# codec/agent.py

import os
import inspect
import pkgutil
import importlib
from typing import Dict

from google import genai
from google.genai import types

# Local imports
import tools
from state import State
from tools.base import BaseTool


SYSTEM_PROMPT = """
You are codec, a autonomous agent that edits videos.
Please keep going until the user’s query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.
"""

# All tools that return a `types.Content` object for the model to "perceive"
# should be included here. This enables the agent to review its own work.
MULTIMODAL_TOOLS = {"view_video", "extract_audio", "view_timeline", "extract_timeline_audio"}

# Alias for brevity, improving readability in the validation block.
FINISH = types.FinishReason

# Define the set of finish reasons that indicate a blocked or incomplete response.
# This is a robust way to handle various failure modes from the API.
BLOCKED_FINISH_REASONS = {
    FINISH.SAFETY,
    FINISH.RECITATION,
    FINISH.BLOCKLIST,
    FINISH.PROHIBITED_CONTENT,
    FINISH.SPII,
    FINISH.IMAGE_SAFETY,
    FINISH.MALFORMED_FUNCTION_CALL,
    FINISH.OTHER,
    FINISH.LANGUAGE,
}
# MAX_TOKENS is intentionally *not* included, as we often want to process partial output.


class Agent:
    """
    The core AI agent responsible for orchestrating LLM calls and tool execution.
    """

    def __init__(self, state: State):
        """
        Initializes the agent, loading the API client and discovering all available tools.
        """
        self.state = state
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-pro") #gemini-2.5-flash-lite-preview-06-17

        print("Loading tools...")
        self.tools = self._load_tools()
        function_declarations = [tool.to_google_tool() for tool in self.tools.values()]
        self.google_tool_set = types.Tool(function_declarations=function_declarations)
        print(f"Loaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")

    def _load_tools(self) -> Dict[str, BaseTool]:
        """
        Dynamically discovers and loads all tool classes from the `tools` directory.
        """
        loaded_tools = {}
        for _, module_name, _ in pkgutil.iter_modules(tools.__path__, tools.__name__ + "."):
            if not module_name.endswith(".base"):
                module = importlib.import_module(module_name)
                for _, cls in inspect.getmembers(module, inspect.isclass):
                    if issubclass(cls, BaseTool) and cls is not BaseTool:
                        tool_instance = cls()
                        loaded_tools[tool_instance.name] = tool_instance
        return loaded_tools

    def run(self, prompt: str):
        """
        Starts the agent's execution loop for a single turn of conversation.
        """
        print("\n--- User Prompt ---")
        print(prompt)
        print("-------------------\n")

        self.state.history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

        while True:
            try:
                token_count_response = self.client.models.count_tokens(
                    model=self.model_name,
                    contents=self.state.history,
                )
                print(f"\n📈 Context size before this turn: {token_count_response.total_tokens} tokens")
            except Exception as e:
                print(f"⚠️  Could not count tokens: {e}")

            config = types.GenerateContentConfig(
                tools=[self.google_tool_set],
                system_instruction=SYSTEM_PROMPT,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True
                )
                # ----------------------------------------------------
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self.state.history,
                config=config,
            )

            # --- BULLET-PROOF RESPONSE VALIDATION ---

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"❌ The prompt was blocked. Reason: {response.prompt_feedback.block_reason.name}")
                print("   Please try rephrasing your request.")
                self.state.history.pop()
                break

            if not response.candidates:
                print("🤖 Agent did not return a candidate. This might be due to a safety filter or other issue. Ending turn.")
                break

            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason or FINISH.FINISH_REASON_UNSPECIFIED

            if finish_reason in BLOCKED_FINISH_REASONS:
                print(f"❌ Response was blocked by the model. Reason: {finish_reason.name}")
                if candidate.safety_ratings:
                    for r in candidate.safety_ratings:
                        print(f"   - {r.category.name}: {r.probability.name}")
                break

            model_response_content = candidate.content

            if not model_response_content or not model_response_content.parts:
                print(f"🤖 Agent returned an empty response. Finish Reason: {finish_reason.name}. Ending turn.")
                break
            # --- END OF VALIDATION BLOCK ---

            self.state.history.append(model_response_content)

            # --- NEW: PROCESS RESPONSE PARTS FOR THOUGHTS AND TEXT ---
            # We loop through the parts to handle thought summaries separately.
            has_thoughts = False
            has_text_response = False
            for part in candidate.content.parts:
                if part.thought:
                    if not has_thoughts:
                        print("\n🤔 Agent is thinking...")
                        has_thoughts = True
                    # The 'part.text' of a thought is the summary.
                    print(part.text)
                elif part.text:
                    print(f"\n🤖 Agent says: {part.text}")
                    has_text_response = True
            
            if has_thoughts:
                print("------------------------")
            # -----------------------------------------------------------

            # Access function calls from the top-level response object.
            function_calls = response.function_calls

            if not function_calls:
                # If there was text or thoughts, the turn might not be "finished"
                # in the sense of requiring more input, but it's done for now.
                if has_text_response or has_thoughts:
                     print("\n✅ Agent has finished its turn.")
                break

            # --- Tool execution logic ---
            special_call = next((fc for fc in function_calls if fc.name in MULTIMODAL_TOOLS), None)

            if special_call:
                tool_name = special_call.name
                tool_args = dict(special_call.args)
                print(f"🤖 Agent wants to call special tool: {tool_name}({tool_args})")
                
                tool_to_execute = self.tools.get(tool_name)
                try:
                    validated_args = tool_to_execute.args_schema(**tool_args)
                    tool_output = tool_to_execute.execute(self.state, validated_args, self.client)
                    
                    if isinstance(tool_output, types.Content):
                        print("🖼️  Agent received a multimodal response. Appending to history and continuing.")
                        self.state.history.append(tool_output)
                    else:
                        print(f"🛠️ Special tool '{tool_name}' returned an error string: {tool_output}")
                        error_content = types.Content(role="tool", parts=[types.Part.from_function_response(
                            name=tool_name,
                            response={"error": str(tool_output)}
                        )])
                        self.state.history.append(error_content)

                except Exception as e:
                    error_content = types.Content(role="tool", parts=[types.Part.from_function_response(
                        name=tool_name,
                        response={"error": f"Error executing tool '{tool_name}': {e}"}
                    )])
                    self.state.history.append(error_content)
                
                continue

            else:
                standard_tool_results = []
                for func_call in function_calls:
                    tool_name = func_call.name
                    tool_args = dict(func_call.args)
                    print(f"🤖 Agent wants to call tool: {tool_name}({tool_args})")

                    tool_to_execute = self.tools.get(tool_name)
                    if not tool_to_execute:
                        result = f"Error: Tool '{tool_name}' not found."
                    else:
                        try:
                            validated_args = tool_to_execute.args_schema(**tool_args)
                            result = tool_to_execute.execute(self.state, validated_args, self.client)
                        except Exception as e:
                            result = f"Error executing tool '{tool_name}': {e}"
                    
                    print(f"🛠️ Tool Result:\n{result}\n")
                    standard_tool_results.append(types.Part.from_function_response(
                        name=tool_name,
                        response={"result": str(result)},
                    ))
                
                self.state.history.append(types.Content(role="tool", parts=standard_tool_results))