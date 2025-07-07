# codec/agent.py

import os
import inspect
import pkgutil
import importlib
from typing import Dict

from google import genai
from google.genai import types
from pydantic import BaseModel

# Local imports
import tools
from state import State
from tools.base import BaseTool

# This makes the logic explicit and easily extensible.
MULTIMODAL_TOOLS = {"view_video"}

class Agent:
    """
    The core AI agent responsible for orchestrating LLM calls and tool execution.
    """

    def __init__(self, state: State):
        """
        Initializes the agent.

        Args:
            state: The state object that holds the session's context.
        """
        self.state = state

        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-pro")

        print("Loading tools...")
        self.tools = self._load_tools()
        function_declarations = [tool.to_google_tool() for tool in self.tools.values()]
        self.google_tools = [types.Tool(function_declarations=function_declarations)]
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

        Args:
            prompt: The user's request to the agent for this turn.
        """
        print("\n--- User Prompt ---")
        print(prompt)
        print("-------------------\n")

        self.state.history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

        while True:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self.state.history,
                config=types.GenerateContentConfig(tools=self.google_tools),
            )

            if not response.candidates:
                print("🤖 Agent did not return a candidate. Ending turn.")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    print(f"❌ The prompt was blocked. Reason: {response.prompt_feedback.block_reason.name}")
                    print("   Please try rephrasing your request.")
                else:
                    print("   The model's response was likely filtered for safety or other reasons.")
                break

            model_response_content = response.candidates[0].content
            self.state.history.append(model_response_content)

            text_parts = [part.text for part in model_response_content.parts if part.text]
            function_calls = [part.function_call for part in model_response_content.parts if part.function_call]

            if text_parts:
                print(f"🤖 Agent says: {''.join(text_parts)}")

            if not function_calls:
                print("\n✅ Agent has finished its turn.")
                break

            special_call = next((fc for fc in function_calls if fc.name in MULTIMODAL_TOOLS), None)

            if special_call:
                tool_name = special_call.name
                tool_args = dict(special_call.args)
                print(f"🤖 Agent wants to call special tool: {tool_name}({tool_args})")
                
                tool_to_execute = self.tools.get(tool_name)
                try:
                    validated_args = tool_to_execute.args_schema(**tool_args)
                    tool_output = tool_to_execute.execute(self.state, validated_args)
                    
                    if isinstance(tool_output, types.Content):
                        print("🖼️  Agent received a multimodal response. Appending to history and continuing.")
                        self.state.history.append(tool_output)
                    else:
                        # --- START OF FIX ---
                        # The tool returned an error string. We must report this specific error back.
                        print(f"🛠️ Special tool '{tool_name}' returned an error string: {tool_output}")
                        error_content = types.Content(role="tool", parts=[types.Part.from_function_response(
                            name=tool_name,
                            # Pass the actual error string from the tool back to the model.
                            response={"error": tool_output}
                        )])
                        self.state.history.append(error_content)
                        # --- END OF FIX ---

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
                            result = tool_to_execute.execute(self.state, validated_args)
                        except Exception as e:
                            result = f"Error executing tool '{tool_name}': {e}"
                    
                    print(f"🛠️ Tool Result:\n{result}\n")
                    standard_tool_results.append(types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result},
                    ))
                
                self.state.history.append(types.Content(role="tool", parts=standard_tool_results))