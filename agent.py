# codec/agent.py

import os
import inspect
import pkgutil
import importlib
from typing import Dict, List
import pprint

# --- MODIFIED: Import our new abstractions ---
from llm.base import LLMConnector
from llm.gemini import GeminiConnector # For Milestone 1, we explicitly use the Gemini connector
from llm.types import Message, ContentPart, ToolCall

# Local imports
import tools
from state import State
from tools.base import BaseTool


# REMOVED: SYSTEM_PROMPT_TEMPLATE is still needed.
SYSTEM_PROMPT_TEMPLATE = """
You are codec, a autonomous agent that edits videos.
Users Request:
{user_request}
Please keep going until the user's request is completely resolved. If the request is generic, make a generic video.
First, you should explore the media and get a lay of the land. This means viewing most of the media using the view_video and extract_audio tools. Once you understand what content you are working with, then you can start actually editing. The edit does not need to be perfect. 
Once you have enough media to make an edit finalize the edit and export it for the user. **You cannot ask any questions to the user. Before at least giving the user a rough draft of the video**
"""

# REMOVED: MULTIMODAL_TOOLS, FINISH, and BLOCKED_FINISH_REASONS are no longer needed here.
# This logic has been moved into the GeminiConnector, where it belongs.


class Agent:
    """
    The core AI agent responsible for orchestrating LLM calls and tool execution.
    This class is now provider-agnostic and works with the LLMConnector interface.
    """

    def __init__(self, state: State):
        """
        Initializes the agent, loading the appropriate LLM connector and discovering tools.
        """
        self.state = state

        # --- MODIFIED: Instantiate a Connector, not a specific client ---
        # For Milestone 1, we hardcode the GeminiConnector. In Milestone 2, this
        # will be decided by an environment variable.
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        gemini_model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")
        self.connector: LLMConnector = GeminiConnector(
            api_key=gemini_api_key,
            model_name=gemini_model_name
        )
        # --- END OF MODIFICATION ---

        print("Loading tools...")
        self.tools = self._load_tools()
        # REMOVED: The agent no longer creates the google_tool_set.
        # This is now the responsibility of the connector.
        print(f"Loaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")

    def _load_tools(self) -> Dict[str, BaseTool]:
        """
        Dynamically discovers and loads all tool classes from the `tools` directory.
        (This method's logic remains unchanged.)
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

        # --- MODIFIED: Use our generic Message and ContentPart types ---
        user_message = Message(role="user", parts=[ContentPart(type='text', text=prompt)])
        self.state.history.append(user_message)
        # --- END OF MODIFICATION ---

        if self.state.initial_prompt is None:
            self.state.initial_prompt = prompt

        while True:
            # REMOVED: Token counting was specific to the google client and can be
            # added to the connector interface later if needed.

            final_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                user_request=self.state.initial_prompt
            )

            # --- MODIFIED: Simplified, provider-agnostic API call ---
            response = self.connector.generate_content(
                history=self.state.history,
                tools=list(self.tools.values()),
                system_prompt=final_system_prompt,
            )
            # --- END OF MODIFICATION ---

            # --- MODIFIED: Simplified, generic response validation ---
            if response.is_blocked or not response.message:
                print(f"❌ Response was blocked or empty. Reason: {response.finish_reason}")
                print("   --- Raw API Response for Debugging ---")
                pprint.pprint(response.raw_response)
                print("   --------------------------------------")
                # If the prompt itself was blocked, remove it to allow the user to rephrase.
                if response.finish_reason in ["SAFETY", "BLOCKLIST"]:
                    self.state.history.pop()
                break
            # --- END OF MODIFICATION ---

            model_message = response.message
            self.state.history.append(model_message)

            # --- MODIFIED: Process generic Message parts for text and tool calls ---
            text_parts = []
            tool_calls: List[ToolCall] = []
            for part in model_message.parts:
                if part.type == 'text' and part.text:
                    text_parts.append(part.text)
                elif part.type == 'tool_call' and part.tool_call:
                    tool_calls.append(part.tool_call)

            if text_parts:
                # The Gemini connector combines thoughts and text. We print it all.
                full_text = "\n".join(text_parts)
                print(f"\n🤖 Agent says: {full_text}")
                print("------------------------")

            if not tool_calls:
                print("\n✅ Agent has finished its turn.")
                break
            # --- END OF MODIFICATION ---

            # --- MODIFIED: Unified tool execution logic ---
            standard_tool_results: List[ContentPart] = []
            multimodal_tool_executed = False

            for call in tool_calls:
                print(f"🤖 Agent wants to call tool: {call.name}({call.args})")
                tool_to_execute = self.tools.get(call.name)

                if not tool_to_execute:
                    result_text = f"Error: Tool '{call.name}' not found."
                else:
                    try:
                        validated_args = tool_to_execute.args_schema(**call.args)
                        # Pass the connector to the tool's execute method
                        tool_output = tool_to_execute.execute(self.state, validated_args, self.connector)
                    except Exception as e:
                        tool_output = f"Error executing tool '{call.name}': {e}"

                # This is the new, elegant way to handle multimodal vs. standard tools.
                if isinstance(tool_output, Message):
                    print("🖼️  Agent received a multimodal response. Appending to history and continuing.")
                    self.state.history.append(tool_output)
                    multimodal_tool_executed = True
                else: # The output is a string
                    result_text = str(tool_output)
                    print(f"🛠️ Tool Result:\n{result_text}\n")
                    standard_tool_results.append(ContentPart(
                        type='tool_result',
                        tool_call_id=call.id,
                        tool_name=call.name,
                        text=result_text
                    ))

            # If a multimodal tool was called, we immediately loop back to the model
            # to let it "perceive" the new content in history.
            if multimodal_tool_executed:
                continue

            # If there were only standard (text-based) tools, we bundle their
            # results into a single "tool" message and append it to history.
            if standard_tool_results:
                tool_response_message = Message(role="tool", parts=standard_tool_results)
                self.state.history.append(tool_response_message)
            # --- END OF MODIFICATION ---