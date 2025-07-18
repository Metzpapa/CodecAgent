# codec/agent.py

import os
import inspect
import pkgutil
import importlib
from typing import Dict, List
import pprint
import sys

# --- MODIFIED: Import all connectors and the base class ---
from llm.base import LLMConnector
from llm.gemini import GeminiConnector
from llm.openai import OpenAIConnector # <-- ADDED
from llm.types import Message, ContentPart, ToolCall

# Local imports
import tools
from state import State
from tools.base import BaseTool


SYSTEM_PROMPT_TEMPLATE = """
You are codec, a autonomous agent that edits videos.
Users Request:
{user_request}
Please keep going until the user's request is completely resolved. If the request is generic, make a generic video.
First, you should explore the media and get a lay of the land. This means viewing most of the media using the view_video and extract_audio tools. Once you understand what content you are working with, then you can start actually editing. The edit does not need to be perfect. 
Once you have enough media to make an edit finalize the edit and export it for the user. **You cannot ask any questions to the user. Before at least giving the user a rough draft of the video**
"""


class Agent:
    """
    The core AI agent responsible for orchestrating LLM calls and tool execution.
    This class is now provider-agnostic and works with the LLMConnector interface.
    """

    def __init__(self, state: State):
        """
        Initializes the agent, loading the appropriate LLM connector based on
        environment variables, and discovering all available tools.
        """
        self.state = state

        # --- MODIFIED: Dynamically select the LLM connector ---
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.connector: LLMConnector

        if provider == "gemini":
            print("🤖 Using Gemini provider.")
            api_key = os.environ.get("GEMINI_API_KEY")
            model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-pro-latest")
            self.connector = GeminiConnector(api_key=api_key, model_name=model_name)
        
        elif provider == "openai":
            print("🤖 Using OpenAI provider.")
            api_key = os.environ.get("OPENAI_API_KEY")
            model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o") # Default to gpt-4o
            self.connector = OpenAIConnector(api_key=api_key, model_name=model_name)
        
        else:
            # This check is also in main.py, but it's good practice to have it here.
            print(f"❌ Error: Unsupported LLM_PROVIDER '{provider}'. Please use 'gemini' or 'openai'.")
            sys.exit(1)
        # --- END OF MODIFICATION ---

        print("Loading tools...")
        self.tools = self._load_tools()
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
        (This method's logic remains unchanged as it's provider-agnostic.)
        """
        print("\n--- User Prompt ---")
        print(prompt)
        print("-------------------\n")

        user_message = Message(role="user", parts=[ContentPart(type='text', text=prompt)])
        self.state.history.append(user_message)

        if self.state.initial_prompt is None:
            self.state.initial_prompt = prompt

        while True:
            final_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                user_request=self.state.initial_prompt
            )

            # This is the core provider-agnostic call.
            response = self.connector.generate_content(
                history=self.state.history,
                tools=list(self.tools.values()),
                system_prompt=final_system_prompt,
            )

            if response.is_blocked or not response.message:
                print(f"❌ Response was blocked or empty. Reason: {response.finish_reason}")
                print("   --- Raw API Response for Debugging ---")
                pprint.pprint(response.raw_response)
                print("   --------------------------------------")
                # If the prompt itself was blocked, remove it to allow the user to rephrase.
                if response.finish_reason in ["SAFETY", "BLOCKLIST", "API_ERROR"]:
                    self.state.history.pop()
                break

            model_message = response.message
            self.state.history.append(model_message)

            text_parts = []
            tool_calls: List[ToolCall] = []
            for part in model_message.parts:
                if part.type == 'text' and part.text:
                    text_parts.append(part.text)
                elif part.type == 'tool_call' and part.tool_call:
                    tool_calls.append(part.tool_call)

            if text_parts:
                full_text = "\n".join(text_parts)
                print(f"\n🤖 Agent says: {full_text}")
                print("------------------------")

            if not tool_calls:
                print("\n✅ Agent has finished its turn.")
                break

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

            if multimodal_tool_executed:
                continue

            if standard_tool_results:
                tool_response_message = Message(role="tool", parts=standard_tool_results)
                self.state.history.append(tool_response_message)