# codec/agent.py

import os
import inspect
import pkgutil
import importlib
from typing import Dict, List, Tuple
import pprint
import sys

# --- MODIFIED: Import all connectors and the base class ---
from llm.base import LLMConnector
from llm.gemini import GeminiConnector
from llm.openai import OpenAIConnector
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

        # --- Dynamically select the LLM connector ---
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.connector: LLMConnector

        if provider == "gemini":
            print("🤖 Using Gemini provider.")
            api_key = os.environ.get("GEMINI_API_KEY")
            model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-pro")
            self.connector = GeminiConnector(api_key=api_key, model_name=model_name)
        
        elif provider == "openai":
            print("🤖 Using OpenAI provider.")
            api_key = os.environ.get("OPENAI_API_KEY")
            model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
            self.connector = OpenAIConnector(api_key=api_key, model_name=model_name)
        
        else:
            print(f"❌ Error: Unsupported LLM_PROVIDER '{provider}'. Please use 'gemini' or 'openai'.")
            sys.exit(1)

        print("Loading tools...")
        self.tools = self._load_tools()
        print(f"✅ Loaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")

    def _load_tools(self) -> Dict[str, BaseTool]:
        """
        Dynamically discovers and loads all tool classes from the `tools` directory,
        filtering them based on the active LLM provider.
        """
        loaded_tools = {}
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()

        for _, module_name, _ in pkgutil.iter_modules(tools.__path__, tools.__name__ + "."):
            if module_name.endswith(".base"):
                continue
            
            module = importlib.import_module(module_name)
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseTool) and cls is not BaseTool:
                    tool_instance = cls()
                    
                    # --- Provider Compatibility Check ---
                    # If a tool specifies a list of supported providers, check if the
                    # current provider is in that list. If not, skip loading the tool.
                    # If the list is None (the default), the tool is considered universal.
                    if tool_instance.supported_providers is not None:
                        if provider not in tool_instance.supported_providers:
                            print(f"  - Skipping tool '{tool_instance.name}' (not supported by '{provider}' provider).")
                            continue
                    
                    loaded_tools[tool_instance.name] = tool_instance
        return loaded_tools

    def run(self, prompt: str):
        """
        Starts the agent's execution loop for a single turn of conversation.
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
            multimodal_user_parts: List[ContentPart] = []

            for call in tool_calls:
                print(f"🤖 Agent wants to call tool: {call.name}({call.args})")
                tool_to_execute = self.tools.get(call.name)

                tool_output = None
                if not tool_to_execute:
                    tool_output = f"Error: Tool '{call.name}' not found."
                else:
                    try:
                        validated_args = tool_to_execute.args_schema(**call.args)
                        tool_output = tool_to_execute.execute(self.state, validated_args, self.connector)
                    except Exception as e:
                        tool_output = f"Error executing tool '{call.name}': {e}"

                if isinstance(tool_output, str):
                    print(f"🛠️ Tool Result:\n{tool_output}\n")
                    standard_tool_results.append(ContentPart(
                        type='tool_result',
                        tool_call_id=call.id,
                        tool_name=call.name,
                        text=tool_output
                    ))
                elif isinstance(tool_output, tuple):
                    confirmation_string, new_multimodal_parts = tool_output
                    print(f"🛠️ Tool Result:\n{confirmation_string}\n")
                    
                    standard_tool_results.append(ContentPart(
                        type='tool_result',
                        tool_call_id=call.id,
                        tool_name=call.name,
                        text=confirmation_string
                    ))
                    multimodal_user_parts.extend(new_multimodal_parts)

            if standard_tool_results:
                tool_response_message = Message(role="tool", parts=standard_tool_results)
                self.state.history.append(tool_response_message)

            if multimodal_user_parts:
                print("🖼️  Presenting new multimodal information to the agent.")
                multimodal_user_message = Message(role="user", parts=multimodal_user_message)
                self.state.history.append(multimodal_user_message)