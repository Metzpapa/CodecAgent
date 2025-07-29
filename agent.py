# codec/agent.py

import os
import inspect
import pkgutil
import importlib
import json
import pprint
import sys
from typing import Dict, List, Any

# --- MODIFIED: Direct OpenAI import, no more abstractions ---
import openai
from openai.types.responses import FunctionCall

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
    The core AI agent, now simplified to work directly and statefully with the
    OpenAI Responses API. All provider-agnostic abstractions have been removed
    to increase prototyping velocity and reduce complexity.
    """

    def __init__(self, state: State):
        """
        Initializes the agent, setting up the OpenAI client and loading all
        available tools from the `tools` directory.
        """
        self.state = state
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ Error: OPENAI_API_KEY is not set. Please add it to your .env file.")
            sys.exit(1)

        # --- MODIFIED: Directly initialize the OpenAI client ---
        # No more provider switching logic. We are all-in on OpenAI.
        print("🤖 Using OpenAI Responses API (Stateful).")
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1-mini")

        print("Loading tools...")
        self.tools = self._load_tools()
        print(f"✅ Loaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")

    def _load_tools(self) -> Dict[str, BaseTool]:
        """
        Dynamically discovers and loads all tool classes from the `tools` directory.
        The provider-specific filter has been removed as we only support OpenAI now.
        """
        loaded_tools = {}
        for _, module_name, _ in pkgutil.iter_modules(tools.__path__, tools.__name__ + "."):
            if module_name.endswith(".base"):
                continue
            
            module = importlib.import_module(module_name)
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseTool) and cls is not BaseTool:
                    tool_instance = cls()
                    loaded_tools[tool_instance.name] = tool_instance
        return loaded_tools

    def _tool_to_openai_tool(self, tool: BaseTool) -> Dict[str, Any]:
        """
        Converts one of our BaseTool instances into the dictionary format
        required by the OpenAI Responses API.
        """
        schema = tool.args_schema.model_json_schema()
        # The 'title' field is not expected by the OpenAI API, so we remove it.
        schema.pop('title', None)
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        }

    def run(self, prompt: str):
        """
        Starts and manages the agent's execution loop for a user's request.
        This loop handles the stateful, multi-step conversation with the
        OpenAI API, including all necessary tool calls.
        """
        print("\n--- User Prompt ---")
        print(prompt)
        print("-------------------\n")

        if self.state.initial_prompt is None:
            self.state.initial_prompt = prompt

        # For the first turn, the input is just the user's prompt.
        current_api_input: List[Dict[str, Any]] = [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
        self.state.history.extend(current_api_input)

        # This loop handles a sequence of tool calls within a single user request.
        while True:
            final_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                user_request=self.state.initial_prompt
            )

            print(f"Sending request to OpenAI... (Previous ID: {self.state.last_response_id})")
            
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=current_api_input,
                    tools=[self._tool_to_openai_tool(t) for t in self.tools.values()],
                    instructions=final_system_prompt,
                    previous_response_id=self.state.last_response_id,
                )
            except openai.APIError as e:
                print(f"❌ OpenAI API Error: {e}")
                pprint.pprint(e.response.json())
                break

            # Save the ID of this response to use in the next call.
            self.state.last_response_id = response.id
            
            # Add the raw response output to our local history for context and debugging.
            self.state.history.extend([item.to_dict() for item in response.output])

            # Extract text and tool calls from the response output
            text_outputs = [
                "".join([c.text for c in item.content if hasattr(c, 'text')])
                for item in response.output if item.type == 'message'
            ]
            tool_calls: List[FunctionCall] = [item for item in response.output if item.type == 'function_call']

            if text_outputs:
                full_text = "\n".join(text_outputs)
                print(f"\n🤖 Agent says: {full_text}")
                print("------------------------")

            if not tool_calls:
                print("\n✅ Agent has finished its turn.")
                break # Exit the tool-calling loop for this user request.

            # --- Execute Tools and Prepare for Next API Call ---
            tool_outputs_for_api = []
            # Clear any multimodal data from the previous tool call cycle.
            self.state.new_file_ids_for_model = []

            for call in tool_calls:
                print(f"🤖 Agent wants to call tool: {call.name}({call.arguments})")
                tool_to_execute = self.tools.get(call.name)
                tool_output_string = f"Error: Tool '{call.name}' not found."

                if tool_to_execute:
                    try:
                        # The arguments are a JSON string, so we parse them.
                        parsed_args = json.loads(call.arguments)
                        validated_args = tool_to_execute.args_schema(**parsed_args)
                        # Pass the OpenAI client directly to tools that need it (e.g., for file uploads)
                        # The tool now returns a simple string and modifies state for multimodal output.
                        tool_output_string = tool_to_execute.execute(self.state, validated_args, self.client)
                    except Exception as e:
                        tool_output_string = f"Error executing tool '{call.name}': {e}"

                print(f"🛠️ Tool Result:\n{tool_output_string}\n")
                tool_outputs_for_api.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": tool_output_string
                })

            # The input for the next iteration of the loop starts with the tool results.
            current_api_input = tool_outputs_for_api
            
            # If any tool generated new files for the model to see, we create a new 'user' message.
            if self.state.new_file_ids_for_model:
                print("🖼️  Presenting new multimodal information to the agent.")
                multimodal_content = [
                    {"type": "input_image", "file_id": file_id}
                    for file_id in self.state.new_file_ids_for_model
                ]
                # This new user message is added to the list of inputs for the next API call.
                current_api_input.append({"role": "user", "content": multimodal_content})

            # Add the inputs for the next turn to our history log.
            self.state.history.extend(current_api_input)