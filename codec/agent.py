# codec/backend/agent.py

import os
import inspect
import pkgutil
import importlib
import json
import pprint
import logging
import time
from typing import Dict, List, Any

# Direct OpenAI import
import openai
from openai.types.responses import ResponseFunctionToolCall

# Local imports
from . import tools
from .state import State
from .tools.base import BaseTool
from .tools.finish_job import JobFinishedException
from .agent_logging import AgentContextLogger


SYSTEM_PROMPT_TEMPLATE = """
You are codec, a autonomous agent that edits videos.
Users Request:
{user_request}
To complete the job, you MUST call the `finish_job` tool. This is your final step.
You can use it to either export a finished timeline and provide a success message,
or to explain why the request could not be completed.
"""


class Agent:
    """
    The core AI agent, now refactored to support both batch and interactive execution.
    All provider-agnostic abstractions have been removed to increase prototyping
    velocity and reduce complexity.
    """

    def __init__(self, state: State, context_logger: AgentContextLogger):
        """
        Initializes the agent, setting up the OpenAI client and loading all
        available tools from the `tools` directory.

        Args:
            state: The current session state object.
            context_logger: The logger instance for this specific job session.
        """
        self.state = state
        self.context_logger = context_logger

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OPENAI_API_KEY is not set. Please add it to your .env file or set it as an environment variable."
            logging.critical(error_msg)
            raise ValueError(error_msg)

        logging.info("Using OpenAI Responses API (Stateful).")
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1-vision")

        logging.info("Loading tools...")
        self.tools = self._load_tools()
        self.openai_tools_payload = [self._tool_to_openai_tool(t) for t in self.tools.values()]
        logging.info(f"Loaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")


    def _load_tools(self) -> Dict[str, BaseTool]:
        """
        Dynamically discovers and loads all tool classes from the `tools` directory.
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
        schema.pop('title', None)
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        }

    # --- REFACTORED: The core, reusable logic block with encapsulated changes ---
    def _execute_turn(self, current_api_input: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
        """
        Executes a single turn of the agent's logic.
        1. Calls the OpenAI API.
        2. Processes the response (which might involve executing tools).
        3. Returns the input for the *next* turn.
        This is the fundamental building block for both run_to_completion and any future interactive modes.
        """
        # --- START: MODIFICATION ---
        # Prepare API parameters. According to Responses API best practices,
        # the `instructions` should only be sent on the first call of a conversation.
        # The API then carries the instructions forward in the state managed by `previous_response_id`.
        api_params = {
            "model": self.model_name,
            "input": current_api_input,
            "tools": self.openai_tools_payload,
        }
        if self.state.last_response_id:
            api_params["previous_response_id"] = self.state.last_response_id
        else:
            # This is the first turn, so we include the instructions.
            api_params["instructions"] = system_prompt
        # --- END: MODIFICATION ---
        
        # 1. Call the API
        try:
            # Use keyword argument unpacking for a cleaner call
            response = self.client.responses.create(**api_params)
        except openai.RateLimitError:
            logging.warning("Rate limit reached. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            # On retry, we want to use the same input again.
            return current_api_input
        except openai.APIError as e:
            logging.error(f"OpenAI API Error: {e}", exc_info=True)
            pprint.pprint(e.response.json())
            self.context_logger.log_tool_result("OpenAI_API", f"FATAL ERROR: {e}")
            # Return an empty list to signal a fatal error and stop the loop.
            return []

        self.state.last_response_id = response.id
        self.context_logger.log_model_response(response)
        self.state.history.extend([item.model_dump() for item in response.output])

        tool_calls: List[ResponseFunctionToolCall] = [
            item for item in response.output if item.type == 'function_call'
        ]

        # If no tool calls, the agent's turn is over. Return empty to stop the loop.
        if not tool_calls:
            return []

        # 2. Process tool calls
        tool_outputs_for_api = []
        self.state.new_file_ids_for_model = []

        for call in tool_calls:
            tool_to_execute = self.tools.get(call.name)
            tool_output_string = f"Error: Tool '{call.name}' not found."

            if tool_to_execute:
                try:
                    parsed_args = json.loads(call.arguments)
                    validated_args = tool_to_execute.args_schema(**parsed_args)
                    tool_output_string = tool_to_execute.execute(self.state, validated_args, self.client)
                except JobFinishedException:
                    # Propagate the exception to be caught by the high-level orchestrator.
                    raise
                except Exception as e:
                    tool_output_string = f"Error executing tool '{call.name}': {e}"
                    logging.error(f"Error during tool execution for '{call.name}'", exc_info=True)

            self.context_logger.log_tool_result(call.name, tool_output_string)

            tool_outputs_for_api.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": tool_output_string
            })

        # 3. Prepare and return the input for the next turn
        next_api_input = list(tool_outputs_for_api)
        if self.state.new_file_ids_for_model:
            multimodal_content = [
                {"type": "input_image", "file_id": file_id}
                for file_id in self.state.new_file_ids_for_model
            ]
            next_api_input.append({"role": "user", "content": multimodal_content})
        
        self.state.history.extend(next_api_input)
        return next_api_input

    # --- This is now a simple orchestrator (no changes needed here) ---
    def run_to_completion(self, prompt: str):
        """
        Starts and manages the agent's execution loop for a user's request.
        This method now uses `_execute_turn` as its core building block.
        """
        if self.state.initial_prompt is None:
            self.state.initial_prompt = prompt

        final_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_request=self.state.initial_prompt)
        self.context_logger.log_initial_setup(
            model_name=self.model_name,
            system_prompt=final_system_prompt,
            tools=self.openai_tools_payload
        )
        self.context_logger.log_user_prompt(prompt)

        # Prepare the very first input
        current_api_input: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        self.state.history.extend(current_api_input)

        # The main loop simply calls the core logic block until it stops.
        while current_api_input:
            current_api_input = self._execute_turn(current_api_input, final_system_prompt)
        
        logging.warning("\nAgent has finished its turn without calling finish_job. This may be an error.")