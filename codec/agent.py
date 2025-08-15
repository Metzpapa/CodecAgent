# codec/backend/agent.py

import os
import inspect
import pkgutil
import importlib
import json
import pprint
import logging
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

    def run_to_completion(self, prompt: str):
        """
        Starts and manages the agent's execution loop for a user's request.
        This is a "fire-and-forget" method intended for batch processing, like
        in a Celery task. It loops internally until the `finish_job` tool is
        called, raising a JobFinishedException.
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

        current_api_input: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        self.state.history.extend(current_api_input)

        while True:
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=current_api_input,
                    tools=self.openai_tools_payload,
                    instructions=final_system_prompt,
                    previous_response_id=self.state.last_response_id,
                )
            except openai.APIError as e:
                logging.error(f"OpenAI API Error: {e}", exc_info=True)
                pprint.pprint(e.response.json())
                self.context_logger.log_tool_result("OpenAI_API", f"FATAL ERROR: {e}")
                break

            self.state.last_response_id = response.id
            self.context_logger.log_model_response(response)
            self.state.history.extend([item.model_dump() for item in response.output])

            tool_calls: List[ResponseFunctionToolCall] = [
                item for item in response.output if item.type == 'function_call'
            ]

            if not tool_calls:
                logging.warning("\nAgent has finished its turn without calling finish_job. This may be an error.")
                break

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
                        logging.info("`finish_job` tool called. Propagating signal to terminate.")
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

            next_api_input = list(tool_outputs_for_api)
            if self.state.new_file_ids_for_model:
                multimodal_content = [
                    {"type": "input_image", "file_id": file_id}
                    for file_id in self.state.new_file_ids_for_model
                ]
                next_api_input.append({"role": "user", "content": multimodal_content})

            current_api_input = next_api_input
            self.state.history.extend(current_api_input)

    def step(self, prompt: str):
        """
        Processes a single turn of the conversation for interactive use (CLI).
        It takes a user prompt, runs the model until it either calls tools and
        gets their output, or responds with a text message, and then waits for
        the next input from the user.
        """
        # If this is the first turn, set the initial prompt and log the setup.
        if not self.state.history:
            self.state.initial_prompt = prompt
            final_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_request=prompt)
            self.context_logger.log_initial_setup(
                model_name=self.model_name,
                system_prompt=final_system_prompt,
                tools=self.openai_tools_payload
            )
        else:
            # On subsequent turns, the system prompt remains the same.
            final_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_request=self.state.initial_prompt)

        self.context_logger.log_user_prompt(prompt)
        current_api_input: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        self.state.history.extend(current_api_input)

        # This loop handles a potential chain of tool calls within a single user turn.
        while True:
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=current_api_input,
                    tools=self.openai_tools_payload,
                    instructions=final_system_prompt,
                    previous_response_id=self.state.last_response_id,
                )
            except openai.APIError as e:
                logging.error(f"OpenAI API Error: {e}", exc_info=True)
                pprint.pprint(e.response.json())
                self.context_logger.log_tool_result("OpenAI_API", f"FATAL ERROR: {e}")
                break

            self.state.last_response_id = response.id
            self.context_logger.log_model_response(response)
            self.state.history.extend([item.model_dump() for item in response.output])

            tool_calls: List[ResponseFunctionToolCall] = [
                item for item in response.output if item.type == 'function_call'
            ]

            # If the model responds with text or no tool calls, its turn is over.
            # We break the loop and wait for the next user input.
            if not tool_calls:
                break

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
                        logging.info("`finish_job` tool called. Propagating signal to terminate.")
                        raise # Re-raise to be caught by the CLI's main loop
                    except Exception as e:
                        tool_output_string = f"Error executing tool '{call.name}': {e}"
                        logging.error(f"Error during tool execution for '{call.name}'", exc_info=True)

                self.context_logger.log_tool_result(call.name, tool_output_string)

                tool_outputs_for_api.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": tool_output_string
                })

            next_api_input = list(tool_outputs_for_api)
            if self.state.new_file_ids_for_model:
                multimodal_content = [
                    {"type": "input_image", "file_id": file_id}
                    for file_id in self.state.new_file_ids_for_model
                ]
                next_api_input.append({"role": "user", "content": multimodal_content})

            # The input for the next iteration of the tool-calling loop is the tool results.
            current_api_input = next_api_input
            self.state.history.extend(current_api_input)