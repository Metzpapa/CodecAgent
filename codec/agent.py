# codec/backend/agent.py

import os
import inspect
import pkgutil
import importlib
import json
import pprint
import logging
import time
import re
import random
from typing import Dict, List, Any

# Direct OpenAI import
import openai
# --- MODIFICATION: Import specific error types for granular handling ---
from openai import RateLimitError, InternalServerError, APITimeoutError, APIError
from openai.types.responses import ResponseFunctionToolCall

# Local imports
from . import tools
from .state import State
from .tools.base import BaseTool
from .tools.finish_job import JobFinishedException
from .agent_logging import AgentContextLogger


SYSTEM_PROMPT_TEMPLATE = """
Users Request:
{user_request}
You are codec, a skilled and autonomous video editing agent. Your single purpose is to fulfill the user's request and produce a video.

**Core Directives:**
1.  **You MUST end every job by calling the `finish_job` tool.** This is your only method of communication with the user and it is non-negotiable.
2.  **NEVER ask for clarification.** The user's request is your complete set of instructions. Interpret it to the best of your ability and act.
3.  **A "best effort" video is REQUIRED.** It is always better to deliver an imperfect or "rough draft" video than to ask a question or report a minor failure. The user will provide feedback by submitting a new job.
4.  **If your first attempt fails, TRY AGAIN.** If an action results in an error (like a black frame from a bad crop), analyze the error, adjust your parameters, and execute the action again. Do not give up and ask the user for help. 
"""

def _parse_wait_time_from_error_message(message: str) -> float:
    """
    Parses the wait time from OpenAI's rate limit error message.
    Example: "Please try again in 31.402s." -> 31.402
    Example: "Please try again in 110ms." -> 0.110
    Returns the time in seconds as a float, or 0.0 if not found.
    """
    match = re.search(r"Please try again in ([\d.]+)(ms|s)", message)
    if not match:
        return 0.0

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "ms":
        return value / 1000.0
    return value


class Agent:
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
        self.client = openai.OpenAI(api_key=api_key, max_retries=0)
        
        self.model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-5")

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

    def _execute_turn(self, current_api_input: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
        """
        Executes a single turn of the agent's logic with a custom retry loop.
        """
        api_params = {
            "model": self.model_name,
            "input": current_api_input,
            "tools": self.openai_tools_payload,
        }
        if self.state.last_response_id:
            api_params["previous_response_id"] = self.state.last_response_id
        else:
            api_params["instructions"] = system_prompt
        
        max_retries = 6
        num_retries = 0
        backoff_delay = 1.0
        response = None

        while num_retries < max_retries:
            try:
                response = self.client.responses.create(**api_params)
                break
            
            except RateLimitError as e:
                num_retries += 1
                wait_time = 0.0
                error_message = ""

                try:
                    body = e.response.json()
                    error_message = body.get("error", {}).get("message", "")
                    wait_time = _parse_wait_time_from_error_message(error_message)
                except (json.JSONDecodeError, AttributeError):
                    pass

                if wait_time > 0:
                    wait_time += 0.5
                    self.context_logger.log_rate_limit_body_hit(
                        error_message=error_message,
                        wait_duration_s=wait_time
                    )
                    time.sleep(wait_time)
                else:
                    current_wait = backoff_delay + random.uniform(0, 1)
                    self.context_logger.log_rate_limit_fallback(
                        attempt=num_retries,
                        max_attempts=max_retries,
                        wait_duration_s=current_wait
                    )
                    time.sleep(current_wait)
                    backoff_delay *= 2
            
            # --- NEW: Handle transient server-side errors ---
            except (InternalServerError, APITimeoutError) as e:
                num_retries += 1
                current_wait = backoff_delay + random.uniform(0, 1)
                self.context_logger.log_server_error_retry(
                    error=e,
                    attempt=num_retries,
                    max_attempts=max_retries,
                    wait_duration_s=current_wait
                )
                time.sleep(current_wait)
                backoff_delay *= 2
            # --- END NEW BLOCK ---

            except APIError as e:
                # This will now catch fatal client-side errors (4xx) that are not RateLimitError
                logging.error(f"Fatal OpenAI API Error: {e}", exc_info=True)
                pprint.pprint(e.response.json())
                self.context_logger.log_tool_result("OpenAI_API", f"FATAL ERROR: {e}")
                return []

        if not response:
            logging.error(f"Failed to get a response from OpenAI after {max_retries} retries.")
            self.context_logger.log_tool_result("OpenAI_API", f"FATAL ERROR: Max retries ({max_retries}) exceeded.")
            return []

        self.state.last_response_id = response.id
        self.context_logger.log_model_response(response)
        self.state.history.extend([item.model_dump() for item in response.output])

        tool_calls: List[ResponseFunctionToolCall] = [
            item for item in response.output if item.type == 'function_call'
        ]

        if not tool_calls:
            return []

        tool_outputs_for_api = []
        self.state.new_multimodal_files = []

        for call in tool_calls:
            tool_to_execute = self.tools.get(call.name)
            tool_output_string = f"Error: Tool '{call.name}' not found."

            if tool_to_execute:
                try:
                    parsed_args = json.loads(call.arguments)
                    validated_args = tool_to_execute.args_schema(**parsed_args)
                    tool_output_string = tool_to_execute.execute(self.state, validated_args, self.client)
                except JobFinishedException:
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
        if self.state.new_multimodal_files:
            # Log the images before sending them to the model
            local_paths = [path for _, path in self.state.new_multimodal_files]
            self.context_logger.log_multimodal_request(local_paths)

            # Prepare the content for the API
            multimodal_content = [
                {"type": "input_image", "file_id": file_id}
                for file_id, _ in self.state.new_multimodal_files
            ]
            next_api_input.append({"role": "user", "content": multimodal_content})
        
        self.state.history.extend(next_api_input)
        return next_api_input

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

        current_api_input: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        self.state.history.extend(current_api_input)

        while current_api_input:
            current_api_input = self._execute_turn(current_api_input, final_system_prompt)
        
        logging.warning("\nAgent has finished its turn without calling finish_job. This may be an error.")