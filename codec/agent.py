# codec/agent.py

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
import tempfile
from typing import Dict, List, Any, Optional

# Direct OpenAI import
import openai
# Import specific error types for granular handling
from openai import RateLimitError, InternalServerError, APITimeoutError, APIError
from openai.types.responses import ResponseFunctionToolCall

# Local imports
from . import tools
from .state import State
from .tools.base import BaseTool
from .agent_logging import AgentContextLogger


SYSTEM_PROMPT_TEMPLATE = """
Users Request:
{user_request}
You are codec, a skilled and collaborative video editing agent. Your purpose is to work with the user to fulfill their request and produce a video.

**Core Directives:**
1.  **Complete Task to best of ability:** Try to do as much as you can with the information given by the user. If the user's request is ambiguous, make reasonable assumptions to move forward.
2.  **Use Your Tools:** Execute tools to manipulate the timeline, find media, and render videos based on the user's instructions.
3.  **Report Your Work:** After you have finished a set of actions, provide a clear, concise text summary of what you have done.
4.  **Cite Your Output:** When you create a file the user needs to see (like a rendered video or an exported timeline), you MUST reference it in your message by placing the exact filename in square brackets. For example: `I have rendered the video for you. You can view it here: [final_render.mp4]`.
5.  **Continuous Conversation:** Your work is part of an ongoing conversation. Do not end the job unless the user tells you the project is complete. The `finish_job` tool has been removed.
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

            except APIError as e:
                logging.error(f"Fatal OpenAI API Error: {e}", exc_info=True)
                pprint.pprint(e.response.json())
                self.context_logger.log_tool_result("OpenAI_API", f"FATAL ERROR: {e}")
                return []

        if not response:
            logging.error(f"Failed to get a response from OpenAI after {max_retries} retries.")
            self.context_logger.log_tool_result("OpenAI_API", f"FATAL ERROR: Max retries ({max_retries}) exceeded.")
            return []

        self.state.last_response_id = response.id
        # --- FIX: We now pass the response object itself to the history for better parsing ---
        self.state.history.extend(response.output)
        self.context_logger.log_model_response(response)

        tool_calls: List[ResponseFunctionToolCall] = [
            item for item in response.output if item.type == 'function_call'
        ]

        if not tool_calls:
            return []

        tool_outputs_for_api = []
        self.state.new_multimodal_files = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for call in tool_calls:
                tool_to_execute = self.tools.get(call.name)
                tool_output_string = f"Error: Tool '{call.name}' not found."

                if tool_to_execute:
                    try:
                        parsed_args = json.loads(call.arguments)
                        validated_args = tool_to_execute.args_schema(**parsed_args)
                        tool_output_string = tool_to_execute.execute(self.state, validated_args, self.client, tmpdir)
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
                local_paths = [path for _, path in self.state.new_multimodal_files]
                self.context_logger.log_multimodal_request(local_paths)

                multimodal_content = [
                    {"type": "input_image", "file_id": file_id}
                    for file_id, _ in self.state.new_multimodal_files
                ]
                next_api_input.append({"role": "user", "content": multimodal_content})
        
        self.state.history.extend(next_api_input)
        return next_api_input

    def process_turn(self, user_prompt: str) -> Optional[str]:
        """
        Processes a single turn of the conversation.
        Takes a user prompt, executes any necessary tool calls, and returns the
        agent's final text response for that turn, or None if no text was generated.
        """
        if self.state.initial_prompt is None:
            self.state.initial_prompt = user_prompt
            final_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(user_request=self.state.initial_prompt)
            self.context_logger.log_initial_setup(
                model_name=self.model_name,
                system_prompt=final_system_prompt,
                tools=self.openai_tools_payload
            )
        else:
            final_system_prompt = ""

        self.context_logger.log_user_prompt(user_prompt)

        # --- FIX: Keep track of where this turn's history starts ---
        turn_start_index = len(self.state.history)
        
        current_api_input: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        self.state.history.append(current_api_input[0])

        while current_api_input:
            current_api_input = self._execute_turn(current_api_input, final_system_prompt)
        
        # --- FIX: Search for the last message within THIS turn only ---
        last_model_message = ""
        # Search backwards from the end of the history to the start of this turn
        for item in reversed(self.state.history[turn_start_index:]):
            # The history now contains Pydantic models, not dicts
            if hasattr(item, 'type') and item.type == 'message' and item.role == 'assistant':
                text_parts = [
                    content.text for content in item.content if hasattr(content, 'text')
                ]
                last_model_message = "".join(text_parts).strip()
                if last_model_message:
                    break

        if not last_model_message:
            logging.warning("Agent turn ended without a final text response from the model.")
            return None

        return last_model_message