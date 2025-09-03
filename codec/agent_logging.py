# codec/backend/agent_logging.py
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional

def _pretty_json(data: Any) -> str:
    """
    Takes a Python object or a JSON string and returns a
    nicely formatted JSON string.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return str(data)
    
    return json.dumps(data, indent=2, ensure_ascii=False)


class AgentContextLogger:
    """
    Manages a dual-logging system for an agent's execution session.
    ...
    """
    def __init__(self,
                 job_id: str,
                 stream_logger: Optional[logging.Logger] = None,
                 logs_dir: Path = Path("logs")):
        self.job_id = job_id
        self.stream_logger = stream_logger
        
        # --- MODIFICATION: Create a dedicated directory for the job ---
        self.job_logs_dir = logs_dir / job_id
        self.job_logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.multimodal_logs_dir = self.job_logs_dir / "multimodal_logs"
        self.multimodal_logs_dir.mkdir(exist_ok=True)
        self.multimodal_request_counter = 0

        raw_log_path = self.job_logs_dir / "agent.raw.log"
        readable_log_path = self.job_logs_dir / "agent.readable.log"
        # --- END MODIFICATION ---

        self.raw_log_file = raw_log_path.open("a", encoding="utf-8")
        self.readable_log_file = readable_log_path.open("a", encoding="utf-8")

    def _write_readable(self, message: str):
        """Writes a message to the readable log and streams it if a logger is configured."""
        self.readable_log_file.write(message)
        self.readable_log_file.flush()
        if self.stream_logger:
            self.stream_logger.info(message.strip())

    def _write_raw(self, event_type: str, data: Dict[str, Any]):
        """Writes a structured event as a JSON line to the raw log."""
        log_entry = {
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **data
        }
        self.raw_log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        self.raw_log_file.flush()

    def log_initial_setup(self, model_name: str, system_prompt: str, tools: List[Dict[str, Any]]):
        """Logs the one-time setup information for the session."""
        self._write_raw("initial_setup", {
            "job_id": self.job_id,
            "model": model_name,
            "system_prompt": system_prompt,
            "tools": tools
        })

        header = [
            "======================================================================",
            "                    CODEC AGENT SESSION LOG",
            "======================================================================",
            f"Job ID:         {self.job_id}",
            f"Model:          {model_name}",
            f"Start Time:     {datetime.utcnow().isoformat()}Z",
            "----------------------------------------------------------------------\n",
            "[SYSTEM INSTRUCTIONS]",
            "---------------------",
            system_prompt.strip(),
            "\n----------------------------------------------------------------------\n",
            "[TOOLS AVAILABLE]",
            "-----------------"
        ]

        for tool in tools:
            header.append(f"- Tool: {tool.get('name', 'N/A')}")
            header.append(f"  Description: {tool.get('description', 'N/A')}")
            params = tool.get('parameters', {})
            if 'properties' in params and params.get('properties'):
                 header.append(f"  Parameters:\n{_pretty_json(params)}")
            else:
                header.append("  Parameters: {}")
            header.append("")

        header.extend([
            "======================================================================",
            "                          CONVERSATION",
            "======================================================================"
        ])
        
        self._write_readable("\n".join(header))

        if self.stream_logger:
            self.stream_logger.info("======================================================================")
            self.stream_logger.info(f"AGENT SESSION STARTING FOR JOB: {self.job_id}")
            self.stream_logger.info(f"Full logs are being written to: {self.job_logs_dir.resolve() / 'agent.readable.log'}")
            self.stream_logger.info("======================================================================")


    def log_user_prompt(self, prompt: str):
        """Logs the initial user request."""
        self._write_raw("user_prompt", {"prompt": prompt})
        self._write_readable(f"\n\nUser: {prompt}")

    def log_multimodal_request(self, local_file_paths: List[str]) -> None:
        """
        Logs the images being sent to the model for a single request.
        It copies the images to a dedicated folder and links to it in the logs.
        """
        if not local_file_paths:
            return

        self.multimodal_request_counter += 1
        request_dir_name = f"request_{self.multimodal_request_counter:02d}"
        request_dir_path = self.multimodal_logs_dir / request_dir_name
        request_dir_path.mkdir()

        copied_files = []
        for i, local_path_str in enumerate(local_file_paths):
            try:
                local_path = Path(local_path_str)
                # Use a generic name in case of collisions from temp files
                dest_path = request_dir_path / f"image_{i+1:03d}{local_path.suffix}"
                shutil.copy2(local_path, dest_path)
                copied_files.append(str(dest_path.resolve()))
            except Exception as e:
                logging.error(f"Could not copy multimodal log file from '{local_path_str}': {e}")
                copied_files.append(f"ERROR_COPYING_{Path(local_path_str).name}")

        log_data = {
            "request_index": self.multimodal_request_counter,
            "image_count": len(local_file_paths),
            "log_directory": str(request_dir_path.resolve()),
            "source_files": local_file_paths,
            "copied_files": copied_files
        }
        self._write_raw("multimodal_request", log_data)

        readable_message = (
            f"\n\n[Multimodal Input]\n"
            f"  - The model was shown {len(local_file_paths)} image(s).\n"
            f"  - View them here: {str(request_dir_path.resolve())}"
        )
        self._write_readable(readable_message)

    def log_model_response(self, response: Any):
        """Logs the model's response, which can be a mix of text and tool calls."""
        self._write_raw("model_response_object", {"response": response.model_dump()})

        for item in response.output:
            self._write_raw("model_output_item", {"item": item.model_dump()})

            if item.type == 'message':
                text_content = "".join([c.text for c in item.content if hasattr(c, 'text')])
                # Instead of calling _write_readable, which streams to the console,
                # we will write this message ONLY to the file log. The CLI is responsible
                # for printing the final agent response to the console.
                message_for_file = f"\n\nModel: {text_content.strip()}"
                self.readable_log_file.write(message_for_file)
                self.readable_log_file.flush()
            
            elif item.type == 'function_call':
                tool_call_str = (
                    f"\n\n[Tool Call]\n"
                    f"  Name: {item.name}\n"
                    f"  Arguments:\n{_pretty_json(item.arguments)}"
                )
                self._write_readable(tool_call_str)

    def log_tool_result(self, tool_name: str, result_string: str):
        """Logs the output from a tool execution."""
        self._write_raw("tool_result", {"tool_name": tool_name, "output": result_string})
        
        indented_result = "\n".join([f"  {line}" for line in result_string.strip().split('\n')])
        self._write_readable(f"\n\nTool Result:\n{indented_result}")

    def log_rate_limit_body_hit(self, error_message: str, wait_duration_s: float):
        """Logs when a rate limit has been hit and the agent is pausing based on the error body."""
        
        log_data = {
            "strategy": "body_based",
            "error_message": error_message,
            "wait_duration_s": wait_duration_s
        }
        self._write_raw("rate_limit_hit", log_data)

        readable_message = (
            f"\n\n[System Notice: Rate Limit Reached (Body-based wait)]\n"
            f"  - API Message: \"{error_message}\"\n"
            f"  - Pausing execution for {wait_duration_s:.2f} seconds before retrying."
        )
        self._write_readable(readable_message)

    def log_rate_limit_fallback(self, attempt: int, max_attempts: int, wait_duration_s: float):
        """Logs when a rate limit has been hit and the agent is using exponential backoff."""
        
        log_data = {
            "strategy": "exponential_backoff",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "wait_duration_s": wait_duration_s
        }
        self._write_raw("rate_limit_hit", log_data)

        readable_message = (
            f"\n\n[System Notice: Rate Limit Reached (Fallback wait)]\n"
            f"  - Body parsing failed. Using exponential backoff.\n"
            f"  - Retry attempt {attempt}/{max_attempts}. Pausing for {wait_duration_s:.2f} seconds."
        )
        self._write_readable(readable_message)

    # --- NEW LOGGING METHOD ---
    def log_server_error_retry(self, error: Exception, attempt: int, max_attempts: int, wait_duration_s: float):
        """Logs when a transient server error occurs and the agent is retrying."""
        
        log_data = {
            "strategy": "server_error_backoff",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "attempt": attempt,
            "max_attempts": max_attempts,
            "wait_duration_s": wait_duration_s
        }
        self._write_raw("server_error_retry", log_data)

        readable_message = (
            f"\n\n[System Notice: OpenAI Server Error (Retrying)]\n"
            f"  - Error: {type(error).__name__}\n"
            f"  - Retry attempt {attempt}/{max_attempts}. Pausing for {wait_duration_s:.2f} seconds."
        )
        self._write_readable(readable_message)
    # --- END NEW METHOD ---

    def log_session_end(self):
        """Logs a footer to signify the end of the session."""
        self._write_raw("session_end", {})
        footer = [
            "\n\n======================================================================",
            "                          SESSION END",
            "======================================================================"
        ]
        self._write_readable("\n".join(footer))

    def close(self):
        """Closes the log files cleanly."""
        self.log_session_end()
        self.raw_log_file.close()
        self.readable_log_file.close()