# codecagent/backend/agent_logging.py
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional

# --- MODIFIED: A more robust pretty-printing helper ---
def _pretty_json(data: Any) -> str:
    """
    Takes a Python object or a JSON string and returns a
    nicely formatted JSON string.
    """
    if isinstance(data, str):
        try:
            # If it's a string, parse it into a Python object first
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            # If it's not a valid JSON string, return it as is.
            return str(data)
    
    # Now, `data` is guaranteed to be a Python object (dict, list, etc.)
    # Dump it to a nicely formatted string.
    return json.dumps(data, indent=2, ensure_ascii=False)


class AgentContextLogger:
    """
    Manages a dual-logging system for an agent's execution session.

    1.  **Raw Log (`.raw.log`):** A machine-readable log where each line is a
        JSON object representing a single event (e.g., setup, model output,
        tool result). This is for perfect, high-fidelity debugging.

    2.  **Readable Log (`.readable.log`):** A human-friendly, narrative log
        that tells the story of the agent's "conversation." This is for
        high-level understanding and sharing.

    3.  **Stream Logging:** It can optionally stream the readable log's content
        to a provided logger instance (like a Celery task logger) for
        real-time console output.
    """
    def __init__(self,
                 job_id: str,
                 stream_logger: Optional[logging.Logger] = None,
                 logs_dir: Path = Path("logs")):
        self.job_id = job_id
        self.stream_logger = stream_logger
        
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        raw_log_path = logs_dir / f"{job_id}.agent.raw.log"
        readable_log_path = logs_dir / f"{job_id}.agent.readable.log"

        # Open files in append mode to be robust against restarts
        self.raw_log_file = raw_log_path.open("a", encoding="utf-8")
        self.readable_log_file = readable_log_path.open("a", encoding="utf-8")

    def _write_readable(self, message: str):
        """Writes a message to the readable log and streams it if a logger is configured."""
        self.readable_log_file.write(message)
        self.readable_log_file.flush()  # <-- FIX: Ensure data is written to disk immediately.
        if self.stream_logger:
            # We strip leading/trailing newlines for cleaner console output
            self.stream_logger.info(message.strip())

    def _write_raw(self, event_type: str, data: Dict[str, Any]):
        """Writes a structured event as a JSON line to the raw log."""
        log_entry = {
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **data
        }
        self.raw_log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        self.raw_log_file.flush() # Also good practice to flush the raw log.

    def log_initial_setup(self, model_name: str, system_prompt: str, tools: List[Dict[str, Any]]):
        """
        Logs the one-time setup information for the session.
        Writes the full header to the file log, but only a concise
        message to the real-time stream.
        """
        # --- Raw Log (Always gets the full data) ---
        self._write_raw("initial_setup", {
            "job_id": self.job_id,
            "model": model_name,
            "system_prompt": system_prompt,
            "tools": tools
        })

        # --- Readable Log File (Gets the full header) ---
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
            # Don't show empty properties dict for cleaner output
            if 'properties' in params and params.get('properties'):
                 # --- THIS LINE NOW WORKS CORRECTLY ---
                 header.append(f"  Parameters:\n{_pretty_json(params)}")
            else:
                header.append("  Parameters: {}")
            header.append("")

        header.extend([
            "======================================================================",
            "                          CONVERSATION",
            "======================================================================"
        ])
        
        # Use _write_readable to ensure it gets flushed
        self._write_readable("\n".join(header))

        # --- Real-time Stream (Gets a concise message) ---
        if self.stream_logger:
            self.stream_logger.info("======================================================================")
            self.stream_logger.info(f"AGENT SESSION STARTING FOR JOB: {self.job_id}")
            self.stream_logger.info(f"Full logs are being written to: logs/{self.job_id}.agent.readable.log")
            self.stream_logger.info("======================================================================")


    def log_user_prompt(self, prompt: str):
        """Logs the initial user request."""
        self._write_raw("user_prompt", {"prompt": prompt})
        self._write_readable(f"\n\nUser: {prompt}")

    def log_model_response(self, response: Any):
        """Logs the model's response, which can be a mix of text and tool calls."""
        self._write_raw("model_response_object", {"response": response.model_dump()})

        for item in response.output:
            self._write_raw("model_output_item", {"item": item.model_dump()})

            if item.type == 'message':
                text_content = "".join([c.text for c in item.content if hasattr(c, 'text')])
                self._write_readable(f"\n\nModel: {text_content.strip()}")
            
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