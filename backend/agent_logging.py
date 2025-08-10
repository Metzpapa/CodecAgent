# codecagent/backend/agent_logging.py
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional

# A helper to pretty-print JSON for the readable log
def _pretty_json(data: Any) -> str:
    try:
        # Assuming data is a string containing JSON
        parsed = json.loads(data)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        # If it's not valid JSON or not a string, return as is
        return str(data)

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

    def log_initial_setup(self, model_name: str, system_prompt: str, tools: List[Dict[str, Any]]):
        """Logs the one-time setup information for the session."""
        # --- Raw Log ---
        self._write_raw("initial_setup", {
            "job_id": self.job_id,
            "model": model_name,
            "system_prompt": system_prompt,
            "tools": tools
        })

        # --- Readable Log ---
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
            if 'properties' in params and params['properties']:
                 header.append(f"  Parameters:\n{_pretty_json(params)}")
            else:
                header.append("  Parameters: {}")
            header.append("") # Spacer line

        header.extend([
            "======================================================================",
            "                          CONVERSATION",
            "======================================================================"
        ])
        
        self._write_readable("\n".join(header))

    def log_user_prompt(self, prompt: str):
        """Logs the initial user request."""
        self._write_raw("user_prompt", {"prompt": prompt})
        self._write_readable(f"\n\nUser: {prompt}")

    def log_model_response(self, response: Any):
        """Logs the model's response, which can be a mix of text and tool calls."""
        # The raw log gets the full, unprocessed response object
        self._write_raw("model_response_object", {"response": response.model_dump()})

        for item in response.output:
            # Each item in the output list is also logged raw
            self._write_raw("model_output_item", {"item": item.model_dump()})

            # And then formatted for the readable log
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
        
        # Indent the result string for readability
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