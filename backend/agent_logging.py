# codecagent/backend/agent_logging.py
import json
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional

class AgentContextLogger:
    """
    Writes a clean, job-scoped log showing exactly what the model sees:
    - System instructions (once)
    - Tools payload (once)
    - For each API call: request payload (input), response, tool outputs sent back, and local history snapshot
    """
    def __init__(self, job_id: str, logs_dir: Path = Path("logs")):
        self.path = logs_dir / f"{job_id}.agent.log"
        self.turn = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Begin file cleanly
        with self.path.open("w", encoding="utf-8") as f:
            f.write(f"=== AGENT SESSION (job_id={job_id}) ===\n")

    def write_header(self, model: str, system_instructions: str, tools: List[Dict[str, Any]]):
        with self.path.open("a", encoding="utf-8") as f:
            f.write("\n----- SYSTEM INSTRUCTIONS (exact) -----\n")
            f.write(system_instructions.strip() + "\n")
            f.write("\n----- TOOLS (exact payload) -----\n")
            f.write(json.dumps(tools, indent=2, ensure_ascii=False) + "\n")

    def log_request(self,
                    previous_response_id: Optional[str],
                    input_payload: List[Dict[str, Any]],
                    local_history_snapshot: List[Dict[str, Any]]):
        self.turn += 1
        entry = {
            "turn": self.turn,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "previous_response_id": previous_response_id,
            "input": input_payload,
            "local_history_snapshot": local_history_snapshot,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"\n=== CALL {self.turn}: REQUEST ===\n")
            f.write(json.dumps(entry, indent=2, ensure_ascii=False) + "\n")

    def log_response(self, response: Any):
        try:
            # For Pydantic v2 models
            data = response.model_dump()
        except Exception:
            try:
                # For older Pydantic or other objects
                data = response.to_dict()
            except Exception:
                try:
                    data = json.loads(response.model_dump_json())
                except Exception:
                    data = str(response)
        entry = {
            "turn": self.turn,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "response": data,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"\n=== CALL {self.turn}: RESPONSE ===\n")
            f.write(json.dumps(entry, indent=2, ensure_ascii=False) + "\n")

    def log_tool_outputs_and_next_input(self,
                                        function_call_outputs: List[Dict[str, Any]],
                                        new_file_ids_for_model: List[str],
                                        next_input_payload: List[Dict[str, Any]]):
        entry = {
            "turn": self.turn,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "function_call_outputs": function_call_outputs,
            "new_files_for_model": new_file_ids_for_model,
            "next_input_payload": next_input_payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"\n=== CALL {self.turn}: TOOL OUTPUTS + NEXT INPUT ===\n")
            f.write(json.dumps(entry, indent=2, ensure_ascii=False) + "\n")