# codec/tools/finish_job.py

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List
import openai

from pydantic import BaseModel, Field

from .base import BaseTool

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from ..state import State


class JobFinishedException(Exception):
    """
    A special exception raised by the finish_job tool to signal that the
    agent's work is complete. This allows for a clean exit from the agent's
    run loop and provides the final result to the calling process (e.g., a Celery task).
    """
    def __init__(self, result: dict):
        self.result = result
        super().__init__("Job finished successfully.")


class FinishJobArgs(BaseModel):
    """Arguments for the finish_job tool."""
    message: str = Field(
        ...,
        description="A final, user-facing message summarizing the work done, explaining the result, or detailing why the request could not be completed. This is always required."
    )
    attachments: Optional[List[str]] = Field(
        None,
        description="A list of filenames from the output directory to present to the user (e.g., ['final_render.mp4', 'project_files.otio']). These files must first be created by other tools like `render_video` or `export_timeline`."
    )


class FinishJobTool(BaseTool):
    """
    The final tool to be called in any job. It stops the agent's work and provides a summary message and any generated files to the user.
    This tool MUST be called to complete a job, whether it was successful or not. It does not generate files itself, but presents files created by other tools.
    """

    @property
    def name(self) -> str:
        return "finish_job"

    @property
    def description(self) -> str:
        return (
            "The single, final tool to end the editing job. Call this when the user's request is fully addressed or when you cannot proceed. "
            "You MUST provide a final `message` for the user. If you have created files with other tools (like `render_video` or `export_timeline`), "
            "provide their filenames in the `attachments` list."
        )

    @property
    def args_schema(self):
        return FinishJobArgs

    def execute(self, state: 'State', args: FinishJobArgs, client: openai.OpenAI, tmpdir: str) -> str:
        final_message = args.message
        resolved_attachment_paths = []
        
        job_dir = Path(state.assets_directory).parent
        output_dir = job_dir / "output"

        if args.attachments:
            logging.info(f"Preparing attachments for final delivery: {args.attachments}")
            
            missing_files = []
            for filename in args.attachments:
                # The agent might provide a nested path, so we handle it safely
                attachment_path = (output_dir / filename).resolve()

                if attachment_path.is_file():
                    resolved_attachment_paths.append(str(attachment_path))
                else:
                    logging.warning(f"Attachment '{filename}' requested by agent not found at expected path '{attachment_path}'")
                    missing_files.append(filename)
            
            if missing_files:
                missing_str = ", ".join(missing_files)
                final_message += f"\n\n[System Note: The agent tried to attach the following files, but they could not be found: {missing_str}]"


        # Prepare the final result payload for the calling process (e.g., Celery task)
        final_result = {
            "status": "COMPLETE",
            "message": final_message,
            "output_paths": resolved_attachment_paths, # Plural to support multiple files
        }

        # Raise the special exception to terminate the agent's run loop
        raise JobFinishedException(final_result)