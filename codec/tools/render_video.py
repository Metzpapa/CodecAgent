# codecagent/codec/tools/render_video.py

import logging
from typing import TYPE_CHECKING
from pathlib import Path

from pydantic import BaseModel, Field
import openai

from .base import BaseTool
from .. import rendering # <-- IMPORT THE NEW UNIFIED RENDERING MODULE

if TYPE_CHECKING:
    from ..state import State


class RenderVideoArgs(BaseModel):
    """Arguments for the render_video tool."""
    output_filename: str = Field(
        "final_render.mp4",
        description="The desired filename for the final rendered video file. Should end in .mp4."
    )


# The complex _create_property_interpolator function and all MoviePy logic are now removed.

class RenderVideoTool(BaseTool):
    """
    Renders the current timeline into a final video file (e.g., an MP4).
    This tool synthesizes all clips, layers, and transformations into a viewable video
    using the unified, high-performance MLT rendering engine.
    """

    @property
    def name(self) -> str:
        return "render_video"

    @property
    def description(self) -> str:
        return (
            "Renders the current timeline into a final video file (e.g., an MP4), applying all transformations and layering. "
            "This is used to produce the final video deliverable. After calling this, you should call `finish_job` with the returned filename in the `attachments` list."
        )

    @property
    def args_schema(self):
        return RenderVideoArgs

    def execute(self, state: 'State', args: RenderVideoArgs, client: openai.OpenAI, tmpdir: str) -> str:
        if not state.timeline:
            return "Error: Cannot render because the timeline is empty."

        # 1. Set up the output path.
        job_dir = Path(state.assets_directory).parent
        output_path = job_dir / "output" / args.output_filename
        output_path.parent.mkdir(exist_ok=True)

        try:
            # 2. Delegate the entire rendering process to the centralized rendering module.
            # All logic for XML generation, command-line execution, and hardware
            # acceleration is handled there.
            rendering.render_final_video(
                state=state,
                output_path=str(output_path),
                tmpdir=tmpdir
            )

            logging.info("Render completed successfully.")
            return f"Successfully rendered video to '{args.output_filename}' in the output directory."

        except Exception as e:
            # Catch any errors from the rendering module and report them to the agent.
            logging.error(f"An unexpected error occurred during rendering with MLT: {e}", exc_info=True)
            return f"An unexpected error occurred during rendering: {e}"