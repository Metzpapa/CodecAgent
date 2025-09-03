# codecagent/codec/tools/view_timeline.py
import os
import logging
from typing import Optional, TYPE_CHECKING, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from pydantic import BaseModel, Field

from .base import BaseTool
from ..utils import hms_to_seconds, seconds_to_hms
from .. import rendering  # <-- IMPORT THE NEW UNIFIED RENDERING MODULE

if TYPE_CHECKING:
    from ..state import State


class ViewTimelineArgs(BaseModel):
    """Arguments for the view_timeline tool."""
    num_frames: int = Field(
        8,
        description="The total number of frames to extract for viewing from the timeline. This controls the granularity of the preview.",
        gt=0
    )
    start_time: Optional[str] = Field(
        None,
        description="The timestamp on the main timeline to start extracting frames from. Format: HH:MM:SS.mmm. If omitted, starts from the beginning of the timeline.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    end_time: Optional[str] = Field(
        None,
        description="The timestamp on the main timeline to stop extracting frames at. Format: HH:MM:SS.mmm. If omitted, uses the full timeline duration.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )


class ViewTimelineTool(BaseTool):
    """
    A tool to extract and 'see' a specified number of evenly-spaced, fully composited
    frames from the timeline within a given time range.
    """

    @property
    def name(self) -> str:
        return "view_timeline"

    @property
    def description(self) -> str:
        return (
            "Extracts and displays a specified number of fully rendered frames from the timeline to 'see' the current edit. "
            "This correctly handles all video layers, transformations, and opacity (e.g., V2 is shown on top of V1). "
            "Use this to get an accurate visual overview of the current state of the video."
        )

    @property
    def args_schema(self):
        return ViewTimelineArgs

    def execute(self, state: 'State', args: ViewTimelineArgs, client: openai.OpenAI, tmpdir: str) -> str:
        if not any(c.track_type == 'video' for c in state.timeline):
            return "Error: The timeline contains no video clips. Cannot view the timeline."

        # --- 1. Determine Time Range & Calculate Sample Timestamps (Largely Unchanged) ---
        start_sec = hms_to_seconds(args.start_time) if args.start_time else 0.0
        end_sec = hms_to_seconds(args.end_time) if args.end_time else state.get_timeline_duration()

        if start_sec >= end_sec:
            return "Error: The start_time must be before the end_time."

        duration_to_sample = end_sec - start_sec
        if duration_to_sample <= 0:
            timeline_timestamps = [start_sec]
        else:
            # Sample frames from the middle of each segment for better representation
            segment_duration = duration_to_sample / args.num_frames
            timeline_timestamps = [start_sec + (i * segment_duration) + (segment_duration / 2) for i in range(args.num_frames)]

        # --- 2. Render and Upload Frames in Parallel (Completely Refactored) ---
        # Instead of complex batching, we now submit a simple render-and-upload
        # task for each timestamp to a thread pool.
        
        logging.info(f"Starting parallel render of {len(timeline_timestamps)} preview frames using MLT...")
        
        successful_frames = 0
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            # Create a future for each frame rendering task
            future_to_ts = {
                executor.submit(self._render_and_upload_frame, state, ts, tmpdir, client): ts
                for ts in timeline_timestamps
            }

            for future in as_completed(future_to_ts):
                ts = future_to_ts[future]
                try:
                    # The future returns the (file_id, local_path) tuple on success
                    file_id, local_path = future.result()
                    state.uploaded_files.append(file_id)
                    state.new_multimodal_files.append((file_id, local_path))
                    successful_frames += 1
                    logging.info(f"Successfully processed frame for timeline at {ts:.3f}s")
                except Exception as e:
                    logging.warning(f"Failed to process frame for timeline at {ts:.3f}s: {e}")
        
        if successful_frames == 0:
            return f"Error: Failed to extract any frames from the timeline between {start_sec:.2f}s and {end_sec:.2f}s."
        
        return (
            f"Successfully rendered and uploaded {successful_frames} frames sampled between {seconds_to_hms(start_sec)} and {seconds_to_hms(end_sec)} "
            f"of the timeline. The agent can now view them."
        )

    def _render_and_upload_frame(
        self, state: 'State', timeline_sec: float, tmpdir: str, client: openai.OpenAI
    ) -> Tuple[str, str]:
        """
        A helper function to render a single frame using the unified rendering
        engine and then upload it. Designed to be run in a ThreadPoolExecutor.
        """
        try:
            # 1. Define a unique path for the output frame.
            output_path = Path(tmpdir) / f"timeline_view_{timeline_sec:.3f}.png"

            # 2. Call the centralized rendering function. This is where the magic happens.
            # It renders the fully composited frame, including all layers and transforms.
            rendering.render_preview_frame(
                state=state,
                timeline_sec=timeline_sec,
                output_path=str(output_path),
                tmpdir=tmpdir
            )

            # 3. Upload the resulting file to OpenAI.
            with open(output_path, "rb") as f:
                uploaded_file = client.files.create(file=f, purpose="vision")
            
            return uploaded_file.id, str(output_path)
        
        except Exception as e:
            # Wrap any exception to be caught by the main loop
            raise RuntimeError(f"Render/upload failed for timestamp {timeline_sec:.3f}s. Details: {e}") from e
