# codec/tools/view_video.py

import os
import ffmpeg
from typing import Optional, Union, List, TYPE_CHECKING, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from pathlib import Path
import logging

from pydantic import BaseModel, Field

# --- MODIFIED: Import our new generic types ---
import openai

from .base import BaseTool
from ..utils import hms_to_seconds, probe_media_file, seconds_to_hms

# --- MODIFIED: Update TYPE_CHECKING imports for the new interface ---
if TYPE_CHECKING:
    from ..state import State


def _extract_and_upload_frame(
    file_path: Union[str, Path],
    timestamp_sec: float,
    display_name: str,
    client: openai.OpenAI,
    tmpdir: str
) -> Tuple[str, str]:
    """
    Core reusable logic to extract a single frame from a video file, save it
    temporarily, and upload it using the provided LLM connector.
    """
    output_path = Path(tmpdir) / f"frame_{os.path.basename(file_path)}_{timestamp_sec:.4f}.jpg"
    try:
        # 1. Extract frame using ffmpeg (this logic is unchanged)
        (
            ffmpeg.input(str(file_path), ss=timestamp_sec)
            .output(str(output_path), vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )

        # 2. Upload the extracted frame using the OpenAI client
        logging.info(f"Uploading frame from '{display_name}' at {timestamp_sec:.3f}s...")
        with open(output_path, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="vision")
        logging.info(f"Upload complete for '{display_name}'. ID: {uploaded_file.id}")
        return uploaded_file.id, str(output_path)

    except Exception as e:
        error_msg = f"Failed to extract or upload frame for '{display_name}' at {timestamp_sec:.3f}s. Details: {e}"
        logging.error(error_msg)
        raise IOError(error_msg) from e


class ViewVideoArgs(BaseModel):
    """Arguments for the view_video tool."""
    source_filename: str = Field(
        ...,
        description="The exact name of the video file from the user's media library to be viewed (e.g., 'interview.mp4')."
    )
    num_frames: int = Field(
        8,
        description="The total number of frames to extract for viewing. This controls the granularity of the preview.",
        gt=0
    )
    start_time: Optional[str] = Field(
        None,
        description="The timestamp in the source video to start extracting frames from. Format: HH:MM:SS.mmm. If omitted, starts from the beginning.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    end_time: Optional[str] = Field(
        None,
        description="The timestamp in a source video to stop extracting frames at. Format: HH:MM:SS.mmm. If omitted, uses the full video duration.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )


class ViewVideoTool(BaseTool):
    """
    A tool to extract and "see" a specified number of evenly-spaced frames
    from a video file within a given time range.
    """

    @property
    def name(self) -> str:
        return "view_video"

    @property
    def description(self) -> str:
        return (
            "Extracts and displays a specified number of frames from a single source video file to 'see' its contents. "
            "You can specify a time range and the number of frames for granularity. If no time range is given, "
            "it will scan the entire video. Use this to get a visual overview, find specific scenes, or "
            "confirm the content of a clip before performing an edit. To view the composed timeline, use 'view_timeline'."
        )

    @property
    def args_schema(self):
        return ViewVideoArgs

    # --- MODIFIED: The execute method signature and return type are updated ---
    def execute(self, state: 'State', args: ViewVideoArgs, client: openai.OpenAI, tmpdir: str) -> str:
        # --- 1. Validation & Setup (No changes needed here) ---
        full_path = os.path.join(state.assets_directory, args.source_filename)
        if not os.path.exists(full_path):
            return f"Error: The source file '{args.source_filename}' does not exist in the assets directory."

        media_info = probe_media_file(full_path)
        if media_info.error:
            return f"Error probing '{args.source_filename}': {media_info.error}"
        
        if not media_info.has_video:
            return f"Error: Source file '{args.source_filename}' does not contain a video stream."
        
        if media_info.duration_sec <= 0:
            return f"Error: Could not determine a valid duration for '{args.source_filename}'."

        source_duration = media_info.duration_sec
        frame_rate = media_info.frame_rate if media_info.frame_rate > 0 else 24.0

        # --- 2. Time & Frame Calculation (No changes needed here) ---
        start_sec = hms_to_seconds(args.start_time) if args.start_time else 0.0
        end_sec = hms_to_seconds(args.end_time) if args.end_time else source_duration

        if start_sec >= end_sec:
            return "Error: The start_time must be before the end_time."
        
        if end_sec > source_duration:
            end_sec = source_duration

        duration_to_sample = end_sec - start_sec

        if duration_to_sample <= 0:
            timestamps = [start_sec]
        else:
            safe_duration = duration_to_sample - (1 / frame_rate) if frame_rate > 0 else duration_to_sample
            segment_duration = safe_duration / max(1, args.num_frames)
            timestamps = [
                start_sec + (i * segment_duration) + (segment_duration / 2)
                for i in range(args.num_frames)
            ]
        
        # --- 3. Parallel Extraction & Upload ---
        logging.info(f"Starting parallel extraction and upload of {len(timestamps)} frames from '{args.source_filename}'...")
        
        upload_results = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_ts = {
                executor.submit(
                    _extract_and_upload_frame,
                    full_path,
                    ts,
                    f"frame-{args.source_filename}-{ts:.2f}s",
                    client,
                    tmpdir
                ): ts
                for ts in timestamps
            }

            for future in as_completed(future_to_ts):
                ts = future_to_ts[future]
                try:
                    result = future.result()
                    upload_results.append((ts, result))
                except Exception as e:
                    upload_results.append((ts, f"An unexpected system error during upload: {e}"))

        # --- 4. Process results and update state ---
        upload_results.sort(key=lambda x: x[0])
        
        successful_uploads = 0
        for ts, result_or_error in upload_results:
            if isinstance(result_or_error, tuple):
                file_id, local_path = result_or_error
                # Add to state for cleanup
                state.uploaded_files.append(file_id)
                # Add to state for the agent to use in the next API call
                state.new_multimodal_files.append((file_id, local_path))
                successful_uploads += 1
            else:
                logging.warning(f"  - Failed to process frame at {ts:.3f}s: {result_or_error}")

        if successful_uploads == 0:
            return f"Error: Failed to extract or upload any frames from '{args.source_filename}'."

        # Return a simple confirmation string. The agent will handle the multimodal part.
        return (
            f"Successfully extracted and uploaded {successful_uploads} frames from '{args.source_filename}' "
            f"between {seconds_to_hms(start_sec)} and {seconds_to_hms(end_sec)}. The agent can now view them."
        )