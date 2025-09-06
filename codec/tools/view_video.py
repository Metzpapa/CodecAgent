# codecagent/codec/tools/view_video.py

import os
import ffmpeg
import logging
from typing import Optional, List, TYPE_CHECKING, Tuple, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from pydantic import BaseModel, Field
from PIL import Image

from .base import BaseTool
from ..utils import hms_to_seconds, probe_media_file, seconds_to_hms
from .. import visuals  # <-- IMPORT THE NEW VISUALS MODULE

if TYPE_CHECKING:
    from ..state import State


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
    overlays: List[Literal["coordinate_grid", "anchor_point"]] = Field(
        default_factory=list,
        description="A list of visual aids to render on top of the frames. 'coordinate_grid' shows a faithful, normalized (0.0 to 1.0) grid. 'anchor_point' shows the clip's default anchor point (0.5, 0.5)."
    )
    side_by_side: bool = Field(
        False,
        description="If true, generates a side-by-side image showing the original frame next to the frame with overlays applied. This is useful for comparison."
    )


class ViewVideoTool(BaseTool):
    """
    A tool to extract and "see" a specified number of frames from a single source video file.
    """

    @property
    def name(self) -> str:
        return "view_video"

    @property
    def description(self) -> str:
        return (
            "Extracts and displays frames from a single source video file. This tool can now add visual overlays like a "
            "coordinate grid or anchor point to aid in positioning. It can also display a side-by-side comparison of "
            "the video with and without overlays. Use this to get a visual overview, find specific scenes, or plan "
            "transformations. To view the composed timeline, use 'view_timeline'."
        )

    @property
    def args_schema(self):
        return ViewVideoArgs

    def execute(self, state: 'State', args: ViewVideoArgs, client: openai.OpenAI, tmpdir: str) -> str:
        # --- 1. Validation & Setup ---
        full_path = Path(state.assets_directory) / args.source_filename
        if not full_path.exists():
            return f"Error: The source file '{args.source_filename}' does not exist in the assets directory."

        media_info = probe_media_file(str(full_path))
        if media_info.error:
            return f"Error probing '{args.source_filename}': {media_info.error}"
        
        if not media_info.has_video:
            return f"Error: Source file '{args.source_filename}' does not contain a video stream."
        
        if media_info.duration_sec <= 0:
            return f"Error: Could not determine a valid duration for '{args.source_filename}'."

        # --- 2. Time & Frame Calculation ---
        source_duration = media_info.duration_sec
        start_sec = hms_to_seconds(args.start_time) if args.start_time else 0.0
        end_sec = hms_to_seconds(args.end_time) if args.end_time else source_duration

        if start_sec >= end_sec:
            return "Error: The start_time must be before the end_time."
        
        end_sec = min(end_sec, source_duration)
        duration_to_sample = end_sec - start_sec

        if duration_to_sample <= 0:
            timestamps = [start_sec]
        else:
            segment_duration = duration_to_sample / args.num_frames
            timestamps = [start_sec + (i * segment_duration) + (segment_duration / 2) for i in range(args.num_frames)]
        
        # --- 3. Parallel Extraction, Processing & Upload ---
        logging.info(f"Starting parallel processing of {len(timestamps)} frames from '{args.source_filename}'...")
        
        successful_uploads = 0
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            future_to_ts = {
                executor.submit(self._process_and_upload_frame, state, args, full_path, ts, client, tmpdir): ts
                for ts in timestamps
            }

            for future in as_completed(future_to_ts):
                ts = future_to_ts[future]
                try:
                    file_id, local_path = future.result()
                    state.uploaded_files.append(file_id)
                    state.new_multimodal_files.append((file_id, local_path))
                    successful_uploads += 1
                    logging.info(f"Successfully processed frame for '{args.source_filename}' at {ts:.3f}s")
                except Exception as e:
                    logging.warning(f"Failed to process frame for '{args.source_filename}' at {ts:.3f}s: {e}")

        if successful_uploads == 0:
            return f"Error: Failed to extract or upload any frames from '{args.source_filename}'."

        # --- 4. Formulate Final Response ---
        return (
            f"Successfully extracted and processed {successful_uploads} frames from '{args.source_filename}' "
            f"between {seconds_to_hms(start_sec)} and {seconds_to_hms(end_sec)}. The agent can now view them."
        )

    def _process_and_upload_frame(
        self, state: 'State', args: ViewVideoArgs, file_path: Path, timestamp_sec: float, client: openai.OpenAI, tmpdir: str
    ) -> Tuple[str, str]:
        """
        A helper to extract a frame, apply visual aids, compose if needed, and upload.
        """
        tmp_path = Path(tmpdir)
        raw_frame_path = tmp_path / f"raw_{file_path.stem}_{timestamp_sec:.3f}.png"

        try:
            # 1. Extract the raw frame using ffmpeg
            (
                ffmpeg.input(str(file_path), ss=timestamp_sec)
                .output(str(raw_frame_path), vframes=1, format='image2', vcodec='png')
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            
            with Image.open(raw_frame_path) as raw_image:
                # Resize to sequence dimensions for consistency
                _, seq_width, seq_height = state.get_sequence_properties()
                raw_image = raw_image.resize((seq_width, seq_height), Image.Resampling.LANCZOS)
                
                final_image = None
                
                # 2. Apply overlays and compose the final image
                if args.side_by_side:
                    # Create a version with overlays
                    overlay_image = visuals.apply_overlays(
                        image=raw_image.copy(),
                        overlays=args.overlays,
                        state=state,
                        clip=None, # No specific clip context here
                        timeline_sec=None
                    )
                    # Compose the two side-by-side
                    final_image = visuals.compose_side_by_side(
                        image_left=raw_image,
                        label_left="Source View",
                        image_right=overlay_image,
                        label_right="Source View (with Overlays)"
                    )
                else:
                    # Apply overlays directly to the single image
                    final_image = visuals.apply_overlays(
                        image=raw_image,
                        overlays=args.overlays,
                        state=state,
                        clip=None,
                        timeline_sec=None
                    )

                # 3. Save the final processed image
                final_output_path = tmp_path / f"processed_{file_path.stem}_{timestamp_sec:.3f}.png"
                final_image.save(final_output_path)

            # 4. Upload the final image to OpenAI
            with open(final_output_path, "rb") as f:
                uploaded_file = client.files.create(file=f, purpose="vision")
            
            return uploaded_file.id, str(final_output_path)

        except Exception as e:
            raise RuntimeError(f"Frame processing failed for timestamp {timestamp_sec:.3f}s. Details: {e}") from e