# codecagent/codec/tools/view_timeline.py
import os
import logging
import ffmpeg
from typing import Optional, TYPE_CHECKING, Tuple, List, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from pydantic import BaseModel, Field
from PIL import Image

from .base import BaseTool
from ..utils import hms_to_seconds, seconds_to_hms
from .. import rendering
from .. import visuals # <-- IMPORT THE NEW VISUALS MODULE

if TYPE_CHECKING:
    from ..state import State, TimelineClip


class SideBySideConfig(BaseModel):
    """Configuration for enabling a side-by-side view."""
    enabled: bool = Field(
        False,
        description="Set to true to enable the side-by-side view."
    )
    source_clip_id: Optional[str] = Field(
        None,
        description="Optional. The clip_id of the source asset to display in the 'Source View'. If omitted, the system will automatically use the source of the topmost visible clip at each requested frame's timestamp."
    )


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
    overlays: List[Literal["coordinate_grid", "anchor_point"]] = Field(
        default_factory=list,
        description="A list of visual aids to render on top of the frames in both the 'Timeline View' and the 'Source View' (if side_by_side is enabled). 'coordinate_grid' shows a faithful, normalized (0.0 to 1.0) grid. 'anchor_point' shows the clip's current anchor point."
    )
    side_by_side: SideBySideConfig = Field(
        default_factory=SideBySideConfig,
        description="Configuration for the side-by-side view. When enabled, it shows the 'Timeline View' on the right and the corresponding 'Source View' on the left. Overlays are applied to both for easy comparison."
    )


class ViewTimelineTool(BaseTool):
    """
    A tool to extract and 'see' fully rendered frames from the timeline, with optional overlays and side-by-side source comparison.
    """

    @property
    def name(self) -> str:
        return "view_timeline"

    @property
    def description(self) -> str:
        return (
            "Extracts and displays fully rendered frames from the timeline to 'see' the current edit. This tool now supports "
            "visual overlays (like a coordinate grid) and a powerful side-by-side view to compare the composed timeline "
            "against the original source media. Use this to verify transformations, check layering, and plan your next edit."
        )

    @property
    def args_schema(self):
        return ViewTimelineArgs

    def execute(self, state: 'State', args: ViewTimelineArgs, client: openai.OpenAI, tmpdir: str) -> str:
        if not any(c.track_type == 'video' for c in state.timeline):
            return "Error: The timeline contains no video clips. Cannot view the timeline."

        # --- 1. Determine Time Range & Calculate Sample Timestamps ---
        start_sec = hms_to_seconds(args.start_time) if args.start_time else 0.0
        end_sec = hms_to_seconds(args.end_time) if args.end_time else state.get_timeline_duration()

        if start_sec >= end_sec:
            return "Error: The start_time must be before the end_time."

        duration_to_sample = end_sec - start_sec
        if duration_to_sample <= 0:
            timeline_timestamps = [start_sec]
        else:
            segment_duration = duration_to_sample / args.num_frames
            timeline_timestamps = [start_sec + (i * segment_duration) + (segment_duration / 2) for i in range(args.num_frames)]

        # --- 2. Render, Process, and Upload Frames in Parallel ---
        logging.info(f"Starting parallel processing of {len(timeline_timestamps)} timeline frames...")
        
        successful_frames = 0
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            future_to_ts = {
                executor.submit(self._process_and_upload_frame, state, args, ts, tmpdir, client): ts
                for ts in timeline_timestamps
            }

            for future in as_completed(future_to_ts):
                ts = future_to_ts[future]
                try:
                    file_id, local_path = future.result()
                    state.uploaded_files.append(file_id)
                    state.new_multimodal_files.append((file_id, local_path))
                    successful_frames += 1
                    logging.info(f"Successfully processed frame for timeline at {ts:.3f}s")
                except Exception as e:
                    logging.warning(f"Failed to process frame for timeline at {ts:.3f}s: {e}", exc_info=True)
        
        if successful_frames == 0:
            return f"Error: Failed to extract any frames from the timeline between {start_sec:.2f}s and {end_sec:.2f}s."
        
        return (
            f"Successfully rendered and processed {successful_frames} frames sampled between {seconds_to_hms(start_sec)} and {seconds_to_hms(end_sec)} "
            f"of the timeline. The agent can now view them."
        )

    def _process_and_upload_frame(
        self, state: 'State', args: ViewTimelineArgs, timeline_sec: float, tmpdir: str, client: openai.OpenAI
    ) -> Tuple[str, str]:
        """
        A helper to render a timeline frame, optionally get its source, apply overlays, compose, and upload.
        """
        tmp_path = Path(tmpdir)
        
        # 1. Render the fully composited "Timeline View" frame
        timeline_frame_path = tmp_path / f"timeline_{timeline_sec:.3f}.png"
        rendering.render_preview_frame(
            state=state,
            timeline_sec=timeline_sec,
            output_path=str(timeline_frame_path),
            tmpdir=tmpdir
        )
        timeline_image = Image.open(timeline_frame_path)
        
        final_image = None
        source_clip_for_overlays: Optional['TimelineClip'] = None

        # 2. Handle Side-by-Side View
        if args.side_by_side.enabled:
            source_image = None
            
            # Find the relevant source clip
            if args.side_by_side.source_clip_id:
                source_clip = state.find_clip_by_id(args.side_by_side.source_clip_id)
            else:
                source_clip = state.get_topmost_clip_at_time(timeline_sec)
            
            source_clip_for_overlays = source_clip # Use this clip for applying overlays later

            if source_clip:
                # Extract the corresponding source frame
                source_time = source_clip.source_in_sec + (timeline_sec - source_clip.timeline_start_sec)
                source_frame_path = tmp_path / f"source_{source_clip.clip_id}_{timeline_sec:.3f}.png"
                try:
                    (
                        ffmpeg.input(source_clip.source_path, ss=source_time)
                        .output(str(source_frame_path), vframes=1, format='image2', vcodec='png')
                        .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                    )
                    source_image = Image.open(source_frame_path)
                    # Ensure source is resized to match timeline for consistent composition
                    source_image = source_image.resize(timeline_image.size, Image.Resampling.LANCZOS)
                except Exception as e:
                    logging.error(f"Could not extract source frame for clip '{source_clip.clip_id}': {e}")

            # If no source clip or extraction failed, create a black placeholder
            if source_image is None:
                source_image = Image.new("RGB", timeline_image.size, "black")

            # Apply overlays to both images
            timeline_image = visuals.apply_overlays(timeline_image, args.overlays, state, source_clip_for_overlays, timeline_sec)
            source_image = visuals.apply_overlays(source_image, args.overlays, state, source_clip_for_overlays, timeline_sec)
            
            # Compose the final side-by-side image
            final_image = visuals.compose_side_by_side(source_image, "Source View", timeline_image, "Timeline View")

        else: # Not side-by-side, just apply overlays to the timeline view
            source_clip_for_overlays = state.get_topmost_clip_at_time(timeline_sec)
            final_image = visuals.apply_overlays(timeline_image, args.overlays, state, source_clip_for_overlays, timeline_sec)

        # 3. Save and Upload the final image
        final_output_path = tmp_path / f"final_view_{timeline_sec:.3f}.png"
        final_image.save(final_output_path)

        with open(final_output_path, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="vision")
        
        return uploaded_file.id, str(final_output_path)