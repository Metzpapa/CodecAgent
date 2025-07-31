# codec/tools/view_timeline.py

import os
import ffmpeg
from typing import Optional, TYPE_CHECKING
import openai
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from pathlib import Path
from collections import defaultdict

from pydantic import BaseModel, Field

from .base import BaseTool
from ..utils import hms_to_seconds

# Use a forward reference for the State class to avoid circular imports.
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
    A tool to extract and 'see' a specified number of evenly-spaced frames
    from the rendered timeline within a given time range.
    """

    @property
    def name(self) -> str:
        return "view_timeline"

    @property
    def description(self) -> str:
        return (
            "Extracts and displays a specified number of frames from the *rendered timeline* to 'see' the current edit. "
            "This correctly handles video layers (e.g., V2 is shown on top of V1). "
            "Use this to get a visual overview, find specific scenes, or confirm the edit. The output will show which source clip each frame comes from. "
            "To view a single source file, use 'view_video'."
        )

    @property
    def args_schema(self):
        return ViewTimelineArgs

    def execute(self, state: 'State', args: ViewTimelineArgs, client: openai.OpenAI) -> str:
        if not state.timeline:
            return "Error: The timeline is empty. Cannot view an empty timeline."

        # --- 1. Determine Time Range & Calculate Sample Timestamps ---
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

        # --- 2. Map Timeline Timestamps to the Topmost Visible Source Frame ---
        timeline_events = []
        for ts in timeline_timestamps:
            # Find all active video clips at this timestamp
            active_video_clips = [
                clip for clip in state.timeline
                if clip.track_type == 'video' and
                   clip.timeline_start_sec <= ts < (clip.timeline_start_sec + clip.duration_sec)
            ]

            if not active_video_clips:
                timeline_events.append({'timeline_ts': ts, 'clip': None})
                continue

            # The topmost clip is the one with the highest track number
            topmost_clip = max(active_video_clips, key=lambda c: c.track_number)
            
            source_time_offset = ts - topmost_clip.timeline_start_sec
            source_ts = topmost_clip.source_in_sec + source_time_offset
            
            timeline_events.append({
                'timeline_ts': ts,
                'clip': topmost_clip,
                'source_ts': source_ts
            })

        # --- 3. Group by Source File for Batch Extraction ---
        frames_by_source = defaultdict(list)
        for event in timeline_events:
            if event['clip']:
                clip = event['clip']
                frame_number = int(round(event['source_ts'] * clip.source_frame_rate))
                event['frame_num'] = frame_number # Store for later
                frames_by_source[clip.source_path].append(event)

        # --- 4. Batch Extraction per Source & Parallel Upload ---
        with tempfile.TemporaryDirectory() as tmpdir:
            logging.info(f"Starting batch extraction for {len(frames_by_source)} source files...")
            
            upload_results = {} # Maps timeline_ts -> result (File or error string)
            with ThreadPoolExecutor(max_workers=16) as executor:
                future_to_ts = {}
                for source_path, tasks in frames_by_source.items():
                    try:
                        frame_numbers = sorted(list(set([t['frame_num'] for t in tasks])))
                        select_filter = "+".join([f"eq(n,{fn})" for fn in frame_numbers])
                        output_pattern = os.path.join(tmpdir, f"{Path(source_path).stem}_frame_%04d.jpg")
                        
                        proc = ffmpeg.input(source_path).filter('select', select_filter).output(output_pattern, vsync='vfr', start_number=0).run_async(pipe_stdout=True, pipe_stderr=True)
                        proc.communicate()

                        extracted_files = sorted(Path(tmpdir).glob(f"{Path(source_path).stem}_frame_*.jpg"))
                        
                        tasks_sorted_by_frame = sorted(tasks, key=lambda t: t['frame_num'])
                        for i, task in enumerate(tasks_sorted_by_frame):
                            if i < len(extracted_files):
                                display_name = f"timeline-frame-{task['clip'].clip_id}-{task['timeline_ts']:.2f}s"
                                future = executor.submit(self._upload_file_from_path, extracted_files[i], display_name, client)
                                future_to_ts[future] = task['timeline_ts']

                    except Exception as e:
                        for task in tasks:
                            upload_results[task['timeline_ts']] = f"Failed to extract frames from {os.path.basename(source_path)}: {e}"

                for future in as_completed(future_to_ts):
                    ts = future_to_ts[future]
                    try:
                        upload_results[ts] = future.result()
                    except Exception as e:
                        upload_results[ts] = f"Upload failed: {e}"

            # --- 5. Process results and update state ---
            successful_frames = 0
            for event in sorted(timeline_events, key=lambda x: x['timeline_ts']):
                ts = event['timeline_ts']
                clip = event.get('clip')

                if not clip:
                    logging.info(f"Timeline at {ts:.3f}s: [GAP ON VIDEO TRACKS]")
                    continue

                result = upload_results.get(ts)
                track_name = f"V{clip.track_number}"
                
                if isinstance(result, str) and "Failed" not in result:
                    file_id = result
                    state.uploaded_files.append(file_id)
                    state.new_file_ids_for_model.append(file_id)
                    successful_frames += 1
                    logging.info(f"Timeline at {ts:.3f}s (from clip: '{clip.clip_id}' on track {track_name})")
                else:
                    error_details = result or "Processing failed."
                    logging.warning(f"  - Failed to process frame from clip '{clip.clip_id}' at timeline {ts:.3f}s: {error_details}")
            
            if successful_frames == 0:
                return f"Error: Failed to extract any frames from the timeline between {start_sec:.2f}s and {end_sec:.2f}s."
            
            return (
                f"Successfully extracted and uploaded {successful_frames} frames sampled between {start_sec:.2f}s and {end_sec:.2f}s "
                f"of the timeline. The agent can now view them."
            )

    def _upload_file_from_path(self, file_path: Path, display_name: str, client: openai.OpenAI) -> str:
        """Helper to upload a single file, intended for use in the executor."""
        try:
            with open(file_path, "rb") as f:
                uploaded_file = client.files.create(file=f, purpose="vision")
            return uploaded_file.id
        except Exception as e:
            return f"Failed to upload file. Details: {str(e)}"