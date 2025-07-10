# codec/tools/view_timeline.py

import os
import ffmpeg
from typing import Optional, TYPE_CHECKING, Union, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from pathlib import Path
from collections import defaultdict

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from .base import BaseTool
from state import TimelineClip

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State


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
            "Use this to get a visual overview, find specific scenes, or confirm the edit. The output will show which source clip each frame comes from. "
            "To view a single source file, use 'view_video'."
        )

    @property
    def args_schema(self):
        return ViewTimelineArgs

    def _hms_to_seconds(self, time_str: str) -> float:
        """Converts HH:MM:SS.mmm format to total seconds."""
        parts = time_str.split(':')
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split('.')
        s = int(s_parts[0])
        ms = int(s_parts[1].ljust(3, '0')) if len(s_parts) > 1 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0

    def execute(self, state: 'State', args: ViewTimelineArgs, client: 'genai.Client') -> str | types.Content:
        if not state.timeline:
            return "Error: The timeline is empty. Cannot view an empty timeline."

        # --- 1. Determine Time Range & Calculate Sample Timestamps ---
        start_sec = self._hms_to_seconds(args.start_time) if args.start_time else 0.0
        end_sec = self._hms_to_seconds(args.end_time) if args.end_time else state.get_timeline_duration()

        if start_sec >= end_sec:
            return "Error: The start_time must be before the end_time."

        duration_to_sample = end_sec - start_sec
        if duration_to_sample <= 0:
            timeline_timestamps = [start_sec]
        else:
            segment_duration = duration_to_sample / args.num_frames
            timeline_timestamps = [start_sec + (i * segment_duration) for i in range(args.num_frames)]

        # --- 2. Map Timeline Timestamps to Source Frames and Group by Source File ---
        frames_by_source = defaultdict(list)
        gaps = []
        for ts in timeline_timestamps:
            found_clip = False
            for clip in state.timeline:
                clip_end_time = clip.timeline_start_sec + clip.duration_sec
                if clip.timeline_start_sec <= ts < clip_end_time:
                    source_time_offset = ts - clip.timeline_start_sec
                    source_ts = clip.source_in_sec + source_time_offset
                    frame_number = int(round(source_ts * clip.source_frame_rate))
                    frames_by_source[clip.source_path].append({'timeline_ts': ts, 'frame_num': frame_number, 'clip': clip})
                    found_clip = True
                    break
            if not found_clip:
                gaps.append(ts)

        # --- 3. Batch Extraction per Source & Parallel Upload ---
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Starting batch extraction for {len(frames_by_source)} source files...")
            
            upload_results = {} # Maps timeline_ts -> result
            with ThreadPoolExecutor(max_workers=16) as executor:
                future_to_ts = {}
                # Step 3a: Batch extract frames from each source file
                for source_path, tasks in frames_by_source.items():
                    try:
                        frame_numbers = sorted(list(set([t['frame_num'] for t in tasks])))
                        select_filter = "+".join([f"eq(n,{fn})" for fn in frame_numbers])
                        output_pattern = os.path.join(tmpdir, f"{Path(source_path).stem}_frame_%04d.jpg")
                        
                        proc = ffmpeg.input(source_path).filter('select', select_filter).output(output_pattern, vsync='vfr', start_number=0).run_async(pipe_stdout=True, pipe_stderr=True)
                        proc.communicate()

                        extracted_files = sorted(Path(tmpdir).glob(f"{Path(source_path).stem}_frame_*.jpg"))
                        
                        # Step 3b: Create upload tasks for the extracted files
                        tasks_sorted_by_frame = sorted(tasks, key=lambda t: t['frame_num'])
                        for i, task in enumerate(tasks_sorted_by_frame):
                            if i < len(extracted_files):
                                display_name = f"timeline-frame-{task['clip'].clip_id}-{task['timeline_ts']:.2f}s"
                                future = executor.submit(self._upload_file_from_path, extracted_files[i], display_name, client)
                                future_to_ts[future] = task['timeline_ts']

                    except Exception as e:
                        for task in tasks:
                            upload_results[task['timeline_ts']] = f"Failed to extract frames from {os.path.basename(source_path)}: {e}"

                # Step 3c: Collect upload results
                for future in as_completed(future_to_ts):
                    ts = future_to_ts[future]
                    try:
                        upload_results[ts] = future.result()
                    except Exception as e:
                        upload_results[ts] = f"Upload failed: {e}"

            # --- 4. Assemble Response ---
            all_parts = [types.Part.from_text(
                text=f"SYSTEM: This is the output of the `view_timeline` tool. "
                     f"Displaying frames sampled between {start_sec:.2f}s and {end_sec:.2f}s of the timeline."
            )]

            all_events = list(frames_by_source.values())
            all_tasks = [item for sublist in all_events for item in sublist]
            all_tasks.extend([{'timeline_ts': ts, 'clip': None} for ts in gaps])
            all_tasks.sort(key=lambda x: x['timeline_ts'])

            for task in all_tasks:
                ts = task['timeline_ts']
                if not task.get('clip'):
                    # THIS IS THE CORRECTED LINE
                    all_parts.append(types.Part.from_text(text=f"Timeline at {ts:.3f}s: [GAP]"))
                    continue

                result = upload_results.get(ts)
                clip_id = task['clip'].clip_id
                
                if isinstance(result, types.File):
                    frame_file = result
                    state.uploaded_files.append(frame_file)
                    all_parts.append(types.Part.from_text(text=f"Timeline at {ts:.3f}s (from clip: '{clip_id}')"))
                    all_parts.append(types.Part.from_uri(file_uri=frame_file.uri, mime_type='image/jpeg'))
                else:
                    error_details = result or "Processing failed."
                    all_parts.append(types.Part.from_text(
                        text=f"SYSTEM: Could not process frame from clip '{clip_id}' at timeline {ts:.3f}s. Error: {error_details}"
                    ))
            
            return types.Content(role="user", parts=all_parts)

    def _upload_file_from_path(self, file_path: Path, display_name: str, client: 'genai.Client') -> Union[types.File, str]:
        """Helper to upload a single file, intended for use in the executor."""
        try:
            with open(file_path, "rb") as f:
                uploaded_file = client.files.upload(
                    file=f,
                    config={"mimeType": "image/jpeg", "displayName": display_name}
                )
            return uploaded_file
        except Exception as e:
            return f"Failed to upload file. Details: {str(e)}"