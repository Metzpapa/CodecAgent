import os
import math
import ffmpeg
from typing import Literal, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

from google import genai
from .base import BaseTool
from state import TimelineClip

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State


class AddToTimelineArgs(BaseModel):
    """Arguments for the add_to_timeline tool."""
    clip_id: str = Field(
        ...,
        description="A unique name for the new clip (e.g., 'intro_scene', 'b-roll_1'). This is used as an identifier for future operations."
    )
    source_filename: str = Field(
        ...,
        description="The exact name of the video file from the user's media library that this clip will be cut from (e.g., 'interview.mp4')."
    )
    source_start_time: str = Field(
        ...,
        description="The timestamp in the source video where the clip begins. Format: HH:MM:SS.mmm",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    source_end_time: str = Field(
        ...,
        description="The timestamp in the source video where the clip ends. Format: HH:MM:SS.mmm",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    timeline_start_time: str = Field(
        "00:00:00.000",
        description="The timestamp on the main timeline where this new clip should be placed. This is ignored when using the 'append' behavior.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    track_index: int = Field(
        0,
        description="The track to place the clip on. 0 is the primary track.",
        ge=0
    )
    clip_description: Optional[str] = Field(
        None,
        description="A human-readable description for organizational purposes."
    )
    insertion_behavior: Literal["append", "insert", "replace"] = Field(
        "append",
        description="Controls how the clip is added. 'append' adds to the end of a specific track. 'insert' shifts subsequent clips. 'replace' overwrites existing content."
    )


class AddToTimelineTool(BaseTool):
    """A tool to add a new clip to the main timeline with precise control over its placement and behavior."""

    @property
    def name(self) -> str:
        return "add_to_timeline"

    @property
    def description(self) -> str:
        return "Adds a clip from a source file to the timeline. Supports appending to a track, inserting with a ripple effect, or replacing existing content."

    @property
    def args_schema(self):
        return AddToTimelineArgs

    def _hms_to_seconds(self, time_str: str) -> float:
        """
        Converts HH:MM:SS.mmm format to total seconds, correctly handling
        partial milliseconds (e.g., '00:00:01.5' -> 1.5s).
        """
        parts = time_str.split(':')
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split('.')
        s = int(s_parts[0])
        
        if len(s_parts) > 1:
            ms_str = s_parts[1].ljust(3, '0')
            ms = int(ms_str)
        else:
            ms = 0
            
        return h * 3600 + m * 60 + s + ms / 1000.0

    def execute(self, state: 'State', args: AddToTimelineArgs, client: 'genai.Client') -> str:
        # --- 1. Pre-flight Validation ---
        if state.clip_id_exists(args.clip_id):
            return f"Error: A clip with the ID '{args.clip_id}' already exists. Please use a unique ID."

        source_path = os.path.join(state.assets_directory, args.source_filename)
        if not os.path.exists(source_path):
            return f"Error: The source file '{args.source_filename}' does not exist in the assets directory."

        # --- 2. Source File Metadata Validation ---
        try:
            probe = ffmpeg.probe(source_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if not video_stream:
                return f"Error: Source file '{args.source_filename}' does not contain a video stream."

            duration_str = video_stream.get('duration') or probe['format'].get('duration')
            if duration_str is None:
                return f"Error: Could not determine duration for source file '{args.source_filename}'."
            source_duration = float(duration_str)

            source_width = video_stream.get('width')
            source_height = video_stream.get('height')
            fr_str = video_stream.get('r_frame_rate', '0/1')
            num, den = map(int, fr_str.split('/'))
            source_fps = num / den if den > 0 else 0

            if not all([source_width, source_height, source_fps > 0]):
                 return f"Error: Could not read essential video properties (resolution, frame rate) from '{args.source_filename}'."

        except Exception as e:
            return f"Error: Could not read metadata from source file '{args.source_filename}'. It may be corrupt. FFmpeg error: {e}"

        source_start_sec = self._hms_to_seconds(args.source_start_time)
        source_end_sec = self._hms_to_seconds(args.source_end_time)

        # --- THE CORRECTED FIX ---
        # Fail if the end time is greater than the duration, UNLESS they are
        # close enough to be considered equal (using a 10ms absolute tolerance).
        if source_end_sec > source_duration and not math.isclose(source_end_sec, source_duration, abs_tol=0.01):
            return f"Error: source_end_time ({source_end_sec:.3f}s) is beyond the source file's total duration ({source_duration:.3f}s)."

        # If the requested end time is slightly beyond the actual duration due to rounding,
        # cap it at the actual duration to ensure data integrity.
        if source_end_sec > source_duration:
            source_end_sec = source_duration

        if source_start_sec >= source_end_sec:
            return "Error: The source_start_time must be before the source_end_time."

        clip_data = {
            "clip_id": args.clip_id,
            "source_path": source_path,
            "source_in_sec": source_start_sec,
            "source_out_sec": source_end_sec,
            "source_total_duration_sec": source_duration,
            "duration_sec": source_end_sec - source_start_sec,
            "track_index": args.track_index,
            "description": args.clip_description,
            "source_frame_rate": source_fps,
            "source_width": source_width,
            "source_height": source_height,
        }

        # --- 3. Dispatch to Behavior Handler ---
        if args.insertion_behavior == "append":
            return self._handle_append(state, clip_data)
        elif args.insertion_behavior == "insert":
            return self._handle_insert(state, args, clip_data)
        elif args.insertion_behavior == "replace":
            return self._handle_replace(state, args, clip_data)
        
        return "Error: Unknown insertion behavior."

    def _handle_append(self, state: 'State', clip_data: dict) -> str:
        timeline_start_sec = state.get_track_duration(clip_data['track_index'])
        clip_data['timeline_start_sec'] = timeline_start_sec
        
        new_clip = TimelineClip(**clip_data)
        state.add_clip(new_clip)
        
        new_track_duration = state.get_track_duration(new_clip.track_index)
        return (
            f"Successfully appended clip '{new_clip.clip_id}' (duration {new_clip.duration_sec:.3f}s) "
            f"to track {new_clip.track_index} at {new_clip.timeline_start_sec:.3f}s. "
            f"The new total duration of track {new_clip.track_index} is now {new_track_duration:.3f}s."
        )

    def _handle_insert(self, state: 'State', args: AddToTimelineArgs, clip_data: dict) -> str:
        timeline_start_sec = self._hms_to_seconds(args.timeline_start_time)
        duration_sec = clip_data['duration_sec']
        clip_data['timeline_start_sec'] = timeline_start_sec

        shifted_count = 0
        for clip in state.get_clips_on_track(args.track_index):
            if clip.timeline_start_sec >= timeline_start_sec:
                clip.timeline_start_sec += duration_sec
                shifted_count += 1
        
        new_clip = TimelineClip(**clip_data)
        state.add_clip(new_clip)

        new_track_duration = state.get_track_duration(new_clip.track_index)
        return (
            f"Successfully inserted clip '{new_clip.clip_id}' (duration {new_clip.duration_sec:.3f}s) "
            f"on track {new_clip.track_index} at {new_clip.timeline_start_sec:.3f}s, shifting {shifted_count} other clips. "
            f"The new total duration of track {new_clip.track_index} is now {new_track_duration:.3f}s."
        )

    def _handle_replace(self, state: 'State', args: AddToTimelineArgs, clip_data: dict) -> str:
        timeline_start_sec = self._hms_to_seconds(args.timeline_start_time)
        duration_sec = clip_data['duration_sec']
        clip_data['timeline_start_sec'] = timeline_start_sec
        replace_end_sec = timeline_start_sec + duration_sec
        
        clips_to_delete = []
        for clip in state.get_clips_on_track(args.track_index):
            clip_end_sec = clip.timeline_start_sec + clip.duration_sec

            if clip.timeline_start_sec < timeline_start_sec and clip_end_sec > replace_end_sec:
                return f"Error: This action would split the existing clip '{clip.clip_id}'. This is not supported. Please adjust the timeline or use an 'insert' operation."

            is_overlapping = max(clip.timeline_start_sec, timeline_start_sec) < min(clip_end_sec, replace_end_sec)
            if is_overlapping:
                clips_to_delete.append(clip.clip_id)

        for clip_id in clips_to_delete:
            state.delete_clip(clip_id)

        new_clip = TimelineClip(**clip_data)
        state.add_clip(new_clip)
        
        new_track_duration = state.get_track_duration(new_clip.track_index)
        return (
            f"Successfully placed clip '{new_clip.clip_id}' (duration {new_clip.duration_sec:.3f}s) "
            f"on track {new_clip.track_index} at {new_clip.timeline_start_sec:.3f}s, deleting {len(clips_to_delete)} overlapping clips. "
            f"The new total duration of track {new_clip.track_index} is now {new_track_duration:.3f}s."
        )