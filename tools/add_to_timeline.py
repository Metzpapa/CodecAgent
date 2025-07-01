import os
import ffmpeg
from typing import Literal, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

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
        """Converts HH:MM:SS.mmm format to total seconds."""
        parts = time_str.split(':')
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split('.')
        s = int(s_parts[0])
        ms = int(s_parts[1]) if len(s_parts) > 1 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0

    def execute(self, state: 'State', args: AddToTimelineArgs) -> str:
        # --- 1. Pre-flight Validation ---
        if state.clip_id_exists(args.clip_id):
            return f"Error: A clip with the ID '{args.clip_id}' already exists. Please use a unique ID."

        source_path = os.path.join(state.assets_directory, args.source_filename)
        if not os.path.exists(source_path):
            return f"Error: The source file '{args.source_filename}' does not exist in the assets directory."

        # --- 2. Source File Metadata Validation ---
        try:
            probe = ffmpeg.probe(source_path)
            # Use the same robust logic as the get_asset_info tool for consistency.
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            duration_str = video_stream.get('duration') if video_stream else probe['format'].get('duration')
            
            if duration_str is None:
                return f"Error: Could not determine duration for source file '{args.source_filename}'."
            source_duration = float(duration_str)
        except Exception as e:
            return f"Error: Could not read metadata from source file '{args.source_filename}'. It may be corrupt. FFmpeg error: {e}"

        source_start_sec = self._hms_to_seconds(args.source_start_time)
        source_end_sec = self._hms_to_seconds(args.source_end_time)

        if source_end_sec > source_duration:
            return f"Error: source_end_time ({source_end_sec:.2f}s) is beyond the source file's total duration ({source_duration:.2f}s)."

        if source_start_sec >= source_end_sec:
            return "Error: The source_start_time must be before the source_end_time."

        # --- 3. Dispatch to Behavior Handler ---
        if args.insertion_behavior == "append":
            return self._handle_append(state, args, source_path, source_start_sec, source_end_sec)
        elif args.insertion_behavior == "insert":
            return self._handle_insert(state, args, source_path, source_start_sec, source_end_sec)
        elif args.insertion_behavior == "replace":
            return self._handle_replace(state, args, source_path, source_start_sec, source_end_sec)
        
        return "Error: Unknown insertion behavior."

    def _handle_append(self, state: 'State', args: AddToTimelineArgs, source_path: str, source_start_sec: float, source_end_sec: float) -> str:
        timeline_start_sec = state.get_track_duration(args.track_index)
        duration_sec = source_end_sec - source_start_sec

        new_clip = TimelineClip(
            clip_id=args.clip_id,
            source_path=source_path,
            source_in_sec=source_start_sec,
            source_out_sec=source_end_sec,
            timeline_start_sec=timeline_start_sec,
            duration_sec=duration_sec,
            track_index=args.track_index,
            description=args.clip_description
        )
        state.add_clip(new_clip)
        return f"Successfully appended clip '{args.clip_id}' to the end of track {args.track_index} at {timeline_start_sec:.3f}s."

    def _handle_insert(self, state: 'State', args: AddToTimelineArgs, source_path: str, source_start_sec: float, source_end_sec: float) -> str:
        timeline_start_sec = self._hms_to_seconds(args.timeline_start_time)
        duration_sec = source_end_sec - source_start_sec

        # Shift subsequent clips on the same track
        shifted_count = 0
        for clip in state.get_clips_on_track(args.track_index):
            if clip.timeline_start_sec >= timeline_start_sec:
                clip.timeline_start_sec += duration_sec
                shifted_count += 1
        
        new_clip = TimelineClip(
            clip_id=args.clip_id,
            source_path=source_path,
            source_in_sec=source_start_sec,
            source_out_sec=source_end_sec,
            timeline_start_sec=timeline_start_sec,
            duration_sec=duration_sec,
            track_index=args.track_index,
            description=args.clip_description
        )
        state.add_clip(new_clip)
        return f"Successfully inserted clip '{args.clip_id}' on track {args.track_index} at {timeline_start_sec:.3f}s, shifting {shifted_count} other clips."

    def _handle_replace(self, state: 'State', args: AddToTimelineArgs, source_path: str, source_start_sec: float, source_end_sec: float) -> str:
        timeline_start_sec = self._hms_to_seconds(args.timeline_start_time)
        duration_sec = source_end_sec - source_start_sec
        replace_end_sec = timeline_start_sec + duration_sec
        
        clips_to_delete = []
        for clip in state.get_clips_on_track(args.track_index):
            clip_end_sec = clip.timeline_start_sec + clip.duration_sec

            # Error case: The new clip would land in the middle of an existing clip.
            if clip.timeline_start_sec < timeline_start_sec and clip_end_sec > replace_end_sec:
                return f"Error: This action would split the existing clip '{clip.clip_id}'. This is not supported. Please adjust the timeline or use an 'insert' operation."

            # Check for any other overlap
            is_overlapping = max(clip.timeline_start_sec, timeline_start_sec) < min(clip_end_sec, replace_end_sec)
            if is_overlapping:
                clips_to_delete.append(clip.clip_id)

        # Perform deletions
        for clip_id in clips_to_delete:
            state.delete_clip(clip_id)

        new_clip = TimelineClip(
            clip_id=args.clip_id,
            source_path=source_path,
            source_in_sec=source_start_sec,
            source_out_sec=source_end_sec,
            timeline_start_sec=timeline_start_sec,
            duration_sec=duration_sec,
            track_index=args.track_index,
            description=args.clip_description
        )
        state.add_clip(new_clip)
        
        deleted_msg = f"and deleted {len(clips_to_delete)} overlapping clips" if clips_to_delete else ""
        return f"Successfully placed clip '{args.clip_id}' on track {args.track_index} at {timeline_start_sec:.3f}s {deleted_msg}."