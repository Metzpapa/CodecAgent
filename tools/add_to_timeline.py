# codec/tools/add_to_timeline.py
import os
import math
import ffmpeg
from typing import Literal, Optional, TYPE_CHECKING, Tuple
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
        description="The exact name of the video or image file from the media library that this clip will be cut from (e.g., 'interview.mp4', 'title_card.png')."
    )
    source_start_time: str = Field(
        ...,
        description="The timestamp where the clip begins in the source asset. For video, this is a specific time like '00:01:30.000'. For static images, this must be '00:00:00.000'.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    source_end_time: str = Field(
        ...,
        description="The timestamp where the clip ends in the source asset. For video, this is a specific time. For static images, this defines the desired display duration (e.g., '00:00:05.000' shows the image for 5 seconds).",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    timeline_start_time: str = Field(
        "00:00:00.000",
        description="The timestamp on the main timeline where this new clip should be placed. When using 'insert' behavior, this must be at an existing cut point. This is ignored when using 'append'.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    track_index: int = Field(
        0,
        description="The track to place the clip on. 0 is the primary track.",
        ge=0
    )
    clip_description: Optional[str] = Field(
        None,
        description="A description for organizational purposes. Use this to describe anything you want to remember about this clip, include specifics if necessary."
    )
    insertion_behavior: Literal["append", "insert", "replace"] = Field(
        "append",
        description="Controls how the clip is added. 'append' adds to the end of a track. 'insert' shifts subsequent clips but requires placing at an existing cut. 'replace' overwrites existing content."
    )


class AddToTimelineTool(BaseTool):
    """A tool to add a new clip to the main timeline with precise control over its placement and behavior."""

    @property
    def name(self) -> str:
        return "add_to_timeline"

    @property
    def description(self) -> str:
        return "Adds a clip from a source file (video or image) to the timeline. Supports appending to a track, inserting at an existing cut point with a ripple effect, or replacing existing content."

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
    
    def _get_or_infer_frame_rate(self, state: 'State') -> float:
        """Gets sequence frame rate from state or infers it from the first video clip."""
        if state.frame_rate:
            return state.frame_rate
        
        first_video_clip = next((c for c in state.timeline if c.source_frame_rate > 0), None)
        
        if first_video_clip:
            return first_video_clip.source_frame_rate
        else:
            # Return a sensible default if no video clips are on the timeline
            return 24.0

    def execute(self, state: 'State', args: AddToTimelineArgs, client: 'genai.Client') -> str:
        # --- 1. Pre-flight Validation ---
        if state.clip_id_exists(args.clip_id):
            return f"Error: A clip with the ID '{args.clip_id}' already exists. Please use a unique ID."

        source_path = os.path.join(state.assets_directory, args.source_filename)
        if not os.path.exists(source_path):
            return f"Error: The source file '{args.source_filename}' does not exist in the assets directory."

        # --- 2. Source File Metadata Extraction (Handles both video and images) ---
        is_image = source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'))

        try:
            probe = ffmpeg.probe(source_path)
            # All valid images and videos will have a 'video' stream for ffmpeg
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if not video_stream:
                return f"Error: Source file '{args.source_filename}' does not contain a video stream or is not a supported image format."

            source_width = video_stream.get('width')
            source_height = video_stream.get('height')
            if not all([source_width, source_height]):
                 return f"Error: Could not read resolution from '{args.source_filename}'."

            if is_image:
                # --- Logic for Static Image Assets ---
                source_start_sec_arg = self._hms_to_seconds(args.source_start_time)
                if not math.isclose(source_start_sec_arg, 0.0, abs_tol=0.001):
                    return f"Error: For image assets, 'source_start_time' must be '00:00:00.000'. You provided '{args.source_start_time}'."
                
                duration_sec = self._hms_to_seconds(args.source_end_time)
                if duration_sec <= 0:
                    return f"Error: For image assets, 'source_end_time' must represent a positive duration (e.g., '00:00:05.000'). You provided '{args.source_end_time}'."

                source_in_sec = 0.0
                source_out_sec = duration_sec
                source_total_duration_sec = duration_sec
                has_audio = False
                # For OTIO compatibility, images need a frame rate. We infer it from the sequence.
                source_fps = self._get_or_infer_frame_rate(state)
            else:
                # --- Logic for Video Assets ---
                audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
                has_audio = audio_stream is not None

                duration_str = video_stream.get('duration') or probe['format'].get('duration')
                if duration_str is None:
                    return f"Error: Could not determine duration for source file '{args.source_filename}'."
                source_total_duration_sec = float(duration_str)

                fr_str = video_stream.get('r_frame_rate', '0/1')
                num, den = map(int, fr_str.split('/'))
                source_fps = num / den if den > 0 else 0
                if not source_fps > 0:
                     return f"Error: Could not read a valid frame rate from '{args.source_filename}'."

                source_in_sec = self._hms_to_seconds(args.source_start_time)
                source_out_sec = self._hms_to_seconds(args.source_end_time)

                if source_out_sec > source_total_duration_sec and not math.isclose(source_out_sec, source_total_duration_sec, abs_tol=0.01):
                    return f"Error: source_end_time ({source_out_sec:.3f}s) is beyond the source file's total duration ({source_total_duration_sec:.3f}s)."
                if source_out_sec > source_total_duration_sec:
                    source_out_sec = source_total_duration_sec
                if source_in_sec >= source_out_sec:
                    return "Error: The source_start_time must be before the source_end_time."
                duration_sec = source_out_sec - source_in_sec
        except Exception as e:
            return f"Error: Could not read metadata from source file '{args.source_filename}'. It may be corrupt or an unsupported format. FFmpeg error: {e}"

        # --- 3. Assemble Common Clip Data ---
        clip_data = {
            "clip_id": args.clip_id,
            "source_path": source_path,
            "source_in_sec": source_in_sec,
            "source_out_sec": source_out_sec,
            "source_total_duration_sec": source_total_duration_sec,
            "duration_sec": duration_sec,
            "track_index": args.track_index,
            "description": args.clip_description,
            "source_frame_rate": source_fps,
            "source_width": source_width,
            "source_height": source_height,
            "has_audio": has_audio,
        }

        # --- 4. Dispatch to Behavior Handler ---
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
        
        # --- "STRICT BUT HELPFUL" LOGIC ---
        # 1. Define a tiny tolerance to account for float precision, not for "snapping".
        #    Half a frame's duration is a perfect, context-aware tolerance.
        timeline_fps = self._get_or_infer_frame_rate(state)
        tolerance = (1.0 / timeline_fps) / 2.0 if timeline_fps > 0 else 0.001

        # 2. Identify all valid cut points on the track.
        #    A valid point is the start (0.0) or the exact end of any existing clip.
        clips_on_track = state.get_clips_on_track(args.track_index)
        valid_cut_points = {0.0}
        for c in clips_on_track:
            valid_cut_points.add(c.timeline_start_sec + c.duration_sec)

        # 3. Check if the requested insertion time matches a valid cut point within tolerance.
        snapped_start_time = None
        for point in valid_cut_points:
            if math.isclose(timeline_start_sec, point, abs_tol=tolerance):
                snapped_start_time = point # Snap to the exact boundary for perfect alignment
                break

        # 4. If no match is found, fail with the "Golden Error Message".
        if snapped_start_time is None:
            sorted_points = sorted(list(valid_cut_points))
            points_str = ", ".join([f"{p:.3f}s" for p in sorted_points])
            return (
                f"Error: No cut exists at the requested insertion time of {timeline_start_sec:.3f}s. "
                f"Valid insertion points on track {args.track_index} are: [{points_str}]. "
                "To create a new cut point, use the 'split_clip' tool. "
                "To see which clips are at these boundaries, use 'get_timeline_summary'."
            )

        # --- If check passed, proceed with the ripple insert ---
        clip_data['timeline_start_sec'] = snapped_start_time
        
        shifted_count = 0
        for clip in clips_on_track:
            # If a clip starts at or after the insertion point, shift it right.
            # Use the snapped time for comparison to avoid float precision issues.
            if clip.timeline_start_sec >= snapped_start_time:
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

            # Prevent splitting an existing clip, as it's an ambiguous operation.
            if clip.timeline_start_sec < timeline_start_sec and clip_end_sec > replace_end_sec:
                return f"Error: This action would split the existing clip '{clip.clip_id}'. This is not supported. Please adjust the timeline or use an 'insert' operation at a boundary."

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