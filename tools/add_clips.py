# codec/tools/add_clips.py
import os
import math
import ffmpeg
from typing import Literal, Optional, TYPE_CHECKING, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from collections import defaultdict

from google import genai
from .base import BaseTool
from state import TimelineClip

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State


class ClipToAdd(BaseModel):
    """A Pydantic model representing a single clip to be added to the timeline."""
    clip_id: str = Field(
        ...,
        description="A unique name for the new clip (e.g., 'intro_scene', 'b-roll_1'). This is used as an identifier for future operations."
    )
    source_filename: str = Field(
        ...,
        description="The exact name of the video or image file from the media library that this clip will be cut from (e.g., 'interview.mp4', 'title_card.png')."
    )
    track: str = Field(
        ...,
        description="The target track for this clip, specified in NLE format (e.g., 'V1', 'A1', 'V2'). 'V' for video, 'A' for audio.",
        pattern=r"^[VAva]\d+$"
    )
    source_in: str = Field(
        ...,
        description="The timestamp where the clip begins in the source asset. For video, this is a specific time like '00:01:30.000'. For static images, this must be '00:00:00.000'.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    source_out: str = Field(
        ...,
        description="The timestamp where the clip ends in the source asset. For video, this is a specific time. For static images, this defines the desired display duration (e.g., '00:00:05.000' shows the image for 5 seconds).",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    timeline_start: str = Field(
        "00:00:00.000",
        description="The timestamp on the main timeline where this new clip should be placed. When using 'insert' behavior, this must be at an existing cut point. This is ignored when using 'append'.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    insertion_behavior: Literal["append", "insert", "replace"] = Field(
        "append",
        description="Controls how the clip is added. 'append' adds to the end of a track. 'insert' shifts subsequent clips but requires placing at an existing cut. 'replace' overwrites existing content."
    )
    description: Optional[str] = Field(
        None,
        description="A description for organizational purposes. Use this to describe anything you want to remember about this clip, include specifics if necessary."
    )


class AddClipsArgs(BaseModel):
    """Arguments for the add_clips tool."""
    clips: List[ClipToAdd] = Field(
        ...,
        min_length=1,
        description="A list of one or more clips to add to the timeline in a single, atomic operation."
    )


class _ValidatedClipInfo(BaseModel):
    """Internal data structure to hold fully validated clip data before commit."""
    # Core TimelineClip fields
    clip_id: str
    source_path: str
    source_in_sec: float
    source_out_sec: float
    source_total_duration_sec: float
    duration_sec: float
    track_type: Literal['video', 'audio']
    track_number: int
    description: Optional[str]
    source_frame_rate: float
    source_width: int
    source_height: int
    has_audio: bool
    # Metadata for the commit phase
    timeline_start_sec: float
    insertion_behavior: Literal["append", "insert", "replace"]


class AddClipsTool(BaseTool):
    """
    A tool to add one or more new clips to the main timeline. This is the primary tool for building an edit.
    It operates atomically: either all clips are added successfully, or none are, preventing partial edits.
    """

    @property
    def name(self) -> str:
        return "add_clips"

    @property
    def description(self) -> str:
        return (
            "Atomically adds one or more clips from source files (video, image, audio) to the timeline. "
            "This is the main tool for constructing the edit. It supports appending to a track, inserting at a cut point, or replacing content. "
            "The entire operation is all-or-nothing; if any single clip fails validation, no changes are made to the timeline."
        )

    @property
    def args_schema(self):
        return AddClipsArgs

    def _hms_to_seconds(self, time_str: str) -> float:
        """Converts HH:MM:SS.mmm format to total seconds."""
        parts = time_str.split(':')
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split('.')
        s = int(s_parts[0])
        ms = int(s_parts[1].ljust(3, '0')) if len(s_parts) > 1 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0

    def _get_or_infer_frame_rate(self, state: 'State') -> float:
        """Gets sequence frame rate from state or infers it from the first video clip."""
        if state.frame_rate:
            return state.frame_rate
        first_video_clip = next((c for c in state.timeline if c.track_type == 'video'), None)
        return first_video_clip.source_frame_rate if first_video_clip else 24.0

    def _validate_single_clip(
        self,
        clip_def: ClipToAdd,
        state: 'State',
        temp_clip_ids: set,
        temp_track_durations: Dict[Tuple[str, int], float]
    ) -> Tuple[Optional[_ValidatedClipInfo], Optional[str]]:
        """
        Validates a single clip definition.
        Returns a tuple of (_ValidatedClipInfo, None) on success, or (None, error_string) on failure.
        """
        # 1a. Unique Clip ID
        if clip_def.clip_id in temp_clip_ids:
            return None, "A clip with this ID already exists or is duplicated in this request."
        
        # 1b. Source File Exists
        source_path = os.path.join(state.assets_directory, clip_def.source_filename)
        if not os.path.exists(source_path):
            return None, f"Source file '{clip_def.source_filename}' not found."

        # 1c. Track Parsing
        track_type = 'video' if clip_def.track[0].lower() == 'v' else 'audio'
        track_number = int(clip_def.track[1:])

        try:
            # 1d. Probe and Media Type Validation
            probe = ffmpeg.probe(source_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

            if track_type == 'video' and not video_stream:
                return None, f"Cannot place '{clip_def.source_filename}' on a video track ('{clip_def.track}') because it contains no video stream."
            if track_type == 'audio' and not audio_stream:
                return None, f"Cannot place '{clip_def.source_filename}' on an audio track ('{clip_def.track}') because it contains no audio stream."

            # 1e. Metadata Extraction (Image vs. Video/Audio)
            is_image = source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
            if is_image:
                if track_type != 'video':
                    return None, "Images can only be placed on video tracks."
                if not math.isclose(self._hms_to_seconds(clip_def.source_in), 0.0):
                    return None, "For images, 'source_in' must be '00:00:00.000'."
                duration_sec = self._hms_to_seconds(clip_def.source_out)
                if duration_sec <= 0:
                    return None, "For images, 'source_out' must be a positive duration."
                source_in_sec, source_out_sec = 0.0, duration_sec
                source_total_duration_sec = duration_sec
                source_fps = self._get_or_infer_frame_rate(state)
                source_width, source_height = video_stream.get('width'), video_stream.get('height')
            else: # Video or Audio-only
                stream = video_stream if track_type == 'video' else audio_stream
                duration_str = stream.get('duration') or probe['format'].get('duration')
                if not duration_str:
                    return None, "Could not determine source duration."
                source_total_duration_sec = float(duration_str)
                source_in_sec = self._hms_to_seconds(clip_def.source_in)
                source_out_sec = self._hms_to_seconds(clip_def.source_out)

                if source_out_sec > source_total_duration_sec and not math.isclose(source_out_sec, source_total_duration_sec, abs_tol=0.01):
                    return None, f"source_out ({source_out_sec:.3f}s) is beyond the source's duration ({source_total_duration_sec:.3f}s)."
                if source_in_sec >= source_out_sec:
                    return None, "source_in must be before source_out."
                duration_sec = source_out_sec - source_in_sec
                
                source_width = video_stream.get('width', 0) if video_stream else 0
                source_height = video_stream.get('height', 0) if video_stream else 0
                source_fps = 0.0
                if video_stream and 'r_frame_rate' in video_stream:
                    num, den = map(int, video_stream['r_frame_rate'].split('/'))
                    if den > 0: source_fps = num / den

            # 1f. Timeline Placement Validation
            timeline_start_sec = self._hms_to_seconds(clip_def.timeline_start)
            if clip_def.insertion_behavior == 'append':
                track_key = (track_type, track_number)
                if track_key not in temp_track_durations:
                    temp_track_durations[track_key] = state.get_specific_track_duration(track_type, track_number)
                timeline_start_sec = temp_track_durations[track_key]
                temp_track_durations[track_key] += duration_sec
            elif clip_def.insertion_behavior == 'insert':
                timeline_fps = self._get_or_infer_frame_rate(state)
                tolerance = (1.0 / timeline_fps) / 2.0 if timeline_fps > 0 else 0.001
                clips_on_track = state.get_clips_on_specific_track(track_type, track_number)
                valid_cut_points = {0.0} | {c.timeline_start_sec + c.duration_sec for c in clips_on_track}
                if not any(math.isclose(timeline_start_sec, p, abs_tol=tolerance) for p in valid_cut_points):
                    points_str = ", ".join([f"{p:.3f}s" for p in sorted(list(valid_cut_points))])
                    return None, f"'insert' requires placing at a valid cut point. Valid points on track {clip_def.track} are: [{points_str}]."

            # 1g. Assemble Validated Data
            validated_info = _ValidatedClipInfo(
                clip_id=clip_def.clip_id, source_path=source_path, source_in_sec=source_in_sec,
                source_out_sec=source_out_sec, source_total_duration_sec=source_total_duration_sec,
                duration_sec=duration_sec, track_type=track_type, track_number=track_number,
                description=clip_def.description, source_frame_rate=source_fps, source_width=source_width,
                source_height=source_height, has_audio=audio_stream is not None,
                timeline_start_sec=timeline_start_sec, insertion_behavior=clip_def.insertion_behavior
            )
            return validated_info, None

        except Exception as e:
            return None, f"An unexpected error occurred during validation: {e}"

    def execute(self, state: 'State', args: AddClipsArgs, client: 'genai.Client') -> str:
        # --- PHASE 1: VALIDATION (ALL OR NOTHING) ---
        validated_clips: List[_ValidatedClipInfo] = []
        errors = []
        temp_clip_ids = {c.clip_id for c in state.timeline}
        temp_track_durations = {}  # Correctly stack 'append' operations within this call

        for i, clip_def in enumerate(args.clips):
            validated_info, error = self._validate_single_clip(clip_def, state, temp_clip_ids, temp_track_durations)
            if error:
                errors.append(f"Error in clip #{i+1} ('{clip_def.clip_id}'): {error}")
            else:
                validated_clips.append(validated_info)
                temp_clip_ids.add(validated_info.clip_id)

        if errors:
            return "Operation failed. Please fix the following errors:\n- " + "\n- ".join(errors)

        # --- PHASE 2: COMMIT (APPLY CHANGES TO STATE) ---
        
        # 2a. Handle 'replace' by identifying clips to delete
        ids_to_delete = set()
        for clip in validated_clips:
            if clip.insertion_behavior == 'replace':
                start, end = clip.timeline_start_sec, clip.timeline_start_sec + clip.duration_sec
                for existing_clip in state.get_clips_on_specific_track(clip.track_type, clip.track_number):
                    existing_end = existing_clip.timeline_start_sec + existing_clip.duration_sec
                    if max(existing_clip.timeline_start_sec, start) < min(existing_end, end):
                        ids_to_delete.add(existing_clip.clip_id)
        
        # 2b. Handle 'insert': calculate total time to shift existing clips
        shifts = defaultdict(float)
        for clip in validated_clips:
            if clip.insertion_behavior == 'insert':
                key = (clip.track_type, clip.track_number, clip.timeline_start_sec)
                shifts[key] += clip.duration_sec

        # 2c. Apply changes to a new timeline list
        new_timeline = []
        for clip in state.timeline:
            if clip.clip_id in ids_to_delete:
                continue  # Clip is deleted by a 'replace' operation

            total_shift = 0.0
            for (track_type, track_number, insert_point), duration in shifts.items():
                # A clip is shifted if it's on the same track and starts at or after the insertion point
                if clip.track_type == track_type and clip.track_number == track_number and clip.timeline_start_sec >= insert_point:
                    total_shift += duration
            
            clip.timeline_start_sec += total_shift
            new_timeline.append(clip)

        # 2d. Add the new clips, correctly handling sequential inserts
        insert_cursors = defaultdict(float) # Tracks start time for sequential inserts
        for clip_info in validated_clips:
            final_start_time = clip_info.timeline_start_sec
            
            if clip_info.insertion_behavior == 'insert':
                key = (clip_info.track_type, clip_info.track_number, clip_info.timeline_start_sec)
                if key not in insert_cursors:
                    insert_cursors[key] = clip_info.timeline_start_sec # Initialize cursor
                
                final_start_time = insert_cursors[key]
                insert_cursors[key] += clip_info.duration_sec # Move cursor for next insert at this point

            new_clip = TimelineClip(
                clip_id=clip_info.clip_id, source_path=clip_info.source_path,
                source_in_sec=clip_info.source_in_sec, source_out_sec=clip_info.source_out_sec,
                source_total_duration_sec=clip_info.source_total_duration_sec,
                timeline_start_sec=final_start_time, duration_sec=clip_info.duration_sec,
                track_type=clip_info.track_type, track_number=clip_info.track_number,
                description=clip_info.description, source_frame_rate=clip_info.source_frame_rate,
                source_width=clip_info.source_width, source_height=clip_info.source_height,
                has_audio=clip_info.has_audio
            )
            new_timeline.append(new_clip)

        # 2e. Finalize state
        state.timeline = new_timeline
        state._sort_timeline()

        return f"Successfully added {len(validated_clips)} clips to the timeline."