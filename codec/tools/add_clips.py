# codec/tools/add_clips.py
import os
import math
from typing import Literal, Optional, TYPE_CHECKING, List, Dict, Any, Tuple
import openai
from pydantic import BaseModel, Field, model_validator
from collections import defaultdict

from .base import BaseTool
from ..state import TimelineClip
from ..utils import hms_to_seconds, probe_media_file

if TYPE_CHECKING:
    from ..state import State


class ClipToAdd(BaseModel):
    clip_id: str = Field(
        ...,
        description="A unique name for the new clip (e.g., 'intro_scene', 'b-roll_1'). If adding both video and audio, this ID will be used as a base for both (e.g., 'intro_scene_v', 'intro_scene_a')."
    )
    source_filename: str = Field(
        ...,
        description="The exact name of the video, image, or audio file from the media library that this clip will be cut from (e.g., 'interview.mp4', 'title_card.png')."
    )
    video_track: Optional[str] = Field(
        None,
        description="The target video track for this clip (e.g., 'V1', 'V2'). Provide this to include the video component. Must be omitted for audio-only clips.",
        pattern=r"^[Vv]\d+$"
    )
    audio_track: Optional[str] = Field(
        None,
        description="The target audio track for this clip (e.g., 'A1', 'A2'). Provide this to include the audio component. Must be omitted for video-only images.",
        pattern=r"^[Aa]\d+$"
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
        description="The timestamp on the main timeline where this new clip should be placed. When using 'insert' or 'replace', this is a precise start time. This is ignored when using 'append'.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    insertion_behavior: Literal["append", "insert", "replace"] = Field(
        "append",
        description="Controls how the clip is added. 'append' adds to the end of the specified track(s). 'insert' shifts subsequent clips. 'replace' overwrites existing content."
    )
    description: Optional[str] = Field(
        None,
        description="A description for organizational purposes. Use this to describe anything you want to remember about this clip, include specifics if necessary."
    )

    @model_validator(mode='after')
    def check_at_least_one_track(self):
        if not self.video_track and not self.audio_track:
            raise ValueError("At least one of 'video_track' or 'audio_track' must be provided.")
        return self


class AddClipsArgs(BaseModel):
    """Arguments for the add_clips tool."""
    clips: List[ClipToAdd] = Field(
        ...,
        min_length=1,
        description="A list of one or more logical clips to add to the timeline. Each item can specify a video track, an audio track, or both to create a linked A/V clip."
    )


class _ValidatedClipInfo(BaseModel):
    """Internal data structure to hold fully validated clip data before commit."""
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
    timeline_start_sec: float
    insertion_behavior: Literal["append", "insert", "replace"]


class AddClipsTool(BaseTool):
    """
    A tool to add one or more new clips to the main timeline. This is the primary tool for building an edit.
    It can add video-only, audio-only, or linked audio/video clips in a single definition.
    It operates atomically: either all clips are added successfully, or none are, preventing partial edits.
    """

    @property
    def name(self) -> str:
        return "add_clips"

    @property
    def description(self) -> str:
        return (
            "Atomically adds one or more clips to the timeline. This is the main tool for constructing the edit. "
            "It can add video-only, audio-only, or linked A/V clips. "
            "Supports appending to track(s), inserting at a cut point, or replacing content. "
            "The entire operation is all-or-nothing; if any single clip fails validation, no changes are made."
        )

    @property
    def args_schema(self):
        return AddClipsArgs

    def _validate_single_clip_group(
        self,
        clip_def: ClipToAdd,
        state: 'State',
        temp_clip_ids: set,
        temp_track_durations: Dict[Tuple[str, int], float]
    ) -> Tuple[Optional[List[_ValidatedClipInfo]], Optional[str]]:
        """
        Validates a single `ClipToAdd` definition, which may result in one (V or A) or two (V+A) clips.
        Returns a tuple of (list_of_validated_clips, None) on success, or (None, error_string) on failure.
        """
        # 1. Source File and Time Validation
        source_path = os.path.join(state.assets_directory, clip_def.source_filename)
        if not os.path.exists(source_path):
            return None, f"Source file '{clip_def.source_filename}' not found."

        media_info = probe_media_file(source_path)
        if media_info.error:
            return None, f"Error probing '{clip_def.source_filename}': {media_info.error}"

        source_in_sec = hms_to_seconds(clip_def.source_in)
        source_out_sec = hms_to_seconds(clip_def.source_out)
        is_image = source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))

        if is_image:
            if clip_def.audio_track:
                return None, "Cannot add an audio track when the source is an image."
            if not math.isclose(source_in_sec, 0.0):
                return None, "For images, 'source_in' must be '00:00:00.000'."
            duration_sec = source_out_sec
            if duration_sec <= 0:
                return None, "For images, 'source_out' must be a positive duration."
            source_total_duration_sec = duration_sec
        else: # Video or Audio-only
            source_total_duration_sec = media_info.duration_sec
            if source_out_sec > source_total_duration_sec and not math.isclose(source_out_sec, source_total_duration_sec, abs_tol=0.01):
                return None, f"source_out ({source_out_sec:.3f}s) is beyond the source's duration ({source_total_duration_sec:.3f}s)."
            if source_in_sec >= source_out_sec:
                return None, "source_in must be before source_out."
            duration_sec = source_out_sec - source_in_sec

        # 2. Prepare for track-specific validation
        validated_group: List[_ValidatedClipInfo] = []
        is_linked_clip = clip_def.video_track and clip_def.audio_track
        
        # 3. Timeline Placement Validation
        timeline_start_sec = hms_to_seconds(clip_def.timeline_start)
        
        if clip_def.insertion_behavior == 'append':
            # For append, find the max end time of all specified tracks
            latest_end_time = 0.0
            tracks_to_check = []
            if clip_def.video_track: tracks_to_check.append(('video', int(clip_def.video_track[1:])))
            if clip_def.audio_track: tracks_to_check.append(('audio', int(clip_def.audio_track[1:])))
            
            for track_type, track_num in tracks_to_check:
                track_key = (track_type, track_num)
                track_duration = temp_track_durations.get(track_key, state.get_specific_track_duration(track_type, track_num))
                latest_end_time = max(latest_end_time, track_duration)
            
            timeline_start_sec = latest_end_time
            # Update temp durations for subsequent clips in this same call
            for track_type, track_num in tracks_to_check:
                temp_track_durations[(track_type, track_num)] = timeline_start_sec + duration_sec

        elif clip_def.insertion_behavior == 'insert':
            timeline_fps = state.get_sequence_properties()[0]
            tolerance = (1.0 / timeline_fps) / 2.0 if timeline_fps > 0 else 0.001
            
            tracks_to_check = []
            if clip_def.video_track: tracks_to_check.append(('video', int(clip_def.video_track[1:]), clip_def.video_track.upper()))
            if clip_def.audio_track: tracks_to_check.append(('audio', int(clip_def.audio_track[1:]), clip_def.audio_track.upper()))

            for track_type, track_num, track_name in tracks_to_check:
                clips_on_track = state.get_clips_on_specific_track(track_type, track_num)
                valid_cut_points = {0.0} | {c.timeline_start_sec + c.duration_sec for c in clips_on_track}
                if not any(math.isclose(timeline_start_sec, p, abs_tol=tolerance) for p in valid_cut_points):
                    points_str = ", ".join([f"{p:.3f}s" for p in sorted(list(valid_cut_points))])
                    return None, f"'insert' requires placing at a valid cut point. '{timeline_start_sec:.3f}s' is not a valid cut on track {track_name}. Valid points are: [{points_str}]."

        # 4. Process Video Track (if specified)
        if clip_def.video_track:
            if not media_info.has_video:
                return None, f"Cannot place '{clip_def.source_filename}' on a video track because it has no video stream."
            
            clip_id = f"{clip_def.clip_id}_v" if is_linked_clip else clip_def.clip_id
            if clip_id in temp_clip_ids:
                return None, f"A clip with ID '{clip_id}' already exists or is duplicated in this request."

            validated_group.append(_ValidatedClipInfo(
                clip_id=clip_id, source_path=source_path, source_in_sec=source_in_sec,
                source_out_sec=source_out_sec, source_total_duration_sec=source_total_duration_sec,
                duration_sec=duration_sec, track_type='video', track_number=int(clip_def.video_track[1:]),
                description=clip_def.description, 
                source_frame_rate=media_info.frame_rate if not is_image else state.get_sequence_properties()[0],
                source_width=media_info.width, source_height=media_info.height, has_audio=media_info.has_audio,
                timeline_start_sec=timeline_start_sec, insertion_behavior=clip_def.insertion_behavior
            ))

        # 5. Process Audio Track (if specified)
        if clip_def.audio_track:
            if not media_info.has_audio:
                return None, f"Cannot place '{clip_def.source_filename}' on an audio track because it has no audio stream."

            clip_id = f"{clip_def.clip_id}_a" if is_linked_clip else clip_def.clip_id
            if clip_id in temp_clip_ids:
                return None, f"A clip with ID '{clip_id}' already exists or is duplicated in this request."

            validated_group.append(_ValidatedClipInfo(
                clip_id=clip_id, source_path=source_path, source_in_sec=source_in_sec,
                source_out_sec=source_out_sec, source_total_duration_sec=source_total_duration_sec,
                duration_sec=duration_sec, track_type='audio', track_number=int(clip_def.audio_track[1:]),
                description=clip_def.description, source_frame_rate=0, source_width=0, source_height=0,
                has_audio=True, timeline_start_sec=timeline_start_sec, insertion_behavior=clip_def.insertion_behavior
            ))

        return validated_group, None

    def execute(self, state: 'State', args: AddClipsArgs, client: openai.OpenAI, tmpdir: str) -> str:
        # --- PHASE 1: VALIDATION (ALL OR NOTHING) ---
        all_validated_groups: List[List[_ValidatedClipInfo]] = []
        errors = []
        temp_clip_ids = {c.clip_id for c in state.timeline}
        temp_track_durations = {}

        for i, clip_def in enumerate(args.clips):
            validated_group, error = self._validate_single_clip_group(clip_def, state, temp_clip_ids, temp_track_durations)
            if error:
                errors.append(f"Error in clip definition #{i+1} ('{clip_def.clip_id}'): {error}")
            else:
                all_validated_groups.append(validated_group)
                for clip_info in validated_group:
                    temp_clip_ids.add(clip_info.clip_id)

        if errors:
            return "Operation failed. Please fix the following errors:\n- " + "\n- ".join(errors)

        # --- PHASE 2: COMMIT (APPLY CHANGES TO STATE) ---
        flat_validated_clips = [clip for group in all_validated_groups for clip in group]

        # 2a. Handle 'replace' by identifying clips to delete
        ids_to_delete = set()
        for clip in flat_validated_clips:
            if clip.insertion_behavior == 'replace':
                start, end = clip.timeline_start_sec, clip.timeline_start_sec + clip.duration_sec
                for existing_clip in state.get_clips_on_specific_track(clip.track_type, clip.track_number):
                    existing_end = existing_clip.timeline_start_sec + existing_clip.duration_sec
                    if max(existing_clip.timeline_start_sec, start) < min(existing_end, end):
                        ids_to_delete.add(existing_clip.clip_id)
        
        # 2b. Handle 'insert': calculate total time to shift existing clips
        shifts = defaultdict(float)
        for clip in flat_validated_clips:
            if clip.insertion_behavior == 'insert':
                key = (clip.track_type, clip.track_number, clip.timeline_start_sec)
                shifts[key] += clip.duration_sec

        # 2c. Apply changes to a new timeline list
        new_timeline = [c for c in state.timeline if c.clip_id not in ids_to_delete]
        for clip in new_timeline:
            total_shift = 0.0
            for (track_type, track_number, insert_point), duration in shifts.items():
                if clip.track_type == track_type and clip.track_number == track_number and clip.timeline_start_sec >= insert_point:
                    total_shift += duration
            clip.timeline_start_sec += total_shift

        # 2d. Add the new clips
        for clip_info in flat_validated_clips:
            new_clip = TimelineClip(
                clip_id=clip_info.clip_id, source_path=clip_info.source_path,
                source_in_sec=clip_info.source_in_sec, source_out_sec=clip_info.source_out_sec,
                source_total_duration_sec=clip_info.source_total_duration_sec,
                timeline_start_sec=clip_info.timeline_start_sec, duration_sec=clip_info.duration_sec,
                track_type=clip_info.track_type, track_number=clip_info.track_number,
                description=clip_info.description, source_frame_rate=clip_info.source_frame_rate,
                source_width=clip_info.source_width, source_height=clip_info.source_height,
                has_audio=clip_info.has_audio
            )
            new_timeline.append(new_clip)

        # 2e. Finalize state
        state.timeline = new_timeline
        state._sort_timeline()

        return f"Successfully added {len(flat_validated_clips)} clips to the timeline from {len(all_validated_groups)} definitions."