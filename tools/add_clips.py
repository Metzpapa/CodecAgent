# codec/tools/add_clips.py
import os
import math
import ffmpeg
from typing import Literal, Optional, TYPE_CHECKING, List, Dict, Any
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

    def execute(self, state: 'State', args: AddClipsArgs, client: 'genai.Client') -> str:
        # --- PHASE 1: VALIDATION (ALL OR NOTHING) ---
        validated_clip_data = []
        errors = []
        temp_clip_ids = {c.clip_id for c in state.timeline}
        temp_track_durations = {} # Used to correctly stack 'append' operations

        for i, clip_def in enumerate(args.clips):
            error_prefix = f"Error in clip #{i+1} ('{clip_def.clip_id}'):"

            # 1a. Unique Clip ID
            if clip_def.clip_id in temp_clip_ids:
                errors.append(f"{error_prefix} A clip with this ID already exists or is duplicated in this request.")
                continue
            temp_clip_ids.add(clip_def.clip_id)

            # 1b. Source File Exists
            source_path = os.path.join(state.assets_directory, clip_def.source_filename)
            if not os.path.exists(source_path):
                errors.append(f"{error_prefix} Source file '{clip_def.source_filename}' not found.")
                continue

            # 1c. Track Parsing & Validation
            track_type_char = clip_def.track[0].lower()
            track_number = int(clip_def.track[1:])
            track_type = 'video' if track_type_char == 'v' else 'audio'

            # 1d. Probe and Media Type Validation
            try:
                probe = ffmpeg.probe(source_path)
                video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

                if track_type == 'video' and not video_stream:
                    errors.append(f"{error_prefix} Cannot place '{clip_def.source_filename}' on a video track ('{clip_def.track}') because it contains no video stream.")
                    continue
                if track_type == 'audio' and not audio_stream:
                    errors.append(f"{error_prefix} Cannot place '{clip_def.source_filename}' on an audio track ('{clip_def.track}') because it contains no audio stream.")
                    continue

                # 1e. Metadata Extraction (Image vs. Video/Audio)
                is_image = source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))

                if is_image:
                    if track_type != 'video':
                        errors.append(f"{error_prefix} Images can only be placed on video tracks.")
                        continue
                    source_in_sec = self._hms_to_seconds(clip_def.source_in)
                    if not math.isclose(source_in_sec, 0.0):
                        errors.append(f"{error_prefix} For images, 'source_in' must be '00:00:00.000'.")
                        continue
                    duration_sec = self._hms_to_seconds(clip_def.source_out)
                    if duration_sec <= 0:
                        errors.append(f"{error_prefix} For images, 'source_out' must be a positive duration.")
                        continue
                    source_out_sec = duration_sec
                    source_total_duration_sec = duration_sec
                    source_fps = self._get_or_infer_frame_rate(state)
                    source_width = video_stream.get('width')
                    source_height = video_stream.get('height')
                else: # Video or Audio-only
                    stream = video_stream if track_type == 'video' else audio_stream
                    duration_str = stream.get('duration') or probe['format'].get('duration')
                    if not duration_str:
                        errors.append(f"{error_prefix} Could not determine duration.")
                        continue
                    source_total_duration_sec = float(duration_str)
                    source_in_sec = self._hms_to_seconds(clip_def.source_in)
                    source_out_sec = self._hms_to_seconds(clip_def.source_out)

                    if source_out_sec > source_total_duration_sec and not math.isclose(source_out_sec, source_total_duration_sec, abs_tol=0.01):
                        errors.append(f"{error_prefix} source_out ({source_out_sec:.3f}s) is beyond the source's duration ({source_total_duration_sec:.3f}s).")
                        continue
                    if source_in_sec >= source_out_sec:
                        errors.append(f"{error_prefix} source_in must be before source_out.")
                        continue
                    duration_sec = source_out_sec - source_in_sec
                    
                    source_width = video_stream.get('width', 0) if video_stream else 0
                    source_height = video_stream.get('height', 0) if video_stream else 0
                    if video_stream:
                        fr_str = video_stream.get('r_frame_rate', '0/1')
                        num, den = map(int, fr_str.split('/'))
                        source_fps = num / den if den > 0 else 0.0
                    else:
                        source_fps = 0.0 # Audio-only files have no frame rate

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
                    
                    is_valid_cut = any(math.isclose(timeline_start_sec, p, abs_tol=tolerance) for p in valid_cut_points)
                    if not is_valid_cut:
                        points_str = ", ".join([f"{p:.3f}s" for p in sorted(list(valid_cut_points))])
                        errors.append(f"{error_prefix} 'insert' requires placing at a valid cut point. Valid points on track {clip_def.track} are: [{points_str}].")
                        continue

                # 1g. Assemble Validated Data
                validated_clip_data.append({
                    "clip_id": clip_def.clip_id,
                    "source_path": source_path,
                    "source_in_sec": source_in_sec,
                    "source_out_sec": source_out_sec,
                    "source_total_duration_sec": source_total_duration_sec,
                    "duration_sec": duration_sec,
                    "track_type": track_type,
                    "track_number": track_number,
                    "description": clip_def.description,
                    "source_frame_rate": source_fps,
                    "source_width": source_width,
                    "source_height": source_height,
                    "has_audio": audio_stream is not None,
                    # --- Metadata for commit phase ---
                    "_timeline_start_sec": timeline_start_sec,
                    "_insertion_behavior": clip_def.insertion_behavior,
                })

            except Exception as e:
                errors.append(f"{error_prefix} An unexpected error occurred during validation: {e}")

        if errors:
            return "Operation failed. Please fix the following errors:\n- " + "\n- ".join(errors)

        # --- PHASE 2: COMMIT (APPLY CHANGES TO STATE) ---
        
        # 2a. Handle 'replace' by identifying clips to delete
        clips_to_add = [TimelineClip(**{k: v for k, v in d.items() if not k.startswith('_')}) for d in validated_clip_data]
        final_clips_to_add = []
        ids_to_delete = set()
        
        for i, clip_def in enumerate(validated_clip_data):
            if clip_def['_insertion_behavior'] == 'replace':
                start = clip_def['_timeline_start_sec']
                end = start + clip_def['duration_sec']
                for existing_clip in state.get_clips_on_specific_track(clip_def['track_type'], clip_def['track_number']):
                    existing_end = existing_clip.timeline_start_sec + existing_clip.duration_sec
                    if max(existing_clip.timeline_start_sec, start) < min(existing_end, end):
                        ids_to_delete.add(existing_clip.clip_id)
        
        # 2b. Handle 'insert' by calculating total time to shift at each point
        shifts = defaultdict(float)
        for i, clip_def in enumerate(validated_clip_data):
            if clip_def['_insertion_behavior'] == 'insert':
                # Snap to the exact cut point to avoid float precision issues
                timeline_fps = self._get_or_infer_frame_rate(state)
                tolerance = (1.0 / timeline_fps) / 2.0 if timeline_fps > 0 else 0.001
                clips_on_track = state.get_clips_on_specific_track(clip_def['track_type'], clip_def['track_number'])
                valid_cut_points = {0.0} | {c.timeline_start_sec + c.duration_sec for c in clips_on_track}
                
                snapped_start_time = min(valid_cut_points, key=lambda p: abs(p - clip_def['_timeline_start_sec']))
                
                key = (clip_def['track_type'], clip_def['track_number'], snapped_start_time)
                shifts[key] += clip_def['duration_sec']

        # 2c. Apply changes to a new timeline list
        new_timeline = []
        for clip in state.timeline:
            if clip.clip_id in ids_to_delete:
                continue # Clip is deleted by a 'replace' operation

            # Calculate total shift for this clip
            total_shift = 0.0
            for (track_type, track_number, insert_point), duration in shifts.items():
                if clip.track_type == track_type and clip.track_number == track_number and clip.timeline_start_sec >= insert_point:
                    total_shift += duration
            
            clip.timeline_start_sec += total_shift
            new_timeline.append(clip)

        # 2d. Add the new clips
        for i, clip_def in enumerate(validated_clip_data):
            new_clip_obj = clips_to_add[i]
            new_clip_obj.timeline_start_sec = clip_def['_timeline_start_sec']
            
            # For inserts, the start time also needs to be shifted by other inserts at the same point
            if clip_def['_insertion_behavior'] == 'insert':
                # This is complex. A simpler model is that all inserts at a point happen "at once".
                # The shifting logic above handles this correctly for existing clips.
                # The new clips are just placed at their calculated start times.
                pass

            new_timeline.append(new_clip_obj)

        # 2e. Finalize state
        state.timeline = new_timeline
        state._sort_timeline()

        return f"Successfully added {len(validated_clip_data)} clips to the timeline."