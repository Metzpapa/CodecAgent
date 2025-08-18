# codec/tools/transform.py

import os
import tempfile
import logging
from typing import List, Optional, Literal, TYPE_CHECKING, Tuple
from pathlib import Path
import openai
import ffmpeg
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont

from .base import BaseTool
from ..state import Keyframe, TimelineClip
from ..utils import hms_to_seconds

if TYPE_CHECKING:
    from ..state import State


# --- Pydantic Models for Tool Arguments ---

class TransformProperties(BaseModel):
    """A dictionary of properties to set for a keyframe."""
    position: Optional[List[float]] = Field(None, min_length=2, max_length=2)
    scale: Optional[float] = None
    rotation: Optional[float] = None
    opacity: Optional[float] = None
    anchor_point: Optional[List[float]] = Field(None, min_length=2, max_length=2)


class Transformation(BaseModel):
    """Defines a single transformation or keyframe to be applied to a clip."""
    clip_id: str = Field(..., description="The unique identifier of the clip to transform.")
    properties: TransformProperties = Field(..., description="A dictionary of properties to set for this keyframe.")
    at_time: Optional[str] = Field(
        None,
        description="The timeline timestamp for this keyframe. Omit for a static transform applied at the start of the clip.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    interpolation: Literal["linear", "easy ease", "hold"] = Field(
        "easy ease",
        description="The interpolation method for this keyframe."
    )


class TransformArgs(BaseModel):
    """Arguments for the transform tool."""
    transformations: List[Transformation] = Field(
        ...,
        description="A list of one or more transformation objects to apply."
    )


class TransformTool(BaseTool):
    """
    A tool to apply spatial transformations and keyframes to clips on the timeline.
    """

    @property
    def name(self) -> str:
        return "transform"

    @property
    def description(self) -> str:
        return (
            "Applies one or more spatial transformations to one or more clips. This is the primary tool for all layout, "
            "animation, and keyframing tasks, including Picture-in-Picture, split screens, and complex motion."
        )

    @property
    def args_schema(self):
        return TransformArgs

    def execute(self, state: 'State', args: TransformArgs, client: openai.OpenAI) -> str:
        # --- PHASE 1: VALIDATE AND APPLY KEYFRAMES TO STATE ---
        modified_clips = set()
        errors = []
        applied_transformations = []

        for i, t in enumerate(args.transformations):
            target_clip = state.find_clip_by_id(t.clip_id)
            if not target_clip:
                errors.append(f"Transformation #{i+1}: Clip with ID '{t.clip_id}' not found.")
                continue

            if t.at_time:
                keyframe_timeline_sec = hms_to_seconds(t.at_time)
                keyframe_relative_sec = keyframe_timeline_sec - target_clip.timeline_start_sec
            else:
                keyframe_timeline_sec = target_clip.timeline_start_sec
                keyframe_relative_sec = 0.0

            if not (-0.001 <= keyframe_relative_sec <= target_clip.duration_sec + 0.001):
                errors.append(
                    f"Transformation #{i+1}: Keyframe time for clip '{t.clip_id}' ({keyframe_relative_sec:.3f}s) "
                    f"is outside its duration on the timeline (0.0s to {target_clip.duration_sec:.3f}s)."
                )
                continue

            keyframe_data = t.properties.model_dump(exclude_none=True)
            if 'position' in keyframe_data: keyframe_data['position'] = tuple(keyframe_data['position'])
            if 'anchor_point' in keyframe_data: keyframe_data['anchor_point'] = tuple(keyframe_data['anchor_point'])

            new_keyframe = Keyframe(time_sec=keyframe_relative_sec, interpolation=t.interpolation, **keyframe_data)
            target_clip.transformations.append(new_keyframe)
            target_clip.transformations.sort(key=lambda kf: kf.time_sec)
            
            modified_clips.add(t.clip_id)
            applied_transformations.append({'clip': target_clip, 'timeline_sec': keyframe_timeline_sec})

        if errors:
            return "Operation failed with errors:\n- " + "\n- ".join(errors)

        # --- PHASE 2: GENERATE AND UPLOAD VISUAL PREVIEWS ---
        preview_count = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            for transform_info in applied_transformations:
                try:
                    file_id, local_path = self._generate_and_upload_preview_frame(
                        state, client, transform_info['clip'], transform_info['timeline_sec'], tmpdir
                    )
                    state.new_multimodal_files.append((file_id, local_path))
                    state.uploaded_files.append(file_id)
                    preview_count += 1
                except Exception as e:
                    logging.error(f"Failed to generate preview for clip '{transform_info['clip'].clip_id}': {e}", exc_info=True)

        # --- PHASE 3: FORMULATE FINAL RESPONSE ---
        confirmation = (
            f"Successfully applied {len(args.transformations)} transformations to {len(modified_clips)} clips: "
            f"{', '.join(sorted(list(modified_clips)))}."
        )
        if preview_count > 0:
            confirmation += f" Generated {preview_count} preview frames for the agent to view."
        
        return confirmation

    # --- HELPER METHOD FOR VISUAL FEEDBACK ---
    def _generate_and_upload_preview_frame(
        self, state: 'State', client: openai.OpenAI, clip: TimelineClip, timeline_sec: float, tmpdir: str
    ) -> Tuple[str, str]:
        """
        Renders a single frame of a clip with its transformations applied at a specific timeline second.
        """
        # 1. Determine source time and extract the raw frame
        source_time_sec = clip.source_in_sec + (timeline_sec - clip.timeline_start_sec)

        # --- FIX: CLAMP THE SOURCE TIME TO PREVENT FILE NOT FOUND ERRORS ---
        # If the requested time is at or beyond the clip's duration, step back one frame.
        one_frame_duration = 1.0 / clip.source_frame_rate if clip.source_frame_rate > 0 else 0.04
        max_source_time = clip.source_total_duration_sec - one_frame_duration
        
        # Clamp the time to be within the valid range [0, max_source_time]
        source_time_sec = max(0, min(source_time_sec, max_source_time))
        # --- END FIX ---

        raw_frame_path = Path(tmpdir) / f"raw_{clip.clip_id}_{timeline_sec:.3f}.jpg"

        try:
            (
                ffmpeg.input(clip.source_path, ss=source_time_sec)
                .output(str(raw_frame_path), vframes=1, format='image2', vcodec='mjpeg')
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            raise IOError(f"FFmpeg failed to extract frame from '{clip.source_path}': {e.stderr.decode()}")

        # 2. Calculate the effective transformation at this point in time
        clip_relative_sec = timeline_sec - clip.timeline_start_sec
        active_kfs = [kf for kf in clip.transformations if kf.time_sec <= clip_relative_sec + 0.001]
        
        final_props = {"scale": 1.0, "rotation": 0.0, "opacity": 1.0, "position": (0.0, 0.0), "anchor_point": (0.0, 0.0)}
        for kf in sorted(active_kfs, key=lambda k: k.time_sec):
            final_props.update(kf.model_dump(exclude_none=True))

        # 3. Apply transformations using Pillow
        frame_img = Image.open(raw_frame_path)
        
        new_size = (
            int(frame_img.width * final_props['scale']),
            int(frame_img.height * final_props['scale'])
        )
        frame_img = frame_img.resize(new_size, Image.Resampling.LANCZOS)
        frame_img = frame_img.rotate(-final_props['rotation'], expand=True, resample=Image.BICUBIC)

        seq_w, seq_h = state.get_sequence_properties()[1:]
        canvas = Image.new("RGBA", (seq_w, seq_h), (0, 0, 0, 255))

        paste_x = int((seq_w / 2) * (1 + final_props['position'][0]) - (frame_img.width / 2) + (final_props['anchor_point'][0] * frame_img.width))
        paste_y = int((seq_h / 2) * (1 - final_props['position'][1]) - (frame_img.height / 2) - (final_props['anchor_point'][1] * frame_img.height))

        if final_props['opacity'] < 1.0:
            alpha = frame_img.getchannel('A') if frame_img.mode == 'RGBA' else Image.new('L', frame_img.size, 255)
            alpha = alpha.point(lambda i: i * final_props['opacity'])
            frame_img.putalpha(alpha)
        
        canvas.paste(frame_img, (paste_x, paste_y), frame_img if frame_img.mode == 'RGBA' else None)
        
        # 4. Save and Upload
        final_frame_path = Path(tmpdir) / f"final_{clip.clip_id}_{timeline_sec:.3f}.jpg"
        canvas.convert("RGB").save(final_frame_path, "JPEG", quality=90)

        with open(final_frame_path, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="vision")
        
        return uploaded_file.id, str(final_frame_path)