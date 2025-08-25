# codecagent/codec/tools/transform.py

import logging
from typing import List, Optional, Literal, TYPE_CHECKING, Tuple
from pathlib import Path
import openai
from pydantic import BaseModel, Field

from .base import BaseTool
from ..state import Keyframe, TimelineClip
from ..utils import hms_to_seconds
from .. import rendering  # <-- IMPORT THE NEW UNIFIED RENDERING MODULE

if TYPE_CHECKING:
    from ..state import State


# --- Pydantic Models for Tool Arguments (Unchanged) ---

class TransformProperties(BaseModel):
    """A dictionary of properties to set for a keyframe."""
    position: Optional[List[float]] = Field(
        None,
        min_length=2,
        max_length=2,
        description="The [x, y] position of the clip's anchor point in pixels. The origin (0,0) is the top-left corner of the sequence."
    )
    scale: Optional[float] = Field(
        None,
        description="The scale of the clip as a multiplier. 1.0 is original size) 0.5 is half size, 2.0 is double size."
    )
    rotation: Optional[float] = Field(
        None,
        description="The rotation of the clip in degrees. Positive values rotate clockwise."
    )
    opacity: Optional[float] = Field(
        None,
        description="The opacity of the clip as a percentage, from 0 (transparent) to 100 (opaque)."
    )
    anchor_point: Optional[List[float]] = Field(
        None,
        min_length=2,
        max_length=2,
        description="The [x, y] anchor point within the clip itself in pixels, relative to its top-left corner. By default, this is the clip's center."
    )


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

    def execute(self, state: 'State', args: TransformArgs, client: openai.OpenAI, tmpdir: str) -> str:
        # --- PHASE 1: VALIDATE AND APPLY KEYFRAMES TO STATE (Unchanged) ---
        # This is the core responsibility of the tool: modifying the state.
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

        # --- PHASE 2: GENERATE AND UPLOAD VISUAL PREVIEWS (Refactored) ---
        # This now delegates the complex rendering work to the unified rendering module.
        preview_count = 0
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

        # --- PHASE 3: FORMULATE FINAL RESPONSE (Unchanged) ---
        confirmation = (
            f"Successfully applied {len(args.transformations)} transformations to {len(modified_clips)} clips: "
            f"{', '.join(sorted(list(modified_clips)))}."
        )
        if preview_count > 0:
            confirmation += f" Generated {preview_count} preview frames for the agent to view."
        
        return confirmation

    # --- HELPER METHOD FOR VISUAL FEEDBACK (COMPLETELY REWRITTEN) ---
    def _generate_and_upload_preview_frame(
        self, state: 'State', client: openai.OpenAI, clip: TimelineClip, timeline_sec: float, tmpdir: str
    ) -> Tuple[str, str]:
        """
        Renders a single, fully composited frame of the timeline at a specific
        second using the unified MLT rendering engine, then uploads it.
        This guarantees the preview is 100% consistent with the final render.
        """
        # 1. Define the output path for the rendered frame.
        final_frame_path = Path(tmpdir) / f"preview_{clip.clip_id}_{timeline_sec:.3f}.jpg"

        # 2. Call the centralized rendering function. All complex logic for
        # compositing, transformations, and keyframing is handled there.
        rendering.render_preview_frame(
            state=state,
            timeline_sec=timeline_sec,
            output_path=str(final_frame_path),
            tmpdir=tmpdir
        )

        # 3. Upload the resulting frame for the agent to see.
        with open(final_frame_path, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="vision")
        
        return uploaded_file.id, str(final_frame_path)