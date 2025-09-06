# codecagent/codec/tools/transform.py

import logging
import ffmpeg
from typing import List, Optional, Literal, TYPE_CHECKING, Tuple
from pathlib import Path
import openai
from pydantic import BaseModel, Field

from PIL import Image

from .base import BaseTool
from ..state import Keyframe, TimelineClip
from ..utils import hms_to_seconds
from .. import rendering
from .. import visuals  # <-- IMPORT THE NEW VISUALS MODULE

if TYPE_CHECKING:
    from ..state import State


# --- Pydantic Models for Tool Arguments (Unchanged) ---

class TransformProperties(BaseModel):
    """A dictionary of properties to set for a keyframe."""
    position: Optional[List[float]] = Field(
        None,
        min_length=2,
        max_length=2,
        description="The [x, y] position of the clip's anchor point in normalized coordinates. (0.0, 0.0) is the top-left corner of the sequence, (1.0, 1.0) is the bottom-right."
    )
    scale: Optional[float] = Field(
        None,
        description="The scale of the clip as a multiplier. 1.0 is original size, 0.5 is half size, 2.0 is double size."
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
        description="The [x, y] anchor point within the clip itself in normalized coordinates, relative to its top-left corner. By default, this is the clip's center (0.5, 0.5)."
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
            "animation, and keyframing tasks. All coordinates for 'position' and 'anchor_point' are normalized, where "
            "(0.0, 0.0) is the top-left corner and (1.0, 1.0) is the bottom-right. To **update** an existing keyframe, "
            "call this tool again with the same `at_time`. To **delete** a keyframe for a specific property, set that "
            "property's value to `null` at the precise `at_time`. For accurate updates or deletions, you should first "
            "use `get_timeline_summary` to find the exact timestamp of the keyframe you wish to modify."
        )

    @property
    def args_schema(self):
        return TransformArgs

    def execute(self, state: 'State', args: TransformArgs, client: openai.OpenAI, tmpdir: str) -> str:
        modified_clips = set()
        errors = []
        applied_transformations = []

        for i, t in enumerate(args.transformations):
            target_clip = state.find_clip_by_id(t.clip_id)
            if not target_clip:
                errors.append(f"Transformation #{i+1}: Clip with ID '{t.clip_id}' not found.")
                continue

            # --- 1. Determine Keyframe Time ---
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

            # --- 2. Find or Create Keyframe (UPSERT LOGIC) ---
            existing_keyframe = None
            for kf in target_clip.transformations:
                if abs(kf.time_sec - keyframe_relative_sec) < 0.001: # Tolerance for float comparison
                    existing_keyframe = kf
                    break
            
            properties_to_apply = t.properties.model_dump() # Includes nulls for deletion

            if existing_keyframe:
                # --- UPDATE/DELETE logic for an existing keyframe ---
                for prop_name, prop_value in properties_to_apply.items():
                    # Convert list to tuple for position/anchor_point before setting
                    if prop_value is not None and prop_name in ['position', 'anchor_point']:
                        prop_value = tuple(prop_value)
                    setattr(existing_keyframe, prop_name, prop_value)
                
                existing_keyframe.interpolation = t.interpolation

                # Clean up keyframe if it has no properties left (and isn't the base keyframe)
                is_base_keyframe = abs(existing_keyframe.time_sec) < 0.001
                has_properties = any([
                    existing_keyframe.position, existing_keyframe.scale is not None,
                    existing_keyframe.rotation is not None, existing_keyframe.opacity is not None,
                    existing_keyframe.anchor_point
                ])
                if not has_properties and not is_base_keyframe:
                    target_clip.transformations.remove(existing_keyframe)

            else:
                # --- CREATE logic for a new keyframe ---
                keyframe_data = {k: v for k, v in properties_to_apply.items() if v is not None}
                if 'position' in keyframe_data: keyframe_data['position'] = tuple(keyframe_data['position'])
                if 'anchor_point' in keyframe_data: keyframe_data['anchor_point'] = tuple(keyframe_data['anchor_point'])

                if keyframe_data: # Only create if there are properties to set
                    new_keyframe = Keyframe(time_sec=keyframe_relative_sec, interpolation=t.interpolation, **keyframe_data)
                    target_clip.transformations.append(new_keyframe)

            target_clip.transformations.sort(key=lambda kf: kf.time_sec)
            modified_clips.add(t.clip_id)
            applied_transformations.append({'clip': target_clip, 'timeline_sec': keyframe_timeline_sec})

        if errors:
            return "Operation failed with errors:\n- " + "\n- ".join(errors)

        # --- PHASE 3: GENERATE AND UPLOAD VISUAL PREVIEWS ---
        preview_count = 0
        for transform_info in applied_transformations:
            try:
                file_id, local_path = self._generate_and_upload_transform_preview(
                    state, client, transform_info['clip'], transform_info['timeline_sec'], tmpdir
                )
                state.new_multimodal_files.append((file_id, local_path))
                state.uploaded_files.append(file_id)
                preview_count += 1
            except Exception as e:
                logging.error(f"Failed to generate preview for clip '{transform_info['clip'].clip_id}': {e}", exc_info=True)

        # --- PHASE 4: FORMULATE FINAL RESPONSE ---
        confirmation = (
            f"Successfully applied {len(args.transformations)} transformations to {len(modified_clips)} clips: "
            f"{', '.join(sorted(list(modified_clips)))}."
        )
        if preview_count > 0:
            confirmation += (
                f" Generated {preview_count} side-by-side preview frames. "
                "On the left is the 'Source Monitor' showing the original frame, and on the right is the 'Program Monitor' "
                "showing the fully transformed and composited result."
            )
        
        return confirmation

    def _generate_and_upload_transform_preview(
        self, state: 'State', client: openai.OpenAI, clip: TimelineClip, timeline_sec: float, tmpdir: str
    ) -> Tuple[str, str]:
        """
        Orchestrates the creation of a side-by-side preview image and uploads it.
        """
        composite_image_path = self._create_side_by_side_preview(state, clip, timeline_sec, tmpdir)
        with open(composite_image_path, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="vision")
        return uploaded_file.id, str(composite_image_path)

    def _create_side_by_side_preview(
        self, state: 'State', clip: TimelineClip, timeline_sec: float, tmpdir: str
    ) -> str:
        """
        Generates a side-by-side image comparing the source frame to the final
        composited frame and returns the path to the final image.
        This method now uses the shared visuals module for composition.
        """
        tmp_path = Path(tmpdir)
        
        # 1. Render the "Program" (timeline) frame
        program_frame_path = tmp_path / f"program_{clip.clip_id}_{timeline_sec:.3f}.png"
        rendering.render_preview_frame(
            state=state,
            timeline_sec=timeline_sec,
            output_path=str(program_frame_path),
            tmpdir=tmpdir
        )

        # 2. Extract the corresponding "Source" frame
        keyframe_relative_sec = timeline_sec - clip.timeline_start_sec
        source_timestamp_sec = clip.source_in_sec + keyframe_relative_sec
        
        source_frame_path = tmp_path / f"source_{clip.clip_id}_{timeline_sec:.3f}.png"
        try:
            (
                ffmpeg.input(clip.source_path, ss=source_timestamp_sec)
                .output(str(source_frame_path), vframes=1, format='image2', vcodec='png')
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg failed to extract source frame: {e.stderr.decode()}")
            raise

        # 3. Load images and compose using the shared visuals module
        _, seq_width, seq_height = state.get_sequence_properties()
        
        with Image.open(source_frame_path) as src_img, Image.open(program_frame_path) as prog_img:
            # Ensure images are consistently sized
            src_img = src_img.resize((seq_width, seq_height), Image.Resampling.LANCZOS)
            prog_img = prog_img.resize((seq_width, seq_height), Image.Resampling.LANCZOS)

            # Use the centralized composition function
            composite_img = visuals.compose_side_by_side(
                image_left=src_img,
                label_left="Source Monitor",
                image_right=prog_img,
                label_right="Program Monitor"
            )

            final_output_path = tmp_path / f"preview_{clip.clip_id}_{timeline_sec:.3f}_composite.png"
            composite_img.save(final_output_path)

        return str(final_output_path)