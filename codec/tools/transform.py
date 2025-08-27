# codec/tools/transform.py

import logging
import ffmpeg
from typing import List, Optional, Literal, TYPE_CHECKING, Tuple
from pathlib import Path
import openai
from pydantic import BaseModel, Field

# --- NEW: Import Pillow for image composition ---
from PIL import Image, ImageDraw, ImageFont

from .base import BaseTool
from ..state import Keyframe, TimelineClip
from ..utils import hms_to_seconds
from .. import rendering

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

        # --- PHASE 3: FORMULATE FINAL RESPONSE (MODIFIED) ---
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

    # --- HELPER METHOD FOR ORCHESTRATING PREVIEW GENERATION ---
    def _generate_and_upload_transform_preview(
        self, state: 'State', client: openai.OpenAI, clip: TimelineClip, timeline_sec: float, tmpdir: str
    ) -> Tuple[str, str]:
        """
        Orchestrates the creation of a side-by-side preview image and uploads it.
        """
        # 1. Create the composite image using our new helper.
        composite_image_path = self._create_side_by_side_preview(state, clip, timeline_sec, tmpdir)

        # 2. Upload the resulting image for the agent to see.
        with open(composite_image_path, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="vision")
        
        return uploaded_file.id, str(composite_image_path)

    # --- NEW: CORE LOGIC FOR CREATING THE SIDE-BY-SIDE PREVIEW IMAGE ---
    def _create_side_by_side_preview(
        self, state: 'State', clip: TimelineClip, timeline_sec: float, tmpdir: str
    ) -> str:
        """
        Generates a side-by-side image comparing the source frame to the final
        composited frame and returns the path to the final image.
        """
        tmp_path = Path(tmpdir)
        
        # --- 1. Generate "Program Monitor" (Right Side) ---
        # This renders the full timeline at the specified moment, ensuring an exact preview.
        program_frame_path = tmp_path / f"program_{clip.clip_id}_{timeline_sec:.3f}.png"
        rendering.render_preview_frame(
            state=state,
            timeline_sec=timeline_sec,
            output_path=str(program_frame_path),
            tmpdir=tmpdir
        )

        # --- 2. Generate "Source Monitor" (Left Side) ---
        # Calculate the corresponding timestamp in the original source file.
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

        # --- 3. Stitch and Label the Images using Pillow ---
        _, seq_width, seq_height = state.get_sequence_properties()
        
        with Image.open(source_frame_path) as src_img, Image.open(program_frame_path) as prog_img:
            # Ensure both images are resized to the sequence dimensions for consistent layout
            src_img = src_img.resize((seq_width, seq_height), Image.Resampling.LANCZOS)
            prog_img = prog_img.resize((seq_width, seq_height), Image.Resampling.LANCZOS)

            # Define layout constants
            padding = 20
            header_height = 50
            font_size = 24
            
            total_width = (seq_width * 2) + (padding * 3)
            total_height = seq_height + header_height + padding

            # Create the final canvas
            composite_img = Image.new('RGB', (total_width, total_height), 'black')
            draw = ImageDraw.Draw(composite_img)
            
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            # Paste the images
            composite_img.paste(src_img, (padding, header_height))
            composite_img.paste(prog_img, (seq_width + padding * 2, header_height))

            # Add labels
            draw.text((padding, padding), "Source Monitor", fill="white", font=font)
            draw.text((seq_width + padding * 2, padding), "Program Monitor", fill="white", font=font)

            # Save the final composite image
            final_output_path = tmp_path / f"preview_{clip.clip_id}_{timeline_sec:.3f}_composite.png"
            composite_img.save(final_output_path)

        return str(final_output_path)