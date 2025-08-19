# codecagent/codec/tools/render_video.py

import os
import logging
import math
from typing import TYPE_CHECKING, List, Callable, Any
from pathlib import Path

# --- NEW: MoviePy v2 Imports ---
# Import specific classes directly from the `moviepy` package.
from moviepy import (
    VideoFileClip, AudioFileClip, CompositeVideoClip,
    CompositeAudioClip, ColorClip
)
import numpy as np  # MoviePy and numpy work well together for interpolation
from pydantic import BaseModel, Field
import openai

from .base import BaseTool
from ..state import Keyframe  # Needed for type hinting

if TYPE_CHECKING:
    from ..state import State


class RenderVideoArgs(BaseModel):
    """Arguments for the render_video tool."""
    output_filename: str = Field(
        "final_render.mp4",
        description="The desired filename for the final rendered video file. Should end in .mp4."
    )


def _create_property_interpolator(
    keyframes: List[Keyframe],
    property_name: str,
    default_value: Any
) -> Callable[[float], Any]:
    """
    Creates a time-based function that interpolates a property based on keyframes.

    This is the core of the animation logic. It returns a function `f(t)` where `t`
    is the time in seconds relative to the start of the clip.

    Args:
        keyframes: The list of all keyframes for a given clip.
        property_name: The name of the property to interpolate (e.g., 'scale').
        default_value: The value to return if no keyframes for this property exist.

    Returns:
        A function that takes a float `t` and returns the interpolated value.
    """
    # Filter for keyframes that actually define this property and sort them by time.
    prop_kfs = sorted(
        [kf for kf in keyframes if getattr(kf, property_name) is not None],
        key=lambda kf: kf.time_sec
    )

    def interpolator(t: float) -> Any:
        if not prop_kfs:
            return default_value

        # Before the first keyframe, hold its value.
        if t <= prop_kfs[0].time_sec:
            return getattr(prop_kfs[0], property_name)

        # After the last keyframe, hold its value.
        if t >= prop_kfs[-1].time_sec:
            return getattr(prop_kfs[-1], property_name)

        # Find the two keyframes that bracket the current time `t`.
        for i in range(len(prop_kfs) - 1):
            kf_prev = prop_kfs[i]
            kf_next = prop_kfs[i + 1]
            if kf_prev.time_sec <= t < kf_next.time_sec:

                # --- Handle different interpolation types ---
                if kf_prev.interpolation == "hold":
                    return getattr(kf_prev, property_name)

                # Calculate progress (0.0 to 1.0) between the two keyframes.
                time_diff = kf_next.time_sec - kf_prev.time_sec
                if time_diff == 0: return getattr(kf_prev, property_name)
                progress = (t - kf_prev.time_sec) / time_diff

                if kf_prev.interpolation == "easy ease":
                    # Apply a standard cubic ease-in-out formula.
                    progress = progress * progress * (3 - 2 * progress)

                # Use numpy for robust linear interpolation of numbers, tuples, etc.
                val_prev = np.array(getattr(kf_prev, property_name))
                val_next = np.array(getattr(kf_next, property_name))
                interpolated_val = val_prev + progress * (val_next - val_prev)

                # Convert back to a tuple if the original was a tuple (e.g., for position).
                return tuple(interpolated_val) if isinstance(getattr(kf_prev, property_name), tuple) else interpolated_val.item()

        return default_value  # Should not be reached if logic is sound.

    return interpolator


class RenderVideoTool(BaseTool):
    """
    Renders the current timeline into a final video file (e.g., an MP4).
    This tool synthesizes all clips, layers, and transformations into a viewable video.
    It uses the MoviePy library to accurately render complex keyframed animations.
    """

    @property
    def name(self) -> str:
        return "render_video"

    @property
    def description(self) -> str:
        return (
            "Renders the current timeline into a final video file (e.g., an MP4), applying all transformations and layering. "
            "This is used to produce the final video deliverable. After calling this, you should call `finish_job` with the returned filename in the `attachments` list."
        )

    @property
    def args_schema(self):
        return RenderVideoArgs

    def execute(self, state: 'State', args: RenderVideoArgs, client: openai.OpenAI, tmpdir: str) -> str:
        if not state.timeline:
            return "Error: Cannot render because the timeline is empty."

        # --- 1. SETUP ---
        job_dir = Path(state.assets_directory).parent
        output_path = job_dir / "output" / args.output_filename
        output_path.parent.mkdir(exist_ok=True)

        duration = state.get_timeline_duration()
        fps, width, height = state.get_sequence_properties()

        processed_video_clips = []
        processed_audio_clips = []

        # --- 2. PROCESS VIDEO CLIPS ---
        video_timeline_clips = sorted(
            [c for c in state.timeline if c.track_type == 'video'],
            key=lambda c: c.track_number  # Lower track numbers are rendered on bottom
        )

        for clip_data in video_timeline_clips:
            try:
                base_clip = VideoFileClip(clip_data.source_path).subclip(
                    clip_data.source_in_sec, clip_data.source_out_sec
                )

                if not clip_data.transformations:
                    # Simple case: No transformations, just place it.
                    final_clip = (base_clip
                        .with_start(clip_data.timeline_start_sec)
                        .with_duration(clip_data.duration_sec)
                        .with_position(('center', 'center')))
                    processed_video_clips.append(final_clip)
                    continue

                # --- Create Interpolator Functions for this clip ---
                scale_func = _create_property_interpolator(clip_data.transformations, 'scale', 1.0)
                rot_func = _create_property_interpolator(clip_data.transformations, 'rotation', 0.0)
                opacity_func = _create_property_interpolator(clip_data.transformations, 'opacity', 100.0)
                pos_func = _create_property_interpolator(clip_data.transformations, 'position', (width / 2, height / 2))
                anchor_func = _create_property_interpolator(clip_data.transformations, 'anchor_point', (base_clip.size[0] / 2, base_clip.size[1] / 2))

                # --- ANIMATION LOGIC (The "Canvas Trick") ---
                # First, apply scale as it affects the clip's intrinsic size.
                scaled_clip = base_clip.resized(lambda t: scale_func(t))
                
                # Determine a safe canvas size that will contain the clip at its largest scaled and rotated size.
                max_scale = max([kf.scale for kf in clip_data.transformations if kf.scale is not None] or [1.0])
                hypot = math.hypot(base_clip.size[0] * max_scale, base_clip.size[1] * max_scale)
                canvas_size = (int(math.ceil(hypot)), int(math.ceil(hypot)))
                canvas_center = (canvas_size[0] / 2, canvas_size[1] / 2)
                
                # 1. Position the scaled clip on the canvas so its anchor point is at the canvas center.
                inner_pos_func = lambda t: (canvas_center[0] - (anchor_func(t)[0] * scale_func(t)),
                                            canvas_center[1] - (anchor_func(t)[1] * scale_func(t)))
                inner_clip = scaled_clip.with_position(inner_pos_func)
                
                # 2. Composite this onto a transparent canvas. `bg_color=None` is crucial for transparency.
                canvas = CompositeVideoClip([inner_clip], size=canvas_size, bg_color=None)
                
                # 3. Rotate the entire canvas around its center. This effectively rotates the clip around its anchor.
                rotated_canvas = canvas.rotated(lambda t: rot_func(t), center=canvas_center)

                # 4. Position the final rotated canvas onto the main sequence.
                # The target position `pos_func(t)` is where the pivot (canvas_center) should be.
                final_pos_func = lambda t: (pos_func(t)[0] - canvas_center[0], pos_func(t)[1] - canvas_center[1])
                final_animated_clip = rotated_canvas.with_position(final_pos_func)

                # 5. Apply opacity, start time, and duration.
                final_clip = (final_animated_clip
                    .with_opacity(lambda t: opacity_func(t) / 100.0)
                    .with_start(clip_data.timeline_start_sec)
                    .with_duration(clip_data.duration_sec))

                processed_video_clips.append(final_clip)

            except Exception as e:
                logging.error(f"Failed to process video clip '{clip_data.clip_id}': {e}", exc_info=True)
                return f"Error: Failed to process video clip '{clip_data.clip_id}'. Details: {e}"

        # --- 3. PROCESS AUDIO CLIPS ---
        audio_timeline_clips = [c for c in state.timeline if c.track_type == 'audio' and c.has_audio]
        for clip_data in audio_timeline_clips:
            audio_clip = (AudioFileClip(clip_data.source_path)
                          .subclip(clip_data.source_in_sec, clip_data.source_out_sec)
                          .with_start(clip_data.timeline_start_sec)
                          .with_duration(clip_data.duration_sec))
            processed_audio_clips.append(audio_clip)

        # --- 4. COMPOSITE AND RENDER ---
        try:
            logging.info("Compositing final video...")
            canvas = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
            
            # Layer the processed video clips on top of the black canvas.
            final_video = CompositeVideoClip([canvas] + processed_video_clips, size=(width, height))
            
            if processed_audio_clips:
                final_audio = CompositeAudioClip(processed_audio_clips)
                final_video.audio = final_audio

            logging.info(f"Rendering final output to {output_path}...")
            final_video.write_videofile(
                str(output_path),
                fps=fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=os.path.join(tmpdir, 'temp-audio.m4a'),
                remove_temp=True,
                threads=os.cpu_count() or 2
            )

            logging.info("Render completed successfully.")
            return f"Successfully rendered video to '{args.output_filename}' in the output directory."

        except Exception as e:
            logging.error(f"An unexpected error occurred during rendering with MoviePy: {e}", exc_info=True)
            return f"An unexpected error occurred during rendering: {e}"