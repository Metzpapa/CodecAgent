# codecagent/codec/tools/render_video.py

import os
import logging
import math
from typing import TYPE_CHECKING, List, Callable, Any
from pathlib import Path

# --- MoviePy v2 Imports ---
from moviepy import (
    VideoFileClip, AudioFileClip, CompositeVideoClip,
    CompositeAudioClip, ColorClip
)
import numpy as np
from pydantic import BaseModel, Field
import openai

from .base import BaseTool
from ..state import Keyframe

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
    Returns a function `f(t)` where `t` is the time relative to the clip's start.
    """
    prop_kfs = sorted(
        [kf for kf in keyframes if getattr(kf, property_name) is not None],
        key=lambda kf: kf.time_sec
    )

    def interpolator(t: float) -> Any:
        if not prop_kfs:
            return default_value
        if t <= prop_kfs[0].time_sec:
            return getattr(prop_kfs[0], property_name)
        if t >= prop_kfs[-1].time_sec:
            return getattr(prop_kfs[-1], property_name)
        for i in range(len(prop_kfs) - 1):
            kf_prev, kf_next = prop_kfs[i], prop_kfs[i + 1]
            if kf_prev.time_sec <= t < kf_next.time_sec:
                if kf_prev.interpolation == "hold":
                    return getattr(kf_prev, property_name)
                time_diff = kf_next.time_sec - kf_prev.time_sec
                if time_diff == 0: return getattr(kf_prev, property_name)
                progress = (t - kf_prev.time_sec) / time_diff
                if kf_prev.interpolation == "easy ease":
                    progress = progress * progress * (3 - 2 * progress)
                val_prev = np.array(getattr(kf_prev, property_name))
                val_next = np.array(getattr(kf_next, property_name))
                interpolated_val = val_prev + progress * (val_next - val_prev)
                return tuple(interpolated_val) if isinstance(getattr(kf_prev, property_name), tuple) else interpolated_val.item()
        return default_value
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

        job_dir = Path(state.assets_directory).parent
        output_path = job_dir / "output" / args.output_filename
        output_path.parent.mkdir(exist_ok=True)

        duration = state.get_timeline_duration()
        fps, width, height = state.get_sequence_properties()

        processed_video_clips, processed_audio_clips = [], []

        video_timeline_clips = sorted(
            [c for c in state.timeline if c.track_type == 'video'],
            key=lambda c: c.track_number
        )

        for clip_data in video_timeline_clips:
            try:
                base_clip = VideoFileClip(clip_data.source_path).subclipped(
                    clip_data.source_in_sec, clip_data.source_out_sec
                )

                if not clip_data.transformations:
                    final_clip = (base_clip
                        .with_start(clip_data.timeline_start_sec)
                        .with_duration(clip_data.duration_sec)
                        .with_position(('center', 'center')))
                    processed_video_clips.append(final_clip)
                    continue

                scale_func = _create_property_interpolator(clip_data.transformations, 'scale', 1.0)
                rot_func = _create_property_interpolator(clip_data.transformations, 'rotation', 0.0)
                opacity_func = _create_property_interpolator(clip_data.transformations, 'opacity', 100.0)
                pos_func = _create_property_interpolator(clip_data.transformations, 'position', (width / 2, height / 2))
                anchor_func = _create_property_interpolator(clip_data.transformations, 'anchor_point', (base_clip.size[0] / 2, base_clip.size[1] / 2))

                scaled_clip = base_clip.resized(lambda t: scale_func(t))
                
                max_scale = max([kf.scale for kf in clip_data.transformations if kf.scale is not None] or [1.0])
                hypot = math.hypot(base_clip.size[0] * max_scale, base_clip.size[1] * max_scale)
                canvas_size = (int(math.ceil(hypot)), int(math.ceil(hypot)))
                canvas_center = (canvas_size[0] / 2, canvas_size[1] / 2)
                
                inner_pos_func = lambda t: (canvas_center[0] - (anchor_func(t)[0] * scale_func(t)),
                                            canvas_center[1] - (anchor_func(t)[1] * scale_func(t)))
                inner_clip = scaled_clip.with_position(inner_pos_func)
                
                canvas = CompositeVideoClip([inner_clip], size=canvas_size, bg_color=None)
                rotated_canvas = canvas.rotated(lambda t: rot_func(t), center=canvas_center)

                final_pos_func = lambda t: (pos_func(t)[0] - canvas_center[0], pos_func(t)[1] - canvas_center[1])
                final_animated_clip = rotated_canvas.with_position(final_pos_func)

                # --- FIX: Manually transform the mask for animated opacity ---
                # The .with_opacity() method doesn't properly handle time-varying functions.
                # The correct way is to create a mask and apply our time function to it directly.
                clip_with_opacity = final_animated_clip.with_mask()
                clip_with_opacity.mask = clip_with_opacity.mask.transform(
                    lambda gf, t: gf(t) * (opacity_func(t) / 100.0)
                )

                final_clip = (clip_with_opacity
                    .with_start(clip_data.timeline_start_sec)
                    .with_duration(clip_data.duration_sec))

                processed_video_clips.append(final_clip)

            except Exception as e:
                logging.error(f"Failed to process video clip '{clip_data.clip_id}': {e}", exc_info=True)
                return f"Error: Failed to process video clip '{clip_data.clip_id}'. Details: {e}"

        audio_timeline_clips = [c for c in state.timeline if c.track_type == 'audio' and c.has_audio]
        for clip_data in audio_timeline_clips:
            audio_clip = (AudioFileClip(clip_data.source_path)
                          .subclipped(clip_data.source_in_sec, clip_data.source_out_sec)
                          .with_start(clip_data.timeline_start_sec)
                          .with_duration(clip_data.duration_sec))
            processed_audio_clips.append(audio_clip)

        try:
            logging.info("Compositing final video...")
            canvas = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
            final_video = CompositeVideoClip([canvas] + processed_video_clips, size=(width, height))
            
            if processed_audio_clips:
                final_audio = CompositeAudioClip(processed_audio_clips)
                final_video = final_video.with_audio(final_audio)

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