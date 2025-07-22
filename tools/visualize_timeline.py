# codec/tools/visualize_timeline.py

import os
import math
import tempfile
from typing import Optional, TYPE_CHECKING, Union, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

import ffmpeg
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont

from .base import BaseTool
from utils import hms_to_seconds
from llm.types import ContentPart, FileObject

if TYPE_CHECKING:
    from state import State
    from llm.base import LLMConnector


class VisualizeTimelineArgs(BaseModel):
    """Arguments for the visualize_timeline tool."""
    start_time: Optional[str] = Field(
        None,
        description="The timeline timestamp to start the visualization from (e.g., '00:01:00.000'). If omitted, starts from the beginning (0s).",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    end_time: Optional[str] = Field(
        None,
        description="The timeline timestamp to end the visualization at. If omitted, goes to the end of the entire timeline.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )


class VisualizeTimelineTool(BaseTool):
    """
    A tool to generate a single image of the timeline, showing all video and audio tracks.
    """

    @property
    def name(self) -> str:
        return "visualize_timeline"

    @property
    def description(self) -> str:
        return (
            "Generates a single image of the timeline, showing all video and audio tracks. "
            "Clips are displayed with thumbnails and are labeled with numbers that correspond to a legend at the bottom of the image. "
            "Use this tool to get a high-level visual understanding of the edit's structure, check for gaps, or identify specific clips by sight."
        )

    @property
    def args_schema(self):
        return VisualizeTimelineArgs

    def execute(self, state: 'State', args: VisualizeTimelineArgs, connector: 'LLMConnector') -> Union[str, Tuple[str, List[ContentPart]]]:
        if not state.timeline:
            return "Error: The timeline is empty. Cannot visualize an empty timeline."

        tmp_file_path = None
        try:
            visualizer = _TimelineVisualizer(state, args)
            final_image = visualizer.render()

            if not final_image:
                return "Error: Failed to generate timeline visualization. This may be due to an internal error."

            # Create the temporary file that will be cleaned up in the finally block
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                final_image.save(tmp_file, format="JPEG", quality=85)
                tmp_file_path = tmp_file.name

            print(f"Uploading timeline visualization from '{tmp_file_path}'...")
            uploaded_file = connector.upload_file(
                file_path=tmp_file_path,
                display_name="timeline_visualization.jpg",
                mime_type="image/jpeg"
            )
            state.uploaded_files.append(uploaded_file)

            confirmation_text = "Successfully generated a visual representation of the timeline. The following image shows the current edit."
            multimodal_parts = [ContentPart(type='image', file=uploaded_file)]

            return (confirmation_text, multimodal_parts)

        except Exception as e:
            print(f"Error during timeline visualization: {e}")
            import traceback
            traceback.print_exc()
            return f"An unexpected error occurred while generating the timeline visualization: {e}"
        
        finally:
            # --- FIX: Ensure temporary file is always cleaned up, even if upload fails ---
            if tmp_file_path and os.path.exists(tmp_file_path):
                print(f"Cleaning up temporary visualization file: {tmp_file_path}")
                os.unlink(tmp_file_path)


class _TimelineVisualizer:
    """Internal helper class to handle all the logic of rendering the timeline image."""

    # --- CONSTANTS ---
    CANVAS_WIDTH = 1920
    RULER_HEIGHT = 40
    TRACK_HEIGHT = 120
    TRACK_LABEL_WIDTH = 60
    LEGEND_LINE_HEIGHT = 30
    MIN_VIEW_DURATION = 1.0
    MIN_CLIP_WIDTH = 1

    # Colors
    COLOR_BG = (20, 20, 20)
    COLOR_RULER_BG = (30, 30, 30)
    COLOR_TRACK_BG = (40, 40, 40)
    COLOR_TEXT = (220, 220, 220)
    COLOR_VIDEO_CLIP = (80, 80, 120)
    COLOR_AUDIO_CLIP = (80, 120, 80)
    COLOR_ERROR = (220, 40, 40)
    COLOR_LABEL_BG = (255, 255, 255)
    COLOR_LABEL_TEXT = (0, 0, 0)

    def __init__(self, state: 'State', args: VisualizeTimelineArgs):
        self.state = state
        self.args = args
        self.font_sm = self._get_font(14)
        self.font_md = self._get_font(20)
        self.font_lg = self._get_font(24)

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        try:
            font_paths = ["/System/Library/Fonts/Supplemental/Arial.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"]
            font_path = next((p for p in font_paths if os.path.exists(p)), None)
            if font_path:
                return ImageFont.truetype(font_path, size)
        except Exception:
            pass
        print(f"Warning: Arial font not found. Using default font for size {size}.")
        return ImageFont.load_default()

    def render(self) -> Optional[Image.Image]:
        """Main orchestration method to generate the image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._prepare_view_window()
            self._collect_and_prepare_clips()
            self._extract_thumbnails(tmpdir)
            return self._draw_image()

    def _prepare_view_window(self):
        """Calculates the time window and scaling factor, handling short durations."""
        self.view_start_sec = hms_to_seconds(self.args.start_time) if self.args.start_time else 0.0
        self.view_end_sec = hms_to_seconds(self.args.end_time) if self.args.end_time else self.state.get_timeline_duration()

        if self.view_end_sec <= self.view_start_sec:
             self.view_end_sec = self.view_start_sec + self.MIN_VIEW_DURATION

        duration = self.view_end_sec - self.view_start_sec
        if duration < self.MIN_VIEW_DURATION:
            center = self.view_start_sec + duration / 2
            self.view_start_sec = max(0, center - self.MIN_VIEW_DURATION / 2)
            self.view_end_sec = self.view_start_sec + self.MIN_VIEW_DURATION
        
        self.view_duration = self.view_end_sec - self.view_start_sec
        self.render_width = self.CANVAS_WIDTH - self.TRACK_LABEL_WIDTH
        self.pixels_per_second = self.render_width / self.view_duration if self.view_duration > 0 else 0

    def _collect_and_prepare_clips(self):
        """Filters, sorts, and prepares clip data for rendering."""
        visible_clips = []
        for clip in self.state.timeline:
            clip_end_sec = clip.timeline_start_sec + clip.duration_sec
            if max(clip.timeline_start_sec, self.view_start_sec) < min(clip_end_sec, self.view_end_sec):
                visible_clips.append(clip)

        visible_clips.sort(key=lambda c: (c.timeline_start_sec, c.track_type, -c.track_number))

        self.prepared_clips = []
        self.thumbnail_jobs = {}
        self.tracks = defaultdict(list)
        
        for i, clip in enumerate(visible_clips):
            label = i + 1
            
            visible_start = max(clip.timeline_start_sec, self.view_start_sec)
            visible_end = min(clip.timeline_start_sec + clip.duration_sec, self.view_end_sec)
            x_pos = self.TRACK_LABEL_WIDTH + (visible_start - self.view_start_sec) * self.pixels_per_second
            width = (visible_end - visible_start) * self.pixels_per_second
            
            prep_info = {
                "clip": clip, "label": label, "x": x_pos, "width": width,
                "thumbnails": []
            }

            is_image = clip.source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
            if clip.track_type == 'video' and width >= self.MIN_CLIP_WIDTH:
                num_thumbs = max(1, int(width // (self.TRACK_HEIGHT * 1.1)))
                
                offset_into_clip = visible_start - clip.timeline_start_sec
                source_start_for_thumb = clip.source_in_sec + offset_into_clip
                source_duration_for_thumb = visible_end - visible_start

                for j in range(num_thumbs):
                    job_id = f"thumb_{label}_{j}"
                    if is_image:
                        source_time = 0
                    else:
                        segment_dur = source_duration_for_thumb / num_thumbs
                        source_time = source_start_for_thumb + (j * segment_dur) + (segment_dur / 2)
                    
                    self.thumbnail_jobs[job_id] = {"source_path": clip.source_path, "time": source_time, "is_image": is_image}
                    prep_info["thumbnails"].append(job_id)

            self.prepared_clips.append(prep_info)
            self.tracks[(clip.track_type, clip.track_number)].append(prep_info)

    def _extract_thumbnails(self, tmpdir: str):
        """Extracts all required frames in parallel, handling errors."""
        self.thumbnail_results = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_job = {
                executor.submit(self._extract_single_thumb, job_id, job_data, tmpdir): job_id
                for job_id, job_data in self.thumbnail_jobs.items()
            }
            for future in as_completed(future_to_job):
                job_id = future_to_job[future]
                try:
                    self.thumbnail_results[job_id] = future.result()
                except Exception as e:
                    print(f"ERROR: Thumbnail job {job_id} failed with system error: {e}")
                    self.thumbnail_results[job_id] = "error"

    def _extract_single_thumb(self, job_id: str, job_data: dict, tmpdir: str) -> str:
        """Logic for one ffmpeg/Pillow job, wrapped for the executor."""
        source_path = job_data["source_path"]
        if job_data["is_image"]:
            return source_path

        try:
            output_path = os.path.join(tmpdir, f"{job_id}.jpg")
            (
                ffmpeg.input(source_path, ss=job_data["time"])
                .filter('format', pix_fmts='yuvj420p')
                .output(output_path, vframes=1, format='image2', vcodec='mjpeg')
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            return output_path
        except ffmpeg.Error as e:
            print(f"ERROR: ffmpeg failed for job {job_id} on '{source_path}'. Stderr: {e.stderr.decode()}")
            return "error"

    def _draw_image(self) -> Image.Image:
        """Composes the final image from all prepared data."""
        # --- FIX: Sort tracks to place video above audio, matching NLE conventions ---
        sort_order = {'video': 0, 'audio': 1}
        sorted_tracks = sorted(self.tracks.keys(), key=lambda t: (sort_order.get(t[0], 99), -t[1]))
        
        legend_height = self.LEGEND_LINE_HEIGHT * (len(self.prepared_clips) + 1)
        canvas_height = self.RULER_HEIGHT + (len(sorted_tracks) * self.TRACK_HEIGHT) + legend_height
        
        img = Image.new("RGB", (self.CANVAS_WIDTH, int(canvas_height)), self.COLOR_BG)
        draw = ImageDraw.Draw(img, "RGBA")

        self._draw_ruler(draw)
        
        y_offset = self.RULER_HEIGHT
        track_map = {}
        for track_type, track_number in sorted_tracks:
            track_map[(track_type, track_number)] = y_offset
            self._draw_track_lane(draw, y_offset, f"{track_type[0].upper()}{track_number}")
            y_offset += self.TRACK_HEIGHT

        for prep_info in self.prepared_clips:
            clip = prep_info["clip"]
            y_pos = track_map[(clip.track_type, clip.track_number)]
            self._draw_clip(img, draw, prep_info, y_pos)

        self._draw_legend(draw, y_offset)
        return img

    def _draw_ruler(self, draw: ImageDraw.Draw):
        draw.rectangle([0, 0, self.CANVAS_WIDTH, self.RULER_HEIGHT], fill=self.COLOR_RULER_BG)
        num_ticks = 10
        for i in range(num_ticks + 1):
            x = self.TRACK_LABEL_WIDTH + (i / num_ticks) * self.render_width
            time_sec = self.view_start_sec + (i / num_ticks) * self.view_duration
            time_str = f"{int(time_sec // 60):02d}:{time_sec % 60:04.1f}"
            draw.line([x, self.RULER_HEIGHT - 10, x, self.RULER_HEIGHT], fill=self.COLOR_TEXT)
            if i < num_ticks:
                draw.text((x + 3, self.RULER_HEIGHT - 25), time_str, font=self.font_sm, fill=self.COLOR_TEXT)

    def _draw_track_lane(self, draw: ImageDraw.Draw, y_pos: int, label: str):
        draw.rectangle([0, y_pos, self.CANVAS_WIDTH, y_pos + self.TRACK_HEIGHT], fill=self.COLOR_TRACK_BG, outline=(50,50,50))
        draw.text((10, y_pos + self.TRACK_HEIGHT/2 - 10), label, font=self.font_lg, fill=self.COLOR_TEXT)

    def _draw_clip(self, img: Image.Image, draw: ImageDraw.Draw, prep_info: dict, y_pos: int):
        x, width = prep_info["x"], prep_info["width"]
        if width < 1: return
        
        clip = prep_info["clip"]
        
        base_color = self.COLOR_VIDEO_CLIP if clip.track_type == 'video' else self.COLOR_AUDIO_CLIP
        draw.rectangle([x, y_pos, x + width, y_pos + self.TRACK_HEIGHT], fill=base_color)

        if clip.track_type == 'video':
            thumb_width = width / len(prep_info["thumbnails"]) if prep_info["thumbnails"] else 0
            for i, job_id in enumerate(prep_info["thumbnails"]):
                result_path = self.thumbnail_results.get(job_id)
                thumb_x = x + i * thumb_width
                if result_path == "error":
                    draw.line([thumb_x + 5, y_pos + 5, thumb_x + thumb_width - 5, y_pos + self.TRACK_HEIGHT - 5], fill=self.COLOR_ERROR, width=3)
                    draw.line([thumb_x + 5, y_pos + self.TRACK_HEIGHT - 5, thumb_x + thumb_width - 5, y_pos + 5], fill=self.COLOR_ERROR, width=3)
                elif result_path:
                    try:
                        thumb_img = Image.open(result_path)
                        thumb_img = self._letterbox(thumb_img, (int(thumb_width), self.TRACK_HEIGHT))
                        img.paste(thumb_img, (int(thumb_x), y_pos))
                    except Exception as e:
                         print(f"ERROR: Failed to paste thumbnail for job {job_id}: {e}")
                         draw.line([thumb_x + 5, y_pos + 5, thumb_x + thumb_width - 5, y_pos + self.TRACK_HEIGHT - 5], fill=self.COLOR_ERROR, width=3)
        else:
            draw.text((x + width/2 - 25, y_pos + self.TRACK_HEIGHT/2 - 10), "AUDIO", font=self.font_md, fill=self.COLOR_TEXT)

        label_text = str(prep_info["label"])
        radius = 12
        center_x, center_y = x + width / 2, y_pos + self.TRACK_HEIGHT / 2
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], fill=self.COLOR_LABEL_BG)
        text_bbox = draw.textbbox((0,0), label_text, font=self.font_md)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.text((center_x - text_w/2, center_y - text_h/2), label_text, font=self.font_md, fill=self.COLOR_LABEL_TEXT)

    def _draw_legend(self, draw: ImageDraw.Draw, y_start: int):
        draw.text((10, y_start + 5), "Legend:", font=self.font_lg, fill=self.COLOR_TEXT)
        y_offset = y_start + self.LEGEND_LINE_HEIGHT + 5
        
        for prep_info in self.prepared_clips:
            label_text = str(prep_info["label"])
            radius = 10
            draw.ellipse([20 - radius, y_offset - radius, 20 + radius, y_offset + radius], fill=self.COLOR_LABEL_BG)
            text_bbox = draw.textbbox((0,0), label_text, font=self.font_sm)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.text((20 - text_w/2, y_offset - text_h/2), label_text, font=self.font_sm, fill=self.COLOR_LABEL_TEXT)
            
            clip = prep_info["clip"]
            legend_str = f"{clip.clip_id} (Source: {os.path.basename(clip.source_path)})"
            draw.text((45, y_offset - 8), legend_str, font=self.font_md, fill=self.COLOR_TEXT)
            y_offset += self.LEGEND_LINE_HEIGHT

    def _letterbox(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resizes an image to fit within target_size, preserving aspect ratio with black bars."""
        target_w, target_h = target_size
        if target_w <= 0 or target_h <= 0: return Image.new("RGB", (1,1), self.COLOR_BG)

        img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
        
        new_img = Image.new("RGB", target_size, self.COLOR_BG)
        paste_x = (target_w - img.width) // 2
        paste_y = (target_h - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img