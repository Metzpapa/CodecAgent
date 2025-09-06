# codecagent/codec/visuals.py

import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from .state import State, TimelineClip, Keyframe

# --- Constants for Visual Styling ---
PADDING = 40
HEADER_HEIGHT = 50
FONT_SIZE_LARGE = 24
FONT_SIZE_SMALL = 14

COLOR_BACKGROUND = "black"
COLOR_TEXT_HEADER = "white"
COLOR_TEXT_LABEL = "#CCCCCC"  # Light gray for grid labels
COLOR_GRID_MAJOR = "#555555"  # Dark gray for major grid lines
COLOR_GRID_MINOR = "#333333"  # Darker gray for minor grid lines
COLOR_ANCHOR = "#FF00FF"  # Bright magenta for the anchor point


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Attempts to load a preferred font, falling back to the default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        logging.debug("Arial font not found. Using default PIL font.")
        if size < 16:
             try:
                 return ImageFont.load_default(size)
             except AttributeError:
                 return ImageFont.load_default()
        return ImageFont.load_default()


def _get_interpolated_position(clip: 'TimelineClip', relative_time_sec: float) -> Tuple[float, float]:
    """
    Calculates the interpolated (x, y) position of a clip at a specific time.
    This is a simplified interpolation logic specifically for visualization.
    """
    default_pos = (0.5, 0.5)
    pos_kfs = sorted(
        [kf for kf in clip.transformations if kf.position is not None],
        key=lambda kf: kf.time_sec
    )
    if not pos_kfs:
        return default_pos

    before_kf = None
    after_kf = None
    for kf in pos_kfs:
        if kf.time_sec <= relative_time_sec:
            before_kf = kf
        if kf.time_sec > relative_time_sec:
            after_kf = kf
            break

    if not before_kf: return pos_kfs[0].position or default_pos
    if not after_kf: return before_kf.position or default_pos
    if abs(before_kf.time_sec - relative_time_sec) < 0.001: return before_kf.position or default_pos

    time_diff = after_kf.time_sec - before_kf.time_sec
    if time_diff < 0.001: return before_kf.position or default_pos

    progress = (relative_time_sec - before_kf.time_sec) / time_diff
    start_x, start_y = before_kf.position or default_pos
    end_x, end_y = after_kf.position or default_pos
    interp_x = start_x + (end_x - start_x) * progress
    interp_y = start_y + (end_y - start_y) * progress
    return (interp_x, interp_y)


def draw_coordinate_grid(draw: ImageDraw.ImageDraw, width: int, height: int):
    """
    Draws a normalized coordinate grid with labels onto the provided image.
    """
    font = _get_font(FONT_SIZE_SMALL)
    for i in range(1, 10):
        x = PADDING + (width * i / 10)
        y = PADDING + (height * i / 10)
        draw.line([(x, PADDING), (x, PADDING + height)], fill=COLOR_GRID_MINOR, width=1)
        draw.line([(PADDING, y), (PADDING + width, y)], fill=COLOR_GRID_MINOR, width=1)

    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for tick in ticks:
        x = PADDING + (width * tick)
        y = PADDING + (height * tick)
        label = f"{tick:.2f}"
        draw.line([(x, PADDING), (x, PADDING + height)], fill=COLOR_GRID_MAJOR, width=1)
        draw.line([(PADDING, y), (PADDING + width, y)], fill=COLOR_GRID_MAJOR, width=1)
        draw.text((x, PADDING - 15), label, fill=COLOR_TEXT_LABEL, font=font, anchor="mb")
        draw.text((PADDING - 10, y), label, fill=COLOR_TEXT_LABEL, font=font, anchor="rm")


def draw_anchor_point(draw: ImageDraw.ImageDraw, state: 'State', clip: 'TimelineClip', timeline_sec: float):
    """
    Draws a marker for the clip's anchor point at its transformed position on the timeline.
    """
    _, seq_width, seq_height = state.get_sequence_properties()
    relative_time_sec = timeline_sec - clip.timeline_start_sec
    pos_x_norm, pos_y_norm = _get_interpolated_position(clip, relative_time_sec)
    abs_x = PADDING + (pos_x_norm * seq_width)
    abs_y = PADDING + (pos_y_norm * seq_height)
    
    marker_size = 10
    draw.line([(abs_x - marker_size, abs_y), (abs_x + marker_size, abs_y)], fill=COLOR_ANCHOR, width=2)
    draw.line([(abs_x, abs_y - marker_size), (abs_x, abs_y + marker_size)], fill=COLOR_ANCHOR, width=2)


def draw_default_anchor_point(draw: ImageDraw.ImageDraw, width: int, height: int):
    """
    Draws a marker at the default center (0.5, 0.5) of a frame, for use in non-timeline contexts.
    """
    abs_x = PADDING + (width / 2)
    abs_y = PADDING + (height / 2)
    
    marker_size = 10
    draw.line([(abs_x - marker_size, abs_y), (abs_x + marker_size, abs_y)], fill=COLOR_ANCHOR, width=2)
    draw.line([(abs_x, abs_y - marker_size), (abs_x, abs_y + marker_size)], fill=COLOR_ANCHOR, width=2)


def apply_overlays(
    image: Image.Image,
    overlays: List[str],
    state: 'State',
    clip: Optional['TimelineClip'] = None,
    timeline_sec: Optional[float] = None
) -> Image.Image:
    """
    Applies a list of specified visual overlays to a given image.
    This function creates a new image with padding to accommodate the overlays.
    """
    if not overlays:
        return image

    padded_width = image.width + PADDING * 2
    padded_height = image.height + PADDING * 2
    padded_image = Image.new("RGB", (padded_width, padded_height), COLOR_BACKGROUND)
    padded_image.paste(image, (PADDING, PADDING))
    draw = ImageDraw.Draw(padded_image)

    if "coordinate_grid" in overlays:
        draw_coordinate_grid(draw, image.width, image.height)

    if "anchor_point" in overlays:
        # If we have a clip and timeline context, draw its transformed position
        if clip and timeline_sec is not None:
            draw_anchor_point(draw, state, clip, timeline_sec)
        # Otherwise (e.g., in view_video), draw the default center anchor
        else:
            draw_default_anchor_point(draw, image.width, image.height)

    return padded_image


def compose_side_by_side(
    image_left: Image.Image,
    label_left: str,
    image_right: Image.Image,
    label_right: str
) -> Image.Image:
    """
    Creates a single composite image by placing two images side-by-side with labels.
    """
    if image_left.size != image_right.size:
        image_right = image_right.resize(image_left.size, Image.Resampling.LANCZOS)
    
    width, height = image_left.size
    total_width = (width * 2) + (PADDING * 3)
    total_height = height + HEADER_HEIGHT + PADDING

    composite_img = Image.new('RGB', (total_width, total_height), COLOR_BACKGROUND)
    draw = ImageDraw.Draw(composite_img)
    font = _get_font(FONT_SIZE_LARGE)

    composite_img.paste(image_left, (PADDING, HEADER_HEIGHT))
    composite_img.paste(image_right, (width + PADDING * 2, HEADER_HEIGHT))

    draw.text((PADDING, PADDING), label_left, fill=COLOR_TEXT_HEADER, font=font)
    draw.text((width + PADDING * 2, PADDING), label_right, fill=COLOR_TEXT_HEADER, font=font)

    return composite_img