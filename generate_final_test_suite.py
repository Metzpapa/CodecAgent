# generate_final_test_suite.py

from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
# Change these values to create different test images (e.g., 4K)
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
TARGET_X = 1250  # The exact X coordinate for our target's center
TARGET_Y = 400   # The exact Y coordinate for our target's center

# --- Drawing Constants ---
RULER_SIZE = 50
BG_COLOR = (240, 240, 240) # Light gray
RULER_COLOR = (220, 220, 220)
FONT_COLOR = (0, 0, 0)
TARGET_COLOR = (255, 0, 0) # Red
MAJOR_TICK_COLOR = (0, 0, 0)
MINOR_TICK_COLOR = (150, 150, 150)
TARGET_RADIUS = 20

def get_font(size: int):
    """Safely loads a system font with a fallback."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()

def create_base_canvas():
    """Creates a blank canvas without any rulers or targets."""
    canvas_width = IMG_WIDTH + RULER_SIZE
    canvas_height = IMG_HEIGHT + RULER_SIZE
    canvas = Image.new('RGB', (canvas_width, canvas_height), BG_COLOR)
    return canvas, ImageDraw.Draw(canvas)

def draw_target(draw):
    """Draws the target object at the configured coordinates."""
    # The target is drawn on the main image area, so we must offset by the ruler size
    draw.ellipse(
        (TARGET_X + RULER_SIZE - TARGET_RADIUS, TARGET_Y + RULER_SIZE - TARGET_RADIUS,
         TARGET_X + RULER_SIZE + TARGET_RADIUS, TARGET_Y + RULER_SIZE + TARGET_RADIUS),
        fill=TARGET_COLOR
    )

def draw_pixel_rulers(draw, font):
    """Draws ruler backgrounds and labels with absolute pixel coordinates."""
    canvas_width = IMG_WIDTH + RULER_SIZE
    canvas_height = IMG_HEIGHT + RULER_SIZE
    draw.rectangle([0, 0, canvas_width, RULER_SIZE], fill=RULER_COLOR) # Top
    draw.rectangle([0, 0, RULER_SIZE, canvas_height], fill=RULER_COLOR) # Left

    for x in range(0, IMG_WIDTH + 1, 50):
        is_major = (x % 200 == 0)
        tick_len = 15 if is_major else 8
        draw.line([(x + RULER_SIZE, RULER_SIZE - tick_len), (x + RULER_SIZE, RULER_SIZE)], fill=MINOR_TICK_COLOR)
        if is_major:
            draw.text((x + RULER_SIZE - 10, 10), str(x), fill=FONT_COLOR, font=font)

    for y in range(0, IMG_HEIGHT + 1, 50):
        is_major = (y % 200 == 0)
        tick_len = 15 if is_major else 8
        draw.line([(RULER_SIZE - tick_len, y + RULER_SIZE), (RULER_SIZE, y + RULER_SIZE)], fill=MINOR_TICK_COLOR)
        if is_major:
            draw.text((10, y + RULER_SIZE - 8), str(y), fill=FONT_COLOR, font=font)

def draw_normalized_rulers(draw, font):
    """Draws ruler backgrounds and labels with normalized float coordinates."""
    canvas_width = IMG_WIDTH + RULER_SIZE
    canvas_height = IMG_HEIGHT + RULER_SIZE
    draw.rectangle([0, 0, canvas_width, RULER_SIZE], fill=RULER_COLOR) # Top
    draw.rectangle([0, 0, RULER_SIZE, canvas_height], fill=RULER_COLOR) # Left

    for i in range(11): # 0.0 to 1.0 in 0.1 increments
        norm_val = i / 10.0
        x = norm_val * IMG_WIDTH
        draw.line([(x + RULER_SIZE, RULER_SIZE), (x + RULER_SIZE, RULER_SIZE - 15)], fill=MAJOR_TICK_COLOR)
        draw.text((x + RULER_SIZE - 10, 10), f"{norm_val:.1f}", fill=FONT_COLOR, font=font)
        y = norm_val * IMG_HEIGHT
        draw.line([(RULER_SIZE, y + RULER_SIZE), (RULER_SIZE - 15, y + RULER_SIZE)], fill=MAJOR_TICK_COLOR)
        draw.text((5, y + RULER_SIZE - 8), f"{norm_val:.1f}", fill=FONT_COLOR, font=font)

def generate_all_images():
    """Main function to generate and save all three test images."""
    font = get_font(16)
    
    # --- 1. Generate Pixel-Based Image ---
    canvas_pixel, draw_pixel = create_base_canvas()
    draw_pixel_rulers(draw_pixel, font)
    draw_target(draw_pixel)
    pixel_filename = f"test_image_{IMG_HEIGHT}p_pixel.png"
    canvas_pixel.save(pixel_filename)
    print(f"Successfully created '{pixel_filename}'")

    # --- 2. Generate Normalized-Based Image ---
    canvas_norm, draw_norm = create_base_canvas()
    draw_normalized_rulers(draw_norm, font)
    draw_target(draw_norm)
    norm_filename = f"test_image_{IMG_HEIGHT}p_normalized.png"
    canvas_norm.save(norm_filename)
    print(f"Successfully created '{norm_filename}'")

    # --- 3. Generate No-Aids (Control) Image ---
    # For this one, we only draw the target on a blank canvas
    canvas_no_aids, draw_no_aids = create_base_canvas()
    draw_target(draw_no_aids)
    no_aids_filename = f"test_image_{IMG_HEIGHT}p_no_aids.png"
    canvas_no_aids.save(no_aids_filename)
    print(f"Successfully created '{no_aids_filename}'")

    # --- Print Ground Truth for Verification ---
    print("\n--- Ground Truth for Verification ---")
    print(f"Target is at pixel coordinates: [{TARGET_X}, {TARGET_Y}]")
    print(f"Target is at normalized coordinates: [{TARGET_X/IMG_WIDTH:.5f}, {TARGET_Y/IMG_HEIGHT:.5f}]")

if __name__ == "__main__":
    generate_all_images()