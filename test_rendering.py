# test_rendering.py

import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Project Imports ---
# This assumes the script is run from the project root.
from codec.state import State, TimelineClip, Keyframe
from codec.utils import probe_media_file
from codec import rendering

# --- Configuration ---
# You can change these to match assets you have available.
# The agent used these files, so they are a good starting point.
ASSET_1_FILENAME = "Cutback at lowers 1 .mp4"
ASSET_2_FILENAME = "cutback at lowers 2.mp4"

# --- Test Case Definitions ---

def setup_test_1_single_clip(state: State, media_info: Dict[str, Any]):
    """A baseline test with a single, full-screen clip on V1."""
    state.add_clip(TimelineClip(
        clip_id="base_clip",
        source_path=media_info[ASSET_1_FILENAME]['path'],
        source_in_sec=0.0,
        source_out_sec=5.0,
        source_total_duration_sec=media_info[ASSET_1_FILENAME]['duration'],
        timeline_start_sec=0.0,
        duration_sec=5.0,
        track_type='video',
        track_number=1,
        description="Baseline full-screen clip",
        **media_info[ASSET_1_FILENAME]['properties']
    ))

def setup_test_2_compositing(state: State, media_info: Dict[str, Any]):
    """Reproduces the agent's core bug report: a scaled clip on V2 over a clip on V1."""
    # V1 Background
    state.add_clip(TimelineClip(
        clip_id="bg",
        source_path=media_info[ASSET_1_FILENAME]['path'],
        source_in_sec=0.0,
        source_out_sec=5.0,
        source_total_duration_sec=media_info[ASSET_1_FILENAME]['duration'],
        timeline_start_sec=0.0,
        duration_sec=5.0,
        track_type='video',
        track_number=1,
        description="Background on V1",
        **media_info[ASSET_1_FILENAME]['properties']
    ))
    # V2 Foreground
    fg_clip = TimelineClip(
        clip_id="fg",
        source_path=media_info[ASSET_2_FILENAME]['path'],
        source_in_sec=0.0,
        source_out_sec=5.0,
        source_total_duration_sec=media_info[ASSET_2_FILENAME]['duration'],
        timeline_start_sec=0.0,
        duration_sec=5.0,
        track_type='video',
        track_number=2,
        description="Foreground on V2, scaled down",
        **media_info[ASSET_2_FILENAME]['properties']
    )
    fg_clip.transformations.append(Keyframe(
        time_sec=0.0,
        position=(0.5, 0.5),
        scale=0.5,
        opacity=80.0
    ))
    state.add_clip(fg_clip)

def setup_test_3_keyframed_motion(state: State, media_info: Dict[str, Any]):
    """Tests keyframed position and scale, which should work smoothly."""
    # V1 Background
    setup_test_2_compositing(state, media_info) # Start with the same base
    # Modify the foreground clip's transformations
    fg_clip = state.find_clip_by_id("fg")
    fg_clip.transformations = [
        Keyframe(time_sec=0.0, position=(0.2, 0.2), scale=0.4, opacity=100.0),
        Keyframe(time_sec=4.5, position=(0.8, 0.8), scale=0.6, opacity=100.0)
    ]

def setup_test_4_keyframed_rotation(state: State, media_info: Dict[str, Any]):
    """Specifically tests rotation, which the user noted was 'glitching'."""
    # V1 Background
    setup_test_2_compositing(state, media_info) # Start with the same base
    # Modify the foreground clip's transformations
    fg_clip = state.find_clip_by_id("fg")
    fg_clip.transformations = [
        Keyframe(time_sec=0.0, position=(0.5, 0.5), scale=0.5, rotation=0.0),
        Keyframe(time_sec=4.5, position=(0.5, 0.5), scale=0.5, rotation=90.0)
    ]

def setup_test_5_full_animation(state: State, media_info: Dict[str, Any]):
    """Combines all transformations to replicate the agent's full test case."""
    # V1 Background
    setup_test_2_compositing(state, media_info) # Start with the same base
    # Modify the foreground clip's transformations
    fg_clip = state.find_clip_by_id("fg")
    fg_clip.transformations = [
        Keyframe(time_sec=0.0, position=(0.8, 0.2), scale=0.5, rotation=0.0, opacity=80.0),
        Keyframe(time_sec=4.5, position=(0.2, 0.8), scale=0.5, rotation=10.0, opacity=80.0)
    ]


# --- Test Runner Logic ---

def run_test_case(
    name: str,
    description: str,
    setup_func,
    media_info: Dict[str, Any],
    output_dir: Path,
    assets_dir: str
) -> Dict[str, str]:
    """
    Runs a single test case: sets up state, renders a preview and video, and returns paths.
    """
    logging.info(f"--- Running Test Case: {name} ---")
    
    # Create a clean state and temporary directory for this test
    state = State(assets_directory=assets_dir)
    tmpdir = output_dir / f"tmp_{name}"
    tmpdir.mkdir(exist_ok=True)

    # Populate the state using the provided setup function
    setup_func(state, media_info)

    # Define output paths
    preview_path = output_dir / f"{name}_preview.png"
    video_path = output_dir / f"{name}_render.mp4"
    xml_path = output_dir / f"{name}_project.mlt" # <-- New path for XML

    result = {
        "name": name,
        "description": description,
        "preview_path": str(preview_path.relative_to(output_dir)),
        "video_path": str(video_path.relative_to(output_dir)),
        "xml_path": str(xml_path.relative_to(output_dir)), # <-- Add XML path to results
        "preview_success": False,
        "render_success": False
    }

    # Generate the MLT XML and save it for debugging
    mlt_xml_content = rendering._state_to_mlt_xml(state)
    with open(xml_path, "w") as f:
        f.write(mlt_xml_content)
    logging.info(f"MLT XML project saved to {xml_path}")

    # Use a temporary file for the melt command
    mlt_project_path = tmpdir / "project.mlt"
    with open(mlt_project_path, "w") as f:
        f.write(mlt_xml_content)

    try:
        # Render a preview frame from the middle of the timeline
        preview_time = state.get_timeline_duration() / 2.0
        logging.info(f"Rendering preview frame at {preview_time:.2f}s...")
        # We can call the internal function directly since we have the XML content
        rendering.render_preview_frame(state, preview_time, str(preview_path), str(tmpdir))
        result["preview_success"] = True
        logging.info(f"Preview frame saved to {preview_path}")
    except Exception as e:
        logging.error(f"Failed to render preview for '{name}': {e}", exc_info=True)

    try:
        # Render the final video
        logging.info("Rendering final video...")
        rendering.render_final_video(state, str(video_path), str(tmpdir))
        result["render_success"] = True
        logging.info(f"Final video saved to {video_path}")
    except Exception as e:
        logging.error(f"Failed to render video for '{name}': {e}", exc_info=True)
    
    # Clean up temp directory for this test
    shutil.rmtree(tmpdir)

    return result


def generate_html_report(results: List[Dict[str, str]], output_dir: Path):
    """Generates a self-contained HTML file to display test results."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Codec Agent Rendering Test Report</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; margin: 0; background-color: #f8f9fa; color: #212529; }
            .container { max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; }
            h1, h2, h3 { color: #343a40; border-bottom: 1px solid #dee2e6; padding-bottom: 10px; }
            h1 { text-align: center; }
            .test-case { margin-bottom: 40px; padding: 20px; border: 1px solid #e9ecef; border-radius: 6px; }
            .description { background-color: #e9f7fd; border-left: 4px solid #17a2b8; padding: 15px; margin: 20px 0; }
            .outputs { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; }
            img, video { max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; }
            .status { font-weight: bold; }
            .rendered { color: #007bff; }
            .failure { color: #dc3545; }
            .debug-link { margin-top: 10px; }
            @media (max-width: 900px) { .outputs { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Codec Agent Rendering Test Report</h1>
    """

    for res in results:
        html_content += f"""
        <div class="test-case">
            <h2>Test: {res['name']}</h2>
            <div class="description">
                <strong>Expected Behavior:</strong>
                <p>{res['description']}</p>
            </div>
            <div class="outputs">
                <div>
                    <h3>Preview Frame (Mid-point)</h3>
        """
        if res['preview_success']:
            html_content += f'<p class="status rendered">RENDERED</p><img src="{res["preview_path"]}" alt="Preview for {res["name"]}" />'
        else:
            html_content += '<p class="status failure">FAILED</p><p>Check logs for error details.</p>'
        
        html_content += """
                </div>
                <div>
                    <h3>Final Video</h3>
        """
        if res['render_success']:
            html_content += f'<p class="status rendered">RENDERED</p><video controls muted loop src="{res["video_path"]}"></video>'
        else:
            html_content += '<p class="status failure">FAILED</p><p>Check logs for error details.</p>'

        html_content += f"""
                </div>
            </div>
            <div class="debug-link">
                <a href="{res['xml_path']}" target="_blank">View Generated MLT XML</a>
            </div>
        </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """
    
    report_path = output_dir / "index.html"
    with open(report_path, "w") as f:
        f.write(html_content)
    logging.info(f"SUCCESS: HTML report generated at {report_path.resolve()}")


def main():
    """Main function to set up and run all rendering tests."""
    project_root = Path(__file__).parent.resolve()
    assets_dir = project_root / "assets"
    output_dir = project_root / "render_tests_output"

    # Clean up previous run
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    # 1. Verify assets exist and get their info
    logging.info("Probing media assets...")
    required_assets = [ASSET_1_FILENAME, ASSET_2_FILENAME]
    media_info = {}
    for filename in required_assets:
        path = assets_dir / filename
        if not path.exists():
            logging.error(f"FATAL: Required asset '{filename}' not found in '{assets_dir}'. Please add it and try again.")
            return
        
        info = probe_media_file(str(path))
        if info.error:
            logging.error(f"FATAL: Could not probe asset '{filename}': {info.error}")
            return
            
        media_info[filename] = {
            'path': str(path),
            'duration': info.duration_sec,
            'properties': {
                'source_frame_rate': info.frame_rate,
                'source_width': info.width,
                'source_height': info.height,
                'has_audio': info.has_audio
            }
        }
    logging.info("All media assets found and probed successfully.")

    # 2. Define all test cases
    test_cases = [
        ("1_single_clip", "A single video clip should play full-screen for 5 seconds.", setup_test_1_single_clip),
        ("2_compositing", "A smaller, semi-transparent clip (V2) should appear centered on top of a full-screen background clip (V1).", setup_test_2_compositing),
        ("3_keyframed_motion", "The V2 clip should smoothly animate from the top-left to the bottom-right, growing slightly larger, composited over the V1 background.", setup_test_3_keyframed_motion),
        ("4_keyframed_rotation", "The V2 clip should rotate 90 degrees clockwise over 5 seconds while staying centered, composited over the V1 background. The rotation should be smooth, not glitchy.", setup_test_4_keyframed_rotation),
        ("5_full_animation", "The V2 clip should animate diagonally from top-right to bottom-left with a slight rotation, composited over the V1 background. This is the agent's full test case.", setup_test_5_full_animation),
    ]

    # 3. Run all tests
    results = []
    for name, description, setup_func in test_cases:
        result = run_test_case(name, description, setup_func, media_info, output_dir, str(assets_dir))
        results.append(result)

    # 4. Generate the final report
    generate_html_report(results, output_dir)
    
    print(f"\nTest suite finished. Please open '{output_dir / 'index.html'}' in your browser to see the results.")


if __name__ == "__main__":
    main()