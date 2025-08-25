# codecagent/codec/rendering.py

import os
import subprocess
import logging
import platform
from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from .state import State, TimelineClip

# --- Public API ---

def render_final_video(state: 'State', output_path: str, tmpdir: str) -> None:
    """
    Renders the complete timeline from the state object to a final video file.

    This function generates an MLT XML project in memory, writes it to a temporary
    file, and then invokes the `melt` command-line tool to perform the render,
    applying hardware acceleration where available.

    Args:
        state: The current agent state containing the timeline.
        output_path: The absolute path for the final rendered video file.
        tmpdir: A temporary directory for intermediate files like the MLT project.
    """
    logging.info("Starting final render process using MLT...")
    
    try:
        mlt_xml_content = _state_to_mlt_xml(state)
        mlt_project_path = os.path.join(tmpdir, "project.mlt")
        with open(mlt_project_path, "w") as f:
            f.write(mlt_xml_content)
        
        logging.debug(f"--- MLT XML Project ---\n{mlt_xml_content}\n-----------------------")

        fps, _, _ = state.get_sequence_properties()

        # Base consumer properties for a high-quality MP4
        consumer_args = [
            f"avformat:{output_path}",
            "acodec=aac",
            "pix_fmt=yuv420p"
        ]

        # Add hardware acceleration options based on platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            logging.info("Apple Silicon detected. Using 'h264_videotoolbox' hardware encoder.")
            consumer_args.append("vcodec=h264_videotoolbox")
        else:
            logging.info("Using 'libx264' software encoder with 'ultrafast' preset.")
            consumer_args.extend([
                "vcodec=libx264",
                "preset=ultrafast",
                f"threads={os.cpu_count() or 2}"
            ])

        command = ["melt", mlt_project_path, "-consumer"] + consumer_args
        
        logging.info(f"Executing melt command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Successfully rendered final video to {output_path}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Melt execution failed with return code {e.returncode}")
        logging.error(f"Melt stderr:\n{e.stderr}")
        raise RuntimeError(f"MLT rendering failed. See logs for details. Stderr: {e.stderr}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during final render: {e}", exc_info=True)
        raise


def render_preview_frame(state: 'State', timeline_sec: float, output_path: str, tmpdir: str) -> None:
    """
    Renders a single, fully composited frame from the timeline at a specific time.

    This function uses the exact same MLT XML generation logic as the final render,
    ensuring that the preview frame is a perfect representation of the final output.
    It instructs `melt` to render only the single requested frame as a JPEG image.

    Args:
        state: The current agent state containing the timeline.
        timeline_sec: The time in seconds on the main timeline to render.
        output_path: The absolute path where the output JPEG image will be saved.
        tmpdir: A temporary directory for the MLT project file.
    """
    logging.info(f"Rendering preview frame at {timeline_sec:.2f}s using MLT...")

    try:
        mlt_xml_content = _state_to_mlt_xml(state)
        mlt_project_path = os.path.join(tmpdir, f"preview_{timeline_sec:.3f}.mlt")
        with open(mlt_project_path, "w") as f:
            f.write(mlt_xml_content)

        fps, _, _ = state.get_sequence_properties()
        frame_num = int(round(timeline_sec * fps))

        # The 'out' property for melt is inclusive. Setting in=out renders one frame.
        command = [
            "melt",
            mlt_project_path,
            f"in={frame_num}",
            f"out={frame_num}",
            "-consumer",
            f"avformat:{output_path}",
            "vcodec=mjpeg"
        ]

        logging.info(f"Executing melt command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Successfully rendered preview frame to {output_path}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Melt preview frame execution failed with return code {e.returncode}")
        logging.error(f"Melt stderr:\n{e.stderr}")
        raise RuntimeError(f"MLT preview rendering failed. Stderr: {e.stderr}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during preview render: {e}", exc_info=True)
        raise


# --- Private MLT XML Generation Logic ---

def _state_to_mlt_xml(state: 'State') -> str:
    """
    Translates the agent's State object into a complete MLT XML project string.
    This is the core of the unified rendering engine.
    """
    fps, width, height = state.get_sequence_properties()
    
    # Use a rational number for frame rate for precision, as MLT prefers it.
    # E.g., 23.976 -> 24000/1001
    fr_num, fr_den = (int(fps * 1001), 1001) if fps % 1 != 0 else (int(fps), 1)

    xml_parts = [
        '<mlt>',
        f'  <profile name="codec-agent-profile" width="{width}" height="{height}" frame_rate_num="{fr_num}" frame_rate_den="{fr_den}" sample_aspect_num="1" sample_aspect_den="1" display_aspect_num="{width}" display_aspect_den="{height}" colorspace="709"/>'
    ]

    # 1. Define producers for each unique media source file
    unique_sources = sorted(list({clip.source_path for clip in state.timeline}))
    source_to_producer_id = {source: f"producer_{i}" for i, source in enumerate(unique_sources)}
    for source, pid in source_to_producer_id.items():
        xml_parts.append(f'  <producer id="{pid}">')
        xml_parts.append(f'    <property name="resource">{os.path.abspath(source)}</property>')
        xml_parts.append(f'  </producer>')

    # 2. Define a playlist for each track (V1, V2, A1, etc.)
    tracks = sorted(
        list({(c.track_type, c.track_number) for c in state.timeline}),
        key=lambda x: (x[0] != 'video', x[1])  # Videos first, then audio, sorted by number
    )
    track_to_playlist_id = {}
    track_zero_based_index = {}
    
    for i, (track_type, track_number) in enumerate(tracks):
        playlist_id = f"playlist_{track_type}{track_number}"
        track_to_playlist_id[(track_type, track_number)] = playlist_id
        track_zero_based_index[(track_type, track_number)] = i
        
        xml_parts.append(f'  <playlist id="{playlist_id}">')
        
        clips_on_track = state.get_clips_on_specific_track(track_type, track_number)
        last_clip_end_frames = 0
        
        for clip in clips_on_track:
            start_frames = int(round(clip.timeline_start_sec * fps))
            
            # Insert blank for any gap between clips
            gap_frames = start_frames - last_clip_end_frames
            if gap_frames > 0:
                xml_parts.append(f'    <blank length="{gap_frames}"/>')

            producer_id = source_to_producer_id[clip.source_path]
            in_frames = int(round(clip.source_in_sec * fps))
            duration_frames = int(round(clip.duration_sec * fps))
            
            # For playlists, we specify the source producer, the start point in the source (in),
            # and how long to play it for (length).
            xml_parts.append(f'    <entry producer="{producer_id}" in="{in_frames}" length="{duration_frames}"/>')
            
            last_clip_end_frames = start_frames + duration_frames
            
        xml_parts.append('  </playlist>')

    # 3. Define the main tractor and multitrack to layer the playlists
    xml_parts.append('  <tractor id="main_tractor">')
    xml_parts.append('    <multitrack>')
    # In MLT, tracks are layered like in GIMP. The last track in the list is the "top" layer.
    # We sort our tracks V1, V2, ..., A1, A2, ... so this ordering is correct.
    for track_type, track_number in tracks:
        playlist_id = track_to_playlist_id[(track_type, track_number)]
        xml_parts.append(f'      <track producer="{playlist_id}"/>')
    xml_parts.append('    </multitrack>')

    # 4. Add affine filters to the tractor for each clip with transformations
    for clip in state.timeline:
        if not clip.transformations or clip.track_type != 'video':
            continue

        track_key = (clip.track_type, clip.track_number)
        track_index = track_zero_based_index.get(track_key)
        if track_index is None: continue

        start_frames = int(round(clip.timeline_start_sec * fps))
        end_frames = start_frames + int(round(clip.duration_sec * fps)) - 1

        master_kfs = _get_master_keyframes(clip, width, height)
        if not master_kfs: continue

        rect_kfs_str = _build_rect_kfs_string(master_kfs, clip, fps)
        rot_kfs_str = _build_generic_kfs_string(master_kfs, 'rotation', fps)

        # A filter is applied to the tractor but constrained to a time range (in/out)
        # and a specific track, effectively applying it to a single clip.
        xml_parts.append(f'    <filter in="{start_frames}" out="{end_frames}">')
        xml_parts.append(f'      <property name="mlt_service">affine</property>')
        xml_parts.append(f'      <property name="track">{track_index}</property>')
        if rect_kfs_str:
            xml_parts.append(f'      <property name="transition.rect">{rect_kfs_str}</property>')
        if rot_kfs_str:
            xml_parts.append(f'      <property name="transition.fix_rotate_z">{rot_kfs_str}</property>')
        
        # This property is crucial for the affine filter to correctly handle the alpha channel
        # of the source clip, allowing for opacity animations.
        xml_parts.append(f'      <property name="transition.b_alpha">1</property>')
        xml_parts.append('    </filter>')

    xml_parts.append('  </tractor>')
    xml_parts.append('</mlt>')
    
    return "\n".join(xml_parts)


# --- Keyframe Generation Helpers ---

def _get_master_keyframes(clip: 'TimelineClip', seq_width: int, seq_height: int) -> List[Dict[str, Any]]:
    """
    Creates a unified list of keyframes, ensuring that every keyframe object
    has a value for every transformable property, carrying forward previous values
    if a property isn't specified at a particular time. This is essential for
    generating valid MLT keyframe strings.
    """
    all_times = sorted(list({kf.time_sec for kf in clip.transformations}))
    if not all_times: return []

    master_kfs = []
    last_props = {
        "position": (seq_width / 2, seq_height / 2),
        "scale": 1.0,
        "rotation": 0.0,
        "opacity": 100.0,
        "anchor_point": (clip.source_width / 2, clip.source_height / 2),
        "interpolation": "easy ease"
    }

    for t in all_times:
        current_kf_props = last_props.copy()
        # Find all keyframes at this exact time and update props
        for kf in clip.transformations:
            if kf.time_sec == t:
                if kf.position is not None: current_kf_props['position'] = kf.position
                if kf.scale is not None: current_kf_props['scale'] = kf.scale
                if kf.rotation is not None: current_kf_props['rotation'] = kf.rotation
                if kf.opacity is not None: current_kf_props['opacity'] = kf.opacity
                if kf.anchor_point is not None: current_kf_props['anchor_point'] = kf.anchor_point
                current_kf_props['interpolation'] = kf.interpolation
        
        current_kf_props['time_sec'] = t
        master_kfs.append(current_kf_props)
        last_props = current_kf_props
    
    return master_kfs


def _build_rect_kfs_string(master_kfs: List[Dict[str, Any]], clip: 'TimelineClip', fps: float) -> str:
    """
    Builds the complex keyframe string for the MLT affine filter's 'rect' property.
    Format: [frame=X/Y:WxH:Opacity:interp; frame2=...]
    """
    kf_strings = []
    for kf in master_kfs:
        frame = int(round(kf['time_sec'] * fps))
        
        scale = kf['scale']
        pos_x, pos_y = kf['position']
        anchor_x, anchor_y = kf['anchor_point']
        opacity = kf['opacity']
        
        # Calculate the scaled width and height of the clip
        w = clip.source_width * scale
        h = clip.source_height * scale
        
        # The 'position' property defines where the 'anchor_point' of the clip should be on the canvas.
        # The MLT 'rect' property's X/Y, however, defines the top-left corner of the transformed clip.
        # We calculate the top-left corner (x, y) based on the desired position and anchor.
        # x = position_x - (anchor_x * scale)
        # y = position_y - (anchor_y * scale)
        x = pos_x - (anchor_x * scale)
        y = pos_y - (anchor_y * scale)
        
        interp_map = {"easy ease": "smooth", "linear": "linear", "hold": "discrete"}
        interp = interp_map.get(kf['interpolation'], "smooth")
        
        kf_strings.append(f"{frame}={x:.3f}/{y:.3f}:{w:.3f}x{h:.3f}:{opacity:.2f}:{interp}")

    return f"[{';'.join(kf_strings)}]"


def _build_generic_kfs_string(master_kfs: List[Dict[str, Any]], prop_name: str, fps: float) -> str:
    """
    Builds a generic keyframe string for simple properties like rotation.
    Format: [frame=value:interp; frame2=value2:interp2;...]
    """
    kf_strings = []
    for kf in master_kfs:
        frame = int(round(kf['time_sec'] * fps))
        value = kf[prop_name]
        
        interp_map = {"easy ease": "smooth", "linear": "linear", "hold": "discrete"}
        interp = interp_map.get(kf['interpolation'], "smooth")
        
        kf_strings.append(f"{frame}={value}:{interp}")
        
    return f"[{';'.join(kf_strings)}]"