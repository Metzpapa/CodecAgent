# codec/tools/export_timeline.py

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional
from collections import defaultdict
import datetime

import opentimelineio as otio
from pydantic import BaseModel, Field

from google import genai
from .base import BaseTool
from state import TimelineClip

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State


class ExportTimelineArgs(BaseModel):
    """Arguments for the export_timeline tool."""
    output_filename: str = Field(
        "codec_edit.otio",
        description="The desired filename for the exported timeline file. The extension determines the format: '.otio' (recommended) or '.xml' (legacy)."
    )
    consolidate: bool = Field(
        True,
        description="If True, creates a self-contained project folder in Downloads with the timeline file and copies of all used media. This is highly recommended for sharing and portability. If False, only the timeline file is created."
    )


class ExportTimelineTool(BaseTool):
    """
    A tool to export the current editing timeline to a standard interchange file.
    Can create a self-contained, portable project package by consolidating all media.
    Use this when the user wants to see, finalize, or share the edit.
    """

    @property
    def name(self) -> str:
        return "export_timeline"

    @property
    def description(self) -> str:
        return "Exports the current timeline and all its media into a self-contained, portable project folder. Use this when the user wants to see, finalize, or share the edit."

    @property
    def args_schema(self):
        return ExportTimelineArgs

    def execute(self, state: 'State', args: ExportTimelineArgs, client: 'genai.Client') -> str:
        if not state.timeline:
            return "Error: The timeline is empty. Please add some clips before exporting."

        try:
            downloads_path = Path.home() / "Downloads"
            if not downloads_path.is_dir():
                raise IOError(f"Could not find the Downloads folder at '{downloads_path}'.")

            if args.consolidate:
                # --- CONSOLIDATION LOGIC ---
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                package_name = f"{Path(args.output_filename).stem}_{timestamp}"
                package_dir = downloads_path / package_name
                media_dir = package_dir / "media"
                
                print(f"Consolidating project into: {package_dir}")
                os.makedirs(media_dir, exist_ok=True)

                # Copy unique source files to the new media directory
                unique_source_paths = {clip.source_path for clip in state.timeline}
                for src_path_str in unique_source_paths:
                    src_path = Path(src_path_str)
                    dest_path = media_dir / src_path.name
                    print(f"  - Copying {src_path.name}...")
                    shutil.copy2(src_path, dest_path)

                # The base for relative paths is the package directory itself
                base_path_for_relinking = package_dir
                output_path = package_dir / args.output_filename
                success_message = f"✅ Successfully consolidated project! Find the complete folder at: {package_dir}"

            else:
                # --- LEGACY (NON-CONSOLIDATED) LOGIC ---
                output_path = downloads_path / args.output_filename
                # The base for relative paths is the Downloads directory
                base_path_for_relinking = downloads_path
                success_message = f"✅ Successfully exported timeline file to: {output_path}"


            # --- COMMON BUILD & WRITE LOGIC ---
            fps, width, height = self._get_or_infer_sequence_properties(state)
            otio_timeline = self._build_otio_timeline(state, fps, width, height, base_path_for_relinking)

            file_ext = output_path.suffix.lower()
            if file_ext == ".otio":
                adapter_name = "otio_json"
            elif file_ext == ".xml":
                adapter_name = "fcp_xml"
            else:
                return f"Error: Unsupported file extension '{file_ext}'. Please use '.otio' or '.xml'."

            otio.adapters.write_to_file(otio_timeline, str(output_path), adapter_name=adapter_name)

        except (ValueError, IOError, Exception) as e:
            return f"Error during export: {e}"

        return success_message

    def _get_or_infer_sequence_properties(self, state: 'State') -> Tuple[float, int, int]:
        # ... (This function remains unchanged)
        if all([state.frame_rate, state.width, state.height]):
            return (state.frame_rate, state.width, state.height)
        first_video_clip = next((c for c in state.timeline if c.source_path.lower().endswith(('.mp4', '.mov', '.mkv', '.mxf'))), None)
        if not first_video_clip:
            raise ValueError("Could not determine timeline properties (frame rate, resolution). No video clips found on timeline to infer from.")
        print(f"✅ Inferred sequence properties from '{os.path.basename(first_video_clip.source_path)}': {first_video_clip.source_width}x{first_video_clip.source_height} @ {first_video_clip.source_frame_rate:.2f} fps")
        return (first_video_clip.source_frame_rate, first_video_clip.source_width, first_video_clip.source_height)

    def _build_otio_timeline(self, state: 'State', fps: float, width: int, height: int, base_path_for_relinking: Path) -> otio.schema.Timeline:
        # ... (This function remains mostly unchanged, just passes the base_path down)
        otio_timeline = otio.schema.Timeline(name="Codec Agent Edit")
        self._inject_sequence_metadata(otio_timeline, fps, width, height)
        tracks_data: Dict[int, List[TimelineClip]] = defaultdict(list)
        for clip in state.timeline:
            tracks_data[clip.track_index].append(clip)
        for track_index in sorted(tracks_data.keys()):
            otio_video_track = otio.schema.Track(name=f"V{track_index+1}", kind=otio.schema.TrackKind.Video)
            otio_audio_track = otio.schema.Track(name=f"A{track_index+1}", kind=otio.schema.TrackKind.Audio)
            last_clip_end_time = 0.0
            for codec_clip in tracks_data[track_index]:
                gap_duration = codec_clip.timeline_start_sec - last_clip_end_time
                if gap_duration > 0.001:
                    gap = otio.schema.Gap(source_range=otio.opentime.TimeRange(duration=otio.opentime.from_seconds(gap_duration, rate=fps)))
                    otio_video_track.append(gap)
                    otio_audio_track.append(gap.clone()) # <-- FIX: Use a clone
                otio_clip = self._create_otio_clip(codec_clip, fps, base_path_for_relinking)
                otio_video_track.append(otio_clip)
                otio_audio_track.append(otio_clip.clone())
                last_clip_end_time = codec_clip.timeline_start_sec + codec_clip.duration_sec
            otio_timeline.tracks.append(otio_video_track)
            otio_timeline.tracks.append(otio_audio_track)
        return otio_timeline

    def _inject_sequence_metadata(self, timeline: otio.schema.Timeline, fps: float, width: int, height: int):
        # ... (This function remains unchanged)
        fcp_meta = timeline.metadata.setdefault("fcp_xml", {})
        is_ntsc = abs(fps - 23.976) < 0.01 or abs(fps - 29.97) < 0.01
        fcp_meta["rate"] = {"timebase": str(int(round(fps))), "ntsc": "TRUE" if is_ntsc else "FALSE"}
        fcp_meta.setdefault("media", {}).setdefault("video", {})["format"] = {
            "samplecharacteristics": { "width": str(width), "height": str(height), "pixelaspectratio": "square", "anamorphic": "FALSE", "fielddominance": "none" }
        }

    def _create_otio_clip(self, codec_clip: TimelineClip, timeline_fps: float, base_path_for_relinking: Path) -> otio.schema.Clip:
        # ... (This function remains mostly unchanged, just uses the passed-in base path)
        def _rt(sec: float): return otio.opentime.from_seconds(sec, rate=timeline_fps)
        
        # If consolidating, the source path is now the *copied* file.
        # Otherwise, it's the original. We calculate the relative path from the
        # timeline file's future location to the media's location.
        if "media" in base_path_for_relinking.parts[-2:]: # Heuristic to check if we are in a package
             media_path_in_package = base_path_for_relinking / "media" / Path(codec_clip.source_path).name
             target_url = os.path.relpath(media_path_in_package, start=base_path_for_relinking)
        else:
             target_url = os.path.relpath(codec_clip.source_path, start=base_path_for_relinking)

        target_url = Path(target_url).as_posix()

        available_range = otio.opentime.TimeRange(start_time=_rt(0), duration=_rt(codec_clip.source_total_duration_sec))
        media_ref = otio.schema.ExternalReference(target_url=target_url, available_range=available_range)
        
        # ... (rest of metadata injection is unchanged)
        is_ntsc = abs(codec_clip.source_frame_rate - 23.976) < 0.01 or abs(codec_clip.source_frame_rate - 29.97) < 0.01
        media_ref.metadata.setdefault("fcp_xml", {})["media"] = {
            "video": { "samplecharacteristics": { "width": str(codec_clip.source_width), "height": str(codec_clip.source_height), "pixelaspectratio": "square" }, "rate": { "timebase": str(int(round(codec_clip.source_frame_rate))), "ntsc": "TRUE" if is_ntsc else "FALSE" } },
            "audio": {"samplecharacteristics": {"samplerate": "48000"}}
        }
        source_range = otio.opentime.TimeRange(start_time=_rt(codec_clip.source_in_sec), duration=_rt(codec_clip.duration_sec))
        return otio.schema.Clip(name=codec_clip.clip_id, media_reference=media_ref, source_range=source_range)