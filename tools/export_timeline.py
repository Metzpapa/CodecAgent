# codec/tools/export_timeline.py

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional
from collections import defaultdict
import datetime

import opentimelineio as otio
from pydantic import BaseModel, Field

from .base import BaseTool
from state import TimelineClip

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State
    from llm.base import LLMConnector


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

    def execute(self, state: 'State', args: ExportTimelineArgs, connector: 'LLMConnector') -> str:
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

                unique_source_paths = {clip.source_path for clip in state.timeline}
                for src_path_str in unique_source_paths:
                    src_path = Path(src_path_str)
                    dest_path = media_dir / src_path.name
                    print(f"  - Copying {src_path.name}...")
                    shutil.copy2(src_path, dest_path)

                base_path_for_relinking = package_dir
                output_path = package_dir / args.output_filename
                success_message = f"✅ Successfully consolidated project! Find the complete folder at: {package_dir}"

            else:
                # --- NON-CONSOLIDATED LOGIC ---
                output_path = downloads_path / args.output_filename
                base_path_for_relinking = downloads_path
                success_message = f"✅ Successfully exported timeline file to: {output_path}"


            # --- COMMON BUILD & WRITE LOGIC ---
            fps, width, height = state.get_sequence_properties()
            
            print(f"✅ Using sequence properties: {width}x{height} @ {fps:.2f} fps")
            # MODIFIED: Pass the `consolidate` flag down to the build methods
            otio_timeline = self._build_otio_timeline(state, fps, width, height, base_path_for_relinking, consolidated=args.consolidate)

            file_ext = output_path.suffix.lower()
            adapter_name = "otio_json" if file_ext == ".otio" else "fcp_xml" if file_ext == ".xml" else None
            if not adapter_name:
                return f"Error: Unsupported file extension '{file_ext}'. Please use '.otio' or '.xml'."

            otio.adapters.write_to_file(otio_timeline, str(output_path), adapter_name=adapter_name)

        except Exception as e:
            return f"Error during export: {e}"

        return success_message

    def _build_otio_timeline(self, state: 'State', fps: float, width: int, height: int, base_path_for_relinking: Path, consolidated: bool) -> otio.schema.Timeline:
        """
        Builds the OTIO timeline by directly translating the state's track
        and clip structure.
        
        MODIFIED: Accepts a `consolidated` flag to pass to the clip creator.
        """
        otio_timeline = otio.schema.Timeline(name="Codec Agent Edit")
        self._inject_sequence_metadata(otio_timeline, fps, width, height)

        # Group clips by their actual track from the state
        clips_by_track = defaultdict(list)
        for clip in state.timeline:
            clips_by_track[(clip.track_type, clip.track_number)].append(clip)

        # Get all unique tracks and sort them (V1, V2, ..., A1, A2, ...)
        sorted_tracks = sorted(clips_by_track.keys(), key=lambda t: (t[0], t[1]))

        for track_type, track_number in sorted_tracks:
            track_name = f"{track_type[0].upper()}{track_number}"
            track_kind = otio.schema.TrackKind.Video if track_type == 'video' else otio.schema.TrackKind.Audio
            otio_track = otio.schema.Track(name=track_name, kind=track_kind)

            last_clip_end_time = 0.0
            # The clips for this track are already sorted by time because state.timeline is sorted
            for codec_clip in clips_by_track[(track_type, track_number)]:
                # Handle gap before the clip
                gap_duration = codec_clip.timeline_start_sec - last_clip_end_time
                if gap_duration > 0.001:
                    gap = otio.schema.Gap(source_range=otio.opentime.TimeRange(duration=otio.opentime.from_seconds(gap_duration, rate=fps)))
                    otio_track.append(gap)
                
                # Create and append the OTIO clip, passing the consolidation flag
                otio_clip = self._create_otio_clip(codec_clip, fps, base_path_for_relinking, consolidated)
                otio_track.append(otio_clip)

                last_clip_end_time = codec_clip.timeline_start_sec + codec_clip.duration_sec
            
            otio_timeline.tracks.append(otio_track)

        return otio_timeline

    def _inject_sequence_metadata(self, timeline: otio.schema.Timeline, fps: float, width: int, height: int):
        """Injects FCP XML-specific metadata for better compatibility."""
        fcp_meta = timeline.metadata.setdefault("fcp_xml", {})
        is_ntsc = abs(fps - 23.976) < 0.01 or abs(fps - 29.97) < 0.01
        fcp_meta["rate"] = {"timebase": str(int(round(fps))), "ntsc": "TRUE" if is_ntsc else "FALSE"}
        fcp_meta.setdefault("media", {}).setdefault("video", {})["format"] = {
            "samplecharacteristics": { "width": str(width), "height": str(height), "pixelaspectratio": "square", "anamorphic": "FALSE", "fielddominance": "none" }
        }

    def _create_otio_clip(self, codec_clip: TimelineClip, timeline_fps: float, base_path_for_relinking: Path, consolidated: bool) -> otio.schema.Clip:
        """
        Creates a single OTIO clip from a TimelineClip, handling path relinking.
        
        MODIFIED: Now correctly generates the `target_url` based on whether the
        export is consolidated or not, fixing the relinking bug.
        """
        def _rt(sec: float): return otio.opentime.from_seconds(sec, rate=timeline_fps)
        
        if consolidated:
            # For a consolidated project, the path is always a simple relative path
            # into the 'media' subfolder. This is the key fix.
            target_url = (Path("media") / Path(codec_clip.source_path).name).as_posix()
        else:
            # For a non-consolidated export, calculate the path relative from the export
            # location to the original asset location.
            target_url = os.path.relpath(codec_clip.source_path, start=base_path_for_relinking)
            target_url = Path(target_url).as_posix()

        available_range = otio.opentime.TimeRange(start_time=_rt(0), duration=_rt(codec_clip.source_total_duration_sec))
        media_ref = otio.schema.ExternalReference(target_url=target_url, available_range=available_range)
        
        # Inject source media metadata for better NLE compatibility
        media_ref_meta = media_ref.metadata.setdefault("fcp_xml", {}).setdefault("media", {})
        if codec_clip.source_width > 0 and codec_clip.source_height > 0:
            is_ntsc = abs(codec_clip.source_frame_rate - 23.976) < 0.01 or abs(codec_clip.source_frame_rate - 29.97) < 0.01
            media_ref_meta["video"] = {
                "samplecharacteristics": { "width": str(codec_clip.source_width), "height": str(codec_clip.source_height), "pixelaspectratio": "square" },
                "rate": { "timebase": str(int(round(codec_clip.source_frame_rate))), "ntsc": "TRUE" if is_ntsc else "FALSE" }
            }
        if codec_clip.has_audio:
            media_ref_meta["audio"] = {"samplecharacteristics": {"samplerate": "48000"}}

        source_range = otio.opentime.TimeRange(start_time=_rt(codec_clip.source_in_sec), duration=_rt(codec_clip.duration_sec))
        return otio.schema.Clip(name=codec_clip.clip_id, media_reference=media_ref, source_range=source_range)