import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional
from collections import defaultdict

import opentimelineio as otio
import ffmpeg
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
        "codec_edit.xml",
        description="The desired filename for the exported XML file. It will be saved in the user's Downloads folder. It should end with '.xml'."
    )


class ExportTimelineTool(BaseTool):
    """
    A tool to export the current editing timeline to a Final Cut Pro 7 XML file.
    This file can be imported into professional video editing software like
    Premiere Pro, Final Cut Pro 7, or DaVinci Resolve.
    """

    @property
    def name(self) -> str:
        return "export_timeline"

    @property
    def description(self) -> str:
        return "Exports the current timeline to a Final Cut Pro 7 compatible XML file for import into video editing software. Use this when the user wants to see or finalize the edit."

    @property
    def args_schema(self):
        return ExportTimelineArgs

    # --- REFACTORED EXECUTE METHOD (PLAN 1) ---
    # Notice how it now reads like a high-level summary of the steps.
    def execute(self, state: 'State', args: ExportTimelineArgs, client: 'genai.Client') -> str:
        if not state.timeline:
            return "Error: The timeline is empty. Please add some clips before exporting."

        try:
            # Step 1: Get sequence properties, raising an error if impossible.
            fps, width, height = self._get_or_infer_sequence_properties(state)

            # Step 2: Resolve the output path, raising an error if invalid.
            output_path = self._resolve_output_path(args.output_filename)

            # Step 3: Build the entire OTIO timeline object.
            otio_timeline = self._build_otio_timeline(state, fps, width, height)

            # Step 4: Write the file.
            otio.adapters.write_to_file(otio_timeline, str(output_path), adapter_name="fcp_xml")

        except (ValueError, IOError, Exception) as e:
            # Catch specific errors from our helpers or general OTIO errors.
            return f"Error: {e}"

        return f"✅ Successfully exported timeline to: {output_path}"


    # --- NEW HELPER 1: Path Resolution (PLAN 1) ---
    def _resolve_output_path(self, filename: str) -> Path:
        """Validates filename and returns the full path in the Downloads folder."""
        if not filename.lower().endswith('.xml'):
            raise ValueError("The output filename must end with '.xml'.")
        
        try:
            downloads_path = Path.home() / "Downloads"
            if not downloads_path.is_dir():
                raise IOError(f"Could not find the Downloads folder at '{downloads_path}'.")
            return downloads_path / filename
        except Exception as e:
            raise IOError(f"Could not determine the user's Downloads folder. System error: {e}")


    # --- NEW HELPER 2: Sequence Property Logic (PLAN 1) ---
    def _get_or_infer_sequence_properties(self, state: 'State') -> Tuple[float, int, int]:
        """
        Returns sequence properties from state, or infers them from the first video clip.
        This determines the overall settings of the timeline itself.
        Raises ValueError if properties cannot be determined.
        """
        # Priority 1: Use properties explicitly set on the state
        if all([state.frame_rate, state.width, state.height]):
            return (state.frame_rate, state.width, state.height)

        # Priority 2: Infer from the first video clip on the timeline
        first_video_clip = next((c for c in state.timeline if c.source_path.lower().endswith(('.mp4', '.mov', '.mkv', '.mxf'))), None)
        if not first_video_clip:
            raise ValueError("Could not determine timeline properties (frame rate, resolution). No video clips found on timeline to infer from.")
        
        # Use the properties we already stored on the clip!
        print(f"✅ Inferred sequence properties from '{os.path.basename(first_video_clip.source_path)}': {first_video_clip.source_width}x{first_video_clip.source_height} @ {first_video_clip.source_frame_rate:.2f} fps")
        return (first_video_clip.source_frame_rate, first_video_clip.source_width, first_video_clip.source_height)


    # --- NEW HELPER 3: Main Timeline Construction (PLAN 1) ---
    def _build_otio_timeline(self, state: 'State', fps: float, width: int, height: int) -> otio.schema.Timeline:
        """Constructs the full OTIO timeline object with all tracks and clips."""
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
                # Add a gap if necessary
                gap_duration = codec_clip.timeline_start_sec - last_clip_end_time
                if gap_duration > 0.001:
                    gap = otio.schema.Gap(source_range=otio.opentime.TimeRange(duration=otio.opentime.from_seconds(gap_duration, rate=fps)))
                    otio_video_track.append(gap)
                    otio_audio_track.append(gap)

                # Create and add the actual clip, passing the sequence fps for timing
                otio_clip = self._create_otio_clip(codec_clip, fps)
                otio_video_track.append(otio_clip)
                otio_audio_track.append(otio_clip.clone()) # Use a clone for the audio track

                last_clip_end_time = codec_clip.timeline_start_sec + codec_clip.duration_sec

            otio_timeline.tracks.append(otio_video_track)
            otio_timeline.tracks.append(otio_audio_track)
            
        return otio_timeline


    # --- NEW HELPER 4: Sequence Metadata Injection (PLAN 1) ---
    def _inject_sequence_metadata(self, timeline: otio.schema.Timeline, fps: float, width: int, height: int):
        """Injects FCP7XML-specific metadata for the main sequence."""
        fcp_meta = timeline.metadata.setdefault("fcp_xml", {})
        is_ntsc = abs(fps - 23.976) < 0.01 or abs(fps - 29.97) < 0.01
        fcp_meta["rate"] = {"timebase": str(int(round(fps))), "ntsc": "TRUE" if is_ntsc else "FALSE"}
        fcp_meta.setdefault("media", {}).setdefault("video", {})["format"] = {
            "samplecharacteristics": {
                "width": str(width), "height": str(height),
                "pixelaspectratio": "square", "anamorphic": "FALSE", "fielddominance": "none"
            }
        }

    # --- NEW HELPER 5: Individual Clip Creation (PLAN 1 + PLAN 2) ---
    def _create_otio_clip(self, codec_clip: TimelineClip, timeline_fps: float) -> otio.schema.Clip:
        """Creates a single OTIO clip from our internal TimelineClip representation."""
        # Define time using the sequence's frame rate for correct placement
        def _rt(sec: float): return otio.opentime.from_seconds(sec, rate=timeline_fps)

        # Create the reference to the external media file
        available_range = otio.opentime.TimeRange(start_time=_rt(0), duration=_rt(codec_clip.source_total_duration_sec))
        media_ref = otio.schema.ExternalReference(target_url=Path(codec_clip.source_path).as_uri(), available_range=available_range)

        # --- CRUCIAL FIX (PLAN 2) ---
        # Use the clip's OWN properties for its source metadata.
        # This tells the NLE the true properties of THIS specific file.
        is_ntsc = abs(codec_clip.source_frame_rate - 23.976) < 0.01 or abs(codec_clip.source_frame_rate - 29.97) < 0.01
        
        media_ref.metadata.setdefault("fcp_xml", {})["media"] = {
            "video": {
                "samplecharacteristics": {
                    "width": str(codec_clip.source_width),
                    "height": str(codec_clip.source_height),
                    "pixelaspectratio": "square"
                },
                "rate": {
                    "timebase": str(int(round(codec_clip.source_frame_rate))),
                    "ntsc": "TRUE" if is_ntsc else "FALSE"
                }
            },
            "audio": {"samplecharacteristics": {"samplerate": "48000"}}
        }

        # Define the in/out points for this clip
        source_range = otio.opentime.TimeRange(start_time=_rt(codec_clip.source_in_sec), duration=_rt(codec_clip.duration_sec))

        return otio.schema.Clip(name=codec_clip.clip_id, media_reference=media_ref, source_range=source_range)