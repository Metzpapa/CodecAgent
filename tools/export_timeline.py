import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional
from collections import defaultdict

import opentimelineio as otio
import ffmpeg
from pydantic import BaseModel, Field

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

    def _get_downloads_path(self) -> Path:
        """
        Returns the path to the user's Downloads folder in a cross-platform way.
        """
        return Path.home() / "Downloads"

    def _infer_timeline_properties(self, state: 'State') -> Optional[Tuple[float, int, int]]:
        """
        Finds the first video clip on the timeline and probes it to get the
        frame rate and resolution for the entire sequence.
        """
        first_video_clip = next((c for c in state.timeline if c.source_path.lower().endswith(('.mp4', '.mov', '.mkv', '.mxf'))), None)
        if not first_video_clip:
            return None
        
        try:
            probe = ffmpeg.probe(first_video_clip.source_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if not video_stream:
                return None

            fr_str = video_stream.get('r_frame_rate', '0/1')
            num, den = map(int, fr_str.split('/'))
            frame_rate = num / den if den > 0 else 0
            width = video_stream.get('width')
            height = video_stream.get('height')

            if all([frame_rate, width, height]):
                print(f"✅ Inferred timeline properties from '{os.path.basename(first_video_clip.source_path)}': {width}x{height} @ {frame_rate:.2f} fps")
                return (frame_rate, width, height)
            return None
        except Exception:
            return None


    def execute(self, state: 'State', args: ExportTimelineArgs) -> str:
        # --- 1. Validation ---
        if not state.timeline:
            return "Error: The timeline is empty. Please add some clips before exporting."

        if not args.output_filename.lower().endswith('.xml'):
            return "Error: The output filename must end with '.xml'."

        # --- 2. Determine Timeline Properties ---
        timeline_fps, timeline_width, timeline_height = state.frame_rate, state.width, state.height
        if not all([timeline_fps, timeline_width, timeline_height]):
            inferred_props = self._infer_timeline_properties(state)
            if not inferred_props:
                return "Error: Could not determine timeline properties (frame rate, resolution). No video clips found on timeline to infer from."
            timeline_fps, timeline_width, timeline_height = inferred_props

        # --- 3. Find Output Path ---
        try:
            downloads_path = self._get_downloads_path()
            if not downloads_path.is_dir():
                return f"Error: Could not find the Downloads folder at '{downloads_path}'. Please ensure it exists."
            output_path = downloads_path / args.output_filename
        except Exception as e:
            return f"Error: Could not determine the user's Downloads folder. System error: {e}"

        # --- 4. Build the OTIO Timeline ---
        otio_timeline = otio.schema.Timeline(name="Codec Agent Edit")

        def _rt(sec: float):
            return otio.opentime.from_seconds(sec, rate=timeline_fps)

        # --- FIX 1: Inject correct SEQUENCE metadata ---
        fcp_meta = otio_timeline.metadata.setdefault("fcp_xml", {})
        
        # Update NTSC detection logic to handle fractional framerates
        is_ntsc = abs(timeline_fps - 23.976) < 0.01 or abs(timeline_fps - 29.97) < 0.01
        timebase = int(round(timeline_fps))
        
        # Rate (keep our ntsc logic, overwrite what OTIO writes)
        fcp_meta["rate"] = {
            "timebase": str(timebase),
            "ntsc": "TRUE" if is_ntsc else "FALSE"
        }
        
        # Video format – THIS is the bit Premiere reads
        seq_format = {
            "samplecharacteristics": {
                "width": str(timeline_width),
                "height": str(timeline_height),
                "pixelaspectratio": "square",
                "anamorphic": "FALSE",
                "fielddominance": "none"
            }
        }
        
        fcp_meta\
            .setdefault("media", {})\
            .setdefault("video", {})\
            ["format"] = seq_format

        tracks_data: Dict[int, List[TimelineClip]] = defaultdict(list)
        for clip in state.timeline:
            tracks_data[clip.track_index].append(clip)

        for track_index in sorted(tracks_data.keys()):
            otio_video_track = otio.schema.Track(name=f"V{track_index+1}", kind=otio.schema.TrackKind.Video)
            otio_audio_track = otio.schema.Track(name=f"A{track_index+1}", kind=otio.schema.TrackKind.Audio)
            
            last_clip_end_time_on_track = 0.0

            for codec_clip in tracks_data[track_index]:
                gap_duration = codec_clip.timeline_start_sec - last_clip_end_time_on_track
                if gap_duration > 0.001:
                    gap = otio.schema.Gap(source_range=otio.opentime.TimeRange(duration=_rt(gap_duration)))
                    otio_video_track.append(gap)
                    otio_audio_track.append(gap)

                available_range = otio.opentime.TimeRange(start_time=_rt(0), duration=_rt(codec_clip.source_total_duration_sec))
                media_ref = otio.schema.ExternalReference(target_url=Path(codec_clip.source_path).as_uri(), available_range=available_range)
                source_range = otio.opentime.TimeRange(start_time=_rt(codec_clip.source_in_sec), duration=_rt(codec_clip.duration_sec))

                # --- FIX 2 (CRUCIAL): Inject correct SOURCE FILE metadata ---
                # This tells Premiere how to interpret the source file itself, preventing fallback to wrong defaults.
                media_ref.metadata.setdefault("fcp_xml", {})["media"] = {
                    "video": {
                        "samplecharacteristics": {
                            "width": str(timeline_width),
                            "height": str(timeline_height),
                            "pixelaspectratio": "square"
                        }
                    },
                    "audio": { # Also good practice to declare audio characteristics
                        "samplecharacteristics": { "samplerate": "48000" }
                    }
                }
                # --- END OF CRUCIAL FIX ---

                otio_clip = otio.schema.Clip(name=codec_clip.clip_id, media_reference=media_ref, source_range=source_range)

                otio_video_track.append(otio_clip)
                otio_audio_track.append(otio_clip.clone())

                last_clip_end_time_on_track = codec_clip.timeline_start_sec + codec_clip.duration_sec

            otio_timeline.tracks.append(otio_video_track)
            otio_timeline.tracks.append(otio_audio_track)

        # --- 5. Write to File ---
        try:
            otio.adapters.write_to_file(otio_timeline, str(output_path), adapter_name="fcp_xml")
        except Exception as e:
            return f"Error: Failed to write the XML file. OTIO error: {e}"

        return f"✅ Successfully exported timeline to: {output_path}"