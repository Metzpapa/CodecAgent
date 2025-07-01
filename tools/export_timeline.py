import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List
from collections import defaultdict

import opentimelineio as otio
from pydantic import BaseModel, Field

from .base import BaseTool
from state import TimelineClip

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State


class ExportTimelineArgs(BaseModel):
    """Arguments for the export_timeline tool."""
    output_filename: str = Field(
        "codec_edit.xml", # CHANGED: Default to .xml for FCP7 format
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

    def execute(self, state: 'State', args: ExportTimelineArgs) -> str:
        # --- 1. Validation ---
        if not state.timeline:
            return "Error: The timeline is empty. Please add some clips before exporting."

        # CHANGED: Validate for .xml extension
        if not args.output_filename.lower().endswith('.xml'):
            return "Error: The output filename must end with '.xml'."

        # --- 2. Find Output Path ---
        try:
            downloads_path = self._get_downloads_path()
            if not downloads_path.is_dir():
                return f"Error: Could not find the Downloads folder at '{downloads_path}'. Please ensure it exists."
            output_path = downloads_path / args.output_filename
        except Exception as e:
            return f"Error: Could not determine the user's Downloads folder. System error: {e}"

        # --- 3. Group Clips by Track ---
        tracks_data: Dict[int, List[TimelineClip]] = defaultdict(list)
        for clip in state.timeline:
            tracks_data[clip.track_index].append(clip)

        # --- 4. Build the OTIO Timeline ---
        otio_timeline = otio.schema.Timeline(name="Codec Agent Edit")

        for track_index in sorted(tracks_data.keys()):
            otio_track = otio.schema.Track(name=f"Track {track_index}")
            last_clip_end_time_on_track = 0.0

            for codec_clip in tracks_data[track_index]:
                # --- A. Handle Gaps ---
                gap_duration = codec_clip.timeline_start_sec - last_clip_end_time_on_track
                if gap_duration > 0.001:
                    gap = otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            duration=otio.opentime.from_seconds(gap_duration)
                        )
                    )
                    otio_track.append(gap)

                # --- B. Create the OTIO Clip ---
                available_range = otio.opentime.TimeRange(
                    start_time=otio.opentime.from_seconds(0),
                    duration=otio.opentime.from_seconds(codec_clip.source_total_duration_sec)
                )
                media_ref = otio.schema.ExternalReference(
                    target_url=Path(codec_clip.source_path).as_uri(),
                    available_range=available_range
                )
                
                source_range = otio.opentime.TimeRange(
                    start_time=otio.opentime.from_seconds(codec_clip.source_in_sec),
                    duration=otio.opentime.from_seconds(codec_clip.duration_sec)
                )

                otio_clip = otio.schema.Clip(
                    name=codec_clip.clip_id,
                    media_reference=media_ref,
                    source_range=source_range
                )
                otio_track.append(otio_clip)

                last_clip_end_time_on_track = codec_clip.timeline_start_sec + codec_clip.duration_sec

            otio_timeline.tracks.append(otio_track)

        # --- 5. Write to File ---
        try:
            # CHANGED: Explicitly specify the 'fcp_xml' adapter for FCP7 format.
            otio.adapters.write_to_file(
                otio_timeline,
                str(output_path),
                adapter_name="fcp_xml"  # Final Cut Pro 7 (.xml) adapter
            )
        except Exception as e:
            return f"Error: Failed to write the XML file. OTIO error: {e}"

        return f"✅ Successfully exported timeline to: {output_path}"