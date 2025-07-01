import os
from typing import List, TYPE_CHECKING

import ffmpeg
from pydantic import BaseModel, Field

from tools.base import BaseTool

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State


class GetAssetInfoArgs(BaseModel):
    """Arguments for the get_asset_info tool."""
    filenames: List[str] = Field(
        ...,
        description="A list of asset filenames to get information for (e.g., ['video1.mp4', 'audio_bg.wav'])."
    )


class GetAssetInfoTool(BaseTool):
    """A tool to retrieve essential metadata from media files."""

    @property
    def name(self) -> str:
        return "get_asset_info"

    @property
    def description(self) -> str:
        return (
            "Retrieves essential metadata (duration, resolution, frame rate) for one or more media files. "
            "Use this to understand the properties of an asset before adding it to the timeline."
        )

    @property
    def args_schema(self) -> type[BaseModel]:
        return GetAssetInfoArgs

    def execute(self, state: 'State', args: GetAssetInfoArgs) -> str:
        """
        Probes each requested file using ffmpeg to extract and format its metadata.
        """
        results = []
        for filename in args.filenames:
            full_path = os.path.join(state.assets_directory, filename)

            if not os.path.isfile(full_path):
                results.append(f"File: {filename}\n  - Status: Error - File not found.")
                continue

            try:
                probe = ffmpeg.probe(full_path)
                
                # Find the primary video stream
                video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                
                # Find the primary audio stream
                audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

                if not video_stream and not audio_stream:
                    results.append(f"File: {filename}\n  - Status: Error - Not a valid media file (no video or audio streams).")
                    continue

                # Format the output string for this file
                info_lines = [f"File: {filename}", "  - Status: OK"]
                
                # Use video stream duration if available, otherwise fall back to format duration
                duration_str = video_stream.get('duration') if video_stream else probe['format'].get('duration', '0')
                info_lines.append(f"  - Duration: {float(duration_str):.2f} seconds")

                if video_stream:
                    width = video_stream.get('width', 'N/A')
                    height = video_stream.get('height', 'N/A')
                    info_lines.append(f"  - Resolution: {width}x{height}")

                    # Safely parse frame rate (it's often a fraction like '30/1')
                    fr_str = video_stream.get('r_frame_rate', '0/1')
                    num, den = map(int, fr_str.split('/'))
                    frame_rate = num / den if den > 0 else 0
                    info_lines.append(f"  - Frame Rate: {frame_rate:.2f} fps")
                
                if audio_stream:
                    sample_rate = audio_stream.get('sample_rate', 'N/A')
                    info_lines.append(f"  - Audio: Yes ({sample_rate} Hz)")
                else:
                    info_lines.append("  - Audio: No")

                results.append("\n".join(info_lines))

            except ffmpeg.Error as e:
                results.append(f"File: {filename}\n  - Status: Error - FFmpeg failed to probe file. It may be corrupt.\n  - FFmpeg Error: {e.stderr.decode('utf-8').strip()}")

        # Join the results for all files into a single string
        return "\n\n".join(results)