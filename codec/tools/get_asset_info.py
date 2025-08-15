import os
from typing import List, TYPE_CHECKING
import openai

from pydantic import BaseModel, Field

from .base import BaseTool
from ..utils import probe_media_file

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from ..state import State


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

    def execute(self, state: 'State', args: GetAssetInfoArgs, client: openai.OpenAI) -> str:
        """
        Probes each requested file using the centralized utility to extract and format its metadata.
        """
        results = []
        for filename in args.filenames:
            full_path = os.path.join(state.assets_directory, filename)

            if not os.path.isfile(full_path):
                results.append(f"File: {filename}\n  - Status: Error - File not found.")
                continue

            media_info = probe_media_file(full_path)

            if media_info.error:
                results.append(f"File: {filename}\n  - Status: Error - {media_info.error}")
                continue

            # Format the output string for this file using the structured MediaInfo object
            info_lines = [f"File: {filename}", "  - Status: OK"]
            info_lines.append(f"  - Duration: {media_info.duration_sec:.2f} seconds")

            if media_info.has_video:
                info_lines.append(f"  - Resolution: {media_info.width}x{media_info.height}")
                info_lines.append(f"  - Frame Rate: {media_info.frame_rate:.2f} fps")
            
            if media_info.has_audio:
                info_lines.append("  - Audio: Yes")
            else:
                info_lines.append("  - Audio: No")

            results.append("\n".join(info_lines))

        # Join the results for all files into a single string
        return "\n\n".join(results)