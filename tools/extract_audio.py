# codec/tools/extract_audio.py

import os
import tempfile
from typing import Optional, TYPE_CHECKING, Union
from pathlib import Path

import ffmpeg
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from .base import BaseTool
from utils import hms_to_seconds, probe_media_file # <-- MODIFIED IMPORT

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State


def _extract_and_upload_audio_segment(
    file_path: Union[str, Path],
    start_sec: float,
    duration_sec: float,
    display_name: str,
    client: 'genai.Client',
    tmpdir: str
) -> Union[types.File, str]:
    """
    Core reusable logic to extract an audio segment from a media file, save it
    temporarily, and upload it to the Gemini API.

    Args:
        file_path: The absolute path to the source media file.
        start_sec: The start time of the segment to extract.
        duration_sec: The duration of the segment to extract.
        display_name: The display name for the uploaded file.
        client: The genai.Client instance.
        tmpdir: The temporary directory to store the extracted audio.

    Returns:
        A google.genai.types.File object on success, or an error string on failure.
    """
    output_path = Path(tmpdir) / f"audio_{os.path.basename(file_path)}_{start_sec:.2f}s.mp3"
    try:
        # 1. Extract audio using ffmpeg
        (
            ffmpeg.input(str(file_path), ss=start_sec, t=duration_sec)
            .output(str(output_path), acodec='libmp3lame', audio_bitrate='192k', format='mp3')
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )

        # 2. Upload the extracted audio
        print(f"Uploading audio from '{display_name}' (duration: {duration_sec:.2f}s)...")
        with open(output_path, "rb") as f:
            audio_file = client.files.upload(
                file=f,
                config={"mimeType": "audio/mpeg", "displayName": display_name}
            )
        print(f"Upload complete for '{display_name}'. Name: {audio_file.name}")
        return audio_file

    except Exception as e:
        error_msg = f"Failed to extract or upload audio for '{display_name}'. Details: {e}"
        print(error_msg)
        return error_msg


class ExtractAudioArgs(BaseModel):
    """Arguments for the extract_audio tool."""
    source_filename: str = Field(
        ...,
        description="The exact name of the video or audio file from the user's media library to extract audio from (e.g., 'interview.mp4', 'background_music.wav')."
    )
    start_time: Optional[str] = Field(
        None,
        description="The timestamp in the source file to start extracting audio from. Format: HH:MM:SS.mmm. If omitted, starts from the beginning.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    end_time: Optional[str] = Field(
        None,
        description="The timestamp in the source file to stop extracting audio at. Format: HH:MM:SS.mmm. If omitted, uses the full file duration.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )


class ExtractAudioTool(BaseTool):
    """
    A tool to extract a segment of audio from a media file and provide it
    to the model for native processing.
    """

    @property
    def name(self) -> str:
        return "extract_audio"

    @property
    def description(self) -> str:
        return (
            "Extracts an audio segment from a single video or audio file. "
            "Use this to 'hear' the contents of a source file. "
            "To hear the audio from the composed timeline, use 'extract_timeline_audio'."
        )

    @property
    def args_schema(self):
        return ExtractAudioArgs

    # REMOVED: _hms_to_seconds helper function

    def execute(self, state: 'State', args: ExtractAudioArgs, client: 'genai.Client') -> str | types.Content:
        # --- 1. Validation & Setup ---
        full_path = os.path.join(state.assets_directory, args.source_filename)
        if not os.path.exists(full_path):
            return f"Error: The source file '{args.source_filename}' does not exist in the assets directory."

        # Use the new utility to probe the file
        media_info = probe_media_file(full_path)
        if media_info.error:
            return f"Error probing '{args.source_filename}': {media_info.error}"
        
        if not media_info.has_audio:
            return f"Error: Source file '{args.source_filename}' does not contain an audio stream."
        
        if media_info.duration_sec <= 0:
            return f"Error: Could not determine a valid duration for '{args.source_filename}'."

        source_duration = media_info.duration_sec

        # --- 2. Time Calculation ---
        start_sec = hms_to_seconds(args.start_time) if args.start_time else 0.0
        end_sec = hms_to_seconds(args.end_time) if args.end_time else source_duration

        if start_sec >= end_sec:
            return "Error: The start_time must be before the end_time."
        
        if end_sec > source_duration:
            end_sec = source_duration
        
        duration_to_extract = end_sec - start_sec
        if duration_to_extract <= 0:
            return "Error: The calculated duration for extraction is zero or negative."

        # --- 3. Audio Extraction using the reusable helper ---
        with tempfile.TemporaryDirectory() as tmpdir:
            display_name = f"audio-{args.source_filename}-{start_sec:.2f}s-{end_sec:.2f}s"
            result = _extract_and_upload_audio_segment(
                full_path, start_sec, duration_to_extract, display_name, client, tmpdir
            )

            if isinstance(result, str): # It's an error string
                return result

            audio_file = result
            state.uploaded_files.append(audio_file)

            # --- 4. Construct the multimodal response ---
            context_text = (
                f"SYSTEM: This is the output of the `extract_audio` tool you called for '{args.source_filename}'. "
                f"This is the audio content from {start_sec:.2f}s to {end_sec:.2f}s."
            )
            
            response_content = types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=context_text),
                    types.Part.from_uri(
                        file_uri=audio_file.uri,
                        mime_type='audio/mpeg'
                    )
                ]
            )
            
            return response_content