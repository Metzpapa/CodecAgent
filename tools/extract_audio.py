# codec/tools/extract_audio.py

import os
import tempfile
from typing import Optional, TYPE_CHECKING

import ffmpeg
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from .base import BaseTool

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State


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
            "Extracts an audio segment from a video or audio file. "
            "Use this to 'hear' the contents of the audio file. "
            "This tool is best used for general understanding of the audio. As it allows you to 'hear' everything. Use this tool to extract the raw audio.  You should be analyzing the raw audio. "
        )

    @property
    def args_schema(self):
        return ExtractAudioArgs

    def _hms_to_seconds(self, time_str: str) -> float:
        """Converts HH:MM:SS.mmm format to total seconds."""
        parts = time_str.split(':')
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split('.')
        s = int(s_parts[0])
        ms = int(s_parts[1].ljust(3, '0')) if len(s_parts) > 1 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0

    def execute(self, state: 'State', args: ExtractAudioArgs, client: 'genai.Client') -> str | types.Content:
        # --- 1. Validation & Setup ---
        full_path = os.path.join(state.assets_directory, args.source_filename)
        if not os.path.exists(full_path):
            return f"Error: The source file '{args.source_filename}' does not exist in the assets directory."

        try:
            probe = ffmpeg.probe(full_path)
            audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            if not audio_stream:
                return f"Error: Source file '{args.source_filename}' does not contain an audio stream."

            duration_str = audio_stream.get('duration') or probe['format'].get('duration', '0')
            source_duration = float(duration_str)
            if source_duration <= 0:
                return f"Error: Could not determine a valid duration for '{args.source_filename}'."
        except ffmpeg.Error as e:
            return f"Error: Failed to probe '{args.source_filename}'. It may be corrupt or not a valid media file. FFmpeg error: {e.stderr.decode()}"

        # --- 2. Time Calculation ---
        start_sec = self._hms_to_seconds(args.start_time) if args.start_time else 0.0
        end_sec = self._hms_to_seconds(args.end_time) if args.end_time else source_duration

        if start_sec >= end_sec:
            return "Error: The start_time must be before the end_time."
        
        if end_sec > source_duration:
            end_sec = source_duration
        
        duration_to_extract = end_sec - start_sec
        if duration_to_extract <= 0:
            return "Error: The calculated duration for extraction is zero or negative."

        # --- 3. Audio Extraction ---
        print(f"Extracting {duration_to_extract:.2f}s of audio from '{args.source_filename}' starting at {start_sec:.2f}s...")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "extracted_audio.mp3")

            try:
                (
                    ffmpeg.input(full_path, ss=start_sec, t=duration_to_extract)
                    .output(output_path, acodec='libmp3lame', audio_bitrate='192k', format='mp3')
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
            except ffmpeg.Error as e:
                return f"Error during audio extraction. Details: {e.stderr.decode()}"

            # --- 4. Upload and Response Assembly ---
            print(f"Extraction complete. Uploading '{output_path}'...")
            try:
                with open(output_path, "rb") as f:
                    audio_file = client.files.upload(
                        file=f,
                        config={
                            "mimeType": "audio/mpeg",
                            "displayName": f"audio-{args.source_filename}-{start_sec:.2f}s-{end_sec:.2f}s"
                        }
                    )
                print(f"Upload complete. File name: {audio_file.name}")
            except Exception as e:
                return f"Failed to upload extracted audio file. Details: {str(e)}"

            # Add the file to the state for cleanup at the end of the session
            state.uploaded_files.append(audio_file)

            # Construct the multimodal response for the agent
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