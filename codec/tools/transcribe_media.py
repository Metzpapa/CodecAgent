# codec/tools/transcribe_media.py

import os
import tempfile
import ffmpeg
import logging
from typing import Optional, Literal, TYPE_CHECKING, Dict, Any
from pydantic import BaseModel, Field
import openai

# Local imports
from .base import BaseTool
from ..utils import probe_media_file, seconds_to_hms

if TYPE_CHECKING:
    from ..state import State

# OpenAI's Whisper API has a 25 MB file size limit.
WHISPER_API_LIMIT_BYTES = 25 * 1024 * 1024




class TranscribeMediaArgs(BaseModel):
    """Arguments for the transcribe_media tool."""
    source_filename: Optional[str] = Field(
        None,
        description="The filename of the asset to transcribe (e.g., 'interview.mp4'). If this is omitted, the tool will transcribe the entire timeline instead."
    )
    granularity: Literal["segment", "word"] = Field(
        "segment",
        description=(
            "The level of detail for timestamps. "
            "'segment': (Default) Groups text into spoken phrases or segments, which is best for general understanding. "
            "'word': Provides a timestamp for every single word, which is useful for precise timing."
        )
    )
    language: Optional[str] = Field(
        None,
        description="Optional. The two-letter ISO-639-1 language code of the audio (e.g., 'en', 'es', 'ja'). If omitted, the language will be auto-detected."
    )
    prompt: Optional[str] = Field(
        None,
        description="Optional. A 'prompt' to provide context, which can improve the recognition of specific names, jargon, or acronyms (e.g., 'This is a podcast about Codec and FFmpeg.')."
    )


class TranscribeMediaTool(BaseTool):
    """
    A tool to extract spoken text from a media source, either a single asset or the entire timeline.
    """

    @property
    def name(self) -> str:
        return "transcribe_media"

    @property
    def description(self) -> str:
        return (
            "Extracts spoken text from a media source. It can operate on a single asset file from the library or on the entire rendered timeline. "
            "The tool returns the full transcription along with timestamps, with adjustable granularity. "
            "Use this to understand dialogue, find specific spoken phrases, or create subtitles."
        )

    @property
    def args_schema(self):
        return TranscribeMediaArgs

    def execute(self, state: 'State', args: TranscribeMediaArgs, client: openai.OpenAI, tmpdir: str) -> str:
        try:
            if args.source_filename:
                return self._transcribe_asset(state, args, client)
            else:
                return self._transcribe_timeline(state, args, client)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"An unexpected error occurred during transcription: {e}"

    def _transcribe_asset(self, state: 'State', args: TranscribeMediaArgs, client: openai.OpenAI) -> str:
        """Handles transcription for a single asset file."""
        source_path = os.path.join(state.assets_directory, args.source_filename)
        if not os.path.exists(source_path):
            return f"Error: Asset file '{args.source_filename}' not found."

        media_info = probe_media_file(source_path)
        if not media_info.has_audio:
            return f"Error: Asset file '{args.source_filename}' contains no audio stream to transcribe."

        logging.info(f"Extracting audio from asset: {args.source_filename}")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp_audio_file:
            tmp_audio_path = tmp_audio_file.name
            try:
                (
                    ffmpeg.input(source_path)
                    .output(tmp_audio_path, acodec='libmp3lame', audio_bitrate='128k', ar='16000', ac=1)
                    .run(overwrite_output=True, quiet=True)
                )
                
                # --- FIX: Proactively check file size before uploading ---
                file_size = os.path.getsize(tmp_audio_path)
                logging.info(f"Audio extracted to temporary file. Size: {file_size / (1024*1024):.2f} MB")
                if file_size > WHISPER_API_LIMIT_BYTES:
                    return (
                        f"Error: The extracted audio from '{args.source_filename}' is too large ({file_size / (1024*1024):.2f} MB) "
                        f"for the transcription API (limit is 25 MB). "
                        f"Consider using the 'find_media' tool with the 'download_range' argument to transcribe a smaller portion of the file."
                    )
                # --- END FIX ---

            except ffmpeg.Error as e:
                return f"Error extracting audio from '{args.source_filename}': {e.stderr.decode()}"

            logging.info(f"Transcribing extracted audio from: {args.source_filename}")
            with open(tmp_audio_path, "rb") as audio_file_handle:
                whisper_result = self._run_whisper(audio_file_handle, args, client)
        
        return self._format_transcription(whisper_result, args.granularity, f"Transcription for '{args.source_filename}'")

    def _transcribe_timeline(self, state: 'State', args: TranscribeMediaArgs, client: openai.OpenAI) -> str:
        """Handles transcription for the entire timeline by rendering its audio first."""
        if not any(c.track_type == 'audio' for c in state.timeline):
            return "Error: The timeline contains no audio clips to transcribe."

        logging.info("Rendering timeline audio for transcription...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_audio_file:
            tmp_audio_path = tmp_audio_file.name
            
            self._render_timeline_audio(state, tmp_audio_path)
            
            # --- FIX: Proactively check file size for timeline audio as well ---
            file_size = os.path.getsize(tmp_audio_path)
            logging.info(f"Timeline audio rendered. Size: {file_size / (1024*1024):.2f} MB")
            if file_size > WHISPER_API_LIMIT_BYTES:
                return (
                    f"Error: The rendered timeline audio is too large ({file_size / (1024*1024):.2f} MB) "
                    f"for the transcription API (limit is 25 MB). "
                    f"Please shorten the timeline or remove audio clips before trying again."
                )
            # --- END FIX ---
            
            logging.info("Transcribing rendered timeline audio...")
            with open(tmp_audio_path, "rb") as audio_file_handle:
                whisper_result = self._run_whisper(audio_file_handle, args, client)
            
            return self._format_transcription(whisper_result, args.granularity, "Transcription for Timeline")

    def _render_timeline_audio(self, state: 'State', output_path: str):
        """Renders all audio clips on the timeline into a single audio file using ffmpeg."""
        audio_clips = [c for c in state.timeline if c.track_type == 'audio' and c.has_audio]
        if not audio_clips:
            raise ValueError("No audio clips with audio streams found on the timeline.")

        input_streams = []
        for clip in audio_clips:
            stream = ffmpeg.input(clip.source_path, ss=clip.source_in_sec, t=clip.duration_sec).audio
            delayed_stream = stream.filter('adelay', f"{int(clip.timeline_start_sec * 1000)}|{int(clip.timeline_start_sec * 1000)}")
            input_streams.append(delayed_stream)

        mixed_audio = ffmpeg.filter(input_streams, 'amix', inputs=len(input_streams), dropout_transition=0)
        
        (
            mixed_audio
            .output(output_path, acodec='pcm_s16le', ar='16000', ac=1)
            .run(overwrite_output=True, quiet=True)
        )

    def _run_whisper(self, audio_file_handle, args: TranscribeMediaArgs, client: openai.OpenAI) -> Dict[str, Any]:
        """Calls the Whisper API and returns the verbose JSON result."""
        logging.info("Sending audio to OpenAI Whisper API...")
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_handle,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
            language=args.language,
            prompt=args.prompt
        )
        logging.info("Received transcription from Whisper API.")
        return response.model_dump()

    def _format_transcription(self, result: Dict[str, Any], granularity: str, header: str) -> str:
        """Formats the Whisper JSON result into a readable string for the agent."""
        output = [f"{header}:", "---"]

        if granularity == "segment":
            if not result.get('segments'):
                return f"{header}\n---\n(No speech detected)"
            for segment in result['segments']:
                start = seconds_to_hms(segment['start'])
                end = seconds_to_hms(segment['end'])
                text = segment['text'].strip()
                output.append(f"[{start} -> {end}] {text}")
        
        elif granularity == "word":
            if not result.get('words'):
                return f"{header}\n---\n(No speech detected or word-level timestamps not available)"
            
            all_words = []
            for word_info in result['words']:
                start = seconds_to_hms(word_info['start'])
                word = word_info['word']
                all_words.append(f"[{start}] {word}")
            output.append(" ".join(all_words))

        return "\n".join(output)