# codec/tools/extract_timeline_audio.py

import os
from typing import Optional, TYPE_CHECKING, Union, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from pathlib import Path

import ffmpeg
from pydantic import BaseModel, Field

from .base import BaseTool
from state import TimelineClip
from .extract_audio import _extract_and_upload_audio_segment # REUSING THE HELPER
from utils import hms_to_seconds
from llm.types import Message, ContentPart, FileObject

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State
    from llm.base import LLMConnector


class ExtractTimelineAudioArgs(BaseModel):
    """Arguments for the extract_timeline_audio tool."""
    start_time: Optional[str] = Field(
        None,
        description="The timestamp on the main timeline to start extracting audio from. Format: HH:MM:SS.mmm. If omitted, starts from the beginning of the timeline.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    end_time: Optional[str] = Field(
        None,
        description="The timestamp on the main timeline to stop extracting audio at. Format: HH:MM:SS.mmm. If omitted, uses the full timeline duration.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )


class ExtractTimelineAudioTool(BaseTool):
    """
    A tool to extract the audio from all clips on audio tracks within a given
    time range on the timeline. Use this to 'hear' the current edit.
    """

    @property
    def name(self) -> str:
        return "extract_timeline_audio"

    @property
    def description(self) -> str:
        return (
            "Extracts the audio from all clips on audio tracks (A1, A2, etc.) within a given time range on the timeline. "
            "This is used to 'hear' the current edit, mixing all audio sources. It returns separate audio segments for each clip found in the range. "
            "To hear a single source file, use 'extract_audio'."
        )

    @property
    def args_schema(self):
        return ExtractTimelineAudioArgs

    def execute(self, state: 'State', args: ExtractTimelineAudioArgs, connector: 'LLMConnector') -> Union[str, Tuple[str, List[ContentPart]]]:
        if not state.timeline:
            return "Error: The timeline is empty. Cannot extract audio from an empty timeline."

        # --- 1. Determine Time Range & Find Overlapping Audio Clips ---
        start_sec = hms_to_seconds(args.start_time) if args.start_time else 0.0
        end_sec = hms_to_seconds(args.end_time) if args.end_time else state.get_timeline_duration()

        if start_sec >= end_sec:
            return "Error: The start_time must be before the end_time."

        tasks_to_process = []
        for clip in state.timeline:
            if clip.track_type != 'audio' or not clip.has_audio:
                continue

            clip_end_sec = clip.timeline_start_sec + clip.duration_sec
            # Check for overlap
            if max(clip.timeline_start_sec, start_sec) < min(clip_end_sec, end_sec):
                # Calculate the exact segment of the clip within the requested range
                timeline_extract_start = max(clip.timeline_start_sec, start_sec)
                timeline_extract_end = min(clip_end_sec, end_sec)
                
                duration_to_extract = timeline_extract_end - timeline_extract_start
                if duration_to_extract <= 0:
                    continue

                source_start_offset = timeline_extract_start - clip.timeline_start_sec
                source_start_ts = clip.source_in_sec + source_start_offset

                tasks_to_process.append({
                    'clip': clip,
                    'source_start_ts': source_start_ts,
                    'duration': duration_to_extract
                })

        if not tasks_to_process:
            return f"No clips with audio were found on audio tracks in the specified timeline range ({start_sec:.2f}s to {end_sec:.2f}s)."

        # --- 2. Parallel Extraction and Upload using the shared helper ---
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Starting parallel audio extraction for {len(tasks_to_process)} clips from timeline...")
            
            upload_results = []
            with ThreadPoolExecutor(max_workers=16) as executor:
                future_to_clip = {
                    executor.submit(
                        _extract_and_upload_audio_segment,
                        task['clip'].source_path,
                        task['source_start_ts'],
                        task['duration'],
                        f"timeline-audio-{task['clip'].clip_id}",
                        connector,
                        tmpdir
                    ): task['clip']
                    for task in tasks_to_process
                }

                for future in as_completed(future_to_clip):
                    clip = future_to_clip[future]
                    try:
                        result = future.result()
                        upload_results.append((clip, result))
                    except Exception as e:
                        upload_results.append((clip, f"Unexpected system error: {e}"))

            # --- 3. MODIFIED: Assemble Response as a tuple ---
            upload_results.sort(key=lambda x: x[0].timeline_start_sec) # Sort by timeline start time
            
            confirmation_text = (
                f"Successfully extracted audio for {len(upload_results)} clips found on audio tracks between {start_sec:.2f}s and {end_sec:.2f}s of the timeline. "
                "The following content contains the audio information."
            )
            
            multimodal_parts: List[ContentPart] = []
            
            for clip, result in upload_results:
                track_name = f"A{clip.track_number}"
                if isinstance(result, FileObject):
                    audio_file = result
                    state.uploaded_files.append(audio_file)
                    multimodal_parts.append(ContentPart(type='text', text=f"Audio from clip: '{clip.clip_id}' on track {track_name} (starts on timeline at {clip.timeline_start_sec:.3f}s)"))
                    multimodal_parts.append(ContentPart(type='audio', file=audio_file))
                else: # It's an error string
                    error_details = result
                    multimodal_parts.append(ContentPart(
                        type='text',
                        text=f"SYSTEM: Could not process audio for clip '{clip.clip_id}' on track {track_name}. Error: {error_details}"
                    ))
            
            return (confirmation_text, multimodal_parts)