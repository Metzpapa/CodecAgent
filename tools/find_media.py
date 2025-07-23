# codec/tools/find_media.py

import os
import json
import tempfile
import ffmpeg
import yt_dlp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional, Tuple, TYPE_CHECKING, Union, List, Dict, Any, Annotated

from pydantic import BaseModel, Field

from .base import BaseTool
from llm.types import ContentPart, FileObject
from utils import hms_to_seconds

if TYPE_CHECKING:
    from state import State
    from llm.base import LLMConnector

# --- Pydantic Models ---

class FindMediaArgs(BaseModel):
    """Arguments for the find_media tool."""

    query: str = Field(
        ...,
        description="A search query (e.g., 'scenic japanese garden') or a direct URL to a video."
    )
    mode: Literal["download", "search_only", "preview"] = Field(
        "preview",
        description=(
            "'download': Immediately downloads the first search result or the specified URL. "
            "'search_only': Returns a text-only list of search results with metadata. "
            "'preview': Returns a rich visual preview of the top search results, including sample frames."
        )
    )
    search_limit: int = Field(
        3,
        description="For 'search_only' and 'preview' modes, the number of top results to return.",
        gt=0,
        le=10
    )
    num_preview_frames: int = Field(
        3,
        description="For 'preview' mode, the number of sample frames to extract from each video result.",
        gt=0,
        le=5
    )
    media_type: Literal["video", "audio"] = Field(
        "video",
        description="Specify whether to acquire a 'video' file or extract 'audio' only. This is applied during the 'download' operation."
    )
    output_filename: Optional[str] = Field(
        None,
        description="A specific name for the downloaded file. If omitted, a name is generated from the video's title. Applied during 'download'."
    )
    quality: Literal["best", "1080p", "720p"] = Field(
        "1080p",
        description="The desired maximum video quality for download. 'best' will get the highest available. Ignored for audio-only."
    )
    # --- FIX: Changed type hint from Tuple to an Annotated List ---
    # This generates a JSON schema that is compliant with the OpenAI API,
    # resolving the "array schema missing items" error.
    download_range: Optional[Annotated[List[str], Field(min_length=2, max_length=2)]] = Field(
        None,
        description="Optional. Download only a specific segment. Provide a list of [start_time, end_time] in 'HH:MM:SS' format. E.g., ['00:01:30', '00:01:45']."
    )

# --- Tool Implementation ---

class FindMediaTool(BaseTool):
    """
    A tool to find, preview, or download video and audio from online sources like YouTube.
    """

    @property
    def name(self) -> str:
        return "find_media"

    @property
    def description(self) -> str:
        return (
            "Finds, previews, or downloads video and audio from online sources like YouTube. This is the primary tool for acquiring new assets. "
            "It can operate in three modes: 'download' (to immediately get a file), 'search_only' (to get a text-based list of results for evaluation), "
            "and 'preview' (to get a rich visual preview with sample frames from the top search results). "
            "The 'preview' mode is the most powerful way to make an informed choice before downloading."
        )

    @property
    def args_schema(self):
        return FindMediaArgs

    def execute(self, state: 'State', args: FindMediaArgs, connector: 'LLMConnector') -> Union[str, Tuple[str, List[ContentPart]]]:
        try:
            if args.mode == "download":
                return self._execute_download(state, args)
            elif args.mode == "search_only":
                return self._execute_search_only(args)
            elif args.mode == "preview":
                return self._execute_preview(args, connector, state)
            else:
                return f"Error: Unknown mode '{args.mode}'."
        except Exception as e:
            print(f"ERROR in find_media tool: {e}")
            import traceback
            traceback.print_exc()
            return f"An unexpected error occurred in the find_media tool: {e}"

    # --- Mode-Specific Execution Helpers ---

    def _execute_download(self, state: 'State', args: FindMediaArgs) -> str:
        print(f"Attempting to download media for query: '{args.query}'")
        
        # 1. Configure yt-dlp options
        output_template = os.path.join(state.assets_directory, args.output_filename or '%(title)s.%(ext)s')
        
        ydl_opts = {
            'outtmpl': output_template,
            'quiet': True,
            'noplaylist': True,
            'default_search': 'ytsearch1', # Download the first result if it's a search
        }

        if args.media_type == 'audio':
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        else: # video
            quality_map = {
                "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
                "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
                "best": "bestvideo+bestaudio/best"
            }
            ydl_opts['format'] = quality_map[args.quality]
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }]

        if args.download_range:
            start_sec = hms_to_seconds(f"{args.download_range[0]}.000")
            end_sec = hms_to_seconds(f"{args.download_range[1]}.000")
            ydl_opts['download_ranges'] = yt_dlp.utils.download_range_func(None, [(start_sec, end_sec)])
            # When downloading a range, yt-dlp needs a specific postprocessor
            ydl_opts.setdefault('postprocessors', []).append({
                'key': 'FFmpegVideoRemuxer',
                'preferedformat': 'mp4'
            })


        # 2. Execute download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(args.query, download=True)
            # If it was a search, info is a playlist dict
            video_info = info['entries'][0] if 'entries' in info else info
            
            # yt-dlp might change the extension, so we need the final path
            final_filepath = ydl.prepare_filename(video_info)
            final_filename = os.path.basename(final_filepath)

        return f"Successfully downloaded '{final_filename}' and added it to the asset library."

    def _execute_search_only(self, args: FindMediaArgs) -> str:
        print(f"Performing search_only for query: '{args.query}'")
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': 'in_playlist', # Faster, gets metadata without deep dive
            'default_search': f"ytsearch{args.search_limit}",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(args.query, download=False)
        
        if not result or 'entries' not in result:
            return "No search results found."

        search_results = []
        for entry in result['entries']:
            if not entry: continue
            search_results.append({
                "title": entry.get('title', 'N/A'),
                "url": entry.get('url', 'N/A'),
                "thumbnail_url": entry.get('thumbnail'),
                "duration": entry.get('duration_string', 'N/A'),
                "channel": entry.get('channel', 'N/A'),
                "view_count": entry.get('view_count'),
                "description_snippet": (entry.get('description') or '')[:150] + '...'
            })
        
        return json.dumps(search_results, indent=2)

    def _execute_preview(self, args: FindMediaArgs, connector: 'LLMConnector', state: 'State') -> Tuple[str, List[ContentPart]]:
        print(f"Generating preview for query: '{args.query}'")
        
        # 1. Get search results metadata (same as search_only)
        ydl_opts = {
            'quiet': True,
            'extract_flat': 'in_playlist',
            'default_search': f"ytsearch{args.search_limit}",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_info = ydl.extract_info(args.query, download=False)

        if not search_info or 'entries' not in search_info:
            return ("No search results found.", [])

        # 2. Create frame extraction jobs
        frame_jobs = []
        results_with_jobs = []
        for i, entry in enumerate(search_info['entries']):
            if not entry or not entry.get('duration'): continue
            
            duration_sec = entry['duration']
            timestamps = self._calculate_timestamps(duration_sec, args.num_preview_frames)
            
            job_ids_for_this_result = []
            for j, ts in enumerate(timestamps):
                job_id = f"result_{i}_frame_{j}"
                frame_jobs.append({
                    "id": job_id,
                    "video_url": entry['url'],
                    "timestamp_sec": ts,
                    "display_name": f"preview_{i+1}_{ts:.1f}s.jpg"
                })
                job_ids_for_this_result.append(job_id)
            
            results_with_jobs.append({
                "metadata": entry,
                "job_ids": job_ids_for_this_result
            })

        # 3. Execute jobs in parallel
        uploaded_frames: Dict[str, Union[FileObject, str]] = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_job = {
                    executor.submit(self._extract_and_upload_frame_from_url, job, connector, tmpdir): job
                    for job in frame_jobs
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        uploaded_frames[job['id']] = future.result()
                    except Exception as e:
                        uploaded_frames[job['id']] = f"System error during frame processing: {e}"

        # 4. Assemble the multimodal response
        multimodal_parts: List[ContentPart] = []
        for i, result_data in enumerate(results_with_jobs):
            entry = result_data['metadata']
            # Add text header for the result
            header_text = (
                f"--- Preview for Result {i+1} ---\n"
                f"Title: {entry.get('title', 'N/A')}\n"
                f"URL: {entry.get('url', 'N/A')}\n"
                f"Duration: {entry.get('duration_string', 'N/A')}\n"
                f"Channel: {entry.get('channel', 'N/A')}"
            )
            multimodal_parts.append(ContentPart(type='text', text=header_text))

            # Add the corresponding frames
            for job_id in result_data['job_ids']:
                result = uploaded_frames.get(job_id)
                if isinstance(result, FileObject):
                    state.uploaded_files.append(result)
                    multimodal_parts.append(ContentPart(type='image', file=result))
                else: # It's an error string
                    error_text = f"SYSTEM: Could not generate preview frame. Error: {result or 'Unknown'}"
                    multimodal_parts.append(ContentPart(type='text', text=error_text))

        confirmation_text = f"Successfully generated a visual preview for {len(results_with_jobs)} search results. The following content contains the previews."
        return (confirmation_text, multimodal_parts)

    # --- Private Helpers ---

    def _calculate_timestamps(self, duration_sec: float, num_frames: int) -> List[float]:
        """Calculates evenly spaced timestamps within a duration."""
        if num_frames <= 0:
            return []
        if num_frames == 1:
            return [duration_sec / 2]
        
        # Inset the first and last frames to avoid black screens at start/end
        start_offset = duration_sec * 0.05
        end_offset = duration_sec * 0.95
        effective_duration = end_offset - start_offset
        
        interval = effective_duration / (num_frames - 1) if (num_frames - 1) > 0 else effective_duration
        return [start_offset + i * interval for i in range(num_frames)]

    def _extract_and_upload_frame_from_url(
        self,
        job: Dict[str, Any],
        connector: 'LLMConnector',
        tmpdir: str
    ) -> Union[FileObject, str]:
        """Worker function to extract a frame from a URL and upload it."""
        output_path = Path(tmpdir) / f"{job['id']}.jpg"
        try:
            # Use ffmpeg to extract a frame directly from the video URL
            (
                ffmpeg.input(job['video_url'], ss=job['timestamp_sec'])
                .output(str(output_path), vframes=1, format='image2', vcodec='mjpeg')
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )

            # Upload the extracted frame
            print(f"Uploading frame: {job['display_name']}")
            uploaded_file_obj = connector.upload_file(
                file_path=str(output_path),
                mime_type="image/jpeg",
                display_name=job['display_name']
            )
            return uploaded_file_obj

        except ffmpeg.Error as e:
            error_msg = f"FFmpeg failed to extract frame. Stderr: {e.stderr.decode()}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Failed to extract or upload frame. Details: {e}"
            print(error_msg)
            return error_msg