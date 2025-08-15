# codec/tools/find_media.py

import os
import json
import tempfile
import ffmpeg
import yt_dlp
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional, Tuple, TYPE_CHECKING, Union, List, Dict, Any, Annotated

from pydantic import BaseModel, Field

from .base import BaseTool
import openai
from ..utils import hms_to_seconds

if TYPE_CHECKING:
    from ..state import State

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

    def execute(self, state: 'State', args: FindMediaArgs, client: openai.OpenAI) -> str:
        try:
            if args.mode == "download":
                return self._execute_download(state, args)
            elif args.mode == "search_only":
                return self._execute_search_only(args)
            elif args.mode == "preview":
                return self._execute_preview(args, client, state)
            else:
                return f"Error: Unknown mode '{args.mode}'."
        except Exception as e:
            logging.error(f"ERROR in find_media tool: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return f"An unexpected error occurred in the find_media tool: {e}"

    # --- Mode-Specific Execution Helpers ---

    def _execute_download(self, state: 'State', args: FindMediaArgs) -> str:
        logging.info(f"Attempting to download media for query: '{args.query}'")
        
        # --- FIX 1: Prevent double-extension bug ---
        # If a filename is provided, strip its extension. yt-dlp's post-processors
        # will add the correct final extension (e.g., .mp3, .mp4).
        if args.output_filename:
            base_name = Path(args.output_filename).stem
            output_template = os.path.join(state.assets_directory, f'{base_name}.%(ext)s')
        else:
            output_template = os.path.join(state.assets_directory, '%(title)s.%(ext)s')

        ydl_opts = {
            'outtmpl': output_template,
            'quiet': True,
            'noplaylist': True,
            'default_search': 'ytsearch1',
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
            ydl_opts.setdefault('postprocessors', []).append({
                'key': 'FFmpegVideoRemuxer',
                'preferedformat': 'mp4'
            })

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(args.query, download=True)
            video_info = info['entries'][0] if 'entries' in info else info
            
            # --- FIX 2: Get the *actual* final filename for an accurate success message ---
            # After post-processing, yt-dlp stores the final path in the info dict.
            final_filepath = video_info.get('filepath')
            if not final_filepath:
                # Fallback in case the 'filepath' key isn't available
                final_filepath = ydl.prepare_filename(video_info)

            final_filename = os.path.basename(final_filepath)

        return f"Successfully downloaded '{final_filename}' and added it to the asset library."

    def _execute_search_only(self, args: FindMediaArgs) -> str:
        logging.info(f"Performing search_only for query: '{args.query}'")
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': 'in_playlist',
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

    def _execute_preview(self, args: FindMediaArgs, client: openai.OpenAI, state: 'State') -> str:
        logging.info(f"Generating preview for query: '{args.query}'")
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': 'in_playlist',
            'default_search': f"ytsearch{args.search_limit}",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_info = ydl.extract_info(args.query, download=False)

        if not search_info or 'entries' not in search_info:
            return ("No search results found.", [])

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

        uploaded_frames: Dict[str, str] = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_job = {
                    executor.submit(self._extract_and_upload_frame_from_url, job, client, tmpdir): job
                    for job in frame_jobs
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        uploaded_frames[job['id']] = future.result()
                    except Exception as e:
                        uploaded_frames[job['id']] = f"System error during frame processing: {e}"

        successful_frames = 0
        for i, result_data in enumerate(results_with_jobs):
            entry = result_data['metadata']
            logging.info(f"--- Preview for Result {i+1} ---")
            logging.info(f"Title: {entry.get('title', 'N/A')}")
            logging.info(f"URL: {entry.get('url', 'N/A')}")
            logging.info(f"Duration: {entry.get('duration_string', 'N/A')}")
            logging.info(f"Channel: {entry.get('channel', 'N/A')}")

            for job_id in result_data['job_ids']:
                result = uploaded_frames.get(job_id)
                if result and "System error" not in result:
                    file_id = result
                    state.uploaded_files.append(file_id)
                    state.new_file_ids_for_model.append(file_id)
                    successful_frames += 1
                else:
                    logging.warning(f"  - Failed to generate frame: {result or 'Unknown error'}")

        if successful_frames == 0:
            return "Error: Failed to generate any preview frames for the search results."

        return f"Successfully generated and uploaded {successful_frames} preview frames from {len(results_with_jobs)} search results. The agent can now view them."

    # --- Private Helpers ---

    def _calculate_timestamps(self, duration_sec: float, num_frames: int) -> List[float]:
        if num_frames <= 0:
            return []
        if num_frames == 1:
            return [duration_sec / 2]
        
        start_offset = duration_sec * 0.05
        end_offset = duration_sec * 0.95
        effective_duration = end_offset - start_offset
        
        interval = effective_duration / (num_frames - 1) if (num_frames - 1) > 0 else effective_duration
        return [start_offset + i * interval for i in range(num_frames)]

    def _extract_and_upload_frame_from_url(
        self,
        job: Dict[str, Any],
        client: openai.OpenAI,
        tmpdir: str
    ) -> str:
        output_path = Path(tmpdir) / f"{job['id']}.jpg"
        try:
            (
                ffmpeg.input(job['video_url'], ss=job['timestamp_sec'])
                .output(str(output_path), vframes=1, format='image2', vcodec='mjpeg')
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )

            logging.info(f"Uploading frame: {job['display_name']}")
            with open(output_path, "rb") as f:
                uploaded_file = client.files.create(file=f, purpose="vision")
            return uploaded_file.id

        except ffmpeg.Error as e:
            error_msg = f"FFmpeg failed to extract frame. Stderr: {e.stderr.decode()}"
            logging.error(error_msg)
            raise IOError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to extract or upload frame. Details: {e}"
            logging.error(error_msg)
            raise IOError(error_msg) from e