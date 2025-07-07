import os
import ffmpeg
from typing import Optional, TYPE_CHECKING, Union
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from .base import BaseTool

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State


class ViewVideoArgs(BaseModel):
    """Arguments for the view_video tool."""
    source_filename: str = Field(
        ...,
        description="The exact name of the video file from the user's media library to be viewed (e.g., 'interview.mp4')."
    )
    num_frames: int = Field(
        8,
        description="The total number of frames to extract for viewing. This controls the granularity of the preview.",
        gt=0
    )
    start_time: Optional[str] = Field(
        None,
        description="The timestamp in the source video to start extracting frames from. Format: HH:MM:SS.mmm. If omitted, starts from the beginning.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    end_time: Optional[str] = Field(
        None,
        description="The timestamp in a source video to stop extracting frames at. Format: HH:MM:SS.mmm. If omitted, uses the full video duration.",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )


class ViewVideoTool(BaseTool):
    """
    A tool to extract and "see" a specified number of evenly-spaced frames
    from a video file within a given time range.
    """

    @property
    def name(self) -> str:
        return "view_video"

    @property
    def description(self) -> str:
        return (
            "Extracts and displays a specified number of frames from a video file to 'see' its contents. "
            "You can specify a time range and the number of frames for granularity. If no time range is given, "
            "it will scan the entire video. Use this to get a visual overview, find specific scenes, or "
            "confirm the content of a clip before performing an edit."
        )

    @property
    def args_schema(self):
        return ViewVideoArgs

    def _hms_to_seconds(self, time_str: str) -> float:
        """Converts HH:MM:SS.mmm format to total seconds."""
        parts = time_str.split(':')
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split('.')
        s = int(s_parts[0])
        ms = int(s_parts[1].ljust(3, '0')) if len(s_parts) > 1 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0

    def _extract_and_upload_frame(
        self,
        ts: float,
        full_path: str,
        source_filename: str,
        client: 'genai.Client'
    ) -> Union[types.File, str]:
        """
        Extracts a single frame, uploads it, and returns the File object or an error string.
        This function is designed to be run in a separate thread.
        """
        try:
            # Step 1: Extract frame bytes
            print(f"Extracting frame from {ts:.3f}s...")
            out, err = (
                ffmpeg.input(full_path, ss=ts)
                .output('pipe:', vframes=1, format='image2', vcodec='mjpeg', strict='-2')
                .run(capture_stdout=True, capture_stderr=True)
            )
            if not out:
                raise ffmpeg.Error('ffmpeg', out, err)

            # Step 2: Upload the frame bytes
            print(f"Uploading frame from {ts:.3f}s...")
            file_obj = BytesIO(out)
            frame_file = client.files.upload(
                file=file_obj,
                config={
                    "mimeType": "image/jpeg",
                    "displayName": f"frame-{source_filename}-{ts:.2f}s"
                }
            )
            print(f"Upload complete for frame at {ts:.3f}s. Name: {frame_file.name}")
            return frame_file

        except ffmpeg.Error as e:
            error_details = e.stderr.decode().strip() if e.stderr else str(e)
            print(f"FFmpeg error for frame at {ts:.3f}s: {error_details}")
            return f"FFmpeg failed to extract frame. Details: {error_details}"
        except Exception as e:
            print(f"Upload error for frame at {ts:.3f}s: {e}")
            return f"Failed to upload frame. Details: {str(e)}"

    def execute(self, state: 'State', args: ViewVideoArgs, client: 'genai.Client') -> str | types.Content:
        # --- 1. Validation & Setup ---
        full_path = os.path.join(state.assets_directory, args.source_filename)
        if not os.path.exists(full_path):
            return f"Error: The source file '{args.source_filename}' does not exist in the assets directory."

        try:
            probe = ffmpeg.probe(full_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if not video_stream:
                return f"Error: Source file '{args.source_filename}' does not contain a video stream."
            
            duration_str = video_stream.get('duration') or probe['format'].get('duration', '0')
            source_duration = float(duration_str)

            if source_duration <= 0:
                return f"Error: Could not determine a valid duration for '{args.source_filename}'."
        except ffmpeg.Error as e:
            return f"Error: Failed to probe '{args.source_filename}'. It may be corrupt or not a valid video file. FFmpeg error: {e.stderr.decode()}"

        # --- 2. Time Range Calculation ---
        start_sec = self._hms_to_seconds(args.start_time) if args.start_time else 0.0
        end_sec = self._hms_to_seconds(args.end_time) if args.end_time else source_duration

        if start_sec >= end_sec:
            return "Error: The start_time must be before the end_time."
        
        if end_sec > source_duration:
            end_sec = source_duration

        safe_end_sec = min(end_sec, source_duration) - 0.05  # 50ms buffer so that ffmpeg doesn't fail at end of video
        duration_to_sample = safe_end_sec - start_sec

        if duration_to_sample <= 0:
            timestamps = [start_sec]
        else:
            segment_duration = duration_to_sample / args.num_frames
            timestamps = [
                start_sec + (i * segment_duration) + (segment_duration / 2)
                for i in range(args.num_frames)
            ]

        # --- 3. Parallel Frame Extraction & Upload ---
        context_text = (
            f"SYSTEM: This is the output of the `view_video` tool you called for '{args.source_filename}'. "
            f"Displaying up to {args.num_frames} frames sampled between {start_sec:.2f}s and {end_sec:.2f}s. "
            "Each image is a frame referenced by the timestamp noted in the accompanying text."
        )
        all_parts = [types.Part.from_text(text=context_text)]
        
        upload_results = []
        # Use a thread pool to upload frames in parallel. max_workers can be tuned.
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Map each timestamp to a future object
            future_to_ts = {
                executor.submit(
                    self._extract_and_upload_frame, ts, full_path, args.source_filename, client
                ): ts for ts in timestamps
            }

            # Process futures as they complete to gather results
            for future in as_completed(future_to_ts):
                ts = future_to_ts[future]
                try:
                    # The result is either a types.File or an error string
                    result = future.result()
                    upload_results.append((ts, result))
                except Exception as e:
                    # This catches errors not handled inside the worker function
                    print(f"Critical error for future of ts {ts:.3f}s: {e}")
                    upload_results.append((ts, f"An unexpected system error occurred: {e}"))

        # --- 4. Assemble Response from Sorted Results ---
        # Sort results by timestamp to maintain chronological order for the LLM
        upload_results.sort(key=lambda x: x[0])

        for ts, result in upload_results:
            if isinstance(result, types.File):
                frame_file = result
                # Store the file object in the state for later cleanup
                state.uploaded_files.append(frame_file)

                # Append the timestamp and the file URI part to the response
                all_parts.append(types.Part.from_text(text=f"Frame at: {ts:.3f}s"))
                all_parts.append(types.Part.from_uri(
                    file_uri=frame_file.uri,
                    mime_type='image/jpeg'
                ))
            else:
                # Result is an error string from the worker function
                error_details = result
                all_parts.append(types.Part.from_text(
                    text=f"SYSTEM: Could not process frame at {ts:.3f}s. Error: {error_details}"
                ))

        return types.Content(role="user", parts=all_parts)