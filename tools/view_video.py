
import os
import ffmpeg
from typing import Optional, TYPE_CHECKING, Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from pathlib import Path

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

    def _upload_frame_from_path(
        self,
        ts: float,
        file_path: Path,
        source_filename: str,
        client: 'genai.Client'
    ) -> Union[types.File, str]:
        """
        Uploads a single frame from a file path and returns the File object or an error string.
        This function is designed to be run in a separate thread for maximum I/O concurrency.
        """
        try:
            print(f"Uploading frame from {ts:.3f}s (path: {file_path.name})...")
            with open(file_path, "rb") as f:
                frame_file = client.files.upload(
                    file=f,
                    config={
                        "mimeType": "image/jpeg",
                        "displayName": f"frame-{source_filename}-{ts:.2f}s"
                    }
                )
            print(f"Upload complete for frame at {ts:.3f}s. Name: {frame_file.name}")
            return frame_file
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

            # Get precise frame rate for frame number calculation
            fr_str = video_stream.get('r_frame_rate', '0/1')
            num, den = map(int, fr_str.split('/'))
            if den == 0:
                return "Error: Could not determine a valid frame rate from the video file."
            frame_rate = num / den

            if source_duration <= 0:
                return f"Error: Could not determine a valid duration for '{args.source_filename}'."
        except ffmpeg.Error as e:
            return f"Error: Failed to probe '{args.source_filename}'. It may be corrupt or not a valid video file. FFmpeg error: {e.stderr.decode()}"

        # --- 2. Time & Frame Number Calculation ---
        start_sec = self._hms_to_seconds(args.start_time) if args.start_time else 0.0
        end_sec = self._hms_to_seconds(args.end_time) if args.end_time else source_duration

        if start_sec >= end_sec:
            return "Error: The start_time must be before the end_time."
        
        if end_sec > source_duration:
            end_sec = source_duration

        duration_to_sample = end_sec - start_sec

        if duration_to_sample <= 0:
            timestamps = [start_sec]
        else:
            # Ensure we don't sample too close to the very end
            safe_duration = duration_to_sample - (1 / frame_rate) 
            segment_duration = safe_duration / args.num_frames
            timestamps = [
                start_sec + (i * segment_duration) + (segment_duration / 2)
                for i in range(args.num_frames)
            ]

        # Convert timestamps to unique frame numbers for reliable extraction
        frame_numbers = sorted(list(set([int(round(ts * frame_rate)) for ts in timestamps])))
        
        if not frame_numbers:
            return "Could not calculate any valid frame numbers for extraction."

        # --- 3. Batch Extraction (Single FFmpeg process) & Parallel Upload ---
        print(f"Starting batch extraction of {len(frame_numbers)} frames using frame numbers...")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 3a: Construct and run a single ffmpeg command using the reliable 'select' filter with frame numbers
            select_filter = "+".join([f"eq(n,{fn})" for fn in frame_numbers])
            output_pattern = os.path.join(tmpdir, "frame_%04d.jpg")

            try:
                proc = (
                    ffmpeg.input(full_path)
                    .filter('select', select_filter)
                    .output(output_pattern, vsync='vfr', start_number=0)
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )
                out, err = proc.communicate()

                if proc.returncode != 0:
                    raise ffmpeg.Error('ffmpeg', out, err)

            except ffmpeg.Error as e:
                return f"Error during batch frame extraction. Details: {e.stderr.decode()}"

            # Step 3b: Get paths of extracted frames and map them back to original timestamps
            extracted_files = sorted(Path(tmpdir).glob("frame_*.jpg"))
            if len(extracted_files) != len(frame_numbers):
                return (f"Error: FFmpeg extracted {len(extracted_files)} frames, but {len(frame_numbers)} were requested. "
                        "This can happen with variable frame rate videos. "
                        f"FFmpeg stderr: {err.decode()}")

            # Map extracted files back to their original timestamps for context
            # We zip the sorted timestamps with the sorted files to ensure correct mapping
            tasks_to_upload = list(zip(sorted(timestamps), extracted_files))

            print(f"Batch extraction complete. Starting parallel upload of {len(extracted_files)} frames.")
            
            # Step 3c: Parallel Upload from the temporary files
            context_text = (
                f"SYSTEM: This is the output of the `view_video` tool you called for '{args.source_filename}'. "
                f"Displaying {len(extracted_files)} frames sampled between {start_sec:.2f}s and {end_sec:.2f}s. "
                "Each image is a frame referenced by the timestamp noted in the accompanying text."
            )
            all_parts = [types.Part.from_text(text=context_text)]
            
            upload_results = []
            with ThreadPoolExecutor(max_workers=16) as executor:
                future_to_ts = {
                    executor.submit(self._upload_frame_from_path, ts, path, args.source_filename, client): ts
                    for ts, path in tasks_to_upload
                }

                for future in as_completed(future_to_ts):
                    ts = future_to_ts[future]
                    try:
                        result = future.result()
                        upload_results.append((ts, result))
                    except Exception as e:
                        upload_results.append((ts, f"An unexpected system error during upload: {e}"))

            # --- 4. Assemble Response from Sorted Results ---
            upload_results.sort(key=lambda x: x[0])

            for ts, result in upload_results:
                if isinstance(result, types.File):
                    frame_file = result
                    state.uploaded_files.append(frame_file)
                    all_parts.append(types.Part.from_text(text=f"Frame at: {ts:.3f}s"))
                    all_parts.append(types.Part.from_uri(
                        file_uri=frame_file.uri,
                        mime_type='image/jpeg'
                    ))
                else:
                    error_details = result
                    all_parts.append(types.Part.from_text(
                        text=f"SYSTEM: Could not process frame at {ts:.3f}s. Error: {error_details}"
                    ))

            return types.Content(role="user", parts=all_parts)