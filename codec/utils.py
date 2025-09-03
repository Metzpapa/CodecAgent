# codec/utils.py
import ffmpeg
from pydantic import BaseModel, Field
from typing import Optional

class MediaInfo(BaseModel):
    """A structured model for holding media file metadata."""
    duration_sec: float = 0.0
    width: int = 0
    height: int = 0
    frame_rate: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    error: Optional[str] = Field(None, description="An error message if probing failed.")

def hms_to_seconds(time_str: str) -> float:
    """
    Converts a time string in HH:MM:SS.mmm format to total seconds.

    Args:
        time_str: The time string to convert.

    Returns:
        The total number of seconds as a float.
    """
    parts = time_str.split(':')
    h, m = int(parts[0]), int(parts[1])
    s_parts = parts[2].split('.')
    s = int(s_parts[0])
    ms = int(s_parts[1].ljust(3, '0')) if len(s_parts) > 1 else 0
    return h * 3600 + m * 60 + s + ms / 1000.0

def seconds_to_hms(seconds: float) -> str:
    """
    Converts total seconds into a standardized HH:MM:SS.mmm format string.
    This function is the single source of truth for time representation to the agent,
    ensuring millisecond precision across all tool outputs.

    Args:
        seconds: The total number of seconds as a float.

    Returns:
        A string formatted as HH:MM:SS.mmm.
    """
    if seconds < 0:
        seconds = 0
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds_rem = divmod(remainder, 60)
    seconds_int = int(seconds_rem)
    milliseconds = int((seconds_rem - seconds_int) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds_int:02d}.{milliseconds:03d}"

def probe_media_file(file_path: str) -> MediaInfo:
    """
    Probes a media file using ffmpeg and returns a structured MediaInfo object.
    This provides a safe and consistent way to get metadata from any media file.

    Args:
        file_path: The absolute path to the media file.

    Returns:
        A MediaInfo object containing the file's properties or an error message.
    """
    try:
        probe = ffmpeg.probe(file_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

        if not video_stream and not audio_stream:
            return MediaInfo(error="Not a valid media file (no video or audio streams).")

        # Use the duration from the most relevant stream, falling back to the format container.
        duration_str = (video_stream or audio_stream).get('duration') or probe['format'].get('duration', '0')
        
        info = MediaInfo(
            duration_sec=float(duration_str),
            has_video=video_stream is not None,
            has_audio=audio_stream is not None,
        )

        if video_stream:
            info.width = video_stream.get('width', 0)
            info.height = video_stream.get('height', 0)
            
            # Safely parse frame rate (it's often a fraction like '30/1')
            fr_str = video_stream.get('r_frame_rate', '0/1')
            num, den = map(int, fr_str.split('/'))
            info.frame_rate = num / den if den > 0 else 0.0
        
        return info

    except ffmpeg.Error as e:
        # Capture and decode the specific ffmpeg error for better debugging.
        error_details = e.stderr.decode('utf-8').strip()
        return MediaInfo(error=f"FFmpeg failed to probe file. It may be corrupt. Error: {error_details}")
    except Exception as e:
        return MediaInfo(error=f"An unexpected error occurred during probing: {e}")