
# codec/tools/get_timeline_summary.py

import os
from typing import Optional, TYPE_CHECKING, Tuple
from collections import defaultdict

from pydantic import BaseModel, Field
from google import genai

from .base import BaseTool

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State


class GetTimelineSummaryArgs(BaseModel):
    """Arguments for the get_timeline_summary tool."""
    track_index: Optional[int] = Field(
        None,
        description="Optional. If provided, the summary will only show clips on this specific track index (e.g., 0 for the first track)."
    )
    start_time: Optional[str] = Field(
        None,
        description="Optional. Filters the summary to only show clips that start at or after this timeline timestamp. Format: HH:MM:SS.mmm",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )
    end_time: Optional[str] = Field(
        None,
        description="Optional. Filters the summary to only show clips that end at or before this timeline timestamp. Format: HH:MM:SS.mmm",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )


class GetTimelineSummaryTool(BaseTool):
    """
    A tool to get a detailed, text-based summary of the current editing timeline.
    """

    @property
    def name(self) -> str:
        return "get_timeline_summary"

    @property
    def description(self) -> str:
        return "Provides a detailed text-based summary of the current editing timeline. By default, it shows all clips on all tracks. Use the optional parameters to filter the summary to a specific track or time range."

    @property
    def args_schema(self):
        return GetTimelineSummaryArgs

    def _hms_to_seconds(self, time_str: str) -> float:
        """Converts HH:MM:SS.mmm format to total seconds."""
        parts = time_str.split(':')
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split('.')
        s = int(s_parts[0])
        ms = int(s_parts[1].ljust(3, '0')) if len(s_parts) > 1 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0

    def _get_or_infer_sequence_properties(self, state: 'State') -> Tuple[float, int, int]:
        """Gets sequence properties from state or infers them from the first video clip."""
        if all([state.frame_rate, state.width, state.height]):
            return (state.frame_rate, state.width, state.height)
        
        # Find the first clip with valid video properties to infer from
        first_video_clip = next((c for c in state.timeline if c.source_width > 0 and c.source_height > 0 and c.source_frame_rate > 0), None)
        
        if first_video_clip:
            return (first_video_clip.source_frame_rate, first_video_clip.source_width, first_video_clip.source_height)
        else:
            # Return sensible defaults if no video clips are on the timeline
            return (24.0, 1920, 1080)

    def execute(self, state: 'State', args: GetTimelineSummaryArgs, client: 'genai.Client') -> str:
        if not state.timeline:
            return "Timeline is currently empty."

        # --- 1. Parse and Apply Filters ---
        start_sec = self._hms_to_seconds(args.start_time) if args.start_time else None
        end_sec = self._hms_to_seconds(args.end_time) if args.end_time else None
        is_filtered = any([args.track_index is not None, start_sec is not None, end_sec is not None])

        clips_to_display = state.timeline
        if args.track_index is not None:
            clips_to_display = [c for c in clips_to_display if c.track_index == args.track_index]
        if start_sec is not None:
            clips_to_display = [c for c in clips_to_display if c.timeline_start_sec >= start_sec]
        if end_sec is not None:
            clips_to_display = [c for c in clips_to_display if (c.timeline_start_sec + c.duration_sec) <= end_sec]

        clips_by_track = defaultdict(list)
        for clip in clips_to_display:
            clips_by_track[clip.track_index].append(clip)

        # --- 2. Build Header ---
        output = []
        header = "TIMELINE SUMMARY (FILTERED)" if is_filtered else "TIMELINE SUMMARY"
        output.append("=" * 40)
        output.append(f"{header:^40}")
        output.append("=" * 40)

        total_duration = state.get_timeline_duration()
        fps, width, height = self._get_or_infer_sequence_properties(state)
        output.append(f"Total Duration: {total_duration:.3f}s")
        output.append(f"Sequence: {width}x{height} @ {fps:.2f}fps")
        output.append(f"Total Tracks: {len(set(c.track_index for c in state.timeline)) if state.timeline else 0}")
        output.append(f"Total Clips: {len(state.timeline)}")

        if is_filtered:
            filter_lines = []
            if args.track_index is not None:
                filter_lines.append(f"Track: {args.track_index}")
            if start_sec is not None or end_sec is not None:
                start_str = f"{start_sec:.3f}s" if start_sec is not None else "start"
                end_str = f"{end_sec:.3f}s" if end_sec is not None else "end"
                filter_lines.append(f"Time Range: {start_str} -> {end_str}")
            output.append(f"Filters Applied: {', '.join(filter_lines)}")
        
        output.append("-" * 40)

        # --- 3. Build Track and Clip Details ---
        track_indices_to_iterate = [args.track_index] if args.track_index is not None else sorted(list(set(c.track_index for c in state.timeline)))

        if not track_indices_to_iterate:
            output.append("No tracks found.")

        for track_index in track_indices_to_iterate:
            output.append(f"\n--- Track {track_index} (V{track_index+1}/A{track_index+1}) ---")
            
            track_clips = clips_by_track.get(track_index, [])
            
            if not track_clips:
                output.append("  (No clips on this track match the specified filters)")
                continue

            last_clip_end_time = start_sec if start_sec is not None else 0.0
            
            for clip in track_clips:
                gap_duration = clip.timeline_start_sec - last_clip_end_time
                if gap_duration > 0.001: # Use a small tolerance for floating point
                    output.append(f"\n  [GAP from {last_clip_end_time:.3f}s to {clip.timeline_start_sec:.3f}s (duration: {gap_duration:.3f}s)]")

                # This check acts as a safety net to report data integrity issues.
                if clip.timeline_start_sec < last_clip_end_time:
                    output.append(f"\n  [!!! WARNING: OVERLAP DETECTED with previous clip !!!]")
                clip_end_time = clip.timeline_start_sec + clip.duration_sec
                clip_info = [
                    f"\n  - Clip ID: {clip.clip_id}",
                    f"    Timeline: {clip.timeline_start_sec:.3f}s -> {clip_end_time:.3f}s (Duration: {clip.duration_sec:.3f}s)",
                    f"    Source: {os.path.basename(clip.source_path)}",
                    f"    Description: {clip.description or 'N/A'}",
                    f"    Source In/Out: {clip.source_in_sec:.3f}s -> {clip.source_out_sec:.3f}s"
                ]
                output.extend(clip_info)
                last_clip_end_time = clip_end_time

        return "\n".join(output)