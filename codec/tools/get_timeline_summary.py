# codec/tools/get_timeline_summary.py

import os
from typing import Optional, TYPE_CHECKING, Tuple
import openai
from collections import defaultdict

from pydantic import BaseModel, Field

from .base import BaseTool
from ..utils import hms_to_seconds

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from ..state import State


class GetTimelineSummaryArgs(BaseModel):
    """Arguments for the get_timeline_summary tool."""
    track: Optional[str] = Field(
        None,
        description="Optional. If provided, the summary will only show clips on this specific track (e.g., 'V1', 'A2').",
        pattern=r"^[VAva]\d+$"
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

    def execute(self, state: 'State', args: GetTimelineSummaryArgs, client: openai.OpenAI) -> str:
        if not state.timeline:
            return "Timeline is currently empty."

        # --- 1. Parse and Apply Filters ---
        start_sec = hms_to_seconds(args.start_time) if args.start_time else None
        end_sec = hms_to_seconds(args.end_time) if args.end_time else None
        
        parsed_track_type = None
        parsed_track_number = None
        if args.track:
            parsed_track_type = 'video' if args.track[0].lower() == 'v' else 'audio'
            parsed_track_number = int(args.track[1:])

        is_filtered = any([args.track, start_sec is not None, end_sec is not None])

        clips_to_display = state.timeline
        if parsed_track_type and parsed_track_number:
            clips_to_display = [c for c in clips_to_display if c.track_type == parsed_track_type and c.track_number == parsed_track_number]
        if start_sec is not None:
            clips_to_display = [c for c in clips_to_display if c.timeline_start_sec >= start_sec]
        if end_sec is not None:
            clips_to_display = [c for c in clips_to_display if (c.timeline_start_sec + c.duration_sec) <= end_sec]

        clips_by_track = defaultdict(list)
        for clip in clips_to_display:
            clips_by_track[(clip.track_type, clip.track_number)].append(clip)

        # --- 2. Build Header ---
        output = []
        header = "TIMELINE SUMMARY (FILTERED)" if is_filtered else "TIMELINE SUMMARY"
        output.append("=" * 40)
        output.append(f"{header:^40}")
        output.append("=" * 40)

        total_duration = state.get_timeline_duration()
        fps, width, height = state.get_sequence_properties() # Use the new state method
        all_tracks = set((c.track_type, c.track_number) for c in state.timeline)
        num_video_tracks = len({t for t in all_tracks if t[0] == 'video'})
        num_audio_tracks = len({t for t in all_tracks if t[0] == 'audio'})

        output.append(f"Total Duration: {total_duration:.3f}s")
        output.append(f"Sequence: {width}x{height} @ {fps:.2f}fps")
        output.append(f"Tracks: {num_video_tracks} Video, {num_audio_tracks} Audio")
        output.append(f"Total Clips: {len(state.timeline)}")

        if is_filtered:
            filter_lines = []
            if args.track:
                filter_lines.append(f"Track: {args.track.upper()}")
            if start_sec is not None or end_sec is not None:
                start_str = f"{start_sec:.3f}s" if start_sec is not None else "start"
                end_str = f"{end_sec:.3f}s" if end_sec is not None else "end"
                filter_lines.append(f"Time Range: {start_str} -> {end_str}")
            output.append(f"Filters Applied: {', '.join(filter_lines)}")
        
        output.append("-" * 40)

        # --- 3. Build Track and Clip Details ---
        if parsed_track_type and parsed_track_number:
            tracks_to_iterate = [(parsed_track_type, parsed_track_number)]
        else:
            # Sort V tracks, then A tracks, both numerically
            tracks_to_iterate = sorted(list(all_tracks), key=lambda t: (t[0], t[1]))

        if not tracks_to_iterate:
            output.append("No tracks found.")

        for track_type, track_number in tracks_to_iterate:
            track_name = f"{track_type[0].upper()}{track_number}"
            output.append(f"\n--- Track {track_name} ---")
            
            track_clips = clips_by_track.get((track_type, track_number), [])
            
            if not track_clips:
                output.append("  (No clips on this track match the specified filters)")
                continue

            last_clip_end_time = start_sec if start_sec is not None else 0.0
            
            for clip in track_clips:
                gap_duration = clip.timeline_start_sec - last_clip_end_time
                if gap_duration > 0.001:
                    output.append(f"\n  [GAP from {last_clip_end_time:.3f}s to {clip.timeline_start_sec:.3f}s (duration: {gap_duration:.3f}s)]")

                if clip.timeline_start_sec < last_clip_end_time:
                    output.append(f"\n  [!!! WARNING: OVERLAP DETECTED with previous clip !!!]")
                
                clip_end_time = clip.timeline_start_sec + clip.duration_sec
                clip_info = [
                    f"\n  - Clip ID: {clip.clip_id}",
                    f"    Timeline: {clip.timeline_start_sec:.3f}s -> {clip_end_time:.3f}s (Duration: {clip.duration_sec:.3f}s)",
                    f"    Source: {os.path.basename(clip.source_path)} ({'Video' if clip.track_type == 'video' else 'Audio'})",
                    f"    Source In/Out: {clip.source_in_sec:.3f}s -> {clip.source_out_sec:.3f}s",
                    f"    Description: {clip.description or 'N/A'}"
                ]
                output.extend(clip_info)
                last_clip_end_time = clip_end_time

        return "\n".join(output)