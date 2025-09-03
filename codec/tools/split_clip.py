# codec/tools/split_clip.py

import os
import math
from typing import TYPE_CHECKING
import openai
from pydantic import BaseModel, Field

from .base import BaseTool
from ..state import TimelineClip
from ..utils import hms_to_seconds, seconds_to_hms

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from ..state import State


class SplitClipArgs(BaseModel):
    """Arguments for the split_clip tool."""
    clip_id: str = Field(
        ...,
        description="The unique identifier of the single clip to be split. This ID must exactly match a `clip_id` from the `get_timeline_summary` tool."
    )
    split_time: str = Field(
        ...,
        description="The timeline timestamp where the cut should be made. This time must be strictly within the target clip's duration on the timeline. Format: HH:MM:SS.mmm",
        pattern=r'^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$'
    )


class SplitClipTool(BaseTool):
    """
    A tool to split a single clip on the timeline at a precise point in time.
    This is a surgical operation that creates a new cut point.
    """

    @property
    def name(self) -> str:
        return "split_clip"

    @property
    def description(self) -> str:
        return (
            "Splits a single clip on the timeline at a specific point in time. "
            "This removes the original clip and replaces it with two new, smaller clips (e.g., 'my_clip' becomes 'my_clip_p1' and 'my_clip_p2'). "
            "Use this to create a new cut point on the timeline before an 'insert' operation."
        )

    @property
    def args_schema(self):
        return SplitClipArgs

    # REMOVED: The local _hms_to_seconds method is no longer needed.

    def execute(self, state: 'State', args: SplitClipArgs, client: openai.OpenAI, tmpdir: str) -> str:
        # --- 1. Find the target clip and convert time ---
        original_clip = state.find_clip_by_id(args.clip_id)
        if not original_clip:
            return f"Error: Clip with ID '{args.clip_id}' not found on the timeline."

        # Use the imported helper function directly
        split_time_sec = hms_to_seconds(args.split_time)

        # --- 2. Perform critical validation ---
        clip_start_sec = original_clip.timeline_start_sec
        clip_end_sec = clip_start_sec + original_clip.duration_sec
        
        # A split must be *strictly* within the clip's bounds.
        # We use a tiny float tolerance to avoid precision issues at the boundaries.
        if not (clip_start_sec < split_time_sec < clip_end_sec):
            return (
                f"Error: The split time {seconds_to_hms(split_time_sec)} is not within the timeline range of clip '{args.clip_id}' "
                f"(from {seconds_to_hms(clip_start_sec)} to {seconds_to_hms(clip_end_sec)}). "
                f"Please provide a time that is strictly between the clip's start and end."
            )

        # --- 3. Calculate properties for the new clips ---
        
        # Part 1 (from original start to split time)
        p1_duration = split_time_sec - clip_start_sec
        p1_data = original_clip.model_copy(deep=True) # Create a deep copy to modify
        p1_data.clip_id = f"{original_clip.clip_id}_p1"
        p1_data.duration_sec = p1_duration
        p1_data.source_out_sec = original_clip.source_in_sec + p1_duration
        p1_data.description = f"Part 1 of split from '{original_clip.clip_id}'"

        # Part 2 (from split time to original end)
        p2_duration = clip_end_sec - split_time_sec
        p2_data = original_clip.model_copy(deep=True)
        p2_data.clip_id = f"{original_clip.clip_id}_p2"
        p2_data.timeline_start_sec = split_time_sec
        p2_data.duration_sec = p2_duration
        p2_data.source_in_sec = p1_data.source_out_sec
        p2_data.description = f"Part 2 of split from '{original_clip.clip_id}'"
        
        # Check for ID collisions before committing changes
        if state.clip_id_exists(p1_data.clip_id) or state.clip_id_exists(p2_data.clip_id):
            return f"Error: The generated clip IDs '{p1_data.clip_id}' or '{p2_data.clip_id}' already exist. Please rename the original clip ('{args.clip_id}') before splitting."

        # --- 4. Atomically update the timeline state ---
        
        # First, remove the original clip
        state.delete_clip(args.clip_id)
        
        # Then, add the two new clips
        clip_p1 = TimelineClip(**p1_data.model_dump())
        clip_p2 = TimelineClip(**p2_data.model_dump())
        state.add_clip(clip_p1)
        state.add_clip(clip_p2)
        
        # --- 5. Return the "Golden" success message ---
        return (
            f"Success: Split clip '{args.clip_id}' at {seconds_to_hms(split_time_sec)}. "
            f"The original clip has been replaced by two new clips: '{p1_data.clip_id}' and '{p2_data.clip_id}'."
        )