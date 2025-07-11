# codec/tools/delete_clip.py

from typing import List, TYPE_CHECKING

from pydantic import BaseModel, Field

from google import genai
from .base import BaseTool

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from state import State


class DeleteClipsArgs(BaseModel):
    """Arguments for the delete_clips tool."""
    clip_ids: List[str] = Field(
        ...,
        description="A list of one or more unique clip identifiers to be deleted. These IDs must exactly match `clip_id`s from the `get_timeline_summary` tool."
    )
    ripple: bool = Field(
        False,
        description="If True, closes the gap left by the deleted clip by shifting all subsequent clips on the same track earlier. This is only supported when deleting a single clip."
    )


class DeleteClipTool(BaseTool):
    """A tool to delete one or more clips from the timeline."""

    @property
    def name(self) -> str:
        return "delete_clips"

    @property
    def description(self) -> str:
        return (
            "Deletes one or more clips from the timeline using their unique `clip_id`s. This action is permanent for the current session. "
            "By default, this leaves a gap. To delete a clip and have all subsequent clips on the same track move earlier to fill the gap, set `ripple` to True (Note: ripple delete is only supported when deleting a single clip). "
            "To find the `clip_id`s, you must first use the `get_timeline_summary` tool."
        )

    @property
    def args_schema(self):
        return DeleteClipsArgs

    def execute(self, state: 'State', args: DeleteClipsArgs, client: 'genai.Client') -> str:
        # --- 1. Input Validation ---
        if args.ripple and len(args.clip_ids) > 1:
            return "Error: Ripple delete is not supported when deleting multiple clips at once. Please provide only one clip_id when ripple is True."

        # --- 2. Ripple Delete Logic ---
        if args.ripple:
            clip_id_to_delete = args.clip_ids[0]
            clip_to_delete = state.find_clip_by_id(clip_id_to_delete)

            if not clip_to_delete:
                return f"Error: No clip with the ID '{clip_id_to_delete}' was found on the timeline."

            # Store properties needed for the ripple effect
            deleted_duration = clip_to_delete.duration_sec
            deleted_track = clip_to_delete.track_index
            deleted_start_time = clip_to_delete.timeline_start_sec

            # Perform the deletion
            state.delete_clip(clip_id_to_delete)

            # Apply the ripple effect to subsequent clips on the same track
            shifted_count = 0
            for clip in state.timeline:
                if clip.track_index == deleted_track and clip.timeline_start_sec > deleted_start_time:
                    clip.timeline_start_sec -= deleted_duration
                    shifted_count += 1
            
            # Re-sort the timeline to ensure order is maintained after shifts
            state._sort_timeline()

            return f"Successfully ripple-deleted clip '{clip_id_to_delete}', shifting {shifted_count} subsequent clips on track {deleted_track}."

        # --- 3. Standard (Batch) Delete Logic ---
        else:
            deleted_ids = []
            failed_ids = []

            for clip_id in args.clip_ids:
                if state.delete_clip(clip_id):
                    deleted_ids.append(clip_id)
                else:
                    failed_ids.append(clip_id)

            # --- 4. Construct Response ---
            if not deleted_ids and not failed_ids:
                return "Error: No clip IDs were provided."
            
            if failed_ids and not deleted_ids:
                return f"Error: Could not find any of the requested clips to delete. Failed IDs: {', '.join(failed_ids)}."
            
            if deleted_ids and not failed_ids:
                return f"Successfully deleted {len(deleted_ids)} clips: {', '.join(deleted_ids)}."
            
            # Mixed results
            return (
                f"Completed with mixed results. "
                f"Successfully deleted {len(deleted_ids)} clips: {', '.join(deleted_ids)}. "
                f"Failed to find {len(failed_ids)} clips: {', '.join(failed_ids)}."
            )