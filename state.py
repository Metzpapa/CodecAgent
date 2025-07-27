# codec/state.py
from typing import List, Optional, Literal, Tuple
from pydantic import BaseModel, Field

from llm.types import Message, FileObject


class TimelineClip(BaseModel):
    """
    Represents a single clip placed on the main timeline, analogous to a clip
    in a non-linear editor (NLE).
    (This class remains unchanged as it's part of the internal editing logic.)
    """
    clip_id: str
    source_path: str
    source_in_sec: float
    source_out_sec: float
    source_total_duration_sec: float
    timeline_start_sec: float
    duration_sec: float
    track_type: Literal['video', 'audio'] = Field(
        ...,
        description="The type of track the clip resides on ('video' or 'audio')."
    )
    track_number: int = Field(
        ...,
        ge=1,
        description="The 1-based index for the track (e.g., V1, A2)."
    )
    description: Optional[str] = Field(
        None,
        description="A user-provided description for the clip's purpose, for the agent to remember context."
    )
    source_frame_rate: float
    source_width: int
    source_height: int
    has_audio: bool = Field(
        ...,
        description="Flag indicating if the original source file for this clip contains an audio stream."
    )


class State:
    """
    Manages the state of the video editing agent session.

    This class acts as the agent's "memory," holding all contextual information
    and providing a clean API for timeline manipulation.
    """

    def __init__(self, assets_directory: str):
        self.assets_directory: str = assets_directory
        self.history: List[Message] = []
        self.timeline: List[TimelineClip] = []
        self.uploaded_files: List[FileObject] = []
        self.frame_rate: Optional[float] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.initial_prompt: Optional[str] = None
        # --- MODIFIED: Add a field to track the conversation for stateful APIs ---
        self.last_response_id: Optional[str] = None

    def _sort_timeline(self):
        """
        Internal helper to sort the timeline by track type (video then audio),
        then by track number, then by start time. This ensures a predictable
        and NLE-like order.
        """
        self.timeline.sort(key=lambda clip: (clip.track_type, clip.track_number, clip.timeline_start_sec))

    # --- Public Timeline Management API ---
    # (No changes needed in the methods below this line)

    def add_clip(self, clip: TimelineClip):
        """
        Adds a new clip to the timeline and re-sorts it.
        This is the primary method for adding clips.
        """
        self.timeline.append(clip)
        self._sort_timeline()

    def delete_clip(self, clip_id: str) -> bool:
        """
        Deletes a clip from the timeline by its unique ID.

        Returns:
            True if a clip was found and deleted, False otherwise.
        """
        clip_to_remove = self.find_clip_by_id(clip_id)
        if clip_to_remove:
            self.timeline.remove(clip_to_remove)
            return True
        return False

    # --- Public Timeline Query API ---

    def get_timeline_duration(self) -> float:
        """
        Calculates the total duration of the timeline by finding the
        end point of the last clip across all tracks.
        """
        if not self.timeline:
            return 0.0
        return max(
            (clip.timeline_start_sec + clip.duration_sec for clip in self.timeline),
            default=0.0
        )

    def get_specific_track_duration(self, track_type: str, track_number: int) -> float:
        """
        Calculates the total duration of a specific track (e.g., 'video', 1)
        by finding the end point of the last clip on that track.
        """
        clips_on_track = self.get_clips_on_specific_track(track_type, track_number)
        if not clips_on_track:
            return 0.0
        
        return max(
            (clip.timeline_start_sec + clip.duration_sec for clip in clips_on_track),
            default=0.0
        )

    def get_sequence_properties(self) -> Tuple[float, int, int]:
        """
        Gets sequence properties from state or infers them from the first video clip.
        It caches the result in the state object for subsequent calls.

        Returns:
            A tuple of (frame_rate, width, height).
        """
        # 1. Check if properties are already set (cached)
        if all([self.frame_rate, self.width, self.height]):
            return (self.frame_rate, self.width, self.height)

        # 2. Infer from the first clip on a video track
        first_video_clip = next((c for c in self.timeline if c.track_type == 'video'), None)
        
        if not first_video_clip:
            # 3. If no video, maybe there's only audio? Fallback to a default.
            return (24.0, 1920, 1080)
        
        # 4. Set the inferred properties on the state for future calls (caching)
        self.frame_rate = first_video_clip.source_frame_rate
        self.width = first_video_clip.source_width
        self.height = first_video_clip.source_height
        
        return (self.frame_rate, self.width, self.height)

    def find_clip_by_id(self, clip_id: str) -> Optional[TimelineClip]:
        """Finds a clip on the timeline by its unique clip_id."""
        return next((clip for clip in self.timeline if clip.clip_id == clip_id), None)

    def clip_id_exists(self, clip_id: str) -> bool:
        """Checks if a clip_id is already in use on the timeline."""
        return any(clip.clip_id == clip_id for clip in self.timeline)

    def get_clips_on_specific_track(self, track_type: str, track_number: int) -> List[TimelineClip]:
        """
        Returns a sorted list of all clips on a specific track (e.g., 'video', 1).
        """
        # The main timeline is already sorted, so we can just filter.
        return [
            clip for clip in self.timeline
            if clip.track_type == track_type and clip.track_number == track_number
        ]