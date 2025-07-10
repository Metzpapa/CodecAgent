from typing import List, Optional
from pydantic import BaseModel
from google.genai import types


class TimelineClip(BaseModel):
    """
    Represents a single clip placed on the main timeline.
    """
    clip_id: str
    source_path: str
    source_in_sec: float
    source_out_sec: float
    source_total_duration_sec: float
    timeline_start_sec: float
    duration_sec: float
    track_index: int
    description: Optional[str] = None
    source_frame_rate: float
    source_width: int
    source_height: int
    has_audio: bool # <-- ADDED: Flag to know if the source file has an audio track.


class State:
    """
    Manages the state of the video editing agent session.

    This class acts as the agent's "memory," holding all contextual information
    and providing a clean API for timeline manipulation.
    """

    def __init__(self, assets_directory: str):
        self.assets_directory: str = assets_directory
        self.history: List[types.Content] = []
        self.timeline: List[TimelineClip] = []
        self.uploaded_files: List[types.File] = []
        self.frame_rate: Optional[float] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None

    def _sort_timeline(self):
        """Internal helper to sort the timeline by track, then by start time."""
        self.timeline.sort(key=lambda clip: (clip.track_index, clip.timeline_start_sec))

    # --- Public Timeline Management API ---

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

    def get_track_duration(self, track_index: int) -> float:
        """
        Calculates the total duration of a specific track by finding the
        end point of the last clip on that track.
        """
        clips_on_track = self.get_clips_on_track(track_index)
        if not clips_on_track:
            return 0.0
        
        return max(
            (clip.timeline_start_sec + clip.duration_sec for clip in clips_on_track),
            default=0.0
        )

    def find_clip_by_id(self, clip_id: str) -> Optional[TimelineClip]:
        """Finds a clip on the timeline by its unique clip_id."""
        return next((clip for clip in self.timeline if clip.clip_id == clip_id), None)

    def clip_id_exists(self, clip_id: str) -> bool:
        """Checks if a clip_id is already in use on the timeline."""
        return any(clip.clip_id == clip_id for clip in self.timeline)

    def get_clips_on_track(self, track_index: int) -> List[TimelineClip]:
        """
        Returns a sorted list of all clips on a specific track.
        This will be essential for the 'add_to_timeline' tool's logic.
        """
        # The main timeline is already sorted, so we can just filter.
        return [clip for clip in self.timeline if clip.track_index == track_index]