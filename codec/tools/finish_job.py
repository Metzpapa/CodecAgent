# codec/tools/finish_job.py

import os
import shutil
from pathlib import Path
import datetime
import logging
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple
from collections import defaultdict
import openai

import opentimelineio as otio
from pydantic import BaseModel, Field

from .base import BaseTool
from ..state import TimelineClip

# Use a forward reference for the State class to avoid circular imports.
if TYPE_CHECKING:
    from ..state import State

# --- Custom Exception to Signal Job Completion ---
class JobFinishedException(Exception):
    """
    A special exception raised by the finish_job tool to signal that the
    agent's work is complete. This allows for a clean exit from the agent's
    run loop and provides the final result to the Celery task.
    """
    def __init__(self, result: dict):
        self.result = result
        super().__init__("Job finished successfully.")


class FinishJobArgs(BaseModel):
    """Arguments for the finish_job tool."""
    message: str = Field(
        ...,
        description="A final, user-facing message summarizing the work done, explaining the result, or detailing why the request could not be completed. This is always required."
    )
    export_timeline: bool = Field(
        False,
        description="Set to True if a timeline has been created and should be exported. If False, the job will finish with only the message."
    )
    output_filename: str = Field(
        "codec_edit.otio",
        description="If exporting, the desired filename for the timeline file. The extension determines the format: '.otio' (recommended) or '.xml' (legacy)."
    )
    consolidate: bool = Field(
        True,
        description="If exporting, creates a self-contained project folder with the timeline file and copies of all used media. This is highly recommended."
    )


class FinishJobTool(BaseTool):
    """
    The final tool to be called in any job. It stops the agent's work and provides a summary message to the user.
    It can optionally export the final timeline into a self-contained, portable project folder.
    This tool MUST be called to complete a job, whether it was successful or not.
    """

    @property
    def name(self) -> str:
        return "finish_job"

    @property
    def description(self) -> str:
        return (
            "The single, final tool to end the editing job. Call this when the user's request is fully addressed or when you cannot proceed. "
            "You MUST provide a final `message` for the user. You can optionally `export_timeline` if you have created one."
        )

    @property
    def args_schema(self):
        return FinishJobArgs

    def execute(self, state: 'State', args: FinishJobArgs, client: openai.OpenAI, tmpdir: str) -> str:
        output_path = None
        export_error = None

        if args.export_timeline:
            if not state.timeline:
                export_error = "Agent requested export, but the timeline is empty."
                logging.warning(export_error)
            else:
                try:
                    output_path = self._export_and_consolidate(state, args)
                except Exception as e:
                    export_error = f"An error occurred during timeline export: {e}"
                    logging.error(export_error, exc_info=True)

        final_message = args.message
        if export_error:
            final_message += f"\n\n[System Note: An error occurred during the final export: {export_error}]"

        # Prepare the final result payload for the Celery task
        final_result = {
            "status": "COMPLETE",
            "message": final_message,
            "output_path": str(output_path.resolve()) if output_path else None
        }

        # Raise the special exception to terminate the agent's run loop
        raise JobFinishedException(final_result)

    def _export_and_consolidate(self, state: 'State', args: FinishJobArgs) -> Path:
        """
        Contains the core logic for building and writing the timeline file,
        and consolidating media if requested. Returns the path to the final .otio/.xml file.
        """
        job_dir = Path(state.assets_directory).parent
        output_dir = job_dir / "output"

        if not output_dir.is_dir():
            raise IOError(f"Could not find the job's output directory at '{output_dir}'.")

        if args.consolidate:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            package_name = f"{Path(args.output_filename).stem}_{timestamp}"
            package_dir = output_dir / package_name
            media_dir = package_dir / "media"
            
            logging.info(f"Consolidating project into: {package_dir}")
            os.makedirs(media_dir, exist_ok=True)

            unique_source_paths = {clip.source_path for clip in state.timeline}
            for src_path_str in unique_source_paths:
                src_path = Path(src_path_str)
                dest_path = media_dir / src_path.name
                shutil.copy2(src_path, dest_path)

            final_timeline_file_path = package_dir / args.output_filename
            base_path_for_relinking = package_dir
        else:
            final_timeline_file_path = output_dir / args.output_filename
            base_path_for_relinking = output_dir

        fps, width, height = state.get_sequence_properties()
        otio_timeline = self._build_otio_timeline(state, fps, width, height, base_path_for_relinking, consolidated=args.consolidate)

        file_ext = final_timeline_file_path.suffix.lower()
        adapter_name = "otio_json" if file_ext == ".otio" else "fcp_xml" if file_ext == ".xml" else None
        if not adapter_name:
            raise ValueError(f"Unsupported file extension '{file_ext}'. Please use '.otio' or '.xml'.")

        otio.adapters.write_to_file(otio_timeline, str(final_timeline_file_path), adapter_name=adapter_name)
        
        logging.info(f"Successfully exported timeline to: {final_timeline_file_path}")
        return final_timeline_file_path

    # --- HELPER METHODS MOVED FROM EXPORT_TIMELINE.PY ---

    def _build_otio_timeline(self, state: 'State', fps: float, width: int, height: int, base_path_for_relinking: Path, consolidated: bool) -> otio.schema.Timeline:
        """Builds the OTIO timeline by directly translating the state's track and clip structure."""
        otio_timeline = otio.schema.Timeline(name="Codec Agent Edit")
        self._inject_sequence_metadata(otio_timeline, fps, width, height)

        clips_by_track = defaultdict(list)
        for clip in state.timeline:
            clips_by_track[(clip.track_type, clip.track_number)].append(clip)

        sorted_tracks = sorted(clips_by_track.keys(), key=lambda t: (t[0], t[1]))

        for track_type, track_number in sorted_tracks:
            track_name = f"{track_type[0].upper()}{track_number}"
            track_kind = otio.schema.TrackKind.Video if track_type == 'video' else otio.schema.TrackKind.Audio
            otio_track = otio.schema.Track(name=track_name, kind=track_kind)

            last_clip_end_time = 0.0
            for codec_clip in clips_by_track[(track_type, track_number)]:
                gap_duration = codec_clip.timeline_start_sec - last_clip_end_time
                if gap_duration > 0.001:
                    gap = otio.schema.Gap(source_range=otio.opentime.TimeRange(duration=otio.opentime.from_seconds(gap_duration, rate=fps)))
                    otio_track.append(gap)
                
                otio_clip = self._create_otio_clip(codec_clip, fps, base_path_for_relinking, consolidated)
                otio_track.append(otio_clip)

                last_clip_end_time = codec_clip.timeline_start_sec + codec_clip.duration_sec
            
            otio_timeline.tracks.append(otio_track)

        return otio_timeline

    def _inject_sequence_metadata(self, timeline: otio.schema.Timeline, fps: float, width: int, height: int):
        """Injects FCP XML-specific metadata for better compatibility."""
        fcp_meta = timeline.metadata.setdefault("fcp_xml", {})
        is_ntsc = abs(fps - 23.976) < 0.01 or abs(fps - 29.97) < 0.01
        fcp_meta["rate"] = {"timebase": str(int(round(fps))), "ntsc": "TRUE" if is_ntsc else "FALSE"}
        fcp_meta.setdefault("media", {}).setdefault("video", {})["format"] = {
            "samplecharacteristics": { "width": str(width), "height": str(height), "pixelaspectratio": "square", "anamorphic": "FALSE", "fielddominance": "none" }
        }

    def _create_otio_clip(self, codec_clip: TimelineClip, timeline_fps: float, base_path_for_relinking: Path, consolidated: bool) -> otio.schema.Clip:
        """Creates a single OTIO clip from a TimelineClip, handling path relinking and custom metadata."""
        def _rt(sec: float): return otio.opentime.from_seconds(sec, rate=timeline_fps)
        
        if consolidated:
            target_url = (Path("media") / Path(codec_clip.source_path).name).as_posix()
        else:
            target_url = os.path.relpath(codec_clip.source_path, start=base_path_for_relinking)
            target_url = Path(target_url).as_posix()

        available_range = otio.opentime.TimeRange(start_time=_rt(0), duration=_rt(codec_clip.source_total_duration_sec))
        media_ref = otio.schema.ExternalReference(target_url=target_url, available_range=available_range)
        
        media_ref_meta = media_ref.metadata.setdefault("fcp_xml", {}).setdefault("media", {})
        if codec_clip.source_width > 0 and codec_clip.source_height > 0:
            is_ntsc = abs(codec_clip.source_frame_rate - 23.976) < 0.01 or abs(codec_clip.source_frame_rate - 29.97) < 0.01
            media_ref_meta["video"] = {
                "samplecharacteristics": { "width": str(codec_clip.source_width), "height": str(codec_clip.source_height), "pixelaspectratio": "square" },
                "rate": { "timebase": str(int(round(codec_clip.source_frame_rate))), "ntsc": "TRUE" if is_ntsc else "FALSE" }
            }
        if codec_clip.has_audio:
            media_ref_meta["audio"] = {"samplecharacteristics": {"samplerate": "48000"}}

        source_range = otio.opentime.TimeRange(start_time=_rt(codec_clip.source_in_sec), duration=_rt(codec_clip.duration_sec))
        
        otio_clip = otio.schema.Clip(name=codec_clip.clip_id, media_reference=media_ref, source_range=source_range)

        # --- MODIFICATION: Serialize transformation data into the clip's metadata ---
        if codec_clip.transformations:
            # Serialize our Keyframe objects into a list of simple dicts.
            # This makes the OTIO file clean and easily parsable by other tools.
            transforms_data = [
                kf.model_dump(exclude_none=True) for kf in codec_clip.transformations
            ]
            otio_clip.metadata['codec_transforms'] = transforms_data
        # --- END MODIFICATION ---

        return otio_clip