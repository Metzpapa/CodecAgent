# backend/tasks.py

import os
import logging
import sys
from pathlib import Path

from celery import Celery
from celery.signals import worker_process_init

# --- Local Imports from your existing codebase ---
# We import the core Agent and State classes.
from .agent import Agent
from .state import State
# We need to explicitly call the export tool at the end of a successful run.
from .tools.export_timeline import ExportTimelineTool, ExportTimelineArgs

# --- Celery Application Setup ---

# Get broker and backend URLs from environment variables, with defaults for local dev.
# This makes the app configurable for production without changing the code.
# You would run your worker like:
# CELERY_BROKER_URL=redis://... CELERY_RESULT_BACKEND_URL=redis://... celery -A backend.tasks worker
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND_URL", "redis://localhost:6379/0")

# Create the Celery application instance.
# The first argument is the name of the current module, which is standard practice.
celery_app = Celery(
    "tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND_URL,
    include=["backend.tasks"] # Explicitly include this module for task discovery.
)

# This setting allows the front-end to see the "STARTED" or "PROGRESS" state.
celery_app.conf.task_track_started = True


# --- Logging Configuration ---
# This is a critical function to provide the per-job logs you need for debugging.

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# We need to store the root logger to remove handlers later.
root_logger = logging.getLogger()
# Store original handlers to avoid removing handlers not added by us.
original_handlers = list(root_logger.handlers)

def setup_job_logging(job_id: str):
    """
    Configures the Python logging module to write to a unique file for each job
    and also stream to the console, mimicking your current workflow.
    """
    # Remove any handlers added by previous tasks in the same worker process.
    # This prevents log duplication.
    for handler in list(root_logger.handlers):
        if handler not in original_handlers:
            root_logger.removeHandler(handler)

    # Create a formatter for a consistent log message format.
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # Create a file handler to save logs to a job-specific file.
    log_file = LOGS_DIR / f"{job_id}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Create a stream handler to print logs to the worker's console.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Add the new handlers to the root logger.
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.INFO)

    logging.info(f"Logging configured for job_id: {job_id}. Outputting to {log_file}")


# --- Celery Task Definition ---

@celery_app.task(bind=True)
def run_editing_job(self, job_id: str, prompt: str, assets_directory: str, output_dir: str):
    """
    The main Celery task that runs the video editing agent.

    Args:
        self: The task instance (automatically passed by `bind=True`).
        job_id: The unique ID for this job.
        prompt: The user's natural language request.
        assets_directory: The path to the directory containing the user's uploaded media.
        output_dir: The path where the final exported file should be saved.
    """
    # 1. Set up logging for this specific job run.
    setup_job_logging(job_id)

    try:
        logging.info(f"Starting job {job_id}. Prompt: '{prompt}'")
        self.update_state(state='PROGRESS', meta={'status': 'Initializing agent...'})

        # 2. Initialize the agent's state and the agent itself.
        # This is the same logic that was in your old main.py.
        session_state = State(assets_directory=assets_directory)
        video_agent = Agent(state=session_state)

        # 3. Run the main agent loop. This is the core, long-running process.
        logging.info("Agent initialized. Starting main execution loop...")
        self.update_state(state='PROGRESS', meta={'status': 'Agent is processing the request...'})
        video_agent.run(prompt=prompt)
        logging.info("Agent has finished its execution loop.")

        # 4. Export the final timeline.
        # After the agent has built the timeline, we must export it to a file.
        self.update_state(state='PROGRESS', meta={'status': 'Exporting final timeline...'})
        
        if not session_state.timeline:
            raise ValueError("Agent finished but the timeline is empty. Nothing to export.")

        export_tool = ExportTimelineTool()
        # We'll create a self-contained project in the job's output directory.
        output_filename = "codec_edit.otio"
        export_args = ExportTimelineArgs(
            output_filename=output_filename,
            consolidate=True # This is important for creating a portable result.
        )
        
        # The export tool needs a slightly different setup now.
        # We'll temporarily change the "home" directory for the export function
        # so it correctly places the consolidated project inside our job's output folder.
        # This is a small hack to adapt the existing tool.
        original_home = Path.home
        Path.home = lambda: Path(output_dir)
        
        export_result = export_tool.execute(state=session_state, args=export_args, client=video_agent.client)
        
        # Restore the original home function
        Path.home = original_home

        if "Error" in export_result:
            raise RuntimeError(f"Failed to export timeline: {export_result}")
        
        logging.info(f"Export successful. Result: {export_result}")

        # The final consolidated folder is inside the output_dir.
        # We need to find the actual .otio file path to return.
        # The export tool creates a folder like "codec_edit_YYYYMMDD_HHMMSS"
        consolidated_folder = next(Path(output_dir).iterdir()) # Get the first (and only) sub-directory
        final_output_path = consolidated_folder / output_filename

        # 5. Return the successful result.
        # This dictionary will be stored as the task's result in Redis.
        return {
            "status": "COMPLETE",
            "output_path": str(final_output_path.resolve())
        }

    except Exception as e:
        logging.error(f"Job {job_id} failed.", exc_info=True) # exc_info=True adds the traceback
        # This ensures the exception is propagated so Celery marks the task as FAILED.
        raise e