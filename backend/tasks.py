# backend/tasks.py

import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

from celery import Celery
from celery.signals import worker_process_init

# --- Local Imports from your existing codebase ---
# We import the core Agent and State classes.
from .agent import Agent
from .state import State
# +++ NEW: Import the custom exception to handle graceful job completion +++
from .tools.finish_job import JobFinishedException

# --- Celery Application Setup ---

load_dotenv()  # Load environment variables from .env file

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
        session_state = State(assets_directory=assets_directory)
        video_agent = Agent(state=session_state)

        # 3. Run the main agent loop. This is the core, long-running process.
        logging.info("Agent initialized. Starting main execution loop...")
        self.update_state(state='PROGRESS', meta={'status': 'Agent is processing the request...'})
        
        # The agent's run method will execute until the `finish_job` tool is called,
        # which raises a JobFinishedException to be caught below.
        video_agent.run(prompt=prompt)

        # --- MODIFIED: This part of the code should no longer be reachable ---
        # If the agent's run loop finishes without raising the special exception,
        # it means the agent stopped without formally finishing the job. This is an error.
        raise RuntimeError("Agent execution loop finished without calling the 'finish_job' tool. The job is incomplete.")

    # +++ NEW: Catch the specific exception for a graceful, controlled finish +++
    except JobFinishedException as e:
        logging.info(f"Job {job_id} finished gracefully via finish_job tool.")
        # The result payload from the exception becomes the final result of the Celery task.
        return e.result

    except Exception as e:
        logging.error(f"Job {job_id} failed with an unhandled exception.", exc_info=True) # exc_info=True adds the traceback
        # This ensures the exception is propagated so Celery marks the task as FAILED.
        raise e