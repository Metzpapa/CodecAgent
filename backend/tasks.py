# backend/tasks.py

import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

from celery import Celery
from celery.signals import worker_process_init
from sqlalchemy.orm import Session

# --- MODIFIED: Import new modules for DB access ---
from .database import SessionLocal, Job

# --- Local Imports from your existing codebase ---
# We import the core Agent and State classes.
from .agent import Agent
from .state import State
# +++ NEW: Import the custom exception to handle graceful job completion +++
from .tools.finish_job import JobFinishedException

# --- Celery Application Setup ---

load_dotenv()  # Load environment variables from .env file

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND_URL,
    include=["backend.tasks"]
)

celery_app.conf.task_track_started = True


# --- Logging Configuration ---
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

root_logger = logging.getLogger()
original_handlers = list(root_logger.handlers)

def setup_job_logging(job_id: str):
    """
    Configures the Python logging module to write to a unique file for each job
    and also stream to the console.
    """
    for handler in list(root_logger.handlers):
        if handler not in original_handlers:
            root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    log_file = LOGS_DIR / f"{job_id}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.INFO)

    logging.info(f"Logging configured for job_id: {job_id}. Outputting to {log_file}")


# --- NEW: Helper function to update the database from within a task ---
def update_job_in_db(job_id: str, status: str, result_payload: dict = None):
    """
    Updates a job's status and result in the database.
    This function creates its own short-lived database session, which is a
    safe practice for operations outside the main FastAPI request-response cycle.
    """
    db: Session = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if job:
            job.status = status
            if result_payload:
                job.result_payload = result_payload
            db.commit()
            logging.info(f"Updated job {job_id} status to {status} in the database.")
        else:
            logging.error(f"Could not find job {job_id} in the database to update.")
    except Exception as e:
        logging.error(f"Database error while updating job {job_id}: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


# --- Celery Task Definition ---

@celery_app.task(bind=True)
def run_editing_job(self, job_id: str, prompt: str, assets_directory: str, output_dir: str):
    """
    The main Celery task that runs the video editing agent.
    It now updates the main application database on completion or failure.
    """
    setup_job_logging(job_id)

    try:
        logging.info(f"Starting job {job_id}. Prompt: '{prompt}'")
        self.update_state(state='PROGRESS', meta={'status': 'Initializing agent...'})
        # --- NEW: Update status in our main database ---
        update_job_in_db(job_id, "PROGRESS")

        session_state = State(assets_directory=assets_directory)
        video_agent = Agent(state=session_state)

        logging.info("Agent initialized. Starting main execution loop...")
        self.update_state(state='PROGRESS', meta={'status': 'Agent is processing the request...'})
        
        video_agent.run(prompt=prompt)

        raise RuntimeError("Agent execution loop finished without calling the 'finish_job' tool. The job is incomplete.")

    except JobFinishedException as e:
        logging.info(f"Job {job_id} finished gracefully via finish_job tool.")
        # --- NEW: Update status and result in our main database ---
        update_job_in_db(job_id, "SUCCESS", e.result)
        # The result payload from the exception becomes the final result of the Celery task.
        return e.result

    except Exception as e:
        logging.error(f"Job {job_id} failed with an unhandled exception.", exc_info=True)
        error_payload = {
            "status": "FAILED",
            "message": f"An unexpected error occurred: {type(e).__name__} - {e}",
            "output_path": None
        }
        # --- NEW: Update status and result in our main database on failure ---
        update_job_in_db(job_id, "FAILURE", error_payload)
        # This ensures the exception is propagated so Celery marks the task as FAILED.
        raise e