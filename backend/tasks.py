# backend/tasks.py

import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

from celery import Celery
# +++ CHANGE: Import Celery's logger +++
from celery.utils.log import get_task_logger
from sqlalchemy.orm import Session

from .database import SessionLocal, Job
from .agent import Agent
from .state import State
from .tools.finish_job import JobFinishedException

load_dotenv()

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND_URL,
    include=["backend.tasks"]
)

celery_app.conf.task_track_started = True

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# --- REFACTORED Logging Setup ---
# This function no longer modifies the global logger. It just creates and returns handlers.
def get_job_log_handlers(job_id: str) -> tuple[logging.Handler, logging.Handler]:
    """
    Creates file and stream handlers for a specific job.
    """
    log_file = LOGS_DIR / f"{job_id}.log"
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    return file_handler, stream_handler

# This helper is unchanged, it's still good.
def update_job_in_db(job_id: str, status: str, result_payload: dict = None):
    db: Session = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if job:
            job.status = status
            if result_payload:
                job.result_payload = result_payload
            db.commit()
            logging.getLogger(__name__).info(f"Updated job {job_id} status to {status} in the database.")
        else:
            logging.getLogger(__name__).error(f"Could not find job {job_id} in the database to update.")
    except Exception as e:
        logging.getLogger(__name__).error(f"Database error while updating job {job_id}: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


@celery_app.task(bind=True)
def run_editing_job(self, job_id: str, prompt: str, assets_directory: str, output_dir: str):
    """
    The main Celery task that runs the video editing agent.
    It now uses Celery's built-in logger for safe, isolated logging.
    """
    # +++ CHANGE: Get the task-specific logger +++
    logger = get_task_logger(__name__)
    logger.propagate = False  # Prevent logs from propagating to the root logger
    logger.setLevel(logging.INFO)
    
    # +++ CHANGE: Set up logging for this task run +++
    # Clear any previous handlers and add our job-specific ones
    logger.handlers.clear()
    file_handler, stream_handler = get_job_log_handlers(job_id)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    try:
        # +++ CHANGE: Use the new task logger +++
        logger.info(f"Starting job {job_id}. Prompt: '{prompt}'")
        self.update_state(state='PROGRESS', meta={'status': 'Initializing agent...'})
        update_job_in_db(job_id, "PROGRESS")

        session_state = State(assets_directory=assets_directory)
        # +++ CHANGE: Pass job_id to the Agent for clean logging +++
        video_agent = Agent(state=session_state, job_id=job_id)

        logger.info("Agent initialized. Starting main execution loop...")
        self.update_state(state='PROGRESS', meta={'status': 'Agent is processing the request...'})
        
        # The agent's internal logging will now be captured by our new handlers
        video_agent.run(prompt=prompt)

        raise RuntimeError("Agent execution loop finished without calling the 'finish_job' tool. The job is incomplete.")

    except JobFinishedException as e:
        logger.info(f"Job {job_id} finished gracefully via finish_job tool.")
        update_job_in_db(job_id, "SUCCESS", e.result)
        return e.result

    except Exception as e:
        logger.error(f"Job {job_id} failed with an unhandled exception.", exc_info=True)
        error_payload = {
            "status": "FAILED",
            "message": f"An unexpected error occurred: {type(e).__name__} - {e}",
            "output_path": None
        }
        update_job_in_db(job_id, "FAILURE", error_payload)
        raise e
    
    finally:
        # +++ CHANGE: Clean up our handlers after the task is done +++
        logger.removeHandler(file_handler)
        logger.removeHandler(stream_handler)
