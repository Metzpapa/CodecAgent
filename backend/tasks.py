# backend/tasks.py

import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

from celery import Celery
from celery.utils.log import get_task_logger
from sqlalchemy.orm import Session

# Local imports
from .database import SessionLocal, Job
from .agent import Agent
from .state import State
from .tools.finish_job import JobFinishedException
from .agent_logging import AgentContextLogger # <-- NEW: Import the new logger

load_dotenv()

# --- Celery Configuration (Unchanged) ---
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND_URL,
    include=["backend.tasks"]
)

celery_app.conf.task_track_started = True

# --- Database Helper (Unchanged) ---
def update_job_in_db(job_id: str, status: str, result_payload: dict = None):
    """Updates the job's status and result in the database."""
    db: Session = SessionLocal()
    try:
        job = db.query(Job).filter(Job.job_id == job_id).first()
        if job:
            job.status = status
            if result_payload:
                job.result_payload = result_payload
            db.commit()
            # Use a generic logger here as this can be called outside the task context
            logging.info(f"Updated job {job_id} status to {status} in the database.")
        else:
            logging.error(f"Could not find job {job_id} in the database to update.")
    except Exception as e:
        logging.error(f"Database error while updating job {job_id}: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


# --- Main Celery Task (Refactored) ---
@celery_app.task(bind=True)
def run_editing_job(self, job_id: str, prompt: str, assets_directory: str, output_dir: str):
    """
    The main Celery task that runs the video editing agent.
    It now uses the AgentContextLogger to manage all logging, including
    real-time console output via the Celery task logger.
    """
    # Get the Celery logger for real-time console streaming.
    logger = get_task_logger(__name__)
    
    # This will be our single source of truth for all logging related to this job.
    context_logger = None

    try:
        # Instantiate our new, powerful logger. It will handle both file logs
        # and streaming to the console via the `logger` instance.
        context_logger = AgentContextLogger(job_id=job_id, stream_logger=logger)

        logger.info(f"Starting job {job_id}. Prompt: '{prompt}'")
        self.update_state(state='PROGRESS', meta={'status': 'Initializing agent...'})
        update_job_in_db(job_id, "PROGRESS")

        session_state = State(assets_directory=assets_directory)
        
        # The Agent is now given the context_logger to handle all its logging needs.
        video_agent = Agent(state=session_state, context_logger=context_logger)

        logger.info("Agent initialized. Starting main execution loop...")
        self.update_state(state='PROGRESS', meta={'status': 'Agent is processing the request...'})
        
        # The agent's run method will now use the context_logger to log
        # its setup, thoughts, tool calls, and tool results in real-time.
        video_agent.run(prompt=prompt)

        # This line should ideally not be reached. If it is, it means the agent
        # finished its loop without calling the mandatory `finish_job` tool.
        raise RuntimeError("Agent execution loop finished without calling the 'finish_job' tool. The job is incomplete.")

    except JobFinishedException as e:
        # This is the expected, graceful exit from the agent's run loop.
        logger.info(f"Job {job_id} finished gracefully via finish_job tool.")
        update_job_in_db(job_id, "SUCCESS", e.result)
        return e.result

    except Exception as e:
        # Catch any other unhandled exceptions during the process.
        logger.error(f"Job {job_id} failed with an unhandled exception.", exc_info=True)
        error_payload = {
            "status": "FAILED",
            "message": f"An unexpected error occurred: {type(e).__name__} - {e}",
            "output_path": None
        }
        update_job_in_db(job_id, "FAILURE", error_payload)
        # Re-raise the exception so Celery marks the task as failed.
        raise e
    
    finally:
        # This block ensures that our log files are always closed properly,
        # and the "SESSION END" footer is written, no matter how the task exits.
        if context_logger:
            logger.info(f"Closing log files for job {job_id}.")
            context_logger.close()