# backend/main.py

import uuid
import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# We will create tasks.py next. This import prepares for that.
# It imports the Celery application instance and the task function we will define.
from .tasks import celery_app, run_editing_job

# --- Configuration ---
# Define a base directory where all job-related files will be stored.
# Using pathlib.Path makes the code work on Windows, macOS, and Linux.
JOBS_BASE_DIR = Path("codec_jobs")
JOBS_BASE_DIR.mkdir(exist_ok=True)

# This is a simple, in-memory dictionary to track job information.
# In a production system, you would replace this with a database like Redis
# or PostgreSQL to persist job state even if the API server restarts.
job_store: Dict[str, Dict] = {}


# --- FastAPI App Initialization ---
app = FastAPI(title="Codec AI Video Editing Backend")

# --- CORS Middleware ---
# The front-end website will run on a different "origin" (e.g., http://localhost:3000)
# than the API (e.g., http://localhost:8000). Browsers block requests between
# different origins by default for security. This middleware tells the browser
# that it's safe to allow requests from our front-end.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, we allow any origin.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all HTTP headers.
)


# --- Helper Functions ---
def cleanup_job_files(job_dir: Path):
    """A helper to safely remove a job's directory and all its contents."""
    logging.info(f"Cleaning up job directory: {job_dir}")
    if job_dir.exists() and job_dir.is_dir():
        shutil.rmtree(job_dir)


# --- API Endpoints ---

@app.get("/")
def read_root():
    """A simple 'health check' endpoint to confirm the API is running."""
    return {"message": "Codec AI Backend is running."}


@app.post("/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    This is the main endpoint to start a new video editing job.
    It's designed to be fast and non-blocking.

    1. Creates a unique job ID and a dedicated directory for its assets.
    2. Saves the user's uploaded files into that directory.
    3. Dispatches the actual editing task to a background Celery worker.
    4. Immediately returns the job ID to the client, so the user isn't left waiting.
    """
    job_id = str(uuid.uuid4())
    job_dir = JOBS_BASE_DIR / job_id
    assets_dir = job_dir / "assets"
    output_dir = job_dir / "output"

    try:
        # Create the necessary folder structure for this specific job.
        assets_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        # Loop through the uploaded files and save them to the job's assets directory.
        for file in files:
            # Basic security: prevent path traversal attacks.
            if ".." in file.filename or "/" in file.filename:
                raise HTTPException(status_code=400, detail=f"Invalid filename: {file.filename}")
            
            file_path = assets_dir / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logging.info(f"Saved asset file for job {job_id}: {file_path}")

    except Exception as e:
        # If anything goes wrong during setup, clean up the partially created
        # directories and inform the client.
        cleanup_job_files(job_dir)
        raise HTTPException(status_code=500, detail=f"Failed to set up job environment: {e}")

    # Store initial information about the job.
    job_store[job_id] = {
        "status": "PENDING",
        "result": None,
        "job_dir": str(job_dir),
        "output_dir": str(output_dir)
    }

    # This is the key step: we ask Celery to run our task in the background.
    # We use .apply_async() to pass arguments and assign our custom job_id as the task_id.
    logging.info(f"Dispatching job {job_id} to Celery worker.")
    run_editing_job.apply_async(
        args=[job_id, prompt, str(assets_dir), str(output_dir)],
        task_id=job_id
    )

    # The status code 202 Accepted is the standard for "I've received your
    # request and will process it, but it's not done yet."
    return {"job_id": job_id, "status": "ACCEPTED"}


@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str):
    """
    Allows the front-end to poll for the status of a specific job.
    It checks the Celery backend (Redis) for the real-time status of the task.
    """
    task_result = celery_app.AsyncResult(job_id)

    status = task_result.status
    result = task_result.result

    # If the job failed, the result will be an Exception object.
    # We convert it to a string to make it JSON-serializable.
    if isinstance(result, Exception):
        result = str(result)

    return {"job_id": job_id, "status": status, "result": result}


@app.get("/jobs/{job_id}/download")
async def download_result(job_id: str, background_tasks: BackgroundTasks):
    """
    Serves the final output file for a completed job.
    Once the download is initiated, it schedules the job's files for cleanup.
    """
    task_result = celery_app.AsyncResult(job_id)

    if not task_result.ready():
        raise HTTPException(status_code=400, detail=f"Job is not complete. Current status: {task_result.status}")

    if task_result.failed():
        raise HTTPException(status_code=404, detail=f"Job failed and has no output file. Error: {task_result.result}")

    result_info = task_result.result
    if not isinstance(result_info, dict) or "output_path" not in result_info:
        raise HTTPException(status_code=404, detail="Job finished, but no output file path was found in the result.")

    output_file_path = Path(result_info["output_path"])

    if not output_file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Output file not found on server at path: {output_file_path}")

    # Use FastAPI's BackgroundTasks to delete the job directory *after*
    # the response has been sent to the user.
    job_dir = output_file_path.parent.parent # e.g., codec_jobs/job_id/output -> codec_jobs/job_id
    background_tasks.add_task(cleanup_job_files, job_dir)

    return FileResponse(
        path=output_file_path,
        filename=output_file_path.name,
        media_type='application/octet-stream'  # A generic type to force browser download.
    )