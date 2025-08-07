# backend/main.py

from dotenv import load_dotenv
load_dotenv()
import uuid
import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime

# --- Local Application Imports ---
from . import database
from .database import Job, get_db
from .auth import get_current_user_id
from .tasks import celery_app, run_editing_job

# --- Configuration ---
# Define a base directory where all job-related files will be stored.
JOBS_BASE_DIR = Path("codec_jobs")
JOBS_BASE_DIR.mkdir(exist_ok=True)

# Initialize database tables on module load.
database.Base.metadata.create_all(bind=database.engine)


# --- FastAPI App Initialization ---
app = FastAPI(title="Codec AI Video Editing Backend")

@app.on_event("startup")
def on_startup():
    """
    This function runs when the FastAPI application starts.
    It calls our database initializer to ensure all tables are created.
    """
    database.init_db()


# --- CORS Middleware ---
# Allows the frontend (running on a different origin) to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow any origin.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions ---
def cleanup_job_files(job_dir: Path):
    """A helper to safely remove a job's directory and all its contents."""
    logging.info(f"Cleaning up job directory: {job_dir}")
    if job_dir.exists() and job_dir.is_dir():
        shutil.rmtree(job_dir)


# --- Pydantic Models for API Responses ---
class JobResponse(BaseModel):
    job_id: str
    prompt: str
    status: str
    created_at: datetime
    result_payload: Optional[dict] = None

    class Config:
        from_attributes = True # Pydantic v2 equivalent of orm_mode


# --- API Endpoints ---

@app.get("/")
def read_root():
    """A simple 'health check' endpoint to confirm the API is running."""
    return {"message": "Codec AI Backend is running."}


@app.get("/jobs", response_model=List[JobResponse])
def get_jobs_for_user(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Retrieves all jobs associated with the currently authenticated user.
    This populates the "My Edits" list on the frontend.
    """
    jobs = db.query(Job).filter(Job.user_id == user_id).order_by(Job.created_at.desc()).all()
    return jobs


@app.post("/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Starts a new video editing job. It's non-blocking.
    1. Creates a unique job ID and a dedicated directory for its assets.
    2. Saves the user's uploaded files into that directory.
    3. Creates a new job record in the database, linking it to the user.
    4. Dispatches the actual editing task to a background Celery worker.
    5. Immediately returns the job ID to the client.
    """
    job_id = str(uuid.uuid4())
    job_dir = JOBS_BASE_DIR / job_id
    assets_dir = job_dir / "assets"
    output_dir = job_dir / "output"

    try:
        assets_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        for file in files:
            if ".." in file.filename or "/" in file.filename:
                raise HTTPException(status_code=400, detail=f"Invalid filename: {file.filename}")
            
            file_path = assets_dir / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logging.info(f"Saved asset file for job {job_id}: {file_path}")

    except Exception as e:
        cleanup_job_files(job_dir)
        raise HTTPException(status_code=500, detail=f"Failed to set up job environment: {e}")

    # Create a new Job record in the database.
    new_job = Job(
        job_id=job_id,
        user_id=user_id,
        prompt=prompt,
        status="PENDING"
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)
    logging.info(f"Job {job_id} for user {user_id} saved to database.")

    logging.info(f"Dispatching job {job_id} to Celery worker.")
    run_editing_job.apply_async(
        args=[job_id, prompt, str(assets_dir), str(output_dir)],
        task_id=job_id
    )

    return {"job_id": job_id, "status": "ACCEPTED"}


@app.get("/jobs/{job_id}/status")
def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Allows the front-end to poll for the status of a specific job.
    It reads directly from the database, which is the single source of truth
    updated by the Celery worker. This endpoint is decoupled from the Celery
    result backend.
    """
    job = db.query(Job).filter(Job.job_id == job_id, Job.user_id == user_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or you do not have permission to view it.")

    # The worker updates the DB, so we just return the DB state. No Celery call needed.
    return {"job_id": job.job_id, "status": job.status, "result": job.result_payload}


@app.get("/jobs/{job_id}/download")
async def download_result(
    job_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Serves the final output file for a completed job, if one exists.
    It verifies ownership and reads the result information directly from the database.
    Once the download is initiated, it schedules the job's files for cleanup.
    """
    job = db.query(Job).filter(Job.job_id == job_id, Job.user_id == user_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or you do not have permission to view it.")

    if job.status != 'SUCCESS':
        raise HTTPException(status_code=400, detail=f"Job is not successfully completed. Current status: {job.status}")

    result_info = job.result_payload
    if not isinstance(result_info, dict):
        raise HTTPException(status_code=404, detail="Job result is not in the expected format.")

    output_path_str = result_info.get("output_path")
    if not output_path_str:
        agent_message = result_info.get("message", "No output file was generated.")
        raise HTTPException(status_code=404, detail=f"Job completed with no downloadable file. Agent message: '{agent_message}'")

    output_file_path = Path(output_path_str)

    if not output_file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Output file not found on server at path: {output_file_path}")

    # Find the parent job directory to schedule for cleanup
    job_dir = None
    for p in output_file_path.parents:
        if p.name == job_id:
            job_dir = p
            break
    
    if job_dir:
        background_tasks.add_task(cleanup_job_files, job_dir)
    else:
        logging.warning(f"Could not determine job directory from path {output_file_path} to schedule cleanup.")

    return FileResponse(
        path=output_file_path,
        filename=output_file_path.name,
        media_type='application/octet-stream'
    )