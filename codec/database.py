# codec/database.py

import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# --- Database Configuration ---

# For a prototype, we'll use a simple SQLite database.
# The file will be created in the root of the project as `codec.db`.
# Using an environment variable allows for easy configuration in production later.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./codec.db")

# --- SQLAlchemy Setup ---

# The engine is the entry point to the database.
# The `connect_args` is specific to SQLite and is needed to allow the database
# to be accessed from multiple threads, which is the case with FastAPI.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# A sessionmaker creates new Session objects when called. A Session is the
# primary interface for all database operations.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# We will inherit from this class to create our ORM models.
Base = declarative_base()


# --- ORM Model Definition ---

class Job(Base):
    """
    SQLAlchemy model representing a single editing job in the database.
    This table replaces the in-memory `job_store` and persists job information,
    linking it to a specific user.
    """
    __tablename__ = "jobs"

    # A standard auto-incrementing primary key.
    id = Column(Integer, primary_key=True, index=True)

    # The unique identifier passed to Celery and used by the frontend.
    job_id = Column(String, unique=True, index=True, nullable=False)

    # The unique 'sub' identifier from the user's Google JWT.
    # Indexed for fast lookups of all jobs belonging to a user.
    user_id = Column(String, index=True, nullable=False)

    # The user's original natural language prompt.
    prompt = Column(String, nullable=False)

    # The timestamp when the job was created. Defaults to the current UTC time.
    created_at = Column(DateTime, default=datetime.utcnow)

    # The current status of the job, mirroring Celery's states (e.g., PENDING, PROGRESS, SUCCESS, FAILURE).
    status = Column(String, default="PENDING")

    # A JSON field to store the final result from the `finish_job` tool,
    # including the user-facing message and the output file path.
    result_payload = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<Job(job_id='{self.job_id}', user_id='{self.user_id}', status='{self.status}')>"


# --- Database Initialization ---

def init_db():
    """
    Creates all database tables defined in the Base metadata.
    This function should be called once on application startup.
    """
    print("Initializing database and creating tables if they don't exist...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")

# --- Dependency for FastAPI ---

def get_db():
    """
    A FastAPI dependency that provides a database session for a single request.
    It ensures the session is always closed after the request is finished,
    even if an error occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()