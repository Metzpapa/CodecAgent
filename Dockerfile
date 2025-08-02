# Dockerfile

# 1. Base Image: Start with a slim, official Python image.
FROM python:3.11-slim

# 2. Set Working Directory: All subsequent commands will run from here.
WORKDIR /app

# 3. Install System Dependencies:
#    - ffmpeg: For all media processing tools.
#    - curl & nodejs: Required to build the React frontend.
#    We combine these into a single RUN layer to optimize image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python Dependencies:
#    Copy the requirements file first to leverage Docker's layer caching.
#    This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code:
#    Copy the entire project directory into the container's working directory.
COPY . .

# 6. Build Frontend:
#    Navigate into the frontend directory, install npm packages, and run the
#    build script. This creates the static `frontend/dist` directory which
#    Nginx will serve.
RUN npm install --prefix frontend && \
    npm run build --prefix frontend

# 7. Expose Port:
#    Inform Docker that the container will listen on port 8000 at runtime.
#    This is the port your FastAPI backend (uvicorn) will use.
EXPOSE 8000

# Note: The command to run the application (e.g., uvicorn or celery) is
# intentionally omitted here. It will be specified in the `docker-compose.yml`
# file, allowing this single image to be used for both the 'backend' and
# 'worker' services with different startup commands.