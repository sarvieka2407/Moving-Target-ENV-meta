# Deployment Dockerfile for Hugging Face Space / OpenEnv build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies (git is useful for VCS-based installs if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Space-specific dependencies first for better layer caching
COPY requirements-space.txt /app/requirements-space.txt
RUN pip install --upgrade pip && pip install -r /app/requirements-space.txt

# Copy project code
COPY . /app

# Expose FastAPI port used by internal environment server
EXPOSE 8000

# Main process: starts internal env server + GRPO loop (defined in app.py)
CMD ["python", "app.py"]
