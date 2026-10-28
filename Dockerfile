# ── Base Image ──────────────────────────────────────────────────────
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860 \
    TORCH_HOME=/app/.cache/torch

# Install system dependencies required for OpenCV, Git (for torch.hub), and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch CPU using --extra-index-url so standard build tools (flit_core, etc.) resolve from PyPI
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cpu

# Install facenet-pytorch with --no-deps, then remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir facenet-pytorch==2.5.3 --no-deps && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application codebase
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY app/ /app/app/
COPY scripts/ /app/scripts/

# Create necessary runtime directories
RUN mkdir -p /app/outputs /app/data/splits /app/.cache/torch

# Pre-cache DINOv2 from torch.hub so the container starts immediately offline
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)"

# Expose Gradio default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Launch the interactive web dashboard
CMD ["python", "app/app.py"]
