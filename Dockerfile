# Production Dockerfile for SynthDistill-LoRA
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SIMULATION_MODE=true \
    PORT=8000

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application source code
COPY src/ ./src/
COPY benchmarks/ ./benchmarks/
COPY tests/ ./tests/
COPY Makefile .
COPY README.md .

# Create persistent storage directories
RUN mkdir -p /app/data /app/checkpoints

EXPOSE 8000

# Default healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI orchestration server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
