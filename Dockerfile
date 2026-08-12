FROM python:3.11-slim

# System packages needed at build/runtime:
# - build-essential + libpq-dev: some wheels (psycopg2, pymilvus's grpc deps)
#   fall back to source builds on platforms without a prebuilt wheel.
# - libmagic1: unstructured's file-type detection (UnstructuredMarkdownLoader).
# - curl: used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libmagic1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code.
COPY . .

# Runs as the api service by default (see docker-compose.yml); the worker
# service reuses this same image and overrides CMD with the celery command,
# so nothing celery-specific needs to live in this file.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/docs || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
