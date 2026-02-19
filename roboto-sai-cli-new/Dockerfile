# Roboto SAI CLI - Docker image
FROM python:3.12-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SIGIL=929

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -u 1000 roboto

WORKDIR /app

# Copy and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ .

# Switch user
USER roboto

# Entry
ENTRYPOINT ["roboto-sai"]