FROM python:3.11-alpine

# Set environment variables for optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=10000

# Install system dependencies (Alpine optimized)
RUN apk add --no-cache \
    build-base \
    git \
    curl \
    && rm -rf /var/cache/apk/*

# Create non-root user for security (skip if already exists)
RUN addgroup -g 1000 -S roboto 2>/dev/null || true && \
    adduser -S -D -H -u 1000 -h /app -s /sbin/nologin -G roboto -g roboto roboto 2>/dev/null || true

WORKDIR /app

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies in virtual environment
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (excluding virtual environments and cache)
COPY api/ ./api/
COPY services/ ./services/
COPY utils/ ./utils/
COPY *.py ./
COPY *.txt ./
COPY *.sql ./

# Create data directory with proper permissions
RUN mkdir -p data && chown -R roboto:roboto /app

# Pre-compile Python bytecode for faster startup
RUN python -m compileall -b .

# Add health check endpoint
RUN echo '#!/bin/bash\n\
python -c "from main_modular import app; print(\"healthy\")" || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Create startup script that handles Render's PORT environment variable
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Use PORT from environment (Render sets this)\n\
PORT=${PORT:-10000}\n\
\n\
echo "Starting Roboto SAI Backend on port $PORT"\n\
\n\
# Start uvicorn with proper configuration\n\
exec uvicorn main_modular:app \\\n\
    --host 0.0.0.0 \\\n\
    --port $PORT \\\n\
    --workers 1 \\\n\
    --loop uvloop \\\n\
    --http httptools \\\n\
    --proxy-headers \\\n\
    --forwarded-allow-ips "*" \\\n\
    --access-log \\\n\
    --log-level info' > /app/start.sh && \
    chmod +x /app/start.sh

# Health check for Render (matches render.yaml healthCheckPath)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD /app/healthcheck.sh

# Switch to non-root user
USER roboto

# Expose the port (Render will override with $PORT)
EXPOSE 10000

# Use the startup script
CMD ["/app/start.sh"]
