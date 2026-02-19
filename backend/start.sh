#!/bin/bash
set -e

PORT=${PORT:-10000}

echo "Starting Roboto SAI Backend..."

exec uvicorn main_modular:app --host 0.0.0.0 --port $PORT