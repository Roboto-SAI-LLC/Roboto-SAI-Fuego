#!/bin/bash
set -e

echo "🚀 Roboto SAI Backend - Starting up..."
echo "PORT: ${PORT:=8000}"
echo "Python version: $(python --version)"

# Log environment
echo "Environment check:"
echo "- XAI_API_KEY: ${XAI_API_KEY:-(not set)}"
echo "- FRONTEND_ORIGIN: ${FRONTEND_ORIGIN:-(not set)}"

# Try to import main module to catch import errors early
echo ""
echo "Checking Python imports..."
python -c "from main_modular import app; print('✅ main_modular.app imported successfully')" 2>&1 || \
{ echo "❌ Failed to import main_modular.app"; exit 1; }

# Start uvicorn with better error logging
echo ""
echo "Starting uvicorn server on port ${PORT:=8000}..."
exec python -m uvicorn main_modular:app \
  --host 0.0.0.0 \
  --port "${PORT:=8000}" \
  --log-level info \
  --access-log
