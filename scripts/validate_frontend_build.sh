#!/usr/bin/env bash
set -euo pipefail

echo "[validate_frontend_build] Installing node dependencies..."
npm ci --legacy-peer-deps

echo "[validate_frontend_build] Building frontend..."
npm run build

echo "[validate_frontend_build] Building Docker production image (multi-stage)..."
docker build --no-cache -t roboto-sai-frontend:local-prod .

echo "[validate_frontend_build] Running container and checking /health..."
CONTAINER_ID=$(docker run -d -e PORT=10000 -p 10000:10000 roboto-sai-frontend:local-prod)
# give the container a few seconds to boot
sleep 5

echo "[validate_frontend_build] Container logs:"
docker logs $CONTAINER_ID || true

if curl -fs "http://localhost:10000/health"; then
  echo "[validate_frontend_build] Healthcheck OK"
  docker kill $CONTAINER_ID >/dev/null
  docker rm $CONTAINER_ID >/dev/null
  exit 0
else
  echo "[validate_frontend_build] Healthcheck failed"
  docker logs $CONTAINER_ID || true
  docker kill $CONTAINER_ID >/dev/null || true
  docker rm $CONTAINER_ID >/dev/null || true
  exit 1
fi
