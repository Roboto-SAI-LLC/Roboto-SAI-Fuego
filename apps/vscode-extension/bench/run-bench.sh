#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
PORT="${LLAMA_SERVER_PORT:-8787}"
URL="http://127.0.0.1:${PORT}/v1/completions"
REQUESTS="${BENCH_REQUESTS:-50}"
SIZES="${BENCH_SIZES:-short,medium,long}"
LANGUAGES="${BENCH_LANGUAGES:-typescript,python}"

mkdir -p "$RESULTS_DIR"

JSON_OUT="$RESULTS_DIR/bench-$TIMESTAMP.json"
TABLE_OUT="$RESULTS_DIR/bench-$TIMESTAMP.txt"
LOG_OUT="$RESULTS_DIR/llama-server-$TIMESTAMP.log"

port_open() {
  if command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "$PORT" >/dev/null 2>&1
    return $?
  fi

  python - <<PY >/dev/null 2>&1
import socket
sock = socket.socket()
try:
    sock.settimeout(1)
    sock.connect(("127.0.0.1", int("$PORT")))
    print("open")
except Exception:
    pass
finally:
    sock.close()
PY
}

if ! port_open; then
  echo "llama-server not detected on port $PORT. Starting..."

  SERVER_PATH="${LLAMA_SERVER_PATH:-}"
  if [[ -z "$SERVER_PATH" ]]; then
    if [[ -x "$REPO_ROOT/llama-server.exe" ]]; then
      SERVER_PATH="$REPO_ROOT/llama-server.exe"
    elif command -v llama-server.exe >/dev/null 2>&1; then
      SERVER_PATH="$(command -v llama-server.exe)"
    elif command -v llama-server >/dev/null 2>&1; then
      SERVER_PATH="$(command -v llama-server)"
    fi
  fi

  if [[ -z "$SERVER_PATH" ]]; then
    echo "llama-server not found. Set LLAMA_SERVER_PATH or add it to PATH."
    exit 1
  fi

  MODEL_PATH="${LLAMA_MODEL_PATH:-$REPO_ROOT/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf}"
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model not found. Set LLAMA_MODEL_PATH to your GGUF model path."
    exit 1
  fi

  ARGS=(--model "$MODEL_PATH" --host 127.0.0.1 --port "$PORT" --ctx-size 4096)
  if [[ -n "${LLAMA_SERVER_ARGS:-}" ]]; then
    read -r -a EXTRA <<< "$LLAMA_SERVER_ARGS"
    ARGS+=("${EXTRA[@]}")
  fi

  "$SERVER_PATH" "${ARGS[@]}" > "$LOG_OUT" 2>&1 &

  STARTED="false"
  for _ in {1..30}; do
    sleep 1
    if port_open; then
      STARTED="true"
      break
    fi
  done

  if [[ "$STARTED" != "true" ]]; then
    echo "llama-server failed to start within 30s. Check $LOG_OUT"
    exit 1
  fi
fi

pushd "$EXT_ROOT" >/dev/null
npx --yes tsc bench/benchmark.ts --outDir bench/dist --module commonjs --target es2020 --lib es2020,dom --moduleResolution node --esModuleInterop
node bench/dist/benchmark.js --requests "$REQUESTS" --url "$URL" --sizes "$SIZES" --languages "$LANGUAGES" --output "$JSON_OUT" --table "$TABLE_OUT"
popd >/dev/null

echo "Saved JSON results to $JSON_OUT"
echo "Saved table results to $TABLE_OUT"
