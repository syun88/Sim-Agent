#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export SIM_BACKEND=genesis

if [[ ! -f .env ]]; then
  echo "Note: .env not found; Neo4j and ai& provider will use their documented fallbacks." >&2
fi

# Prefer the locally installed vision/tool model when the caller has not chosen a planner.
if [[ -z "${OPENARM_LOCAL_PLANNER:-}" ]]; then
  if command -v ollama >/dev/null 2>&1 && ollama list 2>/dev/null | awk '{print $1}' | grep -qx 'qwen3-vl:4b'; then
    export OPENARM_LOCAL_PLANNER=1
  fi
fi
export OPENARM_OLLAMA_MODEL="${OPENARM_OLLAMA_MODEL:-qwen3-vl:4b}"

uv sync

# Start the web backend in the background so the Genesis viewer can poll it.
GENESIS_SHOW_VIEWER= nohup uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --no-access-log --log-level warning > /tmp/sim-agent-backend.log 2>&1 &
BACKEND_PID=$!
trap "kill $BACKEND_PID 2>/dev/null" EXIT

# Wait briefly for the backend to become ready.
for _ in $(seq 1 40); do
  if curl -sS --max-time 1 http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

echo "Web UI:  http://127.0.0.1:8000/simple"
echo "Starting Genesis native viewer..."

# Genesis viewer needs the main thread on macOS.
GENESIS_SHOW_VIEWER=1 uv run python run_genesis_gui.py
