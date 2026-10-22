#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export SIM_BACKEND=genesis

if [[ ! -f .env ]]; then
  echo "Note: .env not found; using the installed local planner or the labelled deterministic fallback." >&2
fi

# Prefer the locally installed vision/tool model when the caller has not chosen a planner.
if [[ -z "${OPENARM_LOCAL_PLANNER:-}" ]]; then
  if command -v ollama >/dev/null 2>&1 && ollama list 2>/dev/null | awk '{print $1}' | grep -qx 'qwen3-vl:4b'; then
    export OPENARM_LOCAL_PLANNER=1
  fi
fi
export OPENARM_OLLAMA_MODEL="${OPENARM_OLLAMA_MODEL:-qwen3-vl:4b}"

uv sync

# Reuse this project's running owner instead of starting another Python window.
EXISTING_SERVICE=$(uv run python - <<'PY'
import httpx
try:
    response = httpx.get('http://127.0.0.1:8000/api/health', timeout=1)
    response.raise_for_status()
    data = response.json()
    if data.get('service') == 'sim-agent' or data.get('robot') == 'OpenArm v2':
        print('viewer' if data.get('viewer_online') or data.get('viewer_starting') else 'broker')
except (httpx.HTTPError, ValueError):
    pass
PY
)
if [[ "$EXISTING_SERVICE" == "viewer" ]]; then
  echo "OpenArm is already running: http://127.0.0.1:8000/simple"
  exit 0
fi

BACKEND_PID=""
if [[ "$EXISTING_SERVICE" != "broker" ]]; then
  GENESIS_SHOW_VIEWER= nohup uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --no-access-log --log-level warning > /tmp/sim-agent-backend.log 2>&1 &
  BACKEND_PID=$!
fi
cleanup() {
  if [[ -n "$BACKEND_PID" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

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
GENESIS_SHOW_VIEWER=1 uv run python run_genesis_gui.py "$@"
