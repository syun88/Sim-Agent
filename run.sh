#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "Note: .env not found; Neo4j and ai& provider will use their documented fallbacks." >&2
fi

# Keep the optional local planner enabled when its model is already installed.
if [[ -z "${OPENARM_LOCAL_PLANNER:-}" ]] && command -v ollama >/dev/null 2>&1; then
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx 'qwen3-vl:4b'; then
    export OPENARM_LOCAL_PLANNER=1
  fi
fi
export OPENARM_OLLAMA_MODEL="${OPENARM_OLLAMA_MODEL:-qwen3-vl:4b}"

if command -v uv >/dev/null 2>&1; then
  uv sync
  RUNNER=(uv run uvicorn)
else
  # Any Python 3.12/3.13 environment with the dependencies from pyproject.toml works as well.
  RUNNER=(python -m uvicorn)
fi
if [[ ! -f frontend/dist/index.html ]]; then
  (cd frontend && npm install && npm run build)
fi

echo "Starting Embodied AI Agent on http://127.0.0.1:8000"
exec "${RUNNER[@]}" backend.app.main:app --host 127.0.0.1 --port 8000
