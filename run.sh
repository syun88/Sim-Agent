#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example and add Neo4j credentials." >&2
  exit 1
fi

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
