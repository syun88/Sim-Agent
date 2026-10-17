#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example and add Neo4j credentials." >&2
  exit 1
fi

uv sync
if [[ ! -f frontend/dist/index.html ]]; then
  (cd frontend && npm install && npm run build)
fi

echo "Starting Embodied AI Agent on http://127.0.0.1:8000"
exec uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
