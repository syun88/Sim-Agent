#!/usr/bin/env bash
# Start the two local model services the app optionally uses, in the /data virtualenv that has transformers.
#
#   d1 decision model  :11435   (OPENARM_DECISION_URL for the app)
#   sam segmentation   :11436   (masks from a text prompt; not on the default path)
#
# Both are optional: with neither running the app records no monitor judgement and no masks, and nothing
# fails. Weights live in /data/hf-cache (not ~/.cache/huggingface) - set HF_HOME accordingly, which is what
# cost a reviewer a confused minute ("weights cache not found").
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${MODEL_VENV:-/data/d1-venv}
CACHE=${HF_HOME:-/data/hf-cache}
PY="$VENV/bin/python"

start() {
  local name=$1 port=$2 script=$3
  if curl -s --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "$name already running on :$port"
    return 0
  fi
  echo "starting $name on :$port"
  setsid systemd-run --user --scope -q -p MemoryMax=12G -p CPUQuota=600% \
    env HF_HOME="$CACHE" HF_HUB_DISABLE_TELEMETRY=1 HOME="${HOME:-/home/yuu}" \
    "$PY" "scripts/$script" --port "$port" --preload \
    >"/tmp/${name}.log" 2>&1 &
  for _ in $(seq 1 30); do
    sleep 2
    if curl -s --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "$name ready"
      return 0
    fi
  done
  echo "$name did not answer within 60 s; see /tmp/${name}.log" >&2
  return 1
}

case "${1:-all}" in
  d1)   start d1_decision_server 11435 d1_decision_server.py ;;
  sam)  start sam_segment_server 11436 sam_segment_server.py ;;
  all)  start d1_decision_server 11435 d1_decision_server.py
        start sam_segment_server 11436 sam_segment_server.py ;;
  *)    echo "usage: $0 [all|d1|sam]" >&2; exit 2 ;;
esac
echo
echo "app: OPENARM_DECISION_URL=http://127.0.0.1:11435 ./run_genesis_gui.sh --headless"
echo "stop: pkill -f d1_decision_server; pkill -f sam_segment_server"
