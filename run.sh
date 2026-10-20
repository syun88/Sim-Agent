#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Both entry points start the one native Genesis owner plus its web command broker.
exec ./run_genesis_gui.sh "$@"
