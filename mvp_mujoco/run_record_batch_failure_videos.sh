#!/usr/bin/env bash
set -euo pipefail

MVP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DANCE_SIM_DIR="$MVP_DIR/../dance_sim"
PY_SITE="$DANCE_SIM_DIR/.venv/lib/python3.11/site-packages"
NVIDIA_LIBS="$(find "$PY_SITE/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$MVP_DIR"
exec "$DANCE_SIM_DIR/.venv/bin/python" scripts/12_record_batch_failure_videos.py "$@"
