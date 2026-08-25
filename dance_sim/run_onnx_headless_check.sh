#!/usr/bin/env bash
set -euo pipefail

DANCE_SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SITE="$DANCE_SIM_DIR/.venv/lib/python3.11/site-packages"
NVIDIA_LIBS="$(find "$PY_SITE/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)"

export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$DANCE_SIM_DIR"
STEPS="${1:-200}"
if [[ $# -gt 0 ]]; then
  shift
fi
DEVICE="${1:-cuda:0}"
if [[ $# -gt 0 ]]; then
  shift
fi

exec .venv/bin/python scripts/play_onnx_mjlab.py \
  --motion-file "$DANCE_SIM_DIR/experiments/full_20p50_to_end.npz" \
  --policy "$DANCE_SIM_DIR/assets/policies/mimic/dance1_subject2/exported/policy.onnx" \
  --viewer none \
  --steps "$STEPS" \
  --device "$DEVICE" \
  "$@"
