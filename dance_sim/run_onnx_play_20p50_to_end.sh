#!/usr/bin/env bash
set -euo pipefail

DANCE_SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SITE="$DANCE_SIM_DIR/.venv/lib/python3.11/site-packages"
NVIDIA_LIBS="$(find "$PY_SITE/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)"

export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$DANCE_SIM_DIR"
DEVICE="${1:-cuda:0}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ "$DEVICE" == "cpu" ]]; then
  ORT_PROVIDER="${ORT_PROVIDER:-cpu}"
else
  ORT_PROVIDER="${ORT_PROVIDER:-cuda}"
fi

exec .venv/bin/python scripts/play_onnx_mjlab.py \
  --motion-file "${MOTION_FILE:-$DANCE_SIM_DIR/assets/policies/mimic/dance1_subject2_16s_faststart/params/dance1_subject2_16s_faststart.npz}" \
  --policy "$DANCE_SIM_DIR/assets/policies/mimic/dance1_subject2_16s_faststart/exported/policy.onnx" \
  --viewer native \
  --device "$DEVICE" \
  --ort-provider "$ORT_PROVIDER" \
  --monitor \
  "$@"
