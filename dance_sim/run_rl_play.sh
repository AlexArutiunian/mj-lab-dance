#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SIM_REPO="${SIM_REPO:-external/unitree_rl_mjlab}"
MOTION="${1:-experiments/full_20p50_36p74_recut.npz}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ ! -d "$SIM_REPO" ]]; then
  echo "Missing simulator repo: $SIM_REPO" >&2
  echo "Put or symlink unitree_rl_mjlab there, then rerun." >&2
  exit 2
fi

if [[ -z "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT=/path/to/model.pt for RL play.py." >&2
  echo "Deploy ONNX is present, but this wrapper targets the original unitree_rl_mjlab scripts/play.py flow." >&2
  exit 2
fi

cd "$SIM_REPO"
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation \
  --checkpoint-file "$CHECKPOINT" \
  --motion-file "$(realpath "../$MOTION")" \
  --num-envs 1 \
  --device cpu \
  --viewer native
