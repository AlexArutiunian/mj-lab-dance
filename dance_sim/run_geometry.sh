#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
MOTION="${1:-experiments/full_20p50_36p74_recut.npz}"

python3 scripts/replay_body_geometry.py \
  --motion "$MOTION" \
  --stride "${STRIDE:-2}" \
  --speed "${SPEED:-1.0}"
