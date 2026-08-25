#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/dance_sim/.venv/bin/python"
UPSTREAM="$ROOT_DIR/dance_sim/external/unitree_rl_mjlab"

test -x "$PYTHON" || { echo "Missing Python environment: $PYTHON"; exit 1; }
test -d "$UPSTREAM" || { echo "Missing upstream MJLab checkout: $UPSTREAM"; exit 1; }
test -f "$ROOT_DIR/dance_sim/assets/policies/mimic/dance1_subject2_16s_faststart/params/dance1_subject2_16s_faststart.npz"
test -f "$ROOT_DIR/dance_sim/assets/policies/mimic/dance1_subject2_16s_faststart/exported/policy.onnx"

"$PYTHON" - <<'PY'
import importlib

for name in ("mujoco", "mujoco_warp", "numpy", "onnxruntime", "torch", "warp"):
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'installed')}")
PY

echo "doctor: OK"
