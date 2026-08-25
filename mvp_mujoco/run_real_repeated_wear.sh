#!/usr/bin/env bash
set -euo pipefail

MVP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DANCE_SIM_DIR="$MVP_DIR/../dance_sim"
PY_SITE="$DANCE_SIM_DIR/.venv/lib/python3.11/site-packages"
NVIDIA_LIBS="$(find "$PY_SITE/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$MVP_DIR"

OUT_DIR="$MVP_DIR/outputs/real_repeated_wear"
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --out-dir)
      if ((i + 1 < ${#ARGS[@]})); then
        OUT_DIR="${ARGS[$((i + 1))]}"
      fi
      ;;
    --out-dir=*)
      OUT_DIR="${ARGS[$i]#--out-dir=}"
      ;;
  esac
done
mkdir -p "$OUT_DIR"
rm -f \
  "$OUT_DIR/summary.csv" \
  "$OUT_DIR/summary.json" \
  "$OUT_DIR/latest_state.json" \
  "$OUT_DIR/run_summary.json" \
  "$OUT_DIR/last_success_trace.npz" \
  "$OUT_DIR/first_failure_trace.npz"

exec "$DANCE_SIM_DIR/.venv/bin/python" scripts/08_run_real_repeated_wear.py "$@"
