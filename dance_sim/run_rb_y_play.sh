#!/usr/bin/env bash
set -euo pipefail

DANCE_SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$DANCE_SIM_DIR/run_onnx_play_20p50_to_end.sh" "${1:-cpu}" "${@:2}"
