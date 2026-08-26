#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/../dance_sim/.venv/bin/python" "$ROOT/scripts/15_run_native_health_sweep.py" \
  --checkpoints 0,500000,550000,600000,650000,700000,750000,800000,850000,900000,950000,1000000 \
  --trials 100 --workers 6 --seed 20260826 --wear-rate-cv 0.25 \
  --common-random-numbers \
  --out-dir "$ROOT/outputs/rb_y_16s/native_health_transition_500k_1m" "$@"
