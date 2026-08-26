#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/../dance_sim/.venv/bin/python" "$ROOT/scripts/15_run_native_health_sweep.py" "$@"
