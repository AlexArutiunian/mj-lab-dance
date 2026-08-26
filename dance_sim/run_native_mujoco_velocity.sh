#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/native_mujoco_velocity_deploy.py" "$@"
