from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import user_env as env
from wearbench.run_utils import FailureConfig, load_config, run_dance

cfg = load_config(ROOT / "config.json")
model, data = env.make_model_and_data()
f = cfg["failure"]
fcfg = FailureConfig(
    base_height_m=float(f["base_height_m"]),
    max_abs_roll_deg=float(f["max_abs_roll_deg"]),
    max_abs_pitch_deg=float(f["max_abs_pitch_deg"]),
    max_tracking_error=f["max_tracking_error"],
    failure_hold_s=float(f["failure_hold_s"]),
)
duration = float(getattr(env, "dance_duration_s", lambda: cfg["dance_duration_s"])())
out = ROOT / "outputs" / "baseline_dance.npz"
res = run_dance(env, model, data, duration, fcfg, out)
print(f"Baseline: failed={res.failed}, reason={res.failure_reason}, t={res.failure_time_s}")
print(out)
if res.failed:
    raise SystemExit("Baseline itself fails. Fix controller/reset/failure thresholds before degradation experiments.")
