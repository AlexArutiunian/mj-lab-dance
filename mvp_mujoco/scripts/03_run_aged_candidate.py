from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import user_env as env
from wearbench.damage import damage_after_repetitions, health_from_damage, torque_scale_from_health
from wearbench.mujoco_degradation import build_joint_actuator_map, ForceRangeDerater
from wearbench.run_utils import FailureConfig, load_config, run_dance

p = argparse.ArgumentParser()
p.add_argument("--repetition", type=int, required=True, help="Run this repetition N after fast-forwarding N-1 completed dances")
p.add_argument("--out", type=str, default=None)
args = p.parse_args()

cfg = load_config(ROOT / "config.json")
profile = json.loads((ROOT / "outputs" / "damage_profile.json").read_text())
alpha = float(profile["alpha_accelerated"])
rows_by_joint = {r["joint"]: r for r in profile["joints"]}

model, data = env.make_model_and_data()
amap = build_joint_actuator_map(model)
sev = np.asarray([rows_by_joint[n]["severity_norm"] for n in amap.joint_names], dtype=float)
damage = damage_after_repetitions(sev, args.repetition - 1, alpha)
health = health_from_damage(damage)
scale = torque_scale_from_health(health, float(cfg["torque_scale_floor"]), float(cfg["health_exponent"]))
derater = ForceRangeDerater(model, amap)
derater.apply_scales(scale)

f = cfg["failure"]
fcfg = FailureConfig(float(f["base_height_m"]), float(f["max_abs_roll_deg"]), float(f["max_abs_pitch_deg"]), f["max_tracking_error"], float(f["failure_hold_s"]))
duration = float(getattr(env, "dance_duration_s", lambda: cfg["dance_duration_s"])())
out = Path(args.out) if args.out else ROOT / "outputs" / f"candidate_rep_{args.repetition:07d}.npz"
res = run_dance(env, model, data, duration, fcfg, out)

print(f"repetition={args.repetition} failed={res.failed} reason={res.failure_reason} t={res.failure_time_s}")
idx = int(np.argmin(scale))
print(f"weakest={amap.joint_names[idx]} health={health[idx]:.4f} torque_scale={scale[idx]:.4f}")
print(out)
