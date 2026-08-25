from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from wearbench.damage import compute_severity, normalized_severity, alpha_for_target_torque_scale
from wearbench.run_utils import load_config

cfg = load_config(ROOT / "config.json")
log = np.load(ROOT / "outputs" / "baseline_dance.npz", allow_pickle=False)
severity = compute_severity(log["time"], log["qvel"], log["torque"])
sev_norm = normalized_severity(severity)
jnames = [str(x) for x in log["joint_names"]]
alpha = alpha_for_target_torque_scale(
    target_repetition=int(cfg["target_failure_repetition"]),
    target_scale=float(cfg["target_weakest_joint_torque_scale"]),
    floor=float(cfg["torque_scale_floor"]),
    exponent=float(cfg["health_exponent"]),
)
order = np.argsort(-severity)
rows = []
for rank, idx in enumerate(order, 1):
    rows.append({
        "rank": rank,
        "joint": jnames[idx],
        "severity_abs": float(severity[idx]),
        "severity_norm": float(sev_norm[idx]),
    })

out_json = ROOT / "outputs" / "damage_profile.json"
out_csv = ROOT / "outputs" / "damage_profile.csv"
out_json.write_text(json.dumps({
    "note": "MVP relative work-like severity; NOT calibrated physical wear.",
    "alpha_accelerated": alpha,
    "target_failure_repetition": int(cfg["target_failure_repetition"]),
    "target_weakest_joint_torque_scale": float(cfg["target_weakest_joint_torque_scale"]),
    "joints": rows,
}, indent=2))
with out_csv.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"Accelerated alpha = {alpha:.6g}")
print("Top loaded joints:")
for row in rows[:10]:
    print(f"  {row['rank']:2d}. {row['joint']:<35} severity_norm={row['severity_norm']:.4f}")
print(out_json)
