from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from wearbench.damage import damage_after_repetitions, health_from_damage, torque_scale_from_health
from wearbench.run_utils import load_config

out = ROOT / "outputs"
cfg = load_config(ROOT / "config.json")
profile = json.loads((out / "damage_profile.json").read_text())
rows = sorted(profile["joints"], key=lambda r: float(r["severity_norm"]))
joint_labels = [r["joint"] for r in rows]
severity_values = np.asarray([float(r["severity_norm"]) for r in rows], dtype=float)

# 1) Damage / severity profile.
fig, ax = plt.subplots(figsize=(8, max(5, 0.24 * len(rows))))
ax.barh(joint_labels, severity_values)
ax.set_xlabel("Normalized work-like severity per dance")
ax.set_title("One-dance relative joint loading (MVP proxy)")
fig.tight_layout(); fig.savefig(out / "fig_damage_per_joint.png", dpi=180); plt.close(fig)

summary = json.loads((out / "failure_summary.json").read_text()) if (out / "failure_summary.json").exists() else None
first_failure = int(summary["first_failure_repetition"]) if summary else int(cfg["target_failure_repetition"])
alpha = float(profile["alpha_accelerated"])
sev = np.asarray([r["severity_norm"] for r in profile["joints"]], dtype=float)
# profile rows are rank-sorted; only plot a handful of top severity joints.
order = np.argsort(-sev)[:5]
reps = np.unique(np.linspace(1, max(2, first_failure), min(200, max(2, first_failure)), dtype=int))

# 2) Torque capability vs repetition for top-5 joints.
fig, ax = plt.subplots(figsize=(8, 5))
for idx in order:
    scales = []
    for n in reps:
        d = damage_after_repetitions(np.asarray([sev[idx]]), n-1, alpha)
        h = health_from_damage(d)
        s = torque_scale_from_health(h, float(cfg["torque_scale_floor"]), float(cfg["health_exponent"]))
        scales.append(float(s[0]))
    ax.plot(reps, scales, label=profile["joints"][idx]["joint"])
ax.axvline(first_failure, linestyle="--", linewidth=1, label="first failure")
ax.set_xlabel("Virtual dance repetition")
ax.set_ylabel("Available actuator force scale")
ax.set_ylim(0, 1.05)
ax.set_title("Accelerated actuator capability decay")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(out / "fig_health_vs_repetition.png", dpi=180); plt.close(fig)


def load_npz(name):
    p = out / name
    return np.load(p, allow_pickle=False) if p.exists() else None

base = load_npz("baseline_dance.npz")
last = load_npz("last_success.npz")
fail = load_npz("first_failure.npz")

if base is not None and fail is not None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(base["time"], base["base_z"], label="healthy baseline")
    if last is not None:
        ax.plot(last["time"], last["base_z"], label="last success")
    ax.plot(fail["time"], fail["base_z"], label="first failure")
    ax.axhline(float(cfg["failure"]["base_height_m"]), linestyle="--", linewidth=1, label="fall threshold")
    ax.set_xlabel("Time, s"); ax.set_ylabel("Root/base height, m")
    ax.set_title("Healthy vs aged dance: base height")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "fig_base_height_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(base["time"], base["tracking_error"], label="healthy baseline")
    if last is not None:
        ax.plot(last["time"], last["tracking_error"], label="last success")
    ax.plot(fail["time"], fail["tracking_error"], label="first failure")
    ax.set_xlabel("Time, s"); ax.set_ylabel("Tracking error")
    ax.set_title("Healthy vs aged dance: tracking degradation")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "fig_tracking_comparison.png", dpi=180); plt.close(fig)

print("Wrote figures to", out)
