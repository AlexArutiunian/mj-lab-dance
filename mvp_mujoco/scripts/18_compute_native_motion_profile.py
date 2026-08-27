#!/usr/bin/env python3
"""Derive a motion-specific accelerated wear profile from a native trace."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / "dance_sim/external/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml"
REFERENCE = ROOT / "outputs/rb_y_16s/damage_profile_1m_scale03.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-profile", type=Path, default=REFERENCE)
    parser.add_argument("--control-dt", type=float, default=0.02)
    args = parser.parse_args()
    trace = np.load(args.trace)
    qvel, torque = np.asarray(trace["qvel"])[:, 6:], np.asarray(trace["ctrl"])
    if qvel.shape != torque.shape:
        raise ValueError(f"Expected joint qvel and ctrl with matching shapes, got {qvel.shape} and {torque.shape}")
    severity = np.trapezoid(np.abs(qvel * torque), dx=args.control_dt, axis=0)
    total = float(np.sum(severity))
    max_severity = float(np.max(severity))
    norm = severity / max(max_severity, 1e-12)
    reference = json.loads(args.reference_profile.read_text())
    reference_total = float(sum(float(row["severity_abs"]) for row in reference["joints"]))
    reference_max = float(max(float(row["severity_abs"]) for row in reference["joints"]))
    # One global work-to-damage constant preserves D_j,m = kappa * S_j,m.
    # alpha is retained only as an equivalent max-normalized representation.
    kappa = float(reference["alpha_accelerated"]) / reference_max
    alpha = kappa * max_severity
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0])) for i in range(model.nu)]
    rows = [
        {
            "joint": name,
            "severity_abs": float(value),
            "severity_norm": float(relative),
            "damage_per_execution": float(kappa * value),
        }
        for name, value, relative in zip(joints, severity, norm, strict=True)
    ]
    profile = {
        "schema_version": 4,
        "status": "MOTION_SPECIFIC_GLOBAL_KAPPA_ACCELERATED_MODEL",
        "motion": args.name,
        "trace": str(args.trace.resolve()),
        "trace_sha256": sha256(args.trace),
        "control_dt_s": args.control_dt,
        "duration_s": float(len(qvel) * args.control_dt),
        "total_abs_work_proxy": total,
        "mean_abs_power_proxy": total / float(len(qvel) * args.control_dt),
        "max_joint_abs_work_proxy": max_severity,
        "reference_profile": str(args.reference_profile.resolve()),
        "reference_total_abs_work_proxy": reference_total,
        "reference_max_joint_abs_work_proxy": reference_max,
        "relative_total_work_to_reference": total / reference_total,
        "relative_max_joint_work_to_reference": max_severity / reference_max,
        "global_kappa_damage_per_work_proxy": kappa,
        "alpha_accelerated": alpha,
        "joints": rows,
        "note": "Damage is direct and joint-wise: D_j,m(R)=R*kappa*S_j,m. Absolute work-like proxy and kappa remain accelerated model assumptions, not physical lifetime calibration.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2))
    print(json.dumps({key: profile[key] for key in ("motion", "duration_s", "total_abs_work_proxy", "mean_abs_power_proxy", "relative_total_work_to_reference", "alpha_accelerated")}, indent=2))


if __name__ == "__main__":
    main()
