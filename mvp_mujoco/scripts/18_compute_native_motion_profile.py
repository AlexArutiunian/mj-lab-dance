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
    norm = severity / max(float(np.max(severity)), 1e-12)
    reference = json.loads(args.reference_profile.read_text())
    reference_total = float(sum(float(row["severity_abs"]) for row in reference["joints"]))
    alpha = float(reference["alpha_accelerated"]) * total / reference_total
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0])) for i in range(model.nu)]
    rows = [{"joint": name, "severity_abs": float(value), "severity_norm": float(relative)} for name, value, relative in zip(joints, severity, norm, strict=True)]
    profile = {
        "schema_version": 3,
        "status": "MOTION_SPECIFIC_ACCELERATED_MODEL",
        "motion": args.name,
        "trace": str(args.trace.resolve()),
        "trace_sha256": sha256(args.trace),
        "control_dt_s": args.control_dt,
        "duration_s": float(len(qvel) * args.control_dt),
        "total_abs_work_proxy": total,
        "reference_profile": str(args.reference_profile.resolve()),
        "reference_total_abs_work_proxy": reference_total,
        "relative_total_work_to_reference": total / reference_total,
        "alpha_accelerated": alpha,
        "joints": rows,
        "note": "Absolute work-like proxy and alpha remain accelerated model assumptions; this corrects policy-specific load shape and magnitude, not physical lifetime calibration.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2))
    print(json.dumps({key: profile[key] for key in ("motion", "duration_s", "total_abs_work_proxy", "relative_total_work_to_reference", "alpha_accelerated")}, indent=2))


if __name__ == "__main__":
    main()
