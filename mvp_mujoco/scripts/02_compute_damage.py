from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wearbench.damage import (  # noqa: E402
    alpha_for_target_torque_scale,
    compute_severity,
    normalized_severity,
)
from wearbench.run_utils import load_config  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Compute a per-joint MVP load profile from one explicit baseline rollout.")
    p.add_argument("--baseline", type=Path, default=ROOT / "outputs" / "baseline_dance.npz")
    p.add_argument("--out-json", type=Path, default=ROOT / "outputs" / "damage_profile.json")
    p.add_argument("--out-csv", type=Path, default=ROOT / "outputs" / "damage_profile.csv")
    p.add_argument("--target-repetition", type=int, default=None)
    p.add_argument("--target-scale", type=float, default=None)
    args = p.parse_args()

    cfg = load_config(ROOT / "config.json")
    baseline = args.baseline.resolve()
    log = np.load(baseline, allow_pickle=False)
    severity = compute_severity(log["time"], log["qvel"], log["torque"])
    sev_norm = normalized_severity(severity)
    joint_names = [str(x) for x in log["joint_names"]]
    target_repetition = int(
        args.target_repetition if args.target_repetition is not None else cfg["target_failure_repetition"]
    )
    target_scale = float(
        args.target_scale if args.target_scale is not None else cfg["target_weakest_joint_torque_scale"]
    )
    alpha = alpha_for_target_torque_scale(
        target_repetition=target_repetition,
        target_scale=target_scale,
        floor=float(cfg["torque_scale_floor"]),
        exponent=float(cfg["health_exponent"]),
    )
    order = np.argsort(-severity)
    rows = [
        {
            "rank": rank,
            "joint": joint_names[idx],
            "severity_abs": float(severity[idx]),
            "severity_norm": float(sev_norm[idx]),
        }
        for rank, idx in enumerate(order, 1)
    ]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "schema_version": 2,
        "status": "MODEL_ASSUMPTION_ONLY",
        "note": "Relative work-like load proxy; alpha is accelerated and not calibrated physical wear.",
        "baseline_path": str(baseline),
        "baseline_sha256": _sha256(baseline),
        "samples": int(len(log["time"])),
        "duration_s": float(log["time"][-1] - log["time"][0]) if len(log["time"]) > 1 else 0.0,
        "alpha_accelerated": alpha,
        "target_repetition": target_repetition,
        "target_weakest_joint_torque_scale": target_scale,
        "joints": rows,
    }
    args.out_json.write_text(json.dumps(profile, indent=2))
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Baseline SHA-256: {profile['baseline_sha256']}")
    print(f"Accelerated alpha = {alpha:.12g}")
    print("Top loaded joints:")
    for row in rows[:10]:
        print(f"  {row['rank']:2d}. {row['joint']:<35} severity_norm={row['severity_norm']:.4f}")
    print(args.out_json)


if __name__ == "__main__":
    main()
