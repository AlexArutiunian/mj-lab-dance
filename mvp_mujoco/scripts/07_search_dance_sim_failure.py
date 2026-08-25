from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Binary-search first failing virtual dance repetition through bundled dance_sim.")
    p.add_argument("--duration", type=float, default=None, help="Candidate dance duration. Default: config dance_duration_s.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-repetitions", type=int, default=1_000_000)
    p.add_argument("--start-upper", type=int, default=None, help="Optional initial failing-bound candidate; default uses config target then expands.")
    args = p.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text())
    duration = float(args.duration if args.duration is not None else cfg.get("dance_duration_s", 120.0))
    upper = int(args.start_upper or cfg.get("target_failure_repetition", 50))
    upper = max(2, upper)

    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    runner = ROOT / "run_dance_sim_candidate.sh"
    records: list[dict[str, object]] = []

    def eval_rep(n: int) -> bool:
        out = outputs / f"_dance_search_rep_{n:09d}.npz"
        cmd = [
            str(runner),
            "--duration",
            str(duration),
            "--device",
            args.device,
            "--repetition",
            str(n),
            "--out",
            str(out),
        ]
        cp = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        print(cp.stdout, end="")
        if cp.returncode != 0:
            print(cp.stderr, file=sys.stderr)
            raise RuntimeError(f"dance_sim candidate failed for repetition {n}")
        z = np.load(out, allow_pickle=False)
        failed = bool(z["failed"][0])
        reason = str(z["failure_reason"][0])
        t = float(z["failure_time_s"][0])
        weakest = int(np.argmin(z["torque_scale"]))
        records.append(
            {
                "repetition": n,
                "failed": int(failed),
                "reason": reason,
                "failure_time_s": t,
                "weakest_joint": str(z["joint_names"][weakest]),
                "weakest_torque_scale": float(z["torque_scale"][weakest]),
            }
        )
        print(f"repetition={n} failed={failed} reason={reason} t={t}")
        return failed

    if eval_rep(1):
        raise SystemExit("Repetition 1 fails; baseline/failure detector is not valid.")

    while upper <= args.max_repetitions and not eval_rep(upper):
        upper *= 2
    if upper > args.max_repetitions:
        upper = args.max_repetitions
        if not eval_rep(upper):
            raise SystemExit(f"No failure found up to {args.max_repetitions}. Increase accelerated aging or lower target torque scale.")

    lower = 1
    while upper - lower > 1:
        mid = (lower + upper) // 2
        if eval_rep(mid):
            upper = mid
        else:
            lower = mid

    def copy_search(n: int, dst_name: str) -> None:
        src = outputs / f"_dance_search_rep_{n:09d}.npz"
        if not src.exists():
            eval_rep(n)
        shutil.copy2(src, outputs / dst_name)

    copy_search(lower, "last_success.npz")
    copy_search(upper, "first_failure.npz")

    with (outputs / "failure_search.csv").open("w", newline="") as fh:
        fieldnames = ["repetition", "failed", "reason", "failure_time_s", "weakest_joint", "weakest_torque_scale"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(records, key=lambda r: int(r["repetition"])))

    summary = {
        "note": "Accelerated uncalibrated MVP result; NOT a real Unitree G1 lifetime prediction.",
        "runner": "dance_sim_mjlab_onnx",
        "last_success_repetition": lower,
        "first_failure_repetition": upper,
    }
    (outputs / "failure_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nRESULT")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
