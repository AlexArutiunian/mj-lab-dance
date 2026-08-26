from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DANCE_SIM = ROOT.parent / "dance_sim"
PYTHON = DANCE_SIM / ".venv" / "bin" / "python"
WORKER = ROOT / "scripts" / "10_run_batched_wear_survival.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _parse_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one device is required")
    duplicated_gpu = any(device != "cpu" and devices.count(device) > 1 for device in set(devices))
    if duplicated_gpu:
        raise ValueError(
            "Each worker must use a distinct physical device; duplicate device entries contaminate MJWarp results"
        )
    return devices


def _run_trial(args, trial: int, device: str) -> dict[str, object]:
    seed = int(args.seed_start + trial)
    trial_dir = args.out_dir / "trials" / f"trial_{trial:04d}_seed_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "worker.log"
    checkpoints = [0] + [int(x) for x in args.checkpoints.split(",") if int(x) != 0]
    cmd = [
        str(PYTHON),
        str(WORKER),
        "--motion-file",
        str(args.motion_file.resolve()),
        "--policy",
        str(args.policy.resolve()),
        "--damage-profile",
        str(args.damage_profile.resolve()),
        "--num-envs",
        "1",
        "--duration",
        str(args.duration),
        "--checkpoints",
        ",".join(map(str, checkpoints)),
        "--target-repetition",
        str(args.target_repetition),
        "--target-scale",
        str(args.target_scale),
        "--seed",
        str(seed),
        "--device",
        device,
        "--physics-origin-mode",
        "local",
        "--out-dir",
        str(trial_dir),
    ]
    if args.pose_xy_jitter_m > 0.0:
        cmd.extend(["--pose-xy-jitter-m", str(args.pose_xy_jitter_m)])
    if args.yaw_jitter_deg > 0.0:
        cmd.extend(["--yaw-jitter-deg", str(args.yaw_jitter_deg)])
    if args.joint_jitter_rad > 0.0:
        cmd.extend(["--joint-jitter-rad", str(args.joint_jitter_rad)])

    started = time.perf_counter()
    with log_path.open("w") as log:
        process = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    run_summary_path = trial_dir / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text()) if run_summary_path.exists() else {}
    records_path = trial_dir / "summary.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    valid = process.returncode == 0 and bool(run_summary.get("valid", False))
    return {
        "trial": trial,
        "seed": seed,
        "device": device,
        "valid": valid,
        "returncode": process.returncode,
        "wall_time_s": elapsed,
        "validation_error": run_summary.get("validation_error"),
        "records": records,
        "log": str(log_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Independent single-world paired healthy/worn trials.")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument(
        "--devices",
        default="cpu",
        help="Comma-separated devices; CPU is the qualified reference, while GPU is diagnostic only.",
    )
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--checkpoints", default="100000,500000,1000000")
    p.add_argument("--target-repetition", type=int, default=1_000_000)
    p.add_argument("--target-scale", type=float, default=0.3)
    p.add_argument("--duration", type=float, default=16.24)
    p.add_argument("--pose-xy-jitter-m", type=float, default=0.0)
    p.add_argument("--yaw-jitter-deg", type=float, default=0.0)
    p.add_argument("--joint-jitter-rad", type=float, default=0.0)
    p.add_argument(
        "--motion-file",
        type=Path,
        default=DANCE_SIM / "assets/policies/mimic/dance1_subject2_16s_faststart/params/dance1_subject2_16s_faststart.npz",
    )
    p.add_argument(
        "--policy",
        type=Path,
        default=DANCE_SIM / "assets/policies/mimic/dance1_subject2_16s_faststart/exported/policy.onnx",
    )
    p.add_argument("--damage-profile", type=Path, default=ROOT / "outputs/rb_y_16s/damage_profile_1m_scale03.json")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/independent_trials_rb_y")
    args = p.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    reset_protocol = (
        "stress_jitter"
        if args.pose_xy_jitter_m > 0.0 or args.yaw_jitter_deg > 0.0 or args.joint_jitter_rad > 0.0
        else "deploy_exact"
    )

    devices = _parse_devices(args.devices)
    all_cpu = all(device == "cpu" for device in devices)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trial_results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {
            pool.submit(_run_trial, args, trial, devices[trial % len(devices)]): trial
            for trial in range(args.trials)
        }
        for future in as_completed(futures):
            result = future.result()
            trial_results.append(result)
            done = len(trial_results)
            avg = (time.perf_counter() - started) / done
            eta = avg * (args.trials - done) / max(1, len(devices))
            print(
                f"[TRIAL] {done}/{args.trials} seed={result['seed']} device={result['device']} "
                f"valid={result['valid']} wall={float(result['wall_time_s']):.1f}s eta={eta/60:.1f}m",
                flush=True,
            )

    trial_results.sort(key=lambda item: int(item["trial"]))
    flat_rows: list[dict[str, object]] = []
    for result in trial_results:
        for record in result["records"]:
            flat_rows.append(
                {
                    "trial": result["trial"],
                    "seed": result["seed"],
                    "device": result["device"],
                    "valid": result["valid"],
                    **record,
                }
            )

    checkpoint_summary: list[dict[str, object]] = []
    checkpoints = sorted({int(row["checkpoint_repetitions"]) for row in flat_rows})
    for checkpoint in checkpoints:
        rows = [row for row in flat_rows if int(row["checkpoint_repetitions"]) == checkpoint and bool(row["valid"])]
        successes = sum(int(row["failed_envs"]) == 0 for row in rows)
        lo, hi = wilson_interval(successes, len(rows))
        checkpoint_summary.append(
            {
                "checkpoint_repetitions": checkpoint,
                "valid_trials": len(rows),
                "successful_trials": successes,
                "failed_trials": len(rows) - successes,
                "survival_rate": successes / len(rows) if rows else math.nan,
                "survival_ci95_low": lo,
                "survival_ci95_high": hi,
                "weakest_health": float(rows[0]["weakest_health"]) if rows else math.nan,
                "weakest_torque_scale": float(rows[0]["weakest_torque_scale"]) if rows else math.nan,
            }
        )

    healthy = next((row for row in checkpoint_summary if row["checkpoint_repetitions"] == 0), None)
    healthy_failure_rate = (
        1.0 - float(healthy["survival_rate"])
        if healthy is not None and not math.isnan(float(healthy["survival_rate"]))
        else math.nan
    )
    for row in checkpoint_summary:
        row["excess_failure_rate_vs_healthy"] = (
            (1.0 - float(row["survival_rate"])) - healthy_failure_rate
            if not math.isnan(healthy_failure_rate) and not math.isnan(float(row["survival_rate"]))
            else math.nan
        )

    summary = {
        "mode": "independent_single_world_paired_trials",
        "status": "VALID" if all(bool(x["valid"]) for x in trial_results) else "INVALID_TRIALS_PRESENT",
        "evidence_class": (
            "ROBUSTNESS_STRESS_TEST"
            if reset_protocol != "deploy_exact"
            else "EXACT_DEPLOY_CPU_REFERENCE"
            if all_cpu
            else "EXACT_DEPLOY_GPU_UNQUALIFIED"
        ),
        "reset_protocol": reset_protocol,
        "model_status": "MODEL_ASSUMPTION_ONLY",
        "trials_requested": args.trials,
        "valid_trials": sum(bool(x["valid"]) for x in trial_results),
        "devices": devices,
        "target_repetition": args.target_repetition,
        "target_scale": args.target_scale,
        "duration_s": args.duration,
        "perturbations": {
            "pose_xy_jitter_m": args.pose_xy_jitter_m,
            "yaw_jitter_deg": args.yaw_jitter_deg,
            "joint_jitter_rad": args.joint_jitter_rad,
        },
        "provenance": {
            "motion_file": str(args.motion_file.resolve()),
            "motion_sha256": _sha256(args.motion_file.resolve()),
            "policy_file": str(args.policy.resolve()),
            "policy_sha256": _sha256(args.policy.resolve()),
            "damage_profile": str(args.damage_profile.resolve()),
            "damage_profile_sha256": _sha256(args.damage_profile.resolve()),
        },
        "wall_time_s": time.perf_counter() - started,
        "checkpoints": checkpoint_summary,
    }
    (args.out_dir / "trial_results.json").write_text(json.dumps(trial_results, indent=2))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if flat_rows:
        with (args.out_dir / "trials.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
