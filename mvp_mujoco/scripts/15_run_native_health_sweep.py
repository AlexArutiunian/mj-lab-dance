#!/usr/bin/env python3
"""Run deterministic native-MuJoCo health checkpoints and compare to healthy motion."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DANCE_SIM = ROOT.parent / "dance_sim"
NATIVE_RUNNER = DANCE_SIM / "run_native_mujoco_deploy.sh"
DEFAULT_PROFILE = ROOT / "outputs/rb_y_16s/damage_profile_1m_scale03.json"
RESULT_LINE = re.compile(
    r"min_pelvis=(?P<min_pelvis>[-+0-9.eE]+) min_torso=(?P<min_torso>[-+0-9.eE]+) "
    r"final_root_z=(?P<final_root_z>[-+0-9.eE]+) failed=(?P<failed>[01]) "
    r"failure_t=(?P<failure_t>[-+0-9.eE]+)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checkpoints(value: str) -> list[int]:
    checkpoints = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not checkpoints or checkpoints[0] != 0 or any(item < 0 for item in checkpoints):
        raise ValueError("--checkpoints must start with 0 and contain non-negative integers")
    return checkpoints


def _quat_angle_deg(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    dots = np.sum(reference * candidate, axis=1)
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))


def _trajectory_metrics(reference: np.lib.npyio.NpzFile, candidate: np.lib.npyio.NpzFile) -> dict[str, float]:
    result: dict[str, float] = {}
    for label, key in (("joint_position", "qpos"), ("joint_velocity", "qvel"), ("action", "actions")):
        a = reference[key]
        b = candidate[key]
        if key == "qpos":
            a, b = a[:, 7:], b[:, 7:]
        elif key == "qvel":
            a, b = a[:, 6:], b[:, 6:]
        diff = a - b
        result[f"{label}_rms"] = float(np.sqrt(np.mean(np.square(diff))))
        result[f"{label}_max_abs"] = float(np.max(np.abs(diff)))
    for label, key in (("pelvis_position", "pelvis_pos"), ("torso_position", "torso_pos")):
        diff = reference[key] - candidate[key]
        norm = np.linalg.norm(diff, axis=1)
        result[f"{label}_rms_m"] = float(np.sqrt(np.mean(np.square(norm))))
        result[f"{label}_max_m"] = float(np.max(norm))
    torso_angle = _quat_angle_deg(reference["torso_quat"], candidate["torso_quat"])
    result["torso_orientation_rms_deg"] = float(np.sqrt(np.mean(np.square(torso_angle))))
    result["torso_orientation_max_deg"] = float(np.max(torso_angle))
    return result


def _run_one(
    runner: Path,
    runner_args: list[str],
    duration: float,
    joint_scales: dict[str, float],
    sample: int,
    wear_rate_multiplier: float,
    weakest_health: float,
    weakest_scale: float,
    log_path: Path,
    trace_path: Path,
) -> dict[str, object]:
    command = [
        str(runner),
        *runner_args,
        "--duration",
        str(duration),
        "--trace",
        str(trace_path),
    ]
    for joint, scale in joint_scales.items():
        command.extend(["--joint-torque-scale", f"{joint}={scale:.10f}"])
    started = time.perf_counter()
    process = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
    elapsed = time.perf_counter() - started
    log_path.write_text(process.stdout + process.stderr)
    match = RESULT_LINE.search(process.stdout)
    if process.returncode != 0 or match is None:
        raise RuntimeError(f"Native run failed for sample={sample}; see {log_path}")
    values = {name: float(value) for name, value in match.groupdict().items() if name != "failed"}
    return {
        "sample": sample,
        "wear_rate_multiplier": wear_rate_multiplier,
        "weakest_health": weakest_health,
        "weakest_torque_scale": weakest_scale,
        "failed": int(match.group("failed")),
        "wall_time_s": elapsed,
        "trace": str(trace_path),
        **values,
    }


def _scales_for_sample(profile: dict[str, object], completed_dances: int, wear_rate: float) -> tuple[dict[str, float], float, float, str]:
    floor = 0.10
    exponent = 1.5
    alpha = float(profile["alpha_accelerated"])
    rows = profile["joints"]
    values: list[tuple[str, float, float]] = []
    for row in rows:
        damage = max(0.0, float(completed_dances) * alpha * float(row["severity_norm"]) * wear_rate)
        health = float(np.clip(1.0 - damage, 0.0, 1.0))
        scale = floor + (1.0 - floor) * health**exponent
        values.append((str(row["joint"]), health, scale))
    weakest_joint, weakest_health, weakest_scale = min(values, key=lambda item: item[2])
    return {joint: scale for joint, _, scale in values}, weakest_health, weakest_scale, weakest_joint


def _quantiles(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        f"{key}_p05": float(np.quantile(values, 0.05)),
        f"{key}_p50": float(np.quantile(values, 0.50)),
        f"{key}_p95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="0,100000,500000,1000000")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--wear-rate-cv",
        type=float,
        default=0.25,
        help="Conditional coefficient of variation for the accelerated wear-rate multiplier.",
    )
    parser.add_argument(
        "--common-random-numbers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse each virtual robot's wear-rate multiplier at every worn checkpoint.",
    )
    parser.add_argument("--duration", type=float, default=16.24)
    parser.add_argument("--runner", type=Path, default=NATIVE_RUNNER)
    parser.add_argument("--runner-arg", action="append", default=[], help="Extra argument passed through to --runner; may be repeated.")
    parser.add_argument("--damage-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/rb_y_16s/native_health_sweep")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.damage_profile = args.damage_profile.resolve()
    checkpoints = _parse_checkpoints(args.checkpoints)
    if args.trials < 1 or args.workers < 1 or args.wear_rate_cv < 0.0:
        raise ValueError("--trials and --workers must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    profile = json.loads(args.damage_profile.read_text())
    reference_trace = args.out_dir / "trace_healthy.npz"
    runner = args.runner.resolve()
    runner_args = list(args.runner_arg)
    subprocess.run(
        [
            str(runner), *runner_args, "--duration", str(args.duration), "--repetition", "1",
            "--damage-profile", str(args.damage_profile.resolve()), "--trace", str(reference_trace),
        ],
        check=True,
        cwd=ROOT,
    )
    reference = np.load(reference_trace)
    sigma = math.sqrt(math.log1p(args.wear_rate_cv**2))
    rng = np.random.default_rng(args.seed)
    base_wear_rates = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=args.trials)
    summary: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        repetition = checkpoint + 1
        scenario_dir = args.out_dir / f"checkpoint_{checkpoint}"
        scenario_dir.mkdir(exist_ok=True)
        if checkpoint == 0:
            wear_rates = np.ones(args.trials, dtype=np.float64)
        elif args.common_random_numbers:
            wear_rates = base_wear_rates
        else:
            # Shift log-space mean so E[wear_rate] is exactly one. This makes
            # the nominal accelerated profile the ensemble mean.
            wear_rates = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=args.trials)
        started = time.perf_counter()
        rows: list[dict[str, object]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for i, wear_rate in enumerate(wear_rates):
                scales, weakest_health, weakest_scale, _ = _scales_for_sample(profile, checkpoint, float(wear_rate))
                futures.append(
                    pool.submit(
                        _run_one,
                        runner,
                        runner_args,
                        args.duration,
                        scales,
                        i,
                        float(wear_rate),
                        weakest_health,
                        weakest_scale,
                        scenario_dir / f"trial_{i:03d}.log",
                        scenario_dir / f"trial_{i:03d}.npz",
                    )
                )
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                rows.append(future.result())
                if index % 10 == 0 or index == args.trials:
                    elapsed = time.perf_counter() - started
                    eta = elapsed / index * (args.trials - index)
                    print(f"[NATIVE HEALTH] checkpoint={checkpoint} {index}/{args.trials} eta={eta:.1f}s", flush=True)
        for row in rows:
            with np.load(str(row["trace"])) as candidate:
                row.update(_trajectory_metrics(reference, candidate))
        rows.sort(key=lambda row: int(row["sample"]))
        with (scenario_dir / "trials.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        signatures = {
            tuple(round(float(row[key]), 7) for key in ("min_pelvis", "min_torso", "final_root_z", "failure_t"))
            + (int(row["failed"]),)
            for row in rows
        }
        _, _, _, weakest_joint = _scales_for_sample(profile, checkpoint, 1.0)
        item: dict[str, object] = {
            "checkpoint_completed_dances": checkpoint,
            "native_repetition_index": repetition,
            "trials": args.trials,
            "completed": sum(not int(row["failed"]) for row in rows),
            "failed": sum(int(row["failed"]) for row in rows),
            "unique_physical_outcomes": len(signatures),
            "weakest_joint": weakest_joint,
        }
        for key in (
            "wear_rate_multiplier",
            "weakest_health",
            "weakest_torque_scale",
            "joint_position_rms",
            "joint_position_max_abs",
            "joint_velocity_rms",
            "joint_velocity_max_abs",
            "action_rms",
            "action_max_abs",
            "pelvis_position_rms_m",
            "pelvis_position_max_m",
            "torso_position_rms_m",
            "torso_position_max_m",
            "torso_orientation_rms_deg",
            "torso_orientation_max_deg",
        ):
            item.update(_quantiles(rows, key))
        summary.append(item)
        print(json.dumps(summary[-1], indent=2), flush=True)
    result = {
        "status": "CONDITIONAL_NATIVE_PARAMETER_ENSEMBLE",
        "protocol": "Fresh exact-start processes per checkpoint. Each worn virtual robot receives an accelerated wear-rate multiplier; it is parameter sensitivity, not measured G1 population reliability.",
        "duration_s": args.duration,
        "trials_per_checkpoint": args.trials,
        "seed": args.seed,
        "common_random_numbers": args.common_random_numbers,
        "wear_rate_distribution": {
            "family": "lognormal",
            "mean": 1.0,
            "median": math.exp(-0.5 * sigma**2),
            "coefficient_of_variation": args.wear_rate_cv,
        },
        "damage_profile": str(args.damage_profile.resolve()),
        "damage_profile_sha256": _sha256(args.damage_profile),
        "results": summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
