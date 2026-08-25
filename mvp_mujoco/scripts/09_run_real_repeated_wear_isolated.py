from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DANCE_SIM = ROOT.parent / "dance_sim"
sys.path.insert(0, str(ROOT))


def main() -> None:
    from wearbench.damage import alpha_for_target_torque_scale, health_from_damage, normalized_severity, torque_scale_from_health

    p = argparse.ArgumentParser(description="Run repeated dances with one isolated Python/MuJoCo-Warp process per repetition.")
    p.add_argument("--max-repetitions", type=int, default=100)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--target-repetition", type=int, default=None)
    p.add_argument("--target-scale", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--alpha-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--wear-effect-deadband", type=float, default=None)
    p.add_argument("--max-attempts-per-repetition", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "real_repeated_wear_isolated")
    p.add_argument("--worker-log", type=Path, default=None)
    args = p.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text())
    profile = json.loads((ROOT / "outputs" / "damage_profile.json").read_text())
    if args.alpha is not None:
        alpha = float(args.alpha)
        alpha_source = "cli_alpha"
    elif args.target_repetition is not None:
        alpha = alpha_for_target_torque_scale(
            target_repetition=int(args.target_repetition),
            target_scale=float(args.target_scale if args.target_scale is not None else cfg["target_weakest_joint_torque_scale"]),
            floor=float(cfg["torque_scale_floor"]),
            exponent=float(cfg["health_exponent"]),
        )
        alpha_source = f"target_repetition_{args.target_repetition}"
    else:
        alpha = float(profile["alpha_accelerated"])
        alpha_source = "damage_profile_alpha_accelerated"
    alpha *= float(args.alpha_scale)

    floor = float(cfg["torque_scale_floor"])
    exponent = float(cfg["health_exponent"])
    duration = float(args.duration if args.duration is not None else cfg.get("dance_duration_s", 120.0))
    deadband = float(args.wear_effect_deadband if args.wear_effect_deadband is not None else cfg.get("wear_effect_deadband", 1e-4))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    worker_root = args.out_dir / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "summary.csv"
    json_path = args.out_dir / "summary.json"
    state_path = args.out_dir / "latest_state.json"
    summary_path = args.out_dir / "run_summary.json"
    worker_log = args.worker_log or (args.out_dir / "worker.log")
    for stale_path in (csv_path, json_path, state_path, summary_path, worker_log):
        stale_path.unlink(missing_ok=True)

    python = DANCE_SIM / ".venv" / "bin" / "python"
    worker_script = ROOT / "scripts" / "08_run_real_repeated_wear.py"

    records: list[dict[str, object]] = []
    joint_names: np.ndarray | None = None
    damage: np.ndarray | None = None
    health: np.ndarray | None = None
    scale: np.ndarray | None = None
    started = time.perf_counter()

    print(f"[ISO] alpha={alpha:.12g} source={alpha_source} deadband={deadband:g}", flush=True)
    for rep in range(1, args.max_repetitions + 1):
        rep_started = time.perf_counter()
        ignored_baseline_failures = 0
        trace = None
        worker_row = None
        final_attempt = 0
        for attempt in range(1, args.max_attempts_per_repetition + 1):
            final_attempt = attempt
            rep_dir = worker_root / f"rep_{rep:04d}_attempt_{attempt:02d}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            scale_file = rep_dir / "initial_scale.json"
            if scale is not None:
                scale_file.write_text(json.dumps(scale.tolist()))

            attempt_seed = int(args.seed) + (rep - 1) * 1000 + (attempt - 1)
            cmd = [
                str(python),
                str(worker_script),
                "--max-repetitions",
                "1",
                "--duration",
                str(duration),
                "--alpha",
                "0",
                "--out-dir",
                str(rep_dir),
                "--save-last-logs",
                "--seed",
                str(attempt_seed),
                "--wear-effect-deadband",
                str(deadband),
            ]
            if scale is not None:
                cmd.extend(["--initial-scale-file", str(scale_file)])

            with worker_log.open("a") as log:
                log.write(f"\n===== isolated repetition {rep} attempt {attempt} seed {attempt_seed} =====\n")
                proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                raise SystemExit(f"Worker failed with exit code {proc.returncode}; see {worker_log}")

            with (rep_dir / "summary.csv").open() as fh:
                worker_row = list(csv.DictReader(fh))[-1]
            trace_path = rep_dir / ("last_success_trace.npz" if worker_row["failed"] == "0" else "first_failure_trace.npz")
            trace = np.load(trace_path, allow_pickle=False)
            if worker_row["failed"] == "0":
                break
            applied_min = float(trace["torque_scale"].min())
            if applied_min < 1.0 - 1e-9:
                break
            ignored_baseline_failures += 1
            print(
                f"[ISO EVENT] rep={rep} attempt={attempt} ignored_baseline_failure "
                f"reason={worker_row['failure_reason'] or '-'} applied_scale={applied_min:.9f}",
                flush=True,
            )

        assert trace is not None and worker_row is not None
        if joint_names is None:
            joint_names = trace["joint_names"]
            damage = np.zeros(len(joint_names), dtype=np.float64)
            health = np.ones(len(joint_names), dtype=np.float64)
            scale = np.ones(len(joint_names), dtype=np.float64)
        assert damage is not None and health is not None and scale is not None and joint_names is not None

        health_before = health.copy()
        scale_before = scale.copy()
        weakest_before = int(np.argmin(scale_before))
        failed = worker_row["failed"] == "1"
        completed = not failed
        if completed:
            damage += alpha * normalized_severity(trace["severity"])
            health = health_from_damage(damage)
            scale = torque_scale_from_health(health, floor, exponent)
        weakest_after = int(np.argmin(scale))
        wall = time.perf_counter() - rep_started
        rec = {
            "repetition": rep,
            "completed": int(completed),
            "failed": int(failed),
            "failure_reason": worker_row["failure_reason"],
            "failure_time_s": worker_row["failure_time_s"],
            "wall_time_s": wall,
            "attempts": final_attempt,
            "ignored_baseline_failures": ignored_baseline_failures,
            "weakest_before": str(joint_names[weakest_before]),
            "health_before": float(health_before[weakest_before]),
            "scale_before": float(scale_before[weakest_before]),
            "weakest_after": str(joint_names[weakest_after]),
            "health_after": float(health[weakest_after]),
            "scale_after": float(scale[weakest_after]),
            "min_base_z": float(worker_row["min_base_z"]),
            "max_abs_roll_deg": float(worker_row["max_abs_roll_deg"]),
            "max_abs_pitch_deg": float(worker_row["max_abs_pitch_deg"]),
        }
        records.append(rec)
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        json_path.write_text(json.dumps(records, indent=2))
        state_path.write_text(json.dumps({"records": records[-10:], "latest": rec}, indent=2))
        avg = sum(float(r["wall_time_s"]) for r in records) / len(records)
        eta_s = (args.max_repetitions - rep) * avg
        print(
            f"[ISO] rep={rep}/{args.max_repetitions} completed={completed} failed={failed} "
            f"reason={rec['failure_reason'] or '-'} health={rec['health_after']:.9f} "
            f"scale={rec['scale_after']:.9f} attempts={final_attempt} "
            f"ignored={ignored_baseline_failures} wall={wall:.1f}s eta={eta_s/60:.1f}m",
            flush=True,
        )
        if failed:
            break

    summary = {
        "mode": "real_repeated_measured_damage_isolated_process",
        "alpha": alpha,
        "alpha_source": alpha_source,
        "alpha_scale": float(args.alpha_scale),
        "seed": int(args.seed),
        "wear_effect_deadband": deadband,
        "max_repetitions_requested": args.max_repetitions,
        "repetitions_run": len(records),
        "failed": bool(records[-1]["failed"]) if records else False,
        "first_failure_repetition": int(records[-1]["repetition"]) if records and records[-1]["failed"] else None,
        "wall_time_s": time.perf_counter() - started,
        "csv": str(csv_path),
        "json": str(json_path),
        "worker_log": str(worker_log),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("[ISO RESULT]")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
