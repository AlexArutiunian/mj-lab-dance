#!/usr/bin/env python3
"""Plot the conditional fall transition from a native health-sweep summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/rb_y_16s/native_health_transition_500k_1m/summary.json"


def _wilson_interval(falls: np.ndarray, trials: np.ndarray, z: float = 1.959963984540054) -> tuple[np.ndarray, np.ndarray]:
    proportion = falls / trials
    denominator = 1.0 + z**2 / trials
    centre = (proportion + z**2 / (2.0 * trials)) / denominator
    radius = z * np.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2)) / denominator
    return np.clip(centre - radius, 0.0, 1.0), np.clip(centre + radius, 0.0, 1.0)


def _switch_report(input_dir: Path, checkpoints: list[int]) -> dict[str, object]:
    statuses: dict[int, list[int]] = {}
    for checkpoint in checkpoints:
        path = input_dir / f"checkpoint_{checkpoint}" / "trials.csv"
        if not path.exists():
            return {"available": False}
        with path.open(newline="") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: int(row["sample"]))
        statuses[checkpoint] = [int(row["failed"]) for row in rows]
    transitions: list[dict[str, int]] = []
    completion_to_fall = 0
    fall_to_completion = 0
    for left, right in zip(checkpoints, checkpoints[1:]):
        before = np.asarray(statuses[left], dtype=np.int8)
        after = np.asarray(statuses[right], dtype=np.int8)
        c_to_f = int(np.sum((before == 0) & (after == 1)))
        f_to_c = int(np.sum((before == 1) & (after == 0)))
        completion_to_fall += c_to_f
        fall_to_completion += f_to_c
        transitions.append({"from": left, "to": right, "completion_to_fall": c_to_f, "fall_to_completion": f_to_c})
    stacked = np.asarray([statuses[checkpoint] for checkpoint in checkpoints], dtype=np.int8)
    return {
        "available": True,
        "ever_failed": int(np.sum(np.any(stacked == 1, axis=0))),
        "failed_at_final_checkpoint": int(np.sum(stacked[-1] == 1)),
        "completion_to_fall_switches": completion_to_fall,
        "fall_to_completion_switches": fall_to_completion,
        "adjacent_checkpoint_switches": transitions,
        "interpretation": "Individual simulated outcomes are not strictly monotone under closed-loop nonlinear dynamics; do not treat first failure as irreversible damage onset.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    rows = sorted(result["results"], key=lambda row: int(row["checkpoint_completed_dances"]))
    rows = [row for row in rows if int(row["checkpoint_completed_dances"]) > 0]
    dances = np.asarray([float(row["checkpoint_completed_dances"]) for row in rows])
    trials = np.asarray([float(row["trials"]) for row in rows])
    falls = np.asarray([float(row["failed"]) for row in rows])
    fraction = falls / trials
    lower, upper = _wilson_interval(falls, trials)
    q_rms = np.asarray([float(row["joint_position_rms_p50"]) for row in rows])
    q_rms_p95 = np.asarray([float(row["joint_position_rms_p95"]) for row in rows])
    pelvis_rms = np.asarray([float(row["pelvis_position_rms_m_p50"]) for row in rows])
    health = np.asarray([float(row["weakest_health_p50"]) for row in rows])
    jumps = np.diff(fraction) / np.diff(dances / 1000.0)
    steepest = int(np.argmax(jumps)) if len(jumps) else 0
    out_dir = args.out_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25})
    figure, (ax_fall, ax_error) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    ax_fall.fill_between(dances / 1000.0, lower, upper, color="#d95f02", alpha=0.18, label="95% Wilson interval")
    ax_fall.plot(dances / 1000.0, fraction, "o-", color="#d95f02", linewidth=2.4, label="conditional floor-fall fraction")
    ax_fall.plot(dances / 1000.0, 1.0 - fraction, "o--", color="#1b9e77", linewidth=1.8, label="completion fraction")
    ax_fall.set_ylim(-0.03, 1.03)
    ax_fall.set_ylabel("Fraction of 100 virtual robots")
    ax_fall.set_title("Native MuJoCo conditional wear transition (common random numbers)")
    ax_fall.legend(loc="upper left", ncol=2)
    if len(jumps):
        left, right = dances[steepest : steepest + 2] / 1000.0
        ax_fall.axvspan(left, right, color="#7570b3", alpha=0.10)
        ax_fall.annotate(
            f"steepest observed step: {left:.0f}k-{right:.0f}k",
            xy=((left + right) / 2.0, max(fraction[steepest], fraction[steepest + 1])),
            xytext=(0, 20), textcoords="offset points", ha="center", color="#4a3f83",
            arrowprops={"arrowstyle": "-", "color": "#4a3f83"},
        )

    ax_error.plot(dances / 1000.0, q_rms, "o-", color="#386cb0", linewidth=2.4, label="joint-position RMS, median [rad]")
    ax_error.plot(dances / 1000.0, q_rms_p95, "o--", color="#386cb0", alpha=0.75, label="joint-position RMS, p95 [rad]")
    ax_error.plot(dances / 1000.0, pelvis_rms, "s-", color="#e7298a", linewidth=2.0, label="pelvis RMS, median [m]")
    axis_health = ax_error.twinx()
    axis_health.plot(dances / 1000.0, health, "d-", color="#66a61e", linewidth=1.8, label="weakest health, median")
    axis_health.set_ylim(-0.03, 1.03)
    axis_health.set_ylabel("Median weakest-joint health")
    ax_error.set_xlabel("Completed dances [thousands]")
    ax_error.set_ylabel("Tracking deviation")
    handles, labels = ax_error.get_legend_handles_labels()
    health_handles, health_labels = axis_health.get_legend_handles_labels()
    ax_error.legend(handles + health_handles, labels + health_labels, loc="upper left", ncol=2)

    png = out_dir / "native_health_transition.png"
    pdf = out_dir / "native_health_transition.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    report = {
        "steepest_observed_interval": [int(dances[steepest]), int(dances[steepest + 1])] if len(jumps) else None,
        "steepest_observed_slope_falls_per_1000_dances": float(jumps[steepest]) if len(jumps) else None,
        "switch_analysis": _switch_report(args.input.parent, [int(value) for value in dances]),
    }
    report_path = out_dir / "native_health_transition_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[PLOT] {png}")
    print(f"[PLOT] {pdf}")
    print(f"[PLOT] {report_path}")
    if len(jumps):
        print(f"[TRANSITION] steepest_interval={dances[steepest] / 1000:.0f}k-{dances[steepest + 1] / 1000:.0f}k slope={jumps[steepest]:.6f}_falls_per_1000_dances")


if __name__ == "__main__":
    main()
