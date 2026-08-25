from pathlib import Path
import argparse
import csv
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser(description="Binary-search first failing virtual dance repetition without replaying all prior repetitions.")
p.add_argument("--max-repetitions", type=int, default=1_000_000)
p.add_argument("--start-upper", type=int, default=None, help="Optional initial failing-bound candidate; default uses config target then expands")
args = p.parse_args()

candidate_script = ROOT / "scripts" / "03_run_aged_candidate.py"
outputs = ROOT / "outputs"
outputs.mkdir(exist_ok=True)
records = []


def eval_rep(n: int) -> bool:
    out = outputs / f"_search_rep_{n:09d}.npz"
    cp = subprocess.run([sys.executable, str(candidate_script), "--repetition", str(n), "--out", str(out)], text=True, capture_output=True)
    print(cp.stdout, end="")
    if cp.returncode != 0:
        print(cp.stderr, file=sys.stderr)
        raise RuntimeError(f"Candidate run failed for repetition {n}")
    import numpy as np
    z = np.load(out, allow_pickle=False)
    failed = bool(z["failed"][0])
    reason = str(z["failure_reason"][0])
    t = float(z["failure_time_s"][0])
    records.append({"repetition": n, "failed": int(failed), "reason": reason, "failure_time_s": t})
    return failed


# Healthy repetition 1 must pass.
if eval_rep(1):
    raise SystemExit("Repetition 1 fails; baseline/failure detector is not valid.")

cfg = json.loads((ROOT / "config.json").read_text())
upper = int(args.start_upper or cfg.get("target_failure_repetition", 50))
upper = max(2, upper)
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

# Save clean neighboring logs.
def copy_search(n: int, dst_name: str):
    src = outputs / f"_search_rep_{n:09d}.npz"
    if not src.exists():
        eval_rep(n)
    shutil.copy2(src, outputs / dst_name)

copy_search(lower, "last_success.npz")
copy_search(upper, "first_failure.npz")

with (outputs / "failure_search.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["repetition", "failed", "reason", "failure_time_s"])
    w.writeheader(); w.writerows(sorted(records, key=lambda r: r["repetition"]))
summary = {
    "note": "Accelerated uncalibrated MVP result; NOT a real Unitree G1 lifetime prediction.",
    "last_success_repetition": lower,
    "first_failure_repetition": upper,
}
(outputs / "failure_summary.json").write_text(json.dumps(summary, indent=2))
print("\nRESULT")
print(json.dumps(summary, indent=2))
