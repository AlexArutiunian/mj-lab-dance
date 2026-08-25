#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from motion_utils import fps_of, load_motion


def main() -> None:
    ap = argparse.ArgumentParser(description="Find where a short motion best matches inside a long motion by joint_pos.")
    ap.add_argument("--long", required=True)
    ap.add_argument("--short", required=True)
    ap.add_argument("--window", type=int, default=120, help="number of first short frames used for matching")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    long = load_motion(args.long)
    short = load_motion(args.short)
    ql = long["joint_pos"]
    qs = short["joint_pos"]
    w = min(args.window, len(qs), len(ql))
    target = qs[:w]

    best: list[tuple[float, int]] = []
    for i in range(0, len(ql) - w + 1, args.stride):
        err = float(np.mean(np.linalg.norm(ql[i : i + w] - target, axis=1)))
        best.append((err, i))
    best.sort()

    fps = fps_of(long)
    print(f"matched first {w} frames of short against long; fps={fps:g}")
    for err, i in best[:10]:
        print(f"start_frame={i} start_sec={i/fps:.3f} mean_joint_err={err:.6f}")


if __name__ == "__main__":
    main()
