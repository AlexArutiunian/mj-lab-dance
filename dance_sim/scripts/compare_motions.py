#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from motion_utils import describe_motion, load_motion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()

    a = load_motion(args.a)
    b = load_motion(args.b)
    print(describe_motion(args.a, a))
    print()
    print(describe_motion(args.b, b))
    print()

    n = min(a["joint_pos"].shape[0], b["joint_pos"].shape[0])
    qa = a["joint_pos"][:n]
    qb = b["joint_pos"][:n]
    err = np.linalg.norm(qa - qb, axis=1)
    print(f"joint_pos first {n} frames diff_norm:")
    print(f"  mean={err.mean():.6f} p50={np.percentile(err,50):.6f} p95={np.percentile(err,95):.6f} max={err.max():.6f}")

    if "body_pos_w" in a and "body_pos_w" in b:
        ra = a["body_pos_w"][:n, 0, :]
        rb = b["body_pos_w"][:n, 0, :]
        re = np.linalg.norm(ra - rb, axis=1)
        print("root_pos first frames diff_norm:")
        print(f"  mean={re.mean():.6f} p50={np.percentile(re,50):.6f} p95={np.percentile(re,95):.6f} max={re.max():.6f}")


if __name__ == "__main__":
    main()
