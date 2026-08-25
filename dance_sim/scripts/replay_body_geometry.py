#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from motion_utils import fps_of, load_motion


def main() -> None:
    ap = argparse.ArgumentParser(description="Lightweight geometric replay from body_pos_w in motion NPZ.")
    ap.add_argument("--motion", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--body-size", type=float, default=18.0)
    args = ap.parse_args()

    motion = load_motion(args.motion)
    if "body_pos_w" not in motion:
        raise SystemExit("motion has no body_pos_w")

    fps = fps_of(motion)
    pos = motion["body_pos_w"]
    root = pos[:, 0, :]
    xyz_min = pos.reshape(-1, 3).min(axis=0)
    xyz_max = pos.reshape(-1, 3).max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    span = float(np.max(xyz_max - xyz_min))
    if span < 1e-3:
        span = 1.0

    plt.ion()
    fig = plt.figure("body geometry replay")
    ax = fig.add_subplot(111, projection="3d")
    scat = ax.scatter([], [], [], s=args.body_size)
    trace, = ax.plot([], [], [], linewidth=1.0)
    title = ax.set_title("")

    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(max(0.0, xyz_min[2] - 0.1), xyz_max[2] + 0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    dt = 1.0 / max(fps * args.speed, 1e-6)
    for i in range(0, len(pos), max(args.stride, 1)):
        p = pos[i]
        scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        trace.set_data(root[: i + 1, 0], root[: i + 1, 1])
        trace.set_3d_properties(root[: i + 1, 2])
        title.set_text(f"{args.motion} | frame {i}/{len(pos)-1} | t={i/fps:.2f}s")
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(dt * max(args.stride, 1))

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
