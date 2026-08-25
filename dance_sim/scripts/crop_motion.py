#!/usr/bin/env python3
from __future__ import annotations

import argparse

from motion_utils import fps_of, load_motion, recompute_joint_vel, save_motion, slice_motion


def main() -> None:
    ap = argparse.ArgumentParser(description="Cut a frame/time range from a Unitree motion NPZ.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start-frame", type=int)
    ap.add_argument("--end-frame", type=int, help="exclusive")
    ap.add_argument("--start-sec", type=float)
    ap.add_argument("--end-sec", type=float)
    ap.add_argument("--recompute-joint-vel", action="store_true")
    args = ap.parse_args()

    motion = load_motion(args.input)
    fps = fps_of(motion)
    start = args.start_frame if args.start_frame is not None else int(round((args.start_sec or 0.0) * fps))
    if args.end_frame is not None:
        end = args.end_frame
    elif args.end_sec is not None:
        end = int(round(args.end_sec * fps))
    else:
        raise SystemExit("Provide --end-frame or --end-sec")

    out = slice_motion(motion, start, end)
    if args.recompute_joint_vel:
        recompute_joint_vel(out)
    save_motion(args.output, out)
    print(f"saved {args.output}")
    print(f"frames {start}:{end} ({(end-start)/fps:.3f}s at {fps:g} fps)")


if __name__ == "__main__":
    main()
