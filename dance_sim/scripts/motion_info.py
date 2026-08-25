#!/usr/bin/env python3
from __future__ import annotations

import argparse

from motion_utils import common_motion_files, describe_motion, load_motion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="NPZ files or directories")
    args = ap.parse_args()

    for path in common_motion_files(args.paths):
        print(describe_motion(str(path), load_motion(path)))
        print()


if __name__ == "__main__":
    main()
