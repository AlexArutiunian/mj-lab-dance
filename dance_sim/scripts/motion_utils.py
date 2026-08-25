#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np


MOTION_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def load_motion(path: str | Path) -> Dict[str, np.ndarray]:
    p = Path(path)
    with np.load(p, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_motion(path: str | Path, motion: Dict[str, np.ndarray]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **motion)


def fps_of(motion: Dict[str, np.ndarray]) -> float:
    fps = motion.get("fps")
    if fps is None:
        return 50.0
    return float(np.asarray(fps).reshape(-1)[0])


def frame_count(motion: Dict[str, np.ndarray]) -> int:
    if "joint_pos" in motion:
        return int(motion["joint_pos"].shape[0])
    for k in MOTION_KEYS:
        if k in motion:
            return int(motion[k].shape[0])
    raise ValueError("No time-series arrays found")


def slice_motion(motion: Dict[str, np.ndarray], start: int, end: int) -> Dict[str, np.ndarray]:
    n = frame_count(motion)
    if start < 0 or end < start or end > n:
        raise ValueError(f"Bad frame range [{start}, {end}) for {n} frames")
    out: Dict[str, np.ndarray] = {}
    for k, v in motion.items():
        if k == "fps":
            out[k] = v.copy()
        elif hasattr(v, "shape") and v.shape and v.shape[0] == n:
            out[k] = v[start:end].copy()
        else:
            out[k] = v.copy()
    return out


def recompute_joint_vel(motion: Dict[str, np.ndarray]) -> None:
    if "joint_pos" not in motion:
        return
    fps = fps_of(motion)
    q = motion["joint_pos"].astype(np.float32)
    vel = np.zeros_like(q)
    if len(q) > 1:
        vel[:-1] = (q[1:] - q[:-1]) * fps
        vel[-1] = vel[-2]
    motion["joint_vel"] = vel


def describe_motion(name: str, motion: Dict[str, np.ndarray]) -> str:
    fps = fps_of(motion)
    n = frame_count(motion)
    lines = [
        f"{name}",
        f"  frames: {n}",
        f"  fps: {fps:g}",
        f"  duration_s: {n / fps:.3f}",
    ]
    if "joint_pos" in motion:
        q = motion["joint_pos"]
        dq = np.linalg.norm(np.diff(q, axis=0), axis=1) if len(q) > 1 else np.array([])
        lines += [
            f"  joint_pos: shape={q.shape} dtype={q.dtype}",
            f"  joint_delta_norm: mean={dq.mean():.6f} p95={np.percentile(dq, 95):.6f} max={dq.max():.6f}" if dq.size else "  joint_delta_norm: n/a",
        ]
    if "body_pos_w" in motion:
        root = motion["body_pos_w"][:, 0, :]
        step = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1) if len(root) > 1 else np.array([])
        path_xy = float(step.sum()) if step.size else 0.0
        lines += [
            f"  root_start_xyz: {np.array2string(root[0], precision=4)}",
            f"  root_end_xyz: {np.array2string(root[-1], precision=4)}",
            f"  root_path_xy_m: {path_xy:.3f}",
            f"  root_z_minmax: {root[:, 2].min():.4f} .. {root[:, 2].max():.4f}",
        ]
    return "\n".join(lines)


def common_motion_files(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.npz")))
        elif p.suffix == ".npz":
            out.append(p)
    return out
