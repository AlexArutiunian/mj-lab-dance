from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .mujoco_degradation import build_joint_actuator_map, read_joint_signals


@dataclass
class FailureConfig:
    base_height_m: float
    max_abs_roll_deg: float
    max_abs_pitch_deg: float
    max_tracking_error: float | None
    failure_hold_s: float


@dataclass
class DanceResult:
    failed: bool
    failure_time_s: float | None
    failure_reason: str | None
    log_path: str | None


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def run_dance(
    env,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration_s: float,
    failure_cfg: FailureConfig,
    out_npz: str | Path | None = None,
) -> DanceResult:
    """Run exactly one dance and log mapped joint signals."""
    amap = build_joint_actuator_map(model)
    env.reset_for_dance(model, data)
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    hold_steps_needed = max(1, int(round(failure_cfg.failure_hold_s / dt)))
    bad_count = 0
    failure_reason = None
    failure_time = None

    times, qvels, torques = [], [], []
    base_zs, rolls, pitches, track_errs = [], [], [], []

    nsteps = int(math.ceil(duration_s / dt))
    for _ in range(nsteps):
        t = float(data.time)
        ctrl = np.asarray(env.compute_control(t, model, data), dtype=np.float64)
        if ctrl.shape != (model.nu,):
            raise ValueError(f"compute_control returned {ctrl.shape}; expected {(model.nu,)}")
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        qvel, tau = read_joint_signals(data, amap)
        z = float(env.root_height(model, data))
        roll, pitch = env.root_roll_pitch(model, data)
        terr = float(env.tracking_error(float(data.time), model, data))

        times.append(float(data.time))
        qvels.append(qvel)
        torques.append(tau)
        base_zs.append(z)
        rolls.append(float(roll))
        pitches.append(float(pitch))
        track_errs.append(terr)

        reason = None
        if z < failure_cfg.base_height_m:
            reason = "base_height"
        elif abs(math.degrees(roll)) > failure_cfg.max_abs_roll_deg:
            reason = "roll"
        elif abs(math.degrees(pitch)) > failure_cfg.max_abs_pitch_deg:
            reason = "pitch"
        elif failure_cfg.max_tracking_error is not None and terr > failure_cfg.max_tracking_error:
            reason = "tracking_error"

        if reason is not None:
            bad_count += 1
            if bad_count >= hold_steps_needed:
                failure_reason = reason
                failure_time = float(data.time)
                break
        else:
            bad_count = 0

    if out_npz is not None:
        out_npz = Path(out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            time=np.asarray(times),
            qvel=np.asarray(qvels),
            torque=np.asarray(torques),
            joint_names=np.asarray(amap.joint_names, dtype=str),
            actuator_ids=amap.actuator_ids,
            joint_ids=amap.joint_ids,
            dof_adrs=amap.dof_adrs,
            base_z=np.asarray(base_zs),
            roll=np.asarray(rolls),
            pitch=np.asarray(pitches),
            tracking_error=np.asarray(track_errs),
            failed=np.asarray([failure_reason is not None]),
            failure_time_s=np.asarray([np.nan if failure_time is None else failure_time]),
            failure_reason=np.asarray(["" if failure_reason is None else failure_reason], dtype=str),
        )

    return DanceResult(
        failed=failure_reason is not None,
        failure_time_s=failure_time,
        failure_reason=failure_reason,
        log_path=None if out_npz is None else str(out_npz),
    )
