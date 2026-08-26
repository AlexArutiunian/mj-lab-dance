#!/usr/bin/env python3
"""Run Unitree's exported G1 velocity policy in the native MuJoCo scene."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml


ROOT = Path(__file__).resolve().parents[1]
UNITREE = ROOT / "external" / "unitree_rl_mjlab"
DEFAULT_POLICY = UNITREE / "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
DEFAULT_DEPLOY = UNITREE / "deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml"
DEFAULT_MODEL = UNITREE / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"


def _quat_inverse_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    xyz = q[1:]
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) - q[0] * vector)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--torque-scale", type=float, default=1.0)
    parser.add_argument("--failure-pelvis-height", type=float, default=0.45)
    parser.add_argument("--failure-torso-height", type=float, default=0.55)
    parser.add_argument("--failure-hold-s", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.duration <= 0.0 or not 0.0 < args.torque_scale <= 1.0:
        raise ValueError("--duration must be positive and --torque-scale must be in (0, 1]")
    cfg = yaml.safe_load(args.deploy.read_text())
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    policy = ort.InferenceSession(str(args.policy.resolve()), providers=["CPUExecutionProvider"])
    if policy.get_inputs()[0].shape != [1, 98]:
        raise ValueError("Expected the Unitree G1 velocity policy with a [1, 98] observation input")
    default_q = np.asarray(cfg["default_joint_pos"], dtype=np.float64)
    kp = np.asarray(cfg["stiffness"], dtype=np.float64)
    kd = np.asarray(cfg["damping"], dtype=np.float64)
    action_cfg = cfg["actions"]["JointPositionAction"]
    action_scale = np.asarray(action_cfg["scale"], dtype=np.float64)
    action_offset = np.asarray(action_cfg["offset"], dtype=np.float64)
    control_dt = float(cfg["step_dt"])
    substeps = int(round(control_dt / model.opt.timestep))
    if not math.isclose(substeps * model.opt.timestep, control_dt, abs_tol=1e-12):
        raise ValueError("Control period is not divisible by the MuJoCo timestep")
    command = np.asarray([args.vx, args.vy, args.yaw_rate], dtype=np.float32)
    command_ranges = cfg["commands"]["base_velocity"]["ranges"]
    minimum = np.asarray([command_ranges["lin_vel_x"][0], command_ranges["lin_vel_y"][0], command_ranges["ang_vel_z"][0]])
    maximum = np.asarray([command_ranges["lin_vel_x"][1], command_ranges["lin_vel_y"][1], command_ranges["ang_vel_z"][1]])
    if np.any(command < minimum) or np.any(command > maximum):
        raise ValueError(f"Velocity command {command.tolist()} is outside {minimum.tolist()}..{maximum.tolist()}")
    data.qpos[:] = model.qpos0
    data.qpos[:3] = (0.0, 0.0, 0.8)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:] = default_q
    mujoco.mj_forward(model, data)
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    gyro_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    gyro_adr = int(model.sensor_adr[gyro_sensor])
    phase = 0.0
    phase_increment = control_dt / float(cfg["observations"]["gait_phase"]["params"]["period"])
    last_action = np.zeros(model.nu, dtype=np.float32)
    torque_limits = model.actuator_ctrlrange * args.torque_scale
    steps = int(math.ceil(args.duration / control_dt))
    hold_steps = max(1, int(math.ceil(args.failure_hold_s / control_dt)))
    min_pelvis, min_torso, bad_steps, failure_step = float("inf"), float("inf"), 0, -1
    failed = False
    trace: dict[str, list[np.ndarray]] = {"obs": [], "actions": [], "qpos": [], "qvel": [], "ctrl": []}

    def step(index: int) -> None:
        nonlocal phase, last_action, min_pelvis, min_torso, bad_steps, failure_step, failed
        phase = (phase + phase_increment) % 1.0
        root_q = data.qpos[3:7].copy()
        projected_gravity = _quat_inverse_rotate(root_q, np.asarray([0.0, 0.0, -1.0]))
        gait_phase = np.asarray([math.sin(phase * 2.0 * math.pi), math.cos(phase * 2.0 * math.pi)])
        obs = np.concatenate(
            (
                data.sensordata[gyro_adr : gyro_adr + 3],
                projected_gravity,
                command,
                gait_phase,
                data.qpos[7:] - default_q,
                data.qvel[6:],
                last_action,
            )
        ).astype(np.float32)[None, :]
        if obs.shape != (1, 98):
            raise RuntimeError(f"Unexpected velocity observation shape: {obs.shape}")
        action = policy.run(None, {policy.get_inputs()[0].name: obs})[0][0].astype(np.float32)
        target_q = action.astype(np.float64) * action_scale + action_offset
        last_action = action
        for _ in range(substeps):
            torque = kp * (target_q - data.qpos[7:]) - kd * data.qvel[6:]
            data.ctrl[:] = np.clip(torque, torque_limits[:, 0], torque_limits[:, 1])
            mujoco.mj_step(model, data)
        pelvis_z, torso_z = float(data.xipos[pelvis_body, 2]), float(data.xipos[torso_body, 2])
        min_pelvis, min_torso = min(min_pelvis, pelvis_z), min(min_torso, torso_z)
        bad_steps = bad_steps + 1 if pelvis_z < args.failure_pelvis_height and torso_z < args.failure_torso_height else 0
        if not failed and bad_steps >= hold_steps:
            failed, failure_step = True, index + 1
        if args.trace is not None:
            for name, value in (("obs", obs[0]), ("actions", action), ("qpos", data.qpos), ("qvel", data.qvel), ("ctrl", data.ctrl)):
                trace[name].append(value.copy())

    started = time.perf_counter()
    if args.viewer:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            for index in range(steps):
                tick = time.perf_counter()
                step(index)
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value, mujoco.mjtGridPos.mjGRID_TOPLEFT.value, "Policy\nCommand\nStep\nStatus\nTorque scale", f"Velocity walk\n{args.vx:.2f}, {args.vy:.2f}, {args.yaw_rate:.2f}\n{index + 1}/{steps}\n{'FALL' if failed else 'RUNNING'}\n{args.torque_scale:.3f}"))
                viewer.sync()
                remaining = control_dt - (time.perf_counter() - tick)
                if remaining > 0.0:
                    time.sleep(remaining)
                if not viewer.is_running():
                    break
    else:
        for index in range(steps):
            tick = time.perf_counter()
            step(index)
            if args.realtime:
                remaining = control_dt - (time.perf_counter() - tick)
                if remaining > 0.0:
                    time.sleep(remaining)
    elapsed = time.perf_counter() - started
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.trace, **{name: np.asarray(values) for name, values in trace.items()})
    print(f"[NATIVE VELOCITY] command=({args.vx:.3f},{args.vy:.3f},{args.yaw_rate:.3f}) steps={steps} sim_s={steps * control_dt:.2f} wall_s={elapsed:.3f} realtime={steps * control_dt / elapsed:.2f}x min_pelvis={min_pelvis:.6f} min_torso={min_torso:.6f} final_root_z={data.qpos[2]:.6f} failed={int(failed)} failure_t={failure_step * control_dt if failed else -1:.2f} torque_scale={args.torque_scale:.3f}")


if __name__ == "__main__":
    main()
