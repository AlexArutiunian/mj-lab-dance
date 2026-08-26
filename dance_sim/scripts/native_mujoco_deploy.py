#!/usr/bin/env python3
"""Run the Unitree G1 mimic deploy policy in native MuJoCo.

This follows the C++ deploy loop instead of MJWarp: 50 Hz ONNX inference,
torque-level PD control, and the Unitree simulator's 2 ms physics step.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml


ROOT = Path(__file__).resolve().parents[1]
MVP = ROOT.parent / "mvp_mujoco"
UNITREE = ROOT / "external" / "unitree_rl_mjlab"
DEFAULT_POLICY_DIR = ROOT / "assets/policies/mimic/dance1_subject2_16s_faststart"


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64) / np.dot(q, q)


def axis_quat(axis: int, angle: float) -> np.ndarray:
    q = np.zeros(4, dtype=np.float64)
    q[0] = math.cos(angle / 2.0)
    q[axis + 1] = math.sin(angle / 2.0)
    return q


def yaw_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return axis_quat(2, yaw)


def torso_quat(root: np.ndarray, joints: np.ndarray) -> np.ndarray:
    q = quat_mul(root, axis_quat(2, float(joints[12])))
    q = quat_mul(q, axis_quat(0, float(joints[13])))
    return quat_mul(q, axis_quat(1, float(joints[14])))


def orientation_6d(real_q: np.ndarray, ref_q: np.ndarray, init_q: np.ndarray) -> np.ndarray:
    rel = quat_mul(quat_inv(quat_mul(init_q, ref_q)), real_q)
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, rel)
    rot = matrix.reshape(3, 3).T
    return np.asarray([rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]])


def load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=UNITREE / "src/assets/robots/unitree_g1/xmls/scene_g1.xml")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_DIR / "exported/policy.onnx")
    parser.add_argument("--motion", type=Path, default=DEFAULT_POLICY_DIR / "params/dance1_subject2_16s_faststart.npz")
    parser.add_argument("--deploy", type=Path, default=DEFAULT_POLICY_DIR / "params/deploy.yaml")
    parser.add_argument("--duration", type=float, default=16.24)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="Pace headless mode to wall time.")
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument(
        "--repetition",
        type=int,
        default=1,
        help="Virtual dance index under the accelerated damage profile; 1 is healthy.",
    )
    parser.add_argument(
        "--damage-profile",
        type=Path,
        default=MVP / "outputs/rb_y_16s/damage_profile_1m_scale03.json",
    )
    parser.add_argument("--torque-scale", type=float, default=1.0)
    parser.add_argument(
        "--joint-torque-scale",
        action="append",
        default=[],
        metavar="JOINT=SCALE",
        help="Override available torque for one joint; may be repeated.",
    )
    parser.add_argument("--failure-pelvis-height", type=float, default=0.45)
    parser.add_argument("--failure-torso-height", type=float, default=0.55)
    parser.add_argument("--failure-hold-s", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = load_args()
    cfg = yaml.safe_load(args.deploy.read_text())
    motion = np.load(args.motion)
    motion_q = np.asarray(motion["joint_pos"], dtype=np.float32)
    motion_dq = np.asarray(motion["joint_vel"], dtype=np.float32)
    motion_root_q = np.asarray(motion["body_quat_w"][:, 0], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    policy = ort.InferenceSession(str(args.policy.resolve()), providers=["CPUExecutionProvider"])
    input_name = policy.get_inputs()[0].name
    output_name = policy.get_outputs()[0].name

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

    if not 0.0 < args.torque_scale <= 1.0:
        raise ValueError("--torque-scale must be in (0, 1]")
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
        for i in range(model.nu)
    ]
    if args.repetition < 1:
        raise ValueError("--repetition must be >= 1")
    health = np.ones(model.nu, dtype=np.float64)
    torque_scales = np.full(model.nu, args.torque_scale, dtype=np.float64)
    if args.repetition > 1:
        sys.path.insert(0, str(MVP))
        from wearbench.damage import damage_after_repetitions, health_from_damage, torque_scale_from_health

        profile = json.loads(args.damage_profile.read_text())
        rows = {str(row["joint"]): row for row in profile["joints"]}
        severity = np.asarray([rows[name]["severity_norm"] for name in joint_names], dtype=np.float64)
        health = health_from_damage(
            damage_after_repetitions(severity, args.repetition - 1, float(profile["alpha_accelerated"]))
        )
        health_scales = torque_scale_from_health(health, floor=0.10, exponent=1.5)
        torque_scales *= health_scales
    for spec in args.joint_torque_scale:
        name, separator, raw_scale = spec.partition("=")
        if not separator or name not in joint_names:
            raise ValueError(f"Invalid --joint-torque-scale {spec!r}; expected one of {joint_names}")
        scale = float(raw_scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"Joint torque scale must be in (0, 1], got {scale}")
        torque_scales[joint_names.index(name)] = scale
    effective_ctrlrange = model.actuator_ctrlrange * torque_scales[:, None]
    weakest_id = int(np.argmin(torque_scales))

    data.qpos[:] = model.qpos0
    data.qpos[0:3] = (0.0, 0.0, 0.8)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:] = default_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    init_q = quat_mul(yaw_quat(torso_quat(data.qpos[3:7], data.qpos[7:])), quat_inv(yaw_quat(motion_root_q[0])))
    last_action = np.zeros(model.nu, dtype=np.float32)
    target_q = default_q.copy()
    steps = min(int(math.ceil(args.duration / control_dt)), len(motion_q))
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    gyro_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    gyro_adr = int(model.sensor_adr[gyro_sensor])

    trace: dict[str, list[np.ndarray | float | int]] = {
        "obs": [],
        "actions": [],
        "qpos": [],
        "qvel": [],
        "ctrl": [],
        "ncon": [],
        "pelvis_pos": [],
        "torso_pos": [],
        "torso_quat": [],
    }
    min_pelvis = float("inf")
    min_torso = float("inf")
    failed = False
    failure_step = -1
    bad_steps = 0
    hold_steps = max(1, int(math.ceil(args.failure_hold_s / control_dt)))
    start = time.perf_counter()

    def control_step(step: int) -> None:
        nonlocal last_action, target_q, min_pelvis, min_torso, failed, failure_step, bad_steps
        frame = min(step, len(motion_q) - 1)
        joints = data.qpos[7:].copy()
        real_torso_q = torso_quat(data.qpos[3:7], joints)
        ref_torso_q = torso_quat(motion_root_q[frame], motion_q[frame])
        obs = np.concatenate(
            (
                motion_q[frame],
                motion_dq[frame],
                orientation_6d(real_torso_q, ref_torso_q, init_q),
                data.sensordata[gyro_adr : gyro_adr + 3],
                joints - default_q,
                data.qvel[6:],
                last_action,
            )
        ).astype(np.float32)[None, :]
        action = policy.run([output_name], {input_name: obs})[0][0].astype(np.float32)
        target_q = action.astype(np.float64) * action_scale + action_offset
        last_action = action
        for _ in range(substeps):
            torque = kp * (target_q - data.qpos[7:]) - kd * data.qvel[6:]
            data.ctrl[:] = np.clip(torque, effective_ctrlrange[:, 0], effective_ctrlrange[:, 1])
            mujoco.mj_step(model, data)
        pelvis_z = float(data.xipos[pelvis_body, 2])
        torso_z = float(data.xipos[torso_body, 2])
        min_pelvis = min(min_pelvis, pelvis_z)
        min_torso = min(min_torso, torso_z)
        bad_steps = bad_steps + 1 if pelvis_z < args.failure_pelvis_height and torso_z < args.failure_torso_height else 0
        if not failed and bad_steps >= hold_steps:
            failed = True
            failure_step = step + 1
        if args.trace is not None:
            trace["obs"].append(obs[0].copy())
            trace["actions"].append(action.copy())
            trace["qpos"].append(data.qpos.copy())
            trace["qvel"].append(data.qvel.copy())
            trace["ctrl"].append(data.ctrl.copy())
            trace["ncon"].append(int(data.ncon))
            trace["pelvis_pos"].append(data.xipos[pelvis_body].copy())
            trace["torso_pos"].append(data.xipos[torso_body].copy())
            trace["torso_quat"].append(data.xquat[torso_body].copy())

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            for step in range(steps):
                tick = time.perf_counter()
                control_step(step)
                overlay = (
                    mujoco.mjtFontScale.mjFONTSCALE_150.value,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                    "Backend\nStep\nStatus\nWeakest joint\nTorque scale",
                    f"Native MuJoCo\n{step + 1}/{steps}\n{'FALL' if failed else 'RUNNING'}\n"
                    f"{joint_names[weakest_id]}\n{torque_scales[weakest_id]:.3f}",
                )
                viewer.set_texts(overlay)
                viewer.sync()
                remaining = control_dt - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)
                if not viewer.is_running():
                    break
    else:
        for step in range(steps):
            tick = time.perf_counter()
            control_step(step)
            if args.realtime:
                remaining = control_dt - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)

    elapsed = time.perf_counter() - start
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.trace, **{key: np.asarray(value) for key, value in trace.items()})
    print(
        f"[NATIVE] steps={steps} sim_s={steps * control_dt:.2f} wall_s={elapsed:.3f} "
        f"realtime={steps * control_dt / elapsed:.2f}x min_pelvis={min_pelvis:.6f} "
        f"min_torso={min_torso:.6f} final_root_z={data.qpos[2]:.6f} "
        f"failed={int(failed)} failure_t={failure_step * control_dt if failed else -1:.2f} "
        f"repetition={args.repetition} weakest={joint_names[weakest_id]} "
        f"health={health[weakest_id]:.3f} scale={torque_scales[weakest_id]:.3f}"
    )


if __name__ == "__main__":
    main()
