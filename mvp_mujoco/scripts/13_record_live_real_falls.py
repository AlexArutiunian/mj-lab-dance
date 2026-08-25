from __future__ import annotations

import argparse
import csv
import importlib.machinery
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT.parent
DANCE_SIM = BUNDLE / "dance_sim"
MJLAB_REPO = DANCE_SIM / "external" / "unitree_rl_mjlab"


def _add_runtime_paths() -> None:
    py_site = DANCE_SIM / ".venv" / "lib" / "python3.11" / "site-packages"
    nvidia = py_site / "nvidia"
    if nvidia.exists():
        libs = [str(p) for p in nvidia.glob("*/lib") if p.is_dir()]
        import os

        old = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([old] if old else []))
    sys.path.insert(0, str(MJLAB_REPO))
    sys.path.insert(0, str(ROOT))


def _overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    line_h = 18
    w = max(draw.textlength(line) for line in lines) + 20
    h = line_h * len(lines) + 14
    draw.rounded_rectangle((10, 10, 10 + w, 10 + h), radius=4, fill=(0, 0, 0, 160))
    y = 17
    for line in lines:
        draw.text((20, y), line, fill=(255, 255, 255, 255))
        y += line_h
    return np.asarray(img)


def _make_env_cfg(args):
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.tasks.tracking.mdp import MotionCommandCfg

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.scene.env_spacing = float(args.env_spacing)
    env_cfg.seed = int(args.seed)
    env_cfg.events = {}
    env_cfg.terminations = {}
    env_cfg.sim.nconmax = max(env_cfg.sim.nconmax, max(128, args.num_envs * 16))
    env_cfg.sim.njmax = max(env_cfg.sim.njmax, max(512, args.num_envs * 64))
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(args.motion_file.resolve())
    motion_cmd.sampling_mode = "start"
    motion_cmd.motion_start_time_s = float(args.motion_start_time_s)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    return env_cfg


def _quat_wxyz_to_roll_pitch(q: np.ndarray) -> tuple[float, float]:
    w, x, y, z = map(float, q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _parse_env_ids(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _copy_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy().copy()
    return np.asarray(value).copy()


def _make_snapshot(raw_env, robot, pelvis_body_id: int, torso_body_id: int, sim_t: float) -> dict[str, np.ndarray | float]:
    sim_data = raw_env.sim.data
    snapshot: dict[str, np.ndarray | float] = {
        "t": float(sim_t),
        "qpos": _copy_array(sim_data.qpos),
        "qvel": _copy_array(sim_data.qvel),
        "root_pos": _copy_array(robot.data.root_link_pos_w),
        "root_quat": _copy_array(robot.data.root_link_quat_w),
        "pelvis_pos": _copy_array(robot.data.body_com_pos_w[:, pelvis_body_id]),
        "torso_pos": _copy_array(robot.data.body_com_pos_w[:, torso_body_id]),
    }
    if raw_env.sim.mj_model.nmocap > 0:
        snapshot["mocap_pos"] = _copy_array(sim_data.mocap_pos)
        snapshot["mocap_quat"] = _copy_array(sim_data.mocap_quat)
    return snapshot


def _render_snapshot(renderer, snapshot: dict[str, np.ndarray | float], env_id: int, nworld: int, lines: list[str]) -> np.ndarray:
    model = renderer._model
    data = renderer._data
    qpos = snapshot["qpos"]
    qvel = snapshot["qvel"]
    assert isinstance(qpos, np.ndarray)
    assert isinstance(qvel, np.ndarray)
    data.qpos[:] = qpos[env_id]
    data.qvel[:] = qvel[env_id]
    if model.nmocap > 0:
        mocap_pos = snapshot["mocap_pos"]
        mocap_quat = snapshot["mocap_quat"]
        assert isinstance(mocap_pos, np.ndarray)
        assert isinstance(mocap_quat, np.ndarray)
        data.mocap_pos[:] = mocap_pos[env_id]
        data.mocap_quat[:] = mocap_quat[env_id]
    mujoco.mj_forward(model, data)
    renderer.renderer.update_scene(data, camera=renderer._cam)

    for extra_id in renderer._get_extra_env_ids(nworld, env_id):
        data.qpos[:] = qpos[extra_id]
        data.qvel[:] = qvel[extra_id]
        if model.nmocap > 0:
            data.mocap_pos[:] = mocap_pos[extra_id]
            data.mocap_quat[:] = mocap_quat[extra_id]
        mujoco.mj_forward(model, data)
        mujoco.mjv_addGeoms(
            model,
            data,
            renderer._opt,
            renderer._pert,
            renderer._catmask.value,
            renderer.renderer.scene,
        )
    return _overlay(renderer.render(), lines)


def main() -> None:
    _add_runtime_paths()
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig

    runner_mod = importlib.machinery.SourceFileLoader(
        "batched_wear_survival", str(ROOT / "scripts" / "10_run_batched_wear_survival.py")
    ).load_module()

    p = argparse.ArgumentParser(description="Record actual same-run falls using pelvis/torso floor-level criteria.")
    p.add_argument("--task", default="Unitree-G1-Tracking-No-State-Estimation")
    p.add_argument("--motion-file", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/params/dance1_subject2.npz")
    p.add_argument("--policy", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/exported/policy.onnx")
    p.add_argument("--motion-start-time-s", type=float, default=0.0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "live_real_fall_videos")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--duration", type=float, default=45.0)
    p.add_argument("--pre-fall-s", type=float, default=8.0)
    p.add_argument("--post-fall-s", type=float, default=8.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--env-spacing", type=float, default=2.0)
    p.add_argument(
        "--physics-origin-mode",
        choices=("local", "grid"),
        default="local",
        help="local removes coordinate-sensitive batch artifacts; grid reproduces the old layout",
    )
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--max-extra-envs", type=int, default=4)
    p.add_argument("--camera-distance", type=float, default=4.0)
    p.add_argument("--camera-elevation", type=float, default=-12.0)
    p.add_argument("--camera-azimuth", type=float, default=135.0)
    p.add_argument("--target-scale", type=float, default=1.0)
    p.add_argument("--uniform-torque-scale", type=float, default=None)
    p.add_argument("--target-repetition", type=int, default=1_000_000)
    p.add_argument("--damage-profile", type=Path, default=ROOT / "outputs" / "damage_profile.json")
    p.add_argument(
        "--watch-env-ids",
        default="",
        help="Deprecated; pre-fall context is now rendered from saved physics snapshots for any falling env.",
    )
    p.add_argument("--pelvis-height-m", type=float, default=None)
    p.add_argument("--torso-height-m", type=float, default=None)
    p.add_argument("--hold-s", type=float, default=None)
    args = p.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text())
    failure_cfg = cfg.get("failure", {})
    pelvis_height_m = float(args.pelvis_height_m if args.pelvis_height_m is not None else failure_cfg.get("pelvis_height_m", 0.35))
    torso_height_m = float(args.torso_height_m if args.torso_height_m is not None else failure_cfg.get("torso_height_m", 0.30))
    hold_s = float(args.hold_s if args.hold_s is not None else failure_cfg.get("failure_hold_s", 0.5))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = runner_mod.BatchedOnnxPolicy(args.policy.resolve(), args.out_dir / "policy_dynamic_batch.onnx")
    agent_cfg = load_rl_cfg(args.task)
    raw_env = ManagerBasedRlEnv(cfg=_make_env_cfg(args), device=args.device, render_mode=None)
    runner_mod._configure_physics_origins(raw_env, args.physics_origin_mode)
    env = runner_mod.FastPlayVecEnv(raw_env, clip_actions=agent_cfg.clip_actions)
    robot = raw_env.scene["robot"]
    joint_names = list(robot.joint_names)
    health = np.ones(len(joint_names), dtype=np.float64)
    scale = np.ones(len(joint_names), dtype=np.float64)
    if args.target_scale < 0.999999:
        from wearbench.damage import alpha_for_target_torque_scale, health_from_damage, torque_scale_from_health

        profile = json.loads(args.damage_profile.read_text())
        rows_by_joint = {r["joint"]: r for r in profile["joints"]}
        severity = np.asarray([rows_by_joint[str(n)]["severity_norm"] for n in joint_names], dtype=np.float64)
        floor = float(cfg["torque_scale_floor"])
        exponent = float(cfg["health_exponent"])
        alpha = alpha_for_target_torque_scale(int(args.target_repetition), float(args.target_scale), floor, exponent)
        damage = int(args.target_repetition) * alpha * severity
        health = health_from_damage(damage)
        scale = torque_scale_from_health(health, floor, exponent)
        runner_mod._apply_actuator_scales(raw_env, scale)
    if args.uniform_torque_scale is not None:
        scale[:] = float(args.uniform_torque_scale)
        health[:] = float(args.uniform_torque_scale)
        runner_mod._apply_actuator_scales(raw_env, scale)
    weakest = int(np.argmin(scale))
    weakest_joint = str(joint_names[weakest]).replace("_joint", "")
    weakest_health = float(health[weakest])
    weakest_scale = float(scale[weakest])
    wear_label = (
        "baseline scale=1.0 no wear"
        if args.target_scale >= 0.999999 and args.uniform_torque_scale is None
        else f"uniform actuator scale={args.uniform_torque_scale:.2f}x"
        if args.uniform_torque_scale is not None
        else f"wear target scale={args.target_scale:.2f} | weakest {weakest_joint} health={weakest_health:.3f} torque={weakest_scale:.3f}x"
    )
    pelvis_body_id = list(robot.body_names).index("pelvis")
    torso_body_id = list(robot.body_names).index("torso_link")
    obs = env.reset(seed=int(args.seed))

    dt = float(raw_env.step_dt)
    steps = int(math.ceil(args.duration / dt))
    hold_steps = max(1, int(round(hold_s / dt)))
    record_every = max(1, int(round((1.0 / args.video_fps) / dt)))
    pre_frames = max(1, int(round(args.pre_fall_s * args.video_fps)))
    post_steps = max(1, int(round(args.post_fall_s / dt)))

    bad_count = torch.zeros(args.num_envs, dtype=torch.int32, device=raw_env.device)
    latched = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
    writers: dict[int, object] = {}
    renderers: dict[int, object] = {}
    video_paths: dict[int, Path] = {}
    stop_step_by_env: dict[int, int] = {}
    snapshot_buffer: deque[dict[str, np.ndarray | float]] = deque(maxlen=pre_frames)
    events: list[dict[str, object]] = []
    started = time.perf_counter()
    watch_env_ids = _parse_env_ids(args.watch_env_ids)
    for env_id in watch_env_ids:
        if env_id < 0 or env_id >= args.num_envs:
            raise ValueError(f"--watch-env-ids contains {env_id}, but num-envs={args.num_envs}")

    def ensure_renderer(env_id: int) -> None:
        if env_id in renderers:
            return
        viewer_cfg = ViewerConfig(
            height=args.height,
            width=args.width,
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            env_idx=env_id,
            distance=float(args.camera_distance),
            elevation=float(args.camera_elevation),
            azimuth=float(args.camera_azimuth),
            max_extra_envs=max(0, int(args.max_extra_envs)),
        )
        renderer = OffscreenRenderer(raw_env.sim.mj_model, viewer_cfg, raw_env.scene)
        renderer.initialize()
        renderers[env_id] = renderer

    if watch_env_ids:
        print("[WARN] --watch-env-ids is ignored; recording uses all-env qpos/qvel pre-roll snapshots.", flush=True)
    print(
        f"[SNAPSHOTS] pre-roll enabled for any env pre={args.pre_fall_s:.1f}s post={args.post_fall_s:.1f}s "
        f"weakest={weakest_joint} health={weakest_health:.4f} scale={weakest_scale:.4f}",
        flush=True,
    )

    try:
        for i in range(steps):
            obs = env.step(policy(obs))
            sim_t = (i + 1) * dt
            pelvis_z_t = robot.data.body_com_pos_w[:, pelvis_body_id, 2]
            torso_z_t = robot.data.body_com_pos_w[:, torso_body_id, 2]
            low = (pelvis_z_t < pelvis_height_m) & (torso_z_t < torso_height_m)
            bad_count = torch.where(low, bad_count + 1, torch.zeros_like(bad_count))
            newly = (~latched) & (bad_count >= hold_steps)
            if bool(torch.any(newly).detach().cpu()):
                ids = torch.where(newly)[0].detach().cpu().numpy().astype(int).tolist()
                pelvis_z = pelvis_z_t.detach().cpu().numpy()
                torso_z = torso_z_t.detach().cpu().numpy()
                for env_id in ids:
                    if len(events) >= args.top:
                        continue
                    rank = len(events) + 1
                    event = {
                        "rank": rank,
                        "env_id": env_id,
                        "failure_time_s": (i + 1) * dt,
                        "failure_step": i + 1,
                        "pelvis_z": float(pelvis_z[env_id]),
                        "torso_z": float(torso_z[env_id]),
                        "criterion": f"pelvis<{pelvis_height_m:.2f} && torso<{torso_height_m:.2f} for {hold_s:.2f}s",
                    }
                    events.append(event)
                    path = args.out_dir / f"rank_{rank:02d}_env_{env_id:03d}_real_fall_{(i + 1) * dt:05.2f}s.mp4"
                    ensure_renderer(env_id)
                    writers[env_id] = imageio.get_writer(path, fps=args.video_fps, codec="libx264", quality=8, macro_block_size=1)
                    video_paths[env_id] = path
                    stop_step_by_env[env_id] = i + 1 + post_steps
                    for snapshot in snapshot_buffer:
                        root_pos = snapshot["root_pos"]
                        pelvis_pos = snapshot["pelvis_pos"]
                        torso_pos = snapshot["torso_pos"]
                        root_quat = snapshot["root_quat"]
                        assert isinstance(root_pos, np.ndarray)
                        assert isinstance(pelvis_pos, np.ndarray)
                        assert isinstance(torso_pos, np.ndarray)
                        assert isinstance(root_quat, np.ndarray)
                        roll, pitch = _quat_wxyz_to_roll_pitch(root_quat[env_id])
                        writers[env_id].append_data(
                            _render_snapshot(
                                renderers[env_id],
                                snapshot,
                                env_id,
                                args.num_envs,
                                [
                                    f"pre-fall baseline rank {rank} | env {env_id} | t={float(snapshot['t']):.1f}s",
                                    str(event["criterion"]),
                                    f"pelvis z: {pelvis_pos[env_id, 2]:.3f}m  torso z: {torso_pos[env_id, 2]:.3f}m",
                                    f"base z: {root_pos[env_id, 2]:.3f}m  roll/pitch: {math.degrees(roll):.1f}/{math.degrees(pitch):.1f} deg",
                                    wear_label,
                                ],
                            )
                        )
                    print(f"[FALL] rank={rank} env={env_id} t={(i + 1) * dt:.2f}s pelvis={pelvis_z[env_id]:.3f} torso={torso_z[env_id]:.3f}", flush=True)
            latched |= newly

            if i % record_every == 0:
                snapshot = _make_snapshot(raw_env, robot, pelvis_body_id, torso_body_id, sim_t)
                snapshot_buffer.append(snapshot)
                for env_id in sorted(writers):
                    root_pos = snapshot["root_pos"]
                    pelvis_pos = snapshot["pelvis_pos"]
                    torso_pos = snapshot["torso_pos"]
                    root_quat = snapshot["root_quat"]
                    assert isinstance(root_pos, np.ndarray)
                    assert isinstance(pelvis_pos, np.ndarray)
                    assert isinstance(torso_pos, np.ndarray)
                    assert isinstance(root_quat, np.ndarray)
                    roll, pitch = _quat_wxyz_to_roll_pitch(root_quat[env_id])
                    event = next((e for e in events if int(e["env_id"]) == env_id), None)
                    if event is None:
                        continue
                    writers[env_id].append_data(
                        _render_snapshot(
                            renderers[env_id],
                            snapshot,
                            env_id,
                            args.num_envs,
                            [
                            f"same-run real fall rank {event['rank']} | env {env_id} | t={sim_t:.1f}s",
                            str(event["criterion"]),
                            f"pelvis z: {pelvis_pos[env_id, 2]:.3f}m  torso z: {torso_pos[env_id, 2]:.3f}m",
                            f"base z: {root_pos[env_id, 2]:.3f}m  roll/pitch: {math.degrees(roll):.1f}/{math.degrees(pitch):.1f} deg",
                            wear_label,
                            ],
                        )
                    )

            for env_id, stop_step in list(stop_step_by_env.items()):
                if i + 1 >= stop_step:
                    writers[env_id].close()
                    del writers[env_id]
                    del video_paths[env_id]
                    del stop_step_by_env[env_id]
            if len(events) >= args.top and not writers:
                break
            if (i + 1) % max(1, int(round(10.0 / dt))) == 0:
                elapsed = time.perf_counter() - started
                fps = (i + 1) / max(elapsed, 1e-9)
                print(f"[PROGRESS] step={i + 1}/{steps} t={(i + 1) * dt:.1f}s falls={len(events)} active_videos={len(writers)} fps={fps:.1f}", flush=True)
    finally:
        for writer in writers.values():
            # On early interruption, buffered frames are intentionally discarded;
            # normal completion flushes them in timestamp order.
            writer.close()
        for renderer in renderers.values():
            renderer.renderer.close()
        env.close()

    if events:
        with (args.out_dir / "fall_events.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(events[0].keys()))
            writer.writeheader()
            writer.writerows(events)
    print(f"[DONE] falls={len(events)} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
