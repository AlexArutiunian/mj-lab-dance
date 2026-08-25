from __future__ import annotations

import argparse
import csv
import importlib.machinery
import json
import math
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
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
        old = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([old] if old else []))
    sys.path.insert(0, str(MJLAB_REPO))
    sys.path.insert(0, str(ROOT))


def _quat_wxyz_to_roll_pitch_torch(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = q.unbind(dim=-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    return roll, pitch


def _overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    line_h = 18
    width = max(draw.textlength(line) for line in lines) + 20
    height = line_h * len(lines) + 14
    draw.rounded_rectangle((10, 10, 10 + width, 10 + height), radius=4, fill=(0, 0, 0, 160))
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
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    return env_cfg


def _reason_name(code: int) -> str:
    return {1: "pelvis_height"}.get(int(code), "unknown")


def _detect_failures(args, runner_mod, policy):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_rl_cfg

    cfg = json.loads((ROOT / "config.json").read_text())
    failure_cfg = cfg.get("failure", {})
    pelvis_height_m = float(failure_cfg.get("pelvis_height_m", 0.45))
    torso_height_m = float(failure_cfg.get("torso_height_m", 0.55))
    hold_s = float(failure_cfg.get("failure_hold_s", 0.5))

    agent_cfg = load_rl_cfg(args.task)
    raw_env = ManagerBasedRlEnv(cfg=_make_env_cfg(args), device=args.device, render_mode=None)
    env = runner_mod.FastPlayVecEnv(raw_env, clip_actions=agent_cfg.clip_actions)
    robot = raw_env.scene["robot"]
    pelvis_body_id = list(robot.body_names).index("pelvis")
    torso_body_id = list(robot.body_names).index("torso_link")
    dt = float(raw_env.step_dt)
    steps = int(math.ceil(args.duration / dt))
    hold_steps = max(1, int(round(hold_s / dt)))
    obs = env.reset(seed=int(args.seed))

    failed = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
    bad_count = torch.zeros(args.num_envs, dtype=torch.int32, device=raw_env.device)
    events: list[dict[str, object]] = []
    started = time.perf_counter()
    for i in range(steps):
        obs = env.step(policy(obs))
        z = robot.data.root_link_pos_w[:, 2]
        pelvis_z = robot.data.body_com_pos_w[:, pelvis_body_id, 2]
        torso_z = robot.data.body_com_pos_w[:, torso_body_id, 2]
        roll, pitch = _quat_wxyz_to_roll_pitch_torch(robot.data.root_link_quat_w)

        reason = torch.zeros(args.num_envs, dtype=torch.int32, device=raw_env.device)
        fallen_low = (pelvis_z < pelvis_height_m) & (torso_z < torso_height_m)
        reason = torch.where(fallen_low, torch.ones_like(reason), reason)
        bad_count = torch.where(reason > 0, bad_count + 1, torch.zeros_like(bad_count))
        newly_failed = (~failed) & (bad_count >= hold_steps)
        if bool(torch.any(newly_failed).detach().cpu()):
            ids = torch.where(newly_failed)[0].detach().cpu().numpy().astype(int).tolist()
            z_cpu = z.detach().cpu().numpy()
            pelvis_z_cpu = pelvis_z.detach().cpu().numpy()
            torso_z_cpu = torso_z.detach().cpu().numpy()
            roll_cpu = roll.detach().cpu().numpy()
            pitch_cpu = pitch.detach().cpu().numpy()
            reason_cpu = reason.detach().cpu().numpy()
            for env_id in ids:
                events.append(
                    {
                        "rank": 0,
                        "env_id": env_id,
                        "failure_time_s": (i + 1) * dt,
                        "failure_step": i + 1,
                        "reason": _reason_name(int(reason_cpu[env_id])),
                        "base_z": float(z_cpu[env_id]),
                        "pelvis_z": float(pelvis_z_cpu[env_id]),
                        "torso_z": float(torso_z_cpu[env_id]),
                        "roll_deg": float(math.degrees(float(roll_cpu[env_id]))),
                        "pitch_deg": float(math.degrees(float(pitch_cpu[env_id]))),
                    }
                )
        failed |= newly_failed
        if (i + 1) % max(1, int(round(10.0 / dt))) == 0:
            elapsed = time.perf_counter() - started
            fps = (i + 1) / max(elapsed, 1e-9)
            print(
                "[DETECT] "
                f"step={i + 1}/{steps} sim_t={(i + 1) * dt:.1f}/{args.duration:.1f}s "
                f"failed={int(torch.sum(failed).detach().cpu())}/{args.num_envs} fps={fps:.1f}",
                flush=True,
            )
    env.close()
    events.sort(key=lambda r: (float(r["failure_time_s"]), int(r["env_id"])))
    for rank, event in enumerate(events, start=1):
        event["rank"] = rank
    return events[: args.top], events


def _record_selected(args, runner_mod, policy, selected: list[dict[str, object]]) -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig

    agent_cfg = load_rl_cfg(args.task)
    raw_env = ManagerBasedRlEnv(cfg=_make_env_cfg(args), device=args.device, render_mode=None)
    env = runner_mod.FastPlayVecEnv(raw_env, clip_actions=agent_cfg.clip_actions)
    robot = raw_env.scene["robot"]
    pelvis_body_id = list(robot.body_names).index("pelvis")
    torso_body_id = list(robot.body_names).index("torso_link")
    obs = env.reset(seed=int(args.seed))

    dt = float(raw_env.step_dt)
    steps = int(math.ceil(args.duration / dt))
    record_every = max(1, int(round((1.0 / args.video_fps) / dt)))
    selected_by_env = {int(e["env_id"]): e for e in selected}
    renderers = {}
    writers = {}
    paths = {}
    for event in selected:
        env_id = int(event["env_id"])
        rank = int(event["rank"])
        viewer_cfg = ViewerConfig(
            height=args.height,
            width=args.width,
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            env_idx=env_id,
            distance=4.0,
            elevation=-12.0,
            azimuth=135.0,
            max_extra_envs=args.max_extra_envs,
        )
        renderer = OffscreenRenderer(raw_env.sim.mj_model, viewer_cfg, raw_env.scene)
        renderer.initialize()
        path = args.out_dir / f"rank_{rank:02d}_env_{env_id:03d}_{event['reason']}_{float(event['failure_time_s']):05.2f}s.mp4"
        renderers[env_id] = renderer
        writers[env_id] = imageio.get_writer(path, fps=args.video_fps, codec="libx264", quality=8, macro_block_size=1)
        paths[env_id] = path

    started = time.perf_counter()
    print(f"[RECORD] writing {len(writers)} videos to {args.out_dir}", flush=True)
    try:
        for i in range(steps):
            obs = env.step(policy(obs))
            if i % record_every != 0:
                continue
            sim_t = (i + 1) * dt
            root_pos = robot.data.root_link_pos_w.detach().cpu().numpy()
            pelvis_pos = robot.data.body_com_pos_w[:, pelvis_body_id].detach().cpu().numpy()
            torso_pos = robot.data.body_com_pos_w[:, torso_body_id].detach().cpu().numpy()
            root_quat = robot.data.root_link_quat_w.detach().cpu().numpy()
            for env_id, renderer in renderers.items():
                event = selected_by_env[env_id]
                renderer.update(raw_env.sim.data)
                frame = renderer.render()
                roll, pitch = _quat_wxyz_to_roll_pitch_torch(
                    torch.as_tensor(root_quat[env_id : env_id + 1], dtype=torch.float32)
                )
                frame = _overlay(
                    frame,
                    [
                        f"baseline batch failure rank {event['rank']} | env {env_id} | t={sim_t:.1f}s",
                        f"latched failure: {event['reason']} at {float(event['failure_time_s']):.2f}s",
                        f"pelvis z: {pelvis_pos[env_id, 2]:.3f}m  torso z: {torso_pos[env_id, 2]:.3f}m",
                        f"base z: {root_pos[env_id, 2]:.3f}m",
                        f"roll/pitch: {math.degrees(float(roll[0])):.1f}/{math.degrees(float(pitch[0])):.1f} deg",
                        f"num_envs={args.num_envs} spacing={args.env_spacing:.1f}m scale=1.0 no wear",
                    ],
                )
                writers[env_id].append_data(frame)
            if sim_t >= args.duration or sim_t - getattr(_record_selected, "_last_progress", 0.0) >= 10.0:
                elapsed = time.perf_counter() - started
                fps = (i + 1) / max(elapsed, 1e-9)
                print(
                    "[RECORD PROGRESS] "
                    f"step={i + 1}/{steps} sim_t={sim_t:.1f}/{args.duration:.1f}s fps={fps:.1f}",
                    flush=True,
                )
                _record_selected._last_progress = sim_t
    finally:
        for writer in writers.values():
            writer.close()
        for renderer in renderers.values():
            renderer.renderer.close()
        env.close()
    for path in paths.values():
        print(f"[VIDEO] {path}", flush=True)


def main() -> None:
    _add_runtime_paths()
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    runner_mod = importlib.machinery.SourceFileLoader(
        "batched_wear_survival", str(ROOT / "scripts" / "10_run_batched_wear_survival.py")
    ).load_module()

    p = argparse.ArgumentParser(description="Record top-N baseline batch failure videos.")
    p.add_argument("--task", default="Unitree-G1-Tracking-No-State-Estimation")
    p.add_argument("--motion-file", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/params/dance1_subject2.npz")
    p.add_argument("--policy", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/exported/policy.onnx")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "batch_failure_videos")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--duration", type=float, default=45.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--env-spacing", type=float, default=2.0)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--max-extra-envs", type=int, default=4)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = runner_mod.BatchedOnnxPolicy(args.policy.resolve(), args.out_dir / "policy_dynamic_batch.onnx")
    selected, events = _detect_failures(args, runner_mod, policy)
    events_csv = args.out_dir / "failure_events.csv"
    if events:
        with events_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(events[0].keys()))
            writer.writeheader()
            writer.writerows(events)
    print(f"[DETECT RESULT] failures={len(events)} top={len(selected)} csv={events_csv}", flush=True)
    print(json.dumps(selected, indent=2), flush=True)
    if selected:
        _record_selected(args, runner_mod, policy, selected)


if __name__ == "__main__":
    main()
