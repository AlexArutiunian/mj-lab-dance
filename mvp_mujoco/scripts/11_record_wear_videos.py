from __future__ import annotations

import argparse
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
from tensordict import TensorDict


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


def _quat_wxyz_to_roll_pitch(q: np.ndarray) -> tuple[float, float]:
    w, x, y, z = map(float, q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _apply_actuator_scales(raw_env, scales: np.ndarray) -> None:
    sim = raw_env.sim
    n = int(scales.shape[0])
    nominal = sim.mj_model.actuator_forcerange[:n].copy()
    scaled = nominal * scales[:, None]
    sim.mj_model.actuator_forcerange[:n] = scaled
    sim.mj_model.actuator_forcelimited[:n] = 1

    device = torch.device(sim.device)
    scaled_t = torch.as_tensor(scaled, dtype=sim.model.actuator_forcerange.dtype, device=device)
    limited_t = torch.ones((n,), dtype=sim.model.actuator_forcelimited.dtype, device=device)
    if len(sim.model.actuator_forcerange.shape) == 3:
        sim.model.actuator_forcerange[0, :n] = scaled_t
    else:
        sim.model.actuator_forcerange[:n] = scaled_t
    if len(sim.model.actuator_forcelimited.shape) == 2:
        sim.model.actuator_forcelimited[0, :n] = limited_t
    else:
        sim.model.actuator_forcelimited[:n] = limited_t
    sim.create_graph()


def _wear_for_target_scale(
    joint_names: list[str],
    target_repetition: int,
    target_scale: float,
    damage_profile: Path,
    floor: float,
    exponent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from wearbench.damage import alpha_for_target_torque_scale, health_from_damage, torque_scale_from_health

    profile = json.loads(damage_profile.read_text())
    rows_by_joint = {r["joint"]: r for r in profile["joints"]}
    severity = np.asarray([rows_by_joint[str(n)]["severity_norm"] for n in joint_names], dtype=np.float64)
    alpha = alpha_for_target_torque_scale(target_repetition, target_scale, floor, exponent)
    damage = target_repetition * alpha * severity
    health = health_from_damage(damage)
    scale = torque_scale_from_health(health, floor, exponent)
    return severity, health, scale


class FastPlayVecEnv:
    def __init__(self, raw_env, clip_actions: float | None = None) -> None:
        self.env = raw_env
        self.clip_actions = clip_actions
        self.num_envs = raw_env.num_envs
        self.device = torch.device(raw_env.device)

    def reset(self) -> TensorDict:
        obs_dict, _ = self.env.reset()
        return TensorDict({"actor": obs_dict["actor"]}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> TensorDict:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        raw = self.env
        raw.action_manager.process_action(actions.to(raw.device))
        for _ in range(raw.cfg.decimation):
            raw._sim_step_counter += 1
            raw.action_manager.apply_action()
            raw.scene.write_data_to_sim()
            raw.sim.step()
            raw.scene.update(dt=raw.physics_dt)

        raw.episode_length_buf += 1
        raw.common_step_counter += 1
        raw.sim.forward()
        raw.command_manager.compute(dt=raw.step_dt)
        raw.sim.sense()
        actor_obs = raw.observation_manager.compute_group("actor", update_history=True)
        raw.obs_buf = {"actor": actor_obs}
        raw.observation_manager._obs_buffer = raw.obs_buf
        return TensorDict(raw.obs_buf, batch_size=[self.num_envs])

    def close(self) -> None:
        self.env.close()


class OnnxPolicy:
    def __init__(self, policy_path: Path) -> None:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(str(policy_path), sess_options=session_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.cuda_enabled = "CUDAExecutionProvider" in self.session.get_providers()
        self.output_tensor: torch.Tensor | None = None
        print(f"[VIDEO] ONNX providers: {self.session.get_providers()}", flush=True)

    def __call__(self, obs: TensorDict) -> torch.Tensor:
        actor = obs["actor"].detach()
        if actor.dtype != torch.float32:
            actor = actor.float()
        actor = actor.contiguous()
        if self.cuda_enabled and actor.is_cuda:
            device_id = actor.device.index if actor.device.index is not None else 0
            if self.output_tensor is None or self.output_tensor.device != actor.device:
                self.output_tensor = torch.empty((1, 29), device=actor.device, dtype=torch.float32)
            binding = self.session.io_binding()
            binding.bind_input(self.input_name, "cuda", device_id, np.float32, tuple(actor.shape), actor.data_ptr())
            binding.bind_output(self.output_name, "cuda", device_id, np.float32, tuple(self.output_tensor.shape), self.output_tensor.data_ptr())
            self.session.run_with_iobinding(binding)
            return self.output_tensor
        actions = self.session.run([self.output_name], {self.input_name: actor.cpu().numpy().astype(np.float32)})[0]
        return torch.from_numpy(actions).to(actor.device)


def _overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    line_h = 18
    w = max(draw.textlength(line) for line in lines) + 20
    h = line_h * len(lines) + 14
    draw.rounded_rectangle((10, 10, 10 + w, 10 + h), radius=4, fill=(0, 0, 0, 150))
    y = 17
    for line in lines:
        draw.text((20, y), line, fill=(255, 255, 255, 255))
        y += line_h
    return np.asarray(img)


def main() -> None:
    _add_runtime_paths()

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.tasks.tracking.mdp import MotionCommandCfg
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig

    p = argparse.ArgumentParser(description="Record healthy/mid/heavy wear dance videos.")
    p.add_argument("--task", default="Unitree-G1-Tracking-No-State-Estimation")
    p.add_argument("--motion-file", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/params/dance1_subject2.npz")
    p.add_argument("--policy", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/exported/policy.onnx")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "wear_videos")
    p.add_argument("--duration", type=float, default=45.0)
    p.add_argument("--video-fps", type=int, default=25)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target-repetition", type=int, default=1_000_000)
    p.add_argument("--damage-profile", type=Path, default=ROOT / "outputs" / "damage_profile.json")
    p.add_argument("--scenarios", default="healthy:1.0,mid_damage:0.5,heavy_damage:0.3")
    args = p.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text())
    floor = float(cfg["torque_scale_floor"])
    exponent = float(cfg["health_exponent"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    policy = OnnxPolicy(args.policy.resolve())
    scenarios: list[tuple[str, float]] = []
    for item in args.scenarios.split(","):
        name, scale_s = item.split(":", 1)
        scenarios.append((name.strip(), float(scale_s)))

    for scenario_name, target_scale in scenarios:
        env_cfg = load_env_cfg(args.task, play=True)
        agent_cfg = load_rl_cfg(args.task)
        env_cfg.scene.num_envs = 1
        env_cfg.events = {}
        env_cfg.terminations = {}
        env_cfg.sim.nconmax = max(env_cfg.sim.nconmax, 128)
        env_cfg.sim.njmax = max(env_cfg.sim.njmax, 512)
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.motion_file = str(args.motion_file.resolve())
        motion_cmd.sampling_mode = "start"
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.joint_position_range = (0.0, 0.0)

        raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        robot = raw_env.scene["robot"]
        joint_names = list(robot.joint_names)
        health = np.ones(len(joint_names), dtype=np.float64)
        scale = np.ones(len(joint_names), dtype=np.float64)
        if target_scale < 0.999999:
            _, health, scale = _wear_for_target_scale(
                joint_names,
                args.target_repetition,
                target_scale,
                args.damage_profile,
                floor,
                exponent,
            )
            _apply_actuator_scales(raw_env, scale)

        weakest = int(np.argmin(scale))
        weakest_joint = joint_names[weakest]
        weakest_health = float(health[weakest])
        weakest_scale = float(scale[weakest])
        env = FastPlayVecEnv(raw_env, clip_actions=agent_cfg.clip_actions)
        obs = env.reset()

        viewer_cfg = ViewerConfig(
            height=args.height,
            width=args.width,
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=4.0,
            elevation=-12.0,
            azimuth=135.0,
            max_extra_envs=0,
        )
        renderer = OffscreenRenderer(model=raw_env.sim.mj_model, cfg=viewer_cfg, scene=raw_env.scene)
        renderer.initialize()

        dt = float(raw_env.step_dt)
        steps = int(math.ceil(args.duration / dt))
        record_every = max(1, int(round((1.0 / args.video_fps) / dt)))
        video_path = args.out_dir / f"{scenario_name}_scale_{target_scale:.1f}.mp4"
        writer = imageio.get_writer(video_path, fps=args.video_fps, codec="libx264", quality=8, macro_block_size=1)
        started = time.perf_counter()
        print(
            "[VIDEO START] "
            f"{scenario_name} target_scale={target_scale:.2f} "
            f"weakest={weakest_joint} health={weakest_health:.4f} scale={weakest_scale:.4f} "
            f"steps={steps} out={video_path}",
            flush=True,
        )
        try:
            last_progress_t = 0.0
            for i in range(steps):
                actions = policy(obs)
                obs = env.step(actions)
                if i % record_every != 0:
                    continue
                renderer.update(raw_env.sim.data)
                frame = renderer.render()
                root_pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
                root_quat = robot.data.root_link_quat_w[0].detach().cpu().numpy()
                roll, pitch = _quat_wxyz_to_roll_pitch(root_quat)
                frame = _overlay(
                    frame,
                    [
                        f"{scenario_name} | t={(i + 1) * dt:.1f}s",
                        f"weakest: {weakest_joint.replace('_joint', '')}",
                        f"health: {weakest_health:.3f}  torque scale: {weakest_scale:.3f}x",
                        f"base z: {float(root_pos[2]):.3f}m  roll/pitch: {math.degrees(roll):.1f}/{math.degrees(pitch):.1f} deg",
                    ],
                )
                writer.append_data(frame)
                sim_t = (i + 1) * dt
                if sim_t - last_progress_t >= 10.0 or i + 1 >= steps:
                    elapsed = time.perf_counter() - started
                    fps = (i + 1) / max(elapsed, 1e-9)
                    eta = (steps - i - 1) / max(fps, 1e-9)
                    print(
                        "[VIDEO PROGRESS] "
                        f"{scenario_name} step={i + 1}/{steps} "
                        f"sim_t={sim_t:.1f}/{args.duration:.1f}s "
                        f"fps={fps:.1f} eta={eta:.1f}s",
                        flush=True,
                    )
                    last_progress_t = sim_t
        finally:
            writer.close()
            renderer.renderer.close()
            env.close()

        print(f"[VIDEO DONE] {scenario_name} -> {video_path}", flush=True)


if __name__ == "__main__":
    main()
