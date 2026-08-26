#!/usr/bin/env python3
"""Play a deployed Unitree ONNX policy in unitree_rl_mjlab/MuJoCo.

This local glue code reuses the upstream mjlab task/env and feeds obs["actor"]
into the exported deploy ONNX policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict


DANCE_SIM_DIR = Path(__file__).resolve().parents[1]
BUNDLE_DIR = DANCE_SIM_DIR.parent
MVP_DIR = BUNDLE_DIR / "mvp_mujoco"


def _add_repo_to_path(repo: Path) -> None:
    sys.path.insert(0, str(repo))


def _quat_wxyz_to_roll_pitch(q: np.ndarray) -> tuple[float, float]:
    w, x, y, z = map(float, q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _repetition_scales(
    joint_names: list[str],
    damage_profile: Path,
    repetition: int,
    torque_scale_floor: float,
    health_exponent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sys.path.insert(0, str(MVP_DIR))
    from wearbench.damage import damage_after_repetitions, health_from_damage, torque_scale_from_health

    profile = json.loads(damage_profile.read_text())
    rows_by_joint = {r["joint"]: r for r in profile["joints"]}
    severity = np.asarray([rows_by_joint[str(n)]["severity_norm"] for n in joint_names], dtype=np.float64)
    damage = damage_after_repetitions(severity, repetition - 1, float(profile["alpha_accelerated"]))
    health = health_from_damage(damage)
    scale = torque_scale_from_health(health, torque_scale_floor, health_exponent)
    return severity, health, scale


def _apply_actuator_scales(raw_env, scales: np.ndarray) -> None:
    sys.path.insert(0, str(MVP_DIR))
    from wearbench.mjlab_mapping import apply_joint_ordered_actuator_scales

    apply_joint_ordered_actuator_scales(raw_env, scales)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default=str(DANCE_SIM_DIR / "external/unitree_rl_mjlab"),
        help="Path to cloned unitree_rl_mjlab repo.",
    )
    parser.add_argument(
        "--task",
        default="Unitree-G1-Tracking-No-State-Estimation",
    )
    parser.add_argument(
        "--motion-file",
        default=str(
            DANCE_SIM_DIR
            / "assets/policies/mimic/dance1_subject2_16s_faststart/params/dance1_subject2_16s_faststart.npz"
        ),
    )
    parser.add_argument(
        "--policy",
        default=str(
            DANCE_SIM_DIR
            / "assets/policies/mimic/dance1_subject2_16s_faststart/exported/policy.onnx"
        ),
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--viewer", choices=["auto", "native", "viser", "none"], default="native")
    parser.add_argument("--steps", type=int, default=0, help="Headless steps for --viewer none; 0 means run forever in viewer.")
    parser.add_argument("--no-terminations", action="store_true", default=True)
    parser.add_argument("--ort-provider", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--profile", action="store_true", help="Print headless policy/env timing.")
    parser.add_argument("--full-env", action="store_true", help="Use the full RL env step with rewards/critic observations.")
    parser.add_argument("--physics-timestep", type=float, default=None)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--solver-iterations", type=int, default=None)
    parser.add_argument("--ls-iterations", type=int, default=None)
    parser.add_argument("--keep-domain-randomization", action="store_true")
    parser.add_argument("--monitor", action="store_true", help="Print live stability metrics.")
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--failure-base-height", type=float, default=0.35)
    parser.add_argument("--failure-max-abs-roll-deg", type=float, default=60.0)
    parser.add_argument("--failure-max-abs-pitch-deg", type=float, default=60.0)
    parser.add_argument("--repetition", type=int, default=1, help="Apply accelerated wear for virtual dance repetition. 1 is healthy.")
    parser.add_argument("--damage-profile", type=Path, default=MVP_DIR / "outputs" / "damage_profile.json")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    _add_repo_to_path(repo)

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.tasks.tracking.mdp import MotionCommandCfg
    from mjlab.viewer.native.viewer import NativeMujocoViewer as NativeMujocoViewerClass
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    def _set_status_overlay_with_wear(self, viewer) -> None:
        import mujoco

        status = self.get_status()
        capped = " [CAPPED]" if status.capped else ""
        text_1 = "Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT"
        text_2 = (
            f"{self.env_idx + 1}/{self.env.num_envs}\n"
            f"{status.step_count}\n"
            f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
            f"{status.speed_label}\n"
            f"{status.target_realtime:.2f}x\n"
            f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)"
        )
        extra = getattr(self.env, "get_overlay_lines", lambda: [])()
        for label, value in extra:
            text_1 += f"\n{label}"
            text_2 += f"\n{value}"
        overlay = (
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            text_1,
            text_2,
        )
        viewer.set_texts(overlay)

    NativeMujocoViewerClass._set_status_overlay = _set_status_overlay_with_wear

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.nconmax = max(env_cfg.sim.nconmax, 128)
    env_cfg.sim.njmax = max(env_cfg.sim.njmax, 512)
    if args.physics_timestep is not None:
        env_cfg.sim.mujoco.timestep = args.physics_timestep
    if args.decimation is not None:
        env_cfg.decimation = args.decimation
    if args.solver_iterations is not None:
        env_cfg.sim.mujoco.iterations = args.solver_iterations
    if args.ls_iterations is not None:
        env_cfg.sim.mujoco.ls_iterations = args.ls_iterations
    if args.no_terminations:
        env_cfg.terminations = {}
    if not args.keep_domain_randomization:
        env_cfg.events = {}

    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = args.motion_file
    motion_cmd.sampling_mode = "start"
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)

    available = ort.get_available_providers()
    if args.ort_provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif args.ort_provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available and torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )

    session = ort.InferenceSession(args.policy, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] Torch device: {device}")
    print(f"[INFO] ONNX providers: {session.get_providers()}")
    print(f"[INFO] Motion: {args.motion_file}")
    print(f"[INFO] Policy: {args.policy}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    robot = env.scene["robot"]
    wear_health = np.ones(len(robot.joint_names), dtype=np.float64)
    wear_scale = np.ones(len(robot.joint_names), dtype=np.float64)
    if args.repetition < 1:
        raise ValueError("--repetition must be >= 1")
    if args.repetition > 1:
        wear_cfg = json.loads((MVP_DIR / "config.json").read_text())
        _, wear_health, wear_scale = _repetition_scales(
            robot.joint_names,
            args.damage_profile,
            args.repetition,
            float(wear_cfg["torque_scale_floor"]),
            float(wear_cfg["health_exponent"]),
        )
        _apply_actuator_scales(env, wear_scale)
        weakest = int(np.argmin(wear_scale))
        print(
            "[WEAR] "
            f"repetition={args.repetition} weakest={robot.joint_names[weakest]} "
            f"health={wear_health[weakest]:.4f} torque_scale={wear_scale[weakest]:.4f}",
            flush=True,
        )
    else:
        print("[WEAR] repetition=1 healthy health=1.0000 torque_scale=1.0000", flush=True)

    class FastPlayVecEnv:
        def __init__(self, raw_env, clip_actions: float | None = None) -> None:
            self.env = raw_env
            self.clip_actions = clip_actions
            self.num_envs = raw_env.num_envs
            self.device = torch.device(raw_env.device)
            self.env.reset()

        @property
        def cfg(self):
            return self.env.cfg

        @property
        def unwrapped(self):
            return self.env.unwrapped

        def get_observations(self) -> TensorDict:
            obs = self.env.observation_manager.compute_group("actor")
            self.env.observation_manager._obs_buffer = {"actor": obs}
            return TensorDict({"actor": obs}, batch_size=[self.num_envs])

        def reset(self):
            obs_dict, extras = self.env.reset()
            return TensorDict({"actor": obs_dict["actor"]}, batch_size=[self.num_envs]), extras

        def step(self, actions: torch.Tensor):
            if self.clip_actions is not None:
                actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

            raw = self.env
            if actions.is_cuda:
                torch.cuda.synchronize(actions.device)
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

            rew = torch.zeros(raw.num_envs, device=raw.device)
            dones = torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device)
            return TensorDict(raw.obs_buf, batch_size=[self.num_envs]), rew, dones, raw.extras

        def close(self) -> None:
            self.env.close()

    env = (
        RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if args.full_env
        else FastPlayVecEnv(env, clip_actions=agent_cfg.clip_actions)
    )

    class StabilityMonitorVecEnv:
        def __init__(self, wrapped) -> None:
            self.wrapped = wrapped
            self._last_log = time.perf_counter()
            self._step = 0
            self._failure_latched = False
            self._weakest_idx = int(np.argmin(wear_scale))
            self._weakest_joint = str(self.wrapped.env.scene["robot"].joint_names[self._weakest_idx])
            self._weakest_health = float(wear_health[self._weakest_idx])
            self._weakest_scale = float(wear_scale[self._weakest_idx])
            self._last_base_z = float("nan")
            self._last_roll_deg = float("nan")
            self._last_pitch_deg = float("nan")
            self._last_event = "OK"

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        @property
        def env(self):
            return self.wrapped.env

        @property
        def cfg(self):
            return self.wrapped.cfg

        @property
        def unwrapped(self):
            return self.wrapped.unwrapped

        def get_observations(self):
            return self.wrapped.get_observations()

        def reset(self):
            self._last_log = time.perf_counter()
            self._step = 0
            self._failure_latched = False
            return self.wrapped.reset()

        def step(self, actions: torch.Tensor):
            action_nan = bool(torch.isnan(actions).any().detach().cpu())
            out = self.wrapped.step(actions)
            self._step += 1

            raw = self.wrapped.env
            robot = raw.scene["robot"]
            root_pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
            root_quat = robot.data.root_link_quat_w[0].detach().cpu().numpy()
            roll, pitch = _quat_wxyz_to_roll_pitch(root_quat)
            z = float(root_pos[2])
            roll_deg = math.degrees(roll)
            pitch_deg = math.degrees(pitch)

            reason = ""
            if action_nan:
                reason = "action_nan"
            elif z < args.failure_base_height:
                reason = "base_height"
            elif abs(roll_deg) > args.failure_max_abs_roll_deg:
                reason = "roll"
            elif abs(pitch_deg) > args.failure_max_abs_pitch_deg:
                reason = "pitch"
            self._last_base_z = z
            self._last_roll_deg = roll_deg
            self._last_pitch_deg = pitch_deg
            self._last_event = "OK" if not reason else reason

            now = time.perf_counter()
            should_log = args.monitor and (now - self._last_log >= args.monitor_interval)
            if should_log or (reason and not self._failure_latched):
                print(
                    "[MON] "
                    f"step={self._step} t={self._step * raw.step_dt:.2f}s "
                    f"base_z={z:.3f} roll={roll_deg:.1f}deg "
                    f"pitch={pitch_deg:.1f}deg action_nan={action_nan} "
                    f"weakest={self._weakest_joint} health={self._weakest_health:.3f} "
                    f"scale={self._weakest_scale:.3f} "
                    f"event={'OK' if not reason else reason}",
                    flush=True,
                )
                self._last_log = now
            if reason:
                self._failure_latched = True
            return out

        def close(self) -> None:
            self.wrapped.close()

        def get_overlay_lines(self) -> list[tuple[str, str]]:
            return [
                ("Rep", str(args.repetition)),
                ("Weakest", self._weakest_joint.replace("_joint", "")),
                ("Health", f"{self._weakest_health:.3f}"),
                ("Torque", f"{self._weakest_scale:.3f}x"),
                ("Base z", f"{self._last_base_z:.3f}m"),
                ("Roll/Pitch", f"{self._last_roll_deg:.1f}/{self._last_pitch_deg:.1f} deg"),
                ("Wear event", self._last_event),
            ]

    if args.monitor:
        env = StabilityMonitorVecEnv(env)

    class OnnxPolicy:
        def __init__(self) -> None:
            self.cuda_enabled = "CUDAExecutionProvider" in session.get_providers()
            self.output_shape = tuple(session.get_outputs()[0].shape)
            self.output_tensor: torch.Tensor | None = None

        def __call__(self, obs) -> torch.Tensor:
            actor = obs["actor"] if isinstance(obs, (dict, TensorDict)) else obs
            actor = actor.detach()
            if actor.is_cuda:
                torch.cuda.synchronize(actor.device)
            if actor.shape[0] != 1:
                actions = [self(actor[i : i + 1]).clone() for i in range(actor.shape[0])]
                return torch.cat(actions, dim=0)

            if self.cuda_enabled and actor.is_cuda:
                actor = actor.contiguous()
                if actor.dtype != torch.float32:
                    actor = actor.float()

                device_id = actor.device.index if actor.device.index is not None else 0
                out_shape = (
                    actor.shape[0],
                    int(self.output_shape[1]) if len(self.output_shape) > 1 else 29,
                )
                if (
                    self.output_tensor is None
                    or tuple(self.output_tensor.shape) != out_shape
                    or self.output_tensor.device != actor.device
                ):
                    self.output_tensor = torch.empty(out_shape, device=actor.device, dtype=torch.float32)

                io_binding = session.io_binding()
                io_binding.bind_input(
                    name=input_name,
                    device_type="cuda",
                    device_id=device_id,
                    element_type=np.float32,
                    shape=tuple(actor.shape),
                    buffer_ptr=actor.data_ptr(),
                )
                io_binding.bind_output(
                    name=output_name,
                    device_type="cuda",
                    device_id=device_id,
                    element_type=np.float32,
                    shape=out_shape,
                    buffer_ptr=self.output_tensor.data_ptr(),
                )
                session.run_with_iobinding(io_binding)
                return self.output_tensor

            actor_np = actor.cpu().numpy().astype(np.float32)
            actions = session.run([output_name], {input_name: actor_np})[0]
            return torch.from_numpy(actions).to(actor.device)

    policy = OnnxPolicy()

    if args.viewer == "none":
        obs = env.get_observations()
        steps = args.steps or 500
        start = time.perf_counter()
        policy_time = 0.0
        step_time = 0.0
        for step in range(steps):
            t0 = time.perf_counter()
            actions = policy(obs)
            t1 = time.perf_counter()
            obs, _, _, _ = env.step(actions)
            t2 = time.perf_counter()
            policy_time += t1 - t0
            step_time += t2 - t1
        elapsed = time.perf_counter() - start
        print(f"[INFO] Headless ONNX play stepped {steps} frames OK")
        print(f"[INFO] Headless speed: {steps / elapsed:.2f} steps/s ({elapsed:.3f}s)")
        if args.profile:
            print(f"[INFO] Policy avg: {policy_time / steps * 1000.0:.2f} ms/step")
            print(f"[INFO] Env avg: {step_time / steps * 1000.0:.2f} ms/step")
        env.close()
        return

    viewer = args.viewer
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"

    if viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()
    env.close()


if __name__ == "__main__":
    main()
