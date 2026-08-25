from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
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


def _apply_actuator_scales(raw_env, nominal_forcerange: np.ndarray, scales: np.ndarray) -> None:
    from wearbench.mjlab_mapping import apply_joint_ordered_actuator_scales

    apply_joint_ordered_actuator_scales(raw_env, scales, nominal_forcerange=nominal_forcerange)


def _applied_wear_scales(scales: np.ndarray, deadband: float) -> np.ndarray:
    """Ignore physically meaningless sub-threshold torque loss in the live rollout."""
    if deadband <= 0.0:
        return scales
    return np.minimum(1.0, np.asarray(scales, dtype=np.float64) + float(deadband))


class OnnxPolicy:
    def __init__(self, policy_path: Path) -> None:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(policy_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.output_tensor: torch.Tensor | None = None
        self.cuda_enabled = "CUDAExecutionProvider" in self.session.get_providers()
        print(f"[INFO] ONNX providers: {self.session.get_providers()}", flush=True)

    def __call__(self, obs) -> torch.Tensor:
        actor = obs["actor"] if isinstance(obs, (dict, TensorDict)) else obs
        actor = actor.detach()
        if actor.shape[0] != 1:
            return torch.cat([self({"actor": actor[i : i + 1]}).clone() for i in range(actor.shape[0])], dim=0)

        if self.cuda_enabled and actor.is_cuda:
            actor = actor.contiguous()
            if actor.dtype != torch.float32:
                actor = actor.float()
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


class FastPlayVecEnv:
    def __init__(self, raw_env, clip_actions: float | None = None) -> None:
        self.env = raw_env
        self.clip_actions = clip_actions
        self.num_envs = raw_env.num_envs
        self.device = torch.device(raw_env.device)

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

    def reset(self, seed: int | None = None) -> TensorDict:
        obs_dict, _ = self.env.reset(seed=seed)
        return TensorDict({"actor": obs_dict["actor"]}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
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


def main() -> None:
    _add_runtime_paths()

    from wearbench.damage import alpha_for_target_torque_scale, health_from_damage, normalized_severity, torque_scale_from_health

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.tasks.tracking.mdp import MotionCommandCfg

    p = argparse.ArgumentParser(description="Run real repeated dances and update wear from measured per-dance torque*qdot.")
    p.add_argument("--task", default="Unitree-G1-Tracking-No-State-Estimation")
    p.add_argument("--motion-file", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/params/dance1_subject2.npz")
    p.add_argument("--policy", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/exported/policy.onnx")
    p.add_argument("--max-repetitions", type=int, default=100)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "real_repeated_wear")
    p.add_argument("--save-last-logs", action="store_true", help="Save last success/failure per-step traces.")
    p.add_argument("--alpha", type=float, default=None, help="Absolute per-dance damage alpha override.")
    p.add_argument("--alpha-scale", type=float, default=1.0, help="Multiplier applied to the selected alpha.")
    p.add_argument("--target-repetition", type=int, default=None, help="Recompute alpha so the max-severity joint reaches --target-scale before this repetition.")
    p.add_argument("--target-scale", type=float, default=None, help="Torque scale target used with --target-repetition. Default: config target_weakest_joint_torque_scale.")
    p.add_argument("--reuse-env", action="store_true", help="Reuse one mjlab env across repetitions. Faster but can leave state behind; fresh env is the default.")
    p.add_argument("--seed", type=int, default=1, help="Fixed seed used before each repeated dance. Use -1 to keep mjlab random seeding.")
    p.add_argument("--wear-effect-deadband", type=float, default=None, help="Do not apply torque derating to physics until 1-scale exceeds this amount. Health is still accumulated and logged.")
    p.add_argument("--initial-scale-file", type=Path, default=None, help="JSON array of model torque scales to apply before the first repetition.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "summary.csv"
    json_path = args.out_dir / "summary.json"
    state_path = args.out_dir / "latest_state.json"
    summary_path = args.out_dir / "run_summary.json"
    for stale_path in (
        csv_path,
        json_path,
        state_path,
        summary_path,
        args.out_dir / "last_success_trace.npz",
        args.out_dir / "first_failure_trace.npz",
    ):
        stale_path.unlink(missing_ok=True)

    cfg = json.loads((ROOT / "config.json").read_text())
    profile = json.loads((ROOT / "outputs" / "damage_profile.json").read_text())
    if args.alpha is not None:
        alpha = float(args.alpha)
        alpha_source = "cli_alpha"
    elif args.target_repetition is not None:
        alpha = alpha_for_target_torque_scale(
            target_repetition=int(args.target_repetition),
            target_scale=float(args.target_scale if args.target_scale is not None else cfg["target_weakest_joint_torque_scale"]),
            floor=float(cfg["torque_scale_floor"]),
            exponent=float(cfg["health_exponent"]),
        )
        alpha_source = f"target_repetition_{args.target_repetition}"
    else:
        alpha = float(profile["alpha_accelerated"])
        alpha_source = "damage_profile_alpha_accelerated"
    alpha *= float(args.alpha_scale)
    floor = float(cfg["torque_scale_floor"])
    exponent = float(cfg["health_exponent"])
    wear_effect_deadband = float(args.wear_effect_deadband if args.wear_effect_deadband is not None else cfg.get("wear_effect_deadband", 1e-4))
    duration = float(args.duration if args.duration is not None else cfg.get("dance_duration_s", 120.0))
    failure_cfg = cfg.get("failure", {})
    base_height_m = float(failure_cfg.get("base_height_m", 0.35))
    max_abs_roll_limit = math.radians(float(failure_cfg.get("max_abs_roll_deg", 60.0)))
    max_abs_pitch_limit = math.radians(float(failure_cfg.get("max_abs_pitch_deg", 60.0)))
    hold_s = float(failure_cfg.get("failure_hold_s", 0.15))

    agent_cfg = load_rl_cfg(args.task)

    fixed_seed = None if args.seed < 0 else int(args.seed)

    def make_env_cfg():
        env_cfg = load_env_cfg(args.task, play=True)
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
        env_cfg.seed = fixed_seed
        return env_cfg

    def seed_everything() -> None:
        if fixed_seed is None:
            return
        random.seed(fixed_seed)
        np.random.seed(fixed_seed)
        torch.manual_seed(fixed_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fixed_seed)

    policy = OnnxPolicy(args.policy.resolve())
    print(f"[REAL] alpha={alpha:.12g} source={alpha_source} alpha_scale={args.alpha_scale}", flush=True)

    persistent_raw_env = None
    persistent_env = None
    if args.reuse_env:
        seed_everything()
        persistent_raw_env = ManagerBasedRlEnv(cfg=make_env_cfg(), device=args.device, render_mode=None)
        persistent_env = FastPlayVecEnv(persistent_raw_env, clip_actions=agent_cfg.clip_actions)

    seed_everything()
    probe_env = persistent_raw_env or ManagerBasedRlEnv(cfg=make_env_cfg(), device=args.device, render_mode=None)
    joint_names = np.asarray(probe_env.scene["robot"].joint_names, dtype=str)
    nominal_forcerange = probe_env.sim.mj_model.actuator_forcerange.copy()
    dt = float(probe_env.step_dt)
    if persistent_raw_env is None:
        probe_env.close()

    steps = int(math.ceil(duration / dt))
    hold_steps = max(1, int(round(hold_s / dt)))
    damage = np.zeros(len(joint_names), dtype=np.float64)
    health = np.ones(len(joint_names), dtype=np.float64)
    scale = np.ones(len(joint_names), dtype=np.float64)
    if args.initial_scale_file is not None:
        loaded_scale = np.asarray(json.loads(args.initial_scale_file.read_text()), dtype=np.float64)
        if loaded_scale.shape != scale.shape:
            raise ValueError(f"--initial-scale-file shape {loaded_scale.shape} != expected {scale.shape}")
        scale = np.clip(loaded_scale, floor, 1.0)
    records: list[dict[str, object]] = []
    started_all = time.perf_counter()
    last_success_trace: dict[str, np.ndarray] | None = None
    failure_trace: dict[str, np.ndarray] | None = None

    for rep in range(1, args.max_repetitions + 1):
        if persistent_raw_env is None:
            seed_everything()
            raw_env = ManagerBasedRlEnv(cfg=make_env_cfg(), device=args.device, render_mode=None)
            env = FastPlayVecEnv(raw_env, clip_actions=agent_cfg.clip_actions)
        else:
            raw_env = persistent_raw_env
            env = persistent_env
            assert env is not None
        robot = raw_env.scene["robot"]
        dof_adrs = robot.indexing.joint_v_adr.detach().cpu().numpy().astype(np.int64)
        applied_scale = _applied_wear_scales(scale, wear_effect_deadband)
        _apply_actuator_scales(raw_env, nominal_forcerange, applied_scale)
        health_before = health.copy()
        scale_before = scale.copy()
        applied_scale_before = applied_scale.copy()
        weakest_before = int(np.argmin(scale))
        obs = env.reset(seed=fixed_seed)
        severity = np.zeros(len(joint_names), dtype=np.float64)
        prev_power: np.ndarray | None = None
        bad_count = 0
        failed = False
        failure_reason = ""
        failure_time_s = math.nan
        min_base_z = math.inf
        max_abs_roll = 0.0
        max_abs_pitch = 0.0
        times: list[float] = []
        base_zs: list[float] = []
        rolls: list[float] = []
        pitches: list[float] = []
        rep_started = time.perf_counter()

        for i in range(steps):
            actions = policy(obs)
            obs = env.step(actions)

            root_pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
            root_quat = robot.data.root_link_quat_w[0].detach().cpu().numpy()
            roll, pitch = _quat_wxyz_to_roll_pitch(root_quat)
            qvel = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float64)
            tau = raw_env.sim.data.qfrc_actuator[0, dof_adrs].detach().cpu().numpy().astype(np.float64)
            power = np.abs(qvel * tau)
            if prev_power is not None:
                severity += 0.5 * (prev_power + power) * dt
            prev_power = power

            t = (i + 1) * dt
            z = float(root_pos[2])
            min_base_z = min(min_base_z, z)
            max_abs_roll = max(max_abs_roll, abs(float(roll)))
            max_abs_pitch = max(max_abs_pitch, abs(float(pitch)))
            if args.save_last_logs:
                times.append(t)
                base_zs.append(z)
                rolls.append(float(roll))
                pitches.append(float(pitch))

            reason = ""
            if z < base_height_m:
                reason = "base_height"
            elif abs(roll) > max_abs_roll_limit:
                reason = "roll"
            elif abs(pitch) > max_abs_pitch_limit:
                reason = "pitch"
            if reason:
                bad_count += 1
                if bad_count >= hold_steps:
                    failed = True
                    failure_reason = reason
                    failure_time_s = t
                    break
            else:
                bad_count = 0

        completed = not failed
        sev_norm = normalized_severity(severity)
        if completed:
            damage += alpha * sev_norm
            health = health_from_damage(damage)
            scale = torque_scale_from_health(health, floor, exponent)
        applied_scale_after = _applied_wear_scales(scale, wear_effect_deadband)

        weakest_after = int(np.argmin(scale))
        elapsed = time.perf_counter() - rep_started
        rec = {
            "repetition": rep,
            "completed": int(completed),
            "failed": int(failed),
            "failure_reason": failure_reason,
            "failure_time_s": failure_time_s,
            "wall_time_s": elapsed,
            "sim_speed_x": (len(base_zs) * dt if args.save_last_logs else (failure_time_s if failed else duration)) / elapsed,
            "weakest_before": str(joint_names[weakest_before]),
            "health_before": float(health_before[weakest_before]),
            "scale_before": float(scale_before[weakest_before]),
            "applied_scale_before": float(applied_scale_before[weakest_before]),
            "weakest_after": str(joint_names[weakest_after]),
            "health_after": float(health[weakest_after]),
            "scale_after": float(scale[weakest_after]),
            "applied_scale_after": float(applied_scale_after[weakest_after]),
            "min_base_z": float(min_base_z),
            "max_abs_roll_deg": math.degrees(float(max_abs_roll)),
            "max_abs_pitch_deg": math.degrees(float(max_abs_pitch)),
        }
        records.append(rec)
        print(
            "[REAL] "
            f"rep={rep}/{args.max_repetitions} completed={completed} failed={failed} "
            f"reason={failure_reason or '-'} t={failure_time_s if failed else duration:.2f}s "
            f"weakest={rec['weakest_after']} health={rec['health_after']:.3f} "
            f"scale={rec['scale_after']:.3f} wall={elapsed:.1f}s",
            flush=True,
        )

        state_path.write_text(json.dumps({"records": records[-10:], "latest": rec}, indent=2))
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        json_path.write_text(json.dumps(records, indent=2))

        if args.save_last_logs:
            trace = {
                "time": np.asarray(times, dtype=np.float64),
                "base_z": np.asarray(base_zs, dtype=np.float64),
                "roll": np.asarray(rolls, dtype=np.float64),
                "pitch": np.asarray(pitches, dtype=np.float64),
                "health": health.copy(),
                "torque_scale": applied_scale_after.copy(),
                "model_torque_scale": scale.copy(),
                "severity": severity.copy(),
                "joint_names": joint_names,
                "repetition": np.asarray([rep], dtype=np.int64),
                "failed": np.asarray([failed]),
                "failure_time_s": np.asarray([failure_time_s]),
                "failure_reason": np.asarray([failure_reason], dtype=str),
            }
            if completed:
                last_success_trace = trace
                np.savez_compressed(args.out_dir / "last_success_trace.npz", **trace)
            else:
                failure_trace = trace
                np.savez_compressed(args.out_dir / "first_failure_trace.npz", **trace)

        if failed:
            if persistent_raw_env is None:
                env.close()
            break
        if persistent_raw_env is None:
            env.close()

    summary = {
        "mode": "real_repeated_measured_damage",
        "alpha": alpha,
        "alpha_source": alpha_source,
        "alpha_scale": float(args.alpha_scale),
        "seed": fixed_seed,
        "wear_effect_deadband": wear_effect_deadband,
        "max_repetitions_requested": args.max_repetitions,
        "repetitions_run": len(records),
        "failed": bool(records[-1]["failed"]) if records else False,
        "first_failure_repetition": int(records[-1]["repetition"]) if records and records[-1]["failed"] else None,
        "wall_time_s": time.perf_counter() - started_all,
        "csv": str(csv_path),
        "json": str(json_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("[REAL RESULT]")
    print(json.dumps(summary, indent=2), flush=True)
    if persistent_env is not None:
        persistent_env.close()


if __name__ == "__main__":
    main()
