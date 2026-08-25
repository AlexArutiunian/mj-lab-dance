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


def _patch_onnx_dynamic_batch(src: Path, dst: Path) -> Path:
    import onnx

    dst.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(src))
    for value_info in list(model.graph.input) + list(model.graph.output):
        shape = value_info.type.tensor_type.shape
        if shape.dim:
            shape.dim[0].ClearField("dim_value")
            shape.dim[0].dim_param = "batch"
    onnx.save(model, str(dst))
    return dst


def _quat_wxyz_to_roll_pitch_torch(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = q.unbind(dim=-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    return roll, pitch


def _apply_actuator_scales(raw_env, scales: np.ndarray) -> None:
    from wearbench.mjlab_mapping import apply_joint_ordered_actuator_scales

    apply_joint_ordered_actuator_scales(raw_env, scales)


class FastPlayVecEnv:
    def __init__(self, raw_env, clip_actions: float | None = None) -> None:
        self.env = raw_env
        self.clip_actions = clip_actions
        self.num_envs = raw_env.num_envs
        self.device = torch.device(raw_env.device)

    def reset(self, seed: int | None = None) -> TensorDict:
        obs_dict, _ = self.env.reset(seed=seed)
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


class BatchedOnnxPolicy:
    def __init__(self, policy_path: Path, dynamic_policy_path: Path) -> None:
        import onnxruntime as ort

        patched = _patch_onnx_dynamic_batch(policy_path, dynamic_policy_path)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self.session = ort.InferenceSession(str(patched), sess_options=session_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.cuda_enabled = "CUDAExecutionProvider" in self.session.get_providers()
        self.output_tensor: torch.Tensor | None = None
        print(f"[BATCH] ONNX providers: {self.session.get_providers()}", flush=True)

    def __call__(self, obs: TensorDict) -> torch.Tensor:
        actor = obs["actor"].detach()
        if actor.dtype != torch.float32:
            actor = actor.float()
        actor = actor.contiguous()
        batch = int(actor.shape[0])
        if self.cuda_enabled and actor.is_cuda:
            device_id = actor.device.index if actor.device.index is not None else 0
            binding = self.session.io_binding()
            binding.bind_input(self.input_name, "cuda", device_id, np.float32, tuple(actor.shape), actor.data_ptr())
            binding.bind_output(self.output_name, "cuda", device_id)
            self.session.run_with_iobinding(binding)
            actions = binding.copy_outputs_to_cpu()[0]
            return torch.from_numpy(actions).to(actor.device)
        actions = self.session.run([self.output_name], {self.input_name: actor.cpu().numpy().astype(np.float32)})[0]
        return torch.from_numpy(actions).to(actor.device)


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip().replace("_", "")) for v in value.split(",") if v.strip()]


def _quantiles(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {k: math.nan for k in ("p05", "p50", "p95", "min", "max")}
    return {
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _body_index(body_names: list[str], preferred: str, fallback: int = 0) -> int:
    try:
        return list(body_names).index(preferred)
    except ValueError:
        return fallback


def _configure_physics_origins(raw_env, mode: str) -> None:
    """Keep independent MJWarp worlds in the same local coordinate frame."""
    if mode == "local":
        raw_env.scene.env_origins.zero_()
    elif mode != "grid":
        raise ValueError(f"Unsupported physics origin mode: {mode}")


def main() -> None:
    _add_runtime_paths()

    from wearbench.damage import alpha_for_target_torque_scale, health_from_damage, torque_scale_from_health

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.tasks.tracking.mdp import MotionCommandCfg

    p = argparse.ArgumentParser(description="Batched survival check at analytically fast-forwarded wear checkpoints.")
    p.add_argument("--task", default="Unitree-G1-Tracking-No-State-Estimation")
    p.add_argument("--motion-file", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/params/dance1_subject2.npz")
    p.add_argument("--policy", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/exported/policy.onnx")
    p.add_argument("--motion-start-time-s", type=float, default=0.0)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoints", default="0,1000,10000,100000,1000000")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--target-repetition", type=int, default=1000000)
    p.add_argument("--target-scale", type=float, default=None)
    p.add_argument("--uniform-torque-scale", type=float, default=None)
    p.add_argument("--damage-profile", type=Path, default=ROOT / "outputs" / "damage_profile.json")
    p.add_argument("--pose-xy-jitter-m", type=float, default=0.0)
    p.add_argument("--yaw-jitter-deg", type=float, default=0.0)
    p.add_argument("--joint-jitter-rad", type=float, default=0.0)
    p.add_argument("--env-spacing", type=float, default=6.0)
    p.add_argument(
        "--physics-origin-mode",
        choices=("local", "grid"),
        default="local",
        help="local keeps independent MJWarp worlds at XY=0; grid reproduces the old coordinate-sensitive layout",
    )
    p.add_argument(
        "--allow-invalid-baseline",
        action="store_true",
        help="Diagnostic only: continue even if identical healthy worlds fall or numerically diverge.",
    )
    p.add_argument("--identical-state-tol", type=float, default=1.0e-4)
    p.add_argument("--progress-interval-s", type=float, default=10.0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "batched_wear_survival")
    args = p.parse_args()
    reset_protocol = (
        "stress_jitter"
        if args.pose_xy_jitter_m > 0.0 or args.yaw_jitter_deg > 0.0 or args.joint_jitter_rad > 0.0
        else "deploy_exact"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = json.loads((ROOT / "config.json").read_text())
    profile = json.loads(args.damage_profile.read_text())
    duration = float(args.duration if args.duration is not None else cfg.get("dance_duration_s", 120.0))
    failure_cfg = cfg.get("failure", {})
    pelvis_height_m = float(failure_cfg.get("pelvis_height_m", 0.45))
    torso_height_m = float(failure_cfg.get("torso_height_m", 0.55))
    hold_s = float(failure_cfg.get("failure_hold_s", 0.5))
    floor = float(cfg["torque_scale_floor"])
    exponent = float(cfg["health_exponent"])
    target_scale = float(args.target_scale if args.target_scale is not None else cfg["target_weakest_joint_torque_scale"])
    alpha = alpha_for_target_torque_scale(args.target_repetition, target_scale, floor, exponent)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "summary.csv"
    json_path = args.out_dir / "summary.json"
    for stale in (csv_path, json_path, args.out_dir / "run_summary.json"):
        stale.unlink(missing_ok=True)

    agent_cfg = load_rl_cfg(args.task)
    policy = BatchedOnnxPolicy(args.policy.resolve(), args.out_dir / "policy_dynamic_batch.onnx")
    checkpoints = _parse_int_list(args.checkpoints)
    if 0 not in checkpoints and not args.allow_invalid_baseline:
        raise ValueError("Validated runs must include checkpoint 0 as the healthy control.")
    records: list[dict[str, object]] = []
    started_all = time.perf_counter()

    for checkpoint in checkpoints:
        checkpoint_total_index = checkpoints.index(checkpoint) + 1
        print(
            "[BATCH START] "
            f"checkpoint={checkpoint} ({checkpoint_total_index}/{len(checkpoints)}) "
            f"envs={args.num_envs} duration={duration:.1f}s",
            flush=True,
        )
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
        motion_cmd.pose_range = {
            "x": (-args.pose_xy_jitter_m, args.pose_xy_jitter_m),
            "y": (-args.pose_xy_jitter_m, args.pose_xy_jitter_m),
            "yaw": (-math.radians(args.yaw_jitter_deg), math.radians(args.yaw_jitter_deg)),
        }
        motion_cmd.velocity_range = {}
        motion_cmd.joint_position_range = (-args.joint_jitter_rad, args.joint_jitter_rad)

        raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
        _configure_physics_origins(raw_env, args.physics_origin_mode)
        env = FastPlayVecEnv(raw_env, clip_actions=agent_cfg.clip_actions)
        robot = raw_env.scene["robot"]
        joint_names = np.asarray(robot.joint_names, dtype=str)
        pelvis_body_id = _body_index(list(robot.body_names), "pelvis", fallback=0)
        torso_body_id = _body_index(list(robot.body_names), "torso_link", fallback=pelvis_body_id)
        rows_by_joint = {r["joint"]: r for r in profile["joints"]}
        severity_profile = np.asarray([rows_by_joint[str(n)]["severity_norm"] for n in joint_names], dtype=np.float64)
        damage = float(checkpoint) * alpha * severity_profile
        health = health_from_damage(damage)
        scale = torque_scale_from_health(health, floor, exponent)
        if args.uniform_torque_scale is not None:
            scale[:] = float(args.uniform_torque_scale)
            health[:] = float(args.uniform_torque_scale)
        _apply_actuator_scales(raw_env, scale)

        dt = float(raw_env.step_dt)
        steps = int(math.ceil(duration / dt))
        hold_steps = max(1, int(round(hold_s / dt)))
        obs = env.reset(seed=int(args.seed))
        print(
            "[BATCH RESET] "
            f"checkpoint={checkpoint} steps={steps} dt={dt:.4f}s "
            f"weakest={joint_names[int(np.argmin(scale))]} "
            f"health={float(np.min(health)):.6f} scale={float(np.min(scale)):.6f}",
            flush=True,
        )

        n = int(args.num_envs)
        failed = torch.zeros(n, dtype=torch.bool, device=raw_env.device)
        bad_count = torch.zeros(n, dtype=torch.int32, device=raw_env.device)
        failure_step = torch.full((n,), -1, dtype=torch.int32, device=raw_env.device)
        failure_reason = torch.zeros(n, dtype=torch.int32, device=raw_env.device)
        min_z = torch.full((n,), float("inf"), dtype=torch.float32, device=raw_env.device)
        min_pelvis_z = torch.full((n,), float("inf"), dtype=torch.float32, device=raw_env.device)
        min_torso_z = torch.full((n,), float("inf"), dtype=torch.float32, device=raw_env.device)
        max_roll = torch.zeros(n, dtype=torch.float32, device=raw_env.device)
        max_pitch = torch.zeros(n, dtype=torch.float32, device=raw_env.device)
        max_state_spread = 0.0

        started = time.perf_counter()
        last_progress = started
        last_step = 0
        for i in range(steps):
            actions = policy(obs)
            obs = env.step(actions)

            if (
                args.physics_origin_mode == "local"
                and args.pose_xy_jitter_m == 0.0
                and args.yaw_jitter_deg == 0.0
                and args.joint_jitter_rad == 0.0
                and n > 1
            ):
                qpos = raw_env.sim.data.qpos
                spread = torch.max(torch.abs(qpos - qpos[0:1])).detach().cpu().item()
                max_state_spread = max(max_state_spread, float(spread))

            z = robot.data.root_link_pos_w[:, 2]
            pelvis_z = robot.data.body_com_pos_w[:, pelvis_body_id, 2]
            torso_z = robot.data.body_com_pos_w[:, torso_body_id, 2]
            roll, pitch = _quat_wxyz_to_roll_pitch_torch(robot.data.root_link_quat_w)
            abs_roll = torch.abs(roll)
            abs_pitch = torch.abs(pitch)
            min_z = torch.minimum(min_z, z)
            min_pelvis_z = torch.minimum(min_pelvis_z, pelvis_z)
            min_torso_z = torch.minimum(min_torso_z, torso_z)
            max_roll = torch.maximum(max_roll, abs_roll)
            max_pitch = torch.maximum(max_pitch, abs_pitch)

            reason = torch.zeros(n, dtype=torch.int32, device=raw_env.device)
            fallen_low = (pelvis_z < pelvis_height_m) & (torso_z < torso_height_m)
            reason = torch.where(fallen_low, torch.ones_like(reason), reason)
            bad_count = torch.where(reason > 0, bad_count + 1, torch.zeros_like(bad_count))
            newly_failed = (~failed) & (bad_count >= hold_steps)
            failed |= newly_failed
            failure_step = torch.where(newly_failed, torch.full_like(failure_step, i + 1), failure_step)
            failure_reason = torch.where(newly_failed, reason, failure_reason)
            if bool(torch.all(failed).detach().cpu()):
                break
            now = time.perf_counter()
            if args.progress_interval_s > 0 and now - last_progress >= args.progress_interval_s:
                done_steps = i + 1
                interval_steps = done_steps - last_step
                interval_s = now - last_progress
                avg_fps = done_steps / max(now - started, 1e-9)
                inst_fps = interval_steps / max(interval_s, 1e-9)
                remaining_s = (steps - done_steps) / max(avg_fps, 1e-9)
                failed_now = int(torch.sum(failed).detach().cpu())
                print(
                    "[BATCH PROGRESS] "
                    f"checkpoint={checkpoint} step={done_steps}/{steps} "
                    f"sim_t={done_steps * dt:.1f}/{duration:.1f}s "
                    f"failed={failed_now}/{n} "
                    f"fps={avg_fps:.1f} inst_fps={inst_fps:.1f} "
                    f"eta={remaining_s:.1f}s",
                    flush=True,
                )
                last_progress = now
                last_step = done_steps

        elapsed = time.perf_counter() - started
        failed_cpu = failed.detach().cpu().numpy().astype(bool)
        min_z_cpu = min_z.detach().cpu().numpy()
        min_pelvis_z_cpu = min_pelvis_z.detach().cpu().numpy()
        min_torso_z_cpu = min_torso_z.detach().cpu().numpy()
        max_roll_deg = np.degrees(max_roll.detach().cpu().numpy())
        max_pitch_deg = np.degrees(max_pitch.detach().cpu().numpy())
        reason_cpu = failure_reason.detach().cpu().numpy()
        step_cpu = failure_step.detach().cpu().numpy()
        success = ~failed_cpu
        weakest = int(np.argmin(scale))
        rec = {
            "checkpoint_repetitions": int(checkpoint),
            "num_envs": n,
            "completed_envs": int(np.sum(success)),
            "failed_envs": int(np.sum(failed_cpu)),
            "survival_rate": float(np.mean(success)),
            "pelvis_height_failures": int(np.sum(reason_cpu == 1)),
            "roll_failures": int(np.sum(reason_cpu == 2)),
            "pitch_failures": int(np.sum(reason_cpu == 3)),
            "first_failure_time_s_min": float(np.min(step_cpu[failed_cpu]) * dt) if np.any(failed_cpu) else math.nan,
            "wall_time_s": elapsed,
            "simulated_env_dances_per_wall_s": float(n / elapsed),
            "weakest_joint": str(joint_names[weakest]),
            "weakest_health": float(health[weakest]),
            "weakest_torque_scale": float(scale[weakest]),
            "min_base_z_p05": _quantiles(min_z_cpu)["p05"],
            "min_base_z_p50": _quantiles(min_z_cpu)["p50"],
            "min_base_z_min": _quantiles(min_z_cpu)["min"],
            "min_pelvis_z_p05": _quantiles(min_pelvis_z_cpu)["p05"],
            "min_pelvis_z_p50": _quantiles(min_pelvis_z_cpu)["p50"],
            "min_pelvis_z_min": _quantiles(min_pelvis_z_cpu)["min"],
            "min_torso_z_p05": _quantiles(min_torso_z_cpu)["p05"],
            "min_torso_z_p50": _quantiles(min_torso_z_cpu)["p50"],
            "min_torso_z_min": _quantiles(min_torso_z_cpu)["min"],
            "max_roll_deg_p95": _quantiles(max_roll_deg)["p95"],
            "max_roll_deg_max": _quantiles(max_roll_deg)["max"],
            "max_pitch_deg_p95": _quantiles(max_pitch_deg)["p95"],
            "max_pitch_deg_max": _quantiles(max_pitch_deg)["max"],
            "max_identical_env_state_spread": max_state_spread,
        }
        baseline_validation = "not_baseline"
        if checkpoint == 0:
            if int(rec["failed_envs"]) > 0:
                baseline_validation = "invalid_healthy_fall"
            elif n > 1 and max_state_spread > float(args.identical_state_tol):
                baseline_validation = "invalid_identical_world_divergence"
            else:
                baseline_validation = "valid"
        rec["baseline_validation"] = baseline_validation
        records.append(rec)
        print(
            "[BATCH] "
            f"checkpoint={checkpoint} envs={n} survival={rec['survival_rate']:.4f} "
            f"failed={rec['failed_envs']} weakest={rec['weakest_joint']} "
            f"health={rec['weakest_health']:.6f} scale={rec['weakest_torque_scale']:.6f} "
            f"wall={elapsed:.1f}s",
            flush=True,
        )
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        json_path.write_text(json.dumps(records, indent=2))
        env.close()
        if checkpoint == 0 and baseline_validation != "valid" and not args.allow_invalid_baseline:
            invalid_summary = {
                "mode": "batched_wear_survival",
                "reset_protocol": reset_protocol,
                "valid": False,
                "validation_error": baseline_validation,
                "message": "Healthy-control batch is numerically contaminated; worn checkpoints were not evaluated.",
                "num_envs": n,
                "physics_origin_mode": args.physics_origin_mode,
                "failed_envs": int(rec["failed_envs"]),
                "max_identical_env_state_spread": max_state_spread,
                "identical_state_tolerance": float(args.identical_state_tol),
                "csv": str(csv_path),
                "json": str(json_path),
            }
            (args.out_dir / "run_summary.json").write_text(json.dumps(invalid_summary, indent=2))
            raise RuntimeError(invalid_summary["message"])

    summary = {
        "mode": "batched_wear_survival",
        "num_envs": int(args.num_envs),
        "checkpoints": checkpoints,
        "duration_s": duration,
        "device": args.device,
        "motion_start_time_s": float(args.motion_start_time_s),
        "reset_protocol": reset_protocol,
        "alpha": alpha,
        "target_repetition": int(args.target_repetition),
        "target_scale": target_scale,
        "uniform_torque_scale": args.uniform_torque_scale,
        "damage_profile": str(args.damage_profile.resolve()),
        "pose_xy_jitter_m": float(args.pose_xy_jitter_m),
        "yaw_jitter_deg": float(args.yaw_jitter_deg),
        "joint_jitter_rad": float(args.joint_jitter_rad),
        "env_spacing": float(args.env_spacing),
        "physics_origin_mode": args.physics_origin_mode,
        "valid": True,
        "baseline_validation": "valid",
        "identical_state_tolerance": float(args.identical_state_tol),
        "wall_time_s": time.perf_counter() - started_all,
        "csv": str(csv_path),
        "json": str(json_path),
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print("[BATCH RESULT]")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
