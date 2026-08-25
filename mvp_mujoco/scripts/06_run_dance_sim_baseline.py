from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT.parent
DANCE_SIM = BUNDLE / "dance_sim"
MJLAB_REPO = DANCE_SIM / "external" / "unitree_rl_mjlab"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_runtime_paths() -> None:
    py_site = DANCE_SIM / ".venv" / "lib" / "python3.11" / "site-packages"
    nvidia = py_site / "nvidia"
    if nvidia.exists():
        libs = [str(p) for p in nvidia.glob("*/lib") if p.is_dir()]
        old = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([old] if old else []))
    sys.path.insert(0, str(MJLAB_REPO))


def _quat_wxyz_to_roll_pitch(q: np.ndarray) -> tuple[float, float]:
    w, x, y, z = map(float, q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def _repetition_scales(
    joint_names: np.ndarray,
    damage_profile: Path,
    repetition: int,
    torque_scale_floor: float,
    health_exponent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sys.path.insert(0, str(ROOT))
    from wearbench.damage import damage_after_repetitions, health_from_damage, torque_scale_from_health

    profile = json.loads(damage_profile.read_text())
    rows_by_joint = {r["joint"]: r for r in profile["joints"]}
    severity = np.asarray([rows_by_joint[str(n)]["severity_norm"] for n in joint_names], dtype=np.float64)
    damage = damage_after_repetitions(severity, repetition - 1, float(profile["alpha_accelerated"]))
    health = health_from_damage(damage)
    scale = torque_scale_from_health(health, torque_scale_floor, health_exponent)
    return severity, health, scale


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
    sim.model.actuator_forcerange[:n] = scaled_t
    sim.model.actuator_forcelimited[:n] = limited_t
    sim.create_graph()


class OnnxPolicy:
    def __init__(self, policy_path: Path) -> None:
        import onnxruntime as ort

        self.ort = ort
        available = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        self.session = self.ort.InferenceSession(str(policy_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.output_tensor: torch.Tensor | None = None
        self.cuda_enabled = "CUDAExecutionProvider" in self.session.get_providers()
        print(f"[INFO] ONNX providers: {self.session.get_providers()}")

    def __call__(self, obs) -> torch.Tensor:
        actor = obs["actor"] if isinstance(obs, dict) else obs["actor"]
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
            binding.bind_input(
                self.input_name,
                "cuda",
                device_id,
                np.float32,
                tuple(actor.shape),
                actor.data_ptr(),
            )
            binding.bind_output(
                self.output_name,
                "cuda",
                device_id,
                np.float32,
                tuple(self.output_tensor.shape),
                self.output_tensor.data_ptr(),
            )
            self.session.run_with_iobinding(binding)
            return self.output_tensor

        actor_np = actor.cpu().numpy().astype(np.float32)
        actions = self.session.run([self.output_name], {self.input_name: actor_np})[0]
        return torch.from_numpy(actions).to(actor.device)


def main() -> None:
    _add_runtime_paths()

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.tasks.tracking.mdp import MotionCommandCfg

    p = argparse.ArgumentParser(description="Run dance_sim ONNX playback and write WearBench baseline_dance.npz.")
    p.add_argument("--task", default="Unitree-G1-Tracking-No-State-Estimation")
    p.add_argument("--motion-file", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/params/dance1_subject2.npz")
    p.add_argument("--policy", type=Path, default=DANCE_SIM / "assets/policies/mimic/dance1_subject2/exported/policy.onnx")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "baseline_dance.npz")
    p.add_argument("--duration", type=float, default=None, help="Seconds to log. Default: config dance_duration_s.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--keep-domain-randomization", action="store_true")
    p.add_argument("--no-stop-on-failure", action="store_true", help="Keep logging after failure threshold is crossed.")
    p.add_argument("--repetition", type=int, default=1, help="Virtual dance repetition after accelerated damage. 1 is healthy.")
    p.add_argument("--damage-profile", type=Path, default=ROOT / "outputs" / "damage_profile.json")
    args = p.parse_args()

    cfg_path = ROOT / "config.json"
    cfg = json.loads(cfg_path.read_text())
    duration = float(args.duration if args.duration is not None else cfg.get("dance_duration_s", 120.0))
    failure_cfg = cfg.get("failure", {})
    pelvis_height_m = float(failure_cfg.get("pelvis_height_m", 0.35))
    torso_height_m = float(failure_cfg.get("torso_height_m", 0.30))
    hold_s = float(failure_cfg.get("failure_hold_s", 0.15))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = int(args.seed)
    if not args.keep_domain_randomization:
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

    policy = OnnxPolicy(args.policy.resolve())
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    robot = raw_env.scene["robot"]
    pelvis_body_id = list(robot.body_names).index("pelvis")
    torso_body_id = list(robot.body_names).index("torso_link")

    dt = float(raw_env.step_dt)
    steps = int(math.ceil(duration / dt))
    joint_names = np.asarray(robot.joint_names, dtype=str)
    actuator_names = np.asarray(robot.actuator_names, dtype=str)
    severity = np.zeros(len(joint_names), dtype=np.float64)
    health = np.ones(len(joint_names), dtype=np.float64)
    torque_scale = np.ones(len(joint_names), dtype=np.float64)
    if args.repetition < 1:
        raise ValueError("--repetition must be >= 1")
    if args.repetition > 1:
        severity, health, torque_scale = _repetition_scales(
            joint_names,
            args.damage_profile,
            args.repetition,
            float(cfg["torque_scale_floor"]),
            float(cfg["health_exponent"]),
        )
        _apply_actuator_scales(raw_env, torque_scale)
        weakest = int(np.argmin(torque_scale))
        print(
            "[INFO] aged repetition "
            f"{args.repetition}: weakest={joint_names[weakest]} "
            f"health={health[weakest]:.4f} torque_scale={torque_scale[weakest]:.4f}"
        )

    times: list[float] = []
    qvels: list[np.ndarray] = []
    torques: list[np.ndarray] = []
    base_z: list[float] = []
    rolls: list[float] = []
    pitches: list[float] = []
    tracking_error: list[float] = []
    pelvis_zs: list[float] = []
    torso_zs: list[float] = []
    failed = False
    failure_time_s = math.nan
    failure_reason = ""
    bad_count = 0
    hold_steps = max(1, int(round(hold_s / dt)))

    obs = env.get_observations()
    started = time.perf_counter()
    for i in range(steps):
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        root_pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
        root_quat = robot.data.root_link_quat_w[0].detach().cpu().numpy()
        roll, pitch = _quat_wxyz_to_roll_pitch(root_quat)
        pelvis_z = float(robot.data.body_com_pos_w[0, pelvis_body_id, 2].detach().cpu())
        torso_z = float(robot.data.body_com_pos_w[0, torso_body_id, 2].detach().cpu())

        times.append((i + 1) * dt)
        qvels.append(robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float64))
        torques.append(raw_env.sim.data.qfrc_actuator[0, 6 : 6 + len(joint_names)].detach().cpu().numpy().astype(np.float64))
        base_z.append(float(root_pos[2]))
        rolls.append(float(roll))
        pitches.append(float(pitch))
        pelvis_zs.append(pelvis_z)
        torso_zs.append(torso_z)

        cmd = raw_env.command_manager.get_command("motion")
        terr = 0.0
        if hasattr(cmd, "metrics") and "error_body_pos" in cmd.metrics:
            terr = float(cmd.metrics["error_body_pos"][0].detach().cpu())
        tracking_error.append(terr)

        reason = ""
        if pelvis_z < pelvis_height_m and torso_z < torso_height_m:
            reason = "floor_level_fall"
        if reason:
            bad_count += 1
            if not failed and bad_count >= hold_steps:
                failed = True
                failure_time_s = (i + 1) * dt
                failure_reason = reason
                print(f"[WARN] baseline failure: reason={failure_reason} t={failure_time_s:.3f}s")
                if not args.no_stop_on_failure:
                    break
        else:
            bad_count = 0

        if (i + 1) % max(1, steps // 10) == 0:
            print(f"[INFO] logged {i + 1}/{steps} steps")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        time=np.asarray(times, dtype=np.float64),
        qvel=np.asarray(qvels, dtype=np.float64),
        torque=np.asarray(torques, dtype=np.float64),
        joint_names=joint_names,
        actuator_names=actuator_names,
        actuator_ids=np.arange(len(joint_names), dtype=np.int32),
        joint_ids=np.arange(len(joint_names), dtype=np.int32),
        dof_adrs=np.arange(len(joint_names), dtype=np.int32),
        severity=severity,
        health=health,
        torque_scale=torque_scale,
        repetition=np.asarray([args.repetition], dtype=np.int64),
        base_z=np.asarray(base_z, dtype=np.float64),
        roll=np.asarray(rolls, dtype=np.float64),
        pitch=np.asarray(pitches, dtype=np.float64),
        tracking_error=np.asarray(tracking_error, dtype=np.float64),
        pelvis_z=np.asarray(pelvis_zs, dtype=np.float64),
        torso_z=np.asarray(torso_zs, dtype=np.float64),
        failed=np.asarray([failed]),
        failure_time_s=np.asarray([failure_time_s]),
        failure_reason=np.asarray([failure_reason], dtype=str),
        source=np.asarray(["dance_sim_mjlab_onnx"], dtype=str),
        seed=np.asarray([args.seed], dtype=np.int64),
        motion_file=np.asarray([str(args.motion_file.resolve())], dtype=str),
        motion_sha256=np.asarray([_sha256(args.motion_file.resolve())], dtype=str),
        policy_file=np.asarray([str(args.policy.resolve())], dtype=str),
        policy_sha256=np.asarray([_sha256(args.policy.resolve())], dtype=str),
    )
    elapsed = time.perf_counter() - started
    print(f"[INFO] wrote {args.out}")
    print(f"[INFO] speed {len(times) / elapsed:.2f} steps/s, logged duration {len(times) * dt:.2f}s")
    env.close()


if __name__ == "__main__":
    main()
