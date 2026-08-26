from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch


@dataclass(frozen=True)
class MjlabJointActuatorMap:
    joint_names: tuple[str, ...]
    joint_ids: np.ndarray
    dof_adrs: np.ndarray
    actuator_ids: np.ndarray
    actuator_names: tuple[str, ...]


def resolve_actuator_ids(actuator_trnid: np.ndarray, joint_ids: np.ndarray) -> np.ndarray:
    """Return one actuator id per joint id, preserving joint order."""
    trnid = np.asarray(actuator_trnid)
    joint_ids = np.asarray(joint_ids, dtype=np.int64)
    result: list[int] = []
    for joint_id in joint_ids:
        matches = np.flatnonzero(trnid[:, 0] == joint_id)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one actuator for joint id {joint_id}, found {matches.tolist()}")
        result.append(int(matches[0]))
    return np.asarray(result, dtype=np.int64)


def degraded_force_limited(nominal: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Enable force limiting only where degradation reduces available torque."""
    nominal = np.asarray(nominal).copy()
    scales = np.asarray(scales, dtype=np.float64)
    if nominal.shape != scales.shape:
        raise ValueError(f"Force-limited flags {nominal.shape} do not match scales {scales.shape}")
    nominal[scales < 1.0 - 1.0e-12] = 1
    return nominal


def build_mjlab_joint_actuator_map(raw_env, entity_name: str = "robot") -> MjlabJointActuatorMap:
    robot = raw_env.scene[entity_name]
    model = raw_env.sim.mj_model
    joint_ids = robot.indexing.joint_ids.detach().cpu().numpy().astype(np.int64)
    dof_adrs = robot.indexing.joint_v_adr.detach().cpu().numpy().astype(np.int64)
    actuator_ids = resolve_actuator_ids(model.actuator_trnid, joint_ids)
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(actuator_id)) or f"actuator_{actuator_id}"
        for actuator_id in actuator_ids
    )
    if len(joint_ids) != len(robot.joint_names) or len(dof_adrs) != len(robot.joint_names):
        raise ValueError("Only one-DoF actuated joints are supported by the current G1 wear mapping")
    return MjlabJointActuatorMap(
        joint_names=tuple(robot.joint_names),
        joint_ids=joint_ids,
        dof_adrs=dof_adrs,
        actuator_ids=actuator_ids,
        actuator_names=actuator_names,
    )


def apply_joint_ordered_actuator_scales(
    raw_env,
    scales: np.ndarray,
    nominal_forcerange: np.ndarray | None = None,
) -> MjlabJointActuatorMap:
    mapping = build_mjlab_joint_actuator_map(raw_env)
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (len(mapping.joint_names),):
        raise ValueError(f"Expected {len(mapping.joint_names)} joint scales, got {scales.shape}")
    if np.all(scales >= 1.0 - 1.0e-12):
        return mapping

    sim = raw_env.sim
    ids = mapping.actuator_ids
    nominal_all = sim.mj_model.actuator_forcerange if nominal_forcerange is None else np.asarray(nominal_forcerange)
    nominal = nominal_all[ids].copy()
    scaled = nominal * scales[:, None]
    limited = degraded_force_limited(sim.mj_model.actuator_forcelimited[ids], scales)
    sim.mj_model.actuator_forcerange[ids] = scaled
    sim.mj_model.actuator_forcelimited[ids] = limited

    device = torch.device(sim.device)
    ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)
    scaled_t = torch.as_tensor(scaled, dtype=sim.model.actuator_forcerange.dtype, device=device)
    limited_t = torch.as_tensor(limited, dtype=sim.model.actuator_forcelimited.dtype, device=device)
    force_range = sim.model.actuator_forcerange
    force_limited = sim.model.actuator_forcelimited
    if force_range.ndim == 3:
        force_range[:, ids_t] = scaled_t.unsqueeze(0)
    else:
        force_range[ids_t] = scaled_t
    if force_limited.ndim == 2:
        force_limited[:, ids_t] = limited_t.unsqueeze(0)
    else:
        force_limited[ids_t] = limited_t
    sim.create_graph()
    return mapping
