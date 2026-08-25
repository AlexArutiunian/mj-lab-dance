from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class JointActuatorMap:
    actuator_ids: np.ndarray
    joint_ids: np.ndarray
    dof_adrs: np.ndarray
    joint_names: list[str]
    actuator_names: list[str]


def _name(model: mujoco.MjModel, objtype, idx: int, fallback: str) -> str:
    s = mujoco.mj_id2name(model, objtype, int(idx))
    return s if s is not None else fallback


def build_joint_actuator_map(model: mujoco.MjModel) -> JointActuatorMap:
    """Map joint-transmission actuators to 1-DoF joints.

    For a standard G1 MJCF with one actuator per hinge this is the useful subset.
    Actuators using tendons/sites are skipped.
    """
    aids, jids, dofs, jnames, anames = [], [], [], [], []
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        jid = int(model.actuator_trnid[a, 0])
        if jid < 0:
            continue
        dof_adr = int(model.jnt_dofadr[jid])
        if dof_adr < 0:
            continue
        aids.append(a)
        jids.append(jid)
        dofs.append(dof_adr)
        jnames.append(_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid, f"joint_{jid}"))
        anames.append(_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a, f"act_{a}"))
    if not aids:
        raise RuntimeError("No joint-transmission actuators found. Adapt mapping for your MJCF.")
    return JointActuatorMap(
        actuator_ids=np.asarray(aids, dtype=np.int32),
        joint_ids=np.asarray(jids, dtype=np.int32),
        dof_adrs=np.asarray(dofs, dtype=np.int32),
        joint_names=jnames,
        actuator_names=anames,
    )


class ForceRangeDerater:
    """Apply persistent health by shrinking actuator forcerange.

    This is preferable to scaling ctrl for position actuators because it limits
    available actuator effort without changing the desired position reference.
    """

    def __init__(self, model: mujoco.MjModel, amap: JointActuatorMap):
        self.model = model
        self.amap = amap
        self.nominal = model.actuator_forcerange[amap.actuator_ids].copy()
        # Ensure force limiting is enabled for selected actuators.
        self.nominal_limited = model.actuator_forcelimited[amap.actuator_ids].copy()
        model.actuator_forcelimited[amap.actuator_ids] = 1

    def restore(self) -> None:
        self.model.actuator_forcerange[self.amap.actuator_ids] = self.nominal
        self.model.actuator_forcelimited[self.amap.actuator_ids] = self.nominal_limited

    def apply_scales(self, scales: np.ndarray) -> None:
        scales = np.asarray(scales, dtype=np.float64)
        if scales.shape != (len(self.amap.actuator_ids),):
            raise ValueError(f"Expected {len(self.amap.actuator_ids)} scales, got {scales.shape}")
        self.model.actuator_forcerange[self.amap.actuator_ids] = self.nominal * scales[:, None]
        self.model.actuator_forcelimited[self.amap.actuator_ids] = 1


def read_joint_signals(data: mujoco.MjData, amap: JointActuatorMap) -> tuple[np.ndarray, np.ndarray]:
    """Return qvel and generalized actuator torque for mapped joint DoFs."""
    qvel = np.asarray(data.qvel[amap.dof_adrs], dtype=np.float64).copy()
    # qfrc_actuator is generalized actuator force after transmission.
    torque = np.asarray(data.qfrc_actuator[amap.dof_adrs], dtype=np.float64).copy()
    return qvel, torque
