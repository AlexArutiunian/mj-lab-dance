"""USER ADAPTER FOR YOUR EXISTING MUJOCO G1 ENVIRONMENT.

Copy this file to user_env.py and implement the four required functions.
The rest of the MVP scripts should then work without knowing your controller.
"""
from __future__ import annotations

import math
from typing import Tuple

import mujoco
import numpy as np


# -----------------------------------------------------------------------------
# REQUIRED HOOK 1
# -----------------------------------------------------------------------------
def make_model_and_data() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """Load your G1 MJCF/XML and return (model, data)."""
    # Example:
    # model = mujoco.MjModel.from_xml_path("/path/to/g1_scene.xml")
    # data = mujoco.MjData(model)
    # return model, data
    raise NotImplementedError("Fill make_model_and_data() in user_env.py")


# -----------------------------------------------------------------------------
# REQUIRED HOOK 2
# -----------------------------------------------------------------------------
def reset_for_dance(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the robot at the exact initial state expected by the dance controller."""
    mujoco.mj_resetData(model, data)
    # Set qpos/qvel/keyframe/etc. here if your dance does not start from XML default.
    # mujoco.mj_forward(model, data)


# -----------------------------------------------------------------------------
# REQUIRED HOOK 3
# -----------------------------------------------------------------------------
def compute_control(t: float, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the actuator ctrl vector for time t during the 2-min dance."""
    # Plug your existing reference trajectory / policy / PD controller here.
    # Must return shape (model.nu,).
    raise NotImplementedError("Fill compute_control() in user_env.py")


# -----------------------------------------------------------------------------
# REQUIRED HOOK 4
# -----------------------------------------------------------------------------
def tracking_error(t: float, model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Return a scalar tracking error. For the very first smoke test, 0.0 is allowed."""
    return 0.0


# -----------------------------------------------------------------------------
# OPTIONAL HOOKS
# -----------------------------------------------------------------------------
def root_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Default assumes freejoint root translation is qpos[0:3]. Override if needed."""
    return float(data.qpos[2])


def root_roll_pitch(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    """Return roll, pitch in radians. Default assumes freejoint quaternion qpos[3:7] = wxyz."""
    if model.nq < 7:
        return 0.0, 0.0
    w, x, y, z = map(float, data.qpos[3:7])
    # quaternion -> roll/pitch
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    return roll, pitch


def dance_duration_s() -> float:
    return 120.0
