from __future__ import annotations

import numpy as np


def compute_severity(time_s: np.ndarray, qvel: np.ndarray, torque: np.ndarray) -> np.ndarray:
    """MVP per-joint work-like loading proxy: integral |tau * qdot| dt.

    time_s: [T]
    qvel:   [T, J]
    torque: [T, J]
    returns [J]
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    torque = np.asarray(torque, dtype=np.float64)
    if qvel.shape != torque.shape:
        raise ValueError(f"qvel {qvel.shape} != torque {torque.shape}")
    if qvel.ndim != 2 or time_s.ndim != 1 or len(time_s) != len(qvel):
        raise ValueError("Expected time [T], qvel [T,J], torque [T,J]")
    power_abs = np.abs(qvel * torque)
    return np.trapezoid(power_abs, x=time_s, axis=0)


def normalized_severity(severity: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    severity = np.asarray(severity, dtype=np.float64)
    m = float(np.max(severity)) if severity.size else 0.0
    if m <= eps:
        return np.zeros_like(severity)
    return severity / m


def torque_scale_from_health(health: np.ndarray, floor: float = 0.10, exponent: float = 1.5) -> np.ndarray:
    health = np.clip(np.asarray(health, dtype=np.float64), 0.0, 1.0)
    floor = float(floor)
    exponent = float(exponent)
    return floor + (1.0 - floor) * np.power(health, exponent)


def health_from_damage(damage: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.asarray(damage, dtype=np.float64), 0.0, 1.0)


def damage_after_repetitions(severity_norm: np.ndarray, repetitions_completed: int, alpha: float) -> np.ndarray:
    """Fast-forward irreversible damage using baseline per-dance severity.

    This is the deliberate MVP approximation: damage profile per repetition is assumed
    equal to the healthy baseline profile. Later versions should update damage online.
    """
    r = max(0, int(repetitions_completed))
    return r * float(alpha) * np.asarray(severity_norm, dtype=np.float64)


def alpha_for_target_torque_scale(
    target_repetition: int,
    target_scale: float,
    floor: float = 0.10,
    exponent: float = 1.5,
) -> float:
    """Choose accelerated-aging alpha so max-severity joint reaches target_scale
    immediately BEFORE the target repetition.

    If target repetition is N, we fast-forward through N-1 completed dances, then run N.
    """
    n_completed = max(1, int(target_repetition) - 1)
    floor = float(floor)
    exponent = float(exponent)
    target_scale = float(target_scale)
    if not (floor < target_scale <= 1.0):
        raise ValueError("target_scale must satisfy floor < target_scale <= 1")
    health_target = ((target_scale - floor) / (1.0 - floor)) ** (1.0 / exponent)
    damage_target = 1.0 - health_target
    return damage_target / n_completed
