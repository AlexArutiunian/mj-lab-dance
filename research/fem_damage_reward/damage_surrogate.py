"""Scaffold for a FEM-calibrated impact-damage surrogate.

This module intentionally contains no fitted physical coefficients.  It defines
an interface that can be integrated into the fast simulator before a real
high-fidelity dataset exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ImpactFeatures:
    force_peak_n: float
    impulse_norm_ns: float
    impact_energy_proxy_j: float
    contact_duration_s: float
    foot_speed_mps: float
    foot_angular_speed_radps: float
    base_speed_mps: float
    base_angular_speed_radps: float


@dataclass(frozen=True)
class DamagePrediction:
    value: float
    in_domain: bool
    model_version: str


class LinearDamageSurrogate:
    """Transparent baseline surrogate.

    This is an engineering/plumbing baseline, not the final scientific model.
    Coefficients must come from a declared fitted dataset.  No defaults are
    provided because invented coefficients would look like physical constants.
    """

    FEATURE_ORDER: Sequence[str] = (
        "force_peak_n",
        "impulse_norm_ns",
        "impact_energy_proxy_j",
        "contact_duration_s",
        "foot_speed_mps",
        "foot_angular_speed_radps",
        "base_speed_mps",
        "base_angular_speed_radps",
    )

    def __init__(
        self,
        coefficients: Mapping[str, float],
        *,
        intercept: float,
        feature_min: Mapping[str, float],
        feature_max: Mapping[str, float],
        model_version: str,
    ) -> None:
        missing = [name for name in self.FEATURE_ORDER if name not in coefficients]
        if missing:
            raise ValueError(f"missing coefficients: {missing}")

        bounds_missing = [
            name
            for name in self.FEATURE_ORDER
            if name not in feature_min or name not in feature_max
        ]
        if bounds_missing:
            raise ValueError(f"missing feature bounds: {bounds_missing}")

        self._coef = dict(coefficients)
        self._intercept = float(intercept)
        self._feature_min = dict(feature_min)
        self._feature_max = dict(feature_max)
        self._model_version = str(model_version)

    def predict(self, x: ImpactFeatures) -> DamagePrediction:
        values = {name: float(getattr(x, name)) for name in self.FEATURE_ORDER}

        if not all(isfinite(v) for v in values.values()):
            raise ValueError("impact features must be finite")

        in_domain = all(
            self._feature_min[name] <= values[name] <= self._feature_max[name]
            for name in self.FEATURE_ORDER
        )

        y = self._intercept
        for name in self.FEATURE_ORDER:
            y += self._coef[name] * values[name]

        # A negative structural-damage target is not useful for the reward.
        # This clamp is a model-interface convention, not a material law.
        y = max(0.0, y)

        return DamagePrediction(
            value=y,
            in_domain=in_domain,
            model_version=self._model_version,
        )


def damage_reward_penalty(
    prediction: DamagePrediction,
    *,
    reward_scale: float,
    reject_ood: bool = True,
) -> float:
    """Return a non-negative penalty to subtract from the RL reward."""

    if reward_scale < 0.0:
        raise ValueError("reward_scale must be non-negative")

    if reject_ood and not prediction.in_domain:
        raise ValueError("DAMAGE_SURROGATE_OOD")

    return reward_scale * prediction.value
