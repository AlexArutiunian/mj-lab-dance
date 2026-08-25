from __future__ import annotations

import numpy as np
import unittest

from wearbench.damage import (
    alpha_for_target_torque_scale,
    compute_severity,
    damage_after_repetitions,
    health_from_damage,
    normalized_severity,
    torque_scale_from_health,
)


class DamageModelTests(unittest.TestCase):
    def test_constant_power_integrates_with_physical_time(self) -> None:
        time_s = np.array([0.0, 0.5, 1.0])
        qvel = np.ones((3, 2))
        torque = np.array([[2.0, -3.0], [2.0, -3.0], [2.0, -3.0]])
        np.testing.assert_allclose(compute_severity(time_s, qvel, torque), [2.0, 3.0])

    def test_damage_health_and_torque_are_monotone(self) -> None:
        severity = normalized_severity(np.array([10.0, 5.0, 0.0]))
        d10 = damage_after_repetitions(severity, 10, 0.01)
        d20 = damage_after_repetitions(severity, 20, 0.01)
        self.assertTrue(np.all(d20 >= d10))
        self.assertTrue(np.all(health_from_damage(d20) <= health_from_damage(d10)))
        self.assertTrue(
            np.all(
                torque_scale_from_health(health_from_damage(d20))
                <= torque_scale_from_health(health_from_damage(d10))
            )
        )

    def test_accelerated_alpha_hits_declared_pre_dance_target(self) -> None:
        target_repetition = 1_000_000
        target_scale = 0.3
        floor = 0.1
        exponent = 1.5
        alpha = alpha_for_target_torque_scale(target_repetition, target_scale, floor, exponent)
        damage = damage_after_repetitions(np.array([1.0]), target_repetition - 1, alpha)
        scale = torque_scale_from_health(health_from_damage(damage), floor, exponent)
        self.assertAlmostEqual(float(scale[0]), target_scale)

    def test_invalid_target_scale_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            alpha_for_target_torque_scale(100, 0.1, floor=0.1)
