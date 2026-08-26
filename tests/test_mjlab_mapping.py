from __future__ import annotations

import numpy as np
import unittest

from mvp_mujoco.wearbench.mjlab_mapping import degraded_force_limited, resolve_actuator_ids


class MjlabMappingTests(unittest.TestCase):
    def test_degraded_force_limited_preserves_healthy_actuators(self) -> None:
        nominal = np.asarray([0, 1, 0, 1], dtype=np.int32)
        scales = np.asarray([1.0, 1.0, 0.8, 0.3])

        resolved = degraded_force_limited(nominal, scales)

        np.testing.assert_array_equal(resolved, np.asarray([0, 1, 1, 1]))

    def test_resolve_actuator_ids_preserves_requested_joint_order(self) -> None:
        actuator_trnid = np.asarray([[30, 0], [10, 0], [20, 0]], dtype=np.int32)

        resolved = resolve_actuator_ids(actuator_trnid, np.asarray([10, 20, 30]))

        np.testing.assert_array_equal(resolved, np.asarray([1, 2, 0]))

    def test_resolve_actuator_ids_rejects_missing_or_duplicate_transmissions(self) -> None:
        invalid_cases = [
            (np.asarray([[10, 0]], dtype=np.int32), np.asarray([20])),
            (np.asarray([[10, 0], [10, 0]], dtype=np.int32), np.asarray([10])),
        ]
        for actuator_trnid, joint_ids in invalid_cases:
            with self.subTest(actuator_trnid=actuator_trnid):
                with self.assertRaisesRegex(ValueError, "exactly one actuator"):
                    resolve_actuator_ids(actuator_trnid, joint_ids)
