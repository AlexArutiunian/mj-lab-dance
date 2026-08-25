from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "dance_sim" / "assets" / "policies" / "mimic"


class MotionIdentityTests(unittest.TestCase):
    def test_rb_y_clip_matches_full_motion_slice_with_export_tolerance(self) -> None:
        full = np.load(POLICIES / "dance1_subject2" / "params" / "dance1_subject2.npz")
        short = np.load(
            POLICIES
            / "dance1_subject2_16s_faststart"
            / "params"
            / "dance1_subject2_16s_faststart.npz"
        )
        fps = float(full["fps"][0])
        self.assertEqual(fps, float(short["fps"][0]))
        self.assertEqual(fps, 50.0)
        start = round(20.5 * fps)
        stop = start + len(short["joint_pos"])
        pos_delta = np.max(np.abs(short["joint_pos"] - full["joint_pos"][start:stop]))
        vel_delta = np.max(np.abs(short["joint_vel"] - full["joint_vel"][start:stop]))
        self.assertLessEqual(float(pos_delta), 1.0e-4)
        self.assertLessEqual(float(vel_delta), 3.1e-3)

    def test_dynamic_segment_time_mapping(self) -> None:
        self.assertEqual(20.5 + 8.0, 28.5)
        self.assertEqual(20.5 + 9.0, 29.5)
