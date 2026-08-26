from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "mvp_mujoco" / "scripts" / "14_run_independent_trials.py"
SPEC = importlib.util.spec_from_file_location("independent_trials", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IndependentTrialTests(unittest.TestCase):
    def test_duplicate_devices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct physical device"):
            MODULE._parse_devices("cuda:0,cuda:0")

    def test_wilson_interval_contains_observed_fraction(self) -> None:
        low, high = MODULE.wilson_interval(8, 10)
        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)

    def test_zero_trial_interval_is_nan(self) -> None:
        low, high = MODULE.wilson_interval(0, 0)
        self.assertTrue(low != low)
        self.assertTrue(high != high)
