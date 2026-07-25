from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "atraumatic_exposure_robot.py"
)
SPEC = importlib.util.spec_from_file_location("dranmar_atraumatic_exposure_robot_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AtraumaticExposureRobotTests(unittest.TestCase):
    def test_force_estimate_uses_compression_and_closing_velocity(self) -> None:
        self.assertAlmostEqual(MODULE.estimate_pad_force_n(-0.002), 2.5)
        self.assertAlmostEqual(
            MODULE.estimate_pad_force_n(-0.002, -0.01),
            2.88,
        )
        self.assertEqual(MODULE.estimate_pad_force_n(0.002, 0.01), 0.0)

    def test_force_estimate_rejects_invalid_numeric_inputs(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                MODULE.estimate_pad_force_n(value)
        with self.assertRaises(ValueError):
            MODULE.estimate_pad_force_n(-0.001, stiffness_n_m=-1.0)

    def test_force_controller_hard_overload_unloads(self) -> None:
        controller = MODULE.ForceControlledRetractionController(
            left_carriage_m=0.020,
            right_carriage_m=-0.020,
        )
        output = controller.update(
            dt=1.0 / 120.0,
            visible_fraction=0.5,
            left_force_n=4.2,
            right_force_n=1.0,
        )
        self.assertEqual(output.mode, "hard_overload_relief")
        self.assertTrue(output.overload["left_hard"])
        self.assertLess(output.joint_targets["left_carriage_joint"], 0.020)
        self.assertGreater(output.joint_targets["right_carriage_joint"], -0.020)

    def test_force_controller_rejects_nonfinite_or_negative_loads(self) -> None:
        controller = MODULE.ForceControlledRetractionController()
        for kwargs in (
            {"dt": math.nan, "visible_fraction": 0.5, "left_force_n": 1.0, "right_force_n": 1.0},
            {"dt": 0.01, "visible_fraction": math.inf, "left_force_n": 1.0, "right_force_n": 1.0},
            {"dt": 0.01, "visible_fraction": 0.5, "left_force_n": -1.0, "right_force_n": 1.0},
        ):
            with self.assertRaises(ValueError):
                controller.update(**kwargs)

if __name__ == "__main__":
    unittest.main()
