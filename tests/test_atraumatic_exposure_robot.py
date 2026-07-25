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
    def test_phase_contract_is_complete_and_bounded(self) -> None:
        expected = set(MODULE.TOOL_JOINTS.values())
        self.assertEqual(
            set(MODULE.PHASE_TARGETS),
            {
                "stowed", "approach", "deploy", "contact", "capture",
                "retract", "hold", "overload_relief", "release",
            },
        )
        for phase, targets in MODULE.PHASE_TARGETS.items():
            self.assertEqual(set(targets), expected, phase)
            self.assertTrue(0.0 <= targets["left_carriage_joint"] <= 0.040)
            self.assertTrue(-0.040 <= targets["right_carriage_joint"] <= 0.0)
            for name in ("left_lift_joint", "right_lift_joint"):
                self.assertTrue(-0.025 <= targets[name] <= 0.030)
            for name in ("left_compliance_joint", "right_compliance_joint"):
                self.assertTrue(-0.006 <= targets[name] <= 0.0)

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

    def test_controller_is_time_step_consistent(self) -> None:
        slow = MODULE.ForceControlledRetractionController()
        fast = MODULE.ForceControlledRetractionController()
        for _ in range(60):
            slow.update(
                dt=1.0 / 60.0,
                visible_fraction=0.82,
                left_force_n=1.2,
                right_force_n=1.2,
            )
        for _ in range(120):
            fast.update(
                dt=1.0 / 120.0,
                visible_fraction=0.82,
                left_force_n=1.2,
                right_force_n=1.2,
            )
        self.assertAlmostEqual(
            slow.left_carriage_m, fast.left_carriage_m, places=4
        )
        self.assertAlmostEqual(slow.left_lift_m, fast.left_lift_m, places=4)

    def test_force_controller_rejects_nonfinite_or_negative_loads(self) -> None:
        controller = MODULE.ForceControlledRetractionController()
        for kwargs in (
            {"dt": math.nan, "visible_fraction": 0.5, "left_force_n": 1.0, "right_force_n": 1.0},
            {"dt": 0.01, "visible_fraction": math.inf, "left_force_n": 1.0, "right_force_n": 1.0},
            {"dt": 0.01, "visible_fraction": 0.5, "left_force_n": -1.0, "right_force_n": 1.0},
        ):
            with self.assertRaises(ValueError):
                controller.update(**kwargs)

    def test_roi_metrics(self) -> None:
        self.assertAlmostEqual(
            MODULE.ROIExposureEstimator.from_edge_gap(0.022), 0.5
        )
        self.assertEqual(MODULE.ROIExposureEstimator.from_edge_gap(-0.01), 0.0)
        self.assertAlmostEqual(
            MODULE.ROIExposureEstimator.bilateral_balance(0.7, 0.9), 0.8
        )
        with self.assertRaises(ValueError):
            MODULE.ROIExposureEstimator.from_edge_gap(math.nan)
        with self.assertRaises(ValueError):
            MODULE.ROIExposureEstimator.from_edge_gap(0.01, 0.0)

    def test_fail_closed_phase_frame_and_dt_lookup(self) -> None:
        with self.assertRaises(KeyError):
            MODULE.phase_targets("invalid")
        with self.assertRaises(KeyError):
            MODULE.frame_path("/World/Tool", "invalid")
        with self.assertRaises(ValueError):
            MODULE.ForceControlledRetractionController().update(
                dt=0.0,
                visible_fraction=0.5,
                left_force_n=1.0,
                right_force_n=1.0,
            )
        self.assertEqual(MODULE.REGISTERED_CAMERA_FRAMES, ("roi_camera",))


if __name__ == "__main__":
    unittest.main()
