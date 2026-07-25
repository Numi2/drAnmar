from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "adaptive_anastomosis_robot.py"
)
SPEC = importlib.util.spec_from_file_location("adaptive_anastomosis_robot", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PhysicalControllerTests(unittest.TestCase):
    def test_patency_pass_failure_and_input_validation(self):
        controller = MODULE.LumenPatencyController()
        report = controller.evaluate(
            [0.0092, 0.0090, 0.0091],
            centerline_offset_m=0.001,
            axis_error_deg=3.0,
        )
        self.assertTrue(report.passed)
        self.assertLessEqual(report.area_fraction, 1.0)
        self.assertFalse(controller.evaluate([0.007], axis_error_deg=1.0).passed)
        with self.assertRaises(ValueError):
            controller.evaluate([float("nan")])

    def test_pressure_decay_model_is_monotonic_and_passes_nominal_case(self):
        controller = MODULE.PressureDecayLeakController()
        sealed = controller.effective_leak_area_m2(
            edge_gap_m=0.0,
            retained_staple_fraction=1.0,
            collar_bond_fraction=1.0,
        )
        open_area = controller.effective_leak_area_m2(
            edge_gap_m=0.001,
            retained_staple_fraction=0.5,
            collar_bond_fraction=0.2,
        )
        self.assertGreater(open_area, sealed)
        controller.pressure_pa = controller.target_pressure_pa
        controller.begin_observation()
        for _ in range(81):
            controller.update(
                0.1,
                edge_gap_m=0.0,
                retained_staple_fraction=1.0,
                collar_bond_fraction=1.0,
            )
        self.assertTrue(controller.complete)
        self.assertTrue(controller.passed)
        self.assertLess(controller.average_leak_ml_min, 2.0)

    def test_capture_force_envelope(self):
        capture = MODULE.BilateralTissueCaptureController(
            "/World/Tool", "/World/Left", "/World/Right"
        )
        self.assertEqual(capture.update_loads(1.6, 1.5)["mode"], "controlled")
        self.assertEqual(capture.update_loads(4.0, 1.5)["mode"], "soft_limit")
        self.assertEqual(capture.update_loads(6.1, 1.5)["mode"], "hard_release")
        with self.assertRaises(ValueError):
            capture.update_loads(float("nan"), 1.0)

    def test_staple_and_collar_threshold_contracts(self):
        retention = MODULE.StapleRingRetentionController()
        retention.register([
            {
                "staple_path": "/World/Staple",
                "attachment_paths": ["/World/Left", "/World/Right"],
                "retained": True,
            }
        ])
        self.assertEqual(retention.apply_loads([1.0]), [])
        with self.assertRaises(ValueError):
            retention.apply_loads([])
        collar = MODULE.ReinforcementCollarBondController()
        bond = MODULE.CollarBond("/World/Collar", [])
        self.assertFalse(collar.apply_sector_load(bond, 0, 0.1))
        collar.bonds.append(bond)
        collar.update(45.0)
        self.assertEqual(bond.cure_fraction, 1.0)
        with self.assertRaises(ValueError):
            collar.apply_sector_load(bond, 16, 0.1)


if __name__ == "__main__":
    unittest.main()
