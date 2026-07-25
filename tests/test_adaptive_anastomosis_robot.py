from __future__ import annotations

import importlib.util
import math
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


class ContractTests(unittest.TestCase):
    def test_phase_contract_is_complete_and_finite(self):
        expected = {
            "inspect", "capture", "align", "mandrel", "approximate", "evert",
            "staple", "release_capture", "reinforce", "occlude", "pressurize",
            "verify", "complete", "abort",
        }
        self.assertEqual(set(MODULE.PHASE_TARGETS), expected)
        self.assertEqual(len(MODULE.TOOL_JOINTS), 14)
        for phase in expected:
            targets = MODULE.phase_targets(phase)
            self.assertEqual(set(targets), set(MODULE.TOOL_JOINTS.values()))
            self.assertTrue(
                all(math.isfinite(value) for value in targets.values())
            )
        with self.assertRaises(KeyError):
            MODULE.phase_targets("invented")
        with self.assertRaises(KeyError):
            MODULE.frame_path("/World/Tool", "invented")
        self.assertEqual(
            MODULE.REGISTERED_CAMERA_FRAMES, ("camera_left", "camera_right")
        )

    def test_sequence_preserves_pressure_at_verification(self):
        sequence = MODULE.AdaptiveAnastomosisSequenceController()
        sequence.transition("pressurize")
        sequence.leak_test.pressure_pa = 8000.0
        sequence.transition("verify")
        self.assertEqual(sequence.leak_test.pressure_pa, 8000.0)
        self.assertEqual(sequence.leak_test.elapsed_s, 0.0)


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

    def test_lumen_measurement_comes_from_supplied_hollow_wall_nodes(self):
        def segment(x0, x1, offset_y=0.0):
            points = []
            for x in (x0, x1):
                for radius in (0.0096, 0.012):
                    for index in range(32):
                        angle = 2.0 * math.pi * index / 32
                        points.append(
                            (
                                x,
                                offset_y + radius * math.cos(angle),
                                radius * math.sin(angle),
                            )
                        )
            return points

        measured = MODULE.measure_lumen_seam_geometry(
            segment(-0.065, -0.001),
            segment(0.001, 0.065, offset_y=0.0004),
        )
        self.assertEqual(
            measured["source"], "live_world_space_simulation_nodes"
        )
        self.assertAlmostEqual(measured["edge_gap_m"], 0.002, places=6)
        self.assertAlmostEqual(
            measured["centerline_offset_m"], 0.0004, places=6
        )
        self.assertAlmostEqual(
            measured["minimum_radius_m"], 0.0096, places=6
        )

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

    def test_leak_ledger_conserves_every_bucket(self):
        ledger = MODULE.LeakTestLedger(initial_reservoir_ml=10.0)
        self.assertEqual(ledger.reservoir_ml, 10.0)
        self.assertEqual(ledger.inject(3.0), 3.0)
        self.assertEqual(ledger.leak(1.5), 1.5)
        ledger.collect(0.5)
        ledger.spill(0.25)
        ledger.discard(0.125)
        self.assertAlmostEqual(ledger.active_leak_ml, 0.625)
        self.assertAlmostEqual(ledger.conservation_error_ml, 0.0, places=12)

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
