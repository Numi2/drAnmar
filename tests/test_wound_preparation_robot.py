"""Dependency-free contract tests for the wound-preparation controller."""
from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "wound_preparation_robot.py"
)
SPEC = importlib.util.spec_from_file_location("dranmar_wound_preparation_robot", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FluidLedgerTests(unittest.TestCase):
    def test_conserves_volume_across_all_sinks(self) -> None:
        ledger = MODULE.FluidLedger(reservoir_capacity_ml=10.0, reservoir_ml=10.0)
        self.assertEqual(ledger.emit(4.0), 4.0)
        self.assertEqual(ledger.aspirate(1.5), 1.5)
        self.assertEqual(ledger.mark_spilled(1.0), 1.0)
        self.assertEqual(ledger.discard(1.5), 1.5)
        self.assertAlmostEqual(ledger.balance_error_ml, 0.0)
        self.assertAlmostEqual(ledger.active_particle_ml, 0.0)

    def test_collection_capacity_leaves_uncaptured_volume_active(self) -> None:
        ledger = MODULE.FluidLedger(
            reservoir_capacity_ml=5.0,
            reservoir_ml=0.0,
            collection_capacity_ml=1.0,
            active_particle_ml=5.0,
        )
        self.assertEqual(ledger.aspirate(5.0), 1.0)
        self.assertEqual(ledger.active_particle_ml, 4.0)
        self.assertEqual(ledger.collection_remaining_ml, 0.0)

    def test_rejects_negative_nan_and_invalid_initial_state(self) -> None:
        ledger = MODULE.FluidLedger()
        for value in (-1.0, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ledger.emit(value)
        with self.assertRaises(ValueError):
            MODULE.FluidLedger(reservoir_capacity_ml=1.0, reservoir_ml=2.0)
        with self.assertRaises(ValueError):
            MODULE.FluidLedger(collection_capacity_ml=1.0, aspirated_ml=2.0)

    def test_snapshot_exposes_both_capacities(self) -> None:
        snapshot = MODULE.FluidLedger().snapshot()
        self.assertIn("reservoir_capacity_ml", snapshot)
        self.assertIn("collection_capacity_ml", snapshot)
        self.assertIn("balance_error_ml", snapshot)


class ProcedureContractTests(unittest.TestCase):
    def test_canonical_phase_targets_are_complete_and_bounded(self) -> None:
        phases = (
            "inspect", "contact", "pre_rinse", "aspirate", "debride",
            "post_rinse", "dry", "verify",
        )
        expected = {
            "contact_guard_joint",
            "debridement_extension_joint",
            "debridement_rotor_joint_velocity",
            "irrigation_valve_joint",
            "suction_valve_joint",
        }
        for phase in phases:
            with self.subTest(phase=phase):
                targets = MODULE.phase_targets(phase)
                self.assertEqual(set(targets), expected)
                self.assertGreaterEqual(targets["contact_guard_joint"], 0.0)
                self.assertLessEqual(targets["contact_guard_joint"], 0.008)
                self.assertGreaterEqual(
                    targets["debridement_extension_joint"], 0.0
                )
                self.assertLessEqual(
                    targets["debridement_extension_joint"], 0.020
                )
                self.assertGreaterEqual(targets["irrigation_valve_joint"], 0.0)
                self.assertLessEqual(targets["irrigation_valve_joint"], 0.006)
                self.assertGreaterEqual(targets["suction_valve_joint"], 0.0)
                self.assertLessEqual(
                    targets["suction_valve_joint"], math.radians(85.0)
                )

    def test_unknown_phase_and_frame_fail_closed(self) -> None:
        with self.assertRaises(KeyError):
            MODULE.phase_targets("close")
        with self.assertRaises(KeyError):
            MODULE.frame_path("/World/Tool", "unknown")

    def test_sequence_snapshot_records_transition_and_fluid_state(self) -> None:
        controller = MODULE.WoundPreparationSequenceController(
            tool_path="/World/Tool",
            wound_root_path="/World/Wound",
        )
        targets = controller.transition("contact")
        snapshot = controller.snapshot()
        self.assertEqual(snapshot["phase"], "contact")
        self.assertEqual(snapshot["history"][0]["targets"], targets)
        self.assertEqual(
            snapshot["status"], "simulation_training_workcell"
        )

    def test_registered_camera_contract_is_complete(self) -> None:
        self.assertEqual(
            MODULE.REGISTERED_CAMERA_FRAMES,
            (
                "camera_left", "camera_right", "depth_camera",
                "fluorescence_camera",
            ),
        )
        for name in MODULE.REGISTERED_CAMERA_FRAMES:
            self.assertIn(name, MODULE.TOOL_FRAME_PATHS)


if __name__ == "__main__":
    unittest.main()
