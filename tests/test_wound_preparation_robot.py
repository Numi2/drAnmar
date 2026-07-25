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

if __name__ == "__main__":
    unittest.main()
