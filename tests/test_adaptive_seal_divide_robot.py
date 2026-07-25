from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "adaptive_seal_divide_robot.py"
)
SPEC = importlib.util.spec_from_file_location("adaptive_seal_divide_robot_tested", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def qualified_energy():
    energy = MODULE.AdaptiveSealEnergyController()
    for _ in range(4000):
        energy.update(0.01, 9.0, 9.0)
        if energy.left.maturity >= 0.975 and energy.right.maturity >= 0.975:
            break
    assert energy.both_ready
    return energy


def qualified_leak(energy):
    leak = MODULE.DualStumpLeakModel()
    for state, zone in ((leak.left, energy.left), (leak.right, energy.right)):
        state.maturity = zone.maturity
        state.residual_gap_fraction = 1.0 - 9.0 / 24.0
    assert leak.flows()["left_ml_min"] < 0.1
    assert leak.flows()["right_ml_min"] < 0.1
    return leak


class FakeStage:
    def GetPrimAtPath(self, _path):
        return FakePrim()


class FakePrim:
    def IsValid(self):
        return False


class FakeBridge:
    released_fraction = 0.0
    complete = False

    def set_cut_progress(self, progress, stage=None):
        self.released_fraction = progress
        self.complete = progress >= 1.0
        return self.released_fraction


class AdaptiveSealDivideSafetyTests(unittest.TestCase):
    def test_energy_overtemperature_latches_fault(self):
        energy = MODULE.AdaptiveSealEnergyController(maximum_temperature_c=40.0)
        for _ in range(100):
            energy.update(0.05, 12.0, 12.0, 45.0, 45.0)
            if energy.left.overtemperature:
                break
        self.assertTrue(energy.left.overtemperature)
        self.assertEqual(energy.recommended_power_w(energy.left), 0.0)
        self.assertFalse(energy.both_ready)

    def test_blade_interlock_requires_all_guards(self):
        energy = qualified_energy()
        leak = qualified_leak(energy)
        interlock = MODULE.BladeInterlockController()
        blocked = interlock.evaluate(energy, leak, 9.0, 9.0, False)
        self.assertFalse(blocked["authorized"])
        self.assertIn("blade_guard_not_retracted", blocked["reasons"])
        allowed = interlock.evaluate(energy, leak, 9.0, 9.0, True)
        self.assertTrue(allowed["authorized"])
        self.assertEqual(allowed["reasons"], [])

    def test_compression_force_envelope_and_hard_release(self):
        controller = MODULE.DualZoneCompressionController("/Tool", "/Vessel")
        controller.engaged = True
        controller.attachment_paths = []
        self.assertEqual(controller.update_force(9.0, 9.0)["mode"], "controlled")
        self.assertEqual(controller.update_force(18.0, 18.0)["mode"], "soft_limit")
        report = controller.update_force(24.0, 24.0, stage=FakeStage())
        self.assertEqual(report["mode"], "hard_release")
        self.assertFalse(report["engaged"])
        with self.assertRaises(ValueError):
            controller.update_force(float("inf"), 1.0)

    def test_division_does_not_advance_before_authorization(self):
        energy = qualified_energy()
        leak = qualified_leak(energy)
        bridge = FakeBridge()
        division = MODULE.TissueDivisionController(bridge)
        blocked = division.advance(
            0.5, energy=energy, leak=leak, upper_force_n=9.0,
            lower_force_n=9.0, guard_retracted=False, stage=FakeStage(),
        )
        self.assertEqual(blocked["blade_progress"], 0.0)
        self.assertEqual(division.violations, 1)
        allowed = division.advance(
            1.0, energy=energy, leak=leak, upper_force_n=9.0,
            lower_force_n=9.0, guard_retracted=True, stage=FakeStage(),
        )
        self.assertTrue(allowed["authorized"])
        self.assertTrue(allowed["division_complete"])
        self.assertEqual(allowed["bridge_release_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
