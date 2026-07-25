from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

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


def test_phase_contract_is_complete_and_exact():
    assert list(MODULE.PHASE_TARGETS) == [
        "inspect", "center", "compress", "seal", "verify_seal",
        "retract_guard", "divide", "release", "verify_stumps",
        "complete", "abort",
    ]
    expected = set(MODULE.TOOL_JOINTS.values())
    for phase in MODULE.PHASE_TARGETS:
        targets = MODULE.phase_targets(phase)
        assert set(targets) == expected
        assert all(math.isfinite(value) for value in targets.values())
    with pytest.raises(KeyError):
        MODULE.phase_targets("unknown")


def test_energy_reaches_maturity_and_rejects_nonfinite_inputs():
    energy = qualified_energy()
    assert energy.left.temperature_c < energy.maximum_temperature_c
    assert energy.right.temperature_c < energy.maximum_temperature_c
    with pytest.raises(ValueError):
        energy.update(float("nan"), 9.0, 9.0)
    with pytest.raises(ValueError):
        energy.update(0.01, -1.0, 9.0)


def test_energy_overtemperature_latches_fault():
    energy = MODULE.AdaptiveSealEnergyController(maximum_temperature_c=40.0)
    for _ in range(100):
        energy.update(0.05, 12.0, 12.0, 45.0, 45.0)
        if energy.left.overtemperature:
            break
    assert energy.left.overtemperature
    assert energy.recommended_power_w(energy.left) == 0.0
    assert not energy.both_ready


def test_leak_model_is_monotonic_and_finite():
    model = MODULE.DualStumpLeakModel()
    open_flow = model.flow_ml_min(model.left)
    model.left.maturity = 0.95
    model.left.residual_gap_fraction = 0.25
    sealed_flow = model.flow_ml_min(model.left)
    model.left.damage = 1.0
    damaged_flow = model.flow_ml_min(model.left)
    assert 0.0 <= sealed_flow < damaged_flow < open_flow
    model.left.maturity = float("nan")
    with pytest.raises(ValueError):
        model.flow_ml_min(model.left)


def test_blade_interlock_requires_all_guards():
    energy = qualified_energy()
    leak = qualified_leak(energy)
    interlock = MODULE.BladeInterlockController()
    blocked = interlock.evaluate(energy, leak, 9.0, 9.0, False)
    assert not blocked["authorized"]
    assert "blade_guard_not_retracted" in blocked["reasons"]
    allowed = interlock.evaluate(energy, leak, 9.0, 9.0, True)
    assert allowed["authorized"]
    assert allowed["reasons"] == []


def test_compression_force_envelope_and_hard_release():
    controller = MODULE.DualZoneCompressionController("/Tool", "/Vessel")
    controller.engaged = True
    controller.attachment_paths = []
    assert controller.update_force(9.0, 9.0)["mode"] == "controlled"
    assert controller.update_force(18.0, 18.0)["mode"] == "soft_limit"
    report = controller.update_force(24.0, 24.0, stage=FakeStage())
    assert report["mode"] == "hard_release"
    assert not report["engaged"]
    with pytest.raises(ValueError):
        controller.update_force(float("inf"), 1.0)


def test_seal_band_break_force_progresses():
    controller = MODULE.TissueSealBandController()
    bond = MODULE.SealBandBond("/Band", "/Vessel", [])
    fresh = controller.break_force_n(bond)
    bond.maturity = 1.0
    mature = controller.break_force_n(bond)
    assert fresh == pytest.approx(0.6)
    assert mature == pytest.approx(7.5)
    assert mature > fresh


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


def test_division_does_not_advance_before_authorization():
    energy = qualified_energy()
    leak = qualified_leak(energy)
    bridge = FakeBridge()
    division = MODULE.TissueDivisionController(bridge)
    blocked = division.advance(
        0.5, energy=energy, leak=leak, upper_force_n=9.0,
        lower_force_n=9.0, guard_retracted=False, stage=FakeStage(),
    )
    assert blocked["blade_progress"] == 0.0
    assert division.violations == 1
    allowed = division.advance(
        1.0, energy=energy, leak=leak, upper_force_n=9.0,
        lower_force_n=9.0, guard_retracted=True, stage=FakeStage(),
    )
    assert allowed["authorized"]
    assert allowed["division_complete"]
    assert allowed["bridge_release_fraction"] == 1.0
