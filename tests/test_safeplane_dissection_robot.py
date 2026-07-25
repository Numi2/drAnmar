from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "safeplane_dissection_robot.py"
)
PACKAGE_TASK_PATH = (
    ROOT
    / "assets/Props/SurgicalDissection/SafePlaneDissectionRobot"
    / "safeplane_dissection_task_contract.json"
)
INSTALLED_TASK_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/data"
    / "Props/SurgicalDissection/SafePlaneDissectionRobot"
    / "safeplane_dissection_task_contract.json"
)
TASK_PATH = PACKAGE_TASK_PATH if PACKAGE_TASK_PATH.is_file() else INSTALLED_TASK_PATH


def load_module():
    spec = importlib.util.spec_from_file_location("safeplane_dissection_test_module", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime = load_module()


class PresentPrim:
    def IsValid(self) -> bool:
        return True

    def IsActive(self) -> bool:
        return True


class MissingPrim:
    def IsValid(self) -> bool:
        return False


class RecordingStage:
    def __init__(self):
        self.removed: list[str] = []

    def GetPrimAtPath(self, path: str):
        return MissingPrim() if str(path) in self.removed else PresentPrim()

    def RemovePrim(self, path: str):
        self.removed.append(str(path))


def first_state(controller, bridge_class: str):
    return next(
        state
        for state in controller.states.values()
        if state.bridge_class == bridge_class
    )


def test_phase_and_task_contracts_are_complete_and_finite():
    assert tuple(runtime.PHASE_TARGETS) == (
        "inspect",
        "capture",
        "traction",
        "blunt",
        "hydro",
        "scissors",
        "energy",
        "verify",
        "complete",
        "abort",
    )
    expected_joints = set(runtime.TOOL_JOINTS.values())
    assert len(expected_joints) == 17
    for targets in runtime.PHASE_TARGETS.values():
        assert set(targets) == expected_joints
        assert all(math.isfinite(value) for value in targets.values())

    task = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    assert task["procedure"] == [
        "inspect",
        "capture",
        "traction",
        "blunt",
        "hydro",
        "guarded_scissors",
        "low_energy",
        "irrigate_and_evacuate",
        "verify_connectivity",
        "release",
        "complete",
    ]


def test_fluid_ledger_conserves_volume_and_rejects_nonfinite_inputs():
    ledger = runtime.FluidLedger(
        reservoir_capacity_ml=5.0,
        reservoir_ml=5.0,
        collection_capacity_ml=3.0,
    )
    assert ledger.emit(4.0) == pytest.approx(4.0)
    assert ledger.aspirate(1.25) == pytest.approx(1.25)
    assert ledger.absorb(0.75) == pytest.approx(0.75)
    assert ledger.spill(0.5) == pytest.approx(0.5)
    assert ledger.active_particle_ml == pytest.approx(1.5)
    assert ledger.balance_error_ml == pytest.approx(0.0, abs=1.0e-12)
    with pytest.raises(ValueError):
        ledger.emit(float("nan"))
    with pytest.raises(ValueError):
        runtime.FluidLedger(reservoir_capacity_ml=1.0, reservoir_ml=2.0)


def test_traction_soft_and_hard_limits_release_physical_cells():
    stage = RecordingStage()
    controller = runtime.BilateralTractionController("/Tool", "/Tissue")
    controller.cells = [
        runtime.TractionCell(side, index, f"/attachment/{side}/{index}")
        for side in ("left", "right")
        for index in range(4)
    ]
    events = controller.update_force(3.0, 5.0, stage=stage)
    assert [event["event"] for event in events] == [
        "peripheral_cell_release",
        "hard_release",
    ]
    assert sum(cell.released for cell in controller.cells if cell.side == "left") == 1
    assert sum(cell.released for cell in controller.cells if cell.side == "right") == 4
    assert len(stage.removed) == 5
    with pytest.raises(ValueError):
        controller.update_force(-0.1, 0.0, stage=stage)


def test_each_bridge_modality_releases_its_intended_class():
    stage = RecordingStage()

    blunt = runtime.AdhesionBridgeController("/Tissue")
    loose = first_state(blunt, "loose_connective_fibre")
    released = blunt.apply_blunt_work(
        loose.position,
        loose.mechanical_threshold_j * 1.01,
        radius_m=1.0e-4,
        stage=stage,
    )
    assert loose.index in released
    assert loose.release_mode == "blunt_spreading"

    hydro = runtime.AdhesionBridgeController("/Tissue")
    loose = first_state(hydro, "loose_connective_fibre")
    released = hydro.apply_hydro_volume(
        loose.position,
        loose.hydro_threshold_ml * 1.01,
        radius_m=1.0e-4,
        stage=stage,
    )
    assert loose.index in released
    assert loose.release_mode == "hydrodissection"

    energy = runtime.AdhesionBridgeController("/Tissue")
    vascular = first_state(energy, "vascularized_adhesion")
    released = energy.apply_energy(
        vascular.position,
        vascular.energy_threshold_j * 1.01,
        radius_m=1.0e-4,
        stage=stage,
    )
    assert vascular.index in released
    assert vascular.release_mode == "low_energy_dissection"

    scissors = runtime.AdhesionBridgeController("/Tissue")
    dense = max(
        (
            state
            for state in scissors.states.values()
            if state.bridge_class == "dense_fibrous_band"
        ),
        key=lambda state: state.clearance_m,
    )
    result = runtime.ScissorsInterlockController().request_cut(
        dense.position,
        0.010,
        scissors,
        runtime.ProtectedStructureController("/Tissue"),
        stage=stage,
    )
    assert result["authorized"] is True
    assert result["released"] is True
    assert result["bridge_index"] == dense.index
    assert dense.release_mode == "guarded_scissors"


def test_protected_structure_interlocks_block_dangerous_actions():
    protected = runtime.ProtectedStructureController("/Tissue")
    vessel_point = protected.topology()["vessel"]["centerline_m"][2]
    for modality in ("blunt", "hydro", "scissors", "energy"):
        result = protected.evaluate_action(vessel_point, modality)
        assert result["authorized"] is False
        assert result["nearest_structure"] == "vessel"

    sequence = runtime.SafePlaneDissectionSequenceController("/Tissue", "/Tool")
    result = sequence.energy_action(
        vessel_point,
        dt=0.1,
        contact_force_n=1.5,
        requested_power_w=22.0,
        stage=RecordingStage(),
    )
    assert result["blocked"] is True
    assert all(state.intact for state in sequence.protected.states.values())
    with pytest.raises(ValueError):
        protected.evaluate_action((float("nan"), 0.0, 0.0), "energy")


def test_scissor_guard_and_energy_overtemperature_latch():
    bridges = runtime.AdhesionBridgeController("/Tissue")
    dense = first_state(bridges, "dense_fibrous_band")
    protected = runtime.ProtectedStructureController("/Tissue")
    scissors = runtime.ScissorsInterlockController()
    blocked = scissors.request_cut(
        dense.position,
        0.0,
        bridges,
        protected,
        stage=RecordingStage(),
    )
    assert blocked["released"] is False
    assert "guard_not_fully_retracted" in blocked["reasons"]

    energy = runtime.LowEnergyDissectionController()
    energy.state.temperature_c = 96.0
    first = energy.update(0.1, 1.5, 22.0)
    assert first["state"].overtemperature is True
    delivered = energy.state.delivered_energy_j
    second = energy.update(0.1, 1.5, 22.0)
    assert second["energy_j"] == pytest.approx(0.0)
    assert energy.state.delivered_energy_j == pytest.approx(delivered)
    with pytest.raises(ValueError):
        energy.update(float("inf"), 1.0)


def test_completion_requires_connectivity_release_and_intact_anatomy():
    bridges = runtime.AdhesionBridgeController("/Tissue")
    protected = runtime.ProtectedStructureController("/Tissue")
    verifier = runtime.DissectionCompletionVerifier(bridges, protected)
    assert verifier.evaluate(visibility_fraction=0.95, traction_stable=True)["complete"] is False
    for state in bridges.states.values():
        state.released = True
    passed = verifier.evaluate(visibility_fraction=0.95, traction_stable=True)
    assert passed["complete"] is True
    protected.states["duct"].intact = False
    assert verifier.evaluate(visibility_fraction=0.95, traction_stable=True)["complete"] is False
    with pytest.raises(ValueError):
        verifier.evaluate(visibility_fraction=1.01, traction_stable=True)
