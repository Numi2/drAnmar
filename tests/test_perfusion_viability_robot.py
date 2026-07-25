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
    / "perfusion_viability_robot.py"
)
ASSET_ROOT = (
    ROOT
    / "source/extensions/orbit.surgical.assets/data"
    / "Props/SurgicalAssessment/PerfusionViabilityRobot"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "perfusion_viability_test_module", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime = load_module()

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

requires_numpy = pytest.mark.skipif(np is None, reason="NumPy is unavailable")


def test_task_phases_have_complete_finite_joint_targets() -> None:
    task = json.loads(
        (ASSET_ROOT / "perfusion_viability_task_contract.json").read_text(
            encoding="utf-8"
        )
    )
    expected_joints = set(runtime.TOOL_JOINTS.values())

    assert tuple(task["phases"]) == runtime.TASK_PHASES
    assert tuple(runtime.PHASE_TARGETS) == runtime.TASK_PHASES
    assert len(expected_joints) == 12
    for phase in runtime.TASK_PHASES:
        targets = runtime.phase_targets(phase)
        assert set(targets) == expected_joints
        assert all(math.isfinite(value) for value in targets.values())
    assert runtime.phase_targets("fused") == runtime.phase_targets("fuse")
    with pytest.raises(ValueError, match="Unknown phase"):
        runtime.phase_targets("unsupported")


@requires_numpy
def test_flow_network_conserves_mass_for_every_condition() -> None:
    solver = runtime.VascularFlowSolver()
    for condition in runtime.VALID_CONDITIONS:
        result = solver.solve(condition)
        assert result.condition == condition
        assert result.total_inflow_ml_s > 0.0
        assert abs(result.conservation_error_ml_s) < 1.0e-8
        assert len(result.region_flows_ml_s) == 24
        assert all(math.isfinite(value) for value in result.node_pressures_kpa.values())
        assert all(math.isfinite(value) for value in result.edge_flows_ml_s.values())

    with pytest.raises(ValueError, match="must exceed"):
        solver.solve("healthy", arterial_pressure_kpa=1.0, venous_pressure_kpa=2.0)
    with pytest.raises(ValueError, match="finite"):
        solver.solve("healthy", arterial_pressure_kpa=float("nan"))


@requires_numpy
@pytest.mark.parametrize(
    ("condition", "cause", "action"),
    (
        (
            "arterial_occlusion",
            "arterial_inflow_obstruction",
            "remove_or_reposition_occluder_or_clip",
        ),
        (
            "venous_congestion",
            "venous_outflow_obstruction",
            "release_venous_compression_or_revise_outflow",
        ),
        (
            "anastomotic_stenosis",
            "anastomotic_stenosis",
            "revise_anastomosis",
        ),
        ("branch_leak", "active_branch_leak", "control_branch_leak"),
        (
            "retraction_ischemia",
            "external_compression",
            "release_retraction_or_reduce_dressing_pressure",
        ),
        (
            "dressing_compression",
            "external_compression",
            "release_retraction_or_reduce_dressing_pressure",
        ),
    ),
)
def test_closed_loop_scan_identifies_and_improves_reversible_faults(
    condition: str, cause: str, action: str
) -> None:
    result = runtime.ClosedLoopPerfusionVerifier().scan_intervene_rescan(
        condition, duration_s=18.0, dt_s=0.15
    )

    assert result["before"].assessment.likely_cause == cause
    assert result["action"] == action
    assert result["after_condition"] == "recovered"
    assert result["viability_gain"] > 0.0
    assert result["nonperfused_fraction_reduction"] >= 0.0
    assert (
        result["after"].flow.conservation_error_ml_s
        == pytest.approx(0.0, abs=1.0e-8)
    )


@requires_numpy
def test_sensor_outputs_and_planner_inputs_are_bounded() -> None:
    verifier = runtime.ClosedLoopPerfusionVerifier()
    scan = verifier.scan("healthy", duration_s=8.0, dt_s=0.2)
    maps = scan.maps
    for array in (
        maps.flow_index,
        maps.icg_intensity,
        maps.icg_extravascular,
        maps.speckle_perfusion,
        maps.temperature_c,
        maps.oxygenation_fraction,
        maps.doppler_speed_m_s,
        maps.ultrasound_patency,
        maps.confidence,
    ):
        assert array.shape == (4, 6)
        assert np.all(np.isfinite(array))

    doppler = verifier.sensor_model.doppler_measure(
        scan.flow, "AT2", beam_direction=(1.0, 0.0, 0.0)
    )
    assert set(doppler) == {
        "edge_id",
        "axial_velocity_m_s",
        "speed_m_s",
        "direction_sign",
    }
    with pytest.raises(ValueError, match="non-zero"):
        verifier.sensor_model.doppler_measure(
            scan.flow, "AT2", beam_direction=(0.0, 0.0, 0.0)
        )

    planner = runtime.PerfusionScanPlanner()
    assert len(planner.optical_raster()) == 35
    assert len(planner.contact_probe_waypoints(modality="ultrasound")) == 24
    with pytest.raises(ValueError, match="non-negative"):
        planner.contact_probe_waypoints(modality="doppler", preload_n=-0.1)
    with pytest.raises(ValueError, match="three-dimensional"):
        planner.contact_probe_waypoints(
            modality="doppler", region_centers=[(0.0, 0.0)]
        )


@requires_numpy
@pytest.mark.parametrize(
    ("duration_s", "dt_s"),
    (
        (0.0, 0.1),
        (-1.0, 0.1),
        (1.0, 0.0),
        (1.0, float("nan")),
    ),
)
def test_scan_rejects_invalid_time_contract(duration_s: float, dt_s: float) -> None:
    with pytest.raises(ValueError):
        runtime.ClosedLoopPerfusionVerifier().scan(
            "healthy", duration_s=duration_s, dt_s=dt_s
        )
