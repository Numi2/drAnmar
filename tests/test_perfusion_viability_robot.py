from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace
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
    assert result["intervention_completed"] is True
    assert result["recovery_fraction"] == pytest.approx(1.0)
    assert result["evidence_source"] == "deterministic_research_fixture"
    assert all(update.accepted for update in result["intervention_updates"])
    assert [
        update.recovery_fraction for update in result["intervention_updates"]
    ] == sorted(
        update.recovery_fraction for update in result["intervention_updates"]
    )
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
    with pytest.raises(ValueError, match="positive"):
        planner.optical_raster(width_m=-0.1)
    with pytest.raises(ValueError, match="positive integers"):
        planner.optical_raster(rows=0)


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


@requires_numpy
@pytest.mark.parametrize(
    "condition",
    (
        "arterial_occlusion",
        "venous_congestion",
        "anastomotic_stenosis",
        "branch_leak",
        "retraction_ischemia",
        "dressing_compression",
    ),
)
def test_diagnosis_is_blind_to_scenario_label_and_latent_flow(condition: str) -> None:
    verifier = runtime.ClosedLoopPerfusionVerifier()
    scan = verifier.scan(condition, duration_s=18.0, dt_s=0.15)
    expected = scan.assessment

    wrong_label = verifier.estimator.estimate(
        "healthy",
        scan.maps,
        icg_metrics=scan.icg_metrics,
    )
    impossible_latent_truth = replace(
        scan.maps, flow_index=np.full((4, 6), 42.0)
    )
    altered_truth = verifier.estimator.estimate(
        impossible_latent_truth,
        icg_metrics=scan.icg_metrics,
        scenario_label="deliberately_wrong",
    )

    assert wrong_label.likely_cause == expected.likely_cause
    assert altered_truth.likely_cause == expected.likely_cause
    assert altered_truth.global_viability_score == pytest.approx(
        expected.global_viability_score
    )
    assert wrong_label.condition == "healthy"
    assert altered_truth.condition == "deliberately_wrong"


@requires_numpy
def test_sensor_faults_consumables_and_registration_drive_abstention() -> None:
    verifier = runtime.ClosedLoopPerfusionVerifier()
    ledger = runtime.SensorConsumableLedger(
        initial_contrast_ml=0.1,
        initial_gel_ml=0.1,
    )
    depleted = verifier.scan(
        "healthy",
        duration_s=4.0,
        dt_s=0.2,
        consumables=ledger,
    )
    assert {"nir_icg", "ultrasound"}.issubset(depleted.maps.faults)
    assert "nir_icg" not in depleted.assessment.usable_modalities
    assert "ultrasound" not in depleted.assessment.usable_modalities
    assert ledger.conservation_error_ml == pytest.approx(0.0)

    faulted = verifier.scan(
        "healthy",
        duration_s=4.0,
        dt_s=0.2,
        operating_state=runtime.SensorOperatingState(sensor_state="fault"),
    )
    assert faulted.assessment.abstained is True
    assert faulted.assessment.likely_cause == "mixed_or_uncertain"
    assert (
        faulted.assessment.recommended_action
        == "repeat_scan_and_inspect_sensor_registration"
    )

    misregistered = verifier.scan(
        "healthy",
        duration_s=4.0,
        dt_s=0.2,
        operating_state=runtime.SensorOperatingState(
            registration_error_m=0.004
        ),
    )
    assert misregistered.assessment.abstained is True


@requires_numpy
def test_degraded_sensor_state_reduces_confidence() -> None:
    ready = runtime.ClosedLoopPerfusionVerifier().scan(
        "healthy", duration_s=4.0, dt_s=0.2
    )
    degraded = runtime.ClosedLoopPerfusionVerifier().scan(
        "healthy",
        duration_s=4.0,
        dt_s=0.2,
        operating_state=runtime.SensorOperatingState(sensor_state="degraded"),
    )
    assert float(np.mean(degraded.maps.confidence)) < float(
        np.mean(ready.maps.confidence)
    )


@requires_numpy
def test_intervention_requires_mechanical_evidence_and_progresses_continuously() -> None:
    controller = runtime.PerfusionConditionController("arterial_occlusion")
    with pytest.raises(ValueError, match="physical intervention evidence"):
        controller.apply("remove_or_reposition_occluder_or_clip")

    partial = controller.update(
        runtime.InterventionEvidence(
            action="remove_or_reposition_occluder_or_clip",
            elapsed_s=0.5,
            displacement_m=0.003,
        )
    )
    assert partial.accepted is True
    assert partial.completed is False
    assert partial.recovery_fraction == pytest.approx(0.5)

    solver = runtime.VascularFlowSolver()
    untreated = solver.solve("arterial_occlusion")
    intermediate = solver.solve("arterial_occlusion", recovery_fraction=0.5)
    treated = solver.solve("arterial_occlusion", recovery_fraction=1.0)
    assert (
        untreated.region_flows_ml_s[23]
        < intermediate.region_flows_ml_s[23]
        < treated.region_flows_ml_s[23]
    )
    assert abs(intermediate.conservation_error_ml_s) < 1.0e-8


def test_probe_contact_controller_has_coupling_and_overload_states() -> None:
    controller = runtime.ProbeContactController()
    coupled = controller.update(measured_force_n=1.2, dt_s=1.0 / 120.0)
    assert coupled.coupled is True
    assert coupled.abort is False
    overloaded = controller.update(measured_force_n=4.2, dt_s=1.0 / 120.0)
    assert overloaded.overload is True
    assert overloaded.abort is True
    assert overloaded.target_extension_delta_m < 0.0


@requires_numpy
def test_registered_sensor_packet_rejects_missing_or_bad_frames() -> None:
    scan = runtime.ClosedLoopPerfusionVerifier().scan(
        "healthy", duration_s=4.0, dt_s=0.2
    )
    names = (
        "rgb_left_camera",
        "rgb_right_camera",
        "nir_fluorescence_camera",
        "speckle_camera",
        "thermal_camera",
        "multispectral_camera",
    )
    frames = {
        name: np.zeros((24, 32, 4), dtype=np.uint8) for name in names
    }
    packet = runtime.build_registered_sensor_packet(
        timestamp_s=scan.final_tracer.time_s,
        camera_frames=frames,
        depth_frame=np.ones((24, 32), dtype=float),
        maps=scan.maps,
    )
    assert packet.valid is True
    broken = runtime.build_registered_sensor_packet(
        timestamp_s=scan.final_tracer.time_s,
        camera_frames={"rgb_left_camera": np.zeros((24, 32), dtype=np.uint8)},
        depth_frame=np.full((24, 32), np.nan),
        maps=scan.maps,
    )
    assert broken.valid is False
    assert broken.errors


@requires_numpy
def test_temporal_icg_metrics_support_isaac_numpy_1_x(monkeypatch) -> None:
    history = runtime.PerfusionTimeSeries(region_count=1)
    history.append(runtime.TracerFrame(0.0, 0.0, {}, {0: 0.0}, {0: 0.0}))
    history.append(runtime.TracerFrame(1.0, 0.0, {}, {0: 2.0}, {0: 0.0}))

    monkeypatch.delattr(np, "trapezoid", raising=False)

    assert history.metrics(0).area_under_curve == pytest.approx(1.0)
