from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "dynamic_abdominal_patient.py"
)


def load_runtime():
    name = "dranmar_dynamic_patient_laparotomy_test_runtime"
    spec = importlib.util.spec_from_file_location(name, RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def runtime():
    return load_runtime()


def sample(runtime, controller=None, **overrides):
    values = {
        "blade_contact": True,
        "normal_force_n": 0.0,
        "tangential_force_n": 6.0,
        "advancement_m": 0.0075,
        "speed_m_s": 0.03,
        "alignment_error_deg": 0.0,
    }
    values.update(overrides)
    if controller is not None:
        advancement = float(values["advancement_m"])
        values.setdefault(
            "path_coordinate_m",
            controller.last_path_coordinate_m + advancement,
        )
        values.setdefault("lateral_offset_m", 0.0)
        values.setdefault("blade_tip_z_m", controller.active_layer_depth_m)
        values.setdefault("dt_s", advancement / float(values["speed_m_s"]))
    return runtime.IncisionContactSample(**values)


def test_incision_requires_physical_gates(runtime):
    patient = runtime.DynamicSurgicalPatient()
    controller = patient.incision

    assert controller.advance(
        sample(runtime, controller, blade_contact=False)
    )["reason"] == "no_blade_contact"
    assert controller.advance(
        sample(runtime, controller, tangential_force_n=1.0)
    )["reason"] == "cutting_force_below_gate"
    assert controller.advance(
        sample(runtime, controller, speed_m_s=0.2)
    )["reason"] == "blade_speed_outside_gate"
    assert controller.advance(
        sample(runtime, controller, alignment_error_deg=30.0)
    )["reason"] == "blade_alignment_outside_gate"
    assert controller.advance(sample(runtime))["reason"] == (
        "missing_spatial_kinematic_evidence"
    )
    assert controller.advance(
        sample(runtime, controller, lateral_offset_m=0.01)
    )["reason"] == "blade_outside_midline_corridor"
    assert controller.progress_fraction == 0.0
    assert patient.tissue_state.access_state == "intact"


def test_midline_incision_releases_ordered_continuity(runtime):
    patient = runtime.DynamicSurgicalPatient()
    controller = patient.incision
    segment = controller.bridge_length_m

    released = []
    for _ in range(controller.calibration.bridges_per_layer):
        result = controller.advance(
            sample(runtime, controller, advancement_m=segment)
        )
        released.extend(result["released_bridge_ids"])

    assert released == [
        f"skin:{index:03d}"
        for index in range(controller.calibration.bridges_per_layer)
    ]
    assert controller.active_layer == "subcutaneous_fat"
    assert controller.progress_fraction == pytest.approx(0.25)
    assert patient.tissue_state.access_state == "intact"
    assert patient.tissue_state.get("skin").cuts == 1


def test_complete_laparotomy_opens_only_after_all_layers(runtime):
    patient = runtime.DynamicSurgicalPatient()
    controller = patient.incision
    segment = controller.bridge_length_m
    sample_count = (
        len(runtime.LAPAROTOMY_INCISED_LAYERS)
        * controller.calibration.bridges_per_layer
    )

    for _ in range(sample_count):
        controller.advance(sample(runtime, controller, advancement_m=segment))

    assert controller.complete
    assert controller.active_layer is None
    assert controller.progress_fraction == 1.0
    assert patient.tissue_state.access_state == "open"
    assert patient.procedure_stage == "access_open"
    assert len(controller.released_bridge_ids) == sample_count
    assert patient.tissue_state.get("abdominal_wall").cuts == 0
    assert any(
        event["kind"] == "physical_incision_progress"
        for event in patient.event_bus.snapshot()
    )


def test_overload_is_retained_as_an_uncalibrated_boundary_signal(runtime):
    patient = runtime.DynamicSurgicalPatient()
    result = patient.incision.advance(
        sample(runtime, patient.incision, tangential_force_n=13.0)
    )

    assert result["accepted"]
    assert result["overload"]
    assert patient.incision.overload_samples == 1
    assert (
        patient.incision.snapshot()["calibration"]["calibration_status"]
        == "cross_tissue_research_envelope_pending_abdominal_layer_bench"
    )


def test_wound_edge_grasp_requires_post_physics_contact_dwell(runtime):
    grasp = runtime.PhysicalWoundEdgeGrasp()
    base = {
        "layer": "skin",
        "side": "left",
        "cell_index": 0,
        "contact": True,
        "normal_force_n": 0.2,
        "relative_speed_m_s": 0.0,
        "edge_offset_m": 0.001,
        "dt_s": 0.05,
    }
    untrusted = grasp.observe(
        runtime.WoundEdgeGraspSample(
            **base,
            contact_authority="commanded_pose",
        )
    )
    assert untrusted["reason"] == "non_physics_contact_authority"
    first = grasp.observe(runtime.WoundEdgeGraspSample(**base))
    second = grasp.observe(runtime.WoundEdgeGraspSample(**base))
    assert first["reason"] == "capture_dwell_accumulating"
    assert second["capture_requested"] is True
    assert grasp.captured_cells == {"skin:left:00"}


def test_wound_edge_grasp_releases_on_overload_or_slip(runtime):
    calibration = runtime.LaparotomyGraspCalibration(capture_dwell_s=0.01)
    grasp = runtime.PhysicalWoundEdgeGrasp(calibration)
    accepted = runtime.WoundEdgeGraspSample(
        layer="fascia",
        side="right",
        cell_index=3,
        contact=True,
        normal_force_n=0.2,
        relative_speed_m_s=0.0,
        edge_offset_m=0.001,
        dt_s=0.02,
    )
    assert grasp.observe(accepted)["capture_requested"] is True
    overload = runtime.WoundEdgeGraspSample(
        **{
            **accepted.__dict__,
            "normal_force_n": 1.3,
        }
    )
    result = grasp.observe(overload)
    assert result["release_requested"] is True
    assert not grasp.captured_cells
