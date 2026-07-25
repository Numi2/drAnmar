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


def sample(runtime, **overrides):
    values = {
        "blade_contact": True,
        "normal_force_n": 0.0,
        "tangential_force_n": 4.0,
        "advancement_m": 0.0075,
        "speed_m_s": 0.03,
        "alignment_error_deg": 0.0,
    }
    values.update(overrides)
    return runtime.IncisionContactSample(**values)


def test_incision_requires_physical_gates(runtime):
    patient = runtime.DynamicSurgicalPatient()
    controller = patient.incision

    assert controller.advance(
        sample(runtime, blade_contact=False)
    )["reason"] == "no_blade_contact"
    assert controller.advance(
        sample(runtime, tangential_force_n=1.0)
    )["reason"] == "cutting_force_below_gate"
    assert controller.advance(
        sample(runtime, speed_m_s=0.2)
    )["reason"] == "blade_speed_outside_gate"
    assert controller.advance(
        sample(runtime, alignment_error_deg=30.0)
    )["reason"] == "blade_alignment_outside_gate"
    assert controller.progress_fraction == 0.0
    assert patient.tissue_state.access_state == "intact"


def test_midline_incision_releases_ordered_continuity(runtime):
    patient = runtime.DynamicSurgicalPatient()
    controller = patient.incision
    segment = controller.bridge_length_m

    released = []
    for _ in range(controller.calibration.bridges_per_layer):
        result = controller.advance(
            sample(runtime, advancement_m=segment)
        )
        released.extend(result["released_bridge_ids"])

    assert released == [
        f"skin:{index:03d}"
        for index in range(controller.calibration.bridges_per_layer)
    ]
    assert controller.active_layer == "subcutaneous_fat"
    assert controller.progress_fraction == pytest.approx(0.2)
    assert patient.tissue_state.access_state == "intact"
    assert patient.tissue_state.get("skin").cuts == 1


def test_complete_laparotomy_opens_only_after_all_layers(runtime):
    patient = runtime.DynamicSurgicalPatient()
    controller = patient.incision
    segment = controller.bridge_length_m
    sample_count = (
        len(runtime.LAPAROTOMY_LAYERS)
        * controller.calibration.bridges_per_layer
    )

    for _ in range(sample_count):
        controller.advance(sample(runtime, advancement_m=segment))

    assert controller.complete
    assert controller.active_layer is None
    assert controller.progress_fraction == 1.0
    assert patient.tissue_state.access_state == "open"
    assert patient.procedure_stage == "access_open"
    assert len(controller.released_bridge_ids) == sample_count
    assert any(
        event["kind"] == "physical_incision_progress"
        for event in patient.event_bus.snapshot()
    )


def test_overload_is_retained_as_an_uncalibrated_boundary_signal(runtime):
    patient = runtime.DynamicSurgicalPatient()
    result = patient.incision.advance(
        sample(runtime, tangential_force_n=13.0)
    )

    assert result["accepted"]
    assert result["overload"]
    assert patient.incision.overload_samples == 1
    assert (
        patient.incision.snapshot()["calibration"]["calibration_status"]
        == "cross_tissue_research_envelope_pending_abdominal_layer_bench"
    )
