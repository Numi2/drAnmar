from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RUNTIME_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "dynamic_abdominal_patient.py"
)


def load_runtime():
    spec = importlib.util.spec_from_file_location(
        "dranmar_dynamic_patient_quality_runtime", RUNTIME_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime = load_runtime()


@pytest.fixture
def route_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime,
        "load_anatomy_manifest",
        lambda: {
            "components": [
                {"id": "peritoneum", "mechanics": "surface_deformable"},
                {"id": "liver", "mechanics": "volume_deformable"},
                {"id": "nerves", "mechanics": "segmented_rod"},
            ]
        },
    )


@pytest.mark.parametrize(
    ("requested", "error"),
    [
        ((), "at least one"),
        (("peritoneum", "peritoneum"), "duplicate"),
        (("invented",), "unknown"),
        (("nerves",), "do not use a native deformable route"),
    ],
)
def test_explicit_deformable_selection_fails_closed(
    route_manifest: None,
    requested: tuple[str, ...],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        runtime.apply_patient_deformables(
            "/World/Patient",
            include=requested,
            stage=object(),
        )


def test_string_is_not_accepted_as_a_component_sequence(route_manifest: None) -> None:
    with pytest.raises(TypeError, match="not a string"):
        runtime.apply_patient_deformables(
            "/World/Patient",
            include="peritoneum",
            stage=object(),
        )


def test_explicit_deformable_selection_applies_only_requested_component(
    route_manifest: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def apply_component(_stage, path, config):
        calls.append((path, config["id"]))
        return {
            "component": config["id"],
            "route": "current_surface_deformable",
        }

    monkeypatch.setattr(runtime, "apply_component_deformable", apply_component)
    results = runtime.apply_patient_deformables(
        "/World/Patient/",
        include=("peritoneum",),
        stage=object(),
    )

    assert calls == [("/World/Patient/Anatomy/peritoneum", "peritoneum")]
    assert results["peritoneum"]["route"] in runtime.NATIVE_DEFORMABLE_ROUTES


def test_access_state_fails_when_patient_prim_is_missing() -> None:
    invalid_prim = SimpleNamespace(IsValid=lambda: False)
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: invalid_prim)

    with pytest.raises(RuntimeError, match="Patient prim does not exist"):
        runtime.set_access_state("/World/Patient", "open", stage=stage)


@pytest.mark.parametrize(
    "component",
    ("skin", "subcutaneous_fat", "fascia", "abdominal_wall", "peritoneum"),
)
def test_access_layer_variants_target_geometry_children(component: str) -> None:
    source = (
        ROOT
        / "source/extensions/orbit.surgical.assets/data/Props/Patients"
        / "DynamicAbdominalPatient/anatomy"
        / f"dranmar_{component}.usda"
    ).read_text(encoding="utf-8")
    variant_contract = source.split('variantSet "access_state"', 1)[1]

    assert variant_contract.count('over "Geometry"') == 2
    assert variant_contract.count('over "Visual"') == 2
    assert variant_contract.count('over "OpenVisual"') == 2


def test_profile_exposes_the_single_lane_and_overall_qualification_boundaries() -> None:
    from dr_anmar_procedures import PROCEDURES_BY_ID

    profile = json.loads(
        (
            ROOT
            / "physics_next/dynamic-patient"
            / "dranmar-dynamic-abdominal-patient-v1.json"
        ).read_text(encoding="utf-8")
    )
    room = PROCEDURES_BY_ID["dr-anmar-dynamic-abdominal-patient"]
    assert (
        profile["deployment"]["default_access_state"]
        == room["dynamic_patient_access_state"]
    )
    assert profile["deployment"]["maximum_solver_active_deformables"] == 1
    assert len(room["dynamic_patient_active_deformables"]) == 1
    assert profile["deployment"]["multi_component_deformables_qualified"] is False
    assert profile["deployment"]["native_volume_deformables_qualified"] is False
    assert profile["validation_scope"]["overall_qualified"] is False

    report = json.loads(
        (
            ROOT
            / "physics_next/benchmarks"
            / "dranmar-dynamic-abdominal-patient-validation.json"
        ).read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert report["passed_scope"] == "checks_executed_by_this_validator_only"
    assert report["overall_qualified"] is False
