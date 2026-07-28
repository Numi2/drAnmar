from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
