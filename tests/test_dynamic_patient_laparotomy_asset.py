from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (
    ROOT
    / "source/extensions/orbit.surgical.assets/data/Props/Patients"
    / "DynamicAbdominalPatient"
)
WOUND_ASSET = ASSET_ROOT / "anatomy/dranmar_laparotomy_wound.usda"
PATIENT_ASSET = ASSET_ROOT / "dranmar_dynamic_abdominal_patient.usda"
RUNTIME = (
    ROOT
    / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
    / "dynamic_abdominal_patient.py"
)
GENERATOR = ROOT / "scripts/generate_dranmar_laparotomy_wound.py"
MANIFEST = ASSET_ROOT / "asset_manifest.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_wound_asset_is_deterministic_and_full_thickness():
    generator = _load(GENERATOR, "dranmar_laparotomy_generator_test")
    text = WOUND_ASSET.read_text(encoding="utf-8")

    assert generator.build_usda() == text
    assert text.count('def TetMesh "SimulationTetMesh"') == 10
    assert text.count('def Mesh "Visual"') == 10
    assert text.count("custom bool drAnmar:clinicalValidation = false") == 11
    assert 'drAnmar:incisionType = "median_laparotomy"' in text
    assert "tissue_plug" not in text


def test_wound_asset_is_registered_with_current_digest():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in payload["files"]
        if entry["path"].endswith(
            "/anatomy/dranmar_laparotomy_wound.usda"
        )
    ]

    assert len(entries) == 1
    assert entries[0]["bytes"] == WOUND_ASSET.stat().st_size
    assert entries[0]["sha256"] == hashlib.sha256(
        WOUND_ASSET.read_bytes()
    ).hexdigest()


def test_every_wound_tetmesh_has_positive_finite_volume():
    generator = _load(GENERATOR, "dranmar_laparotomy_volume_test")
    for layer in generator.LAYERS:
        for side in ("Left", "Right"):
            points, tetrahedra, _ = generator._mesh(
                side,
                center_z=float(layer["center_z"]),
                thickness=float(layer["thickness"]),
                lip_lift=float(layer["lip_lift"]),
            )
            assert len(points) == 350
            assert len(tetrahedra) == 864
            for tetrahedron in tetrahedra:
                determinant = generator._determinant(
                    *(points[index] for index in tetrahedron)
                )
                assert determinant > 0.0


def test_patient_variants_compose_only_the_open_wound():
    text = PATIENT_ASSET.read_text(encoding="utf-8")

    assert (
        "references = "
        "@./anatomy/dranmar_laparotomy_wound.usda@"
        "</DrAnmarLaparotomyWound>"
    ) in re.sub(r"\s+", " ", text)
    intact = text.index('"intact"')
    open_state = text.index('"open"', intact)
    assert 'token visibility = "invisible"' in text[intact:open_state]
    assert 'token visibility = "inherited"' in text[open_state:]


def test_runtime_exposes_all_bilateral_wound_paths():
    runtime = _load(RUNTIME, "dranmar_laparotomy_asset_test_runtime")
    paths = runtime.laparotomy_wound_edge_paths("/World/Patient/")

    assert tuple(paths) == runtime.LAPAROTOMY_LAYERS
    assert sum(len(sides) for sides in paths.values()) == 10
    assert all(tuple(sides) == ("left", "right") for sides in paths.values())
    assert all(
        path.endswith("/Geometry/SimulationTetMesh")
        for sides in paths.values()
        for path in sides.values()
    )
    assert all(
        config["youngs_modulus_pa_seed"] > 0.0
        for config in runtime.LAPAROTOMY_WOUND_LAYER_CONFIGS.values()
    )
