from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPOSITORY_ROOT / "scripts/validate_openusd_layers.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "dranmar_openusd_layer_validator", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discovery_covers_all_usd_layer_suffixes_and_ignores_caches(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    assets = tmp_path / "assets"
    source = tmp_path / "source"
    assets.mkdir()
    source.mkdir()
    expected = {
        assets / "one.usda",
        assets / "two.usd",
        source / "three.usdc",
    }
    for path in expected:
        path.write_text("#usda 1.0\n", encoding="utf-8")
    ignored = source / "__pycache__" / "ignored.usda"
    ignored.parent.mkdir()
    ignored.write_text("#usda 1.0\n", encoding="utf-8")
    (source / "not-a-layer.txt").write_text("{}", encoding="utf-8")

    assert set(validator.discover_usd_layers((assets, source))) == expected


def test_repository_discovery_includes_the_abdominal_rigid_proxy() -> None:
    validator = _load_validator()
    assert validator.DEFAULT_LAYER_ROOTS == (REPOSITORY_ROOT,)
    layers = validator.discover_usd_layers(validator.DEFAULT_LAYER_ROOTS)
    relative = {
        path.relative_to(REPOSITORY_ROOT).as_posix() for path in layers
    }
    assert (
        "source/extensions/orbit.surgical.assets/data/Props/Patients/"
        "DynamicAbdominalPatient/"
        "dranmar_dynamic_abdominal_patient_rigid_proxy.usda"
    ) in relative


def test_abdominal_rigid_proxy_keeps_physics_material_under_looks() -> None:
    from pxr import Usd

    proxy = (
        REPOSITORY_ROOT
        / "source/extensions/orbit.surgical.assets/data/Props/Patients"
        / "DynamicAbdominalPatient"
        / "dranmar_dynamic_abdominal_patient_rigid_proxy.usda"
    )
    stage = Usd.Stage.Open(str(proxy))
    assert stage is not None

    root = "/DrAnmarDynamicAbdominalPatientRigidProxy"
    material_path = f"{root}/Looks/TablePhysics"
    material = stage.GetPrimAtPath(material_path)
    assert material.IsValid()
    assert material.GetTypeName() == "Material"
    assert not stage.GetPrimAtPath(f"{root}/PhysicsMaterials").IsValid()

    binding = stage.GetPrimAtPath(f"{root}/Collisions/TorsoCollider").GetRelationship(
        "material:binding:physics"
    )
    assert [str(path) for path in binding.GetTargets()] == [material_path]
