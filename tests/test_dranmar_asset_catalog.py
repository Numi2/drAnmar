# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / (
    "source/extensions/orbit.surgical.assets/orbit/surgical/assets/"
    "dranmar_asset_catalog.py"
)
INDEX_GENERATOR_PATH = ROOT / "scripts/generate_dranmar_asset_catalog_index.py"
AUTONOMOUS_RESCUE_ASSET_ROOT = ROOT / (
    "source/extensions/orbit.surgical.assets/data/Environments/"
    "SurgicalAutonomy/AutonomousRescueOR"
)


def load_catalog():
    spec = importlib.util.spec_from_file_location(
        "dranmar_asset_catalog_test_module", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_robot_catalog_members_and_usd_dependencies_exist():
    catalog = load_catalog()
    assert len(catalog.DRANMAR_SIM_READY_ASSETS) == 9
    for name, descriptor in catalog.DRANMAR_SIM_READY_ASSETS.items():
        directory = catalog.asset_directory(name)
        assert directory.as_posix().endswith(descriptor.catalog_subpath)
        for member in descriptor.members():
            assert (directory / member).is_file(), (name, member)
        closure = catalog.validate_usd_dependency_closure(name)
        assert closure["missing"] == []
        assert closure["escaping"] == []
    assert (
        catalog.DrAnmarSurgicalRobotAssets.ONCOLOGIC_RESECTION
        == "Props/SurgicalOncology/OncoSurgeryCell/"
        "dranmar_tumor_resection_tool_standalone.usda"
    )
    assert (
        catalog.DrAnmarSurgicalRobotAssets.AUTONOMOUS_RESCUE_OR
        == "Environments/SurgicalAutonomy/AutonomousRescueOR/"
        "dranmar_autonomous_rescue_or.usda"
    )


def test_folder_hash_includes_relative_path_and_content(tmp_path):
    catalog = load_catalog()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "asset.usda").write_bytes(b"usd")
    expected = hashlib.sha256()
    expected.update(b"nested/asset.usda")
    expected.update(b"usd")
    assert catalog.sha256_of_folder(tmp_path) == expected.hexdigest()

    (tmp_path / "nested" / "asset.usda").rename(
        tmp_path / "nested" / "renamed.usda"
    )
    assert catalog.sha256_of_folder(tmp_path) != expected.hexdigest()


def test_catalog_resolution_fails_closed(monkeypatch, tmp_path):
    catalog = load_catalog()
    monkeypatch.setenv(catalog.CATALOG_ROOT_ENV, str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        catalog.asset_data_root()
    with pytest.raises(KeyError):
        catalog.asset_directory("not-an-asset")


def test_catalog_index_is_deterministic_and_dependency_complete():
    spec = importlib.util.spec_from_file_location(
        "dranmar_asset_catalog_index_test_module", INDEX_GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generator
    spec.loader.exec_module(generator)
    first = generator.build_index()
    second = generator.build_index()
    assert first == second
    assert first["asset_count"] == len(first["assets"]) == 9
    assert all(
        len(asset["sha256"]) == 64
        and asset["file_count"] > 0
        and asset["usd_references_checked"] >= 0
        for asset in first["assets"].values()
    )


def test_installed_manifest_entries_match_current_repository_bytes():
    import json

    refresher_path = ROOT / "scripts/refresh_dranmar_installed_manifest_entries.py"
    spec = importlib.util.spec_from_file_location(
        "dranmar_manifest_refresh_test_module", refresher_path
    )
    assert spec is not None and spec.loader is not None
    refresher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = refresher
    spec.loader.exec_module(refresher)
    for relative in refresher.MANIFESTS:
        manifest = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        checked = 0
        for entry in manifest["files"]:
            source = refresher.installed_overlay_source(entry["path"])
            if source is None:
                continue
            checked += 1
            assert entry["bytes"] == source.stat().st_size
            assert entry["sha256"] == refresher.sha256(source)
        assert checked == manifest["installed_overlay_entries_refreshed"]
        assert checked > 0


def test_usd_mesh_normals_use_the_schema_attribute_not_a_primvar_alias():
    props_root = ROOT / (
        "source/extensions/orbit.surgical.assets/data/Props"
    )
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in props_root.rglob("*.usda")
        if "normal3f[] primvars:normals" in path.read_text(
            encoding="utf-8"
        )
    ]
    assert offenders == []


def test_autonomous_rescue_usda_metadata_and_variants_are_parseable():
    typed_field = re.compile(
        r"\b(?:asset|bool|double|float|int|string|token)\s+\w+\s*="
    )
    inline_dictionary = re.compile(
        r"\b(?:assetInfo|customData)\s*=\s*\{[^}\n]+\}"
    )
    inline_variant_body = re.compile(
        r'^\s*"[^"]+"\s*\{\s*(?:custom\s+)?'
        r"(?:asset|bool|double|float|int|string|token)\b[^}\n]*\}\s*$",
        re.MULTILINE,
    )
    offenders = []

    for path in sorted(AUTONOMOUS_RESCUE_ASSET_ROOT.glob("*.usda")):
        source = path.read_text(encoding="utf-8")
        for match in inline_dictionary.finditer(source):
            if len(typed_field.findall(match.group())) > 1:
                line = source.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.name}:{line}: inline metadata dictionary")
        for match in inline_variant_body.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.name}:{line}: inline variant body")

    assert offenders == []
