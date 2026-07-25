# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / (
    "source/extensions/orbit.surgical.assets/orbit/surgical/assets/"
    "dranmar_asset_catalog.py"
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
    assert len(catalog.DRANMAR_SIM_READY_ASSETS) == 8
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
