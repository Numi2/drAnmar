from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dr_anmar_asset_registry import (
    build_lock,
    discover_asset_units,
    load_policy,
    provider_roots,
    resolve_provider_asset,
    sha256_of_folder,
    validate_catalog,
    verify_lock,
)
from dr_anmar_i4h_adapter import (
    _dr_anmar_portfolio_assets,
    _repository_artifact_path,
)
from dr_anmar_procedures import PROCEDURE_ROOMS


def test_i4h_provider_is_fully_pinned() -> None:
    provider = load_policy(ROOT)["providers"]["nvidia_i4h"]

    assert provider["release"] == "v0.7.0"
    assert provider["asset_version"] == "0.7.0"
    assert provider["content_hash"] == "724f82e"
    assert len(provider["catalog_commit"]) == 40
    assert provider["license_review_required"] is True


def test_inventory_spans_extension_and_repository_assets() -> None:
    units = {unit.asset_id: unit for unit in discover_asset_units(ROOT)}

    assert "dr_anmar:Props/SurgicalClosure/SkinStapler" in units
    assert "dr_anmar:Robots/dVRK/PSM" in units
    assert "dr_anmar_repository:assets/dr_anmar/needle" in units
    assert all(unit.entrypoints for unit in units.values())
    assert all(unit.license_path for unit in units.values())


@pytest.mark.parametrize(
    "relative",
    ("../outside.usda", "/tmp/outside.usda", r"..\outside.usda"),
)
def test_provider_resolver_rejects_traversal(relative: str) -> None:
    roots = provider_roots(ROOT, i4h_content_root=ROOT / ".test-i4h")

    with pytest.raises(ValueError):
        resolve_provider_asset("dr_anmar", relative, roots)


def test_provider_resolver_resolves_local_asset() -> None:
    roots = provider_roots(ROOT, i4h_content_root=ROOT / ".test-i4h")

    resolved = resolve_provider_asset(
        "dr_anmar",
        "Props/SurgicalClosure/SkinStapler/skin_stapler_rigid_proxy.usda",
        roots,
        require=True,
    )

    assert resolved.is_file()
    assert ROOT in resolved.parents


def test_catalog_gate_covers_runtime_room_references() -> None:
    report = validate_catalog(
        ROOT,
        procedures=PROCEDURE_ROOMS,
        i4h_content_root=ROOT / ".test-i4h",
    )

    assert report["asset_units"] >= 20
    assert report["entrypoints"] >= 40
    assert report["passed"], json.dumps(report["issues"], indent=2)


def test_capability_payload_covers_the_authoritative_portfolio() -> None:
    portfolio = json.loads(
        (ROOT / "physics_next/dr-anmar-assets.json").read_text(encoding="utf-8")
    )
    assets, portfolio_path, error = _dr_anmar_portfolio_assets()

    assert error is None
    assert portfolio_path == ROOT / "physics_next/dr-anmar-assets.json"
    assert {asset["id"] for asset in assets} == {
        asset["id"] for asset in portfolio["assets"]
    }
    assert all(asset["local_ready"] for asset in assets)
    assert all(asset["clinical_validation"] is False for asset in assets)


@pytest.mark.parametrize(
    "relative",
    ("../outside.usda", "/tmp/outside.usda", r"..\outside.usda"),
)
def test_capability_portfolio_rejects_path_escape(relative: str) -> None:
    with pytest.raises(ValueError):
        _repository_artifact_path(relative)


def test_folder_hash_includes_names_and_contents(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "asset.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (second / "renamed.usda").write_text("#usda 1.0\n", encoding="utf-8")

    assert sha256_of_folder(first) != sha256_of_folder(second)


def test_release_lock_detects_asset_change(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    asset_root = repository / "assets"
    script_root = repository / "scripts"
    config_root = repository / "config"
    asset_root.mkdir(parents=True)
    script_root.mkdir()
    config_root.mkdir()
    (repository / "LICENSE").write_text("BSD-3-Clause\n", encoding="utf-8")
    (asset_root / "fixture.usda").write_text("#usda 1.0\n", encoding="utf-8")
    policy = load_policy(ROOT)
    policy["providers"] = {
        "dr_anmar": {
            "kind": "local",
            "root": "assets",
            "inventory_roots": ["."],
            "license_fallback": "LICENSE",
        },
        "nvidia_i4h": policy["providers"]["nvidia_i4h"],
    }
    (config_root / "dranmar_asset_catalog.json").write_text(
        json.dumps(policy),
        encoding="utf-8",
    )

    lock = build_lock(repository)
    assert verify_lock(lock, repository) == ()

    (asset_root / "fixture.usda").write_text(
        '#usda 1.0\n(\n    defaultPrim = "Changed"\n)\n',
        encoding="utf-8",
    )
    assert verify_lock(lock, repository) == ("Asset unit changed: dr_anmar:.",)
