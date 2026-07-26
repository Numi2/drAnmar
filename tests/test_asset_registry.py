from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dr_anmar_asset_registry import (  # noqa: E402
    build_lock,
    catalog_lock_digest,
    discover_asset_units,
    load_policy,
    provider_roots,
    render_catalog_document,
    resolve_provider_asset,
    sha256_of_folder,
    validate_catalog,
    validate_portfolio,
    validate_release_artifacts,
    verify_lock,
)
from dr_anmar_i4h_adapter import (  # noqa: E402
    _dr_anmar_portfolio_assets,
    _repository_artifact_path,
    asset_catalog_payload,
)
from dr_anmar_procedures import PROCEDURE_ROOMS  # noqa: E402


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
    assert report["bytes"] <= 512 * 1024 * 1024
    assert not any(
        issue["code"].endswith("growth_budget_exceeded")
        for issue in report["issues"]
    )


def test_portfolio_contract_covers_every_declared_artifact() -> None:
    assert validate_portfolio(ROOT) == ()


def test_capability_payload_covers_the_authoritative_portfolio() -> None:
    portfolio = json.loads((ROOT / "physics_next/dr-anmar-assets.json").read_text(encoding="utf-8"))
    assets, portfolio_path, error = _dr_anmar_portfolio_assets()

    assert error is None
    assert portfolio_path == ROOT / "physics_next/dr-anmar-assets.json"
    assert {asset["id"] for asset in assets} == {asset["id"] for asset in portfolio["assets"]}
    assert all(asset["local_ready"] for asset in assets)
    assert all(asset["training_readiness"].startswith("available_") for asset in assets)
    assert all(asset["software_evidence"].startswith("repository_verified_") for asset in assets)
    assert all(asset["clinical_validation"] is False for asset in assets)


def test_capability_payload_exposes_the_release_lock_identity() -> None:
    release = asset_catalog_payload()["dr_anmar_release_lock"]

    assert release["ready"] is True
    assert release["schema"] == "dr.anmar.asset-catalog-lock.v3"
    assert release["self_digest_matches"] is True
    assert release["clinical_validation"] is False
    assert release["asset_units"] >= 20
    assert release["portfolio_assets"] == 21


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
    portfolio_root = repository / "physics_next"
    portfolio_root.mkdir()
    (portfolio_root / "dr-anmar-assets.json").write_text(
        json.dumps(
            {
                "schema": "dr.anmar.asset-portfolio.v2",
                "assets": [
                    {
                        "id": "fixture",
                        "asset": "assets/fixture.usda",
                        "live_integration": "test",
                        "product_capability": "simulation_training_component",
                        "training_readiness": "available_for_training_workcell_composition",
                        "software_evidence": "repository_verified_asset_closure",
                        "native_simulator_evidence": "not_recorded",
                        "real_world_evidence": "not_established",
                        "clinical_validation": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    lock = build_lock(repository)
    assert verify_lock(lock, repository) == ()

    (asset_root / "fixture.usda").write_text(
        '#usda 1.0\n(\n    defaultPrim = "Changed"\n)\n',
        encoding="utf-8",
    )
    assert verify_lock(lock, repository) == ("Asset unit changed: dr_anmar:.",)


def test_canonical_release_lock_and_generated_catalog_are_current() -> None:
    policy = load_policy(ROOT)
    lock_path = ROOT / policy["release"]["lock_path"]
    catalog_path = ROOT / policy["release"]["catalog_document_path"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    assert verify_lock(lock, ROOT) == ()
    assert lock["catalog_sha256"] == catalog_lock_digest(lock)
    assert catalog_path.read_text(encoding="utf-8") == render_catalog_document(lock)
    assert validate_release_artifacts(ROOT) == ()


def test_release_lock_rejects_duplicate_ids_and_stale_self_digest() -> None:
    lock = build_lock(ROOT)
    lock["assets"].append(dict(lock["assets"][0]))

    failures = verify_lock(lock, ROOT)

    assert "Asset-catalog lock contains duplicate asset IDs." in failures
    assert "Asset-catalog lock self-digest does not match." in failures


def test_product_capability_is_not_defined_by_negative_evidence_language() -> None:
    portfolio = json.loads(
        (ROOT / "physics_next/dr-anmar-assets.json").read_text(encoding="utf-8")
    )
    forbidden = ("research", "archived", "reduced_order", "unqualified", "blocked")

    for asset in portfolio["assets"]:
        product_surface = " ".join(
            (
                asset["product_capability"],
                asset["training_readiness"],
                asset["software_evidence"],
            )
        ).lower()
        assert not any(term in product_surface for term in forbidden), asset["id"]
