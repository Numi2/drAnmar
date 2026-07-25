from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dr_anmar_asset_registry import load_policy  # noqa: E402
from dr_anmar_i4h_receipt import (  # noqa: E402
    hash_bundle_subpath,
    update_receipt,
    verify_receipt,
)


def _policy() -> dict:
    policy = load_policy(ROOT)
    policy["i4h_bundles"] = {
        "robot": ["Robots/dVRK"],
        "workcell": ["Props/OperatingTable"],
    }
    return policy


def _content_root(tmp_path: Path) -> Path:
    content_root = tmp_path / "724f82e"
    robot = content_root / "Robots/dVRK"
    table = content_root / "Props/OperatingTable"
    robot.mkdir(parents=True)
    table.mkdir(parents=True)
    (robot / "dvrk.usda").write_text("#usda 1.0\n", encoding="utf-8")
    (table / "table.usda").write_text("#usda 1.0\n", encoding="utf-8")
    return content_root


def test_partial_bundle_receipt_is_content_addressed(tmp_path: Path) -> None:
    policy = _policy()
    content_root = _content_root(tmp_path)

    receipt = update_receipt(policy, content_root, "robot")

    assert verify_receipt(receipt, policy) == ()
    artifact = receipt["bundles"]["robot"]["artifacts"][0]
    assert artifact["path"] == "Robots/dVRK"
    assert artifact["file_count"] == 1
    assert len(artifact["sha256"]) == 64
    assert receipt["clinical_validation"] is False


def test_receipt_detects_downloaded_content_drift(tmp_path: Path) -> None:
    policy = _policy()
    content_root = _content_root(tmp_path)
    receipt = update_receipt(policy, content_root, "robot")
    (content_root / "Robots/dVRK/dvrk.usda").write_text(
        '#usda 1.0\n(\n    defaultPrim = "Changed"\n)\n',
        encoding="utf-8",
    )

    assert verify_receipt(receipt, policy) == ("Bundle subpath changed: robot/Robots/dVRK",)


def test_recording_another_bundle_preserves_verified_bundles(tmp_path: Path) -> None:
    policy = _policy()
    content_root = _content_root(tmp_path)
    robot_receipt = update_receipt(policy, content_root, "robot")

    combined = update_receipt(
        policy,
        content_root,
        "workcell",
        existing=json.loads(json.dumps(robot_receipt)),
    )

    assert sorted(combined["bundles"]) == ["robot", "workcell"]
    assert verify_receipt(combined, policy) == ()


@pytest.mark.parametrize("subpath", ("../outside", "/tmp/outside", r"..\outside"))
def test_bundle_hasher_rejects_path_escape(tmp_path: Path, subpath: str) -> None:
    with pytest.raises(ValueError):
        hash_bundle_subpath(_content_root(tmp_path), subpath)


def test_receipt_rejects_a_different_catalog_pin(tmp_path: Path) -> None:
    policy = _policy()
    receipt = update_receipt(policy, _content_root(tmp_path), "robot")
    receipt["asset_hash"] = "different"

    assert verify_receipt(receipt, policy) == ("Receipt pin mismatch: asset_hash",)
