"""Pure-Python regression gates for fail-closed handover policy bundles."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_ROOT / relative_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUNDLES = _load_module(
    "dr_anmar_policy_bundle",
    "scripts/dr_anmar_policy_bundle.py",
)
PROFILES = _load_module(
    "dr_anmar_controller_profiles",
    (
        "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/handover/controller_profiles.py"
    ),
)


def _synthetic_bundle(checkpoint_sha256: str) -> dict:
    profile = PROFILES.controller_profile("joint-transfer-v23")
    bundle = {
        "schema_version": BUNDLES.POLICY_BUNDLE_SCHEMA_VERSION,
        "bundle_id": "test-bundle",
        "task": "DrAnmar-Test-v0",
        "adaptation_mode": "joint_transfer_acquisition",
        "checkpoint": {"sha256": checkpoint_sha256},
        "environment_runtime_contract_sha256": {
            "DrAnmar-Test-v0": "1" * 64,
        },
        "controller_profile": {
            "name": profile["name"],
            "sha256": profile["sha256"],
        },
        "runtime_expectations": {
            "model_class": "MockPolicy",
            "residual_scale": 0.01,
            "policy_fields": {
                "pickup_recovery_adaptation_enabled": True,
                "joint_transfer_acquisition_adaptation_enabled": True,
            },
            "controller_fields": {
                "controller_profile_name": profile["name"],
                "controller_profile_sha256": profile["sha256"],
            },
        },
    }
    bundle["contract_sha256"] = BUNDLES.bundle_contract_sha256(bundle)
    return bundle


def test_versioned_v23_bundle_and_controller_profile_are_self_consistent():
    bundle_path = (
        REPO_ROOT
        / "config/policy_bundles/joint-transfer-v23.json"
    )
    bundle = BUNDLES.load_policy_bundle(bundle_path)
    profile = PROFILES.controller_profile(
        bundle["controller_profile"]["name"]
    )
    assert profile["sha256"] == bundle["controller_profile"]["sha256"]
    assert bundle["checkpoint"]["sha256"] == (
        "9853a65f75933b1b07578228414c33c4913d83224b669dff44691dac4c1e7d6b"
    )
    assert profile["values"]["canonical_needle_local_frames_enabled"] is False
    assert profile["values"]["custody_preserving_transport_enabled"] is False
    assert profile["values"]["transport_custody_latch_enabled"] is True
    assert profile["values"]["receiver_shaft_guard_all_pickups_enabled"] is False
    assert (
        profile["values"]["receiver_shaft_guard_segment_distance_enabled"]
        is False
    )
    assert profile["values"]["receiver_shaft_guard_preposition_enabled"] is False
    assert profile["values"]["receiver_distal_tool_guard_enabled"] is False
    assert profile["values"]["receiver_swept_tool_guard_enabled"] is False
    assert profile["values"]["receiver_preposition_translation_enabled"] is True
    assert profile["values"]["custody_quality_minimum_transport_scale"] == 0.20
    assert profile["values"]["custody_quality_axial_centering_enabled"] is False
    frontier = PROFILES.controller_profile("frontier-hardening-v24")
    assert frontier["values"]["transport_custody_latch_enabled"] is False
    assert frontier["values"]["receiver_shaft_guard_all_pickups_enabled"] is True
    assert (
        frontier["values"]["receiver_shaft_guard_segment_distance_enabled"]
        is True
    )
    assert frontier["values"]["receiver_shaft_guard_preposition_enabled"] is True
    assert frontier["values"]["receiver_distal_tool_guard_enabled"] is True
    assert frontier["values"]["receiver_swept_tool_guard_enabled"] is True
    assert (
        frontier["values"]["receiver_preposition_translation_enabled"]
        is False
    )
    assert (
        frontier["values"]["receiver_distal_tool_guard_minimum_distance_m"]
        == 0.008
    )
    assert frontier["values"]["custody_quality_minimum_transport_scale"] == 0.0
    assert frontier["values"]["custody_quality_axial_centering_enabled"] is True
    assert set(profile["implementation_sha256"]) == {
        "controller_profiles.py",
        "end_to_end_model.py",
        "residual_model.py",
    }
    assert all(
        len(digest) == 64
        for digest in profile["implementation_sha256"].values()
    )


def test_bundle_binds_exact_task_and_checkpoint_bytes(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"qualified-checkpoint")
    bundle = _synthetic_bundle(BUNDLES.file_sha256(checkpoint))

    BUNDLES.validate_policy_bundle_document(bundle)
    BUNDLES.validate_bundle_invocation(
        bundle,
        task="DrAnmar-Test-v0",
        checkpoint_path=checkpoint,
    )

    checkpoint.write_bytes(b"changed-checkpoint")
    with pytest.raises(BUNDLES.PolicyBundleError, match="hash mismatch"):
        BUNDLES.validate_bundle_invocation(
            bundle,
            task="DrAnmar-Test-v0",
            checkpoint_path=checkpoint,
        )
    with pytest.raises(BUNDLES.PolicyBundleError, match="does not permit"):
        BUNDLES.validate_bundle_invocation(
            bundle,
            task="DrAnmar-Other-v0",
            checkpoint_path=checkpoint,
        )


def test_bundle_rejects_mutation_and_duplicate_json_keys(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    bundle = _synthetic_bundle(BUNDLES.file_sha256(checkpoint))
    bundle["runtime_expectations"]["residual_scale"] = 0.2
    with pytest.raises(BUNDLES.PolicyBundleError, match="contract hash"):
        BUNDLES.validate_policy_bundle_document(bundle)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"one","schema_version":"two"}',
        encoding="utf-8",
    )
    with pytest.raises(BUNDLES.PolicyBundleError, match="duplicate JSON key"):
        BUNDLES.load_policy_bundle(duplicate)


def test_bundle_binds_exact_environment_runtime_contract():
    bundle = _synthetic_bundle("0" * 64)

    BUNDLES.validate_environment_contract(
        bundle,
        "DrAnmar-Test-v0",
        "1" * 64,
    )
    with pytest.raises(
        BUNDLES.PolicyBundleError,
        match="environment runtime contract hash mismatch",
    ):
        BUNDLES.validate_environment_contract(
            bundle,
            "DrAnmar-Test-v0",
            "2" * 64,
        )
    with pytest.raises(
        BUNDLES.PolicyBundleError,
        match="no environment runtime contract",
    ):
        BUNDLES.validate_environment_contract(
            bundle,
            "DrAnmar-Other-v0",
            "1" * 64,
        )


def test_bundle_configures_profile_and_adaptation_before_runtime_check():
    profile = PROFILES.controller_profile("joint-transfer-v23")
    controller = SimpleNamespace(**profile["values"])

    def configure_profile(self, name: str):
        return PROFILES.apply_controller_profile(self, name)

    controller.configure_profile = MethodType(
        configure_profile,
        controller,
    )

    class MockPolicy:
        def __init__(self):
            self.controller = controller
            self.residual_scale = 0.5
            self.pickup_recovery_adaptation_enabled = False
            self.joint_transfer_acquisition_adaptation_enabled = False

        def configure_joint_transfer_acquisition_adaptation(self):
            self.pickup_recovery_adaptation_enabled = True
            self.joint_transfer_acquisition_adaptation_enabled = True

    checkpoint_sha = "0" * 64
    bundle = _synthetic_bundle(checkpoint_sha)
    policy = MockPolicy()
    applied = BUNDLES.configure_policy_from_bundle(policy, bundle)
    assert applied["name"] == "joint-transfer-v23"
    assert policy.residual_scale == 0.01
    assert policy.pickup_recovery_adaptation_enabled is True
    assert policy.joint_transfer_acquisition_adaptation_enabled is True
    assert BUNDLES.policy_runtime_mismatches(
        policy,
        bundle["runtime_expectations"],
    ) == []


def test_policy_bundle_json_is_canonical_json():
    path = REPO_ROOT / "config/policy_bundles/joint-transfer-v23.json"
    assert isinstance(json.loads(path.read_text()), dict)
