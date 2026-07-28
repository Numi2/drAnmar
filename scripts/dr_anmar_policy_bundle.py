#!/usr/bin/env python3
"""Fail-closed policy/checkpoint/controller binding for DrAnmar handovers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_BUNDLE_SCHEMA_VERSION = "dranmar-policy-bundle-1.0"

_ADAPTATION_METHODS = {
    "none": None,
    "giver": "configure_giver_adaptation",
    "pickup_recovery": "configure_pickup_recovery_adaptation",
    "recovery_receiver_grasp_retain": (
        "configure_recovery_receiver_grasp_retain_adaptation"
    ),
    "joint_transfer_acquisition": (
        "configure_joint_transfer_acquisition_adaptation"
    ),
    "transfer_refinement": "configure_transfer_refinement_adaptation",
    "deadline_recovery": "configure_deadline_recovery_adaptation",
    "frontier_hardening": "configure_frontier_hardening_adaptation",
}


class PolicyBundleError(ValueError):
    """A policy bundle is incomplete, inconsistent, or mismatched."""


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data with stable ordering and separators."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """Stream a file into SHA-256 without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_contract_sha256(bundle: dict[str, Any]) -> str:
    """Hash all bundle fields except the self-referential contract hash."""
    contract = dict(bundle)
    contract.pop("contract_sha256", None)
    return canonical_sha256(contract)


def _require_mapping(
    value: Any,
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyBundleError(f"{field} must be an object")
    return value


def validate_policy_bundle_document(
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Validate schema, types, and the immutable self-hash."""
    if bundle.get("schema_version") != POLICY_BUNDLE_SCHEMA_VERSION:
        raise PolicyBundleError(
            "unsupported policy bundle schema "
            f"{bundle.get('schema_version')!r}"
        )
    for field in (
        "bundle_id",
        "task",
        "adaptation_mode",
        "controller_profile",
        "checkpoint",
        "environment_runtime_contract_sha256",
        "runtime_expectations",
        "contract_sha256",
    ):
        if field not in bundle:
            raise PolicyBundleError(f"policy bundle is missing {field!r}")
    if not isinstance(bundle["bundle_id"], str) or not bundle["bundle_id"]:
        raise PolicyBundleError("bundle_id must be a non-empty string")
    if not isinstance(bundle["task"], str) or not bundle["task"]:
        raise PolicyBundleError("task must be a non-empty string")
    adaptation_mode = bundle["adaptation_mode"]
    if adaptation_mode not in _ADAPTATION_METHODS:
        raise PolicyBundleError(
            f"unsupported adaptation_mode {adaptation_mode!r}"
        )
    profile = _require_mapping(
        bundle["controller_profile"],
        "controller_profile",
    )
    if not isinstance(profile.get("name"), str) or not profile["name"]:
        raise PolicyBundleError(
            "controller_profile.name must be a non-empty string"
        )
    if not isinstance(profile.get("sha256"), str) or len(
        profile["sha256"]
    ) != 64:
        raise PolicyBundleError(
            "controller_profile.sha256 must be a SHA-256 hex digest"
        )
    checkpoint = _require_mapping(bundle["checkpoint"], "checkpoint")
    if not isinstance(checkpoint.get("sha256"), str) or len(
        checkpoint["sha256"]
    ) != 64:
        raise PolicyBundleError(
            "checkpoint.sha256 must be a SHA-256 hex digest"
        )
    environment_contracts = _require_mapping(
        bundle["environment_runtime_contract_sha256"],
        "environment_runtime_contract_sha256",
    )
    if not environment_contracts or not all(
        isinstance(task, str)
        and task
        and isinstance(digest, str)
        and len(digest) == 64
        for task, digest in environment_contracts.items()
    ):
        raise PolicyBundleError(
            "environment_runtime_contract_sha256 must map task names to "
            "SHA-256 hex digests"
        )
    _require_mapping(bundle["runtime_expectations"], "runtime_expectations")
    expected_hash = bundle_contract_sha256(bundle)
    if bundle["contract_sha256"] != expected_hash:
        raise PolicyBundleError(
            "policy bundle contract hash mismatch: "
            f"expected {expected_hash}, got {bundle['contract_sha256']}"
        )
    return bundle


def load_policy_bundle(path: Path) -> dict[str, Any]:
    """Load one JSON bundle and reject duplicate keys or trailing ambiguity."""

    def reject_duplicate_pairs(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PolicyBundleError(
                    f"duplicate JSON key {key!r} in policy bundle"
                )
            result[key] = value
        return result

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PolicyBundleError(
            f"cannot read policy bundle {path}: {error}"
        ) from error
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (json.JSONDecodeError, PolicyBundleError) as error:
        raise PolicyBundleError(
            f"invalid policy bundle JSON {path}: {error}"
        ) from error
    return validate_policy_bundle_document(
        _require_mapping(parsed, "policy bundle")
    )


def validate_bundle_invocation(
    bundle: dict[str, Any],
    *,
    task: str,
    checkpoint_path: Path,
    purpose: str = "play",
) -> None:
    """Bind a launch to the exact task and checkpoint bytes in the bundle."""
    if purpose not in {"play", "train"}:
        raise PolicyBundleError(
            f"unsupported policy bundle invocation purpose {purpose!r}"
        )
    allowed_tasks = [bundle["task"]]
    if purpose == "play":
        play_tasks = bundle.get("compatible_play_tasks", [])
        if not isinstance(play_tasks, list) or not all(
            isinstance(value, str) and value
            for value in play_tasks
        ):
            raise PolicyBundleError(
                "compatible_play_tasks must be a list of task names"
            )
        allowed_tasks.extend(play_tasks)
    if purpose == "train":
        training_tasks = bundle.get("compatible_training_tasks", [])
        if not isinstance(training_tasks, list) or not all(
            isinstance(value, str) and value
            for value in training_tasks
        ):
            raise PolicyBundleError(
                "compatible_training_tasks must be a list of task names"
            )
        allowed_tasks.extend(training_tasks)
    if task not in allowed_tasks:
        raise PolicyBundleError(
            f"bundle does not permit {purpose} task {task!r}; "
            f"allowed tasks: {allowed_tasks!r}"
        )
    if not checkpoint_path.is_file():
        raise PolicyBundleError(
            f"checkpoint not found: {checkpoint_path}"
        )
    actual_checkpoint_sha256 = file_sha256(checkpoint_path)
    expected_checkpoint_sha256 = bundle["checkpoint"]["sha256"]
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise PolicyBundleError(
            "checkpoint hash mismatch: expected "
            f"{expected_checkpoint_sha256}, got {actual_checkpoint_sha256}"
        )


def configure_policy_from_bundle(
    policy_model: Any,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Apply named profile and adaptation, then verify critical runtime state."""
    controller = getattr(policy_model, "controller", None)
    if controller is None:
        raise PolicyBundleError(
            "policy model does not expose an analytic controller"
        )
    configure_profile = getattr(controller, "configure_profile", None)
    if configure_profile is None:
        raise PolicyBundleError(
            "policy controller does not support versioned profiles"
        )
    profile = configure_profile(bundle["controller_profile"]["name"])
    if profile["sha256"] != bundle["controller_profile"]["sha256"]:
        raise PolicyBundleError(
            "controller profile hash mismatch: bundle expects "
            f"{bundle['controller_profile']['sha256']}, source provides "
            f"{profile['sha256']}"
        )
    adaptation_method_name = _ADAPTATION_METHODS[
        bundle["adaptation_mode"]
    ]
    if adaptation_method_name is not None:
        adaptation_method = getattr(
            policy_model,
            adaptation_method_name,
            None,
        )
        if adaptation_method is None:
            raise PolicyBundleError(
                "policy model does not support adaptation mode "
                f"{bundle['adaptation_mode']!r}"
            )
        adaptation_method()
    expectations = bundle["runtime_expectations"]
    if "residual_scale" in expectations:
        if not hasattr(policy_model, "residual_scale"):
            raise PolicyBundleError(
                "policy model does not expose residual_scale"
            )
        policy_model.residual_scale = float(expectations["residual_scale"])
    mismatches = policy_runtime_mismatches(policy_model, expectations)
    if mismatches:
        raise PolicyBundleError(
            "policy runtime does not match bundle:\n- "
            + "\n- ".join(mismatches)
        )
    return profile


def validate_environment_contract(
    bundle: dict[str, Any],
    task: str,
    actual_contract_sha256: str,
) -> None:
    """Reject serving under task semantics different from the bundle."""
    expected = bundle["environment_runtime_contract_sha256"].get(task)
    if expected is None:
        raise PolicyBundleError(
            f"bundle has no environment runtime contract for task {task!r}"
        )
    if actual_contract_sha256 != expected:
        raise PolicyBundleError(
            "environment runtime contract hash mismatch: expected "
            f"{expected}, got {actual_contract_sha256}"
        )


def policy_runtime_mismatches(
    policy_model: Any,
    expectations: dict[str, Any],
) -> list[str]:
    """Compare only explicit bundle expectations; unspecified fields are free."""
    mismatches: list[str] = []
    model_class = expectations.get("model_class")
    if model_class is not None and type(policy_model).__name__ != model_class:
        mismatches.append(
            f"model_class expected {model_class!r}, "
            f"got {type(policy_model).__name__!r}"
        )
    for field, expected in expectations.get("policy_fields", {}).items():
        if not hasattr(policy_model, field):
            mismatches.append(f"policy.{field} is missing")
            continue
        actual = getattr(policy_model, field)
        if actual != expected:
            mismatches.append(
                f"policy.{field} expected {expected!r}, got {actual!r}"
            )
    controller = getattr(policy_model, "controller", None)
    for field, expected in expectations.get(
        "controller_fields",
        {},
    ).items():
        if controller is None or not hasattr(controller, field):
            mismatches.append(f"controller.{field} is missing")
            continue
        actual = getattr(controller, field)
        if isinstance(expected, float):
            matches = isinstance(actual, (float, int)) and abs(
                float(actual) - expected
            ) <= 1.0e-12
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(
                f"controller.{field} expected {expected!r}, got {actual!r}"
            )
    return mismatches


def build_policy_bundle_document(
    *,
    bundle_id: str,
    task: str,
    checkpoint_path: Path,
    adaptation_mode: str,
    controller_profile: dict[str, str],
    runtime_expectations: dict[str, Any],
    environment_runtime_contract_sha256: dict[str, str],
    compatible_play_tasks: list[str] | None = None,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct and self-hash a bundle for a newly saved checkpoint."""
    if adaptation_mode not in _ADAPTATION_METHODS:
        raise PolicyBundleError(
            f"unsupported adaptation_mode {adaptation_mode!r}"
        )
    bundle: dict[str, Any] = {
        "schema_version": POLICY_BUNDLE_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "task": task,
        "adaptation_mode": adaptation_mode,
        "checkpoint": {
            "path_hint": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "controller_profile": controller_profile,
        "runtime_expectations": runtime_expectations,
        "environment_runtime_contract_sha256": (
            environment_runtime_contract_sha256
        ),
    }
    if compatible_play_tasks:
        bundle["compatible_play_tasks"] = list(
            compatible_play_tasks
        )
    if lineage:
        bundle["lineage"] = lineage
    bundle["contract_sha256"] = bundle_contract_sha256(bundle)
    return validate_policy_bundle_document(bundle)


def write_policy_bundle(
    path: Path,
    bundle: dict[str, Any],
) -> None:
    """Atomically publish canonical, human-readable JSON."""
    validate_policy_bundle_document(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a DrAnmar policy bundle without launching Isaac"
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        bundle = load_policy_bundle(args.bundle.resolve())
        validate_bundle_invocation(
            bundle,
            task=args.task,
            checkpoint_path=args.checkpoint.expanduser().resolve(),
            purpose="play",
        )
    except PolicyBundleError as error:
        print(f"error: {error}")
        return 2
    print(
        json.dumps(
            {
                "bundle_id": bundle["bundle_id"],
                "contract_sha256": bundle["contract_sha256"],
                "status": "valid",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
