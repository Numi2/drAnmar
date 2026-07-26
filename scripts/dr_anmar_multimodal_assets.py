#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate and safely sample Dr.Anmar multimodal simulation asset bundles.

The contract intentionally separates three kinds of data:

* OpenUSD/PhysX state and native robot actions may be authoritative in a
  simulation episode.
* Camera frames, including generated frames, are observations.
* Generative media is never a robot-control or patient-effect authority.

This module has no Isaac Sim or third-party Python dependency. It is suitable
for release gates, ingestion services, and deterministic unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

BUNDLE_SCHEMA = "dr.anmar.multimodal-asset-bundle.v1"
ACTION_CONTRACT_SCHEMA = "dr.anmar.action-contract.v1"
ACTION_STREAM_SCHEMA = "dr.anmar.timestamped-action-stream.v1"
EVIDENCE_SCHEMA = "dr.anmar.generative-asset-evidence.v1"
BUNDLE_ENTRYPOINT = "asset_bundle.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SEMVER_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

ALLOWED_COMPONENT_ROLES = frozenset(
    {
        "source_image",
        "source_video",
        "source_statistics",
        "source_action_stream",
        "source_manifest",
        "model_checkpoint",
        "text_embedding",
        "action_contract",
        "dranmar_action_stream",
        "evidence",
    }
)
ALLOWED_IMAGE_TRANSFORMS = frozenset({"letterbox", "center_crop"})
GENERATIVE_AUTHORITY = {
    "visual_observation": "non_authoritative_preview_only",
    "simulation_state": "openusd_physx_and_timestamped_native_state_only",
    "robot_control": "never_from_generated_media",
    "patient_effects": "simulator_mechanics_and_sensors_only",
}


@dataclass(frozen=True)
class MultimodalIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ActionSample:
    timestamp_s: float
    action: tuple[float, ...]
    status: str


def image_transform_plan(
    source_size_px: tuple[int, int],
    target_size_px: tuple[int, int],
    mode: str,
) -> dict[str, Any]:
    """Return a deterministic aspect-preserving resize/pad or resize/crop plan."""

    source_width, source_height = source_size_px
    target_width, target_height = target_size_px
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("image dimensions must be positive")
    if mode not in ALLOWED_IMAGE_TRANSFORMS:
        raise ValueError(f"unsupported geometry-preserving transform: {mode!r}")
    width_scale = target_width / source_width
    height_scale = target_height / source_height
    scale = min(width_scale, height_scale) if mode == "letterbox" else max(width_scale, height_scale)
    resized_width = int(round(source_width * scale))
    resized_height = int(round(source_height * scale))
    if mode == "letterbox":
        horizontal = target_width - resized_width
        vertical = target_height - resized_height
        left = horizontal // 2
        top = vertical // 2
        return {
            "mode": mode,
            "resized_content_size_px": [resized_width, resized_height],
            "padding_left_top_right_bottom_px": [
                left,
                top,
                horizontal - left,
                vertical - top,
            ],
            "crop_left_top_right_bottom_px": [0, 0, 0, 0],
        }
    horizontal = resized_width - target_width
    vertical = resized_height - target_height
    left = horizontal // 2
    top = vertical // 2
    return {
        "mode": mode,
        "resized_content_size_px": [resized_width, resized_height],
        "padding_left_top_right_bottom_px": [0, 0, 0, 0],
        "crop_left_top_right_bottom_px": [
            left,
            top,
            horizontal - left,
            vertical - top,
        ],
    }


def _safe_relative(value: Any) -> Path:
    rendered = str(value)
    if not rendered or "\x00" in rendered or "\\" in rendered:
        raise ValueError(f"invalid relative path: {rendered!r}")
    pure = PurePosixPath(rendered)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"path must be normalized and relative: {rendered!r}")
    if ":" in pure.parts[0]:
        raise ValueError(f"path must not contain a drive or URI: {rendered!r}")
    return Path(*pure.parts)


def _contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def discover_multimodal_bundles(repository_root: Path) -> tuple[Path, ...]:
    """Return every repository-local multimodal asset-bundle entrypoint."""

    asset_root = repository_root.expanduser().resolve() / "assets/dr_anmar"
    if not asset_root.is_dir():
        return ()
    return tuple(sorted(asset_root.rglob(BUNDLE_ENTRYPOINT)))


def _error(
    issues: list[MultimodalIssue],
    code: str,
    path: Path | str,
    message: str,
    repository_root: Path,
) -> None:
    rendered = str(path)
    if isinstance(path, Path):
        resolved = path.resolve()
        if _contains(repository_root, resolved):
            rendered = resolved.relative_to(repository_root).as_posix()
    issues.append(MultimodalIssue(code, rendered, message))


def _validate_component(
    component: Mapping[str, Any],
    *,
    bundle_path: Path,
    repository_root: Path,
    issues: list[MultimodalIssue],
) -> None:
    component_id = str(component.get("id", "")).strip() or "<missing>"
    location = f"{bundle_path.as_posix()}#/components/{component_id}"
    role = component.get("role")
    if role not in ALLOWED_COMPONENT_ROLES:
        _error(
            issues,
            "invalid_component_role",
            location,
            f"Unsupported component role: {role!r}.",
            repository_root,
        )
    if not SHA256_PATTERN.fullmatch(str(component.get("sha256", ""))):
        _error(
            issues,
            "invalid_component_sha256",
            location,
            "Every component requires a full lowercase SHA-256.",
            repository_root,
        )
    size = component.get("bytes")
    if not isinstance(size, int) or size <= 0:
        _error(
            issues,
            "invalid_component_size",
            location,
            "Every component requires a positive byte size.",
            repository_root,
        )
    if not str(component.get("license", "")).strip():
        _error(
            issues,
            "missing_component_license",
            location,
            "Every component requires an explicit license identifier.",
            repository_root,
        )

    storage = component.get("storage")
    if storage == "local":
        try:
            relative = _safe_relative(component.get("path"))
        except ValueError as error:
            _error(issues, "unsafe_local_component", location, str(error), repository_root)
            return
        candidate = (bundle_path.parent / relative).resolve()
        if not _contains(bundle_path.parent.resolve(), candidate):
            _error(
                issues,
                "escaping_local_component",
                location,
                "Local components must remain inside their asset bundle.",
                repository_root,
            )
            return
        if not candidate.is_file():
            _error(
                issues,
                "missing_local_component",
                candidate,
                f"Missing component {component_id!r}.",
                repository_root,
            )
            return
        if isinstance(size, int) and candidate.stat().st_size != size:
            _error(
                issues,
                "local_component_size_mismatch",
                candidate,
                f"Expected {size} bytes, found {candidate.stat().st_size}.",
                repository_root,
            )
        expected_hash = str(component.get("sha256", ""))
        if SHA256_PATTERN.fullmatch(expected_hash):
            actual_hash = _sha256(candidate)
            if actual_hash != expected_hash:
                _error(
                    issues,
                    "local_component_hash_mismatch",
                    candidate,
                    f"Expected {expected_hash}, found {actual_hash}.",
                    repository_root,
                )
    elif storage == "external":
        uri = str(component.get("uri", ""))
        parsed = urlparse(uri)
        if parsed.scheme != "https" or not parsed.netloc:
            _error(
                issues,
                "unsafe_external_component_uri",
                location,
                "External components require an HTTPS content URI.",
                repository_root,
            )
        revision = str(component.get("revision", ""))
        if not COMMIT_PATTERN.fullmatch(revision):
            _error(
                issues,
                "unpinned_external_component",
                location,
                "External components require a full 40-character revision.",
                repository_root,
            )
        elif revision not in parsed.path:
            _error(
                issues,
                "mutable_external_component_uri",
                location,
                "The external content URI must contain its declared immutable revision.",
                repository_root,
            )
    else:
        _error(
            issues,
            "invalid_component_storage",
            location,
            f"Unsupported storage mode: {storage!r}.",
            repository_root,
        )

    if role in {"model_checkpoint", "text_embedding"}:
        loading = component.get("loading")
        if not isinstance(loading, Mapping):
            _error(
                issues,
                "missing_model_loading_policy",
                location,
                "Torch artifacts require an explicit loading policy.",
                repository_root,
            )
        else:
            if loading.get("runtime_enabled") is not False:
                _error(
                    issues,
                    "unsafe_model_runtime_enablement",
                    location,
                    "Unbundled Torch artifacts must have runtime_enabled=false.",
                    repository_root,
                )
            if loading.get("policy") != "quarantined_no_deserialization":
                _error(
                    issues,
                    "unsafe_model_loading_policy",
                    location,
                    "Torch pickle artifacts must remain quarantined from Dr.Anmar.",
                    repository_root,
                )


def _validate_media_contract(
    media: Any,
    *,
    bundle_path: Path,
    repository_root: Path,
    issues: list[MultimodalIssue],
) -> None:
    location = f"{bundle_path.as_posix()}#/media_contract"
    if not isinstance(media, Mapping):
        _error(issues, "missing_media_contract", location, "media_contract is required.", repository_root)
        return
    transform = media.get("model_input_transform")
    if not isinstance(transform, Mapping):
        _error(
            issues,
            "missing_image_transform",
            location,
            "A model_input_transform is required.",
            repository_root,
        )
        return
    if transform.get("mode") not in ALLOWED_IMAGE_TRANSFORMS:
        _error(
            issues,
            "distorting_image_transform",
            location,
            "Only letterbox or center_crop transforms preserve geometry.",
            repository_root,
        )
    if transform.get("direct_resize_allowed") is not False:
        _error(
            issues,
            "direct_resize_not_disabled",
            location,
            "direct_resize_allowed must be exactly false.",
            repository_root,
        )
    source = transform.get("source_size_px")
    target = transform.get("target_size_px")
    if not (
        isinstance(source, list)
        and len(source) == 2
        and isinstance(target, list)
        and len(target) == 2
        and all(isinstance(value, int) and value > 0 for value in (*source, *target))
    ):
        _error(
            issues,
            "invalid_image_dimensions",
            location,
            "source_size_px and target_size_px must be positive [width, height] pairs.",
            repository_root,
        )
    elif transform.get("mode") in ALLOWED_IMAGE_TRANSFORMS:
        expected_plan = image_transform_plan(tuple(source), tuple(target), str(transform.get("mode")))
        for key in (
            "resized_content_size_px",
            "padding_left_top_right_bottom_px",
        ):
            if transform.get(key) != expected_plan[key]:
                _error(
                    issues,
                    "incorrect_image_transform_plan",
                    location,
                    f"{key} must be {expected_plan[key]} for the declared transform.",
                    repository_root,
                )
        distortion = transform.get("geometric_aspect_distortion_fraction")
        if not isinstance(distortion, (int, float)) or abs(float(distortion)) > 1.0e-9:
            _error(
                issues,
                "nonzero_geometry_distortion",
                location,
                "Geometry-preserving transforms must declare zero aspect distortion.",
                repository_root,
            )
    if not str(media.get("decoded_color_space", "")).strip():
        _error(
            issues,
            "missing_color_contract",
            location,
            "Decoded color-space handling must be explicit.",
            repository_root,
        )
    if media.get("source_color_metadata") not in {"present", "absent"}:
        _error(
            issues,
            "unknown_color_metadata_state",
            location,
            "source_color_metadata must explicitly be present or absent.",
            repository_root,
        )


def load_action_contract(path: Path) -> dict[str, Any]:
    contract = _read_json(path)
    if not isinstance(contract, dict) or contract.get("schema") != ACTION_CONTRACT_SCHEMA:
        raise ValueError(f"{path} is not a {ACTION_CONTRACT_SCHEMA} contract")
    return contract


def load_action_stream(path: Path) -> dict[str, Any]:
    stream = _read_json(path)
    if not isinstance(stream, dict) or stream.get("schema") != ACTION_STREAM_SCHEMA:
        raise ValueError(f"{path} is not a {ACTION_STREAM_SCHEMA} stream")
    return stream


def _numeric_vector(value: Any, width: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{name} must contain exactly {width} values")
    numeric = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in numeric):
        raise ValueError(f"{name} contains NaN or infinity")
    return numeric


def validate_action_stream(
    stream: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate timing, dimensions, bounds, smoothness, and safe endpoints."""

    failures: list[str] = []
    if contract.get("schema") != ACTION_CONTRACT_SCHEMA:
        failures.append(f"action contract schema must be {ACTION_CONTRACT_SCHEMA}")
    if stream.get("schema") != ACTION_STREAM_SCHEMA:
        failures.append(f"action stream schema must be {ACTION_STREAM_SCHEMA}")
    dimensions = contract.get("dimensions")
    width = contract.get("action_dim")
    if not isinstance(width, int) or width <= 0:
        return ("action_contract.action_dim must be positive",)
    if not isinstance(dimensions, list) or len(dimensions) != width:
        return (f"action contract must describe all {width} dimensions",)
    if len({str(item.get("name")) for item in dimensions if isinstance(item, Mapping)}) != width:
        failures.append("action dimension names must be unique")
    neutral = contract.get("neutral_action")
    try:
        neutral_values = _numeric_vector(neutral, width, "neutral_action")
    except (TypeError, ValueError) as error:
        failures.append(str(error))
        neutral_values = tuple(0.0 for _ in range(width))

    frames = stream.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        return tuple(failures + ["action stream requires at least two frames"])
    if stream.get("frame_count") != len(frames):
        failures.append("frame_count does not match the number of frames")
    if stream.get("sample_hz") != contract.get("sample_hz"):
        failures.append("stream.sample_hz does not match the action contract")
    safety = contract.get("safety")
    if not isinstance(safety, Mapping):
        failures.append("action contract requires a safety object")
    else:
        for key in ("outside_trajectory", "stale_input", "nonfinite_input"):
            if safety.get(key) != "neutral_stop":
                failures.append(f"action contract safety.{key} must be neutral_stop")
        if safety.get("generated_media_may_supply_actions") is not False:
            failures.append("generated media must not supply robot actions")
        if safety.get("operator_lease_required_for_live_control") is not True:
            failures.append("live control must require an operator lease")
    timestamps: list[float] = []
    actions: list[tuple[float, ...]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            failures.append(f"frame {index} must be an object")
            continue
        try:
            timestamp_value = frame.get("timestamp_s")
            if not isinstance(timestamp_value, (int, float)):
                raise TypeError
            timestamp = float(timestamp_value)
            if not math.isfinite(timestamp):
                raise ValueError
        except (TypeError, ValueError):
            failures.append(f"frame {index} timestamp_s must be finite")
            continue
        try:
            action = _numeric_vector(frame.get("action"), width, f"frame {index} action")
        except (TypeError, ValueError) as error:
            failures.append(str(error))
            continue
        timestamps.append(timestamp)
        actions.append(action)
        if not str(frame.get("phase", "")).strip():
            failures.append(f"frame {index} requires a phase")
        for axis, (value, dimension) in enumerate(zip(action, dimensions, strict=True)):
            if not isinstance(dimension, Mapping):
                failures.append(f"dimension {axis} must be an object")
                continue
            bounds = dimension.get("bounds")
            if not (
                isinstance(bounds, list)
                and len(bounds) == 2
                and all(isinstance(item, (int, float)) for item in bounds)
            ):
                failures.append(f"dimension {axis} requires numeric bounds")
                continue
            if value < float(bounds[0]) - 1.0e-9 or value > float(bounds[1]) + 1.0e-9:
                failures.append(f"frame {index} dimension {axis} exceeds bounds")
            allowed = dimension.get("allowed_values")
            if allowed is not None and value not in tuple(float(item) for item in allowed):
                failures.append(f"frame {index} dimension {axis} is not an allowed discrete value")

    if len(timestamps) == len(frames):
        if abs(timestamps[0]) > 1.0e-9:
            failures.append("action stream must start at timestamp 0")
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            failures.append("action timestamps must be strictly increasing")
        sample_hz = contract.get("sample_hz")
        if isinstance(sample_hz, (int, float)) and sample_hz > 0:
            expected_step = 1.0 / float(sample_hz)
            if any(
                abs((current - previous) - expected_step) > 1.0e-6
                for previous, current in zip(timestamps, timestamps[1:])
            ):
                failures.append("action timestamps do not match contract.sample_hz")
        declared_duration = stream.get("duration_s")
        if not isinstance(declared_duration, (int, float)) or abs(
            float(declared_duration) - timestamps[-1]
        ) > 1.0e-6:
            failures.append("duration_s does not match the final frame")

    if actions:
        if any(abs(value - expected) > 1.0e-7 for value, expected in zip(actions[0], neutral_values)):
            failures.append("action stream must begin at the neutral action")
        if any(abs(value - expected) > 1.0e-7 for value, expected in zip(actions[-1], neutral_values)):
            failures.append("action stream must end at the neutral action")
        max_delta = contract.get("maximum_continuous_delta_per_sample")
        if isinstance(max_delta, (int, float)) and max_delta > 0:
            continuous_axes = [
                index
                for index, dimension in enumerate(dimensions)
                if isinstance(dimension, Mapping) and dimension.get("interpolation") == "linear"
            ]
            for previous, current in zip(actions, actions[1:]):
                if any(abs(current[axis] - previous[axis]) > float(max_delta) + 1.0e-9 for axis in continuous_axes):
                    failures.append("continuous action delta exceeds the per-sample safety envelope")
                    break
    if stream.get("paired_visual_observation") is not False:
        failures.append("repository fixture must not claim paired visual observation")
    if stream.get("training_reference_eligible") is not False:
        failures.append("unreviewed repository fixture must not be training-reference eligible")
    if stream.get("clinical_validation") is not False:
        failures.append("clinical_validation must be exactly false")
    return tuple(dict.fromkeys(failures))


def _stream_frames(stream: Mapping[str, Any], width: int) -> tuple[tuple[float, tuple[float, ...]], ...]:
    return tuple(
        (
            float(frame["timestamp_s"]),
            _numeric_vector(frame["action"], width, "action"),
        )
        for frame in stream["frames"]
    )


def safe_action_sample(
    stream: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    target_timestamp_s: float,
    received_monotonic_s: float,
    now_monotonic_s: float,
) -> ActionSample:
    """Interpolate one action or return the neutral action on any timing fault."""

    try:
        width = int(contract["action_dim"])
        neutral = _numeric_vector(contract["neutral_action"], width, "neutral_action")
        target = float(target_timestamp_s)
        received = float(received_monotonic_s)
        now = float(now_monotonic_s)
    except (KeyError, TypeError, ValueError):
        return ActionSample(float("nan"), (), "invalid_contract_neutral_stop")
    if not all(math.isfinite(value) for value in (target, received, now)):
        return ActionSample(target, neutral, "invalid_time_neutral_stop")
    if validate_action_stream(stream, contract):
        return ActionSample(target, neutral, "invalid_stream_neutral_stop")
    stale_timeout = float(contract["stale_timeout_s"])
    if now < received or now - received > stale_timeout:
        return ActionSample(target, neutral, "stale_input_neutral_stop")

    frames = _stream_frames(stream, width)
    if target < frames[0][0] or target > frames[-1][0]:
        return ActionSample(target, neutral, "outside_trajectory_neutral_stop")
    for index, (timestamp, action) in enumerate(frames):
        if abs(target - timestamp) <= 1.0e-12:
            return ActionSample(target, action, "exact")
        if timestamp > target:
            left_time, left = frames[index - 1]
            fraction = (target - left_time) / (timestamp - left_time)
            dimensions = contract["dimensions"]
            sampled = tuple(
                left[axis]
                if dimension.get("interpolation") == "hold"
                else left[axis] + fraction * (action[axis] - left[axis])
                for axis, dimension in enumerate(dimensions)
            )
            return ActionSample(target, sampled, "interpolated")
    return ActionSample(target, neutral, "outside_trajectory_neutral_stop")


def validate_bundle(
    bundle_path: Path,
    repository_root: Path,
) -> tuple[MultimodalIssue, ...]:
    """Validate a complete multimodal asset bundle without fetching externals."""

    repository_root = repository_root.expanduser().resolve()
    bundle_path = bundle_path.expanduser().resolve()
    issues: list[MultimodalIssue] = []
    try:
        bundle = _read_json(bundle_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _error(issues, "invalid_asset_bundle", bundle_path, str(error), repository_root)
        return tuple(issues)
    if not isinstance(bundle, Mapping) or bundle.get("schema") != BUNDLE_SCHEMA:
        _error(
            issues,
            "invalid_asset_bundle_schema",
            bundle_path,
            f"Expected schema {BUNDLE_SCHEMA!r}.",
            repository_root,
        )
        return tuple(issues)
    if not SEMVER_PATTERN.fullmatch(str(bundle.get("version", ""))):
        _error(issues, "invalid_bundle_version", bundle_path, "version must be SemVer.", repository_root)
    if bundle.get("clinical_validation") is not False:
        _error(
            issues,
            "unsafe_bundle_clinical_claim",
            bundle_path,
            "clinical_validation must be exactly false.",
            repository_root,
        )
    if bundle.get("authority") != GENERATIVE_AUTHORITY:
        _error(
            issues,
            "unsafe_generative_authority",
            bundle_path,
            "The bundle must preserve the Dr.Anmar generative-authority boundary.",
            repository_root,
        )

    source = bundle.get("source")
    if not isinstance(source, Mapping) or not COMMIT_PATTERN.fullmatch(str(source.get("commit", ""))):
        _error(
            issues,
            "unpinned_bundle_source",
            bundle_path,
            "source.commit must be a full 40-character Git revision.",
            repository_root,
        )
    source_mapping = source if isinstance(source, Mapping) else {}
    components = bundle.get("components")
    if not isinstance(components, list) or not components:
        _error(issues, "empty_bundle", bundle_path, "components must be non-empty.", repository_root)
        components = []
    component_ids: list[str] = []
    component_map: dict[str, Mapping[str, Any]] = {}
    for component in components:
        if not isinstance(component, Mapping):
            _error(
                issues,
                "invalid_component",
                bundle_path,
                "Every component must be an object.",
                repository_root,
            )
            continue
        component_id = str(component.get("id", "")).strip()
        component_ids.append(component_id)
        component_map[component_id] = component
        _validate_component(
            component,
            bundle_path=bundle_path,
            repository_root=repository_root,
            issues=issues,
        )
    if not all(component_ids) or len(component_ids) != len(set(component_ids)):
        _error(
            issues,
            "duplicate_component_id",
            bundle_path,
            "Component IDs must be present and unique.",
            repository_root,
        )
    required_roles = {
        "source_image",
        "source_video",
        "source_statistics",
        "source_action_stream",
        "source_manifest",
        "model_checkpoint",
        "text_embedding",
        "action_contract",
        "dranmar_action_stream",
        "evidence",
    }
    actual_roles = {str(component.get("role")) for component in components if isinstance(component, Mapping)}
    missing_roles = sorted(required_roles - actual_roles)
    if missing_roles:
        _error(
            issues,
            "incomplete_bundle",
            bundle_path,
            f"Missing component roles: {missing_roles}.",
            repository_root,
        )

    _validate_media_contract(
        bundle.get("media_contract"),
        bundle_path=bundle_path,
        repository_root=repository_root,
        issues=issues,
    )
    pairing = bundle.get("pairing")
    if not isinstance(pairing, Mapping) or pairing.get("source_pairing_status") != "rejected_unpaired":
        _error(
            issues,
            "unsafe_source_pairing_claim",
            bundle_path,
            "The audited source video and trajectory must remain rejected as unpaired.",
            repository_root,
        )
    elif pairing.get("dranmar_control_enabled") is not False:
        _error(
            issues,
            "unsafe_unpaired_control",
            bundle_path,
            "Unpaired source data cannot enable Dr.Anmar control.",
            repository_root,
        )

    local_by_role = {
        str(component.get("role")): component
        for component in components
        if isinstance(component, Mapping) and component.get("storage") == "local"
    }
    contract_component = local_by_role.get("action_contract")
    stream_component = local_by_role.get("dranmar_action_stream")
    evidence_component = local_by_role.get("evidence")
    try:
        if contract_component is None or stream_component is None:
            raise ValueError("local action contract and Dr.Anmar action stream are required")
        contract_path = bundle_path.parent / _safe_relative(contract_component["path"])
        stream_path = bundle_path.parent / _safe_relative(stream_component["path"])
        contract = load_action_contract(contract_path)
        stream = load_action_stream(stream_path)
        for failure in validate_action_stream(stream, contract):
            _error(
                issues,
                "invalid_dranmar_action_stream",
                stream_path,
                failure,
                repository_root,
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _error(
            issues,
            "unreadable_dranmar_action_contract",
            bundle_path,
            str(error),
            repository_root,
        )
    try:
        if evidence_component is None:
            raise ValueError("local evidence component is required")
        evidence_path = bundle_path.parent / _safe_relative(evidence_component["path"])
        evidence = _read_json(evidence_path)
        if not isinstance(evidence, Mapping) or evidence.get("schema") != EVIDENCE_SCHEMA:
            raise ValueError(f"evidence must use schema {EVIDENCE_SCHEMA}")
        if evidence.get("model_executed") is not False:
            raise ValueError("static asset audit must state model_executed=false")
        if evidence.get("clinical_validation") is not False:
            raise ValueError("evidence clinical_validation must be exactly false")
        if evidence.get("source_commit") != source_mapping.get("commit"):
            raise ValueError("evidence source_commit does not match the bundle source")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _error(issues, "invalid_bundle_evidence", bundle_path, str(error), repository_root)
    return tuple(sorted(issues, key=lambda issue: (issue.code, issue.path, issue.message)))


def validate_all_multimodal_bundles(repository_root: Path) -> tuple[MultimodalIssue, ...]:
    issues = [
        issue
        for bundle_path in discover_multimodal_bundles(repository_root)
        for issue in validate_bundle(bundle_path, repository_root)
    ]
    return tuple(sorted(issues, key=lambda issue: (issue.code, issue.path, issue.message)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Dr.Anmar multimodal simulation assets")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("bundle", nargs="?", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    issues = (
        validate_bundle(args.bundle, root)
        if args.bundle is not None
        else validate_all_multimodal_bundles(root)
    )
    payload = {
        "schema": "dr.anmar.multimodal-asset-validation.v1",
        "passed": not issues,
        "bundles": len(discover_multimodal_bundles(root)),
        "issues": [issue.__dict__ for issue in issues],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
