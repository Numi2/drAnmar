# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned controller profiles for reproducible handover policies.

Controller fields are deliberately not checkpoint parameters.  A checkpoint
must therefore be paired with an immutable profile instead of inheriting
whatever defaults happen to be present in the current source tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


CONTROLLER_PROFILE_SCHEMA_VERSION = "dranmar-handover-controller-profile-1.0"

_V23_VALUES: dict[str, bool | float | int] = {
    "position_scale": 0.01,
    "orientation_scale": 0.05,
    "approach_height": 0.02,
    "lateral_alignment_threshold": 0.005,
    "close_distance": 0.005,
    "receiver_close_distance": 0.001,
    "slow_approach_radius": 0.02,
    "slow_approach_action_limit": 0.1,
    "receiver_contact_centering_action_limit": 0.005,
    "recovery_receiver_shaft_guard_start_from_tip_m": 0.025,
    "recovery_receiver_shaft_guard_activation_distance_m": 0.018,
    "recovery_receiver_shaft_guard_minimum_distance_m": 0.015,
    "receiver_jaw_proximal_offset_m": 0.0093,
    "transport_custody_latch_enabled": True,
    "receiver_preposition_enabled": True,
    "receiver_preposition_height": 0.025,
    "recovery_receiver_preposition_height": 0.025,
    "receiver_preposition_action_limit": 0.15,
    "receiver_contact_orientation_error_target_rad": 1.95,
    "receiver_adaptive_arc_enabled": False,
    "receiver_default_arc_fraction": 0.65,
    "needle_provisional_arc_fraction": 0.4,
    "needle_arc_extent_rad": 3.160456172145348,
    "normalized_contact_threshold": 0.002,
    "contact_force_observation_scale": 0.2,
    "giver_lift_contact_force_threshold_n": 0.01,
    "giver_pre_lift_min_contact_jaws": 2,
    "presentation_fraction_from_giver": 0.35,
    "presentation_height_in_robot_frame": -0.13,
    "presentation_ready_tolerance": 0.005,
    "presentation_hold_action_limit": 0.01,
    "minimum_lift_height_in_robot_frame": -0.139,
    "carry_lateral_action_limit": 0.06,
    "recovery_carry_lateral_action_limit": 0.08,
    "carry_lateral_ramp_height": 0.01,
    "pickup_vertical_action_limit": 0.01,
    "pickup_initial_vertical_action_limit": 0.01,
    "recovery_pickup_vertical_action_limit": 0.01,
    "pickup_deceleration_height": 0.01,
    "carry_vertical_action_limit": 0.015,
    "giver_lift_on_live_contact": True,
    "giver_pregrasp_orientation_action_limit": 0.6,
    "giver_pregrasp_orientation_tolerance": 0.035,
    "giver_transport_orientation_action_limit": 0.035,
    "receiver_orientation_action_limit": 0.6,
    "receiver_tangent_delta_rad": 0.7901140430363371,
    "receiver_crossing_angle_rad": 2.351478610553456,
    "receiver_roll_offset_rad": 3.141592653589793,
    "receiver_residual_enabled_for_learning": True,
    "receiver_grasp_retain_residual_enabled_for_learning": False,
    "giver_grasp_x": 0.0007375535249017802,
    "giver_grasp_y": 0.005600696415109648,
    "giver_grasp_z": 0.0006,
    "receiver_grasp_x": 0.0019002163218475414,
    "receiver_grasp_y": -0.009119058578501121,
    "receiver_grasp_z": -0.003,
    # These switches preserve the exact feature/controller semantics that
    # produced the v23 evidence. They are intentionally false.
    "canonical_needle_local_frames_enabled": False,
    "custody_quality_features_enabled": False,
    "custody_preserving_transport_enabled": False,
    "custody_quality_slow_threshold": 0.55,
    "custody_quality_stop_threshold": 0.30,
    "custody_quality_centering_action_limit": 0.02,
    "custody_quality_minimum_transport_scale": 0.20,
}

_PROFILE_VALUES: dict[str, dict[str, bool | float | int]] = {
    "joint-transfer-v23": dict(_V23_VALUES),
    "frontier-hardening-v24": {
        **_V23_VALUES,
        "canonical_needle_local_frames_enabled": True,
        "custody_quality_features_enabled": True,
        "custody_preserving_transport_enabled": True,
        # A two-jaw grasp can be usable while briefly asymmetric.  This score
        # slows transport before physical custody is lost; it does not create
        # contact, success, or permission to open the giver.
        "custody_quality_slow_threshold": 0.55,
        "custody_quality_stop_threshold": 0.30,
        "custody_quality_centering_action_limit": 0.02,
        "custody_quality_minimum_transport_scale": 0.20,
    },
}


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data with stable ordering and separators."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def controller_implementation_sha256() -> dict[str, str]:
    """Bind profiles to the source that interprets their scalar values."""
    source_root = Path(__file__).resolve().parent
    return {
        "controller_profiles.py": _source_file_sha256(
            source_root / "controller_profiles.py"
        ),
        "end_to_end_model.py": _source_file_sha256(
            source_root / "end_to_end_model.py"
        ),
        "residual_model.py": _source_file_sha256(
            source_root / "residual_model.py"
        ),
    }


def controller_profile(name: str) -> dict[str, Any]:
    """Return an isolated profile document, failing closed on unknown names."""
    try:
        values = _PROFILE_VALUES[name]
    except KeyError as error:
        available = ", ".join(sorted(_PROFILE_VALUES))
        raise ValueError(
            f"unknown handover controller profile {name!r}; "
            f"available profiles: {available}"
        ) from error
    document = {
        "schema_version": CONTROLLER_PROFILE_SCHEMA_VERSION,
        "name": name,
        "implementation_sha256": controller_implementation_sha256(),
        "values": copy.deepcopy(values),
    }
    document["sha256"] = canonical_sha256(document)
    return document


def available_controller_profiles() -> tuple[str, ...]:
    """Return stable public profile names."""
    return tuple(sorted(_PROFILE_VALUES))


def apply_controller_profile(controller: Any, name: str) -> dict[str, Any]:
    """Apply every profile field and reject incomplete controller classes."""
    profile = controller_profile(name)
    missing = [
        field
        for field in profile["values"]
        if not hasattr(controller, field)
    ]
    if missing:
        raise AttributeError(
            f"controller cannot apply profile {name!r}; missing fields: "
            + ", ".join(sorted(missing))
        )
    for field, value in profile["values"].items():
        setattr(controller, field, value)
    controller.controller_profile_name = name
    controller.controller_profile_sha256 = profile["sha256"]
    return profile


def controller_profile_mismatches(
    controller: Any,
    profile: dict[str, Any],
) -> list[str]:
    """Describe runtime deviations from an already validated profile."""
    mismatches: list[str] = []
    for field, expected in profile["values"].items():
        if not hasattr(controller, field):
            mismatches.append(f"controller.{field}: missing")
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
                f"controller.{field}: expected {expected!r}, got {actual!r}"
            )
    return mismatches
