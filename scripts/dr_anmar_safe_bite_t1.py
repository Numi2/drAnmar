#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate and sample the CPU-only T1 safe-bite entry-frame contract."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "config/dranmar_safe_bite_t1.json"
DEFAULT_GEOMETRY_CONTRACT = (
    ROOT
    / "source/extensions/orbit.surgical.assets"
    / "data/Props/SurgicalTissue/NeedleReadyTissueUnit"
    / "geometry_contract.json"
)
DEFAULT_GEOMETRY_REPORT = (
    ROOT
    / "source/extensions/orbit.surgical.assets"
    / "data/Props/SurgicalTissue/NeedleReadyTissueUnit"
    / "geometry_report.json"
)


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class SafeBiteEntryFrame:
    """One deterministic entry-frame sample in tissue-local coordinates."""

    seed: int
    environment_index: int
    flap: str
    bite_distance_from_wound_m: float
    surface_point_m: Vec3
    target_tip_position_m: Vec3
    desired_tip_direction: Vec3
    desired_needle_plane_normal: Vec3
    stand_off_m: float
    entry_angle_from_surface_normal_rad: float


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low, high = map(float, value)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{name} is not a finite ordered range")
    return low, high


def _norm(vector: Vec3) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _dot(left: Vec3, right: Vec3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalized(vector: Vec3, name: str) -> Vec3:
    length = _norm(vector)
    if not math.isfinite(length) or length <= 1.0e-12:
        raise ValueError(f"{name} is degenerate")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def orthonormalize_plane_normal(
    desired_tip_direction: Vec3,
    longitudinal_tangent: Vec3,
    surface_normal: Vec3,
) -> Vec3:
    """Gram-Schmidt a live longitudinal axis against the tip direction."""

    tip = _normalized(desired_tip_direction, "desired tip direction")
    tangent = _normalized(longitudinal_tangent, "longitudinal tangent")
    projection = _dot(tangent, tip)
    residual = tuple(tangent[index] - projection * tip[index] for index in range(3))
    if _norm(residual) <= 1.0e-12:
        residual = _cross(
            _normalized(surface_normal, "surface normal"),
            tip,
        )
    plane = _normalized(residual, "needle plane normal")
    if abs(_dot(plane, tip)) > 1.0e-12:
        raise AssertionError("Gram-Schmidt result is not orthogonal")
    return plane


def _expected_surface_particle_nodes(
    geometry_contract: dict[str, Any],
    lod: str,
) -> int:
    """Count structured-grid boundary vertices without reading generated USD."""

    geometry = geometry_contract["geometry"]
    lod_contract = geometry_contract["lods"][lod]
    components = int(geometry["components"])
    cells_x = int(lod_contract["cells_per_flap_x"])
    cells_y = int(lod_contract["cells_y"])
    layers = len(lod_contract["z_fractions"])
    total_per_component = (cells_x + 1) * (cells_y + 1) * layers
    interior_per_component = max(0, cells_x - 1) * max(0, cells_y - 1) * max(0, layers - 2)
    return components * (total_per_component - interior_per_component)


def estimate_contact_pipeline_memory_bytes(
    *,
    soft_candidate_pairs: int,
    soft_contact_capacity: int,
    rigid_sensor_contact_capacity: int,
    memory_estimator: dict[str, Any],
) -> int:
    """Mirror the native manager's conservative contact allocation estimate."""

    return (
        int(soft_candidate_pairs) * int(memory_estimator["bytes_per_candidate_pair"])
        + int(soft_contact_capacity) * int(memory_estimator["bytes_per_soft_contact"])
        + int(rigid_sensor_contact_capacity)
        * int(memory_estimator["bytes_per_rigid_sensor_contact"])
    )


def preflight_launch_profile(
    contract: dict[str, Any],
    geometry_contract: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    """Reject an unbounded or internally inconsistent launch before Isaac."""

    pipeline = contract["scene"]["contact_pipeline"]
    try:
        profile = contract["launch_profiles"][profile_name]
    except KeyError as error:
        raise ValueError(f"Unknown T1 launch profile {profile_name!r}") from error
    environment_count = int(profile["environment_count"])
    if environment_count <= 0:
        raise ValueError("T1 launch environment count must be positive")
    if environment_count > int(pipeline["maximum_environment_count"]):
        raise ValueError("T1 launch exceeds the qualified environment count")
    lod = str(profile["tissue_lod"])
    if lod not in geometry_contract["lods"]:
        raise ValueError("T1 launch selects an unknown tissue LOD")
    surface_particles = _expected_surface_particle_nodes(
        geometry_contract,
        lod,
    )
    expected_surface = int(pipeline["expected_surface_particles_by_lod"][lod])
    if surface_particles != expected_surface:
        raise ValueError("T1 boundary-particle contract drifted")
    if int(profile["expected_surface_particles_per_environment"]) != (surface_particles):
        raise ValueError("T1 launch boundary-particle count drifted")
    shape_count = int(pipeline["expected_approved_soft_shapes_per_environment"])
    if int(profile["expected_approved_soft_shapes_per_environment"]) != (shape_count):
        raise ValueError("T1 launch approved-shape count drifted")

    pair_count = environment_count * surface_particles * shape_count
    if pair_count != int(profile["expected_soft_candidate_pairs"]):
        raise ValueError("T1 launch candidate-pair count drifted")
    if pair_count > int(pipeline["maximum_soft_candidate_pairs"]):
        raise ValueError("T1 launch exceeds the soft candidate-pair limit")
    soft_capacity = environment_count * int(pipeline["soft_contacts_per_environment"])
    rigid_capacity = environment_count * int(pipeline["rigid_sensor_contacts_per_environment"])
    if soft_capacity != int(profile["soft_contact_capacity"]):
        raise ValueError("T1 launch soft-contact capacity drifted")
    if rigid_capacity != int(profile["rigid_sensor_contact_capacity"]):
        raise ValueError("T1 launch rigid-sensor capacity drifted")
    estimated_bytes = estimate_contact_pipeline_memory_bytes(
        soft_candidate_pairs=pair_count,
        soft_contact_capacity=soft_capacity,
        rigid_sensor_contact_capacity=rigid_capacity,
        memory_estimator=pipeline["memory_estimator"],
    )
    if estimated_bytes != int(profile["estimated_contact_pipeline_bytes"]):
        raise ValueError("T1 launch contact-memory estimate drifted")
    if estimated_bytes > int(pipeline["maximum_contact_pipeline_memory_bytes"]):
        raise ValueError("T1 launch exceeds the contact-memory limit")
    if not bool(profile["execution_requires_explicit_user_approval"]):
        raise ValueError("T1 launch profile must retain its approval gate")
    return {
        "profile": profile_name,
        "environment_count": environment_count,
        "tissue_lod": lod,
        "surface_particles_per_environment": surface_particles,
        "approved_soft_shapes_per_environment": shape_count,
        "soft_candidate_pairs": pair_count,
        "soft_contact_capacity": soft_capacity,
        "rigid_sensor_contact_capacity": rigid_capacity,
        "estimated_contact_pipeline_bytes": estimated_bytes,
        "execution_requires_explicit_user_approval": True,
    }


def _expected_outer_attachment_nodes(
    geometry_contract: dict[str, Any],
    lod: str,
) -> int:
    geometry = geometry_contract["geometry"]
    semantics = geometry_contract["semantics"]
    lod_contract = geometry_contract["lods"][lod]
    width = float(geometry["overall_width_m"])
    depth = float(geometry["depth_m"])
    gap = float(geometry["rest_wound_gap_m"])
    bevel = float(geometry["wound_bevel_m"])
    amplitude = float(geometry["wound_irregularity_amplitude_m"])
    wavelength = float(geometry["wound_irregularity_wavelength_m"])
    attachment_width = float(semantics["outer_attachment_width_m"])
    cells_x = int(lod_contract["cells_per_flap_x"])
    cells_y = int(lod_contract["cells_y"])
    edge_power = float(lod_contract["wound_edge_refinement_power"])
    count = 0
    for component in range(2):
        for w_fraction in map(float, lod_contract["z_fractions"]):
            for y_index in range(cells_y + 1):
                y = -depth / 2.0 + depth * y_index / cells_y
                wound_offset = amplitude * math.sin(2.0 * math.pi * (y + depth / 2.0) / wavelength)
                if component == 0:
                    outer_x = -width / 2.0
                    inner_x = -gap / 2.0 + wound_offset - bevel * w_fraction
                else:
                    inner_x = gap / 2.0 + wound_offset + bevel * w_fraction
                    outer_x = width / 2.0
                for x_index in range(cells_x + 1):
                    u_fraction = x_index / cells_x
                    if component == 0:
                        shaped_u = 1.0 - (1.0 - u_fraction) ** edge_power
                        x = outer_x + (inner_x - outer_x) * shaped_u
                    else:
                        shaped_u = u_fraction**edge_power
                        x = inner_x + (outer_x - inner_x) * shaped_u
                    if width / 2.0 - abs(x) <= attachment_width + 1.0e-12:
                        count += 1
    return count


def validate_contract(
    contract: dict[str, Any],
    geometry_contract: dict[str, Any],
    geometry_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail on an impossible or reward-hackable T1 contract."""

    sampling = contract["sampling"]
    success = contract["success"]
    rewards = contract["rewards"]
    puncture = contract["puncture_transition"]
    fixture = contract["scene"]["fixture"]
    curriculum = contract["handover_snapshot_curriculum"]
    contact_pipeline = contract["scene"]["contact_pipeline"]
    geometry = geometry_contract["geometry"]
    semantics = geometry_contract["semantics"]

    bite_low, bite_high = _range(
        sampling["bite_distance_from_wound_m"],
        "sampling.bite_distance_from_wound_m",
    )
    asset_low, asset_high = _range(
        semantics["safe_bite_distance_from_wound_m"],
        "geometry.semantics.safe_bite_distance_from_wound_m",
    )
    if bite_low < asset_low or bite_high > asset_high:
        raise ValueError("T1 bite sampler escapes the asset safe-bite region")

    stand_off_low, stand_off_high = _range(
        sampling["stand_off_m"],
        "sampling.stand_off_m",
    )
    if stand_off_low <= float(success["minimum_tissue_clearance_m"]):
        raise ValueError("T1 stand-off must exceed the contact clearance")
    if float(success["position_tolerance_m"]) > stand_off_high:
        raise ValueError("Position tolerance cannot be larger than the maximum stand-off")

    angle_low, angle_high = _range(
        sampling["entry_angle_from_surface_normal_deg"],
        "sampling.entry_angle_from_surface_normal_deg",
    )
    if not (0.0 < angle_low <= angle_high < 90.0):
        raise ValueError("Entry angle must point inward through the surface")

    depth = float(geometry["depth_m"])
    margin = float(sampling["longitudinal_end_margin_m"])
    if margin < float(semantics["longitudinal_end_margin_m"]):
        raise ValueError("T1 longitudinal margin weakens the asset contract")
    if margin * 2.0 >= depth:
        raise ValueError("T1 longitudinal margin removes the sampling region")
    if fixture["mode"] != "kinematic_authored_outer_attachment_band":
        raise ValueError("T1 must fixture the authored outer attachment band")
    if float(fixture["selection_tolerance_m"]) < 0.0:
        raise ValueError("T1 attachment tolerance cannot be negative")
    expected_by_lod = fixture["expected_anchor_nodes_by_lod"]
    resolved_by_lod = {}
    for lod in geometry_contract["lods"]:
        resolved = _expected_outer_attachment_nodes(
            geometry_contract,
            lod,
        )
        resolved_by_lod[lod] = resolved
        if int(expected_by_lod[lod]) != resolved:
            raise ValueError(f"T1 fixture count disagrees with the {lod} LOD")
        if geometry_report is not None:
            reported = int(geometry_report["lods"][lod]["node_set_counts"]["anchor_outer"])
            if reported != resolved:
                raise ValueError(f"T1 fixture count disagrees with the {lod} geometry report")
    selected_lods = {
        str(contract["scene"]["tissue_lod"]),
        str(contract["scene"]["continuation_tissue_lod"]),
    }
    if not selected_lods.issubset(geometry_contract["lods"]):
        raise ValueError("T1 selects an unknown tissue LOD")
    if not bool(fixture["wound_and_safe_bite_regions_remain_dynamic"]):
        raise ValueError("T1 fixture must leave the wound and bite region dynamic")
    stride = int(curriculum["full_chain_stride"])
    if curriculum["restore_schedule"] != "per_environment_rotating_quota":
        raise ValueError("T1 snapshot curriculum must enforce a rotating quota")
    if stride < 2:
        raise ValueError("T1 full-chain stride must be at least two")
    required_full_chain_fraction = 1.0 / stride
    if not math.isclose(
        required_full_chain_fraction,
        float(curriculum["minimum_full_chain_fraction"]),
        abs_tol=1.0e-12,
    ):
        raise ValueError("T1 full-chain stride disagrees with its minimum")
    if not math.isclose(
        1.0 - required_full_chain_fraction,
        float(curriculum["restore_probability"]),
        abs_tol=1.0e-12,
    ):
        raise ValueError("T1 restore quota disagrees with its probability")

    if int(success["stable_control_steps"]) < 2:
        raise ValueError("A one-frame entry cannot establish stable readiness")
    post_arm_limit = float(success["post_arm_inward_action_limit"])
    if not 0.0 < post_arm_limit <= 0.1:
        raise ValueError("Post-arm inward action must stay in (0.0, 0.1]")
    if float(rewards["holding_still_reward"]) != 0.0:
        raise ValueError("Holding still must not accumulate positive reward")
    if float(rewards["absolute_proximity_reward"]) != 0.0:
        raise ValueError("Absolute proximity must remain diagnostic only")
    if float(rewards["contact_reward"]) != 0.0:
        raise ValueError("Contact cannot be rewarded in T1")
    if rewards["progress_definition"] != (
        "previous normalized pose error minus current normalized pose error"
    ):
        raise ValueError("T1 must use delta progress instead of occupancy reward")
    if bool(puncture["mechanically_blocked_by_t1"]):
        raise ValueError("T1 must not mechanically block authorized puncture")
    if bool(puncture["policy_written_puncture_state_allowed"]):
        raise ValueError("A policy-written puncture flag is not physical evidence")
    if bool(puncture["generic_needle_contact_is_puncture"]):
        raise ValueError("Generic needle contact cannot be called puncture")
    if float(puncture["entry_contact_roi_radius_m"]) <= 0.0:
        raise ValueError("T1 entry contact ROI radius must be positive")
    if not bool(contract["scene"]["contact_observer"]["fail_closed_on_stale_receipt_or_overflow"]):
        raise ValueError("T1 contact authority must fail closed")
    if int(contact_pipeline["maximum_environment_count"]) != 2400:
        raise ValueError("T1 qualified training ceiling must remain 2400")
    launch_preflight = {
        profile_name: preflight_launch_profile(
            contract,
            geometry_contract,
            profile_name,
        )
        for profile_name in contract["launch_profiles"]
    }
    if launch_preflight["training_2400"]["environment_count"] != 2400:
        raise ValueError("T1 training profile must request 2400 environments")
    if launch_preflight["training_2400"]["tissue_lod"] != str(contract["scene"]["tissue_lod"]):
        raise ValueError("T1 training launch LOD disagrees with the scene")
    if launch_preflight["contact_qualification_256"]["tissue_lod"] != str(
        contract["scene"]["continuation_tissue_lod"]
    ):
        raise ValueError("T1 contact launch LOD disagrees with the scene")

    return {
        "schema": "dr.anmar.safe-bite-t1-contract-validation.v2",
        "contract_id": contract["id"],
        "asset_id": geometry_contract["id"],
        "valid": True,
        "safe_bite_range_m": [bite_low, bite_high],
        "stand_off_range_m": [stand_off_low, stand_off_high],
        "entry_angle_range_deg": [angle_low, angle_high],
        "fixture_anchor_nodes_by_lod": resolved_by_lod,
        "fixture_geometry_report_cross_checked": (geometry_report is not None),
        "launch_preflight": launch_preflight,
        "full_chain_stride": stride,
        "puncture_mechanically_blocked": False,
        "clinical_validation": False,
    }


def _surface_point(
    *,
    geometry: dict[str, Any],
    component: int,
    y: float,
    bite_distance: float,
) -> Vec3:
    depth = float(geometry["depth_m"])
    width = float(geometry["overall_width_m"])
    gap = float(geometry["rest_wound_gap_m"])
    bevel = float(geometry["wound_bevel_m"])
    irregularity = float(geometry["wound_irregularity_amplitude_m"])
    irregularity_wavelength = float(geometry["wound_irregularity_wavelength_m"])
    thickness = float(geometry["thickness_m"])
    topography = float(geometry["surface_topography_amplitude_m"])
    topography_wavelength = float(geometry["surface_topography_wavelength_m"])
    wound_offset = irregularity * math.sin(
        2.0 * math.pi * (y + depth / 2.0) / irregularity_wavelength
    )
    if component == 0:
        wound_x = -gap / 2.0 + wound_offset - bevel
        x = wound_x - bite_distance
    else:
        wound_x = gap / 2.0 + wound_offset + bevel
        x = wound_x + bite_distance
    centrality = max(0.0, 1.0 - abs(x) / max(width / 2.0, 1.0e-9))
    z = thickness / 2.0 + (
        topography
        * centrality
        * math.sin(
            2.0 * math.pi * (y + depth / 2.0) / topography_wavelength
            + (0.35 if component == 0 else -0.35)
        )
    )
    return (x, y, z)


def sample_entry_frame(
    contract: dict[str, Any],
    geometry_contract: dict[str, Any],
    *,
    seed: int,
    environment_index: int,
) -> SafeBiteEntryFrame:
    """Return a reproducible sample without importing Isaac Lab or Torch."""

    validate_contract(contract, geometry_contract)
    generator = random.Random(
        (int(seed) * 1_000_003 + int(environment_index) * 97_409) & 0xFFFFFFFF
    )
    sampling = contract["sampling"]
    geometry = geometry_contract["geometry"]
    depth = float(geometry["depth_m"])
    margin = float(sampling["longitudinal_end_margin_m"])
    bite_low, bite_high = map(float, sampling["bite_distance_from_wound_m"])
    stand_off_low, stand_off_high = map(float, sampling["stand_off_m"])
    angle_low, angle_high = map(float, sampling["entry_angle_from_surface_normal_deg"])

    component = 0 if generator.random() < 0.5 else 1
    flap = "left" if component == 0 else "right"
    y = generator.uniform(-depth / 2.0 + margin, depth / 2.0 - margin)
    bite_distance = generator.uniform(bite_low, bite_high)
    stand_off = generator.uniform(stand_off_low, stand_off_high)
    angle = math.radians(generator.uniform(angle_low, angle_high))
    surface = _surface_point(
        geometry=geometry,
        component=component,
        y=y,
        bite_distance=bite_distance,
    )
    inward_x = 1.0 if component == 0 else -1.0
    direction = (
        inward_x * math.sin(angle),
        0.0,
        -math.cos(angle),
    )
    target = (surface[0], surface[1], surface[2] + stand_off)
    plane_normal = orthonormalize_plane_normal(
        direction,
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    if not math.isclose(_norm(direction), 1.0, abs_tol=1.0e-12):
        raise AssertionError("Sampled tip direction is not unit length")
    return SafeBiteEntryFrame(
        seed=int(seed),
        environment_index=int(environment_index),
        flap=flap,
        bite_distance_from_wound_m=bite_distance,
        surface_point_m=surface,
        target_tip_position_m=target,
        desired_tip_direction=direction,
        desired_needle_plane_normal=plane_normal,
        stand_off_m=stand_off,
        entry_angle_from_surface_normal_rad=angle,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
    )
    parser.add_argument(
        "--geometry-contract",
        type=Path,
        default=DEFAULT_GEOMETRY_CONTRACT,
    )
    parser.add_argument(
        "--geometry-report",
        type=Path,
        default=DEFAULT_GEOMETRY_REPORT,
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    contract = load_json(args.contract)
    geometry = load_json(args.geometry_contract)
    geometry_report = load_json(args.geometry_report)
    result = validate_contract(contract, geometry, geometry_report)
    result["samples"] = [
        asdict(
            sample_entry_frame(
                contract,
                geometry,
                seed=args.seed,
                environment_index=index,
            )
        )
        for index in range(max(0, args.samples))
    ]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
