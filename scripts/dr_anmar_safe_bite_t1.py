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


def validate_contract(
    contract: dict[str, Any],
    geometry_contract: dict[str, Any],
) -> dict[str, Any]:
    """Fail on an impossible or reward-hackable T1 contract."""

    sampling = contract["sampling"]
    success = contract["success"]
    rewards = contract["rewards"]
    puncture = contract["puncture_transition"]
    fixture = contract["scene"]["fixture"]
    curriculum = contract["handover_snapshot_curriculum"]
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
        raise ValueError(
            "Position tolerance cannot be larger than the maximum stand-off"
        )

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
    if fixture["mode"] != "kinematic_outer_attachment_edge":
        raise ValueError("T1 must fixture only the authored outer attachment")
    if float(fixture["outer_edge_tolerance_m"]) <= 0.0:
        raise ValueError("T1 outer-edge fixture tolerance must be positive")
    lod = str(contract["scene"]["tissue_lod"])
    lod_geometry = geometry_contract["lods"][lod]
    expected_anchor_nodes = (
        2
        * (int(lod_geometry["cells_y"]) + 1)
        * len(lod_geometry["z_fractions"])
    )
    if (
        int(fixture["expected_training_lod_anchor_nodes"])
        != expected_anchor_nodes
    ):
        raise ValueError("T1 fixture count disagrees with the selected LOD")
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

    return {
        "schema": "dr.anmar.safe-bite-t1-contract-validation.v1",
        "contract_id": contract["id"],
        "asset_id": geometry_contract["id"],
        "valid": True,
        "safe_bite_range_m": [bite_low, bite_high],
        "stand_off_range_m": [stand_off_low, stand_off_high],
        "entry_angle_range_deg": [angle_low, angle_high],
        "fixture_anchor_nodes": expected_anchor_nodes,
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
    irregularity_wavelength = float(
        geometry["wound_irregularity_wavelength_m"]
    )
    thickness = float(geometry["thickness_m"])
    topography = float(geometry["surface_topography_amplitude_m"])
    topography_wavelength = float(
        geometry["surface_topography_wavelength_m"]
    )
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
    bite_low, bite_high = map(
        float, sampling["bite_distance_from_wound_m"]
    )
    stand_off_low, stand_off_high = map(float, sampling["stand_off_m"])
    angle_low, angle_high = map(
        float, sampling["entry_angle_from_surface_normal_deg"]
    )

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
    plane_normal = (0.0, 1.0, 0.0)
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
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    contract = load_json(args.contract)
    geometry = load_json(args.geometry_contract)
    result = validate_contract(contract, geometry)
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
