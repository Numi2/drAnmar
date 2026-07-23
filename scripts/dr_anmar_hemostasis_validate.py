#!/usr/bin/env python3
"""Validate DrAnmar vessel and vascular-clip geometry and mechanics contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from dr_anmar_hemostasis_author import (
    CLIP_ID,
    CLIP_NAME,
    CLIP_ROOT,
    DEFAULT_CLIP_OUTPUT,
    DEFAULT_REPORT,
    DEFAULT_TET_OUTPUT,
    DEFAULT_VESSEL_OUTPUT,
    PACKAGE_ID,
    PACKAGE_NAME,
    PACKAGE_VERSION,
    TET_ROOT,
    VESSEL_ID,
    VESSEL_NAME,
    VESSEL_ROOT,
)
from dr_anmar_hemostasis_model import (
    DEFAULT_HEMOSTASIS_PROFILE_PATH,
    build_clip_mesh,
    build_vessel_mesh,
    clip_closure_state,
    clip_retention_force_n,
    derive_hemostasis,
    load_hemostasis_profile,
    poiseuille_flow_ml_min,
    pressure_diameter_m,
    sample_hemostasis_episode_parameters,
    signed_tetra_volume,
    vessel_occlusion_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_REPORT = (
    REPOSITORY_ROOT / "assets/dr_anmar/hemostasis/DrAnmarHemostasis.validation.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    measured: Any,
    expected: Any,
) -> None:
    checks[name] = {
        "passed": bool(passed),
        "measured": measured,
        "expected": expected,
    }


def validate(
    profile: dict[str, Any],
    vessel_text: str,
    tet_text: str,
    clip_text: str,
    asset_report: dict[str, Any],
    vessel_path: Path,
    tet_path: Path,
    clip_path: Path,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    vessel = build_vessel_mesh(profile)
    clip = build_clip_mesh(profile)
    derived = derive_hemostasis(profile, vessel, clip)
    geometry = profile["geometry"]
    circumferential_cells = int(geometry["circumferential_cells"])
    axial_cells = int(geometry["axial_cells"])
    radial_cells = int(geometry["radial_cells"])
    tetrahedra_per_cell = int(geometry["tetrahedra_per_cell"])
    expected_vessel_points = (
        circumferential_cells * (axial_cells + 1) * (radial_cells + 1)
    )
    expected_vessel_tetrahedra = (
        circumferential_cells * axial_cells * radial_cells * tetrahedra_per_cell
    )
    expected_surface_triangles = (
        4 * circumferential_cells * axial_cells
        + 4 * circumferential_cells * radial_cells
    )
    expected_clip_rings = (
        2 * int(profile["clip"]["arm_segments"])
        + int(profile["clip"]["crown_segments"])
        + 1
    )
    expected_clip_points = (
        expected_clip_rings * int(profile["clip"]["section_segments"]) + 2
    )
    expected_clip_triangles = 2 * (expected_clip_rings - 1) * int(
        profile["clip"]["section_segments"]
    ) + 2 * int(profile["clip"]["section_segments"])

    check(
        checks,
        "profile_identity_and_boundary",
        profile["id"] == "dr-anmar-hemostasis-v1"
        and profile["name"] == PACKAGE_NAME
        and profile["version"] == PACKAGE_VERSION
        and profile["clinical_validation"] is False
        and "pending_solver_bench_and_clinical_validation" in profile["status"],
        {
            "id": profile["id"],
            "name": profile["name"],
            "version": profile["version"],
            "status": profile["status"],
            "clinical_validation": profile["clinical_validation"],
        },
        "canonical independent research-only asset identity",
    )
    check(
        checks,
        "deterministic_vessel_resolution",
        derived.vessel_point_count == expected_vessel_points
        and derived.vessel_tetrahedron_count == expected_vessel_tetrahedra
        and derived.vessel_surface_triangle_count == expected_surface_triangles,
        {
            "points": derived.vessel_point_count,
            "tetrahedra": derived.vessel_tetrahedron_count,
            "surface_triangles": derived.vessel_surface_triangle_count,
        },
        {
            "points": expected_vessel_points,
            "tetrahedra": expected_vessel_tetrahedra,
            "surface_triangles": expected_surface_triangles,
        },
    )
    tetra_volumes = [
        signed_tetra_volume(*(vessel.points[index] for index in tetrahedron))
        for tetrahedron in vessel.tetrahedra
    ]
    check(
        checks,
        "positive_right_handed_vessel_tetrahedra",
        all(volume > 0.0 for volume in tetra_volumes)
        and math.isclose(
            sum(tetra_volumes),
            vessel.wall_volume_m3,
            rel_tol=1.0e-12,
        ),
        {
            "minimum_m3": min(tetra_volumes),
            "total_m3": sum(tetra_volumes),
        },
        "all tetrahedra positive and total volume exact",
    )
    vessel_faces: Counter[tuple[int, int, int]] = Counter()
    for tetrahedron in vessel.tetrahedra:
        a, b, c, d = tetrahedron
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            vessel_faces[tuple(sorted(face))] += 1
    boundary_face_count = sum(1 for count in vessel_faces.values() if count == 1)
    check(
        checks,
        "manifold_hollow_vessel_wall",
        all(count in (1, 2) for count in vessel_faces.values())
        and boundary_face_count == len(vessel.surface_triangles)
        and vessel.connected_components == 1,
        {
            "boundary_faces": boundary_face_count,
            "surface_triangles": len(vessel.surface_triangles),
            "components": vessel.connected_components,
        },
        "one manifold hollow-wall component with exact boundary extraction",
    )
    grouped_tetrahedra = sorted(
        index for indices in vessel.tetrahedron_groups.values() for index in indices
    )
    check(
        checks,
        "complete_three_layer_wall_partition",
        grouped_tetrahedra == list(range(derived.vessel_tetrahedron_count))
        and set(vessel.tetrahedron_groups) == {"intima", "media", "adventitia"}
        and all(vessel.tetrahedron_groups.values()),
        {name: len(indices) for name, indices in vessel.tetrahedron_groups.items()},
        "every tetrahedron belongs to exactly one non-empty vessel-wall layer",
    )
    grouped_surface_faces = sorted(
        index for indices in vessel.surface_groups.values() for index in indices
    )
    check(
        checks,
        "complete_vessel_surface_partition",
        grouped_surface_faces == list(range(derived.vessel_surface_triangle_count))
        and set(vessel.surface_groups) == {"outer", "inner", "inlet", "outlet"}
        and all(vessel.surface_groups.values()),
        {name: len(indices) for name, indices in vessel.surface_groups.items()},
        "every surface triangle belongs to outer, inner, inlet, or outlet",
    )
    check(
        checks,
        "real_open_lumen_and_taper",
        vessel.lumen_volume_m3 > 0.0
        and derived.inlet_inner_diameter_m > derived.outlet_inner_diameter_m > 0.0
        and len(vessel.surface_groups["inner"]) > 0
        and len(vessel.surface_groups["inlet"]) > 0
        and len(vessel.surface_groups["outlet"]) > 0,
        {
            "lumen_volume_ml": derived.vessel_lumen_volume_ml,
            "inlet_inner_diameter_m": derived.inlet_inner_diameter_m,
            "outlet_inner_diameter_m": derived.outlet_inner_diameter_m,
            "inner_faces": len(vessel.surface_groups["inner"]),
        },
        "non-zero lumen with distinct inner wall and tapered open ends",
    )
    y_span = vessel.extent_max[1] - vessel.extent_min[1]
    z_span = vessel.extent_max[2] - vessel.extent_min[2]
    check(
        checks,
        "curved_elliptical_vessel_geometry",
        float(geometry["centerline_curvature_amplitude_m"]) > 0.0
        and float(geometry["ellipticity_fraction"]) > 0.0
        and not math.isclose(y_span, z_span, rel_tol=1.0e-3),
        {"y_span_m": y_span, "z_span_m": z_span},
        "curved centerline and intentionally non-circular cross-section",
    )
    check(
        checks,
        "opposed_end_attachment_capacity",
        derived.attachment_node_count >= 2 * circumferential_cells,
        derived.attachment_node_count,
        "dense node bands at both axial ends",
    )
    check(
        checks,
        "physical_vessel_scale_and_mass",
        0.003 <= float(geometry["outer_diameter_m"]) <= 0.009
        and 0.0001 <= derived.vessel_wall_mass_kg <= 0.005
        and 0.1 <= derived.vessel_lumen_volume_ml <= 5.0,
        {
            "outer_diameter_m": geometry["outer_diameter_m"],
            "wall_mass_kg": derived.vessel_wall_mass_kg,
            "lumen_volume_ml": derived.vessel_lumen_volume_ml,
        },
        "representative medium-vessel scale with positive wall mass and lumen",
    )

    vessel_tokens = [
        f'defaultPrim = "{VESSEL_ROOT}"',
        f'drAnmarAssetId = "{VESSEL_ID}"',
        f'drAnmarAssetName = "{VESSEL_NAME}"',
        'drAnmarWallTopology = "single_watertight_three_layer_hollow_tube"',
        "bool drAnmarHasOpenLumen = true",
        'def Mesh "Wall"',
        'def GeomSubset "OuterFaces"',
        'def GeomSubset "InnerFaces"',
        'def GeomSubset "InletFaces"',
        'def GeomSubset "OutletFaces"',
        "PhysicsCollisionAPI",
    ]
    missing_vessel_tokens = [
        token for token in vessel_tokens if token not in vessel_text
    ]
    check(
        checks,
        "stable_vessel_openusd_contract",
        not missing_vessel_tokens,
        {"missing": missing_vessel_tokens},
        vessel_tokens,
    )
    tet_tokens = [
        f'defaultPrim = "{TET_ROOT}"',
        'drAnmarRepresentation = "explicit_openusd_tetmesh_with_layer_ids_and_lumen_surface"',
        'def TetMesh "Simulation"',
        "int4[] tetVertexIndices",
        "int3[] surfaceFaceVertexIndices",
        'custom uniform token[] drAnmar:tetLayerNames = ["intima", "media", "adventitia"]',
        "custom int[] drAnmar:tetLayerIds",
        'def Mesh "Visual"',
    ]
    missing_tet_tokens = [token for token in tet_tokens if token not in tet_text]
    check(
        checks,
        "explicit_vessel_tetmesh_contract",
        not missing_tet_tokens,
        {"missing": missing_tet_tokens},
        tet_tokens,
    )

    check(
        checks,
        "deterministic_clip_resolution",
        derived.clip_point_count == expected_clip_points
        and derived.clip_triangle_count == expected_clip_triangles
        and derived.clip_centerline_segment_count == expected_clip_rings - 1,
        {
            "points": derived.clip_point_count,
            "triangles": derived.clip_triangle_count,
            "centerline_segments": derived.clip_centerline_segment_count,
        },
        {
            "points": expected_clip_points,
            "triangles": expected_clip_triangles,
            "centerline_segments": expected_clip_rings - 1,
        },
    )
    clip_edges: Counter[tuple[int, int]] = Counter()
    for triangle in clip.triangles:
        for start, end in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            clip_edges[tuple(sorted((start, end)))] += 1
    check(
        checks,
        "watertight_one_piece_clip",
        all(count == 2 for count in clip_edges.values()),
        {
            "edge_count": len(clip_edges),
            "nonmanifold_edges": sum(1 for count in clip_edges.values() if count != 2),
        },
        "closed one-piece clip render mesh with every edge shared twice",
    )
    check(
        checks,
        "physical_clip_scale_and_mass",
        0.005 <= clip.centerline_length_m <= 0.05
        and 1.0e-6 <= derived.clip_mass_kg <= 0.001,
        {
            "centerline_length_m": clip.centerline_length_m,
            "mass_kg": derived.clip_mass_kg,
            "extent_min": clip.extent_min,
            "extent_max": clip.extent_max,
        },
        "millimetre-scale one-piece clip with positive sub-gram mass",
    )
    clip_tokens = [
        f'defaultPrim = "{CLIP_ROOT}"',
        f'drAnmarAssetId = "{CLIP_ID}"',
        f'drAnmarAssetName = "{CLIP_NAME}"',
        'drAnmarConstruction = "one_piece_u_clip_with_swept_elliptical_section_and_inner_serrations"',
        'drAnmarRepresentation = "high_resolution_serrated_mesh_with_centerline_capsule_collision"',
        "PhysicsRigidBodyAPI",
        "PhysicsMassAPI",
        'def Mesh "Visual"',
        'def Xform "Collision"',
    ]
    missing_clip_tokens = [token for token in clip_tokens if token not in clip_text]
    capsule_count = clip_text.count('def Capsule "Segment')
    check(
        checks,
        "clip_openusd_and_collision_contract",
        not missing_clip_tokens
        and capsule_count == derived.clip_centerline_segment_count,
        {
            "missing": missing_clip_tokens,
            "collision_capsules": capsule_count,
        },
        {
            "tokens": clip_tokens,
            "collision_capsules": derived.clip_centerline_segment_count,
        },
    )
    forbidden_tokens = [
        "prepend references",
        "SuturePad",
        "ORBIT",
        "NVIDIA",
        "Lightwheel",
    ]
    present_forbidden = [
        token
        for token in forbidden_tokens
        if token in vessel_text or token in tet_text or token in clip_text
    ]
    check(
        checks,
        "independent_dr_anmar_asset_package",
        not present_forbidden,
        {"forbidden": present_forbidden},
        "no external geometry reference or inherited product identity",
    )

    report_hashes_match = (
        asset_report["vessel_asset_sha256"] == sha256(vessel_path)
        and asset_report["vessel_tetmesh_asset_sha256"] == sha256(tet_path)
        and asset_report["clip_asset_sha256"] == sha256(clip_path)
    )
    check(
        checks,
        "asset_report_matches_generated_package",
        asset_report["asset_id"] == PACKAGE_ID
        and asset_report["vessel_point_count"] == derived.vessel_point_count
        and asset_report["vessel_tetrahedron_count"] == derived.vessel_tetrahedron_count
        and asset_report["clip_point_count"] == derived.clip_point_count
        and asset_report["clip_triangle_count"] == derived.clip_triangle_count
        and report_hashes_match
        and asset_report["clinical_validation"] is False,
        {
            "asset_id": asset_report["asset_id"],
            "vessel_points": asset_report["vessel_point_count"],
            "vessel_tetrahedra": asset_report["vessel_tetrahedron_count"],
            "clip_points": asset_report["clip_point_count"],
            "clip_triangles": asset_report["clip_triangle_count"],
            "hashes_match": report_hashes_match,
            "clinical_validation": asset_report["clinical_validation"],
        },
        "report identity, counts, hashes and boundary match generated assets",
    )
    sample_a = sample_hemostasis_episode_parameters(profile, 7241)
    replay_a = sample_hemostasis_episode_parameters(profile, 7241)
    sample_b = sample_hemostasis_episode_parameters(profile, 7242)
    check(
        checks,
        "deterministic_sim_to_real_sampling",
        sample_a == replay_a and sample_a != sample_b,
        {
            "seed_7241": sample_a.payload(),
            "seed_7241_replay": replay_a.payload(),
            "seed_7242": sample_b.payload(),
        },
        "same seed exactly replays and a different seed changes the domain",
    )
    pressures = [20.0, 80.0, 120.0, 180.0]
    diameters = [
        pressure_diameter_m(profile, pressure_mmhg=pressure) for pressure in pressures
    ]
    low_slope = (diameters[2] - diameters[1]) / (pressures[2] - pressures[1])
    high_slope = (diameters[3] - diameters[2]) / (pressures[3] - pressures[2])
    check(
        checks,
        "nonlinear_pressure_diameter_response",
        diameters == sorted(diameters) and low_slope > high_slope > 0.0,
        {
            "pressures_mmhg": pressures,
            "diameters_m": diameters,
            "low_slope_m_per_mmhg": low_slope,
            "high_slope_m_per_mmhg": high_slope,
        },
        "diameter rises with pressure while collagen recruitment reduces slope",
    )
    flow_small = poiseuille_flow_ml_min(
        pressure_drop_mmhg=80.0,
        lumen_radius_m=0.001,
        length_m=0.05,
        dynamic_viscosity_pa_s=0.0035,
    )
    flow_large = poiseuille_flow_ml_min(
        pressure_drop_mmhg=80.0,
        lumen_radius_m=0.002,
        length_m=0.05,
        dynamic_viscosity_pa_s=0.0035,
    )
    check(
        checks,
        "reduced_order_lumen_flow_scaling",
        math.isclose(flow_large / flow_small, 16.0, rel_tol=1.0e-12),
        {
            "one_mm_radius_ml_min": flow_small,
            "two_mm_radius_ml_min": flow_large,
            "ratio": flow_large / flow_small,
        },
        "laminar-flow proxy preserves fourth-power radius sensitivity",
    )
    forces = [0.0, 1.0, 2.5, 5.0]
    closures = [clip_closure_state(profile, applied_force_n=force) for force in forces]
    check(
        checks,
        "force_gap_and_plastic_set_response",
        [item.loaded_gap_m for item in closures]
        == sorted(
            [item.loaded_gap_m for item in closures],
            reverse=True,
        )
        and closures[0].plastic_fraction == 0.0
        and math.isclose(
            closures[-1].plastic_fraction,
            1.0,
            rel_tol=1.0e-12,
        )
        and closures[-1].residual_gap_m < closures[0].residual_gap_m,
        {
            str(force): closure.__dict__
            for force, closure in zip(forces, closures, strict=True)
        },
        "closing force reduces loaded gap and creates a permanent set only after yield",
    )
    low_yield_closure = clip_closure_state(
        profile,
        applied_force_n=3.0,
        yield_strength_pa=180000000.0,
    )
    high_yield_closure = clip_closure_state(
        profile,
        applied_force_n=3.0,
        yield_strength_pa=480000000.0,
    )
    check(
        checks,
        "yield_strength_dependent_plastic_set",
        low_yield_closure.plastic_fraction > high_yield_closure.plastic_fraction
        and low_yield_closure.residual_gap_m < high_yield_closure.residual_gap_m,
        {
            "low_yield": low_yield_closure.__dict__,
            "high_yield": high_yield_closure.__dict__,
        },
        "higher clip yield strength reduces permanent closure at equal force",
    )
    open_state = vessel_occlusion_state(
        profile,
        clip_gap_m=closures[0].residual_gap_m,
        pressure_mmhg=600.0,
        applied_force_n=0.0,
    )
    closed_state = vessel_occlusion_state(
        profile,
        clip_gap_m=closures[-1].residual_gap_m,
        pressure_mmhg=600.0,
        applied_force_n=forces[-1],
    )
    overload_state = vessel_occlusion_state(
        profile,
        clip_gap_m=closures[-1].residual_gap_m,
        pressure_mmhg=600.0,
        cycles=5,
        applied_force_n=10.0,
    )
    check(
        checks,
        "occlusion_leak_and_overload_states",
        not open_state.qualified_geometry
        and closed_state.qualified_geometry
        and closed_state.occlusion_fraction
        >= float(profile["occlusion"]["target_lumen_area_reduction_fraction"])
        and closed_state.leak_rate_ml_min
        <= float(profile["occlusion"]["maximum_qualified_leak_rate_ml_min"])
        and overload_state.crush_damage > closed_state.crush_damage,
        {
            "open": open_state.__dict__,
            "closed": closed_state.__dict__,
            "overload": overload_state.__dict__,
        },
        "closed geometry seals the reduced-order lumen while overload raises damage",
    )
    weak_retention = clip_retention_force_n(
        profile,
        closure_fraction=0.5,
        vessel_diameter_m=0.006,
        friction_coefficient=0.2,
        pull_angle_degrees=0.0,
    )
    strong_retention = clip_retention_force_n(
        profile,
        closure_fraction=1.0,
        vessel_diameter_m=0.006,
        friction_coefficient=0.4,
        pull_angle_degrees=45.0,
    )
    check(
        checks,
        "closure_friction_and_angle_dependent_retention",
        strong_retention > weak_retention > 0.0,
        {
            "weak_retention_n": weak_retention,
            "strong_retention_n": strong_retention,
        },
        "retention rises with closure, friction and oblique pull contribution",
    )
    sim_gaps = profile["sim_to_real"]["gaps"]
    sampled_parameters = profile["sim_to_real"]["implemented_parameter_sampling"]
    check(
        checks,
        "sim_to_real_gap_register",
        len(sim_gaps) >= 10
        and len(sampled_parameters) >= 12
        and all(
            item.get("id")
            and item.get("risk")
            and item.get("mitigation")
            and item.get("status")
            for item in sim_gaps
        ),
        {
            "gap_count": len(sim_gaps),
            "sampled_parameters": len(sampled_parameters),
            "complete": all(
                item.get("id")
                and item.get("risk")
                and item.get("mitigation")
                and item.get("status")
                for item in sim_gaps
            ),
        },
        "at least ten complete gaps and twelve sampled parameters",
    )
    requirements = profile["qualification"]["requirements"]
    blocked_ids = {
        item["id"] for item in requirements if str(item["status"]).startswith("blocked")
    }
    check(
        checks,
        "fail_closed_qualification",
        profile["qualification"]["policy"] == "fail_closed"
        and {
            "clip_plastic_closure",
            "occlusion_and_leakage",
            "retention_and_slip",
            "damage_and_failure",
            "clinical_use",
        }.issubset(blocked_ids),
        {
            "policy": profile["qualification"]["policy"],
            "blocked_ids": sorted(blocked_ids),
            "requirement_count": len(requirements),
        },
        "plastic closure, pressure seal, retention, failure and clinical use remain blocked",
    )
    evidence = profile["evidence"]
    check(
        checks,
        "traceable_research_provenance",
        len(evidence) >= 6
        and all(item.get("url") and item.get("used_for") for item in evidence),
        len(evidence),
        "at least six traceable platform or primary research sources",
    )

    passed = all(item["passed"] for item in checks.values())
    return {
        "schema": "dr.anmar.hemostasis-validation.v1",
        "profile_id": profile["id"],
        "asset_id": PACKAGE_ID,
        "passed": passed,
        "checks": checks,
        "derived": {
            "vessel_points": derived.vessel_point_count,
            "vessel_tetrahedra": derived.vessel_tetrahedron_count,
            "vessel_surface_triangles": (derived.vessel_surface_triangle_count),
            "vessel_wall_mass_kg": derived.vessel_wall_mass_kg,
            "vessel_lumen_volume_ml": derived.vessel_lumen_volume_ml,
            "clip_points": derived.clip_point_count,
            "clip_triangles": derived.clip_triangle_count,
            "clip_collision_segments": (derived.clip_centerline_segment_count),
            "clip_mass_kg": derived.clip_mass_kg,
            "sim_to_real_gap_count": len(sim_gaps),
        },
        "stable_capability_boundary": asset_report["stable_capabilities"],
        "blocked_capability_boundary": asset_report["gated_capabilities"],
        "clinical_validation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_HEMOSTASIS_PROFILE_PATH,
    )
    parser.add_argument(
        "--vessel",
        type=Path,
        default=DEFAULT_VESSEL_OUTPUT,
    )
    parser.add_argument(
        "--tet",
        type=Path,
        default=DEFAULT_TET_OUTPUT,
    )
    parser.add_argument(
        "--clip",
        type=Path,
        default=DEFAULT_CLIP_OUTPUT,
    )
    parser.add_argument(
        "--asset-report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VALIDATION_REPORT,
    )
    args = parser.parse_args()

    profile = load_hemostasis_profile(args.profile)
    report = validate(
        profile,
        args.vessel.read_text(encoding="utf-8"),
        args.tet.read_text(encoding="utf-8"),
        args.clip.read_text(encoding="utf-8"),
        json.loads(args.asset_report.read_text(encoding="utf-8")),
        args.vessel,
        args.tet,
        args.clip,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
