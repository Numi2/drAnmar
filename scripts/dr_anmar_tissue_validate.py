#!/usr/bin/env python3
"""Validate DrAnmar Suturable Tissue geometry, mechanics, provenance, and integration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from dr_anmar_native_rooms import resolve_native_room
from dr_anmar_procedures import PROCEDURE_ROOMS, PROCEDURE_SUITES
from dr_anmar_tissue_author import (
    ASSET_ID,
    ASSET_NAME,
    ASSET_VERSION,
    DEFAULT_OUTPUT,
    DEFAULT_REPORT,
    DEFAULT_TET_OUTPUT,
    ROOT_PRIM,
    TET_ROOT_PRIM,
)
from dr_anmar_tissue_model import (
    DEFAULT_TISSUE_PROFILE_PATH,
    build_tissue_mesh,
    cyclic_tear_damage_increment,
    derive_tissue,
    load_tissue_profile,
    needle_tissue_force,
    sample_tissue_episode_parameters,
    signed_tetra_volume,
    suture_holding_capacity_n,
    wound_gap_under_tension_m,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSTATION = REPOSITORY_ROOT / "scripts/dr_anmar_workstation.py"
DEFAULT_PROCEDURES = REPOSITORY_ROOT / "scripts/dr_anmar_procedures.py"
DEFAULT_AUDIT = REPOSITORY_ROOT / "docs/DR_ANMAR_SURGICAL_ASSET_GAP_AUDIT.md"
DEFAULT_VALIDATION_REPORT = (
    REPOSITORY_ROOT
    / "physics_next/benchmarks/dr-anmar-suturable-tissue-validation.json"
)
ROOM_ID = "dr-anmar-suturable-tissue"


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
    surface_text: str,
    tet_text: str,
    asset_report: dict[str, Any],
    workstation_text: str,
    procedure_text: str,
    audit_text: str,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    mesh = build_tissue_mesh(profile)
    derived = derive_tissue(profile, mesh)
    geometry = profile["geometry"]
    expected_points = (
        2
        * (int(geometry["cells_per_flap_x"]) + 1)
        * (int(geometry["cells_y"]) + 1)
        * (int(geometry["cells_z"]) + 1)
    )
    expected_tets = (
        2
        * int(geometry["cells_per_flap_x"])
        * int(geometry["cells_y"])
        * int(geometry["cells_z"])
        * int(geometry["tetrahedra_per_cell"])
    )
    check(
        checks,
        "profile_identity_and_boundary",
        profile["id"] == "dr-anmar-suturable-tissue-v1"
        and profile["name"] == ASSET_NAME
        and profile["clinical_validation"] is False
        and "pending_bench_and_clinical_validation" in profile["status"],
        {
            "id": profile["id"],
            "name": profile["name"],
            "clinical_validation": profile["clinical_validation"],
            "status": profile["status"],
        },
        "canonical research-only identity",
    )
    check(
        checks,
        "deterministic_mesh_resolution",
        derived.point_count == expected_points
        and derived.tetrahedron_count == expected_tets
        and derived.surface_triangle_count >= 2000,
        {
            "points": derived.point_count,
            "tetrahedra": derived.tetrahedron_count,
            "surface_triangles": derived.surface_triangle_count,
        },
        {
            "points": expected_points,
            "tetrahedra": expected_tets,
            "surface_triangles": "at least 2000",
        },
    )
    tetra_volumes = [
        signed_tetra_volume(*(mesh.points[index] for index in tet))
        for tet in mesh.tetrahedra
    ]
    check(
        checks,
        "positive_right_handed_tetrahedra",
        bool(tetra_volumes)
        and min(tetra_volumes) > 1.0e-16
        and math.isclose(sum(tetra_volumes), mesh.volume_m3),
        {
            "minimum_m3": min(tetra_volumes),
            "total_m3": sum(tetra_volumes),
        },
        "all positive and total equals reported volume",
    )
    ideal_volume = (
        (float(geometry["overall_width_m"]) - float(geometry["rest_wound_gap_m"]))
        * float(geometry["depth_m"])
        * float(geometry["thickness_m"])
    )
    check(
        checks,
        "physical_scale_volume_and_mass",
        0.95 * ideal_volume <= mesh.volume_m3 <= ideal_volume
        and 0.005 <= derived.mass_kg <= 0.05,
        {
            "volume_m3": mesh.volume_m3,
            "ideal_unbeveled_volume_m3": ideal_volume,
            "mass_kg": derived.mass_kg,
        },
        "beveled volume within five percent of envelope and mass 5-50 g",
    )
    grouped_faces = sorted(
        index for indices in mesh.surface_groups.values() for index in indices
    )
    check(
        checks,
        "complete_layered_surface_partition",
        grouped_faces == list(range(derived.surface_triangle_count))
        and all(mesh.surface_groups.values()),
        {name: len(indices) for name, indices in mesh.surface_groups.items()},
        "every surface face belongs to exactly one non-empty region",
    )
    grouped_tetrahedra = sorted(
        index for indices in mesh.tetrahedron_groups.values() for index in indices
    )
    check(
        checks,
        "complete_layered_tetrahedral_partition",
        grouped_tetrahedra == list(range(derived.tetrahedron_count))
        and set(mesh.tetrahedron_groups) == {"surface", "bulk", "fascia"}
        and all(mesh.tetrahedron_groups.values()),
        {name: len(indices) for name, indices in mesh.tetrahedron_groups.items()},
        (
            "every tetrahedron belongs to exactly one non-empty "
            "surface, bulk, or fascia layer"
        ),
    )
    check(
        checks,
        "real_open_incision_geometry",
        mesh.connected_components == 2
        and math.isclose(
            derived.rest_wound_gap_bottom_m,
            float(geometry["rest_wound_gap_m"]),
        )
        and derived.rest_wound_gap_top_m > derived.rest_wound_gap_bottom_m,
        {
            "components": mesh.connected_components,
            "bottom_gap_m": derived.rest_wound_gap_bottom_m,
            "top_gap_m": derived.rest_wound_gap_top_m,
        },
        "two disconnected flaps with a beveled open gap",
    )
    check(
        checks,
        "opposed_outer_attachment_capacity",
        derived.outer_attachment_node_count >= 100
        and profile["attachments"]["sides"] == ["minimum", "maximum"],
        {
            "attachment_nodes": derived.outer_attachment_node_count,
            "sides": profile["attachments"]["sides"],
        },
        "both outer margins expose a dense attachment band",
    )
    surface_tokens = [
        f'defaultPrim = "{ROOT_PRIM}"',
        f'drAnmarAssetId = "{ASSET_ID}"',
        f'drAnmarAssetName = "{ASSET_NAME}"',
        f'drAnmarAssetVersion = "{ASSET_VERSION}"',
        'drAnmarWoundTopology = "two_disconnected_watertight_flaps_with_open_incision"',
        'drAnmarRepresentation = "watertight_surface_for_native_physx_tetrahedral_cooking"',
        'def Mesh "Surface"',
        'def GeomSubset "SurfaceFaces"',
        'def GeomSubset "BulkFaces"',
        'def GeomSubset "FasciaFaces"',
        'def GeomSubset "WoundFaces"',
        "PhysicsCollisionAPI",
    ]
    missing_surface = [token for token in surface_tokens if token not in surface_text]
    check(
        checks,
        "stable_surface_asset_contract",
        not missing_surface,
        {"missing": missing_surface},
        surface_tokens,
    )
    tet_tokens = [
        f'defaultPrim = "{TET_ROOT_PRIM}"',
        'drAnmarRepresentation = "explicit_openusd_tetmesh_with_surface_visualization"',
        'def TetMesh "Simulation"',
        "int4[] tetVertexIndices",
        "int3[] surfaceFaceVertexIndices",
        'custom uniform token[] drAnmar:tetLayerNames = ["fascia", "bulk", "surface"]',
        "custom int[] drAnmar:tetLayerIds",
        'def Mesh "Visual"',
    ]
    missing_tet = [token for token in tet_tokens if token not in tet_text]
    check(
        checks,
        "backend_neutral_tetmesh_contract",
        not missing_tet,
        {"missing": missing_tet},
        tet_tokens,
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
        if token in surface_text or token in tet_text
    ]
    check(
        checks,
        "independent_dr_anmar_asset",
        not present_forbidden,
        {"forbidden": present_forbidden},
        "no external geometry reference or inherited product identity",
    )
    check(
        checks,
        "asset_report_matches_model",
        asset_report["point_count"] == derived.point_count
        and asset_report["tetrahedron_count"] == derived.tetrahedron_count
        and asset_report["surface_triangle_count"] == derived.surface_triangle_count
        and asset_report["connected_components"] == 2
        and asset_report["tetrahedron_groups"]
        == {name: len(indices) for name, indices in mesh.tetrahedron_groups.items()}
        and asset_report["clinical_validation"] is False,
        {
            "points": asset_report["point_count"],
            "tetrahedra": asset_report["tetrahedron_count"],
            "surface_triangles": asset_report["surface_triangle_count"],
            "components": asset_report["connected_components"],
            "tetrahedron_groups": asset_report["tetrahedron_groups"],
            "clinical_validation": asset_report["clinical_validation"],
        },
        "generated report exactly matches derived model",
    )
    sample_a = sample_tissue_episode_parameters(profile, 4107)
    replay_a = sample_tissue_episode_parameters(profile, 4107)
    sample_b = sample_tissue_episode_parameters(profile, 4108)
    check(
        checks,
        "deterministic_sim_to_real_sampling",
        sample_a == replay_a and sample_a != sample_b,
        {
            "seed_4107": sample_a.payload(),
            "seed_4107_replay": replay_a.payload(),
            "seed_4108": sample_b.payload(),
        },
        "same seed exactly replays and a different seed changes the domain",
    )
    preforces = [
        needle_tissue_force(profile, indentation_m=depth).total_n
        for depth in (0.0, 0.0005, 0.001, 0.0015)
    ]
    penetrated_short = needle_tissue_force(
        profile,
        indentation_m=0.0015,
        punctured=True,
        embedded_arc_length_m=0.003,
        swept_area_m2=0.0,
    )
    penetrated_long = needle_tissue_force(
        profile,
        indentation_m=0.0015,
        punctured=True,
        embedded_arc_length_m=0.012,
        swept_area_m2=1.0e-5,
    )
    check(
        checks,
        "needle_force_phase_model",
        preforces == sorted(preforces)
        and math.isclose(
            preforces[-1],
            float(profile["puncture"]["puncture_force_n_seed"]),
        )
        and penetrated_short.cutting_n > 0.0
        and penetrated_long.shaft_friction_n > penetrated_short.shaft_friction_n
        and penetrated_long.compression_n > 0.0,
        {
            "prepuncture_n": preforces,
            "short_penetration": penetrated_short.__dict__,
            "long_penetration": penetrated_long.__dict__,
        },
        "compression rises to puncture, then cutting, sweep, and shaft friction act",
    )
    capacities = [
        suture_holding_capacity_n(
            profile,
            bite_margin_m=margin,
            engaged_thickness_m=0.006,
        )
        for margin in (0.002, 0.005, 0.008)
    ]
    damaged_capacity = suture_holding_capacity_n(
        profile,
        bite_margin_m=0.005,
        engaged_thickness_m=0.006,
        local_damage=0.5,
    )
    check(
        checks,
        "bite_dependent_suture_holding",
        capacities == sorted(capacities)
        and math.isclose(
            capacities[1],
            float(profile["suture_holding"]["reference_pullout_force_n_seed"]),
        )
        and damaged_capacity < capacities[1],
        {
            "capacities_n": capacities,
            "half_damaged_capacity_n": damaged_capacity,
        },
        "larger bites hold more and local damage weakens only the affected site",
    )
    low_damage = cyclic_tear_damage_increment(
        profile,
        tension_n=2.0,
        holding_capacity_n=8.0,
        duration_s=1.0,
    )
    high_damage = cyclic_tear_damage_increment(
        profile,
        tension_n=7.0,
        holding_capacity_n=8.0,
        duration_s=1.0,
    )
    check(
        checks,
        "cyclic_cheese_wire_damage",
        low_damage == 0.0 and high_damage > 0.0,
        {"low_load": low_damage, "high_load": high_damage},
        "damage begins only above the configured holding utilization",
    )
    gaps = [
        wound_gap_under_tension_m(
            profile,
            total_stitch_tension_n=tension,
        )
        for tension in (0.0, 1.0, 4.0, 8.0)
    ]
    check(
        checks,
        "monotonic_wound_approximation",
        gaps == sorted(gaps, reverse=True)
        and gaps[-1] >= float(profile["wound_closure"]["minimum_residual_gap_m"]),
        gaps,
        "gap decreases with tension without becoming negative",
    )
    sim_gaps = profile["sim_to_real"]["gaps"]
    complete_gaps = all(
        {"id", "risk", "mitigation", "status"}.issubset(item) for item in sim_gaps
    )
    check(
        checks,
        "sim_to_real_gap_register",
        len(sim_gaps) >= 10
        and complete_gaps
        and len(profile["sim_to_real"]["implemented_parameter_sampling"]) >= 10,
        {
            "gap_count": len(sim_gaps),
            "complete": complete_gaps,
            "sampled_parameters": len(
                profile["sim_to_real"]["implemented_parameter_sampling"]
            ),
        },
        "at least ten complete gaps and ten sampled parameters",
    )
    requirements = profile["qualification"]["requirements"]
    clinical = [item for item in requirements if item["id"] == "clinical_use"]
    puncture = [item for item in requirements if item["id"] == "puncture"]
    check(
        checks,
        "fail_closed_qualification",
        profile["qualification"]["policy"] == "fail_closed"
        and len(requirements) >= 7
        and clinical
        == [
            {
                "id": "clinical_use",
                "evidence": "independent clinical validation for the intended use",
                "status": "blocked",
            }
        ]
        and len(puncture) == 1
        and puncture[0]["status"] == "blocked_pending_topology_backend",
        {
            "policy": profile["qualification"]["policy"],
            "requirement_count": len(requirements),
            "clinical": clinical,
            "puncture": puncture,
        },
        "clinical use and topology-changing puncture remain blocked",
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
    native_room = resolve_native_room(ROOM_ID)
    check(
        checks,
        "repository_native_room_binding",
        bool(native_room)
        and native_room["available"]
        and native_room["backend"] == "physx_fem"
        and native_room["repository_asset"] is True
        and Path(native_room["asset_path"]) == DEFAULT_OUTPUT
        and Path(native_room["alternate_tetmesh_asset_path"]) == DEFAULT_TET_OUTPUT
        and native_room["attachment"]["sides"] == ["minimum", "maximum"],
        native_room,
        "available repository-owned PhysX binding plus explicit TetMesh",
    )
    room = next(
        (room for room in PROCEDURE_ROOMS if room["id"] == ROOM_ID),
        None,
    )
    check(
        checks,
        "dedicated_bimanual_research_room",
        bool(room)
        and room["bimanual"] is True
        and room["guide_kind"] == "dr_anmar_wound_closure"
        and "fail-closed" in room["interaction"],
        room,
        "one transparent bimanual wound-closure research room",
    )
    suturing_suite = next(
        (suite for suite in PROCEDURE_SUITES if suite["id"] == "suturing-suite"),
        None,
    )
    check(
        checks,
        "suturing_suite_integration",
        bool(suturing_suite) and ROOM_ID in suturing_suite["rooms"],
        suturing_suite,
        "the tissue room is part of the needle-and-thread progression",
    )
    check(
        checks,
        "opposed_attachment_runtime_support",
        'attachment.get("sides")' in workstation_text
        and 'side == "minimum"' in workstation_text
        and 'side == "maximum"' in workstation_text,
        {
            "supports_sides": 'attachment.get("sides")' in workstation_text,
            "minimum": 'side == "minimum"' in workstation_text,
            "maximum": 'side == "maximum"' in workstation_text,
        },
        "workstation pins both declared outer margins",
    )
    check(
        checks,
        "current_thread_preserved",
        "without replacing any task-owned thread" in procedure_text
        and "current strand—remain untouched" in workstation_text,
        {
            "procedure_boundary": "without replacing any task-owned thread"
            in procedure_text,
            "workstation_boundary": "current strand—remain untouched"
            in workstation_text,
        },
        "the new room and asset do not replace the current thread",
    )
    normalized_audit = " ".join(audit_text.split())
    check(
        checks,
        "research_decision_recorded",
        "DrAnmar Suturable Tissue" in normalized_audit
        and "one static triangle collision mesh" in normalized_audit
        and "highest-impact missing foundational asset" in normalized_audit,
        "decision and inspected platform gap present",
        "auditable asset-selection rationale",
    )
    passed = all(item["passed"] for item in checks.values())
    return {
        "schema": "dr.anmar.suturable-tissue-validation.v1",
        "profile_id": profile["id"],
        "asset_id": ASSET_ID,
        "passed": passed,
        "checks": checks,
        "derived": {
            "points": derived.point_count,
            "tetrahedra": derived.tetrahedron_count,
            "surface_triangles": derived.surface_triangle_count,
            "connected_components": mesh.connected_components,
            "volume_m3": mesh.volume_m3,
            "mass_kg": derived.mass_kg,
            "rest_wound_gap_bottom_m": derived.rest_wound_gap_bottom_m,
            "rest_wound_gap_top_m": derived.rest_wound_gap_top_m,
            "attachment_nodes": derived.outer_attachment_node_count,
            "sim_to_real_gap_count": len(sim_gaps),
        },
        "stable_capability_boundary": [
            "intact_deformation",
            "two_way_contact",
            "grasping",
            "retraction",
            "wound_edge_approximation",
        ],
        "blocked_capability_boundary": [
            "arbitrary_puncture",
            "persistent_tract",
            "thread_passage",
            "cutting",
            "clinical_use",
        ],
        "clinical_validation": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_TISSUE_PROFILE_PATH,
    )
    parser.add_argument("--surface", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tetmesh", type=Path, default=DEFAULT_TET_OUTPUT)
    parser.add_argument("--asset-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workstation", type=Path, default=DEFAULT_WORKSTATION)
    parser.add_argument("--procedures", type=Path, default=DEFAULT_PROCEDURES)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VALIDATION_REPORT,
    )
    args = parser.parse_args()
    report = validate(
        load_tissue_profile(args.profile),
        args.surface.read_text(encoding="utf-8"),
        args.tetmesh.read_text(encoding="utf-8"),
        json.loads(args.asset_report.read_text(encoding="utf-8")),
        args.workstation.read_text(encoding="utf-8"),
        args.procedures.read_text(encoding="utf-8"),
        args.audit.read_text(encoding="utf-8"),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
