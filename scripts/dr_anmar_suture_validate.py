#!/usr/bin/env python3
"""Deterministically validate the Dr.Anmar 4-0 suture contract and USD."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

from dr_anmar_procedures import PROCEDURE_ROOMS
from dr_anmar_needle_model import (
    DEFAULT_NEEDLE_PROFILE_PATH,
    build_needle_mesh,
    derive_needle,
    load_needle_profile,
    sample_episode_parameters,
)
from dr_anmar_suture_integration import (
    DR_ANMAR_NEEDLE_ASSET_ID,
    DR_ANMAR_NEEDLE_ASSET_PATH,
    DR_ANMAR_NEEDLE_ASSET_VERSION,
    DR_ANMAR_NEEDLE_NAME,
    DR_ANMAR_NEEDLE_ROOT_PRIM,
    configure_dr_anmar_needle,
    local_room_ids,
)
from dr_anmar_suture_model import (
    DEFAULT_PROFILE_PATH,
    crush_strength_fraction,
    derive,
    effective_failure_load,
    load_profile,
    monotonic_tension_force,
    sample_suture_runtime_profile,
    self_friction_coefficient,
    stress_retention,
)
from dr_anmar_suture_runtime import SutureRuntime


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = (
    REPOSITORY_ROOT / "assets/dr_anmar/suture/DrAnmarSuture4_0.usda"
)
DEFAULT_NEEDLE = DR_ANMAR_NEEDLE_ASSET_PATH
DEFAULT_WORKSTATION = REPOSITORY_ROOT / "scripts/dr_anmar_workstation.py"


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
    needle_profile: dict[str, Any],
    asset_text: str,
    needle_text: str,
    workstation_text: str,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    derived = derive(profile)
    derived_needle = derive_needle(needle_profile)
    needle_mesh = build_needle_mesh(needle_profile)
    geometry = profile["geometry"]
    material = profile["material"]
    tension = profile["tension"]
    diameter_range = [float(value) for value in geometry["diameter_range_m"]]
    check(
        checks,
        "true_4_0_diameter",
        diameter_range[0] <= derived.diameter_m <= diameter_range[1],
        derived.diameter_m,
        diameter_range,
    )
    density_range = [float(value) for value in material["density_range_kg_m3"]]
    check(
        checks,
        "polymer_density",
        density_range[0] <= float(material["density_kg_m3"]) <= density_range[1],
        float(material["density_kg_m3"]),
        density_range,
    )
    failure_range = [float(value) for value in tension["straight_failure_range_n"]]
    check(
        checks,
        "straight_failure_load",
        failure_range[0] <= derived.straight_failure_load_n <= failure_range[1],
        derived.straight_failure_load_n,
        failure_range,
    )
    failure_strain_range = [float(value) for value in tension["failure_strain_range"]]
    check(
        checks,
        "elongation_at_break",
        failure_strain_range[0]
        <= float(tension["failure_strain"])
        <= failure_strain_range[1],
        float(tension["failure_strain"]),
        failure_strain_range,
    )
    knot_range = [
        float(value) for value in profile["knot"]["strength_efficiency_range"]
    ]
    knot_efficiency = derived.knot_failure_load_n / derived.straight_failure_load_n
    check(
        checks,
        "knot_strength_reduction",
        knot_range[0] <= knot_efficiency <= knot_range[1],
        knot_efficiency,
        knot_range,
    )
    force_yield, failed_yield = monotonic_tension_force(
        profile, float(tension["yield_strain"])
    )
    force_break, failed_break = monotonic_tension_force(
        profile, float(tension["failure_strain"])
    )
    check(
        checks,
        "nonlinear_tension_curve",
        0.0 < force_yield < force_break <= derived.straight_failure_load_n
        and not failed_yield
        and failed_break,
        {"yield_force_n": force_yield, "break_force_n": force_break},
        "positive yield force below a 20-25 N failed endpoint",
    )
    retained_2h = stress_retention(profile, 7200.0)
    retained_long = stress_retention(profile, 1e9)
    check(
        checks,
        "wet_stress_relaxation",
        retained_long <= retained_2h < 1.0
        and math.isclose(
            retained_long,
            float(profile["viscoelasticity"]["retained_stress_asymptote"]),
            abs_tol=1e-6,
        ),
        {"two_hours": retained_2h, "long_time": retained_long},
        "largest early loss with configured asymptote",
    )
    friction_low = self_friction_coefficient(profile, 0.01)
    friction_high = self_friction_coefficient(profile, 2.0)
    check(
        checks,
        "load_dependent_self_friction",
        0.0 < friction_high < friction_low < 1.0,
        {"mu_at_0_01_n": friction_low, "mu_at_2_n": friction_high},
        "positive coefficient decreasing with load while friction force rises",
    )
    check(
        checks,
        "instrument_crush_damage",
        crush_strength_fraction(profile, 0) == 1.0
        and math.isclose(crush_strength_fraction(profile, 1), 0.661)
        and math.isclose(crush_strength_fraction(profile, 5), 0.383),
        {
            "zero_grasps": crush_strength_fraction(profile, 0),
            "one_grasp": crush_strength_fraction(profile, 1),
            "five_grasps": crush_strength_fraction(profile, 5),
        },
        {"zero_grasps": 1.0, "one_grasp": 0.661, "five_grasps": 0.383},
    )
    check(
        checks,
        "combined_knot_and_crush_failure",
        effective_failure_load(profile, knotted=True, grasp_count=1)
        < derived.knot_failure_load_n,
        effective_failure_load(profile, knotted=True, grasp_count=1),
        f"less than {derived.knot_failure_load_n}",
    )
    segment_defs = len(re.findall(r'def Capsule "S\d{4}"', asset_text))
    joint_defs = len(re.findall(r'def PhysicsJoint "J\d{4}"', asset_text))
    check(
        checks,
        "physical_segment_resolution",
        segment_defs == derived.segment_count,
        segment_defs,
        derived.segment_count,
    )
    check(
        checks,
        "breakable_joint_resolution",
        joint_defs == derived.segment_count
        and asset_text.count("physics:breakForce") == derived.segment_count,
        {
            "joint_defs": joint_defs,
            "break_force_attributes": asset_text.count("physics:breakForce"),
        },
        derived.segment_count,
    )
    required_asset_tokens = [
        "PhysicsRigidBodyAPI",
        "PhysicsCollisionAPI",
        "PhysicsDriveAPI:transX",
        "PhysicsDriveAPI:rotY",
        "PhysicsDriveAPI:rotZ",
        "PhysicsFilteredPairsAPI",
        "physxRigidBody:enableCCD",
        'def Capsule "NeedleInterface"',
        "drAnmar:swageFraction",
    ]
    missing_tokens = [
        token for token in required_asset_tokens if token not in asset_text
    ]
    check(
        checks,
        "runtime_physics_contract",
        not missing_tokens,
        {"missing": missing_tokens},
        required_asset_tokens,
    )
    forbidden_tokens = [
        "Rope.usd",
        "SoftMimicGen",
        "nvidia-strand-ring-threading",
    ]
    present_forbidden = [token for token in forbidden_tokens if token in asset_text]
    check(
        checks,
        "independent_from_current_thread",
        not present_forbidden,
        {"forbidden_references": present_forbidden},
        "no current-thread reference",
    )
    check(
        checks,
        "actual_scale_not_visibility_inflated",
        "drAnmarVisibilityScale" not in asset_text
        and math.isclose(derived.diameter_m, 0.00025),
        derived.diameter_m,
        0.00025,
    )
    needle_identity_tokens = [
        f'defaultPrim = "{DR_ANMAR_NEEDLE_ROOT_PRIM}"',
        f'drAnmarAssetId = "{DR_ANMAR_NEEDLE_ASSET_ID}"',
        f'drAnmarAssetName = "{DR_ANMAR_NEEDLE_NAME}"',
        f'drAnmarAssetVersion = "{DR_ANMAR_NEEDLE_ASSET_VERSION}"',
        "prepend references = @../suture/DrAnmarSuture4_0.usda@",
        'drAnmarGeometrySource = "independently_generated_parametric_geometry"',
        'drAnmarRepresentation = "high_resolution_mesh_with_compound_capsule_collision"',
        "drAnmarResetRandomizationCount = 4",
        "drAnmarSimToRealGapCount = 7",
        'def PhysicsFixedJoint "FactorySwage"',
        'def Mesh "Visual"',
        'def Xform "Needle"',
        f"physics:body0 = </{DR_ANMAR_NEEDLE_ROOT_PRIM}/Needle>",
        f"physics:body1 = </{DR_ANMAR_NEEDLE_ROOT_PRIM}/Suture/NeedleInterface>",
        "physics:kinematicEnabled = false",
        'drAnmarAuthorship = "Independent Dr.Anmar geometry, collision, instrument composition and suture physics"',
    ]
    missing_identity_tokens = [
        token for token in needle_identity_tokens if token not in needle_text
    ]
    check(
        checks,
        "dr_anmar_needle_identity_and_provenance",
        not missing_identity_tokens,
        {"missing": missing_identity_tokens},
        needle_identity_tokens,
    )
    forbidden_needle_tokens = [
        "../Surgical_needle",
        "needle_sdf.usd",
        "ORBIT",
    ]
    present_forbidden_needle_tokens = [
        token for token in forbidden_needle_tokens if token in needle_text
    ]
    check(
        checks,
        "independent_dr_anmar_needle_geometry",
        not present_forbidden_needle_tokens,
        {"forbidden_references": present_forbidden_needle_tokens},
        "no external needle geometry or naming",
    )
    authored_collision_capsules = len(
        re.findall(r'def Capsule "C\d{3}"', needle_text)
    )
    check(
        checks,
        "needle_visual_and_collision_resolution",
        len(needle_mesh.points) == derived_needle.visual_vertex_count
        and authored_collision_capsules
        == derived_needle.collision_capsule_count,
        {
            "visual_vertices": len(needle_mesh.points),
            "collision_capsules": authored_collision_capsules,
        },
        {
            "visual_vertices": derived_needle.visual_vertex_count,
            "collision_capsules": derived_needle.collision_capsule_count,
        },
    )
    construction = needle_profile["construction"]
    arc_range = [
        float(value)
        for value in construction["centerline_arc_length_range_m"]
    ]
    diameter_range = [
        float(value)
        for value in construction["body_diameter_range_m"]
    ]
    check(
        checks,
        "needle_scale_and_mass",
        arc_range[0] <= derived_needle.arc_length_m <= arc_range[1]
        and diameter_range[0]
        <= 2.0 * derived_needle.body_radius_m
        <= diameter_range[1]
        and 0.0 < derived_needle.mass_kg < 0.001,
        {
            "arc_length_m": derived_needle.arc_length_m,
            "body_diameter_m": 2.0 * derived_needle.body_radius_m,
            "mass_kg": derived_needle.mass_kg,
        },
        {
            "arc_length_m": arc_range,
            "body_diameter_m": diameter_range,
            "mass_kg": "positive and below 1 gram",
        },
    )
    sim_to_real = needle_profile["sim_to_real"]
    gaps = sim_to_real["gaps"]
    implemented_randomization = sim_to_real[
        "implemented_randomization_on_episode_reset"
    ]
    planned_randomization = sim_to_real[
        "planned_randomization_after_calibration"
    ]
    complete_gaps = all(
        {
            "id",
            "risk",
            "mitigation",
            "calibration_target",
            "status",
        }.issubset(gap)
        for gap in gaps
    )
    check(
        checks,
        "sim_to_real_gap_register",
        len(gaps) >= 7
        and complete_gaps
        and len(implemented_randomization) >= 4
        and len(planned_randomization) >= 4,
        {
            "gap_count": len(gaps),
            "implemented_randomized_parameters": len(
                implemented_randomization
            ),
            "planned_randomized_parameters": len(planned_randomization),
            "complete_gap_records": complete_gaps,
        },
        {
            "gap_count": "at least 7",
            "implemented_randomized_parameters": "at least 4",
            "planned_randomized_parameters": "at least 4",
            "complete_gap_records": True,
        },
    )
    sample_a = sample_episode_parameters(needle_profile, 1701)
    sample_a_replay = sample_episode_parameters(needle_profile, 1701)
    sample_b = sample_episode_parameters(needle_profile, 1702)
    check(
        checks,
        "sim_to_real_randomization_replay",
        sample_a == sample_a_replay and sample_a != sample_b,
        {
            "seed_1701": sample_a.payload(),
            "seed_1701_replay": sample_a_replay.payload(),
            "seed_1702": sample_b.payload(),
        },
        "same seed exactly replays; different seed changes the domain",
    )
    sampled_suture_a, sampled_suture_domain_a = sample_suture_runtime_profile(
        profile,
        2701,
    )
    sampled_suture_replay, sampled_suture_domain_replay = (
        sample_suture_runtime_profile(profile, 2701)
    )
    sampled_suture_b, sampled_suture_domain_b = sample_suture_runtime_profile(
        profile,
        2702,
    )
    suture_gaps = profile["sim_to_real"]["gaps"]
    suture_requirements = profile["qualification"]["requirements"]
    suture_clinical = [
        item for item in suture_requirements if item["id"] == "clinical_use"
    ]
    sampled_self_friction_a = sampled_suture_a["contact"][
        "load_dependent_self_friction"
    ]
    sampled_self_friction_b = sampled_suture_b["contact"][
        "load_dependent_self_friction"
    ]
    check(
        checks,
        "suture_runtime_domain_and_qualification",
        sampled_suture_domain_a == sampled_suture_domain_replay
        and sampled_suture_a == sampled_suture_replay
        and sampled_suture_domain_a != sampled_suture_domain_b
        and sampled_suture_a != sampled_suture_b
        and sampled_self_friction_a != sampled_self_friction_b
        and sampled_suture_a["contact"]["sampled_static_to_dynamic_ratio"]
        == (
            sampled_suture_domain_a["static_friction"]
            / sampled_suture_domain_a["dynamic_friction"]
        )
        and len(suture_gaps) >= 8
        and all(
            {"id", "risk", "mitigation", "status"}.issubset(gap)
            for gap in suture_gaps
        )
        and len(
            profile["sim_to_real"]["runtime_applied_parameter_sampling"]
        )
        >= 7
        and profile["qualification"]["policy"] == "fail_closed"
        and len(suture_requirements) >= 7
        and len(suture_clinical) == 1
        and suture_clinical[0]["status"] == "blocked"
        and profile["clinical_validation"] is False,
        {
            "seed_2701": sampled_suture_domain_a,
            "seed_2701_replay": sampled_suture_domain_replay,
            "seed_2702": sampled_suture_domain_b,
            "gap_count": len(suture_gaps),
            "requirement_count": len(suture_requirements),
            "clinical": suture_clinical,
        },
        "replayable live suture material domain with fail-closed evidence gates",
    )
    qualification = needle_profile["qualification"]
    qualification_gates = qualification["gates"]
    clinical_gates = [
        gate for gate in qualification_gates if gate["id"] == "clinical_use"
    ]
    check(
        checks,
        "fail_closed_sim_to_real_qualification",
        qualification["policy"]
        == "fail_closed_until_each_evidence_gate_is_satisfied"
        and len(qualification_gates) >= 6
        and len(clinical_gates) == 1
        and clinical_gates[0]["status"] == "blocked"
        and needle_profile["clinical_validation"] is False,
        {
            "policy": qualification["policy"],
            "gate_count": len(qualification_gates),
            "clinical_gate": clinical_gates,
            "clinical_validation": needle_profile["clinical_validation"],
        },
        "machine-readable qualification gates with clinical use blocked",
    )
    needle_evidence = needle_profile.get("evidence", [])
    check(
        checks,
        "needle_research_provenance",
        len(needle_evidence) >= 4
        and all(
            item.get("url") and item.get("used_for")
            for item in needle_evidence
        ),
        len(needle_evidence),
        "at least four traceable primary product or regulatory sources",
    )
    first_joint = re.search(
        r'def PhysicsJoint "J0000".*?physics:breakForce = ([0-9.eE+-]+)',
        asset_text,
        re.DOTALL,
    )
    pullout_force_n = float(first_joint.group(1)) if first_joint else None
    check(
        checks,
        "breakable_swage_pullout",
        pullout_force_n is not None
        and math.isclose(
            pullout_force_n,
            float(profile["swage"]["pullout_force_n_seed"]),
        ),
        pullout_force_n,
        float(profile["swage"]["pullout_force_n_seed"]),
    )

    class FakeUsdFileCfg:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeAssetBaseCfg:
        class InitialStateCfg:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeScene:
        pass

    covered_local_rooms: list[str] = []
    for room_id in local_room_ids(PROCEDURE_ROOMS):
        fake_scene = FakeScene()
        configure_dr_anmar_needle(
            fake_scene,
            asset_base_cfg_type=FakeAssetBaseCfg,
            usd_file_cfg_type=FakeUsdFileCfg,
        )
        if getattr(fake_scene, "dr_anmar_needle", None) is not None:
            covered_local_rooms.append(room_id)
    expected_local_rooms = list(local_room_ids(PROCEDURE_ROOMS))
    check(
        checks,
        "all_local_procedure_rooms_receive_instrument",
        covered_local_rooms == expected_local_rooms and bool(covered_local_rooms),
        covered_local_rooms,
        expected_local_rooms,
    )

    syntax_tree = ast.parse(workstation_text)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(syntax_tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    integration_calls = [
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "configure_dr_anmar_needle"
    ]
    direct_main_calls = []
    for call in integration_calls:
        ancestor = parents.get(call)
        guarded = False
        while ancestor is not None and not isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if isinstance(
                ancestor,
                (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith),
            ):
                guarded = True
            ancestor = parents.get(ancestor)
        if (
            isinstance(ancestor, ast.FunctionDef)
            and ancestor.name == "main"
            and not guarded
        ):
            direct_main_calls.append(call.lineno)
    check(
        checks,
        "shared_unconditional_workstation_install",
        len(direct_main_calls) == 1,
        len(direct_main_calls),
        1,
    )
    domain_calls = [
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_dr_anmar_needle_episode_domain"
    ]
    domain_call_owners: list[str] = []
    for call in domain_calls:
        ancestor = parents.get(call)
        while ancestor is not None and not isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            ancestor = parents.get(ancestor)
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            domain_call_owners.append(ancestor.name)
    check(
        checks,
        "live_reset_domain_randomization",
        sorted(domain_call_owners) == ["main", "reset_environment"],
        sorted(domain_call_owners),
        ["main", "reset_environment"],
    )
    runtime_probe = SutureRuntime(profile)
    half_dose_first = runtime_probe.record_instrument_grasp(
        (20,),
        pressure_pa=float(
            profile["instrument_damage"]["reference_crush_pressure_pa"]
        ),
        duration_s=0.5,
    )
    half_dose_second = runtime_probe.record_instrument_grasp(
        (20,),
        pressure_pa=float(
            profile["instrument_damage"]["reference_crush_pressure_pa"]
        ),
        duration_s=0.5,
    )
    live_runtime_tokens = [
        "SutureRuntime(",
        "create_rigid_body_view(",
        "observe_segment_positions(",
        "record_instrument_contact(",
        "apply_to_stage(",
        "force_matrix_w",
        "{ENV_REGEX_NS}/DrAnmarNeedle/Suture/Segments/S.*",
    ]
    missing_live_runtime_tokens = [
        token for token in live_runtime_tokens if token not in workstation_text
    ]
    check(
        checks,
        "live_suture_material_history_wiring",
        not missing_live_runtime_tokens
        and not half_dose_first
        and half_dose_second
        and runtime_probe.joints[20].grasp_count == 1,
        {
            "missing_workstation_tokens": missing_live_runtime_tokens,
            "cumulative_pressure_dose": runtime_probe.joints[20].crush_dose,
            "grasp_count": runtime_probe.joints[20].grasp_count,
        },
        "native tensor poses and filtered per-jaw contact drive cumulative live material history",
    )
    runtime_detection = profile["runtime_detection"]
    broadphase = runtime_detection["self_contact_broadphase"]
    spacing = derived.segment_spacing_m
    straight_positions = [
        (index * spacing, 0.0, 0.0)
        for index in range(derived.segment_count)
    ]
    (
        straight_contacts,
        broadphase_candidates,
        broadphase_overflow_edges,
    ) = (
        runtime_probe._nonadjacent_edge_contacts(
            straight_positions,
            contact_distance_m=float(
                runtime_detection[
                    "self_contact_centerline_distance_m"
                ]
            ),
            minimum_index_separation=int(
                runtime_detection["knot_minimum_index_separation"]
            ),
            cell_size_multiplier=float(
                broadphase["cell_size_to_contact_distance"]
            ),
            maximum_cells_per_edge=int(
                broadphase["maximum_cells_per_edge"]
            ),
        )
    )
    naive_pairs = (
        (
            derived.segment_count
            - 1
            - int(runtime_detection["knot_minimum_index_separation"])
        )
        * (
            derived.segment_count
            - int(runtime_detection["knot_minimum_index_separation"])
        )
        // 2
    )
    check(
        checks,
        "geometry_aware_self_contact_broadphase",
        broadphase["algorithm"]
        == "uniform_spatial_hash_over_expanded_centerline_edge_aabbs"
        and broadphase["narrowphase"]
        == "exact_3d_segment_to_segment_closest_distance"
        and broadphase["deterministic_pair_order"] is True
        and broadphase["overflow_policy"]
        == "exact_test_overflow_edge_against_all_nonadjacent_edges"
        and not straight_contacts
        and broadphase_candidates < naive_pairs * 0.05
        and broadphase_overflow_edges == 0
        and callable(runtime_probe._segment_segment_distance)
        and callable(runtime_probe._point_segment_distance),
        {
            "algorithm": broadphase["algorithm"],
            "narrowphase": broadphase["narrowphase"],
            "straight_contact_count": len(straight_contacts),
            "broadphase_candidates": broadphase_candidates,
            "broadphase_overflow_edges": broadphase_overflow_edges,
            "naive_pairs": naive_pairs,
        },
        "edge-distance contact with deterministic spatial pruning",
    )
    evidence = profile.get("evidence", [])
    check(
        checks,
        "primary_research_provenance",
        len(evidence) >= 6
        and all(item.get("url") and item.get("used_for") for item in evidence),
        len(evidence),
        "at least six traceable experimental/computational sources",
    )
    passed = all(item["passed"] for item in checks.values())
    return {
        "schema": "dr.anmar.suture-validation.v1",
        "profile_id": profile["id"],
        "passed": passed,
        "checks": checks,
        "derived": {
            "mass_kg": derived.mass_kg,
            "segment_mass_kg": derived.segment_mass_kg,
            "axial_rigidity_n": derived.axial_rigidity_n,
            "axial_joint_stiffness_n_m": derived.axial_joint_stiffness_n_m,
            "bend_joint_stiffness_n_m_rad": derived.bend_joint_stiffness_n_m_rad,
            "twist_joint_stiffness_n_m_rad": derived.twist_joint_stiffness_n_m_rad,
            "needle_arc_length_m": derived_needle.arc_length_m,
            "needle_curvature_radius_m": derived_needle.curvature_radius_m,
            "needle_body_diameter_m": 2.0 * derived_needle.body_radius_m,
            "needle_mass_kg": derived_needle.mass_kg,
            "needle_visual_vertex_count": derived_needle.visual_vertex_count,
            "needle_collision_capsule_count": derived_needle.collision_capsule_count,
            "sim_to_real_gap_count": len(gaps),
        },
        "clinical_validation": False,
        "note": "Deterministic engineering validation only; physical bench and clinician validation remain required.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument(
        "--needle-profile",
        type=Path,
        default=DEFAULT_NEEDLE_PROFILE_PATH,
    )
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument(
        "--needle",
        "--assembly",
        dest="needle",
        type=Path,
        default=DEFAULT_NEEDLE,
    )
    parser.add_argument("--workstation", type=Path, default=DEFAULT_WORKSTATION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    needle_profile = load_needle_profile(args.needle_profile)
    asset_text = args.asset.read_text(encoding="utf-8")
    needle_text = args.needle.read_text(encoding="utf-8")
    workstation_text = args.workstation.read_text(encoding="utf-8")
    report = validate(
        profile,
        needle_profile,
        asset_text,
        needle_text,
        workstation_text,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
