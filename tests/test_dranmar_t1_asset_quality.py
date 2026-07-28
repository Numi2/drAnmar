# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config/dranmar_t1_asset_quality.json"
ASSET_DATA = ROOT / "source/extensions/orbit.surgical.assets/data"
TISSUE_ROOT = ASSET_DATA / "Props/SurgicalTissue/NeedleReadyTissueUnit"
SCENE_ROOT = ASSET_DATA / "Props/SurgicalScene/T1"
NEEDLE_ROOT = ASSET_DATA / "Props/SurgicalClosure/NeedleT1Compatibility"
PSM_CANDIDATE_ROOT = ASSET_DATA / "Robots/dVRK/PSM/T1JawContactCandidate"
PSM_COLLIDER_ROOT = ASSET_DATA / "Robots/dVRK/PSM/T1ColliderCandidate"
TABLE_COLLIDER_ROOT = ASSET_DATA / "Props/Table/T1ColliderCandidate"
HANDOVER_NEEDLE_CONFIG = ROOT / (
    "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/surgical/"
    "handover/config/needle/joint_pos_env_cfg.py"
)
T1_TASK_CONFIG = HANDOVER_NEEDLE_CONFIG.with_name("t1_safe_bite_env_cfg.py")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_separates_source_qualification_from_native_claims():
    contract = load_json(CONTRACT_PATH)
    assert contract["schema"] == "dr.anmar.t1-asset-quality-contract.v1"
    assert contract["version"] == "1.1.0"
    assert contract["status"] == (
        "source_static_integrity_qualified_native_qualification_pending"
    )

    claims = contract["claims"]
    for claim in (
        "source_contract_implemented",
        "static_source_integrity_qualified",
        "tissue_v2_1_geometry_quality_qualified",
        "visual_packages_source_qualified",
        "inactive_candidates_source_qualified",
    ):
        assert claims[claim] is True
    for claim in (
        "native_current_tissue_newton_qualified",
        "native_isaac_asset_spawn_qualified",
        "native_tissue_visual_sync_qualified",
        "native_rtx_material_and_render_qualified",
        "native_needle_dynamics_and_task_parity_qualified",
        "native_psm_jaw_contact_qualified",
        "native_psm_collider_qualified",
        "native_table_collider_qualified",
        "biomechanical_calibration_qualified",
        "puncture_or_damage_qualified",
        "clinical_validation",
    ):
        assert claims[claim] is False
    assert claims["numi_execution"] == "not_executed"


def test_training_and_visual_lanes_share_physics_but_not_render_cost():
    contract = load_json(CONTRACT_PATH)
    training = contract["lanes"]["training"]
    visual = contract["lanes"]["visual_qualification"]
    assert training["cameras_enabled"] is False
    assert training["rtx_enabled"] is False
    assert training["maximum_requested_environments"] == 2400
    assert training["execution_requires_explicit_user_approval"] is True
    assert visual["cameras_enabled"] is True
    assert visual["rtx_enabled"] is True
    assert visual["maximum_environments"] == 4
    assert visual["physics_lod_policy"] == "mirror_the_source_episode_exactly"
    assert visual["may_not_be_used_to_infer_training_throughput"] is True
    assert contract["shared_authority"]["physics"][
        "render_meshes_are_never_collision_or_success_authority"
    ] is True


def test_current_t1_manifest_receipts_match_repository_bytes():
    assets = load_json(CONTRACT_PATH)["manifest_and_provenance"]["assets"]
    receipts = (
        (assets["tissue"], "manifest", "manifest_sha256"),
        (assets["tissue"], "visual_manifest", "visual_manifest_sha256"),
        (assets["tissue"], "geometry_report", "geometry_report_sha256"),
        (assets["scene_visuals"], "manifest", "manifest_sha256"),
        (assets["needle_t1_candidate"], "manifest", "manifest_sha256"),
        (assets["psm_jaw_contact_candidate"], "manifest", "manifest_sha256"),
        (assets["psm_collider_candidate"], "manifest", "manifest_sha256"),
        (assets["table_collider_candidate"], "manifest", "manifest_sha256"),
    )
    for receipt, path_key, hash_key in receipts:
        path = ROOT / receipt[path_key]
        assert path.is_file()
        assert sha256(path) == receipt[hash_key]

    assert load_json(TISSUE_ROOT / "asset_manifest.json")["asset_version"] == (
        assets["tissue"]["asset_version"]
    )
    assert load_json(TISSUE_ROOT / "visual_manifest.json")["visual_only"] is True
    scene_manifest = load_json(SCENE_ROOT / "asset_manifest.json")
    assert scene_manifest["version"] == assets["scene_visuals"]["asset_version"]
    assert scene_manifest["dependency_complete_directory"] is False
    assert assets["scene_visuals"]["base_overlay_dependencies_hash_locked"] is True
    assert load_json(NEEDLE_ROOT / "asset_manifest.json")["status"] == (
        "inactive_qualification_candidate"
    )
    psm_manifest = load_json(PSM_CANDIDATE_ROOT / "asset_manifest.json")
    assert psm_manifest["active_replacement"] is False
    assert psm_manifest["dependency_complete_directory"] is False
    assert assets["psm_jaw_contact_candidate"][
        "base_overlay_dependencies_hash_locked"
    ] is True
    for name, root in (
        ("psm_collider_candidate", PSM_COLLIDER_ROOT),
        ("table_collider_candidate", TABLE_COLLIDER_ROOT),
    ):
        manifest = load_json(root / "asset_manifest.json")
        assert manifest["active_replacement"] is False
        assert manifest["dependency_complete_directory"] is False
        assert assets[name]["external_overlay_dependencies_hash_locked"] is True


def test_tissue_v2_1_counts_fixture_and_quality_metrics_are_exact():
    contract = load_json(CONTRACT_PATH)["tissue"]
    report = load_json(TISSUE_ROOT / "geometry_report.json")
    geometry = contract["geometry_quality"]
    assert report["asset_version"] == geometry["asset_version"] == "2.1.0"
    assert geometry["wound_edge_refinement_power"] == 1.1

    expected_points = {"training": 560, "contact": 2470, "validation": 16650}
    expected_fixture = {"training": 80, "contact": 380, "validation": 1998}
    assert contract["visual_sync"]["lod_point_counts"] == expected_points
    assert geometry["fixture_anchor_nodes"] == expected_fixture
    for lod in expected_points:
        observed = report["lods"][lod]
        assert observed["point_count"] == expected_points[lod]
        assert observed["node_set_counts"]["anchor_outer"] == expected_fixture[lod]
        for metric in (
            "minimum_mean_ratio",
            "minimum_scaled_jacobian",
            "maximum_edge_ratio",
        ):
            assert observed["tetrahedron_quality"][metric] == geometry[lod][
                metric
            ]
        gates = observed["tetrahedron_quality_gates"]
        assert geometry[lod]["minimum_mean_ratio"] >= gates["minimum_mean_ratio"]
        assert geometry[lod]["minimum_scaled_jacobian"] >= gates[
            "minimum_scaled_jacobian"
        ]
        assert geometry[lod]["maximum_edge_ratio"] <= gates[
            "maximum_edge_ratio"
        ]

    assert contract["visual_sync"][
        "detached_high_resolution_visual_surface_allowed"
    ] is False
    assert contract["lod_policy"]["silent_lod_substitution_allowed"] is False
    assert geometry["native_deformation_stability_qualified"] is False


def test_tissue_and_scene_material_contracts_are_matte_and_visual_only():
    contract = load_json(CONTRACT_PATH)
    tissue = contract["tissue"]["material"]
    visual_manifest = load_json(TISSUE_ROOT / "visual_manifest.json")
    scene_manifest = load_json(SCENE_ROOT / "asset_manifest.json")

    assert tissue["primary"] == "OpenPBR 1.1 MaterialX"
    assert tissue["fallback"] == "UsdPreviewSurface"
    assert tissue["texture_resolution_px"] == 2048
    assert tissue["normal_encoding"] == "JPEG quality 99 4:4:4"
    assert tissue["surface_and_wound_roughness_minimum"] >= 0.58
    assert tissue["coat_weight"] == 0.0
    assert tissue["metalness"] == 0.0
    assert tissue["thin_walled"] is False
    assert visual_manifest["physics_authority"] is False
    assert visual_manifest["visual_only"] is True
    assert scene_manifest["physics_authority"] is False
    assert scene_manifest["collision_authority"] is False
    assert scene_manifest["visual_only"] is True


def test_needle_scale_quaternion_and_composed_parity_contract_fail_closed():
    contract = load_json(CONTRACT_PATH)
    needle = contract["needle"]
    geometry = load_json(NEEDLE_ROOT / "geometry_contract.json")
    spawn = geometry["runtime_spawn_contract"]
    frames = load_json(NEEDLE_ROOT / "interaction_frames.json")
    assets = contract["manifest_and_provenance"]["assets"]

    assert needle["active_legacy_scale"] == 0.4
    assert assets["needle_legacy_active"]["spawn_scale"] == 0.4
    assert assets["needle_legacy_active"][
        "visual_overlay_must_retain_spawn_scale"
    ] == 0.4
    assert needle["inactive_candidate_scale"] == 1.0
    assert assets["needle_t1_candidate"]["spawn_scale"] == 1.0
    assert spawn["legacy_active_scale_xyz"] == [0.4, 0.4, 0.4]
    assert spawn["candidate_scale_xyz"] == [1.0, 1.0, 1.0]
    assert spawn["path_only_substitution_allowed"] is False
    assert needle["path_only_substitution_allowed"] is False
    assert spawn["promotion_requires_composed_world_space_frame_parity"] is True
    assert set(needle["promotion_composition_gate"]["world_space_parity_required"]) == {
        "tip position",
        "circle centre",
        "centreline radius",
        "grasp frame",
    }

    frames_contract = contract["shared_authority"]["coordinate_frames"]
    assert frames_contract["runtime_and_json_quaternion_order"] == "xyzw"
    assert frames_contract["runtime_identity_quaternion_xyzw"] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert frames_contract["openusd_quaternion_serialization"] == "wxyz"
    assert frames_contract["cross_boundary_tuple_copy_allowed"] is False
    assert spawn["candidate_root_orientation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert frames["coordinate_convention"]["runtime_quaternion_order"] == "xyzw"
    assert frames["coordinate_convention"]["openusd_quaternion_serialization"] == (
        "wxyz"
    )

    active_source = HANDOVER_NEEDLE_CONFIG.read_text(encoding="utf-8")
    assert "Props/Surgical_needle/needle_sdf.usd" in active_source
    assert re.search(r"scale\s*=\s*\(0\.4,\s*0\.4,\s*0\.4\)", active_source)
    assert "NeedleT1Compatibility" not in active_source
    assert "NeedleT1Compatibility" not in T1_TASK_CONFIG.read_text(
        encoding="utf-8"
    )


def test_needle_candidate_geometry_material_and_dynamics_stay_inactive():
    needle = load_json(CONTRACT_PATH)["needle"]
    geometry = needle["candidate_geometry"]
    report = load_json(NEEDLE_ROOT / "geometry_report.json")
    physics = load_json(NEEDLE_ROOT / "physics_profile.json")

    assert report["topology"]["connected_component_count"] == (
        geometry["connected_components"]
    ) == 1
    assert report["topology"]["watertight"] is geometry["watertight"] is True
    assert report["topology"]["vertex_count"] == geometry["vertex_count"] == 12546
    assert report["topology"]["triangle_count"] == (
        geometry["triangle_count"]
    ) == 25088
    assert report["collision_capsule_count"] == (
        geometry["collision_capsule_count"]
    ) == 96
    assert physics["mass_properties"]["mass_kg"] == needle["candidate_dynamics"][
        "mass_kg"
    ]
    assert physics["active_replacement"] is False
    assert physics["status"] == "unqualified_candidate"
    assert needle["candidate_dynamics"]["native_qualified"] is False
    assert needle["candidate_material"]["measured_roughness_range"] == [
        0.475,
        0.6,
    ]
    with Image.open(
        NEEDLE_ROOT / "textures/needle_satin_roughness.png"
    ) as roughness:
        minimum, maximum = roughness.getextrema()
    assert round(minimum / 255.0, 3) == 0.475
    assert round(maximum / 255.0, 3) == 0.6
    assert needle["candidate_material"]["metalness"] == 1.0
    assert needle["candidate_material"]["clearcoat_weight"] == 0.0


def test_psm_contact_candidate_is_friction_only_and_unqualified():
    psm = load_json(CONTRACT_PATH)["psm"]["jaw_contact_candidate"]
    hypothesis = load_json(PSM_CANDIDATE_ROOT / "friction_hypothesis.json")
    qualification = load_json(PSM_CANDIDATE_ROOT / "qualification_contract.json")
    seed = hypothesis["candidate_seed"]
    contact = hypothesis["contact_model"]

    assert seed["static_friction"] == psm["static_friction"] == 0.6
    assert seed["dynamic_friction"] == psm["dynamic_friction"] == 0.45
    assert seed["static_friction"] >= seed["dynamic_friction"]
    for mechanism in ("adhesion", "cohesion", "magnetism", "suction"):
        assert contact[mechanism] == psm[mechanism] == 0.0
    assert psm["uncalibrated_engineering_hypothesis"] is True
    assert psm["active"] is False
    assert qualification["activation"]["default"] == "blocked"
    assert qualification["validation_boundaries"]["physics_calibration"] is False
    assert qualification["validation_boundaries"]["clinical_validation"] is False


def test_low_complexity_collider_candidates_are_inactive_source_hypotheses():
    contract = load_json(CONTRACT_PATH)
    candidates = (
        (
            contract["psm"]["collider_candidate"],
            PSM_COLLIDER_ROOT,
            288052,
            48,
            31,
            "native_contact_and_task_qualified",
        ),
        (
            contract["table_and_fixture"]["collider_candidate"],
            TABLE_COLLIDER_ROOT,
            127622,
            28,
            7,
            "native_contact_and_support_qualified",
        ),
    )
    for candidate, root, legacy_triangles, candidate_triangles, primitives, native_key in (
        candidates
    ):
        report = load_json(root / "approximation_report.json")
        geometry = load_json(root / "geometry_contract.json")
        qualification = load_json(root / "qualification_contract.json")
        manifest = load_json(root / "asset_manifest.json")

        legacy_key = (
            "legacy_enabled_collision_mesh_triangles"
            if "legacy_enabled_collision_mesh_triangles" in report
            else "legacy_collision_mesh_triangles"
        )
        assert report[legacy_key] == legacy_triangles
        assert report["candidate_mesh_triangles"] == candidate_triangles
        assert report["candidate_primitive_count"] == primitives
        assert report["broad_single_shell_used"] is False
        assert candidate["candidate_mesh_triangles"] == candidate_triangles
        assert candidate["candidate_primitive_count"] == primitives
        assert candidate["broad_single_shell_used"] is False
        assert candidate["active"] is False
        assert candidate[native_key] is False
        assert geometry["runtime_contract"]["allowed_runtime_scale"] == [
            1.0,
            1.0,
            1.0,
        ]
        assert geometry["runtime_contract"][
            "root_transform_inherited_exactly"
        ] is True
        assert manifest["runtime_references"] == []
        assert manifest["source_static_validation"] is True
        assert qualification["activation"]["default"] == "blocked"
        assert qualification["validation_boundaries"][
            "native_isaac_spawn_qualified"
        ] is False
        assert qualification["validation_boundaries"]["physics_calibration"] is False
        assert qualification["validation_boundaries"]["clinical_validation"] is False

    table = contract["table_and_fixture"]["collider_candidate"]
    assert table["supported_drape_only"] is True
    assert table["unsupported_hanging_drape_is_rigid"] is False
    assert table["vertical_top_support_error_m"] == 0.0


def test_changed_tissue_topology_does_not_inherit_old_native_evidence():
    boundary = load_json(CONTRACT_PATH)["historical_evidence_boundary"]
    report = load_json(TISSUE_ROOT / "geometry_report.json")
    contact = load_json(
        ROOT / "physics_next/benchmarks/needle-ready-tissue/contact-newton.json"
    )

    assert boundary["current_tissue_version"] == report["asset_version"] == "2.1.0"
    assert boundary["historical_tissue_version"] == "2.0.0"
    assert boundary["historical_contact_sha256"] == contact["asset"]["sha256"]
    assert boundary["current_contact_sha256"] == report["lods"]["contact"][
        "usd_sha256"
    ]
    assert boundary["historical_contact_sha256"] != boundary[
        "current_contact_sha256"
    ]
    for claim in (
        "historical_deterministic_replay_transfers",
        "historical_isaac_spawn_transfers",
        "historical_2400_environment_capacity_transfers",
    ):
        assert boundary[claim] is False
    assert boundary["historical_contact_health"][
        "healthy_calibrated_tissue_qualification"
    ] is False


def test_render_capture_and_documentation_preserve_claim_boundaries():
    contract = load_json(CONTRACT_PATH)
    capture = contract["render_capture"]
    color = capture["exposure_and_color"]
    assert capture["status"] == "source_contract_only_pending_native_rtx_execution"
    assert color["auto_exposure"] is False
    assert color["exposure_mode"] == "manual"
    assert color["tone_mapper"] == "ACES"
    assert color["working_output"] == "scene-linear half-float EXR"
    assert capture["camera"]["motion_blur"] is False
    assert capture["lighting"]["environment_dome_may_not_be_the_only_light"] is True

    documentation = (
        ROOT / "docs/DRANMAR_T1_ASSET_REALISM.md"
    ).read_text(encoding="utf-8")
    for required in (
        "80 | 0.340085",
        "380 | 0.507860",
        "1,998 | 0.505563",
        "scale **0.4**",
        "scale **1.0**",
        "0.475–0.600",
        "(x, y, z, w)",
        "(w, x, y, z)",
        "288,052",
        "127,622",
        "hanging drape remains non-colliding",
        "not healthy calibrated-tissue evidence",
        "No native Isaac, RTX, Numi",
        "clinical validation",
    ):
        assert required in documentation
