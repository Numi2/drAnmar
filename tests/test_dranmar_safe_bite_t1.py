from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config/dranmar_safe_bite_t1.json"
GEOMETRY_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets"
    / "data/Props/SurgicalTissue/NeedleReadyTissueUnit"
    / "geometry_contract.json"
)
GEOMETRY_REPORT_PATH = GEOMETRY_PATH.with_name("geometry_report.json")
TASK_ROOT = (
    ROOT / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/surgical" / "handover"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_contract_module():
    path = ROOT / "scripts/dr_anmar_safe_bite_t1.py"
    spec = importlib.util.spec_from_file_location("dr_anmar_safe_bite_t1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_t1_contract_is_practical_and_does_not_block_puncture():
    contract = load_json(CONTRACT_PATH)
    success = contract["success"]
    rewards = contract["rewards"]
    puncture = contract["puncture_transition"]
    curriculum = contract["handover_snapshot_curriculum"]
    fixture = contract["scene"]["fixture"]
    assert contract["status"] == ("implemented_pending_native_isaac_lab_validation")
    assert contract["version"] == "3.0.0"
    assert success["position_tolerance_m"] == 0.005
    assert math.degrees(success["tip_tangent_tolerance_rad"]) == 25.0
    assert math.isclose(math.degrees(success["needle_plane_tolerance_rad"]), 30.0)
    assert success["stable_control_steps"] == 5
    assert success["post_arm_inward_action_limit"] == 0.03
    assert rewards["holding_still_reward"] == 0.0
    assert rewards["absolute_proximity_reward"] == 0.0
    assert rewards["contact_reward"] == 0.0
    assert rewards["approach_progress_weight"] > 0.0
    assert puncture["mechanically_blocked_by_t1"] is False
    assert puncture["generic_needle_contact_is_puncture"] is False
    assert puncture["recorded_transition_semantics"] == (
        "pre_puncture_tip_entry_contact_onset_not_puncture"
    )
    assert puncture["policy_written_puncture_state_allowed"] is False
    assert curriculum["restore_probability"] == 0.8
    assert curriculum["minimum_full_chain_fraction"] == 0.2
    assert curriculum["restore_schedule"] == ("per_environment_rotating_quota")
    assert curriculum["full_chain_stride"] == 5
    assert curriculum["promotion_uses_snapshot_restore"] is False
    assert fixture["mode"] == "kinematic_authored_outer_attachment_band"
    assert fixture["expected_anchor_nodes_by_lod"] == {
        "training": 80,
        "contact": 380,
        "validation": 1998,
    }
    assert fixture["refinement_power_1_1_physics_requalification_status"] == (
        "required_not_executed"
    )
    assert fixture["wound_and_safe_bite_regions_remain_dynamic"] is True
    pipeline = contract["scene"]["contact_pipeline"]
    assert pipeline["maximum_environment_count"] == 2400
    assert pipeline["maximum_soft_candidate_pairs"] == 15_000_000
    assert pipeline["maximum_contact_pipeline_memory_bytes"] == 536_870_912
    assert (
        contract["launch_profiles"]["training_2400"]["execution_requires_explicit_user_approval"]
        is True
    )


def test_cpu_sampler_stays_inside_versioned_safe_bite_geometry():
    module = load_contract_module()
    contract = load_json(CONTRACT_PATH)
    geometry = load_json(GEOMETRY_PATH)
    geometry_report = load_json(GEOMETRY_REPORT_PATH)
    report = module.validate_contract(contract, geometry, geometry_report)
    assert report["valid"] is True
    assert report["fixture_geometry_report_cross_checked"] is True
    training_preflight = report["launch_preflight"]["training_2400"]
    assert training_preflight == {
        "profile": "training_2400",
        "environment_count": 2400,
        "tissue_lod": "training",
        "surface_particles_per_environment": 400,
        "approved_soft_shapes_per_environment": 14,
        "soft_candidate_pairs": 13_440_000,
        "soft_contact_capacity": 614_400,
        "rigid_sensor_contact_capacity": 153_600,
        "estimated_contact_pipeline_bytes": 298_905_600,
        "execution_requires_explicit_user_approval": True,
    }
    first = module.sample_entry_frame(contract, geometry, seed=17, environment_index=0)
    replay = module.sample_entry_frame(contract, geometry, seed=17, environment_index=0)
    assert first == replay
    samples = [
        module.sample_entry_frame(contract, geometry, seed=17, environment_index=index)
        for index in range(512)
    ]
    assert {sample.flap for sample in samples} == {"left", "right"}
    for sample in samples:
        assert 0.005 <= sample.bite_distance_from_wound_m <= 0.009
        assert 0.005 <= sample.stand_off_m <= 0.008
        assert math.isclose(
            math.sqrt(sum(value * value for value in sample.desired_tip_direction)),
            1.0,
            abs_tol=1.0e-12,
        )
        assert sample.target_tip_position_m[2] > sample.surface_point_m[2]
        assert math.isclose(
            sum(
                left * right
                for left, right in zip(
                    sample.desired_tip_direction,
                    sample.desired_needle_plane_normal,
                    strict=True,
                )
            ),
            0.0,
            abs_tol=1.0e-12,
        )


def test_cpu_frame_orthonormalization_and_launch_preflight_fail_closed():
    module = load_contract_module()
    contract = load_json(CONTRACT_PATH)
    geometry = load_json(GEOMETRY_PATH)
    generator = random.Random(2361)
    for _ in range(256):
        tip = tuple(generator.uniform(-1.0, 1.0) for _ in range(3))
        tangent = tuple(generator.uniform(-1.0, 1.0) for _ in range(3))
        plane = module.orthonormalize_plane_normal(
            tip,
            tangent,
            (0.0, 0.0, 1.0),
        )
        normalized_tip = tuple(value / math.sqrt(sum(item * item for item in tip)) for value in tip)
        assert math.isclose(
            math.sqrt(sum(value * value for value in plane)),
            1.0,
            abs_tol=1.0e-12,
        )
        assert math.isclose(
            sum(
                left * right
                for left, right in zip(
                    normalized_tip,
                    plane,
                    strict=True,
                )
            ),
            0.0,
            abs_tol=1.0e-12,
        )

    drifted = json.loads(json.dumps(contract))
    drifted["launch_profiles"]["training_2400"]["expected_soft_candidate_pairs"] -= 1
    with pytest.raises(ValueError, match="candidate-pair count drifted"):
        module.preflight_launch_profile(
            drifted,
            geometry,
            "training_2400",
        )


def test_t1_is_registered_with_coupled_tissue_and_continuation_task():
    registration = (TASK_ROOT / "config/needle/__init__.py").read_text(encoding="utf-8")
    environment = (TASK_ROOT / "config/needle/t1_safe_bite_env_cfg.py").read_text(encoding="utf-8")
    assert "Isaac-Handover-Needle-Safe-Bite-T1-v0" in registration
    assert "Isaac-Handover-Needle-Safe-Bite-Chain-v0" in registration
    assert "make_needle_ready_tissue_cfg(" in environment
    assert "DrAnmarCoupledMJWarpVBDSolverCfg(" in environment
    assert "maximum_soft_candidate_pairs=int(" in environment
    assert "expected_surface_particles_per_environment=int(" in environment
    assert 'coupling_mode=str(vbd["coupling_mode"])' in environment
    assert 'scene_contract["continuation_tissue_lod"]' in environment
    assert "continuation=True" in environment
    assert "mdp.reset_tissue_outer_fixture" in environment


def test_visual_lane_swaps_only_spawn_usd_paths_for_scene_overlays():
    environment = (
        ROOT
        / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/"
        "surgical/handover/config/needle/t1_safe_bite_env_cfg.py"
    ).read_text(encoding="utf-8")
    visual_function = environment.split(
        "def _configure_visual_qualification_scene",
        maxsplit=1,
    )[1].split("@configclass", maxsplit=1)[0]
    assert visual_function.count(".spawn.replace(") == 4
    assert visual_function.count("usd_path=str(visual_root /") == 4
    for entrypoint in (
        "psm_visual_v1.usda",
        "table_visual_v1.usda",
        "legacy_needle_visual_v1.usda",
    ):
        assert entrypoint in visual_function
    spawn_swap_block = visual_function.split("tissue_position =", maxsplit=1)[0]
    assert "scale=" not in spawn_swap_block
    assert "init_state=" not in spawn_swap_block
    training_function = environment.split(
        "class NeedleHandoverSafeBiteT1EnvCfg",
        maxsplit=1,
    )[1].split(
        "def _configure_visual_qualification_scene",
        maxsplit=1,
    )[0]
    assert "SurgicalScene" not in training_function
    assert "self.terminations.success = None" in environment
    assert "self.rewards.safe_bite_success.weight = 0.0" in environment
    assert '"training_2400"' in environment
    assert '"contact_qualification_256"' in environment


def test_t1_success_and_progress_are_physics_derived():
    state_source = (TASK_ROOT / "mdp/safe_bite.py").read_text(encoding="utf-8")
    handover_source = (TASK_ROOT / "mdp/state.py").read_text(encoding="utf-8")
    assert 'handover["successful_handover"]' in state_source
    assert 'handover["receiver_contact_now"]' in state_source
    assert "minimum_needle_clearance" in state_source
    assert "receiver_tool_clearance" in state_source
    assert "giver_tool_clearance" in state_source
    assert "tissue.data.nodal_pos_w" in state_source
    assert "_live_top_surface_from_material_coordinates" in state_source
    assert 'state["material_uv"]' in state_source
    assert "native_tip_entry_contact" in state_source
    assert "entry_particle_roi" in state_source
    assert "contact_authority_ready" in state_source
    assert "receipt_generation_advanced" in state_source
    assert "obj.data.root_link_lin_vel_w" in state_source
    assert "obj.data.root_link_ang_vel_w" in state_source
    assert "tip_velocity_w = object_root_linear_velocity_w + torch.cross(" in (state_source)
    assert "psm_tool_gripper1_link" in state_source
    assert "psm_tool_gripper2_link" in state_source
    assert "receiver_tool_samples_w" in state_source
    assert "write_nodal_kinematic_target_to_sim_index" in state_source
    assert 'tissue_contact & ~state["entry_armed"]' in state_source
    assert 'cache["reset_attempt_count"][env_ids] += 1' in state_source
    assert "force_full_chain" in state_source
    assert 'state["previous_normalized_error"] - state["normalized_error"]' in (state_source)
    assert (
        'handover["successful_handover"]\n        & handover["receiver_contact_now"]'
    ) in state_source
    assert 'state["entry_armed"] |=' in state_source
    assert 'state["premature_tissue_contact"] |=' in state_source
    assert 'state["authorized_contact_transition"] |=' in state_source
    assert "_dr_anmar_pending_safe_bite_restore" in handover_source
    assert 'state["successful_handover"][restored] = True' in (handover_source)


def test_coupled_manager_accumulates_native_penetration_and_fixes_sensors():
    source = (TASK_ROOT / "newton_contact_manager.py").read_text(encoding="utf-8")
    assert "class DrAnmarCoupledMJWarpVBDManager" in source
    assert "if penetration <= 0.0:" in source
    assert "wp.atomic_max(penetration_seen" in source
    assert "cls._observe_soft_contacts(cls._contacts, state)" in source
    assert "cls._rigid_solver.update_contacts(" in source
    assert "def _build_bounded_soft_pairs(" in source
    assert "surface[surface_world == world]" not in source
    assert "global_shapes_by_world = np.broadcast_to(" in source
    assert "pair_grid[..., 0] = surface_by_world[:, :, None]" in source
    assert "model.particle_count = 0" in source
    assert "class _DrAnmarBoundedSolverVBD(SolverVBD):" in source
    assert "dranmar_soft_contact_max=soft_contact_capacity" in source
    assert "self._dranmar_soft_contact_max" in source
    assert "super()._build_solver(model, solver_cfg)" not in source
    assert "soft_contact_tids_size=int(" in source
    assert "maximum_contact_pipeline_memory_bytes" in source
    assert "cls._rigid_solver.reset(" in source
    assert "cls._soft_solver.reset(" in source
    assert "NewtonManager._contacts = Contacts(" in source
    assert "cls._dr_anmar_rigid_sensor_contacts = Contacts(" in source
    assert 'requested_attributes.add("force")' in source
    assert "_collision_pipeline.collide(" not in source
    assert '"generic_needle_contact_is_puncture": False' in source


def test_newton_qualification_has_bounded_motion_and_recovery_gates():
    source = (ROOT / "scripts/dr_anmar_needle_ready_tissue_newton.py").read_text(encoding="utf-8")
    for gate in (
        "peak_displacement_bounded",
        "peak_speed_bounded",
        "global_volume_error_bounded",
        "recovery_residual_bounded",
        "final_free_speed_bounded",
    ):
        assert gate in source
    assert "and all(sanity_gates.values())" in source


def test_learned_authority_is_receiver_pose_only_after_handover():
    model = (TASK_ROOT / "safe_bite_model.py").read_text(encoding="utf-8")
    end_to_end = (TASK_ROOT / "end_to_end_model.py").read_text(encoding="utf-8")
    assert "receiver_pose_rows[7:13] = 1.0" in model
    assert "receiver_role_mask[:, :6]" in model
    assert "safe_receiver_action" in model
    assert "contact_transition_active" in model
    assert "raw[:, 119:122]" in model
    assert "inward_receiver_action" in model
    assert "SAFE_BITE_HANDOVER_COMPLETE" in model
    assert "self.receiver_adaptation_enabled = True" in model
    assert "for parameter in self.phase_network.trunk.parameters()" not in (model)
    assert "raw[:, 107:]" in end_to_end


def test_benchmark_records_t1_and_accepts_nonterminal_chain_task():
    benchmark = (ROOT / "scripts/dr_anmar_learning_benchmark.py").read_text(encoding="utf-8")
    assert "success_term = term_values.get(" in benchmark
    assert '"first_episode_safe_bite_diagnostics"' in benchmark
    assert '"ever_authorized_contact_transition"' in benchmark
    assert '"ever_native_contact_available"' in benchmark
    assert '"ever_native_contact_overflow"' in benchmark
    assert '"ever_tip_entry_contact"' in benchmark
    assert '"maximum_contact_fresh_generations"' in benchmark
    assert '"safe_bite_t1"' in benchmark
    assert '"snapshot_cached_environments"' in benchmark
    assert '"fixture_anchor_nodes_per_environment"' in benchmark
    assert '"puncture_mechanically_blocked": False' in benchmark
    assert "def _safe_bite_tissue_snapshot(" in benchmark
    assert '"anchor_nodes_per_environment"' in benchmark
    assert '"nodal_position_all_finite"' in benchmark


def test_learning_path_promotes_t1_only_from_full_chain_evidence():
    path = load_json(ROOT / "config/dranmar_learning_path.json")
    t1 = path["stages"][-1]
    assert t1["stage"] == 7
    assert t1["task"] == "DrAnmar-Handover-Needle-Safe-Bite-T1-v0"
    assert t1["learning"]["handover_checkpoint_transfer"] is False
    assert t1["learning"]["holding_still_credit"] is False
    assert t1["learning"]["puncture_mechanically_blocked"] is False
    assert t1["learning"]["handover_snapshot_schedule"] == ("per_environment_rotating_quota")
    assert t1["learning"]["full_chain_stride"] == 5
    assert t1["learning"]["post_arm_inward_action_limit"] == 0.03
    assert t1["promotion"]["minimum_full_chain_success_rate"] == 0.8
    assert t1["promotion"]["execution_requires_explicit_user_approval"] is True
