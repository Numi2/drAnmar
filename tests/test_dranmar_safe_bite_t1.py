from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config/dranmar_safe_bite_t1.json"
GEOMETRY_PATH = (
    ROOT
    / "source/extensions/orbit.surgical.assets"
    / "data/Props/SurgicalTissue/NeedleReadyTissueUnit"
    / "geometry_contract.json"
)
TASK_ROOT = (
    ROOT
    / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/surgical"
    / "handover"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_contract_module():
    path = ROOT / "scripts/dr_anmar_safe_bite_t1.py"
    spec = importlib.util.spec_from_file_location(
        "dr_anmar_safe_bite_t1", path
    )
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
    assert contract["status"] == (
        "implemented_pending_native_isaac_lab_validation"
    )
    assert success["position_tolerance_m"] == 0.005
    assert math.degrees(success["tip_tangent_tolerance_rad"]) == 25.0
    assert math.isclose(
        math.degrees(success["needle_plane_tolerance_rad"]), 30.0
    )
    assert success["stable_control_steps"] == 5
    assert success["post_arm_inward_action_limit"] == 0.03
    assert rewards["holding_still_reward"] == 0.0
    assert rewards["absolute_proximity_reward"] == 0.0
    assert rewards["contact_reward"] == 0.0
    assert rewards["approach_progress_weight"] > 0.0
    assert puncture["mechanically_blocked_by_t1"] is False
    assert puncture["contact_after_entry_armed"] == "allowed_and_recorded"
    assert puncture["policy_written_puncture_state_allowed"] is False
    assert curriculum["restore_probability"] == 0.8
    assert curriculum["minimum_full_chain_fraction"] == 0.2
    assert curriculum["restore_schedule"] == (
        "per_environment_rotating_quota"
    )
    assert curriculum["full_chain_stride"] == 5
    assert curriculum["promotion_uses_snapshot_restore"] is False
    assert fixture["mode"] == "kinematic_outer_attachment_edge"
    assert fixture["expected_training_lod_anchor_nodes"] == 80
    assert fixture["wound_and_safe_bite_regions_remain_dynamic"] is True


def test_cpu_sampler_stays_inside_versioned_safe_bite_geometry():
    module = load_contract_module()
    contract = load_json(CONTRACT_PATH)
    geometry = load_json(GEOMETRY_PATH)
    report = module.validate_contract(contract, geometry)
    assert report["valid"] is True
    first = module.sample_entry_frame(
        contract, geometry, seed=17, environment_index=0
    )
    replay = module.sample_entry_frame(
        contract, geometry, seed=17, environment_index=0
    )
    assert first == replay
    samples = [
        module.sample_entry_frame(
            contract, geometry, seed=17, environment_index=index
        )
        for index in range(512)
    ]
    assert {sample.flap for sample in samples} == {"left", "right"}
    for sample in samples:
        assert 0.005 <= sample.bite_distance_from_wound_m <= 0.009
        assert 0.005 <= sample.stand_off_m <= 0.008
        assert math.isclose(
            math.sqrt(
                sum(value * value for value in sample.desired_tip_direction)
            ),
            1.0,
            abs_tol=1.0e-12,
        )
        assert sample.target_tip_position_m[2] > sample.surface_point_m[2]


def test_t1_is_registered_with_coupled_tissue_and_continuation_task():
    registration = (
        TASK_ROOT / "config/needle/__init__.py"
    ).read_text(encoding="utf-8")
    environment = (
        TASK_ROOT / "config/needle/t1_safe_bite_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "Isaac-Handover-Needle-Safe-Bite-T1-v0" in registration
    assert "Isaac-Handover-Needle-Safe-Bite-Chain-v0" in registration
    assert "make_needle_ready_tissue_cfg(" in environment
    assert "CoupledMJWarpVBDSolverCfg(" in environment
    assert 'coupling_mode=str(vbd["coupling_mode"])' in environment
    assert "mdp.reset_tissue_outer_fixture" in environment
    assert "self.terminations.success = None" in environment
    assert "self.rewards.safe_bite_success.weight = 0.0" in environment


def test_t1_success_and_progress_are_physics_derived():
    state_source = (TASK_ROOT / "mdp/safe_bite.py").read_text(
        encoding="utf-8"
    )
    handover_source = (TASK_ROOT / "mdp/state.py").read_text(
        encoding="utf-8"
    )
    assert 'handover["successful_handover"]' in state_source
    assert 'handover["receiver_contact_now"]' in state_source
    assert "minimum_needle_clearance" in state_source
    assert "receiver_tool_clearance" in state_source
    assert "psm_tool_gripper1_link" in state_source
    assert "psm_tool_gripper2_link" in state_source
    assert "receiver_tool_samples_w" in state_source
    assert "write_nodal_kinematic_target_to_sim_index" in state_source
    assert "tissue_contact & ~state[\"entry_armed\"]" in state_source
    assert "cache[\"reset_attempt_count\"][env_ids] += 1" in state_source
    assert "force_full_chain" in state_source
    assert "state[\"previous_normalized_error\"] - state[\"normalized_error\"]" in (
        state_source
    )
    assert "state[\"entry_armed\"] |=" in state_source
    assert "state[\"premature_tissue_contact\"] |=" in state_source
    assert "state[\"authorized_contact_transition\"] |=" in state_source
    assert "_dr_anmar_pending_safe_bite_restore" in handover_source
    assert "state[\"successful_handover\"][restored] = True" in (
        handover_source
    )


def test_learned_authority_is_receiver_pose_only_after_handover():
    model = (TASK_ROOT / "safe_bite_model.py").read_text(encoding="utf-8")
    end_to_end = (TASK_ROOT / "end_to_end_model.py").read_text(
        encoding="utf-8"
    )
    assert "receiver_pose_rows[7:13] = 1.0" in model
    assert "receiver_role_mask[:, :6]" in model
    assert "safe_receiver_action" in model
    assert "contact_transition_active" in model
    assert "raw[:, 119:122]" in model
    assert "inward_receiver_action" in model
    assert "SAFE_BITE_HANDOVER_COMPLETE" in model
    assert "self.receiver_adaptation_enabled = True" in model
    assert "for parameter in self.phase_network.trunk.parameters()" not in (
        model
    )
    assert "raw[:, 107:]" in end_to_end


def test_benchmark_records_t1_and_accepts_nonterminal_chain_task():
    benchmark = (
        ROOT / "scripts/dr_anmar_learning_benchmark.py"
    ).read_text(encoding="utf-8")
    assert 'success_term = term_values.get(' in benchmark
    assert '"first_episode_safe_bite_diagnostics"' in benchmark
    assert '"ever_authorized_contact_transition"' in benchmark
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
    assert t1["learning"]["handover_snapshot_schedule"] == (
        "per_environment_rotating_quota"
    )
    assert t1["learning"]["full_chain_stride"] == 5
    assert t1["learning"]["post_arm_inward_action_limit"] == 0.03
    assert t1["promotion"]["minimum_full_chain_success_rate"] == 0.8
    assert t1["promotion"]["execution_requires_explicit_user_approval"] is True
