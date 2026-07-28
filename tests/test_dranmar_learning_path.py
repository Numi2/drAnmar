from __future__ import annotations

import ast
import json
import math
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks"


def test_learning_path_manifest_is_ordered_and_branded() -> None:
    manifest = json.loads((ROOT / "config/dranmar_learning_path.json").read_text())
    stages = manifest["stages"]
    assert [stage["stage"] for stage in stages] == list(range(1, 8))
    assert stages[0]["task"] == "DrAnmar-Reach-PSM-IK-Rel-v0"
    assert all(stage["task"].startswith("DrAnmar-") for stage in stages)
    assert manifest["defaults"]["held_out_seeds"]
    assert manifest["defaults"]["num_envs"] == 1200
    assert (
        "target_relative_axis_angle_orientation"
        in manifest["defaults"]["reach_observation_contract"]
    )
    assert manifest["defaults"]["success_source"] == "isaac_lab_termination_manager"
    assert (
        manifest["defaults"]["stage_1_initialization"]["method"]
        == "analytic_relative_ik_base_plus_learned_residual"
    )
    assert (
        manifest["defaults"]["stage_2_initialization"]["method"]
        == "dual_analytic_relative_ik_base_plus_learned_coordination_residual"
    )
    assert (
        "arm_2_target_relative_axis_angle_orientation"
        in manifest["defaults"]["dual_reach_observation_contract"]
    )
    assert manifest["defaults"]["stage_2_initialization"]["dual_loop_gain"] == 0.25
    assert (
        manifest["defaults"]["stage_2_initialization"]["contact_mode"]
        == "disabled_free_space_control_qualification"
    )
    assert (
        manifest["defaults"]["stage_3_initialization"]["method"]
        == "analytic_grasp_lift_base_plus_learned_residual"
    )
    assert manifest["defaults"]["stage_3_initialization"]["approach_height_m"] == 0.02
    assert manifest["defaults"]["stage_3_initialization"]["grasp_height_m"] == 0.0
    assert manifest["defaults"]["stage_3_initialization"]["grasp_offset_m"] == [
        0.0,
        0.0,
        -0.0014,
    ]
    assert (
        manifest["defaults"]["stage_3_initialization"]["grasp_offset_source"]
        == "isaac_lab_parallel_1200_env_first_outcome_contact_sweep"
    )
    assert (
        manifest["defaults"]["stage_3_initialization"]["slow_approach_action_limit"]
        == 0.1
    )
    assert manifest["defaults"]["stage_3_initialization"]["carry_action_limit"] == 0.1
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "carry_lateral_action_limit"
        ]
        == 0.1
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "carry_vertical_action_limit"
        ]
        == 0.18
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "carry_vertical_action_limit_source"
        ]
        == "isaac_lab_parallel_1200_env_full_population_first_outcome_sweep_997_of_1200_successes"
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "carry_target_height_offset_m"
        ]
        == 0.0
    )
    assert (
        manifest["defaults"]["stage_3_initialization"]["gripper_close_rad"]
        == 0.07
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "gripper_effort_limit_nm"
        ]
        == 0.15
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "gripper_effort_qualification"
        ]["total_successful_episodes"]
        == 4379
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "close_distance_to_grasp_m"
        ]
        == 0.005
    )
    assert (
        manifest["defaults"]["stage_3_initialization"]["close_distance_source"]
        == "two_independent_1200_env_shared_distribution_first_outcome_sweeps"
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "lateral_alignment_threshold_m"
        ]
        == 0.005
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "lateral_alignment_source"
        ]
        == "two_independent_1200_env_shared_distribution_first_outcome_sweeps"
    )
    assert (
        manifest["defaults"]["stage_3_initialization"]["residual_phase"]
        == "latched_carry_only"
    )
    assert manifest["defaults"]["stage_3_initialization"]["residual_axes"] == [
        "x",
        "y",
        "z",
    ]
    assert (
        manifest["defaults"]["stage_3_initialization"]["residual_action_limit"]
        == 0.03
    )
    assert (
        manifest["defaults"]["stage_3_initialization"]["residual_initial_std"]
        == 0.01
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "residual_exploration_std_learning"
        ]
        == "fixed"
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "lateral_clearance_below_target_m"
        ]
        == 0.04
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "lateral_clearance_below_target_source"
        ]
        == "isaac_lab_parallel_1200_env_full_population_first_outcome_sweep_1023_of_1200_successes"
    )
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "carry_latch_below_target_m"
        ]
        == 0.062
    )
    stage_3 = stages[2]
    contract = stage_3["qualification_contract"]
    assert contract["initial_object_height_m"] == 0.015
    assert contract["initial_object_height_m"] < contract["minimum_success_height_m"]
    assert contract["minimum_success_height_m"] < contract["target_object_height_m"]
    assert contract["sustained_success_steps"] / contract["control_hz"] == 0.2
    assert contract["requires_orientation_alignment"] is True
    assert contract["requires_stable_angular_motion"] is True
    assert contract["initial_object_quaternion_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert contract["requires_physics_owned_object_motion"] is True
    needle_contract = stages[3]["qualification_contract"]
    assert needle_contract["initial_object_height_m"] == 0.001
    assert needle_contract["grasp_arc_fraction"] == 0.4
    assert (
        needle_contract["grasp_qualification"]["total_successful_episodes"]
        == 2228
    )
    assert needle_contract["grasp_qualification"]["hard_failures"] == 0
    assert needle_contract["initial_object_quaternion_xyzw"] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert needle_contract["requires_goal_pose_alignment"] is False
    assert needle_contract["requires_orientation_alignment"] is False
    assert (
        needle_contract["success_definition"]
        == "native_bilateral_jaw_contact_and_minimum_height_sustained_without_hard_failure"
    )


def test_frontier_imports_and_runner_contract() -> None:
    sources = "\n".join(path.read_text() for path in TASK_ROOT.rglob("*.py"))
    assert "from isaaclab.utils import configclass" not in sources
    assert "AdditiveUniformNoiseCfg" not in sources
    assert "RslRlPpoActorCriticCfg" not in sources
    assert "RslRlMLPModelCfg" in sources
    assert "obs_normalization=True" in sources
    assert "check_for_nan = True" in sources


def test_learning_environments_define_gpu_cloning_and_success() -> None:
    for relative in (
        "surgical/reach/reach_env_cfg.py",
        "surgical/reach_dual/reach_env_cfg.py",
        "surgical/lift/lift_env_cfg.py",
        "surgical/handover/handover_env_cfg.py",
    ):
        source = (TASK_ROOT / relative).read_text()
        assert "clone_in_fabric=True" in source
        assert "success_rate = RewTerm(" in source
        ast.parse(source)


def test_launcher_starts_simulator_before_task_registration() -> None:
    benchmark_path = ROOT / "scripts/dr_anmar_learning_benchmark.py"
    benchmark_source = benchmark_path.read_text()
    tree = ast.parse(benchmark_source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    rendered = ast.unparse(main)
    assert rendered.index("app = AppLauncher(") < rendered.rindex(
        "import orbit.surgical.tasks"
    )
    assert "export_policy_to_jit" in benchmark_source
    assert "export_policy_to_onnx" in benchmark_source
    assert 'termination_manager.get_term("success")' in benchmark_source
    assert "def _reach_teacher_action(" in benchmark_source
    assert "def _lift_teacher_action(" in benchmark_source
    assert "analytic_grasp_lift_base_plus_learned_residual" in benchmark_source
    assert "def _reach_error_offsets(" in benchmark_source
    assert "def _pretrain(" in benchmark_source
    assert "def _probe(" in benchmark_source
    assert '"total_action_dim"' in benchmark_source
    assert "terminated, time_outs" in benchmark_source
    assert "termination_term_counts" in benchmark_source
    assert "initial_procedure_state" in benchmark_source
    assert '"pose_diagnostics": pose_diagnostics' in benchmark_source
    assert '"pose_diagnostic_trace": pose_diagnostic_trace' in benchmark_source


def test_reach_policy_observes_direct_pose_error() -> None:
    cfg_source = (
        TASK_ROOT / "surgical/reach/reach_env_cfg.py"
    ).read_text()
    reward_source = (
        TASK_ROOT / "surgical/reach/mdp/rewards.py"
    ).read_text()
    assert "target_relative_position = ObsTerm(" in cfg_source
    assert "target_relative_orientation = ObsTerm(" in cfg_source
    assert "pose_command_orientation_error_vector" in reward_source
    assert "quat_box_minus(desired_quat_w, current_quat_w)" in reward_source
    residual_source = (
        TASK_ROOT / "surgical/reach/residual_model.py"
    ).read_text()
    assert "class ReachResidualMLPModel(MLPModel):" in residual_source
    assert "self._base_action(obs) + self.residual_scale * residual" in residual_source
    ast.parse(cfg_source)
    ast.parse(reward_source)
    ast.parse(residual_source)


def test_dual_reach_policy_observes_two_pose_errors_and_uses_residual_base() -> None:
    cfg_source = (
        TASK_ROOT / "surgical/reach_dual/reach_env_cfg.py"
    ).read_text()
    model_source = (
        TASK_ROOT / "surgical/reach_dual/residual_model.py"
    ).read_text()
    agent_source = (
        TASK_ROOT / "surgical/reach_dual/config/psm/agents/rsl_rl_cfg.py"
    ).read_text()
    benchmark_source = (
        ROOT / "scripts/dr_anmar_learning_benchmark.py"
    ).read_text()
    assert "target_1_relative_orientation = ObsTerm(" in cfg_source
    assert "target_2_relative_orientation = ObsTerm(" in cfg_source
    assert "class DualReachResidualMLPModel(ReachResidualMLPModel):" in model_source
    assert "dual_reach_residual_actor([256, 128, 64])" in agent_source
    ik_source = (
        TASK_ROOT / "surgical/reach_dual/config/psm/ik_rel_env_cfg.py"
    ).read_text()
    assert ik_source.count(
        "scale=(0.0025, 0.0025, 0.0025, 0.0125, 0.0125, 0.0125)"
    ) == 2
    assert ik_source.count("collision_enabled=False") == 2
    for offset in (46, 49, 52, 55):
        assert f"start={offset}" in benchmark_source
    ast.parse(cfg_source)
    ast.parse(model_source)
    ast.parse(agent_source)


def test_dual_robot_configs_do_not_mutate_shallow_copies() -> None:
    config_paths = (
        *(
            TASK_ROOT / f"surgical/reach_dual/config/{robot}/{control}_env_cfg.py"
            for robot in ("psm", "star")
            for control in ("joint_pos", "ik_abs", "ik_rel")
        ),
        *(
            TASK_ROOT / f"surgical/handover/config/{prop}/{control}_env_cfg.py"
            for prop in ("block", "needle")
            for control in ("joint_pos", "ik_abs", "ik_rel")
        ),
    )
    for path in config_paths:
        source = path.read_text()
        assert ".init_state.pos =" not in source, path
        assert ".init_state.rot =" not in source, path
        assert ".spawn.activate_contact_sensors =" not in source, path
        assert "init_state=" in source, path
        ast.parse(source)


def test_handover_requires_closest_arm_physical_transfer() -> None:
    state_source = (
        TASK_ROOT / "surgical/handover/mdp/state.py"
    ).read_text()
    cfg_source = (
        TASK_ROOT / "surgical/handover/handover_env_cfg.py"
    ).read_text()
    needle_source = (
        TASK_ROOT / "surgical/handover/config/needle/joint_pos_env_cfg.py"
    ).read_text()
    manifest = json.loads((ROOT / "config/dranmar_learning_path.json").read_text())
    contract = manifest["stages"][5]["qualification_contract"]

    assert '"robot_1_jaw_1_object_contact"' in state_source
    assert '"robot_2_jaw_1_object_contact"' in state_source
    assert '"giver_is_robot_1"' in state_source
    assert "robot_1_distance[reset] <= robot_2_distance[reset]" in state_source
    assert "receiver_position_w = torch.where(" in state_source
    assert "giver_identity = ObsTerm(func=mdp.giver_identity)" in cfg_source
    assert "pickup_recovery_context = ObsTerm(" in cfg_source
    assert "func=mdp.pickup_recovery_context" in cfg_source
    assert "func=mdp.role_end_effector_object_distance" in cfg_source
    assert "func=mdp.role_bilateral_grasp" in cfg_source
    assert (
        cfg_source.count(
            "Intermediate physical phases are diagnostics"
        )
        == 1
    )
    phase_progress_block = cfg_source.split(
        "    phase_progress = RewTerm(", 1
    )[1].split("\n", 1)[0]
    assert "weight=0.0" in phase_progress_block
    for reward_name in (
        "giver_reach",
        "giver_grasp",
        "receiver_reach",
        "receiver_grasp",
        "stable_dual_grasp",
    ):
        reward_block = cfg_source.split(
            f"    {reward_name} = RewTerm(", 1
        )[1].split("\n    )", 1)[0]
        assert "weight=0.0" in reward_block
    assert "giver_contact_history" in state_source
    assert "receiver_contact_history" in state_source
    assert "contact_required_steps: int = 3" in state_source
    assert "maximum_pickup_attempts: int = 3" in state_source
    assert "pickup_contact_loss_steps: int = 3" in state_source
    assert "giver_follow_tolerance: float = 0.005" in state_source
    assert "recovery_open_steps: int = 15" in state_source
    assert "recovery_support_clearance: float = 0.005" in state_source
    assert "recovery_linear_speed_limit: float = 0.05" in state_source
    assert "recovery_angular_speed_limit: float = 5.0" in state_source
    assert '"pickup_attempt_count"' in state_source
    assert '"pickup_recovery_count"' in state_source
    assert '"pickup_attempts_exhausted"' in state_source
    assert "phase[recovery_allowed] = 4" in state_source
    assert "phase[recovery_complete] = 0" in state_source
    assert "& ~giver_follows" in state_source
    assert '"progress_phase"' in state_source
    assert "required_receiver_only_steps: int = 10" in state_source
    assert "pickup_clearance: float = 0.01" in state_source
    assert "clearance >= pickup_clearance" in state_source
    assert "reset_height_offset: float = -0.05" in state_source
    assert 'state["support_height_w"]' in state_source
    assert "allowed_receiver_contact_flicker_steps: int = 1" in state_source
    assert "receiver_follows" in state_source
    assert "receiver_contact_now" in state_source
    assert "& ~receiver_contact_now" in state_source
    assert "& receiver_flicker_allowed" in state_source
    assert '"premature_release"' in state_source
    assert "physical_action[:, 6]" in state_source
    assert "physical_action[:, 13]" in state_source
    assert "(giver_open_action > 0.0)" in state_source
    assert "& ~giver_contact_now" in state_source
    assert "& ~state[\"premature_release\"]" in state_source
    assert "(phase == 2) & giver_contact & receiver_contact" in state_source
    assert '"receiver_retention_failed"' in state_source
    assert '"retention_failure_low_clearance"' in state_source
    assert '"retention_failure_follow_error"' in state_source
    assert '"retention_failure_contact_loss"' in state_source
    assert '"last_retention_failure_contact_loss"' in state_source
    assert "receiver_distance < presentation_distance" not in state_source
    assert 'command_name: str = "receiver_pose"' in state_source
    assert "receiver_pose = mdp.UniformPoseCommandCfg(" in cfg_source
    assert 'asset_name="robot_2"' in cfg_source
    assert "self.commands.receiver_pose.body_name" in needle_source
    assert contract["direction"] == "closest_arm_giver_to_other_arm_receiver"
    assert contract["requires_receiver_goal_pose"] is False
    assert contract["minimum_pickup_clearance_m"] == 0.01
    assert contract["contact_window_steps"] == 5
    assert contract["contact_required_steps"] == 3
    assert contract["receiver_contact_flicker_steps"] == 1
    assert (
        manifest["stages"][5]["learning"]["initialization"]
        == "exact_closest_arm_analytic_base_plus_bounded_giver_xy_residual"
    )
    assert (
        manifest["stages"][5]["learning"]["residual_action_limit"]
        == 0.01
    )
    assert (
        manifest["stages"][5]["learning"][
            "analytic_giver_vertical_action_limit"
        ]
        == 0.015
    )
    assert (
        manifest["stages"][5]["learning"][
            "analytic_pickup_vertical_action_limit"
        ]
        == 0.01
    )
    assert (
        manifest["stages"][5]["learning"][
            "analytic_receiver_contact_centering_action_limit"
        ]
        == 0.0025
    )
    assert (
        manifest["stages"][5]["learning"]["residual_initial_std"]
        == 0.01
    )
    assert (
        manifest["stages"][5]["learning"][
            "analytic_pickup_primitive_checkpoint"
        ]
        is None
    )
    assert (
        manifest["stages"][5]["learning"][
            "giver_residual_initialization"
        ]
        == "zero_influence_no_checkpoint_transfer"
    )
    assert (
        manifest["stages"][5]["learning"]["residual_action_limit"]
        == 0.01
    )
    assert manifest["stages"][5]["learning"]["giver_residual_axes"] == [
        "x",
        "y",
    ]
    assert (
        manifest["stages"][5]["learning"]["analytic_vertical_authority"]
        is True
    )
    assert (
        manifest["stages"][5]["learning"]["receiver_residual_enabled"]
        is False
    )
    assert manifest["stages"][5]["learning"]["maximum_pickup_attempts"] == 3
    assert contract["maximum_pickup_attempts"] == 3
    assert contract["first_attempt_and_recovered_success_reported_separately"]
    assert manifest["stages"][5]["learning"]["residual_phases"] == [
        "giver_windowed_contact_pickup_and_transport_xy_before_receiver_contact",
    ]
    reward_contract = manifest["stages"][5]["learning"][
        "reward_contract"
    ]
    assert (
        reward_contract["positive_credit"]
        == "retained_terminal_transfer_only"
    )
    assert reward_contract["intermediate_phase_progress_weight"] == 0.0
    assert reward_contract[
        "continuous_reach_grasp_and_dual_hold_weights"
    ] == 0.0
    assert reward_contract["stalling_credit"] is False
    assert reward_contract["safety_penalties_remain_continuous"] is True
    assert (
        manifest["stages"][5]["learning"][
            "teacher_success_does_not_override_physical_qualification"
        ]
        is True
    )
    promotion = manifest["stages"][5]["promotion"]
    assert promotion["require_complete_first_terminal_population"] is True
    assert promotion["require_matching_analytic_baseline"] is True
    assert (
        promotion["analytic_baseline_retention_comparison"]
        == "strictly_lower_rate_unless_baseline_zero"
    )
    assert promotion["required_policy_contract"] == {
        "policy_residual_scale": 0.01,
        "policy_giver_residual_axes": ["x", "y"],
        "policy_analytic_vertical_authority": True,
        "policy_receiver_residual_enabled": False,
    }
    for prop in ("block", "needle"):
        for control in ("joint_pos", "ik_abs", "ik_rel"):
            robot_cfg_source = (
                TASK_ROOT
                / f"surgical/handover/config/{prop}/{control}_env_cfg.py"
            ).read_text()
            assert "rot=(1.0, 0.0, 0.0, 0.0)" not in robot_cfg_source
            assert robot_cfg_source.count(
                "rot=(0.0, 0.0, 0.0, 1.0)"
            ) >= 2
            assert "pos=(0.2, 0.0, 0.15)" not in robot_cfg_source
            assert "pos=(-0.1, 0.0, 0.15)" in robot_cfg_source
    for source in (state_source, cfg_source, needle_source):
        ast.parse(source)


def test_block_lift_requires_physics_owned_height_and_sustained_contact() -> None:
    cfg_source = (
        TASK_ROOT / "surgical/lift/lift_env_cfg.py"
    ).read_text()
    reward_path = TASK_ROOT / "surgical/lift/mdp/rewards.py"
    termination_path = TASK_ROOT / "surgical/lift/mdp/terminations.py"
    reward_source = reward_path.read_text()
    termination_source = termination_path.read_text()
    block_source = (
        TASK_ROOT / "surgical/lift/config/block/joint_pos_env_cfg.py"
    ).read_text()
    needle_source = (
        TASK_ROOT / "surgical/lift/config/needle/joint_pos_env_cfg.py"
    ).read_text()
    model_source = (
        TASK_ROOT / "surgical/lift/residual_model.py"
    ).read_text()
    agent_source = (
        TASK_ROOT / "surgical/lift/config/block/agents/rsl_rl_cfg.py"
    ).read_text()
    needle_agent_source = (
        TASK_ROOT / "surgical/lift/config/needle/agents/rsl_rl_cfg.py"
    ).read_text()
    handover_agent_source = (
        TASK_ROOT
        / "surgical/handover/config/needle/agents/rsl_rl_cfg.py"
    ).read_text()
    learning_cfg_source = (
        TASK_ROOT / "surgical/learning_cfg.py"
    ).read_text()
    launcher_source = (ROOT / "dr_anmar_learning.sh").read_text()
    benchmark_source = (ROOT / "scripts/dr_anmar_learning_benchmark.py").read_text()

    assert "LIFT_INITIAL_OBJECT_HEIGHT_M = 0.015" in cfg_source
    assert "LIFT_MINIMUM_SUCCESS_HEIGHT_M = 0.06" in cfg_source
    assert "LIFT_TARGET_OBJECT_HEIGHT_M = 0.08" in cfg_source
    assert "LIFT_SUCCESS_DWELL_STEPS = 10" in cfg_source
    assert "pos_z=(-0.07, -0.07)" in cfg_source
    assert cfg_source.count("func=mdp.sustained_lift_success") == 3
    assert "class sustained_lift_success(ManagerTermBase):" in reward_source
    assert "root_pos_w)[:, 2] > minimum_height" in reward_source
    assert "**success_params" not in reward_source
    assert (
        reward_source.count("def successful_lift(")
        + termination_source.count("def successful_lift(")
        == 1
    )
    assert "LIFT_INITIAL_OBJECT_HEIGHT_M" in block_source
    assert "ISAAC_IDENTITY_QUATERNION_XYZW" in block_source
    assert 'orientation_threshold"] = 3.2' not in block_source
    assert "NEEDLE_INITIAL_OBJECT_HEIGHT_M = 0.001" in needle_source
    assert "ISAAC_IDENTITY_QUATERNION_XYZW" in needle_source
    assert "mdp.sustained_pickup_success" in needle_source
    assert "self.rewards.object_goal_tracking.weight = 0.0" in needle_source
    assert "self.rewards.object_goal_orientation.weight = 0.0" in needle_source
    assert "rot=(1, 0, 0, 0)" not in block_source
    assert "rot=(1, 0, 0, 0)" not in needle_source
    assert "class sustained_pickup_success(ManagerTermBase):" in reward_source
    assert "class LiftResidualMLPModel(MLPModel):" in model_source
    assert "carry_mode.unsqueeze(-1)" in model_source
    assert "self.grasp_height = grasp_height" in model_source
    assert "close_distance: float = 0.005" in model_source
    assert "lateral_alignment_threshold: float = 0.005" in model_source
    assert "self.grasp_offset_x = float(grasp_offset[0])" in model_source
    assert "grasp_position[:, 2] += self.grasp_offset_z + self.grasp_height" in model_source
    assert "residual = torch.tanh(self.mlp(latent))" in model_source
    assert "carry_mode.unsqueeze(-1).to(residual.dtype)" in model_source
    assert "torch.zeros_like(residual[:, 3:])" in model_source
    assert 'for parameter_name in ("std_param", "log_std_param")' in model_source
    assert "parameter.requires_grad_(False)" in model_source
    assert "sampled[:, :3]" in model_source
    assert "mean[:, 3:]" in model_source
    assert "self.slow_approach_action_limit = slow_approach_action_limit" in model_source
    assert "self.carry_action_limit = carry_action_limit" in model_source
    assert "self.carry_lateral_action_limit = (" in model_source
    assert "self.carry_vertical_action_limit = (" in model_source
    assert (
        "self.carry_target_height_offset = carry_target_height_offset"
        in model_source
    )
    assert "carry_target[:, 2] += self.carry_target_height_offset" in model_source
    assert "carry_error_action[:, :2].clamp(" in model_source
    assert "carry_error_action[:, 2:].clamp(" in model_source
    assert "object_angular_velocity / self.carry_angular_velocity_scale" not in model_source
    assert "target_position[:, 2] - self.lateral_clearance_below_target" in model_source
    assert "carry_mode = bilateral_contact | lifted_carry" in model_source
    assert "axis_angle_from_quat(object_to_target)" in model_source
    assert "self.carry_orientation_velocity_damping_s" in model_source
    assert "lift_residual_actor([256, 128, 64], initial_std=0.01)" in agent_source
    assert (
        "needle_lift_residual_actor([256, 128, 64], initial_std=0.01)"
        in needle_agent_source
    )
    handover_model_source = (
        TASK_ROOT / "surgical/handover/residual_model.py"
    ).read_text()
    assert "class HandoverResidualMLPModel(MLPModel):" in handover_model_source
    assert "class HandoverAnalyticController(nn.Module):" in handover_model_source
    assert "torch.where(residual_mask, sampled, mean)" in handover_model_source
    assert (
        "(phase == 2)\n            & presentation_stable\n"
        in handover_model_source
    )
    assert "receiver_approach_active = (" in handover_model_source
    assert "receiver_approach_active.unsqueeze(-1)" in handover_model_source
    assert "receiver_orientation_active.unsqueeze(-1)" in handover_model_source
    assert "receiver_residual_enabled = (" in handover_model_source
    assert (
        "& self.receiver_residual_enabled_for_learning"
        in handover_model_source
    )
    assert (
        "giver_residual = torch.zeros_like(giver_action)"
        in handover_model_source
    )
    assert "giver_pre_contact_recovery = (" not in handover_model_source
    assert "giver_pre_lift_centering = (" not in handover_model_source
    assert "giver_pickup_transport_residual = (" in handover_model_source
    assert "(phase >= 1)" in handover_model_source
    assert "& (phase <= 2)" in handover_model_source
    assert (
        "& ~receiver_any_contact"
        in handover_model_source
    )
    assert "giver_pickup_transport_residual" in handover_model_source
    assert "giver_recovery_approach_residual" in handover_model_source
    assert "giver_residual[:, :2]" in handover_model_source
    assert "giver_residual[:, :3]" not in handover_model_source
    assert (
        "giver_channel_output = torch.cat("
        in handover_model_source
    )
    assert handover_model_source.count(
        "network_output[:, 3:5]"
    ) == 2
    assert handover_model_source.count(
        "network_output[:, 10:12]"
    ) == 2
    assert "def configure_giver_adaptation(self)" in handover_model_source
    assert (
        "self.controller.receiver_residual_enabled_for_learning = False"
        in handover_model_source
    )
    assert "giver_row_mask[3:5] = 1.0" in handover_model_source
    assert "giver_row_mask[10:12] = 1.0" in handover_model_source
    assert "giver_transport_active = giver_carry_mode & torch.where(" in handover_model_source
    assert "giver_lift_contact_qualified" in handover_model_source
    assert handover_model_source.count(
        "giver_transport_active.unsqueeze(-1)"
    ) >= 2
    assert handover_model_source.count("~receiver_any_contact") >= 1
    assert ") > 0.5" in handover_model_source
    assert "parameter.requires_grad_(True)" in handover_model_source
    assert (
        "self.receiver_residual_enabled_for_learning = False"
        in handover_model_source
    )
    assert "self.residual_scale = residual_scale" in handover_model_source
    assert "def handover_residual_actor(" in learning_cfg_source
    assert "initial_std: float = 0.01" in learning_cfg_source
    assert (
        "actor = handover_residual_actor("
        in handover_agent_source
    )
    assert "initial_std=0.01" in handover_agent_source
    assert "probe)" in launcher_source
    assert "controller-sweep)" in launcher_source
    assert "handover-sweep)" in launcher_source
    assert "DR_ANMAR_HANDOVER_SWEEP_PARAMETER" in launcher_source
    assert "DR_ANMAR_INIT_CHECKPOINT" in launcher_source
    assert "DR_ANMAR_POLICY_LEARNING_RATE" in launcher_source
    assert "DR_ANMAR_POLICY_RESIDUAL_SCALE" in launcher_source
    assert "DR_ANMAR_HANDOVER_GIVER_ADAPTATION" in launcher_source
    assert (
        "DR_ANMAR_POLICY_PICKUP_VERTICAL_ACTION_LIMIT"
        in launcher_source
    )
    assert (
        "DR_ANMAR_POLICY_PICKUP_INITIAL_VERTICAL_ACTION_LIMIT"
        in launcher_source
    )
    assert (
        "DR_ANMAR_POLICY_RECOVERY_PICKUP_VERTICAL_ACTION_LIMIT"
        in launcher_source
    )
    assert (
        "DR_ANMAR_POLICY_CARRY_LATERAL_ACTION_LIMIT"
        in launcher_source
    )
    assert (
        "DR_ANMAR_POLICY_CARRY_LATERAL_RAMP_HEIGHT"
        in launcher_source
    )
    assert (
        "DR_ANMAR_POLICY_PRESENTATION_FRACTION_FROM_GIVER"
        in launcher_source
    )
    assert (
        "DR_ANMAR_POLICY_PRESENTATION_HEIGHT_IN_ROBOT_FRAME"
        in launcher_source
    )
    assert "DR_ANMAR_POLICY_GIVER_CLOSE_DISTANCE" in launcher_source
    assert (
        "DR_ANMAR_POLICY_GIVER_LIFT_ON_LIVE_CONTACT"
        in launcher_source
    )
    assert "DR_ANMAR_SUCCESS_THRESHOLD" in launcher_source
    assert launcher_source.count('--values="${values}"') == 2
    assert "record)" in launcher_source
    assert "def _controller_sweep(" in benchmark_source
    assert "def _handover_controller_sweep(" in benchmark_source
    assert '"receiver_retention_failure_causes"' in benchmark_source
    assert '"first_episode_handover_diagnostics"' in benchmark_source
    assert '"pickup_causal_diagnostics"' in benchmark_source
    assert '"transport_retention_diagnostics"' in benchmark_source
    assert '"timeouts_after_any_midair_contact_loss"' in benchmark_source
    assert '"maximum_midair_giver_contact_loss_steps"' in benchmark_source
    assert '"maximum_giver_bilateral_contact_steps"' in benchmark_source
    assert '"mean_maximum_clearance_m"' in benchmark_source
    assert '"giver_orientation_at_first_window"' in benchmark_source
    assert '"giver_grasp_error_at_first_window_m"' in benchmark_source
    assert '"giver_jaw_aperture_at_first_window_rad"' in benchmark_source
    assert (
        '"minimum_giver_contact_force_at_first_lift_n"'
        in benchmark_source
    )
    assert (
        '"maximum_giver_contact_force_at_first_lift_n"'
        in benchmark_source
    )
    assert (
        '"giver_contact_force_imbalance_at_first_lift_n"'
        in benchmark_source
    )
    assert '"giver_jaw_aperture_at_first_lift_rad"' in benchmark_source
    assert (
        '"object_linear_speed_at_first_lift_m_s"'
        in benchmark_source
    )
    assert (
        '"object_angular_speed_at_first_lift_rad_s"'
        in benchmark_source
    )
    assert '"maximum_phase_distribution"' in benchmark_source
    assert '"lifted_without_receiver_acquisition"' in benchmark_source
    assert "def _handover_teacher_action(" in benchmark_source
    assert (
        '"exact_closest_arm_handover_base_plus_bounded_residual"'
        in benchmark_source
    )
    assert "teacher_warmup_then_linear_dagger_mixture" in benchmark_source
    assert "predicted_actions.detach()" in benchmark_source
    assert '"student_controlled_frames"' in benchmark_source
    assert (
        'if "Handover-Needle-Dual-PSM-IK-Rel" in task:'
        in benchmark_source
    )
    assert (
        "runner.load(str(initial_checkpoint), load_cfg=load_cfg)"
        in benchmark_source
    )
    assert 'train.add_argument("--checkpoint")' in benchmark_source
    assert 'train.add_argument("--learning_rate", type=float)' in benchmark_source
    assert '"--handover_giver_adaptation"' in benchmark_source
    assert '"optimizer_state_reset": True' in benchmark_source
    assert 'play.add_argument("--residual_scale", type=float)' in benchmark_source
    assert (
        'play.add_argument("--pickup_vertical_action_limit", type=float)'
        in benchmark_source
    )
    assert (
        'play.add_argument("--pickup_initial_vertical_action_limit", type=float)'
        in benchmark_source
    )
    assert (
        'play.add_argument("--recovery_pickup_vertical_action_limit", type=float)'
        in benchmark_source
    )
    assert (
        'play.add_argument("--carry_lateral_action_limit", type=float)'
        in benchmark_source
    )
    assert (
        'play.add_argument("--carry_lateral_ramp_height", type=float)'
        in benchmark_source
    )
    assert (
        'play.add_argument("--presentation_fraction_from_giver", type=float)'
        in benchmark_source
    )
    assert (
        'play.add_argument("--presentation_height_in_robot_frame", type=float)'
        in benchmark_source
    )
    assert (
        'play.add_argument("--giver_close_distance", type=float)'
        in benchmark_source
    )
    assert (
        'play.add_argument("--giver_lift_contact_force_threshold", type=float)'
        in benchmark_source
    )
    assert (
        'play.add_argument("--giver_pre_lift_min_contact_jaws", type=int)'
        in benchmark_source
    )
    assert (
        'play.add_argument("--giver_transport_orientation_action_limit", type=float)'
        in benchmark_source
    )
    assert (
        'action=argparse.BooleanOptionalAction'
        in benchmark_source
    )
    assert '"policy_learning_rate"' in benchmark_source
    assert '"policy_residual_scale"' in benchmark_source
    assert '"policy_pickup_vertical_action_limit"' in benchmark_source
    assert (
        '"policy_pickup_initial_vertical_action_limit"'
        in benchmark_source
    )
    assert (
        '"policy_recovery_pickup_vertical_action_limit"'
        in benchmark_source
    )
    assert '"policy_carry_lateral_action_limit"' in benchmark_source
    assert '"policy_carry_lateral_ramp_height"' in benchmark_source
    assert (
        '"policy_presentation_fraction_from_giver"'
        in benchmark_source
    )
    assert '"full_handover_evaluation_success_unchanged": True' in (
        benchmark_source
    )
    assert (
        '"policy_presentation_height_in_robot_frame"'
        in benchmark_source
    )
    assert '"policy_giver_close_distance_m"' in benchmark_source
    assert '"policy_giver_lift_contact_force_threshold_n"' in benchmark_source
    assert '"policy_giver_pre_lift_min_contact_jaws"' in benchmark_source
    assert (
        '"policy_giver_transport_orientation_action_limit"'
        in benchmark_source
    )
    assert '"policy_giver_lift_on_live_contact"' in benchmark_source
    assert '"first_attempt_successful_episodes"' in benchmark_source
    assert '"recovered_successful_episodes"' in benchmark_source
    assert '"pickup_attempt_histogram"' in benchmark_source
    assert '"checkpoint": None' in benchmark_source
    assert 'play.add_argument("--analytic-only", action="store_true")' in benchmark_source
    assert '"analytic_only": bool(args.analytic_only)' in benchmark_source
    assert 'parameter_group["lr"] = args.learning_rate' in benchmark_source
    assert (
        "return max(1, self._step_count // self.num_steps_per_env)"
        in benchmark_source
    )
    assert '"receiver_grasp_z_offset"' in benchmark_source
    assert '"giver_arc_fraction"' in benchmark_source
    assert '"giver_grasp_offsets_m"' in benchmark_source
    assert '"receiver_roll_offset_rad"' in benchmark_source
    assert '"presentation_fraction_from_giver"' in benchmark_source
    assert '"pickup_vertical_action_limit"' in benchmark_source
    assert "pickup_vertical_action_limits = values" in benchmark_source
    assert '"carry_lateral_action_limit"' in benchmark_source
    assert "carry_lateral_action_limits = values" in benchmark_source
    assert '"carry_vertical_action_limit"' in benchmark_source
    assert "carry_vertical_action_limits = values" in benchmark_source
    assert '"receiver_close_distance"' in benchmark_source
    assert "receiver_close_distances = values" in benchmark_source
    assert '"giver_contact_recovery_action_limit"' in benchmark_source
    assert "giver_contact_recovery_action_limits = values" in benchmark_source
    assert (
        '"receiver_contact_centering_action_limit"'
        in benchmark_source
    )
    assert (
        "receiver_contact_centering_action_limits = values"
        in benchmark_source
    )
    assert '"giver_transport_min_contact_jaws"' in benchmark_source
    assert (
        "giver_transport_min_contact_jaws = ["
        in benchmark_source
    )
    assert (
        '"giver_transport_normalized_contact_threshold"'
        in benchmark_source
    )
    assert (
        "giver_transport_normalized_contact_thresholds = values"
        in benchmark_source
    )
    assert "selected_receiver_z_offset = -0.003" in benchmark_source
    assert '"rule": "minimum_reset_tool_tip_to_needle_distance"' in benchmark_source
    assert '"robot_1_selected_as_giver"' in benchmark_source
    assert '"robot_2_selected_as_giver"' in benchmark_source
    assert "receiver_target_orientation = quat_mul(" in benchmark_source
    assert "receiver_orientation_action_limit: float = 0.6" in benchmark_source
    assert "receiver_close_distance: float = 0.001" in benchmark_source
    assert (
        "receiver_contact_centering_action_limit: float = 0.0025"
        in benchmark_source
    )
    assert "receiver_roll_offsets = [math.pi] * len(values)" in benchmark_source
    assert "giver_bilateral_contact = torch.all(" in benchmark_source
    assert "giver_any_contact = torch.any(" in benchmark_source
    assert "giver_target_orientation[:, 3] = 1.0" in benchmark_source
    assert "giver_orientation_action" in benchmark_source
    assert "receiver_any_contact = torch.any(" in benchmark_source
    assert "receiver_bilateral_contact = torch.all(" in benchmark_source
    assert "& receiver_any_contact" in benchmark_source
    assert "giver_transport_active.unsqueeze(-1)" in benchmark_source
    assert "giver_carry_mode = (phase >= 1) & (phase <= 2)" in benchmark_source
    assert (
        "giver_transport_active = (\n"
        "        giver_carry_mode\n"
        "        & (giver_contact_count >= giver_transport_min_contact_jaws)"
        in benchmark_source
    )
    assert "presentation_fraction_from_giver: float = 0.35" in benchmark_source
    assert "presentation_height_in_robot_frame: float = -0.13" in benchmark_source
    assert "presentation_ready_tolerance: float = 0.005" in benchmark_source
    assert '"presentation_ready_tolerance_m": 0.005' in benchmark_source
    assert "carry_lateral_action_limit: float = 0.06" in benchmark_source
    assert "pickup_vertical_action_limit: float = 0.015" in benchmark_source
    assert "carry_vertical_action_limit: float = 0.015" in benchmark_source
    assert (
        "giver_contact_recovery_action_limit: float = 1.0"
        in benchmark_source
    )
    assert "giver_contact_recovery = giver_approach.clamp(" in benchmark_source
    assert (
        "self.carry_vertical_action_limit = 0.015"
        in handover_model_source
    )
    assert "self.carry_lateral_ramp_height = 0.01" in handover_model_source
    assert "self.giver_lift_on_live_contact = True" in handover_model_source
    assert (
        "self.giver_lift_contact_force_threshold_n = 0.01"
        in handover_model_source
    )
    assert "self.giver_pre_lift_min_contact_jaws = 2" in handover_model_source
    assert (
        "self.giver_transport_orientation_action_limit = 0.035"
        in handover_model_source
    )
    assert (
        "self.giver_pregrasp_orientation_action_limit = 0.6"
        in handover_model_source
    )
    assert (
        "self.giver_pregrasp_orientation_tolerance = 0.035"
        in handover_model_source
    )
    assert "giver_pregrasp_orientation_ready = (" in handover_model_source
    assert "& giver_pregrasp_orientation_ready" in handover_model_source
    assert "rotated_grasp_offset" in handover_model_source
    assert "object_pose_in_giver[:, 3:7]" in handover_model_source
    assert "giver_pregrasp_offset" in handover_model_source
    assert "yaw_sine = 2.0 * (" in handover_model_source
    assert "yaw_cosine = 1.0 - 2.0 * (" in handover_model_source
    assert "object_relative_grasp_offset[:, 1]" in handover_model_source
    assert "pickup_recovery_context = raw[:, 98] > 0.5" in handover_model_source
    assert "pickup_recovery_context.unsqueeze(-1)" in handover_model_source
    assert (
        "giver_tool_target_orientation = identity_tool_orientation"
        in handover_model_source
    )
    assert (
        "carry_ramp_fraction = carry_ramp_fraction * carry_ramp_fraction"
        in handover_model_source
    )
    assert (
        "self.pickup_vertical_action_limit = 0.01"
        in handover_model_source
    )
    assert (
        "self.pickup_initial_vertical_action_limit = 0.01"
        in handover_model_source
    )
    assert (
        "self.recovery_pickup_vertical_action_limit = 0.01"
        in handover_model_source
    )
    assert "pickup_recovery_context," in handover_model_source
    assert "pickup_deceleration_fraction = (" in handover_model_source
    assert (
        "self.receiver_contact_centering_action_limit = 0.005"
        in handover_model_source
    )
    assert (
        '"giver_carry_starts_after_contact_window": True'
        in benchmark_source
    )
    assert "receiver_approach_active = (" in benchmark_source
    assert benchmark_source.count(
        "receiver_approach_active.unsqueeze(-1)"
    ) >= 2
    assert '"giver_move_into_receiver_range"' in benchmark_source
    assert "receiver_wait = torch.zeros_like(receiver_approach)" in benchmark_source
    assert '"receiver_waits_for_presentation": True' in benchmark_source
    assert (
        "& giver_bilateral_contact\n"
        "            & receiver_any_contact"
        in benchmark_source
    )
    assert (
        '"receiver_stops_approach_on_first_contact": True'
        in benchmark_source
    )
    assert "(phase == 3)" in benchmark_source
    assert "& ~receiver_bilateral_contact" in benchmark_source
    assert (
        '"giver_release_waits_for_current_receiver_bilateral": True'
        in benchmark_source
    )
    assert (
        '"giver_release_uses_time_only_settle": False'
        in benchmark_source
    )
    assert "((phase == 3) & giver_any_contact)" in benchmark_source
    assert '"giver_holds_position_until_release": True' in benchmark_source
    assert (
        '"receiver_holds_position_after_acquisition": True'
        in benchmark_source
    )
    assert "receiver_hold_target" not in benchmark_source
    assert (
        '"receiver_orientation_frozen_after_acquisition": True'
        in benchmark_source
    )
    assert (
        '"release_requires_open_command_and_contact_loss": True'
        in benchmark_source
    )
    assert (
        "receiver_orientation_action = torch.where(\n"
        "        receiver_approach_active.unsqueeze(-1)"
        in benchmark_source
    )
    assert (
        '"camera_mode": "focused_environment_neighborhood_oblique"'
        in benchmark_source
    )
    assert 'env_cfg.viewer.origin_type = "world"' in benchmark_source
    assert "env_cfg.viewer.eye = camera_eye" in benchmark_source
    assert "env_cfg.viewer.lookat = camera_target" in benchmark_source
    assert "camera_focus = env.unwrapped.scene.env_origins[" in benchmark_source
    assert "camera_visual_span = min(grid_span, 5.0)" in benchmark_source
    assert '"travel_fraction_of_grid_span": 0.30' in benchmark_source
    assert "flow_eased = 0.5 - 0.5 * math.cos(" in benchmark_source
    assert 'play.add_argument("--video", action="store_true")' in benchmark_source
    assert 'play.add_argument("--video_chunk_length", type=int)' in benchmark_source
    assert 'enable_cameras=bool(getattr(args, "video", False))' in benchmark_source
    assert '"single_environment_episode_trace"' in benchmark_source
    assert '"terminal_frame_inclusive"' in benchmark_source
    assert "--video_chunk_length" in launcher_source
    assert "first_terminal_outcome_per_environment" in benchmark_source
    assert '"all_episode_totals"' in benchmark_source
    assert '"first_episode_lift_diagnostics"' in benchmark_source
    assert '"no_bilateral_contact"' in benchmark_source
    assert '"contact_without_minimum_height"' in benchmark_source
    assert '"minimum_height_without_goal_position"' in benchmark_source
    assert '"ever_midair_bilateral_contact_loss"' in benchmark_source
    assert '"maximum_midair_bilateral_contact_loss_steps"' in benchmark_source
    assert '"at_least_10_steps"' in benchmark_source
    assert '"retention_diagnostics"' in benchmark_source
    assert '"gripper_close_rad"' in benchmark_source
    assert '"gripper_effort_limit_nm"' in benchmark_source
    assert '"environment_level_parameter"' in benchmark_source
    assert '"needle_grasp_arc_fraction"' in benchmark_source
    assert '"needle_grasp_z_offset"' in benchmark_source
    assert '"carry_orientation_action_limit"' in benchmark_source
    assert '"carry_orientation_velocity_damping_s"' in benchmark_source
    assert '"carry_goal_action_limit"' in benchmark_source
    assert '"slow_approach_action_limit"' in benchmark_source
    assert '"grasp_offset_m"' in benchmark_source
    assert '"grasp_frame_source"' in benchmark_source
    assert '"hard_failure_term_counts"' in benchmark_source
    assert '"goal_orientation_inside_frame_rate"' in benchmark_source
    assert '"maximum_non_object_force_n"' in benchmark_source
    assert '"maximum_object_height_m"' in benchmark_source
    assert '"maximum_state_clearance_m"' in benchmark_source
    assert '"environments_with_state_lifted"' in benchmark_source
    assert '"minimum_receiver_distance_m"' in benchmark_source
    assert '"maximum_giver_bilateral_contact_n"' in benchmark_source
    assert '"maximum_receiver_bilateral_contact_n"' in benchmark_source
    assert '"maximum_four_jaw_overlap_contact_n"' in benchmark_source
    assert '"minimum_receiver_grasp_distance_m"' in benchmark_source
    assert '"maximum_receiver_jaw_1_contact_n"' in benchmark_source
    assert '"maximum_receiver_jaw_2_contact_n"' in benchmark_source
    assert '"environments_with_receiver_bilateral_contact"' in benchmark_source
    assert '"environments_with_four_jaw_overlap_contact"' in benchmark_source
    assert '"successful_environment_indices"' in benchmark_source
    assert 'handover_sweep.add_argument("--video", action="store_true")' in benchmark_source
    assert "env.unwrapped.sim.set_camera_view(" in benchmark_source
    assert "DR_ANMAR_HANDOVER_VIDEO_ENV_INDEX" in launcher_source
    assert '"goal_position_without_qualified_state"' in benchmark_source
    assert '"qualified_state_without_sustained_dwell"' in benchmark_source
    assert '"success_by_initial_target_xy_distance"' in benchmark_source
    assert "first_dones = was_first_unresolved & dones.bool()" in benchmark_source
    assert "first_successes = first_dones & successes.bool()" in benchmark_source
    for source in (
        cfg_source,
        reward_source,
        termination_source,
        block_source,
        needle_source,
        model_source,
        agent_source,
        needle_agent_source,
    ):
        ast.parse(source)


def test_needle_geometry_grasp_offsets_follow_composed_arc() -> None:
    scope = runpy.run_path(str(TASK_ROOT / "surgical/lift/grasp_frames.py"))
    grasp_offset = scope["needle_geometry_grasp_offset_m"]
    grasp_frame = scope["needle_geometry_grasp_frame"]

    blunt_side = grasp_offset(0.0)
    one_third = grasp_offset(1.0 / 3.0)
    sharp_side = grasp_offset(1.0)

    assert math.isclose(blunt_side[0], 0.019154, abs_tol=1e-6)
    assert math.isclose(blunt_side[1], 0.018817, abs_tol=1e-6)
    assert math.isclose(one_third[0], 0.0023884, abs_tol=1e-6)
    assert math.isclose(one_third[1], 0.0092818, abs_tol=1e-6)
    assert math.isclose(sharp_side[0], 0.019147, abs_tol=1e-6)
    assert math.isclose(sharp_side[1], -0.019548, abs_tol=1e-6)
    assert math.isclose(one_third[2], -0.00101704, abs_tol=1e-9)
    giver_frame = grasp_frame(0.4, grasp_z_m=0.0006)
    receiver_frame = grasp_frame(0.65, grasp_z_m=-0.003)
    expected_giver = (
        0.0007375535249017802,
        0.005600696415109648,
        0.0006,
    )
    expected_receiver = (
        0.0019002163218475414,
        -0.009119058578501121,
        -0.003,
    )
    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(giver_frame[0], expected_giver, strict=True)
    )
    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(
            receiver_frame[0], expected_receiver, strict=True
        )
    )
    assert math.isclose(
        receiver_frame[1] - giver_frame[1],
        0.790114043036337,
        abs_tol=1e-12,
    )


def test_launcher_fits_parallel_worlds_to_live_ram_and_vram() -> None:
    scope = runpy.run_path(str(ROOT / "scripts/dr_anmar_learning_benchmark.py"))
    fit = scope["_fit_num_envs_to_memory"]
    assert fit(512, 12_803, 11_613) == 256
    assert fit(512, 17_000, 30_000) == 512
    assert fit(1_024, 24_000, 8_500) == 128
