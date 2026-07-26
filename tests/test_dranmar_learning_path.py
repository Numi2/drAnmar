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
    assert [stage["stage"] for stage in stages] == list(range(1, 7))
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


def test_handover_requires_arm_1_to_arm_2_physical_transfer() -> None:
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
    assert 'receiver_frame: FrameTransformer = env.scene["ee_2_frame"]' in state_source
    assert "receiver_only_consecutive" in state_source
    assert "required_receiver_only_steps: int = 10" in state_source
    assert "object_pos_w[:, 2] > minimum_height" in state_source
    assert 'command_name: str = "receiver_pose"' in state_source
    assert "receiver_pose = mdp.UniformPoseCommandCfg(" in cfg_source
    assert 'asset_name="robot_2"' in cfg_source
    assert "self.commands.receiver_pose.body_name" in needle_source
    assert contract["direction"] == "robot_1_giver_to_robot_2_receiver"
    assert contract["requires_receiver_goal_pose"] is False
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
    assert "probe)" in launcher_source
    assert "controller-sweep)" in launcher_source
    assert "handover-sweep)" in launcher_source
    assert "record)" in launcher_source
    assert "def _controller_sweep(" in benchmark_source
    assert "def _handover_controller_sweep(" in benchmark_source
    assert "def _handover_teacher_action(" in benchmark_source
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


def test_launcher_fits_parallel_worlds_to_live_ram_and_vram() -> None:
    scope = runpy.run_path(str(ROOT / "scripts/dr_anmar_learning_benchmark.py"))
    fit = scope["_fit_num_envs_to_memory"]
    assert fit(512, 12_803, 11_613) == 256
    assert fit(512, 17_000, 30_000) == 512
    assert fit(1_024, 24_000, 8_500) == 128
