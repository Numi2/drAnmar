from __future__ import annotations

import ast
import json
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
    assert (
        manifest["defaults"]["stage_3_initialization"]["slow_approach_action_limit"]
        == 0.1
    )
    assert manifest["defaults"]["stage_3_initialization"]["carry_action_limit"] == 0.1
    assert (
        manifest["defaults"]["stage_3_initialization"][
            "lateral_clearance_below_target_m"
        ]
        == 0.02
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
    assert needle_contract["initial_object_quaternion_xyzw"] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert needle_contract["requires_orientation_alignment"] is True


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
    launcher_source = (ROOT / "dr_anmar_learning.sh").read_text()

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
    assert "rot=(1, 0, 0, 0)" not in block_source
    assert "rot=(1, 0, 0, 0)" not in needle_source
    assert "class LiftResidualMLPModel(MLPModel):" in model_source
    assert "bilateral_contact.unsqueeze(-1)" in model_source
    assert "self.grasp_height = grasp_height" in model_source
    assert "self.slow_approach_action_limit = slow_approach_action_limit" in model_source
    assert "self.carry_action_limit = carry_action_limit" in model_source
    assert "object_angular_velocity / self.carry_angular_velocity_scale" not in model_source
    assert "target_position[:, 2] - self.lateral_clearance_below_target" in model_source
    assert "lift_residual_actor([256, 128, 64])" in agent_source
    assert "probe)" in launcher_source
    for source in (
        cfg_source,
        reward_source,
        termination_source,
        block_source,
        needle_source,
        model_source,
        agent_source,
    ):
        ast.parse(source)


def test_launcher_fits_parallel_worlds_to_live_ram_and_vram() -> None:
    scope = runpy.run_path(str(ROOT / "scripts/dr_anmar_learning_benchmark.py"))
    fit = scope["_fit_num_envs_to_memory"]
    assert fit(512, 12_803, 11_613) == 256
    assert fit(512, 17_000, 30_000) == 512
    assert fit(1_024, 24_000, 8_500) == 128
