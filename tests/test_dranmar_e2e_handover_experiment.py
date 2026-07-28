from __future__ import annotations

import ast
import hashlib
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = (
    ROOT
    / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks/surgical"
)


def test_e2e_handover_experiment_is_isolated_and_physics_owned() -> None:
    contract = json.loads(
        (
            ROOT
            / "config/experiments/dranmar_e2e_handover_example.json"
        ).read_text()
    )
    assert contract["status"] == "isolated_research_example_not_stage_qualified"
    assert contract["schema_version"] == "dranmar-e2e-handover-experiment-1.9"
    assert contract["architecture"]["actor"].startswith("frozen_pickup_lift")
    assert contract["architecture"]["learned_authority"].startswith(
        "receiver_xyz"
    )
    assert contract["architecture"]["active_residual_exploration_std"] == 0.005
    assert contract["architecture"]["residual_action_limit"] == 0.01
    baseline = contract["known_good_baseline"]
    baseline_path = ROOT / baseline["evidence_path"]
    assert baseline["retained_handover_success"] == 14
    assert baseline["receiver_acquisition"] == 47
    assert baseline["unsafe_force_or_drop_failures"] == 0
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == (
        baseline["evidence_sha256"]
    )
    baseline_evidence = json.loads(baseline_path.read_text())
    assert baseline_evidence["successful_episodes"] == 14
    assert baseline_evidence["success_rate"] == 0.21875
    rejected = contract["rejected_candidates"][0]
    rejected_path = ROOT / rejected["evidence_path"]
    assert rejected["retained_handover_success"] == 4
    assert hashlib.sha256(rejected_path.read_bytes()).hexdigest() == (
        rejected["evidence_sha256"]
    )
    rejected_by_label = {
        candidate["label"]: candidate
        for candidate in contract["rejected_candidates"]
    }
    giver_xy = rejected_by_label["e2e-giver-xy-v25-model50"]
    assert giver_xy["screen_600_success_rate"] == 0.595
    assert giver_xy["scale_2000_success_rate"] == 0.445
    pickup_15 = rejected_by_label["e2e-pickup15-v26"]
    assert pickup_15["giver_contact_without_10mm_lift"] == 94
    receiver_model_125 = rejected_by_label["e2e-receiver-ppo-v33-model125"]
    assert receiver_model_125["retained_handover_success"] == 378
    assert receiver_model_125["hard_safety_events"] == 3
    for candidate in (giver_xy, pickup_15, receiver_model_125):
        for path_key, hash_key in (
            ("evidence_path", "evidence_sha256"),
            ("screen_600_evidence_path", "screen_600_evidence_sha256"),
            ("scale_2000_evidence_path", "scale_2000_evidence_sha256"),
        ):
            if path_key not in candidate:
                continue
            path = ROOT / candidate[path_key]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == (
                candidate[hash_key]
            )
    calibration = contract["receiver_depth_calibration"]
    calibration_path = ROOT / calibration["evidence_path"]
    assert calibration["selected_receiver_grasp_z_offset_m"] == -0.003
    assert calibration["selected_screen_success_rate"] == 0.385
    assert hashlib.sha256(calibration_path.read_bytes()).hexdigest() == (
        calibration["evidence_sha256"]
    )
    matched_path = ROOT / calibration["matched_checkpoint_evidence_path"]
    assert calibration["matched_checkpoint_retained_success"] == 18
    assert hashlib.sha256(matched_path.read_bytes()).hexdigest() == (
        calibration["matched_checkpoint_evidence_sha256"]
    )
    promotion_path = ROOT / calibration["promotion_evidence_path"]
    scale_path = ROOT / calibration["scale_2000_evidence_path"]
    assert calibration["status"] == "promoted_development_training_baseline"
    assert calibration["scale_2000_successes"] == 946
    assert hashlib.sha256(promotion_path.read_bytes()).hexdigest() == (
        calibration["promotion_evidence_sha256"]
    )
    assert hashlib.sha256(scale_path.read_bytes()).hexdigest() == (
        calibration["scale_2000_evidence_sha256"]
    )
    ppo = contract["ppo_fine_tuning"]
    ppo_training_path = ROOT / ppo["training_evidence_path"]
    ppo_promotion_path = ROOT / ppo["promotion_evidence_path"]
    ppo_screen_path = ROOT / ppo["screen_600_evidence_path"]
    ppo_scale_path = ROOT / ppo["scale_2000_evidence_path"]
    assert ppo["status"] == "promoted_development_training_baseline"
    assert ppo["checkpoint_sha256"] == (
        "5d79cd7e767bdadf2cd83fc9fac472717424dff5ac6dda8b422f2ce26453c5ae"
    )
    assert ppo["screen_600_successes"] == 349
    assert ppo["scale_2000_successes"] == 939
    assert ppo["scale_2000_receiver_retention_losses"] == 47
    for path, expected_hash in (
        (ppo_training_path, ppo["training_evidence_sha256"]),
        (ppo_promotion_path, ppo["promotion_evidence_sha256"]),
        (ppo_screen_path, ppo["screen_600_evidence_sha256"]),
        (ppo_scale_path, ppo["scale_2000_evidence_sha256"]),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    hysteresis = contract["pickup_contact_hysteresis"]
    hysteresis_screen = ROOT / hysteresis["screen_600_evidence_path"]
    hysteresis_scale = ROOT / hysteresis["scale_2000_evidence_path"]
    assert hysteresis["status"] == (
        "promoted_development_controller_baseline_not_stage_qualified"
    )
    assert hysteresis["screen_600_successes"] == 367
    assert hysteresis["scale_2000_successes"] == 1017
    assert hysteresis["scale_2000_giver_contact_without_10mm_lift"] == 175
    assert hysteresis["scale_2000_recovered_successes"] == 67
    assert hysteresis["scale_2000_hard_safety_events"] == 18
    assert hashlib.sha256(hysteresis_screen.read_bytes()).hexdigest() == (
        hysteresis["screen_600_evidence_sha256"]
    )
    assert hashlib.sha256(hysteresis_scale.read_bytes()).hexdigest() == (
        hysteresis["scale_2000_evidence_sha256"]
    )
    latency = contract["receiver_latency_diagnostics"]
    latency_path = ROOT / latency["evidence_path"]
    assert latency["retained_handover_success"] == 367
    assert latency["stable_presentations_without_receiver_contact"] == 76
    assert latency["successful_stable_to_receiver_contact_steps_p50"] == 98.0
    assert hashlib.sha256(latency_path.read_bytes()).hexdigest() == (
        latency["evidence_sha256"]
    )
    receiver_ppo = contract["receiver_ppo_fine_tuning_v33"]
    receiver_training = ROOT / receiver_ppo["training_evidence_path"]
    receiver_screen = ROOT / receiver_ppo["screen_600_evidence_path"]
    receiver_scale = ROOT / receiver_ppo["scale_2000_evidence_path"]
    assert receiver_ppo["checkpoint_sha256"] == (
        "a56f61703855931d9d755beeba530bb8c1ac5232f6d6499e6254399cca8535cf"
    )
    assert receiver_ppo["screen_600_successes"] == 377
    assert receiver_ppo["screen_600_receiver_retention_losses"] == 21
    assert receiver_ppo["scale_2000_successes"] == 1021
    assert receiver_ppo["scale_2000_hard_safety_events"] == 8
    assert receiver_ppo["scale_2000_receiver_retention_losses"] == 45
    for path, expected_hash in (
        (receiver_training, receiver_ppo["training_evidence_sha256"]),
        (receiver_screen, receiver_ppo["screen_600_evidence_sha256"]),
        (receiver_scale, receiver_ppo["scale_2000_evidence_sha256"]),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    contact_centering = contract["receiver_contact_centering_v55"]
    contact_centering_path = ROOT / contact_centering["evidence_path"]
    assert contact_centering["screen_512_successes"] == 325
    assert contact_centering["screen_600_successes"] == 381
    assert contact_centering["screen_600_receiver_contacts"] == 414
    assert contact_centering["screen_600_windowed_bilateral_capture"] == 402
    assert contact_centering["screen_600_retention_losses"] == 18
    assert contact_centering["screen_600_pickup_attempts_exhausted"] == 21
    assert contact_centering["screen_600_hard_safety_events"] == 2
    assert hashlib.sha256(contact_centering_path.read_bytes()).hexdigest() == (
        contact_centering["evidence_sha256"]
    )
    recovery_option = contract["pickup_recovery_option_v4"]
    recovery_option_path = ROOT / recovery_option["evidence_path"]
    assert recovery_option["matched_512_candidate_successes"] == 356
    assert recovery_option["matched_1200_candidate_successes"] == 719
    assert (
        recovery_option["matched_1200_candidate_recovered_successes"] == 66
    )
    assert recovery_option["multiseed_candidate_successes"] == 2197
    assert recovery_option["multiseed_recovered_successes_candidate"] == 243
    assert (
        recovery_option["multiseed_pickup_attempts_exhausted_candidate"]
        == 211
    )
    assert recovery_option["multiseed_seed_deltas"] == [16, 10, 8]
    assert (
        hashlib.sha256(recovery_option_path.read_bytes()).hexdigest()
        == recovery_option["evidence_sha256"]
    )
    recovery_receiver = contract["recovery_receiver_option_v6"]
    recovery_receiver_path = ROOT / recovery_receiver["evidence_path"]
    assert recovery_receiver["v6_baseline_screen_successes"] == 187
    assert recovery_receiver["v6_model_10_screen_successes"] == 187
    assert (
        recovery_receiver["absolute_yaw_multiseed_baseline_successes"]
        == 2197
    )
    assert (
        recovery_receiver["absolute_yaw_multiseed_candidate_successes"]
        == 2148
    )
    assert hashlib.sha256(recovery_receiver_path.read_bytes()).hexdigest() == (
        recovery_receiver["evidence_sha256"]
    )
    joint_transfer = contract["joint_transfer_acquisition_v8"]
    assert joint_transfer["capture_stage"] == "lifted_custody"
    assert joint_transfer["optimizer_schedule"] == "fixed"
    assert joint_transfer["zero_initialized_joint_adapter"] is True
    assert joint_transfer["frozen_promoted_pickup_recovery_policy"] is True
    assert "giver_release" in joint_transfer["analytic_authority"]
    assert joint_transfer["qualification_boundary"].startswith(
        "no_performance_claim"
    )
    refinement = contract["transfer_refinement_v9"]
    assert refinement["capture_stage"] == "stable_presentation"
    assert refinement["rollout_steps_per_env"] == 128
    assert refinement["zero_initialized_refinement_adapter"] is True
    assert refinement["learned_authority"]["phase_2_giver"] == "none"
    assert refinement["learned_authority"]["phase_3_giver"] == "none"
    assert refinement["learned_authority"]["phase_3_receiver"] == "se3"
    assert (
        refinement["promotion_contract"]["minimum_aggregate_improvement"]
        == 0.005
    )
    assert refinement["promotion_contract"]["preregistered_seeds"] == [
        104729,
        130363,
        155921,
    ]
    deadline_recovery = contract["deadline_recovery_option_v10"]
    assert deadline_recovery["zero_impact_adapter"] is True
    assert deadline_recovery["frozen_incumbent_policy"] is True
    assert deadline_recovery["rollout_steps_per_env"] == 128
    assert deadline_recovery["options"] == [
        "continue",
        "reseat",
        "backoff",
    ]
    assert "original_episode_deadline" in (
        deadline_recovery["source_states"]
    )
    assert deadline_recovery["status"].startswith(
        "rejected_discrete_switching"
    )
    continuous_recovery = contract["deadline_recovery_residual_v11"]
    assert continuous_recovery["zero_impact_adapter"] is True
    assert continuous_recovery["frozen_incumbent_policy"] is True
    assert continuous_recovery["rollout_steps_per_env"] == 128
    assert continuous_recovery["control"] == (
        "incumbent_plus_bounded_continuous_receiver_se3_residual"
    )
    assert continuous_recovery["discrete_trajectory_switches"] == []
    transport_recovery = contract["recovered_transport_preposition_v12"]
    assert transport_recovery["zero_impact_adapter"] is True
    assert transport_recovery["original_episode_deadline"] is True
    assert transport_recovery["activation"] == (
        "recovered_lifted_custody_through_presentation"
    )
    two_stage_recovery = contract["two_stage_recovered_handover_v13"]
    assert two_stage_recovery["zero_impact_adapter"] is True
    assert two_stage_recovery["rollout_steps_per_env"] == 384
    assert two_stage_recovery["stage_1"].startswith("bounded_giver_se3")
    assert two_stage_recovery["stage_2"].startswith("bounded_receiver_se3")
    assert two_stage_recovery["status"].startswith("rejected_")
    transport_controller = contract["recovered_transport_controller_v14"]
    assert transport_controller["paired_scale_result"]["candidate_successes"] == 1148
    assert (
        transport_controller["paired_scale_result"][
            "candidate_recovered_successes"
        ]
        == 117
    )
    assert (
        transport_controller["paired_scale_result"][
            "protected_surface_rate_increase"
        ]
        <= transport_controller["paired_scale_result"][
            "maximum_protected_surface_rate_increase"
        ]
    )
    assert contract["anti_reward_hacking"]["analytic_actions_at_inference"] is True
    assert contract["anti_reward_hacking"]["phase_progress_weight"] == 1.0
    assert contract["anti_reward_hacking"]["retained_success_weight"] == 80.0
    assert (
        contract["anti_reward_hacking"]["terminal_transfer_failure_weight"]
        == -80.0
    )
    assert (
        contract["anti_reward_hacking"][
            "unsuccessful_timeout_is_terminal_failure"
        ]
        is True
    )
    assert contract["anti_reward_hacking"]["object_attachment_or_teleportation"] is False
    assert contract["anti_reward_hacking"]["success_source"].startswith(
        "unchanged_isaac_lab_physics"
    )
    assert contract["launch"]["num_envs"] == 2400
    assert contract["launch"]["dagger_teacher_only_warmup_updates"] == 1000
    assert contract["launch"]["dagger_student_segment_steps"] == 64
    assert contract["launch"]["phase_replay_capacity_per_phase"] == 65536
    assert contract["launch"]["phase_balanced_consolidation_updates"] == 2000
    assert contract["launch"]["minimum_deterministic_success_before_ppo"] == 0.25
    assert contract["anti_reward_hacking"]["receiver_retry_steps"] == 15
    assert (
        contract["anti_reward_hacking"]["receiver_approach_timeout_steps"]
        == 0
    )
    assert contract["anti_reward_hacking"]["pickup_contact_loss_steps"] == 3
    assert "fixed_pose_presentation_stable_for_8_steps" in (
        contract["anti_reward_hacking"]["required_sequence"]
    )


def test_e2e_actor_role_normalizes_observations_and_actions() -> None:
    model_path = TASK_ROOT / "handover/end_to_end_model.py"
    source = model_path.read_text()
    controller_source = (
        TASK_ROOT / "handover/residual_model.py"
    ).read_text()
    assert "class EndToEndHandoverMLPModel(MLPModel):" in source
    assert "role_normalize_handover_observation(raw)" in source
    assert "role_action_to_physical(" in source
    assert "learned_role_residual" in source
    assert "def select_handover_role(" in source
    assert "\n    def select(" not in source
    assert "self.phase_network(latent, phase)" in source
    assert "class _PhaseHeadedNetwork(nn.Module):" in source
    assert "class _RecoveryReceiverAdapter(nn.Module):" in source
    assert "class _JointTransferAcquisitionAdapter(nn.Module):" in source
    assert "class _TransferRefinementAdapter(nn.Module):" in source
    assert "class _DeadlineRecoveryAdapter(nn.Module):" in source
    assert "handover_task_features(" in source
    assert "receiver_policy_grasp_offset" in source
    assert "recovery_receiver_canonical_grasp_features" in source
    assert "joint_transfer_acquisition_features" in source
    assert "nn.init.zeros_(self.output.weight)" in source
    assert "nn.init.zeros_(self.output.bias)" in source
    assert "self.recovery_receiver_adapter(" in source
    assert "adapter_role_residual[:, 7:13]" in source
    assert "joint_role_residual[:, 0:6]" in source
    assert "joint_role_residual[:, 7:13]" in source
    assert "joint_physical_action_mask" in source
    assert "refinement_physical_action_mask" in source
    assert "parameter.requires_grad_(True)" in source
    assert "quat_apply(" in source
    assert "HandoverAnalyticController" in source
    assert "self.residual_scale" in source
    assert "* physical_residual" in source
    assert "physical_action_mask = receiver_residual_mask" in source
    assert "or self.pickup_recovery_adaptation_enabled" in source
    assert "giver_residual_mask | receiver_residual_mask" in source
    assert "exploration_mask = giver_residual_mask" in source
    assert "def configure_giver_adaptation(self)" in source
    assert "def configure_pickup_recovery_adaptation(self)" in source
    assert "for phase_index in (0, 1, 2, 4):" in source
    assert "giver_role_row_mask[0:2] = 1.0" in source
    assert "def configure_receiver_adaptation(self)" in source
    assert "receiver_xyz_row_mask[7:10] = 1.0" in source
    assert "self.phase_network.heads[2]" in source
    assert "and not self.receiver_adaptation_enabled" in source
    assert "self.controller.receiver_residual_enabled_for_learning = True" in source
    assert "configure_receiver_grasp_retain_adaptation" in source
    assert (
        "configure_recovery_receiver_grasp_retain_adaptation"
        in source
    )
    assert (
        "configure_joint_transfer_acquisition_adaptation"
        in source
    )
    assert "configure_transfer_refinement_adaptation" in source
    assert "configure_deadline_recovery_adaptation" in source
    assert "deadline_recovery_features" in source
    assert "deadline_option_selection" in source
    assert "reseat_role_action" not in source
    assert "backoff_role_action" not in source
    assert "Always retaining the incumbent action" in source
    assert "deadline_recovery_residual_scale" in source
    assert "last_deadline_option_index" in source
    assert source.count(
        "deadline_active = (\n"
        "            pickup_recovery_context\n"
        "            & (phase == 2)\n"
        "        )"
    ) == 2
    assert source.count(
        "deadline_giver_active = "
        "deadline_active & ~presentation_qualified"
    ) == 2
    assert source.count(
        "deadline_receiver_active = "
        "deadline_active & presentation_qualified"
    ) == 2
    assert "presentation_qualified = raw[:, 103] >= 1.0" in source
    assert "presentation_qualified = obs[:, 103] >= 1.0" in source
    assert "refinement_giver_active = torch.zeros_like(" in source
    assert "receiver_se3_row_mask[7:13] = 1.0" in source
    assert (
        "self.recovery_receiver_grasp_retain_adaptation_enabled"
        in source
    )
    assert "recovery_receiver_reference_network" in source
    assert "reference_role_residual" in source
    assert "pickup_recovery_context.unsqueeze(-1)" in source
    assert "exploration_mask = receiver_residual_mask" in source
    assert "presentation_stable = raw[:, 103]" in controller_source
    assert "receiver_retry_active = raw[:, 105]" in controller_source
    assert "(phase == 3) & ~receiver_bilateral_contact" in controller_source
    assert "receiver_retry_translation" in controller_source
    assert "self.presentation_hold_action_limit = 0.01" in controller_source
    assert "self.receiver_grasp_z = -0.003" in controller_source
    assert "_RECEIVER_ARC_FRACTION = 0.65" in controller_source
    assert "_RECEIVER_TANGENT_DELTA_RAD" in controller_source
    assert "self.receiver_tangent_delta_rad" in controller_source
    assert "self.receiver_crossing_angle_rad" in controller_source
    assert "self.receiver_roll_offset_rad" in controller_source
    assert "receiver_half_roll_offset" in controller_source
    assert "transport_custody_latch_enabled = True" in controller_source
    assert "receiver_preposition_enabled = True" in controller_source
    assert "recovery_receiver_preposition_height" in controller_source
    assert "receiver_contact_orientation_error_target_rad = 1.95" in controller_source
    assert "receiver_adaptive_arc_enabled = False" in controller_source
    assert "phase_two_custody" in controller_source
    assert "receiver_preposition_active" in controller_source
    assert "receiver_grasp_retain_residual_enabled" in controller_source
    assert "giver_presentation_hold = giver_carry.clamp(" in controller_source
    assert "giver_pre_lift_transport_ready = (" in controller_source
    assert "giver_recovery_residual_only_for_learning" in controller_source
    assert "giver_recovery_approach_residual" in controller_source
    assert "recovery_carry_lateral_action_limit" in controller_source
    assert "self.recovery_carry_lateral_action_limit = 0.08" in controller_source
    assert "recovery_transport_qualified = (" in controller_source
    assert "& (phase >= 2)" in controller_source
    assert "& giver_bilateral_contact" in controller_source
    assert (
        "self.recovery_receiver_giver_barrier_activation_distance = 0.018"
        in controller_source
    )
    assert (
        "self.recovery_receiver_giver_minimum_tip_distance = 0.012"
        in controller_source
    )
    assert "recovery_receiver_barrier_active = (" in controller_source
    assert "receiver_barrier_correction" in controller_source
    assert "(phase == 1) | giver_pre_lift_contact" in controller_source
    assert "pickup_contact_loss_steps debounce" in controller_source
    recovery_cfg_source = (
        TASK_ROOT / "handover/config/needle/e2e_ik_rel_env_cfg.py"
    ).read_text()
    assert "def recovery_stable_presentation(env):" in recovery_cfg_source
    assert 'state["giver_custody"]' in recovery_cfg_source
    assert 'state["lifted"]' in recovery_cfg_source
    assert (
        '"recovered_physics_owned_stable_presentation"'
        in recovery_cfg_source
    )
    assert "class PhaseMaskedGaussianDistribution" in source
    assert "object_state_manipulation" not in source
    assert "raw[:, 99:101]" in source
    assert "raw[:, 101:103]" in source
    assert "raw[:, 103:107]" in source
    assert "grasp_z_m=-0.003" in source
    assert "receiver_offset += receiver_policy_grasp_offset" in source
    ast.parse(source)

    benchmark_source = (
        ROOT / "scripts/dr_anmar_learning_benchmark.py"
    ).read_text()
    assert '"first_stable_presentation_frame"' in benchmark_source
    assert '"first_receiver_contact_after_stable_frame"' in benchmark_source
    assert '"receiver_approach_by_maximum_phase"' in benchmark_source
    assert '"receiver_grasp_error_at_first_stable_m"' in benchmark_source
    assert (
        '"receiver_orientation_error_at_first_contact_rad"'
        in benchmark_source
    )
    assert '"object_yaw_in_receiver_at_first_stable_rad"' in benchmark_source
    assert '"object_yaw_in_receiver_at_first_contact_rad"' in benchmark_source
    assert '"outcomes_by_giver_role"' in benchmark_source
    assert '"failure_by_maximum_phase"' in benchmark_source
    assert '"protected_surface_attribution"' in benchmark_source
    assert '"episodes_crossing_limit_by_sensor"' in benchmark_source
    assert '"episodes_crossing_limit_by_tool"' in benchmark_source
    assert (
        '"terminal_protected_surface_force_by_sensor_n"'
        in benchmark_source
    )
    termination_source = (
        TASK_ROOT / "handover/mdp/terminations.py"
    ).read_text()
    assert (
        "_dr_anmar_terminal_protected_surface_forces_n"
        in termination_source
    )
    assert '"terminal_pickup_attempt_histogram"' in benchmark_source
    assert "--receiver_crossing_angle_rad" in benchmark_source
    assert "--receiver_grasp_retain_residual" in benchmark_source
    assert (
        "--recovery_receiver_grasp_retain_adaptation"
        in benchmark_source
    )
    assert (
        "receiver_grasp_retain_residual_enabled_for_learning"
        in benchmark_source
    )
    assert '"optimizer": False' in benchmark_source
    assert '"iteration": False' in benchmark_source


def test_e2e_task_adds_native_contact_history_without_changing_success() -> None:
    observation_source = (
        TASK_ROOT
        / "handover/config/needle/e2e_observations.py"
    ).read_text()
    environment_source = (
        TASK_ROOT
        / "handover/config/needle/e2e_ik_rel_env_cfg.py"
    ).read_text()
    registration_source = (
        TASK_ROOT / "handover/config/needle/__init__.py"
    ).read_text()
    agent_source = (
        TASK_ROOT
        / "handover/config/needle/agents/rsl_rl_e2e_cfg.py"
    ).read_text()
    assert "force_matrix_w_history" in observation_source
    assert "history_index: int = 1" in observation_source
    assert "def transfer_contract_state(" in observation_source
    assert "receiver_retry_active" in observation_source
    assert 'state["giver_release_authorized"].float()' in observation_source
    assert "previous_jaw_contacts = ObsTerm(" in environment_source
    assert "transfer_contract = ObsTerm(" in environment_source
    assert '"presentation_stability_steps": 8' in environment_source
    assert '"receiver_capture_required_steps": 1' in environment_source
    assert '"giver_release_confirmation_steps": 1' in environment_source
    assert '"receiver_attempt_timeout_steps": 30' in environment_source
    assert '"receiver_retry_contact_loss_steps": 8' in environment_source
    assert '"receiver_retry_steps": 15' in environment_source
    assert "self.rewards.phase_progress.weight = 1.0" in environment_source
    assert "self.rewards.success.weight = 80.0" in environment_source
    assert "terminal_transfer_failure" in environment_source
    assert "unsuccessful_timeout" in environment_source
    assert "mdp.time_out(env)" in environment_source
    assert "NeedleHandoverEnvCfg" in environment_source
    assert "NeedleHandoverReceiverCurriculumEnvCfg" in environment_source
    assert "dr_anmar_receiver_curriculum = True" in environment_source
    assert (
        "dr_anmar_receiver_curriculum_restore_probability = 0.8"
        in environment_source
    )
    assert (
        "dr_anmar_receiver_curriculum_cross_environment_sampling = True"
        in environment_source
    )
    assert "NeedleHandoverReceiverGraspRetainEnvCfg" in environment_source
    assert (
        "NeedleHandoverPickupRecoveryCurriculumEnvCfg"
        in environment_source
    )
    assert (
        "dr_anmar_pickup_recovery_curriculum_restore_probability = 0.98"
        in environment_source
    )
    assert (
        "NeedleHandoverRecoveryReceiverGraspRetainEnvCfg"
        in environment_source
    )
    assert (
        "NeedleHandoverJointTransferAcquisitionEnvCfg"
        in environment_source
    )
    assert "NeedleHandoverTransferRefinementEnvCfg" in environment_source
    assert (
        "dr_anmar_receiver_curriculum_require_pickup_recovery = True"
        in environment_source
    )
    assert (
        '"retained_handover_from_recovered_stable_presentation"'
        in environment_source
    )
    assert (
        '"retained_handover_from_physics_owned_lifted_custody"'
        in environment_source
    )
    assert (
        'dr_anmar_receiver_curriculum_capture_stage = "lifted_custody"'
        in environment_source
    )
    assert (
        "dr_anmar_transfer_refinement_rollout_steps_per_env = 128"
        in environment_source
    )
    assert '"presentation_use_filtered_custody": True' in environment_source
    assert "reset_receiver_curriculum_from_cache" in environment_source
    assert "TerminationsCfg" not in environment_source
    assert "RewardsCfg" not in environment_source
    assert "Isaac-Handover-Needle-Dual-PSM-IK-Rel-Structured-v0" in registration_source
    assert (
        "DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-Structured-v0"
        in registration_source
    )
    assert "Isaac-Handover-Needle-Receiver-Curriculum-v0" in (
        registration_source
    )
    assert "DrAnmar-Handover-Needle-Pickup-Recovery-v0" in (
        registration_source
    )
    assert (
        "DrAnmar-Handover-Needle-Recovery-Receiver-Grasp-Retain-v0"
        in registration_source
    )
    assert (
        "DrAnmar-Handover-Needle-Joint-Transfer-Acquisition-v0"
        in registration_source
    )
    assert "DrAnmar-Handover-Needle-Transfer-Refinement-v0" in (
        registration_source
    )
    assert "HandoverNeedleEndToEndPPORunnerCfg" in agent_source
    assert "init_std=0.005" in agent_source
    assert "clip_param=0.05" in agent_source
    assert "desired_kl=0.002" in agent_source
    assert "entropy_coef=0.0" in agent_source
    for source in (
        observation_source,
        environment_source,
        registration_source,
        agent_source,
    ):
        ast.parse(source)
    state_source = (
        TASK_ROOT / "handover/mdp/state.py"
    ).read_text()
    benchmark_source = (
        ROOT / "scripts/dr_anmar_learning_benchmark.py"
    ).read_text()
    assert "def reset_receiver_curriculum_from_cache(" in state_source
    assert (
        "def reset_pickup_recovery_curriculum_from_cache("
        in state_source
    )
    assert '"_dr_anmar_receiver_curriculum_cache"' in state_source
    assert "capture = capture_ready & ~cache" in state_source
    assert 'cache["reset_restores"] +=' in state_source
    assert '"cross_environment_restores"' in state_source
    assert (
        '"dr_anmar_receiver_curriculum_require_pickup_recovery"'
        in state_source
    )
    assert 'capture &= state["pickup_recovery_count"] > 0' in state_source
    assert '"recovery_conditioned_captures"' in state_source
    assert "_RECEIVER_CURRICULUM_STATE_FIELDS" in state_source
    assert '"_dr_anmar_receiver_curriculum_restored"' in state_source
    assert '"restored_source_env_ids"' in state_source
    assert '"markov_state_restores"' in state_source
    assert '"recovery_context_restores"' in state_source
    assert '"last_action"' in state_source
    assert 'capture_stage == "lifted_custody"' in state_source
    assert "presentation_custody" in state_source
    assert '"receiver_approach_step_count"' in state_source
    assert "receiver_approach_stalled" in state_source
    assert "receiver_approach_timeout_steps" in state_source
    assert '"receiver_approach_timeout_steps": 0' in environment_source
    assert "giver_and_deadline_context" in environment_source
    assert "NeedleHandoverDeadlineRecoveryOptionEnvCfg" in (
        environment_source
    )
    assert "NeedleHandoverDeadlineRecoveryResidualEnvCfg" in (
        environment_source
    )
    assert "giver_presentation_then_receiver_" in environment_source
    assert '"restored_episode_length_buf"' in state_source
    assert (
        'env.episode_length_buf[target_env_ids] = receiver_cache['
        in state_source
    )
    assert '"receiver_curriculum_cached_envs"' in benchmark_source
    assert '"receiver_curriculum_reset_restores"' in benchmark_source
    assert '"receiver_curriculum_reset_refreshes"' in benchmark_source
    assert '"receiver_curriculum_restore_probability"' in benchmark_source
    assert "agent_cfg.save_interval = 1" in benchmark_source
    assert 'agent_cfg.algorithm.schedule = "fixed"' in benchmark_source
    assert '"optimizer_learning_rates_final"' in benchmark_source
    assert '"receiver_curriculum_checkpoint_interval"' in benchmark_source
    assert '"receiver_curriculum_adaptation_contract"' in benchmark_source
    assert '"pickup_lift_presentation_policy_frozen": True' in benchmark_source
    assert '"optimizer_state_reset": True' in benchmark_source
    assert "policy_model.configure_receiver_adaptation()" in benchmark_source
    assert "configure_pickup_recovery_adaptation" in benchmark_source
    assert (
        "configure_recovery_receiver_grasp_retain_adaptation"
        in benchmark_source
    )
    assert (
        "configure_joint_transfer_acquisition_adaptation"
        in benchmark_source
    )
    assert "configure_transfer_refinement_adaptation" in benchmark_source
    assert '"transfer_refinement_adaptation_contract"' in benchmark_source
    assert '"policy_runtime_contract_sha256"' in benchmark_source
    assert '"environment_runtime_contract_sha256"' in benchmark_source
    assert "configure_deadline_recovery_adaptation" in benchmark_source
    assert '"deadline_recovery_adaptation_contract"' in benchmark_source
    assert '"deadline_recovery_controller"' in benchmark_source
    assert "deadline-recovery curriculum controller does not " in benchmark_source
    assert '"discrete_trajectory_switches": []' in benchmark_source
    assert "recovered_lifted_custody_with_" in benchmark_source
    assert '"learned_giver_axes_before_stable_presentation"' in (
        benchmark_source
    )
    assert '"deadline_option_step_counts"' in benchmark_source
    assert '"joint_transfer_acquisition_adaptation_contract"' in (
        benchmark_source
    )
    assert '"initial_state_population_sha256"' in benchmark_source
    assert '"tracked_patch_sha256"' in benchmark_source
    assert (
        '"pickup_recovery_policy_frozen_and_active"'
        in benchmark_source
    )
    assert '"pickup_recovery_curriculum_adaptation_contract"' in (
        benchmark_source
    )
    assert '"first_attempt_residual_frozen": True' in benchmark_source


def test_structured_transfer_state_delays_release_and_retries_receiver() -> None:
    state_source = (
        TASK_ROOT / "handover/mdp/state.py"
    ).read_text()
    assert 'getattr(env.cfg, "dr_anmar_handover_contract", None)' in (
        state_source
    )
    assert "presentation_stable_consecutive" in state_source
    assert 'state["presentation_qualified"] |=' in state_source
    assert "receiver_capture_consecutive" in state_source
    assert "receiver_capture_required_steps" in state_source
    assert "giver_release_confirmation_consecutive" in state_source
    assert "giver_release_authorized" in state_source
    assert "release_confirmation_active" in state_source
    assert "& receiver_contact_now" in state_source
    assert "receiver_attempt_timeout_steps" in state_source
    assert "receiver_attempt_stalled" in state_source
    assert "receiver_capture_follows" in state_source
    assert "receiver_retry_step_count" in state_source
    assert "receiver_release_aborted" in state_source
    assert "phase[receiver_release_aborted] = 2" in state_source
    assert "receiver_retention_failed" in state_source
    ast.parse(state_source)


def test_experiment_launcher_keeps_teacher_until_full_trajectories_exist() -> None:
    launcher_source = (ROOT / "dr_anmar_learning.sh").read_text()
    example_source = (
        ROOT / "examples/dranmar_e2e_handover_experiment.sh"
    ).read_text()
    assert "DR_ANMAR_DAGGER_WARMUP_UPDATES" in launcher_source
    assert "DR_ANMAR_DAGGER_MIN_TEACHER_FRACTION" in launcher_source
    assert "DR_ANMAR_E2E_WARMUP_UPDATES:-1000" in example_source
    assert "DR_ANMAR_E2E_MINIMUM_BC_SUCCESS:-0.25" in example_source
    assert "PPO was not started." in example_source
    assert "DR_ANMAR_E2E_STUDENT_SEGMENT_STEPS" in launcher_source
    assert "DR_ANMAR_E2E_REPLAY_CAPACITY_PER_PHASE" in launcher_source
    assert "DR_ANMAR_E2E_CONSOLIDATION_UPDATES" in launcher_source
    assert "dr_anmar_handover_promotion.py" in example_source
    assert "PPO checkpoint rejected" in example_source
    assert launcher_source.count("DR_ANMAR_HANDOVER_GIVER_ADAPTATION") == 2
    assert "--handover_giver_adaptation" in launcher_source
    assert "DR_ANMAR_POLICY_RECEIVER_GRASP_RETAIN_RESIDUAL" in launcher_source
    assert "DR_ANMAR_PICKUP_RECOVERY_ADAPTATION" in launcher_source
    assert "--pickup_recovery_adaptation" in launcher_source
    assert "DR_ANMAR_TRANSFER_REFINEMENT_ADAPTATION" in launcher_source
    assert "--transfer_refinement_adaptation" in launcher_source


def test_promotion_gate_rejects_deterministic_success_and_safety_regression() -> None:
    promotion = runpy.run_path(
        str(ROOT / "scripts/dr_anmar_handover_promotion.py")
    )
    result = promotion["select_checkpoint"](
        ROOT / "docs/experiments/evidence/controller-600-seed17.json",
        [
            ROOT
            / "docs/experiments/evidence/v3-ppo25-600-seed17.json"
        ],
        minimum_success_improvement=0.02,
        maximum_safety_rate_increase=0.0,
    )
    assert result["decision"] == "baseline_retained"
    comparison = result["candidates"][0]
    assert comparison["success_improvement"] < 0.0
    assert comparison["success_gate_passed"] is False
    assert comparison["safety_gate_passed"] is False
    assert "receiver_retention_lost" in comparison["safety_regressions"]
