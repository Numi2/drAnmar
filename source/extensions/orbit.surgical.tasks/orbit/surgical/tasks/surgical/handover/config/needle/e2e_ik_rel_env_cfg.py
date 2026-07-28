# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Isolated end-to-end needle-handover environment configuration."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from ... import mdp
from . import e2e_observations, ik_rel_env_cfg


_JAW_SENSOR_NAMES = (
    "robot_1_jaw_1_object_contact",
    "robot_1_jaw_2_object_contact",
    "robot_2_jaw_1_object_contact",
    "robot_2_jaw_2_object_contact",
)

def _non_success_terminal(env):
    """Return the union of every active termination term except success."""
    failure = env.termination_manager.dones & False
    for term_name in env.termination_manager.active_terms:
        if term_name != "success":
            failure |= env.termination_manager.get_term(term_name)
    return failure


def terminal_transfer_success(env):
    """Reward retained transfer only when no failure fires on that step."""
    return (
        env.termination_manager.get_term("success")
        & ~_non_success_terminal(env)
    ).float()


def terminal_transfer_failure(env):
    """Penalize every actual non-success terminal reported by Isaac Lab.

    The termination manager is computed before the reward manager on each
    environment step. Reading its per-term results keeps the reward fail-closed
    as termination terms evolve and makes failure dominate if success and a
    safety violation occur simultaneously.
    """
    return _non_success_terminal(env).float()


def recovery_stable_presentation(env):
    """Qualify the recovery option at a stable, physically held presentation."""
    state = mdp.handover_state(env)
    return (
        (state["pickup_recovery_count"] > 0)
        & state["presentation_stable"]
        & state["giver_custody"]
        & state["lifted"]
    )


def recovery_stable_presentation_reward(env):
    """Reward only the physical state that terminates the recovery option."""
    return recovery_stable_presentation(env).float()


@configclass
class NeedleHandoverEndToEndEnvCfg(ik_rel_env_cfg.NeedleHandoverEnvCfg):
    """Keep the qualified MDP and append one preceding contact-force frame."""

    def __post_init__(self):
        super().__post_init__()
        # This stricter contract is opt-in for the isolated structured task.
        # The standard handover environments retain their existing phase MDP.
        self.dr_anmar_handover_contract = {
            "presentation_fraction_from_giver": 0.35,
            "presentation_height_in_robot_frame": -0.13,
            "presentation_ready_tolerance": 0.005,
            "presentation_stability_steps": 8,
            "presentation_use_filtered_custody": True,
            "presentation_linear_speed_limit": 0.05,
            "presentation_angular_speed_limit": 5.0,
            "receiver_capture_required_steps": 1,
            "receiver_capture_follow_tolerance": 0.005,
            "receiver_capture_linear_speed_limit": 0.05,
            "receiver_capture_angular_speed_limit": 5.0,
            "giver_release_confirmation_steps": 1,
            "receiver_attempt_timeout_steps": 30,
            # Fixed no-contact retries were rejected at both 75 and 150
            # counts. The retained controller retries only after physical
            # contact loss; complete misses are handled by the isolated,
            # custody- and deadline-aware learned recovery option.
            "receiver_approach_timeout_steps": 0,
            "receiver_retry_contact_loss_steps": 8,
            "receiver_retry_steps": 15,
        }
        self.observations.policy.previous_jaw_contacts = ObsTerm(
            func=e2e_observations.previous_jaw_contact_forces,
            params={
                "sensor_names": _JAW_SENSOR_NAMES,
                "history_index": 1,
                "scale": 0.2,
            },
            clip=(0.0, 5.0),
        )
        self.observations.policy.transfer_contract = ObsTerm(
            func=e2e_observations.transfer_contract_state,
            clip=(0.0, 1.0),
        )
        # The structured actor begins from a competent controller, so terminal
        # retained transfer can dominate the objective. A small one-time phase
        # signal preserves credit assignment without paying early lift or
        # transient receiver contact more than a failed episode costs.
        self.rewards.phase_progress.weight = 1.0
        self.rewards.success.func = terminal_transfer_success
        self.rewards.success.weight = 80.0
        self.rewards.terminal_transfer_failure = RewTerm(
            func=terminal_transfer_failure,
            weight=-80.0,
        )


@configclass
class NeedleHandoverEndToEndEnvCfg_PLAY(NeedleHandoverEndToEndEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class NeedleHandoverDeadlineContextEnvCfg(
    NeedleHandoverEndToEndEnvCfg_PLAY
):
    """Expose deadline context without changing incumbent policy behavior."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.giver_identity = ObsTerm(
            func=e2e_observations.giver_and_deadline_context,
            clip=(0.0, 1.0),
        )


@configclass
class NeedleHandoverReceiverCurriculumEnvCfg(
    NeedleHandoverEndToEndEnvCfg
):
    """Train receiver acquisition from cached physical presentations."""

    def __post_init__(self):
        super().__post_init__()
        self.dr_anmar_receiver_curriculum = True
        self.dr_anmar_receiver_curriculum_restore_probability = 0.8
        self.dr_anmar_receiver_curriculum_cross_environment_sampling = True
        self.events.receiver_curriculum_reset = EventTerm(
            func=mdp.reset_receiver_curriculum_from_cache,
            mode="reset",
        )


@configclass
class NeedleHandoverReceiverGraspRetainEnvCfg(
    NeedleHandoverReceiverCurriculumEnvCfg
):
    """Adapt receiver approach and seating without relearning giver custody."""

    def __post_init__(self):
        super().__post_init__()
        self.dr_anmar_receiver_grasp_retain_curriculum = True
        self.dr_anmar_handover_contract["receiver_capture_required_steps"] = 3
        self.dr_anmar_handover_contract["giver_release_confirmation_steps"] = 3


@configclass
class NeedleHandoverPickupRecoveryCurriculumEnvCfg(
    NeedleHandoverEndToEndEnvCfg
):
    """Train bounded relift corrections from simulator-observed pickup slips."""

    def __post_init__(self):
        super().__post_init__()
        self.dr_anmar_pickup_recovery_curriculum = True
        # Once a simulator-observed slip has been cached, spend almost every
        # reset on the recovery option. The remaining two percent still
        # refresh the cache from an end-to-end physical rollout.
        self.dr_anmar_pickup_recovery_curriculum_restore_probability = 0.98
        self.dr_anmar_pickup_recovery_curriculum_cross_environment_sampling = (
            True
        )
        self.dr_anmar_pickup_recovery_objective = (
            "recovered_physics_owned_stable_presentation"
        )
        self.dr_anmar_pickup_recovery_controller = {
            "recovery_carry_lateral_action_limit": 0.08,
            "recovery_receiver_preposition_height": 0.025,
        }
        self.events.pickup_recovery_curriculum_reset = EventTerm(
            func=mdp.reset_pickup_recovery_curriculum_from_cache,
            mode="reset",
        )
        self.rewards.success = RewTerm(
            func=recovery_stable_presentation_reward,
            weight=80.0,
        )
        self.terminations.success = DoneTerm(
            func=recovery_stable_presentation,
        )


@configclass
class NeedleHandoverRecoveryReceiverGraspRetainEnvCfg(
    NeedleHandoverReceiverGraspRetainEnvCfg
):
    """Train retained transfer only from real post-recovery presentations."""

    def __post_init__(self):
        super().__post_init__()
        # Do not dilute the option with easy reset-aligned presentations.
        # Cache only stable states reached after at least one physical pickup
        # loss and relift, then preserve a two-percent stream of fresh
        # end-to-end rollouts so the replay population cannot go stale.
        self.dr_anmar_receiver_curriculum_require_pickup_recovery = True
        self.dr_anmar_receiver_curriculum_restore_probability = 0.98
        self.dr_anmar_recovery_receiver_grasp_retain_curriculum = True
        self.dr_anmar_recovery_receiver_grasp_retain_objective = (
            "retained_handover_from_recovered_stable_presentation"
        )
        self.dr_anmar_recovery_receiver_controller = {
            "recovery_carry_lateral_action_limit": 0.08,
            "recovery_receiver_preposition_height": 0.025,
        }


@configclass
class NeedleHandoverDeadlineRecoveryResidualEnvCfg(
    NeedleHandoverRecoveryReceiverGraspRetainEnvCfg
):
    """Learn a bounded receiver residual under the original deadline."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.giver_identity = ObsTerm(
            func=e2e_observations.giver_and_deadline_context,
            clip=(0.0, 1.0),
        )
        # More than half of the lifted acquisition failures never reached the
        # old stable-presentation gate. Replay recovered lifted custody so one
        # rollout can learn giver presentation followed by receiver acquisition
        # from the physical state that caused the late deadline.
        self.dr_anmar_receiver_curriculum_capture_stage = "lifted_custody"
        self.dr_anmar_deadline_recovery_curriculum = True
        self.dr_anmar_deadline_recovery_rollout_steps_per_env = 384
        self.dr_anmar_deadline_recovery_objective = (
            "retained_handover_from_recovered_lifted_custody_under_original_"
            "episode_deadline_with_giver_presentation_then_receiver_"
            "acquisition"
        )
        self.dr_anmar_deadline_recovery_control = (
            "incumbent_plus_bounded_two_stage_giver_then_receiver_se3_"
            "residual"
        )
        # Training must use the same qualified recovered-transport contract as
        # full-task play. This field is applied explicitly by the benchmark;
        # inheriting a controller dictionary without applying it previously
        # made the evidence claim a configuration the policy never received.
        self.dr_anmar_deadline_recovery_controller = {
            "recovery_carry_lateral_action_limit": 0.08,
            "recovery_receiver_preposition_height": 0.025,
        }


@configclass
class NeedleHandoverDeadlineRecoveryOptionEnvCfg(
    NeedleHandoverDeadlineRecoveryResidualEnvCfg
):
    """Backward-compatible alias for rejected v10 experiment artifacts."""


@configclass
class NeedleHandoverJointTransferAcquisitionEnvCfg(
    NeedleHandoverReceiverGraspRetainEnvCfg
):
    """Learn coupled giver presentation and receiver acquisition from custody."""

    def __post_init__(self):
        super().__post_init__()
        # Cache as soon as a physics-owned lift has custody. The option must
        # learn the complete presentation-to-acquisition transition instead
        # of starting only after the analytic presentation has already won.
        self.dr_anmar_receiver_curriculum_capture_stage = "lifted_custody"
        self.dr_anmar_receiver_curriculum_restore_probability = 0.95
        self.dr_anmar_joint_transfer_acquisition_curriculum = True
        self.dr_anmar_joint_transfer_acquisition_objective = (
            "retained_handover_from_physics_owned_lifted_custody"
        )
        self.dr_anmar_joint_transfer_acquisition_controller = {
            "recovery_carry_lateral_action_limit": 0.08,
            "recovery_receiver_preposition_height": 0.025,
        }


@configclass
class NeedleHandoverTransferRefinementEnvCfg(
    NeedleHandoverJointTransferAcquisitionEnvCfg
):
    """Refine a frozen joint option from stable presentation to retention."""

    def __post_init__(self):
        super().__post_init__()
        # Model 20 already improved the complete lifted-custody trajectory.
        # Replay the downstream bottleneck instead of spending most PPO
        # samples repeating pickup and transport. A 128-step rollout covers
        # the measured stable-presentation-to-contact latency and lets terminal
        # retention credit reach the acquisition actions that caused it.
        self.dr_anmar_receiver_curriculum_capture_stage = (
            "stable_presentation"
        )
        self.dr_anmar_receiver_curriculum_restore_probability = 0.98
        self.dr_anmar_joint_transfer_acquisition_curriculum = False
        self.dr_anmar_transfer_refinement_curriculum = True
        self.dr_anmar_transfer_refinement_rollout_steps_per_env = 128
        self.dr_anmar_transfer_refinement_objective = (
            "retained_handover_from_physics_owned_stable_presentation"
        )
        self.dr_anmar_transfer_refinement_controller = {
            "recovery_carry_lateral_action_limit": 0.08,
            "recovery_receiver_preposition_height": 0.025,
        }


@configclass
class NeedleHandoverFrontierHardeningEnvCfg(
    NeedleHandoverJointTransferAcquisitionEnvCfg
):
    """Train a zero-impact v24 adapter over the frozen v23 handover stack."""

    def __post_init__(self):
        super().__post_init__()
        self.dr_anmar_joint_transfer_acquisition_curriculum = False
        self.dr_anmar_frontier_hardening_curriculum = True
        self.dr_anmar_failure_stratified_curriculum = True
        self.dr_anmar_frontier_hardening_rollout_steps_per_env = 64
        self.dr_anmar_frontier_hardening_objective = (
            "retained_handover_with_canonical_needle_geometry_balanced_roles_"
            "and_contact_quality_preserving_transport"
        )
        self.dr_anmar_controller_profile = "frontier-hardening-v24"
        self.dr_anmar_policy_serving_task = (
            "DrAnmar-Handover-Needle-Frontier-Eval-v0"
        )
        self.dr_anmar_policy_compatible_play_tasks = [
            "DrAnmar-Handover-Needle-Frontier-Durability-Eval-v0",
        ]
        self.events.balanced_handover_roles = EventTerm(
            func=mdp.assign_balanced_handover_roles,
            mode="reset",
        )
        # A resting needle is planar: randomize the full in-plane heading and
        # measured placement tolerance, not physically impossible free-space
        # roll/pitch.  Dynamics randomization remains disabled until calibrated
        # material/mass receipts exist.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.025, 0.025),
            "y": (-0.025, 0.025),
            "z": (-0.05, -0.05),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }
        self.dr_anmar_randomization_contract = {
            "tier": "calibrated_pose_v1",
            "placement_xy_m": [-0.025, 0.025],
            "resting_roll_pitch_rad": [0.0, 0.0],
            "heading_yaw_rad": [-math.pi, math.pi],
            "mass_randomization_enabled": False,
            "friction_randomization_enabled": False,
            "reason_physics_randomization_disabled": (
                "requires measured instrument-needle calibration receipts"
            ),
        }
        # Terminal +/-80 remains dominant.  Dense credit is a bounded
        # potential difference whose terminal potential is zero, so a failed
        # lift/contact trajectory cannot retain positive return.
        self.rewards.phase_progress.weight = 0.0
        self.rewards.potential_based_progress = RewTerm(
            func=mdp.potential_based_handover_progress,
            params={"gamma": 0.995},
            weight=4.0,
        )


@configclass
class NeedleHandoverFrontierEvalEnvCfg(
    NeedleHandoverDeadlineContextEnvCfg
):
    """Held-out balanced-role evaluation for the v24 handover contract."""

    def __post_init__(self):
        super().__post_init__()
        self.dr_anmar_controller_profile = "frontier-hardening-v24"
        self.events.balanced_handover_roles = EventTerm(
            func=mdp.assign_balanced_handover_roles,
            mode="reset",
        )
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.025, 0.025),
            "y": (-0.025, 0.025),
            "z": (-0.05, -0.05),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi, math.pi),
        }
        self.dr_anmar_randomization_contract = {
            "tier": "calibrated_pose_v1",
            "placement_xy_m": [-0.025, 0.025],
            "resting_roll_pitch_rad": [0.0, 0.0],
            "heading_yaw_rad": [-math.pi, math.pi],
            "mass_randomization_enabled": False,
            "friction_randomization_enabled": False,
            "reason_physics_randomization_disabled": (
                "requires measured instrument-needle calibration receipts"
            ),
        }


@configclass
class NeedleHandoverFrontierDurabilityEnvCfg(
    NeedleHandoverFrontierEvalEnvCfg
):
    """Held-out handover requiring over one second of retained custody."""

    def __post_init__(self):
        super().__post_init__()
        self.dr_anmar_handover_contract[
            "required_receiver_only_steps"
        ] = 60
        self.dr_anmar_durability_contract = {
            "receiver_only_control_steps": 60,
            "control_period_s": self.sim.dt * self.decimation,
            "minimum_receiver_only_duration_s": (
                60 * self.sim.dt * self.decimation
            ),
            "success_remains_physics_owned": True,
            "legacy_ten_step_success_task_unchanged": True,
        }
