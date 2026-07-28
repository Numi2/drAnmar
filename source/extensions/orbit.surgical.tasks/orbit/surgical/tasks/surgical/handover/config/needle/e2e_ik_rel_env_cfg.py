# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Isolated end-to-end needle-handover environment configuration."""

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


def terminal_transfer_failure(env):
    """Penalize physical transfer failures instead of rewarding partial progress."""
    return (
        mdp.needle_dropped_after_pickup(env)
        | mdp.pickup_attempts_exhausted(env)
        | mdp.premature_giver_release(env)
        | mdp.receiver_retention_lost(env)
    ).float()


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
            "recovery_carry_lateral_action_limit": 0.10,
            "recovery_receiver_preposition_height": 0.015,
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
            "recovery_carry_lateral_action_limit": 0.10,
            "recovery_receiver_preposition_height": 0.015,
        }


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
            "recovery_carry_lateral_action_limit": 0.10,
            "recovery_receiver_preposition_height": 0.015,
        }
