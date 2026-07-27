# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Isolated end-to-end needle-handover environment configuration."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
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
            "presentation_linear_speed_limit": 0.05,
            "presentation_angular_speed_limit": 5.0,
            "receiver_capture_required_steps": 1,
            "receiver_capture_follow_tolerance": 0.005,
            "receiver_capture_linear_speed_limit": 0.05,
            "receiver_capture_angular_speed_limit": 5.0,
            "giver_release_confirmation_steps": 1,
            "receiver_attempt_timeout_steps": 30,
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
        self.dr_anmar_receiver_curriculum_restore_probability = 0.5
        self.events.receiver_curriculum_reset = EventTerm(
            func=mdp.reset_receiver_curriculum_from_cache,
            mode="reset",
        )
