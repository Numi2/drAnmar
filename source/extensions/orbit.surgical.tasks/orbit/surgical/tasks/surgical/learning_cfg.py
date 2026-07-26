# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared RSL-RL configurations for the Dr.Anmar Learning Path.

The task packages used to carry separate copies of the pre-4.0 RSL-RL
``ActorCritic`` configuration.  Keeping the current actor, critic, observation
normalization, action clipping, and numerical checks here gives every
Dr.Anmar task the same training-serving contract.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class DrAnmarReachResidualModelCfg(RslRlMLPModelCfg):
    """Residual actor anchored by the Stage 1 analytic relative-IK teacher."""

    class_name = (
        "orbit.surgical.tasks.surgical.reach.residual_model:"
        "ReachResidualMLPModel"
    )
    position_error_start: int = 23
    orientation_error_start: int = 26
    position_scale: float = 0.01
    orientation_scale: float = 0.05
    residual_scale: float = 0.25


@configclass
class DrAnmarDualReachResidualModelCfg(DrAnmarReachResidualModelCfg):
    """Two-controller residual actor for coordinated PSM pose control."""

    class_name = (
        "orbit.surgical.tasks.surgical.reach_dual.residual_model:"
        "DualReachResidualMLPModel"
    )
    arm_1_position_error_start: int = 46
    arm_1_orientation_error_start: int = 49
    arm_2_position_error_start: int = 52
    arm_2_orientation_error_start: int = 55


@configclass
class DrAnmarLiftResidualModelCfg(RslRlMLPModelCfg):
    """Contact-conditioned Stage 3 actor with a bounded learned residual."""

    class_name = (
        "orbit.surgical.tasks.surgical.lift.residual_model:"
        "LiftResidualMLPModel"
    )
    end_effector_position_start: int = 16
    object_position_start: int = 23
    object_velocity_start: int = 30
    target_position_start: int = 36
    contact_force_start: int = 43
    position_scale: float = 0.01
    approach_height: float = 0.02
    grasp_height: float = 0.0
    lateral_alignment_threshold: float = 0.004
    close_distance: float = 0.003
    slow_approach_radius: float = 0.02
    slow_approach_action_limit: float = 0.1
    normalized_contact_threshold: float = 0.002
    carry_angular_velocity_scale: float = 2.5
    carry_stable_angular_speed: float = 1.5
    lateral_clearance_below_target: float = 0.02
    carry_action_limit: float = 0.1
    residual_scale: float = 0.2


def _actor(hidden_dims: list[int], *, initial_std: float = 1.0) -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=initial_std),
    )


def reach_residual_actor(
    hidden_dims: list[int],
    *,
    initial_std: float = 0.25,
) -> DrAnmarReachResidualModelCfg:
    return DrAnmarReachResidualModelCfg(
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=initial_std
        ),
    )


def dual_reach_residual_actor(
    hidden_dims: list[int],
    *,
    initial_std: float = 0.25,
) -> DrAnmarDualReachResidualModelCfg:
    return DrAnmarDualReachResidualModelCfg(
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=initial_std
        ),
    )


def lift_residual_actor(
    hidden_dims: list[int],
    *,
    initial_std: float = 0.25,
) -> DrAnmarLiftResidualModelCfg:
    return DrAnmarLiftResidualModelCfg(
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=initial_std
        ),
    )


def _critic(hidden_dims: list[int]) -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=True,
    )


def _ppo(
    *,
    entropy_coef: float,
    learning_rate: float,
    gamma: float,
    learning_epochs: int,
    mini_batches: int,
) -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=entropy_coef,
        num_learning_epochs=learning_epochs,
        num_mini_batches=mini_batches,
        learning_rate=learning_rate,
        schedule="adaptive",
        gamma=gamma,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class DrAnmarReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Normalized, bounded PPO baseline for single- and dual-arm reaching."""

    num_steps_per_env = 64
    max_iterations = 1200
    save_interval = 25
    experiment_name = "dranmar_reach"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    check_for_nan = True
    empirical_normalization = False
    actor = _actor([256, 128, 64])
    critic = _critic([256, 128, 64])
    algorithm = _ppo(
        entropy_coef=0.005,
        learning_rate=5.0e-4,
        gamma=0.99,
        learning_epochs=5,
        mini_batches=8,
    )


@configclass
class DrAnmarCompactReachPPORunnerCfg(DrAnmarReachPPORunnerCfg):
    """Smaller policy for the lower-dimensional STAR and ECM embodiments."""

    num_steps_per_env = 32
    max_iterations = 800
    actor = _actor([128, 64])
    critic = _critic([128, 64])


@configclass
class DrAnmarManipulationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Contact-aware PPO baseline for lift and handover task families."""

    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 25
    experiment_name = "dranmar_manipulation"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    check_for_nan = True
    empirical_normalization = False
    actor = _actor([256, 128, 64], initial_std=0.8)
    critic = _critic([256, 128, 64])
    algorithm = _ppo(
        entropy_coef=0.004,
        learning_rate=2.5e-4,
        gamma=0.985,
        learning_epochs=5,
        mini_batches=8,
    )
