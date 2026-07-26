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


def _actor(hidden_dims: list[int], *, initial_std: float = 1.0) -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=initial_std),
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
