# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for the isolated end-to-end handover actor."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg
from orbit.surgical.tasks.surgical.learning_cfg import (
    DrAnmarManipulationPPORunnerCfg,
)


@configclass
class EndToEndHandoverModelCfg(RslRlMLPModelCfg):
    class_name = (
        "orbit.surgical.tasks.surgical.handover.end_to_end_model:"
        "EndToEndHandoverMLPModel"
    )


@configclass
class SafeBiteModelCfg(RslRlMLPModelCfg):
    class_name = (
        "orbit.surgical.tasks.surgical.handover.safe_bite_model:"
        "SafeBiteEndToEndMLPModel"
    )


@configclass
class HandoverNeedleEndToEndPPORunnerCfg(DrAnmarManipulationPPORunnerCfg):
    """Single policy, asymmetric critic, DAgger warm start, then PPO."""

    num_steps_per_env = 64
    max_iterations = 3000
    experiment_name = "dranmar_needle_handover_e2e_experimental"
    actor = EndToEndHandoverModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name=(
                "orbit.surgical.tasks.surgical.handover.end_to_end_model:"
                "PhaseMaskedGaussianDistribution"
            ),
            # Noise is applied in physical action space, so keep it below the
            # bounded residual authority instead of overwhelming the servo.
            init_std=0.005,
            std_type="log",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=16,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.5,
    )


@configclass
class HandoverNeedleSafeBitePPORunnerCfg(
    HandoverNeedleEndToEndPPORunnerCfg
):
    """Short-horizon receiver residual around the analytic T1 servo."""

    num_steps_per_env = 48
    max_iterations = 1500
    experiment_name = "dranmar_needle_safe_bite_t1"
    actor = SafeBiteModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name=(
                "orbit.surgical.tasks.surgical.handover.end_to_end_model:"
                "PhaseMaskedGaussianDistribution"
            ),
            init_std=0.003,
            std_type="log",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=16,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.5,
    )
