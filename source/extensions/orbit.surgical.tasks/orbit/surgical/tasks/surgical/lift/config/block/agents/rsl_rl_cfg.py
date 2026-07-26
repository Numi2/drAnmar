# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from orbit.surgical.tasks.surgical.learning_cfg import (
    DrAnmarManipulationPPORunnerCfg,
    lift_residual_actor,
)


@configclass
class LiftBlockPPORunnerCfg(DrAnmarManipulationPPORunnerCfg):
    experiment_name = "dranmar_block_lift"
    actor = lift_residual_actor([256, 128, 64], initial_std=0.01)
