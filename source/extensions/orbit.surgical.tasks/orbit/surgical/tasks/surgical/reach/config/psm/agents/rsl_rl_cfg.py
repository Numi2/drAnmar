# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from orbit.surgical.tasks.surgical.learning_cfg import (
    DrAnmarReachPPORunnerCfg,
    reach_residual_actor,
)


@configclass
class PSMReachPPORunnerCfg(DrAnmarReachPPORunnerCfg):
    experiment_name = "dranmar_psm_reach"
    actor = reach_residual_actor([256, 128, 64])
