# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from orbit.surgical.tasks.surgical.learning_cfg import DrAnmarManipulationPPORunnerCfg


@configclass
class HandoverBlockPPORunnerCfg(DrAnmarManipulationPPORunnerCfg):
    experiment_name = "dranmar_block_handover"
