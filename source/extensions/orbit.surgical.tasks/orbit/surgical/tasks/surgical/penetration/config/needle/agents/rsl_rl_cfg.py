# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from orbit.surgical.tasks.surgical.learning_cfg import DrAnmarPenetrationPPORunnerCfg


@configclass
class PenetrationNeedlePPORunnerCfg(DrAnmarPenetrationPPORunnerCfg):
    pass
