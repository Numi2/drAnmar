# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from ...penetration_env_cfg import (
    PenetrationEnvCfg,
    PenetrationEnvCfg_PLAY,
    ThroughPunctureEnvCfg,
    ThroughPunctureEnvCfg_PLAY,
)
from .agents.rsl_rl_cfg import (
    PenetrationNeedlePPORunnerCfg,
    ThroughPunctureNeedlePPORunnerCfg,
)


gym.register(
    id="DrAnmar-Penetrate-Tissue-Needle-PSM-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PenetrationEnvCfg,
        "rsl_rl_cfg_entry_point": PenetrationNeedlePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="DrAnmar-Through-Puncture-Tissue-Needle-PSM-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ThroughPunctureEnvCfg,
        "rsl_rl_cfg_entry_point": ThroughPunctureNeedlePPORunnerCfg,
    },
    disable_env_checker=True,
)
gym.register(
    id="DrAnmar-Through-Puncture-Tissue-Needle-PSM-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ThroughPunctureEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": ThroughPunctureNeedlePPORunnerCfg,
    },
    disable_env_checker=True,
)
gym.register(
    id="DrAnmar-Penetrate-Tissue-Needle-PSM-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PenetrationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": PenetrationNeedlePPORunnerCfg,
    },
    disable_env_checker=True,
)
