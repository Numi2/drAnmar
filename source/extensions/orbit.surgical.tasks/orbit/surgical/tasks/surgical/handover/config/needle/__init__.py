# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import (
    agents,
    e2e_ik_rel_env_cfg,
    ik_abs_env_cfg,
    ik_rel_env_cfg,
    joint_pos_env_cfg,
    t1_safe_bite_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Handover-Needle-Dual-PSM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.NeedleHandoverEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HandoverNeedlePPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Needle-Dual-PSM-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.NeedleHandoverEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HandoverNeedlePPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Handover-Needle-Dual-PSM-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.NeedleHandoverEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HandoverNeedlePPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Needle-Dual-PSM-IK-Abs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.NeedleHandoverEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HandoverNeedlePPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.NeedleHandoverEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HandoverNeedlePPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Needle-Dual-PSM-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.NeedleHandoverEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HandoverNeedlePPORunnerCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

##
# Experimental end-to-end inverse kinematics control
##

gym.register(
    id="Isaac-Handover-Needle-Dual-PSM-IK-Rel-Structured-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg.NeedleHandoverEndToEndEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Needle-Dual-PSM-IK-Rel-Structured-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg.NeedleHandoverEndToEndEnvCfg_PLAY
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-Structured-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg.NeedleHandoverEndToEndEnvCfg_PLAY
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="DrAnmar-Handover-Needle-Deadline-Context-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg.NeedleHandoverDeadlineContextEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Needle-Receiver-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg.NeedleHandoverReceiverCurriculumEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Needle-Receiver-Grasp-Retain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg.NeedleHandoverReceiverGraspRetainEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="DrAnmar-Handover-Needle-Pickup-Recovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg.NeedleHandoverPickupRecoveryCurriculumEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="DrAnmar-Handover-Needle-Recovery-Receiver-Grasp-Retain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg
            .NeedleHandoverRecoveryReceiverGraspRetainEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="DrAnmar-Handover-Needle-Deadline-Recovery-Option-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg
            .NeedleHandoverDeadlineRecoveryOptionEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="DrAnmar-Handover-Needle-Deadline-Recovery-Residual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg
            .NeedleHandoverDeadlineRecoveryResidualEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="DrAnmar-Handover-Needle-Joint-Transfer-Acquisition-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg
            .NeedleHandoverJointTransferAcquisitionEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="DrAnmar-Handover-Needle-Transfer-Refinement-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            e2e_ik_rel_env_cfg
            .NeedleHandoverTransferRefinementEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleEndToEndPPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

##
# Post-handover deformable-tissue progression
##

gym.register(
    id="Isaac-Handover-Needle-Safe-Bite-T1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            t1_safe_bite_env_cfg.NeedleHandoverSafeBiteT1EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleSafeBitePPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Needle-Safe-Bite-T1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            t1_safe_bite_env_cfg.NeedleHandoverSafeBiteT1EnvCfg_PLAY
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleSafeBitePPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Needle-Safe-Bite-T1-Visual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            t1_safe_bite_env_cfg.NeedleHandoverSafeBiteT1VisualEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleSafeBitePPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Needle-Safe-Bite-Chain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            t1_safe_bite_env_cfg.NeedleHandoverSafeBiteChainEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleSafeBitePPORunnerCfg
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Handover-Needle-Safe-Bite-Chain-Visual-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            t1_safe_bite_env_cfg.NeedleHandoverSafeBiteChainVisualEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            agents.rsl_rl_e2e_cfg.HandoverNeedleSafeBitePPORunnerCfg
        ),
    },
    disable_env_checker=True,
)
