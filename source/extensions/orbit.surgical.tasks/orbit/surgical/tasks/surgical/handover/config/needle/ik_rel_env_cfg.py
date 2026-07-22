# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from orbit.surgical.assets.psm import PSM_HIGH_PD_CFG  # isort: skip


@configclass
class NeedleHandoverEnvCfg(joint_pos_env_cfg.NeedleHandoverEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set PSM as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot_1 = PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        self.scene.robot_1.spawn.activate_contact_sensors = True
        self.scene.robot_1.init_state.joint_pos["psm_tool_gripper1_joint"] = -0.5
        self.scene.robot_1.init_state.joint_pos["psm_tool_gripper2_joint"] = 0.5
        self.scene.robot_1.init_state.pos = (0.2, 0.0, 0.15)
        self.scene.robot_1.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot_2 = PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
        self.scene.robot_2.spawn.activate_contact_sensors = True
        self.scene.robot_2.init_state.joint_pos["psm_tool_gripper1_joint"] = -0.5
        self.scene.robot_2.init_state.joint_pos["psm_tool_gripper2_joint"] = 0.5
        self.scene.robot_2.init_state.pos = (-0.2, 0.0, 0.15)
        self.scene.robot_2.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # Set actions for the specific robot type (PSM)
        self.actions.robot_1_body_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_1",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            body_name="psm_tool_tip_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=(0.01, 0.01, 0.01, 0.05, 0.05, 0.05),
            clip={".*": (-1.0, 1.0)},
        )
        self.actions.robot_2_body_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_2",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            body_name="psm_tool_tip_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=(0.01, 0.01, 0.01, 0.05, 0.05, 0.05),
            clip={".*": (-1.0, 1.0)},
        )


@configclass
class NeedleHandoverEnvCfg_PLAY(NeedleHandoverEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
