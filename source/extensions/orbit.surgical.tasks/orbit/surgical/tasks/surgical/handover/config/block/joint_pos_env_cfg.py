# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.configclass import configclass

from orbit.surgical.tasks.surgical.handover import mdp
from orbit.surgical.tasks.surgical.handover.handover_env_cfg import HandoverEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from orbit.surgical.assets.psm import (  # isort: skip
    PSM_CFG,
    psm_gripper_close_command_expr,
    psm_gripper_open_command_expr,
)


@configclass
class BlockHandoverEnvCfg(HandoverEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set PSM as robot
        self.scene.robot_1 = PSM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_1",
            spawn=PSM_CFG.spawn.replace(activate_contact_sensors=True),
            init_state=PSM_CFG.init_state.replace(
                pos=(-0.2, 0.0, 0.15),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )
        self.scene.robot_2 = PSM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_2",
            spawn=PSM_CFG.spawn.replace(activate_contact_sensors=True),
            init_state=PSM_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.15),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # Set actions for the specific robot type (PSM)
        self.actions.robot_1_body_action = mdp.JointPositionActionCfg(
            asset_name="robot_1",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.robot_2_body_action = mdp.JointPositionActionCfg(
            asset_name="robot_2",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.robot_1_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr=psm_gripper_open_command_expr(),
            close_command_expr=psm_gripper_close_command_expr(),
        )
        self.actions.robot_2_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_2",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr=psm_gripper_open_command_expr(),
            close_command_expr=psm_gripper_close_command_expr(),
        )
        # Set the body name for the end effector
        self.commands.receiver_pose.body_name = "psm_tool_tip_link"

        # Set Peg Block as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, 0.0, 0.05), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Surgical_block/block.usd",
                scale=(0.011, 0.011, 0.011),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=16,
                    max_angular_velocity=0.1,
                    max_linear_velocity=0.1,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_1_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )
        self.scene.ee_2_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_2/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_2/psm_tool_tip_link",
                    name="end_effector",
                ),
            ],
        )


@configclass
class BlockHandoverEnvCfg_PLAY(BlockHandoverEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
