# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-PSM task for complete curved-needle passage and receiver pullout."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from orbit.surgical.assets.psm import (
    PSM_HIGH_PD_CFG,
    psm_gripper_close_command_expr,
    psm_gripper_open_command_expr,
)

from . import mdp
from .penetration_env_cfg import (
    ActionsCfg,
    PenetrationSceneCfg,
    RewardsCfg,
    TerminationsCfg,
    ThroughPunctureEnvCfg,
    ThroughPunctureObservationsCfg,
)


@configclass
class PulloutSceneCfg(PenetrationSceneCfg):
    robot_receiver: ArticulationCfg = MISSING
    receiver_frame: FrameTransformerCfg = MISSING

    receiver_jaw_1_needle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotReceiver/psm_tool_gripper1_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Needle"],
        history_length=2,
    )
    receiver_jaw_2_needle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotReceiver/psm_tool_gripper2_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Needle"],
        history_length=2,
    )
    receiver_tip_tissue_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotReceiver/psm_tool_tip_link",
        history_length=2,
    )
    receiver_jaw_1_tissue_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotReceiver/psm_tool_gripper1_link",
        history_length=2,
    )
    receiver_jaw_2_tissue_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RobotReceiver/psm_tool_gripper2_link",
        history_length=2,
    )


@configclass
class PulloutActionsCfg(ActionsCfg):
    giver_gripper_action: BinaryJointPositionActionCfg = MISSING
    receiver_body_action: DifferentialInverseKinematicsActionCfg = MISSING
    receiver_gripper_action: BinaryJointPositionActionCfg = MISSING


@configclass
class PulloutObservationsCfg:
    @configclass
    class PolicyCfg(ThroughPunctureObservationsCfg.PolicyCfg):
        receiver_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_receiver")},
        )
        receiver_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_receiver")},
        )
        receiver_ee_pose = ObsTerm(func=mdp.pullout_receiver_ee_pose)
        receiver_contacts = ObsTerm(func=mdp.pullout_receiver_contacts)
        receiver_guidance = ObsTerm(func=mdp.pullout_receiver_guidance)
        pullout_phase = ObsTerm(func=mdp.pullout_phase)
        custody = ObsTerm(func=mdp.pullout_custody)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        privileged_state = ObsTerm(func=mdp.privileged_pullout_state)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class PulloutRewardsCfg(RewardsCfg):
    receiver_progress = RewTerm(func=mdp.bounded_receiver_progress, weight=2.0)
    clearance_progress = RewTerm(func=mdp.bounded_clearance_progress, weight=4.0)
    success = RewTerm(func=mdp.successful_pullout, weight=40.0)


@configclass
class PulloutTerminationsCfg(TerminationsCfg):
    success = DoneTerm(func=mdp.successful_pullout)


@configclass
class PulloutEnvCfg(ThroughPunctureEnvCfg):
    pullout: bool = True
    scene: PulloutSceneCfg = PulloutSceneCfg(
        num_envs=12, env_spacing=0.25, clone_in_fabric=True
    )
    observations: PulloutObservationsCfg = PulloutObservationsCfg()
    actions: PulloutActionsCfg = PulloutActionsCfg()
    rewards: PulloutRewardsCfg = PulloutRewardsCfg()
    terminations: PulloutTerminationsCfg = PulloutTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        receiver = PSM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotReceiver",
            spawn=PSM_HIGH_PD_CFG.spawn.replace(activate_contact_sensors=True),
            init_state=PSM_HIGH_PD_CFG.init_state.replace(
                # Park fully lateral to the 70 mm-wide tissue span. The old
                # 50 mm root placed receiver jaw 2 inside the right slab at
                # reset and generated a measured 161.9 N contact.
                pos=(0.080, -0.0730, 0.04676338424909),
                rot=(0.175850305627, -0.684891721377, 0.684891721377, 0.175850305627),
            ),
        )
        receiver.actuators["psm_tool"].effort_limit_sim = 0.8
        receiver.actuators["psm_tool"].stiffness = 1200.0
        receiver.actuators["psm_tool"].damping = 5.0
        receiver.init_state.joint_pos.update(psm_gripper_open_command_expr())
        receiver.init_state.joint_pos.update(
            {
                "psm_yaw_joint": 0.026266563683748245,
                "psm_pitch_end_joint": 0.05557627975940704,
                "psm_main_insertion_joint": 0.07030967622995377,
                "psm_tool_roll_joint": -0.000004311198154027807,
                "psm_tool_pitch_joint": -0.05519611015915871,
                "psm_tool_yaw_joint": -0.0260869599878788,
            }
        )
        self.scene.robot_receiver = receiver
        # Retain the collision-free authored needle proxy inherited from the
        # entry task. The native backend owns needle/tissue resistance and the
        # receiver uses the disclosed sustained geometry custody fallback.
        # Tissue itself stays collision-enabled, so both PSMs remain blocked.
        self.actions.giver_gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr=psm_gripper_open_command_expr(),
            close_command_expr=psm_gripper_close_command_expr(),
        )
        self.actions.receiver_body_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_receiver",
            joint_names=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            body_name="psm_tool_tip_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=(
                0.00025,
                0.00025,
                0.00025,
                0.00872664626,
                0.00872664626,
                0.00872664626,
            ),
            clip={".*": (-1.0, 1.0)},
        )
        self.actions.receiver_gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot_receiver",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr=psm_gripper_open_command_expr(),
            close_command_expr=psm_gripper_close_command_expr(),
        )
        marker_cfg = self.scene.ee_frame.visualizer_cfg.copy()
        marker_cfg.prim_path = "/Visuals/PulloutReceiverFrame"
        self.scene.receiver_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/RobotReceiver/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/RobotReceiver/psm_tool_tip_link",
                    name="receiver_end_effector",
                )
            ],
        )
        self.episode_length_s = 60.0


@configclass
class PulloutEnvCfg_PLAY(PulloutEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_evidence.params = {"fixed_domain": True}
