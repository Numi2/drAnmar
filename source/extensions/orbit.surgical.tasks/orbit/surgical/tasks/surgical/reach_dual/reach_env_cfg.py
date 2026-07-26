# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# Copyright (c) 2026, Dr.Anmar Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import mdp

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.95)),
    )

    table: AssetBaseCfg = MISSING

    # robots
    robot_1: ArticulationCfg = MISSING
    robot_2: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_1_pose: mdp.UniformPoseCommandCfg = MISSING

    ee_2_pose: mdp.UniformPoseCommandCfg = MISSING


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_1_action: ActionTerm = MISSING
    gripper_1_action: ActionTerm | None = None

    arm_2_action: ActionTerm = MISSING
    gripper_2_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_1_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot_1")},
        )
        joint_1_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot_1")},
        )
        joint_2_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot_2")},
        )
        joint_2_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot_2")},
        )
        pose_1_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_1_pose"})
        pose_2_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_2_pose"})
        target_1_relative_position = ObsTerm(
            func=mdp.pose_command_error_vector,
            params={
                "command_name": "ee_1_pose",
                "robot_cfg": SceneEntityCfg("robot_1"),
                "frame_cfg": SceneEntityCfg("ee_1_frame"),
            },
            clip=(-0.25, 0.25),
        )
        target_1_relative_orientation = ObsTerm(
            func=mdp.pose_command_orientation_error_vector,
            params={
                "command_name": "ee_1_pose",
                "robot_cfg": SceneEntityCfg("robot_1"),
                "frame_cfg": SceneEntityCfg("ee_1_frame"),
            },
            clip=(-3.1416, 3.1416),
        )
        target_2_relative_position = ObsTerm(
            func=mdp.pose_command_error_vector,
            params={
                "command_name": "ee_2_pose",
                "robot_cfg": SceneEntityCfg("robot_2"),
                "frame_cfg": SceneEntityCfg("ee_2_frame"),
            },
            clip=(-0.25, 0.25),
        )
        target_2_relative_orientation = ObsTerm(
            func=mdp.pose_command_orientation_error_vector,
            params={
                "command_name": "ee_2_pose",
                "robot_cfg": SceneEntityCfg("robot_2"),
                "frame_cfg": SceneEntityCfg("ee_2_frame"),
            },
            clip=(-3.1416, 3.1416),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_1_joints: EventTerm = MISSING

    reset_robot_2_joints: EventTerm = MISSING


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    end_effector_1_position_tracking = RewTerm(
        func=mdp.position_command_tanh,
        weight=1.25,
        params={
            "command_name": "ee_1_pose",
            "std": 0.03,
            "asset_cfg": SceneEntityCfg("robot_1", body_names=MISSING),
            "robot_cfg": SceneEntityCfg("robot_1"),
            "frame_cfg": SceneEntityCfg("ee_1_frame"),
        },
    )
    end_effector_1_orientation_tracking = RewTerm(
        func=mdp.orientation_command_tanh,
        weight=0.25,
        params={
            "command_name": "ee_1_pose",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot_1", body_names=MISSING),
            "robot_cfg": SceneEntityCfg("robot_1"),
            "frame_cfg": SceneEntityCfg("ee_1_frame"),
        },
    )

    end_effector_2_position_tracking = RewTerm(
        func=mdp.position_command_tanh,
        weight=1.25,
        params={
            "command_name": "ee_2_pose",
            "std": 0.03,
            "asset_cfg": SceneEntityCfg("robot_2", body_names=MISSING),
            "robot_cfg": SceneEntityCfg("robot_2"),
            "frame_cfg": SceneEntityCfg("ee_2_frame"),
        },
    )
    end_effector_2_orientation_tracking = RewTerm(
        func=mdp.orientation_command_tanh,
        weight=0.25,
        params={
            "command_name": "ee_2_pose",
            "std": 0.25,
            "asset_cfg": SceneEntityCfg("robot_2", body_names=MISSING),
            "robot_cfg": SceneEntityCfg("robot_2"),
            "frame_cfg": SceneEntityCfg("ee_2_frame"),
        },
    )
    success = RewTerm(
        func=mdp.successful_dual_reach,
        weight=3.0,
        params={
            "command_1_name": "ee_1_pose",
            "command_2_name": "ee_2_pose",
            "position_threshold": 0.01,
            "orientation_threshold": 0.15,
        },
    )
    success_rate = RewTerm(
        func=mdp.sticky_success_rate,
        weight=0.0,
        params={
            "success_fn": mdp.successful_dual_reach,
            "success_params": {
                "command_1_name": "ee_1_pose",
                "command_2_name": "ee_2_pose",
                "position_threshold": 0.01,
                "orientation_threshold": 0.15,
            },
        },
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    joint_1_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot_1")},
    )
    joint_2_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot_2")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.successful_dual_reach,
        params={
            "command_1_name": "ee_1_pose",
            "command_2_name": "ee_2_pose",
            "position_threshold": 0.01,
            "orientation_threshold": 0.15,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.001, "num_steps": 20_000}
    )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        clone_in_fabric=True,
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 1.0 / 60.0
