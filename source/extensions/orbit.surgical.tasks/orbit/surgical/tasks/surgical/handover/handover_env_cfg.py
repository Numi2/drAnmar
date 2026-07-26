# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.configclass import configclass

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the handover scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot_1: ArticulationCfg = MISSING
    robot_2: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_1_frame: FrameTransformerCfg = MISSING
    ee_2_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Exact per-jaw sensors are required for filtered contact pairs in Isaac
    # Lab. They expose physical object ownership and unintended-contact force.
    robot_1_jaw_1_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_gripper1_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=2,
    )
    robot_1_jaw_2_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/psm_tool_gripper2_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=2,
    )
    robot_2_jaw_1_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_2/psm_tool_gripper1_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=2,
    )
    robot_2_jaw_2_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_2/psm_tool_gripper2_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=2,
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        spawn=UsdFileCfg(usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Table/table.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.95)),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_1_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot_1",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(30.0, 30.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.05, 0.05),
            pos_z=(-0.12, -0.08),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    # set the scale of the visualization markers to (0.01, 0.01, 0.01)
    ee_1_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    ee_1_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    robot_1_body_action: mdp.JointPositionActionCfg = MISSING
    robot_1_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    robot_2_body_action: mdp.JointPositionActionCfg = MISSING
    robot_2_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        robot_1_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        robot_1_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        robot_2_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_2")})
        robot_2_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_2")})
        robot_1_ee_pose = ObsTerm(
            func=mdp.end_effector_pose_in_robot_root_frame,
            params={"frame_cfg": SceneEntityCfg("ee_1_frame"), "robot_cfg": SceneEntityCfg("robot_1")},
        )
        robot_2_ee_pose = ObsTerm(
            func=mdp.end_effector_pose_in_robot_root_frame,
            params={"frame_cfg": SceneEntityCfg("ee_2_frame"), "robot_cfg": SceneEntityCfg("robot_2")},
        )
        object_pose_robot_1 = ObsTerm(
            func=mdp.object_pose_in_robot_root_frame, params={"robot_cfg": SceneEntityCfg("robot_1")}
        )
        object_pose_robot_2 = ObsTerm(
            func=mdp.object_pose_in_robot_root_frame, params={"robot_cfg": SceneEntityCfg("robot_2")}
        )
        object_velocity = ObsTerm(
            func=mdp.object_velocity_in_robot_root_frame, params={"robot_cfg": SceneEntityCfg("robot_1")}
        )
        robot_1_contacts = ObsTerm(
            func=mdp.jaw_contact_forces,
            params={
                "sensor_1_name": "robot_1_jaw_1_object_contact",
                "sensor_2_name": "robot_1_jaw_2_object_contact",
                "scale": 0.2,
            },
            clip=(0.0, 5.0),
        )
        robot_2_contacts = ObsTerm(
            func=mdp.jaw_contact_forces,
            params={
                "sensor_1_name": "robot_2_jaw_1_object_contact",
                "sensor_2_name": "robot_2_jaw_2_object_contact",
                "scale": 0.2,
            },
            clip=(0.0, 5.0),
        )
        receiver_goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_1_pose"})
        handover_phase = ObsTerm(func=mdp.handover_phase)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # The needle asset's default root height is +0.05 m and the native
            # table support surface is at 0.00 m.  Offset it by -0.05 m so the
            # episode begins at physical support height: neither falling from
            # mid-air nor embedded deeply enough for solver depenetration.
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.05, -0.05)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    giver_reach = RewTerm(
        func=mdp.end_effector_object_distance,
        params={"std": 0.06, "frame_name": "ee_2_frame", "minimum_phase": 0},
        weight=2.0,
    )

    giver_grasp = RewTerm(
        func=mdp.bilateral_grasp,
        params={
            "sensor_1_name": "robot_2_jaw_1_object_contact",
            "sensor_2_name": "robot_2_jaw_2_object_contact",
            "threshold": 0.01,
        },
        weight=4.0,
    )

    receiver_reach = RewTerm(
        func=mdp.end_effector_object_distance,
        params={"std": 0.06, "frame_name": "ee_1_frame", "minimum_phase": 1},
        weight=3.0,
    )

    receiver_grasp = RewTerm(
        func=mdp.bilateral_grasp,
        params={
            "sensor_1_name": "robot_1_jaw_1_object_contact",
            "sensor_2_name": "robot_1_jaw_2_object_contact",
            "threshold": 0.01,
            "minimum_phase": 2,
        },
        weight=5.0,
    )

    stable_dual_grasp = RewTerm(
        func=mdp.stable_dual_grasp,
        params={"linear_std": 0.1, "angular_std": 2.0},
        weight=3.0,
    )

    receiver_goal = RewTerm(
        func=mdp.receiver_goal_tracking,
        params={"position_std": 0.04, "orientation_std": 0.5},
        weight=8.0,
    )

    phase_progress = RewTerm(func=mdp.phase_progress, weight=10.0)
    success = RewTerm(func=mdp.successful_handover, weight=40.0)
    success_rate = RewTerm(
        func=mdp.sticky_success_rate,
        params={"success_fn": mdp.successful_handover},
        weight=0.0,
    )

    object_force_excess = RewTerm(
        func=mdp.contact_force_excess,
        params={
            "sensor_names": (
                "robot_1_jaw_1_object_contact",
                "robot_1_jaw_2_object_contact",
                "robot_2_jaw_1_object_contact",
                "robot_2_jaw_2_object_contact",
            ),
            "soft_limit": 1.0,
        },
        weight=-0.5,
    )

    protected_surface_contact = RewTerm(
        func=mdp.non_object_contact_force_excess,
        params={
            "sensor_names": (
                "robot_1_jaw_1_object_contact",
                "robot_1_jaw_2_object_contact",
                "robot_2_jaw_1_object_contact",
                "robot_2_jaw_2_object_contact",
            ),
            "soft_limit": 0.0,
        },
        weight=-2.0,
    )

    robot_1_rcm_motion = RewTerm(
        func=mdp.rcm_motion,
        params={"robot_cfg": SceneEntityCfg("robot_1", body_names="psm_remote_center_link")},
        weight=-2.0,
    )
    robot_2_rcm_motion = RewTerm(
        func=mdp.rcm_motion,
        params={"robot_cfg": SceneEntityCfg("robot_2", body_names="psm_remote_center_link")},
        weight=-2.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.successful_handover)

    excessive_object_force = DoneTerm(
        func=mdp.excessive_contact_force,
        params={
            "sensor_names": (
                "robot_1_jaw_1_object_contact",
                "robot_1_jaw_2_object_contact",
                "robot_2_jaw_1_object_contact",
                "robot_2_jaw_2_object_contact",
            ),
            "hard_limit": 5.0,
        },
    )

    protected_surface_force = DoneTerm(
        func=mdp.excessive_non_object_contact_force,
        params={
            "sensor_names": (
                "robot_1_jaw_1_object_contact",
                "robot_1_jaw_2_object_contact",
                "robot_2_jaw_1_object_contact",
                "robot_2_jaw_2_object_contact",
            ),
            "hard_limit": 2.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 20_000}
    )


##
# Environment configuration
##


@configclass
class HandoverEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the handover environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(
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
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.viewer.eye = (0.0, 0.5, 0.2)
        self.viewer.lookat = (0.0, 0.0, 0.05)
