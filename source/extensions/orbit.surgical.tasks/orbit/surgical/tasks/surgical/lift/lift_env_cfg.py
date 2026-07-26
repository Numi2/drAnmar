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

# The block collision mesh reaches 3.85 mm below its root after asset scale.
# Four millimeters clears the table without adding a free-fall transient.
LIFT_INITIAL_OBJECT_HEIGHT_M = 0.004
LIFT_MINIMUM_SUCCESS_HEIGHT_M = 0.06
LIFT_TARGET_OBJECT_HEIGHT_M = 0.08
LIFT_SUCCESS_DWELL_STEPS = 10
LIFT_CONTACT_THRESHOLD_N = 0.01
LIFT_OBJECT_FORCE_SOFT_LIMIT_N = 1.0
LIFT_OBJECT_FORCE_HARD_LIMIT_N = 5.0
LIFT_PROTECTED_SURFACE_FORCE_HARD_LIMIT_N = 2.0
ISAAC_IDENTITY_QUATERNION_XYZW = (0.0, 0.0, 0.0, 1.0)


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Native PhysX contacts establish a bilateral grasp. The MDP also subtracts
    # filtered object force from total jaw force to detect unintended contact.
    jaw_1_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/psm_tool_gripper1_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=2,
    )
    jaw_2_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/psm_tool_gripper2_link",
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

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.05, 0.05),
            pos_z=(-0.07, -0.07),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    # set the scale of the visualization markers to (0.01, 0.01, 0.01)
    object_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    object_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        end_effector_pose = ObsTerm(func=mdp.end_effector_pose_in_robot_root_frame)
        object_pose = ObsTerm(func=mdp.object_pose_in_robot_root_frame)
        object_velocity = ObsTerm(func=mdp.object_velocity_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        jaw_contact_forces = ObsTerm(func=mdp.jaw_contact_forces, params={"scale": 0.2}, clip=(0.0, 5.0))
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
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=2.0)

    bilateral_grasp = RewTerm(func=mdp.bilateral_grasp, params={"threshold": 0.01}, weight=4.0)

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": LIFT_MINIMUM_SUCCESS_HEIGHT_M},
        weight=6.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.08,
            "minimal_height": LIFT_MINIMUM_SUCCESS_HEIGHT_M,
            "command_name": "object_pose",
        },
        weight=8.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.015,
            "minimal_height": LIFT_MINIMUM_SUCCESS_HEIGHT_M,
            "command_name": "object_pose",
        },
        weight=6.0,
    )

    object_goal_orientation = RewTerm(
        func=mdp.object_goal_orientation,
        params={"std": 0.35, "command_name": "object_pose", "contact_threshold": 0.01},
        weight=4.0,
    )

    stable_grasp = RewTerm(
        func=mdp.stable_object_motion,
        params={"linear_std": 0.08, "angular_std": 1.5, "contact_threshold": 0.01},
        weight=2.0,
    )

    success = RewTerm(
        func=mdp.sustained_lift_success,
        params={
            "required_consecutive_steps": LIFT_SUCCESS_DWELL_STEPS,
            "command_name": "object_pose",
            "minimum_height": LIFT_MINIMUM_SUCCESS_HEIGHT_M,
            "position_threshold": 0.015,
            "orientation_threshold": 0.35,
            "contact_threshold": LIFT_CONTACT_THRESHOLD_N,
            "maximum_linear_speed": 0.08,
            "maximum_angular_speed": 1.5,
        },
        weight=30.0,
    )
    success_rate = RewTerm(
        func=mdp.sustained_lift_success,
        params={
            "required_consecutive_steps": LIFT_SUCCESS_DWELL_STEPS,
            "publish_metric": True,
            "return_zero": True,
            "command_name": "object_pose",
            "minimum_height": LIFT_MINIMUM_SUCCESS_HEIGHT_M,
            "position_threshold": 0.015,
            "orientation_threshold": 0.35,
            "contact_threshold": LIFT_CONTACT_THRESHOLD_N,
            "maximum_linear_speed": 0.08,
            "maximum_angular_speed": 1.5,
        },
        weight=0.0,
    )

    object_force_excess = RewTerm(
        func=mdp.contact_force_excess,
        params={
            "sensor_names": ("jaw_1_object_contact", "jaw_2_object_contact"),
            "soft_limit": LIFT_OBJECT_FORCE_SOFT_LIMIT_N,
        },
        weight=-0.5,
    )

    protected_surface_contact = RewTerm(
        func=mdp.non_object_contact_force_excess,
        params={
            "sensor_names": ("jaw_1_object_contact", "jaw_2_object_contact"),
            "soft_limit": 0.0,
        },
        weight=-2.0,
    )

    rcm_motion = RewTerm(
        func=mdp.rcm_motion,
        params={"robot_cfg": SceneEntityCfg("robot", body_names="psm_remote_center_link")},
        weight=-2.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(
        func=mdp.sustained_lift_success,
        params={
            "required_consecutive_steps": LIFT_SUCCESS_DWELL_STEPS,
            "command_name": "object_pose",
            "minimum_height": LIFT_MINIMUM_SUCCESS_HEIGHT_M,
            "position_threshold": 0.015,
            "orientation_threshold": 0.35,
            "contact_threshold": LIFT_CONTACT_THRESHOLD_N,
            "maximum_linear_speed": 0.08,
            "maximum_angular_speed": 1.5,
        },
    )

    excessive_object_force = DoneTerm(
        func=mdp.excessive_contact_force,
        params={
            "sensor_names": ("jaw_1_object_contact", "jaw_2_object_contact"),
            "hard_limit": LIFT_OBJECT_FORCE_HARD_LIMIT_N,
        },
    )

    protected_surface_force = DoneTerm(
        func=mdp.excessive_non_object_contact_force,
        params={
            "sensor_names": ("jaw_1_object_contact", "jaw_2_object_contact"),
            "hard_limit": LIFT_PROTECTED_SURFACE_FORCE_HARD_LIMIT_N,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 20_000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-3, "num_steps": 20_000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

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
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 1.0 / 200.0
        self.viewer.eye = (0.2, 0.2, 0.1)
        self.viewer.lookat = (0.0, 0.0, 0.04)
