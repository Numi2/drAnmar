# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab configuration for one force-gated needle entry."""

from __future__ import annotations

from dataclasses import MISSING

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.configclass import configclass

from orbit.surgical.assets.psm import PSM_HIGH_PD_CFG, psm_gripper_close_command_expr

from . import mdp


@configclass
class PenetrationSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    needle: RigidObjectCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    jaw_1_needle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/psm_tool_gripper1_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Needle"],
        history_length=2,
    )
    jaw_2_needle_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/psm_tool_gripper2_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Needle"],
        history_length=2,
    )
    robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=2,
    )
    giver_tip_tissue_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/psm_tool_tip_link",
        history_length=2,
    )
    giver_jaw_1_tissue_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/psm_tool_gripper1_link",
        history_length=2,
    )
    giver_jaw_2_tissue_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/psm_tool_gripper2_link",
        history_length=2,
    )
    tissue_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TissueLeft",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.0185, 0.0, 0.05)),
        spawn=sim_utils.CuboidCfg(
            size=(0.033, 0.045, 0.006),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.58, 0.18, 0.16), roughness=0.58
            ),
        ),
    )
    tissue_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TissueRight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0185, 0.0, 0.05)),
        spawn=sim_utils.CuboidCfg(
            size=(0.033, 0.045, 0.006),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.58, 0.18, 0.16), roughness=0.58
            ),
        ),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        # Keep the qualified pickup/handover support transform. Raising the
        # table by 50 mm intersects the rotated PSM insertion chain and creates
        # a six-joint solver launch before the entry controller can act.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.457)),
        spawn=UsdFileCfg(usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/Table/table.usd"),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.95)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    entry_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="psm_tool_tip_link",
        resampling_time_range=(30.0, 30.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Compensate the giver's +20 mm world-Z safety lift in its fixed
            # root frame so the physical entry target remains unchanged.
            pos_x=(-0.01572351386552, -0.01572351386552),
            pos_y=(0.00706205237818, 0.00706205237818),
            pos_z=(-0.0730, -0.0730),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    body_action: DifferentialInverseKinematicsActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        ee_pose = ObsTerm(func=mdp.end_effector_pose)
        needle_tip_pose = ObsTerm(func=mdp.needle_tip_pose)
        needle_velocity = ObsTerm(func=mdp.needle_velocity)
        entry_goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "entry_pose"})
        surface_normal = ObsTerm(func=mdp.entry_surface_normal)
        indentation_depth = ObsTerm(func=mdp.indentation_and_depth)
        jaw_contacts = ObsTerm(func=mdp.jaw_contacts, clip=(0.0, 5.0))
        tissue_wrench = ObsTerm(func=mdp.normalized_tissue_wrench, clip=(-5.0, 5.0))
        force_history = ObsTerm(func=mdp.force_history, clip=(-5.0, 5.0))
        phase = ObsTerm(func=mdp.penetration_phase)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        privileged_state = ObsTerm(func=mdp.privileged_puncture_state)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_pregrasp = EventTerm(func=mdp.reset_pregrasped_needle, mode="reset")
    reset_evidence = EventTerm(func=mdp.reset_penetration_evidence, mode="reset")


@configclass
class RewardsCfg:
    phase_progress = RewTerm(func=mdp.phase_transition_once, weight=2.0)
    entry_progress = RewTerm(func=mdp.bounded_entry_progress, weight=1.0)
    success = RewTerm(func=mdp.successful_entry, weight=20.0)
    force_overshoot = RewTerm(func=mdp.normalized_force_overshoot, weight=-2.0)
    lateral_slip = RewTerm(func=mdp.lateral_slip, weight=-1.0)
    rcm_motion = RewTerm(func=mdp.rcm_motion, weight=-2.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-3)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    hard_failure = DoneTerm(func=mdp.hard_safety_failure)
    success = DoneTerm(func=mdp.successful_entry)


@configclass
class PenetrationEnvCfg(ManagerBasedRLEnvCfg):
    scene: PenetrationSceneCfg = PenetrationSceneCfg(
        num_envs=12, env_spacing=0.25, clone_in_fabric=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        robot = PSM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=PSM_HIGH_PD_CFG.spawn.replace(activate_contact_sensors=True),
            init_state=PSM_HIGH_PD_CFG.init_state.replace(
                pos=(-0.01037645055381, -0.0730, 0.06676338424909),
                rot=(0.175850305627, -0.684891721377, 0.684891721377, 0.175850305627),
            ),
        )
        robot.actuators["psm_tool"].effort_limit_sim = 0.8
        robot.actuators["psm_tool"].stiffness = 1200.0
        robot.actuators["psm_tool"].damping = 5.0
        robot.init_state.joint_pos.update(psm_gripper_close_command_expr())
        robot.init_state.joint_pos.update(
            {
                # Contact-free analytical stand-off obtained from the fixed
                # domain PSM and authored needle grasp. This avoids beginning
                # an episode with the tip already indenting outside the 1 mm
                # entry region.
                "psm_yaw_joint": 0.026266563683748245,
                "psm_pitch_end_joint": 0.05557627975940704,
                "psm_main_insertion_joint": 0.07030967622995377,
                "psm_tool_roll_joint": -0.000004311198154027807,
                "psm_tool_pitch_joint": -0.05519611015915871,
                "psm_tool_yaw_joint": -0.0260869599878788,
            }
        )
        self.scene.robot = robot
        self.scene.needle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Needle",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.07)),
            spawn=UsdFileCfg(
                usd_path=(
                    f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Props/SurgicalClosure/Needle/"
                    "dranmar_needle_entry_proxy.usda"
                ),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=8,
                    max_depenetration_velocity=1.0,
                    disable_gravity=True,
                    enable_gyroscopic_forces=True,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=2.0,
                    dynamic_friction=1.5,
                    restitution=0.0,
                    friction_combine_mode="max",
                ),
            ),
        )
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/PenetrationFrame"
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/psm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/psm_tool_tip_link",
                    name="end_effector",
                )
            ],
        )
        self.actions.body_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
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
            scale=(0.00025, 0.00025, 0.00025, 0.00872664626, 0.00872664626, 0.00872664626),
            clip={".*": (-1.0, 1.0)},
        )
        self.decimation = 10
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.viewer.eye = (0.0, 0.35, 0.18)
        self.viewer.lookat = (0.0, 0.0, 0.05)


@configclass
class PenetrationEnvCfg_PLAY(PenetrationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_evidence.params = {"fixed_domain": True}


@configclass
class ThroughPunctureObservationsCfg:
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        phase = ObsTerm(func=mdp.through_puncture_phase)
        through_progress = ObsTerm(func=mdp.through_puncture_progress)
        exit_delta = ObsTerm(func=mdp.through_exit_delta)

    @configclass
    class CriticCfg(PolicyCfg):
        privileged_state = ObsTerm(func=mdp.privileged_through_puncture_state)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ThroughPunctureRewardsCfg(RewardsCfg):
    exit_progress = RewTerm(func=mdp.bounded_exit_progress, weight=4.0)
    success = RewTerm(func=mdp.successful_through_puncture, weight=30.0)


@configclass
class ThroughPunctureTerminationsCfg(TerminationsCfg):
    success = DoneTerm(func=mdp.successful_through_puncture)


@configclass
class ThroughPunctureEnvCfg(PenetrationEnvCfg):
    """Continue the qualified entry along the needle arc to a grippable exit."""

    through_puncture: bool = True
    observations: ThroughPunctureObservationsCfg = ThroughPunctureObservationsCfg()
    rewards: ThroughPunctureRewardsCfg = ThroughPunctureRewardsCfg()
    terminations: ThroughPunctureTerminationsCfg = ThroughPunctureTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0


@configclass
class ThroughPunctureEnvCfg_PLAY(ThroughPunctureEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.reset_evidence.params = {"fixed_domain": True}
