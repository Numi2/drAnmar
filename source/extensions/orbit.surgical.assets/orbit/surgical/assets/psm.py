# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the da Vinci Research Kit (dVRK) Patient Side Manipulator.

The following configurations are available:

* :obj:`PSM_CFG`: dVRK PSM robot arm
* :obj:`PSM_HIGH_PD_CFG`: dVRK PSM robot arm with stiffer PD control

Reference: https://github.com/med-air/SurRoL
           https://github.com/WPI-AIM/dvrk_env

Jaw posture and actuator values come only from ``config/psm_foundation.json``.
Task files must not carry object- or room-specific jaw tuning.
"""

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from orbit.surgical.assets import ORBITSURGICAL_ASSETS_DATA_DIR

##
# Configuration
##

with (
    Path(__file__).resolve().parents[3] / "config/psm_foundation.json"
).open(encoding="utf-8") as profile_file:
    PSM_FOUNDATION_PROFILE = json.load(profile_file)
PSM_GRIPPER_PROFILE = PSM_FOUNDATION_PROFILE["gripper"]


def psm_gripper_command_expr(aperture_rad: float) -> dict[str, float]:
    """Return symmetric physical jaw targets for NVIDIA binary actions."""

    aperture = float(aperture_rad)
    return {
        "psm_tool_gripper1_joint": -aperture,
        "psm_tool_gripper2_joint": aperture,
    }


def psm_gripper_open_command_expr() -> dict[str, float]:
    return psm_gripper_command_expr(PSM_GRIPPER_PROFILE["open_rad"])


def psm_gripper_close_command_expr() -> dict[str, float]:
    return psm_gripper_command_expr(PSM_GRIPPER_PROFILE["close_rad"])


PSM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ORBITSURGICAL_ASSETS_DATA_DIR}/Robots/dVRK/PSM/psm_col.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "psm_yaw_joint": 0.01,
            "psm_pitch_end_joint": 0.01,
            "psm_main_insertion_joint": 0.07,
            "psm_tool_roll_joint": 0.01,
            "psm_tool_pitch_joint": 0.01,
            "psm_tool_yaw_joint": 0.01,
            **psm_gripper_open_command_expr(),
        },
        pos=(0.0, 0.0, 0.15),
    ),
    actuators={
        "psm": ImplicitActuatorCfg(
            joint_names_expr=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            effort_limit_sim=12.0,
            velocity_limit_sim=1.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "psm_tool": ImplicitActuatorCfg(
            joint_names_expr=["psm_tool_gripper.*"],
            effort_limit_sim=PSM_GRIPPER_PROFILE["effort_limit_nm"],
            velocity_limit_sim=PSM_GRIPPER_PROFILE["velocity_limit_rad_s"],
            stiffness=PSM_GRIPPER_PROFILE["stiffness"],
            damping=PSM_GRIPPER_PROFILE["damping"],
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of dVRK PSM robot arm."""


PSM_HIGH_PD_CFG = PSM_CFG.copy()
PSM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
PSM_HIGH_PD_CFG.actuators["psm"].stiffness = 800.0
PSM_HIGH_PD_CFG.actuators["psm"].damping = 40.0
"""Configuration of dVRK PSM robot arm with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
