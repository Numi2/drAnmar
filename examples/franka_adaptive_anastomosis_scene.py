#!/usr/bin/env python3
"""Minimal scene configuration for the DrAnmar Adaptive Anastomosis Robot."""
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from orbit.surgical.assets.adaptive_anastomosis_robot import (
    make_franka_adaptive_anastomosis_robot_cfg,
    spawn_hollow_tissue_demo,
)

@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = make_franka_adaptive_anastomosis_robot_cfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        staple_state="loaded",
        collar_state="loaded",
        test_medium_state="full",
        collection_state="empty",
    )


def spawn_task_assets():
    return spawn_hollow_tissue_demo(
        "/World/DrAnmarHollowTissue",
        state="initial",
        translation=(0.62, 0.0, 0.82),
    )
