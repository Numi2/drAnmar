#!/usr/bin/env python3
"""Minimal DrAnmar atraumatic exposure scene for Isaac Lab."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from orbit.surgical.assets.atraumatic_exposure_robot import (
    make_franka_exposure_robot_cfg,
    spawn_exposure_tissue_demo,
)

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot = make_franka_exposure_robot_cfg(prim_path="{ENV_REGEX_NS}/Robot", pad_type="fenestrated")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
spawn_exposure_tissue_demo("/World/ExposureTissue", translation=(0.54, 0.0, 0.0))
sim.reset()
while app.is_running():
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
app.close()
