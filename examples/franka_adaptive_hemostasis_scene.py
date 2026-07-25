#!/usr/bin/env python3
"""Minimal DrAnmar Adaptive Hemostasis Robot scene skeleton."""
from isaaclab.app import AppLauncher
app_launcher=AppLauncher(headless=False)
simulation_app=app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from orbit.surgical.assets.adaptive_hemostasis_robot import make_franka_adaptive_hemostasis_robot_cfg, spawn_vessel_demo

class SceneCfg(InteractiveSceneCfg):
    ground=AssetBaseCfg(prim_path="/World/Ground",spawn=sim_utils.GroundPlaneCfg())
    light=AssetBaseCfg(prim_path="/World/Light",spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot=make_franka_adaptive_hemostasis_robot_cfg(prim_path="{ENV_REGEX_NS}/Robot")

sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0",dt=1/120))
scene=InteractiveScene(SceneCfg(num_envs=1,env_spacing=2.0))
spawn_vessel_demo("/World/DrAnmarBleedingVessel",translation=(0.55,0.0,0.02))
sim.reset()
while simulation_app.is_running():
    scene.write_data_to_sim();sim.step();scene.update(sim.get_physics_dt())
simulation_app.close()
