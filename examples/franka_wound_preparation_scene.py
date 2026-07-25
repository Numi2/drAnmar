#!/usr/bin/env python3
"""Minimal DrAnmar wound-preparation scene skeleton.

Run through the matching Isaac Lab launcher. Runtime parameter tuning is left to
the host project.
"""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from orbit.surgical.assets.wound_preparation_robot import (
    FluidLedger,
    WoundPreparationSequenceController,
    apply_wound_surface_deformable,
    attach_demo_debris,
    ensure_irrigation_particle_system,
    make_franka_wound_preparation_robot_cfg,
    spawn_wound_bed_demo,
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot = make_franka_wound_preparation_robot_cfg(
        prim_path="{ENV_REGEX_NS}/Robot", irrigation_state="loaded", collection_state="empty"
    )


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
spawn_wound_bed_demo("/World/DrAnmarWoundBed", translation=(0.55, 0.0, 0.82))
sim.reset()

# Current surface-deformable and PBD particle setup occurs after stage assembly.
apply_wound_surface_deformable("/World/DrAnmarWoundBed")
attachments = attach_demo_debris("/World/DrAnmarWoundBed")
particle_paths = ensure_irrigation_particle_system()

controller = WoundPreparationSequenceController(
    tool_path="/World/envs/env_0/Robot/DrAnmarWoundPreparationTool",
    wound_root_path="/World/DrAnmarWoundBed",
    ledger=FluidLedger(),
)
controller.debridement.register_demo(attachments)
print(controller.snapshot())

while simulation_app.is_running():
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

simulation_app.close()
