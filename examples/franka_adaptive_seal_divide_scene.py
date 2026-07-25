"""DrAnmar Adaptive Seal-and-Divide Robot scene skeleton.

Run through the Isaac Lab launcher on CUDA. The example intentionally leaves
contact-force collection, seal-band deployment, bridge engagement, and
runtime qualification to the host task.
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from orbit.surgical.assets.adaptive_seal_divide_robot import (
    make_franka_adaptive_seal_divide_robot_cfg,
    spawn_vessel_demo,
)

class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot = make_franka_adaptive_seal_divide_robot_cfg(prim_path="{ENV_REGEX_NS}/Robot")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=1/240))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
spawn_vessel_demo("/World/DrAnmarSealDivideVessel", translation=(0.55, 0.0, 0.02))
sim.reset()
while simulation_app.is_running():
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
simulation_app.close()
