#!/usr/bin/env python3
"""Minimal scene skeleton for the DrAnmar perfusion and viability robot."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args=parser.parse_args()
app=AppLauncher(args).app
import isaaclab.sim as sim_utils
from orbit.surgical.assets.perfusion_viability_robot import (
    ClosedLoopPerfusionVerifier,
    make_franka_perfusion_viability_robot_cfg,
    phase_targets,
    spawn_tissue_demo,
)

def main():
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120,device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground",sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500).func("/World/Light",sim_utils.DomeLightCfg(intensity=2500))
    robot_cfg=make_franka_perfusion_viability_robot_cfg(prim_path="/World/Robot")
    robot=robot_cfg.class_type(robot_cfg)
    spawn_tissue_demo("/World/PerfusedTissue",condition="anastomotic_stenosis",translation=(0.55,0.0,0.82))
    sim.reset()
    verifier=ClosedLoopPerfusionVerifier()
    result=verifier.scan_intervene_rescan("anastomotic_stenosis",duration_s=18.0,dt_s=0.15)
    print(result["action"],result["viability_gain"])
    while app.is_running():
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim(); sim.step(); robot.update(sim.get_physics_dt())
if __name__=="__main__":
    main(); app.close()
