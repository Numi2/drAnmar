#!/usr/bin/env python3
"""Minimal Isaac Lab deformable scene for the DrAnmar abdominal patient.

Run this with the repository's Isaac Lab launcher. A successful source-level
validation does not imply that these native PhysX routes have qualified on the
target Isaac/CUDA stack.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
from orbit.surgical.assets.dynamic_abdominal_patient import (  # noqa: E402
    DynamicSurgicalPatient,
    apply_patient_deformables,
    spawn_patient,
)

sim = sim_utils.SimulationContext(
    sim_utils.SimulationCfg(dt=1 / 120, device=args.device)
)
spawn_patient("/World/Patient", access_state="open")
mechanics_routes = apply_patient_deformables(
    "/World/Patient",
    include=("peritoneum",),
)
failed_routes = {
    component: result
    for component, result in mechanics_routes.items()
    if result["route"] == "not_applied"
}
if failed_routes:
    raise RuntimeError(f"Patient deformable setup failed closed: {failed_routes}")
patient = DynamicSurgicalPatient()
sim.set_camera_view([0.65, -0.72, 0.58], [0, 0, 0.02])
sim.reset()
while app.is_running():
    patient.step(sim.get_physics_dt())
    sim.step()
app.close()
