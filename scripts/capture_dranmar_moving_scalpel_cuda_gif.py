#!/usr/bin/env python3
"""Render the real moving-scalpel Warp CUDA trajectory in Isaac Lab."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output", type=Path,
    default=ROOT / "docs/media/dranmar-moving-scalpel-cut-cuda.gif",
)
parser.add_argument("--receipt", type=Path)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=600)
parser.add_argument("--hold-frames", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import numpy as np
import torch
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from PIL import Image, ImageDraw

scripts_path = str(ROOT / "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from dr_anmar_cuttable_tissue_solver import load_profile
from dr_anmar_cuttable_tissue_warp import _clear_force, _neo_hookean_prony_force
from dr_anmar_dynamic_curved_cut_fem import load_curved_profile
from dr_anmar_dynamic_curved_cut_warp import _integrate_nodes
from dr_anmar_moving_scalpel_cut_fem import (
    DEFAULT_MOVING_PROFILE_PATH,
    MovingScalpelCutFEM,
    _path_poses,
    _work_channels,
    load_moving_profile,
)
from dr_anmar_moving_scalpel_cut_warp import _moving_cohesive_and_wedge_force
from dr_anmar_tissue_render_utils import (
    boundary_triangles,
    font,
    material,
    mesh,
    rgb_image,
    set_points,
    set_pose,
)


def _annotate(
    image: Image.Image,
    phase: str,
    segment: int,
    total_segments: int,
    events: int,
    released: int,
) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, canvas.width, 64), fill=(8, 13, 19, 224))
    draw.rectangle((0, canvas.height - 52, canvas.width, canvas.height), fill=(8, 13, 19, 224))
    draw.text((22, 13), "Dr.Anmar · Moving Scalpel Fracture", font=font(21, bold=True), fill=(250, 245, 242, 255))
    label = phase.upper()
    box = draw.textbbox((0, 0), label, font=font(14, bold=True))
    label_width = box[2] - box[0]
    draw.rounded_rectangle(
        (canvas.width - label_width - 48, 15, canvas.width - 20, 48),
        radius=11, fill=(153, 50, 57, 240),
    )
    draw.text((canvas.width - label_width - 34, 22), label, font=font(14, bold=True), fill=(255, 255, 255, 255))
    draw.text(
        (22, canvas.height - 41),
        f"REAL WARP CUDA · ENERGY-GATED EVENTS {events:04d} · RELEASED PAIRS {released:02d}/85",
        font=font(13, bold=True), fill=(245, 190, 181, 255),
    )
    draw.text(
        (22, canvas.height - 22),
        "LIVE DEFORMING FEM · TWO-SIDED WOUND COLLISION · 0× DISPLACEMENT EXAGGERATION",
        font=font(12), fill=(222, 229, 234, 255),
    )
    progress = 0.0 if total_segments <= 0 else segment / total_segments
    x0, x1 = canvas.width - 220, canvas.width - 22
    draw.rounded_rectangle((x0, canvas.height - 29, x1, canvas.height - 17), radius=6, fill=(52, 61, 70, 230))
    draw.rounded_rectangle((x0, canvas.height - 29, x0 + int((x1 - x0) * progress), canvas.height - 17), radius=6, fill=(219, 101, 93, 245))
    return canvas.convert("RGB")


def _cuda_moving_trajectory():
    moving = load_moving_profile(DEFAULT_MOVING_PROFILE_PATH)
    base = load_profile(ROOT / moving["base_profile"])
    curved = load_curved_profile(ROOT / moving["embedded_profile"])
    solver = MovingScalpelCutFEM(base, curved, moving)
    device = "cuda:0"
    wp.init()
    if not wp.get_device(device).is_cuda:
        raise RuntimeError("Moving-scalpel render requires a real CUDA device")
    fem = solver.mesh
    positions = wp.array(solver.position.astype(np.float32), dtype=wp.vec3, device=device)
    velocities = wp.array(solver.velocity.astype(np.float32), dtype=wp.vec3, device=device)
    tetrahedra = wp.array(fem.tetrahedra.astype(np.int32), dtype=wp.vec4i, device=device)
    inverse_rest = wp.array(fem.dm_inverse.astype(np.float32), dtype=wp.mat33, device=device)
    gradients = wp.array(fem.shape_gradients.reshape((-1, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    volumes = wp.array(fem.rest_volume.astype(np.float32), dtype=float, device=device)
    history = wp.array(fem.prony_history.astype(np.float32), dtype=wp.mat33, device=device)
    previous = wp.array(fem.previous_elastic_stress.astype(np.float32), dtype=wp.mat33, device=device)
    masses = wp.array(fem.mass.astype(np.float32), dtype=float, device=device)
    fixed = wp.array(fem.fixed.astype(np.int32), dtype=wp.int32, device=device)
    fixed_positions = wp.array(fem.position.astype(np.float32), dtype=wp.vec3, device=device)
    plus_nodes = wp.array(fem.gap_plus_nodes.astype(np.int32), dtype=wp.int32, device=device)
    minus_nodes = wp.array(fem.gap_minus_nodes.astype(np.int32), dtype=wp.int32, device=device)
    gap_normals = wp.array(fem.gap_normals.astype(np.float32), dtype=wp.vec3, device=device)
    gap_area = wp.array(fem.gap_area.astype(np.float32), dtype=float, device=device)
    released_wp = wp.array(solver.released.astype(np.int32), dtype=wp.int32, device=device)
    node_count = len(fem.position)
    element_count = len(fem.tetrahedra)
    force_x = wp.zeros(node_count, dtype=float, device=device)
    force_y = wp.zeros(node_count, dtype=float, device=device)
    force_z = wp.zeros(node_count, dtype=float, device=device)
    jacobian = wp.zeros(element_count, dtype=float, device=device)
    material_cfg = base["material"]
    fracture = base["fracture"]
    qs = moving["quasi_static_solver"]
    youngs = float(material_cfg["youngs_modulus_pa"])
    poisson = float(material_cfg["poisson_ratio"])
    shear = youngs / (2.0 * (1.0 + poisson))
    lame = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    dt = float(qs["pseudo_time_step_s"])
    decay = math.exp(-dt / float(material_cfg["prony_time_constant_s"]))
    damping = math.exp(-float(qs["velocity_damping_per_s"]) * dt)

    def step(blade_center: np.ndarray | None):
        wp.launch(_clear_force, dim=node_count, inputs=[force_x, force_y, force_z], device=device)
        wp.launch(
            _neo_hookean_prony_force, dim=element_count,
            inputs=[positions, tetrahedra, inverse_rest, gradients, volumes, history, previous,
                    force_x, force_y, force_z, jacobian, shear, lame,
                    float(material_cfg["prony_relaxation_fraction"]), decay], device=device,
        )
        center = wp.vec3(0.0, 0.0, 0.0) if blade_center is None else wp.vec3(*map(float, blade_center))
        wp.launch(
            _moving_cohesive_and_wedge_force, dim=len(fem.gap_plus_nodes),
            inputs=[positions, velocities, plus_nodes, minus_nodes, gap_normals, gap_area, released_wp,
                    force_x, force_y, force_z, float(fracture["penalty_stiffness_pa_m"]),
                    float(fracture["compression_stiffness_pa_m"]), float(fracture["cohesive_viscosity_pa_s_m"]),
                    center, int(blade_center is not None), float(qs["blade_wedge_half_width_m"]),
                    float(qs["blade_wedge_target_gap_m"]), float(qs["blade_wedge_stiffness_pa_m"]),
                    float(qs["blade_wedge_peak_traction_pa"])], device=device,
        )
        wp.launch(
            _integrate_nodes, dim=node_count,
            inputs=[positions, velocities, masses, fixed, fixed_positions,
                    force_x, force_y, force_z, dt, damping], device=device,
        )

    poses = _path_poses(moving, curved)
    work = _work_channels(base, moving)
    frames = [positions.numpy().astype(np.float32)]
    releases = [solver.released.copy()]
    centers: list[np.ndarray | None] = [np.asarray(poses[0].center_m)]
    directions = [np.asarray(poses[0].velocity_m_s) / np.linalg.norm(poses[0].velocity_m_s)]
    events = [0]
    segments = [0]
    labels = ["intact / blade entry"]
    for segment, (start, end) in enumerate(zip(poses[:-1], poses[1:], strict=True), start=1):
        solver.advance_blade(segment - 1, start, end, work)
        released_wp.assign(solver.released.astype(np.int32))
        center = np.asarray(end.center_m, dtype=np.float64)
        for _ in range(int(qs["relaxation_steps_per_segment"])):
            step(center)
        wp.synchronize_device(device)
        frames.append(positions.numpy().astype(np.float32))
        releases.append(solver.released.copy())
        centers.append(center)
        direction = np.asarray(end.velocity_m_s, dtype=np.float64)
        directions.append(direction / np.linalg.norm(direction))
        events.append(solver.field.fracture_event_count)
        segments.append(segment)
        labels.append("energy-gated cutting")
    post_steps = int(qs["post_cut_relaxation_steps"])
    for post in range(1, 13):
        target = int(round(post * post_steps / 12))
        previous_target = int(round((post - 1) * post_steps / 12))
        for _ in range(target - previous_target):
            step(None)
        wp.synchronize_device(device)
        frames.append(positions.numpy().astype(np.float32))
        releases.append(solver.released.copy())
        centers.append(None)
        directions.append(directions[-1])
        events.append(solver.field.fracture_event_count)
        segments.append(len(poses) - 1)
        labels.append("viscoelastic relaxation")
    trace_sha = hashlib.sha256(json.dumps(solver.event_trace, separators=(",", ":")).encode()).hexdigest()
    return solver, np.asarray(frames), releases, centers, directions, events, segments, labels, trace_sha, moving


def _set_triangles(surface, triangles: np.ndarray) -> None:
    surface.GetFaceVertexCountsAttr().Set([3] * len(triangles))
    surface.GetFaceVertexIndicesAttr().Set(triangles.astype(np.int32).reshape(-1).tolist())


def _weld_unreleased_exterior(
    points: np.ndarray,
    triangles: np.ndarray,
    plus_nodes: np.ndarray,
    minus_nodes: np.ndarray,
    released: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hide latent cohesive enrichment until its pair actually fractures."""
    render_points = points.copy()
    render_triangles = triangles.copy()
    for plus, minus, is_released in zip(plus_nodes, minus_nodes, released, strict=True):
        if is_released:
            continue
        plus = int(plus)
        minus = int(minus)
        midpoint = 0.5 * (render_points[plus] + render_points[minus])
        render_points[plus] = midpoint
        render_points[minus] = midpoint
        render_triangles[render_triangles == minus] = plus
    nondegenerate = np.asarray(
        [triangle for triangle in render_triangles if len(set(map(int, triangle))) == 3],
        dtype=np.int32,
    )
    return render_points, nondegenerate.reshape((-1, 3))


def main() -> None:
    solver, trajectory, releases, centers, directions, event_counts, segments, labels, trace_sha, moving = _cuda_moving_trajectory()
    fem = solver.mesh
    final_points = trajectory[-1]
    final_jump = (
        final_points[fem.gap_plus_nodes] - final_points[fem.gap_minus_nodes]
    )
    final_opening = np.maximum(
        np.sum(final_jump * fem.gap_normals, axis=1), 0.0
    )
    boundary_tolerance = float(
        moving["boundary_entry_exit"]["boundary_pair_tolerance_m"]
    )
    gap_x = fem.gap_rest_points[:, 0]
    entry_boundary = np.isclose(
        gap_x, float(moving["path"]["start_x_m"]), rtol=0.0,
        atol=boundary_tolerance,
    )
    exit_boundary = np.isclose(
        gap_x, float(moving["path"]["end_x_m"]), rtol=0.0,
        atol=boundary_tolerance,
    )
    entry_boundary_gap = float(np.mean(final_opening[entry_boundary]))
    exit_boundary_gap = float(np.mean(final_opening[exit_boundary]))
    boundary_opening_passed = (
        bool(np.any(entry_boundary))
        and bool(np.any(exit_boundary))
        and entry_boundary_gap
        >= float(moving["qualification"]["minimum_entry_boundary_mean_gap_m"])
        and exit_boundary_gap
        >= float(moving["qualification"]["minimum_exit_boundary_mean_gap_m"])
    )
    if not boundary_opening_passed:
        raise RuntimeError("CUDA render trajectory failed boundary-opening gates")
    boundary = boundary_triangles(fem.tetrahedra)
    wound_all = np.concatenate(tuple(fem.wound_triangles_by_side.values()), axis=0)
    wound_keys = {tuple(sorted(map(int, triangle))) for triangle in wound_all}
    outer = np.asarray([triangle for triangle in boundary if tuple(sorted(map(int, triangle))) not in wound_keys], dtype=np.int32)
    node_to_pair = solver.node_to_pair
    plus_nodes = fem.gap_plus_nodes
    minus_nodes = fem.gap_minus_nodes

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args.device, gravity=(0.0, 0.0, 0.0)))
    sim_utils.GroundPlaneCfg(color=(0.025, 0.029, 0.035)).func("/World/Ground", sim_utils.GroundPlaneCfg(color=(0.025, 0.029, 0.035)))
    dome = sim_utils.DomeLightCfg(intensity=1550.0, color=(0.72, 0.78, 0.86)); dome.func("/World/Dome", dome)
    key = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 0.82, 0.76), angle=20.0); key.func("/World/Key", key, translation=(0.5, 0.5, 1.0))
    fill = sim_utils.DistantLightCfg(intensity=1200.0, color=(0.62, 0.78, 1.0), angle=28.0); fill.func("/World/Fill", fill, translation=(-0.5, 0.3, 0.6))
    table = sim_utils.CuboidCfg(size=(0.085, 0.060, 0.008), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.045, 0.075, 0.082), metallic=0.04, roughness=0.34))
    table.func("/World/ProcedureTable", table, translation=(0.0, 0.0, -0.0075))
    tissue_mat = material("/World/Looks/Tissue", (0.78, 0.34, 0.31), 0.62)
    wound_mat = material("/World/Looks/Wound", (0.34, 0.018, 0.024), 0.46)
    exterior_mesh = mesh("/World/Tissue/Exterior", trajectory[0], np.concatenate((outer, wound_all), axis=0), tissue_mat)
    wound_mesh = mesh("/World/Tissue/ReleasedWound", trajectory[0], np.empty((0, 3), dtype=np.int32), wound_mat)
    blade_cfg = sim_utils.CuboidCfg(size=(0.0055, 0.00055, 0.018), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.48, 0.52), metallic=0.82, roughness=0.19))
    blade_cfg.func("/World/ScalpelBlade", blade_cfg, translation=tuple(centers[0]))
    edge_cfg = sim_utils.CylinderCfg(radius=0.00018, height=0.018, axis="Z", visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.76, 0.78), metallic=0.92, roughness=0.12))
    edge_cfg.func("/World/ScalpelEdge", edge_cfg, translation=tuple(centers[0]))
    handle_cfg = sim_utils.CylinderCfg(radius=0.0017, height=0.020, axis="Z", visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.12, 0.14), metallic=0.25, roughness=0.35))
    handle_cfg.func("/World/ScalpelHandle", handle_cfg, translation=(centers[0][0], centers[0][1], 0.018))
    camera = Camera(CameraCfg(prim_path="/World/DocumentationCamera", update_period=0.0, height=args.height, width=args.width, data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=38.0, focus_distance=0.075, horizontal_aperture=22.0, clipping_range=(0.005, 2.0))))
    sim.reset()
    camera.set_world_poses_from_view(torch.tensor([[0.047, 0.052, 0.035]], device=args.device), torch.tensor([[0.0, 0.0015, 0.0]], device=args.device))
    for _ in range(18):
        sim.step(render=True); camera.update(sim.get_physics_dt(), force_recompute=True)

    rendered = []
    for index, (points, released, center, direction, event_count, segment, label) in enumerate(
        zip(trajectory, releases, centers, directions, event_counts, segments, labels, strict=True)
    ):
        print(f"Rendering moving-scalpel CUDA frame {index + 1}/{len(trajectory)}: {label}", flush=True)
        released_triangles = []
        for triangle in wound_all:
            is_released = all(released[node_to_pair[int(node)]] for node in triangle)
            if is_released:
                released_triangles.append(triangle)
        released_array = np.asarray(released_triangles, dtype=np.int32).reshape((-1, 3))
        render_points, render_outer = _weld_unreleased_exterior(
            points, outer, plus_nodes, minus_nodes, released
        )
        set_points(exterior_mesh, render_points); set_points(wound_mesh, points)
        # Latent cohesive sheets are a mechanical enrichment, not visible
        # geometry. Unreleased pairs share a render vertex and become two-sided
        # wound geometry only after the fracture authority releases them.
        _set_triangles(exterior_mesh, render_outer)
        _set_triangles(wound_mesh, released_array)
        if center is None:
            hidden = np.asarray((0.0, 0.0, 0.06))
            set_pose("/World/ScalpelEdge", hidden); set_pose("/World/ScalpelBlade", hidden); set_pose("/World/ScalpelHandle", hidden)
        else:
            yaw = math.atan2(float(direction[1]), float(direction[0]))
            body_center = center - 0.0024 * direction
            set_pose("/World/ScalpelEdge", center)
            set_pose("/World/ScalpelBlade", body_center, yaw)
            set_pose("/World/ScalpelHandle", np.asarray((body_center[0], body_center[1], 0.018)), yaw)
        for _ in range(2):
            sim.step(render=True); camera.update(sim.get_physics_dt(), force_recompute=True)
        rendered.append(_annotate(rgb_image(camera.data.output["rgb"]), label, segment, 64, event_count, int(np.count_nonzero(released))))
    held = [rendered[0]] * args.hold_frames + rendered + [rendered[-1]] * args.hold_frames
    args.output.parent.mkdir(parents=True, exist_ok=True)
    held[0].save(args.output, save_all=True, append_images=held[1:], duration=95, loop=0, optimize=True, disposal=2)
    with Image.open(args.output) as gif:
        encoded_frames = int(gif.n_frames)
    receipt_path = args.receipt or args.output.with_suffix(".json")
    payload = {
        "schema": "dr.anmar.moving-scalpel-cuda-render-receipt.v1",
        "output": str(args.output), "gif_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "requested_frame_count": len(held), "encoded_frame_count": encoded_frames,
        "trajectory_frame_count": len(trajectory), "width": args.width, "height": args.height,
        "warp_device": "cuda:0", "moving_profile": moving["id"], "path_segments": 64,
        "fracture_event_count": event_counts[-1], "released_pair_count": int(np.count_nonzero(releases[-1])),
        "retained_anchor_node_count": int(np.count_nonzero(fem.fixed)),
        "entry_boundary_pair_count": int(np.count_nonzero(entry_boundary)),
        "exit_boundary_pair_count": int(np.count_nonzero(exit_boundary)),
        "entry_boundary_mean_gap_m": entry_boundary_gap,
        "exit_boundary_mean_gap_m": exit_boundary_gap,
        "boundary_opening_gates_passed": boundary_opening_passed,
        "event_trace_sha256": trace_sha, "cpu_event_trace_match": trace_sha == "3dcf133190cd7d365b347a98c815a75bc34eb644573c7737dae8db81e493083d",
        "node_count": len(fem.position), "tetrahedron_count": len(fem.tetrahedra),
        "displacement_exaggeration": 1.0, "generated_imagery": False,
        "real_time_transient": False, "biomechanical_validation": False, "clinical_validation": False,
    }
    receipt_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); exit_code = 1
    else:
        exit_code = 0
    try:
        simulation_app.close()
    finally:
        raise SystemExit(exit_code)
