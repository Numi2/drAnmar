#!/usr/bin/env python3
"""Render the real Warp-CUDA curved cut-cell trajectory as an Isaac Lab GIF.

The script advances the qualified curved tissue state with the exact Warp
kernels used by ``dr_anmar_dynamic_curved_cut_warp.py`` and updates USD meshes
from sampled CUDA positions. It does not exaggerate displacement or synthesize
intermediate geometry.
"""

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
    "--output",
    type=Path,
    default=ROOT / "docs/media/dranmar-cuttable-tissue-curved-cuda.gif",
)
parser.add_argument("--receipt", type=Path)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=600)
parser.add_argument("--frames", type=int, default=64)
parser.add_argument("--hold-frames", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import numpy as np
import omni.usd
import torch
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

scripts_path = str(ROOT / "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from dr_anmar_cuttable_tissue_solver import load_profile
from dr_anmar_cuttable_tissue_warp import _clear_force, _neo_hookean_prony_force
from dr_anmar_dynamic_curved_cut_fem import (
    DEFAULT_CURVED_PROFILE_PATH,
    TET_FACES,
    _build_settled_mesh,
    _level_gradient,
    load_curved_profile,
)
from dr_anmar_dynamic_curved_cut_warp import (
    _integrate_nodes,
    _wound_compression_force,
    _wound_opening_force,
)


def _font(size: int, *, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _rgb_image(value) -> Image.Image:
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _annotate(image: Image.Image, phase: str, step: int, total_steps: int) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, canvas.width, 62), fill=(9, 14, 20, 220))
    draw.rectangle((0, canvas.height - 48, canvas.width, canvas.height), fill=(9, 14, 20, 220))
    draw.text(
        (22, 13),
        "Dr.Anmar · Curved Cut-Cell Tissue",
        font=_font(21, bold=True),
        fill=(249, 244, 242, 255),
    )
    label = phase.upper()
    box = draw.textbbox((0, 0), label, font=_font(15, bold=True))
    width = box[2] - box[0]
    draw.rounded_rectangle(
        (canvas.width - width - 48, 14, canvas.width - 20, 47),
        radius=11,
        fill=(153, 50, 57, 238),
    )
    draw.text(
        (canvas.width - width - 34, 21),
        label,
        font=_font(15, bold=True),
        fill=(255, 255, 255, 255),
    )
    draw.text(
        (22, canvas.height - 37),
        "REAL WARP CUDA TRAJECTORY · ISAAC LAB RENDER · 0× DISPLACEMENT EXAGGERATION",
        font=_font(13, bold=True),
        fill=(245, 190, 181, 255),
    )
    progress = 0.0 if total_steps <= 0 else step / total_steps
    x0, x1 = canvas.width - 230, canvas.width - 22
    draw.rounded_rectangle((x0, canvas.height - 30, x1, canvas.height - 17), radius=6, fill=(52, 61, 70, 230))
    draw.rounded_rectangle((x0, canvas.height - 30, x0 + int((x1 - x0) * progress), canvas.height - 17), radius=6, fill=(219, 101, 93, 245))
    return canvas.convert("RGB")


def _boundary_triangles(tetrahedra: np.ndarray) -> np.ndarray:
    counts: dict[tuple[int, int, int], int] = {}
    oriented: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tet in tetrahedra:
        for face in TET_FACES:
            triangle = tuple(int(tet[index]) for index in face)
            key = tuple(sorted(triangle))
            counts[key] = counts.get(key, 0) + 1
            oriented.setdefault(key, triangle)
    return np.asarray([oriented[key] for key, count in counts.items() if count == 1], dtype=np.int32)


def _material(path: str, color: tuple[float, float, float], roughness: float):
    stage = omni.usd.get_context().get_stage()
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _mesh(path: str, points: np.ndarray, triangles: np.ndarray, material) -> UsdGeom.Mesh:
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr().Set(triangles.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr().Set(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def _set_points(mesh: UsdGeom.Mesh, points: np.ndarray) -> None:
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))


def _cuda_trajectory(frame_count: int):
    curved = load_curved_profile(DEFAULT_CURVED_PROFILE_PATH)
    base = load_profile(ROOT / curved["base_profile"])
    solver = _build_settled_mesh(base, curved)
    device = "cuda:0"
    wp.init()
    if not wp.get_device(device).is_cuda:
        raise RuntimeError("Real capture requires a CUDA Warp device")

    positions = wp.array(solver.position.astype(np.float32), dtype=wp.vec3, device=device)
    velocities = wp.array(solver.velocity.astype(np.float32), dtype=wp.vec3, device=device)
    tetrahedra = wp.array(solver.tetrahedra.astype(np.int32), dtype=wp.vec4i, device=device)
    inverse_rest = wp.array(solver.dm_inverse.astype(np.float32), dtype=wp.mat33, device=device)
    gradients = wp.array(solver.shape_gradients.reshape((-1, 3)).astype(np.float32), dtype=wp.vec3, device=device)
    volumes = wp.array(solver.rest_volume.astype(np.float32), dtype=float, device=device)
    history = wp.array(solver.prony_history.astype(np.float32), dtype=wp.mat33, device=device)
    previous = wp.array(solver.previous_elastic_stress.astype(np.float32), dtype=wp.mat33, device=device)
    masses = wp.array(solver.mass.astype(np.float32), dtype=float, device=device)
    fixed = wp.array(solver.fixed.astype(np.int32), dtype=wp.int32, device=device)
    fixed_positions = wp.array(solver.position.astype(np.float32), dtype=wp.vec3, device=device)
    plus_nodes = wp.array(solver.gap_plus_nodes.astype(np.int32), dtype=wp.int32, device=device)
    minus_nodes = wp.array(solver.gap_minus_nodes.astype(np.int32), dtype=wp.int32, device=device)
    gap_normals = wp.array(solver.gap_normals.astype(np.float32), dtype=wp.vec3, device=device)
    gap_area = wp.array(solver.gap_area.astype(np.float32), dtype=float, device=device)

    wound_triangles = np.concatenate(
        (solver.wound_triangles_by_side[-1], solver.wound_triangles_by_side[1]), axis=0
    ).astype(np.int32)
    wound_sides = np.concatenate(
        (-np.ones(len(solver.wound_triangles_by_side[-1]), dtype=np.int32),
         np.ones(len(solver.wound_triangles_by_side[1]), dtype=np.int32))
    )
    wound_centroids = np.mean(solver.rest[wound_triangles], axis=1)
    wound_normals = _level_gradient(wound_centroids, curved["implicit_cut"]).astype(np.float32)
    wound_triangles_wp = wp.array(wound_triangles, dtype=wp.vec3i, device=device)
    wound_sides_wp = wp.array(wound_sides, dtype=wp.int32, device=device)
    wound_normals_wp = wp.array(wound_normals, dtype=wp.vec3, device=device)

    node_count = len(solver.position)
    element_count = len(solver.tetrahedra)
    force_x = wp.zeros(node_count, dtype=float, device=device)
    force_y = wp.zeros(node_count, dtype=float, device=device)
    force_z = wp.zeros(node_count, dtype=float, device=device)
    jacobian = wp.zeros(element_count, dtype=float, device=device)
    material = base["material"]
    fracture = base["fracture"]
    youngs = float(material["youngs_modulus_pa"])
    poisson = float(material["poisson_ratio"])
    shear = youngs / (2.0 * (1.0 + poisson))
    lame = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    dt = float(curved["solver"]["cut_time_step_s"])
    decay = math.exp(-dt / float(material["prony_time_constant_s"]))
    damping = math.exp(-float(curved["solver"]["velocity_damping_per_s"]) * dt)
    phases = (
        ("cohesive release / elastic opening", float(curved["solver"]["opening_load_s"]), float(curved["solver"]["opening_traction_pa"])),
        ("viscoelastic relaxation", float(curved["solver"]["post_load_s"]), 0.0),
    )
    total_steps = sum(int(round(duration / dt)) for _, duration, _ in phases)
    sample_steps = set(np.linspace(0, total_steps, frame_count, dtype=int).tolist())
    frames = [positions.numpy().astype(np.float32)]
    labels = ["settled / intact contact"]
    steps = [0]
    global_step = 0
    for phase, duration, traction in phases:
        for _ in range(int(round(duration / dt))):
            wp.launch(_clear_force, dim=node_count, inputs=[force_x, force_y, force_z], device=device)
            wp.launch(
                _neo_hookean_prony_force,
                dim=element_count,
                inputs=[positions, tetrahedra, inverse_rest, gradients, volumes, history, previous,
                        force_x, force_y, force_z, jacobian, shear, lame,
                        float(material["prony_relaxation_fraction"]), decay],
                device=device,
            )
            wp.launch(
                _wound_compression_force,
                dim=len(solver.gap_plus_nodes),
                inputs=[positions, velocities, plus_nodes, minus_nodes, gap_normals, gap_area,
                        force_x, force_y, force_z, float(fracture["compression_stiffness_pa_m"]),
                        float(fracture["cohesive_viscosity_pa_s_m"])],
                device=device,
            )
            if traction > 0.0:
                wp.launch(
                    _wound_opening_force,
                    dim=len(wound_triangles),
                    inputs=[positions, wound_triangles_wp, wound_sides_wp, wound_normals_wp,
                            force_x, force_y, force_z, traction],
                    device=device,
                )
            wp.launch(
                _integrate_nodes,
                dim=node_count,
                inputs=[positions, velocities, masses, fixed, fixed_positions,
                        force_x, force_y, force_z, dt, damping],
                device=device,
            )
            global_step += 1
            if global_step in sample_steps:
                wp.synchronize_device(device)
                frames.append(positions.numpy().astype(np.float32))
                labels.append(phase)
                steps.append(global_step)
    if steps[-1] != total_steps:
        wp.synchronize_device(device)
        frames.append(positions.numpy().astype(np.float32))
        labels.append(phases[-1][0])
        steps.append(total_steps)
    return solver, np.asarray(frames), labels, steps, total_steps, curved


def main() -> None:
    if args.width < 480 or args.height < 320:
        raise ValueError("Capture resolution must be at least 480 x 320")
    if args.frames < 24:
        raise ValueError("At least 24 trajectory frames are required")

    solver, trajectory, labels, trajectory_steps, total_steps, curved = _cuda_trajectory(args.frames)
    boundary = _boundary_triangles(solver.tetrahedra)
    wound_all = np.concatenate(tuple(solver.wound_triangles_by_side.values()), axis=0)
    wound_keys = {tuple(sorted(map(int, triangle))) for triangle in wound_all}
    exterior = np.asarray(
        [triangle for triangle in boundary if tuple(sorted(map(int, triangle))) not in wound_keys],
        dtype=np.int32,
    )

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args.device, gravity=(0.0, 0.0, 0.0))
    )
    ground = sim_utils.GroundPlaneCfg(color=(0.025, 0.029, 0.035))
    ground.func("/World/Ground", ground)
    dome = sim_utils.DomeLightCfg(intensity=1550.0, color=(0.72, 0.78, 0.86))
    dome.func("/World/Dome", dome)
    key = sim_utils.DistantLightCfg(intensity=2900.0, color=(1.0, 0.82, 0.76), angle=20.0)
    key.func("/World/Key", key, translation=(0.5, 0.5, 1.0))
    fill = sim_utils.DistantLightCfg(intensity=1200.0, color=(0.62, 0.78, 1.0), angle=28.0)
    fill.func("/World/Fill", fill, translation=(-0.5, 0.3, 0.6))
    table = sim_utils.CuboidCfg(
        size=(0.085, 0.060, 0.008),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.045, 0.075, 0.082), metallic=0.04, roughness=0.34
        ),
    )
    table.func("/World/ProcedureTable", table, translation=(0.0, 0.0, -0.0075))

    tissue_material = _material("/World/Looks/Tissue", (0.78, 0.34, 0.31), 0.62)
    wound_negative_material = _material("/World/Looks/WoundNegative", (0.26, 0.018, 0.025), 0.48)
    wound_positive_material = _material("/World/Looks/WoundPositive", (0.48, 0.035, 0.038), 0.52)
    exterior_mesh = _mesh("/World/Tissue/Exterior", trajectory[0], exterior, tissue_material)
    negative_mesh = _mesh(
        "/World/Tissue/WoundNegative",
        trajectory[0],
        solver.wound_triangles_by_side[-1],
        wound_negative_material,
    )
    positive_mesh = _mesh(
        "/World/Tissue/WoundPositive",
        trajectory[0],
        solver.wound_triangles_by_side[1],
        wound_positive_material,
    )
    camera = Camera(
        CameraCfg(
            prim_path="/World/DocumentationCamera",
            update_period=0.0,
            height=args.height,
            width=args.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=38.0,
                focus_distance=0.075,
                horizontal_aperture=22.0,
                clipping_range=(0.005, 2.0),
            ),
        )
    )
    sim.reset()
    eye = torch.tensor([[0.047, 0.052, 0.033]], device=args.device)
    target = torch.tensor([[0.0, 0.0015, -0.0002]], device=args.device)
    camera.set_world_poses_from_view(eye, target)
    for _ in range(18):
        sim.step(render=True)
        camera.update(sim.get_physics_dt(), force_recompute=True)

    rendered: list[Image.Image] = []
    for frame_index, (points, phase, step) in enumerate(
        zip(trajectory, labels, trajectory_steps, strict=True)
    ):
        print(f"Rendering CUDA tissue frame {frame_index + 1}/{len(trajectory)}: {phase}", flush=True)
        _set_points(exterior_mesh, points)
        _set_points(negative_mesh, points)
        _set_points(positive_mesh, points)
        for _ in range(2):
            sim.step(render=True)
            camera.update(sim.get_physics_dt(), force_recompute=True)
        rendered.append(_annotate(_rgb_image(camera.data.output["rgb"]), phase, step, total_steps))

    if not rendered:
        raise RuntimeError("No rendered frames were produced")
    held = [rendered[0]] * args.hold_frames + rendered + [rendered[-1]] * args.hold_frames
    args.output.parent.mkdir(parents=True, exist_ok=True)
    held[0].save(
        args.output,
        save_all=True,
        append_images=held[1:],
        duration=90,
        loop=0,
        optimize=True,
        disposal=2,
    )
    with Image.open(args.output) as encoded_gif:
        encoded_frame_count = int(getattr(encoded_gif, "n_frames", 1))
    receipt_path = args.receipt or args.output.with_suffix(".json")
    payload = {
        "schema": "dr.anmar.curved-cut-cuda-render-receipt.v1",
        "output": str(args.output),
        "gif_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "requested_frame_count": len(held),
        "encoded_frame_count": encoded_frame_count,
        "trajectory_frame_count": len(trajectory),
        "width": args.width,
        "height": args.height,
        "warp_device": "cuda:0",
        "source_profile": curved["id"],
        "trajectory_steps": total_steps,
        "node_count": len(solver.position),
        "tetrahedron_count": len(solver.tetrahedra),
        "exterior_triangle_count": len(exterior),
        "wound_triangle_count": len(wound_all),
        "displacement_exaggeration": 1.0,
        "generated_imagery": False,
        "biomechanical_validation": False,
        "clinical_validation": False
    }
    receipt_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        exit_code = 1
    else:
        exit_code = 0
    try:
        simulation_app.close()
    finally:
        raise SystemExit(exit_code)
