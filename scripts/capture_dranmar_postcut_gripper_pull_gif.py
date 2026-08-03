#!/usr/bin/env python3
"""Render the qualified post-cut bilateral gripper pull in Isaac Lab."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output", type=Path,
    default=ROOT / "docs/media/dranmar-postcut-gripper-pull.gif",
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
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from PIL import Image, ImageDraw

scripts_path = str(ROOT / "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from dr_anmar_postcut_gripper_pull_fem import (
    _run_once,
    load_gripper_profile,
)
from dr_anmar_tissue_render_utils import (
    boundary_triangles,
    font,
    material,
    mesh,
    rgb_image,
    set_points,
    set_pose,
)


def _set_triangles(surface, triangles: np.ndarray) -> None:
    surface.GetFaceVertexCountsAttr().Set([3] * len(triangles))
    surface.GetFaceVertexIndicesAttr().Set(
        triangles.astype(np.int32).reshape(-1).tolist()
    )


def _annotate(
    image: Image.Image,
    phase: str,
    top_contacts: int,
    bottom_contacts: int,
    lateral_reaction_n: float,
) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, canvas.width, 64), fill=(8, 13, 19, 224))
    draw.rectangle((0, canvas.height - 52, canvas.width, canvas.height), fill=(8, 13, 19, 224))
    draw.text(
        (22, 13), "Dr.Anmar · Post-cut flap pull",
        font=font(21, bold=True), fill=(250, 245, 242, 255),
    )
    label = phase.replace("_", " ").upper()
    box = draw.textbbox((0, 0), label, font=font(14, bold=True))
    label_width = box[2] - box[0]
    draw.rounded_rectangle(
        (canvas.width - label_width - 48, 15, canvas.width - 20, 48),
        radius=11, fill=(49, 103, 125, 240),
    )
    draw.text(
        (canvas.width - label_width - 34, 22), label,
        font=font(14, bold=True), fill=(255, 255, 255, 255),
    )
    draw.text(
        (22, canvas.height - 41),
        f"BILATERAL CONTACT {top_contacts}+{bottom_contacts} NODES · LATERAL REACTION {lateral_reaction_n:.3f} N",
        font=font(13, bold=True), fill=(183, 224, 237, 255),
    )
    draw.text(
        (22, canvas.height - 22),
        "POST-CUT FEM · FRICTION-LIMITED JAW CONTACT · 0× DISPLACEMENT EXAGGERATION",
        font=font(12), fill=(222, 229, 234, 255),
    )
    return canvas.convert("RGB")


def main() -> None:
    profile = load_gripper_profile()
    captured: list[dict] = []
    stride = {
        "close": 40,
        "custody": 60,
        "pull": 32,
        "lateral_hold": 80,
        "release": 40,
        "recovery": 140,
    }

    def capture(phase, step, count, solver, gripper, contact):
        if step == 0 or step + 1 == count or step % stride[phase] == 0:
            captured.append({
                "phase": phase,
                "points": solver.position.astype(np.float32).copy(),
                "center_xy": gripper.center_xy.copy(),
                "top_plane_z": gripper.top_plane_z,
                "bottom_plane_z": gripper.bottom_plane_z,
                "top_count": contact.top_count,
                "bottom_count": contact.bottom_count,
                "lateral_reaction_n": contact.lateral_reaction_n,
            })

    result = _run_once(profile, capture=capture)
    if not all(result["gates"].values()):
        failed = [name for name, passed in result["gates"].items() if not passed]
        raise RuntimeError(f"Refusing to render failed gripper mechanics: {failed}")
    solver = result["solver"]
    fem = solver.mesh
    boundary = boundary_triangles(fem.tetrahedra)
    wound_all = np.concatenate(tuple(fem.wound_triangles_by_side.values()), axis=0)
    wound_keys = {tuple(sorted(map(int, triangle))) for triangle in wound_all}
    outer = np.asarray(
        [triangle for triangle in boundary if tuple(sorted(map(int, triangle))) not in wound_keys],
        dtype=np.int32,
    )

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 60.0, device=args.device, gravity=(0.0, 0.0, 0.0)
        )
    )
    sim_utils.GroundPlaneCfg(color=(0.025, 0.029, 0.035)).func(
        "/World/Ground", sim_utils.GroundPlaneCfg(color=(0.025, 0.029, 0.035))
    )
    sim_utils.DomeLightCfg(intensity=1550.0, color=(0.72, 0.78, 0.86)).func(
        "/World/Dome", sim_utils.DomeLightCfg(intensity=1550.0, color=(0.72, 0.78, 0.86))
    )
    sim_utils.DistantLightCfg(
        intensity=3000.0, color=(1.0, 0.82, 0.76), angle=20.0
    ).func("/World/Key", sim_utils.DistantLightCfg(
        intensity=3000.0, color=(1.0, 0.82, 0.76), angle=20.0
    ), translation=(0.5, 0.5, 1.0))
    table = sim_utils.CuboidCfg(
        size=(0.085, 0.070, 0.008),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.045, 0.075, 0.082), metallic=0.04, roughness=0.34
        ),
    )
    table.func("/World/ProcedureTable", table, translation=(0.0, 0.0, -0.0095))
    tissue_mat = material("/World/Looks/Tissue", (0.78, 0.34, 0.31), 0.62)
    wound_mat = material("/World/Looks/Wound", (0.34, 0.018, 0.024), 0.46)
    exterior_mesh = mesh("/World/Tissue/Exterior", captured[0]["points"], outer, tissue_mat)
    wound_mesh = mesh("/World/Tissue/Wound", captured[0]["points"], wound_all, wound_mat)
    _set_triangles(exterior_mesh, outer)
    _set_triangles(wound_mesh, wound_all)
    pad_half = np.asarray(profile["gripper"]["pad_half_extent_m"], dtype=np.float64)
    pad_thickness = 0.0012
    jaw_visual = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.18, 0.29, 0.34), metallic=0.82, roughness=0.20
    )
    pad_cfg = sim_utils.CuboidCfg(
        size=(2.0 * pad_half[0], 2.0 * pad_half[1], pad_thickness),
        visual_material=jaw_visual,
    )
    first = captured[0]
    top_center = (
        first["center_xy"][0], first["center_xy"][1],
        first["top_plane_z"] + 0.5 * pad_thickness,
    )
    bottom_center = (
        first["center_xy"][0], first["center_xy"][1],
        first["bottom_plane_z"] - 0.5 * pad_thickness,
    )
    pad_cfg.func("/World/Gripper/TopJaw", pad_cfg, translation=top_center)
    pad_cfg.func("/World/Gripper/BottomJaw", pad_cfg, translation=bottom_center)
    shank_cfg = sim_utils.CylinderCfg(
        radius=0.0017, height=0.018, axis="Z", visual_material=jaw_visual
    )
    shank_cfg.func(
        "/World/Gripper/Shank", shank_cfg,
        translation=(top_center[0], top_center[1], top_center[2] + 0.009),
    )
    camera = Camera(CameraCfg(
        prim_path="/World/DocumentationCamera", update_period=0.0,
        height=args.height, width=args.width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.0, focus_distance=0.075,
            horizontal_aperture=22.0, clipping_range=(0.005, 2.0),
        ),
    ))
    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([[0.050, 0.058, 0.034]], device=args.device),
        torch.tensor([[0.0, 0.0035, 0.0]], device=args.device),
    )
    for _ in range(18):
        sim.step(render=True)
        camera.update(sim.get_physics_dt(), force_recompute=True)

    rendered = []
    for index, frame in enumerate(captured):
        print(
            f"Rendering post-cut gripper frame {index + 1}/{len(captured)}: {frame['phase']}",
            flush=True,
        )
        set_points(exterior_mesh, frame["points"])
        set_points(wound_mesh, frame["points"])
        top_z = frame["top_plane_z"] + 0.5 * pad_thickness
        bottom_z = frame["bottom_plane_z"] - 0.5 * pad_thickness
        center_xy = frame["center_xy"]
        set_pose("/World/Gripper/TopJaw", np.asarray((center_xy[0], center_xy[1], top_z)))
        set_pose("/World/Gripper/BottomJaw", np.asarray((center_xy[0], center_xy[1], bottom_z)))
        set_pose("/World/Gripper/Shank", np.asarray((center_xy[0], center_xy[1], top_z + 0.009)))
        for _ in range(2):
            sim.step(render=True)
            camera.update(sim.get_physics_dt(), force_recompute=True)
        rendered.append(_annotate(
            rgb_image(camera.data.output["rgb"]), frame["phase"],
            frame["top_count"], frame["bottom_count"], frame["lateral_reaction_n"],
        ))
    held = [rendered[0]] * args.hold_frames + rendered + [rendered[-1]] * args.hold_frames
    args.output.parent.mkdir(parents=True, exist_ok=True)
    held[0].save(
        args.output, save_all=True, append_images=held[1:],
        duration=105, loop=0, optimize=True, disposal=2,
    )
    with Image.open(args.output) as gif:
        encoded_frames = int(gif.n_frames)
    receipt_path = args.receipt or args.output.with_suffix(".json")
    payload = {
        "schema": "dr.anmar.postcut-gripper-pull-render-receipt.v1",
        "output": str(args.output),
        "gif_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "requested_frame_count": len(held),
        "encoded_frame_count": encoded_frames,
        "mechanics_frame_count": len(captured),
        "width": args.width, "height": args.height,
        "render_device": args.device,
        "mechanics_backend": "numpy_float64_reference_fem",
        "mechanics_qualified": True,
        "cut_fracture_event_count": result["cut_events"],
        "released_pair_count": int(np.count_nonzero(solver.released)),
        "pull_bilateral_custody_fraction": result["custody_fraction"],
        "gripped_flap_lateral_displacement_m": result["gripped_displacement"],
        "differential_flap_displacement_m": result["differential"],
        "local_wound_gap_increase_m": result["gap_increase"],
        "maximum_contact_penetration_m": result["peak_penetration"],
        "minimum_jacobian": solver.minimum_jacobian,
        "inversion_observation_count": solver.inversion_observations,
        "topology_event_delta": solver.field.fracture_event_count - result["cut_events"],
        "recovery_residual_m": result["recovery"],
        "trace_sha256": result["trace_sha"],
        "displacement_exaggeration": 1.0,
        "generated_imagery": False,
        "biomechanical_validation": False,
        "clinical_validation": False,
    }
    receipt_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
