#!/usr/bin/env python3
"""Generate the DrAnmar SafePlane Dissection Robot asset family.

This DrAnmar-owned, provider-neutral research asset models a Franka-compatible
end effector for traction-assisted safe-plane dissection in NVIDIA Isaac.
It combines distributed tissue traction, blunt spreading, hydrodissection,
guarded micro-scissors, low-energy dissection, irrigation, smoke evacuation,
protected-structure monitoring, physical adhesion release, and connectivity
verification. It is not clinically validated and is not approved for patient
care.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

VERSION = "0.1.0"
ASSET_NAME = "DrAnmar SafePlane Dissection Robot"
CATALOG_SUBPATH = Path("Props/SurgicalDissection/SafePlaneDissectionRobot")
ROOT_PRIM = "DrAnmarSafePlaneDissectionTool"
STANDALONE_ROOT = "DrAnmarSafePlaneDissectionToolStandalone"
PROXY_ROOT = "DrAnmarSafePlaneDissectionToolRigidProxy"
TISSUE_ROOT = "DrAnmarSafePlaneTissueDemo"
BRIDGE_ROOT = "DrAnmarAdhesionBridge"
VESSEL_ROOT = "DrAnmarProtectedVesselBranch"
NERVE_ROOT = "DrAnmarProtectedNerveBranch"
DUCT_ROOT = "DrAnmarProtectedDuctBranch"
SCISSORS_ROOT = "DrAnmarMicroScissorsCartridge"
HYDRO_PARTICLE_ROOT = "DrAnmarHydrodissectionParticle"
SMOKE_PARTICLE_ROOT = "DrAnmarDissectionSmokeParticle"
BLOOD_PARTICLE_ROOT = "DrAnmarDissectionBloodParticle"
DUCT_FLUID_ROOT = "DrAnmarDuctLeakParticle"

SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[1]
ASSET_ROOT = PACKAGE_ROOT / "assets" / CATALOG_SUBPATH
GLB_ROOT = ASSET_ROOT / "glb"
TEXTURE_ROOT = ASSET_ROOT / "textures"
PREVIEW_ROOT = PACKAGE_ROOT / "previews"
DOCS_ROOT = PACKAGE_ROOT / "docs" / "safeplane_dissection_robot"
EXAMPLE_ROOT = PACKAGE_ROOT / "examples"
EXTENSION_ROOT = PACKAGE_ROOT / "source/extensions/orbit.surgical.assets"
INTEGRATION_PATH = EXTENSION_ROOT / "orbit/surgical/assets/safeplane_dissection_robot.py"
PHYSICS_PROFILE_PATH = PACKAGE_ROOT / "physics_next/surgical-dissection/dranmar-safeplane-dissection-v1.json"

WORK_PLANE_Z = 0.205
FRANKA_HAND_EQUIVALENT_ROTATION_DEG = -45.0
TRACTION_TRAVEL_M = 0.036
PAD_PITCH_DEG = 32.0
PAD_COMPLIANCE_M = 0.008
SPREADER_TRAVEL_M = 0.022
HYDRO_EXTENSION_M = 0.050
HYDRO_PITCH_DEG = 24.0
SCISSOR_EXTENSION_M = 0.052
SCISSOR_GUARD_TRAVEL_M = 0.010
SCISSOR_CLOSE_DEG = 34.0
ENERGY_EXTENSION_M = 0.048
TRACTION_CELL_COUNT_PER_SIDE = 4
ADHESION_BRIDGE_COUNT = 28
PROTECTED_STRUCTURE_NAMES = ("vessel", "nerve", "duct")
HYDRO_JET_COUNT = 7
SUCTION_PORT_COUNT = 10
IRRIGATION_PORT_COUNT = 6
def f(value: float, digits: int = 10) -> str:
    if abs(value) < 1.0e-18:
        value = 0.0
    return f"{float(value):.{digits}g}"


def vec(values: Sequence[float], digits: int = 10) -> str:
    return "(" + ", ".join(f(v, digits) for v in values) + ")"


def quat(values: Sequence[float]) -> str:
    if len(values) != 4:
        raise ValueError("quaternion must use flat wxyz USDA syntax")
    return vec(values)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12 or not math.isfinite(n):
        raise ValueError("cannot normalize vector")
    return v / n


def rotation_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=float))
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.asarray(
        [
            [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
            [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
            [z*x*C-y*s, z*y*C+x*s, z*z*C+c],
        ],
        dtype=float,
    )


def matrix_to_quat_wxyz(m: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(m, dtype=float)
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        q = np.asarray([0.25*s, (m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s])
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
        q = np.asarray([(m[2,1]-m[1,2])/s, 0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s])
    elif m[1,1] > m[2,2]:
        s = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
        q = np.asarray([(m[0,2]-m[2,0])/s, (m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s])
    else:
        s = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
        q = np.asarray([(m[1,0]-m[0,1])/s, (m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return tuple(float(x) for x in q)


def transform(mesh: trimesh.Trimesh, translation=(0.0, 0.0, 0.0), rotation: np.ndarray | None = None, scale=None) -> trimesh.Trimesh:
    mesh = mesh.copy()
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation)
    T[:3, 3] = np.asarray(translation, dtype=float)
    mesh.apply_transform(T)
    if scale is not None:
        mesh.apply_scale(np.asarray(scale, dtype=float))
    return mesh


def box_mesh(size: Sequence[float], center=(0.0, 0.0, 0.0), rotation: np.ndarray | None = None) -> trimesh.Trimesh:
    return transform(trimesh.creation.box(extents=np.asarray(size, dtype=float)), center, rotation)


def cylinder_axis(radius: float, length: float, axis: str = "z", center=(0.0, 0.0, 0.0), sections: int = 48) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    if axis == "x":
        R = rotation_matrix((0, 1, 0), math.pi/2)
    elif axis == "y":
        R = rotation_matrix((1, 0, 0), -math.pi/2)
    elif axis == "z":
        R = np.eye(3)
    else:
        raise ValueError(axis)
    return transform(mesh, center, R)


def ellipsoid_mesh(radii: Sequence[float], center=(0.0, 0.0, 0.0), subdivisions: int = 3) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.apply_scale(np.asarray(radii, dtype=float))
    mesh.apply_translation(np.asarray(center, dtype=float))
    return mesh


def torus_axis(major_radius: float, minor_radius: float, axis: str = "z", center=(0.0, 0.0, 0.0), major_sections=64, minor_sections=16) -> trimesh.Trimesh:
    mesh = trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius, major_sections=major_sections, minor_sections=minor_sections)
    if axis == "x":
        R = rotation_matrix((0, 1, 0), math.pi/2)
    elif axis == "y":
        R = rotation_matrix((1, 0, 0), -math.pi/2)
    else:
        R = np.eye(3)
    return transform(mesh, center, R)


def frustum_axis(radius0: float, radius1: float, height: float, axis: str = "z", center=(0,0,0), sections: int = 48) -> trimesh.Trimesh:
    z0, z1 = -height/2, height/2
    points: list[tuple[float,float,float]] = []
    for z, r in ((z0, radius0), (z1, radius1)):
        for i in range(sections):
            a = 2*math.pi*i/sections
            points.append((r*math.cos(a), r*math.sin(a), z))
    points.extend([(0,0,z0), (0,0,z1)])
    faces: list[tuple[int,int,int]] = []
    for i in range(sections):
        j = (i+1)%sections
        faces += [(i,j,sections+j),(i,sections+j,sections+i),(2*sections,j,i),(2*sections+1,sections+i,sections+j)]
    mesh = trimesh.Trimesh(vertices=np.asarray(points), faces=np.asarray(faces), process=True)
    if axis == "x":
        R = rotation_matrix((0,1,0), math.pi/2)
    elif axis == "y":
        R = rotation_matrix((1,0,0), -math.pi/2)
    else:
        R = np.eye(3)
    return transform(mesh, center, R)


def capsule_between(p0: Sequence[float], p1: Sequence[float], radius: float, sections: int = 24) -> trimesh.Trimesh:
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return transform(trimesh.creation.icosphere(subdivisions=2, radius=radius), p0)
    mesh = trimesh.creation.capsule(radius=radius, height=max(0.0, length-2*radius), count=[sections, sections])
    z = np.asarray([0.0,0.0,1.0])
    d = direction/length
    cross = np.cross(z,d)
    dot = float(np.clip(np.dot(z,d), -1.0, 1.0))
    if np.linalg.norm(cross) <= 1e-12:
        R = np.eye(3) if dot > 0 else rotation_matrix((1,0,0), math.pi)
    else:
        R = rotation_matrix(cross, math.acos(dot))
    return transform(mesh, (p0+p1)/2, R)


def wire_path(points: Sequence[Sequence[float]], radius: float) -> trimesh.Trimesh:
    pts = [np.asarray(p, dtype=float) for p in points]
    parts: list[trimesh.Trimesh] = []
    for a,b in zip(pts[:-1], pts[1:]):
        parts.append(capsule_between(a,b,radius))
    for p in pts[1:-1]:
        parts.append(transform(trimesh.creation.icosphere(subdivisions=2, radius=radius), p))
    return trimesh.util.concatenate(parts)


def wedge_blade_mesh(width_y=0.022, height_z=0.022, thickness_x=0.00075, bevel_z=0.005) -> trimesh.Trimesh:
    x0, x1 = -thickness_x/2, thickness_x/2
    y0, y1 = -width_y/2, width_y/2
    z0, z1 = -height_z/2, height_z/2
    zb = z1 - bevel_z
    vertices = np.asarray([
        (x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
        (x0,y0,zb),(x1,y0,zb),(x1,y1,zb),(x0,y1,zb),
        (0.0,y0,z1),(0.0,y1,z1),
    ],dtype=float)
    faces = np.asarray([
        (0,1,2),(0,2,3),
        (0,4,5),(0,5,1),(3,2,6),(3,6,7),
        (0,3,7),(0,7,4),(1,5,6),(1,6,2),
        (4,7,9),(4,9,8),(5,8,9),(5,9,6),
        (4,8,5),(7,6,9),
    ],dtype=int)
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
    mesh.fix_normals()
    return mesh


def mesh_bounds(meshes: Sequence[trimesh.Trimesh]) -> tuple[np.ndarray,np.ndarray]:
    mins = np.vstack([m.bounds[0] for m in meshes])
    maxs = np.vstack([m.bounds[1] for m in meshes])
    return mins.min(axis=0), maxs.max(axis=0)


def box_mass_properties(meshes: Sequence[trimesh.Trimesh], mass: float) -> dict[str, object]:
    bmin,bmax = mesh_bounds(meshes)
    size = np.maximum(bmax-bmin, 1e-5)
    com = (bmin+bmax)*0.5
    dx,dy,dz = size
    inertia = (mass*(dy*dy+dz*dz)/12, mass*(dx*dx+dz*dz)/12, mass*(dx*dx+dy*dy)/12)
    return {
        "mass_kg": float(mass),
        "center_of_mass_m": [float(x) for x in com],
        "diagonal_inertia_kg_m2": [float(x) for x in inertia],
        "principal_axes_wxyz": [1.0,0.0,0.0,0.0],
        "bounds_min_m": [float(x) for x in bmin],
        "bounds_max_m": [float(x) for x in bmax],
    }


def rounded_bar_mesh(size: Sequence[float], center=(0.0, 0.0, 0.0), radius: float = 0.003) -> trimesh.Trimesh:
    """Create a robust rounded-looking bar without boolean operations."""
    sx, sy, sz = (float(v) for v in size)
    parts = [box_mesh((max(sx - 2 * radius, radius), sy, sz), center)]
    for sign in (-1.0, 1.0):
        x = center[0] + sign * max(0.0, sx / 2.0 - radius)
        parts.append(cylinder_axis(radius, sy, "y", (x, center[1], center[2]), sections=32))
    return trimesh.util.concatenate(parts)


def grid_surface_mesh(
    width: float,
    depth: float,
    nx: int,
    ny: int,
    *,
    z_func,
    center=(0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    points: list[tuple[float, float, float]] = []
    cx, cy, cz = center
    for iy in range(ny):
        y = -depth / 2.0 + depth * iy / (ny - 1)
        for ix in range(nx):
            x = -width / 2.0 + width * ix / (nx - 1)
            points.append((cx + x, cy + y, cz + float(z_func(x, y))))
    faces: list[tuple[int, int, int]] = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix
            b = a + 1
            c = a + nx
            d = c + 1
            if (ix + iy) % 2 == 0:
                faces.extend(((a, b, d), (a, d, c)))
            else:
                faces.extend(((a, b, c), (b, d, c)))
    mesh = trimesh.Trimesh(vertices=np.asarray(points), faces=np.asarray(faces), process=False)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def curved_tube(points: Sequence[Sequence[float]], radius: float, sections: int = 22) -> trimesh.Trimesh:
    return wire_path(points, radius)


def scissor_blade_mesh(length=0.043, width=0.008, thickness=0.0010, curved=True) -> trimesh.Trimesh:
    """Thin guarded microsurgical blade with a rounded heel and tapered tip."""
    x0, x1 = -thickness / 2.0, thickness / 2.0
    y0, y1 = -width / 2.0, width / 2.0
    z0, z1 = 0.0, length
    bend = 0.0025 if curved else 0.0
    vertices = np.asarray(
        [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1 * 0.72), (x1, y0, z1 * 0.72),
            (x1, y1, z1 * 0.72), (x0, y1, z1 * 0.72),
            (0.0, -width * 0.28 + bend, z1), (0.0, width * 0.28 + bend, z1),
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            (0, 1, 2), (0, 2, 3),
            (0, 4, 5), (0, 5, 1),
            (3, 2, 6), (3, 6, 7),
            (0, 3, 7), (0, 7, 4),
            (1, 5, 6), (1, 6, 2),
            (4, 7, 9), (4, 9, 8),
            (5, 8, 9), (5, 9, 6),
            (4, 8, 5), (7, 6, 9),
        ],
        dtype=int,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def fibre_half_mesh(length: float, radius: float, upper: bool, strands: int = 5) -> trimesh.Trimesh:
    parts: list[trimesh.Trimesh] = []
    half = length / 2.0
    z0, z1 = (0.0, -half) if upper else (0.0, half)
    rng = np.random.default_rng(3101 + (1 if upper else 2) + strands)
    for i in range(strands):
        a = 2.0 * math.pi * i / strands
        offset = np.asarray([radius * 1.2 * math.cos(a), radius * 1.2 * math.sin(a), 0.0])
        jitter = rng.normal(0.0, radius * 0.22, 2)
        p0 = offset + np.asarray([jitter[0], jitter[1], z0])
        p1 = 0.35 * offset + np.asarray([-jitter[0] * 0.2, -jitter[1] * 0.2, z1])
        parts.append(capsule_between(p0, p1, radius * 0.34, sections=14))
    return trimesh.util.concatenate(parts)


def ring_sector_mesh(radius: float, tube_radius: float, start: float, end: float, z: float, samples: int = 28) -> trimesh.Trimesh:
    points = [(radius * math.cos(start + (end - start) * i / samples), radius * math.sin(start + (end - start) * i / samples), z) for i in range(samples + 1)]
    return wire_path(points, tube_radius)


@dataclass
class Visual:
    name: str
    mesh: trimesh.Trimesh
    material: str
    labels: tuple[str, ...] = ()


@dataclass
class Collider:
    name: str
    kind: str
    center: tuple[float, float, float]
    size: tuple[float, float, float] | None = None
    radius: float | None = None
    height: float | None = None
    axis: str = "z"
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    physics_material: str = "PolymerPhysics"
    role: str = "collision"
    author_enabled: bool = True
    collision_enabled: bool = True


@dataclass
class Link:
    name: str
    translation: tuple[float, float, float]
    visuals: list[Visual]
    colliders: list[Collider]
    mass_kg: float | None
    labels: tuple[str, ...] = ()
    mass_properties: dict[str, object] | None = field(init=False)

    def __post_init__(self):
        self.mass_properties = None if self.mass_kg is None else box_mass_properties([v.mesh for v in self.visuals], self.mass_kg)


@dataclass
class Joint:
    name: str
    type: str
    body0: str
    body1: str
    axis: str | None
    local_pos0: tuple[float, float, float]
    local_pos1: tuple[float, float, float]
    lower: float | None = None
    upper: float | None = None
    stiffness: float = 0.0
    damping: float = 0.0
    max_force: float = 0.0
    target_velocity: float = 0.0


@dataclass
class BridgeSpec:
    index: int
    position: tuple[float, float, float]
    bridge_class: str
    target: bool
    recommended_mode: str
    mechanical_work_j: float
    hydro_volume_ml: float
    energy_dose_j: float
    nearest_structure: str | None
    clearance_m: float


@dataclass
class ToolBundle:
    links: dict[str, Link]
    joints: list[Joint]
    frames: dict[str, dict[str, object]]
    superficial_mesh: trimesh.Trimesh
    target_bed_mesh: trimesh.Trimesh
    organ_mesh: trimesh.Trimesh
    fixture_mesh: trimesh.Trimesh
    bridge_specs: list[BridgeSpec]
    vessel_left: trimesh.Trimesh
    vessel_right: trimesh.Trimesh
    nerve_left: trimesh.Trimesh
    nerve_right: trimesh.Trimesh
    duct_left: trimesh.Trimesh
    duct_right: trimesh.Trimesh
    scissors_cartridge: trimesh.Trimesh


def _frame(parent: str, position, role: str, orientation=(1.0, 0.0, 0.0, 0.0)) -> dict[str, object]:
    return {"parent_link": parent, "position": list(position), "orientation_wxyz": list(orientation), "role": role}


def build_tool() -> ToolBundle:
    links: dict[str, Link] = {}
    frames: dict[str, dict[str, object]] = {}

    mount_visuals: list[Visual] = [
        Visual("FrankaAdapterPlate", cylinder_axis(0.032, 0.012, "z", (0, 0, 0.006), sections=72), "MountMetal", ("franka_mount",)),
        Visual("QuickReleaseRing", torus_axis(0.0275, 0.003, "z", (0, 0, 0.014), major_sections=72, minor_sections=14), "MountMetal"),
        Visual("MainHousing", ellipsoid_mesh((0.061, 0.052, 0.037), (0, 0, 0.055), subdivisions=3), "BodyPolymer", ("safeplane_dissection_robot",)),
        Visual("HousingCore", rounded_bar_mesh((0.116, 0.094, 0.052), (0, 0, 0.062), 0.009), "BodyPolymer"),
        Visual("TractionRail", rounded_bar_mesh((0.040, 0.184, 0.018), (0, 0, 0.113), 0.006), "RailMetal", ("bilateral_traction_rail",)),
        Visual("InstrumentTower", rounded_bar_mesh((0.088, 0.058, 0.064), (0, 0, 0.132), 0.007), "DarkPolymer", ("multimodal_dissection_head",)),
        Visual("HydroReservoir", cylinder_axis(0.017, 0.042, "y", (-0.034, 0.041, 0.058), sections=48), "TubeClear", ("hydrodissection_reservoir",)),
        Visual("HydroFill", cylinder_axis(0.014, 0.036, "y", (-0.034, 0.041, 0.058), sections=48), "HydroBlue", ("hydrodissection_inventory",)),
        Visual("IrrigationReservoir", cylinder_axis(0.015, 0.038, "y", (0.004, 0.043, 0.058), sections=48), "TubeClear", ("irrigation_reservoir",)),
        Visual("IrrigationFill", cylinder_axis(0.0125, 0.032, "y", (0.004, 0.043, 0.058), sections=48), "SalineBlue", ("irrigation_inventory",)),
        Visual("CollectionCanister", cylinder_axis(0.018, 0.045, "y", (0.040, 0.040, 0.058), sections=48), "TubeClear", ("smoke_and_fluid_collection",)),
        Visual("CollectionFill", cylinder_axis(0.015, 0.013, "y", (0.040, 0.054, 0.058), sections=48), "CollectionDark", ("collected_material",)),
        Visual("EnergyModule", rounded_bar_mesh((0.040, 0.067, 0.026), (0.046, 0, 0.098), 0.005), "AccentPolymer", ("low_energy_dissection_module",)),
        Visual("ScissorsCassetteHousing", rounded_bar_mesh((0.036, 0.050, 0.054), (0.0, 0, 0.117), 0.005), "DarkPolymer", ("guarded_micro_scissors_cassette",)),
        Visual("HydroModule", rounded_bar_mesh((0.034, 0.050, 0.048), (-0.046, 0, 0.116), 0.005), "AccentPolymer", ("hydrodissection_module",)),
        Visual("SuctionManifold", torus_axis(0.039, 0.0032, "z", (0, 0, 0.178), major_sections=80, minor_sections=16), "DarkPolymer", ("annular_smoke_and_fluid_evacuator",)),
        Visual("IrrigationManifold", torus_axis(0.030, 0.0015, "z", (0, 0, 0.181), major_sections=72, minor_sections=12), "SensorBlue", ("field_irrigation_manifold",)),
        Visual("SensorBridge", rounded_bar_mesh((0.070, 0.016, 0.016), (0, -0.050, 0.087), 0.004), "DarkPolymer", ("multimodal_sensor_bridge",)),
        Visual("StereoCameraLeft", cylinder_axis(0.0048, 0.004, "y", (-0.020, -0.059, 0.086), sections=36), "SensorGlass", ("rgb_camera",)),
        Visual("StereoCameraRight", cylinder_axis(0.0048, 0.004, "y", (0.020, -0.059, 0.086), sections=36), "SensorGlass", ("rgb_camera",)),
        Visual("DepthCamera", cylinder_axis(0.0046, 0.004, "y", (0, -0.059, 0.101), sections=36), "DepthGlass", ("depth_camera",)),
        Visual("FluorescenceCamera", cylinder_axis(0.0042, 0.004, "y", (-0.032, -0.058, 0.100), sections=36), "FluorescenceGlass", ("fluorescence_camera",)),
        Visual("ThermalCamera", cylinder_axis(0.0042, 0.004, "y", (0.032, -0.058, 0.100), sections=36), "ThermalGlass", ("thermal_camera",)),
        Visual("SafePlaneIndicator", box_mesh((0.018, 0.0012, 0.009), (-0.022, -0.050, 0.063)), "IndicatorGreen", ("safe_plane_ready_indicator",)),
        Visual("ProtectedStructureIndicator", box_mesh((0.018, 0.0012, 0.009), (0, -0.050, 0.063)), "IndicatorAmber", ("protected_structure_proximity_indicator",)),
        Visual("FaultIndicator", box_mesh((0.018, 0.0012, 0.009), (0.022, -0.050, 0.063)), "IndicatorRed", ("safety_fault_indicator",)),
        Visual("LabelPanel", box_mesh((0.060, 0.0012, 0.022), (0, -0.052, 0.040)), "LabelMaterial"),
    ]
    for i in range(SUCTION_PORT_COUNT):
        a = 2 * math.pi * i / SUCTION_PORT_COUNT
        x, y = 0.039 * math.cos(a), 0.039 * math.sin(a)
        mount_visuals.append(Visual(f"SuctionPort_{i:02d}", frustum_axis(0.0024, 0.0012, 0.008, "z", (x, y, 0.184), sections=26), "DarkPolymer", ("suction_port",)))
    for i in range(IRRIGATION_PORT_COUNT):
        a = 2 * math.pi * (i + 0.5) / IRRIGATION_PORT_COUNT
        x, y = 0.030 * math.cos(a), 0.030 * math.sin(a)
        mount_visuals.append(Visual(f"IrrigationPort_{i:02d}", frustum_axis(0.0014, 0.00055, 0.007, "z", (x, y, 0.184), sections=24), "SensorBlue", ("irrigation_port",)))
    links["Mount"] = Link(
        "Mount",
        (0, 0, 0),
        mount_visuals,
        [
            Collider("AdapterCollider", "cylinder", (0, 0, 0.008), radius=0.032, height=0.016, physics_material="MountPhysics"),
            Collider("HousingCollider", "box", (0, 0, 0.063), size=(0.126, 0.106, 0.082), physics_material="PolymerPhysics"),
            Collider("TractionRailCollider", "box", (0, 0, 0.113), size=(0.046, 0.188, 0.023), physics_material="MountPhysics"),
            Collider("InstrumentTowerCollider", "box", (0, 0, 0.133), size=(0.094, 0.064, 0.070), physics_material="PolymerPhysics"),
            Collider("SuctionRingCollider", "cylinder", (0, 0, 0.178), radius=0.044, height=0.008, physics_material="PolymerPhysics", role="annular_suction_ring"),
        ],
        0.560,
        ("safeplane_dissection_end_effector", "surgical_dissection_device"),
    )

    for side_name, side in (("Left", -1), ("Right", 1)):
        carriage_y = side * 0.058
        links[f"{side_name}TractionCarriage"] = Link(
            f"{side_name}TractionCarriage",
            (0, carriage_y, 0.132),
            [
                Visual("Carriage", rounded_bar_mesh((0.034, 0.032, 0.024), (0, 0, 0), 0.005), "AccentPolymer", ("traction_carriage",)),
                Visual("Arm", rounded_bar_mesh((0.018, 0.048, 0.014), (0, -side * 0.024, 0.028), 0.004), "RailMetal", ("traction_arm",)),
                Visual("ForceWindow", box_mesh((0.014, 0.0012, 0.006), (0, -side * 0.016, 0.002)), "IndicatorGreen", ("traction_force_indicator",)),
            ],
            [
                Collider("CarriageCollider", "box", (0, 0, 0), size=(0.036, 0.034, 0.026), physics_material="PolymerPhysics"),
                Collider("ArmCollider", "box", (0, -side * 0.024, 0.028), size=(0.020, 0.050, 0.016), physics_material="MetalPhysics"),
            ],
            0.068,
            ("bilateral_tissue_traction_carriage",),
        )
        links[f"{side_name}PadPivot"] = Link(
            f"{side_name}PadPivot",
            (0, -side * 0.047, 0.160),
            [
                Visual("Pivot", cylinder_axis(0.006, 0.018, "x", (0, 0, 0), sections=36), "MountMetal", ("traction_pad_pitch_axis",)),
                Visual("PadBack", rounded_bar_mesh((0.034, 0.018, 0.008), (0, 0, 0.014), 0.004), "DarkPolymer", ("traction_pad_backing",)),
            ],
            [
                Collider("PivotCollider", "cylinder", (0, 0, 0), radius=0.006, height=0.018, axis="x", physics_material="MetalPhysics"),
                Collider("PadBackCollider", "box", (0, 0, 0.014), size=(0.036, 0.020, 0.010), physics_material="PolymerPhysics"),
            ],
            0.036,
            ("traction_pad_pitch_link",),
        )
        pad_visuals = [
            Visual("CompliantPad", rounded_bar_mesh((0.038, 0.022, 0.007), (0, 0, 0.005), 0.005), "PadElastomer", ("atraumatic_traction_pad",)),
            Visual("Microtexture", box_mesh((0.034, 0.018, 0.0008), (0, 0, 0.009)), "PadContact", ("distributed_tissue_contact",)),
        ]
        pad_colliders = [
            Collider("PadCollider", "box", (0, 0, 0.005), size=(0.040, 0.024, 0.009), physics_material="PadContactPhysics", role="traction_pad_contact"),
        ]
        for cell in range(TRACTION_CELL_COUNT_PER_SIDE):
            x = -0.0135 + cell * 0.009
            pad_visuals.append(Visual(f"CaptureCell_{cell:02d}", ellipsoid_mesh((0.0035, 0.0065, 0.0014), (x, 0, 0.010), 2), "CaptureCell", ("tissue_capture_cell",)))
            pad_colliders.append(Collider(f"CaptureCell_{cell:02d}", "sphere", (x, 0, 0.010), radius=0.0042, physics_material="PadContactPhysics", role="traction_capture_volume"))
        links[f"{side_name}TractionPad"] = Link(
            f"{side_name}TractionPad",
            (0, 0, 0.026),
            pad_visuals,
            pad_colliders,
            0.028,
            ("distributed_atraumatic_tissue_capture_pad",),
        )

    for side_name, side in (("Left", -1), ("Right", 1)):
        y0 = side * 0.012
        tip_y = -side * 0.020
        spreader_visuals = [
            Visual("SpreaderArm", rounded_bar_mesh((0.011, 0.034, 0.010), (0, tip_y / 2.0, 0.014), 0.003), "JawMetal", ("blunt_spreader_arm",)),
            Visual("BluntPaddle", ellipsoid_mesh((0.010, 0.006, 0.0045), (0, tip_y, 0.031), 3), "PadElastomer", ("blunt_spreading_surface",)),
            Visual("ForceStrip", box_mesh((0.009, 0.0010, 0.004), (0, -side * 0.010, 0.010)), "IndicatorGreen", ("spreader_force_sensor",)),
        ]
        links[f"{side_name}Spreader"] = Link(
            f"{side_name}Spreader",
            (0, y0, 0.166),
            spreader_visuals,
            [
                Collider("ArmCollider", "box", (0, tip_y / 2.0, 0.014), size=(0.014, 0.036, 0.013), physics_material="MetalPhysics"),
                Collider("BluntContact", "sphere", (0, tip_y, 0.031), radius=0.007, physics_material="PadContactPhysics", role="blunt_spreader_contact"),
            ],
            0.034,
            ("atraumatic_blunt_spreader",),
        )

    links["HydroGimbal"] = Link(
        "HydroGimbal",
        (-0.034, 0, 0.146),
        [
            Visual("GimbalBody", cylinder_axis(0.008, 0.020, "x", (0, 0, 0), sections=40), "AccentPolymer", ("hydro_nozzle_gimbal",)),
            Visual("AngleScale", torus_axis(0.0095, 0.0008, "x", (0, 0, 0), major_sections=48, minor_sections=10), "IndicatorBlue", ("hydro_angle_scale",)),
        ],
        [Collider("GimbalCollider", "cylinder", (0, 0, 0), radius=0.0085, height=0.021, axis="x", physics_material="PolymerPhysics")],
        0.028,
        ("hydrodissection_gimbal",),
    )
    hydro_visuals = [
        Visual("NozzleBody", cylinder_axis(0.0038, 0.043, "z", (0, 0, 0.022), sections=42), "NozzleMetal", ("hydrodissection_nozzle",)),
        Visual("TaperedTip", frustum_axis(0.0030, 0.00055, 0.018, "z", (0, 0, 0.052), sections=40), "NozzleMetal", ("hydrodissection_tip",)),
        Visual("SafetyShield", torus_axis(0.0065, 0.0012, "z", (0, 0, 0.043), major_sections=48, minor_sections=10), "SensorBlue", ("hydrojet_splash_shield",)),
    ]
    for i in range(HYDRO_JET_COUNT):
        a = 2 * math.pi * i / HYDRO_JET_COUNT
        hydro_visuals.append(Visual(f"JetPort_{i:02d}", frustum_axis(0.00075, 0.00025, 0.005, "z", (0.0013 * math.cos(a), 0.0013 * math.sin(a), 0.062), sections=20), "HydroBlue", ("hydro_microjet",)))
    links["HydroNozzle"] = Link(
        "HydroNozzle",
        (0, 0, 0.010),
        hydro_visuals,
        [
            Collider("NozzleCollider", "cylinder", (0, 0, 0.022), radius=0.0042, height=0.044, physics_material="MetalPhysics"),
            Collider("TipCollider", "cylinder", (0, 0, 0.052), radius=0.0025, height=0.018, physics_material="MetalPhysics", role="hydrodissection_nozzle_tip"),
        ],
        0.030,
        ("hydrodissection_nozzle",),
    )

    fixed_blade = transform(scissor_blade_mesh(), (-0.0020, 0.0, 0.014), rotation_matrix((0, 0, 1), math.radians(-5)))
    links["ScissorCarriage"] = Link(
        "ScissorCarriage",
        (0, 0, 0.145),
        [
            Visual("CarriageBody", rounded_bar_mesh((0.024, 0.028, 0.038), (0, 0, 0.010), 0.004), "DarkPolymer", ("micro_scissors_carriage",)),
            Visual("FixedBlade", fixed_blade, "BladeSteel", ("fixed_micro_scissor_blade",)),
            Visual("BladePivot", cylinder_axis(0.0045, 0.014, "y", (0, 0, 0.018), sections=36), "MountMetal", ("scissor_pivot",)),
            Visual("FreshIndicator", box_mesh((0.009, 0.001, 0.005), (-0.006, -0.0145, 0.002)), "IndicatorGreen", ("fresh_scissors_indicator",)),
            Visual("SpentIndicator", box_mesh((0.009, 0.001, 0.005), (0.006, -0.0145, 0.002)), "IndicatorRed", ("spent_scissors_indicator",)),
        ],
        [
            Collider("CarriageCollider", "box", (0, 0, 0.010), size=(0.026, 0.030, 0.040), physics_material="PolymerPhysics"),
            Collider("FixedBladeCollider", "box", (-0.002, 0, 0.035), size=(0.0022, 0.010, 0.040), physics_material="BladePhysics", role="fixed_scissor_blade"),
        ],
        0.056,
        ("guarded_micro_scissors_carriage",),
    )
    links["ScissorGuard"] = Link(
        "ScissorGuard",
        (0, 0, 0.036),
        [
            Visual("GuardFrame", rounded_bar_mesh((0.016, 0.018, 0.030), (0, 0, 0.010), 0.003), "CeramicWhite", ("scissor_guard",)),
            Visual("GuardWindow", box_mesh((0.005, 0.012, 0.022), (0, 0, 0.014)), "DarkPolymer", ("scissor_guard_window",)),
        ],
        [Collider("GuardCollider", "box", (0, 0, 0.010), size=(0.018, 0.020, 0.032), physics_material="CeramicPhysics", role="scissor_guard")],
        0.018,
        ("scissor_guard",),
    )
    moving_blade = transform(scissor_blade_mesh(), (0.0020, 0.0, 0.0), rotation_matrix((0, 0, 1), math.radians(6)))
    links["MovingScissorBlade"] = Link(
        "MovingScissorBlade",
        (0, 0, 0.018),
        [Visual("MovingBlade", moving_blade, "BladeSteel", ("moving_micro_scissor_blade",))],
        [Collider("MovingBladeCollider", "box", (0.002, 0, 0.022), size=(0.0022, 0.010, 0.042), physics_material="BladePhysics", role="moving_scissor_blade")],
        0.009,
        ("moving_micro_scissor_blade",),
    )

    links["EnergyTip"] = Link(
        "EnergyTip",
        (0.034, 0, 0.151),
        [
            Visual("Shaft", cylinder_axis(0.0030, 0.040, "z", (0, 0, 0.020), sections=36), "NozzleMetal", ("low_energy_dissection_shaft",)),
            Visual("Spatula", ellipsoid_mesh((0.0065, 0.0038, 0.0015), (0, 0, 0.043), 3), "ElectrodeCopper", ("low_energy_dissection_tip",)),
            Visual("ThermalIndicator", box_mesh((0.007, 0.0010, 0.004), (0, -0.0040, 0.030)), "ThermalGlass", ("energy_tip_temperature_indicator",)),
        ],
        [
            Collider("ShaftCollider", "cylinder", (0, 0, 0.020), radius=0.0033, height=0.040, physics_material="MetalPhysics"),
            Collider("SpatulaContact", "sphere", (0, 0, 0.043), radius=0.0045, physics_material="ElectrodePhysics", role="energy_dissection_contact"),
        ],
        0.025,
        ("low_energy_dissection_probe",),
    )

    links["SuctionValve"] = Link(
        "SuctionValve",
        (0.047, 0.038, 0.092),
        [Visual("Valve", rounded_bar_mesh((0.010, 0.020, 0.012), (0, 0, 0), 0.003), "DarkPolymer", ("suction_metering_valve",))],
        [Collider("ValveCollider", "box", (0, 0, 0), size=(0.012, 0.022, 0.014), physics_material="PolymerPhysics")],
        0.014,
        ("suction_valve",),
    )
    links["HydroValve"] = Link(
        "HydroValve",
        (-0.047, 0.038, 0.092),
        [Visual("Valve", rounded_bar_mesh((0.010, 0.020, 0.012), (0, 0, 0), 0.003), "AccentPolymer", ("hydrodissection_metering_valve",))],
        [Collider("ValveCollider", "box", (0, 0, 0), size=(0.012, 0.022, 0.014), physics_material="PolymerPhysics")],
        0.014,
        ("hydrodissection_valve",),
    )
    links["IrrigationValve"] = Link(
        "IrrigationValve",
        (0.0, 0.043, 0.092),
        [Visual("Valve", rounded_bar_mesh((0.010, 0.020, 0.012), (0, 0, 0), 0.003), "SensorBlue", ("irrigation_metering_valve",))],
        [Collider("ValveCollider", "box", (0, 0, 0), size=(0.012, 0.022, 0.014), physics_material="PolymerPhysics")],
        0.014,
        ("irrigation_valve",),
    )

    joints: list[Joint] = [
        Joint("left_traction_joint", "prismatic", "Mount", "LeftTractionCarriage", "Y", (0, -0.058, 0.132), (0, 0, 0), -TRACTION_TRAVEL_M, 0.0, 4400, 150, 120),
        Joint("right_traction_joint", "prismatic", "Mount", "RightTractionCarriage", "Y", (0, 0.058, 0.132), (0, 0, 0), 0.0, TRACTION_TRAVEL_M, 4400, 150, 120),
        Joint("left_pad_pitch_joint", "revolute", "LeftTractionCarriage", "LeftPadPivot", "X", (0, -0.047, 0.028), (0, 0, 0), -PAD_PITCH_DEG, PAD_PITCH_DEG, 110, 8, 14),
        Joint("right_pad_pitch_joint", "revolute", "RightTractionCarriage", "RightPadPivot", "X", (0, 0.047, 0.028), (0, 0, 0), -PAD_PITCH_DEG, PAD_PITCH_DEG, 110, 8, 14),
        Joint("left_pad_compliance_joint", "prismatic", "LeftPadPivot", "LeftTractionPad", "Z", (0, 0, 0.026), (0, 0, 0), 0.0, PAD_COMPLIANCE_M, 1000, 70, 45),
        Joint("right_pad_compliance_joint", "prismatic", "RightPadPivot", "RightTractionPad", "Z", (0, 0, 0.026), (0, 0, 0), 0.0, PAD_COMPLIANCE_M, 1000, 70, 45),
        Joint("left_spreader_joint", "prismatic", "Mount", "LeftSpreader", "Y", (0, -0.012, 0.166), (0, 0, 0), -SPREADER_TRAVEL_M, 0.0, 3200, 125, 85),
        Joint("right_spreader_joint", "prismatic", "Mount", "RightSpreader", "Y", (0, 0.012, 0.166), (0, 0, 0), 0.0, SPREADER_TRAVEL_M, 3200, 125, 85),
        Joint("hydro_pitch_joint", "revolute", "Mount", "HydroGimbal", "X", (-0.034, 0, 0.146), (0, 0, 0), -HYDRO_PITCH_DEG, HYDRO_PITCH_DEG, 90, 7, 10),
        Joint("hydro_extension_joint", "prismatic", "HydroGimbal", "HydroNozzle", "Z", (0, 0, 0.010), (0, 0, 0), 0.0, HYDRO_EXTENSION_M, 3600, 115, 80),
        Joint("scissor_extension_joint", "prismatic", "Mount", "ScissorCarriage", "Z", (0, 0, 0.145), (0, 0, 0), 0.0, SCISSOR_EXTENSION_M, 5200, 160, 120),
        Joint("scissor_guard_joint", "prismatic", "ScissorCarriage", "ScissorGuard", "Z", (0, 0, 0.036), (0, 0, 0), -SCISSOR_GUARD_TRAVEL_M, 0.0, 3000, 100, 60),
        Joint("scissor_blade_joint", "revolute", "ScissorCarriage", "MovingScissorBlade", "Y", (0, 0, 0.018), (0, 0, 0), 0.0, SCISSOR_CLOSE_DEG, 140, 9, 18),
        Joint("energy_tip_extension_joint", "prismatic", "Mount", "EnergyTip", "Z", (0.034, 0, 0.151), (0, 0, 0), 0.0, ENERGY_EXTENSION_M, 3500, 115, 70),
        Joint("suction_valve_joint", "prismatic", "Mount", "SuctionValve", "Y", (0.047, 0.038, 0.092), (0, 0, 0), 0.0, 0.008, 1600, 48, 25),
        Joint("hydro_valve_joint", "prismatic", "Mount", "HydroValve", "Y", (-0.047, 0.038, 0.092), (0, 0, 0), 0.0, 0.008, 1600, 48, 25),
        Joint("irrigation_valve_joint", "prismatic", "Mount", "IrrigationValve", "Y", (0, 0.043, 0.092), (0, 0, 0), 0.0, 0.008, 1600, 48, 25),
    ]

    frames.update(
        {
            "panda_link8_mount": _frame("Mount", (0, 0, 0), "franka_wrist_mount"),
            "safeplane_tcp": _frame("Mount", (0, 0, WORK_PLANE_Z), "primary_dissection_tcp"),
            "safe_plane_reference": _frame("Mount", (0, 0, WORK_PLANE_Z), "safe_dissection_plane_reference"),
            "roi_center": _frame("Mount", (0, 0, WORK_PLANE_Z + 0.002), "dissection_roi_center"),
            "suction_center": _frame("Mount", (0, 0, 0.184), "annular_suction_center"),
            "irrigation_center": _frame("Mount", (0, 0, 0.184), "field_irrigation_center"),
            "stereo_left": _frame("Mount", (-0.020, -0.059, 0.086), "left_rgb_camera_optical_frame", (0.70710678, 0.70710678, 0, 0)),
            "stereo_right": _frame("Mount", (0.020, -0.059, 0.086), "right_rgb_camera_optical_frame", (0.70710678, 0.70710678, 0, 0)),
            "depth_camera": _frame("Mount", (0, -0.059, 0.101), "depth_camera_optical_frame", (0.70710678, 0.70710678, 0, 0)),
            "fluorescence_camera": _frame("Mount", (-0.032, -0.058, 0.100), "fluorescence_camera_optical_frame", (0.70710678, 0.70710678, 0, 0)),
            "thermal_camera": _frame("Mount", (0.032, -0.058, 0.100), "thermal_camera_optical_frame", (0.70710678, 0.70710678, 0, 0)),
            "protected_structure_probe": _frame("Mount", (0, 0, WORK_PLANE_Z - 0.004), "protected_structure_proximity_probe"),
            "count_reference": _frame("Mount", (0.055, 0.043, 0.052), "inventory_count_reference"),
            "disposal_reference": _frame("Mount", (0.072, 0.0, 0.040), "disposal_reference"),
        }
    )
    for side_name, side in (("Left", -1), ("Right", 1)):
        frames[f"{side_name.lower()}_traction_pad"] = _frame(f"{side_name}TractionPad", (0, 0, 0.010), f"{side_name.lower()}_traction_contact")
        for cell in range(TRACTION_CELL_COUNT_PER_SIDE):
            x = -0.0135 + cell * 0.009
            frames[f"{side_name.lower()}_capture_cell_{cell}"] = _frame(f"{side_name}TractionPad", (x, 0, 0.010), f"{side_name.lower()}_traction_capture_cell")
        frames[f"{side_name.lower()}_spreader_tip"] = _frame(f"{side_name}Spreader", (0, -side * 0.020, 0.031), f"{side_name.lower()}_blunt_spreader_tip")
    frames.update(
        {
            "hydro_nozzle_tip": _frame("HydroNozzle", (0, 0, 0.064), "hydrodissection_nozzle_tip"),
            "hydro_axis": _frame("HydroNozzle", (0, 0, 0.058), "hydrodissection_jet_axis"),
            "scissor_pivot": _frame("ScissorCarriage", (0, 0, 0.018), "micro_scissor_pivot"),
            "scissor_cut_plane": _frame("ScissorCarriage", (0, 0, 0.056), "guarded_scissor_cut_plane"),
            "scissor_guard_tip": _frame("ScissorGuard", (0, 0, 0.026), "scissor_guard_tip"),
            "energy_tip": _frame("EnergyTip", (0, 0, 0.043), "low_energy_dissection_tip"),
            "energy_axis": _frame("EnergyTip", (0, 0, 0.038), "energy_delivery_axis"),
        }
    )

    superficial = grid_surface_mesh(
        0.164,
        0.124,
        25,
        19,
        z_func=lambda x, y: 0.0025 * math.cos(math.pi * x / 0.164) + 0.0015 * math.cos(2 * math.pi * y / 0.124),
        center=(0, 0, 0.036),
    )
    target_bed = grid_surface_mesh(
        0.178,
        0.138,
        27,
        21,
        z_func=lambda x, y: 0.0018 * math.cos(math.pi * x / 0.178) - 0.0012 * math.sin(math.pi * y / 0.138),
        center=(0, 0, 0.014),
    )
    organ = ellipsoid_mesh((0.105, 0.082, 0.030), (0, 0, -0.012), subdivisions=4)
    fixture = rounded_bar_mesh((0.220, 0.180, 0.012), (0, 0, -0.048), 0.012)

    vessel_points = [(-0.075, -0.028, 0.020), (-0.035, -0.020, 0.022), (0.0, -0.014, 0.021), (0.038, -0.006, 0.019), (0.078, 0.006, 0.018)]
    nerve_points = [(-0.075, 0.022, 0.019), (-0.035, 0.016, 0.021), (0.0, 0.020, 0.020), (0.040, 0.028, 0.019), (0.078, 0.034, 0.018)]
    duct_points = [(-0.068, 0.048, 0.018), (-0.030, 0.041, 0.020), (0.006, 0.036, 0.019), (0.043, 0.043, 0.018), (0.072, 0.052, 0.017)]
    vessel_left = curved_tube(vessel_points[:3], 0.0031)
    vessel_right = curved_tube(vessel_points[2:], 0.0031)
    nerve_left = curved_tube(nerve_points[:3], 0.0021)
    nerve_right = curved_tube(nerve_points[2:], 0.0021)
    duct_left = curved_tube(duct_points[:3], 0.0026)
    duct_right = curved_tube(duct_points[2:], 0.0026)

    structures = {
        "vessel": np.asarray(vessel_points[2], dtype=float),
        "nerve": np.asarray(nerve_points[2], dtype=float),
        "duct": np.asarray(duct_points[2], dtype=float),
    }
    bridge_specs: list[BridgeSpec] = []
    positions: list[tuple[float, float]] = []
    for row, y in enumerate(np.linspace(-0.048, 0.050, 5)):
        for col, x in enumerate(np.linspace(-0.060, 0.060, 6)):
            if (row, col) in {(0, 0), (4, 5)}:
                continue
            positions.append((float(x + (0.003 if row % 2 else 0.0)), float(y)))
    positions = positions[:ADHESION_BRIDGE_COUNT]
    for index, (x, y) in enumerate(positions):
        p = np.asarray([x, y, 0.025], dtype=float)
        nearest_name, nearest_distance = min(((name, float(np.linalg.norm(p[:2] - center[:2]))) for name, center in structures.items()), key=lambda item: item[1])
        if index % 7 == 0 or nearest_distance < 0.014:
            bridge_class = "dense_fibrous_band"
            recommended = "guarded_scissors"
            mechanical, hydro, energy = 0.026, 0.55, 1.6
        elif index % 5 == 0:
            bridge_class = "vascularized_adhesion"
            recommended = "low_energy_or_scissors"
            mechanical, hydro, energy = 0.018, 0.42, 1.15
        else:
            bridge_class = "loose_connective_fibre"
            recommended = "blunt_or_hydro"
            mechanical, hydro, energy = 0.008, 0.18, 0.75
        bridge_specs.append(
            BridgeSpec(
                index=index,
                position=(x, y, 0.025),
                bridge_class=bridge_class,
                target=True,
                recommended_mode=recommended,
                mechanical_work_j=mechanical,
                hydro_volume_ml=hydro,
                energy_dose_j=energy,
                nearest_structure=nearest_name,
                clearance_m=nearest_distance,
            )
        )

    scissors_cartridge = trimesh.util.concatenate(
        [
            rounded_bar_mesh((0.022, 0.030, 0.055), (0, 0, 0.016), 0.004),
            fixed_blade,
            moving_blade,
            cylinder_axis(0.0045, 0.014, "y", (0, 0, 0.018), sections=36),
        ]
    )
    return ToolBundle(
        links=links,
        joints=joints,
        frames=frames,
        superficial_mesh=superficial,
        target_bed_mesh=target_bed,
        organ_mesh=organ,
        fixture_mesh=fixture,
        bridge_specs=bridge_specs,
        vessel_left=vessel_left,
        vessel_right=vessel_right,
        nerve_left=nerve_left,
        nerve_right=nerve_right,
        duct_left=duct_left,
        duct_right=duct_right,
        scissors_cartridge=scissors_cartridge,
    )


def visual_materials_scope(root: str) -> str:
    specs = {
        "BodyPolymer": ((0.82, 0.85, 0.88), 0.0, 0.34, 1.0),
        "AccentPolymer": ((0.055, 0.31, 0.62), 0.0, 0.29, 1.0),
        "DarkPolymer": ((0.035, 0.045, 0.055), 0.0, 0.30, 1.0),
        "MountMetal": ((0.44, 0.48, 0.53), 0.82, 0.24, 1.0),
        "RailMetal": ((0.25, 0.29, 0.34), 0.74, 0.26, 1.0),
        "JawMetal": ((0.60, 0.64, 0.68), 0.78, 0.22, 1.0),
        "NozzleMetal": ((0.68, 0.72, 0.76), 0.90, 0.16, 1.0),
        "BladeSteel": ((0.73, 0.76, 0.80), 0.94, 0.11, 1.0),
        "SpentBlade": ((0.30, 0.29, 0.28), 0.70, 0.46, 1.0),
        "ElectrodeCopper": ((0.78, 0.40, 0.13), 0.88, 0.20, 1.0),
        "CeramicWhite": ((0.91, 0.91, 0.88), 0.0, 0.16, 1.0),
        "PadElastomer": ((0.06, 0.53, 0.61), 0.0, 0.52, 1.0),
        "PadContact": ((0.025, 0.28, 0.33), 0.0, 0.60, 1.0),
        "CaptureCell": ((0.03, 0.72, 0.74), 0.0, 0.38, 0.76),
        "SensorGlass": ((0.05, 0.16, 0.24), 0.15, 0.10, 0.78),
        "DepthGlass": ((0.02, 0.20, 0.38), 0.12, 0.08, 0.84),
        "FluorescenceGlass": ((0.06, 0.52, 0.17), 0.10, 0.10, 0.86),
        "ThermalGlass": ((0.31, 0.05, 0.43), 0.18, 0.13, 0.86),
        "SensorBlue": ((0.04, 0.39, 0.86), 0.05, 0.20, 1.0),
        "IndicatorBlue": ((0.06, 0.50, 0.98), 0.0, 0.20, 1.0),
        "IndicatorGreen": ((0.05, 0.86, 0.28), 0.0, 0.20, 1.0),
        "IndicatorAmber": ((1.0, 0.52, 0.02), 0.0, 0.23, 1.0),
        "IndicatorRed": ((0.92, 0.03, 0.04), 0.0, 0.22, 1.0),
        "HydroBlue": ((0.02, 0.48, 0.96), 0.0, 0.05, 0.52),
        "SalineBlue": ((0.05, 0.66, 0.93), 0.0, 0.06, 0.48),
        "CollectionDark": ((0.31, 0.12, 0.08), 0.0, 0.16, 0.72),
        "TubeClear": ((0.63, 0.80, 0.88), 0.0, 0.08, 0.34),
        "LabelMaterial": ((0.96, 0.97, 0.98), 0.0, 0.40, 1.0),
        "SuperficialTissue": ((0.78, 0.43, 0.40), 0.0, 0.50, 1.0),
        "TargetBed": ((0.64, 0.22, 0.20), 0.0, 0.54, 1.0),
        "ProtectedOrgan": ((0.54, 0.12, 0.16), 0.0, 0.48, 1.0),
        "LooseFibre": ((0.93, 0.76, 0.63), 0.0, 0.58, 1.0),
        "DenseFibre": ((0.79, 0.55, 0.40), 0.0, 0.52, 1.0),
        "VascularAdhesion": ((0.66, 0.18, 0.16), 0.0, 0.44, 1.0),
        "VesselMaterial": ((0.55, 0.035, 0.030), 0.0, 0.38, 1.0),
        "VesselInjured": ((0.86, 0.01, 0.01), 0.0, 0.26, 1.0),
        "NerveMaterial": ((0.94, 0.77, 0.16), 0.0, 0.48, 1.0),
        "NerveInjured": ((0.34, 0.18, 0.05), 0.0, 0.66, 1.0),
        "DuctMaterial": ((0.18, 0.70, 0.30), 0.0, 0.48, 1.0),
        "DuctInjured": ((0.56, 0.88, 0.16), 0.0, 0.30, 1.0),
        "FixtureMaterial": ((0.08, 0.09, 0.11), 0.20, 0.42, 1.0),
        "HydroParticle": ((0.04, 0.53, 0.98), 0.0, 0.02, 0.56),
        "SmokeParticle": ((0.64, 0.66, 0.69), 0.0, 0.10, 0.25),
        "BloodParticle": ((0.64, 0.008, 0.006), 0.0, 0.12, 0.78),
        "DuctFluid": ((0.54, 0.86, 0.12), 0.0, 0.08, 0.68),
        "GuideRed": ((0.95, 0.08, 0.08), 0.0, 0.30, 1.0),
        "GuideGreen": ((0.08, 0.90, 0.18), 0.0, 0.30, 1.0),
        "GuideBlue": ((0.08, 0.28, 0.95), 0.0, 0.30, 1.0),
        "CollisionDebug": ((1.0, 0.12, 0.72), 0.0, 0.20, 0.30),
    }
    blocks = []
    for name, (color, metallic, roughness, opacity) in specs.items():
        blocks.append(
            f'''        def Material "{name}"
        {{
            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {vec(color)}
                float inputs:metallic = {f(metallic)}
                float inputs:roughness = {f(roughness)}
                float inputs:opacity = {f(opacity)}
                token outputs:surface
            }}
            token outputs:surface.connect = </{root}/Looks/{name}/PreviewSurface.outputs:surface>
        }}'''
        )
    return '    def Scope "Looks"\n    {\n' + '\n'.join(blocks) + '\n    }'


def physics_materials_scope() -> str:
    specs = {
        "MountPhysics": (0.34, 0.26, 0.02),
        "PolymerPhysics": (0.48, 0.38, 0.03),
        "MetalPhysics": (0.28, 0.21, 0.02),
        "PadContactPhysics": (0.76, 0.60, 0.01),
        "BladePhysics": (0.18, 0.12, 0.00),
        "CeramicPhysics": (0.32, 0.25, 0.01),
        "ElectrodePhysics": (0.42, 0.32, 0.01),
        "TissuePhysics": (0.58, 0.46, 0.00),
        "FibrePhysics": (0.49, 0.37, 0.00),
        "VesselPhysics": (0.50, 0.38, 0.00),
        "NervePhysics": (0.46, 0.34, 0.00),
        "DuctPhysics": (0.45, 0.34, 0.00),
    }
    blocks = []
    for name, (static, dynamic, restitution) in specs.items():
        blocks.append(
            f'''        def Material "{name}" (
            prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
        )
        {{
            float physics:staticFriction = {f(static)}
            float physics:dynamicFriction = {f(dynamic)}
            float physics:restitution = {f(restitution)}
            uniform token physxMaterial:frictionCombineMode = "max"
            uniform token physxMaterial:restitutionCombineMode = "min"
        }}'''
        )
    return '    def Scope "PhysicsMaterials"\n    {\n' + '\n'.join(blocks) + '\n    }'
def mesh_usda(visual: Visual, material_path: str, indent: str = "            ") -> str:
    mesh=visual.mesh
    vertices=np.asarray(mesh.vertices,dtype=float)
    faces=np.asarray(mesh.faces,dtype=int)
    normals=np.asarray(mesh.vertex_normals,dtype=float)
    bmin,bmax=mesh.bounds
    points=",\n".join(indent+"        "+vec(p) for p in vertices)
    counts=", ".join("3" for _ in faces)
    indices=", ".join(str(int(i)) for i in faces.reshape(-1))
    normal_values=",\n".join(indent+"        "+vec(n) for n in normals)
    labels=", ".join(f'"{x}"' for x in visual.labels)
    label_attr=f'{indent}    custom token[] drAnmar:labels = [{labels}]\n' if labels else ""
    return f'''{indent}def Mesh "{visual.name}" (
{indent}    prepend apiSchemas = ["MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    rel material:binding = <{material_path}>
{label_attr}{indent}    float3[] extent = [{vec(bmin)}, {vec(bmax)}]
{indent}    int[] faceVertexCounts = [{counts}]
{indent}    int[] faceVertexIndices = [{indices}]
{indent}    point3f[] points = [
{points}
{indent}    ]
{indent}    normal3f[] normals = [
{normal_values}
{indent}    ] (
{indent}        interpolation = "vertex"
{indent}    )
{indent}    uniform token subdivisionScheme = "none"
{indent}}}'''


def collider_usda(collider: Collider, root_path: str, indent: str = "            ") -> str:
    if not collider.author_enabled:
        return ""
    api='prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]'
    binding=f'{indent}    rel material:binding:physics = <{root_path}/PhysicsMaterials/{collider.physics_material}>\n'
    collision_enabled = str(collider.collision_enabled).lower()
    common=f'''{indent}    custom string drAnmar:role = "{collider.role}"
{binding}{indent}    bool physics:collisionEnabled = {collision_enabled}
{indent}    float physxCollision:contactOffset = 0.00035
{indent}    float physxCollision:restOffset = 0
{indent}    double3 xformOp:translate = {vec(collider.center)}
{indent}    quatf xformOp:orient = {quat(collider.orientation_wxyz)}
'''
    if collider.kind=="box":
        assert collider.size is not None
        return f'''{indent}def Cube "{collider.name}" (
{indent}    {api}
{indent})
{indent}{{
{common}{indent}    double size = 1
{indent}    double3 xformOp:scale = {vec(collider.size)}
{indent}    uniform token purpose = "guide"
{indent}    token visibility = "invisible"
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
{indent}}}'''
    if collider.kind=="cylinder":
        assert collider.radius is not None and collider.height is not None
        return f'''{indent}def Cylinder "{collider.name}" (
{indent}    {api}
{indent})
{indent}{{
{common}{indent}    uniform token axis = "{collider.axis.upper()}"
{indent}    double radius = {f(collider.radius)}
{indent}    double height = {f(collider.height)}
{indent}    uniform token purpose = "guide"
{indent}    token visibility = "invisible"
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{indent}}}'''
    if collider.kind=="sphere":
        assert collider.radius is not None
        return f'''{indent}def Sphere "{collider.name}" (
{indent}    {api}
{indent})
{indent}{{
{common}{indent}    double radius = {f(collider.radius)}
{indent}    uniform token purpose = "guide"
{indent}    token visibility = "invisible"
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{indent}}}'''
    raise ValueError(collider.kind)


def frame_usda(name: str, data: dict[str,object], indent: str = "            ") -> str:
    return f'''{indent}def Xform "{name}"
{indent}{{
{indent}    custom string drAnmar:role = "{data['role']}"
{indent}    double3 xformOp:translate = {vec(data['position'])}
{indent}    quatf xformOp:orient = {quat(data['orientation_wxyz'])}
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{indent}}}'''


def link_usda(link: Link, root_path: str, frames: dict[str,dict[str,object]]) -> str:
    schemas='prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]'
    body_attrs=""
    if link.mass_properties is not None:
        mp=link.mass_properties
        body_attrs=f'''        bool physics:rigidBodyEnabled = true
        bool physics:kinematicEnabled = false
        float physics:mass = {f(mp['mass_kg'])}
        point3f physics:centerOfMass = {vec(mp['center_of_mass_m'])}
        vector3f physics:diagonalInertia = {vec(mp['diagonal_inertia_kg_m2'])}
        quatf physics:principalAxes = {quat(mp['principal_axes_wxyz'])}
        bool physxRigidBody:enableCCD = true
        float physxRigidBody:linearDamping = 0.04
        float physxRigidBody:angularDamping = 0.06
        int physxRigidBody:solverPositionIterationCount = 20
        int physxRigidBody:solverVelocityIterationCount = 6
'''
    visual_blocks="\n".join(mesh_usda(v,f"{root_path}/Looks/{v.material}") for v in link.visuals)
    collider_blocks="\n".join(collider_usda(c,root_path) for c in link.colliders if c.author_enabled)
    frame_blocks="\n".join(frame_usda(n,d) for n,d in frames.items() if d["parent_link"]==link.name)
    labels=", ".join(f'"{x}"' for x in link.labels)
    labels_attr=f'        custom token[] drAnmar:labels = [{labels}]\n' if labels else ''
    return f'''    def Xform "{link.name}" (
        {schemas}
    )
    {{
        double3 xformOp:translate = {vec(link.translation)}
        uniform token[] xformOpOrder = ["xformOp:translate"]
{labels_attr}{body_attrs}        def Scope "Visuals"
        {{
{visual_blocks}
        }}
        def Scope "Collisions"
        {{
{collider_blocks}
        }}
        def Scope "Frames"
        {{
{frame_blocks}
        }}
    }}'''


def joint_usda(joint: Joint, root_path: str) -> str:
    if joint.type == "prismatic":
        typename="PhysicsPrismaticJoint";axis_line=f'        uniform token physics:axis = "{joint.axis}"\n';drive="linear"
    elif joint.type == "revolute":
        typename="PhysicsRevoluteJoint";axis_line=f'        uniform token physics:axis = "{joint.axis}"\n';drive="angular"
    elif joint.type == "fixed":
        typename="PhysicsFixedJoint";axis_line="";drive=None
    else:
        raise ValueError(joint.type)
    api=(
        f'        prepend apiSchemas = ["PhysicsDriveAPI:{drive}"]\n'
        if drive else ""
    )
    limits="";drive_block=""
    if joint.lower is not None and joint.upper is not None:
        limits=f'''        float physics:lowerLimit = {f(joint.lower)}
        float physics:upperLimit = {f(joint.upper)}
'''
    if drive:
        drive_block=f'''        uniform token drive:{drive}:physics:type = "force"
        float drive:{drive}:physics:stiffness = {f(joint.stiffness)}
        float drive:{drive}:physics:damping = {f(joint.damping)}
        float drive:{drive}:physics:maxForce = {f(joint.max_force)}
        float drive:{drive}:physics:targetPosition = 0
        float drive:{drive}:physics:targetVelocity = {f(joint.target_velocity)}
'''
    return f'''    def {typename} "{joint.name}" (
{api}
    )
    {{
{axis_line}        rel physics:body0 = <{root_path}/Links/{joint.body0}>
        rel physics:body1 = <{root_path}/Links/{joint.body1}>
        point3f physics:localPos0 = {vec(joint.local_pos0)}
        point3f physics:localPos1 = {vec(joint.local_pos1)}
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        bool physics:collisionEnabled = false
{limits}{drive_block}    }}'''


def nested_over(path: Sequence[str], body_lines: Sequence[str], *, indent: str="            ") -> str:
    lines=[]
    for depth,name in enumerate(path):
        prefix=indent+"    "*depth
        lines.append(f'{prefix}over "{name}"');lines.append(f"{prefix}{{")
    body_prefix=indent+"    "*len(path)
    lines.extend(f"{body_prefix}{line}" for line in body_lines)
    for depth in reversed(range(len(path))):
        prefix=indent+"    "*depth;lines.append(f"{prefix}}}")
    return "\n".join(lines)


def state_variants() -> str:
    def edits(groups: Sequence[tuple[str, Sequence[tuple[str, str, str]]]]) -> str:
        lines = ['            over "Links"', "            {"]
        for link, values in groups:
            lines.extend([
                f'                over "{link}"', "                {",
                '                    over "Visuals"', "                    {",
            ])
            for prim, attribute, value in values:
                lines.extend([
                    f'                        over "{prim}"',
                    "                        {",
                    f"                            {attribute} = {value}",
                    "                        }",
                ])
            lines.extend(["                    }", "                }"])
        lines.append("            }")
        return "\n".join(lines)

    scissors_fresh = edits([
        ("ScissorCarriage", [
            ("FreshIndicator", "token visibility", '"inherited"'),
            ("SpentIndicator", "token visibility", '"invisible"'),
            ("FixedBlade", "rel material:binding", f"</{ROOT_PRIM}/Looks/BladeSteel>"),
        ]),
        ("MovingScissorBlade", [
            ("MovingBlade", "rel material:binding", f"</{ROOT_PRIM}/Looks/BladeSteel>"),
        ]),
    ])
    scissors_spent = edits([
        ("ScissorCarriage", [
            ("FreshIndicator", "token visibility", '"invisible"'),
            ("SpentIndicator", "token visibility", '"inherited"'),
            ("FixedBlade", "rel material:binding", f"</{ROOT_PRIM}/Looks/SpentBlade>"),
        ]),
        ("MovingScissorBlade", [
            ("MovingBlade", "rel material:binding", f"</{ROOT_PRIM}/Looks/SpentBlade>"),
        ]),
    ])
    hydro_full = edits([("Mount", [("HydroFill", "token visibility", '"inherited"')])])
    hydro_empty = edits([("Mount", [("HydroFill", "token visibility", '"invisible"')])])
    irrigation_full = edits([("Mount", [("IrrigationFill", "token visibility", '"inherited"')])])
    irrigation_empty = edits([("Mount", [("IrrigationFill", "token visibility", '"invisible"')])])
    collection_empty = edits([("Mount", [("CollectionFill", "token visibility", '"invisible"')])])
    collection_visible = edits([("Mount", [("CollectionFill", "token visibility", '"inherited"')])])
    energy_ready = edits([("Mount", [
        ("SafePlaneIndicator", "token visibility", '"inherited"'),
        ("FaultIndicator", "token visibility", '"invisible"'),
    ])])
    energy_fault = edits([("Mount", [
        ("SafePlaneIndicator", "token visibility", '"invisible"'),
        ("FaultIndicator", "token visibility", '"inherited"'),
    ])])
    return f'''    variantSet "scissors_state" = {{
        "fresh"
        {{
{scissors_fresh}
        }}
        "spent"
        {{
{scissors_spent}
        }}
    }}
    variantSet "hydro_state" = {{
        "full"
        {{
{hydro_full}
        }}
        "empty"
        {{
{hydro_empty}
        }}
    }}
    variantSet "irrigation_state" = {{
        "full"
        {{
{irrigation_full}
        }}
        "empty"
        {{
{irrigation_empty}
        }}
    }}
    variantSet "collection_state" = {{
        "empty"
        {{
{collection_empty}
        }}
        "partial"
        {{
{collection_visible}
        }}
        "full"
        {{
{collection_visible}
        }}
    }}
    variantSet "energy_state" = {{
        "ready"
        {{
{energy_ready}
        }}
        "fault"
        {{
{energy_fault}
        }}
    }}'''


def tool_usda(bundle: ToolBundle, articulation_root: bool) -> str:
    root = STANDALONE_ROOT if articulation_root else ROOT_PRIM
    root_path = f"/{root}"
    schemas = (
        '    prepend apiSchemas = ["PhysicsArticulationRootAPI"]'
        if articulation_root else ""
    )
    links = "\n\n".join(link_usda(link, root_path, bundle.frames) for link in bundle.links.values())
    joints = "\n\n".join(joint_usda(joint, root_path) for joint in bundle.joints)
    variants = state_variants().replace(f"/{ROOT_PRIM}/", f"/{root}/")
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "{ASSET_NAME}: bilateral traction, blunt spreading, hydrodissection, guarded cutting, energy dissection, protected-structure monitoring, and connectivity verification."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
{schemas}
    prepend variantSets = ["scissors_state", "hydro_state", "irrigation_state", "collection_state", "energy_state"]
    variants = {{
        string scissors_state = "fresh"
        string hydro_state = "full"
        string irrigation_state = "full"
        string collection_state = "empty"
        string energy_state = "ready"
    }}
    customData = {{
        string drAnmarAssetId = "dranmar-safeplane-dissection-robot-v1"
        string drAnmarAssetVersion = "{VERSION}"
        bool drAnmarClinicalValidation = false
        bool drAnmarMedicalDevice = false
        string drAnmarStatus = "research_only_pending_runtime_and_physical_calibration"
        string drAnmarMount = "replaces_panda_hand_at_panda_link8"
        int drAnmarTractionCellCount = {2 * TRACTION_CELL_COUNT_PER_SIDE}
        int drAnmarDissectionModalityCount = 4
        bool drAnmarProtectedStructureInterlock = true
    }}
)
{{
{visual_materials_scope(root)}
{physics_materials_scope()}
    def Scope "Links"
    {{
{links}
    }}
    def Scope "Joints"
    {{
{joints}
    }}
{variants}
}}
'''


def material_color(material: str) -> tuple[int, int, int, int]:
    colors = {
        "BodyPolymer": (210, 218, 225, 255), "AccentPolymer": (15, 82, 158, 255), "DarkPolymer": (13, 18, 24, 255),
        "MountMetal": (116, 125, 137, 255), "RailMetal": (70, 78, 90, 255), "JawMetal": (157, 165, 174, 255),
        "NozzleMetal": (179, 185, 193, 255), "BladeSteel": (198, 205, 214, 255), "SpentBlade": (79, 76, 73, 255),
        "ElectrodeCopper": (199, 102, 33, 255), "CeramicWhite": (235, 235, 226, 255), "PadElastomer": (15, 135, 155, 255),
        "PadContact": (7, 72, 84, 255), "CaptureCell": (8, 184, 189, 190), "SensorGlass": (10, 42, 65, 190),
        "DepthGlass": (5, 55, 99, 210), "FluorescenceGlass": (14, 132, 45, 215), "ThermalGlass": (79, 13, 111, 215),
        "SensorBlue": (10, 99, 220, 255), "IndicatorBlue": (14, 127, 250, 255), "IndicatorGreen": (13, 219, 70, 255),
        "IndicatorAmber": (255, 133, 5, 255), "IndicatorRed": (235, 8, 10, 255), "HydroBlue": (8, 122, 245, 145),
        "SalineBlue": (13, 168, 238, 130), "CollectionDark": (79, 31, 20, 180), "TubeClear": (160, 204, 225, 95),
        "LabelMaterial": (245, 247, 250, 255), "SuperficialTissue": (204, 111, 105, 255), "TargetBed": (168, 57, 52, 255),
        "ProtectedOrgan": (139, 31, 42, 255), "LooseFibre": (237, 193, 160, 255), "DenseFibre": (201, 140, 102, 255),
        "VascularAdhesion": (168, 46, 41, 255), "VesselMaterial": (145, 8, 7, 255), "VesselInjured": (230, 3, 3, 255),
        "NerveMaterial": (242, 198, 41, 255), "NerveInjured": (88, 46, 13, 255), "DuctMaterial": (46, 180, 77, 255),
        "DuctInjured": (143, 224, 41, 255), "FixtureMaterial": (20, 23, 28, 255), "HydroParticle": (10, 135, 250, 150),
        "SmokeParticle": (160, 166, 172, 70), "BloodParticle": (163, 2, 2, 200), "DuctFluid": (139, 220, 31, 180),
        "GuideRed": (242, 20, 20, 255), "GuideGreen": (20, 230, 46, 255), "GuideBlue": (20, 72, 242, 255),
        "CollisionDebug": (255, 31, 184, 90),
    }
    return colors.get(material, (180, 180, 180, 255))


def pbr(mesh: trimesh.Trimesh, material: str) -> trimesh.Trimesh:
    mesh = mesh.copy()
    rgba = material_color(material)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=np.tile(np.asarray(rgba, dtype=np.uint8), (len(mesh.faces), 1)))
    return mesh


def mesh_usda_unbound(name: str, mesh: trimesh.Trimesh, material_path: str, indent: str = "    ", double_sided: bool = False) -> str:
    return mesh_usda(Visual(name, mesh, material_path.split("/")[-1]), material_path, indent).replace(
        f'{indent}    uniform token subdivisionScheme = "none"',
        f'{indent}    uniform bool doubleSided = {"true" if double_sided else "false"}\n{indent}    uniform token subdivisionScheme = "none"',
    )


def rigid_proxy_usda(bundle: ToolBundle) -> str:
    visuals = []
    for link in bundle.links.values():
        T = np.eye(4)
        T[:3, 3] = np.asarray(link.translation)
        for visual in link.visuals:
            mesh = visual.mesh.copy()
            mesh.apply_transform(T)
            visuals.append(Visual(f"{link.name}_{visual.name}", mesh, visual.material, visual.labels))
    merged = trimesh.util.concatenate([v.mesh for v in visuals])
    mp = box_mass_properties([merged], 1.02)
    visual_blocks = "\n".join(mesh_usda(v, f"/{PROXY_ROOT}/Looks/{v.material}", "        ") for v in visuals)
    return f'''#usda 1.0
(
    defaultPrim = "{PROXY_ROOT}"
    doc = "Rigid perception, planning, handover, and synthetic-data proxy for the DrAnmar SafePlane Dissection Robot."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{PROXY_ROOT}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)
{{
    bool physics:rigidBodyEnabled = true
    bool physics:kinematicEnabled = false
    float physics:mass = {f(mp['mass_kg'])}
    point3f physics:centerOfMass = {vec(mp['center_of_mass_m'])}
    vector3f physics:diagonalInertia = {vec(mp['diagonal_inertia_kg_m2'])}
    quatf physics:principalAxes = {quat(mp['principal_axes_wxyz'])}
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(PROXY_ROOT)}
{physics_materials_scope()}
    def Scope "Visuals"
    {{
{visual_blocks}
    }}
    def Scope "Collisions"
    {{
{collider_usda(Collider('HousingCollider','box',(0,0,0.078),size=(0.132,0.118,0.160),physics_material='PolymerPhysics'), f'/{PROXY_ROOT}', '        ')}
{collider_usda(Collider('HeadCollider','cylinder',(0,0,0.177),radius=0.048,height=0.025,physics_material='PolymerPhysics'), f'/{PROXY_ROOT}', '        ')}
    }}
}}
'''


def orientation_from_z(direction: Sequence[float]) -> tuple[float, float, float, float]:
    d = normalize(np.asarray(direction, dtype=float))
    z = np.asarray([0.0, 0.0, 1.0])
    cross = np.cross(z, d)
    dot = float(np.clip(np.dot(z, d), -1.0, 1.0))
    if np.linalg.norm(cross) <= 1e-12:
        R = np.eye(3) if dot > 0 else rotation_matrix((1, 0, 0), math.pi)
    else:
        R = rotation_matrix(cross, math.acos(dot))
    return matrix_to_quat_wxyz(R)


def polyline_colliders(points: Sequence[Sequence[float]], radius: float, material: str, prefix: str) -> list[Collider]:
    out: list[Collider] = []
    pts = [np.asarray(p, dtype=float) for p in points]
    for index, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        direction = b - a
        length = float(np.linalg.norm(direction))
        out.append(
            Collider(
                f"{prefix}_{index:02d}",
                "cylinder",
                tuple((a + b) / 2.0),
                radius=radius,
                height=length,
                axis="z",
                orientation_wxyz=orientation_from_z(direction),
                physics_material=material,
                role="protected_structure_collision",
            )
        )
    return out


def adhesion_bridge_usda() -> str:
    upper = fibre_half_mesh(0.022, 0.0011, True, 7)
    lower = fibre_half_mesh(0.022, 0.0011, False, 7)
    up = Link("UpperAnchor", (0, 0, 0.011), [Visual("FibreHalf", upper, "DenseFibre", ("adhesion_fibre",))], [Collider("AttachmentVolume", "sphere", (0, 0, 0), radius=0.0026, physics_material="FibrePhysics", role="upper_tissue_attachment")], 0.00002)
    lo = Link("LowerAnchor", (0, 0, -0.011), [Visual("FibreHalf", lower, "DenseFibre", ("adhesion_fibre",))], [Collider("AttachmentVolume", "sphere", (0, 0, 0), radius=0.0026, physics_material="FibrePhysics", role="lower_tissue_attachment")], 0.00002)
    root = BRIDGE_ROOT
    root_path = f"/{root}"
    links = "\n".join(link_usda(link, root_path, {}) for link in (up, lo))
    joint = joint_usda(Joint("ContinuityJoint", "fixed", "UpperAnchor", "LowerAnchor", None, (0, 0, -0.011), (0, 0, 0.011)), root_path)
    intact = nested_over(["Links", "UpperAnchor", "Visuals", "FibreHalf"], [f'rel material:binding = </{root}/Looks/DenseFibre>'])
    released = "\n".join(
        [
            nested_over(["Links", "UpperAnchor", "Visuals", "FibreHalf"], [f'rel material:binding = </{root}/Looks/LooseFibre>']),
            nested_over(["Joints", "ContinuityJoint"], [], indent="            ").replace('over "ContinuityJoint"\n', 'over "ContinuityJoint" (\n                active = false\n            )\n'),
        ]
    )
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Reusable two-anchor physical adhesion bridge for safe-plane dissection research."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend variantSets = "state"
    variants = {{ string state = "intact" }}
    customData = {{
        string drAnmarAssetId = "dranmar-adhesion-bridge-v1"
        bool drAnmarClinicalValidation = false
        string drAnmarReleaseContract = "remove_or_deactivate_continuity_joint"
    }}
)
{{
{visual_materials_scope(root)}
{physics_materials_scope()}
    def Scope "Links"
    {{
{links}
    }}
    def Scope "Joints"
    {{
{joint}
    }}
    variantSet "state" = {{
        "intact"
        {{
{intact}
        }}
        "released"
        {{
{released}
        }}
    }}
}}
'''


def protected_structure_usda(
    *,
    root: str,
    asset_id: str,
    kind: str,
    left_mesh: trimesh.Trimesh,
    right_mesh: trimesh.Trimesh,
    left_points: Sequence[Sequence[float]],
    right_points: Sequence[Sequence[float]],
    radius: float,
    material: str,
    injured_material: str,
    physics_material: str,
    labels: tuple[str, ...],
) -> str:
    left = Link("ProximalSegment", (0, 0, 0), [Visual("Visual", left_mesh, material, labels)], polyline_colliders(left_points, radius * 1.05, physics_material, "C"), 0.0030 if kind == "vessel" else 0.0016, labels)
    right = Link("DistalSegment", (0, 0, 0), [Visual("Visual", right_mesh, material, labels)], polyline_colliders(right_points, radius * 1.05, physics_material, "C"), 0.0030 if kind == "vessel" else 0.0016, labels)
    connection = np.asarray(left_points[-1], dtype=float)
    joint = Joint("ContinuityJoint", "fixed", "ProximalSegment", "DistalSegment", None, tuple(connection), tuple(connection), max_force=0.0)
    frames = {
        "injury_site": _frame("ProximalSegment", connection, f"{kind}_injury_site"),
        "proximal_anchor": _frame("ProximalSegment", left_points[0], f"{kind}_proximal_anchor"),
        "distal_anchor": _frame("DistalSegment", right_points[-1], f"{kind}_distal_anchor"),
        "emission_source": _frame("ProximalSegment", connection, f"{kind}_fluid_or_signal_source"),
    }
    links = "\n\n".join(link_usda(link, f"/{root}", frames) for link in (left, right))
    joint_text = joint_usda(joint, f"/{root}")
    def segment_material(material_name: str) -> str:
        return f'''            over "Links"
            {{
                over "ProximalSegment"
                {{
                    over "Visuals"
                    {{
                        over "Visual"
                        {{
                            rel material:binding = </{root}/Looks/{material_name}>
                        }}
                    }}
                }}
                over "DistalSegment"
                {{
                    over "Visuals"
                    {{
                        over "Visual"
                        {{
                            rel material:binding = </{root}/Looks/{material_name}>
                        }}
                    }}
                }}
            }}'''
    intact = segment_material(material)
    injured_joint = '''            over "Joints"
            {
                over "ContinuityJoint" (
                    active = false
                )
                {
                }
            }'''
    injured = "\n".join(
        [
            segment_material(injured_material),
            injured_joint,
        ]
    )
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "DrAnmar protected {kind} branch with a removable continuity joint and explicit injury state."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend variantSets = "integrity"
    variants = {{ string integrity = "intact" }}
    customData = {{
        string drAnmarAssetId = "{asset_id}"
        string drAnmarStructureKind = "{kind}"
        bool drAnmarClinicalValidation = false
        string drAnmarContinuityContract = "independent_rigid_halves_connected_by_runtime_removable_joint"
    }}
)
{{
{visual_materials_scope(root)}
{physics_materials_scope()}
    def Scope "Links"
    {{
{links}
    }}
    def Scope "Joints"
    {{
{joint_text}
    }}
    variantSet "integrity" = {{
        "intact"
        {{
{intact}
        }}
        "injured"
        {{
{injured}
        }}
    }}
}}
'''


def bridge_block(spec: BridgeSpec) -> str:
    x, y, _ = spec.position
    top_z, bottom_z = 0.036, 0.014
    fibre_material = "DenseFibre" if spec.bridge_class == "dense_fibrous_band" else "VascularAdhesion" if spec.bridge_class == "vascularized_adhesion" else "LooseFibre"
    strands = 9 if spec.bridge_class == "dense_fibrous_band" else 7 if spec.bridge_class == "vascularized_adhesion" else 5
    radius = 0.00125 if spec.bridge_class == "dense_fibrous_band" else 0.00105
    upper_mesh = fibre_half_mesh(top_z - bottom_z, radius, True, strands)
    lower_mesh = fibre_half_mesh(top_z - bottom_z, radius, False, strands)
    upper_visual = mesh_usda(Visual("FibreHalf", upper_mesh, fibre_material, (spec.bridge_class,)), f"/{TISSUE_ROOT}/Looks/{fibre_material}", "                    ")
    lower_visual = mesh_usda(Visual("FibreHalf", lower_mesh, fibre_material, (spec.bridge_class,)), f"/{TISSUE_ROOT}/Looks/{fibre_material}", "                    ")
    upper_col = collider_usda(Collider("AttachmentVolume", "sphere", (0, 0, 0), radius=0.0028, physics_material="FibrePhysics", role="superficial_attachment_volume", collision_enabled=False), f"/{TISSUE_ROOT}", "                    ")
    lower_col = collider_usda(Collider("AttachmentVolume", "sphere", (0, 0, 0), radius=0.0028, physics_material="FibrePhysics", role="target_bed_attachment_volume", collision_enabled=False), f"/{TISSUE_ROOT}", "                    ")
    nearest = spec.nearest_structure or "none"
    return f'''        def Xform "Bridge_{spec.index:02d}" (
            customData = {{
                int bridgeIndex = {spec.index}
                string bridgeClass = "{spec.bridge_class}"
                bool targetBridge = {str(spec.target).lower()}
                string recommendedMode = "{spec.recommended_mode}"
                double mechanicalWorkThresholdJ = {f(spec.mechanical_work_j)}
                double hydroVolumeThresholdMl = {f(spec.hydro_volume_ml)}
                double energyDoseThresholdJ = {f(spec.energy_dose_j)}
                string nearestProtectedStructure = "{nearest}"
                double protectedStructureClearanceM = {f(spec.clearance_m)}
            }}
        )
        {{
            def Xform "UpperAnchor" (
                prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
            )
            {{
                double3 xformOp:translate = {vec((x, y, top_z))}
                uniform token[] xformOpOrder = ["xformOp:translate"]
                bool physics:rigidBodyEnabled = true
                bool physics:kinematicEnabled = false
                float physics:mass = 0.00002
                vector3f physics:diagonalInertia = (2e-9, 2e-9, 2e-9)
                quatf physics:principalAxes = (1, 0, 0, 0)
                float physxRigidBody:linearDamping = 0.18
                float physxRigidBody:angularDamping = 0.22
                def Scope "Visuals"
                {{
{upper_visual}
                }}
                def Scope "Collisions"
                {{
{upper_col}
                }}
                def Scope "Frames"
                {{
                    def Xform "attachment_reference"
                    {{
                        custom string drAnmar:role = "superficial_tissue_attachment_reference"
                    }}
                }}
            }}
            def Xform "LowerAnchor" (
                prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
            )
            {{
                double3 xformOp:translate = {vec((x, y, bottom_z))}
                uniform token[] xformOpOrder = ["xformOp:translate"]
                bool physics:rigidBodyEnabled = true
                bool physics:kinematicEnabled = false
                float physics:mass = 0.00002
                vector3f physics:diagonalInertia = (2e-9, 2e-9, 2e-9)
                quatf physics:principalAxes = (1, 0, 0, 0)
                float physxRigidBody:linearDamping = 0.18
                float physxRigidBody:angularDamping = 0.22
                def Scope "Visuals"
                {{
{lower_visual}
                }}
                def Scope "Collisions"
                {{
{lower_col}
                }}
                def Scope "Frames"
                {{
                    def Xform "attachment_reference"
                    {{
                        custom string drAnmar:role = "target_bed_attachment_reference"
                    }}
                }}
            }}
            def PhysicsFixedJoint "ContinuityJoint"
            {{
                rel physics:body0 = </{TISSUE_ROOT}/AdhesionBridges/Bridge_{spec.index:02d}/UpperAnchor>
                rel physics:body1 = </{TISSUE_ROOT}/AdhesionBridges/Bridge_{spec.index:02d}/LowerAnchor>
                point3f physics:localPos0 = (0, 0, -0.011)
                point3f physics:localPos1 = (0, 0, 0.011)
                quatf physics:localRot0 = (1, 0, 0, 0)
                quatf physics:localRot1 = (1, 0, 0, 0)
                bool physics:collisionEnabled = false
            }}
        }}'''


def tissue_usda(bundle: ToolBundle) -> str:
    superficial = mesh_usda_unbound("SuperficialFlap", bundle.superficial_mesh, f"/{TISSUE_ROOT}/Looks/SuperficialTissue", "    ", True)
    target = mesh_usda_unbound("TargetBed", bundle.target_bed_mesh, f"/{TISSUE_ROOT}/Looks/TargetBed", "    ", True)
    organ = mesh_usda_unbound("ProtectedOrgan", bundle.organ_mesh, f"/{TISSUE_ROOT}/Looks/ProtectedOrgan", "    ")
    fixture = mesh_usda_unbound("FixtureBase", bundle.fixture_mesh, f"/{TISSUE_ROOT}/Looks/FixtureMaterial", "    ")
    bridges = "\n".join(bridge_block(spec) for spec in bundle.bridge_specs)
    return f'''#usda 1.0
(
    defaultPrim = "{TISSUE_ROOT}"
    doc = "Layered safe-plane dissection substrate with 28 removable adhesions and protected vessel, nerve, and duct branches."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{TISSUE_ROOT}" (
    prepend variantSets = "visual_state"
    variants = {{ string visual_state = "initial" }}
    customData = {{
        string drAnmarAssetId = "dranmar-safeplane-tissue-demo-v1"
        bool drAnmarClinicalValidation = false
        int drAnmarAdhesionBridgeCount = {len(bundle.bridge_specs)}
        int drAnmarProtectedStructureCount = 3
        string drAnmarTopologyContract = "two_solver_ready_tissue_surfaces_connected_by_two_anchor_bridge_joints"
    }}
)
{{
{visual_materials_scope(TISSUE_ROOT)}
{physics_materials_scope()}
    def Scope "Anatomy"
    {{
{superficial}
{target}
{organ}
{fixture}
    }}
    def Cube "TargetBedFixtureLeft" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysxRigidBodyAPI"]
    )
    {{
        bool physics:kinematicEnabled = true
        bool physics:collisionEnabled = false
        double size = 1
        double3 xformOp:translate = (-0.078, 0, 0.014)
        double3 xformOp:scale = (0.010, 0.150, 0.012)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        token visibility = "invisible"
        uniform token purpose = "guide"
    }}
    def Cube "TargetBedFixtureRight" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysxRigidBodyAPI"]
    )
    {{
        bool physics:kinematicEnabled = true
        bool physics:collisionEnabled = false
        double size = 1
        double3 xformOp:translate = (0.078, 0, 0.014)
        double3 xformOp:scale = (0.010, 0.150, 0.012)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        token visibility = "invisible"
        uniform token purpose = "guide"
    }}
    def Scope "ProtectedStructures"
    {{
        def Xform "Vessel" (
            prepend references = @./dranmar_protected_vessel_branch.usda@
        )
        {{
        }}
        def Xform "Nerve" (
            prepend references = @./dranmar_protected_nerve_branch.usda@
        )
        {{
        }}
        def Xform "Duct" (
            prepend references = @./dranmar_protected_duct_branch.usda@
        )
        {{
        }}
    }}
    def Scope "AdhesionBridges"
    {{
{bridges}
    }}
    def Scope "Frames"
    {{
        def Xform "safe_plane_reference"
        {{
            custom string drAnmar:role = "safe_dissection_plane"
            double3 xformOp:translate = (0, 0, 0.025)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "superficial_left_capture"
        {{
            custom string drAnmar:role = "left_traction_capture_region"
            double3 xformOp:translate = (0, -0.050, 0.037)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "superficial_right_capture"
        {{
            custom string drAnmar:role = "right_traction_capture_region"
            double3 xformOp:translate = (0, 0.050, 0.037)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "roi_center"
        {{
            custom string drAnmar:role = "dissection_roi_center"
            double3 xformOp:translate = (0, 0, 0.030)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}
    variantSet "visual_state" = {{
        "initial"
        {{
            over "AdhesionBridges"
            {{
                token visibility = "inherited"
            }}
        }}
        "complete"
        {{
            over "AdhesionBridges"
            {{
                token visibility = "invisible"
            }}
        }}
    }}
}}
'''


def scissors_cartridge_usda(bundle: ToolBundle) -> str:
    mesh = mesh_usda_unbound("Visual", bundle.scissors_cartridge, f"/{SCISSORS_ROOT}/Looks/BladeSteel", "    ")
    mp = box_mass_properties([bundle.scissors_cartridge], 0.073)
    fresh = nested_over(["Visual"], [f'rel material:binding = </{SCISSORS_ROOT}/Looks/BladeSteel>'])
    spent = nested_over(["Visual"], [f'rel material:binding = </{SCISSORS_ROOT}/Looks/SpentBlade>'])
    return f'''#usda 1.0
(
    defaultPrim = "{SCISSORS_ROOT}"
    doc = "Replaceable guarded microsurgical scissors cartridge for DrAnmar SafePlane dissection."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{SCISSORS_ROOT}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    prepend variantSets = "state"
    variants = {{ string state = "fresh" }}
)
{{
    bool physics:rigidBodyEnabled = true
    bool physics:kinematicEnabled = false
    float physics:mass = {f(mp['mass_kg'])}
    point3f physics:centerOfMass = {vec(mp['center_of_mass_m'])}
    vector3f physics:diagonalInertia = {vec(mp['diagonal_inertia_kg_m2'])}
    quatf physics:principalAxes = {quat(mp['principal_axes_wxyz'])}
{visual_materials_scope(SCISSORS_ROOT)}
{physics_materials_scope()}
{mesh}
    def Scope "Frames"
    {{
        def Xform "cartridge_grasp"
        {{
            custom string drAnmar:role = "scissors_cartridge_grasp"
            double3 xformOp:translate = (0, 0, 0.010)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "blade_tip"
        {{
            custom string drAnmar:role = "micro_scissors_blade_tip"
            double3 xformOp:translate = (0, 0, 0.056)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}
    variantSet "state" = {{
        "fresh"
        {{
{fresh}
        }}
        "spent"
        {{
{spent}
        }}
    }}
}}
'''


def particle_usda(root: str, material: str, radius: float, asset_id: str, role: str) -> str:
    sphere = ellipsoid_mesh((radius, radius, radius), (0, 0, 0), 2)
    visual = mesh_usda_unbound("Visual", sphere, f"/{root}/Looks/{material}", "    ")
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "{role} visual particle for DrAnmar SafePlane dissection research."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    customData = {{
        string drAnmarAssetId = "{asset_id}"
        bool drAnmarClinicalValidation = false
    }}
)
{{
{visual_materials_scope(root)}
{visual}
}}
'''


def export_scene(path: Path, entries: Sequence[tuple[str, trimesh.Trimesh, str]]) -> None:
    scene = trimesh.Scene()
    for name, mesh, material in entries:
        scene.add_geometry(pbr(mesh, material), node_name=name, geom_name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))


PARENT_LINK = {
    "Mount": None,
    "LeftTractionCarriage": "Mount",
    "RightTractionCarriage": "Mount",
    "LeftPadPivot": "LeftTractionCarriage",
    "RightPadPivot": "RightTractionCarriage",
    "LeftTractionPad": "LeftPadPivot",
    "RightTractionPad": "RightPadPivot",
    "LeftSpreader": "Mount",
    "RightSpreader": "Mount",
    "HydroGimbal": "Mount",
    "HydroNozzle": "HydroGimbal",
    "ScissorCarriage": "Mount",
    "ScissorGuard": "ScissorCarriage",
    "MovingScissorBlade": "ScissorCarriage",
    "EnergyTip": "Mount",
    "SuctionValve": "Mount",
    "HydroValve": "Mount",
    "IrrigationValve": "Mount",
}


def phase_parameters(phase: str) -> dict[str, float]:
    phases = {
        "inspect": dict(lt=0, rt=0, lp=0, rp=0, lc=0, rc=0, ls=0, rs=0, hp=0, he=0, se=0, sg=0, sb=0, ee=0, sv=0, hv=0, iv=0),
        "capture": dict(lt=-0.006, rt=0.006, lp=-8, rp=8, lc=0.003, rc=0.003, ls=0, rs=0, hp=0, he=0, se=0, sg=0, sb=0, ee=0, sv=0.002, hv=0, iv=0),
        "traction": dict(lt=-0.026, rt=0.026, lp=-14, rp=14, lc=0.004, rc=0.004, ls=0, rs=0, hp=0, he=0, se=0, sg=0, sb=0, ee=0, sv=0.003, hv=0, iv=0),
        "blunt": dict(lt=-0.028, rt=0.028, lp=-16, rp=16, lc=0.004, rc=0.004, ls=-0.018, rs=0.018, hp=0, he=0, se=0, sg=0, sb=0, ee=0, sv=0.004, hv=0, iv=0.002),
        "hydro": dict(lt=-0.030, rt=0.030, lp=-17, rp=17, lc=0.004, rc=0.004, ls=-0.016, rs=0.016, hp=12, he=0.042, se=0, sg=0, sb=0, ee=0, sv=0.005, hv=0.007, iv=0.002),
        "scissors": dict(lt=-0.031, rt=0.031, lp=-18, rp=18, lc=0.004, rc=0.004, ls=-0.014, rs=0.014, hp=0, he=0, se=0.046, sg=-0.010, sb=30, ee=0, sv=0.006, hv=0, iv=0.001),
        "energy": dict(lt=-0.031, rt=0.031, lp=-18, rp=18, lc=0.004, rc=0.004, ls=-0.014, rs=0.014, hp=0, he=0, se=0, sg=0, sb=0, ee=0.042, sv=0.006, hv=0, iv=0.001),
        "verify": dict(lt=-0.032, rt=0.032, lp=-18, rp=18, lc=0.003, rc=0.003, ls=0, rs=0, hp=0, he=0, se=0, sg=0, sb=0, ee=0, sv=0.003, hv=0, iv=0),
        "complete": dict(lt=0, rt=0, lp=0, rp=0, lc=0, rc=0, ls=0, rs=0, hp=0, he=0, se=0, sg=0, sb=0, ee=0, sv=0, hv=0, iv=0),
        "abort": dict(lt=0, rt=0, lp=0, rp=0, lc=0, rc=0, ls=0, rs=0, hp=0, he=0, se=0, sg=0, sb=0, ee=0, sv=0.008, hv=0, iv=0.005),
    }
    return phases[phase]


def _axis_vector(axis: str) -> np.ndarray:
    return {"X": np.asarray([1.0, 0.0, 0.0]), "Y": np.asarray([0.0, 1.0, 0.0]), "Z": np.asarray([0.0, 0.0, 1.0])}[axis]


def local_motion_transform(link_name: str, phase: str) -> np.ndarray:
    p = phase_parameters(phase)
    T = np.eye(4)
    translation = np.zeros(3)
    rotation = np.eye(3)
    if link_name == "LeftTractionCarriage": translation[1] = p["lt"]
    elif link_name == "RightTractionCarriage": translation[1] = p["rt"]
    elif link_name == "LeftPadPivot": rotation = rotation_matrix((1, 0, 0), math.radians(p["lp"]))
    elif link_name == "RightPadPivot": rotation = rotation_matrix((1, 0, 0), math.radians(p["rp"]))
    elif link_name == "LeftTractionPad": translation[2] = p["lc"]
    elif link_name == "RightTractionPad": translation[2] = p["rc"]
    elif link_name == "LeftSpreader": translation[1] = p["ls"]
    elif link_name == "RightSpreader": translation[1] = p["rs"]
    elif link_name == "HydroGimbal": rotation = rotation_matrix((1, 0, 0), math.radians(p["hp"]))
    elif link_name == "HydroNozzle": translation[2] = p["he"]
    elif link_name == "ScissorCarriage": translation[2] = p["se"]
    elif link_name == "ScissorGuard": translation[2] = p["sg"]
    elif link_name == "MovingScissorBlade": rotation = rotation_matrix((0, 1, 0), math.radians(p["sb"]))
    elif link_name == "EnergyTip": translation[2] = p["ee"]
    elif link_name == "SuctionValve": translation[1] = p["sv"]
    elif link_name == "HydroValve": translation[1] = p["hv"]
    elif link_name == "IrrigationValve": translation[1] = p["iv"]
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def link_world_transform(bundle: ToolBundle, link_name: str, phase: str, cache: dict[str, np.ndarray] | None = None) -> np.ndarray:
    if cache is None:
        cache = {}
    if link_name in cache:
        return cache[link_name]
    link = bundle.links[link_name]
    base = np.eye(4)
    base[:3, 3] = np.asarray(link.translation, dtype=float)
    local = base @ local_motion_transform(link_name, phase)
    parent = PARENT_LINK[link_name]
    world = local if parent is None else link_world_transform(bundle, parent, phase, cache) @ local
    cache[link_name] = world
    return world


def world_visual_entries(bundle: ToolBundle, phase: str = "inspect") -> list[tuple[str, trimesh.Trimesh, str]]:
    out: list[tuple[str, trimesh.Trimesh, str]] = []
    cache: dict[str, np.ndarray] = {}
    for link_name, link in bundle.links.items():
        T = link_world_transform(bundle, link_name, phase, cache)
        for visual in link.visuals:
            mesh = visual.mesh.copy()
            mesh.apply_transform(T)
            out.append((f"{link_name}_{visual.name}", mesh, visual.material))
    return out


def collider_mesh(collider: Collider) -> trimesh.Trimesh:
    if collider.kind == "box":
        assert collider.size is not None
        mesh = box_mesh(collider.size, collider.center)
    elif collider.kind == "cylinder":
        assert collider.radius is not None and collider.height is not None
        mesh = cylinder_axis(collider.radius, collider.height, collider.axis, collider.center)
    elif collider.kind == "sphere":
        assert collider.radius is not None
        mesh = ellipsoid_mesh((collider.radius, collider.radius, collider.radius), collider.center, 2)
    else:
        raise ValueError(collider.kind)
    R = np.asarray(
        [
            [1 - 2 * (collider.orientation_wxyz[2] ** 2 + collider.orientation_wxyz[3] ** 2), 2 * (collider.orientation_wxyz[1] * collider.orientation_wxyz[2] - collider.orientation_wxyz[0] * collider.orientation_wxyz[3]), 2 * (collider.orientation_wxyz[1] * collider.orientation_wxyz[3] + collider.orientation_wxyz[0] * collider.orientation_wxyz[2])],
            [2 * (collider.orientation_wxyz[1] * collider.orientation_wxyz[2] + collider.orientation_wxyz[0] * collider.orientation_wxyz[3]), 1 - 2 * (collider.orientation_wxyz[1] ** 2 + collider.orientation_wxyz[3] ** 2), 2 * (collider.orientation_wxyz[2] * collider.orientation_wxyz[3] - collider.orientation_wxyz[0] * collider.orientation_wxyz[1])],
            [2 * (collider.orientation_wxyz[1] * collider.orientation_wxyz[3] - collider.orientation_wxyz[0] * collider.orientation_wxyz[2]), 2 * (collider.orientation_wxyz[2] * collider.orientation_wxyz[3] + collider.orientation_wxyz[0] * collider.orientation_wxyz[1]), 1 - 2 * (collider.orientation_wxyz[1] ** 2 + collider.orientation_wxyz[2] ** 2)],
        ]
    )
    # collider primitive functions already applied their translation; rotate about center only when required
    if not np.allclose(R, np.eye(3), atol=1e-9):
        center = np.asarray(collider.center)
        mesh.apply_translation(-center)
        T = np.eye(4)
        T[:3, :3] = R
        mesh.apply_transform(T)
        mesh.apply_translation(center)
    return mesh


def collision_debug_entries(bundle: ToolBundle, phase: str = "traction") -> list[tuple[str, trimesh.Trimesh, str]]:
    out = world_visual_entries(bundle, phase)
    cache: dict[str, np.ndarray] = {}
    for link_name, link in bundle.links.items():
        T = link_world_transform(bundle, link_name, phase, cache)
        for collider in link.colliders:
            if not collider.author_enabled:
                continue
            mesh = collider_mesh(collider)
            mesh.apply_transform(T)
            out.append((f"{link_name}_{collider.name}", mesh, "CollisionDebug"))
    return out


def axis_entries(bundle: ToolBundle, phase: str = "inspect", length: float = 0.012, radius: float = 0.00045) -> list[tuple[str, trimesh.Trimesh, str]]:
    out: list[tuple[str, trimesh.Trimesh, str]] = []
    cache: dict[str, np.ndarray] = {}
    for name, data in bundle.frames.items():
        T = link_world_transform(bundle, str(data["parent_link"]), phase, cache)
        p = T[:3, :3] @ np.asarray(data["position"], dtype=float) + T[:3, 3]
        R = T[:3, :3]
        for index, (direction, material) in enumerate(((R[:, 0], "GuideRed"), (R[:, 1], "GuideGreen"), (R[:, 2], "GuideBlue"))):
            out.append((f"{name}_{index}", capsule_between(p, p + direction * length, radius), material))
    return out


def deform_superficial(mesh: trimesh.Trimesh, traction: float, separation: float) -> trimesh.Trimesh:
    result = mesh.copy()
    vertices = np.asarray(result.vertices).copy()
    y_norm = np.clip(np.abs(vertices[:, 1]) / 0.062, 0.0, 1.0)
    x_norm = np.clip(np.abs(vertices[:, 0]) / 0.082, 0.0, 1.0)
    vertices[:, 2] += traction * (0.007 + 0.027 * y_norm**1.4)
    vertices[:, 1] *= 1.0 + 0.15 * traction
    vertices[:, 2] += separation * 0.010 * (1.0 - 0.45 * x_norm)
    result.vertices = vertices
    result.fix_normals()
    return result


def bridge_entries(bundle: ToolBundle, released_fraction: float = 0.0) -> list[tuple[str, trimesh.Trimesh, str]]:
    out: list[tuple[str, trimesh.Trimesh, str]] = []
    release_count = int(round(len(bundle.bridge_specs) * max(0.0, min(1.0, released_fraction))))
    # release loose fibres first, then vascularized adhesions, then dense bands
    order = sorted(bundle.bridge_specs, key=lambda spec: ({"loose_connective_fibre": 0, "vascularized_adhesion": 1, "dense_fibrous_band": 2}[spec.bridge_class], spec.index))
    released = {spec.index for spec in order[:release_count]}
    for spec in bundle.bridge_specs:
        if spec.index in released:
            continue
        x, y, _ = spec.position
        material = "DenseFibre" if spec.bridge_class == "dense_fibrous_band" else "VascularAdhesion" if spec.bridge_class == "vascularized_adhesion" else "LooseFibre"
        strands = 9 if spec.bridge_class == "dense_fibrous_band" else 7 if spec.bridge_class == "vascularized_adhesion" else 5
        full = fibre_half_mesh(0.022, 0.0012, True, strands)
        lower = fibre_half_mesh(0.022, 0.0012, False, strands)
        full = trimesh.util.concatenate([transform(full, (x, y, 0.036)), transform(lower, (x, y, 0.014))])
        out.append((f"Bridge_{spec.index:02d}", full, material))
    return out


def protected_structure_entries(bundle: ToolBundle, injury: str | None = None) -> list[tuple[str, trimesh.Trimesh, str]]:
    out: list[tuple[str, trimesh.Trimesh, str]] = []
    for name, left, right, material, injured_material in (
        ("Vessel", bundle.vessel_left, bundle.vessel_right, "VesselMaterial", "VesselInjured"),
        ("Nerve", bundle.nerve_left, bundle.nerve_right, "NerveMaterial", "NerveInjured"),
        ("Duct", bundle.duct_left, bundle.duct_right, "DuctMaterial", "DuctInjured"),
    ):
        left_mesh = left.copy()
        right_mesh = right.copy()
        mat = material
        if injury == name.lower():
            left_mesh.apply_translation((-0.003, 0, 0.001))
            right_mesh.apply_translation((0.003, 0, -0.001))
            mat = injured_material
        out.extend(((f"{name}Left", left_mesh, mat), (f"{name}Right", right_mesh, mat)))
    return out


def tissue_entries(bundle: ToolBundle, state: str = "initial", injury: str | None = None) -> list[tuple[str, trimesh.Trimesh, str]]:
    state_params = {
        "initial": (0.0, 0.0, 0.0),
        "capture": (0.10, 0.00, 0.0),
        "traction": (0.42, 0.08, 0.10),
        "blunt": (0.58, 0.24, 0.34),
        "hydro": (0.70, 0.38, 0.66),
        "scissors": (0.78, 0.50, 0.86),
        "energy": (0.82, 0.58, 0.95),
        "complete": (0.90, 0.68, 1.00),
    }
    traction, separation, released = state_params[state]
    superficial = deform_superficial(bundle.superficial_mesh, traction, separation)
    out = [
        ("Fixture", bundle.fixture_mesh.copy(), "FixtureMaterial"),
        ("ProtectedOrgan", bundle.organ_mesh.copy(), "ProtectedOrgan"),
        ("TargetBed", bundle.target_bed_mesh.copy(), "TargetBed"),
        ("SuperficialFlap", superficial, "SuperficialTissue"),
    ]
    out.extend(bridge_entries(bundle, released))
    out.extend(protected_structure_entries(bundle, injury))
    return out


def franka_proxy_entries(bundle: ToolBundle, phase: str = "inspect") -> list[tuple[str, trimesh.Trimesh, str]]:
    out: list[tuple[str, trimesh.Trimesh, str]] = []
    joints = [(0, 0, 0.05), (0, 0, 0.30), (0.18, 0, 0.48), (0.05, 0, 0.68), (0.22, 0, 0.83), (0.05, 0, 1.00), (0.14, 0, 1.12), (0.14, 0, 1.22)]
    for index, (a, b) in enumerate(zip(joints[:-1], joints[1:])):
        out.append((f"ArmLink_{index:02d}", capsule_between(a, b, 0.035 if index < 3 else 0.028), "BodyPolymer"))
    for index, point in enumerate(joints):
        out.append((f"ArmJoint_{index:02d}", ellipsoid_mesh((0.045, 0.045, 0.045), point, 2), "AccentPolymer"))
    tool_offset = np.asarray([0.14, 0, 1.22])
    for name, mesh, material in world_visual_entries(bundle, phase):
        out.append((name, transform(mesh, tool_offset), material))
    return out


def exploded_entries(bundle: ToolBundle) -> list[tuple[str, trimesh.Trimesh, str]]:
    offsets = {
        "Mount": (0, 0, 0),
        "LeftTractionCarriage": (0, -0.060, 0.010), "RightTractionCarriage": (0, 0.060, 0.010),
        "LeftPadPivot": (-0.025, -0.078, 0.025), "RightPadPivot": (-0.025, 0.078, 0.025),
        "LeftTractionPad": (-0.045, -0.078, 0.040), "RightTractionPad": (-0.045, 0.078, 0.040),
        "LeftSpreader": (0.030, -0.045, 0.025), "RightSpreader": (0.030, 0.045, 0.025),
        "HydroGimbal": (-0.060, -0.035, 0.040), "HydroNozzle": (-0.075, -0.035, 0.070),
        "ScissorCarriage": (0, 0, 0.075), "ScissorGuard": (0.025, -0.025, 0.095), "MovingScissorBlade": (-0.025, 0.025, 0.105),
        "EnergyTip": (0.065, 0.035, 0.060), "SuctionValve": (0.055, 0.075, 0), "HydroValve": (-0.055, 0.075, 0), "IrrigationValve": (0, 0.085, 0),
    }
    out: list[tuple[str, trimesh.Trimesh, str]] = []
    cache: dict[str, np.ndarray] = {}
    for link_name, link in bundle.links.items():
        T = link_world_transform(bundle, link_name, "inspect", cache).copy()
        T[:3, 3] += np.asarray(offsets.get(link_name, (0, 0, 0)))
        for visual in link.visuals:
            mesh = visual.mesh.copy()
            mesh.apply_transform(T)
            out.append((f"{link_name}_{visual.name}", mesh, visual.material))
    return out


def redefine_rigid_proxy_usda(bundle: ToolBundle) -> str:
    visuals: list[Visual] = []
    cache: dict[str, np.ndarray] = {}
    for link_name, link in bundle.links.items():
        T = link_world_transform(bundle, link_name, "inspect", cache)
        for visual in link.visuals:
            mesh = visual.mesh.copy()
            mesh.apply_transform(T)
            visuals.append(Visual(f"{link_name}_{visual.name}", mesh, visual.material, visual.labels))
    merged = trimesh.util.concatenate([visual.mesh for visual in visuals])
    mp = box_mass_properties([merged], 1.02)
    visual_blocks = "\n".join(mesh_usda(visual, f"/{PROXY_ROOT}/Looks/{visual.material}", "        ") for visual in visuals)
    return f'''#usda 1.0
(
    defaultPrim = "{PROXY_ROOT}"
    doc = "Rigid perception and planning proxy for the DrAnmar SafePlane Dissection Robot."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{PROXY_ROOT}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)
{{
    bool physics:rigidBodyEnabled = true
    bool physics:kinematicEnabled = false
    float physics:mass = {f(mp['mass_kg'])}
    point3f physics:centerOfMass = {vec(mp['center_of_mass_m'])}
    vector3f physics:diagonalInertia = {vec(mp['diagonal_inertia_kg_m2'])}
    quatf physics:principalAxes = {quat(mp['principal_axes_wxyz'])}
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(PROXY_ROOT)}
{physics_materials_scope()}
    def Scope "Visuals"
    {{
{visual_blocks}
    }}
    def Scope "Collisions"
    {{
{collider_usda(Collider('HousingCollider','box',(0,0,0.090),size=(0.142,0.126,0.180),physics_material='PolymerPhysics'), f'/{PROXY_ROOT}', '        ')}
{collider_usda(Collider('HeadCollider','cylinder',(0,0,0.180),radius=0.050,height=0.028,physics_material='PolymerPhysics'), f'/{PROXY_ROOT}', '        ')}
    }}
}}
'''


# Override the earlier simple proxy with the hierarchy-aware version.
rigid_proxy_usda = redefine_rigid_proxy_usda


def export_glbs(bundle: ToolBundle) -> list[Path]:
    outputs: list[Path] = []
    mapping = {
        "inspect": "initial",
        "capture": "capture",
        "traction": "traction",
        "blunt": "blunt",
        "hydro": "hydro",
        "scissors": "scissors",
        "energy": "energy",
        "verify": "complete",
    }
    for phase, tissue_state in mapping.items():
        entries = world_visual_entries(bundle, phase)
        entries += [(name, transform(mesh, (0, 0, WORK_PLANE_Z)), material) for name, mesh, material in tissue_entries(bundle, tissue_state)]
        path = GLB_ROOT / f"dranmar_safeplane_tool_{phase}.glb"
        export_scene(path, entries)
        outputs.append(path)
    path = GLB_ROOT / "dranmar_safeplane_tool_exploded.glb"
    export_scene(path, exploded_entries(bundle))
    outputs.append(path)
    path = GLB_ROOT / "dranmar_safeplane_tool_collision_debug.glb"
    export_scene(path, collision_debug_entries(bundle, "traction"))
    outputs.append(path)
    path = GLB_ROOT / "dranmar_safeplane_tool_frame_debug.glb"
    export_scene(path, world_visual_entries(bundle, "inspect") + axis_entries(bundle, "inspect"))
    outputs.append(path)
    path = GLB_ROOT / "dranmar_franka_safeplane_assembly.glb"
    export_scene(path, franka_proxy_entries(bundle, "inspect"))
    outputs.append(path)
    for state in ("initial", "traction", "blunt", "hydro", "scissors", "complete"):
        path = GLB_ROOT / f"dranmar_safeplane_tissue_{state}.glb"
        export_scene(path, tissue_entries(bundle, state))
        outputs.append(path)
    for injury in PROTECTED_STRUCTURE_NAMES:
        path = GLB_ROOT / f"dranmar_safeplane_complication_{injury}.glb"
        export_scene(path, tissue_entries(bundle, "complete", injury=injury))
        outputs.append(path)
    path = GLB_ROOT / "dranmar_micro_scissors_cartridge.glb"
    export_scene(path, [("ScissorsCartridge", bundle.scissors_cartridge, "BladeSteel")])
    outputs.append(path)
    for name, left, right, material in (
        ("protected_vessel", bundle.vessel_left, bundle.vessel_right, "VesselMaterial"),
        ("protected_nerve", bundle.nerve_left, bundle.nerve_right, "NerveMaterial"),
        ("protected_duct", bundle.duct_left, bundle.duct_right, "DuctMaterial"),
    ):
        path = GLB_ROOT / f"dranmar_{name}.glb"
        export_scene(path, [(f"{name}_left", left, material), (f"{name}_right", right, material)])
        outputs.append(path)
    return outputs


def add_mesh_to_axis(ax, mesh: trimesh.Trimesh, material: str, max_faces: int = 1000) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = np.asarray(mesh.faces)
    if len(faces) > max_faces:
        faces = faces[np.linspace(0, len(faces) - 1, max_faces, dtype=int)]
    triangles = np.asarray(mesh.vertices)[faces]
    rgba = np.asarray(material_color(material), dtype=float) / 255.0
    collection = Poly3DCollection(triangles, facecolors=[rgba], edgecolors="none", linewidths=0.0, alpha=float(rgba[3]))
    ax.add_collection3d(collection)


def configure_axis(ax, title: str, elev: float = 22, azim: float = -58) -> None:
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(-0.13, 0.13)
    ax.set_zlim(0.0, 0.255)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1.05))


def make_preview(bundle: ToolBundle) -> Path:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 10), dpi=165)
    phases = [
        ("inspect", "1  Inspect + map plane", "initial"),
        ("traction", "2  Distributed counter-traction", "traction"),
        ("blunt", "3  Blunt spreading", "blunt"),
        ("hydro", "4  Hydrodissection", "hydro"),
        ("scissors", "5  Guarded selective cutting", "scissors"),
        ("verify", "6  Connectivity verified", "complete"),
    ]
    for index, (phase, title, tissue_state) in enumerate(phases, 1):
        axis = figure.add_subplot(2, 3, index, projection="3d")
        for _, mesh, material in world_visual_entries(bundle, phase):
            add_mesh_to_axis(axis, mesh, material, 760)
        for _, mesh, material in tissue_entries(bundle, tissue_state):
            add_mesh_to_axis(axis, transform(mesh, (0, 0, WORK_PLANE_Z)), material, 760)
        configure_axis(axis, title)
    figure.suptitle("DrAnmar SafePlane Dissection Robot — traction, blunt separation, hydrodissection, protected cutting, verification", fontsize=16, y=0.98)
    figure.text(0.5, 0.014, "DrAnmar-owned, provider-neutral research asset • protected vessel, nerve and duct integrity tracked independently • Franka hand replacement", ha="center", fontsize=10)
    path = PREVIEW_ROOT / "dranmar_safeplane_dissection_robot_preview.png"
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def make_full_arm_preview(bundle: ToolBundle) -> Path:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(10, 10), dpi=175)
    axis = figure.add_subplot(111, projection="3d")
    for _, mesh, material in franka_proxy_entries(bundle, "inspect"):
        add_mesh_to_axis(axis, mesh, material, 1000)
    axis.set_xlim(-0.18, 0.38)
    axis.set_ylim(-0.26, 0.26)
    axis.set_zlim(0, 1.49)
    axis.view_init(elev=20, azim=-58)
    axis.set_axis_off()
    axis.set_box_aspect((0.6, 0.56, 1.45))
    axis.set_title("DrAnmar SafePlane Dissection Robot mounted at the Franka wrist", fontsize=15, pad=12)
    path = PREVIEW_ROOT / "dranmar_safeplane_dissection_robot_full_arm_preview.png"
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def noise_texture(base: tuple[int, int, int], size: int = 512, strength: int = 18, seed: int = 1) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = np.zeros((size, size, 3), dtype=np.int16)
    array[:] = np.asarray(base, dtype=np.int16)
    array += rng.normal(0, strength, (size, size, 1)).astype(np.int16)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def generate_textures() -> list[Path]:
    TEXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, base, strength, seed in [
        ("polymer_microtexture.png", (212, 218, 224), 9, 51),
        ("metal_microtexture.png", (145, 151, 160), 7, 52),
        ("pad_microtexture.png", (20, 135, 150), 12, 53),
        ("superficial_tissue_basecolor.png", (201, 108, 102), 16, 54),
        ("target_bed_basecolor.png", (166, 55, 51), 14, 55),
        ("organ_basecolor.png", (139, 31, 42), 12, 56),
        ("loose_fibre_basecolor.png", (236, 192, 159), 18, 57),
        ("vessel_basecolor.png", (145, 8, 7), 10, 58),
        ("nerve_basecolor.png", (242, 198, 41), 9, 59),
        ("duct_basecolor.png", (46, 180, 77), 10, 60),
    ]:
        path = TEXTURE_ROOT / name
        noise_texture(base, 512, strength, seed).save(path)
        outputs.append(path)
    image = Image.new("RGB", (1024, 256), (247, 249, 251))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
        small = ImageFont.truetype("DejaVuSans.ttf", 29)
    except OSError:
        font = None
        small = None
    draw.text((36, 44), "DrAnmar", fill=(18, 65, 112), font=font)
    draw.text((40, 148), "SAFEPLANE DISSECTION • RESEARCH ONLY", fill=(35, 45, 55), font=small)
    path = TEXTURE_ROOT / "label_dranmar.png"
    image.save(path)
    outputs.append(path)
    return outputs


def integration_module() -> str:
    if not INTEGRATION_PATH.is_file():
        raise FileNotFoundError(
            f"Canonical integration module is missing: {INTEGRATION_PATH}"
        )
    return INTEGRATION_PATH.read_text(encoding="utf-8")

    # Historical embedded source retained below only for archive compatibility.
    return textwrap.dedent(r'''
    # Copyright (c) 2026, DrAnmar Project Developers.
    # SPDX-License-Identifier: Apache-2.0
    """Isaac Lab integration for the DrAnmar SafePlane Dissection Robot.

    The module provides a Franka hand-replacement spawner, layered tissue
    substrate integration, distributed traction, four dissection modalities,
    protected-structure safety state, particle-fluid helpers, and a topology-
    based completion verifier. All parameters are provisional research values.
    """
    from __future__ import annotations

    import json
    import math
    from collections.abc import Iterable, Mapping, Sequence
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any

    CATALOG_SUBPATH = "Props/SurgicalDissection/SafePlaneDissectionRobot"
    ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
    ASSET_ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
    TOOL_PAYLOAD_USD = ASSET_ROOT / "dranmar_safeplane_dissection_tool_payload.usda"
    TOOL_STANDALONE_USD = ASSET_ROOT / "dranmar_safeplane_dissection_tool_standalone.usda"
    TOOL_RIGID_PROXY_USD = ASSET_ROOT / "dranmar_safeplane_dissection_tool_rigid_proxy.usda"
    TISSUE_DEMO_USD = ASSET_ROOT / "dranmar_safeplane_tissue_demo.usda"
    ADHESION_BRIDGE_USD = ASSET_ROOT / "dranmar_adhesion_bridge.usda"
    PROTECTED_VESSEL_USD = ASSET_ROOT / "dranmar_protected_vessel_branch.usda"
    PROTECTED_NERVE_USD = ASSET_ROOT / "dranmar_protected_nerve_branch.usda"
    PROTECTED_DUCT_USD = ASSET_ROOT / "dranmar_protected_duct_branch.usda"
    MICRO_SCISSORS_USD = ASSET_ROOT / "dranmar_micro_scissors_cartridge.usda"
    DISSECTION_TOPOLOGY_PATH = ASSET_ROOT / "dissection_topology.json"

    VALID_SCISSORS_STATES = frozenset({"fresh", "spent"})
    VALID_FLUID_STATES = frozenset({"full", "empty"})
    VALID_COLLECTION_STATES = frozenset({"empty", "partial", "full"})
    VALID_ENERGY_STATES = frozenset({"ready", "fault"})
    PROTECTED_STRUCTURES = ("vessel", "nerve", "duct")

    TOOL_JOINTS = {
        "left_traction": "left_traction_joint",
        "right_traction": "right_traction_joint",
        "left_pad_pitch": "left_pad_pitch_joint",
        "right_pad_pitch": "right_pad_pitch_joint",
        "left_pad_compliance": "left_pad_compliance_joint",
        "right_pad_compliance": "right_pad_compliance_joint",
        "left_spreader": "left_spreader_joint",
        "right_spreader": "right_spreader_joint",
        "hydro_pitch": "hydro_pitch_joint",
        "hydro_extension": "hydro_extension_joint",
        "scissor_extension": "scissor_extension_joint",
        "scissor_guard": "scissor_guard_joint",
        "scissor_blade": "scissor_blade_joint",
        "energy_tip_extension": "energy_tip_extension_joint",
        "suction_valve": "suction_valve_joint",
        "hydro_valve": "hydro_valve_joint",
        "irrigation_valve": "irrigation_valve_joint",
    }

    TOOL_FRAME_PATHS = {
        "panda_link8_mount": "Links/Mount/Frames/panda_link8_mount",
        "safeplane_tcp": "Links/Mount/Frames/safeplane_tcp",
        "safe_plane_reference": "Links/Mount/Frames/safe_plane_reference",
        "roi_center": "Links/Mount/Frames/roi_center",
        "suction_center": "Links/Mount/Frames/suction_center",
        "irrigation_center": "Links/Mount/Frames/irrigation_center",
        "stereo_left": "Links/Mount/Frames/stereo_left",
        "stereo_right": "Links/Mount/Frames/stereo_right",
        "depth_camera": "Links/Mount/Frames/depth_camera",
        "fluorescence_camera": "Links/Mount/Frames/fluorescence_camera",
        "thermal_camera": "Links/Mount/Frames/thermal_camera",
        "protected_structure_probe": "Links/Mount/Frames/protected_structure_probe",
        "hydro_nozzle_tip": "Links/HydroNozzle/Frames/hydro_nozzle_tip",
        "hydro_axis": "Links/HydroNozzle/Frames/hydro_axis",
        "scissor_cut_plane": "Links/ScissorCarriage/Frames/scissor_cut_plane",
        "scissor_guard_tip": "Links/ScissorGuard/Frames/scissor_guard_tip",
        "energy_tip": "Links/EnergyTip/Frames/energy_tip",
        "energy_axis": "Links/EnergyTip/Frames/energy_axis",
        "left_spreader_tip": "Links/LeftSpreader/Frames/left_spreader_tip",
        "right_spreader_tip": "Links/RightSpreader/Frames/right_spreader_tip",
        "left_traction_pad": "Links/LeftTractionPad/Frames/left_traction_pad",
        "right_traction_pad": "Links/RightTractionPad/Frames/right_traction_pad",
        "count_reference": "Links/Mount/Frames/count_reference",
        "disposal_reference": "Links/Mount/Frames/disposal_reference",
    }
    for _side in ("left", "right"):
        for _index in range(4):
            TOOL_FRAME_PATHS[f"{_side}_capture_cell_{_index}"] = f"Links/{_side.title()}TractionPad/Frames/{_side}_capture_cell_{_index}"

    PARTICLE_RADIUS_M = 0.00072
    PARTICLE_VOLUME_ML = 4.0 / 3.0 * math.pi * PARTICLE_RADIUS_M**3 * 1.0e6


    def frame_path(tool_path: str, name: str) -> str:
        try:
            suffix = TOOL_FRAME_PATHS[name]
        except KeyError as exc:
            raise KeyError(f"Unknown SafePlane frame {name!r}") from exc
        return f"{tool_path.rstrip('/')}/{suffix}"


    def tensor_value(value: Any):
        return value.torch if hasattr(value, "torch") else value


    def _check(value: str, allowed: frozenset[str], label: str) -> str:
        if value not in allowed:
            raise ValueError(f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}")
        return value


    def make_tool_cfg(
        prim_path: str = "/World/DrAnmarSafePlaneDissectionTool",
        *,
        scissors_state: str = "fresh",
        hydro_state: str = "full",
        irrigation_state: str = "full",
        collection_state: str = "empty",
        energy_state: str = "ready",
        position=(0.0, 0.0, 0.35),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    ):
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        _check(scissors_state, VALID_SCISSORS_STATES, "scissors_state")
        _check(hydro_state, VALID_FLUID_STATES, "hydro_state")
        _check(irrigation_state, VALID_FLUID_STATES, "irrigation_state")
        _check(collection_state, VALID_COLLECTION_STATES, "collection_state")
        _check(energy_state, VALID_ENERGY_STATES, "energy_state")
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(TOOL_STANDALONE_USD),
                variants={
                    "scissors_state": scissors_state,
                    "hydro_state": hydro_state,
                    "irrigation_state": irrigation_state,
                    "collection_state": collection_state,
                    "energy_state": energy_state,
                },
                activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=24,
                    solver_velocity_iteration_count=8,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=position,
                rot=orientation_wxyz,
                joint_pos={name: 0.0 for name in TOOL_JOINTS.values()},
            ),
            actuators={
                "traction": ImplicitActuatorCfg(
                    joint_names_expr=[".*traction_joint"], effort_limit_sim=120.0,
                    velocity_limit_sim=0.18, stiffness=4400.0, damping=150.0,
                ),
                "traction_pad": ImplicitActuatorCfg(
                    joint_names_expr=[".*pad_pitch_joint", ".*pad_compliance_joint"],
                    effort_limit_sim=48.0, velocity_limit_sim=1.2,
                    stiffness=1100.0, damping=72.0,
                ),
                "spreader": ImplicitActuatorCfg(
                    joint_names_expr=[".*spreader_joint"], effort_limit_sim=85.0,
                    velocity_limit_sim=0.14, stiffness=3200.0, damping=125.0,
                ),
                "hydro": ImplicitActuatorCfg(
                    joint_names_expr=["hydro_pitch_joint", "hydro_extension_joint"],
                    effort_limit_sim=80.0, velocity_limit_sim=1.0,
                    stiffness=3600.0, damping=115.0,
                ),
                "scissors": ImplicitActuatorCfg(
                    joint_names_expr=["scissor_extension_joint", "scissor_guard_joint", "scissor_blade_joint"],
                    effort_limit_sim=120.0, velocity_limit_sim=1.4,
                    stiffness=5200.0, damping=160.0,
                ),
                "energy_tip": ImplicitActuatorCfg(
                    joint_names_expr=["energy_tip_extension_joint"], effort_limit_sim=70.0,
                    velocity_limit_sim=0.18, stiffness=3500.0, damping=115.0,
                ),
                "valves": ImplicitActuatorCfg(
                    joint_names_expr=[".*_valve_joint"], effort_limit_sim=25.0,
                    velocity_limit_sim=0.25, stiffness=1600.0, damping=48.0,
                ),
            },
        )


    def make_rigid_proxy_cfg(
        prim_path: str = "/World/DrAnmarSafePlaneDissectionProxy",
        *, position=(0.0, 0.0, 0.35), orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    ):
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(TOOL_RIGID_PROXY_USD), activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=orientation_wxyz),
        )


    def _spawn_single_franka_with_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
        from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
        from isaaclab.sim.utils import create_prim, get_current_stage, select_usd_variants
        from pxr import Gf, Sdf, UsdPhysics

        robot = spawn_from_usd(prim_path, cfg, translation, orientation)
        stage = get_current_stage()
        disabled = {
            "panda_hand_joint", "panda_hand", "panda_finger_joint1", "panda_finger_joint2",
            "panda_leftfinger", "panda_rightfinger",
        }
        for prim in list(stage.Traverse()):
            if prim.GetPath().HasPrefix(Sdf.Path(prim_path)) and prim.GetName() in disabled:
                stage.OverridePrim(prim.GetPath()).SetActive(False)
        tool_path = f"{prim_path}/DrAnmarSafePlaneDissectionTool"
        create_prim(tool_path, usd_path=str(TOOL_PAYLOAD_USD), stage=stage)
        select_usd_variants(
            tool_path,
            {
                "scissors_state": cfg.scissors_state,
                "hydro_state": cfg.hydro_state,
                "irrigation_state": cfg.irrigation_state,
                "collection_state": cfg.collection_state,
                "energy_state": cfg.energy_state,
            },
        )
        joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/dranmar_safeplane_mount_joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{prim_path}/panda_link8")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        angle = math.radians(-45.0) / 2.0
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(math.cos(angle), 0, 0, math.sin(angle)))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        return robot


    def spawn_franka_with_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
        from isaaclab.sim.utils import clone
        return clone(_spawn_single_franka_with_tool)(
            prim_path, cfg, translation=translation, orientation=orientation, **kwargs
        )


    def make_franka_safeplane_dissection_robot_cfg(
        *, prim_path="/World/Robot", scissors_state="fresh", hydro_state="full",
        irrigation_state="full", collection_state="empty", energy_state="ready",
    ):
        _check(scissors_state, VALID_SCISSORS_STATES, "scissors_state")
        _check(hydro_state, VALID_FLUID_STATES, "hydro_state")
        _check(irrigation_state, VALID_FLUID_STATES, "irrigation_state")
        _check(collection_state, VALID_COLLECTION_STATES, "collection_state")
        _check(energy_state, VALID_ENERGY_STATES, "energy_state")
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.utils import configclass
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

        @configclass
        class FrankaSafePlaneUsdCfg(sim_utils.UsdFileCfg):
            scissors_state: str = "fresh"
            hydro_state: str = "full"
            irrigation_state: str = "full"
            collection_state: str = "empty"
            energy_state: str = "ready"
            func = spawn_franka_with_tool

        cfg = FRANKA_PANDA_CFG.copy()
        cfg.prim_path = prim_path
        cfg.spawn = FrankaSafePlaneUsdCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
            variants={"Gripper": "Default", "Mesh": "Performance"},
            scissors_state=scissors_state,
            hydro_state=hydro_state,
            irrigation_state=irrigation_state,
            collection_state=collection_state,
            energy_state=energy_state,
            activate_contact_sensors=True,
            rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,
            articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props,
        )
        cfg.init_state.joint_pos = {key: value for key, value in cfg.init_state.joint_pos.items() if "finger" not in key}
        cfg.init_state.joint_pos.update({name: 0.0 for name in TOOL_JOINTS.values()})
        cfg.actuators = {key: value for key, value in cfg.actuators.items() if key != "panda_hand"}
        cfg.actuators.update(
            {
                "safeplane_traction": ImplicitActuatorCfg(joint_names_expr=[".*traction_joint"], effort_limit_sim=120.0, velocity_limit_sim=0.18, stiffness=4400.0, damping=150.0),
                "safeplane_pads": ImplicitActuatorCfg(joint_names_expr=[".*pad_pitch_joint", ".*pad_compliance_joint"], effort_limit_sim=48.0, velocity_limit_sim=1.2, stiffness=1100.0, damping=72.0),
                "safeplane_spreader": ImplicitActuatorCfg(joint_names_expr=[".*spreader_joint"], effort_limit_sim=85.0, velocity_limit_sim=0.14, stiffness=3200.0, damping=125.0),
                "safeplane_hydro": ImplicitActuatorCfg(joint_names_expr=["hydro_pitch_joint", "hydro_extension_joint"], effort_limit_sim=80.0, velocity_limit_sim=1.0, stiffness=3600.0, damping=115.0),
                "safeplane_scissors": ImplicitActuatorCfg(joint_names_expr=["scissor_extension_joint", "scissor_guard_joint", "scissor_blade_joint"], effort_limit_sim=120.0, velocity_limit_sim=1.4, stiffness=5200.0, damping=160.0),
                "safeplane_energy": ImplicitActuatorCfg(joint_names_expr=["energy_tip_extension_joint"], effort_limit_sim=70.0, velocity_limit_sim=0.18, stiffness=3500.0, damping=115.0),
                "safeplane_valves": ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"], effort_limit_sim=25.0, velocity_limit_sim=0.25, stiffness=1600.0, damping=48.0),
            }
        )
        return cfg


    def _current_stage(stage=None):
        if stage is not None:
            return stage
        import omni.usd
        return omni.usd.get_context().get_stage()


    def _world_transform(stage, path: str):
        from pxr import Usd, UsdGeom
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"No valid prim at {path}")
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


    def spawn_tissue_demo(
        prim_path: str = "/World/DrAnmarSafePlaneTissue",
        *, translation=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    ):
        import isaaclab.sim as sim_utils
        cfg = sim_utils.UsdFileCfg(usd_path=str(TISSUE_DEMO_USD))
        return cfg.func(prim_path, cfg, translation=translation, orientation=orientation_wxyz)


    def apply_tissue_surface_deformables(root_path: str, *, self_collision=True, stage=None):
        stage = _current_stage(stage)
        from omni.physx.scripts import deformableUtils
        paths = []
        for child in ("SuperficialFlap", "TargetBed"):
            mesh_path = f"{root_path.rstrip('/')}/Anatomy/{child}"
            mesh = stage.GetPrimAtPath(mesh_path)
            if not mesh or not mesh.IsValid():
                raise ValueError(f"No tissue surface at {mesh_path}")
            success = deformableUtils.set_physics_surface_deformable_body(stage, mesh.GetPath())
            if success is False:
                raise RuntimeError(f"Failed to cook surface deformable at {mesh_path}")
            mesh.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
            if mesh.HasAPI("PhysxSurfaceDeformableBodyAPI"):
                mesh.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
            paths.append(mesh_path)
        return {"root_path": root_path, "mesh_paths": paths, "self_collision": bool(self_collision)}


    def create_deformable_attachment(deformable_path: str, target_path: str, attachment_path: str, *, stage=None):
        stage = _current_stage(stage)
        import omni.kit.commands
        try:
            result = omni.kit.commands.execute(
                "CreateAutoDeformableAttachment",
                target_attachment_path=attachment_path,
                attachable0_path=deformable_path,
                attachable1_path=target_path,
            )
            if result is not False:
                return attachment_path
        except Exception:
            pass
        result = omni.kit.commands.execute(
            "CreatePhysicsAttachment",
            target_attachment_path=attachment_path,
            actor0_path=deformable_path,
            actor1_path=target_path,
        )
        if result is False:
            raise RuntimeError(f"Failed to attach {deformable_path} to {target_path}")
        return attachment_path


    def remove_prims(paths: Iterable[str], *, stage=None):
        stage = _current_stage(stage)
        for path in paths:
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)


    def load_dissection_topology() -> dict[str, Any]:
        return json.loads(DISSECTION_TOPOLOGY_PATH.read_text(encoding="utf-8"))


    @dataclass
    class TractionCell:
        side: str
        index: int
        attachment_path: str
        released: bool = False


    @dataclass
    class BilateralTractionController:
        tool_path: str
        tissue_root: str
        cells: list[TractionCell] = field(default_factory=list)
        nominal_force_n: float = 1.4
        soft_force_limit_n: float = 3.0
        hard_force_limit_n: float = 5.0

        def capture(self, *, stage=None):
            stage = _current_stage(stage)
            superficial = f"{self.tissue_root.rstrip('/')}/Anatomy/SuperficialFlap"
            scope = f"{self.tissue_root.rstrip('/')}/RuntimeTractionAttachments"
            stage.DefinePrim(scope, "Scope")
            created: list[str] = []
            cells: list[TractionCell] = []
            try:
                for side in ("left", "right"):
                    link = side.title()
                    for index in range(4):
                        target = f"{self.tool_path.rstrip('/')}/Links/{link}TractionPad/Collisions/CaptureCell_{index:02d}"
                        attachment = f"{scope}/{side}_{index:02d}"
                        create_deformable_attachment(superficial, target, attachment, stage=stage)
                        created.append(attachment)
                        cells.append(TractionCell(side, index, attachment))
            except Exception:
                remove_prims(created, stage=stage)
                raise
            self.cells = cells
            return list(cells)

        def release_side(self, side: str, *, stage=None):
            stage = _current_stage(stage)
            released = []
            for cell in self.cells:
                if cell.side == side and not cell.released:
                    remove_prims([cell.attachment_path], stage=stage)
                    cell.released = True
                    released.append(cell.index)
            return released

        def update_force(self, left_force_n: float, right_force_n: float, *, stage=None):
            stage = _current_stage(stage)
            events = []
            for side, force in (("left", float(left_force_n)), ("right", float(right_force_n))):
                active = [cell for cell in self.cells if cell.side == side and not cell.released]
                if force >= self.hard_force_limit_n:
                    for cell in active:
                        remove_prims([cell.attachment_path], stage=stage)
                        cell.released = True
                    events.append({"side": side, "event": "hard_release", "force_n": force})
                elif force >= self.soft_force_limit_n and active:
                    cell = sorted(active, key=lambda value: abs(value.index - 1.5), reverse=True)[0]
                    remove_prims([cell.attachment_path], stage=stage)
                    cell.released = True
                    events.append({"side": side, "event": "peripheral_cell_release", "cell": cell.index, "force_n": force})
            return events

        def snapshot(self):
            return {
                side: {"active_cells": [cell.index for cell in self.cells if cell.side == side and not cell.released]}
                for side in ("left", "right")
            }


    @dataclass
    class BridgeRuntimeState:
        index: int
        bridge_class: str
        position: tuple[float, float, float]
        recommended_mode: str
        mechanical_threshold_j: float
        hydro_threshold_ml: float
        energy_threshold_j: float
        nearest_structure: str | None
        clearance_m: float
        attachment_paths: list[str] = field(default_factory=list)
        mechanical_work_j: float = 0.0
        hydro_volume_ml: float = 0.0
        energy_dose_j: float = 0.0
        released: bool = False
        release_mode: str | None = None


    @dataclass
    class AdhesionBridgeController:
        tissue_root: str
        states: dict[int, BridgeRuntimeState] = field(default_factory=dict)

        def __post_init__(self):
            topology = load_dissection_topology()
            self.states = {
                int(item["index"]): BridgeRuntimeState(
                    index=int(item["index"]),
                    bridge_class=str(item["bridge_class"]),
                    position=tuple(float(value) for value in item["position_m"]),
                    recommended_mode=str(item["recommended_mode"]),
                    mechanical_threshold_j=float(item["thresholds"]["mechanical_work_j"]),
                    hydro_threshold_ml=float(item["thresholds"]["hydro_volume_ml"]),
                    energy_threshold_j=float(item["thresholds"]["energy_dose_j"]),
                    nearest_structure=item.get("nearest_structure"),
                    clearance_m=float(item.get("clearance_m", math.inf)),
                )
                for item in topology["adhesion_bridges"]
            }

        def engage(self, *, stage=None):
            stage = _current_stage(stage)
            superficial = f"{self.tissue_root.rstrip('/')}/Anatomy/SuperficialFlap"
            target = f"{self.tissue_root.rstrip('/')}/Anatomy/TargetBed"
            scope = f"{self.tissue_root.rstrip('/')}/RuntimeBridgeAttachments"
            stage.DefinePrim(scope, "Scope")
            created = []
            try:
                for index, state in self.states.items():
                    base = f"{self.tissue_root.rstrip('/')}/AdhesionBridges/Bridge_{index:02d}"
                    paths = []
                    for label, actor, anchor in (
                        ("upper", superficial, f"{base}/UpperAnchor"),
                        ("lower", target, f"{base}/LowerAnchor"),
                    ):
                        attachment = f"{scope}/bridge_{index:02d}_{label}"
                        create_deformable_attachment(actor, anchor, attachment, stage=stage)
                        paths.append(attachment)
                        created.append(attachment)
                    state.attachment_paths = paths
            except Exception:
                remove_prims(created, stage=stage)
                raise
            return self.snapshot()

        def release(self, index: int, mode: str, *, stage=None):
            state = self.states[int(index)]
            if state.released:
                return False
            stage = _current_stage(stage)
            joint_path = f"{self.tissue_root.rstrip('/')}/AdhesionBridges/Bridge_{index:02d}/ContinuityJoint"
            if stage.GetPrimAtPath(joint_path).IsValid():
                stage.RemovePrim(joint_path)
            state.released = True
            state.release_mode = str(mode)
            return True

        @staticmethod
        def _weight(point: Sequence[float], center: Sequence[float], radius_m: float) -> float:
            distance = math.dist(tuple(float(v) for v in point), tuple(float(v) for v in center))
            return max(0.0, 1.0 - distance / max(float(radius_m), 1.0e-9))

        def apply_blunt_work(self, local_position: Sequence[float], work_j: float, *, radius_m: float = 0.018, stage=None):
            released = []
            for state in self.states.values():
                if state.released:
                    continue
                weight = self._weight(local_position, state.position, radius_m)
                if weight <= 0:
                    continue
                class_scale = 1.0 if state.bridge_class == "loose_connective_fibre" else 0.55 if state.bridge_class == "vascularized_adhesion" else 0.28
                state.mechanical_work_j += max(0.0, float(work_j)) * weight * class_scale
                hydration_scale = max(0.28, 1.0 - 0.72 * state.hydro_volume_ml / max(state.hydro_threshold_ml, 1.0e-9))
                if state.mechanical_work_j >= state.mechanical_threshold_j * hydration_scale:
                    if self.release(state.index, "blunt_spreading", stage=stage):
                        released.append(state.index)
            return released

        def apply_hydro_volume(self, local_position: Sequence[float], volume_ml: float, *, radius_m: float = 0.024, stage=None):
            released = []
            for state in self.states.values():
                if state.released:
                    continue
                weight = self._weight(local_position, state.position, radius_m)
                if weight <= 0:
                    continue
                class_scale = 1.0 if state.bridge_class == "loose_connective_fibre" else 0.62 if state.bridge_class == "vascularized_adhesion" else 0.22
                state.hydro_volume_ml += max(0.0, float(volume_ml)) * weight * class_scale
                if state.hydro_volume_ml >= state.hydro_threshold_ml and state.bridge_class != "dense_fibrous_band":
                    if self.release(state.index, "hydrodissection", stage=stage):
                        released.append(state.index)
            return released

        def apply_energy(self, local_position: Sequence[float], energy_j: float, *, radius_m: float = 0.012, stage=None):
            released = []
            for state in self.states.values():
                if state.released:
                    continue
                weight = self._weight(local_position, state.position, radius_m)
                if weight <= 0:
                    continue
                state.energy_dose_j += max(0.0, float(energy_j)) * weight
                if state.energy_dose_j >= state.energy_threshold_j:
                    if self.release(state.index, "low_energy_dissection", stage=stage):
                        released.append(state.index)
            return released

        def nearest_unreleased(self, local_position: Sequence[float], radius_m: float = 0.009):
            candidates = [
                (math.dist(tuple(local_position), state.position), state)
                for state in self.states.values()
                if not state.released
            ]
            if not candidates:
                return None
            distance, state = min(candidates, key=lambda item: item[0])
            return state if distance <= radius_m else None

        def cut_nearest(self, local_position: Sequence[float], *, guard_retracted: bool, blade_closed: bool, stage=None):
            if not guard_retracted:
                return {"released": False, "reason": "scissor_guard_not_retracted"}
            if not blade_closed:
                return {"released": False, "reason": "scissor_blade_not_closed"}
            state = self.nearest_unreleased(local_position)
            if state is None:
                return {"released": False, "reason": "no_bridge_in_cut_volume"}
            released = self.release(state.index, "guarded_scissors", stage=stage)
            return {"released": released, "bridge_index": state.index, "bridge_class": state.bridge_class}

        @property
        def release_fraction(self):
            if not self.states:
                return 0.0
            return sum(state.released for state in self.states.values()) / len(self.states)

        def snapshot(self):
            return {
                "release_fraction": self.release_fraction,
                "released_count": sum(state.released for state in self.states.values()),
                "total_count": len(self.states),
                "bridges": {
                    index: {
                        "class": state.bridge_class,
                        "released": state.released,
                        "release_mode": state.release_mode,
                        "mechanical_work_j": state.mechanical_work_j,
                        "hydro_volume_ml": state.hydro_volume_ml,
                        "energy_dose_j": state.energy_dose_j,
                    }
                    for index, state in self.states.items()
                },
            }


    def _distance_point_segment(point, a, b) -> float:
        p = tuple(float(v) for v in point)
        a = tuple(float(v) for v in a)
        b = tuple(float(v) for v in b)
        ab = tuple(b[i] - a[i] for i in range(3))
        ap = tuple(p[i] - a[i] for i in range(3))
        denom = sum(value * value for value in ab)
        amount = 0.0 if denom <= 1.0e-18 else max(0.0, min(1.0, sum(ap[i] * ab[i] for i in range(3)) / denom))
        closest = tuple(a[i] + amount * ab[i] for i in range(3))
        return math.dist(p, closest)


    @dataclass
    class ProtectedStructureState:
        name: str
        intact: bool = True
        injury_mechanism: str | None = None
        blood_loss_ml: float = 0.0
        duct_leak_ml: float = 0.0
        nerve_conduction_fraction: float = 1.0


    @dataclass
    class ProtectedStructureController:
        tissue_root: str
        states: dict[str, ProtectedStructureState] = field(
            default_factory=lambda: {name: ProtectedStructureState(name) for name in PROTECTED_STRUCTURES}
        )
        attachments: list[str] = field(default_factory=list)

        def topology(self):
            return load_dissection_topology()["protected_structures"]

        def attach_to_target_bed(self, *, stage=None):
            stage = _current_stage(stage)
            target = f"{self.tissue_root.rstrip('/')}/Anatomy/TargetBed"
            scope = f"{self.tissue_root.rstrip('/')}/RuntimeProtectedStructureAttachments"
            stage.DefinePrim(scope, "Scope")
            created = []
            try:
                for name in PROTECTED_STRUCTURES:
                    root = f"{self.tissue_root.rstrip('/')}/ProtectedStructures/{name.title()}"
                    for segment in ("ProximalSegment", "DistalSegment"):
                        attachment = f"{scope}/{name}_{segment.lower()}"
                        create_deformable_attachment(target, f"{root}/Links/{segment}", attachment, stage=stage)
                        created.append(attachment)
            except Exception:
                remove_prims(created, stage=stage)
                raise
            self.attachments = created
            return list(created)

        def distance_to(self, local_position: Sequence[float], structure: str) -> float:
            points = self.topology()[structure]["centerline_m"]
            return min(_distance_point_segment(local_position, a, b) for a, b in zip(points[:-1], points[1:]))

        def nearest(self, local_position: Sequence[float]):
            values = {name: self.distance_to(local_position, name) for name in PROTECTED_STRUCTURES}
            name = min(values, key=values.get)
            return name, values[name], values

        def evaluate_action(self, local_position: Sequence[float], modality: str):
            minimum = {
                "blunt": 0.0025,
                "hydro": 0.0030,
                "scissors": 0.0050,
                "energy": 0.0070,
            }[modality]
            name, distance, all_distances = self.nearest(local_position)
            reasons = []
            if distance < minimum:
                reasons.append(f"{name}_clearance_below_{minimum:.4f}_m")
            if not self.states[name].intact:
                reasons.append(f"{name}_already_injured")
            return {
                "authorized": not reasons,
                "nearest_structure": name,
                "distance_m": distance,
                "minimum_clearance_m": minimum,
                "all_distances_m": all_distances,
                "reasons": reasons,
            }

        def injure(self, structure: str, mechanism: str, *, stage=None):
            if structure not in self.states:
                raise KeyError(structure)
            state = self.states[structure]
            if not state.intact:
                return False
            stage = _current_stage(stage)
            root = f"{self.tissue_root.rstrip('/')}/ProtectedStructures/{structure.title()}"
            joint = f"{root}/Joints/ContinuityJoint"
            if stage.GetPrimAtPath(joint).IsValid():
                stage.RemovePrim(joint)
            prim = stage.GetPrimAtPath(root)
            if prim and prim.IsValid():
                variants = prim.GetVariantSets().GetVariantSet("integrity")
                if variants:
                    variants.SetVariantSelection("injured")
            state.intact = False
            state.injury_mechanism = str(mechanism)
            if structure == "nerve":
                state.nerve_conduction_fraction = 0.0
            return True

        def update_complication(self, dt: float, *, pressure_pa: float = 12000.0, duct_pressure_pa: float = 900.0):
            dt = max(0.0, float(dt))
            vessel = self.states["vessel"]
            duct = self.states["duct"]
            if not vessel.intact:
                vessel.blood_loss_ml += 0.45 * math.sqrt(max(pressure_pa, 0.0) / 12000.0) * dt
            if not duct.intact:
                duct.duct_leak_ml += 0.08 * math.sqrt(max(duct_pressure_pa, 0.0) / 900.0) * dt
            return self.snapshot()

        def snapshot(self):
            return {
                name: {
                    "intact": state.intact,
                    "injury_mechanism": state.injury_mechanism,
                    "blood_loss_ml": state.blood_loss_ml,
                    "duct_leak_ml": state.duct_leak_ml,
                    "nerve_conduction_fraction": state.nerve_conduction_fraction,
                }
                for name, state in self.states.items()
            }


    @dataclass
    class FluidLedger:
        reservoir_capacity_ml: float = 35.0
        reservoir_ml: float = 35.0
        collection_capacity_ml: float = 55.0
        emitted_ml: float = 0.0
        aspirated_ml: float = 0.0
        absorbed_ml: float = 0.0
        spilled_ml: float = 0.0
        active_particle_ml: float = 0.0

        def emit(self, requested_ml: float) -> float:
            amount = min(max(float(requested_ml), 0.0), self.reservoir_ml)
            self.reservoir_ml -= amount
            self.emitted_ml += amount
            self.active_particle_ml += amount
            return amount

        def aspirate(self, requested_ml: float) -> float:
            capacity = max(0.0, self.collection_capacity_ml - self.aspirated_ml)
            amount = min(max(float(requested_ml), 0.0), self.active_particle_ml, capacity)
            self.active_particle_ml -= amount
            self.aspirated_ml += amount
            return amount

        def absorb(self, requested_ml: float) -> float:
            amount = min(max(float(requested_ml), 0.0), self.active_particle_ml)
            self.active_particle_ml -= amount
            self.absorbed_ml += amount
            return amount

        @property
        def balance_error_ml(self):
            return self.reservoir_capacity_ml - (
                self.reservoir_ml + self.active_particle_ml + self.aspirated_ml + self.absorbed_ml + self.spilled_ml
            )

        def snapshot(self):
            return {
                "reservoir_ml": self.reservoir_ml,
                "emitted_ml": self.emitted_ml,
                "active_particle_ml": self.active_particle_ml,
                "aspirated_ml": self.aspirated_ml,
                "absorbed_ml": self.absorbed_ml,
                "spilled_ml": self.spilled_ml,
                "balance_error_ml": self.balance_error_ml,
            }


    def ensure_dissection_particle_system(
        *, stage=None, physics_scene_path="/World/physicsScene",
        root_path="/World/DrAnmarDissectionFluid", particle_radius_m=PARTICLE_RADIUS_M,
    ):
        stage = _current_stage(stage)
        from omni.physx.scripts import particleUtils, physicsUtils
        from pxr import Sdf, UsdGeom, UsdPhysics

        stage.DefinePrim(root_path, "Scope")
        if not stage.GetPrimAtPath(physics_scene_path).IsValid():
            UsdPhysics.Scene.Define(stage, physics_scene_path)
        system_path = f"{root_path}/ParticleSystem"
        set_path = f"{root_path}/Particles"
        material_path = f"{root_path}/PBDMaterial"
        if not stage.GetPrimAtPath(material_path).IsValid():
            particleUtils.add_pbd_particle_material(
                stage, Sdf.Path(material_path), cohesion=0.0015, viscosity=0.0015,
                surface_tension=0.0035, friction=0.04,
            )
        if not stage.GetPrimAtPath(system_path).IsValid():
            particleUtils.add_physx_particle_system(
                stage=stage, particle_system_path=Sdf.Path(system_path),
                simulation_owner=Sdf.Path(physics_scene_path),
                particle_contact_offset=particle_radius_m * 1.15,
                rest_offset=particle_radius_m * 0.90,
                solid_rest_offset=particle_radius_m * 1.80,
                fluid_rest_offset=particle_radius_m * 0.92,
            )
            physicsUtils.add_physics_material_to_prim(
                stage, stage.GetPrimAtPath(system_path), Sdf.Path(material_path)
            )
        if not stage.GetPrimAtPath(set_path).IsValid():
            particleUtils.add_physx_particleset_points(
                stage, Sdf.Path(set_path), [], [], [], Sdf.Path(system_path),
                True, True, 0, 1.0, particle_radius_m * 2.0,
            )
            UsdGeom.Points(stage.GetPrimAtPath(set_path)).GetWidthsAttr().Set([])
        return {"root_path": root_path, "particle_system_path": system_path, "particle_set_path": set_path}


    def emit_hydro_burst(
        tool_path: str, ledger: FluidLedger, *, requested_ml=0.22, jet_speed_m_s=1.35,
        stage=None, particle_set_path="/World/DrAnmarDissectionFluid/Particles",
    ):
        stage = _current_stage(stage)
        from pxr import Gf, UsdGeom, Vt
        points = UsdGeom.Points(stage.GetPrimAtPath(particle_set_path))
        if not points:
            raise ValueError(f"No particle set at {particle_set_path}")
        available = ledger.emit(requested_ml)
        count = int(available / PARTICLE_VOLUME_ML)
        count -= count % 7
        actual_ml = count * PARTICLE_VOLUME_ML
        remainder = available - actual_ml
        ledger.reservoir_ml += remainder
        ledger.emitted_ml -= remainder
        ledger.active_particle_ml -= remainder
        if count <= 0:
            return {"particle_count": 0, "emitted_ml": 0.0}
        transform = _world_transform(stage, frame_path(tool_path, "hydro_nozzle_tip"))
        current_positions = list(points.GetPointsAttr().Get() or [])
        current_velocities = list(points.GetVelocitiesAttr().Get() or [])
        current_widths = list(points.GetWidthsAttr().Get() or [])
        per_jet = count // 7
        for jet in range(7):
            angle = 2.0 * math.pi * jet / 7.0
            origin_local = Gf.Vec3d(0.0012 * math.cos(angle), 0.0012 * math.sin(angle), 0.0)
            direction_local = Gf.Vec3d(0.10 * math.cos(angle), 0.10 * math.sin(angle), 1.0).GetNormalized()
            origin_world = transform.Transform(origin_local)
            direction_world = transform.TransformDir(direction_local).GetNormalized()
            for index in range(per_jet):
                jitter = (index % 5 - 2) * 0.00010
                position = origin_world + direction_world * jitter
                current_positions.append(Gf.Vec3f(position))
                current_velocities.append(Gf.Vec3f(direction_world * jet_speed_m_s))
                current_widths.append(PARTICLE_RADIUS_M * 2.0)
        points.GetPointsAttr().Set(Vt.Vec3fArray(current_positions))
        points.GetVelocitiesAttr().Set(Vt.Vec3fArray(current_velocities))
        points.GetWidthsAttr().Set(current_widths)
        return {"particle_count": count, "emitted_ml": actual_ml, "particle_set_path": particle_set_path}


    @dataclass
    class SuctionFieldController:
        capture_radius_m: float = 0.031
        capture_depth_m: float = 0.040
        throat_radius_m: float = 0.007
        max_acceleration_m_s2: float = 20.0

        def update_particles(
            self, tool_path: str, ledger: FluidLedger, *, dt: float, opening=1.0,
            stage=None, particle_set_path="/World/DrAnmarDissectionFluid/Particles",
        ):
            stage = _current_stage(stage)
            from pxr import Gf, UsdGeom, Vt
            points = UsdGeom.Points(stage.GetPrimAtPath(particle_set_path))
            positions = list(points.GetPointsAttr().Get() or [])
            velocities = list(points.GetVelocitiesAttr().Get() or [])
            widths = list(points.GetWidthsAttr().Get() or [PARTICLE_RADIUS_M * 2.0] * len(positions))
            if not positions:
                return {"active": 0, "captured": 0, "aspirated_ml": 0.0}
            opening = max(0.0, min(1.0, float(opening)))
            capture_transform = _world_transform(stage, frame_path(tool_path, "suction_center"))
            inverse = capture_transform.GetInverse()
            throat = capture_transform.ExtractTranslation()
            kept_positions, kept_velocities, kept_widths = [], [], []
            captured = 0
            for position, velocity, width in zip(positions, velocities, widths):
                world = Gf.Vec3d(position)
                local = inverse.Transform(world)
                radial = math.hypot(local[0], local[1])
                to_throat = throat - world
                distance = max(float(to_throat.GetLength()), 1.0e-8)
                if opening > 0 and distance <= self.throat_radius_m:
                    captured += 1
                    continue
                new_velocity = Gf.Vec3d(velocity)
                if radial <= self.capture_radius_m and abs(local[2]) <= self.capture_depth_m and opening > 0:
                    new_velocity += to_throat / distance * (opening * self.max_acceleration_m_s2 * max(0.0, dt))
                kept_positions.append(Gf.Vec3f(world))
                kept_velocities.append(Gf.Vec3f(new_velocity))
                kept_widths.append(float(width))
            points.GetPointsAttr().Set(Vt.Vec3fArray(kept_positions))
            points.GetVelocitiesAttr().Set(Vt.Vec3fArray(kept_velocities))
            points.GetWidthsAttr().Set(kept_widths)
            aspirated = ledger.aspirate(captured * PARTICLE_VOLUME_ML)
            return {"active": len(kept_positions), "captured": captured, "aspirated_ml": aspirated}


    @dataclass
    class EnergyDissectionState:
        temperature_c: float = 37.0
        delivered_energy_j: float = 0.0
        smoke_generated_ml: float = 0.0
        overtemperature: bool = False


    @dataclass
    class LowEnergyDissectionController:
        target_temperature_c: float = 72.0
        maximum_temperature_c: float = 95.0
        maximum_power_w: float = 22.0
        heat_capacity_j_k: float = 1.1
        heat_loss_w_k: float = 0.18
        state: EnergyDissectionState = field(default_factory=EnergyDissectionState)

        def update(self, dt: float, contact_force_n: float, requested_power_w: float | None = None):
            dt = max(0.0, float(dt))
            force_scale = max(0.0, min(1.0, float(contact_force_n) / 1.5))
            if requested_power_w is None:
                requested_power_w = max(0.0, min(self.maximum_power_w, (self.target_temperature_c - self.state.temperature_c) * 0.8))
            power = 0.0 if self.state.overtemperature else max(0.0, min(self.maximum_power_w, float(requested_power_w)))
            absorbed = power * (0.30 + 0.70 * force_scale)
            loss = self.heat_loss_w_k * max(0.0, self.state.temperature_c - 37.0)
            self.state.temperature_c += (absorbed - loss) * dt / max(self.heat_capacity_j_k, 1.0e-9)
            energy = absorbed * dt
            self.state.delivered_energy_j += energy
            self.state.smoke_generated_ml += max(0.0, self.state.temperature_c - 58.0) * energy * 0.0004
            self.state.overtemperature = self.state.temperature_c > self.maximum_temperature_c
            return {"energy_j": energy, "state": self.state}


    @dataclass
    class ScissorsInterlockController:
        minimum_guard_retraction_m: float = 0.009
        minimum_structure_clearance_m: float = 0.005
        violations: int = 0

        def evaluate(self, local_position: Sequence[float], guard_retraction_m: float, protected: ProtectedStructureController):
            safety = protected.evaluate_action(local_position, "scissors")
            reasons = list(safety["reasons"])
            if float(guard_retraction_m) < self.minimum_guard_retraction_m:
                reasons.append("guard_not_fully_retracted")
            return {**safety, "authorized": not reasons, "reasons": reasons}

        def request_cut(
            self, local_position: Sequence[float], guard_retraction_m: float,
            bridges: AdhesionBridgeController, protected: ProtectedStructureController,
            *, override=False, stage=None,
        ):
            result = self.evaluate(local_position, guard_retraction_m, protected)
            if not result["authorized"] and not override:
                self.violations += 1
                return {**result, "released": False}
            if not result["authorized"] and override:
                protected.injure(result["nearest_structure"], "scissor_override", stage=stage)
            cut = bridges.cut_nearest(local_position, guard_retracted=True, blade_closed=True, stage=stage)
            return {**result, **cut, "override": bool(override)}


    @dataclass
    class DissectionCompletionVerifier:
        bridges: AdhesionBridgeController
        protected: ProtectedStructureController

        def evaluate(self, *, visibility_fraction: float, traction_stable: bool):
            bridge_snapshot = self.bridges.snapshot()
            structure_snapshot = self.protected.snapshot()
            protected_intact = all(value["intact"] for value in structure_snapshot.values())
            residual = bridge_snapshot["total_count"] - bridge_snapshot["released_count"]
            complete = (
                residual == 0
                and protected_intact
                and float(visibility_fraction) >= 0.90
                and bool(traction_stable)
            )
            return {
                "complete": complete,
                "released_bridge_fraction": bridge_snapshot["release_fraction"],
                "residual_bridge_count": residual,
                "protected_structures_intact": protected_intact,
                "visibility_fraction": float(visibility_fraction),
                "traction_stable": bool(traction_stable),
                "complications": [name for name, value in structure_snapshot.items() if not value["intact"]],
            }


    PHASE_TARGETS = {
        "inspect": {name: 0.0 for name in TOOL_JOINTS.values()},
        "capture": {
            **{name: 0.0 for name in TOOL_JOINTS.values()},
            "left_traction_joint": -0.006, "right_traction_joint": 0.006,
            "left_pad_pitch_joint": math.radians(-8), "right_pad_pitch_joint": math.radians(8),
            "left_pad_compliance_joint": 0.003, "right_pad_compliance_joint": 0.003,
            "suction_valve_joint": 0.002,
        },
        "traction": {
            **{name: 0.0 for name in TOOL_JOINTS.values()},
            "left_traction_joint": -0.026, "right_traction_joint": 0.026,
            "left_pad_pitch_joint": math.radians(-14), "right_pad_pitch_joint": math.radians(14),
            "left_pad_compliance_joint": 0.004, "right_pad_compliance_joint": 0.004,
            "suction_valve_joint": 0.003,
        },
        "blunt": {
            **{name: 0.0 for name in TOOL_JOINTS.values()},
            "left_traction_joint": -0.028, "right_traction_joint": 0.028,
            "left_pad_pitch_joint": math.radians(-16), "right_pad_pitch_joint": math.radians(16),
            "left_pad_compliance_joint": 0.004, "right_pad_compliance_joint": 0.004,
            "left_spreader_joint": -0.018, "right_spreader_joint": 0.018,
            "suction_valve_joint": 0.004, "irrigation_valve_joint": 0.002,
        },
        "hydro": {
            **{name: 0.0 for name in TOOL_JOINTS.values()},
            "left_traction_joint": -0.030, "right_traction_joint": 0.030,
            "left_pad_pitch_joint": math.radians(-17), "right_pad_pitch_joint": math.radians(17),
            "left_pad_compliance_joint": 0.004, "right_pad_compliance_joint": 0.004,
            "left_spreader_joint": -0.016, "right_spreader_joint": 0.016,
            "hydro_pitch_joint": math.radians(12), "hydro_extension_joint": 0.042,
            "hydro_valve_joint": 0.007, "suction_valve_joint": 0.005, "irrigation_valve_joint": 0.002,
        },
        "scissors": {
            **{name: 0.0 for name in TOOL_JOINTS.values()},
            "left_traction_joint": -0.031, "right_traction_joint": 0.031,
            "left_pad_pitch_joint": math.radians(-18), "right_pad_pitch_joint": math.radians(18),
            "left_pad_compliance_joint": 0.004, "right_pad_compliance_joint": 0.004,
            "left_spreader_joint": -0.014, "right_spreader_joint": 0.014,
            "scissor_extension_joint": 0.046, "scissor_guard_joint": -0.010,
            "scissor_blade_joint": math.radians(30), "suction_valve_joint": 0.006,
        },
        "energy": {
            **{name: 0.0 for name in TOOL_JOINTS.values()},
            "left_traction_joint": -0.031, "right_traction_joint": 0.031,
            "left_pad_pitch_joint": math.radians(-18), "right_pad_pitch_joint": math.radians(18),
            "left_pad_compliance_joint": 0.004, "right_pad_compliance_joint": 0.004,
            "energy_tip_extension_joint": 0.042, "suction_valve_joint": 0.006,
        },
        "verify": {
            **{name: 0.0 for name in TOOL_JOINTS.values()},
            "left_traction_joint": -0.032, "right_traction_joint": 0.032,
            "left_pad_pitch_joint": math.radians(-18), "right_pad_pitch_joint": math.radians(18),
            "left_pad_compliance_joint": 0.003, "right_pad_compliance_joint": 0.003,
            "suction_valve_joint": 0.003,
        },
        "complete": {name: 0.0 for name in TOOL_JOINTS.values()},
        "abort": {
            **{name: 0.0 for name in TOOL_JOINTS.values()},
            "suction_valve_joint": 0.008, "irrigation_valve_joint": 0.005,
        },
    }


    def phase_targets(phase: str):
        try:
            return dict(PHASE_TARGETS[phase])
        except KeyError as exc:
            raise KeyError(f"Unknown SafePlane phase {phase!r}") from exc


    @dataclass
    class SafePlaneDissectionSequenceController:
        tissue_root: str
        tool_path: str
        phase: str = "inspect"
        history: list[str] = field(default_factory=list)
        traction: BilateralTractionController = field(init=False)
        bridges: AdhesionBridgeController = field(init=False)
        protected: ProtectedStructureController = field(init=False)
        hydro_ledger: FluidLedger = field(default_factory=FluidLedger)
        suction: SuctionFieldController = field(default_factory=SuctionFieldController)
        energy: LowEnergyDissectionController = field(default_factory=LowEnergyDissectionController)
        scissors: ScissorsInterlockController = field(default_factory=ScissorsInterlockController)
        verifier: DissectionCompletionVerifier = field(init=False)

        def __post_init__(self):
            self.traction = BilateralTractionController(self.tool_path, self.tissue_root)
            self.bridges = AdhesionBridgeController(self.tissue_root)
            self.protected = ProtectedStructureController(self.tissue_root)
            self.verifier = DissectionCompletionVerifier(self.bridges, self.protected)

        def transition(self, phase: str):
            targets = phase_targets(phase)
            self.phase = phase
            self.history.append(phase)
            return targets

        def initialize_physical_connections(self, *, stage=None):
            return {
                "traction": self.traction.capture(stage=stage),
                "bridges": self.bridges.engage(stage=stage),
                "protected_structures": self.protected.attach_to_target_bed(stage=stage),
            }

        def blunt_action(self, local_position: Sequence[float], work_j: float, *, override=False, stage=None):
            safety = self.protected.evaluate_action(local_position, "blunt")
            if not safety["authorized"] and not override:
                return {"safety": safety, "released_bridges": [], "blocked": True}
            if not safety["authorized"] and override:
                self.protected.injure(safety["nearest_structure"], "blunt_override", stage=stage)
            released = self.bridges.apply_blunt_work(local_position, work_j, stage=stage)
            return {"safety": safety, "released_bridges": released, "override": bool(override)}

        def hydro_action(self, local_position: Sequence[float], volume_ml: float, *, override=False, stage=None):
            safety = self.protected.evaluate_action(local_position, "hydro")
            if not safety["authorized"] and not override:
                return {"safety": safety, "released_bridges": [], "blocked": True}
            if not safety["authorized"] and override:
                self.protected.injure(safety["nearest_structure"], "hydro_override", stage=stage)
            released = self.bridges.apply_hydro_volume(local_position, volume_ml, stage=stage)
            return {"safety": safety, "released_bridges": released, "override": bool(override)}

        def energy_action(self, local_position: Sequence[float], *, dt: float, contact_force_n: float, requested_power_w: float | None = None, override=False, stage=None):
            safety = self.protected.evaluate_action(local_position, "energy")
            if not safety["authorized"] and not override:
                return {"safety": safety, "released_bridges": [], "blocked": True}
            if not safety["authorized"] and override:
                self.protected.injure(safety["nearest_structure"], "energy_override", stage=stage)
            energy_result = self.energy.update(dt, contact_force_n, requested_power_w)
            released = self.bridges.apply_energy(local_position, energy_result["energy_j"], stage=stage)
            return {"safety": safety, "released_bridges": released, "energy": energy_result, "override": bool(override)}

        def scissors_action(self, local_position: Sequence[float], guard_retraction_m: float, *, override=False, stage=None):
            return self.scissors.request_cut(
                local_position, guard_retraction_m, self.bridges, self.protected,
                override=override, stage=stage,
            )

        def verify(self, *, visibility_fraction: float, traction_stable: bool):
            return self.verifier.evaluate(visibility_fraction=visibility_fraction, traction_stable=traction_stable)
    ''').lstrip()


def interaction_frames(bundle: ToolBundle) -> dict[str, object]:
    return {
        "schema": "dranmar.interaction-frames.v1",
        "asset": "dranmar-safeplane-dissection-robot-v1",
        "frames": bundle.frames,
        "coordinate_contract": {
            "units": "metres",
            "quaternion_order": "wxyz",
            "tool_forward_axis": "+Z from wrist toward tissue",
            "safe_plane_normal": "+Z in tool-local frame",
        },
    }


def mount_contract() -> dict[str, object]:
    return {
        "schema": "dranmar.franka-hand-replacement.v1",
        "asset": "dranmar-safeplane-dissection-robot-v1",
        "parent_link": "panda_link8",
        "payload_link": "DrAnmarSafePlaneDissectionTool/Links/Mount",
        "local_translation_m": [0.0, 0.0, 0.0],
        "local_rotation": {"axis": [0.0, 0.0, 1.0], "degrees": FRANKA_HAND_EQUIVALENT_ROTATION_DEG},
        "composable_asset_compatibility": "reconstruct_fixed_panda_link8_when_nvidia_asset_collapses_terminal_link_into_panda_link7_to_hand_joint",
        "deactivated_stock_prims": [
            "panda_hand_joint", "panda_hand", "panda_finger_joint1", "panda_finger_joint2",
            "panda_leftfinger", "panda_rightfinger",
        ],
        "nested_articulation_in_payload": False,
        "standalone_articulation_available": True,
    }


def dissection_topology(bundle: ToolBundle) -> dict[str, object]:
    vessel = [[-0.075, -0.028, 0.020], [-0.035, -0.020, 0.022], [0.0, -0.014, 0.021], [0.038, -0.006, 0.019], [0.078, 0.006, 0.018]]
    nerve = [[-0.075, 0.022, 0.019], [-0.035, 0.016, 0.021], [0.0, 0.020, 0.020], [0.040, 0.028, 0.019], [0.078, 0.034, 0.018]]
    duct = [[-0.068, 0.048, 0.018], [-0.030, 0.041, 0.020], [0.006, 0.036, 0.019], [0.043, 0.043, 0.018], [0.072, 0.052, 0.017]]
    return {
        "schema": "dranmar.safeplane-dissection-topology.v1",
        "asset": "dranmar-safeplane-tissue-demo-v1",
        "units": "metres",
        "safe_plane": {
            "reference_z_m": 0.025,
            "superficial_surface_path": "Anatomy/SuperficialFlap",
            "target_bed_path": "Anatomy/TargetBed",
            "target_bridge_count": len(bundle.bridge_specs),
            "completion_rule": "all_target_bridge_continuity_joints_released_and_all_protected_structures_intact",
        },
        "protected_structures": {
            "vessel": {"centerline_m": vessel, "nominal_radius_m": 0.0031, "safety_clearance_m": {"blunt": 0.0025, "hydro": 0.0030, "scissors": 0.0050, "energy": 0.0070}},
            "nerve": {"centerline_m": nerve, "nominal_radius_m": 0.0021, "safety_clearance_m": {"blunt": 0.0025, "hydro": 0.0030, "scissors": 0.0050, "energy": 0.0070}},
            "duct": {"centerline_m": duct, "nominal_radius_m": 0.0026, "safety_clearance_m": {"blunt": 0.0025, "hydro": 0.0030, "scissors": 0.0050, "energy": 0.0070}},
        },
        "adhesion_bridges": [
            {
                "index": spec.index,
                "position_m": list(spec.position),
                "bridge_class": spec.bridge_class,
                "target": spec.target,
                "recommended_mode": spec.recommended_mode,
                "thresholds": {
                    "mechanical_work_j": spec.mechanical_work_j,
                    "hydro_volume_ml": spec.hydro_volume_ml,
                    "energy_dose_j": spec.energy_dose_j,
                },
                "nearest_structure": spec.nearest_structure,
                "clearance_m": spec.clearance_m,
                "continuity_joint_path": f"AdhesionBridges/Bridge_{spec.index:02d}/ContinuityJoint",
                "upper_anchor_path": f"AdhesionBridges/Bridge_{spec.index:02d}/UpperAnchor",
                "lower_anchor_path": f"AdhesionBridges/Bridge_{spec.index:02d}/LowerAnchor",
            }
            for spec in bundle.bridge_specs
        ],
    }


def task_contract() -> dict[str, object]:
    return {
        "schema": "dranmar.safeplane-dissection-task.v1",
        "asset": "dranmar-safeplane-dissection-robot-v1",
        "procedure": [
            "inspect", "capture", "traction", "blunt", "hydro", "guarded_scissors",
            "low_energy", "irrigate_and_evacuate", "verify_connectivity", "release", "complete",
        ],
        "required_success_conditions": [
            "all_target_adhesion_bridges_released",
            "vessel_continuity_intact",
            "nerve_continuity_intact",
            "duct_continuity_intact",
            "roi_visibility_fraction_at_least_0_90",
            "traction_stable",
            "no_uncommanded_scissor_or_energy_action_inside_protected_clearance",
        ],
        "metrics": [
            "released_bridge_fraction", "residual_bridge_count", "release_mode_distribution",
            "left_traction_force_n", "right_traction_force_n", "traction_force_asymmetry_n",
            "blunt_work_j", "hydro_volume_ml", "energy_j", "scissor_cut_count",
            "minimum_vessel_clearance_m", "minimum_nerve_clearance_m", "minimum_duct_clearance_m",
            "blood_loss_ml", "duct_leak_ml", "nerve_conduction_fraction",
            "roi_visibility_fraction", "procedure_time_s", "safety_interlock_violations",
        ],
        "failure_states": [
            "traction_overload", "pad_capture_loss", "residual_adhesion", "wrong_plane_dissection",
            "vessel_injury", "nerve_injury", "duct_injury", "hydro_overpressure_proxy",
            "energy_overtemperature", "unguarded_scissor_command", "incomplete_smoke_evacuation",
        ],
        "clinical_validation": False,
        "medical_device": False,
    }


def physics_profile(bundle: ToolBundle) -> dict[str, object]:
    return {
        "schema": "dranmar.safeplane-dissection-profile.v1",
        "id": "dranmar-safeplane-dissection-robot-v1",
        "version": VERSION,
        "status": "research_informed_engineering_model_pending_runtime_metrology_and_clinical_validation",
        "units": "metres_kilograms_seconds",
        "tool": {
            "mount": "panda_link8_hand_replacement",
            "approximate_mass_kg": sum(link.mass_kg or 0.0 for link in bundle.links.values()),
            "joint_count": len(bundle.joints),
            "traction_cell_count": 2 * TRACTION_CELL_COUNT_PER_SIDE,
            "modalities": ["blunt_spreading", "hydrodissection", "guarded_scissors", "low_energy_dissection"],
            "travel": {
                "traction_m": TRACTION_TRAVEL_M,
                "spreader_m": SPREADER_TRAVEL_M,
                "hydro_extension_m": HYDRO_EXTENSION_M,
                "scissor_extension_m": SCISSOR_EXTENSION_M,
                "energy_extension_m": ENERGY_EXTENSION_M,
            },
        },
        "tissue_substrate": {
            "superficial_mesh": {"vertices": int(len(bundle.superficial_mesh.vertices)), "triangles": int(len(bundle.superficial_mesh.faces))},
            "target_bed_mesh": {"vertices": int(len(bundle.target_bed_mesh.vertices)), "triangles": int(len(bundle.target_bed_mesh.faces))},
            "adhesion_bridge_count": len(bundle.bridge_specs),
            "bridge_classes": sorted({spec.bridge_class for spec in bundle.bridge_specs}),
            "physics_contract": "two_deformable_surfaces_each_bridge_uses_two_rigid_anchors_and_one_removable_continuity_joint",
            "surface_parameters": {
                "superficial": {"youngs_modulus_pa": 95000.0, "poissons_ratio": 0.36, "surface_thickness_m": 0.0045, "dynamic_friction": 0.48},
                "target_bed": {"youngs_modulus_pa": 145000.0, "poissons_ratio": 0.38, "surface_thickness_m": 0.0055, "dynamic_friction": 0.52},
            },
        },
        "traction": {
            "nominal_force_per_pad_n": 1.4,
            "soft_force_limit_per_pad_n": 3.0,
            "hard_release_limit_per_pad_n": 5.0,
            "qualification": "provisional_unmeasured",
        },
        "hydrodissection": {
            "reservoir_capacity_ml": 35.0,
            "particle_radius_m": 0.00072,
            "nominal_jet_speed_m_s": 1.35,
            "jet_count": HYDRO_JET_COUNT,
            "bridge_dose_model": "distance_weighted_volume_threshold_proxy",
            "qualification": "provisional_unmeasured_not_a_clinical_pressure_or_flow_setting",
        },
        "guarded_scissors": {
            "guard_retraction_m": SCISSOR_GUARD_TRAVEL_M,
            "blade_closure_deg": SCISSOR_CLOSE_DEG,
            "minimum_protected_structure_clearance_m": 0.005,
            "cut_contract": "nearest_unreleased_bridge_continuity_joint_removed_only_after_guard_and_clearance_interlocks_pass",
        },
        "low_energy_dissection": {
            "target_temperature_c": 72.0,
            "maximum_temperature_c": 95.0,
            "maximum_power_w": 22.0,
            "model": "lumped_thermal_dose_proxy",
            "qualification": "not_electrosurgical_tissue_physics",
        },
        "protected_structures": {
            "vessel": {"radius_m": 0.0031, "injury_consequence": "continuity_joint_removed_and_blood_loss_ledger_activated"},
            "nerve": {"radius_m": 0.0021, "injury_consequence": "continuity_joint_removed_and_conduction_fraction_set_to_zero"},
            "duct": {"radius_m": 0.0026, "injury_consequence": "continuity_joint_removed_and_duct_leak_ledger_activated"},
        },
        "runtime_platforms": ["OpenUSD", "NVIDIA PhysX", "NVIDIA Isaac Lab"],
        "runtime_observation": "archived_smoke_record_not_physical_qualification",
        "clinical_validation": False,
    }


def collider_coverage(bundle: ToolBundle) -> dict[str, object]:
    rows = []
    for link in bundle.links.values():
        if not link.visuals:
            continue
        visual_min, visual_max = mesh_bounds([visual.mesh for visual in link.visuals])
        collider_meshes = [collider_mesh(collider) for collider in link.colliders if collider.author_enabled]
        if collider_meshes:
            collider_min, collider_max = mesh_bounds(collider_meshes)
            visual_size = np.maximum(visual_max - visual_min, 1.0e-9)
            coverage = np.maximum(0.0, collider_max - collider_min) / visual_size
            rows.append({
                "link": link.name,
                "visual_bounds_m": [visual_min.tolist(), visual_max.tolist()],
                "collider_bounds_m": [collider_min.tolist(), collider_max.tolist()],
                "axis_coverage_ratio": coverage.tolist(),
                "deliberate_insets": "sharp_blade_edges_and_sensor_optics_use_inset_colliders_to_avoid_ghost_contact",
            })
    return {"schema": "dranmar.collider-coverage.v1", "asset": "dranmar-safeplane-dissection-robot-v1", "links": rows}


def asset_manifest() -> dict[str, object]:
    return {
        "schema": "dranmar.asset-manifest.v1",
        "asset": "dranmar-safeplane-dissection-robot-v1",
        "version": VERSION,
        "catalog_subpath": CATALOG_SUBPATH.as_posix(),
        "primary_assets": [
            "dranmar_safeplane_dissection_tool_payload.usda",
            "dranmar_safeplane_dissection_tool_standalone.usda",
            "dranmar_safeplane_dissection_tool_rigid_proxy.usda",
            "dranmar_safeplane_tissue_demo.usda",
            "dranmar_adhesion_bridge.usda",
            "dranmar_protected_vessel_branch.usda",
            "dranmar_protected_nerve_branch.usda",
            "dranmar_protected_duct_branch.usda",
            "dranmar_micro_scissors_cartridge.usda",
            "dranmar_hydrodissection_particle.usda",
            "dranmar_dissection_smoke_particle.usda",
            "dranmar_dissection_blood_particle.usda",
            "dranmar_duct_leak_particle.usda",
        ],
        "authorship": "independently_generated_procedural_geometry_and_runtime_contracts",
        "external_geometry_dependencies": [],
        "license": "Apache-2.0",
        "clinical_validation": False,
    }


def example_scene() -> str:
    return textwrap.dedent('''\
    """DrAnmar SafePlane Dissection Robot scene skeleton.

    Run through the Isaac Lab launcher on CUDA. The example spawns the combined
    Franka robot and layered dissection substrate. Runtime physical attachments,
    particle systems, and controller actions remain explicit host-task steps.
    """
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from orbit.surgical.assets.safeplane_dissection_robot import (
        make_franka_safeplane_dissection_robot_cfg,
        spawn_tissue_demo,
    )

    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2600.0))
        robot = make_franka_safeplane_dissection_robot_cfg(prim_path="{ENV_REGEX_NS}/Robot")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=1 / 240))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    spawn_tissue_demo("/World/DrAnmarSafePlaneTissue", translation=(0.55, 0.0, 0.02))
    sim.reset()
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    simulation_app.close()
    ''')


def docs_mechanism() -> str:
    return f'''# DrAnmar SafePlane Dissection Mechanism

The payload replaces the Panda hand at `panda_link8` and provides {len(build_tool().joints)} controlled axes. Two distributed traction pads maintain counter-traction while the central field remains accessible to four dissection modalities:

- bilateral blunt spreaders;
- a seven-port hydrodissection nozzle;
- guarded articulated micro-scissors;
- a retractable low-energy spatula probe.

Annular suction and irrigation operate around the same registered field. Stereo RGB, depth, fluorescence, and thermal frames are fixed to the housing so modality changes do not require scene re-registration.

The mechanism is owned by DrAnmar and provider-neutral. NVIDIA Isaac Sim and
Isaac Lab provide the target runtime. It is research-only.
'''


def docs_physical() -> str:
    return f'''# Physical Safe-Plane Dissection Contract

## Tissue connectivity

The demonstration substrate contains two independent triangular tissue surfaces connected by {ADHESION_BRIDGE_COUNT} discrete adhesion bridges. Each bridge consists of an upper rigid anchor attached to the superficial flap, a lower rigid anchor attached to the target bed, and one removable fixed continuity joint between the anchors. Each joint meets at the authored mid-plane rather than constraining the separated anchor origins together.

The target bed is attached to two explicit kinematic fixtures. Surface
self-collision is disabled by default for portable GPU deformable cooking and
can be enabled explicitly only on a qualified solver configuration.

Releasing a bridge removes its continuity joint. The anchor halves remain attached to their respective tissue layers, so separation is produced by the physics solver and traction mechanism rather than by directly rewriting the tissue transform.

## Four release mechanisms

- blunt spreading accumulates mechanical contact work;
- hydrodissection accumulates local delivered fluid volume and weakens mechanical thresholds;
- guarded scissors release the bridge nearest the cut volume after guard and safety interlocks pass;
- low-energy dissection accumulates a provisional local thermal-energy dose.

All bridge thresholds are category-level engineering seeds and are not biomechanical or clinical claims.
'''


def docs_safety() -> str:
    return '''# Protected-Structure Safety Contract

The tissue substrate includes independent vessel, nerve, and duct assets. Each structure has two rigid segments joined by a removable continuity joint and attached to the target tissue at runtime.

The controller calculates local tool clearance against each authored centerline. Minimum provisional clearances differ by modality. Scissors and energy actions are blocked when the nearest structure lies inside the configured safety envelope unless the host task explicitly overrides the interlock.

An override has a physical consequence:

- vessel injury removes continuity and activates blood-loss state;
- nerve injury removes continuity and sets conduction to zero;
- duct injury removes continuity and activates a leak state.

This is a research complication model, not a validated injury predictor.
'''


def docs_hydro_energy() -> str:
    return '''# Hydrodissection and Energy Models

The hydrodissection helper creates a PhysX PBD particle system, emits quantized seven-port bursts from the authored nozzle frame, and conserves reservoir, active, aspirated, absorbed, and spilled volume in a ledger. Bridge weakening uses a distance-weighted delivered-volume proxy.

The low-energy probe uses a lumped thermal model with force-dependent absorption, heat loss, temperature, delivered energy, smoke generation, and overtemperature state. It does not reproduce electrosurgical current paths, collagen transformation, steam, charring, or tissue-specific thermal spread.

The annular suction controller accelerates fluid particles toward the authored suction center and transfers captured particle volume into the collection ledger.
'''


def docs_franka() -> str:
    return '''# Franka Integration

`make_franka_safeplane_dissection_robot_cfg()` starts from Isaac Lab's Franka configuration, loads the composable Franka USD, deactivates the Panda hand and finger prims, references the custom payload, and creates a fixed joint from `panda_link8` to `Links/Mount`. When NVIDIA's composable asset collapses the terminal URDF link into the `panda_link7`-to-hand joint, the spawner reconstructs a lightweight fixed `panda_link8` compatibility body before mounting the payload.

The payload preserves the standard Panda-hand mounting rotation of −45 degrees around local Z. Tool joints are appended to the same articulation and grouped into traction, pad-compliance, spreader, hydro, scissors, energy-tip, and valve actuator sets.

The rigid proxy is available for perception, synthetic data, collision-aware planning, and handover tasks.
'''


def docs_validation() -> str:
    return '''# Integrity and runtime boundaries

Static gates cover deterministic assets, dependency closure, controller
invariants, protected-structure interlocks, capacity-safe fluid accounting,
fail-closed attachment overlap, and source/container integrity. The optional
Isaac script is diagnostic only.

Injecting threshold-plus-margin work or fluid at an exact authored bridge
coordinate is a controller unit test, not physical release evidence. Physical
bridge release may be promoted only when the accumulated work or delivered
volume is derived from tool contact and the live simulation state.

No current record qualifies safe-plane identification, tissue selectivity,
traction, hydro pressure, cutting, thermal spread, physical calibration,
clinical performance, or patient use.
'''


def readme() -> str:
    return f'''# {ASSET_NAME}

A DrAnmar-owned, provider-neutral NVIDIA Isaac Sim and Isaac Lab research
system for connectivity-aware tissue dissection with protected anatomy.

## Workflow

`inspect → capture → counter-traction → blunt spread → hydrodissect → selectively cut or apply low energy → evacuate → verify topology → release`

## Primary assets

- `dranmar_safeplane_dissection_tool_payload.usda` — Franka payload without a nested articulation root.
- `dranmar_safeplane_dissection_tool_standalone.usda` — standalone articulated mechanism.
- `dranmar_safeplane_dissection_tool_rigid_proxy.usda` — perception/planning proxy.
- `dranmar_safeplane_tissue_demo.usda` — layered tissue, adhesion network, vessel, nerve, duct, and protected organ surface.
- `dissection_topology.json` — bridge graph, thresholds, structure centerlines, and completion contract.

## Boundary

The package is not clinically validated, is not a medical device, and is not approved for patient care. Tissue, fluid, cutting, energy, force, injury, and safety parameters remain provisional research values.
'''


def installer_source() -> str:
    installer = PACKAGE_ROOT / "scripts/install_into_dranmar.py"
    if not installer.is_file():
        raise FileNotFoundError(f"Canonical installer is missing: {installer}")
    return installer.read_text(encoding="utf-8")

    # Historical embedded source retained below only for archive compatibility.
    return r'''#!/usr/bin/env python3
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def copytree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def update_portfolio(repo: Path) -> None:
    path = repo / "physics_next/dr-anmar-assets.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assets = payload.setdefault("assets", [])
    asset_id = "dranmar-safeplane-dissection-robot-v1"
    entry = {
        "id": asset_id,
        "asset": "source/extensions/orbit.surgical.assets/data/Props/SurgicalDissection/SafePlaneDissectionRobot/dranmar_safeplane_dissection_tool_standalone.usda",
        "payload_asset": "source/extensions/orbit.surgical.assets/data/Props/SurgicalDissection/SafePlaneDissectionRobot/dranmar_safeplane_dissection_tool_payload.usda",
        "auxiliary_asset": "source/extensions/orbit.surgical.assets/data/Props/SurgicalDissection/SafePlaneDissectionRobot/dranmar_safeplane_tissue_demo.usda",
        "profile": "physics_next/surgical-dissection/dranmar-safeplane-dissection-v1.json",
        "interaction_frames": "source/extensions/orbit.surgical.assets/data/Props/SurgicalDissection/SafePlaneDissectionRobot/interaction_frames.json",
        "task_contract": "source/extensions/orbit.surgical.assets/data/Props/SurgicalDissection/SafePlaneDissectionRobot/safeplane_dissection_task_contract.json",
        "report": "source/extensions/orbit.surgical.assets/data/Props/SurgicalDissection/SafePlaneDissectionRobot/asset_manifest.json",
        "live_integration": "franka_hand_replacement_connectivity_aware_multimodal_dissection",
        "live_behavior": "distributed_traction_blunt_spreading_hydrodissection_guarded_scissors_low_energy_dissection_protected_structure_state_and_topology_verification",
        "deployment": "enabled_for_research_iteration",
        "native_gpu_qualification": "pending_isaac_sim_cuda_execution",
        "physical_qualification": "provisional_parameters_pending_instrumented_dissection_bench",
        "clinical_validation": False,
    }
    assets[:] = [item for item in assets if item.get("id") != asset_id]
    assets.append(entry)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: install_into_dranmar.py /path/to/drAnmar")
    repo = Path(sys.argv[1]).expanduser().resolve()
    copytree_contents(PACKAGE / "source/extensions/orbit.surgical.assets", repo / "source/extensions/orbit.surgical.assets")
    copytree_contents(PACKAGE / "physics_next", repo / "physics_next")
    copytree_contents(PACKAGE / "docs", repo / "docs")
    copytree_contents(PACKAGE / "examples", repo / "examples")
    shutil.copy2(PACKAGE / "scripts/generate_dranmar_safeplane_dissection_robot.py", repo / "scripts/generate_dranmar_safeplane_dissection_robot.py")
    init = repo / "source/extensions/orbit.surgical.assets/orbit/surgical/assets/__init__.py"
    line = "from .safeplane_dissection_robot import *\n"
    text = init.read_text(encoding="utf-8") if init.exists() else ""
    if line not in text:
        init.parent.mkdir(parents=True, exist_ok=True)
        init.write_text(text + line, encoding="utf-8")
    update_portfolio(repo)
    print(json.dumps({"installed": True, "repository": str(repo), "catalog_subpath": "Props/SurgicalDissection/SafePlaneDissectionRobot"}, indent=2))


if __name__ == "__main__":
    main()
'''


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_sources(bundle: ToolBundle) -> dict[str, str]:
    vessel_points = [(-0.075, -0.028, 0.020), (-0.035, -0.020, 0.022), (0.0, -0.014, 0.021), (0.038, -0.006, 0.019), (0.078, 0.006, 0.018)]
    nerve_points = [(-0.075, 0.022, 0.019), (-0.035, 0.016, 0.021), (0.0, 0.020, 0.020), (0.040, 0.028, 0.019), (0.078, 0.034, 0.018)]
    duct_points = [(-0.068, 0.048, 0.018), (-0.030, 0.041, 0.020), (0.006, 0.036, 0.019), (0.043, 0.043, 0.018), (0.072, 0.052, 0.017)]
    return {
        "dranmar_protected_vessel_branch.usda": protected_structure_usda(
            root=VESSEL_ROOT, asset_id="dranmar-protected-vessel-branch-v1", kind="vessel",
            left_mesh=bundle.vessel_left, right_mesh=bundle.vessel_right,
            left_points=vessel_points[:3], right_points=vessel_points[2:], radius=0.0031,
            material="VesselMaterial", injured_material="VesselInjured", physics_material="VesselPhysics",
            labels=("protected_vessel", "blood_vessel"),
        ),
        "dranmar_protected_nerve_branch.usda": protected_structure_usda(
            root=NERVE_ROOT, asset_id="dranmar-protected-nerve-branch-v1", kind="nerve",
            left_mesh=bundle.nerve_left, right_mesh=bundle.nerve_right,
            left_points=nerve_points[:3], right_points=nerve_points[2:], radius=0.0021,
            material="NerveMaterial", injured_material="NerveInjured", physics_material="NervePhysics",
            labels=("protected_nerve", "nerve_branch"),
        ),
        "dranmar_protected_duct_branch.usda": protected_structure_usda(
            root=DUCT_ROOT, asset_id="dranmar-protected-duct-branch-v1", kind="duct",
            left_mesh=bundle.duct_left, right_mesh=bundle.duct_right,
            left_points=duct_points[:3], right_points=duct_points[2:], radius=0.0026,
            material="DuctMaterial", injured_material="DuctInjured", physics_material="DuctPhysics",
            labels=("protected_duct", "duct_branch"),
        ),
    }


def write_asset_files(bundle: ToolBundle) -> list[Path]:
    for path in (ASSET_ROOT, GLB_ROOT, TEXTURE_ROOT, PREVIEW_ROOT, DOCS_ROOT, EXAMPLE_ROOT, INTEGRATION_PATH.parent, PHYSICS_PROFILE_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    mapping = {
        "dranmar_safeplane_dissection_tool_payload.usda": tool_usda(bundle, False),
        "dranmar_safeplane_dissection_tool_standalone.usda": tool_usda(bundle, True),
        "dranmar_safeplane_dissection_tool_rigid_proxy.usda": rigid_proxy_usda(bundle),
        "dranmar_safeplane_tissue_demo.usda": tissue_usda(bundle),
        "dranmar_adhesion_bridge.usda": adhesion_bridge_usda(),
        "dranmar_micro_scissors_cartridge.usda": scissors_cartridge_usda(bundle),
        "dranmar_hydrodissection_particle.usda": particle_usda(HYDRO_PARTICLE_ROOT, "HydroParticle", 0.00072, "dranmar-hydrodissection-particle-v1", "hydrodissection fluid"),
        "dranmar_dissection_smoke_particle.usda": particle_usda(SMOKE_PARTICLE_ROOT, "SmokeParticle", 0.00085, "dranmar-dissection-smoke-particle-v1", "surgical smoke"),
        "dranmar_dissection_blood_particle.usda": particle_usda(BLOOD_PARTICLE_ROOT, "BloodParticle", 0.00078, "dranmar-dissection-blood-particle-v1", "blood complication"),
        "dranmar_duct_leak_particle.usda": particle_usda(DUCT_FLUID_ROOT, "DuctFluid", 0.00074, "dranmar-duct-leak-particle-v1", "duct leak complication"),
        "README.md": readme(),
    }
    mapping.update(structure_sources(bundle))
    for name, text in mapping.items():
        path = ASSET_ROOT / name
        path.write_text(text, encoding="utf-8")
        files.append(path)
    common_license = Path("/usr/share/common-licenses/Apache-2.0")
    license_text = common_license.read_text(encoding="utf-8") if common_license.exists() else "Apache License 2.0\nCopyright 2026 DrAnmar Project Developers\n"
    license_path = ASSET_ROOT / "LICENSE.txt"
    license_path.write_text(license_text, encoding="utf-8")
    files.append(license_path)
    files += generate_textures()
    files += export_glbs(bundle)
    files += [make_preview(bundle), make_full_arm_preview(bundle)]
    files += [
        write_json(ASSET_ROOT / "interaction_frames.json", interaction_frames(bundle)),
        write_json(ASSET_ROOT / "franka_mount_contract.json", mount_contract()),
        write_json(ASSET_ROOT / "dissection_topology.json", dissection_topology(bundle)),
        write_json(ASSET_ROOT / "safeplane_dissection_task_contract.json", task_contract()),
        write_json(ASSET_ROOT / "physics_profile.json", physics_profile(bundle)),
        write_json(ASSET_ROOT / "collider_coverage.json", collider_coverage(bundle)),
        write_json(ASSET_ROOT / "asset_manifest.json", asset_manifest()),
    ]
    write_json(PHYSICS_PROFILE_PATH, physics_profile(bundle))
    files.append(PHYSICS_PROFILE_PATH)
    INTEGRATION_PATH.write_text(integration_module(), encoding="utf-8")
    files.append(INTEGRATION_PATH)
    for name, text in (
        ("MECHANISM.md", docs_mechanism()),
        ("PHYSICAL_DISSECTION.md", docs_physical()),
        ("PROTECTED_STRUCTURE_SAFETY.md", docs_safety()),
        ("HYDRODISSECTION_AND_ENERGY.md", docs_hydro_energy()),
        ("FRANKA_INTEGRATION.md", docs_franka()),
        ("VALIDATION.md", docs_validation()),
    ):
        path = DOCS_ROOT / name
        path.write_text(text, encoding="utf-8")
        files.append(path)
    example = EXAMPLE_ROOT / "franka_safeplane_dissection_scene.py"
    example.write_text(example_scene(), encoding="utf-8")
    files.append(example)
    installer = PACKAGE_ROOT / "scripts/install_into_dranmar.py"
    installer.write_text(installer_source(), encoding="utf-8")
    installer.chmod(0o755)
    files.append(installer)
    return files


def sync_extension_data() -> None:
    target = EXTENSION_ROOT / "data" / CATALOG_SUBPATH
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ASSET_ROOT, target)


def all_payload_files() -> list[Path]:
    mirror_root = EXTENSION_ROOT / "data" / CATALOG_SUBPATH
    excluded_names = {"asset_manifest.json", "static_build_report.json", ".DS_Store"}
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name not in excluded_names
        and not path.is_relative_to(mirror_root)
    )


def build_manifest(files: Sequence[Path]) -> dict[str, object]:
    return {
        "schema": "dranmar.asset-manifest.v1",
        "asset": "dranmar-safeplane-dissection-robot-v1",
        "version": VERSION,
        "catalog_subpath": CATALOG_SUBPATH.as_posix(),
        "file_count": len(files),
        "files": [
            {
                "path": path.relative_to(PACKAGE_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }


def zip_tree(source: Path, output: Path, *, prefix: str | None = None) -> Path:
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
                or path.name == ".DS_Store"
            ):
                continue
            relative = path.relative_to(source).as_posix()
            archive_path = f"{prefix.rstrip('/')}/{relative}" if prefix else relative
            info = zipfile.ZipInfo(archive_path, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (
                0o755 if path.stat().st_mode & 0o111 else 0o644
            ) << 16
            archive.writestr(
                info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output


def write_checksum(path: Path) -> Path:
    output = path.with_suffix(path.suffix + ".sha256")
    output.write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    return output


def build_overlay() -> Path:
    temp = PACKAGE_ROOT.parent / "_dranmar_safeplane_dissection_overlay"
    shutil.rmtree(temp, ignore_errors=True)
    for subpath in ("source", "physics_next", "docs", "examples", "tests"):
        source = PACKAGE_ROOT / subpath
        if source.exists():
            shutil.copytree(source, temp / subpath, dirs_exist_ok=True)
    (temp / "scripts").mkdir(parents=True, exist_ok=True)
    for name in (
        SCRIPT_PATH.name,
        "install_into_dranmar.py",
        "requirements_safeplane_dissection_generation.txt",
        "validate_dranmar_safeplane_dissection_robot.py",
    ):
        source = PACKAGE_ROOT / "scripts" / name
        if source.exists():
            shutil.copy2(source, temp / "scripts" / name)
    output = PACKAGE_ROOT.parent / "dranmar_safeplane_dissection_robot_repo_overlay_v0.1.0.zip"
    zip_tree(temp, output)
    shutil.rmtree(temp)
    return output


def static_report(files: Sequence[Path]) -> dict[str, object]:
    usda_checks = []
    for path in (item for item in files if item.suffix == ".usda"):
        text = path.read_text(encoding="utf-8")
        usda_checks.append(
            {
                "file": path.relative_to(PACKAGE_ROOT).as_posix(),
                "brace_balance": text.count("{") == text.count("}"),
                "nested_quaternion_suspect": "(1, (" in text or "(0, (" in text,
                "one_line_over_suspect": any(line.strip().startswith("over ") and "{" in line and "}" in line for line in text.splitlines()),
                "flat_quaternion_count": text.count("quatf "),
                "absolute_external_asset_reference_count": sum(1 for line in text.splitlines() if "@/" in line),
            }
        )
    return {
        "schema": "dranmar.static-build-report.v1",
        "asset": "dranmar-safeplane-dissection-robot-v1",
        "usda_checks": usda_checks,
        "python_files": [
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in files if path.suffix == ".py"
        ],
        "runtime_observation": "archived_smoke_record_not_physical_qualification",
        "physical_effect_qualified": False,
    }


def syntax_check_python(paths: Sequence[Path]) -> None:
    for path in paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def generate() -> dict[str, object]:
    for junk in sorted(PACKAGE_ROOT.rglob(".DS_Store")):
        junk.unlink()
    for cache in sorted(PACKAGE_ROOT.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache)
    for bytecode in PACKAGE_ROOT.rglob("*.pyc"):
        bytecode.unlink()
    for legacy in (
        "FRANKA_INTEGRATION.md", "HYDRODISSECTION_AND_ENERGY.md",
        "MECHANISM.md", "PHYSICAL_DISSECTION.md",
        "PROTECTED_STRUCTURE_SAFETY.md",
    ):
        path = PACKAGE_ROOT / "docs" / legacy
        if path.exists():
            path.unlink()
    old_manifest = ASSET_ROOT / "asset_manifest.json"
    if old_manifest.exists():
        old_manifest.unlink()
    bundle = build_tool()
    write_asset_files(bundle)
    files = all_payload_files()
    manifest = write_json(ASSET_ROOT / "asset_manifest.json", build_manifest(files))
    sync_extension_data()
    static_path = write_json(
        PACKAGE_ROOT / "static_build_report.json",
        static_report(all_payload_files()),
    )
    for python_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        compile(python_path.read_text(encoding="utf-8"), str(python_path), "exec")
    development = PACKAGE_ROOT.parent / "dranmar_safeplane_dissection_robot_v0.1.0.zip"
    zip_tree(PACKAGE_ROOT, development, prefix=PACKAGE_ROOT.name)
    catalog = PACKAGE_ROOT.parent / "dranmar_safeplane_dissection_robot_catalog_v0.1.0.zip"
    zip_tree(PACKAGE_ROOT / "assets", catalog)
    overlay = build_overlay()
    for path in (development, catalog, overlay):
        write_checksum(path)
    release = {
        "schema": "dranmar.release.v1",
        "asset": "dranmar-safeplane-dissection-robot-v1",
        "version": VERSION,
        "catalog_subpath": CATALOG_SUBPATH.as_posix(),
        "development_package": {"path": str(development), "sha256": sha256(development)},
        "catalog_package": {"path": str(catalog), "sha256": sha256(catalog)},
        "repository_overlay": {"path": str(overlay), "sha256": sha256(overlay)},
        "primary_assets": [
            str(ASSET_ROOT / name)
            for name in (
                "dranmar_safeplane_dissection_tool_payload.usda",
                "dranmar_safeplane_dissection_tool_standalone.usda",
                "dranmar_safeplane_dissection_tool_rigid_proxy.usda",
                "dranmar_safeplane_tissue_demo.usda",
                "dranmar_adhesion_bridge.usda",
                "dranmar_protected_vessel_branch.usda",
                "dranmar_protected_nerve_branch.usda",
                "dranmar_protected_duct_branch.usda",
                "dranmar_micro_scissors_cartridge.usda",
            )
        ],
        "runtime_validation": static_report(all_payload_files())["runtime_validation"],
        "clinical_validation": False,
    }
    release_path = PACKAGE_ROOT.parent / "dranmar_safeplane_dissection_robot_release_v0.1.0.json"
    write_json(release_path, release)
    return {
        "release": release,
        "release_path": str(release_path),
        "manifest": str(manifest),
        "static_report": str(static_path),
    }


def main() -> None:
    print(json.dumps(generate(), indent=2))


if __name__ == "__main__":
    main()
