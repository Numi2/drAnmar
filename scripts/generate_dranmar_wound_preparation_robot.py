#!/usr/bin/env python3
"""Generate the DrAnmar wound-preparation robotic end-effector asset family.

The package is a manufacturer-neutral engineering research asset for irrigation,
aspiration, controlled debridement, wound-bed inspection, and fluid-accounting
workflows in NVIDIA Isaac Sim / Isaac Lab. It is not clinically validated and is
not approved for patient care.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

VERSION = "0.1.0"
ASSET_NAME = "DrAnmar Wound Preparation End Effector"
CATALOG_SUBPATH = Path("Props/SurgicalPreparation/WoundPreparationRobot")
ROOT_PRIM = "DrAnmarWoundPreparationTool"
STANDALONE_ROOT = "DrAnmarWoundPreparationToolStandalone"
PROXY_ROOT = "DrAnmarWoundPreparationToolRigidProxy"
DROPLET_ROOT = "DrAnmarIrrigationDroplet"
DEBRIS_ROOT = "DrAnmarDebridementFragment"
WOUND_ROOT = "DrAnmarWoundBedDemo"
BRUSH_ROOT = "DrAnmarBrushCartridge"
CURETTE_ROOT = "DrAnmarCuretteCartridge"
PAD_ROOT = "DrAnmarPadCartridge"

SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[1]
ASSET_ROOT = PACKAGE_ROOT / "assets" / CATALOG_SUBPATH
GLB_ROOT = ASSET_ROOT / "glb"
TEXTURE_ROOT = ASSET_ROOT / "textures"
PREVIEW_ROOT = PACKAGE_ROOT / "previews"
DOCS_ROOT = PACKAGE_ROOT / "docs"
EXAMPLE_ROOT = PACKAGE_ROOT / "examples"
EXTENSION_ROOT = PACKAGE_ROOT / "source/extensions/orbit.surgical.assets"
INTEGRATION_PATH = EXTENSION_ROOT / "orbit/surgical/assets/wound_preparation_robot.py"

# Tool coordinates: +Z approaches the wound; +X is lateral; +Y follows the wound.
WORK_PLANE_Z = 0.172
IRRIGATION_RESERVOIR_ML = 45.0
WASTE_CANISTER_ML = 55.0
IRRIGATION_NOZZLE_COUNT = 10
SUCTION_SLOT_COUNT = 12
BRISTLE_COUNT = 48
FRANKA_HAND_EQUIVALENT_ROTATION_DEG = -45.0


def f(value: float, digits: int = 10) -> str:
    if abs(value) < 10 ** (-(digits - 1)):
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
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=float,
    )


def matrix_to_quat_wxyz(m: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(m, dtype=float)
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        q = np.asarray([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.asarray([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.asarray([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.asarray([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
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
        R = rotation_matrix((0, 1, 0), math.pi / 2)
    elif axis == "y":
        R = rotation_matrix((1, 0, 0), -math.pi / 2)
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


def torus_z(major_radius: float, minor_radius: float, center=(0.0, 0.0, 0.0), major_sections=64, minor_sections=16) -> trimesh.Trimesh:
    mesh = trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius, major_sections=major_sections, minor_sections=minor_sections)
    return transform(mesh, center)


def frustum_z(radius0: float, radius1: float, height: float, center=(0.0, 0.0, 0.0), sections=48) -> trimesh.Trimesh:
    z0, z1 = -height / 2, height / 2
    points: list[tuple[float, float, float]] = []
    for z, r in ((z0, radius0), (z1, radius1)):
        for i in range(sections):
            a = 2 * math.pi * i / sections
            points.append((r * math.cos(a), r * math.sin(a), z))
    points.extend([(0, 0, z0), (0, 0, z1)])
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        faces += [(i, j, sections + j), (i, sections + j, sections + i), (2 * sections, j, i), (2 * sections + 1, sections + i, sections + j)]
    return transform(trimesh.Trimesh(vertices=np.asarray(points), faces=np.asarray(faces), process=True), center)


def capsule_between(p0: Sequence[float], p1: Sequence[float], radius: float, sections: int = 24) -> trimesh.Trimesh:
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return transform(trimesh.creation.icosphere(subdivisions=2, radius=radius), p0)
    mesh = trimesh.creation.capsule(radius=radius, height=max(0.0, length - 2 * radius), count=[sections, sections])
    z = np.asarray([0.0, 0.0, 1.0])
    d = direction / length
    cross = np.cross(z, d)
    dot = float(np.clip(np.dot(z, d), -1.0, 1.0))
    if np.linalg.norm(cross) <= 1e-12:
        R = np.eye(3) if dot > 0 else rotation_matrix((1, 0, 0), math.pi)
    else:
        R = rotation_matrix(cross, math.acos(dot))
    return transform(mesh, (p0 + p1) / 2, R)


def wire_path(points: Sequence[Sequence[float]], radius: float) -> trimesh.Trimesh:
    points = [np.asarray(p, dtype=float) for p in points]
    parts: list[trimesh.Trimesh] = []
    for a, b in zip(points[:-1], points[1:]):
        parts.append(capsule_between(a, b, radius))
    for p in points[1:-1]:
        parts.append(transform(trimesh.creation.icosphere(subdivisions=2, radius=radius), p))
    return trimesh.util.concatenate(parts)


def annular_sector_mesh(inner_radius: float, outer_radius: float, height: float, center=(0, 0, 0), start_angle=0.0, end_angle=math.pi / 6, sections=12) -> trimesh.Trimesh:
    z0, z1 = -height / 2, height / 2
    angles = np.linspace(start_angle, end_angle, sections + 1)
    vertices = []
    for z in (z0, z1):
        for r in (inner_radius, outer_radius):
            for a in angles:
                vertices.append((r * math.cos(a), r * math.sin(a), z))
    n = sections + 1
    faces: list[tuple[int, int, int]] = []
    # index helper: layer(0/1), ring(0 inner/1 outer), i
    idx = lambda layer, ring, i: layer * 2 * n + ring * n + i
    for i in range(sections):
        # top and bottom annular faces
        faces += [(idx(1, 0, i), idx(1, 1, i), idx(1, 1, i + 1)), (idx(1, 0, i), idx(1, 1, i + 1), idx(1, 0, i + 1))]
        faces += [(idx(0, 0, i), idx(0, 1, i + 1), idx(0, 1, i)), (idx(0, 0, i), idx(0, 0, i + 1), idx(0, 1, i + 1))]
        # inner and outer walls
        faces += [(idx(0, 0, i), idx(1, 0, i + 1), idx(0, 0, i + 1)), (idx(0, 0, i), idx(1, 0, i), idx(1, 0, i + 1))]
        faces += [(idx(0, 1, i), idx(0, 1, i + 1), idx(1, 1, i + 1)), (idx(0, 1, i), idx(1, 1, i + 1), idx(1, 1, i))]
    # radial end walls
    for i in (0, sections):
        faces += [(idx(0, 0, i), idx(0, 1, i), idx(1, 1, i)), (idx(0, 0, i), idx(1, 1, i), idx(1, 0, i))]
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
    return transform(mesh, center)


def mesh_bounds(meshes: Sequence[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    mins = np.vstack([m.bounds[0] for m in meshes])
    maxs = np.vstack([m.bounds[1] for m in meshes])
    return mins.min(axis=0), maxs.max(axis=0)


def box_mass_properties(meshes: Sequence[trimesh.Trimesh], mass: float) -> dict[str, object]:
    bmin, bmax = mesh_bounds(meshes)
    size = np.maximum(bmax - bmin, 1e-5)
    com = (bmin + bmax) * 0.5
    dx, dy, dz = size
    inertia = (
        mass * (dy * dy + dz * dz) / 12,
        mass * (dx * dx + dz * dz) / 12,
        mass * (dx * dx + dy * dy) / 12,
    )
    return {
        "mass_kg": float(mass),
        "center_of_mass_m": [float(x) for x in com],
        "diagonal_inertia_kg_m2": [float(x) for x in inertia],
        "principal_axes_wxyz": [1.0, 0.0, 0.0, 0.0],
        "bounds_min_m": [float(x) for x in bmin],
        "bounds_max_m": [float(x) for x in bmax],
    }


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
class ToolBundle:
    links: dict[str, Link]
    joints: list[Joint]
    frames: dict[str, dict[str, object]]
    brush_cartridge: trimesh.Trimesh
    curette_cartridge: trimesh.Trimesh
    pad_cartridge: trimesh.Trimesh
    droplet: trimesh.Trimesh
    debris: trimesh.Trimesh
    wound_surface: trimesh.Trimesh
    wound_base: trimesh.Trimesh
    wound_debris: list[trimesh.Trimesh]


# ---------------------------- Geometry ----------------------------

def build_brush_cartridge() -> trimesh.Trimesh:
    parts = [cylinder_axis(0.0105, 0.004, "z", (0, 0, -0.0015), sections=56)]
    rings = [(0.0025, 8), (0.0055, 16), (0.0080, 24)]
    for radius, count in rings:
        for i in range(count):
            a = 2 * math.pi * i / count + (0.2 if count == 16 else 0.0)
            x, y = radius * math.cos(a), radius * math.sin(a)
            lean = np.asarray((-0.0015 * math.cos(a), -0.0015 * math.sin(a), 0.0065))
            parts.append(capsule_between((x, y, 0), (x + lean[0], y + lean[1], lean[2]), 0.00042, sections=12))
    return trimesh.util.concatenate(parts)


def build_curette_cartridge() -> trimesh.Trimesh:
    ring = torus_z(0.0083, 0.0008, (0, 0, 0.003), major_sections=72, minor_sections=12)
    hub = cylinder_axis(0.0040, 0.005, "z", (0, 0, -0.0005), sections=48)
    spokes = [capsule_between((0, 0, 0.001), (0.007 * math.cos(a), 0.007 * math.sin(a), 0.003), 0.00055) for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)]
    return trimesh.util.concatenate([ring, hub] + spokes)


def build_pad_cartridge() -> trimesh.Trimesh:
    base = cylinder_axis(0.0105, 0.003, "z", (0, 0, -0.001), sections=64)
    pad = ellipsoid_mesh((0.0102, 0.0102, 0.0025), (0, 0, 0.002), subdivisions=3)
    bumps = []
    for r, count in ((0.003, 8), (0.0065, 14)):
        for i in range(count):
            a = 2 * math.pi * i / count
            bumps.append(ellipsoid_mesh((0.0008, 0.0008, 0.00055), (r * math.cos(a), r * math.sin(a), 0.0042), subdivisions=1))
    return trimesh.util.concatenate([base, pad] + bumps)


def build_debris(seed: int = 7) -> trimesh.Trimesh:
    rng = np.random.default_rng(seed)
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    vertices = np.asarray(mesh.vertices)
    noise = rng.normal(0.0, 0.12, size=(len(vertices), 1))
    vertices = vertices * (1.0 + noise)
    vertices *= np.asarray((0.0052, 0.0036, 0.0018))
    mesh.vertices = vertices
    mesh.fix_normals()
    return mesh


def build_wound_surface(nx: int = 47, ny: int = 61) -> trimesh.Trimesh:
    xs = np.linspace(-0.070, 0.070, nx)
    ys = np.linspace(-0.055, 0.055, ny)
    vertices: list[tuple[float, float, float]] = []
    for y in ys:
        for x in xs:
            # Elliptical wound crater with rolled margins and mild surface texture.
            r = math.sqrt((x / 0.030) ** 2 + (y / 0.042) ** 2)
            crater = -0.0105 * math.exp(-2.4 * r * r)
            rim = 0.0030 * math.exp(-18.0 * (r - 1.0) ** 2)
            texture = 0.00035 * math.sin(x * 240) * math.cos(y * 170)
            z = crater + rim + texture
            vertices.append((x, y, z))
    faces: list[tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b, c, d = a + 1, a + nx, a + nx + 1
            faces += [(a, b, d), (a, d, c)]
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.fix_normals()
    return mesh


def build_tool() -> ToolBundle:
    brush = build_brush_cartridge()
    curette = build_curette_cartridge()
    pad = build_pad_cartridge()
    droplet = trimesh.creation.icosphere(subdivisions=3, radius=0.00065)
    debris = build_debris()
    wound_surface = build_wound_surface()
    wound_base = box_mesh((0.155, 0.125, 0.018), (0, 0, -0.019))
    wound_debris: list[trimesh.Trimesh] = []
    debris_centers = [(-0.013, -0.020, -0.006), (0.010, -0.016, -0.007), (-0.006, 0.000, -0.009), (0.014, 0.007, -0.006), (-0.012, 0.020, -0.006), (0.005, 0.026, -0.006), (0.0, -0.030, -0.005)]
    for i, center in enumerate(debris_centers):
        fragment = build_debris(20 + i)
        fragment.apply_scale((0.65 + 0.12 * (i % 3), 0.75 + 0.1 * ((i + 1) % 3), 0.7))
        fragment.apply_translation(center)
        wound_debris.append(fragment)

    links: dict[str, Link] = {}

    mount_visuals: list[Visual] = [
        Visual("FrankaAdapterPlate", cylinder_axis(0.032, 0.012, "z", (0, 0, 0.006), sections=64), "MountMetal", ("franka_mount",)),
        Visual("QuickReleaseRing", torus_z(0.028, 0.0030, (0, 0, 0.014)), "MountMetal"),
        Visual("MainHousing", ellipsoid_mesh((0.050, 0.040, 0.028), (0, 0, 0.050), subdivisions=3), "BodyPolymer", ("wound_preparation_end_effector",)),
        Visual("HousingCore", box_mesh((0.084, 0.068, 0.040), (0, 0, 0.050)), "BodyPolymer"),
        Visual("ManifoldBody", cylinder_axis(0.028, 0.032, "z", (0, 0, 0.111), sections=64), "AccentPolymer", ("fluid_manifold",)),
        Visual("SuctionAnnulus", torus_z(0.0165, 0.0030, (0, 0, 0.151), major_sections=72, minor_sections=18), "SuctionDark", ("annular_suction_port",)),
        Visual("IrrigationRing", torus_z(0.0090, 0.0013, (0, 0, 0.149), major_sections=72, minor_sections=14), "MountMetal", ("irrigation_manifold",)),
        Visual("CameraBridge", box_mesh((0.034, 0.012, 0.012), (0, -0.034, 0.096)), "DarkPolymer", ("inspection_sensor_bridge",)),
        Visual("CameraLeft", cylinder_axis(0.0042, 0.004, "y", (-0.010, -0.041, 0.096), sections=32), "SensorGlass", ("rgb_camera",)),
        Visual("CameraRight", cylinder_axis(0.0042, 0.004, "y", (0.010, -0.041, 0.096), sections=32), "SensorGlass", ("rgb_camera",)),
        Visual("DepthProjector", cylinder_axis(0.0035, 0.004, "y", (0, -0.041, 0.106), sections=32), "SensorBlue", ("depth_projector",)),
        Visual("FluorescenceSensor", cylinder_axis(0.0037, 0.004, "y", (0, -0.041, 0.086), sections=32), "SensorGlass", ("fluorescence_camera",)),
        Visual("IlluminationRing", torus_z(0.0225, 0.0011, (0, 0, 0.145), major_sections=72, minor_sections=10), "IndicatorGreen", ("illumination_ring",)),
        Visual("LabelPanel", box_mesh((0.044, 0.0012, 0.020), (0, -0.0405, 0.052)), "LabelMaterial"),
        Visual("ServicePanel", box_mesh((0.050, 0.0015, 0.026), (0, 0.0405, 0.052)), "DarkPolymer", ("service_panel",)),
    ]
    # Inward-converging irrigation micro-nozzles and visible suction slots.
    for i in range(IRRIGATION_NOZZLE_COUNT):
        a = 2 * math.pi * i / IRRIGATION_NOZZLE_COUNT
        p0 = np.asarray((0.0090 * math.cos(a), 0.0090 * math.sin(a), 0.149))
        p1 = np.asarray((0.0062 * math.cos(a), 0.0062 * math.sin(a), 0.1595))
        mount_visuals.append(Visual(f"IrrigationNozzle_{i:02d}", capsule_between(p0, p1, 0.00075, sections=14), "MountMetal", ("irrigation_nozzle",)))
    for i in range(SUCTION_SLOT_COUNT):
        a = 2 * math.pi * i / SUCTION_SLOT_COUNT
        sector = annular_sector_mesh(0.0132, 0.0186, 0.0030, (0, 0, 0.154), a - 0.055, a + 0.055, sections=5)
        mount_visuals.append(Visual(f"SuctionSlot_{i:02d}", sector, "SuctionDark", ("suction_slot",)))
    # Feed and aspiration tubes.
    mount_visuals.append(Visual("IrrigationFeedTube", wire_path([(-0.034, 0.016, 0.082), (-0.026, 0.018, 0.101), (-0.012, 0.010, 0.122), (0, 0, 0.139)], 0.0016), "IrrigationTube"))
    mount_visuals.append(Visual("SuctionReturnTube", wire_path([(0.034, 0.016, 0.082), (0.027, 0.019, 0.104), (0.014, 0.011, 0.128), (0, 0, 0.142)], 0.0021), "SuctionTube"))
    for i in range(6):
        a = 2 * math.pi * i / 6
        mount_visuals.append(Visual(f"MountBolt_{i:02d}", cylinder_axis(0.0020, 0.0020, "z", (0.024 * math.cos(a), 0.024 * math.sin(a), 0.0145), sections=24), "MountMetal"))
    mount_colliders = [
        Collider("AdapterCollider", "cylinder", (0, 0, 0.008), radius=0.032, height=0.016, physics_material="MountPhysics"),
        Collider("HousingCollider", "box", (0, 0, 0.052), size=(0.102, 0.082, 0.064), physics_material="PolymerPhysics"),
        Collider("ManifoldCollider", "cylinder", (0, 0, 0.118), radius=0.029, height=0.050, physics_material="PolymerPhysics"),
        Collider("SuctionCaptureVolume", "cylinder", (0, 0, 0.165), radius=0.023, height=0.026, physics_material="SuctionPhysics", role="suction_capture_volume"),
    ]
    links["Mount"] = Link("Mount", (0, 0, 0), mount_visuals, mount_colliders, 0.245, ("robotic_wound_preparation_end_effector",))

    reservoir_visuals = [
        Visual("ReservoirShell", cylinder_axis(0.016, 0.066, "z", (0, 0, 0), sections=64), "TransparentReservoir", ("irrigation_reservoir",)),
        Visual("ReservoirCap", cylinder_axis(0.017, 0.006, "z", (0, 0, -0.035), sections=56), "AccentPolymer"),
        Visual("ReservoirBase", cylinder_axis(0.017, 0.006, "z", (0, 0, 0.035), sections=56), "AccentPolymer"),
        Visual("IrrigationFluidFull", cylinder_axis(0.0132, 0.052, "z", (0, 0, 0.003), sections=56), "IrrigationFluid", ("sterile_irrigation_fluid",)),
        Visual("IrrigationFluidLow", cylinder_axis(0.0132, 0.018, "z", (0, 0, 0.020), sections=56), "IrrigationFluid", ("sterile_irrigation_fluid",)),
        Visual("VolumeScale", box_mesh((0.0010, 0.010, 0.046), (-0.0164, 0, 0.002)), "LabelMaterial"),
    ]
    links["IrrigationReservoir"] = Link(
        "IrrigationReservoir", (-0.035, 0.018, 0.064), reservoir_visuals,
        [Collider("ReservoirCollider", "cylinder", (0, 0, 0), radius=0.0165, height=0.070, physics_material="PolymerPhysics")],
        None, ("irrigation_inventory",),
    )

    waste_visuals = [
        Visual("CanisterShell", cylinder_axis(0.017, 0.066, "z", (0, 0, 0), sections=64), "TransparentWasteCanister", ("aspiration_collection_canister",)),
        Visual("CanisterCap", cylinder_axis(0.018, 0.007, "z", (0, 0, -0.035), sections=56), "SuctionDark"),
        Visual("CanisterBase", cylinder_axis(0.018, 0.007, "z", (0, 0, 0.035), sections=56), "SuctionDark"),
        Visual("WasteFluidPartial", cylinder_axis(0.0140, 0.026, "z", (0, 0, 0.017), sections=56), "WasteFluid", ("aspirated_fluid",)),
        Visual("WasteFluidFull", cylinder_axis(0.0140, 0.054, "z", (0, 0, 0.002), sections=56), "WasteFluid", ("aspirated_fluid",)),
        Visual("HydrophobicFilter", cylinder_axis(0.009, 0.003, "z", (0, 0, -0.030), sections=48), "FilterMaterial", ("suction_filter",)),
    ]
    links["WasteCanister"] = Link(
        "WasteCanister", (0.035, 0.018, 0.064), waste_visuals,
        [Collider("CanisterCollider", "cylinder", (0, 0, 0), radius=0.0175, height=0.070, physics_material="PolymerPhysics")],
        None, ("aspiration_inventory",),
    )

    guard_visuals = [
        Visual("CompliantGuardRing", torus_z(0.0225, 0.0036, (0, 0, 0)), "GuardElastomer", ("contact_guard",)),
        Visual("GuardSensorRing", torus_z(0.0185, 0.0012, (0, 0, -0.001)), "SensorBlue", ("force_sensor_ring",)),
    ]
    for i in range(8):
        a = 2 * math.pi * i / 8
        guard_visuals.append(Visual(f"GuardPad_{i:02d}", ellipsoid_mesh((0.0045, 0.0025, 0.0015), (0.0225 * math.cos(a), 0.0225 * math.sin(a), 0.0022), subdivisions=2), "GuardElastomer", ("tissue_contact_pad",)))
    links["ContactGuard"] = Link(
        "ContactGuard", (0, 0, 0.157), guard_visuals,
        [Collider("GuardCollider", "cylinder", (0, 0, 0), radius=0.0258, height=0.0075, physics_material="GuardPhysics", role="compliant_contact_guard")],
        0.036, ("compliant_contact_guard",),
    )

    carriage_visuals = [
        Visual("ExtensionSleeve", cylinder_axis(0.0115, 0.035, "z", (0, 0, 0.014), sections=56), "DarkPolymer", ("debridement_extension_carriage",)),
        Visual("LinearScale", box_mesh((0.002, 0.008, 0.028), (-0.0118, 0, 0.014)), "LabelMaterial"),
        Visual("SpindleBearing", torus_z(0.010, 0.0015, (0, 0, 0.031)), "MountMetal"),
    ]
    links["DebridementCarriage"] = Link(
        "DebridementCarriage", (0, 0, 0.112), carriage_visuals,
        [Collider("CarriageCollider", "cylinder", (0, 0, 0.014), radius=0.012, height=0.036, physics_material="PolymerPhysics")],
        0.042, ("debridement_extension_carriage",),
    )

    rotor_visuals = [
        Visual("RotorHub", cylinder_axis(0.0060, 0.012, "z", (0, 0, -0.005), sections=48), "MountMetal", ("debridement_rotor",)),
        Visual("BrushCartridge", brush, "BrushBristle", ("debridement_brush", "removable_cartridge")),
        Visual("CartridgeLockRing", torus_z(0.0108, 0.0010, (0, 0, -0.002)), "IndicatorAmber", ("cartridge_lock",)),
    ]
    links["DebridementRotor"] = Link(
        "DebridementRotor", (0, 0, 0.151), rotor_visuals,
        [
            Collider("RotorHubCollider", "cylinder", (0, 0, -0.005), radius=0.0062, height=0.013, physics_material="SteelPhysics"),
            Collider("BrushContactCollider", "cylinder", (0, 0, 0.003), radius=0.0108, height=0.008, physics_material="BrushPhysics", role="debridement_contact_surface"),
        ],
        0.019, ("rotary_debridement_head",),
    )

    irrigation_valve_visuals = [
        Visual("ValveSpool", cylinder_axis(0.0045, 0.020, "z", (0, 0, 0)), "ValveMetal", ("irrigation_metering_valve",)),
        Visual("ValveIndicator", cylinder_axis(0.0055, 0.002, "z", (0, 0, 0.011)), "IndicatorGreen"),
    ]
    links["IrrigationValve"] = Link(
        "IrrigationValve", (-0.023, 0.020, 0.088), irrigation_valve_visuals,
        [Collider("ValveCollider", "cylinder", (0, 0, 0), radius=0.0048, height=0.021, physics_material="SteelPhysics")],
        0.011, ("irrigation_metering_valve",),
    )

    suction_valve_visuals = [
        Visual("ValveDisk", cylinder_axis(0.010, 0.0035, "z", (0, 0, 0)), "ValveMetal", ("suction_control_valve",)),
        Visual("ValveHandle", box_mesh((0.020, 0.004, 0.004), (0.005, 0, 0.003)), "IndicatorAmber"),
    ]
    links["SuctionValve"] = Link(
        "SuctionValve", (0.023, 0.020, 0.088), suction_valve_visuals,
        [Collider("ValveDiskCollider", "cylinder", (0, 0, 0), radius=0.0102, height=0.004, physics_material="SteelPhysics")],
        0.014, ("suction_control_valve",),
    )

    joints = [
        Joint("irrigation_reservoir_fixed_joint", "fixed", "Mount", "IrrigationReservoir", None, (-0.035, 0.018, 0.064), (0, 0, 0)),
        Joint("waste_canister_fixed_joint", "fixed", "Mount", "WasteCanister", None, (0.035, 0.018, 0.064), (0, 0, 0)),
        Joint("contact_guard_joint", "prismatic", "Mount", "ContactGuard", "Z", (0, 0, 0.157), (0, 0, 0), 0.0, 0.008, 1100.0, 30.0, 32.0),
        Joint("debridement_extension_joint", "prismatic", "Mount", "DebridementCarriage", "Z", (0, 0, 0.112), (0, 0, 0), 0.0, 0.020, 4200.0, 115.0, 45.0),
        Joint("debridement_rotor_joint", "revolute", "DebridementCarriage", "DebridementRotor", "Z", (0, 0, 0.039), (0, 0, 0), -360000.0, 360000.0, 0.0, 0.018, 0.32, 0.0),
        Joint("irrigation_valve_joint", "prismatic", "Mount", "IrrigationValve", "Z", (-0.023, 0.020, 0.088), (0, 0, 0), 0.0, 0.006, 1800.0, 45.0, 25.0),
        Joint("suction_valve_joint", "revolute", "Mount", "SuctionValve", "Z", (0.023, 0.020, 0.088), (0, 0, 0), 0.0, 85.0, 8.0, 0.45, 2.0),
    ]

    frames = {
        "panda_link8_mount": {"position": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "Mount", "role": "franka_mount"},
        "wound_preparation_tcp": {"position": [0, 0, WORK_PLANE_Z], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "Mount", "role": "robot_tcp"},
        "contact_guard_center": {"position": [0, 0, 0.0], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "ContactGuard", "role": "contact_reference"},
        "debridement_contact": {"position": [0, 0, 0.0055], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "DebridementRotor", "role": "debridement_contact"},
        "rotor_axis": {"position": [0, 0, -0.004], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "DebridementRotor", "role": "rotor_axis"},
        "irrigation_jet_origin": {"position": [0, 0, 0.157], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "Mount", "role": "irrigation_emitter_center"},
        "suction_capture_center": {"position": [0, 0, 0.166], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "Mount", "role": "suction_capture_center"},
        "suction_throat": {"position": [0, 0, 0.145], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "Mount", "role": "suction_throat"},
        "camera_left": {"position": [-0.010, -0.041, 0.096], "orientation_wxyz": [0.70710678, 0.70710678, 0, 0], "parent_link": "Mount", "role": "rgb_camera"},
        "camera_right": {"position": [0.010, -0.041, 0.096], "orientation_wxyz": [0.70710678, 0.70710678, 0, 0], "parent_link": "Mount", "role": "rgb_camera"},
        "depth_camera": {"position": [0, -0.041, 0.106], "orientation_wxyz": [0.70710678, 0.70710678, 0, 0], "parent_link": "Mount", "role": "depth_camera"},
        "fluorescence_camera": {"position": [0, -0.041, 0.086], "orientation_wxyz": [0.70710678, 0.70710678, 0, 0], "parent_link": "Mount", "role": "fluorescence_camera"},
        "illumination_ring": {"position": [0, 0, 0.145], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "Mount", "role": "illumination"},
        "irrigation_reservoir_port": {"position": [0, 0, 0.035], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "IrrigationReservoir", "role": "fluid_supply_port"},
        "waste_canister_port": {"position": [0, 0, -0.035], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "WasteCanister", "role": "aspiration_return_port"},
        "cartridge_mount": {"position": [0, 0, -0.002], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "DebridementRotor", "role": "cartridge_mount"},
        "service_reference": {"position": [0, 0.041, 0.052], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "Mount", "role": "service_reference"},
        "count_reference": {"position": [0, -0.041, 0.048], "orientation_wxyz": [1, 0, 0, 0], "parent_link": "Mount", "role": "inventory_reference"},
    }

    return ToolBundle(links, joints, frames, brush, curette, pad, droplet, debris, wound_surface, wound_base, wound_debris)

# ---------------------------- OpenUSD authoring ----------------------------

def uv_for_mesh(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    bmin, bmax = mesh.bounds
    span = np.maximum(bmax - bmin, 1e-9)
    axes = np.argsort(span)[-2:]
    uv = np.empty((len(vertices), 2), dtype=float)
    uv[:, 0] = (vertices[:, axes[0]] - bmin[axes[0]]) / span[axes[0]]
    uv[:, 1] = (vertices[:, axes[1]] - bmin[axes[1]]) / span[axes[1]]
    return uv


def mesh_usda(visual: Visual, material_path: str, indent: str = "                ") -> str:
    mesh = visual.mesh.copy()
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    points = ",\n".join(indent + "    " + vec(p) for p in mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=int)
    counts = ", ".join("3" for _ in faces)
    indices = ", ".join(str(int(x)) for x in faces.reshape(-1))
    normals = ",\n".join(indent + "    " + vec(n) for n in mesh.vertex_normals)
    uvs = uv_for_mesh(mesh)
    uv_text = ",\n".join(indent + "    " + vec(uv) for uv in uvs)
    labels = ""
    if visual.labels:
        labels = "\n" + indent + "    custom token[] drAnmar:labels = [" + ", ".join(f'"{x}"' for x in visual.labels) + "]"
    return f'''{indent}def Mesh "{visual.name}" (
{indent}    prepend apiSchemas = ["MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    uniform bool doubleSided = false
{indent}    float3[] extent = [{vec(mesh.bounds[0])}, {vec(mesh.bounds[1])}]
{indent}    int[] faceVertexCounts = [{counts}]
{indent}    int[] faceVertexIndices = [{indices}]
{indent}    point3f[] points = [
{points}
{indent}    ]
{indent}    normal3f[] normals = [
{normals}
{indent}    ] (
{indent}        interpolation = "vertex"
{indent}    )
{indent}    texCoord2f[] primvars:st = [
{uv_text}
{indent}    ] (
{indent}        interpolation = "vertex"
{indent}    )
{indent}    rel material:binding = <{material_path}>
{indent}    uniform token subdivisionScheme = "none"{labels}
{indent}}}'''


def collider_usda(collider: Collider, root_path: str, indent: str = "                ") -> str:
    schemas = ["PhysicsCollisionAPI", "MaterialBindingAPI", "PhysxCollisionAPI"]
    attrs = [
        f'{indent}    rel material:binding:physics = <{root_path}/PhysicsMaterials/{collider.physics_material}>',
        f'{indent}    custom token drAnmar:role = "{collider.role}"',
        f'{indent}    quatf xformOp:orient = {quat(collider.orientation_wxyz)}',
        f'{indent}    double3 xformOp:translate = {vec(collider.center)}',
        f'{indent}    float physxCollision:contactOffset = 0.0005',
        f'{indent}    float physxCollision:restOffset = 0',
    ]
    if collider.author_enabled:
        attrs.insert(0, f'{indent}    bool physics:collisionEnabled = true')
    if collider.kind == "box":
        assert collider.size is not None
        attrs += [
            f'{indent}    double3 xformOp:scale = {vec(collider.size)}',
            f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]',
        ]
        return f'''{indent}def Cube "{collider.name}" (
{indent}    prepend apiSchemas = [{", ".join(f'"{x}"' for x in schemas)}]
{indent})
{indent}{{
{indent}    double size = 1
{chr(10).join(attrs)}
{indent}}}'''
    if collider.kind in {"cylinder", "capsule"}:
        assert collider.radius is not None and collider.height is not None
        schema_name = "Cylinder" if collider.kind == "cylinder" else "Capsule"
        attrs += [
            f'{indent}    uniform token axis = "{collider.axis.upper()}"',
            f'{indent}    double radius = {f(collider.radius)}',
            f'{indent}    double height = {f(collider.height)}',
            f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
        ]
        return f'''{indent}def {schema_name} "{collider.name}" (
{indent}    prepend apiSchemas = [{", ".join(f'"{x}"' for x in schemas)}]
{indent})
{indent}{{
{chr(10).join(attrs)}
{indent}}}'''
    raise ValueError(collider.kind)


def shader_material(root_name: str, name: str, color: Sequence[float], metallic: float, roughness: float, opacity: float = 1.0, texture: str | None = None) -> str:
    texture_block = ""
    diffuse = f"color3f inputs:diffuseColor = {vec(color)}"
    if texture:
        texture_block = f'''
            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{texture}@
                token inputs:sourceColorSpace = "sRGB"
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                float2 inputs:st.connect = </{root_name}/Looks/{name}/Primvar.outputs:result>
                float3 outputs:rgb
            }}
            def Shader "Primvar"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
'''
        diffuse = f"color3f inputs:diffuseColor.connect = </{root_name}/Looks/{name}/Texture.outputs:rgb>"
    return f'''        def Material "{name}"
        {{
            token outputs:surface.connect = </{root_name}/Looks/{name}/Shader.outputs:surface>
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                {diffuse}
                float inputs:metallic = {f(metallic)}
                float inputs:roughness = {f(roughness)}
                float inputs:opacity = {f(opacity)}
                token outputs:surface
            }}{texture_block}
        }}'''


def looks_scope(root_name: str) -> str:
    definitions = [
        ("MountMetal", (0.28, 0.32, 0.36), 0.88, 0.20, 1.0, "./textures/brushed_metal_basecolor.png"),
        ("ValveMetal", (0.46, 0.50, 0.56), 0.86, 0.22, 1.0, "./textures/brushed_metal_basecolor.png"),
        ("CuretteSteel", (0.69, 0.72, 0.76), 0.94, 0.15, 1.0, "./textures/brushed_metal_basecolor.png"),
        ("BodyPolymer", (0.86, 0.89, 0.92), 0.03, 0.40, 1.0, "./textures/body_polymer_basecolor.png"),
        ("AccentPolymer", (0.07, 0.36, 0.52), 0.04, 0.34, 1.0, "./textures/blue_polymer_basecolor.png"),
        ("DarkPolymer", (0.06, 0.09, 0.12), 0.05, 0.38, 1.0, None),
        ("SuctionDark", (0.025, 0.034, 0.040), 0.32, 0.30, 1.0, None),
        ("TransparentReservoir", (0.50, 0.78, 0.90), 0.0, 0.10, 0.25, None),
        ("TransparentWasteCanister", (0.56, 0.70, 0.72), 0.0, 0.12, 0.28, None),
        ("IrrigationFluid", (0.18, 0.67, 0.96), 0.0, 0.05, 0.62, None),
        ("WasteFluid", (0.54, 0.14, 0.12), 0.0, 0.12, 0.64, None),
        ("IrrigationTube", (0.12, 0.52, 0.78), 0.0, 0.20, 0.78, None),
        ("SuctionTube", (0.12, 0.16, 0.17), 0.0, 0.34, 0.88, None),
        ("GuardElastomer", (0.05, 0.20, 0.21), 0.0, 0.72, 1.0, "./textures/guard_elastomer_basecolor.png"),
        ("BrushBristle", (0.20, 0.78, 0.83), 0.0, 0.68, 1.0, "./textures/brush_bristle_basecolor.png"),
        ("PadFoam", (0.72, 0.86, 0.84), 0.0, 0.82, 1.0, "./textures/pad_foam_basecolor.png"),
        ("FilterMaterial", (0.88, 0.89, 0.86), 0.0, 0.84, 1.0, None),
        ("SensorGlass", (0.04, 0.10, 0.14), 0.20, 0.08, 0.72, None),
        ("SensorBlue", (0.03, 0.56, 0.95), 0.10, 0.18, 1.0, None),
        ("IndicatorGreen", (0.12, 0.96, 0.38), 0.0, 0.18, 1.0, None),
        ("IndicatorAmber", (0.98, 0.48, 0.08), 0.0, 0.24, 1.0, None),
        ("LabelMaterial", (0.98, 0.98, 0.99), 0.0, 0.48, 1.0, "./textures/dranmar_wound_prep_label.png"),
        ("TissueVisual", (0.72, 0.27, 0.25), 0.0, 0.66, 1.0, "./textures/tissue_basecolor.png"),
        ("WoundVisual", (0.43, 0.08, 0.07), 0.0, 0.76, 1.0, "./textures/wound_basecolor.png"),
        ("DebrisVisual", (0.74, 0.65, 0.30), 0.0, 0.78, 1.0, "./textures/debris_basecolor.png"),
        ("WaterVisual", (0.18, 0.68, 0.98), 0.0, 0.03, 0.55, None),
        ("BasePad", (0.18, 0.20, 0.22), 0.0, 0.72, 1.0, None),
    ]
    mats = [shader_material(root_name, *entry) for entry in definitions]
    return '    def Scope "Looks"\n    {\n' + "\n".join(mats) + "\n    }"


def physics_materials_scope() -> str:
    values = {
        "MountPhysics": (0.42, 0.32, 0.02),
        "PolymerPhysics": (0.56, 0.44, 0.03),
        "SteelPhysics": (0.36, 0.26, 0.02),
        "GuardPhysics": (1.10, 0.88, 0.0),
        "BrushPhysics": (0.82, 0.64, 0.0),
        "CurettePhysics": (0.34, 0.24, 0.01),
        "SuctionPhysics": (0.18, 0.12, 0.0),
        "WaterPhysics": (0.04, 0.02, 0.0),
        "DebrisPhysics": (0.76, 0.58, 0.0),
        "TissuePhysics": (0.70, 0.56, 0.0),
    }
    blocks = []
    for name, (static, dynamic, restitution) in values.items():
        blocks.append(f'''        def Material "{name}" (
            prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
        )
        {{
            float physics:staticFriction = {f(static)}
            float physics:dynamicFriction = {f(dynamic)}
            float physics:restitution = {f(restitution)}
            uniform token physxMaterial:frictionCombineMode = "max"
            uniform token physxMaterial:restitutionCombineMode = "min"
        }}''')
    return '    def Scope "PhysicsMaterials"\n    {\n' + "\n".join(blocks) + "\n    }"


def frame_usda(name: str, data: dict[str, object], indent: str = "                ") -> str:
    return f'''{indent}def Xform "{name}"
{indent}{{
{indent}    double3 xformOp:translate = {vec(data["position"])}
{indent}    quatf xformOp:orient = {quat(data["orientation_wxyz"])}
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{indent}    custom token drAnmar:role = "{data["role"]}"
{indent}    custom token drAnmar:parentLink = "{data["parent_link"]}"
{indent}}}'''


def link_usda(link: Link, root_path: str, frames: dict[str, dict[str, object]]) -> str:
    visual_blocks = [mesh_usda(v, f"{root_path}/Looks/{v.material}") for v in link.visuals]
    collider_blocks = [collider_usda(c, root_path) for c in link.colliders]
    labels = ", ".join(f'"{x}"' for x in link.labels)
    local_frames = []
    link_translation = np.asarray(link.translation, dtype=float)
    for name, data in frames.items():
        if data["parent_link"] != link.name:
            continue
        local_data = dict(data)
        # Frame positions in bundle are already link-local unless the parent is Mount.
        if link.name == "Mount":
            local_data["position"] = list(np.asarray(data["position"], dtype=float))
        local_frames.append(frame_usda(name, local_data))
    mass_block = ""
    if link.mass_properties is not None:
        p = link.mass_properties
        mass_block = f'''
            float physics:mass = {f(p["mass_kg"])}
            point3f physics:centerOfMass = {vec(p["center_of_mass_m"])}
            float3 physics:diagonalInertia = {vec(p["diagonal_inertia_kg_m2"])}
            quatf physics:principalAxes = {quat(p["principal_axes_wxyz"])}'''
    return f'''        def Xform "{link.name}" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
        )
        {{
            double3 xformOp:translate = {vec(link.translation)}
            quatf xformOp:orient = (1, 0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
            bool physics:rigidBodyEnabled = true
            bool physics:kinematicEnabled = false{mass_block}
            bool physxRigidBody:enableCCD = true
            bool physxRigidBody:enableSpeculativeCCD = true
            int physxRigidBody:solverPositionIterationCount = 16
            int physxRigidBody:solverVelocityIterationCount = 4
            float physxRigidBody:maxDepenetrationVelocity = 0.45
            custom token[] drAnmar:labels = [{labels}]

            def Scope "Visuals"
            {{
{chr(10).join(visual_blocks)}
            }}
            def Scope "Collisions"
            {{
{chr(10).join(collider_blocks)}
            }}
            def Scope "Frames"
            {{
{chr(10).join(local_frames)}
            }}
        }}'''


def joint_usda(joint: Joint, root_path: str) -> str:
    body0 = f"{root_path}/Links/{joint.body0}"
    body1 = f"{root_path}/Links/{joint.body1}"
    common = f'''            rel physics:body0 = <{body0}>
            rel physics:body1 = <{body1}>
            point3f physics:localPos0 = {vec(joint.local_pos0)}
            point3f physics:localPos1 = {vec(joint.local_pos1)}
            quatf physics:localRot0 = (1, 0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            bool physics:collisionEnabled = false'''
    if joint.type == "fixed":
        return f'''        def PhysicsFixedJoint "{joint.name}"
        {{
{common}
        }}'''
    if joint.type == "prismatic":
        return f'''        def PhysicsPrismaticJoint "{joint.name}" (
            prepend apiSchemas = ["PhysicsDriveAPI:linear"]
        )
        {{
{common}
            uniform token physics:axis = "{joint.axis}"
            float physics:lowerLimit = {f(joint.lower)}
            float physics:upperLimit = {f(joint.upper)}
            uniform token drive:linear:physics:type = "force"
            float drive:linear:physics:targetPosition = 0
            float drive:linear:physics:targetVelocity = {f(joint.target_velocity)}
            float drive:linear:physics:stiffness = {f(joint.stiffness)}
            float drive:linear:physics:damping = {f(joint.damping)}
            float drive:linear:physics:maxForce = {f(joint.max_force)}
        }}'''
    if joint.type == "revolute":
        return f'''        def PhysicsRevoluteJoint "{joint.name}" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular"]
        )
        {{
{common}
            uniform token physics:axis = "{joint.axis}"
            float physics:lowerLimit = {f(joint.lower)}
            float physics:upperLimit = {f(joint.upper)}
            uniform token drive:angular:physics:type = "force"
            float drive:angular:physics:targetPosition = 0
            float drive:angular:physics:targetVelocity = {f(joint.target_velocity)}
            float drive:angular:physics:stiffness = {f(joint.stiffness)}
            float drive:angular:physics:damping = {f(joint.damping)}
            float drive:angular:physics:maxForce = {f(joint.max_force)}
        }}'''
    raise ValueError(joint.type)


def root_header(articulation_root: bool) -> str:
    schemas = ["SemanticsLabelsAPI:class", "SemanticsLabelsAPI:workflow"]
    if articulation_root:
        schemas.append("PhysicsArticulationRootAPI")
    return f'''def Xform "{ROOT_PRIM}" (
    prepend apiSchemas = [{", ".join(f'"{x}"' for x in schemas)}]
    prepend variantSets = ["irrigation_state", "collection_state"]
    variants = {{
        string irrigation_state = "loaded"
        string collection_state = "empty"
    }}
    assetInfo = {{
        string name = "{ROOT_PRIM}"
        string version = "{VERSION}"
    }}
    customData = {{
        string drAnmarStatus = "simulation_training_workcell"
        string drAnmarMountInterface = "franka_panda_link8_hand_replacement"
        string drAnmarMechanism = "concentric_irrigation_annular_aspiration_rotary_debridement_and_multimodal_inspection"
        string drAnmarCoordinateConvention = "+Z approach, +X lateral, +Y wound tangent"
        bool drAnmarClinicalValidation = false
        bool drAnmarMedicalDevice = false
    }}
    kind = "component"
)'''


def visibility_over(name: str, visibility: str, indent: str) -> str:
    return f'''{indent}over "{name}"
{indent}{{
{indent}    token visibility = "{visibility}"
{indent}}}'''


def mass_over(link_name: str, mass: float, com: Sequence[float], inertia: Sequence[float], indent: str = "                ") -> str:
    return f'''{indent}over "{link_name}"
{indent}{{
{indent}    float physics:mass = {f(mass)}
{indent}    point3f physics:centerOfMass = {vec(com)}
{indent}    float3 physics:diagonalInertia = {vec(inertia)}
{indent}    quatf physics:principalAxes = (1, 0, 0, 0)
{indent}}}'''


def state_link_over(
    link_name: str,
    mass: float,
    com: Sequence[float],
    inertia: Sequence[float],
    visibility: Mapping[str, str],
    indent: str = "                ",
) -> str:
    visual_overrides = "\n".join(
        visibility_over(name, value, indent + "        ")
        for name, value in visibility.items()
    )
    return f'''{indent}over "{link_name}"
{indent}{{
{indent}    float physics:mass = {f(mass)}
{indent}    point3f physics:centerOfMass = {vec(com)}
{indent}    float3 physics:diagonalInertia = {vec(inertia)}
{indent}    quatf physics:principalAxes = (1, 0, 0, 0)
{indent}    over "Visuals"
{indent}    {{
{visual_overrides}
{indent}    }}
{indent}}}'''


def state_variants(bundle: ToolBundle) -> str:
    # Deliberately author state-controlled mass and visibility only inside variants.
    reservoir_com = [0, 0, 0]
    reservoir_inertia = [1.8e-5, 1.8e-5, 4.1e-6]
    canister_com = [0, 0, 0]
    canister_inertia = [2.1e-5, 2.1e-5, 4.7e-6]
    return f'''    variantSet "irrigation_state" = {{
        "loaded" {{
            over "Links"
            {{
{state_link_over("IrrigationReservoir", 0.073, reservoir_com, reservoir_inertia, {"IrrigationFluidFull": "inherited", "IrrigationFluidLow": "invisible"})}
            }}
        }}
        "low" {{
            over "Links"
            {{
{state_link_over("IrrigationReservoir", 0.041, reservoir_com, [1.1e-5, 1.1e-5, 3.1e-6], {"IrrigationFluidFull": "invisible", "IrrigationFluidLow": "inherited"})}
            }}
        }}
        "empty" {{
            over "Links"
            {{
{state_link_over("IrrigationReservoir", 0.028, reservoir_com, [8.0e-6, 8.0e-6, 2.4e-6], {"IrrigationFluidFull": "invisible", "IrrigationFluidLow": "invisible"})}
            }}
        }}
    }}

    variantSet "collection_state" = {{
        "empty" {{
            over "Links"
            {{
{state_link_over("WasteCanister", 0.030, canister_com, [8.8e-6, 8.8e-6, 2.8e-6], {"WasteFluidPartial": "invisible", "WasteFluidFull": "invisible"})}
            }}
        }}
        "partial" {{
            over "Links"
            {{
{state_link_over("WasteCanister", 0.057, canister_com, [1.5e-5, 1.5e-5, 4.0e-6], {"WasteFluidPartial": "inherited", "WasteFluidFull": "invisible"})}
            }}
        }}
        "full" {{
            over "Links"
            {{
{state_link_over("WasteCanister", 0.085, canister_com, canister_inertia, {"WasteFluidPartial": "invisible", "WasteFluidFull": "inherited"})}
            }}
        }}
    }}'''


def tool_usda(bundle: ToolBundle, articulation_root: bool) -> str:
    root_path = f"/{ROOT_PRIM}"
    links = "\n".join(link_usda(link, root_path, bundle.frames) for link in bundle.links.values())
    joints = "\n".join(joint_usda(joint, root_path) for joint in bundle.joints)
    root_frames = "\n".join(frame_usda(name, data, indent="        ") for name, data in bundle.frames.items())
    return f'''#usda 1.0
(
    defaultPrim = "{ROOT_PRIM}"
    doc = "Dr.Anmar simulation-training end effector with irrigation, aspiration, rotary debridement, and inspection."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

{root_header(articulation_root)}
{{
    token[] semantics:labels:class = ["robotic_wound_preparation_end_effector", "surgical_preparation_device"]
    token[] semantics:labels:workflow = ["wound_inspection", "irrigation", "aspiration", "debridement"]

{looks_scope(ROOT_PRIM)}
{physics_materials_scope()}

    def Scope "Links"
    {{
{links}
    }}

    def Scope "Joints"
    {{
{joints}
    }}

    def Scope "RestPoseFrames"
    {{
{root_frames}
    }}

{state_variants(bundle)}
}}
'''


def rigid_proxy_usda(bundle: ToolBundle) -> str:
    entries = world_visual_entries(bundle, "ready")
    combined = trimesh.util.concatenate([mesh for _, mesh, _ in entries])
    visual = Visual("CombinedVisual", combined, "BodyPolymer", ("robotic_wound_preparation_end_effector", "rigid_proxy"))
    mass = box_mass_properties([combined], 0.485)
    proxy_root = f"/{PROXY_ROOT}"
    colliders = [
        Collider("Housing", "box", (0, 0, 0.055), size=(0.105, 0.085, 0.070), physics_material="PolymerPhysics"),
        Collider("Manifold", "cylinder", (0, 0, 0.125), radius=0.031, height=0.066, physics_material="PolymerPhysics"),
        Collider("WorkHead", "cylinder", (0, 0, 0.160), radius=0.028, height=0.024, physics_material="GuardPhysics"),
        Collider("Reservoir", "cylinder", (-0.035, 0.018, 0.064), radius=0.018, height=0.071, physics_material="PolymerPhysics"),
        Collider("Canister", "cylinder", (0.035, 0.018, 0.064), radius=0.019, height=0.071, physics_material="PolymerPhysics"),
    ]
    return f'''#usda 1.0
(
    defaultPrim = "{PROXY_ROOT}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{PROXY_ROOT}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "SemanticsLabelsAPI:class"]
    assetInfo = {{
        string name = "{PROXY_ROOT}"
        string version = "{VERSION}"
    }}
    kind = "component"
)
{{
    token[] semantics:labels:class = ["robotic_wound_preparation_end_effector", "rigid_proxy"]
    bool physics:rigidBodyEnabled = true
    float physics:mass = {f(mass["mass_kg"])}
    point3f physics:centerOfMass = {vec(mass["center_of_mass_m"])}
    float3 physics:diagonalInertia = {vec(mass["diagonal_inertia_kg_m2"])}
    quatf physics:principalAxes = (1, 0, 0, 0)
    bool physxRigidBody:enableCCD = true
{looks_scope(PROXY_ROOT)}
{physics_materials_scope()}
    def Scope "Visuals"
    {{
{mesh_usda(visual, f"/{PROXY_ROOT}/Looks/BodyPolymer", indent="        ")}
    }}
    def Scope "Collisions"
    {{
{chr(10).join(collider_usda(c, proxy_root, indent="        ") for c in colliders)}
    }}
    def Scope "Frames"
    {{
{frame_usda("wound_preparation_tcp", {"position":[0,0,WORK_PLANE_Z],"orientation_wxyz":[1,0,0,0],"role":"robot_tcp","parent_link":"proxy"}, indent="        ")}
{frame_usda("count_reference", {"position":[0,-0.041,0.048],"orientation_wxyz":[1,0,0,0],"role":"inventory_reference","parent_link":"proxy"}, indent="        ")}
    }}
}}
'''


def simple_rigid_asset_usda(root_name: str, visual: Visual, collider: Collider, mass_kg: float, class_labels: Sequence[str], frames: dict[str, dict[str, object]], doc: str) -> str:
    p = box_mass_properties([visual.mesh], mass_kg)
    root_path = f"/{root_name}"
    return f'''#usda 1.0
(
    defaultPrim = "{root_name}"
    doc = "{doc}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root_name}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "SemanticsLabelsAPI:class"]
    assetInfo = {{
        string name = "{root_name}"
        string version = "{VERSION}"
    }}
    kind = "component"
)
{{
    token[] semantics:labels:class = [{", ".join(f'"{x}"' for x in class_labels)}]
    bool physics:rigidBodyEnabled = true
    float physics:mass = {f(p["mass_kg"])}
    point3f physics:centerOfMass = {vec(p["center_of_mass_m"])}
    float3 physics:diagonalInertia = {vec(p["diagonal_inertia_kg_m2"])}
    quatf physics:principalAxes = {quat(p["principal_axes_wxyz"])}
    bool physxRigidBody:enableCCD = true
{looks_scope(root_name)}
{physics_materials_scope()}
    def Scope "Visuals"
    {{
{mesh_usda(visual, f"/{root_name}/Looks/{visual.material}", indent="        ")}
    }}
    def Scope "Collisions"
    {{
{collider_usda(collider, root_path, indent="        ")}
    }}
    def Scope "Frames"
    {{
{chr(10).join(frame_usda(name, data, indent="        ") for name, data in frames.items())}
    }}
}}
'''


def cartridge_usda(root_name: str, mesh: trimesh.Trimesh, material: str, collider_kind: str, role: str, mass: float) -> str:
    if collider_kind == "brush":
        collider = Collider("ContactCollider", "cylinder", (0, 0, 0.002), radius=0.0108, height=0.008, physics_material="BrushPhysics", role="debridement_contact_surface")
    elif collider_kind == "curette":
        collider = Collider("ContactCollider", "cylinder", (0, 0, 0.003), radius=0.0094, height=0.003, physics_material="CurettePhysics", role="debridement_contact_surface")
    else:
        collider = Collider("ContactCollider", "cylinder", (0, 0, 0.002), radius=0.0108, height=0.006, physics_material="BrushPhysics", role="debridement_contact_surface")
    frames = {
        "cartridge_mount": {"position": [0, 0, -0.003], "orientation_wxyz": [1, 0, 0, 0], "parent_link": root_name, "role": "cartridge_mount"},
        "contact_reference": {"position": [0, 0, 0.005], "orientation_wxyz": [1, 0, 0, 0], "parent_link": root_name, "role": "debridement_contact"},
    }
    return simple_rigid_asset_usda(root_name, Visual("Visual", mesh, material, (role,)), collider, mass, [role, "debridement_cartridge"], frames, f"Dr.Anmar {role} simulation-training cartridge.")


def wound_surface_mesh_usda(mesh: trimesh.Trimesh, root_name: str) -> str:
    return mesh_usda(Visual("SimulationMesh", mesh, "TissueVisual", ("wound_bed_tissue", "surface_deformable_candidate")), f"/{root_name}/Looks/TissueVisual", indent="        ")


def wound_bed_usda(bundle: ToolBundle) -> str:
    root = WOUND_ROOT
    debris_blocks = []
    for i, fragment in enumerate(bundle.wound_debris):
        local = fragment.copy()
        # Geometry is already in wound coordinates; write as its own rigid body.
        p = box_mass_properties([local], 0.00045 + i * 0.00005)
        visual = Visual("Visual", local, "DebrisVisual", ("adherent_debris", "debridement_target"))
        bmin, bmax = local.bounds
        center = (bmin + bmax) / 2
        size = np.maximum(bmax - bmin, 0.001)
        collider = Collider("AdhesionPatch", "box", tuple(float(x) for x in center), size=tuple(float(x) for x in size * np.asarray((0.85, 0.85, 0.60))), physics_material="DebrisPhysics", role="debris_attachment_region")
        debris_blocks.append(f'''        def Xform "Fragment_{i:02d}" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
        )
        {{
            bool physics:rigidBodyEnabled = true
            float physics:mass = {f(p["mass_kg"])}
            point3f physics:centerOfMass = {vec(p["center_of_mass_m"])}
            float3 physics:diagonalInertia = {vec(p["diagonal_inertia_kg_m2"])}
            quatf physics:principalAxes = (1, 0, 0, 0)
            bool physxRigidBody:enableCCD = true
            custom float drAnmar:adhesionWorkThresholdJ = {f(0.006 + 0.0015 * (i % 4))}
            def Scope "Visuals"
            {{
{mesh_usda(visual, f"/{root}/Looks/DebrisVisual", indent="                ")}
            }}
            def Scope "Collisions"
            {{
{collider_usda(collider, f"/{root}", indent="                ")}
            }}
        }}''')
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Dr.Anmar wound-bed simulation-training surface with detachable debris fragments."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["SemanticsLabelsAPI:class", "SemanticsLabelsAPI:workflow"]
    kind = "component"
)
{{
    token[] semantics:labels:class = ["wound_bed", "debridement_training_surface"]
    token[] semantics:labels:workflow = ["inspection", "irrigation", "debridement", "aspiration"]
{looks_scope(root)}
{physics_materials_scope()}
    def Xform "Base" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
    )
    {{
{mesh_usda(Visual("BaseVisual", bundle.wound_base, "BasePad"), f"/{root}/Looks/BasePad", indent="        ")}
        rel material:binding:physics = <{f'/{root}/PhysicsMaterials/PolymerPhysics'}>
    }}
    def Xform "TissueSurface"
    {{
{wound_surface_mesh_usda(bundle.wound_surface, root)}
        def Xform "AttachmentBand"
        {{
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            custom token drAnmar:role = "wound_surface_attachment_band"
        }}
    }}
    def Scope "Debris"
    {{
{chr(10).join(debris_blocks)}
    }}
    def Scope "Frames"
    {{
{frame_usda("wound_center", {"position":[0,0,-0.009],"orientation_wxyz":[1,0,0,0],"role":"wound_center","parent_link":"TissueSurface"}, indent="        ")}
{frame_usda("scan_reference", {"position":[0,-0.045,0.018],"orientation_wxyz":[1,0,0,0],"role":"inspection_reference","parent_link":"TissueSurface"}, indent="        ")}
{frame_usda("count_reference", {"position":[0,0.050,0],"orientation_wxyz":[1,0,0,0],"role":"inventory_reference","parent_link":"TissueSurface"}, indent="        ")}
    }}
}}
'''

# ---------------------------- GLB and preview ----------------------------

def material_color(material: str) -> tuple[int, int, int, int]:
    colors = {
        "MountMetal": (78, 86, 96, 255),
        "ValveMetal": (128, 136, 148, 255),
        "CuretteSteel": (188, 194, 204, 255),
        "BodyPolymer": (224, 229, 234, 255),
        "AccentPolymer": (25, 107, 150, 255),
        "DarkPolymer": (22, 28, 34, 255),
        "SuctionDark": (8, 13, 16, 255),
        "TransparentReservoir": (108, 190, 224, 80),
        "TransparentWasteCanister": (128, 170, 175, 88),
        "IrrigationFluid": (38, 156, 235, 160),
        "WasteFluid": (126, 40, 35, 175),
        "IrrigationTube": (24, 122, 191, 210),
        "SuctionTube": (25, 40, 42, 235),
        "GuardElastomer": (12, 56, 59, 255),
        "BrushBristle": (43, 194, 204, 255),
        "PadFoam": (175, 217, 211, 255),
        "FilterMaterial": (226, 226, 218, 255),
        "SensorGlass": (10, 28, 38, 210),
        "SensorBlue": (10, 146, 242, 255),
        "IndicatorGreen": (36, 240, 93, 255),
        "IndicatorAmber": (247, 117, 18, 255),
        "LabelMaterial": (247, 248, 250, 255),
        "TissueVisual": (187, 70, 66, 255),
        "WoundVisual": (105, 22, 20, 255),
        "DebrisVisual": (188, 163, 70, 255),
        "WaterVisual": (32, 164, 241, 145),
        "BasePad": (45, 49, 54, 255),
        "RobotWhite": (228, 232, 236, 255),
        "RobotDark": (48, 54, 61, 255),
        "RobotJoint": (94, 102, 113, 255),
        "DebugOrange": (247, 117, 18, 170),
        "DebugGreen": (40, 235, 90, 170),
        "DebugBlue": (15, 145, 240, 170),
    }
    return colors.get(material, (180, 180, 180, 255))


def pbr(mesh: trimesh.Trimesh, material: str) -> trimesh.Trimesh:
    mesh = mesh.copy()
    color = material_color(material)
    metal = 0.85 if any(token in material for token in ("Metal", "Steel")) else 0.02
    roughness = 0.22 if metal > 0.5 else 0.48
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name=material,
            baseColorFactor=np.asarray(color, dtype=np.uint8),
            metallicFactor=metal,
            roughnessFactor=roughness,
            alphaMode="BLEND" if color[3] < 255 else "OPAQUE",
        )
    )
    return mesh


def link_pose(bundle: ToolBundle, phase: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    poses = {name: (np.zeros(3), np.eye(3)) for name in bundle.links}
    if phase in {"contact", "debride", "irrigate", "aspirate"}:
        poses["ContactGuard"] = (np.asarray((0, 0, 0.006)), np.eye(3))
    if phase == "debride":
        poses["DebridementCarriage"] = (np.asarray((0, 0, 0.018)), np.eye(3))
        poses["DebridementRotor"] = (np.asarray((0, 0, 0.018)), rotation_matrix((0, 0, 1), math.radians(28)))
    if phase == "irrigate":
        poses["IrrigationValve"] = (np.asarray((0, 0, 0.006)), np.eye(3))
    if phase == "aspirate":
        poses["SuctionValve"] = (np.zeros(3), rotation_matrix((0, 0, 1), math.radians(76)))
    return poses


def world_visual_entries(bundle: ToolBundle, phase: str = "ready") -> list[tuple[str, trimesh.Trimesh, str]]:
    poses = link_pose(bundle, phase)
    entries: list[tuple[str, trimesh.Trimesh, str]] = []
    for link_name, link in bundle.links.items():
        delta, rotation = poses[link_name]
        base = np.asarray(link.translation) + delta
        for visual in link.visuals:
            mesh = visual.mesh.copy()
            T = np.eye(4)
            T[:3, :3] = rotation
            T[:3, 3] = base
            mesh.apply_transform(T)
            entries.append((f"{link_name}_{visual.name}", mesh, visual.material))
    return entries


def export_scene(path: Path, entries: Sequence[tuple[str, trimesh.Trimesh, str]]) -> None:
    scene = trimesh.Scene()
    for name, mesh, material in entries:
        scene.add_geometry(pbr(mesh, material), node_name=name, geom_name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))


def collider_mesh(collider: Collider) -> trimesh.Trimesh:
    if collider.kind == "box":
        assert collider.size is not None
        return box_mesh(collider.size, collider.center)
    if collider.kind == "cylinder":
        assert collider.radius is not None and collider.height is not None
        return cylinder_axis(collider.radius, collider.height, collider.axis, collider.center)
    if collider.kind == "capsule":
        assert collider.radius is not None and collider.height is not None
        axis = {"x": np.asarray([1, 0, 0]), "y": np.asarray([0, 1, 0]), "z": np.asarray([0, 0, 1])}[collider.axis]
        return capsule_between(np.asarray(collider.center) - axis * collider.height / 2, np.asarray(collider.center) + axis * collider.height / 2, collider.radius)
    raise ValueError(collider.kind)


def collision_debug_entries(bundle: ToolBundle, phase: str = "ready") -> list[tuple[str, trimesh.Trimesh, str]]:
    poses = link_pose(bundle, phase)
    entries: list[tuple[str, trimesh.Trimesh, str]] = []
    for link_name, link in bundle.links.items():
        delta, rotation = poses[link_name]
        base = np.asarray(link.translation) + delta
        for collider in link.colliders:
            mesh = collider_mesh(collider)
            T = np.eye(4)
            T[:3, :3] = rotation
            T[:3, 3] = base
            mesh.apply_transform(T)
            material = "DebugGreen" if "capture" in collider.role or "contact" in collider.role else "DebugOrange"
            entries.append((f"{link_name}_{collider.name}", mesh, material))
    return entries


def frame_world_position(bundle: ToolBundle, frame: dict[str, object], phase: str = "ready") -> tuple[np.ndarray, np.ndarray]:
    parent = str(frame["parent_link"])
    local = np.asarray(frame["position"], dtype=float)
    q = np.asarray(frame["orientation_wxyz"], dtype=float)
    w, x, y, z = q
    frame_rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    delta, parent_rotation = link_pose(bundle, phase)[parent]
    parent_translation = np.asarray(bundle.links[parent].translation) + delta
    return parent_translation + parent_rotation @ local, parent_rotation @ frame_rotation


def axis_entries(bundle: ToolBundle, phase: str = "ready", length=0.012, radius=0.0005) -> list[tuple[str, trimesh.Trimesh, str]]:
    entries = []
    for name, frame in bundle.frames.items():
        position, rotation = frame_world_position(bundle, frame, phase)
        for axis, material in ((rotation[:, 0], "DebugOrange"), (rotation[:, 1], "DebugGreen"), (rotation[:, 2], "DebugBlue")):
            entries.append((f"frame_{name}_{material}", capsule_between(position, position + axis * length, radius), material))
    return entries


def irrigation_stream_entries(bundle: ToolBundle, count_per_nozzle: int = 5) -> list[tuple[str, trimesh.Trimesh, str]]:
    entries: list[tuple[str, trimesh.Trimesh, str]] = []
    for nozzle in range(IRRIGATION_NOZZLE_COUNT):
        angle = 2 * math.pi * nozzle / IRRIGATION_NOZZLE_COUNT
        origin = np.asarray((0.0062 * math.cos(angle), 0.0062 * math.sin(angle), 0.160))
        target = np.asarray((0.0018 * math.cos(angle), 0.0018 * math.sin(angle), WORK_PLANE_Z + 0.002))
        for index in range(count_per_nozzle):
            t = (index + 1) / (count_per_nozzle + 1)
            position = origin * (1 - t) + target * t
            droplet = bundle.droplet.copy()
            droplet.apply_translation(position)
            entries.append((f"jet_{nozzle:02d}_{index:02d}", droplet, "WaterVisual"))
    return entries


def suction_debug_entries(bundle: ToolBundle) -> list[tuple[str, trimesh.Trimesh, str]]:
    entries = []
    for index in range(18):
        angle = 2 * math.pi * index / 18
        radius = 0.020 - 0.0007 * index
        z = 0.174 - 0.0016 * index
        droplet = bundle.droplet.copy()
        droplet.apply_scale(1.4)
        droplet.apply_translation((radius * math.cos(angle), radius * math.sin(angle), z))
        entries.append((f"aspirated_particle_{index:02d}", droplet, "WaterVisual"))
    return entries


def wound_entries(bundle: ToolBundle, include_debris: bool = True) -> list[tuple[str, trimesh.Trimesh, str]]:
    entries = [("wound_base", bundle.wound_base, "BasePad"), ("wound_surface", bundle.wound_surface, "TissueVisual")]
    if include_debris:
        entries += [(f"debris_{i:02d}", mesh, "DebrisVisual") for i, mesh in enumerate(bundle.wound_debris)]
    return entries


def franka_proxy_entries(bundle: ToolBundle, phase: str = "ready") -> list[tuple[str, trimesh.Trimesh, str]]:
    points = [
        np.asarray((-0.30, 0.00, -0.72)), np.asarray((-0.30, 0.00, -0.59)), np.asarray((-0.43, 0.00, -0.44)),
        np.asarray((-0.40, 0.03, -0.20)), np.asarray((-0.27, -0.02, -0.26)), np.asarray((-0.12, -0.02, -0.23)),
        np.asarray((-0.08, 0.02, -0.14)), np.asarray((0.00, 0.00, -0.055)), np.asarray((0.00, 0.00, 0.00)),
    ]
    entries: list[tuple[str, trimesh.Trimesh, str]] = [
        ("arm_base", cylinder_axis(0.105, 0.075, "z", tuple(points[0])), "RobotDark"),
        ("arm_pedestal", cylinder_axis(0.078, 0.10, "z", tuple(points[0] + np.asarray((0, 0, 0.08)))), "RobotWhite"),
    ]
    for index, (a, b) in enumerate(zip(points[1:-1], points[2:])):
        entries.append((f"arm_link_{index:02d}", capsule_between(a, b, 0.052 if index < 3 else 0.043), "RobotWhite"))
    for index, point in enumerate(points[1:-1]):
        joint = trimesh.creation.icosphere(subdivisions=2, radius=0.061 if index < 3 else 0.050)
        joint.apply_translation(point)
        entries.append((f"arm_joint_{index:02d}", joint, "RobotJoint"))
    entries.append(("panda_link8_proxy", cylinder_axis(0.045, 0.11, "z", (0, 0, -0.055)), "RobotDark"))
    entries.extend(world_visual_entries(bundle, phase))
    return entries


def exploded_entries(bundle: ToolBundle) -> list[tuple[str, trimesh.Trimesh, str]]:
    offsets = {
        "Mount": np.asarray((0, 0, 0)),
        "IrrigationReservoir": np.asarray((-0.060, 0.030, 0.0)),
        "WasteCanister": np.asarray((0.060, 0.030, 0.0)),
        "ContactGuard": np.asarray((0, 0, 0.045)),
        "DebridementCarriage": np.asarray((0, 0, 0.030)),
        "DebridementRotor": np.asarray((0, 0, 0.070)),
        "IrrigationValve": np.asarray((-0.035, 0.050, 0.010)),
        "SuctionValve": np.asarray((0.035, 0.050, 0.010)),
    }
    entries = []
    for name, link in bundle.links.items():
        base = np.asarray(link.translation) + offsets.get(name, np.zeros(3))
        for visual in link.visuals:
            mesh = visual.mesh.copy()
            mesh.apply_translation(base)
            entries.append((f"{name}_{visual.name}", mesh, visual.material))
    return entries


def export_glbs(bundle: ToolBundle) -> list[Path]:
    paths: list[Path] = []
    for phase in ("ready", "contact", "debride", "irrigate", "aspirate"):
        entries = world_visual_entries(bundle, phase)
        if phase == "irrigate":
            entries += irrigation_stream_entries(bundle)
        if phase == "aspirate":
            entries += suction_debug_entries(bundle)
        path = GLB_ROOT / f"dranmar_wound_preparation_tool_{phase}.glb"
        export_scene(path, entries)
        paths.append(path)
    path = GLB_ROOT / "dranmar_wound_preparation_tool_collision_debug.glb"
    export_scene(path, world_visual_entries(bundle, "contact") + collision_debug_entries(bundle, "contact"))
    paths.append(path)
    path = GLB_ROOT / "dranmar_wound_preparation_tool_frame_debug.glb"
    export_scene(path, world_visual_entries(bundle, "ready") + axis_entries(bundle))
    paths.append(path)
    path = GLB_ROOT / "dranmar_wound_preparation_tool_exploded.glb"
    export_scene(path, exploded_entries(bundle))
    paths.append(path)
    path = GLB_ROOT / "dranmar_franka_wound_preparation_assembly.glb"
    export_scene(path, franka_proxy_entries(bundle, "ready"))
    paths.append(path)
    path = GLB_ROOT / "dranmar_wound_bed_demo.glb"
    export_scene(path, wound_entries(bundle, True))
    paths.append(path)
    path = GLB_ROOT / "dranmar_wound_bed_cleaned.glb"
    export_scene(path, wound_entries(bundle, False))
    paths.append(path)
    for name, mesh, material in (
        ("brush", bundle.brush_cartridge, "BrushBristle"),
        ("curette", bundle.curette_cartridge, "CuretteSteel"),
        ("pad", bundle.pad_cartridge, "PadFoam"),
        ("irrigation_droplet", bundle.droplet, "WaterVisual"),
        ("debridement_fragment", bundle.debris, "DebrisVisual"),
    ):
        path = GLB_ROOT / f"dranmar_{name}.glb"
        export_scene(path, [(name, mesh, material)])
        paths.append(path)
    return paths


def add_mesh_to_axis(ax, mesh: trimesh.Trimesh, material: str, max_faces: int = 1600) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces = np.asarray(mesh.triangles)
    if len(faces) > max_faces:
        faces = faces[:: max(1, len(faces) // max_faces)]
    rgba = np.asarray(material_color(material), dtype=float) / 255.0
    poly = Poly3DCollection(faces, facecolor=rgba[:3], edgecolor=(0.05, 0.06, 0.07, 0.10), linewidth=0.06, alpha=max(0.24, rgba[3]))
    ax.add_collection3d(poly)


def make_preview(bundle: ToolBundle) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17, 10), dpi=160)
    phases = [("Inspect", "ready"), ("Concentric irrigation", "irrigate"), ("Controlled debridement", "debride"), ("Annular aspiration", "aspirate")]
    for index, (title, phase) in enumerate(phases, 1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        entries = world_visual_entries(bundle, phase)
        wound = [(name, mesh.copy(), material) for name, mesh, material in wound_entries(bundle, include_debris=phase != "aspirate")]
        for _, mesh, _ in wound:
            mesh.apply_translation((0, 0, WORK_PLANE_Z + 0.012))
        entries += wound
        if phase == "irrigate":
            entries += irrigation_stream_entries(bundle, 4)
        if phase == "aspirate":
            entries += suction_debug_entries(bundle)
        for _, mesh, material in entries:
            add_mesh_to_axis(ax, mesh, material)
        ax.set_xlim(-0.09, 0.09)
        ax.set_ylim(-0.09, 0.09)
        ax.set_zlim(-0.01, 0.205)
        ax.view_init(elev=24, azim=-58)
        ax.set_box_aspect((1, 1, 1.18))
        ax.set_axis_off()
        ax.set_title(title, fontsize=14, pad=8)
    fig.suptitle("DrAnmar Wound Preparation End Effector", fontsize=20, y=0.97)
    fig.text(0.5, 0.02, "Franka panda_link8 replacement • inspection • conserved irrigation particles • rotary debridement • annular aspiration", ha="center", fontsize=11)
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    output = PREVIEW_ROOT / "dranmar_wound_preparation_robot_preview.png"
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def make_full_arm_preview(bundle: ToolBundle) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 10), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    for _, mesh, material in franka_proxy_entries(bundle, "ready"):
        add_mesh_to_axis(ax, mesh, material, max_faces=2200)
    ax.set_xlim(-0.58, 0.18)
    ax.set_ylim(-0.35, 0.35)
    ax.set_zlim(-0.82, 0.22)
    ax.view_init(elev=20, azim=-56)
    ax.set_box_aspect((0.76, 0.70, 1.0))
    ax.set_axis_off()
    ax.set_title("DrAnmar Wound Preparation Tool on Franka Interface", fontsize=16, pad=12)
    fig.text(0.5, 0.025, "Host-arm geometry is an inspection proxy; runtime integration uses the Isaac Franka and replaces the Panda hand at panda_link8.", ha="center", fontsize=9)
    output = PREVIEW_ROOT / "dranmar_wound_preparation_robot_full_arm_preview.png"
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output

# ---------------------------- Textures and metadata ----------------------------

def noise_texture(base: tuple[int, int, int], size: int = 512, strength: int = 18, seed: int = 1) -> Image.Image:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, strength, size=(size, size, 1))
    base_array = np.asarray(base, dtype=float).reshape(1, 1, 3)
    image = np.clip(base_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def save_texture(image: Image.Image, name: str) -> Path:
    TEXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEXTURE_ROOT / name
    image.save(path, optimize=True)
    return path


def generate_textures() -> list[Path]:
    outputs = [
        save_texture(noise_texture((218, 224, 230), strength=8, seed=1), "body_polymer_basecolor.png"),
        save_texture(noise_texture((28, 105, 145), strength=10, seed=2), "blue_polymer_basecolor.png"),
        save_texture(noise_texture((15, 58, 59), strength=7, seed=3), "guard_elastomer_basecolor.png"),
        save_texture(noise_texture((47, 191, 198), strength=13, seed=4), "brush_bristle_basecolor.png"),
        save_texture(noise_texture((181, 215, 210), strength=15, seed=5), "pad_foam_basecolor.png"),
        save_texture(noise_texture((145, 153, 164), strength=11, seed=6), "brushed_metal_basecolor.png"),
        save_texture(noise_texture((183, 71, 66), strength=16, seed=7), "tissue_basecolor.png"),
        save_texture(noise_texture((108, 24, 22), strength=20, seed=8), "wound_basecolor.png"),
        save_texture(noise_texture((183, 158, 70), strength=24, seed=9), "debris_basecolor.png"),
    ]
    label = Image.new("RGB", (1024, 512), (246, 248, 250))
    draw = ImageDraw.Draw(label)
    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 92)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 38)
    except OSError:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()
    draw.rounded_rectangle((28, 28, 996, 484), radius=38, outline=(22, 83, 118), width=12)
    draw.text((76, 88), "DrAnmar", fill=(18, 72, 104), font=font_big)
    draw.text((78, 218), "WOUND PREPARATION", fill=(20, 36, 46), font=font_small)
    draw.text((78, 286), "IRRIGATE • DEBRIDE • ASPIRATE", fill=(24, 126, 162), font=font_small)
    draw.text((78, 362), "SIMULATION TRAINING WORKCELL", fill=(164, 62, 38), font=font_small)
    outputs.append(save_texture(label, "dranmar_wound_prep_label.png"))
    return outputs


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def interaction_frames(bundle: ToolBundle) -> dict[str, object]:
    return {
        "schema": "dr.anmar.interaction-frames.v1",
        "asset": ASSET_NAME,
        "version": VERSION,
        "coordinate_convention": {"x": "lateral", "y": "along_wound", "z": "toward_patient"},
        "frames": bundle.frames,
        "microjet_layout": {
            "count": IRRIGATION_NOZZLE_COUNT,
            "ring_radius_m": 0.0090,
            "exit_radius_m": 0.0062,
            "exit_z_m": 0.1595,
            "convergence_target_z_m": WORK_PLANE_Z + 0.002,
        },
        "suction_layout": {
            "slot_count": SUCTION_SLOT_COUNT,
            "annulus_inner_radius_m": 0.0132,
            "annulus_outer_radius_m": 0.0186,
            "capture_radius_m": 0.023,
            "capture_depth_m": 0.026,
        },
    }


def mount_contract() -> dict[str, object]:
    return {
        "schema": "dr.anmar.franka-payload-mount.v1",
        "asset": ASSET_NAME,
        "version": VERSION,
        "host": "Isaac Lab Franka Panda",
        "host_link": "panda_link8",
        "disabled_stock_prims": [
            "panda_hand_joint", "panda_hand", "panda_finger_joint1", "panda_finger_joint2", "panda_leftfinger", "panda_rightfinger"
        ],
        "payload_root": ROOT_PRIM,
        "payload_mount_link": f"{ROOT_PRIM}/Links/Mount",
        "mount_rotation_deg_about_z": FRANKA_HAND_EQUIVALENT_ROTATION_DEG,
        "mount_translation_m": [0.0, 0.0, 0.0],
        "tcp": "wound_preparation_tcp",
        "status": "category_level_research_interface_pending_runtime_and_metrology_iteration",
    }


def task_contract() -> dict[str, object]:
    return {
        "schema": "dr.anmar.wound-preparation-task.v1",
        "asset": ASSET_NAME,
        "version": VERSION,
        "sequence": ["inspect", "contact", "pre_rinse", "aspirate", "debride", "post_rinse", "dry", "verify"],
        "mechanisms": {
            "inspection": ["stereo_rgb", "depth_projector", "fluorescence_camera", "illumination_ring"],
            "irrigation": "ten inward-converging microjets with a metering spool and volume ledger",
            "aspiration": "twelve-slot annular suction crown with capture field and collection ledger",
            "debridement": "extendable rotary cartridge with compliant guard and work-based debris release",
        },
        "success_metrics": [
            "target coverage", "remaining debris fraction", "fluid emitted ml", "fluid recovered ml", "unrecovered fluid ml",
            "peak contact force", "debridement work", "surface dwell time", "collision count", "verification coverage"
        ],
        "blocked_claims": [
            "clinical debridement efficacy", "bacterial reduction", "healing improvement", "tissue viability diagnosis",
            "sterility", "manufacturer equivalence", "patient-care approval"
        ],
    }


def physics_profile(bundle: ToolBundle) -> dict[str, object]:
    return {
        "schema": "dr.anmar.wound-preparation-physics.v1",
        "id": "dranmar-wound-preparation-robot-v1",
        "name": ASSET_NAME,
        "version": VERSION,
        "status": "simulation_training_model",
        "units": "metres-kilograms-seconds",
        "mount": mount_contract(),
        "tool": {
            "work_plane_z_m": WORK_PLANE_Z,
            "nominal_total_mass_loaded_empty_collection_kg": 0.490,
            "irrigation_capacity_ml": IRRIGATION_RESERVOIR_ML,
            "collection_capacity_ml": WASTE_CANISTER_ML,
            "irrigation_nozzles": IRRIGATION_NOZZLE_COUNT,
            "suction_slots": SUCTION_SLOT_COUNT,
            "brush_bristles": BRISTLE_COUNT,
        },
        "joints": {
            joint.name: {
                "type": joint.type,
                "axis": joint.axis,
                "lower": joint.lower,
                "upper": joint.upper,
                "stiffness": joint.stiffness,
                "damping": joint.damping,
                "max_force": joint.max_force,
                "target_velocity": joint.target_velocity,
                "parameter_status": "provisional_engineering_seed",
            }
            for joint in bundle.joints
        },
        "particle_irrigation": {
            "particle_radius_m": 0.00065,
            "particle_volume_ml": 4.0 / 3.0 * math.pi * 0.00065**3 * 1e6,
            "nominal_jet_speed_m_s": 1.20,
            "particle_material": "PhysX PBD liquid",
            "conservation": "ledger tracks reservoir, emitted, active, aspirated, spilled and discarded volumes",
            "status": "particle-scale proxy_not_cfd_or_clinical_dose_model",
        },
        "suction": {
            "capture_radius_m": 0.023,
            "capture_depth_m": 0.026,
            "throat_radius_m": 0.0065,
            "nominal_max_acceleration_m_s2": 18.0,
            "swirl_gain": 0.25,
            "status": "task_level_capture_field_pending_pressure_flow_calibration",
        },
        "debridement": {
            "default_cartridge": "soft_brush",
            "cartridges": ["soft_brush", "ring_curette", "microtextured_pad"],
            "extension_m": 0.020,
            "nominal_rotation_speed_rpm": 420,
            "guard_compression_m": 0.008,
            "debris_release_model": "cumulative_contact_work_removes_temporary_deformable_attachment",
            "status": "no_tissue_cutting_or_viability_claim",
        },
        "wound_demo": {
            "surface_vertices": int(len(bundle.wound_surface.vertices)),
            "surface_triangles": int(len(bundle.wound_surface.faces)),
            "debris_fragments": len(bundle.wound_debris),
            "deformable_route": "portable triangular surface cooked at runtime using current PhysX surface-deformable API",
        },
        "clinical_validation": False,
        "medical_device": False,
    }


def collider_coverage(bundle: ToolBundle) -> dict[str, object]:
    links: dict[str, object] = {}
    for name, link in bundle.links.items():
        vmin, vmax = mesh_bounds([v.mesh for v in link.visuals])
        collider_meshes = [collider_mesh(c) for c in link.colliders]
        cmin, cmax = mesh_bounds(collider_meshes)
        visual_extent = np.maximum(vmax - vmin, 1e-9)
        collider_extent = cmax - cmin
        links[name] = {
            "visual_min_m": vmin.tolist(), "visual_max_m": vmax.tolist(),
            "collider_min_m": cmin.tolist(), "collider_max_m": cmax.tolist(),
            "axis_coverage_ratio": (collider_extent / visual_extent).tolist(),
            "deliberate_exclusions": ["thin visual tubing", "camera lenses", "decorative labels", "individual brush bristles"],
        }
    return {"schema": "dr.anmar.collider-coverage.v1", "asset": ASSET_NAME, "version": VERSION, "links": links}

# ---------------------------- Isaac / DrAnmar integration ----------------------------

def author_integration_module() -> str:
    source = r'''# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab integration for the DrAnmar wound-preparation robot.

The payload replaces the Panda hand at ``panda_link8`` and provides a concentric
irrigation / aspiration head, compliant contact guard, rotary debridement
cartridge, multimodal sensor frames, a particle-volume ledger, and task-level
wound-preparation controllers.

All numerical values are provisional engineering seeds. This package is not a
medical device and is not clinically validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import math
import random

CATALOG_SUBPATH = "Props/SurgicalPreparation/WoundPreparationRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ROOT / "dranmar_wound_preparation_tool_payload.usda"
TOOL_STANDALONE_USD = ROOT / "dranmar_wound_preparation_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_wound_preparation_tool_rigid_proxy.usda"
DROPLET_USD = ROOT / "dranmar_irrigation_droplet.usda"
DEBRIS_USD = ROOT / "dranmar_debridement_fragment.usda"
WOUND_BED_USD = ROOT / "dranmar_wound_bed_demo.usda"
BRUSH_CARTRIDGE_USD = ROOT / "dranmar_debridement_brush_cartridge.usda"
CURETTE_CARTRIDGE_USD = ROOT / "dranmar_debridement_curette_cartridge.usda"
PAD_CARTRIDGE_USD = ROOT / "dranmar_debridement_pad_cartridge.usda"

VALID_IRRIGATION_STATES = frozenset({"loaded", "low", "empty"})
VALID_COLLECTION_STATES = frozenset({"empty", "partial", "full"})
IRRIGATION_NOZZLE_COUNT = 10
PARTICLE_RADIUS_M = 0.00065
PARTICLE_VOLUME_ML = 4.0 / 3.0 * math.pi * PARTICLE_RADIUS_M ** 3 * 1.0e6

TOOL_JOINTS = {
    "contact_guard": "contact_guard_joint",
    "debridement_extension": "debridement_extension_joint",
    "debridement_rotor": "debridement_rotor_joint",
    "irrigation_valve": "irrigation_valve_joint",
    "suction_valve": "suction_valve_joint",
}

TOOL_FRAME_PATHS = {
    "panda_link8_mount": "Links/Mount/Frames/panda_link8_mount",
    "wound_preparation_tcp": "Links/Mount/Frames/wound_preparation_tcp",
    "contact_guard_center": "Links/ContactGuard/Frames/contact_guard_center",
    "debridement_contact": "Links/DebridementRotor/Frames/debridement_contact",
    "rotor_axis": "Links/DebridementRotor/Frames/rotor_axis",
    "irrigation_jet_origin": "Links/Mount/Frames/irrigation_jet_origin",
    "suction_capture_center": "Links/Mount/Frames/suction_capture_center",
    "suction_throat": "Links/Mount/Frames/suction_throat",
    "camera_left": "Links/Mount/Frames/camera_left",
    "camera_right": "Links/Mount/Frames/camera_right",
    "depth_camera": "Links/Mount/Frames/depth_camera",
    "fluorescence_camera": "Links/Mount/Frames/fluorescence_camera",
    "illumination_ring": "Links/Mount/Frames/illumination_ring",
    "irrigation_reservoir_port": "Links/IrrigationReservoir/Frames/irrigation_reservoir_port",
    "waste_canister_port": "Links/WasteCanister/Frames/waste_canister_port",
    "cartridge_mount": "Links/DebridementRotor/Frames/cartridge_mount",
    "service_reference": "Links/Mount/Frames/service_reference",
    "count_reference": "Links/Mount/Frames/count_reference",
}
REGISTERED_CAMERA_FRAMES = (
    "camera_left", "camera_right", "depth_camera", "fluorescence_camera",
)


def frame_path(tool_path: str, name: str) -> str:
    try:
        suffix = TOOL_FRAME_PATHS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown wound-preparation frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any):
    """Return the underlying torch tensor for Isaac 6 proxy tensors."""
    return value.torch if hasattr(value, "torch") else value


def _check(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}")
    return value


def make_tool_cfg(
    prim_path: str = "/World/DrAnmarWoundPreparationTool",
    *,
    irrigation_state: str = "loaded",
    collection_state: str = "empty",
    position=(0.0, 0.0, 0.35),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    """Return the standalone wound-preparation tool articulation."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    _check(irrigation_state, VALID_IRRIGATION_STATES, "irrigation_state")
    _check(collection_state, VALID_COLLECTION_STATES, "collection_state")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants={"irrigation_state": irrigation_state, "collection_state": collection_state},
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=orientation_wxyz,
            joint_pos={
                "contact_guard_joint": 0.0,
                "debridement_extension_joint": 0.0,
                "debridement_rotor_joint": 0.0,
                "irrigation_valve_joint": 0.0,
                "suction_valve_joint": 0.0,
            },
        ),
        actuators={
            "contact_guard": ImplicitActuatorCfg(
                joint_names_expr=["contact_guard_joint"], effort_limit_sim=32.0,
                velocity_limit_sim=0.08, stiffness=1100.0, damping=30.0,
            ),
            "debridement_extension": ImplicitActuatorCfg(
                joint_names_expr=["debridement_extension_joint"], effort_limit_sim=45.0,
                velocity_limit_sim=0.10, stiffness=4200.0, damping=115.0,
            ),
            "debridement_rotor": ImplicitActuatorCfg(
                joint_names_expr=["debridement_rotor_joint"], effort_limit_sim=0.32,
                velocity_limit_sim=90.0, stiffness=0.0, damping=0.018,
            ),
            "irrigation_valve": ImplicitActuatorCfg(
                joint_names_expr=["irrigation_valve_joint"], effort_limit_sim=25.0,
                velocity_limit_sim=2.5, stiffness=1800.0, damping=45.0,
            ),
            "suction_valve": ImplicitActuatorCfg(
                joint_names_expr=["suction_valve_joint"], effort_limit_sim=2.0,
                velocity_limit_sim=2.5, stiffness=8.0, damping=0.45,
            ),
        },
    )


def make_rigid_proxy_cfg(
    prim_path: str = "/World/DrAnmarWoundPreparationToolProxy",
    *, position=(0.0, 0.0, 0.35), orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(TOOL_RIGID_PROXY_USD), activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=orientation_wxyz),
    )


def _spawn_single_franka_with_wound_preparation_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    """Spawn Franka, deactivate the Panda hand, and mount the DrAnmar payload."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import create_prim, get_current_stage, select_usd_variants
    from pxr import Gf, Sdf, UsdPhysics

    robot = spawn_from_usd(prim_path, cfg, translation, orientation)
    stage = get_current_stage()
    names_to_disable = {
        "panda_hand_joint", "panda_hand", "panda_finger_joint1", "panda_finger_joint2",
        "panda_leftfinger", "panda_rightfinger",
    }
    robot_path = Sdf.Path(prim_path)
    hand_joint_prims = [
        prim
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() == "panda_hand_joint"
    ]
    if len(hand_joint_prims) == 1:
        stock_hand_joint = UsdPhysics.Joint(hand_joint_prims[0])
        mount_body_paths = stock_hand_joint.GetBody0Rel().GetTargets()
        mount_local_pos0 = stock_hand_joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)
        mount_local_rot0 = stock_hand_joint.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0)
    else:
        link8_paths = [
            prim.GetPath()
            for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(robot_path) and prim.GetName() == "panda_link8"
        ]
        if len(link8_paths) != 1:
            raise RuntimeError(
                "Could not resolve the Franka hand mount from panda_hand_joint or panda_link8"
            )
        mount_body_paths = link8_paths
        mount_local_pos0 = Gf.Vec3f(0, 0, 0)
        half_angle = math.radians(-45.0) / 2.0
        mount_local_rot0 = Gf.Quatf(
            math.cos(half_angle), 0, 0, math.sin(half_angle)
        )
    if len(mount_body_paths) != 1 or not stage.GetPrimAtPath(mount_body_paths[0]).IsValid():
        raise RuntimeError(f"Invalid Franka hand mount target: {mount_body_paths}")

    candidate_paths = [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(robot_path) and prim.GetName() in names_to_disable
    ]
    paths_to_disable = []
    for path in sorted(candidate_paths, key=lambda item: str(item).count("/")):
        if not any(path.HasPrefix(parent) for parent in paths_to_disable):
            paths_to_disable.append(path)
    for path in paths_to_disable:
        stage.OverridePrim(path).SetActive(False)

    tool_path = f"{prim_path}/DrAnmarWoundPreparationTool"
    create_prim(tool_path, usd_path=str(TOOL_PAYLOAD_USD), stage=stage)
    select_usd_variants(
        tool_path,
        {"irrigation_state": cfg.irrigation_state, "collection_state": cfg.collection_state},
    )

    joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/dranmar_wound_preparation_mount_joint")
    joint.CreateBody0Rel().SetTargets(mount_body_paths)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    joint.CreateLocalPos0Attr().Set(mount_local_pos0)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr().Set(mount_local_rot0)
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return robot


def spawn_franka_with_wound_preparation_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_wound_preparation_tool)(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )


def make_franka_wound_preparation_robot_cfg(
    *, prim_path: str = "/World/Robot", irrigation_state: str = "loaded", collection_state: str = "empty"
):
    """Return the Isaac Lab Franka with its stock hand replaced by this tool."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    _check(irrigation_state, VALID_IRRIGATION_STATES, "irrigation_state")
    _check(collection_state, VALID_COLLECTION_STATES, "collection_state")

    @configclass
    class FrankaWoundPreparationUsdCfg(sim_utils.UsdFileCfg):
        irrigation_state: str = "loaded"
        collection_state: str = "empty"
        func = spawn_franka_with_wound_preparation_tool

    cfg = FRANKA_PANDA_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn = FrankaWoundPreparationUsdCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        variants={"Gripper": "Default", "Mesh": "Performance"},
        irrigation_state=irrigation_state,
        collection_state=collection_state,
        activate_contact_sensors=True,
        rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,
        articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props,
    )
    cfg.init_state.joint_pos = {key: value for key, value in cfg.init_state.joint_pos.items() if "finger" not in key}
    cfg.init_state.joint_pos.update({
        "contact_guard_joint": 0.0,
        "debridement_extension_joint": 0.0,
        "debridement_rotor_joint": 0.0,
        "irrigation_valve_joint": 0.0,
        "suction_valve_joint": 0.0,
    })
    cfg.actuators.pop("panda_hand", None)
    cfg.actuators.update({
        "wound_prep_guard": ImplicitActuatorCfg(
            joint_names_expr=["contact_guard_joint"], effort_limit_sim=32.0,
            velocity_limit_sim=0.08, stiffness=1100.0, damping=30.0,
        ),
        "wound_prep_extension": ImplicitActuatorCfg(
            joint_names_expr=["debridement_extension_joint"], effort_limit_sim=45.0,
            velocity_limit_sim=0.10, stiffness=4200.0, damping=115.0,
        ),
        "wound_prep_rotor": ImplicitActuatorCfg(
            joint_names_expr=["debridement_rotor_joint"], effort_limit_sim=0.32,
            velocity_limit_sim=90.0, stiffness=0.0, damping=0.018,
        ),
        "wound_prep_valves": ImplicitActuatorCfg(
            joint_names_expr=[".*_valve_joint"], effort_limit_sim=25.0,
            velocity_limit_sim=2.5, stiffness=1800.0, damping=45.0,
        ),
    })
    return cfg


def spawn_wound_bed_demo(
    prim_path: str = "/World/DrAnmarWoundBed",
    *, translation=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    cfg = sim_utils.UsdFileCfg(usd_path=str(WOUND_BED_USD), activate_contact_sensors=True)
    return cfg.func(prim_path, cfg, translation=translation, orientation=orientation_wxyz)


def _current_stage(stage=None):
    if stage is not None:
        return stage
    import omni.usd
    return omni.usd.get_context().get_stage()


def apply_wound_surface_deformable(
    wound_root_path: str = "/World/DrAnmarWoundBed",
    *, stage=None, material_path: str = "/World/Materials/DrAnmarWoundSurface",
    youngs_modulus_pa: float = 55_000.0, poissons_ratio: float = 0.45,
    surface_thickness_m: float = 0.007, density_kg_m3: float = 1_050.0,
    dynamic_friction: float = 0.58, elasticity_damping: float = 0.16,
    bend_damping: float = 0.14, self_collision: bool = True,
) -> dict[str, Any]:
    """Cook the portable wound mesh through the current surface-deformable route."""
    stage = _current_stage(stage)
    from omni.physx.scripts import deformableUtils
    from pxr import UsdShade

    mesh_path = f"{wound_root_path.rstrip('/')}/TissueSurface/SimulationMesh"
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim or not mesh_prim.IsValid():
        raise ValueError(f"No wound simulation mesh at {mesh_path}")

    material = UsdShade.Material.Define(stage, material_path)
    prim = material.GetPrim()
    prim.ApplyAPI("OmniPhysicsBaseMaterialAPI")
    prim.GetAttribute("omniphysics:dynamicFriction").Set(float(dynamic_friction))
    prim.GetAttribute("omniphysics:density").Set(float(density_kg_m3))
    prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:youngsModulus").Set(float(youngs_modulus_pa))
    prim.GetAttribute("omniphysics:poissonsRatio").Set(float(poissons_ratio))
    prim.ApplyAPI("OmniPhysicsSurfaceDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:surfaceThickness").Set(float(surface_thickness_m))
    prim.GetAttribute("omniphysics:surfaceBendStiffness").Set(0.0)
    prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(float(elasticity_damping))
    prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(float(bend_damping))

    success = deformableUtils.set_physics_surface_deformable_body(stage, mesh_prim.GetPath())
    if success is False:
        raise RuntimeError(f"PhysX could not create a surface deformable at {mesh_path}")
    mesh_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    if mesh_prim.HasAPI("PhysxSurfaceDeformableBodyAPI"):
        mesh_prim.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
    binding = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return {
        "mesh_path": mesh_path, "material_path": material_path,
        "parameters": {
            "youngs_modulus_pa": youngs_modulus_pa, "poissons_ratio": poissons_ratio,
            "surface_thickness_m": surface_thickness_m, "density_kg_m3": density_kg_m3,
            "dynamic_friction": dynamic_friction, "elasticity_damping": elasticity_damping,
            "bend_damping": bend_damping, "self_collision": self_collision,
            "status": "provisional_engineering_seed",
        },
    }


def create_deformable_attachment(
    deformable_prim_path: str, rigid_prim_path: str, attachment_path: str, *, stage=None
) -> str:
    """Create an overlap-generated rigid/deformable attachment across Isaac generations."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    stage = _current_stage(stage)
    if stage.GetPrimAtPath(attachment_path).IsValid():
        stage.RemovePrim(attachment_path)

    # Isaac Sim 6 replaced the command-authored PhysxPhysicsAttachment with
    # explicit OmniPhysics vertex attachments. Author the current schema
    # directly so headless runtimes do not depend on an optional UI command.
    prim_definition = Usd.SchemaRegistry().FindConcretePrimDefinition(
        "OmniPhysicsVtxXformAttachment"
    )
    if prim_definition:
        deformable_prim = stage.GetPrimAtPath(deformable_prim_path)
        rigid_prim = stage.GetPrimAtPath(rigid_prim_path)
        mesh = UsdGeom.Mesh(deformable_prim)
        points = list(mesh.GetPointsAttr().Get() or [])
        if not deformable_prim.IsValid() or not mesh or not points:
            raise ValueError(f"Attachment source is not a populated mesh: {deformable_prim_path}")
        if not rigid_prim.IsValid() or not UsdGeom.Xformable(rigid_prim):
            raise ValueError(f"Attachment target is not xformable: {rigid_prim_path}")

        mesh_to_world = UsdGeom.Xformable(deformable_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        rigid_to_world = UsdGeom.Xformable(rigid_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        world_to_rigid = rigid_to_world.GetInverse()
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
        ).ComputeWorldBound(rigid_prim).ComputeAlignedRange()
        minimum, maximum = bounds.GetMin(), bounds.GetMax()
        center = (minimum + maximum) * 0.5
        margin = 0.0025
        ranked: list[tuple[float, int, Gf.Vec3d, bool]] = []
        for index, point in enumerate(points):
            world = mesh_to_world.Transform(Gf.Vec3d(point))
            delta = world - center
            distance_sq = float(Gf.Dot(delta, delta))
            overlaps = all(
                minimum[axis] - margin <= world[axis] <= maximum[axis] + margin
                for axis in range(3)
            )
            ranked.append((distance_sq, index, world, overlaps))
        ranked.sort(key=lambda item: item[0])
        selected = [item for item in ranked if item[3]][:12]
        if len(selected) < 4:
            raise RuntimeError(
                f"Attachment capture volume does not overlap enough deformable "
                f"vertices for {attachment_path}: source={deformable_prim_path}, "
                f"target={rigid_prim_path}, overlapping={len(selected)}, "
                "required=4, overlap_margin_m=0.0025"
            )

        attachment = stage.DefinePrim(attachment_path, "OmniPhysicsVtxXformAttachment")
        attachment.CreateRelationship("omniphysics:src0").SetTargets(
            [Sdf.Path(deformable_prim_path)]
        )
        attachment.CreateRelationship("omniphysics:src1").SetTargets(
            [Sdf.Path(rigid_prim_path)]
        )
        attachment.CreateAttribute(
            "omniphysics:vtxIndicesSrc0", Sdf.ValueTypeNames.IntArray
        ).Set(Vt.IntArray([item[1] for item in selected]))
        attachment.CreateAttribute(
            "omniphysics:localPositionsSrc1", Sdf.ValueTypeNames.Point3fArray
        ).Set(
            Vt.Vec3fArray(
                [Gf.Vec3f(world_to_rigid.Transform(item[2])) for item in selected]
            )
        )
        attachment.CreateAttribute(
            "omniphysics:attachmentEnabled", Sdf.ValueTypeNames.Bool
        ).Set(True)
        if (
            not attachment.IsValid()
            or attachment.GetTypeName() != "OmniPhysicsVtxXformAttachment"
            or not attachment.GetRelationship("omniphysics:src0").GetTargets()
            or not attachment.GetRelationship("omniphysics:src1").GetTargets()
        ):
            raise RuntimeError(f"Could not author current attachment schema at {attachment_path}")
        return "OmniPhysicsVtxXformAttachment"

    import omni.kit.commands

    def execute_and_verify(command: str, **kwargs) -> str:
        omni.kit.commands.execute(command, **kwargs)
        attachment = stage.GetPrimAtPath(attachment_path)
        if not attachment.IsValid():
            raise RuntimeError(f"{command} did not author {attachment_path}")
        return command

    try:
        return execute_and_verify(
            "CreateAutoDeformableAttachment",
            target_attachment_path=Sdf.Path(attachment_path),
            attachable0_path=Sdf.Path(deformable_prim_path),
            attachable1_path=Sdf.Path(rigid_prim_path),
        )
    except Exception as current_error:
        if stage.GetPrimAtPath(attachment_path).IsValid():
            stage.RemovePrim(attachment_path)
        try:
            return execute_and_verify(
                "CreatePhysicsAttachment",
                target_attachment_path=Sdf.Path(attachment_path),
                actor0_path=Sdf.Path(deformable_prim_path),
                actor1_path=Sdf.Path(rigid_prim_path),
            )
        except Exception as legacy_error:
            raise RuntimeError(
                f"Could not create attachment {attachment_path}: current={current_error!r}; legacy={legacy_error!r}"
            ) from legacy_error


def attach_demo_debris(
    wound_root_path: str = "/World/DrAnmarWoundBed", *, stage=None
) -> dict[str, str]:
    """Attach every demo fragment to the wound mesh until debridement releases it."""
    stage = _current_stage(stage)
    tissue_path = f"{wound_root_path.rstrip('/')}/TissueSurface/SimulationMesh"
    attachments_root = f"{wound_root_path.rstrip('/')}/RuntimeDebrisAttachments"
    stage.DefinePrim(attachments_root, "Scope")
    result: dict[str, str] = {}
    debris_scope = stage.GetPrimAtPath(f"{wound_root_path.rstrip('/')}/Debris")
    if not debris_scope or not debris_scope.IsValid():
        raise ValueError("Demo debris scope is missing")
    for fragment in debris_scope.GetChildren():
        collider = f"{fragment.GetPath()}/Collisions/AdhesionPatch"
        attachment = f"{attachments_root}/{fragment.GetName()}"
        create_deformable_attachment(tissue_path, collider, attachment, stage=stage)
        result[str(fragment.GetPath())] = attachment
    return result


def _world_transform(stage, prim_path: str):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Invalid frame prim {prim_path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _nonnegative_finite(value: float, label: str) -> float:
    amount = float(value)
    if not math.isfinite(amount) or amount < 0.0:
        raise ValueError(f"{label} must be a finite non-negative value")
    return amount


@dataclass
class FluidLedger:
    """Conservative volume bookkeeping around the particle approximation."""
    reservoir_capacity_ml: float = 45.0
    reservoir_ml: float = 45.0
    collection_capacity_ml: float = 55.0
    emitted_ml: float = 0.0
    aspirated_ml: float = 0.0
    spilled_ml: float = 0.0
    discarded_ml: float = 0.0
    active_particle_ml: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "reservoir_capacity_ml", "reservoir_ml", "collection_capacity_ml",
            "emitted_ml", "aspirated_ml", "spilled_ml", "discarded_ml",
            "active_particle_ml",
        ):
            setattr(self, name, _nonnegative_finite(getattr(self, name), name))
        if self.reservoir_ml > self.reservoir_capacity_ml:
            raise ValueError("reservoir_ml cannot exceed reservoir_capacity_ml")
        if self.aspirated_ml > self.collection_capacity_ml:
            raise ValueError("aspirated_ml cannot exceed collection_capacity_ml")

    def emit(self, requested_ml: float) -> float:
        amount = min(_nonnegative_finite(requested_ml, "requested_ml"), self.reservoir_ml)
        self.reservoir_ml -= amount
        self.emitted_ml += amount
        self.active_particle_ml += amount
        return amount

    def aspirate(self, amount_ml: float) -> float:
        amount = min(
            _nonnegative_finite(amount_ml, "amount_ml"),
            self.active_particle_ml,
            self.collection_remaining_ml,
        )
        self.active_particle_ml -= amount
        self.aspirated_ml += amount
        return amount

    def mark_spilled(self, amount_ml: float) -> float:
        amount = min(_nonnegative_finite(amount_ml, "amount_ml"), self.active_particle_ml)
        self.active_particle_ml -= amount
        self.spilled_ml += amount
        return amount

    def discard(self, amount_ml: float) -> float:
        """Account for particles intentionally culled for numerical maintenance."""
        amount = min(_nonnegative_finite(amount_ml, "amount_ml"), self.active_particle_ml)
        self.active_particle_ml -= amount
        self.discarded_ml += amount
        return amount

    @property
    def collection_remaining_ml(self) -> float:
        return max(0.0, self.collection_capacity_ml - self.aspirated_ml)

    @property
    def balance_error_ml(self) -> float:
        accounted = self.reservoir_ml + self.active_particle_ml + self.aspirated_ml + self.spilled_ml + self.discarded_ml
        return self.reservoir_capacity_ml - accounted

    def snapshot(self) -> dict[str, float]:
        return {
            "reservoir_capacity_ml": self.reservoir_capacity_ml,
            "reservoir_ml": self.reservoir_ml,
            "emitted_ml": self.emitted_ml,
            "active_particle_ml": self.active_particle_ml,
            "aspirated_ml": self.aspirated_ml,
            "spilled_ml": self.spilled_ml,
            "discarded_ml": self.discarded_ml,
            "collection_capacity_ml": self.collection_capacity_ml,
            "collection_remaining_ml": self.collection_remaining_ml,
            "balance_error_ml": self.balance_error_ml,
        }


def ensure_irrigation_particle_system(
    *, stage=None, physics_scene_path: str = "/physicsScene",
    root_path: str = "/World/DrAnmarIrrigationParticles",
    particle_system_path: str | None = None, particle_set_path: str | None = None,
    particle_radius_m: float = PARTICLE_RADIUS_M,
) -> dict[str, str]:
    """Create a PhysX PBD liquid system and an initially empty particle set."""
    stage = _current_stage(stage)
    from omni.physx.scripts import particleUtils, physicsUtils
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage.DefinePrim(root_path, "Scope")
    if not stage.GetPrimAtPath(physics_scene_path).IsValid():
        UsdPhysics.Scene.Define(stage, physics_scene_path)
    particle_system_path = particle_system_path or f"{root_path}/ParticleSystem"
    particle_set_path = particle_set_path or f"{root_path}/Particles"
    material_path = f"{root_path}/PBDMaterial"

    if not stage.GetPrimAtPath(material_path).IsValid():
        particleUtils.add_pbd_particle_material(
            stage, Sdf.Path(material_path), cohesion=0.002, viscosity=0.002,
            surface_tension=0.004, friction=0.05,
        )
    if not stage.GetPrimAtPath(particle_system_path).IsValid():
        particleUtils.add_physx_particle_system(
            stage=stage, particle_system_path=Sdf.Path(particle_system_path),
            simulation_owner=Sdf.Path(physics_scene_path),
            particle_contact_offset=particle_radius_m * 1.15,
            rest_offset=particle_radius_m * 0.90,
            solid_rest_offset=particle_radius_m * 1.80,
            fluid_rest_offset=particle_radius_m * 0.92,
        )
        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(particle_system_path), Sdf.Path(material_path)
        )
    if not stage.GetPrimAtPath(particle_set_path).IsValid():
        particleUtils.add_physx_particleset_points(
            stage, Sdf.Path(particle_set_path), [], [], [], Sdf.Path(particle_system_path),
            True, True, 0, 1.0, particle_radius_m * 2.0,
        )
        points = UsdGeom.Points(stage.GetPrimAtPath(particle_set_path))
        points.GetWidthsAttr().Set([])
    return {
        "root_path": root_path, "material_path": material_path,
        "particle_system_path": particle_system_path, "particle_set_path": particle_set_path,
    }


def emit_irrigation_burst(
    tool_path: str, ledger: FluidLedger, *, requested_ml: float = 0.25,
    jet_speed_m_s: float = 1.20, launch_spread_deg: float = 1.5,
    random_seed: int | None = 0, stage=None,
    particle_set_path: str = "/World/DrAnmarIrrigationParticles/Particles",
) -> dict[str, Any]:
    """Append a multi-nozzle PBD particle burst and debit its exact particle volume."""
    stage = _current_stage(stage)
    from pxr import Gf, UsdGeom, Vt

    points = UsdGeom.Points(stage.GetPrimAtPath(particle_set_path))
    if not points:
        raise ValueError(f"No irrigation particle set at {particle_set_path}")
    jet_speed_m_s = _nonnegative_finite(jet_speed_m_s, "jet_speed_m_s")
    launch_spread_deg = _nonnegative_finite(launch_spread_deg, "launch_spread_deg")
    available = ledger.emit(requested_ml)
    requested_particles = int(available / PARTICLE_VOLUME_ML)
    count = max(0, requested_particles - requested_particles % IRRIGATION_NOZZLE_COUNT)
    actual_ml = count * PARTICLE_VOLUME_ML
    # Return non-emitted quantization remainder to the reservoir.
    remainder = available - actual_ml
    ledger.reservoir_ml += remainder
    ledger.emitted_ml -= remainder
    ledger.active_particle_ml -= remainder
    if count == 0:
        return {"particle_count": 0, "emitted_ml": 0.0, "quantization_remainder_ml": remainder}

    transform = _world_transform(stage, frame_path(tool_path, "irrigation_jet_origin"))
    current_positions = list(points.GetPointsAttr().Get() or [])
    current_velocities = list(points.GetVelocitiesAttr().Get() or [])
    current_widths = list(points.GetWidthsAttr().Get() or [])
    if len(current_velocities) < len(current_positions):
        current_velocities.extend(
            Gf.Vec3f(0.0) for _ in range(len(current_positions) - len(current_velocities))
        )
    if len(current_widths) < len(current_positions):
        current_widths.extend(
            PARTICLE_RADIUS_M * 2.0
            for _ in range(len(current_positions) - len(current_widths))
        )
    current_velocities = current_velocities[:len(current_positions)]
    current_widths = current_widths[:len(current_positions)]
    rng = random.Random(random_seed)
    spread = math.tan(math.radians(launch_spread_deg))
    per_nozzle = count // IRRIGATION_NOZZLE_COUNT
    for nozzle in range(IRRIGATION_NOZZLE_COUNT):
        angle = 2.0 * math.pi * nozzle / IRRIGATION_NOZZLE_COUNT
        local_origin = Gf.Vec3d(0.0062 * math.cos(angle), 0.0062 * math.sin(angle), 0.0030)
        base_direction = Gf.Vec3d(
            -0.0044 * math.cos(angle), -0.0044 * math.sin(angle), 0.0140
        ).GetNormalized()
        tangent = Gf.Vec3d(-math.sin(angle), math.cos(angle), 0.0)
        radial = Gf.Vec3d(math.cos(angle), math.sin(angle), 0.0)
        world_origin = transform.Transform(local_origin)
        for index in range(per_nozzle):
            perturbed = (
                base_direction
                + tangent * (rng.uniform(-1.0, 1.0) * spread)
                + radial * (rng.uniform(-1.0, 1.0) * spread)
            ).GetNormalized()
            world_direction = transform.TransformDir(perturbed).GetNormalized()
            axial_jitter = (index % 5 - 2) * 0.00012
            position = world_origin + world_direction * axial_jitter
            current_positions.append(Gf.Vec3f(position))
            current_velocities.append(Gf.Vec3f(world_direction * jet_speed_m_s))
            current_widths.append(PARTICLE_RADIUS_M * 2.0)
    points.GetPointsAttr().Set(Vt.Vec3fArray(current_positions))
    points.GetVelocitiesAttr().Set(Vt.Vec3fArray(current_velocities))
    points.GetWidthsAttr().Set(current_widths)
    return {
        "particle_count": count, "emitted_ml": actual_ml,
        "particle_volume_ml": PARTICLE_VOLUME_ML,
        "launch_spread_deg": launch_spread_deg,
        "random_seed": random_seed,
        "quantization_remainder_ml": remainder,
        "particle_set_path": particle_set_path,
    }


@dataclass
class SuctionFieldController:
    capture_radius_m: float = 0.023
    capture_depth_m: float = 0.030
    throat_radius_m: float = 0.0065
    max_acceleration_m_s2: float = 18.0
    swirl_gain: float = 0.25

    def update_particles(
        self, tool_path: str, ledger: FluidLedger, *, dt: float, opening: float = 1.0,
        stage=None, particle_set_path: str = "/World/DrAnmarIrrigationParticles/Particles",
    ) -> dict[str, Any]:
        """Apply a converging suction field and remove particles entering the throat."""
        stage = _current_stage(stage)
        from pxr import Gf, UsdGeom, Vt
        dt = _nonnegative_finite(dt, "dt")
        points = UsdGeom.Points(stage.GetPrimAtPath(particle_set_path))
        positions = list(points.GetPointsAttr().Get() or [])
        velocities = list(points.GetVelocitiesAttr().Get() or [])
        if not positions:
            return {"active": 0, "captured": 0, "aspirated_ml": 0.0}
        opening = max(0.0, min(1.0, float(opening)))
        capture_T = _world_transform(stage, frame_path(tool_path, "suction_capture_center"))
        throat_T = _world_transform(stage, frame_path(tool_path, "suction_throat"))
        inverse_capture = capture_T.GetInverse()
        throat_world = throat_T.ExtractTranslation()
        kept_positions, kept_velocities, kept_widths = [], [], []
        widths = list(points.GetWidthsAttr().Get() or [PARTICLE_RADIUS_M * 2] * len(positions))
        if len(velocities) < len(positions):
            velocities.extend(Gf.Vec3f(0.0) for _ in range(len(positions) - len(velocities)))
        if len(widths) < len(positions):
            widths.extend(PARTICLE_RADIUS_M * 2.0 for _ in range(len(positions) - len(widths)))
        velocities = velocities[:len(positions)]
        widths = widths[:len(positions)]
        capture_budget = min(
            int((ledger.collection_remaining_ml + 1.0e-12) / PARTICLE_VOLUME_ML),
            int((ledger.active_particle_ml + 1.0e-12) / PARTICLE_VOLUME_ML),
        )
        captured = 0
        capture_blocked = 0
        for position, velocity, width in zip(positions, velocities, widths):
            world = Gf.Vec3d(position)
            local = inverse_capture.Transform(world)
            radial = math.hypot(local[0], local[1])
            in_capture = radial <= self.capture_radius_m and abs(local[2]) <= self.capture_depth_m * 0.75
            to_throat = throat_world - world
            distance = max(float(to_throat.GetLength()), 1.0e-8)
            if opening > 0 and distance <= self.throat_radius_m:
                if captured < capture_budget:
                    captured += 1
                    continue
                capture_blocked += 1
            new_velocity = Gf.Vec3d(velocity)
            if in_capture and opening > 0:
                direction = to_throat / distance
                swirl = Gf.Vec3d(-local[1], local[0], 0)
                if swirl.GetLength() > 1.0e-9:
                    swirl = capture_T.TransformDir(swirl.GetNormalized())
                gain = opening * self.max_acceleration_m_s2 * max(0.15, 1.0 - radial / self.capture_radius_m)
                new_velocity += (direction + swirl * self.swirl_gain) * gain * dt
            kept_positions.append(Gf.Vec3f(world))
            kept_velocities.append(Gf.Vec3f(new_velocity))
            kept_widths.append(float(width))
        points.GetPointsAttr().Set(Vt.Vec3fArray(kept_positions))
        points.GetVelocitiesAttr().Set(Vt.Vec3fArray(kept_velocities))
        points.GetWidthsAttr().Set(kept_widths)
        aspirated_ml = ledger.aspirate(captured * PARTICLE_VOLUME_ML)
        expected_ml = captured * PARTICLE_VOLUME_ML
        if not math.isclose(aspirated_ml, expected_ml, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise RuntimeError("particle capture and fluid ledger diverged")
        return {
            "active": len(kept_positions),
            "captured": captured,
            "capture_blocked": capture_blocked,
            "aspirated_ml": aspirated_ml,
        }

    def update_rigid_debris(
        self, tool_path: str, debris_paths: Iterable[str], *, dt: float,
        opening: float = 1.0, stage=None,
    ) -> dict[str, list[str]]:
        """Steer released rigid fragments toward the throat and remove captured ones."""
        stage = _current_stage(stage)
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        dt = _nonnegative_finite(dt, "dt")
        opening = max(0.0, min(1.0, _nonnegative_finite(opening, "opening")))
        capture_T = _world_transform(stage, frame_path(tool_path, "suction_capture_center"))
        inverse_capture = capture_T.GetInverse()
        throat_world = _world_transform(stage, frame_path(tool_path, "suction_throat")).ExtractTranslation()
        removed, accelerated = [], []
        for path in debris_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            position = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            local = inverse_capture.Transform(position)
            radial = math.hypot(local[0], local[1])
            to_throat = throat_world - position
            distance = max(float(to_throat.GetLength()), 1.0e-8)
            if opening > 0 and distance <= self.throat_radius_m:
                stage.RemovePrim(path)
                removed.append(path)
                continue
            if radial <= self.capture_radius_m and abs(local[2]) <= self.capture_depth_m and opening > 0:
                velocity_attr = UsdPhysics.RigidBodyAPI(prim).GetVelocityAttr()
                current = Gf.Vec3f(velocity_attr.Get() or Gf.Vec3f(0))
                direction = to_throat / distance
                velocity_attr.Set(current + Gf.Vec3f(direction * (opening * self.max_acceleration_m_s2 * dt)))
                accelerated.append(path)
        return {"removed": removed, "accelerated": accelerated}


@dataclass
class DebrisBond:
    debris_path: str
    attachment_path: str
    threshold_j: float
    accumulated_work_j: float = 0.0
    released: bool = False


@dataclass
class DebridementReleaseController:
    """Release attached debris after cumulative brush/curette contact work."""
    bonds: dict[str, DebrisBond] = field(default_factory=dict)

    def register_demo(self, attachments: Mapping[str, str], *, stage=None) -> None:
        stage = _current_stage(stage)
        for debris_path, attachment_path in attachments.items():
            prim = stage.GetPrimAtPath(debris_path)
            if not prim or not prim.IsValid():
                raise ValueError(f"Invalid debris prim {debris_path}")
            threshold_attr = prim.GetAttribute("drAnmar:adhesionWorkThresholdJ")
            threshold_value = threshold_attr.Get()
            threshold = _nonnegative_finite(
                0.006 if threshold_value is None else threshold_value,
                "adhesion_work_threshold_j",
            )
            if threshold <= 0.0:
                raise ValueError("adhesion work threshold must be greater than zero")
            self.bonds[debris_path] = DebrisBond(debris_path, attachment_path, threshold)

    def update(
        self, contact_forces_n: Mapping[str, float], tangential_speeds_m_s: Mapping[str, float],
        *, dt: float, stage=None,
    ) -> list[str]:
        stage = _current_stage(stage)
        dt = _nonnegative_finite(dt, "dt")
        released: list[str] = []
        for path, bond in self.bonds.items():
            if bond.released:
                continue
            force = _nonnegative_finite(contact_forces_n.get(path, 0.0), "contact_force_n")
            speed = _nonnegative_finite(tangential_speeds_m_s.get(path, 0.0), "tangential_speed_m_s")
            bond.accumulated_work_j += force * speed * dt
            if bond.accumulated_work_j >= bond.threshold_j:
                if stage.GetPrimAtPath(bond.attachment_path).IsValid():
                    stage.RemovePrim(bond.attachment_path)
                bond.released = True
                released.append(path)
        return released

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            path: {
                "threshold_j": bond.threshold_j,
                "accumulated_work_j": bond.accumulated_work_j,
                "released": bond.released,
                "attachment_path": bond.attachment_path,
            }
            for path, bond in self.bonds.items()
        }


def phase_targets(phase: str) -> dict[str, float]:
    """Return joint targets for the canonical wound-preparation sequence."""
    phases = {
        "inspect": {
            "contact_guard_joint": 0.0, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.0,
            "suction_valve_joint": 0.0,
        },
        "contact": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.0,
            "suction_valve_joint": 0.0,
        },
        "pre_rinse": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.006,
            "suction_valve_joint": math.radians(45.0),
        },
        "aspirate": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.0,
            "suction_valve_joint": math.radians(80.0),
        },
        "debride": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.018,
            "debridement_rotor_joint_velocity": math.radians(2520.0),
            "irrigation_valve_joint": 0.002, "suction_valve_joint": math.radians(65.0),
        },
        "post_rinse": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.006,
            "suction_valve_joint": math.radians(75.0),
        },
        "dry": {
            "contact_guard_joint": 0.006, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.0,
            "suction_valve_joint": math.radians(85.0),
        },
        "verify": {
            "contact_guard_joint": 0.0, "debridement_extension_joint": 0.0,
            "debridement_rotor_joint_velocity": 0.0, "irrigation_valve_joint": 0.0,
            "suction_valve_joint": 0.0,
        },
    }
    try:
        return dict(phases[phase])
    except KeyError as exc:
        raise KeyError(f"Unknown wound-preparation phase {phase!r}; expected one of {sorted(phases)}") from exc


@dataclass
class WoundPreparationSequenceController:
    tool_path: str
    wound_root_path: str
    ledger: FluidLedger = field(default_factory=FluidLedger)
    suction: SuctionFieldController = field(default_factory=SuctionFieldController)
    debridement: DebridementReleaseController = field(default_factory=DebridementReleaseController)
    phase: str = "inspect"
    history: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, phase: str) -> dict[str, float]:
        targets = phase_targets(phase)
        self.phase = phase
        self.history.append({"phase": phase, "targets": dict(targets), "fluid": self.ledger.snapshot()})
        return targets

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "tool_path": self.tool_path,
            "wound_root_path": self.wound_root_path,
            "fluid": self.ledger.snapshot(),
            "debridement": self.debridement.snapshot(),
            "history": list(self.history),
            "status": "simulation_training_workcell",
        }
'''
    return source.replace("@VERSION@", VERSION)

# ---------------------------- Documentation and packaging ----------------------------

def readme() -> str:
    return f'''# {ASSET_NAME}

Version: {VERSION}

A Franka-compatible, manufacturer-neutral wound-preparation end effector for
robotic inspection, irrigation, aspiration, controlled debridement, debris
removal, and fluid-accounting research.

## Core design

The tool uses a single concentric work head:

- ten inward-converging irrigation microjets;
- a twelve-slot annular suction crown;
- an extendable rotary debridement cartridge;
- a compliant force-sensing guard ring;
- stereo RGB, depth and fluorescence sensor frames;
- separate irrigation and collection inventories;
- direct replacement of the Panda hand at `panda_link8`.

The shared TCP allows irrigation, suction, debridement and inspection without a
tool exchange or change in robot kinematic reference.

## Catalog path

```text
Props/SurgicalPreparation/WoundPreparationRobot/
```

## Main assets

```text
dranmar_wound_preparation_tool_payload.usda
dranmar_wound_preparation_tool_standalone.usda
dranmar_wound_preparation_tool_rigid_proxy.usda
dranmar_irrigation_droplet.usda
dranmar_debridement_fragment.usda
dranmar_wound_bed_demo.usda
dranmar_debridement_brush_cartridge.usda
dranmar_debridement_curette_cartridge.usda
dranmar_debridement_pad_cartridge.usda
```

## Runtime scope

The supplied integration helper supports:

- standalone Isaac Lab articulation configuration;
- combined Franka + payload configuration;
- current PhysX surface-deformable cooking for the wound bed;
- PBD fluid particle creation and multi-nozzle emission;
- conservative fluid-volume bookkeeping;
- annular suction forces and particle capture;
- temporary debris-to-wound attachments;
- cumulative-work debris release;
- canonical wound-preparation phase targets.

The fluid implementation is a particle-scale task model, not CFD. The
debridement implementation releases discrete debris fragments and does not
claim tissue viability classification, biological cutting, bacterial reduction,
or clinical efficacy.

## Validation

Run the dependency-free package validator and controller tests before promotion:

```bash
python3 scripts/validate_dranmar_wound_preparation_robot.py --require-usdchecker
python3 -m unittest -v tests/test_wound_preparation_robot.py
```

The validator checks OpenUSD parsing, mirrored asset integrity, manifest hashes,
JSON contracts, GLB/PNG containers, Python syntax, release inventory, and
non-portable build-artifact exclusion. The optional Isaac Sim/PhysX CUDA script
is a diagnostic smoke only; `docs/VALIDATION.md` defines the physical effects
that remain unqualified.

## Deterministic regeneration

Install the pinned dependency families and run the generator from any Python
3.10-or-newer environment:

```bash
python3 -m pip install -r scripts/requirements_wound_preparation_generation.txt
python3 scripts/generate_dranmar_wound_preparation_robot.py
```

The generator cleans only its owned output paths, mirrors the catalog into the
extension data tree, writes fixed-timestamp ZIP members, and excludes bytecode
and workstation metadata from manifests and archives.

## Evidence boundary

This simulation-training workcell is available for task execution and
evaluation. Real-world and clinical evidence are not established. All
unmeasured mechanical, fluid, tissue, and contact values remain disclosed
engineering parameters.
'''


def docs_mechanism() -> str:
    return '''# Mechanism

The end effector mounts to `panda_link8` and replaces the stock Panda hand.

## Concentric work head

The central spindle carries the debridement cartridge. Ten irrigation nozzles
surround the spindle and converge toward the work axis. Twelve suction slots
form an outer annulus, reducing the need to reorient the arm between fluid
delivery and recovery.

## Moving links

- `ContactGuard`: spring-driven 8 mm prismatic compression range.
- `DebridementCarriage`: 20 mm tool extension range.
- `DebridementRotor`: continuous revolute spindle.
- `IrrigationValve`: 6 mm metering-spool travel.
- `SuctionValve`: 0–85 degree valve opening.

The guard establishes standoff and provides a contact frame before the brush or
curette reaches the wound bed. The default cartridge is the soft brush. Separate
ring-curette and microtextured-pad assets use the same cartridge frame.
'''


def docs_fluid() -> str:
    return '''# Irrigation and aspiration model

The runtime creates a PhysX PBD liquid particle system and emits particles from
ten nozzle origins. Every particle has a declared volume. `FluidLedger` debits
the reservoir only for particles actually authored and tracks:

- remaining reservoir volume;
- emitted volume;
- active particle volume;
- aspirated volume;
- spilled volume;
- discarded volume;
- balance error.

The emitter uses a deterministic seed by default to add bounded per-particle
directional spread around each authored jet direction. The suction controller
applies a converging field within the annular capture volume, adds a tangential
component to guide particles toward the throat, and removes particles that cross
the throat radius only while the collection canister has volume available.
Captured particle volume is credited to the collection canister. Explicit
simulation culling must be transferred through `FluidLedger.discard()` so the
balance remains auditable.

This provides deterministic volume accounting and visible liquid interaction.
It is not a Navier–Stokes solver, pressure-flow calibration, aerosol model, or
clinical irrigation-dose model.
'''


def docs_debridement() -> str:
    return '''# Debridement model

The demo wound bed contains separate rigid debris fragments. At runtime each
fragment can be attached to the deformable wound surface. The
`DebridementReleaseController` integrates contact work:

```text
work increment = normal force × tangential speed × timestep
```

When cumulative work exceeds a fragment-specific provisional threshold, the
temporary attachment is removed. The released fragment can then be moved by
contact or captured by the annular suction field.

This creates a physical sequence of adhered debris, mechanical mobilization,
release, and aspiration. It does not simulate living-tissue excision, bleeding,
necrosis, bacterial load, tissue viability, or clinical debridement efficacy.
'''


def docs_franka() -> str:
    return '''# Franka integration

`make_franka_wound_preparation_robot_cfg()` starts with Isaac Lab's Franka
configuration, selects the composable Franka USD, deactivates the stock Panda
hand and finger prims, references the DrAnmar payload, and fixes the payload
`Mount` link to `panda_link8`.

The payload deliberately omits `PhysicsArticulationRootAPI`; the Franka root
owns the combined articulation. The standalone asset includes its own
articulation root for isolated mechanism development.

The integration preserves the standard −45 degree hand mounting rotation and
adds five tool-joint actuator groups to the host articulation.
'''


def example_scene() -> str:
    return r'''#!/usr/bin/env python3
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
from isaaclab.utils.configclass import configclass

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
'''


def author_installer() -> str:
    return r'''#!/usr/bin/env python3
"""Install the DrAnmar wound-preparation package into a DrAnmar checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COPY_ROOTS = ("source", "physics_next", "docs", "examples", "tests")
SCRIPT_NAMES = (
    "generate_dranmar_wound_preparation_robot.py",
    "requirements_wound_preparation_generation.txt",
    "validate_dranmar_wound_preparation_robot.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_plan(repository: Path) -> list[tuple[Path, Path]]:
    plan: list[tuple[Path, Path]] = []
    for top in COPY_ROOTS:
        source_root = PACKAGE_ROOT / top
        if not source_root.exists():
            continue
        for source in sorted(source_root.rglob("*")):
            if (
                source.is_file()
                and "__pycache__" not in source.parts
                and source.suffix != ".pyc"
                and source.name != ".DS_Store"
            ):
                plan.append((source, repository / source.relative_to(PACKAGE_ROOT)))
    for name in SCRIPT_NAMES:
        source = PACKAGE_ROOT / "scripts" / name
        if source.is_file():
            plan.append((source, repository / "scripts" / name))
    return plan


def validate_repository(repository: Path) -> None:
    required = (
        repository / "pyproject.toml",
        repository / "source/extensions/orbit.surgical.assets",
        repository / "physics_next/dr-anmar-assets.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Target is not a compatible DrAnmar checkout; missing: " + ", ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the validated wound-preparation overlay into DrAnmar."
    )
    parser.add_argument("repository", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace differing files already present at overlay-owned paths",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the copy plan without modifying the checkout",
    )
    parser.add_argument(
        "--skip-init-export",
        action="store_true",
        help="do not add the convenience import to the assets package __init__.py",
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    validate_repository(repository)
    plan = copy_plan(repository)
    conflicts = [
        str(target.relative_to(repository))
        for source, target in plan
        if target.exists() and sha256(source) != sha256(target)
    ]
    if conflicts and not args.force:
        raise SystemExit(
            "Refusing to overwrite differing files; review them or rerun with --force:\n"
            + "\n".join(f"  {path}" for path in conflicts)
        )

    summary = {
        "repository": str(repository),
        "files_planned": len(plan),
        "differing_existing_files": conflicts,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    for source, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    init_path = repository / "source/extensions/orbit.surgical.assets/orbit/surgical/assets/__init__.py"
    if not args.skip_init_export:
        text = init_path.read_text(encoding="utf-8")
        line = "from .wound_preparation_robot import *"
        if line not in text:
            init_path.write_text(text.rstrip() + "\n" + line + "\n", encoding="utf-8")

    portfolio = repository / "physics_next/dr-anmar-assets.json"
    data = json.loads(portfolio.read_text(encoding="utf-8"))
    assets = data.setdefault("assets", [])
    entry = {
        "id": "dranmar-wound-preparation-robot-v1",
        "asset": "source/extensions/orbit.surgical.assets/data/Props/SurgicalPreparation/WoundPreparationRobot/dranmar_wound_preparation_tool_standalone.usda",
        "payload_asset": "source/extensions/orbit.surgical.assets/data/Props/SurgicalPreparation/WoundPreparationRobot/dranmar_wound_preparation_tool_payload.usda",
        "auxiliary_asset": "source/extensions/orbit.surgical.assets/data/Props/SurgicalPreparation/WoundPreparationRobot/dranmar_wound_bed_demo.usda",
        "profile": "physics_next/surgical-preparation/dranmar-wound-preparation-robot-v1.json",
        "interaction_frames": "source/extensions/orbit.surgical.assets/data/Props/SurgicalPreparation/WoundPreparationRobot/interaction_frames.json",
        "task_contract": "source/extensions/orbit.surgical.assets/data/Props/SurgicalPreparation/WoundPreparationRobot/wound_preparation_task_contract.json",
        "report": "source/extensions/orbit.surgical.assets/data/Props/SurgicalPreparation/WoundPreparationRobot/asset_manifest.json",
        "live_integration": "franka_panda_link8_replacement_with_irrigation_aspiration_debridement_and_inspection",
        "deployment": "enabled_as_training_workcell",
        "product_capability": "executable_training_workcell",
        "training_readiness": "available_for_simulation_training_data_generation_and_evaluation",
        "software_evidence": "repository_verified_asset_task_and_controller_contracts",
        "native_simulator_evidence": "native_cuda_execution_not_yet_recorded",
        "real_world_evidence": "instrumented_wound_preparation_bench_evidence_not_yet_established",
        "clinical_validation": False,
    }
    assets[:] = [item for item in assets if item.get("id") != entry["id"]]
    assets.append(entry)
    portfolio.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    summary["installed"] = True
    summary["init_export_added"] = not args.skip_init_export
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
'''


def write_asset_files(bundle: ToolBundle) -> list[Path]:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    payload_path = ASSET_ROOT / "dranmar_wound_preparation_tool_payload.usda"
    payload_path.write_text(tool_usda(bundle, articulation_root=False), encoding="utf-8")
    paths.append(payload_path)
    standalone_path = ASSET_ROOT / "dranmar_wound_preparation_tool_standalone.usda"
    standalone_path.write_text(tool_usda(bundle, articulation_root=True), encoding="utf-8")
    paths.append(standalone_path)
    proxy_path = ASSET_ROOT / "dranmar_wound_preparation_tool_rigid_proxy.usda"
    proxy_path.write_text(rigid_proxy_usda(bundle), encoding="utf-8")
    paths.append(proxy_path)

    droplet_path = ASSET_ROOT / "dranmar_irrigation_droplet.usda"
    droplet_path.write_text(
        simple_rigid_asset_usda(
            DROPLET_ROOT,
            Visual("Visual", bundle.droplet, "WaterVisual", ("irrigation_droplet",)),
            Collider("Collision", "capsule", (0, 0, 0), radius=0.00065, height=0.0, physics_material="WaterPhysics"),
            1.15e-6,
            ["irrigation_droplet", "fluid_particle_proxy"],
            {"center": {"position": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0], "parent_link": DROPLET_ROOT, "role": "particle_center"}},
            "Rigid irrigation-droplet fallback for simulation training.",
        ),
        encoding="utf-8",
    )
    paths.append(droplet_path)

    debris_bounds = bundle.debris.bounds
    debris_size = np.maximum(debris_bounds[1] - debris_bounds[0], 0.001)
    debris_center = (debris_bounds[0] + debris_bounds[1]) / 2
    debris_path = ASSET_ROOT / "dranmar_debridement_fragment.usda"
    debris_path.write_text(
        simple_rigid_asset_usda(
            DEBRIS_ROOT,
            Visual("Visual", bundle.debris, "DebrisVisual", ("debridement_fragment",)),
            Collider("AdhesionPatch", "box", tuple(debris_center), size=tuple(debris_size * np.asarray((0.85, 0.85, 0.60))), physics_material="DebrisPhysics", role="debris_attachment_region"),
            0.00055,
            ["debridement_fragment", "aspiration_target"],
            {
                "attachment_reference": {"position": debris_center.tolist(), "orientation_wxyz": [1, 0, 0, 0], "parent_link": DEBRIS_ROOT, "role": "debris_attachment"},
                "count_reference": {"position": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0], "parent_link": DEBRIS_ROOT, "role": "inventory_reference"},
            },
            "Detachable wound-debris fragment for simulation training.",
        ),
        encoding="utf-8",
    )
    paths.append(debris_path)

    cartridge_specs = [
        ("dranmar_debridement_brush_cartridge.usda", BRUSH_ROOT, bundle.brush_cartridge, "BrushBristle", "brush", "soft_debridement_brush", 0.008),
        ("dranmar_debridement_curette_cartridge.usda", CURETTE_ROOT, bundle.curette_cartridge, "CuretteSteel", "curette", "ring_curette", 0.011),
        ("dranmar_debridement_pad_cartridge.usda", PAD_ROOT, bundle.pad_cartridge, "PadFoam", "pad", "microtextured_debridement_pad", 0.009),
    ]
    for filename, root, mesh, material, collider_kind, role, mass in cartridge_specs:
        path = ASSET_ROOT / filename
        path.write_text(cartridge_usda(root, mesh, material, collider_kind, role, mass), encoding="utf-8")
        paths.append(path)

    wound_path = ASSET_ROOT / "dranmar_wound_bed_demo.usda"
    wound_path.write_text(wound_bed_usda(bundle), encoding="utf-8")
    paths.append(wound_path)
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def all_payload_files() -> list[Path]:
    return sorted(
        path for path in PACKAGE_ROOT.rglob("*")
        if (
            path.is_file()
            and "_repo_overlay" not in path.parts
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and path.name not in {".DS_Store", "asset_manifest.json"}
            and not path.name.endswith(".zip")
        )
    )


def build_manifest(files: Sequence[Path]) -> dict[str, object]:
    return {
        "schema": "dr.anmar.asset-manifest.v1",
        "asset": ASSET_NAME,
        "version": VERSION,
        "catalog_subpath": CATALOG_SUBPATH.as_posix(),
        "files": [
            {
                "path": path.relative_to(PACKAGE_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }


def sync_extension_data() -> None:
    destination = EXTENSION_ROOT / "data" / CATALOG_SUBPATH
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ASSET_ROOT, destination)


def zip_tree(source: Path, output: Path, *, prefix: str | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if (
                "_repo_overlay" in relative.parts
                or "__pycache__" in relative.parts
                or path.suffix == ".pyc"
                or path.name == ".DS_Store"
            ):
                continue
            arcname = Path(prefix) / relative if prefix else relative
            info = zipfile.ZipInfo(arcname.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (path.stat().st_mode & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output


def write_checksum(path: Path) -> Path:
    checksum_path = Path(str(path) + ".sha256")
    checksum_path.write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    return checksum_path


def build_overlay() -> Path:
    overlay = PACKAGE_ROOT / "_repo_overlay"
    if overlay.exists():
        shutil.rmtree(overlay)
    for top in ("source", "physics_next", "docs", "examples", "tests"):
        src = PACKAGE_ROOT / top
        if src.exists():
            shutil.copytree(src, overlay / top, dirs_exist_ok=True)
    (overlay / "scripts").mkdir(parents=True, exist_ok=True)
    for script_name in (
        SCRIPT_PATH.name,
        "install_into_dranmar.py",
        "requirements_wound_preparation_generation.txt",
        "validate_dranmar_wound_preparation_robot.py",
    ):
        source = PACKAGE_ROOT / "scripts" / script_name
        if source.is_file():
            shutil.copy2(source, overlay / "scripts" / script_name)
    return overlay


def generate() -> dict[str, object]:
    for cache_dir in sorted(PACKAGE_ROOT.rglob("__pycache__"), reverse=True):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
    for compiled in PACKAGE_ROOT.rglob("*.pyc"):
        compiled.unlink()
    for metadata_file in PACKAGE_ROOT.rglob(".DS_Store"):
        metadata_file.unlink()
    overlay = PACKAGE_ROOT / "_repo_overlay"
    if overlay.exists():
        shutil.rmtree(overlay)
    if ASSET_ROOT.exists():
        shutil.rmtree(ASSET_ROOT)
    for preview_name in (
        "dranmar_wound_preparation_robot_preview.png",
        "dranmar_wound_preparation_robot_full_arm_preview.png",
    ):
        (PREVIEW_ROOT / preview_name).unlink(missing_ok=True)
    for directory in (ASSET_ROOT, GLB_ROOT, TEXTURE_ROOT, PREVIEW_ROOT, DOCS_ROOT, EXAMPLE_ROOT, INTEGRATION_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)
    bundle = build_tool()
    asset_files = write_asset_files(bundle)
    texture_files = generate_textures()
    glb_files = export_glbs(bundle)
    previews = [make_preview(bundle), make_full_arm_preview(bundle)]

    metadata_files = [
        write_json(ASSET_ROOT / "interaction_frames.json", interaction_frames(bundle)),
        write_json(ASSET_ROOT / "franka_mount_contract.json", mount_contract()),
        write_json(ASSET_ROOT / "wound_preparation_task_contract.json", task_contract()),
        write_json(ASSET_ROOT / "physics_profile.json", physics_profile(bundle)),
        write_json(ASSET_ROOT / "collider_coverage.json", collider_coverage(bundle)),
    ]
    (ASSET_ROOT / "README.md").write_text(readme(), encoding="utf-8")
    shutil.copy2(ASSET_ROOT / "README.md", PACKAGE_ROOT / "README.md")
    (ASSET_ROOT / "LICENSE.txt").write_text(
        "Copyright 2026 DrAnmar Project Developers\n\nLicensed under the Apache License, Version 2.0.\n",
        encoding="utf-8",
    )

    INTEGRATION_PATH.write_text(author_integration_module(), encoding="utf-8")
    (DOCS_ROOT / "MECHANISM.md").write_text(docs_mechanism(), encoding="utf-8")
    (DOCS_ROOT / "FLUID_MODEL.md").write_text(docs_fluid(), encoding="utf-8")
    (DOCS_ROOT / "DEBRIDEMENT_MODEL.md").write_text(docs_debridement(), encoding="utf-8")
    (DOCS_ROOT / "FRANKA_INTEGRATION.md").write_text(docs_franka(), encoding="utf-8")
    (EXAMPLE_ROOT / "franka_wound_preparation_scene.py").write_text(example_scene(), encoding="utf-8")
    installer_path = PACKAGE_ROOT / "scripts/install_into_dranmar.py"
    installer_path.write_text(author_installer(), encoding="utf-8")
    installer_path.chmod(0o755)

    profile_copy = PACKAGE_ROOT / "physics_next/surgical-preparation/dranmar-wound-preparation-robot-v1.json"
    write_json(profile_copy, physics_profile(bundle))
    sync_extension_data()

    manifest_path = ASSET_ROOT / "asset_manifest.json"
    write_json(manifest_path, build_manifest(all_payload_files()))
    sync_extension_data()

    parent = PACKAGE_ROOT.parent
    canonical_package_name = f"dranmar_wound_preparation_robot_v{VERSION}"
    dev_zip = zip_tree(
        PACKAGE_ROOT,
        parent / f"{canonical_package_name}.zip",
        prefix=canonical_package_name,
    )
    catalog_zip = zip_tree(PACKAGE_ROOT / "assets", parent / "dranmar_wound_preparation_robot_catalog_v0.1.0.zip")
    overlay = build_overlay()
    overlay_zip = zip_tree(overlay, parent / "dranmar_wound_preparation_robot_repo_overlay_v0.1.0.zip")
    shutil.rmtree(overlay)

    release = {
        "schema": "dr.anmar.release.v1",
        "asset": ASSET_NAME,
        "version": VERSION,
        "catalog_subpath": CATALOG_SUBPATH.as_posix(),
        "development_package": {"path": dev_zip.name, "sha256": sha256(dev_zip)},
        "catalog_package": {"path": catalog_zip.name, "sha256": sha256(catalog_zip)},
        "repository_overlay": {"path": overlay_zip.name, "sha256": sha256(overlay_zip)},
        "main_assets": [path.name for path in asset_files],
        "glb_exports": [path.name for path in glb_files],
        "previews": [path.name for path in previews],
        "fluid_model": "PhysX PBD particles with conservative volume ledger",
        "debridement_model": "temporary debris attachments released by cumulative contact work",
        "runtime_validation": "left_to_user_stack_iteration",
        "clinical_validation": False,
        "medical_device": False,
    }
    release_path = parent / "dranmar_wound_preparation_robot_release_v0.1.0.json"
    write_json(release_path, release)
    checksum_paths = [write_checksum(path) for path in (dev_zip, catalog_zip, overlay_zip)]
    return {
        "asset_files": len(asset_files), "texture_files": len(texture_files), "glb_files": len(glb_files),
        "package": str(dev_zip), "catalog": str(catalog_zip), "overlay": str(overlay_zip),
        "release": str(release_path), "checksums": [str(path) for path in checksum_paths],
    }


def main() -> None:
    result = generate()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
