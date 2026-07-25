#!/usr/bin/env python3
"""Generate the DrAnmar force-controlled atraumatic surgical exposure robot.

The asset family is a manufacturer-neutral research platform for bilateral soft-
tissue capture, force-limited retraction, maintained surgical exposure, and ROI
visibility benchmarking in NVIDIA Isaac Sim / Isaac Lab. It is not clinically
validated, is not approved for patient care, and every mechanics value is a
provisional engineering seed until physical calibration is supplied.
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
ASSET_NAME = "DrAnmar Atraumatic Surgical Exposure Robot"
CATALOG_SUBPATH = Path("Props/SurgicalExposure/AtraumaticExposureRobot")
ROOT_PRIM = "DrAnmarAtraumaticExposureTool"
STANDALONE_ROOT = "DrAnmarAtraumaticExposureToolStandalone"
PROXY_ROOT = "DrAnmarAtraumaticExposureToolRigidProxy"
FENESTRATED_PAD_ROOT = "DrAnmarFenestratedRetractionPad"
MICROCUP_PAD_ROOT = "DrAnmarMicrocupRetractionPad"
TISSUE_ROOT = "DrAnmarExposureTissueDemo"
ROI_ROOT = "DrAnmarExposureROITarget"

SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[1]
ASSET_ROOT = PACKAGE_ROOT / "assets" / CATALOG_SUBPATH
GLB_ROOT = ASSET_ROOT / "glb"
TEXTURE_ROOT = ASSET_ROOT / "textures"
PREVIEW_ROOT = PACKAGE_ROOT / "previews"
DOCS_ROOT = PACKAGE_ROOT / "docs" / "atraumatic_exposure_robot"
EXAMPLE_ROOT = PACKAGE_ROOT / "examples"
EXTENSION_ROOT = PACKAGE_ROOT / "source/extensions/orbit.surgical.assets"
INTEGRATION_PATH = EXTENSION_ROOT / "orbit/surgical/assets/atraumatic_exposure_robot.py"

# Tool coordinate convention: +Z approaches the surgical field, +X opens the
# exposure laterally, +Y follows the long axis of the exposed region.
WORK_PLANE_Z = 0.184
PAD_CAPTURE_CELL_COUNT = 6
PAD_MICROCUP_COUNT = 9
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


def rounded_frame(width: float, height: float, radius: float, tube: float, z: float = 0.0) -> trimesh.Trimesh:
    x = width/2 - radius
    y = height/2 - radius
    points = [(-x,-height/2,z), (x,-height/2,z), (width/2,-y,z), (width/2,y,z), (x,height/2,z), (-x,height/2,z), (-width/2,y,z), (-width/2,-y,z), (-x,-height/2,z)]
    return wire_path(points, tube)


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


@dataclass
class Visual:
    name: str
    mesh: trimesh.Trimesh
    material: str
    labels: tuple[str,...] = ()


@dataclass
class Collider:
    name: str
    kind: str
    center: tuple[float,float,float]
    size: tuple[float,float,float] | None = None
    radius: float | None = None
    height: float | None = None
    axis: str = "z"
    orientation_wxyz: tuple[float,float,float,float] = (1.0,0.0,0.0,0.0)
    physics_material: str = "PolymerPhysics"
    role: str = "collision"
    author_enabled: bool = True


@dataclass
class Link:
    name: str
    translation: tuple[float,float,float]
    visuals: list[Visual]
    colliders: list[Collider]
    mass_kg: float | None
    labels: tuple[str,...] = ()
    mass_properties: dict[str,object] | None = field(init=False)
    def __post_init__(self):
        self.mass_properties = None if self.mass_kg is None else box_mass_properties([v.mesh for v in self.visuals], self.mass_kg)


@dataclass
class Joint:
    name: str
    type: str
    body0: str
    body1: str
    axis: str | None
    local_pos0: tuple[float,float,float]
    local_pos1: tuple[float,float,float]
    lower: float | None = None
    upper: float | None = None
    stiffness: float = 0.0
    damping: float = 0.0
    max_force: float = 0.0
    target_velocity: float = 0.0


@dataclass
class ToolBundle:
    links: dict[str,Link]
    joints: list[Joint]
    frames: dict[str,dict[str,object]]
    fenestrated_pad: trimesh.Trimesh
    microcup_pad: trimesh.Trimesh
    left_flap: trimesh.Trimesh
    right_flap: trimesh.Trimesh
    roi_target: trimesh.Trimesh
    tissue_base: trimesh.Trimesh


# ---------------------------- Geometry ----------------------------

def build_fenestrated_pad(side: int = 1) -> trimesh.Trimesh:
    parts: list[trimesh.Trimesh] = []
    parts.append(rounded_frame(0.050, 0.036, 0.008, 0.0018, 0.0))
    for y in (-0.010, 0.0, 0.010):
        parts.append(capsule_between((-0.021,y,0), (0.021,y,0), 0.00125, sections=16))
    for x in (-0.012,0.0,0.012):
        parts.append(capsule_between((x,-0.014,0), (x,0.014,0), 0.00105, sections=16))
    # Three compliant trapping lips curl around the tissue edge without a sharp pinch point.
    lip_x = 0.0225 * side
    for y in (-0.011,0.0,0.011):
        path = [(lip_x,y,-0.001),(lip_x+0.0025*side,y,0.0025),(lip_x+0.0015*side,y,0.0060)]
        parts.append(wire_path(path,0.0014))
    backing = box_mesh((0.052,0.038,0.0035),(0,0,-0.0040))
    parts.append(backing)
    return trimesh.util.concatenate(parts)


def build_microcup_pad(side: int = 1) -> trimesh.Trimesh:
    parts = [ellipsoid_mesh((0.026,0.019,0.0038),(0,0,-0.002),subdivisions=3), box_mesh((0.052,0.038,0.003),(0,0,-0.005))]
    positions = [(-0.015,-0.010), (0,-0.010), (0.015,-0.010), (-0.015,0), (0,0), (0.015,0), (-0.015,0.010), (0,0.010), (0.015,0.010)]
    for x,y in positions:
        parts.append(torus_axis(0.0030,0.00055,"z",(x,y,0.0018),major_sections=36,minor_sections=10))
        parts.append(frustum_axis(0.00265,0.00165,0.0016,"z",(x,y,0.0010),sections=32))
    lip_x = 0.023*side
    for y in (-0.012,0.012):
        parts.append(wire_path([(lip_x,y,-0.001),(lip_x+0.002*side,y,0.003),(lip_x+0.001*side,y,0.006)],0.0012))
    return trimesh.util.concatenate(parts)


def build_tissue_flap(side: int, nx: int = 38, ny: int = 54) -> trimesh.Trimesh:
    if side < 0:
        xs = np.linspace(-0.078, 0.018, nx)
    else:
        xs = np.linspace(-0.018, 0.078, nx)
    ys = np.linspace(-0.060,0.060,ny)
    vertices: list[tuple[float,float,float]] = []
    for y in ys:
        for x in xs:
            inner = max(0.0, 1.0 - abs(x)/(0.020 if abs(x)<0.020 else 0.020))
            dome = 0.0065*math.exp(-1.8*((x/0.070)**2 + (y/0.072)**2))
            overlap = 0.0045*math.exp(-((x/(0.020))**2))*math.exp(-((y/0.056)**4))
            texture = 0.00035*math.sin(145*x+0.6*side)*math.cos(120*y)
            edge_roll = 0.0018*math.exp(-((abs(x)-0.015)/0.006)**2)
            z = dome + overlap + texture + edge_roll
            vertices.append((x,y,z))
    faces: list[tuple[int,int,int]] = []
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i; b=a+1; c=a+nx; d=c+1
            if side < 0:
                faces += [(a,b,d),(a,d,c)]
            else:
                faces += [(a,d,b),(a,c,d)]
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.fix_normals()
    return mesh


def build_roi_target() -> trimesh.Trimesh:
    target = cylinder_axis(0.022,0.0025,"z",(0,0,-0.0045),sections=72)
    inner = cylinder_axis(0.014,0.0010,"z",(0,0,-0.0027),sections=72)
    cross = trimesh.util.concatenate([
        box_mesh((0.032,0.003,0.0012),(0,0,-0.0018)),
        box_mesh((0.003,0.032,0.0012),(0,0,-0.0018)),
    ])
    return trimesh.util.concatenate([target,inner,cross])


def capture_cell_positions() -> list[tuple[float,float,float]]:
    return [(-0.014,-0.010,0.0015),(0,-0.010,0.0015),(0.014,-0.010,0.0015),(-0.014,0.010,0.0015),(0,0.010,0.0015),(0.014,0.010,0.0015)]


def build_tool() -> ToolBundle:
    fenestrated_left = build_fenestrated_pad(-1)
    fenestrated_right = build_fenestrated_pad(1)
    microcup_left = build_microcup_pad(-1)
    microcup_right = build_microcup_pad(1)
    links: dict[str,Link] = {}

    mount_visuals: list[Visual] = [
        Visual("FrankaAdapterPlate", cylinder_axis(0.032,0.012,"z",(0,0,0.006),sections=64), "MountMetal", ("franka_mount",)),
        Visual("QuickReleaseRing", torus_axis(0.0275,0.003,"z",(0,0,0.014)), "MountMetal"),
        Visual("MainHousing", ellipsoid_mesh((0.052,0.043,0.030),(0,0,0.050),subdivisions=3), "BodyPolymer", ("atraumatic_exposure_end_effector",)),
        Visual("HousingCore", box_mesh((0.094,0.074,0.042),(0,0,0.052)), "BodyPolymer"),
        Visual("RetractionRail", box_mesh((0.138,0.022,0.018),(0,0,0.094)), "RailMetal", ("bilateral_retraction_rail",)),
        Visual("RailCover", box_mesh((0.126,0.030,0.009),(0,0,0.082)), "AccentPolymer"),
        Visual("CentralCameraBridge", box_mesh((0.052,0.014,0.014),(0,-0.037,0.071)), "DarkPolymer", ("exposure_sensor_bridge",)),
        Visual("StereoCameraLeft", cylinder_axis(0.0045,0.004,"y",(-0.012,-0.044,0.071),sections=32), "SensorGlass", ("rgb_camera",)),
        Visual("StereoCameraRight", cylinder_axis(0.0045,0.004,"y",(0.012,-0.044,0.071),sections=32), "SensorGlass", ("rgb_camera",)),
        Visual("DepthEmitter", cylinder_axis(0.0037,0.004,"y",(0,-0.044,0.082),sections=32), "SensorBlue", ("depth_projector",)),
        Visual("IlluminationRing", torus_axis(0.024,0.0013,"z",(0,0,0.118),major_sections=72,minor_sections=12), "IndicatorWhite", ("field_illumination",)),
        Visual("ROICamera", cylinder_axis(0.0070,0.008,"z",(0,0,0.119),sections=40), "SensorGlass", ("roi_camera",)),
        Visual("LabelPanel", box_mesh((0.048,0.0012,0.020),(0,-0.0435,0.047)), "LabelMaterial"),
    ]
    for i in range(6):
        a=2*math.pi*i/6
        mount_visuals.append(Visual(f"MountBolt_{i:02d}",cylinder_axis(0.002,0.002,"z",(0.024*math.cos(a),0.024*math.sin(a),0.0145),sections=24),"MountMetal"))
    links["Mount"] = Link("Mount",(0,0,0),mount_visuals,[
        Collider("AdapterCollider","cylinder",(0,0,0.008),radius=0.032,height=0.016,physics_material="MountPhysics"),
        Collider("HousingCollider","box",(0,0,0.052),size=(0.108,0.086,0.068),physics_material="PolymerPhysics"),
        Collider("RailCollider","box",(0,0,0.094),size=(0.142,0.026,0.021),physics_material="MountPhysics"),
    ],0.285,("robotic_surgical_exposure_system",))

    for side_name, side in (("Left",-1),("Right",1)):
        x0=side*0.028
        carriage_visuals=[
            Visual("CarriageBody",box_mesh((0.032,0.036,0.026),(0,0,0)),"AccentPolymer",("retraction_carriage",)),
            Visual("LinearBearing",box_mesh((0.027,0.020,0.010),(0,0,-0.014)),"RailMetal"),
            Visual("PositionScale",box_mesh((0.024,0.0012,0.006),(0,-0.0185,0.006)),"LabelMaterial"),
            Visual("CableGuide",torus_axis(0.010,0.0012,"y",(0,0.014,0),major_sections=40,minor_sections=10),"DarkPolymer"),
        ]
        links[f"{side_name}Carriage"] = Link(f"{side_name}Carriage",(x0,0,0.095),carriage_visuals,[Collider("CarriageCollider","box",(0,0,0),size=(0.034,0.038,0.028),physics_material="PolymerPhysics")],0.070,("lateral_retraction_carriage",))

        lift_visuals=[
            Visual("TelescopingOuter",box_mesh((0.018,0.024,0.050),(0,0,0.025)),"DarkPolymer",("vertical_lift_stage",)),
            Visual("TelescopingInner",box_mesh((0.012,0.018,0.046),(0,0,0.050)),"RailMetal"),
            Visual("LiftCable",wire_path([(0,0.011,0.002),(0,0.014,0.030),(0,0.011,0.065)],0.0012),"CableMaterial"),
        ]
        links[f"{side_name}Lift"] = Link(f"{side_name}Lift",(x0,0,0.095),lift_visuals,[Collider("LiftCollider","box",(0,0,0.032),size=(0.020,0.026,0.066),physics_material="PolymerPhysics")],0.052,("independent_lift_axis",))

        pitch_visuals=[
            Visual("PitchHub",cylinder_axis(0.010,0.026,"y",(0,0,0),sections=48),"MountMetal",("pad_pitch_axis",)),
            Visual("RetractionArm",box_mesh((0.018,0.024,0.052),(0,0,0.026)),"BodyPolymer"),
            Visual("FlexureWindow",rounded_frame(0.014,0.025,0.003,0.0008,0.030),"SensorBlue",("flexure_force_sensor",)),
        ]
        links[f"{side_name}Pitch"] = Link(f"{side_name}Pitch",(x0,0,0.143),pitch_visuals,[Collider("PitchHubCollider","cylinder",(0,0,0),radius=0.0105,height=0.028,axis="y",physics_material="MountPhysics"),Collider("ArmCollider","box",(0,0,0.027),size=(0.020,0.026,0.056),physics_material="PolymerPhysics")],0.046,("pad_orientation_stage","force_sensor_flexure"))

        pad_visuals=[
            Visual("PadBacking",box_mesh((0.056,0.042,0.006),(0,0,-0.004)),"PadBacking",("retraction_pad_backing",)),
            Visual("FenestratedContact",fenestrated_left if side<0 else fenestrated_right,"PadElastomer",("fenestrated_atraumatic_contact",)),
            Visual("MicrocupContact",microcup_left if side<0 else microcup_right,"MicrocupElastomer",("distributed_low_vacuum_contact",)),
            Visual("ForceIndicator",box_mesh((0.018,0.0013,0.005),(0,-0.0215,-0.004)),"IndicatorGreen",("load_indicator",)),
        ]
        # Add visible strain-gauge strips and cup manifolds.
        for y in (-0.012,0.012):
            pad_visuals.append(Visual(f"StrainGauge_{'P' if y>0 else 'N'}",box_mesh((0.030,0.0022,0.0007),(0,y,-0.0072)),"SensorGold",("strain_gauge",)))
        pad_visuals.append(Visual("VacuumManifold",wire_path([(-0.018,0,-0.006),(0,0,-0.010),(0.018,0,-0.006)],0.0012),"VacuumTube",("microvacuum_manifold",)))
        pad_colliders=[Collider("PadBackingCollider","box",(0,0,-0.004),size=(0.057,0.043,0.007),physics_material="PadBackingPhysics",role="pad_backing")]
        for idx,pos in enumerate(capture_cell_positions()):
            pad_colliders.append(Collider(f"TissueCaptureCell_{idx:02d}","box",pos,size=(0.013,0.012,0.0045),physics_material="PadContactPhysics",role="tissue_capture_cell"))
        links[f"{side_name}Pad"] = Link(f"{side_name}Pad",(x0,0,0.184),pad_visuals,pad_colliders,0.054,("atraumatic_tissue_retraction_pad","distributed_capture_array"))

    joints=[
        Joint("left_carriage_joint","prismatic","Mount","LeftCarriage","X",(-0.028,0,0.095),(0,0,0),0.0,0.040,5200,190,95),
        Joint("right_carriage_joint","prismatic","Mount","RightCarriage","X",(0.028,0,0.095),(0,0,0),-0.040,0.0,5200,190,95),
        Joint("left_lift_joint","prismatic","LeftCarriage","LeftLift","Z",(0,0,0),(0,0,0),-0.025,0.030,6200,210,110),
        Joint("right_lift_joint","prismatic","RightCarriage","RightLift","Z",(0,0,0),(0,0,0),-0.025,0.030,6200,210,110),
        Joint("left_pitch_joint","revolute","LeftLift","LeftPitch","Y",(0,0,0.048),(0,0,0),-42.0,72.0,52,2.5,7.0),
        Joint("right_pitch_joint","revolute","RightLift","RightPitch","Y",(0,0,0.048),(0,0,0),-72.0,42.0,52,2.5,7.0),
        Joint("left_compliance_joint","prismatic","LeftPitch","LeftPad","Z",(0,0,0.041),(0,0,0),-0.006,0.0,1250,38,16),
        Joint("right_compliance_joint","prismatic","RightPitch","RightPad","Z",(0,0,0.041),(0,0,0),-0.006,0.0,1250,38,16),
    ]

    frames: dict[str,dict[str,object]]={
        "panda_link8_mount":{"position":[0,0,0],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"robot_mount"},
        "exposure_tcp":{"position":[0,0,WORK_PLANE_Z],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"tool_center_point"},
        "roi_camera":{"position":[0,0,0.123],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"roi_camera"},
        "illumination_center":{"position":[0,0,0.118],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"field_illumination"},
        "exposure_center":{"position":[0,0,WORK_PLANE_Z+0.003],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"exposure_roi_center"},
        "count_reference":{"position":[0,-0.048,0.050],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"inventory_reference"},
    }
    for side_name,side in (("Left",-1),("Right",1)):
        frames[f"{side_name.lower()}_pad_center"]={"position":[0,0,0],"orientation_wxyz":[1,0,0,0],"parent_link":f"{side_name}Pad","role":"pad_contact_center"}
        frames[f"{side_name.lower()}_pad_normal"]={"position":[0,0,0.004],"orientation_wxyz":[1,0,0,0],"parent_link":f"{side_name}Pad","role":"pad_contact_normal"}
        frames[f"{side_name.lower()}_force_sensor"]={"position":[0,0,-0.006],"orientation_wxyz":[1,0,0,0],"parent_link":f"{side_name}Pad","role":"force_sensor_reference"}
        for idx,pos in enumerate(capture_cell_positions()):
            frames[f"{side_name.lower()}_capture_{idx:02d}"]={"position":list(pos),"orientation_wxyz":[1,0,0,0],"parent_link":f"{side_name}Pad","role":"distributed_tissue_capture_cell"}

    left_flap=build_tissue_flap(-1)
    right_flap=build_tissue_flap(1)
    roi=build_roi_target()
    tissue_base=box_mesh((0.180,0.145,0.016),(0,0,-0.016))
    return ToolBundle(links,joints,frames,fenestrated_left,microcup_left,left_flap,right_flap,roi,tissue_base)

# ---------------------------- OpenUSD authoring ----------------------------

def mesh_usda(visual: Visual, material_path: str, indent: str = "                ") -> str:
    mesh=visual.mesh
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    points=",\n".join(f"{indent}        {vec(p)}" for p in np.asarray(mesh.vertices))
    faces=np.asarray(mesh.faces,dtype=int)
    counts=", ".join("3" for _ in faces)
    indices=", ".join(str(int(i)) for i in faces.reshape(-1))
    normals=",\n".join(f"{indent}        {vec(n)}" for n in np.asarray(mesh.vertex_normals))
    bmin,bmax=mesh.bounds
    labels=", ".join(f'"{x}"' for x in visual.labels)
    label_attr=f'\n{indent}    custom token[] drAnmar:labels = [{labels}]' if labels else ""
    return f'''{indent}def Mesh "{visual.name}" (
{indent}    prepend apiSchemas = ["MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    uniform bool doubleSided = false
{indent}    float3[] extent = [{vec(bmin)}, {vec(bmax)}]
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
{indent}    uniform token subdivisionScheme = "none"
{indent}    rel material:binding = <{material_path}>{label_attr}
{indent}}}'''


def collider_usda(collider: Collider, root_path: str, indent: str = "                ") -> str:
    schemas='["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]'
    enabled="true" if collider.author_enabled else "false"
    common=f'''{indent}    bool physics:collisionEnabled = {enabled}
{indent}    float physxCollision:contactOffset = 0.0008
{indent}    float physxCollision:restOffset = 0
{indent}    rel material:binding:physics = <{root_path}/PhysicsMaterials/{collider.physics_material}>
{indent}    custom token drAnmar:role = "{collider.role}"'''
    if collider.kind=="box":
        assert collider.size is not None
        return f'''{indent}def Cube "{collider.name}" (
{indent}    prepend apiSchemas = {schemas}
{indent})
{indent}{{
{indent}    double size = 1
{indent}    double3 xformOp:translate = {vec(collider.center)}
{indent}    quatf xformOp:orient = {quat(collider.orientation_wxyz)}
{indent}    double3 xformOp:scale = {vec(collider.size)}
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
{indent}    uniform token purpose = "guide"
{indent}    token visibility = "invisible"
{common}
{indent}}}'''
    if collider.kind in {"cylinder","capsule"}:
        assert collider.radius is not None and collider.height is not None
        typename="Cylinder" if collider.kind=="cylinder" else "Capsule"
        return f'''{indent}def {typename} "{collider.name}" (
{indent}    prepend apiSchemas = {schemas}
{indent})
{indent}{{
{indent}    uniform token axis = "{collider.axis.upper()}"
{indent}    double radius = {f(collider.radius)}
{indent}    double height = {f(collider.height)}
{indent}    double3 xformOp:translate = {vec(collider.center)}
{indent}    quatf xformOp:orient = {quat(collider.orientation_wxyz)}
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{indent}    uniform token purpose = "guide"
{indent}    token visibility = "invisible"
{common}
{indent}}}'''
    raise ValueError(collider.kind)


def shader_material(root_name: str, name: str, color: tuple[float,float,float], metallic: float, roughness: float, opacity: float=1.0, texture: str|None=None) -> str:
    texture_block=""
    diffuse_line=f"color3f inputs:diffuseColor = {vec(color)}"
    if texture:
        texture_block=f'''
                def Shader "Texture" (
                    prepend apiSchemas = ["NodeDefAPI"]
                )
                {{
                    uniform token info:id = "UsdUVTexture"
                    asset inputs:file = @{texture}@
                    token inputs:sourceColorSpace = "sRGB"
                    float2 inputs:st.connect = <../Primvar.outputs:result>
                    float3 outputs:rgb
                }}
                def Shader "Primvar" (
                    prepend apiSchemas = ["NodeDefAPI"]
                )
                {{
                    uniform token info:id = "UsdPrimvarReader_float2"
                    string inputs:varname = "st"
                    float2 outputs:result
                }}'''
        diffuse_line="color3f inputs:diffuseColor.connect = <../Texture.outputs:rgb>"
    return f'''        def Material "{name}"
        {{
            token outputs:surface.connect = <Shader.outputs:surface>
            def Shader "Shader" (
                prepend apiSchemas = ["NodeDefAPI"]
            )
            {{
                uniform token info:id = "UsdPreviewSurface"
                {diffuse_line}
                float inputs:metallic = {f(metallic)}
                float inputs:roughness = {f(roughness)}
                float inputs:opacity = {f(opacity)}
                token outputs:surface
            }}{texture_block}
        }}'''


def visual_materials_scope(root_name: str) -> str:
    definitions=[
        ("MountMetal",(0.32,0.36,0.42),0.88,0.22,1.0,"./textures/brushed_metal_basecolor.png"),
        ("RailMetal",(0.48,0.52,0.58),0.82,0.28,1.0,None),
        ("BodyPolymer",(0.87,0.90,0.93),0.03,0.38,1.0,"./textures/body_polymer_basecolor.png"),
        ("AccentPolymer",(0.02,0.34,0.56),0.05,0.30,1.0,"./textures/accent_polymer_basecolor.png"),
        ("DarkPolymer",(0.045,0.060,0.075),0.05,0.42,1.0,None),
        ("PadBacking",(0.12,0.16,0.18),0.08,0.46,1.0,None),
        ("PadElastomer",(0.06,0.48,0.43),0.0,0.74,1.0,"./textures/pad_elastomer_basecolor.png"),
        ("MicrocupElastomer",(0.08,0.66,0.58),0.0,0.69,1.0,"./textures/microcup_elastomer_basecolor.png"),
        ("SensorGlass",(0.02,0.07,0.10),0.25,0.07,0.72,None),
        ("SensorBlue",(0.02,0.58,0.96),0.10,0.18,1.0,None),
        ("SensorGold",(0.92,0.62,0.10),0.72,0.20,1.0,None),
        ("IndicatorGreen",(0.10,0.96,0.40),0.0,0.18,1.0,None),
        ("IndicatorAmber",(0.98,0.48,0.08),0.0,0.22,1.0,None),
        ("IndicatorWhite",(0.95,0.98,1.0),0.0,0.10,1.0,None),
        ("VacuumTube",(0.06,0.20,0.24),0.0,0.34,0.86,None),
        ("CableMaterial",(0.08,0.09,0.10),0.0,0.55,1.0,None),
        ("LabelMaterial",(0.98,0.98,0.99),0.0,0.48,1.0,"./textures/dranmar_exposure_label.png"),
        ("TissueLeft",(0.76,0.32,0.30),0.0,0.68,1.0,"./textures/tissue_left_basecolor.png"),
        ("TissueRight",(0.71,0.25,0.25),0.0,0.70,1.0,"./textures/tissue_right_basecolor.png"),
        ("ROIVisual",(0.18,0.88,0.58),0.08,0.26,1.0,None),
        ("ROICenter",(0.98,0.72,0.12),0.25,0.18,1.0,None),
        ("BasePad",(0.16,0.18,0.21),0.0,0.72,1.0,None),
    ]
    return '    def Scope "Looks"\n    {\n'+"\n".join(shader_material(root_name,*entry) for entry in definitions)+"\n    }"


def physics_materials_scope() -> str:
    values={
        "MountPhysics":(0.42,0.32,0.02),
        "PolymerPhysics":(0.58,0.46,0.03),
        "PadBackingPhysics":(0.62,0.49,0.01),
        "PadContactPhysics":(1.08,0.86,0.0),
        "TissuePhysics":(0.72,0.58,0.0),
        "BasePhysics":(0.60,0.48,0.0),
    }
    blocks=[]
    for name,(static,dynamic,restitution) in values.items():
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
    return '    def Scope "PhysicsMaterials"\n    {\n'+"\n".join(blocks)+"\n    }"


def frame_usda(name: str, data: dict[str,object], indent: str="                ") -> str:
    return f'''{indent}def Xform "{name}"
{indent}{{
{indent}    double3 xformOp:translate = {vec(data["position"])}
{indent}    quatf xformOp:orient = {quat(data["orientation_wxyz"])}
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{indent}    custom token drAnmar:role = "{data["role"]}"
{indent}    custom token drAnmar:parentLink = "{data["parent_link"]}"
{indent}}}'''


def link_usda(link: Link, root_path: str, frames: dict[str,dict[str,object]]) -> str:
    visual_blocks=[mesh_usda(v,f"{root_path}/Looks/{v.material}") for v in link.visuals]
    collider_blocks=[collider_usda(c,root_path) for c in link.colliders]
    labels=", ".join(f'"{x}"' for x in link.labels)
    local_frames=[frame_usda(name,data) for name,data in frames.items() if data["parent_link"]==link.name]
    p=link.mass_properties
    mass_block=""
    if p is not None:
        mass_block=f'''
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
            int physxRigidBody:solverPositionIterationCount = 20
            int physxRigidBody:solverVelocityIterationCount = 6
            float physxRigidBody:maxDepenetrationVelocity = 0.35
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
    body0=f"{root_path}/Links/{joint.body0}"
    body1=f"{root_path}/Links/{joint.body1}"
    common=f'''            rel physics:body0 = <{body0}>
            rel physics:body1 = <{body1}>
            point3f physics:localPos0 = {vec(joint.local_pos0)}
            point3f physics:localPos1 = {vec(joint.local_pos1)}
            quatf physics:localRot0 = (1, 0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            bool physics:collisionEnabled = false'''
    if joint.type=="fixed":
        return f'''        def PhysicsFixedJoint "{joint.name}"
        {{
{common}
        }}'''
    if joint.type=="prismatic":
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
    if joint.type=="revolute":
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
    schemas=["SemanticsLabelsAPI:class","SemanticsLabelsAPI:workflow"]
    if articulation_root:
        schemas.append("PhysicsArticulationRootAPI")
    schema_text=", ".join(f'"{x}"' for x in schemas)
    return f'''def Xform "{ROOT_PRIM}" (
    prepend apiSchemas = [{schema_text}]
    prepend variantSets = ["pad_type"]
    variants = {{
        string pad_type = "fenestrated"
    }}
    assetInfo = {{
        string name = "{ROOT_PRIM}"
        string version = "{VERSION}"
    }}
    customData = {{
        string drAnmarStatus = "simulation_training_workcell"
        string drAnmarMountInterface = "franka_panda_link8_hand_replacement"
        string drAnmarMechanism = "bilateral_distributed_capture_force_sensing_lateral_retraction_and_independent_lift"
        string drAnmarCoordinateConvention = "+Z approach, +X lateral opening, +Y ROI tangent"
        bool drAnmarClinicalValidation = false
        bool drAnmarMedicalDevice = false
    }}
    kind = "component"
)'''


def pad_type_variants() -> str:
    def side_block(side: str, fen_vis: str, micro_vis: str) -> str:
        return f'''                    over "{side}Pad"
                    {{
                        over "Visuals"
                        {{
                            over "FenestratedContact"
                            {{
                                token visibility = "{fen_vis}"
                            }}
                            over "MicrocupContact"
                            {{
                                token visibility = "{micro_vis}"
                            }}
                        }}
                    }}'''
    return f'''    variantSet "pad_type" = {{
        "fenestrated" {{
            over "Links"
            {{
{side_block("Left","inherited","invisible")}
{side_block("Right","inherited","invisible")}
            }}
            custom token drAnmar:activePadType = "fenestrated"
        }}
        "microcup" {{
            over "Links"
            {{
{side_block("Left","invisible","inherited")}
{side_block("Right","invisible","inherited")}
            }}
            custom token drAnmar:activePadType = "microcup"
        }}
    }}'''


def tool_usda(bundle: ToolBundle, articulation_root: bool) -> str:
    root_path=f"/{ROOT_PRIM}"
    links="\n".join(link_usda(link,root_path,bundle.frames) for link in bundle.links.values())
    joints="\n".join(joint_usda(joint,root_path) for joint in bundle.joints)
    return f'''#usda 1.0
(
    defaultPrim = "{ROOT_PRIM}"
    doc = "Dr.Anmar force-controlled surgical exposure end effector for simulation training."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

{root_header(articulation_root)}
{{
    token[] semantics:labels:class = ["robotic_surgical_retractor", "atraumatic_exposure_robot"]
    token[] semantics:labels:workflow = ["tissue_capture", "retraction", "exposure", "force_control"]
{visual_materials_scope(ROOT_PRIM)}
{physics_materials_scope()}
    def Scope "Links"
    {{
{links}
    }}
    def Scope "Joints"
    {{
{joints}
    }}
{pad_type_variants()}
}}
'''


def material_color(material: str) -> tuple[int,int,int,int]:
    colors={
        "MountMetal":(82,92,105,255),"RailMetal":(126,136,150,255),"BodyPolymer":(225,231,237,255),
        "AccentPolymer":(13,101,158,255),"DarkPolymer":(18,25,31,255),"PadBacking":(30,42,47,255),
        "PadElastomer":(20,142,126,255),"MicrocupElastomer":(28,190,164,255),"SensorGlass":(5,22,32,205),
        "SensorBlue":(8,146,244,255),"SensorGold":(232,157,24,255),"IndicatorGreen":(30,238,95,255),
        "IndicatorAmber":(248,116,16,255),"IndicatorWhite":(242,250,255,255),"VacuumTube":(16,66,76,230),
        "CableMaterial":(28,30,34,255),"LabelMaterial":(248,249,251,255),"TissueLeft":(194,82,76,255),
        "TissueRight":(176,62,62,255),"ROIVisual":(45,222,144,255),"ROICenter":(248,183,28,255),
        "BasePad":(116,124,134,255),"RobotWhite":(228,232,236,255),"RobotDark":(48,54,61,255),
        "RobotJoint":(94,102,113,255),"DebugOrange":(247,117,18,175),"DebugGreen":(40,235,90,175),
        "DebugBlue":(15,145,240,175),"DebugMagenta":(238,46,170,175),
    }
    return colors.get(material,(180,180,180,255))


def pbr(mesh: trimesh.Trimesh, material: str) -> trimesh.Trimesh:
    mesh=mesh.copy(); color=material_color(material)
    metal=0.85 if any(t in material for t in ("Metal","Rail","Gold")) else 0.02
    rough=0.22 if metal>0.5 else 0.48
    mesh.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(name=material,baseColorFactor=np.asarray(color,dtype=np.uint8),metallicFactor=metal,roughnessFactor=rough,alphaMode="BLEND" if color[3]<255 else "OPAQUE"))
    return mesh


def rigid_proxy_usda(bundle: ToolBundle) -> str:
    entries=[]
    for link in bundle.links.values():
        for visual in link.visuals:
            if visual.name=="MicrocupContact":
                continue
            mesh=visual.mesh.copy(); mesh.apply_translation(np.asarray(link.translation))
            entries.append(Visual(f"{link.name}_{visual.name}",mesh,visual.material,visual.labels))
    all_meshes=[v.mesh for v in entries]
    mass=0.66
    mp=box_mass_properties(all_meshes,mass)
    visual_blocks=[mesh_usda(v,f"/{PROXY_ROOT}/Looks/{v.material}",indent="        ") for v in entries]
    bmin,bmax=mesh_bounds(all_meshes); size=bmax-bmin; center=(bmin+bmax)/2
    return f'''#usda 1.0
(
    defaultPrim = "{PROXY_ROOT}"
    doc = "Rigid perception and planning proxy for the DrAnmar atraumatic exposure robot."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "{PROXY_ROOT}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "SemanticsLabelsAPI:class"]
    kind = "component"
)
{{
    token[] semantics:labels:class = ["robotic_surgical_retractor", "atraumatic_exposure_robot"]
    bool physics:rigidBodyEnabled = true
    float physics:mass = {f(mp["mass_kg"])}
    point3f physics:centerOfMass = {vec(mp["center_of_mass_m"])}
    float3 physics:diagonalInertia = {vec(mp["diagonal_inertia_kg_m2"])}
    quatf physics:principalAxes = {quat(mp["principal_axes_wxyz"])}
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(PROXY_ROOT)}
{physics_materials_scope()}
    def Scope "Visuals"
    {{
{chr(10).join(visual_blocks)}
    }}
    def Scope "Collisions"
    {{
{collider_usda(Collider("ProxyEnvelope","box",tuple(center),size=tuple(size*np.asarray((0.98,0.96,0.98))),physics_material="PolymerPhysics"),f"/{PROXY_ROOT}",indent="        ")}
    }}
}}
'''


def simple_pad_usda(root_name: str, pad_mesh: trimesh.Trimesh, pad_type: str) -> str:
    backing=box_mesh((0.056,0.042,0.006),(0,0,-0.004))
    visuals=[Visual("Backing",backing,"PadBacking"),Visual("Contact",pad_mesh,"PadElastomer" if pad_type=="fenestrated" else "MicrocupElastomer")]
    colliders=[Collider("BackingCollider","box",(0,0,-0.004),size=(0.057,0.043,0.007),physics_material="PadBackingPhysics")]
    for idx,pos in enumerate(capture_cell_positions()):
        colliders.append(Collider(f"TissueCaptureCell_{idx:02d}","box",pos,size=(0.013,0.012,0.0045),physics_material="PadContactPhysics",role="tissue_capture_cell"))
    mp=box_mass_properties([v.mesh for v in visuals],0.054)
    visual_blocks=[mesh_usda(v,f"/{root_name}/Looks/{v.material}",indent="        ") for v in visuals]
    collider_blocks=[collider_usda(c,f"/{root_name}",indent="        ") for c in colliders]
    frames="\n".join(frame_usda(f"capture_{i:02d}",{"position":list(pos),"orientation_wxyz":[1,0,0,0],"parent_link":root_name,"role":"distributed_tissue_capture_cell"},indent="        ") for i,pos in enumerate(capture_cell_positions()))
    return f'''#usda 1.0
(
    defaultPrim = "{root_name}"
    doc = "Replaceable {pad_type} atraumatic retraction pad for simulation training."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "{root_name}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "SemanticsLabelsAPI:class"]
    kind = "component"
)
{{
    token[] semantics:labels:class = ["surgical_retraction_pad", "{pad_type}_contact_pad"]
    bool physics:rigidBodyEnabled = true
    float physics:mass = {f(mp["mass_kg"])}
    point3f physics:centerOfMass = {vec(mp["center_of_mass_m"])}
    float3 physics:diagonalInertia = {vec(mp["diagonal_inertia_kg_m2"])}
    quatf physics:principalAxes = {quat(mp["principal_axes_wxyz"])}
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(root_name)}
{physics_materials_scope()}
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
{frames}
    }}
}}
'''


def tissue_mesh_block(name: str, mesh: trimesh.Trimesh, material: str, indent: str="        ") -> str:
    visual=Visual("SimulationMesh",mesh,material,("deformable_tissue_flap",))
    return f'''{indent}def Xform "{name}" (
{indent}    prepend apiSchemas = ["SemanticsLabelsAPI:class"]
{indent})
{indent}{{
{indent}    token[] semantics:labels:class = ["deformable_tissue_flap", "surgical_exposure_tissue"]
{mesh_usda(visual,f"/{TISSUE_ROOT}/Looks/{material}",indent=indent+"    ")}
{indent}}}'''


def tissue_demo_usda(bundle: ToolBundle) -> str:
    roi_visual=Visual("Visual",bundle.roi_target,"ROIVisual",("region_of_interest",))
    base_visual=Visual("Visual",bundle.tissue_base,"BasePad",("tissue_fixture",))
    roi_mesh=mesh_usda(roi_visual,f"/{TISSUE_ROOT}/Looks/ROIVisual",indent="            ")
    base_mesh=mesh_usda(base_visual,f"/{TISSUE_ROOT}/Looks/BasePad",indent="            ")
    return f'''#usda 1.0
(
    defaultPrim = "{TISSUE_ROOT}"
    doc = "Two-flap deformable exposure training scenario with an underlying ROI target."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "{TISSUE_ROOT}" (
    prepend apiSchemas = ["SemanticsLabelsAPI:workflow"]
    kind = "component"
)
{{
    token[] semantics:labels:workflow = ["tissue_retraction", "roi_exposure", "force_control"]
{visual_materials_scope(TISSUE_ROOT)}
{physics_materials_scope()}
    def Xform "Fixture"
    {{
{base_mesh}
        def Cube "LeftAnchor" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            double size = 1
            double3 xformOp:translate = (-0.073, 0, 0.003)
            double3 xformOp:scale = (0.012, 0.126, 0.014)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            token visibility = "invisible"
            uniform token purpose = "guide"
            rel material:binding:physics = </{TISSUE_ROOT}/PhysicsMaterials/BasePhysics>
        }}
        def Cube "RightAnchor" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            double size = 1
            double3 xformOp:translate = (0.073, 0, 0.003)
            double3 xformOp:scale = (0.012, 0.126, 0.014)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            token visibility = "invisible"
            uniform token purpose = "guide"
            rel material:binding:physics = </{TISSUE_ROOT}/PhysicsMaterials/BasePhysics>
        }}
    }}
{tissue_mesh_block("LeftFlap",bundle.left_flap,"TissueLeft")}
{tissue_mesh_block("RightFlap",bundle.right_flap,"TissueRight")}
    def Xform "ROITarget" (
        prepend apiSchemas = ["SemanticsLabelsAPI:class"]
    )
    {{
        token[] semantics:labels:class = ["surgical_region_of_interest", "exposure_target"]
{roi_mesh}
        def Xform "roi_center"
        {{
            double3 xformOp:translate = (0, 0, -0.001)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            custom token drAnmar:role = "roi_center"
        }}
    }}
}}
'''

# ---------------------------- Inspection exports ----------------------------

def export_scene(path: Path, entries: Sequence[tuple[str,trimesh.Trimesh,str]]) -> None:
    scene=trimesh.Scene()
    for name,mesh,material in entries:
        scene.add_geometry(pbr(mesh,material),node_name=name,geom_name=name)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))


def phase_parameters(phase: str) -> dict[str,float]:
    table={
        "stowed":{"left_carriage":0.0,"right_carriage":0.0,"lift":-0.012,"left_pitch":58.0,"right_pitch":-58.0,"compliance":0.0},
        "deployed":{"left_carriage":0.006,"right_carriage":-0.006,"lift":0.014,"left_pitch":12.0,"right_pitch":-12.0,"compliance":0.0},
        "captured":{"left_carriage":0.006,"right_carriage":-0.006,"lift":0.017,"left_pitch":2.0,"right_pitch":-2.0,"compliance":-0.0020},
        "exposed":{"left_carriage":0.032,"right_carriage":-0.032,"lift":-0.012,"left_pitch":-16.0,"right_pitch":16.0,"compliance":-0.0012},
        "overload_relief":{"left_carriage":0.020,"right_carriage":-0.020,"lift":-0.004,"left_pitch":-5.0,"right_pitch":5.0,"compliance":-0.0004},
    }
    try: return table[phase]
    except KeyError as exc: raise KeyError(phase) from exc


def link_world_transform(bundle: ToolBundle, link_name: str, phase: str) -> np.ndarray:
    p=phase_parameters(phase)
    base=np.asarray(bundle.links[link_name].translation,dtype=float)
    T=np.eye(4)
    if link_name.startswith("Left"):
        base[0]+=p["left_carriage"]
        if link_name in {"LeftLift","LeftPitch","LeftPad"}: base[2]+=p["lift"]
    if link_name.startswith("Right"):
        base[0]+=p["right_carriage"]
        if link_name in {"RightLift","RightPitch","RightPad"}: base[2]+=p["lift"]
    R=np.eye(3)
    if link_name in {"LeftPitch","LeftPad"}: R=rotation_matrix((0,1,0),math.radians(p["left_pitch"]))
    if link_name in {"RightPitch","RightPad"}: R=rotation_matrix((0,1,0),math.radians(p["right_pitch"]))
    if link_name=="LeftPad": base[2]+=p["compliance"]
    if link_name=="RightPad": base[2]+=p["compliance"]
    T[:3,:3]=R; T[:3,3]=base
    return T


def world_visual_entries(bundle: ToolBundle, phase: str="stowed", pad_type: str="fenestrated") -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[]
    for link_name,link in bundle.links.items():
        T=link_world_transform(bundle,link_name,phase)
        for visual in link.visuals:
            if visual.name=="FenestratedContact" and pad_type!="fenestrated": continue
            if visual.name=="MicrocupContact" and pad_type!="microcup": continue
            mesh=visual.mesh.copy(); mesh.apply_transform(T)
            entries.append((f"{link_name}_{visual.name}",mesh,visual.material))
    return entries


def collider_mesh(collider: Collider) -> trimesh.Trimesh:
    if collider.kind=="box":
        assert collider.size is not None
        mesh=box_mesh(collider.size,collider.center)
    elif collider.kind=="cylinder":
        assert collider.radius is not None and collider.height is not None
        mesh=cylinder_axis(collider.radius,collider.height,collider.axis,collider.center)
    elif collider.kind=="capsule":
        assert collider.radius is not None and collider.height is not None
        axis={"x":np.asarray([1,0,0]),"y":np.asarray([0,1,0]),"z":np.asarray([0,0,1])}[collider.axis]
        mesh=capsule_between(np.asarray(collider.center)-axis*collider.height/2,np.asarray(collider.center)+axis*collider.height/2,collider.radius)
    else: raise ValueError(collider.kind)
    q=np.asarray(collider.orientation_wxyz,dtype=float); w,x,y,z=q
    R=np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    if not np.allclose(R,np.eye(3)):
        T=np.eye(4); T[:3,:3]=R; mesh.apply_transform(T)
    return mesh


def collision_debug_entries(bundle: ToolBundle, phase: str="captured") -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[]
    for link_name,link in bundle.links.items():
        T=link_world_transform(bundle,link_name,phase)
        for collider in link.colliders:
            mesh=collider_mesh(collider); mesh.apply_transform(T)
            material="DebugMagenta" if collider.role=="tissue_capture_cell" else "DebugOrange"
            entries.append((f"{link_name}_{collider.name}",mesh,material))
    return entries


def frame_world(bundle: ToolBundle, data: dict[str,object], phase: str="captured") -> tuple[np.ndarray,np.ndarray]:
    T=link_world_transform(bundle,str(data["parent_link"]),phase)
    p=T[:3,:3]@np.asarray(data["position"],dtype=float)+T[:3,3]
    q=np.asarray(data["orientation_wxyz"],dtype=float); w,x,y,z=q
    Rf=np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return p,T[:3,:3]@Rf


def axis_entries(bundle: ToolBundle, phase: str="captured", length: float=0.012, radius: float=0.00048) -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[]
    for name,data in bundle.frames.items():
        if "capture_" in name and not name.endswith(("00","02","03","05")): continue
        p,R=frame_world(bundle,data,phase)
        for axis,mat in ((R[:,0],"DebugOrange"),(R[:,1],"DebugGreen"),(R[:,2],"DebugBlue")):
            entries.append((f"frame_{name}_{mat}",capsule_between(p,p+axis*length,radius),mat))
    return entries


def tissue_entries(bundle: ToolBundle, exposed: bool=False, *, align_z: float=0.0, include_base: bool=True) -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[]
    if include_base:
        base=bundle.tissue_base.copy(); base.apply_translation((0,0,align_z)); entries.append(("tissue_base",base,"BasePad"))
    roi=bundle.roi_target.copy(); roi.apply_translation((0,0,align_z)); entries.append(("roi",roi,"ROIVisual"))
    left=bundle.left_flap.copy(); right=bundle.right_flap.copy()
    if exposed:
        # Inspection-only deformed pose: outward translation, upward lift and gentle roll.
        pivot_l=np.asarray((-0.070,0,0)); pivot_r=np.asarray((0.070,0,0))
        for mesh,pivot,angle,shift in ((left,pivot_l,math.radians(-24),(-0.025,0,-0.010)),(right,pivot_r,math.radians(24),(0.025,0,-0.010))):
            mesh.apply_translation(-pivot)
            T=np.eye(4); T[:3,:3]=rotation_matrix((0,1,0),angle); mesh.apply_transform(T)
            mesh.apply_translation(pivot+np.asarray(shift))
    left.apply_translation((0,0,align_z)); right.apply_translation((0,0,align_z))
    entries += [("left_flap",left,"TissueLeft"),("right_flap",right,"TissueRight")]
    return entries


def franka_proxy_entries(bundle: ToolBundle, phase: str="deployed") -> list[tuple[str,trimesh.Trimesh,str]]:
    points=[np.asarray((-0.30,0,-0.72)),np.asarray((-0.30,0,-0.59)),np.asarray((-0.43,0,-0.44)),np.asarray((-0.40,0.03,-0.20)),np.asarray((-0.27,-0.02,-0.26)),np.asarray((-0.12,-0.02,-0.23)),np.asarray((-0.08,0.02,-0.14)),np.asarray((0,0,-0.055)),np.asarray((0,0,0))]
    entries=[("arm_base",cylinder_axis(0.105,0.075,"z",tuple(points[0])),"RobotDark"),("arm_pedestal",cylinder_axis(0.078,0.10,"z",tuple(points[0]+np.asarray((0,0,0.08)))),"RobotWhite")]
    for i,(a,b) in enumerate(zip(points[1:-1],points[2:])): entries.append((f"arm_link_{i:02d}",capsule_between(a,b,0.052 if i<3 else 0.043),"RobotWhite"))
    for i,p in enumerate(points[1:-1]):
        m=trimesh.creation.icosphere(subdivisions=2,radius=0.061 if i<3 else 0.050); m.apply_translation(p); entries.append((f"arm_joint_{i:02d}",m,"RobotJoint"))
    entries.append(("panda_link8_proxy",cylinder_axis(0.045,0.11,"z",(0,0,-0.055)),"RobotDark"))
    entries.extend(world_visual_entries(bundle,phase))
    return entries


def exploded_entries(bundle: ToolBundle) -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[]
    offsets={"Mount":np.asarray((0,0,0)),"LeftCarriage":np.asarray((-0.08,0,0)),"RightCarriage":np.asarray((0.08,0,0)),"LeftLift":np.asarray((-0.11,0,0.01)),"RightLift":np.asarray((0.11,0,0.01)),"LeftPitch":np.asarray((-0.14,0,0.02)),"RightPitch":np.asarray((0.14,0,0.02)),"LeftPad":np.asarray((-0.17,0,0.03)),"RightPad":np.asarray((0.17,0,0.03))}
    for link_name,link in bundle.links.items():
        base=np.asarray(link.translation)+offsets[link_name]
        for visual in link.visuals:
            if visual.name=="MicrocupContact": continue
            mesh=visual.mesh.copy(); mesh.apply_translation(base); entries.append((f"{link_name}_{visual.name}",mesh,visual.material))
    return entries


def export_glbs(bundle: ToolBundle) -> list[Path]:
    specs={
        "dranmar_exposure_tool_stowed.glb":world_visual_entries(bundle,"stowed","fenestrated"),
        "dranmar_exposure_tool_deployed.glb":world_visual_entries(bundle,"deployed","fenestrated"),
        "dranmar_exposure_tool_captured.glb":world_visual_entries(bundle,"captured","microcup"),
        "dranmar_exposure_tool_exposed.glb":world_visual_entries(bundle,"exposed","fenestrated"),
        "dranmar_exposure_tool_overload_relief.glb":world_visual_entries(bundle,"overload_relief","fenestrated"),
        "dranmar_exposure_tool_exploded.glb":exploded_entries(bundle),
        "dranmar_exposure_tool_collision_debug.glb":world_visual_entries(bundle,"captured")+collision_debug_entries(bundle,"captured"),
        "dranmar_exposure_tool_frame_debug.glb":world_visual_entries(bundle,"captured")+axis_entries(bundle,"captured"),
        "dranmar_franka_exposure_assembly.glb":franka_proxy_entries(bundle,"deployed"),
        "dranmar_exposure_tissue_initial.glb":tissue_entries(bundle,False),
        "dranmar_exposure_tissue_exposed.glb":tissue_entries(bundle,True),
        "dranmar_fenestrated_pad.glb":[("pad",bundle.fenestrated_pad,"PadElastomer")],
        "dranmar_microcup_pad.glb":[("pad",bundle.microcup_pad,"MicrocupElastomer")],
    }
    paths=[]
    for name,entries in specs.items():
        path=GLB_ROOT/name; export_scene(path,entries); paths.append(path)
    return paths


def add_mesh_to_axis(ax, mesh: trimesh.Trimesh, material: str, max_faces: int=1200) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces=np.asarray(mesh.faces)
    if len(faces)>max_faces:
        faces=faces[np.linspace(0,len(faces)-1,max_faces,dtype=int)]
    verts=np.asarray(mesh.vertices)[faces]
    color=np.asarray(material_color(material),dtype=float)/255.0
    poly=Poly3DCollection(verts,facecolor=color,edgecolor=(0,0,0,0.06),linewidth=0.08,alpha=color[3])
    ax.add_collection3d(poly)


def configure_axis(ax, title: str, elev: float=22, azim: float=-58) -> None:
    ax.set_title(title,fontsize=10,pad=5,fontweight="bold")
    ax.view_init(elev=elev,azim=azim)
    ax.set_axis_off(); ax.set_box_aspect((1.35,1.0,1.25))


def make_preview(bundle: ToolBundle) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(15,9),dpi=170)
    panels=[("STOWED",world_visual_entries(bundle,"stowed")),("DEPLOY + CONTACT",world_visual_entries(bundle,"captured","microcup")),("FORCE-CONTROLLED EXPOSURE",world_visual_entries(bundle,"exposed")+tissue_entries(bundle,True,align_z=WORK_PLANE_Z-0.002)),("TISSUE BEFORE",tissue_entries(bundle,False,include_base=False)),("TISSUE EXPOSED",tissue_entries(bundle,True,include_base=False)),("DISTRIBUTED CAPTURE CELLS",world_visual_entries(bundle,"captured")+collision_debug_entries(bundle,"captured"))]
    for idx,(title,entries) in enumerate(panels,1):
        ax=fig.add_subplot(2,3,idx,projection="3d")
        for _,mesh,mat in entries: add_mesh_to_axis(ax,mesh,mat,900)
        configure_axis(ax,title,24,-60 if idx not in (4,5) else -78)
        pts=np.vstack([m.bounds for _,m,_ in entries]).reshape(-1,3); mn=pts.min(0); mx=pts.max(0); c=(mn+mx)/2; r=max(mx-mn)*0.60
        ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    fig.suptitle("DrAnmar Atraumatic Surgical Exposure Robot",fontsize=19,fontweight="bold",y=0.985)
    fig.text(0.5,0.018,"bilateral compliant pads • distributed tissue capture • independent lift • force-limited retraction • ROI exposure control",ha="center",fontsize=10)
    path=PREVIEW_ROOT/"dranmar_atraumatic_exposure_robot_preview.png"; fig.tight_layout(rect=(0,0.04,1,0.96)); fig.savefig(path,bbox_inches="tight"); plt.close(fig); return path


def make_full_arm_preview(bundle: ToolBundle) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    entries=franka_proxy_entries(bundle,"exposed")+tissue_entries(bundle,True,align_z=WORK_PLANE_Z-0.002)
    fig=plt.figure(figsize=(10,10),dpi=180); ax=fig.add_subplot(111,projection="3d")
    for _,mesh,mat in entries: add_mesh_to_axis(ax,mesh,mat,1400)
    configure_axis(ax,"Franka-compatible hand replacement",18,-58)
    pts=np.vstack([m.bounds for _,m,_ in entries]).reshape(-1,3); mn=pts.min(0); mx=pts.max(0); c=(mn+mx)/2; r=max(mx-mn)*0.58
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
    fig.suptitle("DrAnmar Force-Controlled Exposure System",fontsize=18,fontweight="bold",y=0.96)
    path=PREVIEW_ROOT/"dranmar_atraumatic_exposure_robot_full_arm_preview.png"; fig.savefig(path,bbox_inches="tight"); plt.close(fig); return path


# ---------------------------- Textures and metadata ----------------------------

def noise_texture(base: tuple[int,int,int], size: int=512, strength: int=18, seed: int=1) -> Image.Image:
    rng=np.random.default_rng(seed); arr=np.zeros((size,size,3),dtype=np.int16); arr[:]=base
    noise=rng.normal(0,strength,(size,size,1)); arr=np.clip(arr+noise,0,255).astype(np.uint8)
    return Image.fromarray(arr,"RGB")


def save_texture(image: Image.Image, name: str) -> Path:
    path=TEXTURE_ROOT/name; image.save(path,optimize=True); return path


def generate_textures() -> list[Path]:
    paths=[
        save_texture(noise_texture((180,188,198),strength=11,seed=1),"brushed_metal_basecolor.png"),
        save_texture(noise_texture((226,231,237),strength=7,seed=2),"body_polymer_basecolor.png"),
        save_texture(noise_texture((14,103,160),strength=10,seed=3),"accent_polymer_basecolor.png"),
        save_texture(noise_texture((20,145,128),strength=14,seed=4),"pad_elastomer_basecolor.png"),
        save_texture(noise_texture((28,190,164),strength=16,seed=5),"microcup_elastomer_basecolor.png"),
        save_texture(noise_texture((194,82,76),strength=13,seed=6),"tissue_left_basecolor.png"),
        save_texture(noise_texture((176,62,62),strength=14,seed=7),"tissue_right_basecolor.png"),
    ]
    label=Image.new("RGB",(1024,420),(246,249,252)); d=ImageDraw.Draw(label)
    try: font=ImageFont.truetype("DejaVuSans-Bold.ttf",92); small=ImageFont.truetype("DejaVuSans.ttf",38)
    except OSError: font=ImageFont.load_default(); small=ImageFont.load_default()
    d.rounded_rectangle((26,26,998,394),radius=36,outline=(18,90,145),width=12,fill=(249,251,253))
    d.text((72,82),"DrAnmar",font=font,fill=(12,74,122)); d.text((76,226),"ATRAUMATIC EXPOSURE",font=small,fill=(18,96,142)); d.text((76,292),"SIMULATION TRAINING WORKCELL",font=small,fill=(96,104,112))
    paths.append(save_texture(label,"dranmar_exposure_label.png"))
    return paths


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return path


def interaction_frames(bundle: ToolBundle) -> dict[str,object]:
    return {"schema":"dr.anmar.interaction-frames.v1","asset":ASSET_NAME,"coordinate_convention":"+Z approaches field, +X opens exposure, +Y follows ROI","frames":bundle.frames}


def mount_contract() -> dict[str,object]:
    return {"schema":"dr.anmar.franka-mount.v1","parent_link":"resolved_from_stock_panda_hand_joint_body0_with_unique_panda_link8_fallback","disabled_stock_prims":["panda_hand_joint","panda_hand","panda_finger_joint1","panda_finger_joint2","panda_leftfinger","panda_rightfinger"],"payload_root":"DrAnmarAtraumaticExposureTool","payload_mount_link":"Links/Mount","fixed_joint":"dranmar_exposure_mount_joint","frame_source":"stock_panda_hand_joint_local_frame","fallback_local_rotation_wxyz":[math.cos(math.radians(-45)/2),0,0,math.sin(math.radians(-45)/2)],"translation_m":[0,0,0],"status":"composition_observed_physical_effect_unqualified"}


def task_contract() -> dict[str,object]:
    return {"schema":"dr.anmar.surgical-exposure-task.v1","phases":["stowed","approach","deploy","contact","capture","retract","hold","overload_relief","release"],"success_metrics":["roi_visible_fraction","left_pad_force_n","right_pad_force_n","force_asymmetry_n","capture_cell_count","flap_edge_separation_m","retraction_stability","overload_events"],"failure_events":["capture_loss","pad_overload","force_asymmetry","roi_reocclusion","tissue_attachment_break","joint_limit_contact"],"required_observations":["tool_joint_state","pad_compliance_state","pad_contact_force","camera_or_segmentation_roi_visibility","capture_attachment_state"],"clinical_validation":False}


def physics_profile(bundle: ToolBundle) -> dict[str,object]:
    return {"schema":"dr.anmar.atraumatic-exposure-profile.v1","id":"dranmar-atraumatic-exposure-robot-v1","version":VERSION,"status":"research_informed_engineering_model_pending_physical_and_clinical_validation","units":"MKS","mechanism":{"lateral_retraction_range_m":0.040,"independent_lift_range_m":[-0.025,0.030],"pad_pitch_ranges_deg":{"left":[-42,72],"right":[-72,42]},"compliance_travel_m":0.006,"compliance_stiffness_n_m":1250.0,"compliance_damping_n_s_m":38.0,"capture_cells_per_pad":PAD_CAPTURE_CELL_COUNT,"pad_contact_area_m2_seed":0.0015},"force_control":{"nominal_target_force_per_pad_n":1.25,"soft_limit_force_per_pad_n":2.5,"hard_release_force_per_pad_n":4.0,"maximum_force_asymmetry_n":1.0,"controller":"independent_force_limited_impedance_plus_roi_visibility_outer_loop","all_values":"provisional_engineering_seeds"},"capture":{"fenestrated":"geometric_trapping_plus_distributed_overlap_attachments","microcup":"distributed_low_vacuum_proxy_plus_overlap_attachments","attachment_cells":"independent_runtime_deformable_attachments","release":"progressive_outer_cell_release_then_full_release_on_hard_overload"},"tissue":{"representation":"two portable triangle surfaces cooked at runtime by the current surface-deformable route","youngs_modulus_pa_seed":60000.0,"poissons_ratio_seed":0.45,"surface_thickness_m_seed":0.006,"density_kg_m3_seed":1050.0,"self_collision":True},"runtime":{"observed_stack":"Isaac Sim 6.0.1.0 / Isaac Lab distribution 6.1.16","franka_mount":"stock panda_hand_joint frame with unique panda_link8 compatibility fallback","status":"composition_observed_physical_effect_unqualified"},"clinical_validation":False,"medical_device":False}


def collider_coverage(bundle: ToolBundle) -> dict[str,object]:
    links={}
    for name,link in bundle.links.items():
        vbmin,vbmax=mesh_bounds([v.mesh for v in link.visuals])
        cmeshes=[collider_mesh(c) for c in link.colliders]
        cbmin,cbmax=mesh_bounds(cmeshes)
        visual_size=np.maximum(vbmax-vbmin,1e-9); collision_size=cbmax-cbmin
        links[name]={"visual_bounds_m":{"min":vbmin.tolist(),"max":vbmax.tolist()},"collision_bounds_m":{"min":cbmin.tolist(),"max":cbmax.tolist()},"axis_coverage_ratio":(collision_size/visual_size).tolist(),"collider_count":len(link.colliders),"capture_cell_count":sum(c.role=="tissue_capture_cell" for c in link.colliders)}
    return {"schema":"dr.anmar.collider-coverage.v1","asset":ASSET_NAME,"links":links,"note":"Deliberate capture-cell protrusion exists to generate overlap attachments with deformable tissue."}

# ---------------------------- Isaac integration source ----------------------------

def author_integration_module() -> str:
    return r'''# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab integration for the DrAnmar atraumatic surgical exposure robot.

The tool replaces the Panda hand at its verified stock joint frame. Its bilateral
compliant pads use independent, overlap-prioritized vertex attachments. The control helpers
maintain ROI exposure while enforcing provisional pad-force limits. This module
is an engineering research interface, not clinical control software.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import math

CATALOG_SUBPATH = "Props/SurgicalExposure/AtraumaticExposureRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ROOT / "dranmar_atraumatic_exposure_tool_payload.usda"
TOOL_STANDALONE_USD = ROOT / "dranmar_atraumatic_exposure_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_atraumatic_exposure_tool_rigid_proxy.usda"
FENESTRATED_PAD_USD = ROOT / "dranmar_fenestrated_retraction_pad.usda"
MICROCUP_PAD_USD = ROOT / "dranmar_microcup_retraction_pad.usda"
TISSUE_DEMO_USD = ROOT / "dranmar_exposure_tissue_demo.usda"

VALID_PAD_TYPES = frozenset({"fenestrated", "microcup"})
CAPTURE_CELL_COUNT = 6

TOOL_JOINTS = {
    "left_carriage": "left_carriage_joint",
    "right_carriage": "right_carriage_joint",
    "left_lift": "left_lift_joint",
    "right_lift": "right_lift_joint",
    "left_pitch": "left_pitch_joint",
    "right_pitch": "right_pitch_joint",
    "left_compliance": "left_compliance_joint",
    "right_compliance": "right_compliance_joint",
}

TOOL_FRAME_PATHS = {
    "panda_link8_mount": "Links/Mount/Frames/panda_link8_mount",
    "exposure_tcp": "Links/Mount/Frames/exposure_tcp",
    "roi_camera": "Links/Mount/Frames/roi_camera",
    "illumination_center": "Links/Mount/Frames/illumination_center",
    "exposure_center": "Links/Mount/Frames/exposure_center",
    "count_reference": "Links/Mount/Frames/count_reference",
    "left_pad_center": "Links/LeftPad/Frames/left_pad_center",
    "right_pad_center": "Links/RightPad/Frames/right_pad_center",
    "left_pad_normal": "Links/LeftPad/Frames/left_pad_normal",
    "right_pad_normal": "Links/RightPad/Frames/right_pad_normal",
    "left_force_sensor": "Links/LeftPad/Frames/left_force_sensor",
    "right_force_sensor": "Links/RightPad/Frames/right_force_sensor",
}
REGISTERED_CAMERA_FRAMES = ("roi_camera",)
for _side in ("left", "right"):
    for _index in range(CAPTURE_CELL_COUNT):
        TOOL_FRAME_PATHS[f"{_side}_capture_{_index:02d}"] = (
            f"Links/{_side.capitalize()}Pad/Frames/{_side}_capture_{_index:02d}"
        )


def frame_path(tool_path: str, name: str) -> str:
    try:
        suffix = TOOL_FRAME_PATHS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown exposure-tool frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any):
    """Return a native tensor from Isaac 6 tensor proxy objects when required."""
    return value.torch if hasattr(value, "torch") else value


def _check(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}")
    return value


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_finite(value: float, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def make_tool_cfg(
    prim_path: str = "/World/DrAnmarAtraumaticExposureTool",
    *,
    pad_type: str = "fenestrated",
    position=(0.0, 0.0, 0.35),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    """Return a standalone Isaac Lab articulation configuration."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    _check(pad_type, VALID_PAD_TYPES, "pad_type")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants={"pad_type": pad_type},
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=6,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            rot=orientation_wxyz,
            joint_pos={
                "left_carriage_joint": 0.0,
                "right_carriage_joint": 0.0,
                "left_lift_joint": -0.012,
                "right_lift_joint": -0.012,
                "left_pitch_joint": math.radians(58.0),
                "right_pitch_joint": math.radians(-58.0),
                "left_compliance_joint": 0.0,
                "right_compliance_joint": 0.0,
            },
        ),
        actuators={
            "lateral_retraction": ImplicitActuatorCfg(
                joint_names_expr=[".*_carriage_joint"],
                effort_limit_sim=95.0,
                velocity_limit_sim=0.12,
                stiffness=5200.0,
                damping=190.0,
            ),
            "independent_lift": ImplicitActuatorCfg(
                joint_names_expr=[".*_lift_joint"],
                effort_limit_sim=110.0,
                velocity_limit_sim=0.10,
                stiffness=6200.0,
                damping=210.0,
            ),
            "pad_pitch": ImplicitActuatorCfg(
                joint_names_expr=[".*_pitch_joint"],
                effort_limit_sim=7.0,
                velocity_limit_sim=2.0,
                stiffness=52.0,
                damping=2.5,
            ),
            "pad_compliance": ImplicitActuatorCfg(
                joint_names_expr=[".*_compliance_joint"],
                effort_limit_sim=16.0,
                velocity_limit_sim=0.08,
                stiffness=1250.0,
                damping=38.0,
            ),
        },
    )


def make_rigid_proxy_cfg(
    prim_path: str = "/World/DrAnmarAtraumaticExposureProxy",
    *,
    position=(0.0, 0.0, 0.35),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_RIGID_PROXY_USD),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=orientation_wxyz),
    )


def _spawn_single_franka_with_exposure_tool(
    prim_path: str,
    cfg: Any,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn stock Franka, remove Panda hand, and mount the exposure payload."""
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

    tool_path = f"{prim_path}/DrAnmarAtraumaticExposureTool"
    create_prim(tool_path, usd_path=str(TOOL_PAYLOAD_USD), stage=stage)
    select_usd_variants(tool_path, {"pad_type": cfg.pad_type})

    mount_joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/dranmar_exposure_mount_joint")
    mount_joint.CreateBody0Rel().SetTargets(mount_body_paths)
    mount_joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    mount_joint.CreateLocalPos0Attr().Set(mount_local_pos0)
    mount_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    mount_joint.CreateLocalRot0Attr().Set(mount_local_rot0)
    mount_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return robot


def spawn_franka_with_exposure_tool(
    prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs
):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_exposure_tool)(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )


def make_franka_exposure_robot_cfg(
    *,
    prim_path: str = "/World/Robot",
    pad_type: str = "fenestrated",
):
    """Return the stock Isaac Lab Franka with the Panda hand replaced."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    _check(pad_type, VALID_PAD_TYPES, "pad_type")

    @configclass
    class FrankaExposureUsdCfg(sim_utils.UsdFileCfg):
        pad_type: str = "fenestrated"
        func = spawn_franka_with_exposure_tool

    cfg = FRANKA_PANDA_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn = FrankaExposureUsdCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        variants={"Gripper": "Default", "Mesh": "Performance"},
        pad_type=pad_type,
        activate_contact_sensors=True,
        rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,
        articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props,
    )
    cfg.init_state.joint_pos = {
        key: value for key, value in cfg.init_state.joint_pos.items() if "finger" not in key
    }
    cfg.init_state.joint_pos.update({
        "left_carriage_joint": 0.0,
        "right_carriage_joint": 0.0,
        "left_lift_joint": -0.012,
        "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(58.0),
        "right_pitch_joint": math.radians(-58.0),
        "left_compliance_joint": 0.0,
        "right_compliance_joint": 0.0,
    })
    cfg.actuators.pop("panda_hand", None)
    cfg.actuators.update({
        "exposure_lateral": ImplicitActuatorCfg(
            joint_names_expr=[".*_carriage_joint"], effort_limit_sim=95.0,
            velocity_limit_sim=0.12, stiffness=5200.0, damping=190.0,
        ),
        "exposure_lift": ImplicitActuatorCfg(
            joint_names_expr=[".*_lift_joint"], effort_limit_sim=110.0,
            velocity_limit_sim=0.10, stiffness=6200.0, damping=210.0,
        ),
        "exposure_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_pitch_joint"], effort_limit_sim=7.0,
            velocity_limit_sim=2.0, stiffness=52.0, damping=2.5,
        ),
        "exposure_compliance": ImplicitActuatorCfg(
            joint_names_expr=[".*_compliance_joint"], effort_limit_sim=16.0,
            velocity_limit_sim=0.08, stiffness=1250.0, damping=38.0,
        ),
    })
    return cfg


def spawn_exposure_tissue_demo(
    prim_path: str = "/World/DrAnmarExposureTissue",
    *,
    translation=(0.0, 0.0, 0.0),
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
):
    import isaaclab.sim as sim_utils
    cfg = sim_utils.UsdFileCfg(usd_path=str(TISSUE_DEMO_USD))
    return cfg.func(prim_path, cfg, translation=translation, orientation=orientation_wxyz)


def _current_stage(stage=None):
    if stage is not None:
        return stage
    import omni.usd
    return omni.usd.get_context().get_stage()


def apply_exposure_tissue_surface_deformables(
    tissue_root_path: str = "/World/DrAnmarExposureTissue",
    *,
    stage=None,
    material_path: str = "/World/Materials/DrAnmarExposureTissueSurface",
    youngs_modulus_pa: float = 60_000.0,
    poissons_ratio: float = 0.45,
    surface_thickness_m: float = 0.006,
    density_kg_m3: float = 1_050.0,
    dynamic_friction: float = 0.58,
    elasticity_damping: float = 0.16,
    bend_damping: float = 0.14,
    self_collision: bool = True,
) -> dict[str, Any]:
    """Cook both portable flap meshes with the current surface-deformable API."""
    stage = _current_stage(stage)
    from omni.physx.scripts import deformableUtils
    from pxr import UsdShade

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

    result = {"root_path": tissue_root_path, "material_path": material_path, "flaps": {}}
    for side in ("LeftFlap", "RightFlap"):
        actor_path = f"{tissue_root_path.rstrip('/')}/{side}"
        mesh_path = f"{actor_path}/SimulationMesh"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim or not mesh_prim.IsValid():
            raise ValueError(f"No exposure tissue mesh at {mesh_path}")
        success = deformableUtils.set_physics_surface_deformable_body(stage, mesh_prim.GetPath())
        if success is False:
            raise RuntimeError(f"PhysX could not create a surface deformable at {mesh_path}")
        mesh_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        if mesh_prim.HasAPI("PhysxSurfaceDeformableBodyAPI"):
            mesh_prim.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
        UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        result["flaps"][side] = {"actor_path": actor_path, "mesh_path": mesh_path}
    result["parameters"] = {
        "youngs_modulus_pa": youngs_modulus_pa,
        "poissons_ratio": poissons_ratio,
        "surface_thickness_m": surface_thickness_m,
        "density_kg_m3": density_kg_m3,
        "dynamic_friction": dynamic_friction,
        "elasticity_damping": elasticity_damping,
        "bend_damping": bend_damping,
        "self_collision": self_collision,
        "status": "provisional_engineering_seed",
    }
    return result


def create_deformable_attachment(
    deformable_prim_path: str,
    rigid_prim_path: str,
    attachment_path: str,
    *,
    stage=None,
) -> str:
    """Create a verified rigid/deformable attachment across Isaac generations."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    stage = _current_stage(stage)
    if stage.GetPrimAtPath(attachment_path).IsValid():
        stage.RemovePrim(attachment_path)

    prim_definition = Usd.SchemaRegistry().FindConcretePrimDefinition(
        "OmniPhysicsVtxXformAttachment"
    )
    if prim_definition:
        deformable_prim = stage.GetPrimAtPath(deformable_prim_path)
        if deformable_prim.IsValid() and not deformable_prim.IsA(UsdGeom.Mesh):
            candidate = stage.GetPrimAtPath(
                f"{deformable_prim_path.rstrip('/')}/SimulationMesh"
            )
            if candidate.IsValid():
                deformable_prim = candidate
                deformable_prim_path = str(candidate.GetPath())
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
                f"Could not create attachment {attachment_path}: "
                f"current={current_error!r}; legacy={legacy_error!r}"
            ) from legacy_error


def remove_prims(paths: Iterable[str], *, stage=None) -> None:
    stage = _current_stage(stage)
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            stage.RemovePrim(path)


def anchor_tissue_outer_bands(
    tissue_root_path: str = "/World/DrAnmarExposureTissue",
    *,
    stage=None,
) -> list[str]:
    """Attach the outer flap bands to the static fixture anchors."""
    stage = _current_stage(stage)
    attachments = []
    for side in ("Left", "Right"):
        path = f"{tissue_root_path}/Attachments/{side}OuterAnchor"
        create_deformable_attachment(
            f"{tissue_root_path}/{side}Flap",
            f"{tissue_root_path}/Fixture/{side}Anchor",
            path,
            stage=stage,
        )
        attachments.append(path)
    return attachments


@dataclass
class CaptureCell:
    side: str
    index: int
    attachment_path: str
    rigid_cell_path: str
    active: bool = True
    released_reason: str | None = None


@dataclass
class DistributedPadCaptureController:
    """Manage six independent tissue bonds per pad.

    Multiple small attachments distribute pad traction over the contact area.
    Overload handling progressively releases the outermost cells before the
    controller releases an entire pad. This is a research proxy for local loss
    of contact, not a calibrated tissue-injury or vacuum model.
    """

    tool_path: str
    left_tissue_path: str
    right_tissue_path: str
    stage: Any = None
    soft_cell_release_force_n: float = 0.75
    hard_pad_release_force_n: float = 4.0
    cells: list[CaptureCell] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.stage = _current_stage(self.stage)

    def capture(self) -> list[CaptureCell]:
        self.release_all("recapture")
        for side, tissue_path in (("left", self.left_tissue_path), ("right", self.right_tissue_path)):
            side_title = side.capitalize()
            for index in range(CAPTURE_CELL_COUNT):
                rigid_path = f"{self.tool_path}/Links/{side_title}Pad/Collisions/TissueCaptureCell_{index:02d}"
                attachment_path = f"{self.tool_path}/RuntimeAttachments/{side_title}Capture_{index:02d}"
                method = create_deformable_attachment(
                    tissue_path, rigid_path, attachment_path, stage=self.stage
                )
                self.cells.append(CaptureCell(side, index, attachment_path, rigid_path))
                self.events.append({"event": "capture", "side": side, "index": index, "method": method})
        return list(self.cells)

    def active_cells(self, side: str | None = None) -> list[CaptureCell]:
        return [cell for cell in self.cells if cell.active and (side is None or cell.side == side)]

    def release_cell(self, side: str, index: int, reason: str) -> bool:
        for cell in self.cells:
            if cell.side == side and cell.index == index and cell.active:
                remove_prims([cell.attachment_path], stage=self.stage)
                cell.active = False
                cell.released_reason = reason
                self.events.append({"event": "release_cell", "side": side, "index": index, "reason": reason})
                return True
        return False

    def release_all(self, reason: str = "commanded_release") -> None:
        active = self.active_cells()
        remove_prims([cell.attachment_path for cell in active], stage=self.stage)
        for cell in active:
            cell.active = False
            cell.released_reason = reason
        if active:
            self.events.append({"event": "release_all", "reason": reason, "count": len(active)})

    def update_loads(
        self,
        *,
        left_total_force_n: float,
        right_total_force_n: float,
        left_cell_forces_n: Sequence[float] | None = None,
        right_cell_forces_n: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        result = {"released": [], "hard_release": []}
        for side, total, values in (
            ("left", left_total_force_n, left_cell_forces_n),
            ("right", right_total_force_n, right_cell_forces_n),
        ):
            total = _nonnegative_finite(total, f"{side}_total_force_n")
            if total >= self.hard_pad_release_force_n:
                for cell in self.active_cells(side):
                    self.release_cell(side, cell.index, "hard_pad_overload")
                    result["hard_release"].append((side, cell.index))
                continue
            if values is not None:
                for index, force in enumerate(values[:CAPTURE_CELL_COUNT]):
                    force = _nonnegative_finite(force, f"{side}_cell_force_n[{index}]")
                    if force >= self.soft_cell_release_force_n:
                        if self.release_cell(side, index, "local_cell_overload"):
                            result["released"].append((side, index))
            elif total > 0:
                active = self.active_cells(side)
                estimate = total / max(1, len(active))
                if estimate >= self.soft_cell_release_force_n and active:
                    # Release the furthest longitudinal cell first.
                    order = [0, 3, 2, 5, 1, 4]
                    chosen = next((i for i in order if any(c.index == i for c in active)), active[0].index)
                    if self.release_cell(side, chosen, "estimated_distributed_overload"):
                        result["released"].append((side, chosen))
        result["active_left"] = len(self.active_cells("left"))
        result["active_right"] = len(self.active_cells("right"))
        return result


def estimate_pad_force_n(
    compression_m: float,
    compression_velocity_m_s: float = 0.0,
    *,
    stiffness_n_m: float = 1_250.0,
    damping_n_s_m: float = 38.0,
) -> float:
    """Estimate normal pad load from the authored compliant-axis deflection."""
    compression = max(0.0, -_finite(compression_m, "compression_m"))
    closing_velocity = max(
        0.0, -_finite(compression_velocity_m_s, "compression_velocity_m_s")
    )
    stiffness = _nonnegative_finite(stiffness_n_m, "stiffness_n_m")
    damping = _nonnegative_finite(damping_n_s_m, "damping_n_s_m")
    return stiffness * compression + damping * closing_velocity


@dataclass
class ForceControlOutput:
    joint_targets: dict[str, float]
    force_error_n: dict[str, float]
    exposure_error: float
    overload: dict[str, bool]
    mode: str


@dataclass
class ForceControlledRetractionController:
    """Outer-loop ROI controller with independent force-limited pad motion."""

    target_visible_fraction: float = 0.88
    target_force_per_pad_n: float = 1.25
    soft_force_limit_n: float = 2.5
    hard_force_limit_n: float = 4.0
    max_force_asymmetry_n: float = 1.0
    lateral_gain_m_per_fraction: float = 0.010
    lift_gain_m_per_fraction: float = 0.006
    force_gain_m_per_n: float = 0.0018
    integral_gain_m_per_fraction_s: float = 0.0012
    max_integral_m: float = 0.008
    nominal_update_hz: float = 120.0
    left_carriage_m: float = 0.006
    right_carriage_m: float = -0.006
    left_lift_m: float = 0.017
    right_lift_m: float = 0.017
    integral_error: float = 0.0

    def reset(self) -> None:
        self.left_carriage_m = 0.006
        self.right_carriage_m = -0.006
        self.left_lift_m = 0.017
        self.right_lift_m = 0.017
        self.integral_error = 0.0

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def update(
        self,
        *,
        dt: float,
        visible_fraction: float,
        left_force_n: float,
        right_force_n: float,
    ) -> ForceControlOutput:
        dt = _finite(dt, "dt")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        update_scale = dt * _nonnegative_finite(
            self.nominal_update_hz, "nominal_update_hz"
        )
        if update_scale <= 0.0:
            raise ValueError("nominal_update_hz must be positive")
        visible = self._clamp(_finite(visible_fraction, "visible_fraction"), 0.0, 1.0)
        left_force = _nonnegative_finite(left_force_n, "left_force_n")
        right_force = _nonnegative_finite(right_force_n, "right_force_n")
        exposure_error = self.target_visible_fraction - visible
        self.integral_error = self._clamp(
            self.integral_error + exposure_error * dt,
            -self.max_integral_m / max(self.integral_gain_m_per_fraction_s, 1e-9),
            self.max_integral_m / max(self.integral_gain_m_per_fraction_s, 1e-9),
        )

        left_over = left_force >= self.soft_force_limit_n
        right_over = right_force >= self.soft_force_limit_n
        hard_left = left_force >= self.hard_force_limit_n
        hard_right = right_force >= self.hard_force_limit_n

        if hard_left or hard_right:
            # Immediate commanded unloading; capture controller handles bond release.
            relief_step = 0.004 * update_scale
            self.left_carriage_m = max(0.0, self.left_carriage_m - relief_step)
            self.right_carriage_m = min(0.0, self.right_carriage_m + relief_step)
            self.left_lift_m = min(0.030, self.left_lift_m + relief_step)
            self.right_lift_m = min(0.030, self.right_lift_m + relief_step)
            mode = "hard_overload_relief"
        else:
            visibility_step = (
                self.lateral_gain_m_per_fraction * exposure_error
                + self.integral_gain_m_per_fraction_s * self.integral_error
            )
            left_force_error = self.target_force_per_pad_n - left_force
            right_force_error = self.target_force_per_pad_n - right_force
            left_step = (
                visibility_step + self.force_gain_m_per_n * left_force_error
            ) * update_scale
            right_step = (
                visibility_step + self.force_gain_m_per_n * right_force_error
            ) * update_scale
            if left_over:
                left_step = min(
                    left_step,
                    -self.force_gain_m_per_n
                    * (left_force - self.soft_force_limit_n)
                    * update_scale,
                )
            if right_over:
                right_step = min(
                    right_step,
                    -self.force_gain_m_per_n
                    * (right_force - self.soft_force_limit_n)
                    * update_scale,
                )
            self.left_carriage_m = self._clamp(self.left_carriage_m + left_step, 0.0, 0.040)
            self.right_carriage_m = self._clamp(self.right_carriage_m - right_step, -0.040, 0.0)

            # Lift assists exposure but unloads a pad that is already force limited.
            lift_step = (
                self.lift_gain_m_per_fraction * exposure_error * update_scale
            )
            overload_lift_step = 0.002 * update_scale
            self.left_lift_m = self._clamp(
                self.left_lift_m
                - lift_step
                + (overload_lift_step if left_over else 0.0),
                -0.025,
                0.030,
            )
            self.right_lift_m = self._clamp(
                self.right_lift_m
                - lift_step
                + (overload_lift_step if right_over else 0.0),
                -0.025,
                0.030,
            )
            mode = "force_limited_exposure_control"

        # Differential correction reduces excessive bilateral force asymmetry.
        asymmetry = left_force - right_force
        if abs(asymmetry) > self.max_force_asymmetry_n:
            correction = min(0.0025, 0.0012 * abs(asymmetry)) * update_scale
            if asymmetry > 0:
                self.left_carriage_m = max(0.0, self.left_carriage_m - correction)
            else:
                self.right_carriage_m = min(0.0, self.right_carriage_m + correction)

        return ForceControlOutput(
            joint_targets={
                "left_carriage_joint": self.left_carriage_m,
                "right_carriage_joint": self.right_carriage_m,
                "left_lift_joint": self.left_lift_m,
                "right_lift_joint": self.right_lift_m,
                "left_pitch_joint": math.radians(-16.0),
                "right_pitch_joint": math.radians(16.0),
                "left_compliance_joint": 0.0,
                "right_compliance_joint": 0.0,
            },
            force_error_n={
                "left": self.target_force_per_pad_n - left_force,
                "right": self.target_force_per_pad_n - right_force,
                "asymmetry": asymmetry,
            },
            exposure_error=exposure_error,
            overload={"left_soft": left_over, "right_soft": right_over, "left_hard": hard_left, "right_hard": hard_right},
            mode=mode,
        )


class ROIExposureEstimator:
    """Visibility metrics usable with segmentation masks or geometric flap edges."""

    @staticmethod
    def from_masks(roi_mask: Any, occluder_mask: Any) -> float:
        import numpy as np
        roi = np.asarray(tensor_value(roi_mask), dtype=bool)
        occluder = np.asarray(tensor_value(occluder_mask), dtype=bool)
        if roi.shape != occluder.shape:
            raise ValueError(f"mask shape mismatch: roi={roi.shape}, occluder={occluder.shape}")
        total = int(roi.sum())
        if total == 0:
            return 0.0
        visible = np.logical_and(roi, np.logical_not(occluder)).sum()
        return float(visible / total)

    @staticmethod
    def from_edge_gap(gap_width_m: float, target_width_m: float = 0.044) -> float:
        gap = _finite(gap_width_m, "gap_width_m")
        target = _finite(target_width_m, "target_width_m")
        if target <= 0:
            raise ValueError("target_width_m must be positive")
        return max(0.0, min(1.0, gap / target))

    @staticmethod
    def bilateral_balance(left_visible_fraction: float, right_visible_fraction: float) -> float:
        left = max(
            0.0,
            min(1.0, _finite(left_visible_fraction, "left_visible_fraction")),
        )
        right = max(
            0.0,
            min(1.0, _finite(right_visible_fraction, "right_visible_fraction")),
        )
        return 1.0 - abs(left - right)


PHASE_TARGETS = {
    "stowed": {
        "left_carriage_joint": 0.0, "right_carriage_joint": 0.0,
        "left_lift_joint": -0.012, "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(58.0), "right_pitch_joint": math.radians(-58.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "approach": {
        "left_carriage_joint": 0.002, "right_carriage_joint": -0.002,
        "left_lift_joint": 0.004, "right_lift_joint": 0.004,
        "left_pitch_joint": math.radians(35.0), "right_pitch_joint": math.radians(-35.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "deploy": {
        "left_carriage_joint": 0.006, "right_carriage_joint": -0.006,
        "left_lift_joint": 0.014, "right_lift_joint": 0.014,
        "left_pitch_joint": math.radians(12.0), "right_pitch_joint": math.radians(-12.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "contact": {
        "left_carriage_joint": 0.006, "right_carriage_joint": -0.006,
        "left_lift_joint": 0.017, "right_lift_joint": 0.017,
        "left_pitch_joint": math.radians(2.0), "right_pitch_joint": math.radians(-2.0),
        "left_compliance_joint": -0.002, "right_compliance_joint": -0.002,
    },
    "capture": {
        "left_carriage_joint": 0.006, "right_carriage_joint": -0.006,
        "left_lift_joint": 0.017, "right_lift_joint": 0.017,
        "left_pitch_joint": math.radians(2.0), "right_pitch_joint": math.radians(-2.0),
        "left_compliance_joint": -0.002, "right_compliance_joint": -0.002,
    },
    "retract": {
        "left_carriage_joint": 0.032, "right_carriage_joint": -0.032,
        "left_lift_joint": -0.012, "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(-16.0), "right_pitch_joint": math.radians(16.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "hold": {
        "left_carriage_joint": 0.032, "right_carriage_joint": -0.032,
        "left_lift_joint": -0.012, "right_lift_joint": -0.012,
        "left_pitch_joint": math.radians(-16.0), "right_pitch_joint": math.radians(16.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "overload_relief": {
        "left_carriage_joint": 0.020, "right_carriage_joint": -0.020,
        "left_lift_joint": -0.004, "right_lift_joint": -0.004,
        "left_pitch_joint": math.radians(-5.0), "right_pitch_joint": math.radians(5.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
    "release": {
        "left_carriage_joint": 0.010, "right_carriage_joint": -0.010,
        "left_lift_joint": 0.010, "right_lift_joint": 0.010,
        "left_pitch_joint": math.radians(20.0), "right_pitch_joint": math.radians(-20.0),
        "left_compliance_joint": 0.0, "right_compliance_joint": 0.0,
    },
}


def phase_targets(phase: str) -> dict[str, float]:
    try:
        return dict(PHASE_TARGETS[phase])
    except KeyError as exc:
        raise KeyError(f"Unknown exposure phase {phase!r}; expected one of {sorted(PHASE_TARGETS)}") from exc


@dataclass
class ExposureSequenceController:
    """Discrete workflow coordinator around capture and force-aware hold control."""

    tool_path: str
    left_tissue_path: str
    right_tissue_path: str
    stage: Any = None
    phase: str = "stowed"
    capture: DistributedPadCaptureController = field(init=False)
    force_controller: ForceControlledRetractionController = field(default_factory=ForceControlledRetractionController)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.stage = _current_stage(self.stage)
        self.capture = DistributedPadCaptureController(
            tool_path=self.tool_path,
            left_tissue_path=self.left_tissue_path,
            right_tissue_path=self.right_tissue_path,
            stage=self.stage,
        )

    def set_phase(self, phase: str) -> dict[str, float]:
        targets = phase_targets(phase)
        if phase == "capture":
            self.capture.capture()
        elif phase == "release":
            self.capture.release_all("sequence_release")
        elif phase == "stowed":
            self.force_controller.reset()
        self.phase = phase
        self.history.append({"event": "phase", "phase": phase, "targets": targets})
        return targets

    def hold_update(
        self,
        *,
        dt: float,
        visible_fraction: float,
        left_compression_m: float,
        right_compression_m: float,
        left_compression_velocity_m_s: float = 0.0,
        right_compression_velocity_m_s: float = 0.0,
        left_cell_forces_n: Sequence[float] | None = None,
        right_cell_forces_n: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        left_force = estimate_pad_force_n(left_compression_m, left_compression_velocity_m_s)
        right_force = estimate_pad_force_n(right_compression_m, right_compression_velocity_m_s)
        control = self.force_controller.update(
            dt=dt,
            visible_fraction=visible_fraction,
            left_force_n=left_force,
            right_force_n=right_force,
        )
        release = self.capture.update_loads(
            left_total_force_n=left_force,
            right_total_force_n=right_force,
            left_cell_forces_n=left_cell_forces_n,
            right_cell_forces_n=right_cell_forces_n,
        )
        if control.overload["left_hard"] or control.overload["right_hard"]:
            self.phase = "overload_relief"
        else:
            self.phase = "hold"
        result = {
            "phase": self.phase,
            "joint_targets": control.joint_targets,
            "visible_fraction": float(visible_fraction),
            "left_force_n": left_force,
            "right_force_n": right_force,
            "force_error_n": control.force_error_n,
            "exposure_error": control.exposure_error,
            "overload": control.overload,
            "capture_release": release,
            "mode": control.mode,
        }
        self.history.append({"event": "hold_update", **result})
        return result
'''

# ---------------------------- Documentation and packaging ----------------------------

def readme() -> str:
    return textwrap.dedent(f'''\
    # {ASSET_NAME} v{VERSION}

    DrAnmar-owned OpenUSD research assets for bilateral soft-tissue capture,
    force-limited retraction, maintained surgical exposure, and ROI visibility
    benchmarking, integrated with NVIDIA Isaac Lab and Isaac Sim.

    ## Catalog path

    ```text
    {CATALOG_SUBPATH.as_posix()}/
    ```

    ## Primary assets

    - `dranmar_atraumatic_exposure_tool_payload.usda`: hand-replacement payload for `panda_link8`.
    - `dranmar_atraumatic_exposure_tool_standalone.usda`: standalone articulation.
    - `dranmar_atraumatic_exposure_tool_rigid_proxy.usda`: perception/planning proxy.
    - `dranmar_fenestrated_retraction_pad.usda`: replaceable geometric-trapping pad.
    - `dranmar_microcup_retraction_pad.usda`: replaceable distributed low-vacuum proxy pad.
    - `dranmar_exposure_tissue_demo.usda`: two deformable flaps over an ROI target.

    ## Mechanism

    Each side has an independent lateral carriage, vertical lift, pad-pitch axis,
    and 6 mm compliant force-sensing axis. Each pad exposes six independent
    capture cells. Tissue capture is created at runtime from overlap-prioritized,
    explicitly verified deformable vertex attachments. Overload logic can release individual cells before
    releasing a complete pad.

    ## Validation

    Static integrity gates and the optional headless CUDA diagnostic are
    documented in `docs/atraumatic_exposure_robot/VALIDATION.md`. Runtime smoke
    observations do not qualify pad contact, tissue capture, or exposure
    efficacy.

    ## Research boundary

    All dimensions, friction values, tissue mechanics, capture strengths, force
    thresholds, vacuum behavior, and controller gains are provisional engineering
    seeds. The package does not claim calibrated tissue trauma, safe surgical
    force limits, clinical effectiveness, sterility, regulatory approval, or
    suitability for patient care.
    ''')


def docs_mechanism() -> str:
    return textwrap.dedent('''\
    # Mechanism

    The tool uses a symmetric bilateral mechanism. The carriages open laterally;
    the lift stages move each tissue margin independently; the pad pitch axes
    orient the contact surface; and the final compliant axes provide measurable
    travel between the driven arm and tissue-contact pad.

    The fenestrated pad uses distributed ribs and curled edge lips for geometric
    trapping. The microcup pad uses nine shallow cup geometries connected to a
    visible manifold. Both modes use the same six independent runtime capture
    cells so experiments can compare pad geometry while retaining one attachment
    and force-control contract.

    Mechanical states are joint states, not USD variants. The only root variant
    selects the discrete pad construction.
    ''')


def docs_force_control() -> str:
    return textwrap.dedent('''\
    # Force-aware exposure control

    Pad normal force is estimated from the authored compliance-axis displacement
    and velocity. The supplied controller has two loops:

    1. an outer ROI-visibility loop increases lateral separation and lift while
       exposure is below target;
    2. independent force limits unload either side before continuing exposure.

    A bilateral asymmetry correction prevents one pad from carrying a much
    larger load than the other. Hard overload commands immediate unloading and
    allows the capture controller to release the affected pad.

    The numerical thresholds are provisional research seeds, not tissue-specific
    safety limits. Calibration requires instrumented physical specimens and the
    selected target procedure.
    ''')


def docs_tissue() -> str:
    return textwrap.dedent('''\
    # Tissue and distributed capture

    The included benchmark uses two portable triangular flap meshes over a
    central ROI. Runtime code cooks each flap through the current surface-
    deformable route and anchors the outer bands to the fixture.

    Each contact pad contains six small capture volumes. The controller ranks
    tissue vertices by capture-cell overlap and creates one verified vertex
    attachment per cell. If a portable benchmark pose has fewer than four
    overlapping vertices, the nearest four are selected deterministically.
    This distributes traction spatially and allows local loss of capture without
    making the whole pad detach at once.

    The nearest-vertex fallback and capture cells are simulation contracts. They
    do not claim to reproduce a
    specific suction pressure, tissue injury threshold, ischemia response, or
    clinical retractor design.
    ''')


def docs_franka() -> str:
    return textwrap.dedent('''\
    # Franka integration

    `make_franka_exposure_robot_cfg()` references the composable Isaac Franka,
    snapshots the stock `panda_hand_joint` body and local frame, deactivates the
    Panda hand and finger prims, then fixes `Links/Mount` to that verified frame.
    A uniquely resolved `panda_link8` with the standard -45 degree Z frame is used
    only as a compatibility fallback for older Franka layouts.

    The payload intentionally has no articulation root. The Franka articulation
    owns the complete robot and tool. The standalone asset contains its own
    articulation root for isolated mechanism development.
    ''')


def example_scene() -> str:
    return r'''#!/usr/bin/env python3
"""Minimal DrAnmar atraumatic exposure scene for Isaac Lab."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from orbit.surgical.assets.atraumatic_exposure_robot import (
    make_franka_exposure_robot_cfg,
    spawn_exposure_tissue_demo,
)

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot = make_franka_exposure_robot_cfg(prim_path="{ENV_REGEX_NS}/Robot", pad_type="fenestrated")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
spawn_exposure_tissue_demo("/World/ExposureTissue", translation=(0.54, 0.0, 0.0))
sim.reset()
while app.is_running():
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
app.close()
'''


def author_installer() -> str:
    return (PACKAGE_ROOT / "scripts/install_into_dranmar.py").read_text(encoding="utf-8")


def write_asset_files(bundle: ToolBundle) -> list[Path]:
    paths=[]
    payload=ASSET_ROOT/"dranmar_atraumatic_exposure_tool_payload.usda"; payload.write_text(tool_usda(bundle,False),encoding="utf-8"); paths.append(payload)
    standalone=ASSET_ROOT/"dranmar_atraumatic_exposure_tool_standalone.usda"; standalone.write_text(tool_usda(bundle,True),encoding="utf-8"); paths.append(standalone)
    proxy=ASSET_ROOT/"dranmar_atraumatic_exposure_tool_rigid_proxy.usda"; proxy.write_text(rigid_proxy_usda(bundle),encoding="utf-8"); paths.append(proxy)
    fen=ASSET_ROOT/"dranmar_fenestrated_retraction_pad.usda"; fen.write_text(simple_pad_usda(FENESTRATED_PAD_ROOT,bundle.fenestrated_pad,"fenestrated"),encoding="utf-8"); paths.append(fen)
    micro=ASSET_ROOT/"dranmar_microcup_retraction_pad.usda"; micro.write_text(simple_pad_usda(MICROCUP_PAD_ROOT,bundle.microcup_pad,"microcup"),encoding="utf-8"); paths.append(micro)
    tissue=ASSET_ROOT/"dranmar_exposure_tissue_demo.usda"; tissue.write_text(tissue_demo_usda(bundle),encoding="utf-8"); paths.append(tissue)
    return paths


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def all_payload_files() -> list[Path]:
    return sorted(
        p
        for p in PACKAGE_ROOT.rglob("*")
        if p.is_file()
        and "_repo_overlay" not in p.parts
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and p.name not in {".DS_Store", "asset_manifest.json"}
        and not p.name.endswith(".zip")
    )


def build_manifest(files: Sequence[Path]) -> dict[str,object]:
    return {"schema":"dr.anmar.asset-manifest.v1","asset":ASSET_NAME,"version":VERSION,"catalog_subpath":CATALOG_SUBPATH.as_posix(),"files":[{"path":p.relative_to(PACKAGE_ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in files]}


def sync_extension_data() -> None:
    destination=EXTENSION_ROOT/"data"/CATALOG_SUBPATH
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(ASSET_ROOT,destination)


def zip_tree(source: Path, output: Path, *, prefix: str|None=None) -> Path:
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
                or path.name == ".DS_Store"
            ):
                continue
            relative=path.relative_to(source)
            arcname=Path(prefix)/relative if prefix else relative
            info=zipfile.ZipInfo(arcname.as_posix(),date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o755 if path.stat().st_mode & 0o111 else 0o644) << 16
            archive.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    return output


def write_checksum(path: Path) -> Path:
    out=Path(str(path)+".sha256"); out.write_text(f"{sha256(path)}  {path.name}\n",encoding="utf-8"); return out


def build_overlay() -> Path:
    overlay=PACKAGE_ROOT.parent/"_dranmar_atraumatic_exposure_robot_overlay"
    if overlay.exists(): shutil.rmtree(overlay)
    for top in ("source","physics_next","docs","examples","tests"):
        src=PACKAGE_ROOT/top
        if src.exists():
            shutil.copytree(
                src,
                overlay/top,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__","*.pyc",".DS_Store"),
            )
    (overlay/"scripts").mkdir(parents=True,exist_ok=True)
    for name in (
        SCRIPT_PATH.name,
        "install_into_dranmar.py",
        "requirements_atraumatic_exposure_generation.txt",
        "validate_dranmar_atraumatic_exposure_robot.py",
    ):
        source=PACKAGE_ROOT/"scripts"/name
        if source.exists():
            shutil.copy2(source,overlay/"scripts"/name)
    return overlay


def generate() -> dict[str,object]:
    for cache in sorted(PACKAGE_ROOT.rglob("__pycache__"),reverse=True):
        shutil.rmtree(cache)
    for junk in PACKAGE_ROOT.rglob(".DS_Store"):
        junk.unlink()
    for directory in (ASSET_ROOT,GLB_ROOT,TEXTURE_ROOT,PREVIEW_ROOT,DOCS_ROOT,EXAMPLE_ROOT,INTEGRATION_PATH.parent): directory.mkdir(parents=True,exist_ok=True)
    bundle=build_tool()
    assets=write_asset_files(bundle)
    textures=generate_textures()
    glbs=export_glbs(bundle)
    previews=[make_preview(bundle),make_full_arm_preview(bundle)]
    metadata=[
        write_json(ASSET_ROOT/"interaction_frames.json",interaction_frames(bundle)),
        write_json(ASSET_ROOT/"franka_mount_contract.json",mount_contract()),
        write_json(ASSET_ROOT/"surgical_exposure_task_contract.json",task_contract()),
        write_json(ASSET_ROOT/"physics_profile.json",physics_profile(bundle)),
        write_json(ASSET_ROOT/"collider_coverage.json",collider_coverage(bundle)),
    ]
    (ASSET_ROOT/"README.md").write_text(readme(),encoding="utf-8")
    shutil.copy2(ASSET_ROOT/"README.md",PACKAGE_ROOT/"README.md")
    (ASSET_ROOT/"LICENSE.txt").write_text("Copyright 2026 DrAnmar Project Developers\n\nLicensed under the Apache License, Version 2.0.\n",encoding="utf-8")
    (PACKAGE_ROOT/"LICENSE").write_text("Apache License 2.0\n",encoding="utf-8")
    (PACKAGE_ROOT/"NOTICE").write_text("DrAnmar Atraumatic Surgical Exposure Robot. Independently generated simulation-training asset.\n",encoding="utf-8")
    INTEGRATION_PATH.write_text(author_integration_module(),encoding="utf-8")
    (DOCS_ROOT/"MECHANISM.md").write_text(docs_mechanism(),encoding="utf-8")
    (DOCS_ROOT/"FORCE_CONTROL.md").write_text(docs_force_control(),encoding="utf-8")
    (DOCS_ROOT/"TISSUE_CAPTURE.md").write_text(docs_tissue(),encoding="utf-8")
    (DOCS_ROOT/"FRANKA_INTEGRATION.md").write_text(docs_franka(),encoding="utf-8")
    (EXAMPLE_ROOT/"franka_atraumatic_exposure_scene.py").write_text(example_scene(),encoding="utf-8")
    installer=PACKAGE_ROOT/"scripts/install_into_dranmar.py"; installer.write_text(author_installer(),encoding="utf-8"); installer.chmod(0o755)
    write_json(PACKAGE_ROOT/"physics_next/surgical-exposure/dranmar-atraumatic-exposure-robot-v1.json",physics_profile(bundle))
    sync_extension_data()
    manifest=ASSET_ROOT/"asset_manifest.json"; write_json(manifest,build_manifest(all_payload_files())); sync_extension_data()

    parent=PACKAGE_ROOT.parent
    dev=zip_tree(PACKAGE_ROOT,parent/"dranmar_atraumatic_exposure_robot_v0.1.0.zip",prefix=PACKAGE_ROOT.name)
    catalog=zip_tree(PACKAGE_ROOT/"assets",parent/"dranmar_atraumatic_exposure_robot_catalog_v0.1.0.zip")
    overlay=build_overlay(); overlay_zip=zip_tree(overlay,parent/"dranmar_atraumatic_exposure_robot_repo_overlay_v0.1.0.zip")
    release={"schema":"dr.anmar.release.v1","asset":ASSET_NAME,"version":VERSION,"catalog_subpath":CATALOG_SUBPATH.as_posix(),"development_package":{"path":str(dev),"sha256":sha256(dev)},"catalog_package":{"path":str(catalog),"sha256":sha256(catalog)},"repository_overlay":{"path":str(overlay_zip),"sha256":sha256(overlay_zip)},"main_assets":[p.name for p in assets],"glb_exports":[p.name for p in glbs],"previews":[p.name for p in previews],"mechanism":"bilateral distributed-capture force-controlled surgical exposure system","runtime_validation":"headless_cuda_qualification_script_included","clinical_validation":False,"medical_device":False}
    release_path=parent/"dranmar_atraumatic_exposure_robot_release_v0.1.0.json"; write_json(release_path,release)
    checksums=[write_checksum(p) for p in (dev,catalog,overlay_zip)]
    return {"asset_files":len(assets),"texture_files":len(textures),"glb_files":len(glbs),"package":str(dev),"catalog":str(catalog),"overlay":str(overlay_zip),"release":str(release_path),"checksums":[str(x) for x in checksums]}


def main() -> None:
    print(json.dumps(generate(),indent=2))


if __name__ == "__main__":
    main()
