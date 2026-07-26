#!/usr/bin/env python3
"""Generate the DrAnmar Adaptive Seal-and-Divide Robot asset family.

This DrAnmar-owned, provider-neutral research asset models a Franka-compatible
end effector for NVIDIA Isaac Sim and Isaac Lab that centers a vascular pedicle, applies
bilateral compression, forms two physical seal bands, verifies both seals,
retracts a blade guard, and divides the tissue between the protected stumps.
It is not clinically validated and is not approved for patient care.
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
ASSET_NAME = "DrAnmar Adaptive Seal-and-Divide Robot"
CATALOG_SUBPATH = Path("Props/SurgicalDivision/AdaptiveSealDivideRobot")
ROOT_PRIM = "DrAnmarAdaptiveSealDivideTool"
STANDALONE_ROOT = "DrAnmarAdaptiveSealDivideToolStandalone"
PROXY_ROOT = "DrAnmarAdaptiveSealDivideToolRigidProxy"
VESSEL_ROOT = "DrAnmarSealDivideVesselDemo"
SEAL_BAND_ROOT = "DrAnmarTissueSealBand"
BLADE_ROOT = "DrAnmarDivisionBladeCartridge"
VAPOR_ROOT = "DrAnmarSealVaporParticle"

SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[1]
ASSET_ROOT = PACKAGE_ROOT / "assets" / CATALOG_SUBPATH
GLB_ROOT = ASSET_ROOT / "glb"
TEXTURE_ROOT = ASSET_ROOT / "textures"
PREVIEW_ROOT = PACKAGE_ROOT / "previews"
DOCS_ROOT = PACKAGE_ROOT / "docs" / "adaptive_seal_divide_robot"
EXAMPLE_ROOT = PACKAGE_ROOT / "examples"
EXTENSION_ROOT = PACKAGE_ROOT / "source/extensions/orbit.surgical.assets"
INTEGRATION_PATH = EXTENSION_ROOT / "orbit/surgical/assets/adaptive_seal_divide_robot.py"
PHYSICS_PROFILE_PATH = PACKAGE_ROOT / "physics_next/surgical-division/dranmar-adaptive-seal-divide-v1.json"

WORK_PLANE_Z = 0.190
FRANKA_HAND_EQUIVALENT_ROTATION_DEG = -45.0
CENTERING_TRAVEL_M = 0.024
JAW_TRAVEL_M = 0.013
BLADE_TRAVEL_M = 0.041
GUARD_TRAVEL_M = 0.011
BRIDGE_PIN_COUNT = 8
SUCTION_PORT_COUNT = 8
IRRIGATION_PORT_COUNT = 6
SEAL_ZONE_X = 0.0065


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


def tube_wall_segment_mesh(x0: float, x1: float, outer_radius=0.0042, wall=0.00068, axial=16, radial=32) -> trimesh.Trimesh:
    """Create a closed hollow-wall segment whose longitudinal axis is X."""
    inner_radius = outer_radius - wall
    vertices: list[tuple[float,float,float]] = []
    for j in range(axial + 1):
        t = j / axial
        x = x0 + t*(x1-x0)
        cy = 0.00045*math.sin((x/(x1-x0+1e-12))*math.pi*0.65)
        cz = 0.00030*math.cos((x/(x1-x0+1e-12))*math.pi*0.5)
        for r in (outer_radius, inner_radius):
            for i in range(radial):
                a=2*math.pi*i/radial
                elliptic=1.0+0.035*math.cos(2*a)
                vertices.append((x, cy+r*elliptic*math.cos(a), cz+r*(2.0-elliptic)*math.sin(a)))
    faces: list[tuple[int,int,int]]=[]
    ring=2*radial
    for j in range(axial):
        b0=j*ring;b1=(j+1)*ring
        for i in range(radial):
            k=(i+1)%radial
            # outer wall
            faces += [(b0+i,b1+i,b1+k),(b0+i,b1+k,b0+k)]
            # inner wall reverse winding
            faces += [(b0+radial+i,b1+radial+k,b1+radial+i),(b0+radial+i,b0+radial+k,b1+radial+k)]
    # close the annular wall at both longitudinal ends
    for end_j, reverse in ((0,True),(axial,False)):
        b=end_j*ring
        for i in range(radial):
            k=(i+1)%radial
            o0,o1=b+i,b+k
            i0,i1=b+radial+i,b+radial+k
            if reverse:
                faces += [(o0,i1,i0),(o0,o1,i1)]
            else:
                faces += [(o0,i0,i1),(o0,i1,o1)]
    mesh=trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),process=False)
    mesh.remove_unreferenced_vertices();mesh.fix_normals()
    return mesh


def seal_band_mesh(width_x=0.0046, outer_y=0.0052, outer_z=0.0030, thickness=0.00055, sections=72) -> trimesh.Trimesh:
    """Create a flattened elliptical loop around the vessel, extruded along X."""
    vertices=[]
    for x in (-width_x/2,width_x/2):
        for shell in (0,1):
            ry=outer_y-shell*thickness
            rz=outer_z-shell*thickness*0.75
            for i in range(sections):
                a=2*math.pi*i/sections
                vertices.append((x,ry*math.cos(a),rz*math.sin(a)))
    faces=[]
    layer=2*sections
    # outer and inner longitudinal surfaces
    for side in (0,1):
        offset=side*sections
        for i in range(sections):
            j=(i+1)%sections
            a=offset+i;b=offset+j;c=layer+offset+j;d=layer+offset+i
            if side==0:faces += [(a,b,c),(a,c,d)]
            else:faces += [(a,c,b),(a,d,c)]
    # end annuli
    for end in (0,1):
        base=end*layer
        for i in range(sections):
            j=(i+1)%sections
            o0,o1=base+i,base+j;i0,i1=base+sections+i,base+sections+j
            if end==0:faces += [(o0,i1,i0),(o0,o1,i1)]
            else:faces += [(o0,i0,i1),(o0,i1,o1)]
    mesh=trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),process=False)
    mesh.remove_unreferenced_vertices();mesh.fix_normals();return mesh


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
    vessel_left: trimesh.Trimesh
    vessel_right: trimesh.Trimesh
    bridge_visual: trimesh.Trimesh
    seal_band: trimesh.Trimesh
    blade_cartridge: trimesh.Trimesh
    vessel_base: trimesh.Trimesh


def build_tool() -> ToolBundle:
    links: dict[str,Link] = {}
    mount_visuals: list[Visual] = [
        Visual("FrankaAdapterPlate", cylinder_axis(0.032,0.012,"z",(0,0,0.006),sections=72), "MountMetal", ("franka_mount",)),
        Visual("QuickReleaseRing", torus_axis(0.0275,0.003,"z",(0,0,0.014),major_sections=72,minor_sections=14), "MountMetal"),
        Visual("MainHousing", ellipsoid_mesh((0.055,0.047,0.034),(0,0,0.055),subdivisions=3), "BodyPolymer", ("adaptive_seal_divide_robot",)),
        Visual("HousingCore", box_mesh((0.104,0.084,0.052),(0,0,0.058)), "BodyPolymer"),
        Visual("CenteringRail", box_mesh((0.038,0.154,0.018),(0,0,0.112)), "RailMetal", ("tissue_centering_rail",)),
        Visual("JawTower", box_mesh((0.080,0.050,0.058),(0,0,0.125)), "DarkPolymer", ("seal_jaw_drive_housing",)),
        Visual("EnergyModule", box_mesh((0.044,0.072,0.030),(-0.040,0,0.090)), "AccentPolymer", ("adaptive_energy_module",)),
        Visual("BladeCassetteHousing", box_mesh((0.025,0.050,0.060),(0.041,0,0.105)), "DarkPolymer", ("division_blade_cassette",)),
        Visual("SuctionManifold", torus_axis(0.032,0.0030,"z",(0,0,0.166),major_sections=72,minor_sections=14), "DarkPolymer", ("annular_suction_manifold",)),
        Visual("IrrigationManifold", torus_axis(0.024,0.0015,"z",(0,0,0.169),major_sections=72,minor_sections=12), "SensorBlue", ("irrigation_manifold",)),
        Visual("SensorBridge", box_mesh((0.054,0.016,0.016),(0,-0.044,0.086)), "DarkPolymer", ("seal_verification_sensor_bridge",)),
        Visual("StereoCameraLeft", cylinder_axis(0.0048,0.004,"y",(-0.013,-0.053,0.084),sections=36), "SensorGlass", ("rgb_camera",)),
        Visual("StereoCameraRight", cylinder_axis(0.0048,0.004,"y",(0.013,-0.053,0.084),sections=36), "SensorGlass", ("rgb_camera",)),
        Visual("ThermalCamera", cylinder_axis(0.0042,0.004,"y",(0,-0.053,0.098),sections=36), "ThermalGlass", ("thermal_camera",)),
        Visual("ImpedanceMonitor", box_mesh((0.020,0.0012,0.010),(-0.037,-0.043,0.091)), "IndicatorAmber", ("impedance_monitor",)),
        Visual("SealReadyIndicator", box_mesh((0.018,0.0012,0.010),(-0.015,-0.043,0.091)), "IndicatorGreen", ("seal_ready_indicator",)),
        Visual("FaultIndicator", box_mesh((0.018,0.0012,0.010),(0.007,-0.043,0.091)), "IndicatorRed", ("seal_fault_indicator",)),
        Visual("SalineReservoir", cylinder_axis(0.016,0.038,"y",(0.026,0.038,0.058),sections=48), "TubeClear", ("irrigation_reservoir",)),
        Visual("SalineFill", cylinder_axis(0.0135,0.032,"y",(0.026,0.038,0.058),sections=48), "SalineBlue", ("irrigation_inventory",)),
        Visual("CollectionCanister", cylinder_axis(0.018,0.042,"y",(-0.022,0.038,0.058),sections=48), "TubeClear", ("vapor_collection_canister",)),
        Visual("CollectionFill", cylinder_axis(0.015,0.012,"y",(-0.022,0.052,0.058),sections=48), "CollectionDark", ("collected_vapor_condensate",)),
        Visual("LabelPanel", box_mesh((0.054,0.0012,0.023),(0,-0.047,0.052)), "LabelMaterial"),
    ]
    for i in range(SUCTION_PORT_COUNT):
        a=2*math.pi*i/SUCTION_PORT_COUNT
        x=0.032*math.cos(a);y=0.032*math.sin(a)
        mount_visuals.append(Visual(f"SuctionPort_{i:02d}",frustum_axis(0.0023,0.0013,0.008,"z",(x,y,0.172),sections=26),"DarkPolymer",("suction_port",)))
    for i in range(IRRIGATION_PORT_COUNT):
        a=2*math.pi*(i+0.5)/IRRIGATION_PORT_COUNT
        x=0.024*math.cos(a);y=0.024*math.sin(a)
        mount_visuals.append(Visual(f"IrrigationJet_{i:02d}",frustum_axis(0.0013,0.0005,0.007,"z",(x,y,0.173),sections=24),"SensorBlue",("irrigation_microjet",)))
    links["Mount"] = Link("Mount",(0,0,0),mount_visuals,[
        Collider("AdapterCollider","cylinder",(0,0,0.008),radius=0.032,height=0.016,physics_material="MountPhysics"),
        Collider("HousingCollider","box",(0,0,0.060),size=(0.116,0.096,0.078),physics_material="PolymerPhysics"),
        Collider("RailCollider","box",(0,0,0.112),size=(0.044,0.158,0.022),physics_material="MountPhysics"),
        Collider("JawTowerCollider","box",(0,0,0.127),size=(0.084,0.054,0.062),physics_material="PolymerPhysics"),
        Collider("SuctionRingCollider","cylinder",(0,0,0.166),radius=0.036,height=0.007,physics_material="PolymerPhysics",role="suction_ring"),
    ],0.480,("adaptive_seal_divide_end_effector","surgical_division_device"))

    for side_name,side in (("Left",-1),("Right",1)):
        y0=side*0.037
        tip_y=-side*0.018
        centering_visuals=[
            Visual("Carriage",box_mesh((0.030,0.030,0.023),(0,0,0)),"AccentPolymer",("tissue_centering_carriage",)),
            Visual("Arm",box_mesh((0.014,0.044,0.012),(0,tip_y/2,0.020)),"RailMetal",("centering_arm",)),
            Visual("CompliantTip",ellipsoid_mesh((0.010,0.007,0.005),(0,tip_y,0.031),subdivisions=3),"PadElastomer",("atraumatic_centering_contact",)),
            Visual("ForceWindow",box_mesh((0.014,0.0012,0.005),(0,-side*0.015,0.002)),"IndicatorGreen",("centering_force_indicator",)),
        ]
        links[f"{side_name}Centering"] = Link(f"{side_name}Centering",(0,y0,0.137),centering_visuals,[
            Collider("CarriageCollider","box",(0,0,0),size=(0.032,0.032,0.025),physics_material="PolymerPhysics"),
            Collider("ArmCollider","box",(0,tip_y/2,0.020),size=(0.016,0.046,0.014),physics_material="MetalPhysics"),
            Collider("TissueCenteringContact","sphere",(0,tip_y,0.031),radius=0.007,physics_material="PadContactPhysics",role="tissue_centering_contact"),
        ],0.060,("atraumatic_tissue_centering_finger",))

    # Upper and lower seal jaws are symmetric and keep a physical blade channel at X=0.
    # Positive local Z points from the upper jaw toward the tissue plane;
    # negative local Z points from the lower jaw toward that same plane.
    # Keep every electrode, pressure rail, and contact volume on the
    # tissue-facing side of its jaw.
    for jaw_name,sign in (("UpperJaw",1),("LowerJaw",-1)):
        face_z=sign*0.0060
        jaw_visuals=[
            Visual("JawBody",box_mesh((0.056,0.030,0.009),(0,0,0)),"JawMetal",("seal_jaw",)),
            Visual("CeramicInsulator",box_mesh((0.052,0.024,0.003),(0,0,face_z)),"CeramicWhite",("electrical_insulator",)),
            Visual("LeftElectrode",box_mesh((0.0045,0.021,0.0012),(-SEAL_ZONE_X,0,face_z+sign*0.0020)),"ElectrodeCopper",("left_seal_electrode",)),
            Visual("RightElectrode",box_mesh((0.0045,0.021,0.0012),(SEAL_ZONE_X,0,face_z+sign*0.0020)),"ElectrodeCopper",("right_seal_electrode",)),
            Visual("LeftPressureRail",box_mesh((0.0030,0.022,0.0010),(-0.013,0,face_z+sign*0.0015)),"PadElastomer",("pressure_distribution_rail",)),
            Visual("RightPressureRail",box_mesh((0.0030,0.022,0.0010),(0.013,0,face_z+sign*0.0015)),"PadElastomer",("pressure_distribution_rail",)),
            Visual("BladeSlotLeft",box_mesh((0.0010,0.024,0.0010),(-0.0014,0,face_z+sign*0.0010)),"DarkPolymer",("blade_channel",)),
            Visual("BladeSlotRight",box_mesh((0.0010,0.024,0.0010),(0.0014,0,face_z+sign*0.0010)),"DarkPolymer",("blade_channel",)),
            Visual("TemperatureStrip",box_mesh((0.020,0.0012,0.004),(0,-0.0155,-face_z*0.5)),"ThermalGlass",("jaw_temperature_sensor",)),
        ]
        jaw_colliders=[
            Collider("JawBackCollider","box",(0,0,-sign*0.001),size=(0.058,0.032,0.008),physics_material="MetalPhysics"),
            Collider("LeftSealContact","box",(-SEAL_ZONE_X,0,face_z+sign*0.0015),size=(0.0050,0.022,0.0025),physics_material="ElectrodePhysics",role="left_seal_contact"),
            Collider("RightSealContact","box",(SEAL_ZONE_X,0,face_z+sign*0.0015),size=(0.0050,0.022,0.0025),physics_material="ElectrodePhysics",role="right_seal_contact"),
            Collider("LeftPressureContact","box",(-0.013,0,face_z+sign*0.0013),size=(0.0032,0.023,0.0022),physics_material="PadContactPhysics",role="jaw_pressure_contact"),
            Collider("RightPressureContact","box",(0.013,0,face_z+sign*0.0013),size=(0.0032,0.023,0.0022),physics_material="PadContactPhysics",role="jaw_pressure_contact"),
        ]
        z0=0.172 if jaw_name=="UpperJaw" else 0.208
        links[jaw_name]=Link(jaw_name,(0,0,z0),jaw_visuals,jaw_colliders,0.090,("dual_zone_seal_jaw",))

    guard_visuals=[
        Visual("GuardFrame",box_mesh((0.010,0.028,0.034),(0,0,0)),"CeramicWhite",("blade_guard",)),
        Visual("GuardWindow",box_mesh((0.004,0.020,0.028),(0,0,0.004)),"DarkPolymer",("blade_guard_window",)),
        Visual("GuardStatus",box_mesh((0.006,0.0012,0.006),(0,-0.0145,-0.010)),"IndicatorGreen",("blade_guard_status",)),
    ]
    links["BladeGuard"] = Link("BladeGuard",(0,0,0.163),guard_visuals,[
        Collider("GuardLeft","box",(-0.004,0,0),size=(0.0025,0.030,0.036),physics_material="CeramicPhysics"),
        Collider("GuardRight","box",(0.004,0,0),size=(0.0025,0.030,0.036),physics_material="CeramicPhysics"),
    ],0.038,("blade_safety_guard",))

    blade_mesh=wedge_blade_mesh()
    blade_visuals=[
        Visual("BladeCarrier",box_mesh((0.016,0.030,0.040),(0,0,-0.006)),"CeramicWhite",("division_blade_carrier",)),
        Visual("Blade",transform(blade_mesh,(0,0,0.023)),"BladeSteel",("central_division_blade",)),
        Visual("BladeFreshIndicator",box_mesh((0.009,0.0012,0.007),(0,-0.0155,-0.012)),"IndicatorGreen",("fresh_blade_indicator",)),
        Visual("BladeSpentIndicator",box_mesh((0.009,0.0012,0.007),(0,-0.0155,-0.012)),"IndicatorRed",("spent_blade_indicator",)),
    ]
    links["BladeCarriage"] = Link("BladeCarriage",(0,0,0.142),blade_visuals,[
        Collider("CarrierCollider","box",(0,0,-0.006),size=(0.018,0.032,0.042),physics_material="CeramicPhysics"),
        Collider("BladeCollider","box",(0,0,0.020),size=(0.0012,0.023,0.025),physics_material="BladePhysics",role="division_blade_contact"),
    ],0.058,("interlocked_division_blade",))

    for name,side,material in (("SuctionValve",1,"DarkPolymer"),("IrrigationValve",-1,"SensorBlue")):
        visuals=[Visual("ValveStem",cylinder_axis(0.0045,0.024,"y",(0,0,0),sections=36),material,(name.lower(),)),Visual("ValveTab",box_mesh((0.014,0.006,0.010),(0,side*0.012,0)),material)]
        links[name]=Link(name,(side*0.026,0.032,0.082),visuals,[Collider("ValveCollider","cylinder",(0,0,0),radius=0.005,height=0.026,axis="y",physics_material="PolymerPhysics")],0.018,(name.lower(),))

    joints=[
        Joint("left_centering_joint","prismatic","Mount","LeftCentering","Y",(0,-0.037,0.137),(0,0,0),0.0,CENTERING_TRAVEL_M,4200,145,90),
        Joint("right_centering_joint","prismatic","Mount","RightCentering","Y",(0,0.037,0.137),(0,0,0),-CENTERING_TRAVEL_M,0.0,4200,145,90),
        Joint("upper_jaw_joint","prismatic","Mount","UpperJaw","Z",(0,0,0.172),(0,0,0),0.0,JAW_TRAVEL_M,18000,420,360),
        Joint("lower_jaw_joint","prismatic","Mount","LowerJaw","Z",(0,0,0.208),(0,0,0),-JAW_TRAVEL_M,0.0,18000,420,360),
        Joint("blade_guard_joint","prismatic","Mount","BladeGuard","Z",(0,0,0.163),(0,0,0),-GUARD_TRAVEL_M,0.0,8500,220,160),
        Joint("blade_joint","prismatic","Mount","BladeCarriage","Z",(0,0,0.142),(0,0,0),0.0,BLADE_TRAVEL_M,16000,360,280),
        Joint("suction_valve_joint","prismatic","Mount","SuctionValve","Y",(0.026,0.032,0.082),(0,0,0),0.0,0.008,1800,55,30),
        Joint("irrigation_valve_joint","prismatic","Mount","IrrigationValve","Y",(-0.026,0.032,0.082),(0,0,0),0.0,0.006,1800,55,30),
    ]

    frames: dict[str,dict[str,object]]={
        "panda_link8_mount":{"parent_link":"Mount","position":[0,0,0],"orientation_wxyz":[1,0,0,0],"role":"franka_wrist_mount"},
        "seal_divide_tcp":{"parent_link":"Mount","position":[0,0,WORK_PLANE_Z],"orientation_wxyz":[1,0,0,0],"role":"seal_divide_tool_center_point"},
        "tissue_center_reference":{"parent_link":"Mount","position":[0,0,WORK_PLANE_Z],"orientation_wxyz":[1,0,0,0],"role":"tissue_center_reference"},
        "left_seal_zone":{"parent_link":"Mount","position":[-SEAL_ZONE_X,0,WORK_PLANE_Z],"orientation_wxyz":[1,0,0,0],"role":"left_seal_band_center"},
        "right_seal_zone":{"parent_link":"Mount","position":[SEAL_ZONE_X,0,WORK_PLANE_Z],"orientation_wxyz":[1,0,0,0],"role":"right_seal_band_center"},
        "cut_plane":{"parent_link":"Mount","position":[0,0,WORK_PLANE_Z],"orientation_wxyz":[1,0,0,0],"role":"central_division_plane"},
        "suction_center":{"parent_link":"Mount","position":[0,0,0.176],"orientation_wxyz":[1,0,0,0],"role":"annular_suction_center"},
        "irrigation_center":{"parent_link":"Mount","position":[0,0,0.177],"orientation_wxyz":[1,0,0,0],"role":"irrigation_center"},
        "thermal_camera":{"parent_link":"Mount","position":[0,-0.053,0.098],"orientation_wxyz":[0.70710678,0.70710678,0,0],"role":"thermal_camera_optical_frame"},
        "impedance_probe":{"parent_link":"Mount","position":[0,0,0.184],"orientation_wxyz":[1,0,0,0],"role":"impedance_measurement_reference"},
        "seal_verification_probe":{"parent_link":"Mount","position":[0,0,0.180],"orientation_wxyz":[1,0,0,0],"role":"seal_verification_reference"},
        "count_reference":{"parent_link":"Mount","position":[0,0,0.040],"orientation_wxyz":[1,0,0,0],"role":"instrument_count_reference"},
        "disposal_reference":{"parent_link":"Mount","position":[0,0,0.020],"orientation_wxyz":[1,0,0,0],"role":"disposal_reference"},
        "left_centering_contact":{"parent_link":"LeftCentering","position":[0,0.018,0.031],"orientation_wxyz":[1,0,0,0],"role":"left_centering_contact"},
        "right_centering_contact":{"parent_link":"RightCentering","position":[0,-0.018,0.031],"orientation_wxyz":[1,0,0,0],"role":"right_centering_contact"},
        "upper_jaw_contact":{"parent_link":"UpperJaw","position":[0,0,0.008],"orientation_wxyz":[1,0,0,0],"role":"upper_jaw_contact"},
        "lower_jaw_contact":{"parent_link":"LowerJaw","position":[0,0,-0.008],"orientation_wxyz":[1,0,0,0],"role":"lower_jaw_contact"},
        "blade_tip":{"parent_link":"BladeCarriage","position":[0,0,0.034],"orientation_wxyz":[1,0,0,0],"role":"division_blade_tip"},
        "blade_guard_reference":{"parent_link":"BladeGuard","position":[0,0,0.016],"orientation_wxyz":[1,0,0,0],"role":"blade_guard_reference"},
    }

    vessel_left=tube_wall_segment_mesh(-0.042,-0.00020)
    vessel_right=tube_wall_segment_mesh(0.00020,0.042)
    bridge_visual=tube_wall_segment_mesh(-0.00055,0.00055,outer_radius=0.00415,wall=0.00066,axial=4,radial=64)
    seal_band=seal_band_mesh()
    blade_cartridge=trimesh.util.concatenate([box_mesh((0.018,0.032,0.042),(0,0,-0.006)),transform(blade_mesh,(0,0,0.023))])
    vessel_base=box_mesh((0.105,0.050,0.010),(0,0,-0.010))
    return ToolBundle(links,joints,frames,vessel_left,vessel_right,bridge_visual,seal_band,blade_cartridge,vessel_base)


def visual_materials_scope(root: str) -> str:
    specs={
        "BodyPolymer":((0.82,0.85,0.88),0.0,0.34,1.0),
        "AccentPolymer":((0.055,0.31,0.62),0.0,0.29,1.0),
        "DarkPolymer":((0.035,0.045,0.055),0.0,0.30,1.0),
        "MountMetal":((0.44,0.48,0.53),0.82,0.24,1.0),
        "RailMetal":((0.25,0.29,0.34),0.74,0.26,1.0),
        "JawMetal":((0.60,0.64,0.68),0.78,0.22,1.0),
        "ElectrodeCopper":((0.78,0.40,0.13),0.88,0.20,1.0),
        "CeramicWhite":((0.91,0.91,0.88),0.0,0.16,1.0),
        "BladeSteel":((0.70,0.73,0.77),0.92,0.12,1.0),
        "SpentBlade":((0.32,0.31,0.30),0.72,0.42,1.0),
        "PadElastomer":((0.06,0.53,0.61),0.0,0.52,1.0),
        "SensorGlass":((0.05,0.16,0.24),0.15,0.10,0.75),
        "ThermalGlass":((0.31,0.05,0.43),0.18,0.13,0.85),
        "SensorBlue":((0.04,0.39,0.86),0.05,0.20,1.0),
        "IndicatorGreen":((0.05,0.86,0.28),0.0,0.20,1.0),
        "IndicatorAmber":((1.0,0.52,0.02),0.0,0.23,1.0),
        "IndicatorRed":((0.92,0.03,0.04),0.0,0.22,1.0),
        "SalineBlue":((0.04,0.46,0.92),0.0,0.05,0.56),
        "CollectionDark":((0.31,0.12,0.08),0.0,0.16,0.72),
        "TubeClear":((0.63,0.80,0.88),0.0,0.08,0.34),
        "LabelMaterial":((0.96,0.97,0.98),0.0,0.40,1.0),
        "VesselMaterial":((0.52,0.055,0.045),0.0,0.44,1.0),
        "VesselInner":((0.74,0.16,0.13),0.0,0.39,1.0),
        "BridgeMaterial":((0.59,0.075,0.060),0.0,0.42,1.0),
        "SealBandFresh":((0.94,0.62,0.10),0.15,0.24,0.62),
        "SealBandMature":((0.47,0.21,0.05),0.15,0.30,0.78),
        "SealBandFailed":((0.20,0.08,0.04),0.05,0.58,0.58),
        "FixtureMaterial":((0.08,0.09,0.11),0.20,0.42,1.0),
        "VaporMaterial":((0.78,0.84,0.88),0.0,0.08,0.30),
    }
    blocks=[]
    for name,(color,metallic,roughness,opacity) in specs.items():
        blocks.append(f'''        def Material "{name}"
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
        }}''')
    return '    def Scope "Looks"\n    {\n'+'\n'.join(blocks)+'\n    }'


def physics_materials_scope() -> str:
    specs={
        "MountPhysics":(0.34,0.26,0.02),
        "PolymerPhysics":(0.48,0.38,0.03),
        "MetalPhysics":(0.28,0.21,0.02),
        "PadContactPhysics":(0.72,0.58,0.01),
        "ElectrodePhysics":(0.44,0.34,0.01),
        "CeramicPhysics":(0.32,0.25,0.01),
        "BladePhysics":(0.20,0.14,0.00),
        "SealBandPhysics":(0.58,0.44,0.00),
        "VesselPhysics":(0.52,0.40,0.00),
    }
    blocks=[]
    for name,(static,dynamic,restitution) in specs.items():
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
    return '    def Scope "PhysicsMaterials"\n    {\n'+'\n'.join(blocks)+'\n    }'


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
    common=f'''{indent}    custom string drAnmar:role = "{collider.role}"
{binding}{indent}    bool physics:collisionEnabled = true
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
    api=f'prepend apiSchemas = ["PhysicsDriveAPI:{drive}"]' if drive else ''
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
    def edits(link: str, values: Sequence[tuple[str, str, str]]) -> str:
        lines = [
            '            over "Links"',
            "            {",
            f'                over "{link}"',
            "                {",
            '                    over "Visuals"',
            "                    {",
        ]
        for prim, attribute, value in values:
            lines.extend(
                [
                    f'                        over "{prim}"',
                    "                        {",
                    f"                            {attribute} = {value}",
                    "                        }",
                ]
            )
        lines.extend(["                    }", "                }", "            }"])
        return "\n".join(lines)

    fresh=edits("BladeCarriage",[
        ("BladeFreshIndicator","token visibility",'"inherited"'),
        ("BladeSpentIndicator","token visibility",'"invisible"'),
        ("Blade","rel material:binding",f"</{ROOT_PRIM}/Looks/BladeSteel>"),
    ])
    spent=edits("BladeCarriage",[
        ("BladeFreshIndicator","token visibility",'"invisible"'),
        ("BladeSpentIndicator","token visibility",'"inherited"'),
        ("Blade","rel material:binding",f"</{ROOT_PRIM}/Looks/SpentBlade>"),
    ])
    saline_full=edits("Mount",[("SalineFill","token visibility",'"inherited"')])
    saline_empty=edits("Mount",[("SalineFill","token visibility",'"invisible"')])
    collection_empty=edits("Mount",[("CollectionFill","token visibility",'"invisible"')])
    collection_visible=edits("Mount",[("CollectionFill","token visibility",'"inherited"')])
    ready=edits("Mount",[
        ("SealReadyIndicator","token visibility",'"inherited"'),
        ("FaultIndicator","token visibility",'"invisible"'),
    ])
    fault=edits("Mount",[
        ("SealReadyIndicator","token visibility",'"invisible"'),
        ("FaultIndicator","token visibility",'"inherited"'),
    ])
    return f'''    variantSet "cartridge_state" = {{
        "fresh"
        {{
{fresh}
        }}
        "spent"
        {{
{spent}
        }}
    }}
    variantSet "saline_state" = {{
        "full"
        {{
{saline_full}
        }}
        "empty"
        {{
{saline_empty}
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
{ready}
        }}
        "fault"
        {{
{fault}
        }}
    }}'''


def tool_usda(bundle: ToolBundle, articulation_root: bool) -> str:
    root=STANDALONE_ROOT if articulation_root else ROOT_PRIM
    root_path=f"/{root}"
    schemas=(
        '    prepend apiSchemas = ["PhysicsArticulationRootAPI"]'
        if articulation_root else ""
    )
    links="\n\n".join(link_usda(link,root_path,bundle.frames) for link in bundle.links.values())
    joints="\n\n".join(joint_usda(j,root_path) for j in bundle.joints)
    variants=state_variants().replace(f"/{ROOT_PRIM}/",f"/{root}/")
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "{ASSET_NAME}: tissue centering, dual-zone compression sealing, interlocked division, and stump leak verification research asset."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
{schemas}
    prepend variantSets = ["cartridge_state", "saline_state", "collection_state", "energy_state"]
    variants = {{
        string cartridge_state = "fresh"
        string saline_state = "full"
        string collection_state = "empty"
        string energy_state = "ready"
    }}
    customData = {{
        string drAnmarAssetId = "dranmar-adaptive-seal-divide-robot-v1"
        string drAnmarAssetVersion = "{VERSION}"
        bool drAnmarClinicalValidation = false
        bool drAnmarMedicalDevice = false
        string drAnmarStatus = "simulation_training_workcell"
        string drAnmarMount = "replaces_panda_hand_at_panda_link8"
        int drAnmarSealZoneCount = 2
        bool drAnmarBladeInterlockRequired = true
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


def material_color(material: str) -> tuple[int,int,int,int]:
    colors={
        "BodyPolymer":(211,218,224,255),"AccentPolymer":(18,83,158,255),"DarkPolymer":(12,17,22,255),
        "MountMetal":(118,130,142,255),"RailMetal":(70,80,94,255),"JawMetal":(154,163,173,255),
        "ElectrodeCopper":(202,105,33,255),"CeramicWhite":(235,235,228,255),"BladeSteel":(186,193,202,255),
        "SpentBlade":(90,85,80,255),"PadElastomer":(20,160,178,255),"SensorGlass":(18,52,78,215),
        "ThermalGlass":(100,18,136,235),"SensorBlue":(18,104,220,255),"IndicatorGreen":(25,224,78,255),
        "IndicatorAmber":(255,136,8,255),"IndicatorRed":(235,15,18,255),"SalineBlue":(18,120,235,150),
        "CollectionDark":(90,35,20,190),"TubeClear":(185,220,232,90),"LabelMaterial":(245,248,250,255),
        "VesselMaterial":(135,18,17,255),"VesselInner":(185,50,42,255),"BridgeMaterial":(156,28,24,255),
        "SealBandFresh":(240,155,28,170),"SealBandMature":(130,57,12,215),"SealBandFailed":(65,22,10,150),
        "FixtureMaterial":(20,23,28,255),"VaporMaterial":(205,220,230,70),"CollisionDebug":(255,60,15,95),
        "GuideRed":(255,30,30,255),"GuideGreen":(30,255,30,255),"GuideBlue":(30,80,255,255),
    }
    return colors.get(material,(180,180,180,255))


def pbr(mesh: trimesh.Trimesh, material: str) -> trimesh.Trimesh:
    m=mesh.copy();m.visual.vertex_colors=np.tile(np.asarray(material_color(material),dtype=np.uint8),(len(m.vertices),1));return m


def rigid_proxy_usda(bundle: ToolBundle) -> str:
    visuals=[]
    for link in bundle.links.values():
        for v in link.visuals:
            visuals.append(Visual(f"{link.name}_{v.name}",transform(v.mesh,link.translation),v.material,v.labels))
    bmin,bmax=mesh_bounds([v.mesh for v in visuals]);mp=box_mass_properties([v.mesh for v in visuals],0.82)
    blocks="\n".join(mesh_usda(v,f"/{PROXY_ROOT}/Looks/{v.material}",indent="        ") for v in visuals)
    size=bmax-bmin;center=(bmin+bmax)/2
    return f'''#usda 1.0
(
    defaultPrim = "{PROXY_ROOT}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{PROXY_ROOT}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    customData = {{
        bool drAnmarClinicalValidation = false
        string drAnmarRepresentation = "rigid_perception_planning_proxy"
    }}
)
{{
    float physics:mass = {f(mp['mass_kg'])}
    point3f physics:centerOfMass = {vec(mp['center_of_mass_m'])}
    vector3f physics:diagonalInertia = {vec(mp['diagonal_inertia_kg_m2'])}
    quatf physics:principalAxes = (1, 0, 0, 0)
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(PROXY_ROOT)}
{physics_materials_scope()}
    def Scope "Visuals"
    {{
{blocks}
    }}
    def Cube "Collision" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
    )
    {{
        rel material:binding:physics = </{PROXY_ROOT}/PhysicsMaterials/PolymerPhysics>
        double size = 1
        double3 xformOp:translate = {vec(center)}
        double3 xformOp:scale = {vec(size)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        token visibility = "invisible"
        uniform token purpose = "guide"
    }}
}}
'''


def mesh_usda_unbound(name: str, mesh: trimesh.Trimesh, labels: tuple[str,...]=(), indent: str="    ") -> str:
    v=Visual(name,mesh,"",labels)
    text=mesh_usda(v,"/__UNBOUND__",indent=indent)
    text=text.replace(f'{indent}    rel material:binding = </__UNBOUND__>\n',"")
    return text


def vessel_usda(bundle: ToolBundle) -> str:
    left=mesh_usda(Visual("LeftVesselWall",bundle.vessel_left,"VesselMaterial",("left_vessel_segment","deformable_tissue")),f"/{VESSEL_ROOT}/Looks/VesselMaterial",indent="    ")
    right=mesh_usda(Visual("RightVesselWall",bundle.vessel_right,"VesselMaterial",("right_vessel_segment","deformable_tissue")),f"/{VESSEL_ROOT}/Looks/VesselMaterial",indent="    ")
    bridge=mesh_usda(Visual("BridgeVisual",bundle.bridge_visual,"BridgeMaterial",("predivision_bridge_visual",)),f"/{VESSEL_ROOT}/Looks/BridgeMaterial",indent="    ")
    base=mesh_usda(Visual("FixtureBase",bundle.vessel_base,"FixtureMaterial",("fixture_base",)),f"/{VESSEL_ROOT}/Looks/FixtureMaterial",indent="    ")
    pins=[]
    pin_positions=[]
    for i in range(BRIDGE_PIN_COUNT):
        a=2*math.pi*i/BRIDGE_PIN_COUNT
        y=0.00315*math.cos(a);z=0.00255*math.sin(a)
        pin_positions.append([0.0,y,z])
        pins.append(f'''        def Xform "BridgePin_{i:02d}" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
        )
        {{
            bool physics:kinematicEnabled = true
            float physics:mass = 0.00002
            double3 xformOp:translate = {vec((0,y,z))}
            uniform token[] xformOpOrder = ["xformOp:translate"]
            def Cube "Capture" (
                prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
            )
            {{
                rel material:binding:physics = </{VESSEL_ROOT}/PhysicsMaterials/VesselPhysics>
                custom string drAnmar:role = "cut_bridge_capture"
                double size = 1
                double3 xformOp:scale = (0.0014, 0.00135, 0.00135)
                uniform token[] xformOpOrder = ["xformOp:scale"]
                token visibility = "invisible"
                uniform token purpose = "guide"
                bool physics:collisionEnabled = false
                float physxCollision:contactOffset = 0.0002
                float physxCollision:restOffset = 0
            }}
        }}''')
    pin_blocks="\n".join(pins)
    regions=f'''    def Scope "Regions"
    {{
        def Cube "LeftSealZone"
        {{
            custom string drAnmar:role = "left_seal_zone"
            double size = 1
            double3 xformOp:translate = ({f(-SEAL_ZONE_X)}, 0, 0)
            double3 xformOp:scale = (0.0050, 0.0110, 0.0090)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            token visibility = "invisible"
            uniform token purpose = "guide"
        }}
        def Cube "RightSealZone"
        {{
            custom string drAnmar:role = "right_seal_zone"
            double size = 1
            double3 xformOp:translate = ({f(SEAL_ZONE_X)}, 0, 0)
            double3 xformOp:scale = (0.0050, 0.0110, 0.0090)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            token visibility = "invisible"
            uniform token purpose = "guide"
        }}
        def Cube "CutRegion"
        {{
            custom string drAnmar:role = "division_region"
            double size = 1
            double3 xformOp:scale = (0.0014, 0.0120, 0.0100)
            uniform token[] xformOpOrder = ["xformOp:scale"]
            token visibility = "invisible"
            uniform token purpose = "guide"
        }}
    }}'''
    return f'''#usda 1.0
(
    defaultPrim = "{VESSEL_ROOT}"
    doc = "DrAnmar two-body hollow vessel division demo with removable mechanical bridge pins and two future stump seal zones."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{VESSEL_ROOT}" (
    customData = {{
        string drAnmarAssetId = "dranmar-seal-divide-vessel-demo-v1"
        bool drAnmarClinicalValidation = false
        string drAnmarRepresentation = "two_watertight_surface_deformable_halves_with_runtime_bridge_attachments"
        int drAnmarBridgePinCount = {BRIDGE_PIN_COUNT}
        string drAnmarFlowModel = "reduced_order_external_to_surface_solver"
    }}
)
{{
{visual_materials_scope(VESSEL_ROOT)}
{physics_materials_scope()}
{left}
{right}
{bridge}
{base}
    def Scope "BridgePins"
    {{
{pin_blocks}
    }}
    def Cube "LeftFixtureAnchor" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysxRigidBodyAPI"]
    )
    {{
        bool physics:kinematicEnabled = true
        bool physics:collisionEnabled = false
        double size = 1
        double3 xformOp:translate = (-0.040, 0, 0)
        double3 xformOp:scale = (0.004, 0.012, 0.012)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        token visibility = "invisible"
        uniform token purpose = "guide"
    }}
    def Cube "RightFixtureAnchor" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysxRigidBodyAPI"]
    )
    {{
        bool physics:kinematicEnabled = true
        bool physics:collisionEnabled = false
        double size = 1
        double3 xformOp:translate = (0.040, 0, 0)
        double3 xformOp:scale = (0.004, 0.012, 0.012)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        token visibility = "invisible"
        uniform token purpose = "guide"
    }}
{regions}
    def Scope "Frames"
    {{
        def Xform "left_seal_zone"
        {{
            custom string drAnmar:role = "left_stump_seal_center"
            double3 xformOp:translate = ({f(-SEAL_ZONE_X)}, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "right_seal_zone"
        {{
            custom string drAnmar:role = "right_stump_seal_center"
            double3 xformOp:translate = ({f(SEAL_ZONE_X)}, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "cut_center"
        {{
            custom string drAnmar:role = "division_center"
        }}
        def Xform "left_stump_probe"
        {{
            custom string drAnmar:role = "left_stump_flow_probe"
            double3 xformOp:translate = (-0.003, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "right_stump_probe"
        {{
            custom string drAnmar:role = "right_stump_flow_probe"
            double3 xformOp:translate = (0.003, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "anchor_left"
        {{
            custom string drAnmar:role = "fixture_anchor"
            double3 xformOp:translate = (-0.040, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "anchor_right"
        {{
            custom string drAnmar:role = "fixture_anchor"
            double3 xformOp:translate = (0.040, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}
}}
'''


def seal_band_usda(bundle: ToolBundle) -> str:
    mesh_block=mesh_usda_unbound("BandVisual",bundle.seal_band,("tissue_seal_band",),indent="    ")
    fresh_status=mesh_usda(Visual("FreshStatus",box_mesh((0.0010,0.0020,0.0010),(0,0.0055,0.0032)),"IndicatorAmber"),f"/{SEAL_BAND_ROOT}/Looks/IndicatorAmber",indent="    ")
    mature_status=mesh_usda(Visual("MatureStatus",box_mesh((0.0010,0.0020,0.0010),(0,0.0055,0.0032)),"IndicatorGreen"),f"/{SEAL_BAND_ROOT}/Looks/IndicatorGreen",indent="    ")
    failed_status=mesh_usda(Visual("FailedStatus",box_mesh((0.0010,0.0020,0.0010),(0,0.0055,0.0032)),"IndicatorRed"),f"/{SEAL_BAND_ROOT}/Looks/IndicatorRed",indent="    ")
    def bind(material):return nested_over(["BandVisual"],[f'rel material:binding = </{SEAL_BAND_ROOT}/Looks/{material}>'],indent="            ")
    def vis(name,value):return nested_over([name],[f'token visibility = "{value}"'],indent="            ")
    fresh="\n".join([bind("SealBandFresh"),vis("FreshStatus","inherited"),vis("MatureStatus","invisible"),vis("FailedStatus","invisible")])
    mature="\n".join([bind("SealBandMature"),vis("FreshStatus","invisible"),vis("MatureStatus","inherited"),vis("FailedStatus","invisible")])
    failed="\n".join([bind("SealBandFailed"),vis("FreshStatus","invisible"),vis("MatureStatus","invisible"),vis("FailedStatus","inherited")])
    return f'''#usda 1.0
(
    defaultPrim = "{SEAL_BAND_ROOT}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{SEAL_BAND_ROOT}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    prepend variantSets = "state"
    variants = {{ string state = "fresh" }}
    customData = {{
        string drAnmarAssetId = "dranmar-tissue-seal-band-v1"
        bool drAnmarClinicalValidation = false
        string drAnmarRepresentation = "rigid_bond_carrier_with_two_deformable_attachment_regions"
    }}
)
{{
    float physics:mass = 0.0009
    point3f physics:centerOfMass = (0, 0, 0)
    vector3f physics:diagonalInertia = (1.2e-8, 2.0e-8, 2.0e-8)
    quatf physics:principalAxes = (1, 0, 0, 0)
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(SEAL_BAND_ROOT)}
{mesh_block}
{fresh_status}
{mature_status}
{failed_status}
    def Scope "Collisions"
    {{
        def Cube "TopBandCollider" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = </{SEAL_BAND_ROOT}/PhysicsMaterials/SealBandPhysics>
            double size = 1
            double3 xformOp:translate = (0, 0, 0.0027)
            double3 xformOp:scale = (0.0048, 0.0102, 0.0007)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            token visibility = "invisible"
            uniform token purpose = "guide"
        }}
        def Cube "BottomBandCollider" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = </{SEAL_BAND_ROOT}/PhysicsMaterials/SealBandPhysics>
            double size = 1
            double3 xformOp:translate = (0, 0, -0.0027)
            double3 xformOp:scale = (0.0048, 0.0102, 0.0007)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            token visibility = "invisible"
            uniform token purpose = "guide"
        }}
        def Cube "UpperBondVolume" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = </{SEAL_BAND_ROOT}/PhysicsMaterials/SealBandPhysics>
            custom string drAnmar:role = "upper_tissue_bond_volume"
            double size = 1
            double3 xformOp:translate = (0, 0, 0.0015)
            double3 xformOp:scale = (0.0052, 0.0090, 0.0022)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            token visibility = "invisible"
            uniform token purpose = "guide"
        }}
        def Cube "LowerBondVolume" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding:physics = </{SEAL_BAND_ROOT}/PhysicsMaterials/SealBandPhysics>
            custom string drAnmar:role = "lower_tissue_bond_volume"
            double size = 1
            double3 xformOp:translate = (0, 0, -0.0015)
            double3 xformOp:scale = (0.0052, 0.0090, 0.0022)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            token visibility = "invisible"
            uniform token purpose = "guide"
        }}
    }}
    def Scope "Frames"
    {{
        def Xform "seal_center"
        {{
            custom string drAnmar:role = "seal_center"
        }}
        def Xform "upper_bond"
        {{
            custom string drAnmar:role = "upper_bond_reference"
            double3 xformOp:translate = (0, 0, 0.0015)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "lower_bond"
        {{
            custom string drAnmar:role = "lower_bond_reference"
            double3 xformOp:translate = (0, 0, -0.0015)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}
    variantSet "state" = {{
        "fresh"
        {{
{fresh}
        }}
        "mature"
        {{
{mature}
        }}
        "failed"
        {{
{failed}
        }}
    }}
}}
'''


def blade_cartridge_usda(bundle: ToolBundle) -> str:
    carrier=box_mesh((0.018,0.032,0.042),(0,0,-0.006))
    blade=wedge_blade_mesh()
    carrier_block=mesh_usda(Visual("Carrier",carrier,"CeramicWhite",("replaceable_blade_cartridge",)),f"/{BLADE_ROOT}/Looks/CeramicWhite",indent="    ")
    blade_block=mesh_usda_unbound("Blade",transform(blade,(0,0,0.023)),("division_blade",),indent="    ")
    def bind(material):return nested_over(["Blade"],[f'rel material:binding = </{BLADE_ROOT}/Looks/{material}>'],indent="            ")
    return f'''#usda 1.0
(
    defaultPrim = "{BLADE_ROOT}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{BLADE_ROOT}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    prepend variantSets = "state"
    variants = {{ string state = "fresh" }}
    customData = {{
        string drAnmarAssetId = "dranmar-division-blade-cartridge-v1"
        bool drAnmarClinicalValidation = false
    }}
)
{{
    float physics:mass = 0.026
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(BLADE_ROOT)}
{physics_materials_scope()}
{carrier_block}
{blade_block}
    def Scope "Collisions"
    {{
        def Cube "CarrierCollider" ( prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"] )
        {{
            rel material:binding:physics = </{BLADE_ROOT}/PhysicsMaterials/CeramicPhysics>
            double size = 1
            double3 xformOp:translate = (0, 0, -0.006)
            double3 xformOp:scale = (0.018, 0.032, 0.042)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            token visibility = "invisible"
            uniform token purpose = "guide"
        }}
        def Cube "BladeCollider" ( prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"] )
        {{
            rel material:binding:physics = </{BLADE_ROOT}/PhysicsMaterials/BladePhysics>
            custom string drAnmar:role = "division_blade_contact"
            double size = 1
            double3 xformOp:translate = (0, 0, 0.020)
            double3 xformOp:scale = (0.0012, 0.023, 0.025)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            token visibility = "invisible"
            uniform token purpose = "guide"
        }}
    }}
    def Scope "Frames"
    {{
        def Xform "cartridge_mount"
        {{
            custom string drAnmar:role = "cartridge_mount"
        }}
        def Xform "blade_tip"
        {{
            custom string drAnmar:role = "blade_tip"
            double3 xformOp:translate = (0, 0, 0.034)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}
    variantSet "state" = {{
        "fresh"
        {{
{bind('BladeSteel')}
        }}
        "spent"
        {{
{bind('SpentBlade')}
        }}
    }}
}}
'''


def vapor_usda() -> str:
    sphere=trimesh.creation.icosphere(subdivisions=2,radius=0.0008)
    block=mesh_usda(Visual("Vapor",sphere,"VaporMaterial",("seal_vapor_particle",)),f"/{VAPOR_ROOT}/Looks/VaporMaterial",indent="    ")
    return f'''#usda 1.0
(
    defaultPrim = "{VAPOR_ROOT}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{VAPOR_ROOT}" (
    customData = {{
        bool drAnmarClinicalValidation = false
        string drAnmarRepresentation = "visual_particle_prototype"
    }}
)
{{
{visual_materials_scope(VAPOR_ROOT)}
{block}
}}
'''


def export_scene(path: Path, entries: Sequence[tuple[str,trimesh.Trimesh,str]]) -> None:
    scene=trimesh.Scene()
    for name,mesh,material in entries:
        scene.add_geometry(pbr(mesh,material),node_name=name,geom_name=name)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))


def phase_parameters(phase: str) -> dict[str,float]:
    phases={
        "inspect":dict(lc=0,rc=0,uj=0,lj=0,guard=0,blade=0,suction=0,irrigation=0),
        "center":dict(lc=0.022,rc=-0.022,uj=0,lj=0,guard=0,blade=0,suction=0.002,irrigation=0),
        "compress":dict(lc=0.022,rc=-0.022,uj=0.010,lj=-0.010,guard=0,blade=0,suction=0.004,irrigation=0),
        "seal":dict(lc=0.022,rc=-0.022,uj=0.013,lj=-0.013,guard=0,blade=0,suction=0.005,irrigation=0),
        "verify":dict(lc=0.022,rc=-0.022,uj=0.013,lj=-0.013,guard=0,blade=0,suction=0.003,irrigation=0),
        "divide":dict(lc=0.022,rc=-0.022,uj=0.013,lj=-0.013,guard=-0.011,blade=0.041,suction=0.006,irrigation=0),
        "release":dict(lc=0,rc=0,uj=0,lj=0,guard=0,blade=0,suction=0.003,irrigation=0.003),
        "complete":dict(lc=0,rc=0,uj=0,lj=0,guard=0,blade=0,suction=0,irrigation=0),
    }
    return phases[phase]


def link_world_transform(bundle: ToolBundle, link_name: str, phase: str) -> np.ndarray:
    p=phase_parameters(phase);link=bundle.links[link_name];t=np.asarray(link.translation,dtype=float);R=np.eye(3)
    if link_name=="LeftCentering":t=t+np.asarray([0,p["lc"],0])
    elif link_name=="RightCentering":t=t+np.asarray([0,p["rc"],0])
    elif link_name=="UpperJaw":t=t+np.asarray([0,0,p["uj"]])
    elif link_name=="LowerJaw":t=t+np.asarray([0,0,p["lj"]])
    elif link_name=="BladeGuard":t=t+np.asarray([0,0,p["guard"]])
    elif link_name=="BladeCarriage":t=t+np.asarray([0,0,p["blade"]])
    elif link_name=="SuctionValve":t=t+np.asarray([0,p["suction"],0])
    elif link_name=="IrrigationValve":t=t+np.asarray([0,p["irrigation"],0])
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;return T


def world_visual_entries(bundle: ToolBundle, phase: str="inspect") -> list[tuple[str,trimesh.Trimesh,str]]:
    out=[]
    for link_name,link in bundle.links.items():
        T=link_world_transform(bundle,link_name,phase)
        for v in link.visuals:
            m=v.mesh.copy();m.apply_transform(T);out.append((f"{link_name}_{v.name}",m,v.material))
    return out


def collider_mesh(c: Collider) -> trimesh.Trimesh:
    if c.kind=="box":
        assert c.size is not None;m=box_mesh(c.size,c.center)
    elif c.kind=="cylinder":
        assert c.radius is not None and c.height is not None;m=cylinder_axis(c.radius,c.height,c.axis,c.center)
    elif c.kind=="sphere":
        assert c.radius is not None;m=ellipsoid_mesh((c.radius,c.radius,c.radius),c.center,2)
    else:raise ValueError(c.kind)
    return m


def collision_debug_entries(bundle: ToolBundle, phase: str="compress") -> list[tuple[str,trimesh.Trimesh,str]]:
    out=world_visual_entries(bundle,phase)
    for link_name,link in bundle.links.items():
        T=link_world_transform(bundle,link_name,phase)
        for c in link.colliders:
            if not c.author_enabled:continue
            m=collider_mesh(c);m.apply_transform(T);out.append((f"{link_name}_{c.name}",m,"CollisionDebug"))
    return out


def axis_entries(bundle: ToolBundle, phase: str="inspect", length: float=0.012, radius: float=0.00045) -> list[tuple[str,trimesh.Trimesh,str]]:
    out=[]
    for name,data in bundle.frames.items():
        T=link_world_transform(bundle,str(data["parent_link"]),phase)
        p=T[:3,:3]@np.asarray(data["position"],dtype=float)+T[:3,3];R=T[:3,:3]
        for i,(d,mat) in enumerate(((R[:,0],"GuideRed"),(R[:,1],"GuideGreen"),(R[:,2],"GuideBlue"))):
            out.append((f"{name}_{i}",capsule_between(p,p+d*length,radius),mat))
    return out


def vessel_entries(bundle: ToolBundle, state: str="intact") -> list[tuple[str,trimesh.Trimesh,str]]:
    left=bundle.vessel_left.copy();right=bundle.vessel_right.copy();bridge=bundle.bridge_visual.copy()
    if state in {"compressed","sealed","dividing","divided"}:
        left.apply_scale((1.0,1.03,0.62));right.apply_scale((1.0,1.03,0.62));bridge.apply_scale((1.0,1.03,0.62))
    if state=="divided":
        left.apply_translation((-0.0016,0,0));right.apply_translation((0.0016,0,0))
    out=[("FixtureBase",bundle.vessel_base,"FixtureMaterial"),("LeftVessel",left,"VesselMaterial"),("RightVessel",right,"VesselMaterial")]
    if state!="divided":out.append(("BridgeVisual",bridge,"BridgeMaterial"))
    if state in {"sealed","dividing","divided"}:
        band_l=transform(bundle.seal_band,(-SEAL_ZONE_X,0,0));band_r=transform(bundle.seal_band,(SEAL_ZONE_X,0,0))
        out.extend([("LeftSealBand",band_l,"SealBandMature"),("RightSealBand",band_r,"SealBandMature")])
    if state=="dividing":
        out.append(("Blade",transform(wedge_blade_mesh(),(0,0,0.012)),"BladeSteel"))
    return out


def franka_proxy_entries(bundle: ToolBundle, phase: str="inspect") -> list[tuple[str,trimesh.Trimesh,str]]:
    out=[]
    joints=[(0,0,0.05),(0,0,0.30),(0.18,0,0.48),(0.05,0,0.68),(0.22,0,0.83),(0.05,0,1.00),(0.14,0,1.12),(0.14,0,1.22)]
    for i,(a,b) in enumerate(zip(joints[:-1],joints[1:])):out.append((f"ArmLink_{i:02d}",capsule_between(a,b,0.035 if i<3 else 0.028),"BodyPolymer"))
    for i,p in enumerate(joints):out.append((f"ArmJoint_{i:02d}",ellipsoid_mesh((0.045,0.045,0.045),p,2),"AccentPolymer"))
    tool_offset=np.asarray([0.14,0,1.22])
    for n,m,mat in world_visual_entries(bundle,phase):out.append((n,transform(m,tool_offset),mat))
    return out


def exploded_entries(bundle: ToolBundle) -> list[tuple[str,trimesh.Trimesh,str]]:
    offsets={"Mount":(0,0,0),"LeftCentering":(0,-0.065,0.015),"RightCentering":(0,0.065,0.015),"UpperJaw":(-0.055,0,0.010),"LowerJaw":(0.055,0,0.010),"BladeGuard":(0,-0.045,0.045),"BladeCarriage":(0,0.045,0.060),"SuctionValve":(0.030,0.050,0),"IrrigationValve":(-0.030,0.050,0)}
    out=[]
    for link_name,link in bundle.links.items():
        T=link_world_transform(bundle,link_name,"inspect");T[:3,3]+=np.asarray(offsets.get(link_name,(0,0,0)))
        for v in link.visuals:
            m=v.mesh.copy();m.apply_transform(T);out.append((f"{link_name}_{v.name}",m,v.material))
    return out


def export_glbs(bundle: ToolBundle) -> list[Path]:
    outputs=[]
    state_for_phase={"inspect":"intact","center":"intact","compress":"compressed","seal":"sealed","verify":"sealed","divide":"dividing","release":"divided","complete":"divided"}
    for phase in ("inspect","center","compress","seal","verify","divide","release","complete"):
        entries=world_visual_entries(bundle,phase)
        entries += [(n,transform(m,(0,0,WORK_PLANE_Z)),mat) for n,m,mat in vessel_entries(bundle,state_for_phase[phase])]
        p=GLB_ROOT/f"dranmar_seal_divide_tool_{phase}.glb";export_scene(p,entries);outputs.append(p)
    p=GLB_ROOT/"dranmar_seal_divide_tool_exploded.glb";export_scene(p,exploded_entries(bundle));outputs.append(p)
    p=GLB_ROOT/"dranmar_seal_divide_tool_collision_debug.glb";export_scene(p,collision_debug_entries(bundle,"compress"));outputs.append(p)
    p=GLB_ROOT/"dranmar_seal_divide_tool_frame_debug.glb";export_scene(p,world_visual_entries(bundle,"inspect")+axis_entries(bundle,"inspect"));outputs.append(p)
    p=GLB_ROOT/"dranmar_franka_seal_divide_assembly.glb";export_scene(p,franka_proxy_entries(bundle,"inspect"));outputs.append(p)
    for state in ("intact","compressed","sealed","divided"):
        p=GLB_ROOT/f"dranmar_seal_divide_vessel_{state}.glb";export_scene(p,vessel_entries(bundle,state));outputs.append(p)
    p=GLB_ROOT/"dranmar_tissue_seal_band.glb";export_scene(p,[("SealBand",bundle.seal_band,"SealBandMature")]);outputs.append(p)
    p=GLB_ROOT/"dranmar_division_blade_cartridge.glb";export_scene(p,[("BladeCartridge",bundle.blade_cartridge,"BladeSteel")]);outputs.append(p)
    return outputs


def add_mesh_to_axis(ax, mesh: trimesh.Trimesh, material: str, max_faces: int=1000) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces=np.asarray(mesh.faces)
    if len(faces)>max_faces:faces=faces[np.linspace(0,len(faces)-1,max_faces,dtype=int)]
    tri=np.asarray(mesh.vertices)[faces];rgba=np.asarray(material_color(material))/255.0
    coll=Poly3DCollection(tri,facecolors=[rgba],edgecolors="none",linewidths=0.0,alpha=float(rgba[3]));ax.add_collection3d(coll)


def configure_axis(ax,title: str,elev: float=22,azim: float=-58) -> None:
    ax.set_title(title,fontsize=11,pad=8);ax.set_xlim(-0.105,0.105);ax.set_ylim(-0.105,0.105);ax.set_zlim(0.0,0.235)
    ax.view_init(elev=elev,azim=azim);ax.set_axis_off();ax.set_box_aspect((1,1,1.1))


def make_preview(bundle: ToolBundle) -> Path:
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(16,10),dpi=160)
    phases=[("inspect","1  Inspect + localize","intact"),("center","2  Center pedicle","intact"),("compress","3  Compress","compressed"),("seal","4  Form dual seals","sealed"),("divide","5  Interlocked division","dividing"),("release","6  Sealed stumps","divided")]
    for i,(phase,title,vstate) in enumerate(phases,1):
        ax=fig.add_subplot(2,3,i,projection="3d")
        for _,m,mat in world_visual_entries(bundle,phase):add_mesh_to_axis(ax,m,mat,850)
        for _,m,mat in vessel_entries(bundle,vstate):add_mesh_to_axis(ax,transform(m,(0,0,WORK_PLANE_Z)),mat,850)
        configure_axis(ax,title)
    fig.suptitle("DrAnmar Adaptive Seal-and-Divide Robot — center, compress, seal twice, verify, divide",fontsize=17,y=0.98)
    fig.text(0.5,0.015,"DrAnmar-owned, provider-neutral NVIDIA Isaac research asset • blade interlocked until both seal zones qualify",ha="center",fontsize=10)
    path=PREVIEW_ROOT/"dranmar_adaptive_seal_divide_robot_preview.png";fig.savefig(path,bbox_inches="tight",facecolor="white");plt.close(fig);return path


def make_full_arm_preview(bundle: ToolBundle) -> Path:
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(10,10),dpi=170);ax=fig.add_subplot(111,projection="3d")
    for _,m,mat in franka_proxy_entries(bundle,"inspect"):add_mesh_to_axis(ax,m,mat,1050)
    ax.set_xlim(-0.18,0.38);ax.set_ylim(-0.25,0.25);ax.set_zlim(0,1.48);ax.view_init(elev=20,azim=-58);ax.set_axis_off();ax.set_box_aspect((0.6,0.55,1.45))
    ax.set_title("DrAnmar Adaptive Seal-and-Divide Robot mounted at the Franka wrist",fontsize=15,pad=12)
    path=PREVIEW_ROOT/"dranmar_adaptive_seal_divide_robot_full_arm_preview.png";fig.savefig(path,bbox_inches="tight",facecolor="white");plt.close(fig);return path


def noise_texture(base: tuple[int,int,int], size: int=512, strength: int=18, seed: int=1) -> Image.Image:
    rng=np.random.default_rng(seed);arr=np.zeros((size,size,3),dtype=np.int16);arr[:]=np.asarray(base,dtype=np.int16);arr+=rng.normal(0,strength,(size,size,1)).astype(np.int16);arr=np.clip(arr,0,255).astype(np.uint8);return Image.fromarray(arr,"RGB")


def generate_textures() -> list[Path]:
    TEXTURE_ROOT.mkdir(parents=True,exist_ok=True);out=[]
    for name,base,strength,seed in [
        ("vessel_basecolor.png",(132,22,20),13,31),("polymer_microtexture.png",(212,218,224),9,32),
        ("metal_microtexture.png",(145,151,160),7,33),("ceramic_microtexture.png",(232,232,224),5,34),
        ("electrode_microtexture.png",(190,88,24),8,35),("seal_band_basecolor.png",(176,75,15),12,36),
    ]:
        p=TEXTURE_ROOT/name;noise_texture(base,512,strength,seed).save(p);out.append(p)
    img=Image.new("RGB",(1024,256),(247,249,251));d=ImageDraw.Draw(img)
    try:font=ImageFont.truetype("DejaVuSans-Bold.ttf",72);small=ImageFont.truetype("DejaVuSans.ttf",30)
    except OSError:font=None;small=None
    d.text((36,48),"DrAnmar",fill=(18,65,112),font=font);d.text((40,150),"ADAPTIVE SEAL + DIVIDE • TRAINING WORKCELL",fill=(35,45,55),font=small)
    p=TEXTURE_ROOT/"label_dranmar.png";img.save(p);out.append(p);return out


def interaction_frames(bundle: ToolBundle) -> dict[str,object]:
    return {"schema":"dranmar.interaction-frames.v1","asset":"dranmar-adaptive-seal-divide-robot-v1","units":"m","frames":bundle.frames}


def mount_contract() -> dict[str,object]:
    return {
        "schema":"dranmar.franka-mount.v1",
        "parent_link":"panda_link8",
        "payload_link":"DrAnmarAdaptiveSealDivideTool/Links/Mount",
        "local_translation_m":[0,0,0],
        "local_rotation_axis_angle_deg":{"axis":[0,0,1],"angle":FRANKA_HAND_EQUIVALENT_ROTATION_DEG},
        "deactivate":["panda_hand_joint","panda_hand","panda_finger_joint1","panda_finger_joint2","panda_leftfinger","panda_rightfinger"],
        "intended_use":"simulation_training",
    }


def task_contract() -> dict[str,object]:
    return {
        "schema":"dranmar.adaptive-seal-divide-task.v1",
        "phases":["inspect","center","compress","seal","verify_seal","retract_guard","divide","release","verify_stumps","complete","abort"],
        "interlocks":["tissue_centered","compression_within_window","left_seal_mature","right_seal_mature","predicted_stump_flow_below_limit","blade_guard_retracted","no_overtemperature_fault"],
        "success_metrics":["left_seal_maturity","right_seal_maturity","left_stump_flow_ml_min","right_stump_flow_ml_min","peak_jaw_force_n","energy_j","blade_interlock_violations","bridge_cells_released","division_complete","seal_band_retained"],
        "failure_modes":["off_center_capture","insufficient_compression","excess_compression","seal_underenergy","seal_overtemperature","impedance_fault","blade_before_seal","incomplete_division","seal_band_delamination","stump_leak","tissue_crush_proxy"],
        "clinical_validation":False,
    }


def physics_profile(bundle: ToolBundle) -> dict[str,object]:
    return {
        "schema":"dranmar.adaptive-seal-divide-profile.v1",
        "id":"dranmar-adaptive-seal-divide-robot-v1",
        "version":VERSION,
        "status":"simulation_training_model",
        "tool":{"mount":"panda_link8","joint_count":len(bundle.joints),"seal_zone_count":2,"bridge_pin_count":BRIDGE_PIN_COUNT,"suction_ports":SUCTION_PORT_COUNT,"irrigation_ports":IRRIGATION_PORT_COUNT},
        "centering":{"travel_per_side_m":CENTERING_TRAVEL_M,"target_force_per_side_n":0.7,"soft_limit_n":2.2},
        "compression":{"jaw_travel_per_side_m":JAW_TRAVEL_M,"target_total_force_n":18.0,"minimum_force_n":8.0,"soft_limit_n":32.0,"hard_abort_n":45.0},
        "energy_proxy":{"target_temperature_c":78.0,"maximum_temperature_c":105.0,"maximum_power_per_zone_w":45.0,"thermal_capacity_j_k":1.8,"heat_loss_w_k":0.22,"maturity_threshold":0.90,"model":"lumped_thermal_impedance_and_first_order_seal_maturity_proxy"},
        "seal_band":{"outer_envelope_m":[0.0046,0.0104,0.0060],"initial_break_force_n":0.6,"mature_break_force_n":7.5,"bond_regions":["upper","lower"],"state_variants":["fresh","mature","failed"]},
        "blade":{"travel_m":BLADE_TRAVEL_M,"guard_travel_m":GUARD_TRAVEL_M,"interlocked":True,"topology_change":"progressive_release_of_runtime_bridge_attachments"},
        "vessel":{"total_length_m":0.0844,"outer_diameter_m":0.0084,"wall_thickness_m":0.00068,"representation":"two_watertight_surface_deformable_halves_with_runtime_bridge_pins","actual_intraluminal_cfd":False},
        "leak_verification":{"reference_pressure_pa":16000.0,"maximum_flow_per_stump_ml_min":0.1,"model":"reduced_order_orifice_flow_from_seal_maturity_and_damage"},
        "boundaries":["no clinical vessel-sealing claim","no calibrated electrosurgical tissue response","no continuous topology cutting","no validated thermal injury model","no patient-care settings"],
    }


def collider_coverage(bundle: ToolBundle) -> dict[str,object]:
    links={}
    for name,link in bundle.links.items():
        if not link.visuals or not link.colliders:continue
        vbmin,vbmax=mesh_bounds([v.mesh for v in link.visuals]);vsize=np.maximum(vbmax-vbmin,1e-9)
        collider_meshes=[collider_mesh(c) for c in link.colliders if c.author_enabled]
        cbmin,cbmax=mesh_bounds(collider_meshes);csize=cbmax-cbmin
        links[name]={"visual_bounds_min_m":vbmin.tolist(),"visual_bounds_max_m":vbmax.tolist(),"collider_bounds_min_m":cbmin.tolist(),"collider_bounds_max_m":cbmax.tolist(),"axis_coverage_ratio":(csize/vsize).tolist(),"deliberate_gaps":["central blade channel"] if "Jaw" in name else []}
    return {"schema":"dranmar.collider-coverage.v1","asset":"dranmar-adaptive-seal-divide-robot-v1","links":links}


def asset_manifest() -> dict[str,object]:
    return {
        "schema":"dranmar.asset-manifest.v1","id":"dranmar-adaptive-seal-divide-robot-v1","version":VERSION,
        "catalog_subpath":str(CATALOG_SUBPATH),"license":"Apache-2.0","intended_use":"simulation_training","clinical_validation":False,
        "primary_assets":["dranmar_adaptive_seal_divide_tool_payload.usda","dranmar_adaptive_seal_divide_tool_standalone.usda","dranmar_adaptive_seal_divide_tool_rigid_proxy.usda","dranmar_seal_divide_vessel_demo.usda","dranmar_tissue_seal_band.usda","dranmar_division_blade_cartridge.usda","dranmar_seal_vapor_particle.usda"],
        "runtime_capabilities":["franka_hand_replacement","tissue_centering","dual_zone_compression","adaptive_energy_proxy","physical_seal_band_retention","blade_interlock","progressive_bridge_release_division","dual_stump_leak_verification"],
    }


def integration_module() -> str:
    if not INTEGRATION_PATH.is_file():
        raise FileNotFoundError(
            f"Canonical integration module is missing: {INTEGRATION_PATH}"
        )
    return INTEGRATION_PATH.read_text(encoding="utf-8")



def example_scene() -> str:
    return textwrap.dedent('''\
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
    ''')


def docs_mechanism() -> str:
    return f'''# DrAnmar Adaptive Seal-and-Divide Mechanism

The payload replaces the Panda hand at `panda_link8` and uses eight controlled axes:

- bilateral tissue-centering carriages;
- independently driven upper and lower seal jaws;
- retractable ceramic blade guard;
- central division blade carriage;
- suction and irrigation metering valves.

Each jaw contains two separate electrode zones at X = ±{SEAL_ZONE_X*1000:.1f} mm and a deliberate central blade channel. The two future tissue stumps are compressed and sealed independently before the blade can advance.

The mechanism is owned by DrAnmar and provider-neutral. NVIDIA Isaac Sim and
Isaac Lab provide the target runtime. It does not reproduce a commercial
energy device or approved surgical settings.
'''


def docs_physical() -> str:
    return f'''# Physical Seal and Division Contract

## Intact tissue representation

The demo vessel contains two watertight deformable halves connected by {BRIDGE_PIN_COUNT} temporary bridge pins. Each pin is attached physically to both halves. Two explicit kinematic distal fixtures keep the ends grounded. The initial visual bridge conceals the submillimetre seam.

Surface self-collision is disabled by default for portable GPU deformable
cooking. It can be enabled explicitly for a qualified solver configuration.

## Temporary jaw compression

The runtime compression controller creates four verified deformable
attachments: upper and lower jaw seal contacts for each future stump. Force is
reported by the caller and checked against the provisional soft and hard
limits. These attachments are temporary and are released only after both
retained seal bands exist.

## Seal retention

A seal operation deploys one `DrAnmarTissueSealBand` on each future stump. Each band has independent upper and lower bond volumes. The vessel is compressed when those attachments are created, so the band preserves the collapsed wall configuration after jaw release.

Seal-band state progresses from `fresh` to `mature`. Excess load removes the attachments and changes the state to `failed`.

## Division

The blade does not silently replace the vessel mesh. Blade progress releases bridge-pin attachments in a defined order. At complete progress, the two deformable halves are mechanically independent while their seal bands remain attached.

This is a controlled topology-surrogate strategy. It does not claim continuous fracture, histological thermal fusion, or clinically validated stump strength.
'''


def docs_energy() -> str:
    return '''# Energy, Interlock, and Leak Model

The runtime helper uses a lumped thermal model per seal zone:

`C dT/dt = P_absorbed - h (T - T_ambient)`

Compression affects absorbed power. A temperature-dependent dose integrates into a bounded seal-maturity state. Impedance is updated as a research proxy for heating and desiccation.

The blade interlock requires:

1. compression inside the configured force window;
2. both seal zones above the maturity threshold;
3. no overtemperature or impedance fault;
4. predicted flow from both future stumps below the configured limit;
5. the ceramic guard fully retracted.

The leak estimate is a reduced-order orifice-flow model driven by seal maturity, residual gap, pressure, and damage. It is not CFD and is not evidence of clinical seal integrity.

The thermal state is lumped per zone. Spatial thermal spread, tissue
histology, smoke chemistry, electrical current paths, generator waveforms, and
collateral injury are not modeled or calibrated.
'''


def docs_franka() -> str:
    return '''# Franka Integration

`make_franka_adaptive_seal_divide_robot_cfg()` starts from the Isaac Lab Franka configuration, loads the composable Franka USD, deactivates the Panda hand and finger prims, references the custom payload, and creates a fixed joint from `panda_link8` to `Links/Mount`.

The spawner snapshots the stock Panda hand joint's parent path and local frame
before deactivating the hand subtree. The documented −45 degree local Z
relationship is used only as a compatibility fallback when the stock hand
joint is unavailable. Tool joints are appended to the same articulation and
grouped into centering, jaw, blade, and valve actuator sets.

The rigid proxy is available for perception, collision-aware planning, and synthetic data when the articulated mechanism is not required.
'''


def readme() -> str:
    return f'''# {ASSET_NAME}

A DrAnmar-owned, provider-neutral research end effector for NVIDIA Isaac Sim
and Isaac Lab.

## Workflow

`inspect → center → compress → seal left/right → verify → retract guard → divide → release → verify stumps`

## Primary assets

- `dranmar_adaptive_seal_divide_tool_payload.usda` — Franka payload without a nested articulation root.
- `dranmar_adaptive_seal_divide_tool_standalone.usda` — standalone articulated mechanism.
- `dranmar_adaptive_seal_divide_tool_rigid_proxy.usda` — perception/planning proxy.
- `dranmar_seal_divide_vessel_demo.usda` — two-body hollow vessel and bridge-pin scene.
- `dranmar_tissue_seal_band.usda` — physical dual-surface stump bond carrier.
- `dranmar_division_blade_cartridge.usda` — replaceable fresh/spent blade cartridge.

## Important boundary

This package is not clinically validated, is not a medical device, and is not approved for patient care. Energy, compression, seal strength, tissue response, leakage, and cutting parameters are provisional research values.
'''


def docs_validation() -> str:
    return '''# Integrity and runtime boundaries

Static gates cover deterministic assets, dependency closure, controller
invariants, fail-closed attachment overlap, blade interlocks, and
source/container integrity. The optional Isaac script is diagnostic only.

The current surface-shell vessel does not resolve calibrated through-thickness
compression. Measured generalized jaw effort below the authored force envelope
must keep the blade interlock closed; the threshold must not be lowered and a
synthetic force must not be injected to obtain a pass.

Seal efficacy, thermal fusion, burst pressure, division quality, physical
calibration, clinical performance, and patient use remain unqualified pending
a calibrated volumetric vessel/material and instrumented bench data.
'''


def installer_source() -> str:
    installer = PACKAGE_ROOT / "scripts/install_into_dranmar.py"
    if not installer.is_file():
        raise FileNotFoundError(f"Canonical installer is missing: {installer}")
    return installer.read_text(encoding="utf-8")


def write_json(path: Path,payload: Any) -> Path:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");return path


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def write_asset_files(bundle: ToolBundle) -> list[Path]:
    for p in (ASSET_ROOT,GLB_ROOT,TEXTURE_ROOT,PREVIEW_ROOT,DOCS_ROOT,EXAMPLE_ROOT,INTEGRATION_PATH.parent,PHYSICS_PROFILE_PATH.parent):p.mkdir(parents=True,exist_ok=True)
    files=[]
    mapping={
        "dranmar_adaptive_seal_divide_tool_payload.usda":tool_usda(bundle,False),
        "dranmar_adaptive_seal_divide_tool_standalone.usda":tool_usda(bundle,True),
        "dranmar_adaptive_seal_divide_tool_rigid_proxy.usda":rigid_proxy_usda(bundle),
        "dranmar_seal_divide_vessel_demo.usda":vessel_usda(bundle),
        "dranmar_tissue_seal_band.usda":seal_band_usda(bundle),
        "dranmar_division_blade_cartridge.usda":blade_cartridge_usda(bundle),
        "dranmar_seal_vapor_particle.usda":vapor_usda(),
        "README.md":readme(),
    }
    for name,text in mapping.items():
        path=ASSET_ROOT/name;path.write_text(text,encoding="utf-8");files.append(path)
    license_path=ASSET_ROOT/"LICENSE.txt";license_path.write_text("Apache License 2.0\nCopyright 2026 DrAnmar Project Developers\n",encoding="utf-8");files.append(license_path)
    files+=generate_textures();files+=export_glbs(bundle);files+=[make_preview(bundle),make_full_arm_preview(bundle)]
    files += [
        write_json(ASSET_ROOT/"interaction_frames.json",interaction_frames(bundle)),
        write_json(ASSET_ROOT/"franka_mount_contract.json",mount_contract()),
        write_json(ASSET_ROOT/"adaptive_seal_divide_task_contract.json",task_contract()),
        write_json(ASSET_ROOT/"physics_profile.json",physics_profile(bundle)),
        write_json(ASSET_ROOT/"collider_coverage.json",collider_coverage(bundle)),
        write_json(ASSET_ROOT/"asset_manifest.json",asset_manifest()),
    ]
    write_json(PHYSICS_PROFILE_PATH,physics_profile(bundle));files.append(PHYSICS_PROFILE_PATH)
    INTEGRATION_PATH.write_text(integration_module(),encoding="utf-8");files.append(INTEGRATION_PATH)
    for name,text in (("MECHANISM.md",docs_mechanism()),("PHYSICAL_SEAL_AND_DIVIDE.md",docs_physical()),("ENERGY_AND_LEAK_MODEL.md",docs_energy()),("FRANKA_INTEGRATION.md",docs_franka()),("VALIDATION.md",docs_validation())):
        path=DOCS_ROOT/name;path.write_text(text,encoding="utf-8");files.append(path)
    example=EXAMPLE_ROOT/"franka_adaptive_seal_divide_scene.py";example.write_text(example_scene(),encoding="utf-8");files.append(example)
    installer=PACKAGE_ROOT/"scripts/install_into_dranmar.py";installer.write_text(installer_source(),encoding="utf-8");installer.chmod(0o755);files.append(installer)
    return files


def sync_extension_data() -> None:
    target=EXTENSION_ROOT/"data"/CATALOG_SUBPATH
    shutil.rmtree(target,ignore_errors=True);target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(ASSET_ROOT,target)


def all_payload_files() -> list[Path]:
    mirror_root=EXTENSION_ROOT/"data"/CATALOG_SUBPATH
    excluded_names={"asset_manifest.json","static_build_report.json",".DS_Store"}
    return sorted(
        p for p in PACKAGE_ROOT.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix!=".pyc"
        and p.name not in excluded_names
        and not p.is_relative_to(mirror_root)
    )


def build_manifest(files: Sequence[Path]) -> dict[str,object]:
    return {
        "schema":"dranmar.asset-manifest.v1",
        "asset":"dranmar-adaptive-seal-divide-robot-v1",
        "version":VERSION,
        "catalog_subpath":CATALOG_SUBPATH.as_posix(),
        "file_count":len(files),
        "files":[
            {
                "path":path.relative_to(PACKAGE_ROOT).as_posix(),
                "bytes":path.stat().st_size,
                "sha256":sha256(path),
            }
            for path in files
        ],
    }


def zip_tree(source: Path,output: Path,*,prefix: str|None=None) -> Path:
    if output.exists():output.unlink()
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix==".pyc"
                or path.name==".DS_Store"
            ):
                continue
            rel=path.relative_to(source).as_posix()
            arc=f"{prefix.rstrip('/')}/{rel}" if prefix else rel
            info=zipfile.ZipInfo(arc,date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o755 if path.stat().st_mode & 0o111 else 0o644)<<16
            archive.writestr(
                info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output


def write_checksum(path: Path) -> Path:
    out=path.with_suffix(path.suffix+".sha256");out.write_text(f"{sha256(path)}  {path.name}\n",encoding="utf-8");return out


def build_overlay() -> Path:
    temp=PACKAGE_ROOT.parent/"_dranmar_adaptive_seal_divide_overlay"
    shutil.rmtree(temp,ignore_errors=True)
    for sub in ("source","physics_next","docs","examples","tests"):
        src=PACKAGE_ROOT/sub
        if src.exists():shutil.copytree(src,temp/sub,dirs_exist_ok=True)
    (temp/"scripts").mkdir(parents=True,exist_ok=True)
    for name in (
        SCRIPT_PATH.name,
        "install_into_dranmar.py",
        "requirements_adaptive_seal_divide_generation.txt",
        "validate_dranmar_adaptive_seal_divide_robot.py",
    ):
        source=PACKAGE_ROOT/"scripts"/name
        if source.exists():shutil.copy2(source,temp/"scripts"/name)
    output=PACKAGE_ROOT.parent/"dranmar_adaptive_seal_divide_robot_repo_overlay_v0.1.0.zip";zip_tree(temp,output);shutil.rmtree(temp);return output


def static_report(files: Sequence[Path]) -> dict[str,object]:
    checks=[]
    for path in (p for p in files if p.suffix==".usda"):
        text=path.read_text(encoding="utf-8")
        checks.append({
            "file":path.relative_to(PACKAGE_ROOT).as_posix(),
            "brace_balance":text.count("{")==text.count("}"),
            "nested_quaternion_suspect":"(1, (" in text,
            "one_line_over_suspect":any(line.strip().startswith("over ") and "{" in line and "}" in line for line in text.splitlines()),
            "flat_quaternion_count":text.count("quatf "),
        })
    return {
        "schema":"dranmar.static-build-report.v1",
        "asset":"dranmar-adaptive-seal-divide-robot-v1",
        "usda_checks":checks,
        "python_files":[
            p.relative_to(PACKAGE_ROOT).as_posix()
            for p in files if p.suffix==".py"
        ],
        "native_simulator_evidence":"not_recorded",
        "real_world_evidence":"not_established",
    }


def syntax_check_python(paths: Sequence[Path]) -> None:
    for path in paths:
        source=path.read_text(encoding="utf-8");compile(source,str(path),"exec")


def generate() -> dict[str,object]:
    for junk in sorted(PACKAGE_ROOT.rglob(".DS_Store")):junk.unlink()
    for cache in sorted(PACKAGE_ROOT.rglob("__pycache__"),reverse=True):shutil.rmtree(cache)
    for bytecode in PACKAGE_ROOT.rglob("*.pyc"):bytecode.unlink()
    for legacy in (
        "ENERGY_AND_LEAK_MODEL.md","FRANKA_INTEGRATION.md",
        "MECHANISM.md","PHYSICAL_SEAL_AND_DIVIDE.md",
    ):
        path=PACKAGE_ROOT/"docs"/legacy
        if path.exists():path.unlink()
    old_manifest=ASSET_ROOT/"asset_manifest.json"
    if old_manifest.exists():old_manifest.unlink()
    bundle=build_tool();write_asset_files(bundle)
    files=all_payload_files()
    manifest=write_json(ASSET_ROOT/"asset_manifest.json",build_manifest(files))
    sync_extension_data()
    static=write_json(PACKAGE_ROOT/"static_build_report.json",static_report(all_payload_files()))
    for python_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        compile(python_path.read_text(encoding="utf-8"),str(python_path),"exec")
    dev=PACKAGE_ROOT.parent/f"dranmar_adaptive_seal_divide_robot_v{VERSION}.zip";zip_tree(PACKAGE_ROOT,dev,prefix=PACKAGE_ROOT.name)
    catalog=PACKAGE_ROOT.parent/f"dranmar_adaptive_seal_divide_robot_catalog_v{VERSION}.zip";zip_tree(PACKAGE_ROOT/"assets",catalog)
    overlay=build_overlay()
    for path in (dev,catalog,overlay):write_checksum(path)
    release={
        "schema":"dranmar.release.v1","asset":"dranmar-adaptive-seal-divide-robot-v1","version":VERSION,
        "catalog_subpath":CATALOG_SUBPATH.as_posix(),
        "development_package":{"path":str(dev),"sha256":sha256(dev)},
        "catalog_package":{"path":str(catalog),"sha256":sha256(catalog)},
        "repository_overlay":{"path":str(overlay),"sha256":sha256(overlay)},
        "primary_assets":[str(ASSET_ROOT/name) for name in ("dranmar_adaptive_seal_divide_tool_payload.usda","dranmar_adaptive_seal_divide_tool_standalone.usda","dranmar_adaptive_seal_divide_tool_rigid_proxy.usda","dranmar_seal_divide_vessel_demo.usda","dranmar_tissue_seal_band.usda","dranmar_division_blade_cartridge.usda")],
        "runtime_validation":static_report(all_payload_files())["runtime_validation"],"clinical_validation":False,
    }
    release_path=PACKAGE_ROOT.parent/"dranmar_adaptive_seal_divide_robot_release_v0.1.0.json";write_json(release_path,release)
    return {"release":release,"release_path":str(release_path),"manifest":str(manifest),"static_report":str(static)}


def main() -> None:
    print(json.dumps(generate(),indent=2))


if __name__=="__main__":
    main()
