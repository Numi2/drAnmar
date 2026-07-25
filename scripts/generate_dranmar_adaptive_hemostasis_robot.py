#!/usr/bin/env python3
"""Generate the DrAnmar Adaptive Hemostasis Robot asset family.

This is an independently authored DrAnmar research asset for NVIDIA Isaac Sim /
Isaac Lab. It models robotic field clearing, distributed
temporary vessel compression, vascular clip deployment, hemostatic patch
placement, conserved blood-volume bookkeeping, and post-seal leak verification.
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
ASSET_NAME = "DrAnmar Adaptive Hemostasis Robot"
CATALOG_SUBPATH = Path("Props/SurgicalHemostasis/AdaptiveHemostasisRobot")
ROOT_PRIM = "DrAnmarAdaptiveHemostasisTool"
STANDALONE_ROOT = "DrAnmarAdaptiveHemostasisToolStandalone"
PROXY_ROOT = "DrAnmarAdaptiveHemostasisToolRigidProxy"
CLIP_ROOT = "DrAnmarHemostaticClip"
PATCH_ROOT = "DrAnmarHemostaticPatch"
PATCH_PROXY_ROOT = "DrAnmarHemostaticPatchRigidProxy"
VESSEL_ROOT = "DrAnmarBleedingVesselDemo"
DROPLET_ROOT = "DrAnmarBloodDroplet"

SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[1]
ASSET_ROOT = PACKAGE_ROOT / "assets" / CATALOG_SUBPATH
GLB_ROOT = ASSET_ROOT / "glb"
TEXTURE_ROOT = ASSET_ROOT / "textures"
PREVIEW_ROOT = PACKAGE_ROOT / "previews"
DOCS_ROOT = PACKAGE_ROOT / "docs/adaptive_hemostasis_robot"
EXAMPLE_ROOT = PACKAGE_ROOT / "examples"
EXTENSION_ROOT = PACKAGE_ROOT / "source/extensions/orbit.surgical.assets"
INTEGRATION_PATH = EXTENSION_ROOT / "orbit/surgical/assets/adaptive_hemostasis_robot.py"
PHYSICS_PROFILE_PATH = PACKAGE_ROOT / "physics_next/surgical-hemostasis/dranmar-adaptive-hemostasis-v1.json"

WORK_PLANE_Z = 0.184
FRANKA_HAND_EQUIVALENT_ROTATION_DEG = -45.0
CLIP_CAPACITY = 8
PATCH_CAPACITY = 4
SUCTION_PORT_COUNT = 12
IRRIGATION_PORT_COUNT = 8
PATCH_BOND_CELL_COUNT = 8


def f(value: float, digits: int = 10) -> str:
    # Preserve small but physically meaningful values such as sub-gram inertia.
    # Only collapse numerical noise far below the scales used by this asset.
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


def u_clip_mesh(formed: bool, *, scale: float = 1.0) -> trimesh.Trimesh:
    gap = (0.0011 if formed else 0.0060) * scale
    half = gap / 2
    crown_z = 0.0032 * scale
    tip_z = -0.0055 * scale
    if formed:
        points = [(-half, tip_z, 0), (-0.0028*scale, -0.0015*scale, 0), (-0.0034*scale, crown_z, 0),
                  (0.0034*scale, crown_z, 0), (0.0028*scale, -0.0015*scale, 0), (half, tip_z, 0)]
    else:
        points = [(-half, tip_z, 0), (-half, 0.0010*scale, 0), (-0.0036*scale, crown_z, 0),
                  (0.0036*scale, crown_z, 0), (half, 0.0010*scale, 0), (half, tip_z, 0)]
    mesh = wire_path(points, 0.00047*scale)
    # clip local X spans the vessel; local Y is thickness; local Z is insertion.
    mesh.apply_transform(np.block([[rotation_matrix((1,0,0), math.pi/2), np.zeros((3,1))],[np.zeros((1,3)), np.ones((1,1))]]))
    return mesh


def tube_wall_mesh(length: float = 0.070, outer_radius: float = 0.0032, wall: float = 0.00062, axial: int = 72, radial: int = 64) -> trimesh.Trimesh:
    inner_radius = outer_radius - wall
    vertices: list[tuple[float,float,float]] = []
    # Axis is +Y; add a mild centerline curvature in X and Z.
    for j in range(axial + 1):
        t = j / axial
        y = -length/2 + t*length
        cx = 0.0012 * math.sin((t-0.5)*math.pi)
        cz = 0.00055 * math.cos((t-0.5)*math.pi)
        for r in (outer_radius, inner_radius):
            for i in range(radial):
                a = 2*math.pi*i/radial
                vertices.append((cx + r*math.cos(a), y, cz + r*math.sin(a)))
    faces: list[tuple[int,int,int]] = []
    ring = radial*2
    for j in range(axial):
        base0=j*ring; base1=(j+1)*ring
        for i in range(radial):
            n=(i+1)%radial
            # outer surface
            a=base0+i; b=base0+n; c=base1+i; d=base1+n
            faces += [(a,b,d),(a,d,c)]
            # inner surface (reverse winding)
            a=base0+radial+i; b=base1+radial+i; c=base0+radial+n; d=base1+radial+n
            faces += [(a,b,d),(a,d,c)]
    # annular end caps
    for end_j, reverse in ((0, True),(axial, False)):
        base=end_j*ring
        for i in range(radial):
            n=(i+1)%radial
            o0=base+i; o1=base+n; i0=base+radial+i; i1=base+radial+n
            faces += [(o0,i1,o1),(o0,i0,i1)] if reverse else [(o0,o1,i1),(o0,i1,i0)]
    mesh=trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),process=False)
    mesh.remove_unreferenced_vertices(); mesh.fix_normals()
    return mesh


def rounded_patch_mesh(width=0.026, height=0.020, nx=30, ny=24) -> trimesh.Trimesh:
    xs=np.linspace(-width/2,width/2,nx)
    ys=np.linspace(-height/2,height/2,ny)
    vertices=[]
    for y in ys:
        for x in xs:
            r=(x/(width/2))**2+(y/(height/2))**2
            dome=0.00045*max(0.0,1.0-r)
            quilt=0.00008*math.sin(700*x)*math.sin(700*y)
            vertices.append((x,y,dome+quilt))
    faces=[]
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;b=a+1;c=a+nx;d=c+1
            faces += [(a,b,d),(a,d,c)]
    mesh=trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),process=False)
    mesh.fix_normals()
    return mesh


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
    vessel: trimesh.Trimesh
    open_clip: trimesh.Trimesh
    formed_clip: trimesh.Trimesh
    patch: trimesh.Trimesh
    vessel_base: trimesh.Trimesh


def build_tool() -> ToolBundle:
    links: dict[str,Link] = {}
    mount_visuals: list[Visual] = [
        Visual("FrankaAdapterPlate", cylinder_axis(0.032,0.012,"z",(0,0,0.006),sections=72), "MountMetal", ("franka_mount",)),
        Visual("QuickReleaseRing", torus_axis(0.0275,0.003,"z",(0,0,0.014),major_sections=72,minor_sections=14), "MountMetal"),
        Visual("MainHousing", ellipsoid_mesh((0.055,0.047,0.034),(0,0,0.055),subdivisions=3), "BodyPolymer", ("adaptive_hemostasis_robot",)),
        Visual("HousingCore", box_mesh((0.102,0.082,0.048),(0,0,0.056)), "BodyPolymer"),
        Visual("CompressionRail", box_mesh((0.148,0.024,0.018),(0,0,0.105)), "RailMetal", ("bilateral_compression_rail",)),
        Visual("ClipMagazineHousing", box_mesh((0.026,0.070,0.036),(0.043,0,0.092)), "DarkPolymer", ("vascular_clip_magazine",)),
        Visual("PatchCarouselHousing", cylinder_axis(0.034,0.018,"z",(-0.041,0,0.093),sections=64), "AccentPolymer", ("hemostatic_patch_carousel",)),
        Visual("SuctionManifold", torus_axis(0.031,0.0032,"z",(0,0,0.151),major_sections=72,minor_sections=14), "DarkPolymer", ("annular_suction_manifold",)),
        Visual("IrrigationManifold", torus_axis(0.024,0.0017,"z",(0,0,0.156),major_sections=72,minor_sections=12), "SensorBlue", ("irrigation_manifold",)),
        Visual("SensorBridge", box_mesh((0.052,0.016,0.015),(0,-0.043,0.084)), "DarkPolymer", ("hemostasis_sensor_bridge",)),
        Visual("StereoCameraLeft", cylinder_axis(0.0048,0.004,"y",(-0.013,-0.052,0.084),sections=36), "SensorGlass", ("rgb_camera",)),
        Visual("StereoCameraRight", cylinder_axis(0.0048,0.004,"y",(0.013,-0.052,0.084),sections=36), "SensorGlass", ("rgb_camera",)),
        Visual("FluorescenceCamera", cylinder_axis(0.0042,0.004,"y",(0,-0.052,0.096),sections=36), "SensorPurple", ("fluorescence_camera",)),
        Visual("FlowProbe", cylinder_axis(0.0052,0.008,"z",(0,0,0.146),sections=40), "SensorGlass", ("flow_verification_probe",)),
        Visual("IrrigationReservoir", cylinder_axis(0.018,0.042,"y",(0.020,0.038,0.058),sections=48), "TubeClear", ("irrigation_reservoir",)),
        Visual("IrrigationFluid", cylinder_axis(0.0155,0.036,"y",(0.020,0.038,0.058),sections=48), "FluidBlue", ("irrigation_inventory",)),
        Visual("CollectionCanister", cylinder_axis(0.020,0.046,"y",(-0.020,0.038,0.058),sections=48), "TubeClear", ("blood_collection_canister",)),
        Visual("CollectionFill", cylinder_axis(0.017,0.014,"y",(-0.020,0.051,0.058),sections=48), "Blood", ("collected_blood",)),
        Visual("LabelPanel", box_mesh((0.052,0.0012,0.022),(0,-0.047,0.052)), "LabelMaterial"),
    ]
    for i in range(SUCTION_PORT_COUNT):
        a=2*math.pi*i/SUCTION_PORT_COUNT
        x=0.031*math.cos(a); y=0.031*math.sin(a)
        mount_visuals.append(Visual(f"SuctionPort_{i:02d}", frustum_axis(0.0024,0.00145,0.008,"z",(x,y,0.158),sections=28), "DarkPolymer", ("suction_port",)))
    for i in range(IRRIGATION_PORT_COUNT):
        a=2*math.pi*(i+0.5)/IRRIGATION_PORT_COUNT
        x=0.024*math.cos(a); y=0.024*math.sin(a)
        mount_visuals.append(Visual(f"IrrigationJet_{i:02d}", frustum_axis(0.0014,0.00055,0.008,"z",(x,y,0.161),sections=24), "SensorBlue", ("irrigation_microjet",)))
    for i in range(CLIP_CAPACITY-1):
        clip=transform(u_clip_mesh(False,scale=0.75),(0.043,-0.022+i*0.006,0.093),rotation_matrix((1,0,0),math.pi/2))
        mount_visuals.append(Visual(f"StoredClip_{i:02d}",clip,"ClipMetal",("stored_vascular_clip",)))
    # Four replaceable patches are visible on the carousel.
    patch_visual=rounded_patch_mesh(0.022,0.016,18,14)
    for i in range(PATCH_CAPACITY):
        a=2*math.pi*i/PATCH_CAPACITY
        patch=transform(patch_visual,(-0.041+0.018*math.cos(a),0.018*math.sin(a),0.104),rotation_matrix((1,0,0),math.pi))
        mount_visuals.append(Visual(f"StoredPatch_{i:02d}",patch,"PatchMaterial",("stored_hemostatic_patch",)))
    links["Mount"] = Link("Mount",(0,0,0),mount_visuals,[
        Collider("AdapterCollider","cylinder",(0,0,0.008),radius=0.032,height=0.016,physics_material="MountPhysics"),
        Collider("HousingCollider","box",(0,0,0.058),size=(0.114,0.094,0.074),physics_material="PolymerPhysics"),
        Collider("RailCollider","box",(0,0,0.105),size=(0.152,0.028,0.022),physics_material="MountPhysics"),
        Collider("SuctionRingCollider","cylinder",(0,0,0.151),radius=0.035,height=0.008,physics_material="PolymerPhysics",role="suction_ring"),
    ],0.410,("adaptive_hemostasis_end_effector","surgical_hemostasis_device"))

    for side_name,side in (("Left",-1),("Right",1)):
        x0=side*0.043
        carriage_visuals=[
            Visual("CarriageBody",box_mesh((0.032,0.036,0.026),(0,0,0)),"AccentPolymer",("compression_carriage",)),
            Visual("LinearBearing",box_mesh((0.026,0.020,0.009),(0,0,-0.014)),"RailMetal"),
            Visual("ForceScale",box_mesh((0.020,0.0012,0.006),(0,-0.0185,0.006)),"LabelMaterial"),
            Visual("CableLoop",torus_axis(0.009,0.0011,"y",(0,0.013,0),major_sections=40,minor_sections=10),"DarkPolymer"),
        ]
        links[f"{side_name}Compression"] = Link(f"{side_name}Compression",(x0,0,0.105),carriage_visuals,[Collider("CarriageCollider","box",(0,0,0),size=(0.034,0.038,0.028),physics_material="PolymerPhysics")],0.068,("temporary_compression_carriage",))
        pad_visuals=[
            Visual("PadBacking",box_mesh((0.020,0.034,0.006),(0,0,-0.004)),"PadBacking",("compression_pad_backing",)),
            Visual("SoftContact",ellipsoid_mesh((0.010,0.017,0.0032),(0,0,0),subdivisions=3),"PadElastomer",("atraumatic_compression_contact",)),
            Visual("ForceIndicator",box_mesh((0.014,0.0013,0.004),(0,-0.0175,-0.004)),"IndicatorGreen",("compression_force_indicator",)),
        ]
        pad_colliders=[
            Collider("PadBackingCollider","box",(0,0,-0.004),size=(0.021,0.035,0.007),physics_material="PadBackingPhysics"),
            Collider("VesselCaptureVolume","box",(0,0,0.0005),size=(0.014,0.025,0.005),physics_material="PadContactPhysics",role="temporary_vessel_capture"),
        ]
        links[f"{side_name}Pad"] = Link(f"{side_name}Pad",(x0,0,0.176),pad_visuals,pad_colliders,0.038,("atraumatic_vessel_compression_pad",))

    for side_name,side in (("Left",-1),("Right",1)):
        jaw_visuals=[
            Visual("JawBody",box_mesh((0.011,0.024,0.034),(0,0,0.002)),"JawMetal",("clip_forming_jaw",)),
            Visual("AnvilFace",box_mesh((0.004,0.018,0.006),(-side*0.004,0,0.017)),"ClipMetal",("clip_forming_anvil",)),
            Visual("GuideHorn",frustum_axis(0.0038,0.0018,0.010,"z",(-side*0.004,0,0.027),sections=30),"JawMetal"),
        ]
        links[f"{side_name}ClipJaw"] = Link(f"{side_name}ClipJaw",(side*0.010,0,0.154),jaw_visuals,[
            Collider("JawCollider","box",(0,0,0.004),size=(0.012,0.026,0.036),physics_material="MetalPhysics"),
            Collider("AnvilCollider","box",(-side*0.004,0,0.017),size=(0.005,0.020,0.007),physics_material="MetalPhysics",role="clip_forming_contact"),
        ],0.052,("vascular_clip_forming_jaw",))

    chambered=transform(u_clip_mesh(False,scale=1.0),(0,0,0.016),rotation_matrix((1,0,0),math.pi/2))
    driver_visuals=[
        Visual("DriverStem",box_mesh((0.014,0.020,0.040),(0,0,0)),"JawMetal",("clip_driver",)),
        Visual("DriverBlade",box_mesh((0.010,0.018,0.004),(0,0,0.022)),"ClipMetal",("clip_driver_face",)),
        Visual("ChamberedClip",chambered,"ClipMetal",("chambered_vascular_clip",)),
    ]
    links["ClipDriver"] = Link("ClipDriver",(0,0,0.132),driver_visuals,[
        Collider("DriverCollider","box",(0,0,0),size=(0.015,0.022,0.042),physics_material="MetalPhysics"),
        Collider("ClipExitVolume","box",(0,0,0.028),size=(0.014,0.020,0.009),physics_material="MetalPhysics",role="clip_exit"),
    ],0.061,("vascular_clip_driver","clip_deployment_axis"))

    carousel_visuals=[
        Visual("CarouselDisk",cylinder_axis(0.029,0.010,"z",(0,0,0),sections=64),"AccentPolymer",("patch_carousel",)),
        Visual("CarouselHub",cylinder_axis(0.009,0.016,"z",(0,0,0),sections=48),"MountMetal"),
    ]
    links["PatchCarousel"] = Link("PatchCarousel",(-0.041,0,0.093),carousel_visuals,[Collider("CarouselCollider","cylinder",(0,0,0),radius=0.031,height=0.012,physics_material="PolymerPhysics")],0.074,("hemostatic_patch_carousel",))

    platen_visuals=[
        Visual("PlatenStem",box_mesh((0.014,0.014,0.040),(0,0,-0.018)),"DarkPolymer",("patch_applicator_stem",)),
        Visual("PatchPlaten",ellipsoid_mesh((0.016,0.013,0.0035),(0,0,0.004),subdivisions=3),"PadElastomer",("hemostatic_patch_compression_platen",)),
        Visual("LoadedPatch",transform(rounded_patch_mesh(),(0,0,0.008)),"PatchMaterial",("ready_hemostatic_patch",)),
    ]
    links["PatchPlaten"] = Link("PatchPlaten",(-0.041,0,0.128),platen_visuals,[
        Collider("StemCollider","box",(0,0,-0.018),size=(0.015,0.015,0.042),physics_material="PolymerPhysics"),
        Collider("PatchContactVolume","box",(0,0,0.005),size=(0.030,0.024,0.006),physics_material="PatchPhysics",role="patch_application_contact"),
    ],0.048,("hemostatic_patch_applicator",))

    for valve_name,x in (("SuctionValve",0.018),("IrrigationValve",-0.018)):
        visuals=[
            Visual("ValveBody",box_mesh((0.012,0.024,0.014),(0,0,0)),"DarkPolymer",(valve_name.lower(),)),
            Visual("ValveIndicator",box_mesh((0.008,0.0012,0.004),(0,-0.0125,0)),"IndicatorGreen" if valve_name=="SuctionValve" else "SensorBlue"),
        ]
        links[valve_name]=Link(valve_name,(x,0.028,0.090),visuals,[Collider("ValveCollider","box",(0,0,0),size=(0.013,0.025,0.015),physics_material="PolymerPhysics")],0.018,("metering_valve",))

    joints=[
        Joint("left_compression_joint","prismatic","Mount","LeftCompression","X",(-0.043,0,0.105),(0,0,0),0.0,0.030,5200,180,95),
        Joint("right_compression_joint","prismatic","Mount","RightCompression","X",(0.043,0,0.105),(0,0,0),-0.030,0.0,5200,180,95),
        Joint("left_pad_compliance_joint","prismatic","LeftCompression","LeftPad","Z",(0,0,0.071),(0,0,0),-0.008,0.0,1200,42,18),
        Joint("right_pad_compliance_joint","prismatic","RightCompression","RightPad","Z",(0,0,0.071),(0,0,0),-0.008,0.0,1200,42,18),
        Joint("left_clip_jaw_joint","prismatic","Mount","LeftClipJaw","X",(-0.010,0,0.154),(0,0,0),0.0,0.008,8000,260,140),
        Joint("right_clip_jaw_joint","prismatic","Mount","RightClipJaw","X",(0.010,0,0.154),(0,0,0),-0.008,0.0,8000,260,140),
        Joint("clip_driver_joint","prismatic","Mount","ClipDriver","Z",(0,0,0.132),(0,0,0),0.0,0.018,15000,320,220),
        Joint("patch_carousel_joint","revolute","Mount","PatchCarousel","Z",(-0.041,0,0.093),(0,0,0),0.0,90.0,38,2.0,6.0),
        Joint("patch_applicator_joint","prismatic","PatchCarousel","PatchPlaten","Z",(0,0,0.035),(0,0,0),0.0,0.038,6000,190,85),
        Joint("suction_valve_joint","prismatic","Mount","SuctionValve","Y",(0.018,0.028,0.090),(0,0,0),0.0,0.008,1500,48,25),
        Joint("irrigation_valve_joint","prismatic","Mount","IrrigationValve","Y",(-0.018,0.028,0.090),(0,0,0),0.0,0.008,1500,48,25),
    ]

    frames={
        "panda_link8_mount":{"position":[0,0,0],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"robot_mount"},
        "hemostasis_tcp":{"position":[0,0,WORK_PLANE_Z],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"tool_center_point"},
        "bleeding_source_reference":{"position":[0,0,WORK_PLANE_Z],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"bleeding_source_localization"},
        "suction_center":{"position":[0,0,0.161],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"annular_suction_center"},
        "irrigation_center":{"position":[0,0,0.164],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"irrigation_center"},
        "clip_forming_center":{"position":[0,0,0.181],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"clip_forming_center"},
        "clip_exit":{"position":[0,0,0.181],"orientation_wxyz":[1,0,0,0],"parent_link":"ClipDriver","role":"clip_exit"},
        "patch_application":{"position":[0,0,0.010],"orientation_wxyz":[1,0,0,0],"parent_link":"PatchPlaten","role":"patch_application_center"},
        "rgb_camera_left":{"position":[-0.013,-0.052,0.084],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"rgb_camera"},
        "rgb_camera_right":{"position":[0.013,-0.052,0.084],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"rgb_camera"},
        "fluorescence_camera":{"position":[0,-0.052,0.096],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"fluorescence_camera"},
        "flow_probe":{"position":[0,0,0.146],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"flow_verification_probe"},
        "count_reference":{"position":[0,-0.050,0.052],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"inventory_reference"},
        "disposal_reference":{"position":[0,0.048,0.060],"orientation_wxyz":[1,0,0,0],"parent_link":"Mount","role":"disposal_reference"},
    }
    for side_name in ("Left","Right"):
        frames[f"{side_name.lower()}_compression_contact"]={"position":[0,0,0],"orientation_wxyz":[1,0,0,0],"parent_link":f"{side_name}Pad","role":"temporary_vessel_compression_contact"}
        frames[f"{side_name.lower()}_clip_contact"]={"position":[0,0,0.017],"orientation_wxyz":[1,0,0,0],"parent_link":f"{side_name}ClipJaw","role":"vascular_clip_forming_contact"}

    vessel=tube_wall_mesh()
    open_clip=u_clip_mesh(False)
    formed_clip=u_clip_mesh(True)
    patch=rounded_patch_mesh()
    vessel_base=box_mesh((0.095,0.095,0.012),(0,0,-0.014))
    return ToolBundle(links,joints,frames,vessel,open_clip,formed_clip,patch,vessel_base)


# ---------------------------- OpenUSD authoring ----------------------------

def mesh_usda(visual: Visual, material_path: str, indent: str = "                ") -> str:
    mesh=visual.mesh.copy()
    mesh.remove_unreferenced_vertices(); mesh.fix_normals()
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
{indent}    normal3f[] primvars:normals = [
{normals}
{indent}    ] (
{indent}        interpolation = "vertex"
{indent}    )
{indent}    uniform token subdivisionScheme = "none"
{indent}    rel material:binding = <{material_path}>{label_attr}
{indent}}}'''


def collider_usda(collider: Collider, root_path: str, indent: str = "                ") -> str:
    enabled = "true" if collider.author_enabled else "false"
    common_lines = f'''{indent}    bool physics:collisionEnabled = {enabled}
{indent}    rel material:binding:physics = <{root_path}/PhysicsMaterials/{collider.physics_material}>
{indent}    custom string drAnmar:role = "{collider.role}"
{indent}    quatf xformOp:orient = {quat(collider.orientation_wxyz)}
{indent}    double3 xformOp:translate = {vec(collider.center)}'''
    if collider.kind == "box":
        assert collider.size is not None
        size = max(collider.size)
        scale = tuple(v / size for v in collider.size)
        return f'''{indent}def Cube "{collider.name}" (
{indent}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    double size = {f(size)}
{indent}    float3 xformOp:scale = {vec(scale)}
{common_lines}
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
{indent}}}'''
    if collider.kind == "cylinder":
        assert collider.radius is not None and collider.height is not None
        return f'''{indent}def Cylinder "{collider.name}" (
{indent}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    uniform token axis = "{collider.axis.upper()}"
{indent}    double radius = {f(collider.radius)}
{indent}    double height = {f(collider.height)}
{common_lines}
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{indent}}}'''
    if collider.kind == "sphere":
        assert collider.radius is not None
        return f'''{indent}def Sphere "{collider.name}" (
{indent}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]
{indent})
{indent}{{
{indent}    double radius = {f(collider.radius)}
{common_lines}
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{indent}}}'''
    raise ValueError(collider.kind)


def shader_material(root_name: str, name: str, color: tuple[float,float,float], metallic: float, roughness: float, opacity: float=1.0, texture: str|None=None) -> str:
    texture_block=""
    diffuse_input=f"{vec(color)}"
    if texture:
        texture_block=f'''
            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{texture}@
                float2 inputs:st.connect = </{root_name}/Looks/{name}/PrimvarReader.outputs:result>
                token inputs:sourceColorSpace = "sRGB"
                float3 outputs:rgb
            }}
            def Shader "PrimvarReader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
        string inputs:varname = "st"
                float2 outputs:result
            }}'''
        diffuse_input=f"</{root_name}/Looks/{name}/Texture.outputs:rgb>"
    return f'''        def Material "{name}"
        {{
            token outputs:surface.connect = </{root_name}/Looks/{name}/Shader.outputs:surface>
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {diffuse_input}
                float inputs:metallic = {f(metallic)}
                float inputs:roughness = {f(roughness)}
                float inputs:opacity = {f(opacity)}
                token outputs:surface
            }}{texture_block}
        }}'''


def visual_materials_scope(root_name: str) -> str:
    mats={
        "BodyPolymer":((0.84,0.86,0.88),0.0,0.24,1.0,None),
        "AccentPolymer":((0.08,0.35,0.62),0.0,0.25,1.0,None),
        "DarkPolymer":((0.055,0.065,0.075),0.0,0.30,1.0,None),
        "MountMetal":((0.46,0.50,0.55),0.85,0.20,1.0,None),
        "RailMetal":((0.28,0.32,0.37),0.82,0.24,1.0,None),
        "JawMetal":((0.55,0.58,0.62),0.90,0.18,1.0,None),
        "ClipMetal":((0.66,0.68,0.71),0.90,0.16,1.0,None),
        "PadBacking":((0.18,0.20,0.23),0.0,0.38,1.0,None),
        "PadElastomer":((0.10,0.46,0.56),0.0,0.52,1.0,None),
        "SensorGlass":((0.06,0.12,0.18),0.05,0.08,0.78,None),
        "SensorBlue":((0.08,0.48,0.88),0.0,0.20,1.0,None),
        "SensorPurple":((0.48,0.12,0.72),0.0,0.18,1.0,None),
        "IndicatorGreen":((0.12,0.78,0.34),0.0,0.18,1.0,None),
        "IndicatorRed":((0.88,0.08,0.08),0.0,0.18,1.0,None),
        "FluidBlue":((0.10,0.52,0.92),0.0,0.08,0.72,None),
        "Blood":((0.42,0.012,0.016),0.0,0.28,1.0,"./textures/blood_basecolor.png"),
        "VesselMaterial":((0.50,0.055,0.048),0.0,0.46,1.0,"./textures/vessel_basecolor.png"),
        "PatchMaterial":((0.91,0.84,0.55),0.0,0.68,1.0,"./textures/hemostatic_patch_basecolor.png"),
        "TubeClear":((0.72,0.82,0.88),0.0,0.10,0.35,None),
        "LabelMaterial":((0.96,0.97,0.98),0.0,0.38,1.0,"./textures/label_dranmar.png"),
        "GuideRed":((1.0,0.08,0.08),0.0,0.25,1.0,None),
        "GuideGreen":((0.08,1.0,0.08),0.0,0.25,1.0,None),
        "GuideBlue":((0.08,0.24,1.0),0.0,0.25,1.0,None),
        "CollisionDebug":((1.0,0.18,0.04),0.0,0.45,0.34,None),
    }
    return "    def Scope \"Looks\"\n    {\n"+"\n\n".join(shader_material(root_name,n,*v) for n,v in mats.items())+"\n    }"


def physics_materials_scope() -> str:
    specs={
        "PolymerPhysics":(0.50,0.38,0.03),
        "MountPhysics":(0.32,0.24,0.02),
        "MetalPhysics":(0.28,0.20,0.02),
        "PadBackingPhysics":(0.65,0.52,0.02),
        "PadContactPhysics":(0.78,0.64,0.01),
        "PatchPhysics":(0.72,0.58,0.01),
        "ClipPhysics":(0.31,0.23,0.02),
        "VesselPhysics":(0.52,0.38,0.0),
    }
    blocks=[]
    for name,(sf,df,r) in specs.items():
        blocks.append(f'''        def Material "{name}" (
            prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
        )
        {{
            float physics:staticFriction = {f(sf)}
            float physics:dynamicFriction = {f(df)}
            float physics:restitution = {f(r)}
            token physxMaterial:frictionCombineMode = "max"
            token physxMaterial:restitutionCombineMode = "min"
        }}''')
    return "    def Scope \"PhysicsMaterials\"\n    {\n"+"\n\n".join(blocks)+"\n    }"


def frame_usda(name: str, data: dict[str,object], indent: str="                ") -> str:
    return f'''{indent}def Xform "{name}"
{indent}{{
{indent}    custom string drAnmar:role = "{data['role']}"
{indent}    quatf xformOp:orient = {quat(data['orientation_wxyz'])}
{indent}    double3 xformOp:translate = {vec(data['position'])}
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{indent}}}'''


def link_usda(link: Link, root_path: str, frames: dict[str,dict[str,object]]) -> str:
    schemas='prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]' if link.mass_kg is not None else ''
    body_attrs=''
    if link.mass_properties:
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
        int physxRigidBody:solverPositionIterationCount = 16
        int physxRigidBody:solverVelocityIterationCount = 4\n'''
    visual_blocks="\n".join(mesh_usda(v,f"{root_path}/Looks/{v.material}") for v in link.visuals)
    collider_blocks="\n".join(collider_usda(c,root_path) for c in link.colliders)
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
        typename="PhysicsPrismaticJoint"; axis_line=f'        uniform token physics:axis = "{joint.axis}"\n'; drive="linear"
    elif joint.type == "revolute":
        typename="PhysicsRevoluteJoint"; axis_line=f'        uniform token physics:axis = "{joint.axis}"\n'; drive="angular"
    elif joint.type == "fixed":
        typename="PhysicsFixedJoint"; axis_line=""; drive=None
    else:
        raise ValueError(joint.type)
    api=f'prepend apiSchemas = ["PhysicsDriveAPI:{drive}"]' if drive else ''
    limits=""
    drive_block=""
    if joint.lower is not None and joint.upper is not None:
        limits=f'''        float physics:lowerLimit = {f(joint.lower)}
        float physics:upperLimit = {f(joint.upper)}\n'''
    if drive:
        drive_block=f'''        uniform token drive:{drive}:physics:type = "force"
        float drive:{drive}:physics:stiffness = {f(joint.stiffness)}
        float drive:{drive}:physics:damping = {f(joint.damping)}
        float drive:{drive}:physics:maxForce = {f(joint.max_force)}
        float drive:{drive}:physics:targetPosition = 0
        float drive:{drive}:physics:targetVelocity = {f(joint.target_velocity)}\n'''
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


def _nested_over(path: Sequence[str], body_lines: Sequence[str], *, indent: str = "            ") -> str:
    lines: list[str] = []
    for depth, name in enumerate(path):
        prefix = indent + "    " * depth
        lines.append(f'{prefix}over "{name}"')
        lines.append(f"{prefix}{{")
    body_prefix = indent + "    " * len(path)
    lines.extend(f"{body_prefix}{line}" for line in body_lines)
    for depth in reversed(range(len(path))):
        prefix = indent + "    " * depth
        lines.append(f"{prefix}}}")
    return "\n".join(lines)


def state_variants(root_name: str) -> str:
    del root_name  # The paths are relative to the asset root.
    clip_names = [f"StoredClip_{i:02d}" for i in range(CLIP_CAPACITY - 1)]
    patch_names = [f"StoredPatch_{i:02d}" for i in range(PATCH_CAPACITY)]

    def visibility_children(path: Sequence[str], names: Sequence[str], value: str) -> str:
        child_blocks = []
        for name in names:
            child_blocks.append(
                _nested_over([name], [f'token visibility = "{value}"'], indent="                        ")
            )
        return _nested_over(path, "\n".join(child_blocks).splitlines(), indent="            ")

    def visibility_leaf(path: Sequence[str], value: str) -> str:
        return _nested_over(path, [f'token visibility = "{value}"'], indent="            ")

    def grouped_visibility(
        inventory_names: Sequence[str], active_path: Sequence[str], value: str
    ) -> str:
        inventory = visibility_children(["Mount", "Visuals"], inventory_names, value)
        active = visibility_leaf(active_path, value)
        return _nested_over(
            ["Links"],
            (inventory + "\n" + active).splitlines(),
            indent="            ",
        )

    clip_loaded = grouped_visibility(
        clip_names, ["ClipDriver", "Visuals", "ChamberedClip"], "inherited"
    )
    clip_empty = grouped_visibility(
        clip_names, ["ClipDriver", "Visuals", "ChamberedClip"], "invisible"
    )

    patch_loaded = grouped_visibility(
        patch_names, ["PatchPlaten", "Visuals", "LoadedPatch"], "inherited"
    )
    patch_empty = grouped_visibility(
        patch_names, ["PatchPlaten", "Visuals", "LoadedPatch"], "invisible"
    )

    irrigation_full = visibility_leaf(["Links", "Mount", "Visuals", "IrrigationFluid"], "inherited")
    irrigation_empty = visibility_leaf(["Links", "Mount", "Visuals", "IrrigationFluid"], "invisible")
    collection_empty = visibility_leaf(["Links", "Mount", "Visuals", "CollectionFill"], "invisible")
    collection_visible = visibility_leaf(["Links", "Mount", "Visuals", "CollectionFill"], "inherited")

    return f'''    variantSet "clip_state" = {{
        "loaded"
        {{
{clip_loaded}
        }}
        "empty"
        {{
{clip_empty}
        }}
    }}
    variantSet "patch_state" = {{
        "loaded"
        {{
{patch_loaded}
        }}
        "empty"
        {{
{patch_empty}
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
    }}'''


def tool_usda(bundle: ToolBundle, articulation_root: bool) -> str:
    root=STANDALONE_ROOT if articulation_root else ROOT_PRIM
    root_path=f"/{root}"
    schemas='prepend apiSchemas = ["PhysicsArticulationRootAPI"]' if articulation_root else ''
    schema_line=f"    {schemas}\n" if schemas else ""
    links="\n\n".join(link_usda(link,root_path,bundle.frames) for link in bundle.links.values())
    joints="\n\n".join(joint_usda(j,root_path) for j in bundle.joints)
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "{ASSET_NAME}: field clearing, temporary compression, vascular clipping, patch application, and leak verification research asset."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
{schema_line}\
    prepend variantSets = ["clip_state", "patch_state", "irrigation_state", "collection_state"]
    variants = {{
        string clip_state = "loaded"
        string patch_state = "loaded"
        string irrigation_state = "full"
        string collection_state = "empty"
    }}
    customData = {{
        string drAnmarAssetId = "dranmar-adaptive-hemostasis-robot-v1"
        string drAnmarAssetVersion = "{VERSION}"
        bool drAnmarClinicalValidation = false
        bool drAnmarMedicalDevice = false
        string drAnmarStatus = "research_only_runtime_qualified_physical_calibration_pending"
        string drAnmarMount = "replaces_panda_hand_at_panda_link8"
        int drAnmarClipCapacity = {CLIP_CAPACITY}
        int drAnmarPatchCapacity = {PATCH_CAPACITY}
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
{state_variants(root)}
}}
'''


def material_color(material: str) -> tuple[int,int,int,int]:
    colors={
        "BodyPolymer":(214,219,224,255),"AccentPolymer":(25,92,158,255),"DarkPolymer":(18,22,26,255),
        "MountMetal":(122,132,143,255),"RailMetal":(76,84,94,255),"JawMetal":(150,156,164,255),
        "ClipMetal":(180,184,190,255),"PadBacking":(48,52,58,255),"PadElastomer":(36,146,166,255),
        "SensorGlass":(25,55,78,210),"SensorBlue":(28,126,230,255),"SensorPurple":(126,40,184,255),
        "IndicatorGreen":(40,202,86,255),"IndicatorRed":(224,28,30,255),"FluidBlue":(32,135,232,180),
        "Blood":(112,5,10,255),"VesselMaterial":(138,25,22,255),"PatchMaterial":(222,196,116,255),
        "TubeClear":(190,216,228,110),"LabelMaterial":(246,248,250,255),"CollisionDebug":(255,70,18,95),
        "GuideRed":(255,30,30,255),"GuideGreen":(30,255,30,255),"GuideBlue":(30,80,255,255),
    }
    return colors.get(material,(180,180,180,255))


def pbr(mesh: trimesh.Trimesh, material: str) -> trimesh.Trimesh:
    m=mesh.copy(); color=material_color(material)
    m.visual.vertex_colors=np.tile(np.asarray(color,dtype=np.uint8),(len(m.vertices),1))
    return m


def rigid_proxy_usda(bundle: ToolBundle) -> str:
    visuals=[]
    for link in bundle.links.values():
        for v in link.visuals:
            mesh=transform(v.mesh,link.translation)
            visuals.append(Visual(f"{link.name}_{v.name}",mesh,v.material,v.labels))
    bmin,bmax=mesh_bounds([v.mesh for v in visuals])
    mp=box_mass_properties([v.mesh for v in visuals],0.90)
    blocks="\n".join(mesh_usda(v,f"/{PROXY_ROOT}/Looks/{v.material}",indent="        ") for v in visuals)
    size=bmax-bmin; center=(bmin+bmax)/2
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
        string drAnmarRepresentation = "rigid_perception_planning_proxy"
        bool drAnmarClinicalValidation = false
    }}
)
{{
    bool physics:rigidBodyEnabled = true
    float physics:mass = {f(mp['mass_kg'])}
    point3f physics:centerOfMass = {vec(mp['center_of_mass_m'])}
    vector3f physics:diagonalInertia = {vec(mp['diagonal_inertia_kg_m2'])}
    quatf physics:principalAxes = (1, 0, 0, 0)
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
        double size = {f(float(max(size)))}
        float3 xformOp:scale = {vec(tuple(float(v/max(size)) for v in size))}
        double3 xformOp:translate = {vec(center)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        rel material:binding:physics = </{PROXY_ROOT}/PhysicsMaterials/PolymerPhysics>
    }}
}}
'''


def simple_mesh_asset(root: str, visuals: list[Visual], colliders: list[Collider], mass: float|None, *, variants: str="", custom: str="") -> str:
    root_path=f"/{root}"
    schemas='prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]' if mass is not None else ''
    schema_line=f"    {schemas}\n" if schemas else ""
    variants_line=f"    {variants}\n" if variants else ""
    custom_line=f"        {custom}\n" if custom else ""
    body=""
    if mass is not None:
        mp=box_mass_properties([v.mesh for v in visuals],mass)
        body=f'''    bool physics:rigidBodyEnabled = true
    bool physics:kinematicEnabled = false
    float physics:mass = {f(mass)}
    point3f physics:centerOfMass = {vec(mp['center_of_mass_m'])}
    vector3f physics:diagonalInertia = {vec(mp['diagonal_inertia_kg_m2'])}
    quatf physics:principalAxes = (1, 0, 0, 0)
    bool physxRigidBody:enableCCD = true\n'''
    vb="\n".join(mesh_usda(v,f"{root_path}/Looks/{v.material}",indent="        ") for v in visuals)
    cb="\n".join(collider_usda(c,root_path,indent="        ") for c in colliders)
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
{schema_line}{variants_line}\
    customData = {{
        bool drAnmarClinicalValidation = false
        string drAnmarStatus = "research_only"
{custom_line}\
    }}
)
{{
{body}{visual_materials_scope(root)}
{physics_materials_scope()}
    def Scope "Visuals"
    {{
{vb}
    }}
    def Scope "Collisions"
    {{
{cb}
    }}
}}
'''


def clip_usda(bundle: ToolBundle) -> str:
    open_v = Visual("OpenClip", bundle.open_clip, "ClipMetal", ("open_vascular_clip",))
    formed_v = Visual("FormedClip", bundle.formed_clip, "ClipMetal", ("formed_vascular_clip",))
    colliders = [
        Collider("LeftLegAttachment", "box", (-0.0009, 0, -0.0038), size=(0.0016, 0.0030, 0.0048), physics_material="ClipPhysics", role="left_vessel_attachment", author_enabled=False),
        Collider("RightLegAttachment", "box", (0.0009, 0, -0.0038), size=(0.0016, 0.0030, 0.0048), physics_material="ClipPhysics", role="right_vessel_attachment", author_enabled=False),
        Collider("CrownCollider", "box", (0, 0, 0.0026), size=(0.0080, 0.0024, 0.0020), physics_material="ClipPhysics", role="clip_crown"),
    ]
    variants = '''prepend variantSets = "state"
    variants = { string state = "open" }'''
    text = simple_mesh_asset(
        CLIP_ROOT,
        [open_v, formed_v],
        colliders,
        0.000040,
        variants=variants,
        custom='string drAnmarAssetId = "dranmar-hemostatic-clip"',
    )
    insert = '''    variantSet "state" = {
        "open"
        {
            over "Visuals"
            {
                over "OpenClip"
                {
                    token visibility = "inherited"
                }
                over "FormedClip"
                {
                    token visibility = "invisible"
                }
            }
            over "Collisions"
            {
                over "LeftLegAttachment"
                {
                    bool physics:collisionEnabled = false
                }
                over "RightLegAttachment"
                {
                    bool physics:collisionEnabled = false
                }
            }
        }
        "formed"
        {
            over "Visuals"
            {
                over "OpenClip"
                {
                    token visibility = "invisible"
                }
                over "FormedClip"
                {
                    token visibility = "inherited"
                }
            }
            over "Collisions"
            {
                over "LeftLegAttachment"
                {
                    bool physics:collisionEnabled = true
                }
                over "RightLegAttachment"
                {
                    bool physics:collisionEnabled = true
                }
            }
        }
    }
'''
    stripped = text.rstrip()
    if not stripped.endswith("}"):
        raise ValueError("clip asset root did not end with a closing brace")
    return stripped[:-1] + "\n" + insert + "}\n"


def patch_surface_usda(bundle: ToolBundle) -> str:
    mesh=bundle.patch.copy(); mesh.fix_normals()
    v=Visual("SimulationMesh",mesh,"PatchMaterial",("hemostatic_patch","deformable_ready_surface"))
    block=mesh_usda(v,f"/{PATCH_ROOT}/Looks/PatchMaterial",indent="    ")
    return f'''#usda 1.0
(
    defaultPrim = "{PATCH_ROOT}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{PATCH_ROOT}" (
    customData = {{
        bool drAnmarClinicalValidation = false
        string drAnmarRepresentation = "portable_triangular_surface_for_runtime_deformable_cooking"
    }}
)
{{
{visual_materials_scope(PATCH_ROOT)}
{block}
    def Scope "Frames"
    {{
        def Xform "patch_center"
        {{
            custom string drAnmar:role = "patch_center"
        }}
        def Xform "application_normal"
        {{
            custom string drAnmar:role = "application_normal"
        }}
        def Xform "count_reference"
        {{
            custom string drAnmar:role = "count_reference"
        }}
    }}
}}
'''


def patch_proxy_usda(bundle: ToolBundle) -> str:
    visuals=[Visual("PatchVisual",transform(bundle.patch,(0,0,0.0008)),"PatchMaterial",("hemostatic_patch","rigid_bond_carrier"))]
    cells=[]
    positions=[(-0.009,-0.006,0),(-0.003,-0.006,0),(0.003,-0.006,0),(0.009,-0.006,0),(-0.009,0.006,0),(-0.003,0.006,0),(0.003,0.006,0),(0.009,0.006,0)]
    for i,p in enumerate(positions):
        cells.append(Collider(f"BondCell_{i:02d}","box",p,size=(0.0065,0.0060,0.0035),physics_material="PatchPhysics",role="hemostatic_patch_bond_cell"))
    return simple_mesh_asset(PATCH_PROXY_ROOT,visuals,cells,0.00055,custom='string drAnmarRepresentation = "rigid_patch_bond_carrier"')


def vessel_usda(bundle: ToolBundle) -> str:
    vessel=Visual("VesselWall",bundle.vessel,"VesselMaterial",("bleeding_vessel","surface_deformable_ready"))
    base=Visual("FixtureBase",bundle.vessel_base,"DarkPolymer",("vessel_fixture",))
    anchor_min=Visual(
        "AnchorMin",
        transform(trimesh.creation.box(extents=(0.012,0.010,0.010)),(0,-0.034,0)),
        "RailMetal",
        ("vessel_fixture_anchor","negative_y"),
    )
    anchor_max=Visual(
        "AnchorMax",
        transform(trimesh.creation.box(extents=(0.012,0.010,0.010)),(0,0.034,0)),
        "RailMetal",
        ("vessel_fixture_anchor","positive_y"),
    )
    port=Visual("BleedingPort",torus_axis(0.0020,0.00035,"y",(0.0031,0,0.00055),major_sections=44,minor_sections=10),"Blood",("bleeding_source",))
    blocks="\n".join(mesh_usda(v,f"/{VESSEL_ROOT}/Looks/{v.material}",indent="    ") for v in (vessel,base,anchor_min,anchor_max,port))
    return f'''#usda 1.0
(
    defaultPrim = "{VESSEL_ROOT}"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{VESSEL_ROOT}" (
    customData = {{
        string drAnmarAssetId = "dranmar-bleeding-vessel-demo"
        bool drAnmarClinicalValidation = false
        float drAnmarReferencePressurePa = 10665.8
        float drAnmarDefectAreaM2 = 1.8e-6
        string drAnmarStatus = "research_only_reduced_order_flow_source"
    }}
)
{{
{visual_materials_scope(VESSEL_ROOT)}
{physics_materials_scope()}
{blocks}
    def Scope "Frames"
    {{
        def Xform "flow_source"
        {{
            custom string drAnmar:role = "blood_flow_source"
            double3 xformOp:translate = (0.00325, 0, 0.00055)
            quatf xformOp:orient = (0.70710678, 0, 0.70710678, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }}
        def Xform "clip_center"
        {{
            custom string drAnmar:role = "clip_center"
        }}
        def Xform "left_clip_region"
        {{
            custom string drAnmar:role = "left_clip_region"
            double3 xformOp:translate = (-0.0024, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "right_clip_region"
        {{
            custom string drAnmar:role = "right_clip_region"
            double3 xformOp:translate = (0.0024, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "patch_center"
        {{
            custom string drAnmar:role = "patch_center"
            double3 xformOp:translate = (0.0040, 0, 0.0010)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "anchor_min"
        {{
            custom string drAnmar:role = "vessel_anchor"
            double3 xformOp:translate = (0, -0.034, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
        def Xform "anchor_max"
        {{
            custom string drAnmar:role = "vessel_anchor"
            double3 xformOp:translate = (0, 0.034, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}
}}
'''


def droplet_usda() -> str:
    visual=Visual("Droplet",trimesh.creation.icosphere(subdivisions=2,radius=0.00065),"Blood",("blood_particle",))
    return simple_mesh_asset(DROPLET_ROOT,[visual],[],None,custom='string drAnmarRepresentation = "particle_prototype"')


def export_scene(path: Path, entries: Sequence[tuple[str,trimesh.Trimesh,str]]) -> None:
    scene=trimesh.Scene()
    for name,mesh,material in entries:
        scene.add_geometry(pbr(mesh,material),node_name=name,geom_name=name)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))


def phase_parameters(phase: str) -> dict[str,float]:
    phases={
        "inspect":dict(lc=0,rc=0,lp=0,rp=0,lj=0,rj=0,driver=0,carousel=0,platen=0,suction=0,irrigation=0),
        "clear":dict(lc=0,rc=0,lp=0,rp=0,lj=0,rj=0,driver=0,carousel=0,platen=0,suction=0.008,irrigation=0.005),
        "compress":dict(lc=0.026,rc=-0.026,lp=-0.005,rp=-0.005,lj=0,rj=0,driver=0,carousel=0,platen=0,suction=0.006,irrigation=0),
        "clip":dict(lc=0.026,rc=-0.026,lp=-0.005,rp=-0.005,lj=0.007,rj=-0.007,driver=0.017,carousel=0,platen=0,suction=0.004,irrigation=0),
        "patch":dict(lc=0.012,rc=-0.012,lp=0,rp=0,lj=0,rj=0,driver=0,carousel=math.radians(90),platen=0.034,suction=0.002,irrigation=0),
        "verify":dict(lc=0,rc=0,lp=0,rp=0,lj=0,rj=0,driver=0,carousel=math.radians(90),platen=0,suction=0.004,irrigation=0),
        "complete":dict(lc=0,rc=0,lp=0,rp=0,lj=0,rj=0,driver=0,carousel=0,platen=0,suction=0,irrigation=0),
    }
    return phases[phase]


def link_world_transform(bundle: ToolBundle, link_name: str, phase: str) -> np.ndarray:
    p=phase_parameters(phase); link=bundle.links[link_name]; t=np.asarray(link.translation,dtype=float); R=np.eye(3)
    if link_name=="LeftCompression": t=t+np.asarray([p["lc"],0,0])
    elif link_name=="RightCompression": t=t+np.asarray([p["rc"],0,0])
    elif link_name=="LeftPad": t=t+np.asarray([p["lc"],0,p["lp"]])
    elif link_name=="RightPad": t=t+np.asarray([p["rc"],0,p["rp"]])
    elif link_name=="LeftClipJaw": t=t+np.asarray([p["lj"],0,0])
    elif link_name=="RightClipJaw": t=t+np.asarray([p["rj"],0,0])
    elif link_name=="ClipDriver": t=t+np.asarray([0,0,p["driver"]])
    elif link_name=="PatchCarousel": R=rotation_matrix((0,0,1),p["carousel"])
    elif link_name=="PatchPlaten": t=t+np.asarray([0,0,p["platen"]]); R=rotation_matrix((0,0,1),p["carousel"])
    elif link_name=="SuctionValve": t=t+np.asarray([0,p["suction"],0])
    elif link_name=="IrrigationValve": t=t+np.asarray([0,p["irrigation"],0])
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;return T


def world_visual_entries(bundle: ToolBundle, phase: str="inspect") -> list[tuple[str,trimesh.Trimesh,str]]:
    out=[]
    for link_name,link in bundle.links.items():
        T=link_world_transform(bundle,link_name,phase)
        for v in link.visuals:
            m=v.mesh.copy();m.apply_transform(T);out.append((f"{link_name}_{v.name}",m,v.material))
    return out


def collider_mesh(c: Collider) -> trimesh.Trimesh:
    if c.kind=="box": assert c.size is not None; m=box_mesh(c.size,c.center)
    elif c.kind=="cylinder": assert c.radius is not None and c.height is not None; m=cylinder_axis(c.radius,c.height,c.axis,c.center)
    elif c.kind=="sphere": assert c.radius is not None; m=ellipsoid_mesh((c.radius,c.radius,c.radius),c.center,2)
    else: raise ValueError(c.kind)
    R=np.asarray(rotation_matrix((1,0,0),0));
    if c.orientation_wxyz!=(1,0,0,0):
        # Debug exports only; most colliders use identity.
        pass
    return m


def collision_debug_entries(bundle: ToolBundle, phase: str="compress") -> list[tuple[str,trimesh.Trimesh,str]]:
    out=world_visual_entries(bundle,phase)
    for link_name,link in bundle.links.items():
        T=link_world_transform(bundle,link_name,phase)
        for c in link.colliders:
            m=collider_mesh(c);m.apply_transform(T);out.append((f"{link_name}_{c.name}",m,"CollisionDebug"))
    return out


def axis_entries(bundle: ToolBundle, phase: str="inspect", length: float=0.012, radius: float=0.00045) -> list[tuple[str,trimesh.Trimesh,str]]:
    out=[]
    for name,data in bundle.frames.items():
        T=link_world_transform(bundle,str(data["parent_link"]),phase)
        p=T[:3,:3]@np.asarray(data["position"],dtype=float)+T[:3,3]
        R=T[:3,:3]
        axes=[(R[:,0],"GuideRed"),(R[:,1],"GuideGreen"),(R[:,2],"GuideBlue")]
        for i,(d,mat) in enumerate(axes):
            out.append((f"{name}_{i}",capsule_between(p,p+d*length,radius),mat))
    return out


def vessel_entries(bundle: ToolBundle, sealed: bool=False) -> list[tuple[str,trimesh.Trimesh,str]]:
    out=[
        ("FixtureBase",bundle.vessel_base,"DarkPolymer"),
        ("AnchorMin",transform(trimesh.creation.box(extents=(0.012,0.010,0.010)),(0,-0.034,0)),"RailMetal"),
        ("AnchorMax",transform(trimesh.creation.box(extents=(0.012,0.010,0.010)),(0,0.034,0)),"RailMetal"),
        ("Vessel",transform(bundle.vessel,(0,0,0)),"VesselMaterial"),
    ]
    if not sealed:
        out.append(("BleedingPort",torus_axis(0.0020,0.00035,"y",(0.0031,0,0.00055),major_sections=44,minor_sections=10),"Blood"))
    return out


def franka_proxy_entries(bundle: ToolBundle, phase: str="inspect") -> list[tuple[str,trimesh.Trimesh,str]]:
    # Lightweight inspection proxy with Franka-like proportions.
    out=[]
    joints=[(0,0,0.05),(0,0,0.30),(0.18,0,0.48),(0.05,0,0.68),(0.22,0,0.83),(0.05,0,1.00),(0.14,0,1.12),(0.14,0,1.22)]
    for i,(a,b) in enumerate(zip(joints[:-1],joints[1:])):
        out.append((f"ArmLink_{i:02d}",capsule_between(a,b,0.035 if i<3 else 0.028),"BodyPolymer"))
    for i,p in enumerate(joints): out.append((f"ArmJoint_{i:02d}",ellipsoid_mesh((0.045,0.045,0.045),p,2),"AccentPolymer"))
    tool_offset=np.asarray([0.14,0,1.22])
    for n,m,mat in world_visual_entries(bundle,phase): out.append((n,transform(m,tool_offset),mat))
    return out


def exploded_entries(bundle: ToolBundle) -> list[tuple[str,trimesh.Trimesh,str]]:
    out=[]
    offsets={"Mount":(0,0,0),"LeftCompression":(-0.06,0,0),"RightCompression":(0.06,0,0),"LeftPad":(-0.08,0,0.03),"RightPad":(0.08,0,0.03),"LeftClipJaw":(-0.025,-0.03,0.02),"RightClipJaw":(0.025,-0.03,0.02),"ClipDriver":(0,0,0.05),"PatchCarousel":(-0.03,0.04,0),"PatchPlaten":(-0.03,0.04,0.05),"SuctionValve":(0.02,0.05,0),"IrrigationValve":(-0.02,0.05,0)}
    for link_name,link in bundle.links.items():
        T=link_world_transform(bundle,link_name,"inspect");T[:3,3]+=np.asarray(offsets.get(link_name,(0,0,0)))
        for v in link.visuals:
            m=v.mesh.copy();m.apply_transform(T);out.append((f"{link_name}_{v.name}",m,v.material))
    return out


def export_glbs(bundle: ToolBundle) -> list[Path]:
    outputs=[]
    for phase in ("inspect","clear","compress","clip","patch","verify","complete"):
        p=GLB_ROOT/f"dranmar_hemostasis_tool_{phase}.glb";export_scene(p,world_visual_entries(bundle,phase));outputs.append(p)
    p=GLB_ROOT/"dranmar_hemostasis_tool_exploded.glb";export_scene(p,exploded_entries(bundle));outputs.append(p)
    p=GLB_ROOT/"dranmar_hemostasis_tool_collision_debug.glb";export_scene(p,collision_debug_entries(bundle,"compress"));outputs.append(p)
    p=GLB_ROOT/"dranmar_hemostasis_tool_frame_debug.glb";export_scene(p,world_visual_entries(bundle,"inspect")+axis_entries(bundle,"inspect"));outputs.append(p)
    p=GLB_ROOT/"dranmar_franka_hemostasis_assembly.glb";export_scene(p,franka_proxy_entries(bundle,"inspect"));outputs.append(p)
    p=GLB_ROOT/"dranmar_hemostatic_clip_open.glb";export_scene(p,[("OpenClip",bundle.open_clip,"ClipMetal")]);outputs.append(p)
    p=GLB_ROOT/"dranmar_hemostatic_clip_formed.glb";export_scene(p,[("FormedClip",bundle.formed_clip,"ClipMetal")]);outputs.append(p)
    p=GLB_ROOT/"dranmar_hemostatic_patch.glb";export_scene(p,[("Patch",bundle.patch,"PatchMaterial")]);outputs.append(p)
    p=GLB_ROOT/"dranmar_bleeding_vessel.glb";export_scene(p,vessel_entries(bundle,False));outputs.append(p)
    p=GLB_ROOT/"dranmar_sealed_vessel.glb";export_scene(p,vessel_entries(bundle,True)+[("FormedClip",transform(bundle.formed_clip,(0,0,0.006),rotation_matrix((1,0,0),math.pi/2)),"ClipMetal"),("Patch",transform(bundle.patch,(0.004,0,0.004),rotation_matrix((0,1,0),math.pi/2)),"PatchMaterial")]);outputs.append(p)
    return outputs


def add_mesh_to_axis(ax, mesh: trimesh.Trimesh, material: str, max_faces: int=1200) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces=np.asarray(mesh.faces)
    if len(faces)>max_faces:
        faces=faces[np.linspace(0,len(faces)-1,max_faces,dtype=int)]
    tri=np.asarray(mesh.vertices)[faces]
    rgba=np.asarray(material_color(material))/255.0
    coll=Poly3DCollection(tri,facecolors=[rgba],edgecolors="none",linewidths=0.0,alpha=float(rgba[3]))
    ax.add_collection3d(coll)


def configure_axis(ax,title: str,elev: float=22,azim: float=-58) -> None:
    ax.set_title(title,fontsize=11,pad=8)
    ax.set_xlim(-0.10,0.10);ax.set_ylim(-0.10,0.10);ax.set_zlim(0.0,0.22)
    ax.view_init(elev=elev,azim=azim);ax.set_axis_off();ax.set_box_aspect((1,1,1.1))


def make_preview(bundle: ToolBundle) -> Path:
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(16,10),dpi=160)
    phases=[("inspect","1  Inspect + localize"),("clear","2  Clear field"),("compress","3  Distributed compression"),("clip","4  Form + deploy clip"),("patch","5  Apply patch"),("verify","6  Verify seal")]
    for i,(phase,title) in enumerate(phases,1):
        ax=fig.add_subplot(2,3,i,projection="3d")
        entries=world_visual_entries(bundle,phase)
        for _,m,mat in entries:add_mesh_to_axis(ax,m,mat,900)
        # add vessel in the working region for procedure panels
        for _,m,mat in vessel_entries(bundle,sealed=phase in {"patch","verify"}): add_mesh_to_axis(ax,transform(m,(0,0,WORK_PLANE_Z)),mat,900)
        configure_axis(ax,title)
    fig.suptitle("DrAnmar Adaptive Hemostasis Robot — clear, compress, clip, reinforce, verify",fontsize=17,y=0.98)
    fig.text(0.5,0.015,"DrAnmar research asset • NVIDIA Isaac integration • provisional physical parameters",ha="center",fontsize=10)
    path=PREVIEW_ROOT/"dranmar_adaptive_hemostasis_robot_preview.png";fig.savefig(path,bbox_inches="tight",facecolor="white");plt.close(fig);return path


def make_full_arm_preview(bundle: ToolBundle) -> Path:
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(10,10),dpi=170);ax=fig.add_subplot(111,projection="3d")
    for _,m,mat in franka_proxy_entries(bundle,"inspect"):add_mesh_to_axis(ax,m,mat,1100)
    ax.set_xlim(-0.18,0.38);ax.set_ylim(-0.25,0.25);ax.set_zlim(0,1.48);ax.view_init(elev=20,azim=-58);ax.set_axis_off();ax.set_box_aspect((0.6,0.55,1.45))
    ax.set_title("DrAnmar Adaptive Hemostasis Robot mounted at the Franka wrist",fontsize=15,pad=12)
    path=PREVIEW_ROOT/"dranmar_adaptive_hemostasis_robot_full_arm_preview.png";fig.savefig(path,bbox_inches="tight",facecolor="white");plt.close(fig);return path


def noise_texture(base: tuple[int,int,int], size: int=512, strength: int=18, seed: int=1) -> Image.Image:
    rng=np.random.default_rng(seed);arr=np.zeros((size,size,3),dtype=np.int16);arr[:]=np.asarray(base,dtype=np.int16);arr+=rng.normal(0,strength,(size,size,1)).astype(np.int16);arr=np.clip(arr,0,255).astype(np.uint8);return Image.fromarray(arr,"RGB")


def generate_textures() -> list[Path]:
    TEXTURE_ROOT.mkdir(parents=True,exist_ok=True);out=[]
    for name,base,strength,seed in [
        ("vessel_basecolor.png",(130,24,22),13,21),("blood_basecolor.png",(92,4,8),12,22),("hemostatic_patch_basecolor.png",(224,202,135),18,23),
        ("polymer_microtexture.png",(212,218,224),9,24),("metal_microtexture.png",(145,150,156),7,25)]:
        p=TEXTURE_ROOT/name;noise_texture(base,512,strength,seed).save(p);out.append(p)
    img=Image.new("RGB",(1024,256),(247,249,251));d=ImageDraw.Draw(img)
    try: font=ImageFont.truetype("DejaVuSans-Bold.ttf",72);small=ImageFont.truetype("DejaVuSans.ttf",30)
    except OSError: font=None;small=None
    d.text((36,48),"DrAnmar",fill=(18,65,112),font=font);d.text((40,150),"ADAPTIVE HEMOSTASIS • RESEARCH ONLY",fill=(35,45,55),font=small)
    p=TEXTURE_ROOT/"label_dranmar.png";img.save(p);out.append(p)
    return out


def interaction_frames(bundle: ToolBundle) -> dict[str,object]:
    return {"schema":"dranmar.interaction-frames.v1","asset":"dranmar-adaptive-hemostasis-robot-v1","units":"m","frames":bundle.frames}


def mount_contract() -> dict[str,object]:
    return {"schema":"dranmar.franka-mount.v1","parent_link":"resolved_from_stock_panda_hand_joint_body0_with_unique_panda_link8_fallback","payload_link":"DrAnmarAdaptiveHemostasisTool/Links/Mount","local_transform":"copied_from_stock_panda_hand_joint_frame_with_minus_45_degree_z_fallback","fallback_local_translation_m":[0,0,0],"fallback_local_rotation_axis_angle_deg":{"axis":[0,0,1],"angle":FRANKA_HAND_EQUIVALENT_ROTATION_DEG},"deactivate":["panda_hand_joint","panda_hand","panda_finger_joint1","panda_finger_joint2","panda_leftfinger","panda_rightfinger"],"research_only":True}


def task_contract() -> dict[str,object]:
    return {"schema":"dranmar.adaptive-hemostasis-task.v1","phases":["inspect","clear","compress","temporary_control_check","clip","release_compression","patch","pressure_challenge","verify","complete","abort"],"success_metrics":["residual_flow_ml_min","time_to_temporary_control_s","time_to_durable_control_s","clip_retained","patch_bond_fraction","compression_force_n","collected_blood_ml","spilled_blood_ml","pressure_challenge_pass","vessel_damage_proxy"],"failure_modes":["source_not_localized","insufficient_temporary_compression","clip_misalignment","clip_slip","incomplete_occlusion","patch_delamination","continued_bleeding","excess_compression","vessel_rupture_proxy","suction_overcapture"],"clinical_validation":False}


def physics_profile(bundle: ToolBundle) -> dict[str,object]:
    return {
        "schema":"dranmar.adaptive-hemostasis-profile.v1","id":"dranmar-adaptive-hemostasis-robot-v1","version":VERSION,"status":"research_only_runtime_qualified_physical_calibration_pending",
        "tool":{"mount":"panda_link8","joint_count":len(bundle.joints),"clip_capacity":CLIP_CAPACITY,"patch_capacity":PATCH_CAPACITY,"suction_ports":SUCTION_PORT_COUNT,"irrigation_ports":IRRIGATION_PORT_COUNT},
        "compression":{"target_force_per_pad_n":1.8,"soft_limit_n":4.0,"hard_release_n":7.0,"maximum_travel_m":0.030},
        "clip":{"open_gap_m":0.0060,"formed_gap_m":0.0011,"reference_closing_force_n":5.0,"provisional_retention_force_n":2.8,"plastic_forming_not_simulated":True},
        "patch":{"size_m":[0.026,0.020],"bond_cell_count":PATCH_BOND_CELL_COUNT,"cure_time_s":30.0,"initial_tack_break_force_n":0.8,"cured_break_force_n":8.0},
        "blood":{"density_kg_m3":1060.0,"reference_pressure_pa":10665.8,"dynamic_viscosity_pa_s":0.0035,"defect_area_m2":1.8e-6,"particle_nominal_volume_ml":0.002},
        "verification":{"challenge_pressure_pa":26664.5,"maximum_residual_flow_ml_min":0.1,"observation_window_s":5.0},
        "vessel":{"length_m":0.070,"outer_diameter_m":0.0064,"wall_thickness_m":0.00062,"representation":"portable_watertight_surface_for_runtime_deformable_cooking"},
        "boundaries":["no clinical hemostasis claim","no validated clip plasticity","no calibrated vessel damage","no qualified blood CFD","no patient-care settings"]
    }


def collider_coverage(bundle: ToolBundle) -> dict[str,object]:
    rows=[]
    for name,link in bundle.links.items():
        vmin,vmax=mesh_bounds([v.mesh for v in link.visuals]);cmeshes=[collider_mesh(c) for c in link.colliders]
        cmin,cmax=mesh_bounds(cmeshes) if cmeshes else (vmin,vmax)
        rows.append({"link":name,"visual_bounds_m":[vmin.tolist(),vmax.tolist()],"collider_bounds_m":[cmin.tolist(),cmax.tolist()],"axis_coverage_ratio":((cmax-cmin)/np.maximum(vmax-vmin,1e-8)).tolist(),"deliberate_insets":"contact and port openings remain unobstructed"})
    return {"schema":"dranmar.collider-coverage.v1","asset":"dranmar-adaptive-hemostasis-robot-v1","links":rows}


def author_integration_module() -> str:
    # Keep the audited runtime module as the source of truth. This avoids
    # maintaining a second embedded copy that can silently drift.
    if INTEGRATION_PATH.exists():
        return INTEGRATION_PATH.read_text(encoding="utf-8")
    return r'''# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab integration for the DrAnmar Adaptive Hemostasis Robot.

The payload replaces the Panda hand at ``panda_link8``. Runtime helpers provide
conserved blood-volume bookkeeping, PhysX particle emission and suction,
temporary vessel compression, physical clip retention, staged patch bonding,
and a reduced-order pressure/flow verification model. All parameters are
provisional research values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import math

CATALOG_SUBPATH = "Props/SurgicalHemostasis/AdaptiveHemostasisRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ROOT / "dranmar_adaptive_hemostasis_tool_payload.usda"
TOOL_STANDALONE_USD = ROOT / "dranmar_adaptive_hemostasis_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_adaptive_hemostasis_tool_rigid_proxy.usda"
CLIP_USD = ROOT / "dranmar_hemostatic_clip.usda"
PATCH_USD = ROOT / "dranmar_hemostatic_patch.usda"
PATCH_PROXY_USD = ROOT / "dranmar_hemostatic_patch_rigid_proxy.usda"
VESSEL_USD = ROOT / "dranmar_bleeding_vessel_demo.usda"
DROPLET_USD = ROOT / "dranmar_blood_droplet.usda"

VALID_BINARY_STATES = frozenset({"loaded", "empty"})
VALID_IRRIGATION_STATES = frozenset({"full", "empty"})
VALID_COLLECTION_STATES = frozenset({"empty", "partial", "full"})
TOOL_JOINTS = {
    "left_compression":"left_compression_joint","right_compression":"right_compression_joint",
    "left_pad_compliance":"left_pad_compliance_joint","right_pad_compliance":"right_pad_compliance_joint",
    "left_clip_jaw":"left_clip_jaw_joint","right_clip_jaw":"right_clip_jaw_joint",
    "clip_driver":"clip_driver_joint","patch_carousel":"patch_carousel_joint",
    "patch_applicator":"patch_applicator_joint","suction_valve":"suction_valve_joint","irrigation_valve":"irrigation_valve_joint",
}
TOOL_FRAME_PATHS = {
    "panda_link8_mount":"Links/Mount/Frames/panda_link8_mount","hemostasis_tcp":"Links/Mount/Frames/hemostasis_tcp",
    "bleeding_source_reference":"Links/Mount/Frames/bleeding_source_reference","suction_center":"Links/Mount/Frames/suction_center",
    "irrigation_center":"Links/Mount/Frames/irrigation_center","clip_forming_center":"Links/Mount/Frames/clip_forming_center",
    "clip_exit":"Links/ClipDriver/Frames/clip_exit","patch_application":"Links/PatchPlaten/Frames/patch_application",
    "left_compression_contact":"Links/LeftPad/Frames/left_compression_contact","right_compression_contact":"Links/RightPad/Frames/right_compression_contact",
    "left_clip_contact":"Links/LeftClipJaw/Frames/left_clip_contact","right_clip_contact":"Links/RightClipJaw/Frames/right_clip_contact",
    "flow_probe":"Links/Mount/Frames/flow_probe","count_reference":"Links/Mount/Frames/count_reference","disposal_reference":"Links/Mount/Frames/disposal_reference",
}


def frame_path(tool_path: str, name: str) -> str:
    try: suffix=TOOL_FRAME_PATHS[name]
    except KeyError as exc: raise KeyError(f"Unknown hemostasis frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any):
    return value.torch if hasattr(value,"torch") else value


def _check(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed: raise ValueError(f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}")
    return value


def make_tool_cfg(prim_path: str="/World/DrAnmarAdaptiveHemostasisTool", *, clip_state="loaded", patch_state="loaded", irrigation_state="full", collection_state="empty", position=(0,0,0.35), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    _check(clip_state,VALID_BINARY_STATES,"clip_state");_check(patch_state,VALID_BINARY_STATES,"patch_state");_check(irrigation_state,VALID_IRRIGATION_STATES,"irrigation_state");_check(collection_state,VALID_COLLECTION_STATES,"collection_state")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(TOOL_STANDALONE_USD),variants={"clip_state":clip_state,"patch_state":patch_state,"irrigation_state":irrigation_state,"collection_state":collection_state},activate_contact_sensors=True,articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=20,solver_velocity_iteration_count=6)),
        init_state=ArticulationCfg.InitialStateCfg(pos=position,rot=orientation_wxyz,joint_pos={name:0.0 for name in TOOL_JOINTS.values()}),
        actuators={
            "compression":ImplicitActuatorCfg(joint_names_expr=[".*compression_joint",".*pad_compliance_joint"],effort_limit_sim=120.0,velocity_limit_sim=0.18,stiffness=5200.0,damping=190.0),
            "clip":ImplicitActuatorCfg(joint_names_expr=[".*clip_jaw_joint","clip_driver_joint"],effort_limit_sim=240.0,velocity_limit_sim=0.30,stiffness=14000.0,damping=300.0),
            "patch":ImplicitActuatorCfg(joint_names_expr=["patch_.*_joint"],effort_limit_sim=90.0,velocity_limit_sim=1.5,stiffness=6500.0,damping=180.0),
            "valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=30.0,velocity_limit_sim=0.25,stiffness=1800.0,damping=55.0),
        },
    )


def make_rigid_proxy_cfg(prim_path="/World/DrAnmarAdaptiveHemostasisProxy", *, position=(0,0,0.35), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(prim_path=prim_path,spawn=sim_utils.UsdFileCfg(usd_path=str(TOOL_RIGID_PROXY_USD),activate_contact_sensors=True),init_state=RigidObjectCfg.InitialStateCfg(pos=position,rot=orientation_wxyz))


def _spawn_single_franka_with_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import create_prim, get_current_stage, select_usd_variants
    from pxr import Gf, Sdf, UsdPhysics
    robot=spawn_from_usd(prim_path,cfg,translation,orientation);stage=get_current_stage()
    for prim in list(stage.Traverse()):
        if prim.GetPath().HasPrefix(Sdf.Path(prim_path)) and prim.GetName() in {"panda_hand_joint","panda_hand","panda_finger_joint1","panda_finger_joint2","panda_leftfinger","panda_rightfinger"}:
            stage.OverridePrim(prim.GetPath()).SetActive(False)
    tool_path=f"{prim_path}/DrAnmarAdaptiveHemostasisTool";create_prim(tool_path,usd_path=str(TOOL_PAYLOAD_USD),stage=stage)
    select_usd_variants(tool_path,{"clip_state":cfg.clip_state,"patch_state":cfg.patch_state,"irrigation_state":cfg.irrigation_state,"collection_state":cfg.collection_state})
    joint=UsdPhysics.FixedJoint.Define(stage,f"{prim_path}/dranmar_hemostasis_mount_joint");joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{prim_path}/panda_link8")]);joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")]);joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0,0,0));joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0));a=math.radians(-45.0)/2;joint.CreateLocalRot0Attr().Set(Gf.Quatf(math.cos(a),0,0,math.sin(a)));joint.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0));return robot


def spawn_franka_with_tool(prim_path: str,cfg: Any,translation=None,orientation=None,**kwargs):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_tool)(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)


def make_franka_adaptive_hemostasis_robot_cfg(*, prim_path="/World/Robot", clip_state="loaded", patch_state="loaded", irrigation_state="full", collection_state="empty"):
    _check(clip_state,VALID_BINARY_STATES,"clip_state");_check(patch_state,VALID_BINARY_STATES,"patch_state");_check(irrigation_state,VALID_IRRIGATION_STATES,"irrigation_state");_check(collection_state,VALID_COLLECTION_STATES,"collection_state")
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    @configclass
    class FrankaHemostasisUsdCfg(sim_utils.UsdFileCfg):
        clip_state: str="loaded";patch_state: str="loaded";irrigation_state: str="full";collection_state: str="empty";func=spawn_franka_with_tool
    cfg=FRANKA_PANDA_CFG.copy();cfg.prim_path=prim_path
    cfg.spawn=FrankaHemostasisUsdCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",variants={"Gripper":"Default","Mesh":"Performance"},clip_state=clip_state,patch_state=patch_state,irrigation_state=irrigation_state,collection_state=collection_state,activate_contact_sensors=True,rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props)
    cfg.init_state.joint_pos={k:v for k,v in cfg.init_state.joint_pos.items() if "finger" not in k};cfg.init_state.joint_pos.update({name:0.0 for name in TOOL_JOINTS.values()})
    cfg.actuators={k:v for k,v in cfg.actuators.items() if k!="panda_hand"}
    cfg.actuators.update({
        "hemostasis_compression":ImplicitActuatorCfg(joint_names_expr=[".*compression_joint",".*pad_compliance_joint"],effort_limit_sim=120.0,velocity_limit_sim=0.18,stiffness=5200.0,damping=190.0),
        "hemostasis_clip":ImplicitActuatorCfg(joint_names_expr=[".*clip_jaw_joint","clip_driver_joint"],effort_limit_sim=240.0,velocity_limit_sim=0.30,stiffness=14000.0,damping=300.0),
        "hemostasis_patch":ImplicitActuatorCfg(joint_names_expr=["patch_.*_joint"],effort_limit_sim=90.0,velocity_limit_sim=1.5,stiffness=6500.0,damping=180.0),
        "hemostasis_valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=30.0,velocity_limit_sim=0.25,stiffness=1800.0,damping=55.0),
    })
    return cfg


def _current_stage(stage=None):
    if stage is not None:return stage
    import omni.usd
    return omni.usd.get_context().get_stage()


def spawn_vessel_demo(prim_path="/World/DrAnmarBleedingVessel", *, translation=(0,0,0), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    cfg=sim_utils.UsdFileCfg(usd_path=str(VESSEL_USD));return cfg.func(prim_path,cfg,translation=translation,orientation=orientation_wxyz)


def apply_vessel_surface_deformable(root_path: str, *, self_collision=True, stage=None):
    stage=_current_stage(stage);mesh_path=f"{root_path.rstrip('/')}/VesselWall";mesh=stage.GetPrimAtPath(mesh_path)
    if not mesh or not mesh.IsValid():raise ValueError(f"No vessel wall at {mesh_path}")
    from omni.physx.scripts import deformableUtils
    ok=deformableUtils.set_physics_surface_deformable_body(stage,mesh.GetPath())
    if ok is False:raise RuntimeError(f"Failed to cook vessel surface deformable at {mesh_path}")
    mesh.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    if mesh.HasAPI("PhysxSurfaceDeformableBodyAPI"):mesh.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
    return {"root_path":root_path,"mesh_path":mesh_path,"self_collision":bool(self_collision)}


def create_deformable_attachment(deformable_path: str, target_path: str, attachment_path: str, *, stage=None):
    stage=_current_stage(stage)
    import omni.kit.commands
    try:
        result=omni.kit.commands.execute("CreateAutoDeformableAttachment",target_attachment_path=attachment_path,attachable0_path=deformable_path,attachable1_path=target_path)
        if result is not False:return attachment_path
    except Exception:
        pass
    result=omni.kit.commands.execute("CreatePhysicsAttachment",target_attachment_path=attachment_path,actor0_path=deformable_path,actor1_path=target_path)
    if result is False:raise RuntimeError(f"Failed to attach {deformable_path} to {target_path}")
    return attachment_path


def remove_prims(paths: Iterable[str], *, stage=None):
    stage=_current_stage(stage)
    for path in paths:
        if stage.GetPrimAtPath(path).IsValid():stage.RemovePrim(path)


@dataclass
class HemorrhageLedger:
    initial_reservoir_ml: float=250.0
    reservoir_ml: float=250.0
    emitted_ml: float=0.0
    active_particle_ml: float=0.0
    suctioned_ml: float=0.0
    spilled_ml: float=0.0
    discarded_ml: float=0.0
    def emit(self,volume_ml: float):
        amount=min(max(0.0,float(volume_ml)),self.reservoir_ml);self.reservoir_ml-=amount;self.emitted_ml+=amount;self.active_particle_ml+=amount;return amount
    def suction(self,volume_ml: float):
        amount=min(max(0.0,float(volume_ml)),self.active_particle_ml);self.active_particle_ml-=amount;self.suctioned_ml+=amount;return amount
    def spill(self,volume_ml: float):
        amount=min(max(0.0,float(volume_ml)),self.active_particle_ml);self.active_particle_ml-=amount;self.spilled_ml+=amount;return amount
    @property
    def conservation_error_ml(self):return self.initial_reservoir_ml-(self.reservoir_ml+self.active_particle_ml+self.suctioned_ml+self.spilled_ml+self.discarded_ml)


@dataclass
class ReducedOrderBleedModel:
    defect_area_m2: float=1.8e-6
    pressure_pa: float=10665.8
    density_kg_m3: float=1060.0
    discharge_coefficient: float=0.62
    compression_fraction: float=0.0
    clip_occlusion_fraction: float=0.0
    patch_seal_fraction: float=0.0
    vessel_damage_multiplier: float=1.0
    def effective_area_m2(self):
        closure=1.0-(1.0-max(0.0,min(1.0,self.compression_fraction)))*(1.0-max(0.0,min(1.0,self.clip_occlusion_fraction)))*(1.0-max(0.0,min(1.0,self.patch_seal_fraction)))
        return max(0.0,self.defect_area_m2*(1.0-closure)*max(0.0,self.vessel_damage_multiplier))
    def flow_m3_s(self, downstream_pressure_pa: float=0.0):
        dp=max(0.0,self.pressure_pa-float(downstream_pressure_pa));return self.discharge_coefficient*self.effective_area_m2()*math.sqrt(2.0*dp/max(self.density_kg_m3,1e-9))
    def flow_ml_min(self,downstream_pressure_pa: float=0.0):return self.flow_m3_s(downstream_pressure_pa)*60.0*1e6


def ensure_blood_particle_system(*, system_path="/World/DrAnmarBlood/ParticleSystem", material_path="/World/DrAnmarBlood/PBDMaterial", stage=None):
    stage=_current_stage(stage)
    from pxr import Sdf, UsdPhysics
    from omni.physx.scripts import particleUtils, physicsUtils
    scene_path=Sdf.Path("/World/PhysicsScene")
    if not stage.GetPrimAtPath(scene_path).IsValid():UsdPhysics.Scene.Define(stage,scene_path)
    if not stage.GetPrimAtPath(material_path).IsValid():particleUtils.add_pbd_particle_material(stage,Sdf.Path(material_path),friction=0.08,viscosity=0.0035,cohesion=0.01,surface_tension=0.02)
    if not stage.GetPrimAtPath(system_path).IsValid():particleUtils.add_physx_particle_system(stage=stage,particle_system_path=Sdf.Path(system_path),simulation_owner=scene_path,particle_contact_offset=0.0012,solid_rest_offset=0.00065,fluid_rest_offset=0.00055)
    physicsUtils.add_physics_material_to_prim(stage,stage.GetPrimAtPath(system_path),Sdf.Path(material_path))
    return {"particle_system_path":system_path,"material_path":material_path}


def emit_blood_burst(positions: Sequence[Sequence[float]], velocities: Sequence[Sequence[float]], *, particle_volume_ml=0.002, system_path="/World/DrAnmarBlood/ParticleSystem", particles_path="/World/DrAnmarBlood/Particles", ledger: HemorrhageLedger|None=None, stage=None):
    stage=_current_stage(stage);ensure_blood_particle_system(system_path=system_path,stage=stage)
    from pxr import Gf, Sdf
    from omni.physx.scripts import particleUtils
    positions=[Gf.Vec3f(*map(float,p)) for p in positions];velocities=[Gf.Vec3f(*map(float,v)) for v in velocities];widths=[0.0013]*len(positions)
    if ledger is not None:
        allowed=int(ledger.emit(len(positions)*particle_volume_ml)/max(particle_volume_ml,1e-12));positions=positions[:allowed];velocities=velocities[:allowed];widths=widths[:allowed]
    if not positions:return None
    path=Sdf.Path(particles_path);points=particleUtils.add_physx_particleset_points(stage,path,positions,velocities,widths,Sdf.Path(system_path),True,True,0,1.0,0.02)
    return points


@dataclass
class AnnularSuctionController:
    center_world: tuple[float,float,float]
    capture_radius_m: float=0.010
    attraction_radius_m: float=0.050
    acceleration_m_s2: float=14.0
    particle_volume_ml: float=0.002
    def update_positions_velocities(self,positions,velocities,dt: float,ledger: HemorrhageLedger|None=None):
        import numpy as np
        pos=np.asarray(tensor_value(positions),dtype=float);vel=np.asarray(tensor_value(velocities),dtype=float);center=np.asarray(self.center_world,dtype=float);delta=center-pos;dist=np.linalg.norm(delta,axis=-1);direction=delta/np.maximum(dist[...,None],1e-9);mask=dist<self.attraction_radius_m;vel[mask]+=direction[mask]*self.acceleration_m_s2*max(0.0,float(dt));captured=dist<self.capture_radius_m
        count=int(np.count_nonzero(captured))
        if ledger is not None and count:ledger.suction(count*self.particle_volume_ml)
        return pos[~captured],vel[~captured],captured


@dataclass
class TemporaryCompressionController:
    tool_path: str
    vessel_path: str
    attachment_paths: list[str]=field(default_factory=list)
    engaged: bool=False
    def engage(self,*,stage=None):
        if self.engaged:return list(self.attachment_paths)
        stage=_current_stage(stage);pairs=[("LeftPad","left"),("RightPad","right")]
        created=[]
        try:
            for link,label in pairs:
                path=f"{self.tool_path}/RuntimeAttachments/compression_{label}"
                stage.DefinePrim(f"{self.tool_path}/RuntimeAttachments","Scope")
                create_deformable_attachment(self.vessel_path,f"{self.tool_path}/Links/{link}/Collisions/VesselCaptureVolume",path,stage=stage);created.append(path)
        except Exception:
            remove_prims(created,stage=stage);raise
        self.attachment_paths=created;self.engaged=True;return list(created)
    def release(self,*,stage=None):remove_prims(self.attachment_paths,stage=stage);self.attachment_paths.clear();self.engaged=False


def _spawn_reference_at_transform(stage,prim_path: str,usd_path: Path,world_transform: Any,variants: dict[str,str]|None=None):
    import omni.usd
    from pxr import Sdf
    prim=stage.DefinePrim(prim_path,"Xform");prim.GetReferences().AddReference(str(usd_path));omni.kit.commands.execute("TransformPrim",path=Sdf.Path(prim_path),new_transform_matrix=world_transform)
    if variants:
        for name,value in variants.items():prim.GetVariantSets().GetVariantSet(name).SetVariantSelection(value)
    return prim


def deploy_formed_clip(prim_path: str, world_transform: Any, vessel_path: str, *, stage=None):
    stage=_current_stage(stage);_spawn_reference_at_transform(stage,prim_path,CLIP_USD,world_transform,{"state":"formed"});stage.DefinePrim(f"{prim_path}/Attachments","Scope");created=[]
    try:
        for name in ("LeftLegAttachment","RightLegAttachment"):
            ap=f"{prim_path}/Attachments/{name}";create_deformable_attachment(vessel_path,f"{prim_path}/Collisions/{name}",ap,stage=stage);created.append(ap)
    except Exception:
        remove_prims(created+[prim_path],stage=stage);raise
    return {"clip_path":prim_path,"attachment_paths":created,"state":"formed"}


@dataclass
class ClipRetentionBond:
    clip_path: str
    attachment_paths: list[str]
    retained: bool=True


@dataclass
class ClipRetentionController:
    pullout_force_n: float=2.8
    bonds: list[ClipRetentionBond]=field(default_factory=list)
    def register(self,deployment):
        b=ClipRetentionBond(str(deployment["clip_path"]),list(deployment["attachment_paths"]));self.bonds.append(b);return b
    def apply_load(self,bond: ClipRetentionBond,load_n: float,*,stage=None):
        if not bond.retained or abs(float(load_n))<=self.pullout_force_n:return False
        remove_prims(bond.attachment_paths,stage=stage);bond.retained=False;return True


@dataclass
class PatchBond:
    patch_path: str
    attachment_paths: list[str]
    cure_fraction: float=0.0
    broken: bool=False


@dataclass
class HemostaticPatchBondController:
    cure_time_s: float=30.0
    final_break_force_n: float=8.0
    bonds: list[PatchBond]=field(default_factory=list)
    def deploy(self,prim_path: str,world_transform: Any,vessel_path: str,*,stage=None):
        stage=_current_stage(stage);_spawn_reference_at_transform(stage,prim_path,PATCH_PROXY_USD,world_transform);stage.DefinePrim(f"{prim_path}/Attachments","Scope");created=[]
        try:
            for i in range(8):
                ap=f"{prim_path}/Attachments/bond_{i:02d}";create_deformable_attachment(vessel_path,f"{prim_path}/Collisions/BondCell_{i:02d}",ap,stage=stage);created.append(ap)
        except Exception:
            remove_prims(created+[prim_path],stage=stage);raise
        b=PatchBond(prim_path,created);self.bonds.append(b);return b
    def update(self,dt: float):
        for b in self.bonds:
            if not b.broken:b.cure_fraction=min(1.0,b.cure_fraction+max(0.0,float(dt))/max(self.cure_time_s,1e-9))
    def apply_load(self,bond: PatchBond,load_n: float,*,stage=None):
        threshold=max(0.2,self.final_break_force_n*(0.08+0.92*bond.cure_fraction))
        if bond.broken or abs(float(load_n))<=threshold:return False
        remove_prims(bond.attachment_paths,stage=stage);bond.broken=True;return True


@dataclass
class SealVerificationController:
    maximum_flow_ml_min: float=0.1
    observation_window_s: float=5.0
    elapsed_s: float=0.0
    integrated_volume_ml: float=0.0
    peak_flow_ml_min: float=0.0
    def reset(self):self.elapsed_s=0.0;self.integrated_volume_ml=0.0;self.peak_flow_ml_min=0.0
    def update(self,flow_ml_min: float,dt: float):
        flow=max(0.0,float(flow_ml_min));dt=max(0.0,float(dt));self.elapsed_s+=dt;self.integrated_volume_ml+=flow*dt/60.0;self.peak_flow_ml_min=max(self.peak_flow_ml_min,flow)
    @property
    def average_flow_ml_min(self):return 0.0 if self.elapsed_s<=0 else self.integrated_volume_ml*60.0/self.elapsed_s
    @property
    def complete(self):return self.elapsed_s>=self.observation_window_s
    @property
    def passed(self):return self.complete and self.average_flow_ml_min<=self.maximum_flow_ml_min


PHASE_TARGETS={
    "inspect":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.0,"irrigation_valve_joint":0.0},
    "clear":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.008,"irrigation_valve_joint":0.005},
    "compress":{"left_compression_joint":0.026,"right_compression_joint":-0.026,"left_pad_compliance_joint":-0.005,"right_pad_compliance_joint":-0.005,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.006,"irrigation_valve_joint":0.0},
    "temporary_control_check":{"left_compression_joint":0.026,"right_compression_joint":-0.026,"left_pad_compliance_joint":-0.005,"right_pad_compliance_joint":-0.005,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.004,"irrigation_valve_joint":0.0},
    "clip":{"left_compression_joint":0.026,"right_compression_joint":-0.026,"left_pad_compliance_joint":-0.005,"right_pad_compliance_joint":-0.005,"left_clip_jaw_joint":0.007,"right_clip_jaw_joint":-0.007,"clip_driver_joint":0.017,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.004,"irrigation_valve_joint":0.0},
    "release_compression":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.003,"irrigation_valve_joint":0.0},
    "patch":{"left_compression_joint":0.012,"right_compression_joint":-0.012,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":math.radians(90.0),"patch_applicator_joint":0.034,"suction_valve_joint":0.002,"irrigation_valve_joint":0.0},
    "pressure_challenge":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":math.radians(90.0),"patch_applicator_joint":0.0,"suction_valve_joint":0.0,"irrigation_valve_joint":0.0},
    "verify":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":math.radians(90.0),"patch_applicator_joint":0.0,"suction_valve_joint":0.004,"irrigation_valve_joint":0.0},
    "complete":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.0,"irrigation_valve_joint":0.0},
    "abort":{"left_compression_joint":0.0,"right_compression_joint":0.0,"left_pad_compliance_joint":0.0,"right_pad_compliance_joint":0.0,"left_clip_jaw_joint":0.0,"right_clip_jaw_joint":0.0,"clip_driver_joint":0.0,"patch_carousel_joint":0.0,"patch_applicator_joint":0.0,"suction_valve_joint":0.008,"irrigation_valve_joint":0.0},
}

def phase_targets(phase: str):
    try:return dict(PHASE_TARGETS[phase])
    except KeyError as exc:raise KeyError(f"Unknown hemostasis phase {phase!r}") from exc


@dataclass
class AdaptiveHemostasisSequenceController:
    phase: str="inspect"
    bleed_model: ReducedOrderBleedModel=field(default_factory=ReducedOrderBleedModel)
    verifier: SealVerificationController=field(default_factory=SealVerificationController)
    history: list[str]=field(default_factory=list)
    def transition(self,phase: str):
        phase_targets(phase);self.phase=phase;self.history.append(phase)
        if phase=="verify":self.verifier.reset()
        return phase_targets(phase)
    def set_compression(self,fraction: float):self.bleed_model.compression_fraction=max(0.0,min(1.0,float(fraction)))
    def set_clip_occlusion(self,fraction: float):self.bleed_model.clip_occlusion_fraction=max(0.0,min(1.0,float(fraction)))
    def set_patch_seal(self,fraction: float):self.bleed_model.patch_seal_fraction=max(0.0,min(1.0,float(fraction)))
    def update_verification(self,dt: float):
        flow=self.bleed_model.flow_ml_min();self.verifier.update(flow,dt);return {"flow_ml_min":flow,"average_flow_ml_min":self.verifier.average_flow_ml_min,"complete":self.verifier.complete,"passed":self.verifier.passed}
'''


def readme() -> str:
    return f'''# {ASSET_NAME} v{VERSION}

DrAnmar-owned OpenUSD research system for robotic field clearing, temporary vascular compression, physical clip retention, hemostatic patch reinforcement, and post-seal leak verification. It integrates with NVIDIA Isaac Lab and Isaac Sim while remaining provider-independent at the asset-contract layer.

## Catalog path

`Props/SurgicalHemostasis/AdaptiveHemostasisRobot/`

## Main assets

- `dranmar_adaptive_hemostasis_tool_payload.usda` — Franka payload without a nested articulation root.
- `dranmar_adaptive_hemostasis_tool_standalone.usda` — standalone articulated tool.
- `dranmar_adaptive_hemostasis_tool_rigid_proxy.usda` — perception/planning proxy.
- `dranmar_hemostatic_clip.usda` — open/formed clip states and two physical vessel-attachment zones.
- `dranmar_hemostatic_patch.usda` — deformable-ready triangular patch surface.
- `dranmar_hemostatic_patch_rigid_proxy.usda` — stable eight-cell bond carrier.
- `dranmar_bleeding_vessel_demo.usda` — curved hollow vessel wall with reduced-order bleeding source.
- `dranmar_blood_droplet.usda` — particle prototype.

## Procedure

`inspect → clear → compress → temporary control check → clip → release compression → patch → pressure challenge → verify → complete or abort`

All geometry, mechanics, flow, retention, bond, pressure, and damage values are provisional research parameters. This asset is not clinically validated and is not approved for patient care.
'''


def docs_mechanism() -> str:
    return '''# Mechanism

The payload replaces the Panda hand at `panda_link8`. Eleven active joints control bilateral temporary compression, compliant pad contact, two clip-forming jaws, a clip driver, a patch carousel, a patch platen, and independent suction and irrigation valves. The clip and patch inventories use discrete USD variants; joint motion remains articulation-driven.
'''


def docs_physical_hemostasis() -> str:
    return '''# Physical hemostasis contract

The vessel endpoints are held by two explicit fixture-anchor meshes and current-schema vertex attachments. Temporary control uses two independent deformable-to-pad attachments. The pads move through articulated carriages and therefore transmit solver forces into the vessel rather than rewriting vessel transforms. The task-level force envelope targets 1.8 N per pad, flags a soft limit above 4.0 N, and releases both temporary bonds above the 7.0 N hard limit.

A deployed formed clip is an independent rigid body. Two separate attachment volumes on its legs bond to opposite vessel-wall regions. The temporary pad attachments can then be removed while the clip remains load-bearing. A provisional pullout controller removes both leg bonds only when supplied load exceeds its configured threshold.

The patch has eight independent bond cells. Initial placement creates physical vessel-to-patch attachments; cure progression raises the task-level break-force envelope from 0.8 N to 8.0 N over 30 seconds. The stable lane uses a rigid bond carrier, while a portable triangular patch surface is provided for runtime deformable cooking.
'''


def docs_flow() -> str:
    return '''# Blood-flow and leak model

The package separates particle transport from the reduced-order source model. The source computes an orifice-flow estimate from pressure, effective defect area, density, and a discharge coefficient. Temporary compression, clip occlusion, and patch sealing reduce the effective area multiplicatively.

PhysX PBD particles represent emitted blood in the field. `HemorrhageLedger` conserves reservoir, active, suctioned, spilled, and discarded volumes. `AnnularSuctionController` accelerates particles toward the authored suction center and transfers captured particle volume into the collection ledger.

The verification controller integrates residual flow over a pressure-challenge observation window. It is a research benchmark and not proof of clinical hemostasis.
'''


def docs_franka() -> str:
    return '''# Franka integration

The custom spawner references the composable Isaac Franka asset, snapshots the stock `panda_hand_joint` body target and local frame, safely deactivates the stock hand and finger prims, references the DrAnmar payload, and creates a fixed joint to `Links/Mount`. A uniquely resolved `panda_link8` with −45 degrees around local Z is retained only as a compatibility fallback.
'''


def example_scene() -> str:
    return '''#!/usr/bin/env python3
"""Minimal DrAnmar Adaptive Hemostasis Robot scene skeleton."""
from isaaclab.app import AppLauncher
app_launcher=AppLauncher(headless=False)
simulation_app=app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from orbit.surgical.assets.adaptive_hemostasis_robot import make_franka_adaptive_hemostasis_robot_cfg, spawn_vessel_demo

class SceneCfg(InteractiveSceneCfg):
    ground=AssetBaseCfg(prim_path="/World/Ground",spawn=sim_utils.GroundPlaneCfg())
    light=AssetBaseCfg(prim_path="/World/Light",spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot=make_franka_adaptive_hemostasis_robot_cfg(prim_path="{ENV_REGEX_NS}/Robot")

sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0",dt=1/120))
scene=InteractiveScene(SceneCfg(num_envs=1,env_spacing=2.0))
spawn_vessel_demo("/World/DrAnmarBleedingVessel",translation=(0.55,0.0,0.02))
sim.reset()
while simulation_app.is_running():
    scene.write_data_to_sim();sim.step();scene.update(sim.get_physics_dt())
simulation_app.close()
'''


def author_installer() -> str:
    installer = PACKAGE_ROOT / "scripts/install_into_dranmar.py"
    if installer.exists():
        return installer.read_text(encoding="utf-8")
    return '''#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, sys
from pathlib import Path

PACKAGE=Path(__file__).resolve().parents[1]

def copytree_contents(source: Path,destination: Path):
    destination.mkdir(parents=True,exist_ok=True)
    for child in source.iterdir():
        target=destination/child.name
        if child.is_dir():shutil.copytree(child,target,dirs_exist_ok=True)
        else:shutil.copy2(child,target)

def main():
    if len(sys.argv)!=2:raise SystemExit("usage: install_into_dranmar.py /path/to/drAnmar")
    repo=Path(sys.argv[1]).expanduser().resolve()
    copytree_contents(PACKAGE/"source/extensions/orbit.surgical.assets",repo/"source/extensions/orbit.surgical.assets")
    copytree_contents(PACKAGE/"physics_next",repo/"physics_next")
    copytree_contents(PACKAGE/"docs",repo/"docs")
    copytree_contents(PACKAGE/"examples",repo/"examples")
    shutil.copy2(PACKAGE/"scripts/generate_dranmar_adaptive_hemostasis_robot.py",repo/"scripts/generate_dranmar_adaptive_hemostasis_robot.py")
    init=repo/"source/extensions/orbit.surgical.assets/orbit/surgical/assets/__init__.py"
    line="from .adaptive_hemostasis_robot import *\\n"
    text=init.read_text() if init.exists() else ""
    if line not in text:init.write_text(text+line)
    print(json.dumps({"installed":True,"repository":str(repo),"catalog_subpath":"Props/SurgicalHemostasis/AdaptiveHemostasisRobot"},indent=2))
if __name__=="__main__":main()
'''


def write_json(path: Path,payload: Any) -> Path:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");return path


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def write_asset_files(bundle: ToolBundle) -> list[Path]:
    ASSET_ROOT.mkdir(parents=True,exist_ok=True);GLB_ROOT.mkdir(parents=True,exist_ok=True);PREVIEW_ROOT.mkdir(parents=True,exist_ok=True);DOCS_ROOT.mkdir(parents=True,exist_ok=True);EXAMPLE_ROOT.mkdir(parents=True,exist_ok=True);INTEGRATION_PATH.parent.mkdir(parents=True,exist_ok=True);PHYSICS_PROFILE_PATH.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    mapping={
        "dranmar_adaptive_hemostasis_tool_payload.usda":tool_usda(bundle,False),
        "dranmar_adaptive_hemostasis_tool_standalone.usda":tool_usda(bundle,True),
        "dranmar_adaptive_hemostasis_tool_rigid_proxy.usda":rigid_proxy_usda(bundle),
        "dranmar_hemostatic_clip.usda":clip_usda(bundle),
        "dranmar_hemostatic_patch.usda":patch_surface_usda(bundle),
        "dranmar_hemostatic_patch_rigid_proxy.usda":patch_proxy_usda(bundle),
        "dranmar_bleeding_vessel_demo.usda":vessel_usda(bundle),
        "dranmar_blood_droplet.usda":droplet_usda(),
        "README.md":readme(),
    }
    for name,text in mapping.items():p=ASSET_ROOT/name;p.write_text(text,encoding="utf-8");files.append(p)
    lic=ASSET_ROOT/"LICENSE.txt";lic.write_text("Apache License 2.0\nCopyright 2026 DrAnmar Project Developers\n",encoding="utf-8");files.append(lic)
    files+=generate_textures();files+=export_glbs(bundle);files+=[make_preview(bundle),make_full_arm_preview(bundle)]
    files += [write_json(ASSET_ROOT/"interaction_frames.json",interaction_frames(bundle)),write_json(ASSET_ROOT/"franka_mount_contract.json",mount_contract()),write_json(ASSET_ROOT/"adaptive_hemostasis_task_contract.json",task_contract()),write_json(ASSET_ROOT/"physics_profile.json",physics_profile(bundle)),write_json(ASSET_ROOT/"collider_coverage.json",collider_coverage(bundle))]
    write_json(PHYSICS_PROFILE_PATH,physics_profile(bundle));files.append(PHYSICS_PROFILE_PATH)
    INTEGRATION_PATH.write_text(author_integration_module(),encoding="utf-8");files.append(INTEGRATION_PATH)
    for name,text in [("MECHANISM.md",docs_mechanism()),("PHYSICAL_HEMOSTASIS.md",docs_physical_hemostasis()),("FLOW_AND_LEAK_MODEL.md",docs_flow()),("FRANKA_INTEGRATION.md",docs_franka())]:p=DOCS_ROOT/name;p.write_text(text,encoding="utf-8");files.append(p)
    ex=EXAMPLE_ROOT/"franka_adaptive_hemostasis_scene.py";ex.write_text(example_scene(),encoding="utf-8");files.append(ex)
    installer=PACKAGE_ROOT/"scripts/install_into_dranmar.py";installer.write_text(author_installer(),encoding="utf-8");installer.chmod(0o755);files.append(installer)
    return files


def sync_extension_data() -> None:
    target=EXTENSION_ROOT/"data"/CATALOG_SUBPATH
    shutil.rmtree(target,ignore_errors=True);target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(ASSET_ROOT,target)


def all_payload_files() -> list[Path]:
    mirror_root = EXTENSION_ROOT / "data" / CATALOG_SUBPATH
    return sorted(
        p for p in PACKAGE_ROOT.rglob("*")
        if p.is_file()
        and not p.is_relative_to(mirror_root)
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and p.name != ".DS_Store"
        and p.name != "asset_manifest.json"
        and p.name != "static_build_report.json"
    )


def build_manifest(files: Sequence[Path]) -> dict[str,object]:
    return {"schema":"dranmar.asset-manifest.v1","asset":"dranmar-adaptive-hemostasis-robot-v1","version":VERSION,"files":[{"path":p.relative_to(PACKAGE_ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in files]}


def zip_tree(source: Path,output: Path,*,prefix: str|None=None) -> Path:
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(source.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != ".DS_Store":
                rel=p.relative_to(source);arc=Path(prefix)/rel if prefix else rel
                info=zipfile.ZipInfo(arc.as_posix(),date_time=(2026,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=(0o755 if p.stat().st_mode & 0o111 else 0o644)<<16
                z.writestr(info,p.read_bytes())
    return output


def write_checksum(path: Path) -> Path:
    out=path.with_suffix(path.suffix+".sha256");out.write_text(f"{sha256(path)}  {path.name}\n");return out


def build_overlay() -> Path:
    temp=PACKAGE_ROOT/"_overlay";shutil.rmtree(temp,ignore_errors=True)
    for sub in ("source","physics_next","docs","examples","tests"):
        src=PACKAGE_ROOT/sub
        if src.exists():shutil.copytree(src,temp/sub,dirs_exist_ok=True)
    (temp/"scripts").mkdir(parents=True,exist_ok=True)
    for name in (
        SCRIPT_PATH.name,
        "install_into_dranmar.py",
        "validate_dranmar_adaptive_hemostasis_robot.py",
        "requirements_adaptive_hemostasis_generation.txt",
    ):
        source=PACKAGE_ROOT/"scripts"/name
        if source.exists():shutil.copy2(source,temp/"scripts"/name)
    output=PACKAGE_ROOT.parent/"dranmar_adaptive_hemostasis_robot_repo_overlay_v0.1.0.zip";zip_tree(temp,output);shutil.rmtree(temp);return output


def static_report(files: Sequence[Path]) -> dict[str,object]:
    usda=[p for p in files if p.suffix==".usda"]
    checks=[]
    for p in usda:
        text=p.read_text(encoding="utf-8");checks.append({"file":p.relative_to(PACKAGE_ROOT).as_posix(),"brace_balance":text.count("{")==text.count("}"),"flat_quaternion_count":text.count("quatf "),"nested_quaternion_suspect":"(1, (" in text,"one_line_over_suspect":any(line.strip().startswith("over ") and "{" in line and "}" in line for line in text.splitlines())})
    return {"schema":"dranmar.static-build-report.v1","asset":"dranmar-adaptive-hemostasis-robot-v1","usda_checks":checks,"runtime_validation":"headless_cuda_qualification_supplied_separately"}


def generate() -> dict[str,object]:
    for cache in PACKAGE_ROOT.rglob("__pycache__"):
        shutil.rmtree(cache)
    for bytecode in PACKAGE_ROOT.rglob("*.pyc"):
        bytecode.unlink()
    for old_manifest in (ASSET_ROOT/"asset_manifest.json", EXTENSION_ROOT/"data"/CATALOG_SUBPATH/"asset_manifest.json"):
        if old_manifest.exists():old_manifest.unlink()
    bundle=build_tool();files=write_asset_files(bundle)
    manifest=write_json(ASSET_ROOT/"asset_manifest.json",build_manifest(all_payload_files()));files.append(manifest)
    sync_extension_data()
    report=write_json(PACKAGE_ROOT/"static_build_report.json",static_report(files));files.append(report)
    # Compile in memory so release generation never packages bytecode caches.
    for p in all_payload_files():
        if p.suffix==".py":compile(p.read_text(encoding="utf-8"),str(p),"exec")
    dev=PACKAGE_ROOT.parent/"dranmar_adaptive_hemostasis_robot_v0.1.0.zip";zip_tree(PACKAGE_ROOT,dev)
    catalog=PACKAGE_ROOT.parent/"dranmar_adaptive_hemostasis_robot_catalog_v0.1.0.zip";zip_tree(ASSET_ROOT,catalog,prefix=CATALOG_SUBPATH.as_posix())
    overlay=build_overlay()
    for p in (dev,catalog,overlay):write_checksum(p)
    release={"schema":"dranmar.release.v1","asset":"dranmar-adaptive-hemostasis-robot-v1","version":VERSION,"catalog_subpath":CATALOG_SUBPATH.as_posix(),"development_package":{"path":str(dev),"sha256":sha256(dev)},"catalog_package":{"path":str(catalog),"sha256":sha256(catalog)},"repository_overlay":{"path":str(overlay),"sha256":sha256(overlay)},"primary_assets":[str(ASSET_ROOT/n) for n in ("dranmar_adaptive_hemostasis_tool_payload.usda","dranmar_adaptive_hemostasis_tool_standalone.usda","dranmar_adaptive_hemostasis_tool_rigid_proxy.usda","dranmar_hemostatic_clip.usda","dranmar_hemostatic_patch.usda","dranmar_hemostatic_patch_rigid_proxy.usda","dranmar_bleeding_vessel_demo.usda","dranmar_blood_droplet.usda")],"runtime_validation":"qualified_headless_cuda_matrix_report_in_catalog","clinical_validation":False}
    release_path=PACKAGE_ROOT.parent/"dranmar_adaptive_hemostasis_robot_release_v0.1.0.json";write_json(release_path,release)
    return release


def main() -> None:
    print(json.dumps(generate(),indent=2))


if __name__=="__main__":
    main()
