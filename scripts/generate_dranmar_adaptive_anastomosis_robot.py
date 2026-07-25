#!/usr/bin/env python3
"""Generate the DrAnmar Adaptive Anastomosis Robot asset family.

The asset is a DrAnmar-owned, provider-neutral research system for NVIDIA
Isaac Sim / Isaac Lab. It represents bilateral atraumatic tissue capture,
coaxial lumen alignment, edge eversion, circumferential staple deployment,
reinforcement-collar bonding, patency measurement, and pressure-decay leak
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
ASSET_NAME = "DrAnmar Adaptive Anastomosis Robot"
CATALOG_SUBPATH = Path("Props/SurgicalReconstruction/AdaptiveAnastomosisRobot")
ROOT_PRIM = "DrAnmarAdaptiveAnastomosisTool"
STANDALONE_ROOT = "DrAnmarAdaptiveAnastomosisToolStandalone"
PROXY_ROOT = "DrAnmarAdaptiveAnastomosisToolRigidProxy"
TISSUE_ROOT = "DrAnmarHollowTissueDemo"
STAPLE_ROOT = "DrAnmarAnastomosisStaple"
COLLAR_ROOT = "DrAnmarReinforcementCollar"
COLLAR_PROXY_ROOT = "DrAnmarReinforcementCollarRigidProxy"
DROPLET_ROOT = "DrAnmarLeakTestDroplet"

SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[1]
ASSET_ROOT = PACKAGE_ROOT / "assets" / CATALOG_SUBPATH
GLB_ROOT = ASSET_ROOT / "glb"
TEXTURE_ROOT = ASSET_ROOT / "textures"
PREVIEW_ROOT = PACKAGE_ROOT / "previews"
DOCS_ROOT = PACKAGE_ROOT / "docs" / "adaptive_anastomosis_robot"
EXAMPLE_ROOT = PACKAGE_ROOT / "examples"
EXTENSION_ROOT = PACKAGE_ROOT / "source/extensions/orbit.surgical.assets"
INTEGRATION_PATH = EXTENSION_ROOT / "orbit/surgical/assets/adaptive_anastomosis_robot.py"
PHYSICS_PROFILE_PATH = PACKAGE_ROOT / "physics_next/surgical-reconstruction/dranmar-adaptive-anastomosis-v1.json"

WORK_AXIS_Z = 0.205
FRANKA_HAND_EQUIVALENT_ROTATION_DEG = -45.0
STAPLE_COUNT = 16
CAPTURE_CELL_COUNT_PER_SIDE = 6
COLLAR_SECTOR_COUNT = 16
COLLAR_CAPACITY = 3
TEST_RESERVOIR_ML = 60.0


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
    if n <= 1.0e-12 or not math.isfinite(n):
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


def ring_staple_mesh(formed: bool, *, scale: float = 1.0) -> trimesh.Trimesh:
    half_span = 0.0032 * scale
    outer = 0.0036 * scale
    inner = -0.0048 * scale
    if formed:
        points = [
            (-half_span, inner, 0), (-0.0038*scale,-0.0014*scale,0), (-0.0035*scale,outer,0),
            (0.0035*scale,outer,0), (0.0038*scale,-0.0014*scale,0), (half_span,inner,0),
            (0.0016*scale,-0.0032*scale,0), (0.0,-0.0038*scale,0), (-0.0016*scale,-0.0032*scale,0),
        ]
    else:
        points = [
            (-half_span,inner,0), (-half_span,0.0014*scale,0), (-0.0038*scale,outer,0),
            (0.0038*scale,outer,0), (half_span,0.0014*scale,0), (half_span,inner,0),
        ]
    return wire_path(points, 0.00042*scale)


def tube_wall_mesh(length: float = 0.065, outer_radius: float = 0.012, wall: float = 0.0024, axial: int = 16, radial: int = 32) -> trimesh.Trimesh:
    inner_radius = outer_radius - wall
    vertices: list[tuple[float,float,float]] = []
    faces: list[tuple[int,int,int]] = []
    # axis is local X
    for radius in (outer_radius, inner_radius):
        for i in range(axial+1):
            x = -length/2 + length*i/axial
            for j in range(radial):
                a=2*math.pi*j/radial
                vertices.append((x,radius*math.cos(a),radius*math.sin(a)))
    ring=(axial+1)*radial
    # outer surface
    for i in range(axial):
        for j in range(radial):
            k=(j+1)%radial
            a=i*radial+j;b=i*radial+k;c=(i+1)*radial+j;d=(i+1)*radial+k
            faces += [(a,c,d),(a,d,b)]
    # inner surface reverse
    for i in range(axial):
        for j in range(radial):
            k=(j+1)%radial
            a=ring+i*radial+j;b=ring+i*radial+k;c=ring+(i+1)*radial+j;d=ring+(i+1)*radial+k
            faces += [(a,d,c),(a,b,d)]
    # annular end caps
    for end_i in (0, axial):
        for j in range(radial):
            k=(j+1)%radial
            o0=end_i*radial+j;o1=end_i*radial+k
            i0=ring+end_i*radial+j;i1=ring+end_i*radial+k
            if end_i==0:
                faces += [(o0,o1,i1),(o0,i1,i0)]
            else:
                faces += [(o0,i1,o1),(o0,i0,i1)]
    mesh=trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),process=False)
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def reinforcement_collar_mesh(major_radius=0.013, axial_width=0.012, thickness=0.0012, circumferential=96, axial=8) -> trimesh.Trimesh:
    # Ribbon wrapped around local X axis. Thickness is expressed as a slight radial dome.
    vertices=[];faces=[]
    for i in range(axial+1):
        x=-axial_width/2+axial_width*i/axial
        for j in range(circumferential):
            a=2*math.pi*j/circumferential
            r=major_radius+0.00035*math.cos(math.pi*x/(axial_width/2+1e-12))
            vertices.append((x,r*math.cos(a),r*math.sin(a)))
    for i in range(axial):
        for j in range(circumferential):
            k=(j+1)%circumferential
            a=i*circumferential+j;b=i*circumferential+k;c=(i+1)*circumferential+j;d=(i+1)*circumferential+k
            faces += [(a,c,d),(a,d,b)]
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
    tissue_left: trimesh.Trimesh
    tissue_right: trimesh.Trimesh
    open_staple: trimesh.Trimesh
    formed_staple: trimesh.Trimesh
    collar: trimesh.Trimesh


def _staple_at_angle(base: trimesh.Trimesh, angle: float, radius: float) -> trimesh.Trimesh:
    mesh = base.copy()
    mesh.apply_translation((0.0, radius, 0.0))
    mesh.apply_transform(np.block([[rotation_matrix((1,0,0), angle), np.zeros((3,1))],[np.zeros((1,3)), np.ones((1,1))]]))
    return mesh


def _petal_mesh(angle: float, radius: float, x_length: float = 0.019) -> trimesh.Trimesh:
    mesh = ellipsoid_mesh((x_length/2, 0.0045, 0.0028), (0, radius, 0), subdivisions=2)
    mesh.apply_transform(np.block([[rotation_matrix((1,0,0), angle), np.zeros((3,1))],[np.zeros((1,3)), np.ones((1,1))]]))
    return mesh


def build_tool() -> ToolBundle:
    links: dict[str,Link] = {}
    mount_visuals: list[Visual] = [
        Visual("FrankaAdapterPlate", cylinder_axis(0.032,0.012,"z",(0,0,0.006),sections=72), "MountMetal", ("franka_mount",)),
        Visual("QuickReleaseRing", torus_axis(0.0275,0.003,"z",(0,0,0.014),major_sections=72,minor_sections=14), "MountMetal"),
        Visual("MainHousing", ellipsoid_mesh((0.061,0.052,0.035),(0,0,0.058),subdivisions=3), "BodyPolymer", ("adaptive_anastomosis_robot",)),
        Visual("HousingCore", box_mesh((0.122,0.096,0.052),(0,0,0.061)), "BodyPolymer"),
        Visual("ApproximationRail", box_mesh((0.205,0.026,0.020),(0,0,0.116)), "RailMetal", ("bilateral_approximation_rail",)),
        Visual("SensorBridge", box_mesh((0.058,0.017,0.016),(0,-0.050,0.092)), "DarkPolymer", ("anastomosis_sensor_bridge",)),
        Visual("StereoCameraLeft", cylinder_axis(0.0050,0.004,"y",(-0.014,-0.059,0.092),sections=36), "SensorGlass", ("rgb_camera",)),
        Visual("StereoCameraRight", cylinder_axis(0.0050,0.004,"y",(0.014,-0.059,0.092),sections=36), "SensorGlass", ("rgb_camera",)),
        Visual("DepthCamera", cylinder_axis(0.0044,0.004,"y",(0,-0.059,0.105),sections=36), "SensorBlue", ("depth_camera",)),
        Visual("PressureSensor", cylinder_axis(0.0060,0.011,"z",(0.047,0.030,0.090),sections=40), "SensorGlass", ("lumen_pressure_sensor",)),
        Visual("TestReservoir", cylinder_axis(0.019,0.046,"y",(0.024,0.041,0.059),sections=48), "TubeClear", ("leak_test_reservoir",)),
        Visual("TestMedium", cylinder_axis(0.0165,0.038,"y",(0.024,0.041,0.059),sections=48), "TestFluid", ("test_medium_inventory",)),
        Visual("LeakCollectionCanister", cylinder_axis(0.020,0.046,"y",(-0.024,0.041,0.059),sections=48), "TubeClear", ("leak_collection_canister",)),
        Visual("CollectionFill", cylinder_axis(0.017,0.012,"y",(-0.024,0.054,0.059),sections=48), "LeakFluid", ("collected_leak_medium",)),
        Visual("CollarCarouselHousing", cylinder_axis(0.035,0.020,"z",(-0.046,0,0.098),sections=64), "AccentPolymer", ("reinforcement_collar_carousel",)),
        Visual("StapleCrownHousing", torus_axis(0.0185,0.0038,"x",(0.024,0,WORK_AXIS_Z),major_sections=72,minor_sections=14), "DarkPolymer", ("circumferential_staple_crown",)),
        Visual("LabelPanel", box_mesh((0.060,0.0012,0.024),(0,-0.052,0.054)), "LabelMaterial"),
    ]
    collar_mesh = reinforcement_collar_mesh()
    for i in range(COLLAR_CAPACITY):
        a=2*math.pi*i/COLLAR_CAPACITY
        mesh=transform(collar_mesh,(-0.046,0.020*math.cos(a),0.098+0.020*math.sin(a)),rotation_matrix((0,1,0),math.pi/2))
        mount_visuals.append(Visual(f"StoredCollar_{i:02d}",mesh,"CollarMaterial",("stored_reinforcement_collar",)))
    links["Mount"] = Link("Mount",(0,0,0),mount_visuals,[
        Collider("AdapterCollider","cylinder",(0,0,0.008),radius=0.032,height=0.016,physics_material="MountPhysics"),
        Collider("HousingCollider","box",(0,0,0.061),size=(0.128,0.102,0.076),physics_material="PolymerPhysics"),
        Collider("RailCollider","box",(0,0,0.116),size=(0.210,0.030,0.024),physics_material="MountPhysics"),
        Collider("StapleCrownHousingEnvelope","cylinder",(0.024,0,WORK_AXIS_Z),radius=0.023,height=0.010,axis="x",physics_material="PolymerPhysics",role="disabled_solid_debug_envelope_for_annular_staple_housing",author_enabled=False),
    ],0.490,("adaptive_anastomosis_end_effector","surgical_reconstruction_device"))

    for side_name,side in (("Left",-1),("Right",1)):
        x0=side*0.059
        carriage_visuals=[
            Visual("CarriageBody",box_mesh((0.036,0.044,0.030),(0,0,0)),"AccentPolymer",("anastomosis_carriage",)),
            Visual("LinearBearing",box_mesh((0.031,0.022,0.010),(0,0,-0.016)),"RailMetal"),
            Visual("ForceScale",box_mesh((0.023,0.0012,0.006),(0,-0.0225,0.006)),"LabelMaterial"),
            Visual("CableGuide",torus_axis(0.010,0.0012,"y",(0,0.016,0),major_sections=40,minor_sections=10),"DarkPolymer"),
        ]
        links[f"{side_name}Carriage"] = Link(f"{side_name}Carriage",(x0,0,0.116),carriage_visuals,[
            Collider("CarriageCollider","box",(0,0,0),size=(0.038,0.046,0.032),physics_material="PolymerPhysics"),
        ],0.082,("bilateral_tissue_approximation_carriage",))

        sleeve_visuals=[
            Visual("CaptureRing",torus_axis(0.017,0.0028,"x",(0,0,0),major_sections=72,minor_sections=14),"CaptureElastomer",("atraumatic_capture_ring",)),
            Visual("ActuationCone",frustum_axis(0.020,0.015,0.018,"x",(-side*0.004,0,0),sections=64),"AccentPolymer",("capture_sleeve",)),
        ]
        sleeve_colliders=[
            Collider("RingEnvelope","cylinder",(0,0,0),radius=0.021,height=0.012,axis="x",physics_material="CapturePhysics",role="disabled_solid_debug_envelope_for_annular_capture_ring",author_enabled=False),
        ]
        for i in range(CAPTURE_CELL_COUNT_PER_SIDE):
            a=2*math.pi*i/CAPTURE_CELL_COUNT_PER_SIDE
            sleeve_visuals.append(Visual(f"Petal_{i:02d}",_petal_mesh(a,0.0145),"CaptureElastomer",("capture_petal",)))
            R=rotation_matrix((1,0,0),a)
            q=matrix_to_quat_wxyz(R)
            sleeve_colliders.append(Collider(f"CaptureCell_{i:02d}","box",(0,0.0145*math.cos(a),0.0145*math.sin(a)),size=(0.019,0.008,0.0055),orientation_wxyz=q,physics_material="CapturePhysics",role=f"{side_name.lower()}_tissue_capture_cell"))
        links[f"{side_name}CaptureSleeve"] = Link(f"{side_name}CaptureSleeve",(x0,0,WORK_AXIS_Z),sleeve_visuals,sleeve_colliders,0.061,("atraumatic_circumferential_tissue_capture",))

        eversion_visuals=[
            Visual("EversionRing",torus_axis(0.0138,0.0022,"x",(0,0,0),major_sections=72,minor_sections=14),"EversionElastomer",("edge_eversion_ring",)),
            Visual("EversionBacking",torus_axis(0.0178,0.0014,"x",(-side*0.004,0,0),major_sections=72,minor_sections=12),"RailMetal"),
        ]
        for i in range(6):
            a=2*math.pi*(i+0.5)/6
            eversion_visuals.append(Visual(f"EversionLobe_{i:02d}",_petal_mesh(a,0.0124,0.010),"EversionElastomer",("eversion_lobe",)))
        eversion_colliders=[
            Collider("EversionEnvelope","cylinder",(0,0,0),radius=0.017,height=0.007,axis="x",physics_material="EversionPhysics",role=f"disabled_solid_debug_envelope_for_{side_name.lower()}_eversion_ring",author_enabled=False),
        ]
        for i in range(6):
            a=2*math.pi*(i+0.5)/6
            eversion_colliders.append(
                Collider(
                    f"EversionContact_{i:02d}","box",
                    (0,0.0138*math.cos(a),0.0138*math.sin(a)),
                    size=(0.007,0.006,0.004),
                    orientation_wxyz=matrix_to_quat_wxyz(rotation_matrix((1,0,0),a)),
                    physics_material="EversionPhysics",
                    role=f"{side_name.lower()}_edge_eversion_contact_sector",
                )
            )
        links[f"{side_name}EversionRing"] = Link(
            f"{side_name}EversionRing",(side*0.045,0,WORK_AXIS_Z),
            eversion_visuals,eversion_colliders,0.034,
            ("wound_edge_eversion_mechanism",),
        )

    mandrel_visuals=[
        Visual("MandrelShaft",cylinder_axis(0.0058,0.145,"x",(0.0725,0,0),sections=64),"MandrelPolymer",("lumen_alignment_mandrel",)),
        Visual("AtraumaticTip",ellipsoid_mesh((0.008,0.0062,0.0062),(0.146,0,0),subdivisions=3),"MandrelSoftTip",("mandrel_tip",)),
        Visual("PressureLumen",cylinder_axis(0.0015,0.130,"x",(0.067,0,0),sections=36),"TestFluid",("pressure_lumen",)),
        Visual("LeftOccluderCuff",torus_axis(0.0080,0.0022,"x",(0.047,0,0),major_sections=56,minor_sections=14),"OccluderElastomer",("left_test_occluder",)),
        Visual("RightOccluderCuff",torus_axis(0.0080,0.0022,"x",(0.098,0,0),major_sections=56,minor_sections=14),"OccluderElastomer",("right_test_occluder",)),
    ]
    links["Mandrel"] = Link("Mandrel",(-0.075,0,WORK_AXIS_Z),mandrel_visuals,[
        Collider("MandrelCollider","capsule",(0.0725,0,0),radius=0.0063,height=0.132,axis="x",physics_material="MandrelPhysics",role="lumen_patency_mandrel"),
        Collider("AtraumaticTipCollider","sphere",(0.146,0,0),radius=0.0065,physics_material="MandrelTipPhysics",role="atraumatic_mandrel_tip"),
    ],0.072,("lumen_alignment_and_patency_mandrel",))

    expander_visuals=[Visual("ExpansionHub",cylinder_axis(0.0068,0.020,"x",(0,0,0),sections=48),"MandrelPolymer",("patency_expansion_hub",))]
    for i in range(6):
        a=2*math.pi*i/6
        points=[(-0.020,0.006*math.cos(a),0.006*math.sin(a)),(0,0.0105*math.cos(a),0.0105*math.sin(a)),(0.020,0.006*math.cos(a),0.006*math.sin(a))]
        expander_visuals.append(Visual(f"ExpansionRib_{i:02d}",wire_path(points,0.00085),"MandrelSoftTip",("lumen_centering_rib",)))
    expander_colliders=[
        Collider("ExpansionEnvelope","cylinder",(0,0,0),radius=0.011,height=0.044,axis="x",physics_material="MandrelTipPhysics",role="disabled_solid_debug_envelope_for_six_rib_expander",author_enabled=False),
    ]
    for i in range(6):
        a=2*math.pi*i/6
        expander_colliders.append(
            Collider(
                f"ExpansionRibContact_{i:02d}","box",
                (0,0.0102*math.cos(a),0.0102*math.sin(a)),
                size=(0.040,0.003,0.003),
                orientation_wxyz=matrix_to_quat_wxyz(rotation_matrix((1,0,0),a)),
                physics_material="MandrelTipPhysics",
                role="lumen_centering_rib_contact_sector",
            )
        )
    links["MandrelExpander"] = Link(
        "MandrelExpander",(0,0,WORK_AXIS_Z),expander_visuals,
        expander_colliders,0.026,("lumen_centering_expander",),
    )

    # Fixed left anvil ring.
    anvil_visuals=[
        Visual("AnvilRing",torus_axis(0.0152,0.0027,"x",(0,0,0),major_sections=72,minor_sections=14),"AnvilMetal",("circumferential_staple_anvil",)),
        Visual("AnvilBacking",cylinder_axis(0.0205,0.006,"x",(0,0,0),sections=72),"JawMetal"),
    ]
    for i in range(STAPLE_COUNT):
        a=2*math.pi*i/STAPLE_COUNT
        pocket=box_mesh((0.0045,0.0040,0.0015),(0,0.0152*math.cos(a),0.0152*math.sin(a)),rotation_matrix((1,0,0),a))
        anvil_visuals.append(Visual(f"AnvilPocket_{i:02d}",pocket,"AnvilMetal",("staple_forming_pocket",)))
    anvil_colliders=[
        Collider("AnvilEnvelope","cylinder",(0,0,0),radius=0.021,height=0.007,axis="x",physics_material="MetalPhysics",role="disabled_solid_debug_envelope_for_annular_anvil",author_enabled=False),
    ]
    for i in range(STAPLE_COUNT):
        a=2*math.pi*i/STAPLE_COUNT
        anvil_colliders.append(
            Collider(
                f"AnvilContact_{i:02d}","box",
                (0,0.0152*math.cos(a),0.0152*math.sin(a)),
                size=(0.007,0.0045,0.0030),
                orientation_wxyz=matrix_to_quat_wxyz(rotation_matrix((1,0,0),a)),
                physics_material="MetalPhysics",
                role="staple_forming_anvil_sector",
            )
        )
    links["StapleAnvil"] = Link(
        "StapleAnvil",(-0.024,0,WORK_AXIS_Z),anvil_visuals,
        anvil_colliders,0.082,("staple_forming_anvil_ring",),
    )

    open_staple=ring_staple_mesh(False)
    driver_visuals=[
        Visual("DriverRing",torus_axis(0.0168,0.0030,"x",(0,0,0),major_sections=72,minor_sections=14),"JawMetal",("circumferential_staple_driver",)),
        Visual("PusherPlate",cylinder_axis(0.021,0.008,"x",(0.002,0,0),sections=72),"DarkPolymer"),
    ]
    for i in range(STAPLE_COUNT):
        a=2*math.pi*i/STAPLE_COUNT
        driver_visuals.append(Visual(f"ChamberedStaple_{i:02d}",_staple_at_angle(open_staple,a,0.0108),"StapleMetal",("chambered_anastomosis_staple",)))
    driver_colliders=[
        Collider("DriverEnvelope","cylinder",(0,0,0),radius=0.022,height=0.010,axis="x",physics_material="MetalPhysics",role="disabled_solid_debug_envelope_for_annular_driver",author_enabled=False),
        Collider("StapleExitEnvelope","cylinder",(-0.005,0,0),radius=0.018,height=0.008,axis="x",physics_material="StaplePhysics",role="disabled_solid_debug_envelope_for_annular_staple_exit",author_enabled=False),
    ]
    for i in range(STAPLE_COUNT):
        a=2*math.pi*i/STAPLE_COUNT
        driver_colliders.append(
            Collider(
                f"DriverContact_{i:02d}","box",
                (-0.005,0.0168*math.cos(a),0.0168*math.sin(a)),
                size=(0.010,0.0045,0.0030),
                orientation_wxyz=matrix_to_quat_wxyz(rotation_matrix((1,0,0),a)),
                physics_material="StaplePhysics",
                role="circumferential_staple_driver_sector",
            )
        )
    links["StapleDriver"] = Link(
        "StapleDriver",(0.028,0,WORK_AXIS_Z),driver_visuals,
        driver_colliders,0.118,("circumferential_staple_crown",),
    )

    collar_carousel_visuals=[
        Visual("CarouselDisk",cylinder_axis(0.031,0.018,"z",(0,0,0),sections=64),"AccentPolymer",("reinforcement_collar_carousel",)),
        Visual("IndexHub",cylinder_axis(0.010,0.024,"z",(0,0,0),sections=48),"RailMetal"),
    ]
    links["CollarCarousel"] = Link("CollarCarousel",(-0.046,0,0.098),collar_carousel_visuals,[
        Collider("CarouselCollider","cylinder",(0,0,0),radius=0.033,height=0.020,physics_material="PolymerPhysics"),
    ],0.096,("reinforcement_inventory_carousel",))

    applicator_visuals=[
        Visual("ApplicatorStem",box_mesh((0.018,0.018,0.050),(0,0,0)),"RailMetal",("collar_applicator_stem",)),
        Visual("CompliantPlaten",torus_axis(0.016,0.0028,"x",(0,0,0.028),major_sections=72,minor_sections=14),"EversionElastomer",("collar_application_platen",)),
        Visual("LoadedCollar",transform(collar_mesh,(0,0,0.028)),"CollarMaterial",("loaded_reinforcement_collar",)),
    ]
    applicator_colliders=[Collider("StemCollider","box",(0,0,0),size=(0.020,0.020,0.052),physics_material="MountPhysics")]
    for i in range(COLLAR_SECTOR_COUNT):
        a=2*math.pi*i/COLLAR_SECTOR_COUNT
        applicator_colliders.append(Collider(f"CollarBondCell_{i:02d}","box",(0,0.013*math.cos(a),0.028+0.013*math.sin(a)),size=(0.013,0.0045,0.0045),orientation_wxyz=matrix_to_quat_wxyz(rotation_matrix((1,0,0),a)),physics_material="CollarPhysics",role="reinforcement_collar_application_cell"))
    links["CollarApplicator"] = Link("CollarApplicator",(0,0,0.155),applicator_visuals,applicator_colliders,0.074,("bioadhesive_reinforcement_collar_applicator",))

    for side_name,side in (("Left",-1),("Right",1)):
        links[f"{side_name}OccluderValve"] = Link(f"{side_name}OccluderValve",(side*0.024,0.029,0.094),[
            Visual("ValveBody",cylinder_axis(0.006,0.018,"z",(0,0,0),sections=40),"SensorBlue",("occluder_valve",)),
            Visual("ValvePlunger",cylinder_axis(0.0035,0.012,"z",(0,0,0.012),sections=32),"RailMetal"),
        ],[Collider("ValveCollider","cylinder",(0,0,0.004),radius=0.0065,height=0.022,physics_material="PolymerPhysics")],0.018,("lumen_occluder_control",))

    links["PressureValve"] = Link("PressureValve",(0.047,0.030,0.108),[
        Visual("PressureValveBody",cylinder_axis(0.006,0.020,"z",(0,0,0),sections=40),"SensorPurple",("leak_test_pressure_valve",)),
        Visual("PressurePlunger",cylinder_axis(0.0035,0.014,"z",(0,0,0.013),sections=32),"RailMetal"),
    ],[Collider("PressureValveCollider","cylinder",(0,0,0.004),radius=0.0065,height=0.024,physics_material="PolymerPhysics")],0.020,("leak_test_valve",))

    joints=[
        Joint("left_approximation_joint","prismatic","Mount","LeftCarriage","X",(-0.059,0,0.116),(0,0,0),0.0,0.034,9000,260,180),
        Joint("right_approximation_joint","prismatic","Mount","RightCarriage","X",(0.059,0,0.116),(0,0,0),-0.034,0.0,9000,260,180),
        Joint("left_capture_joint","prismatic","LeftCarriage","LeftCaptureSleeve","X",(0,0,WORK_AXIS_Z-0.116),(0,0,0),0.0,0.010,6000,180,110),
        Joint("right_capture_joint","prismatic","RightCarriage","RightCaptureSleeve","X",(0,0,WORK_AXIS_Z-0.116),(0,0,0),-0.010,0.0,6000,180,110),
        Joint("left_eversion_joint","prismatic","Mount","LeftEversionRing","X",(-0.045,0,WORK_AXIS_Z),(0,0,0),0.0,0.008,7200,210,130),
        Joint("right_eversion_joint","prismatic","Mount","RightEversionRing","X",(0.045,0,WORK_AXIS_Z),(0,0,0),-0.008,0.0,7200,210,130),
        Joint("mandrel_extension_joint","prismatic","Mount","Mandrel","X",(-0.075,0,WORK_AXIS_Z),(0,0,0),-0.060,0.0,5400,150,90),
        Joint("mandrel_expansion_joint","prismatic","Mount","MandrelExpander","Z",(0,0,WORK_AXIS_Z),(0,0,0),0.0,0.006,4200,140,70),
        Joint("staple_anvil_mount_joint","fixed","Mount","StapleAnvil",None,(-0.024,0,WORK_AXIS_Z),(0,0,0)),
        Joint("staple_driver_joint","prismatic","Mount","StapleDriver","X",(0.028,0,WORK_AXIS_Z),(0,0,0),-0.030,0.0,17000,360,320),
        Joint("collar_carousel_joint","revolute","Mount","CollarCarousel","Z",(-0.046,0,0.098),(0,0,0),0.0,360.0,160,16,22),
        Joint("collar_applicator_joint","prismatic","Mount","CollarApplicator","Z",(0,0,0.155),(0,0,0),0.0,0.050,7000,190,120),
        Joint("left_occluder_valve_joint","prismatic","Mount","LeftOccluderValve","Z",(-0.024,0.029,0.094),(0,0,0),0.0,0.008,1600,52,24),
        Joint("right_occluder_valve_joint","prismatic","Mount","RightOccluderValve","Z",(0.024,0.029,0.094),(0,0,0),0.0,0.008,1600,52,24),
        Joint("pressure_valve_joint","prismatic","Mount","PressureValve","Z",(0.047,0.030,0.108),(0,0,0),0.0,0.008,1800,58,28),
    ]

    frames={
        "panda_link8_mount":{"parent_link":"Mount","position":(0,0,0),"orientation_wxyz":(1,0,0,0),"role":"Franka panda_link8 mounting frame"},
        "anastomosis_tcp":{"parent_link":"Mount","position":(0,0,WORK_AXIS_Z),"orientation_wxyz":(1,0,0,0),"role":"coaxial anastomosis tool center point"},
        "seam_reference":{"parent_link":"Mount","position":(0,0,WORK_AXIS_Z),"orientation_wxyz":(1,0,0,0),"role":"target anastomosis seam center"},
        "lumen_axis_reference":{"parent_link":"Mount","position":(0,0,WORK_AXIS_Z),"orientation_wxyz":(0.70710678,0,0.70710678,0),"role":"positive local X lumen axis"},
        "camera_left":{"parent_link":"Mount","position":(-0.014,-0.059,0.092),"orientation_wxyz":(0.70710678,0.70710678,0,0),"role":"left RGB camera optical frame"},
        "camera_right":{"parent_link":"Mount","position":(0.014,-0.059,0.092),"orientation_wxyz":(0.70710678,0.70710678,0,0),"role":"right RGB camera optical frame"},
        "pressure_sensor":{"parent_link":"Mount","position":(0.047,0.030,0.090),"orientation_wxyz":(1,0,0,0),"role":"lumen pressure measurement reference"},
        "leak_observation":{"parent_link":"Mount","position":(0,-0.035,WORK_AXIS_Z),"orientation_wxyz":(1,0,0,0),"role":"external seam leak observation reference"},
        "count_reference":{"parent_link":"Mount","position":(0,0,0.035),"orientation_wxyz":(1,0,0,0),"role":"inventory and counting reference"},
        "disposal_reference":{"parent_link":"Mount","position":(0,0,0.020),"orientation_wxyz":(1,0,0,0),"role":"end effector disposal reference"},
        "left_capture_center":{"parent_link":"LeftCaptureSleeve","position":(0,0,0),"orientation_wxyz":(1,0,0,0),"role":"left tissue capture center"},
        "left_tissue_edge_reference":{"parent_link":"LeftCaptureSleeve","position":(0.010,0,0),"orientation_wxyz":(1,0,0,0),"role":"left tissue edge target"},
        "right_capture_center":{"parent_link":"RightCaptureSleeve","position":(0,0,0),"orientation_wxyz":(1,0,0,0),"role":"right tissue capture center"},
        "right_tissue_edge_reference":{"parent_link":"RightCaptureSleeve","position":(-0.010,0,0),"orientation_wxyz":(1,0,0,0),"role":"right tissue edge target"},
        "left_eversion_contact":{"parent_link":"LeftEversionRing","position":(0,0,0),"orientation_wxyz":(1,0,0,0),"role":"left eversion contact ring"},
        "right_eversion_contact":{"parent_link":"RightEversionRing","position":(0,0,0),"orientation_wxyz":(1,0,0,0),"role":"right eversion contact ring"},
        "mandrel_tip":{"parent_link":"Mandrel","position":(0.146,0,0),"orientation_wxyz":(1,0,0,0),"role":"atraumatic lumen insertion tip"},
        "pressure_inlet":{"parent_link":"Mandrel","position":(0.072,0,0),"orientation_wxyz":(1,0,0,0),"role":"leak-test medium inlet"},
        "patency_reference":{"parent_link":"MandrelExpander","position":(0,0,0),"orientation_wxyz":(1,0,0,0),"role":"minimum-lumen patency measurement center"},
        "staple_anvil_reference":{"parent_link":"StapleAnvil","position":(0,0,0),"orientation_wxyz":(1,0,0,0),"role":"circumferential anvil ring center"},
        "staple_crown_reference":{"parent_link":"StapleDriver","position":(-0.005,0,0),"orientation_wxyz":(1,0,0,0),"role":"circumferential staple deployment plane"},
        "collar_application":{"parent_link":"CollarApplicator","position":(0,0,0.028),"orientation_wxyz":(1,0,0,0),"role":"reinforcement collar placement frame"},
    }

    left_tissue=transform(tube_wall_mesh(),(-0.0425,0,0))
    right_tissue=transform(tube_wall_mesh(),(0.0425,0,0))
    return ToolBundle(links,joints,frames,left_tissue,right_tissue,open_staple,ring_staple_mesh(True),collar_mesh)


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
{indent}    normal3f[] normals = [
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
{indent}    float physxCollision:contactOffset = 0.00045
{indent}    float physxCollision:restOffset = 0
{indent}    rel material:binding:physics = <{root_path}/PhysicsMaterials/{collider.physics_material}>
{indent}    custom string drAnmar:role = "{collider.role}"
{indent}    quatf xformOp:orient = {quat(collider.orientation_wxyz)}
{indent}    double3 xformOp:translate = {vec(collider.center)}'''
    schemas='["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]'
    if collider.kind == "box":
        assert collider.size is not None
        size = max(collider.size)
        scale = tuple(v / size for v in collider.size)
        return f'''{indent}def Cube "{collider.name}" (
{indent}    prepend apiSchemas = {schemas}
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
{indent}    prepend apiSchemas = {schemas}
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
{indent}    prepend apiSchemas = {schemas}
{indent})
{indent}{{
{indent}    double radius = {f(collider.radius)}
{common_lines}
{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
{indent}}}'''
    if collider.kind == "capsule":
        assert collider.radius is not None and collider.height is not None
        return f'''{indent}def Capsule "{collider.name}" (
{indent}    prepend apiSchemas = {schemas}
{indent})
{indent}{{
{indent}    uniform token axis = "{collider.axis.upper()}"
{indent}    double radius = {f(collider.radius)}
{indent}    double height = {f(collider.height)}
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
        "BodyPolymer":((0.86,0.88,0.90),0.0,0.23,1.0,None),
        "AccentPolymer":((0.08,0.39,0.67),0.0,0.23,1.0,None),
        "DarkPolymer":((0.045,0.055,0.070),0.0,0.30,1.0,None),
        "MountMetal":((0.48,0.52,0.57),0.88,0.20,1.0,None),
        "RailMetal":((0.28,0.33,0.39),0.84,0.24,1.0,None),
        "JawMetal":((0.57,0.61,0.65),0.90,0.18,1.0,None),
        "AnvilMetal":((0.72,0.74,0.76),0.94,0.14,1.0,None),
        "StapleMetal":((0.67,0.69,0.72),0.92,0.15,1.0,"./textures/metal_microtexture.png"),
        "CaptureElastomer":((0.06,0.52,0.60),0.0,0.54,1.0,None),
        "EversionElastomer":((0.12,0.66,0.55),0.0,0.58,1.0,None),
        "MandrelPolymer":((0.60,0.78,0.90),0.0,0.18,0.52,None),
        "MandrelSoftTip":((0.28,0.72,0.84),0.0,0.48,0.84,None),
        "OccluderElastomer":((0.22,0.58,0.82),0.0,0.52,0.88,None),
        "SensorGlass":((0.05,0.12,0.19),0.05,0.08,0.78,None),
        "SensorBlue":((0.08,0.50,0.92),0.0,0.18,1.0,None),
        "SensorPurple":((0.50,0.13,0.72),0.0,0.18,1.0,None),
        "TestFluid":((0.20,0.62,0.96),0.0,0.08,0.70,None),
        "LeakFluid":((0.16,0.48,0.92),0.0,0.10,0.76,None),
        "TubeClear":((0.72,0.84,0.90),0.0,0.10,0.34,None),
        "TissueOuter":((0.67,0.22,0.18),0.0,0.52,1.0,"./textures/tissue_basecolor.png"),
        "TissueInner":((0.78,0.34,0.28),0.0,0.48,1.0,None),
        "CollarMaterial":((0.94,0.82,0.52),0.0,0.66,0.94,"./textures/reinforcement_collar_basecolor.png"),
        "LabelMaterial":((0.97,0.98,0.99),0.0,0.36,1.0,"./textures/label_dranmar.png"),
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
        "MetalPhysics":(0.27,0.19,0.02),
        "CapturePhysics":(0.76,0.62,0.01),
        "EversionPhysics":(0.68,0.56,0.01),
        "MandrelPhysics":(0.20,0.14,0.01),
        "MandrelTipPhysics":(0.42,0.30,0.01),
        "StaplePhysics":(0.30,0.22,0.02),
        "CollarPhysics":(0.74,0.60,0.01),
        "TissuePhysics":(0.52,0.38,0.0),
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
    schema_line=f"        {schemas}\n" if schemas else ""
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
        int physxRigidBody:solverPositionIterationCount = 18
        int physxRigidBody:solverVelocityIterationCount = 5
'''
    visual_blocks="\n".join(mesh_usda(v,f"{root_path}/Looks/{v.material}") for v in link.visuals)
    collider_blocks="\n".join(collider_usda(c,root_path) for c in link.colliders)
    frame_blocks="\n".join(frame_usda(n,d) for n,d in frames.items() if d["parent_link"]==link.name)
    labels=", ".join(f'"{x}"' for x in link.labels)
    labels_attr=f'        custom token[] drAnmar:labels = [{labels}]\n' if labels else ''
    return f'''    def Xform "{link.name}" (
{schema_line.rstrip()}
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
    api_line=f"        {api}\n" if api else ""
    return f'''    def {typename} "{joint.name}" (
{api_line.rstrip()}
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


def state_variants() -> str:
    staple_names=[f"ChamberedStaple_{i:02d}" for i in range(STAPLE_COUNT)]
    collar_names=[f"StoredCollar_{i:02d}" for i in range(COLLAR_CAPACITY)]

    def visibility_leaf(path: Sequence[str], value: str) -> str:
        return _nested_over(path,[f'token visibility = "{value}"'])

    def visibility_children(path: Sequence[str], names: Sequence[str], value: str) -> str:
        child=[]
        for name in names:
            child.extend(_nested_over([name],[f'token visibility = "{value}"'],indent="                        ").splitlines())
        return _nested_over(path,child)

    staple_loaded=visibility_children(["Links","StapleDriver","Visuals"],staple_names,"inherited")
    staple_empty=visibility_children(["Links","StapleDriver","Visuals"],staple_names,"invisible")
    def collar_visibility(value: str) -> str:
        mount_visuals = []
        for name in collar_names:
            mount_visuals.extend(
                _nested_over(
                    [name],
                    [f'token visibility = "{value}"'],
                    indent="                            ",
                ).splitlines()
            )
        mount = _nested_over(["Mount", "Visuals"], mount_visuals, indent="                ")
        applicator = _nested_over(
            ["CollarApplicator", "Visuals", "LoadedCollar"],
            [f'token visibility = "{value}"'],
            indent="                ",
        )
        return _nested_over(["Links"], (mount + "\n" + applicator).splitlines())

    collar_loaded=collar_visibility("inherited")
    collar_empty=collar_visibility("invisible")
    medium_full=visibility_leaf(["Links","Mount","Visuals","TestMedium"],"inherited")
    medium_empty=visibility_leaf(["Links","Mount","Visuals","TestMedium"],"invisible")
    collection_empty=visibility_leaf(["Links","Mount","Visuals","CollectionFill"],"invisible")
    collection_visible=visibility_leaf(["Links","Mount","Visuals","CollectionFill"],"inherited")
    return f'''    variantSet "staple_state" = {{
        "loaded"
        {{
{staple_loaded}
        }}
        "empty"
        {{
{staple_empty}
        }}
    }}
    variantSet "collar_state" = {{
        "loaded"
        {{
{collar_loaded}
        }}
        "empty"
        {{
{collar_empty}
        }}
    }}
    variantSet "test_medium_state" = {{
        "full"
        {{
{medium_full}
        }}
        "empty"
        {{
{medium_empty}
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
    runtime_status="simulation_training_workcell"
    links="\n\n".join(link_usda(link,root_path,bundle.frames) for link in bundle.links.values())
    joints="\n\n".join(joint_usda(j,root_path) for j in bundle.joints)
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "{ASSET_NAME}: bilateral hollow-tissue capture, coaxial alignment, eversion, circumferential stapling, reinforcement, patency and leak verification research asset."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
{schema_line.rstrip()}
    prepend variantSets = ["staple_state", "collar_state", "test_medium_state", "collection_state"]
    variants = {{
        string staple_state = "loaded"
        string collar_state = "loaded"
        string test_medium_state = "full"
        string collection_state = "empty"
    }}
    customData = {{
        string drAnmarAssetId = "dranmar-adaptive-anastomosis-robot-v1"
        string drAnmarAssetVersion = "{VERSION}"
        bool drAnmarClinicalValidation = false
        bool drAnmarMedicalDevice = false
        string drAnmarStatus = "{runtime_status}"
        string drAnmarMount = "replaces_panda_hand_at_panda_link8"
        string drAnmarProcedure = "end_to_end_hollow_tissue_anastomosis"
        int drAnmarCircumferentialStapleCount = {STAPLE_COUNT}
        int drAnmarCaptureCellCountPerSide = {CAPTURE_CELL_COUNT_PER_SIDE}
        int drAnmarReinforcementSectorCount = {COLLAR_SECTOR_COUNT}
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
{state_variants()}
}}
'''


def material_color(material: str) -> tuple[int,int,int,int]:
    colors={
        "BodyPolymer":(218,224,229,255),"AccentPolymer":(28,107,176,255),"DarkPolymer":(20,25,32,255),
        "MountMetal":(126,139,153,255),"RailMetal":(75,86,101,255),"JawMetal":(150,158,166,255),
        "AnvilMetal":(188,192,196,255),"StapleMetal":(178,182,187,255),"CaptureElastomer":(22,138,151,255),
        "EversionElastomer":(45,170,138,255),"MandrelPolymer":(142,198,225,150),"MandrelSoftTip":(68,184,210,220),
        "OccluderElastomer":(59,144,205,225),"SensorGlass":(18,42,61,210),"SensorBlue":(25,130,235,255),
        "SensorPurple":(129,42,184,255),"TestFluid":(53,159,244,175),"LeakFluid":(44,128,220,190),
        "TubeClear":(190,214,226,100),"TissueOuter":(166,65,54,255),"TissueInner":(201,98,84,255),
        "CollarMaterial":(232,200,121,240),"LabelMaterial":(246,249,252,255),"GuideRed":(250,35,35,255),
        "GuideGreen":(35,245,50,255),"GuideBlue":(35,80,245,255),"CollisionDebug":(255,60,15,90),
    }
    return colors.get(material,(180,180,180,255))


def pbr(mesh: trimesh.Trimesh, material: str) -> trimesh.Trimesh:
    mesh=mesh.copy()
    rgba=material_color(material)
    mesh.visual=trimesh.visual.ColorVisuals(mesh=mesh,face_colors=np.tile(np.asarray(rgba,dtype=np.uint8),(len(mesh.faces),1)))
    return mesh


def rigid_proxy_usda(bundle: ToolBundle) -> str:
    root=PROXY_ROOT;root_path=f"/{root}"
    visuals=[]
    for link in bundle.links.values():
        for visual in link.visuals:
            visuals.append(Visual(f"{link.name}_{visual.name}",transform(visual.mesh,link.translation),visual.material,visual.labels))
    visual_blocks="\n".join(mesh_usda(v,f"{root_path}/Looks/{v.material}",indent="            ") for v in visuals)
    bmin,bmax=mesh_bounds([v.mesh for v in visuals]);size=bmax-bmin;center=(bmin+bmax)/2
    mass=1.45
    mp=box_mass_properties([v.mesh for v in visuals],mass)
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Rigid perception, planning, handover, and collision proxy for the {ASSET_NAME}."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    customData = {{
        string drAnmarAssetVersion = "{VERSION}"
        bool drAnmarClinicalValidation = false
        string drAnmarRepresentation = "rigid_perception_and_planning_proxy"
    }}
)
{{
    bool physics:rigidBodyEnabled = true
    float physics:mass = {f(mass)}
    point3f physics:centerOfMass = {vec(mp['center_of_mass_m'])}
    vector3f physics:diagonalInertia = {vec(mp['diagonal_inertia_kg_m2'])}
    quatf physics:principalAxes = (1, 0, 0, 0)
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(root)}
{physics_materials_scope()}
    def Scope "Visuals"
    {{
{visual_blocks}
    }}
    def Scope "Collisions"
    {{
{collider_usda(Collider('MainHousingCollider','box',tuple(center),size=tuple(size),physics_material='PolymerPhysics',role='whole_tool_proxy'),root_path,indent='        ')}
    }}
}}
'''


def _mesh_at_root(name: str, mesh: trimesh.Trimesh, material_path: str, labels: Sequence[str]=(), indent: str="        ") -> str:
    return mesh_usda(Visual(name,mesh,material_path.split('/')[-1],tuple(labels)),material_path,indent=indent)


def staple_usda(bundle: ToolBundle) -> str:
    root=STAPLE_ROOT;rp=f"/{root}"
    open_block=mesh_usda(Visual("Open",bundle.open_staple,"StapleMetal",("open_anastomosis_staple",)),f"{rp}/Looks/StapleMetal",indent="            ")
    formed_block=mesh_usda(Visual("Formed",bundle.formed_staple,"StapleMetal",("formed_anastomosis_staple",)),f"{rp}/Looks/StapleMetal",indent="            ")
    mass=4.2e-5
    mp=box_mass_properties([bundle.formed_staple],mass)
    def staple_visual_state(open_value: str, formed_value: str) -> str:
        children = []
        for name, value in (("Open", open_value), ("Formed", formed_value)):
            children.extend(
                _nested_over(
                    [name],
                    [f'token visibility = "{value}"'],
                    indent="                ",
                ).splitlines()
            )
        return _nested_over(["Visuals"], children)

    def staple_collision_state(open_enabled: bool, formed_enabled: bool) -> str:
        children = []
        for name, enabled in (
            ("OpenEnvelope", open_enabled),
            ("LeftLegAttachment", formed_enabled),
            ("RightLegAttachment", formed_enabled),
            ("CrownCollider", formed_enabled),
        ):
            children.extend(
                _nested_over(
                    [name],
                    [f"bool physics:collisionEnabled = {str(enabled).lower()}"],
                    indent="                ",
                ).splitlines()
            )
        return _nested_over(["Collisions"], children)

    open_state = (
        staple_visual_state("inherited", "invisible")
        + "\n"
        + staple_collision_state(True, False)
    )
    formed_state = (
        staple_visual_state("invisible", "inherited")
        + "\n"
        + staple_collision_state(False, True)
    )
    collision_blocks = "\n".join(
        collider_usda(collider,rp,indent="        ").replace(
            "bool physics:collisionEnabled = true\n",""
        )
        for collider in (
            Collider("OpenEnvelope","box",(0,0,0),size=(0.009,0.010,0.0018),physics_material="StaplePhysics",role="open_staple_collision"),
            Collider("LeftLegAttachment","box",(-0.0032,-0.0028,0),size=(0.0030,0.0050,0.0020),physics_material="StaplePhysics",role="left_tissue_retention_region"),
            Collider("RightLegAttachment","box",(0.0032,-0.0028,0),size=(0.0030,0.0050,0.0020),physics_material="StaplePhysics",role="right_tissue_retention_region"),
            Collider("CrownCollider","box",(0,0.0023,0),size=(0.0085,0.0040,0.0020),physics_material="StaplePhysics",role="formed_staple_crown"),
        )
    )
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Single open/formed DrAnmar circumferential anastomosis staple with independent left/right tissue-retention regions."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    prepend variantSets = "state"
    variants = {{ string state = "open" }}
    customData = {{
        string drAnmarAssetVersion = "{VERSION}"
        bool drAnmarClinicalValidation = false
        string drAnmarRepresentation = "discrete_open_and_formed_staple_states"
    }}
)
{{
    bool physics:rigidBodyEnabled = true
    float physics:mass = {f(mass)}
    point3f physics:centerOfMass = {vec(mp['center_of_mass_m'])}
    vector3f physics:diagonalInertia = {vec(mp['diagonal_inertia_kg_m2'])}
    quatf physics:principalAxes = (1, 0, 0, 0)
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(root)}
{physics_materials_scope()}
    def Scope "Visuals"
    {{
{open_block}
{formed_block}
    }}
    def Scope "Collisions"
    {{
{collision_blocks}
    }}
    variantSet "state" = {{
        "open"
        {{
{open_state}
        }}
        "formed"
        {{
{formed_state}
        }}
    }}
}}
'''


def collar_surface_usda(bundle: ToolBundle) -> str:
    root=COLLAR_ROOT;rp=f"/{root}"
    mesh=mesh_usda(Visual("SimulationMesh",bundle.collar,"CollarMaterial",("reinforcement_collar","surface_deformable_ready")),f"{rp}/Looks/CollarFresh",indent="        ")
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Connected circumferential reinforcement collar surface for post-staple anastomosis support."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend variantSets = "state"
    variants = {{ string state = "fresh" }}
    customData = {{
        string drAnmarAssetVersion = "{VERSION}"
        bool drAnmarClinicalValidation = false
        string drAnmarRepresentation = "connected_triangular_reinforcement_surface"
    }}
)
{{
    def Scope "Looks"
    {{
{shader_material(root,'CollarFresh',(0.94,0.82,0.52),0.0,0.66,0.94,'./textures/reinforcement_collar_basecolor.png')}
{shader_material(root,'CollarCured',(0.82,0.68,0.34),0.0,0.58,0.98,'./textures/reinforcement_collar_basecolor.png')}
    }}
{mesh}
    variantSet "state" = {{
        "fresh"
        {{
            over "SimulationMesh"
            {{
                rel material:binding = </{root}/Looks/CollarFresh>
            }}
        }}
        "cured"
        {{
            over "SimulationMesh"
            {{
                rel material:binding = </{root}/Looks/CollarCured>
            }}
        }}
    }}
}}
'''


def collar_proxy_usda(bundle: ToolBundle) -> str:
    root=COLLAR_PROXY_ROOT;rp=f"/{root}"
    mass=0.0045;mp=box_mass_properties([bundle.collar],mass)
    mesh=mesh_usda(Visual("Visual",bundle.collar,"CollarMaterial",("reinforcement_collar","rigid_bond_carrier")),f"{rp}/Looks/CollarMaterial",indent="        ")
    cells=[]
    for i in range(COLLAR_SECTOR_COUNT):
        a=2*math.pi*i/COLLAR_SECTOR_COUNT
        q=matrix_to_quat_wxyz(rotation_matrix((1,0,0),a))
        for side,x in (("Left",-0.0035),("Right",0.0035)):
            cells.append(collider_usda(Collider(f"{side}BondCell_{i:02d}","box",(x,0.013*math.cos(a),0.013*math.sin(a)),size=(0.005,0.0045,0.0045),orientation_wxyz=q,physics_material="CollarPhysics",role=f"{side.lower()}_tissue_bond_sector"),rp,indent="        "))
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Rigid bond-carrier proxy for the DrAnmar reinforcement collar, with independent left/right circumferential attachment sectors."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
    prepend variantSets = "state"
    variants = {{ string state = "fresh" }}
)
{{
    bool physics:rigidBodyEnabled = true
    float physics:mass = {f(mass)}
    point3f physics:centerOfMass = {vec(mp['center_of_mass_m'])}
    vector3f physics:diagonalInertia = {vec(mp['diagonal_inertia_kg_m2'])}
    quatf physics:principalAxes = (1, 0, 0, 0)
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(root)}
{physics_materials_scope()}
{mesh}
    def Scope "Collisions"
    {{
{chr(10).join(cells)}
    }}
    variantSet "state" = {{
        "fresh"
        {{
            over "Visual"
            {{
                rel material:binding = </{root}/Looks/CollarMaterial>
            }}
        }}
        "cured"
        {{
            over "Visual"
            {{
                rel material:binding = </{root}/Looks/CollarMaterial>
            }}
        }}
    }}
}}
'''


def tissue_usda(bundle: ToolBundle) -> str:
    root=TISSUE_ROOT;rp=f"/{root}"
    left_local=transform(bundle.tissue_left,(0.0425,0,0))
    right_local=transform(bundle.tissue_right,(-0.0425,0,0))
    left_mesh=mesh_usda(Visual("SimulationMesh",left_local,"TissueOuter",("left_hollow_tissue","surface_deformable_ready")),f"{rp}/Looks/TissueOuter",indent="            ")
    right_mesh=mesh_usda(Visual("SimulationMesh",right_local,"TissueOuter",("right_hollow_tissue","surface_deformable_ready")),f"{rp}/Looks/TissueOuter",indent="            ")
    initial_left=_nested_over(["LeftTissue"],["double3 xformOp:translate = (-0.0425, 0, 0)"])
    initial_right=_nested_over(["RightTissue"],["double3 xformOp:translate = (0.0425, 0, 0)"])
    aligned_left=_nested_over(["LeftTissue"],["double3 xformOp:translate = (-0.0325, 0, 0)"])
    aligned_right=_nested_over(["RightTissue"],["double3 xformOp:translate = (0.0325, 0, 0)"])
    closed_left=_nested_over(["LeftTissue"],["double3 xformOp:translate = (-0.0320, 0, 0)"])
    closed_right=_nested_over(["RightTissue"],["double3 xformOp:translate = (0.0320, 0, 0)"])
    frames='''        def Xform "seam_center"
        {
            custom string drAnmar:role = "target end-to-end anastomosis seam"
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Xform "left_distal_anchor"
        {
            custom string drAnmar:role = "left distal fixture band"
            double3 xformOp:translate = (-0.073, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Xform "right_distal_anchor"
        {
            custom string drAnmar:role = "right distal fixture band"
            double3 xformOp:translate = (0.073, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Xform "lumen_axis"
        {
            custom string drAnmar:role = "shared lumen axis"
            quatf xformOp:orient = (0.70710678, 0, 0.70710678, 0)
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
        }'''
    fixtures = f'''    def Xform "LeftFixtureAnchor" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {{
        bool physics:rigidBodyEnabled = true
        bool physics:kinematicEnabled = true
        double3 xformOp:translate = (-0.073, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Geometry" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding = <{rp}/Looks/RailMetal>
            rel material:binding:physics = <{rp}/PhysicsMaterials/MountPhysics>
            double size = 1
            float3 xformOp:scale = (0.010, 0.032, 0.032)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }}
    }}
    def Xform "RightFixtureAnchor" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {{
        bool physics:rigidBodyEnabled = true
        bool physics:kinematicEnabled = true
        double3 xformOp:translate = (0.073, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Geometry" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
        )
        {{
            rel material:binding = <{rp}/Looks/RailMetal>
            rel material:binding:physics = <{rp}/PhysicsMaterials/MountPhysics>
            double size = 1
            float3 xformOp:scale = (0.010, 0.032, 0.032)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }}
    }}'''
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Two independent hollow tissue conduits with open lumens and a real central separation for capture, alignment, eversion, stapling, reinforcement, patency and leak-test research."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend variantSets = "state"
    variants = {{ string state = "initial" }}
    customData = {{
        string drAnmarAssetVersion = "{VERSION}"
        bool drAnmarClinicalValidation = false
        string drAnmarRepresentation = "two_watertight_hollow_wall_segments_with_open_lumens"
        float drAnmarOuterDiameterM = 0.024
        float drAnmarWallThicknessM = 0.0024
        float drAnmarInitialGapM = 0.020
    }}
)
{{
{visual_materials_scope(root)}
{physics_materials_scope()}
    def Xform "LeftTissue"
    {{
        double3 xformOp:translate = (-0.0425, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
{left_mesh}
    }}
    def Xform "RightTissue"
    {{
        double3 xformOp:translate = (0.0425, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
{right_mesh}
    }}
{fixtures}
    def Scope "Frames"
    {{
{frames}
    }}
    variantSet "state" = {{
        "initial"
        {{
{initial_left}
{initial_right}
        }}
        "aligned"
        {{
{aligned_left}
{aligned_right}
        }}
        "completed"
        {{
{closed_left}
{closed_right}
        }}
    }}
}}
'''


def droplet_usda() -> str:
    root=DROPLET_ROOT;rp=f"/{root}"
    mesh=ellipsoid_mesh((0.0011,0.0011,0.0015),(0,0,0),subdivisions=2)
    return f'''#usda 1.0
(
    defaultPrim = "{root}"
    doc = "Visual and rigid proxy droplet for anastomosis leak-test medium."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]
)
{{
    bool physics:rigidBodyEnabled = true
    float physics:mass = 5e-6
    bool physxRigidBody:enableCCD = true
{visual_materials_scope(root)}
{physics_materials_scope()}
{mesh_usda(Visual('Visual',mesh,'LeakFluid',('leak_test_droplet',)),f'{rp}/Looks/LeakFluid',indent='    ')}
    def Scope "Collisions"
    {{
{collider_usda(Collider('Collision','sphere',(0,0,0),radius=0.0011,physics_material='PolymerPhysics',role='leak_droplet_collision'),rp,indent='        ')}
    }}
}}
'''


def export_scene(path: Path, entries: Sequence[tuple[str,trimesh.Trimesh,str]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    scene=trimesh.Scene()
    for name,mesh,material in entries:
        scene.add_geometry(pbr(mesh,material),node_name=name,geom_name=name)
    path.write_bytes(scene.export(file_type="glb"))


PHASE_PARAMETERS={
    "inspect":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":-0.055,"mandrel_expansion_joint":0.0,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "capture":{"left_approximation_joint":0.004,"right_approximation_joint":-0.004,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":-0.025,"mandrel_expansion_joint":0.0,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "align":{"left_approximation_joint":0.012,"right_approximation_joint":-0.012,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "approximate":{"left_approximation_joint":0.030,"right_approximation_joint":-0.030,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "evert":{"left_approximation_joint":0.030,"right_approximation_joint":-0.030,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.006,"right_eversion_joint":-0.006,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "staple":{"left_approximation_joint":0.030,"right_approximation_joint":-0.030,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.006,"right_eversion_joint":-0.006,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":-0.027,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "reinforce":{"left_approximation_joint":0.016,"right_approximation_joint":-0.016,"left_capture_joint":0.002,"right_capture_joint":-0.002,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":120.0,"collar_applicator_joint":0.046,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "leak_test":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":120.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.008,"right_occluder_valve_joint":0.008,"pressure_valve_joint":0.008},
    "complete":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":-0.055,"mandrel_expansion_joint":0.0,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
}


def phase_parameters(phase: str) -> dict[str,float]:
    try:return dict(PHASE_PARAMETERS[phase])
    except KeyError as exc:raise KeyError(phase) from exc


def link_world_transform(bundle: ToolBundle, link_name: str, phase: str) -> np.ndarray:
    link=bundle.links[link_name]
    p=phase_parameters(phase)
    t=np.asarray(link.translation,dtype=float)
    R=np.eye(3)
    if link_name=="LeftCarriage":t[0]+=p["left_approximation_joint"]
    elif link_name=="RightCarriage":t[0]+=p["right_approximation_joint"]
    elif link_name=="LeftCaptureSleeve":t[0]+=p["left_approximation_joint"]+p["left_capture_joint"]
    elif link_name=="RightCaptureSleeve":t[0]+=p["right_approximation_joint"]+p["right_capture_joint"]
    elif link_name=="LeftEversionRing":t[0]+=p["left_eversion_joint"]
    elif link_name=="RightEversionRing":t[0]+=p["right_eversion_joint"]
    elif link_name=="Mandrel":t[0]+=p["mandrel_extension_joint"]
    elif link_name=="MandrelExpander":t[2]+=p["mandrel_expansion_joint"]
    elif link_name=="StapleDriver":t[0]+=p["staple_driver_joint"]
    elif link_name=="CollarCarousel":R=rotation_matrix((0,0,1),math.radians(p["collar_carousel_joint"]))
    elif link_name=="CollarApplicator":t[2]+=p["collar_applicator_joint"]
    elif link_name=="LeftOccluderValve":t[2]+=p["left_occluder_valve_joint"]
    elif link_name=="RightOccluderValve":t[2]+=p["right_occluder_valve_joint"]
    elif link_name=="PressureValve":t[2]+=p["pressure_valve_joint"]
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
    return T


def world_visual_entries(bundle: ToolBundle, phase: str="inspect") -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[]
    for link_name,link in bundle.links.items():
        T=link_world_transform(bundle,link_name,phase)
        for visual in link.visuals:
            mesh=visual.mesh.copy();mesh.apply_transform(T)
            entries.append((f"{link_name}_{visual.name}",mesh,visual.material))
    return entries


def collider_mesh(c: Collider) -> trimesh.Trimesh:
    if c.kind=="box":mesh=box_mesh(c.size or (0.01,0.01,0.01),c.center)
    elif c.kind=="cylinder":mesh=cylinder_axis(c.radius or 0.005,c.height or 0.01,c.axis,c.center)
    elif c.kind=="sphere":mesh=transform(trimesh.creation.icosphere(subdivisions=2,radius=c.radius or 0.005),c.center)
    elif c.kind=="capsule":
        axis=np.asarray({"x":(1,0,0),"y":(0,1,0),"z":(0,0,1)}[c.axis],dtype=float)
        half=(c.height or 0.01)/2
        mesh=capsule_between(np.asarray(c.center)-axis*half,np.asarray(c.center)+axis*half,c.radius or 0.005)
    else:raise ValueError(c.kind)
    R=np.asarray([[1,0,0],[0,1,0],[0,0,1]],dtype=float)
    # Mesh primitives above are already axis-aligned; apply explicit orientation around the collider center.
    q=c.orientation_wxyz
    w,x,y,z=q
    R=np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    mesh.apply_translation(-np.asarray(c.center));mesh.apply_transform(np.block([[R,np.zeros((3,1))],[np.zeros((1,3)),np.ones((1,1))]]));mesh.apply_translation(np.asarray(c.center))
    return mesh


def collision_debug_entries(bundle: ToolBundle, phase: str="approximate") -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=world_visual_entries(bundle,phase)
    for link_name,link in bundle.links.items():
        T=link_world_transform(bundle,link_name,phase)
        for c in link.colliders:
            mesh=collider_mesh(c);mesh.apply_transform(T)
            entries.append((f"Collider_{link_name}_{c.name}",mesh,"CollisionDebug"))
    return entries


def axis_entries(bundle: ToolBundle, phase: str="inspect", length: float=0.014, radius: float=0.00045) -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=world_visual_entries(bundle,phase)
    for name,data in bundle.frames.items():
        T=link_world_transform(bundle,str(data["parent_link"]),phase)
        p=np.asarray(data["position"],dtype=float)
        ph=np.ones(4);ph[:3]=p;origin=(T@ph)[:3]
        # Frame orientation is intentionally visualized in parent-link axes for easy local inspection.
        entries += [
            (f"{name}_x",capsule_between(origin,origin+np.asarray((length,0,0)),radius),"GuideRed"),
            (f"{name}_y",capsule_between(origin,origin+np.asarray((0,length,0)),radius),"GuideGreen"),
            (f"{name}_z",capsule_between(origin,origin+np.asarray((0,0,length)),radius),"GuideBlue"),
        ]
    return entries


def tissue_entries(bundle: ToolBundle, state: str="initial", completed: bool=False) -> list[tuple[str,trimesh.Trimesh,str]]:
    shifts={"initial":0.0,"aligned":0.010,"completed":0.0105}
    s=shifts[state]
    left=transform(bundle.tissue_left,(s,0,WORK_AXIS_Z))
    right=transform(bundle.tissue_right,(-s,0,WORK_AXIS_Z))
    entries=[("LeftTissue",left,"TissueOuter"),("RightTissue",right,"TissueOuter")]
    if completed:
        for i in range(STAPLE_COUNT):
            a=2*math.pi*i/STAPLE_COUNT
            staple=_staple_at_angle(bundle.formed_staple,a,0.0112)
            staple.apply_translation((0,0,WORK_AXIS_Z))
            entries.append((f"FormedStaple_{i:02d}",staple,"StapleMetal"))
        collar=transform(bundle.collar,(0,0,WORK_AXIS_Z))
        entries.append(("ReinforcementCollar",collar,"CollarMaterial"))
    return entries


def franka_proxy_entries(bundle: ToolBundle, phase: str="inspect") -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[]
    # Provider-neutral Franka visualization for inspection of the DrAnmar payload.
    entries.append(("Base",cylinder_axis(0.105,0.075,"z",(0,0,-0.02),sections=64),"DarkPolymer"))
    joints=[(0,0,0.03),(0.0,0.0,0.20),(0.12,0.0,0.34),(0.05,0.0,0.50),(0.0,0.0,0.62)]
    for i,p in enumerate(joints):entries.append((f"ArmJoint_{i}",ellipsoid_mesh((0.055,0.055,0.050),p,2),"BodyPolymer"))
    for i,(a,b) in enumerate(zip(joints[:-1],joints[1:])):entries.append((f"ArmLink_{i}",capsule_between(a,b,0.035),"BodyPolymer"))
    tool_entries=world_visual_entries(bundle,phase)
    # Shift tool to the top of the proxy arm.
    for name,mesh,material in tool_entries:
        entries.append((name,transform(mesh,(0,0,0.62)),material))
    return entries


def exploded_entries(bundle: ToolBundle) -> list[tuple[str,trimesh.Trimesh,str]]:
    entries=[]
    for index,(link_name,link) in enumerate(bundle.links.items()):
        offset=np.asarray(((index%4-1.5)*0.055,((index//4)%4-1.5)*0.050,(index//16)*0.075))
        for visual in link.visuals:
            entries.append((f"{link_name}_{visual.name}",transform(visual.mesh,np.asarray(link.translation)+offset),visual.material))
    return entries


def export_glbs(bundle: ToolBundle) -> list[Path]:
    outputs=[]
    phases=["inspect","capture","align","approximate","evert","staple","reinforce","leak_test","complete"]
    for phase in phases:
        path=GLB_ROOT/f"dranmar_anastomosis_tool_{phase}.glb";export_scene(path,world_visual_entries(bundle,phase));outputs.append(path)
    for name,entries in (
        ("dranmar_anastomosis_tool_exploded.glb",exploded_entries(bundle)),
        ("dranmar_anastomosis_tool_collision_debug.glb",collision_debug_entries(bundle,"approximate")),
        ("dranmar_anastomosis_tool_frame_debug.glb",axis_entries(bundle,"inspect")),
        ("dranmar_franka_anastomosis_assembly.glb",franka_proxy_entries(bundle,"inspect")),
        ("dranmar_hollow_tissue_initial.glb",tissue_entries(bundle,"initial")),
        ("dranmar_hollow_tissue_aligned.glb",tissue_entries(bundle,"aligned")),
        ("dranmar_hollow_tissue_completed.glb",tissue_entries(bundle,"completed",True)),
        ("dranmar_anastomosis_staple_open.glb",[("OpenStaple",bundle.open_staple,"StapleMetal")]),
        ("dranmar_anastomosis_staple_formed.glb",[("FormedStaple",bundle.formed_staple,"StapleMetal")]),
        ("dranmar_reinforcement_collar.glb",[("Collar",bundle.collar,"CollarMaterial")]),
    ):
        path=GLB_ROOT/name;export_scene(path,entries);outputs.append(path)
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
    ax.set_xlim(-0.115,0.115);ax.set_ylim(-0.105,0.105);ax.set_zlim(0.03,0.26)
    ax.view_init(elev=elev,azim=azim);ax.set_axis_off();ax.set_box_aspect((1,0.95,1.05))


def make_preview(bundle: ToolBundle) -> Path:
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(16,10),dpi=165)
    phases=[("capture","1  Circumferential capture"),("align","2  Lumen alignment"),("approximate","3  Axial approximation"),("staple","4  Staple crown deployment"),("reinforce","5  Reinforcement collar"),("leak_test","6  Patency + leak test")]
    for i,(phase,title) in enumerate(phases,1):
        ax=fig.add_subplot(2,3,i,projection="3d")
        for _,m,mat in world_visual_entries(bundle,phase):add_mesh_to_axis(ax,m,mat,850)
        state="initial" if phase=="capture" else ("aligned" if phase in {"align","approximate"} else "completed")
        for _,m,mat in tissue_entries(bundle,state,completed=phase in {"reinforce","leak_test"}):add_mesh_to_axis(ax,m,mat,900)
        configure_axis(ax,title)
    fig.suptitle("DrAnmar Adaptive Anastomosis Robot — capture, align, evert, staple, reinforce, verify",fontsize=17,y=0.98)
    fig.text(0.5,0.015,"DrAnmar-owned, provider-neutral research system • NVIDIA Isaac/Franka integration • provisional physical parameters",ha="center",fontsize=10)
    path=PREVIEW_ROOT/"dranmar_adaptive_anastomosis_robot_preview.png";fig.savefig(path,bbox_inches="tight",facecolor="white");plt.close(fig);return path


def make_full_arm_preview(bundle: ToolBundle) -> Path:
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(10,10),dpi=175);ax=fig.add_subplot(111,projection="3d")
    for _,m,mat in franka_proxy_entries(bundle,"inspect"):add_mesh_to_axis(ax,m,mat,1000)
    ax.set_xlim(-0.30,0.35);ax.set_ylim(-0.28,0.28);ax.set_zlim(-0.08,0.95);ax.view_init(elev=18,azim=-58);ax.set_axis_off();ax.set_box_aspect((0.72,0.62,1.25))
    ax.set_title("DrAnmar Adaptive Anastomosis Robot mounted at the Franka wrist",fontsize=15,pad=12)
    path=PREVIEW_ROOT/"dranmar_adaptive_anastomosis_robot_full_arm_preview.png";fig.savefig(path,bbox_inches="tight",facecolor="white");plt.close(fig);return path


def noise_texture(base: tuple[int,int,int], size: int=512, strength: int=18, seed: int=1) -> Image.Image:
    rng=np.random.default_rng(seed);arr=np.zeros((size,size,3),dtype=np.int16);arr[:]=np.asarray(base,dtype=np.int16);arr+=rng.normal(0,strength,(size,size,1)).astype(np.int16);arr=np.clip(arr,0,255).astype(np.uint8);return Image.fromarray(arr,"RGB")


def generate_textures() -> list[Path]:
    TEXTURE_ROOT.mkdir(parents=True,exist_ok=True);out=[]
    for name,base,strength,seed in [
        ("tissue_basecolor.png",(166,64,54),14,31),
        ("reinforcement_collar_basecolor.png",(226,194,112),18,32),
        ("polymer_microtexture.png",(216,222,228),9,33),
        ("metal_microtexture.png",(154,159,165),7,34),
        ("test_medium_basecolor.png",(52,150,230),10,35),
    ]:
        p=TEXTURE_ROOT/name;noise_texture(base,512,strength,seed).save(p);out.append(p)
    img=Image.new("RGB",(1024,256),(247,249,251));d=ImageDraw.Draw(img)
    try:font=ImageFont.truetype("DejaVuSans-Bold.ttf",72);small=ImageFont.truetype("DejaVuSans.ttf",29)
    except OSError:font=None;small=None
    d.text((36,46),"DrAnmar",fill=(18,65,112),font=font)
    d.text((40,150),"ADAPTIVE ANASTOMOSIS • TRAINING WORKCELL",fill=(35,45,55),font=small)
    p=TEXTURE_ROOT/"label_dranmar.png";img.save(p);out.append(p)
    return out


def interaction_frames(bundle: ToolBundle) -> dict[str,object]:
    return {"schema":"dranmar.interaction-frames.v1","asset":"dranmar-adaptive-anastomosis-robot-v1","units":"m","frames":bundle.frames}


def mount_contract() -> dict[str,object]:
    return {
        "schema":"dranmar.franka-mount.v1",
        "parent_link":"panda_link8",
        "payload_link":"DrAnmarAdaptiveAnastomosisTool/Links/Mount",
        "local_translation_m":[0,0,0],
        "local_rotation_axis_angle_deg":{"axis":[0,0,1],"angle":FRANKA_HAND_EQUIVALENT_ROTATION_DEG},
        "deactivate":["panda_hand_joint","panda_hand","panda_finger_joint1","panda_finger_joint2","panda_leftfinger","panda_rightfinger"],
        "intended_use":"simulation_training",
    }


def task_contract() -> dict[str,object]:
    return {
        "schema":"dranmar.adaptive-anastomosis-task.v1",
        "procedure":"end_to_end_hollow_tissue_anastomosis",
        "phases":["inspect","capture","align","mandrel","approximate","evert","staple","release_capture","reinforce","occlude","pressurize","verify","complete","abort"],
        "success_metrics":[
            "lumen_axis_error_deg","centerline_offset_m","minimum_lumen_radius_m","patency_area_fraction","edge_apposition_gap_m",
            "eversion_height_m","retained_staple_fraction","reinforcement_bond_fraction","pressure_decay_pa_s","residual_leak_ml_min",
            "integrated_leak_volume_ml","capture_force_n","procedure_time_s","tissue_damage_proxy"
        ],
        "failure_modes":[
            "capture_loss","torsional_misalignment","lumen_obstruction","edge_inversion","incomplete_apposition","staple_misfire","staple_pullout",
            "collar_delamination","pressure_test_leak","tissue_crush","wall_cut_through","mandrel_entrapment","instrument_collision"
        ],
        "clinical_validation":False,
    }


def physics_profile(bundle: ToolBundle) -> dict[str,object]:
    link_masses={name:link.mass_properties for name,link in bundle.links.items() if link.mass_properties}
    return {
        "schema":"dranmar.adaptive-anastomosis-profile.v1",
        "id":"dranmar-adaptive-anastomosis-robot-v1",
        "name":ASSET_NAME,
        "version":VERSION,
        "status":"simulation_training_model",
        "units":"metres-kilograms-seconds",
        "mechanism":{
            "active_joint_count":sum(1 for j in bundle.joints if j.type!="fixed"),
            "authored_joint_prim_count":len(bundle.joints),
            "active_joint_names":[j.name for j in bundle.joints if j.type!="fixed"],
            "fixed_joint_names":[j.name for j in bundle.joints if j.type=="fixed"],
            "staple_count":STAPLE_COUNT,
            "capture_cells_per_side":CAPTURE_CELL_COUNT_PER_SIDE,
            "collar_sector_count":COLLAR_SECTOR_COUNT,
            "collar_capacity":COLLAR_CAPACITY,
            "work_axis_z_m":WORK_AXIS_Z,
            "link_mass_properties":link_masses,
        },
        "tissue_demo":{
            "segment_length_m":0.065,"outer_diameter_m":0.024,"wall_thickness_m":0.0024,"initial_gap_m":0.020,
            "representation":"two_watertight_hollow_wall_surface_meshes_prepared_for_current_surface_deformable_cooking",
            "constitutive_parameters":"provisional_category_level_seeds",
            "surface_self_collision_default":False,
            "surface_self_collision_boundary":"opt_in_after_thickness_and_contact_capacity_tuning",
            "solver_mesh_axial_segments":16,
            "solver_mesh_circumferential_segments":32,
        },
        "capture":{
            "attachment_model":"six_current_vertex_to_xform_attachment_cells_per_side",
            "target_force_per_side_n":1.6,"soft_limit_n":3.5,"hard_release_n":6.0,
        },
        "staples":{
            "representation":"sixteen_independent_rigid_formed_staples_with_left_and_right_attachment_regions",
            "provisional_pullout_force_per_staple_n":1.4,
            "forming":"discrete_open_to_formed_state_pending_plasticity_backend_evidence",
        },
        "reinforcement_collar":{
            "surface_representation":"connected_triangular_circumferential_ribbon",
            "stable_bond_carrier":"rigid_proxy_with_independent_left_and_right_sector_attachments",
            "cure_time_s":45.0,"initial_sector_tack_force_n":0.18,"final_sector_break_force_n":2.2,
        },
        "leak_test":{
            "initial_reservoir_ml":TEST_RESERVOIR_ML,"nominal_challenge_pressure_pa":8000.0,"observation_window_s":8.0,
            "maximum_residual_leak_ml_min":2.0,"model":"lumped_isothermal_pressure_decay_plus_orifice_leak_proxy",
        },
        "patency":{
            "mandrel_radius_m":0.0058,"expanded_reference_radius_m":0.0105,"target_minimum_lumen_radius_m":0.0085,
        },
        "boundaries":[
            "no_clinical_force_or_pressure_claims","no_continuous_staple_plasticity","no_validated_tissue_penetration_or_damage",
            "no_biochemical_adhesive_or_healing_model","no_clinical_leak_threshold_claim","native_simulator_evidence_required",
        ],
    }


def collider_coverage(bundle: ToolBundle) -> dict[str,object]:
    rows=[]
    for name,link in bundle.links.items():
        if not link.visuals or not link.colliders:continue
        vmin,vmax=mesh_bounds([v.mesh for v in link.visuals]);visual=vmax-vmin
        collider_meshes=[collider_mesh(c) for c in link.colliders]
        cmin,cmax=mesh_bounds(collider_meshes);collider=cmax-cmin
        rows.append({"link":name,"visual_bounds_min_m":vmin.tolist(),"visual_bounds_max_m":vmax.tolist(),"collider_bounds_min_m":cmin.tolist(),"collider_bounds_max_m":cmax.tolist(),"axis_coverage_ratio":[float(collider[i]/visual[i]) if visual[i]>1e-9 else None for i in range(3)],"deliberate_insets":"thin contact and attachment volumes are role-specific and not intended to cover decorative geometry"})
    return {"schema":"dranmar.collider-coverage.v1","asset":"dranmar-adaptive-anastomosis-robot-v1","links":rows}


def author_integration_module() -> str:
    if not INTEGRATION_PATH.exists():
        raise FileNotFoundError(
            "The authoritative Isaac integration module is required for generation: "
            f"{INTEGRATION_PATH}"
        )
    return INTEGRATION_PATH.read_text(encoding="utf-8")
    # Unreachable historical template retained only for source provenance.
    return textwrap.dedent(r'''# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab integration for the DrAnmar Adaptive Anastomosis Robot.

The payload replaces the Panda hand at ``panda_link8``. Runtime helpers provide
bilateral hollow-tissue capture, coaxial approximation, surface-deformable
attachment management, circumferential retained-staple deployment,
reinforcement-collar bonding, lumen-patency metrics, and pressure-decay leak
verification. All physical values are provisional research parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import math

CATALOG_SUBPATH = "Props/SurgicalReconstruction/AdaptiveAnastomosisRobot"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
TOOL_PAYLOAD_USD = ROOT / "dranmar_adaptive_anastomosis_tool_payload.usda"
TOOL_STANDALONE_USD = ROOT / "dranmar_adaptive_anastomosis_tool_standalone.usda"
TOOL_RIGID_PROXY_USD = ROOT / "dranmar_adaptive_anastomosis_tool_rigid_proxy.usda"
TISSUE_USD = ROOT / "dranmar_hollow_tissue_demo.usda"
STAPLE_USD = ROOT / "dranmar_anastomosis_staple.usda"
COLLAR_USD = ROOT / "dranmar_reinforcement_collar.usda"
COLLAR_PROXY_USD = ROOT / "dranmar_reinforcement_collar_rigid_proxy.usda"
DROPLET_USD = ROOT / "dranmar_leak_test_droplet.usda"

VALID_BINARY_STATES = frozenset({"loaded", "empty"})
VALID_TEST_MEDIUM_STATES = frozenset({"full", "empty"})
VALID_COLLECTION_STATES = frozenset({"empty", "partial", "full"})
VALID_TISSUE_STATES = frozenset({"initial", "aligned", "completed"})
STAPLE_COUNT = 16
CAPTURE_CELL_COUNT_PER_SIDE = 6
COLLAR_SECTOR_COUNT = 16
STAPLE_RING_RADIUS_M = 0.0112

TOOL_JOINTS = {
    "left_approximation":"left_approximation_joint",
    "right_approximation":"right_approximation_joint",
    "left_capture":"left_capture_joint",
    "right_capture":"right_capture_joint",
    "left_eversion":"left_eversion_joint",
    "right_eversion":"right_eversion_joint",
    "mandrel_extension":"mandrel_extension_joint",
    "mandrel_expansion":"mandrel_expansion_joint",
    "staple_driver":"staple_driver_joint",
    "collar_carousel":"collar_carousel_joint",
    "collar_applicator":"collar_applicator_joint",
    "left_occluder_valve":"left_occluder_valve_joint",
    "right_occluder_valve":"right_occluder_valve_joint",
    "pressure_valve":"pressure_valve_joint",
}
TOOL_FRAME_PATHS = {
    "panda_link8_mount":"Links/Mount/Frames/panda_link8_mount",
    "anastomosis_tcp":"Links/Mount/Frames/anastomosis_tcp",
    "seam_reference":"Links/Mount/Frames/seam_reference",
    "lumen_axis_reference":"Links/Mount/Frames/lumen_axis_reference",
    "camera_left":"Links/Mount/Frames/camera_left",
    "camera_right":"Links/Mount/Frames/camera_right",
    "pressure_sensor":"Links/Mount/Frames/pressure_sensor",
    "leak_observation":"Links/Mount/Frames/leak_observation",
    "count_reference":"Links/Mount/Frames/count_reference",
    "disposal_reference":"Links/Mount/Frames/disposal_reference",
    "left_capture_center":"Links/LeftCaptureSleeve/Frames/left_capture_center",
    "left_tissue_edge_reference":"Links/LeftCaptureSleeve/Frames/left_tissue_edge_reference",
    "right_capture_center":"Links/RightCaptureSleeve/Frames/right_capture_center",
    "right_tissue_edge_reference":"Links/RightCaptureSleeve/Frames/right_tissue_edge_reference",
    "left_eversion_contact":"Links/LeftEversionRing/Frames/left_eversion_contact",
    "right_eversion_contact":"Links/RightEversionRing/Frames/right_eversion_contact",
    "mandrel_tip":"Links/Mandrel/Frames/mandrel_tip",
    "pressure_inlet":"Links/Mandrel/Frames/pressure_inlet",
    "patency_reference":"Links/MandrelExpander/Frames/patency_reference",
    "staple_anvil_reference":"Links/StapleAnvil/Frames/staple_anvil_reference",
    "staple_crown_reference":"Links/StapleDriver/Frames/staple_crown_reference",
    "collar_application":"Links/CollarApplicator/Frames/collar_application",
}


def frame_path(tool_path: str, name: str) -> str:
    try: suffix=TOOL_FRAME_PATHS[name]
    except KeyError as exc: raise KeyError(f"Unknown anastomosis frame {name!r}") from exc
    return f"{tool_path.rstrip('/')}/{suffix}"


def tensor_value(value: Any):
    return value.torch if hasattr(value,"torch") else value


def _check(value: str, allowed: frozenset[str], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"Unsupported {label}={value!r}; expected one of {sorted(allowed)}")
    return value


def make_tool_cfg(
    prim_path: str="/World/DrAnmarAdaptiveAnastomosisTool",
    *,
    staple_state: str="loaded",
    collar_state: str="loaded",
    test_medium_state: str="full",
    collection_state: str="empty",
    position=(0,0,0.35),
    orientation_wxyz=(1,0,0,0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    _check(staple_state,VALID_BINARY_STATES,"staple_state")
    _check(collar_state,VALID_BINARY_STATES,"collar_state")
    _check(test_medium_state,VALID_TEST_MEDIUM_STATES,"test_medium_state")
    _check(collection_state,VALID_COLLECTION_STATES,"collection_state")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TOOL_STANDALONE_USD),
            variants={"staple_state":staple_state,"collar_state":collar_state,"test_medium_state":test_medium_state,"collection_state":collection_state},
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=24,solver_velocity_iteration_count=8),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=position,rot=orientation_wxyz,joint_pos={name:0.0 for name in TOOL_JOINTS.values()}),
        actuators={
            "approximation":ImplicitActuatorCfg(joint_names_expr=[".*approximation_joint"],effort_limit_sim=180.0,velocity_limit_sim=0.16,stiffness=9000.0,damping=260.0),
            "capture_eversion":ImplicitActuatorCfg(joint_names_expr=[".*capture_joint",".*eversion_joint"],effort_limit_sim=130.0,velocity_limit_sim=0.14,stiffness=6800.0,damping=205.0),
            "mandrel":ImplicitActuatorCfg(joint_names_expr=["mandrel_.*_joint"],effort_limit_sim=95.0,velocity_limit_sim=0.20,stiffness=5200.0,damping=155.0),
            "staple":ImplicitActuatorCfg(joint_names_expr=["staple_driver_joint"],effort_limit_sim=340.0,velocity_limit_sim=0.28,stiffness=17000.0,damping=360.0),
            "collar":ImplicitActuatorCfg(joint_names_expr=["collar_.*_joint"],effort_limit_sim=125.0,velocity_limit_sim=1.4,stiffness=7000.0,damping=190.0),
            "valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=32.0,velocity_limit_sim=0.25,stiffness=1750.0,damping=55.0),
        },
    )


def make_rigid_proxy_cfg(prim_path="/World/DrAnmarAdaptiveAnastomosisProxy", *, position=(0,0,0.35), orientation_wxyz=(1,0,0,0)):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(TOOL_RIGID_PROXY_USD),activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position,rot=orientation_wxyz),
    )


def _spawn_single_franka_with_tool(prim_path: str, cfg: Any, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from isaaclab.sim.utils import create_prim, get_current_stage, select_usd_variants
    from pxr import Gf, Sdf, UsdPhysics
    robot=spawn_from_usd(prim_path,cfg,translation,orientation)
    stage=get_current_stage()
    for prim in list(stage.Traverse()):
        if prim.GetPath().HasPrefix(Sdf.Path(prim_path)) and prim.GetName() in {"panda_hand_joint","panda_hand","panda_finger_joint1","panda_finger_joint2","panda_leftfinger","panda_rightfinger"}:
            stage.OverridePrim(prim.GetPath()).SetActive(False)
    tool_path=f"{prim_path}/DrAnmarAdaptiveAnastomosisTool"
    create_prim(tool_path,usd_path=str(TOOL_PAYLOAD_USD),stage=stage)
    select_usd_variants(tool_path,{"staple_state":cfg.staple_state,"collar_state":cfg.collar_state,"test_medium_state":cfg.test_medium_state,"collection_state":cfg.collection_state})
    joint=UsdPhysics.FixedJoint.Define(stage,f"{prim_path}/dranmar_anastomosis_mount_joint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{prim_path}/panda_link8")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{tool_path}/Links/Mount")])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0,0,0));joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
    a=math.radians(-45.0)/2
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(math.cos(a),0,0,math.sin(a)))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))
    return robot


def spawn_franka_with_tool(prim_path: str,cfg: Any,translation=None,orientation=None,**kwargs):
    from isaaclab.sim.utils import clone
    return clone(_spawn_single_franka_with_tool)(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)


def make_franka_adaptive_anastomosis_robot_cfg(
    *,
    prim_path="/World/Robot",
    staple_state="loaded",
    collar_state="loaded",
    test_medium_state="full",
    collection_state="empty",
):
    _check(staple_state,VALID_BINARY_STATES,"staple_state")
    _check(collar_state,VALID_BINARY_STATES,"collar_state")
    _check(test_medium_state,VALID_TEST_MEDIUM_STATES,"test_medium_state")
    _check(collection_state,VALID_COLLECTION_STATES,"collection_state")
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    @configclass
    class FrankaAnastomosisUsdCfg(sim_utils.UsdFileCfg):
        staple_state: str="loaded"
        collar_state: str="loaded"
        test_medium_state: str="full"
        collection_state: str="empty"
        func=spawn_franka_with_tool
    cfg=FRANKA_PANDA_CFG.copy();cfg.prim_path=prim_path
    cfg.spawn=FrankaAnastomosisUsdCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        variants={"Gripper":"Default","Mesh":"Performance"},
        staple_state=staple_state,collar_state=collar_state,test_medium_state=test_medium_state,collection_state=collection_state,
        activate_contact_sensors=True,rigid_props=FRANKA_PANDA_CFG.spawn.rigid_props,articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props,
    )
    cfg.init_state.joint_pos={k:v for k,v in cfg.init_state.joint_pos.items() if "finger" not in k}
    cfg.init_state.joint_pos.update({name:0.0 for name in TOOL_JOINTS.values()})
    cfg.actuators={k:v for k,v in cfg.actuators.items() if k!="panda_hand"}
    cfg.actuators.update({
        "anastomosis_approximation":ImplicitActuatorCfg(joint_names_expr=[".*approximation_joint"],effort_limit_sim=180.0,velocity_limit_sim=0.16,stiffness=9000.0,damping=260.0),
        "anastomosis_capture_eversion":ImplicitActuatorCfg(joint_names_expr=[".*capture_joint",".*eversion_joint"],effort_limit_sim=130.0,velocity_limit_sim=0.14,stiffness=6800.0,damping=205.0),
        "anastomosis_mandrel":ImplicitActuatorCfg(joint_names_expr=["mandrel_.*_joint"],effort_limit_sim=95.0,velocity_limit_sim=0.20,stiffness=5200.0,damping=155.0),
        "anastomosis_staple":ImplicitActuatorCfg(joint_names_expr=["staple_driver_joint"],effort_limit_sim=340.0,velocity_limit_sim=0.28,stiffness=17000.0,damping=360.0),
        "anastomosis_collar":ImplicitActuatorCfg(joint_names_expr=["collar_.*_joint"],effort_limit_sim=125.0,velocity_limit_sim=1.4,stiffness=7000.0,damping=190.0),
        "anastomosis_valves":ImplicitActuatorCfg(joint_names_expr=[".*_valve_joint"],effort_limit_sim=32.0,velocity_limit_sim=0.25,stiffness=1750.0,damping=55.0),
    })
    return cfg


def _current_stage(stage=None):
    if stage is not None:return stage
    import omni.usd
    return omni.usd.get_context().get_stage()


def spawn_hollow_tissue_demo(prim_path="/World/DrAnmarHollowTissue", *, state="initial", translation=(0,0,0), orientation_wxyz=(1,0,0,0)):
    _check(state,VALID_TISSUE_STATES,"state")
    import isaaclab.sim as sim_utils
    cfg=sim_utils.UsdFileCfg(usd_path=str(TISSUE_USD),variants={"state":state})
    return cfg.func(prim_path,cfg,translation=translation,orientation=orientation_wxyz)


def _create_surface_material(stage, material_path: str):
    from pxr import UsdShade
    material=UsdShade.Material.Define(stage,material_path);prim=material.GetPrim()
    for schema in ("OmniPhysicsBaseMaterialAPI","OmniPhysicsDeformableMaterialAPI","OmniPhysicsSurfaceDeformableMaterialAPI","PhysxSurfaceDeformableMaterialAPI"):
        try:prim.ApplyAPI(schema)
        except Exception:pass
    for name,value in {
        "omniphysics:dynamicFriction":0.38,"omniphysics:density":1060.0,"omniphysics:youngsModulus":180000.0,
        "omniphysics:poissonsRatio":0.47,"omniphysics:surfaceThickness":0.0024,"omniphysics:surfaceBendStiffness":0.0,
        "physxDeformableMaterial:elasticityDamping":0.16,"physxDeformableMaterial:bendDamping":0.18,
    }.items():
        attr=prim.GetAttribute(name)
        if attr:attr.Set(value)
    return material


def apply_hollow_tissue_surface_deformables(root_path: str, *, self_collision=True, stage=None):
    from omni.physx.scripts import deformableUtils
    from pxr import UsdShade
    stage=_current_stage(stage);out={}
    material=_create_surface_material(stage,f"{root_path}/RuntimeMaterials/TissueSurface")
    for side in ("LeftTissue","RightTissue"):
        mesh_path=f"{root_path}/{side}/SimulationMesh";mesh=stage.GetPrimAtPath(mesh_path)
        if not mesh or not mesh.IsValid():raise ValueError(f"Missing {mesh_path}")
        result=deformableUtils.set_physics_surface_deformable_body(stage,mesh.GetPath())
        if result is False:raise RuntimeError(f"Failed to cook {mesh_path}")
        try:mesh.ApplyAPI("PhysxSurfaceDeformableBodyAPI");mesh.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
        except Exception:pass
        binding=UsdShade.MaterialBindingAPI.Apply(mesh);binding.Bind(material,UsdShade.Tokens.weakerThanDescendants,"physics")
        out[side]=mesh_path
    return out


def create_deformable_attachment(deformable_path: str,target_path: str,attachment_path: str,*,stage=None):
    stage=_current_stage(stage)
    import omni.kit.commands
    try:
        ok=omni.kit.commands.execute("CreateAutoDeformableAttachment",target_attachment_path=attachment_path,attachable0_path=deformable_path,attachable1_path=target_path)
    except Exception:
        ok=omni.kit.commands.execute("CreatePhysicsAttachment",target_attachment_path=attachment_path,actor0_path=deformable_path,actor1_path=target_path)
    return ok


def remove_prims(paths: Iterable[str],*,stage=None):
    stage=_current_stage(stage)
    for path in paths:
        if stage.GetPrimAtPath(path).IsValid():stage.RemovePrim(path)


@dataclass
class SideCapture:
    side: str
    tissue_path: str
    attachment_paths: list[str]=field(default_factory=list)
    engaged: bool=False


@dataclass
class BilateralTissueCaptureController:
    tool_path: str
    left_tissue_path: str
    right_tissue_path: str
    cells_per_side: int=CAPTURE_CELL_COUNT_PER_SIDE
    left: SideCapture=field(init=False)
    right: SideCapture=field(init=False)
    def __post_init__(self):
        self.left=SideCapture("left",self.left_tissue_path);self.right=SideCapture("right",self.right_tissue_path)
    def engage_side(self,capture: SideCapture,*,stage=None):
        if capture.engaged:return list(capture.attachment_paths)
        stage=_current_stage(stage);stage.DefinePrim(f"{self.tool_path}/RuntimeAttachments","Scope");created=[]
        link="LeftCaptureSleeve" if capture.side=="left" else "RightCaptureSleeve"
        try:
            for i in range(self.cells_per_side):
                ap=f"{self.tool_path}/RuntimeAttachments/{capture.side}_capture_{i:02d}"
                target=f"{self.tool_path}/Links/{link}/Collisions/CaptureCell_{i:02d}"
                create_deformable_attachment(capture.tissue_path,target,ap,stage=stage);created.append(ap)
        except Exception:
            remove_prims(created,stage=stage);raise
        capture.attachment_paths=created;capture.engaged=True;return list(created)
    def engage(self,*,stage=None):
        return {"left":self.engage_side(self.left,stage=stage),"right":self.engage_side(self.right,stage=stage)}
    def release_side(self,capture: SideCapture,*,stage=None):
        remove_prims(capture.attachment_paths,stage=stage);capture.attachment_paths.clear();capture.engaged=False
    def release(self,*,stage=None):self.release_side(self.left,stage=stage);self.release_side(self.right,stage=stage)


def _spawn_reference_at_transform(stage,prim_path: str,usd_path: Path,world_transform: Any,variants: dict[str,str]|None=None):
    import omni.kit.commands
    from pxr import Sdf
    prim=stage.DefinePrim(prim_path,"Xform");prim.GetReferences().AddReference(str(usd_path))
    omni.kit.commands.execute("TransformPrim",path=Sdf.Path(prim_path),new_transform_matrix=world_transform)
    if variants:
        for name,value in variants.items():prim.GetVariantSets().GetVariantSet(name).SetVariantSelection(value)
    return prim


def _ring_local_matrix(angle_rad: float,radius_m: float):
    from pxr import Gf
    rotation=Gf.Matrix4d(1.0);rotation.SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0),math.degrees(angle_rad)))
    translation=Gf.Matrix4d(1.0);translation.SetTranslate(Gf.Vec3d(0,radius_m,0))
    return translation*rotation


def deploy_staple_ring(
    parent_path: str,
    crown_world_transform: Any,
    left_tissue_path: str,
    right_tissue_path: str,
    *,
    staple_count: int=STAPLE_COUNT,
    radius_m: float=STAPLE_RING_RADIUS_M,
    stage=None,
):
    stage=_current_stage(stage);stage.DefinePrim(parent_path,"Scope");deployments=[]
    try:
        for i in range(staple_count):
            angle=2*math.pi*i/staple_count
            path=f"{parent_path}/Staple_{i:02d}"
            world=_ring_local_matrix(angle,radius_m)*crown_world_transform
            _spawn_reference_at_transform(stage,path,STAPLE_USD,world,{"state":"formed"})
            stage.DefinePrim(f"{path}/Attachments","Scope")
            left_ap=f"{path}/Attachments/left";right_ap=f"{path}/Attachments/right"
            create_deformable_attachment(left_tissue_path,f"{path}/Collisions/LeftLegAttachment",left_ap,stage=stage)
            create_deformable_attachment(right_tissue_path,f"{path}/Collisions/RightLegAttachment",right_ap,stage=stage)
            deployments.append({"staple_path":path,"angle_rad":angle,"attachment_paths":[left_ap,right_ap],"retained":True})
    except Exception:
        remove_prims([d["staple_path"] for d in deployments],stage=stage);raise
    return deployments


@dataclass
class StapleRingRetentionController:
    pullout_force_per_staple_n: float=1.4
    deployments: list[dict[str,Any]]=field(default_factory=list)
    def register(self,deployments):self.deployments.extend(deployments);return deployments
    @property
    def retained_fraction(self):
        return 0.0 if not self.deployments else sum(bool(d.get("retained",False)) for d in self.deployments)/len(self.deployments)
    def apply_loads(self,loads_n: Sequence[float],*,stage=None):
        released=[]
        for deployment,load in zip(self.deployments,loads_n):
            if deployment.get("retained",False) and abs(float(load))>self.pullout_force_per_staple_n:
                remove_prims(deployment["attachment_paths"],stage=stage);deployment["retained"]=False;released.append(deployment["staple_path"])
        return released


@dataclass
class CollarBond:
    collar_path: str
    attachment_paths: list[str]
    cure_fraction: float=0.0
    broken_sectors: set[int]=field(default_factory=set)


@dataclass
class ReinforcementCollarBondController:
    cure_time_s: float=45.0
    final_sector_break_force_n: float=2.2
    bonds: list[CollarBond]=field(default_factory=list)
    def deploy(self,prim_path: str,world_transform: Any,left_tissue_path: str,right_tissue_path: str,*,stage=None):
        stage=_current_stage(stage);_spawn_reference_at_transform(stage,prim_path,COLLAR_PROXY_USD,world_transform,{"state":"fresh"});stage.DefinePrim(f"{prim_path}/Attachments","Scope");created=[]
        try:
            for i in range(COLLAR_SECTOR_COUNT):
                for side,tissue in (("Left",left_tissue_path),("Right",right_tissue_path)):
                    ap=f"{prim_path}/Attachments/{side.lower()}_{i:02d}"
                    create_deformable_attachment(tissue,f"{prim_path}/Collisions/{side}BondCell_{i:02d}",ap,stage=stage);created.append(ap)
        except Exception:
            remove_prims(created+[prim_path],stage=stage);raise
        bond=CollarBond(prim_path,created);self.bonds.append(bond);return bond
    def update(self,dt: float):
        for bond in self.bonds:bond.cure_fraction=min(1.0,bond.cure_fraction+max(0.0,float(dt))/max(self.cure_time_s,1e-9))
    def apply_sector_load(self,bond: CollarBond,sector: int,load_n: float,*,stage=None):
        if sector in bond.broken_sectors:return False
        threshold=max(0.15,self.final_sector_break_force_n*(0.08+0.92*bond.cure_fraction))
        if abs(float(load_n))<=threshold:return False
        paths=[f"{bond.collar_path}/Attachments/left_{sector:02d}",f"{bond.collar_path}/Attachments/right_{sector:02d}"]
        remove_prims(paths,stage=stage);bond.broken_sectors.add(int(sector));return True
    def bonded_fraction(self,bond: CollarBond):return max(0.0,1.0-len(bond.broken_sectors)/COLLAR_SECTOR_COUNT)


@dataclass
class PatencyReport:
    minimum_radius_m: float
    mean_radius_m: float
    area_fraction: float
    centerline_offset_m: float
    axis_error_deg: float
    passed: bool


@dataclass
class LumenPatencyController:
    reference_radius_m: float=0.0095
    minimum_accepted_radius_m: float=0.0085
    maximum_centerline_offset_m: float=0.0025
    maximum_axis_error_deg: float=7.0
    def evaluate(self,radial_samples_m: Sequence[float],*,centerline_offset_m: float=0.0,axis_error_deg: float=0.0):
        values=[max(0.0,float(x)) for x in radial_samples_m]
        if not values:raise ValueError("radial_samples_m must not be empty")
        minimum=min(values);mean=sum(values)/len(values);area_fraction=(minimum/max(self.reference_radius_m,1e-9))**2
        passed=minimum>=self.minimum_accepted_radius_m and abs(centerline_offset_m)<=self.maximum_centerline_offset_m and abs(axis_error_deg)<=self.maximum_axis_error_deg
        return PatencyReport(minimum,mean,area_fraction,float(centerline_offset_m),float(axis_error_deg),passed)


@dataclass
class LeakTestLedger:
    initial_reservoir_ml: float=60.0
    reservoir_ml: float=60.0
    injected_ml: float=0.0
    leaked_ml: float=0.0
    collected_ml: float=0.0
    discarded_ml: float=0.0
    def inject(self,volume_ml: float):
        v=max(0.0,min(float(volume_ml),self.reservoir_ml));self.reservoir_ml-=v;self.injected_ml+=v;return v
    def leak(self,volume_ml: float):
        v=max(0.0,float(volume_ml));self.leaked_ml+=v;return v
    def collect(self,volume_ml: float):
        v=max(0.0,min(float(volume_ml),self.leaked_ml-self.collected_ml));self.collected_ml+=v;return v
    @property
    def conservation_error_ml(self):return self.initial_reservoir_ml-(self.reservoir_ml+self.injected_ml)


@dataclass
class PressureDecayLeakController:
    target_pressure_pa: float=8000.0
    chamber_compliance_m3_pa: float=1.8e-11
    fluid_density_kg_m3: float=1000.0
    discharge_coefficient: float=0.62
    observation_window_s: float=8.0
    maximum_residual_leak_ml_min: float=2.0
    pressure_pa: float=0.0
    elapsed_s: float=0.0
    integrated_leak_ml: float=0.0
    peak_leak_ml_min: float=0.0
    history: list[dict[str,float]]=field(default_factory=list)
    def reset(self):self.pressure_pa=0.0;self.elapsed_s=0.0;self.integrated_leak_ml=0.0;self.peak_leak_ml_min=0.0;self.history.clear()
    def effective_leak_area_m2(self,*,edge_gap_m: float,retained_staple_fraction: float,collar_bond_fraction: float):
        circumference=2*math.pi*0.0108
        gap=max(0.0,float(edge_gap_m))
        staple_scale=max(0.0,1.0-max(0.0,min(1.0,float(retained_staple_fraction))))
        collar_scale=max(0.04,1.0-0.94*max(0.0,min(1.0,float(collar_bond_fraction))))
        return max(2.0e-10,circumference*gap*(0.06+0.94*staple_scale)*collar_scale)
    def update(self,dt: float,*,pump_flow_ml_s: float=0.0,edge_gap_m: float=0.0,retained_staple_fraction: float=1.0,collar_bond_fraction: float=1.0):
        dt=max(0.0,float(dt));area=self.effective_leak_area_m2(edge_gap_m=edge_gap_m,retained_staple_fraction=retained_staple_fraction,collar_bond_fraction=collar_bond_fraction)
        q_out_m3_s=self.discharge_coefficient*area*math.sqrt(max(0.0,2.0*self.pressure_pa/self.fluid_density_kg_m3))
        q_in_m3_s=max(0.0,float(pump_flow_ml_s))*1.0e-6
        self.pressure_pa=max(0.0,self.pressure_pa+(q_in_m3_s-q_out_m3_s)*dt/max(self.chamber_compliance_m3_pa,1e-15))
        leak_ml_min=q_out_m3_s*1.0e6*60.0;leak_ml=q_out_m3_s*1.0e6*dt
        self.elapsed_s+=dt;self.integrated_leak_ml+=leak_ml;self.peak_leak_ml_min=max(self.peak_leak_ml_min,leak_ml_min)
        sample={"time_s":self.elapsed_s,"pressure_pa":self.pressure_pa,"leak_ml_min":leak_ml_min,"effective_leak_area_m2":area};self.history.append(sample);return sample
    @property
    def average_leak_ml_min(self):return 0.0 if self.elapsed_s<=0 else self.integrated_leak_ml*60.0/self.elapsed_s
    @property
    def complete(self):return self.elapsed_s>=self.observation_window_s
    @property
    def passed(self):return self.complete and self.average_leak_ml_min<=self.maximum_residual_leak_ml_min and self.pressure_pa>=0.70*self.target_pressure_pa


def ensure_leak_particle_system(*,system_path="/World/DrAnmarLeakTest/ParticleSystem",material_path="/World/DrAnmarLeakTest/PBDMaterial",stage=None):
    stage=_current_stage(stage)
    from omni.physx.scripts import particleUtils,physicsUtils
    from pxr import Sdf,UsdPhysics
    scene_path=Sdf.Path("/World/physicsScene")
    if not stage.GetPrimAtPath(scene_path).IsValid():UsdPhysics.Scene.Define(stage,scene_path)
    if not stage.GetPrimAtPath(material_path).IsValid():particleUtils.add_pbd_particle_material(stage,Sdf.Path(material_path),friction=0.08,damping=0.02,viscosity=1.0)
    if not stage.GetPrimAtPath(system_path).IsValid():particleUtils.add_physx_particle_system(stage=stage,particle_system_path=Sdf.Path(system_path),simulation_owner=scene_path)
    physicsUtils.add_physics_material_to_prim(stage,stage.GetPrimAtPath(system_path),Sdf.Path(material_path));return system_path


def emit_leak_particles(positions: Sequence[Sequence[float]],velocities: Sequence[Sequence[float]],*,system_path="/World/DrAnmarLeakTest/ParticleSystem",particles_path="/World/DrAnmarLeakTest/Particles",stage=None):
    stage=_current_stage(stage);ensure_leak_particle_system(system_path=system_path,stage=stage)
    from omni.physx.scripts import particleUtils
    from pxr import Gf,Sdf
    widths=[0.0018]*len(positions)
    return particleUtils.add_physx_particleset_points(stage,Sdf.Path(particles_path),[Gf.Vec3f(*p) for p in positions],[Gf.Vec3f(*v) for v in velocities],widths,Sdf.Path(system_path),True,True,0,1.0,0.02)


PHASE_TARGETS={
    "inspect":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":-0.055,"mandrel_expansion_joint":0.0,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "capture":{"left_approximation_joint":0.004,"right_approximation_joint":-0.004,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":-0.025,"mandrel_expansion_joint":0.0,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "align":{"left_approximation_joint":0.012,"right_approximation_joint":-0.012,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "mandrel":{"left_approximation_joint":0.012,"right_approximation_joint":-0.012,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.006,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "approximate":{"left_approximation_joint":0.030,"right_approximation_joint":-0.030,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "evert":{"left_approximation_joint":0.030,"right_approximation_joint":-0.030,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.006,"right_eversion_joint":-0.006,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "staple":{"left_approximation_joint":0.030,"right_approximation_joint":-0.030,"left_capture_joint":0.008,"right_capture_joint":-0.008,"left_eversion_joint":0.006,"right_eversion_joint":-0.006,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.004,"staple_driver_joint":-0.027,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "release_capture":{"left_approximation_joint":0.014,"right_approximation_joint":-0.014,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "reinforce":{"left_approximation_joint":0.014,"right_approximation_joint":-0.014,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":120.0,"collar_applicator_joint":0.046,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "occlude":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":120.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.008,"right_occluder_valve_joint":0.008,"pressure_valve_joint":0.0},
    "pressurize":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":120.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.008,"right_occluder_valve_joint":0.008,"pressure_valve_joint":0.008},
    "verify":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":0.0,"mandrel_expansion_joint":0.002,"staple_driver_joint":0.0,"collar_carousel_joint":120.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.008,"right_occluder_valve_joint":0.008,"pressure_valve_joint":0.002},
    "complete":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":-0.055,"mandrel_expansion_joint":0.0,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
    "abort":{"left_approximation_joint":0.0,"right_approximation_joint":0.0,"left_capture_joint":0.0,"right_capture_joint":0.0,"left_eversion_joint":0.0,"right_eversion_joint":0.0,"mandrel_extension_joint":-0.055,"mandrel_expansion_joint":0.0,"staple_driver_joint":0.0,"collar_carousel_joint":0.0,"collar_applicator_joint":0.0,"left_occluder_valve_joint":0.0,"right_occluder_valve_joint":0.0,"pressure_valve_joint":0.0},
}


def phase_targets(phase: str):
    try:return dict(PHASE_TARGETS[phase])
    except KeyError as exc:raise KeyError(f"Unknown anastomosis phase {phase!r}") from exc


@dataclass
class AdaptiveAnastomosisSequenceController:
    phase: str="inspect"
    capture: BilateralTissueCaptureController|None=None
    staples: StapleRingRetentionController=field(default_factory=StapleRingRetentionController)
    collar: ReinforcementCollarBondController=field(default_factory=ReinforcementCollarBondController)
    patency: LumenPatencyController=field(default_factory=LumenPatencyController)
    leak_test: PressureDecayLeakController=field(default_factory=PressureDecayLeakController)
    history: list[str]=field(default_factory=list)
    def transition(self,phase: str):
        targets=phase_targets(phase);self.phase=phase;self.history.append(phase)
        if phase=="verify":self.leak_test.reset()
        return targets
''')


def readme() -> str:
    return f'''# {ASSET_NAME} v{VERSION}

Dr.Anmar executable simulation-training workcell for end-to-end hollow-tissue anastomosis tasks. The provider-neutral Franka-compatible system targets NVIDIA Isaac Sim and Isaac Lab while keeping the OpenUSD asset contract portable.

## Capabilities

- bilateral circumferential tissue capture with {CAPTURE_CELL_COUNT_PER_SIDE} independent cells per side;
- independent axial approximation of the two tissue ends;
- lumen-preserving alignment mandrel and expandable centering cage;
- independent edge-eversion rings;
- one-shot {STAPLE_COUNT}-position circumferential staple crown;
- individual retained staple bodies with left/right tissue attachment regions;
- circumferential reinforcement collar with independent bilateral bond sectors;
- lumen patency scoring and pressure-decay leak verification;
- direct replacement of the Panda hand at `panda_link8`;
- standalone articulated, Franka payload, and rigid proxy representations.
- explicit deformable distal fixtures, temporary-capture release, retained staples, and bilateral collar-sector attachments.

## Primary assets

```text
dranmar_adaptive_anastomosis_tool_payload.usda
dranmar_adaptive_anastomosis_tool_standalone.usda
dranmar_adaptive_anastomosis_tool_rigid_proxy.usda
dranmar_hollow_tissue_demo.usda
dranmar_anastomosis_staple.usda
dranmar_reinforcement_collar.usda
dranmar_reinforcement_collar_rigid_proxy.usda
dranmar_leak_test_droplet.usda
```

The current package represents staple formation as a discrete open-to-formed event and reinforcement as staged mechanical attachments. It does not claim clinically calibrated penetration, plasticity, tissue damage, adhesive chemistry, healing, patency, or leak thresholds. It is not approved for patient care.
'''


def docs_mechanism() -> str:
    return '''# DrAnmar Adaptive Anastomosis Robot mechanism

The end effector uses two coaxial capture collars around a common local X lumen axis. Each collar carries six broad capture cells. Independent carriage motion moves the tissue ends toward the seam. Independent eversion rings advance after alignment to present the tissue rims to the staple crown.

A transparent lumen mandrel crosses both tissue segments and supports centerline alignment. A six-rib centering cage provides an expandable patency reference. The right-side staple driver advances a sixteen-position crown toward a fixed left-side anvil. After staple retention is established, the capture attachments can be removed without releasing the seam.

A separate carousel and compliant ring platen place the reinforcement collar. Two occluder-control valves and a pressure valve support the subsequent leak-test phase.

The articulation contains 14 active degrees of freedom and one additional fixed anvil-mount joint prim. The fixed joint is intentionally excluded from the active joint contract.
'''


def docs_physical_anastomosis() -> str:
    return '''# DrAnmar physical anastomosis contract

## Tissue capture

Each tissue side is connected to six separate capture-cell colliders through current `OmniPhysicsVtxXformAttachment` schemas. Two explicit kinematic distal fixtures prevent unconstrained rigid drift of the cooked surfaces. Carriage motion therefore loads the tissue through the articulated mechanism. The tissue is not moved by directly overwriting its transform.

Surface self-collision is opt-in. The authored geometric wall is 2.4 mm thick and the provisional surface thickness is also 2.4 mm; enabling self-collision without retuning those values creates a full inner/outer-wall contact set rather than a qualified lumen-collapse model.

The portable solver mesh uses 16 axial by 32 circumferential segments per inner and outer wall. This keeps current CUDA surface-contact capacity bounded while retaining a closed, watertight hollow-wall topology; higher-resolution visual or calibrated solver meshes can be substituted through the same prim contract.

## Edge apposition and eversion

The lumen mandrel is inserted before final approximation. Eversion rings then move toward the central seam. Their rounded contact surfaces are intended to interact with the outer wall and rim region while the capture collars maintain distributed support.

## Staple retention

Sixteen formed staple bodies are spawned around the seam. Each staple owns two independent attachment volumes: one for the left tissue and one for the right. The 12 temporary capture constraints can then be released while the 32 staple-leg attachments remain the load-bearing bridge. Pullout is represented by removing a staple's tissue attachments when the caller reports load above a provisional threshold.

Continuous metal plasticity, penetration, puncture damage, wall crushing, ischemia and cut-through require calibrated solver and specimen data and are not claimed by this release.

## Reinforcement collar

The reinforcement collar is supplied as a connected triangular surface and as a stable rigid bond carrier. The rigid carrier contains 16 independent left and 16 independent right bond cells. Bond strength rises from a provisional 0.18 N sector tack value to 2.2 N over 45 seconds. This models the mechanical result of reinforcement but not biochemical adhesion or healing.
'''


def docs_leak_test() -> str:
    return '''# DrAnmar patency and leak verification

The mandrel and centering cage provide a reference for minimum lumen radius, centerline offset and axis alignment. `LumenPatencyController` evaluates user-supplied radial samples and reports minimum radius, mean radius, area fraction, offset, axis error and pass state.

`PressureDecayLeakController` is a reduced-order, dimensionally consistent chamber model. Pump inflow and orifice outflow change pressure through an effective chamber compliance. Effective leak area depends on residual edge gap, retained staple fraction and reinforcement bond fraction. It tracks instantaneous leak flow, pressure, integrated leak volume, observation time and pass state. `LeakTestLedger` conserves test medium across the reservoir, isolated chamber, active leaked medium, collection, spill, and discard buckets.

The leak model is a research benchmark. It is not a clinical leak test, does not reproduce full fluid-structure interaction, and must be calibrated against the selected tissue, test medium, pressure protocol and instrumentation.
'''


def docs_franka() -> str:
    return '''# DrAnmar Franka integration

The combined spawner references the standard NVIDIA Isaac Franka USD, snapshots the stock Panda hand-joint body target and local mounting frame, disables the Panda hand, finger links and finger joints, references the DrAnmar payload, and creates a fixed wrist-to-tool joint using that resolved frame.

This preserves the standard Panda-hand relationship, including the nominal -45 degree local Z rotation, without assuming a particular flattened Franka prim path. The closure mechanism becomes part of the same reduced-coordinate articulation and is controlled through dedicated Isaac Lab actuator groups.

Use `make_franka_adaptive_anastomosis_robot_cfg()` for the combined robot, `make_tool_cfg()` for the standalone mechanism, and `make_rigid_proxy_cfg()` for perception and planning tasks.
'''


def docs_validation() -> str:
    return '''# Integrity and runtime boundaries

Static gates cover deterministic assets, dependency closure, controller
invariants, fail-closed attachment overlap, leak-ledger conservation, and
source/container integrity. The optional Isaac script is diagnostic only.

Articulation motion, attachment creation, staple count, collar sectors, and
nominal reduced-order inputs do not establish patency, edge apposition, or a
pressure-tight seam. Patency and leak status may be promoted only from measured
runtime lumen/seam geometry and calibrated pressure/flow sensing.

No current record qualifies staple penetration or plasticity, collar adhesion,
anastomosis efficacy, physical calibration, clinical performance, or patient
use.
'''


def example_scene() -> str:
    return textwrap.dedent('''#!/usr/bin/env python3
"""Minimal scene configuration for the DrAnmar Adaptive Anastomosis Robot."""
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from orbit.surgical.assets.adaptive_anastomosis_robot import (
    make_franka_adaptive_anastomosis_robot_cfg,
    spawn_hollow_tissue_demo,
)

@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = make_franka_adaptive_anastomosis_robot_cfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        staple_state="loaded",
        collar_state="loaded",
        test_medium_state="full",
        collection_state="empty",
    )


def spawn_task_assets():
    return spawn_hollow_tissue_demo(
        "/World/DrAnmarHollowTissue",
        state="initial",
        translation=(0.62, 0.0, 0.82),
    )
''')


def author_installer() -> str:
    installer_path = PACKAGE_ROOT / "scripts" / "install_into_dranmar.py"
    if not installer_path.exists():
        raise FileNotFoundError(
            "The authoritative safe installer is required for generation: "
            f"{installer_path}"
        )
    return installer_path.read_text(encoding="utf-8")
    # Unreachable historical template retained only for source provenance.
    return textwrap.dedent('''#!/usr/bin/env python3
"""Install the DrAnmar Adaptive Anastomosis Robot into a drAnmar checkout."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

PACKAGE_ROOT=Path(__file__).resolve().parents[1]


def copytree_contents(source: Path,destination: Path):
    destination.mkdir(parents=True,exist_ok=True)
    for path in source.rglob("*"):
        target=destination/path.relative_to(source)
        if path.is_dir():target.mkdir(parents=True,exist_ok=True)
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("repository",type=Path);args=parser.parse_args()
    repo=args.repository.resolve()
    copytree_contents(PACKAGE_ROOT/"source",repo/"source")
    copytree_contents(PACKAGE_ROOT/"physics_next",repo/"physics_next")
    copytree_contents(PACKAGE_ROOT/"docs",repo/"docs")
    copytree_contents(PACKAGE_ROOT/"examples",repo/"examples")
    copytree_contents(PACKAGE_ROOT/"scripts",repo/"scripts")
    init_path=repo/"source/extensions/orbit.surgical.assets/orbit/surgical/assets/__init__.py"
    if init_path.exists():
        text=init_path.read_text(encoding="utf-8")
        line="from .adaptive_anastomosis_robot import *"
        if line not in text:init_path.write_text(text.rstrip()+"\\n"+line+"\\n",encoding="utf-8")
    portfolio=repo/"physics_next/dr-anmar-assets.json"
    entry={
        "id":"dranmar-adaptive-anastomosis-robot-v1",
        "asset":"source/extensions/orbit.surgical.assets/data/Props/SurgicalReconstruction/AdaptiveAnastomosisRobot/dranmar_adaptive_anastomosis_tool_standalone.usda",
        "auxiliary_assets":[
            "source/extensions/orbit.surgical.assets/data/Props/SurgicalReconstruction/AdaptiveAnastomosisRobot/dranmar_hollow_tissue_demo.usda",
            "source/extensions/orbit.surgical.assets/data/Props/SurgicalReconstruction/AdaptiveAnastomosisRobot/dranmar_anastomosis_staple.usda",
            "source/extensions/orbit.surgical.assets/data/Props/SurgicalReconstruction/AdaptiveAnastomosisRobot/dranmar_reinforcement_collar.usda"
        ],
        "profile":"physics_next/surgical-reconstruction/dranmar-adaptive-anastomosis-v1.json",
        "live_integration":"franka_panda_link8_replacement_and_standalone_articulation",
        "live_behavior":"bilateral_capture_alignment_eversion_circumferential_stapling_reinforcement_patency_and_pressure_decay_verification",
        "deployment":"enabled_as_training_workcell",
        "product_capability":"executable_training_workcell",
        "training_readiness":"available_for_simulation_training_data_generation_and_evaluation",
        "software_evidence":"repository_verified_asset_task_and_controller_contracts",
        "native_simulator_evidence":"native_cuda_execution_not_yet_recorded",
        "real_world_evidence":"instrumented_anastomosis_bench_evidence_not_yet_established",
        "clinical_validation":False,
    }
    if portfolio.exists():
        data=json.loads(portfolio.read_text(encoding="utf-8"));assets=data.setdefault("assets",[])
        assets[:]=[x for x in assets if x.get("id")!=entry["id"]];assets.append(entry)
        portfolio.write_text(json.dumps(data,indent=2)+"\\n",encoding="utf-8")
    print(f"Installed DrAnmar Adaptive Anastomosis Robot into {repo}")

if __name__=="__main__":main()
''')


def write_json(path: Path,payload: Any) -> Path:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");return path


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def write_asset_files(bundle: ToolBundle) -> list[Path]:
    ASSET_ROOT.mkdir(parents=True,exist_ok=True);GLB_ROOT.mkdir(parents=True,exist_ok=True);PREVIEW_ROOT.mkdir(parents=True,exist_ok=True);DOCS_ROOT.mkdir(parents=True,exist_ok=True);EXAMPLE_ROOT.mkdir(parents=True,exist_ok=True);INTEGRATION_PATH.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    mapping={
        "dranmar_adaptive_anastomosis_tool_payload.usda":tool_usda(bundle,False),
        "dranmar_adaptive_anastomosis_tool_standalone.usda":tool_usda(bundle,True),
        "dranmar_adaptive_anastomosis_tool_rigid_proxy.usda":rigid_proxy_usda(bundle),
        "dranmar_hollow_tissue_demo.usda":tissue_usda(bundle),
        "dranmar_anastomosis_staple.usda":staple_usda(bundle),
        "dranmar_reinforcement_collar.usda":collar_surface_usda(bundle),
        "dranmar_reinforcement_collar_rigid_proxy.usda":collar_proxy_usda(bundle),
        "dranmar_leak_test_droplet.usda":droplet_usda(),
        "README.md":readme(),
    }
    for name,text in mapping.items():
        p=ASSET_ROOT/name;p.write_text(text,encoding="utf-8");files.append(p)
    license_path=ASSET_ROOT/"LICENSE.txt"
    if not license_path.exists():
        license_path.write_text(
            "Apache License 2.0\n"
            "Copyright (c) 2026 DrAnmar Project Developers\n"
            "See https://www.apache.org/licenses/LICENSE-2.0\n",
            encoding="utf-8",
        )
    files.append(license_path)
    files += generate_textures();files += export_glbs(bundle)
    files += [
        write_json(ASSET_ROOT/"interaction_frames.json",interaction_frames(bundle)),
        write_json(ASSET_ROOT/"franka_mount_contract.json",mount_contract()),
        write_json(ASSET_ROOT/"adaptive_anastomosis_task_contract.json",task_contract()),
        write_json(ASSET_ROOT/"physics_profile.json",physics_profile(bundle)),
        write_json(ASSET_ROOT/"collider_coverage.json",collider_coverage(bundle)),
        write_json(PHYSICS_PROFILE_PATH,physics_profile(bundle)),
    ]
    INTEGRATION_PATH.write_text(author_integration_module(),encoding="utf-8");files.append(INTEGRATION_PATH)
    docs={
        "MECHANISM.md":docs_mechanism(),
        "PHYSICAL_ANASTOMOSIS.md":docs_physical_anastomosis(),
        "PATENCY_AND_LEAK_TEST.md":docs_leak_test(),
        "FRANKA_INTEGRATION.md":docs_franka(),
        "VALIDATION.md":docs_validation(),
    }
    for name,text in docs.items():p=DOCS_ROOT/name;p.write_text(text,encoding="utf-8");files.append(p)
    p=EXAMPLE_ROOT/"franka_adaptive_anastomosis_scene.py";p.write_text(example_scene(),encoding="utf-8");files.append(p)
    p=PACKAGE_ROOT/"scripts/install_into_dranmar.py";p.write_text(author_installer(),encoding="utf-8");p.chmod(0o755);files.append(p)
    files += [make_preview(bundle),make_full_arm_preview(bundle)]
    return files


def sync_extension_data() -> None:
    target=EXTENSION_ROOT/"data"/CATALOG_SUBPATH
    if target.exists():shutil.rmtree(target)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(ASSET_ROOT,target)


def all_payload_files() -> list[Path]:
    mirror_root = EXTENSION_ROOT / "data" / CATALOG_SUBPATH
    excluded_names = {"asset_manifest.json", "static_build_report.json", ".DS_Store"}
    return sorted(
        p
        for p in PACKAGE_ROOT.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and p.name not in excluded_names
        and not p.is_relative_to(mirror_root)
    )


def build_manifest(files: Sequence[Path]) -> dict[str,object]:
    return {"schema":"dranmar.asset-manifest.v1","asset":"dranmar-adaptive-anastomosis-robot-v1","version":VERSION,"catalog_subpath":CATALOG_SUBPATH.as_posix(),"file_count":len(files),"files":[{"path":p.relative_to(PACKAGE_ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in files]}


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
            archive.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    return output


def write_checksum(path: Path) -> Path:
    output=Path(str(path)+".sha256");output.write_text(f"{sha256(path)}  {path.name}\n",encoding="utf-8");return output


def build_overlay() -> Path:
    staging=PACKAGE_ROOT.parent/"_dranmar_adaptive_anastomosis_overlay"
    if staging.exists():shutil.rmtree(staging)
    for name in ("source","physics_next","docs","examples","scripts","tests"):
        src=PACKAGE_ROOT/name
        if src.exists():shutil.copytree(src,staging/name)
    output=PACKAGE_ROOT.parent/f"dranmar_adaptive_anastomosis_robot_repo_overlay_v{VERSION}.zip";zip_tree(staging,output);shutil.rmtree(staging);return output


def static_report(files: Sequence[Path]) -> dict[str,object]:
    usda=[p for p in files if p.suffix==".usda"]
    checks=[]
    for path in usda:
        text=path.read_text(encoding="utf-8")
        checks.append({"path":path.relative_to(PACKAGE_ROOT).as_posix(),"balanced_braces":text.count("{")==text.count("}"),"flat_quaternion_count":text.count("quatf "),"nested_quaternion_pattern_absent":"(1, (" not in text,"one_line_over_absent":all(not(line.strip().startswith("over ") and "{" in line and "}" in line) for line in text.splitlines())})
    return {
        "schema":"dranmar.static-build-report.v1",
        "asset":"dranmar-adaptive-anastomosis-robot-v1",
        "usda_checks":checks,
        "python_files":[p.relative_to(PACKAGE_ROOT).as_posix() for p in files if p.suffix==".py"],
        "native_simulator_evidence":"not_recorded",
        "real_world_evidence":"not_established",
    }


def generate() -> dict[str,object]:
    for cache in sorted(PACKAGE_ROOT.rglob("__pycache__"),reverse=True):
        shutil.rmtree(cache)
    for bytecode in PACKAGE_ROOT.rglob("*.pyc"):
        bytecode.unlink()
    old_manifest=ASSET_ROOT/"asset_manifest.json"
    if old_manifest.exists():
        old_manifest.unlink()
    bundle=build_tool();write_asset_files(bundle)
    files=all_payload_files()
    manifest=write_json(ASSET_ROOT/"asset_manifest.json",build_manifest(files));sync_extension_data()
    report=write_json(PACKAGE_ROOT/"static_build_report.json",static_report(all_payload_files()))
    for python_path in sorted(PACKAGE_ROOT.rglob("*.py")):
        compile(python_path.read_text(encoding="utf-8"),str(python_path),"exec")
    dev_zip=PACKAGE_ROOT.parent/f"dranmar_adaptive_anastomosis_robot_v{VERSION}.zip";zip_tree(PACKAGE_ROOT,dev_zip,prefix=PACKAGE_ROOT.name)
    catalog_zip=PACKAGE_ROOT.parent/f"dranmar_adaptive_anastomosis_robot_catalog_v{VERSION}.zip";zip_tree(PACKAGE_ROOT/"assets",catalog_zip)
    overlay_zip=build_overlay()
    checksums=[write_checksum(p) for p in (dev_zip,catalog_zip,overlay_zip)]
    release={
        "schema":"dranmar.release.v1","asset":"dranmar-adaptive-anastomosis-robot-v1","version":VERSION,
        "package_root":str(PACKAGE_ROOT),"catalog_subpath":CATALOG_SUBPATH.as_posix(),
        "development_package":{"path":str(dev_zip),"sha256":sha256(dev_zip),"bytes":dev_zip.stat().st_size},
        "catalog_package":{"path":str(catalog_zip),"sha256":sha256(catalog_zip),"bytes":catalog_zip.stat().st_size},
        "repository_overlay":{"path":str(overlay_zip),"sha256":sha256(overlay_zip),"bytes":overlay_zip.stat().st_size},
        "primary_assets":["dranmar_adaptive_anastomosis_tool_standalone.usda","dranmar_adaptive_anastomosis_tool_payload.usda","dranmar_hollow_tissue_demo.usda","dranmar_anastomosis_staple.usda","dranmar_reinforcement_collar.usda"],
        "runtime_validation":static_report(all_payload_files())["runtime_validation"],"clinical_validation":False,
    }
    release_path=write_json(PACKAGE_ROOT.parent/f"dranmar_adaptive_anastomosis_robot_release_v{VERSION}.json",release)
    return {"release":release,"release_path":str(release_path),"checksums":[str(x) for x in checksums],"manifest":str(manifest),"static_report":str(report)}


def main() -> None:
    print(json.dumps(generate(),indent=2))


if __name__=="__main__":main()
