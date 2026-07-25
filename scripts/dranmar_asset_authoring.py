"""Shared procedural authoring helpers for DrAnmar OpenUSD research assets.

This module is intentionally limited to geometry, OpenUSD text authoring,
inspection exports, preview rendering, and deterministic package utilities.
It has no Isaac runtime dependency.
"""
from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh


def f(value: float, digits: int = 10) -> str:
    value = float(value)
    if abs(value) < 1.0e-18:
        value = 0.0
    return f"{value:.{digits}g}"


def vec(values: Sequence[float], digits: int = 10) -> str:
    return "(" + ", ".join(f(v, digits) for v in values) + ")"


def quat(values: Sequence[float]) -> str:
    if len(values) != 4:
        raise ValueError("quaternion must use flat wxyz USDA syntax")
    return vec(values)


def normalize(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n <= 1.0e-12:
        raise ValueError("cannot normalize vector")
    return a / n


def rotation_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    x, y, z = normalize(axis)
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.asarray(
        [
            [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
            [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
            [z*x*C-y*s, z*y*C+x*s, z*z*C+c],
        ], dtype=float,
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


def transform(mesh: trimesh.Trimesh, translation=(0.0,0.0,0.0), rotation: np.ndarray|None=None, scale=None) -> trimesh.Trimesh:
    mesh = mesh.copy()
    T = np.eye(4)
    if rotation is not None:
        T[:3,:3] = np.asarray(rotation, dtype=float)
    T[:3,3] = np.asarray(translation, dtype=float)
    mesh.apply_transform(T)
    if scale is not None:
        mesh.apply_scale(np.asarray(scale, dtype=float))
    return mesh


def box_mesh(size: Sequence[float], center=(0.0,0.0,0.0), rotation: np.ndarray|None=None) -> trimesh.Trimesh:
    return transform(trimesh.creation.box(extents=np.asarray(size, dtype=float)), center, rotation)


def rounded_bar_mesh(size: Sequence[float], center=(0.0,0.0,0.0), radius: float=0.003, axis: str="x") -> trimesh.Trimesh:
    sx, sy, sz = [float(v) for v in size]
    if axis == "x":
        core = box_mesh((max(sx-2*radius, radius), sy, sz), center)
        caps = [cylinder_axis(radius, sy, "y", (center[0]+sign*(sx/2-radius), center[1], center[2])) for sign in (-1,1)]
    elif axis == "y":
        core = box_mesh((sx, max(sy-2*radius, radius), sz), center)
        caps = [cylinder_axis(radius, sx, "x", (center[0], center[1]+sign*(sy/2-radius), center[2])) for sign in (-1,1)]
    else:
        core = box_mesh((sx, sy, max(sz-2*radius, radius)), center)
        caps = [cylinder_axis(radius, sx, "x", (center[0], center[1], center[2]+sign*(sz/2-radius))) for sign in (-1,1)]
    return trimesh.util.concatenate([core, *caps])


def cylinder_axis(radius: float, length: float, axis: str="z", center=(0.0,0.0,0.0), sections: int=48) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=float(radius), height=float(length), sections=int(sections))
    if axis == "x":
        R = rotation_matrix((0,1,0), math.pi/2)
    elif axis == "y":
        R = rotation_matrix((1,0,0), -math.pi/2)
    elif axis == "z":
        R = np.eye(3)
    else:
        raise ValueError(axis)
    return transform(mesh, center, R)


def capsule_axis(radius: float, length: float, axis: str="z", center=(0.0,0.0,0.0), sections: int=24) -> trimesh.Trimesh:
    mesh = trimesh.creation.capsule(radius=float(radius), height=max(0.0,float(length)-2*radius), count=[sections, sections])
    if axis == "x":
        R = rotation_matrix((0,1,0), math.pi/2)
    elif axis == "y":
        R = rotation_matrix((1,0,0), -math.pi/2)
    elif axis == "z":
        R = np.eye(3)
    else:
        raise ValueError(axis)
    return transform(mesh, center, R)


def ellipsoid_mesh(radii: Sequence[float], center=(0.0,0.0,0.0), subdivisions: int=3) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.apply_scale(np.asarray(radii, dtype=float))
    mesh.apply_translation(np.asarray(center, dtype=float))
    return mesh


def torus_axis(major_radius: float, minor_radius: float, axis: str="z", center=(0.0,0.0,0.0), major_sections=64, minor_sections=16) -> trimesh.Trimesh:
    mesh = trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius, major_sections=major_sections, minor_sections=minor_sections)
    if axis == "x":
        R = rotation_matrix((0,1,0), math.pi/2)
    elif axis == "y":
        R = rotation_matrix((1,0,0), -math.pi/2)
    else:
        R = np.eye(3)
    return transform(mesh, center, R)


def frustum_axis(radius0: float, radius1: float, height: float, axis: str="z", center=(0,0,0), sections: int=48) -> trimesh.Trimesh:
    z0,z1=-height/2,height/2
    points=[]
    for z,r in ((z0,radius0),(z1,radius1)):
        for i in range(sections):
            a=2*math.pi*i/sections
            points.append((r*math.cos(a),r*math.sin(a),z))
    points.extend([(0,0,z0),(0,0,z1)])
    faces=[]
    for i in range(sections):
        j=(i+1)%sections
        faces += [(i,j,sections+j),(i,sections+j,sections+i),(2*sections,j,i),(2*sections+1,sections+i,sections+j)]
    mesh=trimesh.Trimesh(vertices=np.asarray(points),faces=np.asarray(faces),process=True)
    if axis=="x": R=rotation_matrix((0,1,0),math.pi/2)
    elif axis=="y": R=rotation_matrix((1,0,0),-math.pi/2)
    else: R=np.eye(3)
    return transform(mesh,center,R)


def capsule_between(p0: Sequence[float], p1: Sequence[float], radius: float, sections: int=20) -> trimesh.Trimesh:
    p0,p1=np.asarray(p0,dtype=float),np.asarray(p1,dtype=float)
    d=p1-p0; length=float(np.linalg.norm(d))
    if length<=1e-10:
        return transform(trimesh.creation.icosphere(subdivisions=2,radius=radius),p0)
    mesh=trimesh.creation.capsule(radius=radius,height=max(0.0,length-2*radius),count=[sections,sections])
    z=np.asarray([0.0,0.0,1.0]); u=d/length; cross=np.cross(z,u); dot=float(np.clip(np.dot(z,u),-1.0,1.0))
    if np.linalg.norm(cross)<=1e-12:
        R=np.eye(3) if dot>0 else rotation_matrix((1,0,0),math.pi)
    else:
        R=rotation_matrix(cross,math.acos(dot))
    return transform(mesh,(p0+p1)/2,R)


def wire_path(points: Sequence[Sequence[float]], radius: float, sections: int=20) -> trimesh.Trimesh:
    pts=[np.asarray(p,dtype=float) for p in points]
    parts=[]
    for a,b in zip(pts[:-1],pts[1:]):
        parts.append(capsule_between(a,b,radius,sections))
    for p in pts[1:-1]:
        parts.append(transform(trimesh.creation.icosphere(subdivisions=2,radius=radius),p))
    return trimesh.util.concatenate(parts)


def annular_sector_mesh(r_inner: float, r_outer: float, thickness: float, start: float, end: float, z: float=0.0, segments: int=24) -> trimesh.Trimesh:
    vertices=[]
    for zz in (z-thickness/2,z+thickness/2):
        for r in (r_inner,r_outer):
            for i in range(segments+1):
                a=start+(end-start)*i/segments
                vertices.append((r*math.cos(a),r*math.sin(a),zz))
    vertices=np.asarray(vertices,dtype=float)
    ring=segments+1
    faces=[]
    # top/bottom annular strip
    for i in range(segments):
        bi=i; bo=ring+i; ti=2*ring+i; to=3*ring+i
        faces += [(bi,bo,bo+1),(bi,bo+1,bi+1)]
        faces += [(ti,to+1,to),(ti,ti+1,to+1)]
        faces += [(bo,to,to+1),(bo,to+1,bo+1)]
        faces += [(bi,bi+1,ti+1),(bi,ti+1,ti)]
    # start/end walls
    for i0 in (0,segments):
        bi=i0; bo=ring+i0; ti=2*ring+i0; to=3*ring+i0
        faces += [(bi,ti,to),(bi,to,bo)]
    mesh=trimesh.Trimesh(vertices=vertices,faces=np.asarray(faces),process=True)
    mesh.fix_normals(); return mesh


def grid_surface_mesh(width: float, depth: float, nx: int, ny: int, *, z_func, center=(0,0,0)) -> trimesh.Trimesh:
    cx,cy,cz=center; points=[]
    for iy in range(ny):
        y=-depth/2+depth*iy/(ny-1)
        for ix in range(nx):
            x=-width/2+width*ix/(nx-1)
            points.append((cx+x,cy+y,cz+float(z_func(x,y))))
    faces=[]
    for iy in range(ny-1):
        for ix in range(nx-1):
            a=iy*nx+ix;b=a+1;c=a+nx;d=c+1
            if (ix+iy)%2==0: faces += [(a,b,d),(a,d,c)]
            else: faces += [(a,b,c),(b,d,c)]
    mesh=trimesh.Trimesh(vertices=np.asarray(points),faces=np.asarray(faces),process=False)
    mesh.remove_unreferenced_vertices(); mesh.fix_normals(); return mesh


def mesh_bounds(meshes: Sequence[trimesh.Trimesh]) -> tuple[np.ndarray,np.ndarray]:
    mins=np.vstack([m.bounds[0] for m in meshes]); maxs=np.vstack([m.bounds[1] for m in meshes])
    return mins.min(axis=0),maxs.max(axis=0)


def box_mass_properties(meshes: Sequence[trimesh.Trimesh], mass: float) -> dict[str,object]:
    bmin,bmax=mesh_bounds(meshes); size=np.maximum(bmax-bmin,1e-5); com=(bmin+bmax)*0.5
    dx,dy,dz=size
    inertia=(mass*(dy*dy+dz*dz)/12,mass*(dx*dx+dz*dz)/12,mass*(dx*dx+dy*dy)/12)
    return {"mass_kg":float(mass),"center_of_mass_m":[float(x) for x in com],"diagonal_inertia_kg_m2":[float(x) for x in inertia],"principal_axes_wxyz":[1.0,0.0,0.0,0.0],"bounds_min_m":[float(x) for x in bmin],"bounds_max_m":[float(x) for x in bmax]}


@dataclass
class Visual:
    name: str
    mesh: trimesh.Trimesh
    material: str
    labels: tuple[str,...]=()


@dataclass
class Collider:
    name: str
    kind: str
    center: tuple[float,float,float]
    size: tuple[float,float,float]|None=None
    radius: float|None=None
    height: float|None=None
    axis: str="z"
    orientation_wxyz: tuple[float,float,float,float]=(1.0,0.0,0.0,0.0)
    physics_material: str="PolymerPhysics"
    role: str="collision"
    author_enabled: bool=True


@dataclass
class Link:
    name: str
    translation: tuple[float,float,float]
    visuals: list[Visual]
    colliders: list[Collider]
    mass_kg: float|None
    labels: tuple[str,...]=()
    mass_properties: dict[str,object]|None=field(init=False)
    def __post_init__(self):
        self.mass_properties=None if self.mass_kg is None else box_mass_properties([v.mesh for v in self.visuals],self.mass_kg)


@dataclass
class Joint:
    name: str
    type: str
    body0: str
    body1: str
    axis: str|None
    local_pos0: tuple[float,float,float]
    local_pos1: tuple[float,float,float]
    lower: float|None=None
    upper: float|None=None
    stiffness: float=0.0
    damping: float=0.0
    max_force: float=0.0
    target_velocity: float=0.0


def material_color(material: str, colors: dict[str,tuple[int,int,int,int]]) -> tuple[int,int,int,int]:
    return colors.get(material,(180,180,180,255))


def pbr(mesh: trimesh.Trimesh, material: str, colors: dict[str,tuple[int,int,int,int]]) -> trimesh.Trimesh:
    mesh=mesh.copy(); rgba=material_color(material,colors)
    mesh.visual=trimesh.visual.ColorVisuals(mesh=mesh,face_colors=np.tile(np.asarray(rgba,dtype=np.uint8),(len(mesh.faces),1)))
    return mesh


def mesh_usda(name: str, mesh: trimesh.Trimesh, material_path: str, labels: Sequence[str]=(), indent: str="            ", double_sided: bool=False) -> str:
    vertices=np.asarray(mesh.vertices,dtype=float); faces=np.asarray(mesh.faces,dtype=int); normals=np.asarray(mesh.vertex_normals,dtype=float); bmin,bmax=mesh.bounds
    points=",\n".join(indent+"        "+vec(p) for p in vertices)
    counts=", ".join("3" for _ in faces); indices=", ".join(str(int(i)) for i in faces.reshape(-1)); normal_values=",\n".join(indent+"        "+vec(n) for n in normals)
    label_attr=""
    if labels: label_attr=f'{indent}    custom token[] drAnmar:labels = [{", ".join(repr(x) for x in labels)}]\n'.replace("'",'"')
    return f'''{indent}def Mesh "{name}" (\n{indent}    prepend apiSchemas = ["MaterialBindingAPI"]\n{indent})\n{indent}{{\n{indent}    rel material:binding = <{material_path}>\n{label_attr}{indent}    uniform bool doubleSided = {str(bool(double_sided)).lower()}\n{indent}    float3[] extent = [{vec(bmin)}, {vec(bmax)}]\n{indent}    int[] faceVertexCounts = [{counts}]\n{indent}    int[] faceVertexIndices = [{indices}]\n{indent}    point3f[] points = [\n{points}\n{indent}    ]\n{indent}    normal3f[] primvars:normals = [\n{normal_values}\n{indent}    ] (\n{indent}        interpolation = "vertex"\n{indent}    )\n{indent}    uniform token subdivisionScheme = "none"\n{indent}}}'''


def collider_usda(c: Collider, root_path: str, indent: str="            ") -> str:
    if not c.author_enabled: return ""
    api='prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]'
    common=f'''{indent}    custom string drAnmar:role = "{c.role}"\n{indent}    rel material:binding:physics = <{root_path}/PhysicsMaterials/{c.physics_material}>\n{indent}    bool physics:collisionEnabled = true\n{indent}    float physxCollision:contactOffset = 0.00035\n{indent}    float physxCollision:restOffset = 0\n{indent}    double3 xformOp:translate = {vec(c.center)}\n{indent}    quatf xformOp:orient = {quat(c.orientation_wxyz)}\n'''
    if c.kind=="box":
        assert c.size is not None
        return f'''{indent}def Cube "{c.name}" (\n{indent}    {api}\n{indent})\n{indent}{{\n{common}{indent}    double size = 1\n{indent}    double3 xformOp:scale = {vec(c.size)}\n{indent}    uniform token purpose = "guide"\n{indent}    token visibility = "invisible"\n{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]\n{indent}}}'''
    if c.kind=="cylinder":
        assert c.radius is not None and c.height is not None
        return f'''{indent}def Cylinder "{c.name}" (\n{indent}    {api}\n{indent})\n{indent}{{\n{common}{indent}    uniform token axis = "{c.axis.upper()}"\n{indent}    double radius = {f(c.radius)}\n{indent}    double height = {f(c.height)}\n{indent}    uniform token purpose = "guide"\n{indent}    token visibility = "invisible"\n{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n{indent}}}'''
    if c.kind=="sphere":
        assert c.radius is not None
        return f'''{indent}def Sphere "{c.name}" (\n{indent}    {api}\n{indent})\n{indent}{{\n{common}{indent}    double radius = {f(c.radius)}\n{indent}    uniform token purpose = "guide"\n{indent}    token visibility = "invisible"\n{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n{indent}}}'''
    raise ValueError(c.kind)


def frame_usda(name: str, data: dict[str,object], indent: str="            ") -> str:
    return f'''{indent}def Xform "{name}"\n{indent}{{\n{indent}    custom string drAnmar:role = "{data['role']}"\n{indent}    double3 xformOp:translate = {vec(data['position'])}\n{indent}    quatf xformOp:orient = {quat(data['orientation_wxyz'])}\n{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n{indent}}}'''


def link_usda(link: Link, root_path: str, frames: dict[str,dict[str,object]]) -> str:
    body_attrs=""
    if link.mass_properties is not None:
        mp=link.mass_properties
        body_attrs=f'''        bool physics:rigidBodyEnabled = true\n        bool physics:kinematicEnabled = false\n        float physics:mass = {f(mp['mass_kg'])}\n        point3f physics:centerOfMass = {vec(mp['center_of_mass_m'])}\n        vector3f physics:diagonalInertia = {vec(mp['diagonal_inertia_kg_m2'])}\n        quatf physics:principalAxes = {quat(mp['principal_axes_wxyz'])}\n        bool physxRigidBody:enableCCD = true\n        float physxRigidBody:linearDamping = 0.04\n        float physxRigidBody:angularDamping = 0.06\n        int physxRigidBody:solverPositionIterationCount = 20\n        int physxRigidBody:solverVelocityIterationCount = 6\n'''
    visual_blocks="\n".join(mesh_usda(v.name,v.mesh,f"{root_path}/Looks/{v.material}",v.labels) for v in link.visuals)
    collider_blocks="\n".join(collider_usda(c,root_path) for c in link.colliders if c.author_enabled)
    frame_blocks="\n".join(frame_usda(n,d) for n,d in frames.items() if d["parent_link"]==link.name)
    labels_attr=""
    if link.labels: labels_attr=f'        custom token[] drAnmar:labels = [{", ".join(repr(x) for x in link.labels)}]\n'.replace("'",'"')
    return f'''    def Xform "{link.name}" (\n        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]\n    )\n    {{\n        double3 xformOp:translate = {vec(link.translation)}\n        uniform token[] xformOpOrder = ["xformOp:translate"]\n{labels_attr}{body_attrs}        def Scope "Visuals"\n        {{\n{visual_blocks}\n        }}\n        def Scope "Collisions"\n        {{\n{collider_blocks}\n        }}\n        def Scope "Frames"\n        {{\n{frame_blocks}\n        }}\n    }}'''


def joint_usda(j: Joint, root_path: str) -> str:
    if j.type=="prismatic": typename="PhysicsPrismaticJoint"; drive="linear"; axis=f'        uniform token physics:axis = "{j.axis}"\n'
    elif j.type=="revolute": typename="PhysicsRevoluteJoint"; drive="angular"; axis=f'        uniform token physics:axis = "{j.axis}"\n'
    elif j.type=="fixed": typename="PhysicsFixedJoint"; drive=None; axis=""
    else: raise ValueError(j.type)
    api=f'prepend apiSchemas = ["PhysicsDriveAPI:{drive}"]' if drive else ""
    limits="" if j.lower is None or j.upper is None else f'        float physics:lowerLimit = {f(j.lower)}\n        float physics:upperLimit = {f(j.upper)}\n'
    drive_block=""
    if drive:
        drive_block=f'''        uniform token drive:{drive}:physics:type = "force"\n        float drive:{drive}:physics:stiffness = {f(j.stiffness)}\n        float drive:{drive}:physics:damping = {f(j.damping)}\n        float drive:{drive}:physics:maxForce = {f(j.max_force)}\n        float drive:{drive}:physics:targetPosition = 0\n        float drive:{drive}:physics:targetVelocity = {f(j.target_velocity)}\n'''
    return f'''    def {typename} "{j.name}" (\n        {api}\n    )\n    {{\n{axis}        rel physics:body0 = <{root_path}/Links/{j.body0}>\n        rel physics:body1 = <{root_path}/Links/{j.body1}>\n        point3f physics:localPos0 = {vec(j.local_pos0)}\n        point3f physics:localPos1 = {vec(j.local_pos1)}\n        quatf physics:localRot0 = (1, 0, 0, 0)\n        quatf physics:localRot1 = (1, 0, 0, 0)\n        bool physics:collisionEnabled = false\n{limits}{drive_block}    }}'''


def nested_over(path: Sequence[str], body_lines: Sequence[str], indent: str="            ") -> str:
    lines=[]
    for depth,name in enumerate(path):
        p=indent+"    "*depth; lines += [f'{p}over "{name}"',f'{p}{{']
    body_prefix=indent+"    "*len(path); lines += [f"{body_prefix}{line}" for line in body_lines]
    for depth in reversed(range(len(path))):
        p=indent+"    "*depth; lines.append(f"{p}}}")
    return "\n".join(lines)


def visual_materials_scope(root: str, specs: dict[str,tuple[tuple[float,float,float],float,float,float]]) -> str:
    blocks=[]
    for name,(color,metallic,roughness,opacity) in specs.items():
        blocks.append(f'''        def Material "{name}"\n        {{\n            def Shader "PreviewSurface"\n            {{\n                uniform token info:id = "UsdPreviewSurface"\n                color3f inputs:diffuseColor = {vec(color)}\n                float inputs:metallic = {f(metallic)}\n                float inputs:roughness = {f(roughness)}\n                float inputs:opacity = {f(opacity)}\n                token outputs:surface\n            }}\n            token outputs:surface.connect = </{root}/Looks/{name}/PreviewSurface.outputs:surface>\n        }}''')
    return '    def Scope "Looks"\n    {\n'+"\n".join(blocks)+'\n    }'


def physics_materials_scope(specs: dict[str,tuple[float,float,float]]) -> str:
    blocks=[]
    for name,(static,dynamic,restitution) in specs.items():
        blocks.append(f'''        def Material "{name}" (\n            prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]\n        )\n        {{\n            float physics:staticFriction = {f(static)}\n            float physics:dynamicFriction = {f(dynamic)}\n            float physics:restitution = {f(restitution)}\n            uniform token physxMaterial:frictionCombineMode = "max"\n            uniform token physxMaterial:restitutionCombineMode = "min"\n        }}''')
    return '    def Scope "PhysicsMaterials"\n    {\n'+"\n".join(blocks)+'\n    }'


def export_scene(path: Path, entries: Sequence[tuple[str,trimesh.Trimesh,str]], colors: dict[str,tuple[int,int,int,int]]) -> None:
    scene=trimesh.Scene()
    for name,mesh,material in entries:
        scene.add_geometry(pbr(mesh,material,colors),node_name=name,geom_name=name)
    path.parent.mkdir(parents=True,exist_ok=True); scene.export(path)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return path


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def zip_tree(source: Path, output: Path, *, prefix: str|None=None) -> Path:
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                rel=path.relative_to(source); arc=Path(prefix)/rel if prefix else rel
                archive.write(path,arc.as_posix())
    return output


def write_checksum(path: Path) -> Path:
    out=path.with_suffix(path.suffix+".sha256"); out.write_text(f"{sha256(path)}  {path.name}\n",encoding="utf-8"); return out


def available_font(size: int):
    candidates=["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    for candidate in candidates:
        p=Path(candidate)
        if p.exists(): return ImageFont.truetype(str(p),size)
    return ImageFont.load_default()
