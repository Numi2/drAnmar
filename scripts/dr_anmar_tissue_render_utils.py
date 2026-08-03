"""Small Isaac/USD helpers shared by real DrAnmar tissue render captures."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import omni.usd
from PIL import Image, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

from dr_anmar_dynamic_curved_cut_fem import TET_FACES


def font(size: int, *, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def rgb_image(value) -> Image.Image:
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
    return Image.fromarray(array)


def boundary_triangles(tetrahedra: np.ndarray) -> np.ndarray:
    counts: dict[tuple[int, int, int], int] = {}
    oriented: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tet in tetrahedra:
        for face in TET_FACES:
            triangle = tuple(int(tet[index]) for index in face)
            key = tuple(sorted(triangle))
            counts[key] = counts.get(key, 0) + 1
            oriented.setdefault(key, triangle)
    return np.asarray([oriented[key] for key, count in counts.items() if count == 1], dtype=np.int32)


def material(path: str, color: tuple[float, float, float], roughness: float, metallic: float = 0.0):
    stage = omni.usd.get_context().get_stage()
    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return result


def mesh(path: str, points: np.ndarray, triangles: np.ndarray, surface_material) -> UsdGeom.Mesh:
    stage = omni.usd.get_context().get_stage()
    result = UsdGeom.Mesh.Define(stage, path)
    result.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    result.CreateFaceVertexCountsAttr().Set([3] * len(triangles))
    result.CreateFaceVertexIndicesAttr().Set(triangles.reshape(-1).tolist())
    result.CreateSubdivisionSchemeAttr().Set("none")
    result.CreateDoubleSidedAttr().Set(True)
    UsdShade.MaterialBindingAPI.Apply(result.GetPrim()).Bind(surface_material)
    return result


def set_points(surface: UsdGeom.Mesh, points: np.ndarray) -> None:
    surface.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))


def set_pose(path: str, translation: np.ndarray, yaw_rad: float = 0.0) -> None:
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    xformable = UsdGeom.Xformable(prim)
    translated = False
    oriented = False
    quaternion = Gf.Quatd(math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad))
    for operation in xformable.GetOrderedXformOps():
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            operation.Set(Gf.Vec3d(*map(float, translation)))
            translated = True
        elif operation.GetOpType() == UsdGeom.XformOp.TypeOrient:
            operation.Set(quaternion)
            oriented = True
    if not translated:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*map(float, translation)))
    if not oriented:
        xformable.AddOrientOp().Set(quaternion)
