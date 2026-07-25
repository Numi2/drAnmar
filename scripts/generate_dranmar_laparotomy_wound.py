#!/usr/bin/env python3
"""Generate the full-thickness Dr.Anmar midline laparotomy wound-edge asset.

The asset is deliberately separate from the whole-torso inspection meshes.
It supplies two independently addressable, explicit TetMesh wound edges for
each abdominal layer so an exposure tool can lift and retract the actual
laparotomy margins without inventing a removable central tissue plug.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "source/extensions/orbit.surgical.assets/data/Props/Patients"
    / "DynamicAbdominalPatient/anatomy/dranmar_laparotomy_wound.usda"
)
MANIFEST = OUTPUT.parents[1] / "asset_manifest.json"
MANIFEST_PATH = (
    "assets/Props/Patients/DynamicAbdominalPatient/anatomy/"
    "dranmar_laparotomy_wound.usda"
)

X_SAMPLES = 7
Y_SAMPLES = 25
INNER_HALF_GAP_M = 0.032
OUTER_HALF_WIDTH_M = 0.170
HALF_LENGTH_M = 0.105

LAYERS = (
    {
        "id": "skin",
        "display": "Skin",
        "center_z": 0.105,
        "thickness": 0.006,
        "lip_lift": 0.012,
        "youngs_modulus": 75_000.0,
        "poissons_ratio": 0.46,
        "density": 1_080.0,
        "color": (0.73, 0.34, 0.31),
    },
    {
        "id": "subcutaneous_fat",
        "display": "SubcutaneousFat",
        "center_z": 0.088,
        "thickness": 0.024,
        "lip_lift": 0.010,
        "youngs_modulus": 18_000.0,
        "poissons_ratio": 0.47,
        "density": 920.0,
        "color": (0.91, 0.73, 0.18),
    },
    {
        "id": "fascia",
        "display": "Fascia",
        "center_z": 0.071,
        "thickness": 0.003,
        "lip_lift": 0.008,
        "youngs_modulus": 850_000.0,
        "poissons_ratio": 0.44,
        "density": 1_120.0,
        "color": (0.87, 0.86, 0.78),
    },
    {
        "id": "abdominal_wall",
        "display": "AbdominalWall",
        "center_z": 0.052,
        "thickness": 0.028,
        "lip_lift": 0.007,
        "youngs_modulus": 160_000.0,
        "poissons_ratio": 0.46,
        "density": 1_060.0,
        "color": (0.55, 0.12, 0.10),
    },
    {
        "id": "peritoneum",
        "display": "Peritoneum",
        "center_z": 0.034,
        "thickness": 0.0016,
        "lip_lift": 0.005,
        "youngs_modulus": 1_100_000.0,
        "poissons_ratio": 0.45,
        "density": 1_080.0,
        "color": (0.85, 0.54, 0.50),
    },
)


def _fmt(value: float) -> str:
    if abs(value) < 5.0e-10:
        return "0"
    return f"{value:.9g}"


def _vec3(value: tuple[float, float, float]) -> str:
    return "(" + ", ".join(_fmt(component) for component in value) + ")"


def _determinant(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    d: tuple[float, float, float],
) -> float:
    ux, uy, uz = (b[i] - a[i] for i in range(3))
    vx, vy, vz = (c[i] - a[i] for i in range(3))
    wx, wy, wz = (d[i] - a[i] for i in range(3))
    return (
        ux * (vy * wz - vz * wy)
        - uy * (vx * wz - vz * wx)
        + uz * (vx * wy - vy * wx)
    )


def _mesh(
    side: str,
    *,
    center_z: float,
    thickness: float,
    lip_lift: float,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int]],
]:
    sign = -1.0 if side == "Left" else 1.0
    points: list[tuple[float, float, float]] = []
    for z_index in range(2):
        for y_index in range(Y_SAMPLES):
            y_fraction = y_index / (Y_SAMPLES - 1)
            y = -HALF_LENGTH_M + 2.0 * HALF_LENGTH_M * y_fraction
            longitudinal = math.sin(math.pi * y_fraction) ** 0.70
            for x_index in range(X_SAMPLES):
                outward = x_index / (X_SAMPLES - 1)
                x = sign * (
                    INNER_HALF_GAP_M
                    + (OUTER_HALF_WIDTH_M - INNER_HALF_GAP_M) * outward
                )
                crown = lip_lift * (1.0 - outward) ** 2 * longitudinal
                z_top = center_z + thickness * 0.5 + crown
                z = z_top - thickness if z_index == 0 else z_top
                points.append((x, y, z))

    def index(x_index: int, y_index: int, z_index: int) -> int:
        return (
            z_index * Y_SAMPLES * X_SAMPLES
            + y_index * X_SAMPLES
            + x_index
        )

    tets: list[tuple[int, int, int, int]] = []
    for y_index in range(Y_SAMPLES - 1):
        for x_index in range(X_SAMPLES - 1):
            v000 = index(x_index, y_index, 0)
            v100 = index(x_index + 1, y_index, 0)
            v010 = index(x_index, y_index + 1, 0)
            v110 = index(x_index + 1, y_index + 1, 0)
            v001 = index(x_index, y_index, 1)
            v101 = index(x_index + 1, y_index, 1)
            v011 = index(x_index, y_index + 1, 1)
            v111 = index(x_index + 1, y_index + 1, 1)
            cell = (
                (v000, v100, v110, v111),
                (v000, v110, v010, v111),
                (v000, v010, v011, v111),
                (v000, v011, v001, v111),
                (v000, v001, v101, v111),
                (v000, v101, v100, v111),
            )
            for tet in cell:
                if _determinant(*(points[value] for value in tet)) < 0.0:
                    tet = (tet[0], tet[2], tet[1], tet[3])
                tets.append(tet)

    faces: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    counts: Counter[tuple[int, int, int]] = Counter()
    for a, b, c, d in tets:
        for face in ((a, c, b), (a, b, d), (a, d, c), (b, c, d)):
            key = tuple(sorted(face))
            counts[key] += 1
            faces[key] = face
    surface = [faces[key] for key, count in counts.items() if count == 1]
    return points, tets, surface


def _array(values: list[str], *, indent: str = "                    ") -> str:
    return ",\n".join(f"{indent}{value}" for value in values)


def _material(layer: dict[str, object]) -> str:
    color = layer["color"]
    return f'''        def Material "{layer["display"]}"
        {{
            token outputs:surface.connect = <PreviewSurface.outputs:surface>
            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {_vec3(color)}
                float inputs:metallic = 0
                float inputs:roughness = 0.58
                token outputs:surface
            }}
        }}'''


def _edge(layer: dict[str, object], side: str) -> str:
    points, tets, surface = _mesh(
        side,
        center_z=float(layer["center_z"]),
        thickness=float(layer["thickness"]),
        lip_lift=float(layer["lip_lift"]),
    )
    point_text = _array([_vec3(point) for point in points])
    tet_text = _array(
        ["(" + ", ".join(str(value) for value in tet) + ")" for tet in tets]
    )
    indices = [str(value) for face in surface for value in face]
    index_text = _array(indices)
    counts_text = _array(["3"] * len(surface))
    material_path = (
        f"</DrAnmarLaparotomyWound/Looks/{layer['display']}>"
    )
    mass = (
        float(layer["density"])
        * (OUTER_HALF_WIDTH_M - INNER_HALF_GAP_M)
        * (2.0 * HALF_LENGTH_M)
        * float(layer["thickness"])
    )
    return f'''            def Xform "{side}Edge"
            {{
                custom string drAnmar:role = "{side.lower()}_full_thickness_laparotomy_edge"
                custom string drAnmar:mechanics = "explicit_tetmesh_volume"
                custom float drAnmar:massKg = {_fmt(mass)}
                custom float drAnmar:youngsModulusPaSeed = {_fmt(float(layer["youngs_modulus"]))}
                custom float drAnmar:poissonsRatioSeed = {_fmt(float(layer["poissons_ratio"]))}
                custom bool drAnmar:clinicalValidation = false
                def Xform "Geometry"
                {{
                    def TetMesh "SimulationTetMesh"
                    {{
                        custom string drAnmar:role = "volume_deformable_simulation_mesh"
                        uniform token purpose = "guide"
                        token visibility = "invisible"
                        point3f[] points = [
{point_text}
                        ]
                        int4[] tetVertexIndices = [
{tet_text}
                        ]
                    }}
                    def Mesh "Visual"
                    {{
                        custom string drAnmar:role = "retracted_laparotomy_wound_edge"
                        uniform token subdivisionScheme = "none"
                        bool doubleSided = true
                        rel material:binding = {material_path}
                        int[] faceVertexCounts = [
{counts_text}
                        ]
                        int[] faceVertexIndices = [
{index_text}
                        ]
                        point3f[] points = [
{point_text}
                        ]
                    }}
                }}
                def Scope "Frames"
                {{
                    def Xform "capture_center"
                    {{
                        custom string drAnmar:role = "{side.lower()}_wound_edge_capture_center"
                        double3 xformOp:translate = ({_fmt((-1.0 if side == "Left" else 1.0) * INNER_HALF_GAP_M)}, 0, {_fmt(float(layer["center_z"]) + float(layer["lip_lift"]))})
                        uniform token[] xformOpOrder = ["xformOp:translate"]
                    }}
                }}
            }}'''


def build_usda() -> str:
    materials = "\n".join(_material(layer) for layer in LAYERS)
    layers = []
    for layer in LAYERS:
        layers.append(
            f'''        def Xform "{layer["id"]}"
        {{
{_edge(layer, "Left")}
{_edge(layer, "Right")}
        }}'''
        )
    return f'''#usda 1.0
(
    defaultPrim = "DrAnmarLaparotomyWound"
    doc = "Full-thickness, bilateral midline laparotomy wound edges for Dr.Anmar surgical exposure."
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "DrAnmarLaparotomyWound"
{{
    custom string drAnmar:assetId = "dranmar-laparotomy-wound-v1"
    custom string drAnmar:incisionType = "median_laparotomy"
    custom string drAnmar:representation = "bilateral_layered_explicit_tetmesh_wound_edges"
    custom string drAnmar:status = "engineering_research_geometry_with_provisional_material_parameters"
    custom bool drAnmar:clinicalValidation = false
    custom bool drAnmar:medicalDevice = false
    def Scope "Looks"
    {{
{materials}
    }}
    def Scope "Layers"
    {{
{chr(10).join(layers)}
    }}
}}
'''


def _refresh_manifest() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = {
        "bytes": OUTPUT.stat().st_size,
        "path": MANIFEST_PATH,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    files = [
        current
        for current in payload["files"]
        if current["path"] != MANIFEST_PATH
    ]
    files.append(entry)
    payload["files"] = sorted(files, key=lambda current: current["path"])
    MANIFEST.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_usda(), encoding="utf-8")
    _refresh_manifest()
    print(OUTPUT)


if __name__ == "__main__":
    main()
