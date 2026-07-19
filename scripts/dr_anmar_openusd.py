# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Build clean Dr.Anmar OpenUSD compositions from the installed SUFIA assets.

The upstream archives contain useful organ, room, and ceiling USD layers, but
their ``main_scene.usd`` entry points are not self-contained.  This module
creates deterministic Z-up, metre-scale compositions with only resolvable USD
dependencies.  The runtime environment intentionally excludes the table and
anatomy because ORBIT-Surgical owns their physics; the standalone composition
contains the complete visual digital twin for inspection and export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "dr.anmar.openusd-composition.v1"


def usd_asset_path(asset: Path, layer_dir: Path) -> str:
    return Path(os.path.relpath(asset, layer_dir)).as_posix().replace("@", "%40")


def find_one(scene_root: Path, relative_tail: str) -> Path:
    matches = sorted(path.resolve() for path in scene_root.rglob(Path(relative_tail).name) if path.as_posix().endswith(relative_tail))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {relative_tail} below {scene_root}; found {len(matches)}")
    return matches[0]


def write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitized_geometry(source: Path, cache_root: Path) -> Path:
    """Resolve the deterministic geometry cache created by Isaac Sim."""

    source_hash = file_sha256(source)
    destination = cache_root / f"{source_hash}.usdc"
    if not destination.is_file():
        raise RuntimeError(
            f"Sanitized OpenUSD geometry is missing for {source}; run scripts/dr_anmar_geometry_sanitize.py first"
        )
    return destination


def layer_header(scene_id: str, default_prim: str) -> str:
    return f'''#usda 1.0
(
    defaultPrim = "{default_prim}"
    metersPerUnit = 1
    upAxis = "Z"
    customLayerData = {{
        string drAnmarSchema = "{SCHEMA}"
        string drAnmarSceneId = "{scene_id}"
        string drAnmarUsage = "simulation and research only"
    }}
)
'''


def environment_layer(scene_id: str, room: str, ceiling: str) -> str:
    return layer_header(scene_id, "DrAnmarEnvironment") + f'''
def Xform "DrAnmarEnvironment" (
    kind = "assembly"
)
{{
    custom string drAnmar:sceneId = "{scene_id}"
    custom string drAnmar:sourceAxis = "Y"
    custom double drAnmar:sourceMetersPerUnit = 0.01

    def Scope "Materials"
    {{
        def Material "RoomFinish"
        {{
            token outputs:surface.connect = </DrAnmarEnvironment/Materials/RoomFinish/Shader.outputs:surface>
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.47, 0.55, 0.58)
                float inputs:roughness = 0.72
                token outputs:surface
            }}
        }}
        def Material "CeilingLampFinish"
        {{
            token outputs:surface.connect = </DrAnmarEnvironment/Materials/CeilingLampFinish/Shader.outputs:surface>
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.84, 0.89, 0.92)
                color3f inputs:emissiveColor = (0.34, 0.40, 0.46)
                float inputs:roughness = 0.32
                token outputs:surface
            }}
        }}
    }}

    def Xform "Architecture"
    {{
        double3 xformOp:translate = (0, 0, -0.95)
        double3 xformOp:rotateXYZ = (90, 0, 0)
        double3 xformOp:scale = (0.01, 0.01, 0.01)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]

        def Xform "Room" (
            prepend references = @{room}@</World>
        )
        {{
            custom string drAnmar:component = "operating_room"
            rel material:binding = </DrAnmarEnvironment/Materials/RoomFinish>
        }}

        def Xform "CeilingLamps" (
            prepend references = @{ceiling}@</World>
        )
        {{
            custom string drAnmar:component = "ceiling_lamps"
            rel material:binding = </DrAnmarEnvironment/Materials/CeilingLampFinish>
        }}
    }}
}}
'''


def composed_layer(scene_id: str, environment: str, organs: str, table: str) -> str:
    return layer_header(scene_id, "DrAnmarDigitalTwin") + f'''
def Xform "DrAnmarDigitalTwin" (
    kind = "assembly"
)
{{
    custom string drAnmar:sceneId = "{scene_id}"
    custom string drAnmar:fidelity = "visual anatomy plus ORBIT-Surgical rigid-body task physics"

    def Xform "Environment" (
        prepend references = @{environment}@</DrAnmarEnvironment>
    )
    {{
        custom string drAnmar:component = "repaired_openusd_environment"
    }}

    def Xform "SurgicalTable" (
        prepend references = @{table}@</Table>
    )
    {{
        double3 xformOp:translate = (0, 0, -0.457)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        custom string drAnmar:component = "orbit_surgical_table"
    }}

    def Xform "Anatomy" (
        prepend references = @{organs}@</root>
    )
    {{
        double3 xformOp:translate = (-0.117, -0.0945, -0.144)
        float3 xformOp:scale = (0.35, 0.35, 0.35)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        custom string drAnmar:component = "official_sufia_anatomy"
    }}

    def Scope "SceneCameras"
    {{
        def Camera "EndoscopeOverview"
        {{
            float focalLength = 22
            float focusDistance = 0.25
            float horizontalAperture = 20.955
            float2 clippingRange = (0.01, 2)
            double3 xformOp:translate = (0.45, 0.25, 0.28)
            double3 xformOp:rotateXYZ = (67, 0, 119)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
            custom string drAnmar:cameraRole = "clinical_overview"
        }}

        def Camera "RoomOverview"
        {{
            float focalLength = 24
            float focusDistance = 2
            float horizontalAperture = 20.955
            float2 clippingRange = (0.05, 20)
            double3 xformOp:translate = (2.2, 2.4, 1.8)
            double3 xformOp:rotateXYZ = (67, 0, 137)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
            custom string drAnmar:cameraRole = "operating_room_overview"
        }}
    }}

    def Scope "Lighting"
    {{
        def DomeLight "AmbientFill"
        {{
            float inputs:intensity = 750
            color3f inputs:color = (0.82, 0.88, 1)
            custom string drAnmar:lightRole = "standalone_preview_only"
        }}
    }}
}}
'''


def compose_scene(scene_root: Path, scene_id: str, output_root: Path, table: Path) -> dict[str, Any]:
    organs = find_one(scene_root, "models/organs/models_topo_blender.usdc")
    room = find_one(scene_root, "models/operating_room/room/Over_GRP_Room_Additions_merged.usd")
    ceiling = find_one(
        scene_root,
        "models/operating_room/ceiling_lamps/Over_GRP_CeilingLamps_merged.usd",
    )
    sanitized_room = sanitized_geometry(room, output_root / "_geometry")
    sanitized_ceiling = sanitized_geometry(ceiling, output_root / "_geometry")
    sanitized_organs = sanitized_geometry(organs, output_root / "_geometry")
    sanitized_table = sanitized_geometry(table, output_root / "_geometry")
    destination = output_root / scene_id
    destination.mkdir(parents=True, exist_ok=True)
    environment = destination / "environment.usda"
    composed = destination / "dr_anmar_digital_twin.usda"
    write_if_changed(
        environment,
        environment_layer(
            scene_id,
            usd_asset_path(sanitized_room, destination),
            usd_asset_path(sanitized_ceiling, destination),
        ),
    )
    write_if_changed(
        composed,
        composed_layer(
            scene_id,
            environment.name,
            usd_asset_path(sanitized_organs, destination),
            usd_asset_path(sanitized_table, destination),
        ),
    )
    sources = {
        "organs": organs,
        "room": room,
        "ceiling_lamps": ceiling,
        "table": table,
        "sanitized_room": sanitized_room,
        "sanitized_ceiling_lamps": sanitized_ceiling,
        "sanitized_organs": sanitized_organs,
        "sanitized_table": sanitized_table,
    }
    return {
        "id": scene_id,
        "environment_usd": str(environment),
        "composed_usd": str(composed),
        "runtime_organ_usd": str(sanitized_organs),
        "sources": {
            name: {"path": str(path), "bytes": path.stat().st_size}
            for name, path in sources.items()
        },
        "source_entrypoint_replaced": str(next(iter(sorted(scene_root.rglob("main_scene.usd"))), "")),
        "runtime_ownership": {
            "environment": "Dr.Anmar repaired OpenUSD layer",
            "table_robot_object_physics": "ORBIT-Surgical task",
            "anatomy_visual": "official SUFIA OpenUSD organ layer",
        },
    }


def compose_library(anatomy_root: Path, output_root: Path, table: Path) -> dict[str, Any]:
    if not anatomy_root.is_dir():
        raise FileNotFoundError(f"Anatomy root does not exist: {anatomy_root}")
    if not table.is_file():
        raise FileNotFoundError(f"ORBIT-Surgical table asset does not exist: {table}")
    scenes = []
    for scene_root in sorted(path for path in anatomy_root.iterdir() if path.is_dir()):
        scene_id = scene_root.name
        try:
            scenes.append(compose_scene(scene_root, scene_id, output_root, table))
        except RuntimeError as exc:
            scenes.append({"id": scene_id, "error": str(exc)})
    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "anatomy_root": str(anatomy_root.resolve()),
        "output_root": str(output_root.resolve()),
        "scene_count": len(scenes),
        "ready_count": sum("error" not in scene for scene in scenes),
        "scenes": scenes,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()
    parser = argparse.ArgumentParser(description="Compose the installed Dr.Anmar OpenUSD scene library.")
    parser.add_argument("--anatomy_root", type=Path, default=data_root / "assets/sufia_bc")
    parser.add_argument("--output_root", type=Path, default=data_root / "scenes/openusd")
    parser.add_argument(
        "--table",
        type=Path,
        default=repo_root / "source/extensions/orbit.surgical.assets/data/Props/Table/table.usd",
    )
    args = parser.parse_args()
    payload = compose_library(
        args.anatomy_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        args.table.expanduser().resolve(),
    )
    print(json.dumps({"schema": payload["schema"], "scene_count": payload["scene_count"], "ready_count": payload["ready_count"]}))
    if payload["ready_count"] != payload["scene_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
