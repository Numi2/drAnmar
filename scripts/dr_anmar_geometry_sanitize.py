# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Create dependency-clean OpenUSD geometry layers using the Isaac runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from dr_anmar_openusd import file_sha256


DATA_ROOT = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()
REPO_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Sanitize the installed Dr.Anmar room geometry.")
parser.add_argument("--anatomy_root", type=Path, default=DATA_ROOT / "assets/sufia_bc")
parser.add_argument("--output_root", type=Path, default=DATA_ROOT / "scenes/openusd/_geometry")
parser.add_argument(
    "--table",
    type=Path,
    default=REPO_ROOT / "source/extensions/orbit.surgical.assets/data/Props/Table/table.usd",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Sdf, Usd, UsdShade, UsdUtils


SOURCE_NAMES = {
    "Over_GRP_Room_Additions_merged.usd",
    "Over_GRP_CeilingLamps_merged.usd",
    "models_topo_blender.usdc",
}


def sanitize(source: Path, destination: Path) -> dict[str, object]:
    if destination.is_file():
        _, _, unresolved = UsdUtils.ComputeAllDependencies(str(destination))
        if not unresolved:
            return {"source": str(source), "output": str(destination), "cached": True, "unresolved": []}
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.tmp.usdc")
    temporary.unlink(missing_ok=True)
    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise RuntimeError(f"Could not open source OpenUSD geometry: {source}")
    flattened = source_stage.Flatten()
    if not flattened.Export(str(temporary)):
        raise RuntimeError(f"Could not flatten OpenUSD geometry: {source}")
    clean_stage = Usd.Stage.Open(str(temporary))
    if clean_stage is None:
        raise RuntimeError(f"Could not reopen flattened OpenUSD geometry: {temporary}")
    material_paths = []
    for prim in clean_stage.TraverseAll():
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.GetTypeName() == "NodeGraph":
            material_paths.append(prim.GetPath())
            continue
        for relationship in list(prim.GetRelationships()):
            if relationship.GetName().startswith("material:binding"):
                prim.RemoveProperty(relationship.GetName())
        for attribute in list(prim.GetAttributes()):
            if attribute.GetTypeName() in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                prim.RemoveProperty(attribute.GetName())
    for path in sorted(material_paths, key=lambda value: len(str(value)), reverse=True):
        clean_stage.RemovePrim(path)
    clean_stage.GetRootLayer().Save()
    os.replace(temporary, destination)
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(destination))
    unresolved_strings = [str(path) for path in unresolved]
    if unresolved_strings:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Sanitized geometry still has unresolved dependencies: {unresolved_strings}")
    return {"source": str(source), "output": str(destination), "cached": False, "unresolved": []}


def main() -> None:
    anatomy_root = args.anatomy_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sources = sorted({path.resolve() for path in anatomy_root.rglob("*") if path.is_file() and path.name in SOURCE_NAMES})
    table = args.table.expanduser().resolve()
    if not table.is_file():
        raise RuntimeError(f"ORBIT-Surgical table USD does not exist: {table}")
    sources.append(table)
    if not sources:
        raise RuntimeError(f"No installed operating-room USD geometry found under {anatomy_root}")
    results = []
    completed_hashes: set[str] = set()
    for source in sources:
        source_hash = file_sha256(source)
        if source_hash in completed_hashes:
            continue
        completed_hashes.add(source_hash)
        results.append(sanitize(source, output_root / f"{source_hash}.usdc"))
    print(
        json.dumps(
            {
                "schema": "dr.anmar.openusd-geometry-sanitize.v1",
                "source_count": len(sources),
                "unique_geometry_count": len(results),
                "outputs": results,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
