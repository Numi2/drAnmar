#!/usr/bin/env python3
"""Static release validator for the DrAnmar adaptive-hemostasis package."""
from __future__ import annotations

import argparse
import compileall
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zipfile

CATALOG = Path("Props/SurgicalHemostasis/AdaptiveHemostasisRobot")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_glb(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise AssertionError(f"invalid GLB header: {path}")
    version, length = struct.unpack_from("<II", data, 4)
    if version != 2 or length != len(data):
        raise AssertionError(f"invalid GLB container: {path}")


def validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise AssertionError(f"bad ZIP member {bad} in {path}")
        names = archive.namelist()
        forbidden = [
            name for name in names
            if "__pycache__" in name or name.endswith(".pyc") or name.endswith(".DS_Store")
        ]
        if forbidden:
            raise AssertionError(f"forbidden ZIP members in {path}: {forbidden}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--require-usdchecker", action="store_true")
    parser.add_argument("--skip-archives", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    package_mode = (root / "assets" / CATALOG).is_dir()
    asset_root = (
        root / "assets" / CATALOG
        if package_mode
        else root / "source/extensions/orbit.surgical.assets/data" / CATALOG
    )
    if not asset_root.is_dir():
        raise SystemExit(f"adaptive-hemostasis catalog missing under {root}")

    usda = sorted(asset_root.glob("*.usda"))
    glb = sorted((asset_root / "glb").glob("*.glb"))
    textures = sorted((asset_root / "textures").glob("*.png"))
    if (len(usda), len(glb), len(textures)) != (8, 16, 6):
        raise AssertionError(
            f"unexpected asset counts USDA/GLB/textures={(len(usda),len(glb),len(textures))}"
        )
    for path in glb:
        validate_glb(path)
    for path in textures:
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise AssertionError(f"invalid PNG: {path}")

    usdchecker = shutil.which("usdchecker")
    if args.require_usdchecker and not usdchecker:
        raise AssertionError("usdchecker is required but unavailable")
    if usdchecker:
        for path in usda:
            result = subprocess.run(
                [usdchecker, str(path)], text=True, capture_output=True
            )
            if result.returncode:
                raise AssertionError(
                    f"usdchecker failed for {path}:\n{result.stdout}\n{result.stderr}"
                )

    for path in root.rglob("*.py"):
        if package_mode or any(
            part in {
                "adaptive_hemostasis_robot.py",
                "generate_dranmar_adaptive_hemostasis_robot.py",
                "install_into_dranmar.py",
                "validate_dranmar_adaptive_hemostasis_robot.py",
                "validate_adaptive_hemostasis_runtime.py",
                "test_adaptive_hemostasis_robot.py",
            }
            for part in path.parts
        ):
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    manifest_path = asset_root / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("asset") != "dranmar-adaptive-hemostasis-robot-v1":
        raise AssertionError("wrong manifest asset id")
    if package_mode:
        for row in manifest["files"]:
            path = root / row["path"]
            if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise AssertionError(f"manifest mismatch: {row['path']}")
        mirror = root / "source/extensions/orbit.surgical.assets/data" / CATALOG
        for source in asset_root.rglob("*"):
            if source.is_file():
                target = mirror / source.relative_to(asset_root)
                if not target.is_file() or sha256(source) != sha256(target):
                    raise AssertionError(f"extension mirror mismatch: {source.name}")
        junk = [
            path for path in root.rglob("*")
            if path.is_file()
            and ("__pycache__" in path.parts or path.suffix == ".pyc" or path.name == ".DS_Store")
        ]
        if junk:
            raise AssertionError(f"release junk present: {junk}")
        if not args.skip_archives:
            parent = root.parent
            for name in (
                "dranmar_adaptive_hemostasis_robot_v0.1.0.zip",
                "dranmar_adaptive_hemostasis_robot_catalog_v0.1.0.zip",
                "dranmar_adaptive_hemostasis_robot_repo_overlay_v0.1.0.zip",
            ):
                validate_zip(parent / name)

    task = json.loads((asset_root / "adaptive_hemostasis_task_contract.json").read_text())
    phases = task.get("phases") or task.get("sequence")
    if not phases or len(phases) != 11:
        raise AssertionError("task contract must expose exactly 11 phases")
    report = {
        "validated": True,
        "mode": "package" if package_mode else "installed",
        "primary_usda": len(usda),
        "glb": len(glb),
        "textures": len(textures),
        "manifest_entries": len(manifest["files"]),
        "usdchecker": bool(usdchecker),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
