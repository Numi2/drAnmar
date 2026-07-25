#!/usr/bin/env python3
"""Validate the DrAnmar atraumatic-exposure release or installed overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

CATALOG_SUBPATH = Path("Props/SurgicalExposure/AtraumaticExposureRobot")
PRIMARY_USDA_COUNT = 6
PRIMARY_GLB_COUNT = 13
PRIMARY_TEXTURE_COUNT = 8
PACKAGE_PNG_COUNT = 18
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_manifest_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "_repo_overlay" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name not in {".DS_Store", "asset_manifest.json"}
        and not path.name.endswith(".zip")
    }


def validate_manifest(root: Path, manifest_path: Path, package_mode: bool) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "dr.anmar.asset-manifest.v1", "unexpected manifest schema")
    entries = manifest.get("files")
    require(isinstance(entries, list) and entries, "manifest has no file entries")
    paths = [entry.get("path") for entry in entries]
    require(len(paths) == len(set(paths)), "manifest contains duplicate paths")
    require(
        not any(
            not isinstance(path, str)
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or "__pycache__" in Path(path).parts
            or Path(path).suffix == ".pyc"
            for path in paths
        ),
        "manifest contains an unsafe or non-portable path",
    )

    checked = 0
    installed_names = {
        "scripts/generate_dranmar_atraumatic_exposure_robot.py",
        "scripts/requirements_atraumatic_exposure_generation.txt",
        "scripts/validate_dranmar_atraumatic_exposure_robot.py",
    }
    for entry in entries:
        relative = entry["path"]
        if not package_mode and not (
            relative.startswith(("source/", "physics_next/", "docs/", "examples/", "tests/"))
            or relative in installed_names
        ):
            continue
        path = root / relative
        require(path.is_file(), f"manifest file is missing: {relative}")
        require(path.stat().st_size == entry["bytes"], f"manifest byte mismatch: {relative}")
        require(sha256(path) == entry["sha256"], f"manifest hash mismatch: {relative}")
        checked += 1
    if package_mode:
        require(set(paths) == eligible_manifest_files(root), "manifest coverage differs from payload")
    require(checked > 0, "manifest validation checked no files")
    return checked


def validate_glb(path: Path) -> None:
    data = path.read_bytes()
    require(len(data) >= 20, f"GLB is too short: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", data)
    require(magic == b"glTF", f"invalid GLB magic: {path}")
    require(version == 2, f"unsupported GLB version: {path}")
    require(declared_length == len(data), f"GLB length mismatch: {path}")
    offset = 12
    chunk_types: list[int] = []
    while offset < len(data):
        require(offset + 8 <= len(data), f"truncated GLB chunk header: {path}")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        require(chunk_length % 4 == 0, f"unaligned GLB chunk: {path}")
        require(offset + chunk_length <= len(data), f"truncated GLB chunk: {path}")
        chunk_types.append(chunk_type)
        offset += chunk_length
    require(offset == len(data), f"GLB trailing bytes: {path}")
    require(chunk_types and chunk_types[0] == 0x4E4F534A, f"GLB lacks JSON chunk: {path}")
    json_length = struct.unpack_from("<I", data, 12)[0]
    json.loads(data[20 : 20 + json_length].decode("utf-8").rstrip(" \x00"))


def validate_png(path: Path) -> None:
    data = path.read_bytes()
    require(data.startswith(PNG_SIGNATURE), f"invalid PNG signature: {path}")
    require(len(data) >= 33 and data[12:16] == b"IHDR", f"PNG lacks IHDR: {path}")
    width, height = struct.unpack_from(">II", data, 16)
    require(width > 0 and height > 0, f"PNG has invalid dimensions: {path}")
    require(data[-8:-4] == b"IEND", f"PNG lacks terminal IEND: {path}")


def validate_python(path: Path) -> None:
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def run_usdchecker(path: Path, checker: str) -> None:
    result = subprocess.run(
        [checker, str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(result.returncode == 0, f"usdchecker failed for {path}:\n{result.stdout}")


def resolve_layout(root: Path) -> tuple[bool, Path, Path]:
    package_asset = root / "assets" / CATALOG_SUBPATH
    installed_asset = (
        root / "source/extensions/orbit.surgical.assets/data" / CATALOG_SUBPATH
    )
    if package_asset.is_dir():
        return True, package_asset, (
            root / "source/extensions/orbit.surgical.assets/data" / CATALOG_SUBPATH
        )
    require(installed_asset.is_dir(), f"cannot locate catalog below {root}")
    return False, installed_asset, installed_asset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-usdchecker", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    package_mode, asset_root, mirror_root = resolve_layout(root)
    require((asset_root / "asset_manifest.json").is_file(), "asset manifest is missing")

    if package_mode:
        pyc = list(root.rglob("*.pyc"))
        caches = [path for path in root.rglob("__pycache__") if path.is_dir()]
        require(
            not pyc and not caches,
            "packaged bytecode or __pycache__ directories are present",
        )
        require(not list(root.rglob(".DS_Store")), "workstation metadata is present")

    usda = sorted(asset_root.glob("*.usda"))
    glbs = sorted((asset_root / "glb").glob("*.glb"))
    textures = sorted((asset_root / "textures").glob("*.png"))
    require(len(usda) == PRIMARY_USDA_COUNT, f"expected {PRIMARY_USDA_COUNT} USDA, got {len(usda)}")
    require(len(glbs) == PRIMARY_GLB_COUNT, f"expected {PRIMARY_GLB_COUNT} GLB, got {len(glbs)}")
    require(len(textures) == PRIMARY_TEXTURE_COUNT, f"expected {PRIMARY_TEXTURE_COUNT} textures, got {len(textures)}")
    if package_mode:
        require(len(list(root.rglob("*.png"))) == PACKAGE_PNG_COUNT, "unexpected package PNG count")
        asset_files = {
            path.relative_to(asset_root).as_posix(): sha256(path)
            for path in asset_root.rglob("*")
            if path.is_file()
        }
        mirror_files = {
            path.relative_to(mirror_root).as_posix(): sha256(path)
            for path in mirror_root.rglob("*")
            if path.is_file()
        }
        require(asset_files == mirror_files, "asset and extension-data mirrors differ")

    if package_mode:
        json_files = sorted(root.rglob("*.json"))
        png_files = sorted(root.rglob("*.png"))
        python_files = sorted(root.rglob("*.py"))
    else:
        json_files = sorted(asset_root.rglob("*.json")) + [
            root / "physics_next/dr-anmar-assets.json",
            root / "physics_next/surgical-exposure/dranmar-atraumatic-exposure-robot-v1.json",
        ]
        png_files = sorted(asset_root.rglob("*.png"))
        python_files = [
            root / "examples/franka_atraumatic_exposure_scene.py",
            root / "examples/validate_atraumatic_exposure_runtime.py",
            root / "scripts/generate_dranmar_atraumatic_exposure_robot.py",
            root / "scripts/validate_dranmar_atraumatic_exposure_robot.py",
            root
            / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
            / "atraumatic_exposure_robot.py",
            root / "tests/test_atraumatic_exposure_robot.py",
        ]
        require(all(path.is_file() for path in python_files), "installed Python surface is incomplete")
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    for path in glbs:
        validate_glb(path)
    for path in png_files:
        validate_png(path)
    for path in python_files:
        validate_python(path)

    checker = shutil.which("usdchecker")
    if args.require_usdchecker:
        require(checker is not None, "usdchecker is required but unavailable")
    if checker:
        for path in usda:
            run_usdchecker(path, checker)

    manifest_checked = validate_manifest(
        root, asset_root / "asset_manifest.json", package_mode
    )
    required_docs = {
        "README.md",
        "docs/atraumatic_exposure_robot/VALIDATION.md",
        "docs/atraumatic_exposure_robot/FRANKA_INTEGRATION.md",
        "docs/atraumatic_exposure_robot/FORCE_CONTROL.md",
        "docs/atraumatic_exposure_robot/TISSUE_CAPTURE.md",
        "examples/validate_atraumatic_exposure_runtime.py",
        "tests/test_atraumatic_exposure_robot.py",
    }
    if package_mode:
        require(
            all((root / path).is_file() for path in required_docs),
            "release documentation, tests, or runtime validator are missing",
        )

    result = {
        "schema": "dr.anmar.atraumatic-exposure-static-validation.v1",
        "status": "pass",
        "mode": "package" if package_mode else "installed_overlay",
        "primary_usda": len(usda),
        "primary_glb": len(glbs),
        "primary_textures": len(textures),
        "json_files": len(json_files),
        "python_files_compiled": len(python_files),
        "manifest_files_checked": manifest_checked,
        "usdchecker": bool(checker),
        "clinical_validation": False,
        "medical_device": False,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
