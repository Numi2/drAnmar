#!/usr/bin/env python3
"""Validate the DrAnmar wound-preparation release or installed overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

CATALOG_SUBPATH = Path("Props/SurgicalPreparation/WoundPreparationRobot")
PRIMARY_USDA_COUNT = 9
PRIMARY_GLB_COUNT = 16
PRIMARY_TEXTURE_COUNT = 10
PACKAGE_PNG_COUNT = 22
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ValidationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def eligible_manifest_files(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        if (
            path.is_file()
            and "_repo_overlay" not in path.parts
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and path.name not in {".DS_Store", "asset_manifest.json"}
            and not path.name.endswith(".zip")
        ):
            result.add(path.relative_to(root).as_posix())
    return result


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
    for entry in entries:
        relative = entry["path"]
        if not package_mode:
            installed_overlay_path = (
                relative.startswith("source/")
                or relative.startswith("physics_next/")
                or relative.startswith("docs/")
                or relative.startswith("examples/")
                or relative.startswith("tests/")
                or relative
                in {
                    "scripts/generate_dranmar_wound_preparation_robot.py",
                    "scripts/requirements_wound_preparation_generation.txt",
                    "scripts/validate_dranmar_wound_preparation_robot.py",
                }
            )
            if not installed_overlay_path:
                continue
        path = root / relative
        require(path.is_file(), f"manifest file is missing: {relative}")
        require(path.stat().st_size == entry["bytes"], f"manifest byte count mismatch: {relative}")
        require(sha256(path) == entry["sha256"], f"manifest hash mismatch: {relative}")
        checked += 1

    if package_mode:
        expected = eligible_manifest_files(root)
        require(set(paths) == expected, "manifest coverage differs from package payload")
    require(checked > 0, "manifest validation checked no files")
    return checked


def validate_glb(path: Path) -> None:
    data = path.read_bytes()
    require(len(data) >= 20, f"GLB is too short: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", data)
    require(magic == b"glTF", f"invalid GLB magic: {path}")
    require(version == 2, f"unsupported GLB version in {path}: {version}")
    require(declared_length == len(data), f"GLB length mismatch: {path}")
    offset = 12
    chunks: list[tuple[int, bytes]] = []
    while offset < len(data):
        require(offset + 8 <= len(data), f"truncated GLB chunk header: {path}")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        require(chunk_length % 4 == 0, f"unaligned GLB chunk: {path}")
        require(offset + chunk_length <= len(data), f"truncated GLB chunk: {path}")
        chunks.append((chunk_type, data[offset:offset + chunk_length]))
        offset += chunk_length
    require(offset == len(data) and chunks, f"invalid GLB chunk table: {path}")
    require(chunks[0][0] == 0x4E4F534A, f"first GLB chunk is not JSON: {path}")
    document = json.loads(chunks[0][1].rstrip(b" \x00").decode("utf-8"))
    require(document.get("asset", {}).get("version") == "2.0", f"invalid glTF asset version: {path}")
    require(document.get("scenes"), f"GLB has no scene: {path}")
    require(document.get("nodes"), f"GLB has no nodes: {path}")
    require(document.get("meshes"), f"GLB has no meshes: {path}")


def validate_png(path: Path) -> None:
    data = path.read_bytes()
    require(data.startswith(PNG_SIGNATURE), f"invalid PNG signature: {path}")
    require(len(data) >= 24 and data[12:16] == b"IHDR", f"missing PNG IHDR: {path}")
    width, height = struct.unpack(">II", data[16:24])
    require(width > 0 and height > 0, f"invalid PNG dimensions: {path}")


def validate_contracts(asset_root: Path) -> None:
    frames = json.loads((asset_root / "interaction_frames.json").read_text(encoding="utf-8"))
    frame_names = set(frames["frames"])
    require("wound_preparation_tcp" in frame_names, "TCP frame is missing")
    require("suction_throat" in frame_names, "suction throat frame is missing")
    require("debridement_contact" in frame_names, "debridement contact frame is missing")

    task = json.loads((asset_root / "wound_preparation_task_contract.json").read_text(encoding="utf-8"))
    require(
        task.get("sequence")
        == ["inspect", "contact", "pre_rinse", "aspirate", "debride", "post_rinse", "dry", "verify"],
        "canonical procedure sequence differs from the contract",
    )
    require(
        "clinical debridement efficacy" in task.get("blocked_claims", []),
        "task contract does not block clinical efficacy claims",
    )

    profile = json.loads((asset_root / "physics_profile.json").read_text(encoding="utf-8"))
    require(profile.get("clinical_validation") is False, "physics profile overclaims clinical validation")
    require(profile.get("medical_device") is False, "physics profile marks the asset as a medical device")


def validate_python(root: Path) -> int:
    candidates = (
        root / "scripts/generate_dranmar_wound_preparation_robot.py",
        root / "scripts/install_into_dranmar.py",
        root / "scripts/validate_dranmar_wound_preparation_robot.py",
        root / "examples/franka_wound_preparation_scene.py",
        root / "examples/validate_wound_preparation_runtime.py",
        root / "tests/test_wound_preparation_robot.py",
        root / "source/extensions/orbit.surgical.assets/orbit/surgical/assets/wound_preparation_robot.py",
    )
    checked = 0
    for path in candidates:
        if path.is_file():
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            checked += 1
    require(checked >= 4, "too few Python integration surfaces were found")
    return checked


def validate_usda(asset_root: Path, *, require_checker: bool) -> dict[str, Any]:
    checker = shutil.which("usdchecker")
    if checker is None:
        if require_checker:
            raise ValidationError("usdchecker is required but was not found")
        return {"status": "skipped", "reason": "usdchecker_not_found", "files": 0}
    files = sorted(asset_root.glob("*.usda"))
    for path in files:
        result = subprocess.run(
            [checker, str(path)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            details = (result.stdout + "\n" + result.stderr).strip()
            raise ValidationError(f"OpenUSD validation failed for {path.name}:\n{details[-6000:]}")
    return {"status": "passed", "checker": checker, "files": len(files)}


def validate(root: Path, *, require_checker: bool) -> dict[str, Any]:
    package_asset_root = root / "assets" / CATALOG_SUBPATH
    installed_asset_root = (
        root
        / "source/extensions/orbit.surgical.assets/data"
        / CATALOG_SUBPATH
    )
    package_mode = package_asset_root.is_dir()
    asset_root = package_asset_root if package_mode else installed_asset_root
    require(asset_root.is_dir(), f"wound-preparation catalog is missing under {root}")

    usda_files = sorted(asset_root.glob("*.usda"))
    glb_files = sorted((asset_root / "glb").glob("*.glb"))
    texture_files = sorted((asset_root / "textures").glob("*.png"))
    require(len(usda_files) == PRIMARY_USDA_COUNT, f"expected {PRIMARY_USDA_COUNT} USDA assets")
    require(len(glb_files) == PRIMARY_GLB_COUNT, f"expected {PRIMARY_GLB_COUNT} GLB exports")
    require(len(texture_files) == PRIMARY_TEXTURE_COUNT, f"expected {PRIMARY_TEXTURE_COUNT} textures")

    for path in glb_files:
        validate_glb(path)
    for path in texture_files:
        validate_png(path)

    if package_mode:
        require(installed_asset_root.is_dir(), "install-ready source asset mirror is missing")
        left = {
            path.relative_to(package_asset_root): sha256(path)
            for path in package_asset_root.rglob("*")
            if path.is_file()
        }
        right = {
            path.relative_to(installed_asset_root): sha256(path)
            for path in installed_asset_root.rglob("*")
            if path.is_file()
        }
        require(left == right, "catalog and install-ready source asset trees differ")
        all_png = sorted(
            path for path in root.rglob("*.png")
            if "_repo_overlay" not in path.parts
        )
        require(len(all_png) == PACKAGE_PNG_COUNT, f"expected {PACKAGE_PNG_COUNT} packaged PNG files")
        for path in all_png:
            validate_png(path)
        require(
            (root / "previews/dranmar_wound_preparation_robot_preview.png").is_file(),
            "sequence preview is missing",
        )
        require(
            (root / "previews/dranmar_wound_preparation_robot_full_arm_preview.png").is_file(),
            "full-arm preview is missing",
        )

    manifest_checked = validate_manifest(
        root,
        asset_root / "asset_manifest.json",
        package_mode,
    )
    validate_contracts(asset_root)
    python_checked = validate_python(root)
    usd_result = validate_usda(asset_root, require_checker=require_checker)

    bytecode_candidates = root.rglob("*.pyc") if package_mode else ()
    forbidden = [
        path.relative_to(root).as_posix()
        for path in bytecode_candidates
        if "_repo_overlay" not in path.parts
    ]
    require(not forbidden, "non-portable Python bytecode is present: " + ", ".join(forbidden))

    return {
        "schema": "dr.anmar.wound-preparation-validation.v1",
        "root": str(root),
        "mode": "development_package" if package_mode else "installed_overlay",
        "status": "passed",
        "inventory": {
            "primary_usda": len(usda_files),
            "primary_glb": len(glb_files),
            "primary_textures": len(texture_files),
            "png_files_in_scope": (
                len(
                    [
                        path for path in root.rglob("*.png")
                        if "_repo_overlay" not in path.parts
                    ]
                )
                if package_mode
                else len(texture_files)
            ),
        },
        "manifest_files_checked": manifest_checked,
        "python_files_checked": python_checked,
        "openusd": usd_result,
        "clinical_validation": False,
        "medical_device": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="development package or installed DrAnmar repository root",
    )
    parser.add_argument(
        "--require-usdchecker",
        action="store_true",
        help="fail instead of skipping OpenUSD parsing when usdchecker is unavailable",
    )
    args = parser.parse_args()
    try:
        result = validate(args.root.resolve(), require_checker=args.require_usdchecker)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ValidationError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
