#!/usr/bin/env python3
"""Static release gate for the DrAnmar surgical-oncology integration."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


CATALOG = Path("Props/SurgicalOncology/OncoSurgeryCell")
EXPECTED_COUNTS = {
    "usda": 13,
    "glb": 26,
    "png": 8,
    "json": 10,
}
FORBIDDEN_NAMES = {".DS_Store"}


def load_runtime(root: Path):
    path = (
        root
        / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
        / "oncologic_resection.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dranmar_oncology_static_validation", path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_glb(path: Path) -> None:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise AssertionError(f"Invalid GLB header: {path}")
    version, declared_length = struct.unpack_from("<II", data, 4)
    if version != 2 or declared_length != len(data):
        raise AssertionError(
            f"Invalid GLB version/length: {path} "
            f"version={version} declared={declared_length} actual={len(data)}"
        )


def assert_png(path: Path) -> None:
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"Invalid PNG header: {path}")


def authored_mass(path: Path) -> float:
    values = re.findall(
        r"float physics:mass = ([0-9.eE+-]+)",
        path.read_text(encoding="utf-8"),
    )
    return sum(float(value) for value in values)


def validate(root: Path, *, require_usdchecker: bool) -> dict[str, object]:
    root = root.resolve()
    asset_root = (
        root / "source/extensions/orbit.surgical.assets/data" / CATALOG
    )
    if not asset_root.is_dir():
        raise AssertionError(f"Missing catalog directory: {asset_root}")

    usda = sorted(asset_root.glob("*.usda"))
    glb = sorted((asset_root / "glb").glob("*.glb"))
    png = sorted((asset_root / "textures").glob("*.png"))
    json_paths = sorted(asset_root.glob("*.json"))
    counts = {
        "usda": len(usda),
        "glb": len(glb),
        "png": len(png),
        "json": len(json_paths),
    }
    if counts != EXPECTED_COUNTS:
        raise AssertionError(
            f"Catalog counts {counts} do not match {EXPECTED_COUNTS}"
        )
    for path in glb:
        assert_glb(path)
    for path in png:
        assert_png(path)
    payloads = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in json_paths
    }

    forbidden = [
        path
        for path in asset_root.rglob("*")
        if path.name in FORBIDDEN_NAMES
        or "__pycache__" in path.parts
        or path.suffix == ".pyc"
    ]
    if forbidden:
        raise AssertionError(f"Forbidden generated files: {forbidden[:5]}")

    runtime = load_runtime(root)
    task = payloads["oncologic_resection_task_contract.json"]
    manifest = payloads["asset_manifest.json"]
    profile = payloads["physics_profile.json"]
    training = payloads["isaac_lab_training_contract.json"]
    topology = payloads["resection_topology.json"]
    tumor = payloads["tumor_field.json"]
    frames = payloads["interaction_frames.json"]["frames"]
    modalities = payloads["sensor_modalities.json"]["modalities"]

    if tuple(task["phases"]) != runtime.TASK_PHASES:
        raise AssertionError("Runtime phases do not match the task contract")
    if tuple(runtime.PHASE_TARGETS) != runtime.TASK_PHASES:
        raise AssertionError("Phase targets are missing or out of order")
    expected_joints = set(runtime.TOOL_JOINTS.values())
    if len(expected_joints) != 22:
        raise AssertionError("Oncology tool must expose exactly 22 unique joints")
    for phase, targets in runtime.PHASE_TARGETS.items():
        if set(targets) != expected_joints:
            raise AssertionError(f"Incomplete joint targets for {phase}")
        if not all(math.isfinite(value) for value in targets.values()):
            raise AssertionError(f"Non-finite joint target in {phase}")
    if not set(runtime.TOOL_FRAME_PATHS).issubset(frames):
        missing = sorted(set(runtime.TOOL_FRAME_PATHS) - set(frames))
        raise AssertionError(f"Runtime frames absent from contract: {missing}")
    expected_modalities = {
        "rgb_depth",
        "nir_fluorescence",
        "hyperspectral",
        "ultrasound",
        "oct",
        "raman",
    }
    if set(modalities) != expected_modalities:
        raise AssertionError(f"Unexpected sensor modalities: {modalities}")

    if len(tumor["cells"]) != 3028:
        raise AssertionError("Tumor field must contain 3028 active cells")
    if len({cell["id"] for cell in tumor["cells"]}) != len(tumor["cells"]):
        raise AssertionError("Tumor field contains duplicate cell ids")
    if sum(cell["planned_resection"] for cell in tumor["cells"]) != 220:
        raise AssertionError("Planned resection must contain 220 cells")
    if len(topology["bonds"]) != topology["bond_count"] or (
        topology["bond_count"] != 96
    ):
        raise AssertionError("Resection topology bond count mismatch")
    if sum(bond["seal_required"] for bond in topology["bonds"]) != 12:
        raise AssertionError("Topology must contain 12 protected pedicles")
    for bond in topology["bonds"]:
        center = bond.get("center_m")
        normal = bond.get("normal")
        if (
            not isinstance(center, list)
            or len(center) != 3
            or not all(math.isfinite(float(value)) for value in center)
        ):
            raise AssertionError(f"Invalid bond center: {bond['id']}")
        if (
            not isinstance(normal, list)
            or len(normal) != 3
            or abs(
                math.sqrt(sum(float(value) ** 2 for value in normal)) - 1.0
            )
            > 1.0e-5
        ):
            raise AssertionError(f"Invalid bond normal: {bond['id']}")

    standalone_mass = authored_mass(
        asset_root / "dranmar_tumor_resection_tool_standalone.usda"
    )
    payload_mass = authored_mass(
        asset_root / "dranmar_tumor_resection_tool_payload.usda"
    )
    proxy_mass = authored_mass(
        asset_root / "dranmar_tumor_resection_tool_rigid_proxy.usda"
    )
    for name, value in {
        "standalone": standalone_mass,
        "payload": payload_mass,
        "rigid_proxy": proxy_mass,
    }.items():
        if abs(value - 2.5534) > 1.0e-6:
            raise AssertionError(f"{name} mass is inconsistent: {value}")

    if manifest["version"] != "0.2.0":
        raise AssertionError("Integrated manifest version must be 0.2.0")
    if manifest["runtime"] != "orbit.surgical.assets.oncologic_resection":
        raise AssertionError("Manifest runtime route is missing")
    if profile["tool"]["authored_mass_kg"] != 2.5534:
        raise AssertionError("Physics profile mass is inconsistent")
    native_tissue = manifest.get("native_tissue_route", {})
    deformable = training.get("deformable_tissue", {})
    continuous = profile.get("dynamic_patient_boundary", {}).get(
        "continuous_mechanics", {}
    )
    if (
        native_tissue.get("continuous_mechanics")
        != "dynamic_patient_explicit_tetmesh_gpu_volume_deformable"
        or deformable.get("device") != "CUDA"
        or continuous.get("backend") != "PhysX_GPU_volume_deformable"
    ):
        raise AssertionError("Native volume-deformable liver contract is missing")
    if any(
        value is not False
        for value in (
            native_tissue.get("arbitrary_runtime_mesh_cutting"),
            deformable.get("arbitrary_runtime_mesh_cutting"),
            continuous.get("constitutive_validation"),
        )
    ):
        raise AssertionError("Deformable qualification boundaries are incomplete")
    for path in (
        runtime.DYNAMIC_PATIENT_USD,
        runtime.DYNAMIC_PATIENT_LIVER_USD,
        runtime.DYNAMIC_PATIENT_RUNTIME,
    ):
        if not path.is_file():
            raise AssertionError(f"Dynamic Patient dependency is missing: {path}")
    if training["implementation"] != (
        "orbit.surgical.assets.oncologic_resection.OncologicResectionEpisode"
    ):
        raise AssertionError("Training contract runtime route is missing")
    if any(
        payload.get("clinical_validation") is not False
        for payload in (manifest, training)
    ):
        raise AssertionError("Nonclinical boundary must be explicit")

    interface_names = {
        "dranmar_oncosurgery_tool.usda",
        "dranmar_oncology_liver.usda",
        "dranmar_oncosurgery_workcell.usda",
    }
    for path in usda:
        text = path.read_text(encoding="utf-8")
        if path.name in interface_names and "payload = @./" not in text:
            raise AssertionError(f"Interface is not payload-backed: {path}")
        for asset_path in re.findall(r"@([^@]+)@", text):
            if asset_path.startswith("/") or ":\\" in asset_path:
                raise AssertionError(
                    f"Absolute composition arc in {path.name}: {asset_path}"
                )

    source_paths = [
        root
        / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
        / "oncologic_resection.py",
        root / "tests/test_oncologic_resection.py",
        root / "examples/validate_oncologic_resection_runtime.py",
        root / "scripts/validate_dranmar_oncologic_resection.py",
    ]
    for path in source_paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    portfolio = json.loads(
        (root / "physics_next/dr-anmar-assets.json").read_text(encoding="utf-8")
    )
    entries = {entry["id"]: entry for entry in portfolio["assets"]}
    if "dranmar-oncosurgery-cell-v1" not in entries:
        raise AssertionError("OncoSurgery is missing from the DrAnmar portfolio")
    portfolio_profile = json.loads(
        (
            root
            / "physics_next/surgical-oncology/"
            "dranmar-oncosurgery-cell-v1.json"
        ).read_text(encoding="utf-8")
    )
    if portfolio_profile != profile:
        raise AssertionError(
            "Catalog and physics_next oncology profiles have drifted"
        )

    usdchecker = shutil.which("usdchecker")
    if require_usdchecker and not usdchecker:
        raise AssertionError("usdchecker is required but unavailable")
    if usdchecker:
        for path in usda:
            completed = subprocess.run(
                [usdchecker, str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode:
                raise AssertionError(
                    f"usdchecker failed for {path.name}:\n{completed.stdout}"
                )

    return {
        "status": "pass",
        "catalog_subpath": str(CATALOG),
        "counts": counts,
        "tool_joint_count": len(expected_joints),
        "tumor_cell_count": len(tumor["cells"]),
        "resection_bond_count": len(topology["bonds"]),
        "protected_pedicle_count": sum(
            bond["seal_required"] for bond in topology["bonds"]
        ),
        "native_tissue_route": native_tissue["continuous_mechanics"],
        "arbitrary_runtime_mesh_cutting": False,
        "authored_payload_mass_kg": standalone_mass,
        "usdchecker": bool(usdchecker),
        "clinical_validation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--require-usdchecker", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.root, require_usdchecker=args.require_usdchecker),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
