#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Install checksum-pinned MediaPipe Hand Landmarker assets for local serving."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


TASKS_VERSION = "0.10.35"
TASKS_URL = (
    "https://registry.npmjs.org/@mediapipe/tasks-vision/-/"
    f"tasks-vision-{TASKS_VERSION}.tgz"
)
TASKS_SHA256 = "84597a25e13d123b5f4cbe768bb72e97a2c28c7a465f0ace287d8cbe5246bff0"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
PACKAGE_MEMBERS = {
    "package/vision_bundle.mjs": "vision_bundle.mjs",
    "package/wasm/vision_wasm_internal.js": "wasm/vision_wasm_internal.js",
    "package/wasm/vision_wasm_internal.wasm": "wasm/vision_wasm_internal.wasm",
    "package/wasm/vision_wasm_module_internal.js": "wasm/vision_wasm_module_internal.js",
    "package/wasm/vision_wasm_module_internal.wasm": "wasm/vision_wasm_module_internal.wasm",
    "package/wasm/vision_wasm_nosimd_internal.js": "wasm/vision_wasm_nosimd_internal.js",
    "package/wasm/vision_wasm_nosimd_internal.wasm": "wasm/vision_wasm_nosimd_internal.wasm",
}


def download(url: str, expected_sha256: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "DrAnmar-asset-installer/1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"Checksum mismatch for {url}: expected {expected_sha256}, got {actual}")
    return payload


def install(destination: Path) -> None:
    archive = download(TASKS_URL, TASKS_SHA256)
    model = download(MODEL_URL, MODEL_SHA256)
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
            members = {member.name: member for member in package.getmembers()}
            for package_name, relative_name in PACKAGE_MEMBERS.items():
                member = members.get(package_name)
                if member is None or not member.isfile():
                    raise RuntimeError(f"Pinned package is missing {package_name}")
                target = temporary / relative_name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = package.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not extract {package_name}")
                target.write_bytes(source.read())
        (temporary / "hand_landmarker.task").write_bytes(model)
        (temporary / "SHA256SUMS").write_text(
            f"{TASKS_SHA256}  tasks-vision-{TASKS_VERSION}.tgz\n"
            f"{MODEL_SHA256}  hand_landmarker.task\n",
            encoding="utf-8",
        )
        if destination.exists():
            backup = destination.with_name(f"{destination.name}.previous")
            if backup.exists():
                shutil.rmtree(backup)
            destination.replace(backup)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(
            os.environ.get(
                "DR_ANMAR_HAND_CONTROL_ASSET_ROOT",
                Path.home()
                / ".local/share/dr-anmar/assets/hand-control/"
                f"mediapipe-tasks-vision-{TASKS_VERSION}",
            )
        ),
    )
    args = parser.parse_args()
    install(args.destination)
    print(f"[DR_ANMAR_HAND_CONTROL] installed {args.destination.expanduser().resolve()}")


if __name__ == "__main__":
    main()
