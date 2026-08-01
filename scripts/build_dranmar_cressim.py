#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Build and test the exact locked CRESSim shared C API without package installs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _first(paths: list[Path], marker: str) -> Path:
    for path in paths:
        if (path / marker).is_file():
            return path
    raise SystemExit(f"required build dependency not found: {marker}")


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=ROOT / "config/physics-next-lock.json")
    parser.add_argument("--eigen-include", type=Path)
    parser.add_argument("--glfw-library", type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()

    next_root = args.root.expanduser().resolve()
    lock = json.loads(args.lock.resolve().read_text(encoding="utf-8"))
    source = lock["sources"]["cressim_mpm"]
    build = lock["builds"]["cressim_mpm"]
    cressim_root = next_root / "CRESSim-MPM"
    observed_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cressim_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if observed_revision != source["revision"]:
        raise SystemExit(
            f"CRESSim revision mismatch: {observed_revision} != {source['revision']}"
        )

    python_version = lock["python"]["version"]
    eigen = args.eigen_include or _first(
        [
            Path("/usr/include/eigen3"),
            next_root / f"env_isaaclab/lib/python{python_version}/site-packages/isaacsim/exts/isaacsim.ros1.bridge/noetic/include",
            next_root.parent / "sonogym/env_isaaclab/lib/python3.10/site-packages/isaacsim/exts/isaacsim.ros1.bridge/noetic/include",
        ],
        "Eigen/Dense",
    )
    if args.glfw_library:
        glfw = args.glfw_library
    else:
        candidates = sorted(next_root.glob("env_isaaclab/lib/python*/site-packages/**/libglfw.so"))
        candidates += sorted(next_root.parent.glob("**/libglfw.so"))
        glfw = next((path for path in candidates if path.is_file()), None)
        if glfw is None:
            raise SystemExit("required build dependency not found: libglfw.so")

    build_dir = cressim_root / "build-dranmar"
    env = dict(os.environ)
    env["PATH"] = f"/usr/local/cuda-12.8/bin:{env.get('PATH', '')}"
    _run(
        [
            "cmake", "-S", str(cressim_root), "-B", str(build_dir),
            f"-DCMAKE_BUILD_TYPE={build['build_type']}",
            f"-DCMAKE_CUDA_ARCHITECTURES={build['cuda_architectures']}",
            f"-DENGINE_STATIC={'ON' if build['engine_static'] else 'OFF'}",
            f"-DENABLE_EXAMPLES={'ON' if build['examples'] else 'OFF'}",
            f"-DENABLE_TESTS={'ON' if build['tests'] else 'OFF'}",
            f"-DEIGEN3_INCLUDE_DIR={eigen}",
            f"-Dglfw3_DIR={ROOT / 'physics_next/cmake'}",
            f"-DGLFW_LIBRARY={glfw}",
        ],
        env=env,
    )
    _run(["cmake", "--build", str(build_dir), f"-j{args.jobs}"], env=env)
    _run(["ctest", "--test-dir", str(build_dir), "--output-on-failure"], env=env)
    library = next_root / build["library_relative_path"]
    if not library.is_file():
        raise SystemExit(f"build did not produce locked library: {library}")
    print(library)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
