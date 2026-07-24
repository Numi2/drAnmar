#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect the isolated Dr.Anmar physics-next contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dr_anmar_physics_authority import DEFAULT_MANIFEST, load_physics_authority


def main() -> int:
    parser = argparse.ArgumentParser(description="Dr.Anmar physics-next control plane")
    parser.add_argument("command", choices=("status", "benchmark-plan", "env"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark", default="liver-retraction")
    args = parser.parse_args()

    authority = load_physics_authority(args.manifest)
    if args.command == "status":
        payload = authority.runtime_payload()
        payload["valid"] = True
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "env":
        root = Path(os.environ.get("DR_ANMAR_PHYSICS_NEXT_ROOT", Path.home() / ".local/share/dr-anmar/physics-next"))
        print(f"export DR_ANMAR_PHYSICS_NEXT_ROOT={root.expanduser().resolve()}")
        print(f"export DR_ANMAR_PHYSICS_MANIFEST={authority.manifest_path}")
        print("export DR_ANMAR_ENABLE_EXPERIMENTAL_PHYSICS=0")
        return 0

    benchmark_path = authority.manifest_path.parent / "benchmarks" / f"{args.benchmark}.json"
    if not benchmark_path.is_file():
        parser.error(f"Unknown benchmark: {args.benchmark}")
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    payload["physics_manifest"] = str(authority.manifest_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
