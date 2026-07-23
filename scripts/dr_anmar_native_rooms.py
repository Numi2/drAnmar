# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve OpenUSD assets that have a native Isaac Lab room binding."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BINDINGS_PATH = REPOSITORY_ROOT / "physics_next/room_bindings.json"


@lru_cache(maxsize=1)
def load_native_room_bindings() -> dict[str, Any]:
    payload = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "dr.anmar.native-room-bindings.v1":
        raise ValueError("Unsupported native-room binding schema")
    return payload


def resolve_native_room(procedure_id: str) -> dict[str, Any] | None:
    binding = load_native_room_bindings().get("rooms", {}).get(procedure_id)
    if not binding:
        return None
    data_root = Path(
        os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")
    ).expanduser().resolve()
    resolved = dict(binding)
    if binding.get("runtime_provider") == "nvidia_softmimicgen":
        runtime_root = Path(
            os.environ.get(
                "DR_ANMAR_SOFTMIMICGEN_ROOT",
                data_root / "native-suture-runtime/SoftMimicGen",
            )
        ).expanduser().resolve()
        required = [runtime_root / item for item in binding.get("required_files", [])]
        resolved["runtime_root"] = str(runtime_root)
        resolved["required_paths"] = [str(path) for path in required]
        resolved["available"] = bool(required) and all(path.is_file() for path in required)
        return resolved
    asset_root = REPOSITORY_ROOT if binding.get("repository_asset") else data_root
    resolved["asset_path"] = str((asset_root / binding["asset"]).resolve())
    resolved["asset_contract_path"] = str(
        (asset_root / binding["asset_contract"]).resolve()
    )
    if binding.get("alternate_tetmesh_asset"):
        resolved["alternate_tetmesh_asset_path"] = str(
            (asset_root / binding["alternate_tetmesh_asset"]).resolve()
        )
    auxiliary_assets: list[dict[str, Any]] = []
    for auxiliary in binding.get("auxiliary_assets", []):
        resolved_auxiliary = dict(auxiliary)
        resolved_auxiliary["asset_path"] = str(
            (asset_root / auxiliary["asset"]).resolve()
        )
        auxiliary_assets.append(resolved_auxiliary)
    resolved["auxiliary_assets"] = auxiliary_assets
    resolved["material_path"] = str((REPOSITORY_ROOT / binding["material"]).resolve())
    resolved["available"] = (
        Path(resolved["asset_path"]).is_file()
        and Path(resolved["asset_contract_path"]).is_file()
        and Path(resolved["material_path"]).is_file()
        and all(
            Path(auxiliary["asset_path"]).is_file()
            for auxiliary in auxiliary_assets
        )
    )
    return resolved
