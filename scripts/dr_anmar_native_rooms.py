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
    if payload.get("clinical_validation") is not False:
        raise ValueError("Native room bindings must remain explicitly non-clinical")
    return payload


def resolve_native_room(procedure_id: str) -> dict[str, Any] | None:
    binding = load_native_room_bindings().get("rooms", {}).get(procedure_id)
    if not binding:
        return None
    data_root = Path(
        os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")
    ).expanduser().resolve()
    resolved = dict(binding)
    resolved["asset_path"] = str((data_root / binding["asset"]).resolve())
    resolved["asset_contract_path"] = str((data_root / binding["asset_contract"]).resolve())
    resolved["material_path"] = str((REPOSITORY_ROOT / binding["material"]).resolve())
    resolved["available"] = Path(resolved["asset_path"]).is_file() and Path(
        resolved["asset_contract_path"]
    ).is_file() and Path(resolved["material_path"]).is_file()
    return resolved
