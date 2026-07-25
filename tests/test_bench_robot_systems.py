from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ASSET_ROOT = ROOT / "source/extensions/orbit.surgical.assets/data"
sys.path.insert(0, str(SCRIPTS))

from dr_anmar_bench_systems import (
    BENCH_ROBOT_SYSTEM_CATALOG,
    BENCH_ROBOT_SYSTEMS_BY_ID,
    related_asset_paths,
    resolve_featured_robot_system,
)
from dr_anmar_procedures import PROCEDURES_BY_ID

EXPECTED_SYSTEM_IDS = (
    "wound_preparation_robot",
    "atraumatic_exposure_robot",
    "adaptive_hemostasis_robot",
    "adaptive_anastomosis_robot",
    "adaptive_seal_divide_robot",
    "safeplane_dissection_robot",
    "perfusion_viability_robot",
)


def test_every_new_robot_system_is_selectable_from_the_native_bench() -> None:
    procedure = PROCEDURES_BY_ID["nvidia-native-surgical-bench"]
    catalog = {str(item["id"]): item for item in procedure["bench_asset_catalog"]}

    assert tuple(BENCH_ROBOT_SYSTEMS_BY_ID) == EXPECTED_SYSTEM_IDS
    for asset_id in EXPECTED_SYSTEM_IDS:
        item = catalog[asset_id]
        assert item is BENCH_ROBOT_SYSTEMS_BY_ID[asset_id]
        assert item["provider"] == "dr_anmar"
        assert item["bench_kind"] == "robot_system"
        assert item["representation"] == ("standalone_articulation_with_task_substrate")
        assert item["default"] is False


@pytest.mark.parametrize("item", BENCH_ROBOT_SYSTEM_CATALOG)
def test_robot_station_contract_resolves_every_runtime_representation(
    item: dict[str, object],
) -> None:
    paths = (str(item["path"]), *related_asset_paths(item))

    assert len(paths) == 4
    assert len(set(paths)) == 4
    assert all((ASSET_ROOT / relative_path).is_file() for relative_path in paths)


def test_featured_robot_station_is_exclusive_and_generator_safe() -> None:
    assert resolve_featured_robot_system(iter(["needle"])) is None
    assert (
        resolve_featured_robot_system(iter(["needle", EXPECTED_SYSTEM_IDS[0]]))
        == EXPECTED_SYSTEM_IDS[0]
    )
    with pytest.raises(ValueError, match="Choose one featured"):
        resolve_featured_robot_system(EXPECTED_SYSTEM_IDS[:2])
