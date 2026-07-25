# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / (
    "source/extensions/orbit.surgical.assets/orbit/surgical/assets/"
    "dranmar_camera_scheduler.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "dranmar_camera_scheduler_test_module", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeAdapter:
    def __init__(self):
        self.opened = []
        self.closed = []
        self.active = 0
        self.maximum_active = 0

    def open(self, path):
        self.opened.append(path)
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        return {"path": path}

    def capture(self, sensor):
        return {"rgb": sensor["path"], "depth": 1.0}

    def close(self, sensor):
        self.closed.append(sensor["path"])
        self.active -= 1


def test_scheduler_reuses_current_camera_and_never_opens_two():
    module = load_module()
    adapter = FakeAdapter()
    simulation_time = iter((0.0, 0.01, 0.02))
    scheduler = module.SerializedCameraScheduler(
        {"left": "/Left", "right": "/Right"},
        adapter,
        simulation_clock=lambda: next(simulation_time),
    )
    first = scheduler.capture("left")
    second = scheduler.capture("left")
    third = scheduler.capture("right")
    assert [first.sequence, second.sequence, third.sequence] == [1, 2, 3]
    assert adapter.opened == ["/Left", "/Right"]
    assert adapter.closed == ["/Left"]
    assert adapter.maximum_active == 1
    scheduler.close()
    assert adapter.closed == ["/Left", "/Right"]
    assert adapter.active == 0


def test_cycle_reports_real_skew_instead_of_claiming_synchrony():
    module = load_module()
    adapter = FakeAdapter()
    simulation_time = iter((1.0, 1.04, 1.10))
    with module.SerializedCameraScheduler(
        {"rgb_left": "/L", "rgb_right": "/R", "depth": "/D"},
        adapter,
        simulation_clock=lambda: next(simulation_time),
    ) as scheduler:
        result = scheduler.capture_cycle(maximum_fusion_skew_s=0.05)
    assert result["simulation_skew_s"] == pytest.approx(0.10)
    assert result["within_fusion_skew_gate"] is False
    assert result["maximum_concurrent_pipelines"] == 1
    assert adapter.maximum_active == 1


def test_scheduler_fails_closed_on_unknown_camera_and_empty_data():
    module = load_module()
    adapter = FakeAdapter()
    scheduler = module.SerializedCameraScheduler(
        {"left": "/Left"}, adapter, simulation_clock=lambda: 0.0
    )
    with pytest.raises(KeyError):
        scheduler.capture("missing")
    scheduler.close()
