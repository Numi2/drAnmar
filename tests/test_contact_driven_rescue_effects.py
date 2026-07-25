from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / (
    "source/extensions/orbit.surgical.assets/orbit/surgical/assets/"
    "deformable_rescue.py"
)


def load_effects_module():
    name = "dranmar_contact_driven_rescue_effects_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def vessel_frame(module, step, *, retained, left=1.8, right=1.8):
    return module.PhysicsEvidenceFrame(
        physics_step=step,
        simulation_time_s=step * 0.05,
        dt_s=0.05,
        station_id="rescue",
        tool_id="adaptive_hemostasis",
        target_id="rescue_vessel",
        left_normal_force_n=left,
        right_normal_force_n=right,
        separation_m=0.0005,
        tool_speed_m_s=0.0,
        target_distance_m=0.0,
        retained_attachment_count=retained,
    )


def test_retained_clip_persists_after_tool_release_but_not_attachment_loss():
    module = load_effects_module()
    effects = module.ContactDrivenRescueEffects(seed=9)
    adapter = effects.create_scene_adapter()
    for step in range(1, 9):
        adapter.publish(vessel_frame(module, step, retained=1))

    adapter.publish(
        vessel_frame(module, 9, retained=1, left=0.0, right=0.0)
    )
    retained = effects.snapshot().vessel
    assert retained["retained_clip_fraction"] == pytest.approx(1.0)

    adapter.publish(
        vessel_frame(module, 10, retained=0, left=0.0, right=0.0)
    )
    lost = effects.snapshot().vessel
    assert lost["retained_clip_fraction"] == 0.0
    assert lost["residual_flow_ml_s"] > retained["residual_flow_ml_s"]


def test_retained_geometry_keeps_repair_closed_without_tool_force():
    module = load_effects_module()
    effects = module.ContactDrivenRescueEffects()
    adapter = effects.create_scene_adapter()
    adapter.publish(
        module.PhysicsEvidenceFrame(
            physics_step=1,
            simulation_time_s=0.05,
            dt_s=0.05,
            station_id="primary",
            tool_id="adaptive_anastomosis",
            target_id="bowel_anastomosis",
            left_normal_force_n=0.0,
            right_normal_force_n=0.0,
            separation_m=0.0012,
            tool_speed_m_s=0.0,
            target_distance_m=0.02,
            retained_attachment_count=12,
        )
    )
    repair = effects.snapshot().repairs["bowel_anastomosis"]
    assert repair["approximation_fraction"] == pytest.approx(1.0)
    assert repair["retention_fraction"] == pytest.approx(1.0)
    assert repair["leak_rate_ml_s"] == pytest.approx(0.0)


def test_film_requires_live_bonds_contact_and_pressure_hold():
    module = load_effects_module()
    effects = module.ContactDrivenRescueEffects()
    adapter = effects.create_scene_adapter()
    for step in range(1, 22):
        adapter.publish(
            module.PhysicsEvidenceFrame(
                physics_step=step,
                simulation_time_s=step * 0.05,
                dt_s=0.05,
                station_id="primary",
                tool_id="closure_robot",
                target_id="occlusive_film",
                left_normal_force_n=0.0,
                right_normal_force_n=0.0,
                separation_m=0.0006,
                tool_speed_m_s=0.0,
                target_distance_m=0.0,
                retained_attachment_count=8,
                patch_contact_point_count=24,
                measured_cavity_pressure_kpa=-6.0,
            )
        )
    sealed = effects.snapshot().repairs["occlusive_film"]
    assert sealed["seal_quality"] == pytest.approx(1.0)
    assert sealed["seal_verified"] is True

    adapter.publish(
        module.PhysicsEvidenceFrame(
            physics_step=22,
            simulation_time_s=1.10,
            dt_s=0.05,
            station_id="primary",
            tool_id="closure_robot",
            target_id="occlusive_film",
            left_normal_force_n=0.0,
            right_normal_force_n=0.0,
            separation_m=0.003,
            tool_speed_m_s=0.0,
            target_distance_m=0.0,
            retained_attachment_count=2,
            patch_contact_point_count=5,
            measured_cavity_pressure_kpa=-1.0,
        )
    )
    failed = effects.snapshot().repairs["occlusive_film"]
    assert failed["seal_verified"] is False
    assert failed["leak_rate_ml_s"] > 0.0
