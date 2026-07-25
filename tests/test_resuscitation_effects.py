from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / (
    "source/extensions/orbit.surgical.assets/orbit/surgical/assets/"
    "resuscitation_effects.py"
)


def load_module():
    name = "dranmar_resuscitation_effects_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def blood_frame(
    module,
    step,
    *,
    position_m,
    reservoir_mass_g,
    connected=1,
    line_pressure_kpa=10.0,
):
    return module.PumpEvidenceFrame(
        physics_step=step,
        simulation_time_s=step * 0.05,
        dt_s=0.05,
        channel_id="blood_product",
        access_attachment_count=connected,
        plunger_position_m=position_m,
        downstream_flow_ml_s=600.0,
        reservoir_mass_g=reservoir_mass_g,
        line_pressure_kpa=line_pressure_kpa,
    )


def test_delivery_requires_plunger_flow_mass_and_vascular_attachment():
    module = load_module()
    effects = module.ContactDrivenResuscitationEffects()
    adapter = effects.create_scene_adapter()
    adapter.publish(
        blood_frame(
            module,
            1,
            position_m=0.0,
            reservoir_mass_g=500.0,
        )
    )
    adapter.publish(
        blood_frame(
            module,
            2,
            position_m=0.018,
            reservoir_mass_g=468.2,
        )
    )
    connected = effects.snapshot().channels["blood_product"]
    assert connected["delivered_to_patient_ml"] == pytest.approx(30.0)

    adapter.publish(
        blood_frame(
            module,
            3,
            position_m=0.036,
            reservoir_mass_g=436.4,
            connected=0,
        )
    )
    disconnected = effects.snapshot().channels["blood_product"]
    assert disconnected["delivered_to_patient_ml"] == pytest.approx(30.0)
    assert disconnected["withdrawn_from_reservoir_ml"] == pytest.approx(60.0)
    assert disconnected["wasted_or_extravasated_ml"] == pytest.approx(30.0)


def test_high_line_pressure_cannot_create_patient_volume():
    module = load_module()
    effects = module.ContactDrivenResuscitationEffects()
    adapter = effects.create_scene_adapter()
    adapter.publish(
        blood_frame(
            module,
            1,
            position_m=0.0,
            reservoir_mass_g=500.0,
        )
    )
    adapter.publish(
        blood_frame(
            module,
            2,
            position_m=0.018,
            reservoir_mass_g=468.2,
            line_pressure_kpa=60.0,
        )
    )
    state = effects.snapshot().channels["blood_product"]
    assert state["delivered_to_patient_ml"] == 0.0
    assert state["pressure_damage_fraction"] > 0.0


def test_ventilation_requires_airway_connection_pressure_and_chest_motion():
    module = load_module()
    effects = module.ContactDrivenResuscitationEffects()
    adapter = effects.create_scene_adapter()
    adapter.publish_ventilation(
        module.VentilationEvidenceFrame(
            physics_step=1,
            simulation_time_s=0.05,
            dt_s=0.05,
            airway_attachment_count=1,
            valve_angle_deg=90.0,
            inspiratory_flow_l_min=7.0,
            leaked_flow_l_min=0.0,
            airway_pressure_cmh2o=15.0,
            measured_fio2_fraction=0.5,
            chest_excursion_m=0.02,
        )
    )
    supported = effects.snapshot().ventilation
    assert supported["effective_minute_ventilation_l_min"] == pytest.approx(7.0)
    assert supported["delivered_fio2_fraction"] == pytest.approx(0.5)

    adapter.publish_ventilation(
        module.VentilationEvidenceFrame(
            physics_step=2,
            simulation_time_s=0.10,
            dt_s=0.05,
            airway_attachment_count=0,
            valve_angle_deg=90.0,
            inspiratory_flow_l_min=7.0,
            leaked_flow_l_min=0.0,
            airway_pressure_cmh2o=15.0,
            measured_fio2_fraction=1.0,
            chest_excursion_m=0.02,
        )
    )
    disconnected = effects.snapshot().ventilation
    assert disconnected["effective_minute_ventilation_l_min"] == 0.0
    assert disconnected["delivered_fio2_fraction"] == pytest.approx(0.21)
