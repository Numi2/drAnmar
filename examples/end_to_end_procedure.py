#!/usr/bin/env python3
"""Pure-Python end-to-end patient event example."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_runtime():
    """Load through the extension when installed, or directly from a source checkout."""

    try:
        from orbit.surgical.assets import dynamic_abdominal_patient

        return dynamic_abdominal_patient
    except ModuleNotFoundError:
        runtime_path = (
            Path(__file__).resolve().parents[1]
            / "source/extensions/orbit.surgical.assets/orbit/surgical/assets"
            / "dynamic_abdominal_patient.py"
        )
        spec = importlib.util.spec_from_file_location(
            "dranmar_dynamic_patient_example_runtime", runtime_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load the dynamic-patient runtime")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


runtime = load_runtime()
patient = runtime.DynamicSurgicalPatient()
patient.interventions.set_access_state("open", source_robot="procedure_setup")
patient.contacts.observe(
    runtime.PatientContactFrame(
        target="gallbladder",
        source_robot="atraumatic_exposure_robot",
        interaction="exposure",
        normal_forces_n=(1.25, 1.25),
        tool_position_m=(0.0, 0.0, 0.0),
    )
)
patient.step(0.1)
for _ in range(12):
    patient.contacts.observe(
        runtime.PatientContactFrame(
            target="gallbladder",
            source_robot="atraumatic_exposure_robot",
            interaction="exposure",
            normal_forces_n=(1.25, 1.25),
            tool_position_m=(0.02, 0.0, 0.0),
        )
    )
    patient.step(0.1)
patient.interventions.apply_dissection(
    target="adhesion_00",
    method="guarded_scissors",
    source_robot="safeplane_dissection_robot",
)
patient.bleeding.create_source(
    "cystic_artery_injury",
    "gallbladder",
    vessel_radius_m=0.0008,
    injury_fraction=0.55,
    kind="arterial",
)
for _ in range(80):
    patient.step(0.1)
for _ in range(30):
    patient.contacts.observe(
        runtime.PatientContactFrame(
            target="cystic_artery_injury",
            source_robot="adaptive_hemostasis_robot",
            interaction="hemostasis",
            normal_forces_n=(1.8, 1.8),
            tool_position_m=(0.0, 0.0, 0.0),
        )
    )
    patient.step(0.1)
patient.interventions.apply_closure(
    target="abdominal_wall",
    method="staple_and_adhesive",
    closure_fraction=1.0,
    source_robot="closure_robot",
)
patient.interventions.apply_dressing(
    target="skin", pressure_kpa=-10.0, source_robot="dressing_robot"
)
print(json.dumps(patient.snapshot(), indent=2, sort_keys=True))
