# Copyright (c) 2026, DrAnmar Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic abdominal patient runtime and Isaac integration.

The module exposes a shared physiology contract for all DrAnmar surgical robots:

    patient.respiration
    patient.perfusion
    patient.bleeding
    patient.vital_signs
    patient.tissue_state
    patient.organ_motion
    patient.damage
    patient.contacts
    patient.interventions
    patient.incision

Extended integration surfaces are available as ``patient.robot``,
``patient.event_bus`` and ``patient.fluids``.

The implementation is a research engineering model. Parameters are provisional,
manufacturer-neutral, and not intended for patient care or clinical decisions.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CATALOG_SUBPATH = "Props/Patients/DynamicAbdominalPatient"
_MODULE_PATH = Path(__file__).resolve()
_INSTALLED_DATA_ROOT = _MODULE_PATH.parents[3] / "data"
_DEVELOPMENT_DATA_ROOT = _MODULE_PATH.parents[6] / "assets"
ASSET_DATA_ROOT = (
    _INSTALLED_DATA_ROOT
    if (_INSTALLED_DATA_ROOT / CATALOG_SUBPATH).exists()
    else _DEVELOPMENT_DATA_ROOT
)
ASSET_ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
PATIENT_USD = ASSET_ROOT / "dranmar_dynamic_abdominal_patient.usda"
LAPAROTOMY_WOUND_USD = (
    ASSET_ROOT / "anatomy/dranmar_laparotomy_wound.usda"
)
PATIENT_RIGID_PROXY_USD = (
    ASSET_ROOT / "dranmar_dynamic_abdominal_patient_rigid_proxy.usda"
)
OPERATING_SCENE_USD = (
    ASSET_ROOT / "dranmar_dynamic_abdominal_patient_operating_scene.usda"
)
ANATOMY_MANIFEST_PATH = ASSET_ROOT / "anatomy_manifest.json"
PHYSIOLOGY_NETWORK_PATH = ASSET_ROOT / "physiology_network.json"
MECHANICS_CONTRACT_PATH = ASSET_ROOT / "mechanics_contract.json"
ROBOT_COMPATIBILITY_PATH = ASSET_ROOT / "robot_compatibility.json"
PROCEDURE_SCENARIOS_PATH = ASSET_ROOT / "procedure_scenarios.json"

FLUID_ASSETS = {
    "blood": ASSET_ROOT / "fluids/blood_particle.usda",
    "bile": ASSET_ROOT / "fluids/bile_particle.usda",
    "urine": ASSET_ROOT / "fluids/urine_particle.usda",
    "irrigation": ASSET_ROOT / "fluids/irrigation_particle.usda",
}

FLUID_PHYSICS_PRESETS: dict[str, dict[str, float | int]] = {
    "blood": {
        "density_kg_m3": 1060.0,
        "viscosity": 0.0060,
        "cohesion": 0.0020,
        "particle_group": 1,
    },
    "bile": {
        "density_kg_m3": 1010.0,
        "viscosity": 0.0030,
        "cohesion": 0.0014,
        "particle_group": 2,
    },
    "urine": {
        "density_kg_m3": 1015.0,
        "viscosity": 0.0011,
        "cohesion": 0.0007,
        "particle_group": 3,
    },
    "irrigation": {
        "density_kg_m3": 1000.0,
        "viscosity": 0.0010,
        "cohesion": 0.0005,
        "particle_group": 4,
    },
}

VALID_ACCESS_STATES = frozenset({"intact", "open"})
VALID_FLUIDS = frozenset(FLUID_ASSETS)
VALID_PROCEDURE_STAGES = frozenset(
    {
        "closed",
        "access_open",
        "exposed",
        "dissection",
        "hemostasis",
        "division",
        "reconstruction",
        "closure",
        "dressed",
    }
)
VALID_CONDITIONS = frozenset(
    {
        "healthy",
        "hemorrhage",
        "bowel_ischemia",
        "bile_leak",
        "ureter_injury",
        "liver_tumor",
        "dense_adhesions",
        "postoperative",
    }
)
VALID_HABITUS = frozenset({"baseline", "lean", "increased_visceral_fat"})
VALID_CONTACT_INTERACTIONS = frozenset({"exposure", "hemostasis"})
CONTACT_PERFUSION_TERRITORIES = {
    "mesentery": "small_bowel",
    "major_vessels": "other",
}
NATIVE_DEFORMABLE_ROUTES = frozenset(
    {
        "current_explicit_tetmesh_volume_hierarchy",
        "current_surface_deformable",
        "legacy_surface_deformable",
    }
)

LAPAROTOMY_WOUND_LAYER_CONFIGS: dict[str, dict[str, float | str]] = {
    "skin": {
        "youngs_modulus_pa_seed": 75_000.0,
        "poissons_ratio_seed": 0.46,
        "density_kg_m3": 1_080.0,
        "mass_kg": 0.1877904,
        "damping_seed": 0.16,
        "dynamic_friction": 0.48,
    },
    "subcutaneous_fat": {
        "youngs_modulus_pa_seed": 18_000.0,
        "poissons_ratio_seed": 0.47,
        "density_kg_m3": 920.0,
        "mass_kg": 0.6398784,
        "damping_seed": 0.20,
        "dynamic_friction": 0.35,
    },
    "fascia": {
        "youngs_modulus_pa_seed": 850_000.0,
        "poissons_ratio_seed": 0.44,
        "density_kg_m3": 1_120.0,
        "mass_kg": 0.0973728,
        "damping_seed": 0.14,
        "dynamic_friction": 0.42,
    },
    "abdominal_wall": {
        "youngs_modulus_pa_seed": 160_000.0,
        "poissons_ratio_seed": 0.46,
        "density_kg_m3": 1_060.0,
        "mass_kg": 0.8600904,
        "damping_seed": 0.18,
        "dynamic_friction": 0.44,
    },
    "peritoneum": {
        "youngs_modulus_pa_seed": 1_100_000.0,
        "poissons_ratio_seed": 0.45,
        "density_kg_m3": 1_080.0,
        "mass_kg": 0.05007744,
        "damping_seed": 0.14,
        "dynamic_friction": 0.38,
    },
}


def tensor_value(value: Any):
    """Return a native tensor from Isaac 6 proxy objects when necessary."""
    return value.torch if hasattr(value, "torch") else value


def _clamp(value: float, low: float, high: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"value must be finite, received {value!r}")
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"invalid clamp interval [{low!r}, {high!r}]")
    return max(low, min(high, number))


def _nonnegative(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _smooth(current: float, target: float, dt_s: float, tau_s: float) -> float:
    if tau_s <= 0.0:
        return float(target)
    alpha = 1.0 - math.exp(-max(0.0, dt_s) / tau_s)
    return float(current + alpha * (target - current))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_anatomy_manifest() -> dict[str, Any]:
    return load_json(ANATOMY_MANIFEST_PATH)


def load_physiology_network() -> dict[str, Any]:
    return load_json(PHYSIOLOGY_NETWORK_PATH)


def load_mechanics_contract() -> dict[str, Any]:
    return load_json(MECHANICS_CONTRACT_PATH)


def load_robot_compatibility() -> dict[str, Any]:
    return load_json(ROBOT_COMPATIBILITY_PATH)


def load_procedure_scenarios() -> dict[str, Any]:
    return load_json(PROCEDURE_SCENARIOS_PATH)


@dataclass(frozen=True)
class PatientEvent:
    time_s: float
    kind: str
    payload: dict[str, Any]
    source: str = "patient"


class PatientEventBus:
    """Bounded event stream shared by physiology, damage and robot adapters."""

    def __init__(self, capacity: int = 8192):
        self.capacity = max(16, int(capacity))
        self._events: deque[PatientEvent] = deque(maxlen=self.capacity)
        self._subscribers: dict[str, list[Any]] = {}

    def emit(self, event: PatientEvent) -> None:
        self._events.append(event)
        for callback in self._subscribers.get(event.kind, ()):
            callback(event)
        for callback in self._subscribers.get("*", ()):
            callback(event)

    def subscribe(self, kind: str, callback: Any) -> None:
        self._subscribers.setdefault(str(kind), []).append(callback)

    def snapshot(self, since_s: float | None = None) -> list[dict[str, Any]]:
        return [
            asdict(event)
            for event in self._events
            if since_s is None or event.time_s >= float(since_s)
        ]


@dataclass
class RespirationModel:
    respiratory_rate_bpm: float = 14.0
    tidal_volume_ml: float = 500.0
    spo2_fraction: float = 0.985
    paco2_mmhg: float = 40.0
    phase_rad: float = 0.0
    diaphragm_excursion_m: float = 0.025
    airway_obstruction_fraction: float = 0.0
    inspired_oxygen_fraction: float = 0.21
    minute_ventilation_l_min: float = 7.0

    def step(
        self, dt_s: float, *, metabolic_scale: float = 1.0, perfusion_scale: float = 1.0
    ) -> None:
        rate_hz = max(self.respiratory_rate_bpm, 1.0) / 60.0
        self.phase_rad = (self.phase_rad + 2.0 * math.pi * rate_hz * dt_s) % (
            2.0 * math.pi
        )
        effective_tidal = self.tidal_volume_ml * (
            1.0 - _clamp(self.airway_obstruction_fraction, 0.0, 0.95)
        )
        self.minute_ventilation_l_min = (
            effective_tidal * self.respiratory_rate_bpm / 1000.0
        )
        ventilation_ratio = self.minute_ventilation_l_min / max(
            7.0 * metabolic_scale, 0.1
        )
        target_co2 = 40.0 / max(ventilation_ratio, 0.20)
        self.paco2_mmhg = _smooth(
            self.paco2_mmhg, _clamp(target_co2, 20.0, 100.0), dt_s, 8.0
        )
        oxygen_gain = (self.inspired_oxygen_fraction - 0.21) * 2.4
        ventilation_penalty = max(0.0, 1.0 - ventilation_ratio) * 0.11
        perfusion_penalty = max(0.0, 0.55 - perfusion_scale) * 0.16
        target_spo2 = _clamp(
            0.985 + oxygen_gain - ventilation_penalty - perfusion_penalty, 0.45, 1.0
        )
        self.spo2_fraction = _smooth(self.spo2_fraction, target_spo2, dt_s, 5.0)

    @property
    def displacement_fraction(self) -> float:
        # Smooth respiratory trajectory with slower expiration.
        s = math.sin(self.phase_rad)
        return math.copysign(abs(s) ** 0.78, s)

    @property
    def diaphragm_displacement_m(self) -> float:
        return self.diaphragm_excursion_m * self.displacement_fraction


@dataclass
class CardiovascularModel:
    heart_rate_bpm: float = 72.0
    stroke_volume_ml: float = 70.0
    cardiac_output_l_min: float = 5.04
    systemic_vascular_resistance_mmhg_min_l: float = 17.5
    central_venous_pressure_mmhg: float = 5.0
    mean_arterial_pressure_mmhg: float = 92.0
    systolic_pressure_mmhg: float = 122.0
    diastolic_pressure_mmhg: float = 72.0
    contractility_fraction: float = 1.0
    baroreflex_drive: float = 0.0
    heart_phase_rad: float = 0.0

    def step(
        self,
        dt_s: float,
        *,
        blood_volume_fraction: float,
        oxygenation_fraction: float,
        temperature_c: float,
        anesthetic_depression_fraction: float = 0.0,
        map_target_mmhg: float = 92.0,
    ) -> None:
        bv = _clamp(blood_volume_fraction, 0.25, 1.35)
        preload = _clamp((bv - 0.28) / 0.72, 0.08, 1.22)
        temperature_drive = max(0.0, 36.5 - temperature_c) * 2.5
        hypoxia_drive = max(0.0, 0.94 - oxygenation_fraction) * 180.0
        volume_drive = max(0.0, 1.0 - bv) * 75.0
        reflex_error = max(0.0, map_target_mmhg - self.mean_arterial_pressure_mmhg)
        target_hr = (
            72.0
            + 0.72 * reflex_error
            + volume_drive
            + hypoxia_drive
            + temperature_drive
        )
        target_hr *= 1.0 - 0.45 * _clamp(anesthetic_depression_fraction, 0.0, 0.9)
        target_hr = _clamp(target_hr, 35.0, 190.0)
        self.heart_rate_bpm = _smooth(self.heart_rate_bpm, target_hr, dt_s, 4.5)

        # Compensatory vasoconstriction preserves pressure during modest volume loss
        # without producing a non-physiologic hypertensive response.
        target_svr = 17.5 * (1.0 + 0.9 * max(0.0, 1.0 - bv))
        target_svr *= 1.0 - 0.30 * _clamp(anesthetic_depression_fraction, 0.0, 0.9)
        self.systemic_vascular_resistance_mmhg_min_l = _smooth(
            self.systemic_vascular_resistance_mmhg_min_l,
            _clamp(target_svr, 6.0, 48.0),
            dt_s,
            7.0,
        )

        tachy_penalty = max(0.0, self.heart_rate_bpm - 145.0) / 160.0
        oxygen_contractility = _clamp((oxygenation_fraction - 0.45) / 0.50, 0.25, 1.0)
        self.contractility_fraction = _smooth(
            self.contractility_fraction,
            oxygen_contractility * (1.0 - 0.35 * tachy_penalty),
            dt_s,
            10.0,
        )
        target_sv = 70.0 * preload * self.contractility_fraction
        self.stroke_volume_ml = _smooth(
            self.stroke_volume_ml, _clamp(target_sv, 8.0, 110.0), dt_s, 3.0
        )
        self.cardiac_output_l_min = self.heart_rate_bpm * self.stroke_volume_ml / 1000.0
        self.central_venous_pressure_mmhg = _clamp(
            5.0 * bv - 1.5 * max(0.0, 1.0 - bv), 0.0, 14.0
        )
        target_map = (
            self.cardiac_output_l_min * self.systemic_vascular_resistance_mmhg_min_l
            + self.central_venous_pressure_mmhg
        )
        self.mean_arterial_pressure_mmhg = _smooth(
            self.mean_arterial_pressure_mmhg, _clamp(target_map, 15.0, 170.0), dt_s, 1.6
        )
        pulse_pressure = _clamp(
            38.0 * preload * self.contractility_fraction, 12.0, 70.0
        )
        self.diastolic_pressure_mmhg = _clamp(
            self.mean_arterial_pressure_mmhg - pulse_pressure / 3.0, 10.0, 150.0
        )
        self.systolic_pressure_mmhg = _clamp(
            self.diastolic_pressure_mmhg + pulse_pressure, 20.0, 220.0
        )
        self.heart_phase_rad = (
            self.heart_phase_rad + 2.0 * math.pi * self.heart_rate_bpm / 60.0 * dt_s
        ) % (2.0 * math.pi)


@dataclass
class CoagulationModel:
    platelet_function_fraction: float = 1.0
    fibrinogen_g_l: float = 2.8
    inr: float = 1.0
    clotting_efficiency_fraction: float = 1.0
    dilution_fraction: float = 0.0
    acidosis_fraction: float = 0.0

    def step(
        self,
        dt_s: float,
        *,
        temperature_c: float,
        blood_volume_fraction: float,
        crystalloid_ml: float,
        lactate_mmol_l: float,
    ) -> None:
        self.dilution_fraction = _clamp(
            crystalloid_ml / 6500.0 + max(0.0, 1.0 - blood_volume_fraction) * 0.15,
            0.0,
            0.75,
        )
        self.acidosis_fraction = _clamp((lactate_mmol_l - 2.0) / 8.0, 0.0, 0.8)
        temp_factor = _clamp((temperature_c - 31.0) / 6.0, 0.15, 1.05)
        platelet_factor = _clamp(
            self.platelet_function_fraction * (1.0 - 0.65 * self.dilution_fraction),
            0.05,
            1.0,
        )
        fibrin_factor = _clamp(
            self.fibrinogen_g_l / 2.8 * (1.0 - 0.55 * self.dilution_fraction), 0.05, 1.2
        )
        inr_factor = _clamp(1.0 / max(self.inr, 0.4), 0.25, 1.25)
        acid_factor = 1.0 - 0.55 * self.acidosis_fraction
        target = (
            temp_factor * platelet_factor * fibrin_factor * inr_factor * acid_factor
        )
        self.clotting_efficiency_fraction = _smooth(
            self.clotting_efficiency_fraction, _clamp(target, 0.02, 1.2), dt_s, 5.0
        )

    def clot_growth_rate_per_s(self, flow_ml_s: float) -> float:
        high_flow_penalty = 1.0 / (1.0 + max(0.0, flow_ml_s) / 3.5)
        return 0.018 * self.clotting_efficiency_fraction * high_flow_penalty


@dataclass
class FluidBalanceModel:
    baseline_blood_volume_ml: float = 5000.0
    intravascular_volume_ml: float = 5000.0
    interstitial_volume_ml: float = 11000.0
    crystalloid_input_ml: float = 0.0
    colloid_input_ml: float = 0.0
    transfused_red_cell_ml: float = 0.0
    cumulative_blood_loss_ml: float = 0.0
    urine_output_ml: float = 0.0
    bile_output_ml: float = 0.0
    suction_output_ml: float = 0.0
    irrigation_input_ml: float = 0.0
    irrigation_recovered_ml: float = 0.0

    @property
    def blood_volume_fraction(self) -> float:
        return _clamp(
            self.intravascular_volume_ml / self.baseline_blood_volume_ml, 0.0, 2.0
        )

    def lose_blood(self, volume_ml: float) -> None:
        requested_ml = _nonnegative(volume_ml, "blood-loss volume")
        actual_ml = min(requested_ml, self.intravascular_volume_ml)
        self.intravascular_volume_ml -= actual_ml
        self.cumulative_blood_loss_ml += actual_ml

    def infuse_crystalloid(self, volume_ml: float) -> None:
        v = _nonnegative(volume_ml, "crystalloid volume")
        self.crystalloid_input_ml += v
        self.intravascular_volume_ml += 0.24 * v
        self.interstitial_volume_ml += 0.76 * v

    def transfuse_blood(self, volume_ml: float) -> None:
        v = _nonnegative(volume_ml, "transfusion volume")
        self.transfused_red_cell_ml += v
        self.intravascular_volume_ml += v

    def collect_suction(self, volume_ml: float) -> None:
        self.suction_output_ml += _nonnegative(volume_ml, "suction volume")

    def add_irrigation(self, volume_ml: float) -> None:
        self.irrigation_input_ml += _nonnegative(volume_ml, "irrigation volume")

    def recover_irrigation(self, volume_ml: float) -> None:
        recovered_ml = _nonnegative(volume_ml, "recovered irrigation volume")
        available_ml = max(0.0, self.irrigation_input_ml - self.irrigation_recovered_ml)
        self.irrigation_recovered_ml += min(recovered_ml, available_ml)


@dataclass
class BleedSource:
    id: str
    organ_id: str
    vessel_radius_m: float
    injury_fraction: float
    kind: str
    downstream_pressure_mmhg: float = 5.0
    discharge_coefficient: float = 0.68
    control_effectiveness: float = 0.0
    contact_compression_fraction: float = 0.0
    clot_fraction: float = 0.0
    active: bool = True
    current_flow_ml_s: float = 0.0
    cumulative_loss_ml: float = 0.0
    last_control_method: str | None = None

    def effective_area_m2(self) -> float:
        radius = max(
            1.0e-6, self.vessel_radius_m * _clamp(self.injury_fraction, 0.0, 1.5)
        )
        return (
            math.pi
            * radius
            * radius
            * (1.0 - _clamp(self.control_effectiveness, 0.0, 1.0))
            * (1.0 - _clamp(self.clot_fraction, 0.0, 1.0))
        )


class BleedingModel:
    def __init__(self, patient: "DynamicSurgicalPatient"):
        self.patient = patient
        self.sources: dict[str, BleedSource] = {}
        self.total_flow_ml_s: float = 0.0

    def create_source(
        self,
        source_id: str,
        organ_id: str,
        *,
        vessel_radius_m: float,
        injury_fraction: float = 1.0,
        kind: str = "venous",
        downstream_pressure_mmhg: float | None = None,
    ) -> BleedSource:
        if kind not in {"arterial", "venous", "capillary", "solid_organ"}:
            raise ValueError(f"unsupported bleeding kind {kind!r}")
        if source_id in self.sources:
            raise ValueError(f"bleeding source {source_id!r} already exists")
        radius_m = float(vessel_radius_m)
        injury = float(injury_fraction)
        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("vessel_radius_m must be positive and finite")
        if not math.isfinite(injury) or not 0.0 <= injury <= 1.0:
            raise ValueError("injury_fraction must be finite and between 0 and 1")
        if downstream_pressure_mmhg is None:
            downstream_pressure_mmhg = 5.0 if kind == "arterial" else 0.0
        downstream_pressure = float(downstream_pressure_mmhg)
        if not math.isfinite(downstream_pressure):
            raise ValueError("downstream_pressure_mmhg must be finite")
        source = BleedSource(
            id=source_id,
            organ_id=organ_id,
            vessel_radius_m=radius_m,
            injury_fraction=injury,
            kind=kind,
            downstream_pressure_mmhg=downstream_pressure,
        )
        self.sources[source_id] = source
        return source

    def _set_contact_compression(self, source_id: str, fraction: float) -> None:
        """Apply transient occlusion derived from a physics contact frame."""
        source = self.sources[source_id]
        compression = _clamp(fraction, 0.0, 1.0)
        source.contact_compression_fraction = compression
        source.control_effectiveness = compression
        source.last_control_method = (
            "bilateral_contact_compression" if compression > 0.0 else None
        )
        if compression < 0.999 and source.clot_fraction < 0.999:
            source.active = True

    def reopen(self, source_id: str, fraction: float = 0.5) -> None:
        source = self.sources[source_id]
        source.control_effectiveness *= 1.0 - _clamp(fraction, 0.0, 1.0)
        source.contact_compression_fraction = source.control_effectiveness
        source.active = True

    def step(self, dt_s: float) -> float:
        dt_s = float(dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        rho = 1060.0
        map_mm = self.patient.cardiovascular.mean_arterial_pressure_mmhg
        cvp = self.patient.cardiovascular.central_venous_pressure_mmhg
        candidate_flows: dict[str, float] = {}
        for source in self.sources.values():
            if not source.active:
                source.current_flow_ml_s = 0.0
                continue
            upstream = map_mm if source.kind == "arterial" else max(cvp + 5.0, 8.0)
            if source.kind == "capillary":
                upstream = max(0.45 * map_mm, 12.0)
            elif source.kind == "solid_organ":
                upstream = max(0.62 * map_mm, 15.0)
            pressure_pa = max(
                0.0, (upstream - source.downstream_pressure_mmhg) * 133.322
            )
            q_m3_s = (
                source.discharge_coefficient
                * source.effective_area_m2()
                * math.sqrt(2.0 * pressure_pa / rho)
            )
            candidate_flows[source.id] = _clamp(q_m3_s * 1.0e6, 0.0, 120.0)

        requested_flow = sum(candidate_flows.values())
        requested_loss = requested_flow * dt_s
        available_volume = max(0.0, self.patient.fluid_balance.intravascular_volume_ml)
        actual_loss = min(requested_loss, available_volume)
        flow_scale = actual_loss / requested_loss if requested_loss > 1.0e-12 else 0.0

        total = 0.0
        for source in self.sources.values():
            if source.id not in candidate_flows:
                continue
            flow = candidate_flows[source.id] * flow_scale
            source.current_flow_ml_s = flow
            source.cumulative_loss_ml += flow * dt_s
            total += flow
            growth = self.patient.coagulation.clot_growth_rate_per_s(flow) * dt_s
            source.clot_fraction = _clamp(source.clot_fraction + growth, 0.0, 0.92)
            if source.effective_area_m2() < 1.0e-11:
                source.active = False

        self.total_flow_ml_s = total
        self.patient.fluid_balance.lose_blood(actual_loss)
        return actual_loss

    def active_sources(self) -> list[BleedSource]:
        return [source for source in self.sources.values() if source.active]


@dataclass
class RegionalPerfusionState:
    organ_id: str
    flow_ml_min: float = 0.0
    relative_flow_fraction: float = 1.0
    oxygen_delivery_ml_min: float = 0.0
    oxygen_demand_ml_min: float = 0.0
    oxygen_supply_ratio: float = 1.0
    viability_fraction: float = 1.0
    ischemia_time_s: float = 0.0
    compression_fraction: float = 0.0
    occlusion_fraction: float = 0.0
    leak_fraction: float = 0.0
    temperature_c: float = 37.0


class PerfusionModel:
    def __init__(
        self,
        patient: "DynamicSurgicalPatient",
        regions: Mapping[str, Mapping[str, Any]],
    ):
        self.patient = patient
        self.base_regions = {k: dict(v) for k, v in regions.items()}
        self.regions = {
            organ: RegionalPerfusionState(
                organ_id=organ,
                oxygen_demand_ml_min=float(cfg.get("oxygen_demand_ml_min", 1.0)),
            )
            for organ, cfg in self.base_regions.items()
        }
        self.global_perfusion_fraction = 1.0
        self.abdominal_flow_ml_min = 0.0

    def set_compression(self, organ_id: str, fraction: float) -> None:
        if organ_id in self.regions:
            self.regions[organ_id].compression_fraction = _clamp(fraction, 0.0, 1.0)

    def set_occlusion(self, organ_id: str, fraction: float) -> None:
        if organ_id in self.regions:
            self.regions[organ_id].occlusion_fraction = _clamp(fraction, 0.0, 1.0)

    def set_leak(self, organ_id: str, fraction: float) -> None:
        if organ_id in self.regions:
            self.regions[organ_id].leak_fraction = _clamp(fraction, 0.0, 1.0)

    def step(self, dt_s: float) -> None:
        cv = self.patient.cardiovascular
        baseline_map = float(
            self.patient.physiology_config.get("cardiovascular", {}).get(
                "map_target_mmhg",
                self.patient.physiology_config.get("baseline", {}).get(
                    "mean_arterial_pressure_mmhg", 92.0
                ),
            )
        )
        map_factor = _clamp(
            cv.mean_arterial_pressure_mmhg / max(baseline_map, 1.0), 0.05, 1.25
        )
        abdominal_fraction = 0.46
        self.abdominal_flow_ml_min = (
            cv.cardiac_output_l_min * 1000.0 * abdominal_fraction
        )
        weighted: dict[str, float] = {}
        for organ, cfg in self.base_regions.items():
            state = self.regions[organ]
            base = float(cfg.get("flow_fraction", 0.0))
            local = (1.0 - 0.96 * state.occlusion_fraction) * (
                1.0 - 0.88 * state.compression_fraction
            )
            local *= 1.0 - 0.35 * state.leak_fraction
            if cv.mean_arterial_pressure_mmhg < 65.0:
                if organ in {"skin", "abdominal_wall", "gallbladder", "bladder"}:
                    local *= 0.55
                elif "kidney" in organ:
                    local *= 0.72
            weighted[organ] = max(1.0e-6, base * local)
        total_weight = sum(weighted.values())
        ca_o2_ml_dl = self.patient.oxygen.arterial_oxygen_content_ml_dl
        for organ, weight in weighted.items():
            state = self.regions[organ]
            flow = self.abdominal_flow_ml_min * map_factor * weight / total_weight
            state.flow_ml_min = flow
            base_flow = self.abdominal_flow_ml_min * float(
                self.base_regions[organ].get("flow_fraction", 0.0)
            )
            state.relative_flow_fraction = flow / max(base_flow, 1.0e-6)
            state.oxygen_delivery_ml_min = flow / 100.0 * ca_o2_ml_dl
            # The manifest demand is a lower-bound estimate. Anchor the requirement
            # to each territory's own baseline oxygen delivery so the baseline state
            # is near a modest physiological reserve rather than several-fold excess.
            baseline_delivery = base_flow / 100.0 * 17.9
            effective_requirement = max(
                state.oxygen_demand_ml_min, 0.83 * baseline_delivery, 1.0e-6
            )
            state.oxygen_supply_ratio = (
                state.oxygen_delivery_ml_min / effective_requirement
            )
            if state.oxygen_supply_ratio < 0.72:
                state.ischemia_time_s += dt_s
                damage_rate = (0.72 - state.oxygen_supply_ratio) * dt_s / 900.0
                state.viability_fraction = _clamp(
                    state.viability_fraction - damage_rate, 0.0, 1.0
                )
            else:
                state.ischemia_time_s = max(0.0, state.ischemia_time_s - 0.35 * dt_s)
                state.viability_fraction = _clamp(
                    state.viability_fraction + 0.00015 * dt_s, 0.0, 1.0
                )
            state.temperature_c = (
                self.patient.temperature.core_temperature_c
                - 0.7 * max(0.0, 1.0 - state.relative_flow_fraction)
            )
        if self.regions:
            base_weight_sum = sum(
                max(0.0, float(self.base_regions[organ].get("flow_fraction", 0.0)))
                for organ in self.regions
            )
            if base_weight_sum > 0.0:
                self.global_perfusion_fraction = (
                    sum(
                        max(
                            0.0,
                            float(self.base_regions[organ].get("flow_fraction", 0.0)),
                        )
                        * state.relative_flow_fraction
                        for organ, state in self.regions.items()
                    )
                    / base_weight_sum
                )
            else:
                self.global_perfusion_fraction = 0.0


@dataclass
class OxygenDeliveryModel:
    hemoglobin_g_dl: float = 13.5
    arterial_oxygen_content_ml_dl: float = 17.9
    oxygen_delivery_ml_min: float = 900.0
    oxygen_consumption_ml_min: float = 250.0
    extraction_ratio: float = 0.28
    lactate_mmol_l: float = 1.1

    def step(
        self,
        dt_s: float,
        *,
        cardiac_output_l_min: float,
        spo2_fraction: float,
        blood_volume_fraction: float,
        tissue_supply_ratio: float,
    ) -> None:
        dilution = max(0.0, 1.0 - blood_volume_fraction)
        target_hb = _clamp(13.5 * (1.0 - 0.45 * dilution), 4.0, 18.0)
        self.hemoglobin_g_dl = _smooth(self.hemoglobin_g_dl, target_hb, dt_s, 20.0)
        pao2 = 95.0 * _clamp(spo2_fraction / 0.985, 0.2, 1.15)
        self.arterial_oxygen_content_ml_dl = (
            1.34 * self.hemoglobin_g_dl * spo2_fraction + 0.003 * pao2
        )
        self.oxygen_delivery_ml_min = (
            cardiac_output_l_min * 10.0 * self.arterial_oxygen_content_ml_dl
        )
        metabolic_demand = 250.0
        self.oxygen_consumption_ml_min = metabolic_demand * _clamp(
            tissue_supply_ratio, 0.25, 1.0
        )
        self.extraction_ratio = _clamp(
            self.oxygen_consumption_ml_min / max(self.oxygen_delivery_ml_min, 1.0),
            0.05,
            0.90,
        )
        lactate_target = (
            1.1
            + 8.0 * max(0.0, 0.70 - tissue_supply_ratio)
            + 4.0 * max(0.0, 0.60 - blood_volume_fraction)
        )
        self.lactate_mmol_l = _smooth(
            self.lactate_mmol_l, _clamp(lactate_target, 0.8, 16.0), dt_s, 35.0
        )


@dataclass
class TemperatureModel:
    core_temperature_c: float = 37.0
    ambient_temperature_c: float = 21.0
    warming_power_w: float = 0.0
    cold_fluid_input_ml: float = 0.0

    def step(
        self,
        dt_s: float,
        *,
        blood_loss_ml_s: float,
        irrigation_exposure_fraction: float = 0.0,
    ) -> None:
        heat_capacity_j_c = 72.0 * 3470.0
        passive_loss_w = 58.0 + 28.0 * irrigation_exposure_fraction
        hemorrhage_loss_w = min(120.0, blood_loss_ml_s * 2.3)
        net_w = self.warming_power_w - passive_loss_w - hemorrhage_loss_w
        self.core_temperature_c = _clamp(
            self.core_temperature_c + net_w * dt_s / heat_capacity_j_c, 28.0, 41.5
        )


@dataclass
class RenalModel:
    urine_flow_ml_h: float = 65.0
    cumulative_urine_ml: float = 0.0

    def step(
        self,
        dt_s: float,
        *,
        map_mmhg: float,
        left_flow_fraction: float,
        right_flow_fraction: float,
        vasopressin_drive: float,
    ) -> float:
        pressure = _clamp((map_mmhg - 35.0) / 40.0, 0.0, 1.25)
        perf = _clamp((left_flow_fraction + right_flow_fraction) / 2.0, 0.0, 1.5)
        target = (
            65.0 * pressure * perf * (1.0 - 0.72 * _clamp(vasopressin_drive, 0.0, 1.0))
        )
        self.urine_flow_ml_h = _smooth(
            self.urine_flow_ml_h, _clamp(target, 0.0, 160.0), dt_s, 80.0
        )
        produced = self.urine_flow_ml_h / 3600.0 * dt_s
        self.cumulative_urine_ml += produced
        return produced


@dataclass
class BiliaryModel:
    bile_flow_ml_h: float = 30.0
    cumulative_bile_generated_ml: float = 0.0
    cumulative_bile_leak_ml: float = 0.0
    duct_injury_fraction: float = 0.0
    duct_control_effectiveness: float = 0.0

    def step(
        self, dt_s: float, *, liver_perfusion_fraction: float
    ) -> tuple[float, float]:
        target = 30.0 * _clamp(liver_perfusion_fraction, 0.0, 1.4)
        self.bile_flow_ml_h = _smooth(self.bile_flow_ml_h, target, dt_s, 120.0)
        generated = self.bile_flow_ml_h / 3600.0 * dt_s
        leak = (
            generated
            * _clamp(self.duct_injury_fraction, 0.0, 1.0)
            * (1.0 - _clamp(self.duct_control_effectiveness, 0.0, 1.0))
        )
        self.cumulative_bile_generated_ml += generated
        self.cumulative_bile_leak_ml += leak
        return generated, leak


@dataclass
class VitalSignsModel:
    heart_rate_bpm: float = 72.0
    respiratory_rate_bpm: float = 14.0
    systolic_pressure_mmhg: float = 122.0
    diastolic_pressure_mmhg: float = 72.0
    mean_arterial_pressure_mmhg: float = 92.0
    spo2_fraction: float = 0.985
    etco2_mmhg: float = 35.0
    core_temperature_c: float = 37.0
    cardiac_output_l_min: float = 5.04
    shock_index: float = 0.59
    lactate_mmol_l: float = 1.1
    cumulative_blood_loss_ml: float = 0.0
    active_blood_loss_ml_min: float = 0.0
    urine_output_ml_h: float = 65.0
    bile_leak_ml_h: float = 0.0
    global_perfusion_fraction: float = 1.0
    clinical_status: str = "stable_research_state"

    def update(self, patient: "DynamicSurgicalPatient") -> None:
        cv = patient.cardiovascular
        self.heart_rate_bpm = cv.heart_rate_bpm
        self.respiratory_rate_bpm = patient.respiration.respiratory_rate_bpm
        self.systolic_pressure_mmhg = cv.systolic_pressure_mmhg
        self.diastolic_pressure_mmhg = cv.diastolic_pressure_mmhg
        self.mean_arterial_pressure_mmhg = cv.mean_arterial_pressure_mmhg
        self.spo2_fraction = patient.respiration.spo2_fraction
        self.etco2_mmhg = _clamp(patient.respiration.paco2_mmhg - 5.0, 10.0, 80.0)
        self.core_temperature_c = patient.temperature.core_temperature_c
        self.cardiac_output_l_min = cv.cardiac_output_l_min
        self.shock_index = cv.heart_rate_bpm / max(cv.systolic_pressure_mmhg, 1.0)
        self.lactate_mmol_l = patient.oxygen.lactate_mmol_l
        self.cumulative_blood_loss_ml = patient.fluid_balance.cumulative_blood_loss_ml
        self.active_blood_loss_ml_min = patient.bleeding.total_flow_ml_s * 60.0
        self.urine_output_ml_h = patient.renal.urine_flow_ml_h
        self.bile_leak_ml_h = (
            patient.biliary.bile_flow_ml_h
            * patient.biliary.duct_injury_fraction
            * (1.0 - patient.biliary.duct_control_effectiveness)
        )
        self.global_perfusion_fraction = patient.perfusion.global_perfusion_fraction
        if (
            self.mean_arterial_pressure_mmhg < 45
            or self.spo2_fraction < 0.82
            or self.shock_index > 1.5
        ):
            self.clinical_status = "critical_research_state"
        elif (
            self.mean_arterial_pressure_mmhg < 65
            or self.active_blood_loss_ml_min > 150
            or self.lactate_mmol_l > 4.0
        ):
            self.clinical_status = "unstable_research_state"
        else:
            self.clinical_status = "stable_research_state"


@dataclass
class OrganTissueState:
    organ_id: str
    integrity_fraction: float = 1.0
    closure_fraction: float = 0.0
    seal_fraction: float = 0.0
    retraction_fraction: float = 0.0
    contact_compression_fraction: float = 0.0
    retraction_compression_fraction: float = 0.0
    contamination_fraction: float = 0.0
    edema_fraction: float = 0.0
    removed_fraction: float = 0.0
    punctures: int = 0
    cuts: int = 0
    sutures: int = 0
    staples: int = 0
    active_adhesions: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


class TissueStateRegistry:
    def __init__(self, organ_ids: Iterable[str]):
        self.organs = {
            organ_id: OrganTissueState(organ_id=organ_id) for organ_id in organ_ids
        }
        self.access_state = "intact"
        self.wound_open_fraction = 0.0

    def get(self, organ_id: str) -> OrganTissueState:
        if organ_id not in self.organs:
            self.organs[organ_id] = OrganTissueState(organ_id=organ_id)
        return self.organs[organ_id]

    def set_access_state(self, state: str) -> None:
        if state not in VALID_ACCESS_STATES:
            raise ValueError(state)
        self.access_state = state
        self.wound_open_fraction = 1.0 if state == "open" else 0.0

    def puncture(self, organ_id: str, severity: float = 0.05) -> None:
        state = self.get(organ_id)
        state.punctures += 1
        state.integrity_fraction = _clamp(state.integrity_fraction - severity, 0.0, 1.0)

    def cut(self, organ_id: str, severity: float = 0.20) -> None:
        state = self.get(organ_id)
        state.cuts += 1
        state.integrity_fraction = _clamp(state.integrity_fraction - severity, 0.0, 1.0)

    def apply_contact_response(
        self,
        organ_id: str,
        *,
        retraction_fraction: float,
        compression_fraction: float,
    ) -> None:
        state = self.get(organ_id)
        state.retraction_fraction = _clamp(retraction_fraction, 0.0, 1.0)
        state.contact_compression_fraction = _clamp(
            compression_fraction, 0.0, 1.0
        )
        # Retain the original snapshot field for readers while changing its
        # authority: it is now an observed contact effect, not a caller input.
        state.retraction_compression_fraction = (
            state.contact_compression_fraction
        )

    def suture(self, organ_id: str, closure_fraction: float) -> None:
        state = self.get(organ_id)
        state.sutures += 1
        state.closure_fraction = max(
            state.closure_fraction, _clamp(closure_fraction, 0, 1)
        )

    def staple(self, organ_id: str, closure_fraction: float) -> None:
        state = self.get(organ_id)
        state.staples += 1
        state.closure_fraction = max(
            state.closure_fraction, _clamp(closure_fraction, 0, 1)
        )

    def seal(self, organ_id: str, seal_fraction: float) -> None:
        state = self.get(organ_id)
        state.seal_fraction = max(state.seal_fraction, _clamp(seal_fraction, 0, 1))

    def remove(self, organ_id: str, fraction: float = 1.0) -> None:
        state = self.get(organ_id)
        state.removed_fraction = _clamp(state.removed_fraction + fraction, 0, 1)
        state.integrity_fraction = _clamp(1 - state.removed_fraction, 0, 1)

    def debride(
        self, organ_id: str, fraction: float, contamination_reduction: float
    ) -> None:
        state = self.get(organ_id)
        state.removed_fraction = _clamp(state.removed_fraction + 0.15 * fraction, 0, 1)
        state.contamination_fraction = _clamp(
            state.contamination_fraction - contamination_reduction, 0, 1
        )


@dataclass(frozen=True)
class PatientContactFrame:
    """One post-physics tool/patient contact observation.

    The frame deliberately contains only simulator primitives.  It has no
    success, compression, perfusion, exposure, or hemostasis result for a
    policy or robot caller to write.
    """

    target: str
    source_robot: str
    interaction: str
    normal_forces_n: tuple[float, float]
    tool_position_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("target must not be empty")
        if not self.source_robot:
            raise ValueError("source_robot must not be empty")
        if self.interaction not in VALID_CONTACT_INTERACTIONS:
            raise ValueError(
                f"unsupported contact interaction {self.interaction!r}"
            )
        if len(self.normal_forces_n) != 2:
            raise ValueError("normal_forces_n must contain exactly two pad forces")
        forces = tuple(
            _nonnegative(value, f"normal_forces_n[{index}]")
            for index, value in enumerate(self.normal_forces_n)
        )
        object.__setattr__(self, "normal_forces_n", forces)
        if self.tool_position_m is not None:
            if len(self.tool_position_m) != 3:
                raise ValueError("tool_position_m must contain exactly three values")
            position = tuple(float(value) for value in self.tool_position_m)
            if not all(math.isfinite(value) for value in position):
                raise ValueError("tool_position_m values must be finite")
            object.__setattr__(self, "tool_position_m", position)


@dataclass(frozen=True)
class ContactEffectCalibration:
    """Provisional simulator coupling derived from the authored tool profiles."""

    minimum_contact_force_n: float = 0.05
    exposure_target_force_per_pad_n: float = 1.25
    exposure_soft_force_per_pad_n: float = 2.5
    exposure_hard_force_per_pad_n: float = 4.0
    exposure_maximum_asymmetry_n: float = 1.0
    exposure_full_retraction_distance_m: float = 0.04
    exposure_local_compression_at_target: float = 0.10
    hemostasis_target_force_per_pad_n: float = 1.8
    hemostasis_soft_force_per_pad_n: float = 4.0
    hemostasis_hard_force_per_pad_n: float = 7.0
    hemostasis_maximum_asymmetry_n: float = 1.5
    contact_attack_time_s: float = 0.18
    contact_release_time_s: float = 0.12
    tissue_recovery_time_s: float = 0.35
    overload_damage_fraction_per_s: float = 0.03
    parameter_status: str = "provisional_engineering_seeds"


@dataclass
class ContactEffectState:
    target: str
    source_robot: str
    interaction: str
    contact_active: bool = False
    anchor_position_m: tuple[float, float, float] | None = None
    retraction_fraction: float = 0.0
    compression_fraction: float = 0.0
    hemostatic_control_fraction: float = 0.0
    bilateral_force_n: float = 0.0
    peak_force_n: float = 0.0
    force_asymmetry_n: float = 0.0
    traction_distance_m: float = 0.0
    overload_damage_fraction: float = 0.0
    reported_damage_fraction: float = 0.0


class ContactDrivenPatientEffects:
    """Convert post-physics contact into patient state without result injection."""

    def __init__(
        self,
        patient: "DynamicSurgicalPatient",
        calibration: ContactEffectCalibration | None = None,
    ):
        self.patient = patient
        self.calibration = calibration or ContactEffectCalibration()
        self._pending: dict[tuple[str, str, str], PatientContactFrame] = {}
        self.states: dict[tuple[str, str, str], ContactEffectState] = {}

    def observe(self, frame: PatientContactFrame) -> None:
        """Queue the latest authoritative contact frame for the next patient step."""
        if frame.interaction == "exposure":
            self.patient.tissue_state.get(frame.target)
        elif frame.target not in self.patient.bleeding.sources:
            raise KeyError(f"unknown bleeding source {frame.target!r}")
        key = (frame.source_robot, frame.target, frame.interaction)
        self._pending[key] = frame
        self.states.setdefault(
            key,
            ContactEffectState(
                target=frame.target,
                source_robot=frame.source_robot,
                interaction=frame.interaction,
            ),
        )

    @staticmethod
    def _distance(
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def _force_quality(
        self,
        frame: PatientContactFrame | None,
        *,
        target_force_n: float,
        soft_force_n: float,
        hard_force_n: float,
        maximum_asymmetry_n: float,
    ) -> tuple[bool, float, float, float, float]:
        forces = frame.normal_forces_n if frame is not None else (0.0, 0.0)
        left, right = forces
        bilateral = min(left, right)
        peak = max(left, right)
        asymmetry = abs(left - right)
        active = bilateral >= self.calibration.minimum_contact_force_n
        if not active:
            return False, 0.0, bilateral, peak, asymmetry
        target_quality = _clamp(bilateral / target_force_n, 0.0, 1.0)
        balance_quality = _clamp(
            1.0 - asymmetry / maximum_asymmetry_n, 0.0, 1.0
        )
        overload_quality = (
            1.0
            if peak <= soft_force_n
            else _clamp(
                (hard_force_n - peak) / (hard_force_n - soft_force_n),
                0.0,
                1.0,
            )
        )
        return (
            True,
            target_quality * balance_quality * overload_quality,
            bilateral,
            peak,
            asymmetry,
        )

    def _apply_overload(
        self,
        state: ContactEffectState,
        *,
        dt_s: float,
        soft_force_n: float,
        hard_force_n: float,
    ) -> None:
        overload = max(
            0.0,
            (state.peak_force_n - soft_force_n)
            / max(hard_force_n - soft_force_n, 1.0e-9),
        )
        if overload <= 0.0:
            return
        increment = (
            overload
            * dt_s
            * self.calibration.overload_damage_fraction_per_s
        )
        state.overload_damage_fraction = _clamp(
            state.overload_damage_fraction + increment, 0.0, 1.0
        )
        target = state.target
        if state.interaction == "hemostasis":
            source = self.patient.bleeding.sources[target]
            target = source.organ_id
            source.injury_fraction = _clamp(
                source.injury_fraction + 0.5 * increment, 0.0, 1.0
            )
        tissue = self.patient.tissue_state.get(target)
        tissue.integrity_fraction = _clamp(
            tissue.integrity_fraction - increment, 0.0, 1.0
        )
        if (
            state.overload_damage_fraction - state.reported_damage_fraction
            >= 0.01
        ):
            severity = (
                state.overload_damage_fraction
                - state.reported_damage_fraction
            )
            state.reported_damage_fraction = state.overload_damage_fraction
            event = self.patient.damage.record(
                time_s=self.patient.time_s,
                target=target,
                kind="contact_overload",
                severity=severity,
                source_robot=state.source_robot,
                consequences={
                    "peak_force_n": state.peak_force_n,
                    "cumulative_damage_fraction": (
                        state.overload_damage_fraction
                    ),
                },
            )
            self.patient.event_bus.emit(
                PatientEvent(
                    self.patient.time_s,
                    "damage",
                    asdict(event),
                    source=state.source_robot,
                )
            )

    def step(self, dt_s: float) -> None:
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        calibration = self.calibration
        for key, state in self.states.items():
            frame = self._pending.get(key)
            if state.interaction == "exposure":
                (
                    active,
                    quality,
                    bilateral,
                    peak,
                    asymmetry,
                ) = self._force_quality(
                    frame,
                    target_force_n=calibration.exposure_target_force_per_pad_n,
                    soft_force_n=calibration.exposure_soft_force_per_pad_n,
                    hard_force_n=calibration.exposure_hard_force_per_pad_n,
                    maximum_asymmetry_n=(
                        calibration.exposure_maximum_asymmetry_n
                    ),
                )
                if active and frame is not None:
                    if not state.contact_active:
                        state.anchor_position_m = frame.tool_position_m
                    if (
                        frame.tool_position_m is not None
                        and state.anchor_position_m is not None
                    ):
                        state.traction_distance_m = self._distance(
                            frame.tool_position_m,
                            state.anchor_position_m,
                        )
                    retraction_target = quality * _clamp(
                        state.traction_distance_m
                        / calibration.exposure_full_retraction_distance_m,
                        0.0,
                        1.0,
                    )
                    compression_target = _clamp(
                        calibration.exposure_local_compression_at_target
                        * bilateral
                        / calibration.exposure_target_force_per_pad_n,
                        0.0,
                        1.0,
                    )
                else:
                    state.anchor_position_m = None
                    state.traction_distance_m = 0.0
                    retraction_target = 0.0
                    compression_target = 0.0
                state.retraction_fraction = _smooth(
                    state.retraction_fraction,
                    retraction_target,
                    dt,
                    (
                        calibration.contact_attack_time_s
                        if active
                        else calibration.tissue_recovery_time_s
                    ),
                )
                state.compression_fraction = _smooth(
                    state.compression_fraction,
                    compression_target,
                    dt,
                    (
                        calibration.contact_attack_time_s
                        if active
                        else calibration.contact_release_time_s
                    ),
                )
                soft = calibration.exposure_soft_force_per_pad_n
                hard = calibration.exposure_hard_force_per_pad_n
            else:
                (
                    active,
                    quality,
                    bilateral,
                    peak,
                    asymmetry,
                ) = self._force_quality(
                    frame,
                    target_force_n=calibration.hemostasis_target_force_per_pad_n,
                    soft_force_n=calibration.hemostasis_soft_force_per_pad_n,
                    hard_force_n=calibration.hemostasis_hard_force_per_pad_n,
                    maximum_asymmetry_n=(
                        calibration.hemostasis_maximum_asymmetry_n
                    ),
                )
                state.hemostatic_control_fraction = _smooth(
                    state.hemostatic_control_fraction,
                    quality if active else 0.0,
                    dt,
                    (
                        calibration.contact_attack_time_s
                        if active
                        else calibration.contact_release_time_s
                    ),
                )
                soft = calibration.hemostasis_soft_force_per_pad_n
                hard = calibration.hemostasis_hard_force_per_pad_n
            state.contact_active = active
            state.bilateral_force_n = bilateral
            state.peak_force_n = peak
            state.force_asymmetry_n = asymmetry
            self._apply_overload(
                state,
                dt_s=dt,
                soft_force_n=soft,
                hard_force_n=hard,
            )

        exposure_targets = {
            state.target
            for state in self.states.values()
            if state.interaction == "exposure"
        }
        for target in exposure_targets:
            matching = [
                state
                for state in self.states.values()
                if state.interaction == "exposure" and state.target == target
            ]
            retraction = max(
                (state.retraction_fraction for state in matching),
                default=0.0,
            )
            compression = max(
                (state.compression_fraction for state in matching),
                default=0.0,
            )
            self.patient.tissue_state.apply_contact_response(
                target,
                retraction_fraction=retraction,
                compression_fraction=compression,
            )
            perfusion_target = CONTACT_PERFUSION_TERRITORIES.get(
                target,
                target,
            )
            self.patient.perfusion.set_compression(
                perfusion_target,
                compression,
            )

        for source_id in self.patient.bleeding.sources:
            control = max(
                (
                    state.hemostatic_control_fraction
                    for state in self.states.values()
                    if state.interaction == "hemostasis"
                    and state.target == source_id
                ),
                default=0.0,
            )
            self.patient.bleeding._set_contact_compression(
                source_id,
                control,
            )
        self._pending.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "authority": "post_physics_contact_force_and_tool_pose",
            "calibration": asdict(self.calibration),
            "states": [
                asdict(state)
                for _, state in sorted(self.states.items())
            ],
        }


LAPAROTOMY_LAYERS = (
    "skin",
    "subcutaneous_fat",
    "fascia",
    "abdominal_wall",
    "peritoneum",
)


@dataclass(frozen=True)
class LaparotomyIncisionCalibration:
    """Research-informed gates for a staged midline laparotomy.

    The force envelope is deliberately identified as a surrogate.  It is based
    on ex-vivo porcine aorta scalpel experiments reporting 4--12 N break-in
    force and 2--4 N continuous force (Hu, Sun & Zhang, 2013,
    doi:10.1016/j.jmbbm.2012.10.017).  The 30 mm/s nominal travel is based on
    abdominal-skin simulant cutting work.  These values have not been fitted to
    this patient's abdominal layers and are not clinical limits.
    """

    incision_length_m: float = 0.18
    bridges_per_layer: int = 24
    initiation_force_n: float = 4.0
    propagation_force_n: float = 2.0
    maximum_research_force_n: float = 12.0
    nominal_speed_m_s: float = 0.03
    minimum_speed_m_s: float = 0.001
    maximum_speed_m_s: float = 0.06
    maximum_alignment_error_deg: float = 20.0
    break_in_distance_m: float = 0.005
    source_dois: tuple[str, ...] = (
        "10.1016/j.jmbbm.2012.10.017",
        "10.1088/0957-0233/20/4/045801",
    )
    calibration_status: str = (
        "cross_tissue_research_envelope_pending_abdominal_layer_bench"
    )

    def __post_init__(self) -> None:
        positive = {
            "incision_length_m": self.incision_length_m,
            "bridges_per_layer": float(self.bridges_per_layer),
            "initiation_force_n": self.initiation_force_n,
            "propagation_force_n": self.propagation_force_n,
            "maximum_research_force_n": self.maximum_research_force_n,
            "nominal_speed_m_s": self.nominal_speed_m_s,
            "minimum_speed_m_s": self.minimum_speed_m_s,
            "maximum_speed_m_s": self.maximum_speed_m_s,
            "break_in_distance_m": self.break_in_distance_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_research_force_n < self.initiation_force_n:
            raise ValueError(
                "maximum_research_force_n must cover initiation_force_n"
            )
        if not (
            self.minimum_speed_m_s
            <= self.nominal_speed_m_s
            <= self.maximum_speed_m_s
        ):
            raise ValueError("nominal incision speed is outside its gate")
        if self.bridges_per_layer < 2:
            raise ValueError("bridges_per_layer must be at least two")


@dataclass(frozen=True)
class IncisionContactSample:
    blade_contact: bool
    normal_force_n: float
    tangential_force_n: float
    advancement_m: float
    speed_m_s: float
    alignment_error_deg: float
    source_robot: str = "scalpel"


class PhysicalLaparotomyIncision:
    """Contact-gated controller for removable midline continuity bridges.

    PhysX does not mutate a deformable mesh's topology during a run.  The
    supported physical representation is therefore two pre-segmented wound
    flaps joined by ordered, removable continuity elements.  This controller
    releases those elements only after contact, force, speed, alignment,
    distance and cutting-work gates all pass.
    """

    def __init__(
        self,
        patient: "DynamicSurgicalPatient",
        calibration: LaparotomyIncisionCalibration | None = None,
    ):
        self.patient = patient
        self.calibration = calibration or LaparotomyIncisionCalibration()
        self.layer_index = 0
        self.bridge_index = 0
        self.layer_advancement_m = 0.0
        self.layer_work_j = 0.0
        self.break_in_advancement_m = 0.0
        self.started = False
        self.complete = False
        self.released_bridge_ids: list[str] = []
        self.rejected_samples = 0
        self.overload_samples = 0

    @property
    def active_layer(self) -> str | None:
        if self.complete:
            return None
        return LAPAROTOMY_LAYERS[self.layer_index]

    @property
    def bridge_length_m(self) -> float:
        return (
            self.calibration.incision_length_m
            / self.calibration.bridges_per_layer
        )

    @property
    def progress_fraction(self) -> float:
        completed = (
            self.layer_index * self.calibration.bridges_per_layer
            + self.bridge_index
        )
        total = len(LAPAROTOMY_LAYERS) * self.calibration.bridges_per_layer
        return _clamp(completed / total, 0.0, 1.0)

    def _reject(self, reason: str) -> dict[str, Any]:
        self.rejected_samples += 1
        return {
            "accepted": False,
            "reason": reason,
            "active_layer": self.active_layer,
            "progress_fraction": self.progress_fraction,
            "released_bridge_ids": [],
        }

    def advance(self, sample: IncisionContactSample) -> dict[str, Any]:
        if self.complete:
            return self._reject("incision_complete")
        normal_force = _nonnegative(sample.normal_force_n, "normal_force_n")
        tangential_force = _nonnegative(
            sample.tangential_force_n,
            "tangential_force_n",
        )
        advancement = _nonnegative(sample.advancement_m, "advancement_m")
        speed = _nonnegative(sample.speed_m_s, "speed_m_s")
        alignment = _nonnegative(
            sample.alignment_error_deg,
            "alignment_error_deg",
        )
        if not sample.blade_contact:
            return self._reject("no_blade_contact")
        if alignment > self.calibration.maximum_alignment_error_deg:
            return self._reject("blade_alignment_outside_gate")
        if not (
            self.calibration.minimum_speed_m_s
            <= speed
            <= self.calibration.maximum_speed_m_s
        ):
            return self._reject("blade_speed_outside_gate")
        if advancement <= 0.0:
            return self._reject("no_forward_advancement")

        resultant_force = math.hypot(normal_force, tangential_force)
        force_gate = (
            self.calibration.initiation_force_n
            if not self.started
            else self.calibration.propagation_force_n
        )
        if resultant_force < force_gate:
            return self._reject("cutting_force_below_gate")
        overloaded = (
            resultant_force > self.calibration.maximum_research_force_n
        )
        if overloaded:
            self.overload_samples += 1

        self.break_in_advancement_m += advancement
        if (
            not self.started
            and self.break_in_advancement_m
            < self.calibration.break_in_distance_m
        ):
            return {
                "accepted": True,
                "reason": "break_in_accumulating",
                "active_layer": self.active_layer,
                "progress_fraction": self.progress_fraction,
                "released_bridge_ids": [],
                "overload": overloaded,
            }
        self.started = True
        self.layer_advancement_m += advancement
        self.layer_work_j += tangential_force * advancement

        released: list[str] = []
        work_per_bridge_j = (
            self.calibration.propagation_force_n * self.bridge_length_m
        )
        distance_limited = int(
            self.layer_advancement_m / self.bridge_length_m
        )
        work_limited = int(self.layer_work_j / work_per_bridge_j)
        target_bridge_count = min(
            self.calibration.bridges_per_layer,
            distance_limited,
            work_limited,
        )
        while self.bridge_index < target_bridge_count:
            bridge_id = (
                f"{LAPAROTOMY_LAYERS[self.layer_index]}:"
                f"{self.bridge_index:03d}"
            )
            self.released_bridge_ids.append(bridge_id)
            released.append(bridge_id)
            self.bridge_index += 1

        completed_layer = None
        if self.bridge_index == self.calibration.bridges_per_layer:
            completed_layer = LAPAROTOMY_LAYERS[self.layer_index]
            self.patient.tissue_state.cut(
                completed_layer,
                severity=1.0 / len(LAPAROTOMY_LAYERS),
            )
            self.layer_index += 1
            self.bridge_index = 0
            self.layer_advancement_m = 0.0
            self.layer_work_j = 0.0
            self.break_in_advancement_m = 0.0
            self.started = False
            if self.layer_index == len(LAPAROTOMY_LAYERS):
                self.complete = True
                self.patient.tissue_state.set_access_state("open")
                self.patient.set_procedure_stage("access_open")

        self.patient.tissue_state.wound_open_fraction = self.progress_fraction
        event = {
            "accepted": True,
            "reason": "continuity_released" if released else "propagating",
            "active_layer": self.active_layer,
            "completed_layer": completed_layer,
            "progress_fraction": self.progress_fraction,
            "released_bridge_ids": released,
            "resultant_force_n": resultant_force,
            "overload": overloaded,
            "calibration_status": self.calibration.calibration_status,
        }
        if released:
            self.patient.event_bus.emit(
                PatientEvent(
                    self.patient.time_s,
                    "physical_incision_progress",
                    dict(event),
                    source=sample.source_robot,
                )
            )
        return event

    def snapshot(self) -> dict[str, Any]:
        return {
            "representation": (
                "presegmented_deformable_flaps_with_removable_continuity"
            ),
            "active_layer": self.active_layer,
            "layer_index": self.layer_index,
            "bridge_index": self.bridge_index,
            "progress_fraction": self.progress_fraction,
            "complete": self.complete,
            "released_bridge_ids": list(self.released_bridge_ids),
            "rejected_samples": self.rejected_samples,
            "overload_samples": self.overload_samples,
            "calibration": asdict(self.calibration),
            "clinical_validation": False,
        }


@dataclass
class DamageEvent:
    id: str
    time_s: float
    target: str
    kind: str
    severity: float
    source_robot: str
    active: bool = True
    consequences: dict[str, float] = field(default_factory=dict)
    notes: str = ""


class DamageRegistry:
    def __init__(self):
        self.events: dict[str, DamageEvent] = {}
        self._counter = 0

    def record(
        self,
        *,
        time_s: float,
        target: str,
        kind: str,
        severity: float,
        source_robot: str,
        consequences: Mapping[str, float] | None = None,
        notes: str = "",
    ) -> DamageEvent:
        self._counter += 1
        event = DamageEvent(
            id=f"damage_{self._counter:05d}",
            time_s=float(time_s),
            target=target,
            kind=kind,
            severity=_clamp(severity, 0, 1),
            source_robot=source_robot,
            consequences=dict(consequences or {}),
            notes=notes,
        )
        self.events[event.id] = event
        return event

    def resolve(self, event_id: str) -> None:
        self.events[event_id].active = False

    def active_events(self) -> list[DamageEvent]:
        return [event for event in self.events.values() if event.active]

    def severity_for(self, target: str, kind: str | None = None) -> float:
        return _clamp(
            sum(
                e.severity
                for e in self.active_events()
                if e.target == target and (kind is None or e.kind == kind)
            ),
            0,
            1,
        )


@dataclass
class InterventionEvent:
    id: str
    time_s: float
    source_robot: str
    action: str
    target: str
    parameters: dict[str, Any]
    result: dict[str, Any]


class InterventionRegistry:
    def __init__(self, patient: "DynamicSurgicalPatient"):
        self.patient = patient
        self.history: list[InterventionEvent] = []
        self._counter = 0

    def _record(
        self,
        source_robot: str,
        action: str,
        target: str,
        parameters: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> InterventionEvent:
        self._counter += 1
        event = InterventionEvent(
            id=f"intervention_{self._counter:05d}",
            time_s=self.patient.time_s,
            source_robot=source_robot,
            action=action,
            target=target,
            parameters=dict(parameters),
            result=dict(result),
        )
        self.history.append(event)
        if hasattr(self.patient, "event_bus"):
            self.patient.event_bus.emit(
                PatientEvent(
                    self.patient.time_s,
                    "intervention",
                    {
                        "id": event.id,
                        "action": action,
                        "target": target,
                        "parameters": dict(parameters),
                        "result": dict(result),
                    },
                    source=source_robot,
                )
            )
        return event

    def apply(self, event: Mapping[str, Any]) -> InterventionEvent:
        action = str(event["action"])
        target = str(event.get("target", "patient"))
        source = str(event.get("source_robot", "external"))
        params = dict(event.get("parameters", {}))
        dispatch = {
            "set_access_state": self.set_access_state,
            "dissection": self.apply_dissection,
            "wound_preparation": self.apply_wound_preparation,
            "seal_divide": self.apply_seal_divide,
            "anastomosis": self.apply_anastomosis,
            "closure": self.apply_closure,
            "dressing": self.apply_dressing,
            "perfusion_scan": self.apply_perfusion_scan,
            "crystalloid": self.infuse_crystalloid,
            "blood_transfusion": self.transfuse_blood,
        }
        if action not in dispatch:
            raise KeyError(f"unknown patient intervention {action!r}")
        return dispatch[action](target=target, source_robot=source, **params)

    def set_access_state(
        self,
        state: str | None = None,
        *,
        target: str = "abdomen",
        source_robot: str = "procedure_setup",
        **kwargs,
    ) -> InterventionEvent:
        selected = state or target
        self.patient.tissue_state.set_access_state(selected)
        return self._record(
            source_robot,
            "set_access_state",
            "abdomen",
            {"state": selected},
            {"access_state": selected},
        )

    def apply_dissection(
        self,
        *,
        target: str,
        method: str,
        source_robot: str = "safeplane_dissection_robot",
        protected_structure: str | None = None,
        injury: bool = False,
        **kwargs,
    ) -> InterventionEvent:
        result = {"released": True, "method": method}
        if target.startswith("adhesion_"):
            self.patient.released_adhesions.add(target)
            self.patient.tissue_state.get("adhesions").active_adhesions.discard(target)
        else:
            self.patient.tissue_state.cut(target, 0.06 if method == "blunt" else 0.12)
        if injury and protected_structure:
            severity = float(kwargs.get("severity", 0.45))
            self.patient.damage.record(
                time_s=self.patient.time_s,
                target=protected_structure,
                kind="iatrogenic_dissection_injury",
                severity=severity,
                source_robot=source_robot,
            )
            self.patient._apply_damage_consequence(
                protected_structure, "cut", severity, source_robot
            )
            result["complication"] = protected_structure
        return self._record(
            source_robot,
            "dissection",
            target,
            {
                "method": method,
                "protected_structure": protected_structure,
                "injury": injury,
            },
            result,
        )

    def apply_wound_preparation(
        self,
        *,
        target: str,
        debridement_fraction: float = 0.5,
        contamination_reduction: float = 0.4,
        source_robot: str = "wound_preparation_robot",
        **kwargs,
    ) -> InterventionEvent:
        self.patient.tissue_state.debride(
            target, debridement_fraction, contamination_reduction
        )
        return self._record(
            source_robot,
            "wound_preparation",
            target,
            {
                "debridement_fraction": debridement_fraction,
                "contamination_reduction": contamination_reduction,
            },
            {"prepared": True},
        )

    def apply_seal_divide(
        self,
        *,
        target: str,
        seal_quality: float = 0.95,
        division_complete: bool = True,
        distal_region: str | None = None,
        source_robot: str = "adaptive_seal_divide_robot",
        **kwargs,
    ) -> InterventionEvent:
        self.patient.tissue_state.seal(target, seal_quality)
        if division_complete:
            self.patient.tissue_state.cut(target, 0.35)
        if distal_region:
            self.patient.perfusion.set_occlusion(
                distal_region, 1.0 - seal_quality * 0.15
            )
        return self._record(
            source_robot,
            "seal_divide",
            target,
            {
                "seal_quality": seal_quality,
                "division_complete": division_complete,
                "distal_region": distal_region,
            },
            {"sealed": seal_quality, "divided": division_complete},
        )

    def apply_anastomosis(
        self,
        *,
        target: str,
        patency_fraction: float = 0.9,
        leak_area_mm2: float = 0.0,
        perfusion_restoration: float = 0.9,
        source_robot: str = "adaptive_anastomosis_robot",
        **kwargs,
    ) -> InterventionEvent:
        self.patient.anastomoses[target] = {
            "patency_fraction": _clamp(patency_fraction, 0, 1),
            "leak_area_mm2": max(0.0, leak_area_mm2),
            "perfusion_restoration": _clamp(perfusion_restoration, 0, 1),
        }
        region = str(kwargs.get("region", target))
        if region in self.patient.perfusion.regions:
            self.patient.perfusion.set_occlusion(
                region, 1.0 - _clamp(perfusion_restoration, 0, 1)
            )
            self.patient.perfusion.set_leak(region, _clamp(leak_area_mm2 / 20.0, 0, 1))
        return self._record(
            source_robot,
            "anastomosis",
            target,
            {
                "patency_fraction": patency_fraction,
                "leak_area_mm2": leak_area_mm2,
                "perfusion_restoration": perfusion_restoration,
            },
            {"registered": True},
        )

    def apply_closure(
        self,
        *,
        target: str,
        method: str,
        closure_fraction: float = 1.0,
        source_robot: str = "environment_contact_effects",
        **kwargs,
    ) -> InterventionEvent:
        fraction = _clamp(closure_fraction, 0, 1)
        if "staple" in method:
            self.patient.tissue_state.staple(target, fraction)
        else:
            self.patient.tissue_state.suture(target, fraction)
        if "adhesive" in method:
            self.patient.tissue_state.seal(target, fraction)
        if target in {"skin", "abdominal_wall", "abdomen"}:
            self.patient.tissue_state.wound_open_fraction = _clamp(1.0 - fraction, 0, 1)
            if fraction >= 0.999:
                self.patient.tissue_state.access_state = "intact"
        return self._record(
            source_robot,
            "closure",
            target,
            {"method": method, "closure_fraction": fraction},
            {"wound_open_fraction": self.patient.tissue_state.wound_open_fraction},
        )

    def apply_dressing(
        self,
        *,
        target: str,
        pressure_kpa: float = 0.0,
        seal_fraction: float = 1.0,
        source_robot: str = "environment_contact_effects",
        **kwargs,
    ) -> InterventionEvent:
        self.patient.dressing_state = {
            "target": target,
            "pressure_kpa": float(pressure_kpa),
            "seal_fraction": _clamp(seal_fraction, 0, 1),
            "applied": True,
        }
        compression = _clamp(abs(min(0.0, pressure_kpa)) / 40.0, 0, 0.6)
        self.patient.perfusion.set_compression("skin", compression)
        return self._record(
            source_robot,
            "dressing",
            target,
            {"pressure_kpa": pressure_kpa, "seal_fraction": seal_fraction},
            {"skin_compression_fraction": compression},
        )

    def apply_perfusion_scan(
        self,
        *,
        target: str = "abdomen",
        findings: Mapping[str, Any] | None = None,
        source_robot: str = "perfusion_viability_robot",
        **kwargs,
    ) -> InterventionEvent:
        findings = dict(findings or {})
        if not findings:
            findings = {
                organ: {
                    "relative_flow_fraction": state.relative_flow_fraction,
                    "viability_fraction": state.viability_fraction,
                }
                for organ, state in self.patient.perfusion.regions.items()
            }
        self.patient.last_perfusion_scan = {
            "time_s": self.patient.time_s,
            "target": target,
            "findings": findings,
        }
        return self._record(
            source_robot, "perfusion_scan", target, {}, {"finding_count": len(findings)}
        )

    def infuse_crystalloid(
        self,
        *,
        target: str = "patient",
        volume_ml: float,
        source_robot: str = "anesthesia",
        **kwargs,
    ) -> InterventionEvent:
        self.patient.fluid_balance.infuse_crystalloid(volume_ml)
        return self._record(
            source_robot,
            "crystalloid",
            target,
            {"volume_ml": volume_ml},
            {
                "intravascular_volume_ml": self.patient.fluid_balance.intravascular_volume_ml
            },
        )

    def transfuse_blood(
        self,
        *,
        target: str = "patient",
        volume_ml: float,
        source_robot: str = "anesthesia",
        **kwargs,
    ) -> InterventionEvent:
        self.patient.fluid_balance.transfuse_blood(volume_ml)
        return self._record(
            source_robot,
            "blood_transfusion",
            target,
            {"volume_ml": volume_ml},
            {
                "intravascular_volume_ml": self.patient.fluid_balance.intravascular_volume_ml
            },
        )


class RobotInterventionAdapter:
    """Typed adapter used by all DrAnmar surgical robot systems."""

    def __init__(self, patient: "DynamicSurgicalPatient"):
        self.patient = patient

    def dissection(
        self,
        *,
        target: str,
        method: str,
        protected_structure: str | None = None,
        injury: bool = False,
        severity: float = 0.45,
    ) -> InterventionEvent:
        return self.patient.interventions.apply_dissection(
            target=target,
            method=method,
            protected_structure=protected_structure,
            injury=injury,
            severity=severity,
            source_robot="safeplane_dissection_robot",
        )

    def wound_preparation(
        self,
        *,
        target: str,
        debridement_fraction: float,
        contamination_reduction: float,
        irrigation_ml: float = 0.0,
        aspirated_ml: float = 0.0,
    ) -> InterventionEvent:
        if irrigation_ml > 0.0:
            self.patient.fluid_balance.add_irrigation(irrigation_ml)
        if aspirated_ml > 0.0:
            self.patient.fluid_balance.recover_irrigation(aspirated_ml)
            self.patient.fluid_balance.collect_suction(aspirated_ml)
        return self.patient.interventions.apply_wound_preparation(
            target=target,
            debridement_fraction=debridement_fraction,
            contamination_reduction=contamination_reduction,
            source_robot="wound_preparation_robot",
        )

    def seal_and_divide(
        self,
        *,
        target: str,
        seal_quality: float,
        division_complete: bool,
        distal_region: str | None = None,
    ) -> InterventionEvent:
        return self.patient.interventions.apply_seal_divide(
            target=target,
            seal_quality=seal_quality,
            division_complete=division_complete,
            distal_region=distal_region,
            source_robot="adaptive_seal_divide_robot",
        )

    def anastomosis(
        self,
        *,
        target: str,
        patency_fraction: float,
        leak_area_mm2: float,
        perfusion_restoration: float,
        region: str | None = None,
    ) -> InterventionEvent:
        kwargs: dict[str, Any] = {}
        if region is not None:
            kwargs["region"] = region
        return self.patient.interventions.apply_anastomosis(
            target=target,
            patency_fraction=patency_fraction,
            leak_area_mm2=leak_area_mm2,
            perfusion_restoration=perfusion_restoration,
            source_robot="adaptive_anastomosis_robot",
            **kwargs,
        )

    def close_wound(
        self, *, target: str, method: str, closure_fraction: float
    ) -> InterventionEvent:
        return self.patient.interventions.apply_closure(
            target=target,
            method=method,
            closure_fraction=closure_fraction,
            source_robot="closure_robot",
        )

    def dressing(
        self, *, target: str, pressure_kpa: float, seal_fraction: float = 1.0
    ) -> InterventionEvent:
        return self.patient.interventions.apply_dressing(
            target=target,
            pressure_kpa=pressure_kpa,
            seal_fraction=seal_fraction,
            source_robot="environment_contact_effects",
        )

    def perfusion_scan(
        self, *, target: str = "abdomen", findings: Mapping[str, Any] | None = None
    ) -> InterventionEvent:
        return self.patient.interventions.apply_perfusion_scan(
            target=target,
            findings=findings,
            source_robot="perfusion_viability_robot",
        )

    def crystalloid(self, *, volume_ml: float) -> InterventionEvent:
        return self.patient.interventions.infuse_crystalloid(volume_ml=volume_ml)

    def transfuse(self, *, volume_ml: float) -> InterventionEvent:
        return self.patient.interventions.transfuse_blood(volume_ml=volume_ml)


class OrganMotionModel:
    def __init__(self, components: Sequence[Mapping[str, Any]]):
        self.components = {str(c["id"]): dict(c) for c in components}
        self.displacements_m: dict[str, tuple[float, float, float]] = {
            key: (0, 0, 0) for key in self.components
        }
        self.pulsation_scale: dict[str, float] = {key: 1.0 for key in self.components}

    def step(
        self, respiration: RespirationModel, cardiovascular: CardiovascularModel
    ) -> None:
        respiratory = respiration.displacement_fraction
        pulse = max(0.0, math.sin(cardiovascular.heart_phase_rad))
        for organ, cfg in self.components.items():
            coupling = cfg.get("respiration_coupling_m", [0, 0, 0])
            self.displacements_m[organ] = tuple(
                float(x) * respiratory for x in coupling
            )
            self.pulsation_scale[organ] = (
                1.0 + float(cfg.get("pulsation_fraction", 0.0)) * pulse
            )

    def displacement(self, organ_id: str) -> tuple[float, float, float]:
        return self.displacements_m.get(organ_id, (0, 0, 0))


class DynamicSurgicalPatient:
    """Shared solver-independent abdominal physiology state."""

    def __init__(
        self,
        *,
        anatomy: Mapping[str, Any] | None = None,
        physiology: Mapping[str, Any] | None = None,
        seed: int = 20260725,
        procedure_stage: str = "closed",
        condition: str = "healthy",
        body_habitus: str = "baseline",
    ):
        if procedure_stage not in VALID_PROCEDURE_STAGES:
            raise ValueError(f"unsupported procedure_stage {procedure_stage!r}")
        if condition not in VALID_CONDITIONS:
            raise ValueError(f"unsupported condition {condition!r}")
        if body_habitus not in VALID_HABITUS:
            raise ValueError(f"unsupported body_habitus {body_habitus!r}")
        self.seed = int(seed)
        self._initial_procedure_stage = procedure_stage
        self._initial_condition = condition
        self._initial_body_habitus = body_habitus
        self.procedure_stage = procedure_stage
        self.condition = condition
        self.body_habitus = body_habitus
        self.event_bus = PatientEventBus()
        self.anatomy = dict(anatomy or load_anatomy_manifest())
        self.physiology_config = dict(physiology or load_physiology_network())
        baseline = self.physiology_config["baseline"]
        habitus_blood_scale = {
            "lean": 0.94,
            "baseline": 1.0,
            "increased_visceral_fat": 1.10,
        }[body_habitus]
        blood_volume_ml = float(baseline["blood_volume_ml"]) * habitus_blood_scale
        self.time_s = 0.0
        self.respiration = RespirationModel(
            respiratory_rate_bpm=float(baseline["respiratory_rate_bpm"]),
            tidal_volume_ml=float(baseline["tidal_volume_ml"]),
            spo2_fraction=float(baseline["spo2_fraction"]),
            paco2_mmhg=float(baseline["paco2_mmhg"]),
            diaphragm_excursion_m=float(
                self.physiology_config["respiration"]["diaphragm_excursion_m"]
            ),
        )
        self.cardiovascular = CardiovascularModel(
            heart_rate_bpm=float(baseline["heart_rate_bpm"]),
            stroke_volume_ml=float(baseline["stroke_volume_ml"]),
            cardiac_output_l_min=float(baseline["heart_rate_bpm"])
            * float(baseline["stroke_volume_ml"])
            / 1000.0,
            systemic_vascular_resistance_mmhg_min_l=float(
                baseline["systemic_vascular_resistance_mmhg_min_l"]
            ),
            central_venous_pressure_mmhg=float(
                baseline["central_venous_pressure_mmhg"]
            ),
            mean_arterial_pressure_mmhg=float(baseline["mean_arterial_pressure_mmhg"]),
            systolic_pressure_mmhg=float(baseline["systolic_pressure_mmhg"]),
            diastolic_pressure_mmhg=float(baseline["diastolic_pressure_mmhg"]),
        )
        self.coagulation = CoagulationModel(
            platelet_function_fraction=float(baseline["platelet_function_fraction"]),
            fibrinogen_g_l=float(baseline["fibrinogen_g_l"]),
            inr=float(baseline["inr"]),
        )
        self.fluid_balance = FluidBalanceModel(
            baseline_blood_volume_ml=blood_volume_ml,
            intravascular_volume_ml=blood_volume_ml,
        )
        self.fluids = self.fluid_balance
        self.oxygen = OxygenDeliveryModel(
            hemoglobin_g_dl=float(baseline["hemoglobin_g_dl"]),
            lactate_mmol_l=float(baseline["lactate_mmol_l"]),
        )
        self.temperature = TemperatureModel(
            core_temperature_c=float(baseline["core_temperature_c"]),
            ambient_temperature_c=float(
                self.physiology_config["temperature"]["ambient_temperature_c"]
            ),
        )
        self.renal = RenalModel(urine_flow_ml_h=float(baseline["urine_output_ml_h"]))
        self.biliary = BiliaryModel(bile_flow_ml_h=float(baseline["bile_flow_ml_h"]))
        organ_ids = [str(c["id"]) for c in self.anatomy["components"]]
        self.tissue_state = TissueStateRegistry(organ_ids)
        self.incision = PhysicalLaparotomyIncision(self)
        self.organ_motion = OrganMotionModel(self.anatomy["components"])
        self.damage = DamageRegistry()
        self.perfusion = PerfusionModel(
            self, self.physiology_config["regional_perfusion"]
        )
        self.bleeding = BleedingModel(self)
        self.vital_signs = VitalSignsModel()
        self.contact_effects = ContactDrivenPatientEffects(self)
        self.contacts = self.contact_effects
        self.interventions = InterventionRegistry(self)
        self.robot = RobotInterventionAdapter(self)
        self.released_adhesions: set[str] = set()
        self.anastomoses: dict[str, dict[str, float]] = {}
        self.dressing_state: dict[str, Any] = {
            "applied": False,
            "pressure_kpa": 0.0,
            "seal_fraction": 0.0,
        }
        self.last_perfusion_scan: dict[str, Any] | None = None
        self.anesthetic_depression_fraction = 0.0
        self.irrigation_exposure_fraction = 0.0
        self.exposed_surface_fraction = 0.02
        self._last_bile_leak_ml = 0.0
        self._configure_condition(condition)
        self.set_procedure_stage(procedure_stage, emit=False)
        self.vital_signs.update(self)

    def _configure_condition(self, condition: str) -> None:
        if condition == "hemorrhage":
            self.bleeding.create_source(
                "baseline_mesenteric_hemorrhage",
                "major_vessels",
                vessel_radius_m=0.0018,
                injury_fraction=0.85,
                kind="arterial",
            )
        elif condition == "bowel_ischemia":
            self.perfusion.set_occlusion("small_bowel", 0.82)
        elif condition == "bile_leak":
            self.biliary.duct_injury_fraction = 0.70
            self.tissue_state.cut("gallbladder", 0.20)
        elif condition == "ureter_injury":
            self.tissue_state.cut("ureters", 0.25)
            event = self.damage.record(
                time_s=self.time_s,
                target="ureters",
                kind="urine_leak",
                severity=0.65,
                source_robot="scenario_initialization",
            )
            self.event_bus.emit(
                PatientEvent(
                    self.time_s,
                    "damage",
                    asdict(event),
                    source="scenario_initialization",
                )
            )
        elif condition == "liver_tumor":
            self.tissue_state.get("liver_tumor").integrity_fraction = 0.88
            self.perfusion.set_leak("liver", 0.12)
        elif condition == "dense_adhesions":
            adhesion_ids = {f"adhesion_{index:02d}" for index in range(24)}
            self.tissue_state.get("adhesions").active_adhesions.update(adhesion_ids)
        elif condition == "postoperative":
            self.tissue_state.staple("abdominal_wall", 1.0)
            self.tissue_state.staple("skin", 1.0)
            self.tissue_state.seal("skin", 0.9)
            self.tissue_state.wound_open_fraction = 0.0
            self.dressing_state = {
                "target": "skin",
                "pressure_kpa": -8.0,
                "seal_fraction": 0.95,
                "applied": True,
            }

    def set_procedure_stage(self, stage: str, *, emit: bool = True) -> None:
        if stage not in VALID_PROCEDURE_STAGES:
            raise ValueError(f"unsupported procedure_stage {stage!r}")
        self.procedure_stage = stage
        exposure_by_stage = {
            "closed": 0.02,
            "access_open": 0.10,
            "exposed": 0.22,
            "dissection": 0.28,
            "hemostasis": 0.28,
            "division": 0.28,
            "reconstruction": 0.24,
            "closure": 0.10,
            "dressed": 0.03,
        }
        self.exposed_surface_fraction = exposure_by_stage[stage]
        self.tissue_state.set_access_state(
            "intact" if stage in {"closed", "dressed"} else "open"
        )
        if stage == "closed":
            self.tissue_state.wound_open_fraction = 0.0
        elif stage in {
            "access_open",
            "exposed",
            "dissection",
            "hemostasis",
            "division",
            "reconstruction",
        }:
            self.tissue_state.wound_open_fraction = max(
                self.tissue_state.wound_open_fraction, 0.85
            )
        if emit:
            self.event_bus.emit(
                PatientEvent(
                    self.time_s,
                    "procedure_stage",
                    {"stage": stage},
                    source="procedure_orchestrator",
                )
            )

    def step(self, dt_s: float) -> VitalSignsModel:
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt_s must be positive and finite")
        self.time_s += dt
        self.contact_effects.step(dt)
        tissue_supply = sum(
            s.oxygen_supply_ratio for s in self.perfusion.regions.values()
        ) / max(len(self.perfusion.regions), 1)
        self.respiration.step(
            dt,
            metabolic_scale=1.0
            + 0.2 * max(0, self.temperature.core_temperature_c - 37),
            perfusion_scale=self.perfusion.global_perfusion_fraction,
        )
        self.cardiovascular.step(
            dt,
            blood_volume_fraction=self.fluid_balance.blood_volume_fraction,
            oxygenation_fraction=self.respiration.spo2_fraction,
            temperature_c=self.temperature.core_temperature_c,
            anesthetic_depression_fraction=self.anesthetic_depression_fraction,
            map_target_mmhg=float(
                self.physiology_config["cardiovascular"]["map_target_mmhg"]
            ),
        )
        blood_loss = self.bleeding.step(dt)
        self.perfusion.step(dt)
        tissue_supply = sum(
            s.oxygen_supply_ratio for s in self.perfusion.regions.values()
        ) / max(len(self.perfusion.regions), 1)
        self.oxygen.step(
            dt,
            cardiac_output_l_min=self.cardiovascular.cardiac_output_l_min,
            spo2_fraction=self.respiration.spo2_fraction,
            blood_volume_fraction=self.fluid_balance.blood_volume_fraction,
            tissue_supply_ratio=tissue_supply,
        )
        thermal_exposure = _clamp(
            max(self.irrigation_exposure_fraction, self.exposed_surface_fraction),
            0.0,
            1.0,
        )
        self.temperature.step(
            dt,
            blood_loss_ml_s=blood_loss / dt,
            irrigation_exposure_fraction=thermal_exposure,
        )
        self.coagulation.step(
            dt,
            temperature_c=self.temperature.core_temperature_c,
            blood_volume_fraction=self.fluid_balance.blood_volume_fraction,
            crystalloid_ml=self.fluid_balance.crystalloid_input_ml,
            lactate_mmol_l=self.oxygen.lactate_mmol_l,
        )
        left = self.perfusion.regions.get(
            "left_kidney", RegionalPerfusionState("left_kidney")
        ).relative_flow_fraction
        right = self.perfusion.regions.get(
            "right_kidney", RegionalPerfusionState("right_kidney")
        ).relative_flow_fraction
        vasopressin = _clamp(
            (75 - self.cardiovascular.mean_arterial_pressure_mmhg) / 45, 0, 1
        )
        urine = self.renal.step(
            dt,
            map_mmhg=self.cardiovascular.mean_arterial_pressure_mmhg,
            left_flow_fraction=left,
            right_flow_fraction=right,
            vasopressin_drive=vasopressin,
        )
        self.fluid_balance.urine_output_ml += urine
        self.fluid_balance.intravascular_volume_ml = max(
            0, self.fluid_balance.intravascular_volume_ml - urine
        )
        liver = self.perfusion.regions.get(
            "liver", RegionalPerfusionState("liver")
        ).relative_flow_fraction
        bile, leak = self.biliary.step(dt, liver_perfusion_fraction=liver)
        self.fluid_balance.bile_output_ml += leak
        self._last_bile_leak_ml = leak
        self.organ_motion.step(self.respiration, self.cardiovascular)
        self.vital_signs.update(self)
        self.event_bus.emit(
            PatientEvent(
                self.time_s,
                "physiology_step",
                {
                    "map_mmhg": self.vital_signs.mean_arterial_pressure_mmhg,
                    "heart_rate_bpm": self.vital_signs.heart_rate_bpm,
                    "active_blood_loss_ml_min": self.vital_signs.active_blood_loss_ml_min,
                    "global_perfusion_fraction": self.vital_signs.global_perfusion_fraction,
                    "clinical_status": self.vital_signs.clinical_status,
                },
                source="physiology",
            )
        )
        return self.vital_signs

    def _apply_damage_consequence(
        self, target: str, kind: str, severity: float, source_robot: str
    ) -> None:
        if kind != "puncture":
            self.tissue_state.cut(target, 0.25 * severity)
        lower = target.lower()
        if "vessel" in lower or "arter" in lower or "vein" in lower:
            source_id = f"{target}_bleed_{len(self.bleeding.sources) + 1}"
            self.bleeding.create_source(
                source_id,
                target,
                vessel_radius_m=0.0018 * max(severity, 0.2),
                injury_fraction=severity,
                kind="arterial" if "arter" in lower else "venous",
            )
        if "ureter" in lower:
            self.damage.record(
                time_s=self.time_s,
                target=target,
                kind="urine_leak",
                severity=severity,
                source_robot=source_robot,
            )
        if "duct" in lower or "gallbladder" in lower:
            self.biliary.duct_injury_fraction = max(
                self.biliary.duct_injury_fraction, severity
            )
        if "nerve" in lower:
            self.damage.record(
                time_s=self.time_s,
                target=target,
                kind="nerve_conduction_loss",
                severity=severity,
                source_robot=source_robot,
            )

    def start_bleeding(
        self,
        source_id: str,
        organ_id: str,
        *,
        vessel_radius_m: float,
        injury_fraction: float = 1.0,
        kind: str = "venous",
    ) -> BleedSource:
        source = self.bleeding.create_source(
            source_id,
            organ_id,
            vessel_radius_m=vessel_radius_m,
            injury_fraction=injury_fraction,
            kind=kind,
        )
        self.event_bus.emit(
            PatientEvent(
                self.time_s, "bleeding_started", asdict(source), source="patient"
            )
        )
        return source

    def cut(
        self,
        structure: str,
        severity: float,
        *,
        source_robot: str = "external",
        location_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> DamageEvent:
        severity = _clamp(severity, 0.0, 1.0)
        event = self.damage.record(
            time_s=self.time_s,
            target=structure,
            kind="cut",
            severity=severity,
            source_robot=source_robot,
            consequences={
                "x_m": float(location_m[0]),
                "y_m": float(location_m[1]),
                "z_m": float(location_m[2]),
            },
        )
        self._apply_damage_consequence(structure, "cut", severity, source_robot)
        self.event_bus.emit(
            PatientEvent(self.time_s, "damage", asdict(event), source=source_robot)
        )
        return event

    def puncture(
        self,
        structure: str,
        severity: float = 0.05,
        *,
        source_robot: str = "external",
        location_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> DamageEvent:
        severity = _clamp(severity, 0.0, 1.0)
        self.tissue_state.puncture(structure, severity)
        event = self.damage.record(
            time_s=self.time_s,
            target=structure,
            kind="puncture",
            severity=severity,
            source_robot=source_robot,
            consequences={
                "x_m": float(location_m[0]),
                "y_m": float(location_m[1]),
                "z_m": float(location_m[2]),
            },
        )
        lower = structure.lower()
        if "vessel" in lower or "arter" in lower or "vein" in lower:
            self._apply_damage_consequence(
                structure, "puncture", severity, source_robot
            )
        self.event_bus.emit(
            PatientEvent(self.time_s, "damage", asdict(event), source=source_robot)
        )
        return event

    def infuse(
        self,
        *,
        crystalloid_ml: float = 0.0,
        blood_ml: float = 0.0,
        source: str = "anesthesia",
    ) -> None:
        if crystalloid_ml > 0.0:
            self.interventions.infuse_crystalloid(
                volume_ml=crystalloid_ml, source_robot=source
            )
        if blood_ml > 0.0:
            self.interventions.transfuse_blood(volume_ml=blood_ml, source_robot=source)

    def reset(self) -> None:
        self.__init__(
            anatomy=self.anatomy,
            physiology=self.physiology_config,
            seed=self.seed,
            procedure_stage=self._initial_procedure_stage,
            condition=self._initial_condition,
            body_habitus=self._initial_body_habitus,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "dr.anmar.dynamic-patient-snapshot.v1",
            "time_s": self.time_s,
            "procedure_stage": self.procedure_stage,
            "condition": self.condition,
            "body_habitus": self.body_habitus,
            "seed": self.seed,
            "respiration": asdict(self.respiration),
            "cardiovascular": asdict(self.cardiovascular),
            "oxygen": asdict(self.oxygen),
            "temperature": asdict(self.temperature),
            "coagulation": asdict(self.coagulation),
            "fluid_balance": asdict(self.fluid_balance),
            "renal": asdict(self.renal),
            "biliary": asdict(self.biliary),
            "vital_signs": asdict(self.vital_signs),
            "perfusion": {
                key: asdict(value) for key, value in self.perfusion.regions.items()
            },
            "bleeding": {
                key: asdict(value) for key, value in self.bleeding.sources.items()
            },
            "tissue_state": {
                "access_state": self.tissue_state.access_state,
                "wound_open_fraction": self.tissue_state.wound_open_fraction,
                "organs": {
                    key: {
                        **asdict(value),
                        "active_adhesions": sorted(value.active_adhesions),
                    }
                    for key, value in self.tissue_state.organs.items()
                },
            },
            "organ_motion": {
                "displacements_m": self.organ_motion.displacements_m,
                "pulsation_scale": self.organ_motion.pulsation_scale,
            },
            "damage": {key: asdict(value) for key, value in self.damage.events.items()},
            "contact_effects": self.contact_effects.snapshot(),
            "incision": self.incision.snapshot(),
            "interventions": [asdict(value) for value in self.interventions.history],
            "events": self.event_bus.snapshot(),
            "released_adhesions": sorted(self.released_adhesions),
            "anastomoses": self.anastomoses,
            "dressing": self.dressing_state,
            "intended_use": "simulation_training",
        }

    def observation(self) -> dict[str, Any]:
        return self.snapshot()

    def save_snapshot(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


DynamicPhysiologicalSurgicalPatient = DynamicSurgicalPatient


class ProcedureOrchestrator:
    """Small scenario runner that translates procedure phases into patient events."""

    def __init__(
        self,
        patient: DynamicSurgicalPatient,
        scenarios: Mapping[str, Any] | None = None,
    ):
        self.patient = patient
        payload = dict(scenarios or load_procedure_scenarios())
        self.scenarios = {s["id"]: s for s in payload["scenarios"]}
        self.active_scenario: str | None = None
        self.completed_steps: list[str] = []

    def begin(self, scenario_id: str) -> dict[str, Any]:
        if scenario_id not in self.scenarios:
            raise KeyError(scenario_id)
        self.active_scenario = scenario_id
        self.completed_steps = []
        self.patient.set_procedure_stage("access_open")
        return dict(self.scenarios[scenario_id])

    def mark_step(self, step_id: str, event: Mapping[str, Any] | None = None) -> None:
        scenario = self.scenarios.get(self.active_scenario or "")
        if scenario is None:
            raise RuntimeError("begin a procedure scenario before marking steps")
        planned_steps = tuple(str(step) for step in scenario.get("steps", ()))
        if step_id not in planned_steps:
            raise ValueError(
                f"step {step_id!r} is not part of scenario {self.active_scenario!r}"
            )
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)
        if event is not None:
            self.patient.interventions.apply(event)

    def status(self) -> dict[str, Any]:
        scenario = self.scenarios.get(self.active_scenario or "", {})
        planned = list(scenario.get("steps", []))
        return {
            "scenario": self.active_scenario,
            "completed_steps": list(self.completed_steps),
            "remaining_steps": [s for s in planned if s not in self.completed_steps],
            "patient_status": self.patient.vital_signs.clinical_status,
        }


# ---------------------------------------------------------------------------
# Isaac / OpenUSD integration
# ---------------------------------------------------------------------------


def make_rigid_proxy_cfg(
    prim_path="/World/PatientProxy",
    *,
    position=(0, 0, 0),
    orientation_wxyz=(1, 0, 0, 0),
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(PATIENT_RIGID_PROXY_USD)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=orientation_wxyz),
    )


def spawn_patient(
    prim_path="/World/Patient",
    *,
    access_state="intact",
    translation=(0, 0, 0),
    orientation_wxyz=(1, 0, 0, 0),
):
    if access_state not in VALID_ACCESS_STATES:
        raise ValueError(
            f"unsupported access_state {access_state!r}; "
            f"expected one of {sorted(VALID_ACCESS_STATES)}"
        )
    import isaaclab.sim as sim_utils

    cfg = sim_utils.UsdFileCfg(
        usd_path=str(PATIENT_USD),
        variants={"access_state": access_state},
    )
    return cfg.func(
        prim_path, cfg, translation=translation, orientation=orientation_wxyz
    )


def spawn_operating_scene(
    prim_path="/World/DrAnmarProcedureScene",
    *,
    translation=(0, 0, 0),
    orientation_wxyz=(1, 0, 0, 0),
):
    import isaaclab.sim as sim_utils

    cfg = sim_utils.UsdFileCfg(usd_path=str(OPERATING_SCENE_USD))
    return cfg.func(
        prim_path, cfg, translation=translation, orientation=orientation_wxyz
    )


def set_access_state(patient_path: str, state: str, *, stage=None) -> None:
    if state not in VALID_ACCESS_STATES:
        raise ValueError(
            f"unsupported access_state {state!r}; "
            f"expected one of {sorted(VALID_ACCESS_STATES)}"
        )
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    normalized_path = patient_path.rstrip("/")
    root = stage.GetPrimAtPath(normalized_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Patient prim does not exist: {normalized_path}")
    if root and root.IsValid():
        variants = root.GetVariantSets()
        if variants.HasVariantSet("access_state"):
            variant = variants.GetVariantSet("access_state")
            if state in variant.GetVariantNames():
                if not variant.SetVariantSelection(state):
                    raise RuntimeError(
                        f"Unable to select access_state {state!r} on {normalized_path}"
                    )
                return
    selected_components: list[str] = []
    for name in ("skin", "subcutaneous_fat", "fascia", "abdominal_wall", "peritoneum"):
        prim = stage.GetPrimAtPath(f"{normalized_path}/Anatomy/{name}")
        if not prim or not prim.IsValid():
            continue
        variant = prim.GetVariantSets().GetVariantSet("access_state")
        if state in variant.GetVariantNames():
            if not variant.SetVariantSelection(state):
                raise RuntimeError(
                    f"Unable to select access_state {state!r} on component {name!r}"
                )
            selected_components.append(name)
    if not selected_components:
        raise RuntimeError(
            f"No access_state variant was found below patient prim {normalized_path}"
        )


def _set_usd_attribute_if_valid(prim: Any, name: str, value: Any) -> bool:
    """Set a schema attribute only when it exists in the active Isaac generation."""
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.IsValid():
        return False
    attribute.Set(value)
    return True


def _apply_registered_api(
    prim: Any,
    schema_identifier: str,
    *,
    required: bool = True,
) -> bool:
    """Apply a codeless API only when the active USD runtime registers it."""
    from pxr import Usd

    definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema_identifier)
    if not definition:
        if required:
            raise RuntimeError(
                f"Required USD API schema is unavailable: {schema_identifier}"
            )
        return False
    prim.ApplyAPI(schema_identifier)
    return True


def _selected_visual_mesh_path(stage: Any, component_path: str) -> str:
    """Return the visual mesh selected by the component's access-state variant."""
    base = component_path.rstrip("/") + "/Geometry"
    for name in ("OpenVisual", "Visual"):
        path = f"{base}/{name}"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        visibility = prim.GetAttribute("visibility")
        if (
            not visibility
            or not visibility.IsValid()
            or visibility.Get() != "invisible"
        ):
            return path
    return f"{base}/Visual"


def _surface_thickness_seed(component_cfg: Mapping[str, Any]) -> float:
    explicit = component_cfg.get("surface_thickness_m_seed")
    if explicit is not None:
        return max(1.0e-5, float(explicit))
    component_id = str(component_cfg.get("id", ""))
    if component_id == "peritoneum":
        return 0.0016
    if component_id == "mesentery":
        return 0.0020
    if component_id in {"small_bowel", "colon"}:
        return 0.0030
    if component_id == "diaphragm":
        return 0.0040
    return 0.0020


def _create_deformable_material(
    stage: Any,
    path: str,
    cfg: Mapping[str, Any],
    *,
    surface: bool,
):
    """Create an Omni Physics material without assuming one Isaac schema generation."""
    from pxr import UsdShade

    material = UsdShade.Material.Define(stage, path)
    prim = material.GetPrim()
    _apply_registered_api(prim, "OmniPhysicsBaseMaterialAPI")
    _set_usd_attribute_if_valid(
        prim, "omniphysics:density", float(cfg.get("density_kg_m3", 1050.0))
    )
    _set_usd_attribute_if_valid(
        prim, "omniphysics:dynamicFriction", float(cfg.get("dynamic_friction", 0.38))
    )

    _apply_registered_api(prim, "OmniPhysicsDeformableMaterialAPI")
    _set_usd_attribute_if_valid(
        prim,
        "omniphysics:youngsModulus",
        float(cfg.get("youngs_modulus_pa_seed", 100_000.0)),
    )
    _set_usd_attribute_if_valid(
        prim, "omniphysics:poissonsRatio", float(cfg.get("poissons_ratio_seed", 0.46))
    )

    if surface:
        _apply_registered_api(prim, "OmniPhysicsSurfaceDeformableMaterialAPI")
        _set_usd_attribute_if_valid(
            prim, "omniphysics:surfaceThickness", _surface_thickness_seed(cfg)
        )
        # Zero delegates shell bending to the runtime's thickness-aware derivation.
        _set_usd_attribute_if_valid(prim, "omniphysics:surfaceBendStiffness", 0.0)
        _apply_registered_api(prim, "PhysxSurfaceDeformableMaterialAPI")
        damping = float(cfg.get("damping_seed", 0.16))
        _set_usd_attribute_if_valid(
            prim, "physxDeformableMaterial:elasticityDamping", damping
        )
        _set_usd_attribute_if_valid(
            prim, "physxDeformableMaterial:bendDamping", damping
        )
    else:
        # Isaac 5.1 registers PhysxDeformableBodyMaterialAPI but not the
        # later PhysxBaseDeformableMaterialAPI. Applying an unregistered
        # codeless schema is a runtime error.
        _apply_registered_api(prim, "PhysxDeformableBodyMaterialAPI")
        _apply_registered_api(
            prim,
            "PhysxBaseDeformableMaterialAPI",
            required=False,
        )
        _set_usd_attribute_if_valid(
            prim,
            "physxDeformableMaterial:elasticityDamping",
            float(cfg.get("damping_seed", 0.16)),
        )
    return material


def _configure_deformable_body(
    body_prim: Any,
    cfg: Mapping[str, Any],
    *,
    surface: bool,
    legacy_volume: bool = False,
) -> None:
    """Apply explicit mass and conservative body settings to the resolved body prim."""
    _apply_registered_api(body_prim, "OmniPhysicsDeformableBodyAPI")
    _set_usd_attribute_if_valid(
        body_prim, "omniphysics:mass", float(cfg.get("mass_kg", 0.1))
    )
    if surface:
        _apply_registered_api(body_prim, "PhysxSurfaceDeformableBodyAPI")
    else:
        _apply_registered_api(body_prim, "PhysxBaseDeformableBodyAPI")
        # PhysxDeformableBodyAPI is the legacy single-UsdGeomMesh schema. It
        # must not be applied to modern Xform hierarchies or UsdGeom.TetMesh
        # bodies in Isaac 5.1.
        if legacy_volume:
            _apply_registered_api(body_prim, "PhysxDeformableBodyAPI")
    # Patient components are authored in situ. Until a component has explicit
    # anatomical attachments or kinematic support nodes, allowing free fall
    # makes it leave the body cavity and can drive invalid high-energy contact.
    _set_usd_attribute_if_valid(
        body_prim,
        "physxDeformableBody:disableGravity",
        bool(cfg.get("disable_gravity", True)),
    )
    _set_usd_attribute_if_valid(
        body_prim,
        "physxDeformableBody:selfCollision",
        bool(cfg.get("self_collision", False)),
    )
    _set_usd_attribute_if_valid(
        body_prim,
        "physxDeformableBody:solverPositionIterationCount",
        int(cfg.get("solver_position_iterations", 24)),
    )
    _set_usd_attribute_if_valid(
        body_prim,
        "physxDeformableBody:vertexVelocityDamping",
        float(cfg.get("damping_seed", 0.16)),
    )


def _set_explicit_volume_deformable_hierarchy(
    stage: Any,
    *,
    root_path: str,
    tet_path: str,
    visual_path: str,
) -> bool:
    """Bind an authored TetMesh and render mesh as one volume body.

    Isaac Sim 5.1 updates visible geometry only when the deformable body owns a
    hierarchy containing a simulation TetMesh plus bind-pose geometry. Applying
    the body API directly to the TetMesh can step tetrahedra, but it leaves the
    separately authored organ render mesh static and is therefore not a valid
    patient representation.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    root_prim = stage.GetPrimAtPath(root_path)
    tet_prim = stage.GetPrimAtPath(tet_path)
    visual_prim = stage.GetPrimAtPath(visual_path)
    if (
        not root_prim
        or not root_prim.IsValid()
        or root_prim.IsA(UsdGeom.Gprim)
        or not tet_prim
        or not tet_prim.IsValid()
        or not tet_prim.IsA(UsdGeom.TetMesh)
        or not visual_prim
        or not visual_prim.IsValid()
        or not visual_prim.IsA(UsdGeom.PointBased)
    ):
        return False
    if tet_prim.GetPath().GetParentPath() != root_prim.GetPath():
        return False

    if not root_prim.ApplyAPI("OmniPhysicsDeformableBodyAPI"):
        return False
    if not tet_prim.ApplyAPI("OmniPhysicsVolumeDeformableSimAPI"):
        return False

    tetmesh = UsdGeom.TetMesh(tet_prim)
    tet_prim.GetAttribute("omniphysics:restShapePoints").Set(
        tetmesh.GetPointsAttr().Get()
    )
    tet_prim.GetAttribute("omniphysics:restTetVtxIndices").Set(
        tetmesh.GetTetVertexIndicesAttr().Get()
    )
    if not UsdPhysics.CollisionAPI.Apply(tet_prim):
        return False
    tetmesh.GetSurfaceFaceVertexIndicesAttr().Set(
        UsdGeom.TetMesh.ComputeSurfaceFaces(
            tetmesh,
            Usd.TimeCode.Default(),
        )
    )

    if not visual_prim.ApplyAPI("OmniPhysicsDeformablePoseAPI", "default"):
        return False
    visual_prim.CreateAttribute(
        "deformablePose:default:omniphysics:purposes",
        Sdf.ValueTypeNames.TokenArray,
    ).Set(["bindPose"])
    visual_prim.CreateAttribute(
        "deformablePose:default:omniphysics:points",
        Sdf.ValueTypeNames.Point3fArray,
    ).Set(UsdGeom.PointBased(visual_prim).GetPointsAttr().Get())
    return True


def laparotomy_wound_edge_paths(
    patient_path: str,
) -> dict[str, dict[str, str]]:
    """Return the explicit TetMesh paths for the real patient wound margins."""
    root = (
        f"{patient_path.rstrip('/')}/AccessMechanics/"
        "LaparotomyWound/Layers"
    )
    return {
        layer: {
            side: (
                f"{root}/{layer}/{side.capitalize()}Edge/"
                "Geometry/SimulationTetMesh"
            )
            for side in ("left", "right")
        }
        for layer in LAPAROTOMY_LAYERS
    }


def _create_vertex_xform_attachment(
    stage: Any,
    *,
    attachment_path: str,
    source_path: str,
    target_path: str,
    vertex_indices: Sequence[int],
) -> str:
    """Bind selected simulation vertices to an xformable without snapping."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    source_prim = stage.GetPrimAtPath(source_path)
    target_prim = stage.GetPrimAtPath(target_path)
    if (
        not source_prim
        or not source_prim.IsValid()
        or not source_prim.IsA(UsdGeom.PointBased)
    ):
        raise RuntimeError(
            f"Attachment source is not a point-based simulation mesh: "
            f"{source_path}"
        )
    if (
        not target_prim
        or not target_prim.IsValid()
        or not UsdGeom.Xformable(target_prim)
    ):
        raise RuntimeError(
            f"Attachment target is not xformable: {target_path}"
        )
    unique_indices = tuple(dict.fromkeys(int(value) for value in vertex_indices))
    points = list(UsdGeom.PointBased(source_prim).GetPointsAttr().Get() or [])
    if not unique_indices or any(
        value < 0 or value >= len(points) for value in unique_indices
    ):
        raise RuntimeError(
            f"Attachment vertex selection is invalid for {source_path}"
        )

    source_to_world = UsdGeom.Xformable(
        source_prim
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_to_target = (
        UsdGeom.Xformable(target_prim)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .GetInverse()
    )
    local_positions = [
        Gf.Vec3f(
            world_to_target.Transform(
                source_to_world.Transform(Gf.Vec3d(points[index]))
            )
        )
        for index in unique_indices
    ]

    parent_path = str(Sdf.Path(attachment_path).GetParentPath())
    if not stage.GetPrimAtPath(parent_path).IsValid():
        stage.DefinePrim(parent_path, "Scope")
    existing = stage.GetPrimAtPath(attachment_path)
    if existing and existing.IsValid():
        stage.RemovePrim(attachment_path)
    attachment = stage.DefinePrim(
        attachment_path, "OmniPhysicsVtxXformAttachment"
    )
    attachment.CreateRelationship("omniphysics:src0").SetTargets(
        [Sdf.Path(source_path)]
    )
    attachment.CreateRelationship("omniphysics:src1").SetTargets(
        [Sdf.Path(target_path)]
    )
    attachment.CreateAttribute(
        "omniphysics:vtxIndicesSrc0", Sdf.ValueTypeNames.IntArray
    ).Set(Vt.IntArray(unique_indices))
    attachment.CreateAttribute(
        "omniphysics:localPositionsSrc1",
        Sdf.ValueTypeNames.Point3fArray,
    ).Set(Vt.Vec3fArray(local_positions))
    attachment.CreateAttribute(
        "omniphysics:attachmentEnabled", Sdf.ValueTypeNames.Bool
    ).Set(True)
    return attachment_path


def _wound_edge_band_indices(
    stage: Any,
    tet_path: str,
    *,
    inner: bool,
) -> list[int]:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(tet_path)
    points = list(UsdGeom.PointBased(prim).GetPointsAttr().Get() or [])
    if not points:
        raise RuntimeError(f"Wound edge has no TetMesh points: {tet_path}")
    rows: dict[tuple[float, float], list[int]] = {}
    z_layer_size = len(points) // 2
    for index, point in enumerate(points):
        rows.setdefault(
            (
                round(float(point[1]), 7),
                0 if index < z_layer_size else 1,
            ),
            [],
        ).append(index)
    selected = []
    for indices in rows.values():
        selected.append(
            min(
                indices,
                key=(
                    (lambda index: abs(float(points[index][0])))
                    if inner
                    else (lambda index: -abs(float(points[index][0])))
                ),
            )
        )
    if len(selected) < 12:
        raise RuntimeError(
            f"Wound edge band selection is too sparse at {tet_path}: "
            f"{len(selected)}"
        )
    return selected


def apply_laparotomy_wound_deformables(
    patient_path: str,
    *,
    stage=None,
) -> dict[str, Any]:
    """Activate the full-thickness bilateral wound edges in the real patient.

    Each abdominal layer uses an authored TetMesh hierarchy. The lateral band
    is fixed in the patient frame; the medial band remains free for Dr.Anmar's
    exposure tool. Material values are provisional engineering seeds.
    """
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    from pxr import UsdGeom, UsdShade

    normalized = patient_path.rstrip("/")
    patient_prim = stage.GetPrimAtPath(normalized)
    wound_root = stage.GetPrimAtPath(
        f"{normalized}/AccessMechanics/LaparotomyWound"
    )
    if not patient_prim.IsValid() or not wound_root.IsValid():
        raise RuntimeError(
            f"Patient laparotomy wound asset is missing below {normalized}"
        )
    if (
        UsdGeom.Imageable(wound_root).ComputeVisibility()
        == UsdGeom.Tokens.invisible
    ):
        raise RuntimeError(
            "Laparotomy mechanics require access_state='open' before "
            "deformable initialization"
        )

    paths = laparotomy_wound_edge_paths(normalized)
    results: dict[str, Any] = {}
    for layer, cfg_seed in LAPAROTOMY_WOUND_LAYER_CONFIGS.items():
        cfg = {"id": f"laparotomy_{layer}", **cfg_seed}
        material_path = (
            f"/World/Materials/DrAnmarPatient/LaparotomyWound/{layer}"
        )
        material = _create_deformable_material(
            stage, material_path, cfg, surface=False
        )
        results[layer] = {}
        for side, tet_path in paths[layer].items():
            geometry_path = str(
                stage.GetPrimAtPath(tet_path).GetPath().GetParentPath()
            )
            visual_path = f"{geometry_path}/Visual"
            if not _set_explicit_volume_deformable_hierarchy(
                stage,
                root_path=geometry_path,
                tet_path=tet_path,
                visual_path=visual_path,
            ):
                raise RuntimeError(
                    f"Unable to activate laparotomy TetMesh hierarchy: "
                    f"{geometry_path}"
                )
            body_prim = stage.GetPrimAtPath(geometry_path)
            tet_prim = stage.GetPrimAtPath(tet_path)
            _configure_deformable_body(
                body_prim,
                cfg,
                surface=False,
                legacy_volume=False,
            )
            for prim in (body_prim, tet_prim):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material,
                    UsdShade.Tokens.weakerThanDescendants,
                    "physics",
                )
            edge_path = str(
                body_prim.GetPath().GetParentPath()
            )
            outer_indices = _wound_edge_band_indices(
                stage, tet_path, inner=False
            )
            anchor_path = (
                f"{edge_path}/Attachments/OuterPatientAnchor"
            )
            _create_vertex_xform_attachment(
                stage,
                attachment_path=anchor_path,
                source_path=tet_path,
                target_path=normalized,
                vertex_indices=outer_indices,
            )
            results[layer][side] = {
                "route": "current_explicit_tetmesh_volume_hierarchy",
                "body_prim_path": geometry_path,
                "simulation_mesh_path": tet_path,
                "outer_anchor_path": anchor_path,
                "outer_anchor_vertex_count": len(outer_indices),
                "clinical_validation": False,
            }
    return results


def capture_laparotomy_wound_edges(
    patient_path: str,
    tool_path: str,
    *,
    stage=None,
) -> list[str]:
    """Capture both full-thickness wound margins with the real exposure pads.

    Six independently releasable capture cells are authored per pad and per
    layer. Local attachment coordinates preserve the current pose, avoiding a
    discontinuous snap when capture is requested.
    """
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    normalized_patient = patient_path.rstrip("/")
    normalized_tool = tool_path.rstrip("/")
    paths = laparotomy_wound_edge_paths(normalized_patient)
    attachments: list[str] = []
    for layer in LAPAROTOMY_LAYERS:
        for side in ("left", "right"):
            tet_path = paths[layer][side]
            selected = _wound_edge_band_indices(
                stage, tet_path, inner=True
            )
            point_attr = stage.GetPrimAtPath(tet_path).GetAttribute("points")
            points = list(point_attr.Get() or [])
            by_y: dict[float, list[int]] = {}
            for index in selected:
                by_y.setdefault(
                    round(float(points[index][1]), 7), []
                ).append(index)
            y_values = sorted(by_y)
            for cell in range(6):
                begin = cell * len(y_values) // 6
                end = (cell + 1) * len(y_values) // 6
                cell_indices = [
                    index
                    for y_value in y_values[begin:end]
                    for index in by_y[y_value]
                ]
                side_title = side.capitalize()
                target_path = (
                    f"{normalized_tool}/Links/{side_title}Pad/"
                    f"Collisions/TissueCaptureCell_{cell:02d}"
                )
                attachment_path = (
                    f"{normalized_patient}/AccessMechanics/"
                    "LaparotomyWound/RuntimeAttachments/"
                    f"{side_title}/{layer}/Capture_{cell:02d}"
                )
                attachments.append(
                    _create_vertex_xform_attachment(
                        stage,
                        attachment_path=attachment_path,
                        source_path=tet_path,
                        target_path=target_path,
                        vertex_indices=cell_indices,
                    )
                )
    return attachments


def release_laparotomy_wound_edges(
    patient_path: str,
    *,
    stage=None,
) -> int:
    """Disable active wound-edge capture bonds without deleting the anatomy."""
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    root_path = (
        f"{patient_path.rstrip('/')}/AccessMechanics/"
        "LaparotomyWound/RuntimeAttachments"
    )
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    released = 0
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(root.GetPath()):
            continue
        attribute = prim.GetAttribute("omniphysics:attachmentEnabled")
        if attribute and attribute.IsValid() and bool(attribute.Get()):
            attribute.Set(False)
            released += 1
    return released


def apply_component_deformable(
    stage: Any,
    component_path: str,
    component_cfg: Mapping[str, Any],
    *,
    material_path: str | None = None,
) -> dict[str, Any]:
    """Apply the best available PhysX representation to one patient component.

    Current Isaac generations use a true deformable hierarchy for volumes so the
    detailed render mesh follows an authored TetMesh. Surface components simulate
    the selected triangular mesh directly. Runtime volume cooking is deliberately
    not used by the authored patient route because it was unstable in the
    tested Isaac Sim 5.1 environment.
    """
    mechanics = str(component_cfg.get("mechanics", ""))
    component_path = component_path.rstrip("/")
    geometry_root_path = f"{component_path}/Geometry"
    visual_path = _selected_visual_mesh_path(stage, component_path)
    tet_path = f"{geometry_root_path}/SimulationTetMesh"
    surface = "surface" in mechanics
    volume = "volume" in mechanics
    if material_path is None:
        material_path = f"/World/Materials/DrAnmarPatient/{component_cfg['id']}Material"
    result: dict[str, Any] = {
        "component": component_cfg["id"],
        "mechanics": mechanics,
        "material_path": material_path,
        "visual_path": visual_path,
        "route": None,
    }

    if mechanics in {
        "segmented_rod",
        "breakable_attachment_graph",
        "attached_rigid_or_deformable",
        "host_controlled_presegmented_laparotomy",
    }:
        result["route"] = "host_controlled_" + mechanics
        return result
    if not (surface or volume):
        result["route"] = "unsupported_mechanics_contract"
        return result

    from pxr import UsdShade

    try:
        from omni.physx.scripts import deformableUtils

        material = _create_deformable_material(
            stage, material_path, component_cfg, surface=surface
        )
        body_prim = None
        collision_prim = None
        ok: Any = False

        explicit_tet = volume and stage.GetPrimAtPath(tet_path).IsValid()
        if explicit_tet:
            ok = _set_explicit_volume_deformable_hierarchy(
                stage,
                root_path=geometry_root_path,
                tet_path=tet_path,
                visual_path=visual_path,
            )
            body_prim = stage.GetPrimAtPath(geometry_root_path)
            collision_prim = stage.GetPrimAtPath(tet_path)
            result["simulation_mesh_path"] = tet_path
            result["route"] = "current_explicit_tetmesh_volume_hierarchy"
            result["cooking_triggered"] = False
        elif volume:
            raise RuntimeError(
                "volume component has no authored SimulationTetMesh; "
                "runtime auto-cooking is outside the authored patient route"
            )
        elif surface and hasattr(
            deformableUtils, "set_physics_surface_deformable_body"
        ):
            ok = deformableUtils.set_physics_surface_deformable_body(
                stage, stage.GetPrimAtPath(visual_path).GetPath()
            )
            body_prim = stage.GetPrimAtPath(visual_path)
            collision_prim = body_prim
            result["simulation_mesh_path"] = visual_path
            result["route"] = "current_surface_deformable"
        elif surface and hasattr(deformableUtils, "add_physx_deformable_surface"):
            ok = deformableUtils.add_physx_deformable_surface(
                stage,
                stage.GetPrimAtPath(visual_path).GetPath(),
                solver_position_iteration_count=24,
                vertex_velocity_damping=float(component_cfg.get("damping_seed", 0.12)),
                self_collision=False,
            )
            if hasattr(deformableUtils, "add_deformable_surface_material"):
                deformableUtils.add_deformable_surface_material(
                    stage,
                    material_path,
                    density=float(component_cfg.get("density_kg_m3", 1050.0)),
                    dynamic_friction=float(component_cfg.get("dynamic_friction", 0.38)),
                    poissons_ratio=float(
                        component_cfg.get("poissons_ratio_seed", 0.46)
                    ),
                    thickness=_surface_thickness_seed(component_cfg),
                    youngs_modulus=float(
                        component_cfg.get("youngs_modulus_pa_seed", 100_000.0)
                    ),
                )
            body_prim = stage.GetPrimAtPath(visual_path)
            collision_prim = body_prim
            result["simulation_mesh_path"] = visual_path
            result["route"] = "legacy_surface_deformable"
        else:
            raise RuntimeError("No supported PhysX deformable route is available")

        if ok is False or body_prim is None or not body_prim.IsValid():
            raise RuntimeError(f"deformable creation failed for {component_path}")

        _configure_deformable_body(
            body_prim,
            component_cfg,
            surface=surface,
            legacy_volume=False,
        )
        UsdShade.MaterialBindingAPI.Apply(body_prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        if (
            collision_prim is not None
            and collision_prim.IsValid()
            and collision_prim != body_prim
        ):
            UsdShade.MaterialBindingAPI.Apply(collision_prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
        result["body_prim_path"] = str(body_prim.GetPath())
        result["collision_prim_path"] = (
            str(collision_prim.GetPath())
            if collision_prim is not None and collision_prim.IsValid()
            else None
        )
        return result
    except (
        ImportError,
        ModuleNotFoundError,
        AttributeError,
        RuntimeError,
        ValueError,
    ) as exc:
        result["route"] = "not_applied"
        result["error"] = str(exc)
        return result


def apply_patient_deformables(
    patient_path: str,
    *,
    include: Sequence[str] | None = None,
    include_laparotomy_wound: bool = True,
    stage=None,
) -> dict[str, Any]:
    """Apply mechanics routes after the patient and access variant are spawned.

    An explicit ``include`` is treated as an operational request and therefore
    fails closed for empty, duplicate, unknown, or host-controlled component
    identifiers. Omitting ``include`` retains the inspection behavior that
    reports a route for every manifest component.
    """
    manifest = load_anatomy_manifest()
    components = {
        str(component["id"]): component for component in manifest["components"]
    }
    wanted: set[str] | None = None
    if include is not None:
        if isinstance(include, (str, bytes)):
            raise TypeError("include must be a sequence of component IDs, not a string")
        requested = tuple(str(component_id) for component_id in include)
        if not requested:
            raise ValueError("include must request at least one deformable component")
        if len(set(requested)) != len(requested):
            raise ValueError(f"include contains duplicate component IDs: {requested!r}")
        unknown = sorted(set(requested).difference(components))
        if unknown:
            raise ValueError(f"unknown dynamic-patient component IDs: {unknown}")
        non_deformable = sorted(
            component_id
            for component_id in requested
            if not any(
                token in str(components[component_id].get("mechanics", ""))
                for token in ("surface", "volume")
            )
        )
        if non_deformable:
            raise ValueError(
                "requested components do not use a native deformable route: "
                f"{non_deformable}"
            )
        wanted = set(requested)

    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    results: dict[str, Any] = {}
    for cfg in manifest["components"]:
        if wanted is not None and cfg["id"] not in wanted:
            continue
        path = f"{patient_path.rstrip('/')}/Anatomy/{cfg['id']}"
        results[cfg["id"]] = apply_component_deformable(stage, path, cfg)
    if wanted is None and include_laparotomy_wound:
        from pxr import UsdGeom

        wound_prim = stage.GetPrimAtPath(
            f"{patient_path.rstrip('/')}/AccessMechanics/LaparotomyWound"
        )
        if (
            wound_prim
            and wound_prim.IsValid()
            and UsdGeom.Imageable(wound_prim).ComputeVisibility()
            != UsdGeom.Tokens.invisible
        ):
            results["laparotomy_wound"] = (
                apply_laparotomy_wound_deformables(
                    patient_path, stage=stage
                )
            )
    return results


def configure_patient_internal_collision_filter(
    patient_path: str,
    *,
    stage=None,
    collision_group_path: str | None = None,
) -> dict[str, str]:
    """Disable patient-on-patient contacts while retaining tool contacts.

    The authored organs overlap in their undeformed anatomical pose. Treating
    every pair as a collision pair creates a pathological broad-phase contact
    set before calibrated inter-organ contact layers exist. A self-filtered USD
    collision group removes only contacts between members below the patient
    root; instruments and other scene geometry remain outside the group.
    """
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()

    from pxr import Sdf, Usd, UsdPhysics

    normalized_patient_path = patient_path.rstrip("/")
    if not normalized_patient_path.startswith("/"):
        raise ValueError("patient_path must be an absolute USD prim path")
    if not stage.GetPrimAtPath(normalized_patient_path).IsValid():
        raise RuntimeError(f"Patient prim does not exist: {normalized_patient_path}")

    environment_path = normalized_patient_path.rsplit("/", 1)[0]
    group_path = (
        collision_group_path
        or f"{environment_path}/DynamicAbdominalPatientInternalCollisionGroup"
    )
    collision_group = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path(group_path))
    colliders = Usd.CollectionAPI.Apply(collision_group.GetPrim(), "colliders")
    colliders.CreateExpansionRuleAttr().Set(Usd.Tokens.expandPrims)
    colliders.CreateIncludesRel().AddTarget(Sdf.Path(normalized_patient_path))
    collision_group.CreateFilteredGroupsRel().AddTarget(Sdf.Path(group_path))
    return {
        "patient_path": normalized_patient_path,
        "collision_group_path": group_path,
        "policy": "filter_internal_patient_pairs_preserve_external_tool_contacts",
    }


def create_auto_deformable_attachment(
    stage,
    attachment_path: str,
    deformable_path: str,
    rigid_or_deformable_path: str,
) -> str:
    import omni.kit.commands

    ok = omni.kit.commands.execute(
        "CreateAutoDeformableAttachment",
        target_attachment_path=attachment_path,
        attachable0_path=deformable_path,
        attachable1_path=rigid_or_deformable_path,
    )
    if ok is False:
        raise RuntimeError(f"failed to create attachment {attachment_path}")
    return attachment_path


def apply_proxy_organ_motion(
    stage, patient_path: str, patient: DynamicSurgicalPatient
) -> None:
    """Apply respiration and pulse transforms to non-deformable inspection lanes."""
    from pxr import Gf

    manifest = patient.anatomy
    for cfg in manifest["components"]:
        organ = cfg["id"]
        path = f"{patient_path.rstrip('/')}/Anatomy/{organ}"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        displacement = patient.organ_motion.displacement(organ)
        base = cfg["translation_m"]
        attr = prim.GetAttribute("xformOp:translate")
        if attr:
            attr.Set(
                Gf.Vec3d(*(float(base[i]) + float(displacement[i]) for i in range(3)))
            )
        scale_attr = prim.GetAttribute("xformOp:scale")
        if scale_attr:
            scale = patient.organ_motion.pulsation_scale.get(organ, 1.0)
            scale_attr.Set(Gf.Vec3f(scale, scale, 1.0))


def build_volume_motion_targets(
    deformable_object: Any,
    organ_id: str,
    patient: DynamicSurgicalPatient,
    *,
    env_ids=None,
):
    """Build absolute partial-kinematic targets from the default world-space rest state."""
    target = tensor_value(deformable_object.data.nodal_kinematic_target).clone()
    default_state = tensor_value(deformable_object.data.default_nodal_state_w)
    displacement = patient.organ_motion.displacement(organ_id)
    target[..., :3] = default_state[..., :3]
    target[..., 0] += float(displacement[0])
    target[..., 1] += float(displacement[1])
    target[..., 2] += float(displacement[2])
    # The host chooses constrained anchor nodes. All nodes remain free by default.
    target[..., 3] = 1.0
    return target


def constrain_motion_nodes(target: Any, node_indices: Sequence[int]) -> Any:
    for index in node_indices:
        target[..., int(index), 3] = 0.0
    return target


def ensure_patient_particle_system(
    stage,
    *,
    fluid: str = "blood",
    root_path: str | None = None,
    simulation_owner: str = "/World/physicsScene",
    particle_contact_offset_m: float = 0.0012,
):
    """Create one fluid-specific PBD system with independent material properties."""
    if fluid not in VALID_FLUIDS:
        raise ValueError(fluid)
    from omni.physx.scripts import particleUtils, physicsUtils
    from pxr import PhysxSchema, Sdf

    preset = FLUID_PHYSICS_PRESETS[fluid]
    if root_path is None:
        root_path = f"/World/DrAnmarPatientFluids/{fluid}"
    root = Sdf.Path(root_path)
    material_path = root.AppendChild("PBDMaterial")
    system_path = root.AppendChild("ParticleSystem")
    if not stage.GetPrimAtPath(material_path).IsValid():
        particleUtils.add_pbd_particle_material(
            stage,
            material_path,
            density=float(preset["density_kg_m3"]),
            viscosity=float(preset["viscosity"]),
            cohesion=float(preset["cohesion"]),
        )
    if not stage.GetPrimAtPath(system_path).IsValid():
        particleUtils.add_physx_particle_system(
            stage=stage,
            particle_system_path=system_path,
            simulation_owner=Sdf.Path(simulation_owner),
            particle_contact_offset=float(particle_contact_offset_m),
        )
    system = PhysxSchema.PhysxParticleSystem.Get(stage, system_path)
    physicsUtils.add_physics_material_to_prim(stage, system.GetPrim(), material_path)
    return {
        "fluid": fluid,
        "root_path": str(root),
        "material_path": str(material_path),
        "particle_system_path": str(system_path),
        "particle_group": int(preset["particle_group"]),
    }


def emit_fluid_particles(
    stage,
    *,
    path: str,
    particle_system_path: str,
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
    fluid: str = "blood",
    particle_group: int | None = None,
    radius_m: float = 0.0012,
):
    if fluid not in VALID_FLUIDS:
        raise ValueError(fluid)
    from omni.physx.scripts import particleUtils
    from pxr import Gf, Sdf

    if len(positions) != len(velocities):
        raise ValueError("positions and velocities must have equal length")
    if particle_group is None:
        particle_group = int(FLUID_PHYSICS_PRESETS[fluid]["particle_group"])
    pos = [Gf.Vec3f(*map(float, p)) for p in positions]
    vel = [Gf.Vec3f(*map(float, v)) for v in velocities]
    width = [2.0 * float(radius_m)] * len(pos)
    return particleUtils.add_physx_particleset_points(
        stage,
        Sdf.Path(path),
        pos,
        vel,
        width,
        Sdf.Path(particle_system_path),
        True,
        True,
        int(particle_group),
        1.0,
        0.02,
    )


def patient_adapter_contract() -> dict[str, Any]:
    return {
        "patient_asset": str(PATIENT_USD),
        "shared_runtime_attributes": [
            "respiration",
            "perfusion",
            "bleeding",
            "vital_signs",
            "tissue_state",
            "organ_motion",
            "damage",
            "interventions",
        ],
        "robot_compatibility": load_robot_compatibility(),
        "procedure_scenarios": load_procedure_scenarios(),
        "intended_use": "simulation_training",
    }
