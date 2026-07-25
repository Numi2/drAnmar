# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Scene-evidence-owned circulation and ventilation effects for the rescue OR."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Final, Mapping


CHANNELS: Final = frozenset({"crystalloid", "blood_product", "vasopressor"})


def _finite(value: float, name: str) -> float:
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ValueError(f"{name} must be finite")
    return rendered


def _nonnegative(value: float, name: str) -> float:
    rendered = _finite(value, name)
    if rendered < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return rendered


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class PumpEvidenceFrame:
    """Raw post-physics pump and line measurements.

    No delivered-volume or patient-effect field is accepted. Delivery is the
    mutually supported portion of plunger displacement, outlet flow, and
    reservoir mass loss while a vascular attachment is present.
    """

    physics_step: int
    simulation_time_s: float
    dt_s: float
    channel_id: str
    access_attachment_count: int
    plunger_position_m: float
    downstream_flow_ml_s: float
    reservoir_mass_g: float
    line_pressure_kpa: float
    interval_extravasated_ml: float = 0.0

    def __post_init__(self) -> None:
        if self.physics_step < 0:
            raise ValueError("physics_step must be nonnegative")
        if self.channel_id not in CHANNELS:
            raise ValueError(f"unsupported resuscitation channel {self.channel_id!r}")
        if self.access_attachment_count < 0:
            raise ValueError("access_attachment_count must be nonnegative")
        for name in (
            "simulation_time_s",
            "dt_s",
            "plunger_position_m",
            "downstream_flow_ml_s",
            "reservoir_mass_g",
            "interval_extravasated_ml",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self,
            "line_pressure_kpa",
            _finite(self.line_pressure_kpa, "line_pressure_kpa"),
        )
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be greater than zero")


@dataclass(frozen=True)
class VentilationEvidenceFrame:
    """Raw airway-circuit and chest-motion measurements."""

    physics_step: int
    simulation_time_s: float
    dt_s: float
    airway_attachment_count: int
    valve_angle_deg: float
    inspiratory_flow_l_min: float
    leaked_flow_l_min: float
    airway_pressure_cmh2o: float
    measured_fio2_fraction: float
    chest_excursion_m: float

    def __post_init__(self) -> None:
        if self.physics_step < 0:
            raise ValueError("physics_step must be nonnegative")
        if self.airway_attachment_count < 0:
            raise ValueError("airway_attachment_count must be nonnegative")
        for name in (
            "simulation_time_s",
            "dt_s",
            "valve_angle_deg",
            "inspiratory_flow_l_min",
            "leaked_flow_l_min",
            "airway_pressure_cmh2o",
            "chest_excursion_m",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        fio2 = _finite(self.measured_fio2_fraction, "measured_fio2_fraction")
        if not 0.21 <= fio2 <= 1.0:
            raise ValueError("measured_fio2_fraction must be within [0.21, 1]")
        object.__setattr__(self, "measured_fio2_fraction", fio2)
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be greater than zero")


@dataclass(frozen=True)
class ResuscitationCalibration:
    plunger_stroke_m: float = 0.18
    crystalloid_volume_per_stroke_ml: float = 500.0
    blood_volume_per_stroke_ml: float = 300.0
    vasopressor_volume_per_stroke_ml: float = 10.0
    crystalloid_density_g_ml: float = 1.00
    blood_density_g_ml: float = 1.06
    vasopressor_density_g_ml: float = 1.00
    minimum_line_pressure_kpa: float = 1.0
    maximum_line_pressure_kpa: float = 40.0
    hard_line_pressure_kpa: float = 65.0
    initial_crystalloid_inventory_ml: float = 3000.0
    initial_blood_inventory_ml: float = 1200.0
    initial_vasopressor_inventory_ml: float = 40.0
    crystalloid_intravascular_retention_fraction: float = 0.24
    ventilation_valve_full_open_deg: float = 90.0
    ventilation_target_chest_excursion_m: float = 0.02
    ventilation_minimum_airway_pressure_cmh2o: float = 4.0
    ventilation_maximum_airway_pressure_cmh2o: float = 30.0
    ventilation_hard_airway_pressure_cmh2o: float = 45.0
    parameter_status: str = "provisional_engineering_seeds"

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name == "parameter_status":
                continue
            _nonnegative(value, name)
        if self.plunger_stroke_m <= 0.0:
            raise ValueError("plunger_stroke_m must be positive")
        for name in (
            "crystalloid_volume_per_stroke_ml",
            "blood_volume_per_stroke_ml",
            "vasopressor_volume_per_stroke_ml",
            "crystalloid_density_g_ml",
            "blood_density_g_ml",
            "vasopressor_density_g_ml",
            "ventilation_valve_full_open_deg",
            "ventilation_target_chest_excursion_m",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_line_pressure_kpa > self.maximum_line_pressure_kpa:
            raise ValueError(
                "minimum_line_pressure_kpa must not exceed maximum_line_pressure_kpa"
            )
        if self.hard_line_pressure_kpa <= self.maximum_line_pressure_kpa:
            raise ValueError(
                "hard_line_pressure_kpa must exceed maximum_line_pressure_kpa"
            )
        if (
            self.ventilation_minimum_airway_pressure_cmh2o
            > self.ventilation_maximum_airway_pressure_cmh2o
        ):
            raise ValueError(
                "ventilation minimum pressure must not exceed maximum pressure"
            )
        if (
            self.ventilation_hard_airway_pressure_cmh2o
            <= self.ventilation_maximum_airway_pressure_cmh2o
        ):
            raise ValueError(
                "ventilation hard pressure must exceed maximum pressure"
            )
        if not 0.0 <= self.crystalloid_intravascular_retention_fraction <= 1.0:
            raise ValueError(
                "crystalloid_intravascular_retention_fraction must be a fraction"
            )


@dataclass
class PumpChannelState:
    channel_id: str
    remaining_inventory_ml: float
    delivered_to_patient_ml: float = 0.0
    withdrawn_from_reservoir_ml: float = 0.0
    wasted_or_extravasated_ml: float = 0.0
    pressure_damage_fraction: float = 0.0
    last_plunger_position_m: float | None = None
    last_reservoir_mass_g: float | None = None
    last_physics_step: int = -1
    last_simulation_time_s: float = -1.0


@dataclass
class VentilationState:
    airway_connected: bool = False
    effective_minute_ventilation_l_min: float = 0.0
    delivered_fio2_fraction: float = 0.21
    airway_pressure_cmh2o: float = 0.0
    chest_excursion_m: float = 0.0
    circuit_leak_fraction: float = 0.0
    pressure_damage_fraction: float = 0.0
    cumulative_effective_ventilation_l: float = 0.0
    last_physics_step: int = -1
    last_simulation_time_s: float = -1.0


@dataclass(frozen=True)
class ResuscitationSnapshot:
    physics_step: int
    simulation_time_s: float
    channels: Mapping[str, Mapping[str, float | int]]
    ventilation: Mapping[str, float | int | bool]
    effective_circulating_volume_gain_ml: float
    evidence_frames: int
    rejected_frames: int


class _PumpAuthority:
    __slots__ = ()


class PumpEvidenceAdapter:
    __slots__ = ("_effects", "_authority")

    def __init__(
        self,
        effects: "ContactDrivenResuscitationEffects",
        authority: _PumpAuthority,
    ) -> None:
        self._effects = effects
        self._authority = authority

    def publish(self, frame: PumpEvidenceFrame) -> ResuscitationSnapshot:
        return self._effects._ingest(frame, self._authority)

    def publish_ventilation(
        self,
        frame: VentilationEvidenceFrame,
    ) -> ResuscitationSnapshot:
        return self._effects._ingest_ventilation(frame, self._authority)


class ContactDrivenResuscitationEffects:
    """Integrate conserved fluid delivery and measured ventilation support."""

    def __init__(
        self,
        *,
        calibration: ResuscitationCalibration | None = None,
    ) -> None:
        self.calibration = calibration or ResuscitationCalibration()
        self._authority = _PumpAuthority()
        self._physics_step = -1
        self._simulation_time_s = 0.0
        self._evidence_frames = 0
        self._rejected_frames = 0
        self.channels = self._new_channels()
        self.ventilation = VentilationState()

    def _new_channels(self) -> dict[str, PumpChannelState]:
        cfg = self.calibration
        return {
            "crystalloid": PumpChannelState(
                "crystalloid",
                cfg.initial_crystalloid_inventory_ml,
            ),
            "blood_product": PumpChannelState(
                "blood_product",
                cfg.initial_blood_inventory_ml,
            ),
            "vasopressor": PumpChannelState(
                "vasopressor",
                cfg.initial_vasopressor_inventory_ml,
            ),
        }

    def create_scene_adapter(self) -> PumpEvidenceAdapter:
        return PumpEvidenceAdapter(self, self._authority)

    def reset(self) -> ResuscitationSnapshot:
        self._physics_step = -1
        self._simulation_time_s = 0.0
        self._evidence_frames = 0
        self._rejected_frames = 0
        self.channels = self._new_channels()
        self.ventilation = VentilationState()
        return self.snapshot()

    def _channel_volume_per_stroke(self, channel_id: str) -> float:
        return {
            "crystalloid": self.calibration.crystalloid_volume_per_stroke_ml,
            "blood_product": self.calibration.blood_volume_per_stroke_ml,
            "vasopressor": self.calibration.vasopressor_volume_per_stroke_ml,
        }[channel_id]

    def _channel_density(self, channel_id: str) -> float:
        return {
            "crystalloid": self.calibration.crystalloid_density_g_ml,
            "blood_product": self.calibration.blood_density_g_ml,
            "vasopressor": self.calibration.vasopressor_density_g_ml,
        }[channel_id]

    def _ingest(
        self,
        frame: PumpEvidenceFrame,
        authority: _PumpAuthority,
    ) -> ResuscitationSnapshot:
        if authority is not self._authority:
            self._rejected_frames += 1
            raise PermissionError("resuscitation effects accept scene evidence only")
        state = self.channels[frame.channel_id]
        if frame.physics_step <= state.last_physics_step:
            self._rejected_frames += 1
            raise ValueError(
                "pump evidence must use a strictly increasing step per channel"
            )
        if (
            frame.simulation_time_s <= state.last_simulation_time_s
            and state.last_physics_step >= 0
        ):
            self._rejected_frames += 1
            raise ValueError(
                "pump evidence must use increasing simulation time per channel"
            )

        self._physics_step = max(self._physics_step, frame.physics_step)
        self._simulation_time_s = max(
            self._simulation_time_s,
            frame.simulation_time_s,
        )
        self._evidence_frames += 1
        if (
            state.last_plunger_position_m is None
            or state.last_reservoir_mass_g is None
        ):
            state.last_plunger_position_m = frame.plunger_position_m
            state.last_reservoir_mass_g = frame.reservoir_mass_g
            state.last_physics_step = frame.physics_step
            state.last_simulation_time_s = frame.simulation_time_s
            return self.snapshot()

        plunger_delta_m = max(
            0.0,
            frame.plunger_position_m - state.last_plunger_position_m,
        )
        plunger_volume_ml = (
            plunger_delta_m
            / self.calibration.plunger_stroke_m
            * self._channel_volume_per_stroke(frame.channel_id)
        )
        flow_volume_ml = frame.downstream_flow_ml_s * frame.dt_s
        mass_volume_ml = max(
            0.0,
            state.last_reservoir_mass_g - frame.reservoir_mass_g,
        ) / self._channel_density(frame.channel_id)
        reservoir_withdrawal_ml = min(
            mass_volume_ml,
            state.remaining_inventory_ml,
        )
        supported_delivery_ml = min(
            plunger_volume_ml,
            flow_volume_ml,
            reservoir_withdrawal_ml,
        )
        pressure_valid = (
            self.calibration.minimum_line_pressure_kpa
            <= frame.line_pressure_kpa
            <= self.calibration.maximum_line_pressure_kpa
        )
        connected = frame.access_attachment_count > 0
        extravasated_ml = min(
            supported_delivery_ml,
            frame.interval_extravasated_ml,
        )
        delivered_ml = (
            max(0.0, supported_delivery_ml - extravasated_ml)
            if connected and pressure_valid
            else 0.0
        )
        wasted_ml = reservoir_withdrawal_ml - delivered_ml
        state.withdrawn_from_reservoir_ml += reservoir_withdrawal_ml
        state.delivered_to_patient_ml += delivered_ml
        state.wasted_or_extravasated_ml += wasted_ml
        state.remaining_inventory_ml -= reservoir_withdrawal_ml
        if frame.line_pressure_kpa > self.calibration.maximum_line_pressure_kpa:
            overload = _clamp(
                (
                    frame.line_pressure_kpa
                    - self.calibration.maximum_line_pressure_kpa
                )
                / (
                    self.calibration.hard_line_pressure_kpa
                    - self.calibration.maximum_line_pressure_kpa
                )
            )
            state.pressure_damage_fraction = _clamp(
                state.pressure_damage_fraction + overload * frame.dt_s * 0.04
            )
        state.last_plunger_position_m = frame.plunger_position_m
        state.last_reservoir_mass_g = frame.reservoir_mass_g
        state.last_physics_step = frame.physics_step
        state.last_simulation_time_s = frame.simulation_time_s
        return self.snapshot()

    def _ingest_ventilation(
        self,
        frame: VentilationEvidenceFrame,
        authority: _PumpAuthority,
    ) -> ResuscitationSnapshot:
        if authority is not self._authority:
            self._rejected_frames += 1
            raise PermissionError("ventilation effects accept scene evidence only")
        state = self.ventilation
        if frame.physics_step <= state.last_physics_step:
            self._rejected_frames += 1
            raise ValueError(
                "ventilation evidence must use a strictly increasing step"
            )
        if (
            frame.simulation_time_s <= state.last_simulation_time_s
            and state.last_physics_step >= 0
        ):
            self._rejected_frames += 1
            raise ValueError(
                "ventilation evidence must use increasing simulation time"
            )
        cfg = self.calibration
        self._physics_step = max(self._physics_step, frame.physics_step)
        self._simulation_time_s = max(
            self._simulation_time_s,
            frame.simulation_time_s,
        )
        self._evidence_frames += 1
        state.airway_connected = frame.airway_attachment_count > 0
        valve_fraction = _clamp(
            frame.valve_angle_deg / cfg.ventilation_valve_full_open_deg
        )
        net_flow_l_min = max(
            0.0,
            frame.inspiratory_flow_l_min - frame.leaked_flow_l_min,
        )
        state.circuit_leak_fraction = _clamp(
            frame.leaked_flow_l_min
            / max(frame.inspiratory_flow_l_min, 1.0e-9)
        )
        chest_fraction = _clamp(
            frame.chest_excursion_m
            / cfg.ventilation_target_chest_excursion_m
        )
        pressure_valid = (
            cfg.ventilation_minimum_airway_pressure_cmh2o
            <= frame.airway_pressure_cmh2o
            <= cfg.ventilation_maximum_airway_pressure_cmh2o
        )
        delivery_fraction = (
            min(valve_fraction, chest_fraction)
            if state.airway_connected and pressure_valid
            else 0.0
        )
        state.effective_minute_ventilation_l_min = (
            net_flow_l_min * delivery_fraction
        )
        state.delivered_fio2_fraction = (
            0.21
            + (frame.measured_fio2_fraction - 0.21) * delivery_fraction
        )
        state.airway_pressure_cmh2o = frame.airway_pressure_cmh2o
        state.chest_excursion_m = frame.chest_excursion_m
        state.cumulative_effective_ventilation_l += (
            state.effective_minute_ventilation_l_min
            * frame.dt_s
            / 60.0
        )
        if (
            frame.airway_pressure_cmh2o
            > cfg.ventilation_maximum_airway_pressure_cmh2o
        ):
            overload = _clamp(
                (
                    frame.airway_pressure_cmh2o
                    - cfg.ventilation_maximum_airway_pressure_cmh2o
                )
                / (
                    cfg.ventilation_hard_airway_pressure_cmh2o
                    - cfg.ventilation_maximum_airway_pressure_cmh2o
                )
            )
            state.pressure_damage_fraction = _clamp(
                state.pressure_damage_fraction + overload * frame.dt_s * 0.03
            )
        state.last_physics_step = frame.physics_step
        state.last_simulation_time_s = frame.simulation_time_s
        return self.snapshot()

    def snapshot(self) -> ResuscitationSnapshot:
        channels = MappingProxyType(
            {
                channel_id: MappingProxyType(
                    {
                        "remaining_inventory_ml": state.remaining_inventory_ml,
                        "delivered_to_patient_ml": state.delivered_to_patient_ml,
                        "withdrawn_from_reservoir_ml": (
                            state.withdrawn_from_reservoir_ml
                        ),
                        "wasted_or_extravasated_ml": (
                            state.wasted_or_extravasated_ml
                        ),
                        "pressure_damage_fraction": (
                            state.pressure_damage_fraction
                        ),
                        "last_physics_step": state.last_physics_step,
                    }
                )
                for channel_id, state in self.channels.items()
            }
        )
        cfg = self.calibration
        ventilation = MappingProxyType(
            {
                "airway_connected": self.ventilation.airway_connected,
                "effective_minute_ventilation_l_min": (
                    self.ventilation.effective_minute_ventilation_l_min
                ),
                "delivered_fio2_fraction": (
                    self.ventilation.delivered_fio2_fraction
                ),
                "airway_pressure_cmh2o": (
                    self.ventilation.airway_pressure_cmh2o
                ),
                "chest_excursion_m": self.ventilation.chest_excursion_m,
                "circuit_leak_fraction": (
                    self.ventilation.circuit_leak_fraction
                ),
                "pressure_damage_fraction": (
                    self.ventilation.pressure_damage_fraction
                ),
                "cumulative_effective_ventilation_l": (
                    self.ventilation.cumulative_effective_ventilation_l
                ),
                "last_physics_step": self.ventilation.last_physics_step,
            }
        )
        effective_gain = (
            self.channels["blood_product"].delivered_to_patient_ml
            + cfg.crystalloid_intravascular_retention_fraction
            * self.channels["crystalloid"].delivered_to_patient_ml
        )
        return ResuscitationSnapshot(
            physics_step=self._physics_step,
            simulation_time_s=self._simulation_time_s,
            channels=channels,
            ventilation=ventilation,
            effective_circulating_volume_gain_ml=effective_gain,
            evidence_frames=self._evidence_frames,
            rejected_frames=self._rejected_frames,
        )


__all__ = [
    "ContactDrivenResuscitationEffects",
    "PumpChannelState",
    "PumpEvidenceAdapter",
    "PumpEvidenceFrame",
    "ResuscitationCalibration",
    "ResuscitationSnapshot",
    "VentilationEvidenceFrame",
    "VentilationState",
]
