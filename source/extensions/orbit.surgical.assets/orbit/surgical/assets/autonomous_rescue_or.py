# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Executable orchestration for the DrAnmar Autonomous Rescue OR.

The module separates three surfaces:

* policy intent: reversible requests such as compress, clip, patch, or verify;
* scene evidence: monotonic post-physics contact and attachment observations;
* patient outcome: bleeding, perfusion, repair integrity, and rescue success.

Policy actions never write patient outcomes.  The environment-owned scene
adapter is the only ingress to :mod:`deformable_rescue`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import json
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping, Sequence

from .deformable_rescue import (
    ContactDrivenRescueEffects,
    PhysicsEvidenceFrame,
    RescueEffectsSnapshot,
    SceneEvidenceAdapter,
)
from .resuscitation_effects import (
    ContactDrivenResuscitationEffects,
    PumpEvidenceAdapter,
    PumpEvidenceFrame,
    ResuscitationSnapshot,
    VentilationEvidenceFrame,
)


ASSET_DIRECTORY: Final = (
    Path(__file__).resolve().parents[3]
    / "data/Environments/SurgicalAutonomy/AutonomousRescueOR"
)
AUTONOMOUS_RESCUE_OR_USD: Final = (
    ASSET_DIRECTORY / "dranmar_autonomous_rescue_or.usda"
)
DEFORMABLE_RESCUE_SUITE_USD: Final = (
    ASSET_DIRECTORY / "dranmar_deformable_rescue_suite.usda"
)
RESCUE_VESSEL_USD: Final = ASSET_DIRECTORY / "dranmar_rescue_vessel.usda"
RESUSCITATION_MODULE_USD: Final = (
    ASSET_DIRECTORY / "dranmar_resuscitation_module.usda"
)
TOOL_CHANGER_PAYLOAD_USD: Final = (
    ASSET_DIRECTORY / "dranmar_universal_tool_changer_payload.usda"
)

CONTRACT_FILES: Final = MappingProxyType(
    {
        "benchmark_scenarios": "benchmark_scenarios.json",
        "capability_matrix": "capability_matrix.json",
        "complications": "complication_library.json",
        "dependencies": "dependency_manifest.json",
        "episode_schema": "episode_schema.json",
        "interaction_frames": "interaction_frames.json",
        "procedure_graphs": "procedure_graphs.json",
        "rescue_protocols": "rescue_protocols.json",
        "resources": "resource_inventory.json",
        "robot_stations": "robot_station_contract.json",
        "tools": "tool_registry.json",
        "workspace": "workspace_topology.json",
    }
)


def _load_contract(name: str) -> dict[str, object]:
    try:
        filename = CONTRACT_FILES[name]
    except KeyError as error:
        raise KeyError(
            f"unknown rescue contract {name!r}; expected {sorted(CONTRACT_FILES)}"
        ) from error
    path = ASSET_DIRECTORY / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("schema"):
        raise ValueError(f"invalid DrAnmar rescue contract: {path}")
    return payload


class ActionStatus(str, Enum):
    REQUESTED = "requested"
    EXECUTING = "executing"
    EFFECT_OBSERVED = "effect_observed"
    REJECTED = "rejected"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PolicyAction:
    """A policy request containing no patient outcome fields."""

    action_id: str
    station_id: str
    tool_id: str
    target_id: str
    requested_action: str

    def __post_init__(self) -> None:
        for name in (
            "action_id",
            "station_id",
            "tool_id",
            "target_id",
            "requested_action",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True)
class ActionRecord:
    sequence: int
    physics_step: int
    action: PolicyAction
    status: ActionStatus
    reason: str


@dataclass(frozen=True)
class ComplicationObservation:
    complication_id: str
    priority: int
    rescue_protocol: str
    target_id: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class RescuePlan:
    complication_id: str
    protocol_id: str
    actions: tuple[Mapping[str, object], ...]


@dataclass
class ResourceLedger:
    resources: dict[str, float]
    consumed: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_contract(cls) -> "ResourceLedger":
        payload = _load_contract("resources")
        inventory = payload.get("resources", payload.get("inventory", {}))
        resources: dict[str, float] = {}
        if isinstance(inventory, dict):
            resources = {
                str(key): float(value)
                for key, value in inventory.items()
                if isinstance(value, (int, float))
            }
        elif isinstance(inventory, list):
            for item in inventory:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                quantity = item.get("quantity", item.get("count", 0))
                if isinstance(quantity, (int, float)):
                    resources[str(item["id"])] = float(quantity)
        return cls(resources)

    def available(self, resource_id: str) -> float:
        return self.resources.get(resource_id, 0.0)

    def consume(self, resource_id: str, amount: float = 1.0) -> None:
        amount = float(amount)
        if amount <= 0.0:
            raise ValueError("resource consumption must be positive")
        remaining = self.available(resource_id)
        if remaining < amount:
            raise RuntimeError(
                f"insufficient {resource_id}: requested {amount}, available {remaining}"
            )
        self.resources[resource_id] = remaining - amount
        self.consumed[resource_id] = self.consumed.get(resource_id, 0.0) + amount

    def snapshot(self) -> Mapping[str, float]:
        return MappingProxyType(dict(sorted(self.resources.items())))


class ComplicationDetector:
    """Derive rescue triggers from patient effects, not authored labels."""

    def __init__(self) -> None:
        contract = _load_contract("complications")
        self._definitions = {
            str(item["id"]): item
            for item in contract.get("complications", [])
            if isinstance(item, dict) and item.get("id")
        }

    def _observation(
        self,
        complication_id: str,
        target_id: str,
        evidence: Sequence[str],
    ) -> ComplicationObservation:
        definition = self._definitions[complication_id]
        return ComplicationObservation(
            complication_id=complication_id,
            priority=int(definition.get("priority", 0)),
            rescue_protocol=str(definition["rescue_protocol"]),
            target_id=target_id,
            evidence=tuple(evidence),
        )

    def detect(
        self,
        snapshot: RescueEffectsSnapshot,
        *,
        baseline_blood_volume_ml: float,
        effective_resuscitation_gain_ml: float = 0.0,
        spo2_fraction: float | None = None,
    ) -> tuple[ComplicationObservation, ...]:
        vessel = snapshot.vessel
        observations: list[ComplicationObservation] = []
        residual_flow = float(vessel["residual_flow_ml_s"])
        blood_fraction = (
            (
                float(vessel["blood_volume_ml"])
                + max(0.0, float(effective_resuscitation_gain_ml))
            )
            / baseline_blood_volume_ml
        )
        distal_perfusion = float(vessel["distal_perfusion_fraction"])
        if residual_flow >= 2.0 and "catastrophic_hemorrhage" in self._definitions:
            observations.append(
                self._observation(
                    "catastrophic_hemorrhage",
                    "rescue_vessel",
                    (
                        f"scene_residual_flow_ml_s={residual_flow:.6f}",
                        f"physics_step={snapshot.physics_step}",
                    ),
                )
            )
        if blood_fraction < 0.75 and "hypovolemic_shock" in self._definitions:
            observations.append(
                self._observation(
                    "hypovolemic_shock",
                    "patient",
                    (f"conserved_blood_volume_fraction={blood_fraction:.6f}",),
                )
            )
        if (
            spo2_fraction is not None
            and float(spo2_fraction) < 0.90
            and "hypoxemia" in self._definitions
        ):
            observations.append(
                self._observation(
                    "hypoxemia",
                    "patient",
                    (f"patient_spo2_fraction={float(spo2_fraction):.6f}",),
                )
            )
        if distal_perfusion < 0.45 and "regional_ischemia" in self._definitions:
            observations.append(
                self._observation(
                    "regional_ischemia",
                    "distal_vessel_territory",
                    (f"scene_distal_perfusion_fraction={distal_perfusion:.6f}",),
                )
            )
        for target_id, repair in snapshot.repairs.items():
            leak = float(repair["leak_rate_ml_s"])
            if (
                target_id == "bowel_anastomosis"
                and leak > 0.25
                and "anastomotic_leak" in self._definitions
            ):
                observations.append(
                    self._observation(
                        "anastomotic_leak",
                        target_id,
                        (
                            f"scene_leak_rate_ml_s={leak:.6f}",
                            f"physics_step={snapshot.physics_step}",
                        ),
                    )
                )
            if (
                target_id == "abdominal_wall"
                and float(repair["retention_fraction"]) < 0.9
                and float(repair["approximation_fraction"]) > 0.2
                and "abdominal_wall_dehiscence" in self._definitions
            ):
                observations.append(
                    self._observation(
                        "abdominal_wall_dehiscence",
                        target_id,
                        (
                            "scene_retention_fraction="
                            f"{float(repair['retention_fraction']):.6f}",
                        ),
                    )
                )
            if target_id == "occlusive_film" and int(
                repair["last_physics_step"]
            ) >= 0:
                retention = float(repair["retention_fraction"])
                coverage = float(repair["contact_coverage_fraction"])
                pressure_kpa = float(repair["measured_pressure_kpa"])
                seal_quality = float(repair["seal_quality"])
                if (
                    retention > 0.05
                    and (coverage < 0.75 or seal_quality < 0.60)
                    and "dressing_delamination" in self._definitions
                ):
                    observations.append(
                        self._observation(
                            "dressing_delamination",
                            target_id,
                            (
                                f"scene_retention_fraction={retention:.6f}",
                                f"scene_contact_coverage_fraction={coverage:.6f}",
                                f"scene_seal_quality={seal_quality:.6f}",
                            ),
                        )
                    )
                if (
                    pressure_kpa < -9.0
                    and "dressing_compression_ischemia" in self._definitions
                ):
                    observations.append(
                        self._observation(
                            "dressing_compression_ischemia",
                            target_id,
                            (
                                f"scene_cavity_pressure_kpa={pressure_kpa:.6f}",
                            ),
                        )
                    )
        return tuple(
            sorted(
                observations,
                key=lambda item: (-item.priority, item.complication_id),
            )
        )


class RescuePlanner:
    """Resolve the highest-priority detected complication into a protocol."""

    def __init__(self) -> None:
        contract = _load_contract("rescue_protocols")
        raw = contract.get("protocols", {})
        if not isinstance(raw, dict):
            raise ValueError("rescue_protocols.json must define a protocol map")
        self._protocols = raw

    def plan(
        self,
        complications: Sequence[ComplicationObservation],
    ) -> RescuePlan | None:
        if not complications:
            return None
        selected = sorted(
            complications,
            key=lambda item: (-item.priority, item.complication_id),
        )[0]
        raw_actions = self._protocols.get(selected.rescue_protocol, ())
        if not isinstance(raw_actions, list):
            raise ValueError(
                f"protocol {selected.rescue_protocol!r} must contain an action list"
            )
        return RescuePlan(
            complication_id=selected.complication_id,
            protocol_id=selected.rescue_protocol,
            actions=tuple(
                MappingProxyType(dict(action))
                for action in raw_actions
                if isinstance(action, dict)
            ),
        )


class DynamicPatientRescueBridge:
    """Project contact-derived rescue effects into the shared patient state."""

    def __init__(
        self,
        patient: object,
        *,
        perfusion_region: str = "small_bowel",
        ventilation_target_chest_excursion_m: float = 0.02,
    ) -> None:
        self.patient = patient
        self.perfusion_region = perfusion_region
        self.ventilation_target_chest_excursion_m = float(
            ventilation_target_chest_excursion_m
        )
        if self.ventilation_target_chest_excursion_m <= 0.0:
            raise ValueError(
                "ventilation_target_chest_excursion_m must be positive"
            )
        self._projected_blood_loss_ml = 0.0
        self._projected_crystalloid_ml = 0.0
        self._projected_blood_product_ml = 0.0
        respiration = getattr(patient, "respiration", None)
        self._baseline_tidal_volume_ml = float(
            getattr(respiration, "tidal_volume_ml", 500.0)
        )
        self._baseline_fio2_fraction = float(
            getattr(respiration, "inspired_oxygen_fraction", 0.21)
        )
        self._projected_airway_pressure_damage = 0.0

    def reset(self) -> None:
        self._projected_blood_loss_ml = 0.0
        self._projected_crystalloid_ml = 0.0
        self._projected_blood_product_ml = 0.0
        self._projected_airway_pressure_damage = 0.0
        respiration = getattr(self.patient, "respiration", None)
        if respiration is not None:
            respiration.tidal_volume_ml = self._baseline_tidal_volume_ml
            respiration.inspired_oxygen_fraction = (
                self._baseline_fio2_fraction
            )

    def apply(self, snapshot: RescueEffectsSnapshot) -> None:
        vessel = snapshot.vessel
        cumulative_loss = float(vessel["cumulative_blood_loss_ml"])
        incremental_loss = max(
            0.0,
            cumulative_loss - self._projected_blood_loss_ml,
        )
        if incremental_loss:
            fluid_balance = getattr(self.patient, "fluid_balance", None)
            if fluid_balance is None or not hasattr(fluid_balance, "lose_blood"):
                raise TypeError(
                    "dynamic patient must expose fluid_balance.lose_blood"
                )
            fluid_balance.lose_blood(incremental_loss)
            self._projected_blood_loss_ml = cumulative_loss

        perfusion = getattr(self.patient, "perfusion", None)
        if perfusion is None or not hasattr(perfusion, "set_occlusion"):
            raise TypeError("dynamic patient must expose perfusion.set_occlusion")
        distal = float(vessel["distal_perfusion_fraction"])
        perfusion.set_occlusion(self.perfusion_region, 1.0 - distal)

        anastomosis = snapshot.repairs["bowel_anastomosis"]
        if hasattr(self.patient, "anastomoses"):
            self.patient.anastomoses["autonomous_rescue_bowel"] = {
                "patency_fraction": float(
                    anastomosis["retention_fraction"]
                ),
                "leak_area_mm2": 4.0
                * float(anastomosis["leak_rate_ml_s"]),
                "perfusion_restoration": distal,
            }
        tissue_state = getattr(self.patient, "tissue_state", None)
        if tissue_state is not None and hasattr(tissue_state, "get"):
            wall = snapshot.repairs["abdominal_wall"]
            wall_state = tissue_state.get("abdominal_wall")
            retained_closure = min(
                float(wall["approximation_fraction"]),
                float(wall["retention_fraction"]),
            )
            wall_state.closure_fraction = retained_closure
            wall_state.staples = round(14 * retained_closure)
        film = snapshot.repairs["occlusive_film"]
        if (
            int(film["last_physics_step"]) >= 0
            and hasattr(self.patient, "dressing_state")
        ):
            self.patient.dressing_state = {
                "target": "skin",
                "pressure_kpa": float(film["measured_pressure_kpa"]),
                "seal_fraction": float(film["seal_quality"]),
                "seal_verified": bool(film["seal_verified"]),
                "applied": float(film["retention_fraction"]) > 0.0,
                "source": "autonomous_rescue_contact_effects",
            }
            if perfusion is not None and hasattr(
                perfusion,
                "set_compression",
            ):
                compression = max(
                    0.0,
                    min(
                        1.0,
                        abs(
                            min(
                                0.0,
                                float(film["measured_pressure_kpa"]),
                            )
                        )
                        / 40.0,
                    ),
                )
                perfusion.set_compression("skin", compression)

    def apply_resuscitation(
        self,
        snapshot: ResuscitationSnapshot,
        *,
        physics_step: int,
    ) -> None:
        """Project only scene-supported circulation and ventilation effects."""

        fluid_balance = getattr(self.patient, "fluid_balance", None)
        if fluid_balance is None:
            raise TypeError("dynamic patient must expose fluid_balance")
        crystalloid_ml = float(
            snapshot.channels["crystalloid"]["delivered_to_patient_ml"]
        )
        blood_product_ml = float(
            snapshot.channels["blood_product"]["delivered_to_patient_ml"]
        )
        crystalloid_delta = max(
            0.0,
            crystalloid_ml - self._projected_crystalloid_ml,
        )
        blood_delta = max(
            0.0,
            blood_product_ml - self._projected_blood_product_ml,
        )
        if crystalloid_delta:
            if not hasattr(fluid_balance, "infuse_crystalloid"):
                raise TypeError(
                    "dynamic patient fluid_balance must expose "
                    "infuse_crystalloid"
                )
            fluid_balance.infuse_crystalloid(crystalloid_delta)
            self._projected_crystalloid_ml = crystalloid_ml
        if blood_delta:
            if not hasattr(fluid_balance, "transfuse_blood"):
                raise TypeError(
                    "dynamic patient fluid_balance must expose transfuse_blood"
                )
            fluid_balance.transfuse_blood(blood_delta)
            self._projected_blood_product_ml = blood_product_ml

        ventilation = snapshot.ventilation
        if int(ventilation["last_physics_step"]) >= 0:
            respiration = getattr(self.patient, "respiration", None)
            if respiration is None:
                raise TypeError("dynamic patient must expose respiration")
            if int(ventilation["last_physics_step"]) != physics_step:
                respiration.tidal_volume_ml = self._baseline_tidal_volume_ml
                respiration.inspired_oxygen_fraction = (
                    self._baseline_fio2_fraction
                )
                return
            connected = bool(ventilation["airway_connected"])
            effective_l_min = float(
                ventilation["effective_minute_ventilation_l_min"]
            )
            respiratory_rate = max(
                1.0,
                float(getattr(respiration, "respiratory_rate_bpm", 14.0)),
            )
            flow_supported_tidal_ml = (
                effective_l_min * 1000.0 / respiratory_rate
            )
            chest_supported_tidal_ml = (
                self._baseline_tidal_volume_ml
                * min(
                    2.0,
                    float(ventilation["chest_excursion_m"])
                    / self.ventilation_target_chest_excursion_m,
                )
            )
            delivered_tidal_ml = min(
                flow_supported_tidal_ml,
                chest_supported_tidal_ml,
            )
            respiration.tidal_volume_ml = (
                delivered_tidal_ml
                if connected
                else self._baseline_tidal_volume_ml
            )
            respiration.inspired_oxygen_fraction = (
                float(ventilation["delivered_fio2_fraction"])
                if connected
                else self._baseline_fio2_fraction
            )
            pressure_damage = float(
                ventilation["pressure_damage_fraction"]
            )
            pressure_damage_delta = max(
                0.0,
                pressure_damage - self._projected_airway_pressure_damage,
            )
            if pressure_damage_delta:
                respiration.airway_obstruction_fraction = min(
                    0.95,
                    float(respiration.airway_obstruction_fraction)
                    + pressure_damage_delta,
                )
                self._projected_airway_pressure_damage = pressure_damage

    def advance_physiology(self, dt_s: float) -> None:
        """Advance the shared patient exactly once for one physics interval."""

        if not hasattr(self.patient, "step"):
            raise TypeError("dynamic patient must expose step(dt_s)")
        self.patient.step(dt_s)


class AutonomousRescueORRuntime:
    """Policy/runtime boundary for the autonomous rescue environment."""

    _OUTCOME_FIELD_NAMES: Final = frozenset(
        {
            "bleeding_controlled",
            "closure_fraction",
            "compression_fraction",
            "division_complete",
            "distal_perfusion_fraction",
            "effectiveness",
            "force_n",
            "hemostasis_verified",
            "leak_area_mm2",
            "leak_rate_ml_s",
            "map_mmhg",
            "occlusion_fraction",
            "patch_seal_fraction",
            "patency_fraction",
            "perfusion_restoration",
            "residual_flow_ml_s",
            "seal_fraction",
            "seal_quality",
            "success",
        }
    )

    def __init__(
        self,
        *,
        seed: int = 0,
        dynamic_patient: object | None = None,
    ) -> None:
        self.effects = ContactDrivenRescueEffects(seed=seed)
        self._scene_adapter = self.effects.create_scene_adapter()
        self.resuscitation = ContactDrivenResuscitationEffects()
        self._pump_adapter = self.resuscitation.create_scene_adapter()
        self.patient_bridge = (
            DynamicPatientRescueBridge(
                dynamic_patient,
                ventilation_target_chest_excursion_m=(
                    self.resuscitation.calibration
                    .ventilation_target_chest_excursion_m
                ),
            )
            if dynamic_patient is not None
            else None
        )
        self.detector = ComplicationDetector()
        self.planner = RescuePlanner()
        self.resources = ResourceLedger.from_contract()
        self._actions: list[ActionRecord] = []
        self._sequence = 0
        self._previous_flow_ml_s = 0.0
        self._previous_blood_loss_ml = 0.0
        self._previous_perfusion_fraction = 1.0
        self._previous_overload_damage_fraction = 0.0
        self._pending_support_reward = 0.0
        self._hemostasis_rewarded = False
        self._film_seal_rewarded = False
        self._latest_complications: tuple[ComplicationObservation, ...] = ()
        self._latest_plan: RescuePlan | None = None
        self._last_reward = 0.0
        self._last_resuscitation_reward = 0.0

    @property
    def scene_adapter(self) -> SceneEvidenceAdapter:
        """Environment-only adapter; do not expose it in policy observations."""

        return self._scene_adapter

    @property
    def pump_adapter(self) -> PumpEvidenceAdapter:
        """Environment-only ingress for live pump and vascular-line evidence."""

        return self._pump_adapter

    def _patient_spo2_fraction(self) -> float | None:
        if self.patient_bridge is None:
            return None
        vital_signs = getattr(self.patient_bridge.patient, "vital_signs", None)
        value = getattr(vital_signs, "spo2_fraction", None)
        return None if value is None else float(value)

    def _patient_map_mmhg(self) -> float | None:
        if self.patient_bridge is None:
            return None
        vital_signs = getattr(self.patient_bridge.patient, "vital_signs", None)
        value = getattr(vital_signs, "mean_arterial_pressure_mmhg", None)
        return None if value is None else float(value)

    def _couple_vessel_pressure(
        self,
        frame: PhysicsEvidenceFrame,
    ) -> PhysicsEvidenceFrame:
        """Tie vessel pressure evidence to the shared circulation when present."""

        if frame.target_id != "rescue_vessel":
            return frame
        patient_map = self._patient_map_mmhg()
        if patient_map is None:
            return frame
        measured = frame.measured_upstream_pressure_mmhg
        if measured is None:
            return replace(
                frame,
                measured_upstream_pressure_mmhg=patient_map,
            )
        if abs(measured - patient_map) > 35.0:
            raise ValueError(
                "vessel pressure evidence is inconsistent with shared patient MAP"
            )
        return frame

    def reset(self, *, seed: int | None = None) -> Mapping[str, object]:
        self.effects.reset(seed=seed)
        self.resuscitation.reset()
        if self.patient_bridge is not None:
            self.patient_bridge.reset()
        self.resources = ResourceLedger.from_contract()
        self._actions.clear()
        self._sequence = 0
        self._previous_flow_ml_s = 0.0
        self._previous_blood_loss_ml = 0.0
        self._previous_perfusion_fraction = 1.0
        self._previous_overload_damage_fraction = 0.0
        self._pending_support_reward = 0.0
        self._hemostasis_rewarded = False
        self._film_seal_rewarded = False
        self._latest_complications = ()
        self._latest_plan = None
        self._last_reward = 0.0
        self._last_resuscitation_reward = 0.0
        return self.policy_observation()

    def request_action(
        self,
        action: PolicyAction,
        **unexpected_outcomes: object,
    ) -> ActionRecord:
        """Record intent while rejecting every caller-authored patient result."""

        forbidden = self._OUTCOME_FIELD_NAMES.intersection(unexpected_outcomes)
        if forbidden:
            raise ValueError(
                "policy actions cannot author patient outcomes: "
                + ", ".join(sorted(forbidden))
            )
        if unexpected_outcomes:
            raise ValueError(
                "unsupported policy action fields: "
                + ", ".join(sorted(unexpected_outcomes))
            )
        allowed = {
            "temporary_compression",
            "clip",
            "patch",
            "release_compression",
            "flow_verify",
            "pressure_challenge",
            "capture",
            "align",
            "staple_ring",
            "reinforce",
            "pressure_test",
            "approximate",
            "adhesive",
            "load_test",
            "bond_perimeter",
            "leak_test",
            "transfuse",
            "crystalloid",
            "vasopressor",
            "set_fio2",
            "restore_ventilation",
        }
        if action.requested_action not in allowed:
            record = self._record(
                action,
                ActionStatus.REJECTED,
                "unsupported_policy_intent",
            )
            return record
        record = self._record(
            action,
            ActionStatus.REQUESTED,
            "intent_recorded_waiting_for_scene_effect",
        )
        return record

    def _record(
        self,
        action: PolicyAction,
        status: ActionStatus,
        reason: str,
    ) -> ActionRecord:
        self._sequence += 1
        record = ActionRecord(
            sequence=self._sequence,
            physics_step=self.effects.snapshot().physics_step,
            action=action,
            status=status,
            reason=reason,
        )
        self._actions.append(record)
        return record

    def advance_scene(
        self,
        frame: PhysicsEvidenceFrame,
        *,
        companion_frames: Sequence[PhysicsEvidenceFrame] = (),
    ) -> Mapping[str, object]:
        """Advance outcomes once from a complete post-physics scene interval."""

        previous = self.effects.snapshot()
        previous_spo2 = self._patient_spo2_fraction()
        previous_map = self._patient_map_mmhg()
        frame = self._couple_vessel_pressure(frame)
        observed_target_ids = {frame.target_id}
        current = self._scene_adapter.publish(frame)
        for companion in companion_frames:
            companion = self._couple_vessel_pressure(companion)
            if (
                companion.physics_step != frame.physics_step
                or companion.simulation_time_s != frame.simulation_time_s
                or companion.dt_s != frame.dt_s
            ):
                raise ValueError(
                    "companion scene evidence must share the primary "
                    "physics interval"
                )
            observed_target_ids.add(companion.target_id)
            current = self._scene_adapter.publish(companion)
        current = self._scene_adapter.finalize_interval(
            frozenset(observed_target_ids)
        )
        if self.patient_bridge is not None:
            self.patient_bridge.apply(current)
            self.patient_bridge.apply_resuscitation(
                self.resuscitation.snapshot(),
                physics_step=frame.physics_step,
            )
            self.patient_bridge.advance_physiology(frame.dt_s)
        current_spo2 = self._patient_spo2_fraction()
        current_map = self._patient_map_mmhg()
        self._latest_complications = self.detector.detect(
            current,
            baseline_blood_volume_ml=(
                self.effects.calibration.baseline_blood_volume_ml
            ),
            effective_resuscitation_gain_ml=(
                self.resuscitation.snapshot()
                .effective_circulating_volume_gain_ml
            ),
            spo2_fraction=current_spo2,
        )
        self._latest_plan = self.planner.plan(self._latest_complications)
        previous_flow = float(previous.vessel["residual_flow_ml_s"])
        current_flow = float(current.vessel["residual_flow_ml_s"])
        previous_loss = float(previous.vessel["cumulative_blood_loss_ml"])
        current_loss = float(current.vessel["cumulative_blood_loss_ml"])
        perfusion = float(current.vessel["distal_perfusion_fraction"])
        verified = bool(current.vessel["hemostasis_verified"])
        overload = float(current.vessel["overload_damage_fraction"])
        newly_verified = verified and not self._hemostasis_rewarded
        previous_film = previous.repairs["occlusive_film"]
        current_film = current.repairs["occlusive_film"]
        film_leak = float(current_film["leak_rate_ml_s"])
        film_quality = float(current_film["seal_quality"])
        film_verified = bool(current_film["seal_verified"])
        newly_film_verified = (
            film_verified and not self._film_seal_rewarded
        )
        film_was_observed = int(previous_film["last_physics_step"]) >= 0
        film_leak_improvement = (
            float(previous_film["leak_rate_ml_s"]) - film_leak
            if film_was_observed
            else 0.0
        )
        film_quality_improvement = (
            film_quality - float(previous_film["seal_quality"])
            if film_was_observed
            else 0.0
        )
        flow_improvement = (
            previous_flow - current_flow
            if previous.physics_step >= 0
            else 0.0
        )
        oxygenation_improvement = (
            current_spo2 - previous_spo2
            if current_spo2 is not None and previous_spo2 is not None
            else 0.0
        )
        map_improvement = (
            current_map - previous_map
            if current_map is not None and previous_map is not None
            else 0.0
        )
        self._last_reward = (
            2.0 * flow_improvement
            - 0.5 * (current_loss - previous_loss)
            - 1.5 * max(0.0, self._previous_perfusion_fraction - perfusion)
            - 4.0
            * max(
                0.0,
                overload - self._previous_overload_damage_fraction,
            )
            + (8.0 if newly_verified else 0.0)
            + 1.5 * film_leak_improvement
            + 2.0 * film_quality_improvement
            + (6.0 if newly_film_verified else 0.0)
            + 20.0 * oxygenation_improvement
            + map_improvement
            + self._pending_support_reward
        )
        self._pending_support_reward = 0.0
        self._hemostasis_rewarded = self._hemostasis_rewarded or verified
        self._film_seal_rewarded = (
            self._film_seal_rewarded or film_verified
        )
        self._previous_flow_ml_s = current_flow
        self._previous_blood_loss_ml = current_loss
        self._previous_perfusion_fraction = perfusion
        self._previous_overload_damage_fraction = overload
        return self.policy_observation()

    def advance_resuscitation(
        self,
        frame: PumpEvidenceFrame,
    ) -> Mapping[str, object]:
        """Advance volume support from conserved post-physics pump evidence."""

        previous = self.resuscitation.snapshot()
        current = self._pump_adapter.publish(frame)

        resource_id = {
            "crystalloid": "crystalloid_ml",
            "blood_product": "blood_products_ml",
            "vasopressor": "vasopressor_syringes",
        }[frame.channel_id]
        previous_withdrawn = float(
            previous.channels[frame.channel_id][
                "withdrawn_from_reservoir_ml"
            ]
        )
        current_withdrawn = float(
            current.channels[frame.channel_id][
                "withdrawn_from_reservoir_ml"
            ]
        )
        withdrawn_delta = max(0.0, current_withdrawn - previous_withdrawn)
        resource_amount = withdrawn_delta
        if frame.channel_id == "vasopressor":
            resource_amount /= (
                self.resuscitation.calibration
                .vasopressor_volume_per_stroke_ml
            )
        if resource_amount:
            self.resources.consume(resource_id, resource_amount)

        rescue_snapshot = self.effects.snapshot()
        self._latest_complications = self.detector.detect(
            rescue_snapshot,
            baseline_blood_volume_ml=(
                self.effects.calibration.baseline_blood_volume_ml
            ),
            effective_resuscitation_gain_ml=(
                current.effective_circulating_volume_gain_ml
            ),
            spo2_fraction=self._patient_spo2_fraction(),
        )
        self._latest_plan = self.planner.plan(self._latest_complications)

        gain_delta = max(
            0.0,
            current.effective_circulating_volume_gain_ml
            - previous.effective_circulating_volume_gain_ml,
        )
        deficit_before = max(
            0.0,
            self.effects.calibration.baseline_blood_volume_ml
            - float(rescue_snapshot.vessel["blood_volume_ml"])
            - previous.effective_circulating_volume_gain_ml,
        )
        useful_gain = min(gain_delta, deficit_before)
        previous_channel = previous.channels[frame.channel_id]
        current_channel = current.channels[frame.channel_id]
        waste_delta = max(
            0.0,
            float(current_channel["wasted_or_extravasated_ml"])
            - float(previous_channel["wasted_or_extravasated_ml"]),
        )
        pressure_damage_delta = max(
            0.0,
            float(current_channel["pressure_damage_fraction"])
            - float(previous_channel["pressure_damage_fraction"]),
        )
        self._last_resuscitation_reward = (
            0.01 * useful_gain
            - 0.02 * waste_delta
            - 10.0 * pressure_damage_delta
        )
        self._pending_support_reward += self._last_resuscitation_reward
        self._last_reward = self._last_resuscitation_reward
        return self.policy_observation()

    def advance_ventilation(
        self,
        frame: VentilationEvidenceFrame,
    ) -> Mapping[str, object]:
        """Advance airway support from connected circuit and chest evidence."""

        previous = self.resuscitation.snapshot()
        current = self._pump_adapter.publish_ventilation(frame)
        damage_delta = max(
            0.0,
            float(current.ventilation["pressure_damage_fraction"])
            - float(previous.ventilation["pressure_damage_fraction"]),
        )
        self._last_resuscitation_reward = -10.0 * damage_delta
        self._pending_support_reward += self._last_resuscitation_reward
        self._last_reward = self._last_resuscitation_reward
        return self.policy_observation()

    def policy_observation(self) -> Mapping[str, object]:
        patient = self.effects.policy_observation()
        active = tuple(
            MappingProxyType(
                {
                    "id": item.complication_id,
                    "priority": item.priority,
                    "target_id": item.target_id,
                    "evidence": item.evidence,
                }
            )
            for item in self._latest_complications
        )
        plan = None
        if self._latest_plan is not None:
            plan = MappingProxyType(
                {
                    "complication_id": self._latest_plan.complication_id,
                    "protocol_id": self._latest_plan.protocol_id,
                    "actions": self._latest_plan.actions,
                }
            )
        return MappingProxyType(
            {
                "patient": patient,
                "resuscitation": self.resuscitation.snapshot().channels,
                "ventilation": self.resuscitation.snapshot().ventilation,
                "active_complications": active,
                "rescue_plan": plan,
                "resources": self.resources.snapshot(),
                "last_reward": self._last_reward,
                "last_resuscitation_reward": (
                    self._last_resuscitation_reward
                ),
                "action_count": len(self._actions),
            }
        )

    def trace(self) -> tuple[ActionRecord, ...]:
        return tuple(self._actions)


def autonomous_rescue_or_cfg(
    prim_path: str = "{ENV_REGEX_NS}/AutonomousRescueOR",
):
    """Return a deferred Isaac Lab asset configuration for the OR stage."""

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Isaac Lab is required to create the Autonomous Rescue OR stage"
        ) from error
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(AUTONOMOUS_RESCUE_OR_USD),
        ),
    )


def _create_vertex_xform_attachment(
    stage,
    *,
    attachment_path: str,
    source_path: str,
    target_path: str,
    vertex_indices: Sequence[int],
) -> str:
    """Bind selected TetMesh vertices to an authored fixture without snapping."""

    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    source_prim = stage.GetPrimAtPath(source_path)
    target_prim = stage.GetPrimAtPath(target_path)
    if (
        not source_prim.IsValid()
        or not source_prim.IsA(UsdGeom.PointBased)
    ):
        raise RuntimeError(
            f"rescue attachment source is not point based: {source_path}"
        )
    if not target_prim.IsValid() or not UsdGeom.Xformable(target_prim):
        raise RuntimeError(
            f"rescue attachment target is not xformable: {target_path}"
        )
    points = list(UsdGeom.PointBased(source_prim).GetPointsAttr().Get() or ())
    selected = tuple(dict.fromkeys(int(index) for index in vertex_indices))
    if not selected or any(index < 0 or index >= len(points) for index in selected):
        raise RuntimeError(
            f"rescue attachment has no valid source vertices: {source_path}"
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
        for index in selected
    ]
    parent_path = str(Sdf.Path(attachment_path).GetParentPath())
    if not stage.GetPrimAtPath(parent_path).IsValid():
        stage.DefinePrim(parent_path, "Scope")
    if stage.GetPrimAtPath(attachment_path).IsValid():
        stage.RemovePrim(attachment_path)
    attachment = stage.DefinePrim(
        attachment_path,
        "OmniPhysicsVtxXformAttachment",
    )
    attachment.CreateRelationship("omniphysics:src0").SetTargets(
        [Sdf.Path(source_path)]
    )
    attachment.CreateRelationship("omniphysics:src1").SetTargets(
        [Sdf.Path(target_path)]
    )
    attachment.CreateAttribute(
        "omniphysics:vtxIndicesSrc0",
        Sdf.ValueTypeNames.IntArray,
    ).Set(Vt.IntArray(selected))
    attachment.CreateAttribute(
        "omniphysics:localPositionsSrc1",
        Sdf.ValueTypeNames.Point3fArray,
    ).Set(Vt.Vec3fArray(local_positions))
    attachment.CreateAttribute(
        "omniphysics:attachmentEnabled",
        Sdf.ValueTypeNames.Bool,
    ).Set(True)
    return attachment_path


def anchor_rescue_vessel(
    vessel_path: str,
    *,
    stage=None,
    endpoint_tolerance_m: float = 1.0e-6,
) -> tuple[str, str]:
    """Fix the two vessel ends while leaving the central rescue zone deformable."""

    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    from pxr import UsdGeom

    root = vessel_path.rstrip("/")
    simulation_path = f"{root}/VesselWall/SimulationMesh"
    simulation_prim = stage.GetPrimAtPath(simulation_path)
    if (
        not simulation_prim.IsValid()
        or not simulation_prim.IsA(UsdGeom.PointBased)
    ):
        raise RuntimeError(
            f"rescue vessel has no simulation TetMesh at {simulation_path}"
        )
    points = list(
        UsdGeom.PointBased(simulation_prim).GetPointsAttr().Get() or ()
    )
    if not points:
        raise RuntimeError(f"rescue vessel TetMesh has no points: {simulation_path}")
    x_values = [float(point[0]) for point in points]
    minimum_x = min(x_values)
    maximum_x = max(x_values)
    endpoint_indices = {
        "Left": [
            index
            for index, value in enumerate(x_values)
            if abs(value - minimum_x) <= endpoint_tolerance_m
        ],
        "Right": [
            index
            for index, value in enumerate(x_values)
            if abs(value - maximum_x) <= endpoint_tolerance_m
        ],
    }
    created = []
    for side in ("Left", "Right"):
        indices = endpoint_indices[side]
        if len(indices) < 4:
            raise RuntimeError(
                f"{side.lower()} rescue vessel endpoint is too sparse: "
                f"{len(indices)} vertices"
            )
        created.append(
            _create_vertex_xform_attachment(
                stage,
                attachment_path=f"{root}/RuntimeAttachments/{side}Fixture",
                source_path=simulation_path,
                target_path=f"{root}/AttachmentMasks/{side}Fixture",
                vertex_indices=indices,
            )
        )
    return tuple(created)


def rescue_vessel_cfg(
    prim_path: str = "{ENV_REGEX_NS}/AutonomousRescueVessel",
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.06),
):
    """Return the live, endpoint-anchored rescue substrate for a native bench."""

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Isaac Lab is required to create the Autonomous Rescue vessel"
        ) from error

    spawn = sim_utils.UsdFileCfg(
        usd_path=str(RESCUE_VESSEL_USD),
        variants={"hemostasis_state": "bleeding"},
    )
    source_spawn = spawn.func

    def spawn_anchored_rescue_vessel(
        spawned_prim_path: str,
        cfg: sim_utils.UsdFileCfg,
        translation=None,
        orientation=None,
        **kwargs,
    ):
        root_prim = source_spawn(
            spawned_prim_path,
            cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        anchor_rescue_vessel(str(root_prim.GetPath()))
        return root_prim

    spawn.func = spawn_anchored_rescue_vessel
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(value) for value in position),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=spawn,
    )


def resuscitation_module_cfg(
    prim_path: str = "{ENV_REGEX_NS}/ResuscitationModule",
    *,
    position: tuple[float, float, float] = (-0.75, 0.0, 0.0),
):
    """Return the articulated plunger/ventilation module for a live scene."""

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Isaac Lab is required to create the resuscitation module"
        ) from error
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(RESUSCITATION_MODULE_USD),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(value) for value in position),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*plunger_joint": 0.0,
                "ventilation_valve_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "fluid_pumps": ImplicitActuatorCfg(
                joint_names_expr=[".*plunger_joint"],
                effort_limit_sim=120.0,
                velocity_limit_sim=0.06,
                stiffness=2200.0,
                damping=80.0,
            ),
            "ventilation": ImplicitActuatorCfg(
                joint_names_expr=["ventilation_valve_joint"],
                effort_limit_sim=18.0,
                velocity_limit_sim=1.5,
                stiffness=120.0,
                damping=8.0,
            ),
        },
    )


__all__ = [
    "ASSET_DIRECTORY",
    "AUTONOMOUS_RESCUE_OR_USD",
    "ActionRecord",
    "ActionStatus",
    "AutonomousRescueORRuntime",
    "ComplicationDetector",
    "ComplicationObservation",
    "DEFORMABLE_RESCUE_SUITE_USD",
    "DynamicPatientRescueBridge",
    "PolicyAction",
    "RESCUE_VESSEL_USD",
    "RESUSCITATION_MODULE_USD",
    "RescuePlan",
    "RescuePlanner",
    "ResourceLedger",
    "TOOL_CHANGER_PAYLOAD_USD",
    "anchor_rescue_vessel",
    "autonomous_rescue_or_cfg",
    "rescue_vessel_cfg",
    "resuscitation_module_cfg",
]
