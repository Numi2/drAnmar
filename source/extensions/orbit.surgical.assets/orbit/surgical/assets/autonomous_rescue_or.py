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

from dataclasses import dataclass, field
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
    ) -> tuple[ComplicationObservation, ...]:
        vessel = snapshot.vessel
        observations: list[ComplicationObservation] = []
        residual_flow = float(vessel["residual_flow_ml_s"])
        blood_fraction = (
            float(vessel["blood_volume_ml"]) / baseline_blood_volume_ml
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
    ) -> None:
        self.patient = patient
        self.perfusion_region = perfusion_region
        self._projected_blood_loss_ml = 0.0

    def reset(self) -> None:
        self._projected_blood_loss_ml = 0.0

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
        if tissue_state is not None and hasattr(tissue_state, "staple"):
            wall = snapshot.repairs["abdominal_wall"]
            tissue_state.staple(
                "abdominal_wall",
                float(wall["retention_fraction"]),
            )
        if hasattr(self.patient, "vital_signs") and hasattr(
            self.patient.vital_signs,
            "update",
        ):
            self.patient.vital_signs.update(self.patient)


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
        self.patient_bridge = (
            DynamicPatientRescueBridge(dynamic_patient)
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
        self._hemostasis_rewarded = False
        self._latest_complications: tuple[ComplicationObservation, ...] = ()
        self._latest_plan: RescuePlan | None = None
        self._last_reward = 0.0

    @property
    def scene_adapter(self) -> SceneEvidenceAdapter:
        """Environment-only adapter; do not expose it in policy observations."""

        return self._scene_adapter

    def reset(self, *, seed: int | None = None) -> Mapping[str, object]:
        self.effects.reset(seed=seed)
        if self.patient_bridge is not None:
            self.patient_bridge.reset()
        self.resources = ResourceLedger.from_contract()
        self._actions.clear()
        self._sequence = 0
        self._previous_flow_ml_s = 0.0
        self._previous_blood_loss_ml = 0.0
        self._previous_perfusion_fraction = 1.0
        self._previous_overload_damage_fraction = 0.0
        self._hemostasis_rewarded = False
        self._latest_complications = ()
        self._latest_plan = None
        self._last_reward = 0.0
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
        }
        if action.requested_action not in allowed:
            record = self._record(
                action,
                ActionStatus.REJECTED,
                "unsupported_policy_intent",
            )
            return record
        if action.requested_action in {"pressure_challenge", "flow_verify"}:
            self.effects.start_pressure_challenge()
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
    ) -> Mapping[str, object]:
        """Advance outcomes from one authoritative post-physics frame."""

        previous = self.effects.snapshot()
        current = self._scene_adapter.publish(frame)
        if self.patient_bridge is not None:
            self.patient_bridge.apply(current)
        self._latest_complications = self.detector.detect(
            current,
            baseline_blood_volume_ml=(
                self.effects.calibration.baseline_blood_volume_ml
            ),
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
        flow_improvement = (
            previous_flow - current_flow
            if previous.physics_step >= 0
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
        )
        self._hemostasis_rewarded = self._hemostasis_rewarded or verified
        self._previous_flow_ml_s = current_flow
        self._previous_blood_loss_ml = current_loss
        self._previous_perfusion_fraction = perfusion
        self._previous_overload_damage_fraction = overload
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
                "active_complications": active,
                "rescue_plan": plan,
                "resources": self.resources.snapshot(),
                "last_reward": self._last_reward,
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
    "RescuePlan",
    "RescuePlanner",
    "ResourceLedger",
    "TOOL_CHANGER_PAYLOAD_USD",
    "anchor_rescue_vessel",
    "autonomous_rescue_or_cfg",
    "rescue_vessel_cfg",
]
