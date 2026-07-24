"""Isaac Lab helpers for the DrAnmar topical skin-adhesive system.

The applicator mechanism is an articulated research asset.  The adhesive bead
is a discrete kinematic task representation; this module does not model liquid
flow, curing chemistry, tissue bonding, or clinical performance.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


CATALOG_SUBPATH = Path("Props/SurgicalClosure/SkinAdhesive")
EXTENSION_ROOT = Path(__file__).resolve().parents[3]
ASSET_DATA_ROOT = EXTENSION_ROOT / "data"
ASSET_ROOT_ENV = "DRANMAR_SKIN_ADHESIVE_ASSET_ROOT"

PADDLE_TRAVEL_DEG = 11.0
PADDLE_TRAVEL_RAD = math.radians(PADDLE_TRAVEL_DEG)
PISTON_TRAVEL_M = 0.009
APP_VERSION = "0.1.0"

APPLICATOR_FRAMES = frozenset(
    {
        "body_grasp",
        "paddle_left_contact",
        "paddle_right_contact",
        "tip",
        "dispense_exit",
        "placement_reference",
        "path_tangent_reference",
        "reservoir_center",
        "activation_reference",
        "count_reference",
    }
)
CAP_FRAMES = frozenset({"cap_grasp", "cap_snap_axis", "count_reference"})
BEAD_FRAMES = frozenset(
    {
        "bead_start",
        "bead_mid",
        "bead_end",
        "path_tangent_reference",
        "count_reference",
    }
)


class ApplicatorState(str, Enum):
    SEALED = "sealed"
    ACTIVATED = "activated"
    EMPTY = "empty"


class BeadState(str, Enum):
    FRESH = "fresh"
    CURED = "cured"


def _coerce_enum(value: str | Enum, enum_type: type[Enum], label: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(str(item.value) for item in enum_type)
        raise ValueError(
            f"Unsupported {label} state {value!r}; expected one of: {allowed}"
        ) from exc


def asset_root(*, require: bool = True) -> Path:
    """Resolve the catalog extraction or repository-local asset directory."""

    candidates: list[Path] = []
    if override := os.environ.get(ASSET_ROOT_ENV):
        candidates.append(Path(override).expanduser())
    candidates.append(ASSET_DATA_ROOT / CATALOG_SUBPATH)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    if require:
        rendered = "\n".join(f"- {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "DrAnmar skin-adhesive assets were not found. Checked:\n"
            f"{rendered}\nSet {ASSET_ROOT_ENV} to a catalog extraction."
        )
    return candidates[0]


def articulated_usd() -> Path:
    return asset_root() / "skin_adhesive_applicator_articulated.usda"


def rigid_proxy_usd() -> Path:
    return asset_root() / "skin_adhesive_applicator_rigid_proxy.usda"


def cap_usd() -> Path:
    return asset_root() / "skin_adhesive_cap.usda"


def bead_usd() -> Path:
    return asset_root() / "skin_adhesive_bead.usda"


def physics_profile() -> Path:
    return asset_root() / "physics_profile.json"


def interaction_frames() -> Path:
    return asset_root() / "interaction_frames.json"


def _resolved_path(value: str | Path | None, fallback: Path) -> str:
    selected = fallback if value is None else Path(value).expanduser()
    return str(selected.resolve())


def applicator_variant(state: str | ApplicatorState) -> dict[str, str]:
    selected = _coerce_enum(state, ApplicatorState, "applicator")
    return {"state": str(selected.value)}


def bead_variant(state: str | BeadState) -> dict[str, str]:
    selected = _coerce_enum(state, BeadState, "bead")
    return {"state": str(selected.value)}


def activation_targets(activation: float) -> dict[str, float]:
    """Map normalized activation to coordinated Isaac joint targets.

    Revolute targets are returned in radians and the piston target in metres.
    """

    value = float(activation)
    if not math.isfinite(value):
        raise ValueError("activation must be finite")
    value = min(max(value, 0.0), 1.0)
    return {
        "left_paddle_joint": -PADDLE_TRAVEL_RAD * value,
        "right_paddle_joint": PADDLE_TRAVEL_RAD * value,
        "metering_piston_joint": PISTON_TRAVEL_M * value,
    }


def make_articulated_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/SkinAdhesiveApplicator",
    state: str | ApplicatorState = ApplicatorState.ACTIVATED,
    usd_path: str | Path | None = None,
    disable_gravity: bool = False,
):
    """Build the graspable four-link applicator articulation."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore
    from isaaclab.assets import ArticulationCfg  # type: ignore

    selected = _coerce_enum(state, ApplicatorState, "applicator")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_resolved_path(usd_path, articulated_usd()),
            variants={"state": str(selected.value)},
            semantic_tags=[
                ("class", "skin_adhesive_applicator"),
                ("device_type", "surgical_closure_device"),
                ("workflow_activation", "activation"),
                ("workflow_dispensing", "dispensing"),
                ("state", str(selected.value)),
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=disable_gravity,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),
            joint_pos={
                "left_paddle_joint": 0.0,
                "right_paddle_joint": 0.0,
                "metering_piston_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "paddles": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_paddle_joint",
                    "right_paddle_joint",
                ],
                effort_limit_sim=1.8,
                velocity_limit_sim=2.0,
                stiffness=0.075,
                damping=0.012,
                armature=0.002,
            ),
            "metering_piston": ImplicitActuatorCfg(
                joint_names_expr=["metering_piston_joint"],
                effort_limit_sim=16.0,
                velocity_limit_sim=0.03,
                stiffness=320.0,
                damping=4.5,
                armature=0.02,
            ),
        },
    )


def make_rigid_proxy_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/SkinAdhesiveApplicator",
    state: str | ApplicatorState = ApplicatorState.SEALED,
    usd_path: str | Path | None = None,
):
    """Build the one-body perception and handover representation."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.assets import RigidObjectCfg  # type: ignore

    selected = _coerce_enum(state, ApplicatorState, "applicator")
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_resolved_path(usd_path, rigid_proxy_usd()),
            variants={"state": str(selected.value)},
            semantic_tags=[
                ("class", "skin_adhesive_applicator"),
                ("representation", "rigid_proxy"),
                ("state", str(selected.value)),
            ],
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
    )


def make_cap_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/SkinAdhesiveCap",
    usd_path: str | Path | None = None,
):
    """Build the independently graspable removable cap."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.assets import RigidObjectCfg  # type: ignore

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_resolved_path(usd_path, cap_usd()),
            semantic_tags=[
                ("class", "skin_adhesive_cap"),
                ("workflow_uncapping", "uncapping"),
                ("workflow_disposal", "disposal"),
            ],
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.02)),
    )


def make_bead_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/SkinAdhesiveBead",
    state: str | BeadState = BeadState.FRESH,
    usd_path: str | Path | None = None,
):
    """Build the fresh/cured kinematic deposit task representation."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.assets import RigidObjectCfg  # type: ignore

    selected = _coerce_enum(state, BeadState, "bead")
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_resolved_path(usd_path, bead_usd()),
            variants={"state": str(selected.value)},
            semantic_tags=[
                ("class", "skin_adhesive"),
                ("representation", "kinematic_deposit"),
                ("workflow_closure", "closure"),
                ("state", str(selected.value)),
            ],
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.001)),
    )


def tensor_value(value: Any) -> Any:
    """Return a Torch-compatible tensor from Isaac/Warp proxy values."""

    if hasattr(value, "torch"):
        candidate = value.torch
        return candidate() if callable(candidate) else candidate
    try:
        import warp as wp  # type: ignore

        if isinstance(value, wp.array):
            return wp.to_torch(value)
    except (ImportError, ModuleNotFoundError, TypeError):
        pass
    return value


def set_activation_target(articulation: Any, activation: float) -> dict[str, float]:
    """Apply coordinated targets to a live Isaac Lab articulation."""

    targets = activation_targets(activation)
    joint_names = list(articulation.joint_names)
    ordered_names = list(targets)
    try:
        joint_ids = [joint_names.index(name) for name in ordered_names]
    except ValueError as exc:
        raise RuntimeError(
            "Skin-adhesive articulation is missing one or more mechanism joints"
        ) from exc

    import torch

    positions = tensor_value(articulation.data.joint_pos)
    device = positions.device
    target_tensor = torch.tensor(
        [[targets[name] for name in ordered_names]],
        dtype=positions.dtype,
        device=device,
    )
    articulation.set_joint_position_target(target_tensor, joint_ids=joint_ids)
    return targets


def set_state_variant(
    stage: Any,
    prim_path: str,
    state: str | ApplicatorState | BeadState,
) -> str:
    """Select an authored state before creating the corresponding physics view."""

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid skin-adhesive prim path: {prim_path}")
    variant_set = prim.GetVariantSets().GetVariantSet("state")
    if not variant_set.IsValid():
        raise ValueError(f"{prim_path} has no state variant set")
    value = str(state.value if isinstance(state, Enum) else state).strip().lower()
    if value not in variant_set.GetVariantNames():
        raise ValueError(
            f"Unsupported state {value!r}; available: {variant_set.GetVariantNames()}"
        )
    if not variant_set.SetVariantSelection(value):
        raise RuntimeError(f"Failed to select state {value!r} on {prim_path}")
    return value


def frame_path(root_prim_path: str, name: str, *, asset: str = "applicator") -> str:
    """Return and validate an authored robot-interaction frame path."""

    frame_sets = {
        "applicator": APPLICATOR_FRAMES,
        "cap": CAP_FRAMES,
        "bead": BEAD_FRAMES,
    }
    if asset not in frame_sets:
        raise ValueError(f"Unknown skin-adhesive asset kind: {asset}")
    if name not in frame_sets[asset]:
        raise ValueError(f"Unknown {asset} frame: {name}")
    return f"{root_prim_path.rstrip('/')}/Frames/{name}"


@dataclass(frozen=True)
class DispenseEvent:
    """One logical bead-placement event without fluid or dose claims."""

    sequence: int
    path_length_m: float
    activation: float
    bead_state: str


@dataclass
class DispenseLedger:
    """Auditable logical deposit ledger for task-state transitions."""

    events: list[DispenseEvent] = field(default_factory=list)

    def record(
        self,
        *,
        path_length_m: float,
        activation: float,
        bead_state: str | BeadState = BeadState.FRESH,
    ) -> DispenseEvent:
        length = float(path_length_m)
        if not math.isfinite(length) or length < 0.0:
            raise ValueError("path_length_m must be finite and non-negative")
        normalized = float(activation)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError("activation must be between 0 and 1")
        selected = _coerce_enum(bead_state, BeadState, "bead")
        event = DispenseEvent(
            sequence=len(self.events) + 1,
            path_length_m=length,
            activation=normalized,
            bead_state=str(selected.value),
        )
        self.events.append(event)
        return event

    def reset(self) -> None:
        self.events.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "dr.anmar.skin-adhesive-dispense-ledger.v1",
            "event_count": len(self.events),
            "total_path_length_m": sum(item.path_length_m for item in self.events),
            "dose_model": None,
            "fluid_solver": False,
            "events": [
                {
                    "sequence": item.sequence,
                    "path_length_m": item.path_length_m,
                    "activation": item.activation,
                    "bead_state": item.bead_state,
                }
                for item in self.events
            ],
        }


try:
    from i4h_asset_helper import BaseI4HAssets as _BaseI4HAssets
except (ImportError, ModuleNotFoundError):
    _BaseI4HAssets = object


class SurgicalClosureAssets(_BaseI4HAssets):
    """I4H-compatible relative catalog paths."""

    SKIN_ADHESIVE_APPLICATOR_ARTICULATED = (
        "Props/SurgicalClosure/SkinAdhesive/"
        "skin_adhesive_applicator_articulated.usda"
    )
    SKIN_ADHESIVE_APPLICATOR_RIGID_PROXY = (
        "Props/SurgicalClosure/SkinAdhesive/"
        "skin_adhesive_applicator_rigid_proxy.usda"
    )
    SKIN_ADHESIVE_CAP = (
        "Props/SurgicalClosure/SkinAdhesive/skin_adhesive_cap.usda"
    )
    SKIN_ADHESIVE_BEAD = (
        "Props/SurgicalClosure/SkinAdhesive/skin_adhesive_bead.usda"
    )


__all__ = [
    "APP_VERSION",
    "APPLICATOR_FRAMES",
    "ApplicatorState",
    "BEAD_FRAMES",
    "BeadState",
    "CAP_FRAMES",
    "DispenseEvent",
    "DispenseLedger",
    "PADDLE_TRAVEL_DEG",
    "PADDLE_TRAVEL_RAD",
    "PISTON_TRAVEL_M",
    "SurgicalClosureAssets",
    "activation_targets",
    "applicator_variant",
    "articulated_usd",
    "asset_root",
    "bead_usd",
    "bead_variant",
    "cap_usd",
    "frame_path",
    "interaction_frames",
    "make_articulated_cfg",
    "make_bead_cfg",
    "make_cap_cfg",
    "make_rigid_proxy_cfg",
    "physics_profile",
    "rigid_proxy_usd",
    "set_activation_target",
    "set_state_variant",
    "tensor_value",
]

__version__ = APP_VERSION
