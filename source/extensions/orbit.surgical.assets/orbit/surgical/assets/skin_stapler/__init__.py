"""Helpers for the DrAnmar category-level skin stapler research asset."""

from .closure_task import ClosureLine, PlacementAssessment, assess_placement, spacing_errors_m
from .control import (
    DEPLOYMENT_THRESHOLD_M,
    FIRE_THRESHOLD_DEG,
    FIRE_THRESHOLD_RAD,
    FireCycle,
    PUSHER_TRAVEL_M,
    REARM_THRESHOLD_DEG,
    REARM_THRESHOLD_RAD,
    TRIGGER_LIMIT_DEG,
    TRIGGER_LIMIT_RAD,
    synchronized_joint_targets,
    synchronized_joint_targets_deg,
)
from .deployment import (
    DeploymentEvent,
    StapleDeploymentEvent,
    StapleMagazine,
    TriggerEdgeDeploymentController,
    add_staple_reference,
    deployed_staple_path,
    spawn_staple_at_exit,
    spawn_formed_staple_at_placement,
)
from .isaac_lab_v2 import (
    make_rigid_proxy_cfg as make_rigid_skin_stapler_cfg,
    make_staple_cfg as make_rigid_staple_cfg,
)
from .isaac_lab_v3 import (
    make_articulation_cfg as make_articulated_skin_stapler_cfg,
    make_staple_cfg as make_articulated_staple_cfg,
)
from .paths import (
    SurgicalClosureAssets,
    articulated_usd,
    asset_root,
    interaction_frames,
    physics_profile,
    rigid_proxy_usd,
    staple_usd,
)
from .state import StaplerState, variant_selection
from .tensors import unwrap_tensor

__all__ = [
    "ClosureLine",
    "DEPLOYMENT_THRESHOLD_M",
    "DeploymentEvent",
    "FIRE_THRESHOLD_DEG",
    "FIRE_THRESHOLD_RAD",
    "FireCycle",
    "PUSHER_TRAVEL_M",
    "PlacementAssessment",
    "REARM_THRESHOLD_DEG",
    "REARM_THRESHOLD_RAD",
    "SurgicalClosureAssets",
    "StapleDeploymentEvent",
    "StapleMagazine",
    "StaplerState",
    "TRIGGER_LIMIT_DEG",
    "TRIGGER_LIMIT_RAD",
    "TriggerEdgeDeploymentController",
    "add_staple_reference",
    "articulated_usd",
    "assess_placement",
    "asset_root",
    "deployed_staple_path",
    "interaction_frames",
    "make_articulated_skin_stapler_cfg",
    "make_articulated_staple_cfg",
    "make_rigid_skin_stapler_cfg",
    "make_rigid_staple_cfg",
    "physics_profile",
    "rigid_proxy_usd",
    "spacing_errors_m",
    "spawn_staple_at_exit",
    "spawn_formed_staple_at_placement",
    "staple_usd",
    "synchronized_joint_targets",
    "synchronized_joint_targets_deg",
    "unwrap_tensor",
    "variant_selection",
]

__version__ = "0.2.0"
