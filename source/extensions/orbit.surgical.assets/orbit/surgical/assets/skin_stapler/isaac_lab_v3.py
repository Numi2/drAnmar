"""Isaac Sim 6.0.1 / Isaac Lab 3.0 beta 2 configuration factories."""
from __future__ import annotations

from pathlib import Path

from .paths import articulated_usd, staple_usd
from .semantics import semantic_tags, staple_semantic_tags
from .state import StaplerState, coerce_state, variant_selection


def _path(value: str | Path | None, fallback: Path) -> str:
    resolved = fallback if value is None else Path(value).expanduser()
    return str(resolved.resolve())


def make_articulation_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/SkinStapler",
    state: str | StaplerState = StaplerState.LOADED,
    usd_path: str | Path | None = None,
    disable_gravity: bool = False,
    fix_root_link: bool | None = None,
):
    """Create an ``ArticulationCfg`` with state selected before view creation."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore
    from isaaclab.assets import ArticulationCfg  # type: ignore

    selected = coerce_state(state)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_path(usd_path, articulated_usd()),
            variants=variant_selection(selected),
            semantic_tags=semantic_tags(selected),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=disable_gravity,
                max_depenetration_velocity=2.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                fix_root_link=fix_root_link,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),
            joint_pos={"trigger_joint": 0.0, "pusher_joint": 0.0},
            joint_vel={"trigger_joint": 0.0, "pusher_joint": 0.0},
        ),
        actuators={
            "trigger": ImplicitActuatorCfg(
                joint_names_expr=["trigger_joint"],
                effort_limit_sim=2.0,
                velocity_limit_sim=1.0,
                stiffness=3.2,
                damping=0.24,
                armature=0.015,
            ),
            "pusher": ImplicitActuatorCfg(
                joint_names_expr=["pusher_joint"],
                effort_limit_sim=10.0,
                velocity_limit_sim=0.03,
                stiffness=300.0,
                damping=25.0,
                armature=0.5,
            ),
        },
    )


def make_staple_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/SkinStaple",
    usd_path: str | Path | None = None,
):
    """Create a dynamic standalone formed-staple ``RigidObjectCfg``."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.assets import RigidObjectCfg  # type: ignore

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_path(usd_path, staple_usd()),
            semantic_tags=staple_semantic_tags(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.02)),
    )
