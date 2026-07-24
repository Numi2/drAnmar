"""Isaac Sim 5.1 / Isaac Lab 2.3.2 configuration factories."""
from __future__ import annotations

from pathlib import Path

from .paths import rigid_proxy_usd, staple_usd
from .semantics import semantic_tags, staple_semantic_tags
from .state import StaplerState, coerce_state, variant_selection


def _path(value: str | Path | None, fallback: Path) -> str:
    resolved = fallback if value is None else Path(value).expanduser()
    return str(resolved.resolve())


def make_rigid_proxy_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/SkinStapler",
    state: str | StaplerState = StaplerState.LOADED,
    usd_path: str | Path | None = None,
):
    """Create a v2.3.2 ``RigidObjectCfg`` with spawn-time state selection."""

    import isaaclab.sim as sim_utils  # type: ignore
    from isaaclab.assets import RigidObjectCfg  # type: ignore

    selected = coerce_state(state)
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_path(usd_path, rigid_proxy_usd()),
            variants=variant_selection(selected),
            semantic_tags=semantic_tags(selected),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=2.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
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
