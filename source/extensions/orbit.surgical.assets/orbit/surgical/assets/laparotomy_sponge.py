# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""DrAnmar laparotomy-sponge paths and lazy Isaac runtime integration.

The OpenUSD payload stays portable and simulator-version-neutral. Surface
deformable schemas are applied at runtime because Isaac Sim 5.1/Isaac Lab 2.3
supports the folded rigid proxy, while the selected Isaac Sim 6.0/Isaac Lab 3.0
lane supplies the current triangular surface-deformable API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CATALOG_SUBPATH = "Props/SurgicalCount/LaparotomySponge"
ASSET_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
LAPAROTOMY_SPONGE_ROOT = ASSET_DATA_ROOT / CATALOG_SUBPATH
LAPAROTOMY_SPONGE_UNFOLDED_USD = (
    LAPAROTOMY_SPONGE_ROOT / "lap_sponge_unfolded.usda"
)
LAPAROTOMY_SPONGE_FOLDED_PROXY_USD = (
    LAPAROTOMY_SPONGE_ROOT / "lap_sponge_folded_proxy.usda"
)

VALID_STATES = frozenset({"dry", "wet"})
SIMULATION_SURFACE_AREA_M2 = 0.21021902561187744


try:
    from i4h_asset_helper import BaseI4HAssets as _BaseI4HAssets
except ImportError:
    _BaseI4HAssets = object


class SurgicalCountAssets(_BaseI4HAssets):
    """I4H-compatible relative names for the DrAnmar surgical-count family."""

    LAPAROTOMY_SPONGE_UNFOLDED = (
        "Props/SurgicalCount/LaparotomySponge/lap_sponge_unfolded.usda"
    )
    LAPAROTOMY_SPONGE_FOLDED_PROXY = (
        "Props/SurgicalCount/LaparotomySponge/lap_sponge_folded_proxy.usda"
    )


@dataclass(frozen=True)
class SurfacePhysicsPreset:
    """Category-grounded, provisional surface-solver parameters."""

    mass_kg: float
    youngs_modulus_pa: float
    poissons_ratio: float
    surface_thickness_m: float
    density_kg_m3: float
    dynamic_friction: float
    elasticity_damping: float
    bend_damping: float
    surface_bend_stiffness: float = 0.0


DRY_SURFACE_PRESET = SurfacePhysicsPreset(
    mass_kg=0.022,
    youngs_modulus_pa=100_000.0,
    poissons_ratio=0.35,
    surface_thickness_m=0.004,
    density_kg_m3=26.163188531539117,
    dynamic_friction=0.65,
    elasticity_damping=0.10,
    bend_damping=0.10,
)

WET_SURFACE_PRESET = SurfacePhysicsPreset(
    mass_kg=0.120,
    youngs_modulus_pa=60_000.0,
    poissons_ratio=0.35,
    surface_thickness_m=0.004,
    density_kg_m3=142.70830108112247,
    dynamic_friction=0.55,
    elasticity_damping=0.18,
    bend_damping=0.18,
)

SURFACE_PRESETS = {
    "dry": DRY_SURFACE_PRESET,
    "wet": WET_SURFACE_PRESET,
}

DEFAULT_LABELS: dict[str, list[str]] = {
    "class": ["surgical_sponge", "laparotomy_sponge", "radiopaque_sponge"],
    "workflow": ["surgical_count", "handover", "retrieval", "disposal"],
}


def set_state_variant(stage, prim_path: str, state: str) -> None:
    """Select the coordinated visual and physical dry/wet state."""

    if state not in VALID_STATES:
        raise ValueError(f"Unsupported state {state!r}; expected one of {sorted(VALID_STATES)}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"No valid prim at {prim_path}")
    variant_set = prim.GetVariantSets().GetVariantSet("state")
    names = set(variant_set.GetVariantNames())
    if state not in names:
        raise ValueError(f"Prim {prim_path} has state variants {sorted(names)}, not {state!r}")
    if not variant_set.SetVariantSelection(state):
        raise RuntimeError(f"Failed to set state={state!r} on {prim_path}")


def make_rigid_proxy_cfg(
    prim_path: str = "/World/LaparotomySponge",
    *,
    state: str = "dry",
    position: tuple[float, float, float] = (0.0, 0.0, 0.35),
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    activate_contact_sensors: bool = False,
):
    """Return a lazy Isaac Lab ``RigidObjectCfg`` for the folded proxy."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    if state not in VALID_STATES:
        raise ValueError(f"Unsupported state {state!r}; expected one of {sorted(VALID_STATES)}")
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(LAPAROTOMY_SPONGE_FOLDED_PROXY_USD),
            activate_contact_sensors=activate_contact_sensors,
            variants={"state": state},
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=orientation_wxyz),
    )


def spawn_unfolded_reference(
    prim_path: str = "/World/LaparotomySponge",
    *,
    state: str = "dry",
    translation: tuple[float, float, float] = (0.0, 0.0, 0.50),
):
    """Reference the unfolded component through Isaac Lab's USD spawner."""

    import isaaclab.sim as sim_utils

    if state not in VALID_STATES:
        raise ValueError(f"Unsupported state {state!r}; expected one of {sorted(VALID_STATES)}")
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(LAPAROTOMY_SPONGE_UNFOLDED_USD),
        variants={"state": state},
    )
    return cfg.func(prim_path, cfg, translation=translation)


def make_surface_view_cfg(
    mesh_prim_path: str = "/World/LaparotomySponge/SimulationMesh",
):
    """Return an Isaac Lab surface-deformable view configuration."""

    from isaaclab.assets import DeformableObjectCfg

    return DeformableObjectCfg(
        prim_path=mesh_prim_path,
        spawn=None,
        init_state=DeformableObjectCfg.InitialStateCfg(),
    )


def create_surface_material(stage, material_path: str, preset: SurfacePhysicsPreset):
    """Author the selected Omni Physics/PhysX surface material."""

    from pxr import UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    prim = material.GetPrim()

    prim.ApplyAPI("OmniPhysicsBaseMaterialAPI")
    prim.GetAttribute("omniphysics:dynamicFriction").Set(preset.dynamic_friction)
    prim.GetAttribute("omniphysics:density").Set(preset.density_kg_m3)

    prim.ApplyAPI("OmniPhysicsDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:youngsModulus").Set(preset.youngs_modulus_pa)
    prim.GetAttribute("omniphysics:poissonsRatio").Set(preset.poissons_ratio)

    prim.ApplyAPI("OmniPhysicsSurfaceDeformableMaterialAPI")
    prim.GetAttribute("omniphysics:surfaceThickness").Set(preset.surface_thickness_m)
    # Zero selects the runtime's thickness-aware bend derivation. An explicit
    # membrane-only formula would overstate shell bending stiffness.
    prim.GetAttribute("omniphysics:surfaceBendStiffness").Set(
        preset.surface_bend_stiffness
    )

    prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(
        preset.elasticity_damping
    )
    prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(preset.bend_damping)
    return material


def apply_surface_deformable(
    stage,
    root_prim_path: str,
    *,
    state: str = "dry",
    material_path: str | None = None,
    self_collision: bool = True,
) -> dict[str, object]:
    """Cook the connected triangle mesh as a surface deformable at runtime."""

    if state not in SURFACE_PRESETS:
        raise ValueError(f"Unsupported state {state!r}; expected one of {sorted(SURFACE_PRESETS)}")

    from omni.physx.scripts import deformableUtils
    from pxr import UsdShade

    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        raise ValueError(f"No valid asset root at {root_prim_path}")
    mesh_path = f"{root_prim_path.rstrip('/')}/SimulationMesh"
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim or not mesh_prim.IsValid():
        raise ValueError(f"No SimulationMesh at {mesh_path}")

    set_state_variant(stage, root_prim_path, state)
    success = deformableUtils.set_physics_surface_deformable_body(
        stage, mesh_prim.GetPath()
    )
    if success is False:
        raise RuntimeError(f"PhysX failed to create a surface deformable at {mesh_path}")

    mesh_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    if mesh_prim.HasAPI("PhysxSurfaceDeformableBodyAPI"):
        mesh_prim.GetAttribute("physxDeformableBody:selfCollision").Set(
            bool(self_collision)
        )

    if material_path is None:
        material_path = (
            f"/World/Materials/LaparotomySponge{state.capitalize()}SurfaceMaterial"
        )
    material = create_surface_material(stage, material_path, SURFACE_PRESETS[state])
    binding = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

    return {
        "root_prim_path": root_prim_path,
        "mesh_prim_path": mesh_path,
        "material_path": material_path,
        "state": state,
        "self_collision": bool(self_collision),
        "preset": SURFACE_PRESETS[state],
    }


def _resolve_prim(prim_or_path: Any):
    if not isinstance(prim_or_path, str):
        return prim_or_path
    import omni.usd

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_or_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"No valid prim at {prim_or_path}")
    return prim


def add_labels(prim_or_path: Any, labels: Sequence[str], taxonomy: str = "class") -> str:
    """Apply semantic labels through the first available Isaac API generation."""

    clean = [str(label) for label in labels if str(label)]
    if not clean:
        raise ValueError("At least one non-empty label is required")

    try:
        import isaacsim.core.experimental.utils.semantics as semantics_utils
    except ImportError:
        semantics_utils = None
    if semantics_utils is not None and hasattr(semantics_utils, "add_labels"):
        try:
            semantics_utils.add_labels(
                prim_or_path, labels=clean, taxonomy=taxonomy
            )
            return "isaacsim.core.experimental.utils.semantics.add_labels"
        except (ImportError, ModuleNotFoundError):
            pass

    try:
        import isaacsim.core.utils.semantics as semantics_utils_compat
    except ImportError as exc:
        raise RuntimeError("Isaac Sim semantic utilities are not available") from exc

    prim = _resolve_prim(prim_or_path)
    if hasattr(semantics_utils_compat, "add_labels"):
        try:
            semantics_utils_compat.add_labels(
                prim, clean, instance_name=taxonomy, overwrite=True
            )
            return "isaacsim.core.utils.semantics.add_labels"
        except (ImportError, ModuleNotFoundError):
            pass

    if hasattr(semantics_utils_compat, "add_update_semantics"):
        for index, label in enumerate(clean):
            suffix = "" if index == 0 else f"_{index:02d}"
            semantics_utils_compat.add_update_semantics(
                prim,
                semantic_label=label,
                type_label=taxonomy,
                suffix=suffix,
            )
        return "isaacsim.core.utils.semantics.add_update_semantics"

    raise RuntimeError("No supported Isaac Sim semantic labeling API was found")


def apply_default_labels(
    prim_or_path: Any,
    labels: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """Apply class and workflow taxonomies."""

    selected = DEFAULT_LABELS if labels is None else labels
    return {
        taxonomy: add_labels(prim_or_path, values, taxonomy)
        for taxonomy, values in selected.items()
    }


__all__ = [
    "CATALOG_SUBPATH",
    "DRY_SURFACE_PRESET",
    "LAPAROTOMY_SPONGE_FOLDED_PROXY_USD",
    "LAPAROTOMY_SPONGE_ROOT",
    "LAPAROTOMY_SPONGE_UNFOLDED_USD",
    "SIMULATION_SURFACE_AREA_M2",
    "SURFACE_PRESETS",
    "SurgicalCountAssets",
    "WET_SURFACE_PRESET",
    "apply_default_labels",
    "apply_surface_deformable",
    "make_rigid_proxy_cfg",
    "make_surface_view_cfg",
    "set_state_variant",
    "spawn_unfolded_reference",
]
