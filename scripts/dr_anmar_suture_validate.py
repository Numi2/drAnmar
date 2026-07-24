#!/usr/bin/env python3
"""Deterministically validate the Dr.Anmar 4-0 suture contract and USD."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from dr_anmar_needle_model import (
    DEFAULT_MASS_PROPERTY_INTEGRATION_SLICES,
    DEFAULT_NEEDLE_PROFILE_PATH,
    build_needle_collision_capsules,
    build_needle_mesh,
    derive_needle,
    derive_needle_mass_properties,
    load_needle_profile,
    needle_mesh_collision_coverage,
    needle_mesh_normal_quality,
    reconstruct_inertia_tensor,
    sample_episode_parameters,
)
from dr_anmar_procedures import PROCEDURE_ROOMS
from dr_anmar_suture_integration import (
    DR_ANMAR_NEEDLE_ASSET_ID,
    DR_ANMAR_NEEDLE_ASSET_PATH,
    DR_ANMAR_NEEDLE_ASSET_VERSION,
    DR_ANMAR_NEEDLE_BASE_ASSET_PATH,
    DR_ANMAR_NEEDLE_GEOMETRY_ASSET_PATH,
    DR_ANMAR_NEEDLE_MATERIALS_ASSET_PATH,
    DR_ANMAR_NEEDLE_NAME,
    DR_ANMAR_NEEDLE_PHYSICS_ASSET_PATH,
    DR_ANMAR_NEEDLE_PHYSX_ASSET_PATH,
    DR_ANMAR_NEEDLE_ROOT_PRIM,
    SUTURE_ASSET_PATH,
    SUTURE_BASE_ASSET_PATH,
    SUTURE_GEOMETRY_ASSET_PATH,
    SUTURE_MATERIALS_ASSET_PATH,
    SUTURE_PHYSICS_ASSET_PATH,
    SUTURE_PHYSX_ASSET_PATH,
    configure_dr_anmar_needle,
    local_room_ids,
    needle_mass_properties_for_mass,
)
from dr_anmar_suture_model import (
    DEFAULT_PROFILE_PATH,
    crush_strength_fraction,
    derive,
    effective_failure_load,
    load_profile,
    monotonic_tension_force,
    sample_suture_runtime_profile,
    self_friction_coefficient,
    stress_retention,
)
from dr_anmar_suture_runtime import SutureRuntime

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET = SUTURE_ASSET_PATH
DEFAULT_SUTURE_BASE = SUTURE_BASE_ASSET_PATH
DEFAULT_SUTURE_GEOMETRY = SUTURE_GEOMETRY_ASSET_PATH
DEFAULT_SUTURE_MATERIALS = SUTURE_MATERIALS_ASSET_PATH
DEFAULT_SUTURE_PHYSICS = SUTURE_PHYSICS_ASSET_PATH
DEFAULT_SUTURE_PHYSX = SUTURE_PHYSX_ASSET_PATH
DEFAULT_NEEDLE = DR_ANMAR_NEEDLE_ASSET_PATH
DEFAULT_NEEDLE_BASE = DR_ANMAR_NEEDLE_BASE_ASSET_PATH
DEFAULT_NEEDLE_GEOMETRY = DR_ANMAR_NEEDLE_GEOMETRY_ASSET_PATH
DEFAULT_NEEDLE_MATERIALS = DR_ANMAR_NEEDLE_MATERIALS_ASSET_PATH
DEFAULT_NEEDLE_PHYSICS = DR_ANMAR_NEEDLE_PHYSICS_ASSET_PATH
DEFAULT_NEEDLE_PHYSX = DR_ANMAR_NEEDLE_PHYSX_ASSET_PATH
DEFAULT_WORKSTATION = REPOSITORY_ROOT / "scripts/dr_anmar_workstation.py"
DEFAULT_NATIVE_PROBE = REPOSITORY_ROOT / "scripts/dr_anmar_suture_physics_probe.py"
DEFAULT_INTEGRATION = REPOSITORY_ROOT / "scripts/dr_anmar_suture_integration.py"


def check(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    measured: Any,
    expected: Any,
) -> None:
    checks[name] = {
        "passed": bool(passed),
        "measured": measured,
        "expected": expected,
    }


def missing_text_tokens(text: str, tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in text]


def present_text_tokens(texts: tuple[str, ...], tokens: tuple[str, ...]) -> list[str]:
    return [token for token in tokens if any(token in text for text in texts)]


def read_usd_as_text(path: Path, usdcat_command: str) -> str:
    usdcat_path = shutil.which(usdcat_command)
    if usdcat_path is None:
        raise RuntimeError(f"OpenUSD usdcat is required to validate binary geometry: {usdcat_command}")
    return subprocess.run(
        [usdcat_path, str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def compose_physics_variant(
    asset_path: Path,
    selection: str,
    usdcat_command: str,
) -> str:
    """Flatten one public Physics selection through an external referencing stage."""

    usdcat_path = shutil.which(usdcat_command)
    if usdcat_path is None:
        raise RuntimeError(f"OpenUSD usdcat is required to compose variants: {usdcat_command}")
    asset_reference = asset_path.expanduser().resolve().as_posix()
    wrapper_text = f"""#usda 1.0
(
    defaultPrim = "Probe"
)

def Xform "Probe" (
    prepend references = @{asset_reference}@
    variants = {{
        string Physics = "{selection}"
    }}
)
{{
}}
"""
    with tempfile.TemporaryDirectory(prefix=".dr_anmar_variant_", dir=asset_path.parent) as temporary_directory:
        wrapper = Path(temporary_directory) / f"{selection}.usda"
        wrapper.write_text(wrapper_text, encoding="utf-8")
        return subprocess.run(
            [usdcat_path, "--flatten", str(wrapper)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout


def add_suture_layer_checks(
    checks: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    *,
    segment_count: int,
    entry_text: str,
    base_text: str,
    geometry_text: str,
    geometry_is_usdc: bool,
    materials_text: str,
    physics_text: str,
    physx_text: str,
) -> None:
    """Validate source ownership and scale-aware PhysX tuning for the suture layers."""

    layer_contract = profile["asset_structure"]
    entry_engine_properties = sorted(
        set(re.findall(r"\b(?:physics:|physx[A-Za-z]*:|newton:)[A-Za-z][A-Za-z0-9_]*", entry_text))
    )
    entry_engine_schemas = sorted(set(re.findall(r'"((?:Physics|Physx|Newton)[A-Za-z0-9_]*API)"', entry_text)))
    geometry_forbidden = present_text_tokens(
        (geometry_text,),
        (
            "apiSchemas",
            "material:binding",
            'def Material "',
            'def Shader "',
            "physics:",
            "physx",
            "newton:",
        ),
    )
    materials_forbidden = present_text_tokens(
        (materials_text,),
        (
            'def Capsule "',
            "float height",
            "float radius",
            "physics:",
            "Physx",
            "physx",
            "Newton",
            "newton:",
        ),
    )
    neutral_engine_specific = present_text_tokens(
        (physics_text,),
        (
            '"Physx',
            "physx",
            '"Newton',
            "newton:",
        ),
    )
    physx_neutral_or_newton = present_text_tokens(
        (physx_text,),
        (
            "physics:collisionEnabled",
            "physics:rigidBodyEnabled",
            "physics:mass",
            "physics:body0",
            "physics:breakForce",
            '"PhysicsRigidBodyAPI"',
            '"PhysicsCollisionAPI"',
            '"Newton',
            "newton:",
        ),
    )
    entry_forbidden = present_text_tokens(
        (entry_text,),
        (
            'def Capsule "',
            'def Material "',
            'def Shader "',
            "material:binding",
            "physics:",
            "newton:",
        ),
    )
    base_forbidden = present_text_tokens(
        (base_text,),
        (
            'def Capsule "',
            'def Material "',
            'def Shader "',
            "material:binding",
            "physics:",
            "newton:",
        ),
    )
    base_engine_schemas = sorted(set(re.findall(r'"((?:Physics|Physx|Newton)[A-Za-z0-9_]*API)"', base_text)))
    base_physics_typed_prims = sorted(set(re.findall(r"\bdef\s+(Physics[A-Za-z0-9_]+)\s+\"", base_text)))
    nvidia_references = profile.get("nvidia_stack_references", [])
    check(
        checks,
        "suture_asset_structure_source_ownership",
        profile["version"] == "2.1.0"
        and layer_contract["entry_layer"] == "DrAnmarSuture4_0.usda"
        and layer_contract["base_layer"] == "DrAnmarSuture4_0_base.usda"
        and layer_contract["geometry_layer"] == "DrAnmarSuture4_0_geometry.usd"
        and layer_contract["geometry_format"] == "usdc"
        and layer_contract["materials_layer"] == "DrAnmarSuture4_0_materials.usda"
        and layer_contract["physics_layer"] == "DrAnmarSuture4_0_physics.usda"
        and layer_contract["physx_layer"] == "DrAnmarSuture4_0_physx.usda"
        and layer_contract["composition"]
        == "entry_references_base_and_public_Physics_variant_payloads_none_physics_or_physx"
        and layer_contract["variant_set"] == "Physics"
        and layer_contract["variant_choices"] == ["none", "physics", "physx"]
        and layer_contract["default_runtime"] == "physx"
        and layer_contract["physics_payload_loading"] == "deferred_until_selected_variant_is_loaded"
        and layer_contract["engine_isolation"]
        == "neutral_layer_contains_no_physx_or_newton_opinions_and_physx_layer_contains_no_newton_opinions"
        and f'@{layer_contract["base_layer"]}@' in entry_text
        and f'@{layer_contract["physx_layer"]}@' in entry_text
        and f'@{layer_contract["physics_layer"]}@' in entry_text
        and f'@{layer_contract["materials_layer"]}@' not in entry_text
        and f'@{layer_contract["geometry_layer"]}@' not in entry_text
        and f'@{layer_contract["materials_layer"]}@' in base_text
        and f'@{layer_contract["geometry_layer"]}@' in base_text
        and f'@{layer_contract["physics_layer"]}@' in physx_text
        and 'append variantSets = "Physics"' in entry_text
        and 'string Physics = "physx"' in entry_text
        and entry_text.count('"none" {') == 1
        and entry_text.count('"physics" (') == 1
        and entry_text.count('"physx" (') == 1
        and entry_text.count("prepend payload =") == 2
        and len(entry_text.encode("utf-8")) <= int(layer_contract["entry_layer_max_bytes"])
        and geometry_is_usdc
        and not entry_engine_properties
        and not entry_engine_schemas
        and not entry_forbidden
        and not base_forbidden
        and not base_engine_schemas
        and not base_physics_typed_prims
        and not geometry_forbidden
        and not materials_forbidden
        and not neutral_engine_specific
        and not physx_neutral_or_newton
        and len(re.findall(r'def Capsule "S\d{4}"', geometry_text)) == segment_count
        and geometry_text.count('def Capsule "NeedleInterface"') == 1
        and materials_text.count('def Material "') == 2
        and materials_text.count('def Shader "PreviewSurface"') == 2
        and materials_text.count("rel material:binding =") == 2
        and physics_text.count('"PhysicsRigidBodyAPI"') == segment_count + 1
        and physics_text.count('"PhysicsCollisionAPI"') == segment_count + 1
        and len(re.findall(r'def PhysicsJoint "J\d{4}"', physics_text)) == segment_count
        and physx_text.count('"PhysxRigidBodyAPI", "PhysxCollisionAPI"') == segment_count + 1
        and physx_text.count("physxRigidBody:enableCCD") == segment_count + 1
        and physx_text.count("physxRigidBody:enableSpeculativeCCD") == segment_count + 1
        and physx_text.count("physxCollision:contactOffset") == segment_count + 1
        and physx_text.count("physxCollision:restOffset") == segment_count + 1
        and physx_text.count('"PhysxMaterialAPI"') == 2
        and len(nvidia_references) >= 5
        and all(
            item.get("url", "").startswith(
                (
                    "https://docs.omniverse.nvidia.com/",
                    "https://docs.isaacsim.omniverse.nvidia.com/",
                )
            )
            and item.get("used_for")
            for item in nvidia_references
        ),
        {
            "contract": layer_contract,
            "entry_bytes": len(entry_text.encode("utf-8")),
            "geometry_is_usdc": geometry_is_usdc,
            "entry_engine_properties": entry_engine_properties,
            "entry_engine_schemas": entry_engine_schemas,
            "entry_forbidden": entry_forbidden,
            "base_forbidden": base_forbidden,
            "base_engine_schemas": base_engine_schemas,
            "base_physics_typed_prims": base_physics_typed_prims,
            "geometry_forbidden": geometry_forbidden,
            "materials_forbidden": materials_forbidden,
            "neutral_engine_specific": neutral_engine_specific,
            "physx_neutral_or_newton": physx_neutral_or_newton,
            "nvidia_reference_count": len(nvidia_references),
        },
        "lightweight entry, binary capsule geometry, visual look-development, neutral mechanics, and PhysX tuning"
        " have isolated deterministic source ownership",
    )

    offset_contract = profile["contact"]["contact_offsets"]
    contact_offsets: list[float] = []
    rest_offsets: list[float] = []
    attribute_errors: list[str] = []
    body_names = ["NeedleInterface", *(f"S{index:04d}" for index in range(segment_count))]
    for body_name in body_names:
        geometry_match = re.search(
            rf'def Capsule "{body_name}".*?\n\s*\}}',
            geometry_text,
            flags=re.DOTALL,
        )
        physx_match = re.search(
            rf'over "{body_name}".*?\n\s*\}}',
            physx_text,
            flags=re.DOTALL,
        )
        if geometry_match is None or physx_match is None:
            attribute_errors.append(f"{body_name}:missing_geometry_or_physx_block")
            continue
        geometry_block = geometry_match.group(0)
        physx_block = physx_match.group(0)
        radius_match = re.search(r"float radius = ([0-9.eE+-]+)", geometry_block)
        contact_match = re.search(r"float physxCollision:contactOffset = ([0-9.eE+-]+)", physx_block)
        rest_match = re.search(r"float physxCollision:restOffset = ([0-9.eE+-]+)", physx_block)
        if radius_match is None or contact_match is None or rest_match is None:
            attribute_errors.append(f"{body_name}:missing_radius_or_offset")
            continue
        radius = float(radius_match.group(1))
        contact_offset = float(contact_match.group(1))
        rest_offset = float(rest_match.group(1))
        expected_contact_offset = max(
            float(offset_contract["minimum_m"]),
            min(
                float(offset_contract["maximum_m"]),
                radius * float(offset_contract["collision_radius_fraction"]),
            ),
        )
        if not math.isclose(
            contact_offset,
            expected_contact_offset,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            attribute_errors.append(f"{body_name}:contact_offset")
        if not math.isclose(
            rest_offset,
            float(offset_contract["rest_offset_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            attribute_errors.append(f"{body_name}:rest_offset")
        for token in (
            "bool physxRigidBody:enableCCD = true",
            "bool physxRigidBody:enableSpeculativeCCD = true",
            f'int physxRigidBody:solverPositionIterationCount = {int(profile["solver"]["position_iterations"])}',
            f'int physxRigidBody:solverVelocityIterationCount = {int(profile["solver"]["velocity_iterations"])}',
        ):
            if token not in physx_block:
                attribute_errors.append(f"{body_name}:{token.split(':')[-1].split()[0]}")
        contact_offsets.append(contact_offset)
        rest_offsets.append(rest_offset)
    check(
        checks,
        "suture_hybrid_ccd_and_scale_aware_contact",
        profile["contact"]["sweep_ccd"] is True
        and profile["contact"]["speculative_ccd"] is True
        and profile["contact"]["ccd_mode"] == "hybrid_linear_and_angular"
        and profile["contact"]["combine_mode"] == "max"
        and offset_contract["policy"] == "clamped_fraction_of_authored_capsule_radius"
        and offset_contract["engine_layer"] == "DrAnmarSuture4_0_physx.usda"
        and offset_contract["neutral_layer_policy"] == "no_engine_specific_contact_schema"
        and not attribute_errors
        and len(contact_offsets) == segment_count + 1
        and all(
            0.0 <= rest_offset < contact_offset
            for rest_offset, contact_offset in zip(
                rest_offsets,
                contact_offsets,
                strict=True,
            )
        )
        and min(contact_offsets) >= float(offset_contract["minimum_m"])
        and max(contact_offsets) <= float(offset_contract["maximum_m"])
        and physx_text.count('physxMaterial:frictionCombineMode = "max"') == 2,
        {
            "attribute_errors": attribute_errors,
            "body_count": len(contact_offsets),
            "contact_offset_range_m": [
                min(contact_offsets) if contact_offsets else None,
                max(contact_offsets) if contact_offsets else None,
            ],
            "rest_offset_range_m": [
                min(rest_offsets) if rest_offsets else None,
                max(rest_offsets) if rest_offsets else None,
            ],
            "contract": offset_contract,
        },
        "hybrid linear and angular CCD with bounded radius-scaled PhysX contact offsets on every suture body",
    )


def add_physics_variant_checks(
    checks: dict[str, dict[str, Any]],
    *,
    suture_variants: dict[str, str],
    needle_variants: dict[str, str],
    segment_count: int,
    needle_collision_count: int,
) -> None:
    """Validate all public Physics choices after full OpenUSD composition."""

    def metrics(text: str) -> dict[str, int]:
        return {
            "physics_api_schemas": len(re.findall(r'"Physics[A-Za-z0-9_:]*API(?::[A-Za-z0-9_]+)?"', text)),
            "physx_api_schemas": len(re.findall(r'"Physx[A-Za-z0-9_:]*API(?::[A-Za-z0-9_]+)?"', text)),
            "physics_properties": len(re.findall(r"\bphysics:[A-Za-z][A-Za-z0-9_]*", text)),
            "physx_properties": len(re.findall(r"\bphysx[A-Za-z]*:[A-Za-z][A-Za-z0-9_]*", text)),
            "suture_capsules": len(re.findall(r'def Capsule "S\d{4}"', text)),
            "suture_joints": len(re.findall(r'def PhysicsJoint "J\d{4}"', text)),
            "needle_colliders": len(re.findall(r'def Capsule "C\d{3}"', text)),
            "factory_swage_joints": text.count('def PhysicsFixedJoint "FactorySwage"'),
        }

    suture_metrics = {selection: metrics(text) for selection, text in suture_variants.items()}
    needle_metrics = {selection: metrics(text) for selection, text in needle_variants.items()}
    check(
        checks,
        "suture_public_physics_variants_compose",
        set(suture_variants) == {"none", "physics", "physx"}
        and suture_metrics["none"]["suture_capsules"] == segment_count
        and suture_metrics["none"]["physics_api_schemas"] == 0
        and suture_metrics["none"]["physics_properties"] == 0
        and suture_metrics["none"]["physx_api_schemas"] == 0
        and suture_metrics["none"]["physx_properties"] == 0
        and suture_metrics["none"]["suture_joints"] == 0
        and suture_metrics["physics"]["suture_capsules"] == segment_count
        and suture_metrics["physics"]["physics_api_schemas"] > 0
        and suture_metrics["physics"]["physics_properties"] > 0
        and suture_metrics["physics"]["physx_api_schemas"] == 0
        and suture_metrics["physics"]["physx_properties"] == 0
        and suture_metrics["physics"]["suture_joints"] == segment_count
        and suture_metrics["physx"]["suture_capsules"] == segment_count
        and suture_metrics["physx"]["physics_api_schemas"] > 0
        and suture_metrics["physx"]["physx_api_schemas"] > 0
        and suture_metrics["physx"]["physics_properties"] > 0
        and suture_metrics["physx"]["physx_properties"] > 0
        and suture_metrics["physx"]["suture_joints"] == segment_count,
        suture_metrics,
        "none keeps renderable geometry only, physics adds engine-neutral mechanics, and physx adds PhysX tuning",
    )
    check(
        checks,
        "needle_public_physics_variants_compose_and_synchronize_suture",
        set(needle_variants) == {"none", "physics", "physx"}
        and needle_metrics["none"]["suture_capsules"] == segment_count
        and needle_metrics["none"]["needle_colliders"] == 0
        and needle_metrics["none"]["physics_api_schemas"] == 0
        and needle_metrics["none"]["physics_properties"] == 0
        and needle_metrics["none"]["physx_api_schemas"] == 0
        and needle_metrics["none"]["physx_properties"] == 0
        and needle_metrics["none"]["suture_joints"] == 0
        and needle_metrics["none"]["factory_swage_joints"] == 0
        and needle_metrics["physics"]["suture_capsules"] == segment_count
        and needle_metrics["physics"]["needle_colliders"] == needle_collision_count
        and needle_metrics["physics"]["physics_api_schemas"] > 0
        and needle_metrics["physics"]["physics_properties"] > 0
        and needle_metrics["physics"]["physx_api_schemas"] == 0
        and needle_metrics["physics"]["physx_properties"] == 0
        and needle_metrics["physics"]["suture_joints"] == segment_count
        and needle_metrics["physics"]["factory_swage_joints"] == 1
        and needle_metrics["physx"]["suture_capsules"] == segment_count
        and needle_metrics["physx"]["needle_colliders"] == needle_collision_count
        and needle_metrics["physx"]["physics_api_schemas"] > 0
        and needle_metrics["physx"]["physics_properties"] > 0
        and needle_metrics["physx"]["physx_api_schemas"] > 0
        and needle_metrics["physx"]["physx_properties"] > 0
        and needle_metrics["physx"]["suture_joints"] == segment_count
        and needle_metrics["physx"]["factory_swage_joints"] == 1,
        needle_metrics,
        "the assembly choice switches needle and nested suture together without mixed physics backends",
    )


def validate(
    profile: dict[str, Any],
    needle_profile: dict[str, Any],
    suture_entry_text: str,
    suture_base_text: str,
    suture_geometry_text: str,
    suture_geometry_is_usdc: bool,
    suture_materials_text: str,
    suture_physics_text: str,
    suture_physx_text: str,
    suture_variant_texts: dict[str, str],
    needle_entry_text: str,
    needle_base_text: str,
    needle_geometry_text: str,
    needle_geometry_is_usdc: bool,
    needle_materials_text: str,
    needle_physics_text: str,
    needle_physx_text: str,
    needle_variant_texts: dict[str, str],
    workstation_text: str,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    derived = derive(profile)
    derived_needle = derive_needle(needle_profile)
    needle_mesh = build_needle_mesh(needle_profile)
    needle_normal_quality = needle_mesh_normal_quality(
        needle_profile,
        needle_mesh,
    )
    native_probe_text = DEFAULT_NATIVE_PROBE.read_text(encoding="utf-8")
    integration_text = DEFAULT_INTEGRATION.read_text(encoding="utf-8")
    asset_text = "\n".join(
        (
            suture_entry_text,
            suture_base_text,
            suture_geometry_text,
            suture_materials_text,
            suture_physics_text,
            suture_physx_text,
        )
    )
    needle_text = "\n".join(
        (
            needle_entry_text,
            needle_base_text,
            needle_geometry_text,
            needle_materials_text,
            needle_physics_text,
            needle_physx_text,
        )
    )
    geometry = profile["geometry"]
    material = profile["material"]
    tension = profile["tension"]
    diameter_range = [float(value) for value in geometry["diameter_range_m"]]
    check(
        checks,
        "true_4_0_diameter",
        diameter_range[0] <= derived.diameter_m <= diameter_range[1],
        derived.diameter_m,
        diameter_range,
    )
    density_range = [float(value) for value in material["density_range_kg_m3"]]
    check(
        checks,
        "polymer_density",
        density_range[0] <= float(material["density_kg_m3"]) <= density_range[1],
        float(material["density_kg_m3"]),
        density_range,
    )
    failure_range = [float(value) for value in tension["straight_failure_range_n"]]
    check(
        checks,
        "straight_failure_load",
        failure_range[0] <= derived.straight_failure_load_n <= failure_range[1],
        derived.straight_failure_load_n,
        failure_range,
    )
    failure_strain_range = [float(value) for value in tension["failure_strain_range"]]
    check(
        checks,
        "elongation_at_break",
        failure_strain_range[0] <= float(tension["failure_strain"]) <= failure_strain_range[1],
        float(tension["failure_strain"]),
        failure_strain_range,
    )
    knot_range = [float(value) for value in profile["knot"]["strength_efficiency_range"]]
    knot_efficiency = derived.knot_failure_load_n / derived.straight_failure_load_n
    check(
        checks,
        "knot_strength_reduction",
        knot_range[0] <= knot_efficiency <= knot_range[1],
        knot_efficiency,
        knot_range,
    )
    force_yield, failed_yield = monotonic_tension_force(profile, float(tension["yield_strain"]))
    force_break, failed_break = monotonic_tension_force(profile, float(tension["failure_strain"]))
    check(
        checks,
        "nonlinear_tension_curve",
        0.0 < force_yield < force_break <= derived.straight_failure_load_n and not failed_yield and failed_break,
        {"yield_force_n": force_yield, "break_force_n": force_break},
        "positive yield force below a 20-25 N failed endpoint",
    )
    retained_2h = stress_retention(profile, 7200.0)
    retained_long = stress_retention(profile, 1e9)
    check(
        checks,
        "wet_stress_relaxation",
        retained_long <= retained_2h < 1.0
        and math.isclose(
            retained_long,
            float(profile["viscoelasticity"]["retained_stress_asymptote"]),
            abs_tol=1e-6,
        ),
        {"two_hours": retained_2h, "long_time": retained_long},
        "largest early loss with configured asymptote",
    )
    friction_low = self_friction_coefficient(profile, 0.01)
    friction_high = self_friction_coefficient(profile, 2.0)
    check(
        checks,
        "load_dependent_self_friction",
        0.0 < friction_high < friction_low < 1.0,
        {"mu_at_0_01_n": friction_low, "mu_at_2_n": friction_high},
        "positive coefficient decreasing with load while friction force rises",
    )
    check(
        checks,
        "instrument_crush_damage",
        crush_strength_fraction(profile, 0) == 1.0
        and math.isclose(crush_strength_fraction(profile, 1), 0.661)
        and math.isclose(crush_strength_fraction(profile, 5), 0.383),
        {
            "zero_grasps": crush_strength_fraction(profile, 0),
            "one_grasp": crush_strength_fraction(profile, 1),
            "five_grasps": crush_strength_fraction(profile, 5),
        },
        {"zero_grasps": 1.0, "one_grasp": 0.661, "five_grasps": 0.383},
    )
    check(
        checks,
        "combined_knot_and_crush_failure",
        effective_failure_load(profile, knotted=True, grasp_count=1) < derived.knot_failure_load_n,
        effective_failure_load(profile, knotted=True, grasp_count=1),
        f"less than {derived.knot_failure_load_n}",
    )
    segment_defs = len(re.findall(r'def Capsule "S\d{4}"', asset_text))
    joint_defs = len(re.findall(r'def PhysicsJoint "J\d{4}"', asset_text))
    check(
        checks,
        "physical_segment_resolution",
        segment_defs == derived.segment_count,
        segment_defs,
        derived.segment_count,
    )
    check(
        checks,
        "breakable_joint_resolution",
        joint_defs == derived.segment_count and asset_text.count("physics:breakForce") == derived.segment_count,
        {
            "joint_defs": joint_defs,
            "break_force_attributes": asset_text.count("physics:breakForce"),
        },
        derived.segment_count,
    )
    add_suture_layer_checks(
        checks,
        profile,
        segment_count=derived.segment_count,
        entry_text=suture_entry_text,
        base_text=suture_base_text,
        geometry_text=suture_geometry_text,
        geometry_is_usdc=suture_geometry_is_usdc,
        materials_text=suture_materials_text,
        physics_text=suture_physics_text,
        physx_text=suture_physx_text,
    )
    add_physics_variant_checks(
        checks,
        suture_variants=suture_variant_texts,
        needle_variants=needle_variant_texts,
        segment_count=derived.segment_count,
        needle_collision_count=derived_needle.collision_capsule_count,
    )
    required_asset_tokens = [
        "PhysicsRigidBodyAPI",
        "PhysicsCollisionAPI",
        "PhysicsDriveAPI:transX",
        "PhysicsDriveAPI:rotY",
        "PhysicsDriveAPI:rotZ",
        "PhysicsFilteredPairsAPI",
        "physxRigidBody:enableCCD",
        'def Capsule "NeedleInterface"',
        "drAnmar:swageFraction",
    ]
    missing_tokens = [token for token in required_asset_tokens if token not in asset_text]
    check(
        checks,
        "runtime_physics_contract",
        not missing_tokens,
        {"missing": missing_tokens},
        required_asset_tokens,
    )
    forbidden_tokens = [
        "Rope.usd",
        "SoftMimicGen",
        "nvidia-strand-ring-threading",
    ]
    present_forbidden = [token for token in forbidden_tokens if token in asset_text]
    check(
        checks,
        "independent_from_current_thread",
        not present_forbidden,
        {"forbidden_references": present_forbidden},
        "no current-thread reference",
    )
    check(
        checks,
        "actual_scale_not_visibility_inflated",
        "drAnmarVisibilityScale" not in asset_text and math.isclose(derived.diameter_m, 0.00025),
        derived.diameter_m,
        0.00025,
    )
    needle_identity_tokens = [
        f'defaultPrim = "{DR_ANMAR_NEEDLE_ROOT_PRIM}"',
        f'drAnmarAssetId = "{DR_ANMAR_NEEDLE_ASSET_ID}"',
        f'drAnmarAssetName = "{DR_ANMAR_NEEDLE_NAME}"',
        f'drAnmarAssetVersion = "{DR_ANMAR_NEEDLE_ASSET_VERSION}"',
        "@DrAnmarNeedle_base.usda@",
        "@DrAnmarNeedle_physx.usda@",
        "@DrAnmarNeedle_materials.usda@",
        "@DrAnmarNeedle_geometry.usd@",
        "prepend references = @../suture/DrAnmarSuture4_0.usda@",
        'drAnmarGeometrySource = "independently_generated_parametric_geometry"',
        f"drAnmarMassPropertyIntegrationSlices = {DEFAULT_MASS_PROPERTY_INTEGRATION_SLICES}",
        'drAnmarContactOffsetContract = "scale_aware_physx_engine_layer_authoring"',
        'drAnmarNormalContract = "analytic_taper_and_curvature_aware_indexed_face_varying_primvar"',
        'drAnmarRenderCollisionContract = "separate_visual_mesh_and_guide_purpose_invisible_compound_colliders"',
        'drAnmarMaterialContract = "top_level_looks_with_separate_visual_and_physics_materials"',
        'drAnmarLayerContract = "interface_references_base_with_public_none_physics_physx_payload_variants"',
        'drAnmarRepresentation = "high_resolution_mesh_with_compound_capsule_collision"',
        'drAnmarCollisionContract = "curvature_sagitta_bounded_capsules_with_explicit_extents"',
        '"PhysicsMaterialAPI"',
        '"PhysxMaterialAPI"',
        'physxMaterial:frictionCombineMode = "max"',
        "drAnmarResetRandomizationCount = 4",
        "drAnmarSimToRealGapCount = 7",
        'def PhysicsFixedJoint "FactorySwage"',
        'def Mesh "Visual"',
        'def Xform "Needle"',
        "point3f physics:centerOfMass",
        "float3 physics:diagonalInertia",
        "quatf physics:principalAxes",
        f"physics:body0 = </{DR_ANMAR_NEEDLE_ROOT_PRIM}/Needle>",
        f"physics:body1 = </{DR_ANMAR_NEEDLE_ROOT_PRIM}/Suture/NeedleInterface>",
        "physics:kinematicEnabled = false",
        'drAnmarAuthorship = "Independent Dr.Anmar geometry, collision, instrument composition and suture physics"',
    ]
    missing_identity_tokens = [token for token in needle_identity_tokens if token not in needle_text]
    check(
        checks,
        "dr_anmar_needle_identity_and_provenance",
        not missing_identity_tokens and needle_profile["version"] == DR_ANMAR_NEEDLE_ASSET_VERSION,
        {
            "missing": missing_identity_tokens,
            "profile_version": needle_profile["version"],
            "integration_version": DR_ANMAR_NEEDLE_ASSET_VERSION,
        },
        needle_identity_tokens,
    )
    layer_organization = needle_profile["construction"]["layer_organization"]
    engine_property_pattern = r"\b(?:physics:|physx[A-Za-z]*:|newton:)[A-Za-z][A-Za-z0-9_]*"
    engine_schema_pattern = r'"((?:Physics|Physx|Newton)[A-Za-z0-9_]*API)"'
    entry_physics_properties = sorted(set(re.findall(engine_property_pattern, needle_entry_text)))
    entry_physics_schemas = sorted(set(re.findall(engine_schema_pattern, needle_entry_text)))
    entry_physics_typed_prims = sorted(set(re.findall(r"\bdef\s+(Physics[A-Za-z0-9_]+)\s+\"", needle_entry_text)))
    entry_forbidden_content_tokens = present_text_tokens(
        (needle_entry_text,),
        (
            'def Mesh "Visual"',
            "point3f[] points",
            "faceVertexIndices",
            "primvars:normals",
            'def Material "NeedleSteelVisual"',
            'def Shader "PreviewSurface"',
            "material:binding",
        ),
    )
    base_physics_properties = sorted(set(re.findall(engine_property_pattern, needle_base_text)))
    base_physics_schemas = sorted(set(re.findall(engine_schema_pattern, needle_base_text)))
    base_physics_typed_prims = sorted(set(re.findall(r"\bdef\s+(Physics[A-Za-z0-9_]+)\s+\"", needle_base_text)))
    base_forbidden_content_tokens = present_text_tokens(
        (needle_base_text,),
        (
            'def Mesh "Visual"',
            "point3f[] points",
            "faceVertexIndices",
            "primvars:normals",
            'def Material "NeedleSteelVisual"',
            'def Shader "PreviewSurface"',
            "material:binding",
        ),
    )
    geometry_required_tokens = [
        'defaultPrim = "DrAnmarNeedle"',
        'over "DrAnmarNeedle"',
        'over "Needle"',
        'def Mesh "Visual"',
        "float3[] extent",
        "int[] faceVertexCounts",
        "int[] faceVertexIndices",
        "point3f[] points",
        "normal3f[] primvars:normals",
        "int[] primvars:normals:indices",
        'uniform token subdivisionScheme = "none"',
    ]
    materials_required_tokens = [
        'defaultPrim = "DrAnmarNeedle"',
        'over "DrAnmarNeedle"',
        'def Scope "Looks"',
        'def Material "NeedleSteelVisual"',
        'def Shader "PreviewSurface"',
        'uniform token info:id = "UsdPreviewSurface"',
        'over "Visual" (',
        '"MaterialBindingAPI"',
        "rel material:binding",
    ]
    missing_geometry_layer_tokens = missing_text_tokens(needle_geometry_text, geometry_required_tokens)
    missing_materials_layer_tokens = missing_text_tokens(needle_materials_text, materials_required_tokens)
    geometry_forbidden_content_tokens = present_text_tokens(
        (needle_geometry_text,),
        (
            "apiSchemas",
            "material:binding",
            'def Material "',
            'def Shader "',
            "physics:",
            "physx",
            "newton:",
            "prepend references =",
        ),
    )
    materials_forbidden_content_tokens = present_text_tokens(
        (needle_materials_text,),
        (
            "point3f[] points",
            "faceVertexIndices",
            "primvars:normals",
            "physics:",
            "Physx",
            "physx",
            "Newton",
            "newton:",
            "prepend references =",
        ),
    )
    neutral_required_tokens = [
        'defaultPrim = "DrAnmarNeedle"',
        'over "DrAnmarNeedle"',
        'over "Looks"',
        'def Material "NeedleSteelPhysics"',
        '"PhysicsMaterialAPI"',
        'over "Needle" (',
        '"PhysicsRigidBodyAPI", "PhysicsMassAPI"',
        'def Scope "Collision"',
        '"PhysicsCollisionAPI", "MaterialBindingAPI"',
        'over "NeedleInterface"',
        'def PhysicsFixedJoint "FactorySwage"',
    ]
    physx_required_tokens = [
        "@DrAnmarNeedle_physics.usda@",
        'defaultPrim = "DrAnmarNeedle"',
        'over "DrAnmarNeedle"',
        'over "Looks"',
        'over "NeedleSteelPhysics" (',
        '"PhysxMaterialAPI"',
        'over "Needle" (',
        '"PhysxRigidBodyAPI"',
        'over "Collision"',
        'over "C000" (',
        '"PhysxCollisionAPI"',
        "physxCollision:contactOffset",
        "physxCollision:restOffset",
    ]
    missing_neutral_layer_tokens = missing_text_tokens(needle_physics_text, neutral_required_tokens)
    missing_physx_layer_tokens = missing_text_tokens(needle_physx_text, physx_required_tokens)
    neutral_engine_specific_properties = sorted(
        set(re.findall(r"\b(?:physx[A-Za-z]*:|newton:)[A-Za-z][A-Za-z0-9_]*", needle_physics_text))
    )
    neutral_engine_specific_schemas = sorted(
        set(re.findall(r'"((?:Physx|Newton)[A-Za-z0-9_]*API)"', needle_physics_text))
    )
    physx_neutral_properties = sorted(set(re.findall(r"\bphysics:[A-Za-z][A-Za-z0-9_]*", needle_physx_text)))
    physx_newton_properties = sorted(set(re.findall(r"\bnewton:[A-Za-z][A-Za-z0-9_]*", needle_physx_text)))
    physx_newton_schemas = sorted(set(re.findall(r'"(Newton[A-Za-z0-9_]*API)"', needle_physx_text)))
    physics_visual_payload_tokens = present_text_tokens(
        (needle_physics_text, needle_physx_text),
        (
            'def Mesh "Visual"',
            "point3f[] points",
            'def Shader "PreviewSurface"',
            "prepend references =",
        ),
    )
    check(
        checks,
        "needle_asset_structure_source_ownership",
        layer_organization["entry_layer"] == "DrAnmarNeedle.usda"
        and layer_organization["base_layer"] == "DrAnmarNeedle_base.usda"
        and layer_organization["geometry_layer"] == "DrAnmarNeedle_geometry.usd"
        and layer_organization["geometry_format"] == "usdc"
        and layer_organization["materials_layer"] == "DrAnmarNeedle_materials.usda"
        and layer_organization["entry_layer_max_bytes"] == 8192
        and layer_organization["physics_layer"] == "DrAnmarNeedle_physics.usda"
        and layer_organization["physx_layer"] == "DrAnmarNeedle_physx.usda"
        and layer_organization["composition"]
        == "entry_references_base_and_public_Physics_variant_payloads_none_physics_or_physx"
        and layer_organization["default_runtime"] == "physx"
        and layer_organization["variant_set"] == "Physics"
        and layer_organization["variant_choices"] == ["none", "physics", "physx"]
        and layer_organization["nested_suture_variant_policy"]
        == "assembly_Physics_selection_authors_the_same_selection_on_Suture"
        and layer_organization["physics_payload_loading"] == "deferred_until_selected_variant_is_loaded"
        and layer_organization["entry_layer_owns"]
        == [
            "stage_metadata",
            "base_reference",
            "public_physics_variant_contract",
            "default_physics_selection",
            "nested_suture_physics_selection",
        ]
        and layer_organization["base_layer_owns"]
        == [
            "asset_identity",
            "structural_hierarchy",
            "suture_reference_and_transform",
            "geometry_and_material_sublayer_composition",
        ]
        and layer_organization["geometry_layer_owns"]
        == [
            "visual_mesh_topology",
            "visual_mesh_points",
            "visual_mesh_normals",
            "visual_mesh_extent",
        ]
        and layer_organization["materials_layer_owns"]
        == [
            "visual_material",
            "visual_shader",
            "visual_material_binding",
        ]
        and layer_organization["physics_layer_owns"]
        == [
            "physics_material",
            "rigid_body_schemas_and_attributes",
            "compound_colliders",
            "suture_interface_override",
            "factory_swage_joint",
        ]
        and layer_organization["physx_layer_owns"]
        == [
            "physx_material_schema_and_combine_mode",
            "physx_rigid_body_schema_and_solver_tuning",
            "physx_collision_schemas_and_contact_offsets",
        ]
        and layer_organization["physics_property_prefixes"] == ["physics:"]
        and layer_organization["physics_schema_prefixes"] == ["Physics"]
        and layer_organization["physx_property_prefixes"] == ["physx"]
        and layer_organization["physx_schema_prefixes"] == ["Physx"]
        and layer_organization["forbidden_live_schema_prefixes"] == ["Newton"]
        and layer_organization["engine_isolation"]
        == "neutral_layer_contains_no_physx_or_newton_opinions_and_physx_layer_contains_no_newton_opinions"
        and layer_organization["content_isolation"]
        == "entry_contains_only_interface_composition_base_contains_identity_hierarchy_suture_reference_and_geometry_material_composition_geometry_contains_only_mesh_data_and_materials_contains_only_visual_lookdev_and_binding"
        and "@DrAnmarNeedle_base.usda@" in needle_entry_text
        and "@DrAnmarNeedle_physx.usda@" in needle_entry_text
        and "@DrAnmarNeedle_physics.usda@" in needle_entry_text
        and "@DrAnmarNeedle_materials.usda@" not in needle_entry_text
        and "@DrAnmarNeedle_geometry.usd@" not in needle_entry_text
        and "@DrAnmarNeedle_materials.usda@" in needle_base_text
        and "@DrAnmarNeedle_geometry.usd@" in needle_base_text
        and "prepend references = @../suture/DrAnmarSuture4_0.usda@" in needle_base_text
        and "@DrAnmarNeedle_physics.usda@" in needle_physx_text
        and 'append variantSets = "Physics"' in needle_entry_text
        and 'string Physics = "physx"' in needle_entry_text
        and needle_entry_text.count("prepend payload =") == 2
        and needle_entry_text.count('over "Suture" (') == 3
        and needle_entry_text.count('string Physics = "none"') == 1
        and needle_entry_text.count('string Physics = "physics"') == 1
        and needle_entry_text.count('string Physics = "physx"') == 2
        and len(needle_entry_text.encode("utf-8")) <= int(layer_organization["entry_layer_max_bytes"])
        and needle_geometry_is_usdc
        and not entry_physics_properties
        and not entry_physics_schemas
        and not entry_physics_typed_prims
        and not entry_forbidden_content_tokens
        and not base_physics_properties
        and not base_physics_schemas
        and not base_physics_typed_prims
        and not base_forbidden_content_tokens
        and not missing_geometry_layer_tokens
        and not missing_materials_layer_tokens
        and not geometry_forbidden_content_tokens
        and not materials_forbidden_content_tokens
        and not missing_neutral_layer_tokens
        and not missing_physx_layer_tokens
        and not neutral_engine_specific_properties
        and not neutral_engine_specific_schemas
        and not physx_neutral_properties
        and not physx_newton_properties
        and not physx_newton_schemas
        and not physics_visual_payload_tokens
        and "Newton" not in needle_text
        and "newton:" not in needle_text,
        {
            "contract": layer_organization,
            "entry_physics_properties": entry_physics_properties,
            "entry_physics_schemas": entry_physics_schemas,
            "entry_physics_typed_prims": entry_physics_typed_prims,
            "entry_bytes": len(needle_entry_text.encode("utf-8")),
            "entry_forbidden_content_tokens": entry_forbidden_content_tokens,
            "base_physics_properties": base_physics_properties,
            "base_physics_schemas": base_physics_schemas,
            "base_physics_typed_prims": base_physics_typed_prims,
            "base_forbidden_content_tokens": base_forbidden_content_tokens,
            "geometry_is_usdc": needle_geometry_is_usdc,
            "missing_geometry_layer_tokens": missing_geometry_layer_tokens,
            "geometry_forbidden_content_tokens": geometry_forbidden_content_tokens,
            "missing_materials_layer_tokens": missing_materials_layer_tokens,
            "materials_forbidden_content_tokens": materials_forbidden_content_tokens,
            "missing_neutral_layer_tokens": missing_neutral_layer_tokens,
            "missing_physx_layer_tokens": missing_physx_layer_tokens,
            "neutral_engine_specific_properties": neutral_engine_specific_properties,
            "neutral_engine_specific_schemas": neutral_engine_specific_schemas,
            "physx_neutral_properties": physx_neutral_properties,
            "physx_newton_properties": physx_newton_properties,
            "physx_newton_schemas": physx_newton_schemas,
            "physics_visual_payload_tokens": physics_visual_payload_tokens,
        },
        "lightweight entry, binary mesh, visual look-development, neutral physics, and PhysX tuning each have"
        " isolated source ownership without unqualified cross-engine schemas",
    )
    forbidden_needle_tokens = [
        "../Surgical_needle",
        "needle_sdf.usd",
        "ORBIT",
    ]
    present_forbidden_needle_tokens = [token for token in forbidden_needle_tokens if token in needle_text]
    check(
        checks,
        "independent_dr_anmar_needle_geometry",
        not present_forbidden_needle_tokens,
        {"forbidden_references": present_forbidden_needle_tokens},
        "no external needle geometry or naming",
    )
    authored_collision_capsules = len(re.findall(r'def Capsule "C\d{3}"', needle_text))
    check(
        checks,
        "needle_visual_and_collision_resolution",
        len(needle_mesh.points) == derived_needle.visual_vertex_count
        and authored_collision_capsules == derived_needle.collision_capsule_count,
        {
            "visual_vertices": len(needle_mesh.points),
            "collision_capsules": authored_collision_capsules,
        },
        {
            "visual_vertices": derived_needle.visual_vertex_count,
            "collision_capsules": derived_needle.collision_capsule_count,
        },
    )
    authored_normals_match = re.search(
        r"normal3f\[\] primvars:normals = \[(.*?)\]" r"\s*\(\s*interpolation = \"faceVarying\"\s*\)",
        needle_text,
        flags=re.DOTALL,
    )
    authored_normal_indices_match = re.search(
        r"int\[\] primvars:normals:indices = \[(.*?)\]",
        needle_text,
        flags=re.DOTALL,
    )
    authored_normals: tuple[tuple[float, float, float], ...] = ()
    if authored_normals_match is not None:
        authored_normals = tuple(
            (
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            )
            for match in re.finditer(
                r"\(\s*([-+0-9.eE]+)\s*,\s*" r"([-+0-9.eE]+)\s*,\s*" r"([-+0-9.eE]+)\s*\)",
                authored_normals_match.group(1),
            )
        )
    authored_normal_indices = (
        tuple(int(value.strip()) for value in authored_normal_indices_match.group(1).split(",") if value.strip())
        if authored_normal_indices_match is not None
        else ()
    )
    maximum_authored_normal_error = (
        max(
            abs(authored - expected)
            for authored_normal, expected_normal in zip(
                authored_normals,
                needle_mesh.normals,
                strict=True,
            )
            for authored, expected in zip(
                authored_normal,
                expected_normal,
                strict=True,
            )
        )
        if len(authored_normals) == len(needle_mesh.normals)
        else math.inf
    )
    visual_normal_contract = needle_profile["construction"]["visual_normals"]
    check(
        checks,
        "needle_analytic_indexed_visual_normals",
        visual_normal_contract["representation"] == "primvars:normals"
        and visual_normal_contract["interpolation"] == "faceVarying"
        and visual_normal_contract["indexed"] is True
        and visual_normal_contract["source"] == "analytic_tapered_curved_swept_surface_derivatives"
        and authored_normals_match is not None
        and authored_normal_indices_match is not None
        and needle_text.count("normal3f[] primvars:normals") == 1
        and "normal3f[] normals" not in needle_text
        and len(authored_normals) == len(needle_mesh.normals)
        and authored_normal_indices == needle_mesh.normal_indices
        and maximum_authored_normal_error <= float(visual_normal_contract["binary_storage_component_tolerance"])
        and len(needle_mesh.normals) == len(needle_mesh.points) + 1
        and len(needle_mesh.normal_indices) == len(needle_mesh.face_vertex_indices)
        and int(needle_normal_quality["unused_normal_value_count"]) == 0
        and int(needle_normal_quality["invalid_normal_index_count"]) == 0
        and int(needle_normal_quality["non_finite_normal_component_count"]) == 0
        and float(needle_normal_quality["maximum_unit_length_error"])
        <= float(visual_normal_contract["deterministic_unit_length_tolerance"])
        and float(needle_normal_quality["minimum_face_corner_alignment_dot"])
        > float(visual_normal_contract["minimum_outward_winding_alignment_dot"])
        and int(needle_normal_quality["non_outward_face_corner_count"]) == 0
        and float(needle_normal_quality["minimum_face_area_m2"]) > 0.0
        and float(needle_normal_quality["maximum_surface_tangent_dot"]) <= 1.0e-12
        and float(needle_normal_quality["minimum_cross_section_outward_dot"]) > 0.99
        and float(needle_normal_quality["maximum_swage_cap_side_normal_dot"]) <= 1.0e-12,
        {
            "contract": visual_normal_contract,
            "quality": needle_normal_quality,
            "authored_normal_value_count": len(authored_normals),
            "authored_normal_index_count": len(authored_normal_indices),
            "maximum_authored_normal_error": maximum_authored_normal_error,
        },
        "binary indexed face-varying normals remain within bounded float storage error of analytic tapered-surface"
        " derivatives, outward winding, and hard swage cap",
    )
    needle_collision_capsules = build_needle_collision_capsules(needle_profile)
    collision_contract = needle_profile["construction"]["collision_contract"]
    render_collision_contract = collision_contract["render_collision_separation"]
    material_organization = needle_profile["material"]["usd_organization"]
    visual_material_path = (
        f"/{DR_ANMAR_NEEDLE_ROOT_PRIM}/{material_organization['scope']}/{material_organization['visual_material']}"
    )
    physics_material_path = (
        f"/{DR_ANMAR_NEEDLE_ROOT_PRIM}/{material_organization['scope']}/{material_organization['physics_material']}"
    )
    expected_physics_material_binding = (
        f'rel {render_collision_contract["collider_physics_material_binding"]} = <{physics_material_path}>'
    )
    collision_attribute_errors: list[str] = []
    for index, capsule in enumerate(needle_collision_capsules):
        neutral_match = re.search(
            rf'def Capsule "C{index:03d}".*?\n            \}}',
            needle_physics_text,
            flags=re.DOTALL,
        )
        physx_match = re.search(
            rf'over "C{index:03d}".*?\n            \}}',
            needle_physx_text,
            flags=re.DOTALL,
        )
        if neutral_match is None or physx_match is None:
            collision_attribute_errors.append(f"C{index:03d}:missing_neutral_or_physx_block")
            continue
        neutral_block = neutral_match.group(0)
        physx_block = physx_match.group(0)
        height_match = re.search(r"float height = ([0-9.eE+-]+)", neutral_block)
        radius_match = re.search(r"float radius = ([0-9.eE+-]+)", neutral_block)
        physx_contact_match = re.search(
            r"float physxCollision:contactOffset = ([0-9.eE+-]+)",
            physx_block,
        )
        physx_rest_match = re.search(
            r"float physxCollision:restOffset = ([0-9.eE+-]+)",
            physx_block,
        )
        if height_match is None or not math.isclose(
            float(height_match.group(1)),
            capsule.cylinder_height_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            collision_attribute_errors.append(f"C{index:03d}:height")
        if radius_match is None or not math.isclose(
            float(radius_match.group(1)),
            capsule.collision_radius_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            collision_attribute_errors.append(f"C{index:03d}:radius")
        if "float3[] extent = [" not in neutral_block:
            collision_attribute_errors.append(f"C{index:03d}:extent")
        if f'uniform token purpose = "{render_collision_contract["collider_purpose"]}"' not in neutral_block:
            collision_attribute_errors.append(f"C{index:03d}:purpose")
        if f'token visibility = "{render_collision_contract["collider_visibility"]}"' not in neutral_block:
            collision_attribute_errors.append(f"C{index:03d}:visibility")
        if expected_physics_material_binding not in neutral_block:
            collision_attribute_errors.append(f"C{index:03d}:physics_material_binding")
        if "rel material:binding = " in neutral_block:
            collision_attribute_errors.append(f"C{index:03d}:visual_material_binding")
        expected_contact_attributes = (
            (
                "physx_contact_offset",
                physx_contact_match,
                capsule.contact_offset_m,
            ),
            (
                "physx_rest_offset",
                physx_rest_match,
                capsule.rest_offset_m,
            ),
        )
        for (
            attribute_name,
            attribute_match,
            expected_value,
        ) in expected_contact_attributes:
            if attribute_match is None or not math.isclose(
                float(attribute_match.group(1)),
                expected_value,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                collision_attribute_errors.append(f"C{index:03d}:{attribute_name}")
    maximum_chord_error = max(
        abs(capsule.cylinder_height_m - capsule.chord_length_m) for capsule in needle_collision_capsules
    )
    maximum_sagitta = max(capsule.curvature_sagitta_m for capsule in needle_collision_capsules)
    maximum_seam_margin = max(capsule.visual_seam_margin_m for capsule in needle_collision_capsules)
    collision_coverage = needle_mesh_collision_coverage(
        needle_profile,
        needle_mesh,
    )
    check(
        checks,
        "needle_collision_envelope_matches_centerline_partition",
        not collision_attribute_errors
        and len(needle_collision_capsules) == derived_needle.collision_capsule_count
        and all(
            capsule.collision_radius_m >= capsule.physical_radius_m
            and capsule.cylinder_height_m > 0.0
            and math.isclose(
                capsule.collision_radius_m,
                capsule.physical_radius_m + capsule.curvature_sagitta_m + capsule.visual_seam_margin_m,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            for capsule in needle_collision_capsules
        )
        and maximum_chord_error <= 1.0e-12
        and 0.0 < maximum_sagitta < 1.0e-5
        and 0.0 < maximum_seam_margin < 1.0e-5
        and collision_coverage["uncovered_visual_vertex_count"] == 0
        and collision_coverage["uncovered_visual_face_count"] == 0
        and collision_coverage["minimum_visual_vertex_containment_margin_m"] >= -1.0e-12
        and collision_coverage["minimum_visual_face_containment_margin_m"] >= -1.0e-12
        and needle_text.count("float3[] extent = [") == derived_needle.collision_capsule_count + 1,
        {
            "capsule_count": len(needle_collision_capsules),
            "attribute_errors": collision_attribute_errors,
            "maximum_chord_length_error_m": maximum_chord_error,
            "maximum_curvature_sagitta_m": maximum_sagitta,
            "maximum_visual_seam_margin_m": maximum_seam_margin,
            "visual_mesh_collision_coverage": collision_coverage,
            "authored_extent_count": needle_text.count("float3[] extent = ["),
        },
        "capsule spine equals each assigned chord with curvature-bounded radius, explicit extents, and complete"
        " visual-mesh coverage",
    )
    nvidia_stack_references = needle_profile.get(
        "nvidia_stack_references",
        [],
    )
    render_collision_counts = {
        "guide_purpose": needle_text.count(
            f'uniform token purpose = "{render_collision_contract["collider_purpose"]}"'
        ),
        "invisible": needle_text.count(f'token visibility = "{render_collision_contract["collider_visibility"]}"'),
        "physics_material_binding": needle_text.count(expected_physics_material_binding),
        "visual_material_binding": needle_text.count(
            f'rel {render_collision_contract["visual_material_binding"]} = <{visual_material_path}>'
        ),
    }
    check(
        checks,
        "needle_render_collision_separation",
        render_collision_contract["visual_prim"] == "Needle/Visual"
        and render_collision_contract["collider_prim_pattern"] == "Needle/Collision/C###"
        and render_collision_contract["collider_purpose"] == "guide"
        and render_collision_contract["collider_visibility"] == "invisible"
        and render_collision_contract["collider_physics_material_binding"] == "material:binding:physics"
        and render_collision_contract["collider_physics_material_path"] == "Looks/NeedleSteelPhysics"
        and render_collision_contract["visual_material_binding"] == "material:binding"
        and render_collision_contract["visual_material_path"] == "Looks/NeedleSteelVisual"
        and render_collision_contract["reason"] == "nonrendering_debuggable_collision_geometry"
        and render_collision_counts["guide_purpose"] == derived_needle.collision_capsule_count
        and render_collision_counts["invisible"] == derived_needle.collision_capsule_count
        and render_collision_counts["physics_material_binding"] == derived_needle.collision_capsule_count
        and render_collision_counts["visual_material_binding"] == 1,
        {
            "contract": render_collision_contract,
            "authored_counts": render_collision_counts,
            "collision_capsule_count": derived_needle.collision_capsule_count,
        },
        "one render mesh plus guide-purpose invisible compound colliders with physics-only material bindings",
    )
    visual_material_match = re.search(
        rf'def Material "{re.escape(str(material_organization["visual_material"]))}"\s*' r"\{(.*?)\n        \}",
        needle_materials_text,
        flags=re.DOTALL,
    )
    neutral_physics_material_match = re.search(
        rf'def Material "{re.escape(str(material_organization["physics_material"]))}".*?' r"\{(.*?)\n        \}",
        needle_physics_text,
        flags=re.DOTALL,
    )
    physx_material_match = re.search(
        rf'over "{re.escape(str(material_organization["physics_material"]))}".*?' r"\{(.*?)\n        \}",
        needle_physx_text,
        flags=re.DOTALL,
    )
    visual_material_block = visual_material_match.group(1) if visual_material_match is not None else ""
    neutral_physics_material_block = (
        neutral_physics_material_match.group(1) if neutral_physics_material_match is not None else ""
    )
    physx_material_block = physx_material_match.group(1) if physx_material_match is not None else ""
    check(
        checks,
        "needle_top_level_looks_and_separate_materials",
        material_organization["scope"] == "Looks"
        and material_organization["visual_material"] == "NeedleSteelVisual"
        and material_organization["physics_material"] == "NeedleSteelPhysics"
        and material_organization["separate_by_purpose"] is True
        and material_organization["visual_shader"] == "UsdPreviewSurface"
        and material_organization["physics_api_schemas"]
        == [
            "PhysicsMaterialAPI",
            "PhysxMaterialAPI",
        ]
        and needle_text.count('def Scope "Looks"') == 1
        and 'def Scope "Materials"' not in needle_text
        and len(re.findall(r'def Material "[^"]+"', needle_text)) == 2
        and visual_material_match is not None
        and 'uniform token info:id = "UsdPreviewSurface"' in visual_material_block
        and 'def Shader "PreviewSurface"' in visual_material_block
        and "PhysicsMaterialAPI" not in visual_material_block
        and "PhysxMaterialAPI" not in visual_material_block
        and "physics:staticFriction" not in visual_material_block
        and neutral_physics_material_match is not None
        and '"PhysicsMaterialAPI"' in neutral_physics_material_match.group(0)
        and "PhysxMaterialAPI" not in neutral_physics_material_match.group(0)
        and "physics:staticFriction" in neutral_physics_material_block
        and "physics:dynamicFriction" in neutral_physics_material_block
        and "physics:restitution" in neutral_physics_material_block
        and "physxMaterial:frictionCombineMode" not in neutral_physics_material_block
        and 'def Shader "PreviewSurface"' not in neutral_physics_material_block
        and physx_material_match is not None
        and '"PhysxMaterialAPI"' in physx_material_match.group(0)
        and "PhysicsMaterialAPI" not in physx_material_match.group(0)
        and "physxMaterial:frictionCombineMode" in physx_material_block
        and "physics:staticFriction" not in physx_material_block
        and needle_text.count(visual_material_path) == 2
        and needle_text.count(physics_material_path) == derived_needle.collision_capsule_count,
        {
            "contract": material_organization,
            "visual_material_path": visual_material_path,
            "physics_material_path": physics_material_path,
            "visual_material_found": visual_material_match is not None,
            "neutral_physics_material_found": neutral_physics_material_match is not None,
            "physx_material_overlay_found": physx_material_match is not None,
            "authored_material_count": len(re.findall(r'def Material "[^"]+"', needle_text)),
            "visual_path_reference_count": needle_text.count(visual_material_path),
            "physics_path_reference_count": needle_text.count(physics_material_path),
        },
        "two direct children of top-level Looks with visual, neutral physics, and PhysX material responsibilities"
        " isolated by source layer",
    )
    contact_offset_contract = collision_contract["contact_offsets"]
    contact_offsets = [capsule.contact_offset_m for capsule in needle_collision_capsules]
    rest_offsets = [capsule.rest_offset_m for capsule in needle_collision_capsules]
    radius_offset_pairs = sorted(
        (
            capsule.collision_radius_m,
            capsule.contact_offset_m,
        )
        for capsule in needle_collision_capsules
    )
    contact_offsets_monotonic = all(
        left[1] <= right[1] + 1.0e-15
        for left, right in zip(
            radius_offset_pairs,
            radius_offset_pairs[1:],
        )
    )
    contact_attribute_counts = {
        "PhysxCollisionAPI": needle_text.count('"PhysxCollisionAPI"'),
        "physx_contact_offset": needle_text.count("physxCollision:contactOffset"),
        "physx_rest_offset": needle_text.count("physxCollision:restOffset"),
    }
    forbidden_engine_attribute_counts = {
        "NewtonCollisionAPI": needle_text.count('"NewtonCollisionAPI"'),
        "newton_properties": needle_text.count("newton:"),
    }
    check(
        checks,
        "needle_scale_aware_physx_contact_offsets",
        contact_offset_contract["policy"] == "clamped_fraction_of_final_collision_radius"
        and contact_offset_contract["basis"]
        == "engineering_seed_for_thin_ccd_enabled_colliders_pending_native_velocity_and_timestep_sweep"
        and math.isclose(
            float(contact_offset_contract["collision_radius_fraction"]),
            0.1,
        )
        and all(math.isfinite(value) for value in contact_offsets + rest_offsets)
        and all(
            0.0 <= rest_offset < contact_offset
            for rest_offset, contact_offset in zip(
                rest_offsets,
                contact_offsets,
                strict=True,
            )
        )
        and min(contact_offsets) >= float(contact_offset_contract["minimum_m"])
        and max(contact_offsets) <= float(contact_offset_contract["maximum_m"])
        and max(contact_offsets) < min(capsule.collision_radius_m for capsule in needle_collision_capsules)
        and contact_offsets_monotonic
        and all(count == derived_needle.collision_capsule_count for count in contact_attribute_counts.values())
        and all(count == 0 for count in forbidden_engine_attribute_counts.values())
        and contact_offset_contract["engine_layer"] == "DrAnmarNeedle_physx.usda"
        and contact_offset_contract["neutral_layer_policy"] == "no_engine_specific_contact_schema"
        and contact_offset_contract["unsupported_engine_schema_policy"]
        == "do_not_author_until_the_complete_needle_suture_assembly_is_qualified_on_that_backend",
        {
            "contact_offset_range_m": [
                min(contact_offsets),
                max(contact_offsets),
            ],
            "rest_offset_range_m": [
                min(rest_offsets),
                max(rest_offsets),
            ],
            "minimum_collision_radius_m": min(capsule.collision_radius_m for capsule in needle_collision_capsules),
            "contact_offsets_monotonic_with_radius": contact_offsets_monotonic,
            "authored_attribute_counts": contact_attribute_counts,
            "forbidden_engine_attribute_counts": forbidden_engine_attribute_counts,
            "contract": contact_offset_contract,
        },
        "bounded scale-aware PhysX offsets on every collider with no unqualified cross-engine schema authoring",
    )
    native_probe_tokens = [
        "needle_collision_capsule_count",
        "needle_collision_explicit_extent_count",
        "needle_friction_combine_mode",
        "needle_authored_mass_kg",
        "needle_center_of_mass_m",
        "needle_diagonal_inertia_kg_m2",
        "needle_principal_axes_wxyz",
        "needle_mass_properties_match_geometry",
        "needle_physx_contact_offset_range_m",
        "needle_physx_contact_offsets_match_profile",
        "needle_newton_collision_api_count",
        "needle_engine_schema_isolation_valid",
        "needle_visual_normal_value_count",
        "needle_visual_normal_index_count",
        "needle_visual_normals_valid",
        "needle_collision_guide_purpose_count",
        "needle_collision_invisible_count",
        "needle_collision_physics_material_binding_count",
        "needle_render_collision_separation_valid",
        "needle_material_organization_valid",
        "needle_physics_variant_selection",
        "suture_physics_variant_selection",
        "physics_variant_contract_valid",
        "needle_base_layer_name",
        "needle_asset_structure_source_ownership_valid",
        "suture_base_layer_name",
        "suture_asset_structure_source_ownership_valid",
        "suture_physx_collision_api_count",
        "suture_hybrid_ccd_body_count",
        "suture_physx_contact_offsets_match_profile",
        "suture_material_bindings_valid",
    ]
    missing_native_probe_tokens = [token for token in native_probe_tokens if token not in native_probe_text]
    check(
        checks,
        "needle_nvidia_stack_collision_contract",
        len(nvidia_stack_references) >= 11
        and all(
            item.get("url", "").startswith(
                (
                    "https://docs.omniverse.nvidia.com/",
                    "https://docs.isaacsim.omniverse.nvidia.com/",
                )
            )
            and item.get("used_for")
            for item in nvidia_stack_references
        )
        and collision_contract["primitive"] == "UsdGeomCapsule"
        and collision_contract["height_semantics"] == "cylinder_spine_excluding_spherical_caps"
        and collision_contract["spine_length"] == "assigned_centerline_chord"
        and collision_contract["visual_face_coverage"]
        == "minimum_derived_uniform_seam_margin_for_single_convex_capsule_containment_per_face"
        and 0.0 < float(collision_contract["coverage_epsilon_m"]) <= 1.0e-8
        and collision_contract["extent_policy"] == "explicit_local_extent_on_every_capsule"
        and render_collision_contract["collider_purpose"] == "guide"
        and render_collision_contract["collider_visibility"] == "invisible"
        and render_collision_contract["collider_physics_material_binding"] == "material:binding:physics"
        and material_organization["scope"] == "Looks"
        and material_organization["separate_by_purpose"] is True
        and layer_organization["geometry_layer"].endswith("_geometry.usd")
        and layer_organization["geometry_format"] == "usdc"
        and layer_organization["materials_layer"].endswith("_materials.usda")
        and layer_organization["physics_layer"].endswith("_physics.usda")
        and layer_organization["physx_layer"].endswith("_physx.usda")
        and layer_organization["composition"]
        == "entry_references_base_and_public_Physics_variant_payloads_none_physics_or_physx"
        and layer_organization["engine_isolation"]
        == "neutral_layer_contains_no_physx_or_newton_opinions_and_physx_layer_contains_no_newton_opinions"
        and layer_organization["content_isolation"]
        == "entry_contains_only_interface_composition_base_contains_identity_hierarchy_suture_reference_and_geometry_material_composition_geometry_contains_only_mesh_data_and_materials_contains_only_visual_lookdev_and_binding"
        and contact_offset_contract["engine_layer"] == "DrAnmarNeedle_physx.usda"
        and contact_offset_contract["neutral_layer_policy"] == "no_engine_specific_contact_schema"
        and needle_profile["solver"]["ccd"] is True
        and needle_profile["contact"]["combine_mode"] == "max"
        and not missing_native_probe_tokens,
        {
            "reference_count": len(nvidia_stack_references),
            "collision_contract": collision_contract,
            "ccd": needle_profile["solver"]["ccd"],
            "friction_combine_mode": needle_profile["contact"]["combine_mode"],
            "missing_native_probe_tokens": missing_native_probe_tokens,
        },
        "NVIDIA binary-geometry, look-development, primitive-collider, CCD, material, and engine-isolated"
        " source-layer contract",
    )
    construction = needle_profile["construction"]
    arc_range = [float(value) for value in construction["centerline_arc_length_range_m"]]
    diameter_range = [float(value) for value in construction["body_diameter_range_m"]]
    check(
        checks,
        "needle_scale_and_mass",
        arc_range[0] <= derived_needle.arc_length_m <= arc_range[1]
        and diameter_range[0] <= 2.0 * derived_needle.body_radius_m <= diameter_range[1]
        and 0.0 < derived_needle.mass_kg < 0.001,
        {
            "arc_length_m": derived_needle.arc_length_m,
            "body_diameter_m": 2.0 * derived_needle.body_radius_m,
            "mass_kg": derived_needle.mass_kg,
        },
        {
            "arc_length_m": arc_range,
            "body_diameter_m": diameter_range,
            "mass_kg": "positive and below 1 gram",
        },
    )
    mass_properties = derived_needle.mass_properties
    coarse_mass_properties = derive_needle_mass_properties(
        needle_profile,
        integration_slices=(DEFAULT_MASS_PROPERTY_INTEGRATION_SLICES // 2),
    )
    reconstructed_inertia = reconstruct_inertia_tensor(
        mass_properties.diagonal_inertia_kg_m2,
        mass_properties.principal_axes_wxyz,
    )
    maximum_reconstruction_error = max(
        abs(reconstructed_inertia[row][column] - mass_properties.inertia_tensor_kg_m2[row][column])
        for row in range(3)
        for column in range(3)
    )
    maximum_relative_inertia_convergence_drift = max(
        abs(fine - coarse) / fine
        for fine, coarse in zip(
            mass_properties.diagonal_inertia_kg_m2,
            coarse_mass_properties.diagonal_inertia_kg_m2,
            strict=True,
        )
    )
    maximum_center_of_mass_convergence_drift_m = max(
        abs(fine - coarse)
        for fine, coarse in zip(
            mass_properties.center_of_mass_m,
            coarse_mass_properties.center_of_mass_m,
            strict=True,
        )
    )
    relative_mass_convergence_drift = (
        abs(mass_properties.mass_kg - coarse_mass_properties.mass_kg) / mass_properties.mass_kg
    )
    quaternion_norm = math.sqrt(sum(component * component for component in mass_properties.principal_axes_wxyz))

    def authored_tuple(
        type_name: str,
        attribute_name: str,
        component_count: int,
    ) -> tuple[float, ...] | None:
        match = re.search(
            rf"{re.escape(type_name)} {re.escape(attribute_name)}" rf" = \(([^)]+)\)",
            needle_text,
        )
        if match is None:
            return None
        values = tuple(float(value.strip()) for value in match.group(1).split(","))
        return values if len(values) == component_count else None

    authored_mass_match = re.search(
        r"float physics:mass = ([0-9.eE+-]+)",
        needle_text,
    )
    authored_mass = float(authored_mass_match.group(1)) if authored_mass_match is not None else None
    authored_center_of_mass = authored_tuple(
        "point3f",
        "physics:centerOfMass",
        3,
    )
    authored_diagonal_inertia = authored_tuple(
        "float3",
        "physics:diagonalInertia",
        3,
    )
    authored_principal_axes = authored_tuple(
        "quatf",
        "physics:principalAxes",
        4,
    )

    def tuples_close(
        left: tuple[float, ...] | None,
        right: tuple[float, ...],
        *,
        relative_tolerance: float,
        absolute_tolerance: float,
    ) -> bool:
        return left is not None and all(
            math.isclose(
                left_value,
                right_value,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            )
            for left_value, right_value in zip(
                left,
                right,
                strict=True,
            )
        )

    mass_contract = construction["mass_properties"]
    sampled_mass_parameters = sample_episode_parameters(
        needle_profile,
        1701,
    )
    sampled_mass_properties = needle_mass_properties_for_mass(
        needle_profile,
        sampled_mass_parameters.mass_kg,
    )
    mass_scale = sampled_mass_parameters.mass_kg / mass_properties.mass_kg
    expected_scaled_inertia = tuple(value * mass_scale for value in mass_properties.diagonal_inertia_kg_m2)
    live_mass_property_tokens = [
        "GetCenterOfMassAttr().Set(",
        "GetDiagonalInertiaAttr().Set(",
        "GetPrincipalAxesAttr().Set(",
        "needle_mass_properties_for_mass(",
    ]
    missing_live_mass_property_tokens = [token for token in live_mass_property_tokens if token not in integration_text]
    diagonal_inertia = mass_properties.diagonal_inertia_kg_m2
    check(
        checks,
        "needle_explicit_geometry_mass_properties",
        mass_properties.integration_slices
        == DEFAULT_MASS_PROPERTY_INTEGRATION_SLICES
        == int(mass_contract["integration_slices"])
        and mass_contract["source"] == "numerical_volume_integration_of_tapered_curved_swept_solid"
        and mass_contract["curvature_jacobian"] == "one_plus_outward_radial_coordinate_over_curvature_radius"
        and mass_contract["includes_finite_cross_section_inertia"] is True
        and mass_contract["usd_authoring"]
        == [
            "physics:mass",
            "physics:centerOfMass",
            "physics:diagonalInertia",
            "physics:principalAxes",
        ]
        and math.isclose(
            mass_properties.mass_kg,
            derived_needle.mass_kg,
            rel_tol=0.0,
            abs_tol=1.0e-18,
        )
        and all(math.isfinite(value) for value in diagonal_inertia)
        and all(value > 0.0 for value in diagonal_inertia)
        and all(
            diagonal_inertia[index] <= sum(diagonal_inertia) - diagonal_inertia[index] + 1.0e-20 for index in range(3)
        )
        and all(
            needle_mesh.extent_min[index] - 1.0e-12
            <= mass_properties.center_of_mass_m[index]
            <= needle_mesh.extent_max[index] + 1.0e-12
            for index in range(3)
        )
        and math.isclose(
            quaternion_norm,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and maximum_reconstruction_error <= max(diagonal_inertia) * 1.0e-12
        and relative_mass_convergence_drift < 1.0e-8
        and maximum_center_of_mass_convergence_drift_m < 3.0e-11
        and maximum_relative_inertia_convergence_drift < 5.0e-7
        and authored_mass is not None
        and math.isclose(
            authored_mass,
            mass_properties.mass_kg,
            rel_tol=1.0e-10,
            abs_tol=0.0,
        )
        and tuples_close(
            authored_center_of_mass,
            mass_properties.center_of_mass_m,
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-14,
        )
        and tuples_close(
            authored_diagonal_inertia,
            diagonal_inertia,
            relative_tolerance=1.0e-10,
            absolute_tolerance=0.0,
        )
        and tuples_close(
            authored_principal_axes,
            mass_properties.principal_axes_wxyz,
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
        )
        and tuples_close(
            sampled_mass_properties.diagonal_inertia_kg_m2,
            expected_scaled_inertia,
            relative_tolerance=1.0e-12,
            absolute_tolerance=0.0,
        )
        and sampled_mass_properties.center_of_mass_m == mass_properties.center_of_mass_m
        and sampled_mass_properties.principal_axes_wxyz == mass_properties.principal_axes_wxyz
        and not missing_live_mass_property_tokens,
        {
            "integration_slices": mass_properties.integration_slices,
            "mass_kg": mass_properties.mass_kg,
            "center_of_mass_m": mass_properties.center_of_mass_m,
            "inertia_tensor_kg_m2": mass_properties.inertia_tensor_kg_m2,
            "diagonal_inertia_kg_m2": diagonal_inertia,
            "principal_axes_wxyz": mass_properties.principal_axes_wxyz,
            "principal_axes_norm": quaternion_norm,
            "maximum_reconstruction_error_kg_m2": maximum_reconstruction_error,
            "relative_mass_convergence_drift": relative_mass_convergence_drift,
            "maximum_center_of_mass_convergence_drift_m": maximum_center_of_mass_convergence_drift_m,
            "maximum_relative_inertia_convergence_drift": maximum_relative_inertia_convergence_drift,
            "authored_mass_kg": authored_mass,
            "authored_center_of_mass_m": authored_center_of_mass,
            "authored_diagonal_inertia_kg_m2": authored_diagonal_inertia,
            "authored_principal_axes_wxyz": authored_principal_axes,
            "episode_mass_scale": mass_scale,
            "episode_diagonal_inertia_kg_m2": sampled_mass_properties.diagonal_inertia_kg_m2,
            "missing_live_mass_property_tokens": missing_live_mass_property_tokens,
        },
        "explicit converged geometry-derived USD mass properties with density-consistent episode scaling",
    )
    sim_to_real = needle_profile["sim_to_real"]
    gaps = sim_to_real["gaps"]
    implemented_randomization = sim_to_real["implemented_randomization_on_episode_reset"]
    planned_randomization = sim_to_real["planned_randomization_after_calibration"]
    complete_gaps = all(
        {
            "id",
            "risk",
            "mitigation",
            "calibration_target",
            "status",
        }.issubset(gap)
        for gap in gaps
    )
    check(
        checks,
        "sim_to_real_gap_register",
        len(gaps) >= 7 and complete_gaps and len(implemented_randomization) >= 4 and len(planned_randomization) >= 4,
        {
            "gap_count": len(gaps),
            "implemented_randomized_parameters": len(implemented_randomization),
            "planned_randomized_parameters": len(planned_randomization),
            "complete_gap_records": complete_gaps,
        },
        {
            "gap_count": "at least 7",
            "implemented_randomized_parameters": "at least 4",
            "planned_randomized_parameters": "at least 4",
            "complete_gap_records": True,
        },
    )
    sample_a = sample_episode_parameters(needle_profile, 1701)
    sample_a_replay = sample_episode_parameters(needle_profile, 1701)
    sample_b = sample_episode_parameters(needle_profile, 1702)
    check(
        checks,
        "sim_to_real_randomization_replay",
        sample_a == sample_a_replay and sample_a != sample_b,
        {
            "seed_1701": sample_a.payload(),
            "seed_1701_replay": sample_a_replay.payload(),
            "seed_1702": sample_b.payload(),
        },
        "same seed exactly replays; different seed changes the domain",
    )
    sampled_suture_a, sampled_suture_domain_a = sample_suture_runtime_profile(
        profile,
        2701,
    )
    sampled_suture_replay, sampled_suture_domain_replay = sample_suture_runtime_profile(profile, 2701)
    sampled_suture_b, sampled_suture_domain_b = sample_suture_runtime_profile(
        profile,
        2702,
    )
    suture_gaps = profile["sim_to_real"]["gaps"]
    suture_requirements = profile["qualification"]["requirements"]
    suture_clinical = [item for item in suture_requirements if item["id"] == "clinical_use"]
    sampled_self_friction_a = sampled_suture_a["contact"]["load_dependent_self_friction"]
    sampled_self_friction_b = sampled_suture_b["contact"]["load_dependent_self_friction"]
    check(
        checks,
        "suture_runtime_domain_and_qualification",
        sampled_suture_domain_a == sampled_suture_domain_replay
        and sampled_suture_a == sampled_suture_replay
        and sampled_suture_domain_a != sampled_suture_domain_b
        and sampled_suture_a != sampled_suture_b
        and sampled_self_friction_a != sampled_self_friction_b
        and sampled_suture_a["contact"]["sampled_static_to_dynamic_ratio"]
        == (sampled_suture_domain_a["static_friction"] / sampled_suture_domain_a["dynamic_friction"])
        and len(suture_gaps) >= 8
        and all({"id", "risk", "mitigation", "status"}.issubset(gap) for gap in suture_gaps)
        and len(profile["sim_to_real"]["runtime_applied_parameter_sampling"]) >= 7
        and profile["qualification"]["policy"] == "fail_closed"
        and len(suture_requirements) >= 7
        and len(suture_clinical) == 1
        and suture_clinical[0]["status"] == "blocked"
        and profile["clinical_validation"] is False,
        {
            "seed_2701": sampled_suture_domain_a,
            "seed_2701_replay": sampled_suture_domain_replay,
            "seed_2702": sampled_suture_domain_b,
            "gap_count": len(suture_gaps),
            "requirement_count": len(suture_requirements),
            "clinical": suture_clinical,
        },
        "replayable live suture material domain with fail-closed evidence gates",
    )
    qualification = needle_profile["qualification"]
    qualification_gates = qualification["gates"]
    clinical_gates = [gate for gate in qualification_gates if gate["id"] == "clinical_use"]
    check(
        checks,
        "fail_closed_sim_to_real_qualification",
        qualification["policy"] == "fail_closed_until_each_evidence_gate_is_satisfied"
        and len(qualification_gates) >= 6
        and len(clinical_gates) == 1
        and clinical_gates[0]["status"] == "blocked"
        and needle_profile["clinical_validation"] is False,
        {
            "policy": qualification["policy"],
            "gate_count": len(qualification_gates),
            "clinical_gate": clinical_gates,
            "clinical_validation": needle_profile["clinical_validation"],
        },
        "machine-readable qualification gates with clinical use blocked",
    )
    needle_evidence = needle_profile.get("evidence", [])
    check(
        checks,
        "needle_research_provenance",
        len(needle_evidence) >= 4 and all(item.get("url") and item.get("used_for") for item in needle_evidence),
        len(needle_evidence),
        "at least four traceable primary product or regulatory sources",
    )
    first_joint = re.search(
        r'def PhysicsJoint "J0000".*?physics:breakForce = ([0-9.eE+-]+)',
        asset_text,
        re.DOTALL,
    )
    pullout_force_n = float(first_joint.group(1)) if first_joint else None
    check(
        checks,
        "breakable_swage_pullout",
        pullout_force_n is not None
        and math.isclose(
            pullout_force_n,
            float(profile["swage"]["pullout_force_n_seed"]),
        ),
        pullout_force_n,
        float(profile["swage"]["pullout_force_n_seed"]),
    )

    class FakeUsdFileCfg:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeAssetBaseCfg:
        class InitialStateCfg:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeScene:
        pass

    covered_local_rooms: list[str] = []
    room_physics_variant_contracts: list[dict[str, Any]] = []
    for room_id in local_room_ids(PROCEDURE_ROOMS):
        fake_scene = FakeScene()
        configuration = configure_dr_anmar_needle(
            fake_scene,
            asset_base_cfg_type=FakeAssetBaseCfg,
            usd_file_cfg_type=FakeUsdFileCfg,
        )
        if getattr(fake_scene, "dr_anmar_needle", None) is not None:
            covered_local_rooms.append(room_id)
            room_physics_variant_contracts.append(
                {
                    "variant_set": configuration["physics_variant_set"],
                    "choices": configuration["physics_variant_choices"],
                    "default": configuration["default_physics_variant"],
                }
            )
    expected_local_rooms = list(local_room_ids(PROCEDURE_ROOMS))
    check(
        checks,
        "all_local_procedure_rooms_receive_instrument",
        covered_local_rooms == expected_local_rooms
        and bool(covered_local_rooms)
        and len(room_physics_variant_contracts) == len(expected_local_rooms)
        and all(
            contract
            == {
                "variant_set": "Physics",
                "choices": ["none", "physics", "physx"],
                "default": "physx",
            }
            for contract in room_physics_variant_contracts
        ),
        {
            "covered_rooms": covered_local_rooms,
            "physics_variants": room_physics_variant_contracts,
        },
        {
            "covered_rooms": expected_local_rooms,
            "physics_variant": {
                "variant_set": "Physics",
                "choices": ["none", "physics", "physx"],
                "default": "physx",
            },
        },
    )

    syntax_tree = ast.parse(workstation_text)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(syntax_tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    integration_calls = [
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "configure_dr_anmar_needle"
    ]
    direct_main_calls = []
    for call in integration_calls:
        ancestor = parents.get(call)
        guarded = False
        while ancestor is not None and not isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(
                ancestor,
                (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith),
            ):
                guarded = True
            ancestor = parents.get(ancestor)
        if isinstance(ancestor, ast.FunctionDef) and ancestor.name == "main" and not guarded:
            direct_main_calls.append(call.lineno)
    check(
        checks,
        "shared_unconditional_workstation_install",
        len(direct_main_calls) == 1,
        len(direct_main_calls),
        1,
    )
    domain_calls = [
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_dr_anmar_needle_episode_domain"
    ]
    domain_call_owners: list[str] = []
    for call in domain_calls:
        ancestor = parents.get(call)
        while ancestor is not None and not isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ancestor = parents.get(ancestor)
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            domain_call_owners.append(ancestor.name)
    check(
        checks,
        "live_reset_domain_randomization",
        sorted(domain_call_owners) == ["main", "reset_environment"],
        sorted(domain_call_owners),
        ["main", "reset_environment"],
    )
    runtime_probe = SutureRuntime(profile)
    half_dose_first = runtime_probe.record_instrument_grasp(
        (20,),
        pressure_pa=float(profile["instrument_damage"]["reference_crush_pressure_pa"]),
        duration_s=0.5,
    )
    half_dose_second = runtime_probe.record_instrument_grasp(
        (20,),
        pressure_pa=float(profile["instrument_damage"]["reference_crush_pressure_pa"]),
        duration_s=0.5,
    )
    live_runtime_tokens = [
        "SutureRuntime(",
        "create_rigid_body_view(",
        "observe_segment_positions(",
        "record_instrument_contact(",
        "apply_to_stage(",
        "force_matrix_w",
        "{ENV_REGEX_NS}/DrAnmarNeedle/Suture/Segments/S.*",
    ]
    missing_live_runtime_tokens = [token for token in live_runtime_tokens if token not in workstation_text]
    check(
        checks,
        "live_suture_material_history_wiring",
        not missing_live_runtime_tokens
        and not half_dose_first
        and half_dose_second
        and runtime_probe.joints[20].grasp_count == 1,
        {
            "missing_workstation_tokens": missing_live_runtime_tokens,
            "cumulative_pressure_dose": runtime_probe.joints[20].crush_dose,
            "grasp_count": runtime_probe.joints[20].grasp_count,
        },
        "native tensor poses and filtered per-jaw contact drive cumulative live material history",
    )
    runtime_detection = profile["runtime_detection"]
    broadphase = runtime_detection["self_contact_broadphase"]
    spacing = derived.segment_spacing_m
    straight_positions = [(index * spacing, 0.0, 0.0) for index in range(derived.segment_count)]
    (
        straight_contacts,
        broadphase_candidates,
        broadphase_overflow_edges,
    ) = runtime_probe._nonadjacent_edge_contacts(
        straight_positions,
        contact_distance_m=float(runtime_detection["self_contact_centerline_distance_m"]),
        minimum_index_separation=int(runtime_detection["knot_minimum_index_separation"]),
        cell_size_multiplier=float(broadphase["cell_size_to_contact_distance"]),
        maximum_cells_per_edge=int(broadphase["maximum_cells_per_edge"]),
    )
    naive_pairs = (
        (derived.segment_count - 1 - int(runtime_detection["knot_minimum_index_separation"]))
        * (derived.segment_count - int(runtime_detection["knot_minimum_index_separation"]))
        // 2
    )
    check(
        checks,
        "geometry_aware_self_contact_broadphase",
        broadphase["algorithm"] == "uniform_spatial_hash_over_expanded_centerline_edge_aabbs"
        and broadphase["narrowphase"] == "exact_3d_segment_to_segment_closest_distance"
        and broadphase["deterministic_pair_order"] is True
        and broadphase["overflow_policy"] == "exact_test_overflow_edge_against_all_nonadjacent_edges"
        and not straight_contacts
        and broadphase_candidates < naive_pairs * 0.05
        and broadphase_overflow_edges == 0
        and callable(runtime_probe._segment_segment_distance)
        and callable(runtime_probe._point_segment_distance),
        {
            "algorithm": broadphase["algorithm"],
            "narrowphase": broadphase["narrowphase"],
            "straight_contact_count": len(straight_contacts),
            "broadphase_candidates": broadphase_candidates,
            "broadphase_overflow_edges": broadphase_overflow_edges,
            "naive_pairs": naive_pairs,
        },
        "edge-distance contact with deterministic spatial pruning",
    )
    evidence = profile.get("evidence", [])
    check(
        checks,
        "primary_research_provenance",
        len(evidence) >= 6 and all(item.get("url") and item.get("used_for") for item in evidence),
        len(evidence),
        "at least six traceable experimental/computational sources",
    )
    passed = all(item["passed"] for item in checks.values())
    return {
        "schema": "dr.anmar.suture-validation.v1",
        "profile_id": profile["id"],
        "passed": passed,
        "checks": checks,
        "derived": {
            "mass_kg": derived.mass_kg,
            "segment_mass_kg": derived.segment_mass_kg,
            "axial_rigidity_n": derived.axial_rigidity_n,
            "axial_joint_stiffness_n_m": derived.axial_joint_stiffness_n_m,
            "bend_joint_stiffness_n_m_rad": derived.bend_joint_stiffness_n_m_rad,
            "twist_joint_stiffness_n_m_rad": derived.twist_joint_stiffness_n_m_rad,
            "needle_arc_length_m": derived_needle.arc_length_m,
            "needle_curvature_radius_m": derived_needle.curvature_radius_m,
            "needle_body_diameter_m": 2.0 * derived_needle.body_radius_m,
            "needle_mass_kg": derived_needle.mass_kg,
            "needle_center_of_mass_m": derived_needle.mass_properties.center_of_mass_m,
            "needle_diagonal_inertia_kg_m2": derived_needle.mass_properties.diagonal_inertia_kg_m2,
            "needle_principal_axes_wxyz": derived_needle.mass_properties.principal_axes_wxyz,
            "needle_visual_vertex_count": derived_needle.visual_vertex_count,
            "needle_visual_normal_quality": needle_normal_quality,
            "needle_collision_capsule_count": derived_needle.collision_capsule_count,
            "sim_to_real_gap_count": len(gaps),
        },
        "clinical_validation": False,
        "note": "Deterministic engineering validation only; physical bench and clinician validation remain required.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument(
        "--needle-profile",
        type=Path,
        default=DEFAULT_NEEDLE_PROFILE_PATH,
    )
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument(
        "--suture-base",
        type=Path,
        default=DEFAULT_SUTURE_BASE,
    )
    parser.add_argument(
        "--suture-geometry",
        type=Path,
        default=DEFAULT_SUTURE_GEOMETRY,
    )
    parser.add_argument(
        "--suture-materials",
        type=Path,
        default=DEFAULT_SUTURE_MATERIALS,
    )
    parser.add_argument(
        "--suture-physics",
        type=Path,
        default=DEFAULT_SUTURE_PHYSICS,
    )
    parser.add_argument(
        "--suture-physx",
        type=Path,
        default=DEFAULT_SUTURE_PHYSX,
    )
    parser.add_argument(
        "--needle",
        "--assembly",
        dest="needle",
        type=Path,
        default=DEFAULT_NEEDLE,
    )
    parser.add_argument(
        "--needle-base",
        type=Path,
        default=DEFAULT_NEEDLE_BASE,
    )
    parser.add_argument(
        "--needle-geometry",
        type=Path,
        default=DEFAULT_NEEDLE_GEOMETRY,
    )
    parser.add_argument(
        "--needle-materials",
        type=Path,
        default=DEFAULT_NEEDLE_MATERIALS,
    )
    parser.add_argument(
        "--needle-physics",
        type=Path,
        default=DEFAULT_NEEDLE_PHYSICS,
    )
    parser.add_argument(
        "--needle-physx",
        type=Path,
        default=DEFAULT_NEEDLE_PHYSX,
    )
    parser.add_argument(
        "--usdcat",
        default=shutil.which("usdcat") or "usdcat",
    )
    parser.add_argument("--workstation", type=Path, default=DEFAULT_WORKSTATION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    needle_profile = load_needle_profile(args.needle_profile)
    suture_entry_text = args.asset.read_text(encoding="utf-8")
    suture_base_text = args.suture_base.read_text(encoding="utf-8")
    suture_geometry_text = read_usd_as_text(args.suture_geometry, args.usdcat)
    suture_geometry_is_usdc = args.suture_geometry.read_bytes()[:8] == b"PXR-USDC"
    suture_materials_text = args.suture_materials.read_text(encoding="utf-8")
    suture_physics_text = args.suture_physics.read_text(encoding="utf-8")
    suture_physx_text = args.suture_physx.read_text(encoding="utf-8")
    suture_variant_texts = {
        selection: compose_physics_variant(args.asset, selection, args.usdcat)
        for selection in ("none", "physics", "physx")
    }
    needle_entry_text = args.needle.read_text(encoding="utf-8")
    needle_base_text = args.needle_base.read_text(encoding="utf-8")
    needle_geometry_text = read_usd_as_text(args.needle_geometry, args.usdcat)
    needle_geometry_is_usdc = args.needle_geometry.read_bytes()[:8] == b"PXR-USDC"
    needle_materials_text = args.needle_materials.read_text(encoding="utf-8")
    needle_physics_text = args.needle_physics.read_text(encoding="utf-8")
    needle_physx_text = args.needle_physx.read_text(encoding="utf-8")
    needle_variant_texts = {
        selection: compose_physics_variant(args.needle, selection, args.usdcat)
        for selection in ("none", "physics", "physx")
    }
    workstation_text = args.workstation.read_text(encoding="utf-8")
    report = validate(
        profile,
        needle_profile,
        suture_entry_text,
        suture_base_text,
        suture_geometry_text,
        suture_geometry_is_usdc,
        suture_materials_text,
        suture_physics_text,
        suture_physx_text,
        suture_variant_texts,
        needle_entry_text,
        needle_base_text,
        needle_geometry_text,
        needle_geometry_is_usdc,
        needle_materials_text,
        needle_physics_text,
        needle_physx_text,
        needle_variant_texts,
        workstation_text,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
