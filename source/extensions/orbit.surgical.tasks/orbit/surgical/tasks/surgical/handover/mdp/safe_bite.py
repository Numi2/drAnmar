# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-derived retained-needle approach state for the T1 tissue task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from orbit.surgical.tasks.surgical import mdp_common

from .state import handover_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _contract(env: ManagerBasedRLEnv) -> dict[str, Any]:
    contract = getattr(env.cfg, "dr_anmar_safe_bite_contract", None)
    if not isinstance(contract, dict):
        raise TypeError("T1 requires cfg.dr_anmar_safe_bite_contract")
    return contract


def _step_number(env: ManagerBasedRLEnv) -> int:
    value = env.common_step_counter
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _tensor(
    values: list[float] | tuple[float, ...],
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=env.device)


def _cached_static_tensor(
    env: ManagerBasedRLEnv,
    key: tuple[Any, ...],
    factory,
) -> torch.Tensor:
    """Cache immutable task tensors once per environment instance."""

    cache = getattr(env, "_dr_anmar_safe_bite_static_tensors", None)
    if cache is None:
        cache = {}
        env._dr_anmar_safe_bite_static_tensors = cache
    tensor = cache.get(key)
    if tensor is None:
        tensor = factory()
        cache[key] = tensor
    return tensor


def _cached_vector(
    env: ManagerBasedRLEnv,
    name: str,
    values: list[float] | tuple[float, ...],
) -> torch.Tensor:
    resolved = tuple(map(float, values))
    return _cached_static_tensor(
        env,
        ("vector", name, resolved),
        lambda: torch.tensor(
            resolved,
            dtype=torch.float32,
            device=env.device,
        ),
    )


def _role_root_state(
    env: ManagerBasedRLEnv,
    giver_is_robot_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot_1: Articulation = env.scene["robot_1"]
    robot_2: Articulation = env.scene["robot_2"]
    receiver_is_robot_1 = ~giver_is_robot_1
    receiver_root_pos_w = torch.where(
        receiver_is_robot_1.unsqueeze(-1),
        mdp_common.as_torch(robot_1.data.root_pos_w),
        mdp_common.as_torch(robot_2.data.root_pos_w),
    )
    receiver_root_quat_w = torch.where(
        receiver_is_robot_1.unsqueeze(-1),
        mdp_common.as_torch(robot_1.data.root_quat_w),
        mdp_common.as_torch(robot_2.data.root_quat_w),
    )
    return receiver_root_pos_w, receiver_root_quat_w


def _role_tool_samples_w(
    env: ManagerBasedRLEnv,
    role_is_robot_1: torch.Tensor,
) -> torch.Tensor:
    """Sample both role-selected jaws from tool tip to link center."""

    body_id_cache = getattr(
        env,
        "_dr_anmar_safe_bite_jaw_body_ids",
        None,
    )
    if body_id_cache is None:
        body_id_cache = {}
        for robot_name in ("robot_1", "robot_2"):
            robot: Articulation = env.scene[robot_name]
            body_ids, body_names = robot.find_bodies(
                [
                    "psm_tool_gripper1_link",
                    "psm_tool_gripper2_link",
                ]
            )
            if len(body_ids) != 2:
                raise RuntimeError(
                    f"T1 expected two jaw bodies on {robot_name}, received {body_names}"
                )
            body_id_cache[robot_name] = body_ids
        env._dr_anmar_safe_bite_jaw_body_ids = body_id_cache

    robot_1: Articulation = env.scene["robot_1"]
    robot_2: Articulation = env.scene["robot_2"]
    robot_1_jaws_w = mdp_common.as_torch(robot_1.data.body_pos_w)[:, body_id_cache["robot_1"], :]
    robot_2_jaws_w = mdp_common.as_torch(robot_2.data.body_pos_w)[:, body_id_cache["robot_2"], :]
    role_jaws_w = torch.where(
        role_is_robot_1[:, None, None],
        robot_1_jaws_w,
        robot_2_jaws_w,
    )
    robot_1_tip_w = mdp_common.as_torch(env.scene["ee_1_frame"].data.target_pos_w)[:, 0, :]
    robot_2_tip_w = mdp_common.as_torch(env.scene["ee_2_frame"].data.target_pos_w)[:, 0, :]
    role_tip_w = torch.where(
        role_is_robot_1.unsqueeze(-1),
        robot_1_tip_w,
        robot_2_tip_w,
    )
    guard = _contract(env)["needle_frame"]["clearance_guard"]
    sample_count = int(guard["receiver_tool_segment_sample_count"])
    fractions = _cached_static_tensor(
        env,
        ("tool_segment_fractions", sample_count, role_tip_w.dtype),
        lambda: torch.linspace(
            0.0,
            1.0,
            sample_count,
            dtype=role_tip_w.dtype,
            device=env.device,
        ),
    )
    segments = role_tip_w[:, None, None, :] + (
        fractions[None, None, :, None] * (role_jaws_w[:, :, None, :] - role_tip_w[:, None, None, :])
    )
    return segments.reshape(env.num_envs, -1, 3)


def _needle_centerline_offsets(
    env: ManagerBasedRLEnv,
    contract: dict[str, Any],
) -> torch.Tensor:
    frame = contract["needle_frame"]
    tip = frame["tip_offset_in_needle_root_m"]
    center_x = float(tip[0])
    radius = abs(float(tip[1]))
    maximum_spacing = float(frame["clearance_guard"]["needle_centerline_max_spacing_m"])
    sample_count = max(
        3,
        math.ceil(math.pi * radius / maximum_spacing) + 1,
    )
    key = (
        "needle_centerline_offsets",
        center_x,
        radius,
        maximum_spacing,
        sample_count,
    )

    def build() -> torch.Tensor:
        angles = torch.linspace(
            -0.5 * math.pi,
            -1.5 * math.pi,
            sample_count,
            device=env.device,
        )
        offsets = torch.zeros(
            (sample_count, 3),
            dtype=torch.float32,
            device=env.device,
        )
        offsets[:, 0] = center_x + radius * torch.cos(angles)
        offsets[:, 1] = radius * torch.sin(angles)
        return offsets

    return _cached_static_tensor(env, key, build)


def _apply_quaternion_to_samples(
    quaternion: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    environment_count = quaternion.shape[0]
    sample_count = offsets.shape[0]
    expanded_quaternion = quaternion[:, None, :].expand(-1, sample_count, -1)
    expanded_offsets = offsets[None, :, :].expand(environment_count, -1, -1)
    return quat_apply(
        expanded_quaternion.reshape(-1, 4),
        expanded_offsets.reshape(-1, 3),
    ).reshape(environment_count, sample_count, 3)


def _material_coordinates_from_rest_surface(
    local_points: torch.Tensor,
    contract: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map tissue-local XY to the structured top-surface material chart."""

    geometry = contract["tissue_geometry"]
    lod = str(contract["scene"]["tissue_lod"])
    lod_contract = contract["tissue_lods"][lod]
    x = local_points[..., 0]
    y = local_points[..., 1]
    depth = float(geometry["depth_m"])
    width = float(geometry["overall_width_m"])
    gap = float(geometry["rest_wound_gap_m"])
    bevel = float(geometry["wound_bevel_m"])
    irregularity = float(geometry["wound_irregularity_amplitude_m"])
    wavelength = float(geometry["wound_irregularity_wavelength_m"])
    edge_power = float(lod_contract["wound_edge_refinement_power"])
    wound_offset = irregularity * torch.sin(2.0 * math.pi * (y + depth / 2.0) / wavelength)
    left_inner = -gap / 2.0 + wound_offset - bevel
    right_inner = gap / 2.0 + wound_offset + bevel
    left_outer = -width / 2.0
    right_outer = width / 2.0
    within_y = torch.abs(y) <= depth / 2.0
    left = within_y & (x >= left_outer) & (x <= left_inner)
    right = within_y & (x >= right_inner) & (x <= right_outer)
    inside = left | right
    component = torch.where(
        left,
        torch.zeros_like(x, dtype=torch.long),
        torch.ones_like(x, dtype=torch.long),
    )
    left_shaped = ((x - left_outer) / (left_inner - left_outer).clamp_min(1.0e-9)).clamp(0.0, 1.0)
    right_shaped = ((x - right_inner) / (right_outer - right_inner).clamp_min(1.0e-9)).clamp(
        0.0, 1.0
    )
    left_u = 1.0 - torch.pow(1.0 - left_shaped, 1.0 / edge_power)
    right_u = torch.pow(right_shaped, 1.0 / edge_power)
    u = torch.where(left, left_u, right_u)
    v = ((y + depth / 2.0) / depth).clamp(0.0, 1.0)
    return component, torch.stack((u, v), dim=-1), inside


def _normalized(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(
        vector,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-9)


def _orthonormal_plane_normal(
    desired_tip_direction: torch.Tensor,
    longitudinal_tangent: torch.Tensor,
    surface_normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project the longitudinal axis into the needle-tip normal plane."""

    residual = longitudinal_tangent - (
        torch.sum(
            longitudinal_tangent * desired_tip_direction,
            dim=-1,
            keepdim=True,
        )
        * desired_tip_direction
    )
    residual_norm = torch.linalg.vector_norm(
        residual,
        dim=-1,
        keepdim=True,
    )
    fallback = _normalized(
        torch.cross(
            surface_normal,
            desired_tip_direction,
            dim=-1,
        )
    )
    fallback_sign = torch.sign(torch.sum(fallback * longitudinal_tangent, dim=-1, keepdim=True))
    fallback_sign = torch.where(
        fallback_sign == 0.0,
        torch.ones_like(fallback_sign),
        fallback_sign,
    )
    fallback = fallback * fallback_sign
    valid = (
        (residual_norm.squeeze(-1) > 1.0e-6)
        & torch.isfinite(residual).all(dim=-1)
        & torch.isfinite(desired_tip_direction).all(dim=-1)
    )
    plane_normal = _normalized(torch.where(valid.unsqueeze(-1), residual, fallback))
    return plane_normal, valid


def _live_top_surface_from_material_coordinates(
    env: ManagerBasedRLEnv,
    component: torch.Tensor,
    material_uv: torch.Tensor,
    contract: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate live Newton nodes using the generated mesh's chart."""

    lod = str(contract["scene"]["tissue_lod"])
    lod_contract = contract["tissue_lods"][lod]
    cells_x = int(lod_contract["cells_per_flap_x"])
    cells_y = int(lod_contract["cells_y"])
    z_count = len(lod_contract["z_fractions"])
    original_shape = material_uv.shape[:-1]
    uv = material_uv.reshape(env.num_envs, -1, 2)
    components = component.reshape(env.num_envs, -1).long()
    scaled_u = uv[..., 0].clamp(0.0, 1.0) * cells_x
    scaled_v = uv[..., 1].clamp(0.0, 1.0) * cells_y
    cell_x = torch.floor(scaled_u).long().clamp(0, cells_x - 1)
    cell_y = torch.floor(scaled_v).long().clamp(0, cells_y - 1)
    fraction_u = (scaled_u - cell_x).clamp(0.0, 1.0)
    fraction_v = (scaled_v - cell_y).clamp(0.0, 1.0)
    row_width = cells_x + 1
    layer_width = (cells_y + 1) * row_width
    component_width = z_count * layer_width
    base = components * component_width + (z_count - 1) * layer_width + cell_y * row_width + cell_x
    indices = (base, base + 1, base + row_width, base + row_width + 1)
    tissue: DeformableObject = env.scene["tissue"]
    nodal_pos_w = mdp_common.as_torch(tissue.data.nodal_pos_w)

    def gather(index: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            nodal_pos_w,
            1,
            index.unsqueeze(-1).expand(-1, -1, 3),
        )

    p00, p10, p01, p11 = (gather(index) for index in indices)
    u = fraction_u.unsqueeze(-1)
    v = fraction_v.unsqueeze(-1)
    low = torch.lerp(p00, p10, u)
    high = torch.lerp(p01, p11, u)
    surface_w = torch.lerp(low, high, v)
    tangent_u_w = _normalized(torch.lerp(p10 - p00, p11 - p01, v))
    tangent_v_w = _normalized(torch.lerp(p01 - p00, p11 - p10, u))
    normal_w = _normalized(torch.cross(tangent_u_w, tangent_v_w, dim=-1))
    output_shape = (*original_shape, 3)
    return (
        surface_w.reshape(output_shape),
        normal_w.reshape(output_shape),
        tangent_u_w.reshape(output_shape),
        tangent_v_w.reshape(output_shape),
    )


def _live_surface_guard(
    env: ManagerBasedRLEnv,
    local_points: torch.Tensor,
    contract: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve a live nodal patch for a conservative precontact query."""

    component, material_uv, inside = _material_coordinates_from_rest_surface(
        local_points,
        contract,
    )
    surface_w, normal_w, _, _ = _live_top_surface_from_material_coordinates(
        env,
        component,
        material_uv,
        contract,
    )
    return surface_w, normal_w, inside


def _minimum_role_tool_clearance(
    env: ManagerBasedRLEnv,
    role_tool_samples_w: torch.Tensor,
    tissue_origin_w: torch.Tensor,
    contract: dict[str, Any],
) -> torch.Tensor:
    """Return conservative live-surface clearance for one role's jaws."""

    tool_local = role_tool_samples_w - tissue_origin_w[:, None, :]
    surface_w, surface_normal_w, inside = _live_surface_guard(
        env,
        tool_local,
        contract,
    )
    collision_radius = float(
        contract["needle_frame"]["clearance_guard"]["receiver_tool_collision_radius_m"]
    )
    sample_clearance = (
        torch.sum(
            (role_tool_samples_w - surface_w) * surface_normal_w,
            dim=-1,
        )
        - collision_radius
    )
    sample_clearance = torch.where(
        inside,
        sample_clearance,
        torch.full_like(sample_clearance, torch.inf),
    )
    return torch.amin(sample_clearance, dim=-1)


def _sample_entry_frames(
    env: ManagerBasedRLEnv,
    state: dict[str, Any],
    env_ids: torch.Tensor,
    contract: dict[str, Any],
) -> None:
    if not bool(env_ids.numel()):
        return
    sampling = contract["sampling"]
    geometry = contract["tissue_geometry"]
    count = int(env_ids.numel())
    draws = torch.rand((count, 4), device=env.device)
    component = (draws[:, 0] >= 0.5).long()
    bite_low, bite_high = map(float, sampling["bite_distance_from_wound_m"])
    stand_off_low, stand_off_high = map(float, sampling["stand_off_m"])
    angle_low_deg, angle_high_deg = map(float, sampling["entry_angle_from_surface_normal_deg"])
    depth = float(geometry["depth_m"])
    margin = float(sampling["longitudinal_end_margin_m"])
    y = (-depth / 2.0 + margin) + draws[:, 1] * (depth - 2.0 * margin)
    bite_distance = bite_low + draws[:, 2] * (bite_high - bite_low)
    stand_off = stand_off_low + draws[:, 3] * (stand_off_high - stand_off_low)
    # Reuse a deterministic nonlinear transform of the stand-off draw instead
    # of consuming another random stream and changing all seeded resets.
    angle_fraction = torch.frac(draws[:, 3] * 1.618033988749895)
    angle = torch.deg2rad(angle_low_deg + angle_fraction * (angle_high_deg - angle_low_deg))

    gap = float(geometry["rest_wound_gap_m"])
    bevel = float(geometry["wound_bevel_m"])
    irregularity = float(geometry["wound_irregularity_amplitude_m"])
    irregularity_wavelength = float(geometry["wound_irregularity_wavelength_m"])
    width = float(geometry["overall_width_m"])
    thickness = float(geometry["thickness_m"])
    topography = float(geometry["surface_topography_amplitude_m"])
    topography_wavelength = float(geometry["surface_topography_wavelength_m"])
    wound_offset = irregularity * torch.sin(
        2.0 * math.pi * (y + depth / 2.0) / irregularity_wavelength
    )
    left = component == 0
    wound_x = torch.where(
        left,
        -gap / 2.0 + wound_offset - bevel,
        gap / 2.0 + wound_offset + bevel,
    )
    x = wound_x + torch.where(left, -bite_distance, bite_distance)
    edge_power = float(
        contract["tissue_lods"][str(contract["scene"]["tissue_lod"])]["wound_edge_refinement_power"]
    )
    left_fraction = ((x + width / 2.0) / (wound_x + width / 2.0).clamp_min(1.0e-9)).clamp(0.0, 1.0)
    right_fraction = ((x - wound_x) / (width / 2.0 - wound_x).clamp_min(1.0e-9)).clamp(0.0, 1.0)
    material_u = torch.where(
        left,
        1.0 - torch.pow(1.0 - left_fraction, 1.0 / edge_power),
        torch.pow(right_fraction, 1.0 / edge_power),
    )
    material_v = ((y + depth / 2.0) / depth).clamp(0.0, 1.0)
    centrality = torch.clamp(
        1.0 - torch.abs(x) / max(width / 2.0, 1.0e-9),
        min=0.0,
    )
    surface_z = thickness / 2.0 + (
        topography
        * centrality
        * torch.sin(
            2.0 * math.pi * (y + depth / 2.0) / topography_wavelength
            + torch.where(
                left,
                torch.full_like(y, 0.35),
                torch.full_like(y, -0.35),
            )
        )
    )
    surface_local = torch.stack((x, y, surface_z), dim=-1)
    target_local = surface_local.clone()
    target_local[:, 2] += stand_off
    inward_x = torch.where(
        left,
        torch.ones_like(angle),
        -torch.ones_like(angle),
    )
    desired_tip_direction = torch.stack(
        (
            inward_x * torch.sin(angle),
            torch.zeros_like(angle),
            -torch.cos(angle),
        ),
        dim=-1,
    )
    desired_plane_normal = torch.zeros_like(desired_tip_direction)
    desired_plane_normal[:, 1] = 1.0
    state["component"][env_ids] = component
    state["bite_distance_m"][env_ids] = bite_distance
    state["stand_off_m"][env_ids] = stand_off
    state["entry_angle_rad"][env_ids] = angle
    state["material_uv"][env_ids] = torch.stack(
        (material_u, material_v),
        dim=-1,
    )
    state["surface_point_local"][env_ids] = surface_local
    state["target_tip_local"][env_ids] = target_local
    state["desired_tip_direction_w"][env_ids] = desired_tip_direction
    state["desired_plane_normal_w"][env_ids] = desired_plane_normal


def reset_safe_bite_from_handover_cache(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Restore only snapshots captured from completed physical handovers."""

    cache = getattr(env, "_dr_anmar_safe_bite_handover_cache", None)
    if cache is None:
        return
    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
    curriculum = _contract(env)["handover_snapshot_curriculum"]
    valid = cache["valid"][env_ids]
    stride = int(curriculum["full_chain_stride"])
    if curriculum["restore_schedule"] != "per_environment_rotating_quota":
        raise RuntimeError("unsupported T1 snapshot restore schedule")
    reset_attempt = cache["reset_attempt_count"][env_ids]
    force_full_chain = (env_ids + reset_attempt).remainder(stride) == 0
    restore = valid & ~force_full_chain
    cache["reset_attempt_count"][env_ids] += 1
    selected = env_ids[restore]
    episode_mask = torch.zeros(
        env.num_envs,
        dtype=torch.bool,
        device=env.device,
    )
    episode_mask[selected] = True
    pending = {
        "episode_mask": episode_mask,
        "giver_is_robot_1": cache["giver_is_robot_1"].clone(),
        "receiver_acquisition_offset_w": (cache["receiver_acquisition_offset_w"].clone()),
        "contact_grace_steps": int(curriculum["post_restore_contact_grace_steps"]),
    }
    env._dr_anmar_pending_safe_bite_restore = pending
    if not bool(selected.numel()):
        return
    robot_1: Articulation = env.scene["robot_1"]
    robot_2: Articulation = env.scene["robot_2"]
    obj: RigidObject = env.scene["object"]
    robot_1.write_joint_state_to_sim(
        cache["robot_1_joint_pos"][selected],
        cache["robot_1_joint_vel"][selected],
        env_ids=selected,
    )
    robot_2.write_joint_state_to_sim(
        cache["robot_2_joint_pos"][selected],
        cache["robot_2_joint_vel"][selected],
        env_ids=selected,
    )
    obj.write_root_pose_to_sim(
        cache["object_root_pose_w"][selected],
        env_ids=selected,
    )
    obj.write_root_velocity_to_sim(
        cache["object_root_velocity_w"][selected],
        env_ids=selected,
    )
    cache["restore_count"] += int(selected.numel())


def reset_tissue_outer_fixture(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Reset tissue and pin only its versioned outer attachment edges."""

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
    if not bool(env_ids.numel()):
        return
    from ..newton_contact_manager import (
        DrAnmarCoupledMJWarpVBDManager,
    )

    DrAnmarCoupledMJWarpVBDManager.invalidate_dranmar_contact_receipt(env_ids)
    tissue: DeformableObject = env.scene["tissue"]
    default_state = mdp_common.as_torch(tissue.data.default_nodal_state_w)[env_ids].clone()
    tissue.write_nodal_state_to_sim_index(
        default_state,
        env_ids=env_ids,
    )

    contract = _contract(env)
    geometry = contract["tissue_geometry"]
    fixture = contract["scene"]["fixture"]
    tissue_position = _tensor(contract["scene"]["tissue_position_in_environment_m"], env)
    tissue_origin_w = env.scene.env_origins[env_ids] + tissue_position
    default_position_w = default_state[..., :3]
    local_x = default_position_w[..., 0] - tissue_origin_w[:, None, 0]
    outer_boundary_x = float(geometry["overall_width_m"]) / 2.0
    attachment_width = float(contract["tissue_semantics"]["outer_attachment_width_m"])
    tolerance = float(fixture["selection_tolerance_m"])
    distance_from_outer_edge = outer_boundary_x - torch.abs(local_x)
    anchored = (distance_from_outer_edge >= -tolerance) & (
        distance_from_outer_edge <= attachment_width + tolerance
    )

    targets = torch.zeros(
        (*default_position_w.shape[:-1], 4),
        dtype=default_position_w.dtype,
        device=env.device,
    )
    targets[..., :3] = default_position_w
    targets[..., 3] = (~anchored).to(default_position_w.dtype)
    tissue.write_nodal_kinematic_target_to_sim_index(
        targets,
        env_ids=env_ids,
    )

    lod = str(contract["scene"]["tissue_lod"])
    expected = int(fixture["expected_anchor_nodes_by_lod"][lod])
    anchor_count = anchored.sum(dim=-1)
    if bool((anchor_count != expected).any()):
        counts = sorted(set(anchor_count.detach().cpu().tolist()))
        raise RuntimeError(
            "T1 tissue fixture resolved unexpected outer-anchor counts: "
            f"expected {expected}, received {counts}"
        )
    env._dr_anmar_tissue_fixture_anchor_count = anchor_count


def _capture_handover_snapshots(
    env: ManagerBasedRLEnv,
    state: dict[str, Any],
    handover: dict[str, Any],
) -> None:
    capture = (
        handover["successful_handover"]
        & ~state["handover_snapshot_captured"]
        & ~state["snapshot_initialized"]
    )
    if not bool(capture.any()):
        return
    robot_1: Articulation = env.scene["robot_1"]
    robot_2: Articulation = env.scene["robot_2"]
    obj: RigidObject = env.scene["object"]
    cache = getattr(env, "_dr_anmar_safe_bite_handover_cache", None)
    if cache is None:
        cache = {
            "valid": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "restore_count": 0,
            "capture_count": 0,
            "reset_attempt_count": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "giver_is_robot_1": torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
            "receiver_acquisition_offset_w": torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            ),
            "robot_1_joint_pos": torch.zeros_like(mdp_common.as_torch(robot_1.data.joint_pos)),
            "robot_1_joint_vel": torch.zeros_like(mdp_common.as_torch(robot_1.data.joint_vel)),
            "robot_2_joint_pos": torch.zeros_like(mdp_common.as_torch(robot_2.data.joint_pos)),
            "robot_2_joint_vel": torch.zeros_like(mdp_common.as_torch(robot_2.data.joint_vel)),
            "object_root_pose_w": torch.zeros(
                (env.num_envs, 7), dtype=torch.float32, device=env.device
            ),
            "object_root_velocity_w": torch.zeros(
                (env.num_envs, 6), dtype=torch.float32, device=env.device
            ),
        }
        env._dr_anmar_safe_bite_handover_cache = cache
    cache["valid"][capture] = True
    cache["giver_is_robot_1"][capture] = handover["giver_is_robot_1"][capture]
    cache["receiver_acquisition_offset_w"][capture] = handover["receiver_acquisition_offset_w"][
        capture
    ]
    cache["robot_1_joint_pos"][capture] = mdp_common.as_torch(robot_1.data.joint_pos)[capture]
    cache["robot_1_joint_vel"][capture] = mdp_common.as_torch(robot_1.data.joint_vel)[capture]
    cache["robot_2_joint_pos"][capture] = mdp_common.as_torch(robot_2.data.joint_pos)[capture]
    cache["robot_2_joint_vel"][capture] = mdp_common.as_torch(robot_2.data.joint_vel)[capture]
    cache["object_root_pose_w"][capture, :3] = mdp_common.as_torch(obj.data.root_pos_w)[capture]
    cache["object_root_pose_w"][capture, 3:7] = mdp_common.as_torch(obj.data.root_quat_w)[capture]
    cache["object_root_velocity_w"][capture, :3] = mdp_common.as_torch(obj.data.root_lin_vel_w)[
        capture
    ]
    cache["object_root_velocity_w"][capture, 3:6] = mdp_common.as_torch(obj.data.root_ang_vel_w)[
        capture
    ]
    cache["capture_count"] += int(capture.sum().item())
    state["handover_snapshot_captured"][capture] = True


def _native_soft_contact_receipt(
    env: ManagerBasedRLEnv,
    giver_is_robot_1: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Resolve the custom coupled-manager receipt without CPU copies."""

    from ..newton_contact_manager import (
        DrAnmarCoupledMJWarpVBDManager,
    )

    receipt = DrAnmarCoupledMJWarpVBDManager.get_dranmar_soft_contact_receipt()
    unavailable = torch.zeros(
        env.num_envs,
        dtype=torch.bool,
        device=env.device,
    )
    zero = torch.zeros(
        env.num_envs,
        dtype=torch.float32,
        device=env.device,
    )
    tissue_particle_count = int(env.scene["tissue"].data.nodal_pos_w.shape[1])
    unavailable_particles = torch.zeros(
        (env.num_envs, tissue_particle_count),
        dtype=torch.bool,
        device=env.device,
    )
    if receipt is None:
        return {
            "available": unavailable,
            "overflow": unavailable.clone(),
            "receipt_generation": torch.full(
                (env.num_envs,),
                -1,
                dtype=torch.long,
                device=env.device,
            ),
            "needle_penetration": unavailable.clone(),
            "receiver_tool_penetration": unavailable.clone(),
            "giver_tool_penetration": unavailable.clone(),
            "any_robot_penetration": unavailable.clone(),
            "any_relevant_penetration": unavailable.clone(),
            "needle_tip_particle_penetration_seen": (unavailable_particles),
            "maximum_penetration_m": zero,
        }
    penetration = receipt["penetration_seen"].bool()
    maximum = receipt["maximum_penetration_m"]
    classes = receipt["contact_classes"]
    if penetration.shape[0] != env.num_envs:
        raise RuntimeError(
            "DrAnmar soft-contact receipt environment count mismatch: "
            f"{penetration.shape[0]} != {env.num_envs}"
        )
    tip_particle = receipt["needle_tip_particle_penetration_seen"].bool()
    if tip_particle.shape != unavailable_particles.shape:
        raise RuntimeError(
            "DrAnmar tip-contact particle receipt shape mismatch: "
            f"{tuple(tip_particle.shape)} != "
            f"{tuple(unavailable_particles.shape)}"
        )
    generation = receipt["generation"].reshape(-1)[0]
    environment_generation = receipt["environment_generation"].long()
    fresh = (environment_generation == generation.long()) & (environment_generation >= 0)
    overflow_scalar = receipt["overflow_seen"].reshape(-1)[0].bool()
    overflow = torch.ones_like(fresh) & overflow_scalar
    available = fresh & ~overflow
    needle = penetration[:, int(classes["needle"])] & fresh
    robot_1 = penetration[:, int(classes["robot_1"])] & fresh
    robot_2 = penetration[:, int(classes["robot_2"])] & fresh
    receiver = torch.where(~giver_is_robot_1, robot_1, robot_2)
    giver = torch.where(giver_is_robot_1, robot_1, robot_2)
    maximum_receiver = torch.where(
        ~giver_is_robot_1,
        maximum[:, int(classes["robot_1"])],
        maximum[:, int(classes["robot_2"])],
    )
    maximum_giver = torch.where(
        giver_is_robot_1,
        maximum[:, int(classes["robot_1"])],
        maximum[:, int(classes["robot_2"])],
    )
    maximum_relevant_raw = torch.maximum(
        maximum[:, int(classes["needle"])],
        torch.maximum(maximum_receiver, maximum_giver),
    )
    maximum_relevant = torch.where(
        fresh,
        maximum_relevant_raw,
        torch.zeros_like(maximum_relevant_raw),
    )
    return {
        "available": available,
        "overflow": overflow,
        "receipt_generation": environment_generation,
        "needle_penetration": needle,
        "receiver_tool_penetration": receiver,
        "giver_tool_penetration": giver,
        "any_robot_penetration": robot_1 | robot_2,
        "any_relevant_penetration": needle | robot_1 | robot_2,
        "needle_tip_particle_penetration_seen": (tip_particle & fresh.unsqueeze(-1)),
        "maximum_penetration_m": maximum_relevant,
    }


def safe_bite_state(env: ManagerBasedRLEnv) -> dict[str, Any]:
    """Return sampled entry geometry and monotonic physical T1 state."""

    contract = _contract(env)
    success = contract["success"]
    handover = handover_state(env)
    step = _step_number(env)
    state = getattr(env, "_dr_anmar_safe_bite_state", None)
    if state is None:
        zeros = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        state = {
            "component": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "bite_distance_m": zeros.clone(),
            "stand_off_m": zeros.clone(),
            "entry_angle_rad": zeros.clone(),
            "material_uv": torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device),
            "surface_point_local": torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            ),
            "target_tip_local": torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            ),
            "desired_tip_direction_w": torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            ),
            "desired_plane_normal_w": torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            ),
            "stable_consecutive": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "contact_fresh_consecutive": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "contact_fresh_required": torch.ones(env.num_envs, dtype=torch.long, device=env.device),
            "last_contact_generation": torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=env.device
            ),
            "entry_armed": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "premature_tissue_contact": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "authorized_contact_transition": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "handover_snapshot_captured": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "snapshot_initialized": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "has_previous_error": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "previous_normalized_error": zeros.clone(),
            "last_reset_step": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
            "last_step": -1,
        }
        env._dr_anmar_safe_bite_state = state

    reset = (env.episode_length_buf == 0) & (state["last_reset_step"] != step)
    reset_ids = torch.nonzero(reset, as_tuple=False).squeeze(-1)
    if bool(reset_ids.numel()):
        _sample_entry_frames(env, state, reset_ids, contract)
        pending = getattr(env, "_dr_anmar_pending_safe_bite_restore", None)
        restored = (
            pending["episode_mask"][reset_ids]
            if isinstance(pending, dict)
            else torch.zeros_like(reset_ids, dtype=torch.bool)
        )
        state["snapshot_initialized"][reset_ids] = restored
        state["stable_consecutive"][reset_ids] = 0
        grace_steps = int(pending["contact_grace_steps"]) if isinstance(pending, dict) else 1
        state["contact_fresh_consecutive"][reset_ids] = 0
        state["contact_fresh_required"][reset_ids] = torch.where(
            restored,
            torch.full_like(reset_ids, max(1, grace_steps)),
            torch.ones_like(reset_ids),
        )
        state["last_contact_generation"][reset_ids] = -1
        state["entry_armed"][reset_ids] = False
        state["premature_tissue_contact"][reset_ids] = False
        state["authorized_contact_transition"][reset_ids] = False
        state["handover_snapshot_captured"][reset_ids] = False
        state["has_previous_error"][reset_ids] = False
        state["previous_normalized_error"][reset_ids] = 0.0
        state["last_reset_step"][reset_ids] = step
    if state["last_step"] == step and not bool(reset.any()):
        return state

    _capture_handover_snapshots(env, state, handover)
    obj: RigidObject = env.scene["object"]
    object_pos_w = mdp_common.as_torch(obj.data.root_pos_w)
    object_quat_w = mdp_common.as_torch(obj.data.root_quat_w)
    frame = contract["needle_frame"]
    tip_offset = _cached_vector(
        env,
        "needle_tip_offset",
        frame["tip_offset_in_needle_root_m"],
    )
    tip_forward_axis = _cached_vector(
        env,
        "needle_tip_forward_axis",
        frame["tip_forward_axis_in_needle_root"],
    )
    plane_normal_axis = _cached_vector(
        env,
        "needle_plane_normal_axis",
        frame["needle_plane_normal_in_needle_root"],
    )
    tip_pos_w = object_pos_w + quat_apply(
        object_quat_w,
        tip_offset.unsqueeze(0).expand(env.num_envs, -1),
    )
    tip_direction_w = quat_apply(
        object_quat_w,
        tip_forward_axis.unsqueeze(0).expand(env.num_envs, -1),
    )
    plane_normal_w = quat_apply(
        object_quat_w,
        plane_normal_axis.unsqueeze(0).expand(env.num_envs, -1),
    )
    tissue_position = _cached_vector(
        env,
        "tissue_position",
        contract["scene"]["tissue_position_in_environment_m"],
    )
    tissue_origin_w = env.scene.env_origins + tissue_position
    (
        live_surface_w,
        live_surface_normal_w,
        live_tangent_u_w,
        live_tangent_v_w,
    ) = _live_top_surface_from_material_coordinates(
        env,
        state["component"],
        state["material_uv"],
        contract,
    )
    inward_tangent_w = torch.where(
        (state["component"] == 0).unsqueeze(-1),
        live_tangent_u_w,
        -live_tangent_u_w,
    )
    entry_angle = state["entry_angle_rad"].unsqueeze(-1)
    desired_tip_direction_w = _normalized(
        inward_tangent_w * torch.sin(entry_angle) - live_surface_normal_w * torch.cos(entry_angle)
    )
    desired_plane_normal_w, plane_frame_valid = _orthonormal_plane_normal(
        desired_tip_direction_w,
        live_tangent_v_w,
        live_surface_normal_w,
    )
    live_frame_valid = (
        plane_frame_valid
        & (torch.linalg.vector_norm(live_tangent_u_w, dim=-1) > 0.99)
        & (torch.linalg.vector_norm(live_tangent_v_w, dim=-1) > 0.99)
        & (torch.linalg.vector_norm(live_surface_normal_w, dim=-1) > 0.99)
    )
    desired_frame_orthogonality_error = torch.abs(
        torch.sum(
            desired_tip_direction_w * desired_plane_normal_w,
            dim=-1,
        )
    )
    live_frame_valid &= desired_frame_orthogonality_error <= 1.0e-5
    target_tip_w = live_surface_w + live_surface_normal_w * state["stand_off_m"].unsqueeze(-1)
    state["surface_point_local"][:] = live_surface_w - tissue_origin_w
    state["target_tip_local"][:] = target_tip_w - tissue_origin_w
    state["desired_tip_direction_w"][:] = desired_tip_direction_w
    state["desired_plane_normal_w"][:] = desired_plane_normal_w
    position_error_w = target_tip_w - tip_pos_w
    position_error = torch.linalg.vector_norm(position_error_w, dim=-1)
    tangent_dot = torch.sum(tip_direction_w * state["desired_tip_direction_w"], dim=-1).clamp(
        -1.0, 1.0
    )
    tangent_error = torch.acos(tangent_dot)
    plane_dot = torch.abs(
        torch.sum(plane_normal_w * state["desired_plane_normal_w"], dim=-1)
    ).clamp(0.0, 1.0)
    plane_error = torch.acos(plane_dot)

    receiver_root_pos_w, receiver_root_quat_w = _role_root_state(env, handover["giver_is_robot_1"])
    position_error_receiver = quat_apply_inverse(
        receiver_root_quat_w,
        position_error_w,
    )
    desired_tip_direction_receiver = quat_apply_inverse(
        receiver_root_quat_w,
        state["desired_tip_direction_w"],
    )
    tip_direction_receiver = quat_apply_inverse(
        receiver_root_quat_w,
        tip_direction_w,
    )
    desired_plane_normal_receiver = quat_apply_inverse(
        receiver_root_quat_w,
        state["desired_plane_normal_w"],
    )
    plane_normal_receiver = quat_apply_inverse(
        receiver_root_quat_w,
        plane_normal_w,
    )
    tangent_rotation_error_receiver = torch.cross(
        tip_direction_receiver,
        desired_tip_direction_receiver,
        dim=-1,
    )
    plane_sign = torch.sign(
        torch.sum(
            plane_normal_receiver * desired_plane_normal_receiver,
            dim=-1,
        )
    )
    plane_sign = torch.where(plane_sign == 0.0, torch.ones_like(plane_sign), plane_sign)
    plane_rotation_error_receiver = torch.cross(
        plane_normal_receiver,
        desired_plane_normal_receiver * plane_sign.unsqueeze(-1),
        dim=-1,
    )

    offsets = _needle_centerline_offsets(env, contract)
    needle_samples_w = object_pos_w[:, None, :] + (
        _apply_quaternion_to_samples(object_quat_w, offsets)
    )
    needle_samples_local = needle_samples_w - tissue_origin_w[:, None, :]
    (
        needle_surface_w,
        needle_surface_normal_w,
        inside_tissue,
    ) = _live_surface_guard(
        env,
        needle_samples_local,
        contract,
    )
    collision_radius = float(frame["body_collision_radius_m"])
    sample_clearance = (
        torch.sum(
            (needle_samples_w - needle_surface_w) * needle_surface_normal_w,
            dim=-1,
        )
        - collision_radius
    )
    sample_clearance = torch.where(
        inside_tissue,
        sample_clearance,
        torch.full_like(sample_clearance, torch.inf),
    )
    minimum_needle_clearance = torch.amin(sample_clearance, dim=-1)

    receiver_tool_samples_w = _role_tool_samples_w(
        env,
        ~handover["giver_is_robot_1"],
    )
    giver_tool_samples_w = _role_tool_samples_w(
        env,
        handover["giver_is_robot_1"],
    )
    receiver_tool_clearance = _minimum_role_tool_clearance(
        env,
        receiver_tool_samples_w,
        tissue_origin_w,
        contract,
    )
    giver_tool_clearance = _minimum_role_tool_clearance(
        env,
        giver_tool_samples_w,
        tissue_origin_w,
        contract,
    )
    minimum_tool_clearance = torch.minimum(
        receiver_tool_clearance,
        giver_tool_clearance,
    )

    contact_clearance = float(success["minimum_tissue_clearance_m"])
    tool_clearance = float(success["minimum_tool_clearance_m"])
    live_surface_guard_contact = (minimum_needle_clearance <= contact_clearance) | (
        minimum_tool_clearance <= tool_clearance
    )
    native_contact = _native_soft_contact_receipt(
        env,
        handover["giver_is_robot_1"],
    )
    receipt_generation_advanced = native_contact["available"] & (
        native_contact["receipt_generation"] != state["last_contact_generation"]
    )
    state["contact_fresh_consecutive"][:] = torch.where(
        receipt_generation_advanced,
        state["contact_fresh_consecutive"] + 1,
        torch.where(
            native_contact["available"],
            state["contact_fresh_consecutive"],
            torch.zeros_like(state["contact_fresh_consecutive"]),
        ),
    )
    state["last_contact_generation"][:] = torch.where(
        receipt_generation_advanced,
        native_contact["receipt_generation"],
        state["last_contact_generation"],
    )
    contact_authority_ready = native_contact["available"] & (
        state["contact_fresh_consecutive"] >= state["contact_fresh_required"]
    )
    tissue_nodal_pos_w = mdp_common.as_torch(env.scene["tissue"].data.nodal_pos_w)
    entry_contact_radius = float(contract["puncture_transition"]["entry_contact_roi_radius_m"])
    entry_particle_roi = (
        torch.linalg.vector_norm(
            tissue_nodal_pos_w - live_surface_w[:, None, :],
            dim=-1,
        )
        <= entry_contact_radius
    )
    native_tip_entry_contact = torch.any(
        native_contact["needle_tip_particle_penetration_seen"] & entry_particle_roi,
        dim=-1,
    )
    tissue_contact = (
        live_surface_guard_contact
        | native_contact["any_relevant_penetration"]
        | native_contact["overflow"]
    )
    object_com_linear_velocity_w = mdp_common.as_torch(obj.data.root_com_lin_vel_w)
    object_com_angular_velocity_w = mdp_common.as_torch(obj.data.root_com_ang_vel_w)
    object_linear_speed = torch.linalg.vector_norm(
        object_com_linear_velocity_w,
        dim=-1,
    )
    object_angular_speed = torch.linalg.vector_norm(
        object_com_angular_velocity_w,
        dim=-1,
    )
    ready_now = (
        handover["successful_handover"]
        & handover["receiver_contact_now"]
        & live_frame_valid
        & contact_authority_ready
        & (position_error <= float(success["position_tolerance_m"]))
        & (tangent_error <= float(success["tip_tangent_tolerance_rad"]))
        & (plane_error <= float(success["needle_plane_tolerance_rad"]))
        & (object_linear_speed <= float(success["linear_speed_limit_m_s"]))
        & (object_angular_speed <= float(success["angular_speed_limit_rad_s"]))
        & ~tissue_contact
    )
    state["stable_consecutive"][:] = torch.where(
        ready_now,
        state["stable_consecutive"] + 1,
        torch.zeros_like(state["stable_consecutive"]),
    )
    state["entry_armed"] |= state["stable_consecutive"] >= int(success["stable_control_steps"])
    state["premature_tissue_contact"] |= tissue_contact & ~state["entry_armed"]
    tip_offset_w = quat_apply(
        object_quat_w,
        tip_offset.unsqueeze(0).expand(env.num_envs, -1),
    )
    object_root_linear_velocity_w = mdp_common.as_torch(obj.data.root_link_lin_vel_w)
    object_root_angular_velocity_w = mdp_common.as_torch(obj.data.root_link_ang_vel_w)
    tip_velocity_w = object_root_linear_velocity_w + torch.cross(
        object_root_angular_velocity_w,
        tip_offset_w,
        dim=-1,
    )
    inward_speed = torch.sum(
        tip_velocity_w * state["desired_tip_direction_w"],
        dim=-1,
    )
    state["authorized_contact_transition"] |= (
        state["entry_armed"]
        & contact_authority_ready
        & native_tip_entry_contact
        & (inward_speed > 0.0)
    )

    normalized_error = (
        position_error / float(success["position_tolerance_m"])
        + tangent_error / float(success["tip_tangent_tolerance_rad"])
        + plane_error / float(success["needle_plane_tolerance_rad"])
    ) / 3.0
    state.update(
        {
            "last_step": step,
            "target_tip_w": target_tip_w,
            "tip_pos_w": tip_pos_w,
            "tip_direction_w": tip_direction_w,
            "plane_normal_w": plane_normal_w,
            "position_error_w": position_error_w,
            "position_error_receiver": position_error_receiver,
            "desired_tip_direction_receiver": (desired_tip_direction_receiver),
            "tangent_rotation_error_receiver": (tangent_rotation_error_receiver),
            "plane_rotation_error_receiver": (plane_rotation_error_receiver),
            "position_error": position_error,
            "tangent_error": tangent_error,
            "plane_error": plane_error,
            "minimum_needle_clearance": minimum_needle_clearance,
            "receiver_tool_clearance": receiver_tool_clearance,
            "giver_tool_clearance": giver_tool_clearance,
            "minimum_tool_clearance": minimum_tool_clearance,
            "tissue_contact": tissue_contact,
            "live_surface_guard_contact": live_surface_guard_contact,
            "live_frame_valid": live_frame_valid,
            "desired_frame_orthogonality_error": (desired_frame_orthogonality_error),
            "contact_authority_ready": contact_authority_ready,
            "contact_fresh_consecutive": (state["contact_fresh_consecutive"]),
            "native_contact_available": native_contact["available"],
            "native_contact_overflow": native_contact["overflow"],
            "native_needle_penetration": (native_contact["needle_penetration"]),
            "native_tip_entry_contact": native_tip_entry_contact,
            "native_receiver_tool_penetration": (native_contact["receiver_tool_penetration"]),
            "native_giver_tool_penetration": (native_contact["giver_tool_penetration"]),
            "native_maximum_penetration_m": (native_contact["maximum_penetration_m"]),
            "ready_now": ready_now,
            "normalized_error": normalized_error,
            "object_linear_speed": object_linear_speed,
            "object_angular_speed": object_angular_speed,
            "tip_velocity_w": tip_velocity_w,
            "inward_speed": inward_speed,
            "receiver_root_pos_w": receiver_root_pos_w,
        }
    )
    return state


def safe_bite_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Goal-conditioned servo error plus auditable physical diagnostics."""

    state = safe_bite_state(env)
    contract = _contract(env)
    success = contract["success"]
    stable_steps = float(success["stable_control_steps"])
    servo = torch.cat(
        (
            (state["position_error_receiver"] / 0.02).clamp(-5.0, 5.0),
            state["tangent_rotation_error_receiver"].clamp(-1.0, 1.0),
            state["plane_rotation_error_receiver"].clamp(-1.0, 1.0),
            handover_state(env)["successful_handover"].float().unsqueeze(-1),
            state["entry_armed"].float().unsqueeze(-1),
            state["tissue_contact"].float().unsqueeze(-1),
            state["desired_tip_direction_receiver"].clamp(-1.0, 1.0),
        ),
        dim=-1,
    )
    diagnostics = torch.stack(
        (
            (state["position_error"] / float(success["position_tolerance_m"])).clamp(0.0, 5.0),
            (state["tangent_error"] / math.pi).clamp(0.0, 1.0),
            (state["plane_error"] / math.pi).clamp(0.0, 1.0),
            (state["minimum_needle_clearance"] / 0.01).clamp(-5.0, 5.0),
            (state["minimum_tool_clearance"] / 0.01).clamp(-5.0, 5.0),
            (state["stable_consecutive"].float() / stable_steps).clamp(0.0, 1.0),
            state["snapshot_initialized"].float(),
            state["authorized_contact_transition"].float(),
        ),
        dim=-1,
    )
    return torch.cat((servo, diagnostics), dim=-1)


def safe_bite_approach_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward error reduction once; stationary occupancy returns exactly zero."""

    state = safe_bite_state(env)
    handover = handover_state(env)
    active = (
        handover["successful_handover"]
        & handover["receiver_contact_now"]
        & state["contact_authority_ready"]
        & state["live_frame_valid"]
        & ~state["entry_armed"]
    )
    has_previous = state["has_previous_error"]
    progress = state["previous_normalized_error"] - state["normalized_error"]
    progress = torch.where(
        active & has_previous,
        progress,
        torch.zeros_like(progress),
    )
    clip = float(_contract(env)["rewards"]["approach_progress_clip_per_step"])
    progress = progress.clamp(-clip, clip)
    state["previous_normalized_error"][:] = state["normalized_error"]
    state["has_previous_error"][:] = active
    return progress


def safe_bite_entry_armed(env: ManagerBasedRLEnv) -> torch.Tensor:
    return safe_bite_state(env)["entry_armed"]


def safe_bite_premature_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    return safe_bite_state(env)["premature_tissue_contact"]


def safe_bite_authorized_contact_transition(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return safe_bite_state(env)["authorized_contact_transition"].float()
