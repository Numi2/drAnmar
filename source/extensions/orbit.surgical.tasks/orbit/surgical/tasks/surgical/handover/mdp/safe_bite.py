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


_NEEDLE_CENTERLINE_SAMPLE_COUNT = 13


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


def _receiver_tool_samples_w(
    env: ManagerBasedRLEnv,
    giver_is_robot_1: torch.Tensor,
) -> torch.Tensor:
    """Sample both receiver jaws from tool tip to link center."""

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
                    f"T1 expected two jaw bodies on {robot_name}, "
                    f"received {body_names}"
                )
            body_id_cache[robot_name] = body_ids
        env._dr_anmar_safe_bite_jaw_body_ids = body_id_cache

    robot_1: Articulation = env.scene["robot_1"]
    robot_2: Articulation = env.scene["robot_2"]
    robot_1_jaws_w = mdp_common.as_torch(
        robot_1.data.body_pos_w
    )[:, body_id_cache["robot_1"], :]
    robot_2_jaws_w = mdp_common.as_torch(
        robot_2.data.body_pos_w
    )[:, body_id_cache["robot_2"], :]
    receiver_jaws_w = torch.where(
        (~giver_is_robot_1)[:, None, None],
        robot_1_jaws_w,
        robot_2_jaws_w,
    )
    robot_1_tip_w = mdp_common.as_torch(
        env.scene["ee_1_frame"].data.target_pos_w
    )[:, 0, :]
    robot_2_tip_w = mdp_common.as_torch(
        env.scene["ee_2_frame"].data.target_pos_w
    )[:, 0, :]
    receiver_tip_w = torch.where(
        (~giver_is_robot_1).unsqueeze(-1),
        robot_1_tip_w,
        robot_2_tip_w,
    )
    fractions = torch.tensor(
        [0.0, 0.5, 1.0],
        dtype=receiver_tip_w.dtype,
        device=env.device,
    )
    segments = receiver_tip_w[:, None, None, :] + (
        fractions[None, None, :, None]
        * (receiver_jaws_w[:, :, None, :] - receiver_tip_w[:, None, None, :])
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
    angles = torch.linspace(
        -0.5 * math.pi,
        -1.5 * math.pi,
        _NEEDLE_CENTERLINE_SAMPLE_COUNT,
        device=env.device,
    )
    offsets = torch.zeros(
        (_NEEDLE_CENTERLINE_SAMPLE_COUNT, 3),
        dtype=torch.float32,
        device=env.device,
    )
    offsets[:, 0] = center_x + radius * torch.cos(angles)
    offsets[:, 1] = radius * torch.sin(angles)
    return offsets


def _apply_quaternion_to_samples(
    quaternion: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    environment_count = quaternion.shape[0]
    sample_count = offsets.shape[0]
    expanded_quaternion = quaternion[:, None, :].expand(
        -1, sample_count, -1
    )
    expanded_offsets = offsets[None, :, :].expand(
        environment_count, -1, -1
    )
    return quat_apply(
        expanded_quaternion.reshape(-1, 4),
        expanded_offsets.reshape(-1, 3),
    ).reshape(environment_count, sample_count, 3)


def _surface_geometry(
    local_points: torch.Tensor,
    geometry: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return conservative top-surface height and tissue-footprint membership."""

    x = local_points[..., 0]
    y = local_points[..., 1]
    depth = float(geometry["depth_m"])
    width = float(geometry["overall_width_m"])
    gap = float(geometry["rest_wound_gap_m"])
    bevel = float(geometry["wound_bevel_m"])
    irregularity = float(geometry["wound_irregularity_amplitude_m"])
    irregularity_wavelength = float(
        geometry["wound_irregularity_wavelength_m"]
    )
    thickness = float(geometry["thickness_m"])
    topography = float(geometry["surface_topography_amplitude_m"])
    topography_wavelength = float(
        geometry["surface_topography_wavelength_m"]
    )
    wound_offset = irregularity * torch.sin(
        2.0 * math.pi * (y + depth / 2.0) / irregularity_wavelength
    )
    left_inner = -gap / 2.0 + wound_offset - bevel
    right_inner = gap / 2.0 + wound_offset + bevel
    within_y = torch.abs(y) <= depth / 2.0
    left = within_y & (x >= -width / 2.0) & (x <= left_inner)
    right = within_y & (x >= right_inner) & (x <= width / 2.0)
    inside = left | right
    centrality = torch.clamp(
        1.0 - torch.abs(x) / max(width / 2.0, 1.0e-9),
        min=0.0,
    )
    phase = torch.where(
        left,
        torch.full_like(x, 0.35),
        torch.full_like(x, -0.35),
    )
    surface_z = thickness / 2.0 + (
        topography
        * centrality
        * torch.sin(
            2.0 * math.pi * (y + depth / 2.0) / topography_wavelength
            + phase
        )
    )
    return surface_z, inside


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
    bite_low, bite_high = map(
        float, sampling["bite_distance_from_wound_m"]
    )
    stand_off_low, stand_off_high = map(float, sampling["stand_off_m"])
    angle_low_deg, angle_high_deg = map(
        float, sampling["entry_angle_from_surface_normal_deg"]
    )
    depth = float(geometry["depth_m"])
    margin = float(sampling["longitudinal_end_margin_m"])
    y = (-depth / 2.0 + margin) + draws[:, 1] * (
        depth - 2.0 * margin
    )
    bite_distance = bite_low + draws[:, 2] * (bite_high - bite_low)
    stand_off = stand_off_low + draws[:, 3] * (
        stand_off_high - stand_off_low
    )
    # Reuse a deterministic nonlinear transform of the stand-off draw instead
    # of consuming another random stream and changing all seeded resets.
    angle_fraction = torch.frac(draws[:, 3] * 1.618033988749895)
    angle = torch.deg2rad(
        angle_low_deg + angle_fraction * (angle_high_deg - angle_low_deg)
    )

    gap = float(geometry["rest_wound_gap_m"])
    bevel = float(geometry["wound_bevel_m"])
    irregularity = float(geometry["wound_irregularity_amplitude_m"])
    irregularity_wavelength = float(
        geometry["wound_irregularity_wavelength_m"]
    )
    width = float(geometry["overall_width_m"])
    thickness = float(geometry["thickness_m"])
    topography = float(geometry["surface_topography_amplitude_m"])
    topography_wavelength = float(
        geometry["surface_topography_wavelength_m"]
    )
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
    force_full_chain = (
        (env_ids + reset_attempt).remainder(stride) == 0
    )
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
        "receiver_acquisition_offset_w": (
            cache["receiver_acquisition_offset_w"].clone()
        ),
        "contact_grace_steps": int(
            curriculum["post_restore_contact_grace_steps"]
        ),
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
    tissue: DeformableObject = env.scene["tissue"]
    default_state = mdp_common.as_torch(
        tissue.data.default_nodal_state_w
    )[env_ids].clone()
    tissue.write_nodal_state_to_sim_index(
        default_state,
        env_ids=env_ids,
    )

    contract = _contract(env)
    geometry = contract["tissue_geometry"]
    fixture = contract["scene"]["fixture"]
    tissue_position = _tensor(
        contract["scene"]["tissue_position_in_environment_m"], env
    )
    tissue_origin_w = env.scene.env_origins[env_ids] + tissue_position
    default_position_w = default_state[..., :3]
    local_x = default_position_w[..., 0] - tissue_origin_w[:, None, 0]
    outer_boundary_x = float(geometry["overall_width_m"]) / 2.0
    tolerance = float(fixture["outer_edge_tolerance_m"])
    anchored = torch.abs(torch.abs(local_x) - outer_boundary_x) <= tolerance

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

    expected = int(fixture["expected_training_lod_anchor_nodes"])
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
            "valid": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "restore_count": 0,
            "capture_count": 0,
            "reset_attempt_count": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "giver_is_robot_1": torch.ones(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "receiver_acquisition_offset_w": torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            ),
            "robot_1_joint_pos": torch.zeros_like(
                mdp_common.as_torch(robot_1.data.joint_pos)
            ),
            "robot_1_joint_vel": torch.zeros_like(
                mdp_common.as_torch(robot_1.data.joint_vel)
            ),
            "robot_2_joint_pos": torch.zeros_like(
                mdp_common.as_torch(robot_2.data.joint_pos)
            ),
            "robot_2_joint_vel": torch.zeros_like(
                mdp_common.as_torch(robot_2.data.joint_vel)
            ),
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
    cache["receiver_acquisition_offset_w"][capture] = handover[
        "receiver_acquisition_offset_w"
    ][capture]
    cache["robot_1_joint_pos"][capture] = mdp_common.as_torch(
        robot_1.data.joint_pos
    )[capture]
    cache["robot_1_joint_vel"][capture] = mdp_common.as_torch(
        robot_1.data.joint_vel
    )[capture]
    cache["robot_2_joint_pos"][capture] = mdp_common.as_torch(
        robot_2.data.joint_pos
    )[capture]
    cache["robot_2_joint_vel"][capture] = mdp_common.as_torch(
        robot_2.data.joint_vel
    )[capture]
    cache["object_root_pose_w"][capture, :3] = mdp_common.as_torch(
        obj.data.root_pos_w
    )[capture]
    cache["object_root_pose_w"][capture, 3:7] = mdp_common.as_torch(
        obj.data.root_quat_w
    )[capture]
    cache["object_root_velocity_w"][capture, :3] = mdp_common.as_torch(
        obj.data.root_lin_vel_w
    )[capture]
    cache["object_root_velocity_w"][capture, 3:6] = mdp_common.as_torch(
        obj.data.root_ang_vel_w
    )[capture]
    cache["capture_count"] += int(capture.sum().item())
    state["handover_snapshot_captured"][capture] = True


def safe_bite_state(env: ManagerBasedRLEnv) -> dict[str, Any]:
    """Return sampled entry geometry and monotonic physical T1 state."""

    contract = _contract(env)
    success = contract["success"]
    handover = handover_state(env)
    step = _step_number(env)
    state = getattr(env, "_dr_anmar_safe_bite_state", None)
    if state is None:
        zeros = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        state = {
            "component": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "bite_distance_m": zeros.clone(),
            "stand_off_m": zeros.clone(),
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
            "stable_consecutive": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "entry_armed": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "premature_tissue_contact": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "authorized_contact_transition": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "handover_snapshot_captured": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "snapshot_initialized": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "has_previous_error": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "previous_normalized_error": zeros.clone(),
            "last_reset_step": torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=env.device
            ),
            "last_step": -1,
        }
        env._dr_anmar_safe_bite_state = state

    reset = (env.episode_length_buf == 0) & (state["last_reset_step"] != step)
    reset_ids = torch.nonzero(reset, as_tuple=False).squeeze(-1)
    if bool(reset_ids.numel()):
        _sample_entry_frames(env, state, reset_ids, contract)
        pending = getattr(
            env, "_dr_anmar_pending_safe_bite_restore", None
        )
        restored = (
            pending["episode_mask"][reset_ids]
            if isinstance(pending, dict)
            else torch.zeros_like(reset_ids, dtype=torch.bool)
        )
        state["snapshot_initialized"][reset_ids] = restored
        state["stable_consecutive"][reset_ids] = 0
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
    tip_offset = _tensor(frame["tip_offset_in_needle_root_m"], env)
    tip_forward_axis = _tensor(
        frame["tip_forward_axis_in_needle_root"], env
    )
    plane_normal_axis = _tensor(
        frame["needle_plane_normal_in_needle_root"], env
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
    tissue_position = _tensor(
        contract["scene"]["tissue_position_in_environment_m"], env
    )
    tissue_origin_w = env.scene.env_origins + tissue_position
    target_tip_w = tissue_origin_w + state["target_tip_local"]
    position_error_w = target_tip_w - tip_pos_w
    position_error = torch.linalg.vector_norm(position_error_w, dim=-1)
    tangent_dot = torch.sum(
        tip_direction_w * state["desired_tip_direction_w"], dim=-1
    ).clamp(-1.0, 1.0)
    tangent_error = torch.acos(tangent_dot)
    plane_dot = torch.abs(
        torch.sum(
            plane_normal_w * state["desired_plane_normal_w"], dim=-1
        )
    ).clamp(0.0, 1.0)
    plane_error = torch.acos(plane_dot)

    receiver_root_pos_w, receiver_root_quat_w = _role_root_state(
        env, handover["giver_is_robot_1"]
    )
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
    plane_sign = torch.where(
        plane_sign == 0.0, torch.ones_like(plane_sign), plane_sign
    )
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
    surface_z, inside_tissue = _surface_geometry(
        needle_samples_local,
        contract["tissue_geometry"],
    )
    collision_radius = float(frame["body_collision_radius_m"])
    sample_clearance = (
        needle_samples_local[..., 2] - surface_z - collision_radius
    )
    sample_clearance = torch.where(
        inside_tissue,
        sample_clearance,
        torch.full_like(sample_clearance, torch.inf),
    )
    minimum_needle_clearance = torch.amin(sample_clearance, dim=-1)

    receiver_tool_samples_w = _receiver_tool_samples_w(
        env,
        handover["giver_is_robot_1"],
    )
    receiver_tool_local = (
        receiver_tool_samples_w - tissue_origin_w[:, None, :]
    )
    tool_surface_z, tool_inside = _surface_geometry(
        receiver_tool_local,
        contract["tissue_geometry"],
    )
    receiver_tool_sample_clearance = (
        receiver_tool_local[..., 2] - tool_surface_z
    )
    receiver_tool_sample_clearance = torch.where(
        tool_inside,
        receiver_tool_sample_clearance,
        torch.full_like(receiver_tool_sample_clearance, torch.inf),
    )
    receiver_tool_clearance = torch.amin(
        receiver_tool_sample_clearance,
        dim=-1,
    )

    contact_clearance = float(success["minimum_tissue_clearance_m"])
    tool_clearance = float(success["minimum_tool_clearance_m"])
    tissue_contact = (
        (minimum_needle_clearance <= contact_clearance)
        | (receiver_tool_clearance <= tool_clearance)
    )
    object_linear_speed = torch.linalg.vector_norm(
        mdp_common.as_torch(obj.data.root_lin_vel_w), dim=-1
    )
    object_angular_speed = torch.linalg.vector_norm(
        mdp_common.as_torch(obj.data.root_ang_vel_w), dim=-1
    )
    ready_now = (
        handover["successful_handover"]
        & handover["receiver_contact_now"]
        & (position_error <= float(success["position_tolerance_m"]))
        & (tangent_error <= float(success["tip_tangent_tolerance_rad"]))
        & (plane_error <= float(success["needle_plane_tolerance_rad"]))
        & (
            object_linear_speed
            <= float(success["linear_speed_limit_m_s"])
        )
        & (
            object_angular_speed
            <= float(success["angular_speed_limit_rad_s"])
        )
        & ~tissue_contact
    )
    state["stable_consecutive"][:] = torch.where(
        ready_now,
        state["stable_consecutive"] + 1,
        torch.zeros_like(state["stable_consecutive"]),
    )
    state["entry_armed"] |= (
        state["stable_consecutive"] >= int(success["stable_control_steps"])
    )
    state["premature_tissue_contact"] |= (
        tissue_contact & ~state["entry_armed"]
    )
    inward_speed = torch.sum(
        mdp_common.as_torch(obj.data.root_lin_vel_w)
        * state["desired_tip_direction_w"],
        dim=-1,
    )
    state["authorized_contact_transition"] |= (
        state["entry_armed"] & tissue_contact & (inward_speed > 0.0)
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
            "desired_tip_direction_receiver": (
                desired_tip_direction_receiver
            ),
            "tangent_rotation_error_receiver": (
                tangent_rotation_error_receiver
            ),
            "plane_rotation_error_receiver": (
                plane_rotation_error_receiver
            ),
            "position_error": position_error,
            "tangent_error": tangent_error,
            "plane_error": plane_error,
            "minimum_needle_clearance": minimum_needle_clearance,
            "receiver_tool_clearance": receiver_tool_clearance,
            "tissue_contact": tissue_contact,
            "ready_now": ready_now,
            "normalized_error": normalized_error,
            "object_linear_speed": object_linear_speed,
            "object_angular_speed": object_angular_speed,
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
            (
                state["position_error_receiver"] / 0.02
            ).clamp(-5.0, 5.0),
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
            (
                state["position_error"]
                / float(success["position_tolerance_m"])
            ).clamp(0.0, 5.0),
            (state["tangent_error"] / math.pi).clamp(0.0, 1.0),
            (state["plane_error"] / math.pi).clamp(0.0, 1.0),
            (state["minimum_needle_clearance"] / 0.01).clamp(-5.0, 5.0),
            (state["receiver_tool_clearance"] / 0.01).clamp(-5.0, 5.0),
            (state["stable_consecutive"].float() / stable_steps).clamp(
                0.0, 1.0
            ),
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
    active = handover["successful_handover"] & ~state["entry_armed"]
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
