#!/usr/bin/env python3
"""Deterministic controller invariants required before frontier Isaac runs."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import torch

from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_error_magnitude,
    subtract_frame_transforms,
)

from orbit.surgical.tasks.surgical.handover.mdp.state import (
    assign_balanced_handover_roles,
    place_object_in_assigned_giver_frame,
)
from orbit.surgical.tasks.surgical.handover.residual_model import (
    HandoverAnalyticController,
)


def _yaw_quaternion(
    yaw: torch.Tensor,
) -> torch.Tensor:
    quaternion = torch.zeros(
        (yaw.numel(), 4),
        dtype=yaw.dtype,
        device=yaw.device,
    )
    quaternion[:, 2] = torch.sin(0.5 * yaw)
    quaternion[:, 3] = torch.cos(0.5 * yaw)
    return quaternion


def _phase_zero_observation(
    controller: HandoverAnalyticController,
    yaw: torch.Tensor,
    *,
    canonical: bool,
) -> torch.Tensor:
    count = yaw.numel()
    raw = torch.zeros(
        (count, 106),
        dtype=yaw.dtype,
        device=yaw.device,
    )
    object_orientation = _yaw_quaternion(yaw)
    identity = torch.zeros_like(object_orientation)
    identity[:, 3] = 1.0
    object_position = torch.zeros(
        (count, 3),
        dtype=yaw.dtype,
        device=yaw.device,
    )
    object_position[:, 2] = -0.15
    local_offset = torch.tensor(
        [
            controller.giver_grasp_x,
            controller.giver_grasp_y,
            controller.giver_grasp_z,
        ],
        dtype=yaw.dtype,
        device=yaw.device,
    ).expand(count, -1)
    grasp_offset = (
        quat_apply(object_orientation, local_offset)
        if canonical
        else local_offset
    )
    giver_position = object_position + grasp_offset
    giver_position[:, 2] += controller.approach_height
    raw[:, 32:35] = giver_position
    raw[:, 35:39] = object_orientation if canonical else identity
    raw[:, 39:42] = torch.tensor(
        [0.05, 0.0, -0.13],
        dtype=yaw.dtype,
        device=yaw.device,
    )
    raw[:, 42:46] = identity
    raw[:, 46:49] = object_position
    raw[:, 49:53] = object_orientation
    raw[:, 53:56] = object_position
    raw[:, 56:60] = object_orientation
    raw[:, 77] = 1.0
    raw[:, 82] = 1.0
    return raw


def _assert_balanced_episode_zero_roles(device: torch.device) -> dict:
    count = 1200
    initial_state = {
        "giver_is_robot_1": torch.ones(
            count,
            dtype=torch.bool,
            device=device,
        )
    }
    env = SimpleNamespace(
        num_envs=count,
        device=device,
        _dr_anmar_handover_state=initial_state,
    )
    env_ids = torch.arange(count, device=device)
    assign_balanced_handover_roles(env, env_ids)
    first_roles = initial_state["giver_is_robot_1"].clone()
    first_robot_1 = int(first_roles.sum().item())
    if first_robot_1 != count // 2:
        raise AssertionError(
            f"episode-zero role population is {first_robot_1}/{count}"
        )
    assign_balanced_handover_roles(env, env_ids)
    second_roles = initial_state["giver_is_robot_1"].clone()
    if not torch.equal(second_roles, ~first_roles):
        raise AssertionError("roles did not alternate on the next reset")
    second_robot_1 = int(second_roles.sum().item())
    if second_robot_1 != count // 2:
        raise AssertionError(
            f"second role population is {second_robot_1}/{count}"
        )
    return {
        "episode_zero_robot_1": first_robot_1,
        "episode_zero_robot_2": count - first_robot_1,
        "second_reset_robot_1": second_robot_1,
        "second_reset_robot_2": count - second_robot_1,
        "roles_alternated": True,
    }


def _assert_role_conditioned_object_pose(
    device: torch.device,
) -> dict:
    count = 4
    identity = torch.zeros(
        (count, 4),
        dtype=torch.float32,
        device=device,
    )
    identity[:, 3] = 1.0
    robot_1_position = torch.tensor(
        [-0.2, 0.0, 0.15],
        dtype=torch.float32,
        device=device,
    ).expand(count, -1).clone()
    robot_2_position = torch.tensor(
        [-0.1, 0.05, 0.15],
        dtype=torch.float32,
        device=device,
    ).expand(count, -1).clone()
    robot_2_orientation = _yaw_quaternion(
        torch.full(
            (count,),
            0.5 * math.pi,
            dtype=torch.float32,
            device=device,
        )
    )
    local_position = torch.tensor(
        [
            [0.01, -0.02, -0.15],
            [-0.01, 0.015, -0.15],
            [0.02, 0.01, -0.15],
            [-0.02, -0.01, -0.15],
        ],
        dtype=torch.float32,
        device=device,
    )
    local_orientation = _yaw_quaternion(
        torch.tensor(
            [-2.4, -0.7, 0.8, 2.5],
            dtype=torch.float32,
            device=device,
        )
    )
    object_position, object_orientation = combine_frame_transforms(
        robot_1_position,
        identity,
        local_position,
        local_orientation,
    )

    class FakeAsset:
        def __init__(
            self,
            position: torch.Tensor,
            orientation: torch.Tensor,
        ) -> None:
            self.data = SimpleNamespace(
                root_pos_w=position.clone(),
                root_quat_w=orientation.clone(),
            )

    class FakeObject(FakeAsset):
        def write_root_pose_to_sim(
            self,
            pose: torch.Tensor,
            *,
            env_ids: torch.Tensor,
        ) -> None:
            self.data.root_pos_w[env_ids] = pose[:, :3]
            self.data.root_quat_w[env_ids] = pose[:, 3:7]

    env = SimpleNamespace(
        num_envs=count,
        device=device,
        scene={
            "robot_1": FakeAsset(robot_1_position, identity),
            "robot_2": FakeAsset(
                robot_2_position,
                robot_2_orientation,
            ),
            "object": FakeObject(
                object_position,
                object_orientation,
            ),
        },
    )
    env_ids = torch.arange(count, device=device)
    assign_balanced_handover_roles(env, env_ids)
    place_object_in_assigned_giver_frame(env, env_ids)
    giver_is_robot_1 = env._dr_anmar_forced_giver_is_robot_1
    giver_position = torch.where(
        giver_is_robot_1.unsqueeze(-1),
        robot_1_position,
        robot_2_position,
    )
    giver_orientation = torch.where(
        giver_is_robot_1.unsqueeze(-1),
        identity,
        robot_2_orientation,
    )
    actual_local_position, actual_local_orientation = (
        subtract_frame_transforms(
            giver_position,
            giver_orientation,
            env.scene["object"].data.root_pos_w,
            env.scene["object"].data.root_quat_w,
        )
    )
    position_error = torch.linalg.vector_norm(
        actual_local_position - local_position,
        dim=-1,
    )
    orientation_error = quat_error_magnitude(
        actual_local_orientation,
        local_orientation,
    )
    if float(position_error.max().item()) > 1.0e-6:
        raise AssertionError(
            "object placement changed the giver-local position distribution"
        )
    if float(orientation_error.max().item()) > 1.0e-6:
        raise AssertionError(
            "object placement changed the giver-local orientation distribution"
        )
    return {
        "robot_1_giver_count": int(giver_is_robot_1.sum().item()),
        "robot_2_giver_count": int((~giver_is_robot_1).sum().item()),
        "maximum_local_position_error_m": float(
            position_error.max().item()
        ),
        "maximum_local_orientation_error_rad": float(
            orientation_error.max().item()
        ),
        "tested_nonidentity_robot_2_root_orientation": True,
    }


def _assert_yaw_equivariant_pickup(device: torch.device) -> dict:
    yaw = torch.tensor(
        [0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi],
        dtype=torch.float32,
        device=device,
    )
    controller = HandoverAnalyticController().to(device)
    controller.configure_profile("frontier-hardening-v24")
    canonical_raw = _phase_zero_observation(
        controller,
        yaw,
        canonical=True,
    )
    canonical_action, _, _ = controller(canonical_raw)
    pickup_xy_error = torch.linalg.vector_norm(
        canonical_action[:, 0:2],
        dim=-1,
    )
    pickup_orientation_error = torch.linalg.vector_norm(
        canonical_action[:, 3:6],
        dim=-1,
    )
    if float(pickup_xy_error.max().item()) > 1.0e-5:
        raise AssertionError(
            "canonical pickup target did not rotate with needle yaw"
        )
    if float(pickup_orientation_error.max().item()) > 1.0e-5:
        raise AssertionError(
            "yaw-aligned tool emitted a spurious pregrasp rotation"
        )

    legacy = HandoverAnalyticController().to(device)
    legacy.configure_profile("joint-transfer-v23")
    legacy_raw = _phase_zero_observation(
        legacy,
        yaw,
        canonical=False,
    )
    legacy_action, _, _ = legacy(legacy_raw)
    legacy_xy_error = torch.linalg.vector_norm(
        legacy_action[:, 0:2],
        dim=-1,
    )
    legacy_orientation_error = torch.linalg.vector_norm(
        legacy_action[:, 3:6],
        dim=-1,
    )
    if float(legacy_xy_error.max().item()) > 1.0e-5:
        raise AssertionError("v23 pickup position semantics changed")
    if float(legacy_orientation_error.max().item()) > 1.0e-5:
        raise AssertionError("v23 pickup orientation semantics changed")

    transport_raw = canonical_raw.clone()
    transport_raw[:, 77:82] = 0.0
    transport_raw[:, 78] = 1.0
    transport_raw[:, 66:68] = 0.2
    transport_action, _, _ = controller(transport_raw)
    transport_orientation = torch.linalg.vector_norm(
        transport_action[:, 3:6],
        dim=-1,
    )
    if float(transport_orientation.max().item()) > 1.0e-5:
        raise AssertionError(
            "yaw-aligned transport emitted a global-identity twist"
        )
    return {
        "sampled_yaw_rad": [float(value) for value in yaw.tolist()],
        "maximum_canonical_pickup_xy_action": float(
            pickup_xy_error.max().item()
        ),
        "maximum_canonical_aligned_orientation_action": float(
            pickup_orientation_error.max().item()
        ),
        "maximum_v23_pickup_xy_action": float(
            legacy_xy_error.max().item()
        ),
        "maximum_v23_orientation_action": float(
            legacy_orientation_error.max().item()
        ),
        "maximum_canonical_transport_orientation_action": float(
            transport_orientation.max().item()
        ),
    }


def _assert_finite_segment_collision_geometry(
    device: torch.device,
) -> dict:
    dtype = torch.float32
    first_start = torch.tensor(
        [[-1.0, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    first_end = torch.tensor(
        [[1.0, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    second_start = torch.tensor(
        [[0.0, -1.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    second_end = torch.tensor(
        [[0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    endpoint_distance = torch.linalg.vector_norm(
        second_start - first_start,
        dim=-1,
    ).minimum(
        torch.linalg.vector_norm(second_end - first_end, dim=-1)
    )
    crossing_delta = HandoverAnalyticController._segment_to_segment_delta(
        first_start,
        first_end,
        second_start,
        second_end,
    )
    crossing_distance = torch.linalg.vector_norm(
        crossing_delta,
        dim=-1,
    )
    if float(endpoint_distance.item()) <= 1.0:
        raise AssertionError("synthetic endpoints do not expose the sampling gap")
    if float(crossing_distance.item()) > 1.0e-6:
        raise AssertionError("finite-segment geometry missed an interior crossing")
    preferred_side = torch.tensor(
        [[0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    crossing_direction = (
        HandoverAnalyticController._stable_segment_separation_direction(
            crossing_delta,
            first_end - first_start,
            second_end - second_start,
            preferred_side,
        )
    )
    if not torch.allclose(
        crossing_direction,
        preferred_side,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "interior crossing did not produce a stable separating normal"
        )
    controller = HandoverAnalyticController().to(device)
    projected_action, guard_active = (
        controller._project_translation_outside_capsule(
            -preferred_side,
            crossing_direction,
            crossing_distance,
            minimum_distance_m=0.008,
            activation_distance_m=0.012,
            eligible=torch.ones(1, dtype=torch.bool, device=device),
        )
    )
    if not bool(guard_active.item()):
        raise AssertionError("interior crossing failed to activate the guard")
    projected_inward_action = (
        projected_action[:, :3] * crossing_direction
    ).sum(dim=-1)
    if float(projected_inward_action.item()) < -1.0e-6:
        raise AssertionError(
            "interior crossing retained an unsafe inward action"
        )
    controller.receiver_distal_tool_guard_enabled = True
    giver_ee = torch.tensor(
        [[0.0, 0.0, -0.1]],
        dtype=dtype,
        device=device,
    )
    receiver_ee = torch.tensor(
        [[0.0081, 0.0, -0.09]],
        dtype=dtype,
        device=device,
    )
    identity_orientation = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    shaft_distance, distal_distance = (
        controller._receiver_tool_capsule_distances(
            giver_ee,
            receiver_ee,
            identity_orientation,
        )
    )
    zero_translation = torch.zeros_like(giver_ee)
    inward_orientation_action = torch.tensor(
        [[0.0, -0.6, 0.0]],
        dtype=dtype,
        device=device,
    )
    inward_receiver_scale, inward_giver_scale, swept_active = (
        controller._collision_safe_intertool_motion_scales(
            giver_ee=giver_ee,
            receiver_ee_in_giver=receiver_ee,
            receiver_orientation=identity_orientation,
            giver_translation=zero_translation,
            receiver_translation=zero_translation,
            receiver_orientation_action=inward_orientation_action,
            current_shaft_distance=shaft_distance,
            current_distal_distance=distal_distance,
            eligible=torch.ones(1, dtype=torch.bool, device=device),
        )
    )
    if (
        not bool(swept_active.item())
        or float(inward_receiver_scale.item()) >= 1.0
    ):
        raise AssertionError(
            "swept guard retained a full unsafe orientation command"
        )
    outward_receiver_scale, outward_giver_scale, outward_active = (
        controller._collision_safe_intertool_motion_scales(
            giver_ee=giver_ee,
            receiver_ee_in_giver=receiver_ee,
            receiver_orientation=identity_orientation,
            giver_translation=zero_translation,
            receiver_translation=zero_translation,
            receiver_orientation_action=-inward_orientation_action,
            current_shaft_distance=shaft_distance,
            current_distal_distance=distal_distance,
            eligible=torch.ones(1, dtype=torch.bool, device=device),
        )
    )
    if bool(outward_active.item()) or not torch.allclose(
        outward_receiver_scale,
        torch.ones_like(outward_receiver_scale),
    ) or not torch.allclose(
        outward_giver_scale,
        torch.ones_like(outward_giver_scale),
    ):
        raise AssertionError(
            "swept guard altered an orientation command that increases clearance"
        )

    separated_delta = HandoverAnalyticController._segment_to_segment_delta(
        first_start,
        first_end,
        second_start + torch.tensor(
            [[0.0, 0.0, 0.25]],
            dtype=dtype,
            device=device,
        ),
        second_end + torch.tensor(
            [[0.0, 0.0, 0.25]],
            dtype=dtype,
            device=device,
        ),
    )
    separated_distance = torch.linalg.vector_norm(
        separated_delta,
        dim=-1,
    )
    if abs(float(separated_distance.item()) - 0.25) > 1.0e-6:
        raise AssertionError("finite-segment distance is not metric-correct")
    return {
        "endpoint_sample_distance_for_crossing_m": float(
            endpoint_distance.item()
        ),
        "exact_crossing_distance_m": float(crossing_distance.item()),
        "exact_separated_distance_m": float(separated_distance.item()),
        "crossing_separation_direction": [
            float(value)
            for value in crossing_direction[0].detach().cpu().tolist()
        ],
        "crossing_projected_inward_action": float(
            projected_inward_action.item()
        ),
        "unsafe_receiver_motion_scale": float(
            inward_receiver_scale.item()
        ),
        "unsafe_giver_motion_scale": float(inward_giver_scale.item()),
        "safe_receiver_motion_scale": float(
            outward_receiver_scale.item()
        ),
        "safe_giver_motion_scale": float(outward_giver_scale.item()),
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evidence = {
        "schema_version": "dranmar-frontier-invariants-1.0",
        "device": str(device),
        "balanced_roles": _assert_balanced_episode_zero_roles(device),
        "role_conditioned_object_pose": (
            _assert_role_conditioned_object_pose(device)
        ),
        "yaw_equivariant_controller": _assert_yaw_equivariant_pickup(
            device
        ),
        "finite_segment_collision_geometry": (
            _assert_finite_segment_collision_geometry(device)
        ),
        "passed": True,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
