#!/usr/bin/env python3
"""Deterministic controller invariants required before frontier Isaac runs."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import torch

from isaaclab.utils.math import quat_apply

from orbit.surgical.tasks.surgical.handover.mdp.state import (
    assign_balanced_handover_roles,
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


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evidence = {
        "schema_version": "dranmar-frontier-invariants-1.0",
        "device": str(device),
        "balanced_roles": _assert_balanced_episode_zero_roles(device),
        "yaw_equivariant_controller": _assert_yaw_equivariant_pickup(
            device
        ),
        "passed": True,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
