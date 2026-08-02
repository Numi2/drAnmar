#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Solve and receipt a joint-safe canonical PSM posture for needle entry."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task", default="DrAnmar-Penetrate-Tissue-Needle-PSM-IK-Rel-Play-v0"
)
parser.add_argument("--starts", type=int, default=96)
parser.add_argument("--iterations", type=int, default=120)
parser.add_argument("--side", choices=("giver", "receiver"), default="giver")
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab.utils.math import (  # noqa: E402
    compute_pose_error,
    quat_apply,
    quat_conjugate,
    quat_mul,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import orbit.surgical.tasks  # noqa: E402, F401


ARM_JOINT_NAMES = (
    "psm_yaw_joint",
    "psm_pitch_end_joint",
    "psm_main_insertion_joint",
    "psm_tool_roll_joint",
    "psm_tool_pitch_joint",
    "psm_tool_yaw_joint",
)
NEEDLE_RADIUS_M = 0.0070028174960433945
NEEDLE_TRAILING_GRASP_POSITION_M = (0.0, 0.0062, 0.0)
REVOLUTE_MARGIN_RAD = math.radians(10.0)
INSERTION_MARGIN_M = 0.005


STANDARD_GIVER_ROOT_POSITION_M = (-0.040, -0.120, 0.140)
STANDARD_GIVER_ROOT_QUAT_CFG = (0.4816338, -0.15930964, 0.0, 0.86177104)
STANDARD_RECEIVER_ROOT_POSITION_M = (0.040, -0.120, 0.140)
STANDARD_RECEIVER_ROOT_QUAT_CFG = (0.4816338, 0.15930964, 0.0, 0.86177104)
DESIRED_NEEDLE_QUAT_XYZW = (0.5, 0.5, 0.5, -0.5)
AUTHORED_GRASP_QUAT_XYZW = (0.50904141575, 0.860742027004, 0.0, 0.0)
DESIRED_ENTRY_TIP_WORLD_M = (-0.007, 0.01043433237818, 0.058)


def _desired_grasp_pose(
    env_origins: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    *,
    receiver: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = env_origins.shape[0]
    desired_tip = torch.tensor(
        DESIRED_ENTRY_TIP_WORLD_M, device=device, dtype=dtype
    ).repeat(count, 1) + env_origins
    desired_needle_quat = torch.tensor(
        DESIRED_NEEDLE_QUAT_XYZW, device=device, dtype=dtype
    ).repeat(count, 1)
    if receiver:
        desired_ee_pos = torch.tensor(
            (0.007, 0.01043433237818, 0.085), device=device, dtype=dtype
        ).repeat(count, 1) + env_origins
        return desired_ee_pos, desired_needle_quat
    tip_local = torch.tensor(
        (0.0, -NEEDLE_RADIUS_M, 0.0), device=device, dtype=dtype
    ).repeat(count, 1)
    grasp_local = torch.tensor(
        NEEDLE_TRAILING_GRASP_POSITION_M, device=device, dtype=dtype
    ).repeat(count, 1)
    needle_root = desired_tip - quat_apply(desired_needle_quat, tip_local)
    desired_ee_pos = needle_root + quat_apply(desired_needle_quat, grasp_local)
    return desired_ee_pos, desired_needle_quat


def _needle_entry_error(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    desired_ee_pos: torch.Tensor,
    desired_needle_quat: torch.Tensor,
    grasp_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = ee_pos.shape[0]
    dtype = ee_pos.dtype
    device = ee_pos.device
    tip_local = torch.tensor(
        (0.0, -NEEDLE_RADIUS_M, 0.0), device=device, dtype=dtype
    ).repeat(count, 1)
    grasp_local = torch.tensor(
        NEEDLE_TRAILING_GRASP_POSITION_M, device=device, dtype=dtype
    ).repeat(count, 1)
    needle_quat = quat_mul(ee_quat, quat_conjugate(grasp_quat))
    jaw_offset = torch.tensor(
        (0.0, 0.0, 0.0), device=device, dtype=dtype
    ).repeat(count, 1)
    needle_root = (
        ee_pos
        + quat_apply(ee_quat, jaw_offset)
        - quat_apply(needle_quat, grasp_local)
    )
    tip = needle_root + quat_apply(needle_quat, tip_local)
    tangent = quat_apply(
        needle_quat,
        torch.tensor((-1.0, 0.0, 0.0), device=device, dtype=dtype).repeat(count, 1),
    )
    plane = quat_apply(
        needle_quat,
        torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype).repeat(count, 1),
    )
    desired_tangent = torch.tensor(
        (0.0, 0.0, -1.0), device=device, dtype=dtype
    ).repeat(count, 1)
    desired_plane = torch.tensor(
        (0.0, 1.0, 0.0), device=device, dtype=dtype
    ).repeat(count, 1)
    plane_sign = torch.where(
        torch.sum(plane * desired_plane, dim=-1, keepdim=True) < 0.0,
        -torch.ones((count, 1), device=device, dtype=dtype),
        torch.ones((count, 1), device=device, dtype=dtype),
    )
    signed_plane = plane * plane_sign
    tangent_error = torch.acos(
        torch.sum(tangent * desired_tangent, dim=-1).clamp(-1.0, 1.0)
    )
    plane_error = torch.acos(
        torch.sum(signed_plane * desired_plane, dim=-1).clamp(-1.0, 1.0)
    )
    return desired_ee_pos - ee_pos, grasp_quat, tangent_error, plane_error, tip


def _margin_bounds(limits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    margin = torch.full_like(limits[:, 0], REVOLUTE_MARGIN_RAD)
    margin[2] = INSERTION_MARGIN_M
    return limits[:, 0] + margin, limits[:, 1] - margin


def main() -> int:
    num_envs = min(12, args.starts)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=num_envs, use_fabric=False)
    receiver = args.side == "receiver"
    root_position = (
        STANDARD_RECEIVER_ROOT_POSITION_M if receiver else STANDARD_GIVER_ROOT_POSITION_M
    )
    root_quaternion = (
        STANDARD_RECEIVER_ROOT_QUAT_CFG if receiver else STANDARD_GIVER_ROOT_QUAT_CFG
    )
    env_cfg.scene.robot.init_state.pos = root_position
    env_cfg.scene.robot.init_state.rot = root_quaternion
    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    joint_ids, _ = robot.find_joints(list(ARM_JOINT_NAMES))
    body_ids, _ = robot.find_bodies("psm_tool_tip_link")
    body_id = body_ids[0]
    jacobian_body_id = body_id - 1 if robot.is_fixed_base else body_id
    limits = robot.data.soft_joint_pos_limits.torch[0, joint_ids, :]
    lower, upper = _margin_bounds(limits)
    desired_ee_pos, desired_needle_quat = _desired_grasp_pose(
        unwrapped.scene.env_origins,
        unwrapped.device,
        robot.data.root_pos_w.torch.dtype,
        receiver=receiver,
    )
    grasp_angles = torch.arange(
        num_envs, device=unwrapped.device, dtype=desired_needle_quat.dtype
    ) * (torch.pi / num_envs)
    authored_grasp = torch.zeros_like(desired_needle_quat)
    authored_grasp[:, 0] = torch.cos(grasp_angles)
    authored_grasp[:, 1] = torch.sin(grasp_angles)
    desired_ee_quat = quat_mul(desired_needle_quat, authored_grasp)
    generator = torch.Generator(device=unwrapped.device)
    generator.manual_seed(20260801)
    best_score = float("inf")
    best: dict[str, object] | None = None
    tested = 0

    try:
        while tested < args.starts:
            active = min(num_envs, args.starts - tested)
            q = lower + (upper - lower) * torch.rand(
                (num_envs, len(joint_ids)), generator=generator, device=unwrapped.device
            )
            if active < num_envs:
                q[active:] = q[0]
            velocity = torch.zeros_like(q)
            for _ in range(args.iterations):
                robot.write_joint_state_to_sim(q, velocity, joint_ids=joint_ids)
                robot.set_joint_position_target_index(target=q, joint_ids=joint_ids)
                unwrapped.scene.write_data_to_sim()
                unwrapped.sim.step(render=False)
                unwrapped.scene.update(dt=unwrapped.physics_dt)
                ee_pos = robot.data.body_pos_w.torch[:, body_id, :]
                ee_quat = robot.data.body_quat_w.torch[:, body_id, :]
                position_error, _, _, _, _ = _needle_entry_error(
                    ee_pos, ee_quat, desired_ee_pos, desired_needle_quat, authored_grasp
                )
                _, rotation_error = compute_pose_error(
                    ee_pos, ee_quat, desired_ee_pos, desired_ee_quat
                )
                error = torch.cat((position_error, rotation_error), dim=-1)
                jacobian = robot.data.body_link_jacobian_w.torch[
                    :, jacobian_body_id, :, joint_ids
                ]
                identity = torch.eye(6, device=unwrapped.device).expand(num_envs, 6, 6)
                delta = torch.bmm(
                    jacobian.transpose(1, 2),
                    torch.linalg.solve(
                        torch.bmm(jacobian, jacobian.transpose(1, 2)) + 0.0025 * identity,
                        error.unsqueeze(-1),
                    ),
                ).squeeze(-1)
                delta[:, 2] = delta[:, 2].clamp(-0.002, 0.002)
                delta[:, (0, 1, 3, 4, 5)] = delta[:, (0, 1, 3, 4, 5)].clamp(
                    -0.05, 0.05
                )
                q = torch.maximum(torch.minimum(q + delta, upper), lower)

            robot.write_joint_state_to_sim(q, velocity, joint_ids=joint_ids)
            robot.set_joint_position_target_index(target=q, joint_ids=joint_ids)
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.step(render=False)
            unwrapped.scene.update(dt=unwrapped.physics_dt)
            ee_pos = robot.data.body_pos_w.torch[:, body_id, :]
            ee_quat = robot.data.body_quat_w.torch[:, body_id, :]
            position_error, grasp_quat, tangent_error, plane_error, tip = _needle_entry_error(
                ee_pos, ee_quat, desired_ee_pos, desired_needle_quat, authored_grasp
            )
            position_norm = torch.linalg.vector_norm(position_error, dim=-1)
            orientation_error = torch.maximum(tangent_error, plane_error)
            joint_center = 0.5 * (lower + upper)
            # Reserve distal range for the subsequent clockwise half-turn:
            # the standard reset must not begin near the positive yaw or
            # negative pitch limits that the curved drive consumes.
            joint_center[4] = 0.70
            joint_center[5] = -0.34
            joint_span = (upper - lower).clamp_min(1.0e-6)
            posture_cost = torch.mean(torch.square((q - joint_center) / joint_span), dim=-1)
            score = position_norm / 0.0005 + posture_cost
            for index in range(active):
                value = float(score[index])
                if value < best_score:
                    best_score = value
                    best = {
                        "joint_positions": {
                            name: float(q[index, offset])
                            for offset, name in enumerate(ARM_JOINT_NAMES)
                        },
                        "position_error_m": float(position_norm[index]),
                        "orientation_error_deg": math.degrees(float(orientation_error[index])),
                        "needle_tip_world_m": [float(value) for value in tip[index]],
                        "derived_grasp_quaternion_xyzw": [
                            float(value) for value in grasp_quat[index]
                        ],
                        "grasp_axis_angle_deg": math.degrees(float(grasp_angles[index])),
                        "robot_root_position_m": list(root_position),
                        "robot_root_quaternion_cfg": list(root_quaternion),
                        "side": args.side,
                        "minimum_revolute_margin_deg": math.degrees(
                            float(
                                torch.min(
                                    torch.minimum(
                                        q[index, (0, 1, 3, 4, 5)]
                                        - limits[(0, 1, 3, 4, 5), 0],
                                        limits[(0, 1, 3, 4, 5), 1]
                                        - q[index, (0, 1, 3, 4, 5)],
                                    )
                                )
                            )
                        ),
                        "insertion_margin_m": float(
                            torch.min(
                                q[index, 2] - limits[2, 0],
                                limits[2, 1] - q[index, 2],
                            )
                        ),
                    }
            tested += active

        assert best is not None
        best["schema"] = "dr.anmar.tissue-entry-reset-calibration.v1"
        best["starts"] = tested
        best["iterations_per_start"] = args.iterations
        best["qualified"] = bool(
            best["position_error_m"] <= 0.0005
            and best["orientation_error_deg"] <= 5.0
            and best["minimum_revolute_margin_deg"] >= 10.0
            and best["insertion_margin_m"] >= INSERTION_MARGIN_M
        )
        best["clinical_validation"] = False
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(f".{args.report.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.report)
        print("[DR_ANMAR_TISSUE_ENTRY_RESET] " + json.dumps(best, sort_keys=True))
        return 0 if best["qualified"] else 1
    except BaseException as error:
        print(f"[DR_ANMAR_TISSUE_ENTRY_RESET_ERROR] {error!r}", flush=True)
        raise
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
