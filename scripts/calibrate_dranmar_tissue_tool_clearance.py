#!/usr/bin/env python3
"""Calibrate a collision-clear PSM reset posture above the tissue surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    default="DrAnmar-Puncture-Pullout-Tissue-Needle-PSM-IK-Rel-Play-v0",
)
parser.add_argument("--lift_m", type=float, default=0.020)
parser.add_argument("--iterations", type=int, default=160)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab.utils.math import compute_pose_error  # noqa: E402
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


def main() -> int:
    if not 0.005 <= args.lift_m <= 0.040:
        raise ValueError("lift_m must be in [0.005, 0.040]")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=False)
    # Calibration only: admit the legacy pose, lift in task space, and receipt
    # the resulting joints. Qualification always recreates the environment with
    # the real collision-enabled tissue configuration.
    env_cfg.scene.tissue_left.spawn.collision_props.collision_enabled = False
    env_cfg.scene.tissue_right.spawn.collision_props.collision_enabled = False
    env = gym.make(args.task, cfg=env_cfg)
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        joint_ids, _ = robot.find_joints(list(ARM_JOINT_NAMES))
        body_ids, _ = robot.find_bodies("psm_tool_tip_link")
        body_id = body_ids[0]
        jacobian_body_id = body_id - 1 if robot.is_fixed_base else body_id
        joint_position_data = robot.data.joint_pos
        if hasattr(joint_position_data, "torch"):
            joint_position_data = joint_position_data.torch
        joint_positions = joint_position_data[:, joint_ids].clone()
        joint_velocity = torch.zeros_like(joint_positions)
        body_pos = robot.data.body_pos_w
        body_quat = robot.data.body_quat_w
        if hasattr(body_pos, "torch"):
            body_pos = body_pos.torch
            body_quat = body_quat.torch
        desired_pos = body_pos[:, body_id, :].clone()
        desired_pos[:, 2] += args.lift_m
        desired_quat = body_quat[:, body_id, :].clone()
        for _ in range(args.iterations):
            robot.write_joint_state_to_sim(
                joint_positions, joint_velocity, joint_ids=joint_ids
            )
            env.unwrapped.sim.forward()
            body_pos = robot.data.body_pos_w
            body_quat = robot.data.body_quat_w
            jacobians = robot.data.body_link_jacobian_w
            if hasattr(body_pos, "torch"):
                body_pos = body_pos.torch
                body_quat = body_quat.torch
                jacobians = jacobians.torch
            position_error, rotation_error = compute_pose_error(
                body_pos[:, body_id, :],
                body_quat[:, body_id, :],
                desired_pos,
                desired_quat,
            )
            error = torch.cat((position_error, rotation_error), dim=-1)
            jacobian = jacobians[:, jacobian_body_id, :, joint_ids]
            identity = torch.eye(6, device=env.unwrapped.device).expand(1, 6, 6)
            delta = torch.bmm(
                jacobian.transpose(1, 2),
                torch.linalg.solve(
                    torch.bmm(jacobian, jacobian.transpose(1, 2))
                    + 0.0025 * identity,
                    error.unsqueeze(-1),
                ),
            ).squeeze(-1)
            delta[:, 2] = delta[:, 2].clamp(-0.002, 0.002)
            delta[:, (0, 1, 3, 4, 5)] = delta[:, (0, 1, 3, 4, 5)].clamp(
                -0.05, 0.05
            )
            joint_positions += delta
        robot.write_joint_state_to_sim(
            joint_positions, joint_velocity, joint_ids=joint_ids
        )
        env.unwrapped.sim.forward()
        from orbit.surgical.tasks.surgical.penetration.mdp.state import (
            penetration_state,
        )

        state = penetration_state(env.unwrapped)
        body_pos = robot.data.body_pos_w
        if hasattr(body_pos, "torch"):
            body_pos = body_pos.torch
        state = env.unwrapped._dr_anmar_penetration_state
        report = {
            "schema": "dr.anmar.tissue-tool-clearance-calibration.v1",
            "task": args.task,
            "iterations": args.iterations,
            "commanded_lift_m": args.lift_m,
            "joint_positions": {
                name: float(joint_positions[0, index])
                for index, name in enumerate(ARM_JOINT_NAMES)
            },
            "tool_position_world_m": [
                float(value) for value in body_pos[0, body_id]
            ],
            "tip_position_world_m": [
                float(value) for value in state["measurement"]["tip_pos"][0]
            ],
            "giver_distal_forces_n": [
                float(value) for value in state["giver_tissue_forces"][0]
            ],
            "clinical_validation": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("[DR_ANMAR_TISSUE_CLEARANCE_CALIBRATION] " + json.dumps(report, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
