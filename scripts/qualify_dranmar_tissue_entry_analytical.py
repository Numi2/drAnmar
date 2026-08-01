#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the fixed-domain analytical tissue-entry gate before PPO training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task", default="DrAnmar-Penetrate-Tissue-Needle-PSM-IK-Rel-Play-v0"
)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import orbit.surgical.tasks  # noqa: E402, F401
from orbit.surgical.tasks.surgical.penetration.residual_model import (  # noqa: E402
    PenetrationAnalyticController,
)


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    env = gym.make(args.task, cfg=env_cfg)
    controller = PenetrationAnalyticController().to(env.unwrapped.device)
    observation, _ = env.reset()
    reset_rejections = 0
    warmup_steps = 0
    while warmup_steps < 128:
        warmup_steps += 1
        with torch.inference_mode():
            observation, _, terminated, truncated, _ = env.step(
                torch.zeros((1, 6), device=env.unwrapped.device)
            )
        state = env.unwrapped._dr_anmar_penetration_state
        if bool(terminated[0] or truncated[0]):
            reset_rejections += 1
            continue
        if int(state["settle_control_steps"][0]) == 0 and bool(
            state["custody_valid"][0]
        ):
            break
    else:
        raise RuntimeError("could not obtain settled bilateral needle custody in 128 steps")
    robot = env.unwrapped.scene["robot"]
    initial_joint_positions = robot.data.joint_pos[0, :6].detach().cpu().tolist()
    initial_needle_quaternion = observation["policy"][0, 26:30].detach().cpu().tolist()
    initial_end_effector_quaternion = observation["policy"][0, 19:23].detach().cpu().tolist()
    initial_end_effector_position = observation["policy"][0, 16:19].detach().cpu().tolist()
    completed_steps = 0
    terminated_by: list[str] = []
    max_rotation_action = 0.0
    action = torch.zeros((1, 6), device=env.unwrapped.device)
    try:
        for completed_steps in range(1, args.steps + 1):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                action, _, _ = controller(observation["policy"])
                max_rotation_action = max(
                    max_rotation_action, float(torch.max(torch.abs(action[:, 3:])))
                )
                observation, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0] or truncated[0]):
                terminated_by = [
                    name
                    for name in env.unwrapped.termination_manager.active_terms
                    if bool(env.unwrapped.termination_manager.get_term(name)[0])
                ]
                break

        state = env.unwrapped._dr_anmar_penetration_state
        successful = "success" in terminated_by
        success_receipt = getattr(env.unwrapped, "_dr_anmar_last_successful_entry", None)
        evidence = success_receipt[0] if success_receipt else {
            "event_count": int(state["event_count"][0]),
            "phase": int(state["phase"][0]),
            "hard_failures": tuple(sorted(state["gates"][0].hard_failures)),
            "entry_error_m": float(state["measurement"]["entry_error"][0]),
            "tangent_error_deg": float(state["measurement"]["tangent_error"][0]),
            "plane_error_deg": float(state["measurement"]["plane_error"][0]),
            "embedded_depth_m": float(state["measurement"]["embedded_depth"][0]),
            "peak_force_n": float(state["normal_force"][0]),
        }
        report = {
            "schema": "dr.anmar.tissue-entry-analytical-gate.v1",
            "task": args.task,
            "completed_steps": completed_steps,
            "warmup_steps": warmup_steps,
            "reset_rejections": reset_rejections,
            "terminated_by": terminated_by,
            "successful": successful,
            "event_count": evidence["event_count"],
            "phase": evidence["phase"],
            "hard_failures": list(evidence["hard_failures"]),
            "entry_error_m": evidence["entry_error_m"],
            "tangent_error_deg": evidence["tangent_error_deg"],
            "plane_error_deg": evidence["plane_error_deg"],
            "embedded_depth_m": evidence["embedded_depth_m"],
            "peak_force_n": evidence["peak_force_n"],
            "wrench_finite": bool(torch.isfinite(state["wrench"]).all()),
            "max_rotation_action": max_rotation_action,
            "last_action": action[0].detach().cpu().tolist(),
            "needle_quaternion_xyzw": observation["policy"][0, 26:30].detach().cpu().tolist(),
            "surface_normal_robot": observation["policy"][0, 43:46].detach().cpu().tolist(),
            "initial_joint_positions": initial_joint_positions,
            "initial_needle_quaternion_xyzw": initial_needle_quaternion,
            "initial_end_effector_quaternion_xyzw": initial_end_effector_quaternion,
            "initial_end_effector_position_robot_m": initial_end_effector_position,
            "final_end_effector_quaternion_xyzw": observation["policy"][0, 19:23]
            .detach()
            .cpu()
            .tolist(),
            "final_joint_positions": robot.data.joint_pos[0, :6].detach().cpu().tolist(),
            "qualified_for_ppo": bool(
                successful
                and evidence["event_count"] == 1
                and not evidence["hard_failures"]
            ),
            "evidence_level": "simulator_engineering_only",
            "clinical_validation": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(f".{args.report.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.report)
        print("[DR_ANMAR_TISSUE_ENTRY_ANALYTICAL] " + json.dumps(report, sort_keys=True))
        return 0 if report["qualified_for_ppo"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
