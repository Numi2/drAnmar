# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded smoke test for the ported ORBIT-Surgical dVRK environments."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Load and step a Dr.Anmar surgical environment.")
parser.add_argument("--task", default="Isaac-Reach-PSM-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--report", type=Path, default=Path("logs/dr_anmar_smoke.json"))
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The ECM task authors a camera in its native scene configuration. Isaac Lab
# requires the renderer to be enabled before that environment is constructed.
if "Reach-ECM" in args_cli.task:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import orbit.surgical.tasks  # noqa: E402, F401


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    scene = env.unwrapped.scene
    robot_names: list[str] = []
    robots = []
    for name in ("robot", "robot_1", "robot_2"):
        try:
            robots.append(scene[name])
            robot_names.append(name)
        except KeyError:
            continue
    if not robot_names:
        raise RuntimeError(f"No supported robot entity was found in {args_cli.task}")
    completed_steps = 0
    termination_count = 0
    truncation_count = 0
    for completed_steps in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            _, _, terminated, truncated, _ = env.step(actions)
            termination_count += int(terminated.sum().item())
            truncation_count += int(truncated.sum().item())

    report = {
        "project": "Dr.Anmar",
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "completed_steps": completed_steps,
        "termination_count": termination_count,
        "truncation_count": truncation_count,
        "device": str(env.unwrapped.device),
        "robot_count": len(robots),
        "robots": [
            {
                "name": name,
                "joint_count": len(robot.joint_names),
                "joint_names": list(robot.joint_names),
                "body_count": len(robot.body_names),
                "body_names": list(robot.body_names),
                "joint_positions_finite": bool(torch.isfinite(robot.data.joint_pos).all().item()),
                "joint_velocities_finite": bool(torch.isfinite(robot.data.joint_vel).all().item()),
            }
            for name, robot in zip(robot_names, robots, strict=True)
        ],
        # Preserve the original single-robot summary for existing tooling.
        "joint_count": len(robots[0].joint_names),
        "joint_names": list(robots[0].joint_names),
        "body_count": len(robots[0].body_names),
        "body_names": list(robots[0].body_names),
        "joint_positions_finite": all(bool(torch.isfinite(robot.data.joint_pos).all().item()) for robot in robots),
        "joint_velocities_finite": all(bool(torch.isfinite(robot.data.joint_vel).all().item()) for robot in robots),
    }
    penetration_state = getattr(env.unwrapped, "_dr_anmar_penetration_state", None)
    if penetration_state is not None:
        report["tissue_entry"] = {
            "episode_length_steps": env.unwrapped.episode_length_buf.detach().cpu().tolist(),
            "phase": penetration_state["phase"].detach().cpu().tolist(),
            "event_count": penetration_state["event_count"].detach().cpu().tolist(),
            "hard_failure": penetration_state["hard_failure"].detach().cpu().tolist(),
            "jaw_forces_n": penetration_state["jaw_forces"].detach().cpu().tolist(),
            "custody_valid": penetration_state["custody_valid"].detach().cpu().tolist(),
            "grasp_position_error_m": penetration_state["grasp_position_error"]
            .detach()
            .cpu()
            .tolist(),
            "grasp_angle_error_deg": penetration_state["grasp_angle_error_deg"]
            .detach()
            .cpu()
            .tolist(),
            "wrench_finite": bool(torch.isfinite(penetration_state["wrench"]).all().item()),
            "entry_error_m": penetration_state["measurement"]["entry_error"].detach().cpu().tolist(),
            "tip_position_w": penetration_state["measurement"]["tip_pos"].detach().cpu().tolist(),
            "needle_quaternion_xyzw": penetration_state["measurement"]["tip_quat"]
            .detach()
            .cpu()
            .tolist(),
            "target_position_w": penetration_state["measurement"]["target"].detach().cpu().tolist(),
            "indentation_m": penetration_state["measurement"]["indentation"].detach().cpu().tolist(),
            "hard_failure_flags": [sorted(gate.hard_failures) for gate in penetration_state["gates"]],
            "last_hard_failure_flags": getattr(
                env.unwrapped, "_dr_anmar_last_hard_failures", None
            ),
            "termination_terms": {
                name: env.unwrapped.termination_manager.get_term(name).detach().cpu().tolist()
                for name in env.unwrapped.termination_manager.active_terms
            },
        }
    args_cli.report.parent.mkdir(parents=True, exist_ok=True)
    args_cli.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[DR_ANMAR_SMOKE] " + json.dumps(report, sort_keys=True))
    env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
