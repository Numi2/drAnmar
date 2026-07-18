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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import orbit.surgical.tasks  # noqa: F401


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    completed_steps = 0
    for completed_steps in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    report = {
        "project": "Dr.Anmar",
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "completed_steps": completed_steps,
        "device": str(env.unwrapped.device),
        "joint_count": len(robot.joint_names),
        "joint_names": list(robot.joint_names),
        "body_count": len(robot.body_names),
        "body_names": list(robot.body_names),
        "joint_positions_finite": bool(torch.isfinite(robot.data.joint_pos).all().item()),
        "joint_velocities_finite": bool(torch.isfinite(robot.data.joint_vel).all().item()),
    }
    args_cli.report.parent.mkdir(parents=True, exist_ok=True)
    args_cli.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[DR_ANMAR_SMOKE] " + json.dumps(report, sort_keys=True))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
