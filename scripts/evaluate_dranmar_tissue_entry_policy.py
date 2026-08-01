#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate analytical or recurrent-residual tissue entry on isolated episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task", default="DrAnmar-Penetrate-Tissue-Needle-PSM-IK-Rel-v0"
)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--episodes", type=int, default=48)
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import orbit.surgical.tasks  # noqa: E402, F401
from orbit.surgical.tasks.surgical.penetration.residual_model import (  # noqa: E402
    PenetrationAnalyticController,
)
from orbit.surgical.tasks.surgical.penetration.config.needle.agents.rsl_rl_cfg import (  # noqa: E402
    PenetrationNeedlePPORunnerCfg,
)


DEPRECATED_MODEL_KEYS = (
    "stochastic",
    "init_noise_std",
    "noise_std_type",
    "state_dependent_std",
)


def _runner_cfg() -> PenetrationNeedlePPORunnerCfg:
    cfg = PenetrationNeedlePPORunnerCfg()
    cfg.seed = args.seed
    cfg.device = args.device
    return cfg


def _sanitized_runner_dict(cfg: PenetrationNeedlePPORunnerCfg) -> dict:
    result = cfg.to_dict()
    for model_name in ("actor", "critic"):
        for key in DEPRECATED_MODEL_KEYS:
            result[model_name].pop(key, None)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=True,
    )
    env_cfg.seed = args.seed
    gym_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env)
    runner = None
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint is None:
        controller = PenetrationAnalyticController().to(env.unwrapped.device)

        def policy(observation):
            return controller(observation["policy"])[0]

        policy_name = "analytical"
    else:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        agent_cfg = _runner_cfg()
        runner = OnPolicyRunner(
            env,
            _sanitized_runner_dict(agent_cfg),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(
            str(checkpoint),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
        )
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_name = "recurrent_residual_ppo"

    observation = env.get_observations()
    completed = 0
    successes = 0
    hard_failures = 0
    timeouts = 0
    entry_errors: list[float] = []
    tangent_errors: list[float] = []
    plane_errors: list[float] = []
    embedded_depths: list[float] = []
    normalized_peak_forces: list[float] = []
    event_counts: list[int] = []
    failure_flags: dict[str, int] = {}
    control_steps = 0
    try:
        while completed < args.episodes and simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(observation)
                observation, _, dones, _ = env.step(actions)
                if runner is not None:
                    policy.reset(dones)
            control_steps += 1
            if not bool(torch.any(dones)):
                continue
            termination_manager = env.unwrapped.termination_manager
            success_mask = termination_manager.get_term("success")
            hard_mask = termination_manager.get_term("hard_failure")
            timeout_mask = termination_manager.get_term("time_out")
            success_receipts = getattr(
                env.unwrapped, "_dr_anmar_last_successful_entry", None
            )
            hard_receipts = getattr(
                env.unwrapped, "_dr_anmar_last_hard_failures", None
            )
            for env_index in torch.nonzero(dones, as_tuple=False).squeeze(-1).tolist():
                if completed >= args.episodes:
                    break
                completed += 1
                if bool(success_mask[env_index]):
                    successes += 1
                    receipt = success_receipts[env_index]
                    entry_errors.append(float(receipt["entry_error_m"]))
                    tangent_errors.append(float(receipt["tangent_error_deg"]))
                    plane_errors.append(float(receipt["plane_error_deg"]))
                    embedded_depths.append(float(receipt["embedded_depth_m"]))
                    normalized_peak_forces.append(
                        float(receipt["peak_force_n"])
                        / float(receipt["sampled_puncture_force_n"])
                    )
                    event_counts.append(int(receipt["event_count"]))
                elif bool(hard_mask[env_index]):
                    hard_failures += 1
                    flags = hard_receipts[env_index] if hard_receipts else ("unknown",)
                    for flag in flags:
                        failure_flags[flag] = failure_flags.get(flag, 0) + 1
                elif bool(timeout_mask[env_index]):
                    timeouts += 1
                else:
                    failure_flags["unclassified_done"] = (
                        failure_flags.get("unclassified_done", 0) + 1
                    )
        if completed != args.episodes:
            raise RuntimeError(
                f"simulation stopped after {completed}/{args.episodes} episodes"
            )
        report = {
            "schema": "dr.anmar.tissue-entry-isolated-evaluation.v1",
            "task": args.task,
            "policy": policy_name,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint else None,
            "seed": args.seed,
            "episodes": completed,
            "successes": successes,
            "success_rate": successes / completed,
            "hard_safety_failures": hard_failures,
            "timeouts": timeouts,
            "failure_flags": failure_flags,
            "control_steps": control_steps,
            "entry_error_m_max": max(entry_errors, default=None),
            "entry_error_m_mean": (
                sum(entry_errors) / len(entry_errors) if entry_errors else None
            ),
            "tangent_error_deg_max": max(tangent_errors, default=None),
            "plane_error_deg_max": max(plane_errors, default=None),
            "embedded_depth_m_min": min(embedded_depths, default=None),
            "embedded_depth_m_max": max(embedded_depths, default=None),
            "normalized_peak_force_max": max(normalized_peak_forces, default=None),
            "normalized_peak_force_mean": (
                sum(normalized_peak_forces) / len(normalized_peak_forces)
                if normalized_peak_forces
                else None
            ),
            "exactly_one_event_per_success": bool(event_counts)
            and all(count == 1 for count in event_counts),
            "custody_model": "pregrasped_pose_coupling",
            "evidence_level": "simulator_engineering_only",
            "clinical_validation": False,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(f".{args.report.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.report)
        print("[DR_ANMAR_TISSUE_ENTRY_EVALUATION] " + json.dumps(report, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        os._exit(1)
    simulation_app.close()
    raise SystemExit(exit_code)
