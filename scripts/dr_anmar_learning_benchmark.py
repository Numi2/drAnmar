#!/usr/bin/env python3
"""Train and evaluate DrAnmar policies on the active Isaac Lab runtime.

Isaac Sim is launched before the extension is imported. This ordering is
required for extension assets that still resolve OpenUSD and simulator modules
eagerly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import resource
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _prepare_imports(repo_root: Path, isaaclab_root: Path) -> None:
    paths = (
        repo_root / "source/extensions/orbit.surgical.tasks",
        repo_root / "source/extensions/orbit.surgical.assets",
        isaaclab_root,
    )
    for path in reversed(paths):
        sys.path.insert(0, str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _free_gpu_memory_mib() -> int | None:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    try:
        return int(output.splitlines()[0].strip())
    except ValueError:
        return None


def _system_memory_mib() -> tuple[int | None, int | None]:
    """Return total and currently available system memory in MiB."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values: dict[str, int] = {}
        for line in meminfo.read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator:
                values[key] = int(value.strip().split()[0]) // 1024
        return values.get("MemTotal"), values.get("MemAvailable")
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page_size // (1024**2)
        available = os.sysconf("SC_AVPHYS_PAGES") * page_size // (1024**2)
        return total, available
    except (OSError, TypeError, ValueError):
        return None, None


def _parallel_world_cap(available_mib: int, *, gpu: bool) -> int | None:
    if gpu:
        if available_mib < 3072:
            return 8
        if available_mib < 4096:
            return 16
        if available_mib < 6144:
            return 32
        if available_mib < 10_000:
            return 64
        if available_mib < 14_000:
            return 256
        if available_mib < 18_000:
            return 512
        if available_mib < 22_000:
            return 1024
        return None
    if available_mib < 4096:
        return 8
    if available_mib < 6144:
        return 32
    if available_mib < 8192:
        return 64
    if available_mib < 10_240:
        return 128
    if available_mib < 16_384:
        return 256
    if available_mib < 24_576:
        return 512
    if available_mib < 40_960:
        return 1024
    return None


def _fit_num_envs_to_memory(
    requested: int,
    free_gpu_mib: int | None,
    available_system_mib: int | None,
) -> int:
    """Cap parallel worlds using the stricter live RAM or VRAM allowance."""
    fitted = requested
    if free_gpu_mib is not None:
        cap = _parallel_world_cap(free_gpu_mib, gpu=True)
        if cap is not None:
            fitted = min(fitted, cap)
    if available_system_mib is not None:
        cap = _parallel_world_cap(available_system_mib, gpu=False)
        if cap is not None:
            fitted = min(fitted, cap)
    if fitted != requested:
        print(
            "[DrAnmar] Memory fit: "
            f"{free_gpu_mib if free_gpu_mib is not None else 'unknown'} MiB GPU free, "
            f"{available_system_mib if available_system_mib is not None else 'unknown'} "
            f"MiB system available; using {fitted} of {requested} requested environments"
        )
    return fitted


def _peak_process_memory_mib() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024**2)
    return peak / 1024


def _runtime_evidence(repo_root: Path) -> dict[str, Any]:
    import torch

    packages = {}
    for package in ("isaacsim", "isaaclab", "isaaclab-rl", "rsl-rl-lib", "torch"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    gpu = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "packages": packages,
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "gpu": gpu,
        },
        "source": {
            "dranmar_revision": _command_output(["git", "rev-parse", "HEAD"], repo_root),
            "asset_revision": _command_output(
                ["git", "rev-parse", "HEAD"],
                repo_root / "source/extensions/orbit.surgical.assets",
            ),
        },
    }


def _write_evidence(output_dir: Path, prefix: str, evidence: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{prefix}_{stamp}.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"[DrAnmar] Evidence: {path}")
    return path


def _load_configs(task: str, num_envs: int, seed: int):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=num_envs, use_fabric=True)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = seed
    env_cfg.seed = seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    return env_cfg, agent_cfg


def _pose_error_action(
    policy_obs,
    *,
    position_start: int,
    orientation_start: int,
    position_scale: float,
    orientation_scale: float,
):
    import torch

    position_error = policy_obs[:, position_start : position_start + 3]
    orientation_error = policy_obs[:, orientation_start : orientation_start + 3]
    return torch.cat(
        (
            position_error / position_scale,
            orientation_error / orientation_scale,
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)


def _reach_teacher_action(
    obs,
    task: str,
    *,
    position_scale: float,
    orientation_scale: float,
):
    """Map a declared reach pose-error observation to relative-IK actions."""
    import torch

    policy_obs = obs["policy"]
    if "Reach-Dual-PSM-IK-Rel" in task:
        return torch.cat(
            (
                _pose_error_action(
                    policy_obs,
                    position_start=46,
                    orientation_start=49,
                    position_scale=position_scale,
                    orientation_scale=orientation_scale,
                ),
                _pose_error_action(
                    policy_obs,
                    position_start=52,
                    orientation_start=55,
                    position_scale=position_scale,
                    orientation_scale=orientation_scale,
                ),
            ),
            dim=-1,
        )
    if "Reach-PSM-IK-Rel" in task:
        return _pose_error_action(
            policy_obs,
            position_start=23,
            orientation_start=26,
            position_scale=position_scale,
            orientation_scale=orientation_scale,
        )
    raise ValueError(f"no analytic reach teacher is declared for task: {task}")


def _reach_error_offsets(task: str) -> tuple[tuple[str, int, int], ...]:
    if "Reach-Dual-PSM-IK-Rel" in task:
        return (
            ("arm_1", 46, 49),
            ("arm_2", 52, 55),
        )
    if "Reach-PSM-IK-Rel" in task:
        return (("arm", 23, 26),)
    return ()


def _lift_teacher_action(
    obs,
    *,
    position_scale: float,
    approach_height: float = 0.02,
    grasp_height: float = 0.0,
    lateral_alignment_threshold: float = 0.005,
    close_distance: float = 0.005,
    slow_approach_radius: float = 0.02,
    slow_approach_action_limit: float = 0.1,
    normalized_contact_threshold: float = 0.002,
    lateral_clearance_below_target: float = 0.04,
    carry_latch_below_target: float = 0.062,
    carry_action_limit: float = 0.1,
    carry_lateral_action_limit: float | None = None,
    carry_vertical_action_limit: float | None = 0.18,
    carry_orientation_action_limit: float | None = None,
    carry_orientation_scale: float = 0.05,
    carry_orientation_velocity_damping_s: float = 0.0,
    carry_goal_action_limit: float | None = None,
    carry_goal_position_radius: float = 0.015,
    carry_target_height_offset: float = 0.0,
    grasp_offset: tuple[float, float, float] | None = None,
):
    """Contact-conditioned analytic approach, grasp, and lift action."""
    import torch

    if grasp_offset is None:
        from orbit.surgical.tasks.surgical.lift.grasp_frames import (
            BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_M,
        )

        grasp_offset = BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_M

    policy_obs = obs["policy"]
    ee_position = policy_obs[:, 16:19]
    object_position = policy_obs[:, 23:26]
    target_position = policy_obs[:, 36:39]
    contact_forces = policy_obs[:, 43:45]

    grasp_position = object_position.clone()
    grasp_position[:, 0] += grasp_offset[0]
    grasp_position[:, 1] += grasp_offset[1]
    grasp_position[:, 2] += grasp_offset[2] + grasp_height
    ee_to_grasp = grasp_position - ee_position
    lateral_distance = torch.linalg.vector_norm(ee_to_grasp[:, :2], dim=-1)
    above_object = grasp_position.clone()
    above_object[:, 2] += approach_height
    grasp_distance = torch.linalg.vector_norm(
        grasp_position - ee_position,
        dim=-1,
    )
    approach_position = torch.where(
        (lateral_distance > lateral_alignment_threshold).unsqueeze(-1),
        above_object,
        grasp_position,
    )
    bilateral_contact = torch.all(
        contact_forces > normalized_contact_threshold,
        dim=-1,
    )
    lifted_carry = object_position[:, 2] > (
        target_position[:, 2] - carry_latch_below_target
    )
    carry_mode = bilateral_contact | lifted_carry
    approach_action = (
        (approach_position - ee_position) / position_scale
    ).clamp(-1.0, 1.0)
    slow_approach_action = approach_action.clamp(
        -slow_approach_action_limit,
        slow_approach_action_limit,
    )
    approach_action = torch.where(
        (grasp_distance < slow_approach_radius).unsqueeze(-1),
        slow_approach_action,
        approach_action,
    )
    vertical_only = object_position[:, 2] < (
        target_position[:, 2] - lateral_clearance_below_target
    )
    carry_target = target_position.clone()
    carry_target[:, 2] += carry_target_height_offset
    carry_target[:, :2] = torch.where(
        vertical_only.unsqueeze(-1),
        object_position[:, :2],
        target_position[:, :2],
    )
    lateral_limit = (
        carry_action_limit
        if carry_lateral_action_limit is None
        else carry_lateral_action_limit
    )
    vertical_limit = (
        carry_action_limit
        if carry_vertical_action_limit is None
        else carry_vertical_action_limit
    )
    carry_error_action = (
        carry_target - object_position
    ) / position_scale
    carry_action = torch.cat(
        (
            carry_error_action[:, :2].clamp(
                -lateral_limit,
                lateral_limit,
            ),
            carry_error_action[:, 2:].clamp(
                -vertical_limit,
                vertical_limit,
            ),
        ),
        dim=-1,
    )
    if carry_goal_action_limit is not None:
        goal_error_action = (
            target_position - object_position
        ) / position_scale
        goal_action = goal_error_action.clamp(
            -carry_goal_action_limit,
            carry_goal_action_limit,
        )
        inside_goal_radius = torch.linalg.vector_norm(
            target_position - object_position,
            dim=-1,
        ) < carry_goal_position_radius
        carry_action = torch.where(
            inside_goal_radius.unsqueeze(-1),
            goal_action,
            carry_action,
        )
    translation_action = torch.where(
        carry_mode.unsqueeze(-1),
        carry_action,
        approach_action,
    )
    orientation_action = torch.zeros_like(translation_action)
    if carry_orientation_action_limit is not None:
        from isaaclab.utils.math import (
            axis_angle_from_quat,
            quat_conjugate,
            quat_mul,
        )

        object_orientation = policy_obs[:, 26:30]
        target_orientation = policy_obs[:, 39:43]
        object_angular_velocity = policy_obs[:, 33:36]
        object_to_target = quat_mul(
            target_orientation,
            quat_conjugate(object_orientation),
        )
        carry_orientation_action = (
            (
                axis_angle_from_quat(object_to_target)
                - carry_orientation_velocity_damping_s
                * object_angular_velocity
            )
            / carry_orientation_scale
        ).clamp(
            -carry_orientation_action_limit,
            carry_orientation_action_limit,
        )
        orientation_action = torch.where(
            carry_mode.unsqueeze(-1),
            carry_orientation_action,
            orientation_action,
        )
    body_action = torch.cat(
        (translation_action, orientation_action),
        dim=-1,
    ).clamp(-1.0, 1.0)
    closing = (
        grasp_distance < close_distance
    ) | torch.any(
        contact_forces > normalized_contact_threshold,
        dim=-1,
    ) | lifted_carry
    gripper_action = torch.where(
        closing,
        -torch.ones_like(grasp_distance),
        torch.ones_like(grasp_distance),
    ).unsqueeze(-1)
    return torch.cat((body_action, gripper_action), dim=-1)


def _teacher_action(
    obs,
    task: str,
    *,
    position_scale: float,
    orientation_scale: float,
):
    if "Lift-Block-PSM-IK-Rel" in task:
        return _lift_teacher_action(obs, position_scale=position_scale)
    if "Lift-Needle-PSM-IK-Rel" in task:
        from orbit.surgical.tasks.surgical.lift.grasp_frames import (
            NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
        )

        return _lift_teacher_action(
            obs,
            position_scale=position_scale,
            grasp_offset=NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
            carry_orientation_action_limit=0.035,
            carry_orientation_scale=orientation_scale,
            carry_orientation_velocity_damping_s=0.001,
        )
    return _reach_teacher_action(
        obs,
        task,
        position_scale=position_scale,
        orientation_scale=orientation_scale,
    )


def _handover_teacher_action(
    obs,
    *,
    receiver_grasp_offset: tuple[float, float, float],
    receiver_roll_offset_rad: float = 0.0,
    position_scale: float = 0.01,
    orientation_scale: float = 0.05,
    receiver_orientation_action_limit: float = 0.3,
    approach_height: float = 0.02,
    lateral_alignment_threshold: float = 0.005,
    close_distance: float = 0.005,
    slow_approach_radius: float = 0.02,
    slow_approach_action_limit: float = 0.1,
    normalized_contact_threshold: float = 0.002,
    presentation_height_in_robot_frame: float = -0.07,
    minimum_lift_height_in_robot_frame: float = -0.09,
    carry_lateral_action_limit: float = 0.1,
    carry_vertical_action_limit: float = 0.18,
):
    """Stage a physical Arm 1 pickup and Arm 2 custody transfer."""
    import math

    import torch

    from orbit.surgical.tasks.surgical.lift.grasp_frames import (
        NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
    )

    policy_obs = obs["policy"]
    giver_ee = policy_obs[:, 32:35]
    giver_orientation = policy_obs[:, 35:39]
    receiver_ee = policy_obs[:, 39:42]
    receiver_orientation = policy_obs[:, 42:46]
    object_in_giver = policy_obs[:, 46:49]
    object_in_receiver = policy_obs[:, 53:56]
    giver_contacts = policy_obs[:, 66:68]
    receiver_contacts = policy_obs[:, 68:70]
    phase = torch.argmax(policy_obs[:, 77:82], dim=-1)

    def approach_action(
        ee_position,
        object_position,
        grasp_offset,
    ):
        grasp_position = object_position.clone()
        grasp_position[:, 0] += grasp_offset[0]
        grasp_position[:, 1] += grasp_offset[1]
        grasp_position[:, 2] += grasp_offset[2]
        delta = grasp_position - ee_position
        lateral_distance = torch.linalg.vector_norm(delta[:, :2], dim=-1)
        above = grasp_position.clone()
        above[:, 2] += approach_height
        target = torch.where(
            (lateral_distance > lateral_alignment_threshold).unsqueeze(-1),
            above,
            grasp_position,
        )
        distance = torch.linalg.vector_norm(
            grasp_position - ee_position,
            dim=-1,
        )
        action = ((target - ee_position) / position_scale).clamp(-1.0, 1.0)
        action = torch.where(
            (distance < slow_approach_radius).unsqueeze(-1),
            action.clamp(
                -slow_approach_action_limit,
                slow_approach_action_limit,
            ),
            action,
        )
        return action, distance

    giver_approach, giver_distance = approach_action(
        giver_ee,
        object_in_giver,
        NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
    )
    receiver_approach, receiver_distance = approach_action(
        receiver_ee,
        object_in_receiver,
        receiver_grasp_offset,
    )

    root_2_in_giver = object_in_giver - object_in_receiver
    midpoint_in_giver = 0.5 * root_2_in_giver
    midpoint_in_receiver = -0.5 * root_2_in_giver
    giver_target = midpoint_in_giver.clone()
    giver_target[:, 2] = presentation_height_in_robot_frame
    vertical_only = (
        object_in_giver[:, 2] < minimum_lift_height_in_robot_frame
    )
    giver_target[:, :2] = torch.where(
        vertical_only.unsqueeze(-1),
        object_in_giver[:, :2],
        giver_target[:, :2],
    )
    giver_error = (giver_target - object_in_giver) / position_scale
    giver_carry = torch.cat(
        (
            giver_error[:, :2].clamp(
                -carry_lateral_action_limit,
                carry_lateral_action_limit,
            ),
            giver_error[:, 2:].clamp(
                -carry_vertical_action_limit,
                carry_vertical_action_limit,
            ),
        ),
        dim=-1,
    )
    giver_bilateral_contact = torch.all(
        giver_contacts > normalized_contact_threshold,
        dim=-1,
    )
    giver_lifted = (
        object_in_giver[:, 2] > minimum_lift_height_in_robot_frame
    )
    giver_carry_mode = (
        ((phase == 1) & giver_bilateral_contact)
        | ((phase == 2) & (giver_bilateral_contact | giver_lifted))
    )

    receiver_stage_target = midpoint_in_receiver.clone()
    receiver_stage_target[:, 2] = (
        presentation_height_in_robot_frame + approach_height
    )
    receiver_stage = (
        (receiver_stage_target - receiver_ee) / position_scale
    ).clamp(-carry_lateral_action_limit, carry_lateral_action_limit)
    receiver_hold_target = midpoint_in_receiver.clone()
    receiver_hold_target[:, 2] = presentation_height_in_robot_frame
    receiver_hold_error = (
        receiver_hold_target - object_in_receiver
    ) / position_scale
    receiver_hold = torch.cat(
        (
            receiver_hold_error[:, :2].clamp(
                -carry_lateral_action_limit,
                carry_lateral_action_limit,
            ),
            receiver_hold_error[:, 2:].clamp(
                -carry_vertical_action_limit,
                carry_vertical_action_limit,
            ),
        ),
        dim=-1,
    )

    giver_translation = torch.where(
        giver_carry_mode.unsqueeze(-1),
        giver_carry,
        giver_approach,
    )
    giver_retreat = torch.zeros_like(giver_translation)
    giver_retreat[:, 2] = carry_lateral_action_limit
    giver_translation = torch.where(
        (phase >= 3).unsqueeze(-1),
        giver_retreat,
        giver_translation,
    )
    receiver_translation = torch.where(
        (
            (phase <= 1)
            | ((phase == 2) & ~giver_lifted)
        ).unsqueeze(-1),
        receiver_stage,
        receiver_approach,
    )
    receiver_translation = torch.where(
        (phase >= 3).unsqueeze(-1),
        receiver_hold,
        receiver_translation,
    )

    giver_closing = (
        (giver_distance < close_distance)
        | torch.any(
            giver_contacts > normalized_contact_threshold,
            dim=-1,
        )
        | ((phase >= 1) & (phase <= 2))
    ) & (phase < 3)
    receiver_closing = (
        (phase >= 2)
        & (
            (receiver_distance < close_distance)
            | torch.any(
                receiver_contacts > normalized_contact_threshold,
                dim=-1,
            )
            | (phase >= 3)
        )
    )
    giver_gripper = torch.where(
        giver_closing,
        -torch.ones_like(giver_distance),
        torch.ones_like(giver_distance),
    ).unsqueeze(-1)
    receiver_gripper = torch.where(
        receiver_closing,
        -torch.ones_like(receiver_distance),
        torch.ones_like(receiver_distance),
    ).unsqueeze(-1)
    zero_orientation = torch.zeros_like(giver_translation)
    from isaaclab.utils.math import (
        axis_angle_from_quat,
        quat_conjugate,
        quat_mul,
    )

    receiver_roll = torch.zeros_like(giver_orientation)
    half_roll = 0.5 * receiver_roll_offset_rad
    receiver_roll[:, 2] = math.sin(half_roll)
    receiver_roll[:, 3] = math.cos(half_roll)
    receiver_target_orientation = quat_mul(
        receiver_roll,
        giver_orientation,
    )
    receiver_orientation_error = axis_angle_from_quat(
        quat_mul(
            receiver_target_orientation,
            quat_conjugate(receiver_orientation),
        )
    )
    receiver_orientation_action = (
        receiver_orientation_error / orientation_scale
    ).clamp(
        -receiver_orientation_action_limit,
        receiver_orientation_action_limit,
    )
    return torch.cat(
        (
            giver_translation,
            zero_orientation,
            giver_gripper,
            receiver_translation,
            receiver_orientation_action,
            receiver_gripper,
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)


def _pretraining_algorithm(task: str) -> str:
    if "Lift-" in task:
        return "analytic_grasp_lift_base_plus_learned_residual"
    return "analytic_relative_ik_base_plus_learned_residual"


def _pretrain(args: argparse.Namespace, repo_root: Path) -> int:
    """Initialize and validate a task-declared analytic-base residual actor."""
    import gymnasium as gym
    import torch
    import torch.nn.functional as functional
    from orbit.surgical.tasks.surgical.lift.grasp_frames import (
        BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_M,
        BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_SOURCE,
        NEEDLE_GEOMETRY_GRASP_OFFSET_SOURCE,
        NEEDLE_PROVISIONAL_ARC_FRACTION,
        NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
        NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M,
    )
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env_cfg, agent_cfg = _load_configs(args.task, args.num_envs, args.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_path).resolve() / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(run_dir)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device
    )
    runner.logger.git_status_repos = []
    policy = runner.alg.get_policy()
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    obs = env.get_observations().to(agent_cfg.device)
    losses: list[float] = []
    teacher_successes = 0
    teacher_completed = 0
    started = time.perf_counter()
    try:
        for _ in range(args.updates):
            teacher_actions = _teacher_action(
                obs,
                args.task,
                position_scale=args.position_scale,
                orientation_scale=args.orientation_scale,
            )
            policy.update_normalization(obs)
            predicted_actions = policy(obs)
            loss = functional.smooth_l1_loss(predicted_actions, teacher_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))

            with torch.no_grad():
                obs, _, dones, _ = env.step(teacher_actions)
                successes = env.unwrapped.termination_manager.get_term("success")
            obs = obs.to(agent_cfg.device)
            teacher_successes += int(successes.sum().item())
            teacher_completed += int(dones.sum().item())

        policy.eval()
        obs, _ = env.reset()
        obs = obs.to(agent_cfg.device)
        validation_successes = 0
        validation_completed = 0
        error_offsets = _reach_error_offsets(args.task)
        diagnostic_totals = {
            name: torch.zeros(6, device=agent_cfg.device)
            for name, _, _ in error_offsets
        }
        simultaneous_pose_inside = torch.zeros(1, device=agent_cfg.device)
        diagnostic_trace_frames = {
            0,
            1,
            2,
            5,
            10,
            20,
            40,
            80,
            120,
            149,
            150,
            300,
            args.validation_frames - 1,
        }
        pose_diagnostic_trace = []
        with torch.no_grad():
            for frame_index in range(args.validation_frames):
                policy_obs = obs["policy"]
                pose_inside_terms = []
                trace_entry = {"frame": frame_index, "arms": {}}
                for name, position_start, orientation_start in error_offsets:
                    position_error = torch.linalg.vector_norm(
                        policy_obs[:, position_start : position_start + 3],
                        dim=-1,
                    )
                    orientation_error = torch.linalg.vector_norm(
                        policy_obs[:, orientation_start : orientation_start + 3],
                        dim=-1,
                    )
                    position_inside = position_error < 0.01
                    orientation_inside = orientation_error < 0.15
                    pose_inside = position_inside & orientation_inside
                    pose_inside_terms.append(pose_inside)
                    diagnostic_totals[name] += torch.stack(
                        (
                            position_error.new_tensor(position_error.numel()),
                            position_error.sum(),
                            orientation_error.sum(),
                            position_inside.sum().to(position_error.dtype),
                            orientation_inside.sum().to(position_error.dtype),
                            pose_inside.sum().to(position_error.dtype),
                        )
                    )
                    if frame_index in diagnostic_trace_frames:
                        trace_entry["arms"][name] = {
                            "mean_position_error_m": float(
                                position_error.mean().item()
                            ),
                            "mean_orientation_error_rad": float(
                                orientation_error.mean().item()
                            ),
                        }
                if pose_inside_terms:
                    simultaneous = pose_inside_terms[0]
                    for pose_inside in pose_inside_terms[1:]:
                        simultaneous = simultaneous & pose_inside
                    simultaneous_pose_inside += simultaneous.sum()
                actions = policy(obs)
                if frame_index in diagnostic_trace_frames:
                    for action_index, (name, _, _) in enumerate(error_offsets):
                        action_start = action_index * 6
                        trace_entry["arms"][name]["mean_abs_action"] = float(
                            actions[
                                :,
                                action_start : action_start + 6,
                            ]
                            .abs()
                            .mean()
                            .item()
                        )
                    pose_diagnostic_trace.append(trace_entry)
                obs, _, dones, _ = env.step(actions)
                successes = env.unwrapped.termination_manager.get_term("success")
                obs = obs.to(agent_cfg.device)
                validation_successes += int(successes.sum().item())
                validation_completed += int(dones.sum().item())

        duration = time.perf_counter() - started
        checkpoint = run_dir / "model_final.pt"
        runner.save(str(checkpoint))
        simulated_frames = env.unwrapped.num_envs * (
            args.updates + args.validation_frames
        )
        pose_diagnostics = {}
        for name, totals in diagnostic_totals.items():
            values = totals.cpu().tolist()
            samples = int(values[0])
            pose_diagnostics[name] = {
                "samples": samples,
                "mean_position_error_m": values[1] / samples,
                "mean_orientation_error_rad": values[2] / samples,
                "position_inside_rate": values[3] / samples,
                "orientation_inside_rate": values[4] / samples,
                "pose_inside_rate": values[5] / samples,
            }
        if error_offsets:
            pose_diagnostics["simultaneous_pose_inside_rate"] = (
                float(simultaneous_pose_inside.item())
                / (env.unwrapped.num_envs * args.validation_frames)
            )
        evidence = {
            "schema_version": "dranmar-learning-evidence-1.0",
            "kind": "training",
            "algorithm": _pretraining_algorithm(args.task),
            "task": args.task,
            "seed": args.seed,
            "requested_num_envs": args.requested_num_envs,
            "num_envs": env.unwrapped.num_envs,
            "trusted_requested_num_envs": args.trusted_requested_num_envs,
            "free_gpu_memory_before_launch_mib": args.free_gpu_memory_before_launch_mib,
            "system_memory_total_mib": args.system_memory_total_mib,
            "system_memory_available_before_launch_mib": (
                args.system_memory_available_before_launch_mib
            ),
            "updates": args.updates,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "teacher_action_scales": {
                "position_m": args.position_scale,
                "orientation_rad": args.orientation_scale,
            },
            "teacher_controller": (
                (
                    {
                        "approach_height_m": 0.02,
                        "grasp_height_m": 0.0,
                        "grasp_offset_m": list(
                            NEEDLE_PROVISIONAL_GRASP_OFFSET_M
                        ),
                        "grasp_offset_source": (
                            f"{NEEDLE_GEOMETRY_GRASP_OFFSET_SOURCE};"
                            f"provisional_arc_fraction="
                            f"{NEEDLE_PROVISIONAL_ARC_FRACTION};"
                            f"provisional_z_offset_m="
                            f"{NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M}"
                        ),
                        "lateral_alignment_threshold_m": 0.005,
                        "close_distance_to_grasp_m": 0.005,
                        "slow_approach_radius_m": 0.02,
                        "slow_approach_action_limit": 0.1,
                        "normalized_contact_threshold": 0.002,
                        "lateral_clearance_below_target_m": 0.04,
                        "carry_latch_below_target_m": 0.062,
                        "carry_action_limit": 0.1,
                        "carry_lateral_action_limit": 0.1,
                        "carry_vertical_action_limit": 0.18,
                        "carry_orientation_action_limit": 0.035,
                        "carry_orientation_scale_rad": args.orientation_scale,
                        "carry_orientation_velocity_damping_s": 0.001,
                        "carry_target_height_offset_m": 0.0,
                        "qualification_status": "provisional_not_stage_qualified",
                    }
                    if "Lift-Needle-PSM-IK-Rel" in args.task
                    else {
                    "approach_height_m": 0.02,
                    "grasp_height_m": 0.0,
                    "grasp_offset_m": list(BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_M),
                    "grasp_offset_source": BLOCK_CONTACT_CALIBRATED_GRASP_OFFSET_SOURCE,
                    "lateral_alignment_threshold_m": 0.005,
                    "close_distance_to_grasp_m": 0.005,
                    "slow_approach_radius_m": 0.02,
                    "slow_approach_action_limit": 0.1,
                    "normalized_contact_threshold": 0.002,
                    "lateral_clearance_below_target_m": 0.04,
                    "carry_latch_below_target_m": 0.062,
                    "carry_action_limit": 0.1,
                    "carry_lateral_action_limit": 0.1,
                    "carry_vertical_action_limit": 0.18,
                    "carry_target_height_offset_m": 0.0,
                    }
                )
                if "Lift-" in args.task
                else None
            ),
            "loss": {
                "initial": losses[0] if losses else None,
                "final": losses[-1] if losses else None,
                "minimum": min(losses) if losses else None,
            },
            "teacher_rollout": {
                "completed_episodes": teacher_completed,
                "successful_episodes": teacher_successes,
                "success_rate": (
                    teacher_successes / teacher_completed
                    if teacher_completed
                    else None
                ),
            },
            "deterministic_validation": {
                "frames_per_env": args.validation_frames,
                "completed_episodes": validation_completed,
                "successful_episodes": validation_successes,
                "success_rate": (
                    validation_successes / validation_completed
                    if validation_completed
                    else None
                ),
                "pose_diagnostics": pose_diagnostics,
                "pose_diagnostic_trace": pose_diagnostic_trace,
            },
            "wall_time_s": duration,
            "simulated_frames": simulated_frames,
            "total_fps": simulated_frames / duration if duration > 0 else None,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
            },
            "gpu_peak_memory_bytes": (
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else None
            ),
            "process_peak_memory_mib": _peak_process_memory_mib(),
            "runtime": _runtime_evidence(repo_root),
        }
        _write_evidence(Path(args.output_path), "dranmar_pretraining", evidence)
        return 0
    finally:
        env.close()


class _EarlyStopConverged(Exception):
    """Raised after the direct simulator success rate satisfies its gate."""


class _TerminationSuccessEarlyStop:
    """Track exact episode success from Isaac Lab termination tensors."""

    def __init__(
        self,
        env,
        runner,
        threshold: float,
        window: int,
        num_steps_per_env: int,
        stop_on_convergence: bool = True,
        success_term: str = "success",
    ) -> None:
        self.env = env
        self.runner = runner
        self.threshold = threshold
        self.window = window
        self.num_steps_per_env = num_steps_per_env
        self.stop_on_convergence = stop_on_convergence
        self.success_term = success_term
        self.history: list[float] = []
        self._step_count = 0
        self._iteration_successes = 0
        self._iteration_completed = 0
        self._orig_step = env.step
        self.tracker = self

    def __enter__(self):
        self.env.step = self._step
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.env.step = self._orig_step
        if exc_type is _EarlyStopConverged:
            self._runner_cleanup()
            print(
                "[DrAnmar] Early stop: direct termination success converged at "
                f"iteration {self.framework_iteration_count} "
                f"(tail mean {self.tail_mean:.4f})"
            )
            return True
        return False

    def _step(self, actions):
        result = self._orig_step(actions)
        dones = result[2]
        success = self.env.unwrapped.termination_manager.get_term(self.success_term)
        self._iteration_successes += int(success.sum().item())
        self._iteration_completed += int(dones.sum().item())
        self._step_count += 1

        current_rate = (
            self._iteration_successes / self._iteration_completed
            if self._iteration_completed
            else 0.0
        )
        result[3].setdefault("log", {})["Metrics/success_rate"] = current_rate

        if self._step_count % self.num_steps_per_env == 0:
            self.history.append(current_rate)
            self._iteration_successes = 0
            self._iteration_completed = 0
            if self.stop_on_convergence and self.converged:
                raise _EarlyStopConverged()
        return result

    def _runner_cleanup(self) -> None:
        if self.runner.logger.writer is not None:
            iteration = self.runner.current_learning_iteration
            self.runner.save(
                os.path.join(self.runner.logger.log_dir, f"model_{iteration}.pt")
            )
            self.runner.logger.stop_logging_writer()

    @property
    def framework_iteration_count(self) -> int:
        return max(
            self._step_count // self.num_steps_per_env,
            self.runner.current_learning_iteration + 1,
        )

    @property
    def converged(self) -> bool:
        return len(self.history) >= self.window and all(
            value >= self.threshold for value in self.history[-self.window :]
        )

    @property
    def tail_mean(self) -> float:
        if not self.history:
            return 0.0
        return statistics.mean(self.history[-self.window :])


def _train(args: argparse.Namespace, repo_root: Path) -> int:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env_cfg, agent_cfg = _load_configs(args.task, args.num_envs, args.seed)
    agent_cfg.max_iterations = args.max_iterations
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.output_path).resolve() / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(run_dir)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)
    runner.logger.git_status_repos = []

    started = time.perf_counter()
    early = _TerminationSuccessEarlyStop(
        env,
        runner,
        threshold=args.success_threshold,
        window=args.success_window,
        num_steps_per_env=agent_cfg.num_steps_per_env,
        stop_on_convergence=args.check_success,
    )
    try:
        with early:
            runner.learn(
                num_learning_iterations=agent_cfg.max_iterations,
                init_at_random_ep_len=True,
            )
        duration = time.perf_counter() - started
        checkpoint = run_dir / "model_final.pt"
        runner.save(str(checkpoint))
        iterations = max(1, early.framework_iteration_count)
        simulated_frames = (
            env.unwrapped.num_envs * agent_cfg.num_steps_per_env * iterations
        )
        success_history = [float(value) for value in early.tracker.history]
        evidence = {
            "schema_version": "dranmar-learning-evidence-1.0",
            "kind": "training",
            "task": args.task,
            "seed": args.seed,
            "requested_num_envs": args.requested_num_envs,
            "num_envs": env.unwrapped.num_envs,
            "trusted_requested_num_envs": args.trusted_requested_num_envs,
            "free_gpu_memory_before_launch_mib": args.free_gpu_memory_before_launch_mib,
            "system_memory_total_mib": args.system_memory_total_mib,
            "system_memory_available_before_launch_mib": (
                args.system_memory_available_before_launch_mib
            ),
            "rollout_steps_per_env": agent_cfg.num_steps_per_env,
            "iterations_requested": agent_cfg.max_iterations,
            "iterations_completed": iterations,
            "wall_time_s": duration,
            "simulated_frames": simulated_frames,
            "total_fps": simulated_frames / duration if duration > 0 else None,
            "success": {
                "threshold": args.success_threshold,
                "window": args.success_window,
                "history": success_history,
                "tail_mean": (
                    float(early.tracker.tail_mean) if success_history else None
                ),
                "converged": bool(early.tracker.converged),
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
            },
            "gpu_peak_memory_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
            ),
            "process_peak_memory_mib": _peak_process_memory_mib(),
            "runtime": _runtime_evidence(repo_root),
        }
        _write_evidence(Path(args.output_path), "dranmar_training", evidence)
        return 0
    finally:
        env.close()


def _lift_procedure_snapshot(env) -> dict[str, Any]:
    """Summarize simulator-owned lift geometry, motion, and contact state."""
    import torch

    from isaaclab.managers import SceneEntityCfg
    from orbit.surgical.tasks.surgical import mdp_common

    unwrapped = env.unwrapped
    object_pos = mdp_common.as_torch(unwrapped.scene["object"].data.root_pos_w)
    ee_pos = mdp_common.as_torch(
        unwrapped.scene["ee_frame"].data.target_pos_w
    )[:, 0, :]
    distance = torch.linalg.vector_norm(object_pos - ee_pos, dim=-1)
    forces = mdp_common.paired_contact_forces(
        unwrapped,
        "jaw_1_object_contact",
        "jaw_2_object_contact",
    )
    non_object_forces = torch.stack(
        (
            mdp_common.non_object_contact_force_magnitude(
                unwrapped, "jaw_1_object_contact"
            ),
            mdp_common.non_object_contact_force_magnitude(
                unwrapped, "jaw_2_object_contact"
            ),
        ),
        dim=-1,
    )
    motion = mdp_common.object_motion(unwrapped)
    goal_position_error, goal_orientation_error = mdp_common.object_goal_errors(
        unwrapped,
        "object_pose",
        SceneEntityCfg("robot"),
        SceneEntityCfg("object"),
    )

    def stats(value) -> dict[str, float]:
        return {
            "minimum": float(value.min().item()),
            "mean": float(value.float().mean().item()),
            "maximum": float(value.max().item()),
        }

    return {
        "object_height_m": stats(object_pos[:, 2]),
        "end_effector_object_distance_m": stats(distance),
        "jaw_object_force_n": stats(forces),
        "jaw_non_object_force_n": stats(non_object_forces),
        "object_linear_speed_m_s": stats(motion[:, 0]),
        "object_angular_speed_rad_s": stats(motion[:, 1]),
        "goal_position_error_m": stats(goal_position_error),
        "goal_orientation_error_rad": stats(goal_orientation_error),
        "bilateral_contact_fraction": float(
            torch.all(forces > 0.01, dim=-1).float().mean().item()
        ),
    }


def _probe(args: argparse.Namespace, repo_root: Path) -> int:
    """Exercise a task without training and record its native runtime contract."""
    import gymnasium as gym
    import torch

    env_cfg, _ = _load_configs(args.task, args.num_envs, args.seed)
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    initial_procedure_state = (
        _lift_procedure_snapshot(env) if "Lift-" in args.task else None
    )
    manager = env.unwrapped.termination_manager
    term_counts = {name: 0 for name in manager.active_terms}
    action_manager = env.unwrapped.action_manager
    action_dim = getattr(action_manager, "total_action_dim", None)
    if action_dim is None:
        action_dim = action_manager.action_dim
    action = torch.zeros(
        env.unwrapped.num_envs,
        action_dim,
        device=env.unwrapped.device,
    )
    done_count = 0
    started = time.perf_counter()
    try:
        for _ in range(args.num_frames):
            obs, _, terminated, time_outs, _ = env.step(action)
            dones = terminated | time_outs
            done_count += int(dones.sum().item())
            for name in manager.active_terms:
                term_counts[name] += int(manager.get_term(name).sum().item())
        duration = time.perf_counter() - started
        final_procedure_state = (
            _lift_procedure_snapshot(env) if "Lift-" in args.task else None
        )
        evidence = {
            "schema_version": "dranmar-learning-evidence-1.0",
            "kind": "task_probe",
            "task": args.task,
            "seed": args.seed,
            "requested_num_envs": args.requested_num_envs,
            "num_envs": env.unwrapped.num_envs,
            "trusted_requested_num_envs": args.trusted_requested_num_envs,
            "free_gpu_memory_before_launch_mib": args.free_gpu_memory_before_launch_mib,
            "system_memory_total_mib": args.system_memory_total_mib,
            "system_memory_available_before_launch_mib": (
                args.system_memory_available_before_launch_mib
            ),
            "frames_per_env": args.num_frames,
            "policy_observation_shape": list(obs["policy"].shape),
            "action_shape": [env.unwrapped.num_envs, action_dim],
            "completed_episodes": done_count,
            "termination_term_counts": term_counts,
            "initial_procedure_state": initial_procedure_state,
            "final_procedure_state": final_procedure_state,
            "wall_time_s": duration,
            "total_fps": (
                env.unwrapped.num_envs * args.num_frames / duration
                if duration > 0
                else None
            ),
            "process_peak_memory_mib": _peak_process_memory_mib(),
            "runtime": _runtime_evidence(repo_root),
        }
        _write_evidence(Path(args.output_path), "dranmar_probe", evidence)
        return 0
    finally:
        env.close()


def _handover_controller_sweep(
    args: argparse.Namespace,
    repo_root: Path,
) -> int:
    """Compare Arm 2 grasp points for the staged physical needle handover."""
    import math

    import gymnasium as gym
    import torch

    from orbit.surgical.tasks.surgical import mdp_common
    from orbit.surgical.tasks.surgical.lift.grasp_frames import (
        NEEDLE_GEOMETRY_GRASP_OFFSET_SOURCE,
        NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
        NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M,
        needle_geometry_grasp_offset_m,
    )

    if "Handover-Needle-Dual-PSM-IK-Rel" not in args.task:
        return _fail("handover-sweep requires the dual-PSM IK-relative needle task")
    values = [float(value) for value in args.values.split(",") if value.strip()]
    if len(values) < 2:
        return _fail("handover-sweep requires at least two values")
    if args.num_envs % len(values):
        return _fail("number of environments must divide evenly across sweep values")
    parameter = args.parameter
    receiver_offsets = []
    receiver_roll_offsets = [0.0] * len(values)
    fixed_receiver_arc_fraction = 0.65
    if parameter == "receiver_arc_fraction":
        if any(not 0.0 <= value <= 1.0 for value in values):
            return _fail("receiver arc fractions must be between 0.0 and 1.0")
        for value in values:
            geometry_offset = needle_geometry_grasp_offset_m(value)
            receiver_offsets.append(
                (
                    geometry_offset[0],
                    geometry_offset[1],
                    NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M,
                )
            )
    elif parameter == "receiver_grasp_z_offset":
        if any(not -0.02 <= value <= 0.02 for value in values):
            return _fail("receiver grasp z offsets must be within +/- 0.02 m")
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (geometry_offset[0], geometry_offset[1], value)
            for value in values
        ]
    elif parameter == "receiver_roll_offset_rad":
        if any(not -math.pi <= value <= math.pi for value in values):
            return _fail("receiver roll offsets must be within +/- pi rad")
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (
                geometry_offset[0],
                geometry_offset[1],
                NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M,
            )
            for _ in values
        ]
        receiver_roll_offsets = values
    else:
        return _fail(
            "handover-sweep parameter must be receiver_arc_fraction "
            "receiver_grasp_z_offset, or receiver_roll_offset_rad"
        )

    env_cfg, _ = _load_configs(args.task, args.num_envs, args.seed)
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()

    def world_state_snapshot():
        robot = env.unwrapped.scene["robot_1"]
        obj = env.unwrapped.scene["object"]
        ee_frame = env.unwrapped.scene["ee_1_frame"]
        return {
            "robot_root_position_world_m": (
                mdp_common.as_torch(robot.data.root_pos_w)[0].tolist()
            ),
            "robot_root_orientation_world_wxyz": (
                mdp_common.as_torch(robot.data.root_quat_w)[0].tolist()
            ),
            "ee_position_world_m": (
                mdp_common.as_torch(
                    ee_frame.data.target_pos_w
                )[0, 0].tolist()
            ),
            "object_position_world_m": (
                mdp_common.as_torch(obj.data.root_pos_w)[0].tolist()
            ),
            "object_default_root_pose": (
                mdp_common.as_torch(obj.data.default_root_pose)[0].tolist()
            ),
        }

    group_size = env.unwrapped.num_envs // len(values)
    unresolved = torch.ones(
        env.unwrapped.num_envs,
        dtype=torch.bool,
        device=env.unwrapped.device,
    )
    max_phase = torch.argmax(obs["policy"][:, 77:82], dim=-1)
    completed = torch.zeros(len(values), dtype=torch.int64, device=env.unwrapped.device)
    successes = torch.zeros_like(completed)
    timeouts = torch.zeros_like(completed)
    hard_failures = torch.zeros_like(completed)
    failure_term_counts = {
        name: torch.zeros_like(completed)
        for name in (
            "object_dropping",
            "excessive_object_force",
            "protected_surface_force",
        )
    }
    maximum_object_force = torch.zeros(
        len(values),
        dtype=torch.float64,
        device=env.unwrapped.device,
    )
    maximum_non_object_force = torch.zeros_like(maximum_object_force)
    minimum_giver_grasp_distance = torch.full(
        (len(values),),
        float("inf"),
        dtype=torch.float64,
        device=env.unwrapped.device,
    )
    maximum_giver_translation_action = torch.zeros_like(
        minimum_giver_grasp_distance
    )
    maximum_object_height = torch.full_like(
        minimum_giver_grasp_distance,
        float("-inf"),
    )
    minimum_receiver_distance = torch.full_like(
        minimum_giver_grasp_distance,
        float("inf"),
    )
    maximum_giver_bilateral_contact = torch.zeros_like(
        minimum_giver_grasp_distance
    )
    maximum_receiver_bilateral_contact = torch.zeros_like(
        minimum_giver_grasp_distance
    )
    minimum_receiver_grasp_distance = torch.full_like(
        minimum_giver_grasp_distance,
        float("inf"),
    )
    maximum_receiver_jaw_1_contact = torch.zeros_like(
        minimum_giver_grasp_distance
    )
    maximum_receiver_jaw_2_contact = torch.zeros_like(
        minimum_giver_grasp_distance
    )
    initial_giver_state = {
        "ee_position_robot_frame_m": obs["policy"][0, 32:35].tolist(),
        "object_position_robot_frame_m": obs["policy"][0, 46:49].tolist(),
        "world": world_state_snapshot(),
    }
    manager = env.unwrapped.termination_manager
    sensor_names = (
        "robot_1_jaw_1_object_contact",
        "robot_1_jaw_2_object_contact",
        "robot_2_jaw_1_object_contact",
        "robot_2_jaw_2_object_contact",
    )
    started = time.perf_counter()
    try:
        for _ in range(args.num_frames):
            was_unresolved = unresolved.clone()
            current_phase = torch.argmax(obs["policy"][:, 77:82], dim=-1)
            max_phase = torch.maximum(max_phase, current_phase)
            actions = torch.zeros(
                env.unwrapped.num_envs,
                14,
                device=env.unwrapped.device,
            )
            for group_index, receiver_offset in enumerate(receiver_offsets):
                start = group_index * group_size
                stop = start + group_size
                group_obs = {
                    name: observation[start:stop]
                    for name, observation in obs.items()
                }
                actions[start:stop] = _handover_teacher_action(
                    group_obs,
                    receiver_grasp_offset=receiver_offset,
                    receiver_roll_offset_rad=(
                        receiver_roll_offsets[group_index]
                    ),
                )
                giver_ee = group_obs["policy"][:, 32:35]
                giver_grasp = group_obs["policy"][:, 46:49].clone()
                giver_grasp[:, 0] += NEEDLE_PROVISIONAL_GRASP_OFFSET_M[0]
                giver_grasp[:, 1] += NEEDLE_PROVISIONAL_GRASP_OFFSET_M[1]
                giver_grasp[:, 2] += NEEDLE_PROVISIONAL_GRASP_OFFSET_M[2]
                minimum_giver_grasp_distance[group_index] = torch.minimum(
                    minimum_giver_grasp_distance[group_index],
                    torch.linalg.vector_norm(
                        giver_grasp - giver_ee,
                        dim=-1,
                    ).min().double(),
                )
                maximum_giver_translation_action[group_index] = torch.maximum(
                    maximum_giver_translation_action[group_index],
                    actions[start:stop, :3].abs().max().double(),
                )
                receiver_ee = group_obs["policy"][:, 39:42]
                receiver_grasp = group_obs["policy"][:, 53:56].clone()
                receiver_grasp[:, 0] += receiver_offset[0]
                receiver_grasp[:, 1] += receiver_offset[1]
                receiver_grasp[:, 2] += receiver_offset[2]
                minimum_receiver_grasp_distance[group_index] = torch.minimum(
                    minimum_receiver_grasp_distance[group_index],
                    torch.linalg.vector_norm(
                        receiver_grasp - receiver_ee,
                        dim=-1,
                    ).min().double(),
                )

            obs, _, terminated, time_out_flags, _ = env.step(actions)
            dones = terminated | time_out_flags
            success_term = manager.get_term("success")
            failure_terms = {
                name: manager.get_term(name)
                for name in failure_term_counts
            }
            hard_failure = (
                failure_terms["object_dropping"]
                | failure_terms["excessive_object_force"]
                | failure_terms["protected_surface_force"]
            )
            time_out_term = manager.get_term("time_out")
            first_done = was_unresolved & dones
            max_phase = torch.where(
                first_done & success_term & ~hard_failure,
                torch.full_like(max_phase, 4),
                max_phase,
            )

            giver_forces = mdp_common.paired_contact_forces(
                env.unwrapped,
                sensor_names[0],
                sensor_names[1],
            )
            receiver_forces = mdp_common.paired_contact_forces(
                env.unwrapped,
                sensor_names[2],
                sensor_names[3],
            )
            object_forces = torch.cat(
                (giver_forces, receiver_forces),
                dim=-1,
            )
            non_object_forces = mdp_common.maximum_non_object_contact_force(
                env.unwrapped,
                sensor_names,
            )
            object_height = mdp_common.as_torch(
                env.unwrapped.scene["object"].data.root_pos_w
            )[:, 2]
            receiver_position = mdp_common.as_torch(
                env.unwrapped.scene["ee_2_frame"].data.target_pos_w
            )[:, 0, :]
            receiver_distance = torch.linalg.vector_norm(
                receiver_position
                - mdp_common.as_torch(
                    env.unwrapped.scene["object"].data.root_pos_w
                ),
                dim=-1,
            )
            for group_index in range(len(values)):
                start = group_index * group_size
                stop = start + group_size
                active = was_unresolved[start:stop]
                first = first_done[start:stop]
                completed[group_index] += first.sum()
                successes[group_index] += (
                    first
                    & success_term[start:stop]
                    & ~hard_failure[start:stop]
                ).sum()
                hard_failures[group_index] += (
                    first & hard_failure[start:stop]
                ).sum()
                for name, term in failure_terms.items():
                    failure_term_counts[name][group_index] += (
                        first & term[start:stop]
                    ).sum()
                timeouts[group_index] += (
                    first
                    & time_out_term[start:stop]
                    & ~success_term[start:stop]
                    & ~hard_failure[start:stop]
                ).sum()
                if active.any():
                    maximum_object_force[group_index] = torch.maximum(
                        maximum_object_force[group_index],
                        object_forces[start:stop][active].max().double(),
                    )
                    maximum_non_object_force[group_index] = torch.maximum(
                        maximum_non_object_force[group_index],
                        non_object_forces[start:stop][active].max().double(),
                    )
                    maximum_object_height[group_index] = torch.maximum(
                        maximum_object_height[group_index],
                        object_height[start:stop][active].max().double(),
                    )
                    minimum_receiver_distance[group_index] = torch.minimum(
                        minimum_receiver_distance[group_index],
                        receiver_distance[start:stop][active].min().double(),
                    )
                    maximum_giver_bilateral_contact[group_index] = torch.maximum(
                        maximum_giver_bilateral_contact[group_index],
                        giver_forces[start:stop][active]
                        .min(dim=-1)
                        .values.max()
                        .double(),
                    )
                    maximum_receiver_bilateral_contact[group_index] = torch.maximum(
                        maximum_receiver_bilateral_contact[group_index],
                        receiver_forces[start:stop][active]
                        .min(dim=-1)
                        .values.max()
                        .double(),
                    )
                    maximum_receiver_jaw_1_contact[group_index] = torch.maximum(
                        maximum_receiver_jaw_1_contact[group_index],
                        receiver_forces[start:stop, 0][active].max().double(),
                    )
                    maximum_receiver_jaw_2_contact[group_index] = torch.maximum(
                        maximum_receiver_jaw_2_contact[group_index],
                        receiver_forces[start:stop, 1][active].max().double(),
                    )
            unresolved &= ~first_done
            if not unresolved.any():
                break

        duration = time.perf_counter() - started
        results = []
        for group_index, value in enumerate(values):
            start = group_index * group_size
            stop = start + group_size
            completed_count = int(completed[group_index].item())
            success_count = int(successes[group_index].item())
            group_max_phase = max_phase[start:stop]
            results.append(
                {
                    "parameter": parameter,
                    "parameter_value": value,
                    "receiver_grasp_arc_fraction": (
                        value
                        if parameter == "receiver_arc_fraction"
                        else fixed_receiver_arc_fraction
                    ),
                    "receiver_grasp_z_offset_m": (
                        receiver_offsets[group_index][2]
                    ),
                    "receiver_roll_offset_rad": (
                        receiver_roll_offsets[group_index]
                    ),
                    "receiver_grasp_offset_m": list(
                        receiver_offsets[group_index]
                    ),
                    "assigned_environments": group_size,
                    "completed_episodes": completed_count,
                    "successful_episodes": success_count,
                    "success_rate": (
                        success_count / completed_count
                        if completed_count
                        else None
                    ),
                    "time_outs": int(timeouts[group_index].item()),
                    "hard_failures": int(
                        hard_failures[group_index].item()
                    ),
                    "hard_failure_term_counts": {
                        name: int(counts[group_index].item())
                        for name, counts in failure_term_counts.items()
                    },
                    "maximum_phase_reached": {
                        str(phase): int((group_max_phase == phase).sum().item())
                        for phase in range(5)
                    },
                    "maximum_object_force_n": float(
                        maximum_object_force[group_index].item()
                    ),
                    "maximum_non_object_force_n": float(
                        maximum_non_object_force[group_index].item()
                    ),
                    "minimum_giver_grasp_distance_m": float(
                        minimum_giver_grasp_distance[group_index].item()
                    ),
                    "maximum_giver_translation_action": float(
                        maximum_giver_translation_action[group_index].item()
                    ),
                    "maximum_object_height_m": float(
                        maximum_object_height[group_index].item()
                    ),
                    "minimum_receiver_distance_m": float(
                        minimum_receiver_distance[group_index].item()
                    ),
                    "maximum_giver_bilateral_contact_n": float(
                        maximum_giver_bilateral_contact[group_index].item()
                    ),
                    "maximum_receiver_bilateral_contact_n": float(
                        maximum_receiver_bilateral_contact[group_index].item()
                    ),
                    "minimum_receiver_grasp_distance_m": float(
                        minimum_receiver_grasp_distance[group_index].item()
                    ),
                    "maximum_receiver_jaw_1_contact_n": float(
                        maximum_receiver_jaw_1_contact[group_index].item()
                    ),
                    "maximum_receiver_jaw_2_contact_n": float(
                        maximum_receiver_jaw_2_contact[group_index].item()
                    ),
                }
            )
        evidence = {
            "schema_version": "dranmar-handover-sweep-evidence-1.0",
            "kind": "handover_controller_sweep",
            "task": args.task,
            "seed": args.seed,
            "num_envs": env.unwrapped.num_envs,
            "frames_per_env": args.num_frames,
            "first_terminal_outcome_per_environment": True,
            "giver": "robot_1",
            "receiver": "robot_2",
            "giver_grasp_arc_fraction": 0.4,
            "parameter": parameter,
            "initial_giver_state": initial_giver_state,
            "final_giver_state": {
                "ee_position_robot_frame_m": (
                    obs["policy"][0, 32:35].tolist()
                ),
                "object_position_robot_frame_m": (
                    obs["policy"][0, 46:49].tolist()
                ),
                "world": world_state_snapshot(),
            },
            "receiver_grasp_frame_source": (
                NEEDLE_GEOMETRY_GRASP_OFFSET_SOURCE
            ),
            "values": values,
            "results": results,
            "wall_time_s": duration,
            "total_fps": (
                env.unwrapped.num_envs * args.num_frames / duration
                if duration > 0
                else None
            ),
            "process_peak_memory_mib": _peak_process_memory_mib(),
            "runtime": _runtime_evidence(repo_root),
        }
        _write_evidence(
            Path(args.output_path),
            "dranmar_handover_sweep",
            evidence,
        )
        return 0
    finally:
        env.close()


_LIFT_SWEEP_PARAMETERS = {
    "carry_action_limit",
    "carry_goal_action_limit",
    "carry_lateral_action_limit",
    "carry_orientation_action_limit",
    "carry_orientation_velocity_damping_s",
    "carry_target_height_offset",
    "carry_vertical_action_limit",
    "close_distance",
    "gripper_close_rad",
    "gripper_effort_limit_nm",
    "lateral_alignment_threshold",
    "lateral_clearance_below_target",
    "needle_grasp_arc_fraction",
    "needle_grasp_z_offset",
    "slow_approach_action_limit",
}
_ENVIRONMENT_LEVEL_LIFT_SWEEP_PARAMETERS = {
    "gripper_close_rad",
    "gripper_effort_limit_nm",
}
_NEEDLE_PROVISIONAL_ORIENTATION_ACTION_LIMIT = 0.035
_NEEDLE_PROVISIONAL_ORIENTATION_VELOCITY_DAMPING_S = 0.001


def _controller_sweep(args: argparse.Namespace, repo_root: Path) -> int:
    """Compare analytic lift-controller variants from one shared reset batch."""
    import gymnasium as gym
    import torch

    from isaaclab.managers import SceneEntityCfg
    from orbit.surgical.tasks.surgical import mdp_common

    block_task = "Lift-Block-PSM-IK-Rel" in args.task
    needle_task = "Lift-Needle-PSM-IK-Rel" in args.task
    if not (block_task or needle_task):
        return _fail("controller-sweep requires a block or needle IK-relative lift task")
    if args.parameter not in _LIFT_SWEEP_PARAMETERS:
        return _fail(f"unsupported controller-sweep parameter: {args.parameter}")
    needle_grasp_sweep = args.parameter in {
        "needle_grasp_arc_fraction",
        "needle_grasp_z_offset",
    }
    needle_orientation_sweep = needle_task and args.parameter in {
        "carry_orientation_action_limit",
        "carry_orientation_velocity_damping_s",
    }
    needle_orientation_damping_sweep = (
        needle_task
        and args.parameter == "carry_orientation_velocity_damping_s"
    )
    needle_transport_sweep = needle_task and args.parameter in {
        "carry_action_limit",
        "carry_goal_action_limit",
        "carry_lateral_action_limit",
        "carry_vertical_action_limit",
        "lateral_clearance_below_target",
    }
    needle_approach_sweep = needle_task and args.parameter in {
        "close_distance",
        "lateral_alignment_threshold",
        "slow_approach_action_limit",
    }
    needle_environment_sweep = (
        needle_task
        and args.parameter in _ENVIRONMENT_LEVEL_LIFT_SWEEP_PARAMETERS
    )
    if needle_task and not (
        needle_grasp_sweep
        or needle_orientation_sweep
        or needle_transport_sweep
        or needle_approach_sweep
        or needle_environment_sweep
    ):
        return _fail(
            "needle controller-sweep requires a needle grasp-frame or "
            "approach/carry-controller parameter, or a full-population physical "
            "challenger until a contact-qualified needle controller exists"
        )
    if block_task and needle_grasp_sweep:
        return _fail("needle_grasp_arc_fraction requires the needle lift task")
    values = [float(value) for value in args.values.split(",") if value.strip()]
    if len(values) < 2:
        return _fail("controller-sweep requires at least two comma-separated values")
    if args.parameter == "carry_orientation_action_limit" and any(
        value <= 0.0 for value in values
    ):
        return _fail("carry_orientation_action_limit must be positive")
    if args.parameter == "carry_orientation_velocity_damping_s" and any(
        value < 0.0 for value in values
    ):
        return _fail("carry_orientation_velocity_damping_s must be non-negative")
    if args.parameter == "carry_goal_action_limit" and any(
        value <= 0.0 for value in values
    ):
        return _fail("carry_goal_action_limit must be positive")
    if args.num_envs % len(values):
        return _fail("number of environments must divide evenly across sweep values")
    environment_level_parameter = (
        args.parameter in _ENVIRONMENT_LEVEL_LIFT_SWEEP_PARAMETERS
    )
    if environment_level_parameter and len(set(values)) != 1:
        return _fail(
            f"{args.parameter} is environment-level; repeat one value "
            "to run a full-population qualification"
        )

    needle_grasp_offsets: list[tuple[float, float, float] | None] = [
        None for _ in values
    ]
    grasp_frame_source = None
    if needle_grasp_sweep:
        from orbit.surgical.tasks.surgical.lift.grasp_frames import (
            NEEDLE_GEOMETRY_GRASP_OFFSET_SOURCE,
            NEEDLE_PROVISIONAL_ARC_FRACTION,
            NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M,
            needle_geometry_grasp_offset_m,
        )

        if args.parameter == "needle_grasp_arc_fraction":
            try:
                geometry_offsets = [
                    needle_geometry_grasp_offset_m(value) for value in values
                ]
            except ValueError as error:
                return _fail(str(error))
            needle_grasp_offsets = [
                (offset[0], offset[1], NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M)
                for offset in geometry_offsets
            ]
            grasp_frame_source = (
                f"{NEEDLE_GEOMETRY_GRASP_OFFSET_SOURCE};"
                f"fixed_z_offset_m={NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M}"
            )
        else:
            provisional_offset = needle_geometry_grasp_offset_m(
                NEEDLE_PROVISIONAL_ARC_FRACTION
            )
            needle_grasp_offsets = [
                (provisional_offset[0], provisional_offset[1], value)
                for value in values
            ]
            grasp_frame_source = (
                f"{NEEDLE_GEOMETRY_GRASP_OFFSET_SOURCE};"
                f"fixed_arc_fraction={NEEDLE_PROVISIONAL_ARC_FRACTION}"
            )
    elif (
        needle_orientation_sweep
        or needle_transport_sweep
        or needle_approach_sweep
        or needle_environment_sweep
    ):
        from orbit.surgical.tasks.surgical.lift.grasp_frames import (
            NEEDLE_GEOMETRY_GRASP_OFFSET_SOURCE,
            NEEDLE_PROVISIONAL_ARC_FRACTION,
            NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M,
            needle_geometry_grasp_offset_m,
        )

        provisional_offset = needle_geometry_grasp_offset_m(
            NEEDLE_PROVISIONAL_ARC_FRACTION
        )
        needle_grasp_offsets = [
            (
                provisional_offset[0],
                provisional_offset[1],
                NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M,
            )
            for _ in values
        ]
        grasp_frame_source = (
            f"{NEEDLE_GEOMETRY_GRASP_OFFSET_SOURCE};"
            f"fixed_arc_fraction={NEEDLE_PROVISIONAL_ARC_FRACTION};"
            f"fixed_z_offset_m={NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M}"
        )

    env_cfg, _ = _load_configs(args.task, args.num_envs, args.seed)
    environment_override = None
    if args.parameter == "gripper_close_rad":
        if not 0.0 <= values[0] <= 0.5:
            return _fail("gripper_close_rad must be between 0.0 and 0.5")
        from orbit.surgical.assets.psm import psm_gripper_command_expr

        env_cfg.actions.gripper_action.close_command_expr = (
            psm_gripper_command_expr(values[0])
        )
        environment_override = {
            "gripper_close_rad": values[0],
            "close_command_expr": (
                env_cfg.actions.gripper_action.close_command_expr
            ),
        }
    elif args.parameter == "gripper_effort_limit_nm":
        if values[0] <= 0.0:
            return _fail("gripper_effort_limit_nm must be positive")
        env_cfg.scene.robot.actuators["psm_tool"].effort_limit_sim = values[0]
        environment_override = {
            "gripper_effort_limit_nm": values[0],
        }
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    group_size = env.unwrapped.num_envs // len(values)
    unresolved = torch.ones(
        env.unwrapped.num_envs,
        dtype=torch.bool,
        device=env.unwrapped.device,
    )
    completed = torch.zeros(len(values), dtype=torch.int64, device=env.unwrapped.device)
    successes = torch.zeros_like(completed)
    timeouts = torch.zeros_like(completed)
    hard_failures = torch.zeros_like(completed)
    failure_term_counts = {
        name: torch.zeros_like(completed)
        for name in (
            "object_dropping",
            "excessive_object_force",
            "protected_surface_force",
        )
    }
    contact_frames = torch.zeros(len(values), dtype=torch.float64, device=env.unwrapped.device)
    angular_valid_frames = torch.zeros_like(contact_frames)
    above_height_frames = torch.zeros_like(contact_frames)
    goal_position_frames = torch.zeros_like(contact_frames)
    goal_orientation_frames = torch.zeros_like(contact_frames)
    active_samples = torch.zeros_like(contact_frames)
    maximum_force = torch.zeros_like(contact_frames)
    maximum_non_object_force = torch.zeros_like(contact_frames)
    manager = env.unwrapped.termination_manager
    started = time.perf_counter()
    try:
        for _ in range(args.num_frames):
            was_unresolved = unresolved.clone()
            actions = torch.zeros(
                env.unwrapped.num_envs,
                7,
                device=env.unwrapped.device,
            )
            for group_index, value in enumerate(values):
                start = group_index * group_size
                stop = start + group_size
                if needle_grasp_sweep:
                    controller_kwargs = {
                        "grasp_offset": needle_grasp_offsets[group_index],
                        "carry_orientation_action_limit": (
                            _NEEDLE_PROVISIONAL_ORIENTATION_ACTION_LIMIT
                        ),
                        "carry_orientation_velocity_damping_s": (
                            _NEEDLE_PROVISIONAL_ORIENTATION_VELOCITY_DAMPING_S
                        ),
                    }
                elif needle_orientation_sweep:
                    controller_kwargs = {
                        "grasp_offset": needle_grasp_offsets[group_index],
                        args.parameter: value,
                    }
                    if needle_orientation_damping_sweep:
                        controller_kwargs["carry_orientation_action_limit"] = (
                            _NEEDLE_PROVISIONAL_ORIENTATION_ACTION_LIMIT
                        )
                    else:
                        controller_kwargs[
                            "carry_orientation_velocity_damping_s"
                        ] = _NEEDLE_PROVISIONAL_ORIENTATION_VELOCITY_DAMPING_S
                elif needle_environment_sweep:
                    controller_kwargs = {
                        "grasp_offset": needle_grasp_offsets[group_index],
                        "carry_orientation_action_limit": (
                            _NEEDLE_PROVISIONAL_ORIENTATION_ACTION_LIMIT
                        ),
                        "carry_orientation_velocity_damping_s": (
                            _NEEDLE_PROVISIONAL_ORIENTATION_VELOCITY_DAMPING_S
                        ),
                    }
                elif needle_transport_sweep or needle_approach_sweep:
                    controller_kwargs = {
                        "grasp_offset": needle_grasp_offsets[group_index],
                        "carry_orientation_action_limit": (
                            _NEEDLE_PROVISIONAL_ORIENTATION_ACTION_LIMIT
                        ),
                        "carry_orientation_velocity_damping_s": (
                            _NEEDLE_PROVISIONAL_ORIENTATION_VELOCITY_DAMPING_S
                        ),
                        args.parameter: value,
                    }
                else:
                    controller_kwargs = (
                        {}
                        if environment_level_parameter
                        else {args.parameter: value}
                    )
                group_obs = {
                    name: observation[start:stop]
                    for name, observation in obs.items()
                }
                actions[start:stop] = _lift_teacher_action(
                    group_obs,
                    position_scale=0.01,
                    **controller_kwargs,
                )

            obs, _, terminated, time_out_flags, _ = env.step(actions)
            dones = terminated | time_out_flags
            success_term = manager.get_term("success")
            failure_terms = {
                name: manager.get_term(name)
                for name in failure_term_counts
            }
            hard_failure = (
                failure_terms["object_dropping"]
                | failure_terms["excessive_object_force"]
                | failure_terms["protected_surface_force"]
            )
            time_out_term = manager.get_term("time_out")
            first_done = was_unresolved & dones

            forces = mdp_common.paired_contact_forces(
                env.unwrapped,
                "jaw_1_object_contact",
                "jaw_2_object_contact",
            )
            non_object_forces = mdp_common.maximum_non_object_contact_force(
                env.unwrapped,
                ("jaw_1_object_contact", "jaw_2_object_contact"),
            )
            goal_position_error, goal_orientation_error = (
                mdp_common.object_goal_errors(
                    env.unwrapped,
                    "object_pose",
                    SceneEntityCfg("robot"),
                )
            )
            motion = mdp_common.object_motion(env.unwrapped)
            object_height = mdp_common.as_torch(
                env.unwrapped.scene["object"].data.root_pos_w
            )[:, 2]
            for group_index in range(len(values)):
                start = group_index * group_size
                stop = start + group_size
                active = was_unresolved[start:stop]
                first = first_done[start:stop]
                completed[group_index] += first.sum()
                successes[group_index] += (
                    first & success_term[start:stop] & ~hard_failure[start:stop]
                ).sum()
                hard_failures[group_index] += (
                    first & hard_failure[start:stop]
                ).sum()
                for name, term in failure_terms.items():
                    failure_term_counts[name][group_index] += (
                        first & term[start:stop]
                    ).sum()
                timeouts[group_index] += (
                    first
                    & time_out_term[start:stop]
                    & ~success_term[start:stop]
                    & ~hard_failure[start:stop]
                ).sum()
                if active.any():
                    group_forces = forces[start:stop]
                    active_samples[group_index] += active.sum()
                    contact_frames[group_index] += (
                        torch.all(group_forces > 0.01, dim=-1) & active
                    ).sum()
                    angular_valid_frames[group_index] += (
                        (motion[start:stop, 1] < 1.5) & active
                    ).sum()
                    above_height_frames[group_index] += (
                        (object_height[start:stop] > 0.06) & active
                    ).sum()
                    goal_position_frames[group_index] += (
                        (goal_position_error[start:stop] < 0.015) & active
                    ).sum()
                    goal_orientation_frames[group_index] += (
                        (goal_orientation_error[start:stop] < 0.35) & active
                    ).sum()
                    maximum_force[group_index] = torch.maximum(
                        maximum_force[group_index],
                        group_forces[active].max().double(),
                    )
                    maximum_non_object_force[group_index] = torch.maximum(
                        maximum_non_object_force[group_index],
                        non_object_forces[start:stop][active].max().double(),
                    )
            unresolved &= ~first_done
            if not unresolved.any():
                break

        duration = time.perf_counter() - started
        results = []
        for index, value in enumerate(values):
            completed_count = int(completed[index].item())
            sample_count = float(active_samples[index].item())
            success_count = int(successes[index].item())
            results.append(
                {
                    "value": value,
                    "grasp_offset_m": (
                        list(needle_grasp_offsets[index])
                        if needle_grasp_offsets[index] is not None
                        else None
                    ),
                    "assigned_environments": group_size,
                    "completed_episodes": completed_count,
                    "successful_episodes": success_count,
                    "success_rate": (
                        success_count / completed_count if completed_count else None
                    ),
                    "time_outs": int(timeouts[index].item()),
                    "hard_failures": int(hard_failures[index].item()),
                    "hard_failure_term_counts": {
                        name: int(counts[index].item())
                        for name, counts in failure_term_counts.items()
                    },
                    "bilateral_contact_frame_rate": (
                        float(contact_frames[index].item()) / sample_count
                        if sample_count
                        else None
                    ),
                    "angular_speed_inside_frame_rate": (
                        float(angular_valid_frames[index].item()) / sample_count
                        if sample_count
                        else None
                    ),
                    "above_minimum_height_frame_rate": (
                        float(above_height_frames[index].item()) / sample_count
                        if sample_count
                        else None
                    ),
                    "goal_position_inside_frame_rate": (
                        float(goal_position_frames[index].item()) / sample_count
                        if sample_count
                        else None
                    ),
                    "goal_orientation_inside_frame_rate": (
                        float(goal_orientation_frames[index].item()) / sample_count
                        if sample_count
                        else None
                    ),
                    "maximum_object_force_n": float(maximum_force[index].item()),
                    "maximum_non_object_force_n": float(
                        maximum_non_object_force[index].item()
                    ),
                }
            )
        evidence = {
            "schema_version": "dranmar-controller-sweep-evidence-1.0",
            "kind": "controller_sweep",
            "task": args.task,
            "seed": args.seed,
            "num_envs": env.unwrapped.num_envs,
            "frames_per_env": args.num_frames,
            "parameter": args.parameter,
            "values": values,
            "grasp_frame_source": grasp_frame_source,
            "environment_level_parameter": environment_level_parameter,
            "environment_override": environment_override,
            "shared_reset_distribution": True,
            "first_terminal_outcome_per_environment": True,
            "results": results,
            "wall_time_s": duration,
            "total_fps": (
                env.unwrapped.num_envs * args.num_frames / duration
                if duration > 0
                else None
            ),
            "runtime": _runtime_evidence(repo_root),
        }
        _write_evidence(
            Path(args.output_path),
            "dranmar_controller_sweep",
            evidence,
        )
        return 0
    finally:
        env.close()


def _play(args: argparse.Namespace, repo_root: Path) -> int:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.managers import SceneEntityCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        return _fail(f"checkpoint not found: {checkpoint}")

    env_cfg, agent_cfg = _load_configs(args.task, args.num_envs, args.seed)
    env_kwargs: dict[str, Any] = {"cfg": env_cfg}
    if args.video:
        env_cfg.viewer.resolution = (args.video_width, args.video_height)
        env_kwargs["render_mode"] = "rgb_array"
    env = gym.make(args.task, **env_kwargs)
    if args.video:
        video_folder = Path(
            args.video_folder or Path(args.output_path).resolve() / "videos"
        ).resolve()
        video_folder.mkdir(parents=True, exist_ok=True)
        video_chunk_length = args.video_chunk_length or 0
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            step_trigger=(
                (lambda step: step % video_chunk_length == 0)
                if video_chunk_length
                else (lambda step: step == 0)
            ),
            video_length=(
                video_chunk_length
                if video_chunk_length
                else (args.video_length or args.num_frames)
            ),
            name_prefix=f"{args.task}-seed{args.seed}",
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_dir = Path(args.output_path).resolve() / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_jit(path=str(export_dir), filename="policy.pt")
    runner.export_policy_to_onnx(path=str(export_dir), filename="policy.onnx")

    rewards: list[float] = []
    done_count = 0
    success_count = 0
    termination_manager = env.unwrapped.termination_manager
    termination_names = list(termination_manager.active_terms)
    termination_counts = {name: 0 for name in termination_names}
    failure_names = [
        name for name in termination_names if name not in {"success", "time_out"}
    ]
    if "time_out" in termination_names:
        failure_names.append("time_out")
    failure_distribution = {name: 0 for name in failure_names}
    first_unresolved = torch.ones(
        env.unwrapped.num_envs,
        dtype=torch.bool,
        device=env.unwrapped.device,
    )
    first_outcome_success = torch.zeros_like(first_unresolved)
    first_termination_counts = {name: 0 for name in termination_names}
    first_failure_distribution = {name: 0 for name in failure_names}
    lift_diagnostics = None
    first_lift_history = None
    lift_mdp_common = None
    procedure_diagnostic_trace = None
    diagnostic_trace_frames = {
        0,
        1,
        2,
        5,
        10,
        20,
        40,
        80,
        120,
        149,
        150,
        300,
        args.num_frames - 1,
    }
    single_environment_episode_trace = (
        [] if args.video and args.num_envs == 1 else None
    )
    single_environment_episode_start_frame = 0
    if "Lift-" in args.task:
        from orbit.surgical.tasks.surgical import mdp_common as lift_mdp_common

        procedure_diagnostic_trace = []
        lift_diagnostics = {
            "samples": torch.zeros(1, device=env.unwrapped.device),
            "bilateral_contact": torch.zeros(1, device=env.unwrapped.device),
            "above_minimum_height": torch.zeros(1, device=env.unwrapped.device),
            "goal_position_inside": torch.zeros(1, device=env.unwrapped.device),
            "goal_orientation_inside": torch.zeros(
                1, device=env.unwrapped.device
            ),
            "linear_speed_inside": torch.zeros(1, device=env.unwrapped.device),
            "angular_speed_inside": torch.zeros(1, device=env.unwrapped.device),
            "instantaneous_success": torch.zeros(1, device=env.unwrapped.device),
            "goal_position_error_sum": torch.zeros(
                1, device=env.unwrapped.device
            ),
            "goal_orientation_error_sum": torch.zeros(
                1, device=env.unwrapped.device
            ),
            "object_height_sum": torch.zeros(1, device=env.unwrapped.device),
            "maximum_object_force_n": torch.zeros(1, device=env.unwrapped.device),
            "maximum_non_object_force_n": torch.zeros(
                1, device=env.unwrapped.device
            ),
            "maximum_object_height_m": torch.full(
                (1,),
                -torch.inf,
                device=env.unwrapped.device,
            ),
        }
    obs = env.get_observations()
    if lift_diagnostics is not None:
        policy_observation = obs["policy"]
        initial_object_xy = policy_observation[:, 23:25].clone()
        initial_target_xy = policy_observation[:, 36:38].clone()
        first_lift_history = {
            "ever_bilateral_contact": torch.zeros_like(first_unresolved),
            "ever_airborne_transport": torch.zeros_like(first_unresolved),
            "ever_midair_bilateral_contact_loss": torch.zeros_like(
                first_unresolved
            ),
            "current_midair_bilateral_contact_loss_steps": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "maximum_midair_bilateral_contact_loss_steps": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "ever_above_minimum_height": torch.zeros_like(first_unresolved),
            "ever_goal_position_inside": torch.zeros_like(first_unresolved),
            "ever_goal_orientation_inside": torch.zeros_like(first_unresolved),
            "ever_linear_speed_inside": torch.zeros_like(first_unresolved),
            "ever_angular_speed_inside": torch.zeros_like(first_unresolved),
            "ever_instantaneous_success": torch.zeros_like(first_unresolved),
            "maximum_object_height_m": torch.full(
                (env.unwrapped.num_envs,),
                -torch.inf,
                device=env.unwrapped.device,
            ),
            "minimum_goal_position_error_m": torch.full(
                (env.unwrapped.num_envs,),
                torch.inf,
                device=env.unwrapped.device,
            ),
            "initial_object_xy": initial_object_xy,
            "initial_target_xy": initial_target_xy,
        }
    started = time.perf_counter()
    try:
        for frame_index in range(args.num_frames):
            with torch.inference_mode():
                was_first_unresolved = first_unresolved.clone()
                if first_lift_history is not None:
                    assert lift_mdp_common is not None
                    first_forces = lift_mdp_common.paired_contact_forces(
                        env.unwrapped,
                        "jaw_1_object_contact",
                        "jaw_2_object_contact",
                    )
                    first_object_height = lift_mdp_common.as_torch(
                        env.unwrapped.scene["object"].data.root_pos_w
                    )[:, 2]
                    (
                        first_goal_position_error,
                        first_goal_orientation_error,
                    ) = lift_mdp_common.object_goal_errors(
                        env.unwrapped,
                        "object_pose",
                        SceneEntityCfg("robot"),
                        SceneEntityCfg("object"),
                    )
                    first_motion = lift_mdp_common.object_motion(env.unwrapped)
                    first_bilateral_contact = torch.all(
                        first_forces > 0.01,
                        dim=-1,
                    )
                    first_above_minimum_height = first_object_height > 0.06
                    first_goal_position_inside = (
                        first_goal_position_error < 0.015
                    )
                    first_goal_orientation_inside = (
                        first_goal_orientation_error < 0.35
                    )
                    first_linear_speed_inside = first_motion[:, 0] < 0.08
                    first_angular_speed_inside = first_motion[:, 1] < 1.5
                    first_instantaneous_success = (
                        first_bilateral_contact
                        & first_above_minimum_height
                        & first_goal_position_inside
                        & first_goal_orientation_inside
                        & first_linear_speed_inside
                        & first_angular_speed_inside
                    )
                    first_airborne_transport = first_object_height > 0.03
                    first_midair_bilateral_contact_loss = (
                        (
                            first_lift_history["ever_airborne_transport"]
                            | first_airborne_transport
                        )
                        & first_lift_history["ever_bilateral_contact"]
                        & ~first_bilateral_contact
                        & first_airborne_transport
                    )
                    current_midair_loss_steps = torch.where(
                        was_first_unresolved
                        & first_midair_bilateral_contact_loss,
                        first_lift_history[
                            "current_midair_bilateral_contact_loss_steps"
                        ]
                        + 1,
                        0,
                    )
                    first_lift_history[
                        "current_midair_bilateral_contact_loss_steps"
                    ] = current_midair_loss_steps
                    first_lift_history[
                        "maximum_midair_bilateral_contact_loss_steps"
                    ] = torch.maximum(
                        first_lift_history[
                            "maximum_midair_bilateral_contact_loss_steps"
                        ],
                        current_midair_loss_steps,
                    )
                    for key, value in (
                        ("ever_bilateral_contact", first_bilateral_contact),
                        (
                            "ever_airborne_transport",
                            first_airborne_transport,
                        ),
                        (
                            "ever_midair_bilateral_contact_loss",
                            first_midair_bilateral_contact_loss,
                        ),
                        (
                            "ever_above_minimum_height",
                            first_above_minimum_height,
                        ),
                        (
                            "ever_goal_position_inside",
                            first_goal_position_inside,
                        ),
                        (
                            "ever_goal_orientation_inside",
                            first_goal_orientation_inside,
                        ),
                        (
                            "ever_linear_speed_inside",
                            first_linear_speed_inside,
                        ),
                        (
                            "ever_angular_speed_inside",
                            first_angular_speed_inside,
                        ),
                        (
                            "ever_instantaneous_success",
                            first_instantaneous_success,
                        ),
                    ):
                        first_lift_history[key] |= was_first_unresolved & value
                    first_lift_history["maximum_object_height_m"] = torch.where(
                        was_first_unresolved,
                        torch.maximum(
                            first_lift_history["maximum_object_height_m"],
                            first_object_height,
                        ),
                        first_lift_history["maximum_object_height_m"],
                    )
                    first_lift_history["minimum_goal_position_error_m"] = (
                        torch.where(
                            was_first_unresolved,
                            torch.minimum(
                                first_lift_history[
                                    "minimum_goal_position_error_m"
                                ],
                                first_goal_position_error,
                            ),
                            first_lift_history[
                                "minimum_goal_position_error_m"
                            ],
                        )
                    )
                actions = policy(obs)
                obs, reward, dones, extras = env.step(actions)
                term_values = {
                    name: termination_manager.get_term(name)
                    for name in termination_names
                }
                hard_failure = torch.zeros_like(dones, dtype=torch.bool)
                for name in failure_names:
                    if name != "time_out":
                        hard_failure |= term_values[name]
                successes = term_values["success"] & ~hard_failure
                unassigned_failures = dones & ~successes
                for name in failure_names:
                    assigned = unassigned_failures & term_values[name]
                    failure_distribution[name] += int(assigned.sum().item())
                    unassigned_failures &= ~assigned
                if unassigned_failures.any().item():
                    failure_distribution.setdefault("unclassified", 0)
                    failure_distribution["unclassified"] += int(
                        unassigned_failures.sum().item()
                    )
                for name, value in term_values.items():
                    termination_counts[name] += int(value.sum().item())
                if (
                    single_environment_episode_trace is not None
                    and bool(dones.bool()[0].item())
                ):
                    if bool(successes.bool()[0].item()):
                        episode_outcome = "success"
                    else:
                        episode_outcome = next(
                            (
                                name
                                for name in failure_names
                                if bool(term_values[name].bool()[0].item())
                            ),
                            "unclassified",
                        )
                    single_environment_episode_trace.append(
                        {
                            "episode": len(
                                single_environment_episode_trace
                            ),
                            "start_frame_inclusive": (
                                single_environment_episode_start_frame
                            ),
                            "terminal_frame_inclusive": frame_index,
                            "frame_count": (
                                frame_index
                                - single_environment_episode_start_frame
                                + 1
                            ),
                            "outcome": episode_outcome,
                        }
                    )
                    single_environment_episode_start_frame = frame_index + 1
                first_dones = was_first_unresolved & dones.bool()
                first_successes = first_dones & successes.bool()
                first_outcome_success |= first_successes
                first_unassigned_failures = first_dones & ~first_successes
                for name in failure_names:
                    assigned = (
                        first_unassigned_failures
                        & term_values[name].bool()
                    )
                    first_failure_distribution[name] += int(
                        assigned.sum().item()
                    )
                    first_unassigned_failures &= ~assigned
                if first_unassigned_failures.any().item():
                    first_failure_distribution.setdefault("unclassified", 0)
                    first_failure_distribution["unclassified"] += int(
                        first_unassigned_failures.sum().item()
                    )
                for name, value in term_values.items():
                    first_termination_counts[name] += int(
                        (first_dones & value.bool()).sum().item()
                    )
                if first_lift_history is not None:
                    for key in (
                        "ever_bilateral_contact",
                        "ever_airborne_transport",
                        "ever_above_minimum_height",
                        "ever_goal_position_inside",
                        "ever_goal_orientation_inside",
                        "ever_linear_speed_inside",
                        "ever_angular_speed_inside",
                        "ever_instantaneous_success",
                    ):
                        first_lift_history[key] |= first_successes
                first_unresolved &= ~first_dones
                if lift_diagnostics is not None:
                    assert lift_mdp_common is not None
                    assert procedure_diagnostic_trace is not None
                    forces = lift_mdp_common.paired_contact_forces(
                        env.unwrapped,
                        "jaw_1_object_contact",
                        "jaw_2_object_contact",
                    )
                    non_object_forces = torch.stack(
                        (
                            lift_mdp_common.non_object_contact_force_magnitude(
                                env.unwrapped, "jaw_1_object_contact"
                            ),
                            lift_mdp_common.non_object_contact_force_magnitude(
                                env.unwrapped, "jaw_2_object_contact"
                            ),
                        ),
                        dim=-1,
                    )
                    object_height = lift_mdp_common.as_torch(
                        env.unwrapped.scene["object"].data.root_pos_w
                    )[:, 2]
                    goal_position_error, goal_orientation_error = (
                        lift_mdp_common.object_goal_errors(
                            env.unwrapped,
                            "object_pose",
                            SceneEntityCfg("robot"),
                            SceneEntityCfg("object"),
                        )
                    )
                    motion = lift_mdp_common.object_motion(env.unwrapped)
                    bilateral_contact = torch.all(forces > 0.01, dim=-1)
                    above_minimum_height = object_height > 0.06
                    goal_position_inside = goal_position_error < 0.015
                    goal_orientation_inside = goal_orientation_error < 0.35
                    linear_speed_inside = motion[:, 0] < 0.08
                    angular_speed_inside = motion[:, 1] < 1.5
                    instantaneous_success = (
                        bilateral_contact
                        & above_minimum_height
                        & goal_position_inside
                        & goal_orientation_inside
                        & linear_speed_inside
                        & angular_speed_inside
                    )
                    lift_diagnostics["samples"] += env.unwrapped.num_envs
                    lift_diagnostics["bilateral_contact"] += (
                        bilateral_contact.sum()
                    )
                    lift_diagnostics["above_minimum_height"] += (
                        above_minimum_height.sum()
                    )
                    lift_diagnostics["goal_position_inside"] += (
                        goal_position_inside.sum()
                    )
                    lift_diagnostics["goal_orientation_inside"] += (
                        goal_orientation_inside.sum()
                    )
                    lift_diagnostics["linear_speed_inside"] += (
                        linear_speed_inside.sum()
                    )
                    lift_diagnostics["angular_speed_inside"] += (
                        angular_speed_inside.sum()
                    )
                    lift_diagnostics["instantaneous_success"] += (
                        instantaneous_success.sum()
                    )
                    lift_diagnostics["goal_position_error_sum"] += (
                        goal_position_error.sum()
                    )
                    lift_diagnostics["goal_orientation_error_sum"] += (
                        goal_orientation_error.sum()
                    )
                    lift_diagnostics["object_height_sum"] += object_height.sum()
                    lift_diagnostics["maximum_object_force_n"] = torch.maximum(
                        lift_diagnostics["maximum_object_force_n"],
                        forces.max(),
                    )
                    lift_diagnostics["maximum_non_object_force_n"] = (
                        torch.maximum(
                            lift_diagnostics["maximum_non_object_force_n"],
                            non_object_forces.max(),
                        )
                    )
                    lift_diagnostics["maximum_object_height_m"] = torch.maximum(
                        lift_diagnostics["maximum_object_height_m"],
                        object_height.max(),
                    )
                    if frame_index in diagnostic_trace_frames:
                        ee_position = lift_mdp_common.as_torch(
                            env.unwrapped.scene["ee_frame"].data.target_pos_w
                        )[:, 0, :]
                        object_position = lift_mdp_common.as_torch(
                            env.unwrapped.scene["object"].data.root_pos_w
                        )
                        procedure_diagnostic_trace.append(
                            {
                                "frame": frame_index,
                                "bilateral_contact_fraction": float(
                                    bilateral_contact.float().mean().item()
                                ),
                                "instantaneous_success_fraction": float(
                                    instantaneous_success.float().mean().item()
                                ),
                                "mean_end_effector_object_distance_m": float(
                                    torch.linalg.vector_norm(
                                        object_position - ee_position,
                                        dim=-1,
                                    )
                                    .mean()
                                    .item()
                                ),
                                "mean_object_height_m": float(
                                    object_height.mean().item()
                                ),
                                "mean_goal_position_error_m": float(
                                    goal_position_error.mean().item()
                                ),
                                "mean_goal_orientation_error_rad": float(
                                    goal_orientation_error.mean().item()
                                ),
                                "mean_object_linear_speed_m_s": float(
                                    motion[:, 0].mean().item()
                                ),
                                "mean_object_angular_speed_rad_s": float(
                                    motion[:, 1].mean().item()
                                ),
                                "maximum_object_force_n": float(
                                    forces.max().item()
                                ),
                            }
                        )
                policy.reset(dones)
            rewards.append(float(reward.float().mean().item()))
            done_count += int(dones.sum().item())
            success_count += int(successes.sum().item())
        duration = time.perf_counter() - started
        jit_path = export_dir / "policy.pt"
        onnx_path = export_dir / "policy.onnx"
        procedure_diagnostics = None
        first_episode_lift_diagnostics = None
        if lift_diagnostics is not None:
            samples = float(lift_diagnostics["samples"].item())
            procedure_diagnostics = {
                "bilateral_contact_frame_rate": (
                    float(lift_diagnostics["bilateral_contact"].item()) / samples
                    if samples
                    else None
                ),
                "above_minimum_height_frame_rate": (
                    float(lift_diagnostics["above_minimum_height"].item())
                    / samples
                    if samples
                    else None
                ),
                "goal_position_inside_frame_rate": (
                    float(lift_diagnostics["goal_position_inside"].item())
                    / samples
                    if samples
                    else None
                ),
                "goal_orientation_inside_frame_rate": (
                    float(lift_diagnostics["goal_orientation_inside"].item())
                    / samples
                    if samples
                    else None
                ),
                "linear_speed_inside_frame_rate": (
                    float(lift_diagnostics["linear_speed_inside"].item())
                    / samples
                    if samples
                    else None
                ),
                "angular_speed_inside_frame_rate": (
                    float(lift_diagnostics["angular_speed_inside"].item())
                    / samples
                    if samples
                    else None
                ),
                "instantaneous_success_frame_rate": (
                    float(lift_diagnostics["instantaneous_success"].item())
                    / samples
                    if samples
                    else None
                ),
                "mean_goal_position_error_m": (
                    float(lift_diagnostics["goal_position_error_sum"].item())
                    / samples
                    if samples
                    else None
                ),
                "mean_goal_orientation_error_rad": (
                    float(lift_diagnostics["goal_orientation_error_sum"].item())
                    / samples
                    if samples
                    else None
                ),
                "mean_object_height_m": (
                    float(lift_diagnostics["object_height_sum"].item()) / samples
                    if samples
                    else None
                ),
                "maximum_object_force_n": float(
                    lift_diagnostics["maximum_object_force_n"].item()
                ),
                "maximum_non_object_force_n": float(
                    lift_diagnostics["maximum_non_object_force_n"].item()
                ),
                "maximum_object_height_m": float(
                    lift_diagnostics["maximum_object_height_m"].item()
                ),
            }
        if first_lift_history is not None:
            first_completed = ~first_unresolved
            first_failed = first_completed & ~first_outcome_success
            no_contact = (
                first_failed
                & ~first_lift_history["ever_bilateral_contact"]
            )
            contact_without_height = (
                first_failed
                & first_lift_history["ever_bilateral_contact"]
                & ~first_lift_history["ever_above_minimum_height"]
            )
            height_without_goal = (
                first_failed
                & first_lift_history["ever_above_minimum_height"]
                & ~first_lift_history["ever_goal_position_inside"]
            )
            goal_without_qualified_state = (
                first_failed
                & first_lift_history["ever_goal_position_inside"]
                & ~first_lift_history["ever_instantaneous_success"]
            )
            qualified_without_dwell = (
                first_failed
                & first_lift_history["ever_instantaneous_success"]
            )
            failed_after_midair_contact_loss = (
                first_failed
                & first_lift_history["ever_midair_bilateral_contact_loss"]
            )
            successful_after_midair_contact_loss = (
                first_outcome_success
                & first_lift_history["ever_midair_bilateral_contact_loss"]
            )

            def retention_cohort_stats(mask) -> dict[str, float | int | None]:
                count = int(mask.sum().item())
                maximum_loss_steps = first_lift_history[
                    "maximum_midair_bilateral_contact_loss_steps"
                ]
                if not count:
                    return {
                        "count": 0,
                        "mean_maximum_consecutive_loss_steps": None,
                        "median_maximum_consecutive_loss_steps": None,
                        "maximum_consecutive_loss_steps": None,
                        "at_least_2_steps": 0,
                        "at_least_5_steps": 0,
                        "at_least_10_steps": 0,
                    }
                cohort_steps = maximum_loss_steps[mask]
                return {
                    "count": count,
                    "mean_maximum_consecutive_loss_steps": float(
                        cohort_steps.float().mean().item()
                    ),
                    "median_maximum_consecutive_loss_steps": float(
                        cohort_steps.float().median().item()
                    ),
                    "maximum_consecutive_loss_steps": int(
                        cohort_steps.max().item()
                    ),
                    "at_least_2_steps": int(
                        (cohort_steps >= 2).sum().item()
                    ),
                    "at_least_5_steps": int(
                        (cohort_steps >= 5).sum().item()
                    ),
                    "at_least_10_steps": int(
                        (cohort_steps >= 10).sum().item()
                    ),
                }

            def cohort_stats(mask) -> dict[str, float | int | None]:
                count = int(mask.sum().item())
                if not count:
                    return {
                        "count": 0,
                        "mean_initial_target_xy_distance_m": None,
                        "mean_maximum_object_height_m": None,
                        "mean_minimum_goal_position_error_m": None,
                    }
                initial_target_xy_distance = torch.linalg.vector_norm(
                    first_lift_history["initial_target_xy"]
                    - first_lift_history["initial_object_xy"],
                    dim=-1,
                )
                return {
                    "count": count,
                    "mean_initial_target_xy_distance_m": float(
                        initial_target_xy_distance[mask].mean().item()
                    ),
                    "mean_maximum_object_height_m": float(
                        first_lift_history["maximum_object_height_m"][mask]
                        .mean()
                        .item()
                    ),
                    "mean_minimum_goal_position_error_m": float(
                        first_lift_history["minimum_goal_position_error_m"][
                            mask
                        ]
                        .mean()
                        .item()
                    ),
                }

            initial_target_xy_distance = torch.linalg.vector_norm(
                first_lift_history["initial_target_xy"]
                - first_lift_history["initial_object_xy"],
                dim=-1,
            )
            target_distance_bins = []
            for lower, upper in (
                (0.0, 0.02),
                (0.02, 0.04),
                (0.04, 0.06),
                (0.06, 0.08),
                (0.08, float("inf")),
            ):
                in_bin = (
                    first_completed
                    & (initial_target_xy_distance >= lower)
                    & (initial_target_xy_distance < upper)
                )
                bin_count = int(in_bin.sum().item())
                bin_successes = int(
                    (in_bin & first_outcome_success).sum().item()
                )
                target_distance_bins.append(
                    {
                        "lower_inclusive_m": lower,
                        "upper_exclusive_m": (
                            upper if upper != float("inf") else None
                        ),
                        "completed_episodes": bin_count,
                        "successful_episodes": bin_successes,
                        "success_rate": (
                            bin_successes / bin_count if bin_count else None
                        ),
                    }
                )
            first_episode_lift_diagnostics = {
                "stage_distribution": {
                    "success": int(first_outcome_success.sum().item()),
                    "no_bilateral_contact": int(no_contact.sum().item()),
                    "contact_without_minimum_height": int(
                        contact_without_height.sum().item()
                    ),
                    "minimum_height_without_goal_position": int(
                        height_without_goal.sum().item()
                    ),
                    "goal_position_without_qualified_state": int(
                        goal_without_qualified_state.sum().item()
                    ),
                    "qualified_state_without_sustained_dwell": int(
                        qualified_without_dwell.sum().item()
                    ),
                    "unresolved": int(first_unresolved.sum().item()),
                },
                "reached_fraction": {
                    key.removeprefix("ever_"): float(
                        value.float().mean().item()
                    )
                    for key, value in first_lift_history.items()
                    if key.startswith("ever_")
                },
                "outcome_cohorts": {
                    "successful": cohort_stats(first_outcome_success),
                    "failed": cohort_stats(first_failed),
                },
                "retention_diagnostics": {
                    "airborne_height_threshold_m": 0.03,
                    "failed_after_midair_bilateral_contact_loss": int(
                        failed_after_midair_contact_loss.sum().item()
                    ),
                    "successful_after_midair_bilateral_contact_loss": int(
                        successful_after_midair_contact_loss.sum().item()
                    ),
                    "failed_episode_loss_duration": retention_cohort_stats(
                        first_failed
                    ),
                    "successful_episode_loss_duration": (
                        retention_cohort_stats(first_outcome_success)
                    ),
                },
                "success_by_initial_target_xy_distance": target_distance_bins,
            }
        first_completed_count = int((~first_unresolved).sum().item())
        first_success_count = int(first_outcome_success.sum().item())
        evidence = {
            "schema_version": "dranmar-learning-evidence-1.0",
            "kind": "held_out_play",
            "task": args.task,
            "seed": args.seed,
            "requested_num_envs": args.requested_num_envs,
            "num_envs": env.unwrapped.num_envs,
            "trusted_requested_num_envs": args.trusted_requested_num_envs,
            "free_gpu_memory_before_launch_mib": args.free_gpu_memory_before_launch_mib,
            "system_memory_total_mib": args.system_memory_total_mib,
            "system_memory_available_before_launch_mib": (
                args.system_memory_available_before_launch_mib
            ),
            "frames_per_env": args.num_frames,
            "wall_time_s": duration,
            "total_fps": (
                env.unwrapped.num_envs * args.num_frames / duration
                if duration > 0
                else None
            ),
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "first_terminal_outcome_per_environment": True,
            "completed_episodes": first_completed_count,
            "successful_episodes": first_success_count,
            "failed_episodes": first_completed_count - first_success_count,
            "unresolved_episodes": int(first_unresolved.sum().item()),
            "failure_distribution": first_failure_distribution,
            "termination_term_counts": first_termination_counts,
            "all_episode_totals": {
                "completed_episodes": done_count,
                "successful_episodes": success_count,
                "failed_episodes": done_count - success_count,
                "failure_distribution": failure_distribution,
                "termination_term_counts": termination_counts,
            },
            "single_environment_episode_trace": (
                single_environment_episode_trace
            ),
            "video_capture": (
                {
                    "resolution": [args.video_width, args.video_height],
                    "chunk_length_frames": args.video_chunk_length,
                    "folder": str(
                        Path(
                            args.video_folder
                            or Path(args.output_path).resolve() / "videos"
                        ).resolve()
                    ),
                }
                if args.video
                else None
            ),
            "procedure_diagnostics": procedure_diagnostics,
            "procedure_diagnostic_trace": procedure_diagnostic_trace,
            "first_episode_lift_diagnostics": (
                first_episode_lift_diagnostics
            ),
            "process_peak_memory_mib": _peak_process_memory_mib(),
            "success_rate": (
                first_success_count / first_completed_count
                if first_completed_count
                else None
            ),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
            },
            "exports": {
                "jit": {"path": str(jit_path), "sha256": _sha256(jit_path)},
                "onnx": {"path": str(onnx_path), "sha256": _sha256(onnx_path)},
            },
            "runtime": _runtime_evidence(repo_root),
        }
        _write_evidence(Path(args.output_path), "dranmar_play", evidence)
        return 0
    finally:
        env.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DrAnmar Learning Path runtime")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("list")

    probe = subparsers.add_parser("probe")
    probe.add_argument("--task", required=True)
    probe.add_argument("--num_envs", type=int, required=True)
    probe.add_argument("--num_frames", type=int, default=10)
    probe.add_argument("--seed", type=int, default=17)
    probe.add_argument("--output_path", required=True)
    probe.add_argument("--benchmark_formatter", default="schema,json")

    controller_sweep = subparsers.add_parser("controller-sweep")
    controller_sweep.add_argument("--task", required=True)
    controller_sweep.add_argument("--num_envs", type=int, required=True)
    controller_sweep.add_argument("--num_frames", type=int, default=500)
    controller_sweep.add_argument("--parameter", required=True)
    controller_sweep.add_argument("--values", required=True)
    controller_sweep.add_argument("--seed", type=int, default=17)
    controller_sweep.add_argument("--output_path", required=True)
    controller_sweep.add_argument("--benchmark_formatter", default="schema,json")

    handover_sweep = subparsers.add_parser("handover-sweep")
    handover_sweep.add_argument("--task", required=True)
    handover_sweep.add_argument("--num_envs", type=int, required=True)
    handover_sweep.add_argument("--num_frames", type=int, default=1000)
    handover_sweep.add_argument(
        "--parameter",
        default="receiver_arc_fraction",
    )
    handover_sweep.add_argument("--values", required=True)
    handover_sweep.add_argument("--seed", type=int, default=17)
    handover_sweep.add_argument("--output_path", required=True)
    handover_sweep.add_argument("--benchmark_formatter", default="schema,json")

    train = subparsers.add_parser("train")
    train.add_argument("--task", required=True)
    train.add_argument("--num_envs", type=int, required=True)
    train.add_argument("--max_iterations", type=int, required=True)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--output_path", required=True)
    train.add_argument("--benchmark_formatter", default="schema,json")
    train.add_argument("--check_success", action="store_true")
    train.add_argument("--success_threshold", type=float, default=0.95)
    train.add_argument("--success_window", type=int, default=10)

    pretrain = subparsers.add_parser("pretrain")
    pretrain.add_argument("--task", required=True)
    pretrain.add_argument("--num_envs", type=int, required=True)
    pretrain.add_argument("--updates", type=int, default=32)
    pretrain.add_argument("--validation_frames", type=int, default=500)
    pretrain.add_argument("--learning_rate", type=float, default=3e-4)
    pretrain.add_argument("--weight_decay", type=float, default=1e-6)
    pretrain.add_argument("--position_scale", type=float, default=0.01)
    pretrain.add_argument("--orientation_scale", type=float, default=0.05)
    pretrain.add_argument("--seed", type=int, default=17)
    pretrain.add_argument("--output_path", required=True)
    pretrain.add_argument("--benchmark_formatter", default="schema,json")

    play = subparsers.add_parser("play")
    play.add_argument("--task", required=True)
    play.add_argument("--checkpoint", required=True)
    play.add_argument("--num_envs", type=int, required=True)
    play.add_argument("--num_frames", type=int, required=True)
    play.add_argument("--seed", type=int, default=2361)
    play.add_argument("--output_path", required=True)
    play.add_argument("--video", action="store_true")
    play.add_argument("--video_length", type=int)
    play.add_argument("--video_chunk_length", type=int)
    play.add_argument("--video_width", type=int, default=1280)
    play.add_argument("--video_height", type=int, default=720)
    play.add_argument("--video_folder")
    play.add_argument("--benchmark_formatter", default="schema,json")
    return parser


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    isaaclab_value = os.environ.get("DR_ANMAR_ISAACLAB_ROOT")
    if not isaaclab_value:
        return _fail("DR_ANMAR_ISAACLAB_ROOT must point to the active Isaac Lab checkout")
    isaaclab_root = Path(isaaclab_value).expanduser().resolve()
    if not (isaaclab_root / "source/isaaclab").is_dir():
        return _fail(f"Isaac Lab source not found under {isaaclab_root}")
    _prepare_imports(repo_root, isaaclab_root)

    args = _parser().parse_args(argv)
    if args.mode == "list":
        import orbit.surgical.tasks as dranmar_tasks

        for task_id in dranmar_tasks.DRANMAR_LEARNING_TASK_IDS:
            print(task_id)
        return 0
    if not args.task.startswith("DrAnmar-"):
        return _fail("--task must name a registered DrAnmar learning task")

    minimum_free_mib = int(os.environ.get("DR_ANMAR_MIN_FREE_GPU_MIB", "1024"))
    free_mib = _free_gpu_memory_mib()
    if free_mib is not None and free_mib < minimum_free_mib:
        return _fail(
            f"GPU preflight has {free_mib} MiB free; "
            f"DrAnmar requires {minimum_free_mib} MiB before simulator launch"
        )
    system_total_mib, system_available_mib = _system_memory_mib()
    minimum_system_mib = int(
        os.environ.get("DR_ANMAR_MIN_AVAILABLE_SYSTEM_MIB", "4096")
    )
    if system_available_mib is not None and system_available_mib < minimum_system_mib:
        return _fail(
            f"system memory preflight has {system_available_mib} MiB available; "
            f"DrAnmar requires {minimum_system_mib} MiB before simulator launch"
        )
    args.requested_num_envs = args.num_envs
    args.free_gpu_memory_before_launch_mib = free_mib
    args.system_memory_total_mib = system_total_mib
    args.system_memory_available_before_launch_mib = system_available_mib
    args.trusted_requested_num_envs = (
        os.environ.get("DR_ANMAR_TRUST_REQUESTED_NUM_ENVS", "0") == "1"
    )
    if args.trusted_requested_num_envs:
        print(
            "[DrAnmar] Qualified environment-count override: "
            f"using all {args.num_envs} requested environments"
        )
    else:
        args.num_envs = _fit_num_envs_to_memory(
            args.num_envs,
            free_mib,
            system_available_mib,
        )

    # Reuse one process-owned CUDA context across Torch, Warp, and PhysX. This
    # avoids a second large primary context when other GPU services are active.
    import torch

    torch.cuda.set_device(0)
    cuda_context_guard = torch.zeros(1, device="cuda:0")
    from isaaclab.app import AppLauncher

    app = AppLauncher(
        headless=True,
        enable_cameras=bool(getattr(args, "video", False)),
        multi_gpu=False,
        anti_aliasing=0,
        denoiser=False,
        kit_args="--/persistent/physics/useActiveCudaContext=true",
    ).app
    try:
        if not app.is_running():
            return _fail("Isaac Sim did not remain running after launch")
        # Kit startup shares the active Torch context to minimize its footprint.
        # Scene creation then gives PhysX its own thread-safe context because its
        # cooking tasks are not guaranteed to execute on Torch's calling thread.
        import carb

        carb.settings.get_settings().set_bool(
            "/persistent/physics/useActiveCudaContext", False
        )
        import orbit.surgical.tasks  # noqa: F401

        if args.mode == "probe":
            result = _probe(args, repo_root)
        elif args.mode == "controller-sweep":
            result = _controller_sweep(args, repo_root)
        elif args.mode == "handover-sweep":
            result = _handover_controller_sweep(args, repo_root)
        elif args.mode == "train":
            result = _train(args, repo_root)
        elif args.mode == "pretrain":
            result = _pretrain(args, repo_root)
        else:
            result = _play(args, repo_root)
        del cuda_context_guard
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    else:
        app.close()
        return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
