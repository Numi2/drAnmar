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
    lateral_alignment_threshold: float = 0.004,
    close_distance: float = 0.003,
    slow_approach_radius: float = 0.02,
    slow_approach_action_limit: float = 0.1,
    normalized_contact_threshold: float = 0.002,
    lateral_clearance_below_target: float = 0.02,
    carry_latch_below_target: float = 0.062,
    carry_action_limit: float = 0.1,
):
    """Contact-conditioned analytic approach, grasp, and lift action."""
    import torch

    policy_obs = obs["policy"]
    ee_position = policy_obs[:, 16:19]
    object_position = policy_obs[:, 23:26]
    target_position = policy_obs[:, 36:39]
    contact_forces = policy_obs[:, 43:45]

    ee_to_object = object_position - ee_position
    lateral_distance = torch.linalg.vector_norm(ee_to_object[:, :2], dim=-1)
    above_object = object_position.clone()
    above_object[:, 2] += approach_height
    grasp_position = object_position.clone()
    grasp_position[:, 2] += grasp_height
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
    carry_target[:, :2] = torch.where(
        vertical_only.unsqueeze(-1),
        object_position[:, :2],
        target_position[:, :2],
    )
    carry_action = (
        (carry_target - object_position) / position_scale
    ).clamp(-carry_action_limit, carry_action_limit)
    translation_action = torch.where(
        carry_mode.unsqueeze(-1),
        carry_action,
        approach_action,
    )
    orientation_action = torch.zeros_like(translation_action)
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
    return _reach_teacher_action(
        obs,
        task,
        position_scale=position_scale,
        orientation_scale=orientation_scale,
    )


def _pretraining_algorithm(task: str) -> str:
    if "Lift-Block-PSM-IK-Rel" in task:
        return "analytic_grasp_lift_base_plus_learned_residual"
    return "analytic_relative_ik_base_plus_learned_residual"


def _pretrain(args: argparse.Namespace, repo_root: Path) -> int:
    """Initialize and validate a task-declared analytic-base residual actor."""
    import gymnasium as gym
    import torch
    import torch.nn.functional as functional
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
                {
                    "approach_height_m": 0.02,
                    "grasp_height_m": 0.0,
                    "lateral_alignment_threshold_m": 0.004,
                    "close_distance_to_grasp_m": 0.003,
                    "slow_approach_radius_m": 0.02,
                    "slow_approach_action_limit": 0.1,
                    "normalized_contact_threshold": 0.002,
                    "lateral_clearance_below_target_m": 0.02,
                    "carry_latch_below_target_m": 0.062,
                    "carry_action_limit": 0.1,
                }
                if "Lift-Block-PSM-IK-Rel" in args.task
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
    env = gym.make(args.task, cfg=env_cfg)
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
    lift_diagnostics = None
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
    started = time.perf_counter()
    try:
        for frame_index in range(args.num_frames):
            with torch.inference_mode():
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
            "completed_episodes": done_count,
            "successful_episodes": success_count,
            "failed_episodes": done_count - success_count,
            "failure_distribution": failure_distribution,
            "termination_term_counts": termination_counts,
            "procedure_diagnostics": procedure_diagnostics,
            "procedure_diagnostic_trace": procedure_diagnostic_trace,
            "process_peak_memory_mib": _peak_process_memory_mib(),
            "success_rate": success_count / done_count if done_count else None,
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
        enable_cameras=False,
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
