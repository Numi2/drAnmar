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


def _reach_teacher_action(obs, *, position_scale: float, orientation_scale: float):
    """Map the explicit pose-error observation to the relative-IK action."""
    import torch

    policy_obs = obs["policy"]
    position_error = policy_obs[:, 23:26]
    orientation_error = policy_obs[:, 26:29]
    return torch.cat(
        (
            position_error / position_scale,
            orientation_error / orientation_scale,
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)


def _pretrain(args: argparse.Namespace, repo_root: Path) -> int:
    """Initialize the reach actor from the controller encoded by its action space."""
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
            teacher_actions = _reach_teacher_action(
                obs,
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

            with torch.inference_mode():
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
        with torch.inference_mode():
            for _ in range(args.validation_frames):
                actions = policy(obs)
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
        evidence = {
            "schema_version": "dranmar-learning-evidence-1.0",
            "kind": "training",
            "algorithm": "analytic_relative_ik_teacher_behavior_cloning",
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


def _play(args: argparse.Namespace, repo_root: Path) -> int:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

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
    obs = env.get_observations()
    started = time.perf_counter()
    try:
        for _ in range(args.num_frames):
            with torch.inference_mode():
                actions = policy(obs)
                obs, reward, dones, extras = env.step(actions)
                successes = env.unwrapped.termination_manager.get_term("success")
                policy.reset(dones)
            rewards.append(float(reward.float().mean().item()))
            done_count += int(dones.sum().item())
            success_count += int(successes.sum().item())
        duration = time.perf_counter() - started
        jit_path = export_dir / "policy.pt"
        onnx_path = export_dir / "policy.onnx"
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
    pretrain.add_argument("--updates", type=int, default=400)
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

        if args.mode == "train":
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
