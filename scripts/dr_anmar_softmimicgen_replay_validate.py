#!/usr/bin/env python3
"""Replay NVIDIA's SoftMimicGen threading episode and validate the physical state.

NVIDIA's bundled ``replay_demos.py --validate_states`` currently compares a
saved state with its leading environment dimension intact and exits on the
first robot root pose.  It also omits deformable objects.  This validator keeps
the released task, initial state and actions unchanged, fixes the shape
comparison, and includes every FEM strand node in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Thread-PSM-IK-Rel-v0")
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--episode", default="demo_0")
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch

import softmimicgen_tasks  # noqa: F401  Registers the pinned task.
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from softmimicgen_tasks.surgical_threading.mdp import object_reached_goal


def array(value: Any) -> np.ndarray:
    """Return a detached CPU array for Isaac Lab tensors and NumPy values."""

    value = getattr(value, "torch", value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def align_reference(reference: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Remove only the saved singleton environment dimension when necessary."""

    if reference.shape == live.shape:
        return reference
    if reference.ndim == live.ndim + 1 and reference.shape[0] == 1 and reference.shape[1:] == live.shape:
        return reference[0]
    raise ValueError(f"state shape mismatch: dataset={reference.shape}, live={live.shape}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_success(episode: h5py.Group, index: int) -> bool:
    """Evaluate NVIDIA's published threading predicate on a recorded state."""

    nodes = torch.from_numpy(episode["states/deformable_object/object/nodal_position"][index])
    ring_pose = torch.from_numpy(episode["states/rigid_object/ring/root_pose"][index])
    ee_position = torch.from_numpy(episode["obs/robot0_eef_pos"][index])
    if nodes.ndim == 2:
        nodes = nodes.unsqueeze(0)
    if ring_pose.ndim == 1:
        ring_pose = ring_pose.unsqueeze(0)
    if ee_position.ndim == 1:
        ee_position = ee_position.unsqueeze(0)
    nodes_b, _ = subtract_frame_transforms(ring_pose[:, :3], ring_pose[:, 3:7], nodes)
    ee_b, _ = subtract_frame_transforms(ring_pose[:, :3], ring_pose[:, 3:7], ee_position)
    passed = (
        (nodes_b[..., 0] >= 0.005)
        & (torch.abs(nodes_b[..., 1]) <= 0.01)
        & (torch.abs(nodes_b[..., 2]) <= 0.01)
        & (ee_b[:, None, 0] < 0.0)
    )
    return bool(torch.any(passed).item())


def main() -> int:
    dataset_path = args_cli.dataset.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # Match NVIDIA's official replay behavior: the replay itself must not be
    # auto-reset by success, drop, or timeout terms.
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 3600.0

    # Cameras do not participate in physics-state validation.  Removing them
    # avoids shader work and leaves enough 4090 memory for the live Dr.Anmar
    # and Jetbot workloads that share Gilgamesh.
    for camera_name in ("agentview_image", "robot0_eye_in_hand_image"):
        if hasattr(env_cfg.scene, camera_name):
            setattr(env_cfg.scene, camera_name, None)
        if hasattr(env_cfg.observations.policy, camera_name):
            setattr(env_cfg.observations.policy, camera_name, None)

    # The upstream config reserves buffers for thousands of training rooms.
    # These capacities are only reservations; reducing them for one room does
    # not alter the task assets, material parameters, integration step, or
    # contact/deformable solver behavior.
    one_room_physx = {
        "gpu_max_rigid_contact_count": 2**16,
        "gpu_max_rigid_patch_count": 2**14,
        "gpu_found_lost_pairs_capacity": 2**16,
        "gpu_found_lost_aggregate_pairs_capacity": 2**16,
        "gpu_total_aggregate_pairs_capacity": 2**16,
        "gpu_collision_stack_size": 2**24,
        "gpu_heap_capacity": 2**24,
        "gpu_temp_buffer_capacity": 2**22,
        "gpu_max_soft_body_contacts": 2**16,
        "gpu_max_particle_contacts": 2**14,
    }
    for setting, value in one_room_physx.items():
        if hasattr(env_cfg.sim.physx, setting):
            setattr(env_cfg.sim.physx, setting, value)

    handler = HDF5DatasetFileHandler()
    handler.open(str(dataset_path))
    if args_cli.episode not in set(handler.get_episode_names()):
        raise KeyError(f"episode {args_cli.episode!r} is not present")
    episode_data = handler.load_episode(args_cli.episode, args_cli.device)
    actions = array(episode_data.data["actions"]).astype(np.float32)

    with h5py.File(dataset_path, "r") as dataset:
        episode = dataset[f"data/{args_cli.episode}"]
        reference_states = episode["states"]
        reference_terminal_success = reference_success(episode, -1)

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        metrics: dict[str, dict[str, Any]] = {}
        success_steps: list[int] = []
        shape_errors: list[str] = []
        non_finite_values = 0

        with torch.inference_mode():
            env.reset_to(episode_data.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
            for step_index, action_value in enumerate(actions):
                action_tensor = torch.from_numpy(action_value).to(device=env.device).reshape(1, -1)
                env.step(action_tensor)
                live_state = env.scene.get_state(is_relative=True)
                if bool(object_reached_goal(env)[0].detach().cpu().item()):
                    success_steps.append(step_index)

                for category_name, category_group in reference_states.items():
                    live_category = live_state.get(category_name, {})
                    for asset_name, asset_group in category_group.items():
                        live_asset = live_category.get(asset_name, {})
                        for state_name, state_dataset in asset_group.items():
                            metric_name = f"{category_name}/{asset_name}/{state_name}"
                            if state_name not in live_asset:
                                shape_errors.append(f"missing live state: {metric_name}")
                                continue
                            reference_value = np.asarray(state_dataset[step_index])
                            live_value = array(live_asset[state_name][0])
                            try:
                                reference_value = align_reference(reference_value, live_value)
                            except ValueError as exc:
                                shape_errors.append(f"{metric_name}: {exc}")
                                continue
                            finite = np.isfinite(reference_value) & np.isfinite(live_value)
                            non_finite_values += int(finite.size - np.count_nonzero(finite))
                            if not np.all(finite):
                                continue
                            delta = live_value.astype(np.float64) - reference_value.astype(np.float64)
                            max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
                            rmse = float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0
                            metric = metrics.setdefault(
                                metric_name,
                                {
                                    "shape": list(live_value.shape),
                                    "worst_max_abs": 0.0,
                                    "worst_rmse": 0.0,
                                    "worst_step": 0,
                                    "final_max_abs": 0.0,
                                    "final_rmse": 0.0,
                                },
                            )
                            if max_abs > metric["worst_max_abs"]:
                                metric["worst_max_abs"] = max_abs
                                metric["worst_step"] = step_index
                            metric["worst_rmse"] = max(metric["worst_rmse"], rmse)
                            metric["final_max_abs"] = max_abs
                            metric["final_rmse"] = rmse

        live_terminal_success = bool(object_reached_goal(env)[0].detach().cpu().item())
        robot_error = metrics.get("articulation/robot/joint_position", {}).get("final_max_abs")
        ring_error = metrics.get("rigid_object/ring/root_pose", {}).get("final_max_abs")
        strand_metric = metrics.get("deformable_object/object/nodal_position", {})
        tolerances = {
            # NVIDIA's bundled rigid-state validator uses 0.01.  The strand
            # bound is tighter and tied to the task's published 0.005 m
            # crossing threshold so a replay cannot pass on predicate alone
            # while materially diverging from the released deformation.
            "robot_joint_position_final_max_abs": 0.01,
            "ring_root_pose_final_max_abs": 0.01,
            "strand_nodal_position_worst_max_abs_m": 0.005,
        }
        position_replay_within_tolerance = bool(
            robot_error is not None
            and ring_error is not None
            and strand_metric.get("worst_max_abs") is not None
            and robot_error <= tolerances["robot_joint_position_final_max_abs"]
            and ring_error <= tolerances["ring_root_pose_final_max_abs"]
            and strand_metric["worst_max_abs"] <= tolerances["strand_nodal_position_worst_max_abs_m"]
        )
        report = {
            "schema": "dr.anmar.softmimicgen-replay-validation.v1",
            "task": args_cli.task,
            "episode": args_cli.episode,
            "action_count": int(len(actions)),
            "action_shape": list(actions.shape),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256(dataset_path),
            "reference_terminal_success": reference_terminal_success,
            "live_terminal_success": live_terminal_success,
            "live_success_steps": success_steps,
            "first_live_success_step": success_steps[0] if success_steps else None,
            "non_finite_values": non_finite_values,
            "shape_errors": sorted(set(shape_errors)),
            "tolerances": tolerances,
            "position_replay_within_tolerance": position_replay_within_tolerance,
            "summary": {
                "robot_joint_position_final_max_abs": robot_error,
                "ring_root_pose_final_max_abs": ring_error,
                "strand_nodal_position_final_rmse_m": strand_metric.get("final_rmse"),
                "strand_nodal_position_final_max_abs_m": strand_metric.get("final_max_abs"),
                "strand_nodal_position_worst_rmse_m": strand_metric.get("worst_rmse"),
                "strand_nodal_position_worst_max_abs_m": strand_metric.get("worst_max_abs"),
            },
            "metrics": metrics,
            "pass": bool(
                reference_terminal_success
                and live_terminal_success
                and position_replay_within_tolerance
                and not shape_errors
                and non_finite_values == 0
            ),
            "clinical_validation": False,
            "note": "Physical replay validation of NVIDIA's released research task; not clinical validation.",
        }
        args_cli.report.parent.mkdir(parents=True, exist_ok=True)
        args_cli.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("pass", "reference_terminal_success", "live_terminal_success", "first_live_success_step", "summary")}, separators=(",", ":")))
        env.close()
        handler.close()
        return 0 if report["pass"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
