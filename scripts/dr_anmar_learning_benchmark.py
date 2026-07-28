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


def _policy_runtime_contract(policy_model, task: str) -> dict[str, Any]:
    """Return the deterministic serving configuration paired with a checkpoint."""
    controller = getattr(policy_model, "controller", None)
    controller_scalars = {}
    if controller is not None:
        controller_scalars = {
            key: value
            for key, value in vars(controller).items()
            if isinstance(value, (bool, int, float, str)) or value is None
        }
    return {
        "task": task,
        "model_class": type(policy_model).__name__,
        "residual_scale": getattr(policy_model, "residual_scale", None),
        "giver_adaptation_enabled": bool(
            getattr(policy_model, "giver_adaptation_enabled", False)
        ),
        "pickup_recovery_adaptation_enabled": bool(
            getattr(
                policy_model,
                "pickup_recovery_adaptation_enabled",
                False,
            )
        ),
        "recovery_receiver_grasp_retain_adaptation_enabled": bool(
            getattr(
                policy_model,
                "recovery_receiver_grasp_retain_adaptation_enabled",
                False,
            )
        ),
        "joint_transfer_acquisition_adaptation_enabled": bool(
            getattr(
                policy_model,
                "joint_transfer_acquisition_adaptation_enabled",
                False,
            )
        ),
        "transfer_refinement_adaptation_enabled": bool(
            getattr(
                policy_model,
                "transfer_refinement_adaptation_enabled",
                False,
            )
        ),
        "deadline_recovery_adaptation_enabled": bool(
            getattr(
                policy_model,
                "deadline_recovery_adaptation_enabled",
                False,
            )
        ),
        "controller_scalars": controller_scalars,
    }


def _environment_runtime_contract(env_cfg, task: str) -> dict[str, Any]:
    """Return the task-side serving contract that can change policy behavior."""
    giver_identity_term = getattr(
        getattr(env_cfg.observations, "policy", None),
        "giver_identity",
        None,
    )
    giver_identity_function = getattr(giver_identity_term, "func", None)
    return {
        "task": task,
        "environment_config_class": type(env_cfg).__name__,
        "handover_contract": getattr(
            env_cfg,
            "dr_anmar_handover_contract",
            None,
        ),
        "episode_length_s": getattr(env_cfg, "episode_length_s", None),
        "decimation": getattr(env_cfg, "decimation", None),
        "giver_identity_observation_function": (
            f"{giver_identity_function.__module__}:"
            f"{giver_identity_function.__name__}"
            if giver_identity_function is not None
            else None
        ),
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _initial_state_population_sha256(env) -> str:
    """Hash the exact reset population used for paired qualification."""
    import torch

    unwrapped = env.unwrapped
    origins = unwrapped.scene.env_origins
    obj = unwrapped.scene["object"]
    object_pose = torch.cat(
        (
            obj.data.root_pos_w - origins,
            obj.data.root_quat_w,
        ),
        dim=-1,
    )
    named_tensors = (
        ("robot_1_joint_pos", unwrapped.scene["robot_1"].data.joint_pos),
        ("robot_1_joint_vel", unwrapped.scene["robot_1"].data.joint_vel),
        ("robot_2_joint_pos", unwrapped.scene["robot_2"].data.joint_pos),
        ("robot_2_joint_vel", unwrapped.scene["robot_2"].data.joint_vel),
        ("object_root_pose_env", object_pose),
        (
            "object_root_velocity",
            torch.cat(
                (
                    obj.data.root_lin_vel_w,
                    obj.data.root_ang_vel_w,
                ),
                dim=-1,
            ),
        ),
    )
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = (
            torch.as_tensor(tensor)
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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
    worktree_status = _command_output(
        ["git", "status", "--porcelain=v1"],
        repo_root,
    )
    tracked_patch = _command_output(
        ["git", "diff", "--binary", "HEAD"],
        repo_root,
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
            "working_tree_dirty": bool(worktree_status),
            "working_tree_status": (
                worktree_status.splitlines() if worktree_status else []
            ),
            "tracked_patch_sha256": hashlib.sha256(
                (tracked_patch or "").encode("utf-8")
            ).hexdigest(),
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
    if "Handover-Needle-Dual-PSM-IK-Rel" in task:
        import math

        from orbit.surgical.tasks.surgical.lift.grasp_frames import (
            NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
            needle_geometry_grasp_offset_m,
        )

        receiver_offset = needle_geometry_grasp_offset_m(0.65)
        return _handover_teacher_action(
            obs,
            giver_grasp_offset=NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
            receiver_grasp_offset=(
                receiver_offset[0],
                receiver_offset[1],
                -0.003,
            ),
            receiver_roll_offset_rad=math.pi,
            position_scale=position_scale,
            orientation_scale=orientation_scale,
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
    giver_grasp_offset: tuple[float, float, float],
    receiver_grasp_offset: tuple[float, float, float],
    receiver_roll_offset_rad: float = 0.0,
    position_scale: float = 0.01,
    orientation_scale: float = 0.05,
    receiver_orientation_action_limit: float = 0.6,
    approach_height: float = 0.02,
    lateral_alignment_threshold: float = 0.005,
    close_distance: float = 0.005,
    receiver_close_distance: float = 0.001,
    slow_approach_radius: float = 0.02,
    slow_approach_action_limit: float = 0.1,
    receiver_contact_centering_action_limit: float = 0.0025,
    normalized_contact_threshold: float = 0.002,
    presentation_fraction_from_giver: float = 0.35,
    presentation_height_in_robot_frame: float = -0.13,
    presentation_ready_tolerance: float = 0.005,
    presentation_hold_action_limit: float = 0.01,
    minimum_lift_height_in_robot_frame: float = -0.139,
    carry_lateral_action_limit: float = 0.06,
    pickup_vertical_action_limit: float = 0.015,
    recovery_pickup_vertical_action_limit: float = 0.01,
    carry_vertical_action_limit: float = 0.015,
    giver_transport_min_contact_jaws: int = 2,
    giver_transport_normalized_contact_threshold: float = 0.002,
    giver_contact_recovery_action_limit: float = 1.0,
):
    """Stage a closest-arm pickup and other-arm physical custody transfer."""
    import math

    import torch

    policy_obs = obs["policy"]
    giver_is_robot_1 = policy_obs[:, 82] > 0.5

    def select_role(robot_1_value, robot_2_value, use_robot_1):
        return torch.where(
            use_robot_1.unsqueeze(-1),
            robot_1_value,
            robot_2_value,
        )

    giver_ee = select_role(
        policy_obs[:, 32:35],
        policy_obs[:, 39:42],
        giver_is_robot_1,
    )
    giver_orientation = select_role(
        policy_obs[:, 35:39],
        policy_obs[:, 42:46],
        giver_is_robot_1,
    )
    receiver_ee = select_role(
        policy_obs[:, 32:35],
        policy_obs[:, 39:42],
        ~giver_is_robot_1,
    )
    receiver_orientation = select_role(
        policy_obs[:, 35:39],
        policy_obs[:, 42:46],
        ~giver_is_robot_1,
    )
    object_pose_in_giver = select_role(
        policy_obs[:, 46:53],
        policy_obs[:, 53:60],
        giver_is_robot_1,
    )
    object_pose_in_receiver = select_role(
        policy_obs[:, 46:53],
        policy_obs[:, 53:60],
        ~giver_is_robot_1,
    )
    object_in_giver = object_pose_in_giver[:, :3]
    object_in_receiver = object_pose_in_receiver[:, :3]
    giver_contacts = select_role(
        policy_obs[:, 66:68],
        policy_obs[:, 68:70],
        giver_is_robot_1,
    )
    receiver_contacts = select_role(
        policy_obs[:, 66:68],
        policy_obs[:, 68:70],
        ~giver_is_robot_1,
    )
    phase = torch.argmax(policy_obs[:, 77:82], dim=-1)
    pickup_recovery_context = policy_obs[:, 98] > 0.5

    def approach_action(
        ee_position,
        object_position,
        object_orientation,
        use_object_relative_grasp,
        grasp_offset,
    ):
        local_grasp_offset = torch.zeros_like(object_position)
        local_grasp_offset[:, 0] = grasp_offset[0]
        local_grasp_offset[:, 1] = grasp_offset[1]
        local_grasp_offset[:, 2] = grasp_offset[2]
        quaternion_x = object_orientation[:, 0]
        quaternion_y = object_orientation[:, 1]
        quaternion_z = object_orientation[:, 2]
        quaternion_w = object_orientation[:, 3]
        yaw_sine = 2.0 * (
            quaternion_w * quaternion_z
            + quaternion_x * quaternion_y
        )
        yaw_cosine = 1.0 - 2.0 * (
            quaternion_y * quaternion_y
            + quaternion_z * quaternion_z
        )
        object_relative_grasp_offset = local_grasp_offset.clone()
        object_relative_grasp_offset[:, 0] = (
            yaw_cosine * local_grasp_offset[:, 0]
            - yaw_sine * local_grasp_offset[:, 1]
        )
        object_relative_grasp_offset[:, 1] = (
            yaw_sine * local_grasp_offset[:, 0]
            + yaw_cosine * local_grasp_offset[:, 1]
        )
        rotated_grasp_offset = torch.where(
            use_object_relative_grasp.unsqueeze(-1),
            object_relative_grasp_offset,
            local_grasp_offset,
        )
        grasp_position = object_position.clone()
        grasp_position += rotated_grasp_offset
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
        object_pose_in_giver[:, 3:7],
        pickup_recovery_context,
        giver_grasp_offset,
    )
    giver_recovery_position = object_in_giver.clone()
    giver_recovery_offset = torch.zeros_like(object_in_giver)
    giver_recovery_offset[:, 0] = giver_grasp_offset[0]
    giver_recovery_offset[:, 1] = giver_grasp_offset[1]
    giver_recovery_offset[:, 2] = giver_grasp_offset[2]
    giver_quaternion_x = object_pose_in_giver[:, 3]
    giver_quaternion_y = object_pose_in_giver[:, 4]
    giver_quaternion_z = object_pose_in_giver[:, 5]
    giver_quaternion_w = object_pose_in_giver[:, 6]
    giver_yaw_sine = 2.0 * (
        giver_quaternion_w * giver_quaternion_z
        + giver_quaternion_x * giver_quaternion_y
    )
    giver_yaw_cosine = 1.0 - 2.0 * (
        giver_quaternion_y * giver_quaternion_y
        + giver_quaternion_z * giver_quaternion_z
    )
    giver_object_relative_recovery_offset = (
        giver_recovery_offset.clone()
    )
    giver_object_relative_recovery_offset[:, 0] = (
        giver_yaw_cosine * giver_recovery_offset[:, 0]
        - giver_yaw_sine * giver_recovery_offset[:, 1]
    )
    giver_object_relative_recovery_offset[:, 1] = (
        giver_yaw_sine * giver_recovery_offset[:, 0]
        + giver_yaw_cosine * giver_recovery_offset[:, 1]
    )
    giver_recovery_position += torch.where(
        pickup_recovery_context.unsqueeze(-1),
        giver_object_relative_recovery_offset,
        giver_recovery_offset,
    )
    giver_recovery_position[:, 2] += approach_height
    giver_recovery_action = (
        (giver_recovery_position - giver_ee) / position_scale
    ).clamp(
        -slow_approach_action_limit,
        slow_approach_action_limit,
    )
    receiver_approach, receiver_distance = approach_action(
        receiver_ee,
        object_in_receiver,
        object_pose_in_receiver[:, 3:7],
        torch.zeros_like(pickup_recovery_context),
        receiver_grasp_offset,
    )

    root_2_in_giver = object_in_giver - object_in_receiver
    presentation_in_giver = (
        presentation_fraction_from_giver * root_2_in_giver
    )
    giver_target = presentation_in_giver.clone()
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
    giver_vertical_limit = torch.where(
        vertical_only,
        torch.full_like(
            giver_error[:, 2],
            pickup_vertical_action_limit,
        ),
        torch.full_like(
            giver_error[:, 2],
            carry_vertical_action_limit,
        ),
    ).unsqueeze(-1)
    giver_vertical_limit = torch.where(
        (
            vertical_only
            & pickup_recovery_context
        ).unsqueeze(-1),
        torch.full_like(
            giver_vertical_limit,
            recovery_pickup_vertical_action_limit,
        ),
        giver_vertical_limit,
    )
    giver_vertical_action = torch.maximum(
        torch.minimum(
            giver_error[:, 2:],
            giver_vertical_limit,
        ),
        -giver_vertical_limit,
    )
    giver_carry = torch.cat(
        (
            giver_error[:, :2].clamp(
                -carry_lateral_action_limit,
                carry_lateral_action_limit,
            ),
            giver_vertical_action,
        ),
        dim=-1,
    )
    giver_bilateral_contact = torch.all(
        giver_contacts > normalized_contact_threshold,
        dim=-1,
    )
    giver_contact_count = torch.sum(
        giver_contacts
        > giver_transport_normalized_contact_threshold,
        dim=-1,
    )
    giver_any_contact = torch.any(
        giver_contacts > normalized_contact_threshold,
        dim=-1,
    )
    receiver_any_contact = torch.any(
        receiver_contacts > normalized_contact_threshold,
        dim=-1,
    )
    receiver_bilateral_contact = torch.all(
        receiver_contacts > normalized_contact_threshold,
        dim=-1,
    )
    giver_carry_mode = (phase >= 1) & (phase <= 2)
    giver_transport_active = (
        giver_carry_mode
        & (giver_contact_count >= giver_transport_min_contact_jaws)
    )

    presentation_ready = (
        torch.linalg.vector_norm(
            giver_target - object_in_giver,
            dim=-1,
        )
        < presentation_ready_tolerance
    )
    if policy_obs.shape[1] >= 107:
        presentation_stable = policy_obs[:, 103] >= 1.0
        receiver_retry_active = policy_obs[:, 105] > 0.5
    else:
        presentation_stable = presentation_ready
        receiver_retry_active = torch.zeros_like(
            presentation_ready
        )
    receiver_approach_active = (
        (phase == 2)
        & presentation_stable
        & giver_bilateral_contact
        & ~receiver_any_contact
        & ~receiver_retry_active
    )

    giver_translation = torch.where(
        giver_transport_active.unsqueeze(-1),
        giver_carry,
        giver_approach,
    )
    giver_contact_recovery = giver_approach.clamp(
        -giver_contact_recovery_action_limit,
        giver_contact_recovery_action_limit,
    )
    giver_translation = torch.where(
        (
            giver_carry_mode
            & ~giver_transport_active
        ).unsqueeze(-1),
        giver_contact_recovery,
        giver_translation,
    )
    giver_presentation_hold = giver_carry.clamp(
        -presentation_hold_action_limit,
        presentation_hold_action_limit,
    )
    giver_translation = torch.where(
        (
            (phase == 2)
            & giver_bilateral_contact
            & presentation_stable
            & ~receiver_any_contact
        ).unsqueeze(-1),
        giver_presentation_hold,
        giver_translation,
    )
    giver_translation = torch.where(
        (
            (phase == 2)
            & giver_bilateral_contact
            & receiver_any_contact
        ).unsqueeze(-1),
        torch.zeros_like(giver_translation),
        giver_translation,
    )
    giver_retreat = torch.zeros_like(giver_translation)
    giver_retreat[:, 2] = carry_lateral_action_limit
    giver_release_translation = torch.where(
        ((phase == 3) & giver_any_contact).unsqueeze(-1),
        torch.zeros_like(giver_translation),
        giver_retreat,
    )
    giver_translation = torch.where(
        (phase >= 3).unsqueeze(-1),
        giver_release_translation,
        giver_translation,
    )
    giver_translation = torch.where(
        (phase == 4).unsqueeze(-1),
        giver_recovery_action,
        giver_translation,
    )
    receiver_wait = torch.zeros_like(receiver_approach)
    receiver_translation = torch.where(
        receiver_approach_active.unsqueeze(-1),
        receiver_approach,
        receiver_wait,
    )
    receiver_translation = torch.where(
        (phase >= 3).unsqueeze(-1),
        torch.zeros_like(receiver_translation),
        receiver_translation,
    )
    receiver_translation = torch.where(
        ((phase == 2) & receiver_any_contact).unsqueeze(-1),
        torch.zeros_like(receiver_translation),
        receiver_translation,
    )
    receiver_retry_translation = torch.zeros_like(
        receiver_translation
    )
    receiver_retry_translation[:, 2] = slow_approach_action_limit
    receiver_translation = torch.where(
        (
            (phase == 2) & receiver_retry_active
        ).unsqueeze(-1),
        receiver_retry_translation,
        receiver_translation,
    )
    receiver_contact_imbalance = (
        receiver_contacts[:, 1] - receiver_contacts[:, 0]
    )
    receiver_contact_centering = torch.sign(
        receiver_contact_imbalance
    ) * receiver_contact_centering_action_limit
    receiver_translation[:, 2] += torch.where(
        (
            (phase == 2)
            & giver_bilateral_contact
            & receiver_any_contact
            & ~receiver_retry_active
        ),
        receiver_contact_centering,
        torch.zeros_like(receiver_contact_centering),
    )

    giver_closing = (
        (giver_distance < close_distance)
        | torch.any(
            giver_contacts > normalized_contact_threshold,
            dim=-1,
        )
        | ((phase >= 1) & (phase <= 2))
    ) & (phase < 3)
    giver_closing |= (
        (phase == 3)
        & ~receiver_bilateral_contact
    )
    receiver_closing = (
        ((phase == 2) | (phase == 3))
        & ~receiver_retry_active
        & (
            (receiver_distance < receiver_close_distance)
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
    from isaaclab.utils.math import (
        axis_angle_from_quat,
        quat_conjugate,
        quat_mul,
    )

    giver_object_orientation = object_pose_in_giver[:, 3:7]
    giver_object_angular_velocity = policy_obs[:, 63:66]
    giver_target_orientation = torch.zeros_like(
        giver_object_orientation
    )
    giver_target_orientation[:, 3] = 1.0
    giver_orientation_error = axis_angle_from_quat(
        quat_mul(
            giver_target_orientation,
            quat_conjugate(giver_object_orientation),
        )
    )
    giver_orientation_action = (
        (
            giver_orientation_error
            - 0.001 * giver_object_angular_velocity
        )
        / orientation_scale
    ).clamp(-0.035, 0.035)
    giver_orientation_action = torch.where(
        giver_transport_active.unsqueeze(-1),
        giver_orientation_action,
        torch.zeros_like(giver_orientation_action),
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
    receiver_orientation_action = torch.where(
        receiver_approach_active.unsqueeze(-1),
        receiver_orientation_action,
        torch.zeros_like(receiver_orientation_action),
    )
    giver_action = torch.cat(
        (
            giver_translation,
            giver_orientation_action,
            giver_gripper,
        ),
        dim=-1,
    )
    receiver_action = torch.cat(
        (
            receiver_translation,
            receiver_orientation_action,
            receiver_gripper,
        ),
        dim=-1,
    )
    robot_1_action = torch.where(
        giver_is_robot_1.unsqueeze(-1),
        giver_action,
        receiver_action,
    )
    robot_2_action = torch.where(
        giver_is_robot_1.unsqueeze(-1),
        receiver_action,
        giver_action,
    )
    return torch.cat(
        (robot_1_action, robot_2_action),
        dim=-1,
    ).clamp(-1.0, 1.0)


def _pretraining_algorithm(task: str) -> str:
    if task.endswith("-IK-Rel-Structured-v0"):
        return "physics_structured_phase_residual_dagger_distillation"
    if "Handover-Needle-Dual-PSM-IK-Rel" in task:
        return "exact_closest_arm_handover_base_plus_bounded_residual"
    if "Lift-" in task:
        return "analytic_grasp_lift_base_plus_learned_residual"
    return "analytic_relative_ik_base_plus_learned_residual"


def _pretrain(args: argparse.Namespace, repo_root: Path) -> int:
    """Distill a declared teacher, then validate the resulting policy."""
    if args.dagger_warmup_updates < 0:
        return _fail("DAgger warm-up updates must be non-negative")
    if not 0.0 <= args.dagger_min_teacher_fraction <= 1.0:
        return _fail(
            "DAgger minimum teacher fraction must be between 0 and 1"
        )
    if args.e2e_replay_capacity_per_phase <= 0:
        return _fail("end-to-end replay capacity must be positive")
    if args.e2e_replay_batch_size <= 0:
        return _fail("end-to-end replay batch size must be positive")
    if args.e2e_samples_per_phase_step <= 0:
        return _fail("end-to-end phase samples per step must be positive")
    if args.e2e_student_segment_steps <= 0:
        return _fail("end-to-end student segment must be positive")
    if args.e2e_teacher_recovery_steps <= 0:
        return _fail("end-to-end teacher recovery must be positive")
    import gymnasium as gym
    import torch
    import torch.nn.functional as functional
    from tensordict import TensorDict
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
    is_end_to_end_handover = args.task.endswith(
        "-IK-Rel-Structured-v0"
    )

    class PhaseBalancedReplay:
        """Bounded CPU replay with equal sampling across observed task phases."""

        def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            capacity_per_phase: int,
            samples_per_phase_step: int,
        ) -> None:
            self.capacity = capacity_per_phase
            self.samples_per_phase_step = samples_per_phase_step
            self.observations = torch.empty(
                (5, capacity_per_phase, observation_dim),
                dtype=torch.float32,
            )
            self.actions = torch.empty(
                (5, capacity_per_phase, action_dim),
                dtype=torch.float32,
            )
            self.counts = [0] * 5
            self.positions = [0] * 5

        def add(
            self,
            policy_observation: torch.Tensor,
            teacher_action: torch.Tensor,
        ) -> None:
            phase = torch.argmax(
                policy_observation[:, 77:82],
                dim=-1,
            )
            for phase_index in range(5):
                indices = torch.nonzero(
                    phase == phase_index,
                    as_tuple=False,
                ).flatten()
                if indices.numel() == 0:
                    continue
                if indices.numel() > self.samples_per_phase_step:
                    selected = torch.randperm(
                        indices.numel(),
                        device=indices.device,
                    )[: self.samples_per_phase_step]
                    indices = indices[selected]
                phase_observations = (
                    policy_observation[indices].detach().cpu()
                )
                phase_actions = teacher_action[indices].detach().cpu()
                count = phase_observations.shape[0]
                destination = (
                    torch.arange(count) + self.positions[phase_index]
                ) % self.capacity
                self.observations[
                    phase_index, destination
                ] = phase_observations
                self.actions[phase_index, destination] = phase_actions
                self.positions[phase_index] = int(
                    (self.positions[phase_index] + count) % self.capacity
                )
                self.counts[phase_index] = min(
                    self.capacity,
                    self.counts[phase_index] + count,
                )

        @property
        def size(self) -> int:
            return sum(self.counts)

        def sample(
            self,
            batch_size: int,
            device: str,
        ) -> tuple[TensorDict, torch.Tensor]:
            available_phases = [
                index for index, count in enumerate(self.counts) if count
            ]
            if not available_phases:
                raise RuntimeError("phase replay is empty")
            samples_per_phase = max(
                1,
                batch_size // len(available_phases),
            )
            observation_batches = []
            action_batches = []
            for phase_index in available_phases:
                sample_indices = torch.randint(
                    self.counts[phase_index],
                    (samples_per_phase,),
                )
                observation_batches.append(
                    self.observations[
                        phase_index, sample_indices
                    ]
                )
                action_batches.append(
                    self.actions[phase_index, sample_indices]
                )
            replay_observation = torch.cat(
                observation_batches,
                dim=0,
            ).to(device)
            replay_action = torch.cat(
                action_batches,
                dim=0,
            ).to(device)
            shuffle = torch.randperm(
                replay_observation.shape[0],
                device=device,
            )
            replay_observation = replay_observation[shuffle]
            replay_action = replay_action[shuffle]
            return (
                TensorDict(
                    {"policy": replay_observation},
                    batch_size=[replay_observation.shape[0]],
                ),
                replay_action,
            )

    replay = (
        PhaseBalancedReplay(
            observation_dim=obs["policy"].shape[-1],
            action_dim=env.unwrapped.action_manager.total_action_dim,
            capacity_per_phase=args.e2e_replay_capacity_per_phase,
            samples_per_phase_step=args.e2e_samples_per_phase_step,
        )
        if is_end_to_end_handover
        else None
    )

    def imitation_loss(
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not is_end_to_end_handover:
            return functional.smooth_l1_loss(predicted, target)
        per_channel = functional.smooth_l1_loss(
            predicted,
            target,
            reduction="none",
            beta=0.01,
        )
        channel_weights = predicted.new_tensor(
            [
                4.0,
                4.0,
                4.0,
                2.0,
                2.0,
                2.0,
                6.0,
                4.0,
                4.0,
                4.0,
                2.0,
                2.0,
                2.0,
                6.0,
            ]
        )
        active_weights = 1.0 + 2.0 * (
            target.abs() > 0.02
        ).to(predicted.dtype)
        weighted = per_channel * channel_weights * active_weights
        return weighted.sum() / (
            channel_weights.sum() * predicted.shape[0]
        )

    losses: list[float] = []
    teacher_successes = 0
    teacher_completed = 0
    teacher_controlled_frames = 0
    student_controlled_frames = 0
    dagger_teacher_fractions: list[float] = []
    contiguous_student_steps = 0
    contiguous_teacher_recovery_steps = 0
    started = time.perf_counter()
    try:
        for update_index in range(args.updates):
            teacher_actions = _teacher_action(
                obs,
                args.task,
                position_scale=args.position_scale,
                orientation_scale=args.orientation_scale,
            )
            policy.update_normalization(obs)
            predicted_actions = policy(obs)
            if replay is not None:
                replay.add(obs["policy"], teacher_actions)
            if replay is not None and replay.size >= args.e2e_replay_batch_size:
                replay_obs, replay_actions = replay.sample(
                    args.e2e_replay_batch_size,
                    agent_cfg.device,
                )
                policy.update_normalization(replay_obs)
                replay_predictions = policy(replay_obs)
                loss = imitation_loss(
                    replay_predictions,
                    replay_actions,
                )
            else:
                loss = imitation_loss(
                    predicted_actions,
                    teacher_actions,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))

            with torch.no_grad():
                rollout_actions = teacher_actions
                teacher_fraction = 1.0
                if is_end_to_end_handover:
                    warmup_updates = min(
                        args.dagger_warmup_updates,
                        args.updates,
                    )
                    if update_index >= warmup_updates:
                        cycle_index = (
                            update_index - warmup_updates
                        ) % (
                            args.e2e_student_segment_steps
                            + args.e2e_teacher_recovery_steps
                        )
                        student_control = (
                            cycle_index
                            < args.e2e_student_segment_steps
                        )
                        if student_control:
                            rollout_actions = predicted_actions.detach()
                            teacher_fraction = 0.0
                            contiguous_student_steps += 1
                        else:
                            contiguous_teacher_recovery_steps += 1
                    if teacher_fraction == 1.0:
                        teacher_controlled_frames += env.unwrapped.num_envs
                    else:
                        student_controlled_frames += env.unwrapped.num_envs
                    dagger_teacher_fractions.append(teacher_fraction)
                elif "Handover-Needle-Dual-PSM-IK-Rel" in args.task:
                    warmup_updates = min(
                        args.dagger_warmup_updates,
                        args.updates,
                    )
                    if update_index >= warmup_updates:
                        schedule_steps = max(
                            args.updates - warmup_updates - 1,
                            1,
                        )
                        progress = (
                            update_index - warmup_updates
                        ) / schedule_steps
                        teacher_fraction = max(
                            args.dagger_min_teacher_fraction,
                            1.0 - progress,
                        )
                    teacher_mask = (
                        torch.rand(
                            env.unwrapped.num_envs,
                            device=agent_cfg.device,
                        )
                        < teacher_fraction
                    )
                    rollout_actions = torch.where(
                        teacher_mask.unsqueeze(-1),
                        teacher_actions,
                        predicted_actions.detach(),
                    )
                    teacher_controlled_frames += int(
                        teacher_mask.sum().item()
                    )
                    student_controlled_frames += int(
                        (~teacher_mask).sum().item()
                    )
                    dagger_teacher_fractions.append(
                        teacher_fraction
                    )
                else:
                    teacher_controlled_frames += (
                        env.unwrapped.num_envs
                    )
                obs, _, dones, _ = env.step(rollout_actions)
                successes = env.unwrapped.termination_manager.get_term("success")
            obs = obs.to(agent_cfg.device)
            teacher_successes += int(successes.sum().item())
            teacher_completed += int(dones.sum().item())

        consolidation_losses: list[float] = []
        if replay is not None:
            for _ in range(args.e2e_consolidation_updates):
                replay_obs, replay_actions = replay.sample(
                    args.e2e_replay_batch_size,
                    agent_cfg.device,
                )
                policy.update_normalization(replay_obs)
                replay_predictions = policy(replay_obs)
                consolidation_loss = imitation_loss(
                    replay_predictions,
                    replay_actions,
                )
                optimizer.zero_grad(set_to_none=True)
                consolidation_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
                value = float(consolidation_loss.detach().item())
                consolidation_losses.append(value)
                losses.append(value)

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
                "control_schedule": (
                    (
                        "teacher_warmup_then_contiguous_student_dagger_segments"
                        if is_end_to_end_handover
                        else "teacher_warmup_then_linear_dagger_mixture"
                    )
                    if dagger_teacher_fractions
                    else "teacher_only"
                ),
                "teacher_controlled_frames": teacher_controlled_frames,
                "student_controlled_frames": student_controlled_frames,
                "teacher_fraction_initial": (
                    dagger_teacher_fractions[0]
                    if dagger_teacher_fractions
                    else 1.0
                ),
                "teacher_fraction_final": (
                    dagger_teacher_fractions[-1]
                    if dagger_teacher_fractions
                    else 1.0
                ),
                "completed_episodes": teacher_completed,
                "successful_episodes": teacher_successes,
                "success_rate": (
                    teacher_successes / teacher_completed
                    if teacher_completed
                    else None
                ),
            },
            "end_to_end_replay": (
                {
                    "capacity_per_phase": replay.capacity,
                    "stored_samples_per_phase": replay.counts,
                    "balanced_batch_size": args.e2e_replay_batch_size,
                    "samples_per_phase_per_step": (
                        args.e2e_samples_per_phase_step
                    ),
                    "loss_beta": 0.01,
                    "channel_weighting": (
                        "translation_4_orientation_2_jaw_6_"
                        "active_teacher_channels_x3"
                    ),
                    "student_segment_steps": (
                        args.e2e_student_segment_steps
                    ),
                    "teacher_recovery_steps": (
                        args.e2e_teacher_recovery_steps
                    ),
                    "contiguous_student_segments_total_steps": (
                        contiguous_student_steps
                    ),
                    "contiguous_teacher_recovery_total_steps": (
                        contiguous_teacher_recovery_steps
                    ),
                    "consolidation_updates": (
                        args.e2e_consolidation_updates
                    ),
                    "consolidation_loss_initial": (
                        consolidation_losses[0]
                        if consolidation_losses
                        else None
                    ),
                    "consolidation_loss_final": (
                        consolidation_losses[-1]
                        if consolidation_losses
                        else None
                    ),
                    "consolidation_loss_minimum": (
                        min(consolidation_losses)
                        if consolidation_losses
                        else None
                    ),
                }
                if replay is not None
                else None
            ),
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
        return max(1, self._step_count // self.num_steps_per_env)

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
    if args.learning_rate is not None:
        if args.learning_rate <= 0.0:
            return _fail("training learning rate must be positive")
        agent_cfg.algorithm.learning_rate = args.learning_rate
    receiver_curriculum = bool(
        getattr(env_cfg, "dr_anmar_receiver_curriculum", False)
    )
    receiver_grasp_retain_curriculum = bool(
        getattr(
            env_cfg,
            "dr_anmar_receiver_grasp_retain_curriculum",
            False,
        )
    )
    recovery_receiver_grasp_retain_curriculum = bool(
        getattr(
            env_cfg,
            "dr_anmar_recovery_receiver_grasp_retain_curriculum",
            False,
        )
    )
    joint_transfer_acquisition_curriculum = bool(
        getattr(
            env_cfg,
            "dr_anmar_joint_transfer_acquisition_curriculum",
            False,
        )
    )
    transfer_refinement_curriculum = bool(
        getattr(
            env_cfg,
            "dr_anmar_transfer_refinement_curriculum",
            False,
        )
    )
    deadline_recovery_curriculum = bool(
        getattr(
            env_cfg,
            "dr_anmar_deadline_recovery_curriculum",
            False,
        )
    )
    pickup_recovery_curriculum = bool(
        getattr(
            env_cfg,
            "dr_anmar_pickup_recovery_curriculum",
            False,
        )
    )
    focused_curriculum = (
        receiver_curriculum or pickup_recovery_curriculum
    )
    if focused_curriculum and not args.checkpoint:
        return _fail(
            "focused handover curriculum requires a qualified initial checkpoint"
        )
    if focused_curriculum:
        # Receiver-only adaptation has a narrow optimum: the physical replay
        # cache needs several iterations to fill, while later PPO updates can
        # overfit that cache. Preserve every update so promotion can be based
        # on full-task qualification instead of assuming the final update is
        # the best policy. Focused option learning also uses a fixed optimizer
        # schedule: adaptive KL scheduling previously amplified an explicitly
        # requested 1e-5 rate to 1e-2 before the replay cache was representative.
        agent_cfg.save_interval = 1
        agent_cfg.algorithm.schedule = "fixed"
    if transfer_refinement_curriculum:
        refinement_rollout_steps = int(
            getattr(
                env_cfg,
                "dr_anmar_transfer_refinement_rollout_steps_per_env",
                128,
            )
        )
        if refinement_rollout_steps <= 0:
            return _fail(
                "transfer-refinement rollout steps must be positive"
            )
        agent_cfg.num_steps_per_env = refinement_rollout_steps
    if deadline_recovery_curriculum:
        deadline_rollout_steps = int(
            getattr(
                env_cfg,
                "dr_anmar_deadline_recovery_rollout_steps_per_env",
                128,
            )
        )
        if deadline_rollout_steps <= 0:
            return _fail(
                "deadline-recovery rollout steps must be positive"
            )
        agent_cfg.num_steps_per_env = deadline_rollout_steps
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
    initial_checkpoint = None
    if args.checkpoint:
        initial_checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not initial_checkpoint.is_file():
            env.close()
            return _fail(
                f"initial checkpoint not found: {initial_checkpoint}"
            )
        load_cfg = None
        if (
            args.handover_giver_adaptation
            or receiver_curriculum
            or pickup_recovery_curriculum
        ):
            if (
                args.handover_giver_adaptation
                and "Handover-Needle-Dual-PSM-IK-Rel" not in args.task
            ):
                env.close()
                return _fail(
                    "handover adaptation requires the dual-PSM needle handover task"
                )
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            }
        runner.load(str(initial_checkpoint), load_cfg=load_cfg)
        if deadline_recovery_curriculum:
            policy_model = runner.alg.get_policy()
            configure_deadline_recovery = getattr(
                policy_model,
                "configure_deadline_recovery_adaptation",
                None,
            )
            if configure_deadline_recovery is None:
                env.close()
                return _fail(
                    "handover policy does not support deadline-aware recovery"
                )
            configure_deadline_recovery()
        elif transfer_refinement_curriculum:
            policy_model = runner.alg.get_policy()
            configure_refinement = getattr(
                policy_model,
                "configure_transfer_refinement_adaptation",
                None,
            )
            if configure_refinement is None:
                env.close()
                return _fail(
                    "handover policy does not support transfer refinement"
                )
            configure_refinement()
            refinement_controller_cfg = getattr(
                env_cfg,
                "dr_anmar_transfer_refinement_controller",
                {},
            )
            refinement_controller = getattr(
                policy_model,
                "controller",
                None,
            )
            for attribute, value in refinement_controller_cfg.items():
                if refinement_controller is None or not hasattr(
                    refinement_controller,
                    attribute,
                ):
                    env.close()
                    return _fail(
                        "transfer-refinement curriculum controller does not "
                        f"expose {attribute}"
                    )
                setattr(refinement_controller, attribute, value)
        elif joint_transfer_acquisition_curriculum:
            policy_model = runner.alg.get_policy()
            configure_joint_transfer = getattr(
                policy_model,
                "configure_joint_transfer_acquisition_adaptation",
                None,
            )
            if configure_joint_transfer is None:
                env.close()
                return _fail(
                    "handover policy does not support joint "
                    "transfer-acquisition adaptation"
                )
            configure_joint_transfer()
            joint_controller_cfg = getattr(
                env_cfg,
                "dr_anmar_joint_transfer_acquisition_controller",
                {},
            )
            joint_controller = getattr(
                policy_model,
                "controller",
                None,
            )
            for attribute, value in joint_controller_cfg.items():
                if joint_controller is None or not hasattr(
                    joint_controller,
                    attribute,
                ):
                    env.close()
                    return _fail(
                        "joint transfer-acquisition curriculum controller "
                        f"does not expose {attribute}"
                    )
                setattr(joint_controller, attribute, value)
        elif pickup_recovery_curriculum:
            policy_model = runner.alg.get_policy()
            if not hasattr(
                policy_model,
                "configure_pickup_recovery_adaptation",
            ):
                env.close()
                return _fail(
                    "handover policy does not support pickup-recovery adaptation"
                )
            policy_model.configure_pickup_recovery_adaptation()
            recovery_controller_cfg = getattr(
                env_cfg,
                "dr_anmar_pickup_recovery_controller",
                {},
            )
            recovery_controller = getattr(
                policy_model,
                "controller",
                None,
            )
            for attribute, value in recovery_controller_cfg.items():
                if recovery_controller is None or not hasattr(
                    recovery_controller,
                    attribute,
                ):
                    env.close()
                    return _fail(
                        "pickup-recovery curriculum controller does not "
                        f"expose {attribute}"
                    )
                setattr(recovery_controller, attribute, value)
        elif recovery_receiver_grasp_retain_curriculum:
            policy_model = runner.alg.get_policy()
            configure_recovery_receiver = getattr(
                policy_model,
                "configure_recovery_receiver_grasp_retain_adaptation",
                None,
            )
            if configure_recovery_receiver is None:
                env.close()
                return _fail(
                    "handover policy does not support recovery-conditioned "
                    "receiver grasp-retain adaptation"
                )
            configure_recovery_receiver()
            recovery_receiver_controller_cfg = getattr(
                env_cfg,
                "dr_anmar_recovery_receiver_controller",
                {},
            )
            recovery_receiver_controller = getattr(
                policy_model,
                "controller",
                None,
            )
            for attribute, value in (
                recovery_receiver_controller_cfg.items()
            ):
                if (
                    recovery_receiver_controller is None
                    or not hasattr(
                        recovery_receiver_controller,
                        attribute,
                    )
                ):
                    env.close()
                    return _fail(
                        "recovery-conditioned receiver curriculum "
                        f"controller does not expose {attribute}"
                    )
                setattr(
                    recovery_receiver_controller,
                    attribute,
                    value,
                )
        elif args.handover_giver_adaptation:
            policy_model = runner.alg.get_policy()
            if not hasattr(
                policy_model, "configure_giver_adaptation"
            ):
                env.close()
                return _fail(
                    "handover policy does not support giver adaptation"
                )
            policy_model.configure_giver_adaptation()
        elif receiver_grasp_retain_curriculum:
            policy_model = runner.alg.get_policy()
            if not hasattr(
                policy_model,
                "configure_receiver_grasp_retain_adaptation",
            ):
                env.close()
                return _fail(
                    "handover policy does not support receiver grasp-retain adaptation"
                )
            policy_model.configure_receiver_grasp_retain_adaptation()
        elif receiver_curriculum:
            policy_model = runner.alg.get_policy()
            if not hasattr(
                policy_model, "configure_receiver_adaptation"
            ):
                env.close()
                return _fail(
                    "handover policy does not support receiver adaptation"
                )
            policy_model.configure_receiver_adaptation()
        if args.learning_rate is not None:
            for parameter_group in runner.alg.optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
    elif args.handover_giver_adaptation:
        if "Handover-Needle-Dual-PSM-IK-Rel" not in args.task:
            env.close()
            return _fail(
                "giver adaptation requires the dual-PSM needle handover task"
            )
        policy_model = runner.alg.get_policy()
        if not hasattr(policy_model, "configure_giver_adaptation"):
            env.close()
            return _fail(
                "handover policy does not support giver adaptation"
            )
        policy_model.configure_giver_adaptation()
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
                init_at_random_ep_len=not focused_curriculum,
            )
        duration = time.perf_counter() - started
        checkpoint = run_dir / "model_final.pt"
        runner.save(str(checkpoint))
        iterations = max(1, early.framework_iteration_count)
        simulated_frames = (
            env.unwrapped.num_envs * agent_cfg.num_steps_per_env * iterations
        )
        success_history = [float(value) for value in early.tracker.history]
        receiver_curriculum_cache = getattr(
            env.unwrapped,
            "_dr_anmar_receiver_curriculum_cache",
            None,
        )
        pickup_recovery_curriculum_cache = getattr(
            env.unwrapped,
            "_dr_anmar_pickup_recovery_curriculum_cache",
            None,
        )
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
            "policy_learning_rate": float(
                args.learning_rate
                if args.learning_rate is not None
                else agent_cfg.algorithm.learning_rate
            ),
            "policy_learning_rate_schedule": str(
                agent_cfg.algorithm.schedule
            ),
            "optimizer_learning_rates_final": [
                float(group["lr"])
                for group in runner.alg.optimizer.param_groups
            ],
            "handover_giver_adaptation": bool(
                args.handover_giver_adaptation
            ),
            "receiver_curriculum": receiver_curriculum,
            "receiver_grasp_retain_curriculum": (
                receiver_grasp_retain_curriculum
            ),
            "recovery_receiver_grasp_retain_curriculum": (
                recovery_receiver_grasp_retain_curriculum
            ),
            "joint_transfer_acquisition_curriculum": (
                joint_transfer_acquisition_curriculum
            ),
            "transfer_refinement_curriculum": (
                transfer_refinement_curriculum
            ),
            "deadline_recovery_curriculum": (
                deadline_recovery_curriculum
            ),
            "deadline_recovery_objective": (
                getattr(
                    env_cfg,
                    "dr_anmar_deadline_recovery_objective",
                    None,
                )
                if deadline_recovery_curriculum
                else None
            ),
            "deadline_recovery_adaptation_contract": (
                {
                    "source_states": (
                        "physics_owned_recovered_stable_presentation_with_"
                        "original_episode_deadline_and_complete_markov_state"
                    ),
                    "options": [
                        "continue",
                        "reseat",
                        "backoff",
                    ],
                    "option_success": "unchanged_retained_handover_terminal",
                    "incumbent_checkpoint_frozen_and_active": True,
                    "optimizer_state_reset": True,
                    "optimizer_schedule": "fixed",
                    "observation_normalizer_frozen": True,
                    "zero_impact_adapter": True,
                    "learned_receiver_axes": [
                        "x",
                        "y",
                        "z",
                        "roll",
                        "pitch",
                        "yaw",
                    ],
                    "learned_giver_axes": [],
                    "analytic_release_authority": True,
                    "hard_terminations_unchanged": True,
                    "episode_horizon_unchanged": True,
                }
                if deadline_recovery_curriculum
                else None
            ),
            "transfer_refinement_objective": (
                getattr(
                    env_cfg,
                    "dr_anmar_transfer_refinement_objective",
                    None,
                )
                if transfer_refinement_curriculum
                else None
            ),
            "transfer_refinement_controller": (
                getattr(
                    env_cfg,
                    "dr_anmar_transfer_refinement_controller",
                    None,
                )
                if transfer_refinement_curriculum
                else None
            ),
            "transfer_refinement_adaptation_contract": (
                {
                    "source_states": (
                        "physics_owned_stable_presentation_with_complete_"
                        "markov_state"
                    ),
                    "option_success": "unchanged_retained_handover_terminal",
                    "joint_transfer_checkpoint_frozen_and_active": True,
                    "optimizer_state_reset": True,
                    "optimizer_schedule": "fixed",
                    "observation_normalizer_frozen": True,
                    "zero_initialized_adapter": True,
                    "rollout_steps_per_env": agent_cfg.num_steps_per_env,
                    "learned_giver_axes": [],
                    "learned_receiver_axes_after_presentation_qualification": [
                        "x",
                        "y",
                        "z",
                        "roll",
                        "pitch",
                        "yaw",
                    ],
                    "analytic_gripper_authority": True,
                    "analytic_release_authority": True,
                    "hard_terminations_unchanged": True,
                }
                if transfer_refinement_curriculum
                else None
            ),
            "joint_transfer_acquisition_objective": (
                getattr(
                    env_cfg,
                    "dr_anmar_joint_transfer_acquisition_objective",
                    None,
                )
                if joint_transfer_acquisition_curriculum
                else None
            ),
            "joint_transfer_acquisition_controller": (
                getattr(
                    env_cfg,
                    "dr_anmar_joint_transfer_acquisition_controller",
                    None,
                )
                if joint_transfer_acquisition_curriculum
                else None
            ),
            "joint_transfer_acquisition_adaptation_contract": (
                {
                    "source_states": (
                        "physics_owned_lifted_custody_with_complete_markov_state"
                    ),
                    "option_success": "unchanged_retained_handover_terminal",
                    "promoted_pickup_recovery_policy_frozen_and_active": True,
                    "optimizer_state_reset": True,
                    "optimizer_schedule": "fixed",
                    "observation_normalizer_frozen": True,
                    "zero_initialized_adapter": True,
                    "learned_giver_axes": [
                        "x",
                        "y",
                        "z",
                        "roll",
                        "pitch",
                        "yaw",
                    ],
                    "learned_receiver_axes": [
                        "x",
                        "y",
                        "z",
                        "roll",
                        "pitch",
                        "yaw",
                    ],
                    "analytic_gripper_authority": True,
                    "analytic_release_authority": True,
                    "hard_terminations_unchanged": True,
                }
                if joint_transfer_acquisition_curriculum
                else None
            ),
            "recovery_receiver_grasp_retain_objective": (
                getattr(
                    env_cfg,
                    "dr_anmar_recovery_receiver_grasp_retain_objective",
                    None,
                )
                if recovery_receiver_grasp_retain_curriculum
                else None
            ),
            "recovery_receiver_grasp_retain_controller": (
                getattr(
                    env_cfg,
                    "dr_anmar_recovery_receiver_controller",
                    None,
                )
                if recovery_receiver_grasp_retain_curriculum
                else None
            ),
            "pickup_recovery_curriculum": pickup_recovery_curriculum,
            "pickup_recovery_curriculum_objective": (
                getattr(
                    env_cfg,
                    "dr_anmar_pickup_recovery_objective",
                    None,
                )
                if pickup_recovery_curriculum
                else None
            ),
            "pickup_recovery_curriculum_controller": (
                getattr(
                    env_cfg,
                    "dr_anmar_pickup_recovery_controller",
                    None,
                )
                if pickup_recovery_curriculum
                else None
            ),
            "pickup_recovery_curriculum_cached_envs": (
                int(
                    pickup_recovery_curriculum_cache[
                        "valid"
                    ].sum().item()
                )
                if pickup_recovery_curriculum_cache is not None
                else 0
            ),
            "pickup_recovery_curriculum_reset_restores": (
                int(pickup_recovery_curriculum_cache["reset_restores"])
                if pickup_recovery_curriculum_cache is not None
                else 0
            ),
            "pickup_recovery_curriculum_reset_refreshes": (
                int(pickup_recovery_curriculum_cache["reset_refreshes"])
                if pickup_recovery_curriculum_cache is not None
                else 0
            ),
            "pickup_recovery_curriculum_cross_environment_restores": (
                int(
                    pickup_recovery_curriculum_cache[
                        "cross_environment_restores"
                    ]
                )
                if pickup_recovery_curriculum_cache is not None
                else 0
            ),
            "pickup_recovery_curriculum_adaptation_contract": (
                {
                    "source_states": (
                        "simulator_observed_physical_pickup_loss_transitions"
                    ),
                    "option_success": (
                        "recovered_physics_owned_stable_presentation"
                    ),
                    "full_handover_evaluation_success_unchanged": True,
                    "optimizer_state_reset": True,
                    "shared_actor_features_trainable": False,
                    "observation_normalizer_frozen": True,
                    "trainable_phase_heads": [0, 1, 2, 4],
                    "trainable_role_output_rows": [0, 1],
                    "learned_giver_axes": ["x", "y"],
                    "analytic_giver_axes": [
                        "z",
                        "roll",
                        "pitch",
                        "yaw",
                        "gripper",
                    ],
                    "first_attempt_residual_frozen": True,
                    "receiver_policy_frozen_and_active": True,
                    "hard_terminations_unchanged": True,
                }
                if pickup_recovery_curriculum
                else None
            ),
            "receiver_curriculum_cached_envs": (
                int(receiver_curriculum_cache["valid"].sum().item())
                if receiver_curriculum_cache is not None
                else 0
            ),
            "receiver_curriculum_reset_restores": (
                int(receiver_curriculum_cache["reset_restores"])
                if receiver_curriculum_cache is not None
                else 0
            ),
            "receiver_curriculum_reset_refreshes": (
                int(receiver_curriculum_cache["reset_refreshes"])
                if receiver_curriculum_cache is not None
                else 0
            ),
            "receiver_curriculum_cross_environment_restores": (
                int(receiver_curriculum_cache["cross_environment_restores"])
                if receiver_curriculum_cache is not None
                else 0
            ),
            "receiver_curriculum_recovery_conditioned_captures": (
                int(
                    receiver_curriculum_cache[
                        "recovery_conditioned_captures"
                    ]
                )
                if receiver_curriculum_cache is not None
                else 0
            ),
            "receiver_curriculum_markov_state_restores": (
                int(receiver_curriculum_cache["markov_state_restores"])
                if receiver_curriculum_cache is not None
                else 0
            ),
            "receiver_curriculum_recovery_context_restores": (
                int(receiver_curriculum_cache["recovery_context_restores"])
                if receiver_curriculum_cache is not None
                else 0
            ),
            "receiver_curriculum_capture_stage": (
                str(
                    getattr(
                        env_cfg,
                        "dr_anmar_receiver_curriculum_capture_stage",
                        "stable_presentation",
                    )
                )
                if receiver_curriculum
                else None
            ),
            "receiver_curriculum_restore_probability": (
                float(
                    getattr(
                        env_cfg,
                        "dr_anmar_receiver_curriculum_restore_probability",
                        0.0,
                    )
                )
                if receiver_curriculum
                else 0.0
            ),
            "receiver_curriculum_checkpoint_interval": (
                int(agent_cfg.save_interval)
                if receiver_curriculum
                else None
            ),
            "receiver_curriculum_adaptation_contract": (
                {
                    "optimizer_state_reset": True,
                    "shared_actor_features_trainable": False,
                    "observation_normalizer_frozen": True,
                    "trainable_phase_head": (
                        [2, 3]
                        if receiver_grasp_retain_curriculum
                        else 2
                    ),
                    "trainable_role_output_rows": (
                        [7, 8, 9, 10, 11, 12]
                        if receiver_grasp_retain_curriculum
                        else [7, 8, 9]
                    ),
                    "learned_receiver_axes": (
                        ["x", "y", "z", "roll", "pitch", "yaw"]
                        if receiver_grasp_retain_curriculum
                        else ["x", "y", "z"]
                    ),
                    "analytic_receiver_axes": (
                        ["gripper"]
                        if receiver_grasp_retain_curriculum
                        else ["roll", "pitch", "yaw", "gripper"]
                    ),
                    "post_contact_authority": (
                        receiver_grasp_retain_curriculum
                    ),
                    "source_states": (
                        "simulator_observed_recovered_stable_presentations"
                        if recovery_receiver_grasp_retain_curriculum
                        else "simulator_observed_stable_presentations"
                    ),
                    "pickup_recovery_policy_frozen_and_active": (
                        recovery_receiver_grasp_retain_curriculum
                    ),
                    "option_success": (
                        "retained_handover_from_recovered_stable_presentation"
                        if recovery_receiver_grasp_retain_curriculum
                        else "unchanged_retained_handover"
                    ),
                    "full_handover_evaluation_success_unchanged": True,
                    "cross_environment_state_sampling": bool(
                        getattr(
                            env_cfg,
                            "dr_anmar_receiver_curriculum_cross_environment_sampling",
                            False,
                        )
                    ),
                    "pickup_lift_presentation_policy_frozen": True,
                    "exploration_scale_frozen": True,
                    "initial_policy_influence": "loaded_checkpoint",
                }
                if receiver_curriculum
                else None
            ),
            "handover_giver_adaptation_contract": (
                (
                    {
                        "optimizer_state_reset": True,
                        "shared_actor_features_trainable": True,
                        "non_giver_output_rows_gradient_masked": True,
                        "trainable_output_rows": [3, 4, 10, 11],
                        "learned_giver_axes": ["x", "y"],
                        "analytic_giver_axes": [
                            "z",
                            "roll",
                            "pitch",
                            "yaw",
                            "gripper",
                        ],
                        "initial_policy_influence": (
                            "loaded_checkpoint"
                            if initial_checkpoint is not None
                            else "zero_residual_analytic_base"
                        ),
                        "receiver_residual_disabled": True,
                    }
                    if not hasattr(
                        runner.alg.get_policy(),
                        "giver_adaptation_enabled",
                    )
                    else {
                        "optimizer_state_reset": True,
                        "shared_actor_features_trainable": False,
                        "observation_normalizer_frozen": True,
                        "non_giver_output_rows_gradient_masked": True,
                        "trainable_role_output_rows": [0, 1],
                        "learned_giver_axes": ["x", "y"],
                        "analytic_giver_axes": [
                            "z",
                            "roll",
                            "pitch",
                            "yaw",
                            "gripper",
                        ],
                        "receiver_policy_frozen_and_active": True,
                        "receiver_exploration_disabled": True,
                        "initial_policy_influence": "loaded_checkpoint",
                    }
                )
                if args.handover_giver_adaptation
                else None
            ),
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
            "initial_checkpoint": (
                {
                    "path": str(initial_checkpoint),
                    "sha256": _sha256(initial_checkpoint),
                }
                if initial_checkpoint is not None
                else None
            ),
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
    """Compare receiver grasp points for the staged physical needle handover."""
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
    giver_offsets = [NEEDLE_PROVISIONAL_GRASP_OFFSET_M] * len(values)
    giver_arc_fractions = [0.4] * len(values)
    receiver_roll_offsets = [0.0] * len(values)
    presentation_fractions = [0.35] * len(values)
    pickup_vertical_action_limits = [0.015] * len(values)
    carry_lateral_action_limits = [0.06] * len(values)
    carry_vertical_action_limits = [0.015] * len(values)
    receiver_close_distances = [0.001] * len(values)
    receiver_contact_centering_action_limits = [0.0025] * len(values)
    giver_transport_min_contact_jaws = [2] * len(values)
    giver_transport_normalized_contact_thresholds = [0.002] * len(values)
    giver_contact_recovery_action_limits = [1.0] * len(values)
    fixed_receiver_arc_fraction = 0.65
    selected_receiver_z_offset = -0.003
    if parameter == "giver_arc_fraction":
        if any(not 0.0 <= value <= 1.0 for value in values):
            return _fail("giver arc fractions must be between 0.0 and 1.0")
        giver_offsets = []
        for value in values:
            geometry_offset = needle_geometry_grasp_offset_m(value)
            giver_offsets.append(
                (
                    geometry_offset[0],
                    geometry_offset[1],
                    NEEDLE_PROVISIONAL_GRASP_Z_OFFSET_M,
                )
            )
        giver_arc_fractions = values
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (
                geometry_offset[0],
                geometry_offset[1],
                selected_receiver_z_offset,
            )
            for _ in values
        ]
        receiver_roll_offsets = [math.pi] * len(values)
    elif parameter == "receiver_arc_fraction":
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
        receiver_roll_offsets = [math.pi] * len(values)
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
    elif parameter == "presentation_fraction_from_giver":
        if any(not 0.1 <= value <= 0.9 for value in values):
            return _fail(
                "presentation fractions must be between 0.1 and 0.9"
            )
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (
                geometry_offset[0],
                geometry_offset[1],
                selected_receiver_z_offset,
            )
            for _ in values
        ]
        receiver_roll_offsets = [math.pi] * len(values)
        presentation_fractions = values
    elif parameter == "carry_vertical_action_limit":
        if any(not 0.01 <= value <= 0.20 for value in values):
            return _fail(
                "carry vertical action limits must be between 0.01 and 0.20"
            )
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (
                geometry_offset[0],
                geometry_offset[1],
                selected_receiver_z_offset,
            )
            for _ in values
        ]
        receiver_roll_offsets = [math.pi] * len(values)
        carry_vertical_action_limits = values
    elif parameter == "pickup_vertical_action_limit":
        if any(not 0.01 <= value <= 0.30 for value in values):
            return _fail(
                "pickup vertical action limits must be between 0.01 and 0.30"
            )
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (
                geometry_offset[0],
                geometry_offset[1],
                selected_receiver_z_offset,
            )
            for _ in values
        ]
        receiver_roll_offsets = [math.pi] * len(values)
        pickup_vertical_action_limits = values
    elif parameter == "carry_lateral_action_limit":
        if any(not 0.001 <= value <= 0.10 for value in values):
            return _fail(
                "carry lateral action limits must be between 0.001 and 0.10"
            )
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (
                geometry_offset[0],
                geometry_offset[1],
                selected_receiver_z_offset,
            )
            for _ in values
        ]
        receiver_roll_offsets = [math.pi] * len(values)
        carry_lateral_action_limits = values
    elif parameter == "receiver_close_distance":
        if any(not 0.0002 <= value <= 0.01 for value in values):
            return _fail(
                "receiver close distances must be between 0.0002 and 0.01 m"
            )
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (
                geometry_offset[0],
                geometry_offset[1],
                selected_receiver_z_offset,
            )
            for _ in values
        ]
        receiver_roll_offsets = [math.pi] * len(values)
        receiver_close_distances = values
    elif parameter == "receiver_contact_centering_action_limit":
        if any(not 0.0 <= value <= 0.10 for value in values):
            return _fail(
                "receiver contact centering action limits must be between "
                "0.0 and 0.10"
            )
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (
                geometry_offset[0],
                geometry_offset[1],
                selected_receiver_z_offset,
            )
            for _ in values
        ]
        receiver_roll_offsets = [math.pi] * len(values)
        receiver_contact_centering_action_limits = values
    elif parameter == "giver_transport_min_contact_jaws":
        if any(value not in {1.0, 2.0} for value in values):
            return _fail(
                "giver transport minimum contact jaws must be 1 or 2"
            )
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (
                geometry_offset[0],
                geometry_offset[1],
                selected_receiver_z_offset,
            )
            for _ in values
        ]
        receiver_roll_offsets = [math.pi] * len(values)
        giver_transport_min_contact_jaws = [
            int(value) for value in values
        ]
    elif parameter == "giver_transport_normalized_contact_threshold":
        if any(not 0.0001 <= value <= 0.005 for value in values):
            return _fail(
                "giver transport normalized contact thresholds must be "
                "between 0.0001 and 0.005"
            )
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (
                geometry_offset[0],
                geometry_offset[1],
                selected_receiver_z_offset,
            )
            for _ in values
        ]
        receiver_roll_offsets = [math.pi] * len(values)
        giver_transport_normalized_contact_thresholds = values
    elif parameter == "giver_contact_recovery_action_limit":
        if any(not 0.001 <= value <= 1.0 for value in values):
            return _fail(
                "giver contact recovery action limits must be between "
                "0.001 and 1.0"
            )
        geometry_offset = needle_geometry_grasp_offset_m(
            fixed_receiver_arc_fraction
        )
        receiver_offsets = [
            (
                geometry_offset[0],
                geometry_offset[1],
                selected_receiver_z_offset,
            )
            for _ in values
        ]
        receiver_roll_offsets = [math.pi] * len(values)
        giver_contact_recovery_action_limits = values
    else:
        return _fail(
            "handover-sweep parameter must be giver_arc_fraction, "
            "receiver_arc_fraction, "
            "receiver_grasp_z_offset, receiver_roll_offset_rad, or "
            "presentation_fraction_from_giver, or "
            "pickup_vertical_action_limit, carry_lateral_action_limit, "
            "carry_vertical_action_limit, "
            "receiver_close_distance, or "
            "receiver_contact_centering_action_limit, or "
            "giver_transport_min_contact_jaws, or "
            "giver_transport_normalized_contact_threshold, or "
            "giver_contact_recovery_action_limit"
        )

    env_cfg, _ = _load_configs(args.task, args.num_envs, args.seed)
    env_kwargs: dict[str, Any] = {"cfg": env_cfg}
    camera_eye = None
    camera_target = None
    if args.video:
        grid_side = math.ceil(math.sqrt(args.num_envs))
        grid_span = max(
            (grid_side - 1) * float(env_cfg.scene.env_spacing),
            1.0,
        )
        camera_visual_span = min(grid_span, 5.0)
        camera_eye = (
            0.0,
            -0.333 * camera_visual_span,
            0.245 * camera_visual_span,
        )
        camera_target = (0.0, 0.0, 0.12)
        env_cfg.viewer.resolution = (
            args.video_width,
            args.video_height,
        )
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.env_index = args.video_env_index
        env_cfg.viewer.eye = camera_eye
        env_cfg.viewer.lookat = camera_target
        env_kwargs["render_mode"] = "rgb_array"
    env = gym.make(args.task, **env_kwargs)
    if args.video:
        camera_focus = env.unwrapped.scene.env_origins[
            args.video_env_index
        ]
        camera_eye = (
            camera_eye[0] + float(camera_focus[0].item()),
            camera_eye[1] + float(camera_focus[1].item()),
            camera_eye[2],
        )
        camera_target = (
            camera_target[0] + float(camera_focus[0].item()),
            camera_target[1] + float(camera_focus[1].item()),
            camera_target[2],
        )
        env.unwrapped.sim.set_camera_view(
            camera_eye,
            camera_target,
        )
        video_folder = Path(
            args.video_folder
            or Path(args.output_path).resolve() / "videos"
        ).resolve()
        video_folder.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            step_trigger=lambda step: step == 0,
            video_length=args.num_frames,
            name_prefix=(
                f"{args.task}-seed{args.seed}"
                f"-env{args.video_env_index}"
            ),
            disable_logger=True,
        )
    obs, _ = env.reset()
    initial_giver_is_robot_1 = obs["policy"][:, 82] > 0.5
    initial_robot_1_distance = torch.linalg.vector_norm(
        obs["policy"][:, 46:49] - obs["policy"][:, 32:35],
        dim=-1,
    )
    initial_robot_2_distance = torch.linalg.vector_norm(
        obs["policy"][:, 53:56] - obs["policy"][:, 39:42],
        dim=-1,
    )

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
    initial_handover_state = getattr(
        env.unwrapped,
        "_dr_anmar_handover_state",
    )
    max_phase = initial_handover_state["progress_phase"].clone()
    completed = torch.zeros(len(values), dtype=torch.int64, device=env.unwrapped.device)
    successes = torch.zeros_like(completed)
    timeouts = torch.zeros_like(completed)
    hard_failures = torch.zeros_like(completed)
    procedural_failures = torch.zeros_like(completed)
    retention_failure_causes = {
        "low_clearance": torch.zeros_like(completed),
        "receiver_follow_error": torch.zeros_like(completed),
        "receiver_contact_loss": torch.zeros_like(completed),
    }
    hard_failure_names = (
        "object_dropping",
        "excessive_object_force",
        "protected_surface_force",
    )
    procedural_failure_names = (
        "needle_dropped_after_pickup",
        "pickup_attempts_exhausted",
        "premature_giver_release",
        "receiver_retention_lost",
    )
    failure_term_counts = {
        name: torch.zeros_like(completed)
        for name in hard_failure_names + procedural_failure_names
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
    maximum_state_clearance = torch.full_like(
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
    maximum_four_jaw_overlap_contact = torch.zeros_like(
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
    ever_receiver_jaw_1_contact = torch.zeros(
        env.unwrapped.num_envs,
        dtype=torch.bool,
        device=env.unwrapped.device,
    )
    ever_receiver_jaw_2_contact = torch.zeros_like(
        ever_receiver_jaw_1_contact
    )
    ever_receiver_bilateral_contact = torch.zeros_like(
        ever_receiver_jaw_1_contact
    )
    ever_four_jaw_overlap_contact = torch.zeros_like(
        ever_receiver_jaw_1_contact
    )
    successful_environment = torch.zeros_like(
        ever_receiver_jaw_1_contact
    )
    first_attempt_successful_environment = torch.zeros_like(
        ever_receiver_jaw_1_contact
    )
    recovered_successful_environment = torch.zeros_like(
        ever_receiver_jaw_1_contact
    )
    first_terminal_pickup_attempts = torch.zeros(
        env.unwrapped.num_envs,
        dtype=torch.long,
        device=env.unwrapped.device,
    )
    maximum_pickup_attempts_observed = torch.zeros_like(
        first_terminal_pickup_attempts
    )
    maximum_pickup_recoveries_observed = torch.zeros_like(
        first_terminal_pickup_attempts
    )
    ever_state_lifted = torch.zeros_like(
        ever_receiver_jaw_1_contact
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
        for frame_index in range(args.num_frames):
            if args.video:
                flow_progress = frame_index / max(
                    args.num_frames - 1,
                    1,
                )
                flow_eased = 0.5 - 0.5 * math.cos(
                    math.pi * flow_progress
                )
                flow_x = (
                    (flow_eased - 0.5)
                    * 0.30
                    * camera_visual_span
                )
                flow_y = (
                    0.04
                    * math.sin(math.pi * flow_progress)
                    * camera_visual_span
                )
                env.unwrapped.sim.set_camera_view(
                    (
                        camera_eye[0] + flow_x,
                        camera_eye[1] + flow_y,
                        camera_eye[2],
                    ),
                    (
                        camera_target[0] + flow_x,
                        camera_target[1] + flow_y,
                        camera_target[2],
                    ),
                )
            was_unresolved = unresolved.clone()
            current_handover_state = getattr(
                env.unwrapped,
                "_dr_anmar_handover_state",
            )
            max_phase = torch.maximum(
                max_phase,
                current_handover_state["progress_phase"],
            )
            maximum_pickup_attempts_observed = torch.maximum(
                maximum_pickup_attempts_observed,
                current_handover_state["pickup_attempt_count"],
            )
            maximum_pickup_recoveries_observed = torch.maximum(
                maximum_pickup_recoveries_observed,
                current_handover_state["pickup_recovery_count"],
            )
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
                    giver_grasp_offset=giver_offsets[group_index],
                    receiver_grasp_offset=receiver_offset,
                    receiver_roll_offset_rad=(
                        receiver_roll_offsets[group_index]
                    ),
                    presentation_fraction_from_giver=(
                        presentation_fractions[group_index]
                    ),
                    pickup_vertical_action_limit=(
                        pickup_vertical_action_limits[group_index]
                    ),
                    carry_lateral_action_limit=(
                        carry_lateral_action_limits[group_index]
                    ),
                    carry_vertical_action_limit=(
                        carry_vertical_action_limits[group_index]
                    ),
                    receiver_close_distance=(
                        receiver_close_distances[group_index]
                    ),
                    receiver_contact_centering_action_limit=(
                        receiver_contact_centering_action_limits[
                            group_index
                        ]
                    ),
                    giver_transport_min_contact_jaws=(
                        giver_transport_min_contact_jaws[group_index]
                    ),
                    giver_transport_normalized_contact_threshold=(
                        giver_transport_normalized_contact_thresholds[
                            group_index
                        ]
                    ),
                    giver_contact_recovery_action_limit=(
                        giver_contact_recovery_action_limits[
                            group_index
                        ]
                    ),
                )
                giver_ee = group_obs["policy"][:, 32:35]
                giver_grasp = group_obs["policy"][:, 46:49].clone()
                giver_grasp[:, 0] += giver_offsets[group_index][0]
                giver_grasp[:, 1] += giver_offsets[group_index][1]
                giver_grasp[:, 2] += giver_offsets[group_index][2]
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
            hard_failure = torch.stack(
                [failure_terms[name] for name in hard_failure_names],
                dim=-1,
            ).any(dim=-1)
            procedural_failure = torch.stack(
                [
                    failure_terms[name]
                    for name in procedural_failure_names
                ],
                dim=-1,
            ).any(dim=-1)
            any_failure = hard_failure | procedural_failure
            time_out_term = manager.get_term("time_out")
            first_done = was_unresolved & dones
            first_success = (
                first_done & success_term & ~any_failure
            )
            successful_environment |= first_success
            handover_state_data = getattr(
                env.unwrapped,
                "_dr_anmar_handover_state",
            )
            terminal_pickup_attempts = torch.maximum(
                handover_state_data["pickup_attempt_count"],
                handover_state_data["last_pickup_attempt_count"],
            )
            terminal_pickup_recoveries = torch.maximum(
                handover_state_data["pickup_recovery_count"],
                handover_state_data["last_pickup_recovery_count"],
            )
            first_terminal_pickup_attempts[first_done] = (
                terminal_pickup_attempts[first_done]
            )
            maximum_pickup_attempts_observed = torch.maximum(
                maximum_pickup_attempts_observed,
                terminal_pickup_attempts,
            )
            maximum_pickup_recoveries_observed = torch.maximum(
                maximum_pickup_recoveries_observed,
                terminal_pickup_recoveries,
            )
            first_attempt_successful_environment |= (
                first_success & (terminal_pickup_attempts == 1)
            )
            recovered_successful_environment |= (
                first_success & (terminal_pickup_attempts > 1)
            )
            max_phase = torch.where(
                first_success,
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
            giver_bilateral_now = torch.all(
                giver_forces > 0.01,
                dim=-1,
            )
            receiver_bilateral_now = torch.all(
                receiver_forces > 0.01,
                dim=-1,
            )
            ever_receiver_jaw_1_contact |= (
                was_unresolved & (receiver_forces[:, 0] > 0.01)
            )
            ever_receiver_jaw_2_contact |= (
                was_unresolved & (receiver_forces[:, 1] > 0.01)
            )
            ever_receiver_bilateral_contact |= (
                was_unresolved & receiver_bilateral_now
            )
            ever_four_jaw_overlap_contact |= (
                was_unresolved
                & giver_bilateral_now
                & receiver_bilateral_now
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
            state_clearance = handover_state_data["clearance"]
            ever_state_lifted |= was_unresolved & handover_state_data["lifted"]
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
                    & ~any_failure[start:stop]
                ).sum()
                hard_failures[group_index] += (
                    first & hard_failure[start:stop]
                ).sum()
                procedural_failures[group_index] += (
                    first & procedural_failure[start:stop]
                ).sum()
                for name, term in failure_terms.items():
                    failure_term_counts[name][group_index] += (
                        first & term[start:stop]
                    ).sum()
                receiver_retention_failure = (
                    first
                    & failure_terms["receiver_retention_lost"][start:stop]
                )
                for cause, state_name in (
                    (
                        "low_clearance",
                        "last_retention_failure_low_clearance",
                    ),
                    (
                        "receiver_follow_error",
                        "last_retention_failure_follow_error",
                    ),
                    (
                        "receiver_contact_loss",
                        "last_retention_failure_contact_loss",
                    ),
                ):
                    retention_failure_causes[cause][group_index] += (
                        receiver_retention_failure
                        & handover_state_data[state_name][start:stop]
                    ).sum()
                timeouts[group_index] += (
                    first
                    & time_out_term[start:stop]
                    & ~success_term[start:stop]
                    & ~any_failure[start:stop]
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
                    maximum_state_clearance[group_index] = torch.maximum(
                        maximum_state_clearance[group_index],
                        state_clearance[start:stop][active].max().double(),
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
                    maximum_four_jaw_overlap_contact[group_index] = torch.maximum(
                        maximum_four_jaw_overlap_contact[group_index],
                        torch.cat(
                            (
                                giver_forces[start:stop][active],
                                receiver_forces[start:stop][active],
                            ),
                            dim=-1,
                        )
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
                    "giver_grasp_arc_fraction": (
                        giver_arc_fractions[group_index]
                    ),
                    "giver_grasp_offset_m": list(
                        giver_offsets[group_index]
                    ),
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
                    "presentation_fraction_from_giver": (
                        presentation_fractions[group_index]
                    ),
                    "pickup_vertical_action_limit": (
                        pickup_vertical_action_limits[group_index]
                    ),
                    "carry_lateral_action_limit": (
                        carry_lateral_action_limits[group_index]
                    ),
                    "carry_vertical_action_limit": (
                        carry_vertical_action_limits[group_index]
                    ),
                    "receiver_close_distance_m": (
                        receiver_close_distances[group_index]
                    ),
                    "receiver_contact_centering_action_limit": (
                        receiver_contact_centering_action_limits[
                            group_index
                        ]
                    ),
                    "giver_transport_min_contact_jaws": (
                        giver_transport_min_contact_jaws[group_index]
                    ),
                    "giver_transport_normalized_contact_threshold": (
                        giver_transport_normalized_contact_thresholds[
                            group_index
                        ]
                    ),
                    "giver_contact_recovery_action_limit": (
                        giver_contact_recovery_action_limits[
                            group_index
                        ]
                    ),
                    "receiver_grasp_offset_m": list(
                        receiver_offsets[group_index]
                    ),
                    "assigned_environments": group_size,
                    "robot_1_selected_as_giver": int(
                        initial_giver_is_robot_1[start:stop].sum().item()
                    ),
                    "robot_2_selected_as_giver": int(
                        (~initial_giver_is_robot_1[start:stop]).sum().item()
                    ),
                    "completed_episodes": completed_count,
                    "successful_episodes": success_count,
                    "first_attempt_successful_episodes": int(
                        first_attempt_successful_environment[
                            start:stop
                        ].sum().item()
                    ),
                    "recovered_successful_episodes": int(
                        recovered_successful_environment[
                            start:stop
                        ].sum().item()
                    ),
                    "pickup_attempt_histogram": {
                        str(attempt): int(
                            (
                                first_terminal_pickup_attempts[start:stop]
                                == attempt
                            ).sum().item()
                        )
                        for attempt in range(4)
                    },
                    "environments_with_pickup_recovery": int(
                        (
                            maximum_pickup_recoveries_observed[start:stop]
                            > 0
                        ).sum().item()
                    ),
                    "success_rate": (
                        success_count / completed_count
                        if completed_count
                        else None
                    ),
                    "time_outs": int(timeouts[group_index].item()),
                    "hard_failures": int(
                        hard_failures[group_index].item()
                    ),
                    "procedural_failures": int(
                        procedural_failures[group_index].item()
                    ),
                    "hard_failure_term_counts": {
                        name: int(counts[group_index].item())
                        for name, counts in failure_term_counts.items()
                        if name in hard_failure_names
                    },
                    "procedural_failure_term_counts": {
                        name: int(counts[group_index].item())
                        for name, counts in failure_term_counts.items()
                        if name in procedural_failure_names
                    },
                    "receiver_retention_failure_causes": {
                        name: int(counts[group_index].item())
                        for name, counts in retention_failure_causes.items()
                    },
                    "failure_term_counts": {
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
                    "maximum_state_clearance_m": float(
                        maximum_state_clearance[group_index].item()
                    ),
                    "environments_with_state_lifted": int(
                        ever_state_lifted[start:stop].sum().item()
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
                    "maximum_four_jaw_overlap_contact_n": float(
                        maximum_four_jaw_overlap_contact[group_index].item()
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
                    "environments_with_receiver_jaw_1_contact": int(
                        ever_receiver_jaw_1_contact[start:stop].sum().item()
                    ),
                    "environments_with_receiver_jaw_2_contact": int(
                        ever_receiver_jaw_2_contact[start:stop].sum().item()
                    ),
                    "environments_with_receiver_bilateral_contact": int(
                        ever_receiver_bilateral_contact[start:stop].sum().item()
                    ),
                    "environments_with_four_jaw_overlap_contact": int(
                        ever_four_jaw_overlap_contact[start:stop].sum().item()
                    ),
                    "successful_environment_indices": (
                        torch.nonzero(
                            successful_environment[start:stop],
                            as_tuple=False,
                        )
                        .squeeze(-1)
                        .add(start)
                        .tolist()
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
            "checkpoint": None,
            "video_capture": (
                {
                    "environment_index": args.video_env_index,
                    "camera_mode": "focused_environment_neighborhood_oblique",
                    "focus_environment_index": args.video_env_index,
                    "camera_eye_world_m": list(camera_eye),
                    "camera_target_world_m": list(camera_target),
                    "camera_flow": {
                        "axis": "world_x",
                        "easing": "cosine",
                        "travel_fraction_of_grid_span": 0.30,
                        "forward_arc_fraction_of_grid_span": 0.04,
                    },
                    "resolution": [
                        args.video_width,
                        args.video_height,
                    ],
                    "folder": str(
                        Path(
                            args.video_folder
                            or Path(args.output_path).resolve()
                            / "videos"
                        ).resolve()
                    ),
                }
                if args.video
                else None
            ),
            "giver": "closest_tool_tip_selected_per_environment_at_reset",
            "receiver": "other_tool",
            "giver_selection": {
                "rule": "minimum_reset_tool_tip_to_needle_distance",
                "robot_1_selected_count": int(
                    initial_giver_is_robot_1.sum().item()
                ),
                "robot_2_selected_count": int(
                    (~initial_giver_is_robot_1).sum().item()
                ),
                "robot_1_initial_distance_m": {
                    "minimum": float(initial_robot_1_distance.min().item()),
                    "maximum": float(initial_robot_1_distance.max().item()),
                },
                "robot_2_initial_distance_m": {
                    "minimum": float(initial_robot_2_distance.min().item()),
                    "maximum": float(initial_robot_2_distance.max().item()),
                },
            },
            "giver_grasp_arc_fractions": giver_arc_fractions,
            "giver_grasp_offsets_m": [
                list(offset) for offset in giver_offsets
            ],
            "handover_motion_contract": {
                "sequence": [
                    "closest_arm_pickup",
                    "up_to_two_safe_regrasp_relift_recoveries",
                    "giver_move_into_receiver_range",
                    "physical_ownership_transfer",
                ],
                "maximum_pickup_attempts": 3,
                "pickup_recovery_requires_physics_owned_custody_loss": True,
                "pickup_recovery_success_credit": False,
                "presentation_fraction_from_giver": 0.35,
                "presentation_height_in_robot_frame_m": -0.13,
                "presentation_ready_tolerance_m": 0.005,
                "minimum_lift_height_in_robot_frame_m": -0.139,
                "carry_lateral_action_limits": (
                    carry_lateral_action_limits
                ),
                "pickup_vertical_action_limits": (
                    pickup_vertical_action_limits
                ),
                "carry_vertical_action_limits": (
                    carry_vertical_action_limits
                ),
                "giver_carry_starts_after_contact_window": True,
                "giver_transport_min_contact_jaws": (
                    giver_transport_min_contact_jaws
                ),
                "giver_transport_normalized_contact_thresholds": (
                    giver_transport_normalized_contact_thresholds
                ),
                "giver_contact_recovery_action_limits": (
                    giver_contact_recovery_action_limits
                ),
                "receiver_close_distances_m": receiver_close_distances,
                "receiver_contact_centering_action_limits": (
                    receiver_contact_centering_action_limits
                ),
                "receiver_waits_for_presentation": True,
                "receiver_stops_approach_on_first_contact": True,
                "giver_release_waits_for_current_receiver_bilateral": True,
                "giver_release_uses_time_only_settle": False,
                "giver_release_confirmation_steps": int(
                    getattr(
                        env.unwrapped.cfg,
                        "dr_anmar_handover_contract",
                        {},
                    ).get("giver_release_confirmation_steps", 0)
                ),
                "giver_holds_position_until_release": True,
                "receiver_holds_position_after_acquisition": True,
                "receiver_orientation_frozen_after_acquisition": True,
                "release_requires_open_command_and_contact_loss": True,
            },
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
    import math

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import (
        axis_angle_from_quat,
        quat_conjugate,
        quat_mul,
    )
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    checkpoint = None
    if args.analytic_only:
        if args.checkpoint:
            return _fail(
                "analytic-only play must not load a checkpoint"
            )
    else:
        if not args.checkpoint:
            return _fail(
                "play requires --checkpoint unless --analytic-only is set"
            )
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            return _fail(f"checkpoint not found: {checkpoint}")

    env_cfg, agent_cfg = _load_configs(args.task, args.num_envs, args.seed)
    if args.presentation_use_filtered_custody is not None:
        contract = getattr(
            env_cfg,
            "dr_anmar_handover_contract",
            None,
        )
        if contract is None:
            return _fail(
                "filtered presentation custody requires a structured handover task"
            )
        contract["presentation_use_filtered_custody"] = (
            args.presentation_use_filtered_custody
        )
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
    if checkpoint is not None:
        runner.load(
            str(checkpoint),
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
        )
    policy_model = runner.alg.get_policy()
    adaptation_modes = sum(
        (
            bool(args.handover_giver_adaptation),
            bool(args.pickup_recovery_adaptation),
            bool(args.recovery_receiver_grasp_retain_adaptation),
            bool(args.joint_transfer_acquisition_adaptation),
            bool(args.transfer_refinement_adaptation),
            bool(args.deadline_recovery_adaptation),
        )
    )
    if adaptation_modes > 1:
        env.close()
        return _fail(
            "giver, pickup-recovery, recovery-conditioned receiver, joint "
            "transfer-acquisition, transfer-refinement, and deadline-recovery "
            "modes are mutually exclusive"
        )
    if args.deadline_recovery_adaptation:
        configure_deadline_recovery = getattr(
            policy_model,
            "configure_deadline_recovery_adaptation",
            None,
        )
        if configure_deadline_recovery is None:
            env.close()
            return _fail(
                "loaded policy does not support deadline-aware recovery"
            )
        configure_deadline_recovery()
    elif args.transfer_refinement_adaptation:
        configure_refinement = getattr(
            policy_model,
            "configure_transfer_refinement_adaptation",
            None,
        )
        if configure_refinement is None:
            env.close()
            return _fail(
                "loaded policy does not support transfer refinement"
            )
        configure_refinement()
    elif args.joint_transfer_acquisition_adaptation:
        configure_joint_transfer = getattr(
            policy_model,
            "configure_joint_transfer_acquisition_adaptation",
            None,
        )
        if configure_joint_transfer is None:
            env.close()
            return _fail(
                "loaded policy does not support joint "
                "transfer-acquisition adaptation"
            )
        configure_joint_transfer()
    elif args.recovery_receiver_grasp_retain_adaptation:
        configure_recovery_receiver = getattr(
            policy_model,
            "configure_recovery_receiver_grasp_retain_adaptation",
            None,
        )
        if configure_recovery_receiver is None:
            env.close()
            return _fail(
                "loaded policy does not support recovery-conditioned "
                "receiver grasp-retain adaptation"
            )
        configure_recovery_receiver()
    elif args.pickup_recovery_adaptation:
        if not hasattr(
            policy_model,
            "configure_pickup_recovery_adaptation",
        ):
            env.close()
            return _fail(
                "loaded policy does not support pickup-recovery adaptation"
            )
        policy_model.configure_pickup_recovery_adaptation()
    if args.handover_giver_adaptation:
        if not hasattr(policy_model, "configure_giver_adaptation"):
            env.close()
            return _fail(
                "loaded policy does not support giver adaptation"
            )
        policy_model.configure_giver_adaptation()
    controller = getattr(policy_model, "controller", None)
    controller_boolean_overrides = (
        (
            "transport_custody_latch_enabled",
            args.transport_custody_latch,
        ),
        (
            "receiver_preposition_enabled",
            args.receiver_preposition,
        ),
        (
            "receiver_adaptive_arc_enabled",
            args.receiver_adaptive_arc,
        ),
        (
            "receiver_grasp_retain_residual_enabled_for_learning",
            args.receiver_grasp_retain_residual,
        ),
    )
    for attribute, value in controller_boolean_overrides:
        if value is None:
            continue
        if controller is None or not hasattr(controller, attribute):
            env.close()
            return _fail(
                f"loaded policy does not expose {attribute}"
            )
        setattr(controller, attribute, value)
    if args.receiver_preposition_height is not None:
        if not 0.01 <= args.receiver_preposition_height <= 0.05:
            env.close()
            return _fail(
                "receiver preposition height must be in [0.01, 0.05] m"
            )
        if controller is None or not hasattr(
            controller,
            "receiver_preposition_height",
        ):
            env.close()
            return _fail(
                "loaded policy does not expose receiver preposition height"
            )
        controller.receiver_preposition_height = (
            args.receiver_preposition_height
        )
    if args.recovery_receiver_preposition_height is not None:
        if not 0.01 <= args.recovery_receiver_preposition_height <= 0.05:
            env.close()
            return _fail(
                "recovery receiver preposition height must be "
                "in [0.01, 0.05] m"
            )
        if controller is None or not hasattr(
            controller,
            "recovery_receiver_preposition_height",
        ):
            env.close()
            return _fail(
                "loaded policy does not expose recovery receiver "
                "preposition height"
            )
        controller.recovery_receiver_preposition_height = (
            args.recovery_receiver_preposition_height
        )
    if args.analytic_only:
        if not hasattr(policy_model, "residual_scale"):
            env.close()
            return _fail(
                "analytic-only play requires an analytic residual policy"
            )
        policy_model.residual_scale = 0.0
    if args.residual_scale is not None:
        if args.residual_scale < 0.0:
            env.close()
            return _fail("play residual scale must be non-negative")
        if not hasattr(policy_model, "residual_scale"):
            env.close()
            return _fail(
                "loaded policy does not expose a residual scale"
            )
        policy_model.residual_scale = args.residual_scale
    if args.pickup_vertical_action_limit is not None:
        if not 0.0 < args.pickup_vertical_action_limit <= 0.3:
            env.close()
            return _fail(
                "play pickup vertical action limit must be in (0.0, 0.3]"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "pickup_vertical_action_limit"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose a pickup vertical action limit"
            )
        controller.pickup_vertical_action_limit = (
            args.pickup_vertical_action_limit
        )
    if args.pickup_initial_vertical_action_limit is not None:
        if not 0.0 < args.pickup_initial_vertical_action_limit <= 0.3:
            env.close()
            return _fail(
                "play initial pickup vertical action limit must be in (0.0, 0.3]"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "pickup_initial_vertical_action_limit"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose an initial pickup vertical action limit"
            )
        controller.pickup_initial_vertical_action_limit = (
            args.pickup_initial_vertical_action_limit
        )
    if args.recovery_pickup_vertical_action_limit is not None:
        if not 0.0 < args.recovery_pickup_vertical_action_limit <= 0.3:
            env.close()
            return _fail(
                "play recovery pickup vertical action limit must be in (0.0, 0.3]"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "recovery_pickup_vertical_action_limit"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose a recovery pickup vertical action limit"
            )
        controller.recovery_pickup_vertical_action_limit = (
            args.recovery_pickup_vertical_action_limit
        )
    if args.carry_lateral_action_limit is not None:
        if not 0.0 < args.carry_lateral_action_limit <= 0.1:
            env.close()
            return _fail(
                "play carry lateral action limit must be in (0.0, 0.1]"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "carry_lateral_action_limit"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose a carry lateral action limit"
            )
        controller.carry_lateral_action_limit = (
            args.carry_lateral_action_limit
        )
    if args.recovery_carry_lateral_action_limit is not None:
        if not 0.0 < args.recovery_carry_lateral_action_limit <= 0.1:
            env.close()
            return _fail(
                "play recovery carry lateral action limit "
                "must be in (0.0, 0.1]"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "recovery_carry_lateral_action_limit"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose a recovery carry "
                "lateral action limit"
            )
        controller.recovery_carry_lateral_action_limit = (
            args.recovery_carry_lateral_action_limit
        )
    if args.carry_lateral_ramp_height is not None:
        if not 0.001 <= args.carry_lateral_ramp_height <= 0.02:
            env.close()
            return _fail(
                "play carry lateral ramp height must be in [0.001, 0.02]"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "carry_lateral_ramp_height"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose a carry lateral ramp height"
            )
        controller.carry_lateral_ramp_height = (
            args.carry_lateral_ramp_height
        )
    if args.presentation_fraction_from_giver is not None:
        if not 0.1 <= args.presentation_fraction_from_giver <= 0.6:
            env.close()
            return _fail(
                "play presentation fraction from giver must be in [0.1, 0.6]"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "presentation_fraction_from_giver"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose a giver presentation fraction"
            )
        controller.presentation_fraction_from_giver = (
            args.presentation_fraction_from_giver
        )
    if args.receiver_crossing_angle_rad is not None:
        if not 0.0 <= args.receiver_crossing_angle_rad <= math.pi:
            env.close()
            return _fail(
                "play receiver crossing angle must be in [0.0, pi]"
            )
        controller = getattr(policy_model, "controller", None)
        if (
            controller is None
            or not hasattr(controller, "receiver_tangent_delta_rad")
            or not hasattr(controller, "receiver_crossing_angle_rad")
            or not hasattr(controller, "receiver_roll_offset_rad")
        ):
            env.close()
            return _fail(
                "loaded policy does not expose receiver crossing-angle control"
            )
        controller.receiver_crossing_angle_rad = (
            args.receiver_crossing_angle_rad
        )
        controller.receiver_roll_offset_rad = (
            controller.receiver_tangent_delta_rad
            + controller.receiver_crossing_angle_rad
        )
    if args.presentation_height_in_robot_frame is not None:
        if not -0.139 < args.presentation_height_in_robot_frame <= -0.12:
            env.close()
            return _fail(
                "play presentation height in robot frame must be in (-0.139, -0.12]"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "presentation_height_in_robot_frame"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose a giver presentation height"
            )
        controller.presentation_height_in_robot_frame = (
            args.presentation_height_in_robot_frame
        )
    if args.giver_close_distance is not None:
        if not 0.0005 <= args.giver_close_distance <= 0.02:
            env.close()
            return _fail(
                "play giver close distance must be in [0.0005, 0.02]"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(controller, "close_distance"):
            env.close()
            return _fail(
                "loaded policy does not expose a giver close distance"
            )
        controller.close_distance = args.giver_close_distance
    if args.giver_lift_contact_force_threshold is not None:
        if not 0.01 <= args.giver_lift_contact_force_threshold <= 0.1:
            env.close()
            return _fail(
                "play giver lift contact force threshold must be in [0.01, 0.1] N"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "giver_lift_contact_force_threshold_n"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose a giver lift contact force threshold"
            )
        controller.giver_lift_contact_force_threshold_n = (
            args.giver_lift_contact_force_threshold
        )
    if args.giver_pre_lift_min_contact_jaws is not None:
        if args.giver_pre_lift_min_contact_jaws not in {1, 2}:
            env.close()
            return _fail(
                "play giver pre-lift minimum contact jaws must be 1 or 2"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "giver_pre_lift_min_contact_jaws"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose giver pre-lift contact continuity"
            )
        controller.giver_pre_lift_min_contact_jaws = (
            args.giver_pre_lift_min_contact_jaws
        )
    if args.giver_transport_orientation_action_limit is not None:
        if not 0.0 <= args.giver_transport_orientation_action_limit <= 0.2:
            env.close()
            return _fail(
                "play giver transport orientation action limit must be in [0.0, 0.2]"
            )
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "giver_transport_orientation_action_limit"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose giver transport orientation authority"
            )
        controller.giver_transport_orientation_action_limit = (
            args.giver_transport_orientation_action_limit
        )
    if args.giver_lift_on_live_contact is not None:
        controller = getattr(policy_model, "controller", None)
        if controller is None or not hasattr(
            controller, "giver_lift_on_live_contact"
        ):
            env.close()
            return _fail(
                "loaded policy does not expose live-contact giver lift"
            )
        controller.giver_lift_on_live_contact = (
            args.giver_lift_on_live_contact
        )
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
    initial_state_population_sha256 = (
        _initial_state_population_sha256(env)
    )
    first_handover_max_phase = None
    first_handover_history = None
    if "Handover-" in args.task:
        from orbit.surgical.tasks.surgical.lift.grasp_frames import (
            NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
            needle_geometry_grasp_offset_m,
        )

        receiver_grasp_offset = needle_geometry_grasp_offset_m(0.65)
        receiver_roll_offset_rad = float(
            getattr(
                getattr(policy_model, "controller", None),
                "receiver_roll_offset_rad",
                math.pi,
            )
        )
        handover_observation = obs["policy"]
        initial_handover_state = getattr(
            env.unwrapped,
            "_dr_anmar_handover_state",
        )
        first_handover_max_phase = initial_handover_state[
            "progress_phase"
        ].clone()
        giver_is_robot_1 = handover_observation[:, 82] > 0.5
        initial_object_in_giver = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            handover_observation[:, 46:49],
            handover_observation[:, 53:56],
        ).clone()
        first_handover_history = {
            "giver_is_robot_1": giver_is_robot_1.clone(),
            "initial_object_in_giver": initial_object_in_giver,
            "ever_giver_bilateral_contact": torch.zeros_like(
                first_unresolved
            ),
            "ever_windowed_giver_contact": torch.zeros_like(
                first_unresolved
            ),
            "giver_bilateral_contact_steps": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "current_giver_bilateral_contact_steps": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "maximum_giver_bilateral_contact_steps": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "ever_midair_giver_contact_loss": torch.zeros_like(
                first_unresolved
            ),
            "current_midair_giver_contact_loss_steps": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "maximum_midair_giver_contact_loss_steps": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "first_giver_bilateral_contact_frame": torch.full(
                (env.unwrapped.num_envs,),
                -1,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "first_windowed_giver_contact_frame": torch.full(
                (env.unwrapped.num_envs,),
                -1,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "giver_orientation_at_first_window": torch.full(
                (env.unwrapped.num_envs, 4),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "object_orientation_at_first_window": torch.full(
                (env.unwrapped.num_envs, 4),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "giver_grasp_error_at_first_window_m": torch.full(
                (env.unwrapped.num_envs, 3),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "giver_jaw_aperture_at_first_window_rad": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "first_lift_frame": torch.full(
                (env.unwrapped.num_envs,),
                -1,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "minimum_giver_contact_force_at_first_lift_n": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "maximum_giver_contact_force_at_first_lift_n": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "giver_contact_force_imbalance_at_first_lift_n": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "giver_jaw_aperture_at_first_lift_rad": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "giver_grasp_error_at_first_lift_m": torch.full(
                (env.unwrapped.num_envs, 3),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "object_orientation_at_first_lift": torch.full(
                (env.unwrapped.num_envs, 4),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "object_linear_speed_at_first_lift_m_s": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "object_angular_speed_at_first_lift_rad_s": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "maximum_clearance_m": torch.full(
                (env.unwrapped.num_envs,),
                -torch.inf,
                device=env.unwrapped.device,
            ),
            "maximum_pickup_attempts": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "maximum_pickup_recoveries": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "ever_presentation_ready": torch.zeros_like(
                first_unresolved
            ),
            "ever_presentation_stable": torch.zeros_like(
                first_unresolved
            ),
            "first_stable_presentation_frame": torch.full(
                (env.unwrapped.num_envs,),
                -1,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "first_receiver_contact_after_stable_frame": torch.full(
                (env.unwrapped.num_envs,),
                -1,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "remaining_steps_at_first_stable": torch.full(
                (env.unwrapped.num_envs,),
                -1,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "remaining_steps_at_first_receiver_contact": torch.full(
                (env.unwrapped.num_envs,),
                -1,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "maximum_receiver_initial_approach_steps": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "receiver_grasp_error_at_first_stable_m": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "receiver_orientation_error_at_first_stable_rad": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "object_yaw_in_receiver_at_first_stable_rad": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "receiver_grasp_error_at_first_contact_m": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "receiver_orientation_error_at_first_contact_rad": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "object_yaw_in_receiver_at_first_contact_rad": torch.full(
                (env.unwrapped.num_envs,),
                torch.nan,
                device=env.unwrapped.device,
            ),
            "maximum_presentation_stable_steps": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "minimum_presentation_error_m": torch.full(
                (env.unwrapped.num_envs,),
                torch.inf,
                device=env.unwrapped.device,
            ),
            "ever_receiver_any_contact": torch.zeros_like(
                first_unresolved
            ),
            "ever_receiver_jaw_1_contact": torch.zeros_like(
                first_unresolved
            ),
            "ever_receiver_jaw_2_contact": torch.zeros_like(
                first_unresolved
            ),
            "ever_receiver_windowed_bilateral_contact": torch.zeros_like(
                first_unresolved
            ),
            "ever_receiver_capture_follows": torch.zeros_like(
                first_unresolved
            ),
            "ever_receiver_close_command": torch.zeros_like(
                first_unresolved
            ),
            "ever_receiver_capture_motion_inside": torch.zeros_like(
                first_unresolved
            ),
            "maximum_receiver_jaw_1_contact_n": torch.zeros(
                env.unwrapped.num_envs,
                device=env.unwrapped.device,
            ),
            "maximum_receiver_jaw_2_contact_n": torch.zeros(
                env.unwrapped.num_envs,
                device=env.unwrapped.device,
            ),
            "minimum_receiver_grasp_error_m": torch.full(
                (env.unwrapped.num_envs,),
                torch.inf,
                device=env.unwrapped.device,
            ),
            "maximum_receiver_capture_steps": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "maximum_giver_release_confirmation_steps": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "ever_giver_release_authorized": torch.zeros_like(
                first_unresolved
            ),
            "maximum_receiver_retries": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "maximum_receiver_release_aborts": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "deadline_option_step_counts": torch.zeros(
                3,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "ever_deadline_reseat": torch.zeros_like(first_unresolved),
            "ever_deadline_backoff": torch.zeros_like(first_unresolved),
            "terminal_pickup_attempts": torch.zeros(
                env.unwrapped.num_envs,
                dtype=torch.int64,
                device=env.unwrapped.device,
            ),
            "terminal_time_out": torch.zeros_like(first_unresolved),
        }
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
                if first_handover_max_phase is not None:
                    first_stable_presentation = torch.zeros_like(
                        was_first_unresolved
                    )
                    first_receiver_contact_after_stable = torch.zeros_like(
                        was_first_unresolved
                    )
                    current_handover_state = getattr(
                        env.unwrapped,
                        "_dr_anmar_handover_state",
                    )
                    current_handover_phase = current_handover_state[
                        "progress_phase"
                    ]
                    first_handover_max_phase = torch.where(
                        was_first_unresolved,
                        torch.maximum(
                            first_handover_max_phase,
                            current_handover_phase,
                        ),
                        first_handover_max_phase,
                    )
                    assert first_handover_history is not None
                    first_handover_history[
                        "maximum_pickup_attempts"
                    ] = torch.maximum(
                        first_handover_history[
                            "maximum_pickup_attempts"
                        ],
                        current_handover_state["pickup_attempt_count"],
                    )
                    first_handover_history[
                        "maximum_pickup_recoveries"
                    ] = torch.maximum(
                        first_handover_history[
                            "maximum_pickup_recoveries"
                        ],
                        current_handover_state["pickup_recovery_count"],
                    )
                    if "presentation_ready_now" in current_handover_state:
                        first_handover_history[
                            "ever_presentation_ready"
                        ] |= (
                            was_first_unresolved
                            & current_handover_state[
                                "presentation_ready_now"
                            ]
                        )
                        first_handover_history[
                            "ever_presentation_stable"
                        ] |= (
                            was_first_unresolved
                            & current_handover_state[
                                "presentation_stable"
                            ]
                        )
                        first_stable_presentation = (
                            was_first_unresolved
                            & current_handover_state[
                                "presentation_stable"
                            ]
                            & (
                                first_handover_history[
                                    "first_stable_presentation_frame"
                                ]
                                < 0
                            )
                        )
                        first_handover_history[
                            "first_stable_presentation_frame"
                        ][first_stable_presentation] = frame_index
                        first_handover_history[
                            "remaining_steps_at_first_stable"
                        ][first_stable_presentation] = (
                            env.unwrapped.max_episode_length
                            - env.unwrapped.episode_length_buf[
                                first_stable_presentation
                            ]
                        )
                        first_receiver_contact_after_stable = (
                            was_first_unresolved
                            & current_handover_state[
                                "presentation_stable"
                            ]
                            & current_handover_state[
                                "receiver_any_contact_now"
                            ]
                            & (
                                first_handover_history[
                                    "first_receiver_contact_after_stable_frame"
                                ]
                                < 0
                            )
                        )
                        first_handover_history[
                            "first_receiver_contact_after_stable_frame"
                        ][first_receiver_contact_after_stable] = frame_index
                        first_handover_history[
                            "remaining_steps_at_first_receiver_contact"
                        ][first_receiver_contact_after_stable] = (
                            env.unwrapped.max_episode_length
                            - env.unwrapped.episode_length_buf[
                                first_receiver_contact_after_stable
                            ]
                        )
                        first_handover_history[
                            "maximum_receiver_initial_approach_steps"
                        ] = torch.maximum(
                            first_handover_history[
                                "maximum_receiver_initial_approach_steps"
                            ],
                            current_handover_state[
                                "receiver_approach_step_count"
                            ],
                        )
                        first_handover_history[
                            "maximum_presentation_stable_steps"
                        ] = torch.maximum(
                            first_handover_history[
                                "maximum_presentation_stable_steps"
                            ],
                            current_handover_state[
                                "presentation_stable_consecutive"
                            ],
                        )
                        first_handover_history[
                            "minimum_presentation_error_m"
                        ] = torch.where(
                            was_first_unresolved,
                            torch.minimum(
                                first_handover_history[
                                    "minimum_presentation_error_m"
                                ],
                                current_handover_state[
                                    "presentation_error"
                                ],
                            ),
                            first_handover_history[
                                "minimum_presentation_error_m"
                            ],
                        )
                        first_handover_history[
                            "ever_receiver_any_contact"
                        ] |= (
                            was_first_unresolved
                            & current_handover_state[
                                "receiver_any_contact_now"
                            ]
                        )
                        first_handover_history[
                            "ever_receiver_windowed_bilateral_contact"
                        ] |= (
                            was_first_unresolved
                            & current_handover_state[
                                "receiver_contact"
                            ]
                        )
                        first_handover_history[
                            "ever_receiver_capture_follows"
                        ] |= (
                            was_first_unresolved
                            & current_handover_state[
                                "receiver_capture_follows"
                            ]
                        )
                        receiver_close_action = torch.where(
                            current_handover_state[
                                "giver_is_robot_1"
                            ],
                            env.unwrapped.action_manager.action[:, 13],
                            env.unwrapped.action_manager.action[:, 6],
                        )
                        first_handover_history[
                            "ever_receiver_close_command"
                        ] |= (
                            was_first_unresolved
                            & (receiver_close_action < 0.0)
                        )
                        first_handover_history[
                            "ever_receiver_capture_motion_inside"
                        ] |= (
                            was_first_unresolved
                            & (
                                current_handover_state["motion"][:, 0]
                                <= 0.05
                            )
                            & (
                                current_handover_state["motion"][:, 1]
                                <= 5.0
                            )
                        )
                        first_handover_history[
                            "maximum_receiver_capture_steps"
                        ] = torch.maximum(
                            first_handover_history[
                                "maximum_receiver_capture_steps"
                            ],
                            current_handover_state[
                                "receiver_capture_consecutive"
                            ],
                        )
                        first_handover_history[
                            "maximum_giver_release_confirmation_steps"
                        ] = torch.maximum(
                            first_handover_history[
                                "maximum_giver_release_confirmation_steps"
                            ],
                            current_handover_state[
                                "giver_release_confirmation_consecutive"
                            ],
                        )
                        first_handover_history[
                            "ever_giver_release_authorized"
                        ] |= (
                            was_first_unresolved
                            & current_handover_state[
                                "giver_release_authorized"
                            ]
                        )
                        first_handover_history[
                            "maximum_receiver_retries"
                        ] = torch.maximum(
                            first_handover_history[
                                "maximum_receiver_retries"
                            ],
                            current_handover_state[
                                "receiver_retry_count"
                            ],
                        )
                        first_handover_history[
                            "maximum_receiver_release_aborts"
                        ] = torch.maximum(
                            first_handover_history[
                                "maximum_receiver_release_aborts"
                            ],
                            current_handover_state[
                                "receiver_release_abort_count"
                            ],
                        )
                    handover_observation = obs["policy"]
                    giver_is_robot_1 = handover_observation[:, 82] > 0.5
                    giver_contacts = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 66:68],
                        handover_observation[:, 68:70],
                    )
                    receiver_contacts = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 68:70],
                        handover_observation[:, 66:68],
                    )
                    giver_bilateral_contact = torch.all(
                        giver_contacts > 0.002,
                        dim=-1,
                    )
                    object_in_giver = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 46:49],
                        handover_observation[:, 53:56],
                    )
                    object_in_receiver = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 53:56],
                        handover_observation[:, 46:49],
                    )
                    object_orientation_in_giver = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 49:53],
                        handover_observation[:, 56:60],
                    )
                    object_orientation_in_receiver = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 56:60],
                        handover_observation[:, 49:53],
                    )
                    object_yaw_in_receiver = torch.atan2(
                        2.0
                        * (
                            object_orientation_in_receiver[:, 3]
                            * object_orientation_in_receiver[:, 2]
                            + object_orientation_in_receiver[:, 0]
                            * object_orientation_in_receiver[:, 1]
                        ),
                        1.0
                        - 2.0
                        * (
                            object_orientation_in_receiver[:, 1].square()
                            + object_orientation_in_receiver[:, 2].square()
                        ),
                    )
                    giver_ee = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 32:35],
                        handover_observation[:, 39:42],
                    )
                    receiver_ee = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 39:42],
                        handover_observation[:, 32:35],
                    )
                    giver_orientation = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 35:39],
                        handover_observation[:, 42:46],
                    )
                    receiver_orientation = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 42:46],
                        handover_observation[:, 35:39],
                    )
                    receiver_roll = torch.zeros_like(giver_orientation)
                    receiver_roll[:, 2] = math.sin(
                        0.5 * receiver_roll_offset_rad
                    )
                    receiver_roll[:, 3] = math.cos(
                        0.5 * receiver_roll_offset_rad
                    )
                    receiver_target_orientation = quat_mul(
                        receiver_roll,
                        giver_orientation,
                    )
                    receiver_orientation_error = torch.linalg.vector_norm(
                        axis_angle_from_quat(
                            quat_mul(
                                receiver_target_orientation,
                                quat_conjugate(receiver_orientation),
                            )
                        ),
                        dim=-1,
                    )
                    receiver_grasp_position = object_in_receiver.clone()
                    receiver_grasp_position[:, 0] += receiver_grasp_offset[0]
                    receiver_grasp_position[:, 1] += receiver_grasp_offset[1]
                    receiver_grasp_position[:, 2] += -0.003
                    receiver_grasp_error = torch.linalg.vector_norm(
                        receiver_grasp_position - receiver_ee,
                        dim=-1,
                    )
                    first_handover_history[
                        "receiver_grasp_error_at_first_stable_m"
                    ][first_stable_presentation] = receiver_grasp_error[
                        first_stable_presentation
                    ]
                    first_handover_history[
                        "receiver_orientation_error_at_first_stable_rad"
                    ][first_stable_presentation] = receiver_orientation_error[
                        first_stable_presentation
                    ]
                    first_handover_history[
                        "object_yaw_in_receiver_at_first_stable_rad"
                    ][first_stable_presentation] = object_yaw_in_receiver[
                        first_stable_presentation
                    ]
                    first_handover_history[
                        "receiver_grasp_error_at_first_contact_m"
                    ][
                        first_receiver_contact_after_stable
                    ] = receiver_grasp_error[
                        first_receiver_contact_after_stable
                    ]
                    first_handover_history[
                        "receiver_orientation_error_at_first_contact_rad"
                    ][
                        first_receiver_contact_after_stable
                    ] = receiver_orientation_error[
                        first_receiver_contact_after_stable
                    ]
                    first_handover_history[
                        "object_yaw_in_receiver_at_first_contact_rad"
                    ][
                        first_receiver_contact_after_stable
                    ] = object_yaw_in_receiver[
                        first_receiver_contact_after_stable
                    ]
                    first_handover_history[
                        "minimum_receiver_grasp_error_m"
                    ] = torch.where(
                        was_first_unresolved,
                        torch.minimum(
                            first_handover_history[
                                "minimum_receiver_grasp_error_m"
                            ],
                            receiver_grasp_error,
                        ),
                        first_handover_history[
                            "minimum_receiver_grasp_error_m"
                        ],
                    )
                    receiver_jaw_contact = receiver_contacts > 0.002
                    first_handover_history[
                        "ever_receiver_jaw_1_contact"
                    ] |= (
                        was_first_unresolved
                        & receiver_jaw_contact[:, 0]
                    )
                    first_handover_history[
                        "ever_receiver_jaw_2_contact"
                    ] |= (
                        was_first_unresolved
                        & receiver_jaw_contact[:, 1]
                    )
                    first_handover_history[
                        "maximum_receiver_jaw_1_contact_n"
                    ] = torch.where(
                        was_first_unresolved,
                        torch.maximum(
                            first_handover_history[
                                "maximum_receiver_jaw_1_contact_n"
                            ],
                            receiver_contacts[:, 0] / 0.2,
                        ),
                        first_handover_history[
                            "maximum_receiver_jaw_1_contact_n"
                        ],
                    )
                    first_handover_history[
                        "maximum_receiver_jaw_2_contact_n"
                    ] = torch.where(
                        was_first_unresolved,
                        torch.maximum(
                            first_handover_history[
                                "maximum_receiver_jaw_2_contact_n"
                            ],
                            receiver_contacts[:, 1] / 0.2,
                        ),
                        first_handover_history[
                            "maximum_receiver_jaw_2_contact_n"
                        ],
                    )
                    giver_orientation = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 35:39],
                        handover_observation[:, 42:46],
                    )
                    giver_joint_position = torch.where(
                        giver_is_robot_1.unsqueeze(-1),
                        handover_observation[:, 0:8],
                        handover_observation[:, 16:24],
                    )
                    clearance = (
                        object_in_giver[:, 2]
                        - first_handover_history[
                            "initial_object_in_giver"
                        ][:, 2]
                    )
                    windowed_contact = current_handover_phase >= 1
                    first_contact = (
                        was_first_unresolved
                        & giver_bilateral_contact
                        & (
                            first_handover_history[
                                "first_giver_bilateral_contact_frame"
                            ]
                            < 0
                        )
                    )
                    first_window = (
                        was_first_unresolved
                        & windowed_contact
                        & (
                            first_handover_history[
                                "first_windowed_giver_contact_frame"
                            ]
                            < 0
                        )
                    )
                    first_handover_history[
                        "first_giver_bilateral_contact_frame"
                    ][first_contact] = frame_index
                    first_handover_history[
                        "first_windowed_giver_contact_frame"
                    ][first_window] = frame_index
                    giver_grasp_position = object_in_giver.clone()
                    giver_grasp_position[:, 0] += (
                        NEEDLE_PROVISIONAL_GRASP_OFFSET_M[0]
                    )
                    giver_grasp_position[:, 1] += (
                        NEEDLE_PROVISIONAL_GRASP_OFFSET_M[1]
                    )
                    giver_grasp_position[:, 2] += (
                        NEEDLE_PROVISIONAL_GRASP_OFFSET_M[2]
                    )
                    first_handover_history[
                        "giver_orientation_at_first_window"
                    ][first_window] = giver_orientation[first_window]
                    first_handover_history[
                        "object_orientation_at_first_window"
                    ][first_window] = object_orientation_in_giver[first_window]
                    first_handover_history[
                        "giver_grasp_error_at_first_window_m"
                    ][first_window] = (
                        giver_grasp_position[first_window]
                        - giver_ee[first_window]
                    )
                    first_handover_history[
                        "giver_jaw_aperture_at_first_window_rad"
                    ][first_window] = (
                        1.0
                        + giver_joint_position[first_window, 7]
                        - giver_joint_position[first_window, 6]
                    )
                    first_lift = (
                        was_first_unresolved
                        & (current_handover_phase >= 2)
                        & (
                            first_handover_history[
                                "first_lift_frame"
                            ]
                            < 0
                        )
                    )
                    first_handover_history["first_lift_frame"][
                        first_lift
                    ] = frame_index
                    first_handover_history[
                        "minimum_giver_contact_force_at_first_lift_n"
                    ][first_lift] = (
                        torch.min(
                            giver_contacts[first_lift],
                            dim=-1,
                        ).values
                        / 0.2
                    )
                    first_handover_history[
                        "maximum_giver_contact_force_at_first_lift_n"
                    ][first_lift] = (
                        torch.max(
                            giver_contacts[first_lift],
                            dim=-1,
                        ).values
                        / 0.2
                    )
                    first_handover_history[
                        "giver_contact_force_imbalance_at_first_lift_n"
                    ][first_lift] = (
                        torch.abs(
                            giver_contacts[first_lift, 1]
                            - giver_contacts[first_lift, 0]
                        )
                        / 0.2
                    )
                    first_handover_history[
                        "giver_jaw_aperture_at_first_lift_rad"
                    ][first_lift] = (
                        1.0
                        + giver_joint_position[first_lift, 7]
                        - giver_joint_position[first_lift, 6]
                    )
                    first_handover_history[
                        "giver_grasp_error_at_first_lift_m"
                    ][first_lift] = (
                        giver_grasp_position[first_lift]
                        - giver_ee[first_lift]
                    )
                    first_handover_history[
                        "object_orientation_at_first_lift"
                    ][first_lift] = object_orientation_in_giver[first_lift]
                    object_linear_velocity = handover_observation[:, 60:63]
                    object_angular_velocity = handover_observation[:, 63:66]
                    first_handover_history[
                        "object_linear_speed_at_first_lift_m_s"
                    ][first_lift] = torch.linalg.vector_norm(
                        object_linear_velocity[first_lift],
                        dim=-1,
                    )
                    first_handover_history[
                        "object_angular_speed_at_first_lift_rad_s"
                    ][first_lift] = torch.linalg.vector_norm(
                        object_angular_velocity[first_lift],
                        dim=-1,
                    )
                    first_handover_history[
                        "ever_giver_bilateral_contact"
                    ] |= was_first_unresolved & giver_bilateral_contact
                    first_handover_history[
                        "ever_windowed_giver_contact"
                    ] |= was_first_unresolved & windowed_contact
                    first_handover_history[
                        "giver_bilateral_contact_steps"
                    ] += (
                        was_first_unresolved & giver_bilateral_contact
                    ).to(torch.int64)
                    current_contact_steps = torch.where(
                        was_first_unresolved & giver_bilateral_contact,
                        first_handover_history[
                            "current_giver_bilateral_contact_steps"
                        ]
                        + 1,
                        0,
                    )
                    first_handover_history[
                        "current_giver_bilateral_contact_steps"
                    ] = current_contact_steps
                    first_handover_history[
                        "maximum_giver_bilateral_contact_steps"
                    ] = torch.maximum(
                        first_handover_history[
                            "maximum_giver_bilateral_contact_steps"
                        ],
                        current_contact_steps,
                    )
                    midair_before_receiver = (
                        was_first_unresolved
                        & (current_handover_phase >= 2)
                        & (current_handover_phase < 3)
                        & (clearance >= 0.01)
                    )
                    midair_giver_contact_loss = (
                        midair_before_receiver
                        & ~giver_bilateral_contact
                    )
                    current_midair_loss_steps = torch.where(
                        midair_giver_contact_loss,
                        first_handover_history[
                            "current_midair_giver_contact_loss_steps"
                        ]
                        + 1,
                        0,
                    )
                    first_handover_history[
                        "current_midair_giver_contact_loss_steps"
                    ] = current_midair_loss_steps
                    first_handover_history[
                        "maximum_midair_giver_contact_loss_steps"
                    ] = torch.maximum(
                        first_handover_history[
                            "maximum_midair_giver_contact_loss_steps"
                        ],
                        current_midair_loss_steps,
                    )
                    first_handover_history[
                        "ever_midair_giver_contact_loss"
                    ] |= midair_giver_contact_loss
                    first_handover_history["maximum_clearance_m"] = (
                        torch.where(
                            was_first_unresolved,
                            torch.maximum(
                                first_handover_history[
                                    "maximum_clearance_m"
                                ],
                                clearance,
                            ),
                            first_handover_history[
                                "maximum_clearance_m"
                            ],
                        )
                    )
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
                if first_handover_history is not None:
                    option_index = getattr(
                        policy_model,
                        "last_deadline_option_index",
                        None,
                    )
                    option_active = getattr(
                        policy_model,
                        "last_deadline_option_active",
                        None,
                    )
                    if option_index is not None and option_active is not None:
                        counted_option = (
                            was_first_unresolved & option_active
                        )
                        for option in range(3):
                            first_handover_history[
                                "deadline_option_step_counts"
                            ][option] += (
                                counted_option & (option_index == option)
                            ).sum()
                        first_handover_history[
                            "ever_deadline_reseat"
                        ] |= counted_option & (option_index == 1)
                        first_handover_history[
                            "ever_deadline_backoff"
                        ] |= counted_option & (option_index == 2)
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
                if first_handover_max_phase is not None:
                    assert first_handover_history is not None
                    post_handover_state = getattr(
                        env.unwrapped,
                        "_dr_anmar_handover_state",
                    )
                    terminal_pickup_attempts = torch.maximum(
                        post_handover_state["pickup_attempt_count"],
                        post_handover_state[
                            "last_pickup_attempt_count"
                        ],
                    )
                    terminal_pickup_recoveries = torch.maximum(
                        post_handover_state["pickup_recovery_count"],
                        post_handover_state[
                            "last_pickup_recovery_count"
                        ],
                    )
                    first_handover_history[
                        "terminal_pickup_attempts"
                    ][first_dones] = terminal_pickup_attempts[first_dones]
                    first_handover_history[
                        "terminal_time_out"
                    ][first_dones] = term_values["time_out"][
                        first_dones
                    ].bool()
                    first_handover_history[
                        "maximum_pickup_attempts"
                    ] = torch.maximum(
                        first_handover_history[
                            "maximum_pickup_attempts"
                        ],
                        terminal_pickup_attempts,
                    )
                    first_handover_history[
                        "maximum_pickup_recoveries"
                    ] = torch.maximum(
                        first_handover_history[
                            "maximum_pickup_recoveries"
                        ],
                        terminal_pickup_recoveries,
                    )
                    first_handover_max_phase = torch.where(
                        first_successes,
                        torch.full_like(first_handover_max_phase, 4),
                        first_handover_max_phase,
                    )
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
        first_episode_handover_diagnostics = None
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
        if first_handover_max_phase is not None:
            assert first_handover_history is not None
            first_completed = ~first_unresolved
            phase_labels = (
                "no_giver_bilateral_contact",
                "giver_contact_without_10mm_lift",
                "lifted_without_receiver_acquisition",
                "receiver_acquired_without_retained_success",
                "success",
            )
            def handover_cohort_stats(mask: torch.Tensor) -> dict:
                count = int(mask.sum().item())
                if not count:
                    return {"count": 0}
                initial_xy = first_handover_history[
                    "initial_object_in_giver"
                ][mask, :2]
                first_contact_frame = first_handover_history[
                    "first_giver_bilateral_contact_frame"
                ][mask]
                contacted = first_contact_frame >= 0
                first_window_frame = first_handover_history[
                    "first_windowed_giver_contact_frame"
                ][mask]
                windowed = first_window_frame >= 0
                giver_orientation = first_handover_history[
                    "giver_orientation_at_first_window"
                ][mask][windowed]
                object_orientation = first_handover_history[
                    "object_orientation_at_first_window"
                ][mask][windowed]
                giver_grasp_error = first_handover_history[
                    "giver_grasp_error_at_first_window_m"
                ][mask][windowed]
                giver_jaw_aperture = first_handover_history[
                    "giver_jaw_aperture_at_first_window_rad"
                ][mask][windowed]
                first_lift_frame = first_handover_history[
                    "first_lift_frame"
                ][mask]
                lifted = first_lift_frame >= 0
                minimum_lift_contact_force = first_handover_history[
                    "minimum_giver_contact_force_at_first_lift_n"
                ][mask][lifted]
                maximum_lift_contact_force = first_handover_history[
                    "maximum_giver_contact_force_at_first_lift_n"
                ][mask][lifted]
                lift_contact_force_imbalance = first_handover_history[
                    "giver_contact_force_imbalance_at_first_lift_n"
                ][mask][lifted]
                giver_lift_jaw_aperture = first_handover_history[
                    "giver_jaw_aperture_at_first_lift_rad"
                ][mask][lifted]
                giver_lift_grasp_error = first_handover_history[
                    "giver_grasp_error_at_first_lift_m"
                ][mask][lifted]
                object_lift_orientation = first_handover_history[
                    "object_orientation_at_first_lift"
                ][mask][lifted]
                object_lift_linear_speed = first_handover_history[
                    "object_linear_speed_at_first_lift_m_s"
                ][mask][lifted]
                object_lift_angular_speed = first_handover_history[
                    "object_angular_speed_at_first_lift_rad_s"
                ][mask][lifted]

                def scalar_quantiles(values: torch.Tensor) -> dict | None:
                    if not bool(values.numel()):
                        return None
                    quantiles = torch.quantile(
                        values.float(),
                        torch.tensor(
                            [0.1, 0.5, 0.9],
                            device=values.device,
                        ),
                    )
                    return {
                        "p10": float(quantiles[0].item()),
                        "p50": float(quantiles[1].item()),
                        "p90": float(quantiles[2].item()),
                    }

                return {
                    "count": count,
                    "ever_giver_bilateral_contact": int(
                        first_handover_history[
                            "ever_giver_bilateral_contact"
                        ][mask]
                        .sum()
                        .item()
                    ),
                    "ever_windowed_giver_contact": int(
                        first_handover_history[
                            "ever_windowed_giver_contact"
                        ][mask]
                        .sum()
                        .item()
                    ),
                    "mean_initial_object_xy_in_giver_m": [
                        float(initial_xy[:, axis].mean().item())
                        for axis in range(2)
                    ],
                    "mean_initial_object_radial_offset_m": float(
                        torch.linalg.vector_norm(initial_xy, dim=-1)
                        .mean()
                        .item()
                    ),
                    "mean_giver_bilateral_contact_steps": float(
                        first_handover_history[
                            "giver_bilateral_contact_steps"
                        ][mask]
                        .float()
                        .mean()
                        .item()
                    ),
                    "mean_maximum_consecutive_bilateral_contact_steps": float(
                        first_handover_history[
                            "maximum_giver_bilateral_contact_steps"
                        ][mask]
                        .float()
                        .mean()
                        .item()
                    ),
                    "mean_first_bilateral_contact_frame": (
                        float(first_contact_frame[contacted].float().mean().item())
                        if bool(contacted.any().item())
                        else None
                    ),
                    "mean_first_windowed_contact_frame": (
                        float(first_window_frame[windowed].float().mean().item())
                        if bool(windowed.any().item())
                        else None
                    ),
                    "mean_giver_orientation_at_first_window_xyzw": (
                        [
                            float(
                                giver_orientation[:, axis]
                                .mean()
                                .item()
                            )
                            for axis in range(4)
                        ]
                        if bool(windowed.any().item())
                        else None
                    ),
                    "mean_object_orientation_at_first_window_xyzw": (
                        [
                            float(
                                object_orientation[:, axis]
                                .mean()
                                .item()
                            )
                            for axis in range(4)
                        ]
                        if bool(windowed.any().item())
                        else None
                    ),
                    "mean_giver_grasp_error_at_first_window_m": (
                        [
                            float(
                                giver_grasp_error[:, axis]
                                .mean()
                                .item()
                            )
                            for axis in range(3)
                        ]
                        if bool(windowed.any().item())
                        else None
                    ),
                    "mean_giver_jaw_aperture_at_first_window_rad": (
                        float(giver_jaw_aperture.mean().item())
                        if bool(windowed.any().item())
                        else None
                    ),
                    "mean_first_lift_frame": (
                        float(first_lift_frame[lifted].float().mean().item())
                        if bool(lifted.any().item())
                        else None
                    ),
                    "minimum_giver_contact_force_at_first_lift_n": (
                        scalar_quantiles(minimum_lift_contact_force)
                    ),
                    "maximum_giver_contact_force_at_first_lift_n": (
                        scalar_quantiles(maximum_lift_contact_force)
                    ),
                    "giver_contact_force_imbalance_at_first_lift_n": (
                        scalar_quantiles(lift_contact_force_imbalance)
                    ),
                    "giver_jaw_aperture_at_first_lift_rad": (
                        scalar_quantiles(giver_lift_jaw_aperture)
                    ),
                    "mean_giver_grasp_error_at_first_lift_m": (
                        [
                            float(
                                giver_lift_grasp_error[:, axis]
                                .mean()
                                .item()
                            )
                            for axis in range(3)
                        ]
                        if bool(lifted.any().item())
                        else None
                    ),
                    "mean_object_orientation_at_first_lift_xyzw": (
                        [
                            float(
                                object_lift_orientation[:, axis]
                                .mean()
                                .item()
                            )
                            for axis in range(4)
                        ]
                        if bool(lifted.any().item())
                        else None
                    ),
                    "object_linear_speed_at_first_lift_m_s": (
                        scalar_quantiles(object_lift_linear_speed)
                    ),
                    "object_angular_speed_at_first_lift_rad_s": (
                        scalar_quantiles(object_lift_angular_speed)
                    ),
                    "mean_maximum_clearance_m": float(
                        first_handover_history["maximum_clearance_m"][mask]
                        .mean()
                        .item()
                    ),
                    "environments_with_recovery": int(
                        (
                            first_handover_history[
                                "maximum_pickup_recoveries"
                            ][mask]
                            > 0
                        )
                        .sum()
                        .item()
                    ),
                    "terminal_pickup_attempt_histogram": {
                        str(attempt): int(
                            (
                                first_handover_history[
                                    "terminal_pickup_attempts"
                                ][mask]
                                == attempt
                            )
                            .sum()
                            .item()
                        )
                        for attempt in range(4)
                    },
                }

            phase_masks = {
                label: first_completed
                & (first_handover_max_phase == phase)
                for phase, label in enumerate(phase_labels)
            }

            def handover_retention_stats(mask: torch.Tensor) -> dict:
                count = int(mask.sum().item())
                if not count:
                    return {"count": 0}
                maximum_loss_steps = first_handover_history[
                    "maximum_midair_giver_contact_loss_steps"
                ][mask]
                return {
                    "count": count,
                    "episodes_with_any_midair_bilateral_contact_loss": int(
                        (
                            first_handover_history[
                                "ever_midair_giver_contact_loss"
                            ][mask]
                        )
                        .sum()
                        .item()
                    ),
                    "episodes_with_sustained_midair_loss_3_steps": int(
                        (maximum_loss_steps >= 3).sum().item()
                    ),
                    "maximum_loss_steps_p50": float(
                        torch.quantile(
                            maximum_loss_steps.float(),
                            0.5,
                        ).item()
                    ),
                    "maximum_loss_steps_p90": float(
                        torch.quantile(
                            maximum_loss_steps.float(),
                            0.9,
                        ).item()
                    ),
                    "maximum_loss_steps_max": int(
                        maximum_loss_steps.max().item()
                    ),
                }

            def handover_scalar_quantiles(
                values: torch.Tensor,
            ) -> dict | None:
                if not bool(values.numel()):
                    return None
                quantiles = torch.quantile(
                    values.float(),
                    torch.tensor(
                        [0.1, 0.5, 0.9, 0.99],
                        device=values.device,
                    ),
                )
                return {
                    "p10": float(quantiles[0].item()),
                    "p50": float(quantiles[1].item()),
                    "p90": float(quantiles[2].item()),
                    "p99": float(quantiles[3].item()),
                    "max": float(values.max().item()),
                }

            def receiver_approach_stats(mask: torch.Tensor) -> dict:
                stable = (
                    mask
                    & (
                        first_handover_history[
                            "first_stable_presentation_frame"
                        ]
                        >= 0
                    )
                )
                contacted = (
                    stable
                    & (
                        first_handover_history[
                            "first_receiver_contact_after_stable_frame"
                        ]
                        >= 0
                    )
                )
                stable_frames = first_handover_history[
                    "first_stable_presentation_frame"
                ][stable]
                contact_frames = first_handover_history[
                    "first_receiver_contact_after_stable_frame"
                ][contacted]
                latencies = (
                    first_handover_history[
                        "first_receiver_contact_after_stable_frame"
                    ][contacted]
                    - first_handover_history[
                        "first_stable_presentation_frame"
                    ][contacted]
                )
                stable_grasp_errors = first_handover_history[
                    "receiver_grasp_error_at_first_stable_m"
                ][stable]
                stable_orientation_errors = first_handover_history[
                    "receiver_orientation_error_at_first_stable_rad"
                ][stable]
                contact_grasp_errors = first_handover_history[
                    "receiver_grasp_error_at_first_contact_m"
                ][contacted]
                contact_orientation_errors = first_handover_history[
                    "receiver_orientation_error_at_first_contact_rad"
                ][contacted]
                stable_object_yaw = first_handover_history[
                    "object_yaw_in_receiver_at_first_stable_rad"
                ][stable]
                contact_object_yaw = first_handover_history[
                    "object_yaw_in_receiver_at_first_contact_rad"
                ][contacted]
                remaining_at_stable = first_handover_history[
                    "remaining_steps_at_first_stable"
                ][stable]
                remaining_at_contact = first_handover_history[
                    "remaining_steps_at_first_receiver_contact"
                ][contacted]
                maximum_initial_approach_steps = first_handover_history[
                    "maximum_receiver_initial_approach_steps"
                ][mask]
                return {
                    "count": int(mask.sum().item()),
                    "stable_presentations": int(stable.sum().item()),
                    "contacts_after_stable_presentation": int(
                        contacted.sum().item()
                    ),
                    "first_stable_presentation_frame": (
                        handover_scalar_quantiles(stable_frames)
                    ),
                    "first_receiver_contact_frame": (
                        handover_scalar_quantiles(contact_frames)
                    ),
                    "stable_to_receiver_contact_steps": (
                        handover_scalar_quantiles(latencies)
                    ),
                    "remaining_steps_at_first_stable": (
                        handover_scalar_quantiles(remaining_at_stable)
                    ),
                    "remaining_steps_at_first_receiver_contact": (
                        handover_scalar_quantiles(remaining_at_contact)
                    ),
                    "maximum_receiver_initial_approach_steps": (
                        handover_scalar_quantiles(
                            maximum_initial_approach_steps
                        )
                    ),
                    "receiver_grasp_error_at_first_stable_m": (
                        handover_scalar_quantiles(stable_grasp_errors)
                    ),
                    "receiver_orientation_error_at_first_stable_rad": (
                        handover_scalar_quantiles(
                            stable_orientation_errors
                        )
                    ),
                    "receiver_grasp_error_at_first_contact_m": (
                        handover_scalar_quantiles(contact_grasp_errors)
                    ),
                    "receiver_orientation_error_at_first_contact_rad": (
                        handover_scalar_quantiles(
                            contact_orientation_errors
                        )
                    ),
                    "object_yaw_in_receiver_at_first_stable_rad": (
                        handover_scalar_quantiles(stable_object_yaw)
                    ),
                    "object_yaw_in_receiver_at_first_contact_rad": (
                        handover_scalar_quantiles(contact_object_yaw)
                    ),
                }

            first_episode_handover_diagnostics = {
                "pickup_attempt_outcomes": {
                    "maximum_attempts": 3,
                    "first_attempt_successful_episodes": int(
                        (
                            first_outcome_success
                            & (
                                first_handover_history[
                                    "terminal_pickup_attempts"
                                ]
                                == 1
                            )
                        ).sum().item()
                    ),
                    "recovered_successful_episodes": int(
                        (
                            first_outcome_success
                            & (
                                first_handover_history[
                                    "terminal_pickup_attempts"
                                ]
                                > 1
                            )
                        ).sum().item()
                    ),
                    "environments_with_recovery": int(
                        (
                            first_handover_history[
                                "maximum_pickup_recoveries"
                            ]
                            > 0
                        ).sum().item()
                    ),
                    "terminal_attempt_histogram": {
                        str(attempt): int(
                            (
                                first_handover_history[
                                    "terminal_pickup_attempts"
                                ]
                                == attempt
                            ).sum().item()
                        )
                        for attempt in range(4)
                    },
                },
                "maximum_phase_distribution": {
                    label: int(mask.sum().item())
                    for label, mask in phase_masks.items()
                },
                "outcomes_by_giver_role": {
                    role: {
                        "count": int(role_mask.sum().item()),
                        "success": int(
                            (role_mask & first_outcome_success)
                            .sum()
                            .item()
                        ),
                        "reached_10mm_lift": int(
                            (
                                role_mask
                                & (first_handover_max_phase >= 2)
                            )
                            .sum()
                            .item()
                        ),
                        "reached_stable_presentation": int(
                            (
                                role_mask
                                & first_handover_history[
                                    "ever_presentation_stable"
                                ]
                            )
                            .sum()
                            .item()
                        ),
                        "reached_receiver_contact": int(
                            (
                                role_mask
                                & first_handover_history[
                                    "ever_receiver_any_contact"
                                ]
                            )
                            .sum()
                            .item()
                        ),
                        "environments_with_recovery": int(
                            (
                                role_mask
                                & (
                                    first_handover_history[
                                        "maximum_pickup_recoveries"
                                    ]
                                    > 0
                                )
                            )
                            .sum()
                            .item()
                        ),
                        "recovered_success": int(
                            (
                                role_mask
                                & first_outcome_success
                                & (
                                    first_handover_history[
                                        "terminal_pickup_attempts"
                                    ]
                                    > 1
                                )
                            )
                            .sum()
                            .item()
                        ),
                    }
                    for role, role_mask in {
                        "robot_1": (
                            first_completed
                            & first_handover_history[
                                "giver_is_robot_1"
                            ]
                        ),
                        "robot_2": (
                            first_completed
                            & ~first_handover_history[
                                "giver_is_robot_1"
                            ]
                        ),
                    }.items()
                },
                "reached_phase_fraction": {
                    f"phase_{phase}_{label}": float(
                        (
                            first_handover_max_phase >= phase
                        )
                        .float()
                        .mean()
                        .item()
                    )
                    for phase, label in enumerate(phase_labels)
                },
                "pickup_causal_diagnostics": {
                    "contact_threshold_n": 0.01,
                    "contact_window": "three_of_five_control_steps",
                    "overall": handover_cohort_stats(first_completed),
                    "by_maximum_phase": {
                        label: handover_cohort_stats(mask)
                        for label, mask in phase_masks.items()
                    },
                },
                "transport_retention_diagnostics": {
                    "minimum_clearance_m": 0.01,
                    "sustained_loss_steps": 3,
                    "physical_signal": (
                        "native_giver_bilateral_contact_loss_"
                        "after_10mm_lift_before_receiver_acquisition"
                    ),
                    "overall": handover_retention_stats(first_completed),
                    "by_maximum_phase": {
                        label: handover_retention_stats(mask)
                        for label, mask in phase_masks.items()
                    },
                    "timeouts_after_any_midair_contact_loss": int(
                        (
                            first_handover_history[
                                "terminal_time_out"
                            ]
                            & first_handover_history[
                                "ever_midair_giver_contact_loss"
                            ]
                        )
                        .sum()
                        .item()
                    ),
                    "successes_after_any_midair_contact_loss": int(
                        (
                            first_outcome_success
                            & first_handover_history[
                                "ever_midair_giver_contact_loss"
                            ]
                        )
                        .sum()
                        .item()
                    ),
                },
                "structured_transfer_diagnostics": {
                    "receiver_approach_by_maximum_phase": {
                        label: receiver_approach_stats(mask)
                        for label, mask in phase_masks.items()
                    },
                    "environments_reaching_instant_presentation": int(
                        first_handover_history[
                            "ever_presentation_ready"
                        ][first_completed]
                        .sum()
                        .item()
                    ),
                    "environments_reaching_stable_presentation": int(
                        first_handover_history[
                            "ever_presentation_stable"
                        ][first_completed]
                        .sum()
                        .item()
                    ),
                    "maximum_presentation_stable_steps": int(
                        first_handover_history[
                            "maximum_presentation_stable_steps"
                        ][first_completed]
                        .max()
                        .item()
                    )
                    if bool(first_completed.any().item())
                    else 0,
                    "minimum_presentation_error_m": float(
                        first_handover_history[
                            "minimum_presentation_error_m"
                        ][first_completed]
                        .min()
                        .item()
                    )
                    if bool(first_completed.any().item())
                    else None,
                    "environments_with_receiver_contact": int(
                        first_handover_history[
                            "ever_receiver_any_contact"
                        ][first_completed]
                        .sum()
                        .item()
                    ),
                    "environments_with_receiver_jaw_1_contact": int(
                        first_handover_history[
                            "ever_receiver_jaw_1_contact"
                        ][first_completed]
                        .sum()
                        .item()
                    ),
                    "environments_with_receiver_jaw_2_contact": int(
                        first_handover_history[
                            "ever_receiver_jaw_2_contact"
                        ][first_completed]
                        .sum()
                        .item()
                    ),
                    "environments_with_windowed_bilateral_contact": int(
                        first_handover_history[
                            "ever_receiver_windowed_bilateral_contact"
                        ][first_completed]
                        .sum()
                        .item()
                    ),
                    "environments_with_capture_follow": int(
                        first_handover_history[
                            "ever_receiver_capture_follows"
                        ][first_completed]
                        .sum()
                        .item()
                    ),
                    "environments_with_receiver_close_command": int(
                        first_handover_history[
                            "ever_receiver_close_command"
                        ][first_completed]
                        .sum()
                        .item()
                    ),
                    "environments_with_capture_motion_inside": int(
                        first_handover_history[
                            "ever_receiver_capture_motion_inside"
                        ][first_completed]
                        .sum()
                        .item()
                    ),
                    "maximum_receiver_jaw_1_contact_n": (
                        handover_scalar_quantiles(
                            first_handover_history[
                                "maximum_receiver_jaw_1_contact_n"
                            ][first_completed]
                        )
                    ),
                    "maximum_receiver_jaw_2_contact_n": (
                        handover_scalar_quantiles(
                            first_handover_history[
                                "maximum_receiver_jaw_2_contact_n"
                            ][first_completed]
                        )
                    ),
                    "minimum_receiver_grasp_error_m": (
                        handover_scalar_quantiles(
                            first_handover_history[
                                "minimum_receiver_grasp_error_m"
                            ][first_completed]
                        )
                    ),
                    "maximum_receiver_capture_steps": int(
                        first_handover_history[
                            "maximum_receiver_capture_steps"
                        ][first_completed]
                        .max()
                        .item()
                    )
                    if bool(first_completed.any().item())
                    else 0,
                    "maximum_giver_release_confirmation_steps": int(
                        first_handover_history[
                            "maximum_giver_release_confirmation_steps"
                        ][first_completed]
                        .max()
                        .item()
                    )
                    if bool(first_completed.any().item())
                    else 0,
                    "environments_with_giver_release_authorized": int(
                        first_handover_history[
                            "ever_giver_release_authorized"
                        ][first_completed]
                        .sum()
                        .item()
                    ),
                    "environments_with_receiver_retry": int(
                        (
                            first_handover_history[
                                "maximum_receiver_retries"
                            ][first_completed]
                            > 0
                        )
                        .sum()
                        .item()
                    ),
                    "maximum_receiver_retries": int(
                        first_handover_history[
                            "maximum_receiver_retries"
                        ][first_completed]
                        .max()
                        .item()
                    )
                    if bool(first_completed.any().item())
                    else 0,
                    "receiver_release_aborts": int(
                        first_handover_history[
                            "maximum_receiver_release_aborts"
                        ][first_completed]
                        .sum()
                        .item()
                    ),
                    "deadline_option_step_counts": {
                        label: int(
                            first_handover_history[
                                "deadline_option_step_counts"
                            ][index].item()
                        )
                        for index, label in enumerate(
                            ("continue", "reseat", "backoff")
                        )
                    },
                    "environments_with_deadline_reseat": int(
                        first_handover_history[
                            "ever_deadline_reseat"
                        ][first_completed].sum().item()
                    ),
                    "environments_with_deadline_backoff": int(
                        first_handover_history[
                            "ever_deadline_backoff"
                        ][first_completed].sum().item()
                    ),
                },
                "unresolved": int(first_unresolved.sum().item()),
            }
        first_completed_count = int((~first_unresolved).sum().item())
        first_success_count = int(first_outcome_success.sum().item())
        policy_runtime_contract = _policy_runtime_contract(
            policy_model,
            args.task,
        )
        environment_runtime_contract = _environment_runtime_contract(
            env_cfg,
            args.task,
        )
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
            "initial_state_population_sha256": (
                initial_state_population_sha256
            ),
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
            "first_episode_handover_diagnostics": (
                first_episode_handover_diagnostics
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
            }
            if checkpoint is not None
            else None,
            "policy_runtime_contract": policy_runtime_contract,
            "policy_runtime_contract_sha256": _canonical_sha256(
                policy_runtime_contract
            ),
            "environment_runtime_contract": (
                environment_runtime_contract
            ),
            "environment_runtime_contract_sha256": _canonical_sha256(
                environment_runtime_contract
            ),
            "analytic_only": bool(args.analytic_only),
            "policy_giver_adaptation_enabled": bool(
                getattr(
                    policy_model,
                    "giver_adaptation_enabled",
                    args.handover_giver_adaptation,
                )
            ),
            "policy_pickup_recovery_adaptation_enabled": bool(
                getattr(
                    policy_model,
                    "pickup_recovery_adaptation_enabled",
                    args.pickup_recovery_adaptation,
                )
            ),
            "policy_recovery_receiver_grasp_retain_adaptation_enabled": bool(
                getattr(
                    policy_model,
                    "recovery_receiver_grasp_retain_adaptation_enabled",
                    args.recovery_receiver_grasp_retain_adaptation,
                )
            ),
            "policy_joint_transfer_acquisition_adaptation_enabled": bool(
                getattr(
                    policy_model,
                    "joint_transfer_acquisition_adaptation_enabled",
                    args.joint_transfer_acquisition_adaptation,
                )
            ),
            "policy_transfer_refinement_adaptation_enabled": bool(
                getattr(
                    policy_model,
                    "transfer_refinement_adaptation_enabled",
                    args.transfer_refinement_adaptation,
                )
            ),
            "policy_deadline_recovery_adaptation_enabled": bool(
                getattr(
                    policy_model,
                    "deadline_recovery_adaptation_enabled",
                    args.deadline_recovery_adaptation,
                )
            ),
            "policy_residual_scale": (
                float(policy_model.residual_scale)
                if hasattr(policy_model, "residual_scale")
                else None
            ),
            "policy_giver_residual_axes": (
                ["x", "y"]
                if hasattr(policy_model, "controller")
                and hasattr(policy_model, "residual_scale")
                else None
            ),
            "policy_analytic_vertical_authority": (
                True
                if hasattr(policy_model, "controller")
                and hasattr(policy_model, "residual_scale")
                else None
            ),
            "policy_receiver_residual_enabled": (
                bool(
                    policy_model.controller.receiver_residual_enabled_for_learning
                )
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "receiver_residual_enabled_for_learning",
                )
                else None
            ),
            "policy_receiver_grasp_retain_residual_enabled": (
                bool(
                    policy_model.controller.receiver_grasp_retain_residual_enabled_for_learning
                )
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "receiver_grasp_retain_residual_enabled_for_learning",
                )
                else None
            ),
            "policy_transport_custody_latch_enabled": (
                bool(policy_model.controller.transport_custody_latch_enabled)
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "transport_custody_latch_enabled",
                )
                else None
            ),
            "policy_receiver_preposition_enabled": (
                bool(policy_model.controller.receiver_preposition_enabled)
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "receiver_preposition_enabled",
                )
                else None
            ),
            "policy_receiver_preposition_height_m": (
                float(policy_model.controller.receiver_preposition_height)
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "receiver_preposition_height",
                )
                else None
            ),
            "policy_recovery_receiver_preposition_height_m": (
                float(
                    policy_model.controller.recovery_receiver_preposition_height
                )
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "recovery_receiver_preposition_height",
                )
                else None
            ),
            "policy_receiver_contact_orientation_error_target_rad": (
                float(
                    policy_model.controller.receiver_contact_orientation_error_target_rad
                )
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "receiver_contact_orientation_error_target_rad",
                )
                else None
            ),
            "policy_receiver_adaptive_arc_enabled": (
                bool(policy_model.controller.receiver_adaptive_arc_enabled)
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "receiver_adaptive_arc_enabled",
                )
                else None
            ),
            "policy_presentation_use_filtered_custody": bool(
                getattr(
                    env_cfg,
                    "dr_anmar_handover_contract",
                    {},
                ).get(
                    "presentation_use_filtered_custody",
                    False,
                )
            ),
            "policy_pickup_vertical_action_limit": (
                float(policy_model.controller.pickup_vertical_action_limit)
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "pickup_vertical_action_limit",
                )
                else None
            ),
            "policy_pickup_initial_vertical_action_limit": (
                float(
                    policy_model.controller.pickup_initial_vertical_action_limit
                )
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "pickup_initial_vertical_action_limit",
                )
                else None
            ),
            "policy_recovery_pickup_vertical_action_limit": (
                float(
                    policy_model.controller.recovery_pickup_vertical_action_limit
                )
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "recovery_pickup_vertical_action_limit",
                )
                else None
            ),
            "policy_carry_lateral_action_limit": (
                float(policy_model.controller.carry_lateral_action_limit)
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "carry_lateral_action_limit",
                )
                else None
            ),
            "policy_recovery_carry_lateral_action_limit": (
                float(
                    policy_model.controller.recovery_carry_lateral_action_limit
                )
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "recovery_carry_lateral_action_limit",
                )
                else None
            ),
            "policy_carry_lateral_ramp_height": (
                float(policy_model.controller.carry_lateral_ramp_height)
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "carry_lateral_ramp_height",
                )
                else None
            ),
            "policy_presentation_fraction_from_giver": (
                float(
                    policy_model.controller.presentation_fraction_from_giver
                )
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "presentation_fraction_from_giver",
                )
                else None
            ),
            "policy_receiver_tangent_delta_rad": (
                float(policy_model.controller.receiver_tangent_delta_rad)
                if hasattr(
                    getattr(policy_model, "controller", None),
                    "receiver_tangent_delta_rad",
                )
                else None
            ),
            "policy_receiver_crossing_angle_rad": (
                float(policy_model.controller.receiver_crossing_angle_rad)
                if hasattr(
                    getattr(policy_model, "controller", None),
                    "receiver_crossing_angle_rad",
                )
                else None
            ),
            "policy_receiver_roll_offset_rad": (
                float(policy_model.controller.receiver_roll_offset_rad)
                if hasattr(
                    getattr(policy_model, "controller", None),
                    "receiver_roll_offset_rad",
                )
                else None
            ),
            "policy_presentation_height_in_robot_frame": (
                float(
                    policy_model.controller.presentation_height_in_robot_frame
                )
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "presentation_height_in_robot_frame",
                )
                else None
            ),
            "policy_giver_close_distance_m": (
                float(policy_model.controller.close_distance)
                if hasattr(policy_model, "controller")
                and hasattr(policy_model.controller, "close_distance")
                else None
            ),
            "policy_giver_lift_contact_force_threshold_n": (
                float(
                    policy_model.controller.giver_lift_contact_force_threshold_n
                )
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "giver_lift_contact_force_threshold_n",
                )
                else None
            ),
            "policy_giver_pre_lift_min_contact_jaws": (
                int(policy_model.controller.giver_pre_lift_min_contact_jaws)
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "giver_pre_lift_min_contact_jaws",
                )
                else None
            ),
            "policy_giver_transport_orientation_action_limit": (
                float(
                    policy_model.controller.giver_transport_orientation_action_limit
                )
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "giver_transport_orientation_action_limit",
                )
                else None
            ),
            "policy_giver_lift_on_live_contact": (
                bool(policy_model.controller.giver_lift_on_live_contact)
                if hasattr(policy_model, "controller")
                and hasattr(
                    policy_model.controller,
                    "giver_lift_on_live_contact",
                )
                else None
            ),
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
    handover_sweep.add_argument("--video", action="store_true")
    handover_sweep.add_argument("--video_env_index", type=int, default=0)
    handover_sweep.add_argument("--video_width", type=int, default=1280)
    handover_sweep.add_argument("--video_height", type=int, default=720)
    handover_sweep.add_argument("--video_folder")
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
    train.add_argument("--checkpoint")
    train.add_argument("--learning_rate", type=float)
    train.add_argument(
        "--handover_giver_adaptation",
        action="store_true",
    )

    pretrain = subparsers.add_parser("pretrain")
    pretrain.add_argument("--task", required=True)
    pretrain.add_argument("--num_envs", type=int, required=True)
    pretrain.add_argument("--updates", type=int, default=32)
    pretrain.add_argument("--validation_frames", type=int, default=500)
    pretrain.add_argument("--learning_rate", type=float, default=3e-4)
    pretrain.add_argument("--weight_decay", type=float, default=1e-6)
    pretrain.add_argument("--position_scale", type=float, default=0.01)
    pretrain.add_argument("--orientation_scale", type=float, default=0.05)
    pretrain.add_argument("--dagger_warmup_updates", type=int, default=100)
    pretrain.add_argument(
        "--dagger_min_teacher_fraction",
        type=float,
        default=0.1,
    )
    pretrain.add_argument(
        "--e2e_replay_capacity_per_phase",
        type=int,
        default=65_536,
    )
    pretrain.add_argument(
        "--e2e_replay_batch_size",
        type=int,
        default=4096,
    )
    pretrain.add_argument(
        "--e2e_samples_per_phase_step",
        type=int,
        default=64,
    )
    pretrain.add_argument(
        "--e2e_student_segment_steps",
        type=int,
        default=64,
    )
    pretrain.add_argument(
        "--e2e_teacher_recovery_steps",
        type=int,
        default=32,
    )
    pretrain.add_argument(
        "--e2e_consolidation_updates",
        type=int,
        default=2000,
    )
    pretrain.add_argument("--seed", type=int, default=17)
    pretrain.add_argument("--output_path", required=True)
    pretrain.add_argument("--benchmark_formatter", default="schema,json")

    play = subparsers.add_parser("play")
    play.add_argument("--task", required=True)
    play.add_argument("--checkpoint")
    play.add_argument("--analytic-only", action="store_true")
    play.add_argument("--handover_giver_adaptation", action="store_true")
    play.add_argument(
        "--pickup_recovery_adaptation",
        action="store_true",
    )
    play.add_argument(
        "--recovery_receiver_grasp_retain_adaptation",
        action="store_true",
    )
    play.add_argument(
        "--joint_transfer_acquisition_adaptation",
        action="store_true",
    )
    play.add_argument(
        "--transfer_refinement_adaptation",
        action="store_true",
    )
    play.add_argument(
        "--deadline_recovery_adaptation",
        action="store_true",
    )
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
    play.add_argument("--residual_scale", type=float)
    play.add_argument("--pickup_vertical_action_limit", type=float)
    play.add_argument("--pickup_initial_vertical_action_limit", type=float)
    play.add_argument("--recovery_pickup_vertical_action_limit", type=float)
    play.add_argument("--carry_lateral_action_limit", type=float)
    play.add_argument(
        "--recovery_carry_lateral_action_limit",
        type=float,
    )
    play.add_argument("--carry_lateral_ramp_height", type=float)
    play.add_argument("--presentation_fraction_from_giver", type=float)
    play.add_argument("--receiver_crossing_angle_rad", type=float)
    play.add_argument(
        "--transport_custody_latch",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    play.add_argument(
        "--receiver_preposition",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    play.add_argument("--receiver_preposition_height", type=float)
    play.add_argument(
        "--recovery_receiver_preposition_height",
        type=float,
    )
    play.add_argument(
        "--receiver_adaptive_arc",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    play.add_argument(
        "--receiver_grasp_retain_residual",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    play.add_argument(
        "--presentation_use_filtered_custody",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    play.add_argument("--presentation_height_in_robot_frame", type=float)
    play.add_argument("--giver_close_distance", type=float)
    play.add_argument("--giver_lift_contact_force_threshold", type=float)
    play.add_argument("--giver_pre_lift_min_contact_jaws", type=int)
    play.add_argument("--giver_transport_orientation_action_limit", type=float)
    play.add_argument(
        "--giver_lift_on_live_contact",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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
