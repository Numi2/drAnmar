# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Native NVIDIA PSM action and recording adapter.

This module runs inside the pinned Isaac for Healthcare Agentic Arena
environment.  It does not implement inverse kinematics or robot control.
Doctor/state-machine Cartesian commands continue through NVIDIA's
``DifferentialInverseKinematicsAction`` and ``DifferentialIKController``.

The adapter accepts NVIDIA's seven-value relative-IK or eight-value absolute-IK
Cartesian command and records the joint-space command produced by that native
path in the seven-value action contract consumed by NVIDIA's PSM policy mode:

    six normalized joint-position inputs + one canonical binary gripper input

The native recorder also retains the original Cartesian action as
``cartesian_actions`` and the absolute joint targets, so the conversion is
auditable in both directions while Isaac Lab's standard ``actions`` key remains
directly replayable in joint-position mode. The live Cartesian gripper slot may
be proportional, but policy recordings deliberately keep the released binary
contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from dr_anmar_psm_gripper import (
    CANONICAL_PSM_GRIPPER_PROFILE,
    apply_psm_gripper_action_profile,
)

PSM_POLICY_ACTION_DIM = 7
PSM_CARTESIAN_ACTION_DIM = 8
PSM_ARM_DIM = 6
CONTRACT_NAME = "dr_anmar.nvidia_psm_policy_action.v1"
PSM_ARM_NAMES = (
    "psm_yaw_joint",
    "psm_pitch_end_joint",
    "psm_main_insertion_joint",
    "psm_tool_roll_joint",
    "psm_tool_pitch_joint",
    "psm_tool_yaw_joint",
)
_PATCHED = False


class NativePsmControlAdapter:
    """Thin doctor-control facade over NVIDIA's relative- or absolute-IK PSM environment."""

    def __init__(self, env: Any) -> None:
        self.env = env
        base = _base_env(env)
        action_dim = int(base.action_manager.total_action_dim)
        if action_dim not in (PSM_CARTESIAN_ACTION_DIM - 1, PSM_CARTESIAN_ACTION_DIM):
            raise RuntimeError(
                "The doctor PSM adapter requires NVIDIA's seven-value relative-IK or "
                "eight-value absolute-IK action mode; "
                f"the active environment exposes {action_dim} values"
            )
        robot = base.scene["robot"]
        arm_term, _gripper_term = _native_terms_for_robot(base, robot)
        if arm_term.__class__.__name__ != "DifferentialInverseKinematicsAction":
            raise RuntimeError(
                "The doctor PSM adapter requires NVIDIA DifferentialInverseKinematicsAction; "
                f"got {arm_term.__class__.__name__}"
            )
        self.action_dim = action_dim

    def step(self, cartesian_action: Any) -> dict[str, Any]:
        """Apply one doctor command through NVIDIA IK and expose its native result."""

        import torch

        base = _base_env(self.env)
        action = torch.as_tensor(cartesian_action, dtype=torch.float32, device=base.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape != (base.num_envs, self.action_dim):
            raise ValueError(
                f"doctor PSM command has shape {tuple(action.shape)}, "
                f"expected ({base.num_envs}, {self.action_dim})"
            )
        _require_finite(action, "doctor Cartesian actions")
        step_result = self.env.step(action)
        return {
            "step_result": step_result,
            "cartesian_action": action.clone(),
            "policy_action": canonical_policy_actions(self.env),
            "resolved_joint_target": resolved_joint_targets(self.env),
        }


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _psm_arm_names() -> tuple[str, ...]:
    try:
        from common.config import get_robot_config

        names = tuple(get_robot_config("psm").body_joint_names)
    except (ImportError, ModuleNotFoundError):
        names = PSM_ARM_NAMES
    if len(names) != PSM_ARM_DIM:
        raise RuntimeError(f"NVIDIA PSM config exposes {len(names)} arm joints; expected {PSM_ARM_DIM}")
    return names


def _joint_ids(robot: Any, names: Sequence[str]) -> list[int]:
    indices: list[int] = []
    for fallback, name in enumerate(names):
        found = robot.find_joints(name, preserve_order=True)
        candidate = found[0]
        if hasattr(candidate, "numel"):
            if candidate.numel() != 1:
                raise RuntimeError(f"PSM joint {name!r} resolved to {candidate.numel()} entries")
            indices.append(int(candidate[0]))
        elif len(candidate) == 1:
            indices.append(int(candidate[0]))
        else:
            raise RuntimeError(f"PSM joint {name!r} did not resolve uniquely")
    return indices


def _gripper_ids(robot: Any) -> list[int]:
    found = robot.find_joints("psm_tool_gripper.*_joint", preserve_order=True)
    candidate = found[0]
    values = candidate.tolist() if hasattr(candidate, "tolist") else list(candidate)
    if len(values) != 2:
        raise RuntimeError(f"NVIDIA PSM gripper resolved to {len(values)} physical joints; expected 2")
    return [int(value) for value in values]


def _joint_policy_scale() -> float:
    """Read the scale from NVIDIA's own joint-position PSM action config."""

    try:
        from arena.embodiments.psm import PsmEmbodiment

        cfg = PsmEmbodiment(enable_cameras=False, action_device="joint_position").get_action_cfg().arm_action
    except (ImportError, ModuleNotFoundError):
        # NVIDIA PSM v0.7 declares this value in its JointPositionActionCfg.
        return 0.5
    scale = cfg.scale
    if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) == 0.0:
        raise RuntimeError(f"unsupported NVIDIA PSM joint-position scale: {scale!r}")
    if not bool(cfg.use_default_offset):
        raise RuntimeError("NVIDIA PSM joint-position mode no longer uses its default pose as the action offset")
    return float(scale)


def _native_gripper_apertures() -> tuple[float, float]:
    """Return the logical (open, close) apertures from NVIDIA's PSM config."""

    try:
        from arena.embodiments.psm import PsmEmbodiment

        cfg = PsmEmbodiment(enable_cameras=False, action_device="joint_position").get_action_cfg().gripper_action
    except (ImportError, ModuleNotFoundError):
        return (
            CANONICAL_PSM_GRIPPER_PROFILE.open_rad,
            CANONICAL_PSM_GRIPPER_PROFILE.close_rad,
        )
    names = ("psm_tool_gripper1_joint", "psm_tool_gripper2_joint")

    def aperture(command: dict[str, float]) -> float:
        return (float(command[names[1]]) - float(command[names[0]])) * 0.5

    return aperture(cfg.open_command_expr), aperture(cfg.close_command_expr)


def _physical_gripper_targets(base: Any, robot: Any):
    return robot.data.joint_pos_target[:, _gripper_ids(robot)].clone()


def _native_terms_for_robot(base: Any, robot: Any) -> tuple[Any, Any | None]:
    """Find Isaac Lab's native IK and binary-gripper terms for one articulation."""

    arm_term = None
    gripper_term = None
    for term_name in base.action_manager.active_terms:
        term = base.action_manager.get_term(term_name)
        if getattr(term, "_asset", None) is not robot:
            continue
        class_name = term.__class__.__name__
        if class_name == "DifferentialInverseKinematicsAction":
            arm_term = term
        elif class_name in {
            "BinaryJointAction",
            "BinaryJointPositionAction",
            "ProportionalJointPositionAction",
        }:
            gripper_term = term
    if arm_term is None:
        raise RuntimeError("PSM articulation is not controlled by NVIDIA DifferentialInverseKinematicsAction")
    return arm_term, gripper_term


def native_ik_action_scales(env: Any, robot_names: Sequence[str]) -> list[list[float]]:
    """Read the six real per-axis scales from each active NVIDIA IK term."""

    base = _base_env(env)
    result: list[list[float]] = []
    for robot_name in robot_names:
        robot = base.scene[robot_name]
        arm_term, _gripper_term = _native_terms_for_robot(base, robot)
        scale = getattr(arm_term, "_scale", None)
        if scale is None:
            scale = getattr(arm_term.cfg, "scale", None)
        if scale is None:
            raise RuntimeError(f"NVIDIA IK term for {robot_name} does not expose its action scale")
        if hasattr(scale, "detach"):
            values = scale.detach().reshape(-1).cpu().tolist()
        elif isinstance(scale, Sequence) and not isinstance(scale, (str, bytes)):
            values = list(scale)
        else:
            values = [scale]
        if len(values) == 1:
            values *= PSM_ARM_DIM
        if len(values) != PSM_ARM_DIM:
            raise RuntimeError(
                f"NVIDIA IK term for {robot_name} exposes {len(values)} scales; expected {PSM_ARM_DIM}"
            )
        numeric = [abs(float(value)) for value in values]
        if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
            raise RuntimeError(f"NVIDIA IK term for {robot_name} exposes invalid scales: {numeric}")
        result.append(numeric)
    return result


def resolved_joint_targets(env: Any, robot_name: str = "robot"):
    """Return NVIDIA's applied six-joint target plus logical gripper aperture."""

    import torch

    base = _base_env(env)
    robot = base.scene[robot_name]
    arm_ids = _joint_ids(robot, _psm_arm_names())
    arm_target = robot.data.joint_pos_target[:, arm_ids].clone()
    _arm_term, gripper_term = _native_terms_for_robot(base, robot)
    if gripper_term is not None:
        physical = gripper_term.processed_actions
    else:
        # NVIDIA's reach state machine intentionally omits its gripper action.
        # Preserve the jaw target already owned by the articulation.
        physical = _physical_gripper_targets(base, robot)
    if physical.ndim != 2 or physical.shape[-1] != 2:
        raise RuntimeError(
            "NVIDIA PSM gripper action must resolve to two physical jaw targets; "
            f"got shape {tuple(physical.shape)}"
        )
    aperture = (physical[:, 1:2] - physical[:, 0:1]) * 0.5
    targets = torch.cat((arm_target, aperture), dim=-1)
    _require_finite(targets, "resolved PSM joint targets")
    return targets


def canonical_policy_actions(env: Any, robot_name: str = "robot"):
    """Encode NVIDIA's applied targets as its native seven-value policy input."""

    import torch

    base = _base_env(env)
    robot = base.scene[robot_name]
    arm_ids = _joint_ids(robot, _psm_arm_names())
    targets = robot.data.joint_pos_target[:, arm_ids]
    offsets = robot.data.default_joint_pos[:, arm_ids]
    arm_action = (targets - offsets) / _joint_policy_scale()
    _arm_term, gripper_term = _native_terms_for_robot(base, robot)
    if gripper_term is not None:
        raw_gripper = gripper_term.raw_actions
        if raw_gripper.ndim != 2 or raw_gripper.shape[-1] != 1:
            raise RuntimeError(
                "NVIDIA PSM gripper action must expose one logical input; "
                f"got shape {tuple(raw_gripper.shape)}"
            )
        # Keep the released policy channel binary even though the live Cartesian
        # gripper term now preserves proportional aperture.
        gripper_action = torch.where(
            raw_gripper < 0,
            torch.full_like(raw_gripper, -1.0),
            torch.full_like(raw_gripper, 1.0),
        )
    else:
        physical = _physical_gripper_targets(base, robot)
        aperture = (physical[:, 1:2] - physical[:, 0:1]) * 0.5
        open_aperture, close_aperture = _native_gripper_apertures()
        gripper_action = torch.where(
            torch.abs(aperture - close_aperture) < torch.abs(aperture - open_aperture),
            torch.full_like(aperture, -1.0),
            torch.full_like(aperture, 1.0),
        )
    actions = torch.cat((arm_action, gripper_action), dim=-1)
    if actions.shape[-1] != PSM_POLICY_ACTION_DIM:
        raise RuntimeError(f"resolved PSM policy action has shape {tuple(actions.shape)}, expected (*, 7)")
    _require_finite(actions, "canonical PSM policy actions")
    return actions


def canonical_policy_contract(env: Any) -> tuple[Any, Any, tuple[str, ...]]:
    """Return the native 7D-per-PSM policy action and resolved targets.

    Single-arm environments return seven values. Dual-arm environments return
    fourteen values in scene order. This function reads only Isaac Lab action
    terms and articulation targets after ``env.step``; it never writes state.
    """

    import torch

    base = _base_env(env)
    scene_names = tuple(getattr(base.scene, "articulations", {}).keys())
    robot_names = tuple(
        name
        for name in ("robot", "robot_1", "robot_2")
        if name in scene_names
        and all(joint in tuple(base.scene[name].joint_names) for joint in _psm_arm_names())
    )
    if not robot_names:
        raise RuntimeError("The active scene has no NVIDIA-compatible PSM articulation")
    if robot_names == ("robot", "robot_1", "robot_2"):
        raise RuntimeError("Ambiguous PSM scene contains both single- and dual-arm articulation names")
    actions = torch.cat([canonical_policy_actions(env, name) for name in robot_names], dim=-1)
    targets = torch.cat([resolved_joint_targets(env, name) for name in robot_names], dim=-1)
    expected = PSM_POLICY_ACTION_DIM * len(robot_names)
    if actions.shape[-1] != expected or targets.shape[-1] != expected:
        raise RuntimeError(
            f"native PSM contract has action/target shapes {tuple(actions.shape)}/{tuple(targets.shape)}; "
            f"expected (*, {expected})"
        )
    return actions, targets, robot_names


def _require_finite(values: Any, label: str) -> None:
    import torch

    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError(f"{label} contain NaN or infinity")


def _recorder_term_cfgs():
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import RecorderTerm, RecorderTermCfg
    from isaaclab.utils.configclass import configclass

    class CanonicalPolicyActionRecorder(RecorderTerm):
        def record_post_step(self):
            actions, _targets, _robot_names = canonical_policy_contract(self._env)
            return "processed_actions", actions

    class ResolvedJointTargetRecorder(RecorderTerm):
        def record_post_step(self):
            _actions, targets, _robot_names = canonical_policy_contract(self._env)
            return "resolved_joint_targets", targets

    class EndEffectorPoseRecorder(RecorderTerm):
        def record_post_step(self):
            import torch

            scene = _base_env(self._env).scene
            frame_names = tuple(
                name for name in ("ee_frame", "ee_2_frame", "ee_frame_1", "ee_frame_2") if name in scene.sensors
            )
            if not frame_names:
                raise RuntimeError("The PSM scene exposes no end-effector frame sensor")
            poses = []
            for frame_name in frame_names:
                frame = scene[frame_name].data
                poses.append(torch.cat((frame.target_pos_w[:, 0, :], frame.target_quat_w[:, 0, :]), dim=-1))
            return "native_ee_pose_w", torch.cat(poses, dim=-1)

    @configclass
    class CanonicalPolicyActionRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = CanonicalPolicyActionRecorder

    @configclass
    class ResolvedJointTargetRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = ResolvedJointTargetRecorder

    @configclass
    class EndEffectorPoseRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = EndEffectorPoseRecorder

    @configclass
    class NativePsmRecorderManagerCfg(ActionStateRecorderManagerCfg):
        record_post_step_processed_actions = CanonicalPolicyActionRecorderCfg()
        record_post_step_resolved_joint_targets = ResolvedJointTargetRecorderCfg()
        record_post_step_native_ee_pose_w = EndEffectorPoseRecorderCfg()

    return NativePsmRecorderManagerCfg


def install_native_recorder(*, env_id: str, bootstrap_root: Path) -> None:
    """Install deferred PSM recording and jaw-target hooks after Kit starts."""

    global _PATCHED
    if _PATCHED:
        return

    from arena.environments import get_environment

    bootstrap_root.mkdir(parents=True, exist_ok=True)
    env_cls = get_environment(env_id)
    original_build = env_cls.build

    def build_with_native_psm_recorder(self, args):
        _patch_psm_recorder_config(bootstrap_root)
        return original_build(self, args)

    env_cls.build = build_with_native_psm_recorder
    _PATCHED = True


def _patch_psm_recorder_config(bootstrap_root: Path) -> None:
    """Patch the embodiment only after AppLauncher has initialized Omniverse."""

    from arena.embodiments.psm import DualPsmEmbodiment, PsmEmbodiment

    recorder_manager_cfg = _recorder_term_cfgs()

    def set_native_gripper_close(action_cfg):
        apply_psm_gripper_action_profile(action_cfg)
        return action_cfg

    def get_recorder_term_cfg(self):
        cfg = recorder_manager_cfg()
        # Arena constructs Isaac Lab's RecorderManager before setup_recording()
        # applies --record-to.  Give that bootstrap manager a writable, isolated
        # target rather than Isaac Lab's process-global /tmp/isaaclab default.
        cfg.dataset_export_dir_path = str(bootstrap_root)
        cfg.dataset_filename = f"psm-recorder-bootstrap-{os.getpid()}"
        return cfg

    for embodiment_cls in (PsmEmbodiment, DualPsmEmbodiment):
        if not getattr(embodiment_cls, "_dr_anmar_native_gripper_close", False):
            original_get_action_cfg = embodiment_cls.get_action_cfg

            def get_action_cfg(self, _original=original_get_action_cfg):
                return set_native_gripper_close(_original(self))

            embodiment_cls.get_action_cfg = get_action_cfg
            embodiment_cls._dr_anmar_native_gripper_close = True
        embodiment_cls.get_recorder_term_cfg = get_recorder_term_cfg
        embodiment_cls._dr_anmar_native_recorder = True
    _install_recording_close_hook()


def _install_recording_close_hook() -> None:
    """Finalize the HDF5 before SimulationApp.close terminates the process."""

    from arena import recording

    if getattr(recording.close_recording, "_dr_anmar_psm_finalize", False):
        return
    original = recording.close_recording

    def close_and_finalize(ctx):
        original(ctx)
        dataset_path = getattr(ctx, "recording_dataset_path", None)
        if dataset_path:
            report = finalize_hdf5(Path(dataset_path))
            print("[dr-anmar-psm] canonical action contract", flush=True)
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    close_and_finalize._dr_anmar_psm_finalize = True
    recording.close_recording = close_and_finalize

    # The state-machine module imports close_recording directly.  Replace that
    # module-global alias when present so its _Recorder.close uses the hook.
    try:
        from arena.statemachine.core import machine

        machine.close_recording = close_and_finalize
    except ImportError:
        pass


def finalize_hdf5(path: Path) -> dict[str, Any]:
    """Make NVIDIA's converter consume the canonical action and validate it."""

    import h5py
    import numpy as np

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    episode_reports = []
    with h5py.File(path, "r+") as h5_file:
        data = h5_file.get("data")
        if data is None:
            raise RuntimeError(f"{path} has no /data group")

        for name in sorted(data.keys()):
            episode = data[name]
            if not hasattr(episode, "keys"):
                continue
            if "processed_actions" not in episode or "resolved_joint_targets" not in episode:
                raise RuntimeError(f"{name} is missing the native PSM action contract")

            policy_actions = np.asarray(episode["processed_actions"])
            joint_targets = np.asarray(episode["resolved_joint_targets"])
            cartesian_key = "cartesian_actions" if "cartesian_actions" in episode else "actions"
            cartesian_actions = np.asarray(episode[cartesian_key]) if cartesian_key in episode else None
            _validate_episode_arrays(name, policy_actions, joint_targets, cartesian_actions)

            if "cartesian_actions" not in episode and "actions" in episode:
                episode.move("actions", "cartesian_actions")
            elif "actions" in episode:
                del episode["actions"]
            episode["actions"] = episode["processed_actions"]

            obs = episode.get("obs")
            if obs is None:
                raise RuntimeError(f"{name} is missing observations")
            policy_dim = int(policy_actions.shape[-1])
            psm_count = policy_dim // PSM_POLICY_ACTION_DIM
            if "actions" in obs:
                observed_actions = np.asarray(obs["actions"])
                cartesian_dims = _cartesian_action_dims(psm_count)
                if observed_actions.shape == policy_actions.shape and np.array_equal(
                    observed_actions, policy_actions
                ):
                    del obs["actions"]
                elif observed_actions.shape[-1] in cartesian_dims:
                    if "previous_cartesian_actions" in obs:
                        del obs["previous_cartesian_actions"]
                    obs.move("actions", "previous_cartesian_actions")
                elif observed_actions.shape[-1] != policy_dim:
                    raise RuntimeError(
                        f"{name}/obs/actions has {observed_actions.shape[-1]} values; "
                        f"expected {policy_dim} policy or one of {cartesian_dims} Cartesian"
                    )
                else:
                    del obs["actions"]
            obs["actions"] = episode["processed_actions"]

            episode.attrs["dr_anmar_action_contract"] = CONTRACT_NAME
            episode.attrs["dr_anmar_policy_action_dim"] = policy_dim
            episode.attrs["dr_anmar_psm_count"] = psm_count
            episode.attrs["dr_anmar_cartesian_action_dim"] = (
                int(cartesian_actions.shape[-1]) if cartesian_actions is not None else 0
            )
            episode.attrs["dr_anmar_policy_action_semantics"] = (
                "per PSM: six NVIDIA JointPositionAction raw inputs plus one BinaryJointAction sign"
            )
            episode.attrs["dr_anmar_resolved_target_semantics"] = (
                "per PSM: six absolute NVIDIA DifferentialIK joint targets plus logical jaw aperture"
            )
            episode_reports.append(
                {
                    "episode": name,
                    "steps": int(policy_actions.shape[0]),
                    "policy_action_shape": list(policy_actions.shape),
                    "resolved_target_shape": list(joint_targets.shape),
                    "cartesian_action_shape": list(cartesian_actions.shape) if cartesian_actions is not None else None,
                    "max_roundtrip_error": _roundtrip_error(policy_actions, joint_targets),
                }
            )

        if not episode_reports:
            raise RuntimeError(f"{path} contains no recorded episodes")
        h5_file.attrs["dr_anmar_action_contract"] = CONTRACT_NAME
        h5_file.attrs["dr_anmar_action_contract_json"] = json.dumps(
            {
                "schema": CONTRACT_NAME,
                "policy_action_dims": sorted({report["policy_action_shape"][-1] for report in episode_reports}),
                "source": "NVIDIA DifferentialIKController and PSM action configuration",
                "episodes": episode_reports,
            },
            sort_keys=True,
        )
        h5_file.flush()

    return {"path": str(path), "contract": CONTRACT_NAME, "episodes": episode_reports}


def inspect_hdf5(path: Path) -> dict[str, Any]:
    """Validate a finalized file without modifying it."""

    import h5py
    import numpy as np

    path = path.expanduser().resolve()
    reports = []
    with h5py.File(path, "r") as h5_file:
        if h5_file.attrs.get("dr_anmar_action_contract") != CONTRACT_NAME:
            raise RuntimeError(f"{path} is not finalized with {CONTRACT_NAME}")
        for name, episode in sorted(h5_file["data"].items()):
            policy_actions = np.asarray(episode["processed_actions"])
            targets = np.asarray(episode["resolved_joint_targets"])
            cartesian = np.asarray(episode["cartesian_actions"]) if "cartesian_actions" in episode else None
            replay_actions = np.asarray(episode["actions"])
            observed = np.asarray(episode["obs/actions"])
            _validate_episode_arrays(name, policy_actions, targets, cartesian)
            if replay_actions.shape != policy_actions.shape or not np.array_equal(replay_actions, policy_actions):
                raise RuntimeError(f"{name}/actions is not the canonical replayable policy-action dataset")
            if observed.shape != policy_actions.shape or not np.array_equal(observed, policy_actions):
                raise RuntimeError(f"{name}/obs/actions is not the canonical policy-action dataset")
            reports.append(
                {
                    "episode": name,
                    "steps": int(policy_actions.shape[0]),
                    "policy_action_shape": list(policy_actions.shape),
                    "resolved_target_shape": list(targets.shape),
                    "cartesian_action_shape": list(cartesian.shape) if cartesian is not None else None,
                    "max_roundtrip_error": _roundtrip_error(policy_actions, targets),
                }
            )
    return {"path": str(path), "contract": CONTRACT_NAME, "episodes": reports}


def _validate_episode_arrays(name: str, policy_actions, targets, cartesian) -> None:
    import numpy as np

    policy_dim = int(policy_actions.shape[-1]) if policy_actions.ndim == 2 else 0
    if policy_dim not in (PSM_POLICY_ACTION_DIM, 2 * PSM_POLICY_ACTION_DIM):
        raise RuntimeError(f"{name}/processed_actions has shape {policy_actions.shape}, expected (T, 7) or (T, 14)")
    psm_count = policy_dim // PSM_POLICY_ACTION_DIM
    if targets.shape != policy_actions.shape:
        raise RuntimeError(f"{name}/resolved_joint_targets has shape {targets.shape}, expected {policy_actions.shape}")
    if cartesian is not None and (
        cartesian.ndim != 2
        or cartesian.shape[0] != policy_actions.shape[0]
        or cartesian.shape[-1] not in _cartesian_action_dims(psm_count)
    ):
        raise RuntimeError(
            f"{name}/actions has shape {cartesian.shape}, expected Cartesian width in "
            f"{_cartesian_action_dims(psm_count)}"
        )
    if not np.isfinite(policy_actions).all() or not np.isfinite(targets).all():
        raise RuntimeError(f"{name} contains NaN or infinity in the PSM action contract")
    if not np.isin(policy_actions[:, 6::PSM_POLICY_ACTION_DIM], (-1.0, 1.0)).all():
        raise RuntimeError(f"{name} has a non-binary logical gripper action")
    error = _roundtrip_error(policy_actions, targets)
    if error > 2.0e-6:
        raise RuntimeError(f"{name} joint-action round trip error {error:.8f} exceeds 2e-6")


def _roundtrip_error(policy_actions, targets) -> float:
    import numpy as np

    # NVIDIA's PSM policy action term uses scale=0.5 and the configured home
    # pose as its default offset.  Read the scale live when Isaac is available;
    # the pinned v0.7 fallback keeps offline inspection possible.
    try:
        scale = _joint_policy_scale()
    except (ImportError, ModuleNotFoundError):
        scale = 0.5
    default_arm = np.asarray((0.01, 0.01, 0.07, 0.01, 0.01, 0.01), dtype=np.float32)
    errors = []
    for start in range(0, policy_actions.shape[-1], PSM_POLICY_ACTION_DIM):
        arm_slice = slice(start, start + PSM_ARM_DIM)
        decoded = default_arm.reshape(1, -1) + float(scale) * policy_actions[:, arm_slice]
        errors.append(np.max(np.abs(decoded - targets[:, arm_slice])))
    return float(max(errors))


def _cartesian_action_dims(psm_count: int) -> tuple[int, ...]:
    if psm_count == 1:
        return (PSM_CARTESIAN_ACTION_DIM - 1, PSM_CARTESIAN_ACTION_DIM)
    if psm_count == 2:
        # NVIDIA's v0.7 dual absolute-IK embodiment emits two poses without jaw
        # commands (14), while Dr.Anmar's bimanual relative-IK room includes one
        # logical gripper per PSM (14) and a future complete absolute mode is 16.
        return (14, 16)
    raise RuntimeError(f"unsupported PSM count: {psm_count}")


def _record_to_arg(argv: Sequence[str]) -> Path | None:
    for index, value in enumerate(argv):
        if value == "--record-to" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if value.startswith("--record-to="):
            return Path(value.split("=", 1)[1])
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inspect-hdf5", type=Path)
    return parser


def main() -> None:
    args, remaining = _build_parser().parse_known_args()
    if args.inspect_hdf5:
        print(json.dumps(inspect_hdf5(args.inspect_hdf5), indent=2, sort_keys=True))
        return

    record_to = _record_to_arg(remaining)
    if record_to is None:
        raise SystemExit("The native PSM adapter requires --record-to <file.hdf5>.")
    env_id = None
    for index, value in enumerate(remaining):
        if value == "--env" and index + 1 < len(remaining):
            env_id = remaining[index + 1]
        elif value.startswith("--env="):
            env_id = value.split("=", 1)[1]
    if not env_id:
        raise SystemExit("The native PSM adapter requires --env <NVIDIA environment>.")
    from common.config import get_env_robot_config

    robot_id = get_env_robot_config(env_id).id
    if robot_id not in {"psm", "dual_psm"}:
        raise SystemExit("This adapter accepts NVIDIA PSM and dual-PSM surgical environments.")

    bootstrap = Path(
        os.environ.get(
            "DR_ANMAR_PSM_RECORDER_BOOTSTRAP",
            str(record_to.expanduser().resolve().parent / ".recorder-bootstrap"),
        )
    )
    install_native_recorder(
        env_id=env_id,
        bootstrap_root=bootstrap,
    )

    from arena import run as arena_run

    sys.argv = [sys.argv[0], *remaining]
    arena_error: BaseException | None = None
    try:
        arena_run.main()
    except BaseException as exc:
        arena_error = exc
    finally:
        if record_to.expanduser().resolve().is_file():
            report = finalize_hdf5(record_to)
            print("[dr-anmar-psm] canonical action contract")
            print(json.dumps(report, indent=2, sort_keys=True))
    if arena_error is not None:
        raise arena_error


if __name__ == "__main__":
    main()
