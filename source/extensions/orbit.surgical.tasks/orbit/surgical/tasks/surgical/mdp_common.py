# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Native Isaac Lab state helpers shared by surgical manipulation MDPs.

These functions only read simulator-owned articulation, rigid-body, frame-transform,
command, and contact-sensor tensors. They never attach or teleport an object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse, quat_error_magnitude, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def as_torch(value) -> torch.Tensor:
    """Return an Isaac Lab data view as a torch tensor across runtime generations."""
    return value.torch if hasattr(value, "torch") else value


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position and orientation expressed in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    pos_b, quat_b = subtract_frame_transforms(
        as_torch(robot.data.root_pos_w),
        as_torch(robot.data.root_quat_w),
        as_torch(obj.data.root_pos_w),
        as_torch(obj.data.root_quat_w),
    )
    return torch.cat((pos_b, quat_b), dim=-1)


def object_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object linear and angular velocity expressed in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    lin_b = quat_apply_inverse(as_torch(robot.data.root_quat_w), as_torch(obj.data.root_lin_vel_w))
    ang_b = quat_apply_inverse(as_torch(robot.data.root_quat_w), as_torch(obj.data.root_ang_vel_w))
    return torch.cat((lin_b, ang_b), dim=-1)


def end_effector_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Measured tool-tip pose expressed in the corresponding robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    frame: FrameTransformer = env.scene[frame_cfg.name]
    pos_b, quat_b = subtract_frame_transforms(
        as_torch(robot.data.root_pos_w),
        as_torch(robot.data.root_quat_w),
        as_torch(frame.data.target_pos_w)[:, 0, :],
        as_torch(frame.data.target_quat_w)[:, 0, :],
    )
    return torch.cat((pos_b, quat_b), dim=-1)


def contact_force_magnitude(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Maximum filtered normal contact force reported by one native sensor."""
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    forces = as_torch(sensor.data.force_matrix_w) if sensor.data.force_matrix_w is not None else None
    if forces is None:
        forces = as_torch(sensor.data.net_forces_w) if sensor.data.net_forces_w is not None else None
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.linalg.vector_norm(forces.reshape(env.num_envs, -1, 3), dim=-1).amax(dim=-1)


def filtered_contact_force_magnitudes(
    env: ManagerBasedRLEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Per-filter native contact magnitudes for one sensing body.

    PhysX one-to-many filtered contact reporting preserves one column per
    configured partner collider in this runtime. Diagnostic users can
    therefore attribute a jaw contact without altering the aggregated contact
    signal used by the task.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    filtered = as_torch(sensor.data.force_matrix_w) if sensor.data.force_matrix_w is not None else None
    if filtered is None:
        return torch.zeros((env.num_envs, 0), device=env.device)
    return torch.linalg.vector_norm(filtered.reshape(env.num_envs, -1, 3), dim=-1)


def non_object_contact_force_magnitude(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Magnitude of native jaw force not accounted for by the filtered object.

    This uses the vector residual between the sensor's total force and its
    object-filtered force. It avoids unsupported GPU filtering against complex
    table/anatomy collider hierarchies while remaining entirely PhysX-derived.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    net = as_torch(sensor.data.net_forces_w) if sensor.data.net_forces_w is not None else None
    filtered = as_torch(sensor.data.force_matrix_w) if sensor.data.force_matrix_w is not None else None
    if net is None:
        return torch.zeros(env.num_envs, device=env.device)
    net_vector = net.reshape(env.num_envs, -1, 3).sum(dim=1)
    if filtered is None:
        return torch.linalg.vector_norm(net_vector, dim=-1)
    object_vector = filtered.reshape(env.num_envs, -1, 3).sum(dim=1)
    return torch.linalg.vector_norm(net_vector - object_vector, dim=-1)


def paired_contact_forces(
    env: ManagerBasedRLEnv, sensor_1_name: str, sensor_2_name: str, scale: float = 1.0
) -> torch.Tensor:
    """Two measured jaw-contact magnitudes, optionally normalized for policy input."""
    forces = torch.stack(
        (contact_force_magnitude(env, sensor_1_name), contact_force_magnitude(env, sensor_2_name)), dim=-1
    )
    return forces * scale


def bilateral_contact(
    env: ManagerBasedRLEnv, sensor_1_name: str, sensor_2_name: str, threshold: float
) -> torch.Tensor:
    """Whether both opposing jaws have measured contact above ``threshold``."""
    return torch.all(paired_contact_forces(env, sensor_1_name, sensor_2_name) > threshold, dim=-1)


def contact_force_excess(env: ManagerBasedRLEnv, sensor_names: tuple[str, ...], soft_limit: float) -> torch.Tensor:
    """Bounded excess over a research force envelope from native contacts."""
    forces = torch.stack([contact_force_magnitude(env, name) for name in sensor_names], dim=-1)
    return torch.square(torch.tanh(torch.relu(forces - soft_limit))).sum(dim=-1)


def maximum_contact_force(env: ManagerBasedRLEnv, sensor_names: tuple[str, ...]) -> torch.Tensor:
    """Largest contact magnitude across a collection of native sensors."""
    return torch.stack([contact_force_magnitude(env, name) for name in sensor_names], dim=-1).amax(dim=-1)


def non_object_contact_force_excess(
    env: ManagerBasedRLEnv, sensor_names: tuple[str, ...], soft_limit: float
) -> torch.Tensor:
    """Bounded unintended-contact excess across native jaw sensors."""
    forces = torch.stack([non_object_contact_force_magnitude(env, name) for name in sensor_names], dim=-1)
    return torch.square(torch.tanh(torch.relu(forces - soft_limit))).sum(dim=-1)


def maximum_non_object_contact_force(env: ManagerBasedRLEnv, sensor_names: tuple[str, ...]) -> torch.Tensor:
    """Largest native force not produced by the manipulated object."""
    return torch.stack(
        [non_object_contact_force_magnitude(env, name) for name in sensor_names], dim=-1
    ).amax(dim=-1)


def rcm_linear_speed(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Measured remote-center-link speed; zero is the ideal fixed RCM condition."""
    robot: Articulation = env.scene[robot_cfg.name]
    velocity = as_torch(robot.data.body_lin_vel_w)[:, robot_cfg.body_ids, :]
    return torch.linalg.vector_norm(velocity, dim=-1).amax(dim=-1)


def commanded_object_pose_w(
    env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform a pose command from the robot root into world coordinates."""
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    return combine_frame_transforms(
        as_torch(robot.data.root_pos_w),
        as_torch(robot.data.root_quat_w),
        command[:, :3],
        command[:, 3:7],
    )


def object_goal_errors(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Position and quaternion angular error to a commanded object pose."""
    obj: RigidObject = env.scene[object_cfg.name]
    goal_pos_w, goal_quat_w = commanded_object_pose_w(env, command_name, robot_cfg)
    pos_error = torch.linalg.vector_norm(goal_pos_w - as_torch(obj.data.root_pos_w), dim=-1)
    rot_error = quat_error_magnitude(as_torch(obj.data.root_quat_w), goal_quat_w)
    return pos_error, rot_error


def object_motion(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Measured object linear and angular speed magnitudes."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.stack(
        (
            torch.linalg.vector_norm(as_torch(obj.data.root_lin_vel_w), dim=-1),
            torch.linalg.vector_norm(as_torch(obj.data.root_ang_vel_w), dim=-1),
        ),
        dim=-1,
    )


class sticky_success_rate(ManagerTermBase):
    """Track whether each environment ever meets a task success condition.

    The term intentionally returns zero reward. On reset it publishes the
    completed-episode mean at the unified benchmark key.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
            self._succeeded[env_ids].float().mean().item()
        )
        self._succeeded[env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, success_fn, success_params: dict | None = None) -> torch.Tensor:
        self._succeeded |= success_fn(env, **(success_params or {})).bool()
        return torch.zeros(env.num_envs, device=env.device)
