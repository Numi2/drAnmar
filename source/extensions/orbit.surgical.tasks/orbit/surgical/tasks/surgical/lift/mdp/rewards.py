# Copyright (c) 2024-2026, The ORBIT-Surgical and Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from orbit.surgical.tasks.surgical import mdp_common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    contact_threshold: float = 0.01,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_1_name: str = "jaw_1_object_contact",
    sensor_2_name: str = "jaw_2_object_contact",
) -> torch.Tensor:
    """Reward lifting only while both jaws physically contact the object."""
    obj: RigidObject = env.scene[object_cfg.name]
    grasped = mdp_common.bilateral_contact(env, sensor_1_name, sensor_2_name, contact_threshold)
    return ((mdp_common.as_torch(obj.data.root_pos_w)[:, 2] > minimal_height) & grasped).float()


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward measured end-effector proximity to the object."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    distance = torch.linalg.vector_norm(
        mdp_common.as_torch(obj.data.root_pos_w)
        - mdp_common.as_torch(ee_frame.data.target_pos_w)[:, 0, :],
        dim=-1,
    )
    return 1.0 - torch.tanh(distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    contact_threshold: float = 0.01,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_1_name: str = "jaw_1_object_contact",
    sensor_2_name: str = "jaw_2_object_contact",
) -> torch.Tensor:
    """Reward commanded position only for a grasped object above the surface."""
    obj: RigidObject = env.scene[object_cfg.name]
    distance, _ = mdp_common.object_goal_errors(env, command_name, robot_cfg, object_cfg)
    grasped = mdp_common.bilateral_contact(env, sensor_1_name, sensor_2_name, contact_threshold)
    return ((mdp_common.as_torch(obj.data.root_pos_w)[:, 2] > minimal_height) & grasped).float() * (
        1.0 - torch.tanh(distance / std)
    )


def bilateral_grasp(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_1_name: str = "jaw_1_object_contact",
    sensor_2_name: str = "jaw_2_object_contact",
) -> torch.Tensor:
    """Reward physical contact on both opposing jaws, not jaw closure alone."""
    return mdp_common.bilateral_contact(env, sensor_1_name, sensor_2_name, threshold).float()


def object_goal_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    contact_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_1_name: str = "jaw_1_object_contact",
    sensor_2_name: str = "jaw_2_object_contact",
) -> torch.Tensor:
    """Reward commanded needle orientation only while physically grasped."""
    _, rot_error = mdp_common.object_goal_errors(env, command_name, robot_cfg, object_cfg)
    grasped = mdp_common.bilateral_contact(env, sensor_1_name, sensor_2_name, contact_threshold)
    return grasped.float() * (1.0 - torch.tanh(rot_error / std))


def stable_object_motion(
    env: ManagerBasedRLEnv,
    linear_std: float,
    angular_std: float,
    contact_threshold: float,
    sensor_1_name: str = "jaw_1_object_contact",
    sensor_2_name: str = "jaw_2_object_contact",
) -> torch.Tensor:
    """Reward controlled object motion only while both jaws hold it."""
    motion = mdp_common.object_motion(env)
    grasped = mdp_common.bilateral_contact(env, sensor_1_name, sensor_2_name, contact_threshold)
    stable = (1.0 - torch.tanh(motion[:, 0] / linear_std)) * (
        1.0 - torch.tanh(motion[:, 1] / angular_std)
    )
    return grasped.float() * stable


def successful_lift(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimum_height: float,
    position_threshold: float,
    orientation_threshold: float,
    contact_threshold: float,
    maximum_linear_speed: float,
    maximum_angular_speed: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_1_name: str = "jaw_1_object_contact",
    sensor_2_name: str = "jaw_2_object_contact",
) -> torch.Tensor:
    """Sparse procedure completion for a grasped, aligned, stable object at goal."""
    obj: RigidObject = env.scene[object_cfg.name]
    pos_error, rot_error = mdp_common.object_goal_errors(env, command_name, robot_cfg, object_cfg)
    motion = mdp_common.object_motion(env, object_cfg)
    grasped = mdp_common.bilateral_contact(env, sensor_1_name, sensor_2_name, contact_threshold)
    return (
        grasped
        & (mdp_common.as_torch(obj.data.root_pos_w)[:, 2] > minimum_height)
        & (pos_error < position_threshold)
        & (rot_error < orientation_threshold)
        & (motion[:, 0] < maximum_linear_speed)
        & (motion[:, 1] < maximum_angular_speed)
    ).float()


class sustained_lift_success(ManagerTermBase):
    """Require physics-qualified lift success for consecutive control steps."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._consecutive = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._publish_metric = bool(cfg.params.get("publish_metric", False))

    def reset(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        if self._publish_metric:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self._succeeded[env_ids].float().mean().item()
            )
        self._consecutive[env_ids] = 0
        self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        required_consecutive_steps: int,
        command_name: str,
        minimum_height: float,
        position_threshold: float,
        orientation_threshold: float,
        contact_threshold: float,
        maximum_linear_speed: float,
        maximum_angular_speed: float,
        publish_metric: bool = False,
        return_zero: bool = False,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        sensor_1_name: str = "jaw_1_object_contact",
        sensor_2_name: str = "jaw_2_object_contact",
    ) -> torch.Tensor:
        del publish_metric
        current = successful_lift(
            env,
            command_name=command_name,
            minimum_height=minimum_height,
            position_threshold=position_threshold,
            orientation_threshold=orientation_threshold,
            contact_threshold=contact_threshold,
            maximum_linear_speed=maximum_linear_speed,
            maximum_angular_speed=maximum_angular_speed,
            robot_cfg=robot_cfg,
            object_cfg=object_cfg,
            sensor_1_name=sensor_1_name,
            sensor_2_name=sensor_2_name,
        ).bool()
        self._consecutive[:] = torch.where(
            current,
            self._consecutive + 1,
            torch.zeros_like(self._consecutive),
        )
        sustained = self._consecutive >= required_consecutive_steps
        self._succeeded |= sustained
        if return_zero:
            return torch.zeros(env.num_envs, device=env.device)
        return sustained


class sustained_pickup_success(ManagerTermBase):
    """Require a bilateral, physics-owned pickup for consecutive control steps."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._consecutive = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._publish_metric = bool(cfg.params.get("publish_metric", False))

    def reset(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        if self._publish_metric:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = (
                self._succeeded[env_ids].float().mean().item()
            )
        self._consecutive[env_ids] = 0
        self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        required_consecutive_steps: int,
        minimum_height: float,
        contact_threshold: float,
        publish_metric: bool = False,
        return_zero: bool = False,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        sensor_1_name: str = "jaw_1_object_contact",
        sensor_2_name: str = "jaw_2_object_contact",
    ) -> torch.Tensor:
        del publish_metric
        current = object_is_lifted(
            env,
            minimal_height=minimum_height,
            contact_threshold=contact_threshold,
            object_cfg=object_cfg,
            sensor_1_name=sensor_1_name,
            sensor_2_name=sensor_2_name,
        ).bool()
        self._consecutive[:] = torch.where(
            current,
            self._consecutive + 1,
            torch.zeros_like(self._consecutive),
        )
        sustained = self._consecutive >= required_consecutive_steps
        self._succeeded |= sustained
        if return_zero:
            return torch.zeros(env.num_envs, device=env.device)
        return sustained


def contact_force_excess(
    env: ManagerBasedRLEnv, sensor_names: tuple[str, ...], soft_limit: float
) -> torch.Tensor:
    """Penalize force above the configured research envelope."""
    return mdp_common.contact_force_excess(env, sensor_names, soft_limit)


def non_object_contact_force_excess(
    env: ManagerBasedRLEnv, sensor_names: tuple[str, ...], soft_limit: float
) -> torch.Tensor:
    """Penalize jaw contact with anything except the manipulated object."""
    return mdp_common.non_object_contact_force_excess(env, sensor_names, soft_limit)


def rcm_motion(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="psm_remote_center_link"),
) -> torch.Tensor:
    """Penalize movement of the physical PSM remote-center link."""
    return torch.square(mdp_common.rcm_linear_speed(env, robot_cfg))
