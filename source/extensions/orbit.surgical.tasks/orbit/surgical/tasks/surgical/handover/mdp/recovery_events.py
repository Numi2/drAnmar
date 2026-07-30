# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Reset terms used only by generated recovery-demonstration sweeps."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils


class reset_root_state_uniform_grouped(ManagerTermBase):
    """Reset replica groups to identical local object states.

    A group starts from one sampled object state and allocates that state to
    several candidate lanes.  Contact-rich GPU environments are not assumed
    to remain trajectory-identical across spatial replicas.  Exact causal
    replay must pair the same environment index across separate, identically
    seeded runs and verify pre-intervention tensor equality.

    The implementation follows Isaac Lab's BSD-3-Clause
    ``reset_root_state_uniform`` state semantics while sampling once per
    replica group.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        keys = ("x", "y", "z", "roll", "pitch", "yaw")
        pose_range = cfg.params.get("pose_range", {})
        velocity_range = cfg.params.get("velocity_range", {})
        self._pose_ranges = torch.tensor(
            [tuple(pose_range.get(key, (0.0, 0.0))) for key in keys],
            device=env.device,
        )
        self._velocity_ranges = torch.tensor(
            [tuple(velocity_range.get(key, (0.0, 0.0))) for key in keys],
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        replicas: int = 1,
    ) -> None:
        if replicas < 1:
            raise ValueError("recovery sweep replicas must be positive")
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        complete_groups = (
            len(env_ids) % replicas == 0
            and bool(
                torch.all(
                    env_ids.reshape(-1, replicas)[:, 0] % replicas == 0
                )
            )
            and bool(
                torch.all(
                    env_ids.reshape(-1, replicas)
                    == (
                        env_ids.reshape(-1, replicas)[:, :1]
                        + torch.arange(
                            replicas,
                            device=env_ids.device,
                        )
                    )
                )
            )
        )
        active_replicas = replicas if complete_groups else 1
        group_count = len(env_ids) // active_replicas
        default_pose = asset.data.default_root_pose.torch[env_ids]
        default_velocity = asset.data.default_root_vel.torch[env_ids]

        pose_samples = math_utils.sample_uniform(
            self._pose_ranges[:, 0],
            self._pose_ranges[:, 1],
            (group_count, 6),
            device=asset.device,
        ).repeat_interleave(active_replicas, dim=0)
        positions = (
            default_pose[:, :3]
            + env.scene.env_origins[env_ids]
            + pose_samples[:, :3]
        )
        orientation_delta = math_utils.quat_from_euler_xyz(
            pose_samples[:, 3],
            pose_samples[:, 4],
            pose_samples[:, 5],
        )
        orientations = math_utils.quat_mul(
            default_pose[:, 3:7],
            orientation_delta,
        )
        velocity_samples = math_utils.sample_uniform(
            self._velocity_ranges[:, 0],
            self._velocity_ranges[:, 1],
            (group_count, 6),
            device=asset.device,
        ).repeat_interleave(active_replicas, dim=0)
        velocities = default_velocity + velocity_samples

        asset.write_root_pose_to_sim_index(
            root_pose=torch.cat((positions, orientations), dim=-1),
            env_ids=env_ids,
        )
        asset.write_root_velocity_to_sim_index(
            root_velocity=velocities,
            env_ids=env_ids,
        )
