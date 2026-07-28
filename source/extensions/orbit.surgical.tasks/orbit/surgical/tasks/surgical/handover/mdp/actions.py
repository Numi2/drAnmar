# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task-space safety projection for physical handover actions."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)

from orbit.surgical.tasks.surgical import mdp_common


class HandoverProtectedSurfaceShieldAction(
    DifferentialInverseKinematicsAction
):
    """Retract a recovered receiver after soft non-object contact.

    The learned policy and analytic controller remain unchanged. This action
    projection sees native PhysX contact before applying the next IK command,
    so it can prevent a soft collision from escalating to the unchanged 2 N
    hard termination.
    """

    soft_force_limit_n = 0.25
    hold_steps = 3
    retreat_action_z = 0.05

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._shield_steps_remaining = torch.zeros(
            self.num_envs,
            dtype=torch.int64,
            device=self.device,
        )
        self.shield_activation_steps = torch.zeros(
            (),
            dtype=torch.int64,
            device=self.device,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._shield_steps_remaining.zero_()
        else:
            self._shield_steps_remaining[env_ids] = 0

    def process_actions(self, actions: torch.Tensor) -> None:
        state = getattr(self._env, "_dr_anmar_handover_state", None)
        if state is None:
            super().process_actions(actions)
            return

        asset_name = str(self.cfg.asset_name)
        giver_is_robot_1 = state["giver_is_robot_1"]
        receiver_role = (
            giver_is_robot_1
            if asset_name == "robot_2"
            else ~giver_is_robot_1
        )
        recovery_receiver = (
            receiver_role
            & (state["pickup_recovery_count"] > 0)
            & (state["progress_phase"] == 2)
        )
        sensor_names = (
            f"{asset_name}_jaw_1_object_contact",
            f"{asset_name}_jaw_2_object_contact",
        )
        maximum_non_object_force = torch.stack(
            [
                mdp_common.non_object_contact_force_magnitude(
                    self._env,
                    sensor_name,
                )
                for sensor_name in sensor_names
            ],
            dim=-1,
        ).amax(dim=-1)
        soft_contact = (
            recovery_receiver
            & (maximum_non_object_force > self.soft_force_limit_n)
        )
        self._shield_steps_remaining = torch.where(
            soft_contact,
            torch.full_like(
                self._shield_steps_remaining,
                self.hold_steps,
            ),
            torch.clamp(self._shield_steps_remaining - 1, min=0),
        )
        shield_active = (
            recovery_receiver
            & (self._shield_steps_remaining > 0)
        )
        shielded_actions = actions.clone()
        shielded_actions[shield_active, :6] = 0.0
        shielded_actions[shield_active, 2] = self.retreat_action_z
        self.shield_activation_steps += shield_active.sum()
        super().process_actions(shielded_actions)
