# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Additional physics observations for the isolated end-to-end experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.sensors import ContactSensor

from orbit.surgical.tasks.surgical import mdp_common

from ...mdp.state import handover_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def previous_jaw_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    history_index: int = 1,
    scale: float = 1.0,
) -> torch.Tensor:
    """Return the preceding filtered force magnitude for each physical jaw."""
    values = []
    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        history = sensor.data.force_matrix_w_history
        if history is None:
            values.append(mdp_common.contact_force_magnitude(env, sensor_name))
            continue
        history_tensor = mdp_common.as_torch(history)
        selected_index = min(history_index, history_tensor.shape[1] - 1)
        forces = history_tensor[:, selected_index]
        values.append(
            torch.linalg.vector_norm(
                forces.reshape(env.num_envs, -1, 3),
                dim=-1,
            ).amax(dim=-1)
        )
    return torch.stack(values, dim=-1) * scale


def transfer_contract_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expose physical presentation, capture, retry, and release authority."""
    state = handover_state(env)
    contract = env.cfg.dr_anmar_handover_contract
    presentation_steps = float(contract["presentation_stability_steps"])
    capture_steps = float(contract["receiver_capture_required_steps"])
    presentation_progress = torch.where(
        state["presentation_qualified"],
        torch.ones_like(state["presentation_stable_consecutive"]).float(),
        (
            state["presentation_stable_consecutive"].float()
            / presentation_steps
        ).clamp(0.0, 1.0),
    )
    capture_progress = (
        state["receiver_capture_consecutive"].float()
        / capture_steps
    ).clamp(0.0, 1.0)
    return torch.stack(
        (
            presentation_progress,
            capture_progress,
            state["receiver_retry_active"].float(),
            state["giver_release_authorized"].float(),
        ),
        dim=-1,
    )


def giver_and_deadline_context(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Preserve giver routing and expose the real remaining episode budget.

    The end-to-end actor routes roles from channel zero and canonicalizes this
    two-channel term before the frozen phase network sees it. Channel one is
    therefore available to a recovery-only option without changing the
    incumbent policy's output or the 107-value observation contract.
    """
    state = handover_state(env)
    giver_is_robot_1 = state["giver_is_robot_1"].float()
    maximum_steps = float(env.max_episode_length)
    remaining_fraction = (
        1.0 - env.episode_length_buf.float() / maximum_steps
    ).clamp(0.0, 1.0)
    return torch.stack(
        (giver_is_robot_1, remaining_fraction),
        dim=-1,
    )
