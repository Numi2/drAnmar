# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-grounded phase state for physical two-instrument handover."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from orbit.surgical.tasks.surgical import mdp_common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_RECEIVER_CURRICULUM_STATE_FIELDS = (
    "phase",
    "progress_phase",
    "rewarded_phase",
    "giver_is_robot_1",
    "receiver_only_consecutive",
    "giver_contact_history",
    "receiver_contact_history",
    "presentation_stable_consecutive",
    "presentation_qualified",
    "receiver_capture_consecutive",
    "giver_release_confirmation_consecutive",
    "giver_release_authorized",
    "receiver_capture_offset_w",
    "receiver_attempt_active",
    "receiver_attempt_step_count",
    "receiver_approach_step_count",
    "receiver_pre_release_loss_consecutive",
    "receiver_retry_step_count",
    "receiver_retry_count",
    "receiver_release_abort_count",
    "receiver_loss_consecutive",
    "giver_release_observed",
    "pickup_attempt_count",
    "pickup_recovery_count",
    "pickup_contact_loss_consecutive",
    "recovery_open_step_count",
    "pickup_attempts_exhausted",
    "giver_acquisition_offset_w",
    "receiver_acquisition_offset_w",
)


def assign_balanced_handover_roles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Assign alternating giver roles without changing physical observations.

    Initial populations are exactly balanced when the environment count is
    even.  Each environment swaps giver on every subsequent reset, preventing
    the closest-arm reset geometry from collapsing all experience onto one
    physical robot.
    """
    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
    forced_roles = getattr(
        env,
        "_dr_anmar_forced_giver_is_robot_1",
        None,
    )
    generations = getattr(
        env,
        "_dr_anmar_balanced_role_generation",
        None,
    )
    if forced_roles is None:
        forced_roles = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        env._dr_anmar_forced_giver_is_robot_1 = forced_roles
    if generations is None:
        generations = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        env._dr_anmar_balanced_role_generation = generations
    forced_roles[env_ids] = (
        (env_ids + generations[env_ids]) % 2 == 0
    )
    generations[env_ids] += 1


def _failure_stratified_receiver_sources(
    cache: dict[str, Any],
    target_env_ids: torch.Tensor,
) -> torch.Tensor:
    """Sample every available role/recovery stratum, weighted by failures."""
    available = torch.nonzero(
        cache["valid"],
        as_tuple=False,
    ).squeeze(-1)
    if not bool(available.numel()):
        return target_env_ids.to(dtype=torch.long)
    giver_is_robot_1 = cache["handover_state"][
        "giver_is_robot_1"
    ][available]
    recovered = cache["handover_state"][
        "pickup_recovery_count"
    ][available] > 0
    stratum = giver_is_robot_1.long() + 2 * recovered.long()
    nonempty_strata = [
        value
        for value in range(4)
        if bool((stratum == value).any())
    ]
    # Isaac's reset IDs may be int32 while ``torch.nonzero`` and tensor
    # indexing use int64.  Source IDs are indices, so keep them long
    # regardless of the event-manager input dtype.
    selected = torch.empty(
        target_env_ids.shape,
        dtype=torch.long,
        device=target_env_ids.device,
    )
    for target_offset, stratum_value in enumerate(
        nonempty_strata
    ):
        target_positions = torch.arange(
            target_offset,
            target_env_ids.numel(),
            len(nonempty_strata),
            device=target_env_ids.device,
        )
        if not bool(target_positions.numel()):
            continue
        candidates = available[stratum == stratum_value]
        weights = cache["source_failure_priority"][
            candidates
        ].clamp_min(0.05)
        sampled = torch.multinomial(
            weights,
            target_positions.numel(),
            replacement=True,
        )
        selected[target_positions] = candidates[sampled]
    cache["failure_stratified_restores"] += int(
        target_env_ids.numel()
    )
    return selected


def _update_receiver_source_failure_priority(
    env: ManagerBasedRLEnv,
    cache: dict[str, Any],
    env_ids: torch.Tensor,
) -> None:
    """Feed simulator terminal outcomes back into replay sampling priority."""
    previously_restored = cache["active_restore_valid"][env_ids]
    if not bool(previously_restored.any()):
        return
    state = getattr(env, "_dr_anmar_handover_state", None)
    if state is None:
        return
    completed_env_ids = env_ids[previously_restored]
    source_ids = cache["active_restore_source_env_ids"][
        completed_env_ids
    ]
    succeeded = state["successful_handover"][completed_env_ids]
    target_priority = torch.where(
        succeeded,
        torch.full_like(
            source_ids,
            0.25,
            dtype=torch.float32,
        ),
        torch.full_like(
            source_ids,
            2.0,
            dtype=torch.float32,
        ),
    )
    for source_id in torch.unique(source_ids):
        source_mask = source_ids == source_id
        observed_priority = target_priority[source_mask].mean()
        cache["source_failure_priority"][source_id] = (
            0.8 * cache["source_failure_priority"][source_id]
            + 0.2 * observed_priority
        )
    cache["failure_priority_updates"] += int(
        source_ids.numel()
    )
    cache["active_restore_valid"][completed_env_ids] = False


def reset_receiver_curriculum_from_cache(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Restore complete Markov states captured during physical handover.

    A replayed rigid-body state without its latched giver, phase, contact
    filters, counters, acquisition offsets, and previous action is not the
    state that generated the transition. The cache therefore restores both
    simulator state here and the corresponding logical state on the next
    ``handover_state`` update. This remains training-only and never modifies
    an active end-to-end qualification episode.
    """
    cache = getattr(env, "_dr_anmar_receiver_curriculum_cache", None)
    if cache is None:
        return
    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
    restored_mask = getattr(
        env,
        "_dr_anmar_receiver_curriculum_restored",
        None,
    )
    if restored_mask is None:
        restored_mask = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        setattr(
            env,
            "_dr_anmar_receiver_curriculum_restored",
            restored_mask,
        )
    if bool(
        getattr(
            env.cfg,
            "dr_anmar_failure_stratified_curriculum",
            False,
        )
    ):
        _update_receiver_source_failure_priority(
            env,
            cache,
            env_ids,
        )
    cache["active_restore_valid"][env_ids] = False
    restored_mask[env_ids] = False
    valid = cache["valid"][env_ids]
    restore_probability = float(
        getattr(
            env.cfg,
            "dr_anmar_receiver_curriculum_restore_probability",
            0.5,
        )
    )
    restore_draw = torch.rand(
        env_ids.shape,
        device=env.device,
    )
    restore = valid & (restore_draw < restore_probability)
    refresh_env_ids = env_ids[valid & ~restore]
    if bool(refresh_env_ids.numel()):
        cache["valid"][refresh_env_ids] = False
        cache["reset_refreshes"] += int(refresh_env_ids.numel())
    target_env_ids = env_ids[restore]
    if not bool(target_env_ids.numel()):
        return
    cache["reset_restores"] += int(target_env_ids.numel())
    source_env_ids = target_env_ids
    cross_environment_sampling = bool(
        getattr(
            env.cfg,
            "dr_anmar_receiver_curriculum_cross_environment_sampling",
            False,
        )
    )
    if cross_environment_sampling:
        if bool(
            getattr(
                env.cfg,
                "dr_anmar_failure_stratified_curriculum",
                False,
            )
        ):
            source_env_ids = _failure_stratified_receiver_sources(
                cache,
                target_env_ids,
            )
        else:
            available_source_ids = torch.nonzero(
                cache["valid"],
                as_tuple=False,
            ).squeeze(-1)
            if bool(available_source_ids.numel()):
                source_env_ids = available_source_ids[
                    torch.randint(
                        available_source_ids.numel(),
                        (target_env_ids.numel(),),
                        device=env.device,
                    )
                ]
        if bool(source_env_ids.numel()):
            cache["cross_environment_restores"] += int(
                target_env_ids.numel()
            )
    robot_1: Articulation = env.scene["robot_1"]
    robot_2: Articulation = env.scene["robot_2"]
    obj: RigidObject = env.scene["object"]
    robot_1.write_joint_state_to_sim(
        cache["robot_1_joint_pos"][source_env_ids],
        cache["robot_1_joint_vel"][source_env_ids],
        env_ids=target_env_ids,
    )
    robot_2.write_joint_state_to_sim(
        cache["robot_2_joint_pos"][source_env_ids],
        cache["robot_2_joint_vel"][source_env_ids],
        env_ids=target_env_ids,
    )
    object_root_pose_w = cache["object_root_pose_w"][source_env_ids].clone()
    if cross_environment_sampling:
        source_origins = env.scene.env_origins[source_env_ids]
        target_origins = env.scene.env_origins[target_env_ids]
        object_root_pose_w[:, :3] += target_origins - source_origins
    obj.write_root_pose_to_sim(
        object_root_pose_w,
        env_ids=target_env_ids,
    )
    obj.write_root_velocity_to_sim(
        cache["object_root_velocity_w"][source_env_ids],
        env_ids=target_env_ids,
    )
    cache["restored_source_env_ids"][target_env_ids] = source_env_ids
    cache["active_restore_source_env_ids"][
        target_env_ids
    ] = source_env_ids
    cache["active_restore_valid"][target_env_ids] = True
    cache["restored_episode_length_buf"][target_env_ids] = cache[
        "episode_length_buf"
    ][source_env_ids]
    cache["markov_state_restores"] += int(target_env_ids.numel())
    cache["recovery_context_restores"] += int(
        (
            cache["handover_state"]["pickup_recovery_count"][
                source_env_ids
            ]
            > 0
        )
        .sum()
        .item()
    )
    restored_mask[target_env_ids] = True


def reset_pickup_recovery_curriculum_from_cache(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Restore simulator states captured at a real pickup-loss transition."""
    cache = getattr(env, "_dr_anmar_pickup_recovery_curriculum_cache", None)
    if cache is None:
        return
    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
    restored_mask = getattr(
        env,
        "_dr_anmar_pickup_recovery_curriculum_restored",
        None,
    )
    if restored_mask is None:
        restored_mask = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        setattr(
            env,
            "_dr_anmar_pickup_recovery_curriculum_restored",
            restored_mask,
        )
    restored_mask[env_ids] = False
    valid = cache["valid"][env_ids]
    restore_probability = float(
        getattr(
            env.cfg,
            "dr_anmar_pickup_recovery_curriculum_restore_probability",
            0.9,
        )
    )
    restore = valid & (
        torch.rand(env_ids.shape, device=env.device) < restore_probability
    )
    refresh_env_ids = env_ids[valid & ~restore]
    if bool(refresh_env_ids.numel()):
        cache["valid"][refresh_env_ids] = False
        cache["reset_refreshes"] += int(refresh_env_ids.numel())
    target_env_ids = env_ids[restore]
    if not bool(target_env_ids.numel()):
        return
    cache["reset_restores"] += int(target_env_ids.numel())
    source_env_ids = target_env_ids
    if bool(
        getattr(
            env.cfg,
            "dr_anmar_pickup_recovery_curriculum_cross_environment_sampling",
            False,
        )
    ):
        available_source_ids = torch.nonzero(
            cache["valid"],
            as_tuple=False,
        ).squeeze(-1)
        if bool(available_source_ids.numel()):
            source_env_ids = available_source_ids[
                torch.randint(
                    available_source_ids.numel(),
                    (target_env_ids.numel(),),
                    device=env.device,
                )
            ]
            cache["cross_environment_restores"] += int(
                target_env_ids.numel()
            )
    robot_1: Articulation = env.scene["robot_1"]
    robot_2: Articulation = env.scene["robot_2"]
    obj: RigidObject = env.scene["object"]
    robot_1.write_joint_state_to_sim(
        cache["robot_1_joint_pos"][source_env_ids],
        cache["robot_1_joint_vel"][source_env_ids],
        env_ids=target_env_ids,
    )
    robot_2.write_joint_state_to_sim(
        cache["robot_2_joint_pos"][source_env_ids],
        cache["robot_2_joint_vel"][source_env_ids],
        env_ids=target_env_ids,
    )
    object_root_pose_w = cache["object_root_pose_w"][source_env_ids].clone()
    source_origins = env.scene.env_origins[source_env_ids]
    target_origins = env.scene.env_origins[target_env_ids]
    object_root_pose_w[:, :3] += target_origins - source_origins
    obj.write_root_pose_to_sim(
        object_root_pose_w,
        env_ids=target_env_ids,
    )
    obj.write_root_velocity_to_sim(
        cache["object_root_velocity_w"][source_env_ids],
        env_ids=target_env_ids,
    )
    cache["restored_giver_is_robot_1"][target_env_ids] = cache[
        "giver_is_robot_1"
    ][source_env_ids]
    restored_mask[target_env_ids] = True


def _step_number(env: ManagerBasedRLEnv) -> int:
    value = env.common_step_counter
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def handover_state(
    env: ManagerBasedRLEnv,
    contact_threshold: float = 0.01,
    pickup_clearance: float = 0.01,
    contact_window_steps: int = 5,
    contact_required_steps: int = 3,
    maximum_pickup_attempts: int = 3,
    pickup_contact_loss_steps: int = 3,
    giver_follow_tolerance: float = 0.005,
    recovery_open_steps: int = 15,
    recovery_support_clearance: float = 0.005,
    recovery_linear_speed_limit: float = 0.05,
    recovery_angular_speed_limit: float = 5.0,
    required_receiver_only_steps: int = 10,
    allowed_receiver_contact_flicker_steps: int = 1,
    receiver_follow_tolerance: float = 0.005,
    presentation_fraction_from_giver: float = 0.35,
    presentation_height_in_robot_frame: float = -0.13,
    presentation_ready_tolerance: float = 0.005,
    presentation_stability_steps: int = 1,
    presentation_linear_speed_limit: float = 0.05,
    presentation_angular_speed_limit: float = 5.0,
    receiver_capture_required_steps: int = 1,
    receiver_capture_follow_tolerance: float = 0.005,
    receiver_capture_linear_speed_limit: float = 0.05,
    receiver_capture_angular_speed_limit: float = 5.0,
    giver_release_confirmation_steps: int = 5,
    receiver_attempt_timeout_steps: int = 30,
    receiver_approach_timeout_steps: int = 0,
    receiver_retry_contact_loss_steps: int = 2,
    receiver_retry_steps: int = 0,
    reset_height_offset: float = -0.05,
    command_name: str = "receiver_pose",
) -> dict[str, Any]:
    """Update and return the monotonic physical handover phase.

    Phases are: 0 closest-arm approach, 1 giver grasp, 2 lifted presentation,
    3 receiver acquisition, 4 safe pickup recovery. The giver identity is
    latched at reset from the two physical tool-tip distances to the needle.
    Pickup accepts bilateral PhysX contact in three of five control steps.
    The base task accepts receiver acquisition on the same filtered contact;
    an environment may opt into a stricter presentation/capture contract via
    ``cfg.dr_anmar_handover_contract``. Under that contract the giver first
    settles at a fixed presentation pose, the receiver must retain a stable
    bilateral capture, both tools hold still and closed for a separate release
    confirmation period, and a pre-release miss returns to a receiver-only
    retry while the giver keeps custody. Retention permits one missing contact
    frame only while the elevated needle preserves its receiver-relative offset.
    Before receiver acquisition, three consecutive frames without live giver
    custody trigger an open-jaw recovery and analytic reacquisition. The first
    attempt plus recoveries are capped at ``maximum_pickup_attempts``.
    """
    contract = getattr(env.cfg, "dr_anmar_handover_contract", None)
    if contract:
        presentation_fraction_from_giver = float(
            contract.get(
                "presentation_fraction_from_giver",
                presentation_fraction_from_giver,
            )
        )
        presentation_height_in_robot_frame = float(
            contract.get(
                "presentation_height_in_robot_frame",
                presentation_height_in_robot_frame,
            )
        )
        presentation_ready_tolerance = float(
            contract.get(
                "presentation_ready_tolerance",
                presentation_ready_tolerance,
            )
        )
        presentation_stability_steps = int(
            contract.get(
                "presentation_stability_steps",
                presentation_stability_steps,
            )
        )
        presentation_linear_speed_limit = float(
            contract.get(
                "presentation_linear_speed_limit",
                presentation_linear_speed_limit,
            )
        )
        presentation_angular_speed_limit = float(
            contract.get(
                "presentation_angular_speed_limit",
                presentation_angular_speed_limit,
            )
        )
        receiver_capture_required_steps = int(
            contract.get(
                "receiver_capture_required_steps",
                receiver_capture_required_steps,
            )
        )
        receiver_capture_follow_tolerance = float(
            contract.get(
                "receiver_capture_follow_tolerance",
                receiver_capture_follow_tolerance,
            )
        )
        receiver_capture_linear_speed_limit = float(
            contract.get(
                "receiver_capture_linear_speed_limit",
                receiver_capture_linear_speed_limit,
            )
        )
        receiver_capture_angular_speed_limit = float(
            contract.get(
                "receiver_capture_angular_speed_limit",
                receiver_capture_angular_speed_limit,
            )
        )
        giver_release_confirmation_steps = int(
            contract.get(
                "giver_release_confirmation_steps",
                giver_release_confirmation_steps,
            )
        )
        receiver_attempt_timeout_steps = int(
            contract.get(
                "receiver_attempt_timeout_steps",
                receiver_attempt_timeout_steps,
            )
        )
        receiver_approach_timeout_steps = int(
            contract.get(
                "receiver_approach_timeout_steps",
                receiver_approach_timeout_steps,
            )
        )
        receiver_retry_contact_loss_steps = int(
            contract.get(
                "receiver_retry_contact_loss_steps",
                receiver_retry_contact_loss_steps,
            )
        )
        receiver_retry_steps = int(
            contract.get("receiver_retry_steps", receiver_retry_steps)
        )
        required_receiver_only_steps = int(
            contract.get(
                "required_receiver_only_steps",
                required_receiver_only_steps,
            )
        )
        if required_receiver_only_steps <= 0:
            raise ValueError(
                "required_receiver_only_steps must be positive"
            )
    step = _step_number(env)
    state = getattr(env, "_dr_anmar_handover_state", None)
    if state is None:
        obj: RigidObject = env.scene["object"]
        support_height_w = (
            mdp_common.as_torch(obj.data.default_root_state)[:, 2]
            + reset_height_offset
        )
        state = {
            "phase": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "progress_phase": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "giver_is_robot_1": torch.ones(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "rewarded_phase": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "start_object_pos": mdp_common.as_torch(obj.data.root_pos_w).clone(),
            "support_height_w": support_height_w,
            "last_reset_step": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
            "receiver_only_consecutive": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "giver_contact_history": torch.zeros(
                (env.num_envs, contact_window_steps),
                dtype=torch.bool,
                device=env.device,
            ),
            "receiver_contact_history": torch.zeros(
                (env.num_envs, contact_window_steps),
                dtype=torch.bool,
                device=env.device,
            ),
            "presentation_stable_consecutive": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "presentation_qualified": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "receiver_capture_consecutive": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "giver_release_confirmation_consecutive": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "giver_release_authorized": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "receiver_capture_offset_w": torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            ),
            "receiver_attempt_active": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "receiver_attempt_step_count": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "receiver_approach_step_count": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "receiver_pre_release_loss_consecutive": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "receiver_retry_step_count": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "receiver_retry_count": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "receiver_release_abort_count": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "receiver_loss_consecutive": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "giver_release_observed": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "successful_handover": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "pickup_attempt_count": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "pickup_recovery_count": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "pickup_contact_loss_consecutive": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "recovery_open_step_count": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "pickup_attempts_exhausted": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "giver_acquisition_offset_w": torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            ),
            "last_pickup_attempt_count": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "last_pickup_recovery_count": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "last_pickup_attempts_exhausted": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "last_successful_attempt": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "last_success_was_recovered": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "last_progress_phase": torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            "premature_release": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "receiver_retention_failed": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "retention_failure_low_clearance": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "retention_failure_follow_error": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "retention_failure_contact_loss": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "last_retention_failure_low_clearance": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "last_retention_failure_follow_error": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "last_retention_failure_contact_loss": torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            "receiver_acquisition_offset_w": torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            ),
            "last_step": -1,
        }
        setattr(env, "_dr_anmar_handover_state", state)
    obj = env.scene["object"]
    reset = (env.episode_length_buf == 0) & (state["last_reset_step"] != step)
    object_pos_w = mdp_common.as_torch(obj.data.root_pos_w)
    robot_1_frame: FrameTransformer = env.scene["ee_1_frame"]
    robot_2_frame: FrameTransformer = env.scene["ee_2_frame"]
    robot_1_position_w = mdp_common.as_torch(
        robot_1_frame.data.target_pos_w
    )[:, 0, :]
    robot_2_position_w = mdp_common.as_torch(
        robot_2_frame.data.target_pos_w
    )[:, 0, :]
    robot_1_distance = torch.linalg.vector_norm(
        robot_1_position_w - object_pos_w, dim=-1
    )
    robot_2_distance = torch.linalg.vector_norm(
        robot_2_position_w - object_pos_w, dim=-1
    )
    forced_giver_is_robot_1 = getattr(
        env,
        "_dr_anmar_forced_giver_is_robot_1",
        None,
    )
    if forced_giver_is_robot_1 is None:
        state["giver_is_robot_1"][reset] = (
            robot_1_distance[reset] <= robot_2_distance[reset]
        )
    else:
        state["giver_is_robot_1"][reset] = (
            forced_giver_is_robot_1[reset]
        )
    state["last_pickup_attempt_count"][reset] = state[
        "pickup_attempt_count"
    ][reset]
    state["last_pickup_recovery_count"][reset] = state[
        "pickup_recovery_count"
    ][reset]
    state["last_pickup_attempts_exhausted"][reset] = state[
        "pickup_attempts_exhausted"
    ][reset]
    state["last_successful_attempt"][reset] = torch.where(
        state["successful_handover"][reset],
        state["pickup_attempt_count"][reset],
        torch.zeros_like(state["pickup_attempt_count"][reset]),
    )
    state["last_success_was_recovered"][reset] = (
        state["successful_handover"][reset]
        & (state["pickup_attempt_count"][reset] > 1)
    )
    state["last_progress_phase"][reset] = state["progress_phase"][reset]
    state["phase"][reset] = 0
    state["progress_phase"][reset] = 0
    state["rewarded_phase"][reset] = 0
    state["receiver_only_consecutive"][reset] = 0
    state["giver_contact_history"][reset] = False
    state["receiver_contact_history"][reset] = False
    state["presentation_stable_consecutive"][reset] = 0
    state["presentation_qualified"][reset] = False
    state["receiver_capture_consecutive"][reset] = 0
    state["giver_release_confirmation_consecutive"][reset] = 0
    state["giver_release_authorized"][reset] = False
    state["receiver_capture_offset_w"][reset] = 0.0
    state["receiver_attempt_active"][reset] = False
    state["receiver_attempt_step_count"][reset] = 0
    state["receiver_approach_step_count"][reset] = 0
    state["receiver_pre_release_loss_consecutive"][reset] = 0
    state["receiver_retry_step_count"][reset] = 0
    state["receiver_retry_count"][reset] = 0
    state["receiver_release_abort_count"][reset] = 0
    state["receiver_loss_consecutive"][reset] = 0
    state["giver_release_observed"][reset] = False
    state["successful_handover"][reset] = False
    state["pickup_attempt_count"][reset] = 0
    state["pickup_recovery_count"][reset] = 0
    state["pickup_contact_loss_consecutive"][reset] = 0
    state["recovery_open_step_count"][reset] = 0
    state["pickup_attempts_exhausted"][reset] = False
    state["giver_acquisition_offset_w"][reset] = 0.0
    state["premature_release"][reset] = False
    state["last_retention_failure_low_clearance"][reset] = state[
        "retention_failure_low_clearance"
    ][reset]
    state["last_retention_failure_follow_error"][reset] = state[
        "retention_failure_follow_error"
    ][reset]
    state["last_retention_failure_contact_loss"][reset] = state[
        "retention_failure_contact_loss"
    ][reset]
    state["receiver_retention_failed"][reset] = False
    state["retention_failure_low_clearance"][reset] = False
    state["retention_failure_follow_error"][reset] = False
    state["retention_failure_contact_loss"][reset] = False
    state["receiver_acquisition_offset_w"][reset] = 0.0
    state["start_object_pos"][reset] = object_pos_w[reset]
    state["start_object_pos"][reset, 2] = state["support_height_w"][reset]
    restored_recovery = getattr(
        env,
        "_dr_anmar_pickup_recovery_curriculum_restored",
        None,
    )
    if restored_recovery is not None:
        restored_recovery = reset & restored_recovery
        recovery_cache = getattr(
            env,
            "_dr_anmar_pickup_recovery_curriculum_cache",
            None,
        )
        state["phase"][restored_recovery] = 4
        state["pickup_attempt_count"][restored_recovery] = 1
        state["pickup_recovery_count"][restored_recovery] = 1
        if recovery_cache is not None:
            state["giver_is_robot_1"][restored_recovery] = recovery_cache[
                "restored_giver_is_robot_1"
            ][restored_recovery]
        getattr(
            env,
            "_dr_anmar_pickup_recovery_curriculum_restored",
        )[restored_recovery] = False
    restored_receiver = getattr(
        env,
        "_dr_anmar_receiver_curriculum_restored",
        None,
    )
    if restored_receiver is not None:
        restored_receiver = reset & restored_receiver
        receiver_cache = getattr(
            env,
            "_dr_anmar_receiver_curriculum_cache",
            None,
        )
        if receiver_cache is not None and bool(restored_receiver.any()):
            target_env_ids = torch.nonzero(
                restored_receiver,
                as_tuple=False,
            ).squeeze(-1)
            source_env_ids = receiver_cache[
                "restored_source_env_ids"
            ][target_env_ids]
            for field in _RECEIVER_CURRICULUM_STATE_FIELDS:
                state[field][target_env_ids] = receiver_cache[
                    "handover_state"
                ][field][source_env_ids]
            # Event-manager reset ordering may clear action buffers after the
            # physical restore. Restore the transition's previous action here,
            # when observations request the logical handover state.
            mdp_common.as_torch(env.action_manager.action)[
                target_env_ids
            ] = receiver_cache["last_action"][source_env_ids]
            # Terminal outcomes never cross an episode boundary. Every other
            # restored tensor is part of the state that generated the cached
            # transition and is required for a Markov-complete replay.
            env.episode_length_buf[target_env_ids] = receiver_cache[
                "restored_episode_length_buf"
            ][target_env_ids]
            state["successful_handover"][target_env_ids] = False
            state["premature_release"][target_env_ids] = False
            state["receiver_retention_failed"][target_env_ids] = False
            state["retention_failure_low_clearance"][target_env_ids] = False
            state["retention_failure_follow_error"][target_env_ids] = False
            state["retention_failure_contact_loss"][target_env_ids] = False
        getattr(
            env,
            "_dr_anmar_receiver_curriculum_restored",
        )[restored_receiver] = False
    state["last_reset_step"][reset] = step
    # A downstream curriculum may restore a reset-time snapshot that was
    # captured only after this exact physical handover contract succeeded.
    # Restore never changes an active episode and never fabricates a cache
    # from target poses.  Rehydrate the monotonic handover state so the
    # receiver can train the successor skill without replaying the full
    # prerequisite on every rollout.
    pending_safe_bite_restore = getattr(
        env,
        "_dr_anmar_pending_safe_bite_restore",
        None,
    )
    if isinstance(pending_safe_bite_restore, dict):
        restored = reset & pending_safe_bite_restore["episode_mask"]
        if bool(restored.any()):
            state["giver_is_robot_1"][restored] = (
                pending_safe_bite_restore["giver_is_robot_1"][restored]
            )
            state["phase"][restored] = 3
            state["progress_phase"][restored] = 4
            state["rewarded_phase"][restored] = 4
            state["giver_release_observed"][restored] = True
            state["giver_release_authorized"][restored] = True
            state["successful_handover"][restored] = True
            state["receiver_only_consecutive"][restored] = (
                required_receiver_only_steps
            )
            state["receiver_acquisition_offset_w"][restored] = (
                pending_safe_bite_restore[
                    "receiver_acquisition_offset_w"
                ][restored]
            )
            # Newton contact buffers need a few substeps to repopulate after
            # a reset write.  A negative counter is a bounded debounce, not
            # artificial custody: geometric follow and lifted-state checks
            # remain live and any sustained loss still terminates.
            state["receiver_loss_consecutive"][restored] = -int(
                pending_safe_bite_restore["contact_grace_steps"]
            )
    if state["last_step"] == step and not bool(torch.any(reset)):
        return state

    robot_1_contact_forces = mdp_common.paired_contact_forces(
        env,
        "robot_1_jaw_1_object_contact",
        "robot_1_jaw_2_object_contact",
    )
    robot_2_contact_forces = mdp_common.paired_contact_forces(
        env,
        "robot_2_jaw_1_object_contact",
        "robot_2_jaw_2_object_contact",
    )
    robot_1_contact_now = torch.all(
        robot_1_contact_forces > contact_threshold,
        dim=-1,
    )
    robot_2_contact_now = torch.all(
        robot_2_contact_forces > contact_threshold,
        dim=-1,
    )
    robot_1_any_contact_now = torch.any(
        robot_1_contact_forces > contact_threshold,
        dim=-1,
    )
    robot_2_any_contact_now = torch.any(
        robot_2_contact_forces > contact_threshold,
        dim=-1,
    )
    giver_is_robot_1 = state["giver_is_robot_1"]
    giver_contact_now = torch.where(
        giver_is_robot_1,
        robot_1_contact_now,
        robot_2_contact_now,
    )
    giver_any_contact_now = torch.where(
        giver_is_robot_1,
        robot_1_any_contact_now,
        robot_2_any_contact_now,
    )
    receiver_contact_now = torch.where(
        giver_is_robot_1,
        robot_2_contact_now,
        robot_1_contact_now,
    )
    receiver_any_contact_now = torch.where(
        giver_is_robot_1,
        robot_2_any_contact_now,
        robot_1_any_contact_now,
    )
    state["giver_contact_history"] = torch.roll(
        state["giver_contact_history"], shifts=-1, dims=-1
    )
    state["receiver_contact_history"] = torch.roll(
        state["receiver_contact_history"], shifts=-1, dims=-1
    )
    state["giver_contact_history"][:, -1] = giver_contact_now
    state["receiver_contact_history"][:, -1] = receiver_contact_now
    giver_contact = (
        state["giver_contact_history"].sum(dim=-1) >= contact_required_steps
    )
    receiver_contact = (
        state["receiver_contact_history"].sum(dim=-1) >= contact_required_steps
    )
    receiver_position_w = torch.where(
        giver_is_robot_1.unsqueeze(-1),
        robot_2_position_w,
        robot_1_position_w,
    )
    giver_position_w = torch.where(
        giver_is_robot_1.unsqueeze(-1),
        robot_1_position_w,
        robot_2_position_w,
    )
    receiver_distance = torch.linalg.vector_norm(
        receiver_position_w - object_pos_w, dim=-1
    )
    clearance = object_pos_w[:, 2] - state["support_height_w"]
    lifted = clearance >= pickup_clearance
    pos_error, rot_error = mdp_common.object_goal_errors(
        env, command_name, SceneEntityCfg("robot_2"), SceneEntityCfg("object")
    )
    motion = mdp_common.object_motion(env)
    object_pose_robot_1 = mdp_common.object_pose_in_robot_root_frame(
        env,
        SceneEntityCfg("robot_1"),
        SceneEntityCfg("object"),
    )
    object_pose_robot_2 = mdp_common.object_pose_in_robot_root_frame(
        env,
        SceneEntityCfg("robot_2"),
        SceneEntityCfg("object"),
    )
    object_pose_in_giver = torch.where(
        giver_is_robot_1.unsqueeze(-1),
        object_pose_robot_1,
        object_pose_robot_2,
    )
    object_pose_in_receiver = torch.where(
        giver_is_robot_1.unsqueeze(-1),
        object_pose_robot_2,
        object_pose_robot_1,
    )
    root_2_in_giver = (
        object_pose_in_giver[:, :3]
        - object_pose_in_receiver[:, :3]
    )
    presentation_target_in_giver = (
        presentation_fraction_from_giver * root_2_in_giver
    )
    presentation_target_in_giver[:, 2] = (
        presentation_height_in_robot_frame
    )
    presentation_error = torch.linalg.vector_norm(
        presentation_target_in_giver - object_pose_in_giver[:, :3],
        dim=-1,
    )
    structured_transfer_contract = bool(contract)

    phase = state["phase"]
    new_pickup_attempt = (phase == 0) & giver_contact
    state["pickup_attempt_count"][new_pickup_attempt] += 1
    state["giver_acquisition_offset_w"][new_pickup_attempt] = (
        object_pos_w[new_pickup_attempt]
        - giver_position_w[new_pickup_attempt]
    )
    phase[new_pickup_attempt] = 1
    state["progress_phase"][new_pickup_attempt] = torch.maximum(
        state["progress_phase"][new_pickup_attempt],
        torch.ones_like(state["progress_phase"][new_pickup_attempt]),
    )
    before_acquisition = (phase >= 1) & (phase < 3)
    physical_action = mdp_common.as_torch(
        env.action_manager.action
    )
    giver_open_action = torch.where(
        giver_is_robot_1,
        physical_action[:, 6],
        physical_action[:, 13],
    )
    state["premature_release"] |= (
        before_acquisition
        & (giver_open_action > 0.0)
    )
    newly_lifted = (
        (phase == 1)
        & lifted
        & ~state["premature_release"]
    )
    phase[newly_lifted] = 2
    state["progress_phase"][newly_lifted] = torch.maximum(
        state["progress_phase"][newly_lifted],
        torch.full_like(state["progress_phase"][newly_lifted], 2),
    )
    giver_relative_offset = object_pos_w - giver_position_w
    giver_follow_error = torch.linalg.vector_norm(
        giver_relative_offset - state["giver_acquisition_offset_w"],
        dim=-1,
    )
    giver_follows = giver_follow_error <= giver_follow_tolerance
    giver_custody = giver_contact | (
        lifted & giver_follows
    )
    use_filtered_presentation_custody = bool(
        contract
        and contract.get(
            "presentation_use_filtered_custody",
            False,
        )
    )
    presentation_custody = (
        giver_custody
        if use_filtered_presentation_custody
        else giver_contact_now
    )
    presentation_ready_now = (
        (phase == 2)
        & presentation_custody
        & lifted
        & (presentation_error <= presentation_ready_tolerance)
        & (motion[:, 0] <= presentation_linear_speed_limit)
        & (motion[:, 1] <= presentation_angular_speed_limit)
    )
    state["presentation_stable_consecutive"][:] = torch.where(
        presentation_ready_now,
        state["presentation_stable_consecutive"] + 1,
        torch.zeros_like(state["presentation_stable_consecutive"]),
    )
    state["presentation_qualified"] |= (
        state["presentation_stable_consecutive"]
        >= presentation_stability_steps
    )
    presentation_stable = state["presentation_qualified"]
    if bool(
        getattr(env.cfg, "dr_anmar_receiver_curriculum", False)
    ):
        cache = getattr(
            env,
            "_dr_anmar_receiver_curriculum_cache",
            None,
        )
        robot_1: Articulation = env.scene["robot_1"]
        robot_2: Articulation = env.scene["robot_2"]
        if cache is None:
            cache = {
                "valid": torch.zeros(
                    env.num_envs,
                    dtype=torch.bool,
                    device=env.device,
                ),
                "reset_restores": 0,
                "reset_refreshes": 0,
                "cross_environment_restores": 0,
                "failure_stratified_restores": 0,
                "failure_priority_updates": 0,
                "recovery_conditioned_captures": 0,
                "markov_state_restores": 0,
                "recovery_context_restores": 0,
                "restored_source_env_ids": torch.arange(
                    env.num_envs,
                    dtype=torch.long,
                    device=env.device,
                ),
                "active_restore_source_env_ids": torch.arange(
                    env.num_envs,
                    dtype=torch.long,
                    device=env.device,
                ),
                "active_restore_valid": torch.zeros(
                    env.num_envs,
                    dtype=torch.bool,
                    device=env.device,
                ),
                "episode_length_buf": torch.zeros(
                    env.num_envs,
                    dtype=torch.long,
                    device=env.device,
                ),
                "restored_episode_length_buf": torch.zeros(
                    env.num_envs,
                    dtype=torch.long,
                    device=env.device,
                ),
                "robot_1_joint_pos": torch.zeros_like(
                    mdp_common.as_torch(robot_1.data.joint_pos)
                ),
                "robot_1_joint_vel": torch.zeros_like(
                    mdp_common.as_torch(robot_1.data.joint_vel)
                ),
                "robot_2_joint_pos": torch.zeros_like(
                    mdp_common.as_torch(robot_2.data.joint_pos)
                ),
                "robot_2_joint_vel": torch.zeros_like(
                    mdp_common.as_torch(robot_2.data.joint_vel)
                ),
                "object_root_pose_w": torch.zeros(
                    (env.num_envs, 7),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "object_root_velocity_w": torch.zeros(
                    (env.num_envs, 6),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "last_action": torch.zeros_like(
                    mdp_common.as_torch(env.action_manager.action)
                ),
                "source_failure_priority": torch.ones(
                    env.num_envs,
                    dtype=torch.float32,
                    device=env.device,
                ),
                "handover_state": {
                    field: torch.zeros_like(state[field])
                    for field in _RECEIVER_CURRICULUM_STATE_FIELDS
                },
            }
            setattr(
                env,
                "_dr_anmar_receiver_curriculum_cache",
                cache,
            )
        capture_stage = str(
            getattr(
                env.cfg,
                "dr_anmar_receiver_curriculum_capture_stage",
                "stable_presentation",
            )
        )
        if capture_stage == "stable_presentation":
            capture_ready = presentation_stable
        elif capture_stage == "lifted_custody":
            capture_ready = (phase == 2) & lifted & giver_custody
        else:
            raise ValueError(
                "receiver curriculum capture stage must be "
                "'stable_presentation' or 'lifted_custody'"
            )
        capture = capture_ready & ~cache["valid"]
        require_pickup_recovery = bool(
            getattr(
                env.cfg,
                "dr_anmar_receiver_curriculum_require_pickup_recovery",
                False,
            )
        )
        if require_pickup_recovery:
            capture &= state["pickup_recovery_count"] > 0
        if bool(capture.any()):
            if require_pickup_recovery:
                cache["recovery_conditioned_captures"] += int(
                    capture.sum().item()
                )
            cache["robot_1_joint_pos"][capture] = mdp_common.as_torch(
                robot_1.data.joint_pos
            )[capture]
            cache["robot_1_joint_vel"][capture] = mdp_common.as_torch(
                robot_1.data.joint_vel
            )[capture]
            cache["robot_2_joint_pos"][capture] = mdp_common.as_torch(
                robot_2.data.joint_pos
            )[capture]
            cache["robot_2_joint_vel"][capture] = mdp_common.as_torch(
                robot_2.data.joint_vel
            )[capture]
            cache["object_root_pose_w"][capture, :3] = object_pos_w[
                capture
            ]
            cache["object_root_pose_w"][capture, 3:7] = mdp_common.as_torch(
                obj.data.root_quat_w
            )[capture]
            cache["object_root_velocity_w"][capture, :3] = (
                mdp_common.as_torch(obj.data.root_lin_vel_w)[capture]
            )
            cache["object_root_velocity_w"][capture, 3:6] = (
                mdp_common.as_torch(obj.data.root_ang_vel_w)[capture]
            )
            cache["last_action"][capture] = mdp_common.as_torch(
                env.action_manager.action
            )[capture]
            cache["episode_length_buf"][capture] = (
                env.episode_length_buf[capture]
            )
            for field in _RECEIVER_CURRICULUM_STATE_FIELDS:
                cache["handover_state"][field][capture] = state[field][
                    capture
                ]
            cache["valid"][capture] = True
            cache["source_failure_priority"][capture] = 1.0

    retry_was_active = state["receiver_retry_step_count"] > 0
    state["receiver_retry_step_count"][:] = torch.where(
        retry_was_active,
        state["receiver_retry_step_count"] + 1,
        state["receiver_retry_step_count"],
    )
    retry_complete = (
        retry_was_active
        & (state["receiver_retry_step_count"] > receiver_retry_steps)
    )
    state["receiver_retry_step_count"][retry_complete] = 0
    receiver_retry_active = state["receiver_retry_step_count"] > 0
    receiver_approach_active = (
        structured_transfer_contract
        & (phase == 2)
        & presentation_stable
        & ~receiver_any_contact_now
        & ~receiver_retry_active
        & ~state["receiver_attempt_active"]
        & (state["receiver_retry_count"] == 0)
    )
    state["receiver_approach_step_count"][:] = torch.where(
        receiver_approach_active,
        state["receiver_approach_step_count"] + 1,
        torch.zeros_like(state["receiver_approach_step_count"]),
    )
    receiver_capture_began = (
        structured_transfer_contract
        & (phase == 2)
        & presentation_stable
        & receiver_any_contact_now
        & ~receiver_retry_active
        & ~state["receiver_attempt_active"]
    )
    state["receiver_attempt_active"] |= receiver_capture_began
    state["receiver_attempt_step_count"][:] = torch.where(
        (
            (phase == 2)
            & state["receiver_attempt_active"]
            & ~receiver_retry_active
        ),
        state["receiver_attempt_step_count"] + 1,
        torch.zeros_like(state["receiver_attempt_step_count"]),
    )
    state["receiver_capture_offset_w"][receiver_capture_began] = (
        object_pos_w[receiver_capture_began]
        - receiver_position_w[receiver_capture_began]
    )
    receiver_capture_relative_offset = object_pos_w - receiver_position_w
    receiver_capture_follow_error = torch.linalg.vector_norm(
        receiver_capture_relative_offset
        - state["receiver_capture_offset_w"],
        dim=-1,
    )
    receiver_capture_follows = (
        receiver_capture_follow_error
        <= receiver_capture_follow_tolerance
    )
    receiver_capture_qualified = (
        structured_transfer_contract
        & (phase == 2)
        & presentation_stable
        & receiver_contact
        & ~receiver_retry_active
    )
    state["receiver_capture_consecutive"][:] = torch.where(
        receiver_capture_qualified,
        state["receiver_capture_consecutive"] + 1,
        torch.zeros_like(state["receiver_capture_consecutive"]),
    )
    state["receiver_pre_release_loss_consecutive"][:] = torch.where(
        (
            ((phase == 2) & state["receiver_attempt_active"])
            | (
                (phase == 3)
                & ~state["giver_release_observed"]
            )
        )
        & ~receiver_any_contact_now
        & ~receiver_retry_active,
        state["receiver_pre_release_loss_consecutive"] + 1,
        torch.zeros_like(
            state["receiver_pre_release_loss_consecutive"]
        ),
    )
    receiver_missed_before_capture = (
        structured_transfer_contract
        & (phase == 2)
        & state["receiver_attempt_active"]
        & (
            state["receiver_pre_release_loss_consecutive"]
            >= receiver_retry_contact_loss_steps
        )
    )
    receiver_attempt_stalled = (
        structured_transfer_contract
        & (phase == 2)
        & state["receiver_attempt_active"]
        & (
            state["receiver_attempt_step_count"]
            >= receiver_attempt_timeout_steps
        )
        & (
            state["receiver_capture_consecutive"]
            < receiver_capture_required_steps
        )
    )
    receiver_approach_stalled = (
        receiver_approach_timeout_steps > 0
    ) & (
        receiver_approach_active
        & (
            state["receiver_approach_step_count"]
            >= receiver_approach_timeout_steps
        )
    )
    receiver_release_aborted = (
        structured_transfer_contract
        & (phase == 3)
        & ~state["giver_release_observed"]
        & giver_contact_now
        & (
            state["receiver_pre_release_loss_consecutive"]
            >= receiver_retry_contact_loss_steps
        )
    )
    start_receiver_retry = (
        receiver_retry_steps > 0
    ) & (
        receiver_approach_stalled
        | receiver_missed_before_capture
        | receiver_attempt_stalled
        | receiver_release_aborted
    )
    state["receiver_retry_count"][start_receiver_retry] += 1
    state["receiver_release_abort_count"][receiver_release_aborted] += 1
    state["receiver_retry_step_count"][start_receiver_retry] = 1
    state["receiver_attempt_active"][start_receiver_retry] = False
    state["receiver_attempt_step_count"][start_receiver_retry] = 0
    state["receiver_approach_step_count"][start_receiver_retry] = 0
    state["receiver_capture_consecutive"][start_receiver_retry] = 0
    state["giver_release_confirmation_consecutive"][
        start_receiver_retry
    ] = 0
    state["giver_release_authorized"][start_receiver_retry] = False
    state["receiver_pre_release_loss_consecutive"][
        start_receiver_retry
    ] = 0
    state["receiver_contact_history"][start_receiver_retry] = False
    phase[receiver_release_aborted] = 2
    receiver_retry_active = state["receiver_retry_step_count"] > 0

    if structured_transfer_contract:
        receiver_acquired = (
            (phase == 2)
            & (
                state["receiver_capture_consecutive"]
                >= receiver_capture_required_steps
            )
            & ~receiver_retry_active
        )
    else:
        receiver_acquired = (
            (phase == 2) & giver_contact & receiver_contact
        )
    state["receiver_acquisition_offset_w"][receiver_acquired] = (
        object_pos_w[receiver_acquired]
        - receiver_position_w[receiver_acquired]
    )
    state["receiver_attempt_active"][receiver_acquired] = False
    state["receiver_attempt_step_count"][receiver_acquired] = 0
    state["receiver_approach_step_count"][receiver_acquired] = 0
    phase[receiver_acquired] = 3
    state["progress_phase"][receiver_acquired] = torch.maximum(
        state["progress_phase"][receiver_acquired],
        torch.full_like(state["progress_phase"][receiver_acquired], 3),
    )
    release_confirmation_active = (
        structured_transfer_contract
        & (phase == 3)
        & ~state["giver_release_observed"]
        & receiver_contact
        & receiver_contact_now
    )
    state["giver_release_confirmation_consecutive"][:] = torch.where(
        release_confirmation_active,
        state["giver_release_confirmation_consecutive"] + 1,
        torch.zeros_like(
            state["giver_release_confirmation_consecutive"]
        ),
    )
    state["giver_release_authorized"][:] = (
        release_confirmation_active
        & (
            state["giver_release_confirmation_consecutive"]
            >= giver_release_confirmation_steps
        )
    )
    state["giver_release_observed"] |= (
        (phase == 3)
        & (giver_open_action > 0.0)
        & ~giver_contact_now
    )
    receiver_relative_offset = object_pos_w - receiver_position_w
    receiver_follow_error = torch.linalg.vector_norm(
        receiver_relative_offset - state["receiver_acquisition_offset_w"],
        dim=-1,
    )
    receiver_follows = receiver_follow_error <= receiver_follow_tolerance
    retention_active = (
        (phase == 3)
        & state["giver_release_observed"]
    )
    state["receiver_loss_consecutive"][:] = torch.where(
        retention_active & ~receiver_contact_now,
        state["receiver_loss_consecutive"] + 1,
        torch.zeros_like(state["receiver_loss_consecutive"]),
    )
    receiver_flicker_allowed = (
        state["receiver_loss_consecutive"]
        <= allowed_receiver_contact_flicker_steps
    )
    receiver_only = (
        retention_active
        & lifted
        & (
            receiver_contact_now
            | (
                receiver_flicker_allowed
                & receiver_follows
            )
        )
    )
    retention_failure_low_clearance = retention_active & ~lifted
    retention_failure_follow_error = (
        retention_active
        & ~receiver_contact_now
        & receiver_flicker_allowed
        & ~receiver_follows
    )
    retention_failure_contact_loss = (
        retention_active
        & ~receiver_contact_now
        & ~receiver_flicker_allowed
    )
    state["retention_failure_low_clearance"] |= (
        retention_failure_low_clearance
    )
    state["retention_failure_follow_error"] |= (
        retention_failure_follow_error
    )
    state["retention_failure_contact_loss"] |= (
        retention_failure_contact_loss
    )
    state["receiver_retention_failed"] |= (
        retention_failure_low_clearance
        | retention_failure_follow_error
        | retention_failure_contact_loss
    )
    state["receiver_only_consecutive"][:] = torch.where(
        receiver_only,
        state["receiver_only_consecutive"] + 1,
        torch.zeros_like(state["receiver_only_consecutive"]),
    )
    successful_now = (
        (phase == 3)
        & (state["receiver_only_consecutive"] >= required_receiver_only_steps)
    )
    state["successful_handover"] |= successful_now
    state["progress_phase"][successful_now] = 4

    pickup_active = (phase == 1) | (phase == 2)
    state["pickup_contact_loss_consecutive"][:] = torch.where(
        pickup_active & ~giver_contact_now,
        state["pickup_contact_loss_consecutive"] + 1,
        torch.zeros_like(state["pickup_contact_loss_consecutive"]),
    )
    pickup_attempt_failed = (
        pickup_active
        & (
            state["pickup_contact_loss_consecutive"]
            >= pickup_contact_loss_steps
        )
        & ~giver_follows
        & ~receiver_contact_now
    )
    pickup_attempt_failed |= (
        (phase == 2)
        & (clearance < 0.005)
        & ~receiver_contact_now
    )
    recovery_allowed = (
        pickup_attempt_failed
        & (state["pickup_attempt_count"] < maximum_pickup_attempts)
    )
    attempts_exhausted = (
        pickup_attempt_failed
        & (state["pickup_attempt_count"] >= maximum_pickup_attempts)
    )
    if bool(
        getattr(
            env.cfg,
            "dr_anmar_pickup_recovery_curriculum",
            False,
        )
    ):
        cache = getattr(
            env,
            "_dr_anmar_pickup_recovery_curriculum_cache",
            None,
        )
        robot_1: Articulation = env.scene["robot_1"]
        robot_2: Articulation = env.scene["robot_2"]
        if cache is None:
            cache = {
                "valid": torch.zeros(
                    env.num_envs,
                    dtype=torch.bool,
                    device=env.device,
                ),
                "reset_restores": 0,
                "reset_refreshes": 0,
                "cross_environment_restores": 0,
                "robot_1_joint_pos": torch.zeros_like(
                    mdp_common.as_torch(robot_1.data.joint_pos)
                ),
                "robot_1_joint_vel": torch.zeros_like(
                    mdp_common.as_torch(robot_1.data.joint_vel)
                ),
                "robot_2_joint_pos": torch.zeros_like(
                    mdp_common.as_torch(robot_2.data.joint_pos)
                ),
                "robot_2_joint_vel": torch.zeros_like(
                    mdp_common.as_torch(robot_2.data.joint_vel)
                ),
                "object_root_pose_w": torch.zeros(
                    (env.num_envs, 7),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "object_root_velocity_w": torch.zeros(
                    (env.num_envs, 6),
                    dtype=torch.float32,
                    device=env.device,
                ),
                "giver_is_robot_1": torch.ones(
                    env.num_envs,
                    dtype=torch.bool,
                    device=env.device,
                ),
                "restored_giver_is_robot_1": torch.ones(
                    env.num_envs,
                    dtype=torch.bool,
                    device=env.device,
                ),
            }
            setattr(
                env,
                "_dr_anmar_pickup_recovery_curriculum_cache",
                cache,
            )
        capture = recovery_allowed & ~cache["valid"]
        if bool(capture.any()):
            cache["robot_1_joint_pos"][capture] = mdp_common.as_torch(
                robot_1.data.joint_pos
            )[capture]
            cache["robot_1_joint_vel"][capture] = mdp_common.as_torch(
                robot_1.data.joint_vel
            )[capture]
            cache["robot_2_joint_pos"][capture] = mdp_common.as_torch(
                robot_2.data.joint_pos
            )[capture]
            cache["robot_2_joint_vel"][capture] = mdp_common.as_torch(
                robot_2.data.joint_vel
            )[capture]
            cache["object_root_pose_w"][capture, :3] = object_pos_w[capture]
            cache["object_root_pose_w"][capture, 3:7] = (
                mdp_common.as_torch(obj.data.root_quat_w)[capture]
            )
            cache["object_root_velocity_w"][capture, :3] = (
                mdp_common.as_torch(obj.data.root_lin_vel_w)[capture]
            )
            cache["object_root_velocity_w"][capture, 3:6] = (
                mdp_common.as_torch(obj.data.root_ang_vel_w)[capture]
            )
            cache["giver_is_robot_1"][capture] = giver_is_robot_1[capture]
            cache["valid"][capture] = True
    state["pickup_attempts_exhausted"] |= attempts_exhausted
    state["pickup_recovery_count"][recovery_allowed] += 1
    state["recovery_open_step_count"][recovery_allowed] = 0
    phase[recovery_allowed] = 4
    state["giver_contact_history"][recovery_allowed] = False
    state["receiver_contact_history"][recovery_allowed] = False
    state["presentation_stable_consecutive"][recovery_allowed] = 0
    state["presentation_qualified"][recovery_allowed] = False
    state["receiver_capture_consecutive"][recovery_allowed] = 0
    state["giver_release_confirmation_consecutive"][recovery_allowed] = 0
    state["giver_release_authorized"][recovery_allowed] = False
    state["receiver_capture_offset_w"][recovery_allowed] = 0.0
    state["receiver_attempt_active"][recovery_allowed] = False
    state["receiver_attempt_step_count"][recovery_allowed] = 0
    state["receiver_approach_step_count"][recovery_allowed] = 0
    state["receiver_pre_release_loss_consecutive"][
        recovery_allowed
    ] = 0
    state["receiver_retry_step_count"][recovery_allowed] = 0
    state["pickup_contact_loss_consecutive"][recovery_allowed] = 0

    recovery_active = phase == 4
    state["recovery_open_step_count"][:] = torch.where(
        recovery_active,
        state["recovery_open_step_count"] + 1,
        torch.zeros_like(state["recovery_open_step_count"]),
    )
    recovery_complete = (
        recovery_active
        & (state["recovery_open_step_count"] >= recovery_open_steps)
        & (clearance <= recovery_support_clearance)
        & (motion[:, 0] <= recovery_linear_speed_limit)
        & (motion[:, 1] <= recovery_angular_speed_limit)
        & ~giver_contact_now
    )
    phase[recovery_complete] = 0
    state["giver_contact_history"][recovery_complete] = False
    state["receiver_contact_history"][recovery_complete] = False
    state["presentation_stable_consecutive"][recovery_complete] = 0
    state["presentation_qualified"][recovery_complete] = False
    state["receiver_capture_consecutive"][recovery_complete] = 0
    state["giver_release_confirmation_consecutive"][recovery_complete] = 0
    state["giver_release_authorized"][recovery_complete] = False
    state["receiver_capture_offset_w"][recovery_complete] = 0.0
    state["receiver_attempt_active"][recovery_complete] = False
    state["receiver_attempt_step_count"][recovery_complete] = 0
    state["receiver_approach_step_count"][recovery_complete] = 0
    state["receiver_pre_release_loss_consecutive"][
        recovery_complete
    ] = 0
    state["receiver_retry_step_count"][recovery_complete] = 0
    state["recovery_open_step_count"][recovery_complete] = 0

    state.update(
        {
            "last_step": step,
            "giver_contact": giver_contact,
            "receiver_contact": receiver_contact,
            "giver_is_robot_1": giver_is_robot_1,
            "robot_1_contact_now": robot_1_contact_now,
            "robot_2_contact_now": robot_2_contact_now,
            "giver_contact_now": giver_contact_now,
            "giver_any_contact_now": giver_any_contact_now,
            "giver_custody": giver_custody,
            "receiver_contact_now": receiver_contact_now,
            "receiver_any_contact_now": receiver_any_contact_now,
            "giver_distance": torch.where(
                giver_is_robot_1,
                robot_1_distance,
                robot_2_distance,
            ),
            "receiver_distance": receiver_distance,
            "clearance": clearance,
            "lifted": lifted,
            "presentation_error": presentation_error,
            "presentation_ready_now": presentation_ready_now,
            "presentation_stable": presentation_stable,
            "receiver_capture_follow_error": (
                receiver_capture_follow_error
            ),
            "receiver_capture_follows": receiver_capture_follows,
            "receiver_capture_qualified": receiver_capture_qualified,
            "receiver_retry_active": receiver_retry_active,
            "receiver_follows": receiver_follows,
            "receiver_follow_error": receiver_follow_error,
            "giver_follows": giver_follows,
            "giver_follow_error": giver_follow_error,
            "needle_dropped": (phase == 3) & (clearance < 0.005),
            "position_error": pos_error,
            "orientation_error": rot_error,
            "motion": motion,
        }
    )
    return state
