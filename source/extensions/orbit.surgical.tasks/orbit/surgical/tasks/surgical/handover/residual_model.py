# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Closest-arm physical handover controller with bounded learned residuals."""

from __future__ import annotations

import copy
import math

import torch
from rsl_rl.models import MLPModel
from torch import nn

from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
)

from orbit.surgical.tasks.surgical.lift.grasp_frames import (
    NEEDLE_ARC_EXTENT_RAD,
    NEEDLE_PROVISIONAL_ARC_FRACTION,
    NEEDLE_PROVISIONAL_GRASP_OFFSET_M,
    NEEDLE_PROVISIONAL_TANGENT_YAW_RAD,
    needle_geometry_grasp_frame,
)

from .controller_profiles import apply_controller_profile

_RECEIVER_ARC_FRACTION = 0.65
_RECEIVER_ARC_CANDIDATES = (0.60, 0.65, 0.70)
_RECEIVER_CANDIDATE_OFFSETS = tuple(
    needle_geometry_grasp_frame(fraction)[0]
    for fraction in _RECEIVER_ARC_CANDIDATES
)
(
    _RECEIVER_OFFSET,
    _RECEIVER_TANGENT_YAW_RAD,
) = needle_geometry_grasp_frame(
    _RECEIVER_ARC_FRACTION,
    grasp_z_m=-0.003,
)
_RECEIVER_TANGENT_DELTA_RAD = (
    _RECEIVER_TANGENT_YAW_RAD - NEEDLE_PROVISIONAL_TANGENT_YAW_RAD
)
_RECEIVER_BASELINE_CROSSING_ANGLE_RAD = (
    math.pi - _RECEIVER_TANGENT_DELTA_RAD
)


class HandoverAnalyticController(nn.Module):
    """Exact ordered pickup, presentation, acquisition, and release base."""

    def __init__(self) -> None:
        super().__init__()
        self.position_scale = 0.01
        self.orientation_scale = 0.05
        self.approach_height = 0.02
        self.lateral_alignment_threshold = 0.005
        self.close_distance = 0.005
        self.receiver_close_distance = 0.001
        self.slow_approach_radius = 0.02
        self.slow_approach_action_limit = 0.1
        self.receiver_contact_centering_action_limit = 0.005
        # The receiver must approach the needle beside the giver's jaws, but
        # it must never cross the giver's long insertion shaft. The protected
        # segment starts 25 mm behind the giver tip so intended distal-tool
        # acquisition remains unchanged.
        self.recovery_receiver_shaft_guard_start_from_tip_m = 0.025
        self.recovery_receiver_shaft_guard_activation_distance_m = 0.018
        self.recovery_receiver_shaft_guard_minimum_distance_m = 0.015
        self.receiver_shaft_guard_all_pickups_enabled = False
        self.receiver_jaw_proximal_offset_m = 0.0093
        self.register_buffer(
            "last_recovery_receiver_shaft_guard_active",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "last_recovery_receiver_shaft_distance_m",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "last_giver_custody_quality",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "last_giver_transport_scale",
            torch.empty(0),
            persistent=False,
        )
        self.transport_custody_latch_enabled = True
        self.receiver_preposition_enabled = True
        self.receiver_preposition_height = 0.025
        self.recovery_receiver_preposition_height = 0.025
        self.receiver_preposition_action_limit = 0.15
        # Commanded stop includes the measured ~0.25 rad actuator lag so the
        # physical jaw pose lands at the retained v33 contact boundary.
        self.receiver_contact_orientation_error_target_rad = 1.95
        self.receiver_adaptive_arc_enabled = False
        self.receiver_default_arc_fraction = float(_RECEIVER_ARC_FRACTION)
        self.needle_provisional_arc_fraction = float(NEEDLE_PROVISIONAL_ARC_FRACTION)
        self.needle_arc_extent_rad = float(NEEDLE_ARC_EXTENT_RAD)
        self.register_buffer(
            "receiver_candidate_offsets",
            torch.tensor(_RECEIVER_CANDIDATE_OFFSETS),
            persistent=False,
        )
        self.register_buffer(
            "receiver_candidate_fractions",
            torch.tensor(_RECEIVER_ARC_CANDIDATES),
            persistent=False,
        )
        self.normalized_contact_threshold = 0.002
        self.contact_force_observation_scale = 0.2
        self.giver_lift_contact_force_threshold_n = 0.01
        self.giver_pre_lift_min_contact_jaws = 2
        self.presentation_fraction_from_giver = 0.35
        self.presentation_height_in_robot_frame = -0.13
        self.presentation_ready_tolerance = 0.005
        self.presentation_hold_action_limit = 0.01
        self.minimum_lift_height_in_robot_frame = -0.139
        self.carry_lateral_action_limit = 0.06
        # Recovered grasps are physically less repeatable than reset-aligned
        # grasps. Keep the qualified first-attempt transport unchanged while
        # giving only contact-qualified, already-lifted recovery transport
        # enough lateral authority to reach presentation before the original
        # episode deadline. Relift and live-contact loss fall back to the
        # qualified 0.06 first-attempt limit.
        self.recovery_carry_lateral_action_limit = 0.08
        self.carry_lateral_ramp_height = 0.01
        self.pickup_vertical_action_limit = 0.01
        self.pickup_initial_vertical_action_limit = 0.01
        # A fallen needle is not the reset-aligned Stage 4 pickup geometry.
        # Keep recovery at the qualified handover pickup authority; the
        # 0.18 standalone-lift setting exhausted 110/1,200 handover retries
        # and recovered only one episode.
        self.recovery_pickup_vertical_action_limit = 0.01
        self.pickup_deceleration_height = 0.01
        self.carry_vertical_action_limit = 0.015
        self.giver_lift_on_live_contact = True
        self.giver_pregrasp_orientation_action_limit = 0.6
        self.giver_pregrasp_orientation_tolerance = 0.035
        self.giver_transport_orientation_action_limit = 0.035
        self.receiver_orientation_action_limit = 0.6
        self.receiver_tangent_delta_rad = _RECEIVER_TANGENT_DELTA_RAD
        self.receiver_crossing_angle_rad = (
            _RECEIVER_BASELINE_CROSSING_ANGLE_RAD
        )
        self.receiver_roll_offset_rad = (
            self.receiver_tangent_delta_rad
            + self.receiver_crossing_angle_rad
        )
        # Keep training and serving identical for giver adaptation.  This is
        # deliberately disabled by default because this flag is controller
        # configuration, not checkpoint state.  A checkpoint must never gain
        # an untrained receiver residual merely by being reloaded for play.
        self.receiver_residual_enabled_for_learning = False
        self.receiver_grasp_retain_residual_enabled_for_learning = False
        self.giver_recovery_residual_only_for_learning = False
        self.giver_grasp_x = float(
            NEEDLE_PROVISIONAL_GRASP_OFFSET_M[0]
        )
        self.giver_grasp_y = float(
            NEEDLE_PROVISIONAL_GRASP_OFFSET_M[1]
        )
        self.giver_grasp_z = float(
            NEEDLE_PROVISIONAL_GRASP_OFFSET_M[2]
        )
        self.receiver_grasp_x = float(_RECEIVER_OFFSET[0])
        self.receiver_grasp_y = float(_RECEIVER_OFFSET[1])
        self.receiver_grasp_z = -0.003
        # New geometric and custody semantics are opt-in through a versioned
        # controller profile.  Keeping these defaults false preserves every
        # unbundled legacy checkpoint exactly.
        self.canonical_needle_local_frames_enabled = False
        self.custody_quality_features_enabled = False
        self.custody_preserving_transport_enabled = False
        self.custody_quality_slow_threshold = 0.55
        self.custody_quality_stop_threshold = 0.30
        self.custody_quality_centering_action_limit = 0.02
        self.custody_quality_minimum_transport_scale = 0.20
        self.controller_profile_name = "unbundled-source-default"
        self.controller_profile_sha256 = None

    def configure_profile(self, name: str) -> dict[str, object]:
        """Apply a versioned profile to every non-checkpoint controller field."""
        return apply_controller_profile(self, name)

    def _select_role(
        self,
        robot_1_value: torch.Tensor,
        robot_2_value: torch.Tensor,
        use_robot_1: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            use_robot_1.unsqueeze(-1),
            robot_1_value,
            robot_2_value,
        )

    def _object_relative_offset(
        self,
        object_orientation: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate a needle-local offset with legacy-compatible semantics."""
        if self.canonical_needle_local_frames_enabled:
            return quat_apply(object_orientation, offset)
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
        object_relative_grasp_offset = offset.clone()
        object_relative_grasp_offset[:, 0] = (
            yaw_cosine * offset[:, 0]
            - yaw_sine * offset[:, 1]
        )
        object_relative_grasp_offset[:, 1] = (
            yaw_sine * offset[:, 0]
            + yaw_cosine * offset[:, 1]
        )
        return object_relative_grasp_offset

    def _approach_action(
        self,
        ee_position: torch.Tensor,
        object_position: torch.Tensor,
        object_orientation: torch.Tensor,
        use_object_relative_grasp: torch.Tensor,
        grasp_x: float,
        grasp_y: float,
        grasp_z: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grasp_offset = torch.zeros_like(object_position)
        grasp_offset[:, 0] = grasp_x
        grasp_offset[:, 1] = grasp_y
        grasp_offset[:, 2] = grasp_z
        object_relative_grasp_offset = self._object_relative_offset(
            object_orientation,
            grasp_offset,
        )
        rotated_grasp_offset = torch.where(
            use_object_relative_grasp.unsqueeze(-1),
            object_relative_grasp_offset,
            grasp_offset,
        )
        grasp_position = object_position.clone()
        grasp_position += rotated_grasp_offset
        delta = grasp_position - ee_position
        lateral_distance = torch.linalg.vector_norm(
            delta[:, :2],
            dim=-1,
        )
        above = grasp_position.clone()
        above[:, 2] += self.approach_height
        target = torch.where(
            (
                lateral_distance
                > self.lateral_alignment_threshold
            ).unsqueeze(-1),
            above,
            grasp_position,
        )
        distance = torch.linalg.vector_norm(
            grasp_position - ee_position,
            dim=-1,
        )
        action = (
            (target - ee_position) / self.position_scale
        ).clamp(-1.0, 1.0)
        action = torch.where(
            (distance < self.slow_approach_radius).unsqueeze(-1),
            action.clamp(
                -self.slow_approach_action_limit,
                self.slow_approach_action_limit,
            ),
            action,
        )
        return action, distance

    def _receiver_approach_action(
        self,
        ee_position: torch.Tensor,
        object_position: torch.Tensor,
        object_orientation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select a reachable, object-relative grasp frame on the needle arc."""
        candidate_offsets = self.receiver_candidate_offsets.to(
            dtype=object_position.dtype,
            device=object_position.device,
        ).unsqueeze(0)
        candidate_offsets = candidate_offsets.expand(
            object_position.shape[0],
            -1,
            -1,
        )
        if self.canonical_needle_local_frames_enabled:
            expanded_orientation = object_orientation.unsqueeze(1).expand(
                -1,
                candidate_offsets.shape[1],
                -1,
            )
            rotated_offsets = quat_apply(
                expanded_orientation.reshape(-1, 4),
                candidate_offsets.reshape(-1, 3),
            ).reshape_as(candidate_offsets)
        else:
            quaternion_x = object_orientation[:, 0].unsqueeze(-1)
            quaternion_y = object_orientation[:, 1].unsqueeze(-1)
            quaternion_z = object_orientation[:, 2].unsqueeze(-1)
            quaternion_w = object_orientation[:, 3].unsqueeze(-1)
            yaw_sine = 2.0 * (
                quaternion_w * quaternion_z
                + quaternion_x * quaternion_y
            )
            yaw_cosine = 1.0 - 2.0 * (
                quaternion_y * quaternion_y
                + quaternion_z * quaternion_z
            )
            rotated_offsets = candidate_offsets.clone()
            rotated_offsets[:, :, 0] = (
                yaw_cosine * candidate_offsets[:, :, 0]
                - yaw_sine * candidate_offsets[:, :, 1]
            )
            rotated_offsets[:, :, 1] = (
                yaw_sine * candidate_offsets[:, :, 0]
                + yaw_cosine * candidate_offsets[:, :, 1]
            )
        candidate_positions = (
            object_position.unsqueeze(1) + rotated_offsets
        )
        candidate_distances = torch.linalg.vector_norm(
            candidate_positions - ee_position.unsqueeze(1),
            dim=-1,
        )
        selected_index = torch.argmin(candidate_distances, dim=-1)
        batch_index = torch.arange(
            object_position.shape[0],
            device=object_position.device,
        )
        grasp_position = candidate_positions[batch_index, selected_index]
        selected_fraction = self.receiver_candidate_fractions.to(
            dtype=object_position.dtype,
            device=object_position.device,
        )[selected_index]
        delta = grasp_position - ee_position
        lateral_distance = torch.linalg.vector_norm(
            delta[:, :2],
            dim=-1,
        )
        above = grasp_position.clone()
        above[:, 2] += self.approach_height
        target = torch.where(
            (
                lateral_distance
                > self.lateral_alignment_threshold
            ).unsqueeze(-1),
            above,
            grasp_position,
        )
        distance = torch.linalg.vector_norm(delta, dim=-1)
        action = (
            (target - ee_position) / self.position_scale
        ).clamp(-1.0, 1.0)
        action = torch.where(
            (distance < self.slow_approach_radius).unsqueeze(-1),
            action.clamp(
                -self.slow_approach_action_limit,
                self.slow_approach_action_limit,
            ),
            action,
        )
        return action, distance, selected_fraction, grasp_position

    def forward(
        self,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        giver_is_robot_1 = raw[:, 82] > 0.5
        giver_ee = self._select_role(
            raw[:, 32:35],
            raw[:, 39:42],
            giver_is_robot_1,
        )
        giver_orientation = self._select_role(
            raw[:, 35:39],
            raw[:, 42:46],
            giver_is_robot_1,
        )
        receiver_ee = self._select_role(
            raw[:, 32:35],
            raw[:, 39:42],
            ~giver_is_robot_1,
        )
        receiver_orientation = self._select_role(
            raw[:, 35:39],
            raw[:, 42:46],
            ~giver_is_robot_1,
        )
        object_pose_in_giver = self._select_role(
            raw[:, 46:53],
            raw[:, 53:60],
            giver_is_robot_1,
        )
        object_pose_in_receiver = self._select_role(
            raw[:, 46:53],
            raw[:, 53:60],
            ~giver_is_robot_1,
        )
        object_in_giver = object_pose_in_giver[:, :3]
        object_in_receiver = object_pose_in_receiver[:, :3]
        giver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            giver_is_robot_1,
        )
        receiver_contacts = self._select_role(
            raw[:, 66:68],
            raw[:, 68:70],
            ~giver_is_robot_1,
        )
        previous_giver_contacts = self._select_role(
            raw[:, 99:101],
            raw[:, 101:103],
            giver_is_robot_1,
        )
        phase = torch.argmax(raw[:, 77:82], dim=-1)
        pickup_recovery_context = raw[:, 98] > 0.5
        presentation_stable = raw[:, 103] >= 1.0
        receiver_retry_active = raw[:, 105] > 0.5
        identity_tool_orientation = torch.zeros_like(
            giver_orientation
        )
        identity_tool_orientation[:, 3] = 1.0
        if self.canonical_needle_local_frames_enabled:
            giver_tool_target_orientation = object_pose_in_giver[:, 3:7]
        else:
            giver_tool_target_orientation = identity_tool_orientation
        giver_tool_orientation_error = axis_angle_from_quat(
            quat_mul(
                giver_tool_target_orientation,
                quat_conjugate(giver_orientation),
            )
        )
        giver_pregrasp_orientation_action = (
            giver_tool_orientation_error / self.orientation_scale
        ).clamp(
            -self.giver_pregrasp_orientation_action_limit,
            self.giver_pregrasp_orientation_action_limit,
        )
        giver_pregrasp_orientation_ready = (
            torch.linalg.vector_norm(
                giver_tool_orientation_error,
                dim=-1,
            )
            < self.giver_pregrasp_orientation_tolerance
        )

        giver_uses_object_frame = (
            pickup_recovery_context.unsqueeze(-1)
            | torch.full_like(
                pickup_recovery_context.unsqueeze(-1),
                self.canonical_needle_local_frames_enabled,
            )
        )
        giver_approach, giver_distance = self._approach_action(
            giver_ee,
            object_in_giver,
            object_pose_in_giver[:, 3:7],
            giver_uses_object_frame.squeeze(-1),
            self.giver_grasp_x,
            self.giver_grasp_y,
            self.giver_grasp_z,
        )
        giver_pregrasp_position = object_in_giver.clone()
        giver_pregrasp_offset = torch.zeros_like(object_in_giver)
        giver_pregrasp_offset[:, 0] = self.giver_grasp_x
        giver_pregrasp_offset[:, 1] = self.giver_grasp_y
        giver_pregrasp_offset[:, 2] = self.giver_grasp_z
        giver_object_relative_pregrasp_offset = (
            self._object_relative_offset(
                object_pose_in_giver[:, 3:7],
                giver_pregrasp_offset,
            )
        )
        giver_pregrasp_position += torch.where(
            giver_uses_object_frame,
            giver_object_relative_pregrasp_offset,
            giver_pregrasp_offset,
        )
        giver_pregrasp_position[:, 2] += self.approach_height
        giver_orientation_wait_action = (
            (giver_pregrasp_position - giver_ee)
            / self.position_scale
        ).clamp(-1.0, 1.0)
        giver_approach = torch.where(
            (
                (phase == 0)
                & ~giver_pregrasp_orientation_ready
            ).unsqueeze(-1),
            giver_orientation_wait_action,
            giver_approach,
        )
        if self.receiver_adaptive_arc_enabled:
            (
                receiver_approach,
                receiver_distance,
                receiver_arc_fraction,
                _,
            ) = self._receiver_approach_action(
                receiver_ee,
                object_in_receiver,
                object_pose_in_receiver[:, 3:7],
            )
        else:
            receiver_uses_object_frame = torch.full_like(
                pickup_recovery_context,
                self.canonical_needle_local_frames_enabled,
            )
            receiver_approach, receiver_distance = self._approach_action(
                receiver_ee,
                object_in_receiver,
                object_pose_in_receiver[:, 3:7],
                receiver_uses_object_frame,
                self.receiver_grasp_x,
                self.receiver_grasp_y,
                self.receiver_grasp_z,
            )
            receiver_arc_fraction = torch.full_like(
                receiver_distance,
                self.receiver_default_arc_fraction,
            )
        root_2_in_giver = object_in_giver - object_in_receiver
        # Both PSM roots have the same fixed orientation in this task. Express
        # the receiver's distal jaw segment in the giver root frame, then keep
        # its nearest endpoint outside a capsule around the giver insertion
        # shaft. This projects only the inward component and preserves the
        # tangential motion needed to acquire the curved needle.
        receiver_ee_in_giver = receiver_ee + root_2_in_giver
        receiver_jaw_offset = torch.zeros_like(receiver_ee_in_giver)
        receiver_jaw_offset[:, 2] = self.receiver_jaw_proximal_offset_m
        receiver_jaw_proximal_in_giver = (
            receiver_ee_in_giver
            + quat_apply(receiver_orientation, receiver_jaw_offset)
        )
        giver_to_rcm = -giver_ee
        giver_to_rcm_direction = giver_to_rcm / torch.linalg.vector_norm(
            giver_to_rcm,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        giver_shaft_start = (
            giver_ee
            + giver_to_rcm_direction
            * self.recovery_receiver_shaft_guard_start_from_tip_m
        )
        giver_shaft_vector = -giver_shaft_start
        giver_shaft_length_squared = (
            giver_shaft_vector * giver_shaft_vector
        ).sum(dim=-1).clamp_min(1e-8)

        receiver_shaft_points = torch.stack(
            (receiver_ee_in_giver, receiver_jaw_proximal_in_giver),
            dim=1,
        )
        shaft_fraction = (
            (
                (
                    receiver_shaft_points
                    - giver_shaft_start.unsqueeze(1)
                )
                * giver_shaft_vector.unsqueeze(1)
            ).sum(dim=-1)
            / giver_shaft_length_squared.unsqueeze(-1)
        ).clamp(0.0, 1.0)
        closest_shaft_points = (
            giver_shaft_start.unsqueeze(1)
            + shaft_fraction.unsqueeze(-1)
            * giver_shaft_vector.unsqueeze(1)
        )
        receiver_shaft_deltas = (
            receiver_shaft_points - closest_shaft_points
        )
        receiver_shaft_distances = torch.linalg.vector_norm(
            receiver_shaft_deltas,
            dim=-1,
        )
        receiver_tip_shaft_distance = receiver_shaft_distances[:, 0]
        receiver_proximal_shaft_distance = receiver_shaft_distances[:, 1]
        proximal_is_closer = (
            receiver_proximal_shaft_distance
            < receiver_tip_shaft_distance
        )
        receiver_shaft_distance = torch.where(
            proximal_is_closer,
            receiver_proximal_shaft_distance,
            receiver_tip_shaft_distance,
        )
        receiver_from_shaft = torch.where(
            proximal_is_closer.unsqueeze(-1),
            receiver_shaft_deltas[:, 1],
            receiver_shaft_deltas[:, 0],
        )
        receiver_from_shaft_direction = (
            receiver_from_shaft
            / receiver_shaft_distance.clamp_min(1e-6).unsqueeze(-1)
        )
        receiver_shaft_radial_action = (
            receiver_approach[:, :3] * receiver_from_shaft_direction
        ).sum(dim=-1)
        maximum_receiver_shaft_inward_action = (
            (
                receiver_shaft_distance
                - self.recovery_receiver_shaft_guard_minimum_distance_m
            )
            / self.position_scale
        ).clamp_min(0.0)
        projected_receiver_shaft_radial_action = torch.maximum(
            receiver_shaft_radial_action,
            -maximum_receiver_shaft_inward_action,
        )
        receiver_shaft_guard_context = (
            pickup_recovery_context
            | torch.full_like(
                pickup_recovery_context,
                self.receiver_shaft_guard_all_pickups_enabled,
            )
        )
        recovery_receiver_shaft_guard_active = (
            receiver_shaft_guard_context
            & (phase == 2)
            & (
                receiver_shaft_distance
                < self.recovery_receiver_shaft_guard_activation_distance_m
            )
            & (
                receiver_shaft_radial_action
                < -maximum_receiver_shaft_inward_action
            )
        )
        receiver_shaft_correction = (
            projected_receiver_shaft_radial_action
            - receiver_shaft_radial_action
        ).unsqueeze(-1) * receiver_from_shaft_direction
        receiver_approach[:, :3] = torch.where(
            recovery_receiver_shaft_guard_active.unsqueeze(-1),
            receiver_approach[:, :3] + receiver_shaft_correction,
            receiver_approach[:, :3],
        )
        self.last_recovery_receiver_shaft_guard_active = (
            recovery_receiver_shaft_guard_active.detach()
        )
        self.last_recovery_receiver_shaft_distance_m = (
            receiver_shaft_distance.detach()
        )
        presentation_in_giver = (
            self.presentation_fraction_from_giver
            * root_2_in_giver
        )
        presentation_in_giver[:, 2] = (
            self.presentation_height_in_robot_frame
        )
        presentation_in_receiver = (
            presentation_in_giver - root_2_in_giver
        )
        if self.receiver_adaptive_arc_enabled:
            identity_orientation = torch.zeros_like(
                object_pose_in_receiver[:, 3:7]
            )
            identity_orientation[:, 3] = 1.0
            (
                _,
                _,
                _,
                receiver_future_grasp_position,
            ) = self._receiver_approach_action(
                receiver_ee,
                presentation_in_receiver,
                identity_orientation,
            )
        else:
            receiver_future_grasp_offset = torch.zeros_like(
                presentation_in_receiver
            )
            receiver_future_grasp_offset[:, 0] = self.receiver_grasp_x
            receiver_future_grasp_offset[:, 1] = self.receiver_grasp_y
            receiver_future_grasp_offset[:, 2] = self.receiver_grasp_z
            if self.canonical_needle_local_frames_enabled:
                receiver_future_grasp_offset = (
                    self._object_relative_offset(
                        object_pose_in_receiver[:, 3:7],
                        receiver_future_grasp_offset,
                    )
                )
            receiver_future_grasp_position = (
                presentation_in_receiver
                + receiver_future_grasp_offset
            )
        receiver_preposition_target = receiver_future_grasp_position.clone()
        receiver_preposition_height = torch.where(
            pickup_recovery_context,
            torch.full_like(
                receiver_distance,
                self.recovery_receiver_preposition_height,
            ),
            torch.full_like(
                receiver_distance,
                self.receiver_preposition_height,
            ),
        )
        receiver_preposition_target[:, 2] += receiver_preposition_height
        receiver_preposition = (
            (receiver_preposition_target - receiver_ee)
            / self.position_scale
        ).clamp(
            -self.receiver_preposition_action_limit,
            self.receiver_preposition_action_limit,
        )
        giver_target = presentation_in_giver.clone()
        vertical_only = (
            object_in_giver[:, 2]
            < self.minimum_lift_height_in_robot_frame
        )
        giver_target[:, :2] = torch.where(
            vertical_only.unsqueeze(-1),
            object_in_giver[:, :2],
            giver_target[:, :2],
        )
        giver_error = (
            giver_target - object_in_giver
        ) / self.position_scale
        pickup_progress = (
            (
                object_in_giver[:, 2]
                - (
                    self.minimum_lift_height_in_robot_frame
                    - self.pickup_deceleration_height
                )
            )
            / self.pickup_deceleration_height
        ).clamp(0.0, 1.0)
        pickup_deceleration_fraction = (
            pickup_progress
            * pickup_progress
            * (3.0 - 2.0 * pickup_progress)
        )
        pickup_vertical_limit = (
            self.pickup_initial_vertical_action_limit
            + (
                self.pickup_vertical_action_limit
                - self.pickup_initial_vertical_action_limit
            )
            * pickup_deceleration_fraction
        )
        pickup_vertical_limit = torch.where(
            pickup_recovery_context,
            torch.full_like(
                pickup_vertical_limit,
                self.recovery_pickup_vertical_action_limit,
            ),
            pickup_vertical_limit,
        )
        giver_vertical_limit = torch.where(
            vertical_only,
            pickup_vertical_limit,
            torch.full_like(
                giver_error[:, 2],
                self.carry_vertical_action_limit,
            ),
        ).unsqueeze(-1)
        giver_vertical_action = torch.maximum(
            torch.minimum(
                giver_error[:, 2:],
                giver_vertical_limit,
            ),
            -giver_vertical_limit,
        )
        carry_ramp_fraction = (
            (
                object_in_giver[:, 2]
                - self.minimum_lift_height_in_robot_frame
            )
            / self.carry_lateral_ramp_height
        ).clamp(0.0, 1.0)
        carry_ramp_fraction = carry_ramp_fraction * carry_ramp_fraction * (
            3.0 - 2.0 * carry_ramp_fraction
        )
        giver_bilateral_contact = torch.all(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        contact_reference = max(
            self.normalized_contact_threshold,
            (
                self.giver_lift_contact_force_threshold_n
                * self.contact_force_observation_scale
            ),
        )
        minimum_contact_confidence = (
            torch.amin(giver_contacts, dim=-1) / contact_reference
        ).clamp(0.0, 1.0)
        contact_balance = (
            1.0
            - torch.abs(giver_contacts[:, 0] - giver_contacts[:, 1])
            / giver_contacts.sum(dim=-1).clamp_min(contact_reference)
        ).clamp(0.0, 1.0)
        minimum_contact_trend = torch.amin(
            giver_contacts - previous_giver_contacts,
            dim=-1,
        )
        contact_trend_quality = (
            1.0 + minimum_contact_trend / contact_reference
        ).clamp(0.0, 1.0)
        object_linear_speed = torch.linalg.vector_norm(
            raw[:, 60:63],
            dim=-1,
        )
        object_angular_speed = torch.linalg.vector_norm(
            raw[:, 63:66],
            dim=-1,
        )
        motion_quality = (
            1.0
            - 0.5 * torch.tanh(object_linear_speed / 0.05)
            - 0.5 * torch.tanh(object_angular_speed / 5.0)
        ).clamp(0.0, 1.0)
        giver_custody_quality = (
            0.45 * minimum_contact_confidence
            + 0.20 * contact_balance
            + 0.20 * contact_trend_quality
            + 0.15 * motion_quality
        ).clamp(0.0, 1.0)
        if not self.custody_quality_features_enabled:
            giver_custody_quality = torch.ones_like(
                giver_custody_quality
            )
        quality_span = max(
            self.custody_quality_slow_threshold
            - self.custody_quality_stop_threshold,
            1.0e-6,
        )
        custody_transport_scale = (
            (
                giver_custody_quality
                - self.custody_quality_stop_threshold
            )
            / quality_span
        ).clamp(
            self.custody_quality_minimum_transport_scale,
            1.0,
        )
        custody_transport_scale = torch.where(
            giver_custody_quality
            >= self.custody_quality_slow_threshold,
            torch.ones_like(custody_transport_scale),
            custody_transport_scale,
        )
        recovery_transport_qualified = (
            pickup_recovery_context
            & (phase >= 2)
            & giver_bilateral_contact
        )
        recovery_lateral_action_limit = torch.where(
            recovery_transport_qualified,
            torch.full_like(
                carry_ramp_fraction,
                self.recovery_carry_lateral_action_limit,
            ),
            torch.full_like(
                carry_ramp_fraction,
                self.carry_lateral_action_limit,
            ),
        )
        carry_lateral_action_limit = torch.where(
            pickup_recovery_context,
            recovery_lateral_action_limit,
            torch.full_like(
                carry_ramp_fraction,
                self.carry_lateral_action_limit,
            ),
        )
        carry_lateral_limit = (
            carry_lateral_action_limit * carry_ramp_fraction
        ).unsqueeze(-1)
        giver_lateral_action = torch.maximum(
            torch.minimum(
                giver_error[:, :2],
                carry_lateral_limit,
            ),
            -carry_lateral_limit,
        )
        giver_carry = torch.cat(
            (
                giver_lateral_action,
                giver_vertical_action,
            ),
            dim=-1,
        )
        if self.custody_preserving_transport_enabled:
            giver_carry = (
                giver_carry
                * custody_transport_scale.unsqueeze(-1)
            )
            giver_grasp_target = (
                object_in_giver
                + giver_object_relative_pregrasp_offset
            )
            giver_centering_error = (
                (giver_grasp_target - giver_ee) / self.position_scale
            ).clamp(
                -self.custody_quality_centering_action_limit,
                self.custody_quality_centering_action_limit,
            )
            giver_carry[:, :2] += (
                giver_centering_error[:, :2]
                * (1.0 - custody_transport_scale).unsqueeze(-1)
            )
        else:
            custody_transport_scale = torch.ones_like(
                custody_transport_scale
            )
        self.last_giver_custody_quality = (
            giver_custody_quality.detach()
        )
        self.last_giver_transport_scale = (
            custody_transport_scale.detach()
        )
        giver_lift_contact_qualified = torch.all(
            giver_contacts
            > (
                self.giver_lift_contact_force_threshold_n
                * self.contact_force_observation_scale
            ),
            dim=-1,
        )
        giver_any_contact = torch.any(
            giver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        giver_pre_lift_contact = torch.where(
            torch.full_like(
                giver_any_contact,
                self.giver_pre_lift_min_contact_jaws == 1,
            ),
            giver_any_contact,
            giver_lift_contact_qualified,
        )
        receiver_any_contact = torch.any(
            receiver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        receiver_bilateral_contact = torch.all(
            receiver_contacts > self.normalized_contact_threshold,
            dim=-1,
        )
        phase_zero_lift_enabled = torch.full_like(
            phase,
            self.giver_lift_on_live_contact,
            dtype=torch.bool,
        )
        giver_carry_mode = (
            ((phase >= 1) & (phase <= 2))
            | ((phase == 0) & phase_zero_lift_enabled)
        )
        # Phase 1 is entered only after filtered bilateral giver contact. Keep
        # lifting through a brief contact-sensor flicker so the action policy
        # agrees with the state's pickup_contact_loss_steps debounce. A real
        # loss still moves the episode to phase 4, where recovery takes over.
        giver_pre_lift_transport_ready = (
            (phase == 1) | giver_pre_lift_contact
        )
        phase_two_custody = torch.where(
            torch.full_like(
                giver_bilateral_contact,
                self.transport_custody_latch_enabled,
            ),
            phase == 2,
            giver_bilateral_contact,
        )
        giver_transport_active = giver_carry_mode & torch.where(
            phase <= 1,
            giver_pre_lift_transport_ready,
            phase_two_custody,
        )
        receiver_approach_active = (
            (phase == 2)
            & presentation_stable
            & phase_two_custody
            & ~receiver_any_contact
            & ~receiver_retry_active
        )
        receiver_preposition_active = (
            torch.full_like(
                phase,
                self.receiver_preposition_enabled,
                dtype=torch.bool,
            )
            & (
                (phase <= 1)
                | (
                    (phase == 2)
                    & ~presentation_stable
                    & phase_two_custody
                )
            )
        )
        giver_translation = torch.where(
            giver_transport_active.unsqueeze(-1),
            giver_carry,
            giver_approach,
        )
        giver_presentation_hold = giver_carry.clamp(
            -self.presentation_hold_action_limit,
            self.presentation_hold_action_limit,
        )
        giver_translation = torch.where(
            (
                (phase == 2)
                & phase_two_custody
                & presentation_stable
                & ~receiver_any_contact
            ).unsqueeze(-1),
            giver_presentation_hold,
            giver_translation,
        )
        giver_translation = torch.where(
            (
                (phase == 2)
                & phase_two_custody
                & receiver_any_contact
            ).unsqueeze(-1),
            torch.zeros_like(giver_translation),
            giver_translation,
        )
        giver_retreat = torch.zeros_like(giver_translation)
        giver_retreat[:, 2] = self.carry_lateral_action_limit
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
        giver_recovery_translation = (
            (giver_pregrasp_position - giver_ee)
            / self.position_scale
        ).clamp(
            -self.slow_approach_action_limit,
            self.slow_approach_action_limit,
        )
        giver_translation = torch.where(
            (phase == 4).unsqueeze(-1),
            giver_recovery_translation,
            giver_translation,
        )

        receiver_translation = torch.where(
            receiver_approach_active.unsqueeze(-1),
            receiver_approach,
            torch.where(
                receiver_preposition_active.unsqueeze(-1),
                receiver_preposition,
                torch.zeros_like(receiver_approach),
            ),
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
        receiver_retry_translation[:, 2] = (
            self.slow_approach_action_limit
        )
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
        receiver_contact_centering = (
            torch.sign(receiver_contact_imbalance)
            * self.receiver_contact_centering_action_limit
        )
        receiver_translation[:, 2] += torch.where(
            (
                (phase == 2)
                & phase_two_custody
                & receiver_any_contact
                & ~receiver_retry_active
            ),
            receiver_contact_centering,
            torch.zeros_like(receiver_contact_centering),
        )

        giver_closing = (
            (
                (
                    (giver_distance < self.close_distance)
                    & giver_pregrasp_orientation_ready
                )
                | giver_any_contact
                | ((phase >= 1) & (phase <= 2))
            )
            & (phase < 3)
        )
        giver_closing |= (phase == 3) & ~receiver_bilateral_contact
        receiver_closing = (
            ((phase == 2) | (phase == 3))
            & ~receiver_retry_active
            & (
                (
                    receiver_distance
                    < self.receiver_close_distance
                )
                | receiver_any_contact
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

        giver_object_orientation = object_pose_in_giver[:, 3:7]
        giver_object_angular_velocity = raw[:, 63:66]
        if self.canonical_needle_local_frames_enabled:
            # Preserve the yaw used to acquire the needle. Relative IK holds
            # the live tool frame when its angular command is zero; matching
            # the target to the measured object quaternion makes the
            # proportional term exactly zero while retaining bounded angular
            # velocity damping. Using the live tool quaternion as the target
            # here would instead feed the tool-object grasp offset back as a
            # spurious rotation command.
            giver_target_orientation = (
                giver_object_orientation.detach()
            )
        else:
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
            / self.orientation_scale
        ).clamp(
            -self.giver_transport_orientation_action_limit,
            self.giver_transport_orientation_action_limit,
        )
        giver_orientation_action = torch.where(
            giver_transport_active.unsqueeze(-1),
            giver_orientation_action,
            torch.zeros_like(giver_orientation_action),
        )
        giver_orientation_action = torch.where(
            (
                (phase == 0)
                & ~giver_any_contact
            ).unsqueeze(-1),
            giver_pregrasp_orientation_action,
            giver_orientation_action,
        )
        giver_orientation_action = torch.where(
            (phase == 4).unsqueeze(-1),
            giver_pregrasp_orientation_action,
            giver_orientation_action,
        )

        # The giver and receiver grasp different points on a curved needle.
        # Express receiver roll as local tangent change plus a physics-
        # calibrated crossing angle. Zero crossing is parallel to the needle
        # and was rejected because the needle slipped after release; the
        # baseline crossing keeps the prior pi roll until a matched sweep
        # identifies a better contact-retaining jaw angle.
        receiver_roll = torch.zeros_like(giver_orientation)
        selected_tangent_delta = (
            receiver_arc_fraction - self.needle_provisional_arc_fraction
        ) * self.needle_arc_extent_rad
        selected_roll_offset = (
            selected_tangent_delta + self.receiver_crossing_angle_rad
        )
        receiver_half_roll_offset = 0.5 * selected_roll_offset
        receiver_roll[:, 2] = torch.sin(receiver_half_roll_offset)
        receiver_roll[:, 3] = torch.cos(receiver_half_roll_offset)
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
            receiver_orientation_error / self.orientation_scale
        ).clamp(
            -self.receiver_orientation_action_limit,
            self.receiver_orientation_action_limit,
        )
        receiver_orientation_error_norm = torch.linalg.vector_norm(
            receiver_orientation_error,
            dim=-1,
        )
        if self.receiver_preposition_enabled:
            # Calibrate the pre-contact jaw angle from the retained v33
            # population, then hold it through the final approach.  Fully
            # aligning to the pi-roll target before contact produced bilateral
            # acquisition but post-release slip.
            receiver_orientation_active = (
                (receiver_preposition_active | receiver_approach_active)
                & (
                    receiver_orientation_error_norm
                    > self.receiver_contact_orientation_error_target_rad
                )
            )
        else:
            receiver_orientation_active = receiver_approach_active
        receiver_orientation_action = torch.where(
            receiver_orientation_active.unsqueeze(-1),
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
        base_action = torch.cat(
            (robot_1_action, robot_2_action),
            dim=-1,
        ).clamp(-1.0, 1.0)

        giver_residual = torch.zeros_like(giver_action)
        giver_pickup_transport_residual = (
            (phase >= 1)
            & (phase <= 2)
            & torch.where(
                phase == 2,
                phase_two_custody,
                giver_bilateral_contact,
            )
            & ~receiver_any_contact
        )
        giver_pickup_transport_residual &= torch.where(
            torch.full_like(
                pickup_recovery_context,
                self.giver_recovery_residual_only_for_learning,
            ),
            pickup_recovery_context,
            torch.ones_like(pickup_recovery_context),
        )
        giver_recovery_approach_residual = (
            pickup_recovery_context
            & ((phase == 0) | (phase == 4))
            & torch.full_like(
                pickup_recovery_context,
                self.giver_recovery_residual_only_for_learning,
            )
        )
        # The analytic controller remains the sole authority for vertical
        # lift.  A learned correction may center the grasp in the table plane,
        # but cannot reverse or accelerate the qualified pickup trajectory.
        giver_residual[:, :2] = (
            (
                giver_pickup_transport_residual
                | giver_recovery_approach_residual
            ).unsqueeze(-1)
        )
        receiver_residual = torch.zeros_like(receiver_action)
        receiver_residual_enabled = (
            receiver_approach_active
            & self.receiver_residual_enabled_for_learning
        )
        receiver_residual[:, :3] = receiver_residual_enabled.unsqueeze(-1)
        receiver_grasp_retain_residual_enabled = torch.zeros_like(
            receiver_approach_active
        )
        if self.receiver_grasp_retain_residual_enabled_for_learning:
            receiver_grasp_retain_residual_enabled = (
                receiver_approach_active
                | (
                    (phase == 2)
                    & presentation_stable
                    & ~receiver_retry_active
                )
                | (phase == 3)
            )
        receiver_residual[:, :6] = torch.where(
            receiver_grasp_retain_residual_enabled.unsqueeze(-1),
            torch.ones_like(receiver_residual[:, :6]),
            receiver_residual[:, :6],
        )
        no_giver_residual = torch.zeros_like(giver_residual)
        no_receiver_residual = torch.zeros_like(receiver_residual)
        robot_1_giver_residual = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            giver_residual,
            no_giver_residual,
        )
        robot_2_giver_residual = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            no_giver_residual,
            giver_residual,
        )
        robot_1_receiver_residual = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            no_receiver_residual,
            receiver_residual,
        )
        robot_2_receiver_residual = torch.where(
            giver_is_robot_1.unsqueeze(-1),
            receiver_residual,
            no_receiver_residual,
        )
        giver_residual_mask = torch.cat(
            (robot_1_giver_residual, robot_2_giver_residual),
            dim=-1,
        ) > 0.5
        receiver_residual_mask = torch.cat(
            (
                robot_1_receiver_residual,
                robot_2_receiver_residual,
            ),
            dim=-1,
        ) > 0.5
        return (
            base_action,
            giver_residual_mask,
            receiver_residual_mask,
        )


class HandoverResidualMLPModel(MLPModel):
    """Learn only bounded giver XY corrections around the physical sequence."""

    def __init__(
        self,
        *args,
        residual_scale: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.controller = HandoverAnalyticController()
        self.residual_scale = residual_scale
        final_linear = next(
            module
            for module in reversed(self.mlp)
            if isinstance(module, nn.Linear)
        )
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        if self.distribution is not None:
            for parameter_name in ("std_param", "log_std_param"):
                parameter = getattr(
                    self.distribution,
                    parameter_name,
                    None,
                )
                if parameter is not None:
                    parameter.requires_grad_(False)

    def configure_giver_adaptation(self) -> None:
        """Adapt shared features and only the zero-influence giver XY rows."""
        self.controller.receiver_residual_enabled_for_learning = False
        final_linear = next(
            module
            for module in reversed(self.mlp)
            if isinstance(module, nn.Linear)
        )
        # The previous contract froze the randomly initialized feature
        # extractor and reduced PPO to a linear probe over random features.
        # Shared features may now adapt, while gradient masking below keeps
        # every non-giver output row inert.
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(True)
        giver_row_mask = torch.zeros(
            final_linear.out_features,
            dtype=final_linear.weight.dtype,
            device=final_linear.weight.device,
        )
        giver_row_mask[3:5] = 1.0
        giver_row_mask[10:12] = 1.0
        final_linear.weight.register_hook(
            lambda gradient: gradient
            * giver_row_mask.unsqueeze(-1)
        )
        final_linear.bias.register_hook(
            lambda gradient: gradient * giver_row_mask
        )

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        raw = torch.cat(
            [obs[group] for group in self.obs_groups],
            dim=-1,
        )
        (
            base,
            giver_residual_mask,
            receiver_residual_mask,
        ) = self.controller(raw)
        network_output = torch.tanh(self.mlp(latent))
        giver_channel_output = torch.cat(
            (
                network_output[:, 3:5],
                torch.zeros_like(network_output[:, 2:7]),
                network_output[:, 10:12],
                torch.zeros_like(network_output[:, 9:14]),
            ),
            dim=-1,
        )
        residual = self.residual_scale * (
            giver_channel_output
            * giver_residual_mask.to(raw.dtype)
            + network_output
            * receiver_residual_mask.to(raw.dtype)
        )
        residual_mask = giver_residual_mask | receiver_residual_mask
        mean = (base + residual).clamp(-1.0, 1.0)
        if self.distribution is None:
            return mean
        if stochastic_output:
            self.distribution.update(mean)
            sampled = self.distribution.sample()
            return torch.where(residual_mask, sampled, mean)
        return self.distribution.deterministic_output(mean)

    def as_jit(self) -> nn.Module:
        return _HandoverResidualExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _HandoverResidualOnnxExport(self, verbose)


class _HandoverResidualExport(nn.Module):
    """TorchScript-compatible deterministic handover policy."""

    def __init__(self, model: HandoverResidualMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.controller = copy.deepcopy(model.controller)
        self.residual_scale = model.residual_scale
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        (
            base,
            giver_residual_mask,
            receiver_residual_mask,
        ) = self.controller(obs)
        network_output = torch.tanh(
            self.mlp(self.obs_normalizer(obs))
        )
        giver_channel_output = torch.cat(
            (
                network_output[:, 3:5],
                torch.zeros_like(network_output[:, 2:7]),
                network_output[:, 10:12],
                torch.zeros_like(network_output[:, 9:14]),
            ),
            dim=-1,
        )
        residual = self.residual_scale * (
            giver_channel_output
            * giver_residual_mask.to(obs.dtype)
            + network_output
            * receiver_residual_mask.to(obs.dtype)
        )
        return self.deterministic_output(
            (base + residual).clamp(-1.0, 1.0)
        )

    @torch.jit.export
    def reset(self) -> None:
        pass


class _HandoverResidualOnnxExport(_HandoverResidualExport):
    """ONNX metadata for the deterministic residual handover policy."""

    is_recurrent: bool = False

    def __init__(
        self,
        model: HandoverResidualMLPModel,
        verbose: bool,
    ) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
