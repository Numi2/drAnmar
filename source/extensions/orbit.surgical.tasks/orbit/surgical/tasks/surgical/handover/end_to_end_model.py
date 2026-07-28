# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Role-normalized end-to-end actor for the experimental needle handover."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution
from torch import nn
from torch.distributions import Normal

from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
)

from orbit.surgical.tasks.surgical.lift.grasp_frames import (
    needle_geometry_grasp_frame,
)

from .residual_model import HandoverAnalyticController

_TASK_FEATURE_DIM = 24
_RECOVERY_RECEIVER_ADAPTER_FEATURE_DIM = 12
_JOINT_TRANSFER_ACQUISITION_FEATURE_DIM = 31
_TRANSFER_REFINEMENT_FEATURE_DIM = 31
_RECEIVER_POLICY_GRASP_OFFSET, _ = needle_geometry_grasp_frame(
    0.65,
    grasp_z_m=-0.003,
)


def select_handover_role(
    robot_1_value: torch.Tensor,
    robot_2_value: torch.Tensor,
    use_robot_1: torch.Tensor,
) -> torch.Tensor:
    """Select a physical-arm tensor using the episode's role assignment."""
    return torch.where(
        use_robot_1.unsqueeze(-1),
        robot_1_value,
        robot_2_value,
    )


def role_normalize_handover_observation(raw: torch.Tensor) -> torch.Tensor:
    """Express both arms in giver/receiver order without hiding physical state."""
    giver_is_robot_1 = raw[:, 82] > 0.5

    giver_joint_position = select_handover_role(
        raw[:, 0:8], raw[:, 16:24], giver_is_robot_1
    )
    giver_joint_velocity = select_handover_role(
        raw[:, 8:16], raw[:, 24:32], giver_is_robot_1
    )
    receiver_joint_position = select_handover_role(
        raw[:, 0:8], raw[:, 16:24], ~giver_is_robot_1
    )
    receiver_joint_velocity = select_handover_role(
        raw[:, 8:16], raw[:, 24:32], ~giver_is_robot_1
    )
    giver_end_effector = select_handover_role(
        raw[:, 32:39], raw[:, 39:46], giver_is_robot_1
    )
    receiver_end_effector = select_handover_role(
        raw[:, 32:39], raw[:, 39:46], ~giver_is_robot_1
    )
    object_in_giver = select_handover_role(
        raw[:, 46:53], raw[:, 53:60], giver_is_robot_1
    )
    object_in_receiver = select_handover_role(
        raw[:, 46:53], raw[:, 53:60], ~giver_is_robot_1
    )
    giver_contacts = select_handover_role(
        raw[:, 66:68], raw[:, 68:70], giver_is_robot_1
    )
    receiver_contacts = select_handover_role(
        raw[:, 66:68], raw[:, 68:70], ~giver_is_robot_1
    )
    giver_last_action = select_handover_role(
        raw[:, 84:91], raw[:, 91:98], giver_is_robot_1
    )
    receiver_last_action = select_handover_role(
        raw[:, 84:91], raw[:, 91:98], ~giver_is_robot_1
    )
    giver_contact_history = select_handover_role(
        raw[:, 99:101], raw[:, 101:103], giver_is_robot_1
    )
    receiver_contact_history = select_handover_role(
        raw[:, 99:101], raw[:, 101:103], ~giver_is_robot_1
    )

    canonical_identity = torch.zeros_like(raw[:, 82:84])
    canonical_identity[:, 0] = 1.0
    return torch.cat(
        (
            giver_joint_position,
            giver_joint_velocity,
            receiver_joint_position,
            receiver_joint_velocity,
            giver_end_effector,
            receiver_end_effector,
            object_in_giver,
            object_in_receiver,
            raw[:, 60:66],
            giver_contacts,
            receiver_contacts,
            raw[:, 70:77],
            raw[:, 77:82],
            canonical_identity,
            giver_last_action,
            receiver_last_action,
            raw[:, 98:99],
            giver_contact_history,
            receiver_contact_history,
            raw[:, 103:107],
        ),
        dim=-1,
    )


def role_action_to_physical(
    role_action: torch.Tensor,
    giver_is_robot_1: torch.Tensor,
) -> torch.Tensor:
    """Map canonical giver/receiver actions back to Robot 1/Robot 2 order."""
    giver_action = role_action[:, 0:7]
    receiver_action = role_action[:, 7:14]
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
    return torch.cat((robot_1_action, robot_2_action), dim=-1)


def handover_task_features(
    role_observation: torch.Tensor,
    receiver_policy_grasp_offset: torch.Tensor,
) -> torch.Tensor:
    """Build local grasp, presentation, orientation, and contact-change features."""
    giver_ee_position = role_observation[:, 32:35]
    giver_ee_orientation = role_observation[:, 35:39]
    receiver_ee_position = role_observation[:, 39:42]
    receiver_ee_orientation = role_observation[:, 42:46]
    object_in_giver = role_observation[:, 46:53]
    object_in_receiver = role_observation[:, 53:60]

    giver_offset = torch.zeros_like(giver_ee_position)
    giver_offset[:, 0] = 0.0007375535249017802
    giver_offset[:, 1] = 0.005600696415109648
    giver_offset[:, 2] = 0.0006
    object_quaternion = object_in_giver[:, 3:7]
    yaw_sine = 2.0 * (
        object_quaternion[:, 3] * object_quaternion[:, 2]
        + object_quaternion[:, 0] * object_quaternion[:, 1]
    )
    yaw_cosine = 1.0 - 2.0 * (
        object_quaternion[:, 1] * object_quaternion[:, 1]
        + object_quaternion[:, 2] * object_quaternion[:, 2]
    )
    rotated_giver_offset = giver_offset.clone()
    rotated_giver_offset[:, 0] = (
        yaw_cosine * giver_offset[:, 0]
        - yaw_sine * giver_offset[:, 1]
    )
    rotated_giver_offset[:, 1] = (
        yaw_sine * giver_offset[:, 0]
        + yaw_cosine * giver_offset[:, 1]
    )
    recovery_context = role_observation[:, 98] > 0.5
    giver_offset = torch.where(
        recovery_context.unsqueeze(-1),
        rotated_giver_offset,
        giver_offset,
    )
    giver_grasp_error = (
        object_in_giver[:, :3] + giver_offset - giver_ee_position
    ) / 0.02

    receiver_offset = torch.zeros_like(receiver_ee_position)
    receiver_offset += receiver_policy_grasp_offset
    receiver_grasp_error = (
        object_in_receiver[:, :3] + receiver_offset - receiver_ee_position
    ) / 0.02

    root_2_in_giver = object_in_giver[:, :3] - object_in_receiver[:, :3]
    presentation_target = 0.35 * root_2_in_giver
    presentation_target[:, 2] = -0.13
    presentation_error = (
        presentation_target - object_in_giver[:, :3]
    ) / 0.05

    identity_orientation = torch.zeros_like(giver_ee_orientation)
    identity_orientation[:, 3] = 1.0
    giver_orientation_error = axis_angle_from_quat(
        quat_mul(
            identity_orientation,
            quat_conjugate(giver_ee_orientation),
        )
    ) / 3.141592653589793
    receiver_roll = torch.zeros_like(receiver_ee_orientation)
    receiver_roll[:, 2] = 1.0
    receiver_target_orientation = quat_mul(
        receiver_roll,
        giver_ee_orientation,
    )
    receiver_orientation_error = axis_angle_from_quat(
        quat_mul(
            receiver_target_orientation,
            quat_conjugate(receiver_ee_orientation),
        )
    ) / 3.141592653589793

    pickup_clearance = (
        (object_in_giver[:, 2:3] + 0.139) / 0.02
    ).clamp(-5.0, 5.0)
    contact_change = (
        role_observation[:, 66:70] - role_observation[:, 99:103]
    ).clamp(-5.0, 5.0)
    transfer_contract = role_observation[:, 103:107]
    return torch.cat(
        (
            giver_grasp_error,
            receiver_grasp_error,
            presentation_error,
            giver_orientation_error,
            receiver_orientation_error,
            pickup_clearance,
            contact_change,
            transfer_contract,
        ),
        dim=-1,
    ).clamp(-5.0, 5.0)


def recovery_receiver_canonical_grasp_features(
    role_observation: torch.Tensor,
    receiver_policy_grasp_offset: torch.Tensor,
) -> torch.Tensor:
    """Describe the receiver target in the physically observed needle frame."""

    receiver_ee_position = role_observation[:, 39:42]
    receiver_ee_orientation = role_observation[:, 42:46]
    object_in_receiver = role_observation[:, 53:60]
    receiver_offset = torch.zeros_like(receiver_ee_position)
    receiver_offset += receiver_policy_grasp_offset
    receiver_target_position = (
        object_in_receiver[:, :3]
        + quat_apply(object_in_receiver[:, 3:7], receiver_offset)
    )
    receiver_position_error = (
        receiver_target_position - receiver_ee_position
    ) / 0.02

    receiver_roll = torch.zeros_like(receiver_ee_orientation)
    receiver_roll[:, 2] = 1.0
    receiver_target_orientation = quat_mul(
        object_in_receiver[:, 3:7],
        receiver_roll,
    )
    receiver_orientation_error = axis_angle_from_quat(
        quat_mul(
            receiver_target_orientation,
            quat_conjugate(receiver_ee_orientation),
        )
    ) / 3.141592653589793
    receiver_contacts = role_observation[:, 68:70]
    transfer_contract = role_observation[:, 103:107]
    return torch.cat(
        (
            receiver_position_error,
            receiver_orientation_error,
            receiver_contacts,
            transfer_contract,
        ),
        dim=-1,
    ).clamp(-5.0, 5.0)


def joint_transfer_acquisition_features(
    role_observation: torch.Tensor,
    receiver_policy_grasp_offset: torch.Tensor,
) -> torch.Tensor:
    """Describe coupled giver stabilization and receiver acquisition.

    The compact option sees the needle-local receiver grasp error, the
    presentation error, object twist, giver contact confidence/history, and
    both tools' previous Cartesian actions. It cannot command either gripper;
    release remains owned by the physics-derived capture contract.
    """
    task_features = handover_task_features(
        role_observation,
        receiver_policy_grasp_offset,
    )
    return torch.cat(
        (
            recovery_receiver_canonical_grasp_features(
                role_observation,
                receiver_policy_grasp_offset,
            ),
            task_features[:, 6:9],
            role_observation[:, 60:66],
            role_observation[:, 66:68],
            role_observation[:, 99:101],
            role_observation[:, 84:87],
            role_observation[:, 91:94],
        ),
        dim=-1,
    ).clamp(-5.0, 5.0)


class PhaseMaskedGaussianDistribution(GaussianDistribution):
    """Gaussian exploration with near-zero variance on structurally inactive actions."""

    def __init__(self, *args, inactive_std: float = 1.0e-6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inactive_std = inactive_std
        self._action_mask: torch.Tensor | None = None

    def set_action_mask(self, action_mask: torch.Tensor) -> None:
        self._action_mask = action_mask

    def update(self, mlp_output: torch.Tensor) -> None:
        super().update(mlp_output)
        if self._action_mask is None:
            return
        active_std = self._distribution.stddev
        masked_std = torch.where(
            self._action_mask,
            active_std,
            torch.full_like(mlp_output, self.inactive_std),
        )
        self._distribution = Normal(mlp_output, masked_std)


class _PhaseHeadedNetwork(nn.Module):
    """Shared geometric encoder with a separate motor head for each physical phase."""

    def __init__(self, trunk: nn.Module, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.heads = nn.ModuleList(
            nn.Linear(input_dim, output_dim) for _ in range(5)
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        latent: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.trunk(latent)
        candidates = torch.stack(
            [head(encoded) for head in self.heads],
            dim=1,
        )
        batch_indices = torch.arange(
            encoded.shape[0],
            device=encoded.device,
        )
        return candidates[batch_indices, phase]


class _RecoveryReceiverAdapter(nn.Module):
    """Zero-initialized bounded correction from needle-local grasp error."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(_RECOVERY_RECEIVER_ADAPTER_FEATURE_DIM, 64),
            nn.ELU(),
        )
        self.output = nn.Linear(64, 6)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.output(self.encoder(features)))


class _JointTransferAcquisitionAdapter(nn.Module):
    """Zero-impact coupled SE(3) residual for the two tools."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(_JOINT_TRANSFER_ACQUISITION_FEATURE_DIM, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.output = nn.Linear(64, 12)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.output(self.encoder(features)))


class _TransferRefinementAdapter(nn.Module):
    """Zero-impact residual that refines acquisition and receiver retention."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(_TRANSFER_REFINEMENT_FEATURE_DIM, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.output = nn.Linear(64, 12)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.output(self.encoder(features)))


class EndToEndHandoverMLPModel(MLPModel):
    """Physics-structured servo plus bounded phase-specialized learned residual."""

    def __init__(
        self,
        *args,
        residual_scale: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        built_mlp = self.mlp
        modules = list(built_mlp)
        final_linear = modules[-1]
        if not isinstance(final_linear, nn.Linear):
            raise TypeError("end-to-end handover actor requires a linear output")
        trunk = nn.Sequential(*modules[:-1])
        del self.mlp
        self.phase_network = _PhaseHeadedNetwork(
            trunk,
            final_linear.in_features,
            final_linear.out_features,
        )
        self.controller = HandoverAnalyticController()
        # This actor's qualified default is receiver-only residual learning.
        # The legacy handover actor disables receiver residuals by default so
        # giver-adaptation checkpoints cannot silently change at serving time.
        self.controller.receiver_residual_enabled_for_learning = True
        self.residual_scale = residual_scale
        self.giver_adaptation_enabled = False
        self.pickup_recovery_adaptation_enabled = False
        self.receiver_adaptation_enabled = False
        self.recovery_receiver_grasp_retain_adaptation_enabled = False
        self.joint_transfer_acquisition_adaptation_enabled = False
        self.transfer_refinement_adaptation_enabled = False
        self.register_buffer(
            "receiver_policy_grasp_offset",
            torch.tensor(_RECEIVER_POLICY_GRASP_OFFSET),
            persistent=False,
        )
        self.recovery_receiver_adapter = _RecoveryReceiverAdapter()
        for parameter in self.recovery_receiver_adapter.parameters():
            parameter.requires_grad_(False)
        self.joint_transfer_acquisition_adapter = (
            _JointTransferAcquisitionAdapter()
        )
        for parameter in self.joint_transfer_acquisition_adapter.parameters():
            parameter.requires_grad_(False)
        self.transfer_refinement_adapter = _TransferRefinementAdapter()
        for parameter in self.transfer_refinement_adapter.parameters():
            parameter.requires_grad_(False)
        self.recovery_receiver_reference_network: (
            _PhaseHeadedNetwork | None
        ) = None

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load older option checkpoints while preserving an exact-zero adapter."""
        adapter_prefix = "recovery_receiver_adapter."
        if not any(key.startswith(adapter_prefix) for key in state_dict):
            state_dict = state_dict.copy()
            for key, value in self.recovery_receiver_adapter.state_dict().items():
                state_dict[f"{adapter_prefix}{key}"] = value
        joint_adapter_prefix = "joint_transfer_acquisition_adapter."
        if not any(
            key.startswith(joint_adapter_prefix) for key in state_dict
        ):
            state_dict = state_dict.copy()
            for key, value in (
                self.joint_transfer_acquisition_adapter.state_dict().items()
            ):
                state_dict[f"{joint_adapter_prefix}{key}"] = value
        refinement_adapter_prefix = "transfer_refinement_adapter."
        if not any(
            key.startswith(refinement_adapter_prefix) for key in state_dict
        ):
            state_dict = state_dict.copy()
            for key, value in self.transfer_refinement_adapter.state_dict().items():
                state_dict[f"{refinement_adapter_prefix}{key}"] = value
        if (
            self.recovery_receiver_reference_network is None
            and any(
                key.startswith("recovery_receiver_reference_network.")
                for key in state_dict
            )
        ):
            self.recovery_receiver_reference_network = copy.deepcopy(
                self.phase_network
            )
        return super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

    def _get_latent_dim(self) -> int:
        return self.obs_dim + _TASK_FEATURE_DIM

    def _role_latent(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        return role_normalize_handover_observation(raw), raw[:, 82] > 0.5

    def get_latent(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
    ) -> torch.Tensor:
        del masks, hidden_state
        role_observation, _ = self._role_latent(obs)
        normalized = self.obs_normalizer(role_observation)
        return torch.cat(
            (
                normalized,
                handover_task_features(
                    role_observation,
                    self.receiver_policy_grasp_offset,
                ),
            ),
            dim=-1,
        )

    def update_normalization(self, obs) -> None:
        if (
            self.obs_normalization
            and not self.giver_adaptation_enabled
            and not self.pickup_recovery_adaptation_enabled
            and not self.receiver_adaptation_enabled
            and not self.joint_transfer_acquisition_adaptation_enabled
            and not self.transfer_refinement_adaptation_enabled
        ):
            role_observation, _ = self._role_latent(obs)
            self.obs_normalizer.update(role_observation)

    def configure_receiver_adaptation(self) -> None:
        """Adapt only receiver XYZ in the acquisition phase.

        The promoted pickup, lift, presentation representation, observation
        normalization, and every non-acquisition motor row remain immutable.
        """
        self.receiver_adaptation_enabled = True
        for parameter in self.phase_network.parameters():
            parameter.requires_grad_(False)
        receiver_head = self.phase_network.heads[2]
        receiver_head.weight.requires_grad_(True)
        receiver_head.bias.requires_grad_(True)
        receiver_xyz_row_mask = torch.zeros(
            14,
            dtype=receiver_head.weight.dtype,
            device=receiver_head.weight.device,
        )
        receiver_xyz_row_mask[7:10] = 1.0
        receiver_head.weight.register_hook(
            lambda gradient: gradient
            * receiver_xyz_row_mask.unsqueeze(-1)
        )
        receiver_head.bias.register_hook(
            lambda gradient: gradient * receiver_xyz_row_mask
        )
        if self.distribution is not None:
            for parameter_name in ("std_param", "log_std_param"):
                parameter = getattr(
                    self.distribution,
                    parameter_name,
                    None,
                )
                if parameter is not None:
                    parameter.requires_grad_(False)

    def configure_receiver_grasp_retain_adaptation(self) -> None:
        """Adapt receiver SE(3) through approach, seating, and release."""
        self.receiver_adaptation_enabled = True
        self.controller.receiver_grasp_retain_residual_enabled_for_learning = (
            True
        )
        for parameter in self.phase_network.parameters():
            parameter.requires_grad_(False)
        receiver_se3_row_mask = torch.zeros(
            14,
            dtype=self.phase_network.heads[2].weight.dtype,
            device=self.phase_network.heads[2].weight.device,
        )
        receiver_se3_row_mask[7:13] = 1.0
        for phase_index in (2, 3):
            receiver_head = self.phase_network.heads[phase_index]
            receiver_head.weight.requires_grad_(True)
            receiver_head.bias.requires_grad_(True)
            receiver_head.weight.register_hook(
                lambda gradient, row_mask=receiver_se3_row_mask: (
                    gradient * row_mask.unsqueeze(-1)
                )
            )
            receiver_head.bias.register_hook(
                lambda gradient, row_mask=receiver_se3_row_mask: (
                    gradient * row_mask
                )
            )
        if self.distribution is not None:
            for parameter_name in ("std_param", "log_std_param"):
                parameter = getattr(
                    self.distribution,
                    parameter_name,
                    None,
                )
                if parameter is not None:
                    parameter.requires_grad_(False)

    def configure_recovery_receiver_grasp_retain_adaptation(self) -> None:
        """Chain the frozen pickup-recovery option into receiver adaptation.

        The loaded checkpoint's learned giver recovery remains active at
        inference, but only the zero-initialized receiver adapter receives
        gradients and exploration. Its output is restricted to receiver
        SE(3). This prevents the downstream option from erasing the
        qualified recovery behavior that generates its source states.
        """
        self.pickup_recovery_adaptation_enabled = True
        self.receiver_adaptation_enabled = True
        self.recovery_receiver_grasp_retain_adaptation_enabled = True
        self.controller.giver_recovery_residual_only_for_learning = True
        self.controller.receiver_grasp_retain_residual_enabled_for_learning = (
            True
        )
        if self.recovery_receiver_reference_network is None:
            self.recovery_receiver_reference_network = copy.deepcopy(
                self.phase_network
            )
        for parameter in (
            self.recovery_receiver_reference_network.parameters()
        ):
            parameter.requires_grad_(False)
        for parameter in self.phase_network.parameters():
            parameter.requires_grad_(False)
        for parameter in self.recovery_receiver_adapter.parameters():
            parameter.requires_grad_(True)
        if self.distribution is not None:
            for parameter_name in ("std_param", "log_std_param"):
                parameter = getattr(
                    self.distribution,
                    parameter_name,
                    None,
                )
                if parameter is not None:
                    parameter.requires_grad_(False)

    def configure_joint_transfer_acquisition_adaptation(self) -> None:
        """Learn coupled tool motion without altering the promoted option.

        The loaded pickup-recovery policy remains active and frozen. A new
        exact-zero adapter alone receives gradients, and it is gated to phase
        two SE(3) motion for both tools. Gripper closure and giver release are
        still fully analytic and require the unchanged filtered contact
        contract.
        """
        self.pickup_recovery_adaptation_enabled = True
        self.receiver_adaptation_enabled = True
        self.joint_transfer_acquisition_adaptation_enabled = True
        self.controller.giver_recovery_residual_only_for_learning = True
        for parameter in self.phase_network.parameters():
            parameter.requires_grad_(False)
        for parameter in self.recovery_receiver_adapter.parameters():
            parameter.requires_grad_(False)
        for parameter in (
            self.joint_transfer_acquisition_adapter.parameters()
        ):
            parameter.requires_grad_(True)
        if self.distribution is not None:
            for parameter_name in ("std_param", "log_std_param"):
                parameter = getattr(
                    self.distribution,
                    parameter_name,
                    None,
                )
                if parameter is not None:
                    parameter.requires_grad_(False)

    def configure_transfer_refinement_adaptation(self) -> None:
        """Refine the frozen joint option through acquisition and retention.

        A loaded joint-transfer checkpoint remains active and immutable.
        The new exact-zero adapter alone receives gradients. It may correct
        receiver SE(3) only after the physical presentation is qualified and
        through bilateral acquisition/retention. The giver, both grippers, and
        giver release stay under the unchanged analytic contact contract.
        """
        self.pickup_recovery_adaptation_enabled = True
        self.receiver_adaptation_enabled = True
        self.joint_transfer_acquisition_adaptation_enabled = True
        self.transfer_refinement_adaptation_enabled = True
        self.controller.giver_recovery_residual_only_for_learning = True
        for parameter in self.phase_network.parameters():
            parameter.requires_grad_(False)
        for parameter in self.recovery_receiver_adapter.parameters():
            parameter.requires_grad_(False)
        for parameter in (
            self.joint_transfer_acquisition_adapter.parameters()
        ):
            parameter.requires_grad_(False)
        for parameter in self.transfer_refinement_adapter.parameters():
            parameter.requires_grad_(True)
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
        """Learn giver XY while preserving the promoted receiver policy."""
        self.giver_adaptation_enabled = True
        for parameter in self.phase_network.parameters():
            parameter.requires_grad_(False)
        giver_role_row_mask = torch.zeros(
            14,
            dtype=self.phase_network.heads[0].weight.dtype,
            device=self.phase_network.heads[0].weight.device,
        )
        giver_role_row_mask[0:2] = 1.0
        for head in self.phase_network.heads:
            head.weight.requires_grad_(True)
            head.bias.requires_grad_(True)
            head.weight.register_hook(
                lambda gradient, row_mask=giver_role_row_mask: (
                    gradient * row_mask.unsqueeze(-1)
                )
            )
            head.bias.register_hook(
                lambda gradient, row_mask=giver_role_row_mask: (
                    gradient * row_mask
                )
            )

    def configure_pickup_recovery_adaptation(self) -> None:
        """Adapt giver XY only on post-slip relift phases."""
        self.pickup_recovery_adaptation_enabled = True
        self.controller.giver_recovery_residual_only_for_learning = True
        for parameter in self.phase_network.parameters():
            parameter.requires_grad_(False)
        giver_xy_row_mask = torch.zeros(
            14,
            dtype=self.phase_network.heads[1].weight.dtype,
            device=self.phase_network.heads[1].weight.device,
        )
        giver_xy_row_mask[0:2] = 1.0
        for phase_index in (0, 1, 2, 4):
            head = self.phase_network.heads[phase_index]
            head.weight.requires_grad_(True)
            head.bias.requires_grad_(True)
            head.weight.register_hook(
                lambda gradient, row_mask=giver_xy_row_mask: (
                    gradient * row_mask.unsqueeze(-1)
                )
            )
            head.bias.register_hook(
                lambda gradient, row_mask=giver_xy_row_mask: (
                    gradient * row_mask
                )
            )
        if self.distribution is not None:
            for parameter_name in ("std_param", "log_std_param"):
                parameter = getattr(
                    self.distribution,
                    parameter_name,
                    None,
                )
                if parameter is not None:
                    parameter.requires_grad_(False)

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        phase = torch.argmax(raw[:, 77:82], dim=-1)
        current_role_residual = torch.tanh(
            self.phase_network(latent, phase)
        )
        learned_role_residual = current_role_residual
        joint_role_residual = torch.zeros_like(learned_role_residual)
        refinement_role_residual = torch.zeros_like(learned_role_residual)
        if self.recovery_receiver_grasp_retain_adaptation_enabled:
            if self.recovery_receiver_reference_network is None:
                raise RuntimeError(
                    "recovery receiver adaptation requires a frozen "
                    "reference network"
                )
            reference_role_residual = torch.tanh(
                self.recovery_receiver_reference_network(latent, phase)
            )
            pickup_recovery_context = raw[:, 98] > 0.5
            learned_role_residual = torch.where(
                pickup_recovery_context.unsqueeze(-1),
                current_role_residual,
                reference_role_residual,
            )
            role_observation = role_normalize_handover_observation(raw)
            adapter_features = recovery_receiver_canonical_grasp_features(
                role_observation,
                self.receiver_policy_grasp_offset,
            )
            receiver_adapter = self.recovery_receiver_adapter(
                adapter_features
            )
            adapter_role_residual = torch.zeros_like(
                learned_role_residual
            )
            adapter_role_residual[:, 7:13] = receiver_adapter
            learned_role_residual = (
                learned_role_residual
                + pickup_recovery_context.unsqueeze(-1)
                * adapter_role_residual
            ).clamp(-1.0, 1.0)
        joint_active = phase == 2
        if self.joint_transfer_acquisition_adaptation_enabled:
            role_observation = role_normalize_handover_observation(raw)
            joint_adapter = self.joint_transfer_acquisition_adapter(
                joint_transfer_acquisition_features(
                    role_observation,
                    self.receiver_policy_grasp_offset,
                )
            )
            joint_role_residual[:, 0:6] = joint_adapter[:, 0:6]
            joint_role_residual[:, 7:13] = joint_adapter[:, 6:12]
            joint_role_residual *= joint_active.unsqueeze(-1)
        presentation_qualified = raw[:, 103] >= 1.0
        refinement_giver_active = torch.zeros_like(
            presentation_qualified
        )
        refinement_receiver_active = (
            ((phase == 2) & presentation_qualified) | (phase == 3)
        )
        if self.transfer_refinement_adaptation_enabled:
            role_observation = role_normalize_handover_observation(raw)
            refinement_adapter = self.transfer_refinement_adapter(
                joint_transfer_acquisition_features(
                    role_observation,
                    self.receiver_policy_grasp_offset,
                )
            )
            refinement_role_residual[:, 0:6] = (
                refinement_adapter[:, 0:6]
                * refinement_giver_active.unsqueeze(-1)
            )
            refinement_role_residual[:, 7:13] = (
                refinement_adapter[:, 6:12]
                * refinement_receiver_active.unsqueeze(-1)
            )
        physical_residual = role_action_to_physical(
            learned_role_residual,
            raw[:, 82] > 0.5,
        )
        joint_physical_residual = role_action_to_physical(
            joint_role_residual,
            raw[:, 82] > 0.5,
        )
        refinement_physical_residual = role_action_to_physical(
            refinement_role_residual,
            raw[:, 82] > 0.5,
        )
        joint_role_action_mask = torch.zeros_like(
            learned_role_residual,
            dtype=torch.bool,
        )
        joint_role_action_mask[:, 0:6] = joint_active.unsqueeze(-1)
        joint_role_action_mask[:, 7:13] = joint_active.unsqueeze(-1)
        joint_physical_action_mask = role_action_to_physical(
            joint_role_action_mask,
            raw[:, 82] > 0.5,
        )
        refinement_role_action_mask = torch.zeros_like(
            learned_role_residual,
            dtype=torch.bool,
        )
        refinement_role_action_mask[:, 0:6] = (
            refinement_giver_active.unsqueeze(-1)
        )
        refinement_role_action_mask[:, 7:13] = (
            refinement_receiver_active.unsqueeze(-1)
        )
        refinement_physical_action_mask = role_action_to_physical(
            refinement_role_action_mask,
            raw[:, 82] > 0.5,
        )
        (
            base_action,
            giver_residual_mask,
            receiver_residual_mask,
        ) = self.controller(raw)
        # The promoted receiver mean remains active during giver adaptation,
        # but exploration and gradients are restricted to giver XY. Z and all
        # rotations/jaws stay under the analytic physics sequence.
        physical_action_mask = receiver_residual_mask
        exploration_mask = receiver_residual_mask
        if self.recovery_receiver_grasp_retain_adaptation_enabled:
            physical_action_mask = (
                giver_residual_mask | receiver_residual_mask
            )
            exploration_mask = receiver_residual_mask
        elif (
            self.giver_adaptation_enabled
            or self.pickup_recovery_adaptation_enabled
        ):
            physical_action_mask = (
                giver_residual_mask | receiver_residual_mask
            )
            exploration_mask = giver_residual_mask
        if self.joint_transfer_acquisition_adaptation_enabled:
            exploration_mask = (
                exploration_mask | joint_physical_action_mask
            )
        if self.transfer_refinement_adaptation_enabled:
            exploration_mask = (
                exploration_mask | refinement_physical_action_mask
            )
        physical_mean = (
            base_action
            + self.residual_scale
            * physical_residual
            * physical_action_mask.to(base_action.dtype)
            + self.residual_scale
            * joint_physical_residual
            * joint_physical_action_mask.to(base_action.dtype)
            + self.residual_scale
            * refinement_physical_residual
            * refinement_physical_action_mask.to(base_action.dtype)
        ).clamp(-1.0, 1.0)
        if self.distribution is None:
            return physical_mean
        if stochastic_output:
            set_action_mask = getattr(
                self.distribution,
                "set_action_mask",
                None,
            )
            if set_action_mask is not None:
                set_action_mask(exploration_mask)
            self.distribution.update(physical_mean)
            return self.distribution.sample()
        return self.distribution.deterministic_output(physical_mean)

    def as_jit(self) -> nn.Module:
        return _EndToEndHandoverExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _EndToEndHandoverOnnxExport(self, verbose)


class _EndToEndHandoverExport(nn.Module):
    """TorchScript-compatible deterministic end-to-end handover policy."""

    def __init__(self, model: EndToEndHandoverMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.phase_network = copy.deepcopy(model.phase_network)
        self.controller = copy.deepcopy(model.controller)
        self.residual_scale = model.residual_scale
        self.giver_adaptation_enabled = model.giver_adaptation_enabled
        self.pickup_recovery_adaptation_enabled = (
            model.pickup_recovery_adaptation_enabled
        )
        self.recovery_receiver_grasp_retain_adaptation_enabled = (
            model.recovery_receiver_grasp_retain_adaptation_enabled
        )
        self.joint_transfer_acquisition_adaptation_enabled = (
            model.joint_transfer_acquisition_adaptation_enabled
        )
        self.transfer_refinement_adaptation_enabled = (
            model.transfer_refinement_adaptation_enabled
        )
        self.recovery_receiver_reference_network = copy.deepcopy(
            model.recovery_receiver_reference_network
        )
        self.recovery_receiver_adapter = copy.deepcopy(
            model.recovery_receiver_adapter
        )
        self.joint_transfer_acquisition_adapter = copy.deepcopy(
            model.joint_transfer_acquisition_adapter
        )
        self.transfer_refinement_adapter = copy.deepcopy(
            model.transfer_refinement_adapter
        )
        self.register_buffer(
            "receiver_policy_grasp_offset",
            model.receiver_policy_grasp_offset.detach().clone(),
            persistent=False,
        )
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None
            else nn.Identity()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        giver_is_robot_1 = obs[:, 82] > 0.5
        role_observation = role_normalize_handover_observation(obs)
        latent = torch.cat(
            (
                self.obs_normalizer(role_observation),
                handover_task_features(
                    role_observation,
                    self.receiver_policy_grasp_offset,
                ),
            ),
            dim=-1,
        )
        phase = torch.argmax(obs[:, 77:82], dim=-1)
        current_role_residual = torch.tanh(
            self.phase_network(latent, phase)
        )
        learned_role_residual = current_role_residual
        joint_role_residual = torch.zeros_like(learned_role_residual)
        refinement_role_residual = torch.zeros_like(learned_role_residual)
        if self.recovery_receiver_grasp_retain_adaptation_enabled:
            if self.recovery_receiver_reference_network is None:
                raise RuntimeError(
                    "recovery receiver export requires a reference network"
                )
            reference_role_residual = torch.tanh(
                self.recovery_receiver_reference_network(latent, phase)
            )
            pickup_recovery_context = obs[:, 98] > 0.5
            learned_role_residual = torch.where(
                pickup_recovery_context.unsqueeze(-1),
                current_role_residual,
                reference_role_residual,
            )
            adapter_features = recovery_receiver_canonical_grasp_features(
                role_observation,
                self.receiver_policy_grasp_offset,
            )
            receiver_adapter = self.recovery_receiver_adapter(
                adapter_features
            )
            adapter_role_residual = torch.zeros_like(
                learned_role_residual
            )
            adapter_role_residual[:, 7:13] = receiver_adapter
            learned_role_residual = (
                learned_role_residual
                + pickup_recovery_context.unsqueeze(-1)
                * adapter_role_residual
            ).clamp(-1.0, 1.0)
        joint_active = phase == 2
        if self.joint_transfer_acquisition_adaptation_enabled:
            joint_adapter = self.joint_transfer_acquisition_adapter(
                joint_transfer_acquisition_features(
                    role_observation,
                    self.receiver_policy_grasp_offset,
                )
            )
            joint_role_residual[:, 0:6] = joint_adapter[:, 0:6]
            joint_role_residual[:, 7:13] = joint_adapter[:, 6:12]
            joint_role_residual *= joint_active.unsqueeze(-1)
        presentation_qualified = obs[:, 103] >= 1.0
        refinement_giver_active = torch.zeros_like(
            presentation_qualified
        )
        refinement_receiver_active = (
            ((phase == 2) & presentation_qualified) | (phase == 3)
        )
        if self.transfer_refinement_adaptation_enabled:
            refinement_adapter = self.transfer_refinement_adapter(
                joint_transfer_acquisition_features(
                    role_observation,
                    self.receiver_policy_grasp_offset,
                )
            )
            refinement_role_residual[:, 0:6] = (
                refinement_adapter[:, 0:6]
                * refinement_giver_active.unsqueeze(-1)
            )
            refinement_role_residual[:, 7:13] = (
                refinement_adapter[:, 6:12]
                * refinement_receiver_active.unsqueeze(-1)
            )
        physical_residual = role_action_to_physical(
            learned_role_residual,
            giver_is_robot_1,
        )
        joint_physical_residual = role_action_to_physical(
            joint_role_residual,
            giver_is_robot_1,
        )
        refinement_physical_residual = role_action_to_physical(
            refinement_role_residual,
            giver_is_robot_1,
        )
        joint_role_action_mask = torch.zeros_like(
            learned_role_residual,
            dtype=torch.bool,
        )
        joint_role_action_mask[:, 0:6] = joint_active.unsqueeze(-1)
        joint_role_action_mask[:, 7:13] = joint_active.unsqueeze(-1)
        joint_physical_action_mask = role_action_to_physical(
            joint_role_action_mask,
            giver_is_robot_1,
        )
        refinement_role_action_mask = torch.zeros_like(
            learned_role_residual,
            dtype=torch.bool,
        )
        refinement_role_action_mask[:, 0:6] = (
            refinement_giver_active.unsqueeze(-1)
        )
        refinement_role_action_mask[:, 7:13] = (
            refinement_receiver_active.unsqueeze(-1)
        )
        refinement_physical_action_mask = role_action_to_physical(
            refinement_role_action_mask,
            giver_is_robot_1,
        )
        (
            base_action,
            giver_residual_mask,
            receiver_residual_mask,
        ) = self.controller(obs)
        physical_action_mask = receiver_residual_mask
        if (
            self.giver_adaptation_enabled
            or self.pickup_recovery_adaptation_enabled
        ):
            physical_action_mask = (
                giver_residual_mask | receiver_residual_mask
            )
        physical_mean = (
            base_action
            + self.residual_scale
            * physical_residual
            * physical_action_mask.to(base_action.dtype)
            + self.residual_scale
            * joint_physical_residual
            * joint_physical_action_mask.to(base_action.dtype)
            + self.residual_scale
            * refinement_physical_residual
            * refinement_physical_action_mask.to(base_action.dtype)
        ).clamp(-1.0, 1.0)
        return self.deterministic_output(physical_mean)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _EndToEndHandoverOnnxExport(_EndToEndHandoverExport):
    """ONNX metadata for the deterministic end-to-end policy."""

    is_recurrent: bool = False

    def __init__(
        self,
        model: EndToEndHandoverMLPModel,
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
