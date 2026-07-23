# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T N1.5 modality contract for NVIDIA's seven-action PSM."""

from __future__ import annotations

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoColorJitter, VideoCrop, VideoResize, VideoToNumpy, VideoToTensor
from gr00t.experiment.data_config import DATA_CONFIG_MAP, BaseDataConfig
from gr00t.model.transforms import GR00TTransform

DATA_CONFIG_NAME = "psm_singlecam"
PSM_STATE_DIM = 8
PSM_ACTION_DIM = 7


class PsmSingleCameraDataConfig(BaseDataConfig):
    video_keys = ["video.room"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self) -> ComposedModalityTransform:
        keys = self.state_keys + self.action_keys
        return ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=self.video_keys),
                VideoCrop(apply_to=self.video_keys, scale=0.95),
                VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.2,
                    contrast=0.25,
                    saturation=0.2,
                    hue=0.04,
                ),
                VideoToNumpy(apply_to=self.video_keys),
                StateActionToTensor(apply_to=keys),
                StateActionTransform(
                    apply_to=keys,
                    normalization_modes={key: "min_max" for key in keys},
                ),
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
                GR00TTransform(state_horizon=1, action_horizon=16, max_state_dim=64, max_action_dim=32),
            ]
        )


def register() -> PsmSingleCameraDataConfig:
    existing = DATA_CONFIG_MAP.get(DATA_CONFIG_NAME)
    if existing is not None:
        if not isinstance(existing, PsmSingleCameraDataConfig):
            raise RuntimeError(f"{DATA_CONFIG_NAME} is already registered by an incompatible provider")
        return existing
    config = PsmSingleCameraDataConfig()
    DATA_CONFIG_MAP[DATA_CONFIG_NAME] = config
    return config


register()
