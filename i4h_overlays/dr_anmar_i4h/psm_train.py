# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T N1.5 fine-tuning entry for NVIDIA's seven-action PSM."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro
from common.policy_stack import default_base_model, policy_default, policy_train_default
from dr_anmar_i4h.psm_data_config import (
    DATA_CONFIG_NAME,
    PSM_ACTION_DIM,
    PSM_STATE_DIM,
    register,
)
from huggingface_hub.constants import HF_HOME

logger = logging.getLogger("dr_anmar_i4h.psm_train")
ENV_ID = "surgical_lift_needle"


@dataclass
class TrainConfig:
    dataset_path: list[str]
    output_dir: str = policy_train_default(ENV_ID, "output_dir", "/tmp/dr_anmar_psm_gr00t")
    data_config: str = policy_default(ENV_ID, "data_config", DATA_CONFIG_NAME)
    batch_size: int = 4
    max_steps: int = policy_train_default(ENV_ID, "max_steps", 10_000)
    save_steps: int = policy_train_default(ENV_ID, "save_steps", 1_000)
    dataloader_num_workers: int = 4
    base_model_path: str = field(
        default_factory=lambda: default_base_model(ENV_ID, "nvidia/GR00T-N1.5-3B")
    )
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    embodiment_tag: str = "new_embodiment"
    video_backend: Literal["decord", "torchvision_av"] = "decord"
    report_to: Literal["tensorboard", "wandb"] = "tensorboard"
    resume: bool = False
    validate_only: bool = False


def _resolve_dataset_path(path_or_repo_id: str) -> Path:
    direct = Path(path_or_repo_id).expanduser()
    if direct.exists():
        return direct.resolve()
    cached = Path(os.getenv("HF_LEROBOT_HOME", Path(HF_HOME) / "lerobot")) / path_or_repo_id
    if cached.exists():
        return cached.resolve()
    raise SystemExit(f"dataset path does not exist: {path_or_repo_id}")


def validate_lerobot_contract(path: Path) -> dict:
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"LeRobot dataset metadata is missing: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features") or {}
    expected = {
        "action": PSM_ACTION_DIM,
        "observation.state": PSM_STATE_DIM,
    }
    observed = {}
    for key, dimension in expected.items():
        feature = features.get(key) or {}
        shape = feature.get("shape")
        if not isinstance(shape, list) or shape != [dimension]:
            raise SystemExit(f"{path}: {key} shape is {shape!r}; expected [{dimension}]")
        observed[key] = shape
    room = features.get("observation.images.room") or features.get("video.room")
    if not isinstance(room, dict):
        raise SystemExit(f"{path}: single room-camera feature is missing")
    return {
        "dataset": str(path),
        "action_dim": PSM_ACTION_DIM,
        "state_dim": PSM_STATE_DIM,
        "room_camera": room.get("shape"),
        "episodes": int(info.get("total_episodes", 0)),
        "frames": int(info.get("total_frames", 0)),
    }


def _build_dataset(cfg: TrainConfig):
    from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP

    register()
    data_config = DATA_CONFIG_MAP[cfg.data_config]
    resolved = [_resolve_dataset_path(path) for path in cfg.dataset_path]
    for path in resolved:
        validate_lerobot_contract(path)

    def single(path: Path) -> LeRobotSingleDataset:
        return LeRobotSingleDataset(
            dataset_path=str(path),
            modality_configs=data_config.modality_config(),
            transforms=data_config.transform(),
            embodiment_tag=EmbodimentTag(cfg.embodiment_tag),
            video_backend=cfg.video_backend,
        )

    datasets = [single(path) for path in resolved]
    if len(datasets) == 1:
        return datasets[0]
    return LeRobotMixtureDataset(
        data_mixture=[(dataset, 1.0) for dataset in datasets],
        mode="train",
        balance_dataset_weights=True,
        balance_trajectory_weights=True,
        seed=42,
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )


def _train(cfg: TrainConfig) -> None:
    from gr00t.experiment import runner as gr00t_runner
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from transformers import TrainingArguments

    dataset = _build_dataset(cfg)
    if cfg.validate_only:
        sample = dataset[0]
        print(
            json.dumps(
                {
                    "status": "ready",
                    "data_config": cfg.data_config,
                    "dataset_length": len(dataset),
                    "sample_keys": sorted(sample),
                },
                indent=2,
            )
        )
        return

    if not torch.cuda.is_available():
        raise SystemExit("GR00T PSM fine-tuning requires a CUDA GPU")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=cfg.base_model_path,
        tune_llm=cfg.tune_llm,
        tune_visual=cfg.tune_visual,
        tune_projector=cfg.tune_projector,
        tune_diffusion_model=cfg.tune_diffusion_model,
    )
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    arguments = TrainingArguments(
        output_dir=cfg.output_dir,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=cfg.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        max_steps=cfg.max_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=5,
        report_to=cfg.report_to,
        seed=42,
        do_eval=False,
    )
    gr00t_runner.TrainRunner(
        train_dataset=dataset,
        model=model,
        training_args=arguments,
        resume_from_checkpoint=cfg.resume,
    ).train()


def main() -> None:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    cfg = tyro.cli(TrainConfig)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    if not cfg.dataset_path:
        raise SystemExit("--dataset-path is required")
    if cfg.data_config != DATA_CONFIG_NAME:
        raise SystemExit(f"PSM training requires --data-config {DATA_CONFIG_NAME}")
    if cfg.batch_size < 1 or cfg.max_steps < 1 or cfg.save_steps < 1:
        raise SystemExit("batch-size, max-steps, and save-steps must be positive")
    _train(cfg)


if __name__ == "__main__":
    main()
