# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab composition for the four-arm Autonomous Rescue OR."""

from __future__ import annotations

import json
import math
from typing import Final

from .autonomous_rescue_or import (
    ASSET_DIRECTORY,
    anchor_rescue_vessel,
    autonomous_rescue_or_cfg,
)


STATION_CONTRACT: Final = ASSET_DIRECTORY / "robot_station_contract.json"
STATION_IDS: Final = ("primary", "exposure", "rescue", "assessment")


def _station_poses() -> dict[str, tuple[tuple[float, ...], float]]:
    payload = json.loads(STATION_CONTRACT.read_text(encoding="utf-8"))
    stations: dict[str, tuple[tuple[float, ...], float]] = {}
    for item in payload["stations"]:
        station_id = str(item["id"])
        stations[station_id] = (
            tuple(float(value) for value in item["pose"]["position_m"]),
            float(item["pose"]["yaw_deg"]),
        )
    missing = sorted(set(STATION_IDS) - stations.keys())
    if missing:
        raise ValueError(f"robot station contract is missing {missing}")
    return stations


def _yaw_quaternion_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(float(yaw_deg)) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _training_environment_cfg():
    """Disable visual station placeholders when real articulations are present."""

    cfg = autonomous_rescue_or_cfg()
    source_spawn = cfg.spawn.func

    def spawn_training_environment(
        prim_path: str,
        spawn_cfg,
        translation=None,
        orientation=None,
        **kwargs,
    ):
        root_prim = source_spawn(
            prim_path,
            spawn_cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        stage = root_prim.GetStage()
        root_path = str(root_prim.GetPath())
        for prim_name in ("Primary", "Exposure", "Rescue", "Assessment"):
            placeholder = stage.GetPrimAtPath(
                f"{root_path}/RobotStations/{prim_name}/UniversalToolChanger"
            )
            if placeholder.IsValid():
                placeholder.SetActive(False)
        from pxr import UsdGeom

        rescue_suite_path = f"{root_path}/DeformableRescueSuite"
        rescue_suite = stage.GetPrimAtPath(rescue_suite_path)
        if not rescue_suite.IsValid():
            raise RuntimeError(
                f"rescue scene is missing {rescue_suite_path}"
            )
        UsdGeom.Imageable(rescue_suite).MakeVisible()
        for inactive_substrate in (
            "AbdominalWall",
            "BowelAnastomosis",
            "OcclusiveFilm",
        ):
            substrate = stage.GetPrimAtPath(
                f"{rescue_suite_path}/{inactive_substrate}"
            )
            if substrate.IsValid():
                substrate.SetActive(False)
        anchor_rescue_vessel(
            f"{rescue_suite_path}/Vessel",
            stage=stage,
        )
        return root_prim

    cfg.spawn.func = spawn_training_environment
    return cfg


def autonomous_rescue_scene_cfg(
    *,
    num_envs: int = 1,
    env_spacing: float = 5.0,
):
    """Build the OR with four controllable Franka/tool articulations.

    The room USDA owns the patient, hemorrhage vessel, resource carousel,
    monitor, and station anchors. Other rescue substrates remain inactive so
    the current PhysX scene has one bounded deformable lane. Isaac Lab owns the
    robot articulations so each station exposes real joint/contact state.
    """

    try:
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Isaac Lab is required to compose the Autonomous Rescue OR scene"
        ) from error
    from .adaptive_anastomosis_robot import (
        make_franka_adaptive_anastomosis_robot_cfg,
    )
    from .adaptive_hemostasis_robot import (
        make_franka_adaptive_hemostasis_robot_cfg,
    )
    from .atraumatic_exposure_robot import make_franka_exposure_robot_cfg
    from .perfusion_viability_robot import (
        make_franka_perfusion_viability_robot_cfg,
    )

    @configclass
    class AutonomousRescueSceneCfg(InteractiveSceneCfg):
        environment = _training_environment_cfg()
        primary = make_franka_adaptive_anastomosis_robot_cfg(
            prim_path="{ENV_REGEX_NS}/PrimaryRobot"
        )
        exposure = make_franka_exposure_robot_cfg(
            prim_path="{ENV_REGEX_NS}/ExposureRobot",
            pad_type="fenestrated",
        )
        rescue = make_franka_adaptive_hemostasis_robot_cfg(
            prim_path="{ENV_REGEX_NS}/RescueRobot"
        )
        assessment = make_franka_perfusion_viability_robot_cfg(
            prim_path="{ENV_REGEX_NS}/AssessmentRobot"
        )

    scene = AutonomousRescueSceneCfg(
        num_envs=int(num_envs),
        env_spacing=float(env_spacing),
    )
    poses = _station_poses()
    for station_id in STATION_IDS:
        position, yaw_deg = poses[station_id]
        robot_cfg = getattr(scene, station_id)
        robot_cfg.init_state.pos = position
        robot_cfg.init_state.rot = _yaw_quaternion_wxyz(yaw_deg)
    return scene


__all__ = [
    "STATION_CONTRACT",
    "STATION_IDS",
    "autonomous_rescue_scene_cfg",
]
