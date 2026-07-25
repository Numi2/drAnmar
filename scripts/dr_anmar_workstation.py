# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Browser-operated, simulation-only Dr.Anmar surgical workstation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import platform
import queue
import signal
import subprocess
import threading
import time
import traceback
import zipfile
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

from dr_anmar_asset_layout import asset_landing as dr_anmar_asset_landing
from dr_anmar_asset_registry import provider_roots, resolve_provider_asset
from dr_anmar_bench_systems import (
    BENCH_ROBOT_SYSTEMS_BY_ID,
    FEATURED_ROBOT_POSITION_M,
    FEATURED_SUBSTRATE_POSITION_M,
    resolve_featured_robot_system,
)
from dr_anmar_hemostasis_model import (
    sample_hemostasis_episode_parameters,
    stable_physx_vessel_proxy_parameters,
)
from dr_anmar_native_rooms import resolve_native_room
from dr_anmar_procedures import PROCEDURES_BY_ID
from dr_anmar_psm_gripper import (
    CANONICAL_PSM_GRIPPER_PROFILE,
    apply_psm_gripper_action_profile,
    apply_psm_gripper_articulation_profile,
    complete_psm_actions_from_nvidia_orbit,
    psm_articulation_names,
    psm_gripper_profile_manifest,
    resolve_psm_gripper_profile,
)
from dr_anmar_suture_integration import (
    apply_dr_anmar_needle_episode_domain,
    configure_dr_anmar_needle,
    configure_nvidia_needle_dr_anmar_suture,
)
from dr_anmar_suture_model import (
    load_profile as load_suture_profile,
    sample_suture_runtime_profile,
)
from dr_anmar_tissue_model import (
    sample_tissue_episode_parameters,
    stable_physx_proxy_parameters,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()
I4H_ASSET_HASH = os.environ.get("DR_ANMAR_I4H_ASSET_HASH", "724f82e")
I4H_ASSET_CONTENT_ROOT = Path(
    os.environ.get(
        "DR_ANMAR_I4H_ASSET_CONTENT_ROOT",
        DATA_ROOT / "assets/i4h-catalog" / I4H_ASSET_HASH,
    )
).expanduser()
HAND_CONTROL_ASSET_ROOT = Path(
    os.environ.get(
        "DR_ANMAR_HAND_CONTROL_ASSET_ROOT",
        DATA_ROOT / "assets/hand-control/mediapipe-tasks-vision-0.10.35",
    )
).expanduser()
HAND_CONTROL_ASSET_FILES = {
    "vision_bundle.mjs",
    "hand_landmarker.task",
    "wasm/vision_wasm_internal.js",
    "wasm/vision_wasm_internal.wasm",
    "wasm/vision_wasm_module_internal.js",
    "wasm/vision_wasm_module_internal.wasm",
    "wasm/vision_wasm_nosimd_internal.js",
    "wasm/vision_wasm_nosimd_internal.wasm",
}
HAND_CONTROL_CLIENT_PATH = Path(__file__).resolve().parents[1] / "web/hand_control.mjs"
HAND_CONTROL_WORKER_PATH = Path(__file__).resolve().parents[1] / "web/hand_control_worker.mjs"


def positive_environment_number(name: str, default: float, minimum: float) -> float:
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


MAX_DEMO_FRAMES = int(positive_environment_number("DR_ANMAR_MAX_DEMO_FRAMES", 60_000, 1_000))
MAX_DEMO_SECONDS = positive_environment_number("DR_ANMAR_MAX_DEMO_SECONDS", 300.0, 30.0)
MAX_DEMO_BYTES = int(positive_environment_number("DR_ANMAR_MAX_DEMO_BYTES", 1_500_000_000, 50_000_000))
MEMORY_WARNING_BYTES = int(
    positive_environment_number("DR_ANMAR_MEMORY_WARNING_BYTES", 16_000_000_000, 1_000_000_000)
)
SENSOR_PROFILES = {"efficient", "stereo", "research"}
DEFAULT_SCENARIO_SEED = 7777
EXTERNAL_OPERATOR_SENSORS_ENABLED = os.environ.get("DR_ANMAR_ENABLE_EXTERNAL_OPERATOR_SENSORS", "0") == "1"
STUDY_ID = os.environ.get("DR_ANMAR_STUDY_ID", "").strip()
CONSENT_PROTOCOL = os.environ.get("DR_ANMAR_CONSENT_PROTOCOL", "").strip()
ACTION_CONTRACT = {
    "id": "dr_anmar.nvidia_psm_policy_action.v1",
    "dimensions_per_psm": 7,
    "arm": "six raw NVIDIA JointPositionAction inputs",
    "gripper": "one canonical binary policy sign",
    "runtime_gripper": "one proportional Cartesian slot; -1 closed, +1 open",
    "source": "NVIDIA DifferentialInverseKinematicsAction resolved joint targets",
    "doctor_intent_key": "cartesian_actions",
}
NON_PSM_ACTION_CONTRACT = {
    "id": "dr-anmar-cartesian-ik-relative-v1",
    "bounds": [-1.0, 1.0],
    "translation_scale_m": 0.01,
    "rotation_scale_rad": 0.05,
    "gripper": "binary normalized action per instrument",
}

parser = argparse.ArgumentParser(description="Run the Dr.Anmar browser workstation.")
parser.add_argument("--task", default="Isaac-Lift-Needle-PSM-IK-Rel-v0")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=2361)
parser.add_argument("--demo_dir", type=Path, default=DATA_ROOT / "demos")
parser.add_argument("--camera_width", type=int, default=960)
parser.add_argument("--camera_height", type=int, default=640)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--procedure", default="")
parser.add_argument("--anatomy_scene", type=Path)
parser.add_argument("--openusd_environment", type=Path)
parser.add_argument("--anatomy_scene_id", default="")
parser.add_argument("--anatomy_title", default="")
parser.add_argument(
    "--bench_assets",
    default="default",
    help="comma-separated NVIDIA bench prop ids, 'default', or 'none'",
)
parser.add_argument("--gripper_open_rad", type=float)
parser.add_argument("--gripper_close_rad", type=float)
parser.add_argument(
    "--sensor_profile",
    choices=sorted(SENSOR_PROFILES),
    default=os.environ.get("DR_ANMAR_SENSOR_PROFILE", "research"),
    help="all profiles include one down-axis RGB camera per PSM; efficient=left RGB, stereo=left RGB-D+right RGB, research=full research sensors",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
_softmimicgen_task = args_cli.task.startswith("Isaac-Thread-PSM-")
_softmimicgen_root = Path(
    os.environ.get(
        "DR_ANMAR_SOFTMIMICGEN_ROOT",
        DATA_ROOT / "native-suture-runtime/SoftMimicGen",
    )
).expanduser().resolve()

_requested_procedure = PROCEDURES_BY_ID.get(args_cli.procedure) if args_cli.procedure else None
_native_room = resolve_native_room(args_cli.procedure) if args_cli.procedure else None
if args_cli.procedure:
    if _requested_procedure is None:
        parser.error(f"Unknown Dr.Anmar procedure room: {args_cli.procedure}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
if args_cli.headless:
    # Camera render products are independent of the desktop viewport. NVIDIA
    # recommends disabling that otherwise-hidden viewport in headless sensor
    # workloads so the 4090 does not render every frame twice.
    try:
        from omni.kit.viewport.utility import get_active_viewport

        active_viewport = get_active_viewport()
        if active_viewport is not None:
            active_viewport.updates_enabled = False
    except (ImportError, AttributeError, RuntimeError):
        pass

import gymnasium as gym
import h5py
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.math import quat_conjugate, quat_mul
from isaaclab_tasks.utils import parse_env_cfg

import orbit.surgical.tasks  # noqa: F401
from orbit.surgical.tasks.surgical.handover.config.needle.ik_rel_env_cfg import (
    NeedleHandoverEnvCfg as ORBIT_NEEDLE_HANDOVER_CFG,
)

if _softmimicgen_task:
    import softmimicgen_tasks  # noqa: F401

from orbit.surgical.assets.psm import PSM_HIGH_PD_CFG as ORBIT_PSM_HIGH_PD_CFG
from orbit.surgical.assets.needle_thread import (
    make_needle_cfg as make_dranmar_v030_needle_cfg,
    make_needle_thread_rigid_proxy_cfg,
    make_segmented_needle_thread_cfg,
)
from orbit.surgical.assets.skin_adhesive import (
    activation_targets as skin_adhesive_activation_targets,
    make_articulated_cfg as make_articulated_skin_adhesive_cfg,
    set_activation_target as set_skin_adhesive_activation_target,
)
from orbit.surgical.assets.closure_robot import (
    ClosurePhase,
    ClosureSequenceController,
    anchor_tissue_outer_edges,
    apply_tissue_demo_surface_deformables,
    closure_phase_targets,
    make_franka_closure_robot_cfg,
    set_joint_targets as set_closure_robot_joint_targets,
)
from orbit.surgical.assets.wound_preparation_robot import (
    make_tool_cfg as make_wound_preparation_tool_cfg,
)
from orbit.surgical.assets.atraumatic_exposure_robot import (
    make_tool_cfg as make_atraumatic_exposure_tool_cfg,
)
from orbit.surgical.assets.adaptive_hemostasis_robot import (
    make_tool_cfg as make_adaptive_hemostasis_tool_cfg,
)
from orbit.surgical.assets.adaptive_anastomosis_robot import (
    make_tool_cfg as make_adaptive_anastomosis_tool_cfg,
)
from orbit.surgical.assets.adaptive_seal_divide_robot import (
    make_tool_cfg as make_adaptive_seal_divide_tool_cfg,
)
from orbit.surgical.assets.safeplane_dissection_robot import (
    make_tool_cfg as make_safeplane_dissection_tool_cfg,
)
from orbit.surgical.assets.perfusion_viability_robot import (
    make_tool_cfg as make_perfusion_viability_tool_cfg,
)
from orbit.surgical.assets.dynamic_abdominal_patient import (
    DynamicSurgicalPatient,
    PatientContactFrame,
)
from orbit.surgical.assets.autonomous_rescue_or import (
    AutonomousRescueORRuntime,
    PhysicsEvidenceFrame,
    rescue_vessel_cfg,
)
from orbit.surgical.assets.skin_stapler import (
    ClosureLine,
    FIRE_THRESHOLD_DEG,
    REARM_THRESHOLD_DEG,
    TRIGGER_LIMIT_DEG,
    StapleMagazine,
    TriggerEdgeDeploymentController,
    add_staple_reference,
    assess_placement,
    make_articulated_skin_stapler_cfg,
    spacing_errors_m,
    synchronized_joint_targets_deg,
)

BENCH_ROBOT_SYSTEM_FACTORIES = {
    "wound_preparation_robot": make_wound_preparation_tool_cfg,
    "atraumatic_exposure_robot": make_atraumatic_exposure_tool_cfg,
    "adaptive_hemostasis_robot": make_adaptive_hemostasis_tool_cfg,
    "adaptive_anastomosis_robot": make_adaptive_anastomosis_tool_cfg,
    "adaptive_seal_divide_robot": make_adaptive_seal_divide_tool_cfg,
    "safeplane_dissection_robot": make_safeplane_dissection_tool_cfg,
    "perfusion_viability_robot": make_perfusion_viability_tool_cfg,
}

from dr_anmar_expert import EXPERT_CONTROLLER_VERSION, EXPERT_PHASES, ExpertDemonstrationController
from dr_anmar_operator import ACCESS_COOKIE, OPERATOR_HEADER, OperatorLease, access_is_authorized
from dr_anmar_psm_native_adapter import (
    CONTRACT_NAME as PSM_POLICY_CONTRACT_NAME,
    PSM_ARM_NAMES,
    canonical_policy_contract,
    native_ik_action_scales,
)
from dr_anmar_hand_teleop import (
    HandTeleopRuntime,
    camera_pose_to_action_frame,
    proportional_gripper_action,
    validate_hand_frame,
)


STAPLER_CLOSURE_STATION_OFFSETS_M = (
    -0.018,
    -0.012,
    -0.006,
    0.0,
    0.006,
    0.012,
    0.018,
)
STAPLER_CLOSURE_STATION_SPACING_M = 0.006
STAPLER_TEST_DEVICE_MOUNT_Z_M = 0.0592
STAPLER_CLOSURE_TARGET_CENTER_M = (0.096, 0.0, 0.0602)
STAPLER_CLOSURE_TISSUE_CENTER_M = (0.095, 0.0, 0.055)
STAPLER_TISSUE_APPROXIMATION_DURATION_S = 0.85
STAPLER_TISSUE_TARGET_GAP_M = 0.0008
STAPLER_TISSUE_STATION_HALF_WIDTH_M = 0.0032
STAPLER_TISSUE_EDGE_CAPTURE_M = 0.0075
STAPLER_CLOSURE_TISSUE_ROTATION_WXYZ = (
    0.70710678,
    0.0,
    0.0,
    0.70710678,
)


FAILURE_SCENARIOS = (
    {
        "id": "baseline",
        "title": "Clinical baseline",
        "difficulty": "Foundation",
        "description": "Standard camera, lighting, and control response for learning the task.",
        "doctor_focus": "Complete approach, grasp, lift, and recovery with deliberate motion.",
    },
    {
        "id": "camera_shift",
        "title": "Shifted endoscope",
        "difficulty": "Intermediate",
        "description": "The endoscope begins from a displaced viewpoint while the task stays unchanged.",
        "doctor_focus": "Rebuild depth and orientation cues before committing to the grasp.",
    },
    {
        "id": "low_light",
        "title": "Reduced illumination",
        "difficulty": "Intermediate",
        "description": "The endoscopic feed is darkened to challenge visual feature dependence.",
        "doctor_focus": "Use instrument silhouette, needle geometry, and controlled camera-relative motion.",
    },
    {
        "id": "glare",
        "title": "Specular glare",
        "difficulty": "Advanced",
        "description": "A bright specular region obscures part of the operative view.",
        "doctor_focus": "Maintain control while important visual texture is temporarily unreliable.",
    },
    {
        "id": "partial_occlusion",
        "title": "Partial occlusion",
        "difficulty": "Advanced",
        "description": "A foreground obstruction masks part of the endoscopic image.",
        "doctor_focus": "Avoid guessing; reposition or hand back when the grasp target is not observable.",
    },
    {
        "id": "combined_visual",
        "title": "Combined visual stress",
        "difficulty": "Research challenge",
        "description": "Camera displacement, reduced light, blur, and glare are applied together.",
        "doctor_focus": "Demonstrate conservative recovery and explicit hand-back under uncertainty.",
    },
    {
        "id": "target_lateral_offset",
        "title": "Lateral target variation",
        "difficulty": "Intermediate",
        "description": "The task object begins 25 mm to the side inside the simulator instead of at the memorized pose.",
        "doctor_focus": "Re-plan the approach from the target that is present, not from the usual starting position.",
    },
    {
        "id": "target_depth_offset",
        "title": "Depth target variation",
        "difficulty": "Advanced",
        "description": "The task object begins 25 mm deeper in the operative workspace inside the simulator.",
        "doctor_focus": "Use parallax and controlled tool motion to confirm depth before closing the gripper.",
    },
    {
        "id": "dropped_object_recovery",
        "title": "Dropped needle recovery",
        "difficulty": "Research challenge",
        "description": "The needle begins displaced from its usual presentation after a simulated drop.",
        "doctor_focus": "Stop, rebuild the camera view, reacquire with controlled contact, and return to stable custody.",
    },
    {
        "id": "calibration_bias",
        "title": "Control calibration bias",
        "difficulty": "Research challenge",
        "description": "A reproducible seven-degree translational calibration bias changes how commands map into the robot frame.",
        "doctor_focus": "Detect the systematic drift, slow down, compensate deliberately, or hand control back.",
    },
    {
        "id": "stereo_miscalibration",
        "title": "Stereo calibration drift",
        "difficulty": "Research challenge",
        "description": "The right endoscope receives a reproducible vertical and baseline error while the left view remains stable.",
        "doctor_focus": "Recognize inconsistent depth cues and rely on deliberate camera motion or hand control back.",
    },
    {
        "id": "sensor_dropout",
        "title": "Intermittent camera dropout",
        "difficulty": "Research challenge",
        "description": "A seeded periodic dropout removes all camera observations for brief intervals.",
        "doctor_focus": "Stop during missing observations and resume only after the operative field returns.",
    },
    {
        "id": "stiff_tissue_response",
        "title": "Stiffer tissue response",
        "difficulty": "Advanced",
        "description": "The bounded OpenUSD surface responds with lower compliance to the same interaction.",
        "doctor_focus": "Compare displacement and force proxies without assuming one tissue model represents clinical material.",
    },
    {
        "id": "anatomy_context",
        "title": "Multi-organ anatomy context",
        "difficulty": "Advanced",
        "description": "The native OpenUSD scene reveals the surrounding CT prostate and bladder geometry beside the liver.",
        "doctor_focus": "Keep attention on the task target while using the additional anatomy as spatial context.",
    },
)
SCENARIOS_BY_ID = {item["id"]: item for item in FAILURE_SCENARIOS}
SCENARIO_NATIVE_PROFILES = {
    "target_lateral_offset": {"object_offset_m": (0.0, 0.025, 0.0)},
    "target_depth_offset": {"object_offset_m": (0.025, 0.0, 0.0)},
    "dropped_object_recovery": {"object_offset_m": (0.035, -0.030, 0.0)},
    "calibration_bias": {"translation_yaw_deg": 7.0, "axis_scale": (1.08, 0.92, 1.0)},
    "stereo_miscalibration": {"right_camera_offset_m": (0.0, 0.010, 0.004)},
    "sensor_dropout": {"dropout_frames": 8, "dropout_period_frames": 40},
    "stiff_tissue_response": {"surface_compliance_scale": 0.45},
    "anatomy_context": {"show_multi_organ": True},
}
RESEARCH_ADVISORY_LIMITS = {
    "contact_force_n": 2.0,
    "tissue_displacement_m": 0.015,
    "deformation_gradient_proxy": 0.50,
}
PROCEDURE_PHASES = {
    "setup": 0,
    "rest": 0,
    "access": 1,
    "approach": 1,
    "align": 2,
    "contact": 3,
    "grasp": 4,
    "manipulate": 5,
    "manipulation": 5,
    "verify": 6,
    "recover": 7,
    "recovery": 7,
}
PROCEDURE_EVENTS = {
    "none": 0,
    "target_visible": 1,
    "contact": 2,
    "grasp": 3,
    "task_complete": 4,
    "handoff": 5,
    "safety_review": 6,
}
OPERATOR_INPUT_SOURCES = {
    "none": 0,
    "keyboard_pointer": 1,
    "gamepad": 2,
    "external_teleop": 3,
    "xr": 4,
    "haptic": 5,
    "keyboard_smart_action": 6,
    "supervised_replay": 7,
    "automation_policy": 8,
    "voice": 9,
    "gamepad_smart_action": 10,
    "voice_smart_action": 11,
    "webcam_hands": 12,
}
NATIVE_NEEDLE_GUIDE_KINDS = {"pickup", "handover", "needle_pass", "recovery"}


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Dr.Anmar Surgical Workstation</title>
  <style>
    :root{color-scheme:dark;--bg:#071016;--panel:#0d1a22;--line:#24404d;--cyan:#2cd2e8;--cyan2:#1795ae;--ink:#e9f8fa;--muted:#88a6b2;--red:#ff5c68;--green:#42e49b}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.35 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI";min-height:100vh}
    header{height:64px;display:flex;align-items:center;gap:14px;padding:0 16px;border-bottom:1px solid var(--line);background:#09151c}
    .brand{font-weight:900;letter-spacing:.08em;white-space:nowrap}.brand span{color:var(--cyan)}
    .live{margin-left:auto;display:flex;align-items:center;gap:7px;color:var(--muted);font-size:13px}.dot{width:9px;height:9px;border-radius:50%;background:var(--red)}.dot.ok{background:var(--green);box-shadow:0 0 12px #42e49b99}
    main{display:grid;grid-template-columns:minmax(0,1fr) 470px;height:calc(100vh - 64px)}
    .view{position:relative;overflow:hidden;background:#020608;display:flex;align-items:center;justify-content:center}.view img{width:100%;height:100%;object-fit:contain}.view-toolbar{padding:9px;border:1px solid #294956;border-radius:10px;background:#07151c}.view-toolbar-row{display:grid;grid-template-columns:62px 1fr;align-items:center;gap:7px}.view-toolbar-row+.view-toolbar-row{margin-top:7px}.view-toolbar-label{color:#6f909b;font:800 9px/1 ui-monospace,SFMono-Regular,Menlo;letter-spacing:.1em;text-transform:uppercase}.camera-tabs,.view-presets{display:grid;gap:5px}.camera-tabs{grid-template-columns:repeat(4,1fr)}.view-presets{grid-template-columns:repeat(9,1fr)}.camera-tabs button,.view-presets button{min-height:32px;padding:0 5px;font-size:9px}.camera-tabs button.active,.view-presets button.active{background:var(--cyan);border-color:var(--cyan);color:#031014}.header-camera-toolbar{min-width:0;flex:1;display:flex;align-items:center;gap:10px;padding:0;border:0;border-radius:0;background:transparent}.header-camera-toolbar .view-toolbar-row{display:flex;min-width:0;gap:6px}.header-camera-toolbar .view-toolbar-row+.view-toolbar-row{margin-top:0}.header-camera-toolbar .camera-tabs,.header-camera-toolbar .view-presets{display:flex;min-width:0;gap:4px}.header-camera-toolbar button{min-width:56px;min-height:34px;padding:0 6px}.header-camera-toolbar .view-presets button{min-width:64px}.gaze-cursor{position:absolute;width:18px;height:18px;border:1px solid #fff;border-radius:50%;translate:-50% -50%;pointer-events:none;opacity:0;box-shadow:0 0 0 3px #2cd2e855}.view.gaze-on .gaze-cursor{opacity:.85}.view.free-camera{cursor:grab;touch-action:none}.view.free-camera.dragging{cursor:grabbing}.free-camera-hud{position:absolute;left:14px;bottom:14px;padding:7px 10px;border:1px solid #3b6472;border-radius:7px;background:#031017d9;color:#dffbff;font:800 10px/1.2 ui-monospace,SFMono-Regular,Menlo;pointer-events:none;backdrop-filter:blur(5px)}
    .aim-reticle{position:absolute;left:50%;top:50%;width:28px;height:28px;translate:-50% -50%;pointer-events:none;opacity:.2}.aim-reticle:before,.aim-reticle:after{content:"";position:absolute;background:#dffcff}.aim-reticle:before{left:0;right:0;top:13px;height:1px}.aim-reticle:after{top:0;bottom:0;left:13px;width:1px}.proximity{margin:0 0 10px;padding:9px 11px;border:1px solid #294956;border-radius:8px;background:#061219;color:#9fc0c9;font:10px/1.45 ui-monospace,SFMono-Regular,Menlo}.proximity b{display:inline;color:var(--ink);font-size:10px;margin-right:7px}.proximity.near{border-color:#7a693d}.proximity.held{border-color:#34715f;color:var(--green)}.proximity.guard{border-color:#2c7180;color:var(--cyan)}.proximity.puncture{border-color:#78483e;color:#ffb09e}
    .recflag{display:none;position:absolute;right:18px;top:18px;color:#fff;background:#c91f2f;padding:8px 12px;border-radius:99px;font-size:12px;font-weight:900;letter-spacing:.08em}.recflag.on{display:block}
    aside{overflow:auto;padding:17px;background:var(--panel);border-left:1px solid var(--line)}
    h2{font-size:12px;letter-spacing:.14em;color:#a9c1ca;margin:3px 0 11px;text-transform:uppercase}.card{border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:13px;background:#0a171e}
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.grid.two{grid-template-columns:repeat(2,1fr)}
    button{min-height:42px;border:1px solid #315462;border-radius:7px;background:#10252e;color:var(--ink);font-weight:750;cursor:pointer;touch-action:manipulation;user-select:none;-webkit-user-select:none}button:hover{border-color:var(--cyan);background:#153540}button:active{transform:translateY(1px);background:var(--cyan2)}
    button.primary{background:var(--cyan);border-color:var(--cyan);color:#041014}button.danger{background:#31171c;border-color:#74414a;color:#ffabb2}button.stop{grid-column:1/-1;background:#27323a;border-color:#5f727c}
    .speedbar{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:11px}.speedbar button{min-height:35px;font-size:11px}.speedbar button.active{background:var(--cyan);border-color:var(--cyan);color:#041014}.dpad{display:grid;grid-template-columns:repeat(3,1fr);grid-template-areas:"blank up blank2" "left stop right" "blank3 down blank4";gap:6px}.dpad .up{grid-area:up}.dpad .left{grid-area:left}.dpad .stop-center{grid-area:stop;min-height:54px;background:#26343b;border-color:#617681}.stop-center small{display:block;color:#9bb0b8;font-size:10px;margin-top:2px}.dpad .right{grid-area:right}.dpad .down{grid-area:down}.depthgrid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}.anglegrid{display:grid;grid-template-columns:1fr 1fr;gap:6px}.move-button{min-height:54px;touch-action:none;position:relative}.move-button small{display:block;color:#86a5af;font-size:10px;margin-top:2px}.move-button.held{background:var(--cyan);border-color:var(--cyan);color:#041014;box-shadow:0 0 16px #2cd2e855}.move-button.held small{color:#0a5260}.control-readout{display:flex;align-items:center;gap:7px;margin-top:10px;color:var(--muted);font-size:11px}.control-readout i{width:7px;height:7px;border-radius:50%;background:#536a73}.control-readout.moving{color:var(--green)}.control-readout.moving i{background:var(--green);box-shadow:0 0 10px #42e49b99}
    .hint{color:var(--muted);font-size:12px;margin-top:9px}.hidden{display:none}.arm.active,.autonomy.active{background:var(--cyan);color:#041014;border-color:var(--cyan)}
    .procedure-title{font-size:15px;font-weight:850}.procedure-objective{color:#b9ccd2;font-size:11px;margin:6px 0 10px}.procedure-progress{height:4px;background:#19313b;margin:8px 0}.procedure-progress i{display:block;height:100%;background:var(--cyan);width:0}.procedure-step{display:grid;grid-template-columns:21px 1fr;gap:7px;padding:6px 0;border-top:1px solid #19313b;color:#738d96;font-size:10px}.procedure-step b{color:#9eb5bd}.procedure-step.complete b{color:var(--green)}.procedure-step.active b{color:var(--cyan)}.procedure-step span:first-child{font:10px ui-monospace,monospace}.patient-access{width:100%;min-height:38px;margin-top:8px;background:#963f46;border-color:#cf737a;color:#fff;text-align:left}.patient-access.open{background:#183c33;border-color:#3b7a67;color:#baf6df}.patient-access:disabled{cursor:not-allowed;opacity:.8}
    .supervision{border-color:#356475;background:linear-gradient(135deg,#0d2731,#09171e)}.supervision-state{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:9px}.supervision-state b{color:var(--cyan)}.cue{min-height:32px;margin-top:9px;padding:8px;border-left:2px solid var(--cyan);background:#061219;color:#9fc0c9;font-size:11px}
    .safety-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.safety-metric{padding:8px;background:#061219;border:1px solid #1c3742}.safety-metric b{display:block;color:var(--green);font:15px ui-monospace,monospace}.safety-metric span{color:var(--muted);font-size:9px}
    .stapler-cell{padding:12px;border:1px solid #5d6140;border-radius:11px;background:linear-gradient(135deg,#252417,#151b1f)}#staplerCell.hidden{display:none!important}.stapler-cell-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:9px}.stapler-cell-head b{display:block;color:#f1e6c1;font-size:14px}.stapler-cell-head small{display:block;margin-top:2px;color:#aaa58d;font-size:9px}.stapler-phase{padding:4px 7px;border-radius:5px;background:#393622;color:#f0cf77;font:800 8px ui-monospace,monospace;text-transform:uppercase}.stapler-metrics{display:grid;grid-template-columns:repeat(9,1fr);gap:5px}.stapler-metric{padding:7px;border:1px solid #4c4930;border-radius:6px;background:#181b18}.stapler-metric b{display:block;color:#e9d98e;font:14px ui-monospace,monospace}.stapler-metric span{color:#928f7d;font-size:7px;letter-spacing:.05em}.stapler-progress{height:5px;margin-top:7px;overflow:hidden;border-radius:4px;background:#111411}.stapler-progress i{display:block;width:0;height:100%;background:#d8b750;transition:width .25s ease}.stapler-controls{display:grid;grid-template-columns:.8fr 1.6fr .8fr repeat(3,1fr);gap:6px;margin-top:8px}#closureRobotCell .stapler-controls{grid-template-columns:repeat(3,1fr)}.stapler-target{display:grid;grid-column:span 3;grid-template-columns:1fr auto;align-items:center;gap:4px 8px;padding:6px 8px;border:1px solid #4c4930;border-radius:7px;background:#181b18}.stapler-target b{color:#d9d3b6;font-size:9px}.stapler-target output{color:#e9d98e;font:12px ui-monospace,monospace}.stapler-target input{grid-column:1/-1;width:100%;accent-color:#d8b750}.stapler-controls button{min-height:42px;background:#29291d;border-color:#565135;font-size:9px}.stapler-controls button.primary{background:#d8b750;border-color:#d8b750;color:#221f10}.stapler-controls button:disabled{cursor:not-allowed;opacity:.45}.stapler-boundary{margin:7px 0 0;color:#8f8c7f;font-size:8px}
    .control-dock{position:relative;margin:0 0 10px;padding:34px 10px 8px;border:1px solid #294651;border-radius:9px;background:#0a171e;box-shadow:none}.control-dock:before{content:"Robot controls";position:absolute;left:12px;top:10px;color:#dffbff;font:800 12px/1 ui-sans-serif,system-ui}.control-dock:after{display:none}.control-dock .move-button{min-height:43px;padding:4px 2px;border:1px solid #31515d;background:#0d2028;font-size:10px;line-height:1.05}.control-dock .move-button small{font-size:8px;margin-top:2px}.control-dock .stop-center{width:100%;min-height:34px;padding:3px 8px;border:1px solid #68444b;background:#25181c;color:#ffc2c7;font-size:9px}.control-dock .hint{display:none}.instrument-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.instrument-grid.single{grid-template-columns:1fr}.instrument-card{min-width:0;padding:7px;border:1px solid #1d3540;border-radius:8px;background:#08131a}.instrument-head{display:flex;align-items:center;gap:7px;margin-bottom:5px}.instrument-head button{flex:1;min-height:30px;padding:3px 7px;text-align:left;font-size:10px}.instrument-head .arm.active{border-color:#426775;background:#132a33;color:#dffbff}.instrument-head span{color:#708b95;font:750 8px/1 ui-monospace,monospace;letter-spacing:.08em;white-space:nowrap}.hand-key-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}.hand-key{display:flex;flex-direction:column;align-items:center;justify-content:center}.hand-key kbd{height:18px;min-width:22px;padding:0 4px;font-size:9px}.instrument-actions{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:5px}.instrument-actions .modifier-chip,.instrument-actions button{min-height:29px;display:flex;align-items:center;justify-content:center;gap:3px;padding:2px;font-size:8px}.instrument-actions button,.instrument-actions .primary{border-color:#31515d;background:#0d2028;color:#dffbff}.direct-roll{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:4px}.direct-roll .move-button{min-height:29px;font-size:8px}.direct-roll kbd{height:16px;min-width:18px;padding:0 3px;font-size:8px}.hand-speeds{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:4px}.hand-speeds button{min-height:29px;padding:2px;font-size:8px}.hand-speeds button.active{border-color:#527480;background:#132a33}.hand-speeds kbd{height:16px;min-width:17px;padding:0 3px;font-size:8px}.control-stop-row{margin-top:7px;padding-top:6px;border-top:1px solid #1d3540}.control-dock .control-readout{min-height:15px;margin-top:3px;font-size:8px}.control-dock .control-readout i{width:5px;height:5px}
    kbd{display:inline-flex;align-items:center;justify-content:center;min-width:22px;height:20px;padding:0 6px;border:1px solid #4a6570;border-bottom-width:2px;border-radius:5px;background:#09141a;color:#dffbff;font:800 10px/1 ui-monospace,SFMono-Regular,Menlo;white-space:nowrap}button kbd{pointer-events:none}.header-keyboard{min-height:32px;margin-left:4px;padding:0 10px;background:#10252e;color:#cfe7eb;font-size:11px}.header-keyboard kbd{margin-right:5px}.keyboard-quick{display:grid;grid-template-columns:.9fr 1.1fr;gap:5px;margin:0;padding:0;border:0;background:transparent}.keyboard-quick-head{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;gap:8px;margin:0 0 1px}.keyboard-quick-head b{color:#8eabb5;font-size:10px;font-weight:750;letter-spacing:.04em}.keyboard-quick-head span{display:none}.keyboard-input-display{display:flex;align-items:center;gap:6px;min-height:38px;margin:0;padding:5px 7px;border:1px solid #1d3540;border-radius:6px;background:#08131a;color:#a7bbc2;font-size:10px}.keyboard-input-display kbd{min-width:38px;height:18px;font-size:8px;color:var(--green);border-color:#3b7a67}.keyboard-input-display.active{border-color:var(--green);box-shadow:none}.keyboard-input-display.active span{color:#e5ffff}.smart-action{width:100%;min-height:38px;margin:0;background:#2fc5d8;border-color:#52d7e8;color:#031014;text-align:left;padding:5px 8px;box-shadow:none}.smart-action strong{display:block;font-size:11px}.smart-action strong kbd{height:17px;min-width:24px;padding:0 4px;font-size:8px}.smart-action small{display:block;overflow:hidden;color:#174851;font-size:8px;line-height:1.1;white-space:nowrap;text-overflow:ellipsis}.proximity{grid-column:1/-1;margin:0;padding:4px 7px;border:0;border-radius:5px;background:#071219;font-size:9px;line-height:1.2}.proximity b{font-size:9px;margin-right:5px}.control-feel{grid-column:1/-1;display:flex;align-items:center;justify-content:center;gap:7px;min-height:23px;border:1px solid #1d3540;border-radius:5px;background:#071219;color:#86a5af;font:8px/1 ui-monospace,SFMono-Regular,Menlo}.control-feel b{color:#dffbff}.modifier-row{grid-column:1/-1;display:flex;gap:4px;margin:0}.modifier-chip{flex:1;padding:3px;border:0;border-radius:5px;background:#071219;color:#829aa3;font-size:9px;text-align:center}.modifier-chip kbd{height:16px;min-width:20px;padding:0 3px;font-size:8px}.modifier-chip.active{color:var(--green);background:#0b2b25}.keyboard-coverage{display:none}.keyboard-coverage.bad{color:var(--red)}button.key-active,button.state-active{border-color:var(--green)!important;box-shadow:0 0 0 1px #42e49b77!important;background:#174a42!important;color:#efffff!important}button.key-active kbd,button.state-active kbd{border-color:#9bffe0;background:#dcfff5;color:#09281f}.smart-action.key-active{background:#8bffe0!important;color:#041a13!important;transform:none}
    .teleop-strip{grid-column:1/-1;display:grid;grid-template-columns:1.05fr .95fr;gap:5px}.gamepad-status{min-width:0;min-height:42px;padding:4px 7px;display:grid;grid-template-columns:7px minmax(0,1fr) auto;align-items:center;gap:7px;text-align:left;border-color:#294b57;background:#081820;color:#a9c2ca;overflow:hidden}.gamepad-status.connected{border-color:#387c68;color:var(--green);background:linear-gradient(120deg,#0a241f,#081820)}.gamepad-status.mode{border-color:var(--cyan);box-shadow:inset 0 0 12px #2cd2e817}.gamepad-dot{width:7px;height:7px;border-radius:50%;background:#60767d}.gamepad-status.connected .gamepad-dot{background:var(--green);box-shadow:0 0 8px #42e49baa}.gamepad-copy{min-width:0}.gamepad-copy b,.gamepad-copy small{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.gamepad-copy b{color:#dffbff;font:800 9px/1.1 ui-sans-serif,system-ui}.gamepad-copy small{margin-top:2px;color:#7f9ca5;font:8px/1.1 ui-monospace,SFMono-Regular,Menlo}.gamepad-status.connected .gamepad-copy small{color:#89bbae}.gamepad-sticks{display:flex;gap:4px}.stick-meter{position:relative;width:20px;height:20px;border:1px solid #3c5c66;border-radius:50%;background:#061219}.stick-meter:before,.stick-meter:after{content:"";position:absolute;background:#294550}.stick-meter:before{left:3px;right:3px;top:9px;height:1px}.stick-meter:after{top:3px;bottom:3px;left:9px;width:1px}.stick-meter i{position:absolute;left:7px;top:7px;width:5px;height:5px;border-radius:50%;background:#66828b;transition:transform 45ms linear}.gamepad-status.connected .stick-meter i{background:var(--cyan);box-shadow:0 0 5px #2cd2e899}.voice-form{display:grid;grid-template-columns:minmax(0,1fr) 36px 36px;gap:4px}.voice-form input{min-width:0;height:34px;padding:0 8px;border:1px solid #294b57;border-radius:6px;background:#061219;color:#ddf7fa;font:9px/1 ui-sans-serif,system-ui}.voice-form input:focus{outline:1px solid var(--cyan);border-color:var(--cyan)}.voice-form button{min-height:34px;padding:0;font-size:12px}.voice-mic.listening{border-color:#ff8b93;background:#4b1f28;color:#fff;box-shadow:0 0 12px #ff4f6670}.voice-status{grid-column:1/-1;min-height:8px;color:#718f99;font:8px/1.1 ui-monospace,SFMono-Regular,Menlo;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.voice-status.listening{color:#ff9da5}.voice-status.ok{color:var(--green)}.voice-status.error{color:#ffb1b6}
    .keyboard-help{position:fixed;inset:0;z-index:50;display:grid;place-items:center;padding:24px;background:#02080dd9;backdrop-filter:blur(7px)}.keyboard-help.hidden{display:none}.keyboard-help-panel{width:min(940px,96vw);max-height:90vh;overflow:auto;border:1px solid #4c7c8d;border-radius:14px;background:#09171e;box-shadow:0 24px 90px #000;padding:18px}.keyboard-help-head{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:14px}.keyboard-help-head h1{margin:0;color:#e7fbfd;font-size:21px}.keyboard-help-head p{margin:3px 0 0;color:var(--muted);font-size:11px}.keyboard-help-head button{min-height:36px;padding:0 12px}.shortcut-columns{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.shortcut-group{padding:12px;border:1px solid #203e49;border-radius:9px;background:#071219}.shortcut-group h3{margin:0 0 8px;color:var(--cyan);font-size:11px;letter-spacing:.11em}.shortcut-line{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:5px 0;border-top:1px solid #152c35;color:#bdd1d7;font-size:10px}.shortcut-line:first-of-type{border-top:0}.shortcut-line span{text-align:right}.shortcut-group.wide{grid-column:span 3}.shortcut-group.wide .shortcut-list{display:grid;grid-template-columns:repeat(3,1fr);gap:0 14px}
    #toast{position:fixed;left:50%;bottom:20px;translate:-50% 20px;opacity:0;background:#e9f8fa;color:#061116;border-radius:8px;padding:10px 15px;font-weight:750;transition:.2s;pointer-events:none}#toast.show{opacity:1;translate:-50% 0}
    @media(max-width:1250px){.header-camera-toolbar .view-toolbar-label{display:none}.header-camera-toolbar button{min-width:48px;padding:0 4px}.header-camera-toolbar .view-presets button{min-width:54px}.header-camera-toolbar kbd{display:none}}
    @media(max-width:1100px){main{grid-template-columns:minmax(0,1fr) 430px}.shortcut-columns{grid-template-columns:repeat(2,1fr)}.shortcut-group.wide{grid-column:span 2}.shortcut-group.wide .shortcut-list{grid-template-columns:repeat(2,1fr)}.header-keyboard{display:none}}
    @media(max-width:880px){header{height:auto;min-height:64px;padding:7px 12px;flex-wrap:wrap}.header-camera-toolbar{order:3;flex-basis:100%;overflow-x:auto;padding-bottom:2px}.header-camera-toolbar .view-toolbar-label{display:block}.header-camera-toolbar kbd{display:inline-flex}.header-camera-toolbar button{min-width:62px}.header-camera-toolbar .view-presets button{min-width:70px}main{display:block;height:auto}.view{height:52vh}aside{border-left:0;border-top:1px solid var(--line)}}
    /* Calm camera-first workstation: the live view owns the canvas; details open below it. */
    :root{--bg:#101518;--panel:#171d21;--line:#303a40;--cyan:#8bc6cd;--cyan2:#5f9ca4;--ink:#edf1f2;--muted:#a3afb3;--red:#d77b77;--green:#79c8a2}
    body{background:var(--bg);font:14px/1.42 "Avenir Next","SF Pro Text",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    header{height:40px;gap:5px;padding:4px 6px;background:#171d21;border-bottom-color:#303a40;overflow:hidden}
    header .brand{display:none}
    .live{font-size:0;gap:0;margin-left:3px}.live #connection{display:none}.dot{width:6px;height:6px}.dot.ok{box-shadow:none}
    .header-camera-toolbar{gap:4px;overflow-x:auto;scrollbar-width:none}.header-camera-toolbar::-webkit-scrollbar{display:none}.header-camera-toolbar .view-toolbar-row{gap:3px}.header-camera-toolbar .camera-tabs,.header-camera-toolbar .view-presets{gap:3px}.header-camera-toolbar button{min-width:44px;min-height:26px;padding:0 7px;border-radius:6px;background:#222a2f;border-color:#354147;color:#dce3e5;font-size:10px;font-weight:650}.header-camera-toolbar .view-presets button{min-width:48px}.header-camera-toolbar button:hover{background:#2b3439;border-color:#71979d}.header-camera-toolbar .camera-tabs button.active,.header-camera-toolbar .view-presets button.active{background:#8bc6cd;border-color:#8bc6cd;color:#152126}
    .header-keyboard{min-width:28px;min-height:26px;margin-left:2px;padding:0 4px;border-radius:6px;background:#222a2f;border-color:#354147;color:#dce4e6;font-size:0}.header-keyboard kbd{width:18px;min-width:18px;height:18px;margin:0;padding:0;border:0;background:transparent;font-size:10px}
    main{display:grid;grid-template-columns:1fr;grid-template-rows:minmax(0,1fr) auto;height:calc(100vh - 40px);overflow:hidden}
    .view{min-height:0;background:#090c0e}.view img{object-fit:contain}.free-camera-hud{left:12px;bottom:12px;border-color:#4b5d64;border-radius:9px;background:#11181cdd;color:#d9e2e4;font-weight:650}
    aside{max-height:52px;overflow:hidden;padding:5px 7px;border-left:0;border-top:1px solid #303a40;background:#171d21;transition:max-height .22s ease}
    aside>:not(.workstation-dockbar){display:none}
    .workstation-dockbar{display:flex;align-items:center;gap:6px;min-height:40px}
    .dock-hands{display:flex;align-items:center;gap:6px;min-width:0;flex:1}
    .dock-hand{display:flex;align-items:center;gap:6px;min-width:0;padding:4px 7px;border:1px solid #354147;border-radius:7px;background:#1d252a;color:#b7c2c5}
    .dock-hand b{color:#eff3f4;font-size:10px;white-space:nowrap}.dock-hand span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:9px}
    .dock-actions{display:flex;align-items:center;gap:4px}.dock-actions button{min-height:30px;padding:0 8px;border-radius:7px;background:#242d32;border-color:#3c494f;color:#e9eef0;font-size:10px}.dock-actions .primary{background:#8bc6cd;border-color:#8bc6cd;color:#142126}
    body.panel-open aside{max-height:min(50vh,500px);overflow:auto;display:grid;grid-template-columns:1fr;gap:10px;align-items:start}
    body.panel-open aside>.workstation-dockbar{grid-column:1/-1}
    body.panel-open aside>.control-dock{display:block;grid-column:1/-1}
    body.panel-open aside>.stapler-cell{display:block;grid-column:1/-1}
    body.panel-open aside>.session-details{display:block;grid-column:1/-1}
    .control-dock{margin:0;padding:36px 10px 9px;border-color:#354147;border-radius:12px;background:#1b2328}.control-dock:before{content:"Instrument controls";color:#dfe6e8;font-family:inherit;font-weight:720}
    .control-dock .move-button,.instrument-actions button,.instrument-actions .primary,.direct-roll .move-button,.hand-speeds button{background:#242d32;border-color:#3a484f;color:#e7edef}
    .instrument-card{border-color:#354147;border-radius:10px;background:#171e22}.instrument-head .arm.active,.hand-speeds button.active{border-color:#6d949a;background:#2b3a40;color:#f2f6f7}
    .teleop-strip,.control-feel{display:none}.proximity{background:#141a1e;color:#a8b5b9}
    .session-details{border:1px solid #354147;border-radius:12px;background:#1a2125}.session-details>summary{cursor:pointer;padding:10px 12px;color:#c9d2d5;font-size:11px;font-weight:700;list-style:none}.session-details>summary::-webkit-details-marker{display:none}.session-details>summary:after{content:"Show";float:right;color:#8bc6cd;font-weight:650}.session-details[open]>summary:after{content:"Hide"}.session-details-grid{display:grid;grid-template-columns:1.25fr .8fr .8fr 1fr;gap:8px;padding:0 10px 10px}.session-section h2{margin:0 0 6px;color:#9eaaae;font-size:9px;letter-spacing:.08em}.session-section .card{height:calc(100% - 18px);margin:0;padding:9px;border-color:#354147;background:#171e22}.procedure-title{font-size:13px}.procedure-objective{font-size:10px}.procedure-step{padding:4px 0}.safety-grid{grid-template-columns:1fr}.safety-metric{padding:5px 7px;background:#141a1e;border-color:#2f393e}.cue{background:#141a1e}
    button{border-radius:9px;font-family:inherit}button.primary{background:#8bc6cd;border-color:#8bc6cd;color:#142126}button:hover{border-color:#73999f;background:#2c373d}
    kbd{border-color:#526168;background:#141a1e;color:#e6edef}
    @media(max-width:920px){header{height:40px;min-height:40px;flex-wrap:nowrap}.header-camera-toolbar{order:initial;flex-basis:auto;overflow-x:auto}.live{display:flex}.header-keyboard{display:flex}main{display:grid;grid-template-columns:1fr;grid-template-rows:minmax(0,1fr) auto;height:calc(100vh - 40px)}.view{height:auto}.workstation-dockbar{overflow-x:auto}.dock-hand span{display:none}body.panel-open aside{grid-template-columns:1fr}.control-dock,.stapler-cell,.session-details{grid-column:1!important}.stapler-metrics{grid-template-columns:repeat(2,1fr)}.stapler-controls{grid-template-columns:1fr 1fr}.stapler-target{grid-column:1/-1}.session-details-grid{grid-template-columns:1fr 1fr}}
  </style>
</head>
<body>
<header><div class="brand">DR.<span>ANMAR</span></div><section class="view-toolbar header-camera-toolbar" aria-label="Camera controls"><div class="view-toolbar-row"><span class="view-toolbar-label">Camera</span><div class="camera-tabs"><button class="active" data-camera="endoscope_left" data-shortcut="4" onclick="setCamera('endoscope_left',this)">Left <kbd>4</kbd></button><button data-camera="endoscope_right" data-shortcut="5" onclick="setCamera('endoscope_right',this)">Right <kbd>5</kbd></button><button data-camera="wrist_1" data-shortcut="6" onclick="setCamera('wrist_1',this)">Wrist 1 <kbd>6</kbd></button><button id="wrist2Tab" class="hidden" data-camera="wrist_2" data-shortcut="7" onclick="setCamera('wrist_2',this)">Wrist 2 <kbd>7</kbd></button><button id="nvidiaEcmTab" class="hidden" data-camera="nvidia_ecm" data-shortcut="C" onclick="setCamera('nvidia_ecm',this)">ECM <kbd>C</kbd></button></div></div><div class="view-toolbar-row"><span class="view-toolbar-label">Angle</span><div class="view-presets"><button data-view-mode="operative" data-shortcut="F1" onclick="setCameraView('operative',this)">Operative <kbd>F1</kbd></button><button data-view-mode="close" data-shortcut="F2" onclick="setCameraView('close',this)">Close <kbd>F2</kbd></button><button data-view-mode="overview" data-shortcut="F3" onclick="setCameraView('overview',this)">Wide <kbd>F3</kbd></button><button data-view-mode="overhead" data-shortcut="F4" onclick="setCameraView('overhead',this)">Overhead <kbd>F4</kbd></button><button data-view-mode="left_oblique" data-shortcut="F5" onclick="setCameraView('left_oblique',this)">Left angle <kbd>F5</kbd></button><button data-view-mode="right_oblique" data-shortcut="F6" onclick="setCameraView('right_oblique',this)">Right angle <kbd>F6</kbd></button><button data-view-mode="opposite" data-shortcut="F7" onclick="setCameraView('opposite',this)">Opposite <kbd>F7</kbd></button><button id="freeCameraButton" class="active" data-shortcut="F8" onclick="toggleFreeCamera()">Free <kbd>F8</kbd></button><button id="resetCameraButton" class="state-active" data-shortcut="Home" onclick="resetFreeCamera()">Reset <kbd>Home</kbd></button></div></div></section><button class="header-keyboard" aria-label="Keyboard shortcuts" data-shortcut="?" onclick="toggleKeyboardHelp()"><kbd>?</kbd></button><div class="live"><i id="dot" class="dot"></i><span id="connection">Connecting…</span></div></header>
<main>
  <section id="cameraView" class="view free-camera"><img id="cameraImage" alt="Live simulated medical sensor view"><div id="recflag" class="recflag">● RECORDING</div><div id="gazeCursor" class="gaze-cursor"></div><div class="aim-reticle"></div><div id="freeCameraHud" class="free-camera-hud">Drag orbit · Shift-drag pan · wheel zoom</div></section>
  <aside>
    <div class="workstation-dockbar">
      <div class="dock-hands"><div class="dock-hand"><b>Left instrument</b><span>QWE · ASD · Space grip</span></div><div class="dock-hand"><b>Right instrument</b><span>UIO · JKL · Enter grip</span></div></div>
      <div class="dock-actions"><button id="panelToggle" data-shortcut="Tab" aria-expanded="false" onclick="toggleControlPanel()">Controls <kbd>Tab</kbd></button></div>
    </div>
    <section class="control-dock">
      <div class="keyboard-quick"><div class="keyboard-quick-head"><b>Two-hand surgical controls</b><span>Each hand owns one robot · release = stop</span></div><div id="keyActionDisplay" class="keyboard-input-display" aria-live="polite"><kbd>READY</kbd><span>Tap to nudge · hold to glide</span></div><button id="smartActionButton" class="smart-action" data-shortcut="F12" onclick="smartAction()"><strong><kbd>F12</kbd> Smart assist</strong><small id="smartActionLabel">Nudge toward the target</small></button><div id="proximity" class="proximity"><b>Next</b><span>Acquiring target…</span></div><div class="teleop-strip"><button id="gamepadStatus" class="gamepad-status" data-shortcut="?" onclick="toggleKeyboardHelp(true)" aria-label="Xbox controller status and map"><span class="gamepad-dot"></span><span class="gamepad-copy"><b id="gamepadTitle">Connect Xbox controller</b><small id="gamepadMode">One pad · both robots</small></span><span class="gamepad-sticks" aria-hidden="true"><span class="stick-meter"><i id="gamepadLeftStick"></i></span><span class="stick-meter"><i id="gamepadRightStick"></i></span></span></button><form id="voiceForm" class="voice-form" onsubmit="submitVoiceCommand(event)"><input id="voiceCommand" autocomplete="off" spellcheck="false" aria-label="Voice or typed robot command" placeholder="Say or type: left robot up"><button id="voiceMic" class="voice-mic" type="button" data-shortcut="`" aria-label="Hold to talk">●</button><button type="submit" data-shortcut="↵" aria-label="Run typed command">↵</button><div id="voiceStatus" class="voice-status" aria-live="polite">Push to talk or type a bounded command</div></form></div><div class="control-feel"><b>Game feel</b><span>tap = micro · hold = smooth speed · Option = fine</span></div><div id="keyboardCoverage" class="keyboard-coverage">Auditing keyboard coverage…</div></div>
      <div id="instrumentGrid" class="instrument-grid">
        <section class="instrument-card left-instrument"><div class="instrument-head"><button id="arm0" class="arm active" data-shortcut="[" onclick="setArm(0)">Instrument 1 <kbd>[</kbd></button><span>LEFT HAND</span></div><div class="hand-key-grid">
          <button class="move-button hand-key" data-arm="0" data-key="KeyQ" data-shortcut="Q" data-axis="0" data-direction="-1"><kbd>Q</kbd><small>Toward</small></button><button class="move-button hand-key" data-arm="0" data-key="KeyW" data-shortcut="W" data-axis="2" data-direction="1"><kbd>W</kbd><small>Up</small></button><button class="move-button hand-key" data-arm="0" data-key="KeyE" data-shortcut="E" data-axis="0" data-direction="1"><kbd>E</kbd><small>Away</small></button>
          <button class="move-button hand-key" data-arm="0" data-key="KeyA" data-shortcut="A" data-axis="1" data-direction="1"><kbd>A</kbd><small>Left</small></button><button class="move-button hand-key" data-arm="0" data-key="KeyS" data-shortcut="S" data-axis="2" data-direction="-1"><kbd>S</kbd><small>Down</small></button><button class="move-button hand-key" data-arm="0" data-key="KeyD" data-shortcut="D" data-axis="1" data-direction="-1"><kbd>D</kbd><small>Right</small></button>
        </div><div class="instrument-actions"><div id="leftRotateModifier" class="modifier-chip"><kbd>L⇧</kbd> Angle</div><div id="leftPrecisionModifier" class="modifier-chip"><kbd>L⌥</kbd> Fine</div><button id="gripOpenButton" class="gripper-control" data-shortcut="Space" onclick="toggleGrip(0)"><kbd>Space</kbd> Grip</button></div><div class="direct-roll"><button class="move-button" data-arm="0" data-key="KeyZ" data-shortcut="Z" data-axis="3" data-direction="-1"><kbd>Z</kbd> Roll ↶</button><button class="move-button" data-arm="0" data-key="KeyX" data-shortcut="X" data-axis="3" data-direction="1"><kbd>X</kbd> Roll ↷</button></div><div class="hand-speeds"><button data-hand-speed-arm="0" data-hand-speed=".35" data-shortcut="1" onclick="setHandSpeed(0,.35,'1')"><kbd>1</kbd> Fine</button><button class="active" data-hand-speed-arm="0" data-hand-speed="1" data-shortcut="2" onclick="setHandSpeed(0,1,'2')"><kbd>2</kbd> Normal</button><button data-hand-speed-arm="0" data-hand-speed="1.7" data-shortcut="3" onclick="setHandSpeed(0,1.7,'3')"><kbd>3</kbd> Fast</button></div></section>
        <section id="rightInstrumentControls" class="instrument-card right-instrument"><div class="instrument-head"><button id="arm1" class="arm" data-shortcut="]" onclick="setArm(1)">Instrument 2 <kbd>]</kbd></button><span>RIGHT HAND</span></div><div class="hand-key-grid">
          <button class="move-button hand-key" data-arm="1" data-key="KeyU" data-shortcut="U" data-axis="0" data-direction="-1"><kbd>U</kbd><small>Toward</small></button><button class="move-button hand-key" data-arm="1" data-key="KeyI" data-shortcut="I" data-axis="2" data-direction="1"><kbd>I</kbd><small>Up</small></button><button class="move-button hand-key" data-arm="1" data-key="KeyO" data-shortcut="O" data-axis="0" data-direction="1"><kbd>O</kbd><small>Away</small></button>
          <button class="move-button hand-key" data-arm="1" data-key="KeyJ" data-shortcut="J" data-axis="1" data-direction="1"><kbd>J</kbd><small>Left</small></button><button class="move-button hand-key" data-arm="1" data-key="KeyK" data-shortcut="K" data-axis="2" data-direction="-1"><kbd>K</kbd><small>Down</small></button><button class="move-button hand-key" data-arm="1" data-key="KeyL" data-shortcut="L" data-axis="1" data-direction="-1"><kbd>L</kbd><small>Right</small></button>
        </div><div class="instrument-actions"><div id="rightRotateModifier" class="modifier-chip"><kbd>R⇧</kbd> Angle</div><div id="rightPrecisionModifier" class="modifier-chip"><kbd>R⌥</kbd> Fine</div><button id="gripCloseButton" class="gripper-control primary" data-shortcut="Enter" onclick="toggleGrip(1)"><kbd>Enter</kbd> Grip</button></div><div class="direct-roll"><button class="move-button" data-arm="1" data-key="KeyN" data-shortcut="N" data-axis="3" data-direction="-1"><kbd>N</kbd> Roll ↶</button><button class="move-button" data-arm="1" data-key="KeyM" data-shortcut="M" data-axis="3" data-direction="1"><kbd>M</kbd> Roll ↷</button></div><div class="hand-speeds"><button data-hand-speed-arm="1" data-hand-speed=".35" data-shortcut="8" onclick="setHandSpeed(1,.35,'8')"><kbd>8</kbd> Fine</button><button class="active" data-hand-speed-arm="1" data-hand-speed="1" data-shortcut="9" onclick="setHandSpeed(1,1,'9')"><kbd>9</kbd> Normal</button><button data-hand-speed-arm="1" data-hand-speed="1.7" data-shortcut="0" onclick="setHandSpeed(1,1.7,'0')"><kbd>0</kbd> Fast</button></div></section>
      </div><div class="control-stop-row"><button class="stop-center" data-shortcut="Esc" onclick="emergencyStop()">Stop both robots <kbd>Esc / ⌫</kbd></button></div><div id="controlReadout" class="control-readout" aria-live="polite"><i></i><span>Ready · hold a key to move either instrument</span></div>
    </section>
    <section id="closureRobotCell" class="stapler-cell hidden" aria-label="Approximate staple seal robot controls">
      <div class="stapler-cell-head"><div><b>Dr.Anmar approximate–staple–seal robot</b><small>Franka link8 mount · surface-deformable tissue · measured physical phase gates</small></div><span id="closureRobotPhase" class="stapler-phase">READY</span></div>
      <div class="stapler-metrics"><div class="stapler-metric"><b id="closureRobotApproximation">0 / 0 mm</b><span>CARRIAGES L / R</span></div><div class="stapler-metric"><b id="closureRobotClamps">28 / −28°</b><span>CLAMPS L / R</span></div><div class="stapler-metric"><b id="closureRobotDriver">0.0 mm</b><span>STAPLE DRIVER</span></div><div class="stapler-metric"><b id="closureRobotStaple">0 · 0 bonds</b><span>RETAINED STAPLE</span></div><div class="stapler-metric"><b id="closureRobotAdhesive">0.0 / 0.0 mm</b><span>DEPLOY / METER</span></div><div class="stapler-metric"><b id="closureRobotBonds">0 · 0 bonds</b><span>ADHESIVE BEAD</span></div><div class="stapler-metric"><b id="closureRobotCapture">0</b><span>TEMP CAPTURE</span></div><div class="stapler-metric"><b id="closureRobotPhysics">PHYSX</b><span>TISSUE AUTHORITY</span></div></div>
      <div class="stapler-controls"><button id="closureRobotRun" class="primary" data-shortcut="CLOSE-RUN" onclick="closureRobotCommand('run')">Run complete<br>physical closure</button><button id="closureRobotStop" data-shortcut="CLOSE-STOP" onclick="closureRobotCommand('stop')">Hold mechanism<br>keep attachments</button><button data-shortcut="CLOSE-RESET" onclick="closureRobotCommand('reset')">Reset robot<br>and closure</button></div>
      <p class="stapler-boundary">The stock Panda hand and fingers are inactive; the payload is fixed directly to panda_link8. Clamp capture, the two formed-staple legs and all six cured-bead regions use PhysxPhysicsAttachment. The runtime never rewrites tissue transforms or nodal positions. Penetration, metal forming, adhesive chemistry, damage and clinical strength are not claimed.</p>
    </section>
    <section id="staplerCell" class="stapler-cell hidden" aria-label="Stapler test cell controls">
      <div class="stapler-cell-head"><div><b>Dr.Anmar physical tissue closure bench</b><small>PhysX FEM tissue · pre-fire wound approximation · rigid retained staples · no robot grip required</small></div><span id="staplerPhase" class="stapler-phase">READY</span></div>
      <div class="stapler-metrics"><div class="stapler-metric"><b id="staplerStation">1 / 7</b><span>CLOSURE STATION</span></div><div class="stapler-metric"><b id="staplerClosure">0 / 7</b><span>STAPLES PLACED</span></div><div class="stapler-metric"><b id="staplerGap">—</b><span>LIVE TISSUE GAP</span></div><div class="stapler-metric"><b id="staplerApproximation">0%</b><span>APPROXIMATION</span></div><div class="stapler-metric"><b id="staplerSpacing">6.0 mm</b><span>GUIDED SPACING</span></div><div class="stapler-metric"><b id="staplerTrigger">0.0°</b><span>ACTUAL TRIGGER</span></div><div class="stapler-metric"><b id="staplerPusher">0.0 mm</b><span>PUSHER TRAVEL</span></div><div class="stapler-metric"><b id="staplerMagazine">35 / 35</b><span>MAGAZINE</span></div><div class="stapler-metric"><b id="staplerRetention">OPEN</b><span>PHYSICAL RETENTION</span></div></div>
      <div class="stapler-progress" aria-label="Closure placement progress"><i id="staplerProgress"></i></div>
      <div class="stapler-controls"><button id="staplerPrevious" data-shortcut="CELL-PREV" onclick="staplerCommand('previous_station')">← Previous<br>station</button><button id="staplerFire" class="primary" data-shortcut="CELL-FIRE" onclick="runStaplerCycle()">Staple &amp; advance<br>one full cycle</button><button id="staplerNext" data-shortcut="CELL-NEXT" onclick="staplerCommand('next_station')">Next<br>station →</button><button data-shortcut="CELL-20" onclick="setStaplerTarget(20)">Mechanism check<br>partial 20°</button><button data-shortcut="CELL-RELEASE" onclick="staplerCommand('release')">Release<br>&lt; 8°</button><button data-shortcut="CELL-RESET" onclick="staplerCommand('reset')">Reset closure</button></div>
      <p class="stapler-boundary">The tissue is a live PhysX FEM body. Each cycle first approximates both wound edges, then a collision-enabled rigid staple retains two local FEM attachment bands after release. The backend does not cut tissue or model needle penetration, metal plastic forming, calibrated pullout strength or clinical performance.</p>
    </section>
    <details class="session-details"><summary>Procedure details and session tools</summary><div class="session-details-grid">
      <section class="session-section"><h2>Procedure</h2><div class="card"><div id="procedureTitle" class="procedure-title">Free practice</div><div id="procedureObjective" class="procedure-objective">Use the robot controls to explore the digital twin.</div><div class="procedure-progress"><i id="procedureProgress"></i></div><div id="procedureSteps"></div></div></section>
      <section class="session-section"><h2>Guidance</h2><div class="card supervision"><div class="supervision-state"><span>Control</span><b id="autonomyState">Manual</b></div><div class="grid two"><button id="manualMode" class="autonomy active" data-shortcut="⇧G" onclick="setAutonomy('manual')">Manual <kbd>⇧G</kbd></button><button id="guidedMode" class="autonomy" data-shortcut="G" onclick="setAutonomy('guided')">Guided <kbd>G</kbd></button></div><div id="coachingCue" class="cue">You command every movement. Dr.Anmar records telemetry for coaching.</div></div></section>
      <section class="session-section"><h2>Live signals</h2><div class="card"><div class="safety-grid"><div class="safety-metric"><b id="forceMetric">—</b><span>CONTACT N</span></div><div class="safety-metric"><b id="deformMetric">—</b><span>TISSUE MM</span></div><div class="safety-metric"><b id="stressMetric">—</b><span>STRESS PA</span></div></div></div></section>
      <section class="session-section"><h2>Session</h2><div class="card"><div class="grid two"><button data-shortcut="T" onclick="recording(false)">Stop & save <kbd>T</kbd></button><button data-shortcut="R" onclick="replay()">Replay last <kbd>R</kbd></button><button data-shortcut="Delete" onclick="resetScene()">Reset scene <kbd>Delete</kbd></button></div><div class="hint" id="lastDemo">Robot state and camera observations are saved together.</div></div></section>
    </div></details>
    <section id="skinAdhesiveCell" class="stapler-cell hidden" aria-label="Topical skin adhesive controls">
      <div class="stapler-cell-head"><div><b>Dr.Anmar topical skin adhesive tool</b><small>Instrument 1 fixed-joint end effector · native IK · proportional dispense</small></div><span id="skinAdhesivePhase" class="stapler-phase">MOUNTED</span></div>
      <div class="stapler-metrics"><div class="stapler-metric"><b id="skinAdhesiveActivation">0%</b><span>ACTUAL DISPENSE</span></div><div class="stapler-metric"><b id="skinAdhesiveLeft">0.0°</b><span>LEFT PADDLE</span></div><div class="stapler-metric"><b id="skinAdhesiveRight">0.0°</b><span>RIGHT PADDLE</span></div><div class="stapler-metric"><b id="skinAdhesivePiston">0.00 mm</b><span>PISTON TRAVEL</span></div><div class="stapler-metric"><b id="skinAdhesiveMount">INSTRUMENT 1</b><span>FIXED PHYSX MOUNT</span></div><div class="stapler-metric"><b id="skinAdhesiveOutlet">EXPOSED</b><span>DISPENSE OUTLET</span></div></div>
      <div class="stapler-controls"><label class="stapler-target"><b>Instrument 1 dispense command</b><output id="skinAdhesiveTargetOutput">0%</output><input id="skinAdhesiveTarget" type="range" min="0" max="100" step="1" value="0" onchange="setSkinAdhesiveActivation(this.value)" oninput="document.getElementById('skinAdhesiveTargetOutput').value=`${this.value}%`"></label><button data-shortcut="ADH-0" onclick="setSkinAdhesiveActivation(0)">Release<br>0%</button><button data-shortcut="ADH-50" onclick="setSkinAdhesiveActivation(50)">Half squeeze<br>50%</button><button class="primary" data-shortcut="ADH-100" onclick="setSkinAdhesiveActivation(100)">Full squeeze<br>100%</button></div>
      <p class="stapler-boundary">Only the physical end-effector mechanism is simulated here. No bead is fabricated: liquid flow, wetting, polymerization, tissue bonding, dose, bond strength and clinical performance remain outside this room.</p>
    </section>
  </aside>
</main>
<div id="keyboardHelp" class="keyboard-help hidden" role="dialog" aria-modal="true" aria-labelledby="keyboardHelpTitle"><div class="keyboard-help-panel"><div class="keyboard-help-head"><div><h1 id="keyboardHelpTitle">Two-hand surgical controls</h1><p>Each hand permanently owns one robot. Tap for a micro-movement, hold to accelerate smoothly, and release to stop. Escape or Backspace stops both.</p></div><button data-shortcut="?" onclick="toggleKeyboardHelp(false)">Close <kbd>?</kbd></button></div><div class="shortcut-columns">
  <div class="shortcut-group"><h3>LEFT ROBOT · LEFT HAND</h3><div class="shortcut-line"><kbd>W / S</kbd><span>Up / down</span></div><div class="shortcut-line"><kbd>A / D</kbd><span>Left / right</span></div><div class="shortcut-line"><kbd>Q / E</kbd><span>Toward / away</span></div><div class="shortcut-line"><kbd>Z / X</kbd><span>Direct tool roll</span></div><div class="shortcut-line"><kbd>Left Shift + QWEASD</kbd><span>Roll / pitch / yaw</span></div><div class="shortcut-line"><kbd>Left Option</kbd><span>Hold for fine control</span></div><div class="shortcut-line"><kbd>Space</kbd><span>Toggle left gripper</span></div></div>
  <div class="shortcut-group"><h3>RIGHT ROBOT · RIGHT HAND</h3><div class="shortcut-line"><kbd>I / K</kbd><span>Up / down</span></div><div class="shortcut-line"><kbd>J / L</kbd><span>Left / right</span></div><div class="shortcut-line"><kbd>U / O</kbd><span>Toward / away</span></div><div class="shortcut-line"><kbd>N / M</kbd><span>Direct tool roll</span></div><div class="shortcut-line"><kbd>Right Shift + UIOJKL</kbd><span>Roll / pitch / yaw</span></div><div class="shortcut-line"><kbd>Right Option</kbd><span>Hold for fine control</span></div><div class="shortcut-line"><kbd>Enter</kbd><span>Toggle right gripper</span></div></div>
  <div class="shortcut-group"><h3>CONTROL FEEL + SAFETY</h3><div class="shortcut-line"><kbd>Tap movement</kbd><span>Micro-nudge</span></div><div class="shortcut-line"><kbd>Hold movement</kbd><span>Smooth acceleration</span></div><div class="shortcut-line"><kbd>1 / 2 / 3</kbd><span>Left fine / normal / fast</span></div><div class="shortcut-line"><kbd>8 / 9 / 0</kbd><span>Right fine / normal / fast</span></div><div class="shortcut-line"><kbd>Backspace / Esc</kbd><span>Stop both robots</span></div><div class="shortcut-line"><kbd>[ / ]</kbd><span>Select pointer-control robot</span></div></div>
  <div class="shortcut-group"><h3>SURGICAL COMBINATIONS</h3><div class="shortcut-line"><kbd>Q/E + Z/X</kbd><span>Left advance/retract + roll</span></div><div class="shortcut-line"><kbd>U/O + N/M</kbd><span>Right advance/retract + roll</span></div><div class="shortcut-line"><kbd>Both hand clusters</kbd><span>True simultaneous bimanual motion</span></div><div class="shortcut-line"><kbd>F12</kbd><span>Bounded context assist</span></div><div class="shortcut-line"><kbd>Space / Enter</kbd><span>Independent grippers</span></div></div>
  <div class="shortcut-group"><h3>CAMERAS</h3><div class="shortcut-line"><kbd>4 / 5</kbd><span>Stereo left / right</span></div><div class="shortcut-line"><kbd>6 / 7</kbd><span>Wrist 1 / 2</span></div><div class="shortcut-line"><kbd>F1 / F2 / F3</kbd><span>Operative / close / wide</span></div><div class="shortcut-line"><kbd>F4–F7</kbd><span>Overhead / oblique / opposite</span></div><div class="shortcut-line"><kbd>F8 / Home</kbd><span>Free camera / reset</span></div><div class="shortcut-line"><kbd>Drag / ⇧Drag / Wheel</kbd><span>Orbit / pan / zoom</span></div><div class="shortcut-line"><kbd>C / ⇧C</kbd><span>Next sensor / next angle</span></div></div>
  <div class="shortcut-group"><h3>EXPERT + SESSION</h3><div class="shortcut-line"><kbd>F9 / F10</kbd><span>Run / pause expert</span></div><div class="shortcut-line"><kbd>F11</kbd><span>Modeled abdominal access</span></div><div class="shortcut-line"><kbd>Y / T</kbd><span>Start / stop + save</span></div><div class="shortcut-line"><kbd>R / H</kbd><span>Replay / path guide</span></div><div class="shortcut-line"><kbd>G / Shift+G</kbd><span>Guided / manual</span></div><div class="shortcut-line"><kbd>Delete</kbd><span>Reset scene</span></div></div>
  <div class="shortcut-group"><h3>XBOX · BIMANUAL</h3><div class="shortcut-line"><kbd>Left / right stick</kbd><span>Move left / right robot</span></div><div class="shortcut-line"><kbd>Hold X + sticks</kbd><span>Depth + roll both wrists</span></div><div class="shortcut-line"><kbd>Hold Y + sticks</kbd><span>Pitch + yaw both wrists</span></div><div class="shortcut-line"><kbd>LB / LT</kbd><span>Close / open left gripper</span></div><div class="shortcut-line"><kbd>RB / RT</kbd><span>Close / open right gripper</span></div><div class="shortcut-line"><kbd>L3 / R3</kbd><span>Precision for that robot</span></div><div class="shortcut-line"><kbd>A / B</kbd><span>Smart assist / emergency stop</span></div></div>
  <div class="shortcut-group"><h3>XBOX · CAMERA + SESSION</h3><div class="shortcut-line"><kbd>Hold View</kbd><span>Camera control layer</span></div><div class="shortcut-line"><kbd>Camera: sticks</kbd><span>Pan / orbit</span></div><div class="shortcut-line"><kbd>Camera: LT / RT</kbd><span>Zoom out / in</span></div><div class="shortcut-line"><kbd>Camera: LB / RB</kbd><span>Sensor / angle</span></div><div class="shortcut-line"><kbd>D-pad ↑ / ↓</kbd><span>Faster / slower tools</span></div><div class="shortcut-line"><kbd>Hold Menu</kbd><span>Session layer</span></div><div class="shortcut-line"><kbd>Menu + A / X / Y</kbd><span>Record / expert / guidance</span></div><div class="shortcut-line"><kbd>Menu + ↑ / ↓</kbd><span>Replay / reset scene</span></div></div>
  <div class="shortcut-group"><h3>VOICE CONTROL</h3><div class="shortcut-line"><kbd>Hold `</kbd><span>Push to talk</span></div><div class="shortcut-line"><kbd>Say: left robot up</kbd><span>Bounded robot nudge</span></div><div class="shortcut-line"><kbd>Say: right robot toward</kbd><span>Choose robot + direction</span></div><div class="shortcut-line"><kbd>Say: close left gripper</kbd><span>Explicit jaw command</span></div><div class="shortcut-line"><kbd>Say: camera overhead</kbd><span>Switch view</span></div><div class="shortcut-line"><kbd>Say: stop</kbd><span>Stop both robots</span></div><div class="shortcut-line"><kbd>Type + ↵</kbd><span>Works without microphone support</span></div></div>
</div></div></div><div id="toast"></div>
<script>
const operatorId=(()=>{const query=new URLSearchParams(location.search).get('operator');if(query){sessionStorage.setItem('drAnmarOperatorId',query);return query}let value=sessionStorage.getItem('drAnmarOperatorId');if(!value){const random=crypto.randomUUID?crypto.randomUUID():`${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;value=`browser-${random}`;sessionStorage.setItem('drAnmarOperatorId',value)}return value})();
const handKeyMaps=[{KeyW:[2,1,'Up'],KeyS:[2,-1,'Down'],KeyA:[1,1,'Left'],KeyD:[1,-1,'Right'],KeyQ:[0,-1,'Toward'],KeyE:[0,1,'Away']},{KeyI:[2,1,'Up'],KeyK:[2,-1,'Down'],KeyJ:[1,1,'Left'],KeyL:[1,-1,'Right'],KeyU:[0,-1,'Toward'],KeyO:[0,1,'Away']}];
const rotationKeyMaps=[{KeyW:[4,-1,'Pitch up'],KeyS:[4,1,'Pitch down'],KeyA:[5,-1,'Yaw left'],KeyD:[5,1,'Yaw right'],KeyQ:[3,-1,'Roll left'],KeyE:[3,1,'Roll right']},{KeyI:[4,-1,'Pitch up'],KeyK:[4,1,'Pitch down'],KeyJ:[5,-1,'Yaw left'],KeyL:[5,1,'Yaw right'],KeyU:[3,-1,'Roll left'],KeyO:[3,1,'Roll right']}];
const directRollKeyMaps=[{KeyZ:[3,-1,'Roll left'],KeyX:[3,1,'Roll right']},{KeyN:[3,-1,'Roll left'],KeyM:[3,1,'Roll right']}];
const dualMovementCodes=new Set([...Object.keys(handKeyMaps[0]),...Object.keys(handKeyMaps[1]),...Object.keys(directRollKeyMaps[0]),...Object.keys(directRollKeyMaps[1])]);
let activeArm=0,driveSpeed=1,keyboardSpeeds=[1,1],driveInFlight=false,queuedDrive=null,driveWasActive=false,bimanualInFlight=false,queuedBimanual=null,bimanualWasActive=false,inputSource='keyboard_pointer',lastGazeSend=0,currentCamera='endoscope_left',currentViewMode='free',latestStatus=null,workerInstanceId=null,macroPulseTimer=null,voicePulseTimer=null,keyFlashTimer=null,toastTimer=null,cameraAdjustMode=true,cameraDrag=null,cameraAdjustPending={},cameraAdjustPendingCamera=null,cameraAdjustTimer=null,cameraAdjustInFlight=false,cameraFeedGeneration=0,cameraFeedController=null,cameraObjectUrl=null,refreshInFlight=false,heartbeatInFlight=false,pageDisposed=false,parentKeyboardActive=false,voiceRecognition=null,voiceListening=false,gamepadSafetyLatched=false,gamepadSpeed=1,gamepadFocusArm=0,gamepadAnimationFrame=null,gamepadCameraRequested=false;
const heldKeys=new Set(),heldModifiers=new Set(),heldKeyStartedAt=new Map(),pointerMoves=new Map();
const activeFetchControllers=new Set();
const gamepadButtonStates=new Map();
const gamepadKnownIndices=new Set();
let latestGamepadCommands=new Map(),gamepadVisualState={mode:'BIMANUAL · NORMAL',left:[0,0],right:[0,0]};
const previousGamepadContacts=[false,false];
async function requestJson(url,options={},timeoutMs=5000){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeoutMs);activeFetchControllers.add(controller);try{const r=await fetch(url,{...options,signal:controller.signal});let data={};try{data=await r.json()}catch(_error){}if(!r.ok)throw Error(data.detail||'Request failed');return data}catch(error){if(error.name==='AbortError')throw Error('Simulator request timed out');throw error}finally{clearTimeout(timer);activeFetchControllers.delete(controller)}}
async function post(url,body={},timeoutMs=5000){return requestJson(url,{method:'POST',headers:{'content-type':'application/json','x-dr-anmar-operator':operatorId},body:JSON.stringify(body)},timeoutMs)}
async function setSkinAdhesiveActivation(percent){const activation=Math.max(0,Math.min(1,(Number(percent)||0)/100));try{await post('/api/skin-adhesive/activation',{activation});toast(`Skin adhesive activation ${Math.round(activation*100)}%`);await refresh()}catch(e){toast(e.message)}}
function renderSkinAdhesive(system={}){const panel=document.getElementById('skinAdhesiveCell'),enabled=!!system.enabled;panel.classList.toggle('hidden',!enabled);if(!enabled)return;const actual=Number(system.actual_activation||0),target=Number(system.target_activation||0),slider=document.getElementById('skinAdhesiveTarget'),arm=Number(system.mounted_arm||1);document.getElementById('skinAdhesivePhase').textContent=String(system.workflow_state||system.applicator_state||'mounted').replaceAll('_',' ');document.getElementById('skinAdhesiveActivation').textContent=`${Math.round(actual*100)}%`;document.getElementById('skinAdhesiveLeft').textContent=`${Number(system.left_paddle_deg||0).toFixed(1)}°`;document.getElementById('skinAdhesiveRight').textContent=`${Number(system.right_paddle_deg||0).toFixed(1)}°`;document.getElementById('skinAdhesivePiston').textContent=`${Number(system.piston_travel_mm||0).toFixed(2)} mm`;document.getElementById('skinAdhesiveMount').textContent=`INSTRUMENT ${arm}`;document.getElementById('skinAdhesiveOutlet').textContent=String(system.outlet_state||'exposed').toUpperCase();const gripButton=document.getElementById(arm===1?'gripOpenButton':'gripCloseButton');if(gripButton)gripButton.innerHTML=`<kbd>${arm===1?'Space':'Enter'}</kbd> Dispense`;if(document.activeElement!==slider)slider.value=String(Math.round(target*100));document.getElementById('skinAdhesiveTargetOutput').value=`${Math.round(target*100)}%`}
async function closureRobotCommand(action){try{await post('/api/closure-robot/command',{action});toast({run:'Physical closure started',stop:'Closure mechanism held',reset:'Closure robot reset'}[action]||action);await refresh()}catch(e){toast(e.message)}}
function renderClosureRobot(system={}){const panel=document.getElementById('closureRobotCell'),enabled=!!system.enabled;panel.classList.toggle('hidden',!enabled);if(!enabled)return;const phase=String(system.phase||'ready').replaceAll('_',' '),running=!!system.cycle_running,complete=!!system.cycle_complete;document.getElementById('closureRobotPhase').textContent=system.last_error?'safety hold':complete?'complete':system.held?`held · ${phase}`:phase;document.getElementById('closureRobotApproximation').textContent=`${Number(system.left_approximation_mm||0).toFixed(1)} / ${Number(system.right_approximation_mm||0).toFixed(1)} mm`;document.getElementById('closureRobotClamps').textContent=`${Number(system.left_clamp_deg||0).toFixed(1)} / ${Number(system.right_clamp_deg||0).toFixed(1)}°`;document.getElementById('closureRobotDriver').textContent=`${Number(system.staple_driver_mm||0).toFixed(1)} mm`;document.getElementById('closureRobotStaple').textContent=`${Number(system.formed_staple_count||0)} · ${Number(system.staple_attachment_count||0)} bonds`;document.getElementById('closureRobotAdhesive').textContent=`${Number(system.adhesive_deploy_mm||0).toFixed(1)} / ${Number(system.adhesive_meter_mm||0).toFixed(1)} mm`;document.getElementById('closureRobotBonds').textContent=`${Number(system.adhesive_bead_count||0)} · ${Number(system.adhesive_bond_attachment_count||0)} bonds`;document.getElementById('closureRobotCapture').textContent=String(Number(system.capture_attachment_count||0));document.getElementById('closureRobotPhysics').textContent=String(system.tissue_backend||'physx').replace('physx_','').toUpperCase();const run=document.getElementById('closureRobotRun'),stop=document.getElementById('closureRobotStop');run.disabled=running;stop.disabled=!running;run.innerHTML=complete?'Run a new<br>physical closure':'Run complete<br>physical closure';if(system.last_error)run.title=system.last_error;else run.removeAttribute('title')}
function toast(s){const e=document.getElementById('toast');e.textContent=s;e.classList.add('show');if(toastTimer)clearTimeout(toastTimer);toastTimer=setTimeout(()=>{toastTimer=null;e.classList.remove('show')},1600)}
async function staplerCommand(action,targetDeg=null){try{const body={action};if(targetDeg!==null)body.target_deg=Number(targetDeg);await post('/api/stapler/command',body);const message={fire:'Stapler cycle started',reset:'Tissue closure reset',release:'Stapler released',previous_station:'Fixture indexed to previous station',next_station:'Fixture indexed to next station'}[action]||`Trigger target ${Number(targetDeg).toFixed(0)}°`;toast(message);await refresh()}catch(e){toast(e.message)}}
function setStaplerTarget(value){const target=Math.max(0,Math.min(28,Number(value)||0));return staplerCommand('set_target',target)}
function runStaplerCycle(){return staplerCommand('fire')}
function renderStaplerCell(cell={}){const panel=document.getElementById('staplerCell'),enabled=!!cell.enabled;panel.classList.toggle('hidden',!enabled);if(!enabled)return;const station=Number(cell.station_index||1),stationCount=Number(cell.station_count||7),placed=Number(cell.closed_station_count||0),running=!!cell.cycle_running,stationPlaced=cell.station_state==='placed',stationReady=cell.station_ready!==false,gap=cell.tissue_gap_mm;document.getElementById('staplerPhase').textContent=cell.closure_complete?'closure complete':!stationReady?'indexing':String(cell.cycle_phase||'ready').replaceAll('_',' ');document.getElementById('staplerStation').textContent=`${station} / ${stationCount}`;document.getElementById('staplerClosure').textContent=`${placed} / ${stationCount}`;document.getElementById('staplerGap').textContent=gap===null||gap===undefined?'—':`${Number(gap).toFixed(2)} mm`;document.getElementById('staplerApproximation').textContent=`${Number(cell.approximation_progress_percent||0).toFixed(0)}%`;document.getElementById('staplerRetention').textContent=String(cell.retention_state||'open').replaceAll('_',' ').toUpperCase();document.getElementById('staplerSpacing').textContent=`${Number(cell.station_spacing_mm||6).toFixed(1)} mm`;document.getElementById('staplerTrigger').textContent=`${Number(cell.actual_trigger_deg||0).toFixed(1)}°`;document.getElementById('staplerPusher').textContent=`${Number(cell.pusher_travel_mm||0).toFixed(2)} mm`;document.getElementById('staplerMagazine').textContent=`${cell.magazine_remaining??0} / ${cell.magazine_capacity??35}`;document.getElementById('staplerProgress').style.width=`${Number(cell.closure_progress_percent||0)}%`;document.getElementById('staplerPrevious').disabled=running||!stationReady||station<=1;document.getElementById('staplerNext').disabled=running||!stationReady||station>=stationCount;const fire=document.getElementById('staplerFire');fire.disabled=running||!stationReady||stationPlaced||!!cell.closure_complete;fire.innerHTML=cell.closure_complete?'Closure complete<br>reset to repeat':!stationReady?'Indexing fixture<br>hold position':stationPlaced?'Staple retaining tissue<br>choose next':'Approximate, staple &amp; retain<br>one physical cycle'}
function showKeyAction(key,label,active=true){const display=document.getElementById('keyActionDisplay');display.classList.toggle('active',active);display.querySelector('kbd').textContent=key;display.querySelector('span').textContent=label}
function flashShortcut(shortcut,label,duration=850){if(keyFlashTimer)clearTimeout(keyFlashTimer);document.querySelectorAll('button.key-active').forEach(button=>button.classList.remove('key-active'));document.querySelectorAll('button[data-shortcut]').forEach(button=>{if(button.dataset.shortcut===shortcut)button.classList.add('key-active')});showKeyAction(shortcut,label,true);keyFlashTimer=setTimeout(()=>{document.querySelectorAll('button.key-active').forEach(button=>button.classList.remove('key-active'));showKeyAction('READY','Keyboard control ready',false)},duration)}
function runShortcut(shortcut,label,action){flashShortcut(shortcut,label);action()}
function setArm(arm){const arms=latestStatus?.arms||1;if(arm>=arms){toast(`Instrument ${arm+1} is not available in this room`);return}stopDrive(false);activeArm=arm;document.getElementById('arm0').classList.toggle('active',arm===0);document.getElementById('arm1').classList.toggle('active',arm===1);toast(`Instrument ${arm+1} active`)}
function setSpeed(speed,button){driveSpeed=speed;document.querySelectorAll('[data-speed]').forEach(x=>x.classList.toggle('active',x===button));updateControlReadout(false,`${button?.textContent.trim()||'Selected'} speed`)}
function setSpeedShortcut(speed){setSpeed(speed,document.querySelector(`[data-speed="${speed}"]`))}
function setHandSpeed(arm,speed,label){keyboardSpeeds[arm]=speed;document.querySelectorAll(`[data-hand-speed-arm="${arm}"]`).forEach(button=>button.classList.toggle('active',Number(button.dataset.handSpeed)===speed));flashShortcut(label,`Instrument ${arm+1} · ${speed===.35?'precision':speed===1?'normal':'fast'} speed`);toast(`Instrument ${arm+1} speed · ${speed===.35?'precision':speed===1?'normal':'fast'}`)}
function deadzone(value){return Math.abs(value)<0.18?0:Math.sign(value)*(Math.abs(value)-0.18)/0.82}
function standardGamepads(){return (navigator.getGamepads?[...navigator.getGamepads()]:[]).filter(Boolean)}
function radialStick(pad,xIndex,yIndex){const rawX=pad.axes[xIndex]||0,rawY=pad.axes[yIndex]||0,magnitude=Math.min(1,Math.hypot(rawX,rawY)),dead=.14;if(magnitude<=dead)return[0,0];const normalized=(magnitude-dead)/(1-dead),curved=.58*normalized+.42*normalized**3,scale=curved/magnitude;return[rawX*scale,rawY*scale]}
function gamepadButtonEdges(pad){const prefix=`${pad.index}:`,known=[...gamepadButtonStates.keys()].some(key=>key.startsWith(prefix));return pad.buttons.map((button,index)=>{const key=`${prefix}${index}`,pressed=!!button?.pressed,previous=gamepadButtonStates.get(key)||false;gamepadButtonStates.set(key,pressed);return known&&pressed&&!previous})}
function gamepadHaptic(pad,{duration=70,weak=.22,strong=.12}={}){const actuator=pad?.vibrationActuator||pad?.hapticActuators?.[0];if(!actuator?.playEffect)return;const effects=actuator.effects||[];if(effects.length&&!effects.includes('dual-rumble'))return;actuator.playEffect('dual-rumble',{startDelay:0,duration,weakMagnitude:Math.max(0,Math.min(1,weak)),strongMagnitude:Math.max(0,Math.min(1,strong))}).catch(()=>{})}
function setStickIndicator(id,stick=[0,0]){const dot=document.getElementById(id);if(dot)dot.style.transform=`translate(${(stick[0]*6).toFixed(1)}px,${(stick[1]*6).toFixed(1)}px)`}
function updateGamepadStatus(pads=standardGamepads(),state=gamepadVisualState){const button=document.getElementById('gamepadStatus'),title=document.getElementById('gamepadTitle'),mode=document.getElementById('gamepadMode');if(!button||!title||!mode)return;const connected=pads.length>0;title.textContent=connected?(pads.length>1?'Xbox · both robots · extra pad standby':'Xbox · both robots'):'Connect Xbox controller';mode.textContent=connected?state.mode:'One pad · both robots';button.classList.toggle('connected',connected);button.classList.toggle('mode',connected&&!state.mode.startsWith('BIMANUAL'));setStickIndicator('gamepadLeftStick',connected?state.left:[0,0]);setStickIndicator('gamepadRightStick',connected?state.right:[0,0]);button.setAttribute('aria-label',connected?`Xbox connected. ${state.mode}. Open controller map`:'Connect an Xbox controller. Open controller map')}
function setGamepadSpeed(direction,pad){const steps=[.35,1,1.7],index=Math.max(0,steps.indexOf(gamepadSpeed)),next=steps[Math.max(0,Math.min(steps.length-1,index+direction))];if(next===gamepadSpeed)return;gamepadSpeed=next;gamepadHaptic(pad,{duration:55,weak:.18,strong:.08});toast(`Controller speed · ${next===.35?'precision':next===1?'normal':'fast'}`)}
function gamepadExpertAction(){const status=latestStatus?.expert_demonstration?.status;if(status==='running'||status==='paused')toggleExpertPause();else startExpert()}
async function enableGamepadCamera(){if(cameraAdjustMode||gamepadCameraRequested)return;gamepadCameraRequested=true;try{const cameraName=ensureAdjustableCameraSensor();const result=await post('/api/camera-adjust',{camera_name:cameraName,enabled:true});currentViewMode=result.mode;renderFreeCamera(result);document.querySelectorAll('[data-view-mode]').forEach(x=>x.classList.remove('active'))}catch(e){toast(e.message)}finally{gamepadCameraRequested=false}}
function readGamepadCommands(){const pads=standardGamepads(),commands=new Map(),arms=Math.min(latestStatus?.arms||1,2),livePrefixes=new Set(pads.map(pad=>`${pad.index}:`));for(const key of gamepadButtonStates.keys())if(![...livePrefixes].some(prefix=>key.startsWith(prefix)))gamepadButtonStates.delete(key);if(!pads.length){gamepadKnownIndices.clear();gamepadSafetyLatched=false;gamepadVisualState={mode:'BIMANUAL · NORMAL',left:[0,0],right:[0,0]};updateGamepadStatus(pads);return commands}const pad=pads[0];if(!gamepadKnownIndices.has(pad.index)){gamepadKnownIndices.add(pad.index);gamepadSafetyLatched=true}const edges=gamepadButtonEdges(pad),left=radialStick(pad,0,1),right=radialStick(pad,2,3),systemLayer=!!pad.buttons[9]?.pressed,cameraLayer=!systemLayer&&!!pad.buttons[8]?.pressed,wristLayer=!systemLayer&&!cameraLayer&&!!pad.buttons[3]?.pressed,depthLayer=!systemLayer&&!cameraLayer&&!wristLayer&&!!pad.buttons[2]?.pressed;let mode='BIMANUAL',anyMotion=false;
  if(edges[1]){gamepadSafetyLatched=true;gamepadHaptic(pad,{duration:240,weak:.8,strong:1});emergencyStop('gamepad')}
  if(systemLayer){mode='SESSION';if(edges[9])gamepadHaptic(pad,{duration:55,weak:.22,strong:.1});if(edges[0])recording(!latestStatus?.recording);if(edges[2])gamepadExpertAction();if(edges[3])setAutonomy(latestStatus?.autonomy_mode==='guided'?'manual':'guided');if(edges[4])cycleSensorCamera();if(edges[5])cycleCameraView();if(edges[8])resetFreeCamera();if(edges[10])toggleReferenceGhost();if(edges[11])takeControl();if(edges[12])replay();if(edges[13])resetScene()}
  else if(cameraLayer){mode='CAMERA';if(edges[8]){enableGamepadCamera();gamepadHaptic(pad,{duration:65,weak:.28,strong:.12})}if(edges[4])cycleSensorCamera();if(edges[5])cycleCameraView();if(edges[3]||edges[11])resetFreeCamera();const zoom=(pad.buttons[7]?.value||0)-(pad.buttons[6]?.value||0);if(Math.abs(left[0])>.01||Math.abs(left[1])>.01||Math.abs(right[0])>.01||Math.abs(right[1])>.01||Math.abs(zoom)>.04)queueCameraAdjustment({pan_x_delta_m:-left[0]*.0011,pan_y_delta_m:left[1]*.0011,orbit_yaw_delta_deg:right[0]*1.25,orbit_pitch_delta_deg:right[1]*1.1,zoom_delta:-zoom*.035})}
  else {if(depthLayer&&edges[2])gamepadHaptic(pad,{duration:48,weak:.12,strong:.2});if(wristLayer&&edges[3])gamepadHaptic(pad,{duration:48,weak:.2,strong:.12});if(edges[0]){activeArm=Math.min(gamepadFocusArm,arms-1);smartAction('gamepad_smart_action');gamepadHaptic(pad,{duration:75,weak:.25,strong:.12})}if(edges[4]){grip(false,0,'gamepad');gamepadHaptic(pad,{duration:60,weak:.34,strong:.08})}if(edges[6]){grip(true,0,'gamepad');gamepadHaptic(pad,{duration:45,weak:.16,strong:.06})}if(arms>1&&edges[5]){grip(false,1,'gamepad');gamepadHaptic(pad,{duration:60,weak:.08,strong:.34})}if(arms>1&&edges[7]){grip(true,1,'gamepad');gamepadHaptic(pad,{duration:45,weak:.06,strong:.16})}if(edges[12])setGamepadSpeed(1,pad);if(edges[13])setGamepadSpeed(-1,pad);if(edges[14])cycleSensorCamera();if(edges[15])cycleCameraView();mode=wristLayer?'WRIST PITCH + YAW':depthLayer?'DEPTH + ROLL':'BIMANUAL';const sticks=[left,right];for(let arm=0;arm<arms;arm++){const [x,y]=sticks[arm],values=Array(6).fill(0);if(wristLayer){values[5]+=x;values[4]+=y}else if(depthLayer){values[3]+=x;values[0]+=y}else{values[1]-=x;values[2]-=y}const normalized=normalizeDrive(values),precision=!!pad.buttons[10+arm]?.pressed,speed=precision?.35:gamepadSpeed;if(Math.hypot(x,y)>.16)gamepadFocusArm=arm;anyMotion=anyMotion||normalized.some(value=>Math.abs(value)>.01);commands.set(arm,{values:normalized,labels:[wristLayer?'Xbox wrist':depthLayer?'Xbox depth + roll':'Xbox camera-plane'],speed});if(precision)mode+=arm===0?' · LEFT PRECISION':' · RIGHT PRECISION'}}
  if(gamepadSafetyLatched){if(!anyMotion&&!pad.buttons[1]?.pressed)gamepadSafetyLatched=false;else{commands.clear();mode='SAFETY STOP · CENTER STICKS'}}gamepadVisualState={mode:`${mode} · ${gamepadSpeed===.35?'FINE':gamepadSpeed===1?'NORMAL':'FAST'}`,left,right};updateGamepadStatus(pads,gamepadVisualState);return commands}
function pollGamepads(){if(pageDisposed)return;if(!document.hidden)latestGamepadCommands=readGamepadCommands();else latestGamepadCommands=new Map();gamepadAnimationFrame=requestAnimationFrame(pollGamepads)}
function normalizeDrive(values){for(const [start,end] of [[0,3],[3,6]]){const norm=Math.hypot(...values.slice(start,end));if(norm>1)for(let i=start;i<end;i++)values[i]/=norm}return values.map(value=>Math.max(-1,Math.min(1,value)))}
function buildDrive(){const values=Array(6).fill(0);pointerMoves.forEach(move=>{if(move.values)move.values.forEach((value,index)=>values[index]+=value);else values[move.axis]+=move.direction});return normalizeDrive(values)}
function keyboardInputGain(code){const started=heldKeyStartedAt.get(code);if(started===undefined)return .42;const progress=Math.max(0,Math.min(1,(performance.now()-started)/520)),smooth=progress*progress*(3-2*progress);return .42+.58*smooth}
function keyboardArmDrive(arm){const shiftRotate=heldModifiers.has(arm===0?'rotate-left':'rotate-right'),primaryMap=shiftRotate?rotationKeyMaps[arm]:handKeyMaps[arm],rollMap=directRollKeyMaps[arm],values=Array(6).fill(0),labels=[];let translationActive=false,rotationActive=false;heldKeys.forEach(code=>{const primaryMove=primaryMap[code],rollMove=rollMap[code],move=primaryMove||rollMove;if(!move)return;values[move[0]]+=move[1]*keyboardInputGain(code);labels.push(move[2]);translationActive=translationActive||move[0]<3;rotationActive=rotationActive||move[0]>=3});const mode=translationActive&&rotationActive?'combined':rotationActive?'wrist':'tool';return {values:normalizeDrive(values),labels,mode}}
function keyboardSpeedForArm(arm){const fine=heldModifiers.has(arm===0?'precision-left':'precision-right');return fine?Math.min(keyboardSpeeds[arm],.35):keyboardSpeeds[arm]}
function buildBimanualCommands(gamepadCommands=new Map()){const arms=Math.min(latestStatus?.arms||1,2),commands=[];for(let arm=0;arm<arms;arm++){const hand=keyboardArmDrive(arm),pad=gamepadCommands.get(arm),values=hand.values.slice(),labels=hand.labels.slice(),handActive=hand.values.some(value=>Math.abs(value)>.01),padActive=pad?.values.some(value=>Math.abs(value)>.01),keyboardSpeed=keyboardSpeedForArm(arm);if(pad){pad.values.forEach((value,index)=>values[index]+=value);labels.push(...pad.labels)}commands.push({arm,values:normalizeDrive(values),speed:handActive&&padActive?Math.min(keyboardSpeed,pad.speed):padActive?pad.speed:keyboardSpeed,labels,mode:hand.mode})}return commands}
function effectiveSpeed(){const fine=heldModifiers.has(activeArm===0?'precision-left':'precision-right');return fine?Math.min(driveSpeed,.35):driveSpeed}
function activeDriveLabel(commands=buildBimanualCommands()){const labels=commands.filter(x=>x.values.some(v=>Math.abs(v)>.01)).map(x=>`R${x.arm+1} ${x.mode}: ${x.labels.join(' + ')}`);return labels.join(' · ')||'Moving'}
function updateControlReadout(moving,label){const readout=document.getElementById('controlReadout');readout.classList.toggle('moving',moving);readout.querySelector('span').textContent=moving?(label||'Moving · release to stop'):'Ready · hold a control to move'}
async function flushDrive(){if(pageDisposed){queuedDrive=null;return}if(driveInFlight||!queuedDrive)return;const next=queuedDrive;queuedDrive=null;driveInFlight=true;try{await post('/api/drive',{values:next.values,arm:activeArm,speed:next.speed,source:next.source})}catch(e){if(!pageDisposed)toast(e.message)}finally{driveInFlight=false;if(!pageDisposed&&queuedDrive)flushDrive()}}
function sendDrive(values,speed=effectiveSpeed(),source=inputSource){if(pageDisposed)return;queuedDrive={values,speed,source};flushDrive()}
async function flushBimanual(){if(pageDisposed){queuedBimanual=null;return}if(bimanualInFlight||!queuedBimanual)return;const next=queuedBimanual;queuedBimanual=null;bimanualInFlight=true;try{await post('/api/drive/bimanual',{commands:next.commands,source:next.source})}catch(e){if(!pageDisposed)toast(e.message)}finally{bimanualInFlight=false;if(!pageDisposed&&queuedBimanual)flushBimanual()}}
function sendBimanual(commands,source='keyboard_pointer'){if(pageDisposed)return;queuedBimanual={commands:commands.map(({arm,values,speed})=>({arm,values,speed})),source};flushBimanual()}
function syncKeyVisuals(){document.querySelectorAll('[data-key]').forEach(button=>button.classList.toggle('held',heldKeys.has(button.dataset.key)||[...pointerMoves.values()].some(move=>move.button===button)));document.getElementById('leftRotateModifier').classList.toggle('active',heldModifiers.has('rotate-left'));document.getElementById('rightRotateModifier').classList.toggle('active',heldModifiers.has('rotate-right'));document.getElementById('leftPrecisionModifier').classList.toggle('active',heldModifiers.has('precision-left'));document.getElementById('rightPrecisionModifier').classList.toggle('active',heldModifiers.has('precision-right'))}
function updateDrive(){if(pageDisposed||document.hidden)return;const gamepadCommands=latestGamepadCommands,commands=buildBimanualCommands(gamepadCommands),bimanualActive=commands.some(command=>command.values.some(value=>Math.abs(value)>.01)),gamepadActive=[...gamepadCommands.values()].some(command=>command.values.some(value=>Math.abs(value)>.01)),pointerValues=buildDrive(),pointerActive=pointerValues.some(value=>Math.abs(value)>.01);if((bimanualActive||pointerActive)&&macroPulseTimer){clearTimeout(macroPulseTimer);macroPulseTimer=null}if(bimanualActive||bimanualWasActive)sendBimanual(commands,gamepadActive?'gamepad':'keyboard_pointer');bimanualWasActive=bimanualActive;if(!bimanualActive&&(pointerActive||driveWasActive))sendDrive(pointerValues);driveWasActive=pointerActive;syncKeyVisuals();updateControlReadout(bimanualActive||pointerActive,activeDriveLabel(commands))}
function clearHeldControls(){heldKeys.clear();heldModifiers.clear();heldKeyStartedAt.clear();pointerMoves.clear();syncKeyVisuals()}
function stopDrive(showToast=true,source='keyboard_pointer'){if(macroPulseTimer){clearTimeout(macroPulseTimer);macroPulseTimer=null}if(voicePulseTimer){clearTimeout(voicePulseTimer);voicePulseTimer=null}clearHeldControls();driveWasActive=false;bimanualWasActive=false;sendDrive(Array(6).fill(0),effectiveSpeed(),source);sendBimanual(buildBimanualCommands(),source);updateControlReadout(false);if(showToast)toast('Both instruments stopped')}
async function stopTool(){stopDrive();try{await post('/api/stop')}catch(e){toast(e.message)}}
async function emergencyStop(source='keyboard_pointer'){gamepadSafetyLatched=true;flashShortcut('Esc','Emergency stop · manual control');stopDrive(false,source);try{await post('/api/stop',{source});if(latestStatus?.autonomy_mode&&latestStatus.autonomy_mode!=='manual')await post('/api/handoff');toast('Stopped · manual control')}catch(e){toast(e.message)}}
async function grip(open,arm=activeArm,source='keyboard_pointer'){try{await post('/api/gripper',{open,arm,source});toast(`Instrument ${arm+1} · ${open?'gripper open':'gripper closed'}`)}catch(e){toast(e.message)}}
async function toggleGrip(arm=activeArm,source='keyboard_pointer'){if(arm>=(latestStatus?.arms||1)){toast(`Instrument ${arm+1} is not available in this room`);return}try{const result=await post('/api/gripper/toggle',{arm,source}),adhesiveArm=Number(latestStatus?.skin_adhesive_system?.mounted_arm||0)-1;toast(arm===adhesiveArm?`Adhesive tool · ${result.open?'dispense released':'full dispense'}`:`Instrument ${arm+1} · ${result.open?'gripper open':'gripper closed'}`)}catch(e){toast(e.message)}}
async function recording(start){try{await post(start?'/api/record/start':'/api/record/stop');toast(start?'Recording started':'Saving demonstration…')}catch(e){toast(e.message)}}
async function replay(){try{const x=await post('/api/replay-last');toast(x.message)}catch(e){toast(e.message)}}
async function referenceGhost(enabled){try{const x=await post('/api/reference-ghost',{enabled});toast(x.message)}catch(e){toast(e.message)}}
const cameraDelay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
for(const [name,label] of [['wrist_1','Gripper 1'],['wrist_2','Gripper 2']]){const button=document.querySelector(`[data-camera="${name}"]`);if(button?.firstChild)button.firstChild.textContent=`${label} `}
function startCameraFeed(name){currentCamera=name;cameraFeedGeneration+=1;const generation=cameraFeedGeneration,image=document.getElementById('cameraImage');cameraFeedController?.abort();cameraFeedController=new AbortController();const controller=cameraFeedController;activeFetchControllers.add(controller);(async()=>{try{while(!pageDisposed&&generation===cameraFeedGeneration){if(document.hidden){await cameraDelay(250);continue}try{const requestStarted=performance.now(),response=await fetch(`/frame/${encodeURIComponent(name)}.jpg?t=${Date.now()}`,{cache:'no-store',signal:controller.signal});if(!response.ok)throw Error(`Camera frame ${response.status}`);const nextUrl=URL.createObjectURL(await response.blob()),previousUrl=cameraObjectUrl;cameraObjectUrl=nextUrl;image.src=nextUrl;if(previousUrl)URL.revokeObjectURL(previousUrl);await cameraDelay(Math.max(0,55-(performance.now()-requestStarted)))}catch(error){if(controller.signal.aborted)return;await cameraDelay(250)}}}finally{activeFetchControllers.delete(controller)}})()}
function cameraAdjustmentTarget(name=currentCamera){if(name.startsWith('wrist_'))return name;if(name.startsWith('endoscope_'))return'endoscope_left';return'endoscope_left'}
function selectedCameraAdjustment(status=latestStatus){const target=cameraAdjustmentTarget();return status?.camera_adjustable_by_name?.[target]||status?.camera_adjustable||{}}
function setCamera(name,button){if(currentCamera!==name||!cameraFeedController||cameraFeedController.signal.aborted)startCameraFeed(name);document.querySelectorAll('[data-camera]').forEach(x=>x.classList.toggle('active',x===button));renderFreeCamera(selectedCameraAdjustment())}
function setCameraShortcut(name){const button=document.querySelector(`[data-camera="${name}"]`);if(!button||button.classList.contains('hidden')){toast(`${name.replace('_',' ')} is not available in this room`);return}setCamera(name,button);toast(`${button.textContent.trim()} camera`)}
async function setCameraView(mode,button){try{const result=await post('/api/camera-view',{mode});currentViewMode=result.mode;renderFreeCamera({enabled:false});document.querySelectorAll('[data-view-mode]').forEach(x=>x.classList.toggle('active',x.dataset.viewMode===result.mode));toast(`${button?.textContent||result.mode} camera ready`)}catch(e){toast(e.message)}}
function renderFreeCamera(adjustable={}){cameraAdjustMode=!!adjustable.enabled;const view=document.getElementById('cameraView');view.classList.toggle('free-camera',cameraAdjustMode);if(!cameraAdjustMode)view.classList.remove('dragging');const hud=document.getElementById('freeCameraHud');hud.textContent=currentCamera.startsWith('wrist_')?'Drag aim · Shift-drag mount · wheel zoom':'Drag orbit · Shift-drag pan · wheel zoom';hud.classList.toggle('hidden',!cameraAdjustMode);document.getElementById('freeCameraButton').classList.toggle('active',cameraAdjustMode);document.getElementById('resetCameraButton').classList.toggle('state-active',cameraAdjustMode)}
function ensureAdjustableCameraSensor(){if(currentCamera.startsWith('endoscope_')||currentCamera.startsWith('wrist_'))return cameraAdjustmentTarget();const button=document.querySelector('[data-camera="endoscope_left"]');setCamera('endoscope_left',button);return'endoscope_left'}
async function toggleFreeCamera(){try{const cameraName=ensureAdjustableCameraSensor(),enable=!cameraAdjustMode;const result=await post('/api/camera-adjust',{camera_name:cameraName,enabled:enable});if(cameraName.startsWith('endoscope_'))currentViewMode=result.mode;renderFreeCamera(result);document.querySelectorAll('[data-view-mode]').forEach(x=>x.classList.remove('active'));if(!result.enabled&&!cameraName.startsWith('wrist_'))document.querySelector(`[data-view-mode="${result.mode}"]`)?.classList.add('active');toast(result.enabled?(cameraName.startsWith('wrist_')?'Gripper camera adjustable · drag, Shift-drag, or scroll':'Free camera · drag, Shift-drag, or scroll'):'Fixed camera restored')}catch(e){toast(e.message)}}
async function resetFreeCamera(){try{const cameraName=ensureAdjustableCameraSensor();const result=await post('/api/camera-adjust',{camera_name:cameraName,enabled:true,reset:true});if(cameraName.startsWith('endoscope_'))currentViewMode='free';renderFreeCamera(result);document.querySelectorAll('[data-view-mode]').forEach(x=>x.classList.remove('active'));toast(cameraName.startsWith('wrist_')?'Gripper camera reset':'Free camera reset')}catch(e){toast(e.message)}}
function scheduleCameraAdjustment(){if(cameraAdjustTimer||cameraAdjustInFlight||!Object.keys(cameraAdjustPending).length)return;cameraAdjustTimer=setTimeout(flushCameraAdjustment,45)}
function queueCameraAdjustment(delta){const cameraName=ensureAdjustableCameraSensor();if(cameraAdjustPendingCamera&&cameraAdjustPendingCamera!==cameraName)cameraAdjustPending={};cameraAdjustPendingCamera=cameraName;for(const [key,value] of Object.entries(delta))cameraAdjustPending[key]=(cameraAdjustPending[key]||0)+value;scheduleCameraAdjustment()}
async function flushCameraAdjustment(){cameraAdjustTimer=null;if(cameraAdjustInFlight||!Object.keys(cameraAdjustPending).length)return;const delta=cameraAdjustPending,cameraName=cameraAdjustPendingCamera||ensureAdjustableCameraSensor();cameraAdjustPending={};cameraAdjustPendingCamera=null;cameraAdjustInFlight=true;try{const result=await post('/api/camera-adjust',{camera_name:cameraName,enabled:true,...delta});if(cameraName.startsWith('endoscope_'))currentViewMode='free';if(cameraAdjustmentTarget()===cameraName)renderFreeCamera(result)}catch(e){toast(e.message)}finally{cameraAdjustInFlight=false;scheduleCameraAdjustment()}}
function cycleCameraView(){const modes=['operative','close','overview','overhead','left_oblique','right_oblique','opposite'],mode=modes[(modes.indexOf(currentViewMode)+1)%modes.length],button=document.querySelector(`[data-view-mode="${mode}"]`);setCameraView(mode,button)}
function cycleSensorCamera(){const buttons=[...document.querySelectorAll('[data-camera]:not(.hidden)')];if(!buttons.length)return;const index=buttons.findIndex(button=>button.dataset.camera===currentCamera),button=buttons[(index+1)%buttons.length];setCamera(button.dataset.camera,button);toast(`${button.textContent.trim()} camera`)}
async function annotatePhase(phase){try{const x=await post('/api/annotation',{phase});toast(x.message)}catch(e){toast(e.message)}}
async function annotateEvent(event){try{const x=await post('/api/annotation',{event});toast('Procedure event saved')}catch(e){toast(e.message)}}
async function resetScene(){try{await post('/api/reset');toast('Scene reset')}catch(e){toast(e.message)}}
async function setAutonomy(mode){try{const x=await post('/api/autonomy',{mode});toast(x.message)}catch(e){toast(e.message)}}
async function takeControl(){stopDrive(false);try{const x=await post('/api/handoff');toast(x.message)}catch(e){toast(e.message)}}
async function startExpert(){try{const x=await post('/api/expert/start');toast(x.message)}catch(e){toast(e.message)}}
async function toggleExpertPause(){const status=latestStatus?.expert_demonstration?.status;try{const x=await post(status==='paused'?'/api/expert/resume':'/api/expert/pause');toast(x.message)}catch(e){toast(e.message)}}
function renderExpert(){}
function toggleReferenceGhost(){referenceGhost(!latestStatus?.reference_ghost?.enabled)}
function toggleKeyboardHelp(force){const help=document.getElementById('keyboardHelp'),show=force??help.classList.contains('hidden');help.classList.toggle('hidden',!show);if(show)stopDrive(false)}
function toggleControlPanel(force){const open=typeof force==='boolean'?force:!document.body.classList.contains('panel-open'),button=document.getElementById('panelToggle');document.body.classList.toggle('panel-open',open);button.setAttribute('aria-expanded',String(open));button.innerHTML=open?'Hide controls <kbd>Tab</kbd>':'Controls <kbd>Tab</kbd>'}
function auditKeyboardCoverage(){const buttons=[...document.querySelectorAll('button')],missing=buttons.filter(button=>!button.dataset.shortcut);const coverage=document.getElementById('keyboardCoverage');coverage.classList.toggle('bad',missing.length>0);coverage.textContent=missing.length?`${buttons.length-missing.length}/${buttons.length} controls mapped · ${missing.length} missing`:`✓ ${buttons.length}/${buttons.length} controls mapped to keyboard`;if(missing.length)console.warn('Buttons missing keyboard shortcuts',missing)}
function simulatorReadablePulse(minimum=550){const fps=Math.max(.5,latestStatus?.sim_fps||2);return Math.max(minimum,Math.ceil(1400/fps))}
async function pulseDrive(values,label,duration=simulatorReadablePulse(),speed=.35,source='keyboard_smart_action'){if(macroPulseTimer)clearTimeout(macroPulseTimer);clearHeldControls();driveWasActive=false;try{await post('/api/stop',{source});await post('/api/drive',{values,arm:activeArm,speed,source});updateControlReadout(true,`${label} · bounded pulse`);macroPulseTimer=setTimeout(()=>{macroPulseTimer=null;post('/api/stop',{source}).catch(()=>{});updateControlReadout(false)},duration)}catch(e){toast(e.message)}}
function smartTargetNudge(source='keyboard_smart_action'){const offset=latestStatus?.tool_to_object_offset_m?.[activeArm];if(!offset){toast('Target pose is not available yet');return}const ranked=offset.map((value,axis)=>({axis,value,magnitude:Math.abs(value)})).filter(item=>item.magnitude>.0025).sort((a,b)=>b.magnitude-a.magnitude).slice(0,2),values=Array(6).fill(0);ranked.forEach(({axis,value})=>{values[axis]=axis===0?(value<0?-.72:.72):(value>0?.72:-.72)});if(!ranked.length){toast('Target aligned · close the jaws');return}pulseDrive(values,'Target-guided nudge',simulatorReadablePulse(700),.5,source)}
function smartAction(source='keyboard_smart_action'){flashShortcut('F12',document.getElementById('smartActionLabel').textContent,1050);const s=latestStatus;if(!s){toast('Waiting for simulator state');return}const open=s.grippers_open?.[activeArm],nativeContact=s.native_grasp_contact_active?.[activeArm],distance=s.tool_to_object_distance_m?.[activeArm],capture=s.grasp_capture_radius_m||.018;if(open===undefined){smartTargetNudge(source);return}if(open&&distance!==null&&distance!==undefined&&distance<=capture){grip(false,activeArm,source);return}if(open){smartTargetNudge(source);return}if(nativeContact){pulseDrive([.65,0,.65,0,0,0],'Lift + retract',simulatorReadablePulse(850),.35,source);return}grip(true,activeArm,source)}
function setVoiceStatus(message,state=''){const status=document.getElementById('voiceStatus');status.textContent=message;status.className=`voice-status ${state}`.trim()}
async function voiceNudge(arm,values,label){if(voicePulseTimer)clearTimeout(voicePulseTimer);activeArm=arm;document.getElementById('arm0').classList.toggle('active',arm===0);document.getElementById('arm1').classList.toggle('active',arm===1);const duration=Math.min(1100,simulatorReadablePulse(650));try{await post('/api/stop',{source:'voice'});await post('/api/drive',{values:normalizeDrive(values),arm,speed:.35,source:'voice'});updateControlReadout(true,`Voice · instrument ${arm+1} ${label}`);setVoiceStatus(`Heard: instrument ${arm+1} ${label} · ${duration} ms pulse`,'ok');voicePulseTimer=setTimeout(()=>{voicePulseTimer=null;post('/api/stop',{source:'voice'}).catch(()=>{});updateControlReadout(false)},duration)}catch(e){setVoiceStatus(e.message,'error');toast(e.message)}}
function voiceArmAndText(command){let arm=activeArm,text=command;if(/\b(left robot|left instrument|instrument one|robot one|left gripper)\b/.test(text)){arm=0;text=text.replace(/\b(left robot|left instrument|instrument one|robot one)\b/g,' ').replace(/\bleft gripper\b/g,'gripper')}else if(/\b(right robot|right instrument|instrument two|robot two|right gripper)\b/.test(text)){arm=1;text=text.replace(/\b(right robot|right instrument|instrument two|robot two)\b/g,' ').replace(/\bright gripper\b/g,'gripper')}return {arm,text:text.replace(/\s+/g,' ').trim()}}
function executeVoiceCommand(rawCommand){const command=rawCommand.toLowerCase().replace(/[^a-z0-9 ]+/g,' ').replace(/\s+/g,' ').trim();document.getElementById('voiceCommand').value=rawCommand;if(!command){setVoiceStatus('Say or type a command','error');return false}if(/^(emergency )?stop( both( robots)?)?$/.test(command)){setVoiceStatus(`Heard: ${command}`,'ok');emergencyStop('voice');return true}const parsed=voiceArmAndText(command),arms=latestStatus?.arms||1,selection=/^(select |use )?(left|right)( robot| instrument)?$/.test(command)||/^(select |use )?(instrument|robot) (one|two)$/.test(command);if(selection){const selected=/\b(right|two)\b/.test(command)?1:0;if(selected>=arms){setVoiceStatus('The right robot is not available in this room','error');return false}setArm(selected);setVoiceStatus(`Instrument ${selected+1} selected`,'ok');return true}if(parsed.arm>=arms){setVoiceStatus('The right robot is not available in this room','error');return false}const sensorMatch=command.match(/^camera (left|right|wrist one|wrist two)$/);if(sensorMatch){const name={left:'endoscope_left',right:'endoscope_right','wrist one':'wrist_1','wrist two':'wrist_2'}[sensorMatch[1]];setCameraShortcut(name);setVoiceStatus(`Camera: ${sensorMatch[1]}`,'ok');return true}const cameraMatch=command.match(/^camera (operative|close|wide|overview|overhead|left angle|right angle|opposite|free|reset)$/);if(cameraMatch){const spoken=cameraMatch[1],mode={wide:'overview','left angle':'left_oblique','right angle':'right_oblique'}[spoken]||spoken;if(mode==='free'){if(!cameraAdjustMode)toggleFreeCamera()}else if(mode==='reset')resetFreeCamera();else setCameraView(mode,document.querySelector(`[data-view-mode="${mode}"]`));setVoiceStatus(`Camera: ${spoken}`,'ok');return true}const speedMatch=parsed.text.match(/^(precision|fine|normal|fast)( speed)?$/);if(speedMatch){const speed={precision:.35,fine:.35,normal:1,fast:1.7}[speedMatch[1]],label=parsed.arm===0?(speed===.35?'1':speed===1?'2':'3'):(speed===.35?'8':speed===1?'9':'0');setHandSpeed(parsed.arm,speed,label);setVoiceStatus(`Instrument ${parsed.arm+1}: ${speedMatch[1]} speed`,'ok');return true}const gripMatch=parsed.text.match(/^(open|close)( the)? gripper$/);if(gripMatch){grip(gripMatch[1]==='open',parsed.arm,'voice');setVoiceStatus(`${gripMatch[1]} instrument ${parsed.arm+1} gripper`,'ok');return true}if(/^(smart assist|assist)$/.test(parsed.text)){activeArm=parsed.arm;smartAction('voice_smart_action');setVoiceStatus(`Smart assist · instrument ${parsed.arm+1}`,'ok');return true}const directions={toward:[0,-1],forward:[0,-1],away:[0,1],back:[0,1],backward:[0,1],left:[1,1],right:[1,-1],up:[2,1],down:[2,-1]},words=parsed.text.split(/\s+(?:and\s+)?/).filter(Boolean),values=Array(6).fill(0),labels=[];for(const word of words){const move=directions[word];if(move){values[move[0]]+=move[1];labels.push(word)}}if(labels.length&&labels.length===words.length){voiceNudge(parsed.arm,values,labels.join(' + '));return true}setVoiceStatus(`Command not recognized: ${rawCommand}`,'error');return false}
function submitVoiceCommand(event){event.preventDefault();const input=document.getElementById('voiceCommand');executeVoiceCommand(input.value);input.select()}
function ensureVoiceRecognition(){if(voiceRecognition)return voiceRecognition;const Recognition=window.SpeechRecognition||window.webkitSpeechRecognition;if(!Recognition){setVoiceStatus('Microphone voice is unavailable here · type commands instead','error');return null}const recognition=new Recognition();recognition.continuous=false;recognition.interimResults=false;recognition.maxAlternatives=1;recognition.lang='en-US';recognition.onresult=event=>{const transcript=event.results[event.results.length-1][0].transcript;executeVoiceCommand(transcript)};recognition.onerror=event=>setVoiceStatus(`Voice error: ${event.error} · type commands instead`,'error');recognition.onend=()=>{voiceListening=false;document.getElementById('voiceMic').classList.remove('listening');const status=document.getElementById('voiceStatus');if(!status.classList.contains('ok')&&!status.classList.contains('error'))setVoiceStatus('Push to talk or type a bounded command')};voiceRecognition=recognition;return recognition}
function startVoiceInput(){if(voiceListening||pageDisposed)return;const recognition=ensureVoiceRecognition();if(!recognition)return;try{voiceListening=true;document.getElementById('voiceMic').classList.add('listening');setVoiceStatus('LISTENING · release to run command','listening');recognition.start()}catch(error){voiceListening=false;setVoiceStatus('Voice input is already active','error')}}
function finishVoiceInput(){if(!voiceListening)return;voiceListening=false;document.getElementById('voiceMic').classList.remove('listening');setVoiceStatus('Processing voice command…');try{voiceRecognition?.stop()}catch(_error){}}
const voiceMic=document.getElementById('voiceMic');voiceMic.addEventListener('pointerdown',event=>{event.preventDefault();voiceMic.setPointerCapture(event.pointerId);startVoiceInput()});const releaseVoice=event=>{if(voiceMic.hasPointerCapture?.(event.pointerId))voiceMic.releasePointerCapture(event.pointerId);finishVoiceInput()};voiceMic.addEventListener('pointerup',releaseVoice);voiceMic.addEventListener('pointercancel',releaseVoice);voiceMic.addEventListener('lostpointercapture',finishVoiceInput);
window.addEventListener('gamepadconnected',event=>{gamepadSafetyLatched=true;updateGamepadStatus();gamepadHaptic(event.gamepad,{duration:110,weak:.32,strong:.18});toast(`${event.gamepad.id||'Controller'} connected · center sticks to begin`)});window.addEventListener('gamepaddisconnected',event=>{gamepadKnownIndices.delete(event.gamepad.index);gamepadButtonStates.clear();latestGamepadCommands=new Map();updateGamepadStatus();if(!pageDisposed)emergencyStop('gamepad')});
function bindPointerHold(button,movement){button.addEventListener('pointerdown',event=>{event.preventDefault();if(isTypingTarget(document.activeElement))document.activeElement.blur();if(Number.isInteger(movement.arm)&&movement.arm<(latestStatus?.arms||1)){activeArm=movement.arm;document.getElementById('arm0').classList.toggle('active',activeArm===0);document.getElementById('arm1').classList.toggle('active',activeArm===1)}inputSource='keyboard_pointer';button.setPointerCapture(event.pointerId);pointerMoves.set(event.pointerId,{...movement,button});showKeyAction(button.dataset.shortcut,movement.label||button.textContent.trim(),true);syncKeyVisuals();updateDrive()});const release=event=>{pointerMoves.delete(event.pointerId);syncKeyVisuals();updateDrive();if(!pointerMoves.size)showKeyAction('READY','Released · motion stopped',false)};button.addEventListener('pointerup',release);button.addEventListener('pointercancel',release);button.addEventListener('lostpointercapture',release);button.addEventListener('contextmenu',event=>event.preventDefault())}
document.querySelectorAll('.move-button').forEach(button=>bindPointerHold(button,{axis:Number(button.dataset.axis),direction:Number(button.dataset.direction),arm:button.dataset.arm===undefined?undefined:Number(button.dataset.arm)}));
function isTypingTarget(target){return ['INPUT','SELECT','TEXTAREA'].includes(target.tagName)||target.isContentEditable}
function annotationShortcut(code){return {Digit1:['⌥1','Approach annotation',()=>annotatePhase('approach')],Digit2:['⌥2','Grasp annotation',()=>annotatePhase('grasp')],Digit3:['⌥3','Manipulation annotation',()=>annotatePhase('manipulation')],Digit4:['⌥4','Recovery annotation',()=>annotatePhase('recovery')],Digit5:['⌥5','Task event',()=>annotateEvent('task_complete')],Digit6:['⌥6','Safety event',()=>annotateEvent('safety_review')]}[code]}
function handleDiscreteShortcut(event){const {code}=event;if(code==='Slash'&&event.shiftKey){if(!event.repeat)runShortcut('?','Keyboard map',()=>toggleKeyboardHelp());return true}if(code==='F8'){if(!event.repeat)runShortcut('F8','Toggle free camera',()=>toggleFreeCamera());return true}if(code==='Home'){if(!event.repeat)runShortcut('Home','Reset free camera',()=>resetFreeCamera());return true}const annotation=event.altKey?annotationShortcut(code):null;if(annotation){if(!event.repeat)runShortcut(...annotation);return true}const speeds={Digit1:[0,.35,'1'],Digit2:[0,1,'2'],Digit3:[0,1.7,'3'],Digit8:[1,.35,'8'],Digit9:[1,1,'9'],Digit0:[1,1.7,'0'],Numpad8:[1,.35,'8'],Numpad9:[1,1,'9'],Numpad0:[1,1.7,'0']};if(speeds[code]){if(!event.repeat)setHandSpeed(...speeds[code]);return true}const cameraSensors={Digit4:['4','Stereo left camera','endoscope_left'],Digit5:['5','Stereo right camera','endoscope_right'],Digit6:['6','Wrist 1 camera','wrist_1'],Digit7:['7','Wrist 2 camera','wrist_2']},cameraViews={F1:['F1','Operative view','operative'],F2:['F2','Close view','close'],F3:['F3','Wide view','overview'],F4:['F4','Overhead view','overhead'],F5:['F5','Left oblique view','left_oblique'],F6:['F6','Right oblique view','right_oblique'],F7:['F7','Opposite-side view','opposite']};if(cameraSensors[code]){if(!event.repeat){const [shortcut,label,name]=cameraSensors[code];runShortcut(shortcut,label,()=>setCameraShortcut(name))}return true}if(cameraViews[code]){if(!event.repeat){const [shortcut,label,mode]=cameraViews[code];runShortcut(shortcut,label,()=>setCameraView(mode,document.querySelector(`[data-view-mode="${mode}"]`)))}return true}const commands={
  Tab:['Tab','Toggle instrument controls',()=>toggleControlPanel()],
  Space:['Space','Instrument 1 gripper',()=>toggleGrip(0)],Enter:['Enter','Instrument 2 gripper',()=>toggleGrip((latestStatus?.arms||1)>1?1:0)],NumpadEnter:['Enter','Instrument 2 gripper',()=>toggleGrip((latestStatus?.arms||1)>1?1:0)],Backspace:null,Escape:null,
  BracketLeft:['[','Pointer controls · instrument 1',()=>setArm(0)],BracketRight:[']','Pointer controls · instrument 2',()=>setArm(1)],KeyC:[event.shiftKey?'⇧C':'C',event.shiftKey?'Next camera view':'Next camera sensor',()=>event.shiftKey?cycleCameraView():cycleSensorCamera()],
  Comma:[',','Pointer precision speed',()=>setSpeedShortcut(.35)],Period:['.','Pointer normal speed',()=>setSpeedShortcut(1)],Slash:['/','Pointer fast speed',()=>setSpeedShortcut(1.7)],
  KeyG:[event.shiftKey?'⇧G':'G',event.shiftKey?'Manual control':'Guided control',()=>setAutonomy(event.shiftKey?'manual':'guided')],KeyH:['H','Toggle clinician path',()=>toggleReferenceGhost()],F9:['F9','Run live expert',()=>startExpert()],F10:['F10','Pause or resume expert',()=>toggleExpertPause()],F12:['F12','Smart context action',()=>smartAction()],
  KeyY:['Y','Start recording',()=>recording(true)],KeyT:['T','Stop and save',()=>recording(false)],KeyR:['R','Replay last',()=>replay()],Delete:['Delete','Reset scene',()=>resetScene()]
};if(code==='Backspace'||code==='Escape'){if(!event.repeat)emergencyStop();return true}const command=commands[code];if(!command)return false;if(!event.repeat)runShortcut(...command);return true}
document.addEventListener('keydown',event=>{if(event.code==='Backquote'&&!isTypingTarget(event.target)&&!event.metaKey&&!event.ctrlKey){event.preventDefault();if(!event.repeat)startVoiceInput();return}if(isTypingTarget(event.target)||event.metaKey||event.ctrlKey)return;const helpOpen=!document.getElementById('keyboardHelp').classList.contains('hidden');if(helpOpen&&event.code!=='Slash'&&event.code!=='Escape'&&event.code!=='Backspace'){event.preventDefault();return}if(event.code==='ShiftLeft'||event.code==='ShiftRight'){event.preventDefault();heldModifiers.add(event.code==='ShiftLeft'?'rotate-left':'rotate-right');showKeyAction(event.code==='ShiftLeft'?'L⇧':'R⇧',`${event.code==='ShiftLeft'?'Left':'Right'} wrist angle mode`,true);syncKeyVisuals();updateDrive();return}if(event.code==='AltLeft'||event.code==='AltRight'){event.preventDefault();const modifier=event.code==='AltLeft'?'precision-left':'precision-right';heldModifiers.add(modifier);showKeyAction(event.code==='AltLeft'?'L⌥':'R⌥',`${event.code==='AltLeft'?'Left':'Right'} precision clutch`,true);syncKeyVisuals();updateDrive();return}if(handleDiscreteShortcut(event)){event.preventDefault();if((event.code==='Escape'||event.code==='Backspace')&&helpOpen)toggleKeyboardHelp(false);return}if(!dualMovementCodes.has(event.code))return;event.preventDefault();inputSource='keyboard_pointer';if(!heldKeys.has(event.code))heldKeyStartedAt.set(event.code,performance.now());heldKeys.add(event.code);updateDrive();showKeyAction(event.key.length===1?event.key.toUpperCase():event.key,activeDriveLabel(),true)});
document.addEventListener('keyup',event=>{if(event.code==='Backquote'&&!isTypingTarget(event.target)){event.preventDefault();finishVoiceInput();return}if(event.code==='ShiftLeft'||event.code==='ShiftRight'){event.preventDefault();heldModifiers.delete(event.code==='ShiftLeft'?'rotate-left':'rotate-right');syncKeyVisuals();updateDrive();if(!bimanualWasActive)showKeyAction('READY','Wrist angle mode released',false);return}if(event.code==='AltLeft'||event.code==='AltRight'){event.preventDefault();heldModifiers.delete(event.code==='AltLeft'?'precision-left':'precision-right');syncKeyVisuals();updateDrive();return}if(!dualMovementCodes.has(event.code))return;event.preventDefault();heldKeys.delete(event.code);heldKeyStartedAt.delete(event.code);updateDrive();showKeyAction(heldKeys.size?'HOLD':'READY',heldKeys.size?activeDriveLabel():'Released · motion stopped',heldKeys.size>0)});
window.addEventListener('message',event=>{let sameHost=false;try{sameHost=new URL(event.origin).hostname===location.hostname}catch(_error){}const message=event.data;if(!sameHost||event.source!==parent||message?.type!=='dr-anmar-control-key')return;if(message.releaseAll){parentKeyboardActive=true;stopDrive(false);parentKeyboardActive=false;showKeyAction('READY','All controls released',false);return}if(!message.code||typeof message.down!=='boolean')return;parentKeyboardActive=true;document.dispatchEvent(new KeyboardEvent(message.down?'keydown':'keyup',{code:message.code,key:message.key||'',repeat:!!message.repeat,shiftKey:!!message.shiftKey,altKey:!!message.altKey,bubbles:true,cancelable:true}));if(!message.down&&!heldKeys.size&&!heldModifiers.size)parentKeyboardActive=false});
window.addEventListener('blur',()=>stopDrive(false));document.addEventListener('visibilitychange',()=>{if(document.hidden)stopDrive(false)});
const cameraView=document.getElementById('cameraView');
cameraView.addEventListener('pointerdown',event=>{if(!cameraAdjustMode||event.button!==0)return;event.preventDefault();cameraView.setPointerCapture(event.pointerId);cameraDrag={pointerId:event.pointerId,x:event.clientX,y:event.clientY,pan:event.shiftKey};cameraView.classList.add('dragging');cameraView.classList.remove('gaze-on')});
cameraView.addEventListener('pointermove',event=>{if(cameraDrag&&cameraDrag.pointerId===event.pointerId){event.preventDefault();const dx=event.clientX-cameraDrag.x,dy=event.clientY-cameraDrag.y;cameraDrag.x=event.clientX;cameraDrag.y=event.clientY;if(cameraDrag.pan)queueCameraAdjustment({pan_x_delta_m:-dx*.00055,pan_y_delta_m:dy*.00055});else queueCameraAdjustment({orbit_yaw_delta_deg:dx*.22,orbit_pitch_delta_deg:dy*.20});return}const rect=cameraView.getBoundingClientRect(),u=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),v=Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height));const cursor=document.getElementById('gazeCursor');cursor.style.left=`${u*100}%`;cursor.style.top=`${v*100}%`;cameraView.classList.add('gaze-on');const now=performance.now();if(now-lastGazeSend>100){lastGazeSend=now;post('/api/gaze',{u,v,valid:true,source:'pointer_attention_proxy'}).catch(()=>{})}});
function finishCameraDrag(event){if(!cameraDrag||cameraDrag.pointerId!==event.pointerId)return;cameraDrag=null;cameraView.classList.remove('dragging')}
cameraView.addEventListener('pointerup',finishCameraDrag);cameraView.addEventListener('pointercancel',finishCameraDrag);cameraView.addEventListener('lostpointercapture',finishCameraDrag);cameraView.addEventListener('pointerleave',()=>{if(!cameraDrag)cameraView.classList.remove('gaze-on')});cameraView.addEventListener('wheel',event=>{if(!cameraAdjustMode)return;event.preventDefault();queueCameraAdjustment({zoom_delta:Math.sign(event.deltaY)*.08})},{passive:false});
function targetDirections(offset){if(!offset)return'';const choices=[];if(Math.abs(offset[2])>.004)choices.push([Math.abs(offset[2]),offset[2]>0?'Up':'Down']);if(Math.abs(offset[1])>.004)choices.push([Math.abs(offset[1]),offset[1]>0?'Left':'Right']);if(Math.abs(offset[0])>.004)choices.push([Math.abs(offset[0]),offset[0]<0?'Toward':'Away']);return choices.sort((a,b)=>b[0]-a[0]).slice(0,2).map(x=>x[1]).join(' + ')}
async function refresh(){if(refreshInFlight||pageDisposed||document.hidden)return;refreshInFlight=true;try{
  const s=await requestJson('/api/status/live',{cache:'no-store'},2500);if(workerInstanceId&&s.instance_id!==workerInstanceId){location.reload();return}workerInstanceId=s.instance_id;latestStatus=s;if(activeArm>=s.arms){activeArm=0;document.getElementById('arm0').classList.add('active');document.getElementById('arm1').classList.remove('active')}document.getElementById('dot').classList.add('ok');document.getElementById('connection').textContent='Isaac Lab live';const contactPad=standardGamepads()[0];for(let arm=0;arm<2;arm++){const contact=!!s.native_grasp_contact_active?.[arm];if(contact&&!previousGamepadContacts[arm])gamepadHaptic(contactPad,{duration:95,weak:arm===0 ? .48 : .12,strong:arm===1 ? .48 : .12});previousGamepadContacts[arm]=contact}
  const p=s.procedure||{};document.getElementById('procedureTitle').textContent=p.title||'Free practice';document.getElementById('procedureObjective').textContent=p.objective||'Use the robot controls to explore the digital twin.';document.getElementById('procedureProgress').style.width=`${p.progress_percent||0}%`;const procedureMarkup=(p.steps||[]).map((x,i)=>`<div class="procedure-step ${x.status}"><span>${String(i+1).padStart(2,'0')}</span><div><b>${x.title}</b><br>${x.instruction}</div></div>`).join(''),procedureSteps=document.getElementById('procedureSteps');if(procedureSteps.dataset.markup!==procedureMarkup){procedureSteps.innerHTML=procedureMarkup;procedureSteps.dataset.markup=procedureMarkup}const patient=s.dynamic_patient||{},patientRoom=patient.access_state==='intact'||patient.access_state==='open',sessionDetails=document.querySelector('.session-details');if(patientRoom&&!sessionDetails.dataset.patientRoomShown){sessionDetails.open=true;sessionDetails.dataset.patientRoomShown='true'}
  document.querySelectorAll('[data-camera]').forEach(button=>button.classList.toggle('hidden',!s.camera_names.includes(button.dataset.camera)));document.getElementById('rightInstrumentControls').classList.toggle('hidden',s.arms<2);document.getElementById('instrumentGrid').classList.toggle('single',s.arms<2);document.querySelectorAll('.gripper-control').forEach(button=>button.classList.toggle('hidden',!s.has_grippers));
  currentViewMode=s.camera_view_mode||currentViewMode;renderFreeCamera(selectedCameraAdjustment(s));document.querySelectorAll('[data-view-mode]').forEach(x=>x.classList.toggle('active',!cameraAdjustMode&&x.dataset.viewMode===currentViewMode));
  document.getElementById('recflag').classList.toggle('on',s.recording);document.getElementById('record')?.classList.toggle('state-active',s.recording);document.getElementById('gripOpenButton').classList.toggle('state-active',s.grippers_open?.[0]===false);document.getElementById('gripCloseButton').classList.toggle('state-active',s.grippers_open?.[(s.arms||1)>1?1:0]===false);
	  const proximity=document.getElementById('proximity'),distance=s.tool_to_object_distance_m?.[activeArm],offset=s.tool_to_object_offset_m?.[activeArm],clearance=s.closest_anatomy_clearance_m;proximity.className='proximity';let guidance='Move toward the target';if(s.native_grasp_contact_active?.[activeArm]){guidance='Native jaw contact detected · lift smoothly';proximity.classList.add('held')}else if(distance!==null&&distance!==undefined&&distance<=(s.grasp_capture_radius_m||.018)){guidance=`Aligned ${Math.round(distance*1000)} mm · close jaws`;proximity.classList.add('near')}else if(distance!==null&&distance!==undefined){guidance=`Target ${Math.round(distance*1000)} mm · ${targetDirections(offset)||'hold course'}`}else if(clearance!==null&&clearance!==undefined){guidance=`Anatomy clearance ${Math.round(clearance*1000)} mm`};proximity.innerHTML=`<b>Next</b><span>${guidance}</span>`;const smartLabel=document.getElementById('smartActionLabel'),open=s.grippers_open?.[activeArm],contact=s.native_grasp_contact_active?.[activeArm];smartLabel.textContent=open===undefined?'Precision nudge toward target':open&&distance!==null&&distance!==undefined&&distance<=(s.grasp_capture_radius_m||.018)?'Close jaws on aligned target':open?'Precision nudge toward target':contact?'Lift the physically held object':'Open jaws and retry';
  const labels={manual:'L0 · Manual',guided:'L1 · Guided',supervised_replay:'L2 · Supervised replay',expert_demonstration:'L2 · Live expert'};document.getElementById('autonomyState').textContent=labels[s.autonomy_mode]||s.autonomy_mode;document.getElementById('manualMode').classList.toggle('active',s.autonomy_mode==='manual');document.getElementById('guidedMode').classList.toggle('active',s.autonomy_mode==='guided');document.getElementById('coachingCue').textContent=s.coaching_cue;document.getElementById('forceMetric').textContent=s.safety?.max_contact_force_n===null?'—':Number(s.safety.max_contact_force_n).toFixed(2);document.getElementById('deformMetric').textContent=s.safety?.max_tissue_displacement_m===null?'—':(Number(s.safety.max_tissue_displacement_m)*1000).toFixed(1);document.getElementById('stressMetric').textContent=s.safety?.max_tissue_stress_pa===null?'—':Number(s.safety.max_tissue_stress_pa).toExponential(1);renderClosureRobot(s.closure_robot_system);renderStaplerCell(s.stapler_test_cell);renderSkinAdhesive(s.skin_adhesive_system);renderExpert(s.expert_demonstration);
	  if(s.last_demo)document.getElementById('lastDemo').innerHTML=`Last saved: <a href="/demos/${s.last_demo}" style="color:#2cd2e8">${s.last_demo}</a>`;
}catch(e){document.getElementById('dot').classList.remove('ok');document.getElementById('connection').textContent='Reconnecting…'}finally{refreshInFlight=false}}
async function heartbeat(){if(heartbeatInFlight||pageDisposed||document.hidden)return;heartbeatInFlight=true;try{await post('/api/operator/heartbeat',{},3000)}catch(_error){}finally{heartbeatInFlight=false}}
function releasePageResources(){if(pageDisposed)return;pageDisposed=true;queuedDrive=null;queuedBimanual=null;clearInterval(driveInterval);clearInterval(refreshInterval);clearInterval(heartbeatInterval);if(gamepadAnimationFrame!==null)cancelAnimationFrame(gamepadAnimationFrame);if(macroPulseTimer)clearTimeout(macroPulseTimer);if(voicePulseTimer)clearTimeout(voicePulseTimer);if(keyFlashTimer)clearTimeout(keyFlashTimer);if(toastTimer)clearTimeout(toastTimer);if(cameraAdjustTimer)clearTimeout(cameraAdjustTimer);try{voiceRecognition?.abort()}catch(_error){}voiceRecognition=null;voiceListening=false;gamepadButtonStates.clear();gamepadKnownIndices.clear();latestGamepadCommands=new Map();activeFetchControllers.forEach(controller=>controller.abort());activeFetchControllers.clear();cameraFeedController=null;if(cameraObjectUrl){URL.revokeObjectURL(cameraObjectUrl);cameraObjectUrl=null}clearHeldControls();const image=document.getElementById('cameraImage');image.removeAttribute('src');const options={method:'POST',headers:{'content-type':'application/json','x-dr-anmar-operator':operatorId},body:JSON.stringify({source:'keyboard_pointer'}),keepalive:true};fetch('/api/stop',options).catch(()=>{});fetch('/api/operator/release',{...options,body:'{}'}).catch(()=>{})}
auditKeyboardCoverage();if(!(window.SpeechRecognition||window.webkitSpeechRecognition))setVoiceStatus('Microphone unavailable here · type commands instead');updateGamepadStatus();startCameraFeed(currentCamera);const driveInterval=setInterval(updateDrive,33),refreshInterval=setInterval(refresh,500),heartbeatInterval=setInterval(heartbeat,1000);gamepadAnimationFrame=requestAnimationFrame(pollGamepads);window.addEventListener('pagehide',releasePageResources,{once:true});window.addEventListener('pageshow',event=>{if(event.persisted&&pageDisposed)location.reload()});document.addEventListener('visibilitychange',()=>{if(!document.hidden){refresh();heartbeat()}});refresh();
</script><script type="module" src="./hand-control.mjs"></script></body></html>"""


class JogRequest(BaseModel):
    axis: int
    direction: int
    arm: int = 0


class DriveRequest(BaseModel):
    values: list[float]
    arm: int = 0
    speed: float = 1.0
    source: str = "web_control"


class BimanualArmDrive(BaseModel):
    arm: int
    values: list[float]
    speed: float = 1.0


class BimanualDriveRequest(BaseModel):
    commands: list[BimanualArmDrive]
    source: str = "keyboard_pointer"


class StopRequest(BaseModel):
    source: str = "keyboard_pointer"


class GripperRequest(BaseModel):
    open: bool
    arm: int = 0
    source: str = "keyboard_pointer"


class GripperToggleRequest(BaseModel):
    arm: int = 0
    source: str = "keyboard_pointer"


class HandTeleopInput(BaseModel):
    arm: int
    tracked: bool
    motion_engaged: bool
    translation_offset_m: list[float]
    rotation_vector_rad: list[float]
    aperture_normalized: float
    confidence: float


class HandTeleopRequest(BaseModel):
    sequence: int
    hands: list[HandTeleopInput]
    captured_at_ms: float | None = None
    inference_ms: float | None = None
    client_sent_at_ms: float | None = None
    transport_drops: int = 0


class HandTeleopControlRequest(BaseModel):
    enabled: bool
    reason: str = "operator"


class CameraViewRequest(BaseModel):
    mode: str


class CameraAdjustRequest(BaseModel):
    camera_name: str = "endoscope_left"
    enabled: bool = True
    orbit_yaw_delta_deg: float = 0.0
    orbit_pitch_delta_deg: float = 0.0
    zoom_delta: float = 0.0
    pan_x_delta_m: float = 0.0
    pan_y_delta_m: float = 0.0
    reset: bool = False


class StaplerCommandRequest(BaseModel):
    action: str
    target_deg: float | None = None


class SkinAdhesiveActivationRequest(BaseModel):
    activation: float


class ClosureRobotCommandRequest(BaseModel):
    action: str


class ScenarioRequest(BaseModel):
    scenario_id: str
    seed: int = 7777


class AutonomyRequest(BaseModel):
    mode: str


class EvaluationRequest(BaseModel):
    demo: str
    scenario_id: str
    seed: int = 7777


class ReferenceGhostRequest(BaseModel):
    enabled: bool = True
    demo: str | None = None


class GazeRequest(BaseModel):
    u: float
    v: float
    valid: bool = True
    source: str = "pointer_attention_proxy"


class ProcedureAnnotationRequest(BaseModel):
    phase: str | None = None
    event: str | None = None
    note: str = ""


@dataclass
class SharedState:
    task: str
    camera_width: int
    camera_height: int
    demo_dir: Path
    action_dim: int
    arms: int
    has_grippers: bool
    robot_names: list[str]
    robot_body_names: dict[str, list[str]]
    anatomy_showcase: str | None = None
    anatomy_scene_id: str = ""
    anatomy_asset: str = ""
    openusd_environment: str = ""
    procedure: dict[str, Any] = field(default_factory=dict)
    openusd_scene_loaded: bool = False
    anatomy_collision_meshes: int = 0
    instance_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    lock: threading.Lock = field(default_factory=threading.Lock)
    wake_event: threading.Event = field(default_factory=threading.Event)
    frame_jpeg: bytes = b""
    frame_id: int = 0
    camera_frames_jpeg: dict[str, bytes] = field(default_factory=dict)
    camera_frame_ids: dict[str, int] = field(default_factory=dict)
    camera_names: list[str] = field(default_factory=list)
    camera_subscribers: dict[str, int] = field(default_factory=dict)
    camera_poll_last_seen: float = 0.0
    camera_poll_last_seen_by_name: dict[str, float] = field(default_factory=dict)
    jpeg_queue_depth: int = 0
    jpeg_frames_dropped: int = 0
    render_fps: float = 0.0
    sim_fps: float = 0.0
    sim_step: int = 0
    pulse: np.ndarray = field(init=False)
    pulse_steps: int = 0
    drive: np.ndarray = field(init=False)
    drive_until: float = 0.0
    drive_min_steps_remaining: int = 0
    drive_stop_pending: bool = False
    control_sequence: int = 0
    control_last_kind: str = "idle"
    control_last_source: str = "none"
    control_last_action: list[float] = field(default_factory=list)
    control_last_at: float = 0.0
    control_last_nonzero_sequence: int = 0
    control_last_nonzero_action: list[float] = field(default_factory=list)
    grippers_open: list[bool] = field(init=False)
    gripper_apertures: list[float] = field(init=False)
    hand_teleop: HandTeleopRuntime = field(init=False)
    native_ik_scales: list[list[float]] = field(default_factory=list)
    hand_camera_to_action_basis: list[list[list[float]]] = field(default_factory=list)
    hand_camera_control_name: str = "endoscope_left"
    hand_camera_control_revision: int = 0
    hand_last_received_sequence: int = -1
    hand_last_received_at: float = 0.0
    hand_last_applied_sequence: int = -1
    hand_last_applied_at: float = 0.0
    hand_latest_transport_age_ms: float | None = None
    hand_latest_capture_age_ms: float | None = None
    hand_latest_inference_ms: float | None = None
    hand_transport_drops: int = 0
    hand_authority_reason: str = "disabled"
    native_grasp_contact_active: list[bool] = field(init=False)
    tool_to_object_distance_m: list[float | None] = field(init=False)
    tool_to_object_offset_m: list[list[float] | None] = field(init=False)
    grasp_capture_radius_m: float = 0.018
    camera_view_mode: str = "free"
    camera_view_request: str | None = None
    camera_free_enabled: bool = True
    camera_free_base_mode: str = "operative"
    camera_free_yaw_deg: float = 0.0
    camera_free_pitch_deg: float = 0.0
    camera_free_zoom: float = 1.0
    camera_free_pan_x_m: float = 0.0
    camera_free_pan_y_m: float = 0.0
    gripper_camera_adjustments: dict[str, dict[str, float | bool | str]] = field(
        default_factory=dict
    )
    virtual_fixture_enabled: bool = False
    virtual_fixture_active: bool = False
    closest_anatomy_clearance_m: float | None = None
    needle_tip_clearance_m: float | None = None
    needle_surface_outward: list[float] | None = None
    needle_surface_direction: list[float] | None = None
    needle_entry_direction: list[float] | None = None
    adaptive_precision_active: bool = False
    reset_requested: bool = False
    record_request: str | None = None
    recording: bool = False
    recorded_frames: int = 0
    recorded_bytes_estimate: int = 0
    recording_queue_depth: int = 0
    recording_buffered_frames: int = 0
    sensor_profile: str = "research"
    last_demo: str | None = None
    replay_request: str | None = None
    replaying: bool = False
    expert_request: str | None = None
    expert_demonstration: dict[str, Any] = field(default_factory=dict)
    expert_reference_pending: bool = False
    expert_reference_demo: str | None = None
    expert_clean_run: bool = False
    scenario_id: str = "baseline"
    scenario_seed: int = DEFAULT_SCENARIO_SEED
    autonomy_mode: str = "manual"
    intervention_count: int = 0
    coaching_cue: str = "You command every movement. Dr.Anmar records telemetry for coaching."
    evaluation_status: str = "idle"
    evaluation_source: str | None = None
    evaluation_output: str | None = None
    camera_intrinsics: list[list[float]] | None = None
    semantic_labels: dict[str, Any] = field(default_factory=dict)
    runtime_provenance: dict[str, Any] = field(default_factory=dict)
    camera_valid_depth_fraction: float | None = None
    camera_foreground_fraction: float | None = None
    camera_mean_luminance: float | None = None
    camera_nonblank_seen: bool = False
    needle_visual_ready: bool = False
    deformable_strand_ready: bool = False
    native_rigid_object_names: list[str] = field(default_factory=list)
    native_deformable_object_names: list[str] = field(default_factory=list)
    native_rigid_object_positions_m: dict[str, list[float]] = field(default_factory=dict)
    native_tool_positions_m: list[list[float] | None] = field(default_factory=list)
    native_psm_policy_contract: bool = False
    native_psm_policy_dim: int = 0
    native_psm_robot_names: list[str] = field(default_factory=list)
    gripper_profile: dict[str, Any] = field(default_factory=dict)
    ring_physics_ready: bool = False
    strand_self_collision_ready: bool = False
    reference_ghost_enabled: bool = False
    reference_ghost_demo: str | None = None
    reference_ghost_update: str | None = None
    reference_ghost_points: int = 0
    max_contact_force_n: float | None = None
    max_tissue_displacement_m: float | None = None
    max_tissue_deformation_proxy: float | None = None
    max_tissue_stress_pa: float | None = None
    gaze_uv: tuple[float, float] = (0.5, 0.5)
    gaze_valid: bool = False
    gaze_source: str = "none"
    operator_input_source: str = "none"
    procedure_phase: str = "setup"
    procedure_event_code: int = 0
    procedure_event_sequence: int = 0
    procedure_events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=4096))
    dynamic_patient_access_state: str = ""
    dynamic_patient_cut_events: int = 0
    procedure_waypoints_total: int = 0
    procedure_waypoints_completed: int = 0
    procedure_motion_seen: bool = False
    procedure_grasp_seen: bool = False
    procedure_object_lift_m: float = 0.0
    procedure_object_motion_m: float = 0.0
    procedure_started_at: float = 0.0
    procedure_last_motion_at: float = 0.0
    native_telemetry: dict[str, Any] = field(default_factory=dict)
    stapler_command_request: str | None = None
    stapler_station_request: int | None = None
    stapler_manual_target_deg: float = 0.0
    stapler_test_cell: dict[str, Any] = field(default_factory=dict)
    skin_adhesive_target: float = 0.0
    skin_adhesive_system: dict[str, Any] = field(default_factory=dict)
    closure_robot_command_request: str | None = None
    closure_robot_system: dict[str, Any] = field(default_factory=dict)
    dr_anmar_needle_domain: dict[str, float | int] = field(
        default_factory=dict
    )
    upstream_task_success: bool | None = None
    performance_timings_ms: dict[str, float] = field(default_factory=dict)
    simulation_profile: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pulse = np.zeros(self.action_dim, dtype=np.float32)
        self.drive = np.zeros(self.action_dim, dtype=np.float32)
        self.grippers_open = [True] * self.arms
        self.gripper_apertures = [1.0] * self.arms
        self.hand_teleop = HandTeleopRuntime(self.arms)
        self.native_grasp_contact_active = [False] * self.arms
        self.tool_to_object_distance_m = [None] * self.arms
        self.tool_to_object_offset_m = [None] * self.arms

    def body_action_slice(self, arm: int) -> slice:
        group_width = 7 if self.has_grippers else 6
        start = arm * group_width
        return slice(start, start + 6)

    def gripper_action_index(self, arm: int) -> int:
        if not self.has_grippers:
            raise ValueError("This task has no gripper action")
        return arm * 7 + 6

    def note_control(self, kind: str, source: str, action: np.ndarray) -> None:
        """Record one fixed-size control observation while the caller owns ``lock``."""
        self.control_sequence += 1
        self.control_last_kind = kind
        self.control_last_source = source
        action_array = np.asarray(action, dtype=np.float32)
        self.control_last_action = action_array.tolist()
        self.control_last_at = time.monotonic()
        if bool(np.any(action_array)):
            self.control_last_nonzero_sequence = self.control_sequence
            self.control_last_nonzero_action = action_array.tolist()

    def disable_hand_motion(self, *, require_unclutched: bool = True) -> None:
        """Freeze and invalidate all pending webcam pose displacement."""

        self.hand_teleop.disable_motion(require_unclutched=require_unclutched)
        self.hand_authority_reason = "frozen:other_control"

    def hand_teleop_snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        """Return hand state plus the live camera-aligned IK control frame."""

        timestamp = time.monotonic() if now is None else float(now)
        snapshot = self.hand_teleop.snapshot(now=timestamp)
        snapshot["control_frame"] = {
            "name": self.hand_camera_control_name,
            "revision": self.hand_camera_control_revision,
            "semantic_axes": ["camera_forward", "camera_right", "camera_up"],
            "camera_to_action_basis": [
                [list(row) for row in basis]
                for basis in self.hand_camera_to_action_basis
            ],
        }
        snapshot["transport"] = {
            "received_sequence": self.hand_last_received_sequence,
            "applied_sequence": self.hand_last_applied_sequence,
            "latest_age_ms": self.hand_latest_transport_age_ms,
            "capture_age_ms": self.hand_latest_capture_age_ms,
            "inference_ms": self.hand_latest_inference_ms,
            "dropped_client_frames": self.hand_transport_drops,
            "received_age_ms": (
                round(max(0.0, timestamp - self.hand_last_received_at) * 1000)
                if self.hand_last_received_at > 0.0
                else None
            ),
            "applied_age_ms": (
                round(max(0.0, timestamp - self.hand_last_applied_at) * 1000)
                if self.hand_last_applied_at > 0.0
                else None
            ),
            "authority_reason": self.hand_authority_reason,
        }
        return snapshot

    def camera_adjustment(self, camera_name: str = "endoscope_left") -> dict[str, float | bool | str]:
        """Return one camera's adjustment while the caller owns ``lock``."""

        if camera_name.startswith("wrist_"):
            adjustment = self.gripper_camera_adjustments.get(camera_name)
            if adjustment is None:
                adjustment = {
                    "enabled": True,
                    "base_mode": "tool_axis",
                    "yaw_deg": 0.0,
                    "pitch_deg": 0.0,
                    "zoom": 1.0,
                    "pan_x_m": 0.0,
                    "pan_y_m": 0.0,
                }
                self.gripper_camera_adjustments[camera_name] = adjustment
            return {"camera_name": camera_name, **adjustment}
        return {
            "camera_name": camera_name,
            "enabled": self.camera_free_enabled,
            "base_mode": self.camera_free_base_mode,
            "yaw_deg": self.camera_free_yaw_deg,
            "pitch_deg": self.camera_free_pitch_deg,
            "zoom": self.camera_free_zoom,
            "pan_x_m": self.camera_free_pan_x_m,
            "pan_y_m": self.camera_free_pan_y_m,
        }

    def camera_adjustments(self) -> dict[str, dict[str, float | bool | str]]:
        """Return adjustments for the camera streams that support direct manipulation."""

        names = [
            name
            for name in self.camera_names
            if name.startswith("endoscope_") or name.startswith("wrist_")
        ]
        return {name: self.camera_adjustment(name) for name in names}

    def status(self) -> dict[str, Any]:
        with self.lock:
            procedure_status = self._procedure_status()
            guide_kind = str(self.procedure.get("guide_kind", ""))
            active_bench_assets = set(
                self.procedure.get("active_bench_assets", ())
            )
            authored_suture_selected = bool(
                active_bench_assets.intersection(
                    {
                        "dr_anmar_needle_suture",
                        "nvidia_needle_dr_anmar_suture",
                        "dr_anmar_needle_thread_coiled",
                        "dr_anmar_needle_thread_extended",
                        "dr_anmar_needle_thread_proxy",
                    }
                )
            )
            thread_required = bool(
                guide_kind == "softmimicgen_threading"
                or authored_suture_selected
            )
            needle_required = bool(
                guide_kind in NATIVE_NEEDLE_GUIDE_KINDS
                or (
                    guide_kind == "softmimicgen_threading"
                    and self.procedure.get("bimanual")
                )
                or active_bench_assets.intersection(
                    {
                        "needle",
                        "dr_anmar_needle",
                        "dr_anmar_needle_suture",
                        "nvidia_needle_dr_anmar_suture",
                        "dr_anmar_needle_v030",
                        "dr_anmar_needle_thread_coiled",
                        "dr_anmar_needle_thread_extended",
                        "dr_anmar_needle_thread_proxy",
                    }
                )
            )
            raw_thread_geometry_ready = bool(
                active_bench_assets.intersection(
                    {
                        "dr_anmar_needle_thread_coiled",
                        "dr_anmar_needle_thread_extended",
                        "dr_anmar_needle_thread_proxy",
                    }
                )
            )
            thread_geometry_ready = (
                not thread_required
                or self.deformable_strand_ready
                or raw_thread_geometry_ready
            )
            needle_geometry_ready = not needle_required or self.needle_visual_ready
            camera_frame_ready = bool(self.frame_id > 0 and self.frame_jpeg and self.camera_nonblank_seen)
            render_contract = {
                # A real camera frame is the only condition for presenting the
                # room. Needle and strand fields below are research telemetry,
                # never UI gates.
                "ready": camera_frame_ready,
                "camera_frame_ready": camera_frame_ready,
                "camera_nonblank_seen": self.camera_nonblank_seen,
                "needle_required": needle_required,
                "needle_geometry_ready": needle_geometry_ready,
                "thread_required": thread_required,
                "thread_geometry_ready": thread_geometry_ready,
                "target_anchor_markers": max(2, int(self.procedure.get("target_anchors", 2))) if thread_required else 0,
            }
            return {
                "task": self.task,
                "instance_id": self.instance_id,
                "camera_width": self.camera_width,
                "camera_height": self.camera_height,
                "camera_names": self.camera_names,
                "active_camera_streams": sum(1 for count in self.camera_subscribers.values() if count > 0),
                "jpeg_queue_depth": self.jpeg_queue_depth,
                "jpeg_frames_dropped": self.jpeg_frames_dropped,
                "frame_id": self.frame_id,
                "render_fps": self.render_fps,
                "sim_fps": self.sim_fps,
                "sim_step": self.sim_step,
                "action_dim": self.action_dim,
                "action_contract": ACTION_CONTRACT if self.native_psm_policy_contract else NON_PSM_ACTION_CONTRACT,
                "native_psm_policy_dim": self.native_psm_policy_dim,
                "gripper_profile": self.gripper_profile,
                "arms": self.arms,
                "has_grippers": self.has_grippers,
                "robot_names": self.robot_names,
                "robot_body_names": self.robot_body_names,
                "anatomy_showcase": self.anatomy_showcase,
                "anatomy_scene_id": self.anatomy_scene_id,
                "anatomy_asset": self.anatomy_asset,
                "openusd_environment": self.openusd_environment,
                "openusd_scene_loaded": self.openusd_scene_loaded,
                "anatomy_collision_meshes": self.anatomy_collision_meshes,
                "procedure": procedure_status,
                "dynamic_patient": {
                    "access_state": self.dynamic_patient_access_state or None,
                    "access_pending": False,
                    "cut_events": self.dynamic_patient_cut_events,
                    "active_deformables": list(
                        self.procedure.get(
                            "dynamic_patient_active_deformables", ()
                        )
                    ),
                },
                "upstream_task_success": self.upstream_task_success,
                "performance_timings_ms": dict(self.performance_timings_ms),
                "simulation_profile": dict(self.simulation_profile),
                "grippers_open": self.grippers_open,
                "gripper_apertures": self.gripper_apertures,
                "hand_teleop": self.hand_teleop_snapshot(),
                "native_grasp_contact_active": self.native_grasp_contact_active,
                "tool_to_object_distance_m": self.tool_to_object_distance_m,
                "tool_to_object_offset_m": self.tool_to_object_offset_m,
                "grasp_capture_radius_m": self.grasp_capture_radius_m,
                "camera_view_mode": self.camera_view_mode,
                "camera_adjustable": self.camera_adjustment(),
                "camera_adjustable_by_name": self.camera_adjustments(),
                "virtual_fixture_enabled": self.virtual_fixture_enabled,
                "virtual_fixture_active": self.virtual_fixture_active,
                "closest_anatomy_clearance_m": self.closest_anatomy_clearance_m,
                "needle_tip_clearance_m": self.needle_tip_clearance_m,
                "needle_surface_outward": self.needle_surface_outward,
                "needle_surface_direction": self.needle_surface_direction,
                "needle_entry_direction": self.needle_entry_direction,
                "adaptive_precision_active": self.adaptive_precision_active,
                "recording": self.recording,
                "recorded_frames": self.recorded_frames,
                "recorded_bytes_estimate": self.recorded_bytes_estimate,
                "recording_limit_bytes": MAX_DEMO_BYTES,
                "recording_queue_depth": self.recording_queue_depth,
                "recording_buffered_frames": self.recording_buffered_frames,
                "recording_storage": "bounded-hdf5-spool-plus-npz",
                "sensor_profile": self.sensor_profile,
                "runtime_provenance": self.runtime_provenance,
                "last_demo": self.last_demo,
                "replaying": self.replaying,
                "expert_demonstration": self.expert_demonstration,
                "scenario_id": self.scenario_id,
                "scenario_seed": self.scenario_seed,
                "scenario_title": SCENARIOS_BY_ID[self.scenario_id]["title"],
                "autonomy_mode": self.autonomy_mode,
                "intervention_count": self.intervention_count,
                "coaching_cue": self.coaching_cue,
                "evaluation_status": self.evaluation_status,
                "evaluation_source": self.evaluation_source,
                "evaluation_output": self.evaluation_output,
                "reference_ghost": {
                    "enabled": self.reference_ghost_enabled,
                    "reference": self.reference_ghost_demo,
                    "point_count": self.reference_ghost_points,
                },
                "safety": {
                    "max_contact_force_n": self.max_contact_force_n,
                    "max_tissue_displacement_m": self.max_tissue_displacement_m,
                    "max_tissue_deformation_proxy": self.max_tissue_deformation_proxy,
                    "max_tissue_stress_pa": self.max_tissue_stress_pa,
                },
                "native_telemetry": self.native_telemetry,
                "stapler_test_cell": dict(self.stapler_test_cell),
                "skin_adhesive_system": dict(self.skin_adhesive_system),
                "closure_robot_system": dict(self.closure_robot_system),
                "dr_anmar_needle_domain": self.dr_anmar_needle_domain,
                "sensor_quality": {
                    "valid_depth_fraction": self.camera_valid_depth_fraction,
                    "semantic_foreground_fraction": self.camera_foreground_fraction,
                    "mean_luminance": self.camera_mean_luminance,
                },
                "render_contract": render_contract,
                "native_scene_contract": {
                    "rigid_objects": self.native_rigid_object_names,
                    "deformable_objects": self.native_deformable_object_names,
                    "rigid_object_positions_m": self.native_rigid_object_positions_m,
                    "tool_positions_m": self.native_tool_positions_m,
                    "ring_physics_ready": self.ring_physics_ready,
                    "strand_self_collision_ready": self.strand_self_collision_ready,
                    "bimanual_ready": self.arms == 2 and self.action_dim == 14,
                },
                "operator_study": {
                    "gaze_valid": self.gaze_valid,
                    "gaze_source": self.gaze_source,
                    "input_source": self.operator_input_source,
                    "procedure_phase": self.procedure_phase,
                    "annotation_count": len(self.procedure_events),
                },
                "drive_active": (
                    self.drive_until > time.monotonic() and bool(np.any(self.drive))
                ) or self.expert_demonstration.get("status") == "running",
                "control_contract": {
                    "sequence": self.control_sequence,
                    "kind": self.control_last_kind,
                    "source": self.control_last_source,
                    "action": list(self.control_last_action),
                    "last_nonzero_sequence": self.control_last_nonzero_sequence,
                    "last_nonzero_action": list(self.control_last_nonzero_action),
                    "minimum_steps_remaining": self.drive_min_steps_remaining,
                    "stop_pending": self.drive_stop_pending,
                    "age_ms": round(max(0.0, time.monotonic() - self.control_last_at) * 1000)
                    if self.control_last_at
                    else None,
                },
            }

    def live_status(self) -> dict[str, Any]:
        """Return only the rapidly changing fields consumed by the workstation UI."""
        with self.lock:
            procedure = self._procedure_status()
            steps = [
                {
                    "title": item.get("title", ""),
                    "instruction": item.get("instruction", ""),
                    "status": item.get("status", "pending"),
                }
                for item in procedure.get("steps", [])
            ]
            expert = dict(self.expert_demonstration)
            expert["phases"] = [dict(item) for item in self.expert_demonstration.get("phases", [])]
            return {
                "instance_id": self.instance_id,
                "camera_names": list(self.camera_names),
                "arms": self.arms,
                "has_grippers": self.has_grippers,
                "procedure": {
                    "title": procedure.get("title", "Free practice"),
                    "objective": procedure.get("objective", "Use the robot controls to explore the digital twin."),
                    "progress_percent": procedure.get("progress_percent", 0),
                    "steps": steps,
                },
                "grippers_open": list(self.grippers_open),
                "gripper_apertures": list(self.gripper_apertures),
                "hand_teleop": self.hand_teleop_snapshot(),
                "native_grasp_contact_active": list(self.native_grasp_contact_active),
                "tool_to_object_distance_m": list(self.tool_to_object_distance_m),
                "tool_to_object_offset_m": [list(value) if value is not None else None for value in self.tool_to_object_offset_m],
                "grasp_capture_radius_m": self.grasp_capture_radius_m,
                "camera_view_mode": self.camera_view_mode,
                "camera_adjustable": self.camera_adjustment(),
                "camera_adjustable_by_name": self.camera_adjustments(),
                "closest_anatomy_clearance_m": self.closest_anatomy_clearance_m,
                "recording": self.recording,
                "last_demo": self.last_demo,
                "expert_demonstration": expert,
                "autonomy_mode": self.autonomy_mode,
                "coaching_cue": self.coaching_cue,
                "safety": {
                    "max_contact_force_n": self.max_contact_force_n,
                    "max_tissue_displacement_m": self.max_tissue_displacement_m,
                    "max_tissue_stress_pa": self.max_tissue_stress_pa,
                },
                "stapler_test_cell": dict(self.stapler_test_cell),
                "skin_adhesive_system": dict(self.skin_adhesive_system),
                "closure_robot_system": dict(self.closure_robot_system),
            }

    def _procedure_status(self) -> dict[str, Any]:
        if not self.procedure:
            return {}
        now = time.monotonic()
        kind = self.procedure.get("guide_kind")
        step_count = len(self.procedure.get("steps", []))
        if kind == "softmimicgen_threading":
            # Only NVIDIA's published ring-crossing predicate may complete the
            # upstream task. In the extended bimanual room the expert's native
            # custody state separately confirms the handoff.
            if self.procedure.get("bimanual"):
                completed = 3 if self.upstream_task_success is True else 0
                if completed and self.expert_demonstration.get("ring_handoff_complete"):
                    completed = 4
            else:
                completed = step_count if self.upstream_task_success is True else 0
            if not completed:
                completed += int(self.procedure_motion_seen)
                completed += int(self.procedure_grasp_seen)
                completed = min(completed, max(0, step_count - 1))
        elif kind == "stapler_test_cell":
            cell = self.stapler_test_cell
            completed = int(bool(cell.get("enabled")))
            completed += int(
                float(
                    cell.get("tissue_max_displacement_mm", 0.0)
                )
                >= 0.5
            )
            completed += int(
                int(cell.get("retained_attachment_count", 0)) >= 1
            )
            completed += int(
                bool(cell.get("closure_complete"))
            )
            completed += int(
                bool(cell.get("closure_complete"))
                and not bool(cell.get("cycle_running"))
            )
        elif kind == "dynamic_abdominal_patient":
            completed = int(self.camera_nonblank_seen)
            contact_states = (
                self.native_telemetry
                .get("dynamic_patient_effects", {})
                .get("states", [])
            )
            completed += int(
                any(
                    float(item.get("retraction_fraction", 0.0)) > 0.02
                    for item in contact_states
                )
            )
        elif kind == "autonomous_rescue_or":
            completed = int(self.camera_nonblank_seen)
            rescue = self.native_telemetry.get(
                "autonomous_rescue_or",
                {},
            )
            measured = rescue.get("measured_contact", {})
            vessel = rescue.get("vessel", {})
            completed += int(
                min(
                    float(measured.get("left_normal_force_n", 0.0)),
                    float(measured.get("right_normal_force_n", 0.0)),
                )
                > 0.12
            )
            completed += int(
                float(
                    vessel.get(
                        "transient_compression_fraction",
                        0.0,
                    )
                )
                > 0.1
            )
            completed += int(
                bool(rescue.get("release_observed", False))
            )
        elif kind == "native_suturing_bench":
            # This dry-lab room deliberately avoids synthetic completion
            # predicates.  Only an observed physical grasp and subsequent
            # rigid-body transport advance the first two phases; regrasp,
            # handoff, scissors exchange and recovery remain clinician-led
            # until native per-object custody predicates are available.
            completed = int(self.procedure_grasp_seen)
            completed += int(
                self.procedure_grasp_seen
                and self.procedure_motion_seen
                and self.procedure_object_motion_m >= 0.018
            )
        elif kind == "needle_pass":
            completed = int(self.procedure_motion_seen)
            completed += int(self.procedure_waypoints_completed >= 2)
            completed += int(self.procedure_grasp_seen)
            completed += int(self.procedure_object_motion_m >= 0.018)
            if completed >= max(1, step_count - 1) and now - self.procedure_last_motion_at > 0.8:
                completed = step_count
        elif kind == "navigation":
            completed = int(self.procedure_motion_seen)
            if self.procedure_waypoints_total and self.procedure_waypoints_completed:
                completed = max(
                    completed,
                    1 + round(3 * self.procedure_waypoints_completed / self.procedure_waypoints_total),
                )
            if self.procedure_waypoints_completed >= self.procedure_waypoints_total > 0 and now - self.procedure_last_motion_at > 0.8:
                completed = step_count
        else:
            completed = int(self.procedure_motion_seen)
            completed += int(self.procedure_grasp_seen)
            completed += int(self.procedure_object_lift_m >= 0.008)
            completed += int(self.procedure_object_motion_m >= 0.018)
            if completed >= max(1, step_count - 1) and now - self.procedure_last_motion_at > 0.8:
                completed = step_count
        completed = min(completed, step_count)
        steps = []
        for index, item in enumerate(self.procedure.get("steps", [])):
            status = "complete" if index < completed else "active" if index == completed else "pending"
            steps.append({**item, "status": status})
        return {
            **self.procedure,
            "steps": steps,
            "completed_steps": completed,
            "step_count": step_count,
            "progress_percent": round(100 * completed / step_count) if step_count else 0,
            "waypoints_completed": self.procedure_waypoints_completed,
            "waypoints_total": self.procedure_waypoints_total,
            "object_lift_m": round(self.procedure_object_lift_m, 4),
            "object_motion_m": round(self.procedure_object_motion_m, 4),
            "upstream_task_success": self.upstream_task_success,
            "native_telemetry": self.native_telemetry,
        }


def process_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def build_web_app(state: SharedState) -> FastAPI:
    app = FastAPI(title="Dr.Anmar Surgical Workstation", docs_url=None, redoc_url=None)
    operator_lease = OperatorLease()
    hand_watchdog_stop = threading.Event()
    hand_watchdog_thread: threading.Thread | None = None

    def run_hand_watchdog() -> None:
        while not hand_watchdog_stop.wait(0.010):
            with state.lock:
                expired = state.hand_teleop.expire_stale()
            if expired:
                state.wake_event.set()

    @app.on_event("startup")
    def start_hand_watchdog() -> None:
        nonlocal hand_watchdog_thread
        hand_watchdog_stop.clear()
        hand_watchdog_thread = threading.Thread(
            target=run_hand_watchdog,
            daemon=True,
            name="dr-anmar-hand-watchdog",
        )
        hand_watchdog_thread.start()

    @app.on_event("shutdown")
    def stop_hand_watchdog() -> None:
        hand_watchdog_stop.set()
        if hand_watchdog_thread is not None:
            hand_watchdog_thread.join(timeout=1.0)

    @app.middleware("http")
    async def protect_browser_requests(request: Request, call_next):
        if not access_is_authorized(request.cookies.get(ACCESS_COOKIE)):
            return JSONResponse({"detail": "Dr.Anmar access token required"}, status_code=401)
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            try:
                from urllib.parse import urlparse

                if urlparse(origin).hostname != request.url.hostname:
                    return JSONResponse({"detail": "Cross-site state changes are not allowed"}, status_code=403)
            except ValueError:
                return JSONResponse({"detail": "Invalid request origin"}, status_code=403)
            claimed, detail = operator_lease.claim(request.headers.get(OPERATOR_HEADER))
            if not claimed:
                return JSONResponse({"detail": detail}, status_code=423)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Resource-Policy"] = "same-site"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(self)"
        return response

    @app.get("/hand-control-assets/{asset_path:path}")
    def hand_control_asset(asset_path: str) -> FileResponse:
        if asset_path not in HAND_CONTROL_ASSET_FILES:
            raise HTTPException(404, "Unknown hand-control asset")
        path = (HAND_CONTROL_ASSET_ROOT / asset_path).resolve()
        if HAND_CONTROL_ASSET_ROOT.resolve() not in path.parents or not path.is_file():
            raise HTTPException(
                503,
                "Pinned hand-control assets are not installed on this workstation",
            )
        return FileResponse(path)

    @app.get("/hand-control.mjs")
    def hand_control_client() -> FileResponse:
        return FileResponse(HAND_CONTROL_CLIENT_PATH, media_type="text/javascript")

    @app.get("/hand-control-worker.mjs")
    def hand_control_worker() -> FileResponse:
        return FileResponse(HAND_CONTROL_WORKER_PATH, media_type="text/javascript")

    @app.get("/api/hand-control/assets")
    def hand_control_asset_status() -> dict[str, Any]:
        present = sorted(
            asset
            for asset in HAND_CONTROL_ASSET_FILES
            if (HAND_CONTROL_ASSET_ROOT / asset).is_file()
        )
        return {
            "ready": len(present) == len(HAND_CONTROL_ASSET_FILES),
            "version": "0.10.35",
            "present": present,
            "required": sorted(HAND_CONTROL_ASSET_FILES),
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return APP_HTML

    @app.get("/api/status")
    def status() -> JSONResponse:
        return JSONResponse({**state.status(), "operator_lease": operator_lease.status()})

    @app.get("/api/status/live")
    def live_status() -> JSONResponse:
        return JSONResponse(state.live_status())

    @app.get("/api/health/runtime")
    def runtime_health() -> JSONResponse:
        rss_bytes = process_rss_bytes()
        cuda_allocated = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        cuda_reserved = int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
        try:
            open_descriptors = len(list(Path("/proc/self/fd").iterdir()))
        except OSError:
            open_descriptors = None
        with state.lock:
            payload = {
                "schema": "dr.anmar.runtime-health.v1",
                "instance_id": state.instance_id,
                "rss_bytes": rss_bytes,
                "rss_warning_bytes": MEMORY_WARNING_BYTES,
                "rss_warning": rss_bytes is not None and rss_bytes >= MEMORY_WARNING_BYTES,
                "cuda_allocated_bytes": cuda_allocated,
                "cuda_reserved_bytes": cuda_reserved,
                "thread_count": threading.active_count(),
                "open_descriptors": open_descriptors,
                "active_camera_streams": sum(1 for count in state.camera_subscribers.values() if count > 0),
                "recording": state.recording,
                "recording_frames": state.recorded_frames,
                "recording_payload_bytes": state.recorded_bytes_estimate,
                "recording_queue_depth": state.recording_queue_depth,
                "recording_buffered_frames": state.recording_buffered_frames,
                "recording_queue_capacity": BoundedCaptureSpool.MAX_QUEUED_BATCHES,
                "jpeg_queue_depth": state.jpeg_queue_depth,
                "jpeg_queue_capacity": BoundedJpegEncoder.MAX_QUEUED_JOBS,
                "jpeg_frames_dropped": state.jpeg_frames_dropped,
                "sim_fps": state.sim_fps,
                "render_fps": state.render_fps,
                "control_sequence": state.control_sequence,
                "control_last_kind": state.control_last_kind,
                "control_last_source": state.control_last_source,
                "control_last_nonzero_sequence": state.control_last_nonzero_sequence,
            }
        return JSONResponse(payload)

    @app.post("/api/operator/heartbeat")
    def operator_heartbeat() -> dict[str, Any]:
        return {"ok": True, "operator_lease": operator_lease.status()}

    @app.post("/api/operator/release")
    def operator_release(request: Request) -> dict[str, Any]:
        return {"ok": operator_lease.release(request.headers.get(OPERATOR_HEADER))}

    @app.post("/api/jog")
    def jog(request: JogRequest) -> dict[str, Any]:
        if request.axis not in range(6) or request.direction not in (-1, 1):
            raise HTTPException(400, "axis must be 0–5 and direction must be -1 or 1")
        if request.arm not in range(state.arms):
            raise HTTPException(400, f"arm must be between 0 and {state.arms - 1}")
        command = np.zeros(state.action_dim, dtype=np.float32)
        body_slice = state.body_action_slice(request.arm)
        command[body_slice.start + request.axis] = 0.5 * request.direction
        with state.lock:
            state.disable_hand_motion()
            state.pulse = command
            state.pulse_steps = 1
            state.note_control("jog", "keyboard_pointer", command)
        state.wake_event.set()
        return {"ok": True, "action": command.tolist()}

    @app.post("/api/drive")
    def drive(request: DriveRequest) -> dict[str, Any]:
        if request.arm not in range(state.arms):
            raise HTTPException(400, f"arm must be between 0 and {state.arms - 1}")
        if len(request.values) != 6:
            raise HTTPException(400, "drive values must contain six axes")
        values = np.asarray(request.values, dtype=np.float32)
        if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0):
            raise HTTPException(400, "drive axes must be finite values between -1 and 1")
        if not 0.25 <= request.speed <= 2.0:
            raise HTTPException(400, "speed must be between 0.25 and 2.0")
        if request.source not in OPERATOR_INPUT_SOURCES:
            raise HTTPException(400, "Unknown operator input source")
        with state.lock:
            state.disable_hand_motion()
            scenario_id = state.scenario_id
        profile = SCENARIO_NATIVE_PROFILES.get(scenario_id, {})
        translation = values[:3].copy()
        yaw_degrees = float(profile.get("translation_yaw_deg", 0.0))
        if yaw_degrees:
            radians = np.deg2rad(yaw_degrees)
            cosine, sine = np.cos(radians), np.sin(radians)
            translation[:2] = np.asarray(
                (cosine * translation[0] - sine * translation[1], sine * translation[0] + cosine * translation[1]),
                dtype=np.float32,
            )
        translation *= np.asarray(profile.get("axis_scale", (1.0, 1.0, 1.0)), dtype=np.float32)
        calibrated_values = values.copy()
        calibrated_values[:3] = translation
        command = np.zeros(state.action_dim, dtype=np.float32)
        with state.lock:
            semantic_far_field = (
                request.source in {"keyboard_smart_action", "gamepad_smart_action", "voice_smart_action"}
                and state.native_grasp_contact_active[request.arm]
                and state.needle_tip_clearance_m is not None
                and state.needle_tip_clearance_m > 0.020
            )
            semantic_target_far = (
                request.source in {"keyboard_smart_action", "gamepad_smart_action", "voice_smart_action"}
                and not state.native_grasp_contact_active[request.arm]
                and state.tool_to_object_distance_m[request.arm] is not None
                and state.tool_to_object_distance_m[request.arm] > 0.050
            )
        translation_boost = 6.0 if semantic_far_field else 3.0 if semantic_target_far else 1.0
        calibrated_values[:3] *= translation_boost
        command[state.body_action_slice(request.arm)] = np.clip(
            calibrated_values * request.speed, -1.0, 1.0
        )
        active = bool(np.any(values))
        with state.lock:
            # Keep a command alive long enough for at least one slow Isaac step.
            # The workstation normally refreshes held keys continuously, but
            # bounded keyboard macros intentionally send one command followed
            # by an explicit stop. A fixed 300 ms expiry could disappear
            # between frames when a photorealistic scene renders near 2 Hz.
            hold_seconds = max(0.30, min(1.25, 1.4 / max(state.sim_fps, 1.0)))
            if active:
                state.drive = command
                state.drive_min_steps_remaining = max(state.drive_min_steps_remaining, 1)
                state.drive_stop_pending = False
            elif state.drive_min_steps_remaining > 0 and bool(np.any(state.drive)):
                state.drive_stop_pending = True
            else:
                state.drive = command
                state.drive_stop_pending = False
            state.operator_input_source = request.source
            state.drive_until = time.monotonic() + hold_seconds if active else 0.0
            state.note_control("drive", request.source, command)
            if active:
                if state.expert_demonstration.get("status") in {"running", "paused"}:
                    state.expert_request = "take_over"
                    state.expert_clean_run = False
                if state.replaying or state.autonomy_mode == "supervised_replay":
                    state.intervention_count += 1
                    state.replaying = False
                state.autonomy_mode = "manual" if state.autonomy_mode == "supervised_replay" else state.autonomy_mode
                state.replay_request = "stop"
                magnitude = float(np.linalg.norm(values[:3]))
                angular = float(np.linalg.norm(values[3:]))
                if state.autonomy_mode == "guided" and request.speed > 1.2 and magnitude > 0.65:
                    state.coaching_cue = "Fast translation detected. Use Precision near the needle or tissue."
                elif state.autonomy_mode == "guided" and magnitude > 0.55 and angular > 0.55:
                    state.coaching_cue = "Position and angle are changing together. Separate them for a more readable demonstration."
                elif state.autonomy_mode == "guided":
                    state.coaching_cue = "Motion is being captured. Approach deliberately and include a stable recovery."
        state.wake_event.set()
        return {
            "ok": True,
            "active": active,
            "action": command.tolist(),
            "expires_ms": round(hold_seconds * 1000),
        }

    @app.post("/api/drive/bimanual")
    def drive_bimanual(request: BimanualDriveRequest) -> dict[str, Any]:
        """Apply both hand-held keyboard commands as one atomic simulator action."""
        if request.source not in OPERATOR_INPUT_SOURCES:
            raise HTTPException(400, "Unknown operator input source")
        if not request.commands or len(request.commands) > state.arms:
            raise HTTPException(400, f"commands must contain between 1 and {state.arms} instruments")
        command = np.zeros(state.action_dim, dtype=np.float32)
        seen_arms: set[int] = set()
        with state.lock:
            profile = SCENARIO_NATIVE_PROFILES.get(state.scenario_id, {})
        yaw_degrees = float(profile.get("translation_yaw_deg", 0.0))
        axis_scale = np.asarray(profile.get("axis_scale", (1.0, 1.0, 1.0)), dtype=np.float32)
        active = False
        for arm_command in request.commands:
            if arm_command.arm not in range(state.arms) or arm_command.arm in seen_arms:
                raise HTTPException(400, "each available instrument may appear once")
            if len(arm_command.values) != 6:
                raise HTTPException(400, "every bimanual drive command must contain six axes")
            values = np.asarray(arm_command.values, dtype=np.float32)
            if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0):
                raise HTTPException(400, "drive axes must be finite values between -1 and 1")
            if not 0.25 <= arm_command.speed <= 2.0:
                raise HTTPException(400, "speed must be between 0.25 and 2.0")
            calibrated_values = values.copy()
            translation = values[:3].copy()
            if yaw_degrees:
                radians = np.deg2rad(yaw_degrees)
                cosine, sine = np.cos(radians), np.sin(radians)
                translation[:2] = np.asarray(
                    (cosine * translation[0] - sine * translation[1], sine * translation[0] + cosine * translation[1]),
                    dtype=np.float32,
                )
            calibrated_values[:3] = translation * axis_scale
            command[state.body_action_slice(arm_command.arm)] = np.clip(
                calibrated_values * arm_command.speed, -1.0, 1.0
            )
            active = active or bool(np.any(values))
            seen_arms.add(arm_command.arm)

        with state.lock:
            state.disable_hand_motion()
            hold_seconds = max(0.30, min(1.25, 1.4 / max(state.sim_fps, 1.0)))
            if active:
                state.drive = command
                state.drive_min_steps_remaining = max(state.drive_min_steps_remaining, 1)
                state.drive_stop_pending = False
            elif state.drive_min_steps_remaining > 0 and bool(np.any(state.drive)):
                state.drive_stop_pending = True
            else:
                state.drive = command
                state.drive_stop_pending = False
            state.drive_until = time.monotonic() + hold_seconds if active else 0.0
            state.operator_input_source = request.source
            state.note_control("bimanual", request.source, command)
            if active:
                if state.expert_demonstration.get("status") in {"running", "paused"}:
                    state.expert_request = "take_over"
                    state.expert_clean_run = False
                if state.replaying or state.autonomy_mode == "supervised_replay":
                    state.intervention_count += 1
                    state.replaying = False
                state.autonomy_mode = "manual" if state.autonomy_mode == "supervised_replay" else state.autonomy_mode
                state.replay_request = "stop"
                if state.autonomy_mode == "guided":
                    state.coaching_cue = "Bimanual motion is being captured. Coordinate both instruments deliberately."
        state.wake_event.set()
        return {
            "ok": True,
            "active": active,
            "arms": sorted(seen_arms),
            "action": command.tolist(),
            "expires_ms": round(hold_seconds * 1000),
        }

    @app.post("/api/teleop/hands")
    def teleop_hands(request: HandTeleopRequest, http_request: Request) -> dict[str, Any]:
        """Accept one complete, calibrated bimanual master-pose frame."""

        claimed, detail = operator_lease.claim(http_request.headers.get(OPERATOR_HEADER))
        if not claimed:
            raise HTTPException(423, detail)
        if not state.native_psm_policy_contract or len(state.native_ik_scales) != state.arms:
            raise HTTPException(409, "Webcam hand control requires the native NVIDIA PSM IK room")
        raw_hands = [
            {
                "arm": hand.arm,
                "tracked": hand.tracked,
                "motion_engaged": hand.motion_engaged,
                "translation_offset_m": hand.translation_offset_m,
                "rotation_vector_rad": hand.rotation_vector_rad,
                "aperture_normalized": hand.aperture_normalized,
                "confidence": hand.confidence,
            }
            for hand in request.hands
        ]
        try:
            sequence, hands = validate_hand_frame(
                request.sequence,
                raw_hands,
                arms=state.arms,
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error
        if request.transport_drops < 0:
            raise HTTPException(400, "transport_drops must be non-negative")
        for value, label, maximum in (
            (request.captured_at_ms, "captured_at_ms", None),
            (request.client_sent_at_ms, "client_sent_at_ms", None),
            (request.inference_ms, "inference_ms", 2000.0),
        ):
            if value is None:
                continue
            if not math.isfinite(float(value)) or (maximum is not None and not 0.0 <= float(value) <= maximum):
                raise HTTPException(400, f"{label} is outside its valid range")
        command = np.zeros(state.action_dim, dtype=np.float32)
        now = time.monotonic()
        wall_now_ms = time.time() * 1000.0
        transport_age_ms = (
            max(0.0, wall_now_ms - float(request.client_sent_at_ms))
            if request.client_sent_at_ms is not None
            and abs(wall_now_ms - float(request.client_sent_at_ms)) <= 60_000.0
            else None
        )
        capture_age_ms = (
            max(0.0, wall_now_ms - float(request.captured_at_ms))
            if request.captured_at_ms is not None
            and abs(wall_now_ms - float(request.captured_at_ms)) <= 60_000.0
            else None
        )
        with state.lock:
            if len(state.hand_camera_to_action_basis) != state.arms:
                raise HTTPException(
                    409,
                    "The live endoscope-to-instrument control frame is not ready",
                )
            action_hands = []
            for hand in hands:
                transformed = camera_pose_to_action_frame(
                    hand,
                    state.hand_camera_to_action_basis[hand["arm"]],
                )
                transformed["translation_offset_m"] = np.clip(
                    np.asarray(
                        transformed["translation_offset_m"],
                        dtype=np.float64,
                    ),
                    -0.12,
                    0.12,
                ).tolist()
                transformed["rotation_vector_rad"] = np.clip(
                    np.asarray(
                        transformed["rotation_vector_rad"],
                        dtype=np.float64,
                    ),
                    -0.8,
                    0.8,
                ).tolist()
                action_hands.append(transformed)
            try:
                state.hand_teleop.submit(sequence, action_hands, now=now)
            except ValueError as error:
                raise HTTPException(409, str(error)) from error
            state.hand_last_received_sequence = sequence
            state.hand_last_received_at = now
            state.hand_latest_transport_age_ms = (
                round(transport_age_ms, 2)
                if transport_age_ms is not None
                else None
            )
            state.hand_latest_capture_age_ms = (
                round(capture_age_ms, 2)
                if capture_age_ms is not None
                else None
            )
            state.hand_latest_inference_ms = (
                round(float(request.inference_ms), 2)
                if request.inference_ms is not None
                else None
            )
            state.hand_transport_drops = request.transport_drops
            for hand in action_hands:
                arm = hand["arm"]
                arm_state = state.hand_teleop.arm_states[arm]
                if arm_state.tracked and state.has_grippers and state.hand_teleop.enabled:
                    aperture = arm_state.aperture_normalized
                    state.gripper_apertures[arm] = aperture
                    state.grippers_open[arm] = aperture >= 0.5
                    command[state.gripper_action_index(arm)] = proportional_gripper_action(aperture)
                if arm_state.tracked and arm_state.motion_engaged:
                    offsets = arm_state.target_offset
                    scales = state.native_ik_scales[arm]
                    command[state.body_action_slice(arm)] = np.clip(
                        np.asarray(offsets, dtype=np.float32)
                        / np.asarray(scales, dtype=np.float32),
                        -1.0,
                        1.0,
                    )
            state.operator_input_source = "webcam_hands"
            state.note_control("webcam_hands", "webcam_hands", command)
            snapshot = state.hand_teleop_snapshot(now=now)
        state.wake_event.set()
        return {"ok": True, "hand_teleop": snapshot}

    @app.post("/api/teleop/hands/control")
    def teleop_hands_control(
        request: HandTeleopControlRequest,
        http_request: Request,
    ) -> dict[str, Any]:
        claimed, detail = operator_lease.claim(http_request.headers.get(OPERATOR_HEADER))
        if not claimed:
            raise HTTPException(423, detail)
        if not state.native_psm_policy_contract:
            raise HTTPException(409, "Webcam hand control requires the native NVIDIA PSM IK room")
        reason = request.reason.strip()[:64] or "operator"
        with state.lock:
            if request.enabled:
                # Webcam authority is exclusive. Clear every buffered manual
                # command and stop replay/expert motion before arming the hand
                # controller so releasing its clutch cannot reveal an older
                # command underneath.
                state.pulse.fill(0.0)
                state.pulse_steps = 0
                state.drive.fill(0.0)
                state.drive_until = 0.0
                state.drive_min_steps_remaining = 0
                state.drive_stop_pending = False
                if state.expert_demonstration.get("status") in {"running", "paused"}:
                    state.expert_request = "take_over"
                    state.expert_clean_run = False
                if state.replaying or state.autonomy_mode == "supervised_replay":
                    state.intervention_count += 1
                    state.replaying = False
                    state.replay_request = "stop"
                    if state.autonomy_mode == "supervised_replay":
                        state.autonomy_mode = "manual"
                state.hand_teleop.enable_motion()
                state.operator_input_source = "webcam_hands"
                state.hand_authority_reason = f"enabled:{reason}"
                message = "Hand control enabled; send an unclutched frame before engaging motion"
            else:
                state.hand_teleop.disable_motion()
                state.hand_authority_reason = f"frozen:{reason}"
                message = "Hand control frozen"
            state.note_control(
                "webcam_hands_enable" if request.enabled else "webcam_hands_disable",
                "webcam_hands",
                np.zeros(state.action_dim, dtype=np.float32),
            )
            snapshot = state.hand_teleop_snapshot()
        state.wake_event.set()
        return {"ok": True, "message": message, "hand_teleop": snapshot}

    @app.post("/api/stop")
    def stop(request: StopRequest = StopRequest()) -> dict[str, bool]:
        if request.source not in OPERATOR_INPUT_SOURCES:
            raise HTTPException(400, "Unknown operator input source")
        with state.lock:
            state.disable_hand_motion()
            state.pulse.fill(0.0)
            state.pulse_steps = 0
            if state.drive_min_steps_remaining > 0 and bool(np.any(state.drive)):
                state.drive_stop_pending = True
                state.drive_until = 0.0
            else:
                state.drive.fill(0.0)
                state.drive_until = 0.0
                state.drive_stop_pending = False
            state.replay_request = "stop"
            state.note_control("stop", request.source, np.zeros(state.action_dim, dtype=np.float32))
            if state.expert_demonstration.get("status") in {"running", "paused"}:
                state.expert_request = "take_over"
                state.expert_clean_run = False
        state.wake_event.set()
        return {"ok": True}

    @app.post("/api/camera-view")
    def camera_view(request: CameraViewRequest) -> dict[str, Any]:
        camera_modes = {
            "operative",
            "close",
            "overview",
            "overhead",
            "left_oblique",
            "right_oblique",
            "opposite",
        }
        if request.mode not in camera_modes:
            raise HTTPException(400, f"camera view must be one of: {', '.join(sorted(camera_modes))}")
        with state.lock:
            state.disable_hand_motion()
            state.camera_view_mode = request.mode
            state.camera_view_request = request.mode
            state.camera_free_enabled = False
            state.camera_free_base_mode = request.mode
            state.camera_free_yaw_deg = 0.0
            state.camera_free_pitch_deg = 0.0
            state.camera_free_zoom = 1.0
            state.camera_free_pan_x_m = 0.0
            state.camera_free_pan_y_m = 0.0
        state.wake_event.set()
        return {"ok": True, "mode": request.mode}

    @app.post("/api/camera-adjust")
    def camera_adjust(request: CameraAdjustRequest) -> dict[str, Any]:
        def finite_delta(value: float, limit: float) -> float:
            numeric = float(value)
            if not np.isfinite(numeric):
                raise HTTPException(400, "camera adjustments must be finite numbers")
            return float(np.clip(numeric, -limit, limit))

        requested_camera = str(request.camera_name or "endoscope_left")
        if requested_camera.startswith("endoscope_"):
            adjustment_camera = "endoscope_left"
        elif requested_camera.startswith("wrist_") and requested_camera in state.camera_names:
            adjustment_camera = requested_camera
        else:
            raise HTTPException(400, "Only the adjustable camera and gripper cameras can be aimed")

        with state.lock:
            state.disable_hand_motion()
            if adjustment_camera.startswith("wrist_"):
                adjustment = state.camera_adjustment(adjustment_camera)
                if request.reset:
                    adjustment.update(
                        {
                            "yaw_deg": 0.0,
                            "pitch_deg": 0.0,
                            "zoom": 1.0,
                            "pan_x_m": 0.0,
                            "pan_y_m": 0.0,
                        }
                    )
                adjustment["yaw_deg"] = float(
                    np.clip(
                        float(adjustment["yaw_deg"]) + finite_delta(request.orbit_yaw_delta_deg, 45.0),
                        -65.0,
                        65.0,
                    )
                )
                adjustment["pitch_deg"] = float(
                    np.clip(
                        float(adjustment["pitch_deg"]) + finite_delta(request.orbit_pitch_delta_deg, 30.0),
                        -55.0,
                        55.0,
                    )
                )
                adjustment["zoom"] = float(
                    np.clip(float(adjustment["zoom"]) + finite_delta(request.zoom_delta, 0.30), 0.55, 1.75)
                )
                adjustment["pan_x_m"] = float(
                    np.clip(
                        float(adjustment["pan_x_m"]) + finite_delta(request.pan_x_delta_m, 0.02),
                        -0.025,
                        0.025,
                    )
                )
                adjustment["pan_y_m"] = float(
                    np.clip(
                        float(adjustment["pan_y_m"]) + finite_delta(request.pan_y_delta_m, 0.02),
                        -0.025,
                        0.025,
                    )
                )
                adjustment["enabled"] = bool(request.enabled)
                state.gripper_camera_adjustments[adjustment_camera] = {
                    key: value for key, value in adjustment.items() if key != "camera_name"
                }
                result = {"mode": "tool_axis", **state.camera_adjustment(adjustment_camera)}
            elif not request.enabled:
                state.camera_free_enabled = False
                state.camera_view_mode = state.camera_free_base_mode
                state.camera_view_request = state.camera_free_base_mode
                result = {"mode": state.camera_view_mode, **state.camera_adjustment(adjustment_camera)}
            else:
                if not state.camera_free_enabled and state.camera_view_mode != "free":
                    state.camera_free_base_mode = state.camera_view_mode
                if request.reset:
                    state.camera_free_yaw_deg = 0.0
                    state.camera_free_pitch_deg = 0.0
                    state.camera_free_zoom = 1.0
                    state.camera_free_pan_x_m = 0.0
                    state.camera_free_pan_y_m = 0.0
                state.camera_free_yaw_deg = (
                    state.camera_free_yaw_deg + finite_delta(request.orbit_yaw_delta_deg, 45.0) + 180.0
                ) % 360.0 - 180.0
                state.camera_free_pitch_deg = float(
                    np.clip(
                        state.camera_free_pitch_deg + finite_delta(request.orbit_pitch_delta_deg, 30.0),
                        -75.0,
                        75.0,
                    )
                )
                state.camera_free_zoom = float(
                    np.clip(state.camera_free_zoom + finite_delta(request.zoom_delta, 0.35), 0.35, 2.5)
                )
                state.camera_free_pan_x_m = float(
                    np.clip(state.camera_free_pan_x_m + finite_delta(request.pan_x_delta_m, 0.1), -0.45, 0.45)
                )
                state.camera_free_pan_y_m = float(
                    np.clip(state.camera_free_pan_y_m + finite_delta(request.pan_y_delta_m, 0.1), -0.45, 0.45)
                )
                state.camera_free_enabled = True
                state.camera_view_mode = "free"
                state.camera_view_request = "free"
                result = {"mode": "free", **state.camera_adjustment(adjustment_camera)}
        state.wake_event.set()
        return {"ok": True, **result}

    @app.post("/api/gripper")
    def gripper(request: GripperRequest) -> dict[str, Any]:
        if not state.has_grippers:
            raise HTTPException(409, "This robot has no gripper action")
        if request.arm not in range(state.arms):
            raise HTTPException(400, f"arm must be between 0 and {state.arms - 1}")
        if request.source not in OPERATOR_INPUT_SOURCES:
            raise HTTPException(400, "Unknown operator input source")
        with state.lock:
            state.disable_hand_motion()
            if state.expert_demonstration.get("status") in {"running", "paused"}:
                state.expert_request = "take_over"
                state.expert_clean_run = False
            state.grippers_open[request.arm] = request.open
            state.gripper_apertures[request.arm] = 1.0 if request.open else 0.0
            gripper_action = np.zeros(state.action_dim, dtype=np.float32)
            gripper_action[state.gripper_action_index(request.arm)] = 1.0 if request.open else -1.0
            state.note_control("gripper", request.source, gripper_action)
        state.wake_event.set()
        return {"ok": True, "open": request.open, "arm": request.arm}

    @app.post("/api/gripper/toggle")
    def toggle_gripper(request: GripperToggleRequest) -> dict[str, Any]:
        if not state.has_grippers:
            raise HTTPException(409, "This robot has no gripper action")
        if request.arm not in range(state.arms):
            raise HTTPException(400, f"arm must be between 0 and {state.arms - 1}")
        if request.source not in OPERATOR_INPUT_SOURCES:
            raise HTTPException(400, "Unknown operator input source")
        with state.lock:
            state.disable_hand_motion()
            if state.expert_demonstration.get("status") in {"running", "paused"}:
                state.expert_request = "take_over"
                state.expert_clean_run = False
            state.grippers_open[request.arm] = not state.grippers_open[request.arm]
            is_open = state.grippers_open[request.arm]
            state.gripper_apertures[request.arm] = 1.0 if is_open else 0.0
            gripper_action = np.zeros(state.action_dim, dtype=np.float32)
            gripper_action[state.gripper_action_index(request.arm)] = 1.0 if is_open else -1.0
            state.note_control("gripper", request.source, gripper_action)
        state.wake_event.set()
        return {"ok": True, "open": is_open, "arm": request.arm}

    @app.post("/api/reset")
    def reset() -> dict[str, bool]:
        with state.lock:
            state.disable_hand_motion()
            state.reset_requested = True
            state.drive.fill(0.0)
            state.drive_until = 0.0
            state.drive_min_steps_remaining = 0
            state.drive_stop_pending = False
            state.note_control("reset", "keyboard_pointer", state.drive)
            state.replay_request = "stop"
            state.expert_request = "cancel"
            state.expert_clean_run = False
            state.grippers_open = [True] * state.arms
            state.gripper_apertures = [1.0] * state.arms
            state.skin_adhesive_target = 0.0
            if state.closure_robot_system.get("enabled", False):
                state.closure_robot_command_request = "reset"
        state.wake_event.set()
        return {"ok": True}

    @app.post("/api/stapler/command")
    def stapler_command(request: StaplerCommandRequest) -> dict[str, Any]:
        action = request.action.strip().lower()
        with state.lock:
            if not state.stapler_test_cell.get("enabled", False):
                raise HTTPException(
                    409,
                    "Open the Dr.Anmar stapler test cell before commanding its fixture",
                )
            if action == "set_target":
                if request.target_deg is None:
                    raise HTTPException(
                        400,
                        "set_target requires target_deg",
                    )
                target = float(request.target_deg)
                if (
                    not np.isfinite(target)
                    or not 0.0 <= target < FIRE_THRESHOLD_DEG
                ):
                    raise HTTPException(
                        400,
                        f"manual target must stay below the "
                        f"{FIRE_THRESHOLD_DEG:.0f} degree deployment threshold; "
                        "use fire for a full cycle",
                    )
                state.stapler_manual_target_deg = target
                state.stapler_command_request = "manual"
                state.coaching_cue = (
                    f"Fixture target set to {target:.1f}°. "
                    "A partial stroke below 24° must not deploy."
                )
            elif action in {"fire", "release", "reset"}:
                if (
                    action == "fire"
                    and state.stapler_test_cell.get("station_state")
                    == "placed"
                ):
                    raise HTTPException(
                        409,
                        "This closure station already has a staple; choose an open station",
                    )
                if (
                    action == "fire"
                    and not state.stapler_test_cell.get(
                        "station_ready",
                        True,
                    )
                ):
                    raise HTTPException(
                        409,
                        "The indexing fixture is still settling at this station",
                    )
                if (
                    action == "fire"
                    and state.stapler_test_cell.get("closure_complete", False)
                ):
                    raise HTTPException(
                        409,
                        "All closure stations are complete; reset the closure to run again",
                    )
                state.stapler_command_request = action
                if action == "release":
                    state.stapler_manual_target_deg = 0.0
                state.coaching_cue = {
                    "fire": (
                        "Approximating both wound edges before firing. The "
                        "formed staple will retain the FEM tissue after "
                        "release, then the fixture will advance."
                    ),
                    "release": "Returning the actuator below the 8° rearm threshold.",
                    "reset": "Resetting the closure, fixture and magazine evidence.",
                }[action]
            elif action in {"previous_station", "next_station"}:
                if state.stapler_test_cell.get("cycle_running", False):
                    raise HTTPException(
                        409,
                        "Wait for the current staple cycle to release before indexing",
                    )
                if not state.stapler_test_cell.get("station_ready", True):
                    raise HTTPException(
                        409,
                        "Wait for the indexing fixture to settle",
                    )
                current_index = int(
                    state.stapler_test_cell.get("station_index", 1)
                ) - 1
                direction = -1 if action == "previous_station" else 1
                station_count = int(
                    state.stapler_test_cell.get("station_count", 1)
                )
                requested_index = max(
                    0,
                    min(station_count - 1, current_index + direction),
                )
                state.stapler_station_request = requested_index
                state.coaching_cue = (
                    f"Indexing the fixture to closure station "
                    f"{requested_index + 1} of {station_count}."
                )
            else:
                raise HTTPException(
                    400,
                    "action must be fire, release, reset, set_target, "
                    "previous_station, or next_station",
                )
            result = dict(state.stapler_test_cell)
        state.wake_event.set()
        return {"ok": True, "action": action, "test_cell": result}

    @app.post("/api/skin-adhesive/activation")
    def skin_adhesive_activation(
        request: SkinAdhesiveActivationRequest,
    ) -> dict[str, Any]:
        target = float(request.activation)
        if not np.isfinite(target) or not 0.0 <= target <= 1.0:
            raise HTTPException(400, "activation must be between 0 and 1")
        with state.lock:
            if not state.skin_adhesive_system.get("enabled", False):
                raise HTTPException(
                    409,
                    "Add the Dr.Anmar topical skin adhesive system to this bench first",
                )
            mounted_arm = (
                int(state.skin_adhesive_system.get("mounted_arm", 1)) - 1
            )
            if mounted_arm not in range(state.arms):
                raise HTTPException(
                    409,
                    "The mounted adhesive end effector has no active robot arm",
                )
            state.skin_adhesive_target = target
            aperture = 1.0 - target
            state.gripper_apertures[mounted_arm] = aperture
            state.grippers_open[mounted_arm] = aperture >= 0.5
            gripper_action = np.zeros(
                state.action_dim,
                dtype=np.float32,
            )
            gripper_action[
                state.gripper_action_index(mounted_arm)
            ] = proportional_gripper_action(aperture)
            state.note_control(
                "skin_adhesive_dispense",
                "keyboard_pointer",
                gripper_action,
            )
            state.coaching_cue = (
                f"Instrument {mounted_arm + 1} adhesive dispense set to "
                f"{target * 100.0:.0f}%. Both paddles and the metering "
                "piston move together."
            )
            result = dict(state.skin_adhesive_system)
        state.wake_event.set()
        return {
            "ok": True,
            "activation": target,
            "targets": skin_adhesive_activation_targets(target),
            "skin_adhesive_system": result,
        }

    @app.post("/api/closure-robot/command")
    def closure_robot_command(
        request: ClosureRobotCommandRequest,
    ) -> dict[str, Any]:
        action = request.action.strip().lower()
        if action not in {"run", "stop", "reset"}:
            raise HTTPException(400, "action must be run, stop, or reset")
        with state.lock:
            if not state.closure_robot_system.get("enabled", False):
                raise HTTPException(
                    409,
                    "Add the Dr.Anmar approximate–staple–seal robot to this bench first",
                )
            if (
                action == "run"
                and state.closure_robot_system.get("cycle_running", False)
            ):
                raise HTTPException(409, "The physical closure cycle is already running")
            state.closure_robot_command_request = action
            state.operator_input_source = "keyboard_pointer"
            state.coaching_cue = {
                "run": (
                    "Closure robot started. Each phase advances only after the "
                    "articulated mechanism reaches its measured physical target."
                ),
                "stop": (
                    "Closure robot held at its current articulated target. "
                    "PhysX attachments remain physical and active."
                ),
                "reset": (
                    "Resetting the closure mechanism and removing its runtime "
                    "staple, adhesive and temporary capture attachments."
                ),
            }[action]
            result = dict(state.closure_robot_system)
        state.wake_event.set()
        return {"ok": True, "action": action, "closure_robot_system": result}

    @app.get("/api/scenarios")
    def scenarios() -> dict[str, Any]:
        with state.lock:
            current = state.scenario_id
            seed = state.scenario_seed
        return {"scenarios": FAILURE_SCENARIOS, "current": current, "seed": seed}

    @app.post("/api/scenario")
    def apply_scenario(request: ScenarioRequest) -> dict[str, Any]:
        scenario = SCENARIOS_BY_ID.get(request.scenario_id)
        if scenario is None:
            raise HTTPException(404, "Unknown failure scenario")
        if not 0 <= request.seed <= 2_147_483_647:
            raise HTTPException(400, "Seed must be between 0 and 2147483647")
        with state.lock:
            state.scenario_id = request.scenario_id
            state.scenario_seed = request.seed
            state.reset_requested = True
            state.replay_request = "stop"
            state.replaying = False
            state.autonomy_mode = "guided"
            state.coaching_cue = scenario["doctor_focus"]
        state.wake_event.set()
        return {"ok": True, "scenario": scenario, "seed": request.seed, "message": f"{scenario['title']} loaded"}

    @app.post("/api/evaluate")
    def evaluate(request: EvaluationRequest) -> dict[str, Any]:
        if Path(request.demo).name != request.demo or not request.demo.endswith(".npz"):
            raise HTTPException(400, "Invalid demonstration name")
        demo_path = state.demo_dir / request.demo
        if not demo_path.is_file():
            raise HTTPException(404, "Demonstration not found")
        require_replayable_demo(demo_path)
        scenario = SCENARIOS_BY_ID.get(request.scenario_id)
        if scenario is None:
            raise HTTPException(404, "Unknown failure scenario")
        if not 0 <= request.seed <= 2_147_483_647:
            raise HTTPException(400, "Seed must be between 0 and 2147483647")
        with state.lock:
            if state.recording or state.replaying or state.evaluation_status in {"running", "saving"}:
                raise HTTPException(409, "The workstation is already recording or evaluating")
            state.scenario_id = request.scenario_id
            state.scenario_seed = request.seed
            state.reset_requested = True
            state.record_request = "start"
            state.replay_request = request.demo
            state.replaying = False
            state.autonomy_mode = "supervised_replay"
            state.intervention_count = 0
            state.evaluation_status = "running"
            state.evaluation_source = request.demo
            state.evaluation_output = None
            state.coaching_cue = "Challenge evaluation is running. Take control if the behavior becomes unsafe or uncertain."
        state.wake_event.set()
        return {
            "ok": True,
            "status": "running",
            "demo": request.demo,
            "scenario": scenario,
            "seed": request.seed,
        }

    @app.post("/api/autonomy")
    def set_autonomy(request: AutonomyRequest) -> dict[str, Any]:
        if request.mode not in {"manual", "guided"}:
            raise HTTPException(400, "Choose manual or guided control")
        with state.lock:
            if state.expert_demonstration.get("status") in {"running", "paused"}:
                state.expert_request = "take_over"
                state.expert_clean_run = False
            state.autonomy_mode = request.mode
            state.coaching_cue = (
                "Dr.Anmar will surface movement cues while you retain full control."
                if request.mode == "guided"
                else "You command every movement. Dr.Anmar records telemetry for coaching."
            )
        return {"ok": True, "mode": request.mode, "message": "Guided coaching active" if request.mode == "guided" else "Manual control active"}

    @app.post("/api/handoff")
    def handoff() -> dict[str, Any]:
        with state.lock:
            state.disable_hand_motion()
            expert_active = state.expert_demonstration.get("status") in {"running", "paused"}
            was_automatic = state.replaying or state.autonomy_mode in {"supervised_replay", "expert_demonstration"}
            if was_automatic:
                state.intervention_count += 1
            state.replay_request = "stop"
            state.replaying = False
            if expert_active:
                state.expert_request = "take_over"
                state.expert_clean_run = False
            state.autonomy_mode = "manual"
            state.operator_input_source = "keyboard_pointer"
            state.drive.fill(0.0)
            state.drive_until = 0.0
            state.drive_min_steps_remaining = 0
            state.drive_stop_pending = False
            if state.evaluation_status in {"running", "saving"}:
                state.evaluation_status = "interrupted"
                state.record_request = "stop"
            state.coaching_cue = "Control returned to the doctor. The intervention is recorded in this session."
        state.wake_event.set()
        return {"ok": True, "intervention_recorded": was_automatic, "message": "Manual control restored immediately"}

    @app.post("/api/expert/start")
    def expert_start() -> dict[str, Any]:
        with state.lock:
            if not state.procedure:
                raise HTTPException(409, "Load a procedure room before starting its expert demonstration")
            if state.recording or state.record_request == "start":
                raise HTTPException(409, "Stop the current recording before starting the expert")
            if state.replaying or state.evaluation_status in {"running", "saving"}:
                raise HTTPException(409, "Stop replay or evaluation before starting the expert")
            if state.expert_demonstration.get("status") in {"running", "paused"}:
                raise HTTPException(409, "The expert demonstration is already active")
            state.disable_hand_motion()
            state.reset_requested = True
            state.record_request = "start"
            state.expert_request = "start"
            state.expert_reference_pending = False
            state.expert_clean_run = True
            state.intervention_count = 0
            state.autonomy_mode = "expert_demonstration"
            state.operator_input_source = "automation_policy"
            state.coaching_cue = "Expert demonstration starting from the neutral pose. Pause or take control at any phase."
        state.wake_event.set()
        return {"ok": True, "message": "Live expert demonstration starting", "phases": list(EXPERT_PHASES)}

    @app.post("/api/expert/pause")
    def expert_pause() -> dict[str, Any]:
        with state.lock:
            if state.expert_demonstration.get("status") != "running":
                raise HTTPException(409, "The expert demonstration is not running")
            state.expert_request = "pause"
        state.wake_event.set()
        return {"ok": True, "message": "Expert paused; simulator state is preserved"}

    @app.post("/api/expert/resume")
    def expert_resume() -> dict[str, Any]:
        with state.lock:
            if state.expert_demonstration.get("status") != "paused":
                raise HTTPException(409, "The expert demonstration is not paused")
            state.expert_request = "resume"
        state.wake_event.set()
        return {"ok": True, "message": "Expert demonstration resumed"}

    @app.post("/api/expert/take-control")
    def expert_take_control() -> dict[str, Any]:
        return handoff()

    @app.post("/api/record/start")
    def record_start() -> dict[str, bool]:
        with state.lock:
            if state.recording:
                raise HTTPException(409, "A demonstration is already recording")
            state.record_request = "start"
        state.wake_event.set()
        return {"ok": True}

    @app.post("/api/record/stop")
    def record_stop() -> dict[str, bool]:
        with state.lock:
            if not state.recording and state.record_request != "start":
                raise HTTPException(409, "No demonstration is recording")
            state.record_request = "stop"
        state.wake_event.set()
        return {"ok": True}

    @app.post("/api/replay-last")
    def replay_last() -> dict[str, Any]:
        with state.lock:
            if state.recording:
                raise HTTPException(409, "Stop the recording before replaying")
            if not state.last_demo:
                raise HTTPException(404, "Record a demonstration first")
            require_replayable_demo(state.demo_dir / state.last_demo)
            state.disable_hand_motion()
            state.replay_request = state.last_demo
            state.autonomy_mode = "supervised_replay"
            state.coaching_cue = "The recorded behavior is running under supervision. Take control at any time."
            name = state.last_demo
        state.wake_event.set()
        return {"ok": True, "message": f"Replaying {name}"}

    @app.post("/api/replay/{name}")
    def replay_named(name: str) -> dict[str, Any]:
        if Path(name).name != name or not name.endswith(".npz"):
            raise HTTPException(400, "Invalid demonstration name")
        demo_path = state.demo_dir / name
        if not demo_path.is_file():
            raise HTTPException(404, "Demonstration not found")
        require_replayable_demo(demo_path)
        with state.lock:
            if state.recording:
                raise HTTPException(409, "Stop the recording before replaying")
            state.disable_hand_motion()
            state.replay_request = name
            state.autonomy_mode = "supervised_replay"
            state.coaching_cue = "The selected behavior is running under supervision. Take control at any time."
        state.wake_event.set()
        return {"ok": True, "message": f"Replaying {name}"}

    @app.get("/api/demos")
    def demos(limit: int = 100, offset: int = 0) -> dict[str, Any]:
        if not 1 <= limit <= 500 or offset < 0:
            raise HTTPException(400, "limit must be 1–500 and offset must be non-negative")
        files = sorted(state.demo_dir.glob("dr_anmar_*.npz"), reverse=True)
        references = read_reference_map(state.demo_dir)
        items = []
        for path in files[offset : offset + limit]:
            manifest = read_demo_manifest(path)
            integrity = inspect_demo_file(path)
            task = manifest.get("task")
            items.append(
                {
                    "name": path.name,
                    "manifest": path.with_suffix(".json").name if path.with_suffix(".json").is_file() else None,
                    "task": task,
                    "finished_at": manifest.get("finished_at"),
                    "frames": manifest.get("frames"),
                    "duration_s": manifest.get("analysis", {}).get("duration_s"),
                    "analysis": manifest.get("analysis"),
                    "context": manifest.get("context", {}),
                    "modalities": manifest.get("modalities", {}),
                    "integrity": integrity,
                    "is_reference": bool(task and references.get(task) == path.name),
                }
            )
        return {
            "demos": items,
            "total": len(files),
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(items) < len(files),
            "analysis_notice": "Telemetry-derived research coaching; clinician validation is pending.",
        }

    @app.get("/api/demos/{name}/analysis")
    def demo_analysis(name: str) -> dict[str, Any]:
        if Path(name).name != name or not name.endswith(".npz"):
            raise HTTPException(400, "Invalid demonstration name")
        path = state.demo_dir / name
        if not path.is_file():
            raise HTTPException(404, "Demonstration not found")
        return read_demo_manifest(path)

    @app.post("/api/demos/{name}/reference")
    def set_demo_reference(name: str) -> dict[str, Any]:
        if Path(name).name != name or not name.endswith(".npz"):
            raise HTTPException(400, "Invalid demonstration name")
        path = state.demo_dir / name
        if not path.is_file():
            raise HTTPException(404, "Demonstration not found")
        require_replayable_demo(path)
        manifest = read_demo_manifest(path)
        task = manifest.get("task")
        if not task:
            raise HTTPException(409, "This legacy demonstration has no task manifest")
        references = read_reference_map(state.demo_dir)
        references[task] = name
        write_reference_map(state.demo_dir, references)
        with state.lock:
            state.reference_ghost_demo = name
            state.reference_ghost_enabled = True
            state.reference_ghost_update = name
        state.wake_event.set()
        return {
            "ok": True,
            "task": task,
            "reference": name,
            "message": "Clinician reference selected and its registered tool path is visible in the operating room",
        }

    @app.post("/api/reference-ghost")
    def reference_ghost(request: ReferenceGhostRequest) -> dict[str, Any]:
        demo = request.demo
        if request.enabled and not demo:
            demo = read_reference_map(state.demo_dir).get(state.task)
        if request.enabled:
            if not demo or Path(demo).name != demo or not demo.endswith(".npz"):
                raise HTTPException(409, "Select a clinician reference in Skills Twin first")
            if not (state.demo_dir / demo).is_file():
                raise HTTPException(404, "The selected clinician reference file is missing")
            require_replayable_demo(state.demo_dir / demo)
        with state.lock:
            state.reference_ghost_enabled = request.enabled
            if demo:
                state.reference_ghost_demo = demo
            state.reference_ghost_update = demo if request.enabled else "__hide__"
        state.wake_event.set()
        return {
            "ok": True,
            "enabled": request.enabled,
            "reference": demo,
            "message": "Clinician path shown in the room" if request.enabled else "Clinician path hidden",
        }

    @app.post("/api/gaze")
    def gaze(request: GazeRequest) -> dict[str, Any]:
        if not 0.0 <= request.u <= 1.0 or not 0.0 <= request.v <= 1.0:
            raise HTTPException(400, "Normalized gaze coordinates must be between zero and one")
        if request.source not in {"pointer_attention_proxy", "external_eye_tracker", "xr_eye_tracking"}:
            raise HTTPException(400, "Unknown gaze source")
        if request.source != "pointer_attention_proxy" and (
            not EXTERNAL_OPERATOR_SENSORS_ENABLED or not STUDY_ID or not CONSENT_PROTOCOL
        ):
            raise HTTPException(
                403,
                "External gaze is disabled until a study ID, consent protocol, and external-sensor opt-in are configured",
            )
        with state.lock:
            state.gaze_uv = (request.u, request.v)
            state.gaze_valid = request.valid
            state.gaze_source = request.source
        return {"ok": True, "gaze_uv": [request.u, request.v], "valid": request.valid, "source": request.source}

    @app.post("/api/annotation")
    def annotate(request: ProcedureAnnotationRequest) -> dict[str, Any]:
        if request.phase is not None and request.phase not in PROCEDURE_PHASES:
            raise HTTPException(400, "Unknown procedure phase")
        if request.event is not None and request.event not in PROCEDURE_EVENTS:
            raise HTTPException(400, "Unknown procedure event")
        if len(request.note) > 240:
            raise HTTPException(400, "Annotation note is limited to 240 characters")
        with state.lock:
            if request.phase is not None:
                state.procedure_phase = request.phase
            if request.event is not None:
                state.procedure_event_code = PROCEDURE_EVENTS[request.event]
                state.procedure_event_sequence += 1
            annotation = {
                "time": datetime.now(timezone.utc).isoformat(),
                "recorded_frame": state.recorded_frames,
                "frame_alignment": "next_control_frame_index",
                "sim_step": state.sim_step,
                "phase": state.procedure_phase,
                "event": request.event,
                "event_sequence": state.procedure_event_sequence,
                "note": request.note,
            }
            state.procedure_events.append(annotation)
        return {"ok": True, "annotation": annotation, "message": f"{state.procedure_phase.title()} annotation saved"}

    @app.get("/api/demos/{name}/comparison")
    def compare_demo(name: str) -> dict[str, Any]:
        if Path(name).name != name or not name.endswith(".npz"):
            raise HTTPException(400, "Invalid demonstration name")
        candidate = state.demo_dir / name
        if not candidate.is_file():
            raise HTTPException(404, "Demonstration not found")
        require_replayable_demo(candidate)
        manifest = read_demo_manifest(candidate)
        task = manifest.get("task")
        reference_name = read_reference_map(state.demo_dir).get(task)
        if not reference_name:
            raise HTTPException(404, "Select a clinician reference for this task first")
        reference = state.demo_dir / reference_name
        if not reference.is_file():
            raise HTTPException(404, "The selected clinician reference file is missing")
        require_replayable_demo(reference)
        return compare_demonstrations(candidate, reference)

    @app.get("/demos/{name}")
    def download_demo(name: str) -> FileResponse:
        if Path(name).name != name or not name.endswith((".npz", ".json", ".hdf5")):
            raise HTTPException(400, "Invalid demonstration name")
        path = state.demo_dir / name
        if not path.is_file():
            raise HTTPException(404, "Demonstration not found")
        return FileResponse(path, filename=name)

    @app.get("/video")
    async def video() -> StreamingResponse:
        async def frames():
            last_id = -1
            with state.lock:
                state.camera_subscribers["endoscope_left"] = state.camera_subscribers.get("endoscope_left", 0) + 1
            try:
                while True:
                    with state.lock:
                        frame_id = state.frame_id
                        jpeg = state.frame_jpeg
                    if jpeg and frame_id != last_id:
                        last_id = frame_id
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    await asyncio.sleep(0.04)
            finally:
                with state.lock:
                    state.camera_subscribers["endoscope_left"] = max(
                        0, state.camera_subscribers.get("endoscope_left", 1) - 1
                    )

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/video/{camera_name}")
    async def camera_video(camera_name: str) -> StreamingResponse:
        if camera_name not in state.camera_names:
            raise HTTPException(404, "Unknown simulated camera")

        async def frames():
            last_id = -1
            with state.lock:
                state.camera_subscribers[camera_name] = state.camera_subscribers.get(camera_name, 0) + 1
            try:
                while True:
                    with state.lock:
                        frame_id = state.camera_frame_ids.get(camera_name, -1)
                        jpeg = state.camera_frames_jpeg.get(camera_name, b"")
                    if jpeg and frame_id != last_id:
                        last_id = frame_id
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    await asyncio.sleep(0.04)
            finally:
                with state.lock:
                    state.camera_subscribers[camera_name] = max(
                        0, state.camera_subscribers.get(camera_name, 1) - 1
                    )

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/frame.jpg")
    def still_frame() -> Response:
        with state.lock:
            poll_time = time.monotonic()
            state.camera_poll_last_seen = poll_time
            state.camera_poll_last_seen_by_name["endoscope_left"] = poll_time
            jpeg = state.frame_jpeg
        if not jpeg:
            raise HTTPException(503, "The first camera frame is not ready")
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/frame/{camera_name}.jpg")
    def camera_still_frame(camera_name: str) -> Response:
        if camera_name not in state.camera_names:
            raise HTTPException(404, "Unknown simulated camera")
        with state.lock:
            poll_time = time.monotonic()
            state.camera_poll_last_seen = poll_time
            state.camera_poll_last_seen_by_name[camera_name] = poll_time
            jpeg = state.camera_frames_jpeg.get(camera_name, b"")
        if not jpeg:
            raise HTTPException(503, "The first camera frame is not ready")
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app


def apply_visual_scenario(image: Image.Image, scenario_id: str) -> Image.Image:
    if scenario_id in {"low_light", "combined_visual"}:
        image = ImageEnhance.Brightness(image).enhance(0.48 if scenario_id == "low_light" else 0.38)
        image = ImageEnhance.Contrast(image).enhance(1.12)
    if scenario_id == "combined_visual":
        image = image.filter(ImageFilter.GaussianBlur(radius=1.6))
    if scenario_id in {"glare", "combined_visual"}:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        draw.ellipse(
            (int(width * 0.52), int(height * 0.06), int(width * 1.03), int(height * 0.54)),
            fill=(255, 252, 232, 116 if scenario_id == "glare" else 88),
        )
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    if scenario_id == "partial_occlusion":
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        draw.rounded_rectangle(
            (int(width * 0.67), int(height * 0.18), int(width * 1.04), int(height * 0.88)),
            radius=max(12, int(width * 0.025)),
            fill=(4, 10, 13, 226),
        )
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return image


def rgb_tensor_to_array(rgb: torch.Tensor) -> np.ndarray:
    array = rgb[..., :3].detach().cpu().numpy()
    if np.issubdtype(array.dtype, np.floating):
        return np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array.astype(np.uint8, copy=False)


def rgb_array_to_image(array: np.ndarray, scenario_id: str = "baseline", dropout: bool = False) -> Image.Image:
    image = Image.fromarray(array)
    return Image.new("RGB", image.size, (0, 0, 0)) if dropout else apply_visual_scenario(image, scenario_id)


def rgb_tensor_to_image(rgb: torch.Tensor, scenario_id: str = "baseline", dropout: bool = False) -> Image.Image:
    return rgb_array_to_image(rgb_tensor_to_array(rgb), scenario_id, dropout)


def encode_jpeg_array(array: np.ndarray, scenario_id: str = "baseline", dropout: bool = False) -> tuple[bytes, float]:
    image = rgb_array_to_image(array, scenario_id, dropout)
    sample = np.asarray(image.resize((32, 24), Image.Resampling.BILINEAR), dtype=np.float32)
    mean_luminance = float(sample.mean() / 255.0)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=86, optimize=False)
    return buffer.getvalue(), mean_luminance


def encode_jpeg(rgb: torch.Tensor, scenario_id: str = "baseline", dropout: bool = False) -> tuple[bytes, float]:
    return encode_jpeg_array(rgb_tensor_to_array(rgb), scenario_id, dropout)


class BoundedJpegEncoder:
    """Encode only the newest available camera job without blocking simulation."""

    MAX_QUEUED_JOBS = 1

    def __init__(self, state: SharedState) -> None:
        self.state = state
        self._queue: queue.Queue[tuple[dict[str, np.ndarray], str, bool, float] | None] = queue.Queue(
            maxsize=self.MAX_QUEUED_JOBS
        )
        self._closed = False
        self._last_completed = 0.0
        self._thread = threading.Thread(target=self._run, name="dr-anmar-jpeg", daemon=True)
        self._thread.start()

    def submit(
        self,
        frames: dict[str, np.ndarray],
        scenario_id: str,
        dropout: bool,
        submitted_at: float,
    ) -> None:
        if self._closed or not frames:
            return
        job = (frames, scenario_id, dropout, submitted_at)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(job)
            with self.state.lock:
                self.state.jpeg_frames_dropped += 1
        with self.state.lock:
            self.state.jpeg_queue_depth = self._queue.qsize()

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                try:
                    frames, scenario_id, dropout, _submitted_at = job
                    rendered: dict[str, bytes] = {}
                    left_luminance: float | None = None
                    for camera_name, array in frames.items():
                        jpeg, luminance = encode_jpeg_array(array, scenario_id, dropout)
                        rendered[camera_name] = jpeg
                        if camera_name == "endoscope_left":
                            left_luminance = luminance
                    completed_at = time.monotonic()
                    with self.state.lock:
                        self.state.camera_frames_jpeg.update(rendered)
                        for camera_name in rendered:
                            self.state.camera_frame_ids[camera_name] = self.state.camera_frame_ids.get(camera_name, 0) + 1
                        self.state.frame_jpeg = rendered.get("endoscope_left", self.state.frame_jpeg)
                        self.state.frame_id += 1
                        elapsed = completed_at - self._last_completed
                        self.state.render_fps = 1.0 / elapsed if self._last_completed and elapsed > 0 else 0.0
                        self.state.jpeg_queue_depth = self._queue.qsize()
                        if left_luminance is not None:
                            self.state.camera_mean_luminance = left_luminance
                            if left_luminance > 0.01:
                                self.state.camera_nonblank_seen = True
                    self._last_completed = completed_at
                    del frames
                except Exception:
                    traceback.print_exc()
                    with self.state.lock:
                        self.state.jpeg_frames_dropped += 1
                        self.state.jpeg_queue_depth = self._queue.qsize()
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                self._queue.put(None, timeout=0.25)
                break
            except queue.Full:
                continue
        self._thread.join(timeout=5.0)
        with self.state.lock:
            self.state.jpeg_queue_depth = 0


def depth_to_point_cloud(depth_m: np.ndarray, intrinsics: np.ndarray, stride: int = 16) -> np.ndarray:
    """Unproject metric endoscope depth to a compact camera-frame XYZ cloud."""
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        return np.zeros((0, 3), dtype=np.float32)
    rows = np.arange(0, depth.shape[0], stride)
    columns = np.arange(0, depth.shape[1], stride)
    u, v = np.meshgrid(columns, rows)
    z = depth[v, u]
    valid = np.isfinite(z) & (z > 0.0)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    x = (u.astype(np.float32) - cx) * z / max(fx, 1e-6)
    y = (v.astype(np.float32) - cy) * z / max(fy, 1e-6)
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3).astype(np.float32)
    points[~valid.reshape(-1)] = 0.0
    return points


def camera_semantic_labels(camera) -> dict[str, Any]:
    """Normalize Isaac Lab camera metadata across single- and multi-env layouts."""

    info = getattr(camera.data, "info", {})
    if isinstance(info, (list, tuple)):
        info = info[0] if info else {}
    if not isinstance(info, dict):
        return {}
    segmentation = info.get("semantic_segmentation", {})
    if isinstance(segmentation, (list, tuple)):
        segmentation = segmentation[0] if segmentation else {}
    if not isinstance(segmentation, dict):
        return {}
    labels = segmentation.get("idToLabels", {})
    return {str(key): value for key, value in labels.items()} if isinstance(labels, dict) else {}


def read_demo_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_DEMO_INSPECTION_CACHE: OrderedDict[tuple[str, int, int], dict[str, Any]] = OrderedDict()
MAX_DEMO_INSPECTION_CACHE = 256


def inspect_demo_file(path: Path) -> dict[str, Any]:
    """Bounded, cached structural validation for replay and dataset use."""
    try:
        stat = path.stat()
    except OSError as exc:
        return {"valid": False, "training_eligible": False, "error": str(exc)}
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _DEMO_INSPECTION_CACHE.get(cache_key)
    if cached is not None:
        _DEMO_INSPECTION_CACHE.move_to_end(cache_key)
        return dict(cached)
    result: dict[str, Any]
    try:
        with np.load(path, allow_pickle=False) as data:
            if "actions" not in data.files:
                raise ValueError("missing actions array")
            actions = np.asarray(data["actions"])
            if actions.ndim != 2 or actions.shape[1] < 1:
                raise ValueError("actions must be a two-dimensional trajectory")
            if not np.issubdtype(actions.dtype, np.number) or not np.all(np.isfinite(actions)):
                raise ValueError("actions contain non-finite or non-numeric values")
            frame_count = int(actions.shape[0])
            if "time_s" in data.files:
                times = np.asarray(data["time_s"]).reshape(-1)
                if len(times) != frame_count or not np.all(np.isfinite(times)):
                    raise ValueError("time_s is not finite and frame-aligned")
                if len(times) > 1 and np.any(np.diff(times) < 0):
                    raise ValueError("time_s is not monotonic")
            manifest = read_demo_manifest(path)
            contract = manifest.get("action_contract", {})
            contract_matches = contract.get("id") in {
                ACTION_CONTRACT["id"],
                NON_PSM_ACTION_CONTRACT["id"],
            }
            warnings = [] if frame_count >= 2 else ["Recording has fewer than two control frames"]
            if not contract_matches:
                warnings.append("Recording predates the current native PSM or Cartesian action contract")
            result = {
                "valid": True,
                "training_eligible": frame_count >= 2 and contract_matches,
                "frames": frame_count,
                "action_dim": int(actions.shape[1]),
                "action_contract": contract or None,
                "action_contract_current": contract_matches,
                "bytes": stat.st_size,
                "warnings": warnings,
                "error": None,
            }
    except Exception as exc:
        result = {
            "valid": False,
            "training_eligible": False,
            "bytes": stat.st_size,
            "warnings": [],
            "error": f"Unreadable demonstration: {exc}",
        }
    for stale_key in [key for key in _DEMO_INSPECTION_CACHE if key[0] == str(path) and key != cache_key]:
        _DEMO_INSPECTION_CACHE.pop(stale_key, None)
    while len(_DEMO_INSPECTION_CACHE) >= MAX_DEMO_INSPECTION_CACHE:
        _DEMO_INSPECTION_CACHE.popitem(last=False)
    _DEMO_INSPECTION_CACHE[cache_key] = result
    return dict(result)


def require_replayable_demo(path: Path) -> dict[str, Any]:
    inspection = inspect_demo_file(path)
    if not inspection.get("valid"):
        raise HTTPException(422, inspection.get("error", "The demonstration is unreadable"))
    if not inspection.get("training_eligible"):
        if not inspection.get("action_contract_current", False):
            raise HTTPException(422, "This demonstration uses a legacy action scale and must be migrated before replay or training")
        raise HTTPException(422, "The demonstration is too short to replay or train from")
    return inspection


def read_reference_map(demo_dir: Path) -> dict[str, str]:
    try:
        value = json.loads((demo_dir / "clinician_references.json").read_text(encoding="utf-8"))
        return {str(key): str(name) for key, name in value.items()} if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_reference_map(demo_dir: Path, value: dict[str, str]) -> None:
    path = demo_dir / "clinician_references.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normalized_action_trace(actions: np.ndarray, points: int = 160) -> np.ndarray:
    if len(actions) == 0:
        return np.zeros((points, actions.shape[1] if actions.ndim == 2 else 1), dtype=np.float64)
    source = np.linspace(0.0, 1.0, len(actions))
    target = np.linspace(0.0, 1.0, points)
    return np.stack([np.interp(target, source, actions[:, axis]) for axis in range(actions.shape[1])], axis=1)


def compare_demonstrations(candidate_path: Path, reference_path: Path) -> dict[str, Any]:
    with np.load(candidate_path, allow_pickle=False) as candidate_data, np.load(reference_path, allow_pickle=False) as reference_data:
        candidate_actions = np.asarray(candidate_data["actions"], dtype=np.float64)
        reference_actions = np.asarray(reference_data["actions"], dtype=np.float64)
    dimensions = min(candidate_actions.shape[1], reference_actions.shape[1])
    candidate_trace = _normalized_action_trace(candidate_actions[:, :dimensions])
    reference_trace = _normalized_action_trace(reference_actions[:, :dimensions])
    action_rmse = float(np.sqrt(np.mean((candidate_trace - reference_trace) ** 2)))
    reference_scale = float(np.sqrt(np.mean(reference_trace**2)))
    normalized_error = action_rmse / max(reference_scale, 1e-4)
    similarity = int(round(max(0.0, min(100.0, 100.0 * np.exp(-0.85 * normalized_error)))))
    candidate_manifest = read_demo_manifest(candidate_path)
    reference_manifest = read_demo_manifest(reference_path)
    candidate_analysis = candidate_manifest.get("analysis", {})
    reference_analysis = reference_manifest.get("analysis", {})
    candidate_metrics = candidate_analysis.get("metrics", {})
    reference_metrics = reference_analysis.get("metrics", {})
    duration_delta = float(candidate_analysis.get("duration_s", 0.0)) - float(reference_analysis.get("duration_s", 0.0))
    path_delta = float(candidate_metrics.get("tool_path_m", 0.0)) - float(reference_metrics.get("tool_path_m", 0.0))
    recovery_delta = float(candidate_metrics.get("recovery_hold_s", 0.0)) - float(reference_metrics.get("recovery_hold_s", 0.0))
    guidance = []
    if similarity < 55:
        guidance.append("The action rhythm differs substantially from the selected reference; compare approach and grasp timing first.")
    if duration_delta > 2.0:
        guidance.append("This attempt is slower than the reference. Inspect idle segments before increasing movement speed.")
    if path_delta > 0.04:
        guidance.append("The tool travelled farther than the reference; look for avoidable corrections or indirect approach motion.")
    if recovery_delta < -0.25:
        guidance.append("The selected reference holds a longer stable recovery pose.")
    if not guidance:
        guidance.append("The action structure is close to the clinician-selected reference; challenge it under a new scenario next.")
    return {
        "schema": "dr.anmar.reference-comparison.v1",
        "validation_status": "research_proxy_pending_clinician_validation",
        "candidate": candidate_path.name,
        "reference": reference_path.name,
        "action_similarity": similarity,
        "action_rmse": round(action_rmse, 7),
        "duration_delta_s": round(duration_delta, 2),
        "tool_path_delta_m": round(path_delta, 4),
        "recovery_hold_delta_s": round(recovery_delta, 2),
        "guidance": guidance,
    }


def reference_tool_path(path: Path, max_points_per_tool: int = 54) -> tuple[np.ndarray, np.ndarray]:
    """Extract registered task-native tool-tip paths, with a legacy mobility fallback."""
    manifest = read_demo_manifest(path)
    body_names_by_robot = manifest.get("robot_body_names", {})
    allow_legacy_fallback = not str(manifest.get("schema", "")).startswith("dr.anmar.demonstration.v2")
    preferred_tip_names = ("psm_tool_tip_link", "endo360_needle", "ecm_end_link", "tool_tip", "end_effector")
    with np.load(path, allow_pickle=False) as data:
        candidates: list[np.ndarray] = []
        legacy_candidates: list[tuple[float, np.ndarray]] = []
        for key in data.files:
            if not key.endswith("_body_positions_w"):
                continue
            positions = np.asarray(data[key], dtype=np.float32)
            if positions.ndim != 3 or len(positions) < 2:
                continue
            robot_name = key.removesuffix("_body_positions_w")
            body_names = list(body_names_by_robot.get(robot_name, []))
            preferred_index = next(
                (body_names.index(name) for name in preferred_tip_names if name in body_names),
                None,
            )
            if preferred_index is not None and preferred_index < positions.shape[1]:
                candidates.append(positions[:, preferred_index, :3])
                continue
            if allow_legacy_fallback:
                path_lengths = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=2), axis=0)
                body_index = int(np.argmax(path_lengths))
                legacy_candidates.append((float(path_lengths[body_index]), positions[:, body_index, :3]))
    if not candidates:
        if not legacy_candidates:
            raise ValueError("The reference has no registered task-native tool-tip trajectory")
        candidates = [max(legacy_candidates, key=lambda item: item[0])[1]]
    all_points = []
    all_phases = []
    for tool_path in candidates:
        tool_path = tool_path[np.all(np.isfinite(tool_path), axis=1)]
        if not len(tool_path):
            continue
        indices = np.unique(
            np.linspace(0, len(tool_path) - 1, min(max_points_per_tool, len(tool_path))).round().astype(int)
        )
        points = tool_path[indices]
        normalized_time = indices / max(len(tool_path) - 1, 1)
        phases = np.where(normalized_time < 0.35, 0, np.where(normalized_time < 0.78, 1, 2)).astype(np.int32)
        all_points.append(points)
        all_phases.append(phases)
    if not all_points:
        raise ValueError("The reference tool trajectory contains no finite points")
    return np.concatenate(all_points, axis=0), np.concatenate(all_phases, axis=0)


def apply_native_object_scenario(objects: dict[str, Any], scenario_id: str, seed: int) -> None:
    profile = SCENARIO_NATIVE_PROFILES.get(scenario_id, {})
    offset = profile.get("object_offset_m")
    if offset is None or not objects:
        return
    if "object" in objects:
        targets = {"object": objects["object"]}
    elif "suture_needle" in objects:
        targets = {"suture_needle": objects["suture_needle"]}
    else:
        return
    generator = np.random.default_rng(seed)
    seeded_jitter = generator.uniform(-0.0015, 0.0015, size=3).astype(np.float32)
    seeded_jitter[2] = 0.0
    for rigid_object in targets.values():
        pose = rigid_object.data.root_pose_w.clone()
        delta = torch.tensor(np.asarray(offset, dtype=np.float32) + seeded_jitter, device=pose.device)
        pose[:, :3] += delta
        rigid_object.write_root_pose_to_sim(pose)
        rigid_object.write_root_velocity_to_sim(torch.zeros((pose.shape[0], 6), device=pose.device))


def sample_deformable_safety(
    deformables: dict[str, Any], *, include_material_metrics: bool = True
) -> dict[str, float]:
    telemetry: dict[str, float] = {}
    for name, deformable in deformables.items():
        try:
            nodal_position = deformable.data.nodal_pos_w[0]
            default_position = deformable.data.default_nodal_state_w[0, :, :3]
            displacement = torch.linalg.vector_norm(nodal_position - default_position, dim=-1).max()
            telemetry[f"{name}_max_tissue_displacement_m"] = float(displacement.detach().cpu().item())
            if not include_material_metrics:
                continue
            deformation = deformable.data.sim_element_deform_gradient_w[0]
            identity = torch.eye(3, device=deformation.device).reshape(1, 3, 3)
            deformation_proxy = torch.linalg.matrix_norm(deformation - identity, dim=(-2, -1)).max()
            stress = deformable.data.sim_element_stress_w[0]
            max_stress = torch.linalg.matrix_norm(stress, dim=(-2, -1)).max()
            telemetry[f"{name}_max_deformation_gradient_proxy"] = float(deformation_proxy.detach().cpu().item())
            telemetry[f"{name}_max_tissue_stress_pa"] = float(max_stress.detach().cpu().item())
        except (AttributeError, RuntimeError, IndexError, ValueError):
            continue
    return telemetry


def _first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def _last_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[-1]) if len(indices) else None


def action_channel_views(actions: np.ndarray, arms: int) -> tuple[np.ndarray, np.ndarray]:
    """Split Isaac action vectors for both single and interleaved dual-arm tasks."""
    group_width = 7 if actions.shape[1] >= arms * 7 else 6
    motion = np.stack(
        [actions[:, arm * group_width : arm * group_width + 6] for arm in range(arms)],
        axis=1,
    )
    if group_width == 7:
        grippers = np.stack([actions[:, arm * 7 + 6] for arm in range(arms)], axis=1)
    else:
        grippers = np.zeros((actions.shape[0], 0), dtype=actions.dtype)
    return motion, grippers


def analyze_demo(
    arrays: dict[str, np.ndarray],
    task: str,
    arms: int,
    robot_body_names: dict[str, list[str]],
    procedure_id: str = "",
) -> dict[str, Any]:
    """Evaluate recorded evidence without reconstructing simulator physics."""

    actions = np.asarray(arrays.get("actions", []), dtype=np.float64)
    times = np.asarray(arrays.get("time_s", []), dtype=np.float64).reshape(-1)
    motion, _grippers = action_channel_views(actions, arms)
    translations = motion[..., :3]
    rotations = motion[..., 3:6]
    duration_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    native_success = np.asarray(arrays.get("environment_success", []), dtype=np.float64).reshape(-1)
    native_success_available = bool(len(native_success) and np.any(native_success >= 0.0))
    rewards = np.asarray(arrays.get("environment_reward", []), dtype=np.float64).reshape(-1)
    terminations = np.asarray(arrays.get("environment_terminated", []), dtype=np.bool_).reshape(-1)
    truncations = np.asarray(arrays.get("environment_truncated", []), dtype=np.bool_).reshape(-1)
    contact_arrays = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for key, value in arrays.items()
        if key.endswith("_max_contact_force_n")
    ]
    deformable_arrays = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for key, value in arrays.items()
        if key.endswith("_max_tissue_displacement_m")
    ]
    peak_contact = max((float(np.nanmax(value)) for value in contact_arrays if value.size), default=None)
    peak_displacement = max((float(np.nanmax(value)) for value in deformable_arrays if value.size), default=None)
    return {
        "schema": "dr.anmar.native-demonstration-analysis.v1",
        "task": task,
        "procedure_id": procedure_id or None,
        "frames": int(len(actions)),
        "duration_s": round(duration_s, 4),
        "motion": {
            "translation_effort": round(float(np.linalg.norm(translations, axis=-1).sum()), 5)
            if translations.size
            else 0.0,
            "rotation_effort": round(float(np.linalg.norm(rotations, axis=-1).sum()), 5)
            if rotations.size
            else 0.0,
            "active_fraction": round(float(np.mean(np.linalg.norm(actions, axis=-1) > 1e-5)), 4)
            if actions.size
            else 0.0,
        },
        "native_outcome": {
            "available": native_success_available,
            "success": bool(np.any(native_success > 0.5)) if native_success_available else None,
            "reward_max": round(float(np.nanmax(rewards)), 5) if rewards.size else None,
            "terminated": bool(np.any(terminations)),
            "truncated": bool(np.any(truncations)),
        },
        "native_safety": {
            "peak_contact_force_n": round(peak_contact, 5) if peak_contact is not None else None,
            "peak_tissue_displacement_m": round(peak_displacement, 6)
            if peak_displacement is not None
            else None,
        },
        "interpretation": (
            "Physical outcomes come only from the active Isaac Lab task and NVIDIA backend. "
            "Dr.Anmar evaluates recorded actions, observations and native solver telemetry."
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_provenance(state: SharedState) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        revision = None
    gpu_name = None
    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except (RuntimeError, AssertionError):
        pass
    configuration = {
        "task": state.task,
        "procedure": state.procedure,
        "anatomy_scene_id": state.anatomy_scene_id,
        "sensor_profile": state.sensor_profile,
    }
    return {
        "source_revision": revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "gpu": gpu_name,
        "task_id": state.task,
        "task_configuration_sha256": hashlib.sha256(
            json.dumps(configuration, sort_keys=True, default=str, separators=(",", ":")).encode()
        ).hexdigest(),
        "policy_checkpoint_sha256": None,
    }


def array_payload_bytes(frame: dict[str, np.ndarray]) -> int:
    return sum(int(np.asarray(value).nbytes) for value in frame.values())


def write_npz_from_hdf5(
    destination: Path,
    arrays: dict[str, h5py.Dataset],
    chunk_budget_bytes: int = 16 * 1024 * 1024,
) -> None:
    """Write a NumPy-compatible compressed archive without loading full datasets."""
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=3,
            allowZip64=True,
        ) as archive:
            for key, dataset in arrays.items():
                header = {
                    "descr": np.lib.format.dtype_to_descr(dataset.dtype),
                    "fortran_order": False,
                    "shape": tuple(dataset.shape),
                }
                with archive.open(f"{key}.npy", mode="w", force_zip64=True) as member:
                    np.lib.format.write_array_header_2_0(member, header)
                    if not dataset.shape:
                        member.write(memoryview(np.ascontiguousarray(dataset[()])).cast("B"))
                        continue
                    if dataset.shape[0] == 0:
                        continue
                    row_bytes = max(1, int(dataset.dtype.itemsize * np.prod(dataset.shape[1:], dtype=np.int64)))
                    rows_per_chunk = max(1, chunk_budget_bytes // row_bytes)
                    for start in range(0, dataset.shape[0], rows_per_chunk):
                        stop = min(dataset.shape[0], start + rows_per_chunk)
                        chunk = np.ascontiguousarray(dataset[start:stop])
                        member.write(memoryview(chunk).cast("B"))
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class BoundedCaptureSpool:
    """Append recording chunks to HDF5 on one bounded background writer."""

    CONTROL_BATCH = 128
    VISION_BATCH = 8
    MAX_QUEUED_BATCHES = 6
    VISION_KEYS = {
        "time_s": "endoscope_time_s",
        "rgb": "endoscope_rgb",
        "sensor_dropout_active": "endoscope_sensor_dropout_active",
        "depth_m": "endoscope_depth_m",
        "semantic_id": "endoscope_semantic_id",
        "point_cloud_camera_m": "endoscope_point_cloud_camera_m",
    }

    def __init__(self, demo_dir: Path) -> None:
        demo_dir.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}-{time.time_ns()}"
        self.path = demo_dir / f".dr-anmar-capture-{token}.hdf5"
        self.control_count = 0
        self.vision_count = 0
        self.payload_bytes = 0
        self._buffers: dict[str, list[dict[str, np.ndarray]]] = {"control": [], "vision": []}
        self._queue: queue.Queue[tuple[str, list[dict[str, np.ndarray]]]] = queue.Queue(
            maxsize=self.MAX_QUEUED_BATCHES
        )
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._write_loop, name="dr-anmar-recorder", daemon=True)
        self._thread.start()

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def buffered_frames(self) -> int:
        return sum(len(batch) for batch in self._buffers.values())

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Recording spool failed: {self._error}") from self._error

    def _enqueue(self, item: tuple[str, list[dict[str, np.ndarray]]]) -> None:
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(item, timeout=0.25)
                return
            except queue.Full:
                continue

    def _flush_buffer(self, kind: str) -> None:
        batch = self._buffers[kind]
        if not batch:
            return
        self._buffers[kind] = []
        self._enqueue((kind, batch))

    def _append(self, kind: str, frame: dict[str, np.ndarray]) -> None:
        if self._closed:
            raise RuntimeError("Recording spool is already closed")
        self._raise_if_failed()
        self.payload_bytes += array_payload_bytes(frame)
        if kind == "control":
            self.control_count += 1
            limit = self.CONTROL_BATCH
        else:
            self.vision_count += 1
            limit = self.VISION_BATCH
        self._buffers[kind].append(frame)
        if len(self._buffers[kind]) >= limit:
            self._flush_buffer(kind)

    def append_control(self, frame: dict[str, np.ndarray]) -> None:
        self._append("control", frame)

    def append_vision(self, frame: dict[str, np.ndarray]) -> None:
        self._append("vision", frame)

    @classmethod
    def _dataset_name(cls, kind: str, key: str) -> str:
        return cls.VISION_KEYS.get(key, key) if kind == "vision" else key

    @classmethod
    def _write_batch(cls, destination: h5py.File, kind: str, batch: list[dict[str, np.ndarray]]) -> None:
        start = int(destination.attrs.get(f"{kind}_frames", 0))
        end = start + len(batch)
        source_keys = sorted(set().union(*(frame.keys() for frame in batch)))
        written_names: set[str] = set()
        for source_key in source_keys:
            target_key = cls._dataset_name(kind, source_key)
            template = next(np.asarray(frame[source_key]) for frame in batch if source_key in frame)
            fill = np.nan if np.issubdtype(template.dtype, np.floating) else 0
            values = np.stack(
                [
                    np.asarray(frame[source_key])
                    if source_key in frame
                    else np.full(template.shape, fill, dtype=template.dtype)
                    for frame in batch
                ]
            )
            if target_key not in destination:
                chunk_rows = max(1, min(len(batch), cls.VISION_BATCH if kind == "vision" else cls.CONTROL_BATCH))
                dataset = destination.create_dataset(
                    target_key,
                    shape=(start, *values.shape[1:]),
                    maxshape=(None, *values.shape[1:]),
                    chunks=(chunk_rows, *values.shape[1:]),
                    dtype=values.dtype,
                    compression="lzf",
                    shuffle=values.dtype.itemsize > 1,
                )
                dataset.attrs["capture_kind"] = kind
            dataset = destination[target_key]
            if dataset.shape[1:] != values.shape[1:] or dataset.dtype != values.dtype:
                raise ValueError(f"Recording field changed shape or dtype: {target_key}")
            dataset.resize((end, *dataset.shape[1:]))
            dataset[start:end] = values
            written_names.add(target_key)
        for target_key, dataset in destination.items():
            if dataset.attrs.get("capture_kind") == kind and target_key not in written_names:
                dataset.resize((end, *dataset.shape[1:]))
        destination.attrs[f"{kind}_frames"] = end
        destination.flush()
        batch.clear()

    def _write_loop(self) -> None:
        try:
            with h5py.File(self.path, "w", libver="latest") as destination:
                destination.attrs["schema"] = "dr.anmar.capture-spool.v1"
                while True:
                    kind, batch = self._queue.get()
                    try:
                        if kind == "stop":
                            return
                        self._write_batch(destination, kind, batch)
                    finally:
                        self._queue.task_done()
        except BaseException as exc:
            self._error = exc

    def finish(self) -> Path:
        if self._closed:
            self._raise_if_failed()
            return self.path
        self._flush_buffer("control")
        self._flush_buffer("vision")
        self._enqueue(("stop", []))
        self._thread.join()
        self._closed = True
        self._raise_if_failed()
        return self.path

    def abort(self) -> None:
        for batch in self._buffers.values():
            batch.clear()
        while True:
            try:
                _kind, batch = self._queue.get_nowait()
                batch.clear()
                self._queue.task_done()
            except queue.Empty:
                break
        if self._thread.is_alive():
            self._queue.put(("stop", []))
            self._thread.join(timeout=5.0)
        self._closed = True
        self.path.unlink(missing_ok=True)


def write_native_psm_training_hdf5(
    spool_path: Path,
    destination: Path,
    state: SharedState,
) -> Path:
    """Export the live room recording in Isaac Lab's native episode schema."""

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(spool_path, "r") as source, h5py.File(temporary, "w", libver="latest") as output:
            required = {"actions", "cartesian_actions", "resolved_joint_targets", "time_s"}
            missing = sorted(required.difference(source.keys()))
            if missing:
                raise ValueError(f"native PSM recording is missing {', '.join(missing)}")
            actions = source["actions"]
            cartesian = source["cartesian_actions"]
            targets = source["resolved_joint_targets"]
            expected_dim = 7 * len(state.native_psm_robot_names)
            if actions.ndim != 2 or actions.shape[1] != expected_dim:
                raise ValueError(f"native PSM policy actions have shape {actions.shape}; expected (T, {expected_dim})")
            if targets.shape != actions.shape or cartesian.shape[0] != actions.shape[0]:
                raise ValueError("native PSM action, Cartesian intent, and target streams are not frame-aligned")
            if not np.isfinite(actions[:]).all() or not np.isfinite(targets[:]).all():
                raise ValueError("native PSM action contract contains NaN or infinity")
            if not np.isin(actions[:, 6::7], (-1.0, 1.0)).all():
                raise ValueError("native PSM policy recording contains a non-binary gripper action")

            data = output.create_group("data")
            episode = data.create_group("demo_0")
            observations = episode.create_group("obs")
            states = episode.create_group("states").create_group("articulation")
            initial = episode.create_group("initial_state").create_group("articulation")
            _copy_hdf5_dataset(actions, episode, "actions")
            episode["processed_actions"] = episode["actions"]
            observations["actions"] = episode["actions"]
            _copy_hdf5_dataset(cartesian, episode, "cartesian_actions")
            _copy_hdf5_dataset(targets, episode, "resolved_joint_targets")

            joint_positions = []
            joint_velocities = []
            for robot_name in state.native_psm_robot_names:
                position_key = f"{robot_name}_joint_positions"
                velocity_key = f"{robot_name}_joint_velocities"
                if position_key not in source or velocity_key not in source:
                    raise ValueError(f"native PSM recording is missing state for {robot_name}")
                joint_positions.append(np.asarray(source[position_key]))
                joint_velocities.append(np.asarray(source[velocity_key]))
            position_values = np.concatenate(joint_positions, axis=-1)
            velocity_values = np.concatenate(joint_velocities, axis=-1)
            observations.create_dataset(
                "joint_pos",
                data=position_values,
                chunks=(min(128, len(position_values)), position_values.shape[1]),
                compression="lzf",
            )
            observations.create_dataset(
                "joint_vel",
                data=velocity_values,
                chunks=(min(128, len(velocity_values)), velocity_values.shape[1]),
                compression="lzf",
            )

            for robot_name in state.native_psm_robot_names:
                robot_state = states.create_group(robot_name)
                robot_initial = initial.create_group(robot_name)
                for source_suffix, target_name in (
                    ("joint_positions", "joint_position"),
                    ("joint_velocities", "joint_velocity"),
                    ("root_pose_w", "root_pose"),
                    ("root_velocity_w", "root_velocity"),
                ):
                    source_key = f"{robot_name}_{source_suffix}"
                    if source_key not in source:
                        raise ValueError(f"native PSM recording is missing {source_key}")
                    _copy_hdf5_dataset(source[source_key], robot_state, target_name)
                    robot_initial.create_dataset(target_name, data=np.asarray(source[source_key][0:1]))

            _write_aligned_room_observations(source, observations, len(actions))
            success = bool(np.asarray(source.get("environment_success", [-1.0]))[-1] > 0.5)
            episode.attrs.update(
                {
                    "num_samples": int(len(actions)),
                    "success": success,
                    "agentic_env_id": state.task,
                    "dr_anmar_action_contract": PSM_POLICY_CONTRACT_NAME,
                    "dr_anmar_policy_action_dim": int(expected_dim),
                    "dr_anmar_cartesian_action_dim": int(cartesian.shape[1]),
                    "dr_anmar_policy_action_semantics": (
                        "seven NVIDIA-native joint-and-gripper values per PSM articulation"
                    ),
                }
            )
            data.attrs["total"] = int(len(actions))
            data.attrs["env_args"] = json.dumps(
                {
                    "env_name": state.task,
                    "type": 2,
                    "sim_args": {"dt": 0.02, "decimation": 1, "render_interval": 1, "num_envs": 1},
                },
                sort_keys=True,
            )
            output.attrs["dr_anmar_action_contract"] = PSM_POLICY_CONTRACT_NAME
            output.flush()
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _copy_hdf5_dataset(source: h5py.Dataset, destination: h5py.Group, name: str) -> h5py.Dataset:
    rows = int(source.shape[0])
    chunks = (min(128, max(1, rows)), *source.shape[1:])
    target = destination.create_dataset(
        name,
        shape=source.shape,
        dtype=source.dtype,
        chunks=chunks,
        compression="lzf",
        shuffle=source.dtype.itemsize > 1,
    )
    for start in range(0, rows, 128):
        target[start : start + 128] = source[start : start + 128]
    return target


def _write_aligned_room_observations(source: h5py.File, observations: h5py.Group, control_frames: int) -> None:
    if "endoscope_rgb" not in source or "endoscope_time_s" not in source:
        return
    control_time = np.asarray(source["time_s"], dtype=np.float64)
    vision_time = np.asarray(source["endoscope_time_s"], dtype=np.float64)
    if not len(vision_time):
        return
    indices = np.searchsorted(vision_time, control_time, side="right") - 1
    indices = np.clip(indices, 0, len(vision_time) - 1)
    rgb = source["endoscope_rgb"]
    room = observations.create_dataset(
        "room",
        shape=(control_frames, *rgb.shape[1:]),
        dtype=rgb.dtype,
        chunks=(min(12, max(1, control_frames)), *rgb.shape[1:]),
        compression="lzf",
    )
    for start in range(0, control_frames, 12):
        selected = indices[start : start + 12]
        room[start : start + len(selected)] = np.stack([rgb[int(index)] for index in selected])


def save_demo(
    state: SharedState,
    capture: BoundedCaptureSpool,
    started_at: str,
) -> str | None:
    if not capture.control_count:
        capture.abort()
        return None
    spool_path = capture.finish()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    task_slug = state.task.lower().replace("isaac-", "").replace("-v0", "").replace("-", "_")
    name = f"dr_anmar_{task_slug}_{stamp}.npz"
    path = state.demo_dir / name
    training_hdf5_path = path.with_suffix(".hdf5") if state.native_psm_policy_contract else None
    control_frame_count = capture.control_count
    vision_frame_count = capture.vision_count
    uncompressed_payload_bytes = capture.payload_bytes
    try:
        with h5py.File(spool_path, "r") as spool:
            arrays = {key: spool[key] for key in spool.keys()}
            analysis = analyze_demo(
                arrays,
                state.task,
                state.arms,
                state.robot_body_names,
                str(state.procedure.get("id", "")),
            )
            write_npz_from_hdf5(path, arrays)
            times = np.asarray(arrays.get("time_s", []), dtype=np.float64).reshape(-1)
            array_shapes = {key: list(value.shape) for key, value in arrays.items()}
            array_keys = set(arrays)
        if training_hdf5_path is not None:
            write_native_psm_training_hdf5(spool_path, training_hdf5_path, state)
    finally:
        spool_path.unlink(missing_ok=True)
    observed_control_hz = 0.0
    if len(times) > 1 and times[-1] > times[0]:
        observed_control_hz = float((len(times) - 1) / (times[-1] - times[0]))
    with state.lock:
        context = {
            "scenario_id": state.scenario_id,
            "scenario_title": SCENARIOS_BY_ID[state.scenario_id]["title"],
            "scenario_seed": state.scenario_seed,
            "autonomy_mode": "supervised_replay" if state.evaluation_source else state.autonomy_mode,
            "intervention_count": state.intervention_count,
            "run_kind": "challenge_evaluation" if state.evaluation_source else "demonstration",
            "evaluation_source": state.evaluation_source,
            "procedure_id": state.procedure.get("id"),
            "procedure_title": state.procedure.get("title"),
            "anatomy_scene_id": state.anatomy_scene_id,
            "anatomy_asset": state.anatomy_asset,
            "final_native_telemetry": json.loads(json.dumps(state.native_telemetry)),
            "expert_demonstration": json.loads(json.dumps(state.expert_demonstration)),
            "expert_controller": EXPERT_CONTROLLER_VERSION if state.expert_demonstration else None,
            "behavior_cloning_reference_candidate": bool(state.expert_clean_run),
            "reference_origin": "simulation_expert" if state.expert_clean_run else "operator_demonstration",
            "reference_review_status": "pending_clinician_review" if state.expert_clean_run else "not_applicable",
        }
        procedure_annotations = list(state.procedure_events)
        camera_intrinsics = state.camera_intrinsics
        semantic_labels = dict(state.semantic_labels)
    manifest = {
        "schema": "dr.anmar.demonstration.v2",
        "simulation_only": True,
        "task": state.task,
        "robots": state.robot_names,
        "robot_body_names": state.robot_body_names,
        "action_dim": state.action_dim,
        "action_contract": ACTION_CONTRACT if state.native_psm_policy_contract else NON_PSM_ACTION_CONTRACT,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "frames": control_frame_count,
        "vision_frames": vision_frame_count,
        "sensor_profile": state.sensor_profile,
        "uncompressed_payload_bytes": uncompressed_payload_bytes,
        "control_hz": round(observed_control_hz, 2),
        "control_hz_nominal": 50,
        "arrays": array_shapes,
        "data_file": name,
        "data_bytes": path.stat().st_size,
        "training_hdf5": training_hdf5_path.name if training_hdf5_path is not None else None,
        "training_hdf5_sha256": sha256_file(training_hdf5_path) if training_hdf5_path is not None else None,
        "modalities": {
            "robot_state_hz": round(observed_control_hz, 2),
            "robot_state_hz_nominal": 50,
            "endoscope_rgb_hz": 5 if vision_frame_count else 0,
            "endoscope_rgb_resolution": [360, 240] if vision_frame_count else None,
            "endoscope_depth_hz": 5 if "endoscope_depth_m" in array_keys else 0,
            "endoscope_depth_units": "metres" if "endoscope_depth_m" in array_keys else None,
            "endoscope_semantic_hz": 5 if "endoscope_semantic_id" in array_keys else 0,
            "endoscope_semantic_encoding": "uint32 semantic id" if "endoscope_semantic_id" in array_keys else None,
            "endoscope_point_cloud_hz": 5 if "endoscope_point_cloud_camera_m" in array_keys else 0,
            "endoscope_point_cloud_frame": "left endoscope camera optical frame",
            "stereo_right_rgb_hz": 5 if "endoscope_right_rgb" in array_keys else 0,
            "instrument_wrist_rgb_hz": 5 if "wrist_1_rgb" in array_keys else 0,
            "instrument_wrist_camera_count": sum(1 for key in ("wrist_1_rgb", "wrist_2_rgb") if key in array_keys),
            "camera_intrinsics": camera_intrinsics,
            "semantic_labels": semantic_labels,
            "simulator_outcome": "environment_reward, termination, truncation, and success when exposed by the task",
            "contact": "maximum force per available contact sensor",
            "deformable_tissue": "nodal displacement, deformation-gradient proxy, and simulator stress when exposed",
            "physics_state": "Native Isaac Lab contact, rigid-body, articulation and deformable telemetry when exposed by the active backend",
            "robot_and_anatomy_pose": "world-frame tool bodies, task objects, and showcase anatomy transform at 50 Hz",
            "joint_torque": "applied and computed joint torque when exposed by the articulation",
            "operator_study": "input source, normalized gaze/attention coordinates, procedure phase, and event codes at 50 Hz",
        },
        "research_safety_advisories": {
            "limits": RESEARCH_ADVISORY_LIMITS,
            "clinical_thresholds_validated": False,
        },
        "context": context,
        "runtime_provenance": state.runtime_provenance or runtime_provenance(state),
        "data_governance": {
            "study_id": STUDY_ID or None,
            "consent_protocol": CONSENT_PROTOCOL or None,
            "external_operator_sensors_enabled": EXTERNAL_OPERATOR_SENSORS_ENABLED,
            "pointer_gaze_is_attention_proxy": True,
            "patient_data_expected": False,
            "retention_requires_study_protocol": True,
        },
        "procedure_annotations": procedure_annotations,
        "annotation_vocabulary": {
            "procedure_phases": PROCEDURE_PHASES,
            "procedure_events": PROCEDURE_EVENTS,
            "operator_input_sources": OPERATOR_INPUT_SOURCES,
            "gaze_sources": {"none": 0, "pointer_attention_proxy": 1, "external_eye_tracker": 2, "xr_eye_tracking": 3},
        },
        "analysis": analysis,
        "data_sha256": sha256_file(path),
    }
    manifest_path = path.with_suffix(".json")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return name


def scenario_camera_pose(base_eye: np.ndarray, base_target: np.ndarray, scenario_id: str) -> tuple[np.ndarray, np.ndarray]:
    eye = base_eye.copy()
    target = base_target.copy()
    if scenario_id in {"camera_shift", "combined_visual"}:
        eye += np.asarray((0.055, -0.045, 0.035), dtype=np.float32)
        target += np.asarray((-0.015, 0.020, 0.010), dtype=np.float32)
    return eye, target


def camera_view_pose(eye: np.ndarray, target: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Move the existing stereo pair without adding another rendered sensor."""
    view_vector = eye - target
    distance = max(float(np.linalg.norm(view_vector)), 0.20)
    if mode == "close":
        eye = target + view_vector * 0.68
    elif mode == "overview":
        eye = target + view_vector * 1.48 + np.asarray((0.0, 0.0, 0.11), dtype=np.float32)
        target = target + np.asarray((0.0, 0.0, 0.015), dtype=np.float32)
    elif mode == "overhead":
        # A slight fore-aft offset avoids the look-at singularity of a perfectly
        # vertical camera while keeping the operative field centered.
        eye = target + np.asarray((0.0, -0.12 * distance, 1.05 * distance), dtype=np.float32)
    elif mode in {"left_oblique", "right_oblique"}:
        angle = np.deg2rad(58.0 if mode == "left_oblique" else -58.0)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotated = np.asarray(
            (
                cosine * view_vector[0] - sine * view_vector[1],
                sine * view_vector[0] + cosine * view_vector[1],
                max(view_vector[2] * 0.92, 0.18 * distance),
            ),
            dtype=np.float32,
        )
        eye = target + rotated
    elif mode == "opposite":
        eye = target - view_vector * 0.92 + np.asarray((0.0, 0.0, 0.32 * distance), dtype=np.float32)
        target = target + np.asarray((0.0, 0.0, 0.02 * distance), dtype=np.float32)
    return eye.astype(np.float32), target.astype(np.float32)


def rotate_camera_vector(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-6 or abs(angle_deg) < 1.0e-6:
        return vector.copy()
    unit_axis = axis / axis_norm
    angle = np.deg2rad(angle_deg)
    return (
        vector * np.cos(angle)
        + np.cross(unit_axis, vector) * np.sin(angle)
        + unit_axis * float(np.dot(unit_axis, vector)) * (1.0 - np.cos(angle))
    ).astype(np.float32)


def adjustable_camera_pose(
    eye: np.ndarray,
    target: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    zoom: float,
    pan_x_m: float,
    pan_y_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply bounded clinician camera controls to an Isaac camera pose."""
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    view_vector = rotate_camera_vector(eye - target, world_up, yaw_deg)
    forward = -view_vector / max(float(np.linalg.norm(view_vector)), 1.0e-6)
    right = np.cross(forward, world_up).astype(np.float32)
    if float(np.linalg.norm(right)) < 1.0e-5:
        right = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    view_vector = rotate_camera_vector(view_vector, right, pitch_deg) * float(zoom)
    forward = -view_vector / max(float(np.linalg.norm(view_vector)), 1.0e-6)
    right = np.cross(forward, world_up).astype(np.float32)
    if float(np.linalg.norm(right)) < 1.0e-5:
        right = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    right /= max(float(np.linalg.norm(right)), 1.0e-6)
    camera_up = np.cross(right, forward).astype(np.float32)
    camera_up /= max(float(np.linalg.norm(camera_up)), 1.0e-6)
    pan = right * float(pan_x_m) + camera_up * float(pan_y_m)
    adjusted_target = target + pan
    adjusted_eye = adjusted_target + view_vector
    return adjusted_eye.astype(np.float32), adjusted_target.astype(np.float32)


def scalar_value(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, torch.Tensor):
            return float(value.detach().reshape(-1)[0].cpu().item())
        array = np.asarray(value)
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError, IndexError, AttributeError):
        return default


def native_success_from_info(info: Any) -> float:
    if not isinstance(info, dict):
        return -1.0
    for key in ("success", "is_success", "task_success", "successes"):
        if key in info:
            return 1.0 if scalar_value(info[key]) > 0.5 else 0.0
    for nested_key in ("log", "episode", "metrics"):
        nested = info.get(nested_key)
        if isinstance(nested, dict):
            result = native_success_from_info(nested)
            if result >= 0.0:
                return result
    return -1.0


def procedure_waypoints(procedure: dict[str, Any]) -> np.ndarray:
    """Return ordered world-space guidance points in the native PSM workspace."""
    authored = procedure.get("waypoints")
    if authored:
        points = np.asarray(authored, dtype=np.float32).reshape(-1, 3)
        if len(points):
            return points
    kind = procedure.get("guide_kind")
    if kind == "navigation":
        return np.asarray(
            ((-0.045, -0.030, 0.070), (-0.015, -0.005, 0.052), (0.018, 0.020, 0.046), (0.050, 0.036, 0.060)),
            dtype=np.float32,
        )
    return np.zeros((0, 3), dtype=np.float32)


def preferred_tool_position(robots: dict[str, Any], body_names: dict[str, list[str]]) -> np.ndarray | None:
    preferred = ("psm_tool_tip_link", "endo360_needle", "ecm_end_link", "tool_tip", "end_effector")
    for name, robot in robots.items():
        names = body_names.get(name, [])
        index = next((names.index(candidate) for candidate in preferred if candidate in names), None)
        if index is None:
            continue
        try:
            return robot.data.body_pos_w[0, index, :3].detach().cpu().numpy().astype(np.float32)
        except (AttributeError, IndexError, RuntimeError):
            continue
    return None


def main() -> None:
    args_cli.demo_dir.mkdir(parents=True, exist_ok=True)
    procedure = dict(PROCEDURES_BY_ID.get(args_cli.procedure, {}))
    if args_cli.procedure and not procedure:
        raise ValueError(f"Unknown Dr.Anmar procedure room: {args_cli.procedure}")
    dr_anmar_parametric_needle_enabled = bool(
        procedure.get("dr_anmar_needle_asset")
    )
    suture_physics_lod = os.environ.get(
        "DR_ANMAR_SUTURE_PHYSICS_LOD",
        str(procedure.get("suture_physics_lod", "full_360")),
    )
    suture_native_segment_rendering = bool(
        procedure.get("suture_native_segment_rendering")
    )
    single_active_camera_renderer = bool(
        procedure.get("single_active_camera_renderer", True)
    )
    nvidia_native_bench = bool(procedure.get("nvidia_native_bench"))
    dynamic_abdominal_patient_enabled = bool(
        procedure.get("dynamic_abdominal_patient")
    )
    autonomous_rescue_or_enabled = bool(
        procedure.get("autonomous_rescue_or")
    )
    if dynamic_abdominal_patient_enabled and autonomous_rescue_or_enabled:
        raise ValueError(
            "Select one contact-driven patient substrate per procedure room"
        )
    contact_driven_patient_effects_enabled = bool(
        dynamic_abdominal_patient_enabled or autonomous_rescue_or_enabled
    )
    stapler_test_cell_enabled = bool(procedure.get("stapler_test_cell"))
    selected_bench_assets: set[str] = set()
    bench_asset_paths: dict[str, Path] = {}
    featured_robot_system_id: str | None = None
    featured_robot_system_paths: dict[str, Path] = {}
    if nvidia_native_bench:
        bench_catalog = tuple(procedure.get("bench_asset_catalog", ()))
        allowed_bench_assets = {str(item["id"]) for item in bench_catalog}
        if args_cli.bench_assets == "default":
            selected_bench_assets = {
                str(item["id"]) for item in bench_catalog if item.get("default")
            }
        elif args_cli.bench_assets == "none":
            selected_bench_assets = set()
        else:
            selected_bench_assets = {
                item.strip() for item in args_cli.bench_assets.split(",") if item.strip()
            }
        unknown_bench_assets = sorted(selected_bench_assets - allowed_bench_assets)
        if unknown_bench_assets:
            raise ValueError(
                "Unknown operating-room bench assets: "
                + ", ".join(unknown_bench_assets)
            )
        featured_robot_system_id = resolve_featured_robot_system(
            selected_bench_assets
        )
        procedure["active_bench_assets"] = [
            str(item["id"])
            for item in bench_catalog
            if str(item["id"]) in selected_bench_assets
        ]
        procedure["featured_robot_system"] = featured_robot_system_id
        bench_asset_provider_roots = provider_roots(
            REPOSITORY_ROOT,
            i4h_content_root=I4H_ASSET_CONTENT_ROOT,
        )
        core_bench_assets = {
            "psm": resolve_provider_asset(
                "nvidia_i4h",
                "Robots/dVRK/PSM/psm.usd",
                bench_asset_provider_roots,
            ),
            "needle_runtime": resolve_provider_asset(
                "nvidia_i4h",
                "Props/SutureNeedle/needle_sdf.usd",
                bench_asset_provider_roots,
            ),
            "table": resolve_provider_asset(
                "nvidia_i4h",
                "Props/Table/table.usd",
                bench_asset_provider_roots,
            ),
        }
        unknown_bench_providers = sorted(
            {
                str(item.get("provider", "nvidia_i4h"))
                for item in bench_catalog
                if str(item["id"]) in selected_bench_assets
            }
            - bench_asset_provider_roots.keys()
        )
        if unknown_bench_providers:
            raise ValueError(
                "Unknown operating-room asset providers: "
                + ", ".join(unknown_bench_providers)
            )
        bench_asset_paths = {
            **core_bench_assets,
            **{
                str(item["id"]): resolve_provider_asset(
                    str(item.get("provider", "nvidia_i4h")),
                    str(item["path"]),
                    bench_asset_provider_roots,
                )
                for item in bench_catalog
                if str(item["id"]) in selected_bench_assets
            },
        }
        if featured_robot_system_id is not None:
            featured_robot_spec = BENCH_ROBOT_SYSTEMS_BY_ID[
                featured_robot_system_id
            ]
            featured_robot_system_paths = {
                "standalone": bench_asset_paths[featured_robot_system_id],
                **{
                    key.removesuffix("_path"): resolve_provider_asset(
                        str(
                            featured_robot_spec.get(
                                "provider",
                                "nvidia_i4h",
                            )
                        ),
                        str(featured_robot_spec[key]),
                        bench_asset_provider_roots,
                    )
                    for key in (
                        "payload_path",
                        "rigid_proxy_path",
                        "auxiliary_path",
                    )
                },
            }
        missing_bench_assets = [
            f"{name}: {path}"
            for name, path in bench_asset_paths.items()
            if not path.is_file()
        ]
        missing_bench_assets.extend(
            f"{featured_robot_system_id}.{name}: {path}"
            for name, path in featured_robot_system_paths.items()
            if name != "standalone" and not path.is_file()
        )
        if missing_bench_assets:
            raise RuntimeError(
                "The operating-room bench is missing required assets: "
                + "; ".join(missing_bench_assets)
            )
    dynamic_abdominal_patient_path = (
        REPOSITORY_ROOT
        / "source/extensions/orbit.surgical.assets/data/Props/Patients"
        / "DynamicAbdominalPatient/dranmar_dynamic_abdominal_patient.usda"
    )
    if (
        dynamic_abdominal_patient_enabled
        and not dynamic_abdominal_patient_path.is_file()
    ):
        raise RuntimeError(
            "The dynamic abdominal patient room is missing its primary asset: "
            f"{dynamic_abdominal_patient_path}"
        )
    autonomous_rescue_vessel_path = (
        REPOSITORY_ROOT
        / "source/extensions/orbit.surgical.assets/data/Environments"
        / "SurgicalAutonomy/AutonomousRescueOR/dranmar_rescue_vessel.usda"
    )
    if (
        autonomous_rescue_or_enabled
        and not autonomous_rescue_vessel_path.is_file()
    ):
        raise RuntimeError(
            "The Autonomous Rescue OR is missing its live vessel substrate: "
            f"{autonomous_rescue_vessel_path}"
        )
    bench_dr_anmar_suture_enabled = bool(
        nvidia_native_bench
        and "dr_anmar_needle_suture" in selected_bench_assets
    )
    skin_adhesive_enabled = bool(
        nvidia_native_bench
        and "skin_adhesive_system" in selected_bench_assets
    )
    closure_robot_enabled = bool(
        nvidia_native_bench
        and "approximate_staple_seal_robot" in selected_bench_assets
    )
    skin_adhesive_paths: dict[str, Path] = {}
    if skin_adhesive_enabled:
        skin_adhesive_paths = {
            "applicator": bench_asset_paths["skin_adhesive_system"],
            "mounted_psm": REPOSITORY_ROOT
            / "source/extensions/orbit.surgical.assets/data/Robots/dVRK/PSM"
            / "psm_skin_adhesive.usda",
        }
        missing_skin_adhesive_assets = [
            f"{name}: {path}"
            for name, path in skin_adhesive_paths.items()
            if not path.is_file()
        ]
        if missing_skin_adhesive_assets:
            raise RuntimeError(
                "The Dr.Anmar topical skin-adhesive system is incomplete: "
                + "; ".join(missing_skin_adhesive_assets)
            )
    closure_robot_asset_root = (
        REPOSITORY_ROOT
        / "source/extensions/orbit.surgical.assets/data/Props/"
        "SurgicalClosure/ClosureRobot"
    )
    closure_robot_paths: dict[str, Path] = {}
    if closure_robot_enabled:
        closure_robot_paths = {
            "payload": bench_asset_paths["approximate_staple_seal_robot"],
            "standalone": closure_robot_asset_root
            / "dranmar_closure_tool_standalone.usda",
            "tissue": closure_robot_asset_root
            / "dranmar_closure_tissue_demo.usda",
            "staple": closure_robot_asset_root
            / "dranmar_closure_staple.usda",
            "adhesive_bead": closure_robot_asset_root
            / "dranmar_closure_adhesive_bead.usda",
            "physics_profile": closure_robot_asset_root
            / "physics_profile.json",
            "mount_contract": closure_robot_asset_root
            / "franka_mount_contract.json",
        }
        missing_closure_robot_assets = [
            f"{name}: {path}"
            for name, path in closure_robot_paths.items()
            if not path.is_file()
        ]
        if missing_closure_robot_assets:
            raise RuntimeError(
                "The Dr.Anmar approximate–staple–seal robot is incomplete: "
                + "; ".join(missing_closure_robot_assets)
            )
    stapler_test_cell_paths = {
        "fixture": REPOSITORY_ROOT
        / "source/extensions/orbit.surgical.assets/data/Props/"
        "SurgicalClosure/StaplerTestCell/stapler_test_fixture.usda",
        "device": REPOSITORY_ROOT
        / "source/extensions/orbit.surgical.assets/data/Props/"
        "SurgicalClosure/StaplerTestCell/stapler_test_device.usda",
        "tissue_left": REPOSITORY_ROOT
        / "assets/dr_anmar/tissue/DrAnmarSuturableTissue.left.usda",
        "tissue_right": REPOSITORY_ROOT
        / "assets/dr_anmar/tissue/DrAnmarSuturableTissue.right.usda",
        "tissue_profile": REPOSITORY_ROOT
        / "physics_next/tissues/dr-anmar-suturable-tissue-v1.json",
    }
    stapler_tissue_material_runtime: dict[str, float | str] = {}
    stapler_tissue_episode_payload: dict[str, float | int] = {}
    if stapler_test_cell_enabled:
        missing_test_cell_assets = [
            f"{name}: {path}"
            for name, path in stapler_test_cell_paths.items()
            if not path.is_file()
        ]
        if missing_test_cell_assets:
            raise RuntimeError(
                "The Dr.Anmar stapler test cell is incomplete: "
                + "; ".join(missing_test_cell_assets)
            )
        stapler_tissue_profile = json.loads(
            stapler_test_cell_paths["tissue_profile"].read_text(
                encoding="utf-8"
            )
        )
        stapler_tissue_episode = sample_tissue_episode_parameters(
            stapler_tissue_profile,
            DEFAULT_SCENARIO_SEED,
        )
        stapler_tissue_episode_payload = (
            stapler_tissue_episode.payload()
        )
        stapler_tissue_proxy = stable_physx_proxy_parameters(
            stapler_tissue_profile,
            stapler_tissue_episode,
        )
        stapler_tissue_material_runtime = {
            "density_kg_m3": float(
                stapler_tissue_proxy["density_kg_m3"]
            ),
            "dynamic_friction": float(
                stapler_tissue_proxy["dynamic_friction"]
            ),
            "youngs_modulus_pa": float(
                min(
                    120000.0,
                    float(
                        stapler_tissue_proxy[
                            "youngs_modulus_pa"
                        ]
                    ),
                )
            ),
            "poisson_ratio": float(
                min(
                    0.40,
                    float(
                        stapler_tissue_proxy["poisson_ratio"]
                    ),
                )
            ),
            "vertex_velocity_damping": min(
                1.0,
                max(
                    0.05,
                    float(
                        stapler_tissue_proxy["damping_ratio"]
                    ),
                ),
            ),
            "solver_position_iterations": float(
                max(
                    32,
                    int(
                        stapler_tissue_profile["solver"][
                            "position_iterations"
                        ]
                    ),
                )
            ),
            "target_youngs_modulus_pa": float(
                stapler_tissue_proxy["youngs_modulus_pa"]
            ),
            "target_poisson_ratio": float(
                stapler_tissue_proxy["poisson_ratio"]
            ),
            "stability_proxy": (
                "bounded_linear_tangent_for_interactive_physx"
            ),
        }
    nvidia_needle_dr_anmar_suture_enabled = bool(
        nvidia_native_bench
        and "nvidia_needle_dr_anmar_suture" in selected_bench_assets
    )
    if (
        nvidia_needle_dr_anmar_suture_enabled
        and "DR_ANMAR_SUTURE_PHYSICS_LOD" not in os.environ
        and "suture_physics_lod" not in procedure
    ):
        # Preserve the authored 4-0 dimensions and joint behavior while using
        # the real-time discretization intended for camera teleoperation.
        suture_physics_lod = "interactive_90"
    dr_anmar_needle_enabled = bool(
        dr_anmar_parametric_needle_enabled
        or nvidia_needle_dr_anmar_suture_enabled
    )
    if "-IK-Rel" not in args_cli.task:
        raise ValueError("The browser workstation accepts relative-IK tasks. Other variants remain available via the CLI.")
    guide_kind = str(procedure.get("guide_kind", ""))
    bimanual_softmimicgen = bool(_softmimicgen_task and procedure.get("bimanual"))
    softmimicgen_goal = None
    if _softmimicgen_task:
        from softmimicgen_tasks.surgical_threading.mdp import object_reached_goal

        softmimicgen_goal = object_reached_goal
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        # Keep every interactive room on Isaac Lab's native Fabric transform
        # path. The authored suture is rendered from its PhysX tensor poses
        # below, so custom assets never disable live robot articulation.
        use_fabric=not args_cli.disable_fabric,
    )
    # Isaac Lab otherwise shares /tmp/isaaclab across workstation accounts.
    # Keep each Dr.Anmar installation's logs inside its configured writable
    # data root so Numi qualification cannot collide with another account.
    isaaclab_log_dir = DATA_ROOT / "logs" / "isaaclab"
    isaaclab_log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.sim.log_dir = str(isaaclab_log_dir)
    # Every interactive PSM room inherits its control/physics cadence from the
    # released ORBIT needle-handover configuration that already works in the
    # NVIDIA native bench. Rooms may compose different OpenUSD assets, but
    # they do not author their own PSM timestep, decimation, or jaw actions.
    orbit_psm_foundation = (
        ORBIT_NEEDLE_HANDOVER_CFG()
        if "PSM" in args_cli.task and not _softmimicgen_task
        else None
    )
    if orbit_psm_foundation is not None:
        env_cfg.sim.dt = orbit_psm_foundation.sim.dt
        env_cfg.decimation = orbit_psm_foundation.decimation
        env_cfg.sim.render_interval = orbit_psm_foundation.sim.render_interval
    if stapler_test_cell_enabled or closure_robot_enabled:
        # The imported 6 mm tissue coupon requires the profile's 1 ms FEM
        # cadence. Preserve the PSM action period by increasing decimation
        # while giving the deformable solver the substeps it needs.
        foundation_action_period_s = (
            float(env_cfg.sim.dt) * int(env_cfg.decimation)
        )
        env_cfg.sim.dt = 0.001
        env_cfg.decimation = max(
            1,
            int(round(foundation_action_period_s / env_cfg.sim.dt)),
        )
    microscopic_bench_assets = selected_bench_assets.intersection(
        {
            "dr_anmar_needle",
            "dr_anmar_needle_suture",
            "nvidia_needle_dr_anmar_suture",
            "dr_anmar_needle_v030",
            "dr_anmar_needle_thread_coiled",
            "dr_anmar_needle_thread_extended",
            "dr_anmar_needle_thread_proxy",
        }
    )
    interactive_rendering_mode = procedure.get("interactive_rendering_mode")
    if interactive_rendering_mode:
        # Use Isaac Lab's official real-time RTX preset for the doctor-facing
        # camera. It preserves the scene, shadows, materials, native sensors,
        # and all PhysX state while disabling expensive sampled-lighting and
        # denoising features intended for offline-quality output.
        env_cfg.sim.render.rendering_mode = str(interactive_rendering_mode)
    suture_profile = load_suture_profile()
    interactive_camera_period_s = float(
        procedure.get("interactive_camera_update_period_s", 0.04)
    )
    env_cfg.sim.render_interval = max(
        env_cfg.decimation,
        int(round(interactive_camera_period_s / env_cfg.sim.dt)),
    )
    if hasattr(env_cfg.sim.physx, "enable_external_forces_every_iteration"):
        env_cfg.sim.physx.enable_external_forces_every_iteration = True
    if microscopic_bench_assets:
        # Isaac Lab's global PhysX CCD switch must be enabled in addition to
        # the rigid-body CCD schemas authored on surgical-scale assets.
        # Without the scene switch, the sub-millimetre needle can tunnel
        # through the NVIDIA table even though its USD requests CCD.
        env_cfg.sim.physx.enable_ccd = True
    if nvidia_native_bench:
        # Compose the room exclusively from the pinned Isaac for Healthcare
        # catalog. The existing handover task continues to own both PSMs,
        # relative IK actions, jaw contacts, needle state, resets and stepping.
        # Coordinates retain NVIDIA/ORBIT's opposed PSM roots, but this
        # clinician-facing bench narrows their separation so both neutral
        # tools begin inside a practical shared handoff workspace. The table
        # top (z=0) remains the single setup datum.
        psm_root_spacing_m = float(procedure.get("psm_root_spacing_m", 0.40))
        if not 0.12 <= psm_root_spacing_m <= 0.40:
            raise ValueError(
                "NVIDIA bench psm_root_spacing_m must be between 0.12 and 0.40 m"
            )
        psm_root_half_spacing_m = psm_root_spacing_m / 2.0
        psm_root_height_m = float(
            procedure.get("psm_root_height_m", 0.15)
        )
        if not 0.12 <= psm_root_height_m <= 0.40:
            raise ValueError(
                "NVIDIA bench psm_root_height_m must be between 0.12 and 0.40 m"
            )
        psm_root_positions = {
            "robot_1": (
                psm_root_half_spacing_m,
                0.0,
                psm_root_height_m,
            ),
            "robot_2": (
                -psm_root_half_spacing_m,
                0.0,
                psm_root_height_m,
            ),
        }
        for robot_name, root_position in psm_root_positions.items():
            robot_cfg = getattr(env_cfg.scene, robot_name)
            mounted_adhesive_arm = bool(
                skin_adhesive_enabled and robot_name == "robot_1"
            )
            robot_cfg.spawn.usd_path = str(
                skin_adhesive_paths["mounted_psm"]
                if mounted_adhesive_arm
                else bench_asset_paths["psm"]
            )
            if mounted_adhesive_arm:
                adhesive_mechanism_cfg = (
                    make_articulated_skin_adhesive_cfg(
                        state="activated",
                        usd_path=skin_adhesive_paths["applicator"],
                    )
                )
                # The standalone hand-held asset uses intentionally compliant
                # paddle drives. Once mounted sideways on the PSM, that
                # compliance lets gravity bias the two paddle angles away from
                # the proportional dispense command. Keep the authored effort
                # limit, but make the mounted drives stiff enough to hold the
                # requested physical angle without turning them kinematic.
                mounted_paddle_actuator = (
                    adhesive_mechanism_cfg.actuators["paddles"]
                )
                mounted_paddle_actuator.stiffness = 8.0
                mounted_paddle_actuator.damping = 0.18
                robot_cfg.actuators = {
                    **robot_cfg.actuators,
                    "skin_adhesive_paddles": (
                        mounted_paddle_actuator
                    ),
                    "skin_adhesive_metering_piston": (
                        adhesive_mechanism_cfg.actuators[
                            "metering_piston"
                        ]
                    ),
                }
                robot_cfg.init_state.joint_pos.update(
                    {
                        "left_paddle_joint": 0.0,
                        "right_paddle_joint": 0.0,
                        "metering_piston_joint": 0.0,
                    }
                )
            robot_cfg.init_state.pos = root_position
            robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        if closure_robot_enabled:
            env_cfg.scene.replicate_physics = False
            closure_robot_cfg = make_franka_closure_robot_cfg(
                prim_path="{ENV_REGEX_NS}/ClosureRobot",
                staple_state="loaded",
                adhesive_state="full",
            )
            # The standard Franka ready pose places link8 over the shared
            # operative center. The tissue spawn below derives its setup
            # transform from the authored closure TCP before PhysX starts.
            # Offset Isaac 5.1's ready-pose link8 so the diagonal closure axis
            # lands the authored TCP on the shared operative target rather
            # than below the table or outside the endoscope view.
            closure_robot_cfg.init_state.pos = (-0.499, 0.22, -0.045)
            closure_robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
            env_cfg.scene.closure_robot = closure_robot_cfg

            closure_tissue_spawn = sim_utils.UsdFileCfg(
                usd_path=str(closure_robot_paths["tissue"]),
            )
            source_closure_tissue_spawn = closure_tissue_spawn.func

            def spawn_closure_tissue_at_tool(
                prim_path: str,
                cfg: sim_utils.UsdFileCfg,
                translation=None,
                orientation=None,
                **kwargs: Any,
            ) -> Any:
                tissue_root = source_closure_tissue_spawn(
                    prim_path,
                    cfg,
                    translation=translation,
                    orientation=orientation,
                    **kwargs,
                )
                import omni.usd
                from pxr import UsdGeom

                stage = omni.usd.get_context().get_stage()
                resolved_root_path = str(tissue_root.GetPath())
                environment_path = resolved_root_path.rsplit("/", 1)[0]
                closure_tcp_path = (
                    f"{environment_path}/ClosureRobot/"
                    "DrAnmarClosureTool/Links/Mount/Frames/closure_tcp"
                )
                closure_tcp = stage.GetPrimAtPath(closure_tcp_path)
                if not closure_tcp.IsValid():
                    raise RuntimeError(
                        "The composed Franka closure robot has no authored closure_tcp"
                    )
                cache = UsdGeom.XformCache()
                environment_prim = tissue_root.GetParent()
                closure_tcp_world = cache.GetLocalToWorldTransform(closure_tcp)
                environment_world = cache.GetLocalToWorldTransform(
                    environment_prim
                )
                tissue_local = closure_tcp_world * environment_world.GetInverse()
                tissue_xform = UsdGeom.Xformable(tissue_root)
                tissue_xform.ClearXformOpOrder()
                tissue_xform.AddTransformOp().Set(tissue_local)
                if not sim_utils.standardize_xform_ops(tissue_root):
                    raise RuntimeError(
                        "The closure tissue root could not be converted to "
                        "Isaac Lab's canonical transform stack"
                    )

                tissue_info = apply_tissue_demo_surface_deformables(
                    resolved_root_path
                )
                anchor_tissue_outer_edges(
                    stage,
                    tissue_root_path=resolved_root_path,
                    left_tissue_path=str(
                        tissue_info["left_tissue_path"]
                    ),
                    right_tissue_path=str(
                        tissue_info["right_tissue_path"]
                    ),
                )
                return tissue_root

            closure_tissue_spawn.func = spawn_closure_tissue_at_tool
            env_cfg.scene.closure_tissue = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/ClosureTissue",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                spawn=closure_tissue_spawn,
            )
        if featured_robot_system_id is not None:
            # The large Dr.Anmar systems share one featured station so the
            # bench remains readable and GPU-bounded. Each selection composes
            # the real standalone articulation and its authored task substrate;
            # the payload and rigid planning proxy are validated with the same
            # catalog contract but are not duplicated into the live station.
            env_cfg.scene.replicate_physics = False
            featured_robot_cfg = BENCH_ROBOT_SYSTEM_FACTORIES[
                featured_robot_system_id
            ](
                prim_path="{ENV_REGEX_NS}/FeaturedRobotSystem",
                position=FEATURED_ROBOT_POSITION_M,
            )
            featured_robot_cfg.spawn.usd_path = str(
                featured_robot_system_paths["standalone"]
            )
            env_cfg.scene.featured_robot_system = featured_robot_cfg
            env_cfg.scene.featured_robot_substrate = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/FeaturedRobotSubstrate",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=FEATURED_SUBSTRATE_POSITION_M,
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(featured_robot_system_paths["auxiliary"]),
                ),
            )
        if dynamic_abdominal_patient_enabled:
            env_cfg.scene.replicate_physics = False
            # Keep bounded CUDA headroom for the single permitted deformable
            # lane and external tool contacts. Multi-component deformable
            # contact is intentionally rejected below until it has native
            # evidence on the target PhysX/CUDA stack.
            dynamic_patient_physx = {
                "gpu_collision_stack_size": 2**30,
                "gpu_heap_capacity": 2**28,
                "gpu_temp_buffer_capacity": 2**26,
                "gpu_max_soft_body_contacts": 2**20,
            }
            for setting, value in dynamic_patient_physx.items():
                if hasattr(env_cfg.sim.physx, setting):
                    setattr(env_cfg.sim.physx, setting, value)
            dynamic_patient_active_deformables = tuple(
                str(component)
                for component in procedure.get(
                    "dynamic_patient_active_deformables", ()
                )
            )
            if len(dynamic_patient_active_deformables) != 1:
                raise ValueError(
                    "The current dynamic patient safety boundary requires exactly one "
                    "explicitly selected solver-active deformable component"
                )
            patient_access_state = str(
                procedure.get("dynamic_patient_access_state", "open")
            )
            if patient_access_state not in {"intact", "open"}:
                raise ValueError(
                    "dynamic_patient_access_state must be 'intact' or 'open'"
                )
            dynamic_patient_spawn = sim_utils.UsdFileCfg(
                usd_path=str(dynamic_abdominal_patient_path),
                variants={"access_state": patient_access_state},
            )
            source_dynamic_patient_spawn = dynamic_patient_spawn.func

            def spawn_dynamic_abdominal_patient(
                prim_path: str,
                cfg: sim_utils.UsdFileCfg,
                translation=None,
                orientation=None,
                **kwargs: Any,
            ) -> Any:
                patient_root = source_dynamic_patient_spawn(
                    prim_path,
                    cfg,
                    translation=translation,
                    orientation=orientation,
                    **kwargs,
                )
                from orbit.surgical.assets.dynamic_abdominal_patient import (
                    NATIVE_DEFORMABLE_ROUTES,
                    apply_patient_deformables,
                )

                mechanics_routes = apply_patient_deformables(
                    str(patient_root.GetPath()),
                    include=dynamic_patient_active_deformables,
                )
                failed_routes = {
                    component: result
                    for component, result in mechanics_routes.items()
                    if result["route"] not in NATIVE_DEFORMABLE_ROUTES
                }
                if failed_routes:
                    raise RuntimeError(
                        "Dynamic abdominal patient mechanics failed closed: "
                        f"{failed_routes}"
                    )
                collision_filter = {
                    "patient_path": str(patient_root.GetPath()),
                    "policy": "not_required_for_single_active_deformable",
                }
                print(
                    "[DR_ANMAR_DYNAMIC_PATIENT_MECHANICS] "
                    + json.dumps(
                        {
                            "active_deformables": dynamic_patient_active_deformables,
                            "collision_filter": collision_filter,
                            "routes": mechanics_routes,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return patient_root

            dynamic_patient_spawn.func = spawn_dynamic_abdominal_patient
            dynamic_patient_position_raw = tuple(
                procedure.get(
                    "dynamic_patient_position_m", (0.0, 0.0, 0.0)
                )
            )
            if len(dynamic_patient_position_raw) != 3:
                raise ValueError(
                    "dynamic_patient_position_m must contain exactly three values"
                )
            dynamic_patient_position = tuple(
                float(value) for value in dynamic_patient_position_raw
            )
            if not all(math.isfinite(value) for value in dynamic_patient_position):
                raise ValueError(
                    "dynamic_patient_position_m values must all be finite"
                )
            env_cfg.scene.dynamic_abdominal_patient = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/DynamicAbdominalPatient",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=dynamic_patient_position,
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                spawn=dynamic_patient_spawn,
            )
        if autonomous_rescue_or_enabled:
            env_cfg.scene.replicate_physics = False
            for setting, value in {
                "gpu_collision_stack_size": 2**30,
                "gpu_heap_capacity": 2**28,
                "gpu_temp_buffer_capacity": 2**26,
                "gpu_max_soft_body_contacts": 2**20,
            }.items():
                if hasattr(env_cfg.sim.physx, setting):
                    setattr(env_cfg.sim.physx, setting, value)
            rescue_vessel_position_raw = tuple(
                procedure.get(
                    "rescue_vessel_position_m",
                    (0.0, 0.0, 0.055),
                )
            )
            if len(rescue_vessel_position_raw) != 3:
                raise ValueError(
                    "rescue_vessel_position_m must contain exactly three values"
                )
            rescue_vessel_position = tuple(
                float(value) for value in rescue_vessel_position_raw
            )
            if not all(
                math.isfinite(value) for value in rescue_vessel_position
            ):
                raise ValueError(
                    "rescue_vessel_position_m values must all be finite"
                )
            env_cfg.scene.autonomous_rescue_vessel = rescue_vessel_cfg(
                position=rescue_vessel_position,
            )
        env_cfg.scene.table.spawn.usd_path = str(bench_asset_paths["table"])
        env_cfg.scene.table.init_state.pos = (0.0, 0.0, -0.457)
        env_cfg.scene.object.spawn.usd_path = str(bench_asset_paths["needle_runtime"])
        env_cfg.scene.object.spawn.scale = (0.4, 0.4, 0.4)
        env_cfg.scene.object.init_state.pos = (
            (-0.195, 0.015, 0.0008)
            if "needle" in selected_bench_assets
            else (0.0, 0.0, -1.5)
        )
        env_cfg.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        if getattr(env_cfg, "events", None) is not None:
            # The upstream handover randomizer targets a broad bare table.
            # This bench uses a fixed, visible sterile landing on the table.
            env_cfg.events.reset_object_position = None

        if stapler_test_cell_enabled:
            env_cfg.scene.replicate_physics = False
            for setting, value in {
                "gpu_max_soft_body_contacts": 2**17,
                "gpu_max_particle_contacts": 2**14,
                "gpu_heap_capacity": 2**24,
                "gpu_temp_buffer_capacity": 2**22,
            }.items():
                if hasattr(env_cfg.sim.physx, setting):
                    setattr(env_cfg.sim.physx, setting, value)
            env_cfg.scene.stapler_test_fixture = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/StaplerTestFixture",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(stapler_test_cell_paths["fixture"]),
                ),
            )
            env_cfg.scene.stapler_test_device = (
                make_articulated_skin_stapler_cfg(
                    prim_path="{ENV_REGEX_NS}/StaplerTestDevice",
                    state="loaded",
                    usd_path=stapler_test_cell_paths["device"],
                    disable_gravity=True,
                )
            )
            stapler_trigger_actuator = (
                env_cfg.scene.stapler_test_device.actuators["trigger"]
            )
            stapler_trigger_actuator.effort_limit_sim = 6.0
            stapler_trigger_actuator.stiffness = 8.0
            stapler_trigger_actuator.damping = 0.40
            # The fixture runtime holds this authored root datum while the
            # trigger and pusher remain dynamic articulation coordinates.
            env_cfg.scene.stapler_test_device.init_state.pos = (
                STAPLER_CLOSURE_STATION_OFFSETS_M[0],
                0.0,
                STAPLER_TEST_DEVICE_MOUNT_Z_M,
            )
            env_cfg.scene.stapler_test_device.init_state.rot = (
                1.0,
                0.0,
                0.0,
                0.0,
            )
            # PhysX cooks the two disconnected watertight flaps into a
            # volumetric deformable. The local +90 degree rotation makes the
            # incision follow the fixture's X rail while staple crowns bridge
            # the gap along Y.
            def make_stapler_tissue_flap_cfg(
                *,
                usd_path: Path,
                prim_name: str,
            ) -> DeformableObjectCfg:
                tissue_spawn = sim_utils.UsdFileCfg(
                    usd_path=str(usd_path),
                    deformable_props=(
                        sim_utils.DeformableBodyPropertiesCfg(
                            deformable_enabled=True,
                            kinematic_enabled=True,
                            self_collision=False,
                            solver_position_iteration_count=int(
                                stapler_tissue_material_runtime[
                                    "solver_position_iterations"
                                ]
                            ),
                            vertex_velocity_damping=float(
                                stapler_tissue_material_runtime[
                                    "vertex_velocity_damping"
                                ]
                            ),
                            sleep_damping=0.0,
                            sleep_threshold=0.0,
                            settling_threshold=0.0,
                            simulation_hexahedral_resolution=12,
                            contact_offset=0.0005,
                            rest_offset=0.0001,
                            max_depenetration_velocity=0.10,
                        )
                    ),
                )
                source_spawn = tissue_spawn.func

                def spawn_tissue_flap(
                    prim_path: str,
                    cfg: sim_utils.UsdFileCfg,
                    translation=None,
                    orientation=None,
                    **kwargs: Any,
                ) -> Any:
                    deformable_props = cfg.deformable_props
                    source_cfg = cfg.replace(deformable_props=None)
                    root_prim = source_spawn(
                        prim_path,
                        source_cfg,
                        translation=translation,
                        orientation=orientation,
                        **kwargs,
                    )
                    root_path = str(root_prim.GetPath())
                    if root_prim.GetTypeName() == "Mesh":
                        PhysxSchema.PhysxDeformableBodyAPI.Apply(
                            root_prim
                        )
                        sim_utils.modify_deformable_body_properties(
                            root_path,
                            deformable_props,
                        )
                    else:
                        sim_utils.define_deformable_body_properties(
                            root_path,
                            deformable_props,
                        )
                    material_path = (
                        f"{root_path}/DrAnmarStaplerTissueMaterial"
                    )
                    material_cfg = (
                        sim_utils.DeformableBodyMaterialCfg(
                            density=float(
                                stapler_tissue_material_runtime[
                                    "density_kg_m3"
                                ]
                            ),
                            dynamic_friction=float(
                                stapler_tissue_material_runtime[
                                    "dynamic_friction"
                                ]
                            ),
                            youngs_modulus=float(
                                stapler_tissue_material_runtime[
                                    "youngs_modulus_pa"
                                ]
                            ),
                            poissons_ratio=float(
                                stapler_tissue_material_runtime[
                                    "poisson_ratio"
                                ]
                            ),
                        )
                    )
                    material_cfg.func(material_path, material_cfg)
                    sim_utils.bind_physics_material(
                        root_path,
                        material_path,
                    )
                    return root_prim

                tissue_spawn.func = spawn_tissue_flap
                return DeformableObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
                    init_state=DeformableObjectCfg.InitialStateCfg(
                        pos=STAPLER_CLOSURE_TISSUE_CENTER_M,
                        rot=STAPLER_CLOSURE_TISSUE_ROTATION_WXYZ,
                    ),
                    spawn=tissue_spawn,
                )

            env_cfg.scene.stapler_closure_tissue_left = (
                make_stapler_tissue_flap_cfg(
                    usd_path=stapler_test_cell_paths["tissue_left"],
                    prim_name="StaplerClosureTissueLeft",
                )
            )
            env_cfg.scene.stapler_closure_tissue_right = (
                make_stapler_tissue_flap_cfg(
                    usd_path=stapler_test_cell_paths["tissue_right"],
                    prim_name="StaplerClosureTissueRight",
                )
            )

        # The pad is NVIDIA-authored static collision geometry. It intentionally
        # remains rigid: this room must never turn contact into a fake puncture.
        if "suture_pad" in selected_bench_assets:
            env_cfg.scene.suture_pad = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/SuturePad",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(0.030, 0.055, 0.0005),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(bench_asset_paths["suture_pad"]),
                ),
            )

        # NVIDIA authors the scissors in centimetres. Preserve its native
        # collider and PhysX response while normalizing it into the metre stage.
        if "scissors" in selected_bench_assets:
            env_cfg.scene.surgical_scissors = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/SurgicalScissors",
                init_state=RigidObjectCfg.InitialStateCfg(
                    # The 190 mm instrument rests on a separate table landing.
                    pos=(-0.135, -0.130, 0.0114),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(bench_asset_paths["scissors"]),
                    scale=(0.01, 0.01, 0.01),
                ),
            )
        if "tray" in selected_bench_assets:
            env_cfg.scene.surgical_tray = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/SurgicalTray",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(0.135, -0.125, 0.001),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(bench_asset_paths["tray"]),
                    scale=(0.01, 0.01, 0.01),
                ),
            )
        if "skin_stapler" in selected_bench_assets:
            env_cfg.scene.skin_stapler = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/SkinStapler",
                init_state=RigidObjectCfg.InitialStateCfg(
                    # Keep the broad-side landing inside the shared PSM field.
                    pos=dr_anmar_asset_landing("skin_stapler"),
                    rot=(0.70710678, 0.70710678, 0.0, 0.0),
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(bench_asset_paths["skin_stapler"]),
                    variants={"state": "loaded"},
                    semantic_tags=[
                        ("class", "skin_stapler"),
                        ("device_type", "surgical_closure_device"),
                        ("workflow_handover", "handover"),
                        ("workflow_closure", "closure"),
                        ("state", "loaded"),
                    ],
                    activate_contact_sensors=True,
                ),
            )
        if "dr_anmar_needle" in selected_bench_assets:
            env_cfg.scene.dr_anmar_standalone_needle = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/DrAnmarStandaloneNeedle",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=dr_anmar_asset_landing("dr_anmar_needle"),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(bench_asset_paths["dr_anmar_needle"]),
                    variants={"Physics": "physx"},
                    activate_contact_sensors=True,
                ),
            )
        if "dr_anmar_needle_v030" in selected_bench_assets:
            env_cfg.scene.dr_anmar_needle_v030 = make_dranmar_v030_needle_cfg(
                prim_path="{ENV_REGEX_NS}/DrAnmarNeedleV030",
                usd_path=bench_asset_paths["dr_anmar_needle_v030"],
            )
            env_cfg.scene.dr_anmar_needle_v030.init_state.pos = dr_anmar_asset_landing(
                "dr_anmar_needle_v030"
            )
        if "dr_anmar_needle_thread_coiled" in selected_bench_assets:
            env_cfg.scene.dr_anmar_needle_thread_coiled = (
                make_segmented_needle_thread_cfg(
                    configuration="coiled",
                    prim_path="{ENV_REGEX_NS}/DrAnmarNeedleThreadCoiled",
                    usd_path=bench_asset_paths[
                        "dr_anmar_needle_thread_coiled"
                    ],
                )
            )
            env_cfg.scene.dr_anmar_needle_thread_coiled.init_state.pos = dr_anmar_asset_landing(
                "dr_anmar_needle_thread_coiled"
            )
        if "dr_anmar_needle_thread_extended" in selected_bench_assets:
            env_cfg.scene.dr_anmar_needle_thread_extended = (
                make_segmented_needle_thread_cfg(
                    configuration="extended",
                    prim_path="{ENV_REGEX_NS}/DrAnmarNeedleThreadExtended",
                    usd_path=bench_asset_paths[
                        "dr_anmar_needle_thread_extended"
                    ],
                )
            )
            env_cfg.scene.dr_anmar_needle_thread_extended.init_state.pos = dr_anmar_asset_landing(
                "dr_anmar_needle_thread_extended"
            )
        if "dr_anmar_needle_thread_proxy" in selected_bench_assets:
            env_cfg.scene.dr_anmar_needle_thread_proxy = (
                make_needle_thread_rigid_proxy_cfg(
                    prim_path="{ENV_REGEX_NS}/DrAnmarNeedleThreadProxy",
                    usd_path=bench_asset_paths[
                        "dr_anmar_needle_thread_proxy"
                    ],
                )
            )
            env_cfg.scene.dr_anmar_needle_thread_proxy.init_state.pos = dr_anmar_asset_landing(
                "dr_anmar_needle_thread_proxy"
            )
        if "dr_anmar_tissue" in selected_bench_assets:
            env_cfg.scene.dr_anmar_tissue = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/DrAnmarSuturableTissue",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=dr_anmar_asset_landing("dr_anmar_tissue"),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(bench_asset_paths["dr_anmar_tissue"]),
                ),
            )
        if "vascular_clip" in selected_bench_assets:
            env_cfg.scene.dr_anmar_vascular_clip = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/DrAnmarVascularClip",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=dr_anmar_asset_landing("vascular_clip"),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(bench_asset_paths["vascular_clip"]),
                    activate_contact_sensors=True,
                ),
            )
        if "laparotomy_sponge" in selected_bench_assets:
            env_cfg.scene.laparotomy_sponge = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/LaparotomySponge",
                init_state=RigidObjectCfg.InitialStateCfg(
                    # Keep the large folded proxy near both instruments.
                    pos=dr_anmar_asset_landing("laparotomy_sponge"),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(bench_asset_paths["laparotomy_sponge"]),
                    variants={"state": "dry"},
                    semantic_tags=[
                        ("class", "laparotomy_sponge"),
                        ("count_category", "sponge"),
                        ("workflow_counting", "counting"),
                        ("workflow_retrieval", "retrieval"),
                        ("state", "dry"),
                    ],
                    activate_contact_sensors=True,
                ),
            )

    if bench_dr_anmar_suture_enabled:
        # Use SoftMimicGen's released deformable Rope.usd for the strand and
        # the same native PhysX auto-attachment pattern already proven in the
        # NVIDIA threading room. Dr.Anmar contributes the needle geometry and
        # its explicit swage anchor; no rigid-link rope or projected curve is
        # introduced in the main operating room.
        env_cfg.scene.replicate_physics = False
        rope_usd = (
            _softmimicgen_root
            / "source/softmimicgen_assets/data/Props/Rope/Rope.usd"
        )
        threaded_needle_usd = (
            REPOSITORY_ROOT
            / "assets/dr_anmar/needle/DrAnmarNeedleOnly.usda"
        )
        if not rope_usd.is_file():
            raise RuntimeError(
                f"Pinned NVIDIA SoftMimicGen Rope.usd is missing: {rope_usd}"
            )
        if not threaded_needle_usd.is_file():
            raise RuntimeError(
                f"Dr.Anmar needle runtime asset is missing: {threaded_needle_usd}"
            )
        one_room_physx = {
            "gpu_max_rigid_contact_count": 2**16,
            "gpu_max_rigid_patch_count": 2**14,
            "gpu_found_lost_pairs_capacity": 2**16,
            "gpu_found_lost_aggregate_pairs_capacity": 2**16,
            "gpu_total_aggregate_pairs_capacity": 2**16,
            "gpu_collision_stack_size": 2**24,
            "gpu_heap_capacity": 2**24,
            "gpu_temp_buffer_capacity": 2**22,
            "gpu_max_soft_body_contacts": 2**16,
            "gpu_max_particle_contacts": 2**14,
        }
        for setting, value in one_room_physx.items():
            if hasattr(env_cfg.sim.physx, setting):
                setattr(env_cfg.sim.physx, setting, value)
        env_cfg.scene.dr_anmar_native_suture = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/DrAnmarNativeSuture",
            init_state=DeformableObjectCfg.InitialStateCfg(
                pos=(-0.1, 0.0, 0.01),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(rope_usd),
                # Rope.usd has a 10 mm local radius. Preserve NVIDIA's 0.2
                # longitudinal scale and set both rendered and physical
                # diameter to the proven 0.8 mm SoftMimicGen configuration.
                scale=(0.2, 0.04, 0.04),
            ),
            debug_vis=False,
        )
        threaded_needle_spawn = sim_utils.UsdFileCfg(
            usd_path=str(threaded_needle_usd),
            activate_contact_sensors=True,
        )
        source_threaded_needle_spawn = threaded_needle_spawn.func

        def spawn_dr_anmar_threaded_needle(
            prim_path: str,
            cfg: sim_utils.UsdFileCfg,
            translation: tuple[float, float, float] | None = None,
            orientation: tuple[float, float, float, float] | None = None,
            **kwargs: Any,
        ) -> Any:
            needle_prim = source_threaded_needle_spawn(
                prim_path,
                cfg,
                translation=translation,
                orientation=orientation,
                **kwargs,
            )
            import omni.usd
            from pxr import PhysxSchema, Sdf

            stage = omni.usd.get_context().get_stage()
            resolved_root_path = str(needle_prim.GetPath())
            environment_path = resolved_root_path.rsplit("/", 1)[0]
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(
                stage,
                Sdf.Path(
                    f"{environment_path}/DrAnmarSutureNeedleAttachment"
                ),
            )
            attachment.GetActor0Rel().SetTargets(
                [
                    Sdf.Path(
                        f"{environment_path}/DrAnmarNativeSuture/Xform"
                    )
                ]
            )
            attachment.GetActor1Rel().SetTargets(
                [Sdf.Path(f"{resolved_root_path}/Needle")]
            )
            auto_attachment = (
                PhysxSchema.PhysxAutoAttachmentAPI.Apply(
                    attachment.GetPrim()
                )
            )
            auto_attachment.CreateDeformableVertexOverlapOffsetAttr(
                0.0012
            )
            auto_attachment.CreateCollisionFilteringOffsetAttr(0.0012)
            return needle_prim

        threaded_needle_spawn.func = spawn_dr_anmar_threaded_needle
        softmimicgen_swage_target_world = np.asarray(
            (-0.0201500003, 0.0000380001, 0.0093380004),
            dtype=np.float64,
        )
        dr_anmar_swage_anchor_local = np.asarray(
            (0.0, 0.00700281749604, 0.0),
            dtype=np.float64,
        )
        dr_anmar_threaded_needle_position = (
            softmimicgen_swage_target_world
            - dr_anmar_swage_anchor_local
        )
        env_cfg.scene.dr_anmar_threaded_needle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/DrAnmarThreadedNeedle",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(dr_anmar_threaded_needle_position.tolist()),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=threaded_needle_spawn,
        )

    if _softmimicgen_task:
        # There is one interactive room, so physics replication provides no
        # benefit and can drop cross-asset attachment relationships while the
        # source environment is cloned.
        env_cfg.scene.replicate_physics = False
        # SoftMimicGen randomizes the standalone strand and ring pose on every
        # RL reset. A factory-swaged needle must share that transform, so the
        # clinician workstation uses the authored deterministic composition.
        # Training keeps the upstream randomization unchanged outside here.
        interactive_events = getattr(env_cfg, "events", None)
        if interactive_events is not None:
            deterministic_events = ["reset_object_position", "reset_ring_position"]
            if bimanual_softmimicgen:
                deterministic_events.append("randomize_psm_joint_state")
            for event_name in deterministic_events:
                if hasattr(interactive_events, event_name):
                    setattr(interactive_events, event_name, None)
        # The upstream task defaults to training-scale PhysX reservations even
        # when num_envs=1.  Size the native buffers for one interactive room so
        # it can coexist with other Gilgamesh workloads without changing any
        # rope, ring, robot, action, reset, or contact behavior.
        one_room_physx = {
            "gpu_max_rigid_contact_count": 2**16,
            "gpu_max_rigid_patch_count": 2**14,
            "gpu_found_lost_pairs_capacity": 2**16,
            "gpu_found_lost_aggregate_pairs_capacity": 2**16,
            "gpu_total_aggregate_pairs_capacity": 2**16,
            "gpu_collision_stack_size": 2**24,
            "gpu_heap_capacity": 2**24,
            "gpu_temp_buffer_capacity": 2**22,
            "gpu_max_soft_body_contacts": 2**16,
            "gpu_max_particle_contacts": 2**14,
        }
        for setting, value in one_room_physx.items():
            if hasattr(env_cfg.sim.physx, setting):
                setattr(env_cfg.sim.physx, setting, value)
        # Rope.usd has a 0.01-unit local radius. Keep its 0.2 longitudinal
        # scale and use a 0.04 cross-section scale for a 0.4 mm radius / 0.8 mm
        # diameter. This is the native deformable mesh, so rendering and PhysX
        # collision/deformation use the same physical dimensions.
        env_cfg.scene.object.spawn.scale = (0.2, 0.04, 0.04)
        if procedure.get("enable_strand_self_collision"):
            # PhysX owns strand self-contact; no projected curve, teleport or
            # workstation-side constraint stands in for contact.
            env_cfg.scene.object.spawn.deformable_props = sim_utils.DeformableBodyPropertiesCfg(
                self_collision=True,
                self_collision_filter_distance=0.0012,
                solver_position_iteration_count=32,
            )
        if bimanual_softmimicgen:
            # Use the complete ORBIT-Surgical PSM configuration from the
            # needle rooms that already have matched rendered and collision
            # jaws. SoftMimicGen's psm_forceps asset has a different jaw
            # visual/collider relationship, which can report contact while a
            # visible gap remains. The strand, ring, attachment and solver
            # remain the native NVIDIA SoftMimicGen implementation.
            env_cfg.scene.robot = ORBIT_PSM_HIGH_PD_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot"
            )
            env_cfg.scene.robot.spawn.activate_contact_sensors = True
            # The shared PSM profile below owns jaw actuation for every room.
            # The native thread does not introduce a room-specific override.
            env_cfg.scene.robot.init_state.pos = (0.1, 0.0, 0.15)
            env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
            env_cfg.scene.robot_2 = env_cfg.scene.robot.replace(
                prim_path="{ENV_REGEX_NS}/Robot_2"
            )
            env_cfg.scene.robot_2.init_state.pos = (-0.1, 0.0, 0.15)
            env_cfg.scene.robot_2.init_state.rot = (1.0, 0.0, 0.0, 0.0)
            # Use ORBIT's complete bimanual needle-handover action stack and
            # ordering. SoftMimicGen remains responsible only for the native
            # deformable strand, ring, and their PhysX behavior.
            orbit_reference_cfg = ORBIT_NEEDLE_HANDOVER_CFG()
            orbit_actions = orbit_reference_cfg.actions
            orbit_actions.robot_1_body_action.asset_name = "robot"
            orbit_actions.robot_1_gripper_action.asset_name = "robot"
            orbit_actions.robot_2_body_action.asset_name = "robot_2"
            orbit_actions.robot_2_gripper_action.asset_name = "robot_2"
            env_cfg.actions = orbit_actions
            # Reuse the exact ORBIT surgical table asset and transform from
            # the working needle-handover room instead of SoftMimicGen's
            # visually different dry-lab table.
            env_cfg.scene.table = orbit_reference_cfg.scene.table
        needle_usd = (
            Path(__file__).resolve().parents[1]
            / "source/extensions/orbit.surgical.assets/data/Props/Surgical_needle/needle_sdf.usd"
        )
        if not needle_usd.is_file():
            raise RuntimeError(f"ORBIT-Surgical needle asset is missing: {needle_usd}")
        needle_digest = hashlib.sha256(needle_usd.read_bytes()).hexdigest()
        if needle_digest != "2b317a61f93631a7192e7ed2839ef20f7a75c05aa5f84a3905696134a64f36d7":
            raise RuntimeError("The pinned ORBIT needle mesh changed; re-derive its swage anchor before use")
        # Center of the blunt factory-swaged endpoint in the pinned ORBIT
        # needle default prim. This is derived from the mesh end cap, not from
        # whichever needle surface happens to be closest to the strand.
        orbit_needle_swage_anchor_m = (0.0478657183, 0.0491908647, 0.0009574010)
        # Use the same canonical rendered and collision scale as the proven
        # ORBIT needle rooms.
        native_needle_reference_scale = 0.4
        native_needle_scale = 0.4
        native_needle_reference_position = np.asarray(
            (-0.001003713, 0.019714346, 0.008955040), dtype=np.float64
        )
        swage_rotation_sign = np.asarray((-1.0, -1.0, 1.0), dtype=np.float64)
        native_swage_target_world = native_needle_reference_position + (
            swage_rotation_sign
            * np.asarray(orbit_needle_swage_anchor_m, dtype=np.float64)
            * native_needle_reference_scale
        )
        # Use ORBIT's graspable identity orientation. Solve the root position
        # from the fixed swage target so changing orientation and thickness
        # does not move the thread endpoint in world space.
        native_needle_position = native_swage_target_world - (
            np.asarray(orbit_needle_swage_anchor_m, dtype=np.float64)
            * native_needle_scale
        )
        needle_spawn = sim_utils.UsdFileCfg(
            usd_path=str(needle_usd),
            scale=(native_needle_scale,) * 3,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
                max_angular_velocity=200.0,
                max_linear_velocity=200.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
        )
        source_needle_spawn = needle_spawn.func

        def spawn_suture_needle_with_attachment(
            prim_path: str,
            cfg: sim_utils.UsdFileCfg,
            translation: tuple[float, float, float] | None = None,
            orientation: tuple[float, float, float, float] | None = None,
            **kwargs: Any,
        ) -> Any:
            """Spawn the rigid needle and author its native endpoint attachment."""

            needle_prim = source_needle_spawn(
                prim_path,
                cfg,
                translation=translation,
                orientation=orientation,
                **kwargs,
            )
            import omni.usd
            from pxr import Gf, PhysxSchema, Sdf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            resolved_needle_path = str(needle_prim.GetPath())
            environment_path = resolved_needle_path.rsplit("/", 1)[0]
            swage_anchor = UsdGeom.Xform.Define(
                stage,
                Sdf.Path(resolved_needle_path).AppendChild("SutureAnchor"),
            )
            swage_anchor.AddTranslateOp().Set(Gf.Vec3d(*orbit_needle_swage_anchor_m))
            thread_path = Sdf.Path(f"{environment_path}/Object/Xform")
            needle_path = Sdf.Path(resolved_needle_path)
            attachment_path = Sdf.Path(f"{environment_path}/SutureNeedleAttachment")
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
            attachment.GetActor0Rel().SetTargets([thread_path])
            attachment.GetActor1Rel().SetTargets([needle_path])
            auto_attachment = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
            # Exact swage placement permits a local surgical-scale selection
            # radius. It cannot reach the needle's middle or sharp endpoint.
            auto_attachment.CreateDeformableVertexOverlapOffsetAttr(0.0012)
            auto_attachment.CreateCollisionFilteringOffsetAttr(0.0012)
            grip_material_path = f"{resolved_needle_path}/GripPhysicsMaterial"
            grip_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="max",
            )
            grip_material.func(grip_material_path, grip_material)
            sim_utils.bind_physics_material(resolved_needle_path, grip_material_path)
            return needle_prim

        needle_spawn.func = spawn_suture_needle_with_attachment
        env_cfg.scene.suture_needle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SutureNeedle",
            init_state=RigidObjectCfg.InitialStateCfg(
                # Align the named swage anchor with the four terminal FEM
                # surface nodes while matching ORBIT's grasp-ready needle
                # orientation. The arc lies left of the swage and the free
                # strand trails right into the ring workspace.
                pos=tuple(native_needle_position.tolist()),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=needle_spawn,
        )
        # Dr.Anmar supplies one doctor-adjustable operative camera.  Remove the
        # two policy cameras from this interactive process only; NVIDIA's
        # recorded dataset and task assets remain unchanged on disk.
        for camera_name in ("agentview_image", "robot0_eye_in_hand_image"):
            if hasattr(env_cfg.scene, camera_name):
                setattr(env_cfg.scene, camera_name, None)
            if hasattr(env_cfg.observations.policy, camera_name):
                setattr(env_cfg.observations.policy, camera_name, None)
    # Keep the legacy authored research-room assembly quarantined here. The
    # main operating bench uses SoftMimicGen's native deformable strand above;
    # NVIDIA/ORBIT/SoftMimicGen rooms otherwise keep their native assets and
    # solver contracts unchanged.
    dr_anmar_needle_manifest = (
        configure_nvidia_needle_dr_anmar_suture(
            env_cfg.scene,
            asset_base_cfg_type=AssetBaseCfg,
            usd_file_cfg_type=sim_utils.UsdFileCfg,
            physics_lod=suture_physics_lod,
        )
        if nvidia_needle_dr_anmar_suture_enabled
        else configure_dr_anmar_needle(
            env_cfg.scene,
            asset_base_cfg_type=AssetBaseCfg,
            usd_file_cfg_type=sim_utils.UsdFileCfg,
            physics_lod=suture_physics_lod,
        )
        if dr_anmar_parametric_needle_enabled
        else None
    )
    # RL environments end and auto-reset episodes on success, dropped
    # objects, force thresholds, and time limits.  That is correct during
    # policy training but disastrous during clinician teleoperation: a
    # contact or mistake must remain visible and recoverable until the doctor
    # explicitly presses Reset.  Preserve the terms in the task configs for
    # RL and disable them only for this interactive workstation process.
    interactive_terminations = getattr(env_cfg, "terminations", None)
    if interactive_terminations is not None:
        for term_name in (
            "time_out",
            "object_dropping",
            "success",
            "excessive_object_force",
            "protected_surface_force",
        ):
            if hasattr(interactive_terminations, term_name):
                setattr(interactive_terminations, term_name, None)
    env_cfg.episode_length_s = 3600.0
    env_cfg.scene.num_envs = 1
    native_room = _native_room if _native_room and _native_room.get("available") else None
    native_deformable_enabled = bool(
        native_room
        and native_room.get("backend")
        in {"physx_fem", "physx_fem_hemostasis"}
    )
    native_hemostasis_enabled = bool(
        native_room and native_room.get("backend") == "physx_fem_hemostasis"
    )
    native_tissue_enabled = bool(
        native_deformable_enabled
        and native_room.get("representation") != "upstream_softmimicgen_task"
        and native_room.get("runtime_provider") != "nvidia_softmimicgen"
    )
    native_static_collision_enabled = bool(
        native_room
        and native_room.get("backend") == "openusd_static_collision"
    )
    if native_static_collision_enabled:
        spawn = native_room["spawn"]
        setattr(
            env_cfg.scene,
            str(native_room["stage_key"]),
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/DrAnmarSuturableTissue",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=tuple(spawn["translation_m"])
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(native_room["asset_path"]),
                    scale=tuple(spawn["scale"]),
                ),
            ),
        )
    configured_psm_articulations = psm_articulation_names(env_cfg.scene)
    runtime_psm_gripper_profile = resolve_psm_gripper_profile(
        open_rad=args_cli.gripper_open_rad,
        close_rad=args_cli.gripper_close_rad,
    )
    for robot_attribute in configured_psm_articulations:
        robot_cfg = getattr(env_cfg.scene, robot_attribute, None)
        if robot_cfg is not None:
            if getattr(robot_cfg, "spawn", None) is not None:
                robot_cfg.spawn.activate_contact_sensors = True
            apply_psm_gripper_articulation_profile(
                robot_cfg,
                runtime_psm_gripper_profile,
            )
    reference_actions = ORBIT_NEEDLE_HANDOVER_CFG().actions
    configured_psm_action_terms = complete_psm_actions_from_nvidia_orbit(
        env_cfg.actions,
        env_cfg.scene,
        reference_actions,
    )
    apply_psm_gripper_action_profile(
        env_cfg.actions,
        runtime_psm_gripper_profile,
    )
    configured_psm_gripper_profile = psm_gripper_profile_manifest(
        profile=runtime_psm_gripper_profile,
        action_terms=configured_psm_action_terms,
        articulations=configured_psm_articulations,
    )
    camera_target = np.asarray(
        procedure.get(
            "interactive_camera_target_m",
            (-0.045, 0.220, 0.355)
            if closure_robot_enabled
            else (-0.070, -0.020, 0.055)
            if nvidia_native_bench
            else env_cfg.viewer.lookat,
        ),
        dtype=np.float32,
    )
    # Start from the room-facing side used by the official OR scene so the
    # doctor sees the instrument, liver, table, and surrounding environment.
    camera_eye = np.asarray(
        procedure.get(
            "interactive_camera_eye_m",
            (0.32, 0.62, 0.52)
            if closure_robot_enabled
            else (0.38, -0.44, 0.32)
            if nvidia_native_bench
            else (0.36, 0.36, 0.21)
            if bimanual_softmimicgen
            else (0.20, 0.20, 0.11)
            if _softmimicgen_task
            else (0.45, 0.25, 0.28),
        ),
        dtype=np.float32,
    )
    interactive_camera_width = int(
        procedure.get("interactive_camera_width_px", args_cli.camera_width)
    )
    interactive_camera_height = int(
        procedure.get("interactive_camera_height_px", args_cli.camera_height)
    )
    endoscope_data_types = (
        ["rgb"]
        if procedure.get("interactive_rgb_only")
        or args_cli.sensor_profile == "efficient"
        else ["rgb", "distance_to_image_plane", "semantic_segmentation"]
    )
    env_cfg.scene.endoscope = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Endoscope",
        update_period=interactive_camera_period_s,
        height=interactive_camera_height,
        width=interactive_camera_width,
        data_types=endoscope_data_types,
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=0.25,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=tuple(camera_eye.tolist()), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    if (
        not single_active_camera_renderer
        and (
            procedure.get("interactive_multiview")
            or args_cli.sensor_profile in {"stereo", "research"}
        )
    ):
        env_cfg.scene.endoscope_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/EndoscopeRight",
            update_period=0.04,
            height=interactive_camera_height,
            width=interactive_camera_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=22.0,
                focus_distance=0.25,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=tuple(camera_eye.tolist()), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        )
    wrist_tip_name = "endo360_needle" if "STAR" in args_cli.task else "ecm_end_link" if "ECM" in args_cli.task else "psm_tool_tip_link"
    wrist_robot_names = (
        ("Robot", "Robot_2")
        if bimanual_softmimicgen
        else ("Robot_1", "Robot_2")
        if "Dual" in args_cli.task
        else ("Robot",)
    )
    contact_effect_filter_prim = procedure.get("contact_effect_filter_prim")
    if (
        dynamic_abdominal_patient_enabled
        and not contact_effect_filter_prim
    ):
        contact_effect_filter_prim = (
            "{ENV_REGEX_NS}/DynamicAbdominalPatient/Anatomy/"
            f"{procedure.get('dynamic_patient_contact_target', 'mesentery')}"
            "/Geometry/Visual"
        )
    contact_effect_filter = (
        [str(contact_effect_filter_prim)]
        if (
            contact_driven_patient_effects_enabled
            and contact_effect_filter_prim
        )
        else []
    )
    # One native net-force sensor per rigid jaw is sufficient for grasp
    # detection. Per-segment filter matrices scale every jaw against the full
    # suture body chain and were never consumed after PhysX became the sole
    # thread authority.
    for contact_index, contact_robot_name in enumerate(wrist_robot_names, start=1):
        for jaw_index in (1, 2):
            setattr(
                env_cfg.scene,
                f"gripper_contact_{contact_index}_jaw_{jaw_index}",
                ContactSensorCfg(
                    prim_path=(
                        f"{{ENV_REGEX_NS}}/{contact_robot_name}/"
                        f"psm_tool_gripper{jaw_index}_link"
                    ),
                    update_period=0.0,
                    history_length=3,
                    track_air_time=False,
                    filter_prim_paths_expr=contact_effect_filter,
                ),
            )
    if not single_active_camera_renderer:
        for wrist_index, wrist_robot_name in enumerate(wrist_robot_names, start=1):
            setattr(
                env_cfg.scene,
                f"wrist_{wrist_index}",
                CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/DrAnmarWristCamera{wrist_index}",
                    update_period=CANONICAL_PSM_GRIPPER_PROFILE.camera_update_period_s,
                    height=CANONICAL_PSM_GRIPPER_PROFILE.camera_height_px,
                    width=CANONICAL_PSM_GRIPPER_PROFILE.camera_width_px,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=CANONICAL_PSM_GRIPPER_PROFILE.camera_focal_length_mm,
                        focus_distance=0.10,
                        horizontal_aperture=20.955,
                        clipping_range=(0.005, 0.50),
                    ),
                    offset=CameraCfg.OffsetCfg(
                        pos=(0.20, 0.20, 0.14),
                        rot=(1.0, 0.0, 0.0, 0.0),
                        convention="world",
                    ),
                ),
            )
    organ_usd = args_cli.anatomy_scene.expanduser().resolve() if args_cli.anatomy_scene else (
        DATA_ROOT
        / "assets/sufia_bc/OR_scene_CTLiver-Prostate-Bladder"
        / "OR_scene_CTLiver-Prostate-Bladder/models/organs/models_topo_blender.usdc"
    )
    allowed_anatomy_root = (DATA_ROOT / "assets/sufia_bc").resolve()
    allowed_composed_root = (DATA_ROOT / "scenes/openusd").resolve()
    if organ_usd.is_file() and allowed_anatomy_root not in organ_usd.parents and allowed_composed_root not in organ_usd.parents:
        raise ValueError("The composed anatomy asset must be inside the installed Dr.Anmar OpenUSD library")
    openusd_environment = args_cli.openusd_environment.expanduser().resolve() if args_cli.openusd_environment else None
    allowed_environment_root = (DATA_ROOT / "scenes/openusd").resolve()
    if openusd_environment and (
        not openusd_environment.is_file() or allowed_environment_root not in openusd_environment.parents
    ):
        raise ValueError("The OpenUSD environment must be a prepared Dr.Anmar scene inside the composed library")
    if openusd_environment and not _softmimicgen_task:
        env_cfg.scene.openusd_operating_room = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/OpenUSDOperatingRoom",
            spawn=sim_utils.UsdFileCfg(usd_path=str(openusd_environment)),
        )
    elif not _softmimicgen_task and not nvidia_native_bench:
        # Offline fallback for development installations that do not yet have
        # the repaired OpenUSD compositions generated.
        wall_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.27, 0.30), roughness=0.72)
        env_cfg.scene.or_wall_x = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ORWallX",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.25, 0.0, 0.45)),
            spawn=sim_utils.CuboidCfg(size=(0.05, 2.5, 2.8), visual_material=wall_material),
        )
        env_cfg.scene.or_wall_y = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ORWallY",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -1.25, 0.45)),
            spawn=sim_utils.CuboidCfg(size=(2.5, 0.05, 2.8), visual_material=wall_material),
        )
        env_cfg.scene.or_backdrop = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ORBackdrop",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-1.0, -0.55, 0.45),
                rot=(0.9681476, 0.0, 0.0, 0.2503800),
            ),
            spawn=sim_utils.CuboidCfg(size=(0.05, 3.5, 2.8), visual_material=wall_material),
        )
    procedure_id = str(procedure.get("id", ""))
    guide_kind = str(procedure.get("guide_kind", ""))
    room_waypoints = procedure_waypoints(procedure)
    if guide_kind == "navigation":
        anatomy_position = (-0.117, -0.0945, -0.189)
    else:
        anatomy_position = (-0.117, -0.1945, -0.164)
    native_episode_domain: dict[str, Any] = {}
    if native_tissue_enabled:
        spawn = native_room["spawn"]
        native_deformable_prim_name = str(
            native_room.get(
                "stage_prim_name",
                str(native_room["stage_key"])
                .replace("_", " ")
                .title()
                .replace(" ", ""),
            )
        )
        material_contract = json.loads(Path(native_room["material_path"]).read_text(encoding="utf-8"))
        if material_contract.get("schema") == "dr.anmar.hemostasis-profile.v1":
            hemostasis_episode = sample_hemostasis_episode_parameters(
                material_contract,
                DEFAULT_SCENARIO_SEED,
            )
            stable_vessel_proxy = stable_physx_vessel_proxy_parameters(
                material_contract,
                hemostasis_episode,
            )
            native_material_runtime = {
                "density_kg_m3": float(stable_vessel_proxy["density_kg_m3"]),
                "dynamic_friction": float(
                    stable_vessel_proxy["dynamic_friction"]
                ),
                "static_friction": float(stable_vessel_proxy["static_friction"]),
                "youngs_modulus_pa": float(
                    stable_vessel_proxy["youngs_modulus_pa"]
                ),
                "poisson_ratio": float(stable_vessel_proxy["poisson_ratio"]),
                "vertex_velocity_damping": min(
                    1.0,
                    float(stable_vessel_proxy["damping_ratio"])
                    * float(
                        material_contract["stable_physx_proxy"][
                            "vertex_velocity_damping_scale"
                        ]
                    ),
                ),
                "solver_position_iterations": int(
                    material_contract["solver"]["position_iterations"]
                ),
            }
            native_episode_domain = {
                "schema": "dr.anmar.native-deformable-domain.v1",
                "profile_id": material_contract["id"],
                "setup_seed": DEFAULT_SCENARIO_SEED,
                "parameters": hemostasis_episode.payload(),
                "stable_backend_proxy": stable_vessel_proxy,
                "parameter_application": "setup_before_first_physics_step",
                "per_layer_mechanics": "homogenized_proxy",
                "explicit_layer_ids_preserved": True,
                "backend_applied_parameters": [
                    "density_kg_m3",
                    "dynamic_friction",
                    "youngs_modulus_pa",
                    "poisson_ratio",
                    "vertex_velocity_damping",
                    "solver_position_iterations",
                ],
                "static_friction": (
                    "recorded_target_not_exposed_by_isaac_lab_2_3_"
                    "deformable_material_cfg"
                ),
                "flow_model": "not_present_in_native_physx_room",
                "plastic_clip_forming": False,
                "clinical_validation": False,
            }
        elif material_contract.get("schema") == "dr.anmar.suturable-tissue-profile.v1":
            tissue_episode = sample_tissue_episode_parameters(
                material_contract,
                DEFAULT_SCENARIO_SEED,
            )
            stable_tissue_proxy = stable_physx_proxy_parameters(
                material_contract,
                tissue_episode,
            )
            native_material_runtime = {
                "density_kg_m3": float(stable_tissue_proxy["density_kg_m3"]),
                "dynamic_friction": float(stable_tissue_proxy["dynamic_friction"]),
                "static_friction": float(stable_tissue_proxy["static_friction"]),
                "youngs_modulus_pa": float(stable_tissue_proxy["youngs_modulus_pa"]),
                "poisson_ratio": float(stable_tissue_proxy["poisson_ratio"]),
                "vertex_velocity_damping": min(
                    1.0,
                    float(stable_tissue_proxy["damping_ratio"])
                    * float(
                        material_contract["stable_physx_proxy"][
                            "vertex_velocity_damping_scale"
                        ]
                    ),
                ),
                "solver_position_iterations": int(
                    material_contract["solver"]["position_iterations"]
                ),
            }
            native_episode_domain = {
                "schema": "dr.anmar.native-deformable-domain.v1",
                "profile_id": material_contract["id"],
                "setup_seed": DEFAULT_SCENARIO_SEED,
                "parameters": tissue_episode.payload(),
                "stable_backend_proxy": stable_tissue_proxy,
                "parameter_application": "setup_before_first_physics_step",
                "per_layer_mechanics": "homogenized_proxy",
                "explicit_layer_ids_preserved": True,
                "backend_applied_parameters": [
                    "density_kg_m3",
                    "dynamic_friction",
                    "youngs_modulus_pa",
                    "poisson_ratio",
                    "vertex_velocity_damping",
                    "solver_position_iterations",
                ],
                "static_friction": (
                    "recorded_target_not_exposed_by_isaac_lab_2_3_"
                    "deformable_material_cfg"
                ),
                "clinical_validation": False,
            }
        else:
            tissue_material = material_contract["intact_tissue"]
            contact_material = material_contract["contact"]
            native_material_runtime = {
                "density_kg_m3": float(tissue_material["density_kg_m3_seed"]),
                "dynamic_friction": float(contact_material["dynamic_friction_seed"]),
                "static_friction": float(contact_material["static_friction_seed"]),
                "youngs_modulus_pa": float(
                    tissue_material["youngs_modulus_pa_seed"]
                ),
                "poisson_ratio": float(tissue_material["poisson_ratio_seed"]),
                "vertex_velocity_damping": 0.005,
                "solver_position_iterations": 16,
            }
        native_spawn = sim_utils.UsdFileCfg(
            usd_path=str(native_room["asset_path"]),
            scale=tuple(spawn["scale"]),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                deformable_enabled=True,
                self_collision=True,
                solver_position_iteration_count=int(
                    native_material_runtime["solver_position_iterations"]
                ),
                vertex_velocity_damping=float(
                    native_material_runtime["vertex_velocity_damping"]
                ),
                sleep_damping=0.0,
                sleep_threshold=0.0,
                settling_threshold=0.0,
            ),
        )
        native_usd_spawn = native_spawn.func

        def spawn_native_deformable_with_material(
            prim_path: str,
            cfg,
            translation=None,
            orientation=None,
            **kwargs,
        ):
            """Spawn the watertight surface, let PhysX cook it, and bind its material."""

            deformable_props = cfg.deformable_props
            source_cfg = cfg.replace(deformable_props=None)
            root_prim = native_usd_spawn(
                prim_path,
                source_cfg,
                translation=translation,
                orientation=orientation,
                **kwargs,
            )
            root_path = str(root_prim.GetPath())
            sim_utils.define_deformable_body_properties(root_path, deformable_props)
            material_path = f"{root_path}/DrAnmarNativeTissueMaterial"
            material_cfg = sim_utils.DeformableBodyMaterialCfg(
                density=float(native_material_runtime["density_kg_m3"]),
                dynamic_friction=float(native_material_runtime["dynamic_friction"]),
                youngs_modulus=float(native_material_runtime["youngs_modulus_pa"]),
                poissons_ratio=float(native_material_runtime["poisson_ratio"]),
            )
            material_cfg.func(material_path, material_cfg)
            sim_utils.bind_physics_material(root_path, material_path)
            return root_prim

        native_spawn.func = spawn_native_deformable_with_material
        setattr(
            env_cfg.scene,
            str(native_room["stage_key"]),
            DeformableObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{native_deformable_prim_name}",
                init_state=DeformableObjectCfg.InitialStateCfg(
                    pos=tuple(spawn["translation_m"])
                ),
                spawn=native_spawn,
            ),
        )
        for auxiliary in native_room.get("auxiliary_assets", []):
            setattr(
                env_cfg.scene,
                str(auxiliary["stage_key"]),
                AssetBaseCfg(
                    prim_path=(
                        f"{{ENV_REGEX_NS}}/{auxiliary['prim_name']}"
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=tuple(auxiliary["translation_m"]),
                        rot=tuple(auxiliary["rotation_wxyz"]),
                    ),
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=str(auxiliary["asset_path"])
                    ),
                ),
            )
    elif organ_usd.is_file() and not procedure.get("hide_anatomy") and not _softmimicgen_task:
        env_cfg.scene.liver_showcase = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/LiverShowcase",
            init_state=AssetBaseCfg.InitialStateCfg(pos=anatomy_position),
            spawn=sim_utils.UsdFileCfg(usd_path=str(organ_usd), scale=(0.35, 0.35, 0.35)),
        )
    env_cfg.num_rerenders_on_reset = 3

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import (
        Gf,
        PhysxSchema,
        Sdf,
        Usd,
        UsdGeom,
        UsdPhysics,
        UsdShade,
        Vt,
    )

    suture_stage = omni.usd.get_context().get_stage()
    suture_root_path = ""
    suture_segment_count = 0
    suture_root = None
    suture_rigid_bodies: list[Any] = []
    initial_dr_anmar_needle_domain: dict[str, Any] = {}
    suture_runtime_domain_state: list[dict[str, Any]] = [{}]
    suture_physics_view = SimulationManager.get_physics_sim_view()
    suture_needle_view = None
    suture_interface_view = None
    suture_segment_view = None
    if dr_anmar_needle_enabled:
        if dr_anmar_needle_manifest is None:
            raise RuntimeError(
                "The authored Dr.Anmar suture room has no instrument manifest"
            )
        suture_root_path = str(dr_anmar_needle_manifest["prim_path"])
        suture_segment_count = int(
            dr_anmar_needle_manifest["segment_count"]
        )
        suture_root = suture_stage.GetPrimAtPath(suture_root_path)
        if not suture_root.IsValid():
            raise RuntimeError(
                "The Dr.Anmar needle-suture instrument did not enter the room"
            )
        suture_rigid_bodies = [
            prim
            for prim in Usd.PrimRange(suture_root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        factory_swage = UsdPhysics.FixedJoint.Get(
            suture_stage,
            f"{suture_root_path}/FactorySwage",
        )
        pullout_joint = UsdPhysics.Joint.Get(
            suture_stage,
            f"{suture_root_path}/Suture/Joints/J0000",
        )
        needle_interface = suture_stage.GetPrimAtPath(
            f"{suture_root_path}/Suture/NeedleInterface"
        )
        interface_kinematic = UsdPhysics.RigidBodyAPI(
            needle_interface
        ).GetKinematicEnabledAttr().Get()
        if (
            len(suture_rigid_bodies) != suture_segment_count + 2
            or not factory_swage.GetPrim().IsValid()
            or not pullout_joint.GetPrim().IsValid()
            or interface_kinematic is not False
        ):
            raise RuntimeError(
                "The Dr.Anmar needle-suture physics composition is incomplete: "
                f"rigid_bodies={len(suture_rigid_bodies)}, "
                f"factory_swage={factory_swage.GetPrim().IsValid()}, "
                f"pullout_joint={pullout_joint.GetPrim().IsValid()}, "
                f"interface_kinematic={interface_kinematic}"
            )
        if dr_anmar_parametric_needle_enabled:
            initial_dr_anmar_needle_domain = (
                apply_dr_anmar_needle_episode_domain(
                    suture_stage,
                    seed=DEFAULT_SCENARIO_SEED,
                    root_path=suture_root_path,
                )
            )
        else:
            initial_dr_anmar_needle_domain = {
                "needle_provider": "NVIDIA_ORBIT",
                "needle_domain_randomization": False,
                "reason": "preserve_pinned_native_needle_physics",
            }
        _initial_suture_runtime_profile, initial_suture_domain = (
            sample_suture_runtime_profile(
                suture_profile,
                DEFAULT_SCENARIO_SEED,
            )
        )
        suture_runtime_domain_state[0] = initial_suture_domain
        suture_segment_paths = [
            f"{suture_root_path}/Suture/Segments/S{index:04d}"
            for index in range(suture_segment_count)
        ]
        suture_segment_view = suture_physics_view.create_rigid_body_view(
            suture_segment_paths
        )
        if (
            suture_segment_view._backend is None
            or suture_segment_view.count != suture_segment_count
        ):
            raise RuntimeError(
                "The Dr.Anmar live suture tensor view is incomplete: "
                f"segments="
                f"{suture_segment_view.count if suture_segment_view._backend else 0}"
            )
        suture_needle_view = suture_physics_view.create_rigid_body_view(
            f"{suture_root_path}/Needle"
        )
        suture_interface_view = suture_physics_view.create_rigid_body_view(
            f"{suture_root_path}/Suture/NeedleInterface"
        )
        if (
            suture_needle_view._backend is None
            or suture_needle_view.count != 1
            or suture_interface_view._backend is None
            or suture_interface_view.count != 1
        ):
            raise RuntimeError(
                "The Dr.Anmar needle-suture rigid-body views are incomplete"
            )
    hemostasis_clip_view = None
    if native_hemostasis_enabled:
        hemostasis_clip_view = suture_physics_view.create_rigid_body_view(
            "/World/envs/env_0/DrAnmarVascularClip"
        )
        if (
            hemostasis_clip_view._backend is None
            or hemostasis_clip_view.count != 1
        ):
            raise RuntimeError(
                "The DrAnmar Vascular Clip did not create one native rigid body"
            )
    if native_episode_domain:
        native_root = suture_stage.GetPrimAtPath(
            f"/World/envs/env_0/{native_deformable_prim_name}"
        )
        roughness = (
            native_episode_domain.get("stable_backend_proxy", {}).get(
                "surface_roughness"
            )
        )
        if native_root.IsValid() and roughness is not None:
            for prim in Usd.PrimRange(native_root):
                roughness_attribute = prim.GetAttribute("inputs:roughness")
                if roughness_attribute.IsValid():
                    roughness_attribute.Set(float(roughness))
        print(
            "[DR_ANMAR_NATIVE_DEFORMABLE_DOMAIN] "
            + json.dumps(native_episode_domain, sort_keys=True),
            flush=True,
        )
    if dr_anmar_needle_enabled:
        print(
            "[DR_ANMAR_NEEDLE] "
            + json.dumps(
                {
                    **(dr_anmar_needle_manifest or {}),
                    "rigid_body_count": len(suture_rigid_bodies),
                    "factory_swage": True,
                    "breakable_pullout_joint": True,
                    "episode_domain": initial_dr_anmar_needle_domain,
                    "live_material_history_controller": False,
                    "physics_authority": "OpenUSD_PhysX",
                    "live_segment_tensor_count": suture_segment_view.count,
                    "live_suture_domain": suture_runtime_domain_state[0],
                    "clinical_validation": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    scene = env.unwrapped.scene
    camera = scene["endoscope"]
    shared_camera_renderer = single_active_camera_renderer
    stereo_right_camera = (
        None
        if shared_camera_renderer
        else scene["endoscope_right"]
        if args_cli.sensor_profile in {"stereo", "research"}
        or procedure.get("interactive_multiview")
        else None
    )
    wrist_cameras = (
        []
        if shared_camera_renderer
        else [
            scene[f"wrist_{index}"]
            for index in range(1, len(wrist_robot_names) + 1)
        ]
    )
    camera_sources = {"endoscope_left": camera}
    if shared_camera_renderer:
        if (
            procedure.get("interactive_multiview")
            or args_cli.sensor_profile in {"stereo", "research"}
        ):
            camera_sources["endoscope_right"] = camera
        camera_sources.update(
            {
                f"wrist_{index}": camera
                for index in range(1, len(wrist_robot_names) + 1)
            }
        )
    elif stereo_right_camera is not None:
        camera_sources["endoscope_right"] = stereo_right_camera
    if not shared_camera_renderer:
        camera_sources.update(
            {
                f"wrist_{index}": wrist_camera
                for index, wrist_camera in enumerate(wrist_cameras, start=1)
            }
        )
    all_robot_names = sorted(scene.articulations.keys())
    robot_names = [
        name
        for name in all_robot_names
        if all(
            joint_name in tuple(scene[name].joint_names)
            for joint_name in PSM_ARM_NAMES
        )
    ]
    if not robot_names:
        robot_names = all_robot_names
    robots = {name: scene[name] for name in robot_names}
    robot_body_names = {name: list(getattr(robot, "body_names", [])) for name, robot in robots.items()}
    stapler_articulation = (
        scene.articulations.get("stapler_test_device")
        if stapler_test_cell_enabled
        else None
    )
    closure_robot_articulation = (
        scene.articulations.get("closure_robot")
        if closure_robot_enabled
        else None
    )
    closure_robot_joint_indices: dict[str, int] = {}
    closure_robot_arm_joint_indices: list[int] = []
    closure_robot_body_indices: dict[str, int] = {}
    if closure_robot_enabled:
        if closure_robot_articulation is None:
            raise RuntimeError(
                "The setup bench did not create the Franka closure articulation"
            )
        closure_joint_names = list(closure_robot_articulation.joint_names)
        required_closure_joint_names = tuple(
            closure_phase_targets(ClosurePhase.READY)
        )
        try:
            closure_robot_joint_indices = {
                name: closure_joint_names.index(name)
                for name in required_closure_joint_names
            }
        except ValueError as exc:
            raise RuntimeError(
                "The composed Franka articulation is missing one or more "
                "approximation, clamp, staple, or adhesive joints"
            ) from exc
        try:
            closure_robot_arm_joint_indices = [
                closure_joint_names.index(f"panda_joint{index}")
                for index in range(1, 8)
            ]
        except ValueError as exc:
            raise RuntimeError(
                "The composed Franka articulation is missing one or more "
                "of its seven arm joints"
            ) from exc
        closure_body_names = list(closure_robot_articulation.body_names)
        forbidden_closure_bodies = sorted(
            {
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
            }.intersection(closure_body_names)
        )
        if forbidden_closure_bodies:
            raise RuntimeError(
                "The Franka closure articulation still contains stock hand "
                "bodies: " + ", ".join(forbidden_closure_bodies)
            )
        required_closure_bodies = (
            "panda_link8",
            "Mount",
            "LeftCarriage",
            "RightCarriage",
            "LeftClamp",
            "RightClamp",
            "StapleDriver",
            "AdhesiveCarriage",
        )
        try:
            closure_robot_body_indices = {
                name: closure_body_names.index(name)
                for name in required_closure_bodies
            }
        except ValueError as exc:
            raise RuntimeError(
                "The Franka closure articulation is missing its physical "
                "mount or one or more payload links"
            ) from exc
    stapler_trigger_joint_index: int | None = None
    stapler_pusher_joint_index: int | None = None
    stapler_housing_body_index: int | None = None
    stapler_fixture_position_w: torch.Tensor | None = None
    stapler_fixture_quaternion_w: torch.Tensor | None = None
    if stapler_test_cell_enabled:
        if stapler_articulation is None:
            raise RuntimeError(
                "The stapler test cell did not create its articulated device"
            )
        stapler_joint_names = list(stapler_articulation.joint_names)
        try:
            stapler_trigger_joint_index = stapler_joint_names.index(
                "trigger_joint"
            )
            stapler_pusher_joint_index = stapler_joint_names.index(
                "pusher_joint"
            )
        except ValueError as exc:
            raise RuntimeError(
                "The stapler test cell requires trigger_joint and pusher_joint"
            ) from exc
        stapler_body_names = list(stapler_articulation.body_names)
        try:
            stapler_housing_body_index = stapler_body_names.index("Housing")
        except ValueError as exc:
            raise RuntimeError(
                "The stapler test cell requires the authored Housing link"
            ) from exc
    skin_adhesive_mounted_arm = 0
    skin_adhesive_articulation = (
        robots[robot_names[skin_adhesive_mounted_arm]]
        if skin_adhesive_enabled
        and len(robot_names) > skin_adhesive_mounted_arm
        else None
    )
    skin_adhesive_joint_indices: dict[str, int] = {}
    if skin_adhesive_enabled:
        if skin_adhesive_articulation is None:
            raise RuntimeError(
                "The topical skin-adhesive end effector has no mounted PSM"
            )
        skin_adhesive_joint_names = list(
            skin_adhesive_articulation.joint_names
        )
        try:
            skin_adhesive_joint_indices = {
                name: skin_adhesive_joint_names.index(name)
                for name in (
                    "left_paddle_joint",
                    "right_paddle_joint",
                    "metering_piston_joint",
                )
            }
        except ValueError as exc:
            raise RuntimeError(
                "The topical skin-adhesive applicator requires both paddle "
                "joints and the metering-piston joint"
            ) from exc
    closure_robot_controller: ClosureSequenceController | None = None
    closure_robot_tool_path = (
        "/World/envs/env_0/ClosureRobot/DrAnmarClosureTool"
    )
    closure_tissue_root_path = "/World/envs/env_0/ClosureTissue"
    closure_left_tissue_path = (
        f"{closure_tissue_root_path}/LeftTissue/SimulationMesh"
    )
    closure_right_tissue_path = (
        f"{closure_tissue_root_path}/RightTissue/SimulationMesh"
    )
    if closure_robot_enabled:
        for required_path in (
            closure_robot_tool_path,
            closure_left_tissue_path,
            closure_right_tissue_path,
        ):
            if not suture_stage.GetPrimAtPath(required_path).IsValid():
                raise RuntimeError(
                    "The Franka closure setup is missing its composed runtime "
                    f"prim: {required_path}"
                )
        closure_robot_controller = ClosureSequenceController(
            stage=suture_stage,
            tool_path=closure_robot_tool_path,
            left_tissue_path=closure_left_tissue_path,
            right_tissue_path=closure_right_tissue_path,
        )
    closure_robot_arm_hold_targets: torch.Tensor | None = None
    closure_robot_mount_reference_w: torch.Tensor | None = None
    closure_robot_max_mount_error_mm = 0.0
    if (
        closure_robot_articulation is not None
        and closure_robot_arm_joint_indices
        and "panda_link8" in closure_robot_body_indices
    ):
        closure_robot_arm_hold_targets = (
            closure_robot_articulation.data.joint_pos[
                :,
                closure_robot_arm_joint_indices,
            ]
            .detach()
            .clone()
        )
        closure_robot_mount_reference_w = (
            closure_robot_articulation.data.body_pos_w[
                0,
                closure_robot_body_indices["panda_link8"],
            ]
            .detach()
            .clone()
        )
    object_names = sorted(scene.rigid_objects.keys())
    objects = {name: scene[name] for name in object_names}
    deformable_names = sorted(getattr(scene, "deformable_objects", {}).keys())
    deformables = {name: scene[name] for name in deformable_names}
    native_tissue = deformables.get(str(native_room.get("stage_key", ""))) if native_room else None
    stapler_closure_tissues = (
        [
            deformables.get("stapler_closure_tissue_left"),
            deformables.get("stapler_closure_tissue_right"),
        ]
        if stapler_test_cell_enabled
        else []
    )
    if stapler_test_cell_enabled and any(
        tissue is None for tissue in stapler_closure_tissues
    ):
        raise RuntimeError(
            "The stapler test cell did not create both PhysX FEM tissue flaps"
        )
    stapler_closure_tissues = [
        tissue
        for tissue in stapler_closure_tissues
        if tissue is not None
    ]
    interactive_deformable = (
        deformables.get("object")
        if _softmimicgen_task
        else deformables.get("dr_anmar_native_suture")
        if bench_dr_anmar_suture_enabled
        else native_tissue
    )
    ring_physics_ready = "ring" in objects
    strand_self_collision_ready = not bool(procedure.get("enable_strand_self_collision"))
    self_collision_attributes: dict[str, Any] = {}
    if _softmimicgen_task and procedure.get("enable_strand_self_collision"):
        import omni.usd
        from pxr import Usd

        diagnostic_stage = omni.usd.get_context().get_stage()
        deformable_root = diagnostic_stage.GetPrimAtPath("/World/envs/env_0/Object")
        if not deformable_root.IsValid():
            raise RuntimeError("The native deformable strand is missing from the OpenUSD stage")
        for prim in Usd.PrimRange(deformable_root):
            for attribute in prim.GetAttributes():
                attribute_name = attribute.GetName()
                normalized_name = "".join(character for character in attribute_name.lower() if character.isalnum())
                if "selfcollision" not in normalized_name:
                    continue
                value = attribute.Get()
                self_collision_attributes[f"{prim.GetPath()}:{attribute_name}"] = value
                if isinstance(value, bool) and value:
                    strand_self_collision_ready = True
        print(
            "[DR_ANMAR_NATIVE_STRAND_SELF_COLLISION] "
            + json.dumps(
                {
                    "ready": strand_self_collision_ready,
                    "attributes": self_collision_attributes,
                },
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
    if bimanual_softmimicgen and not ring_physics_ready:
        raise RuntimeError("The bimanual thread room requires SoftMimicGen's native rigid ring")
    if _softmimicgen_task and interactive_deformable is not None and "suture_needle" in objects:
        # Report the named swage-to-terminal-surface separation once. This is
        # read-only: OpenUSD defines the anchor and PhysX owns all motion.
        import omni.usd

        diagnostic_stage = omni.usd.get_context().get_stage()
        swage_anchor = diagnostic_stage.GetPrimAtPath(
            "/World/envs/env_0/SutureNeedle/SutureAnchor"
        )
        if not swage_anchor.IsValid():
            raise RuntimeError("The native suture needle is missing its OpenUSD swage anchor")
        nodal_value = interactive_deformable.data.nodal_pos_w
        nodal_positions = getattr(nodal_value, "torch", nodal_value)[0].detach().cpu().numpy()
        default_value = interactive_deformable.data.default_nodal_state_w
        default_positions = getattr(default_value, "torch", default_value)[0, :, :3].detach().cpu().numpy()
        endpoint_mask = default_positions[:, 0] >= float(default_positions[:, 0].max()) - 0.0002
        default_endpoint_position = default_positions[endpoint_mask].mean(axis=0)
        endpoint_position = nodal_positions[endpoint_mask].mean(axis=0)
        needle_state = objects["suture_needle"].data
        needle_position = needle_state.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        needle_quaternion = needle_state.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
        needle_default_state = needle_state.default_root_state[0].detach().cpu().numpy().astype(np.float64)

        def rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
            quaternion = quaternion / np.linalg.norm(quaternion)
            vector_part = quaternion[1:]
            doubled_cross = 2.0 * np.cross(vector_part, vector)
            return vector + quaternion[0] * doubled_cross + np.cross(vector_part, doubled_cross)

        local_swage = (
            np.asarray(orbit_needle_swage_anchor_m, dtype=np.float64)
            * native_needle_scale
        )
        swage_position = needle_position + rotate_wxyz(needle_quaternion, local_swage)
        default_swage_position = needle_default_state[:3] + rotate_wxyz(
            needle_default_state[3:7], local_swage
        )
        swage_distance = float(np.linalg.norm(endpoint_position - swage_position))
        default_swage_distance = float(
            np.linalg.norm(default_endpoint_position - default_swage_position)
        )
        print(
            "[DR_ANMAR_NATIVE_SUTURE_GEOMETRY] "
            + json.dumps(
                {
                    "anchor_semantics": "ORBIT needle blunt swaged endpoint",
                    "needle_uniform_scale": native_needle_scale,
                    "strand_diameter_m": 0.0008,
                    "terminal_surface_nodes": int(endpoint_mask.sum()),
                    "strand_endpoint_m": endpoint_position.round(6).tolist(),
                    "strand_default_endpoint_m": default_endpoint_position.round(6).tolist(),
                    "swage_anchor_m": swage_position.round(6).tolist(),
                    "swage_default_anchor_m": default_swage_position.round(6).tolist(),
                    "swage_to_strand_delta_m": (
                        endpoint_position - swage_position
                    ).round(6).tolist(),
                    "swage_to_strand_distance_m": round(swage_distance, 6),
                    "swage_default_to_strand_distance_m": round(default_swage_distance, 6),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    native_attachment_targets: torch.Tensor | None = None

    def initialize_native_attachment() -> None:
        """Pin the OpenUSD asset's declared attachment band through PhysX."""

        nonlocal native_attachment_targets
        if native_tissue is None or native_room is None:
            native_attachment_targets = None
            return
        default_state_value = native_tissue.data.default_nodal_state_w
        default_state = getattr(default_state_value, "torch", default_state_value).clone()
        target_value = native_tissue.data.nodal_kinematic_target
        targets = getattr(target_value, "torch", target_value).clone()
        positions = default_state[..., :3]
        targets[..., :3] = positions
        targets[..., 3] = 1.0
        attachment = native_room["attachment"]
        axis = {"x": 0, "y": 1, "z": 2}[str(attachment["axis"])]
        coordinates = positions[0, :, axis]
        width_m = float(attachment["width_m"])
        sides = attachment.get("sides") or [attachment["side"]]
        mask = torch.zeros_like(coordinates, dtype=torch.bool)
        for side in sides:
            if side == "minimum":
                mask |= coordinates <= torch.min(coordinates) + width_m
            elif side == "maximum":
                mask |= coordinates >= torch.max(coordinates) - width_m
            else:
                raise RuntimeError(
                    f"Unsupported native tissue attachment side: {side}"
                )
        if int(mask.sum().item()) < 1:
            raise RuntimeError("The native tissue asset exposes no nodes in its attachment region")
        targets[0, mask, :3] = positions[0, mask]
        targets[0, mask, 3] = 0.0
        native_attachment_targets = targets

    def write_native_attachment() -> None:
        if native_tissue is None or native_attachment_targets is None:
            return
        writer = getattr(native_tissue, "write_nodal_kinematic_target_to_sim_index", None)
        if writer is None:
            writer = native_tissue.write_nodal_kinematic_target_to_sim
        writer(native_attachment_targets)
        native_tissue.write_data_to_sim()

    initialize_native_attachment()
    write_native_attachment()
    contact_sensors = {}
    for name in sorted(getattr(scene, "sensors", {}).keys()):
        sensor = scene[name]
        if getattr(sensor.data, "net_forces_w", None) is not None:
            contact_sensors[name] = sensor
    dynamic_patient_runtime = (
        DynamicSurgicalPatient(
            seed=DEFAULT_SCENARIO_SEED,
            procedure_stage="access_open",
            condition=str(
                procedure.get("dynamic_patient_condition", "healthy")
            ),
        )
        if dynamic_abdominal_patient_enabled
        else None
    )
    autonomous_rescue_patient_runtime = (
        DynamicSurgicalPatient(
            seed=DEFAULT_SCENARIO_SEED,
            procedure_stage="access_open",
            condition="healthy",
        )
        if autonomous_rescue_or_enabled
        else None
    )
    autonomous_rescue_runtime = (
        AutonomousRescueORRuntime(
            seed=DEFAULT_SCENARIO_SEED,
            dynamic_patient=autonomous_rescue_patient_runtime,
        )
        if autonomous_rescue_or_enabled
        else None
    )
    rescue_physics_step = -1
    rescue_simulation_time_s = 0.0
    rescue_previous_tool_positions: dict[int, np.ndarray] = {}
    rescue_target_position_w: np.ndarray | None = None
    if autonomous_rescue_or_enabled:
        from pxr import Usd, UsdGeom

        rescue_target_path = (
            "/World/envs/env_0/AutonomousRescueVessel/"
            "Frames/temporary_compression"
        )
        rescue_target_prim = suture_stage.GetPrimAtPath(
            rescue_target_path
        )
        if (
            not rescue_target_prim.IsValid()
            or not UsdGeom.Xformable(rescue_target_prim)
        ):
            raise RuntimeError(
                "Autonomous Rescue OR is missing its authored physical "
                f"target frame: {rescue_target_path}"
            )
        rescue_target_position_w = np.asarray(
            UsdGeom.Xformable(rescue_target_prim)
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            .ExtractTranslation(),
            dtype=np.float32,
        )
    showcase_children: list[Any] = []
    default_showcase_names: set[str] = {"Liver_topo_blender"}
    collision_mesh_count = 0
    anatomy_guard_volumes: list[tuple[np.ndarray, np.ndarray, str]] = []
    anatomy_surface_samples: list[tuple[np.ndarray, np.ndarray, str]] = []
    anatomy_collision_prims: list[Any] = []
    stage = None
    showcase_prim = None
    if (
        organ_usd.is_file()
        and not native_tissue_enabled
        and not _softmimicgen_task
        and not procedure.get("hide_anatomy")
    ):
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

        stage = omni.usd.get_context().get_stage()
        showcase_path = "/World/envs/env_0/LiverShowcase"
        showcase_prim = stage.GetPrimAtPath(showcase_path)
        showcase_children = [
            child for child in showcase_prim.GetChildren() if child.GetName() != "_materials" and child.IsA(UsdGeom.Imageable)
        ]
        drape_path = "/World/envs/env_0/DrAnmarSurgicalDrape"
        drape = UsdGeom.Cube.Define(stage, drape_path)
        drape.CreateSizeAttr(1.0)
        drape_transform = UsdGeom.Xformable(drape.GetPrim())
        drape_transform.ClearXformOpOrder()
        # This is a visual-only cover over the native task table.  Keep its
        # rendered top just below the z=0 physical support plane; otherwise
        # thin native objects such as the 1 mm suture needle are depth-hidden
        # until the robot lifts them above the cover.
        drape_transform.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.12, -0.0015))
        drape_transform.AddScaleOp().Set(Gf.Vec3f(0.62, 0.50, 0.002))
        drape_material = UsdShade.Material.Define(stage, f"{drape_path}/Material")
        drape_shader = UsdShade.Shader.Define(stage, f"{drape_path}/Material/Shader")
        drape_shader.CreateIdAttr("UsdPreviewSurface")
        drape_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.025, 0.12, 0.15))
        drape_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
        drape_material.CreateSurfaceOutput().ConnectToSource(drape_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(drape.GetPrim()).Bind(drape_material)
        focus = str(procedure.get("anatomy_focus", "Liver"))
        if "multi-organ" in focus.lower() or procedure.get("guide_kind") == "navigation":
            default_showcase_names = {child.GetName() for child in showcase_children}
        else:
            focus_lower = focus.lower()
            organ_terms = ("liver", "gallbladder", "bladder", "prostate", "kidney", "pancreas", "spleen")
            selected_terms = {term for term in organ_terms if term in focus_lower} or {focus_lower.replace(" surface", "")}
            matching = {
                child.GetName()
                for child in showcase_children
                if any(term in child.GetName().lower() for term in selected_terms)
            }
            if matching:
                default_showcase_names = matching
        if procedure.get("hide_anatomy"):
            default_showcase_names = set()
        for child in showcase_children:
            if child.GetName() in default_showcase_names:
                UsdGeom.Imageable(child).MakeVisible()
            else:
                UsdGeom.Imageable(child).MakeInvisible()

        organ_colors = {
            "Liver_topo_blender": (0.48, 0.055, 0.035),
            "Gallbladder_topo_blender": (0.20, 0.42, 0.12),
            "Bladder_topo_blender": (0.72, 0.30, 0.28),
            "Prostate_topo_blender": (0.70, 0.45, 0.28),
            "Kidney_topo_blender": (0.48, 0.16, 0.12),
            "Pancreas_topo_blender": (0.78, 0.48, 0.34),
            "Spleen_topo_blender": (0.38, 0.09, 0.16),
        }
        for organ_name, color in organ_colors.items():
            mesh = stage.GetPrimAtPath(f"{showcase_path}/{organ_name}/{organ_name}")
            if not mesh.IsValid():
                continue
            material_path = f"{showcase_path}/DrAnmarMaterials/{organ_name}"
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.38)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            collision_enabled = organ_name in default_showcase_names
            collision_api = UsdPhysics.CollisionAPI.Apply(mesh)
            collision_api.CreateCollisionEnabledAttr().Set(collision_enabled)
            anatomy_collision_prims.append(mesh)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")
            if collision_enabled:
                collision_mesh_count += 1


    def refresh_anatomy_guard_volumes() -> None:
        """Cache visible OpenUSD organ surfaces for shape-aware proximity and a bounds fallback."""
        anatomy_guard_volumes.clear()
        anatomy_surface_samples.clear()
        if stage is None:
            return
        bounds_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        for child in showcase_children:
            if UsdGeom.Imageable(child).ComputeVisibility(Usd.TimeCode.Default()) == UsdGeom.Tokens.invisible:
                continue
            mesh = stage.GetPrimAtPath(f"{showcase_path}/{child.GetName()}/{child.GetName()}")
            if not mesh.IsValid():
                continue
            collision_enabled = UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get()
            if collision_enabled is False:
                continue
            aligned_range = bounds_cache.ComputeWorldBound(mesh).ComputeAlignedRange()
            minimum = np.asarray(tuple(aligned_range.GetMin()), dtype=np.float32)
            maximum = np.asarray(tuple(aligned_range.GetMax()), dtype=np.float32)
            center = (minimum + maximum) * 0.5
            radii = np.maximum((maximum - minimum) * 0.48, np.asarray((0.008, 0.008, 0.008), dtype=np.float32))
            if np.all(np.isfinite(center)) and np.all(np.isfinite(radii)):
                anatomy_guard_volumes.append((center, radii, child.GetName()))
            mesh_points = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
            if mesh_points:
                stride = max(1, int(np.ceil(len(mesh_points) / 3000)))
                transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(mesh)
                sampled = np.asarray(
                    [tuple(transform.Transform(point)) for point in list(mesh_points)[::stride]],
                    dtype=np.float32,
                )
                if len(sampled) and np.all(np.isfinite(sampled)):
                    anatomy_surface_samples.append((sampled, center, child.GetName()))


    def anatomy_surface_query(
        point: np.ndarray,
    ) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
        """Return signed clearance, outward normal, and actual sampled surface point."""
        candidates: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        for points, center, _name in anatomy_surface_samples:
            offsets = point[None, :] - points
            nearest_index = int(np.argmin(np.einsum("ij,ij->i", offsets, offsets)))
            surface = points[nearest_index]
            outward = surface - center
            outward_length = float(np.linalg.norm(outward))
            if outward_length < 1e-6:
                continue
            outward /= outward_length
            clearance = float(np.dot(point - surface, outward))
            candidates.append((abs(clearance), clearance, outward.astype(np.float32), surface.astype(np.float32)))
        if candidates:
            _absolute, clearance, outward, surface = min(candidates, key=lambda item: item[0])
            return clearance, outward, surface
        # Bounds-only fallback for anatomy assets that do not expose readable mesh points.
        for center, radii, _name in anatomy_guard_volumes:
            delta = point - center
            normalized_length = float(np.linalg.norm(delta / radii))
            if normalized_length < 1e-6:
                continue
            clearance = (normalized_length - 1.0) * float(np.min(radii))
            outward = delta / np.square(radii)
            outward_length = float(np.linalg.norm(outward))
            if outward_length < 1e-6:
                continue
            outward /= outward_length
            surface = center + delta / normalized_length
            candidates.append((abs(clearance), clearance, outward.astype(np.float32), surface.astype(np.float32)))
        if not candidates:
            return None, None, None
        _absolute, clearance, outward, surface = min(candidates, key=lambda item: item[0])
        return clearance, outward, surface


    refresh_anatomy_guard_volumes()
    ghost_markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/World/DrAnmarClinicianPath",
            markers={
                "approach": sim_utils.SphereCfg(
                    radius=0.0032,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.95, 0.98),
                        emissive_color=(0.04, 0.45, 0.52),
                        opacity=0.72,
                    ),
                ),
                "manipulation": sim_utils.SphereCfg(
                    radius=0.0036,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.64, 0.18),
                        emissive_color=(0.42, 0.18, 0.02),
                        opacity=0.82,
                    ),
                ),
                "recovery": sim_utils.SphereCfg(
                    radius=0.0032,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.22, 0.94, 0.55),
                        emissive_color=(0.02, 0.36, 0.16),
                        opacity=0.72,
                    ),
                ),
            },
        )
    )
    ghost_markers.visualize(translations=np.zeros((1, 3), dtype=np.float32))
    ghost_markers.set_visibility(False)
    procedure_markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/World/DrAnmarProcedureGuide",
            markers={
                "start": sim_utils.SphereCfg(
                    radius=0.0026,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.08, 0.92, 1.0), emissive_color=(0.01, 0.12, 0.16), opacity=0.42
                    ),
                ),
                "path": sim_utils.SphereCfg(
                    radius=0.0022,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.64, 0.16), emissive_color=(0.12, 0.04, 0.0), opacity=0.34
                    ),
                ),
                "finish": sim_utils.SphereCfg(
                    radius=0.0026,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.20, 0.95, 0.48), emissive_color=(0.01, 0.14, 0.05), opacity=0.42
                    ),
                ),
            },
        )
    )
    visible_procedure_waypoint_index: int | None = None
    shared_camera_pose_cache: dict[str, Any] = {
        "name": None,
        "eye": None,
        "target": None,
    }

    def update_procedure_waypoint_marker(index: int, force: bool = False) -> None:
        """Show one unobtrusive next-step cue instead of covering the field."""
        nonlocal visible_procedure_waypoint_index
        with state.lock:
            guided_markers_active = state.autonomy_mode == "guided"
        if not procedure.get("show_waypoint_markers", True) or not guided_markers_active:
            visible_procedure_waypoint_index = -1
            procedure_markers.set_visibility(False)
            return
        normalized_index = int(index) if 0 <= int(index) < len(room_waypoints) else -1
        if not force and normalized_index == visible_procedure_waypoint_index:
            return
        visible_procedure_waypoint_index = normalized_index
        if normalized_index < 0:
            procedure_markers.set_visibility(False)
            return
        marker_kind = 0 if normalized_index == 0 else 2 if normalized_index == len(room_waypoints) - 1 else 1
        procedure_markers.visualize(
            translations=room_waypoints[normalized_index : normalized_index + 1],
            marker_indices=np.asarray([marker_kind], dtype=np.int32),
        )
        procedure_markers.set_visibility(True)

    def wrist_camera_pose(
        arm: int,
        adjustment: dict[str, float | bool | str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the physical gripper-camera mount pose for one PSM."""
        if arm >= len(robot_names):
            return None
        world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        fallback_axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
        robot_name = robot_names[arm]
        robot = robots[robot_name]
        names = robot_body_names.get(robot_name, [])
        tip_index = next(
            (
                names.index(candidate)
                for candidate in (
                    wrist_tip_name,
                    "psm_tool_tip_link",
                    "endo360_needle",
                    "ecm_end_link",
                )
                if candidate in names
            ),
            None,
        )
        if tip_index is None:
            return None
        rear_index = next(
            (
                names.index(candidate)
                for candidate in (
                    "psm_tool_roll_link",
                    "psm_main_insertion_link_3",
                    "endo360_link",
                    "ecm_yaw_link",
                )
                if candidate in names
            ),
            None,
        )
        positions = (
            robot.data.body_pos_w[0, :, :3]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        tip = positions[tip_index]
        shaft = (
            tip - positions[rear_index]
            if rear_index is not None
            else np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        )
        shaft_norm = float(np.linalg.norm(shaft))
        if shaft_norm < 1e-6:
            return None
        shaft /= shaft_norm
        lateral = np.cross(shaft, world_up)
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm < 1e-6:
            lateral = np.cross(shaft, fallback_axis)
            lateral_norm = float(np.linalg.norm(lateral))
        lateral /= max(lateral_norm, 1e-6)
        camera_up = np.cross(lateral, shaft).astype(np.float32)
        camera_up /= max(float(np.linalg.norm(camera_up)), 1e-6)
        adjustment = adjustment or {}
        adjustable = bool(adjustment.get("enabled", False))
        zoom = float(adjustment.get("zoom", 1.0)) if adjustable else 1.0
        mount_offset = (
            lateral * float(adjustment.get("pan_x_m", 0.0))
            + camera_up * float(adjustment.get("pan_y_m", 0.0))
            if adjustable
            else np.zeros(3, dtype=np.float32)
        )
        adhesive_tool_camera = bool(
            skin_adhesive_enabled
            and arm == skin_adhesive_mounted_arm
        )
        eye = (
            tip
            - shaft
            * CANONICAL_PSM_GRIPPER_PROFILE.camera_backoff_m
            * zoom
            + lateral
            * (
                CANONICAL_PSM_GRIPPER_PROFILE.camera_lateral_offset_m
                + (0.035 if adhesive_tool_camera else 0.0)
            )
            + mount_offset
        )
        aim = shaft
        if adhesive_tool_camera:
            # The applicator is wider and longer than the replaced forceps.
            # Put its wrist camera outside the housing and converge on the
            # dispensing end instead of looking through the mounted body.
            aim = tip + shaft * 0.10 - eye
            aim /= max(float(np.linalg.norm(aim)), 1e-6)
        if adjustable:
            aim = rotate_camera_vector(
                aim,
                camera_up,
                float(adjustment.get("yaw_deg", 0.0)),
            )
            aimed_right = np.cross(aim, camera_up).astype(np.float32)
            aimed_right /= max(float(np.linalg.norm(aimed_right)), 1e-6)
            aim = rotate_camera_vector(
                aim,
                aimed_right,
                float(adjustment.get("pitch_deg", 0.0)),
            )
            aim /= max(float(np.linalg.norm(aim)), 1e-6)
        target = (
            eye
            + aim * CANONICAL_PSM_GRIPPER_PROFILE.camera_lookahead_m
        )
        return eye, target

    def set_shared_camera_pose(
        camera_name: str,
        eye: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Move the shared RTX sensor only when its physical pose changed."""
        cached_eye = shared_camera_pose_cache["eye"]
        cached_target = shared_camera_pose_cache["target"]
        unchanged = (
            shared_camera_pose_cache["name"] == camera_name
            and cached_eye is not None
            and cached_target is not None
            and float(np.max(np.abs(cached_eye - eye))) < 1e-5
            and float(np.max(np.abs(cached_target - target))) < 1e-5
        )
        if unchanged:
            return
        camera.set_world_poses_from_view(
            torch.tensor([eye.tolist()], device=camera.device),
            torch.tensor([target.tolist()], device=camera.device),
        )
        shared_camera_pose_cache["name"] = camera_name
        shared_camera_pose_cache["eye"] = eye.copy()
        shared_camera_pose_cache["target"] = target.copy()

    def update_wrist_camera_poses(
        adjustments_by_name: dict[str, dict[str, float | bool | str]] | None = None,
        active_camera_name: str = "endoscope_left",
    ) -> None:
        """Keep native or shared render sensors on their physical mounts."""
        if shared_camera_renderer:
            if not active_camera_name.startswith("wrist_"):
                return
            try:
                arm = int(active_camera_name.split("_", 1)[1]) - 1
            except (ValueError, IndexError):
                return
            pose = wrist_camera_pose(
                arm,
                (adjustments_by_name or {}).get(active_camera_name, {}),
            )
            if pose is None:
                return
            eye, target = pose
            set_shared_camera_pose(
                active_camera_name,
                eye,
                target,
            )
            return
        for arm, wrist_camera in enumerate(wrist_cameras):
            pose = wrist_camera_pose(
                arm,
                (adjustments_by_name or {}).get(f"wrist_{arm + 1}", {}),
            )
            if pose is None:
                continue
            eye, target = pose
            wrist_camera.set_world_poses_from_view(
                torch.tensor([eye.tolist()], device=wrist_camera.device),
                torch.tensor([target.tolist()], device=wrist_camera.device),
            )

    def tool_position_for_arm(arm: int) -> np.ndarray | None:
        if arm >= len(robot_names):
            return None
        robot_name = robot_names[arm]
        names = robot_body_names.get(robot_name, [])
        tip_index = next(
            (names.index(candidate) for candidate in (wrist_tip_name, "psm_tool_tip_link", "endo360_needle", "ecm_end_link") if candidate in names),
            None,
        )
        if tip_index is None:
            return None
        return robots[robot_name].data.body_pos_w[0, tip_index, :3].detach().cpu().numpy().astype(np.float32)

    def native_gripper_contact_force(arm: int) -> float:
        names = [
            f"gripper_contact_{arm + 1}_jaw_1",
            f"gripper_contact_{arm + 1}_jaw_2",
        ]
        observed = []
        for name in names:
            sensor = contact_sensors.get(name)
            if sensor is None:
                continue
            try:
                forces = sensor.data.net_forces_w[0]
                observed.append(float(torch.linalg.vector_norm(forces, dim=-1).max().detach().cpu().item()))
            except (AttributeError, IndexError, RuntimeError):
                continue
        return max(observed, default=0.0)

    def contact_effect_jaw_force(arm: int, jaw: int) -> float | None:
        """Return only the configured jaw/substrate collision-pair force."""
        sensor = contact_sensors.get(
            f"gripper_contact_{arm + 1}_jaw_{jaw}"
        )
        if sensor is None:
            return None
        filtered = getattr(sensor.data, "force_matrix_w", None)
        if filtered is None:
            return None
        try:
            return float(
                torch.linalg.vector_norm(
                    filtered[0],
                    dim=-1,
                )
                .max()
                .detach()
                .cpu()
                .item()
            )
        except (AttributeError, IndexError, RuntimeError):
            return None

    def jaw_positions_for_arm(
        arm: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Read the two physical jaw poses used to measure aperture."""

        if arm >= len(robot_names):
            return None
        robot_name = robot_names[arm]
        names = robot_body_names.get(robot_name, [])
        try:
            left_index = names.index("psm_tool_gripper1_link")
            right_index = names.index("psm_tool_gripper2_link")
        except ValueError:
            return None
        try:
            positions = robots[robot_name].data.body_pos_w[0]
            left = (
                positions[left_index, :3]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            right = (
                positions[right_index, :3]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        except (AttributeError, IndexError, RuntimeError):
            return None
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            return None
        return left, right

    def suture_segment_positions() -> np.ndarray:
        if suture_segment_view is None:
            return np.empty((0, 3), dtype=np.float64)
        transforms = (
            suture_segment_view.get_transforms()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if (
            transforms.shape != (suture_segment_count, 7)
            or not np.isfinite(transforms).all()
        ):
            finite_rows = (
                int(np.isfinite(transforms).all(axis=1).sum())
                if transforms.ndim == 2 and transforms.shape[1] == 7
                else 0
            )
            first_invalid_row = next(
                (
                    index
                    for index, row in enumerate(transforms)
                    if not np.isfinite(row).all()
                ),
                None,
            )
            raise RuntimeError(
                "The Dr.Anmar live suture tensor state is non-finite or incomplete: "
                f"shape={transforms.shape}, "
                f"finite_rows={finite_rows}/{suture_segment_count}, "
                f"first_invalid_row={first_invalid_row}"
            )
        return transforms[:, :3]

    def suture_body_position(view: Any) -> np.ndarray | None:
        if view is None or view._backend is None or view.count != 1:
            return None
        transforms = (
            view.get_transforms()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if transforms.shape != (1, 7) or not np.isfinite(transforms).all():
            return None
        return transforms[0, :3].astype(np.float32)

    def suture_render_positions() -> np.ndarray:
        segment_positions = suture_segment_positions()
        interface_position = suture_body_position(suture_interface_view)
        if interface_position is None:
            if len(segment_positions) == 0:
                return segment_positions
            interface_position = segment_positions[0]
        return np.concatenate(
            (interface_position.reshape(1, 3), segment_positions),
            axis=0,
        )

    def disabled_suture_curve_update(
        _world_positions: np.ndarray,
    ) -> None:
        return

    update_realtime_suture_curve = disabled_suture_curve_update
    if dr_anmar_needle_enabled and not suture_native_segment_rendering:
        # Present the native PhysX strand to RTX as one dynamic curve. Drawing
        # one detailed mesh per physical segment forces Hydra to synchronize
        # every moving visual prim on every camera frame. The curve reads the
        # native segment poses and retains the authored 0.25 mm diameter.
        realtime_suture_curve = UsdGeom.BasisCurves.Define(
            suture_stage,
            f"{suture_root_path}/Suture/RealtimeVisual",
        )
        realtime_suture_curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        realtime_suture_curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        realtime_suture_curve.CreateCurveVertexCountsAttr(
            Vt.IntArray([suture_segment_count + 1])
        )
        realtime_suture_curve.CreateWidthsAttr(Vt.FloatArray([0.00025]))
        realtime_suture_curve.SetWidthsInterpolation(
            UsdGeom.Tokens.constant
        )
        realtime_suture_curve.CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(0.86, 0.82, 0.72)])
        )
        realtime_suture_material = UsdShade.Material.Get(
            suture_stage,
            f"{suture_root_path}/Suture/Looks/SutureVisual",
        )
        if realtime_suture_material.GetPrim().IsValid():
            UsdShade.MaterialBindingAPI.Apply(
                realtime_suture_curve.GetPrim()
            ).Bind(realtime_suture_material)
        for segment_index in range(suture_segment_count):
            segment_visual = suture_stage.GetPrimAtPath(
                f"{suture_root_path}/Suture/Segments/"
                f"S{segment_index:04d}/Visual"
            )
            if segment_visual.IsValid():
                UsdGeom.Imageable(segment_visual).MakeInvisible()
        suture_root_world_inverse = (
            UsdGeom.Xformable(suture_root)
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            .GetInverse()
        )

        def realtime_suture_local_points(
            world_positions: np.ndarray,
        ) -> list[Gf.Vec3d]:
            return [
                suture_root_world_inverse.Transform(
                    Gf.Vec3d(
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                    )
                )
                for point in world_positions
            ]

        initial_suture_world_positions = suture_render_positions()
        initial_suture_local_points = realtime_suture_local_points(
            initial_suture_world_positions
        )
        realtime_suture_curve.CreatePointsAttr().Set(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                    )
                    for point in initial_suture_local_points
                ]
            )
        )
        initial_suture_local_array = np.asarray(
            [
                (float(point[0]), float(point[1]), float(point[2]))
                for point in initial_suture_local_points
            ],
            dtype=np.float32,
        )
        suture_extent_padding_m = max(
            0.25,
            float(
                np.linalg.norm(
                    np.ptp(initial_suture_local_array, axis=0)
                )
            )
            * 2.0,
        )
        realtime_suture_curve.CreateExtentAttr().Set(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(
                        *(
                            initial_suture_local_array.min(axis=0)
                            - suture_extent_padding_m
                        ).tolist()
                    ),
                    Gf.Vec3f(
                        *(
                            initial_suture_local_array.max(axis=0)
                            + suture_extent_padding_m
                        ).tolist()
                    ),
                ]
            )
        )

        realtime_suture_points = realtime_suture_curve.GetPointsAttr()

        def usd_suture_curve_update(
            world_positions: np.ndarray,
        ) -> None:
            local_points = realtime_suture_local_points(
                world_positions
            )
            realtime_suture_points.Set(
                Vt.Vec3fArray(
                    [
                        Gf.Vec3f(
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                        )
                        for point in local_points
                    ]
                )
            )

        update_realtime_suture_curve = usd_suture_curve_update
        update_realtime_suture_curve(
            initial_suture_world_positions
        )

    def apply_endoscope_camera_view(
        selected_scenario: str,
        view_mode: str,
        adjustment: dict[str, float] | None = None,
        active_camera_name: str = "endoscope_left",
    ) -> None:
        selected_eye, selected_target = scenario_camera_pose(camera_eye, camera_target, selected_scenario)
        base_mode = str((adjustment or {}).get("base_mode", "operative")) if view_mode == "free" else view_mode
        selected_eye, selected_target = camera_view_pose(selected_eye, selected_target, base_mode)
        if view_mode == "free" and adjustment:
            selected_eye, selected_target = adjustable_camera_pose(
                selected_eye,
                selected_target,
                float(adjustment.get("yaw_deg", 0.0)),
                float(adjustment.get("pitch_deg", 0.0)),
                float(adjustment.get("zoom", 1.0)),
                float(adjustment.get("pan_x_m", 0.0)),
                float(adjustment.get("pan_y_m", 0.0)),
            )
        right_offset = SCENARIO_NATIVE_PROFILES.get(selected_scenario, {}).get(
            "right_camera_offset_m", (0.0, 0.006, 0.0)
        )
        selected_right_eye = selected_eye + np.asarray(right_offset, dtype=np.float32)
        primary_eye = (
            selected_right_eye
            if shared_camera_renderer
            and active_camera_name == "endoscope_right"
            else selected_eye
        )
        if shared_camera_renderer:
            set_shared_camera_pose(
                active_camera_name,
                primary_eye,
                selected_target,
            )
        else:
            camera.set_world_poses_from_view(
                torch.tensor([primary_eye.tolist()], device=camera.device),
                torch.tensor([selected_target.tolist()], device=camera.device),
            )
        if stereo_right_camera is not None:
            stereo_right_camera.set_world_poses_from_view(
                torch.tensor([selected_right_eye.tolist()], device=stereo_right_camera.device),
                torch.tensor([selected_target.tolist()], device=stereo_right_camera.device),
            )

    anatomy_showcase_position_w = np.asarray(anatomy_position, dtype=np.float32)
    anatomy_showcase_quaternion_w = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    if showcase_prim is not None and showcase_prim.IsValid():
        try:
            anatomy_world = UsdGeom.Xformable(showcase_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = anatomy_world.ExtractTranslation()
            rotation = anatomy_world.ExtractRotationQuat()
            imaginary = rotation.GetImaginary()
            anatomy_showcase_position_w = np.asarray(tuple(translation), dtype=np.float32)
            anatomy_showcase_quaternion_w = np.asarray(
                (rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]), dtype=np.float32
            )
        except (AttributeError, RuntimeError):
            pass

    action_dim = int(env.action_space.shape[-1])
    arms = min(2, len(robot_names))
    has_grippers = action_dim >= arms * 7
    psm_scene_names = [
        name
        for name in robot_names
        if all(joint_name in tuple(robots[name].joint_names) for joint_name in PSM_ARM_NAMES)
    ]
    native_psm_policy_dim = 0
    if psm_scene_names:
        with torch.inference_mode():
            initial_policy_action, initial_joint_targets, native_robot_names = canonical_policy_contract(env)
        native_psm_policy_dim = int(initial_policy_action.shape[-1])
        if tuple(psm_scene_names) != native_robot_names:
            raise RuntimeError(
                f"PSM scene order {tuple(psm_scene_names)} does not match the native action contract "
                f"{native_robot_names}"
            )
        if initial_joint_targets.shape[-1] != native_psm_policy_dim:
            raise RuntimeError("NVIDIA PSM policy action and resolved target dimensions disagree")
    native_ik_scales = (
        native_ik_action_scales(env, psm_scene_names)
        if len(psm_scene_names) == arms
        else []
    )
    apply_endoscope_camera_view(
        "baseline",
        "free",
        {
            "base_mode": "operative",
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "zoom": 1.0,
            "pan_x_m": 0.0,
            "pan_y_m": 0.0,
        },
    )
    update_wrist_camera_poses()

    stapler_magazine = StapleMagazine(capacity=35, remaining=35)
    stapler_deployment_controller = TriggerEdgeDeploymentController(
        magazine=stapler_magazine,
    )
    stapler_cycle_started_at: float | None = None
    stapler_cycle_threshold_at: float | None = None
    stapler_cycle_release_started_at: float | None = None
    stapler_cycle_count = 0
    stapler_last_event: dict[str, Any] | None = None
    stapler_partial_candidate = False
    stapler_partial_start_deployments = 0
    stapler_partial_peak_deg = 0.0
    stapler_partial_stroke_attempts = 0
    stapler_partial_stroke_passes = 0
    stapler_closure_line = ClosureLine(
        (
            STAPLER_CLOSURE_TARGET_CENTER_M[0]
            + STAPLER_CLOSURE_STATION_OFFSETS_M[0],
            STAPLER_CLOSURE_TARGET_CENTER_M[1],
            STAPLER_CLOSURE_TARGET_CENTER_M[2],
        ),
        (
            STAPLER_CLOSURE_TARGET_CENTER_M[0]
            + STAPLER_CLOSURE_STATION_OFFSETS_M[-1],
            STAPLER_CLOSURE_TARGET_CENTER_M[1],
            STAPLER_CLOSURE_TARGET_CENTER_M[2],
        ),
    )
    stapler_closure_targets_m = tuple(
        (
            STAPLER_CLOSURE_TARGET_CENTER_M[0] + station_offset,
            STAPLER_CLOSURE_TARGET_CENTER_M[1],
            STAPLER_CLOSURE_TARGET_CENTER_M[2],
        )
        for station_offset in STAPLER_CLOSURE_STATION_OFFSETS_M
    )
    stapler_active_station_index = 0
    stapler_station_settle_until = 0.0
    stapler_closed_station_indices: set[int] = set()
    stapler_pending_advance = False
    stapler_last_placement: dict[str, Any] | None = None
    stapler_visual_root_path = "/World/envs/env_0/StaplerClosurePlacements"
    stapler_tissue_default_state: torch.Tensor | None = None
    stapler_tissue_default_states: list[torch.Tensor] = []
    stapler_tissue_node_slices: list[slice] = []
    stapler_tissue_default_positions: torch.Tensor | None = None
    stapler_tissue_base_targets: torch.Tensor | None = None
    stapler_tissue_station_masks: list[torch.Tensor] = []
    stapler_tissue_station_closed_positions: list[torch.Tensor] = []
    stapler_tissue_initial_gaps_mm: list[float] = []
    stapler_tissue_outer_anchor_count = 0
    stapler_tissue_detected_wound_axis = "unknown"
    stapler_tissue_detected_source_gap_mm = 0.0
    stapler_tissue_wound_axis = 1
    stapler_tissue_longitudinal_axis = 0
    stapler_tissue_wound_center = float(
        STAPLER_CLOSURE_TARGET_CENTER_M[1]
    )
    stapler_tissue_longitudinal_center = float(
        STAPLER_CLOSURE_TARGET_CENTER_M[0]
    )
    stapler_tissue_station_longitudinal_coordinates = [
        float(target[0]) for target in stapler_closure_targets_m
    ]
    stapler_tissue_approximation_progress = 0.0
    stapler_tissue_max_displacement_mm = 0.0
    stapler_approximation_paddle_translate_ops: dict[
        str,
        Any,
    ] = {}
    if stapler_closure_tissues:
        target_parts: list[torch.Tensor] = []
        node_start = 0
        for tissue_flap in stapler_closure_tissues:
            default_state_value = (
                tissue_flap.data.nodal_state_w
            )
            flap_default_state = getattr(
                default_state_value,
                "torch",
                default_state_value,
            ).clone()
            flap_default_state[..., 3:] = 0.0
            stapler_tissue_default_states.append(
                flap_default_state
            )
            node_stop = node_start + flap_default_state.shape[1]
            stapler_tissue_node_slices.append(
                slice(node_start, node_stop)
            )
            node_start = node_stop
            target_value = tissue_flap.data.nodal_kinematic_target
            target_parts.append(
                getattr(target_value, "torch", target_value).clone()
            )
        stapler_tissue_default_state = torch.cat(
            stapler_tissue_default_states,
            dim=1,
        )
        # PhysX exposes cooked simulation vertices in the USD asset's authored
        # coordinate frame rather than the visual prim's intuitive local
        # frame. The authored wound axis is the wider in-plane dimension, so
        # detect its robust span and map the FEM mesh into the test-cell frame.
        # This keeps both tissue flaps, their tetrahedra, and their separation
        # intact without assuming how the imported USD was authored.
        cooked_positions = stapler_tissue_default_state[0, :, :3]
        in_plane_spans: list[float] = []
        in_plane_centers: list[float] = []
        for candidate_axis in (0, 1):
            ordered_coordinates = torch.sort(
                cooked_positions[:, candidate_axis]
            ).values
            lower_quantile = torch.quantile(
                ordered_coordinates,
                0.05,
            )
            upper_quantile = torch.quantile(
                ordered_coordinates,
                0.95,
            )
            in_plane_spans.append(
                float((upper_quantile - lower_quantile).item())
            )
            in_plane_centers.append(
                float(
                    ((lower_quantile + upper_quantile) * 0.5).item()
                )
            )
        wound_axis = max(
            range(2),
            key=lambda axis_index: in_plane_spans[axis_index],
        )
        longitudinal_axis = 1 - wound_axis
        wound_coordinates = cooked_positions[:, wound_axis]
        wound_center = in_plane_centers[wound_axis]
        raw_left_component = wound_coordinates < wound_center
        raw_right_component = ~raw_left_component
        wound_left_inner = torch.max(
            wound_coordinates[raw_left_component]
        )
        wound_right_inner = torch.min(
            wound_coordinates[raw_right_component]
        )
        detected_gap_m = float(
            (wound_right_inner - wound_left_inner).item()
        )
        stapler_tissue_default_state[0, :, 3:] = 0.0
        stapler_tissue_wound_axis = wound_axis
        stapler_tissue_longitudinal_axis = longitudinal_axis
        stapler_tissue_wound_center = wound_center
        stapler_tissue_longitudinal_center = in_plane_centers[
            longitudinal_axis
        ]
        stapler_tissue_station_longitudinal_coordinates = [
            stapler_tissue_longitudinal_center + offset
            for offset in STAPLER_CLOSURE_STATION_OFFSETS_M
        ]
        stapler_tissue_detected_wound_axis = (
            "x" if wound_axis == 0 else "y"
        )
        stapler_tissue_detected_source_gap_mm = (
            detected_gap_m * 1000.0
        )
        stapler_tissue_default_positions = (
            stapler_tissue_default_state[..., :3].clone()
        )
        stapler_tissue_base_targets = torch.cat(
            target_parts,
            dim=1,
        )
        stapler_tissue_base_targets[..., :3] = (
            stapler_tissue_default_positions
        )
        stapler_tissue_base_targets[..., 3] = 1.0
        default_positions = stapler_tissue_default_positions[0]
        closure_center_y = stapler_tissue_wound_center
        lateral_distance = torch.abs(
            default_positions[:, stapler_tissue_wound_axis]
            - closure_center_y
        )
        outer_anchor_threshold = torch.quantile(
            lateral_distance,
            0.60,
        )
        outer_anchor_mask = lateral_distance >= outer_anchor_threshold
        stapler_tissue_base_targets[
            0,
            outer_anchor_mask,
            :3,
        ] = default_positions[outer_anchor_mask]
        stapler_tissue_base_targets[
            0,
            outer_anchor_mask,
            3,
        ] = 0.0
        stapler_tissue_outer_anchor_count = int(
            outer_anchor_mask.sum().item()
        )
        for station_index, target_position_m in enumerate(
            stapler_closure_targets_m
        ):
            tissue_station_coordinate = (
                stapler_tissue_station_longitudinal_coordinates[
                    station_index
                ]
            )
            longitudinal_mask = (
                torch.abs(
                    default_positions[
                        :,
                        stapler_tissue_longitudinal_axis,
                    ]
                    - tissue_station_coordinate
                )
                <= STAPLER_TISSUE_STATION_HALF_WIDTH_M
            )
            left_flap = (
                default_positions[:, stapler_tissue_wound_axis]
                < closure_center_y
            )
            right_flap = (
                default_positions[:, stapler_tissue_wound_axis]
                > closure_center_y
            )
            left_candidates = longitudinal_mask & left_flap
            right_candidates = longitudinal_mask & right_flap
            minimum_station_candidates = 2
            if (
                int(left_candidates.sum().item())
                < minimum_station_candidates
            ):
                left_flap_indices = torch.nonzero(
                    left_flap,
                    as_tuple=False,
                ).flatten()
                nearest_left_longitudinal = left_flap_indices[
                    torch.argsort(
                        torch.abs(
                            default_positions[
                                left_flap_indices,
                                stapler_tissue_longitudinal_axis,
                            ]
                            - tissue_station_coordinate
                        )
                    )[:minimum_station_candidates]
                ]
                left_candidates[nearest_left_longitudinal] = True
            if (
                int(right_candidates.sum().item())
                < minimum_station_candidates
            ):
                right_flap_indices = torch.nonzero(
                    right_flap,
                    as_tuple=False,
                ).flatten()
                nearest_right_longitudinal = right_flap_indices[
                    torch.argsort(
                        torch.abs(
                            default_positions[
                                right_flap_indices,
                                stapler_tissue_longitudinal_axis,
                            ]
                            - tissue_station_coordinate
                        )
                    )[:minimum_station_candidates]
                ]
                right_candidates[
                    nearest_right_longitudinal
                ] = True
            if (
                not bool(left_candidates.any())
                or not bool(right_candidates.any())
            ):
                raise RuntimeError(
                    "The stapler FEM tissue is missing a flap at closure "
                    f"station x={target_position_m[0]:.6f}"
                )
            left_inner = torch.max(
                default_positions[
                    left_candidates,
                    stapler_tissue_wound_axis,
                ]
            )
            right_inner = torch.min(
                default_positions[
                    right_candidates,
                    stapler_tissue_wound_axis,
                ]
            )
            left_edge_distance = (
                left_inner
                - default_positions[:, stapler_tissue_wound_axis]
            )
            right_edge_distance = (
                default_positions[:, stapler_tissue_wound_axis]
                - right_inner
            )
            left_candidate_indices = torch.nonzero(
                left_candidates,
                as_tuple=False,
            ).flatten()
            right_candidate_indices = torch.nonzero(
                right_candidates,
                as_tuple=False,
            ).flatten()
            left_mask = torch.zeros_like(left_candidates)
            right_mask = torch.zeros_like(right_candidates)
            nearest_left_indices = left_candidate_indices[
                torch.argsort(
                    left_edge_distance[left_candidate_indices]
                )[:2]
            ]
            nearest_right_indices = right_candidate_indices[
                torch.argsort(
                    right_edge_distance[right_candidate_indices]
                )[:2]
            ]
            left_mask[nearest_left_indices] = True
            right_mask[nearest_right_indices] = True
            if (
                int(left_mask.sum().item()) < 2
                or int(right_mask.sum().item()) < 2
            ):
                raise RuntimeError(
                    "The stapler FEM tissue exposes too few wound-edge "
                    "nodes at a closure station: "
                    f"target_x={target_position_m[0]:.6f}, "
                    f"x_extent=({float(default_positions[:, 0].min().item()):.6f},"
                    f"{float(default_positions[:, 0].max().item()):.6f}), "
                    f"y_extent=({float(default_positions[:, 1].min().item()):.6f},"
                    f"{float(default_positions[:, 1].max().item()):.6f}), "
                    f"x_q=({float(torch.quantile(default_positions[:, 0], 0.05).item()):.6f},"
                    f"{float(torch.quantile(default_positions[:, 0], 0.50).item()):.6f},"
                    f"{float(torch.quantile(default_positions[:, 0], 0.95).item()):.6f}), "
                    f"y_q=({float(torch.quantile(default_positions[:, 1], 0.05).item()):.6f},"
                    f"{float(torch.quantile(default_positions[:, 1], 0.50).item()):.6f},"
                    f"{float(torch.quantile(default_positions[:, 1], 0.95).item()):.6f}), "
                    f"left_candidates={int(left_candidates.sum().item())}, "
                    f"right_candidates={int(right_candidates.sum().item())}, "
                    f"left={int(left_mask.sum().item())}, "
                    f"right={int(right_mask.sum().item())}"
                )
            stapler_tissue_initial_gaps_mm.append(
                float((right_inner - left_inner).item() * 1000.0)
            )
            station_closed_positions = default_positions.clone()
            target_half_gap = STAPLER_TISSUE_TARGET_GAP_M / 2.0
            full_capture_distance = 0.0020
            blend_span = max(
                1.0e-6,
                STAPLER_TISSUE_EDGE_CAPTURE_M
                - full_capture_distance,
            )
            left_capture_weights = torch.clamp(
                (
                    STAPLER_TISSUE_EDGE_CAPTURE_M
                    - left_edge_distance
                )
                / blend_span,
                0.0,
                1.0,
            )
            right_capture_weights = torch.clamp(
                (
                    STAPLER_TISSUE_EDGE_CAPTURE_M
                    - right_edge_distance
                )
                / blend_span,
                0.0,
                1.0,
            )
            left_capture_weights[left_mask] = torch.clamp(
                left_capture_weights[left_mask],
                min=0.25,
            )
            right_capture_weights[right_mask] = torch.clamp(
                right_capture_weights[right_mask],
                min=0.25,
            )
            station_closed_positions[
                left_mask,
                stapler_tissue_wound_axis,
            ] = (
                default_positions[
                    left_mask,
                    stapler_tissue_wound_axis,
                ]
                + left_capture_weights[left_mask]
                * (
                    closure_center_y
                    - target_half_gap
                    - left_inner
                )
            )
            station_closed_positions[
                right_mask,
                stapler_tissue_wound_axis,
            ] = (
                default_positions[
                    right_mask,
                    stapler_tissue_wound_axis,
                ]
                + right_capture_weights[right_mask]
                * (
                    closure_center_y
                    + target_half_gap
                    - right_inner
                )
            )
            stapler_tissue_station_masks.append(
                left_mask | right_mask
            )
            stapler_tissue_station_closed_positions.append(
                station_closed_positions
            )
        approximation_root_path = (
            "/World/envs/env_0/StaplerTissueApproximator"
        )
        suture_stage.DefinePrim(approximation_root_path, "Xform")
        for side, color in (
            ("Left", (0.16, 0.62, 0.72)),
            ("Right", (0.16, 0.62, 0.72)),
        ):
            paddle_path = (
                f"{approximation_root_path}/{side}ApproximationFoot"
            )
            paddle = UsdGeom.Cube.Define(
                suture_stage,
                paddle_path,
            )
            paddle.CreateSizeAttr(1.0)
            paddle.CreateDisplayColorAttr(
                [Gf.Vec3f(*color)]
            )
            xformable = UsdGeom.Xformable(paddle.GetPrim())
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble
            )
            xformable.AddScaleOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Vec3d(0.0042, 0.0012, 0.0008))
            stapler_approximation_paddle_translate_ops[
                side.lower()
            ] = translate_op
    state = SharedState(
        task=args_cli.task,
        camera_width=interactive_camera_width,
        camera_height=interactive_camera_height,
        demo_dir=args_cli.demo_dir,
        action_dim=action_dim,
        arms=arms,
        has_grippers=has_grippers,
        robot_names=robot_names,
        robot_body_names=robot_body_names,
        anatomy_showcase=str(procedure.get("anatomy_focus") or "Operative field"),
        anatomy_scene_id=args_cli.anatomy_scene_id,
        anatomy_asset=(
            str(organ_usd)
            if organ_usd.is_file()
            and not _softmimicgen_task
            and not procedure.get("hide_anatomy")
            else ""
        ),
        openusd_environment=str(openusd_environment) if openusd_environment else "",
        procedure=procedure,
        dynamic_patient_access_state=(
            str(procedure.get("dynamic_patient_access_state", "intact"))
            if dynamic_abdominal_patient_enabled
            else ""
        ),
        openusd_scene_loaded=bool(
            nvidia_native_bench
            or native_deformable_enabled
            or native_static_collision_enabled
            or (openusd_environment and organ_usd.is_file() and showcase_children)
        ),
        anatomy_collision_meshes=collision_mesh_count,
        sensor_profile=args_cli.sensor_profile,
        needle_visual_ready=bool(
            "suture_needle" in objects
            or "dr_anmar_standalone_needle" in objects
            or "dr_anmar_needle_v030" in objects
            or "dr_anmar_threaded_needle" in objects
            or suture_body_position(suture_needle_view) is not None
            or "object" in objects
            or selected_bench_assets.intersection(
                {
                    "dr_anmar_needle_thread_coiled",
                    "dr_anmar_needle_thread_extended",
                    "dr_anmar_needle_thread_proxy",
                }
            )
        ),
        deformable_strand_ready=bool(
            "object" in deformables
            or "dr_anmar_native_suture" in deformables
            or selected_bench_assets.intersection(
                {
                    "dr_anmar_needle_thread_coiled",
                    "dr_anmar_needle_thread_extended",
                    "dr_anmar_needle_thread_proxy",
                }
            )
            or (
                suture_segment_view is not None
                and suture_segment_view._backend is not None
                and suture_segment_view.count == suture_segment_count
            )
        ),
        native_rigid_object_names=object_names,
        native_deformable_object_names=deformable_names,
        dr_anmar_needle_domain=initial_dr_anmar_needle_domain,
        native_psm_policy_contract=bool(psm_scene_names),
        native_psm_policy_dim=native_psm_policy_dim,
        native_psm_robot_names=psm_scene_names,
        native_ik_scales=native_ik_scales,
        gripper_profile=configured_psm_gripper_profile,
        ring_physics_ready=ring_physics_ready,
        strand_self_collision_ready=strand_self_collision_ready,
        stapler_test_cell={
            "enabled": stapler_test_cell_enabled,
            "mode": "fem_preapproximation_and_retained_staple_constraints",
            "cycle_phase": "ready" if stapler_test_cell_enabled else "disabled",
            "cycle_running": False,
            "target_trigger_deg": 0.0,
            "actual_trigger_deg": 0.0,
            "pusher_travel_mm": 0.0,
            "max_pusher_travel_mm": 0.0,
            "joint_limit_violation_deg": 0.0,
            "fixture_translation_error_mm": 0.0,
            "fixture_rotation_error_deg": 0.0,
            "max_fixture_translation_error_mm": 0.0,
            "max_fixture_rotation_error_deg": 0.0,
            "fire_threshold_deg": FIRE_THRESHOLD_DEG,
            "rearm_threshold_deg": REARM_THRESHOLD_DEG,
            "magazine_capacity": stapler_magazine.capacity,
            "magazine_remaining": stapler_magazine.remaining,
            "deployment_count": 0,
            "cycle_count": 0,
            "partial_stroke_attempts": 0,
            "partial_stroke_passes": 0,
            "tissue_asset_id": "dr-anmar-suturable-tissue",
            "tissue_runtime": "physx_fem_two_flap_deformable",
            "closure_model": "preapproximation_then_rigid_staple_fem_retention",
            "detected_wound_axis": stapler_tissue_detected_wound_axis,
            "detected_source_gap_mm": round(
                stapler_tissue_detected_source_gap_mm,
                4,
            ),
            "approximation_duration_s": (
                STAPLER_TISSUE_APPROXIMATION_DURATION_S
            ),
            "approximation_progress_percent": 0.0,
            "initial_tissue_gap_mm": (
                round(stapler_tissue_initial_gaps_mm[0], 4)
                if stapler_tissue_initial_gaps_mm
                else None
            ),
            "tissue_gap_mm": (
                round(stapler_tissue_initial_gaps_mm[0], 4)
                if stapler_tissue_initial_gaps_mm
                else None
            ),
            "target_tissue_gap_mm": (
                STAPLER_TISSUE_TARGET_GAP_M * 1000.0
            ),
            "tissue_max_displacement_mm": 0.0,
            "outer_anchor_node_count": (
                stapler_tissue_outer_anchor_count
            ),
            "retained_attachment_count": 0,
            "retained_node_count": 0,
            "retained_verified_count": 0,
            "retained_station_gaps_mm": {},
            "retention_state": "open",
            "staple_rigid_body_mode": (
                "kinematic_retainer_with_collision"
            ),
            "tissue_attachment_mode": (
                "fem_nodal_staple_leg_constraint"
            ),
            "tissue_material": dict(
                stapler_tissue_material_runtime
            ),
            "tissue_episode": dict(
                stapler_tissue_episode_payload
            ),
            "station_index": 1,
            "station_count": len(STAPLER_CLOSURE_STATION_OFFSETS_M),
            "station_spacing_mm": STAPLER_CLOSURE_STATION_SPACING_M
            * 1000.0,
            "station_state": "open",
            "station_ready": True,
            "closed_station_count": 0,
            "closed_station_indices": [],
            "closure_progress_percent": 0.0,
            "closure_complete": False,
            "current_target_m": list(stapler_closure_targets_m[0]),
            "max_spacing_error_mm": 0.0,
            "last_placement": None,
            "last_event": None,
            "parameter_status": "provisional_unmeasured",
            "clinical_validation": False,
        },
        skin_adhesive_system={
            "enabled": skin_adhesive_enabled,
            "asset_id": "dranmar-skin-adhesive-system-v1",
            "version": "0.1.0",
            "applicator_state": (
                "activated" if skin_adhesive_enabled else "disabled"
            ),
            "target_activation": 0.0,
            "actual_activation": 0.0,
            "left_paddle_deg": 0.0,
            "right_paddle_deg": 0.0,
            "piston_travel_mm": 0.0,
            "workflow_state": (
                "mounted_ready" if skin_adhesive_enabled else "disabled"
            ),
            "mounted_arm": (
                skin_adhesive_mounted_arm + 1
                if skin_adhesive_enabled
                else None
            ),
            "tool_type": "topical_skin_adhesive_end_effector",
            "mount_type": "physx_fixed_joint",
            "mount_joint": "skin_adhesive_mount_joint",
            "pose_write_attachment": False,
            "jaw_geometry_replaced": skin_adhesive_enabled,
            "control_mapping": "activation_equals_one_minus_aperture",
            "outlet_state": "exposed" if skin_adhesive_enabled else "disabled",
            "mechanism": "coordinated_dual_paddle_and_metering_piston",
            "deposit_representation": "none",
            "material_release_model": "not_simulated",
            "fluid_solver": False,
            "curing_solver": False,
            "clinical_validation": False,
        },
        closure_robot_system={
            "enabled": closure_robot_enabled,
            "asset_id": (
                "dranmar-approximate-staple-seal-end-effector-v1"
            ),
            "version": "0.1.0",
            "phase": "ready" if closure_robot_enabled else "disabled",
            "cycle_running": False,
            "cycle_complete": False,
            "mount_robot": "Franka Panda",
            "mount_link": "panda_link8",
            "stock_hand_active": False,
            "fixed_mount": True,
            "articulation_joint_count": (
                len(closure_robot_articulation.joint_names)
                if closure_robot_articulation is not None
                else 0
            ),
            "articulation_body_count": (
                len(closure_robot_articulation.body_names)
                if closure_robot_articulation is not None
                else 0
            ),
            "arm_hold": "gravity_compensated_physical_joint_position_drive",
            "mount_translation_error_mm": 0.0,
            "max_mount_translation_error_mm": 0.0,
            "joint_target_source": "measured_phase_gated_sequence",
            "left_approximation_mm": 0.0,
            "right_approximation_mm": 0.0,
            "left_clamp_deg": 28.0,
            "right_clamp_deg": -28.0,
            "staple_driver_mm": 0.0,
            "adhesive_deploy_mm": 0.0,
            "adhesive_meter_mm": 0.0,
            "temporary_capture_attachment_count": 0,
            "formed_staple_count": 0,
            "staple_attachment_count": 0,
            "adhesive_bead_count": 0,
            "adhesive_bond_attachment_count": 0,
            "attachment_prim_count": 0,
            "attachment_enabled_count": 0,
            "attachment_actor_pair_count": 0,
            "attachment_auto_overlap_count": 0,
            "attachment_explicit_point_count": 0,
            "tissue_backend": "physx_surface_deformable",
            "attachment_backend": "PhysxPhysicsAttachment",
            "transform_writes": False,
            "kinematic_tissue_motion": False,
            "staple_dynamic_rigid_body": True,
            "parameter_status": "provisional_unmeasured",
            "clinical_validation": False,
        },
    )
    closure_cycle_running = False
    closure_cycle_phase = (
        ClosurePhase.READY
        if closure_robot_enabled
        else None
    )
    closure_phase_started_at = time.monotonic()
    closure_hold_targets = (
        closure_phase_targets(ClosurePhase.READY)
        if closure_robot_enabled
        else {}
    )
    closure_staple_deployed = False
    closure_bead_deposited = False

    state.simulation_profile = {
        "scene_authority": "OpenUSD",
        "simulation_authority": "Isaac Lab",
        "physics_authority": "NVIDIA PhysX",
        "device": str(env.unwrapped.device),
        "physics_dt_s": float(env_cfg.sim.dt),
        "action_decimation": int(env_cfg.decimation),
        "action_period_s": float(env_cfg.sim.dt * env_cfg.decimation),
        "scene_ccd_enabled": bool(
            getattr(env_cfg.sim.physx, "enable_ccd", False)
        ),
        "render_interval_physics_steps": int(env_cfg.sim.render_interval),
        "single_active_camera_renderer": shared_camera_renderer,
        "native_segment_rendering": suture_native_segment_rendering,
        "suture_physics_lod": (
            suture_physics_lod if dr_anmar_needle_enabled else None
        ),
    }

    def refresh_hand_camera_control_frame() -> None:
        """Align camera forward/right/up with each PSM's native IK root frame."""

        cached_eye = shared_camera_pose_cache.get("eye")
        cached_target = shared_camera_pose_cache.get("target")
        if cached_eye is None or cached_target is None:
            return
        eye = np.asarray(cached_eye, dtype=np.float64)
        target = np.asarray(cached_target, dtype=np.float64)
        forward = target - eye
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1.0e-7:
            return
        forward /= forward_norm
        world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        right = np.cross(forward, world_up)
        if float(np.linalg.norm(right)) < 1.0e-6:
            right = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        right /= max(float(np.linalg.norm(right)), 1.0e-7)
        camera_up = np.cross(right, forward)
        camera_up /= max(float(np.linalg.norm(camera_up)), 1.0e-7)
        camera_basis_world = np.column_stack((forward, right, camera_up))
        bases: list[list[list[float]]] = []
        for arm, robot_name in enumerate(robot_names[: state.arms]):
            quaternion = (
                robots[robot_name]
                .data.root_quat_w[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
            w, x, y, z = quaternion
            root_rotation_world = np.asarray(
                (
                    (
                        1.0 - 2.0 * (y * y + z * z),
                        2.0 * (x * y - z * w),
                        2.0 * (x * z + y * w),
                    ),
                    (
                        2.0 * (x * y + z * w),
                        1.0 - 2.0 * (x * x + z * z),
                        2.0 * (y * z - x * w),
                    ),
                    (
                        2.0 * (x * z - y * w),
                        2.0 * (y * z + x * w),
                        1.0 - 2.0 * (x * x + y * y),
                    ),
                ),
                dtype=np.float64,
            )
            action_basis = root_rotation_world.T @ camera_basis_world
            bases.append(action_basis.round(9).tolist())
        if len(bases) != state.arms:
            return
        control_name = str(
            shared_camera_pose_cache.get("name") or "endoscope_left"
        )
        with state.lock:
            previous = state.hand_camera_to_action_basis
            changed = (
                state.hand_camera_control_name != control_name
                or len(previous) != len(bases)
                or not np.allclose(
                    np.asarray(previous, dtype=np.float64),
                    np.asarray(bases, dtype=np.float64),
                    atol=1.0e-6,
                    rtol=0.0,
                )
            )
            if changed:
                if previous:
                    state.disable_hand_motion()
                state.hand_camera_to_action_basis = bases
                state.hand_camera_control_name = control_name
                state.hand_camera_control_revision += 1

    refresh_hand_camera_control_frame()
    state.camera_names = list(camera_sources)
    update_procedure_waypoint_marker(0, force=True)
    expert_controller = ExpertDemonstrationController(
        procedure_id=str(procedure.get("id", "")),
        guide_kind=guide_kind,
        action_dim=action_dim,
        arms=arms,
        has_grippers=has_grippers,
        waypoints=room_waypoints,
    )
    upstream_expert_handler = None
    upstream_expert_initial_state = None
    upstream_expert_actions: np.ndarray | None = None
    if _softmimicgen_task:
        from isaaclab.utils.datasets import HDF5DatasetFileHandler

        upstream_dataset = (
            _softmimicgen_root
            / "datasets/annotated_dataset/annotated_dataset_surgical_threading.hdf5"
        )
        if not upstream_dataset.is_file():
            raise RuntimeError(f"Pinned SoftMimicGen expert dataset is missing: {upstream_dataset}")
        upstream_expert_handler = HDF5DatasetFileHandler()
        upstream_expert_handler.open(str(upstream_dataset))
        upstream_episode_name = next(iter(upstream_expert_handler.get_episode_names()))
        upstream_episode = upstream_expert_handler.load_episode(
            upstream_episode_name,
            env.unwrapped.device,
        )
        upstream_expert_initial_state = upstream_episode.get_initial_state()
        recorded_deformable_state = upstream_expert_initial_state.get("deformable_object", {}).get("object", {})
        recorded_nodal_positions = recorded_deformable_state.get("nodal_position")
        live_nodal_positions = interactive_deformable.data.nodal_pos_w if interactive_deformable is not None else None
        if (
            recorded_nodal_positions is not None
            and live_nodal_positions is not None
            and recorded_nodal_positions.shape[-2] != live_nodal_positions.shape[-2]
        ):
            # Scaling Rope.usd radially changes PhysX's cooked FEM topology.
            # Restore the recorded robot and ring poses, while retaining the
            # new strand's own native reset state rather than forcing an
            # incompatible 549-node tensor into the 244-node physical body.
            native_scaled_state = scene.get_state(is_relative=True)
            upstream_expert_initial_state = dict(upstream_expert_initial_state)
            upstream_expert_initial_state["deformable_object"] = native_scaled_state["deformable_object"]
        native_scaled_state = scene.get_state(is_relative=True)
        recorded_articulations = dict(upstream_expert_initial_state.get("articulation", {}))
        for articulation_name, articulation_state in native_scaled_state.get("articulation", {}).items():
            recorded_articulations.setdefault(articulation_name, articulation_state)
        recorded_rigid_objects = dict(upstream_expert_initial_state.get("rigid_object", {}))
        for object_name, object_state in native_scaled_state.get("rigid_object", {}).items():
            recorded_rigid_objects.setdefault(object_name, object_state)
        upstream_expert_initial_state = dict(upstream_expert_initial_state)
        upstream_expert_initial_state["articulation"] = recorded_articulations
        upstream_expert_initial_state["rigid_object"] = recorded_rigid_objects
        upstream_actions_value = upstream_episode.data["actions"]
        upstream_expert_actions = (
            upstream_actions_value.detach().cpu().numpy().astype(np.float32)
            if isinstance(upstream_actions_value, torch.Tensor)
            else np.asarray(upstream_actions_value, dtype=np.float32)
        )
        if (
            bimanual_softmimicgen
            and upstream_expert_actions.ndim == 2
            and upstream_expert_actions.shape[1] == 7
            and action_dim == 14
        ):
            # Preserve NVIDIA's seven primary-arm values exactly. The added
            # receiving PSM remains stationary with its jaws open during the
            # upstream reference replay; clinician bimanual demonstrations
            # record the full fourteen-dimensional room action instead.
            bimanual_actions = np.zeros(
                (upstream_expert_actions.shape[0], action_dim), dtype=np.float32
            )
            bimanual_actions[:, :7] = upstream_expert_actions
            bimanual_actions[:, 13] = 1.0
            upstream_expert_actions = bimanual_actions
        if upstream_expert_actions.ndim != 2 or upstream_expert_actions.shape[1] != action_dim:
            raise RuntimeError("Pinned SoftMimicGen expert action shape does not match the live task")
    state.expert_demonstration = expert_controller.snapshot()
    state.runtime_provenance = runtime_provenance(state)
    state.camera_frame_ids = {name: 0 for name in camera_sources}
    state.camera_subscribers = {name: 0 for name in camera_sources}
    state.camera_poll_last_seen_by_name = {name: 0.0 for name in camera_sources}

    def active_logical_camera_name(now: float | None = None) -> str:
        """Resolve the one doctor-selected view backed by the shared sensor."""
        if not shared_camera_renderer:
            return "endoscope_left"
        now = time.monotonic() if now is None else now
        with state.lock:
            subscribed = [
                name
                for name, count in state.camera_subscribers.items()
                if count > 0
            ]
            if subscribed:
                return max(
                    subscribed,
                    key=lambda name: state.camera_poll_last_seen_by_name.get(
                        name, 0.0
                    ),
                )
            camera_name, last_seen = max(
                state.camera_poll_last_seen_by_name.items(),
                key=lambda item: item[1],
                default=("endoscope_left", 0.0),
            )
        return camera_name if now - last_seen < 1.5 else "endoscope_left"

    jpeg_encoder = BoundedJpegEncoder(state)
    state.procedure_waypoints_total = len(room_waypoints)
    state.procedure_started_at = time.monotonic()
    state.procedure_last_motion_at = state.procedure_started_at
    try:
        state.camera_intrinsics = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float).tolist()
        state.semantic_labels = camera_semantic_labels(camera)
    except (AttributeError, KeyError, TypeError, RuntimeError):
        pass
    initial_object_positions = {
        name: rigid_object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32).copy()
        for name, rigid_object in objects.items()
    }

    def native_tissue_centroid() -> np.ndarray | None:
        if native_tissue is None:
            return None
        nodal_value = native_tissue.data.nodal_pos_w
        nodal_positions = getattr(nodal_value, "torch", nodal_value)
        return nodal_positions[0].mean(dim=0).detach().cpu().numpy().astype(np.float32)

    initial_native_centroid: list[np.ndarray | None] = [native_tissue_centroid()]

    stapler_placement_longitudinal_m: dict[int, float] = {}

    def stapler_tissue_station_gap_mm(
        station_index: int,
    ) -> float | None:
        if (
            not stapler_closure_tissues
            or stapler_tissue_default_positions is None
            or not stapler_tissue_station_masks
        ):
            return None
        bounded_index = max(
            0,
            min(
                len(stapler_tissue_station_masks) - 1,
                int(station_index),
            ),
        )
        current_positions = torch.cat(
            [
                getattr(
                    tissue_flap.data.nodal_pos_w,
                    "torch",
                    tissue_flap.data.nodal_pos_w,
                )
                for tissue_flap in stapler_closure_tissues
            ],
            dim=1,
        )[0]
        default_positions = stapler_tissue_default_positions[0]
        station_mask = stapler_tissue_station_masks[bounded_index]
        closure_center_y = stapler_tissue_wound_center
        left_mask = station_mask & (
            default_positions[:, stapler_tissue_wound_axis]
            < closure_center_y
        )
        right_mask = station_mask & (
            default_positions[:, stapler_tissue_wound_axis]
            > closure_center_y
        )
        if not bool(left_mask.any()) or not bool(right_mask.any()):
            return None
        left_edge = torch.max(
            current_positions[left_mask, stapler_tissue_wound_axis]
        )
        right_edge = torch.min(
            current_positions[right_mask, stapler_tissue_wound_axis]
        )
        return max(
            0.0,
            float((right_edge - left_edge).item() * 1000.0),
        )

    def write_stapler_tissue_constraints(
        approximation_progress: float,
    ) -> None:
        nonlocal stapler_tissue_approximation_progress

        if (
            not stapler_closure_tissues
            or stapler_tissue_base_targets is None
            or stapler_tissue_default_positions is None
        ):
            return
        progress = max(
            0.0,
            min(1.0, float(approximation_progress)),
        )
        targets = stapler_tissue_base_targets.clone()
        default_positions = stapler_tissue_default_positions[0]
        for retained_index in sorted(
            stapler_closed_station_indices
        ):
            retained_mask = stapler_tissue_station_masks[
                retained_index
            ]
            targets[0, retained_mask, :3] = (
                stapler_tissue_station_closed_positions[
                    retained_index
                ][retained_mask]
            )
            targets[0, retained_mask, 3] = 0.0
        if (
            stapler_active_station_index
            not in stapler_closed_station_indices
            and progress > 0.0
        ):
            active_mask = stapler_tissue_station_masks[
                stapler_active_station_index
            ]
            closed_positions = (
                stapler_tissue_station_closed_positions[
                    stapler_active_station_index
                ]
            )
            targets[0, active_mask, :3] = (
                default_positions[active_mask]
                + progress
                * (
                    closed_positions[active_mask]
                    - default_positions[active_mask]
                )
            )
            targets[0, active_mask, 3] = 0.0
        for tissue_flap, node_slice in zip(
            stapler_closure_tissues,
            stapler_tissue_node_slices,
            strict=True,
        ):
            tissue_flap.write_nodal_kinematic_target_to_sim(
                targets[:, node_slice, :]
            )
            tissue_flap.write_data_to_sim()
        paddle_open_offset_m = 0.0062
        paddle_closed_offset_m = 0.0023
        paddle_offset_m = (
            paddle_open_offset_m
            + progress
            * (
                paddle_closed_offset_m
                - paddle_open_offset_m
            )
        )
        paddle_x_m = stapler_closure_targets_m[
            stapler_active_station_index
        ][0]
        for side, direction in (("left", -1.0), ("right", 1.0)):
            translate_op = (
                stapler_approximation_paddle_translate_ops.get(side)
            )
            if translate_op is not None:
                translate_op.Set(
                    Gf.Vec3d(
                        paddle_x_m,
                        direction * paddle_offset_m,
                        0.0590,
                    )
                )
        stapler_tissue_approximation_progress = progress

    def reset_stapler_tissue_physics() -> None:
        nonlocal stapler_tissue_approximation_progress
        nonlocal stapler_tissue_max_displacement_mm

        if (
            not stapler_closure_tissues
            or stapler_tissue_default_state is None
            or stapler_tissue_base_targets is None
        ):
            return
        for (
            tissue_flap,
            flap_default_state,
            node_slice,
        ) in zip(
            stapler_closure_tissues,
            stapler_tissue_default_states,
            stapler_tissue_node_slices,
            strict=True,
        ):
            tissue_flap.write_nodal_state_to_sim(
                flap_default_state
            )
            tissue_flap.write_nodal_kinematic_target_to_sim(
                stapler_tissue_base_targets[:, node_slice, :]
            )
            tissue_flap.write_data_to_sim()
        stapler_tissue_approximation_progress = 0.0
        stapler_tissue_max_displacement_mm = 0.0
        write_stapler_tissue_constraints(0.0)

    reset_stapler_tissue_physics()

    def stapler_closure_payload() -> dict[str, Any]:
        spacing_errors = spacing_errors_m(
            tuple(stapler_placement_longitudinal_m.values()),
            STAPLER_CLOSURE_STATION_SPACING_M,
        )
        station_count = len(STAPLER_CLOSURE_STATION_OFFSETS_M)
        closed_count = len(stapler_closed_station_indices)
        station_ready = (
            time.monotonic() >= stapler_station_settle_until
        )
        tissue_gap_mm = stapler_tissue_station_gap_mm(
            stapler_active_station_index
        )
        retained_mask = None
        if stapler_tissue_station_masks:
            retained_mask = torch.zeros_like(
                stapler_tissue_station_masks[0],
                dtype=torch.bool,
            )
            for retained_index in stapler_closed_station_indices:
                retained_mask |= stapler_tissue_station_masks[
                    retained_index
                ]
        retained_node_count = (
            int(retained_mask.sum().item())
            if retained_mask is not None
            else 0
        )
        current_station_retained = (
            stapler_active_station_index
            in stapler_closed_station_indices
        )
        retained_station_gaps_mm = {
            retained_index + 1: stapler_tissue_station_gap_mm(
                retained_index
            )
            for retained_index in sorted(
                stapler_closed_station_indices
            )
        }
        retained_gap_limit_mm = (
            STAPLER_TISSUE_TARGET_GAP_M * 1000.0 + 0.25
        )
        retained_verified_count = sum(
            gap_mm is not None
            and gap_mm <= retained_gap_limit_mm
            for gap_mm in retained_station_gaps_mm.values()
        )
        retention_state = (
            "retained_complete"
            if closed_count == station_count
            else "retained"
            if current_station_retained
            else "approximating"
            if stapler_tissue_approximation_progress > 0.0
            else "open"
        )
        return {
            "station_index": stapler_active_station_index + 1,
            "station_count": station_count,
            "station_spacing_mm": STAPLER_CLOSURE_STATION_SPACING_M
            * 1000.0,
            "station_state": (
                "placed"
                if stapler_active_station_index
                in stapler_closed_station_indices
                else "open"
                if station_ready
                else "indexing"
            ),
            "station_ready": station_ready,
            "closed_station_count": closed_count,
            "closed_station_indices": [
                index + 1
                for index in sorted(stapler_closed_station_indices)
            ],
            "closure_progress_percent": round(
                100.0 * closed_count / station_count,
                2,
            ),
            "closure_complete": closed_count == station_count,
            "approximation_progress_percent": round(
                stapler_tissue_approximation_progress * 100.0,
                2,
            ),
            "initial_tissue_gap_mm": (
                round(
                    stapler_tissue_initial_gaps_mm[
                        stapler_active_station_index
                    ],
                    4,
                )
                if stapler_tissue_initial_gaps_mm
                else None
            ),
            "tissue_gap_mm": (
                round(tissue_gap_mm, 4)
                if tissue_gap_mm is not None
                else None
            ),
            "target_tissue_gap_mm": (
                STAPLER_TISSUE_TARGET_GAP_M * 1000.0
            ),
            "detected_wound_axis": stapler_tissue_detected_wound_axis,
            "detected_source_gap_mm": round(
                stapler_tissue_detected_source_gap_mm,
                4,
            ),
            "tissue_max_displacement_mm": round(
                stapler_tissue_max_displacement_mm,
                4,
            ),
            "retained_attachment_count": closed_count,
            "retained_node_count": retained_node_count,
            "retained_verified_count": retained_verified_count,
            "retained_station_gaps_mm": {
                str(station): (
                    round(gap_mm, 4)
                    if gap_mm is not None
                    else None
                )
                for station, gap_mm
                in retained_station_gaps_mm.items()
            },
            "retention_state": retention_state,
            "retention_verified": bool(
                current_station_retained
                and tissue_gap_mm is not None
                and tissue_gap_mm
                <= STAPLER_TISSUE_TARGET_GAP_M * 1000.0 + 0.25
            ),
            "current_target_m": list(
                stapler_closure_targets_m[stapler_active_station_index]
            ),
            "max_spacing_error_mm": round(
                max(spacing_errors, default=0.0) * 1000.0,
                4,
            ),
            "last_placement": stapler_last_placement,
        }

    def move_stapler_to_station(station_index: int) -> None:
        nonlocal stapler_active_station_index
        nonlocal stapler_station_settle_until
        nonlocal stapler_fixture_position_w
        nonlocal stapler_fixture_quaternion_w

        if stapler_articulation is None:
            return
        bounded_index = max(
            0,
            min(
                len(STAPLER_CLOSURE_STATION_OFFSETS_M) - 1,
                int(station_index),
            ),
        )
        if bounded_index == stapler_active_station_index:
            return
        delta_x_m = (
            STAPLER_CLOSURE_STATION_OFFSETS_M[bounded_index]
            - STAPLER_CLOSURE_STATION_OFFSETS_M[
                stapler_active_station_index
            ]
        )
        root_pose = stapler_articulation.data.root_pose_w.clone()
        root_pose[:, 0] += delta_x_m
        stapler_articulation.write_root_pose_to_sim(root_pose)
        stapler_articulation.write_root_velocity_to_sim(
            torch.zeros_like(stapler_articulation.data.root_vel_w)
        )
        # Re-capture the ideal fixture datum after the explicit rail index.
        # This keeps indexing motion separate from measured housing drift.
        stapler_fixture_position_w = None
        stapler_fixture_quaternion_w = None
        stapler_active_station_index = bounded_index
        stapler_station_settle_until = time.monotonic() + 0.75

    def clear_stapler_closure() -> None:
        nonlocal stapler_last_placement

        if suture_stage.GetPrimAtPath(stapler_visual_root_path).IsValid():
            suture_stage.RemovePrim(stapler_visual_root_path)
        stapler_closed_station_indices.clear()
        stapler_placement_longitudinal_m.clear()
        stapler_last_placement = None
        move_stapler_to_station(0)
        reset_stapler_tissue_physics()

    def record_stapler_placement() -> dict[str, Any]:
        nonlocal stapler_last_placement

        station_index = stapler_active_station_index
        if station_index in stapler_closed_station_indices:
            return dict(stapler_last_placement or {})
        output_path = (
            f"{stapler_visual_root_path}/"
            f"Staple_{station_index + 1:02d}"
        )
        target_position_m = stapler_closure_targets_m[station_index]
        staple_prim = add_staple_reference(
            suture_stage,
            output_path,
            translation_m=target_position_m,
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        # The formed staple remains a kinematic rigid retainer with collision.
        # Its two FEM attachment bands are driven below through PhysX nodal
        # constraints. This models post-release retention without pretending
        # that the current backend simulates puncture or metal plasticity.
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(staple_prim)
        rigid_body.CreateRigidBodyEnabledAttr().Set(True)
        rigid_body.CreateKinematicEnabledAttr().Set(True)
        for descendant in Usd.PrimRange(staple_prim):
            if descendant.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(
                    descendant
                ).CreateCollisionEnabledAttr().Set(True)
        staple_prim.CreateAttribute(
            "drAnmar:retentionMode",
            Sdf.ValueTypeNames.String,
        ).Set("physx_fem_nodal_staple_leg_constraint")
        staple_prim.CreateAttribute(
            "drAnmar:rigidRetainer",
            Sdf.ValueTypeNames.Bool,
        ).Set(True)
        staple_prim.CreateAttribute(
            "drAnmar:clinicalValidation",
            Sdf.ValueTypeNames.Bool,
        ).Set(False)
        placement = assess_placement(
            stapler_closure_line,
            target_position_m,
            (0.0, 1.0, 0.0),
        )
        tissue_gap_at_deployment_mm = (
            stapler_tissue_station_gap_mm(station_index)
        )
        stapler_closed_station_indices.add(station_index)
        write_stapler_tissue_constraints(1.0)
        stapler_placement_longitudinal_m[
            station_index
        ] = placement.longitudinal_m
        stapler_last_placement = {
            "station_index": station_index + 1,
            "prim_path": output_path,
            "position_m": [
                round(float(component), 6)
                for component in target_position_m
            ],
            "lateral_error_mm": round(
                placement.lateral_error_m * 1000.0,
                4,
            ),
            "orientation_error_deg": round(
                placement.orientation_error_deg,
                4,
            ),
            "within_line_extent": placement.within_line_extent,
            "tissue_gap_mm_at_deployment": (
                round(tissue_gap_at_deployment_mm, 4)
                if tissue_gap_at_deployment_mm is not None
                else None
            ),
            "retained_node_count": int(
                stapler_tissue_station_masks[station_index]
                .sum()
                .item()
            ),
            "representation": (
                "formed_staple_rigid_fem_retainer"
            ),
        }
        return dict(stapler_last_placement)

    def reset_environment(selected_scenario: str, selected_seed: int) -> None:
        nonlocal stapler_cycle_started_at
        nonlocal stapler_cycle_threshold_at
        nonlocal stapler_cycle_release_started_at
        nonlocal stapler_cycle_count
        nonlocal stapler_last_event
        nonlocal stapler_partial_candidate
        nonlocal stapler_partial_start_deployments
        nonlocal stapler_partial_peak_deg
        nonlocal stapler_partial_stroke_attempts
        nonlocal stapler_partial_stroke_passes
        nonlocal stapler_active_station_index
        nonlocal stapler_station_settle_until
        nonlocal stapler_pending_advance
        nonlocal stapler_last_placement
        nonlocal stapler_fixture_position_w
        nonlocal stapler_fixture_quaternion_w
        nonlocal closure_cycle_running
        nonlocal closure_cycle_phase
        nonlocal closure_phase_started_at
        nonlocal closure_hold_targets
        nonlocal closure_staple_deployed
        nonlocal closure_bead_deposited
        nonlocal closure_robot_arm_hold_targets
        nonlocal closure_robot_mount_reference_w
        nonlocal closure_robot_max_mount_error_mm
        nonlocal rescue_physics_step
        nonlocal rescue_simulation_time_s
        native_grasp_arms.clear()
        update_procedure_waypoint_marker(0, force=True)
        np.random.seed(selected_seed)
        torch.manual_seed(selected_seed)
        env.reset(seed=selected_seed)
        latest_dynamic_patient_telemetry.clear()
        latest_autonomous_rescue_telemetry.clear()
        if dynamic_abdominal_patient_enabled:
            if dynamic_patient_runtime is None:
                raise RuntimeError(
                    "Dynamic patient physiology runtime is unavailable"
                )
            dynamic_patient_runtime.reset()
            patient_prim = suture_stage.GetPrimAtPath(
                "/World/envs/env_0/DynamicAbdominalPatient"
            )
            if not patient_prim.IsValid():
                raise RuntimeError(
                    "Dynamic patient reset failed closed: patient prim is missing"
                )
            initial_access_state = str(
                procedure.get("dynamic_patient_access_state", "intact")
            )
            access_variant = patient_prim.GetVariantSets().GetVariantSet(
                "access_state"
            )
            if (
                not access_variant.IsValid()
                or not access_variant.SetVariantSelection(
                    initial_access_state
                )
            ):
                raise RuntimeError(
                    "Dynamic patient reset failed closed: "
                    f"{initial_access_state}"
                )
        if autonomous_rescue_or_enabled:
            if autonomous_rescue_runtime is None:
                raise RuntimeError(
                    "Autonomous Rescue OR effects runtime is unavailable"
                )
            if autonomous_rescue_patient_runtime is None:
                raise RuntimeError(
                    "Autonomous Rescue OR shared patient runtime is unavailable"
                )
            autonomous_rescue_patient_runtime.reset()
            autonomous_rescue_runtime.reset(seed=selected_seed)
            rescue_physics_step = -1
            rescue_simulation_time_s = 0.0
            rescue_previous_tool_positions.clear()
            vessel_prim = suture_stage.GetPrimAtPath(
                "/World/envs/env_0/AutonomousRescueVessel"
            )
            if not vessel_prim.IsValid():
                raise RuntimeError(
                    "Autonomous Rescue OR reset failed: vessel prim is missing"
                )
        if stapler_test_cell_enabled:
            if suture_stage.GetPrimAtPath(
                stapler_visual_root_path
            ).IsValid():
                suture_stage.RemovePrim(stapler_visual_root_path)
            stapler_deployment_controller.reset(reset_magazine=True)
            stapler_cycle_started_at = None
            stapler_cycle_threshold_at = None
            stapler_cycle_release_started_at = None
            stapler_cycle_count = 0
            stapler_last_event = None
            stapler_partial_candidate = False
            stapler_partial_start_deployments = 0
            stapler_partial_peak_deg = 0.0
            stapler_partial_stroke_attempts = 0
            stapler_partial_stroke_passes = 0
            stapler_active_station_index = 0
            stapler_station_settle_until = 0.0
            stapler_closed_station_indices.clear()
            stapler_placement_longitudinal_m.clear()
            stapler_pending_advance = False
            stapler_last_placement = None
            stapler_fixture_position_w = None
            stapler_fixture_quaternion_w = None
            reset_stapler_tissue_physics()
        if (
            closure_robot_enabled
            and closure_robot_controller is not None
            and closure_robot_articulation is not None
        ):
            closure_robot_controller.reset()
            closure_robot_arm_hold_targets = (
                closure_robot_articulation.data.joint_pos[
                    :,
                    closure_robot_arm_joint_indices,
                ]
                .detach()
                .clone()
            )
            closure_robot_mount_reference_w = (
                closure_robot_articulation.data.body_pos_w[
                    0,
                    closure_robot_body_indices["panda_link8"],
                ]
                .detach()
                .clone()
            )
            closure_robot_max_mount_error_mm = 0.0
            closure_cycle_running = False
            closure_cycle_phase = ClosurePhase.READY
            closure_phase_started_at = time.monotonic()
            closure_hold_targets = closure_phase_targets(
                ClosurePhase.READY
            )
            closure_staple_deployed = False
            closure_bead_deposited = False
            set_closure_robot_joint_targets(
                closure_robot_articulation,
                closure_hold_targets,
            )
        dr_anmar_needle_domain: dict[str, Any] = {}
        if dr_anmar_parametric_needle_enabled:
            dr_anmar_needle_domain = (
                apply_dr_anmar_needle_episode_domain(
                    suture_stage,
                    seed=selected_seed,
                    root_path=suture_root_path,
                )
            )
        if dr_anmar_needle_enabled:
            (
                _suture_runtime_profile,
                suture_runtime_domain_state[0],
            ) = sample_suture_runtime_profile(
                suture_profile,
                selected_seed,
            )
            suture_last_sample_time[0] = time.monotonic()
        if native_episode_domain:
            native_episode_domain["requested_reset_seed"] = selected_seed
            native_episode_domain["requires_scene_rebuild_for_new_material_domain"] = (
                selected_seed != native_episode_domain["setup_seed"]
            )
        initialize_native_attachment()
        write_native_attachment()
        apply_native_object_scenario(objects, selected_scenario, selected_seed)
        initial_object_positions.clear()
        initial_object_positions.update(
            {
                name: rigid_object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32).copy()
                for name, rigid_object in objects.items()
            }
        )
        initial_native_centroid[0] = native_tissue_centroid()
        profile = SCENARIO_NATIVE_PROFILES.get(selected_scenario, {})
        show_multi_organ = bool(profile.get("show_multi_organ"))
        enabled_colliders = 0
        for child in showcase_children:
            imageable = UsdGeom.Imageable(child)
            visible = show_multi_organ or child.GetName() in default_showcase_names
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
            child_mesh = stage.GetPrimAtPath(f"{showcase_path}/{child.GetName()}/{child.GetName()}")
            if child_mesh.IsValid():
                UsdPhysics.CollisionAPI.Apply(child_mesh).CreateCollisionEnabledAttr().Set(visible)
                enabled_colliders += int(visible)
        refresh_anatomy_guard_volumes()
        with state.lock:
            state.anatomy_showcase = (
                "Multi-organ operative field"
                if show_multi_organ
                else str(procedure.get("anatomy_focus") or "Operative field")
            )
            state.procedure_waypoints_completed = 0
            state.procedure_motion_seen = False
            state.procedure_grasp_seen = False
            state.procedure_object_lift_m = 0.0
            state.procedure_object_motion_m = 0.0
            state.procedure_started_at = time.monotonic()
            state.procedure_last_motion_at = time.monotonic()
            state.anatomy_collision_meshes = enabled_colliders
            state.native_grasp_contact_active = [False] * state.arms
            state.gripper_apertures = [1.0] * state.arms
            state.grippers_open = [True] * state.arms
            state.disable_hand_motion()
            state.tool_to_object_distance_m = [None] * state.arms
            state.tool_to_object_offset_m = [None] * state.arms
            state.virtual_fixture_active = False
            state.closest_anatomy_clearance_m = None
            state.needle_tip_clearance_m = None
            state.needle_surface_outward = None
            state.needle_surface_direction = None
            state.needle_entry_direction = None
            state.adaptive_precision_active = False
            state.native_telemetry = {}
            if dynamic_abdominal_patient_enabled:
                state.dynamic_patient_access_state = initial_access_state
                state.dynamic_patient_cut_events = 0
            if skin_adhesive_enabled:
                state.skin_adhesive_target = 0.0
                state.skin_adhesive_system.update(
                    {
                        "workflow_state": "mounted_ready",
                        "mounted_arm": skin_adhesive_mounted_arm + 1,
                        "target_activation": 0.0,
                        "actual_activation": 0.0,
                    }
                )
            if stapler_test_cell_enabled:
                state.stapler_command_request = None
                state.stapler_station_request = None
                state.stapler_manual_target_deg = 0.0
                state.stapler_test_cell.update(
                    {
                        "cycle_phase": "ready",
                        "cycle_running": False,
                        "target_trigger_deg": 0.0,
                        "actual_trigger_deg": 0.0,
                        "pusher_travel_mm": 0.0,
                        "max_pusher_travel_mm": 0.0,
                        "joint_limit_violation_deg": 0.0,
                        "fixture_translation_error_mm": 0.0,
                        "fixture_rotation_error_deg": 0.0,
                        "max_fixture_translation_error_mm": 0.0,
                        "max_fixture_rotation_error_deg": 0.0,
                        "magazine_remaining": stapler_magazine.remaining,
                        "deployment_count": 0,
                        "cycle_count": 0,
                        "partial_stroke_attempts": 0,
                        "partial_stroke_passes": 0,
                        "max_trigger_deg": 0.0,
                        **stapler_closure_payload(),
                        "last_event": None,
                    }
                )
            if closure_robot_enabled:
                state.closure_robot_command_request = None
                state.closure_robot_system.update(
                    {
                        "phase": "ready",
                        "cycle_running": False,
                        "cycle_complete": False,
                        "temporary_capture_attachment_count": 0,
                        "formed_staple_count": 0,
                        "staple_attachment_count": 0,
                        "adhesive_bead_count": 0,
                        "adhesive_bond_attachment_count": 0,
                        "attachment_prim_count": 0,
                        "attachment_enabled_count": 0,
                        "attachment_actor_pair_count": 0,
                        "attachment_auto_overlap_count": 0,
                        "attachment_explicit_point_count": 0,
                    }
                )
            state.dr_anmar_needle_domain = dr_anmar_needle_domain
            state.upstream_task_success = False if _softmimicgen_task else None
        selected_active_camera = active_logical_camera_name()
        with state.lock:
            selected_view_mode = state.camera_view_mode
            selected_camera_adjustment = state.camera_adjustment(
                selected_active_camera
            )
            selected_wrist_camera_adjustments = {
                name: state.camera_adjustment(name)
                for name in state.camera_names
                if name.startswith("wrist_")
            }
        apply_endoscope_camera_view(
            selected_scenario,
            selected_view_mode,
            selected_camera_adjustment,
            selected_active_camera,
        )
        update_wrist_camera_poses(
            selected_wrist_camera_adjustments,
            selected_active_camera,
        )
    task_slug = args_cli.task.lower().replace("isaac-", "").replace("-v0", "").replace("-", "_")
    existing = sorted(args_cli.demo_dir.glob(f"dr_anmar_{task_slug}_*.npz"), reverse=True)
    if existing:
        state.last_demo = existing[0].name

    server = uvicorn.Server(
        uvicorn.Config(build_web_app(state), host=args_cli.host, port=args_cli.port, log_level="info", access_log=False)
    )
    server_thread = threading.Thread(target=server.run, name="dr-anmar-web", daemon=True)
    server_thread.start()

    stop_event = threading.Event()

    def request_shutdown(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    capture_spool: BoundedCaptureSpool | None = None
    demo_started_at = ""
    demo_started_monotonic = 0.0
    last_vision_sample_time = 0.0
    last_safety_sample_time = 0.0
    suture_last_sample_time = [time.monotonic()]
    latest_contact_forces: dict[str, float] = {}
    latest_deformable_safety: dict[str, float] = {}
    latest_suture_telemetry: dict[str, Any] = {}
    latest_dynamic_patient_telemetry: dict[str, Any] = {}
    latest_autonomous_rescue_telemetry: dict[str, Any] = {}
    replay_actions: np.ndarray | None = None
    upstream_expert_active = False
    upstream_expert_index = 0
    replay_index = 0
    # Observational state only. Membership is derived from native jaw contact;
    # it never authors a joint, teleports an object, or changes collisions.
    native_grasp_arms: set[int] = set()
    last_loop_time = time.monotonic()
    last_fps_time = last_loop_time
    fps_steps = 0
    last_frame_time = 0.0
    frame_count = 0
    environment_reward = 0.0
    environment_terminated = False
    environment_truncated = False
    environment_success = -1.0
    print(f"[DR_ANMAR_WORKSTATION] http://{args_cli.host}:{args_cli.port}", flush=True)

    while simulation_app.is_running() and not stop_event.is_set():
        loop_started = time.monotonic()
        refresh_hand_camera_control_frame()
        action_uses_upstream_softmimicgen_units = False
        selected_active_camera = active_logical_camera_name(loop_started)
        with state.lock:
            reset_requested = state.reset_requested
            state.reset_requested = False
            record_request = state.record_request
            state.record_request = None
            replay_request = state.replay_request
            state.replay_request = None
            expert_request = state.expert_request
            state.expert_request = None
            stapler_command_request = state.stapler_command_request
            state.stapler_command_request = None
            stapler_station_request = state.stapler_station_request
            state.stapler_station_request = None
            stapler_manual_target_deg = state.stapler_manual_target_deg
            closure_robot_command_request = (
                state.closure_robot_command_request
            )
            state.closure_robot_command_request = None
            scenario_id = state.scenario_id
            scenario_seed = state.scenario_seed
            camera_view_request = state.camera_view_request
            state.camera_view_request = None
            selected_camera_view_mode = state.camera_view_mode
            camera_adjustment = state.camera_adjustment(
                selected_active_camera
            )
            wrist_camera_adjustments = {
                name: state.camera_adjustment(name)
                for name in state.camera_names
                if name.startswith("wrist_")
            }
            ghost_update = state.reference_ghost_update
            state.reference_ghost_update = None
            ghost_enabled = state.reference_ghost_enabled
            if state.drive_min_steps_remaining > 0 and bool(np.any(state.drive)):
                manual_action = state.drive.copy()
                state.drive_min_steps_remaining -= 1
                if state.drive_min_steps_remaining == 0 and state.drive_stop_pending:
                    state.drive.fill(0.0)
                    state.drive_until = 0.0
                    state.drive_stop_pending = False
            elif state.drive_until > loop_started:
                manual_action = state.drive.copy()
            elif state.pulse_steps > 0:
                manual_action = state.pulse.copy()
                state.pulse_steps -= 1
            else:
                manual_action = np.zeros(state.action_dim, dtype=np.float32)
            if state.native_ik_scales:
                hand_commands = state.hand_teleop.consume(
                    state.native_ik_scales,
                    now=loop_started,
                )
                for arm, hand_command in enumerate(hand_commands):
                    if state.hand_teleop.arm_states[arm].motion_engaged:
                        manual_action[state.body_action_slice(arm)] = np.asarray(
                            hand_command,
                            dtype=np.float32,
                        )
                if (
                    state.hand_teleop.enabled
                    and any(
                        arm_state.motion_engaged
                        for arm_state in state.hand_teleop.arm_states
                    )
                ):
                    state.hand_last_applied_sequence = state.hand_teleop.last_sequence
                    state.hand_last_applied_at = loop_started
            grippers_open = list(state.grippers_open)
            gripper_apertures = list(state.gripper_apertures)
            virtual_fixture_enabled = state.virtual_fixture_enabled

        if ghost_update is not None:
            if ghost_enabled and ghost_update != "__hide__":
                try:
                    path_points, path_phases = reference_tool_path(args_cli.demo_dir / Path(ghost_update).name)
                    ghost_markers.visualize(translations=path_points, marker_indices=path_phases)
                    ghost_markers.set_visibility(True)
                    with state.lock:
                        state.reference_ghost_points = len(path_points)
                except (OSError, KeyError, ValueError):
                    ghost_markers.set_visibility(False)
                    with state.lock:
                        state.reference_ghost_enabled = False
                        state.reference_ghost_points = 0
                        state.coaching_cue = "The clinician reference has no usable world-space tool path."
            else:
                ghost_markers.set_visibility(False)
                with state.lock:
                    state.reference_ghost_points = 0

        if reset_requested:
            with torch.inference_mode():
                reset_environment(scenario_id, scenario_seed)

        stapler_target_deg = 0.0
        stapler_cycle_phase = "disabled"
        stapler_tissue_progress_target = 0.0
        if stapler_test_cell_enabled and stapler_articulation is not None:
            if (
                stapler_station_request is not None
                and stapler_cycle_started_at is None
            ):
                move_stapler_to_station(stapler_station_request)
            if stapler_command_request == "reset":
                clear_stapler_closure()
                stapler_deployment_controller.reset(reset_magazine=True)
                stapler_cycle_started_at = None
                stapler_cycle_threshold_at = None
                stapler_cycle_release_started_at = None
                stapler_cycle_count = 0
                stapler_last_event = None
                stapler_partial_candidate = False
                stapler_partial_start_deployments = 0
                stapler_partial_peak_deg = 0.0
                stapler_partial_stroke_attempts = 0
                stapler_partial_stroke_passes = 0
                stapler_manual_target_deg = 0.0
                stapler_pending_advance = False
                with state.lock:
                    state.stapler_manual_target_deg = 0.0
                    state.stapler_test_cell["max_trigger_deg"] = 0.0
                    state.stapler_test_cell["max_pusher_travel_mm"] = 0.0
                    state.stapler_test_cell[
                        "max_fixture_translation_error_mm"
                    ] = 0.0
                    state.stapler_test_cell[
                        "max_fixture_rotation_error_deg"
                    ] = 0.0
            elif stapler_command_request == "fire":
                if stapler_cycle_started_at is None:
                    stapler_cycle_started_at = loop_started
                    stapler_cycle_threshold_at = None
                    stapler_cycle_release_started_at = None
            elif stapler_command_request == "release":
                stapler_cycle_started_at = None
                stapler_cycle_threshold_at = None
                stapler_cycle_release_started_at = None
                stapler_manual_target_deg = 0.0
            elif stapler_command_request == "manual":
                stapler_cycle_started_at = None
                stapler_cycle_threshold_at = None
                stapler_cycle_release_started_at = None
                stapler_partial_candidate = bool(
                    REARM_THRESHOLD_DEG
                    < stapler_manual_target_deg
                    < FIRE_THRESHOLD_DEG
                )
                stapler_partial_start_deployments = (
                    stapler_magazine.deployed
                )
                stapler_partial_peak_deg = 0.0

            if stapler_cycle_started_at is not None:
                cycle_elapsed_s = max(
                    0.0,
                    loop_started - stapler_cycle_started_at,
                )
                measured_trigger_deg = float(
                    np.degrees(
                        stapler_articulation.data.joint_pos[
                            0,
                            stapler_trigger_joint_index,
                        ].item()
                    )
                )
                approximation_fraction = min(
                    1.0,
                    cycle_elapsed_s
                    / STAPLER_TISSUE_APPROXIMATION_DURATION_S,
                )
                stapler_tissue_progress_target = (
                    approximation_fraction
                    * approximation_fraction
                    * (3.0 - 2.0 * approximation_fraction)
                )
                press_elapsed_s = max(
                    0.0,
                    cycle_elapsed_s
                    - STAPLER_TISSUE_APPROXIMATION_DURATION_S,
                )
                if approximation_fraction < 1.0:
                    stapler_cycle_phase = "approximate_tissue"
                    stapler_target_deg = 0.0
                elif stapler_cycle_threshold_at is None:
                    stapler_target_deg = min(
                        TRIGGER_LIMIT_DEG,
                        TRIGGER_LIMIT_DEG * press_elapsed_s,
                    )
                    stapler_cycle_phase = (
                        "press"
                        if stapler_target_deg < TRIGGER_LIMIT_DEG
                        else "await_threshold"
                    )
                    if (
                        stapler_target_deg >= FIRE_THRESHOLD_DEG
                        and measured_trigger_deg >= FIRE_THRESHOLD_DEG
                    ):
                        stapler_cycle_threshold_at = loop_started
                        stapler_cycle_phase = "hold"
                        stapler_target_deg = TRIGGER_LIMIT_DEG
                elif stapler_cycle_release_started_at is None:
                    stapler_cycle_phase = "hold"
                    stapler_target_deg = TRIGGER_LIMIT_DEG
                    if (
                        loop_started - stapler_cycle_threshold_at
                        >= 0.50
                    ):
                        stapler_cycle_release_started_at = loop_started
                else:
                    stapler_cycle_phase = "release"
                    release_elapsed_s = (
                        loop_started - stapler_cycle_release_started_at
                    )
                    stapler_target_deg = TRIGGER_LIMIT_DEG * max(
                        0.0,
                        1.0 - release_elapsed_s,
                    )
                    if (
                        stapler_target_deg <= 0.0
                        and measured_trigger_deg <= REARM_THRESHOLD_DEG
                    ):
                        stapler_cycle_phase = "complete"
                        stapler_cycle_started_at = None
                        stapler_cycle_threshold_at = None
                        stapler_cycle_release_started_at = None
                        stapler_cycle_count += 1
                        stapler_tissue_progress_target = 0.0
                        if stapler_pending_advance:
                            open_station_indices = [
                                index
                                for index in range(
                                    len(
                                        STAPLER_CLOSURE_STATION_OFFSETS_M
                                    )
                                )
                                if index
                                not in stapler_closed_station_indices
                                and index > stapler_active_station_index
                            ]
                            if open_station_indices:
                                move_stapler_to_station(
                                    open_station_indices[0]
                                )
                            stapler_pending_advance = False
                        with state.lock:
                            state.stapler_manual_target_deg = 0.0
            else:
                stapler_cycle_phase = (
                    "ready"
                    if stapler_manual_target_deg <= REARM_THRESHOLD_DEG
                    else "manual"
                )
                stapler_target_deg = stapler_manual_target_deg
                stapler_tissue_progress_target = 0.0

            write_stapler_tissue_constraints(
                stapler_tissue_progress_target
            )

            stapler_targets = synchronized_joint_targets_deg(
                stapler_target_deg
            )
            stapler_device = stapler_articulation.data.joint_pos.device
            stapler_articulation.set_joint_position_target(
                torch.tensor(
                    [[stapler_targets["trigger_joint"]]],
                    dtype=torch.float32,
                    device=stapler_device,
                ),
                joint_ids=[stapler_trigger_joint_index],
            )
            stapler_articulation.set_joint_position_target(
                torch.tensor(
                    [[stapler_targets["pusher_joint"]]],
                    dtype=torch.float32,
                    device=stapler_device,
                ),
                joint_ids=[stapler_pusher_joint_index],
            )

        if (
            closure_robot_enabled
            and closure_robot_controller is not None
            and closure_robot_articulation is not None
            and closure_cycle_phase is not None
        ):
            try:
                closure_positions = closure_robot_articulation.data.joint_pos[
                    0
                ]

                def closure_position(name: str) -> float:
                    return float(
                        closure_positions[
                            closure_robot_joint_indices[name]
                        ].item()
                    )

                release_clamps_open = False
                if closure_robot_command_request == "reset":
                    closure_robot_controller.reset()
                    closure_cycle_running = False
                    closure_cycle_phase = ClosurePhase.READY
                    closure_phase_started_at = loop_started
                    closure_staple_deployed = False
                    closure_bead_deposited = False
                    closure_hold_targets = closure_phase_targets(
                        ClosurePhase.READY
                    )
                elif closure_robot_command_request == "run":
                    closure_robot_controller.reset()
                    closure_cycle_running = True
                    closure_cycle_phase = ClosurePhase.CAPTURE
                    closure_robot_controller.phase = (
                        ClosurePhase.CAPTURE
                    )
                    closure_phase_started_at = loop_started
                    closure_staple_deployed = False
                    closure_bead_deposited = False
                    closure_hold_targets = closure_phase_targets(
                        ClosurePhase.CAPTURE
                    )
                elif closure_robot_command_request == "stop":
                    closure_cycle_running = False
                    closure_hold_targets = {
                        name: closure_position(name)
                        for name in closure_robot_joint_indices
                    }

                if closure_cycle_running:
                    closure_hold_targets = closure_phase_targets(
                        closure_cycle_phase
                    )
                    phase_elapsed_s = max(
                        0.0,
                        loop_started - closure_phase_started_at,
                    )
                    left_clamp_rad = closure_position(
                        "left_clamp_joint"
                    )
                    right_clamp_rad = closure_position(
                        "right_clamp_joint"
                    )
                    left_approximation_m = closure_position(
                        "left_approximation_joint"
                    )
                    right_approximation_m = closure_position(
                        "right_approximation_joint"
                    )
                    driver_m = closure_position(
                        "staple_driver_joint"
                    )
                    adhesive_deploy_m = closure_position(
                        "adhesive_deploy_joint"
                    )
                    adhesive_meter_m = closure_position(
                        "adhesive_meter_joint"
                    )
                    release_clamps_open = bool(
                        left_clamp_rad >= math.radians(25.0)
                        and right_clamp_rad <= math.radians(-25.0)
                    )
                    if (
                        closure_cycle_phase is ClosurePhase.RELEASE
                        and not release_clamps_open
                    ):
                        # Remove the temporary tissue capture first, then open
                        # both clamps while the approximation carriages remain
                        # centered. Retracting under a still-closed clamp can
                        # wedge real deformable tissue between the upper clamp
                        # and lower shoe.
                        closure_hold_targets[
                            "left_approximation_joint"
                        ] = 0.022
                        closure_hold_targets[
                            "right_approximation_joint"
                        ] = -0.022

                    if (
                        closure_cycle_phase is ClosurePhase.CAPTURE
                        and phase_elapsed_s >= 0.20
                        and abs(left_clamp_rad) <= math.radians(2.0)
                        and abs(right_clamp_rad) <= math.radians(2.0)
                    ):
                        closure_robot_controller.capture()
                        closure_cycle_phase = ClosurePhase.APPROXIMATE
                        closure_robot_controller.phase = (
                            ClosurePhase.APPROXIMATE
                        )
                        closure_phase_started_at = loop_started
                    elif (
                        closure_cycle_phase
                        is ClosurePhase.APPROXIMATE
                        and left_approximation_m >= 0.020
                        and right_approximation_m <= -0.020
                    ):
                        closure_cycle_phase = ClosurePhase.STAPLE
                        closure_robot_controller.phase = (
                            ClosurePhase.STAPLE
                        )
                        closure_phase_started_at = loop_started
                    elif (
                        closure_cycle_phase is ClosurePhase.STAPLE
                        and driver_m >= 0.012
                        and not closure_staple_deployed
                    ):
                        suture_stage.DefinePrim(
                            f"{closure_tissue_root_path}/Deployments",
                            "Xform",
                        )
                        closure_robot_controller.staple_retention.deploy(
                            prim_path=(
                                f"{closure_tissue_root_path}/Deployments/"
                                "FormedStaple_01"
                            ),
                            left_tissue_path=closure_left_tissue_path,
                            right_tissue_path=closure_right_tissue_path,
                            translation_m=(0.0, 0.0, 0.0034),
                            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                        )
                        closure_staple_deployed = True
                        closure_robot_controller.release_capture()
                        closure_cycle_phase = ClosurePhase.RELEASE
                        closure_phase_started_at = loop_started
                    elif (
                        closure_cycle_phase is ClosurePhase.RELEASE
                        and release_clamps_open
                        and left_approximation_m <= 0.003
                        and right_approximation_m >= -0.003
                    ):
                        closure_cycle_phase = ClosurePhase.ADHESIVE
                        closure_robot_controller.phase = (
                            ClosurePhase.ADHESIVE
                        )
                        closure_phase_started_at = loop_started
                    elif (
                        closure_cycle_phase is ClosurePhase.ADHESIVE
                        and adhesive_deploy_m <= -0.025
                        and adhesive_meter_m >= 0.008
                        and not closure_bead_deposited
                    ):
                        closure_robot_controller.adhesive_bonds.deposit(
                            prim_path=(
                                f"{closure_tissue_root_path}/Deployments/"
                                "AdhesiveBead_01"
                            ),
                            translation_m=(0.0, 0.0, 0.0022),
                            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                        )
                        closure_bead_deposited = True
                        closure_cycle_phase = (
                            ClosurePhase.CURE_LEADING
                        )
                        closure_robot_controller.phase = (
                            ClosurePhase.CURE_LEADING
                        )
                        closure_phase_started_at = loop_started
                    elif (
                        closure_cycle_phase
                        is ClosurePhase.CURE_LEADING
                        and phase_elapsed_s >= 0.75
                    ):
                        closure_robot_controller.adhesive_bonds.set_cure_fraction(
                            0,
                            0.5,
                        )
                        closure_cycle_phase = (
                            ClosurePhase.CURE_TRAILING
                        )
                        closure_robot_controller.phase = (
                            ClosurePhase.CURE_TRAILING
                        )
                        closure_phase_started_at = loop_started
                    elif (
                        closure_cycle_phase
                        is ClosurePhase.CURE_TRAILING
                        and phase_elapsed_s >= 0.75
                    ):
                        closure_robot_controller.adhesive_bonds.set_cure_fraction(
                            0,
                            1.0,
                        )
                        closure_cycle_phase = ClosurePhase.COMPLETE
                        closure_robot_controller.phase = (
                            ClosurePhase.COMPLETE
                        )
                        closure_phase_started_at = loop_started
                        closure_cycle_running = False
                        closure_hold_targets = closure_phase_targets(
                            ClosurePhase.COMPLETE
                        )
                        with state.lock:
                            state.procedure_event_code = (
                                PROCEDURE_EVENTS["task_complete"]
                            )
                            state.procedure_event_sequence += 1
                            state.coaching_cue = (
                                "Physical closure complete: the dynamic "
                                "formed staple retains two tissue attachments "
                                "and the cured bead retains six."
                            )

                set_closure_robot_joint_targets(
                    closure_robot_articulation,
                    closure_hold_targets,
                )
                if (
                    closure_robot_arm_hold_targets is None
                    or closure_robot_mount_reference_w is None
                ):
                    raise RuntimeError(
                        "The Franka closure arm has no initialized physical "
                        "joint-space hold"
                    )
                closure_robot_articulation.set_joint_position_target(
                    closure_robot_arm_hold_targets,
                    joint_ids=closure_robot_arm_joint_indices,
                )
                closure_snapshot = closure_robot_controller.snapshot()
                closure_body_positions = (
                    closure_robot_articulation.data.body_pos_w[0]
                    .detach()
                    .cpu()
                )
                mount_position_w = closure_robot_articulation.data.body_pos_w[
                    0,
                    closure_robot_body_indices["panda_link8"],
                ]
                mount_translation_error_mm = float(
                    torch.linalg.vector_norm(
                        mount_position_w - closure_robot_mount_reference_w
                    )
                    .detach()
                    .cpu()
                    .item()
                    * 1000.0
                )
                closure_robot_max_mount_error_mm = max(
                    closure_robot_max_mount_error_mm,
                    mount_translation_error_mm,
                )
                closure_body_positions_m = {
                    name: [
                        round(float(component), 6)
                        for component in closure_body_positions[index].tolist()
                    ]
                    for name, index in closure_robot_body_indices.items()
                }
                with state.lock:
                    state.closure_robot_system.update(
                        {
                            **closure_snapshot,
                            "body_positions_m": closure_body_positions_m,
                            "mount_translation_error_mm": round(
                                mount_translation_error_mm,
                                4,
                            ),
                            "max_mount_translation_error_mm": round(
                                closure_robot_max_mount_error_mm,
                                4,
                            ),
                            "phase": (
                                closure_cycle_phase.value
                                if closure_cycle_phase is not None
                                else "disabled"
                            ),
                            "cycle_running": closure_cycle_running,
                            "cycle_complete": (
                                closure_cycle_phase
                                is ClosurePhase.COMPLETE
                            ),
                            "held": bool(
                                not closure_cycle_running
                                and closure_cycle_phase
                                not in {
                                    ClosurePhase.READY,
                                    ClosurePhase.COMPLETE,
                                }
                            ),
                            "release_stage": (
                                "open_clamps"
                                if closure_cycle_phase
                                is ClosurePhase.RELEASE
                                and not release_clamps_open
                                else "retract_carriages"
                                if closure_cycle_phase
                                is ClosurePhase.RELEASE
                                else None
                            ),
                            "left_approximation_mm": round(
                                closure_position(
                                    "left_approximation_joint"
                                )
                                * 1000.0,
                                4,
                            ),
                            "right_approximation_mm": round(
                                closure_position(
                                    "right_approximation_joint"
                                )
                                * 1000.0,
                                4,
                            ),
                            "left_clamp_deg": round(
                                math.degrees(
                                    closure_position(
                                        "left_clamp_joint"
                                    )
                                ),
                                4,
                            ),
                            "right_clamp_deg": round(
                                math.degrees(
                                    closure_position(
                                        "right_clamp_joint"
                                    )
                                ),
                                4,
                            ),
                            "staple_driver_mm": round(
                                closure_position(
                                    "staple_driver_joint"
                                )
                                * 1000.0,
                                4,
                            ),
                            "adhesive_deploy_mm": round(
                                closure_position(
                                    "adhesive_deploy_joint"
                                )
                                * 1000.0,
                                4,
                            ),
                            "adhesive_meter_mm": round(
                                closure_position(
                                    "adhesive_meter_joint"
                                )
                                * 1000.0,
                                4,
                            ),
                            "last_error": None,
                        }
                    )
            except (
                AttributeError,
                IndexError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                closure_cycle_running = False
                with state.lock:
                    state.closure_robot_system.update(
                        {
                            "phase": "safety_hold",
                            "cycle_running": False,
                            "cycle_complete": False,
                            "last_error": str(exc),
                        }
                    )
                    state.coaching_cue = (
                        "Closure robot safety hold: " + str(exc)
                    )

        if expert_request == "start":
            expert_controller.start()
            if upstream_expert_actions is not None and upstream_expert_initial_state is not None:
                with torch.inference_mode():
                    env.unwrapped.reset_to(
                        upstream_expert_initial_state,
                        torch.tensor([0], device=env.unwrapped.device),
                        is_relative=True,
                    )
                upstream_expert_index = 0
                upstream_expert_active = True
            with state.lock:
                state.upstream_task_success = False if _softmimicgen_task else None
                state.expert_demonstration = expert_controller.snapshot(state.expert_reference_demo)
                state.procedure_phase = "rest"
                state.operator_input_source = "automation_policy"
                state.autonomy_mode = "expert_demonstration"
                state.coaching_cue = EXPERT_PHASES[0]["instruction"]
        elif expert_request == "pause":
            expert_controller.pause()
            with state.lock:
                state.expert_demonstration = expert_controller.snapshot(state.expert_reference_demo)
                state.coaching_cue = expert_controller.paused_reason or "Expert paused."
        elif expert_request == "resume":
            expert_controller.resume()
            with state.lock:
                state.expert_demonstration = expert_controller.snapshot(state.expert_reference_demo)
                state.coaching_cue = next(
                    (phase["instruction"] for phase in EXPERT_PHASES if phase["id"] == expert_controller.phase),
                    "Expert demonstration resumed.",
                )
        elif expert_request == "take_over":
            expert_controller.take_over()
            upstream_expert_active = False
            with state.lock:
                state.expert_demonstration = expert_controller.snapshot(state.expert_reference_demo)
                state.autonomy_mode = "manual"
                state.operator_input_source = "keyboard_pointer"
                state.coaching_cue = (
                    f"Control transferred during {expert_controller.takeover_phase or 'the procedure'}. "
                    "The current simulator state and recording are preserved."
                )
        elif expert_request == "cancel":
            expert_controller.cancel()
            upstream_expert_active = False
            with state.lock:
                state.expert_demonstration = expert_controller.snapshot(state.expert_reference_demo)
                state.expert_clean_run = False
                if state.recording:
                    state.record_request = "stop"

        if (
            not reset_requested
            and (
                camera_view_request is not None
                or (
                    shared_camera_renderer
                    and selected_active_camera.startswith("endoscope_")
                )
            )
        ):
            with torch.inference_mode():
                apply_endoscope_camera_view(
                    scenario_id,
                    camera_view_request or selected_camera_view_mode,
                    camera_adjustment,
                    selected_active_camera,
                )

        if record_request == "start":
            if capture_spool is not None:
                capture_spool.abort()
            capture_spool = BoundedCaptureSpool(state.demo_dir)
            demo_started_at = datetime.now(timezone.utc).isoformat()
            demo_started_monotonic = time.monotonic()
            last_vision_sample_time = 0.0
            with state.lock:
                state.recording = True
                state.recorded_frames = 0
                state.recorded_bytes_estimate = 0
                state.recording_queue_depth = 0
                state.recording_buffered_frames = 0
                state.intervention_count = 0
                state.procedure_phase = expert_controller.phase if expert_controller.active else "setup"
                state.procedure_event_code = 0
                state.procedure_event_sequence = 0
                state.procedure_events.clear()
                if expert_controller.active:
                    state.procedure_events.append(
                        {
                            "time": datetime.now(timezone.utc).isoformat(),
                            "recorded_frame": 0,
                            "frame_alignment": "next_control_frame_index",
                            "sim_step": state.sim_step,
                            "phase": "rest",
                            "event": None,
                            "event_sequence": 0,
                            "note": "Expert state machine started from the neutral pose",
                        }
                    )
        elif record_request == "stop":
            try:
                name = save_demo(state, capture_spool, demo_started_at) if capture_spool is not None else None
                save_error = None
            except Exception as exc:
                name = None
                save_error = f"Demonstration could not be saved: {exc}"
                traceback.print_exc()
            finally:
                if capture_spool is not None:
                    capture_spool.abort()
                capture_spool = None
                with state.lock:
                    state.recording_queue_depth = 0
                    state.recording_buffered_frames = 0
            with state.lock:
                expert_reference_pending = state.expert_reference_pending
            expert_reference_saved = False
            if name and expert_reference_pending:
                try:
                    references = read_reference_map(state.demo_dir)
                    references[state.task] = name
                    write_reference_map(state.demo_dir, references)
                    expert_reference_saved = True
                except OSError as exc:
                    save_error = f"Expert trajectory saved, but its reference index could not be updated: {exc}"
            with state.lock:
                state.recording = False
                state.last_demo = name or state.last_demo
                state.expert_reference_pending = False
                if expert_reference_saved and name:
                    state.expert_reference_demo = name
                    state.reference_ghost_demo = name
                    state.reference_ghost_enabled = True
                    state.reference_ghost_update = name
                    state.expert_demonstration = expert_controller.snapshot(name)
                    state.coaching_cue = (
                        "Simulation expert trajectory saved and selected as a Behavior Cloning reference candidate. "
                        "Clinician review is still required before research use."
                    )
                if save_error:
                    state.coaching_cue = save_error
                    state.evaluation_status = "failed"
                if state.evaluation_status == "saving":
                    state.evaluation_status = "complete"
                state.evaluation_output = name if state.evaluation_status in {"complete", "interrupted", "failed"} else state.evaluation_output
                if state.evaluation_status in {"complete", "interrupted", "failed"}:
                    state.evaluation_source = None

        if replay_request == "stop":
            replay_actions = None
            replay_index = 0
            with state.lock:
                state.replaying = False
                if state.autonomy_mode == "supervised_replay":
                    state.autonomy_mode = "manual"
                    state.coaching_cue = "Replay stopped. Manual control is active."
        elif replay_request:
            replay_path = args_cli.demo_dir / Path(replay_request).name
            try:
                with np.load(replay_path, allow_pickle=False) as replay_data:
                    replay_key = (
                        "cartesian_actions"
                        if "cartesian_actions" in replay_data.files
                        else "actions"
                    )
                    replay_actions = np.asarray(replay_data[replay_key], dtype=np.float32)
                replay_index = 0
                if not reset_requested:
                    with torch.inference_mode():
                        reset_environment(scenario_id, scenario_seed)
                with state.lock:
                    state.replaying = True
            except (OSError, KeyError, ValueError):
                replay_actions = None
                with state.lock:
                    state.replaying = False
                    state.autonomy_mode = "manual"
                    if state.evaluation_status == "running":
                        state.evaluation_status = "failed"
                        state.record_request = "stop"
                    state.coaching_cue = "Replay could not start. Manual control remains active."

        if upstream_expert_active and expert_controller.active:
            action_np = np.zeros(state.action_dim, dtype=np.float32)
            action_uses_upstream_softmimicgen_units = True
            if expert_controller.status == "running" and upstream_expert_actions is not None:
                if upstream_expert_index < len(upstream_expert_actions):
                    action_np = upstream_expert_actions[upstream_expert_index].copy()
                    upstream_expert_index += 1
                    progress = upstream_expert_index / max(1, len(upstream_expert_actions))
                    phase_index = min(len(EXPERT_PHASES) - 1, int(progress * len(EXPERT_PHASES)))
                    expert_controller.phase_index = phase_index
                    expert_controller.phase_ticks = upstream_expert_index
                    expert_controller.completed_phases = [
                        item["id"] for item in EXPERT_PHASES[:phase_index]
                    ]
                    with state.lock:
                        state.grippers_open = [
                            bool(action_np[state.gripper_action_index(arm)] > 0.0)
                            for arm in range(state.arms)
                        ]
                        state.gripper_apertures = [
                            float(np.clip(
                                (action_np[state.gripper_action_index(arm)] + 1.0) * 0.5,
                                0.0,
                                1.0,
                            ))
                            for arm in range(state.arms)
                        ]
                        state.operator_input_source = "nvidia_softmimicgen_expert"
                        state.procedure_phase = expert_controller.phase or "manipulate"
                        state.expert_demonstration = expert_controller.snapshot(
                            state.expert_reference_demo
                        )
                        state.coaching_cue = (
                            f"NVIDIA SoftMimicGen expert · frame {upstream_expert_index}/"
                            f"{len(upstream_expert_actions)}. Pause or take control at any time."
                        )
                if upstream_expert_index >= len(upstream_expert_actions):
                    upstream_expert_active = False
                    expert_controller.status = "completed"
                    expert_controller.phase_index = len(EXPERT_PHASES) - 1
                    expert_controller.completed_phases = [item["id"] for item in EXPERT_PHASES]
                    with state.lock:
                        state.expert_demonstration = expert_controller.snapshot(
                            state.expert_reference_demo
                        )
                        state.procedure_phase = "recover"
                        state.procedure_event_code = PROCEDURE_EVENTS["task_complete"]
                        state.procedure_event_sequence += 1
                        state.record_request = "stop"
                        state.coaching_cue = (
                            "NVIDIA's final expert action is being evaluated by the live task predicate."
                        )
        elif expert_controller.active:
            expert_tools = {
                arm: position
                for arm in range(state.arms)
                if (position := tool_position_for_arm(arm)) is not None
            }
            expert_object = None
            if bimanual_softmimicgen and "suture_needle" in objects:
                expert_object = (
                    objects["suture_needle"].data.root_pos_w[0, :3]
                    .detach().cpu().numpy().astype(np.float32)
                )
            elif (
                bench_dr_anmar_suture_enabled
                and "dr_anmar_threaded_needle" in objects
            ):
                expert_object = (
                    objects["dr_anmar_threaded_needle"]
                    .data.root_pos_w[0, :3]
                    .detach().cpu().numpy().astype(np.float32)
                )
            elif (
                "dr_anmar_needle" not in selected_bench_assets
                and dr_anmar_needle_enabled
            ):
                expert_object = suture_body_position(suture_needle_view)
            elif objects:
                expert_object = next(iter(objects.values())).data.root_pos_w[0, :3].detach().cpu().numpy().astype(np.float32)
            elif native_tissue is not None:
                nodal_position_value = native_tissue.data.nodal_pos_w
                nodal_positions = getattr(nodal_position_value, "torch", nodal_position_value)
                expert_object = nodal_positions[0].mean(dim=0).detach().cpu().numpy().astype(np.float32)
            expert_command = expert_controller.step(
                expert_tools,
                expert_object,
                grippers_open,
                safety_envelope_active=False,
                hoop_passed=state.upstream_task_success is True,
                native_grasp_contact_active=[arm in native_grasp_arms for arm in range(state.arms)],
            )
            action_np = expert_command.action
            grippers_open = expert_command.grippers_open
            if state.has_grippers:
                for arm, is_open in enumerate(grippers_open):
                    action_np[state.gripper_action_index(arm)] = 1.0 if is_open else -1.0
            with state.lock:
                state.grippers_open = list(grippers_open)
                state.gripper_apertures = [1.0 if value else 0.0 for value in grippers_open]
                state.operator_input_source = "automation_policy"
                state.expert_demonstration = expert_controller.snapshot(state.expert_reference_demo)
                state.expert_clean_run = state.expert_clean_run and not expert_controller.degraded_reasons
                if expert_command.phase_changed:
                    phase = expert_controller.phase or "recover"
                    state.procedure_phase = phase
                    event = {
                        "approach": "target_visible",
                        "contact": "contact",
                        "grasp": "grasp",
                    }.get(phase)
                    if event:
                        state.procedure_event_code = PROCEDURE_EVENTS[event]
                        state.procedure_event_sequence += 1
                    annotation = {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "recorded_frame": state.recorded_frames,
                        "frame_alignment": "next_control_frame_index",
                        "sim_step": state.sim_step,
                        "phase": phase,
                        "event": event,
                        "event_sequence": state.procedure_event_sequence,
                        "note": "Expert state-machine phase transition",
                    }
                    state.procedure_events.append(annotation)
                    state.coaching_cue = next(
                        (item["instruction"] for item in EXPERT_PHASES if item["id"] == phase),
                        "Expert demonstration running.",
                    )
                if expert_command.completed:
                    state.procedure_phase = "recover"
                    state.procedure_event_code = PROCEDURE_EVENTS["task_complete"]
                    state.procedure_event_sequence += 1
                    state.procedure_events.append(
                        {
                            "time": datetime.now(timezone.utc).isoformat(),
                            "recorded_frame": state.recorded_frames,
                            "frame_alignment": "next_control_frame_index",
                            "sim_step": state.sim_step,
                            "phase": "recover",
                            "event": "task_complete",
                            "event_sequence": state.procedure_event_sequence,
                            "note": "Expert state machine completed all eight phases",
                        }
                    )
                    state.expert_reference_pending = bool(state.expert_clean_run)
                    state.record_request = "stop"
                    state.coaching_cue = (
                        "Expert trajectory complete. Saving a Behavior Cloning reference candidate for clinician review."
                        if state.expert_clean_run
                        else "Expert trajectory completed with qualification warnings; saving without automatic reference approval."
                    )
        elif replay_actions is not None and replay_index < len(replay_actions):
            action_np = replay_actions[replay_index].copy()
            replay_index += 1
            action_uses_upstream_softmimicgen_units = _softmimicgen_task
            with state.lock:
                state.operator_input_source = "supervised_replay"
        else:
            if replay_actions is not None:
                replay_actions = None
                with state.lock:
                    state.replaying = False
                    state.autonomy_mode = "manual"
                    if state.evaluation_status == "running":
                        state.evaluation_status = "saving"
                        state.record_request = "stop"
                        state.coaching_cue = "Challenge rollout finished. Saving synchronized evidence."
                    else:
                        state.coaching_cue = "Supervised replay finished. Manual control is active."
            action_np = manual_action.copy()
        if state.has_grippers:
            for arm, aperture in enumerate(gripper_apertures):
                action_np[state.gripper_action_index(arm)] = proportional_gripper_action(aperture)

        skin_adhesive_target = 0.0
        if (
            skin_adhesive_enabled
            and skin_adhesive_articulation is not None
            and state.has_grippers
        ):
            adhesive_gripper_action = float(
                action_np[
                    state.gripper_action_index(
                        skin_adhesive_mounted_arm
                    )
                ]
            )
            skin_adhesive_target = float(
                np.clip(
                    (1.0 - adhesive_gripper_action) * 0.5,
                    0.0,
                    1.0,
                )
            )
            set_skin_adhesive_activation_target(
                skin_adhesive_articulation,
                skin_adhesive_target,
            )
            with state.lock:
                state.skin_adhesive_target = skin_adhesive_target

        if _softmimicgen_task and not action_uses_upstream_softmimicgen_units:
            # NVIDIA's released task accepts small metric deltas directly and
            # then applies its scalar 0.5 controller scale.  Dr.Anmar's doctor
            # controls use normalized game-style axes.  Convert only operator
            # and Dr.Anmar controller input; recorded upstream actions are
            # replayed byte-for-byte in their original seven-dimensional space.
            for arm in range(state.arms):
                body = state.body_action_slice(arm)
                action_np[body.start : body.start + 3] *= 0.02
                action_np[body.start + 3 : body.start + 6] *= 0.10

        grasp_distances: list[float | None] = [None] * state.arms
        grasp_offsets: list[list[float] | None] = [None] * state.arms
        grasp_target_position = None
        if bimanual_softmimicgen and "suture_needle" in objects:
            grasp_target_position = (
                objects["suture_needle"].data.root_pos_w[0, :3]
                .detach().cpu().numpy().astype(np.float32)
            )
        elif (
            bench_dr_anmar_suture_enabled
            and "dr_anmar_threaded_needle" in objects
        ):
            grasp_target_position = (
                objects["dr_anmar_threaded_needle"]
                .data.root_pos_w[0, :3]
                .detach().cpu().numpy().astype(np.float32)
            )
        elif interactive_deformable is not None:
            nodal_position_value = interactive_deformable.data.nodal_pos_w
            nodal_positions = getattr(nodal_position_value, "torch", nodal_position_value)
            grasp_target_position = nodal_positions[0].mean(dim=0).detach().cpu().numpy().astype(np.float32)
        elif (
            "dr_anmar_needle" not in selected_bench_assets
            and dr_anmar_needle_enabled
        ):
            grasp_target_position = suture_body_position(suture_needle_view)
        elif objects:
            grasp_object = next(iter(objects.values()))
            grasp_target_position = grasp_object.data.root_pos_w[0, :3].detach().cpu().numpy().astype(np.float32)
        if state.has_grippers and grasp_target_position is not None:
            previous_native_grasps = set(native_grasp_arms)
            observed_native_grasps: set[int] = set()
            for arm, is_open in enumerate(grippers_open):
                tool_position_for_grasp = tool_position_for_arm(arm)
                if tool_position_for_grasp is not None:
                    object_offset = grasp_target_position - tool_position_for_grasp
                    grasp_distances[arm] = float(np.linalg.norm(object_offset))
                    grasp_offsets[arm] = object_offset.astype(float).round(5).tolist()
                if (
                    not is_open
                    and native_gripper_contact_force(arm) > 0.005
                    and (
                        interactive_deformable is not None
                        or (
                            grasp_distances[arm] is not None
                            and grasp_distances[arm] <= state.grasp_capture_radius_m
                        )
                    )
                ):
                    observed_native_grasps.add(arm)
                    if arm not in previous_native_grasps:
                        with state.lock:
                            state.coaching_cue = "Physical jaw contact detected. Lift smoothly and verify that the object follows."
            native_grasp_arms.clear()
            native_grasp_arms.update(observed_native_grasps)
            with state.lock:
                # Compatibility field for saved demonstrations. It now reports
                # observed PhysX contact and never means an authored attachment.
                state.native_grasp_contact_active = [arm in native_grasp_arms for arm in range(state.arms)]
                state.tool_to_object_distance_m = [round(value, 5) if value is not None else None for value in grasp_distances]
                state.tool_to_object_offset_m = grasp_offsets


        # Dr.Anmar forwards the requested action unchanged. PhysX/Isaac Lab own
        # contacts, limits, constraints and the resulting physical state.
        actions = torch.from_numpy(action_np).to(device=env.unwrapped.device).reshape(1, -1)
        env_step_started = time.perf_counter()
        with torch.inference_mode():
            if stapler_articulation is not None:
                # The test cell replaces a robot grasp with an idealized
                # six-DOF proportional velocity fixture. It restores the
                # captured housing datum without overwriting the root pose or
                # either articulated mechanism coordinate.
                stapler_root_velocity = stapler_articulation.data.root_vel_w
                fixture_velocity = torch.zeros_like(
                    stapler_root_velocity
                )
                if (
                    stapler_housing_body_index is not None
                    and stapler_fixture_position_w is not None
                    and stapler_fixture_quaternion_w is not None
                ):
                    current_fixture_position = (
                        stapler_articulation.data.body_pos_w[
                            0,
                            stapler_housing_body_index,
                        ]
                    )
                    current_fixture_quaternion = (
                        stapler_articulation.data.body_quat_w[
                            0,
                            stapler_housing_body_index,
                        ]
                    )
                    fixture_velocity[0, :3] = torch.clamp(
                        25.0
                        * (
                            stapler_fixture_position_w
                            - current_fixture_position
                        ),
                        -0.05,
                        0.05,
                    )
                    quaternion_error = quat_mul(
                        stapler_fixture_quaternion_w.unsqueeze(0),
                        quat_conjugate(
                            current_fixture_quaternion.unsqueeze(0)
                        ),
                    )[0]
                    quaternion_sign = torch.where(
                        quaternion_error[0] >= 0.0,
                        torch.ones_like(quaternion_error[0]),
                        -torch.ones_like(quaternion_error[0]),
                    )
                    fixture_velocity[0, 3:] = torch.clamp(
                        50.0
                        * quaternion_sign
                        * quaternion_error[1:],
                        -0.5,
                        0.5,
                    )
                stapler_articulation.write_root_velocity_to_sim(
                    fixture_velocity
                )
            write_native_attachment()
            _observations, reward, terminated, truncated, info = env.step(actions)
            if dynamic_patient_runtime is not None:
                patient_contact_target = str(
                    procedure.get(
                        "dynamic_patient_contact_target",
                        "mesentery",
                    )
                )
                patient_contact_interaction = str(
                    procedure.get(
                        "dynamic_patient_contact_interaction",
                        "exposure",
                    )
                )
                sensor_pairs_observed = 0
                for arm in range(arms):
                    left_force = contact_effect_jaw_force(arm, 1)
                    right_force = contact_effect_jaw_force(arm, 2)
                    tool_position = tool_position_for_arm(arm)
                    if (
                        left_force is None
                        or right_force is None
                        or tool_position is None
                    ):
                        continue
                    dynamic_patient_runtime.contacts.observe(
                        PatientContactFrame(
                            target=patient_contact_target,
                            source_robot=f"psm_{arm + 1}",
                            interaction=patient_contact_interaction,
                            normal_forces_n=(
                                left_force,
                                right_force,
                            ),
                            tool_position_m=tuple(
                                float(value)
                                for value in tool_position
                            ),
                        )
                    )
                    sensor_pairs_observed += 1
                dynamic_patient_runtime.step(
                    float(env_cfg.sim.dt * env_cfg.decimation)
                )
                target_tissue_state = (
                    dynamic_patient_runtime.tissue_state.get(
                        patient_contact_target
                    )
                )
                latest_dynamic_patient_telemetry = {
                    **dynamic_patient_runtime.contact_effects.snapshot(),
                    "sensor_pair_count": sensor_pairs_observed,
                    "sensor_authority_available": (
                        sensor_pairs_observed == arms
                    ),
                    "vital_signs": asdict(
                        dynamic_patient_runtime.vital_signs
                    ),
                    "target_tissue": {
                        **asdict(target_tissue_state),
                        "active_adhesions": sorted(
                            target_tissue_state.active_adhesions
                        ),
                    },
                }
            if autonomous_rescue_runtime is not None:
                if rescue_target_position_w is None:
                    raise RuntimeError(
                        "Autonomous Rescue OR target frame is unavailable"
                    )
                action_period_s = float(
                    env_cfg.sim.dt * env_cfg.decimation
                )
                rescue_candidates: list[dict[str, Any]] = []
                for arm in range(arms):
                    left_force = contact_effect_jaw_force(arm, 1)
                    right_force = contact_effect_jaw_force(arm, 2)
                    jaw_positions = jaw_positions_for_arm(arm)
                    tool_position = tool_position_for_arm(arm)
                    if (
                        left_force is None
                        or right_force is None
                        or jaw_positions is None
                        or tool_position is None
                    ):
                        continue
                    previous_tool_position = (
                        rescue_previous_tool_positions.get(arm)
                    )
                    tool_speed_m_s = (
                        float(
                            np.linalg.norm(
                                tool_position - previous_tool_position
                            )
                            / action_period_s
                        )
                        if previous_tool_position is not None
                        else 0.0
                    )
                    rescue_previous_tool_positions[arm] = (
                        tool_position.copy()
                    )
                    jaw_center_w = 0.5 * (
                        jaw_positions[0] + jaw_positions[1]
                    )
                    target_distance_m = float(
                        np.linalg.norm(
                            jaw_center_w - rescue_target_position_w
                        )
                    )
                    rescue_candidates.append(
                        {
                            "arm": arm,
                            "left_force_n": left_force,
                            "right_force_n": right_force,
                            "separation_m": float(
                                np.linalg.norm(
                                    jaw_positions[0] - jaw_positions[1]
                                )
                            ),
                            "tool_speed_m_s": tool_speed_m_s,
                            "target_distance_m": target_distance_m,
                        }
                    )
                selected_contact = (
                    max(
                        rescue_candidates,
                        key=lambda item: (
                            min(
                                float(item["left_force_n"]),
                                float(item["right_force_n"]),
                            )
                            * max(
                                0.0,
                                1.0
                                - float(item["target_distance_m"])
                                / (
                                    autonomous_rescue_runtime.effects
                                    .calibration.maximum_target_radius_m
                                ),
                            )
                        ),
                    )
                    if rescue_candidates
                    else {
                        "arm": 0,
                        "left_force_n": 0.0,
                        "right_force_n": 0.0,
                        "separation_m": 0.02,
                        "tool_speed_m_s": 0.0,
                        "target_distance_m": 1.0,
                    }
                )
                rescue_physics_step += 1
                rescue_simulation_time_s += action_period_s
                rescue_observation = (
                    autonomous_rescue_runtime.advance_scene(
                        PhysicsEvidenceFrame(
                            physics_step=rescue_physics_step,
                            simulation_time_s=(
                                rescue_simulation_time_s
                            ),
                            dt_s=action_period_s,
                            station_id=(
                                f"psm_{int(selected_contact['arm']) + 1}"
                            ),
                            tool_id="psm_grasper",
                            target_id="rescue_vessel",
                            left_normal_force_n=float(
                                selected_contact["left_force_n"]
                            ),
                            right_normal_force_n=float(
                                selected_contact["right_force_n"]
                            ),
                            separation_m=float(
                                selected_contact["separation_m"]
                            ),
                            tool_speed_m_s=float(
                                selected_contact["tool_speed_m_s"]
                            ),
                            target_distance_m=float(
                                selected_contact["target_distance_m"]
                            ),
                        )
                    )
                )
                rescue_patient = rescue_observation["patient"]
                rescue_vessel = rescue_patient["vessel"]
                rescue_complications = [
                    {
                        "id": str(item["id"]),
                        "priority": int(item["priority"]),
                        "target_id": str(item["target_id"]),
                    }
                    for item in rescue_observation[
                        "active_complications"
                    ]
                ]
                rescue_plan = rescue_observation["rescue_plan"]
                current_compression_fraction = float(
                    rescue_vessel[
                        "transient_compression_fraction"
                    ]
                )
                peak_compression_fraction = max(
                    float(
                        latest_autonomous_rescue_telemetry.get(
                            "peak_compression_fraction",
                            0.0,
                        )
                    ),
                    current_compression_fraction,
                )
                release_observed = bool(
                    latest_autonomous_rescue_telemetry.get(
                        "release_observed",
                        False,
                    )
                    or (
                        peak_compression_fraction > 0.1
                        and current_compression_fraction < 0.02
                    )
                )
                latest_autonomous_rescue_telemetry = {
                    "physics_step": int(
                        rescue_patient["physics_step"]
                    ),
                    "simulation_time_s": float(
                        rescue_patient["simulation_time_s"]
                    ),
                    "sensor_pair_count": len(rescue_candidates),
                    "sensor_authority_available": bool(
                        rescue_candidates
                    ),
                    "selected_arm": (
                        int(selected_contact["arm"]) + 1
                    ),
                    "peak_compression_fraction": (
                        peak_compression_fraction
                    ),
                    "release_observed": release_observed,
                    "measured_contact": {
                        "left_normal_force_n": float(
                            selected_contact["left_force_n"]
                        ),
                        "right_normal_force_n": float(
                            selected_contact["right_force_n"]
                        ),
                        "jaw_separation_m": float(
                            selected_contact["separation_m"]
                        ),
                        "tool_speed_m_s": float(
                            selected_contact["tool_speed_m_s"]
                        ),
                        "target_distance_m": float(
                            selected_contact["target_distance_m"]
                        ),
                    },
                    "vessel": {
                        key: value
                        for key, value in rescue_vessel.items()
                    },
                    "vital_signs": asdict(
                        autonomous_rescue_patient_runtime.vital_signs
                    ),
                    "fluid_balance": asdict(
                        autonomous_rescue_patient_runtime.fluid_balance
                    ),
                    "active_complications": rescue_complications,
                    "rescue_plan": (
                        {
                            "complication_id": str(
                                rescue_plan["complication_id"]
                            ),
                            "protocol_id": str(
                                rescue_plan["protocol_id"]
                            ),
                            "action_count": len(
                                rescue_plan["actions"]
                            ),
                        }
                        if rescue_plan is not None
                        else None
                    ),
                    "contact_owned_reward": float(
                        rescue_observation["last_reward"]
                    ),
                    "outcome_authority": (
                        "post_physics_filtered_local_contact"
                    ),
                }
            if (
                stapler_articulation is not None
                and stapler_housing_body_index is not None
                and stapler_fixture_position_w is None
                and stapler_fixture_quaternion_w is None
            ):
                # Isaac's articulation tensors become authoritative after the
                # first PhysX advance. Capture the fixture datum here, then
                # measure every later frame against it.
                stapler_fixture_position_w = (
                    stapler_articulation.data.body_pos_w[
                        0,
                        stapler_housing_body_index,
                    ]
                    .detach()
                    .clone()
                )
                stapler_fixture_quaternion_w = (
                    stapler_articulation.data.body_quat_w[
                        0,
                        stapler_housing_body_index,
                    ]
                    .detach()
                    .clone()
                )
            env_step_finished = time.perf_counter()
            with state.lock:
                suture_visual_active = (
                    state.recording
                    or any(
                        count > 0
                        for count in state.camera_subscribers.values()
                    )
                    or time.monotonic() - state.camera_poll_last_seen < 1.0
                )
            if suture_visual_active:
                curve_update_started = time.perf_counter()
                update_realtime_suture_curve(suture_render_positions())
                curve_update_ms = (
                    time.perf_counter() - curve_update_started
                ) * 1000.0
            else:
                curve_update_ms = 0.0
            update_wrist_camera_poses(
                wrist_camera_adjustments,
                selected_active_camera,
            )
            if state.native_psm_policy_contract:
                native_policy_tensor, native_target_tensor, native_robot_names = canonical_policy_contract(env)
                if native_robot_names != tuple(state.native_psm_robot_names):
                    raise RuntimeError("The active PSM articulation order changed during the episode")
                native_policy_action_np = native_policy_tensor[0].detach().cpu().numpy().astype(np.float32)
                native_joint_targets_np = native_target_tensor[0].detach().cpu().numpy().astype(np.float32)
            else:
                native_policy_action_np = None
                native_joint_targets_np = None
        if stapler_test_cell_enabled and stapler_articulation is not None:
            actual_trigger_rad = float(
                stapler_articulation.data.joint_pos[
                    0,
                    stapler_trigger_joint_index,
                ].item()
            )
            actual_trigger_deg = float(np.degrees(actual_trigger_rad))
            actual_pusher_mm = (
                float(
                    stapler_articulation.data.joint_pos[
                        0,
                        stapler_pusher_joint_index,
                    ].item()
                )
                * 1000.0
            )
            actual_trigger_velocity_deg_s = float(
                np.degrees(
                    stapler_articulation.data.joint_vel[
                        0,
                        stapler_trigger_joint_index,
                    ].item()
                )
            )
            actual_pusher_velocity_mm_s = (
                float(
                    stapler_articulation.data.joint_vel[
                        0,
                        stapler_pusher_joint_index,
                    ].item()
                )
                * 1000.0
            )
            if (
                stapler_closure_tissues
                and stapler_tissue_default_positions is not None
            ):
                tissue_positions = torch.cat(
                    [
                        getattr(
                            tissue_flap.data.nodal_pos_w,
                            "torch",
                            tissue_flap.data.nodal_pos_w,
                        )
                        for tissue_flap in stapler_closure_tissues
                    ],
                    dim=1,
                )
                current_tissue_displacement_mm = float(
                    torch.linalg.vector_norm(
                        tissue_positions
                        - stapler_tissue_default_positions,
                        dim=-1,
                    )
                    .max()
                    .item()
                    * 1000.0
                )
                stapler_tissue_max_displacement_mm = max(
                    stapler_tissue_max_displacement_mm,
                    current_tissue_displacement_mm,
                )
            fixture_translation_error_mm = 0.0
            fixture_rotation_error_deg = 0.0
            if (
                stapler_housing_body_index is not None
                and stapler_fixture_position_w is not None
                and stapler_fixture_quaternion_w is not None
            ):
                current_fixture_position = (
                    stapler_articulation.data.body_pos_w[
                        0,
                        stapler_housing_body_index,
                    ]
                )
                current_fixture_quaternion = (
                    stapler_articulation.data.body_quat_w[
                        0,
                        stapler_housing_body_index,
                    ]
                )
                fixture_translation_error_mm = float(
                    torch.linalg.vector_norm(
                        current_fixture_position
                        - stapler_fixture_position_w
                    ).item()
                    * 1000.0
                )
                quaternion_dot = torch.clamp(
                    torch.abs(
                        torch.dot(
                            current_fixture_quaternion,
                            stapler_fixture_quaternion_w,
                        )
                    ),
                    0.0,
                    1.0,
                )
                fixture_rotation_error_deg = float(
                    np.degrees(
                        2.0 * np.arccos(float(quaternion_dot.item()))
                    )
                )
            applied_effort = getattr(
                stapler_articulation.data,
                "applied_torque",
                None,
            )
            trigger_effort = (
                float(
                    applied_effort[
                        0,
                        stapler_trigger_joint_index,
                    ].item()
                )
                if applied_effort is not None
                else None
            )
            pusher_effort = (
                float(
                    applied_effort[
                        0,
                        stapler_pusher_joint_index,
                    ].item()
                )
                if applied_effort is not None
                else None
            )
            # A logical deployment requires both an explicit full-stroke
            # command and a measured threshold crossing.  Physical overshoot
            # from a partial command is therefore surfaced as a failed
            # mechanism check rather than misreported as a valid firing edge.
            deployment_event = None
            if actual_trigger_deg <= REARM_THRESHOLD_DEG or (
                stapler_target_deg >= FIRE_THRESHOLD_DEG
                and actual_trigger_deg >= FIRE_THRESHOLD_DEG
            ):
                deployment_event = stapler_deployment_controller.update(
                    actual_trigger_rad
                )
            if stapler_partial_candidate:
                stapler_partial_peak_deg = max(
                    stapler_partial_peak_deg,
                    actual_trigger_deg,
                )
                if (
                    stapler_partial_peak_deg > REARM_THRESHOLD_DEG
                    and actual_trigger_deg <= REARM_THRESHOLD_DEG
                ):
                    stapler_partial_stroke_attempts += 1
                    if (
                        stapler_magazine.deployed
                        == stapler_partial_start_deployments
                        and stapler_partial_peak_deg
                        < FIRE_THRESHOLD_DEG
                    ):
                        stapler_partial_stroke_passes += 1
                    stapler_partial_candidate = False
            if deployment_event is not None:
                placement_event = record_stapler_placement()
                stapler_pending_advance = True
                stapler_last_event = {
                    "sequence_index": deployment_event.sequence_index,
                    "trigger_deg": round(
                        deployment_event.trigger_position_deg,
                        4,
                    ),
                    "remaining": deployment_event.remaining,
                    "state": deployment_event.state.value,
                    "sim_step": fps_steps,
                    "closure_station": stapler_active_station_index + 1,
                    "placement": placement_event,
                }
            with state.lock:
                previous_max_trigger = float(
                    state.stapler_test_cell.get("max_trigger_deg", 0.0)
                )
                previous_max_pusher = float(
                    state.stapler_test_cell.get(
                        "max_pusher_travel_mm",
                        0.0,
                    )
                )
                previous_max_fixture_translation = float(
                    state.stapler_test_cell.get(
                        "max_fixture_translation_error_mm",
                        0.0,
                    )
                )
                previous_max_fixture_rotation = float(
                    state.stapler_test_cell.get(
                        "max_fixture_rotation_error_deg",
                        0.0,
                    )
                )
                state.stapler_test_cell.update(
                    {
                        "cycle_phase": stapler_cycle_phase,
                        "cycle_running": stapler_cycle_started_at
                        is not None,
                        "target_trigger_deg": round(
                            stapler_target_deg,
                            4,
                        ),
                        "actual_trigger_deg": round(
                            actual_trigger_deg,
                            4,
                        ),
                        "trigger_velocity_deg_s": round(
                            actual_trigger_velocity_deg_s,
                            4,
                        ),
                        "pusher_travel_mm": round(
                            actual_pusher_mm,
                            4,
                        ),
                        "max_pusher_travel_mm": round(
                            max(previous_max_pusher, actual_pusher_mm),
                            4,
                        ),
                        "pusher_velocity_mm_s": round(
                            actual_pusher_velocity_mm_s,
                            4,
                        ),
                        "tracking_error_deg": round(
                            stapler_target_deg - actual_trigger_deg,
                            4,
                        ),
                        "joint_limit_violation_deg": round(
                            max(
                                0.0,
                                -actual_trigger_deg,
                                actual_trigger_deg - TRIGGER_LIMIT_DEG,
                            ),
                            4,
                        ),
                        "fixture_translation_error_mm": round(
                            fixture_translation_error_mm,
                            4,
                        ),
                        "fixture_rotation_error_deg": round(
                            fixture_rotation_error_deg,
                            4,
                        ),
                        "max_fixture_translation_error_mm": round(
                            max(
                                previous_max_fixture_translation,
                                fixture_translation_error_mm,
                            ),
                            4,
                        ),
                        "max_fixture_rotation_error_deg": round(
                            max(
                                previous_max_fixture_rotation,
                                fixture_rotation_error_deg,
                            ),
                            4,
                        ),
                        "trigger_drive_effort_provisional": (
                            round(trigger_effort, 6)
                            if trigger_effort is not None
                            else None
                        ),
                        "pusher_drive_effort_provisional": (
                            round(pusher_effort, 6)
                            if pusher_effort is not None
                            else None
                        ),
                        "magazine_remaining": stapler_magazine.remaining,
                        "deployment_count": stapler_magazine.deployed,
                        "cycle_count": stapler_cycle_count,
                        "partial_stroke_attempts": (
                            stapler_partial_stroke_attempts
                        ),
                        "partial_stroke_passes": (
                            stapler_partial_stroke_passes
                        ),
                        "max_trigger_deg": round(
                            max(previous_max_trigger, actual_trigger_deg),
                            4,
                        ),
                        **stapler_closure_payload(),
                        "last_event": stapler_last_event,
                    }
                )
                if deployment_event is not None:
                    state.procedure_event_code = PROCEDURE_EVENTS[
                        "task_complete"
                    ]
                    state.procedure_event_sequence += 1
                    closure_complete = (
                        len(stapler_closed_station_indices)
                        == len(STAPLER_CLOSURE_STATION_OFFSETS_M)
                    )
                    state.coaching_cue = (
                        "All seven rigid staples are retaining the approximated "
                        "FEM tissue. Review the closure evidence or reset."
                        if closure_complete
                        else "The rigid staple is retaining both FEM tissue "
                        "attachment bands. The fixture will advance after "
                        "release."
                    )
        if (
            skin_adhesive_enabled
            and skin_adhesive_articulation is not None
        ):
            adhesive_joint_positions = (
                skin_adhesive_articulation.data.joint_pos[0]
            )
            left_paddle_rad = float(
                adhesive_joint_positions[
                    skin_adhesive_joint_indices["left_paddle_joint"]
                ].item()
            )
            right_paddle_rad = float(
                adhesive_joint_positions[
                    skin_adhesive_joint_indices["right_paddle_joint"]
                ].item()
            )
            piston_travel_m = float(
                adhesive_joint_positions[
                    skin_adhesive_joint_indices["metering_piston_joint"]
                ].item()
            )
            fully_activated = skin_adhesive_activation_targets(1.0)
            activation_components = (
                abs(
                    left_paddle_rad
                    / fully_activated["left_paddle_joint"]
                ),
                abs(
                    right_paddle_rad
                    / fully_activated["right_paddle_joint"]
                ),
                abs(
                    piston_travel_m
                    / fully_activated["metering_piston_joint"]
                ),
            )
            actual_activation = float(
                np.clip(np.mean(activation_components), 0.0, 1.0)
            )
            with state.lock:
                state.skin_adhesive_system.update(
                    {
                        "workflow_state": (
                            "dispensing"
                            if skin_adhesive_target > 0.001
                            else "mounted_ready"
                        ),
                        "target_activation": round(
                            skin_adhesive_target,
                            6,
                        ),
                        "actual_activation": round(
                            actual_activation,
                            6,
                        ),
                        "left_paddle_deg": round(
                            float(np.degrees(left_paddle_rad)),
                            4,
                        ),
                        "right_paddle_deg": round(
                            float(np.degrees(right_paddle_rad)),
                            4,
                        ),
                        "piston_travel_mm": round(
                            piston_travel_m * 1000.0,
                            4,
                        ),
                        "tracking_error": round(
                            skin_adhesive_target - actual_activation,
                            6,
                        ),
                    }
                )
        if native_joint_targets_np is not None:
            close_rad = float(state.gripper_profile["close_rad"])
            open_rad = float(state.gripper_profile["open_rad"])
            span_rad = open_rad - close_rad
            resolved_apertures = [
                float(np.clip(
                    (native_joint_targets_np[arm * 7 + 6] - close_rad) / span_rad,
                    0.0,
                    1.0,
                ))
                for arm in range(state.arms)
            ]
            with state.lock:
                # Keep the operator's latest aperture as the command source.
                # A slow PhysX step may finish after a newer API frame arrives;
                # feeding that older resolved target back into command state
                # would otherwise erase the newer proportional request.
                state.gripper_profile["resolved_aperture_normalized"] = [
                    round(value, 6) for value in resolved_apertures
                ]
        with state.lock:
            state.performance_timings_ms.update(
                {
                    "env_step": round(
                        (env_step_finished - env_step_started) * 1000.0,
                        3,
                    ),
                    "suture_curve": round(curve_update_ms, 3),
                }
            )
        environment_reward = scalar_value(reward)
        environment_terminated = bool(scalar_value(terminated))
        environment_truncated = bool(scalar_value(truncated))
        environment_success = native_success_from_info(info)
        if autonomous_rescue_or_enabled:
            rescue_vessel_telemetry = (
                latest_autonomous_rescue_telemetry.get("vessel", {})
            )
            environment_reward = float(
                latest_autonomous_rescue_telemetry.get(
                    "contact_owned_reward",
                    0.0,
                )
            )
            environment_success = (
                1.0
                if bool(
                    rescue_vessel_telemetry.get(
                        "hemostasis_verified",
                        False,
                    )
                )
                else 0.0
            )
        if softmimicgen_goal is not None:
            environment_success = 1.0 if bool(
                softmimicgen_goal(env.unwrapped)[0].detach().cpu().item()
            ) else 0.0
            with state.lock:
                state.upstream_task_success = environment_success > 0.5
                if expert_controller.status == "completed":
                    state.expert_clean_run = state.upstream_task_success
                    state.expert_reference_pending = state.upstream_task_success
                    if state.upstream_task_success:
                        state.coaching_cue = (
                            "NVIDIA's released expert episode satisfied the native ring-crossing predicate. "
                            "The trajectory is ready for clinician review."
                        )
                    else:
                        state.coaching_cue = (
                            "The released trajectory finished, but the native ring-crossing predicate is not satisfied. "
                            "It will not be promoted as a reference."
                        )


        current_tool_positions = {
            arm: position
            for arm in range(state.arms)
            if (position := tool_position_for_arm(arm)) is not None
        }
        current_object_positions = {
            name: rigid_object.data.root_pos_w[0, :3]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            for name, rigid_object in objects.items()
        }
        hemostasis_clip_pose_w = None
        if hemostasis_clip_view is not None:
            clip_transforms = (
                hemostasis_clip_view.get_transforms()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            if clip_transforms.shape == (1, 7) and np.isfinite(
                clip_transforms
            ).all():
                hemostasis_clip_pose_w = clip_transforms[0]
        native_clearances = [
            clearance
            for position in current_tool_positions.values()
            if (clearance := anatomy_surface_query(position)[0]) is not None
        ]
        with state.lock:
            state.closest_anatomy_clearance_m = (
                round(min(native_clearances, key=abs), 5) if native_clearances else None
            )
            state.virtual_fixture_active = False
            state.adaptive_precision_active = False
            state.native_telemetry = {
                "contact_forces_n": dict(latest_contact_forces),
                "deformable": dict(latest_deformable_safety),
                "dr_anmar_suture": dict(latest_suture_telemetry),
                "dynamic_patient_effects": dict(
                    latest_dynamic_patient_telemetry
                ),
                "autonomous_rescue_or": dict(
                    latest_autonomous_rescue_telemetry
                ),
                "native_deformable_domain": dict(native_episode_domain),
                "dr_anmar_hemostasis": {
                    "clip_pose_w": (
                        hemostasis_clip_pose_w.round(7).tolist()
                        if hemostasis_clip_pose_w is not None
                        else None
                    ),
                    "plastic_forming": False,
                    "flow_model": "not_present_in_native_physx_room",
                    "clinical_validation": False,
                }
                if native_hemostasis_enabled
                else {},
            }
            state.native_rigid_object_positions_m = {
                name: position.astype(float).round(5).tolist()
                for name, position in current_object_positions.items()
            }
            state.native_tool_positions_m = [
                (
                    current_tool_positions[arm]
                    .astype(float)
                    .round(5)
                    .tolist()
                    if arm in current_tool_positions
                    else None
                )
                for arm in range(state.arms)
            ]
            if native_joint_targets_np is not None:
                state.gripper_profile["resolved_aperture_rad"] = [
                    round(float(native_joint_targets_np[arm * 7 + 6]), 5)
                    for arm in range(state.arms)
                ]

        motion_active = any(bool(np.any(action_np[state.body_action_slice(arm)])) for arm in range(state.arms))
        current_time = time.monotonic()
        if (
            procedure.get("nvidia_native_bench")
            and not motion_active
            and not native_grasp_arms
        ):
            with state.lock:
                untouched_native_bench = (
                    not state.procedure_motion_seen
                    and not state.procedure_grasp_seen
                )
            if untouched_native_bench:
                # Rigid assets may settle a few millimetres under PhysX after
                # reset. Keep that passive motion out of procedure progress.
                # The last fully idle pose becomes the user-action baseline.
                initial_object_positions.clear()
                initial_object_positions.update(
                    {
                        name: rigid_object.data.root_pos_w[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                        .copy()
                        for name, rigid_object in objects.items()
                    }
                )
        max_object_lift = 0.0
        max_object_motion = 0.0
        for name, rigid_object in objects.items():
            start = initial_object_positions.get(name)
            if start is None:
                continue
            current_position = rigid_object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
            max_object_lift = max(max_object_lift, float(current_position[2] - start[2]))
            max_object_motion = max(max_object_motion, float(np.linalg.norm(current_position - start)))
        current_native_centroid = native_tissue_centroid()
        if current_native_centroid is not None and initial_native_centroid[0] is not None:
            native_delta = current_native_centroid - initial_native_centroid[0]
            max_object_lift = max(max_object_lift, float(native_delta[2]))
            max_object_motion = max(max_object_motion, float(np.linalg.norm(native_delta)))
        tool_position = preferred_tool_position(robots, robot_body_names)
        with state.lock:
            if motion_active:
                state.procedure_motion_seen = True
                state.procedure_last_motion_at = current_time
            if native_grasp_arms:
                state.procedure_grasp_seen = True
            state.procedure_object_lift_m = max(state.procedure_object_lift_m, max_object_lift)
            state.procedure_object_motion_m = max(state.procedure_object_motion_m, max_object_motion)
            waypoint_index = state.procedure_waypoints_completed
            if tool_position is not None and waypoint_index < len(room_waypoints):
                if float(np.linalg.norm(tool_position - room_waypoints[waypoint_index])) <= 0.014:
                    state.procedure_waypoints_completed += 1
                    state.procedure_last_motion_at = current_time
            next_waypoint_index = state.procedure_waypoints_completed
        update_procedure_waypoint_marker(next_waypoint_index)

        with state.lock:
            is_recording = state.recording
            camera_stream_active = (
                any(count > 0 for count in state.camera_subscribers.values())
                or current_time - state.camera_poll_last_seen < 1.0
            )
        # A doctor watching any live camera is an interactive workload even
        # while the instruments are stationary.  Previously the loop fell to
        # its 500 ms unattended cadence in that state, hard-capping every
        # visible camera at 2 FPS despite an otherwise idle GPU.  Keep the
        # camera sensors at their native 25 Hz whenever a stream is connected;
        # the low-power cadence remains available after the last viewer leaves.
        interactive_active = (
            bool(np.any(manual_action))
            or replay_actions is not None
            or is_recording
            or camera_stream_active
        )
        safety_due = is_recording or loop_started - last_safety_sample_time >= 0.20
        if safety_due:
            latest_contact_forces = {}
            for name, sensor in contact_sensors.items():
                forces = sensor.data.net_forces_w[0]
                max_force = torch.linalg.vector_norm(forces, dim=-1).max()
                latest_contact_forces[f"{name}_max_contact_force_n"] = float(max_force.detach().cpu().item())
            # SoftMimicGen publishes strand kinematics and a task predicate,
            # not a calibrated surgical material/stress contract.  Keep the
            # visible displacement but do not mislabel raw FEM tensors as
            # clinically meaningful tissue stress in this dry-lab room.
            latest_deformable_safety = sample_deformable_safety(
                deformables,
                include_material_metrics=not (
                    _softmimicgen_task
                    or bench_dr_anmar_suture_enabled
                ),
            )
            suture_sample_time = time.monotonic()
            suture_dt_s = max(
                0.0,
                suture_sample_time - suture_last_sample_time[0],
            )
            suture_sample_period_s = max(
                float(suture_profile["runtime_detection"]["sample_period_s"]),
                float(procedure.get("suture_telemetry_period_s", 0.0))
                if not is_recording
                else 0.0,
            )
            if (
                dr_anmar_needle_enabled
                and suture_dt_s >= suture_sample_period_s
            ):
                live_suture_positions = suture_segment_positions()
                adjacent_suture_distances = np.linalg.norm(
                    np.diff(live_suture_positions, axis=0),
                    axis=1,
                )
                suture_axis_span = np.ptp(
                    live_suture_positions,
                    axis=0,
                )
                interface_position = suture_body_position(
                    suture_interface_view
                )
                latest_suture_telemetry = {
                    "schema": "dr.anmar.native-suture-telemetry.v1",
                    "profile_id": suture_profile["id"],
                    "physics_authority": "OpenUSD_PhysX",
                    "segment_count": int(live_suture_positions.shape[0]),
                    "native_state_finite": bool(
                        np.isfinite(live_suture_positions).all()
                    ),
                    "material_history_controller": False,
                    "episode_domain": dict(suture_runtime_domain_state[0]),
                    "sample_period_s": suture_dt_s,
                    "evidence_source": (
                        "native_physx_tensor_poses"
                    ),
                    "axis_span_m": [
                        round(float(value), 6)
                        for value in suture_axis_span
                    ],
                    "path_length_m": round(
                        float(adjacent_suture_distances.sum()),
                        6,
                    ),
                    "maximum_adjacent_segment_gap_m": round(
                        float(adjacent_suture_distances.max()),
                        6,
                    ),
                    "needle_interface_gap_m": (
                        round(
                            float(
                                np.linalg.norm(
                                    interface_position
                                    - live_suture_positions[0]
                                )
                            ),
                            6,
                        )
                        if interface_position is not None
                        else None
                    ),
                }
                suture_last_sample_time[0] = suture_sample_time
            elif (
                bench_dr_anmar_suture_enabled
                and interactive_deformable is not None
                and "dr_anmar_threaded_needle" in objects
                and suture_dt_s >= suture_sample_period_s
            ):
                # Observe the released SoftMimicGen deformable directly. The
                # nodal cloud is intentionally not treated as an ordered
                # centerline or a clinical material metric.
                nodal_value = interactive_deformable.data.nodal_pos_w
                nodal_positions = (
                    getattr(nodal_value, "torch", nodal_value)[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                default_value = interactive_deformable.data.default_nodal_state_w
                default_positions = (
                    getattr(default_value, "torch", default_value)[0, :, :3]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                endpoint_mask = (
                    default_positions[:, 0]
                    >= float(default_positions[:, 0].max()) - 0.0002
                )
                endpoint_position = nodal_positions[endpoint_mask].mean(axis=0)
                needle_state = objects["dr_anmar_threaded_needle"].data
                needle_position = (
                    needle_state.root_pos_w[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                needle_quaternion = (
                    needle_state.root_quat_w[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                needle_quaternion /= np.linalg.norm(needle_quaternion)
                quaternion_vector = needle_quaternion[1:]
                doubled_cross = 2.0 * np.cross(
                    quaternion_vector,
                    dr_anmar_swage_anchor_local,
                )
                rotated_anchor = (
                    dr_anmar_swage_anchor_local
                    + needle_quaternion[0] * doubled_cross
                    + np.cross(quaternion_vector, doubled_cross)
                )
                swage_position = needle_position + rotated_anchor
                latest_suture_telemetry = {
                    "schema": "dr.anmar.native-suture-telemetry.v1",
                    "profile_id": "nvidia-softmimicgen-rope-0.8mm",
                    "physics_authority": "NVIDIA_SoftMimicGen_PhysX",
                    "native_state_finite": bool(
                        np.isfinite(nodal_positions).all()
                    ),
                    "material_history_controller": False,
                    "sample_period_s": suture_dt_s,
                    "evidence_source": "native_deformable_nodal_positions",
                    "attachment": "PhysxAutoAttachmentAPI",
                    "rendered_diameter_m": 0.0008,
                    "physical_diameter_m": 0.0008,
                    "node_count": int(nodal_positions.shape[0]),
                    "terminal_surface_nodes": int(endpoint_mask.sum()),
                    "axis_span_m": [
                        round(float(value), 6)
                        for value in np.ptp(nodal_positions, axis=0)
                    ],
                    "swage_endpoint_gap_m": round(
                        float(
                            np.linalg.norm(
                                endpoint_position - swage_position
                            )
                        ),
                        6,
                    ),
                }
                suture_last_sample_time[0] = suture_sample_time
            max_force_value = max(latest_contact_forces.values(), default=None)
            max_displacement = max(
                (value for key, value in latest_deformable_safety.items() if key.endswith("_max_tissue_displacement_m")),
                default=None,
            )
            max_deformation = max(
                (value for key, value in latest_deformable_safety.items() if key.endswith("_max_deformation_gradient_proxy")),
                default=None,
            )
            max_stress = max(
                (value for key, value in latest_deformable_safety.items() if key.endswith("_max_tissue_stress_pa")),
                default=None,
            )
            with state.lock:
                state.max_contact_force_n = max_force_value
                state.max_tissue_displacement_m = max_displacement
                state.max_tissue_deformation_proxy = max_deformation
                state.max_tissue_stress_pa = max_stress
                state.native_telemetry["dr_anmar_suture"] = dict(
                    latest_suture_telemetry
                )
            last_safety_sample_time = loop_started
        if is_recording:
            with state.lock:
                gaze_uv = state.gaze_uv
                gaze_valid = state.gaze_valid
                gaze_source = state.gaze_source
                operator_source = state.operator_input_source
                procedure_phase = state.procedure_phase
                procedure_event_code = state.procedure_event_code
                procedure_event_sequence = state.procedure_event_sequence
                native_grasp_contact_state = list(state.native_grasp_contact_active)
                tool_object_distances = list(state.tool_to_object_distance_m)
                gripper_state = list(state.grippers_open)
                hand_teleop_frame = state.hand_teleop.snapshot()
                camera_valid_depth_fraction = state.camera_valid_depth_fraction
                camera_foreground_fraction = state.camera_foreground_fraction
                camera_mean_luminance = state.camera_mean_luminance
                state.procedure_event_code = 0
            frame = {
                "time_s": np.array(time.monotonic() - demo_started_monotonic, dtype=np.float64),
                "actions": (
                    native_policy_action_np.copy()
                    if native_policy_action_np is not None
                    else action_np.copy()
                ),
                "environment_reward": np.array(environment_reward, dtype=np.float32),
                "environment_terminated": np.array(environment_terminated, dtype=np.bool_),
                "environment_truncated": np.array(environment_truncated, dtype=np.bool_),
                "environment_success": np.array(environment_success, dtype=np.float32),
                "operator_gaze_uv": np.asarray(gaze_uv, dtype=np.float32),
                "operator_gaze_valid": np.array(gaze_valid, dtype=np.bool_),
                "operator_gaze_source_code": np.array(
                    {"none": 0, "pointer_attention_proxy": 1, "external_eye_tracker": 2, "xr_eye_tracking": 3}.get(
                        gaze_source, 0
                    ),
                    dtype=np.int16,
                ),
                "operator_input_source_code": np.array(OPERATOR_INPUT_SOURCES.get(operator_source, 0), dtype=np.int16),
                "webcam_hand_tracked": np.asarray(
                    [item["tracked"] for item in hand_teleop_frame["arms"]],
                    dtype=np.bool_,
                ),
                "webcam_motion_engaged": np.asarray(
                    [item["motion_engaged"] for item in hand_teleop_frame["arms"]],
                    dtype=np.bool_,
                ),
                "webcam_hand_confidence": np.asarray(
                    [item["confidence"] for item in hand_teleop_frame["arms"]],
                    dtype=np.float32,
                ),
                "webcam_hand_target_offsets": np.asarray(
                    [item["target_offset"] for item in hand_teleop_frame["arms"]],
                    dtype=np.float32,
                ),
                "webcam_hand_consumed_offsets": np.asarray(
                    [item["consumed_offset"] for item in hand_teleop_frame["arms"]],
                    dtype=np.float32,
                ),
                "webcam_gripper_aperture": np.asarray(
                    [item["aperture_normalized"] for item in hand_teleop_frame["arms"]],
                    dtype=np.float32,
                ),
                "procedure_phase_code": np.array(PROCEDURE_PHASES.get(procedure_phase, 0), dtype=np.int16),
                "procedure_event_code": np.array(procedure_event_code, dtype=np.int16),
                "procedure_event_sequence": np.array(procedure_event_sequence, dtype=np.int64),
                "task_grasp_state_code": np.asarray(
                    [
                        2 if native_grasp_contact_state[arm] else 1 if not gripper_state[arm] else 0
                        for arm in range(state.arms)
                    ],
                    dtype=np.int16,
                ),
                "tool_to_object_distance_m": np.asarray(
                    [value if value is not None else np.nan for value in tool_object_distances], dtype=np.float32
                ),
                "camera_valid_depth_fraction": np.array(
                    camera_valid_depth_fraction if camera_valid_depth_fraction is not None else np.nan, dtype=np.float32
                ),
                "camera_semantic_foreground_fraction": np.array(
                    camera_foreground_fraction if camera_foreground_fraction is not None else np.nan, dtype=np.float32
                ),
                "camera_mean_luminance": np.array(
                    camera_mean_luminance if camera_mean_luminance is not None else np.nan, dtype=np.float32
                ),
                "anatomy_showcase_position_w": anatomy_showcase_position_w.copy(),
                "anatomy_showcase_quaternion_w": anatomy_showcase_quaternion_w.copy(),
            }
            if native_policy_action_np is not None and native_joint_targets_np is not None:
                frame["cartesian_actions"] = action_np.copy()
                frame["resolved_joint_targets"] = native_joint_targets_np.copy()
            for name, robot in robots.items():
                frame[f"{name}_joint_positions"] = robot.data.joint_pos[0].detach().cpu().numpy().copy()
                frame[f"{name}_joint_velocities"] = robot.data.joint_vel[0].detach().cpu().numpy().copy()
                frame[f"{name}_root_pose_w"] = np.concatenate(
                    (
                        robot.data.root_pos_w[0].detach().cpu().numpy(),
                        robot.data.root_quat_w[0].detach().cpu().numpy(),
                    )
                ).astype(np.float32)
                frame[f"{name}_root_velocity_w"] = robot.data.root_vel_w[0].detach().cpu().numpy().astype(np.float32)
                frame[f"{name}_body_positions_w"] = robot.data.body_pos_w[0].detach().cpu().numpy().copy()
                frame[f"{name}_body_quaternions_w"] = robot.data.body_quat_w[0].detach().cpu().numpy().copy()
                applied_torque = getattr(robot.data, "applied_torque", None)
                if applied_torque is not None:
                    frame[f"{name}_applied_joint_torque"] = applied_torque[0].detach().cpu().numpy().copy()
                computed_torque = getattr(robot.data, "computed_torque", None)
                if computed_torque is not None:
                    frame[f"{name}_computed_joint_torque"] = computed_torque[0].detach().cpu().numpy().copy()
            for name, rigid_object in objects.items():
                frame[f"{name}_position_w"] = rigid_object.data.root_pos_w[0].detach().cpu().numpy().copy()
                frame[f"{name}_quaternion_w"] = rigid_object.data.root_quat_w[0].detach().cpu().numpy().copy()
            if current_native_centroid is not None:
                frame["native_tissue_centroid_w"] = current_native_centroid.copy()
            if hemostasis_clip_pose_w is not None:
                frame["dr_anmar_vascular_clip_pose_w"] = (
                    hemostasis_clip_pose_w.copy()
                )
            for key, value in latest_contact_forces.items():
                frame[key] = np.array(value, dtype=np.float32)
            for key, value in latest_deformable_safety.items():
                frame[key] = np.array(value, dtype=np.float32)
            frame["dr_anmar_suture_minimum_break_force_n"] = np.array(
                latest_suture_telemetry.get("minimum_break_force_n", np.nan),
                dtype=np.float32,
            )
            frame["dr_anmar_suture_maximum_observed_strain"] = np.array(
                latest_suture_telemetry.get("maximum_observed_strain", np.nan),
                dtype=np.float32,
            )
            frame["dr_anmar_suture_failed_joint_count"] = np.array(
                latest_suture_telemetry.get("failed_joint_count", 0),
                dtype=np.int32,
            )
            frame["dr_anmar_suture_compacted_knot_joint_count"] = np.array(
                latest_suture_telemetry.get("compacted_knot_joint_count", 0),
                dtype=np.int32,
            )
            if capture_spool is None:
                with state.lock:
                    state.record_request = "stop"
                    state.coaching_cue = "Recording storage was unavailable; saving stopped safely."
                continue
            capture_spool.append_control(frame)
            with state.lock:
                state.recorded_frames = capture_spool.control_count
                state.recorded_bytes_estimate = capture_spool.payload_bytes
                state.recording_queue_depth = capture_spool.queue_depth
                state.recording_buffered_frames = capture_spool.buffered_frames
                if (
                    capture_spool.control_count >= MAX_DEMO_FRAMES
                    or loop_started - demo_started_monotonic >= MAX_DEMO_SECONDS
                    or capture_spool.payload_bytes >= MAX_DEMO_BYTES
                ):
                    state.record_request = "stop"
                    state.coaching_cue = (
                        "The recording reached its configured safety limit and is being saved automatically."
                    )

        now = time.monotonic()
        fps_steps += 1
        frame_interval = 0.04 if interactive_active else 0.20
        if camera.data.output.get("rgb") is not None and now - last_frame_time >= frame_interval:
            camera_extract_started = time.perf_counter()
            camera_arrays: dict[str, np.ndarray] = {}
            dropout_profile = SCENARIO_NATIVE_PROFILES.get(scenario_id, {})
            dropout_period = int(dropout_profile.get("dropout_period_frames", 0))
            dropout_frames = int(dropout_profile.get("dropout_frames", 0))
            dropout_active = bool(
                dropout_period and (frame_count + scenario_seed % dropout_period) % dropout_period < dropout_frames
            )
            with state.lock:
                requested_cameras = {
                    name for name, count in state.camera_subscribers.items() if count > 0
                }
                requested_cameras.update(
                    name
                    for name, last_seen in state.camera_poll_last_seen_by_name.items()
                    if now - last_seen < 1.0
                )
            if shared_camera_renderer:
                # Every logical view is backed by the same physical RTX
                # render product. Encode only its current mount; duplicating
                # the same GPU tensor under multiple names wastes CPU/JPEG
                # work and can publish a wrist view as the endoscope during
                # a view switch.
                requested_cameras = {selected_active_camera}
            else:
                requested_cameras.add("endoscope_left")
            for camera_name in requested_cameras:
                sensor_camera = camera_sources.get(camera_name)
                if sensor_camera is None:
                    continue
                camera_output = sensor_camera.data.output.get("rgb")
                if camera_output is not None:
                    camera_arrays[camera_name] = rgb_tensor_to_array(camera_output[0])
            jpeg_encoder.submit(camera_arrays, scenario_id, dropout_active, now)
            camera_rgb = camera.data.output["rgb"][0]
            if is_recording and now - last_vision_sample_time >= 0.20:
                observation = rgb_tensor_to_image(camera_rgb, scenario_id, dropout_active).resize((360, 240), Image.Resampling.BILINEAR)
                vision_frame = {
                    "time_s": np.array(now - demo_started_monotonic, dtype=np.float64),
                    "rgb": np.asarray(observation, dtype=np.uint8),
                    "sensor_dropout_active": np.array(dropout_active, dtype=np.bool_),
                }
                vision_frame["mean_luminance"] = np.array(np.asarray(observation, dtype=np.float32).mean() / 255.0, dtype=np.float32)
                depth_tensor = camera.data.output.get("distance_to_image_plane")
                if depth_tensor is not None:
                    depth = depth_tensor[0].detach().cpu().numpy().astype(np.float32)
                    depth = np.squeeze(depth)
                    depth = np.nan_to_num(depth, nan=0.0, posinf=2.0, neginf=0.0)
                    if dropout_active:
                        depth.fill(0.0)
                    vision_frame["valid_depth_fraction"] = np.array(np.mean(depth > 0.0), dtype=np.float32)
                    if state.camera_intrinsics is None:
                        try:
                            with state.lock:
                                state.camera_intrinsics = (
                                    camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float).tolist()
                                )
                        except (AttributeError, RuntimeError, IndexError):
                            pass
                    intrinsics = np.asarray(state.camera_intrinsics or [], dtype=np.float32)
                    if intrinsics.shape == (3, 3):
                        vision_frame["point_cloud_camera_m"] = depth_to_point_cloud(depth, intrinsics)
                    depth_image = Image.fromarray(depth).resize((360, 240), Image.Resampling.BILINEAR)
                    vision_frame["depth_m"] = np.asarray(depth_image, dtype=np.float32)
                semantic_tensor = camera.data.output.get("semantic_segmentation")
                if semantic_tensor is not None:
                    semantic = np.squeeze(semantic_tensor[0].detach().cpu().numpy()).astype(np.int32)
                    if dropout_active:
                        semantic.fill(0)
                    semantic_image = Image.fromarray(semantic, mode="I").resize((360, 240), Image.Resampling.NEAREST)
                    vision_frame["semantic_id"] = np.asarray(semantic_image, dtype=np.uint32)
                    vision_frame["semantic_foreground_fraction"] = np.array(np.mean(semantic != 0), dtype=np.float32)
                    if not state.semantic_labels:
                        with state.lock:
                            state.semantic_labels = camera_semantic_labels(camera)
                if not shared_camera_renderer:
                    for camera_name in (
                        "endoscope_right",
                        "wrist_1",
                        "wrist_2",
                    ):
                        sensor_camera = camera_sources.get(camera_name)
                        sensor_rgb = (
                            sensor_camera.data.output.get("rgb")
                            if sensor_camera is not None
                            else None
                        )
                        if sensor_rgb is not None:
                            sensor_image = rgb_tensor_to_image(
                                sensor_rgb[0],
                                scenario_id,
                                dropout_active,
                            ).resize(
                                (360, 240),
                                Image.Resampling.BILINEAR,
                            )
                            vision_frame[f"{camera_name}_rgb"] = np.asarray(
                                sensor_image,
                                dtype=np.uint8,
                            )
                if capture_spool is not None:
                    capture_spool.append_vision(vision_frame)
                with state.lock:
                    state.camera_mean_luminance = float(vision_frame["mean_luminance"])
                    if "valid_depth_fraction" in vision_frame:
                        state.camera_valid_depth_fraction = float(vision_frame["valid_depth_fraction"])
                    if "semantic_foreground_fraction" in vision_frame:
                        state.camera_foreground_fraction = float(vision_frame["semantic_foreground_fraction"])
                with state.lock:
                    state.recorded_bytes_estimate = capture_spool.payload_bytes if capture_spool is not None else 0
                    state.recording_queue_depth = capture_spool.queue_depth if capture_spool is not None else 0
                    state.recording_buffered_frames = capture_spool.buffered_frames if capture_spool is not None else 0
                    if capture_spool is not None and capture_spool.payload_bytes >= MAX_DEMO_BYTES:
                        state.record_request = "stop"
                        state.coaching_cue = "The recording reached its memory budget and is being saved automatically."
                last_vision_sample_time = now
            frame_count += 1
            last_frame_time = now
            with state.lock:
                state.performance_timings_ms["camera_extract_submit"] = round(
                    (time.perf_counter() - camera_extract_started) * 1000.0,
                    3,
                )
        if now - last_fps_time >= 1.0:
            with state.lock:
                state.sim_fps = fps_steps / (now - last_fps_time)
                state.sim_step += fps_steps
            fps_steps = 0
            last_fps_time = now

        target_step = 0.02 if interactive_active else 0.50
        sleep_for = target_step - (time.monotonic() - loop_started)
        if sleep_for > 0:
            state.wake_event.wait(sleep_for)
            state.wake_event.clear()

    if state.recording:
        try:
            name = save_demo(state, capture_spool, demo_started_at) if capture_spool is not None else None
            if name:
                state.last_demo = name
        except Exception:
            traceback.print_exc()
        finally:
            if capture_spool is not None:
                capture_spool.abort()
    server.should_exit = True
    server_thread.join(timeout=5.0)
    jpeg_encoder.close()
    if upstream_expert_handler is not None:
        upstream_expert_handler.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
