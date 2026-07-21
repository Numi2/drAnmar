# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Browser-operated, simulation-only Dr.Anmar surgical workstation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import platform
import signal
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

from dr_anmar_procedures import PROCEDURES_BY_ID
from dr_anmar_physics_authority import load_physics_authority
from dr_anmar_native_rooms import resolve_native_room


DATA_ROOT = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()


def positive_environment_number(name: str, default: float, minimum: float) -> float:
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


MAX_DEMO_FRAMES = int(positive_environment_number("DR_ANMAR_MAX_DEMO_FRAMES", 60_000, 1_000))
MAX_DEMO_SECONDS = positive_environment_number("DR_ANMAR_MAX_DEMO_SECONDS", 300.0, 30.0)
MAX_DEMO_BYTES = int(positive_environment_number("DR_ANMAR_MAX_DEMO_BYTES", 1_500_000_000, 50_000_000))
SENSOR_PROFILES = {"efficient", "stereo", "research"}
EXTERNAL_OPERATOR_SENSORS_ENABLED = os.environ.get("DR_ANMAR_ENABLE_EXTERNAL_OPERATOR_SENSORS", "0") == "1"
STUDY_ID = os.environ.get("DR_ANMAR_STUDY_ID", "").strip()
CONSENT_PROTOCOL = os.environ.get("DR_ANMAR_CONSENT_PROTOCOL", "").strip()

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
    "--sensor_profile",
    choices=sorted(SENSOR_PROFILES),
    default=os.environ.get("DR_ANMAR_SENSOR_PROFILE", "research"),
    help="efficient=left RGB, stereo=left RGB-D+right RGB, research=stereo plus wrist cameras",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Refuse unsupported physics before Isaac Sim claims the GPU.  This is the
# same single capability decision used by the hub, so direct CLI launches
# cannot bypass the native-solver boundary.
_physics_authority = load_physics_authority()
_requested_procedure = PROCEDURES_BY_ID.get(args_cli.procedure) if args_cli.procedure else None
_native_room = resolve_native_room(args_cli.procedure) if args_cli.procedure else None
_effective_backend = (
    str(_native_room["backend"])
    if _native_room and _native_room.get("available")
    else None
)
_physics_runtime = _physics_authority.runtime_payload(
    runtime_family="isaac-sim-5.1-stable",
    effective_backend=_effective_backend,
)
if args_cli.procedure:
    if _requested_procedure is None:
        parser.error(f"Unknown Dr.Anmar procedure room: {args_cli.procedure}")
    _physics_readiness = _physics_authority.procedure_readiness(_requested_procedure, _physics_runtime)
    if not _physics_readiness["ready"]:
        parser.error(_physics_readiness["reason"])

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab_tasks.utils import parse_env_cfg

import orbit.surgical.tasks  # noqa: F401

from dr_anmar_expert import EXPERT_CONTROLLER_VERSION, EXPERT_PHASES, ExpertDemonstrationController
from dr_anmar_operator import ACCESS_COOKIE, OPERATOR_HEADER, OperatorLease, access_is_authorized


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
PROCEDURE_EVENTS = {"none": 0, "target_visible": 1, "contact": 2, "grasp": 3, "task_complete": 4, "handoff": 5, "safety_review": 6}
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
    .view{position:relative;overflow:hidden;background:#020608;display:flex;align-items:center;justify-content:center}.view img{width:100%;height:100%;object-fit:contain}.view-toolbar{padding:9px;border:1px solid #294956;border-radius:10px;background:#07151c}.view-toolbar-row{display:grid;grid-template-columns:62px 1fr;align-items:center;gap:7px}.view-toolbar-row+.view-toolbar-row{margin-top:7px}.view-toolbar-label{color:#6f909b;font:800 9px/1 ui-monospace,SFMono-Regular,Menlo;letter-spacing:.1em;text-transform:uppercase}.camera-tabs,.view-presets{display:grid;gap:5px}.camera-tabs{grid-template-columns:repeat(4,1fr)}.view-presets{grid-template-columns:repeat(7,1fr)}.camera-tabs button,.view-presets button{min-height:32px;padding:0 5px;font-size:9px}.camera-tabs button.active,.view-presets button.active{background:var(--cyan);border-color:var(--cyan);color:#031014}.header-camera-toolbar{min-width:0;flex:1;display:flex;align-items:center;gap:10px;padding:0;border:0;border-radius:0;background:transparent}.header-camera-toolbar .view-toolbar-row{display:flex;min-width:0;gap:6px}.header-camera-toolbar .view-toolbar-row+.view-toolbar-row{margin-top:0}.header-camera-toolbar .camera-tabs,.header-camera-toolbar .view-presets{display:flex;min-width:0;gap:4px}.header-camera-toolbar button{min-width:56px;min-height:34px;padding:0 6px}.header-camera-toolbar .view-presets button{min-width:64px}.gaze-cursor{position:absolute;width:18px;height:18px;border:1px solid #fff;border-radius:50%;translate:-50% -50%;pointer-events:none;opacity:0;box-shadow:0 0 0 3px #2cd2e855}.view.gaze-on .gaze-cursor{opacity:.85}
    .aim-reticle{position:absolute;left:50%;top:50%;width:28px;height:28px;translate:-50% -50%;pointer-events:none;opacity:.2}.aim-reticle:before,.aim-reticle:after{content:"";position:absolute;background:#dffcff}.aim-reticle:before{left:0;right:0;top:13px;height:1px}.aim-reticle:after{top:0;bottom:0;left:13px;width:1px}.proximity{margin:0 0 10px;padding:9px 11px;border:1px solid #294956;border-radius:8px;background:#061219;color:#9fc0c9;font:10px/1.45 ui-monospace,SFMono-Regular,Menlo}.proximity b{display:inline;color:var(--ink);font-size:10px;margin-right:7px}.proximity.near{border-color:#7a693d}.proximity.held{border-color:#34715f;color:var(--green)}.proximity.guard{border-color:#2c7180;color:var(--cyan)}.proximity.puncture{border-color:#78483e;color:#ffb09e}
    .mechanics-details{margin:0 0 12px;border:1px solid #24404d;border-radius:8px;background:#07151c}.mechanics-details summary{cursor:pointer;padding:9px 11px;color:#8ba8b2;font-size:10px;font-weight:800;letter-spacing:.08em;list-style:none}.mechanics-details summary::-webkit-details-marker{display:none}.mechanics-details summary:after{content:"+";float:right;color:var(--cyan)}.mechanics-details[open] summary:after{content:"−"}.procedure-sensor{margin:0 9px 9px;padding:10px;border-top:1px solid #23434f;font:9px/1.5 ui-monospace,SFMono-Regular,Menlo;color:#9fb9c1}.procedure-sensor.hidden{display:none}.procedure-sensor b{display:block;color:var(--cyan);font-size:10px;letter-spacing:.08em;margin-bottom:3px}.procedure-sensor strong{color:#efffff}.procedure-sensor .ok{color:var(--green)}.procedure-sensor .warn{color:#ffd978}
    .recflag{display:none;position:absolute;right:18px;top:18px;color:#fff;background:#c91f2f;padding:8px 12px;border-radius:99px;font-size:12px;font-weight:900;letter-spacing:.08em}.recflag.on{display:block}
    aside{overflow:auto;padding:17px;background:var(--panel);border-left:1px solid var(--line)}
    h2{font-size:12px;letter-spacing:.14em;color:#a9c1ca;margin:3px 0 11px;text-transform:uppercase}.card{border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:13px;background:#0a171e}
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.grid.two{grid-template-columns:repeat(2,1fr)}
    button{min-height:42px;border:1px solid #315462;border-radius:7px;background:#10252e;color:var(--ink);font-weight:750;cursor:pointer;touch-action:manipulation;user-select:none;-webkit-user-select:none}button:hover{border-color:var(--cyan);background:#153540}button:active{transform:translateY(1px);background:var(--cyan2)}
    button.primary{background:var(--cyan);border-color:var(--cyan);color:#041014}button.danger{background:#31171c;border-color:#74414a;color:#ffabb2}button.stop{grid-column:1/-1;background:#27323a;border-color:#5f727c}
    .speedbar{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:11px}.speedbar button{min-height:35px;font-size:11px}.speedbar button.active{background:var(--cyan);border-color:var(--cyan);color:#041014}.dpad{display:grid;grid-template-columns:repeat(3,1fr);grid-template-areas:"blank up blank2" "left stop right" "blank3 down blank4";gap:6px}.dpad .up{grid-area:up}.dpad .left{grid-area:left}.dpad .stop-center{grid-area:stop;min-height:54px;background:#26343b;border-color:#617681}.stop-center small{display:block;color:#9bb0b8;font-size:10px;margin-top:2px}.dpad .right{grid-area:right}.dpad .down{grid-area:down}.depthgrid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}.anglegrid{display:grid;grid-template-columns:1fr 1fr;gap:6px}.move-button{min-height:54px;touch-action:none;position:relative}.move-button small{display:block;color:#86a5af;font-size:10px;margin-top:2px}.move-button.held{background:var(--cyan);border-color:var(--cyan);color:#041014;box-shadow:0 0 16px #2cd2e855}.move-button.held small{color:#0a5260}.control-readout{display:flex;align-items:center;gap:7px;margin-top:10px;color:var(--muted);font-size:11px}.control-readout i{width:7px;height:7px;border-radius:50%;background:#536a73}.control-readout.moving{color:var(--green)}.control-readout.moving i{background:var(--green);box-shadow:0 0 10px #42e49b99}
    .hint{color:var(--muted);font-size:12px;margin-top:9px}.status{font:12px/1.65 ui-monospace,SFMono-Regular,Menlo;color:#bdd2d8;word-break:break-word}.status b{color:var(--green)}.hidden{display:none}.arm.active,.autonomy.active{background:var(--cyan);color:#041014;border-color:var(--cyan)}
    .procedure-title{font-size:15px;font-weight:850}.procedure-objective{color:#b9ccd2;font-size:11px;margin:6px 0 10px}.procedure-progress{height:4px;background:#19313b;margin:8px 0}.procedure-progress i{display:block;height:100%;background:var(--cyan);width:0}.procedure-step{display:grid;grid-template-columns:21px 1fr;gap:7px;padding:6px 0;border-top:1px solid #19313b;color:#738d96;font-size:10px}.procedure-step b{color:#9eb5bd}.procedure-step.complete b{color:var(--green)}.procedure-step.active b{color:var(--cyan)}.procedure-step span:first-child{font:10px ui-monospace,monospace}.fidelity-note{margin-top:8px;padding:7px;border-left:2px solid #f0b94e;background:#201a0d;color:#d8c18c;font-size:9px}
    .supervision{border-color:#356475;background:linear-gradient(135deg,#0d2731,#09171e)}.supervision-state{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:9px}.supervision-state b{color:var(--cyan)}.cue{min-height:32px;margin-top:9px;padding:8px;border-left:2px solid var(--cyan);background:#061219;color:#9fc0c9;font-size:11px}.take-control{width:100%;margin-top:8px;background:#ffd978;color:#251b02;border-color:#ffd978}
    .safety-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.safety-metric{padding:8px;background:#061219;border:1px solid #1c3742}.safety-metric b{display:block;color:var(--green);font:15px ui-monospace,monospace}.safety-metric span{color:var(--muted);font-size:9px}.ghost-state{margin-top:8px;color:var(--muted);font-size:11px}.ghost-state.on{color:var(--green)}
    .control-dock{position:relative;margin:0 0 12px;padding:32px 8px 8px;border:2px solid var(--cyan);border-radius:11px;background:linear-gradient(145deg,#0d2630,#08151c 68%);box-shadow:0 0 0 1px #2cd2e829,0 0 24px #2cd2e820}.control-dock:before{content:"ROBOT NAVIGATION + INSTRUMENT CONTROL";position:absolute;left:10px;top:10px;color:var(--cyan);font:900 9px/1 ui-sans-serif,system-ui;letter-spacing:.1em}.control-dock:after{content:"LIVE";position:absolute;right:10px;top:7px;padding:3px 6px;border-radius:99px;background:#123a32;color:var(--green);font:900 8px/1 ui-monospace,monospace;letter-spacing:.1em}.control-dock h2{color:#d7f9fc;margin:0 0 5px;font-size:9px;letter-spacing:.1em}.control-dock .card{border-color:#356675;background:#07151c;margin:0;padding:6px;border-radius:7px}.control-dock .move-button{min-height:38px;border-width:1px;border-color:#3d6d7b;font-size:10px;line-height:1.1}.control-dock .stop-center{min-height:38px;border-width:1px;font-size:9px;line-height:1.1}.control-dock .move-button small,.control-dock .stop-center small{font-size:8px;margin-top:1px}.control-dock #gripperPanel .primary{box-shadow:0 0 12px #2cd2e82e}.control-dock .hint{display:none}.control-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:7px;align-items:start}.control-section{min-width:0}.control-position{grid-column:1}.control-angle{grid-column:2}.control-speed{grid-column:1}.control-grippers{grid-column:2}.control-dock .dpad,.control-dock .anglegrid{gap:4px}.control-dock .depthgrid{gap:4px;margin-top:4px}.control-dock .control-readout{min-height:17px;margin-top:4px;font-size:8px}.control-dock .control-readout i{width:5px;height:5px}.control-dock .speedbar{gap:4px;margin:0}.control-dock .speedbar button{min-height:34px;padding:2px;font-size:9px}.control-dock .grid{gap:4px}.control-dock .grid button{min-height:34px;padding:3px;font-size:9px}.control-dock .grid button kbd,.control-dock .speedbar button kbd{height:16px;min-width:18px;padding:0 4px;font-size:8px}
    .expert-demo{margin:2px 0 14px;padding:13px;border:1px solid #557586;border-radius:12px;background:linear-gradient(145deg,#102a35,#07151d 72%);box-shadow:0 0 24px #2cd2e81c}.expert-head{display:flex;align-items:start;justify-content:space-between;gap:10px}.expert-head .eyebrow{color:var(--cyan);font:900 10px/1 ui-monospace,SFMono-Regular,Menlo;letter-spacing:.13em}.expert-head b{display:block;margin-top:5px;font-size:15px}.expert-status{padding:4px 7px;border:1px solid #365867;border-radius:99px;color:#a8c0c8;font:800 9px/1 ui-monospace,monospace;text-transform:uppercase}.expert-status.running{border-color:var(--green);color:var(--green)}.expert-status.paused{border-color:#ffd978;color:#ffd978}.expert-rail{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin:11px 0}.expert-phase{min-width:0;padding:7px 3px;border:1px solid #203f4b;background:#071219;color:#6f8b95;text-align:center;font:800 8px/1 ui-monospace,monospace;text-transform:uppercase}.expert-phase.complete{border-color:#2d725c;color:var(--green);background:#0a251f}.expert-phase.active{border-color:var(--cyan);color:#eaffff;background:#103a48;box-shadow:0 0 12px #2cd2e82b}.expert-instruction{min-height:42px;padding:8px;border-left:2px solid var(--cyan);background:#061219;color:#b7cbd1;font-size:10px}.expert-actions{display:grid;grid-template-columns:1.25fr 1fr 1fr;gap:6px;margin-top:9px}.expert-actions button{min-height:43px;font-size:10px}.expert-meta{display:flex;justify-content:space-between;gap:10px;margin-top:8px;color:#718d97;font:9px/1.4 ui-monospace,monospace}.expert-meta .ready{color:var(--green)}
    kbd{display:inline-flex;align-items:center;justify-content:center;min-width:22px;height:20px;padding:0 6px;border:1px solid #567482;border-bottom-width:2px;border-radius:5px;background:#09141a;color:#dffbff;font:800 10px/1 ui-monospace,SFMono-Regular,Menlo;white-space:nowrap}button kbd{pointer-events:none}.header-keyboard{min-height:32px;margin-left:4px;padding:0 10px;background:#10252e;color:#cfe7eb;font-size:11px}.header-keyboard kbd{margin-right:5px}.keyboard-quick{display:grid;grid-template-columns:.9fr 1.1fr;gap:5px;margin:0;padding:6px;border:1px solid #46788a;border-radius:8px;background:linear-gradient(135deg,#0b222c,#07151c)}.keyboard-quick-head{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;gap:8px;margin:0}.keyboard-quick-head b{color:var(--cyan);font-size:9px;letter-spacing:.1em}.keyboard-quick-head span{color:var(--muted);font-size:8px}.keyboard-input-display{display:flex;align-items:center;gap:5px;min-height:34px;margin:0;padding:4px 6px;border:1px solid #284d5b;border-radius:6px;background:#061219;color:#9eb7bf;font-size:9px}.keyboard-input-display kbd{min-width:38px;height:17px;font-size:8px;color:var(--green);border-color:#3b7a67}.keyboard-input-display.active{border-color:var(--green);box-shadow:inset 0 0 14px #42e49b16}.keyboard-input-display.active span{color:#e5ffff}.smart-action{width:100%;min-height:34px;margin:0;background:linear-gradient(90deg,var(--cyan),#5ee8ca);border-color:#8ff7f5;color:#031014;text-align:left;padding:4px 7px;box-shadow:0 0 14px #2cd2e829}.smart-action strong{display:block;font-size:10px}.smart-action strong kbd{height:17px;min-width:24px;padding:0 4px;font-size:8px}.smart-action small{display:block;overflow:hidden;color:#174851;font-size:8px;line-height:1.1;white-space:nowrap;text-overflow:ellipsis}.proximity{grid-column:1/-1;margin:0;padding:4px 6px;border-radius:6px;font-size:8px;line-height:1.2}.proximity b{font-size:8px;margin-right:5px}.combo-grid{grid-column:1/-1;display:grid;grid-template-columns:repeat(6,1fr);gap:3px}.combo-button{min-height:34px;padding:3px 2px;border-color:#315a69;background:#0c2029;font-size:8px;line-height:1.05}.combo-button kbd{display:inline-flex;width:auto;height:16px;min-width:17px;margin:0 2px 0 0;padding:0 3px;font-size:8px}.combo-button.held{background:var(--cyan);color:#031014;box-shadow:0 0 15px #2cd2e855}.combo-button.held kbd{border-color:#174851;background:#d9fbff;color:#031014}.modifier-row{grid-column:1/-1;display:flex;gap:3px;margin:0}.modifier-chip{flex:1;padding:3px;border:1px solid #244653;border-radius:5px;color:#89a8b3;font-size:8px;text-align:center}.modifier-chip kbd{height:16px;min-width:20px;padding:0 3px;font-size:8px}.modifier-chip.active{border-color:var(--green);color:var(--green);background:#0b2b25}.keyboard-coverage{grid-column:1/-1;margin:0;color:var(--green);font:8px/1.1 ui-monospace,SFMono-Regular,Menlo}.keyboard-coverage.bad{color:var(--red)}button.key-active,button.state-active{border-color:var(--green)!important;box-shadow:0 0 0 2px #42e49b66,0 0 22px #42e49b45!important;background:#174a42!important;color:#efffff!important}button.key-active kbd,button.state-active kbd{border-color:#9bffe0;background:#dcfff5;color:#09281f}.smart-action.key-active{background:linear-gradient(90deg,#8bffe0,#edff9d)!important;color:#041a13!important;transform:scale(1.015)}
    .keyboard-help{position:fixed;inset:0;z-index:50;display:grid;place-items:center;padding:24px;background:#02080dd9;backdrop-filter:blur(7px)}.keyboard-help.hidden{display:none}.keyboard-help-panel{width:min(940px,96vw);max-height:90vh;overflow:auto;border:1px solid #4c7c8d;border-radius:14px;background:#09171e;box-shadow:0 24px 90px #000;padding:18px}.keyboard-help-head{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:14px}.keyboard-help-head h1{margin:0;color:#e7fbfd;font-size:21px}.keyboard-help-head p{margin:3px 0 0;color:var(--muted);font-size:11px}.keyboard-help-head button{min-height:36px;padding:0 12px}.shortcut-columns{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.shortcut-group{padding:12px;border:1px solid #203e49;border-radius:9px;background:#071219}.shortcut-group h3{margin:0 0 8px;color:var(--cyan);font-size:11px;letter-spacing:.11em}.shortcut-line{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:5px 0;border-top:1px solid #152c35;color:#bdd1d7;font-size:10px}.shortcut-line:first-of-type{border-top:0}.shortcut-line span{text-align:right}.shortcut-group.wide{grid-column:span 3}.shortcut-group.wide .shortcut-list{display:grid;grid-template-columns:repeat(3,1fr);gap:0 14px}
    #toast{position:fixed;left:50%;bottom:20px;translate:-50% 20px;opacity:0;background:#e9f8fa;color:#061116;border-radius:8px;padding:10px 15px;font-weight:750;transition:.2s;pointer-events:none}#toast.show{opacity:1;translate:-50% 0}
    @media(max-width:1250px){.header-camera-toolbar .view-toolbar-label{display:none}.header-camera-toolbar button{min-width:48px;padding:0 4px}.header-camera-toolbar .view-presets button{min-width:54px}.header-camera-toolbar kbd{display:none}}
    @media(max-width:1100px){main{grid-template-columns:minmax(0,1fr) 430px}.shortcut-columns{grid-template-columns:repeat(2,1fr)}.shortcut-group.wide{grid-column:span 2}.shortcut-group.wide .shortcut-list{grid-template-columns:repeat(2,1fr)}.header-keyboard{display:none}}
    @media(max-width:880px){header{height:auto;min-height:64px;padding:7px 12px;flex-wrap:wrap}.header-camera-toolbar{order:3;flex-basis:100%;overflow-x:auto;padding-bottom:2px}.header-camera-toolbar .view-toolbar-label{display:block}.header-camera-toolbar kbd{display:inline-flex}.header-camera-toolbar button{min-width:62px}.header-camera-toolbar .view-presets button{min-width:70px}main{display:block;height:auto}.view{height:52vh}aside{border-left:0;border-top:1px solid var(--line)}}
  </style>
</head>
<body>
<header><div class="brand">DR.<span>ANMAR</span></div><section class="view-toolbar header-camera-toolbar" aria-label="Camera controls"><div class="view-toolbar-row"><span class="view-toolbar-label">Camera</span><div class="camera-tabs"><button class="active" data-camera="endoscope_left" data-shortcut="4" onclick="setCamera('endoscope_left',this)">Left <kbd>4</kbd></button><button data-camera="endoscope_right" data-shortcut="5" onclick="setCamera('endoscope_right',this)">Right <kbd>5</kbd></button><button data-camera="wrist_1" data-shortcut="6" onclick="setCamera('wrist_1',this)">Wrist 1 <kbd>6</kbd></button><button id="wrist2Tab" class="hidden" data-camera="wrist_2" data-shortcut="7" onclick="setCamera('wrist_2',this)">Wrist 2 <kbd>7</kbd></button></div></div><div class="view-toolbar-row"><span class="view-toolbar-label">Angle</span><div class="view-presets"><button class="active" data-view-mode="operative" data-shortcut="F1" onclick="setCameraView('operative',this)">Operative <kbd>F1</kbd></button><button data-view-mode="close" data-shortcut="F2" onclick="setCameraView('close',this)">Close <kbd>F2</kbd></button><button data-view-mode="overview" data-shortcut="F3" onclick="setCameraView('overview',this)">Wide <kbd>F3</kbd></button><button data-view-mode="overhead" data-shortcut="F4" onclick="setCameraView('overhead',this)">Overhead <kbd>F4</kbd></button><button data-view-mode="left_oblique" data-shortcut="F5" onclick="setCameraView('left_oblique',this)">Left angle <kbd>F5</kbd></button><button data-view-mode="right_oblique" data-shortcut="F6" onclick="setCameraView('right_oblique',this)">Right angle <kbd>F6</kbd></button><button data-view-mode="opposite" data-shortcut="F7" onclick="setCameraView('opposite',this)">Opposite <kbd>F7</kbd></button></div></div></section><button class="header-keyboard" data-shortcut="?" onclick="toggleKeyboardHelp()"><kbd>?</kbd> Keyboard map</button><div class="live"><i id="dot" class="dot"></i><span id="connection">Connecting…</span></div></header>
<main>
  <section id="cameraView" class="view"><img id="cameraImage" src="/video/endoscope_left" alt="Live simulated medical sensor view"><div id="recflag" class="recflag">● RECORDING</div><div id="gazeCursor" class="gaze-cursor"></div><div class="aim-reticle"></div></section>
  <aside>
    <section class="control-dock">
      <div class="keyboard-quick"><div class="keyboard-quick-head"><b>BIMANUAL GAME CONTROLS</b><span>Both hands move together · release = stop</span></div><div id="keyActionDisplay" class="keyboard-input-display" aria-live="polite"><kbd>READY</kbd><span>Keyboard control ready</span></div><button id="smartActionButton" class="smart-action" data-shortcut="F12" onclick="smartAction()"><strong><kbd>F12</kbd> Smart context action</strong><small id="smartActionLabel">Nudge toward the target</small></button><div id="proximity" class="proximity"><b>Next</b><span>Acquiring target…</span></div><div class="combo-grid">
        <button class="combo-button" data-combo-key="KeyZ" data-shortcut="Z"><kbd>Z</kbd>Orbit left</button><button class="combo-button" data-combo-key="KeyX" data-shortcut="X"><kbd>X</kbd>Orbit right</button><button class="combo-button" data-combo-key="KeyV" data-shortcut="V"><kbd>V</kbd>Drive needle</button>
        <button class="combo-button" data-combo-key="KeyB" data-shortcut="B"><kbd>B</kbd>Reverse needle</button><button class="combo-button" data-combo-key="KeyN" data-shortcut="N"><kbd>N</kbd>Lift + retract</button><button class="combo-button" data-combo-key="KeyF" data-shortcut="F"><kbd>F</kbd>Lower + approach</button>
      </div><div class="modifier-row"><div id="leftRotateModifier" class="modifier-chip"><kbd>L⇧</kbd> left wrist</div><div id="precisionModifier" class="modifier-chip"><kbd>⌥</kbd> pointer precision</div><div id="rightRotateModifier" class="modifier-chip"><kbd>R⇧</kbd> right wrist</div></div><div id="keyboardCoverage" class="keyboard-coverage">Auditing keyboard coverage…</div></div>
      <div id="armPanel" class="hidden"><h2>Pointer-control instrument</h2><div class="card"><div class="grid two"><button id="arm0" class="arm active" data-shortcut="[" onclick="setArm(0)">Instrument 1 <kbd>[</kbd></button><button id="arm1" class="arm" data-shortcut="]" onclick="setArm(1)">Instrument 2 <kbd>]</kbd></button></div><div class="hint">Keyboard movement does not require selection: each hand always owns its robot.</div></div></div>
      <div class="control-grid"><div class="control-section control-position"><h2>Tool position</h2><div class="card"><div class="dpad">
        <button class="move-button up" data-key="KeyW" data-shortcut="W" data-axis="2" data-direction="1">↑ Up<small>W / I</small></button>
        <button class="move-button left" data-key="KeyA" data-shortcut="A" data-axis="1" data-direction="1">← Left<small>A</small></button>
        <button class="stop-center" data-shortcut="Backspace" onclick="emergencyStop()">■ Stop both<small>Backspace / Esc</small></button>
        <button class="move-button right" data-key="KeyD" data-shortcut="D" data-axis="1" data-direction="-1">Right →<small>D</small></button>
        <button class="move-button down" data-key="KeyS" data-shortcut="S" data-axis="2" data-direction="-1">↓ Down<small>S / K</small></button>
      </div><div class="depthgrid"><button class="move-button" data-key="KeyQ" data-shortcut="Q" data-axis="0" data-direction="-1">Toward patient<small>Q / U</small></button><button class="move-button" data-key="KeyE" data-shortcut="E" data-axis="0" data-direction="1">Away from patient<small>E / O</small></button></div><div id="controlReadout" class="control-readout" aria-live="polite"><i></i><span>Ready · hold a control to move</span></div></div>
      </div>
      <div class="control-section control-angle"><h2>Tool angle</h2><div class="card"><div class="anglegrid">
        <button class="move-button" data-shortcut="⇧Q" data-axis="3" data-direction="-1">↶ Roll left<small>Shift + Q / U</small></button><button class="move-button" data-shortcut="⇧E" data-axis="3" data-direction="1">Roll right ↷<small>Shift + E / O</small></button>
        <button class="move-button" data-shortcut="⇧W" data-axis="4" data-direction="-1">Pitch up<small>Shift + W / I</small></button><button class="move-button" data-shortcut="⇧S" data-axis="4" data-direction="1">Pitch down<small>Shift + S / K</small></button>
        <button class="move-button" data-shortcut="⇧A" data-axis="5" data-direction="-1">← Yaw left<small>Shift + A / J</small></button><button class="move-button" data-shortcut="⇧D" data-axis="5" data-direction="1">Yaw right →<small>Shift + D / L</small></button>
      </div><div class="hint"><kbd>WASD + QE</kbd> left robot · <kbd>IJKL + UO</kbd> right robot · hold that hand’s <kbd>Shift</kbd> to rotate its wrist.</div></div></div>
      <div class="control-section control-speed"><h2>Pointer / gamepad speed</h2><div class="card"><div class="speedbar"><button data-speed="0.35" data-shortcut="," onclick="setSpeed(0.35,this)">Precision <kbd>,</kbd></button><button class="active" data-speed="1" data-shortcut="." onclick="setSpeed(1,this)">Normal <kbd>.</kbd></button><button data-speed="1.7" data-shortcut="/" onclick="setSpeed(1.7,this)">Fast <kbd>/</kbd></button></div><div class="hint">Keyboard speed is independent: <kbd>1/2/3</kbd> for the left robot and <kbd>8/9/0</kbd> for the right.</div></div></div>
      <div id="gripperPanel" class="control-section control-grippers"><h2>Grippers</h2><div class="card"><div class="grid two"><button id="gripOpenButton" data-shortcut="Space" onclick="toggleGrip(0)">Left robot <kbd>Space</kbd></button><button id="gripCloseButton" class="primary" data-shortcut="Enter" onclick="toggleGrip((latestStatus?.arms||1)>1?1:0)">Right robot <kbd>Enter</kbd></button></div><div class="hint">Each thumb owns its nearest gripper. <kbd>Backspace</kbd> or <kbd>Esc</kbd> stops both robots instantly.</div></div></div></div>
    </section>
    <section id="expertDemo" class="expert-demo"><div class="expert-head"><div><div class="eyebrow">EXECUTABLE TEACHING</div><b>Watch the robot perform this room</b></div><span id="expertStatus" class="expert-status">READY</span></div><div id="expertRail" class="expert-rail"></div><div id="expertInstruction" class="expert-instruction">The expert executes the full procedure in the live simulation. Pause at any phase, inspect the views and forces, or take control from the current pose.</div><div class="expert-actions"><button id="expertStart" class="primary" data-shortcut="F9" onclick="startExpert()">Watch expert <kbd>F9</kbd></button><button id="expertPause" data-shortcut="F10" onclick="toggleExpertPause()" disabled>Pause <kbd>F10</kbd></button><button id="expertTakeover" data-shortcut="Esc" onclick="takeControl()" disabled>Take control <kbd>Esc</kbd></button></div><div class="expert-meta"><span id="expertProgress">8 teachable phases</span><span id="expertReference">BC reference captured on completion</span></div></section>
    <h2>Procedure room</h2><div class="card"><div id="procedureTitle" class="procedure-title">Free practice</div><div id="procedureObjective" class="procedure-objective">Use the robot controls to explore the digital twin.</div><div class="procedure-progress"><i id="procedureProgress"></i></div><div id="procedureSteps"></div><div id="procedureTruth" class="fidelity-note hidden"></div></div><details class="mechanics-details"><summary>Live mechanics details</summary><div id="procedureSensor" class="procedure-sensor hidden"></div></details>
    <h2>Supervision</h2><div class="card supervision"><div class="supervision-state"><span>Autonomy level</span><b id="autonomyState">L0 · Manual</b></div><div class="grid two"><button id="manualMode" class="autonomy active" data-shortcut="M" onclick="setAutonomy('manual')">Manual <kbd>M</kbd></button><button id="guidedMode" class="autonomy" data-shortcut="G" onclick="setAutonomy('guided')">Guided <kbd>G</kbd></button></div><button class="take-control" data-shortcut="Esc" onclick="emergencyStop()">Take control now <kbd>Esc</kbd></button><div id="coachingCue" class="cue">You command every movement. Dr.Anmar records telemetry for coaching.</div></div>
    <h2>Expert path guide</h2><div class="card"><div class="grid two"><button class="primary" data-shortcut="H" onclick="referenceGhost(true)">Show clinician path <kbd>H</kbd></button><button data-shortcut="H" onclick="referenceGhost(false)">Hide path <kbd>H</kbd></button></div><div id="ghostState" class="ghost-state">Select a clinician reference in Skills Twin first.</div></div>
    <h2>Procedure annotation</h2><div class="card"><div class="grid"><button data-shortcut="⌥1" onclick="annotatePhase('approach')">Approach <kbd>⌥1</kbd></button><button data-shortcut="⌥2" onclick="annotatePhase('grasp')">Grasp <kbd>⌥2</kbd></button><button data-shortcut="⌥3" onclick="annotatePhase('manipulation')">Manipulate <kbd>⌥3</kbd></button><button data-shortcut="⌥4" onclick="annotatePhase('recovery')">Recovery <kbd>⌥4</kbd></button><button data-shortcut="⌥5" onclick="annotateEvent('task_complete')">Task event <kbd>⌥5</kbd></button><button data-shortcut="⌥6" onclick="annotateEvent('safety_review')">Safety event <kbd>⌥6</kbd></button></div><div class="hint">Phase labels and events are synchronized into the training trajectory.</div></div>
    <h2>Research safety monitor</h2><div class="card"><div class="safety-grid"><div class="safety-metric"><b id="forceMetric">—</b><span>CONTACT N</span></div><div class="safety-metric"><b id="deformMetric">—</b><span>TISSUE MM</span></div><div class="safety-metric"><b id="stressMetric">—</b><span>STRESS PA</span></div></div><div class="hint">Simulator signals only. Limits are engineering advisories, not clinical thresholds.</div></div>
    <h2>Demonstration</h2><div class="card"><div class="grid two"><button id="record" class="primary" data-shortcut="Y" onclick="recording(true)">Start recording <kbd>Y</kbd></button><button data-shortcut="T" onclick="recording(false)">Stop & save <kbd>T</kbd></button><button data-shortcut="R" onclick="replay()">Replay last <kbd>R</kbd></button><button data-shortcut="Delete" onclick="resetScene()">Reset scene <kbd>Delete</kbd></button></div><div class="hint" id="lastDemo">Actions, joints, RGB-D, segmentation, object state, and safety telemetry are saved together.</div></div>
    <h2>System</h2><div class="card status" id="status">Starting…</div>
  </aside>
</main>
<div id="keyboardHelp" class="keyboard-help hidden" role="dialog" aria-modal="true" aria-labelledby="keyboardHelpTitle"><div class="keyboard-help-panel"><div class="keyboard-help-head"><div><h1 id="keyboardHelpTitle">Two-hand surgical game controls</h1><p>Each hand permanently owns one robot. Hold keys to move both at once; releasing stops that motion. Backspace or Escape stops everything.</p></div><button data-shortcut="?" onclick="toggleKeyboardHelp(false)">Close <kbd>?</kbd></button></div><div class="shortcut-columns">
  <div class="shortcut-group"><h3>LEFT ROBOT · LEFT HAND</h3><div class="shortcut-line"><kbd>W / S</kbd><span>Up / down</span></div><div class="shortcut-line"><kbd>A / D</kbd><span>Left / right</span></div><div class="shortcut-line"><kbd>Q / E</kbd><span>Toward / away</span></div><div class="shortcut-line"><kbd>Left Shift + move</kbd><span>Rotate wrist</span></div><div class="shortcut-line"><kbd>Space</kbd><span>Toggle left gripper</span></div></div>
  <div class="shortcut-group"><h3>RIGHT ROBOT · RIGHT HAND</h3><div class="shortcut-line"><kbd>I / K</kbd><span>Up / down</span></div><div class="shortcut-line"><kbd>J / L</kbd><span>Left / right</span></div><div class="shortcut-line"><kbd>U / O</kbd><span>Toward / away</span></div><div class="shortcut-line"><kbd>Right Shift + move</kbd><span>Rotate wrist</span></div><div class="shortcut-line"><kbd>Enter</kbd><span>Toggle right gripper</span></div></div>
  <div class="shortcut-group"><h3>SPEED + SAFETY</h3><div class="shortcut-line"><kbd>1 / 2 / 3</kbd><span>Left precision / normal / fast</span></div><div class="shortcut-line"><kbd>8 / 9 / 0</kbd><span>Right precision / normal / fast</span></div><div class="shortcut-line"><kbd>Backspace</kbd><span>Stop both robots</span></div><div class="shortcut-line"><kbd>Esc</kbd><span>Stop + take control</span></div><div class="shortcut-line"><kbd>[ / ]</kbd><span>Select pointer-control robot</span></div></div>
  <div class="shortcut-group"><h3>BOUNDED COMBINED MOVES</h3><div class="shortcut-line"><kbd>Z / X</kbd><span>Orbit left / right</span></div><div class="shortcut-line"><kbd>V / B</kbd><span>Drive / reverse needle</span></div><div class="shortcut-line"><kbd>N / F</kbd><span>Lift / lower + approach</span></div><div class="shortcut-line"><kbd>F12</kbd><span>Context action</span></div><div class="shortcut-line"><kbd>Space / Enter</kbd><span>Independent grippers</span></div></div>
  <div class="shortcut-group"><h3>CAMERAS</h3><div class="shortcut-line"><kbd>4 / 5</kbd><span>Stereo left / right</span></div><div class="shortcut-line"><kbd>6 / 7</kbd><span>Wrist 1 / 2</span></div><div class="shortcut-line"><kbd>F1 / F2 / F3</kbd><span>Operative / close / wide</span></div><div class="shortcut-line"><kbd>F4 / F5 / F6 / F7</kbd><span>Overhead / left / right / opposite</span></div><div class="shortcut-line"><kbd>C / ⇧C</kbd><span>Next sensor / next angle</span></div></div>
  <div class="shortcut-group"><h3>EXPERT + SESSION</h3><div class="shortcut-line"><kbd>F9 / F10</kbd><span>Run / pause expert</span></div><div class="shortcut-line"><kbd>Y / T</kbd><span>Start / stop + save</span></div><div class="shortcut-line"><kbd>R / H</kbd><span>Replay / path guide</span></div><div class="shortcut-line"><kbd>M / G</kbd><span>Manual / guided</span></div><div class="shortcut-line"><kbd>Delete</kbd><span>Reset scene</span></div></div>
  <div class="shortcut-group wide"><h3>PROCEDURE ANNOTATIONS</h3><div class="shortcut-list"><div class="shortcut-line"><kbd>⌥1</kbd><span>Approach</span></div><div class="shortcut-line"><kbd>⌥2</kbd><span>Grasp</span></div><div class="shortcut-line"><kbd>⌥3</kbd><span>Manipulate</span></div><div class="shortcut-line"><kbd>⌥4</kbd><span>Recovery</span></div><div class="shortcut-line"><kbd>⌥5</kbd><span>Task event</span></div><div class="shortcut-line"><kbd>⌥6</kbd><span>Safety event</span></div></div></div>
</div></div></div><div id="toast"></div>
<script>
const operatorId=(()=>{const query=new URLSearchParams(location.search).get('operator');if(query){sessionStorage.setItem('drAnmarOperatorId',query);return query}let value=sessionStorage.getItem('drAnmarOperatorId');if(!value){const random=crypto.randomUUID?crypto.randomUUID():`${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;value=`browser-${random}`;sessionStorage.setItem('drAnmarOperatorId',value)}return value})();
const keyMap={KeyW:[2,1,'Up'],KeyS:[2,-1,'Down'],KeyA:[1,1,'Left'],KeyD:[1,-1,'Right'],KeyQ:[0,-1,'Toward'],KeyE:[0,1,'Away']};
const handKeyMaps=[{KeyW:[2,1,'Up'],KeyS:[2,-1,'Down'],KeyA:[1,1,'Left'],KeyD:[1,-1,'Right'],KeyQ:[0,-1,'Toward'],KeyE:[0,1,'Away']},{KeyI:[2,1,'Up'],KeyK:[2,-1,'Down'],KeyJ:[1,1,'Left'],KeyL:[1,-1,'Right'],KeyU:[0,-1,'Toward'],KeyO:[0,1,'Away']}];
const rotationKeyMaps=[{KeyW:[4,-1,'Pitch up'],KeyS:[4,1,'Pitch down'],KeyA:[5,-1,'Yaw left'],KeyD:[5,1,'Yaw right'],KeyQ:[3,-1,'Roll left'],KeyE:[3,1,'Roll right']},{KeyI:[4,-1,'Pitch up'],KeyK:[4,1,'Pitch down'],KeyJ:[5,-1,'Yaw left'],KeyL:[5,1,'Yaw right'],KeyU:[3,-1,'Roll left'],KeyO:[3,1,'Roll right']}];
const dualMovementCodes=new Set([...Object.keys(handKeyMaps[0]),...Object.keys(handKeyMaps[1])]);
const comboMap={KeyZ:{label:'Orbit left',values:[0,.72,0,0,0,-.72]},KeyX:{label:'Orbit right',values:[0,-.72,0,0,0,.72]},KeyV:{label:'Drive needle',values:[-.68,0,0,.68,0,0]},KeyB:{label:'Reverse needle',values:[.68,0,0,-.68,0,0]},KeyN:{label:'Lift + retract',values:[.68,0,.68,0,0,0]},KeyF:{label:'Lower + approach',values:[-.68,0,-.68,0,0,0]}};
function comboValues(code){return comboMap[code]?.values||Array(6).fill(0)}
let activeArm=0,driveSpeed=1,keyboardSpeeds=[1,1],driveInFlight=false,queuedDrive=null,driveWasActive=false,bimanualInFlight=false,queuedBimanual=null,bimanualWasActive=false,inputSource='keyboard_pointer',lastGazeSend=0,currentCamera='endoscope_left',currentViewMode='operative',lastGamepadGrip=false,lastGamepadCamera=false,latestStatus=null,macroPulseTimer=null,keyFlashTimer=null;
const heldKeys=new Set(),heldModifiers=new Set(),pointerMoves=new Map(),keyDownAt=new Map();
async function post(url,body={}){const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json','x-dr-anmar-operator':operatorId},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.detail||'Request failed');return data}
function toast(s){const e=document.getElementById('toast');e.textContent=s;e.classList.add('show');setTimeout(()=>e.classList.remove('show'),1600)}
function showKeyAction(key,label,active=true){const display=document.getElementById('keyActionDisplay');display.classList.toggle('active',active);display.querySelector('kbd').textContent=key;display.querySelector('span').textContent=label}
function flashShortcut(shortcut,label,duration=850){if(keyFlashTimer)clearTimeout(keyFlashTimer);document.querySelectorAll('button.key-active').forEach(button=>button.classList.remove('key-active'));document.querySelectorAll('button[data-shortcut]').forEach(button=>{if(button.dataset.shortcut===shortcut)button.classList.add('key-active')});showKeyAction(shortcut,label,true);keyFlashTimer=setTimeout(()=>{document.querySelectorAll('button.key-active').forEach(button=>button.classList.remove('key-active'));showKeyAction('READY','Keyboard control ready',false)},duration)}
function runShortcut(shortcut,label,action){flashShortcut(shortcut,label);action()}
function setArm(arm){const arms=latestStatus?.arms||1;if(arm>=arms){toast(`Instrument ${arm+1} is not available in this room`);return}stopDrive(false);activeArm=arm;document.getElementById('arm0').classList.toggle('active',arm===0);document.getElementById('arm1').classList.toggle('active',arm===1);toast(`Instrument ${arm+1} active`)}
function setSpeed(speed,button){driveSpeed=speed;document.querySelectorAll('[data-speed]').forEach(x=>x.classList.toggle('active',x===button));updateControlReadout(false,`${button?.textContent.trim()||'Selected'} speed`)}
function setSpeedShortcut(speed){setSpeed(speed,document.querySelector(`[data-speed="${speed}"]`))}
function setHandSpeed(arm,speed,label){keyboardSpeeds[arm]=speed;flashShortcut(label,`Instrument ${arm+1} · ${speed===.35?'precision':speed===1?'normal':'fast'} speed`);toast(`Instrument ${arm+1} speed · ${speed===.35?'precision':speed===1?'normal':'fast'}`);if(bimanualWasActive)updateDrive()}
function deadzone(value){return Math.abs(value)<0.18?0:Math.sign(value)*(Math.abs(value)-0.18)/0.82}
function gamepadDrive(){const values=Array(6).fill(0);const pads=navigator.getGamepads?navigator.getGamepads():[];const pad=[...pads].find(Boolean);if(!pad){lastGamepadGrip=false;lastGamepadCamera=false;return values}inputSource='gamepad';values[1]-=deadzone(pad.axes[0]||0);values[0]+=deadzone(pad.axes[1]||0);values[5]+=deadzone(pad.axes[2]||0);values[4]+=deadzone(pad.axes[3]||0);values[3]+=(pad.buttons[5]?.value||0)-(pad.buttons[4]?.value||0);values[2]+=(pad.buttons[7]?.value||0)-(pad.buttons[6]?.value||0);const gripPressed=!!pad.buttons[0]?.pressed,cameraPressed=!!pad.buttons[3]?.pressed;if(gripPressed&&!lastGamepadGrip)toggleGrip();if(cameraPressed&&!lastGamepadCamera)cycleCameraView();lastGamepadGrip=gripPressed;lastGamepadCamera=cameraPressed;return values}
function normalizeDrive(values){for(const [start,end] of [[0,3],[3,6]]){const norm=Math.hypot(...values.slice(start,end));if(norm>1)for(let i=start;i<end;i++)values[i]/=norm}return values.map(value=>Math.max(-1,Math.min(1,value)))}
function buildDrive(){const values=gamepadDrive();pointerMoves.forEach(move=>{if(move.comboCode)comboValues(move.comboCode).forEach((value,index)=>values[index]+=value);else if(move.values)move.values.forEach((value,index)=>values[index]+=value);else values[move.axis]+=move.direction});return normalizeDrive(values)}
function keyboardArmDrive(arm){const rotate=heldModifiers.has(arm===0?'rotate-left':'rotate-right'),map=rotate?rotationKeyMaps[arm]:handKeyMaps[arm],values=Array(6).fill(0),labels=[];heldKeys.forEach(code=>{const move=map[code];if(move){values[move[0]]+=move[1];labels.push(move[2])}});return {values:normalizeDrive(values),labels,rotate}}
function buildBimanualCommands(){const arms=Math.min(latestStatus?.arms||1,2),commands=[];for(let arm=0;arm<arms;arm++){const hand=keyboardArmDrive(arm);commands.push({arm,values:hand.values,speed:keyboardSpeeds[arm],labels:hand.labels,rotate:hand.rotate})}return commands}
function effectiveSpeed(){if(heldModifiers.has('precision'))return Math.min(driveSpeed,.35);return driveSpeed}
function activeDriveLabel(){const labels=buildBimanualCommands().filter(x=>x.values.some(v=>Math.abs(v)>.01)).map(x=>`R${x.arm+1} ${x.rotate?'wrist':'tool'}: ${x.labels.join(' + ')}`);return labels.join(' · ')||'Moving'}
function updateControlReadout(moving,label){const readout=document.getElementById('controlReadout');readout.classList.toggle('moving',moving);readout.querySelector('span').textContent=moving?(label||'Moving · release to stop'):'Ready · hold a control to move'}
async function flushDrive(){if(driveInFlight||!queuedDrive)return;const next=queuedDrive;queuedDrive=null;driveInFlight=true;try{await post('/api/drive',{values:next.values,arm:activeArm,speed:next.speed,source:next.source})}catch(e){toast(e.message)}finally{driveInFlight=false;if(queuedDrive)flushDrive()}}
function sendDrive(values,speed=effectiveSpeed(),source=inputSource){queuedDrive={values,speed,source};flushDrive()}
async function flushBimanual(){if(bimanualInFlight||!queuedBimanual)return;const next=queuedBimanual;queuedBimanual=null;bimanualInFlight=true;try{await post('/api/drive/bimanual',{commands:next,source:'keyboard_pointer'})}catch(e){toast(e.message)}finally{bimanualInFlight=false;if(queuedBimanual)flushBimanual()}}
function sendBimanual(commands){queuedBimanual=commands.map(({arm,values,speed})=>({arm,values,speed}));flushBimanual()}
function syncKeyVisuals(){document.querySelectorAll('[data-key]').forEach(button=>button.classList.toggle('held',heldKeys.has(button.dataset.key)||[...pointerMoves.values()].some(move=>move.button===button)));document.querySelectorAll('[data-combo-key]').forEach(button=>button.classList.toggle('held',heldKeys.has(button.dataset.comboKey)||[...pointerMoves.values()].some(move=>move.button===button)));document.getElementById('precisionModifier').classList.toggle('active',heldModifiers.has('precision'));document.getElementById('leftRotateModifier').classList.toggle('active',heldModifiers.has('rotate-left'));document.getElementById('rightRotateModifier').classList.toggle('active',heldModifiers.has('rotate-right'))}
function updateDrive(){const commands=buildBimanualCommands(),keyboardActive=commands.some(command=>command.values.some(value=>Math.abs(value)>.01)),pointerValues=buildDrive(),pointerActive=pointerValues.some(value=>Math.abs(value)>.01);if((keyboardActive||pointerActive)&&macroPulseTimer){clearTimeout(macroPulseTimer);macroPulseTimer=null}if(keyboardActive||bimanualWasActive)sendBimanual(commands);bimanualWasActive=keyboardActive;if(!keyboardActive&&(pointerActive||driveWasActive))sendDrive(pointerValues);driveWasActive=pointerActive;syncKeyVisuals();updateControlReadout(keyboardActive||pointerActive,keyboardActive?activeDriveLabel():pointerActive?'Pointer / gamepad motion':null)}
function clearHeldControls(){heldKeys.clear();heldModifiers.clear();pointerMoves.clear();syncKeyVisuals()}
function stopDrive(showToast=true){if(macroPulseTimer){clearTimeout(macroPulseTimer);macroPulseTimer=null}clearHeldControls();driveWasActive=false;bimanualWasActive=false;sendDrive(Array(6).fill(0));sendBimanual(buildBimanualCommands());updateControlReadout(false);if(showToast)toast('Both instruments stopped')}
async function stopTool(){stopDrive();try{await post('/api/stop')}catch(e){toast(e.message)}}
async function emergencyStop(){flashShortcut('Esc','Emergency stop · manual control');stopDrive(false);try{await post('/api/stop');if(latestStatus?.autonomy_mode&&latestStatus.autonomy_mode!=='manual')await post('/api/handoff');toast('Stopped · manual control')}catch(e){toast(e.message)}}
async function grip(open,arm=activeArm){try{await post('/api/gripper',{open,arm});toast(`Instrument ${arm+1} · ${open?'gripper open':'gripper closed'}`)}catch(e){toast(e.message)}}
async function toggleGrip(arm=activeArm){if(arm>=(latestStatus?.arms||1)){toast(`Instrument ${arm+1} is not available in this room`);return}try{const result=await post('/api/gripper/toggle',{arm});toast(`Instrument ${arm+1} · ${result.open?'gripper open':'gripper closed'}`)}catch(e){toast(e.message)}}
async function recording(start){try{await post(start?'/api/record/start':'/api/record/stop');toast(start?'Recording started':'Saving demonstration…')}catch(e){toast(e.message)}}
async function replay(){try{const x=await post('/api/replay-last');toast(x.message)}catch(e){toast(e.message)}}
async function referenceGhost(enabled){try{const x=await post('/api/reference-ghost',{enabled});toast(x.message)}catch(e){toast(e.message)}}
function setCamera(name,button){currentCamera=name;document.getElementById('cameraImage').src=`/video/${name}?t=${Date.now()}`;document.querySelectorAll('[data-camera]').forEach(x=>x.classList.toggle('active',x===button))}
function setCameraShortcut(name){const button=document.querySelector(`[data-camera="${name}"]`);if(!button||button.classList.contains('hidden')){toast(`${name.replace('_',' ')} is not available in this room`);return}setCamera(name,button);toast(`${button.textContent.trim()} camera`)}
async function setCameraView(mode,button){try{const result=await post('/api/camera-view',{mode});currentViewMode=result.mode;document.querySelectorAll('[data-view-mode]').forEach(x=>x.classList.toggle('active',x.dataset.viewMode===result.mode));toast(`${button?.textContent||result.mode} camera ready`)}catch(e){toast(e.message)}}
function cycleCameraView(){const modes=['operative','close','overview','overhead','left_oblique','right_oblique','opposite'],mode=modes[(modes.indexOf(currentViewMode)+1)%modes.length],button=document.querySelector(`[data-view-mode="${mode}"]`);setCameraView(mode,button)}
function cycleSensorCamera(){const buttons=[...document.querySelectorAll('[data-camera]:not(.hidden)')];if(!buttons.length)return;const index=buttons.findIndex(button=>button.dataset.camera===currentCamera),button=buttons[(index+1)%buttons.length];setCamera(button.dataset.camera,button);toast(`${button.textContent.trim()} camera`)}
async function annotatePhase(phase){try{const x=await post('/api/annotation',{phase});toast(x.message)}catch(e){toast(e.message)}}
async function annotateEvent(event){try{const x=await post('/api/annotation',{event});toast('Procedure event saved')}catch(e){toast(e.message)}}
async function resetScene(){try{await post('/api/reset');toast('Scene reset')}catch(e){toast(e.message)}}
async function setAutonomy(mode){try{const x=await post('/api/autonomy',{mode});toast(x.message)}catch(e){toast(e.message)}}
async function takeControl(){stopDrive(false);try{const x=await post('/api/handoff');toast(x.message)}catch(e){toast(e.message)}}
async function startExpert(){try{const x=await post('/api/expert/start');toast(x.message)}catch(e){toast(e.message)}}
async function toggleExpertPause(){const status=latestStatus?.expert_demonstration?.status;try{const x=await post(status==='paused'?'/api/expert/resume':'/api/expert/pause');toast(x.message)}catch(e){toast(e.message)}}
function renderExpert(expert={}){const phases=expert.phases||['rest','approach','align','contact','grasp','manipulate','verify','recover'].map(id=>({id,title:id,status:'pending'})),status=expert.status||'idle',active=status==='running'||status==='paused';document.getElementById('expertRail').innerHTML=phases.map(phase=>`<div class="expert-phase ${phase.status||'pending'}" title="${phase.instruction||phase.title}">${phase.title}</div>`).join('');const current=phases.find(phase=>phase.id===expert.phase),statusLabel={idle:'ready',running:'executing',paused:'paused',completed:'complete',taken_over:'doctor control',cancelled:'cancelled'}[status]||status.replaceAll('_',' '),badge=document.getElementById('expertStatus');badge.textContent=statusLabel;badge.className=`expert-status ${status}`;document.getElementById('expertInstruction').textContent=status==='paused'?(expert.paused_reason||expert.procedure_instruction||current?.instruction||'Paused for inspection.'):status==='taken_over'?`You took control during ${expert.takeover_phase||'the procedure'}. The simulation pose and recording were preserved.`:status==='completed'?'All eight phases completed in the live room. Review the generated trajectory before using it for research.':(expert.procedure_instruction||current?.instruction||'The expert executes the full procedure in the live simulation. Pause, inspect, or take control at any phase.');document.getElementById('expertStart').disabled=active;const pause=document.getElementById('expertPause');pause.disabled=!active;pause.innerHTML=status==='paused'?'Resume <kbd>F10</kbd>':'Pause <kbd>F10</kbd>';document.getElementById('expertTakeover').disabled=!active;document.getElementById('expertProgress').textContent=`${expert.completed_phases?.length||0}/8 phases · ${expert.progress_percent||0}%`;const reference=document.getElementById('expertReference');reference.textContent=expert.reference_demo?'Simulation expert reference saved':active?'Recording synchronized BC candidate':'BC candidate saved after a complete run';reference.classList.toggle('ready',!!expert.reference_demo)}
function toggleReferenceGhost(){referenceGhost(!latestStatus?.reference_ghost?.enabled)}
function toggleKeyboardHelp(force){const help=document.getElementById('keyboardHelp'),show=force??help.classList.contains('hidden');help.classList.toggle('hidden',!show);if(show)stopDrive(false)}
function auditKeyboardCoverage(){const buttons=[...document.querySelectorAll('button')],missing=buttons.filter(button=>!button.dataset.shortcut);const coverage=document.getElementById('keyboardCoverage');coverage.classList.toggle('bad',missing.length>0);coverage.textContent=missing.length?`${buttons.length-missing.length}/${buttons.length} controls mapped · ${missing.length} missing`:`✓ ${buttons.length}/${buttons.length} controls mapped to keyboard`;if(missing.length)console.warn('Buttons missing keyboard shortcuts',missing)}
function simulatorReadablePulse(minimum=550){const fps=Math.max(.5,latestStatus?.sim_fps||2);return Math.max(minimum,Math.ceil(1400/fps))}
async function pulseDrive(values,label,duration=simulatorReadablePulse(),speed=.35){if(macroPulseTimer)clearTimeout(macroPulseTimer);clearHeldControls();driveWasActive=false;try{await post('/api/stop');await post('/api/drive',{values,arm:activeArm,speed,source:'keyboard_smart_action'});updateControlReadout(true,`${label} · bounded pulse`);macroPulseTimer=setTimeout(()=>{macroPulseTimer=null;post('/api/stop').catch(()=>{});updateControlReadout(false)},duration)}catch(e){toast(e.message)}}
function smartTargetNudge(){const offset=latestStatus?.tool_to_object_offset_m?.[activeArm];if(!offset){toast('Target pose is not available yet');return}const ranked=offset.map((value,axis)=>({axis,value,magnitude:Math.abs(value)})).filter(item=>item.magnitude>.0025).sort((a,b)=>b.magnitude-a.magnitude).slice(0,2),values=Array(6).fill(0);ranked.forEach(({axis,value})=>{values[axis]=axis===0?(value<0?-.72:.72):(value>0?.72:-.72)});if(!ranked.length){toast('Target aligned · close the jaws');return}pulseDrive(values,'Target-guided nudge',simulatorReadablePulse(700),.5)}
function smartAction(){flashShortcut('Enter',document.getElementById('smartActionLabel').textContent,1050);const s=latestStatus;if(!s){toast('Waiting for simulator state');return}const open=s.grippers_open?.[activeArm],nativeContact=s.native_grasp_contact_active?.[activeArm],distance=s.tool_to_object_distance_m?.[activeArm],capture=s.grasp_capture_radius_m||.018;if(open===undefined){smartTargetNudge();return}if(open&&distance!==null&&distance!==undefined&&distance<=capture){grip(false);return}if(open){smartTargetNudge();return}if(nativeContact){pulseDrive([.65,0,.65,0,0,0],'Lift + retract',simulatorReadablePulse(850),.35);return}grip(true)}
function bindPointerHold(button,movement){button.addEventListener('pointerdown',event=>{event.preventDefault();inputSource='keyboard_pointer';button.setPointerCapture(event.pointerId);pointerMoves.set(event.pointerId,{...movement,button});showKeyAction(button.dataset.shortcut,movement.label||button.textContent.trim(),true);syncKeyVisuals();updateDrive()});const release=event=>{pointerMoves.delete(event.pointerId);syncKeyVisuals();updateDrive();if(!pointerMoves.size)showKeyAction('READY','Released · motion stopped',false)};button.addEventListener('pointerup',release);button.addEventListener('pointercancel',release);button.addEventListener('lostpointercapture',release);button.addEventListener('contextmenu',event=>event.preventDefault())}
document.querySelectorAll('.move-button').forEach(button=>bindPointerHold(button,{axis:Number(button.dataset.axis),direction:Number(button.dataset.direction)}));
document.querySelectorAll('.combo-button').forEach(button=>bindPointerHold(button,{comboCode:button.dataset.comboKey,label:comboMap[button.dataset.comboKey].label}));
function isTypingTarget(target){return ['INPUT','SELECT','TEXTAREA'].includes(target.tagName)||target.isContentEditable}
function annotationShortcut(code){return {Digit1:['⌥1','Approach annotation',()=>annotatePhase('approach')],Digit2:['⌥2','Grasp annotation',()=>annotatePhase('grasp')],Digit3:['⌥3','Manipulation annotation',()=>annotatePhase('manipulation')],Digit4:['⌥4','Recovery annotation',()=>annotatePhase('recovery')],Digit5:['⌥5','Task event',()=>annotateEvent('task_complete')],Digit6:['⌥6','Safety event',()=>annotateEvent('safety_review')]}[code]}
function handleDiscreteShortcut(event){const {code}=event;if(code==='Slash'&&event.shiftKey){if(!event.repeat)runShortcut('?','Keyboard map',()=>toggleKeyboardHelp());return true}const annotation=event.altKey?annotationShortcut(code):null;if(annotation){if(!event.repeat)runShortcut(...annotation);return true}const speeds={Digit1:[0,.35,'1'],Digit2:[0,1,'2'],Digit3:[0,1.7,'3'],Digit8:[1,.35,'8'],Digit9:[1,1,'9'],Digit0:[1,1.7,'0'],Numpad8:[1,.35,'8'],Numpad9:[1,1,'9'],Numpad0:[1,1.7,'0']};if(speeds[code]){if(!event.repeat)setHandSpeed(...speeds[code]);return true}const cameraSensors={Digit4:['4','Stereo left camera','endoscope_left'],Digit5:['5','Stereo right camera','endoscope_right'],Digit6:['6','Wrist 1 camera','wrist_1'],Digit7:['7','Wrist 2 camera','wrist_2']},cameraViews={F1:['F1','Operative view','operative'],F2:['F2','Close view','close'],F3:['F3','Wide view','overview'],F4:['F4','Overhead view','overhead'],F5:['F5','Left oblique view','left_oblique'],F6:['F6','Right oblique view','right_oblique'],F7:['F7','Opposite-side view','opposite']};if(cameraSensors[code]){if(!event.repeat){const [shortcut,label,name]=cameraSensors[code];runShortcut(shortcut,label,()=>setCameraShortcut(name))}return true}if(cameraViews[code]){if(!event.repeat){const [shortcut,label,mode]=cameraViews[code];runShortcut(shortcut,label,()=>setCameraView(mode,document.querySelector(`[data-view-mode="${mode}"]`)))}return true}const commands={
  Space:['Space','Instrument 1 gripper',()=>toggleGrip(0)],Enter:['Enter','Instrument 2 gripper',()=>toggleGrip((latestStatus?.arms||1)>1?1:0)],NumpadEnter:['Enter','Instrument 2 gripper',()=>toggleGrip((latestStatus?.arms||1)>1?1:0)],Backspace:null,Escape:null,
  BracketLeft:['[','Pointer controls · instrument 1',()=>setArm(0)],BracketRight:[']','Pointer controls · instrument 2',()=>setArm(1)],KeyC:[event.shiftKey?'⇧C':'C',event.shiftKey?'Next camera view':'Next camera sensor',()=>event.shiftKey?cycleCameraView():cycleSensorCamera()],
  Comma:[',','Pointer precision speed',()=>setSpeedShortcut(.35)],Period:['.','Pointer normal speed',()=>setSpeedShortcut(1)],Slash:['/','Pointer fast speed',()=>setSpeedShortcut(1.7)],
  KeyM:['M','Manual control',()=>setAutonomy('manual')],KeyG:['G','Guided control',()=>setAutonomy('guided')],KeyH:['H','Toggle clinician path',()=>toggleReferenceGhost()],F9:['F9','Run live expert',()=>startExpert()],F10:['F10','Pause or resume expert',()=>toggleExpertPause()],F12:['F12','Smart context action',()=>smartAction()],
  KeyY:['Y','Start recording',()=>recording(true)],KeyT:['T','Stop and save',()=>recording(false)],KeyR:['R','Replay last',()=>replay()],Delete:['Delete','Reset scene',()=>resetScene()]
};if(code==='Backspace'||code==='Escape'){if(!event.repeat)emergencyStop();return true}const command=commands[code];if(!command)return false;if(!event.repeat)runShortcut(...command);return true}
document.addEventListener('keydown',event=>{if(isTypingTarget(event.target)||event.metaKey||event.ctrlKey)return;const helpOpen=!document.getElementById('keyboardHelp').classList.contains('hidden');if(helpOpen&&event.code!=='Slash'&&event.code!=='Escape'&&event.code!=='Backspace'){event.preventDefault();return}if(event.code==='ShiftLeft'||event.code==='ShiftRight'){event.preventDefault();heldModifiers.add(event.code==='ShiftLeft'?'rotate-left':'rotate-right');showKeyAction(event.code==='ShiftLeft'?'L⇧':'R⇧',`${event.code==='ShiftLeft'?'Left':'Right'} wrist rotation mode`,true);syncKeyVisuals();updateDrive();return}if(event.code==='AltLeft'||event.code==='AltRight'){event.preventDefault();heldModifiers.add('precision');syncKeyVisuals();return}if(handleDiscreteShortcut(event)){event.preventDefault();if((event.code==='Escape'||event.code==='Backspace')&&helpOpen)toggleKeyboardHelp(false);return}if(!dualMovementCodes.has(event.code)&&!comboMap[event.code])return;event.preventDefault();if(!event.repeat&&!heldKeys.has(event.code))keyDownAt.set(event.code,performance.now());inputSource='keyboard_pointer';heldKeys.add(event.code);updateDrive();showKeyAction(event.key.length===1?event.key.toUpperCase():event.key,activeDriveLabel(),true)});
document.addEventListener('keyup',event=>{if(event.code==='ShiftLeft'||event.code==='ShiftRight'){event.preventDefault();heldModifiers.delete(event.code==='ShiftLeft'?'rotate-left':'rotate-right');syncKeyVisuals();updateDrive();if(!bimanualWasActive)showKeyAction('READY','Wrist rotation mode released',false);return}if(event.code==='AltLeft'||event.code==='AltRight'){event.preventDefault();heldModifiers.delete('precision');syncKeyVisuals();return}if(!dualMovementCodes.has(event.code)&&!comboMap[event.code])return;event.preventDefault();const pressedAt=keyDownAt.get(event.code);keyDownAt.delete(event.code);heldKeys.delete(event.code);updateDrive();if(comboMap[event.code]&&pressedAt!==undefined&&performance.now()-pressedAt<220){const label=comboMap[event.code].label,shortcut=event.key.length===1?event.key.toUpperCase():event.key;flashShortcut(shortcut,`${label} · precision tap`,1050);pulseDrive(comboValues(event.code),`${label} · precision tap`,simulatorReadablePulse(700),Math.max(.35,effectiveSpeed()*.55));return}showKeyAction(heldKeys.size?'HOLD':'READY',heldKeys.size?activeDriveLabel():'Released · motion stopped',heldKeys.size>0)});
window.addEventListener('blur',()=>stopDrive(false));document.addEventListener('visibilitychange',()=>{if(document.hidden)stopDrive(false)});
document.getElementById('cameraView').addEventListener('pointermove',event=>{const view=event.currentTarget,rect=view.getBoundingClientRect();const u=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),v=Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height));const cursor=document.getElementById('gazeCursor');cursor.style.left=`${u*100}%`;cursor.style.top=`${v*100}%`;view.classList.add('gaze-on');const now=performance.now();if(now-lastGazeSend>100){lastGazeSend=now;post('/api/gaze',{u,v,valid:true,source:'pointer_attention_proxy'}).catch(()=>{})}});document.getElementById('cameraView').addEventListener('pointerleave',()=>document.getElementById('cameraView').classList.remove('gaze-on'));
function targetDirections(offset){if(!offset)return'';const choices=[];if(Math.abs(offset[2])>.004)choices.push([Math.abs(offset[2]),offset[2]>0?'Up':'Down']);if(Math.abs(offset[1])>.004)choices.push([Math.abs(offset[1]),offset[1]>0?'Left':'Right']);if(Math.abs(offset[0])>.004)choices.push([Math.abs(offset[0]),offset[0]<0?'Toward':'Away']);return choices.sort((a,b)=>b[0]-a[0]).slice(0,2).map(x=>x[1]).join(' + ')}
async function refresh(){try{
  const s=await(await fetch('/api/status',{cache:'no-store'})).json();latestStatus=s;if(activeArm>=s.arms){activeArm=0;document.getElementById('arm0').classList.add('active');document.getElementById('arm1').classList.remove('active')}document.getElementById('dot').classList.add('ok');document.getElementById('connection').textContent='Isaac Lab live';
  const p=s.procedure||{};document.getElementById('procedureTitle').textContent=p.title||'Free practice';document.getElementById('procedureObjective').textContent=p.objective||'Use the robot controls to explore the digital twin.';document.getElementById('procedureProgress').style.width=`${p.progress_percent||0}%`;document.getElementById('procedureSteps').innerHTML=(p.steps||[]).map((x,i)=>`<div class="procedure-step ${x.status}"><span>${String(i+1).padStart(2,'0')}</span><div><b>${x.title}</b><br>${x.instruction}</div></div>`).join('');
  const truth=document.getElementById('procedureTruth');truth.textContent=p.truth_note||'';truth.classList.toggle('hidden',!p.truth_note);document.querySelectorAll('[data-camera]').forEach(button=>button.classList.toggle('hidden',!s.camera_names.includes(button.dataset.camera)));document.getElementById('armPanel').classList.toggle('hidden',s.arms<2);document.getElementById('gripperPanel').classList.toggle('hidden',!s.has_grippers);
  currentViewMode=s.camera_view_mode||currentViewMode;document.querySelectorAll('[data-view-mode]').forEach(x=>x.classList.toggle('active',x.dataset.viewMode===currentViewMode));
  document.getElementById('recflag').classList.toggle('on',s.recording);document.getElementById('record').classList.toggle('state-active',s.recording);document.getElementById('gripOpenButton').classList.toggle('state-active',s.grippers_open?.[0]===false);document.getElementById('gripCloseButton').classList.toggle('state-active',s.grippers_open?.[(s.arms||1)>1?1:0]===false);
	  const proximity=document.getElementById('proximity'),distance=s.tool_to_object_distance_m?.[activeArm],offset=s.tool_to_object_offset_m?.[activeArm],clearance=s.closest_anatomy_clearance_m;proximity.className='proximity';let guidance='Move toward the target';if(s.native_grasp_contact_active?.[activeArm]){guidance='Native jaw contact detected · lift smoothly';proximity.classList.add('held')}else if(distance!==null&&distance!==undefined&&distance<=(s.grasp_capture_radius_m||.018)){guidance=`Aligned ${Math.round(distance*1000)} mm · close jaws`;proximity.classList.add('near')}else if(distance!==null&&distance!==undefined){guidance=`Target ${Math.round(distance*1000)} mm · ${targetDirections(offset)||'hold course'}`}else if(clearance!==null&&clearance!==undefined){guidance=`Anatomy clearance ${Math.round(clearance*1000)} mm`};proximity.innerHTML=`<b>Next</b><span>${guidance}</span>`;const sensor=document.getElementById('procedureSensor');sensor.innerHTML='';sensor.classList.add('hidden');const smartLabel=document.getElementById('smartActionLabel'),open=s.grippers_open?.[activeArm],contact=s.native_grasp_contact_active?.[activeArm];smartLabel.textContent=open===undefined?'Precision nudge toward target':open&&distance!==null&&distance!==undefined&&distance<=(s.grasp_capture_radius_m||.018)?'Close jaws on aligned target':open?'Precision nudge toward target':contact?'Lift the physically held object':'Open jaws and retry';
  const labels={manual:'L0 · Manual',guided:'L1 · Guided',supervised_replay:'L2 · Supervised replay',expert_demonstration:'L2 · Live expert'};document.getElementById('autonomyState').textContent=labels[s.autonomy_mode]||s.autonomy_mode;document.getElementById('manualMode').classList.toggle('active',s.autonomy_mode==='manual');document.getElementById('guidedMode').classList.toggle('active',s.autonomy_mode==='guided');document.getElementById('coachingCue').textContent=s.coaching_cue;document.getElementById('forceMetric').textContent=s.safety?.max_contact_force_n===null?'—':Number(s.safety.max_contact_force_n).toFixed(2);document.getElementById('deformMetric').textContent=s.safety?.max_tissue_displacement_m===null?'—':(Number(s.safety.max_tissue_displacement_m)*1000).toFixed(1);document.getElementById('stressMetric').textContent=s.safety?.max_tissue_stress_pa===null?'—':Number(s.safety.max_tissue_stress_pa).toExponential(1);renderExpert(s.expert_demonstration);
	  const ghost=document.getElementById('ghostState');ghost.classList.toggle('on',!!s.reference_ghost?.enabled);ghost.textContent=s.reference_ghost?.enabled?`${s.reference_ghost.point_count} registered path points · ${s.reference_ghost.reference}`:'Clinician path hidden';document.getElementById('status').innerHTML=`Task<br><b>${s.task}</b><br>Procedure: ${p.title||'Free practice'}<br>Anatomy: ${s.anatomy_showcase||'—'}<br>Scenario: ${s.scenario_title}<br>Robots: ${s.robot_names.join(', ')}<br>Autonomy: ${labels[s.autonomy_mode]||s.autonomy_mode}<br>Phase: ${s.operator_study.procedure_phase}<br>Input: ${s.operator_study.input_source}<br>Annotations: ${s.operator_study.annotation_count}<br>Interventions: ${s.intervention_count}<br>Simulation: ${s.sim_fps.toFixed(1)} Hz<br>Controls: ${s.drive_active?'moving':'ready'}<br>Physics: ${s.physics_authority?.effective_backend||'native'}<br>Recorded frames: ${s.recorded_frames}<br>Replay: ${s.replaying?'running':'idle'}`;if(s.last_demo)document.getElementById('lastDemo').innerHTML=`Last saved: <a href="/demos/${s.last_demo}" style="color:#2cd2e8">${s.last_demo}</a>`;
}catch(e){document.getElementById('dot').classList.remove('ok');document.getElementById('connection').textContent='Reconnecting…'}}
auditKeyboardCoverage();setInterval(updateDrive,90);setInterval(refresh,500);setInterval(()=>post('/api/operator/heartbeat').catch(()=>{}),10000);refresh();
</script></body></html>"""


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


class GripperRequest(BaseModel):
    open: bool
    arm: int = 0


class GripperToggleRequest(BaseModel):
    arm: int = 0


class CameraViewRequest(BaseModel):
    mode: str


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
    render_fps: float = 0.0
    sim_fps: float = 0.0
    sim_step: int = 0
    pulse: np.ndarray = field(init=False)
    pulse_steps: int = 0
    drive: np.ndarray = field(init=False)
    drive_until: float = 0.0
    grippers_open: list[bool] = field(init=False)
    native_grasp_contact_active: list[bool] = field(init=False)
    tool_to_object_distance_m: list[float | None] = field(init=False)
    tool_to_object_offset_m: list[list[float] | None] = field(init=False)
    grasp_capture_radius_m: float = 0.018
    camera_view_mode: str = "operative"
    camera_view_request: str | None = None
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
    scenario_seed: int = 7777
    autonomy_mode: str = "manual"
    intervention_count: int = 0
    coaching_cue: str = "You command every movement. Dr.Anmar records telemetry for coaching."
    evaluation_status: str = "idle"
    evaluation_source: str | None = None
    evaluation_output: str | None = None
    camera_intrinsics: list[list[float]] | None = None
    semantic_labels: dict[str, Any] = field(default_factory=dict)
    runtime_provenance: dict[str, Any] = field(default_factory=dict)
    physics_authority: dict[str, Any] = field(default_factory=dict)
    camera_valid_depth_fraction: float | None = None
    camera_foreground_fraction: float | None = None
    camera_mean_luminance: float | None = None
    camera_nonblank_seen: bool = False
    needle_visual_ready: bool = False
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
    procedure_events: list[dict[str, Any]] = field(default_factory=list)
    procedure_waypoints_total: int = 0
    procedure_waypoints_completed: int = 0
    procedure_motion_seen: bool = False
    procedure_grasp_seen: bool = False
    procedure_object_lift_m: float = 0.0
    procedure_object_motion_m: float = 0.0
    procedure_started_at: float = 0.0
    procedure_last_motion_at: float = 0.0
    native_telemetry: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pulse = np.zeros(self.action_dim, dtype=np.float32)
        self.drive = np.zeros(self.action_dim, dtype=np.float32)
        self.grippers_open = [True] * self.arms
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

    def status(self) -> dict[str, Any]:
        with self.lock:
            procedure_status = self._procedure_status()
            guide_kind = str(self.procedure.get("guide_kind", ""))
            thread_required = False
            needle_required = guide_kind in NATIVE_NEEDLE_GUIDE_KINDS
            thread_geometry_ready = True
            needle_geometry_ready = not needle_required or self.needle_visual_ready
            camera_frame_ready = bool(self.frame_id > 0 and self.frame_jpeg and self.camera_nonblank_seen)
            render_contract = {
                "ready": bool(camera_frame_ready and needle_geometry_ready and thread_geometry_ready),
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
                "frame_id": self.frame_id,
                "render_fps": self.render_fps,
                "sim_fps": self.sim_fps,
                "sim_step": self.sim_step,
                "action_dim": self.action_dim,
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
                "grippers_open": self.grippers_open,
                "native_grasp_contact_active": self.native_grasp_contact_active,
                "tool_to_object_distance_m": self.tool_to_object_distance_m,
                "tool_to_object_offset_m": self.tool_to_object_offset_m,
                "grasp_capture_radius_m": self.grasp_capture_radius_m,
                "camera_view_mode": self.camera_view_mode,
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
                "sensor_profile": self.sensor_profile,
                "runtime_provenance": self.runtime_provenance,
                "physics_authority": self.physics_authority,
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
                "sensor_quality": {
                    "valid_depth_fraction": self.camera_valid_depth_fraction,
                    "semantic_foreground_fraction": self.camera_foreground_fraction,
                    "mean_luminance": self.camera_mean_luminance,
                },
                "render_contract": render_contract,
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
            }

    def _procedure_status(self) -> dict[str, Any]:
        if not self.procedure:
            return {}
        now = time.monotonic()
        kind = self.procedure.get("guide_kind")
        step_count = len(self.procedure.get("steps", []))
        if kind == "needle_pass":
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
            "native_telemetry": self.native_telemetry,
        }


def build_web_app(state: SharedState) -> FastAPI:
    app = FastAPI(title="Dr.Anmar Surgical Workstation", docs_url=None, redoc_url=None)
    operator_lease = OperatorLease()

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
        return response

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return APP_HTML

    @app.get("/api/status")
    def status() -> JSONResponse:
        return JSONResponse({**state.status(), "operator_lease": operator_lease.status()})

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
        command[body_slice.start + request.axis] = (0.02 if request.axis < 3 else 0.08) * request.direction
        with state.lock:
            state.pulse = command
            state.pulse_steps = 1
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
                request.source == "keyboard_smart_action"
                and state.native_grasp_contact_active[request.arm]
                and state.needle_tip_clearance_m is not None
                and state.needle_tip_clearance_m > 0.020
            )
            semantic_target_far = (
                request.source == "keyboard_smart_action"
                and not state.native_grasp_contact_active[request.arm]
                and state.tool_to_object_distance_m[request.arm] is not None
                and state.tool_to_object_distance_m[request.arm] > 0.050
            )
        translation_boost = 6.0 if semantic_far_field else 3.0 if semantic_target_far else 1.0
        scales = np.asarray(
            (0.006 * translation_boost,) * 3 + (0.03, 0.03, 0.03),
            dtype=np.float32,
        )
        command[state.body_action_slice(request.arm)] = calibrated_values * scales * request.speed
        active = bool(np.any(values))
        with state.lock:
            # Keep a command alive long enough for at least one slow Isaac step.
            # The workstation normally refreshes held keys continuously, but
            # bounded keyboard macros intentionally send one command followed
            # by an explicit stop. A fixed 300 ms expiry could disappear
            # between frames when a photorealistic scene renders near 2 Hz.
            hold_seconds = max(0.30, min(1.25, 1.4 / max(state.sim_fps, 1.0)))
            state.drive = command
            state.operator_input_source = request.source
            state.drive_until = time.monotonic() + hold_seconds if active else 0.0
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
        scales = np.asarray((0.006,) * 3 + (0.03, 0.03, 0.03), dtype=np.float32)
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
            command[state.body_action_slice(arm_command.arm)] = calibrated_values * scales * arm_command.speed
            active = active or bool(np.any(values))
            seen_arms.add(arm_command.arm)

        with state.lock:
            hold_seconds = max(0.30, min(1.25, 1.4 / max(state.sim_fps, 1.0)))
            state.drive = command
            state.drive_until = time.monotonic() + hold_seconds if active else 0.0
            state.operator_input_source = request.source
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

    @app.post("/api/stop")
    def stop() -> dict[str, bool]:
        with state.lock:
            state.pulse.fill(0.0)
            state.pulse_steps = 0
            state.drive.fill(0.0)
            state.drive_until = 0.0
            state.replay_request = "stop"
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
            state.camera_view_mode = request.mode
            state.camera_view_request = request.mode
        state.wake_event.set()
        return {"ok": True, "mode": request.mode}

    @app.post("/api/gripper")
    def gripper(request: GripperRequest) -> dict[str, Any]:
        if not state.has_grippers:
            raise HTTPException(409, "This robot has no gripper action")
        if request.arm not in range(state.arms):
            raise HTTPException(400, f"arm must be between 0 and {state.arms - 1}")
        with state.lock:
            if state.expert_demonstration.get("status") in {"running", "paused"}:
                state.expert_request = "take_over"
                state.expert_clean_run = False
            state.grippers_open[request.arm] = request.open
        state.wake_event.set()
        return {"ok": True, "open": request.open, "arm": request.arm}

    @app.post("/api/gripper/toggle")
    def toggle_gripper(request: GripperToggleRequest) -> dict[str, Any]:
        if not state.has_grippers:
            raise HTTPException(409, "This robot has no gripper action")
        if request.arm not in range(state.arms):
            raise HTTPException(400, f"arm must be between 0 and {state.arms - 1}")
        with state.lock:
            if state.expert_demonstration.get("status") in {"running", "paused"}:
                state.expert_request = "take_over"
                state.expert_clean_run = False
            state.grippers_open[request.arm] = not state.grippers_open[request.arm]
            is_open = state.grippers_open[request.arm]
        state.wake_event.set()
        return {"ok": True, "open": is_open, "arm": request.arm}

    @app.post("/api/reset")
    def reset() -> dict[str, bool]:
        with state.lock:
            state.reset_requested = True
            state.drive.fill(0.0)
            state.drive_until = 0.0
            state.replay_request = "stop"
            state.expert_request = "cancel"
            state.expert_clean_run = False
            state.grippers_open = [True] * state.arms
        state.wake_event.set()
        return {"ok": True}

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
            if not state.procedure.get("simulation_ready", True):
                raise HTTPException(
                    409,
                    state.procedure.get(
                        "readiness_reason",
                        "This room is unavailable until its native physics is ready.",
                    ),
                )
            if state.recording or state.record_request == "start":
                raise HTTPException(409, "Stop the current recording before starting the expert")
            if state.replaying or state.evaluation_status in {"running", "saving"}:
                raise HTTPException(409, "Stop replay or evaluation before starting the expert")
            if state.expert_demonstration.get("status") in {"running", "paused"}:
                raise HTTPException(409, "The expert demonstration is already active")
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
        if Path(name).name != name or not name.endswith((".npz", ".json")):
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
            jpeg = state.frame_jpeg
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


def rgb_tensor_to_image(rgb: torch.Tensor, scenario_id: str = "baseline", dropout: bool = False) -> Image.Image:
    array = rgb[..., :3].detach().cpu().numpy()
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8, copy=False)
    image = Image.fromarray(array)
    return Image.new("RGB", image.size, (0, 0, 0)) if dropout else apply_visual_scenario(image, scenario_id)


def encode_jpeg(rgb: torch.Tensor, scenario_id: str = "baseline", dropout: bool = False) -> tuple[bytes, float]:
    image = rgb_tensor_to_image(rgb, scenario_id, dropout)
    sample = np.asarray(image.resize((32, 24), Image.Resampling.BILINEAR), dtype=np.float32)
    mean_luminance = float(sample.mean() / 255.0)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=86, optimize=False)
    return buffer.getvalue(), mean_luminance


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


_DEMO_INSPECTION_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def inspect_demo_file(path: Path) -> dict[str, Any]:
    """Bounded, cached structural validation for replay and dataset use."""
    try:
        stat = path.stat()
    except OSError as exc:
        return {"valid": False, "training_eligible": False, "error": str(exc)}
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _DEMO_INSPECTION_CACHE.get(cache_key)
    if cached is not None:
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
            warnings = [] if frame_count >= 2 else ["Recording has fewer than two control frames"]
            result = {
                "valid": True,
                "training_eligible": frame_count >= 2,
                "frames": frame_count,
                "action_dim": int(actions.shape[1]),
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
    if len(_DEMO_INSPECTION_CACHE) >= 512:
        _DEMO_INSPECTION_CACHE.clear()
    _DEMO_INSPECTION_CACHE[cache_key] = result
    return dict(result)


def require_replayable_demo(path: Path) -> dict[str, Any]:
    inspection = inspect_demo_file(path)
    if not inspection.get("valid"):
        raise HTTPException(422, inspection.get("error", "The demonstration is unreadable"))
    if not inspection.get("training_eligible"):
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
    targets = {"object": objects["object"]} if "object" in objects else objects
    generator = np.random.default_rng(seed)
    seeded_jitter = generator.uniform(-0.0015, 0.0015, size=3).astype(np.float32)
    seeded_jitter[2] = 0.0
    for rigid_object in targets.values():
        pose = rigid_object.data.root_pose_w.clone()
        delta = torch.tensor(np.asarray(offset, dtype=np.float32) + seeded_jitter, device=pose.device)
        pose[:, :3] += delta
        rigid_object.write_root_pose_to_sim(pose)
        rigid_object.write_root_velocity_to_sim(torch.zeros((pose.shape[0], 6), device=pose.device))


def sample_deformable_safety(deformables: dict[str, Any]) -> dict[str, float]:
    telemetry: dict[str, float] = {}
    for name, deformable in deformables.items():
        try:
            nodal_position = deformable.data.nodal_pos_w[0]
            default_position = deformable.data.default_nodal_state_w[0, :, :3]
            displacement = torch.linalg.vector_norm(nodal_position - default_position, dim=-1).max()
            deformation = deformable.data.sim_element_deform_gradient_w[0]
            identity = torch.eye(3, device=deformation.device).reshape(1, 3, 3)
            deformation_proxy = torch.linalg.matrix_norm(deformation - identity, dim=(-2, -1)).max()
            stress = deformable.data.sim_element_stress_w[0]
            max_stress = torch.linalg.matrix_norm(stress, dim=(-2, -1)).max()
            telemetry[f"{name}_max_tissue_displacement_m"] = float(displacement.detach().cpu().item())
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
        "physics_authority": state.physics_authority,
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
        "physics_authority": state.physics_authority,
    }


def array_payload_bytes(frame: dict[str, np.ndarray]) -> int:
    return sum(int(np.asarray(value).nbytes) for value in frame.values())


def save_demo(
    state: SharedState,
    frames: list[dict[str, np.ndarray]],
    vision_frames: list[dict[str, np.ndarray]],
    started_at: str,
) -> str | None:
    if not frames:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    task_slug = state.task.lower().replace("isaac-", "").replace("-v0", "").replace("-", "_")
    name = f"dr_anmar_{task_slug}_{stamp}.npz"
    path = state.demo_dir / name
    keys = tuple(sorted(set.intersection(*(set(frame.keys()) for frame in frames))))
    arrays = {key: np.stack([frame[key] for frame in frames]) for key in keys}
    if vision_frames:
        arrays["endoscope_time_s"] = np.stack([frame["time_s"] for frame in vision_frames])
        arrays["endoscope_rgb"] = np.stack([frame["rgb"] for frame in vision_frames])
        arrays["endoscope_sensor_dropout_active"] = np.stack(
            [frame.get("sensor_dropout_active", np.array(False, dtype=np.bool_)) for frame in vision_frames]
        )
        if all("depth_m" in frame for frame in vision_frames):
            arrays["endoscope_depth_m"] = np.stack([frame["depth_m"] for frame in vision_frames])
        if all("semantic_id" in frame for frame in vision_frames):
            arrays["endoscope_semantic_id"] = np.stack([frame["semantic_id"] for frame in vision_frames])
        if all("point_cloud_camera_m" in frame for frame in vision_frames):
            point_counts = {len(frame["point_cloud_camera_m"]) for frame in vision_frames}
            if len(point_counts) == 1:
                arrays["endoscope_point_cloud_camera_m"] = np.stack(
                    [frame["point_cloud_camera_m"] for frame in vision_frames]
                )
        for camera_name in ("endoscope_right", "wrist_1", "wrist_2"):
            key = f"{camera_name}_rgb"
            if all(key in frame for frame in vision_frames):
                arrays[key] = np.stack([frame[key] for frame in vision_frames])
    analysis = analyze_demo(
        arrays,
        state.task,
        state.arms,
        state.robot_body_names,
        str(state.procedure.get("id", "")),
    )
    temporary_data = path.with_suffix(".npz.tmp")
    with temporary_data.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary_data.replace(path)
    times = np.asarray(arrays.get("time_s", []), dtype=np.float64).reshape(-1)
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
            "procedure_fidelity": state.procedure.get("fidelity"),
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
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "frames": len(frames),
        "vision_frames": len(vision_frames),
        "sensor_profile": state.sensor_profile,
        "uncompressed_payload_bytes": sum(array_payload_bytes(frame) for frame in frames + vision_frames),
        "control_hz": round(observed_control_hz, 2),
        "control_hz_nominal": 50,
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
        "data_file": name,
        "data_bytes": path.stat().st_size,
        "modalities": {
            "robot_state_hz": round(observed_control_hz, 2),
            "robot_state_hz_nominal": 50,
            "endoscope_rgb_hz": 5 if vision_frames else 0,
            "endoscope_rgb_resolution": [360, 240] if vision_frames else None,
            "endoscope_depth_hz": 5 if "endoscope_depth_m" in arrays else 0,
            "endoscope_depth_units": "metres" if "endoscope_depth_m" in arrays else None,
            "endoscope_semantic_hz": 5 if "endoscope_semantic_id" in arrays else 0,
            "endoscope_semantic_encoding": "uint32 semantic id" if "endoscope_semantic_id" in arrays else None,
            "endoscope_point_cloud_hz": 5 if "endoscope_point_cloud_camera_m" in arrays else 0,
            "endoscope_point_cloud_frame": "left endoscope camera optical frame",
            "stereo_right_rgb_hz": 5 if "endoscope_right_rgb" in arrays else 0,
            "instrument_wrist_rgb_hz": 5 if "wrist_1_rgb" in arrays else 0,
            "instrument_wrist_camera_count": sum(1 for key in ("wrist_1_rgb", "wrist_2_rgb") if key in arrays),
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
        "physics_authority": state.physics_authority,
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
    if kind == "threading":
        return np.asarray(
            ((-0.038, -0.018, 0.025), (-0.018, -0.002, 0.038), (0.006, 0.012, 0.044), (0.032, 0.022, 0.032)),
            dtype=np.float32,
        )
    if kind == "cutting_path":
        return np.asarray(
            ((-0.055, -0.018, 0.052), (-0.030, -0.010, 0.046), (-0.004, 0.000, 0.043), (0.024, 0.011, 0.040), (0.050, 0.020, 0.036)),
            dtype=np.float32,
        )
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
    procedure_physics = _physics_authority.procedure_readiness(
        procedure or {"fidelity": "anatomy_context"},
        _physics_runtime,
    )
    procedure["physics"] = procedure_physics
    procedure["simulation_ready"] = procedure_physics["ready"]
    procedure["readiness_reason"] = procedure_physics["reason"]
    if "-IK-Rel" not in args_cli.task:
        raise ValueError("The browser workstation accepts relative-IK tasks. Other variants remain available via the CLI.")
    guide_kind = str(procedure.get("guide_kind", ""))
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.episode_length_s = 3600.0
    env_cfg.scene.num_envs = 1
    native_room = _native_room if _native_room and _native_room.get("available") else None
    native_deformable_enabled = bool(native_room and native_room.get("backend") == "physx_fem")
    for robot_attribute in ("robot", "robot_1", "robot_2"):
        robot_cfg = getattr(env_cfg.scene, robot_attribute, None)
        if robot_cfg is not None and getattr(robot_cfg, "spawn", None) is not None:
            robot_cfg.spawn.activate_contact_sensors = True
    if native_deformable_enabled:
        env_cfg.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["psm_tool_gripper.*_joint"],
            open_command_expr={"psm_tool_gripper1_joint": -0.5, "psm_tool_gripper2_joint": 0.5},
            close_command_expr={"psm_tool_gripper1_joint": -0.09, "psm_tool_gripper2_joint": 0.09},
        )
    camera_target = np.asarray(env_cfg.viewer.lookat, dtype=np.float32)
    # Start from the room-facing side used by the official OR scene so the
    # doctor sees the instrument, liver, table, and surrounding environment.
    camera_eye = np.asarray((0.45, 0.25, 0.28), dtype=np.float32)
    endoscope_data_types = ["rgb"] if args_cli.sensor_profile == "efficient" else ["rgb", "distance_to_image_plane", "semantic_segmentation"]
    env_cfg.scene.endoscope = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Endoscope",
        update_period=0.04,
        height=args_cli.camera_height,
        width=args_cli.camera_width,
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
    if args_cli.sensor_profile in {"stereo", "research"}:
        env_cfg.scene.endoscope_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/EndoscopeRight",
            update_period=0.04,
            height=args_cli.camera_height,
            width=args_cli.camera_width,
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
    wrist_robot_names = ("Robot_1", "Robot_2") if "Dual" in args_cli.task else ("Robot",)
    for contact_index, contact_robot_name in enumerate(wrist_robot_names, start=1):
        if native_deformable_enabled:
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
                    ),
                )
        else:
            setattr(
                env_cfg.scene,
                f"gripper_contact_{contact_index}",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{contact_robot_name}/psm_tool_gripper.*_link",
                    update_period=0.0,
                    history_length=3,
                    track_air_time=False,
                ),
            )
    for wrist_index, wrist_robot_name in enumerate(wrist_robot_names, start=1) if args_cli.sensor_profile == "research" else ():
        setattr(
            env_cfg.scene,
            f"wrist_{wrist_index}",
            CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/DrAnmarWristCamera{wrist_index}",
                update_period=0.04,
                height=360,
                width=480,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0,
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
    if openusd_environment:
        env_cfg.scene.openusd_operating_room = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/OpenUSDOperatingRoom",
            spawn=sim_utils.UsdFileCfg(usd_path=str(openusd_environment)),
        )
    else:
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
    elif procedure_id in {"needle-pickup", "needle-transfer"}:
        anatomy_position = (-0.117, -0.2445, -0.144)
    else:
        anatomy_position = (-0.117, -0.1945, -0.164)
    if native_deformable_enabled:
        spawn = native_room["spawn"]
        material_contract = json.loads(Path(native_room["material_path"]).read_text(encoding="utf-8"))
        tissue_material = material_contract["intact_tissue"]
        contact_material = material_contract["contact"]
        native_spawn = sim_utils.UsdFileCfg(
            usd_path=str(native_room["asset_path"]),
            scale=tuple(spawn["scale"]),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                deformable_enabled=True,
                self_collision=True,
                solver_position_iteration_count=16,
                vertex_velocity_damping=0.005,
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
            """Spawn the OpenUSD TetMesh and bind one native PhysX material."""

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
                density=float(tissue_material["density_kg_m3_seed"]),
                dynamic_friction=float(contact_material["dynamic_friction_seed"]),
                youngs_modulus=float(tissue_material["youngs_modulus_pa_seed"]),
                poissons_ratio=float(tissue_material["poisson_ratio_seed"]),
            )
            material_cfg.func(material_path, material_cfg)
            sim_utils.bind_physics_material(root_path, material_path)
            return root_prim

        native_spawn.func = spawn_native_deformable_with_material
        env_cfg.scene.native_tissue = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/NativeTissue",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=tuple(spawn["translation_m"])),
            spawn=native_spawn,
        )
    elif organ_usd.is_file():
        env_cfg.scene.liver_showcase = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/LiverShowcase",
            init_state=AssetBaseCfg.InitialStateCfg(pos=anatomy_position),
            spawn=sim_utils.UsdFileCfg(usd_path=str(organ_usd), scale=(0.35, 0.35, 0.35)),
        )
    env_cfg.num_rerenders_on_reset = 3

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    scene = env.unwrapped.scene
    camera = scene["endoscope"]
    stereo_right_camera = scene["endoscope_right"] if args_cli.sensor_profile in {"stereo", "research"} else None
    wrist_cameras = [scene[f"wrist_{index}"] for index in range(1, len(wrist_robot_names) + 1)] if args_cli.sensor_profile == "research" else []
    camera_sources = {"endoscope_left": camera}
    if stereo_right_camera is not None:
        camera_sources["endoscope_right"] = stereo_right_camera
    camera_sources.update({f"wrist_{index}": wrist_camera for index, wrist_camera in enumerate(wrist_cameras, start=1)})
    robot_names = sorted(scene.articulations.keys())
    robots = {name: scene[name] for name in robot_names}
    robot_body_names = {name: list(getattr(robot, "body_names", [])) for name, robot in robots.items()}
    object_names = sorted(scene.rigid_objects.keys())
    objects = {name: scene[name] for name in object_names}
    deformable_names = sorted(getattr(scene, "deformable_objects", {}).keys())
    deformables = {name: scene[name] for name in deformable_names}
    native_tissue = deformables.get(str(native_room.get("stage_key", ""))) if native_room else None
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
        if attachment["side"] == "minimum":
            mask = coordinates <= torch.min(coordinates) + width_m
        else:
            mask = coordinates >= torch.max(coordinates) - width_m
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
    showcase_children: list[Any] = []
    default_showcase_names: set[str] = {"Liver_topo_blender"}
    collision_mesh_count = 0
    anatomy_guard_volumes: list[tuple[np.ndarray, np.ndarray, str]] = []
    anatomy_surface_samples: list[tuple[np.ndarray, np.ndarray, str]] = []
    anatomy_collision_prims: list[Any] = []
    stage = None
    showcase_prim = None
    if organ_usd.is_file() and not native_deformable_enabled:
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

    def update_wrist_camera_poses() -> None:
        """Keep a close oblique camera behind each instrument and aimed at its jaws."""
        world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        fallback_axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
        for arm, wrist_camera in enumerate(wrist_cameras):
            if arm >= len(robot_names):
                continue
            robot_name = robot_names[arm]
            robot = robots[robot_name]
            names = robot_body_names.get(robot_name, [])
            tip_index = next(
                (names.index(candidate) for candidate in (wrist_tip_name, "psm_tool_tip_link", "endo360_needle", "ecm_end_link") if candidate in names),
                None,
            )
            if tip_index is None:
                continue
            rear_index = next(
                (
                    names.index(candidate)
                    for candidate in ("psm_tool_roll_link", "psm_main_insertion_link_3", "endo360_link", "ecm_yaw_link")
                    if candidate in names
                ),
                None,
            )
            positions = robot.data.body_pos_w[0, :, :3].detach().cpu().numpy().astype(np.float32)
            tip = positions[tip_index]
            shaft = tip - positions[rear_index] if rear_index is not None else np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
            shaft_norm = float(np.linalg.norm(shaft))
            if shaft_norm < 1e-6:
                continue
            shaft /= shaft_norm
            lateral = np.cross(shaft, world_up)
            lateral_norm = float(np.linalg.norm(lateral))
            if lateral_norm < 1e-6:
                lateral = np.cross(shaft, fallback_axis)
                lateral_norm = float(np.linalg.norm(lateral))
            lateral /= max(lateral_norm, 1e-6)
            eye = tip - shaft * 0.055 + lateral * 0.022 + world_up * 0.014
            target = tip + shaft * 0.028
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
        names = (
            [f"gripper_contact_{arm + 1}_jaw_1", f"gripper_contact_{arm + 1}_jaw_2"]
            if native_deformable_enabled
            else [f"gripper_contact_{arm + 1}"]
        )
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


    def apply_endoscope_camera_view(selected_scenario: str, view_mode: str) -> None:
        selected_eye, selected_target = scenario_camera_pose(camera_eye, camera_target, selected_scenario)
        selected_eye, selected_target = camera_view_pose(selected_eye, selected_target, view_mode)
        camera.set_world_poses_from_view(
            torch.tensor([selected_eye.tolist()], device=camera.device),
            torch.tensor([selected_target.tolist()], device=camera.device),
        )
        right_offset = SCENARIO_NATIVE_PROFILES.get(selected_scenario, {}).get(
            "right_camera_offset_m", (0.0, 0.006, 0.0)
        )
        selected_right_eye = selected_eye + np.asarray(right_offset, dtype=np.float32)
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
    arms = 2 if "Dual" in args_cli.task else 1
    has_grippers = action_dim >= arms * 7
    apply_endoscope_camera_view("baseline", "operative")
    update_wrist_camera_poses()

    state = SharedState(
        task=args_cli.task,
        camera_width=args_cli.camera_width,
        camera_height=args_cli.camera_height,
        demo_dir=args_cli.demo_dir,
        action_dim=action_dim,
        arms=arms,
        has_grippers=has_grippers,
        robot_names=robot_names,
        robot_body_names=robot_body_names,
        anatomy_showcase=str(procedure.get("anatomy_focus") or "Operative field"),
        anatomy_scene_id=args_cli.anatomy_scene_id,
        anatomy_asset=str(organ_usd) if organ_usd.is_file() else "",
        openusd_environment=str(openusd_environment) if openusd_environment else "",
        procedure=procedure,
        openusd_scene_loaded=bool(
            native_deformable_enabled
            or (openusd_environment and organ_usd.is_file() and showcase_children)
        ),
        anatomy_collision_meshes=collision_mesh_count,
        sensor_profile=args_cli.sensor_profile,
        needle_visual_ready=bool(objects) if guide_kind in NATIVE_NEEDLE_GUIDE_KINDS else True,
    )
    state.camera_names = list(camera_sources)
    update_procedure_waypoint_marker(0, force=True)
    state.physics_authority = load_physics_authority().runtime_payload(
        native_deformable_count=len(deformables),
        runtime_family="isaac-sim-5.1-stable",
        effective_backend=(str(native_room["backend"]) if native_room else None),
    )
    expert_controller = ExpertDemonstrationController(
        procedure_id=str(procedure.get("id", "")),
        guide_kind=guide_kind,
        action_dim=action_dim,
        arms=arms,
        has_grippers=has_grippers,
        waypoints=room_waypoints,
    )
    state.expert_demonstration = expert_controller.snapshot()
    state.runtime_provenance = runtime_provenance(state)
    state.camera_frame_ids = {name: 0 for name in camera_sources}
    state.camera_subscribers = {name: 0 for name in camera_sources}
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

    def reset_environment(selected_scenario: str, selected_seed: int) -> None:
        native_grasp_arms.clear()
        update_procedure_waypoint_marker(0, force=True)
        np.random.seed(selected_seed)
        torch.manual_seed(selected_seed)
        env.reset(seed=selected_seed)
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
        with state.lock:
            selected_view_mode = state.camera_view_mode
        apply_endoscope_camera_view(selected_scenario, selected_view_mode)
        update_wrist_camera_poses()
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

    demo_frames: list[dict[str, np.ndarray]] = []
    vision_frames: list[dict[str, np.ndarray]] = []
    demo_started_at = ""
    demo_started_monotonic = 0.0
    recorded_bytes_estimate = 0
    last_vision_sample_time = 0.0
    last_safety_sample_time = 0.0
    latest_contact_forces: dict[str, float] = {}
    latest_deformable_safety: dict[str, float] = {}
    replay_actions: np.ndarray | None = None
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
        with state.lock:
            reset_requested = state.reset_requested
            state.reset_requested = False
            record_request = state.record_request
            state.record_request = None
            replay_request = state.replay_request
            state.replay_request = None
            expert_request = state.expert_request
            state.expert_request = None
            scenario_id = state.scenario_id
            scenario_seed = state.scenario_seed
            camera_view_request = state.camera_view_request
            state.camera_view_request = None
            ghost_update = state.reference_ghost_update
            state.reference_ghost_update = None
            ghost_enabled = state.reference_ghost_enabled
            if state.drive_until > loop_started:
                manual_action = state.drive.copy()
            elif state.pulse_steps > 0:
                manual_action = state.pulse.copy()
                state.pulse_steps -= 1
            else:
                manual_action = np.zeros(state.action_dim, dtype=np.float32)
            grippers_open = list(state.grippers_open)
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

        if expert_request == "start":
            expert_controller.start()
            with state.lock:
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
            with state.lock:
                state.expert_demonstration = expert_controller.snapshot(state.expert_reference_demo)
                state.expert_clean_run = False
                if state.recording:
                    state.record_request = "stop"

        if camera_view_request is not None and not reset_requested:
            with torch.inference_mode():
                apply_endoscope_camera_view(scenario_id, camera_view_request)

        if record_request == "start":
            demo_frames.clear()
            vision_frames.clear()
            demo_started_at = datetime.now(timezone.utc).isoformat()
            demo_started_monotonic = time.monotonic()
            last_vision_sample_time = 0.0
            recorded_bytes_estimate = 0
            with state.lock:
                state.recording = True
                state.recorded_frames = 0
                state.recorded_bytes_estimate = 0
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
                name = save_demo(state, demo_frames, vision_frames, demo_started_at)
                save_error = None
            except Exception as exc:
                name = None
                save_error = f"Demonstration could not be saved: {exc}"
                traceback.print_exc()
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
                    replay_actions = np.asarray(replay_data["actions"], dtype=np.float32)
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

        if expert_controller.active:
            expert_tools = {
                arm: position
                for arm in range(state.arms)
                if (position := tool_position_for_arm(arm)) is not None
            }
            expert_object = None
            if objects:
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
                thread_tail_position=None,
                needle_points=np.empty((0, 3), dtype=np.float32),
                hoop_passed=False,
                knot_secure=False,
                native_grasp_contact_active=[arm in native_grasp_arms for arm in range(state.arms)],
            )
            action_np = expert_command.action
            grippers_open = expert_command.grippers_open
            if state.has_grippers:
                for arm, is_open in enumerate(grippers_open):
                    action_np[state.gripper_action_index(arm)] = 1.0 if is_open else -1.0
            with state.lock:
                state.grippers_open = list(grippers_open)
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
                for arm, is_open in enumerate(grippers_open):
                    action_np[state.gripper_action_index(arm)] = 1.0 if is_open else -1.0

        grasp_distances: list[float | None] = [None] * state.arms
        grasp_offsets: list[list[float] | None] = [None] * state.arms
        grasp_target_position = None
        if objects:
            grasp_object = next(iter(objects.values()))
            grasp_target_position = grasp_object.data.root_pos_w[0, :3].detach().cpu().numpy().astype(np.float32)
        elif native_tissue is not None:
            nodal_position_value = native_tissue.data.nodal_pos_w
            nodal_positions = getattr(nodal_position_value, "torch", nodal_position_value)
            grasp_target_position = nodal_positions[0].mean(dim=0).detach().cpu().numpy().astype(np.float32)
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
                        native_tissue is not None
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
        with torch.inference_mode():
            write_native_attachment()
            _observations, reward, terminated, truncated, info = env.step(actions)
            update_wrist_camera_poses()
        environment_reward = scalar_value(reward)
        environment_terminated = bool(scalar_value(terminated))
        environment_truncated = bool(scalar_value(truncated))
        environment_success = native_success_from_info(info)


        current_tool_positions = {
            arm: position
            for arm in range(state.arms)
            if (position := tool_position_for_arm(arm)) is not None
        }
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
            }

        motion_active = any(bool(np.any(action_np[state.body_action_slice(arm)])) for arm in range(state.arms))
        current_time = time.monotonic()
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
            camera_stream_active = any(count > 0 for count in state.camera_subscribers.values())
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
            latest_deformable_safety = sample_deformable_safety(deformables)
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
                camera_valid_depth_fraction = state.camera_valid_depth_fraction
                camera_foreground_fraction = state.camera_foreground_fraction
                camera_mean_luminance = state.camera_mean_luminance
                state.procedure_event_code = 0
            frame = {
                "time_s": np.array(time.monotonic() - demo_started_monotonic, dtype=np.float64),
                "actions": action_np.copy(),
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
            for name, robot in robots.items():
                frame[f"{name}_joint_positions"] = robot.data.joint_pos[0].detach().cpu().numpy().copy()
                frame[f"{name}_joint_velocities"] = robot.data.joint_vel[0].detach().cpu().numpy().copy()
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
            for key, value in latest_contact_forces.items():
                frame[key] = np.array(value, dtype=np.float32)
            for key, value in latest_deformable_safety.items():
                frame[key] = np.array(value, dtype=np.float32)
            demo_frames.append(frame)
            recorded_bytes_estimate += array_payload_bytes(frame)
            with state.lock:
                state.recorded_frames = len(demo_frames)
                state.recorded_bytes_estimate = recorded_bytes_estimate
                if (
                    len(demo_frames) >= MAX_DEMO_FRAMES
                    or now - demo_started_monotonic >= MAX_DEMO_SECONDS
                    or recorded_bytes_estimate >= MAX_DEMO_BYTES
                ):
                    state.record_request = "stop"
                    state.coaching_cue = (
                        "The recording reached its configured safety limit and is being saved automatically."
                    )

        now = time.monotonic()
        fps_steps += 1
        frame_interval = 0.04 if interactive_active else 0.20
        if camera.data.output.get("rgb") is not None and now - last_frame_time >= frame_interval:
            rendered_jpegs = {}
            rendered_luminance: float | None = None
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
            requested_cameras.add("endoscope_left")
            for camera_name in requested_cameras:
                sensor_camera = camera_sources.get(camera_name)
                if sensor_camera is None:
                    continue
                camera_output = sensor_camera.data.output.get("rgb")
                if camera_output is not None:
                    jpeg, mean_luminance = encode_jpeg(camera_output[0], scenario_id, dropout_active)
                    rendered_jpegs[camera_name] = jpeg
                    if camera_name == "endoscope_left":
                        rendered_luminance = mean_luminance
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
                for camera_name in ("endoscope_right", "wrist_1", "wrist_2"):
                    sensor_camera = camera_sources.get(camera_name)
                    sensor_rgb = sensor_camera.data.output.get("rgb") if sensor_camera is not None else None
                    if sensor_rgb is not None:
                        sensor_image = rgb_tensor_to_image(sensor_rgb[0], scenario_id, dropout_active).resize(
                            (360, 240), Image.Resampling.BILINEAR
                        )
                        vision_frame[f"{camera_name}_rgb"] = np.asarray(sensor_image, dtype=np.uint8)
                vision_frames.append(vision_frame)
                with state.lock:
                    state.camera_mean_luminance = float(vision_frame["mean_luminance"])
                    if "valid_depth_fraction" in vision_frame:
                        state.camera_valid_depth_fraction = float(vision_frame["valid_depth_fraction"])
                    if "semantic_foreground_fraction" in vision_frame:
                        state.camera_foreground_fraction = float(vision_frame["semantic_foreground_fraction"])
                recorded_bytes_estimate += array_payload_bytes(vision_frame)
                with state.lock:
                    state.recorded_bytes_estimate = recorded_bytes_estimate
                    if recorded_bytes_estimate >= MAX_DEMO_BYTES:
                        state.record_request = "stop"
                        state.coaching_cue = "The recording reached its memory budget and is being saved automatically."
                last_vision_sample_time = now
            frame_count += 1
            elapsed = max(now - last_frame_time, 1e-6)
            with state.lock:
                state.camera_frames_jpeg.update(rendered_jpegs)
                for camera_name in rendered_jpegs:
                    state.camera_frame_ids[camera_name] = state.camera_frame_ids.get(camera_name, 0) + 1
                state.frame_jpeg = rendered_jpegs.get("endoscope_left", state.frame_jpeg)
                state.frame_id += 1
                state.render_fps = 1.0 / elapsed if last_frame_time else 0.0
                if rendered_luminance is not None:
                    state.camera_mean_luminance = rendered_luminance
                    if rendered_luminance > 0.01:
                        state.camera_nonblank_seen = True
            last_frame_time = now
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
            name = save_demo(state, demo_frames, vision_frames, demo_started_at)
            if name:
                state.last_demo = name
        except Exception:
            traceback.print_exc()
    server.should_exit = True
    server_thread.join(timeout=5.0)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
