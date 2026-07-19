# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Browser-operated, simulation-only Dr.Anmar surgical workstation."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import signal
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

from dr_anmar_procedures import PROCEDURES_BY_ID


DATA_ROOT = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()

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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils import parse_env_cfg

import orbit.surgical.tasks  # noqa: F401


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
    "anatomy_context": {"show_multi_organ": True},
}
RESEARCH_ADVISORY_LIMITS = {
    "contact_force_n": 2.0,
    "tissue_displacement_m": 0.015,
    "deformation_gradient_proxy": 0.50,
}
PROCEDURE_PHASES = {"setup": 0, "approach": 1, "grasp": 2, "manipulation": 3, "recovery": 4}
PROCEDURE_EVENTS = {"none": 0, "target_visible": 1, "contact": 2, "grasp": 3, "task_complete": 4, "handoff": 5, "safety_review": 6}
OPERATOR_INPUT_SOURCES = {"none": 0, "keyboard_pointer": 1, "gamepad": 2, "external_teleop": 3, "xr": 4, "haptic": 5}


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Dr.Anmar Surgical Workstation</title>
  <style>
    :root{color-scheme:dark;--bg:#071016;--panel:#0d1a22;--line:#24404d;--cyan:#2cd2e8;--cyan2:#1795ae;--ink:#e9f8fa;--muted:#88a6b2;--red:#ff5c68;--green:#42e49b}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.35 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI";min-height:100vh}
    header{height:58px;display:flex;align-items:center;gap:16px;padding:0 20px;border-bottom:1px solid var(--line);background:#09151c}
    .brand{font-weight:900;letter-spacing:.08em}.brand span{color:var(--cyan)}.tag{font-size:11px;font-weight:800;letter-spacing:.08em;padding:5px 9px;border-radius:99px;border:1px solid #76505a;color:#ff9aa2;background:#2b151b}
    .live{margin-left:auto;display:flex;align-items:center;gap:7px;color:var(--muted);font-size:13px}.dot{width:9px;height:9px;border-radius:50%;background:var(--red)}.dot.ok{background:var(--green);box-shadow:0 0 12px #42e49b99}
    main{display:grid;grid-template-columns:minmax(0,1fr) 400px;height:calc(100vh - 58px)}
    .view{position:relative;overflow:hidden;background:#020608;display:flex;align-items:center;justify-content:center}.view img{width:100%;height:100%;object-fit:contain}.camera-tabs{position:absolute;left:50%;bottom:16px;translate:-50% 0;display:flex;gap:5px;padding:5px;background:#061118cc;border:1px solid #ffffff24;border-radius:8px}.camera-tabs button,.view-presets button{min-height:31px;padding:0 10px;font-size:10px}.camera-tabs button.active,.view-presets button.active{background:var(--cyan);color:#031014}.view-presets{position:absolute;right:16px;bottom:16px;display:flex;gap:5px;padding:5px;background:#061118cc;border:1px solid #ffffff24;border-radius:8px}.gaze-cursor{position:absolute;width:18px;height:18px;border:1px solid #fff;border-radius:50%;translate:-50% -50%;pointer-events:none;opacity:0;box-shadow:0 0 0 3px #2cd2e855}.view.gaze-on .gaze-cursor{opacity:.85}
    .aim-reticle{position:absolute;left:50%;top:50%;width:34px;height:34px;translate:-50% -50%;pointer-events:none;opacity:.48}.aim-reticle:before,.aim-reticle:after{content:"";position:absolute;background:#dffcff}.aim-reticle:before{left:0;right:0;top:16px;height:1px}.aim-reticle:after{top:0;bottom:0;left:16px;width:1px}.proximity{position:absolute;right:16px;top:16px;min-width:180px;padding:9px 11px;border:1px solid #ffffff24;border-radius:8px;background:#051016d9;color:#9fc0c9;font:11px/1.5 ui-monospace,SFMono-Regular,Menlo;backdrop-filter:blur(6px)}.proximity b{display:block;color:var(--ink);font-size:13px}.proximity.near{border-color:#ffd978}.proximity.held{border-color:var(--green);color:var(--green)}.proximity.guard{border-color:var(--cyan);color:var(--cyan)}.proximity.puncture{border-color:#ff9b83;color:#ffb09e;box-shadow:0 0 18px #ff775533}
    .hud{position:absolute;left:16px;top:16px;padding:10px 13px;border:1px solid #ffffff24;border-radius:8px;background:#051016c9;backdrop-filter:blur(6px);font:12px/1.5 ui-monospace,SFMono-Regular,Menlo;color:#cfe7eb}.hud strong{color:var(--cyan)}
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
    .control-dock{position:relative;margin:2px 0 18px;padding:42px 13px 4px;border:2px solid var(--cyan);border-radius:13px;background:linear-gradient(145deg,#0d2630,#08151c 68%);box-shadow:0 0 0 1px #2cd2e829,0 0 24px #2cd2e820}.control-dock:before{content:"INSTRUMENT CONTROL";position:absolute;left:14px;top:12px;color:var(--cyan);font:900 13px/1 ui-sans-serif,system-ui;letter-spacing:.15em}.control-dock:after{content:"LIVE";position:absolute;right:14px;top:9px;padding:4px 7px;border-radius:99px;background:#123a32;color:var(--green);font:900 9px/1 ui-monospace,monospace;letter-spacing:.12em}.control-dock h2{color:#d7f9fc;margin-top:8px}.control-dock .card{border-color:#356675;background:#07151c;margin-bottom:10px}.control-dock .move-button{min-height:62px;border-width:2px;border-color:#3d6d7b;font-size:14px}.control-dock .stop-center{border-width:2px}.control-dock #gripperPanel .primary{box-shadow:0 0 18px #2cd2e82e}.control-dock .hint{font-size:10px}
    #toast{position:fixed;left:50%;bottom:20px;translate:-50% 20px;opacity:0;background:#e9f8fa;color:#061116;border-radius:8px;padding:10px 15px;font-weight:750;transition:.2s;pointer-events:none}#toast.show{opacity:1;translate:-50% 0}
    @media(max-width:1100px){.view-presets{right:10px;top:74px;bottom:auto;flex-direction:column}.proximity{right:10px;top:10px;min-width:155px}.camera-tabs{bottom:10px}}
    @media(max-width:880px){main{display:block;height:auto}.view{height:52vh}aside{border-left:0;border-top:1px solid var(--line)}header{padding:0 12px}.tag{display:none}.view-presets{display:none}.proximity{font-size:9px;min-width:130px}}
  </style>
</head>
<body>
<header><div class="brand">DR.<span>ANMAR</span></div><div class="tag">SIMULATION ONLY · NO PHYSICAL ROBOT OUTPUT</div><div class="live"><i id="dot" class="dot"></i><span id="connection">Connecting…</span></div></header>
<main>
  <section id="cameraView" class="view"><img id="cameraImage" src="/video/endoscope_left" alt="Live simulated medical sensor view"><div class="hud"><strong id="cameraLabel">STEREO ENDOSCOPE · LEFT</strong><br><span id="hud">Waiting for Isaac Lab…</span></div><div id="recflag" class="recflag">● RECORDING</div><div id="gazeCursor" class="gaze-cursor"></div><div class="aim-reticle"></div><div id="proximity" class="proximity"><b>Tool guidance</b><span>Acquiring target…</span></div><div class="camera-tabs"><button class="active" data-camera="endoscope_left" onclick="setCamera('endoscope_left',this)">Stereo left</button><button data-camera="endoscope_right" onclick="setCamera('endoscope_right',this)">Stereo right</button><button data-camera="wrist_1" onclick="setCamera('wrist_1',this)">Wrist 1</button><button id="wrist2Tab" class="hidden" data-camera="wrist_2" onclick="setCamera('wrist_2',this)">Wrist 2</button></div><div class="view-presets"><button class="active" data-view-mode="operative" onclick="setCameraView('operative',this)">Operative</button><button data-view-mode="close" onclick="setCameraView('close',this)">Close</button><button data-view-mode="overview" onclick="setCameraView('overview',this)">Overview</button></div></section>
  <aside>
    <section class="control-dock">
      <div id="armPanel" class="hidden"><h2>Active instrument</h2><div class="card"><div class="grid two"><button id="arm0" class="arm active" onclick="setArm(0)">Instrument 1</button><button id="arm1" class="arm" onclick="setArm(1)">Instrument 2</button></div></div></div>
      <h2>Movement speed</h2><div class="card"><div class="speedbar"><button data-speed="0.35" onclick="setSpeed(0.35,this)">Precision</button><button class="active" data-speed="1" onclick="setSpeed(1,this)">Normal</button><button data-speed="1.7" onclick="setSpeed(1.7,this)">Fast</button></div><div class="hint">Fast in open space; automatic precision near the target.</div></div>
      <h2>Tool position</h2><div class="card"><div class="dpad">
        <button class="move-button up" data-axis="2" data-direction="1">↑ Up<small>R</small></button>
        <button class="move-button left" data-axis="1" data-direction="1">← Left<small>A</small></button>
        <button class="stop-center" onclick="stopTool()">■ Stop<small>Esc</small></button>
        <button class="move-button right" data-axis="1" data-direction="-1">Right →<small>D</small></button>
        <button class="move-button down" data-axis="2" data-direction="-1">↓ Down<small>F</small></button>
      </div><div class="depthgrid"><button class="move-button" data-axis="0" data-direction="-1">Toward patient<small>W</small></button><button class="move-button" data-axis="0" data-direction="1">Away from patient<small>S</small></button></div><div id="controlReadout" class="control-readout"><i></i><span>Ready · hold a control to move</span></div></div>
      <div id="gripperPanel"><h2>Gripper</h2><div class="card"><div class="grid two"><button onclick="grip(true)">Open jaws</button><button class="primary" onclick="grip(false)">Close / grasp</button></div><div class="hint">Space or gamepad A toggles the jaws.</div></div></div>
      <h2>Tool angle</h2><div class="card"><div class="anglegrid">
        <button class="move-button" data-axis="3" data-direction="-1">↶ Roll left<small>Q</small></button><button class="move-button" data-axis="3" data-direction="1">Roll right ↷<small>E</small></button>
        <button class="move-button" data-axis="4" data-direction="-1">Pitch up<small>↑</small></button><button class="move-button" data-axis="4" data-direction="1">Pitch down<small>↓</small></button>
        <button class="move-button" data-axis="5" data-direction="-1">← Yaw left<small>←</small></button><button class="move-button" data-axis="5" data-direction="1">Yaw right →<small>→</small></button>
      </div><div class="hint">WASD + R/F position · arrows + Q/E angle · C camera · Space gripper · Esc stop.</div></div>
    </section>
    <h2>Procedure room</h2><div class="card"><div id="procedureTitle" class="procedure-title">Free practice</div><div id="procedureObjective" class="procedure-objective">Use the robot controls to explore the digital twin.</div><div class="procedure-progress"><i id="procedureProgress"></i></div><div id="procedureSteps"></div><div id="procedureTruth" class="fidelity-note hidden"></div></div>
    <h2>Supervision</h2><div class="card supervision"><div class="supervision-state"><span>Autonomy level</span><b id="autonomyState">L0 · Manual</b></div><div class="grid two"><button id="manualMode" class="autonomy active" onclick="setAutonomy('manual')">Manual</button><button id="guidedMode" class="autonomy" onclick="setAutonomy('guided')">Guided</button></div><button class="take-control" onclick="takeControl()">Take control now</button><div id="coachingCue" class="cue">You command every movement. Dr.Anmar records telemetry for coaching.</div></div>
    <h2>Expert path guide</h2><div class="card"><div class="grid two"><button class="primary" onclick="referenceGhost(true)">Show clinician path</button><button onclick="referenceGhost(false)">Hide path</button></div><div id="ghostState" class="ghost-state">Select a clinician reference in Skills Twin first.</div></div>
    <h2>Procedure annotation</h2><div class="card"><div class="grid"><button onclick="annotatePhase('approach')">Approach</button><button onclick="annotatePhase('grasp')">Grasp</button><button onclick="annotatePhase('manipulation')">Manipulate</button><button onclick="annotatePhase('recovery')">Recovery</button><button onclick="annotateEvent('task_complete')">Task event</button><button onclick="annotateEvent('safety_review')">Safety event</button></div><div class="hint">Phase labels and events are synchronized into the training trajectory.</div></div>
    <h2>Research safety monitor</h2><div class="card"><div class="safety-grid"><div class="safety-metric"><b id="forceMetric">—</b><span>CONTACT N</span></div><div class="safety-metric"><b id="deformMetric">—</b><span>TISSUE MM</span></div><div class="safety-metric"><b id="stressMetric">—</b><span>STRESS PA</span></div></div><div class="hint">Simulator signals only. Limits are engineering advisories, not clinical thresholds.</div></div>
    <h2>Demonstration</h2><div class="card"><div class="grid two"><button id="record" class="primary" onclick="recording(true)">Start recording</button><button onclick="recording(false)">Stop & save</button><button onclick="replay()">Replay last</button><button onclick="resetScene()">Reset scene</button></div><div class="hint" id="lastDemo">Actions, joints, RGB-D, segmentation, object state, and safety telemetry are saved together.</div></div>
    <h2>System</h2><div class="card status" id="status">Starting…</div>
  </aside>
</main><div id="toast"></div>
<script>
const keyMap={w:[0,-1],s:[0,1],a:[1,1],d:[1,-1],r:[2,1],f:[2,-1],q:[3,-1],e:[3,1],arrowup:[4,-1],arrowdown:[4,1],arrowleft:[5,-1],arrowright:[5,1]};
let activeArm=0,driveSpeed=1,driveInFlight=false,queuedDrive=null,driveWasActive=false,inputSource='keyboard_pointer',lastGazeSend=0,currentCamera='endoscope_left',currentViewMode='operative',lastGamepadGrip=false,lastGamepadCamera=false;
const heldKeys=new Set(),pointerMoves=new Map();
async function post(url,body={}){const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.detail||'Request failed');return data}
function toast(s){const e=document.getElementById('toast');e.textContent=s;e.classList.add('show');setTimeout(()=>e.classList.remove('show'),1600)}
function setArm(arm){stopDrive(false);activeArm=arm;document.getElementById('arm0').classList.toggle('active',arm===0);document.getElementById('arm1').classList.toggle('active',arm===1)}
function setSpeed(speed,button){driveSpeed=speed;document.querySelectorAll('[data-speed]').forEach(x=>x.classList.toggle('active',x===button));updateControlReadout(false,`${button.textContent} speed`)}
function deadzone(value){return Math.abs(value)<0.18?0:Math.sign(value)*(Math.abs(value)-0.18)/0.82}
function gamepadDrive(){const values=Array(6).fill(0);const pads=navigator.getGamepads?navigator.getGamepads():[];const pad=[...pads].find(Boolean);if(!pad){lastGamepadGrip=false;lastGamepadCamera=false;return values}inputSource='gamepad';values[1]-=deadzone(pad.axes[0]||0);values[0]+=deadzone(pad.axes[1]||0);values[5]+=deadzone(pad.axes[2]||0);values[4]+=deadzone(pad.axes[3]||0);values[3]+=(pad.buttons[5]?.value||0)-(pad.buttons[4]?.value||0);values[2]+=(pad.buttons[7]?.value||0)-(pad.buttons[6]?.value||0);const gripPressed=!!pad.buttons[0]?.pressed,cameraPressed=!!pad.buttons[3]?.pressed;if(gripPressed&&!lastGamepadGrip)toggleGrip();if(cameraPressed&&!lastGamepadCamera)cycleCameraView();lastGamepadGrip=gripPressed;lastGamepadCamera=cameraPressed;return values}
function buildDrive(){const values=gamepadDrive();heldKeys.forEach(key=>{const move=keyMap[key];if(move)values[move[0]]+=move[1]});pointerMoves.forEach(move=>values[move.axis]+=move.direction);return values.map(value=>Math.max(-1,Math.min(1,value)))}
function updateControlReadout(moving,label){const readout=document.getElementById('controlReadout');readout.classList.toggle('moving',moving);readout.querySelector('span').textContent=moving?(label||'Moving · release to stop'):'Ready · hold a control to move'}
async function flushDrive(){if(driveInFlight||!queuedDrive)return;const next=queuedDrive;queuedDrive=null;driveInFlight=true;try{await post('/api/drive',{values:next,arm:activeArm,speed:driveSpeed,source:inputSource})}catch(e){toast(e.message)}finally{driveInFlight=false;if(queuedDrive)flushDrive()}}
function sendDrive(values){queuedDrive=values;flushDrive()}
function updateDrive(){const values=buildDrive();const active=values.some(value=>Math.abs(value)>0.01);if(active||driveWasActive)sendDrive(values);driveWasActive=active;updateControlReadout(active,active?'Moving · release to stop':null)}
function stopDrive(showToast=true){heldKeys.clear();pointerMoves.clear();document.querySelectorAll('.move-button.held').forEach(x=>x.classList.remove('held'));driveWasActive=false;sendDrive(Array(6).fill(0));updateControlReadout(false);if(showToast)toast('Tool stopped')}
async function stopTool(){stopDrive();try{await post('/api/stop')}catch(e){toast(e.message)}}
async function grip(open){try{await post('/api/gripper',{open,arm:activeArm});toast(open?'Gripper open':'Gripper closed')}catch(e){toast(e.message)}}
async function toggleGrip(){try{const result=await post('/api/gripper/toggle',{arm:activeArm});toast(result.open?'Gripper open':'Gripper closed')}catch(e){toast(e.message)}}
async function recording(start){try{await post(start?'/api/record/start':'/api/record/stop');toast(start?'Recording started':'Saving demonstration…')}catch(e){toast(e.message)}}
async function replay(){try{const x=await post('/api/replay-last');toast(x.message)}catch(e){toast(e.message)}}
async function referenceGhost(enabled){try{const x=await post('/api/reference-ghost',{enabled});toast(x.message)}catch(e){toast(e.message)}}
function setCamera(name,button){currentCamera=name;document.getElementById('cameraImage').src=`/video/${name}?t=${Date.now()}`;document.querySelectorAll('[data-camera]').forEach(x=>x.classList.toggle('active',x===button));const labels={endoscope_left:'STEREO ENDOSCOPE · LEFT',endoscope_right:'STEREO ENDOSCOPE · RIGHT',wrist_1:'INSTRUMENT WRIST · 1',wrist_2:'INSTRUMENT WRIST · 2'};document.getElementById('cameraLabel').textContent=labels[name]||name.toUpperCase()}
async function setCameraView(mode,button){try{const result=await post('/api/camera-view',{mode});currentViewMode=result.mode;document.querySelectorAll('[data-view-mode]').forEach(x=>x.classList.toggle('active',x.dataset.viewMode===result.mode));toast(`${button?.textContent||result.mode} camera ready`)}catch(e){toast(e.message)}}
function cycleCameraView(){const modes=['operative','close','overview'],mode=modes[(modes.indexOf(currentViewMode)+1)%modes.length],button=document.querySelector(`[data-view-mode="${mode}"]`);setCameraView(mode,button)}
async function annotatePhase(phase){try{const x=await post('/api/annotation',{phase});toast(x.message)}catch(e){toast(e.message)}}
async function annotateEvent(event){try{const x=await post('/api/annotation',{event});toast('Procedure event saved')}catch(e){toast(e.message)}}
async function resetScene(){try{await post('/api/reset');toast('Scene reset')}catch(e){toast(e.message)}}
async function setAutonomy(mode){try{const x=await post('/api/autonomy',{mode});toast(x.message)}catch(e){toast(e.message)}}
async function takeControl(){stopDrive(false);try{const x=await post('/api/handoff');toast(x.message)}catch(e){toast(e.message)}}
document.querySelectorAll('.move-button').forEach(button=>{button.addEventListener('pointerdown',event=>{event.preventDefault();inputSource='keyboard_pointer';button.setPointerCapture(event.pointerId);pointerMoves.set(event.pointerId,{axis:Number(button.dataset.axis),direction:Number(button.dataset.direction),button});button.classList.add('held');updateDrive()});const release=event=>{const move=pointerMoves.get(event.pointerId);pointerMoves.delete(event.pointerId);if(move&&![...pointerMoves.values()].some(x=>x.button===move.button))move.button.classList.remove('held');updateDrive()};button.addEventListener('pointerup',release);button.addEventListener('pointercancel',release);button.addEventListener('lostpointercapture',release);button.addEventListener('contextmenu',event=>event.preventDefault())});
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;const key=event.key.toLowerCase();if(event.code==='Space'){event.preventDefault();if(!event.repeat)toggleGrip();return}if(key==='c'){event.preventDefault();if(!event.repeat)cycleCameraView();return}if(key==='escape'){event.preventDefault();stopTool();return}if(!keyMap[key])return;event.preventDefault();inputSource='keyboard_pointer';heldKeys.add(key);updateDrive()});
document.addEventListener('keyup',event=>{if(event.code==='Space'){event.preventDefault();return}const key=event.key.toLowerCase();if(!keyMap[key])return;event.preventDefault();heldKeys.delete(key);updateDrive()});
window.addEventListener('blur',()=>stopDrive(false));document.addEventListener('visibilitychange',()=>{if(document.hidden)stopDrive(false)});
document.getElementById('cameraView').addEventListener('pointermove',event=>{const view=event.currentTarget,rect=view.getBoundingClientRect();const u=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),v=Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height));const cursor=document.getElementById('gazeCursor');cursor.style.left=`${u*100}%`;cursor.style.top=`${v*100}%`;view.classList.add('gaze-on');const now=performance.now();if(now-lastGazeSend>100){lastGazeSend=now;post('/api/gaze',{u,v,valid:true,source:'pointer_attention_proxy'}).catch(()=>{})}});document.getElementById('cameraView').addEventListener('pointerleave',()=>document.getElementById('cameraView').classList.remove('gaze-on'));
function targetDirections(offset){if(!offset)return'';const choices=[];if(Math.abs(offset[2])>.004)choices.push([Math.abs(offset[2]),offset[2]>0?'Up':'Down']);if(Math.abs(offset[1])>.004)choices.push([Math.abs(offset[1]),offset[1]>0?'Left':'Right']);if(Math.abs(offset[0])>.004)choices.push([Math.abs(offset[0]),offset[0]<0?'Toward':'Away']);return choices.sort((a,b)=>b[0]-a[0]).slice(0,2).map(x=>x[1]).join(' + ')}
async function refresh(){try{
  const s=await(await fetch('/api/status',{cache:'no-store'})).json();document.getElementById('dot').classList.add('ok');document.getElementById('connection').textContent='Isaac Lab live';
  const p=s.procedure||{};document.getElementById('procedureTitle').textContent=p.title||'Free practice';document.getElementById('procedureObjective').textContent=p.objective||'Use the robot controls to explore the digital twin.';document.getElementById('procedureProgress').style.width=`${p.progress_percent||0}%`;document.getElementById('procedureSteps').innerHTML=(p.steps||[]).map((x,i)=>`<div class="procedure-step ${x.status}"><span>${String(i+1).padStart(2,'0')}</span><div><b>${x.title}</b><br>${x.instruction}</div></div>`).join('');
  const truth=document.getElementById('procedureTruth');truth.textContent=p.truth_note||'';truth.classList.toggle('hidden',!p.truth_note);document.querySelectorAll('[data-camera]').forEach(button=>button.classList.toggle('hidden',!s.camera_names.includes(button.dataset.camera)));document.getElementById('armPanel').classList.toggle('hidden',s.arms<2);document.getElementById('gripperPanel').classList.toggle('hidden',!s.has_grippers);
  currentViewMode=s.camera_view_mode||currentViewMode;document.querySelectorAll('[data-view-mode]').forEach(x=>x.classList.toggle('active',x.dataset.viewMode===currentViewMode));
  const grip=s.has_grippers?(s.grippers_open[activeArm]?' · GRIPPER OPEN':' · GRIPPER CLOSED'):'',moving=s.drive_active?' · MOVING':'';document.getElementById('hud').innerHTML=`<strong>${s.anatomy_showcase||'SURGICAL WORKSPACE'}</strong><br>${s.camera_width}×${s.camera_height} · ${s.render_fps.toFixed(1)} FPS · ${currentViewMode.toUpperCase()}<br>${p.title||s.scenario_title}${grip}${moving}`;document.getElementById('recflag').classList.toggle('on',s.recording);
  const proximity=document.getElementById('proximity'),distance=s.tool_to_object_distance_m?.[activeArm],offset=s.tool_to_object_offset_m?.[activeArm],clearance=s.closest_anatomy_clearance_m,tipClearance=s.needle_tip_clearance_m,depth=s.needle_penetration_depth_m||0;proximity.className='proximity';let guidance='Move toward the target';if(s.needle_puncture_active){guidance=`Needle inserted ${Math.round(depth*1000)} mm · rotate through the arc`;proximity.classList.add('puncture')}else if(s.assisted_grasp_active?.[activeArm]&&tipClearance!==null&&tipClearance!==undefined&&tipClearance<=.006){guidance=`Needle tip ${Math.max(0,Math.round(tipClearance*1000))} mm from tissue · advance gently`;proximity.classList.add('near')}else if(s.assisted_grasp_active?.[activeArm]){guidance='Needle held · Space releases';proximity.classList.add('held')}else if(s.virtual_fixture_active){guidance='Instrument boundary · tangential motion only';proximity.classList.add('guard')}else if(distance!==null&&distance!==undefined&&distance<=s.grasp_capture_radius_m){guidance=`Aligned ${Math.round(distance*1000)} mm · close jaws`;proximity.classList.add('near')}else if(distance!==null&&distance!==undefined){const direction=targetDirections(offset);guidance=`Target ${Math.round(distance*1000)} mm · ${direction||'hold course'}${s.adaptive_precision_active?' · auto precision':''}`}else if(clearance!==null&&clearance!==undefined){guidance=`Anatomy clearance ${Math.round(clearance*1000)} mm`};proximity.innerHTML=`<b>Tool guidance</b><span>${guidance}</span>`;
  const labels={manual:'L0 · Manual',guided:'L1 · Guided',supervised_replay:'L2 · Supervised replay'};document.getElementById('autonomyState').textContent=labels[s.autonomy_mode]||s.autonomy_mode;document.getElementById('manualMode').classList.toggle('active',s.autonomy_mode==='manual');document.getElementById('guidedMode').classList.toggle('active',s.autonomy_mode==='guided');document.getElementById('coachingCue').textContent=s.coaching_cue;document.getElementById('forceMetric').textContent=s.safety?.max_contact_force_n===null?'—':Number(s.safety.max_contact_force_n).toFixed(2);document.getElementById('deformMetric').textContent=s.safety?.max_tissue_displacement_m===null?'—':(Number(s.safety.max_tissue_displacement_m)*1000).toFixed(1);document.getElementById('stressMetric').textContent=s.safety?.max_tissue_stress_pa===null?'—':Number(s.safety.max_tissue_stress_pa).toExponential(1);
  const ghost=document.getElementById('ghostState');ghost.classList.toggle('on',!!s.reference_ghost?.enabled);ghost.textContent=s.reference_ghost?.enabled?`${s.reference_ghost.point_count} registered path points · ${s.reference_ghost.reference}`:'Clinician path hidden';document.getElementById('status').innerHTML=`Task<br><b>${s.task}</b><br>Procedure: ${p.title||'Free practice'}<br>Anatomy: ${s.anatomy_showcase||'—'}<br>Scenario: ${s.scenario_title}<br>Robots: ${s.robot_names.join(', ')}<br>Autonomy: ${labels[s.autonomy_mode]||s.autonomy_mode}<br>Phase: ${s.operator_study.procedure_phase}<br>Input: ${s.operator_study.input_source}<br>Annotations: ${s.operator_study.annotation_count}<br>Interventions: ${s.intervention_count}<br>Simulation: ${s.sim_fps.toFixed(1)} Hz<br>Controls: ${s.drive_active?'moving':'ready'}<br>Instrument guard: ${s.virtual_fixture_enabled?'on':'off'}<br>Needle entry: ${s.needle_puncture_active?`${Math.round(depth*1000)} mm`:'ready'}<br>Recorded frames: ${s.recorded_frames}<br>Replay: ${s.replaying?'running':'idle'}`;if(s.last_demo)document.getElementById('lastDemo').innerHTML=`Last saved: <a href="/demos/${s.last_demo}" style="color:#2cd2e8">${s.last_demo}</a>`;
}catch(e){document.getElementById('dot').classList.remove('ok');document.getElementById('connection').textContent='Reconnecting…'}}
setInterval(updateDrive,90);setInterval(refresh,500);refresh();
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
    organ_proxy_visual_ready: bool = False
    anatomy_collision_meshes: int = 0
    instance_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    lock: threading.Lock = field(default_factory=threading.Lock)
    wake_event: threading.Event = field(default_factory=threading.Event)
    frame_jpeg: bytes = b""
    frame_id: int = 0
    camera_frames_jpeg: dict[str, bytes] = field(default_factory=dict)
    camera_frame_ids: dict[str, int] = field(default_factory=dict)
    camera_names: list[str] = field(default_factory=list)
    render_fps: float = 0.0
    sim_fps: float = 0.0
    sim_step: int = 0
    pulse: np.ndarray = field(init=False)
    pulse_steps: int = 0
    drive: np.ndarray = field(init=False)
    drive_until: float = 0.0
    grippers_open: list[bool] = field(init=False)
    assisted_grasp_active: list[bool] = field(init=False)
    tool_to_object_distance_m: list[float | None] = field(init=False)
    tool_to_object_offset_m: list[list[float] | None] = field(init=False)
    grasp_capture_radius_m: float = 0.018
    camera_view_mode: str = "operative"
    camera_view_request: str | None = None
    virtual_fixture_enabled: bool = True
    virtual_fixture_active: bool = False
    closest_anatomy_clearance_m: float | None = None
    needle_tip_clearance_m: float | None = None
    needle_puncture_active: bool = False
    needle_penetration_depth_m: float = 0.0
    needle_max_penetration_m: float = 0.012
    adaptive_precision_active: bool = False
    reset_requested: bool = False
    record_request: str | None = None
    recording: bool = False
    recorded_frames: int = 0
    last_demo: str | None = None
    replay_request: str | None = None
    replaying: bool = False
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
    procedure_events: list[dict[str, Any]] = field(default_factory=list)
    procedure_waypoints_total: int = 0
    procedure_waypoints_completed: int = 0
    procedure_motion_seen: bool = False
    procedure_grasp_seen: bool = False
    procedure_object_lift_m: float = 0.0
    procedure_object_motion_m: float = 0.0
    procedure_started_at: float = 0.0
    procedure_last_motion_at: float = 0.0

    def __post_init__(self) -> None:
        self.pulse = np.zeros(self.action_dim, dtype=np.float32)
        self.drive = np.zeros(self.action_dim, dtype=np.float32)
        self.grippers_open = [True] * self.arms
        self.assisted_grasp_active = [False] * self.arms
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
            return {
                "task": self.task,
                "instance_id": self.instance_id,
                "camera_width": self.camera_width,
                "camera_height": self.camera_height,
                "camera_names": self.camera_names,
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
                "organ_proxy_visual_ready": self.organ_proxy_visual_ready,
                "anatomy_collision_meshes": self.anatomy_collision_meshes,
                "procedure": procedure_status,
                "grippers_open": self.grippers_open,
                "assisted_grasp_active": self.assisted_grasp_active,
                "tool_to_object_distance_m": self.tool_to_object_distance_m,
                "tool_to_object_offset_m": self.tool_to_object_offset_m,
                "grasp_capture_radius_m": self.grasp_capture_radius_m,
                "camera_view_mode": self.camera_view_mode,
                "virtual_fixture_enabled": self.virtual_fixture_enabled,
                "virtual_fixture_active": self.virtual_fixture_active,
                "closest_anatomy_clearance_m": self.closest_anatomy_clearance_m,
                "needle_tip_clearance_m": self.needle_tip_clearance_m,
                "needle_puncture_active": self.needle_puncture_active,
                "needle_penetration_depth_m": self.needle_penetration_depth_m,
                "needle_max_penetration_m": self.needle_max_penetration_m,
                "adaptive_precision_active": self.adaptive_precision_active,
                "recording": self.recording,
                "recorded_frames": self.recorded_frames,
                "last_demo": self.last_demo,
                "replaying": self.replaying,
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
                "operator_study": {
                    "gaze_valid": self.gaze_valid,
                    "gaze_source": self.gaze_source,
                    "input_source": self.operator_input_source,
                    "procedure_phase": self.procedure_phase,
                    "annotation_count": len(self.procedure_events),
                },
                "drive_active": self.drive_until > time.monotonic() and bool(np.any(self.drive)),
            }

    def _procedure_status(self) -> dict[str, Any]:
        if not self.procedure:
            return {}
        now = time.monotonic()
        kind = self.procedure.get("guide_kind")
        step_count = len(self.procedure.get("steps", []))
        if kind in {"threading", "cutting_path", "navigation"}:
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
        }


def build_web_app(state: SharedState) -> FastAPI:
    app = FastAPI(title="Dr.Anmar Surgical Workstation", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return APP_HTML

    @app.get("/api/status")
    def status() -> JSONResponse:
        return JSONResponse(state.status())

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
        scales = np.asarray((0.006, 0.006, 0.006, 0.03, 0.03, 0.03), dtype=np.float32)
        command[state.body_action_slice(request.arm)] = calibrated_values * scales * request.speed
        active = bool(np.any(values))
        with state.lock:
            state.drive = command
            state.operator_input_source = request.source
            state.drive_until = time.monotonic() + 0.30 if active else 0.0
            if active:
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
        return {"ok": True, "active": active, "action": command.tolist(), "expires_ms": 300}

    @app.post("/api/stop")
    def stop() -> dict[str, bool]:
        with state.lock:
            state.pulse.fill(0.0)
            state.pulse_steps = 0
            state.drive.fill(0.0)
            state.drive_until = 0.0
            state.replay_request = "stop"
        state.wake_event.set()
        return {"ok": True}

    @app.post("/api/camera-view")
    def camera_view(request: CameraViewRequest) -> dict[str, Any]:
        if request.mode not in {"operative", "close", "overview"}:
            raise HTTPException(400, "camera view must be operative, close, or overview")
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
        if not (state.demo_dir / request.demo).is_file():
            raise HTTPException(404, "Demonstration not found")
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
            was_automatic = state.replaying or state.autonomy_mode == "supervised_replay"
            if was_automatic:
                state.intervention_count += 1
            state.replay_request = "stop"
            state.replaying = False
            state.autonomy_mode = "manual"
            state.drive.fill(0.0)
            state.drive_until = 0.0
            if state.evaluation_status in {"running", "saving"}:
                state.evaluation_status = "interrupted"
                state.record_request = "stop"
            state.coaching_cue = "Control returned to the doctor. The intervention is recorded in this session."
        state.wake_event.set()
        return {"ok": True, "intervention_recorded": was_automatic, "message": "Manual control restored immediately"}

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
        if not (state.demo_dir / name).is_file():
            raise HTTPException(404, "Demonstration not found")
        with state.lock:
            if state.recording:
                raise HTTPException(409, "Stop the recording before replaying")
            state.replay_request = name
            state.autonomy_mode = "supervised_replay"
            state.coaching_cue = "The selected behavior is running under supervision. Take control at any time."
        state.wake_event.set()
        return {"ok": True, "message": f"Replaying {name}"}

    @app.get("/api/demos")
    def demos() -> dict[str, Any]:
        files = sorted(state.demo_dir.glob("dr_anmar_*.npz"), reverse=True)
        references = read_reference_map(state.demo_dir)
        items = []
        for path in files[:50]:
            manifest = read_demo_manifest(path)
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
                    "is_reference": bool(task and references.get(task) == path.name),
                }
            )
        return {"demos": items, "analysis_notice": "Telemetry-derived research coaching; clinician validation is pending."}

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
            annotation = {
                "time": datetime.now(timezone.utc).isoformat(),
                "recorded_frame": state.recorded_frames,
                "sim_step": state.sim_step,
                "phase": state.procedure_phase,
                "event": request.event,
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
        manifest = read_demo_manifest(candidate)
        task = manifest.get("task")
        reference_name = read_reference_map(state.demo_dir).get(task)
        if not reference_name:
            raise HTTPException(404, "Select a clinician reference for this task first")
        reference = state.demo_dir / reference_name
        if not reference.is_file():
            raise HTTPException(404, "The selected clinician reference file is missing")
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
            while True:
                with state.lock:
                    frame_id = state.frame_id
                    jpeg = state.frame_jpeg
                if jpeg and frame_id != last_id:
                    last_id = frame_id
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(0.04)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/video/{camera_name}")
    async def camera_video(camera_name: str) -> StreamingResponse:
        if camera_name not in state.camera_names:
            raise HTTPException(404, "Unknown simulated camera")

        async def frames():
            last_id = -1
            while True:
                with state.lock:
                    frame_id = state.camera_frame_ids.get(camera_name, -1)
                    jpeg = state.camera_frames_jpeg.get(camera_name, b"")
                if jpeg and frame_id != last_id:
                    last_id = frame_id
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(0.04)

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


def rgb_tensor_to_image(rgb: torch.Tensor, scenario_id: str = "baseline") -> Image.Image:
    array = rgb[..., :3].detach().cpu().numpy()
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8, copy=False)
    return apply_visual_scenario(Image.fromarray(array), scenario_id)


def encode_jpeg(rgb: torch.Tensor, scenario_id: str = "baseline") -> bytes:
    buffer = io.BytesIO()
    rgb_tensor_to_image(rgb, scenario_id).save(buffer, "JPEG", quality=86, optimize=False)
    return buffer.getvalue()


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
    with np.load(candidate_path) as candidate_data, np.load(reference_path) as reference_data:
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
    preferred_tip_names = ("psm_tool_tip_link", "endo360_needle", "ecm_end_link", "tool_tip", "end_effector")
    with np.load(path) as data:
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
            path_lengths = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=2), axis=0)
            body_index = int(np.argmax(path_lengths))
            legacy_candidates.append((float(path_lengths[body_index]), positions[:, body_index, :3]))
    if not candidates:
        if not legacy_candidates:
            raise ValueError("The reference does not contain a robot body trajectory")
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
    robot_body_names: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    times = np.asarray(arrays["time_s"], dtype=np.float64).reshape(-1)
    actions = np.asarray(arrays["actions"], dtype=np.float64)
    frame_count = len(times)
    duration = float(times[-1] - times[0]) if frame_count > 1 else 0.0
    motion, gripper_values = action_channel_views(actions, arms)
    translation = np.linalg.norm(motion[:, :, :3], axis=2)
    rotation = np.linalg.norm(motion[:, :, 3:], axis=2)
    activity = np.max(np.concatenate((translation, rotation), axis=1), axis=1) > 1e-5
    first_motion = _first_index(activity)
    last_motion = _last_index(activity)
    idle_ratio = float(1.0 - np.mean(activity)) if frame_count else 1.0

    corrections = 0
    flat_motion = motion.reshape(frame_count, arms * 6)
    if frame_count > 1:
        previous = flat_motion[:-1]
        current = flat_motion[1:]
        corrections = int(np.sum((previous * current < 0) & (np.abs(previous) > 1e-5) & (np.abs(current) > 1e-5)))
    acceleration = np.diff(flat_motion, n=2, axis=0) if frame_count > 2 else np.zeros((0, flat_motion.shape[1]))
    smoothness_proxy = float(np.mean(np.linalg.norm(acceleration, axis=1))) if len(acceleration) else 0.0

    robot_body_names = robot_body_names or {}
    preferred_tip_names = ("psm_tool_tip_link", "endo360_needle", "ecm_end_link", "tool_tip", "end_effector")
    tip_path_m = 0.0
    explicit_tip_positions: list[np.ndarray] = []
    for key, values in arrays.items():
        if not key.endswith("_body_positions_w"):
            continue
        positions = np.asarray(values, dtype=np.float64)
        if positions.ndim != 3 or len(positions) < 2:
            continue
        body_paths = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=2), axis=0)
        robot_name = key.removesuffix("_body_positions_w")
        names = robot_body_names.get(robot_name, [])
        tip_index = next((names.index(name) for name in preferred_tip_names if name in names), None)
        if tip_index is not None and tip_index < positions.shape[1]:
            tip_path_m += float(body_paths[tip_index])
            explicit_tip_positions.append(positions[:, tip_index, :3])
        else:
            tip_path_m += float(np.max(body_paths))

    object_lift_m = 0.0
    object_motion_m = 0.0
    object_motion_index: int | None = None
    for key, values in arrays.items():
        if not key.endswith("_position_w") or "body_positions" in key:
            continue
        positions = np.asarray(values, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] < 3 or len(positions) < 2:
            continue
        lift = positions[:, 2] - positions[0, 2]
        travel = np.linalg.norm(positions - positions[0], axis=1)
        if float(np.max(lift)) > object_lift_m:
            object_lift_m = float(np.max(lift))
        if float(np.max(travel)) > object_motion_m:
            object_motion_m = float(np.max(travel))
            object_motion_index = int(np.argmax(travel))

    gripper_index: int | None = None
    if gripper_values.shape[1]:
        gripper_index = _first_index(np.min(gripper_values, axis=1) < -0.5)

    grasp_relative_drift_m = 0.0
    if gripper_index is not None and explicit_tip_positions:
        object_tracks = [
            np.asarray(values, dtype=np.float64)
            for key, values in arrays.items()
            if key.endswith("_position_w") and "body_positions" not in key and np.asarray(values).ndim == 2
        ]
        if object_tracks:
            tip = explicit_tip_positions[0]
            object_track = object_tracks[0]
            length = min(len(tip), len(object_track))
            start = min(gripper_index, max(length - 1, 0))
            if length > start:
                relative_distance = np.linalg.norm(object_track[:length, :3] - tip[:length, :3], axis=1)
                baseline = float(np.median(relative_distance[start : min(start + 10, length)]))
                grasp_relative_drift_m = float(np.max(np.abs(relative_distance[start:] - baseline)))

    native_reward = np.asarray(arrays.get("environment_reward", []), dtype=np.float64).reshape(-1)
    native_success = np.asarray(arrays.get("environment_success", []), dtype=np.float64).reshape(-1)
    native_success_observed = bool(np.any(native_success > 0.5)) if len(native_success) else False
    native_success_available = bool(np.any(native_success >= 0.0)) if len(native_success) else False
    max_contact_force_n = 0.0
    max_tissue_displacement_m = 0.0
    max_tissue_deformation_proxy = 0.0
    max_tissue_stress_pa = 0.0
    safety_series: dict[str, np.ndarray] = {}
    for key, values in arrays.items():
        if key.endswith("_max_contact_force_n"):
            series = np.asarray(values, dtype=np.float64).reshape(-1)
            max_contact_force_n = max(max_contact_force_n, float(np.max(series)))
            safety_series["contact_force_n"] = np.maximum(safety_series.get("contact_force_n", np.zeros_like(series)), series)
        elif key.endswith("_max_tissue_displacement_m"):
            series = np.asarray(values, dtype=np.float64).reshape(-1)
            max_tissue_displacement_m = max(max_tissue_displacement_m, float(np.max(series)))
            safety_series["tissue_displacement_m"] = np.maximum(
                safety_series.get("tissue_displacement_m", np.zeros_like(series)), series
            )
        elif key.endswith("_max_deformation_gradient_proxy"):
            series = np.asarray(values, dtype=np.float64).reshape(-1)
            max_tissue_deformation_proxy = max(max_tissue_deformation_proxy, float(np.max(series)))
            safety_series["deformation_gradient_proxy"] = np.maximum(
                safety_series.get("deformation_gradient_proxy", np.zeros_like(series)), series
            )
        elif key.endswith("_max_tissue_stress_pa"):
            max_tissue_stress_pa = max(max_tissue_stress_pa, float(np.max(np.asarray(values, dtype=np.float64))))

    start_idle_s = float(times[first_motion] - times[0]) if first_motion is not None else duration
    recovery_s = float(times[-1] - times[last_motion]) if last_motion is not None else 0.0
    completeness = 100.0
    if frame_count < 50:
        completeness -= 30
    if duration < 2.0:
        completeness -= 25
    if start_idle_s > max(2.0, duration * 0.25):
        completeness -= 15
    if recovery_s < 0.35:
        completeness -= 15
    if "Lift" in task and object_lift_m < 0.008:
        completeness -= 25
    completeness = max(0.0, completeness)

    correction_rate = corrections / max(duration, 1.0)
    control_score = max(0.0, 100.0 - correction_rate * 9.0)
    economy_score = max(0.0, 100.0 - max(0.0, idle_ratio - 0.18) * 100.0)
    smoothness_score = max(0.0, 100.0 - smoothness_proxy * 18_000.0)
    if native_success_available:
        task_score = 100.0 if native_success_observed else 0.0
    else:
        task_score = 100.0 if "Lift" not in task else min(100.0, object_lift_m / 0.03 * 100.0)
    overall = round(0.30 * completeness + 0.22 * control_score + 0.18 * economy_score + 0.15 * smoothness_score + 0.15 * task_score)

    timeline = [{"time_s": 0.0, "kind": "start", "label": "Recording started"}]
    if first_motion is not None:
        timeline.append({"time_s": round(float(times[first_motion]), 2), "kind": "approach", "label": "First deliberate tool movement"})
    if gripper_index is not None:
        timeline.append({"time_s": round(float(times[gripper_index]), 2), "kind": "contact", "label": "Gripper close command"})
    if object_motion_index is not None and object_motion_m > 0.002:
        timeline.append({"time_s": round(float(times[object_motion_index]), 2), "kind": "task", "label": "Largest detected object displacement"})
    if native_success_observed:
        success_index = _first_index(native_success > 0.5)
        if success_index is not None and success_index < len(times):
            timeline.append({"time_s": round(float(times[success_index]), 2), "kind": "success", "label": "Simulator-native task success signal"})
    safety_events = []
    safety_labels = {
        "contact_force_n": "Contact-force engineering advisory crossed",
        "tissue_displacement_m": "Tissue-displacement engineering advisory crossed",
        "deformation_gradient_proxy": "Deformation-gradient engineering advisory crossed",
    }
    for metric_name, limit in RESEARCH_ADVISORY_LIMITS.items():
        series = safety_series.get(metric_name)
        if series is None:
            continue
        crossing = _first_index(series > limit)
        if crossing is not None:
            event = {
                "metric": metric_name,
                "limit": limit,
                "observed": round(float(np.max(series)), 5),
                "time_s": round(float(times[min(crossing, len(times) - 1)]), 2),
                "status": "research_advisory_not_clinically_validated",
            }
            safety_events.append(event)
            timeline.append({"time_s": event["time_s"], "kind": "safety", "label": safety_labels[metric_name]})
    if last_motion is not None:
        timeline.append({"time_s": round(float(times[last_motion]), 2), "kind": "recovery", "label": "Last tool movement; recovery hold begins"})
    timeline.append({"time_s": round(duration, 2), "kind": "finish", "label": "Recording finished"})
    timeline.sort(key=lambda item: item["time_s"])

    coaching = []
    if start_idle_s > max(2.0, duration * 0.25):
        coaching.append("Begin closer to the first intentional movement so the policy receives a clean starting example.")
    if correction_rate > 1.2:
        coaching.append("Use Precision speed near the needle and separate position changes from angle changes.")
    if recovery_s < 0.35:
        coaching.append("Hold a stable final pose before stopping so the recovery phase is represented.")
    if "Lift" in task and object_lift_m < 0.008:
        coaching.append("The recording does not yet show a clear object lift; include grasp, elevation, and stable recovery.")
    if safety_events:
        coaching.append("A research safety advisory was crossed. Review the marked instant before reusing this demonstration.")
    if not coaching:
        coaching.append("This trajectory has a readable start, controlled execution, and stable finish. Add anatomy and camera variation next.")

    return {
        "schema": "dr.anmar.skills-twin-analysis.v1",
        "validation_status": "research_proxy_pending_clinician_validation",
        "overall_score": int(max(0, min(100, overall))),
        "grade": "Ready to challenge" if overall >= 80 else "Developing" if overall >= 55 else "Needs another demonstration",
        "duration_s": round(duration, 2),
        "frames": frame_count,
        "metrics": {
            "tool_path_m": round(tip_path_m, 4),
            "object_lift_m": round(object_lift_m, 4),
            "object_motion_m": round(object_motion_m, 4),
            "direction_corrections": corrections,
            "idle_ratio": round(idle_ratio, 3),
            "recovery_hold_s": round(recovery_s, 2),
            "smoothness_proxy": round(smoothness_proxy, 7),
            "max_contact_force_n": round(max_contact_force_n, 3),
            "max_tissue_displacement_m": round(max_tissue_displacement_m, 5),
            "max_tissue_deformation_proxy": round(max_tissue_deformation_proxy, 5),
            "max_tissue_stress_pa": round(max_tissue_stress_pa, 3),
            "grasp_relative_drift_m": round(grasp_relative_drift_m, 5),
            "max_environment_reward": round(float(np.max(native_reward)), 4) if len(native_reward) else None,
            "native_success": native_success_observed if native_success_available else None,
        },
        "subscores": {
            "completeness": round(completeness),
            "control": round(control_score),
            "economy": round(economy_score),
            "smoothness": round(smoothness_score),
            "task_evidence": round(task_score),
        },
        "timeline": timeline,
        "coaching": coaching,
        "safety": {
            "advisory_limits": RESEARCH_ADVISORY_LIMITS,
            "events": safety_events,
            "clinical_thresholds_validated": False,
        },
    }


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
    np.savez_compressed(path, **arrays)
    analysis = analyze_demo(arrays, state.task, state.arms, state.robot_body_names)
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
        }
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
        "control_hz": 50,
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
        "data_file": name,
        "modalities": {
            "robot_state_hz": 50,
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
            "camera_intrinsics": state.camera_intrinsics,
            "semantic_labels": state.semantic_labels,
            "simulator_outcome": "environment_reward, termination, truncation, and success when exposed by the task",
            "contact": "maximum force per available contact sensor",
            "deformable_tissue": "nodal displacement, deformation-gradient proxy, and simulator stress when exposed",
            "robot_and_anatomy_pose": "world-frame tool bodies, task objects, and showcase anatomy transform at 50 Hz",
            "joint_torque": "applied and computed joint torque when exposed by the articulation",
            "operator_study": "input source, normalized gaze/attention coordinates, procedure phase, and event codes at 50 Hz",
        },
        "research_safety_advisories": {
            "limits": RESEARCH_ADVISORY_LIMITS,
            "clinical_thresholds_validated": False,
        },
        "context": context,
        "procedure_annotations": list(state.procedure_events),
        "annotation_vocabulary": {
            "procedure_phases": PROCEDURE_PHASES,
            "procedure_events": PROCEDURE_EVENTS,
            "operator_input_sources": OPERATOR_INPUT_SOURCES,
            "gaze_sources": {"none": 0, "pointer_attention_proxy": 1, "external_eye_tracker": 2, "xr_eye_tracking": 3},
        },
        "analysis": analysis,
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
    if mode == "close":
        eye = target + view_vector * 0.68
    elif mode == "overview":
        eye = target + view_vector * 1.48 + np.asarray((0.0, 0.0, 0.11), dtype=np.float32)
        target = target + np.asarray((0.0, 0.0, 0.015), dtype=np.float32)
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
    if "-IK-Rel" not in args_cli.task:
        raise ValueError("The browser workstation accepts relative-IK tasks. Other variants remain available via the CLI.")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.episode_length_s = 3600.0
    env_cfg.scene.num_envs = 1
    camera_target = np.asarray(env_cfg.viewer.lookat, dtype=np.float32)
    # Start from the room-facing side used by the official OR scene so the
    # doctor sees the instrument, liver, table, and surrounding environment.
    camera_eye = np.asarray((0.45, 0.25, 0.28), dtype=np.float32)
    env_cfg.scene.endoscope = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Endoscope",
        update_period=0.04,
        height=args_cli.camera_height,
        width=args_cli.camera_width,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=0.25,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=tuple(camera_eye.tolist()), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
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
    for wrist_index, wrist_robot_name in enumerate(wrist_robot_names, start=1):
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
    if guide_kind in {"threading", "cutting_path", "navigation"}:
        anatomy_position = (-0.117, -0.0945, -0.189)
    elif procedure_id in {"needle-pickup", "needle-transfer"} or procedure.get("proxy_organ"):
        anatomy_position = (-0.117, -0.2445, -0.144)
    else:
        anatomy_position = (-0.117, -0.1945, -0.164)
    if organ_usd.is_file():
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
    stereo_right_camera = scene["endoscope_right"]
    wrist_cameras = [scene[f"wrist_{index}"] for index in range(1, len(wrist_robot_names) + 1)]
    camera_sources = {"endoscope_left": camera, "endoscope_right": stereo_right_camera}
    camera_sources.update({f"wrist_{index}": wrist_camera for index, wrist_camera in enumerate(wrist_cameras, start=1)})
    robot_names = sorted(scene.articulations.keys())
    robots = {name: scene[name] for name in robot_names}
    robot_body_names = {name: list(getattr(robot, "body_names", [])) for name, robot in robots.items()}
    object_names = sorted(scene.rigid_objects.keys())
    objects = {name: scene[name] for name in object_names}
    deformable_names = sorted(getattr(scene, "deformable_objects", {}).keys())
    deformables = {name: scene[name] for name in deformable_names}
    contact_sensors = {}
    for name in sorted(getattr(scene, "sensors", {}).keys()):
        sensor = scene[name]
        if getattr(sensor.data, "net_forces_w", None) is not None:
            contact_sensors[name] = sensor
    showcase_children: list[Any] = []
    default_showcase_names: set[str] = {"Liver_topo_blender"}
    proxy_visual_ready = False
    collision_mesh_count = 0
    anatomy_guard_volumes: list[tuple[np.ndarray, np.ndarray, str]] = []
    anatomy_surface_samples: list[tuple[np.ndarray, np.ndarray, str]] = []
    anatomy_collision_prims: list[Any] = []
    puncture_marker_translate = None
    puncture_marker_prim = None
    stage = None
    if organ_usd.is_file():
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

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
        drape_transform.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.12, 0.001))
        drape_transform.AddScaleOp().Set(Gf.Vec3f(0.62, 0.50, 0.002))
        drape_material = UsdShade.Material.Define(stage, f"{drape_path}/Material")
        drape_shader = UsdShade.Shader.Define(stage, f"{drape_path}/Material/Shader")
        drape_shader.CreateIdAttr("UsdPreviewSurface")
        drape_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.025, 0.12, 0.15))
        drape_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
        drape_material.CreateSurfaceOutput().ConnectToSource(drape_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(drape.GetPrim()).Bind(drape_material)
        puncture_marker_path = "/World/envs/env_0/DrAnmarNeedleEntry"
        puncture_marker = UsdGeom.Sphere.Define(stage, puncture_marker_path)
        puncture_marker.CreateRadiusAttr(0.0024)
        puncture_marker_prim = puncture_marker.GetPrim()
        puncture_marker_translate = UsdGeom.Xformable(puncture_marker_prim).AddTranslateOp()
        puncture_marker_material = UsdShade.Material.Define(stage, f"{puncture_marker_path}/Material")
        puncture_marker_shader = UsdShade.Shader.Define(stage, f"{puncture_marker_path}/Material/Shader")
        puncture_marker_shader.CreateIdAttr("UsdPreviewSurface")
        puncture_marker_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.18, 0.012, 0.008))
        puncture_marker_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
        puncture_marker_material.CreateSurfaceOutput().ConnectToSource(puncture_marker_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(puncture_marker_prim).Bind(puncture_marker_material)
        UsdGeom.Imageable(puncture_marker_prim).MakeInvisible()
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

        proxy_organ = procedure.get("proxy_organ")
        object_prim = stage.GetPrimAtPath("/World/envs/env_0/Object")
        if proxy_organ and object_prim.IsValid():
            for child in object_prim.GetChildren():
                if child.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(child).MakeInvisible()
            proxy_path = "/World/envs/env_0/Object/DrAnmarOrganProxy"
            proxy_prim = stage.DefinePrim(proxy_path, "Xform")
            proxy_prim.GetReferences().AddReference(str(organ_usd))
            for child in proxy_prim.GetChildren():
                if child.IsA(UsdGeom.Imageable):
                    if child.GetName() == proxy_organ:
                        UsdGeom.Imageable(child).MakeVisible()
                    else:
                        UsdGeom.Imageable(child).MakeInvisible()
            proxy_mesh = stage.GetPrimAtPath(f"{proxy_path}/{proxy_organ}/{proxy_organ}")
            if proxy_mesh.IsValid():
                bounds = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(proxy_mesh).ComputeAlignedRange()
                center = bounds.GetMidpoint()
                size = bounds.GetSize()
                longest = max(float(size[0]), float(size[1]), float(size[2]), 1e-6)
                scale = min(0.055 / longest, 0.32)
                transform = UsdGeom.Xformable(proxy_prim)
                transform.ClearXformOpOrder()
                transform.AddTranslateOp().Set(Gf.Vec3d(-center[0] * scale, -center[1] * scale, -center[2] * scale + 0.006))
                transform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
                source_material = stage.GetPrimAtPath(f"{showcase_path}/DrAnmarMaterials/{proxy_organ}")
                if source_material.IsValid():
                    UsdShade.MaterialBindingAPI.Apply(proxy_mesh).Bind(
                        UsdShade.Material(source_material),
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    )
                proxy_visual_ready = True

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

    def set_anatomy_collision_enabled(enabled: bool) -> None:
        """Temporarily open a needle-only channel while the tool guard remains active."""
        for mesh in anatomy_collision_prims:
            parent = mesh.GetParent()
            visible = UsdGeom.Imageable(parent).ComputeVisibility(Usd.TimeCode.Default()) != UsdGeom.Tokens.invisible
            UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr().Set(bool(enabled and visible))

    def show_puncture_marker(position: np.ndarray | None) -> None:
        if puncture_marker_prim is None or puncture_marker_translate is None:
            return
        imageable = UsdGeom.Imageable(puncture_marker_prim)
        if position is None:
            imageable.MakeInvisible()
            return
        puncture_marker_translate.Set(Gf.Vec3d(*position.astype(float).tolist()))
        imageable.MakeVisible()

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

    def derive_needle_tip_offsets() -> np.ndarray:
        """Derive the two open ends of the official semicircular needle in Object-local metres."""
        if stage is None:
            return np.empty((0, 3), dtype=np.float32)
        object_prim = stage.GetPrimAtPath("/World/envs/env_0/Object")
        if not object_prim.IsValid():
            return np.empty((0, 3), dtype=np.float32)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world_to_object = cache.GetLocalToWorldTransform(object_prim).GetInverse()
        local_points: list[np.ndarray] = []
        for prim in Usd.PrimRange(object_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if not points:
                continue
            mesh_to_world = cache.GetLocalToWorldTransform(prim)
            for point in points:
                world_point = mesh_to_world.Transform(point)
                local_point = world_to_object.Transform(world_point)
                local_points.append(np.asarray(tuple(local_point), dtype=np.float32))
        if not local_points:
            return np.empty((0, 3), dtype=np.float32)
        vertices = np.stack(local_points)
        span = np.ptp(vertices, axis=0)
        thickness_axis = int(np.argmin(span))
        curve_axes = [axis for axis in range(3) if axis != thickness_axis]
        arc_axis = max(curve_axes, key=lambda axis: float(span[axis]))
        ordered = np.argsort(vertices[:, arc_axis])
        endpoint_count = max(12, min(64, len(vertices) // 100))
        return np.stack(
            (
                np.mean(vertices[ordered[:endpoint_count]], axis=0),
                np.mean(vertices[ordered[-endpoint_count:]], axis=0),
            )
        ).astype(np.float32)

    def rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        vector_part = quaternion[1:4]
        cross = 2.0 * np.cross(vector_part, vector)
        return vector + quaternion[0] * cross + np.cross(vector_part, cross)

    def constrain_anatomy_translation(
        tool_position: np.ndarray | None,
        translation: np.ndarray,
    ) -> tuple[np.ndarray, float | None, bool]:
        if tool_position is None or not anatomy_guard_volumes:
            return translation, None, False
        adjusted = translation.astype(np.float32).copy()
        clearance, outward, _surface = anatomy_surface_query(tool_position)
        if clearance is None or outward is None:
            return adjusted, None, False
        inward_component = float(np.dot(adjusted, outward))
        guard_active = clearance <= 0.004 and inward_component < 0.0
        if guard_active:
            remaining_fraction = float(np.clip(max(clearance, 0.0) / 0.004, 0.0, 1.0))
            adjusted -= outward * inward_component * (1.0 - remaining_fraction)
        return adjusted, clearance, guard_active

    refresh_anatomy_guard_volumes()
    needle_tip_offsets_local = derive_needle_tip_offsets()
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
    room_waypoints = procedure_waypoints(procedure)
    procedure_markers = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/World/DrAnmarProcedureGuide",
            markers={
                "start": sim_utils.SphereCfg(
                    radius=0.006,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.08, 0.92, 1.0), emissive_color=(0.02, 0.38, 0.50), opacity=0.82
                    ),
                ),
                "path": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.64, 0.16), emissive_color=(0.40, 0.16, 0.01), opacity=0.78
                    ),
                ),
                "finish": sim_utils.SphereCfg(
                    radius=0.006,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.20, 0.95, 0.48), emissive_color=(0.02, 0.38, 0.14), opacity=0.84
                    ),
                ),
            },
        )
    )
    if len(room_waypoints):
        marker_indices = np.ones(len(room_waypoints), dtype=np.int32)
        marker_indices[0] = 0
        marker_indices[-1] = 2
        procedure_markers.visualize(translations=room_waypoints, marker_indices=marker_indices)
        procedure_markers.set_visibility(True)
    else:
        procedure_markers.visualize(translations=np.zeros((1, 3), dtype=np.float32))
        procedure_markers.set_visibility(False)

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

    def needle_tip_positions_world() -> np.ndarray:
        if not len(needle_tip_offsets_local) or not objects:
            return np.empty((0, 3), dtype=np.float32)
        needle = next(iter(objects.values()))
        position = needle.data.root_pos_w[0, :3].detach().cpu().numpy().astype(np.float32)
        quaternion = needle.data.root_quat_w[0, :4].detach().cpu().numpy().astype(np.float32)
        return np.stack([position + rotate_wxyz(quaternion, offset) for offset in needle_tip_offsets_local])

    def apply_endoscope_camera_view(selected_scenario: str, view_mode: str) -> None:
        selected_eye, selected_target = scenario_camera_pose(camera_eye, camera_target, selected_scenario)
        selected_eye, selected_target = camera_view_pose(selected_eye, selected_target, view_mode)
        camera.set_world_poses_from_view(
            torch.tensor([selected_eye.tolist()], device=camera.device),
            torch.tensor([selected_target.tolist()], device=camera.device),
        )
        selected_right_eye = selected_eye + np.asarray((0.0, 0.006, 0.0), dtype=np.float32)
        stereo_right_camera.set_world_poses_from_view(
            torch.tensor([selected_right_eye.tolist()], device=stereo_right_camera.device),
            torch.tensor([selected_target.tolist()], device=stereo_right_camera.device),
        )

    action_dim = int(env.action_space.shape[-1])
    arms = 2 if "Dual" in args_cli.task else 1
    has_grippers = "Lift" in args_cli.task or "Handover" in args_cli.task
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
        anatomy_showcase=args_cli.anatomy_title or "Official CT liver",
        anatomy_scene_id=args_cli.anatomy_scene_id,
        anatomy_asset=str(organ_usd) if organ_usd.is_file() else "",
        openusd_environment=str(openusd_environment) if openusd_environment else "",
        procedure=procedure,
        openusd_scene_loaded=bool(openusd_environment and organ_usd.is_file() and showcase_children),
        organ_proxy_visual_ready=proxy_visual_ready,
        anatomy_collision_meshes=collision_mesh_count,
    )
    state.camera_names = list(camera_sources)
    state.camera_frame_ids = {name: 0 for name in camera_sources}
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

    def reset_environment(selected_scenario: str, selected_seed: int) -> None:
        if stage is not None:
            for joint_path in assisted_grasp_joints.values():
                stage.RemovePrim(joint_path)
        assisted_grasp_joints.clear()
        show_puncture_marker(None)
        np.random.seed(selected_seed)
        torch.manual_seed(selected_seed)
        env.reset(seed=selected_seed)
        apply_native_object_scenario(objects, selected_scenario, selected_seed)
        initial_object_positions.clear()
        initial_object_positions.update(
            {
                name: rigid_object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32).copy()
                for name, rigid_object in objects.items()
            }
        )
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
                f"{args_cli.anatomy_title or 'Official anatomy'} · multi-organ context"
                if show_multi_organ
                else args_cli.anatomy_title or "Official CT liver"
            )
            state.procedure_waypoints_completed = 0
            state.procedure_motion_seen = False
            state.procedure_grasp_seen = False
            state.procedure_object_lift_m = 0.0
            state.procedure_object_motion_m = 0.0
            state.procedure_started_at = time.monotonic()
            state.procedure_last_motion_at = time.monotonic()
            state.anatomy_collision_meshes = enabled_colliders
            state.assisted_grasp_active = [False] * state.arms
            state.tool_to_object_distance_m = [None] * state.arms
            state.tool_to_object_offset_m = [None] * state.arms
            state.virtual_fixture_active = False
            state.closest_anatomy_clearance_m = None
            state.needle_tip_clearance_m = None
            state.needle_puncture_active = False
            state.needle_penetration_depth_m = 0.0
            state.adaptive_precision_active = False
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
    last_vision_sample_time = 0.0
    last_safety_sample_time = 0.0
    latest_contact_forces: dict[str, float] = {}
    latest_deformable_safety: dict[str, float] = {}
    replay_actions: np.ndarray | None = None
    replay_index = 0
    assisted_grasp_joints: dict[int, str] = {}
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

        if camera_view_request is not None and not reset_requested:
            with torch.inference_mode():
                apply_endoscope_camera_view(scenario_id, camera_view_request)

        if record_request == "start":
            demo_frames.clear()
            vision_frames.clear()
            demo_started_at = datetime.now(timezone.utc).isoformat()
            demo_started_monotonic = time.monotonic()
            last_vision_sample_time = 0.0
            with state.lock:
                state.recording = True
                state.recorded_frames = 0
                state.intervention_count = 0
                state.procedure_phase = "setup"
                state.procedure_event_code = 0
                state.procedure_events.clear()
        elif record_request == "stop":
            name = save_demo(state, demo_frames, vision_frames, demo_started_at)
            with state.lock:
                state.recording = False
                state.last_demo = name or state.last_demo
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
                replay_actions = np.load(replay_path)["actions"].astype(np.float32)
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

        if replay_actions is not None and replay_index < len(replay_actions):
            action_np = replay_actions[replay_index].copy()
            replay_index += 1
            replay_profile = SCENARIO_NATIVE_PROFILES.get(scenario_id, {})
            replay_yaw = float(replay_profile.get("translation_yaw_deg", 0.0))
            if replay_yaw:
                radians = np.deg2rad(replay_yaw)
                cosine, sine = np.cos(radians), np.sin(radians)
                axis_scale = np.asarray(replay_profile.get("axis_scale", (1.0, 1.0, 1.0)), dtype=np.float32)
                for arm in range(state.arms):
                    start = state.body_action_slice(arm).start
                    translation = action_np[start : start + 3].copy()
                    translation[:2] = (
                        cosine * translation[0] - sine * translation[1],
                        sine * translation[0] + cosine * translation[1],
                    )
                    action_np[start : start + 3] = translation * axis_scale
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
        if state.has_grippers and objects and stage is not None:
            grasp_object = next(iter(objects.values()))
            object_position = grasp_object.data.root_pos_w[0, :3].detach().cpu().numpy().astype(np.float32)
            for arm, is_open in enumerate(grippers_open):
                tool_position_for_grasp = tool_position_for_arm(arm)
                if tool_position_for_grasp is not None:
                    object_offset = object_position - tool_position_for_grasp
                    grasp_distances[arm] = float(np.linalg.norm(object_offset))
                    grasp_offsets[arm] = object_offset.astype(float).round(5).tolist()
                if is_open and arm in assisted_grasp_joints:
                    stage.RemovePrim(assisted_grasp_joints.pop(arm))
                elif (
                    not is_open
                    and arm not in assisted_grasp_joints
                    and grasp_distances[arm] is not None
                    and grasp_distances[arm] <= state.grasp_capture_radius_m
                ):
                    robot_prim_name = "Robot" if state.arms == 1 else f"Robot_{arm + 1}"
                    tool_body_path = Sdf.Path(f"/World/envs/env_0/{robot_prim_name}/{wrist_tip_name}")
                    object_body_path = Sdf.Path("/World/envs/env_0/Object")
                    if stage.GetPrimAtPath(tool_body_path).IsValid() and stage.GetPrimAtPath(object_body_path).IsValid():
                        joint_path = f"/World/envs/env_0/DrAnmarAssistedGrasp{arm + 1}"
                        fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                        fixed_joint.CreateBody0Rel().SetTargets([tool_body_path])
                        fixed_joint.CreateBody1Rel().SetTargets([object_body_path])
                        fixed_joint.CreateCollisionEnabledAttr().Set(False)
                        fixed_joint.CreateBreakForceAttr().Set(1_000_000.0)
                        fixed_joint.CreateBreakTorqueAttr().Set(1_000_000.0)
                        assisted_grasp_joints[arm] = joint_path
                        with state.lock:
                            state.coaching_cue = "Needle secured between the jaws. Open the gripper to release it."
            with state.lock:
                state.assisted_grasp_active = [arm in assisted_grasp_joints for arm in range(state.arms)]
                state.tool_to_object_distance_m = [round(value, 5) if value is not None else None for value in grasp_distances]
                state.tool_to_object_offset_m = grasp_offsets

        with state.lock:
            puncture_was_active = state.needle_puncture_active
            max_penetration = state.needle_max_penetration_m
        needle_clearance: float | None = None
        needle_outward: np.ndarray | None = None
        needle_surface: np.ndarray | None = None
        if assisted_grasp_joints:
            tip_queries = []
            for tip in needle_tip_positions_world():
                clearance, outward, surface = anatomy_surface_query(tip)
                if clearance is not None and outward is not None and surface is not None:
                    tip_queries.append((abs(clearance), clearance, outward, surface))
            if tip_queries:
                _absolute, needle_clearance, needle_outward, needle_surface = min(
                    tip_queries, key=lambda item: item[0]
                )

        adaptive_precision_active = False
        virtual_fixture_active = False
        puncture_active = bool(puncture_was_active and assisted_grasp_joints)
        anatomy_clearances: list[float] = []
        for arm in range(state.arms):
            body_slice = state.body_action_slice(arm)
            translation = action_np[body_slice.start : body_slice.start + 3].copy()
            if replay_actions is None and grasp_distances[arm] is not None and grasp_distances[arm] < 0.035:
                feather = 0.35 + 0.65 * float(np.clip(grasp_distances[arm] / 0.035, 0.0, 1.0))
                translation *= feather
                adaptive_precision_active = adaptive_precision_active or bool(np.any(translation))
            if arm in assisted_grasp_joints and needle_clearance is not None and needle_outward is not None:
                needle_inward = float(np.dot(translation, needle_outward))
                if not puncture_active and needle_clearance <= 0.0025 and needle_inward < 0.0:
                    puncture_active = True
                penetration_depth = max(0.0, -needle_clearance)
                if puncture_active and penetration_depth >= max_penetration and needle_inward < 0.0:
                    translation -= needle_outward * needle_inward
                    virtual_fixture_active = True
            if virtual_fixture_enabled:
                translation, clearance, guard_active = constrain_anatomy_translation(
                    tool_position_for_arm(arm),
                    translation,
                )
                if clearance is not None:
                    anatomy_clearances.append(clearance)
                virtual_fixture_active = virtual_fixture_active or guard_active
            action_np[body_slice.start : body_slice.start + 3] = translation
        if puncture_active and needle_clearance is not None and needle_clearance > 0.006:
            puncture_active = False
        if puncture_active != puncture_was_active:
            set_anatomy_collision_enabled(not puncture_active)
        show_puncture_marker(needle_surface if puncture_active else None)
        penetration_depth = max(0.0, -needle_clearance) if needle_clearance is not None else 0.0
        with state.lock:
            state.adaptive_precision_active = adaptive_precision_active
            state.virtual_fixture_active = virtual_fixture_active
            state.closest_anatomy_clearance_m = (
                round(min(anatomy_clearances, key=abs), 5) if anatomy_clearances else None
            )
            state.needle_tip_clearance_m = round(needle_clearance, 5) if needle_clearance is not None else None
            state.needle_puncture_active = puncture_active
            state.needle_penetration_depth_m = round(min(penetration_depth, max_penetration), 5)
            if puncture_active and penetration_depth >= max_penetration:
                state.coaching_cue = "Maximum rehearsal depth reached. Rotate along the needle arc or withdraw."
            elif puncture_active:
                state.coaching_cue = "Needle tip is inside the tissue proxy. Rotate through the curved needle arc; the shaft remains excluded."
            elif virtual_fixture_active:
                state.coaching_cue = "Instrument boundary reached. The shaft cannot enter; align the needle tip or withdraw."

        actions = torch.from_numpy(action_np).to(device=env.unwrapped.device).reshape(1, -1)
        with torch.inference_mode():
            _observations, reward, terminated, truncated, info = env.step(actions)
            update_wrist_camera_poses()
        environment_reward = scalar_value(reward)
        environment_terminated = bool(scalar_value(terminated))
        environment_truncated = bool(scalar_value(truncated))
        environment_success = native_success_from_info(info)

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
        tool_position = preferred_tool_position(robots, robot_body_names)
        with state.lock:
            if motion_active:
                state.procedure_motion_seen = True
                state.procedure_last_motion_at = current_time
            if state.has_grippers and any(not is_open for is_open in grippers_open):
                state.procedure_grasp_seen = True
            state.procedure_object_lift_m = max(state.procedure_object_lift_m, max_object_lift)
            state.procedure_object_motion_m = max(state.procedure_object_motion_m, max_object_motion)
            waypoint_index = state.procedure_waypoints_completed
            if tool_position is not None and waypoint_index < len(room_waypoints):
                if float(np.linalg.norm(tool_position - room_waypoints[waypoint_index])) <= 0.014:
                    state.procedure_waypoints_completed += 1
                    state.procedure_last_motion_at = current_time

        with state.lock:
            is_recording = state.recording
        interactive_active = bool(np.any(manual_action)) or replay_actions is not None or is_recording
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
                needle_puncture_active = state.needle_puncture_active
                needle_penetration_depth_m = state.needle_penetration_depth_m
                needle_tip_clearance_m = state.needle_tip_clearance_m
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
                "needle_puncture_active": np.array(needle_puncture_active, dtype=np.bool_),
                "needle_penetration_depth_m": np.array(needle_penetration_depth_m, dtype=np.float32),
                "needle_tip_clearance_m": np.array(
                    needle_tip_clearance_m if needle_tip_clearance_m is not None else np.nan,
                    dtype=np.float32,
                ),
                "anatomy_showcase_position_w": np.asarray((-0.117, -0.0945, -0.144), dtype=np.float32),
                "anatomy_showcase_quaternion_w": np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
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
            for key, value in latest_contact_forces.items():
                frame[key] = np.array(value, dtype=np.float32)
            for key, value in latest_deformable_safety.items():
                frame[key] = np.array(value, dtype=np.float32)
            demo_frames.append(frame)
            with state.lock:
                state.recorded_frames = len(demo_frames)

        now = time.monotonic()
        fps_steps += 1
        frame_interval = 0.04 if interactive_active else 0.20
        if camera.data.output.get("rgb") is not None and now - last_frame_time >= frame_interval:
            rendered_jpegs = {}
            for camera_name, sensor_camera in camera_sources.items():
                camera_output = sensor_camera.data.output.get("rgb")
                if camera_output is not None:
                    rendered_jpegs[camera_name] = encode_jpeg(camera_output[0], scenario_id)
            camera_rgb = camera.data.output["rgb"][0]
            if is_recording and now - last_vision_sample_time >= 0.20:
                observation = rgb_tensor_to_image(camera_rgb, scenario_id).resize((360, 240), Image.Resampling.BILINEAR)
                vision_frame = {
                    "time_s": np.array(now - demo_started_monotonic, dtype=np.float64),
                    "rgb": np.asarray(observation, dtype=np.uint8),
                }
                depth_tensor = camera.data.output.get("distance_to_image_plane")
                if depth_tensor is not None:
                    depth = depth_tensor[0].detach().cpu().numpy().astype(np.float32)
                    depth = np.squeeze(depth)
                    depth = np.nan_to_num(depth, nan=0.0, posinf=2.0, neginf=0.0)
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
                    semantic_image = Image.fromarray(semantic, mode="I").resize((360, 240), Image.Resampling.NEAREST)
                    vision_frame["semantic_id"] = np.asarray(semantic_image, dtype=np.uint32)
                    if not state.semantic_labels:
                        with state.lock:
                            state.semantic_labels = camera_semantic_labels(camera)
                for camera_name in ("endoscope_right", "wrist_1", "wrist_2"):
                    sensor_camera = camera_sources.get(camera_name)
                    sensor_rgb = sensor_camera.data.output.get("rgb") if sensor_camera is not None else None
                    if sensor_rgb is not None:
                        sensor_image = rgb_tensor_to_image(sensor_rgb[0], scenario_id).resize(
                            (360, 240), Image.Resampling.BILINEAR
                        )
                        vision_frame[f"{camera_name}_rgb"] = np.asarray(sensor_image, dtype=np.uint8)
                vision_frames.append(vision_frame)
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
        name = save_demo(state, demo_frames, vision_frames, demo_started_at)
        if name:
            state.last_demo = name
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
