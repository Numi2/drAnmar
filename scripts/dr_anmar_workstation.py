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


DATA_ROOT = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()

parser = argparse.ArgumentParser(description="Run the Dr.Anmar browser workstation.")
parser.add_argument("--task", default="Isaac-Lift-Needle-PSM-IK-Rel-v0")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=2361)
parser.add_argument("--demo_dir", type=Path, default=DATA_ROOT / "demos")
parser.add_argument("--camera_width", type=int, default=960)
parser.add_argument("--camera_height", type=int, default=640)
parser.add_argument("--disable_fabric", action="store_true", default=False)
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
    .view{position:relative;overflow:hidden;background:#020608;display:flex;align-items:center;justify-content:center}.view img{width:100%;height:100%;object-fit:contain}.camera-tabs{position:absolute;left:50%;bottom:16px;translate:-50% 0;display:flex;gap:5px;padding:5px;background:#061118cc;border:1px solid #ffffff24;border-radius:8px}.camera-tabs button{min-height:31px;padding:0 10px;font-size:10px}.camera-tabs button.active{background:var(--cyan);color:#031014}.gaze-cursor{position:absolute;width:18px;height:18px;border:1px solid #fff;border-radius:50%;translate:-50% -50%;pointer-events:none;opacity:0;box-shadow:0 0 0 3px #2cd2e855}.view.gaze-on .gaze-cursor{opacity:.85}
    .hud{position:absolute;left:16px;top:16px;padding:10px 13px;border:1px solid #ffffff24;border-radius:8px;background:#051016c9;backdrop-filter:blur(6px);font:12px/1.5 ui-monospace,SFMono-Regular,Menlo;color:#cfe7eb}.hud strong{color:var(--cyan)}
    .recflag{display:none;position:absolute;right:18px;top:18px;color:#fff;background:#c91f2f;padding:8px 12px;border-radius:99px;font-size:12px;font-weight:900;letter-spacing:.08em}.recflag.on{display:block}
    aside{overflow:auto;padding:17px;background:var(--panel);border-left:1px solid var(--line)}
    h2{font-size:12px;letter-spacing:.14em;color:#a9c1ca;margin:3px 0 11px;text-transform:uppercase}.card{border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:13px;background:#0a171e}
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.grid.two{grid-template-columns:repeat(2,1fr)}
    button{min-height:42px;border:1px solid #315462;border-radius:7px;background:#10252e;color:var(--ink);font-weight:750;cursor:pointer;touch-action:manipulation;user-select:none;-webkit-user-select:none}button:hover{border-color:var(--cyan);background:#153540}button:active{transform:translateY(1px);background:var(--cyan2)}
    button.primary{background:var(--cyan);border-color:var(--cyan);color:#041014}button.danger{background:#31171c;border-color:#74414a;color:#ffabb2}button.stop{grid-column:1/-1;background:#27323a;border-color:#5f727c}
    .speedbar{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:11px}.speedbar button{min-height:35px;font-size:11px}.speedbar button.active{background:var(--cyan);border-color:var(--cyan);color:#041014}.dpad{display:grid;grid-template-columns:repeat(3,1fr);grid-template-areas:"blank up blank2" "left stop right" "blank3 down blank4";gap:6px}.dpad .up{grid-area:up}.dpad .left{grid-area:left}.dpad .stop-center{grid-area:stop;min-height:54px;background:#26343b;border-color:#617681}.stop-center small{display:block;color:#9bb0b8;font-size:10px;margin-top:2px}.dpad .right{grid-area:right}.dpad .down{grid-area:down}.depthgrid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}.anglegrid{display:grid;grid-template-columns:1fr 1fr;gap:6px}.move-button{min-height:54px;touch-action:none;position:relative}.move-button small{display:block;color:#86a5af;font-size:10px;margin-top:2px}.move-button.held{background:var(--cyan);border-color:var(--cyan);color:#041014;box-shadow:0 0 16px #2cd2e855}.move-button.held small{color:#0a5260}.control-readout{display:flex;align-items:center;gap:7px;margin-top:10px;color:var(--muted);font-size:11px}.control-readout i{width:7px;height:7px;border-radius:50%;background:#536a73}.control-readout.moving{color:var(--green)}.control-readout.moving i{background:var(--green);box-shadow:0 0 10px #42e49b99}
    .hint{color:var(--muted);font-size:12px;margin-top:9px}.status{font:12px/1.65 ui-monospace,SFMono-Regular,Menlo;color:#bdd2d8;word-break:break-word}.status b{color:var(--green)}.hidden{display:none}.arm.active,.autonomy.active{background:var(--cyan);color:#041014;border-color:var(--cyan)}
    .supervision{border-color:#356475;background:linear-gradient(135deg,#0d2731,#09171e)}.supervision-state{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:9px}.supervision-state b{color:var(--cyan)}.cue{min-height:32px;margin-top:9px;padding:8px;border-left:2px solid var(--cyan);background:#061219;color:#9fc0c9;font-size:11px}.take-control{width:100%;margin-top:8px;background:#ffd978;color:#251b02;border-color:#ffd978}
    .safety-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.safety-metric{padding:8px;background:#061219;border:1px solid #1c3742}.safety-metric b{display:block;color:var(--green);font:15px ui-monospace,monospace}.safety-metric span{color:var(--muted);font-size:9px}.ghost-state{margin-top:8px;color:var(--muted);font-size:11px}.ghost-state.on{color:var(--green)}
    #toast{position:fixed;left:50%;bottom:20px;translate:-50% 20px;opacity:0;background:#e9f8fa;color:#061116;border-radius:8px;padding:10px 15px;font-weight:750;transition:.2s;pointer-events:none}#toast.show{opacity:1;translate:-50% 0}
    @media(max-width:880px){main{display:block;height:auto}.view{height:52vh}aside{border-left:0;border-top:1px solid var(--line)}header{padding:0 12px}.tag{display:none}}
  </style>
</head>
<body>
<header><div class="brand">DR.<span>ANMAR</span></div><div class="tag">SIMULATION ONLY · NO PHYSICAL ROBOT OUTPUT</div><div class="live"><i id="dot" class="dot"></i><span id="connection">Connecting…</span></div></header>
<main>
  <section id="cameraView" class="view"><img id="cameraImage" src="/video/endoscope_left" alt="Live simulated medical sensor view"><div class="hud"><strong id="cameraLabel">STEREO ENDOSCOPE · LEFT</strong><br><span id="hud">Waiting for Isaac Lab…</span></div><div id="recflag" class="recflag">● RECORDING</div><div id="gazeCursor" class="gaze-cursor"></div><div class="camera-tabs"><button class="active" data-camera="endoscope_left" onclick="setCamera('endoscope_left',this)">Stereo left</button><button data-camera="endoscope_right" onclick="setCamera('endoscope_right',this)">Stereo right</button><button data-camera="wrist_1" onclick="setCamera('wrist_1',this)">Wrist 1</button><button id="wrist2Tab" class="hidden" data-camera="wrist_2" onclick="setCamera('wrist_2',this)">Wrist 2</button></div></section>
  <aside>
    <h2>Supervision</h2><div class="card supervision"><div class="supervision-state"><span>Autonomy level</span><b id="autonomyState">L0 · Manual</b></div><div class="grid two"><button id="manualMode" class="autonomy active" onclick="setAutonomy('manual')">Manual</button><button id="guidedMode" class="autonomy" onclick="setAutonomy('guided')">Guided</button></div><button class="take-control" onclick="takeControl()">Take control now</button><div id="coachingCue" class="cue">You command every movement. Dr.Anmar records telemetry for coaching.</div></div>
    <div id="armPanel" class="hidden"><h2>Active instrument</h2><div class="card"><div class="grid two"><button id="arm0" class="arm active" onclick="setArm(0)">Instrument 1</button><button id="arm1" class="arm" onclick="setArm(1)">Instrument 2</button></div></div></div>
    <h2>Movement speed</h2><div class="card"><div class="speedbar"><button data-speed="0.45" onclick="setSpeed(0.45,this)">Precision</button><button class="active" data-speed="1" onclick="setSpeed(1,this)">Normal</button><button data-speed="1.8" onclick="setSpeed(1.8,this)">Fast</button></div><div class="hint">Hold controls to move. Release to stop. Fast is for open space; Precision is for grasping.</div></div>
    <h2>Tool position</h2><div class="card"><div class="dpad">
      <button class="move-button up" data-axis="2" data-direction="1">↑ Up<small>R</small></button>
      <button class="move-button left" data-axis="1" data-direction="1">← Left<small>A</small></button>
      <button class="stop-center" onclick="stopTool()">■ Stop<small>Esc</small></button>
      <button class="move-button right" data-axis="1" data-direction="-1">Right →<small>D</small></button>
      <button class="move-button down" data-axis="2" data-direction="-1">↓ Down<small>F</small></button>
    </div><div class="depthgrid"><button class="move-button" data-axis="0" data-direction="-1">Toward patient<small>W</small></button><button class="move-button" data-axis="0" data-direction="1">Away from patient<small>S</small></button></div><div id="controlReadout" class="control-readout"><i></i><span>Ready · hold a control to move</span></div></div>
    <h2>Tool angle</h2><div class="card"><div class="anglegrid">
      <button class="move-button" data-axis="3" data-direction="-1">↶ Roll left<small>Q</small></button><button class="move-button" data-axis="3" data-direction="1">Roll right ↷<small>E</small></button>
      <button class="move-button" data-axis="4" data-direction="-1">Pitch up<small>↑</small></button><button class="move-button" data-axis="4" data-direction="1">Pitch down<small>↓</small></button>
      <button class="move-button" data-axis="5" data-direction="-1">← Yaw left<small>←</small></button><button class="move-button" data-axis="5" data-direction="1">Yaw right →<small>→</small></button>
    </div><div class="hint">Keyboard: WASD + R/F for position, arrows + Q/E for angle, and Esc to stop. Standard gamepads are supported.</div></div>
    <div id="gripperPanel"><h2>Gripper</h2><div class="card"><div class="grid two"><button onclick="grip(true)">Open</button><button class="primary" onclick="grip(false)">Close / grasp</button></div><div class="hint">Press Space to toggle open / close.</div></div></div>
    <h2>Expert path guide</h2><div class="card"><div class="grid two"><button class="primary" onclick="referenceGhost(true)">Show clinician path</button><button onclick="referenceGhost(false)">Hide path</button></div><div id="ghostState" class="ghost-state">Select a clinician reference in Skills Twin first.</div></div>
    <h2>Procedure annotation</h2><div class="card"><div class="grid"><button onclick="annotatePhase('approach')">Approach</button><button onclick="annotatePhase('grasp')">Grasp</button><button onclick="annotatePhase('manipulation')">Manipulate</button><button onclick="annotatePhase('recovery')">Recovery</button><button onclick="annotateEvent('task_complete')">Task event</button><button onclick="annotateEvent('safety_review')">Safety event</button></div><div class="hint">Phase labels and events are synchronized into the training trajectory.</div></div>
    <h2>Research safety monitor</h2><div class="card"><div class="safety-grid"><div class="safety-metric"><b id="forceMetric">—</b><span>CONTACT N</span></div><div class="safety-metric"><b id="deformMetric">—</b><span>TISSUE MM</span></div><div class="safety-metric"><b id="stressMetric">—</b><span>STRESS PA</span></div></div><div class="hint">Simulator signals only. Limits are engineering advisories, not clinical thresholds.</div></div>
    <h2>Demonstration</h2><div class="card"><div class="grid two"><button id="record" class="primary" onclick="recording(true)">Start recording</button><button onclick="recording(false)">Stop & save</button><button onclick="replay()">Replay last</button><button onclick="resetScene()">Reset scene</button></div><div class="hint" id="lastDemo">Actions, joints, RGB-D, segmentation, object state, and safety telemetry are saved together.</div></div>
    <h2>System</h2><div class="card status" id="status">Starting…</div>
  </aside>
</main><div id="toast"></div>
<script>
const keyMap={w:[0,-1],s:[0,1],a:[1,1],d:[1,-1],r:[2,1],f:[2,-1],q:[3,-1],e:[3,1],arrowup:[4,-1],arrowdown:[4,1],arrowleft:[5,-1],arrowright:[5,1]};
let activeArm=0,driveSpeed=1,driveInFlight=false,queuedDrive=null,driveWasActive=false,inputSource='keyboard_pointer',lastGazeSend=0;
const heldKeys=new Set(),pointerMoves=new Map();
async function post(url,body={}){const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.detail||'Request failed');return data}
function toast(s){const e=document.getElementById('toast');e.textContent=s;e.classList.add('show');setTimeout(()=>e.classList.remove('show'),1600)}
function setArm(arm){stopDrive(false);activeArm=arm;document.getElementById('arm0').classList.toggle('active',arm===0);document.getElementById('arm1').classList.toggle('active',arm===1)}
function setSpeed(speed,button){driveSpeed=speed;document.querySelectorAll('[data-speed]').forEach(x=>x.classList.toggle('active',x===button));updateControlReadout(false,`${button.textContent} speed`)}
function deadzone(value){return Math.abs(value)<0.18?0:Math.sign(value)*(Math.abs(value)-0.18)/0.82}
function gamepadDrive(){const values=Array(6).fill(0);const pads=navigator.getGamepads?navigator.getGamepads():[];const pad=[...pads].find(Boolean);if(!pad)return values;inputSource='gamepad';values[1]-=deadzone(pad.axes[0]||0);values[0]+=deadzone(pad.axes[1]||0);values[5]+=deadzone(pad.axes[2]||0);values[4]+=deadzone(pad.axes[3]||0);values[3]+=(pad.buttons[5]?.value||0)-(pad.buttons[4]?.value||0);values[2]+=(pad.buttons[7]?.value||0)-(pad.buttons[6]?.value||0);return values}
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
function setCamera(name,button){document.getElementById('cameraImage').src=`/video/${name}?t=${Date.now()}`;document.querySelectorAll('[data-camera]').forEach(x=>x.classList.toggle('active',x===button));const labels={endoscope_left:'STEREO ENDOSCOPE · LEFT',endoscope_right:'STEREO ENDOSCOPE · RIGHT',wrist_1:'INSTRUMENT WRIST · 1',wrist_2:'INSTRUMENT WRIST · 2'};document.getElementById('cameraLabel').textContent=labels[name]||name.toUpperCase()}
async function annotatePhase(phase){try{const x=await post('/api/annotation',{phase});toast(x.message)}catch(e){toast(e.message)}}
async function annotateEvent(event){try{const x=await post('/api/annotation',{event});toast('Procedure event saved')}catch(e){toast(e.message)}}
async function resetScene(){try{await post('/api/reset');toast('Scene reset')}catch(e){toast(e.message)}}
async function setAutonomy(mode){try{const x=await post('/api/autonomy',{mode});toast(x.message)}catch(e){toast(e.message)}}
async function takeControl(){stopDrive(false);try{const x=await post('/api/handoff');toast(x.message)}catch(e){toast(e.message)}}
document.querySelectorAll('.move-button').forEach(button=>{button.addEventListener('pointerdown',event=>{event.preventDefault();inputSource='keyboard_pointer';button.setPointerCapture(event.pointerId);pointerMoves.set(event.pointerId,{axis:Number(button.dataset.axis),direction:Number(button.dataset.direction),button});button.classList.add('held');updateDrive()});const release=event=>{const move=pointerMoves.get(event.pointerId);pointerMoves.delete(event.pointerId);if(move&&![...pointerMoves.values()].some(x=>x.button===move.button))move.button.classList.remove('held');updateDrive()};button.addEventListener('pointerup',release);button.addEventListener('pointercancel',release);button.addEventListener('lostpointercapture',release);button.addEventListener('contextmenu',event=>event.preventDefault())});
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;const key=event.key.toLowerCase();if(event.code==='Space'){event.preventDefault();if(!event.repeat)toggleGrip();return}if(key==='escape'){event.preventDefault();stopTool();return}if(!keyMap[key])return;event.preventDefault();inputSource='keyboard_pointer';heldKeys.add(key);updateDrive()});
document.addEventListener('keyup',event=>{if(event.code==='Space'){event.preventDefault();return}const key=event.key.toLowerCase();if(!keyMap[key])return;event.preventDefault();heldKeys.delete(key);updateDrive()});
window.addEventListener('blur',()=>stopDrive(false));document.addEventListener('visibilitychange',()=>{if(document.hidden)stopDrive(false)});
document.getElementById('cameraView').addEventListener('pointermove',event=>{const view=event.currentTarget,rect=view.getBoundingClientRect();const u=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),v=Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height));const cursor=document.getElementById('gazeCursor');cursor.style.left=`${u*100}%`;cursor.style.top=`${v*100}%`;view.classList.add('gaze-on');const now=performance.now();if(now-lastGazeSend>100){lastGazeSend=now;post('/api/gaze',{u,v,valid:true,source:'pointer_attention_proxy'}).catch(()=>{})}});document.getElementById('cameraView').addEventListener('pointerleave',()=>document.getElementById('cameraView').classList.remove('gaze-on'));
async function refresh(){try{const s=await(await fetch('/api/status',{cache:'no-store'})).json();document.getElementById('dot').classList.add('ok');document.getElementById('connection').textContent='Isaac Lab live';document.querySelectorAll('[data-camera]').forEach(button=>button.classList.toggle('hidden',!s.camera_names.includes(button.dataset.camera)));document.getElementById('armPanel').classList.toggle('hidden',s.arms<2);document.getElementById('gripperPanel').classList.toggle('hidden',!s.has_grippers);const grip=s.has_grippers?(s.grippers_open[activeArm]?' · GRIPPER OPEN':' · GRIPPER CLOSED'):'';const moving=s.drive_active?' · MOVING':'';document.getElementById('hud').innerHTML=`<strong>${s.anatomy_showcase||'SURGICAL WORKSPACE'}</strong><br>${s.camera_width}×${s.camera_height} · ${s.render_fps.toFixed(1)} FPS<br>${s.scenario_title}${grip}${moving}`;document.getElementById('recflag').classList.toggle('on',s.recording);const labels={manual:'L0 · Manual',guided:'L1 · Guided',supervised_replay:'L2 · Supervised replay'};document.getElementById('autonomyState').textContent=labels[s.autonomy_mode]||s.autonomy_mode;document.getElementById('manualMode').classList.toggle('active',s.autonomy_mode==='manual');document.getElementById('guidedMode').classList.toggle('active',s.autonomy_mode==='guided');document.getElementById('coachingCue').textContent=s.coaching_cue;document.getElementById('forceMetric').textContent=s.safety?.max_contact_force_n===null?'—':Number(s.safety.max_contact_force_n).toFixed(2);document.getElementById('deformMetric').textContent=s.safety?.max_tissue_displacement_m===null?'—':(Number(s.safety.max_tissue_displacement_m)*1000).toFixed(1);document.getElementById('stressMetric').textContent=s.safety?.max_tissue_stress_pa===null?'—':Number(s.safety.max_tissue_stress_pa).toExponential(1);const ghost=document.getElementById('ghostState');ghost.classList.toggle('on',!!s.reference_ghost?.enabled);ghost.textContent=s.reference_ghost?.enabled?`${s.reference_ghost.point_count} registered path points · ${s.reference_ghost.reference}`:'Clinician path hidden';document.getElementById('status').innerHTML=`Task<br><b>${s.task}</b><br>Scenario: ${s.scenario_title}<br>Robots: ${s.robot_names.join(', ')}<br>Autonomy: ${labels[s.autonomy_mode]||s.autonomy_mode}<br>Phase: ${s.operator_study.procedure_phase}<br>Input: ${s.operator_study.input_source}<br>Annotations: ${s.operator_study.annotation_count}<br>Interventions: ${s.intervention_count}<br>Simulation: ${s.sim_fps.toFixed(1)} Hz<br>Controls: ${s.drive_active?'moving':'ready'}<br>Recorded frames: ${s.recorded_frames}<br>Replay: ${s.replaying?'running':'idle'}`;if(s.last_demo)document.getElementById('lastDemo').innerHTML=`Last saved: <a href="/demos/${s.last_demo}" style="color:#2cd2e8">${s.last_demo}</a>`;}catch(e){document.getElementById('dot').classList.remove('ok');document.getElementById('connection').textContent='Reconnecting…'}}
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

    def __post_init__(self) -> None:
        self.pulse = np.zeros(self.action_dim, dtype=np.float32)
        self.drive = np.zeros(self.action_dim, dtype=np.float32)
        self.grippers_open = [True] * self.arms

    def status(self) -> dict[str, Any]:
        with self.lock:
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
                "grippers_open": self.grippers_open,
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
        command[request.arm * 6 + request.axis] = (0.02 if request.axis < 3 else 0.08) * request.direction
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
        command[request.arm * 6 : request.arm * 6 + 6] = calibrated_values * scales * request.speed
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
    motion = actions[:, : arms * 6].reshape(frame_count, arms, 6)
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
    if actions.shape[1] > arms * 6:
        gripper_values = actions[:, arms * 6 :]
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


def main() -> None:
    args_cli.demo_dir.mkdir(parents=True, exist_ok=True)
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
                prim_path=f"{{ENV_REGEX_NS}}/{wrist_robot_name}/{wrist_tip_name}/DrAnmarWristCamera{wrist_index}",
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
                    pos=(-0.025, 0.0, 0.012),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    convention="ros",
                ),
            ),
        )
    organ_usd = (
        DATA_ROOT
        / "assets/sufia_bc/OR_scene_CTLiver-Prostate-Bladder"
        / "OR_scene_CTLiver-Prostate-Bladder/models/organs/models_topo_blender.usdc"
    )
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
    if organ_usd.is_file():
        env_cfg.scene.liver_showcase = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/LiverShowcase",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.117, -0.0945, -0.144)),
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
    if organ_usd.is_file():
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()
        showcase_path = "/World/envs/env_0/LiverShowcase"
        showcase_prim = stage.GetPrimAtPath(showcase_path)
        showcase_children = [
            child for child in showcase_prim.GetChildren() if child.GetName() != "_materials" and child.IsA(UsdGeom.Imageable)
        ]
        for child in showcase_prim.GetChildren():
            if child.GetName() not in {"Liver_topo_blender", "_materials"} and child.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(child).MakeInvisible()
        material = UsdShade.Material.Define(stage, f"{showcase_path}/DrAnmarLiverMaterial")
        shader = UsdShade.Shader.Define(stage, f"{showcase_path}/DrAnmarLiverMaterial/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.48, 0.055, 0.035))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.38)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        liver_mesh = stage.GetPrimAtPath(f"{showcase_path}/Liver_topo_blender/Liver_topo_blender")
        UsdShade.MaterialBindingAPI.Apply(liver_mesh).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
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
    action_dim = int(env.action_space.shape[-1])
    arms = 2 if "Dual" in args_cli.task else 1
    has_grippers = "Lift" in args_cli.task or "Handover" in args_cli.task
    initial_eye, initial_target = scenario_camera_pose(camera_eye, camera_target, "baseline")
    camera.set_world_poses_from_view(
        torch.tensor([initial_eye.tolist()], device=camera.device),
        torch.tensor([initial_target.tolist()], device=camera.device),
    )
    right_eye = initial_eye + np.asarray((0.0, 0.006, 0.0), dtype=np.float32)
    stereo_right_camera.set_world_poses_from_view(
        torch.tensor([right_eye.tolist()], device=stereo_right_camera.device),
        torch.tensor([initial_target.tolist()], device=stereo_right_camera.device),
    )

    state = SharedState(
        args_cli.task,
        args_cli.camera_width,
        args_cli.camera_height,
        args_cli.demo_dir,
        action_dim,
        arms,
        has_grippers,
        robot_names,
        robot_body_names,
        "Official CT liver",
    )
    state.camera_names = list(camera_sources)
    state.camera_frame_ids = {name: 0 for name in camera_sources}
    try:
        state.camera_intrinsics = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float).tolist()
        state.semantic_labels = camera_semantic_labels(camera)
    except (AttributeError, KeyError, TypeError, RuntimeError):
        pass

    def reset_environment(selected_scenario: str, selected_seed: int) -> None:
        np.random.seed(selected_seed)
        torch.manual_seed(selected_seed)
        env.reset(seed=selected_seed)
        apply_native_object_scenario(objects, selected_scenario, selected_seed)
        profile = SCENARIO_NATIVE_PROFILES.get(selected_scenario, {})
        show_multi_organ = bool(profile.get("show_multi_organ"))
        for child in showcase_children:
            imageable = UsdGeom.Imageable(child)
            if show_multi_organ or child.GetName() == "Liver_topo_blender":
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        with state.lock:
            state.anatomy_showcase = "Official CT liver + pelvic anatomy" if show_multi_organ else "Official CT liver"
        scenario_eye, scenario_target = scenario_camera_pose(camera_eye, camera_target, selected_scenario)
        camera.set_world_poses_from_view(
            torch.tensor([scenario_eye.tolist()], device=camera.device),
            torch.tensor([scenario_target.tolist()], device=camera.device),
        )
        scenario_right_eye = scenario_eye + np.asarray((0.0, 0.006, 0.0), dtype=np.float32)
        stereo_right_camera.set_world_poses_from_view(
            torch.tensor([scenario_right_eye.tolist()], device=stereo_right_camera.device),
            torch.tensor([scenario_target.tolist()], device=stereo_right_camera.device),
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
                    start = arm * 6
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
                gripper_offset = state.arms * 6
                for arm, is_open in enumerate(grippers_open):
                    action_np[gripper_offset + arm] = 1.0 if is_open else -1.0

        actions = torch.from_numpy(action_np).to(device=env.unwrapped.device).reshape(1, -1)
        with torch.inference_mode():
            _observations, reward, terminated, truncated, info = env.step(actions)
        environment_reward = scalar_value(reward)
        environment_terminated = bool(scalar_value(terminated))
        environment_truncated = bool(scalar_value(truncated))
        environment_success = native_success_from_info(info)

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
