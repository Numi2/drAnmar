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
from PIL import Image

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils import parse_env_cfg

import orbit.surgical.tasks  # noqa: F401


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
    .view{position:relative;overflow:hidden;background:#020608;display:flex;align-items:center;justify-content:center}.view img{width:100%;height:100%;object-fit:contain}
    .hud{position:absolute;left:16px;top:16px;padding:10px 13px;border:1px solid #ffffff24;border-radius:8px;background:#051016c9;backdrop-filter:blur(6px);font:12px/1.5 ui-monospace,SFMono-Regular,Menlo;color:#cfe7eb}.hud strong{color:var(--cyan)}
    .recflag{display:none;position:absolute;right:18px;top:18px;color:#fff;background:#c91f2f;padding:8px 12px;border-radius:99px;font-size:12px;font-weight:900;letter-spacing:.08em}.recflag.on{display:block}
    aside{overflow:auto;padding:17px;background:var(--panel);border-left:1px solid var(--line)}
    h2{font-size:12px;letter-spacing:.14em;color:#a9c1ca;margin:3px 0 11px;text-transform:uppercase}.card{border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:13px;background:#0a171e}
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.grid.two{grid-template-columns:repeat(2,1fr)}
    button{min-height:42px;border:1px solid #315462;border-radius:7px;background:#10252e;color:var(--ink);font-weight:750;cursor:pointer;touch-action:manipulation;user-select:none;-webkit-user-select:none}button:hover{border-color:var(--cyan);background:#153540}button:active{transform:translateY(1px);background:var(--cyan2)}
    button.primary{background:var(--cyan);border-color:var(--cyan);color:#041014}button.danger{background:#31171c;border-color:#74414a;color:#ffabb2}button.stop{grid-column:1/-1;background:#27323a;border-color:#5f727c}
    .speedbar{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:11px}.speedbar button{min-height:35px;font-size:11px}.speedbar button.active{background:var(--cyan);border-color:var(--cyan);color:#041014}.dpad{display:grid;grid-template-columns:repeat(3,1fr);grid-template-areas:"blank up blank2" "left stop right" "blank3 down blank4";gap:6px}.dpad .up{grid-area:up}.dpad .left{grid-area:left}.dpad .stop-center{grid-area:stop;min-height:54px;background:#26343b;border-color:#617681}.stop-center small{display:block;color:#9bb0b8;font-size:10px;margin-top:2px}.dpad .right{grid-area:right}.dpad .down{grid-area:down}.depthgrid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}.anglegrid{display:grid;grid-template-columns:1fr 1fr;gap:6px}.move-button{min-height:54px;touch-action:none;position:relative}.move-button small{display:block;color:#86a5af;font-size:10px;margin-top:2px}.move-button.held{background:var(--cyan);border-color:var(--cyan);color:#041014;box-shadow:0 0 16px #2cd2e855}.move-button.held small{color:#0a5260}.control-readout{display:flex;align-items:center;gap:7px;margin-top:10px;color:var(--muted);font-size:11px}.control-readout i{width:7px;height:7px;border-radius:50%;background:#536a73}.control-readout.moving{color:var(--green)}.control-readout.moving i{background:var(--green);box-shadow:0 0 10px #42e49b99}
    .hint{color:var(--muted);font-size:12px;margin-top:9px}.status{font:12px/1.65 ui-monospace,SFMono-Regular,Menlo;color:#bdd2d8;word-break:break-word}.status b{color:var(--green)}.hidden{display:none}.arm.active{background:var(--cyan);color:#041014;border-color:var(--cyan)}
    #toast{position:fixed;left:50%;bottom:20px;translate:-50% 20px;opacity:0;background:#e9f8fa;color:#061116;border-radius:8px;padding:10px 15px;font-weight:750;transition:.2s;pointer-events:none}#toast.show{opacity:1;translate:-50% 0}
    @media(max-width:880px){main{display:block;height:auto}.view{height:52vh}aside{border-left:0;border-top:1px solid var(--line)}header{padding:0 12px}.tag{display:none}}
  </style>
</head>
<body>
<header><div class="brand">DR.<span>ANMAR</span></div><div class="tag">SIMULATION ONLY · NO PHYSICAL ROBOT OUTPUT</div><div class="live"><i id="dot" class="dot"></i><span id="connection">Connecting…</span></div></header>
<main>
  <section class="view"><img src="/video" alt="Live simulated endoscopic view"><div class="hud"><strong>ENDOSCOPE A</strong><br><span id="hud">Waiting for Isaac Lab…</span></div><div id="recflag" class="recflag">● RECORDING</div></section>
  <aside>
    <div id="armPanel" class="hidden"><h2>Active instrument</h2><div class="card"><div class="grid two"><button id="arm0" class="arm active" onclick="setArm(0)">Instrument 1</button><button id="arm1" class="arm" onclick="setArm(1)">Instrument 2</button></div></div></div>
    <h2>Movement speed</h2><div class="card"><div class="speedbar"><button data-speed="0.45" onclick="setSpeed(0.45,this)">Precision</button><button class="active" data-speed="1" onclick="setSpeed(1,this)">Normal</button><button data-speed="1.8" onclick="setSpeed(1.8,this)">Fast</button></div><div class="hint">Hold controls to move. Release to stop. Fast is for open space; Precision is for grasping.</div></div>
    <h2>Tool position</h2><div class="card"><div class="dpad">
      <button class="move-button up" data-axis="2" data-direction="1">↑ Up<small>R</small></button>
      <button class="move-button left" data-axis="1" data-direction="1">← Left<small>A</small></button>
      <button class="stop-center" onclick="stopTool()">■ Stop<small>Space</small></button>
      <button class="move-button right" data-axis="1" data-direction="-1">Right →<small>D</small></button>
      <button class="move-button down" data-axis="2" data-direction="-1">↓ Down<small>F</small></button>
    </div><div class="depthgrid"><button class="move-button" data-axis="0" data-direction="-1">Toward patient<small>W</small></button><button class="move-button" data-axis="0" data-direction="1">Away from patient<small>S</small></button></div><div id="controlReadout" class="control-readout"><i></i><span>Ready · hold a control to move</span></div></div>
    <h2>Tool angle</h2><div class="card"><div class="anglegrid">
      <button class="move-button" data-axis="3" data-direction="-1">↶ Roll left<small>Q</small></button><button class="move-button" data-axis="3" data-direction="1">Roll right ↷<small>E</small></button>
      <button class="move-button" data-axis="4" data-direction="-1">Pitch up<small>↑</small></button><button class="move-button" data-axis="4" data-direction="1">Pitch down<small>↓</small></button>
      <button class="move-button" data-axis="5" data-direction="-1">← Yaw left<small>←</small></button><button class="move-button" data-axis="5" data-direction="1">Yaw right →<small>→</small></button>
    </div><div class="hint">Keyboard: WASD + R/F for position, arrows + Q/E for angle. Standard gamepads are supported.</div></div>
    <div id="gripperPanel"><h2>Gripper</h2><div class="card"><div class="grid two"><button onclick="grip(true)">Open</button><button class="primary" onclick="grip(false)">Close / grasp</button></div></div></div>
    <h2>Demonstration</h2><div class="card"><div class="grid two"><button id="record" class="primary" onclick="recording(true)">Start recording</button><button onclick="recording(false)">Stop & save</button><button onclick="replay()">Replay last</button><button onclick="resetScene()">Reset scene</button></div><div class="hint" id="lastDemo">Actions, joints, tool and needle poses are saved together.</div></div>
    <h2>System</h2><div class="card status" id="status">Starting…</div>
  </aside>
</main><div id="toast"></div>
<script>
const keyMap={w:[0,-1],s:[0,1],a:[1,1],d:[1,-1],r:[2,1],f:[2,-1],q:[3,-1],e:[3,1],arrowup:[4,-1],arrowdown:[4,1],arrowleft:[5,-1],arrowright:[5,1]};
let activeArm=0,driveSpeed=1,driveInFlight=false,queuedDrive=null,driveWasActive=false;
const heldKeys=new Set(),pointerMoves=new Map();
async function post(url,body={}){const r=await fetch(url,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.detail||'Request failed');return data}
function toast(s){const e=document.getElementById('toast');e.textContent=s;e.classList.add('show');setTimeout(()=>e.classList.remove('show'),1600)}
function setArm(arm){stopDrive(false);activeArm=arm;document.getElementById('arm0').classList.toggle('active',arm===0);document.getElementById('arm1').classList.toggle('active',arm===1)}
function setSpeed(speed,button){driveSpeed=speed;document.querySelectorAll('[data-speed]').forEach(x=>x.classList.toggle('active',x===button));updateControlReadout(false,`${button.textContent} speed`)}
function deadzone(value){return Math.abs(value)<0.18?0:Math.sign(value)*(Math.abs(value)-0.18)/0.82}
function gamepadDrive(){const values=Array(6).fill(0);const pads=navigator.getGamepads?navigator.getGamepads():[];const pad=[...pads].find(Boolean);if(!pad)return values;values[1]-=deadzone(pad.axes[0]||0);values[0]+=deadzone(pad.axes[1]||0);values[5]+=deadzone(pad.axes[2]||0);values[4]+=deadzone(pad.axes[3]||0);values[3]+=(pad.buttons[5]?.value||0)-(pad.buttons[4]?.value||0);values[2]+=(pad.buttons[7]?.value||0)-(pad.buttons[6]?.value||0);return values}
function buildDrive(){const values=gamepadDrive();heldKeys.forEach(key=>{const move=keyMap[key];if(move)values[move[0]]+=move[1]});pointerMoves.forEach(move=>values[move.axis]+=move.direction);return values.map(value=>Math.max(-1,Math.min(1,value)))}
function updateControlReadout(moving,label){const readout=document.getElementById('controlReadout');readout.classList.toggle('moving',moving);readout.querySelector('span').textContent=moving?(label||'Moving · release to stop'):'Ready · hold a control to move'}
async function flushDrive(){if(driveInFlight||!queuedDrive)return;const next=queuedDrive;queuedDrive=null;driveInFlight=true;try{await post('/api/drive',{values:next,arm:activeArm,speed:driveSpeed})}catch(e){toast(e.message)}finally{driveInFlight=false;if(queuedDrive)flushDrive()}}
function sendDrive(values){queuedDrive=values;flushDrive()}
function updateDrive(){const values=buildDrive();const active=values.some(value=>Math.abs(value)>0.01);if(active||driveWasActive)sendDrive(values);driveWasActive=active;updateControlReadout(active,active?'Moving · release to stop':null)}
function stopDrive(showToast=true){heldKeys.clear();pointerMoves.clear();document.querySelectorAll('.move-button.held').forEach(x=>x.classList.remove('held'));driveWasActive=false;sendDrive(Array(6).fill(0));updateControlReadout(false);if(showToast)toast('Tool stopped')}
async function stopTool(){stopDrive();try{await post('/api/stop')}catch(e){toast(e.message)}}
async function grip(open){try{await post('/api/gripper',{open,arm:activeArm});toast(open?'Gripper open':'Gripper closed')}catch(e){toast(e.message)}}
async function recording(start){try{await post(start?'/api/record/start':'/api/record/stop');toast(start?'Recording started':'Saving demonstration…')}catch(e){toast(e.message)}}
async function replay(){try{const x=await post('/api/replay-last');toast(x.message)}catch(e){toast(e.message)}}
async function resetScene(){try{await post('/api/reset');toast('Scene reset')}catch(e){toast(e.message)}}
document.querySelectorAll('.move-button').forEach(button=>{button.addEventListener('pointerdown',event=>{event.preventDefault();button.setPointerCapture(event.pointerId);pointerMoves.set(event.pointerId,{axis:Number(button.dataset.axis),direction:Number(button.dataset.direction),button});button.classList.add('held');updateDrive()});const release=event=>{const move=pointerMoves.get(event.pointerId);pointerMoves.delete(event.pointerId);if(move&&![...pointerMoves.values()].some(x=>x.button===move.button))move.button.classList.remove('held');updateDrive()};button.addEventListener('pointerup',release);button.addEventListener('pointercancel',release);button.addEventListener('lostpointercapture',release);button.addEventListener('contextmenu',event=>event.preventDefault())});
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;const key=event.key.toLowerCase();if(key===' '){event.preventDefault();stopTool();return}if(!keyMap[key])return;event.preventDefault();heldKeys.add(key);updateDrive()});
document.addEventListener('keyup',event=>{const key=event.key.toLowerCase();if(!keyMap[key])return;event.preventDefault();heldKeys.delete(key);updateDrive()});
window.addEventListener('blur',()=>stopDrive(false));document.addEventListener('visibilitychange',()=>{if(document.hidden)stopDrive(false)});
async function refresh(){try{const s=await(await fetch('/api/status',{cache:'no-store'})).json();document.getElementById('dot').classList.add('ok');document.getElementById('connection').textContent='Isaac Lab live';document.getElementById('armPanel').classList.toggle('hidden',s.arms<2);document.getElementById('gripperPanel').classList.toggle('hidden',!s.has_grippers);const grip=s.has_grippers?(s.grippers_open[activeArm]?' · GRIPPER OPEN':' · GRIPPER CLOSED'):'';const moving=s.drive_active?' · MOVING':'';document.getElementById('hud').innerHTML=`<strong>${s.anatomy_showcase||'SURGICAL WORKSPACE'}</strong><br>${s.camera_width}×${s.camera_height} · ${s.render_fps.toFixed(1)} FPS<br>Step ${s.sim_step}${grip}${moving}`;document.getElementById('recflag').classList.toggle('on',s.recording);document.getElementById('status').innerHTML=`Task<br><b>${s.task}</b><br>Showcase: ${s.anatomy_showcase||'none'}<br>Robots: ${s.robot_names.join(', ')}<br>Action dimensions: ${s.action_dim}<br>Simulation: ${s.sim_fps.toFixed(1)} Hz<br>Controls: ${s.drive_active?'moving':'ready'}<br>Recorded frames: ${s.recorded_frames}<br>Replay: ${s.replaying?'running':'idle'}`;if(s.last_demo)document.getElementById('lastDemo').innerHTML=`Last saved: <a href="/demos/${s.last_demo}" style="color:#2cd2e8">${s.last_demo}</a>`;}catch(e){document.getElementById('dot').classList.remove('ok');document.getElementById('connection').textContent='Reconnecting…'}}
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


class GripperRequest(BaseModel):
    open: bool
    arm: int = 0


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
    anatomy_showcase: str | None = None
    instance_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    lock: threading.Lock = field(default_factory=threading.Lock)
    wake_event: threading.Event = field(default_factory=threading.Event)
    frame_jpeg: bytes = b""
    frame_id: int = 0
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
                "frame_id": self.frame_id,
                "render_fps": self.render_fps,
                "sim_fps": self.sim_fps,
                "sim_step": self.sim_step,
                "action_dim": self.action_dim,
                "arms": self.arms,
                "has_grippers": self.has_grippers,
                "robot_names": self.robot_names,
                "anatomy_showcase": self.anatomy_showcase,
                "grippers_open": self.grippers_open,
                "recording": self.recording,
                "recorded_frames": self.recorded_frames,
                "last_demo": self.last_demo,
                "replaying": self.replaying,
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
        command = np.zeros(state.action_dim, dtype=np.float32)
        scales = np.asarray((0.006, 0.006, 0.006, 0.03, 0.03, 0.03), dtype=np.float32)
        command[request.arm * 6 : request.arm * 6 + 6] = values * scales * request.speed
        active = bool(np.any(values))
        with state.lock:
            state.drive = command
            state.drive_until = time.monotonic() + 0.30 if active else 0.0
            if active:
                state.replay_request = "stop"
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

    @app.post("/api/reset")
    def reset() -> dict[str, bool]:
        with state.lock:
            state.reset_requested = True
            state.drive.fill(0.0)
            state.drive_until = 0.0
            state.replay_request = "stop"
        state.wake_event.set()
        return {"ok": True}

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
            name = state.last_demo
        state.wake_event.set()
        return {"ok": True, "message": f"Replaying {name}"}

    @app.get("/api/demos")
    def demos() -> dict[str, Any]:
        files = sorted(state.demo_dir.glob("dr_anmar_*.npz"), reverse=True)
        return {"demos": [item.name for item in files[:50]]}

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

    @app.get("/frame.jpg")
    def still_frame() -> Response:
        with state.lock:
            jpeg = state.frame_jpeg
        if not jpeg:
            raise HTTPException(503, "The first camera frame is not ready")
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app


def encode_jpeg(rgb: torch.Tensor) -> bytes:
    array = rgb[..., :3].detach().cpu().numpy()
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8, copy=False)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "JPEG", quality=86, optimize=False)
    return buffer.getvalue()


def save_demo(state: SharedState, frames: list[dict[str, np.ndarray]], started_at: str) -> str | None:
    if not frames:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    task_slug = state.task.lower().replace("isaac-", "").replace("-v0", "").replace("-", "_")
    name = f"dr_anmar_{task_slug}_{stamp}.npz"
    path = state.demo_dir / name
    keys = tuple(frames[0].keys())
    arrays = {key: np.stack([frame[key] for frame in frames]) for key in keys}
    np.savez_compressed(path, **arrays)
    manifest = {
        "schema": "dr.anmar.demonstration.v1",
        "simulation_only": True,
        "task": state.task,
        "robots": state.robot_names,
        "action_dim": state.action_dim,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "frames": len(frames),
        "control_hz": 50,
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
        "data_file": name,
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return name


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
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.0,
            focus_distance=0.25,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=tuple(camera_eye.tolist()), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
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
    robot_names = sorted(scene.articulations.keys())
    robots = {name: scene[name] for name in robot_names}
    object_names = sorted(scene.rigid_objects.keys())
    objects = {name: scene[name] for name in object_names}
    if organ_usd.is_file():
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()
        showcase_path = "/World/envs/env_0/LiverShowcase"
        showcase_prim = stage.GetPrimAtPath(showcase_path)
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
    action_dim = int(env.action_space.shape[-1])
    arms = 2 if "Dual" in args_cli.task else 1
    has_grippers = "Lift" in args_cli.task or "Handover" in args_cli.task
    camera.set_world_poses_from_view(
        torch.tensor([camera_eye.tolist()], device=camera.device),
        torch.tensor([camera_target.tolist()], device=camera.device),
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
        "Official CT liver",
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
    demo_started_at = ""
    demo_started_monotonic = 0.0
    replay_actions: np.ndarray | None = None
    replay_index = 0
    last_loop_time = time.monotonic()
    last_fps_time = last_loop_time
    fps_steps = 0
    last_frame_time = 0.0
    frame_count = 0
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
            if state.drive_until > loop_started:
                manual_action = state.drive.copy()
            elif state.pulse_steps > 0:
                manual_action = state.pulse.copy()
                state.pulse_steps -= 1
            else:
                manual_action = np.zeros(state.action_dim, dtype=np.float32)
            grippers_open = list(state.grippers_open)

        if reset_requested:
            with torch.inference_mode():
                env.reset()
                camera.set_world_poses_from_view(
                    torch.tensor([camera_eye.tolist()], device=camera.device),
                    torch.tensor([camera_target.tolist()], device=camera.device),
                )

        if record_request == "start":
            demo_frames.clear()
            demo_started_at = datetime.now(timezone.utc).isoformat()
            demo_started_monotonic = time.monotonic()
            with state.lock:
                state.recording = True
                state.recorded_frames = 0
        elif record_request == "stop":
            name = save_demo(state, demo_frames, demo_started_at)
            with state.lock:
                state.recording = False
                state.last_demo = name or state.last_demo

        if replay_request == "stop":
            replay_actions = None
            replay_index = 0
            with state.lock:
                state.replaying = False
        elif replay_request:
            replay_path = args_cli.demo_dir / Path(replay_request).name
            try:
                replay_actions = np.load(replay_path)["actions"].astype(np.float32)
                replay_index = 0
                with torch.inference_mode():
                    env.reset()
                with state.lock:
                    state.replaying = True
            except (OSError, KeyError, ValueError):
                replay_actions = None
                with state.lock:
                    state.replaying = False

        if replay_actions is not None and replay_index < len(replay_actions):
            action_np = replay_actions[replay_index]
            replay_index += 1
        else:
            if replay_actions is not None:
                replay_actions = None
                with state.lock:
                    state.replaying = False
            action_np = manual_action.copy()
            if state.has_grippers:
                gripper_offset = state.arms * 6
                for arm, is_open in enumerate(grippers_open):
                    action_np[gripper_offset + arm] = 1.0 if is_open else -1.0

        actions = torch.from_numpy(action_np).to(device=env.unwrapped.device).reshape(1, -1)
        with torch.inference_mode():
            env.step(actions)

        with state.lock:
            is_recording = state.recording
        interactive_active = bool(np.any(manual_action)) or replay_actions is not None or is_recording
        if is_recording:
            frame = {
                "time_s": np.array(time.monotonic() - demo_started_monotonic, dtype=np.float64),
                "actions": action_np.copy(),
            }
            for name, robot in robots.items():
                frame[f"{name}_joint_positions"] = robot.data.joint_pos[0].detach().cpu().numpy().copy()
                frame[f"{name}_joint_velocities"] = robot.data.joint_vel[0].detach().cpu().numpy().copy()
                frame[f"{name}_body_positions_w"] = robot.data.body_pos_w[0].detach().cpu().numpy().copy()
                frame[f"{name}_body_quaternions_w"] = robot.data.body_quat_w[0].detach().cpu().numpy().copy()
            for name, rigid_object in objects.items():
                frame[f"{name}_position_w"] = rigid_object.data.root_pos_w[0].detach().cpu().numpy().copy()
                frame[f"{name}_quaternion_w"] = rigid_object.data.root_quat_w[0].detach().cpu().numpy().copy()
            demo_frames.append(frame)
            with state.lock:
                state.recorded_frames = len(demo_frames)

        now = time.monotonic()
        fps_steps += 1
        frame_interval = 0.04 if interactive_active else 0.20
        if camera.data.output.get("rgb") is not None and now - last_frame_time >= frame_interval:
            jpeg = encode_jpeg(camera.data.output["rgb"][0])
            frame_count += 1
            elapsed = max(now - last_frame_time, 1e-6)
            with state.lock:
                state.frame_jpeg = jpeg
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
        name = save_demo(state, demo_frames, demo_started_at)
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
