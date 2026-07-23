# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Browser bridge around unmodified SonoGym Isaac Lab environments.

The bridge only transports user actions and native observations.  It does not
replace SonoGym rewards, kinematics, ultrasound generation, scene assets,
contacts, simulation stepping, or safety constraints.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import signal
import sys
import threading
import time
import traceback
import types
import uuid
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dr_anmar_operator import ACCESS_COOKIE, OPERATOR_HEADER, OperatorLease, access_is_authorized


parser = argparse.ArgumentParser(description="Dr.Anmar browser bridge for SonoGym")
parser.add_argument("--sonogym-root", type=Path, required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--procedure-id", required=True)
parser.add_argument("--title", required=True)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=2361)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.sonogym_root = args.sonogym_root.expanduser().resolve()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


TASK_META = {
    "Isaac-robot-US-guidance-v0": {
        "action_dimensions": 3,
        "objective": "Find and hold the transverse ultrasound plane through the centre of L4.",
        "interaction": "SonoGym native CT-derived ultrasound navigation and KUKA probe control.",
        "robot": "KUKA LBR with ultrasound probe",
        "instrument": "Robotic ultrasound probe",
        "difficulty": "Foundation",
        "anatomy": "SonoGym CT-derived lumbar patient · L4 vertebra",
        "steps": (
            ("survey", "Survey the lumbar surface"),
            ("orient", "Rotate toward a transverse L4 view"),
            ("centre", "Centre the L4 anatomy in ultrasound"),
            ("hold", "Hold the target plane steadily"),
        ),
    },
    "Isaac-robot-US-reconstruction-v0": {
        "action_dimensions": 4,
        "objective": "Acquire enough complementary ultrasound views to reconstruct the L4 surface.",
        "interaction": "SonoGym native surface-reconstruction observation and coverage reward.",
        "robot": "KUKA LBR with ultrasound probe",
        "instrument": "Robotic ultrasound probe",
        "difficulty": "Intermediate",
        "anatomy": "SonoGym CT-derived lumbar patient · L4 vertebra",
        "steps": (
            ("localize", "Localize L4"),
            ("sweep", "Sweep across complementary surface views"),
            ("inspect", "Inspect uncovered reconstruction regions"),
            ("complete", "Finish with stable surface coverage"),
        ),
    },
    "Isaac-robot-US-guided-surgery-v0": {
        "action_dimensions": 6,
        "objective": "Use ultrasound localization to guide the orthopedic instrument toward L4 while respecting SonoGym safety constraints.",
        "interaction": "SonoGym native dual-robot ultrasound-guided surgery task and safe-action cost.",
        "robot": "FR3 ultrasound robot + KUKA orthopedic instrument robot",
        "instrument": "Ultrasound probe and orthopedic drill trajectory",
        "difficulty": "Advanced research",
        "anatomy": "SonoGym CT-derived lumbar patient · L4 vertebra",
        "steps": (
            ("localize", "Localize L4 with ultrasound"),
            ("align", "Align the planned instrument trajectory"),
            ("advance", "Advance under image guidance"),
            ("verify", "Verify target and safety state"),
        ),
    },
}


class BridgeState:
    def __init__(self, action_dimensions: int) -> None:
        self.lock = threading.Lock()
        self.action = np.zeros(action_dimensions, dtype=np.float32)
        self.action_updated = 0.0
        self.stop_requested = False
        self.frame_jpeg: bytes | None = None
        self.frame_id = 0
        self.step = 0
        self.fps = 0.0
        self.reward: float | None = None
        self.applied_action = np.zeros(action_dimensions, dtype=np.float32)
        self.native_feedback: dict[str, Any] = {}
        self.last_error: str | None = None
        self.started_at = time.time()


meta = TASK_META[args.task]
state = BridgeState(meta["action_dimensions"])
instance_id = f"sonogym-{uuid.uuid4().hex[:12]}"
operator_lease = OperatorLease()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def status_payload() -> dict[str, Any]:
    with state.lock:
        frame_id = state.frame_id
        return {
            "task": args.task,
            "instance_id": instance_id,
            "worker_kind": "sonogym_native",
            "frame_id": frame_id,
            "fps": round(state.fps, 1),
            "reward": state.reward,
            "applied_action": state.applied_action.tolist(),
            "command_active": bool(np.any(np.abs(state.applied_action) > 1.0e-5)),
            "native_feedback": dict(state.native_feedback),
            "last_error": state.last_error,
            "recording": False,
            "recorded_frames": 0,
            "autonomy_mode": "manual",
            "expert_demonstration": {"status": "idle"},
            "render_contract": {"ready": frame_id > 0, "view": "native_ultrasound"},
            "runtime_provenance": {
                "provider": "SonoGym",
                "task": args.task,
                "source_root": str(args.sonogym_root),
                "simulator": "Isaac Lab",
            },
            "anatomy_scene_id": "sonogym-lumbar-l4",
            "procedure": {
                "id": args.procedure_id,
                "title": args.title,
                "objective": meta["objective"],
                "interaction": meta["interaction"],
                "robot": meta["robot"],
                "instrument": meta["instrument"],
                "difficulty": meta["difficulty"],
                "anatomy_focus": meta["anatomy"],
                "anatomy_title": meta["anatomy"],
                "steps": [
                    {"id": step_id, "title": title, "instruction": title, "status": "pending"}
                    for step_id, title in meta["steps"]
                ],
            },
        }


def page_html() -> bytes:
    action_dimensions = meta["action_dimensions"]
    controls = {
        3: "W/S scan forward and back · A/D scan left and right · Q/E rotate probe",
        4: "W/S and A/D sweep · Q/E tilt · R/F rotate",
        6: "W/S depth · A/D lateral · R/F height · Q/E roll · arrow keys pitch/yaw",
    }[action_dimensions]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{args.title}</title><style>
:root{{--bg:#020608;--panel:#071923;--line:#244651;--cyan:#35d0e6;--green:#57e5aa;--text:#edf7f8;--muted:#8ca7af}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif;overflow:hidden}}
.shell{{height:100vh;min-height:0;display:grid;grid-template-rows:58px minmax(0,1fr) auto}}header{{display:flex;align-items:center;justify-content:space-between;padding:0 20px;background:#06141c;border-bottom:1px solid var(--line)}}
header b{{font-size:17px}}header span,.hint{{color:var(--muted)}}.live{{color:var(--green)}}main{{display:grid;place-items:center;min-height:0;overflow:hidden;padding:10px;background:radial-gradient(circle at center,#0a2631 0,#020608 68%)}}
.scan{{width:min(100%,calc((100vh - 184px)*4/3));aspect-ratio:4/3;max-height:100%;border:1px solid var(--line);background:#000;position:relative;display:grid;place-items:center}}
.scan img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;image-rendering:auto}}.badge{{position:absolute;left:12px;top:12px;padding:7px 9px;background:#06141ce8;border:1px solid var(--line);font:12px ui-monospace,monospace}}
footer{{min-height:116px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:16px;align-items:center;padding:10px 20px;background:var(--panel);border-top:1px solid var(--line)}}
.control-state{{display:flex;align-items:center;gap:8px;margin-bottom:7px;font-weight:750;color:var(--green)}}.control-state i{{width:8px;height:8px;border-radius:50%;background:currentColor;box-shadow:0 0 10px currentColor}}.control-state.moving{{color:var(--cyan)}}
.keys{{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:6px}}button{{min-width:54px;height:44px;border:1px solid #376371;background:#0a202a;color:var(--text);font-weight:800;cursor:pointer;border-radius:3px}}button:hover{{border-color:var(--cyan)}}button.active{{background:var(--cyan);border-color:var(--cyan);color:#001117;box-shadow:0 0 18px #35d0e655}}button.reset{{min-width:110px}}.native-progress{{font:12px ui-monospace,monospace;color:var(--muted);margin-top:4px}}
@media(max-width:800px){{header{{padding:0 12px}}.shell{{grid-template-rows:54px minmax(0,1fr) auto}}.scan{{width:min(100%,calc((100vh - 230px)*4/3))}}footer{{grid-template-columns:1fr;padding:9px 12px}}button.reset{{position:absolute;right:12px;bottom:10px}}.hint{{padding-right:120px}}}}
</style></head><body tabindex="0"><div class="shell"><header><div><b>{args.title}</b><div class="hint">{meta['anatomy']}</div></div><div class="live">● SonoGym native</div></header>
<main><div class="scan"><img id="frame" alt="Live SonoGym ultrasound"><div class="badge" id="telemetry">Starting native environment…</div></div></main>
<footer><div><div class="control-state" id="controlState"><i></i><span>Keyboard ready — hold a movement key</span></div><div class="keys" id="keys"></div><div class="hint">{controls}</div><div class="native-progress" id="nativeProgress">Waiting for native simulator state…</div></div><button class="reset" onclick="resetRoom()">Reset</button></footer></div>
<script>
const operatorId=new URLSearchParams(location.search).get('operator')||`browser-${{Date.now().toString(36)}}-${{Math.random().toString(36).slice(2)}}`,dims={action_dimensions},held=new Set();let lastCommand='',lastCommandAt=0,actionError='';
const maps=dims===3?{{w:[0,1],s:[0,-1],a:[1,-1],d:[1,1],q:[2,-1],e:[2,1]}}:dims===4?{{w:[0,1],s:[0,-1],a:[1,-1],d:[1,1],q:[2,-1],e:[2,1],r:[3,1],f:[3,-1]}}:{{w:[0,1],s:[0,-1],a:[1,-1],d:[1,1],r:[2,1],f:[2,-1],q:[3,-1],e:[3,1],arrowup:[4,1],arrowdown:[4,-1],arrowleft:[5,-1],arrowright:[5,1]}};
const labels={{arrowup:'↑',arrowdown:'↓',arrowleft:'←',arrowright:'→'}},commandNames=dims===3?{{w:'Scanning forward',s:'Scanning back',a:'Scanning left',d:'Scanning right',q:'Rotating left',e:'Rotating right'}}:dims===4?{{w:'Sweeping forward',s:'Sweeping back',a:'Sweeping left',d:'Sweeping right',q:'Tilting left',e:'Tilting right',r:'Rotating clockwise',f:'Rotating counter-clockwise'}}:{{w:'Advancing in depth',s:'Retracting in depth',a:'Moving left',d:'Moving right',r:'Moving up',f:'Moving down',q:'Rolling left',e:'Rolling right',arrowup:'Pitching up',arrowdown:'Pitching down',arrowleft:'Yawing left',arrowright:'Yawing right'}};
document.getElementById('keys').innerHTML=Object.keys(maps).map(k=>`<button data-key="${{k}}" aria-label="${{commandNames[k]}}">${{labels[k]||k.toUpperCase()}}</button>`).join('');
function renderControlState(){{const names=[...held].map(k=>commandNames[k]).filter(Boolean),el=document.getElementById('controlState'),recent=lastCommand&&Date.now()-lastCommandAt<1800;el.classList.toggle('moving',names.length>0);el.querySelector('span').textContent=actionError||names.length&&names.join(' + ')||recent&&`Last command — ${{lastCommand}}`||'Keyboard ready — hold a movement key';document.querySelectorAll('[data-key]').forEach(b=>b.classList.toggle('active',held.has(b.dataset.key)))}}
function send(){{const a=Array(dims).fill(0);for(const k of held){{const m=maps[k];if(m)a[m[0]]+=m[1]}}renderControlState();fetch('/api/action',{{method:'POST',headers:{{'content-type':'application/json','x-dr-anmar-operator':operatorId}},body:JSON.stringify({{action:a}})}}).then(async r=>{{if(!r.ok){{const body=await r.json().catch(()=>({{}}));throw Error(body.detail||body.error||'Control command was rejected')}}actionError=''}}).catch(e=>{{actionError=e.message;renderControlState()}})}}
function setHeld(key,down){{const k=String(key||'').toLowerCase();if(!maps[k])return;if(down){{held.add(k);lastCommand=commandNames[k];lastCommandAt=Date.now()}}else held.delete(k);send()}}
function releaseAll(){{if(!held.size)return;held.clear();send()}}
addEventListener('keydown',e=>{{const k=e.key.toLowerCase();if(maps[k]){{e.preventDefault();setHeld(k,true)}}}});addEventListener('keyup',e=>{{const k=e.key.toLowerCase();if(maps[k]){{e.preventDefault();setHeld(k,false)}}}});addEventListener('blur',releaseAll);
addEventListener('message',e=>{{let sameHost=false;try{{sameHost=new URL(e.origin).hostname===location.hostname}}catch(_error){{}}if(e.source!==parent||!sameHost||e.data?.type!=='dr-anmar-sonogym-key')return;if(e.data.releaseAll)releaseAll();else setHeld(e.data.key,!!e.data.down)}});
document.querySelectorAll('[data-key]').forEach(b=>{{b.onpointerdown=e=>{{e.preventDefault();b.setPointerCapture?.(e.pointerId);setHeld(b.dataset.key,true)}};b.onpointerup=b.onpointercancel=b.onpointerleave=()=>setHeld(b.dataset.key,false)}});
setInterval(()=>{{if(held.size)send();else renderControlState()}},75);
async function resetRoom(){{releaseAll();let parentOrigin='';try{{parentOrigin=new URL(document.referrer).origin}}catch(_error){{}}if(parent!==window&&parentOrigin){{parent.postMessage({{type:'dr-anmar-sonogym-reset'}},parentOrigin);document.getElementById('nativeProgress').textContent='Restarting the native SonoGym room…';return}}const response=await fetch('/api/reset',{{method:'POST',headers:{{'x-dr-anmar-operator':operatorId}}}});if(!response.ok)document.getElementById('nativeProgress').textContent='Open this lab through Dr.Anmar to restart it safely.'}}
const img=document.getElementById('frame');function nextFrame(){{img.src='/frame.jpg?t='+Date.now()}}img.onload=()=>setTimeout(nextFrame,70);img.onerror=()=>setTimeout(nextFrame,500);nextFrame();
async function status(){{try{{const s=await fetch('/api/status').then(r=>r.json()),feedback=s.native_feedback||{{}};document.getElementById('telemetry').textContent=`${{s.fps}} FPS · native step ${{s.frame_id}}`;let progress='Native action and ultrasound state are live';if(feedback.coverage_ratio!==undefined)progress=`Reconstruction coverage ${{(feedback.coverage_ratio*100).toFixed(1)}}% · path ${{(feedback.path_length||0).toFixed(3)}}`;else if(feedback.tip_to_trajectory_m!==undefined)progress=`Trajectory gap ${{(feedback.tip_to_trajectory_m*1000).toFixed(1)}} mm · insertion ${{(feedback.insertion_m*1000).toFixed(1)}} mm · ${{feedback.safe?'safe envelope':'outside safe envelope'}}`;else if(feedback.distance_to_goal!==undefined)progress=`Native target error ${{feedback.distance_to_goal.toFixed(3)}} · ${{feedback.trend||'tracking'}}`;document.getElementById('nativeProgress').textContent=progress}}catch(e){{}}setTimeout(status,300)}}status();document.body.focus();
</script></body></html>""".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "DrAnmarSonoGym/1"

    def log_message(self, fmt: str, *values: Any) -> None:
        return

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _access_allowed(self) -> bool:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("cookie", ""))
        except Exception:
            return False
        value = cookie.get(ACCESS_COOKIE)
        return access_is_authorized(value.value if value else None)

    def _mutation_allowed(self) -> tuple[bool, str]:
        origin = self.headers.get("origin")
        if not origin:
            return True, ""
        try:
            origin_host = urlparse(origin).hostname
            request_host = self.headers.get("host", "").split(":", 1)[0]
        except ValueError:
            return False, "Invalid request origin"
        if origin_host != request_host:
            return False, "Cross-site state changes are not allowed"
        return operator_lease.claim(self.headers.get(OPERATOR_HEADER))

    def do_GET(self) -> None:  # noqa: N802
        if not self._access_allowed():
            self._send(_json_bytes({"detail": "Dr.Anmar access token required"}), "application/json", 401)
            return
        if self.path == "/" or self.path.startswith("/?"):
            self._send(page_html(), "text/html; charset=utf-8")
            return
        if self.path.startswith("/api/status"):
            self._send(_json_bytes(status_payload()), "application/json")
            return
        if self.path.startswith("/frame.jpg"):
            with state.lock:
                frame = state.frame_jpeg
            if frame is None:
                self._send(b"", "image/jpeg", HTTPStatus.SERVICE_UNAVAILABLE)
            else:
                self._send(frame, "image/jpeg")
            return
        self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if not self._access_allowed():
            self._send(_json_bytes({"detail": "Dr.Anmar access token required"}), "application/json", 401)
            return
        allowed, detail = self._mutation_allowed()
        if not allowed:
            self._send(_json_bytes({"detail": detail}), "application/json", 423)
            return
        length = min(int(self.headers.get("content-length", "0") or 0), 32_768)
        body = self.rfile.read(length) if length else b"{}"
        if self.path == "/api/action":
            try:
                value = json.loads(body).get("action", [])
                action = np.asarray(value, dtype=np.float32)
                if action.shape != state.action.shape or not np.isfinite(action).all():
                    raise ValueError("invalid action")
                action = np.clip(action, -1.0, 1.0)
                with state.lock:
                    state.action = action
                    state.action_updated = time.monotonic()
                self._send(_json_bytes({"ok": True}), "application/json")
            except (ValueError, TypeError, json.JSONDecodeError):
                self._send(_json_bytes({"error": "Invalid action vector"}), "application/json", 400)
            return
        if self.path == "/api/reset":
            self._send(
                _json_bytes({"detail": "Restart this native room through the Dr.Anmar hub"}),
                "application/json",
                HTTPStatus.CONFLICT,
            )
            return
        self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in ("reconstruction", "observation", "policy"):
            if key in value:
                found = _first_tensor(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _observation_image(env: Any, observation: Any) -> bytes | None:
    unwrapped = env.unwrapped
    tensor = None
    slicer = getattr(unwrapped, "US_slicer", None)
    if slicer is not None:
        tensor = getattr(slicer, "us_img_tensor", None)
    if tensor is None:
        tensor = _first_tensor(observation)
    if tensor is None:
        return None
    array = tensor.detach().float().cpu().numpy()
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    while array.ndim > 2:
        smallest_axis = int(np.argmin(array.shape))
        if array.shape[smallest_axis] <= 8:
            array = np.take(array, array.shape[smallest_axis] // 2, axis=smallest_axis)
        else:
            array = np.max(array, axis=0)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(array, (1.0, 99.5))
    if high <= low:
        high = low + 1.0
    image = np.clip((array - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    image = np.rot90(image)
    pil = Image.fromarray(image, mode="L").resize((800, 600), Image.Resampling.BILINEAR)
    output = io.BytesIO()
    pil.save(output, format="JPEG", quality=88, optimize=False)
    return output.getvalue()


def _configure_upstream_task() -> None:
    module_name = {
        "Isaac-robot-US-guidance-v0": "spinal_surgery.tasks.robot_US_guidance.robotic_US_guidance",
        "Isaac-robot-US-reconstruction-v0": "spinal_surgery.tasks.robot_US_reconstruction.robotic_US_reconstruction",
        "Isaac-robot-US-guided-surgery-v0": "spinal_surgery.tasks.robot_US_guided_surgery.robotic_US_guided_surgery",
    }[args.task]
    module = importlib.import_module(module_name)
    config = module.scene_cfg
    config.setdefault("sim", {})["vis_us"] = False
    config["sim"]["vis_seg_map"] = False
    config["sim"]["vis_rec"] = False
    config["if_record_traj"] = False


def request_stop(_signum: int, _frame: Any) -> None:
    print(f"Dr.Anmar SonoGym bridge received signal {_signum}", flush=True)
    with state.lock:
        state.stop_requested = True


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)


def _native_task_feedback(env: Any) -> dict[str, Any]:
    """Expose native task state for human feedback without changing task behavior."""
    unwrapped = env.unwrapped
    result: dict[str, Any] = {}
    distance = getattr(unwrapped, "distance_to_goal", None)
    if isinstance(distance, torch.Tensor) and distance.numel():
        result["distance_to_goal"] = float(distance.detach().reshape(-1)[0].item())
    current = getattr(unwrapped, "cur_cmd_pose", None)
    goal = getattr(unwrapped, "goal_cmd_pose", None)
    if isinstance(current, torch.Tensor) and current.numel():
        result["current_pose"] = current.detach().reshape(current.shape[0], -1)[0].float().cpu().tolist()
    if isinstance(goal, torch.Tensor) and goal.numel():
        result["goal_pose"] = goal.detach().reshape(goal.shape[0], -1)[0].float().cpu().tolist()
    scalar_fields = {
        "cov_ratio": "coverage_ratio",
        "total_length": "path_length",
        "tip_to_traj_dist": "tip_to_trajectory_m",
        "tip_pos_along_traj": "insertion_m",
        "angle": "trajectory_angle_deg",
    }
    for source_name, result_name in scalar_fields.items():
        value = getattr(unwrapped, source_name, None)
        if isinstance(value, torch.Tensor) and value.numel():
            result[result_name] = float(value.detach().reshape(-1)[0].item())
    if hasattr(unwrapped, "surface_reconstructor"):
        incremental = getattr(unwrapped.surface_reconstructor, "incremental_cov", None)
        if isinstance(incremental, torch.Tensor) and incremental.numel():
            result["incremental_coverage"] = float(incremental.detach().reshape(-1)[0].item())
    cost = getattr(unwrapped, "extras", {}).get("cost") if isinstance(getattr(unwrapped, "extras", None), dict) else None
    if isinstance(cost, torch.Tensor) and cost.numel():
        result["safe"] = bool(cost.detach().reshape(-1)[0].item() < 0.5)
    elif "tip_to_trajectory_m" in result:
        result["safe"] = True
    return result


def main() -> None:
    source_root = args.sonogym_root / "source/spinal_surgery"
    sys.path.insert(0, str(source_root))
    os.chdir(args.sonogym_root)
    # SonoGym's current package initializer publishes these paths after it
    # imports task modules, while the task modules read them during import.
    # Seed the same upstream constants before importing the registry.  This is
    # an import-order adapter only; no task or simulator state is replaced.
    package_root = source_root / "spinal_surgery"
    spinal_surgery = types.ModuleType("spinal_surgery")
    spinal_surgery.__file__ = str(package_root / "__init__.py")
    spinal_surgery.__path__ = [str(package_root)]
    spinal_surgery.ASSETS_EXT_DIR = str(package_root / "assets")
    spinal_surgery.ASSETS_DATA_DIR = str(package_root / "assets/data")
    spinal_surgery.PACKAGE_DIR = str(package_root)
    spinal_surgery.PROJECT_DIR = str(args.sonogym_root)
    sys.modules["spinal_surgery"] = spinal_surgery
    importlib.import_module("spinal_surgery.tasks")

    _configure_upstream_task()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    env_cfg.episode_length_s = 3600.0
    env = gym.make(args.task, cfg=env_cfg)
    observation, _info = env.reset()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True, name="sonogym-web")
    server_thread.start()
    last_fps_time = time.monotonic()
    last_fps_step = 0
    last_frame_time = 0.0
    try:
        while simulation_app.is_running():
            with state.lock:
                if state.stop_requested:
                    break
                action = state.action.copy()
                stale = time.monotonic() - state.action_updated > 0.25
            if stale:
                action.fill(0.0)
            actions = torch.as_tensor(action, device=env.unwrapped.device).reshape(1, -1)
            observation, reward, _terminated, _truncated, _info = env.step(actions)
            native_feedback = _native_task_feedback(env)
            now = time.monotonic()
            with state.lock:
                state.step += 1
                state.reward = float(torch.as_tensor(reward).float().mean().item())
                previous_distance = state.native_feedback.get("distance_to_goal")
                current_distance = native_feedback.get("distance_to_goal")
                if current_distance is not None and previous_distance is not None:
                    delta = previous_distance - current_distance
                    native_feedback["trend"] = "closer" if delta > 1.0e-4 else "farther" if delta < -1.0e-4 else "steady"
                state.native_feedback = native_feedback
                state.applied_action = action.copy()
            if now - last_frame_time >= 1.0 / 15.0:
                frame = _observation_image(env, observation)
                if frame is not None:
                    with state.lock:
                        state.frame_jpeg = frame
                        state.frame_id += 1
                last_frame_time = now
            if now - last_fps_time >= 1.0:
                with state.lock:
                    state.fps = (state.step - last_fps_step) / (now - last_fps_time)
                last_fps_step = state.step
                last_fps_time = now
        print(
            f"SonoGym loop exited: app_running={simulation_app.is_running()} stop_requested={state.stop_requested}",
            flush=True,
        )
        if not state.stop_requested:
            raise RuntimeError("Isaac Sim stopped before the native SonoGym room was closed by Dr.Anmar")
    except Exception as exc:
        with state.lock:
            state.last_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        server.shutdown()
        server.server_close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            simulation_app.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
