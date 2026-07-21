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
        self.reset_requested = False
        self.stop_requested = False
        self.frame_jpeg: bytes | None = None
        self.frame_id = 0
        self.step = 0
        self.fps = 0.0
        self.reward: float | None = None
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
                "truth_note": "SonoGym owns this native environment, patient data, ultrasound simulation, robot state, rewards and safety constraints. Simulation and research only.",
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
:root{{--bg:#020608;--panel:#071923;--line:#244651;--cyan:#35d0e6;--text:#edf7f8;--muted:#8ca7af}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif;overflow:hidden}}
.shell{{height:100vh;display:grid;grid-template-rows:58px minmax(0,1fr) 86px}}header{{display:flex;align-items:center;justify-content:space-between;padding:0 22px;background:#06141c;border-bottom:1px solid var(--line)}}
header b{{font-size:17px}}header span,.hint{{color:var(--muted)}}.live{{color:#57e5aa}}main{{display:grid;place-items:center;min-height:0;padding:14px;background:radial-gradient(circle at center,#0a2631 0,#020608 68%)}}
.scan{{height:100%;max-height:760px;aspect-ratio:4/3;max-width:100%;border:1px solid var(--line);background:#000;position:relative;display:grid;place-items:center}}
.scan img{{width:100%;height:100%;object-fit:contain;image-rendering:auto}}.badge{{position:absolute;left:14px;top:14px;padding:8px 10px;background:#06141cdd;border:1px solid var(--line);font:12px ui-monospace,monospace}}
footer{{display:grid;grid-template-columns:1fr auto;gap:16px;align-items:center;padding:12px 22px;background:var(--panel);border-top:1px solid var(--line)}}
.keys{{display:flex;gap:7px;flex-wrap:wrap}}button{{min-width:52px;height:42px;border:1px solid var(--line);background:#0a202a;color:var(--text);font-weight:700;cursor:pointer}}button.active{{background:var(--cyan);color:#001117}}button.reset{{min-width:100px}}@media(max-width:800px){{footer{{grid-template-columns:1fr}}}}
</style></head><body tabindex="0"><div class="shell"><header><div><b>{args.title}</b><div class="hint">{meta['anatomy']}</div></div><div class="live">● SonoGym native</div></header>
<main><div class="scan"><img id="frame" alt="Live SonoGym ultrasound"><div class="badge" id="telemetry">Starting native environment…</div></div></main>
<footer><div><div class="keys" id="keys"></div><div class="hint">{controls}</div></div><button class="reset" onclick="resetRoom()">Reset</button></footer></div>
<script>
const operatorId=new URLSearchParams(location.search).get('operator')||`browser-${{Date.now().toString(36)}}-${{Math.random().toString(36).slice(2)}}`,dims={action_dimensions},held=new Set();
const maps=dims===3?{{w:[0,1],s:[0,-1],a:[1,-1],d:[1,1],q:[2,-1],e:[2,1]}}:dims===4?{{w:[0,1],s:[0,-1],a:[1,-1],d:[1,1],q:[2,-1],e:[2,1],r:[3,1],f:[3,-1]}}:{{w:[0,1],s:[0,-1],a:[1,-1],d:[1,1],r:[2,1],f:[2,-1],q:[3,-1],e:[3,1],arrowup:[4,1],arrowdown:[4,-1],arrowleft:[5,-1],arrowright:[5,1]}};
const labels={{arrowup:'↑',arrowdown:'↓',arrowleft:'←',arrowright:'→'}};document.getElementById('keys').innerHTML=Object.keys(maps).map(k=>`<button data-key="${{k}}">${{labels[k]||k.toUpperCase()}}</button>`).join('');
function send(){{const a=Array(dims).fill(0);for(const k of held){{const m=maps[k];if(m)a[m[0]]+=m[1]}}fetch('/api/action',{{method:'POST',headers:{{'content-type':'application/json','x-dr-anmar-operator':operatorId}},body:JSON.stringify({{action:a}})}}).catch(()=>{{}});document.querySelectorAll('[data-key]').forEach(b=>b.classList.toggle('active',held.has(b.dataset.key)))}}
addEventListener('keydown',e=>{{const k=e.key.toLowerCase();if(maps[k]){{e.preventDefault();held.add(k);send()}}}});addEventListener('keyup',e=>{{const k=e.key.toLowerCase();if(maps[k]){{e.preventDefault();held.delete(k);send()}}}});addEventListener('blur',()=>{{held.clear();send()}});
document.querySelectorAll('[data-key]').forEach(b=>{{b.onpointerdown=()=>{{held.add(b.dataset.key);send()}};b.onpointerup=b.onpointerleave=()=>{{held.delete(b.dataset.key);send()}}}});
setInterval(()=>{{if(held.size)send()}},75);
async function resetRoom(){{await fetch('/api/reset',{{method:'POST',headers:{{'x-dr-anmar-operator':operatorId}}}})}}
const img=document.getElementById('frame');function nextFrame(){{img.src='/frame.jpg?t='+Date.now()}}img.onload=()=>setTimeout(nextFrame,70);img.onerror=()=>setTimeout(nextFrame,500);nextFrame();
async function status(){{try{{const s=await fetch('/api/status').then(r=>r.json());document.getElementById('telemetry').textContent=`${{s.fps}} FPS · step ${{s.frame_id}}${{s.reward===null?'':' · reward '+s.reward.toFixed(3)}}`}}catch(e){{}}setTimeout(status,500)}}status();document.body.focus();
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
            with state.lock:
                state.reset_requested = True
            self._send(_json_bytes({"ok": True}), "application/json")
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
    with state.lock:
        state.stop_requested = True


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)


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
                do_reset = state.reset_requested
                state.reset_requested = False
                action = state.action.copy()
                stale = time.monotonic() - state.action_updated > 0.25
            if stale:
                action.fill(0.0)
            if do_reset:
                observation, _info = env.reset()
            actions = torch.as_tensor(action, device=env.unwrapped.device).reshape(1, -1)
            observation, reward, _terminated, _truncated, _info = env.step(actions)
            now = time.monotonic()
            with state.lock:
                state.step += 1
                state.reward = float(torch.as_tensor(reward).float().mean().item())
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
