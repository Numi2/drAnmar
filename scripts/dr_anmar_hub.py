# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stable control hub and doctor-facing learning studio for Dr.Anmar."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dr_anmar_catalog import CATALOG, PRIMARY_TASKS, TASKS_BY_ID
from dr_anmar_curriculum import curriculum_payload


parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=2360)
parser.add_argument("--worker_port", type=int, default=2361)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
args.root = args.root.expanduser().resolve()

DR_ANMAR_ROOT = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()
WEB_ROOT = args.root / "web"
RESEARCH_ROOT = DR_ANMAR_ROOT / "research/sufia-bc/static"
ANATOMY_ROOT = DR_ANMAR_ROOT / "assets/sufia_bc"
ASSET_STATUS_PATH = DR_ANMAR_ROOT / "run/sufia_assets_status.json"
PROGRESS_PATH = DR_ANMAR_ROOT / "state/doctor_progress.json"
TRAINING_ROOT = DR_ANMAR_ROOT / "training"


ANATOMY_SCENES = (
    {
        "archive": "OR_scene_CTLiver-Prostate-Bladder.zip",
        "title": "CT liver, prostate & bladder",
        "kind": "Photorealistic organ scene",
        "description": "A detailed multi-organ operating-room scene for anatomy-aware visual policy experiments.",
        "organs": ["liver", "prostate", "bladder"],
        "preview": "/research/videos/tissue.mp4",
    },
    {
        "archive": "OR_scene_MAISI_imagesTr_liver_27_relabel_resample1_syn_seed6_postprocess.zip",
        "title": "MAISI synthetic liver 27",
        "kind": "Synthetic anatomy variation",
        "description": "A generated liver scene for increasing anatomical variation in synthetic datasets.",
        "organs": ["liver"],
        "preview": "/research/videos/Tissue_view_train.mp4",
    },
    {
        "archive": "OR_scene_MAISI_s0253_ct_relabel_resample1_syn_seed6_postprocess.zip",
        "title": "MAISI anatomy s0253",
        "kind": "Synthetic CT-derived scene",
        "description": "One of five anatomy variants used to prevent a visual policy from memorizing one patient geometry.",
        "organs": ["multi-organ anatomy"],
        "preview": "/research/videos/Tissue_view_1.mp4",
    },
    {
        "archive": "OR_scene_MAISI_s0702_ct_relabel_resample2_syn_seed6_postprocess.zip",
        "title": "MAISI anatomy s0702",
        "kind": "Synthetic CT-derived scene",
        "description": "A second synthetic anatomy configuration for viewpoint and geometry robustness studies.",
        "organs": ["multi-organ anatomy"],
        "preview": "/research/videos/Tissue_view_2.mp4",
    },
    {
        "archive": "OR_scene_MAISI_s0994_ct_relabel_resample2_syn_seed6_postprocess.zip",
        "title": "MAISI anatomy s0994",
        "kind": "Synthetic CT-derived scene",
        "description": "A distinct patient-anatomy variation packaged as an OpenUSD operating-room scene.",
        "organs": ["multi-organ anatomy"],
        "preview": "/research/videos/tissue.mp4",
    },
    {
        "archive": "OR_scene_MAISI_s1269_ct_relabel_resample1_syn_seed6_postprocess.zip",
        "title": "MAISI anatomy s1269",
        "kind": "Synthetic CT-derived scene",
        "description": "Synthetic anatomy for testing whether a learned visual behavior transfers across geometries.",
        "organs": ["multi-organ anatomy"],
        "preview": "/research/videos/Tissue_view_train.mp4",
    },
    {
        "archive": "OR_scene_s1371.zip",
        "title": "Surgical anatomy s1371",
        "kind": "OpenUSD surgical scene",
        "description": "A complete official surgical scene ready for Isaac Sim inspection and dataset authoring.",
        "organs": ["multi-organ anatomy"],
        "preview": "/research/videos/Tissue_view_1.mp4",
    },
)


class LaunchRequest(BaseModel):
    task: str


class AnatomyLaunchRequest(BaseModel):
    room_id: str


class ProgressRequest(BaseModel):
    lesson_id: str
    completed: bool = True


class TrainingRequest(BaseModel):
    backend: str = "rsl_rl"
    task: str
    num_envs: int = 32
    max_iterations: int = 50
    resume_workstation: bool = True


@dataclass
class HubState:
    switching: bool = False
    requested_task: str | None = None
    error: str | None = None
    training_process: subprocess.Popen | None = None
    training_status: str = "idle"
    training_task: str | None = None
    training_backend: str | None = None
    training_log: str | None = None
    training_started_at: str | None = None
    training_exit_code: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


state = HubState()
app = FastAPI(title="Dr.Anmar Doctor Studio", docs_url=None, redoc_url=None)
if RESEARCH_ROOT.is_dir():
    app.mount("/research", StaticFiles(directory=RESEARCH_ROOT), name="sufia-research")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def worker_json(path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload or {}).encode() if method != "GET" else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.worker_port}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", str(exc))
        except (ValueError, AttributeError):
            detail = str(exc)
        raise HTTPException(exc.code, detail) from exc
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise HTTPException(503, "The operating-room worker is not ready") from exc


def worker_status() -> dict[str, Any] | None:
    try:
        return worker_json("/api/status")
    except HTTPException:
        return None


def switch_worker(task: str) -> None:
    try:
        command = [str(args.root / "dr_anmar_workstation.sh"), "restart", str(args.worker_port), task]
        result = subprocess.run(command, cwd=args.root, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            current = worker_status()
            if current and current.get("task") == task and current.get("frame_id", 0) > 0:
                with state.lock:
                    state.error = None
                    state.switching = False
                return
            time.sleep(1.0)
        raise TimeoutError(f"Timed out starting {task}")
    except Exception as exc:
        with state.lock:
            state.error = str(exc)
            state.switching = False


def switch_anatomy_worker(room_id: str, room_title: str, scene: Path) -> None:
    try:
        command = [
            str(args.root / "dr_anmar_workstation.sh"),
            "restart-anatomy",
            str(args.worker_port),
            room_id,
            str(scene),
            room_title,
        ]
        result = subprocess.run(command, cwd=args.root, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline:
            current = worker_status()
            if current and current.get("mode") == "anatomy" and current.get("room_id") == room_id and current.get("frame_id", 0) > 0:
                with state.lock:
                    state.error = None
                    state.switching = False
                return
            time.sleep(1.0)
        raise TimeoutError(f"Timed out starting anatomy room {room_title}")
    except Exception as exc:
        with state.lock:
            state.error = str(exc)
            state.switching = False


@lru_cache(maxsize=len(ANATOMY_SCENES))
def installed_usd_inventory(directory: str) -> tuple[str, ...]:
    root = Path(directory)
    return tuple(sorted(str(path.relative_to(ANATOMY_ROOT)) for path in root.rglob("*.usd")))


def anatomy_payload() -> dict[str, Any]:
    installer = read_json(ASSET_STATUS_PATH, {})
    scenes: list[dict[str, Any]] = []
    for metadata in ANATOMY_SCENES:
        archive = metadata["archive"]
        directory = ANATOMY_ROOT / archive.removesuffix(".zip")
        installed = (directory / ".installed").is_file()
        usd_files = installed_usd_inventory(str(directory)) if installed else ()
        if installed:
            scene_state = "ready"
        elif installer.get("current") == archive:
            scene_state = installer.get("phase", "downloading")
        else:
            scene_state = "queued"
        item = dict(metadata)
        primary_usd = next((path for path in usd_files if Path(path).name == "main_scene.usd"), None)
        item.update(
            {
                "id": archive.removesuffix(".zip"),
                "state": scene_state,
                "openusd_ready": bool(usd_files),
                "usd_count": len(usd_files),
                "primary_usd": primary_usd,
            }
        )
        scenes.append(item)
    total = int(installer.get("total_bytes") or 0)
    downloaded = min(int(installer.get("downloaded_bytes") or 0), total) if total else 0
    return {
        "installer": installer,
        "scenes": scenes,
        "ready_count": sum(scene["openusd_ready"] for scene in scenes),
        "scene_count": len(scenes),
        "download_percent": round(downloaded * 100 / total, 1) if total else 0.0,
        "install_root": str(ANATOMY_ROOT),
        "source": "Official ORBIT-Surgical v0.1.0 release assets",
        "runtime_note": "OpenUSD scenes are installed for Isaac Sim inspection and synthetic-data authoring; they are not clinical patient models.",
    }


def progress_payload() -> dict[str, Any]:
    progress = read_json(PROGRESS_PATH, {"completed": {}, "active_lesson": "needle-lift"})
    lesson_count = curriculum_payload()["lesson_count"]
    completed = progress.get("completed", {})
    progress["completed_count"] = sum(bool(value) for value in completed.values())
    progress["lesson_count"] = lesson_count
    progress["percent"] = round(progress["completed_count"] * 100 / lesson_count) if lesson_count else 0
    return progress


def training_payload() -> dict[str, Any]:
    with state.lock:
        process = state.training_process
        if process is not None and process.poll() is not None and state.training_status == "running":
            state.training_exit_code = process.returncode
            state.training_status = "complete" if process.returncode == 0 else "failed"
        return {
            "status": state.training_status,
            "task": state.training_task,
            "backend": state.training_backend,
            "log": state.training_log,
            "started_at": state.training_started_at,
            "exit_code": state.training_exit_code,
        }


def monitor_training(process: subprocess.Popen, log_file, resume_task: str | None) -> None:
    code = process.wait()
    log_file.close()
    with state.lock:
        state.training_exit_code = code
        state.training_status = "complete" if code == 0 else "failed"
    if resume_task:
        with state.lock:
            state.switching = True
            state.requested_task = resume_task
        switch_worker(resume_task)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = WEB_ROOT / "doctor_studio.html"
    if not path.is_file():
        raise HTTPException(503, "Doctor Studio interface is not installed")
    return HTMLResponse(path.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    current = worker_status()
    return {"catalog": CATALOG, "primary": PRIMARY_TASKS, "current_task": current.get("task") if current else None}


@app.get("/api/curriculum")
def curriculum() -> dict[str, Any]:
    return curriculum_payload()


@app.get("/api/progress")
def get_progress() -> dict[str, Any]:
    return progress_payload()


@app.post("/api/progress")
def update_progress(request: ProgressRequest) -> dict[str, Any]:
    valid_lessons = {lesson["id"] for course in curriculum_payload()["courses"] for lesson in course["lessons"]}
    if request.lesson_id not in valid_lessons:
        raise HTTPException(404, "Unknown lesson")
    progress = read_json(PROGRESS_PATH, {"completed": {}})
    progress.setdefault("completed", {})[request.lesson_id] = request.completed
    progress["active_lesson"] = request.lesson_id
    progress["updated_at"] = utc_now()
    write_json(PROGRESS_PATH, progress)
    return progress_payload()


@app.get("/api/anatomy")
def anatomy() -> dict[str, Any]:
    return anatomy_payload()


@app.post("/api/anatomy/launch")
def launch_anatomy(request: AnatomyLaunchRequest) -> dict[str, Any]:
    room = next((item for item in anatomy_payload()["scenes"] if item["id"] == request.room_id), None)
    if room is None:
        raise HTTPException(404, "Unknown anatomy operating-room preset")
    if not room["openusd_ready"] or not room["primary_usd"]:
        raise HTTPException(409, "This anatomy room has not finished installing")
    scene = (ANATOMY_ROOT / room["primary_usd"]).resolve()
    if ANATOMY_ROOT.resolve() not in scene.parents or scene.name != "main_scene.usd" or not scene.is_file():
        raise HTTPException(409, "The installed anatomy room path is invalid")
    training = training_payload()
    if training["status"] in {"running", "stopping"}:
        raise HTTPException(409, "Stop the training lab before loading an anatomy room")
    current = worker_status()
    if current and current.get("mode") == "anatomy" and current.get("room_id") == request.room_id and current.get("frame_id", 0) > 0:
        return {"ok": True, "room_id": request.room_id, "already_ready": True}
    with state.lock:
        if state.switching:
            raise HTTPException(409, f"Already loading {state.requested_task}")
        state.switching = True
        state.requested_task = room["title"]
        state.error = None
    threading.Thread(
        target=switch_anatomy_worker,
        args=(request.room_id, room["title"], scene),
        daemon=True,
        name="dr-anmar-anatomy-switch",
    ).start()
    return {"ok": True, "room_id": request.room_id, "title": room["title"]}


@app.get("/api/demos")
def demonstrations() -> dict[str, Any]:
    return worker_json("/api/demos")


@app.post("/api/worker/{command}")
def worker_command(command: str) -> dict[str, Any]:
    paths = {
        "reset": "/api/reset",
        "record-start": "/api/record/start",
        "record-stop": "/api/record/stop",
        "replay-last": "/api/replay-last",
        "stop": "/api/stop",
    }
    path = paths.get(command)
    if path is None:
        raise HTTPException(404, "Unknown workstation command")
    return worker_json(path, method="POST", payload={})


@app.get("/api/training")
def training_status() -> dict[str, Any]:
    return training_payload()


@app.post("/api/training/start")
def start_training(request: TrainingRequest) -> dict[str, Any]:
    if request.backend not in {"rsl_rl", "rl_games", "sb3", "skrl"}:
        raise HTTPException(400, "Choose RSL-RL, RL-Games, SB3, or SKRL")
    item = TASKS_BY_ID.get(request.task)
    if item is None or item["play"] or item["variant"] != "joint":
        raise HTTPException(400, "Training requires a registered non-play joint-position task")
    if request.backend in {"sb3", "skrl"} and item["procedure"] not in {"lift", "handover"}:
        raise HTTPException(409, f"{request.backend} is configured here for lift and handover tasks")
    if request.num_envs not in range(8, 129):
        raise HTTPException(400, "Starter labs support 8 to 128 parallel environments")
    if request.max_iterations not in range(1, 201):
        raise HTTPException(400, "Starter labs support 1 to 200 iterations")
    with state.lock:
        if state.training_process is not None and state.training_process.poll() is None:
            raise HTTPException(409, "A training lab is already running")
    current = worker_status()
    resume_task = current.get("task") if current and current.get("mode") != "anatomy" and request.resume_workstation else None
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = TRAINING_ROOT / f"{stamp}_{request.backend}_{item['slug']}.log"
    command = [
        str(args.root / "dr_anmar_train.sh"),
        request.backend,
        request.task,
        "--num_envs",
        str(request.num_envs),
        "--max_iterations",
        str(request.max_iterations),
    ]
    log_file = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            command,
            cwd=args.root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise
    with state.lock:
        state.training_process = process
        state.training_status = "running"
        state.training_task = request.task
        state.training_backend = request.backend
        state.training_log = str(log_path)
        state.training_started_at = utc_now()
        state.training_exit_code = None
    threading.Thread(
        target=monitor_training,
        args=(process, log_file, resume_task),
        daemon=True,
        name="dr-anmar-training",
    ).start()
    return {"ok": True, **training_payload(), "command": command}


@app.post("/api/training/stop")
def stop_training() -> dict[str, Any]:
    with state.lock:
        process = state.training_process
    if process is None or process.poll() is not None:
        raise HTTPException(409, "No training lab is running")
    os.killpg(process.pid, signal.SIGTERM)
    with state.lock:
        state.training_status = "stopping"
    return {"ok": True, **training_payload()}


@app.get("/api/status")
def status() -> JSONResponse:
    with state.lock:
        hub = {"switching": state.switching, "requested_task": state.requested_task, "error": state.error}
    hub.update(
        {
            "worker": worker_status(),
            "catalog_tasks": len(CATALOG),
            "interactive_tasks": len(PRIMARY_TASKS),
            "anatomy": anatomy_payload(),
            "training": training_payload(),
        }
    )
    return JSONResponse(hub)


@app.post("/api/launch")
def launch(request: LaunchRequest) -> dict[str, Any]:
    item = TASKS_BY_ID.get(request.task)
    if item is None:
        raise HTTPException(404, "Unknown ORBIT-Surgical task")
    if not item["browser_control"] or item["play"]:
        raise HTTPException(409, "Use the relative-IK non-play variant for the interactive workstation")
    training = training_payload()
    if training["status"] in {"running", "stopping"}:
        raise HTTPException(409, "Stop the training lab before loading an operating room")
    current = worker_status()
    if current and current.get("task") == request.task and current.get("frame_id", 0) > 0:
        return {"ok": True, "task": request.task, "already_ready": True}
    with state.lock:
        if state.switching:
            raise HTTPException(409, f"Already loading {state.requested_task}")
        state.switching = True
        state.requested_task = request.task
        state.error = None
    threading.Thread(target=switch_worker, args=(request.task,), daemon=True, name="dr-anmar-switch").start()
    return {"ok": True, "task": request.task}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)
