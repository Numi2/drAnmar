# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stable control hub and doctor-facing learning studio for Dr.Anmar."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from dr_anmar_catalog import CATALOG, PRIMARY_TASKS, TASKS_BY_ID
from dr_anmar_bench_systems import (
    related_asset_paths as related_bench_asset_paths,
    resolve_featured_robot_system,
)
from dr_anmar_curriculum import curriculum_payload
from dr_anmar_i4h_adapter import (
    I4H_ROOT,
    I4H_RELEASE,
    I4H_ASSET_DOWNLOAD_DIR,
    I4H_ASSET_HASH,
    HOLOHUB_CLI_COMMIT,
    MODALITY_CATALOG,
    POLICY_STARTING_POINTS,
    WORKFLOW_BINDINGS,
    platform_payload,
    runtime_prerequisites,
    study_manifest,
    workflow_modes,
)
from dr_anmar_procedures import PROCEDURES_BY_ID, PROCEDURE_ROOMS, procedure_payload
from dr_anmar_native_rooms import resolve_native_room
from dr_anmar_sonogym_adapter import (
    SONOGYM_COMMIT,
    SONOGYM_ROOT,
    launch_command as sonogym_launch_command,
    platform_workflow as sonogym_platform_workflow,
    runtime_prerequisites as sonogym_runtime_prerequisites,
    workflow_modes as sonogym_workflow_modes,
)
from dr_anmar_operator import (
    ACCESS_COOKIE,
    OPERATOR_HEADER,
    OperatorLease,
    access_cookie_value,
    access_is_authorized,
    configured_access_token,
)
from dr_anmar_psm_gripper import CANONICAL_PSM_GRIPPER_PROFILE


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
OPENUSD_ROOT = DR_ANMAR_ROOT / "scenes/openusd"
ASSET_STATUS_PATH = DR_ANMAR_ROOT / "run/sufia_assets_status.json"
PROGRESS_PATH = DR_ANMAR_ROOT / "state/doctor_progress.json"
TRAINING_ROOT = DR_ANMAR_ROOT / "training"
EXPERIMENT_ROOT = DR_ANMAR_ROOT / "experiments"
DEMO_ROOT = DR_ANMAR_ROOT / "demos"
DATASET_CARD_ROOT = DR_ANMAR_ROOT / "dataset_cards"
POLICY_CARD_ROOT = DR_ANMAR_ROOT / "policy_evaluation_cards"
STUDY_ROOT = DR_ANMAR_ROOT / "studies"
HEALTHCARE_JOB_ROOT = DR_ANMAR_ROOT / "healthcare_jobs"
WORKSTATION_LOG_PATH = DR_ANMAR_ROOT / "logs/workstation.log"
WORKER_FATAL_MARKERS = (
    "Traceback (most recent call last):",
    "Out of GPU memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "ModuleNotFoundError:",
)


def bench_asset_selection(
    procedure: dict[str, Any], requested: list[str] | tuple[str, ...] | None
) -> tuple[str, ...] | None:
    """Resolve one ordered, allow-listed operating-room bench composition."""

    catalog = tuple(procedure.get("bench_asset_catalog", ()))
    if not catalog:
        if requested:
            raise HTTPException(400, "This room does not support configurable bench assets")
        return None
    allowed = {str(item["id"]) for item in catalog}
    selected = (
        {str(item["id"]) for item in catalog if item.get("default")}
        if requested is None
        else set(requested)
    )
    unknown = sorted(selected - allowed)
    if unknown:
        raise HTTPException(400, "Unknown operating-room bench assets: " + ", ".join(unknown))
    try:
        resolve_featured_robot_system(selected)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return tuple(str(item["id"]) for item in catalog if str(item["id"]) in selected)


def psm_gripper_selection(
    procedure: dict[str, Any],
    requested_open_rad: float | None,
    requested_close_rad: float | None,
) -> tuple[float, float] | None:
    """Resolve explicit numeric jaw targets for the native NVIDIA bench."""

    if not procedure.get("nvidia_native_bench"):
        if requested_open_rad is not None or requested_close_rad is not None:
            raise HTTPException(400, "This room does not expose PSM jaw target settings")
        return None
    open_rad = (
        CANONICAL_PSM_GRIPPER_PROFILE.open_rad
        if requested_open_rad is None
        else float(requested_open_rad)
    )
    close_rad = (
        CANONICAL_PSM_GRIPPER_PROFILE.close_rad
        if requested_close_rad is None
        else float(requested_close_rad)
    )
    if not math.isfinite(open_rad) or not 0.10 <= open_rad <= 0.60:
        raise HTTPException(400, "Open target must be between 0.10 and 0.60 radians")
    if not math.isfinite(close_rad) or not 0.00 <= close_rad <= 0.15:
        raise HTTPException(400, "Closed target must be between 0.00 and 0.15 radians")
    if close_rad >= open_rad:
        raise HTTPException(400, "Closed target must be smaller than the open target")
    return open_rad, close_rad


def missing_required_bench_assets(
    procedure: dict[str, Any], bench_assets: tuple[str, ...] | None = None
) -> list[str]:
    """Return missing NVIDIA or DrAnmar paths from a room's asset contract."""

    content_root = I4H_ASSET_DOWNLOAD_DIR / I4H_ASSET_HASH
    required: list[tuple[Path, str]] = [
        (content_root, str(path))
        for path in procedure.get("required_nvidia_assets", ())
    ]
    required.extend(
        (args.root, str(path))
        for path in procedure.get("required_repository_assets", ())
    )
    selected = set(bench_assets or ())
    provider_roots = {
        "nvidia_i4h": content_root,
        "dr_anmar": args.root / "source/extensions/orbit.surgical.assets/data",
        "dr_anmar_repository": args.root,
    }
    for item in procedure.get("bench_asset_catalog", ()):
        if str(item["id"]) not in selected:
            continue
        provider = str(item.get("provider", "nvidia_i4h"))
        root = provider_roots.get(provider)
        if root is None:
            raise RuntimeError(f"Unknown operating-room asset provider: {provider}")
        required.append((root, str(item["path"])))
        required.extend(
            (root, relative_path)
            for relative_path in related_bench_asset_paths(item)
        )
    return [
        str(relative_path)
        for root, relative_path in required
        if not (root / relative_path).is_file()
    ]


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


class ProcedureLaunchRequest(BaseModel):
    procedure_id: str
    anatomy_scene: str | None = None
    bench_assets: list[str] | None = None
    gripper_open_rad: float | None = None
    gripper_close_rad: float | None = None


class ProgressRequest(BaseModel):
    lesson_id: str
    completed: bool = True


class TrainingRequest(BaseModel):
    backend: str = "rsl_rl"
    task: str
    num_envs: int = 32
    max_iterations: int = 50
    resume_workstation: bool = True


class ScenarioApplicationRequest(BaseModel):
    scenario_id: str
    seed: int = 7777


class AutonomyModeRequest(BaseModel):
    mode: str


class ChallengeMatrixRequest(BaseModel):
    demo: str
    scenario_ids: list[str]
    seeds: list[int]


class ReferenceGhostRequest(BaseModel):
    enabled: bool = True
    demo: str | None = None


class DatasetCardRequest(BaseModel):
    title: str = "Dr.Anmar surgical behavior dataset"
    demos: list[str]
    intended_use: str = "Simulation-only behavior cloning and supervised-autonomy research"


class PolicyEvaluationCardRequest(BaseModel):
    title: str
    dataset_id: str
    training_experiment_id: str
    challenge_matrix_id: str
    checkpoint_path: str


class MultimodalStudyRequest(BaseModel):
    title: str
    clinical_question: str
    task: str
    modalities: list[str]
    policy: str = "behavior_cloning"
    teleoperation: str = "keyboard_pointer"


class HealthcareWorkflowRequest(BaseModel):
    workflow: str
    mode: str
    resume_workstation: bool = True


class AccessSessionRequest(BaseModel):
    token: str


@dataclass
class HubState:
    shutting_down: bool = False
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
    training_manifest: str | None = None
    healthcare_process: subprocess.Popen | None = None
    healthcare_status: str = "idle"
    healthcare_job_id: str | None = None
    healthcare_workflow: str | None = None
    healthcare_mode: str | None = None
    healthcare_log: str | None = None
    healthcare_started_at: str | None = None
    healthcare_exit_code: int | None = None
    healthcare_manifest: str | None = None
    healthcare_resume_task: str | None = None
    healthcare_resume_context: dict[str, Any] | None = None
    healthcare_skip_resume_job_id: str | None = None
    matrix_status: str = "idle"
    matrix_id: str | None = None
    matrix_demo: str | None = None
    matrix_total: int = 0
    matrix_completed: int = 0
    matrix_results: list[dict[str, Any]] = field(default_factory=list)
    matrix_aggregate: dict[str, Any] = field(default_factory=dict)
    matrix_manifest: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


state = HubState()
operator_lease = OperatorLease()
access_attempts: dict[str, list[float]] = {}
access_attempts_lock = threading.Lock()
app = FastAPI(title="Dr.Anmar Doctor Studio", docs_url=None, redoc_url=None)
if RESEARCH_ROOT.is_dir():
    app.mount("/research", StaticFiles(directory=RESEARCH_ROOT), name="sufia-research")


@app.middleware("http")
async def protect_browser_requests(request: Request, call_next):
    """Reject cross-site mutations while preserving local API and tailnet use."""
    if request.url.path != "/api/session" and not access_is_authorized(request.cookies.get(ACCESS_COOKIE)):
        if request.method == "GET" and request.url.path == "/":
            return HTMLResponse(LOGIN_HTML)
        return JSONResponse({"detail": "Dr.Anmar access token required"}, status_code=401)
    origin = request.headers.get("origin")
    if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
        try:
            from urllib.parse import urlparse

            if urlparse(origin).hostname != request.url.hostname:
                return JSONResponse({"detail": "Cross-site state changes are not allowed"}, status_code=403)
        except ValueError:
            return JSONResponse({"detail": "Invalid request origin"}, status_code=403)
        if request.url.path != "/api/session":
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


LOGIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dr.Anmar access</title><style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#061119;color:#e9f8fa;font:16px system-ui}.card{width:min(420px,90vw);padding:32px;border:1px solid #28505e;background:#0a1a23}h1{margin-top:0}p{color:#92adb7}input,button{width:100%;height:48px;box-sizing:border-box;margin-top:12px;border:1px solid #3b6878;background:#0c222c;color:#fff;padding:0 12px}button{background:#2cd2e8;color:#031014;font-weight:800;cursor:pointer}#error{color:#ff9ca4}</style></head><body><form class="card" onsubmit="login(event)"><h1>Dr.Anmar</h1><p>This research workstation requires its operator access token.</p><input id="token" type="password" autocomplete="current-password" autofocus aria-label="Access token"><button>Open Doctor Studio</button><p id="error"></p></form><script>async function login(event){event.preventDefault();const response=await fetch('/api/session',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({token:document.getElementById('token').value})});if(response.ok)location.reload();else document.getElementById('error').textContent='The access token was not accepted.'}</script></body></html>"""


@app.post("/api/session")
def create_access_session(request: AccessSessionRequest, http_request: Request) -> JSONResponse:
    if configured_access_token() is None:
        return JSONResponse({"ok": True, "access_control": "disabled"})
    if not request.token or len(request.token) > 4096:
        raise HTTPException(400, "Invalid access-token format")
    client = http_request.client.host if http_request.client else "unknown"
    now = time.monotonic()
    with access_attempts_lock:
        for stale_client, attempts in list(access_attempts.items()):
            active = [attempt for attempt in attempts if now - attempt < 60.0]
            if active:
                access_attempts[stale_client] = active
            else:
                access_attempts.pop(stale_client, None)
        recent = [attempt for attempt in access_attempts.get(client, []) if now - attempt < 60.0]
        if len(recent) >= 5:
            raise HTTPException(429, "Too many access attempts; try again in one minute")
        if recent:
            access_attempts[client] = recent
    if not access_is_authorized(None, request.token):
        with access_attempts_lock:
            access_attempts.setdefault(client, []).append(now)
        raise HTTPException(401, "Invalid Dr.Anmar access token")
    with access_attempts_lock:
        access_attempts.pop(client, None)
    response = JSONResponse({"ok": True, "access_control": "enabled"})
    response.set_cookie(
        ACCESS_COOKIE,
        access_cookie_value(request.token),
        max_age=43_200,
        httponly=True,
        secure=os.environ.get("DR_ANMAR_COOKIE_SECURE", "0") == "1",
        samesite="strict",
        path="/",
    )
    return response


@app.post("/api/operator/heartbeat")
def operator_heartbeat(request: Request) -> dict[str, Any]:
    return {"ok": True, "operator_lease": operator_lease.status()}


@app.post("/api/operator/release")
def operator_release(request: Request) -> dict[str, Any]:
    return {"ok": operator_lease.release(request.headers.get(OPERATOR_HEADER))}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


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


def process_command(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        return ""


def reconcile_orphaned_jobs() -> None:
    """Stop jobs left by a previous hub and make their manifests truthful."""
    if not EXPERIMENT_ROOT.is_dir():
        return
    for path in EXPERIMENT_ROOT.glob("*.json"):
        manifest = read_json(path, {})
        if manifest.get("status") not in {"preparing", "starting", "running", "stopping"}:
            continue
        if manifest.get("kind") not in {
            "policy_training",
            "isaac_for_healthcare_workflow",
            "sonogym_orthopedic_workflow",
            "challenge_matrix",
        }:
            continue
        pid = manifest.get("pid")
        command = process_command(pid) if isinstance(pid, int) and pid > 1 else ""
        expected = {
            "policy_training": "dr_anmar_train",
            "isaac_for_healthcare_workflow": "i4h",
            "sonogym_orthopedic_workflow": "dr_anmar_sonogym_worker",
        }.get(manifest.get("kind"))
        if expected and command and expected in command:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        manifest.update(
            {
                "status": "interrupted",
                "interrupted_at": utc_now(),
                "interruption_reason": "hub_restart_reconciliation",
            }
        )
        write_json(path, manifest)


def terminate_managed_jobs() -> None:
    with state.lock:
        state.shutting_down = True
        processes = (state.training_process, state.healthcare_process)
        state.healthcare_resume_context = None
    for process in processes:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass


@app.on_event("startup")
def startup_reconciliation() -> None:
    reconcile_orphaned_jobs()


@app.on_event("shutdown")
def shutdown_managed_jobs() -> None:
    terminate_managed_jobs()


def repository_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def source_revision() -> str | None:
    return repository_revision(args.root)


def worker_json(path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload or {}).encode() if method != "GET" else None
    headers = {"Content-Type": "application/json"}
    if (access_token := configured_access_token()) is not None:
        headers["Cookie"] = f"{ACCESS_COOKIE}={access_cookie_value(access_token)}"
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.worker_port}{path}",
        data=body,
        method=method,
        headers=headers,
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


def proxy_worker_request(
    path: str,
    method: str,
    query: str,
    body: bytes,
    incoming_headers: dict[str, str],
) -> tuple[int, bytes, dict[str, str]]:
    """Forward one browser request to the private workstation process."""
    target = f"http://127.0.0.1:{args.worker_port}/{path}"
    if query:
        target = f"{target}?{query}"
    headers: dict[str, str] = {}
    for name in ("content-type", "accept", "range", OPERATOR_HEADER):
        if value := incoming_headers.get(name):
            headers[name] = value
    if (access_token := configured_access_token()) is not None:
        headers["Cookie"] = f"{ACCESS_COOKIE}={access_cookie_value(access_token)}"
    request = urllib.request.Request(
        target,
        data=body if method not in {"GET", "HEAD"} else None,
        method=method,
        headers=headers,
    )
    try:
        response = urllib.request.urlopen(request, timeout=10.0)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        payload = response.read()
        status = response.status
        response_headers = {
            name: value
            for name, value in response.headers.items()
            if name.lower() in {"content-type", "content-disposition", "content-range", "accept-ranges"}
        }
    content_type = response_headers.get("Content-Type", response_headers.get("content-type", ""))
    if path == "" and "text/html" in content_type:
        # The worker uses absolute browser paths because it can also run alone.
        # Keep that implementation intact while mounting it behind the one
        # public Doctor Studio address.
        html = payload.decode("utf-8")
        for prefix in ("api", "frame", "demos"):
            html = html.replace(f"'/{prefix}/", f"'/workstation/{prefix}/")
            html = html.replace(f'"/{prefix}/', f'"/workstation/{prefix}/')
            html = html.replace(f"`/{prefix}/", f"`/workstation/{prefix}/")
        payload = html.encode("utf-8")
    return status, payload, response_headers


@app.api_route(
    "/workstation/{path:path}",
    methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def workstation_proxy(path: str, request: Request) -> Response:
    """Expose the one private simulation worker through the public hub."""
    body = await request.body()
    try:
        status, payload, headers = await run_in_threadpool(
            proxy_worker_request,
            path,
            request.method,
            request.url.query,
            body,
            {name.lower(): value for name, value in request.headers.items()},
        )
    except (OSError, urllib.error.URLError) as exc:
        raise HTTPException(503, "The operating-room worker is not ready") from exc
    return Response(content=payload, status_code=status, headers=headers)


def capture_worker_context(current: dict[str, Any] | None, enabled: bool = True) -> dict[str, Any] | None:
    """Capture enough state to restore the exact procedure and anatomy room."""
    if not enabled or not current or not current.get("task"):
        return None
    context: dict[str, Any] = {
        "task": current["task"],
        "procedure_id": current.get("procedure", {}).get("id", ""),
        "anatomy_scene_id": current.get("anatomy_scene_id", ""),
        "anatomy_title": current.get("anatomy_showcase", ""),
    }
    active_bench_assets = current.get("procedure", {}).get("active_bench_assets")
    if isinstance(active_bench_assets, list):
        context["bench_assets"] = tuple(str(item) for item in active_bench_assets)
    gripper_profile = current.get("gripper_profile")
    if isinstance(gripper_profile, dict):
        context["gripper_open_rad"] = float(gripper_profile["open_rad"])
        context["gripper_close_rad"] = float(gripper_profile["close_rad"])
    anatomy_scene = current.get("anatomy_asset")
    environment = current.get("openusd_environment")
    if anatomy_scene:
        context["anatomy_scene"] = Path(anatomy_scene)
    if environment:
        context["openusd_environment"] = Path(environment)
    return context


def resume_worker(context: dict[str, Any] | None) -> None:
    if not context:
        return
    with state.lock:
        if state.shutting_down:
            return
        state.switching = True
        state.requested_task = context.get("procedure_id") or context["task"]
    switch_worker(**context)


def ensure_worker_available(operation: str) -> None:
    """Keep GPU-owning activities and workstation mutations mutually exclusive."""
    training = training_payload()
    if training["status"] in {"preparing", "running", "stopping"}:
        raise HTTPException(409, f"Stop policy training before {operation}")
    healthcare = healthcare_job_payload()
    if healthcare["status"] in {"preparing", "running", "stopping"}:
        raise HTTPException(409, f"Stop the native healthcare workflow before {operation}")
    matrix = matrix_payload()
    if matrix["status"] in {"preparing", "running"}:
        raise HTTPException(409, f"Finish the Failure Lab matrix before {operation}")
    with state.lock:
        if state.switching:
            raise HTTPException(409, f"The operating room is already loading {state.requested_task}")


def reserve_worker_switch(label: str) -> None:
    """Atomically reserve the interactive worker for a room transition."""
    with state.lock:
        if state.switching:
            raise HTTPException(409, f"The operating room is already loading {state.requested_task}")
        if state.training_status in {"preparing", "running", "stopping"}:
            raise HTTPException(409, "Stop policy training before loading another room")
        if state.healthcare_status in {"preparing", "running", "stopping"}:
            raise HTTPException(409, "Stop the native healthcare workflow before loading another room")
        if state.matrix_status in {"preparing", "running"}:
            raise HTTPException(409, "Finish the Failure Lab matrix before loading another room")
        state.switching = True
        state.requested_task = label
        state.error = None


def worker_startup_failure(log_offset: int) -> str | None:
    """Return a concise fatal startup error written after *log_offset*."""
    try:
        with WORKSTATION_LOG_PATH.open("rb") as stream:
            stream.seek(min(log_offset, WORKSTATION_LOG_PATH.stat().st_size))
            output = stream.read(2_000_000).decode("utf-8", errors="replace")
    except OSError:
        return None
    if not any(marker in output for marker in WORKER_FATAL_MARKERS):
        return None
    if "Out of GPU memory" in output or "CUBLAS_STATUS_ALLOC_FAILED" in output:
        return (
            "The operating room could not start because the GPU is out of memory. "
            "Pause another GPU workload or select the efficient sensor profile, then try again."
        )
    exception_lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith(("RuntimeError:", "ModuleNotFoundError:", "ImportError:", "KeyError:"))
    ]
    return f"The operating-room worker failed during startup: {exception_lines[-1]}" if exception_lines else (
        "The operating-room worker failed during startup. Review the workstation log for the complete traceback."
    )


def switch_worker(
    task: str,
    procedure_id: str = "",
    anatomy_scene: Path | None = None,
    anatomy_scene_id: str = "",
    anatomy_title: str = "",
    openusd_environment: Path | None = None,
    bench_assets: tuple[str, ...] | None = None,
    gripper_open_rad: float | None = None,
    gripper_close_rad: float | None = None,
) -> None:
    log_offset = WORKSTATION_LOG_PATH.stat().st_size if WORKSTATION_LOG_PATH.exists() else 0
    try:
        command = [
            str(args.root / "dr_anmar_workstation.sh"),
            "restart",
            str(args.worker_port),
            task,
            procedure_id,
            str(anatomy_scene or ""),
            anatomy_scene_id,
            anatomy_title,
            str(openusd_environment or ""),
            "default"
            if bench_assets is None
            else ",".join(bench_assets) if bench_assets else "none",
            "" if gripper_open_rad is None else f"{gripper_open_rad:.6g}",
            "" if gripper_close_rad is None else f"{gripper_close_rad:.6g}",
        ]
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
            if failure := worker_startup_failure(log_offset):
                try:
                    stop_interactive_worker()
                except Exception:
                    pass
                raise RuntimeError(failure)
            time.sleep(1.0)
        raise TimeoutError(f"Timed out starting {task}")
    except Exception as exc:
        try:
            stop_interactive_worker()
        except Exception:
            pass
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
    paths = tuple(root.rglob("*.usd")) + tuple(root.rglob("*.usdc"))
    return tuple(sorted(str(path.relative_to(ANATOMY_ROOT)) for path in paths))


def anatomy_payload() -> dict[str, Any]:
    installer = read_json(ASSET_STATUS_PATH, {})
    composed_manifest = read_json(OPENUSD_ROOT / "manifest.json", {})
    composed_by_id = {
        item.get("id"): item
        for item in composed_manifest.get("scenes", [])
        if isinstance(item, dict) and item.get("id") and not item.get("error")
    }
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
        composed = composed_by_id.get(archive.removesuffix(".zip"), {})
        environment_usd = composed.get("environment_usd")
        composed_usd = composed.get("composed_usd")
        runtime_organ_usd = composed.get("runtime_organ_usd")
        environment_ready = bool(environment_usd and Path(environment_usd).is_file())
        composed_ready = bool(composed_usd and Path(composed_usd).is_file())
        runtime_organ_ready = bool(runtime_organ_usd and Path(runtime_organ_usd).is_file())
        primary_usd = next((path for path in usd_files if Path(path).name == "main_scene.usd"), None)
        organ_usd = next((path for path in usd_files if Path(path).name == "models_topo_blender.usdc"), None)
        item.update(
            {
                "id": archive.removesuffix(".zip"),
                "state": scene_state,
                "openusd_ready": bool(organ_usd and runtime_organ_ready and environment_ready and composed_ready),
                "usd_count": len(usd_files),
                "primary_usd": primary_usd,
                "organ_usd": organ_usd,
                "environment_usd": environment_usd,
                "composed_usd": composed_usd,
                "runtime_organ_usd": runtime_organ_usd,
                "source_entrypoint_replaced": composed.get("source_entrypoint_replaced"),
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
        "runtime_note": "Each preset uses a repaired, dependency-clean OpenUSD room plus the official anatomy layer. These are simulation and research assets, not clinical patient models.",
    }


def anatomy_room(room_id: str) -> dict[str, Any] | None:
    return next((item for item in anatomy_payload()["scenes"] if item["id"] == room_id), None)


def anatomy_asset(room: dict[str, Any]) -> Path:
    raw_path = room.get("runtime_organ_usd")
    if not raw_path:
        raise HTTPException(409, "This anatomy package has no dependency-clean organ OpenUSD asset")
    asset = Path(raw_path).expanduser().resolve()
    if OPENUSD_ROOT.resolve() not in asset.parents or asset.suffix != ".usdc" or not asset.is_file():
        raise HTTPException(409, "The prepared anatomy OpenUSD asset path is invalid")
    return asset


def openusd_environment_asset(room: dict[str, Any]) -> Path:
    raw_path = room.get("environment_usd")
    if not raw_path:
        raise HTTPException(409, "This anatomy package has no prepared OpenUSD operating-room layer")
    asset = Path(raw_path).expanduser().resolve()
    if OPENUSD_ROOT.resolve() not in asset.parents or asset.name != "environment.usda" or not asset.is_file():
        raise HTTPException(409, "The prepared OpenUSD operating-room path is invalid")
    return asset


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
            "manifest": state.training_manifest,
        }


def healthcare_job_payload() -> dict[str, Any]:
    with state.lock:
        return {
            "status": state.healthcare_status,
            "job_id": state.healthcare_job_id,
            "workflow": state.healthcare_workflow,
            "mode": state.healthcare_mode,
            "log": state.healthcare_log,
            "started_at": state.healthcare_started_at,
            "exit_code": state.healthcare_exit_code,
            "manifest": state.healthcare_manifest,
            "resume_task": state.healthcare_resume_task,
        }


def matrix_payload() -> dict[str, Any]:
    with state.lock:
        return {
            "status": state.matrix_status,
            "matrix_id": state.matrix_id,
            "demo": state.matrix_demo,
            "total": state.matrix_total,
            "completed": state.matrix_completed,
            "results": list(state.matrix_results),
            "aggregate": dict(state.matrix_aggregate),
            "manifest": state.matrix_manifest,
        }


def descriptive_interval(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"n": len(values), "mean": round(mean, 4), "ci95": [None, None]}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = 1.96 * math.sqrt(variance / len(values))
    return {
        "n": len(values),
        "mean": round(mean, 4),
        "ci95": [round(mean - half_width, 4), round(mean + half_width, 4)],
    }


def aggregate_matrix_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("status") in {"complete", "interrupted"}]
    scores = [float(item["overall_score"]) for item in completed if item.get("overall_score") is not None]
    native = [1.0 if item.get("native_success") else 0.0 for item in completed if item.get("native_success") is not None]
    interventions = [1.0 if int(item.get("intervention_count", 0)) > 0 else 0.0 for item in completed]
    safety = [1.0 if int(item.get("safety_event_count", 0)) > 0 else 0.0 for item in completed]
    contacts = [float(item["max_contact_force_n"]) for item in completed if item.get("max_contact_force_n") is not None]
    per_scenario = {}
    for scenario_id in sorted({str(item.get("scenario_id")) for item in completed}):
        group = [item for item in completed if item.get("scenario_id") == scenario_id]
        group_scores = [float(item["overall_score"]) for item in group if item.get("overall_score") is not None]
        per_scenario[scenario_id] = {
            "rollouts": len(group),
            "skills_score": descriptive_interval(group_scores),
            "native_success_rate": descriptive_interval(
                [1.0 if item.get("native_success") else 0.0 for item in group if item.get("native_success") is not None]
            ),
            "intervention_rate": round(sum(1 for item in group if int(item.get("intervention_count", 0)) > 0) / len(group), 4),
            "safety_event_rate": round(sum(1 for item in group if int(item.get("safety_event_count", 0)) > 0) / len(group), 4),
        }
    return {
        "schema": "dr.anmar.challenge-summary.v1",
        "validation_status": "descriptive_research_statistics_not_clinically_validated",
        "completed_rollouts": len(completed),
        "skills_score": descriptive_interval(scores),
        "native_success_rate": descriptive_interval(native),
        "intervention_rate": descriptive_interval(interventions),
        "safety_event_rate": descriptive_interval(safety),
        "max_contact_force_n": descriptive_interval(contacts),
        "per_scenario": per_scenario,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4096)
def cached_sha256_file(path_value: str, modified_ns: int, size: int) -> str:
    """Cache immutable-artifact hashes without trusting a possibly stale manifest."""
    del modified_ns, size
    return sha256_file(Path(path_value))


def artifact_sha256(path: Path) -> str:
    stat = path.stat()
    return cached_sha256_file(str(path), stat.st_mtime_ns, stat.st_size)


def run_challenge_matrix(demo: str, cases: list[tuple[str, int]], manifest_path: Path) -> None:
    results: list[dict[str, Any]] = []
    for scenario_id, seed in cases:
        result: dict[str, Any] = {"scenario_id": scenario_id, "seed": seed, "status": "starting"}
        try:
            worker_json(
                "/api/evaluate",
                method="POST",
                payload={"demo": demo, "scenario_id": scenario_id, "seed": seed},
            )
            deadline = time.monotonic() + 360.0
            worker = None
            while time.monotonic() < deadline:
                worker = worker_status()
                if worker and worker.get("evaluation_status") in {"complete", "interrupted", "failed"}:
                    break
                time.sleep(0.5)
            if not worker or worker.get("evaluation_status") not in {"complete", "interrupted", "failed"}:
                try:
                    worker_json("/api/handoff", method="POST", payload={})
                except HTTPException:
                    pass
                raise TimeoutError("Challenge rollout exceeded 360 seconds")
            output = worker.get("evaluation_output")
            analysis = worker_json(f"/api/demos/{output}/analysis") if output else {}
            skills = analysis.get("analysis", {})
            result.update(
                {
                    "status": worker.get("evaluation_status"),
                    "output": output,
                    "overall_score": skills.get("overall_score"),
                    "native_success": skills.get("metrics", {}).get("native_success"),
                    "max_contact_force_n": skills.get("metrics", {}).get("max_contact_force_n"),
                    "max_tissue_displacement_m": skills.get("metrics", {}).get("max_tissue_displacement_m"),
                    "safety_event_count": len(skills.get("safety", {}).get("events", [])),
                    "intervention_count": analysis.get("context", {}).get("intervention_count", 0),
                    "analysis": skills,
                }
            )
        except Exception as exc:
            result.update({"status": "failed", "error": str(exc)})
        results.append(result)
        aggregate = aggregate_matrix_results(results)
        with state.lock:
            state.matrix_completed = len(results)
            state.matrix_results = list(results)
            state.matrix_aggregate = aggregate
        manifest = read_json(manifest_path, {})
        manifest.update(
            {"status": "running", "completed": len(results), "results": results, "aggregate": aggregate, "updated_at": utc_now()}
        )
        write_json(manifest_path, manifest)
    aggregate = aggregate_matrix_results(results)
    with state.lock:
        state.matrix_status = "complete"
        state.matrix_results = list(results)
        state.matrix_aggregate = aggregate
    manifest = read_json(manifest_path, {})
    manifest.update(
        {"status": "complete", "completed": len(results), "results": results, "aggregate": aggregate, "finished_at": utc_now()}
    )
    write_json(manifest_path, manifest)


def monitor_training(process: subprocess.Popen, log_file, resume_context: dict[str, Any] | None) -> None:
    code = process.wait()
    log_file.close()
    with state.lock:
        if state.training_process is process:
            state.training_process = None
        state.training_exit_code = code
        state.training_status = "complete" if code == 0 else "failed"
        manifest_path = Path(state.training_manifest) if state.training_manifest else None
    if manifest_path:
        manifest = read_json(manifest_path, {})
        manifest.update({"status": "complete" if code == 0 else "failed", "exit_code": code, "finished_at": utc_now()})
        write_json(manifest_path, manifest)
    resume_worker(resume_context)


def stop_interactive_worker() -> None:
    command = [str(args.root / "dr_anmar_workstation.sh"), "stop", str(args.worker_port)]
    result = subprocess.run(command, cwd=args.root, capture_output=True, text=True, timeout=45)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "Could not pause the interactive operating room")


def monitor_healthcare_workflow(
    process: subprocess.Popen,
    log_file,
    job_id: str,
    manifest_path: Path,
    resume_context: dict[str, Any] | None,
) -> None:
    code = process.wait()
    log_file.close()
    with state.lock:
        requested_stop = state.healthcare_job_id == job_id and state.healthcare_status == "stopping"
        final_status = "stopped" if requested_stop else ("complete" if code == 0 else "failed")
        skip_resume = state.healthcare_skip_resume_job_id == job_id
        if skip_resume:
            state.healthcare_skip_resume_job_id = None
        resume_target = state.healthcare_resume_context if state.healthcare_job_id == job_id else resume_context
        if state.healthcare_job_id == job_id:
            if state.healthcare_process is process:
                state.healthcare_process = None
            state.healthcare_exit_code = code
            state.healthcare_status = final_status
            state.healthcare_resume_context = None
    manifest = read_json(manifest_path, {})
    manifest.update({"status": final_status, "exit_code": code, "finished_at": utc_now()})
    write_json(manifest_path, manifest)
    if not skip_resume:
        resume_worker(resume_target)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = WEB_ROOT / "doctor_studio.html"
    if not path.is_file():
        raise HTTPException(503, "Doctor Studio interface is not installed")
    return HTMLResponse(path.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    current = worker_status()
    return {"catalog": CATALOG, "primary": PRIMARY_TASKS, "current_task": current.get("task") if current else None}


@app.get("/api/curriculum")
def curriculum() -> dict[str, Any]:
    return curriculum_payload()


@app.get("/api/healthcare-platform")
def healthcare_platform() -> dict[str, Any]:
    payload = platform_payload(ANATOMY_ROOT)
    payload["workflows"].append(sonogym_platform_workflow())
    payload["runtime_boundary"]["native_research_providers"] = {
        "SonoGym": [
            "orthopedic patient assets",
            "ultrasound simulation",
            "robot environments",
            "observations and rewards",
            "safe-action constraints",
        ]
    }
    return payload


@app.get("/api/healthcare-job")
def healthcare_job() -> dict[str, Any]:
    return healthcare_job_payload()


@app.get("/api/healthcare-job/log")
def healthcare_job_log() -> dict[str, Any]:
    with state.lock:
        log_value = state.healthcare_log
        job_id = state.healthcare_job_id
    if not log_value:
        return {"job_id": job_id, "text": "No native research workflow has been launched yet."}
    path = Path(log_value).resolve()
    root = HEALTHCARE_JOB_ROOT.resolve()
    if root not in path.parents or not path.is_file():
        return {"job_id": job_id, "text": "The workflow log is not available yet."}
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 48 * 1024))
            output = stream.read().decode("utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(503, "The workflow log could not be read") from exc
    return {"job_id": job_id, "text": output[-24000:]}


@app.post("/api/healthcare-job/start")
def start_healthcare_job(request: HealthcareWorkflowRequest) -> dict[str, Any]:
    is_sonogym = request.workflow == "sonogym_orthopedics"
    is_agentic = request.workflow == "agentic"
    if is_sonogym:
        definition = sonogym_platform_workflow()
        mode_catalog = sonogym_workflow_modes()
    else:
        definition = WORKFLOW_BINDINGS.get(request.workflow)
        if definition is None:
            raise HTTPException(404, "Unknown native healthcare workflow")
        mode_catalog = workflow_modes(request.workflow)
    mode = next((item for item in mode_catalog["modes"] if item["id"] == request.mode), None)
    if mode is None:
        raise HTTPException(404, "This mode is not present in the pinned provider metadata")
    if not mode["launchable"]:
        raise HTTPException(409, mode["blocked_reason"] or "This mode needs advanced workstation setup")
    if not mode.get("launch_ready"):
        missing = ", ".join(mode.get("missing_prerequisites", [])) or "provider runtime"
        raise HTTPException(409, f"Install or configure {missing} before launching this lab")
    if is_sonogym:
        workflow_root = SONOGYM_ROOT
        prerequisites = sonogym_runtime_prerequisites()
        if not all(item["ready"] for item in prerequisites.values()):
            raise HTTPException(409, "The pinned SonoGym source, runtime, assets and ultrasound models must all be ready")
        command = sonogym_launch_command(
            mode_id=request.mode,
            bridge=args.root / "scripts/dr_anmar_sonogym_worker.py",
            port=args.worker_port,
        )
        process_cwd = SONOGYM_ROOT
        process_env = os.environ.copy()
        process_env.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
        process_env.setdefault("PRIVACY_CONSENT", "Y")
        process_env.setdefault("WANDB_MODE", "disabled")
        process_env.setdefault("WANDB_SILENT", "true")
    elif is_agentic:
        workflow_root = I4H_ROOT / definition["directory"]
        arena_runner = workflow_root / "arena/run.sh"
        if not workflow_root.is_dir() or not arena_runner.is_file():
            raise HTTPException(409, "Install the pinned NVIDIA Agentic workflow before launching this environment")
        command = [
            str(arena_runner),
            "--env",
            request.mode,
            "--state-machine",
            "--episodes",
            "1",
            "--num_envs",
            "1",
            "--headless",
            "--disable-cameras",
        ]
        process_cwd = I4H_ROOT
        process_env = os.environ.copy()
        process_env["I4H_WORKFLOWS"] = str(I4H_ROOT)
        uv_path = mode_catalog["agentic_runtime_prerequisites"]["uv"]["path"]
        if uv_path:
            process_env["PATH"] = f"{Path(uv_path).parent}:{process_env.get('PATH', '')}"
        process_env.setdefault("UV_CACHE_DIR", str(args.root / "cache/uv"))
        process_env.setdefault("UV_PYTHON_INSTALL_DIR", str(args.root / "runtime/uv-python"))
    else:
        workflow_root = I4H_ROOT / definition["directory"]
        i4h_cli = I4H_ROOT / "i4h"
        if not workflow_root.is_dir() or not i4h_cli.is_file():
            raise HTTPException(409, "Install the pinned Isaac for Healthcare workflows before launching this mode")
        prerequisites = runtime_prerequisites()
        if not prerequisites["container_runtime"]["ready"]:
            raise HTTPException(409, "Gilgamesh needs Docker Engine before official NVIDIA workflow containers can launch")
        if mode["requires_rti"] and not prerequisites["rti_dds_license"]["ready"]:
            raise HTTPException(409, "This workflow needs an RTI Connext DDS license file before it can launch")
        command = [str(i4h_cli), "run", request.workflow, request.mode]
        process_cwd = I4H_ROOT
        process_env = os.environ.copy()
        process_env["CLI_PINNED_COMMIT"] = HOLOHUB_CLI_COMMIT
        if not (I4H_ROOT / "tools/utilities/cli/holohub.py").is_file():
            process_env["CLI_FORCE_UPDATE"] = "1"
    training = training_payload()
    if training["status"] in {"preparing", "running", "stopping"}:
        raise HTTPException(409, "Stop the policy training lab before launching another native workflow")
    if matrix_payload()["status"] in {"preparing", "running"}:
        raise HTTPException(409, "Finish the active Failure Lab matrix before launching another simulator")
    with state.lock:
        if state.switching:
            raise HTTPException(409, f"The operating room is already loading {state.requested_task}")
        if state.healthcare_status in {"preparing", "running", "stopping"}:
            raise HTTPException(409, "A native healthcare workflow is already running")
    current = worker_status()
    resume_context = capture_worker_context(current, request.resume_workstation)
    resume_task = resume_context["task"] if resume_context else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    job_id = f"healthcare_{stamp}_{request.workflow}_{request.mode}"
    HEALTHCARE_JOB_ROOT.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = HEALTHCARE_JOB_ROOT / f"{job_id}.log"
    manifest_path = EXPERIMENT_ROOT / f"{job_id}.json"
    manifest = {
        "schema": "dr.anmar.healthcare-workflow-job.v1",
        "experiment_id": job_id,
        "kind": (
            "sonogym_orthopedic_workflow"
            if is_sonogym
            else "isaac_for_healthcare_agentic_workflow"
            if is_agentic
            else "isaac_for_healthcare_workflow"
        ),
        "simulation_only": True,
        "clinical_use": False,
        "created_at": utc_now(),
        "source_revision": source_revision(),
        "runtime_provenance": (current or {}).get("runtime_provenance", {}),
        "i4h_revision": None if is_sonogym else repository_revision(I4H_ROOT),
        "sonogym_revision": repository_revision(SONOGYM_ROOT) if is_sonogym else None,
        "workflow_metadata_sha256": mode_catalog.get("metadata_sha256"),
        "workflow": request.workflow,
        "workflow_title": definition["title"],
        "mode": request.mode,
        "mode_title": mode["title"],
        "mode_description": mode["description"],
        "configuration": {
            "resume_workstation": request.resume_workstation,
            "resume_context": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in (resume_context or {}).items()
            },
            "hardware_access": False,
            "custom_arguments": False,
            "holohub_cli_commit": None if is_sonogym or is_agentic else HOLOHUB_CLI_COMMIT,
            "sonogym_source_commit": SONOGYM_COMMIT if is_sonogym else None,
        },
        "command": command,
        "log": str(log_path),
        "status": "preparing",
    }
    write_json(manifest_path, manifest)
    with state.lock:
        if state.switching:
            raise HTTPException(409, f"The operating room is already loading {state.requested_task}")
        if state.training_status in {"preparing", "running", "stopping"}:
            raise HTTPException(409, "Stop policy training before launching a native healthcare workflow")
        if state.matrix_status in {"preparing", "running"}:
            raise HTTPException(409, "Finish the Failure Lab matrix before launching another simulator")
        if state.healthcare_status in {"preparing", "running", "stopping"}:
            raise HTTPException(409, "A native healthcare workflow is already running")
        state.healthcare_status = "preparing"
        state.healthcare_job_id = job_id
        state.healthcare_workflow = request.workflow
        state.healthcare_mode = request.mode
        state.healthcare_log = str(log_path)
        state.healthcare_started_at = None
        state.healthcare_exit_code = None
        state.healthcare_manifest = str(manifest_path)
        state.healthcare_resume_task = resume_task
        state.healthcare_resume_context = resume_context
    log_file = log_path.open("ab", buffering=0)
    try:
        # Stop by PID even when the old workstation is still booting and has
        # not begun answering /api/status yet.
        stop_interactive_worker()
        process = subprocess.Popen(
            command,
            cwd=process_cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=process_env,
        )
    except Exception as exc:
        log_file.close()
        with state.lock:
            state.healthcare_status = "failed"
            state.healthcare_exit_code = -1
        manifest.update({"status": "failed", "error": str(exc), "finished_at": utc_now()})
        write_json(manifest_path, manifest)
        if resume_context:
            threading.Thread(target=resume_worker, args=(resume_context,), daemon=True, name="dr-anmar-resume").start()
        raise HTTPException(500, f"The native workflow could not start: {exc}") from exc
    started_at = utc_now()
    with state.lock:
        state.healthcare_process = process
        state.healthcare_status = "running"
        state.healthcare_started_at = started_at
    manifest.update({"status": "running", "pid": process.pid, "started_at": started_at})
    write_json(manifest_path, manifest)
    threading.Thread(
        target=monitor_healthcare_workflow,
        args=(process, log_file, job_id, manifest_path, resume_context),
        daemon=True,
        name="dr-anmar-healthcare-workflow",
    ).start()
    return {"ok": True, **healthcare_job_payload()}


@app.post("/api/healthcare-job/stop")
def stop_healthcare_job() -> dict[str, Any]:
    with state.lock:
        process = state.healthcare_process
    if process is None or process.poll() is not None:
        raise HTTPException(409, "No native healthcare workflow is running")
    os.killpg(process.pid, signal.SIGTERM)
    with state.lock:
        state.healthcare_status = "stopping"
    return {"ok": True, **healthcare_job_payload()}


@app.get("/api/multimodal-studies")
def multimodal_studies() -> dict[str, Any]:
    files = sorted(STUDY_ROOT.glob("study_*.json"), reverse=True) if STUDY_ROOT.is_dir() else []
    return {"studies": [read_json(path, {}) for path in files[:100] if read_json(path, {})], "root": str(STUDY_ROOT)}


@app.post("/api/multimodal-studies")
def create_multimodal_study(request: MultimodalStudyRequest) -> dict[str, Any]:
    if not request.title.strip() or len(request.title) > 120:
        raise HTTPException(400, "Study title must contain 1 to 120 characters")
    if not request.clinical_question.strip() or len(request.clinical_question) > 500:
        raise HTTPException(400, "Describe one concise clinical research question")
    available_modalities = {item["id"] for item in MODALITY_CATALOG}
    modalities = list(dict.fromkeys(request.modalities))
    unknown_modalities = sorted(set(modalities) - available_modalities)
    if not modalities or unknown_modalities:
        raise HTTPException(400, f"Choose valid study modalities; unknown: {', '.join(unknown_modalities)}")
    available_policies = {item["id"] for item in POLICY_STARTING_POINTS}
    if request.policy not in available_policies:
        raise HTTPException(400, "Unknown policy starting point")
    if request.teleoperation not in {"keyboard_pointer", "gamepad", "external_teleop", "xr", "haptic"}:
        raise HTTPException(400, "Unknown teleoperation method")
    if request.task not in TASKS_BY_ID and not request.task.startswith("Isaac-"):
        raise HTTPException(400, "Unknown surgical task")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    study_id = f"study_{stamp}"
    manifest = study_manifest(
        study_id=study_id,
        title=request.title.strip(),
        clinical_question=request.clinical_question.strip(),
        task=request.task,
        modalities=modalities,
        policy=request.policy,
        teleoperation=request.teleoperation,
        created_at=utc_now(),
        source_revision=source_revision(),
    )
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(STUDY_ROOT / f"{study_id}.json", manifest)
    return {"ok": True, "study": manifest, "download": f"/api/multimodal-studies/{study_id}/download"}


@app.get("/api/multimodal-studies/{study_id}/download")
def download_multimodal_study(study_id: str) -> FileResponse:
    if not study_id.startswith("study_") or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in study_id):
        raise HTTPException(400, "Invalid study ID")
    path = STUDY_ROOT / f"{study_id}.json"
    if not path.is_file():
        raise HTTPException(404, "Study manifest not found")
    return FileResponse(path, filename=path.name, media_type="application/json")


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


@app.get("/api/procedure-rooms")
def procedure_rooms() -> dict[str, Any]:
    """Return the room catalog without duplicating runtime launch checks.

    The main NVIDIA bench and experimental research rooms share one launch
    endpoint. Asset and provider errors are reported by that endpoint when the
    user actually opens a room; stale preflight booleans must never hide a
    running simulator or disable the clinician UI.
    """

    payload = procedure_payload()
    available_anatomy = {scene["id"]: scene for scene in anatomy_payload()["scenes"]}
    for room in payload["rooms"]:
        if room.get("nvidia_native_bench"):
            room["gripper_profile"] = {
                "open_rad": CANONICAL_PSM_GRIPPER_PROFILE.open_rad,
                "close_rad": CANONICAL_PSM_GRIPPER_PROFILE.close_rad,
            }
        anatomy = available_anatomy.get(room["anatomy_scene"])
        room["location"] = (
            "main" if room["id"] == payload["default"] else "research"
        )
        if room.get("external_provider") == "nvidia_robotic_ultrasound":
            room["anatomy_title"] = "NVIDIA robotic ultrasound patient model"
            continue
        if room.get("external_provider") == "sonogym_orthopedics":
            room["anatomy_title"] = "SonoGym CT-derived lumbar patient · L4 vertebra"
            continue
        room["anatomy_title"] = (
            str(room.get("anatomy_focus") or "Dry-lab field")
            if room.get("hide_anatomy")
            else anatomy["title"] if anatomy else room.get("anatomy_focus", "")
        )
    return payload


def launch_sonogym_procedure(procedure: dict[str, Any], force_restart: bool = False) -> dict[str, Any]:
    """Launch or seamlessly replace one native SonoGym operating room."""
    requested_mode = str(procedure["provider_mode"])
    current_worker = worker_status()
    with state.lock:
        active = state.healthcare_status in {"preparing", "running", "stopping"}
        active_workflow = state.healthcare_workflow
        active_mode = state.healthcare_mode
        active_process = state.healthcare_process
        active_job_id = state.healthcare_job_id
        preserved_resume = state.healthcare_resume_context
    if (
        active
        and active_workflow == "sonogym_orthopedics"
        and active_mode == requested_mode
        and not force_restart
        and current_worker
        and current_worker.get("worker_kind") == "sonogym_native"
        and current_worker.get("frame_id", 0) > 0
    ):
        return {"ok": True, **healthcare_job_payload(), "already_ready": True}
    replacing = active and active_workflow == "sonogym_orthopedics"
    if replacing:
        if active_process is None or active_process.poll() is not None:
            with state.lock:
                state.healthcare_status = "failed"
            raise HTTPException(503, "The previous SonoGym room exited unexpectedly; choose the room again")
        with state.lock:
            state.healthcare_skip_resume_job_id = active_job_id
            state.healthcare_status = "stopping"
        os.killpg(active_process.pid, signal.SIGTERM)
        try:
            active_process.wait(timeout=45)
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(504, "The previous SonoGym room did not finish closing") from exc
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with state.lock:
                finished = state.healthcare_status not in {"preparing", "running", "stopping"}
            if finished:
                break
            time.sleep(0.05)
        else:
            raise HTTPException(504, "The previous SonoGym room did not release the native runtime")
    result = start_healthcare_job(
        HealthcareWorkflowRequest(
            workflow="sonogym_orthopedics",
            mode=requested_mode,
            resume_workstation=not replacing,
        )
    )
    if replacing and preserved_resume:
        with state.lock:
            state.healthcare_resume_context = preserved_resume
            state.healthcare_resume_task = preserved_resume.get("task")
            manifest_value = state.healthcare_manifest
        if manifest_value:
            manifest_path = Path(manifest_value)
            manifest = read_json(manifest_path, {})
            manifest.setdefault("configuration", {})["resume_workstation"] = True
            manifest.setdefault("configuration", {})["resume_context"] = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in preserved_resume.items()
            }
            write_json(manifest_path, manifest)
        result["resume_task"] = preserved_resume.get("task")
    return result


def reserve_sonogym_room_switch(label: str) -> bool:
    """Reserve a normal room while a native SonoGym worker is active."""
    with state.lock:
        active_sonogym = (
            state.healthcare_status in {"preparing", "running", "stopping"}
            and state.healthcare_workflow == "sonogym_orthopedics"
        )
        if not active_sonogym:
            return False
        if state.switching:
            raise HTTPException(409, f"The operating room is already loading {state.requested_task}")
        if state.training_status in {"preparing", "running", "stopping"}:
            raise HTTPException(409, "Stop policy training before loading another room")
        if state.matrix_status in {"preparing", "running"}:
            raise HTTPException(409, "Finish the Failure Lab matrix before loading another room")
        state.switching = True
        state.requested_task = label
        state.error = None
        return True


def stop_sonogym_for_room_switch() -> None:
    """Stop SonoGym without restoring the room that preceded it."""
    with state.lock:
        process = state.healthcare_process
        job_id = state.healthcare_job_id
        active_sonogym = (
            state.healthcare_status in {"preparing", "running", "stopping"}
            and state.healthcare_workflow == "sonogym_orthopedics"
        )
        if not active_sonogym:
            return
        state.healthcare_skip_resume_job_id = job_id
        state.healthcare_status = "stopping"
    if process is None:
        raise RuntimeError("The active SonoGym worker has no managed process")
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("The SonoGym room did not finish closing") from exc
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with state.lock:
            released = (
                state.healthcare_job_id != job_id
                or state.healthcare_status not in {"preparing", "running", "stopping"}
            )
        if released:
            return
        time.sleep(0.05)
    raise TimeoutError("The SonoGym room did not release the native runtime")


def switch_sonogym_to_worker(
    task: str,
    procedure_id: str,
    anatomy_scene: Path | None,
    anatomy_scene_id: str,
    anatomy_title: str,
    openusd_environment: Path | None,
    bench_assets: tuple[str, ...] | None = None,
    gripper_open_rad: float | None = None,
    gripper_close_rad: float | None = None,
) -> None:
    """Replace the active SonoGym process with the selected Isaac room."""
    try:
        stop_sonogym_for_room_switch()
    except Exception as exc:
        with state.lock:
            state.error = str(exc)
            state.switching = False
        return
    switch_worker(
        task,
        procedure_id,
        anatomy_scene,
        anatomy_scene_id,
        anatomy_title,
        openusd_environment,
        bench_assets,
        gripper_open_rad,
        gripper_close_rad,
    )


@app.post("/api/healthcare-job/restart")
def restart_healthcare_job() -> dict[str, Any]:
    with state.lock:
        workflow = state.healthcare_workflow
        mode = state.healthcare_mode
        running = state.healthcare_status in {"preparing", "running", "stopping"}
    if not running or workflow != "sonogym_orthopedics" or not mode:
        raise HTTPException(409, "A native SonoGym room is not running")
    procedure = next(
        (
            item
            for item in PROCEDURE_ROOMS
            if item.get("external_provider") == "sonogym_orthopedics" and item.get("provider_mode") == mode
        ),
        None,
    )
    if procedure is None:
        raise HTTPException(404, "The active SonoGym procedure is not in the room catalog")
    return launch_sonogym_procedure(procedure, force_restart=True)


@app.post("/api/procedure-rooms/launch")
def launch_procedure_room(request: ProcedureLaunchRequest) -> dict[str, Any]:
    procedure = PROCEDURES_BY_ID.get(request.procedure_id)
    if procedure is None:
        raise HTTPException(404, "Unknown procedure room")
    if procedure.get("external_provider") == "nvidia_robotic_ultrasound":
        result = start_healthcare_job(
            HealthcareWorkflowRequest(
                workflow="robotic_ultrasound",
                mode="teleop_with_ultrasound",
                resume_workstation=True,
            )
        )
        return {
            **result,
            "procedure_id": request.procedure_id,
            "title": procedure["title"],
            "native_provider": f"NVIDIA Isaac for Healthcare {I4H_RELEASE}",
        }
    if procedure.get("external_provider") == "sonogym_orthopedics":
        result = launch_sonogym_procedure(procedure)
        return {
            **result,
            "procedure_id": request.procedure_id,
            "title": procedure["title"],
            "native_provider": "SonoGym on Isaac Lab 2.1.0",
        }
    selected_bench_assets = bench_asset_selection(procedure, request.bench_assets)
    selected_gripper_profile = psm_gripper_selection(
        procedure,
        request.gripper_open_rad,
        request.gripper_close_rad,
    )
    binding = resolve_native_room(str(procedure["id"]))
    if binding and not binding.get("available"):
        raise HTTPException(409, "Required room assets are not installed on this worker.")
    missing_bench_assets = missing_required_bench_assets(procedure, selected_bench_assets)
    if missing_bench_assets:
        raise HTTPException(
            409,
            "Missing required room assets: " + ", ".join(missing_bench_assets) + ".",
        )
    if procedure.get("hide_anatomy"):
        selected_anatomy = ""
        room_title = str(procedure.get("anatomy_focus") or "NVIDIA dry-lab field")
        asset = None
        environment_scene_id = str(procedure.get("operating_room_environment", ""))
        environment_room = anatomy_room(environment_scene_id) if environment_scene_id else None
        if environment_scene_id and environment_room is None:
            raise HTTPException(404, "Unknown OpenUSD operating-room environment")
        environment = (
            openusd_environment_asset(environment_room)
            if environment_room is not None
            else None
        )
    else:
        selected_anatomy = request.anatomy_scene or procedure["anatomy_scene"]
        room = anatomy_room(selected_anatomy)
        if room is None:
            raise HTTPException(404, "Unknown OpenUSD anatomy scene")
        room_title = room["title"]
        asset = anatomy_asset(room)
        environment = openusd_environment_asset(room)
    replacing_sonogym = reserve_sonogym_room_switch(procedure["title"])
    if not replacing_sonogym:
        ensure_worker_available("loading an operating room")
    current = worker_status()
    if (
        not replacing_sonogym
        and current
        and current.get("task") == procedure["task"]
        and current.get("procedure", {}).get("id") == request.procedure_id
        and current.get("anatomy_scene_id") == selected_anatomy
        and (
            selected_bench_assets is None
            or tuple(current.get("procedure", {}).get("active_bench_assets", ()))
            == selected_bench_assets
        )
        and (
            selected_gripper_profile is None
            or (
                float(current.get("gripper_profile", {}).get("open_rad", -1.0)),
                float(current.get("gripper_profile", {}).get("close_rad", -1.0)),
            )
            == selected_gripper_profile
        )
        and current.get("frame_id", 0) > 0
    ):
        return {"ok": True, "procedure_id": request.procedure_id, "already_ready": True}
    if not replacing_sonogym:
        reserve_worker_switch(procedure["title"])
    threading.Thread(
        target=switch_sonogym_to_worker if replacing_sonogym else switch_worker,
        args=(
            procedure["task"],
            request.procedure_id,
            asset,
            selected_anatomy,
            room_title,
            environment,
            selected_bench_assets,
            selected_gripper_profile[0] if selected_gripper_profile else None,
            selected_gripper_profile[1] if selected_gripper_profile else None,
        ),
        daemon=True,
        name="dr-anmar-sonogym-room-switch" if replacing_sonogym else "dr-anmar-procedure-switch",
    ).start()
    return {
        "ok": True,
        "procedure_id": request.procedure_id,
        "title": procedure["title"],
        "anatomy_scene": selected_anatomy,
        "anatomy_title": room_title,
        "bench_assets": list(selected_bench_assets or ()),
        "gripper_open_rad": (
            selected_gripper_profile[0] if selected_gripper_profile else None
        ),
        "gripper_close_rad": (
            selected_gripper_profile[1] if selected_gripper_profile else None
        ),
    }


@app.post("/api/anatomy/launch")
def launch_anatomy(request: AnatomyLaunchRequest) -> dict[str, Any]:
    room = anatomy_room(request.room_id)
    if room is None:
        raise HTTPException(404, "Unknown anatomy operating-room preset")
    if not room["openusd_ready"] or not room["organ_usd"]:
        raise HTTPException(409, "This anatomy room has not finished installing")
    scene = anatomy_asset(room)
    environment = openusd_environment_asset(room)
    ensure_worker_available("loading an anatomy room")
    current = worker_status()
    if current and current.get("anatomy_scene_id") == request.room_id and current.get("frame_id", 0) > 0:
        return {"ok": True, "room_id": request.room_id, "already_ready": True}
    reserve_worker_switch(room["title"])
    threading.Thread(
        target=switch_worker,
        args=(
            PROCEDURES_BY_ID["synthetic-anatomy-navigation"]["task"],
            "synthetic-anatomy-navigation",
            scene,
            request.room_id,
            room["title"],
            environment,
        ),
        daemon=True,
        name="dr-anmar-anatomy-switch",
    ).start()
    return {"ok": True, "room_id": request.room_id, "title": room["title"]}


@app.get("/api/demos")
def demonstrations(limit: int = 100, offset: int = 0) -> dict[str, Any]:
    if not 1 <= limit <= 500 or offset < 0:
        raise HTTPException(400, "limit must be 1–500 and offset must be non-negative")
    return worker_json(f"/api/demos?limit={limit}&offset={offset}")


@app.get("/api/demos/{name}/analysis")
def demonstration_analysis(name: str) -> dict[str, Any]:
    if Path(name).name != name or not name.endswith(".npz"):
        raise HTTPException(400, "Invalid demonstration name")
    return worker_json(f"/api/demos/{name}/analysis")


@app.post("/api/demos/{name}/replay")
def replay_demonstration(name: str) -> dict[str, Any]:
    ensure_worker_available("replaying a demonstration")
    if Path(name).name != name or not name.endswith(".npz"):
        raise HTTPException(400, "Invalid demonstration name")
    return worker_json(f"/api/replay/{name}", method="POST", payload={})


@app.post("/api/demos/{name}/reference")
def set_demonstration_reference(name: str) -> dict[str, Any]:
    ensure_worker_available("changing the clinician reference")
    if Path(name).name != name or not name.endswith(".npz"):
        raise HTTPException(400, "Invalid demonstration name")
    return worker_json(f"/api/demos/{name}/reference", method="POST", payload={})


@app.post("/api/reference-ghost")
def reference_ghost(request: ReferenceGhostRequest) -> dict[str, Any]:
    ensure_worker_available("changing the reference guide")
    if request.demo and (Path(request.demo).name != request.demo or not request.demo.endswith(".npz")):
        raise HTTPException(400, "Invalid demonstration name")
    return worker_json(
        "/api/reference-ghost",
        method="POST",
        payload={"enabled": request.enabled, "demo": request.demo},
    )


@app.get("/api/demos/{name}/comparison")
def demonstration_comparison(name: str) -> dict[str, Any]:
    if Path(name).name != name or not name.endswith(".npz"):
        raise HTTPException(400, "Invalid demonstration name")
    return worker_json(f"/api/demos/{name}/comparison")


@app.get("/api/failure-scenarios")
def failure_scenarios() -> dict[str, Any]:
    return worker_json("/api/scenarios")


@app.post("/api/failure-scenarios/apply")
def apply_failure_scenario(request: ScenarioApplicationRequest) -> dict[str, Any]:
    ensure_worker_available("applying a failure scenario")
    current = worker_status()
    if current is None or current.get("mode") == "anatomy":
        raise HTTPException(409, "Load a runnable lesson robot before applying a challenge")
    result = worker_json(
        "/api/scenario",
        method="POST",
        payload={"scenario_id": request.scenario_id, "seed": request.seed},
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    experiment_id = f"failure_{stamp}_{request.scenario_id}"
    manifest_path = EXPERIMENT_ROOT / f"{experiment_id}.json"
    write_json(
        manifest_path,
        {
            "schema": "dr.anmar.experiment.v1",
            "experiment_id": experiment_id,
            "kind": "failure_scenario",
            "simulation_only": True,
            "created_at": utc_now(),
            "source_revision": source_revision(),
            "runtime_provenance": current.get("runtime_provenance", {}),
            "task": current.get("task"),
            "configuration": {"scenario_id": request.scenario_id, "seed": request.seed},
            "status": "ready_for_demonstration",
            "result": result,
        },
    )
    result["experiment_id"] = experiment_id
    result["manifest"] = str(manifest_path)
    return result


@app.get("/api/challenge-matrix")
def challenge_matrix_status() -> dict[str, Any]:
    return matrix_payload()


@app.post("/api/challenge-matrix")
def start_challenge_matrix(request: ChallengeMatrixRequest) -> dict[str, Any]:
    if Path(request.demo).name != request.demo or not request.demo.endswith(".npz"):
        raise HTTPException(400, "Invalid demonstration name")
    if not request.scenario_ids or not request.seeds:
        raise HTTPException(400, "Choose at least one scenario and one seed")
    if len(request.scenario_ids) * len(request.seeds) > 30:
        raise HTTPException(400, "A single challenge matrix is limited to 30 rollouts")
    if any(not 0 <= seed <= 2_147_483_647 for seed in request.seeds):
        raise HTTPException(400, "Seeds must be between 0 and 2147483647")
    available = worker_json("/api/scenarios").get("scenarios", [])
    available_ids = {item["id"] for item in available}
    unknown = sorted(set(request.scenario_ids) - available_ids)
    if unknown:
        raise HTTPException(404, f"Unknown failure scenarios: {', '.join(unknown)}")
    worker_demos = worker_json("/api/demos").get("demos", [])
    if request.demo not in {item.get("name") for item in worker_demos}:
        raise HTTPException(404, "Demonstration not found")
    current = worker_status()
    with state.lock:
        if state.switching:
            raise HTTPException(409, f"The operating room is already loading {state.requested_task}")
        if state.training_status in {"preparing", "running", "stopping"}:
            raise HTTPException(409, "Stop policy training before starting a Failure Lab matrix")
        if state.healthcare_status in {"preparing", "running", "stopping"}:
            raise HTTPException(409, "Stop the native healthcare workflow before starting a Failure Lab matrix")
        if state.matrix_status in {"preparing", "running"}:
            raise HTTPException(409, "A challenge matrix is already running")
        state.matrix_status = "preparing"
    cases = [(scenario_id, seed) for scenario_id in request.scenario_ids for seed in request.seeds]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    matrix_id = f"matrix_{stamp}"
    manifest_path = EXPERIMENT_ROOT / f"{matrix_id}.json"
    manifest = {
        "schema": "dr.anmar.challenge-matrix.v1",
        "experiment_id": matrix_id,
        "kind": "challenge_matrix",
        "simulation_only": True,
        "created_at": utc_now(),
        "source_revision": source_revision(),
        "runtime_provenance": (current or {}).get("runtime_provenance", {}),
        "demo": request.demo,
        "scenario_ids": request.scenario_ids,
        "seeds": request.seeds,
        "total": len(cases),
        "completed": 0,
        "status": "preparing",
        "results": [],
    }
    write_json(manifest_path, manifest)
    with state.lock:
        state.matrix_id = matrix_id
        state.matrix_demo = request.demo
        state.matrix_total = len(cases)
        state.matrix_completed = 0
        state.matrix_results = []
        state.matrix_aggregate = {}
        state.matrix_manifest = str(manifest_path)
    with state.lock:
        state.matrix_status = "running"
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    threading.Thread(
        target=run_challenge_matrix,
        args=(request.demo, cases, manifest_path),
        daemon=True,
        name="dr-anmar-challenge-matrix",
    ).start()
    return {"ok": True, **matrix_payload()}


@app.post("/api/autonomy")
def autonomy(request: AutonomyModeRequest) -> dict[str, Any]:
    ensure_worker_available("changing autonomy mode")
    return worker_json("/api/autonomy", method="POST", payload={"mode": request.mode})


@app.post("/api/handoff")
def handoff() -> dict[str, Any]:
    ensure_worker_available("starting an assisted handoff")
    return worker_json("/api/handoff", method="POST", payload={})


@app.get("/api/dataset-cards")
def dataset_cards(limit: int = 100, offset: int = 0) -> dict[str, Any]:
    if not 1 <= limit <= 500 or offset < 0:
        raise HTTPException(400, "limit must be 1–500 and offset must be non-negative")
    files = sorted(DATASET_CARD_ROOT.glob("dataset_*.json"), reverse=True) if DATASET_CARD_ROOT.is_dir() else []
    items = []
    for path in files[offset : offset + limit]:
        card = read_json(path, {})
        if card:
            items.append(card)
    return {"dataset_cards": items, "total": len(files), "offset": offset, "limit": limit, "has_more": offset + len(items) < len(files), "root": str(DATASET_CARD_ROOT)}


@app.post("/api/dataset-cards")
def create_dataset_card(request: DatasetCardRequest) -> dict[str, Any]:
    demo_names = list(dict.fromkeys(request.demos))
    if not demo_names:
        raise HTTPException(400, "Choose at least one demonstration")
    if len(demo_names) > 100:
        raise HTTPException(400, "A dataset card can contain at most 100 demonstrations")
    if not request.title.strip() or len(request.title) > 120 or len(request.intended_use) > 500:
        raise HTTPException(400, "Dataset title or intended use is invalid")
    worker_items = {item.get("name"): item for item in worker_json("/api/demos").get("demos", [])}
    entries = []
    for name in demo_names:
        if Path(name).name != name or not name.endswith(".npz"):
            raise HTTPException(400, f"Invalid demonstration name: {name}")
        path = DEMO_ROOT / name
        item = worker_items.get(name)
        if item is None or not path.is_file():
            raise HTTPException(404, f"Demonstration not found: {name}")
        integrity = item.get("integrity", {})
        if not integrity.get("valid"):
            raise HTTPException(422, f"Demonstration is unreadable: {name}")
        if not integrity.get("training_eligible"):
            raise HTTPException(422, f"Demonstration is too short for a dataset: {name}")
        manifest_path = path.with_suffix(".json")
        source_manifest = read_json(manifest_path, {}) if manifest_path.is_file() else {}
        data_hash = artifact_sha256(path)
        entries.append(
            {
                "name": name,
                "sha256": data_hash,
                "bytes": path.stat().st_size,
                "manifest": manifest_path.name if manifest_path.is_file() else None,
                "manifest_sha256": artifact_sha256(manifest_path) if manifest_path.is_file() else None,
                "task": item.get("task"),
                "frames": item.get("frames"),
                "duration_s": item.get("duration_s"),
                "modalities": item.get("modalities", {}),
                "context": item.get("context", {}),
                "runtime_provenance": source_manifest.get("runtime_provenance", {}),
                "clinician_reference": bool(item.get("is_reference")),
            }
        )
    canonical = {
        "title": request.title.strip(),
        "intended_use": request.intended_use.strip(),
        "demonstrations": entries,
    }
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    dataset_id = f"dataset_{fingerprint[:16]}"
    tasks = sorted({entry["task"] for entry in entries if entry.get("task")})
    card = {
        "schema": "dr.anmar.dataset-card.v1",
        "dataset_id": dataset_id,
        "immutable": True,
        "simulation_only": True,
        "clinical_use": False,
        "created_at": utc_now(),
        "source_revision": source_revision(),
        "content_sha256": fingerprint,
        "title": canonical["title"],
        "intended_use": canonical["intended_use"],
        "tasks": tasks,
        "summary": {
            "demonstrations": len(entries),
            "frames": sum(int(entry.get("frames") or 0) for entry in entries),
            "duration_s": round(sum(float(entry.get("duration_s") or 0.0) for entry in entries), 2),
            "clinician_references": sum(1 for entry in entries if entry["clinician_reference"]),
        },
        "demonstrations": entries,
        "limitations": [
            "Simulation and preclinical research evidence only.",
            "Telemetry-derived coaching and research advisories are not clinically validated.",
            "Checksums detect any later change to the referenced demonstration files.",
        ],
    }
    DATASET_CARD_ROOT.mkdir(parents=True, exist_ok=True)
    path = DATASET_CARD_ROOT / f"{dataset_id}.json"
    if not path.exists():
        write_json(path, card)
    else:
        card = read_json(path, card)
    return {"ok": True, "dataset_card": card, "download": f"/api/dataset-cards/{dataset_id}/download"}


@app.get("/api/dataset-cards/{dataset_id}/download")
def download_dataset_card(dataset_id: str) -> FileResponse:
    if not dataset_id.startswith("dataset_") or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in dataset_id):
        raise HTTPException(400, "Invalid dataset card ID")
    path = DATASET_CARD_ROOT / f"{dataset_id}.json"
    if not path.is_file():
        raise HTTPException(404, "Dataset card not found")
    return FileResponse(path, filename=path.name, media_type="application/json")


def _artifact_manifest(root: Path, artifact_id: str, prefix: str) -> tuple[Path, dict[str, Any]]:
    if not artifact_id.startswith(prefix) or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in artifact_id):
        raise HTTPException(400, f"Invalid {prefix.rstrip('_')} ID")
    path = root / f"{artifact_id}.json"
    if not path.is_file():
        raise HTTPException(404, f"Artifact not found: {artifact_id}")
    payload = read_json(path, {})
    if not payload:
        raise HTTPException(422, f"Artifact is unreadable: {artifact_id}")
    return path, payload


@app.get("/api/policy-evaluation-cards")
def policy_evaluation_cards(limit: int = 100, offset: int = 0) -> dict[str, Any]:
    if not 1 <= limit <= 500 or offset < 0:
        raise HTTPException(400, "limit must be 1–500 and offset must be non-negative")
    files = sorted(POLICY_CARD_ROOT.glob("policy_*.json"), reverse=True) if POLICY_CARD_ROOT.is_dir() else []
    cards = [card for path in files[offset : offset + limit] if (card := read_json(path, {}))]
    return {"policy_evaluation_cards": cards, "total": len(files), "offset": offset, "limit": limit, "has_more": offset + len(cards) < len(files)}


@app.post("/api/policy-evaluation-cards")
def create_policy_evaluation_card(request: PolicyEvaluationCardRequest) -> dict[str, Any]:
    if not request.title.strip() or len(request.title) > 120:
        raise HTTPException(400, "Policy card title is required and limited to 120 characters")
    dataset_path, dataset = _artifact_manifest(DATASET_CARD_ROOT, request.dataset_id, "dataset_")
    training_path, training = _artifact_manifest(EXPERIMENT_ROOT, request.training_experiment_id, "training_")
    matrix_path, matrix = _artifact_manifest(EXPERIMENT_ROOT, request.challenge_matrix_id, "matrix_")
    if training.get("kind") != "policy_training" or training.get("status") != "complete":
        raise HTTPException(422, "The selected policy-training run must be complete")
    if matrix.get("kind") != "challenge_matrix" or matrix.get("status") != "complete":
        raise HTTPException(422, "The selected challenge matrix must be complete")
    dataset_demos = {item.get("name") for item in dataset.get("demonstrations", [])}
    if matrix.get("demo") not in dataset_demos:
        raise HTTPException(422, "The challenge matrix demonstration is not part of the selected dataset")
    checkpoint = Path(request.checkpoint_path).expanduser().resolve()
    allowed_root = DR_ANMAR_ROOT.resolve()
    if not checkpoint.is_file() or not checkpoint.is_relative_to(allowed_root):
        raise HTTPException(400, "Checkpoint must be an existing file inside the Dr.Anmar data root")
    evidence = {
        "dataset": {"id": request.dataset_id, "sha256": artifact_sha256(dataset_path), "content_sha256": dataset.get("content_sha256")},
        "training": {"id": request.training_experiment_id, "sha256": artifact_sha256(training_path), "status": training.get("status")},
        "checkpoint": {"path": str(checkpoint.relative_to(allowed_root)), "sha256": artifact_sha256(checkpoint), "bytes": checkpoint.stat().st_size},
        "challenge_matrix": {"id": request.challenge_matrix_id, "sha256": artifact_sha256(matrix_path), "aggregate": matrix.get("aggregate", {})},
    }
    policy_card_content = {"title": request.title.strip(), "task": training.get("task"), "evidence": evidence}
    content_hash = hashlib.sha256(
        json.dumps(policy_card_content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    card_id = f"policy_{content_hash[:16]}"
    card = {
        "schema": "dr.anmar.policy-evaluation-card.v1",
        "policy_card_id": card_id,
        "immutable": True,
        "simulation_only": True,
        "clinical_use": False,
        "created_at": utc_now(),
        "source_revision": source_revision(),
        "content_sha256": content_hash,
        "title": request.title.strip(),
        "task": training.get("task"),
        "evidence": evidence,
        "limitations": [
            "Simulation and preclinical research evidence only.",
            "Challenge statistics and coaching metrics require clinician and construct validation.",
            "The card binds immutable hashes; it does not certify clinical safety or efficacy.",
        ],
    }
    POLICY_CARD_ROOT.mkdir(parents=True, exist_ok=True)
    path = POLICY_CARD_ROOT / f"{card_id}.json"
    if not path.exists():
        write_json(path, card)
    else:
        card = read_json(path, card)
    return {"ok": True, "policy_evaluation_card": card, "download": f"/api/policy-evaluation-cards/{card_id}/download"}


@app.get("/api/policy-evaluation-cards/{card_id}/download")
def download_policy_evaluation_card(card_id: str) -> FileResponse:
    path, _ = _artifact_manifest(POLICY_CARD_ROOT, card_id, "policy_")
    return FileResponse(path, filename=path.name, media_type="application/json")


@app.get("/api/experiments")
def experiments(limit: int = 100, offset: int = 0) -> dict[str, Any]:
    if not 1 <= limit <= 500 or offset < 0:
        raise HTTPException(400, "limit must be 1–500 and offset must be non-negative")
    files = sorted(EXPERIMENT_ROOT.glob("*.json"), reverse=True) if EXPERIMENT_ROOT.is_dir() else []
    items = []
    for path in files[offset : offset + limit]:
        manifest = read_json(path, {})
        if manifest:
            items.append({"file": path.name, **manifest})
    return {"experiments": items, "total": len(files), "offset": offset, "limit": limit, "has_more": offset + len(items) < len(files), "root": str(EXPERIMENT_ROOT)}


@app.get("/api/experiments/{experiment_id}/download")
def download_experiment(experiment_id: str) -> FileResponse:
    if not experiment_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in experiment_id):
        raise HTTPException(400, "Invalid experiment ID")
    path = EXPERIMENT_ROOT / f"{experiment_id}.json"
    if not path.is_file():
        raise HTTPException(404, "Experiment manifest not found")
    return FileResponse(path, filename=path.name, media_type="application/json")


@app.post("/api/worker/{command}")
def worker_command(command: str) -> dict[str, Any]:
    ensure_worker_available("controlling the workstation")
    paths = {
        "reset": "/api/reset",
        "record-start": "/api/record/start",
        "record-stop": "/api/record/stop",
        "replay-last": "/api/replay-last",
        "stop": "/api/stop",
        "expert-start": "/api/expert/start",
        "expert-pause": "/api/expert/pause",
        "expert-resume": "/api/expert/resume",
        "expert-take-control": "/api/expert/take-control",
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
    if request.backend != "rsl_rl":
        raise HTTPException(400, "The validated Cartesian training backbone currently uses RSL-RL")
    item = TASKS_BY_ID.get(request.task)
    if item is None or item["play"] or item["variant"] != "ik-rel":
        raise HTTPException(400, "Training requires a registered non-play Cartesian IK-relative task")
    if request.num_envs not in range(8, 129):
        raise HTTPException(400, "Starter labs support 8 to 128 parallel environments")
    if request.max_iterations not in range(1, 201):
        raise HTTPException(400, "Starter labs support 1 to 200 iterations")
    if matrix_payload()["status"] in {"preparing", "running"}:
        raise HTTPException(409, "Finish the Failure Lab matrix before starting policy training")
    with state.lock:
        if state.switching:
            raise HTTPException(409, f"The operating room is already loading {state.requested_task}")
        if state.training_process is not None and state.training_process.poll() is None:
            raise HTTPException(409, "A training lab is already running")
        if state.healthcare_status in {"preparing", "running", "stopping"}:
            raise HTTPException(409, "Stop the native healthcare workflow before starting policy training")
    current = worker_status()
    resume_context = capture_worker_context(current, request.resume_workstation)
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = TRAINING_ROOT / f"{stamp}_{request.backend}_{item['slug']}.log"
    experiment_id = f"training_{stamp}_{request.backend}_{item['slug']}"
    manifest_path = EXPERIMENT_ROOT / f"{experiment_id}.json"
    command = [
        str(args.root / "dr_anmar_train.sh"),
        request.backend,
        request.task,
        "--num_envs",
        str(request.num_envs),
        "--max_iterations",
        str(request.max_iterations),
    ]
    write_json(
        manifest_path,
        {
            "schema": "dr.anmar.experiment.v1",
            "experiment_id": experiment_id,
            "kind": "policy_training",
            "simulation_only": True,
            "created_at": utc_now(),
            "source_revision": source_revision(),
            "runtime_provenance": (current or {}).get("runtime_provenance", {}),
            "task": request.task,
            "configuration": {
                "backend": request.backend,
                "num_envs": request.num_envs,
                "max_iterations": request.max_iterations,
                "resume_workstation": request.resume_workstation,
                "resume_context": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in (resume_context or {}).items()
                },
            },
            "command": command,
            "log": str(log_path),
            "status": "starting",
        },
    )
    with state.lock:
        if state.switching:
            raise HTTPException(409, f"The operating room is already loading {state.requested_task}")
        if state.healthcare_status in {"preparing", "running", "stopping"}:
            raise HTTPException(409, "Stop the native healthcare workflow before starting policy training")
        if state.matrix_status in {"preparing", "running"}:
            raise HTTPException(409, "Finish the Failure Lab matrix before starting policy training")
        if state.training_process is not None and state.training_process.poll() is None:
            raise HTTPException(409, "A training lab is already running")
        state.training_status = "preparing"
        state.training_task = request.task
        state.training_backend = request.backend
        state.training_log = str(log_path)
        state.training_started_at = None
        state.training_exit_code = None
        state.training_manifest = str(manifest_path)
    log_file = log_path.open("ab", buffering=0)
    try:
        if current:
            stop_interactive_worker()
        process = subprocess.Popen(
            command,
            cwd=args.root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:
        log_file.close()
        with state.lock:
            state.training_status = "failed"
            state.training_exit_code = -1
        manifest = read_json(manifest_path, {})
        manifest.update({"status": "failed", "error": str(exc), "finished_at": utc_now()})
        write_json(manifest_path, manifest)
        if resume_context:
            threading.Thread(target=resume_worker, args=(resume_context,), daemon=True, name="dr-anmar-resume").start()
        raise HTTPException(500, f"The training lab could not start: {exc}") from exc
    with state.lock:
        state.training_process = process
        state.training_status = "running"
        state.training_started_at = utc_now()
    manifest = read_json(manifest_path, {})
    manifest.update({"status": "running", "pid": process.pid, "started_at": state.training_started_at})
    write_json(manifest_path, manifest)
    threading.Thread(
        target=monitor_training,
        args=(process, log_file, resume_context),
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
            "worker_port": args.worker_port,
            "catalog_tasks": len(CATALOG),
            "interactive_tasks": len(PRIMARY_TASKS),
            "procedure_rooms": len(PROCEDURE_ROOMS),
            "anatomy": anatomy_payload(),
            "training": training_payload(),
            "healthcare_job": healthcare_job_payload(),
            "challenge_matrix": matrix_payload(),
            "operator_lease": operator_lease.status(),
            "access_control_enabled": configured_access_token() is not None,
        }
    )
    return JSONResponse(hub)


@app.get("/api/health/runtime")
def runtime_health() -> JSONResponse:
    try:
        open_descriptors = len(list(Path("/proc/self/fd").iterdir()))
    except OSError:
        open_descriptors = None
    try:
        worker = worker_json("/api/health/runtime")
    except (HTTPException, OSError, ValueError):
        worker = None
    with access_attempts_lock:
        access_clients = len(access_attempts)
        access_entries = sum(len(attempts) for attempts in access_attempts.values())
    with state.lock:
        payload = {
            "schema": "dr.anmar.hub-runtime-health.v1",
            "rss_bytes": process_rss_bytes(),
            "thread_count": threading.active_count(),
            "open_descriptors": open_descriptors,
            "switching": state.switching,
            "training_process_active": state.training_process is not None,
            "healthcare_process_active": state.healthcare_process is not None,
            "matrix_results_retained": len(state.matrix_results),
            "access_attempt_clients": access_clients,
            "access_attempt_entries": access_entries,
            "sha256_cache": cached_sha256_file.cache_info()._asdict(),
            "anatomy_cache": installed_usd_inventory.cache_info()._asdict(),
            "worker": worker,
        }
    return JSONResponse(payload)


@app.post("/api/launch")
def launch(request: LaunchRequest) -> dict[str, Any]:
    item = TASKS_BY_ID.get(request.task)
    if item is None:
        raise HTTPException(404, "Unknown ORBIT-Surgical task")
    if not item["browser_control"] or item["play"]:
        raise HTTPException(409, "Use the relative-IK non-play variant for the interactive workstation")
    ensure_worker_available("loading the Dr.Anmar operating room")
    # Task-only launches intentionally do not guess among several procedure rooms
    # that may share the same registered Isaac task.
    selected_anatomy = anatomy_room(ANATOMY_SCENES[0]["archive"].removesuffix(".zip"))
    anatomy_scene = anatomy_asset(selected_anatomy) if selected_anatomy else None
    openusd_environment = openusd_environment_asset(selected_anatomy) if selected_anatomy else None
    procedure_id = ""
    current = worker_status()
    if (
        current
        and current.get("task") == request.task
        and current.get("procedure", {}).get("id", "") == procedure_id
        and current.get("frame_id", 0) > 0
    ):
        return {"ok": True, "task": request.task, "already_ready": True}
    reserve_worker_switch(request.task)
    threading.Thread(
        target=switch_worker,
        args=(
            request.task,
            procedure_id,
            anatomy_scene,
            selected_anatomy["id"] if selected_anatomy else "",
            selected_anatomy["title"] if selected_anatomy else "",
            openusd_environment,
        ),
        daemon=True,
        name="dr-anmar-switch",
    ).start()
    return {"ok": True, "task": request.task}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)
