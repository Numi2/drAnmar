# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Low-overhead browser viewer for official SuFIA-BC operating-room presets."""

from __future__ import annotations

import argparse
import asyncio
import io
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from dr_anmar_operator import host_is_loopback


parser = argparse.ArgumentParser(description="View an official Dr.Anmar anatomy operating room.")
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--room_id", required=True)
parser.add_argument("--room_title", required=True)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=2361)
parser.add_argument("--camera_width", type=int, default=960)
parser.add_argument("--camera_height", type=int, default=640)
args = parser.parse_args()
if not host_is_loopback(args.host):
    parser.error("The read-only anatomy worker must remain bound to loopback")


APP_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dr.Anmar Anatomy Operating Room</title><style>
:root{color-scheme:dark;--bg:#030b10;--line:#24404d;--cyan:#2cd2e8;--text:#e9f8fa;--muted:#8aa7b1;--green:#42e49b}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,"Segoe UI";height:100vh;overflow:hidden}header{height:58px;display:flex;align-items:center;gap:14px;padding:0 18px;background:#07141c;border-bottom:1px solid var(--line)}.brand{font-weight:900;letter-spacing:.08em}.brand span{color:var(--cyan)}.mode{padding:5px 8px;border:1px solid #31515e;color:#a9c0c8;font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}.live{margin-left:auto;display:flex;align-items:center;gap:7px;color:var(--muted);font-size:12px}.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 10px #42e49b88}.view{height:calc(100vh - 58px);position:relative;background:#000;display:grid;place-items:center}.view img{width:100%;height:100%;object-fit:contain}.hud{position:absolute;left:16px;top:16px;max-width:470px;padding:12px 14px;background:#041018df;border:1px solid #ffffff25;backdrop-filter:blur(7px)}.hud .eyebrow{font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--cyan);font-weight:850}.hud h1{font-size:18px;margin:6px 0 3px}.hud p{margin:0;color:#a9bec5;font-size:11px}.badge{position:absolute;right:16px;top:16px;padding:7px 10px;background:#07171fdf;border:1px solid #31515e;color:#a9c0c8;font:11px ui-monospace,monospace}
</style></head><body><header><div class="brand">DR.<span>ANMAR</span></div><div class="mode">Official organ-room preset</div><div class="live"><i class="dot"></i><span>Room ready · CPU-idle</span></div></header><main class="view"><img src="/frame.jpg" alt="Official anatomy operating-room preview"><div class="hud"><div class="eyebrow">SuFIA-BC anatomy room</div><h1 id="title">Loading room…</h1><p>Instant official stage preview · the complete OpenUSD scene remains installed for Isaac Sim research</p></div><div id="badge" class="badge">CameraOR preview</div></main><script>
fetch('/api/status',{cache:'no-store'}).then(r=>r.json()).then(s=>{document.getElementById('title').textContent=s.room_title;document.getElementById('badge').textContent=`Official preview · ${s.camera_width}×${s.camera_height} · CPU-idle`});
</script></body></html>"""


@dataclass
class ViewerState:
    room_id: str
    room_title: str
    scene: str
    preview: str
    camera_width: int
    camera_height: int
    frame_jpeg: bytes
    instance_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))

    def status(self) -> dict[str, Any]:
        return {
            "task": f"Anatomy-{self.room_id}",
            "mode": "anatomy",
            "room_id": self.room_id,
            "room_title": self.room_title,
            "scene": self.scene,
            "preview": self.preview,
            "camera": "/CameraOR",
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
            "instance_id": self.instance_id,
            "frame_id": 1,
            "render_fps": 0.0,
            "sim_fps": 0.0,
            "sim_step": 0,
            "static_frame": True,
            "preview_mode": True,
            "action_dim": 0,
            "arms": 0,
            "has_grippers": False,
            "robot_names": [],
            "grippers_open": [],
            "recording": False,
            "recorded_frames": 0,
            "last_demo": None,
            "replaying": False,
            "drive_active": False,
        }


def make_preview(source: Path, width: int, height: int) -> bytes:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    background = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    background = ImageEnhance.Brightness(background.filter(ImageFilter.GaussianBlur(18))).enhance(0.34)
    foreground = ImageOps.contain(image, (height, height), method=Image.Resampling.LANCZOS)
    x = (width - foreground.width) // 2
    y = (height - foreground.height) // 2
    background.paste(foreground, (x, y))
    buffer = io.BytesIO()
    background.save(buffer, "JPEG", quality=91, optimize=True)
    return buffer.getvalue()


def build_app(state: ViewerState) -> FastAPI:
    app = FastAPI(title="Dr.Anmar Anatomy Operating Room", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return APP_HTML

    @app.get("/api/status")
    def status() -> JSONResponse:
        return JSONResponse(state.status())

    @app.get("/api/demos")
    def demos() -> dict[str, list]:
        return {"demos": []}

    @app.post("/api/reset")
    @app.post("/api/stop")
    def no_op() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/api/drive")
    @app.post("/api/jog")
    @app.post("/api/gripper")
    def unavailable() -> None:
        raise HTTPException(409, "This preset is an anatomy inspection room. Select a robot task for tool control.")

    @app.get("/frame.jpg")
    def frame() -> Response:
        return Response(state.frame_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/video")
    async def video() -> StreamingResponse:
        async def frames():
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + state.frame_jpeg + b"\r\n"
            while True:
                await asyncio.sleep(60.0)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def main() -> None:
    scene = args.scene.resolve()
    data_root = Path(os.environ.get("DR_ANMAR_ROOT", Path.home() / ".local/share/dr-anmar")).expanduser()
    allowed_root = (data_root / "assets/sufia_bc").resolve()
    if not scene.is_file() or scene.name != "main_scene.usd" or allowed_root not in scene.parents:
        raise ValueError(f"Official anatomy scene not found: {scene}")
    preview = scene.parent / ".thumbs/256x256/main_scene.usd.png"
    if not preview.is_file():
        raise ValueError(f"Official anatomy preview not found: {preview}")
    frame = make_preview(preview, args.camera_width, args.camera_height)
    state = ViewerState(
        args.room_id,
        args.room_title,
        str(scene),
        str(preview),
        args.camera_width,
        args.camera_height,
        frame,
    )
    print(f"[DR_ANMAR_ANATOMY] {args.room_id} · instant preview · CPU-idle", flush=True)
    uvicorn.run(build_app(state), host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
