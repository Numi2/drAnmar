#!/usr/bin/env python3
"""Dependency-free browser preview for the workstation hand-control surface."""

from __future__ import annotations

import argparse
import ast
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (
    Path.home()
    / ".local/share/dr-anmar/assets/hand-control/mediapipe-tasks-vision-0.10.35"
)


def workstation_html() -> bytes:
    tree = ast.parse((ROOT / "scripts/dr_anmar_workstation.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if any(isinstance(target, ast.Name) and target.id == "APP_HTML" for target in node.targets):
            return str(node.value.value).encode()
    raise RuntimeError("APP_HTML was not found")


class PreviewHandler(BaseHTTPRequestHandler):
    server_version = "DrAnmarHandPreview/1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Permissions-Policy", "camera=(self)")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.respond(200, workstation_html(), "text/html; charset=utf-8")
            return
        if path == "/hand-control.mjs":
            self.respond(
                200,
                (ROOT / "web/hand_control.mjs").read_bytes(),
                "text/javascript; charset=utf-8",
            )
            return
        if path.startswith("/hand-control-assets/"):
            relative = path.removeprefix("/hand-control-assets/")
            asset = (ASSET_ROOT / relative).resolve()
            if ASSET_ROOT.resolve() not in asset.parents or not asset.is_file():
                self.respond(404, b"", "text/plain")
                return
            content_type = (
                "application/wasm"
                if asset.suffix == ".wasm"
                else "text/javascript"
                if asset.suffix in {".js", ".mjs"}
                else "application/octet-stream"
            )
            self.respond(200, asset.read_bytes(), content_type)
            return
        if path == "/api/hand-control/assets":
            required = [
                "vision_bundle.mjs",
                "hand_landmarker.task",
                "wasm/vision_wasm_internal.js",
                "wasm/vision_wasm_internal.wasm",
            ]
            self.respond(
                200,
                json.dumps(
                    {
                        "ready": all((ASSET_ROOT / name).is_file() for name in required),
                        "version": "0.10.35",
                    }
                ).encode(),
                "application/json",
            )
            return
        if path in {"/api/status", "/api/status/live"}:
            self.respond(
                200,
                json.dumps(
                    {
                        "instance_id": "hand-control-preview",
                        "arms": 2,
                        "has_grippers": True,
                        "camera_names": [],
                        "grippers_open": [True, True],
                        "gripper_apertures": [1.0, 1.0],
                        "hand_teleop": {
                            "enabled": False,
                            "sequence": -1,
                            "watchdog_ms": 250,
                            "arms": [],
                        },
                        "procedure": {
                            "title": "Bimanual webcam preview",
                            "objective": "Validate the hand-control interface.",
                            "progress_percent": 0,
                            "steps": [],
                        },
                        "expert_demonstration": {},
                        "autonomy_mode": "manual",
                        "coaching_cue": "Preview only.",
                        "safety": {},
                    }
                ).encode(),
                "application/json",
            )
            return
        self.respond(404, b"", "text/plain")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if path == "/api/teleop/hands/control":
            enabled = bool(body.get("enabled"))
            self.respond(
                200,
                json.dumps(
                    {
                        "ok": True,
                        "hand_teleop": {
                            "enabled": enabled,
                            "sequence": -1,
                            "watchdog_ms": 250,
                            "arms": [
                                {"arm": 0, "reacquire_unclutched": True},
                                {"arm": 1, "reacquire_unclutched": True},
                            ],
                        },
                    }
                ).encode(),
                "application/json",
            )
            return
        if path == "/api/teleop/hands":
            self.respond(
                200,
                json.dumps(
                    {
                        "ok": True,
                        "hand_teleop": {
                            "enabled": True,
                            "sequence": body.get("sequence", -1),
                            "watchdog_ms": 250,
                            "arms": [
                                {"arm": 0, "reacquire_unclutched": False},
                                {"arm": 1, "reacquire_unclutched": False},
                            ],
                        },
                    }
                ).encode(),
                "application/json",
            )
            return
        self.respond(200, b'{"ok":true}', "application/json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8236)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), PreviewHandler)
    print(f"[DR_ANMAR_HAND_PREVIEW] http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
