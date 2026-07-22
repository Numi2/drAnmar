#!/usr/bin/env python3
"""Dependency-free audit of the Dr.Anmar workstation keyboard surface."""

from __future__ import annotations

import ast
import html
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSTATION = ROOT / "scripts" / "dr_anmar_workstation.py"
BUTTON_RE = re.compile(r"<button\b(?P<attrs>[^>]*)>(?P<label>.*?)</button>", re.IGNORECASE | re.DOTALL)
ATTRIBUTE_RE = re.compile(r"([\w-]+)=(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))")
TAG_RE = re.compile(r"<[^>]+>")


def app_html() -> str:
    tree = ast.parse(WORKSTATION.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if any(isinstance(target, ast.Name) and target.id == "APP_HTML" for target in node.targets):
            return str(node.value.value)
    raise RuntimeError("APP_HTML was not found")


def attributes(source: str) -> dict[str, str]:
    return {
        match.group(1): next(value for value in match.groups()[1:] if value is not None)
        for match in ATTRIBUTE_RE.finditer(source)
    }


def main() -> int:
    workstation_source = WORKSTATION.read_text()
    page = app_html()
    buttons = []
    for match in BUTTON_RE.finditer(page):
        attrs = attributes(match.group("attrs"))
        label = html.unescape(TAG_RE.sub(" ", match.group("label")))
        buttons.append((" ".join(label.split()), attrs))

    missing = [(label, attrs) for label, attrs in buttons if not attrs.get("data-shortcut", "").strip()]
    key_codes = set(re.findall(r"data-key=\"([^\"]+)\"", page))
    combo_codes = set(re.findall(r"data-combo-key=\"([^\"]+)\"", page))
    javascript = re.search(r"<script>(.*)</script>", page, re.DOTALL)
    script = javascript.group(1) if javascript else ""
    missing_key_codes = sorted(code for code in key_codes if f"{code}:[" not in script)
    missing_combo_codes = sorted(code for code in combo_codes if f"{code}:{{" not in script)

    print(f"Keyboard coverage: {len(buttons) - len(missing)}/{len(buttons)} buttons mapped")
    if missing:
        for label, _ in missing:
            print(f"  missing shortcut: {label}")
    if missing_key_codes:
        print(f"  movement keys missing from keyMap: {', '.join(missing_key_codes)}")
    if missing_combo_codes:
        print(f"  combo keys missing from comboMap: {', '.join(missing_combo_codes)}")

    required_safety = {
        "if(code==='Backspace'||code==='Escape'){if(!event.repeat)emergencyStop()": "Escape and Backspace emergency stop",
        "window.addEventListener('blur',()=>stopDrive(false))": "focus-loss stop",
        "visibilitychange": "hidden-page stop",
        "source='keyboard_smart_action'": "smart-action provenance",
        "window.addEventListener('pagehide',releasePageResources": "page-exit resource release",
        "activeFetchControllers.forEach(controller=>controller.abort())": "in-flight request cancellation",
        "if(refreshInFlight||pageDisposed||document.hidden)return": "bounded status polling",
        "heldKeys.forEach(code=>{if(comboMap[code])": "held combined-move control",
        "navigator.getGamepads": "browser gamepad polling",
        "gamepaddisconnected": "controller-disconnect emergency stop",
        "gamepadSafetyLatched": "controller neutral-before-resume safety latch",
        "window.SpeechRecognition||window.webkitSpeechRecognition": "push-to-talk speech recognition",
        "voicePulseTimer=setTimeout": "bounded voice movement pulse",
        "type commands instead": "typed voice-command fallback",
    }
    missing_safety = [label for source, label in required_safety.items() if source not in script]
    for label in missing_safety:
        print(f"  missing safety behavior: {label}")

    required_backend = {
        '"keyboard_smart_action": 6': "smart-action input-source registration",
        '"voice": 9': "voice input-source registration",
        '"gamepad_smart_action": 10': "gamepad smart-action registration",
        '"voice_smart_action": 11': "voice smart-action registration",
        "hold_seconds = max(0.30": "simulator-rate-aware command lifetime",
        "semantic_target_far": "far-target semantic travel scaling",
        "needle_entry_direction": "stable tissue entry vector",
        'camera_view_mode: str = "free"': "adjustable camera default mode",
        "camera_free_enabled: bool = True": "adjustable camera enabled by default",
    }
    missing_backend = [
        label for source, label in required_backend.items() if source not in workstation_source
    ]
    for label in missing_backend:
        print(f"  missing backend behavior: {label}")

    control_contract = {
        "KeyW:[2,1,'Up']": "left W maps to up",
        "KeyS:[2,-1,'Down']": "left S maps to down",
        "KeyA:[1,1,'Left']": "left A maps to left",
        "KeyD:[1,-1,'Right']": "left D maps to right",
        "KeyQ:[0,-1,'Toward']": "left Q maps toward patient",
        "KeyE:[0,1,'Away']": "left E maps away from patient",
        "KeyI:[2,1,'Up']": "right I maps to up",
        "KeyK:[2,-1,'Down']": "right K maps to down",
        "KeyJ:[1,1,'Left']": "right J maps to left",
        "KeyL:[1,-1,'Right']": "right L maps to right",
        "KeyU:[0,-1,'Toward']": "right U maps toward patient",
        "KeyO:[0,1,'Away']": "right O maps away from patient",
    }
    missing_contract = [label for source, label in control_contract.items() if source not in script]
    for label in missing_contract:
        print(f"  incorrect control contract: {label}")

    return 1 if missing or missing_key_codes or missing_combo_codes or missing_safety or missing_backend or missing_contract else 0


if __name__ == "__main__":
    sys.exit(main())
