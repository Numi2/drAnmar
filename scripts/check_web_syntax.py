#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Check Doctor Studio, workstation, and hand-control JavaScript with Node."""

from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def workstation_html() -> str:
    tree = ast.parse((ROOT / "scripts/dr_anmar_workstation.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if any(isinstance(target, ast.Name) and target.id == "APP_HTML" for target in node.targets):
            return str(node.value.value)
    raise RuntimeError("APP_HTML was not found")


def node_check(source: str, label: str, suffix: str = ".js") -> int:
    with tempfile.NamedTemporaryFile("w", suffix=suffix, encoding="utf-8") as stream:
        stream.write(source)
        stream.flush()
        result = subprocess.run(["node", "--check", stream.name], check=False)
    if result.returncode == 0:
        print(f"{label} JavaScript syntax passed")
    return result.returncode


def main() -> int:
    pages = {
        "Doctor Studio": (ROOT / "web/doctor_studio.html").read_text(encoding="utf-8"),
        "Surgical workstation": workstation_html(),
    }
    return_code = 0
    for label, html in pages.items():
        scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
        if not scripts:
            raise SystemExit(f"{label} inline script was not found")
        for index, source in enumerate(scripts, start=1):
            return_code |= node_check(source, f"{label} inline script {index}")
    hand_control = (ROOT / "web/hand_control.mjs").read_text(encoding="utf-8")
    return_code |= node_check(hand_control, "Webcam hand control", suffix=".mjs")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
