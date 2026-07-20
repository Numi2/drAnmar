#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Extract Doctor Studio's inline JavaScript and check it with Node."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    html = (ROOT / "web/doctor_studio.html").read_text(encoding="utf-8")
    start = html.rfind("<script>")
    end = html.find("</script>", start)
    if start < 0 or end < 0:
        raise SystemExit("Doctor Studio inline script was not found")
    source = html[start + len("<script>") : end]
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as stream:
        stream.write(source)
        stream.flush()
        result = subprocess.run(["node", "--check", stream.name], check=False)
    if result.returncode == 0:
        print("Doctor Studio JavaScript syntax passed")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
