#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Fail when a source snapshot contains private-machine or generated data."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
REQUIRED = (
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".env.example",
)
RUNTIME_DIRS = ("assets", "downloads", "demos", "logs", "run", "state", "training", "tmp")
FORBIDDEN = (
    ("Gilgamesh user path", re.compile(r"/home/(?:numi|gilgamesh)(?:/|\b)")),
    ("private Gilgamesh address", re.compile(r"\b100[.]98[.]17[.]98\b")),
    ("Hugging Face access token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("private key", re.compile(r"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY")),
)
MAX_GIT_FILE = 95 * 1024 * 1024


def candidate_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        yield path


def main() -> int:
    problems: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            problems.append(f"missing required public file: {relative}")

    for relative in RUNTIME_DIRS:
        if (ROOT / relative).exists():
            problems.append(f"generated/runtime directory must not be committed: {relative}/")

    for path in candidate_files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_GIT_FILE:
            problems.append(f"file approaches Git hosting size limit: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in FORBIDDEN:
            if pattern.search(content):
                problems.append(f"{label} found in {relative}")

    if problems:
        print("Public-release check failed:", file=sys.stderr)
        for problem in sorted(set(problems)):
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("Public-release check passed: no private paths, credentials, or generated data found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
