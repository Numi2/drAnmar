#!/usr/bin/env python3
# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Fail when a source snapshot contains private-machine or generated data."""

from __future__ import annotations

import re
import queue
import subprocess
import sys
import threading
import time
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
CANONICAL_SOURCE_PREFIXES = ("assets/dr_anmar",)
FORBIDDEN = (
    ("Gilgamesh user path", re.compile(r"/home/(?:numi|gilgamesh)(?:/|\b)")),
    ("private Gilgamesh address", re.compile(r"\b100[.]98[.]17[.]98\b")),
    ("Hugging Face access token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("private key", re.compile(r"BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY")),
)
MAX_GIT_FILE = 95 * 1024 * 1024
MAX_TEXT_FILE = 8 * 1024 * 1024


def candidate_files():
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            timeout=15,
        )
        for raw in result.stdout.split(b"\0"):
            if not raw:
                continue
            path = ROOT / raw.decode("utf-8", errors="surrogateescape")
            if path.is_file():
                yield path
        return
    except (OSError, subprocess.SubprocessError):
        pass
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
            yield path


def scan_text(content: str, relative: str, problems: list[str]) -> None:
    for label, pattern in FORBIDDEN:
        if pattern.search(content):
            problems.append(f"{label} found in {relative}")


def is_forbidden_runtime_path(relative: str) -> bool:
    """Distinguish committed catalog source from external/generated runtime data."""
    for source_prefix in CANONICAL_SOURCE_PREFIXES:
        if relative == source_prefix or relative.startswith(f"{source_prefix}/"):
            return False
    return any(
        relative == runtime_dir or relative.startswith(f"{runtime_dir}/")
        for runtime_dir in RUNTIME_DIRS
    )


def read_texts_bounded(paths: list[Path], timeout: float = 12.0) -> tuple[dict[Path, str], list[str]]:
    """Read in parallel so cloud-evicted files cannot serialize into a frozen release gate."""
    outputs: dict[Path, bytes] = {}
    errors: list[str] = []
    pending: queue.Queue[Path] = queue.Queue()
    eligible: list[Path] = []

    def collect() -> None:
        while True:
            try:
                path = pending.get_nowait()
            except queue.Empty:
                return
            try:
                outputs[path] = path.read_bytes()
            except OSError:
                pass
            finally:
                pending.task_done()

    for path in paths:
        if path.stat().st_size > MAX_TEXT_FILE:
            errors.append(f"text file exceeds bounded security-scan size: {path.relative_to(ROOT)}")
            continue
        eligible.append(path)
        pending.put(path)

    deadline = time.monotonic() + timeout
    workers = [threading.Thread(target=collect, daemon=True) for _ in range(min(128, len(eligible)))]
    for worker in workers:
        worker.start()
    while pending.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.05)
    for path in eligible:
        if path not in outputs:
            errors.append(f"could not read {path.relative_to(ROOT)} within {timeout:.0f} seconds")

    decoded: dict[Path, str] = {}
    for path, content in list(outputs.items()):
        try:
            decoded[path] = content.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return decoded, errors


def main() -> int:
    problems: list[str] = []
    candidates = list(candidate_files())
    relative_candidates = {path.relative_to(ROOT).as_posix() for path in candidates}
    text_paths = [path for path in candidates if path.suffix.lower() in TEXT_SUFFIXES]
    text_content, read_errors = read_texts_bounded(text_paths)
    problems.extend(read_errors)

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            problems.append(f"missing required public file: {relative}")

    forbidden_runtime_roots = {
        item.split("/", 1)[0]
        for item in relative_candidates
        if is_forbidden_runtime_path(item)
    }
    for relative in sorted(forbidden_runtime_roots):
        problems.append(f"generated/runtime directory must not be committed: {relative}/")

    for path in candidates:
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_GIT_FILE:
            problems.append(f"file approaches Git hosting size limit: {relative}")
        if path not in text_content:
            continue
        scan_text(text_content[path], relative.as_posix(), problems)

    if problems:
        print("Public-release check failed:", file=sys.stderr)
        for problem in sorted(set(problems)):
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("Public-release check passed: no private paths, credentials, or generated data found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
