import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_public_release.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dranmar_public_release", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_repository_assets_are_source_not_runtime_data():
    module = _load_module()
    assert not module.is_forbidden_runtime_path("assets/dr_anmar")
    assert not module.is_forbidden_runtime_path("assets/dr_anmar/nvidia_needle_suture/scene.usda")
    assert module.is_forbidden_runtime_path("assets/downloaded/archive.zip")
    assert module.is_forbidden_runtime_path("run/session.json")


def test_public_gate_ignores_external_runtime_root(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "private-session.json").write_text(
        '{"token":"hf_not_a_real_but_intentionally_private_runtime_value"}\n',
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["DR_ANMAR_ROOT"] = str(runtime_root)
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
