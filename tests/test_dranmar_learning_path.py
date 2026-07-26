from __future__ import annotations

import ast
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "source/extensions/orbit.surgical.tasks/orbit/surgical/tasks"


def test_learning_path_manifest_is_ordered_and_branded() -> None:
    manifest = json.loads((ROOT / "config/dranmar_learning_path.json").read_text())
    stages = manifest["stages"]
    assert [stage["stage"] for stage in stages] == list(range(1, 7))
    assert stages[0]["task"] == "DrAnmar-Reach-PSM-IK-Rel-v0"
    assert all(stage["task"].startswith("DrAnmar-") for stage in stages)
    assert manifest["defaults"]["held_out_seeds"]
    assert manifest["defaults"]["num_envs"] == 512


def test_frontier_imports_and_runner_contract() -> None:
    sources = "\n".join(path.read_text() for path in TASK_ROOT.rglob("*.py"))
    assert "from isaaclab.utils import configclass" not in sources
    assert "AdditiveUniformNoiseCfg" not in sources
    assert "RslRlPpoActorCriticCfg" not in sources
    assert "RslRlMLPModelCfg" in sources
    assert "obs_normalization=True" in sources
    assert "check_for_nan = True" in sources


def test_learning_environments_define_gpu_cloning_and_success() -> None:
    for relative in (
        "surgical/reach/reach_env_cfg.py",
        "surgical/reach_dual/reach_env_cfg.py",
        "surgical/lift/lift_env_cfg.py",
        "surgical/handover/handover_env_cfg.py",
    ):
        source = (TASK_ROOT / relative).read_text()
        assert "clone_in_fabric=True" in source
        assert "success_rate = RewTerm(" in source
        ast.parse(source)


def test_launcher_starts_simulator_before_task_registration() -> None:
    benchmark_path = ROOT / "scripts/dr_anmar_learning_benchmark.py"
    benchmark_source = benchmark_path.read_text()
    tree = ast.parse(benchmark_source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    rendered = ast.unparse(main)
    assert rendered.index("app = AppLauncher(") < rendered.rindex(
        "import orbit.surgical.tasks"
    )
    assert "export_policy_to_jit" in benchmark_source
    assert "export_policy_to_onnx" in benchmark_source


def test_launcher_fits_parallel_worlds_to_live_ram_and_vram() -> None:
    scope = runpy.run_path(str(ROOT / "scripts/dr_anmar_learning_benchmark.py"))
    fit = scope["_fit_num_envs_to_memory"]
    assert fit(512, 12_803, 11_613) == 256
    assert fit(512, 17_000, 30_000) == 512
    assert fit(1_024, 24_000, 8_500) == 128
