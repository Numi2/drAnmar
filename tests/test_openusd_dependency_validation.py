import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_openusd_dependencies.py"


def test_every_repository_usd_dependency_resolves_natively():
    spec = importlib.util.spec_from_file_location("dranmar_openusd_dependencies", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.validate_dependencies()
    assert report["passed"], report["issues"]
    assert report["layer_count"] >= 170
