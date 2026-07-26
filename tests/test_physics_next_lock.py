import json
import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_physics_next_sources_are_full_commit_pins():
    lock = json.loads((ROOT / "config/physics-next-lock.json").read_text(encoding="utf-8"))
    assert lock["schema"] == "dr.anmar.physics-next-lock.v1"
    for source in lock["sources"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
        assert source["repository"].startswith("https://github.com/")
        assert source["repository"].endswith(".git")


def test_installer_consumes_lock_and_writes_hashed_receipt():
    source = (ROOT / "dr_anmar_physics_next.sh").read_text(encoding="utf-8")
    assert 'LOCK_PATH="${REPOSITORY_ROOT}/config/physics-next-lock.json"' in source
    assert "verify_dranmar_physics_next_receipt.py" in source
    assert "verify_dranmar_physics_next_environment.py" in source
    assert "python-freeze.txt" in source
    assert 'isaaclab.sh --install "${isaaclab_install_profile}"' in source
    assert '"torchaudio==${torchaudio_version}"' in source
    assert 'shasum -a 256 "${next_root}/runtime.json"' in source


def test_runtime_package_lock_matches_the_supported_isaac_stack():
    lock = json.loads((ROOT / "config/physics-next-lock.json").read_text(encoding="utf-8"))
    assert lock["runtime_packages"] == {
        "isaacsim": "6.0.1.0",
        "torch": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
        "torchaudio": "2.11.0+cu128",
    }


def test_dependency_policy_is_scoped_and_fail_closed():
    lock = json.loads((ROOT / "config/physics-next-lock.json").read_text(encoding="utf-8"))
    policy = lock["dependency_policy"]
    assert policy["isaaclab_install_profile"] == "core"
    assert policy["pytorch_index_url"] == "https://download.pytorch.org/whl/cu128"
    assert len(policy["allowed_pip_check_conflicts"]) == 6
    assert all("isaacsim-" in line for line in policy["allowed_pip_check_conflicts"])

    path = ROOT / "scripts/verify_dranmar_physics_next_environment.py"
    spec = importlib.util.spec_from_file_location("physics_next_environment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    exact = module.evaluate(
        policy["allowed_pip_check_conflicts"],
        policy["allowed_pip_check_conflicts"],
    )
    assert exact["passed"]
    unexpected = module.evaluate(
        [*policy["allowed_pip_check_conflicts"], "unexpected dependency conflict"],
        policy["allowed_pip_check_conflicts"],
    )
    assert not unexpected["passed"]
    assert unexpected["unexpected_conflicts"] == ["unexpected dependency conflict"]
    missing = module.evaluate(
        policy["allowed_pip_check_conflicts"][:-1],
        policy["allowed_pip_check_conflicts"],
    )
    assert not missing["passed"]
    assert len(missing["missing_expected_conflicts"]) == 1
