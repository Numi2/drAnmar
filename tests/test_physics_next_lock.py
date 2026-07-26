import json
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
    assert "python-freeze.txt" in source
    assert '"${env_root}/bin/python" -m pip check' in source
    assert 'shasum -a 256 "${next_root}/runtime.json"' in source


def test_runtime_package_lock_matches_the_supported_isaac_stack():
    lock = json.loads((ROOT / "config/physics-next-lock.json").read_text(encoding="utf-8"))
    assert lock["runtime_packages"] == {
        "isaacsim": "6.0.1.0",
        "torch": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
        "torchaudio": "2.11.0+cu128",
    }
