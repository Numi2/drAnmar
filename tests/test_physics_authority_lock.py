import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dr_anmar_physics_authority import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_PHYSICS_NEXT_LOCK,
    load_physics_authority,
)


def test_physics_authority_uses_the_current_exact_stack(monkeypatch, tmp_path):
    lock = json.loads(
        DEFAULT_PHYSICS_NEXT_LOCK.read_text(encoding="utf-8")
    )
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert (
        manifest["source_pins"]["isaac_lab"]["revision"]
        == lock["sources"]["isaaclab"]["revision"]
    )

    monkeypatch.setenv("DR_ANMAR_ROOT", str(tmp_path))
    payload = load_physics_authority().runtime_payload()
    assert payload["next_runtime"] == {
        "isaac_sim": lock["simulator"]["version"],
        "isaac_lab": lock["sources"]["isaaclab"]["revision"],
        "torch": lock["runtime_packages"]["torch"],
        "installation_profile": "core",
        "newton": "VBD experimental integration",
        "isolated": True,
        "ready_marker": False,
        "receipt_verified": False,
        "module_probe_current_process": {
            "isaaclab": False,
            "isaaclab_newton": False,
            "warp": False,
        },
    }
