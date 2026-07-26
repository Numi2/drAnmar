import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/install_sufia_assets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dranmar_sufia_installer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_assets_have_full_sha256_pins():
    module = _load_module()
    assert len(module.ASSETS) == 7
    for _name, size, digest in module.ASSETS:
        assert size > 0
        assert len(digest) == 64
        int(digest, 16)


def test_archive_verification_checks_content_not_only_size(tmp_path):
    module = _load_module()
    archive = tmp_path / "asset.zip"
    archive.write_bytes(b"same-size")
    assert not module.archive_is_verified(archive, len(b"same-size"), "0" * 64)
    assert module.archive_is_verified(
        archive,
        len(b"same-size"),
        module.sha256_file(archive),
    )


def test_safe_extract_rejects_parent_traversal(tmp_path):
    module = _load_module()
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe archive member"):
        module.safe_extract(archive, tmp_path / "destination")


def test_install_receipt_binds_extraction_to_archive_hash(tmp_path):
    module = _load_module()
    archive = tmp_path / "scene.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("main_scene.usd", "#usda 1.0\n")
    destination = tmp_path / "installed"
    digest = module.sha256_file(archive)
    module.install_archive(archive, destination, digest)
    receipt = json.loads(
        (destination / ".installed.json").read_text(encoding="utf-8")
    )
    assert receipt["archive_sha256"] == digest
    assert receipt["extracted_file_count"] == 1
