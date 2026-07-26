from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = (
    ROOT
    / "physics_next/benchmarks/dranmar-portfolio-evidence-index.json"
)


def test_portfolio_evidence_index_is_complete_and_content_addressed() -> None:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    assert index["schema"] == "dr.anmar.portfolio-evidence-index.v1"
    assert index["asset_count"] == len(index["assets"]) == 21
    assert len(index["source"]["parent_revision"]) == 40
    assert len(index["source"]["asset_submodule_revision"]) == 40

    for asset in index["assets"]:
        assert asset["artifact_count"] == len(asset["artifacts"])
        assert asset["all_declared_artifacts_content_addressed"]
        assert asset["claims"]["clinical_validation"] is False
        for artifact in asset["artifacts"]:
            path = ROOT / artifact["path"]
            assert path.is_file()
            assert path.stat().st_size == artifact["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
            assert len(artifact["containing_revision"]) == 40
