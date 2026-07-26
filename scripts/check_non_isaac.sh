#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"
VALIDATION_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"

uv run python -m compileall -q \
    scripts \
    tests \
    source/extensions/orbit.surgical.assets/orbit/surgical/assets
uv run ruff check \
    scripts/check_public_release.py \
    scripts/dr_anmar_asset_registry.py \
    scripts/dr_anmar_multimodal_assets.py \
    scripts/dr_anmar_operator.py \
    scripts/dr_anmar_telemetry.py \
    scripts/generate_dranmar_evidence_index.py \
    scripts/generate_dranmar_multimodal_fixture.py \
    scripts/install_sufia_assets.py \
    scripts/localize_openusd_materials.py \
    scripts/validate_dranmar_dynamic_abdominal_patient.py \
    scripts/validate_openusd_dependencies.py \
    scripts/verify_dranmar_physics_next_receipt.py \
    tests/test_dranmar_operator_security.py \
    tests/test_dranmar_telemetry.py \
    tests/test_install_sufia_assets.py \
    tests/test_multimodal_assets.py \
    tests/test_openusd_dependency_validation.py \
    tests/test_physics_next_lock.py \
    tests/test_public_release_policy.py
uv run pyright --pythonpath "${VALIDATION_PYTHON}" \
    scripts/check_public_release.py \
    scripts/dr_anmar_multimodal_assets.py \
    scripts/dr_anmar_operator.py \
    scripts/dr_anmar_telemetry.py \
    scripts/generate_dranmar_evidence_index.py \
    scripts/generate_dranmar_multimodal_fixture.py \
    scripts/validate_openusd_dependencies.py \
    scripts/verify_dranmar_physics_next_receipt.py
uv run python scripts/check_public_release.py
uv run python scripts/validate_openusd_layers.py
uv run python scripts/validate_openusd_dependencies.py
uv run python scripts/localize_openusd_materials.py --check
uv run python scripts/generate_dranmar_dynamic_abdominal_patient_rigid_proxy.py --check
uv run python scripts/refresh_dynamic_patient_asset_manifest.py --check
uv run python scripts/generate_dranmar_multimodal_fixture.py --check
uv run python scripts/generate_dranmar_evidence_index.py --check
uv run python scripts/dr_anmar_multimodal_assets.py
uv run python scripts/dr_anmar_asset_registry.py verify
uv run python -m pytest -q tests
node --test tests/hand_control.test.mjs
