#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DR_ANMAR_ROOT:-${HOME}/dr_anmar/dr-anmar-runtime}"
PROFILE="${REPOSITORY_ROOT}/physics_next/sutures/dr-anmar-suture-4-0.json"
NEEDLE_PROFILE="${REPOSITORY_ROOT}/physics_next/needles/dr-anmar-needle-v1.json"
ASSET="${REPOSITORY_ROOT}/assets/dr_anmar/suture/DrAnmarSuture4_0.usda"
SUTURE_BASE="${REPOSITORY_ROOT}/assets/dr_anmar/suture/DrAnmarSuture4_0_base.usda"
SUTURE_GEOMETRY="${REPOSITORY_ROOT}/assets/dr_anmar/suture/DrAnmarSuture4_0_geometry.usd"
SUTURE_MATERIALS="${REPOSITORY_ROOT}/assets/dr_anmar/suture/DrAnmarSuture4_0_materials.usda"
SUTURE_TEXTURE="${REPOSITORY_ROOT}/assets/dr_anmar/suture/textures/DrAnmarSuture4_0_braid_normal_roughness.png"
SUTURE_PHYSICS="${REPOSITORY_ROOT}/assets/dr_anmar/suture/DrAnmarSuture4_0_physics.usda"
SUTURE_PHYSX="${REPOSITORY_ROOT}/assets/dr_anmar/suture/DrAnmarSuture4_0_physx.usda"
DR_ANMAR_NEEDLE="${REPOSITORY_ROOT}/assets/dr_anmar/needle/DrAnmarNeedle.usda"
DR_ANMAR_NEEDLE_BASE="${REPOSITORY_ROOT}/assets/dr_anmar/needle/DrAnmarNeedle_base.usda"
DR_ANMAR_NEEDLE_GEOMETRY="${REPOSITORY_ROOT}/assets/dr_anmar/needle/DrAnmarNeedle_geometry.usd"
DR_ANMAR_NEEDLE_MATERIALS="${REPOSITORY_ROOT}/assets/dr_anmar/needle/DrAnmarNeedle_materials.usda"
DR_ANMAR_NEEDLE_PHYSICS="${REPOSITORY_ROOT}/assets/dr_anmar/needle/DrAnmarNeedle_physics.usda"
DR_ANMAR_NEEDLE_PHYSX="${REPOSITORY_ROOT}/assets/dr_anmar/needle/DrAnmarNeedle_physx.usda"
REPORT="${REPOSITORY_ROOT}/physics_next/benchmarks/dr-anmar-suture-4-0-validation.json"
ISAAC_PYTHON="${DR_ANMAR_STABLE_ISAAC_PYTHON:-}"

case "${1:-validate}" in
    author)
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_suture_author.py" \
            --profile "${PROFILE}" \
            --needle-profile "${NEEDLE_PROFILE}" \
            --output "${ASSET}" \
            --base-output "${SUTURE_BASE}" \
            --geometry-output "${SUTURE_GEOMETRY}" \
            --materials-output "${SUTURE_MATERIALS}" \
            --texture-output "${SUTURE_TEXTURE}" \
            --physics-output "${SUTURE_PHYSICS}" \
            --physx-output "${SUTURE_PHYSX}" \
            --needle-output "${DR_ANMAR_NEEDLE}" \
            --needle-base-output "${DR_ANMAR_NEEDLE_BASE}" \
            --needle-geometry-output "${DR_ANMAR_NEEDLE_GEOMETRY}" \
            --needle-materials-output "${DR_ANMAR_NEEDLE_MATERIALS}" \
            --needle-physics-output "${DR_ANMAR_NEEDLE_PHYSICS}" \
            --needle-physx-output "${DR_ANMAR_NEEDLE_PHYSX}"
        ;;
    validate)
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_suture_validate.py" \
            --profile "${PROFILE}" \
            --needle-profile "${NEEDLE_PROFILE}" \
            --asset "${ASSET}" \
            --suture-base "${SUTURE_BASE}" \
            --suture-geometry "${SUTURE_GEOMETRY}" \
            --suture-materials "${SUTURE_MATERIALS}" \
            --suture-texture "${SUTURE_TEXTURE}" \
            --suture-physics "${SUTURE_PHYSICS}" \
            --suture-physx "${SUTURE_PHYSX}" \
            --needle "${DR_ANMAR_NEEDLE}" \
            --needle-base "${DR_ANMAR_NEEDLE_BASE}" \
            --needle-geometry "${DR_ANMAR_NEEDLE_GEOMETRY}" \
            --needle-materials "${DR_ANMAR_NEEDLE_MATERIALS}" \
            --needle-physics "${DR_ANMAR_NEEDLE_PHYSICS}" \
            --needle-physx "${DR_ANMAR_NEEDLE_PHYSX}" \
            --output "${REPORT}"
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_suture_runtime.py" \
            --profile "${PROFILE}" \
            --self-test
        ;;
    inspect)
        [[ -x "${ISAAC_PYTHON}" ]] || {
            echo "Isaac Python not found: ${ISAAC_PYTHON}" >&2
            exit 1
        }
        "${ISAAC_PYTHON}" -c \
            'from pxr import Usd; import sys; stage=Usd.Stage.Open(sys.argv[1]); print(stage.GetDefaultPrim()); print("prims=", sum(1 for _ in stage.Traverse()))' \
            "${DR_ANMAR_NEEDLE}"
        ;;
    physics-probe)
        [[ -x "${ISAAC_PYTHON}" ]] || {
            echo "Isaac Python not found: ${ISAAC_PYTHON}" >&2
            exit 1
        }
        portable_root="${DATA_ROOT}/isaac_portable-suture-asset"
        temporary_root="${DATA_ROOT}/tmp"
        mkdir -p "${portable_root}" "${temporary_root}"
        output="${REPOSITORY_ROOT}/physics_next/benchmarks/dr-anmar-suture-4-0-physx.json"
        probe_status=0
        TMPDIR="${temporary_root}" \
            "${ISAAC_PYTHON}" "${REPOSITORY_ROOT}/scripts/dr_anmar_suture_physics_probe.py" \
            --asset "${DR_ANMAR_NEEDLE}" \
            --output "${output}" \
            --device cuda:0 \
            --headless \
            --kit_args "--portable-root ${portable_root}" || probe_status=$?
        python3 -c \
            'import json,sys; report=json.load(open(sys.argv[1])); raise SystemExit(0 if report.get("passed") is True else 1)' \
            "${output}" || exit 1
        [[ "${probe_status}" -eq 0 ]] || exit "${probe_status}"
        ;;
    rebuild)
        "$0" author
        "$0" validate
        ;;
    *)
        echo "Usage: $0 {author|validate|inspect|physics-probe|rebuild}" >&2
        exit 2
        ;;
esac
