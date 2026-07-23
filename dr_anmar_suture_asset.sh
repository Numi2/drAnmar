#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${REPOSITORY_ROOT}/physics_next/sutures/dr-anmar-suture-4-0.json"
NEEDLE_PROFILE="${REPOSITORY_ROOT}/physics_next/needles/dr-anmar-needle-v1.json"
ASSET="${REPOSITORY_ROOT}/assets/dr_anmar/suture/DrAnmarSuture4_0.usda"
DR_ANMAR_NEEDLE="${REPOSITORY_ROOT}/assets/dr_anmar/needle/DrAnmarNeedle.usda"
REPORT="${REPOSITORY_ROOT}/physics_next/benchmarks/dr-anmar-suture-4-0-validation.json"
ISAAC_PYTHON="${DR_ANMAR_STABLE_ISAAC_PYTHON:-}"

case "${1:-validate}" in
    author)
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_suture_author.py" \
            --profile "${PROFILE}" \
            --needle-profile "${NEEDLE_PROFILE}" \
            --output "${ASSET}" \
            --needle-output "${DR_ANMAR_NEEDLE}"
        ;;
    validate)
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_suture_validate.py" \
            --profile "${PROFILE}" \
            --needle-profile "${NEEDLE_PROFILE}" \
            --asset "${ASSET}" \
            --needle "${DR_ANMAR_NEEDLE}" \
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
        output="${REPOSITORY_ROOT}/physics_next/benchmarks/dr-anmar-suture-4-0-physx.json"
        "${ISAAC_PYTHON}" "${REPOSITORY_ROOT}/scripts/dr_anmar_suture_physics_probe.py" \
            --asset "${DR_ANMAR_NEEDLE}" \
            --output "${output}" \
            --device cuda:0 \
            --headless
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
