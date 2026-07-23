#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${REPOSITORY_ROOT}/physics_next/sutures/dr-anmar-suture-4-0.json"
ASSET="${REPOSITORY_ROOT}/source/extensions/orbit.surgical.assets/data/Props/DrAnmarSuture/DrAnmarSuture4_0.usda"
ASSEMBLY="${REPOSITORY_ROOT}/source/extensions/orbit.surgical.assets/data/Props/DrAnmarSuture/DrAnmarNeedleSuture4_0.usda"
REPORT="${REPOSITORY_ROOT}/physics_next/benchmarks/dr-anmar-suture-4-0-validation.json"
ISAAC_PYTHON="${DR_ANMAR_STABLE_ISAAC_PYTHON:-/home/gilgamesh/isaaclab_pip/env_isaaclab/bin/python}"

case "${1:-validate}" in
    author)
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_suture_author.py" \
            --profile "${PROFILE}" \
            --output "${ASSET}" \
            --assembly-output "${ASSEMBLY}"
        ;;
    validate)
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_suture_validate.py" \
            --profile "${PROFILE}" \
            --asset "${ASSET}" \
            --assembly "${ASSEMBLY}" \
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
            "${ASSEMBLY}"
        ;;
    physics-probe)
        [[ -x "${ISAAC_PYTHON}" ]] || {
            echo "Isaac Python not found: ${ISAAC_PYTHON}" >&2
            exit 1
        }
        output="${REPOSITORY_ROOT}/physics_next/benchmarks/dr-anmar-suture-4-0-physx.json"
        "${ISAAC_PYTHON}" "${REPOSITORY_ROOT}/scripts/dr_anmar_suture_physics_probe.py" \
            --asset "${ASSEMBLY}" \
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
