#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${REPOSITORY_ROOT}/physics_next/tissues/dr-anmar-suturable-tissue-v1.json"
SURFACE="${REPOSITORY_ROOT}/assets/dr_anmar/tissue/DrAnmarSuturableTissue.usda"
TETMESH="${REPOSITORY_ROOT}/assets/dr_anmar/tissue/DrAnmarSuturableTissue.tet.usda"
ASSET_REPORT="${REPOSITORY_ROOT}/assets/dr_anmar/tissue/DrAnmarSuturableTissue.report.json"
USD_PYTHON="${DR_ANMAR_USD_PYTHON:-${DR_ANMAR_STABLE_ISAAC_PYTHON:-}}"

case "${1:-author}" in
    author)
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_tissue_author.py" \
            --profile "${PROFILE}" \
            --output "${SURFACE}" \
            --tet-output "${TETMESH}" \
            --report "${ASSET_REPORT}"
        ;;
    rebuild)
        "$0" author
        ;;
    inspect)
        [[ -x "${USD_PYTHON}" ]] || {
            echo "A Python runtime with OpenUSD is required: ${USD_PYTHON}" >&2
            exit 1
        }
        "${USD_PYTHON}" -c \
            'from pxr import Usd; import sys; [print(path, Usd.Stage.Open(path).GetDefaultPrim(), sum(1 for _ in Usd.Stage.Open(path).Traverse())) for path in sys.argv[1:]]' \
            "${SURFACE}" "${TETMESH}"
        ;;
    *)
        echo "Usage: $0 {author|rebuild|inspect}" >&2
        exit 2
        ;;
esac
