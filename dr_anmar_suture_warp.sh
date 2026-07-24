#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DR_ANMAR_ROOT:-${HOME}/.local/share/dr-anmar}"
REPORT_ROOT="${DR_ANMAR_WARP_SUTURE_REPORT_ROOT:-${DATA_ROOT}/native-suture/reports}"
ISAAC_PYTHON="${DR_ANMAR_STABLE_ISAAC_PYTHON:-}"
DEVICE="${DR_ANMAR_WARP_DEVICE:-cuda:0}"

case "${1:-qualify}" in
    qualify)
        [[ -x "${ISAAC_PYTHON}" ]] || {
            echo "Set DR_ANMAR_STABLE_ISAAC_PYTHON to the Isaac Lab Python." >&2
            exit 1
        }
        mkdir -p "${REPORT_ROOT}"
        output="${REPORT_ROOT}/dr-anmar-warp-suture-$(date -u +%Y%m%dT%H%M%SZ).json"
        "${ISAAC_PYTHON}" \
            "${REPOSITORY_ROOT}/scripts/dr_anmar_warp_suture.py" \
            --profile \
            "${REPOSITORY_ROOT}/physics_next/sutures/dr-anmar-suture-4-0.json" \
            --device "${DEVICE}" \
            --output "${output}"
        echo "Warp suture qualification: ${output}"
        ;;
    *)
        echo "Usage: $0 qualify" >&2
        exit 2
        ;;
esac
