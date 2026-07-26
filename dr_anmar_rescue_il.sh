#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${REPOSITORY_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPOSITORY_ROOT}/.env"
    set +a
fi

PYTHON="${ISAAC_PYTHON:-python3}"
DATA_ROOT="${DR_ANMAR_ROOT:-${HOME}/.local/share/dr-anmar}"
TASK="Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0"
PROCEDURE="dr-anmar-autonomous-rescue-or"
CONFIG="${REPOSITORY_ROOT}/source/standalone/workflows/robomimic/config/autonomous_rescue_bc.json"

command_name="${1:-}"
case "${command_name}" in
    room)
        port="${2:-2361}"
        exec "${REPOSITORY_ROOT}/dr_anmar_workstation.sh" \
            start "${port}" "${TASK}" "${PROCEDURE}"
        ;;
    expert)
        port="${2:-2361}"
        curl --fail --silent --show-error \
            --request POST \
            "http://127.0.0.1:${port}/api/expert/start"
        printf '\n'
        ;;
    collect)
        port="${2:-2361}"
        episodes="${3:-10}"
        shift "$(( $# >= 3 ? 3 : $# ))"
        exec "${PYTHON}" \
            "${REPOSITORY_ROOT}/scripts/collect_autonomous_rescue_experts.py" \
            --url "http://127.0.0.1:${port}" \
            --episodes "${episodes}" \
            "$@"
        ;;
    pack)
        output="${2:?An output HDF5 path is required}"
        shift 2
        if [[ "$#" -lt 1 ]]; then
            echo "At least one rescue episode HDF5 path is required" >&2
            exit 2
        fi
        exec "${PYTHON}" \
            "${REPOSITORY_ROOT}/source/standalone/workflows/robomimic/prepare_rescue_dataset.py" \
            --output "${output}" \
            "$@"
        ;;
    train)
        dataset="${2:?A merged rescue HDF5 dataset is required}"
        shift 2
        cd "${REPOSITORY_ROOT}"
        exec "${PYTHON}" \
            source/standalone/workflows/robomimic/train.py \
            --config "${CONFIG}" \
            --dataset "${dataset}" \
            "$@"
        ;;
    episodes)
        demo_dir="${2:-${DATA_ROOT}/demos}"
        find "${demo_dir}" -maxdepth 1 -type f -name '*.hdf5' -print | sort
        ;;
    *)
        cat >&2 <<'EOF'
Usage:
  dr_anmar_rescue_il.sh room [PORT]
  dr_anmar_rescue_il.sh expert [PORT]
  dr_anmar_rescue_il.sh collect [PORT] [EPISODES] [collector arguments...]
  dr_anmar_rescue_il.sh episodes [DEMO_DIR]
  dr_anmar_rescue_il.sh pack OUTPUT.hdf5 EPISODE.hdf5...
  dr_anmar_rescue_il.sh train DATASET.hdf5 [Robomimic arguments...]

The expert command starts a reset, records the contact-driven rescue episode,
and saves both the workstation NPZ and transition-aligned HDF5 training file.
EOF
        exit 2
        ;;
esac
