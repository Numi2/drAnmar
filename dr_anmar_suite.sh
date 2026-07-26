#!/usr/bin/env bash
set -euo pipefail

ORBIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${ORBIT_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ORBIT_ROOT}/.env"
    set +a
fi

ROOT="${DR_ANMAR_ROOT:-${HOME}/.local/share/dr-anmar}"
export DR_ANMAR_ROOT="${ROOT}"
export DR_ANMAR_I4H_ROOT="${DR_ANMAR_I4H_ROOT:-${ROOT}/vendor/i4h-workflows-current}"
export DR_ANMAR_I4H_ASSET_CATALOG_ROOT="${DR_ANMAR_I4H_ASSET_CATALOG_ROOT:-${ROOT}/vendor/i4h-asset-catalog-current}"
export I4H_ASSET_DOWNLOAD_DIR="${I4H_ASSET_DOWNLOAD_DIR:-${ROOT}/assets/i4h-catalog}"
if [[ -d "${HOME}/.local/bin" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT}/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${ROOT}/runtime/uv-python}"
PYTHON="${ISAAC_PYTHON:-python3}"
HUB_PORT="${DR_ANMAR_HUB_PORT:-2360}"
WORKER_PORT="${DR_ANMAR_WORKER_PORT:-2361}"
HUB_HOST="${DR_ANMAR_HUB_HOST:-127.0.0.1}"
PID_FILE="${ROOT}/run/hub.pid"
LOG_FILE="${ROOT}/logs/hub.log"

mkdir -p "${ROOT}/run" "${ROOT}/logs"

hub_running() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid command
    pid="$(cat "${PID_FILE}")"
    kill -0 "${pid}" 2>/dev/null || return 1
    if [[ -r "/proc/${pid}/cmdline" ]]; then
        command="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
    else
        command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
    fi
    [[ "${command}" == *"dr_anmar_hub.py"* ]]
}

case "${1:-status}" in
    start)
        if hub_running && "${ORBIT_ROOT}/dr_anmar_workstation.sh" status "${WORKER_PORT}" >/dev/null 2>&1; then
            echo "Dr.Anmar suite is already ready: http://localhost:${HUB_PORT}/"
            exit 0
        fi
        if [[ "${DR_ANMAR_REBUILD_OPENUSD:-0}" == "1" ]] || \
            ! "${PYTHON}" "${ORBIT_ROOT}/scripts/dr_anmar_openusd_preflight.py" --root "${ROOT}"; then
            echo "Preparing the OpenUSD catalog (set DR_ANMAR_REBUILD_OPENUSD=1 to force this later)..."
            "${PYTHON}" "${ORBIT_ROOT}/scripts/dr_anmar_geometry_sanitize.py" \
                --headless \
                --kit_args "--portable-root ${ROOT}/isaac_portable" >/dev/null
            "${PYTHON}" "${ORBIT_ROOT}/scripts/dr_anmar_openusd.py" >/dev/null
        fi
        if ! hub_running; then
            cd "${ORBIT_ROOT}"
            nohup "${PYTHON}" scripts/dr_anmar_hub.py --host "${HUB_HOST}" --port "${HUB_PORT}" --worker_port "${WORKER_PORT}" --root "${ORBIT_ROOT}" >>"${LOG_FILE}" 2>&1 &
            echo "$!" >"${PID_FILE}"
        fi
        if ! "${ORBIT_ROOT}/dr_anmar_workstation.sh" status "${WORKER_PORT}" >/dev/null 2>&1; then
            "${ORBIT_ROOT}/dr_anmar_workstation.sh" start \
                "${WORKER_PORT}" \
                "Isaac-Handover-Needle-Dual-PSM-IK-Rel-v0" \
                "nvidia-native-surgical-bench" \
                "" \
                "" \
                "NVIDIA surgical dry-lab bench" \
                ""
        fi
        echo "Dr.Anmar suite starting: http://localhost:${HUB_PORT}/"
        ;;
    stop)
        "${ORBIT_ROOT}/dr_anmar_workstation.sh" stop "${WORKER_PORT}" || true
        if hub_running; then kill "$(cat "${PID_FILE}")" || true; fi
        rm -f "${PID_FILE}"
        echo "Dr.Anmar suite stopped"
        ;;
    restart)
        "$0" stop
        "$0" start
        ;;
    status)
        echo "Hub:"
        curl -fsS --max-time 2 "http://127.0.0.1:${HUB_PORT}/api/status" || true
        echo
        echo "Worker:"
        "${ORBIT_ROOT}/dr_anmar_workstation.sh" status "${WORKER_PORT}" || true
        ;;
    logs)
        tail -n 100 "${LOG_FILE}"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}" >&2
        exit 2
        ;;
esac
