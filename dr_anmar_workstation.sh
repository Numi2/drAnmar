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
PORT="${2:-2361}"
TASK="${3:-Isaac-Lift-Needle-PSM-IK-Rel-v0}"
PROCEDURE="${4:-}"
ANATOMY_SCENE="${5:-}"
ANATOMY_SCENE_ID="${6:-}"
ANATOMY_TITLE="${7:-}"
OPENUSD_ENVIRONMENT="${8:-}"
PID_FILE="${ROOT}/run/workstation.pid"
LOG_FILE="${ROOT}/logs/workstation.log"

mkdir -p "${ROOT}/run" "${ROOT}/logs" "${ROOT}/demos"

is_running() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid command
    pid="$(cat "${PID_FILE}")"
    kill -0 "${pid}" 2>/dev/null || return 1
    if [[ -r "/proc/${pid}/cmdline" ]]; then
        command="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
    else
        command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
    fi
    [[ "${command}" == *"dr_anmar"* ]]
}

case "${1:-status}" in
    start)
        if is_running; then
            echo "Dr.Anmar workstation is already running (PID $(cat "${PID_FILE}"))"
            exit 0
        fi
        rm -f "${PID_FILE}"
        cd "${ORBIT_ROOT}"
        nohup ./dr_anmar.sh workstation "${PORT}" "${TASK}" "${PROCEDURE}" "${ANATOMY_SCENE}" "${ANATOMY_SCENE_ID}" "${ANATOMY_TITLE}" "${OPENUSD_ENVIRONMENT}" >>"${LOG_FILE}" 2>&1 &
        echo "$!" >"${PID_FILE}"
        echo "Starting Dr.Anmar workstation on port ${PORT}: ${TASK} (PID $!)"
        echo "Log: ${LOG_FILE}"
        ;;
    start-anatomy)
        room_id="${3:?A room id is required}"
        scene="${4:?An official main_scene.usd path is required}"
        room_title="${5:?A room title is required}"
        if is_running; then
            echo "Dr.Anmar worker is already running (PID $(cat "${PID_FILE}"))"
            exit 0
        fi
        rm -f "${PID_FILE}"
        cd "${ORBIT_ROOT}"
        nohup ./dr_anmar.sh anatomy-viewer "${PORT}" "${scene}" "${room_id}" "${room_title}" >>"${LOG_FILE}" 2>&1 &
        echo "$!" >"${PID_FILE}"
        echo "Starting Dr.Anmar anatomy room on port ${PORT}: ${room_title} (PID $!)"
        echo "Log: ${LOG_FILE}"
        ;;
    stop)
        if ! is_running; then
            rm -f "${PID_FILE}"
            echo "Dr.Anmar workstation is not running"
            exit 0
        fi
        pid="$(cat "${PID_FILE}")"
        kill "${pid}"
        for _ in $(seq 1 20); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}"
        fi
        rm -f "${PID_FILE}"
        echo "Dr.Anmar workstation stopped"
        ;;
    restart)
        "$0" stop || true
        "$0" start "${PORT}" "${TASK}" "${PROCEDURE}" "${ANATOMY_SCENE}" "${ANATOMY_SCENE_ID}" "${ANATOMY_TITLE}" "${OPENUSD_ENVIRONMENT}"
        ;;
    restart-anatomy)
        room_id="${3:?A room id is required}"
        scene="${4:?An official main_scene.usd path is required}"
        room_title="${5:?A room title is required}"
        "$0" stop || true
        "$0" start-anatomy "${PORT}" "${room_id}" "${scene}" "${room_title}"
        ;;
    status)
        if is_running; then
            echo "Dr.Anmar workstation is running (PID $(cat "${PID_FILE}"))"
            curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/api/status" || true
            echo
        else
            echo "Dr.Anmar workstation is stopped"
            exit 1
        fi
        ;;
    logs)
        tail -n 120 "${LOG_FILE}"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|start-anatomy|restart-anatomy|status|logs} [PORT] [TASK_OR_ROOM] [PROCEDURE_OR_SCENE] [ANATOMY_SCENE_OR_TITLE] [ANATOMY_ID] [ANATOMY_TITLE]" >&2
        exit 2
        ;;
esac
