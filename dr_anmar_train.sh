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
PYTHON="${ISAAC_PYTHON:-python3}"
PORTABLE_ROOT="${DR_ANMAR_PORTABLE_ROOT:-${ROOT}/isaac_portable}"

backend="${1:-}"
task="${2:-}"
if [[ -z "${backend}" || -z "${task}" ]]; then
    echo "Usage: $0 {rsl_rl|rl_games|sb3|skrl} TASK [training arguments...]" >&2
    exit 2
fi
shift 2

mkdir -p "${ROOT}/logs" "${ROOT}/tmp" "${PORTABLE_ROOT}"
export TMPDIR="${ROOT}/tmp"
export PYTHONPATH="${ORBIT_ROOT}/source/extensions/orbit.surgical.ext:${ORBIT_ROOT}/source/extensions/orbit.surgical.assets:${ORBIT_ROOT}/source/extensions/orbit.surgical.tasks:${PYTHONPATH:-}"

# Release only the Dr.Anmar interactive worker before allocating training envs.
"${ORBIT_ROOT}/dr_anmar_workstation.sh" stop "${DR_ANMAR_WORKER_PORT:-2361}" || true
cd "${ORBIT_ROOT}"

case "${backend}" in
    rsl_rl) script="source/standalone/workflows/rsl_rl/train.py" ;;
    rl_games) script="source/standalone/workflows/rl_games/train.py" ;;
    sb3) script="source/standalone/workflows/sb3/train.py" ;;
    skrl) script="source/standalone/workflows/skrl/train.py" ;;
    *) echo "Unknown backend: ${backend}" >&2; exit 2 ;;
esac

exec "${PYTHON}" "${script}" --headless --task "${task}" "$@" --kit_args "--portable-root ${PORTABLE_ROOT}"
