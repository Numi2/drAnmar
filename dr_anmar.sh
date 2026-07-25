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

mkdir -p "${ROOT}/logs" "${ROOT}/tmp" "${PORTABLE_ROOT}"
export TMPDIR="${ROOT}/tmp"
export PYTHONPATH="${ORBIT_ROOT}/source/extensions/orbit.surgical.ext:${ORBIT_ROOT}/source/extensions/orbit.surgical.assets:${ORBIT_ROOT}/source/extensions/orbit.surgical.tasks:${PYTHONPATH:-}"

activate_softmimicgen() {
    local root isaaclab
    root="${DR_ANMAR_SOFTMIMICGEN_ROOT:-${ROOT}/native-suture-runtime/SoftMimicGen}"
    isaaclab="${root}/third_party/IsaacLab"
    [[ -f "${root}/source/softmimicgen_tasks/softmimicgen_tasks/__init__.py" ]] || {
        echo "Pinned SoftMimicGen runtime is not installed: ${root}" >&2
        echo "Run ./dr_anmar_suture_native.sh install-upstream first." >&2
        exit 1
    }
    [[ -f "${isaaclab}/source/isaaclab/isaaclab/__init__.py" ]] || {
        echo "Pinned SoftMimicGen Isaac Lab fork is not installed: ${isaaclab}" >&2
        exit 1
    }
    export DR_ANMAR_SOFTMIMICGEN_ROOT="${root}"
    export PYTHONPATH="${isaaclab}/source/isaaclab:${isaaclab}/source/isaaclab_assets:${isaaclab}/source/isaaclab_mimic:${isaaclab}/source/isaaclab_rl:${isaaclab}/source/isaaclab_tasks:${root}/source/softmimicgen:${root}/source/softmimicgen_assets:${root}/source/softmimicgen_tasks:${PYTHONPATH}"
}

cd "${ORBIT_ROOT}"

command="${1:-smoke}"
case "${command}" in
    list)
        exec "${PYTHON}" source/standalone/environments/list_envs.py \
            --headless \
            --kit_args "--portable-root ${PORTABLE_ROOT}"
        ;;
    smoke)
        task="${2:-Isaac-Reach-PSM-v0}"
        steps="${3:-120}"
        slug="$(printf '%s' "${task}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')"
        exec "${PYTHON}" scripts/dr_anmar_smoke.py \
            --headless \
            --task "${task}" \
            --num_envs 1 \
            --steps "${steps}" \
            --report "${ROOT}/logs/${slug}_smoke.json" \
            --kit_args "--portable-root ${PORTABLE_ROOT}"
        ;;
    workstation)
        port="${2:-2361}"
        task="${3:-Isaac-Lift-Needle-PSM-IK-Rel-v0}"
        procedure="${4:-}"
        anatomy_scene="${5:-}"
        anatomy_scene_id="${6:-}"
        anatomy_title="${7:-}"
        openusd_environment="${8:-}"
        bench_assets="${9:-default}"
        gripper_open_rad="${10:-}"
        gripper_close_rad="${11:-}"
        if [[ "${task}" == Isaac-Thread-PSM-* ]]; then
            activate_softmimicgen
        fi
        workstation_args=(
            --headless
            --enable_cameras
            --host "${DR_ANMAR_WORKER_HOST:-127.0.0.1}"
            --task "${task}"
            --port "${port}"
            --demo_dir "${ROOT}/demos"
        )
        [[ -n "${procedure}" ]] && workstation_args+=(--procedure "${procedure}")
        [[ -n "${anatomy_scene}" ]] && workstation_args+=(--anatomy_scene "${anatomy_scene}")
        [[ -n "${anatomy_scene_id}" ]] && workstation_args+=(--anatomy_scene_id "${anatomy_scene_id}")
        [[ -n "${anatomy_title}" ]] && workstation_args+=(--anatomy_title "${anatomy_title}")
        [[ -n "${openusd_environment}" ]] && workstation_args+=(--openusd_environment "${openusd_environment}")
        workstation_args+=(--bench_assets "${bench_assets}")
        [[ -n "${gripper_open_rad}" ]] && workstation_args+=(--gripper_open_rad "${gripper_open_rad}")
        [[ -n "${gripper_close_rad}" ]] && workstation_args+=(--gripper_close_rad "${gripper_close_rad}")
        exec "${PYTHON}" scripts/dr_anmar_workstation.py \
            "${workstation_args[@]}" \
            --kit_args "--portable-root ${PORTABLE_ROOT}"
        ;;
    anatomy-viewer)
        port="${2:-2361}"
        scene="${3:?An official main_scene.usd path is required}"
        room_id="${4:?A room id is required}"
        room_title="${5:?A room title is required}"
        exec "${PYTHON}" scripts/dr_anmar_anatomy_viewer.py \
            --scene "${scene}" \
            --room_id "${room_id}" \
            --room_title "${room_title}" \
            --port "${port}"
        ;;
    catalog)
        exec "${PYTHON}" scripts/dr_anmar_catalog.py
        ;;
    doctor)
        exec "${PYTHON}" scripts/dr_anmar_doctor.py
        ;;
    laparotomy)
        steps="${2:-0}"
        capture_path="${3:-}"
        if [[ -n "${capture_path}" ]]; then
            exec "${PYTHON}" examples/dynamic_abdominal_patient_scene.py \
                --headless \
                --enable_cameras \
                --device cuda:0 \
                --steps "${steps}" \
                --capture_path "${capture_path}" \
                --kit_args "--portable-root ${PORTABLE_ROOT}"
        fi
        exec "${PYTHON}" examples/dynamic_abdominal_patient_scene.py \
            --device cuda:0 \
            --steps "${steps}" \
            --kit_args "--portable-root ${PORTABLE_ROOT}"
        ;;
    *)
        echo "Usage: $0 {list|catalog|doctor|smoke [TASK] [STEPS]|workstation [PORT] [TASK]|anatomy-viewer [PORT] [SCENE] [ROOM_ID] [ROOM_TITLE]|laparotomy [STEPS] [CAPTURE_PNG]}" >&2
        exit 2
        ;;
esac
