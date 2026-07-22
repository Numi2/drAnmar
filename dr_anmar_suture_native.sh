#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DR_ANMAR_ROOT:-${HOME}/.local/share/dr-anmar}"
RUNTIME_ROOT="${DR_ANMAR_SUTURE_NATIVE_ROOT:-${DATA_ROOT}/native-suture}"
ASSET_ROOT="${RUNTIME_ROOT}/softmimicgen-assets"
REPORT_ROOT="${RUNTIME_ROOT}/reports"
UPSTREAM_ROOT="${DR_ANMAR_SOFTMIMICGEN_ROOT:-${DATA_ROOT}/native-suture-runtime/SoftMimicGen}"
ISAAC_PYTHON="${DR_ANMAR_STABLE_ISAAC_PYTHON:-/home/gilgamesh/isaaclab_pip/env_isaaclab/bin/python}"
NEEDLE_USD="${DR_ANMAR_NEEDLE_USD:-${REPOSITORY_ROOT}/source/extensions/orbit.surgical.assets/data/Props/Surgical_needle/needle_sdf.usd}"

mkdir -p "${ASSET_ROOT}" "${REPORT_ROOT}" "${RUNTIME_ROOT}/tmp"

manifest_value() {
    python3 -c 'import json,sys; value=json.load(open(sys.argv[1]));
for key in sys.argv[2].split("."): value=value[key]
print(value)' "${REPOSITORY_ROOT}/physics_next/softmimicgen.json" "$1"
}

install_upstream() {
    local source_revision fork_url fork_revision
    source_revision="$(manifest_value repository.revision)"
    fork_url="$(manifest_value compatibility.official_isaaclab_fork.url)"
    fork_revision="$(manifest_value compatibility.official_isaaclab_fork.revision)"
    mkdir -p "$(dirname -- "${UPSTREAM_ROOT}")"
    if [[ ! -d "${UPSTREAM_ROOT}/.git" ]]; then
        git clone --filter=blob:none "$(manifest_value repository.url)" "${UPSTREAM_ROOT}"
    fi
    git -C "${UPSTREAM_ROOT}" fetch --filter=blob:none origin "${source_revision}"
    git -C "${UPSTREAM_ROOT}" checkout --detach "${source_revision}"
    if [[ ! -d "${UPSTREAM_ROOT}/third_party/IsaacLab/.git" ]]; then
        git clone --filter=blob:none "${fork_url}" "${UPSTREAM_ROOT}/third_party/IsaacLab"
    fi
    git -C "${UPSTREAM_ROOT}/third_party/IsaacLab" fetch --filter=blob:none origin "${fork_revision}"
    git -C "${UPSTREAM_ROOT}/third_party/IsaacLab" checkout --detach "${fork_revision}"
    python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_fetch_softmimicgen.py" --output "${ASSET_ROOT}"
    install -D -m 0644 "${ASSET_ROOT}/Rope.usd" "${UPSTREAM_ROOT}/source/softmimicgen_assets/data/Props/Rope/Rope.usd"
    install -D -m 0644 "${ASSET_ROOT}/Ring.usd" "${UPSTREAM_ROOT}/source/softmimicgen_assets/data/Props/Ring/Ring.usd"
    install -D -m 0644 "${ASSET_ROOT}/Table.usd" "${UPSTREAM_ROOT}/source/softmimicgen_assets/data/Props/Table/Table.usd"
    install -D -m 0644 "${ASSET_ROOT}/psm_forceps.usd" "${UPSTREAM_ROOT}/source/softmimicgen_assets/data/Robots/dVRK/PSM/psm_forceps.usd"
    install -D -m 0644 "${ASSET_ROOT}/annotated_dataset_surgical_threading.hdf5" \
        "${UPSTREAM_ROOT}/datasets/annotated_dataset/annotated_dataset_surgical_threading.hdf5"
}

upstream_pythonpath() {
    printf '%s' \
        "${UPSTREAM_ROOT}/third_party/IsaacLab/source/isaaclab:"\
"${UPSTREAM_ROOT}/third_party/IsaacLab/source/isaaclab_assets:"\
"${UPSTREAM_ROOT}/third_party/IsaacLab/source/isaaclab_mimic:"\
"${UPSTREAM_ROOT}/third_party/IsaacLab/source/isaaclab_rl:"\
"${UPSTREAM_ROOT}/third_party/IsaacLab/source/isaaclab_tasks:"\
"${UPSTREAM_ROOT}/source/softmimicgen:"\
"${UPSTREAM_ROOT}/source/softmimicgen_assets:"\
"${UPSTREAM_ROOT}/source/softmimicgen_tasks:"\
"${PYTHONPATH:-}"
}

case "${1:-status}" in
    fetch)
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_fetch_softmimicgen.py" \
            --output "${ASSET_ROOT}"
        ;;
    install-upstream)
        install_upstream
        echo "Pinned NVIDIA SoftMimicGen runtime installed: ${UPSTREAM_ROOT}"
        ;;
    status)
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_fetch_softmimicgen.py" \
            --output "${ASSET_ROOT}" --verify-only
        latest="$(find "${REPORT_ROOT}" -maxdepth 1 -name 'softmimicgen-suture-*.json' -type f -print 2>/dev/null | sort | tail -1)"
        if [[ -n "${latest}" ]]; then
            echo "Latest qualification: ${latest}"
            python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("core_passed=",r.get("core_passed"),"promotion_passed=",r.get("promotion_passed"))' "${latest}"
        fi
        latest_replay="$(find "${REPORT_ROOT}" -maxdepth 1 -name 'softmimicgen-replay-*.json' -type f -print 2>/dev/null | sort | tail -1)"
        if [[ -n "${latest_replay}" ]]; then
            echo "Latest upstream replay: ${latest_replay}"
            python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("pass=",r.get("pass"),"live_terminal_success=",r.get("live_terminal_success"),"first_success_step=",r.get("first_live_success_step"))' "${latest_replay}"
        fi
        ;;
    qualify)
        [[ -x "${ISAAC_PYTHON}" ]] || { echo "Isaac Python not found: ${ISAAC_PYTHON}" >&2; exit 1; }
        [[ -f "${NEEDLE_USD}" ]] || { echo "Needle asset not found: ${NEEDLE_USD}" >&2; exit 1; }
        python3 "${REPOSITORY_ROOT}/scripts/dr_anmar_fetch_softmimicgen.py" --output "${ASSET_ROOT}"
        gate="${2:-core}"
        [[ "${gate}" == "core" || "${gate}" == "promotion" ]] || { echo "gate must be core or promotion" >&2; exit 2; }
        output="${REPORT_ROOT}/softmimicgen-suture-$(date -u +%Y%m%dT%H%M%SZ).json"
        TMPDIR="${RUNTIME_ROOT}/tmp" "${ISAAC_PYTHON}" "${REPOSITORY_ROOT}/scripts/dr_anmar_softmimicgen_suture_probe.py" \
            --asset-dir "${ASSET_ROOT}" \
            --needle-usd "${NEEDLE_USD}" \
            --gate "${gate}" \
            --output "${output}" \
            --headless
        python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); raise SystemExit(0 if r.get("passed") is True else 1)' "${output}"
        echo "Qualification: ${output}"
        ;;
    validate-upstream)
        [[ -x "${ISAAC_PYTHON}" ]] || { echo "Isaac Python not found: ${ISAAC_PYTHON}" >&2; exit 1; }
        dataset="${UPSTREAM_ROOT}/datasets/annotated_dataset/annotated_dataset_surgical_threading.hdf5"
        [[ -f "${dataset}" ]] || { echo "Install the pinned upstream runtime first." >&2; exit 1; }
        output="${REPORT_ROOT}/softmimicgen-replay-$(date -u +%Y%m%dT%H%M%SZ).json"
        PYTHONPATH="$(upstream_pythonpath)" TMPDIR="${RUNTIME_ROOT}/tmp" \
            "${ISAAC_PYTHON}" "${REPOSITORY_ROOT}/scripts/dr_anmar_softmimicgen_replay_validate.py" \
            --headless \
            --task Isaac-Thread-PSM-IK-Rel-v0 \
            --dataset "${dataset}" \
            --episode "${2:-demo_0}" \
            --report "${output}" \
            --kit_args "--portable-root ${DATA_ROOT}/isaac_portable-softmimicgen-validation"
        echo "Replay validation: ${output}"
        ;;
    *)
        echo "Usage: $0 {fetch|install-upstream|status|qualify [core|promotion]|validate-upstream [demo_0]}" >&2
        exit 2
        ;;
esac
