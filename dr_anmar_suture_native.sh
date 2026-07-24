#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DR_ANMAR_ROOT:-${HOME}/.local/share/dr-anmar}"
RUNTIME_ROOT="${DR_ANMAR_SUTURE_NATIVE_ROOT:-${DATA_ROOT}/native-suture}"
ASSET_ROOT="${RUNTIME_ROOT}/softmimicgen-assets"
UPSTREAM_ROOT="${DR_ANMAR_SOFTMIMICGEN_ROOT:-${DATA_ROOT}/native-suture-runtime/SoftMimicGen}"

mkdir -p "${ASSET_ROOT}" "${RUNTIME_ROOT}/tmp"

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
        ;;
    *)
        echo "Usage: $0 {fetch|install-upstream|status}" >&2
        exit 2
        ;;
esac
