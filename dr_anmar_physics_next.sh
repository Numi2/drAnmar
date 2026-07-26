#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${REPOSITORY_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPOSITORY_ROOT}/.env"
    set +a
fi

DATA_ROOT="${DR_ANMAR_ROOT:-${HOME}/.local/share/dr-anmar}"
NEXT_ROOT="${DR_ANMAR_PHYSICS_NEXT_ROOT:-${DATA_ROOT}/physics-next}"
ENV_ROOT="${NEXT_ROOT}/env_isaaclab"
ISAACLAB_ROOT="${NEXT_ROOT}/IsaacLab"
LOG_ROOT="${NEXT_ROOT}/logs"
INSTALL_LOG="${LOG_ROOT}/install.log"
PID_FILE="${NEXT_ROOT}/install.pid"
LOCK_PATH="${REPOSITORY_ROOT}/config/physics-next-lock.json"

lock_value() {
    python3 - "${LOCK_PATH}" "$1" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

PYTHON_VERSION="$(lock_value python.version)"
PIP_VERSION="$(lock_value python.pip)"
ISAACSIM_VERSION="$(lock_value simulator.version)"
ISAACLAB_REPOSITORY="$(lock_value sources.isaaclab.repository)"
ISAACLAB_COMMIT="$(lock_value sources.isaaclab.revision)"
CRESSIM_REPOSITORY="$(lock_value sources.cressim_mpm.repository)"
CRESSIM_COMMIT="$(lock_value sources.cressim_mpm.revision)"
PYTETWILD_VERSION="$(lock_value mesh_environment.pytetwild)"
SCIPY_VERSION="$(lock_value mesh_environment.scipy)"

mkdir -p "${NEXT_ROOT}" "${LOG_ROOT}"

install_running() {
    [[ -f "${PID_FILE}" ]] || return 1
    local pid
    pid="$(cat "${PID_FILE}")"
    kill -0 "${pid}" 2>/dev/null
}

install_runtime() {
    if [[ -f "${NEXT_ROOT}/READY" ]]; then
        echo "physics-next runtime is already ready"
        return 0
    fi
    if install_running; then
        echo "physics-next installation already running (PID $(cat "${PID_FILE}"))"
        return 0
    fi
    nohup bash -euo pipefail -c '
        next_root="$1"
        env_root="$2"
        isaaclab_root="$3"
        rm -f "${next_root}/READY" "${next_root}/INSTALL_FAILED"
        finish_install() {
            code=$?
            if [[ "${code}" -ne 0 ]]; then
                printf "exit_code=%s\nfailed_at=%s\n" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${next_root}/INSTALL_FAILED"
            fi
        }
        trap finish_install EXIT
        lock_path="$4"
        python_version="$5"
        pip_version="$6"
        isaacsim_version="$7"
        isaaclab_repository="$8"
        isaaclab_commit="$9"
        cressim_repository="${10}"
        cressim_commit="${11}"
        "python${python_version}" -m venv "${env_root}"
        "${env_root}/bin/python" -m pip install "pip==${pip_version}"
        "${env_root}/bin/python" -m pip install "isaacsim[all,extscache]==${isaacsim_version}" --extra-index-url https://pypi.nvidia.com
        if [[ ! -d "${isaaclab_root}/.git" ]]; then
            git clone --filter=blob:none --no-checkout "${isaaclab_repository}" "${isaaclab_root}"
        fi
        cd "${isaaclab_root}"
        git fetch --depth 1 origin "${isaaclab_commit}"
        git checkout --detach "${isaaclab_commit}"
        [[ "$(git rev-parse HEAD)" == "${isaaclab_commit}" ]]
        PATH="${env_root}/bin:${PATH}" VIRTUAL_ENV="${env_root}" ./isaaclab.sh --install
        cressim_root="${next_root}/CRESSim-MPM"
        if [[ ! -d "${cressim_root}/.git" ]]; then
            git clone --filter=blob:none --no-checkout "${cressim_repository}" "${cressim_root}"
        fi
        cd "${cressim_root}"
        git fetch --depth 1 origin "${cressim_commit}"
        git checkout --detach "${cressim_commit}"
        [[ "$(git rev-parse HEAD)" == "${cressim_commit}" ]]
        "${env_root}/bin/python" -m pip check
        "${env_root}/bin/python" -m pip freeze --all > "${next_root}/python-freeze.txt"
        ISAACLAB_COMMIT="${isaaclab_commit}" CRESSIM_COMMIT="${cressim_commit}" \
            "${env_root}/bin/python" - "${lock_path}" "${next_root}/python-freeze.txt" <<PY > "${next_root}/runtime.json"
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def package(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

try:
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip().splitlines()
except (OSError, subprocess.SubprocessError):
    driver = []

installed_packages = {
    name: package(name)
    for name in lock["runtime_packages"]
}
package_mismatches = {
    name: {
        "expected": expected,
        "observed": installed_packages.get(name),
    }
    for name, expected in lock["runtime_packages"].items()
    if installed_packages.get(name) != expected
}
if package_mismatches:
    raise SystemExit(
        "runtime package lock mismatch: "
        + json.dumps(package_mismatches, sort_keys=True)
    )

receipt = {
    "schema": "dr.anmar.physics-next-installation.v1",
    "ready": True,
    "lock_sha256": sha256(sys.argv[1]),
    "python_freeze_sha256": sha256(sys.argv[2]),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "gpu_driver": driver,
    "packages": installed_packages,
    "sources": {
        "isaaclab": os.environ["ISAACLAB_COMMIT"],
        "cressim_mpm": os.environ["CRESSIM_COMMIT"],
    },
    "clinical_validation": False,
}
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
        cat "${next_root}/runtime.json"
        shasum -a 256 "${next_root}/runtime.json" | awk "{print \$1}" > "${next_root}/READY"
        rm -f "${next_root}/INSTALL_FAILED"
    ' bash "${NEXT_ROOT}" "${ENV_ROOT}" "${ISAACLAB_ROOT}" "${LOCK_PATH}" \
        "${PYTHON_VERSION}" "${PIP_VERSION}" "${ISAACSIM_VERSION}" \
        "${ISAACLAB_REPOSITORY}" "${ISAACLAB_COMMIT}" \
        "${CRESSIM_REPOSITORY}" "${CRESSIM_COMMIT}" >>"${INSTALL_LOG}" 2>&1 &
    echo "$!" >"${PID_FILE}"
    echo "Started isolated physics-next installation (PID $!)"
    echo "Log: ${INSTALL_LOG}"
}

case "${1:-status}" in
    install)
        install_runtime
        ;;
    status)
        if [[ -f "${NEXT_ROOT}/READY" ]]; then
            if python3 "${REPOSITORY_ROOT}/scripts/verify_dranmar_physics_next_receipt.py" \
                --root "${NEXT_ROOT}" --lock "${LOCK_PATH}" >/dev/null; then
                echo "physics-next runtime: ready and receipt-verified"
            else
                echo "physics-next runtime: stale or unverifiable" >&2
                python3 "${REPOSITORY_ROOT}/scripts/verify_dranmar_physics_next_receipt.py" \
                    --root "${NEXT_ROOT}" --lock "${LOCK_PATH}" >&2
                exit 1
            fi
        elif install_running; then
            echo "physics-next runtime: installing (PID $(cat "${PID_FILE}"))"
        elif [[ -f "${NEXT_ROOT}/INSTALL_FAILED" ]]; then
            echo "physics-next runtime: install failed"
            cat "${NEXT_ROOT}/INSTALL_FAILED"
        else
            echo "physics-next runtime: not installed"
        fi
        ;;
    logs)
        tail -n 160 "${INSTALL_LOG}"
        ;;
    probe)
        [[ -x "${ENV_ROOT}/bin/python" ]] || { echo "physics-next environment is not installed" >&2; exit 1; }
        "${ENV_ROOT}/bin/python" "${REPOSITORY_ROOT}/scripts/dr_anmar_physics_next.py" status
        ;;
    benchmark)
        backend="${2:-physx}"
        [[ "${backend}" == "physx" || "${backend}" == "newton" ]] || {
            echo "backend must be physx or newton" >&2
            exit 2
        }
        [[ -f "${NEXT_ROOT}/READY" ]] || { echo "physics-next runtime is not ready" >&2; exit 1; }
        if [[ "${backend}" == "physx" && "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]]; then
            echo "PhysX FEM requires the Isaac Sim / Omniverse Kit runtime." >&2
            echo "NVIDIA requires explicit EULA acceptance before that runtime starts:" >&2
            echo "https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html" >&2
            echo "After you accept it, rerun with OMNI_KIT_ACCEPT_EULA=YES. Dr.Anmar will not accept legal terms for you." >&2
            exit 3
        fi
        output="${NEXT_ROOT}/benchmarks/liver-retraction-${backend}-$(date -u +%Y%m%dT%H%M%SZ).json"
        mkdir -p "$(dirname "${output}")"
        cd "${ISAACLAB_ROOT}"
        arguments=(--backend "${backend}" --viz none --device cuda:0 --output "${output}")
        if [[ "${backend}" == "newton" ]]; then
            reference="${output%.json}-replay-reference.json"
            "${ENV_ROOT}/bin/python" "${REPOSITORY_ROOT}/scripts/dr_anmar_physics_next_runtime.py" \
                --backend newton --viz none --device cuda:0 --output "${reference}"
            arguments+=(--replay-reference "${reference}")
        fi
        "${ENV_ROOT}/bin/python" "${REPOSITORY_ROOT}/scripts/dr_anmar_physics_next_runtime.py" "${arguments[@]}"
        echo "Benchmark: ${output}"
        ;;
    extract-liver)
        scene="${2:?Usage: $0 extract-liver SCENE_USD [OUTPUT_DIR] [PRIM_PATH]}"
        output="${3:-${NEXT_ROOT}/assets/ct-liver-prostate-bladder/liver}"
        prim="${4:-/root/Liver_topo_blender/Liver_topo_blender}"
        "${ISAAC_PYTHON:-python3}" "${REPOSITORY_ROOT}/scripts/dr_anmar_physics_asset_prepare.py" \
            --scene "${scene}" --prim "${prim}" --output "${output}" --headless
        ;;
    tetrahedralize-liver)
        extraction="${2:?Usage: $0 tetrahedralize-liver EXTRACTION_JSON [OUTPUT_DIR] [EDGE_LENGTH_M]}"
        output="${3:-$(dirname "${extraction}")/interactive-8mm}"
        edge_length="${4:-0.008}"
        mesh_python="${NEXT_ROOT}/env_mesh/bin/python"
        if [[ ! -x "${mesh_python}" ]]; then
            "python${PYTHON_VERSION}" -m venv "${NEXT_ROOT}/env_mesh"
            "${mesh_python}" -m pip install "pip==${PIP_VERSION}"
            "${mesh_python}" -m pip install \
                "pytetwild[all]==${PYTETWILD_VERSION}" \
                "scipy==${SCIPY_VERSION}"
        fi
        "${mesh_python}" "${REPOSITORY_ROOT}/scripts/dr_anmar_tetrahedralize.py" \
            --extraction "${extraction}" --output "${output}" --edge-length-m "${edge_length}" --threads 1
        ;;
    author-liver-usd)
        mesh="${2:?Usage: $0 author-liver-usd TETRA_NPZ [OUTPUT_USD]}"
        output="${3:-$(dirname "${mesh}")/liver-tetmesh.usda}"
        usd_python="${ENV_ROOT}/bin/python"
        [[ -x "${usd_python}" ]] || usd_python="${ISAAC_PYTHON:-python3}"
        canonical="$(dirname "${mesh}")/canonical-asset.json"
        arguments=(--mesh "${mesh}" --output "${output}")
        [[ -f "${canonical}" ]] && arguments+=(--canonical "${canonical}")
        "${usd_python}" "${REPOSITORY_ROOT}/scripts/dr_anmar_tet_to_usd.py" "${arguments[@]}"
        ;;
    author-liver-physx)
        surface="${2:?Usage: $0 author-liver-physx SURFACE_OBJ [OUTPUT_USD] [ISAACLAB_ROOT]}"
        output="${3:-$(dirname "${surface}")/stable-physx/liver-surface.usd}"
        converter_root="${4:-${DR_ANMAR_STABLE_ISAACLAB_ROOT:-${ISAACLAB_ROOT}}}"
        converter="${converter_root}/scripts/tools/convert_mesh.py"
        converter_python="${ISAAC_PYTHON:-${ENV_ROOT}/bin/python}"
        [[ -f "${converter}" ]] || { echo "Isaac Lab mesh converter not found: ${converter}" >&2; exit 1; }
        [[ -x "${converter_python}" ]] || { echo "Isaac Python not found: ${converter_python}" >&2; exit 1; }
        mkdir -p "$(dirname "${output}")"
        "${converter_python}" "${converter}" "${surface}" "${output}" \
            --collision-approximation none \
            --headless \
            --kit_args "--portable-root ${DATA_ROOT}/isaac_portable"
        echo "PhysX surface asset: ${output}"
        ;;
    patient-liver-smoke)
        mesh="${2:-${NEXT_ROOT}/assets/ct-liver-prostate-bladder/liver/interactive-8mm/simulation-tetrahedra.npz}"
        output="${3:-${NEXT_ROOT}/benchmarks/patient-liver-newton-$(date -u +%Y%m%dT%H%M%SZ).json}"
        [[ -f "${mesh}" ]] || { echo "patient liver TetMesh not found: ${mesh}" >&2; exit 1; }
        [[ -x "${ENV_ROOT}/bin/python" ]] || { echo "physics-next environment is not installed" >&2; exit 1; }
        "${ENV_ROOT}/bin/python" "${REPOSITORY_ROOT}/scripts/dr_anmar_patient_liver_newton_smoke.py" \
            --mesh "${mesh}" --output "${output}" --device cuda:0
        echo "Patient liver smoke result: ${output}"
        ;;
    compare)
        shift
        [[ "$#" -gt 0 ]] || { echo "Usage: $0 compare RESULT_JSON [...]" >&2; exit 2; }
        "${ISAAC_PYTHON:-python3}" "${REPOSITORY_ROOT}/scripts/dr_anmar_physics_next_compare.py" "$@"
        ;;
    *)
        echo "Usage: $0 {install|status|logs|probe|benchmark [physx|newton]|compare RESULT [...]|extract-liver SCENE [OUTPUT] [PRIM]|tetrahedralize-liver EXTRACTION [OUTPUT] [EDGE]|author-liver-usd TETRA_NPZ [OUTPUT_USD]|author-liver-physx SURFACE_OBJ [OUTPUT_USD] [ISAACLAB_ROOT]|patient-liver-smoke [TETRA_NPZ] [OUTPUT]}" >&2
        exit 2
        ;;
esac
