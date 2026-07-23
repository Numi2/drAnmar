#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${project_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${project_root}/.env"
  set +a
fi

app_root="${DR_ANMAR_ROOT:-$HOME/.local/share/dr-anmar}"
i4h_root="${DR_ANMAR_I4H_ROOT:-${app_root}/vendor/i4h-workflows-current}"
agentic_root="${i4h_root}/workflows/agentic"
arena_root="${agentic_root}/arena"

[[ -x "${arena_root}/run.sh" ]] || {
  echo "NVIDIA Agentic Arena is not installed at ${arena_root}." >&2
  echo "Run scripts/install_i4h_workflows.sh and scripts/setup_i4h_agentic.sh first." >&2
  exit 1
}
if [[ -d "${HOME}/.local/bin" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || {
  echo "uv is required by NVIDIA's Agentic workflow." >&2
  exit 1
}

export WORKFLOW_ROOT="${agentic_root}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${app_root}/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${app_root}/runtime/uv-python}"
export PYTHONPATH="${arena_root}:${agentic_root}/common${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

if [[ "$(uname -m)" == "aarch64" ]]; then
  sys_libgomp="$(ls /lib/*/libgomp.so.1 2>/dev/null | head -1 || true)"
  if [[ -n "${sys_libgomp}" ]]; then
    export LD_PRELOAD="${sys_libgomp}${LD_PRELOAD:+:${LD_PRELOAD}}"
  fi
  export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=2000000}"
else
  libgomp_path="$(
    cd "${arena_root}"
    env -u VIRTUAL_ENV uv run --no-sync python -c \
      'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib" / "libgomp.so.1")' \
      2>/dev/null || true
  )"
  if [[ -n "${libgomp_path}" && -e "${libgomp_path}" ]]; then
    export LD_PRELOAD="${libgomp_path}${LD_PRELOAD:+:${LD_PRELOAD}}"
  fi
fi

cd "${arena_root}"
exec env -u VIRTUAL_ENV uv run --no-sync python \
  "${project_root}/scripts/dr_anmar_psm_native_adapter.py" "$@"
