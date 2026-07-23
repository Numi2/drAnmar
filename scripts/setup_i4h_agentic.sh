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
setup="${i4h_root}/workflows/agentic/setup.sh"

if [[ -d "${HOME}/.local/bin" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || {
  echo "uv is required by NVIDIA's Agentic workflow setup." >&2
  echo "Install it from https://docs.astral.sh/uv/getting-started/installation/." >&2
  exit 1
}
[[ -x "${setup}" ]] || {
  echo "NVIDIA Agentic setup was not found at ${setup}." >&2
  echo "Run scripts/install_i4h_workflows.sh first." >&2
  exit 1
}

export UV_CACHE_DIR="${UV_CACHE_DIR:-${app_root}/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${app_root}/runtime/uv-python}"
export I4H_WORKFLOWS="${i4h_root}"
mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

exec "${setup}" "$@"
