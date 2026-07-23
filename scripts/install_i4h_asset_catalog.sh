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
release="${DR_ANMAR_I4H_ASSET_CATALOG_RELEASE:-v0.7.0}"
release_commit="${DR_ANMAR_I4H_ASSET_CATALOG_COMMIT:-b0b7ad39f26490d58d12407cfa74b3c9ad861769}"
asset_version="${DR_ANMAR_I4H_ASSET_VERSION:-0.7.0}"
asset_hash="${DR_ANMAR_I4H_ASSET_HASH:-724f82e}"
vendor_root="${app_root}/vendor"
root="${DR_ANMAR_I4H_ASSET_CATALOG_INSTALL_ROOT:-${vendor_root}/i4h-asset-catalog-${release}}"
active_link="${DR_ANMAR_I4H_ASSET_CATALOG_ACTIVE_LINK:-${vendor_root}/i4h-asset-catalog-current}"
runtime_root="${app_root}/runtime/i4h-asset-catalog-${release}"
venv="${runtime_root}/.venv"
repository="https://github.com/isaac-for-healthcare/i4h-asset-catalog.git"

uv="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${uv}" && -x "${HOME}/.local/bin/uv" ]]; then
  uv="${HOME}/.local/bin/uv"
fi
if [[ -z "${uv}" ]]; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun." >&2
  exit 1
fi

mkdir -p "${vendor_root}" "${runtime_root}" "${app_root}/cache/uv" "${app_root}/runtime/uv-python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${app_root}/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${app_root}/runtime/uv-python}"

if [[ -d "${root}/.git" ]]; then
  git -C "${root}" fetch --tags --prune origin
  git -C "${root}" checkout --detach "${release}"
else
  git clone --depth 1 --branch "${release}" "${repository}" "${root}"
fi

resolved_release="$(git -C "${root}" rev-parse HEAD)"
if [[ "${resolved_release}" != "${release_commit}" ]]; then
  echo "Resolved ${release} to ${resolved_release}, expected ${release_commit}" >&2
  exit 1
fi

installed_hash="$(
  python3 - "${root}/i4h_asset_helper/assets_sha256.json" "${asset_version}" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2], ""))
PY
)"
if [[ "${installed_hash}" != "${asset_hash}" ]]; then
  echo "Catalog ${asset_version} resolves to ${installed_hash}, expected ${asset_hash}" >&2
  exit 1
fi

"${uv}" venv --python 3.10 "${venv}"
"${uv}" pip install --python "${venv}/bin/python" "${root}"
"${venv}/bin/python" - "${asset_version}" "${asset_hash}" <<'PY'
import sys
from i4h_asset_helper import get_i4h_asset_hash

actual = get_i4h_asset_hash(sys.argv[1])
if actual != sys.argv[2]:
    raise SystemExit(f"Installed helper resolves {sys.argv[1]} to {actual}, expected {sys.argv[2]}")
PY

if [[ -e "${active_link}" && ! -L "${active_link}" ]]; then
  echo "Refusing to replace non-symlink provider path: ${active_link}" >&2
  exit 1
fi
ln -sfn "${root}" "${active_link}"

echo "Isaac for Healthcare asset catalog: ${root}"
echo "Pinned release: ${release}"
echo "Resolved catalog commit: ${resolved_release}"
echo "Asset content: ${asset_version}/${asset_hash}"
echo "Helper environment: ${venv}"
echo "Active Dr.Anmar asset provider: ${active_link}"
echo "No asset payload was downloaded automatically."
echo "Retrieve the surgical core with: ${project_root}/scripts/fetch_i4h_assets.sh surgical-core"
