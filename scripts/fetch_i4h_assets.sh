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
policy_path="${project_root}/config/dranmar_asset_catalog.json"
read -r policy_release policy_version policy_hash < <(
  python3 - "${policy_path}" <<'PY'
import json
import sys

provider = json.load(open(sys.argv[1], encoding="utf-8"))["providers"]["nvidia_i4h"]
print(provider["release"], provider["asset_version"], provider["content_hash"])
PY
)
release="${DR_ANMAR_I4H_ASSET_CATALOG_RELEASE:-${policy_release}}"
asset_version="${DR_ANMAR_I4H_ASSET_VERSION:-${policy_version}}"
asset_hash="${DR_ANMAR_I4H_ASSET_HASH:-${policy_hash}}"
download_dir="${I4H_ASSET_DOWNLOAD_DIR:-${app_root}/assets/i4h-catalog}"
helper="${app_root}/runtime/i4h-asset-catalog-${release}/.venv/bin/i4h-asset-retrieve"
bundle="${1:-}"

if [[ ! -x "${helper}" ]]; then
  echo "The pinned asset helper is not installed. Run scripts/install_i4h_asset_catalog.sh first." >&2
  exit 1
fi

paths=()
while IFS= read -r asset_subpath; do
  paths+=("${asset_subpath}")
done < <(
  python3 - "${policy_path}" "${bundle}" <<'PY'
import json
import sys

policy = json.load(open(sys.argv[1], encoding="utf-8"))
for asset_path in policy.get("i4h_bundles", {}).get(sys.argv[2], ()):
    print(asset_path)
PY
)
if [[ "${#paths[@]}" -eq 0 ]]; then
  available="$(
    python3 - "${policy_path}" <<'PY'
import json
import sys

print("|".join(json.load(open(sys.argv[1], encoding="utf-8"))["i4h_bundles"]))
PY
  )"
  echo "Usage: $0 {${available}}" >&2
  exit 2
fi

mkdir -p "${download_dir}" "${app_root}/run"
export I4H_ASSET_ENV=production
export I4H_ASSET_VERSION="${asset_version}"
export I4H_ASSET_SHA256_HASH="${asset_hash}"
export I4H_ASSET_DOWNLOAD_DIR="${download_dir}"

for path in "${paths[@]}"; do
  "${helper}" \
    --version "${asset_version}" \
    --hash "${asset_hash}" \
    --download-dir "${download_dir}" \
    --sub-path "${path}"
done

python3 - "${app_root}/run/i4h_asset_catalog.json" "${bundle}" "${asset_version}" "${asset_hash}" \
  "${download_dir}/${asset_hash}" "${paths[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output, bundle, version, content_hash, content_root, *requested = sys.argv[1:]
root = Path(content_root)
files = [path for path in root.rglob("*") if path.is_file()]
manifest = {
    "schema": "dr.anmar.i4h-asset-catalog-installation.v1",
    "provider": "NVIDIA Isaac for Healthcare asset catalog",
    "asset_version": version,
    "asset_hash": content_hash,
    "last_retrieved_bundle": bundle,
    "last_requested_subpaths": requested,
    "content_root": str(root),
    "cache_file_count": len(files),
    "cache_bytes": sum(path.stat().st_size for path in files),
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "license_review_required": True,
}
Path(output).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY
