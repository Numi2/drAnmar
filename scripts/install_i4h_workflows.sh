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
root="${DR_ANMAR_I4H_ROOT:-${app_root}/vendor/i4h-workflows}"
release="${DR_ANMAR_I4H_RELEASE:-v0.6.0}"
holohub_cli_commit="${DR_ANMAR_HOLOHUB_CLI_COMMIT:-f7e791dac061e01c560d3a2c5b7da82350915b69}"
repository="https://github.com/isaac-for-healthcare/i4h-workflows.git"

mkdir -p "$(dirname "$root")"

if [[ -d "$root/.git" ]]; then
  git -C "$root" fetch --tags --prune origin
  git -C "$root" checkout --detach "$release"
else
  git clone --depth 1 --branch "$release" "$repository" "$root"
fi

installed_cli_commit=""
if [[ -f "$root/tools/utilities/cli/.cli_commit_hash" ]]; then
  installed_cli_commit="$(tr -d '[:space:]' < "$root/tools/utilities/cli/.cli_commit_hash")"
fi
if [[ ! -f "$root/tools/utilities/cli/holohub.py" || "$installed_cli_commit" != "$holohub_cli_commit" ]]; then
  CLI_PINNED_COMMIT="$holohub_cli_commit" CLI_FORCE_UPDATE=1 "$root/i4h" list >/dev/null
fi

resolved_release="$(git -C "$root" rev-parse HEAD)"

echo "Isaac for Healthcare workflows: $root"
echo "Pinned release: $release"
echo "Resolved workflow commit: $resolved_release"
echo "Compatible HoloHub CLI: $holohub_cli_commit"
echo "Set DR_ANMAR_I4H_ROOT=$root when launching Dr.Anmar."
echo "Inspect available modes with: $root/i4h modes <workflow>"
