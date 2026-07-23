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
release="${DR_ANMAR_I4H_RELEASE:-v0.7.0}"
release_commit="${DR_ANMAR_I4H_RELEASE_COMMIT:-9b526c6d107254727d3b113c612fb860fc65a5b2}"
vendor_root="${app_root}/vendor"
root="${DR_ANMAR_I4H_INSTALL_ROOT:-${vendor_root}/i4h-workflows-${release}}"
active_link="${DR_ANMAR_I4H_ACTIVE_LINK:-${vendor_root}/i4h-workflows-current}"
holohub_cli_commit="${DR_ANMAR_HOLOHUB_CLI_COMMIT:-f7e791dac061e01c560d3a2c5b7da82350915b69}"
repository="https://github.com/isaac-for-healthcare/i4h-workflows.git"

mkdir -p "$(dirname "$root")"

if [[ -d "$root/.git" ]]; then
  git -C "$root" fetch --tags --prune origin
  git -C "$root" checkout --detach "$release"
else
  git clone --depth 1 --branch "$release" "$repository" "$root"
fi

resolved_release="$(git -C "$root" rev-parse HEAD)"
if [[ "${resolved_release}" != "${release_commit}" ]]; then
  echo "Resolved ${release} to ${resolved_release}, expected ${release_commit}" >&2
  exit 1
fi

installed_cli_commit=""
if [[ -f "$root/tools/utilities/cli/.cli_commit_hash" ]]; then
  installed_cli_commit="$(tr -d '[:space:]' < "$root/tools/utilities/cli/.cli_commit_hash")"
fi
if [[ ! -f "$root/tools/utilities/cli/holohub.py" || "$installed_cli_commit" != "$holohub_cli_commit" ]]; then
  CLI_PINNED_COMMIT="$holohub_cli_commit" CLI_FORCE_UPDATE=1 "$root/i4h" list >/dev/null
fi

if [[ -e "${active_link}" && ! -L "${active_link}" ]]; then
  echo "Refusing to replace non-symlink provider path: ${active_link}" >&2
  echo "Move that checkout to a versioned path, then rerun the installer." >&2
  exit 1
fi
ln -sfn "${root}" "${active_link}"

echo "Isaac for Healthcare workflows: $root"
echo "Pinned release: $release"
echo "Resolved workflow commit: $resolved_release"
echo "Compatible HoloHub CLI: $holohub_cli_commit"
echo "Active Dr.Anmar provider: $active_link"
echo "Previous versioned checkouts remain available for rollback."
echo "Set DR_ANMAR_I4H_ROOT=$root to bypass the active provider link."
echo "Inspect available modes with: $root/i4h modes <workflow>"
echo "Inspect NVIDIA surgical environments with: $root/workflows/agentic/arena/run.sh --list-envs"
