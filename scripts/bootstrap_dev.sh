#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is required; install the pinned uv==0.6.14 CLI first.\n' >&2
    exit 1
fi
if [[ "$(uv --version)" != "uv 0.6.14 "* ]]; then
    printf 'This lock was generated with uv 0.6.14; found %s.\n' "$(uv --version)" >&2
    exit 1
fi

uv sync --locked --dev

if [[ "${1:-}" == "--check" ]]; then
    exec "${REPOSITORY_ROOT}/scripts/check_non_isaac.sh"
fi

printf 'Dr.Anmar validation environment is ready. Run scripts/check_non_isaac.sh.\n'
