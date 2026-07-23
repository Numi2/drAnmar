#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${project_root}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${project_root}/.env"
  set +a
fi

mode="${1:-}"
if [[ -z "${mode}" ]]; then
  echo "Usage: $0 {dry-run|validate-data|train|infer} [arguments...]" >&2
  exit 2
fi
shift

app_root="${DR_ANMAR_ROOT:-$HOME/.local/share/dr-anmar}"
i4h_root="${DR_ANMAR_I4H_ROOT:-${app_root}/vendor/i4h-workflows-current}"
agentic_root="${i4h_root}/workflows/agentic"
stack_root="${agentic_root}/policy/gr00t_n15"
policy_bin="${stack_root}/.venv/bin/i4h-agentic-gr00t-n15"
train_bin="${stack_root}/.venv/bin/i4h-agentic-gr00t-n15-train"

[[ -x "${policy_bin}" && -x "${train_bin}" ]] || {
  echo "NVIDIA GR00T N1.5 Agentic policy environment is not installed under ${stack_root}." >&2
  exit 1
}

export WORKFLOW_ROOT="${agentic_root}"
export ENVIRONMENT_CONFIG="${project_root}/config/i4h/psm_foundation.yaml"
export PYTHONPATH="${project_root}/i4h_overlays:${stack_root}:${agentic_root}/common${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${app_root}/cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${app_root}/cache/lerobot}"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p "${HF_HOME}" "${HF_LEROBOT_HOME}"

case "${mode}" in
  dry-run)
    exec env -u VIRTUAL_ENV "${policy_bin}" --env surgical_lift_needle --dry-run "$@"
    ;;
  validate-data)
    exec env -u VIRTUAL_ENV "${train_bin}" \
      --env surgical_lift_needle \
      --validate-only \
      "$@"
    ;;
  train)
    exec env -u VIRTUAL_ENV "${train_bin}" --env surgical_lift_needle "$@"
    ;;
  infer)
    exec env -u VIRTUAL_ENV "${policy_bin}" --env surgical_lift_needle "$@"
    ;;
  *)
    echo "Unknown mode ${mode}; choose dry-run, validate-data, train, or infer." >&2
    exit 2
    ;;
esac
