#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-Structured-v0"
OUTPUT="${1:-${REPO_ROOT}/output/e2e-handover-experiment}"
NUM_ENVS="${DR_ANMAR_E2E_NUM_ENVS:-2400}"
UPDATES="${DR_ANMAR_E2E_DAGGER_UPDATES:-2400}"
VALIDATION_FRAMES="${DR_ANMAR_E2E_VALIDATION_FRAMES:-1000}"
PPO_ITERATIONS="${DR_ANMAR_E2E_PPO_ITERATIONS:-100}"
MINIMUM_BC_SUCCESS="${DR_ANMAR_E2E_MINIMUM_BC_SUCCESS:-0.25}"
SCREEN_ENVS="${DR_ANMAR_E2E_SCREEN_ENVS:-600}"
MINIMUM_PROMOTION_IMPROVEMENT="${DR_ANMAR_E2E_MINIMUM_PROMOTION_IMPROVEMENT:-0.02}"

export DR_ANMAR_TRUST_REQUESTED_NUM_ENVS=1
export DR_ANMAR_DAGGER_WARMUP_UPDATES="${DR_ANMAR_E2E_WARMUP_UPDATES:-1000}"

"${REPO_ROOT}/dr_anmar_learning.sh" pretrain \
    "${TASK}" \
    "${NUM_ENVS}" \
    "${UPDATES}" \
    "${VALIDATION_FRAMES}" \
    "${OUTPUT}/dagger"

checkpoint="$(
    find "${OUTPUT}/dagger/runs" -name model_final.pt -type f -print \
        | sort \
        | tail -1
)"
if [[ -z "${checkpoint}" ]]; then
    echo "error: DAgger checkpoint was not produced" >&2
    exit 1
fi

evidence="$(
    find "${OUTPUT}/dagger" -name 'dranmar_pretraining_*.json' -type f -print \
        | sort \
        | tail -1
)"
validation_rate="$(
    python3 - "${evidence}" <<'PY'
import json
import sys

evidence = json.load(open(sys.argv[1], encoding="utf-8"))
rate = evidence["deterministic_validation"]["success_rate"]
print(0.0 if rate is None else rate)
PY
)"
if ! python3 - "${validation_rate}" "${MINIMUM_BC_SUCCESS}" <<'PY'
import sys

raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
then
    echo "qualification blocked: deterministic BC success ${validation_rate} is below ${MINIMUM_BC_SUCCESS}" >&2
    echo "PPO was not started." >&2
    exit 3
fi

DR_ANMAR_INIT_CHECKPOINT="${checkpoint}" \
DR_ANMAR_POLICY_LEARNING_RATE="${DR_ANMAR_E2E_PPO_LR:-0.0001}" \
"${REPO_ROOT}/dr_anmar_learning.sh" train \
    "${TASK}" \
    "${NUM_ENVS}" \
    "${PPO_ITERATIONS}" \
    "${OUTPUT}/ppo"

candidate_checkpoint="$(
    find "${OUTPUT}/ppo/runs" -name model_final.pt -type f -print \
        | sort \
        | tail -1
)"
if [[ -z "${candidate_checkpoint}" ]]; then
    echo "error: PPO checkpoint was not produced" >&2
    exit 1
fi

DR_ANMAR_SEED=17 "${REPO_ROOT}/dr_anmar_learning.sh" play \
    "${checkpoint}" \
    "${TASK}" \
    "${SCREEN_ENVS}" \
    "${VALIDATION_FRAMES}" \
    "${OUTPUT}/promotion-screen/baseline"
DR_ANMAR_SEED=17 "${REPO_ROOT}/dr_anmar_learning.sh" play \
    "${candidate_checkpoint}" \
    "${TASK}" \
    "${SCREEN_ENVS}" \
    "${VALIDATION_FRAMES}" \
    "${OUTPUT}/promotion-screen/candidate"

baseline_evidence="$(
    find "${OUTPUT}/promotion-screen/baseline" \
        -name 'dranmar_play_*.json' -type f -print \
        | sort \
        | tail -1
)"
candidate_evidence="$(
    find "${OUTPUT}/promotion-screen/candidate" \
        -name 'dranmar_play_*.json' -type f -print \
        | sort \
        | tail -1
)"
promotion_evidence="${OUTPUT}/promotion-screen/promotion.json"
python3 "${REPO_ROOT}/scripts/dr_anmar_handover_promotion.py" \
    --baseline "${baseline_evidence}" \
    --candidate "${candidate_evidence}" \
    --minimum-success-improvement "${MINIMUM_PROMOTION_IMPROVEMENT}" \
    --maximum-safety-rate-increase 0.0 \
    --output "${promotion_evidence}"

decision="$(
    python3 - "${promotion_evidence}" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])
PY
)"
if [[ "${decision}" != "candidate_promoted" ]]; then
    echo "PPO checkpoint rejected; deterministic baseline retained." >&2
    exit 4
fi
