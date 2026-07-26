#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DR_ANMAR_RUNTIME_ROOT="${DR_ANMAR_ROOT:-${HOME}/dr_anmar}"
DR_ANMAR_ISAACLAB_ROOT="${DR_ANMAR_ISAACLAB_ROOT:-${DR_ANMAR_RUNTIME_ROOT}/physics-next/IsaacLab}"
DR_ANMAR_ISAAC_PYTHON="${DR_ANMAR_ISAAC_PYTHON:-${DR_ANMAR_RUNTIME_ROOT}/physics-next/env_isaaclab/bin/python}"
DR_ANMAR_LEARNING_OUTPUT="${DR_ANMAR_LEARNING_OUTPUT:-${REPO_ROOT}/output/dranmar-learning}"
DR_ANMAR_TASK="${DR_ANMAR_TASK:-DrAnmar-Reach-PSM-IK-Rel-v0}"
DR_ANMAR_NUM_ENVS="${DR_ANMAR_NUM_ENVS:-1200}"
DR_ANMAR_SEED="${DR_ANMAR_SEED:-17}"

export DR_ANMAR_ISAACLAB_ROOT
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONPATH="${REPO_ROOT}/source/extensions/orbit.surgical.tasks:${REPO_ROOT}/source/extensions/orbit.surgical.assets:${PYTHONPATH:-}"

usage() {
    echo "Usage: $0 {validate|list|smoke|benchmark|sweep|pretrain|train|play|tqta-start|tqta-ingest|tqta-report} [arguments]"
    echo "  benchmark [task] [num_envs] [iterations] [output]"
    echo "  sweep     [task] [iterations] [comma-separated env counts] [output]"
    echo "  pretrain  [task] [num_envs] [updates] [validation_frames] [output]"
    echo "  train     [task] [num_envs] [iterations] [output]"
    echo "  play      CHECKPOINT [task] [num_envs] [frames] [output]"
    echo "  tqta-start  [task] [tracker]"
    echo "  tqta-ingest TRACKER EVIDENCE_JSON..."
    echo "  tqta-report [tracker]"
}

require_runtime() {
    if [[ ! -x "${DR_ANMAR_ISAAC_PYTHON}" ]]; then
        echo "error: executable Isaac Python not found: ${DR_ANMAR_ISAAC_PYTHON}" >&2
        exit 2
    fi
    if [[ ! -d "${DR_ANMAR_ISAACLAB_ROOT}/scripts/benchmarks" ]]; then
        echo "error: Isaac Lab checkout not found: ${DR_ANMAR_ISAACLAB_ROOT}" >&2
        exit 2
    fi
}

run_train_benchmark() {
    local task="$1"
    local num_envs="$2"
    local iterations="$3"
    local output="$4"
    shift 4
    mkdir -p "${output}"
    "${DR_ANMAR_ISAAC_PYTHON}" "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" train \
        --task "${task}" \
        --num_envs "${num_envs}" \
        --max_iterations "${iterations}" \
        --seed "${DR_ANMAR_SEED}" \
        --benchmark_formatter schema,json \
        --output_path "${output}" \
        "$@"
}

command="${1:-}"
case "${command}" in
    validate)
        python3 "${REPO_ROOT}/scripts/validate_dranmar_learning.py"
        ;;
    list)
        require_runtime
        "${DR_ANMAR_ISAAC_PYTHON}" "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" list
        ;;
    smoke)
        require_runtime
        run_train_benchmark "${DR_ANMAR_TASK}" 64 1 "${DR_ANMAR_LEARNING_OUTPUT}/smoke"
        ;;
    benchmark)
        require_runtime
        run_train_benchmark \
            "${2:-${DR_ANMAR_TASK}}" \
            "${3:-${DR_ANMAR_NUM_ENVS}}" \
            "${4:-10}" \
            "${5:-${DR_ANMAR_LEARNING_OUTPUT}/benchmark}"
        ;;
    sweep)
        require_runtime
        task="${2:-${DR_ANMAR_TASK}}"
        iterations="${3:-10}"
        counts="${4:-128,256,512,1024}"
        output="${5:-${DR_ANMAR_LEARNING_OUTPUT}/sweep}"
        IFS=',' read -r -a env_counts <<< "${counts}"
        for count in "${env_counts[@]}"; do
            run_train_benchmark "${task}" "${count}" "${iterations}" "${output}/${count}-envs"
        done
        ;;
    pretrain)
        require_runtime
        task="${2:-${DR_ANMAR_TASK}}"
        num_envs="${3:-${DR_ANMAR_NUM_ENVS}}"
        updates="${4:-400}"
        validation_frames="${5:-500}"
        output="${6:-${DR_ANMAR_LEARNING_OUTPUT}/pretrain}"
        mkdir -p "${output}"
        "${DR_ANMAR_ISAAC_PYTHON}" \
            "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" pretrain \
            --task "${task}" \
            --num_envs "${num_envs}" \
            --updates "${updates}" \
            --validation_frames "${validation_frames}" \
            --seed "${DR_ANMAR_SEED}" \
            --benchmark_formatter schema,json \
            --output_path "${output}"
        ;;
    train)
        require_runtime
        run_train_benchmark \
            "${2:-${DR_ANMAR_TASK}}" \
            "${3:-${DR_ANMAR_NUM_ENVS}}" \
            "${4:-1200}" \
            "${5:-${DR_ANMAR_LEARNING_OUTPUT}/train}" \
            --check_success \
            --success_threshold 0.95 \
            --success_window 10
        ;;
    play)
        require_runtime
        checkpoint="${2:-}"
        if [[ -z "${checkpoint}" ]]; then
            usage
            exit 2
        fi
        task="${3:-${DR_ANMAR_TASK}}"
        num_envs="${4:-256}"
        frames="${5:-500}"
        output="${6:-${DR_ANMAR_LEARNING_OUTPUT}/play}"
        mkdir -p "${output}"
        "${DR_ANMAR_ISAAC_PYTHON}" "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" play \
            --task "${task}" \
            --checkpoint "${checkpoint}" \
            --num_envs "${num_envs}" \
            --num_frames "${frames}" \
            --seed "${DR_ANMAR_SEED}" \
            --benchmark_formatter schema,json \
            --output_path "${output}"
        ;;
    tqta-start)
        task="${2:-${DR_ANMAR_TASK}}"
        tracker="${3:-${DR_ANMAR_LEARNING_OUTPUT}/tqta/${task}.json}"
        python3 "${REPO_ROOT}/scripts/dr_anmar_tqta.py" start \
            --task "${task}" \
            --tracker "${tracker}"
        ;;
    tqta-ingest)
        tracker="${2:-}"
        if [[ -z "${tracker}" || "$#" -lt 3 ]]; then
            usage
            exit 2
        fi
        shift 2
        python3 "${REPO_ROOT}/scripts/dr_anmar_tqta.py" ingest \
            --tracker "${tracker}" \
            "$@"
        ;;
    tqta-report)
        tracker="${2:-${DR_ANMAR_LEARNING_OUTPUT}/tqta/${DR_ANMAR_TASK}.json}"
        python3 "${REPO_ROOT}/scripts/dr_anmar_tqta.py" report \
            --tracker "${tracker}"
        ;;
    *)
        usage
        exit 2
        ;;
esac
