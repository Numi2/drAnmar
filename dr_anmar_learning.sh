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
    echo "Usage: $0 {validate|list|probe|controller-sweep|handover-sweep|smoke|benchmark|sweep|pretrain|train|receiver-attempt-bootstrap|receiver-attempt-update|receiver-selector-bootstrap|receiver-selector-update|receiver-selector-sweep|receiver-retry-sweep|receiver-retry-candidate|receiver-retry-portfolio|attempt-handover|selector-handover|portfolio-handover|promoted-handover|play|record|tqta-start|tqta-ingest|tqta-report} [arguments]"
    echo "  probe     [task] [num_envs] [frames] [output]"
    echo "  controller-sweep [task] [num_envs] [frames] [parameter] [comma-values] [output]"
    echo "  handover-sweep [task] [num_envs] [frames] [receiver-arc-values] [output]"
    echo "  benchmark [task] [num_envs] [iterations] [output]"
    echo "  sweep     [task] [iterations] [comma-separated env counts] [output]"
    echo "  pretrain  [task] [num_envs] [updates] [validation_frames] [output]"
    echo "  train     [task] [num_envs] [iterations] [output]"
    echo "  receiver-attempt-bootstrap BASE CANDIDATE OUTPUT DATASET..."
    echo "  receiver-attempt-update CHECKPOINT OUTPUT ROLLOUT..."
    echo "  receiver-selector-bootstrap BASE CANDIDATE OUTPUT DATASET..."
    echo "  receiver-selector-update CHECKPOINT OUTPUT DATASET..."
    echo "  receiver-selector-sweep DATASET [num_envs] [frames] [output] [task] [stream_offset]"
    echo "  receiver-retry-sweep DATASET [num_envs] [frames] [output] [task] [stream_offset]"
    echo "  receiver-retry-candidate INDEX DATASET [num_envs] [frames] [output] [task] [stream_offset]"
    echo "  receiver-retry-portfolio BASE CANDIDATE OUTPUT DATASET..."
    echo "  attempt-handover ATTEMPT_CHECKPOINT [num_envs] [frames] [output] [task]"
    echo "  selector-handover SELECTOR_CHECKPOINT [num_envs] [frames] [output] [task]"
    echo "  portfolio-handover PORTFOLIO_CHECKPOINT [num_envs] [frames] [output] [task] [dataset]"
    echo "  promoted-handover [num_envs] [frames] [output] [task]"
    echo "  play      CHECKPOINT [task] [num_envs] [frames] [output]"
    echo "  record    CHECKPOINT [task] [frames] [output] [chunk_frames]"
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

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{print $1}'
    else
        shasum -a 256 "${path}" | awk '{print $1}'
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
    probe)
        require_runtime
        task="${2:-${DR_ANMAR_TASK}}"
        num_envs="${3:-${DR_ANMAR_NUM_ENVS}}"
        frames="${4:-10}"
        output="${5:-${DR_ANMAR_LEARNING_OUTPUT}/probe}"
        mkdir -p "${output}"
        "${DR_ANMAR_ISAAC_PYTHON}" \
            "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" probe \
            --task "${task}" \
            --num_envs "${num_envs}" \
            --num_frames "${frames}" \
            --seed "${DR_ANMAR_SEED}" \
            --benchmark_formatter schema,json \
            --output_path "${output}"
        ;;
    controller-sweep)
        require_runtime
        task="${2:-DrAnmar-Lift-Block-PSM-IK-Rel-v0}"
        num_envs="${3:-${DR_ANMAR_NUM_ENVS}}"
        frames="${4:-500}"
        parameter="${5:-close_distance}"
        values="${6:-0.001,0.002,0.003,0.004,0.005,0.006}"
        output="${7:-${DR_ANMAR_LEARNING_OUTPUT}/controller-sweep}"
        mkdir -p "${output}"
        "${DR_ANMAR_ISAAC_PYTHON}" \
            "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" controller-sweep \
            --task "${task}" \
            --num_envs "${num_envs}" \
            --num_frames "${frames}" \
            --parameter "${parameter}" \
            --values="${values}" \
            --seed "${DR_ANMAR_SEED}" \
            --benchmark_formatter schema,json \
            --output_path "${output}"
        ;;
    handover-sweep)
        require_runtime
        task="${2:-DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0}"
        num_envs="${3:-${DR_ANMAR_NUM_ENVS}}"
        frames="${4:-1000}"
        values="${5:-0.50,0.55,0.60,0.65,0.70,0.75}"
        output="${6:-${DR_ANMAR_LEARNING_OUTPUT}/handover-sweep}"
        mkdir -p "${output}"
        video_args=()
        if [[ "${DR_ANMAR_HANDOVER_VIDEO:-0}" == "1" ]]; then
            video_args=(
                --video
                --video_env_index
                "${DR_ANMAR_HANDOVER_VIDEO_ENV_INDEX:-0}"
                --video_width
                "${DR_ANMAR_HANDOVER_VIDEO_WIDTH:-1280}"
                --video_height
                "${DR_ANMAR_HANDOVER_VIDEO_HEIGHT:-720}"
            )
        fi
        "${DR_ANMAR_ISAAC_PYTHON}" \
            "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" handover-sweep \
            --task "${task}" \
            --num_envs "${num_envs}" \
            --num_frames "${frames}" \
            --parameter "${DR_ANMAR_HANDOVER_SWEEP_PARAMETER:-receiver_arc_fraction}" \
            --values="${values}" \
            --seed "${DR_ANMAR_SEED}" \
            --benchmark_formatter schema,json \
            --output_path "${output}" \
            "${video_args[@]}"
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
        updates="${4:-32}"
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
        checkpoint_args=()
        learning_rate_args=()
        if [[ -n "${DR_ANMAR_INIT_CHECKPOINT:-}" ]]; then
            checkpoint_args=(
                --checkpoint
                "${DR_ANMAR_INIT_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_POLICY_LEARNING_RATE:-}" ]]; then
            learning_rate_args=(
                --learning_rate
                "${DR_ANMAR_POLICY_LEARNING_RATE}"
            )
        fi
        giver_adaptation_args=()
        if [[ "${DR_ANMAR_HANDOVER_GIVER_ADAPTATION:-0}" == "1" ]]; then
            giver_adaptation_args=(--handover_giver_adaptation)
        fi
        run_train_benchmark \
            "${2:-${DR_ANMAR_TASK}}" \
            "${3:-${DR_ANMAR_NUM_ENVS}}" \
            "${4:-1200}" \
            "${5:-${DR_ANMAR_LEARNING_OUTPUT}/train}" \
            --check_success \
            --success_threshold "${DR_ANMAR_SUCCESS_THRESHOLD:-0.95}" \
            --success_window 10 \
            "${checkpoint_args[@]}" \
            "${learning_rate_args[@]}" \
            "${giver_adaptation_args[@]}"
        ;;
    receiver-attempt-bootstrap)
        require_runtime
        base_checkpoint="${2:-}"
        candidate_checkpoint="${3:-}"
        output="${4:-}"
        if [[ -z "${base_checkpoint}" || -z "${candidate_checkpoint}" || -z "${output}" || "$#" -lt 5 ]]; then
            usage
            exit 2
        fi
        shift 4
        dataset_args=()
        for dataset in "$@"; do
            dataset_args+=(--dataset "${dataset}")
        done
        "${DR_ANMAR_ISAAC_PYTHON}" \
            "${REPO_ROOT}/scripts/train_dranmar_receiver_attempt_ppo.py" \
            bootstrap \
            --base_checkpoint "${base_checkpoint}" \
            --candidate_checkpoint "${candidate_checkpoint}" \
            --output "${output}" \
            "${dataset_args[@]}"
        ;;
    receiver-attempt-update)
        require_runtime
        attempt_checkpoint="${2:-}"
        output="${3:-}"
        if [[ -z "${attempt_checkpoint}" || -z "${output}" || "$#" -lt 4 ]]; then
            usage
            exit 2
        fi
        shift 3
        rollout_args=()
        for rollout in "$@"; do
            rollout_args+=(--rollout "${rollout}")
        done
        "${DR_ANMAR_ISAAC_PYTHON}" \
            "${REPO_ROOT}/scripts/train_dranmar_receiver_attempt_ppo.py" \
            update \
            --checkpoint "${attempt_checkpoint}" \
            --output "${output}" \
            "${rollout_args[@]}"
        ;;
    receiver-selector-bootstrap)
        require_runtime
        base_checkpoint="${2:-}"
        candidate_checkpoint="${3:-}"
        output="${4:-}"
        if [[ -z "${base_checkpoint}" || -z "${candidate_checkpoint}" || -z "${output}" || "$#" -lt 5 ]]; then
            usage
            exit 2
        fi
        shift 4
        dataset_args=()
        for dataset in "$@"; do
            dataset_args+=(--dataset "${dataset}")
        done
        "${DR_ANMAR_ISAAC_PYTHON}" \
            "${REPO_ROOT}/scripts/train_dranmar_receiver_context_selector.py" \
            bootstrap \
            --base_checkpoint "${base_checkpoint}" \
            --candidate_checkpoint "${candidate_checkpoint}" \
            --output "${output}" \
            "${dataset_args[@]}"
        ;;
    receiver-selector-update)
        require_runtime
        selector_checkpoint="${2:-}"
        output="${3:-}"
        if [[ -z "${selector_checkpoint}" || -z "${output}" || "$#" -lt 4 ]]; then
            usage
            exit 2
        fi
        shift 3
        dataset_args=()
        for dataset in "$@"; do
            dataset_args+=(--dataset "${dataset}")
        done
        "${DR_ANMAR_ISAAC_PYTHON}" \
            "${REPO_ROOT}/scripts/train_dranmar_receiver_context_selector.py" \
            update \
            --checkpoint "${selector_checkpoint}" \
            --output "${output}" \
            "${dataset_args[@]}"
        ;;
    receiver-retry-portfolio)
        require_runtime
        base_checkpoint="${2:-}"
        candidate_checkpoint="${3:-}"
        output="${4:-}"
        if [[ -z "${base_checkpoint}" || -z "${candidate_checkpoint}" || -z "${output}" || "$#" -lt 5 ]]; then
            usage
            exit 2
        fi
        shift 4
        dataset_args=()
        for dataset in "$@"; do
            dataset_args+=(--dataset "${dataset}")
        done
        "${DR_ANMAR_ISAAC_PYTHON}" \
            "${REPO_ROOT}/scripts/build_dranmar_receiver_retry_portfolio.py" \
            --base_checkpoint "${base_checkpoint}" \
            --candidate_checkpoint "${candidate_checkpoint}" \
            --output "${output}" \
            "${dataset_args[@]}"
        ;;
    receiver-selector-sweep)
        dataset="${2:-}"
        if [[ -z "${dataset}" ]]; then
            usage
            exit 2
        fi
        exec env \
            DR_ANMAR_PROMOTED_ALLOW_SELECTOR_SWEEP=1 \
            DR_ANMAR_RECEIVER_RECOVERY_DATASET="${dataset}" \
            DR_ANMAR_SEED_STREAM_OFFSET="${7:-0}" \
            "${BASH_SOURCE[0]}" promoted-handover \
            "${3:-1200}" \
            "${4:-2000}" \
            "${5:-${DR_ANMAR_LEARNING_OUTPUT}/receiver-selector-sweep}" \
            "${6:-DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0}"
        ;;
    receiver-retry-sweep)
        dataset="${2:-}"
        if [[ -z "${dataset}" ]]; then
            usage
            exit 2
        fi
        exec env \
            DR_ANMAR_PROMOTED_ALLOW_RETRY_SWEEP=1 \
            DR_ANMAR_RECEIVER_RECOVERY_DATASET="${dataset}" \
            DR_ANMAR_SEED_STREAM_OFFSET="${7:-0}" \
            "${BASH_SOURCE[0]}" promoted-handover \
            "${3:-256}" \
            "${4:-2000}" \
            "${5:-${DR_ANMAR_LEARNING_OUTPUT}/receiver-retry-sweep}" \
            "${6:-DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0}"
        ;;
    receiver-retry-candidate)
        candidate_index="${2:-}"
        dataset="${3:-}"
        if [[ -z "${candidate_index}" || -z "${dataset}" ]]; then
            usage
            exit 2
        fi
        exec env \
            DR_ANMAR_PROMOTED_ALLOW_RETRY_CANDIDATE=1 \
            DR_ANMAR_RECEIVER_RETRY_CANDIDATE_INDEX="${candidate_index}" \
            DR_ANMAR_RECEIVER_RECOVERY_DATASET="${dataset}" \
            DR_ANMAR_SEED_STREAM_OFFSET="${8:-0}" \
            "${BASH_SOURCE[0]}" promoted-handover \
            "${4:-256}" \
            "${5:-2000}" \
            "${6:-${DR_ANMAR_LEARNING_OUTPUT}/receiver-retry-candidate-${candidate_index}}" \
            "${7:-DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0}"
        ;;
    attempt-handover)
        attempt_checkpoint="${2:-}"
        if [[ -z "${attempt_checkpoint}" ]]; then
            usage
            exit 2
        fi
        exec env \
            DR_ANMAR_PROMOTED_ALLOW_ATTEMPT=1 \
            DR_ANMAR_RECEIVER_ATTEMPT_CHECKPOINT="${attempt_checkpoint}" \
            "${BASH_SOURCE[0]}" promoted-handover \
            "${3:-1200}" \
            "${4:-2000}" \
            "${5:-${DR_ANMAR_LEARNING_OUTPUT}/attempt-handover}" \
            "${6:-DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0}"
        ;;
    selector-handover)
        selector_checkpoint="${2:-}"
        if [[ -z "${selector_checkpoint}" ]]; then
            usage
            exit 2
        fi
        exec env \
            DR_ANMAR_PROMOTED_ALLOW_SELECTOR=1 \
            DR_ANMAR_RECEIVER_CONTEXT_SELECTOR_CHECKPOINT="${selector_checkpoint}" \
            "${BASH_SOURCE[0]}" promoted-handover \
            "${3:-1200}" \
            "${4:-2000}" \
            "${5:-${DR_ANMAR_LEARNING_OUTPUT}/selector-handover}" \
            "${6:-DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0}"
        ;;
    portfolio-handover)
        portfolio_checkpoint="${2:-}"
        if [[ -z "${portfolio_checkpoint}" ]]; then
            usage
            exit 2
        fi
        exec env \
            DR_ANMAR_PROMOTED_ALLOW_PORTFOLIO=1 \
            DR_ANMAR_RECEIVER_RETRY_PORTFOLIO_CHECKPOINT="${portfolio_checkpoint}" \
            DR_ANMAR_RECEIVER_RECOVERY_DATASET="${7:-}" \
            "${BASH_SOURCE[0]}" promoted-handover \
            "${3:-256}" \
            "${4:-2000}" \
            "${5:-${DR_ANMAR_LEARNING_OUTPUT}/portfolio-handover}" \
            "${6:-DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0}"
        ;;
    promoted-handover)
        require_runtime
        base_sha256="f33e41883f80f4dd791d0033568a4241bf366adcf2eb739c20c9ffd9ab568aad"
        receiver_sha256="582669a8a1c71bace2bdd3cb8e3fcdf12a64bb077e82f87a1a5ce7cc0ad64b79"
        base_checkpoint="${DR_ANMAR_RUNTIME_ROOT}/repro/recovery80/immutable/${base_sha256}/model.pt"
        receiver_checkpoint="${DR_ANMAR_RUNTIME_ROOT}/repro/recovery80/immutable/${receiver_sha256}/model.pt"
        num_envs="${2:-1200}"
        frames="${3:-2000}"
        output="${4:-${DR_ANMAR_LEARNING_OUTPUT}/promoted-handover}"
        task="${5:-DrAnmar-Handover-Needle-Dual-PSM-IK-Rel-v0}"
        if [[ ! -f "${base_checkpoint}" ]]; then
            echo "error: promoted base checkpoint not found: ${base_checkpoint}" >&2
            exit 2
        fi
        if [[ ! -f "${receiver_checkpoint}" ]]; then
            echo "error: promoted receiver checkpoint not found: ${receiver_checkpoint}" >&2
            exit 2
        fi
        if [[ "$(sha256_file "${base_checkpoint}")" != "${base_sha256}" ]]; then
            echo "error: promoted base checkpoint hash mismatch: ${base_checkpoint}" >&2
            exit 2
        fi
        if [[ "$(sha256_file "${receiver_checkpoint}")" != "${receiver_sha256}" ]]; then
            echo "error: promoted receiver checkpoint hash mismatch: ${receiver_checkpoint}" >&2
            exit 2
        fi
        attempt_checkpoint_env=""
        attempt_stochastic_env=0
        attempt_dataset_env=""
        selector_checkpoint_env=""
        retry_portfolio_checkpoint_env=""
        custody_confirmation_steps_env=0
        receiver_disable_retries_env=1
        receiver_recovery_dataset_env=""
        selector_sweep_random_env=0
        selector_sweep_replicas_env=""
        selector_sweep_sobol_seed_env=""
        selector_sweep_id_env=""
        selector_sweep_dataset_env=""
        selector_seed_stream_offset_env=""
        retry_candidate_sweep_env=0
        retry_candidate_index_env=""
        if [[ "${DR_ANMAR_PROMOTED_ALLOW_ATTEMPT:-0}" == "1" ]]; then
            attempt_checkpoint_env="${DR_ANMAR_RECEIVER_ATTEMPT_CHECKPOINT:-}"
            attempt_stochastic_env="${DR_ANMAR_RECEIVER_ATTEMPT_STOCHASTIC:-0}"
            attempt_dataset_env="${DR_ANMAR_RECEIVER_ATTEMPT_DATASET:-}"
            if [[ -z "${attempt_checkpoint_env}" ]]; then
                echo "error: attempt-handover requires an attempt checkpoint" >&2
                exit 2
            fi
        fi
        if [[ "${DR_ANMAR_PROMOTED_ALLOW_SELECTOR:-0}" == "1" ]]; then
            selector_checkpoint_env="${DR_ANMAR_RECEIVER_CONTEXT_SELECTOR_CHECKPOINT:-}"
            if [[ -z "${selector_checkpoint_env}" ]]; then
                echo "error: selector-handover requires a selector checkpoint" >&2
                exit 2
            fi
        fi
        if [[ "${DR_ANMAR_PROMOTED_ALLOW_PORTFOLIO:-0}" == "1" ]]; then
            retry_portfolio_checkpoint_env="${DR_ANMAR_RECEIVER_RETRY_PORTFOLIO_CHECKPOINT:-}"
            receiver_disable_retries_env=0
            receiver_recovery_dataset_env="${DR_ANMAR_RECEIVER_RECOVERY_DATASET:-}"
            if [[ -z "${retry_portfolio_checkpoint_env}" ]]; then
                echo "error: portfolio-handover requires a portfolio checkpoint" >&2
                exit 2
            fi
        fi
        if [[ "${DR_ANMAR_PROMOTED_ALLOW_SELECTOR_SWEEP:-0}" == "1" ]]; then
            selector_sweep_random_env=1
            selector_sweep_replicas_env=16
            selector_sweep_sobol_seed_env=104730
            selector_sweep_id_env=attempt-selector-common16-new-v1
            selector_sweep_dataset_env="${DR_ANMAR_RECEIVER_RECOVERY_DATASET:-}"
            receiver_recovery_dataset_env="${selector_sweep_dataset_env}"
            selector_seed_stream_offset_env="${DR_ANMAR_SEED_STREAM_OFFSET:-0}"
            if [[ -z "${selector_sweep_dataset_env}" ]]; then
                echo "error: receiver-selector-sweep requires a dataset" >&2
                exit 2
            fi
        fi
        if [[ "${DR_ANMAR_PROMOTED_ALLOW_RETRY_SWEEP:-0}" == "1" ]]; then
            retry_candidate_sweep_env=1
            receiver_disable_retries_env=0
            selector_sweep_replicas_env=16
            selector_sweep_id_env=receiver-retry-common16-v1
            receiver_recovery_dataset_env="${DR_ANMAR_RECEIVER_RECOVERY_DATASET:-}"
            selector_seed_stream_offset_env="${DR_ANMAR_SEED_STREAM_OFFSET:-0}"
            if [[ -z "${receiver_recovery_dataset_env}" ]]; then
                echo "error: receiver-retry-sweep requires a dataset" >&2
                exit 2
            fi
        fi
        if [[ "${DR_ANMAR_PROMOTED_ALLOW_RETRY_CANDIDATE:-0}" == "1" ]]; then
            retry_candidate_index_env="${DR_ANMAR_RECEIVER_RETRY_CANDIDATE_INDEX:-}"
            receiver_disable_retries_env=0
            receiver_recovery_dataset_env="${DR_ANMAR_RECEIVER_RECOVERY_DATASET:-}"
            selector_sweep_id_env="receiver-retry-candidate-${retry_candidate_index_env}-v1"
            selector_seed_stream_offset_env="${DR_ANMAR_SEED_STREAM_OFFSET:-0}"
            if [[ -z "${retry_candidate_index_env}" || -z "${receiver_recovery_dataset_env}" ]]; then
                echo "error: receiver-retry-candidate requires an index and dataset" >&2
                exit 2
            fi
        fi
        exec env \
            DR_ANMAR_POLICY_RESIDUAL_SCALE=0.03 \
            DR_ANMAR_POLICY_PICKUP_VERTICAL_ACTION_LIMIT=0.01 \
            DR_ANMAR_POLICY_PICKUP_INITIAL_VERTICAL_ACTION_LIMIT=0.01 \
            DR_ANMAR_POLICY_CARRY_LATERAL_ACTION_LIMIT=0.06 \
            DR_ANMAR_POLICY_CARRY_LATERAL_RAMP_HEIGHT=0.01 \
            DR_ANMAR_POLICY_PRESENTATION_FRACTION_FROM_GIVER=0.35 \
            DR_ANMAR_POLICY_PRESENTATION_HEIGHT_IN_ROBOT_FRAME=-0.13 \
            DR_ANMAR_POLICY_GIVER_CLOSE_DISTANCE=0.005 \
            DR_ANMAR_POLICY_GIVER_LIFT_CONTACT_FORCE_THRESHOLD_N=0.01 \
            DR_ANMAR_POLICY_GIVER_PRE_LIFT_MIN_CONTACT_JAWS=2 \
            DR_ANMAR_POLICY_GIVER_LIFT_ON_LIVE_CONTACT=1 \
            DR_ANMAR_PICKUP_RECOVERY=1 \
            DR_ANMAR_PICKUP_RECOVERY_CHECKPOINT= \
            DR_ANMAR_PICKUP_RECOVERY_FIXED_CORRECTION="-0.000964653600000,0.000437389650000,-0.000506129775000,0.458332861313,1.03786434447,0.981207699374" \
            DR_ANMAR_PICKUP_RECOVERY_FIXED_CORRECTION_AFTER_FIRST_RETRY= \
            DR_ANMAR_PICKUP_RECOVERY_CORRECTION_CANDIDATES= \
            DR_ANMAR_PICKUP_RECOVERY_RANDOM_CORRECTIONS=0 \
            DR_ANMAR_PICKUP_RECOVERY_POSITION_CAP_M=0.001875 \
            DR_ANMAR_PICKUP_RECOVERY_ORIENTATION_CAP_DEG=1.5 \
            DR_ANMAR_RECEIVER_RECOVERY=1 \
            DR_ANMAR_RECEIVER_DISABLE_RETRIES="${receiver_disable_retries_env}" \
            DR_ANMAR_RECEIVER_RECOVERY_CHECKPOINT= \
            DR_ANMAR_RECEIVER_CANDIDATE_VALUE_CHECKPOINT="${receiver_checkpoint}" \
            DR_ANMAR_RECEIVER_CANDIDATE_LOCAL_REFINEMENT=1 \
            DR_ANMAR_RECEIVER_CANDIDATE_MIN_LOGIT_ADVANTAGE=0.0 \
            DR_ANMAR_RECEIVER_CANDIDATE_FIRST_ATTEMPT=1 \
            DR_ANMAR_RECEIVER_GATE_STEP=50 \
            DR_ANMAR_RECEIVER_RETRY_GATE_CHECKPOINT= \
            DR_ANMAR_RECEIVER_STABILIZATION_GATE_CHECKPOINT= \
            DR_ANMAR_RECEIVER_STABILIZE_GIVER_DURING_ACQUISITION=0 \
            DR_ANMAR_RECEIVER_SECURE_SETTLE_STEPS=0 \
            DR_ANMAR_RECEIVER_CUSTODY_CONFIRMATION_STEPS="${custody_confirmation_steps_env}" \
            DR_ANMAR_RECEIVER_RECOVERY_FIXED_CORRECTION= \
            DR_ANMAR_RECEIVER_RECOVERY_RANDOM_CORRECTIONS="${selector_sweep_random_env}" \
            DR_ANMAR_RECEIVER_RECOVERY_SWEEP_REPLICAS="${selector_sweep_replicas_env}" \
            DR_ANMAR_RECEIVER_RETRY_CANDIDATE_SWEEP="${retry_candidate_sweep_env}" \
            DR_ANMAR_RECEIVER_RETRY_CANDIDATE_INDEX="${retry_candidate_index_env}" \
            DR_ANMAR_RECEIVER_RECOVERY_SOBOL_SEED="${selector_sweep_sobol_seed_env}" \
            DR_ANMAR_RECEIVER_RECOVERY_SWEEP_ID="${selector_sweep_id_env}" \
            DR_ANMAR_RECEIVER_RECOVERY_DATASET="${receiver_recovery_dataset_env}" \
            DR_ANMAR_RECEIVER_RECOVERY_POSITION_CAP_M=0.0025 \
            DR_ANMAR_RECEIVER_RECOVERY_ORIENTATION_CAP_DEG=2.0 \
            DR_ANMAR_RECEIVER_RECOVERY_LOCAL_SOBOL_SEED=104748 \
            DR_ANMAR_RECEIVER_RECOVERY_LOCAL_POSITION_RADIUS_M=0.001 \
            DR_ANMAR_RECEIVER_RECOVERY_LOCAL_ORIENTATION_RADIUS_DEG=1.0 \
            DR_ANMAR_RECEIVER_ATTEMPT_CHECKPOINT="${attempt_checkpoint_env}" \
            DR_ANMAR_RECEIVER_ATTEMPT_STOCHASTIC="${attempt_stochastic_env}" \
            DR_ANMAR_RECEIVER_ATTEMPT_DATASET="${attempt_dataset_env}" \
            DR_ANMAR_RECEIVER_ATTEMPT_POSITION_CAP_M=0.001 \
            DR_ANMAR_RECEIVER_ATTEMPT_ORIENTATION_CAP_DEG=1.0 \
            DR_ANMAR_RECEIVER_CONTEXT_SELECTOR_CHECKPOINT="${selector_checkpoint_env}" \
            DR_ANMAR_RECEIVER_RETRY_PORTFOLIO_CHECKPOINT="${retry_portfolio_checkpoint_env}" \
            DR_ANMAR_SEED_STREAM_OFFSET="${selector_seed_stream_offset_env}" \
            "${BASH_SOURCE[0]}" play \
            "${base_checkpoint}" \
            "${task}" \
            "${num_envs}" \
            "${frames}" \
            "${output}"
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
        seed_stream_args=()
        if [[ -n "${DR_ANMAR_SEED_STREAM_OFFSET:-}" ]]; then
            seed_stream_args=(
                --seed_stream_offset
                "${DR_ANMAR_SEED_STREAM_OFFSET}"
            )
        fi
        residual_scale_args=()
        if [[ -n "${DR_ANMAR_POLICY_RESIDUAL_SCALE:-}" ]]; then
            residual_scale_args=(
                --residual_scale
                "${DR_ANMAR_POLICY_RESIDUAL_SCALE}"
            )
        fi
        pickup_vertical_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_PICKUP_VERTICAL_ACTION_LIMIT:-}" ]]; then
            pickup_vertical_action_limit_args=(
                --pickup_vertical_action_limit
                "${DR_ANMAR_POLICY_PICKUP_VERTICAL_ACTION_LIMIT}"
            )
        fi
        pickup_initial_vertical_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_PICKUP_INITIAL_VERTICAL_ACTION_LIMIT:-}" ]]; then
            pickup_initial_vertical_action_limit_args=(
                --pickup_initial_vertical_action_limit
                "${DR_ANMAR_POLICY_PICKUP_INITIAL_VERTICAL_ACTION_LIMIT}"
            )
        fi
        carry_lateral_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_CARRY_LATERAL_ACTION_LIMIT:-}" ]]; then
            carry_lateral_action_limit_args=(
                --carry_lateral_action_limit
                "${DR_ANMAR_POLICY_CARRY_LATERAL_ACTION_LIMIT}"
            )
        fi
        carry_lateral_ramp_height_args=()
        if [[ -n "${DR_ANMAR_POLICY_CARRY_LATERAL_RAMP_HEIGHT:-}" ]]; then
            carry_lateral_ramp_height_args=(
                --carry_lateral_ramp_height
                "${DR_ANMAR_POLICY_CARRY_LATERAL_RAMP_HEIGHT}"
            )
        fi
        presentation_fraction_from_giver_args=()
        if [[ -n "${DR_ANMAR_POLICY_PRESENTATION_FRACTION_FROM_GIVER:-}" ]]; then
            presentation_fraction_from_giver_args=(
                --presentation_fraction_from_giver
                "${DR_ANMAR_POLICY_PRESENTATION_FRACTION_FROM_GIVER}"
            )
        fi
        presentation_height_in_robot_frame_args=()
        if [[ -n "${DR_ANMAR_POLICY_PRESENTATION_HEIGHT_IN_ROBOT_FRAME:-}" ]]; then
            presentation_height_in_robot_frame_args=(
                --presentation_height_in_robot_frame
                "${DR_ANMAR_POLICY_PRESENTATION_HEIGHT_IN_ROBOT_FRAME}"
            )
        fi
        giver_close_distance_args=()
        if [[ -n "${DR_ANMAR_POLICY_GIVER_CLOSE_DISTANCE:-}" ]]; then
            giver_close_distance_args=(
                --giver_close_distance
                "${DR_ANMAR_POLICY_GIVER_CLOSE_DISTANCE}"
            )
        fi
        giver_lift_contact_force_threshold_args=()
        if [[ -n "${DR_ANMAR_POLICY_GIVER_LIFT_CONTACT_FORCE_THRESHOLD_N:-}" ]]; then
            giver_lift_contact_force_threshold_args=(
                --giver_lift_contact_force_threshold
                "${DR_ANMAR_POLICY_GIVER_LIFT_CONTACT_FORCE_THRESHOLD_N}"
            )
        fi
        giver_pre_lift_min_contact_jaws_args=()
        if [[ -n "${DR_ANMAR_POLICY_GIVER_PRE_LIFT_MIN_CONTACT_JAWS:-}" ]]; then
            giver_pre_lift_min_contact_jaws_args=(
                --giver_pre_lift_min_contact_jaws
                "${DR_ANMAR_POLICY_GIVER_PRE_LIFT_MIN_CONTACT_JAWS}"
            )
        fi
        giver_lift_on_live_contact_args=()
        if [[ -n "${DR_ANMAR_POLICY_GIVER_LIFT_ON_LIVE_CONTACT:-}" ]]; then
            case "${DR_ANMAR_POLICY_GIVER_LIFT_ON_LIVE_CONTACT}" in
                1) giver_lift_on_live_contact_args=(--giver_lift_on_live_contact) ;;
                0) giver_lift_on_live_contact_args=(--no-giver_lift_on_live_contact) ;;
                *)
                    echo "DR_ANMAR_POLICY_GIVER_LIFT_ON_LIVE_CONTACT must be 0 or 1" >&2
                    exit 2
                    ;;
                esac
        fi
        pickup_recovery_args=()
        if [[ -n "${DR_ANMAR_RECOVERY_DEMO_ROTATION_DEG:-}" ]]; then
            pickup_recovery_args+=(
                --recovery_demo_rotation_deg
                "${DR_ANMAR_RECOVERY_DEMO_ROTATION_DEG}"
            )
        fi
        if [[ "${DR_ANMAR_PICKUP_RECOVERY:-0}" == "1" ]]; then
            pickup_recovery_args+=(--pickup_recovery)
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_CHECKPOINT:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_checkpoint
                "${DR_ANMAR_PICKUP_RECOVERY_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_POSITION_CAP_M:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_position_cap
                "${DR_ANMAR_PICKUP_RECOVERY_POSITION_CAP_M}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_ORIENTATION_CAP_DEG:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_orientation_cap_deg
                "${DR_ANMAR_PICKUP_RECOVERY_ORIENTATION_CAP_DEG}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_FIXED_CORRECTION:-}" ]]; then
            pickup_recovery_args+=(
                "--pickup_recovery_fixed_correction=${DR_ANMAR_PICKUP_RECOVERY_FIXED_CORRECTION}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_FIXED_CORRECTION_AFTER_FIRST_RETRY:-}" ]]; then
            pickup_recovery_args+=(
                "--pickup_recovery_fixed_correction_after_first_retry=${DR_ANMAR_PICKUP_RECOVERY_FIXED_CORRECTION_AFTER_FIRST_RETRY}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_CORRECTION_CANDIDATES:-}" ]]; then
            pickup_recovery_args+=(
                "--pickup_recovery_correction_candidates=${DR_ANMAR_PICKUP_RECOVERY_CORRECTION_CANDIDATES}"
            )
        fi
        if [[ "${DR_ANMAR_PICKUP_RECOVERY_RANDOM_CORRECTIONS:-0}" == "1" ]]; then
            pickup_recovery_args+=(--pickup_recovery_random_corrections)
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_SWEEP_REPLICAS:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_sweep_replicas
                "${DR_ANMAR_PICKUP_RECOVERY_SWEEP_REPLICAS}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_SOBOL_START:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_sobol_start
                "${DR_ANMAR_PICKUP_RECOVERY_SOBOL_START}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_SOBOL_CANDIDATE:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_sobol_candidate
                "${DR_ANMAR_PICKUP_RECOVERY_SOBOL_CANDIDATE}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_LOCAL_SOBOL_CANDIDATE:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_local_sobol_candidate
                "${DR_ANMAR_PICKUP_RECOVERY_LOCAL_SOBOL_CANDIDATE}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_LOCAL_POSITION_RADIUS_M:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_local_position_radius
                "${DR_ANMAR_PICKUP_RECOVERY_LOCAL_POSITION_RADIUS_M}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_LOCAL_ORIENTATION_RADIUS_DEG:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_local_orientation_radius_deg
                "${DR_ANMAR_PICKUP_RECOVERY_LOCAL_ORIENTATION_RADIUS_DEG}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_SWEEP_ID:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_sweep_id
                "${DR_ANMAR_PICKUP_RECOVERY_SWEEP_ID}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_DATASET:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_dataset
                "${DR_ANMAR_PICKUP_RECOVERY_DATASET}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_RECOVERY:-0}" == "1" ]]; then
            pickup_recovery_args+=(--receiver_recovery)
        fi
        if [[ "${DR_ANMAR_RECEIVER_DISABLE_RETRIES:-0}" == "1" ]]; then
            pickup_recovery_args+=(--receiver_disable_retries)
        fi
        if [[ "${DR_ANMAR_RECEIVER_STABILIZE_GIVER_DURING_ACQUISITION:-0}" == "1" ]]; then
            pickup_recovery_args+=(
                --receiver_stabilize_giver_during_acquisition
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_GIVER_STABILIZATION_START_STEP:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_giver_stabilization_start_step
                "${DR_ANMAR_RECEIVER_GIVER_STABILIZATION_START_STEP}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_SECURE_SETTLE_STEPS:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_secure_settle_steps
                "${DR_ANMAR_RECEIVER_SECURE_SETTLE_STEPS}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_CUSTODY_CONFIRMATION_STEPS:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_custody_confirmation_steps
                "${DR_ANMAR_RECEIVER_CUSTODY_CONFIRMATION_STEPS}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_RETENTION_CONTACT_CENTERING:-0}" == "1" ]]; then
            pickup_recovery_args+=(
                --receiver_retention_contact_centering
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_CHECKPOINT:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_checkpoint
                "${DR_ANMAR_RECEIVER_RECOVERY_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_CANDIDATE_VALUE_CHECKPOINT:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_candidate_value_checkpoint
                "${DR_ANMAR_RECEIVER_CANDIDATE_VALUE_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_CONTEXT_SELECTOR_CHECKPOINT:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_context_selector_checkpoint
                "${DR_ANMAR_RECEIVER_CONTEXT_SELECTOR_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RETRY_PORTFOLIO_CHECKPOINT:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_retry_portfolio_checkpoint
                "${DR_ANMAR_RECEIVER_RETRY_PORTFOLIO_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_ATTEMPT_CHECKPOINT:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_attempt_checkpoint
                "${DR_ANMAR_RECEIVER_ATTEMPT_CHECKPOINT}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_ATTEMPT_STOCHASTIC:-0}" == "1" ]]; then
            pickup_recovery_args+=(--receiver_attempt_stochastic)
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_ATTEMPT_DATASET:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_attempt_dataset
                "${DR_ANMAR_RECEIVER_ATTEMPT_DATASET}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_ATTEMPT_POSITION_CAP_M:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_attempt_position_cap
                "${DR_ANMAR_RECEIVER_ATTEMPT_POSITION_CAP_M}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_ATTEMPT_ORIENTATION_CAP_DEG:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_attempt_orientation_cap_deg
                "${DR_ANMAR_RECEIVER_ATTEMPT_ORIENTATION_CAP_DEG}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_CANDIDATE_LOCAL_REFINEMENT:-0}" == "1" ]]; then
            pickup_recovery_args+=(
                --receiver_candidate_local_refinement
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_CANDIDATE_MIN_LOGIT_ADVANTAGE:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_candidate_min_logit_advantage
                "${DR_ANMAR_RECEIVER_CANDIDATE_MIN_LOGIT_ADVANTAGE}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_CANDIDATE_FIRST_ATTEMPT:-0}" == "1" ]]; then
            pickup_recovery_args+=(
                --receiver_candidate_first_attempt
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_GATE_STEP:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_gate_step
                "${DR_ANMAR_RECEIVER_GATE_STEP}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RETRY_GATE_CHECKPOINT:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_retry_gate_checkpoint
                "${DR_ANMAR_RECEIVER_RETRY_GATE_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RETRY_GATE_THRESHOLD:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_retry_gate_threshold
                "${DR_ANMAR_RECEIVER_RETRY_GATE_THRESHOLD}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_STABILIZATION_GATE_CHECKPOINT:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_stabilization_gate_checkpoint
                "${DR_ANMAR_RECEIVER_STABILIZATION_GATE_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_STABILIZATION_GATE_THRESHOLD:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_stabilization_gate_threshold
                "${DR_ANMAR_RECEIVER_STABILIZATION_GATE_THRESHOLD}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_POSITION_CAP_M:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_position_cap
                "${DR_ANMAR_RECEIVER_RECOVERY_POSITION_CAP_M}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_ORIENTATION_CAP_DEG:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_orientation_cap_deg
                "${DR_ANMAR_RECEIVER_RECOVERY_ORIENTATION_CAP_DEG}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_ACQUISITION_TIMEOUT_STEPS:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_acquisition_timeout_steps
                "${DR_ANMAR_RECEIVER_RECOVERY_ACQUISITION_TIMEOUT_STEPS}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_FIXED_CORRECTION:-}" ]]; then
            pickup_recovery_args+=(
                "--receiver_recovery_fixed_correction=${DR_ANMAR_RECEIVER_RECOVERY_FIXED_CORRECTION}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_RECOVERY_RANDOM_CORRECTIONS:-0}" == "1" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_random_corrections
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_SWEEP_REPLICAS:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_sweep_replicas
                "${DR_ANMAR_RECEIVER_RECOVERY_SWEEP_REPLICAS}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_RETRY_CANDIDATE_SWEEP:-0}" == "1" ]]; then
            pickup_recovery_args+=(
                --receiver_retry_candidate_sweep
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RETRY_CANDIDATE_INDEX:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_retry_candidate_index
                "${DR_ANMAR_RECEIVER_RETRY_CANDIDATE_INDEX}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_SOBOL_SEED:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_sobol_seed
                "${DR_ANMAR_RECEIVER_RECOVERY_SOBOL_SEED}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_SOBOL_START:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_sobol_start
                "${DR_ANMAR_RECEIVER_RECOVERY_SOBOL_START}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_SOBOL_CANDIDATE:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_sobol_candidate
                "${DR_ANMAR_RECEIVER_RECOVERY_SOBOL_CANDIDATE}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_SOBOL_CANDIDATE:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_local_sobol_candidate
                "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_SOBOL_CANDIDATE}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_SOBOL_SEED:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_local_sobol_seed
                "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_SOBOL_SEED}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_POSITION_RADIUS_M:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_local_position_radius
                "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_POSITION_RADIUS_M}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_ORIENTATION_RADIUS_DEG:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_local_orientation_radius_deg
                "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_ORIENTATION_RADIUS_DEG}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_SWEEP_ID:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_sweep_id
                "${DR_ANMAR_RECEIVER_RECOVERY_SWEEP_ID}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_DATASET:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_dataset
                "${DR_ANMAR_RECEIVER_RECOVERY_DATASET}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_GATE_DATASET:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_gate_dataset
                "${DR_ANMAR_RECEIVER_RECOVERY_GATE_DATASET}"
            )
        fi
        "${DR_ANMAR_ISAAC_PYTHON}" "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" play \
            --task "${task}" \
            --checkpoint "${checkpoint}" \
            --num_envs "${num_envs}" \
            --num_frames "${frames}" \
            --seed "${DR_ANMAR_SEED}" \
            "${seed_stream_args[@]}" \
            --benchmark_formatter schema,json \
            --output_path "${output}" \
            "${residual_scale_args[@]}" \
            "${pickup_vertical_action_limit_args[@]}" \
            "${pickup_initial_vertical_action_limit_args[@]}" \
            "${carry_lateral_action_limit_args[@]}" \
            "${carry_lateral_ramp_height_args[@]}" \
            "${presentation_fraction_from_giver_args[@]}" \
            "${presentation_height_in_robot_frame_args[@]}" \
            "${giver_close_distance_args[@]}" \
            "${giver_lift_contact_force_threshold_args[@]}" \
            "${giver_pre_lift_min_contact_jaws_args[@]}" \
            "${giver_lift_on_live_contact_args[@]}" \
            "${pickup_recovery_args[@]}"
        ;;
    record)
        require_runtime
        checkpoint="${2:-}"
        if [[ -z "${checkpoint}" ]]; then
            usage
            exit 2
        fi
        task="${3:-${DR_ANMAR_TASK}}"
        frames="${4:-500}"
        output="${5:-${DR_ANMAR_LEARNING_OUTPUT}/record}"
        chunk_frames="${6:-${frames}}"
        mkdir -p "${output}/videos"
        pickup_vertical_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_PICKUP_VERTICAL_ACTION_LIMIT:-}" ]]; then
            pickup_vertical_action_limit_args=(
                --pickup_vertical_action_limit
                "${DR_ANMAR_POLICY_PICKUP_VERTICAL_ACTION_LIMIT}"
            )
        fi
        pickup_initial_vertical_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_PICKUP_INITIAL_VERTICAL_ACTION_LIMIT:-}" ]]; then
            pickup_initial_vertical_action_limit_args=(
                --pickup_initial_vertical_action_limit
                "${DR_ANMAR_POLICY_PICKUP_INITIAL_VERTICAL_ACTION_LIMIT}"
            )
        fi
        carry_lateral_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_CARRY_LATERAL_ACTION_LIMIT:-}" ]]; then
            carry_lateral_action_limit_args=(
                --carry_lateral_action_limit
                "${DR_ANMAR_POLICY_CARRY_LATERAL_ACTION_LIMIT}"
            )
        fi
        carry_lateral_ramp_height_args=()
        if [[ -n "${DR_ANMAR_POLICY_CARRY_LATERAL_RAMP_HEIGHT:-}" ]]; then
            carry_lateral_ramp_height_args=(
                --carry_lateral_ramp_height
                "${DR_ANMAR_POLICY_CARRY_LATERAL_RAMP_HEIGHT}"
            )
        fi
        presentation_fraction_from_giver_args=()
        if [[ -n "${DR_ANMAR_POLICY_PRESENTATION_FRACTION_FROM_GIVER:-}" ]]; then
            presentation_fraction_from_giver_args=(
                --presentation_fraction_from_giver
                "${DR_ANMAR_POLICY_PRESENTATION_FRACTION_FROM_GIVER}"
            )
        fi
        presentation_height_in_robot_frame_args=()
        if [[ -n "${DR_ANMAR_POLICY_PRESENTATION_HEIGHT_IN_ROBOT_FRAME:-}" ]]; then
            presentation_height_in_robot_frame_args=(
                --presentation_height_in_robot_frame
                "${DR_ANMAR_POLICY_PRESENTATION_HEIGHT_IN_ROBOT_FRAME}"
            )
        fi
        giver_close_distance_args=()
        if [[ -n "${DR_ANMAR_POLICY_GIVER_CLOSE_DISTANCE:-}" ]]; then
            giver_close_distance_args=(
                --giver_close_distance
                "${DR_ANMAR_POLICY_GIVER_CLOSE_DISTANCE}"
            )
        fi
        giver_lift_contact_force_threshold_args=()
        if [[ -n "${DR_ANMAR_POLICY_GIVER_LIFT_CONTACT_FORCE_THRESHOLD_N:-}" ]]; then
            giver_lift_contact_force_threshold_args=(
                --giver_lift_contact_force_threshold
                "${DR_ANMAR_POLICY_GIVER_LIFT_CONTACT_FORCE_THRESHOLD_N}"
            )
        fi
        giver_pre_lift_min_contact_jaws_args=()
        if [[ -n "${DR_ANMAR_POLICY_GIVER_PRE_LIFT_MIN_CONTACT_JAWS:-}" ]]; then
            giver_pre_lift_min_contact_jaws_args=(
                --giver_pre_lift_min_contact_jaws
                "${DR_ANMAR_POLICY_GIVER_PRE_LIFT_MIN_CONTACT_JAWS}"
            )
        fi
        giver_lift_on_live_contact_args=()
        if [[ -n "${DR_ANMAR_POLICY_GIVER_LIFT_ON_LIVE_CONTACT:-}" ]]; then
            case "${DR_ANMAR_POLICY_GIVER_LIFT_ON_LIVE_CONTACT}" in
                1) giver_lift_on_live_contact_args=(--giver_lift_on_live_contact) ;;
                0) giver_lift_on_live_contact_args=(--no-giver_lift_on_live_contact) ;;
                *)
                    echo "DR_ANMAR_POLICY_GIVER_LIFT_ON_LIVE_CONTACT must be 0 or 1" >&2
                    exit 2
                    ;;
            esac
        fi
        recovery_record_args=()
        if [[ "${DR_ANMAR_PICKUP_RECOVERY:-0}" == "1" ]]; then
            recovery_record_args+=(--pickup_recovery)
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_CHECKPOINT:-}" ]]; then
            recovery_record_args+=(
                --pickup_recovery_checkpoint
                "${DR_ANMAR_PICKUP_RECOVERY_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_POSITION_CAP_M:-}" ]]; then
            recovery_record_args+=(
                --pickup_recovery_position_cap
                "${DR_ANMAR_PICKUP_RECOVERY_POSITION_CAP_M}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_ORIENTATION_CAP_DEG:-}" ]]; then
            recovery_record_args+=(
                --pickup_recovery_orientation_cap_deg
                "${DR_ANMAR_PICKUP_RECOVERY_ORIENTATION_CAP_DEG}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_FIXED_CORRECTION:-}" ]]; then
            recovery_record_args+=(
                "--pickup_recovery_fixed_correction=${DR_ANMAR_PICKUP_RECOVERY_FIXED_CORRECTION}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_FIXED_CORRECTION_AFTER_FIRST_RETRY:-}" ]]; then
            recovery_record_args+=(
                "--pickup_recovery_fixed_correction_after_first_retry=${DR_ANMAR_PICKUP_RECOVERY_FIXED_CORRECTION_AFTER_FIRST_RETRY}"
            )
        fi
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_CORRECTION_CANDIDATES:-}" ]]; then
            recovery_record_args+=(
                "--pickup_recovery_correction_candidates=${DR_ANMAR_PICKUP_RECOVERY_CORRECTION_CANDIDATES}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_RECOVERY:-0}" == "1" ]]; then
            recovery_record_args+=(--receiver_recovery)
        fi
        if [[ "${DR_ANMAR_RECEIVER_DISABLE_RETRIES:-0}" == "1" ]]; then
            recovery_record_args+=(--receiver_disable_retries)
        fi
        if [[ "${DR_ANMAR_RECEIVER_STABILIZE_GIVER_DURING_ACQUISITION:-0}" == "1" ]]; then
            recovery_record_args+=(
                --receiver_stabilize_giver_during_acquisition
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_GIVER_STABILIZATION_START_STEP:-}" ]]; then
            recovery_record_args+=(
                --receiver_giver_stabilization_start_step
                "${DR_ANMAR_RECEIVER_GIVER_STABILIZATION_START_STEP}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_SECURE_SETTLE_STEPS:-}" ]]; then
            recovery_record_args+=(
                --receiver_secure_settle_steps
                "${DR_ANMAR_RECEIVER_SECURE_SETTLE_STEPS}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_RETENTION_CONTACT_CENTERING:-0}" == "1" ]]; then
            recovery_record_args+=(
                --receiver_retention_contact_centering
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_CHECKPOINT:-}" ]]; then
            recovery_record_args+=(
                --receiver_recovery_checkpoint
                "${DR_ANMAR_RECEIVER_RECOVERY_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_CANDIDATE_VALUE_CHECKPOINT:-}" ]]; then
            recovery_record_args+=(
                --receiver_candidate_value_checkpoint
                "${DR_ANMAR_RECEIVER_CANDIDATE_VALUE_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_ATTEMPT_CHECKPOINT:-}" ]]; then
            recovery_record_args+=(
                --receiver_attempt_checkpoint
                "${DR_ANMAR_RECEIVER_ATTEMPT_CHECKPOINT}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_ATTEMPT_STOCHASTIC:-0}" == "1" ]]; then
            recovery_record_args+=(--receiver_attempt_stochastic)
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_ATTEMPT_POSITION_CAP_M:-}" ]]; then
            recovery_record_args+=(
                --receiver_attempt_position_cap
                "${DR_ANMAR_RECEIVER_ATTEMPT_POSITION_CAP_M}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_ATTEMPT_ORIENTATION_CAP_DEG:-}" ]]; then
            recovery_record_args+=(
                --receiver_attempt_orientation_cap_deg
                "${DR_ANMAR_RECEIVER_ATTEMPT_ORIENTATION_CAP_DEG}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_CANDIDATE_LOCAL_REFINEMENT:-0}" == "1" ]]; then
            recovery_record_args+=(
                --receiver_candidate_local_refinement
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_CANDIDATE_MIN_LOGIT_ADVANTAGE:-}" ]]; then
            recovery_record_args+=(
                --receiver_candidate_min_logit_advantage
                "${DR_ANMAR_RECEIVER_CANDIDATE_MIN_LOGIT_ADVANTAGE}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_CANDIDATE_FIRST_ATTEMPT:-0}" == "1" ]]; then
            recovery_record_args+=(
                --receiver_candidate_first_attempt
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_GATE_STEP:-}" ]]; then
            recovery_record_args+=(
                --receiver_gate_step
                "${DR_ANMAR_RECEIVER_GATE_STEP}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RETRY_GATE_CHECKPOINT:-}" ]]; then
            recovery_record_args+=(
                --receiver_retry_gate_checkpoint
                "${DR_ANMAR_RECEIVER_RETRY_GATE_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RETRY_GATE_THRESHOLD:-}" ]]; then
            recovery_record_args+=(
                --receiver_retry_gate_threshold
                "${DR_ANMAR_RECEIVER_RETRY_GATE_THRESHOLD}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_STABILIZATION_GATE_CHECKPOINT:-}" ]]; then
            recovery_record_args+=(
                --receiver_stabilization_gate_checkpoint
                "${DR_ANMAR_RECEIVER_STABILIZATION_GATE_CHECKPOINT}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_STABILIZATION_GATE_THRESHOLD:-}" ]]; then
            recovery_record_args+=(
                --receiver_stabilization_gate_threshold
                "${DR_ANMAR_RECEIVER_STABILIZATION_GATE_THRESHOLD}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_POSITION_CAP_M:-}" ]]; then
            recovery_record_args+=(
                --receiver_recovery_position_cap
                "${DR_ANMAR_RECEIVER_RECOVERY_POSITION_CAP_M}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_ORIENTATION_CAP_DEG:-}" ]]; then
            recovery_record_args+=(
                --receiver_recovery_orientation_cap_deg
                "${DR_ANMAR_RECEIVER_RECOVERY_ORIENTATION_CAP_DEG}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_SOBOL_SEED:-}" ]]; then
            recovery_record_args+=(
                --receiver_recovery_local_sobol_seed
                "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_SOBOL_SEED}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_POSITION_RADIUS_M:-}" ]]; then
            recovery_record_args+=(
                --receiver_recovery_local_position_radius
                "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_POSITION_RADIUS_M}"
            )
        fi
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_ORIENTATION_RADIUS_DEG:-}" ]]; then
            recovery_record_args+=(
                --receiver_recovery_local_orientation_radius_deg
                "${DR_ANMAR_RECEIVER_RECOVERY_LOCAL_ORIENTATION_RADIUS_DEG}"
            )
        fi
        "${DR_ANMAR_ISAAC_PYTHON}" "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" play \
            --task "${task}" \
            --checkpoint "${checkpoint}" \
            --num_envs 1 \
            --num_frames "${frames}" \
            --seed "${DR_ANMAR_SEED}" \
            --video \
            --stop_after_first_episode \
            --video_length "${chunk_frames}" \
            --video_chunk_length "${chunk_frames}" \
            --video_width "${DR_ANMAR_VIDEO_WIDTH:-640}" \
            --video_height "${DR_ANMAR_VIDEO_HEIGHT:-360}" \
            --video_folder "${output}/videos" \
            --benchmark_formatter schema,json \
            --output_path "${output}" \
            "${pickup_vertical_action_limit_args[@]}" \
            "${pickup_initial_vertical_action_limit_args[@]}" \
            "${carry_lateral_action_limit_args[@]}" \
            "${carry_lateral_ramp_height_args[@]}" \
            "${presentation_fraction_from_giver_args[@]}" \
            "${presentation_height_in_robot_frame_args[@]}" \
            "${giver_close_distance_args[@]}" \
            "${giver_lift_contact_force_threshold_args[@]}" \
            "${giver_pre_lift_min_contact_jaws_args[@]}" \
            "${giver_lift_on_live_contact_args[@]}" \
            "${recovery_record_args[@]}"
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
