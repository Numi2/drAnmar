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
    echo "Usage: $0 {validate|list|probe|controller-sweep|handover-sweep|smoke|benchmark|sweep|pretrain|train|play|record|tqta-start|tqta-ingest|tqta-report} [arguments]"
    echo "  probe     [task] [num_envs] [frames] [output]"
    echo "  controller-sweep [task] [num_envs] [frames] [parameter] [comma-values] [output]"
    echo "  handover-sweep [task] [num_envs] [frames] [receiver-arc-values] [output]"
    echo "  benchmark [task] [num_envs] [iterations] [output]"
    echo "  sweep     [task] [iterations] [comma-separated env counts] [output]"
    echo "  pretrain  [task] [num_envs] [updates] [validation_frames] [output]"
    echo "  train     [task] [num_envs] [iterations] [output]"
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
        if [[ "${DR_ANMAR_PICKUP_RECOVERY:-0}" == "1" ]]; then
            pickup_recovery_args=(--pickup_recovery)
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
                --pickup_recovery_fixed_correction
                "${DR_ANMAR_PICKUP_RECOVERY_FIXED_CORRECTION}"
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
        if [[ -n "${DR_ANMAR_PICKUP_RECOVERY_SOBOL_CANDIDATE:-}" ]]; then
            pickup_recovery_args+=(
                --pickup_recovery_sobol_candidate
                "${DR_ANMAR_PICKUP_RECOVERY_SOBOL_CANDIDATE}"
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
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_CHECKPOINT:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_checkpoint
                "${DR_ANMAR_RECEIVER_RECOVERY_CHECKPOINT}"
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
        if [[ -n "${DR_ANMAR_RECEIVER_RECOVERY_FIXED_CORRECTION:-}" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_fixed_correction
                "${DR_ANMAR_RECEIVER_RECOVERY_FIXED_CORRECTION}"
            )
        fi
        if [[ "${DR_ANMAR_RECEIVER_RECOVERY_RANDOM_CORRECTIONS:-0}" == "1" ]]; then
            pickup_recovery_args+=(
                --receiver_recovery_random_corrections
            )
        fi
        "${DR_ANMAR_ISAAC_PYTHON}" "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" play \
            --task "${task}" \
            --checkpoint "${checkpoint}" \
            --num_envs "${num_envs}" \
            --num_frames "${frames}" \
            --seed "${DR_ANMAR_SEED}" \
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
        "${DR_ANMAR_ISAAC_PYTHON}" "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" play \
            --task "${task}" \
            --checkpoint "${checkpoint}" \
            --num_envs 1 \
            --num_frames "${frames}" \
            --seed "${DR_ANMAR_SEED}" \
            --video \
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
            "${giver_lift_on_live_contact_args[@]}"
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
