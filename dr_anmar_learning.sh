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
        dagger_args=()
        if [[ -n "${DR_ANMAR_DAGGER_WARMUP_UPDATES:-}" ]]; then
            dagger_args+=(
                --dagger_warmup_updates
                "${DR_ANMAR_DAGGER_WARMUP_UPDATES}"
            )
        fi
        if [[ -n "${DR_ANMAR_DAGGER_MIN_TEACHER_FRACTION:-}" ]]; then
            dagger_args+=(
                --dagger_min_teacher_fraction
                "${DR_ANMAR_DAGGER_MIN_TEACHER_FRACTION}"
            )
        fi
        dagger_args+=(
            --e2e_replay_capacity_per_phase
            "${DR_ANMAR_E2E_REPLAY_CAPACITY_PER_PHASE:-65536}"
            --e2e_replay_batch_size
            "${DR_ANMAR_E2E_REPLAY_BATCH_SIZE:-4096}"
            --e2e_samples_per_phase_step
            "${DR_ANMAR_E2E_SAMPLES_PER_PHASE_STEP:-64}"
            --e2e_student_segment_steps
            "${DR_ANMAR_E2E_STUDENT_SEGMENT_STEPS:-64}"
            --e2e_teacher_recovery_steps
            "${DR_ANMAR_E2E_TEACHER_RECOVERY_STEPS:-32}"
            --e2e_consolidation_updates
            "${DR_ANMAR_E2E_CONSOLIDATION_UPDATES:-2000}"
        )
        mkdir -p "${output}"
        "${DR_ANMAR_ISAAC_PYTHON}" \
            "${REPO_ROOT}/scripts/dr_anmar_learning_benchmark.py" pretrain \
            --task "${task}" \
            --num_envs "${num_envs}" \
            --updates "${updates}" \
            --validation_frames "${validation_frames}" \
            --seed "${DR_ANMAR_SEED}" \
            --benchmark_formatter schema,json \
            --output_path "${output}" \
            "${dagger_args[@]}"
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
        giver_adaptation_args=()
        if [[ "${DR_ANMAR_HANDOVER_GIVER_ADAPTATION:-0}" == "1" ]]; then
            giver_adaptation_args=(--handover_giver_adaptation)
        fi
        pickup_recovery_adaptation_args=()
        if [[ "${DR_ANMAR_PICKUP_RECOVERY_ADAPTATION:-0}" == "1" ]]; then
            pickup_recovery_adaptation_args=(--pickup_recovery_adaptation)
        fi
        recovery_receiver_grasp_retain_adaptation_args=()
        if [[ "${DR_ANMAR_RECOVERY_RECEIVER_GRASP_RETAIN_ADAPTATION:-0}" == "1" ]]; then
            recovery_receiver_grasp_retain_adaptation_args=(
                --recovery_receiver_grasp_retain_adaptation
            )
        fi
        joint_transfer_acquisition_adaptation_args=()
        if [[ "${DR_ANMAR_JOINT_TRANSFER_ACQUISITION_ADAPTATION:-0}" == "1" ]]; then
            joint_transfer_acquisition_adaptation_args=(
                --joint_transfer_acquisition_adaptation
            )
        fi
        transfer_refinement_adaptation_args=()
        if [[ "${DR_ANMAR_TRANSFER_REFINEMENT_ADAPTATION:-0}" == "1" ]]; then
            transfer_refinement_adaptation_args=(
                --transfer_refinement_adaptation
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
        recovery_pickup_vertical_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_RECOVERY_PICKUP_VERTICAL_ACTION_LIMIT:-}" ]]; then
            recovery_pickup_vertical_action_limit_args=(
                --recovery_pickup_vertical_action_limit
                "${DR_ANMAR_POLICY_RECOVERY_PICKUP_VERTICAL_ACTION_LIMIT}"
            )
        fi
        carry_lateral_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_CARRY_LATERAL_ACTION_LIMIT:-}" ]]; then
            carry_lateral_action_limit_args=(
                --carry_lateral_action_limit
                "${DR_ANMAR_POLICY_CARRY_LATERAL_ACTION_LIMIT}"
            )
        fi
        recovery_carry_lateral_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_RECOVERY_CARRY_LATERAL_ACTION_LIMIT:-}" ]]; then
            recovery_carry_lateral_action_limit_args=(
                --recovery_carry_lateral_action_limit
                "${DR_ANMAR_POLICY_RECOVERY_CARRY_LATERAL_ACTION_LIMIT}"
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
        receiver_crossing_angle_args=()
        if [[ -n "${DR_ANMAR_POLICY_RECEIVER_CROSSING_ANGLE_RAD:-}" ]]; then
            receiver_crossing_angle_args=(
                --receiver_crossing_angle_rad
                "${DR_ANMAR_POLICY_RECEIVER_CROSSING_ANGLE_RAD}"
            )
        fi
        transport_custody_latch_args=()
        if [[ -n "${DR_ANMAR_POLICY_TRANSPORT_CUSTODY_LATCH:-}" ]]; then
            case "${DR_ANMAR_POLICY_TRANSPORT_CUSTODY_LATCH}" in
                1) transport_custody_latch_args=(--transport_custody_latch) ;;
                0) transport_custody_latch_args=(--no-transport_custody_latch) ;;
                *)
                    echo "DR_ANMAR_POLICY_TRANSPORT_CUSTODY_LATCH must be 0 or 1" >&2
                    exit 2
                    ;;
            esac
        fi
        receiver_preposition_args=()
        if [[ -n "${DR_ANMAR_POLICY_RECEIVER_PREPOSITION:-}" ]]; then
            case "${DR_ANMAR_POLICY_RECEIVER_PREPOSITION}" in
                1) receiver_preposition_args=(--receiver_preposition) ;;
                0) receiver_preposition_args=(--no-receiver_preposition) ;;
                *)
                    echo "DR_ANMAR_POLICY_RECEIVER_PREPOSITION must be 0 or 1" >&2
                    exit 2
                    ;;
            esac
        fi
        receiver_preposition_height_args=()
        if [[ -n "${DR_ANMAR_POLICY_RECEIVER_PREPOSITION_HEIGHT_M:-}" ]]; then
            receiver_preposition_height_args=(
                --receiver_preposition_height
                "${DR_ANMAR_POLICY_RECEIVER_PREPOSITION_HEIGHT_M}"
            )
        fi
        recovery_receiver_preposition_height_args=()
        if [[ -n "${DR_ANMAR_POLICY_RECOVERY_RECEIVER_PREPOSITION_HEIGHT_M:-}" ]]; then
            recovery_receiver_preposition_height_args=(
                --recovery_receiver_preposition_height
                "${DR_ANMAR_POLICY_RECOVERY_RECEIVER_PREPOSITION_HEIGHT_M}"
            )
        fi
        receiver_adaptive_arc_args=()
        if [[ -n "${DR_ANMAR_POLICY_RECEIVER_ADAPTIVE_ARC:-}" ]]; then
            case "${DR_ANMAR_POLICY_RECEIVER_ADAPTIVE_ARC}" in
                1) receiver_adaptive_arc_args=(--receiver_adaptive_arc) ;;
                0) receiver_adaptive_arc_args=(--no-receiver_adaptive_arc) ;;
                *)
                    echo "DR_ANMAR_POLICY_RECEIVER_ADAPTIVE_ARC must be 0 or 1" >&2
                    exit 2
                    ;;
            esac
        fi
        receiver_grasp_retain_residual_args=()
        if [[ -n "${DR_ANMAR_POLICY_RECEIVER_GRASP_RETAIN_RESIDUAL:-}" ]]; then
            case "${DR_ANMAR_POLICY_RECEIVER_GRASP_RETAIN_RESIDUAL}" in
                1) receiver_grasp_retain_residual_args=(--receiver_grasp_retain_residual) ;;
                0) receiver_grasp_retain_residual_args=(--no-receiver_grasp_retain_residual) ;;
                *)
                    echo "DR_ANMAR_POLICY_RECEIVER_GRASP_RETAIN_RESIDUAL must be 0 or 1" >&2
                    exit 2
                    ;;
            esac
        fi
        presentation_filtered_custody_args=()
        if [[ -n "${DR_ANMAR_PRESENTATION_USE_FILTERED_CUSTODY:-}" ]]; then
            case "${DR_ANMAR_PRESENTATION_USE_FILTERED_CUSTODY}" in
                1) presentation_filtered_custody_args=(--presentation_use_filtered_custody) ;;
                0) presentation_filtered_custody_args=(--no-presentation_use_filtered_custody) ;;
                *)
                    echo "DR_ANMAR_PRESENTATION_USE_FILTERED_CUSTODY must be 0 or 1" >&2
                    exit 2
                    ;;
            esac
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
        giver_transport_orientation_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_GIVER_TRANSPORT_ORIENTATION_ACTION_LIMIT:-}" ]]; then
            giver_transport_orientation_action_limit_args=(
                --giver_transport_orientation_action_limit
                "${DR_ANMAR_POLICY_GIVER_TRANSPORT_ORIENTATION_ACTION_LIMIT}"
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
            --num_envs "${num_envs}" \
            --num_frames "${frames}" \
            --seed "${DR_ANMAR_SEED}" \
            --benchmark_formatter schema,json \
            --output_path "${output}" \
            "${residual_scale_args[@]}" \
            "${giver_adaptation_args[@]}" \
            "${pickup_recovery_adaptation_args[@]}" \
            "${recovery_receiver_grasp_retain_adaptation_args[@]}" \
            "${joint_transfer_acquisition_adaptation_args[@]}" \
            "${transfer_refinement_adaptation_args[@]}" \
            "${pickup_vertical_action_limit_args[@]}" \
            "${pickup_initial_vertical_action_limit_args[@]}" \
            "${recovery_pickup_vertical_action_limit_args[@]}" \
            "${carry_lateral_action_limit_args[@]}" \
            "${recovery_carry_lateral_action_limit_args[@]}" \
            "${carry_lateral_ramp_height_args[@]}" \
            "${presentation_fraction_from_giver_args[@]}" \
            "${receiver_crossing_angle_args[@]}" \
            "${transport_custody_latch_args[@]}" \
            "${receiver_preposition_args[@]}" \
            "${receiver_preposition_height_args[@]}" \
            "${recovery_receiver_preposition_height_args[@]}" \
            "${receiver_adaptive_arc_args[@]}" \
            "${receiver_grasp_retain_residual_args[@]}" \
            "${presentation_filtered_custody_args[@]}" \
            "${presentation_height_in_robot_frame_args[@]}" \
            "${giver_close_distance_args[@]}" \
            "${giver_lift_contact_force_threshold_args[@]}" \
            "${giver_pre_lift_min_contact_jaws_args[@]}" \
            "${giver_transport_orientation_action_limit_args[@]}" \
            "${giver_lift_on_live_contact_args[@]}"
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
        recovery_pickup_vertical_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_RECOVERY_PICKUP_VERTICAL_ACTION_LIMIT:-}" ]]; then
            recovery_pickup_vertical_action_limit_args=(
                --recovery_pickup_vertical_action_limit
                "${DR_ANMAR_POLICY_RECOVERY_PICKUP_VERTICAL_ACTION_LIMIT}"
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
        giver_transport_orientation_action_limit_args=()
        if [[ -n "${DR_ANMAR_POLICY_GIVER_TRANSPORT_ORIENTATION_ACTION_LIMIT:-}" ]]; then
            giver_transport_orientation_action_limit_args=(
                --giver_transport_orientation_action_limit
                "${DR_ANMAR_POLICY_GIVER_TRANSPORT_ORIENTATION_ACTION_LIMIT}"
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
            "${recovery_pickup_vertical_action_limit_args[@]}" \
            "${carry_lateral_action_limit_args[@]}" \
            "${carry_lateral_ramp_height_args[@]}" \
            "${presentation_fraction_from_giver_args[@]}" \
            "${presentation_height_in_robot_frame_args[@]}" \
            "${giver_close_distance_args[@]}" \
            "${giver_lift_contact_force_threshold_args[@]}" \
            "${giver_pre_lift_min_contact_jaws_args[@]}" \
            "${giver_transport_orientation_action_limit_args[@]}" \
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
