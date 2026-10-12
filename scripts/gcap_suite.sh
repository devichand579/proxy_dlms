#!/usr/bin/env bash
# Run four K=8 G-cap benchmarks on eight GPUs, then score every candidate pool.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
    TASKS=("$@")
else
    TASKS=(humaneval mbpp gsm8k minerva_math500)
fi

BATCH_SIZE="${BATCH_SIZE:-64}"
PD_STEPS="${PD_STEPS:-512}"
BD_STEPS="${BD_STEPS:-512}"
TOP_K_C_STEPS="${TOP_K_C_STEPS:-256}"
TOP_K_M_STEPS="${TOP_K_M_STEPS:-256}"
DUS_STEPS="${DUS_STEPS:-128}"
EB_STEPS="${EB_STEPS:-128}"
GEN_LENGTH="${GEN_LENGTH:-512}"
GCAP_K="${GCAP_K:-8}"
GCAP_PREFIX_STEPS="${GCAP_PREFIX_STEPS:-32}"
TEMPERATURE="${TEMPERATURE:-0.7}"
WANDB="${WANDB:-True}"
CODE_WORKERS="${CODE_WORKERS:-8}"
LIMIT="${LIMIT:-}"
LOG_DIR="${LOG_DIR:-logs/gcap_suite}"
SUMMARY_DIR="${SUMMARY_DIR:-outputs/gcap_suite_summary}"
SUMMARY_CSV="$SUMMARY_DIR/summary.csv"

mkdir -p "$LOG_DIR" "$SUMMARY_DIR"

# PD and BD use two GPUs each because their 512-step budgets dominate wall
# time. The existing top-k-m and EB two-shard checkpoints are preserved, but
# each scheduler's two shards run sequentially on one GPU.
GPU=(0 1 2 3 4 5 6 7)
SCHED=(pd pd bd bd top_k_c dus top_k_m eb)
SHARDS=("0" "1" "0" "1" "0" "0" "0 1" "0 1")
NSHARDS=(2 2 2 2 1 1 2 2)

steps_for() {
    case "$1" in
        pd) echo "$PD_STEPS" ;;
        bd) echo "$BD_STEPS" ;;
        top_k_c) echo "$TOP_K_C_STEPS" ;;
        top_k_m) echo "$TOP_K_M_STEPS" ;;
        dus) echo "$DUS_STEPS" ;;
        eb) echo "$EB_STEPS" ;;
        *)
            echo "Unknown scheduler: $1" >&2
            return 1
            ;;
    esac
}

run_id() {
    local task="$1" scheduler="$2" shard="$3" nshards="$4"
    local steps
    steps="$(steps_for "$scheduler")"
    local temp_tag="${TEMPERATURE//./p}"
    echo "llada_${task}_gcap_${scheduler}_per_commit_k${GCAP_K}_t${temp_tag}_s${steps}_shard${shard}of${nshards}"
}

fewshot_for() {
    case "$1" in
        humaneval) echo 0 ;;
        mbpp) echo 3 ;;
        gsm8k|minerva_math500) echo 4 ;;
        *) echo 0 ;;
    esac
}

for task in "${TASKS[@]}"; do
    echo "Launching task=$task on ${#GPU[@]} GPUs"
    pids=()
    names=()
    for slot in "${!GPU[@]}"; do
        gpu="${GPU[$slot]}"
        scheduler="${SCHED[$slot]}"
        nshards="${NSHARDS[$slot]}"
        steps="$(steps_for "$scheduler")"
        shard_list="${SHARDS[$slot]}"
        names+=("${scheduler}_gpu${gpu}_shards${shard_list// /_}")
        (
            export ROCR_VISIBLE_DEVICES="$gpu"
            # ROCR narrows the process to one physical AMD GPU and remaps it to
            # logical device 0. Select that logical device for HIP/PyTorch.
            export HIP_VISIBLE_DEVICES=0
            export CUDA_VISIBLE_DEVICES=0
            limit_args=()
            if [[ -n "$LIMIT" ]]; then
                limit_args=(--limit "$LIMIT")
            fi
            for shard in $shard_list; do
                name="$(run_id "$task" "$scheduler" "$shard" "$nshards")"
                log="$LOG_DIR/${name}.log"
                if ! ./scripts/eval.sh \
                    --batch_size "$BATCH_SIZE" \
                    --mode gcap \
                    --gcap_scheduler "$scheduler" \
                    --gcap_k "$GCAP_K" \
                    --gcap_score_mode per_commit \
                    --gcap_prefix_steps "$GCAP_PREFIX_STEPS" \
                    --gcap_random_control false \
                    --gcap_selection_seed 7070 \
                    --temperature "$TEMPERATURE" \
                    --task_name "$task" \
                    --model llada \
                    --steps "$steps" \
                    --gen_length "$GEN_LENGTH" \
                    --dataset_num_shards "$nshards" \
                    --dataset_shard_index "$shard" \
                    --run_id "$name" \
                    --device cuda \
                    --resumable \
                    --wandb "$WANDB" \
                    "${limit_args[@]}" \
                    >>"$log" 2>&1; then
                    echo "Worker failed: $name (see $log)" >&2
                    exit 1
                fi
            done
        ) &
        pids+=("$!")
    done

    failed=0
    for slot in "${!pids[@]}"; do
        if ! wait "${pids[$slot]}"; then
            echo "Worker failed: ${names[$slot]} (see scheduler logs under $LOG_DIR)"
            failed=1
        fi
    done
    if (( failed )); then
        echo "Task $task stopped with failed workers. Rerun this script to resume."
        exit 1
    fi
    echo "Generation complete for task=$task"
done

partial_args=()
if [[ -n "$LIMIT" ]]; then
    partial_args=(--allow-partial)
fi
for task in "${TASKS[@]}"; do
    fewshot="$(fewshot_for "$task")"
    for scheduler in pd bd dus top_k_c top_k_m eb; do
        if [[ "$scheduler" == "pd" || "$scheduler" == "bd" || "$scheduler" == "top_k_m" || "$scheduler" == "eb" ]]; then
            run_args=(
                --run-dir "outputs/$(run_id "$task" "$scheduler" 0 2)"
                --run-dir "outputs/$(run_id "$task" "$scheduler" 1 2)"
            )
        else
            run_args=(--run-dir "outputs/$(run_id "$task" "$scheduler" 0 1)")
        fi
        python3 -m schedulerlens.gcap_summary \
            --task "$task" \
            --scheduler "$scheduler" \
            "${run_args[@]}" \
            --prefix-steps "$GCAP_PREFIX_STEPS" \
            --k "$GCAP_K" \
            --num-fewshot "$fewshot" \
            --code-workers "$CODE_WORKERS" \
            "${partial_args[@]}" \
            --output "$SUMMARY_DIR/${task}_${scheduler}.json" \
            --summary-csv "$SUMMARY_CSV"
    done
done

echo "G-cap suite complete: $SUMMARY_CSV"
