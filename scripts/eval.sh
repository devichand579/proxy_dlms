#!/bin/bash
#SBATCH --job-name=eval_strace
#SBATCH --output=job_outputs/strace_job_out_%j.out
#SBATCH --error=job_outputs/strace_job_err_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=47:00:00
#SBATCH --partition=gpu_h100 #gpu_l40, gpu_h100, dibdp (AIRAWAT)
#SBATCH --gres=gpu:1  # ignore if your cluster doesn't have a qos for GPUs

########################################################################################
# DGX setup: uncomment the block below and comment out (partition through gres) above
########################################################################################
##SBATCH --partition=dgx2
##SBATCH --qos=gpu2
##SBATCH --gres=gpu:2
##SBATCH --mem=72GB
##SBATCH --time=72:00:00
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --open-mode=append
##SBATCH --signal=USR2@120
##SBATCH --wckey=submitit
##SBATCH --output=job_outputs/%j/%j_0_log.out
##SBATCH --error=job_outputs/%j/%j_0_log.err

########################################################################################
# Single eval job via sbatch: run this script with your args and --slurm, e.g.:
#   sbatch scripts/eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --slurm
# Resumable (checkpointing): add --resumable; re-run same command to resume after crash
# Alternative (Hydra): sbatch --wrap "conda run -n strace python hydra_runners/launch_eval.py task_name=gsm8k mode=pd"
# Multirun on SLURM (one job per config): python hydra_runners/launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun launcher=slurm


########################################################################################
# Parse command-line arguments
# Usage: ./scripts/eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada [opts...]
#   Full flags: see Error/Unknown option messages below; e.g. --resumable [--run_id] [--slurm] [--use_eos_harness / --no_eos_harness]
# Mode shortcuts: pd, bd, eb, fdm, fhs, top_k_c, top_k_m, dus
# Intervention and wandb are optional. Datasets: gsm8k, minerva_math500, humaneval, mbpp, arc_challenge_chat
# --model: basename of a folder under ./models/ (must match lm-eval registration). Examples: llada, llada1_5, dream.
#          Weights/tokenizer load from ./models/<model>/; mask/EOS/EoT ids come from that folder's config when not overridden.
# --slurm: run the eval command under srun (use when script is submitted via sbatch so the job uses the allocated GPU).
# Multi-GPU: use --parallel_config <YAML> -> hydra_runners/launch_eval.py parallel_runs_file= (see config/parallel_eval_runs.yaml).
# Local GPU: run without --slurm; use --device 0|1|cuda:0 or export CUDA_VISIBLE_DEVICES before running.
# Offline (cached datasets + local W&B): add --offline (run scripts/download_datasets.sh first).

# shellcheck source=_repo_root.sh
_script_dir="$(dirname "${BASH_SOURCE[0]}")"
# When sbatch copies the script to its spool dir, _repo_root.sh won't be there;
# fall back to the original submission directory.
if [[ ! -f "$_script_dir/_repo_root.sh" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _script_dir="${SLURM_SUBMIT_DIR}/scripts"
fi
source "$_script_dir/_repo_root.sh"
unset _script_dir
BATCH_SIZE=""
MODE=""
STRATEGY=""
TASK_NAME=""
MODEL=""
COMPILE_MODEL=""
INTERVENTION_MODE=""
INTERVENTION_STEPS=""
INTERVENTION_K=""
WANDB=""
TRACING_ENABLED_ARG=""
TRACE_HF_UPLOAD_ARG=""
TRACE_HF_DATASET_REPO_ARG=""
TRACE_HF_NAMESPACE_ARG=""
NUM_FEWSHOT_OVERRIDE=""
LIMIT_OVERRIDE=""
BLOCK_LENGTH_OVERRIDE=""
TOP_K_OVERRIDE=""
STEPS_OVERRIDE=""
GEN_LENGTH_OVERRIDE=""
SLURM_MODE=false
RESUMABLE=false
RUN_ID=""
USE_EOS_HARNESS="true"
PARALLEL_CONFIG=""
OFFLINE=false
DEVICE=""
LAUNCHER=""
MODEL_DEVICE="cuda"
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --task_name)
            TASK_NAME="$2"
            shift 2
            ;;
        # See header: must be llada | llada1_5 | dream (directory name under ./models/)
        --model)
            MODEL="$2"
            shift 2
            ;;
        --compile_model)
            COMPILE_MODEL="$2"
            shift 2
            ;;
        --intervention_mode)
            INTERVENTION_MODE="$2"
            shift 2
            ;;
        --intervention_steps)
            INTERVENTION_STEPS="$2"
            shift 2
            ;;
        --intervention_k)
            INTERVENTION_K="$2"
            shift 2
            ;;
        --wandb)
            WANDB="$2"
            shift 2
            ;;
        --tracing_enabled)
            TRACING_ENABLED_ARG="$2"
            shift 2
            ;;
        --trace_hf_upload)
            TRACE_HF_UPLOAD_ARG="$2"
            shift 2
            ;;
        --trace_hf_dataset_repo)
            TRACE_HF_DATASET_REPO_ARG="$2"
            shift 2
            ;;
        --trace_hf_namespace)
            TRACE_HF_NAMESPACE_ARG="$2"
            shift 2
            ;;
        --num_fewshot)
            NUM_FEWSHOT_OVERRIDE="$2"
            shift 2
            ;;
        --limit|--num_samples)
            LIMIT_OVERRIDE="$2"
            shift 2
            ;;
        --block_length)
            BLOCK_LENGTH_OVERRIDE="$2"
            shift 2
            ;;
        --top_k)
            TOP_K_OVERRIDE="$2"
            shift 2
            ;;
        --steps)
            STEPS_OVERRIDE="$2"
            shift 2
            ;;
        --gen_length)
            GEN_LENGTH_OVERRIDE="$2"
            shift 2
            ;;
        --parallel_config|--parallel_runs_file)
            PARALLEL_CONFIG="$2"
            shift 2
            ;;
        --slurm)
            SLURM_MODE=true
            shift
            ;;
        --resumable)
            RESUMABLE=true
            shift
            ;;
        --run_id)
            RUN_ID="$2"
            shift 2
            ;;
        --use_eos_harness)
            USE_EOS_HARNESS="$2"
            shift 2
            ;;
        --no_eos_harness)
            USE_EOS_HARNESS="false"
            shift
            ;;
        --offline)
            OFFLINE=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --launcher)
            LAUNCHER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --batch_size <size> --mode <mode> --task_name <task> --model <model> [options]"
            echo "  Core optional: --num_fewshot --limit (alias: --num_samples) --block_length --top_k --steps --gen_length --compile_model"
            echo "  Harness (eval.LLaDAEvalHarness / --model_args): --intervention_mode --intervention_steps --intervention_k"
            echo "  EOS decode: --use_eos_harness <True|False> or --no_eos_harness (default true; false = full steps + full-tail decode)"
            echo "  Alternative multi-GPU: --parallel_config <parallel_runs.yaml> (per-GPU overrides: mode/model/task/batch — delegates to hydra_runners/launch_eval.py parallel_runs_file=)"
            echo "Run: --resumable [--run_id] --slurm  |  Multi-GPU: --parallel_config <parallel_runs.yaml>"
            echo "  Offline: --offline (HF cache + WANDB_MODE=offline; prefetch with scripts/download_datasets.sh)"
            echo "  GPU: --device <0|1|cuda|cuda:0|cpu>  (default cuda on GPU 0 when not --slurm)"
            echo "  Cluster parallel: --parallel_config <yaml> --launcher slurm_dgx  (auto sbatch N GPUs from login node)"
            exit 1
            ;;
    esac
done

if [[ -n "$LIMIT_OVERRIDE" && ! "$LIMIT_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --limit/--num_samples must be a positive integer"
    exit 1
fi

if [[ -n "$PARALLEL_CONFIG" ]]; then
    if [[ -f "$PARALLEL_CONFIG" ]]; then
        :
    elif [[ -f "$REPO_ROOT/$PARALLEL_CONFIG" ]]; then
        PARALLEL_CONFIG="$REPO_ROOT/$PARALLEL_CONFIG"
    else
        echo "Error: --parallel_config not found: $PARALLEL_CONFIG (tried repo path under $REPO_ROOT)"
        exit 1
    fi
fi

# Check required arguments (intervention and wandb are optional)
if [[ -z "$PARALLEL_CONFIG" ]]; then
if [[ -z "$BATCH_SIZE" || -z "$MODE" || -z "$TASK_NAME" || -z "$MODEL" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 --batch_size <size> --mode <mode> --task_name <task> --model <model> [options]"
    echo "  Core optional: --num_fewshot <n> --limit <n> (alias: --num_samples) --block_length <n> --top_k <n> --steps <n> --gen_length <n> --compile_model <bool>"
    echo "  Harness (see eval.LLaDAEvalHarness): --intervention_mode drop|swap, --intervention_steps 0,128,256, --intervention_k <k>"
    echo "  --wandb <True|False>  --tracing_enabled <True|False>  (tracing: trace_outputs/ or JSONL when --resumable)"
    echo "  --trace_hf_upload <True|False>  --trace_hf_dataset_repo <id>  --trace_hf_namespace <user|org>  (Hub dataset push; default repo schedulerlens when tracing on)"
    echo "  Decode: --use_eos_harness <True|False> or --no_eos_harness (default true; false = full steps + full-tail decode, see eval.py)"
    echo "  Run: --slurm (srun for sbatch)  --resumable [--run_id]  (checkpoint under outputs/…)"
    echo "  Multi-GPU: --parallel_config <yaml> per-GPU settings (launch_eval parallel_runs_file=); see docs/evaluation.md"
    echo "  Offline: --offline  |  GPU: --device <0|1|cuda|cuda:0|cpu>"
    echo "  Mode shortcuts: pd, bd, eb, fdm, fhs, top_k_c, top_k_m, dus"
    echo "  Few-shot defaults: humaneval/arc 0, gsm8k/math500 4, mbpp 3, arc_challenge_chat 5; block_length defaults: bd 32, fdm 64, dus 16, else gen_length"
    echo "Example: $0 --batch_size 2 --mode pd --task_name gpqa_main_zeroshot --model llada --compile_model False"
    echo "Example: $0 --batch_size 2 --mode pd --task_name gsm8k --model llada --intervention_mode drop --intervention_steps 0,128,256 --intervention_k 1 --wandb True"
    exit 1
fi
fi

# Convert mode shortcuts to full mode names
if [[ -z "$PARALLEL_CONFIG" ]]; then
case "$MODE" in
    pd)
        MODE="pure_diffusion"
        STRATEGY="confidence"
        ;;
    bd)
        MODE="block_diffusion"
        STRATEGY="confidence"
        ;;
    eb)
        MODE="eb_sampler"
        STRATEGY="confidence"
        ;;
    fdm)
        MODE="fdm"
        STRATEGY="confidence"  # unused for fdm, kept for model_args
        ;;
    fhs)
        MODE="fhs"
        STRATEGY="confidence"  # unused for fhs, kept for model_args
        ;;
    top_k_c)
        MODE="top_k"
        STRATEGY="confidence"
        ;;
    top_k_m)
        MODE="top_k"
        STRATEGY="top_margin"
        ;;
    dus)
        MODE="dus"
        STRATEGY="confidence"
        ;;
    *)
        echo "Warning: Unknown mode '$MODE'. Using as-is. Valid modes: pd, bd, eb, fdm, fhs, top_k_c, top_k_m, dus"
        ;;
esac

# ``top_k_c`` / ``top_k_m`` both use mode=top_k; include strategy so resumable dirs / W&B names differ.
TOPK_TAG=""
if [[ "$MODE" == "top_k" ]]; then
    TOPK_TAG="_${STRATEGY:-confidence}"
fi

fi

# Initialize conda for this shell session
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate strace
export PATH="$CONDA_PREFIX/bin:$PATH" 

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export LMEVAL_LOG_LEVEL=INFO
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=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

if [[ "$OFFLINE" == "true" ]]; then
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export WANDB_MODE=offline
    export TRACE_HF_UPLOAD=FALSE
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
fi

# When running locally (not --slurm), pin GPU via --device or default GPU 0.
if [[ "$SLURM_MODE" != "true" ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    if [[ -n "$DEVICE" ]]; then
        case "$DEVICE" in
            cpu|CPU)
                export CUDA_VISIBLE_DEVICES=""
                MODEL_DEVICE="cpu"
                ;;
            cuda)
                export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
                MODEL_DEVICE="cuda"
                ;;
            cuda:*)
                GPU_IDX="${DEVICE#cuda:}"
                export CUDA_VISIBLE_DEVICES="$GPU_IDX"
                MODEL_DEVICE="cuda"
                ;;
            [0-9]*)
                export CUDA_VISIBLE_DEVICES="$DEVICE"
                MODEL_DEVICE="cuda"
                ;;
            *)
                MODEL_DEVICE="$DEVICE"
                ;;
        esac
    else
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        MODEL_DEVICE="cuda"
    fi
fi

export WANDB_API_KEY=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
export WANDB_PROJECT=strace

# Tracing: enable only when --tracing_enabled True (default off to avoid overhead)
if [[ "${TRACING_ENABLED_ARG,,}" == "true" ]]; then
    export TRACING_ENABLED=TRUE
else
    export TRACING_ENABLED=FALSE
fi

# HF dataset upload for final trace file (default: on when tracing is on -> repo schedulerlens)
if [[ -n "$TRACE_HF_UPLOAD_ARG" ]]; then
    if [[ "${TRACE_HF_UPLOAD_ARG,,}" == "true" ]]; then
        export TRACE_HF_UPLOAD=TRUE
    else
        export TRACE_HF_UPLOAD=FALSE
    fi
elif [[ "${TRACING_ENABLED_ARG,,}" == "true" ]]; then
    export TRACE_HF_UPLOAD=TRUE
else
    export TRACE_HF_UPLOAD=FALSE
fi
export TRACE_HF_DATASET_REPO="${TRACE_HF_DATASET_REPO_ARG:-schedulerlens}"
[[ -n "$TRACE_HF_NAMESPACE_ARG" ]] && export TRACE_HF_NAMESPACE="$TRACE_HF_NAMESPACE_ARG"

export TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ -n "$PARALLEL_CONFIG" ]]; then
    ABS_PARALLEL_CFG="$(realpath "$PARALLEL_CONFIG")"
    if [[ ! -f "$ABS_PARALLEL_CFG" ]]; then
        echo "Error: --parallel_config must resolve to a file: $PARALLEL_CONFIG"
        exit 1
    fi
    cd "$REPO_ROOT" || exit 1
    PARALLEL_SLOTS=$(python -c "from hydra_runners.launch_eval import count_parallel_slots; print(count_parallel_slots('$ABS_PARALLEL_CFG'))")
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        SLURM_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-}}"
        if [[ -n "$SLURM_GPUS" && "$SLURM_GPUS" -lt "$PARALLEL_SLOTS" ]]; then
            echo "Error: parallel_config has ${PARALLEL_SLOTS} slots but SLURM job has ${SLURM_GPUS} GPU(s)."
            echo "  Set #SBATCH --gres=gpu:${PARALLEL_SLOTS} in eval.sh (or submit from login with --launcher slurm_dgx)."
            exit 1
        fi
    fi
    OVERRIDES=(parallel_runs_file="$ABS_PARALLEL_CFG")
    [[ -n "$BATCH_SIZE" ]] && OVERRIDES+=(batch_size="$BATCH_SIZE")
    [[ -n "$WANDB" ]] && OVERRIDES+=(wandb="$WANDB")
    [[ -n "$TRACING_ENABLED_ARG" ]] && OVERRIDES+=(tracing_enabled="$TRACING_ENABLED_ARG")
    [[ -n "$TRACE_HF_UPLOAD_ARG" ]] && OVERRIDES+=(trace_hf_upload="$TRACE_HF_UPLOAD_ARG")
    [[ -n "$TRACE_HF_DATASET_REPO_ARG" ]] && OVERRIDES+=(trace_hf_dataset_repo="$TRACE_HF_DATASET_REPO_ARG")
    [[ -n "$TRACE_HF_NAMESPACE_ARG" ]] && OVERRIDES+=(trace_hf_namespace="$TRACE_HF_NAMESPACE_ARG")
    [[ "$RESUMABLE" == "true" ]] && OVERRIDES+=(resumable=true)
    [[ -n "$RUN_ID" ]] && OVERRIDES+=(run_id="$RUN_ID")
    [[ -n "$COMPILE_MODEL" ]] && OVERRIDES+=(compile_model="$COMPILE_MODEL")
    [[ -n "$BLOCK_LENGTH_OVERRIDE" ]] && OVERRIDES+=(block_length="$BLOCK_LENGTH_OVERRIDE")
    [[ -n "$TOP_K_OVERRIDE" ]] && OVERRIDES+=(top_k="$TOP_K_OVERRIDE")
    [[ -n "$STEPS_OVERRIDE" ]] && OVERRIDES+=(steps="$STEPS_OVERRIDE")
    [[ -n "$GEN_LENGTH_OVERRIDE" ]] && OVERRIDES+=(gen_length="$GEN_LENGTH_OVERRIDE")
    [[ -n "$LIMIT_OVERRIDE" ]] && OVERRIDES+=(limit="$LIMIT_OVERRIDE")
    [[ "$OFFLINE" == "true" ]] && OVERRIDES+=(offline=true)
    [[ -n "$DEVICE" ]] && OVERRIDES+=(device="$DEVICE")
    [[ -n "$LAUNCHER" ]] && OVERRIDES+=(launcher="$LAUNCHER")
    exec python hydra_runners/launch_eval.py "${OVERRIDES[@]}"
fi

run_eval_for_task() {
    export TASK_NAME="$1"

    # Set gen_length and steps (ARC / CommonsenseQA: 128; others: 512)
    if [[ "$TASK_NAME" == *"arc_challenge_chat"* ]] || [[ "$TASK_NAME" == *"arc_easy"* ]] || [[ "$TASK_NAME" == *"commonsense"* ]]; then
        export GEN_LENGTH=128
        export STEPS=128
    else
        export GEN_LENGTH=512
        export STEPS=512
    fi
    if [[ -n "$GEN_LENGTH_OVERRIDE" ]]; then
        export GEN_LENGTH="$GEN_LENGTH_OVERRIDE"
    fi
    if [[ -n "$STEPS_OVERRIDE" ]]; then
        export STEPS="$STEPS_OVERRIDE"
    fi
    if (( 10#$STEPS > 10#$GEN_LENGTH )); then
        echo "Error: steps ($STEPS) must be <= gen_length ($GEN_LENGTH). Adjust --steps / --gen_length or task defaults."
        exit 1
    fi

    # Set block_length: override if --block_length passed, else defaults by mode
    # block_diffusion: 32, fdm: 64, dus: 16, Otherwise: gen_length
    if [[ -n "$BLOCK_LENGTH_OVERRIDE" ]]; then
        export BLOCK_LENGTH=$BLOCK_LENGTH_OVERRIDE
    elif [[ "$MODE" == "block_diffusion" ]]; then
        export BLOCK_LENGTH=32
    elif [[ "$MODE" == "fdm" ]]; then
        export BLOCK_LENGTH=64
    elif [[ "$MODE" == "dus" ]]; then
        export BLOCK_LENGTH=16
    else
        export BLOCK_LENGTH=$GEN_LENGTH
    fi

    # Set top_k: override if --top_k passed, else default 2 (top_k scheduler only)
    if [[ -n "$TOP_K_OVERRIDE" ]]; then
        export TOP_K=$TOP_K_OVERRIDE
    else
        export TOP_K=2
    fi

    # Passed through to eval harness as model_path=...; AutoModel/AutoTokenizer resolve from this directory.
    export MODEL_PATH=./models/${MODEL}

    # Default compile_model if not set
    if [[ -z "$COMPILE_MODEL" ]]; then
        COMPILE_MODEL="False"
    fi

    # MODEL_TASK_MODE[_intv_…][_s{steps}]; explicit --run_id wins as-is
    BASE_SUFFIX="${MODEL}_${TASK_NAME}_${MODE}${TOPK_TAG}"
    if [[ -n "$INTERVENTION_MODE" ]]; then
        INTERVENTION_STEPS="${INTERVENTION_STEPS:-0}"
        INTERVENTION_STEPS_SAFE="${INTERVENTION_STEPS//,/_}"
        BASE_SUFFIX="${BASE_SUFFIX}_intv_${INTERVENTION_MODE}_steps${INTERVENTION_STEPS_SAFE}_k${INTERVENTION_K:-1}"
    fi
    if [[ -z "$RUN_ID" ]]; then
        BASE_SUFFIX="${BASE_SUFFIX}_s${STEPS}"
    fi
    if [[ "$RESUMABLE" == "true" ]]; then
        if [[ -n "$RUN_ID" ]]; then
            OUTPUT_SUFFIX="$RUN_ID"
        else
            OUTPUT_SUFFIX="$BASE_SUFFIX"
        fi
        CHECKPOINT_DIR="outputs/${OUTPUT_SUFFIX}"
        mkdir -p "$CHECKPOINT_DIR"
    else
        OUTPUT_SUFFIX="${BASE_SUFFIX}_${TIMESTAMP}"
        CHECKPOINT_DIR=""
    fi
    WANDB_NAME="${OUTPUT_SUFFIX}"

    # Set num_fewshot: override if provided, else per-task default
    if [[ -n "$NUM_FEWSHOT_OVERRIDE" ]]; then
        NUM_FEWSHOT="$NUM_FEWSHOT_OVERRIDE"
    else
        NUM_FEWSHOT=0  # Default
        if [[ "$TASK_NAME" == *"humaneval"* ]]; then
            NUM_FEWSHOT=0
        elif [[ "$TASK_NAME" == *"gsm8k"* ]]; then
            NUM_FEWSHOT=4
        elif [[ "$TASK_NAME" == *"math500"* ]] || [[ "$TASK_NAME" == *"minerva_math500"* ]]; then
            NUM_FEWSHOT=4
        elif [[ "$TASK_NAME" == *"mbpp"* ]]; then
            NUM_FEWSHOT=3
        elif [[ "$TASK_NAME" == *"arc_challenge_chat"* ]]; then
            NUM_FEWSHOT=5
        fi
    fi

    # Build model_args (base args + optional intervention params + checkpoint paths when resumable)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},mode=${MODE},strategy=${STRATEGY},top_k=${TOP_K},compile_model=${COMPILE_MODEL},task_name=${TASK_NAME},trace_name=${OUTPUT_SUFFIX},use_eos_harness=${USE_EOS_HARNESS},device=${MODEL_DEVICE}"
    if [[ "$RESUMABLE" == "true" ]]; then
        MODEL_ARGS="${MODEL_ARGS},resumable=true,output_dir=${CHECKPOINT_DIR}"
    fi
    if [[ -n "$INTERVENTION_MODE" ]]; then
        INTERVENTION_STEPS_ARG="${INTERVENTION_STEPS//,/_}"
        MODEL_ARGS="${MODEL_ARGS},intervention_mode=${INTERVENTION_MODE},intervention_steps=${INTERVENTION_STEPS_ARG},intervention_k=${INTERVENTION_K:-1}"
    fi

    echo "Configuration:"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Mode: $MODE"
    echo "  Task Name: $TASK_NAME"
    echo "  Model: $MODEL"
    echo "  Model Path: $MODEL_PATH"
    echo "  Gen Length: $GEN_LENGTH"
    echo "  Steps: $STEPS"
    echo "  Block Length: $BLOCK_LENGTH"
    echo "  Top-k: $TOP_K"
    echo "  Compile Model: $COMPILE_MODEL"
    echo "  Device: $MODEL_DEVICE (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
    echo "  Offline: ${OFFLINE}"
    echo "  use_eos_harness: $USE_EOS_HARNESS (false = full steps + full-tail decode, reference LLaDA-style)"
    echo "  Num Fewshot: $NUM_FEWSHOT"
    echo "  Sample limit: ${LIMIT_OVERRIDE:-all}"
    if [[ -n "$INTERVENTION_MODE" ]]; then
        echo "  Intervention Mode: $INTERVENTION_MODE"
        echo "  Intervention Steps: $INTERVENTION_STEPS"
        echo "  Intervention K: ${INTERVENTION_K:-1}"
    fi
    echo "  Wandb: ${WANDB:-False}"
    echo "  Tracing enabled: ${TRACING_ENABLED_ARG:-False}"
    echo "  Trace HF upload: ${TRACE_HF_UPLOAD:-False} (repo: ${TRACE_HF_DATASET_REPO:-schedulerlens})"
    echo "  SLURM mode (srun outer): ${SLURM_MODE}"
    if [[ -n "$WANDB_NAME" ]]; then
        echo "  Wandb Name: $WANDB_NAME"
    fi
    if [[ "$RESUMABLE" == "true" ]]; then
        echo "  Resumable: yes (checkpoint=${CHECKPOINT_DIR})"
    else
        echo "  Output: outputs/${OUTPUT_SUFFIX}.json"
    fi
    echo ""

    RESUME_ENV=""
    WANDB_RESUME_ARGS=""
    if [[ "$RESUMABLE" == "true" ]]; then
        WANDB_ID_FILE="${CHECKPOINT_DIR}/.wandb_id"
        if [[ -f "$WANDB_ID_FILE" ]] && [[ "${WANDB,,}" == "true" ]]; then
            WANDB_RUN_ID=$(cat "$WANDB_ID_FILE")
            WANDB_RESUME_ARGS=",resume=allow,id=${WANDB_RUN_ID}"
            echo "  Resuming wandb run: ${WANDB_RUN_ID}"
        fi
        RESUME_ENV="RESUMABLE_RUN_ID=${OUTPUT_SUFFIX} CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    fi

    if [[ "$RESUMABLE" == "true" ]]; then
        ACCEL_CMD="accelerate launch eval.py --tasks ${TASK_NAME} --batch_size ${BATCH_SIZE} --num_fewshot ${NUM_FEWSHOT} --model ${MODEL} --model_args ${MODEL_ARGS} --write_out --output_path ${CHECKPOINT_DIR}/results.json --log_samples --cache_requests true"
    else
        ACCEL_CMD="accelerate launch eval.py --tasks ${TASK_NAME} --batch_size ${BATCH_SIZE} --num_fewshot ${NUM_FEWSHOT} --model ${MODEL} --model_args ${MODEL_ARGS} --write_out --output_path outputs/${OUTPUT_SUFFIX}.json --log_samples --cache_requests true"
    fi
    if [[ -n "$LIMIT_OVERRIDE" ]]; then
        ACCEL_CMD="${ACCEL_CMD} --limit ${LIMIT_OVERRIDE}"
    fi
    if [[ "${WANDB,,}" == "true" ]]; then
        ACCEL_CMD="${ACCEL_CMD} --wandb_args project=${WANDB_PROJECT},name=${WANDB_NAME}${WANDB_RESUME_ARGS}"
    fi
    if [[ -n "$RESUME_ENV" ]]; then
        ACCEL_CMD="env ${RESUME_ENV} ${ACCEL_CMD}"
    fi

    if [[ "$SLURM_MODE" == "true" ]]; then
        srun bash -c "$ACCEL_CMD"
    else
        eval $ACCEL_CMD
    fi
}

run_eval_for_task "$TASK_NAME"
exit $?
