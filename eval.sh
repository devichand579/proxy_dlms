#!/bin/bash
#SBATCH --job-name=eval_strace
#SBATCH --output=job_outputs/strace_job_out_%j.out
#SBATCH --error=job_outputs/strace_job_err_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=42gb
#SBATCH --time=47:00:00
#SBATCH --partition=gpupart_a40  # gpupart_v100, gpupart_a40, gpu_h100, dibdp (AIRAWAT)
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
#   sbatch eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --slurm
# Resumable (checkpointing): add --resumable; re-run same command to resume after crash
# Alternative (Hydra): sbatch --wrap "conda run -n strace python launch_eval.py task_name=gsm8k mode=pd"
# Multirun on SLURM (one job per config): python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun launcher=slurm
#
# CST (optional, via eval.LLaDAEvalHarness in eval.py):
#   --cst_checkpoint PATH         forces --capture_pre_logits true, buffers .pt to cst_target_staging_gb, eval+delete
#   [--cst_config_path PATH]      default config/cst.yaml
#   [--cst_ntf_conf_low HIGH]   [--cst_ntf_conf_high HIGH]  [--cst_target_staging_gb GB]
# Example: --capture_pre_logits true --cst_checkpoint ./outputs/cst_ckpt/cst_llada_scheduler_pd_k4096_w4_train_best.pt



########################################################################################
# Parse command-line arguments
# Usage: ./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada [--block_length 64] [--compile_model False] [--intervention_mode drop] [--intervention_steps 0,128,256] [--intervention_k 1] [--wandb True] [--tracing_enabled True] [--capture_pre_logits True] [--hidden_capture_path outputs/my_hidden] [--slurm]
# Mode shortcuts: pd (pure_diffusion), bd (block_diffusion), eb (eb_sampler), fdm (fdm), fhs (fhs), top_k_c (top_k_confidence), top_k_m (top_k_margin), dus (dus)
# Intervention and wandb are optional. Datasets: gsm8k, minerva_math500, humaneval, mbpp, arc_challenge_chat
# --model: basename of a folder under ./models/ (must match lm-eval registration). Examples: llada, llada1_5, dream.
#          Weights/tokenizer load from ./models/<model>/; mask/EOS/EoT ids come from that folder's config when not overridden.
# --slurm: run the eval command under srun (use when script is submitted via sbatch so the job uses the allocated GPU).
# Local GPU: run without --slurm; optionally export CUDA_VISIBLE_DEVICES=0 or 1  before running.

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
NUM_FEWSHOT_OVERRIDE=""
BLOCK_LENGTH_OVERRIDE=""
TOP_K_OVERRIDE=""
SLURM_MODE=false
RESUMABLE=false
RUN_ID=""
CAPTURE_PRE_LOGITS=""
HIDDEN_CAPTURE_PATH=""
CST_CHECKPOINT=""
CST_CONFIG_PATH=""
CST_NTF_LO=""
CST_NTF_HI=""
CST_TGB=""
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
        --num_fewshot)
            NUM_FEWSHOT_OVERRIDE="$2"
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
        --capture_pre_logits)
            CAPTURE_PRE_LOGITS="$2"
            shift 2
            ;;
        --hidden_capture_path)
            HIDDEN_CAPTURE_PATH="$2"
            shift 2
            ;;
        --cst_checkpoint)
            CST_CHECKPOINT="$2"
            shift 2
            ;;
        --cst_config_path)
            CST_CONFIG_PATH="$2"
            shift 2
            ;;
        --cst_ntf_conf_low)
            CST_NTF_LO="$2"
            shift 2
            ;;
        --cst_ntf_conf_high)
            CST_NTF_HI="$2"
            shift 2
            ;;
        --cst_target_staging_gb)
            CST_TGB="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --batch_size <size> --mode <mode> --task_name <task> --model <model> [--num_fewshot <n>] [--block_length <n>] [--top_k <n>] [--compile_model <bool>] [--intervention_mode <mode>] [--intervention_steps <steps>] [--intervention_k <k>] [--wandb <True|False>] [--tracing_enabled <True|False>] [--capture_pre_logits <True|False>] [--hidden_capture_path <dir>] [--slurm] [--resumable] [--run_id <id>]"
            exit 1
            ;;
    esac
done

# CST needs hidden .pt (same as launch_eval.py: cst_checkpoint => capture on)
if [[ -n "$CST_CHECKPOINT" ]]; then
    CAPTURE_PRE_LOGITS="true"
fi

# Check required arguments (intervention and wandb are optional)
if [[ -z "$BATCH_SIZE" || -z "$MODE" || -z "$TASK_NAME" || -z "$MODEL" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 --batch_size <size> --mode <mode> --task_name <task> --model <model> [--num_fewshot <n>] [--block_length <n>] [--top_k <n>] [--compile_model <bool>] [--intervention_mode <mode>] [--intervention_steps <steps>] [--intervention_k <k>] [--wandb <True|False>] [--tracing_enabled <True|False>] [--slurm]"
    echo "  Mode shortcuts: pd (pure_diffusion), bd (block_diffusion), eb (eb_sampler), fdm (fdm), fhs (fhs), top_k_c (top_k_confidence), top_k_m (top_k_margin), dus (dus)"
    echo "  Optional - Intervention: --intervention_mode drop|swap, --intervention_steps 0,128,256, --intervention_k 1"
    echo "  Optional - Wandb: --wandb True to log to wandb"
    echo "  Optional - Few-shot: --num_fewshot <n> to override per-task default (defaults: humaneval/arc_challenge_chat 0, gsm8k/math500 4, mbpp 3)"
    echo "  Optional - Block length: --block_length <n> to override (defaults: block_diffusion 32, fdm 64, dus 16, else gen_length)"
    echo "  Optional - Top-k: --top_k <n> for top_k scheduler (default 2, tokens to transfer per step)"
    echo "  Optional - Tracing: --tracing_enabled True to enable scheduler tracing (writes trace_outputs/<trace_name>.json)"
            echo "  Optional - Hidden states: --capture_pre_logits True [--hidden_capture_path <dir>] (default dir under outputs/ or checkpoint/hidden)"
            echo "  Optional - CST: --cst_checkpoint <.pt> (implies capture; also --cst_config_path, --cst_target_staging_gb, --cst_ntf_conf_low/high)"
    echo "  Optional - SLURM: --slurm to run the eval under srun (for sbatch submissions)"
    echo "  Optional - Resumable: --resumable writes outputs/{run_id}/checkpoint.jsonl and resumes skipped samples; traces use JSONL when tracing is on. Use --run_id for custom id."
    echo "Example: $0 --batch_size 2 --mode pd --task_name gpqa_main_zeroshot --model llada --compile_model False"
    echo "Example with intervention: $0 --batch_size 2 --mode pd --task_name gsm8k --model llada --intervention_mode drop --intervention_steps 0,128,256 --intervention_k 1 --wandb True"
    exit 1
fi

# Convert mode shortcuts to full mode names
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
        echo "Warning: Unknown mode '$MODE'. Using as-is. Valid modes: pd, bd, pure_diffusion, block_diffusion"
        ;;
esac

# Initialize conda for this shell session
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate strace

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export LMEVAL_LOG_LEVEL=INFO
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=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

# When running locally (not --slurm), you can set CUDA_VISIBLE_DEVICES; when --slurm, leave to SLURM allocation.
if [[ "$SLURM_MODE" != "true" ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

# Wandb configuration 
export WANDB_API_KEY=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
export WANDB_PROJECT=strace

# Tracing: enable only when --tracing_enabled True (default off to avoid overhead)
if [[ "${TRACING_ENABLED_ARG,,}" == "true" ]]; then
    export TRACING_ENABLED=TRUE
else
    export TRACING_ENABLED=FALSE
fi

# Set gen_length and steps (arc_challenge_chat: 128; others: 512)
if [[ "$TASK_NAME" == *"arc_challenge_chat"* ]]; then
    export GEN_LENGTH=128
    export STEPS=128
else
    export GEN_LENGTH=512
    export STEPS=512
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
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default compile_model if not set
if [[ -z "$COMPILE_MODEL" ]]; then
    COMPILE_MODEL="False"
fi

# Generate wandb name and output suffix: MODEL_TASK_MODE; use stable run_id when resumable for checkpointing
if [[ "$RESUMABLE" == "true" ]]; then
    if [[ -n "$RUN_ID" ]]; then
        OUTPUT_SUFFIX="$RUN_ID"
    else
        OUTPUT_SUFFIX="${MODEL}_${TASK_NAME}_${MODE}"
        if [[ -n "$INTERVENTION_MODE" ]]; then
            INTERVENTION_STEPS="${INTERVENTION_STEPS:-0}"
            INTERVENTION_STEPS_SAFE="${INTERVENTION_STEPS//,/_}"
            OUTPUT_SUFFIX="${MODEL}_${TASK_NAME}_${MODE}_intv_${INTERVENTION_MODE}_steps${INTERVENTION_STEPS_SAFE}_k${INTERVENTION_K:-1}"
        fi
    fi
    CHECKPOINT_DIR="outputs/${OUTPUT_SUFFIX}"
    mkdir -p "$CHECKPOINT_DIR"
else
    OUTPUT_SUFFIX="${MODEL}_${TASK_NAME}_${MODE}_${TIMESTAMP}"
    if [[ -n "$INTERVENTION_MODE" ]]; then
        INTERVENTION_STEPS="${INTERVENTION_STEPS:-0}"
        INTERVENTION_STEPS_SAFE="${INTERVENTION_STEPS//,/_}"
        OUTPUT_SUFFIX="${MODEL}_${TASK_NAME}_${MODE}_intv_${INTERVENTION_MODE}_steps${INTERVENTION_STEPS_SAFE}_k${INTERVENTION_K:-1}_${TIMESTAMP}"
    fi
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
MODEL_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},mode=${MODE},strategy=${STRATEGY},top_k=${TOP_K},compile_model=${COMPILE_MODEL},task_name=${TASK_NAME},trace_name=${OUTPUT_SUFFIX}"
if [[ "$RESUMABLE" == "true" ]]; then
    MODEL_ARGS="${MODEL_ARGS},resumable=true,output_dir=${CHECKPOINT_DIR}"
fi
if [[ -n "$INTERVENTION_MODE" ]]; then
    # Use underscores for intervention_steps so commas don't break model_args parsing
    INTERVENTION_STEPS_ARG="${INTERVENTION_STEPS//,/_}"
    MODEL_ARGS="${MODEL_ARGS},intervention_mode=${INTERVENTION_MODE},intervention_steps=${INTERVENTION_STEPS_ARG},intervention_k=${INTERVENTION_K:-1}"
fi

# Pre-logit hidden capture (cst): paths must not contain commas (lm-eval model_args)
if [[ "${CAPTURE_PRE_LOGITS,,}" == "true" ]]; then
    if [[ -n "$HIDDEN_CAPTURE_PATH" ]]; then
        HIDDEN_DIR="$HIDDEN_CAPTURE_PATH"
    elif [[ "$RESUMABLE" == "true" && -n "$CHECKPOINT_DIR" ]]; then
        HIDDEN_DIR="${CHECKPOINT_DIR}/hidden"
    else
        HIDDEN_DIR="outputs/${OUTPUT_SUFFIX}/hidden"
    fi
    mkdir -p "$HIDDEN_DIR"
    MODEL_ARGS="${MODEL_ARGS},capture_pre_logits=true,hidden_capture_path=${HIDDEN_DIR}"
fi

# CST: appended to --model_args (see eval.LLaDAEvalHarness; paths must not contain commas)
if [[ -n "$CST_CHECKPOINT" ]]; then
    NLO="${CST_NTF_LO:-0.2}"
    NHI="${CST_NTF_HI:-0.8}"
    TGB="${CST_TGB:-1.0}"
    MODEL_ARGS="${MODEL_ARGS},cst_checkpoint=${CST_CHECKPOINT},cst_ntf_conf_low=${NLO},cst_ntf_conf_high=${NHI},cst_target_staging_gb=${TGB}"
    if [[ -n "$CST_CONFIG_PATH" ]]; then
        MODEL_ARGS="${MODEL_ARGS},cst_config_path=${CST_CONFIG_PATH}"
    fi
fi

# Display configuration
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
echo "  Num Fewshot: $NUM_FEWSHOT"
if [[ -n "$INTERVENTION_MODE" ]]; then
    echo "  Intervention Mode: $INTERVENTION_MODE"
    echo "  Intervention Steps: $INTERVENTION_STEPS"
    echo "  Intervention K: ${INTERVENTION_K:-1}"
fi
echo "  Wandb: ${WANDB:-False}"
echo "  Tracing enabled: ${TRACING_ENABLED_ARG:-False}"
if [[ "${CAPTURE_PRE_LOGITS,,}" == "true" ]]; then
    echo "  Capture pre-logit hiddens: yes (dir=${HIDDEN_DIR})"
fi
if [[ -n "$CST_CHECKPOINT" ]]; then
    echo "  CST: checkpoint=${CST_CHECKPOINT} target_gb=${CST_TGB:-1.0} ntf=[$NLO,$NHI]"
    [[ -n "$CST_CONFIG_PATH" ]] && echo "  CST config: ${CST_CONFIG_PATH}"
fi
echo "  SLURM mode (srun): ${SLURM_MODE}"
if [[ -n "$WANDB_NAME" ]]; then
    echo "  Wandb Name: $WANDB_NAME"
fi
if [[ "$RESUMABLE" == "true" ]]; then
    echo "  Resumable: yes (checkpoint=${CHECKPOINT_DIR})"
else
    echo "  Output: outputs/${OUTPUT_SUFFIX}.json"
fi
echo ""

# Build accelerate launch command
# When resumable: checkpoint.jsonl skips completed samples; pass wandb resume id if .wandb_id exists
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
    # Resumable: checkpoint-based skip for completed samples; output_path in checkpoint dir for final metrics
    ACCEL_CMD="accelerate launch eval.py --tasks ${TASK_NAME} --batch_size ${BATCH_SIZE} --num_fewshot ${NUM_FEWSHOT} --model ${MODEL} --model_args ${MODEL_ARGS} --write_out --output_path ${CHECKPOINT_DIR}/results.json --log_samples --cache_requests true"
else
    ACCEL_CMD="accelerate launch eval.py --tasks ${TASK_NAME} --batch_size ${BATCH_SIZE} --num_fewshot ${NUM_FEWSHOT} --model ${MODEL} --model_args ${MODEL_ARGS} --write_out --output_path outputs/${OUTPUT_SUFFIX}.json --log_samples --cache_requests true"
fi
if [[ "${WANDB,,}" == "true" ]]; then
    ACCEL_CMD="${ACCEL_CMD} --wandb_args project=${WANDB_PROJECT},name=${WANDB_NAME}${WANDB_RESUME_ARGS}"
fi
if [[ -n "$RESUME_ENV" ]]; then
    ACCEL_CMD="env ${RESUME_ENV} ${ACCEL_CMD}"
fi

# Run: under srun when --slurm (for sbatch), else directly (local GPU)
if [[ "$SLURM_MODE" == "true" ]]; then
    srun bash -c "$ACCEL_CMD"
else
    eval $ACCEL_CMD
fi

