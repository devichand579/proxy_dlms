#!/bin/bash
#SBATCH --job-name=eval_strace
#SBATCH --output=job_outputs/strace_job_out_%j.out
#SBATCH --error=job_outputs/strace_job_err_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=72:00:00
#SBATCH --partition=dgx2  # gpupart_v100, gpupart_a40, gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu2  # ignore if your cluster doesn't have a qos for GPUs


########################################################################################
# Single eval job via sbatch: run this script with your args and --slurm, e.g.:
#   sbatch eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --slurm
# Alternative (Hydra): sbatch --wrap "conda run -n strace python launch_eval.py task_name=gsm8k mode=pd"
# Multirun on SLURM (one job per config): python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun launcher=slurm



########################################################################################
# Parse command-line arguments
# Usage: ./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada [--compile_model False] [--intervention_mode drop] [--intervention_steps 0,128,256] [--intervention_k 1] [--wandb True] [--tracing_enabled True] [--slurm]
# Mode shortcuts: pd (pure_diffusion), bd (block_diffusion), eb (eb_sampler), fdm (fdm), fhs (fhs), top_k_c (top_k_confidence), top_k_m (top_k_margin), dus (dus)
# Intervention and wandb are optional. Datasets: gsm8k, minerva_math500, humaneval, mbpp, arc_challenge_chat
# --slurm: run the eval command under srun (use when script is submitted via sbatch so the job uses the allocated GPU).
# Local GPU: run without --slurm; optionally export CUDA_VISIBLE_DEVICES=0 before running.

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
SLURM_MODE=false
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
        --slurm)
            SLURM_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --batch_size <size> --mode <mode> --task_name <task> --model <model> [--compile_model <bool>] [--intervention_mode <mode>] [--intervention_steps <steps>] [--intervention_k <k>] [--wandb <True|False>] [--tracing_enabled <True|False>] [--slurm]"
            exit 1
            ;;
    esac
done

# Check required arguments (intervention and wandb are optional)
if [[ -z "$BATCH_SIZE" || -z "$MODE" || -z "$TASK_NAME" || -z "$MODEL" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 --batch_size <size> --mode <mode> --task_name <task> --model <model> [--compile_model <bool>] [--intervention_mode <mode>] [--intervention_steps <steps>] [--intervention_k <k>] [--wandb <True|False>] [--tracing_enabled <True|False>] [--slurm]"
    echo "  Mode shortcuts: pd (pure_diffusion), bd (block_diffusion), eb (eb_sampler), fdm (fdm), fhs (fhs), top_k_c (top_k_confidence), top_k_m (top_k_margin), dus (dus)"
    echo "  Optional - Intervention: --intervention_mode drop|swap, --intervention_steps 0,128,256, --intervention_k 1"
    echo "  Optional - Wandb: --wandb True to log to wandb"
    echo "  Optional - Tracing: --tracing_enabled True to enable scheduler tracing (writes trace_outputs/<trace_name>.json)"
    echo "  Optional - SLURM: --slurm to run the eval under srun (for sbatch submissions)"
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
        ;;
    fhs)
        MODE="fhs"
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

# Set block_length based on mode
# If mode is "block_diffusion": block_length=32
# Otherwise: block_length=gen_length
if [[ "$MODE" == "block_diffusion" ]]; then
    export BLOCK_LENGTH=32
else
    export BLOCK_LENGTH=$GEN_LENGTH
fi

export MODEL_PATH=./models/${MODEL}
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default compile_model if not set
if [[ -z "$COMPILE_MODEL" ]]; then
    COMPILE_MODEL="False"
fi

# Generate wandb name and output suffix: MODEL_TASK_MODE_TIMESTAMP; append intervention info when intervention_mode is set
OUTPUT_SUFFIX="${MODEL}_${TASK_NAME}_${MODE}_${TIMESTAMP}"
WANDB_NAME="${OUTPUT_SUFFIX}"
if [[ -n "$INTERVENTION_MODE" ]]; then
    INTERVENTION_STEPS="${INTERVENTION_STEPS:-0}"
    INTERVENTION_STEPS_SAFE="${INTERVENTION_STEPS//,/_}"
    OUTPUT_SUFFIX="${MODEL}_${TASK_NAME}_${MODE}_intv_${INTERVENTION_MODE}_steps${INTERVENTION_STEPS_SAFE}_k${INTERVENTION_K:-1}_${TIMESTAMP}"
    WANDB_NAME="${OUTPUT_SUFFIX}"
fi

# Set num_fewshot based on task name
NUM_FEWSHOT=0  # Default
if [[ "$TASK_NAME" == *"humaneval"* ]]; then
    NUM_FEWSHOT=0
elif [[ "$TASK_NAME" == *"gsm8k"* ]]; then
    NUM_FEWSHOT=4
elif [[ "$TASK_NAME" == *"math500"* ]] || [[ "$TASK_NAME" == *"minerva_math500"* ]]; then
    NUM_FEWSHOT=4
elif [[ "$TASK_NAME" == *"mbpp3"* ]]; then
    NUM_FEWSHOT=3
elif [[ "$TASK_NAME" == *"mbpp"* ]]; then
    NUM_FEWSHOT=3
elif [[ "$TASK_NAME" == *"arc_challenge_chat"* ]]; then
    NUM_FEWSHOT=5
fi

# Build model_args (base args + optional intervention params)
MODEL_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},mode=${MODE},compile_model=${COMPILE_MODEL},task_name=${TASK_NAME},trace_name=${OUTPUT_SUFFIX}"
if [[ -n "$INTERVENTION_MODE" ]]; then
    # Use underscores for intervention_steps so commas don't break model_args parsing
    INTERVENTION_STEPS_ARG="${INTERVENTION_STEPS//,/_}"
    MODEL_ARGS="${MODEL_ARGS},intervention_mode=${INTERVENTION_MODE},intervention_steps=${INTERVENTION_STEPS_ARG},intervention_k=${INTERVENTION_K:-1}"
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
echo "  Compile Model: $COMPILE_MODEL"
echo "  Num Fewshot: $NUM_FEWSHOT"
if [[ -n "$INTERVENTION_MODE" ]]; then
    echo "  Intervention Mode: $INTERVENTION_MODE"
    echo "  Intervention Steps: $INTERVENTION_STEPS"
    echo "  Intervention K: ${INTERVENTION_K:-1}"
fi
echo "  Wandb: ${WANDB:-False}"
echo "  Tracing enabled: ${TRACING_ENABLED_ARG:-False}"
echo "  SLURM mode (srun): ${SLURM_MODE}"
if [[ -n "$WANDB_NAME" ]]; then
    echo "  Wandb Name: $WANDB_NAME"
fi
echo "  Output: outputs/${OUTPUT_SUFFIX}.json"
echo ""

# Build accelerate launch command (add --wandb_args only when --wandb True)
ACCEL_CMD="accelerate launch eval.py --tasks ${TASK_NAME} --batch_size ${BATCH_SIZE} --num_fewshot ${NUM_FEWSHOT} --model ${MODEL} --model_args ${MODEL_ARGS} --write_out --output_path outputs/${OUTPUT_SUFFIX}.json --log_samples --cache_requests true"
if [[ "${WANDB,,}" == "true" ]]; then
    ACCEL_CMD="${ACCEL_CMD} --wandb_args project=${WANDB_PROJECT},name=${WANDB_NAME}"
fi

# Run: under srun when --slurm (for sbatch), else directly (local GPU)
if [[ "$SLURM_MODE" == "true" ]]; then
    srun bash -c "$ACCEL_CMD"
else
    eval $ACCEL_CMD
fi

