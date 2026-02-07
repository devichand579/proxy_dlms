#!/bin/bash

# Parse command-line arguments
# Usage: ./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --compile_model False
# Mode shortcuts: pd (pure_diffusion), bd (block_diffusion)
# Datasets : gsm8k, minerva_math500, humaneval, mbpp, logiqa, bbh

BATCH_SIZE=""
MODE=""
TASK_NAME=""
MODEL=""
COMPILE_MODEL=""
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --batch_size <size> --mode <mode> --task_name <task> --model <model> --compile_model <compile_model>"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$BATCH_SIZE" || -z "$MODE" || -z "$TASK_NAME" || -z "$MODEL" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 --batch_size <size> --mode <mode> --task_name <task> --model <model> --compile_model <compile_model>"
    echo "  Mode shortcuts: pd (pure_diffusion), bd (block_diffusion)"
    echo "Example: $0 --batch_size 2 --mode pd --task_name gpqa_main_zeroshot --model llada --compile_model False"
    exit 1
fi

# Convert mode shortcuts to full mode names
case "$MODE" in
    pd)
        MODE="pure_diffusion"
        ;;
    bd)
        MODE="block_diffusion"
        ;;
    pure_diffusion|block_diffusion)
        # Already full name, keep as is
        ;;
    *)
        echo "Warning: Unknown mode '$MODE'. Using as-is. Valid modes: pd, bd, pure_diffusion, block_diffusion"
        ;;
esac

conda init
conda activate strace

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export LMEVAL_LOG_LEVEL=INFO
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=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
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0  # UNSET if are running with SLURM

# Wandb configuration 
export WANDB_API_KEY=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
export WANDB_PROJECT=strace
export TRACING_ENABLED=FALSE

# Set gen_length and steps based on task
# For humaneval and mbpp: gen_length=512, steps=512
# For others: gen_length=256, steps=256
if [[ "$TASK_NAME" == *"humaneval"* ]] || [[ "$TASK_NAME" == *"mbpp"* ]]; then
    export GEN_LENGTH=512
    export STEPS=512
else
    export GEN_LENGTH=256
    export STEPS=256
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

# Generate wandb name from same pattern as output file: MODEL_TASK_MODE_TIMESTAMP
WANDB_NAME="${MODEL}_${TASK_NAME}_${MODE}_${TIMESTAMP}"

# Set num_fewshot based on task name
NUM_FEWSHOT=0  # Default
if [[ "$TASK_NAME" == *"humaneval"* ]]; then
    NUM_FEWSHOT=0
elif [[ "$TASK_NAME" == *"gsm8k"* ]]; then
    NUM_FEWSHOT=4
elif [[ "$TASK_NAME" == *"math500"* ]] || [[ "$TASK_NAME" == *"minerva_math500"* ]]; then
    NUM_FEWSHOT=4
elif [[ "$TASK_NAME" == *"mbpp"* ]]; then
    NUM_FEWSHOT=4
elif [[ "$TASK_NAME" == *"bbh"* ]]; then
    NUM_FEWSHOT=3
elif [[ "$TASK_NAME" == *"truthfulqa"* ]]; then
    NUM_FEWSHOT=0
fi

# Display configuration
echo "Configuration:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Mode: $MODE"
echo "  Task Name: $TASK_NAME"
echo "  Wandb Name: $WANDB_NAME"
echo "  Model: $MODEL"
echo "  Model Path: $MODEL_PATH"
echo "  Gen Length: $GEN_LENGTH"
echo "  Steps: $STEPS"
echo "  Block Length: $BLOCK_LENGTH"
echo "  Compile Model: $COMPILE_MODEL"
echo "  Num Fewshot: $NUM_FEWSHOT"
echo ""
accelerate launch eval.py -L 4 --tasks ${TASK_NAME} --batch_size ${BATCH_SIZE} --num_fewshot ${NUM_FEWSHOT} --model ${MODEL} --model_args model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},mode=${MODE},compile_model=${COMPILE_MODEL},task_name=${TASK_NAME} --wandb_args project=${WANDB_PROJECT},name=${WANDB_NAME} --write_out --output_path outputs/${MODEL}_${TASK_NAME}_${MODE}_${TIMESTAMP}.json --log_samples --cache_requests

