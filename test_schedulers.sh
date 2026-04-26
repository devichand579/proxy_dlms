#!/bin/bash
# ONLY FOR TESTING
# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Test different schedulers with configurable environment variables and options.

ENVIRONMENT VARIABLE OPTIONS:
  --tracing_enabled VALUE     Set TRACING_ENABLED (default: True)
  --wandb_api_key KEY         Set WANDB_API_KEY (default: preset key)
  --wandb_project PROJECT     Set WANDB_PROJECT (default: strace)

SCHEDULER OPTIONS:
  --scheduler SCHEDULER        Scheduler to test. Available schedulers:
                              - pure_diffusion
                              - block_diffusion
                              - top_k_confidence
                              - top_k_margin
                              - fhs
                              - eb_sampler
                              - dus
                              - fdm
                              If not specified, all schedulers will be tested.

WANDB OPTIONS:
  --wandb                     Enable wandb logging and trace uploads
  --wandb_name NAME           Custom wandb run name (default: auto-generated)
  --wandb_project PROJECT     Wandb project name (default: strace)

INTERVENTION OPTIONS:
  --intervention_mode MODE    Intervention mode: "drop" or "swap"
                              "drop": prevent tokens from being unmasked
                              "swap": swap tokens

BATCH OPTIONS:
  --batch                     Run batch execution tests only (batched generation)
  --batch_size N              Batch size for --batch tests (default: 2)

EOS HARNESS (pass-through; same as: python test_schedulers.py --no-eos-harness):
  --no-eos-harness            Full step budget + full-tail decode (reference LLaDA-style; no EOS early stop)

DEBUG (pass-through; python test_schedulers.py --show-original-tokens):
  --show-original-tokens      Print full token id lists and decode (skip_special_tokens=False) before stripped decode

MODEL OPTIONS:
  --model_path DIR            Local checkpoint directory; repeat to test several models
                              (e.g. ./models/llada and ./models/dream). Default: ./models/llada

OTHER OPTIONS:
  --help, -h                  Show this help message

EXAMPLES:
  # Test pure_diffusion scheduler with drop intervention
  $0 --scheduler pure_diffusion --intervention_mode drop

  # Test FHS scheduler with wandb logging
  $0 --scheduler fhs --wandb --wandb_project myproject

  # Test with custom tracing disabled
  $0 --scheduler top_k_confidence --tracing_enabled False

  # Run batch execution tests (all schedulers, batch_size=2)
  $0 --batch

  # Run batch tests for one scheduler with batch_size 4
  $0 --batch --scheduler pure_diffusion --batch_size 4

  # Test all schedulers on two local models
  $0 --model_path ./models/llada --model_path ./models/dream

  # Test all schedulers (default behavior, default model)
  $0

EOF
}

# Default values (mirror .env.example / repo_env_defaults.py)
TRACING_ENABLED="${TRACING_ENABLED:-True}"
HF_TOKEN="${HF_TOKEN:-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}"
WANDB_API_KEY="${WANDB_API_KEY:-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}"
WANDB_PROJECT="${WANDB_PROJECT:-strace}"

# Parse arguments
USE_WANDB=false
WANDB_NAME=""
SCHEDULER=""
INTERVENTION_MODE=""
USE_BATCH=false
BATCH_SIZE=2
MODEL_PATHS=()
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --tracing_enabled)
            TRACING_ENABLED="$2"
            shift 2
            ;;
        --wandb_api_key)
            WANDB_API_KEY="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --wandb_name)
            WANDB_NAME="$2"
            shift 2
            ;;
        --scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        --intervention_mode)
            INTERVENTION_MODE="$2"
            shift 2
            ;;
        --batch)
            USE_BATCH=true
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATHS+=("$2")
            shift 2
            ;;
        *)
            # Pass through any other arguments to the Python script
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

# Initialize conda for this shell session
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate strace

# Export environment variables
export TRACING_ENABLED
export HF_TOKEN
export WANDB_API_KEY
export WANDB_PROJECT
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Build Python command array
PYTHON_CMD_ARGS=("test_schedulers.py")

if [ -n "$SCHEDULER" ]; then
    PYTHON_CMD_ARGS+=("--scheduler" "$SCHEDULER")
fi

if [ "$USE_WANDB" = true ]; then
    PYTHON_CMD_ARGS+=("--wandb")
fi

if [ -n "$WANDB_NAME" ]; then
    PYTHON_CMD_ARGS+=("--wandb_name" "$WANDB_NAME")
fi

if [ -n "$WANDB_PROJECT" ]; then
    PYTHON_CMD_ARGS+=("--wandb_project" "$WANDB_PROJECT")
fi

if [ -n "$INTERVENTION_MODE" ]; then
    PYTHON_CMD_ARGS+=("--intervention_mode" "$INTERVENTION_MODE")
fi

for mp in "${MODEL_PATHS[@]}"; do
    PYTHON_CMD_ARGS+=("--model_path" "$mp")
done

if [ "$USE_BATCH" = true ]; then
    PYTHON_CMD_ARGS+=("--batch")
    PYTHON_CMD_ARGS+=("--batch_size" "$BATCH_SIZE")
fi

# Add any additional arguments passed through
if [ ${#PYTHON_ARGS[@]} -gt 0 ]; then
    PYTHON_CMD_ARGS+=("${PYTHON_ARGS[@]}")
fi

# Run the Python script
python "${PYTHON_CMD_ARGS[@]}"