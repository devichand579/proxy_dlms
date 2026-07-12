#!/bin/bash
#SBATCH --job-name=lladalens
#SBATCH --output=job_outputs/lladalens_job_out_%j.out
#SBATCH --error=job_outputs/lladalens_job_err_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=47:00:00
#SBATCH --partition=gpu_h100  # gpu_l40, gpu_h100, dibdp (AIRAWAT)
#SBATCH --gres=gpu:1

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
# LLaDALens PCFG train / scheduler eval
# Config defaults: config/lladalens.yaml  |  Hydra entry: run_lladalens.py
#
# Flags below override config only when set; omitted flags use config/lladalens.yaml.
#
# Interactive:
#   ./lladalens.sh --mode train --variant 100m --hf_dataset pcfg-code --batch_size 128
#   ./lladalens.sh --mode eval --hf_model username/llada-100m-pcfg --schedulers all
#
# Slurm:
#   sbatch lladalens.sh --mode train --hf_dataset pcfg-code --batch_size 128 --slurm
#   sbatch lladalens.sh --mode train --nproc 4 --hf_dataset pcfg-code --slurm
#
# Hydra multirun (login node, one job per combo):
#   python run_lladalens.py mode=train train.variant=100m,200m --multirun launcher=slurm_h100
########################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
LLaDALens launcher — config defaults in config/lladalens.yaml; flags override Hydra.

  ./lladalens.sh [OPTIONS] [extra hydra overrides...]

Run:
  --mode train|eval       (default: train)
  --slurm                 use srun inside a Slurm allocation
  --device DEV            0|1|cuda|cuda:0|cpu
  --offline               cached HF + local W&B
  --launcher NAME         Hydra launcher group (local|slurm_h100|slurm_l40|...)

Train (maps to train.* in config):
  --variant 100m|200m|300m
  --corpus PATH           local save_to_disk corpus (lladalens/-relative)
  --hf_dataset NAME       Hub dataset short name or org/repo
  --hf_namespace ORG      org override for short --hf_dataset
  --train_tokens N        on-the-fly sample budget (mutually exclusive with corpus/hf)
  --batch_size N
  --grad_accum N
  --num_workers N
  --nproc N               torchrun DDP GPUs (train.nproc_per_node; match #SBATCH --gres)
  --peak_lr F --mid_lr F --min_lr F
  --warmup_steps N --warmup_fraction F --warmup_max N
  --epochs N --steps N
  --weight_decay F --max_grad_norm F --seed N
  --out_dir PATH --resume PATH
  --log_every N --save_every N
  --val_fraction F --val_every N
  --push_model            push checkpoint-best_val to Hub after train
  --hf_model REPO         HF model repo for --push_model
  --hf_model_namespace ORG
  --hf_model_private
  --amp_dtype float16|bfloat16
  --no_amp
  --wandb true|false
  --wandb_project NAME --wandb_run_name NAME --wandb_id ID

Eval (maps to eval.* in config):
  --model_path PATH       local checkpoint dir (lladalens/-relative)
  --hf_model REPO         HF model repo (alternative to --model_path)
  --variant LABEL         trace/output label (default 100m in config)
  --corpus PATH           corpus with test split
  --hf_dataset NAME
  --hf_namespace ORG
  --schedulers LIST       all or comma list: pd,bd,eb,fdm,fhs,top_k_c,top_k_m,dus
  --max_samples N
  --steps N --batch_size N
  --out_dir PATH
  --tracing_enabled true|false
  --trace_hf_upload true|false
  --trace_hf_dataset_repo REPO
  --trace_hf_namespace ORG
  --no_upload_traces
  --no_replay_oracle

Examples:
  ./lladalens.sh --mode train --variant 100m --hf_dataset pcfg-code --batch_size 128 --epochs 3
  ./lladalens.sh --mode train --corpus data/pcfg --steps 5000 --val_every 2000
  ./lladalens.sh --mode eval --model_path checkpoints/llada-100m-pcfg/checkpoint-best_val --schedulers all
  sbatch lladalens.sh --mode train --hf_dataset pcfg-code --nproc 4 --slurm

Extra overrides (advanced): append hydra key=value after flags, e.g. train.mid_lr=2e-4
EOF
}

# --- run ---
MODE="train"
SLURM_MODE=false
DEVICE=""
OFFLINE=false
LAUNCHER=""

# --- train ---
VARIANT=""
CORPUS=""
HF_DATASET=""
HF_NAMESPACE=""
TRAIN_TOKENS=""
BATCH_SIZE=""
GRAD_ACCUM=""
NUM_WORKERS=""
NPROC=""
PEAK_LR=""
MID_LR=""
MIN_LR=""
WARMUP_STEPS=""
WARMUP_FRACTION=""
WARMUP_MAX=""
EPOCHS=""
STEPS=""
WEIGHT_DECAY=""
MAX_GRAD_NORM=""
SEED=""
OUT_DIR=""
RESUME=""
LOG_EVERY=""
SAVE_EVERY=""
VAL_FRACTION=""
VAL_EVERY=""
PUSH_MODEL=false
HF_MODEL=""
HF_MODEL_NAMESPACE=""
HF_MODEL_PRIVATE=false
AMP_DTYPE=""
NO_AMP=false
WANDB=""
WANDB_PROJECT=""
WANDB_RUN_NAME=""
WANDB_ID=""

# --- eval ---
MODEL_PATH=""
EVAL_VARIANT=""
EVAL_CORPUS=""
EVAL_HF_DATASET=""
EVAL_HF_NAMESPACE=""
SCHEDULERS=""
MAX_SAMPLES=""
EVAL_STEPS=""
EVAL_BATCH_SIZE=""
EVAL_OUT_DIR=""
TRACING_ENABLED=""
TRACE_HF_UPLOAD=""
TRACE_HF_DATASET_REPO=""
TRACE_HF_NAMESPACE=""
NO_UPLOAD_TRACES=false
NO_REPLAY_ORACLE=false

EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --slurm)
      SLURM_MODE=true
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --offline)
      OFFLINE=true
      shift
      ;;
    --launcher)
      LAUNCHER="$2"
      shift 2
      ;;

    # train data
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --corpus)
      CORPUS="$2"
      shift 2
      ;;
    --hf_dataset|--hf-dataset)
      HF_DATASET="$2"
      shift 2
      ;;
    --hf_namespace|--hf-namespace)
      HF_NAMESPACE="$2"
      shift 2
      ;;
    --train_tokens|--train-tokens)
      TRAIN_TOKENS="$2"
      shift 2
      ;;

    # train hparams
    --batch_size|--batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad_accum|--grad-accum)
      GRAD_ACCUM="$2"
      shift 2
      ;;
    --num_workers|--num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --peak_lr|--peak-lr)
      PEAK_LR="$2"
      shift 2
      ;;
    --mid_lr|--mid-lr)
      MID_LR="$2"
      shift 2
      ;;
    --min_lr|--min-lr)
      MIN_LR="$2"
      shift 2
      ;;
    --warmup_steps|--warmup-steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    --warmup_fraction|--warmup-fraction)
      WARMUP_FRACTION="$2"
      shift 2
      ;;
    --warmup_max|--warmup-max)
      WARMUP_MAX="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --steps)
      if [[ "$MODE" == "eval" ]]; then
        EVAL_STEPS="$2"
      else
        STEPS="$2"
      fi
      shift 2
      ;;
    --weight_decay|--weight-decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --max_grad_norm|--max-grad-norm)
      MAX_GRAD_NORM="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --out_dir|--out-dir)
      if [[ "$MODE" == "eval" ]]; then
        EVAL_OUT_DIR="$2"
      else
        OUT_DIR="$2"
      fi
      shift 2
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    --log_every|--log-every)
      LOG_EVERY="$2"
      shift 2
      ;;
    --save_every|--save-every)
      SAVE_EVERY="$2"
      shift 2
      ;;
    --val_fraction|--val-fraction)
      VAL_FRACTION="$2"
      shift 2
      ;;
    --val_every|--val-every)
      VAL_EVERY="$2"
      shift 2
      ;;
    --push_model|--push-model)
      PUSH_MODEL=true
      shift
      ;;
    --hf_model|--hf-model)
      HF_MODEL="$2"
      shift 2
      ;;
    --hf_model_namespace|--hf-model-namespace)
      HF_MODEL_NAMESPACE="$2"
      shift 2
      ;;
    --hf_model_private|--hf-model-private)
      HF_MODEL_PRIVATE=true
      shift
      ;;
    --amp_dtype|--amp-dtype)
      AMP_DTYPE="$2"
      shift 2
      ;;
    --no_amp|--no-amp)
      NO_AMP=true
      shift
      ;;
    --wandb)
      WANDB="$2"
      shift 2
      ;;
    --wandb_project|--wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb_run_name|--wandb-run-name)
      WANDB_RUN_NAME="$2"
      shift 2
      ;;
    --wandb_id|--wandb-id)
      WANDB_ID="$2"
      shift 2
      ;;

    # eval
    --model_path|--model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --schedulers)
      SCHEDULERS="$2"
      shift 2
      ;;
    --max_samples|--max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --tracing_enabled|--tracing-enabled)
      TRACING_ENABLED="$2"
      shift 2
      ;;
    --trace_hf_upload|--trace-hf-upload)
      TRACE_HF_UPLOAD="$2"
      shift 2
      ;;
    --trace_hf_dataset_repo|--trace-hf-dataset-repo)
      TRACE_HF_DATASET_REPO="$2"
      shift 2
      ;;
    --trace_hf_namespace|--trace-hf-namespace)
      TRACE_HF_NAMESPACE="$2"
      shift 2
      ;;
    --no_upload_traces|--no-upload-traces)
      NO_UPLOAD_TRACES=true
      shift
      ;;
    --no_replay_oracle|--no-replay-oracle)
      NO_REPLAY_ORACLE=true
      shift
      ;;

    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

# Disambiguate --corpus / --variant / --hf_dataset for eval vs train when mode=eval
# (re-parse not needed: eval uses EVAL_* only when we add separate flags below)
# For eval mode, --corpus sets eval corpus; for train, train corpus.
if [[ "$MODE" == "eval" ]]; then
  [[ -n "$CORPUS" && -z "$EVAL_CORPUS" ]] && EVAL_CORPUS="$CORPUS"
  [[ -n "$VARIANT" && -z "$EVAL_VARIANT" ]] && EVAL_VARIANT="$VARIANT"
  [[ -n "$HF_DATASET" && -z "$EVAL_HF_DATASET" ]] && EVAL_HF_DATASET="$HF_DATASET"
  [[ -n "$HF_NAMESPACE" && -z "$EVAL_HF_NAMESPACE" ]] && EVAL_HF_NAMESPACE="$HF_NAMESPACE"
  [[ -n "$BATCH_SIZE" && -z "$EVAL_BATCH_SIZE" ]] && EVAL_BATCH_SIZE="$BATCH_SIZE"
fi

append_if_set() {
  local key="$1"
  local val="$2"
  if [[ -n "$val" ]]; then
    OVERRIDES+=("${key}=${val}")
  fi
}

append_bool_if_set() {
  local key="$1"
  local val="$2"
  if [[ -n "$val" ]]; then
    OVERRIDES+=("${key}=${val}")
  fi
}

OVERRIDES=( "mode=${MODE}" )

if [[ "$MODE" == "train" ]]; then
  append_if_set train.variant "$VARIANT"
  append_if_set train.corpus "$CORPUS"
  append_if_set train.hf_dataset "$HF_DATASET"
  append_if_set train.hf_namespace "$HF_NAMESPACE"
  append_if_set train.train_tokens "$TRAIN_TOKENS"
  append_if_set train.batch_size "$BATCH_SIZE"
  append_if_set train.grad_accum "$GRAD_ACCUM"
  append_if_set train.num_workers "$NUM_WORKERS"
  append_if_set train.nproc_per_node "$NPROC"
  append_if_set train.peak_lr "$PEAK_LR"
  append_if_set train.mid_lr "$MID_LR"
  append_if_set train.min_lr "$MIN_LR"
  append_if_set train.warmup_steps "$WARMUP_STEPS"
  append_if_set train.warmup_fraction "$WARMUP_FRACTION"
  append_if_set train.warmup_max "$WARMUP_MAX"
  append_if_set train.epochs "$EPOCHS"
  append_if_set train.steps "$STEPS"
  append_if_set train.weight_decay "$WEIGHT_DECAY"
  append_if_set train.max_grad_norm "$MAX_GRAD_NORM"
  append_if_set train.seed "$SEED"
  append_if_set train.out_dir "$OUT_DIR"
  append_if_set train.resume "$RESUME"
  append_if_set train.log_every "$LOG_EVERY"
  append_if_set train.save_every "$SAVE_EVERY"
  append_if_set train.val_fraction "$VAL_FRACTION"
  append_if_set train.val_every "$VAL_EVERY"
  append_if_set train.hf_model "$HF_MODEL"
  append_if_set train.hf_model_namespace "$HF_MODEL_NAMESPACE"
  append_if_set train.amp_dtype "$AMP_DTYPE"
  append_bool_if_set train.wandb "$WANDB"
  append_if_set train.wandb_run_name "$WANDB_RUN_NAME"
  append_if_set train.wandb_id "$WANDB_ID"
  [[ -n "$DEVICE" ]] && append_if_set train.device "$DEVICE"
  [[ "$PUSH_MODEL" == "true" ]] && OVERRIDES+=(train.push_model=true)
  [[ "$HF_MODEL_PRIVATE" == "true" ]] && OVERRIDES+=(train.hf_model_private=true)
  [[ "$NO_AMP" == "true" ]] && OVERRIDES+=(train.amp=false)
else
  append_if_set eval.model_path "$MODEL_PATH"
  append_if_set eval.hf_model "$HF_MODEL"
  append_if_set eval.variant "$EVAL_VARIANT"
  append_if_set eval.corpus "$EVAL_CORPUS"
  append_if_set eval.hf_dataset "$EVAL_HF_DATASET"
  append_if_set eval.hf_namespace "$EVAL_HF_NAMESPACE"
  append_if_set eval.schedulers "$SCHEDULERS"
  append_if_set eval.max_samples "$MAX_SAMPLES"
  append_if_set eval.steps "$EVAL_STEPS"
  append_if_set eval.batch_size "$EVAL_BATCH_SIZE"
  append_if_set eval.out_dir "$EVAL_OUT_DIR"
  append_bool_if_set eval.tracing_enabled "$TRACING_ENABLED"
  append_bool_if_set eval.trace_hf_upload "$TRACE_HF_UPLOAD"
  append_if_set eval.trace_hf_dataset_repo "$TRACE_HF_DATASET_REPO"
  append_if_set eval.trace_hf_namespace "$TRACE_HF_NAMESPACE"
  [[ -n "$DEVICE" ]] && append_if_set eval.device "$DEVICE"
  [[ "$NO_UPLOAD_TRACES" == "true" ]] && OVERRIDES+=(eval.upload_traces=false)
  [[ "$NO_REPLAY_ORACLE" == "true" ]] && OVERRIDES+=(eval.replay_oracle=false)
fi

append_if_set wandb_project "$WANDB_PROJECT"
[[ "$OFFLINE" == "true" ]] && OVERRIDES+=(offline=true)
[[ -n "$LAUNCHER" ]] && OVERRIDES+=(launcher="$LAUNCHER")

eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
conda activate strace
export PATH="$CONDA_PREFIX/bin:$PATH"

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export TOKENIZERS_PARALLELISM=false

if [[ "$SLURM_MODE" == "true" ]]; then
  export LLADALENS_SLURM_WRAPPED=1
fi

if [[ "${LLADALENS_SLURM_WRAPPED:-}" == "1" ]] || [[ "${SLURM_JOB_ID:-}" != "" ]]; then
  RUN=(srun)
else
  RUN=()
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  if [[ -z "$DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  fi
fi

echo "Configuration (unset flags use config/lladalens.yaml):"
echo "  mode=${MODE}"
echo "  config=config/lladalens.yaml"
if [[ ${#OVERRIDES[@]} -gt 1 || ${#EXTRA[@]} -gt 0 ]]; then
  echo "  overrides: ${OVERRIDES[*]} ${EXTRA[*]}"
else
  echo "  overrides: mode=${MODE} only (all other values from config)"
fi
echo "  slurm=${SLURM_MODE} offline=${OFFLINE} device=${DEVICE:-config default}"
echo ""

"${RUN[@]}" python run_lladalens.py "${OVERRIDES[@]}" "${EXTRA[@]}"
