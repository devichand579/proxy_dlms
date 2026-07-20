#!/bin/bash
# #SBATCH --job-name=sl_compare
# #SBATCH --output=job_outputs/sl_compare_out_%j.out
# #SBATCH --error=job_outputs/sl_compare_err_%j.err
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64gb
# #SBATCH --time=47:00:00
# #SBATCH --partition=gpu_h100  # gpu_l40, gpu_h100, dibdp (AIRAWAT)
# #SBATCH --gres=gpu:1

########################################################################################
# DGX setup: uncomment the block below and comment out (partition through gres) above
########################################################################################
#SBATCH --partition=dgx_all
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:2
##SBATCH --mem=72GB
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --wckey=submitit
#SBATCH --output=job_outputs/%j/%j_0_log.out
#SBATCH --error=job_outputs/%j/%j_0_log.err

########################################################################################
# SchedulerLens Hub/local compare on SLURM (or interactive GPU).
#
# Interactive:
#   ./scripts/schedulerlens_compare.sh \
#     --repo llada-pcfg-traces --pattern 'llada-300m-pcfg-1B-pcfg_*.npz' \
#     --pcfg-only --replay-oracle --pcfg -o outputs/pcfg_compare_1b
#
# Slurm:
#   sbatch scripts/schedulerlens_compare.sh --repo llada-pcfg-traces \
#     --pattern 'llada-300m-pcfg-1B-pcfg_pd_s128.npz' --pcfg-only \
#     --replay-oracle --pcfg -o outputs/pcfg_compare --slurm
#
# One scheduler on the 1B model:
#   sbatch scripts/schedulerlens_compare.sh --repo llada-pcfg-traces \
#     --pattern 'llada-300m-pcfg-1B-pcfg_top_k_m_s128.npz' \
#     --pcfg-only --replay-oracle --pcfg -o outputs/pcfg_compare --slurm
#
# Requires: conda env ``strace``, HF auth for Hub downloads (``hf auth login`` or HF_TOKEN).
########################################################################################

set -euo pipefail

# shellcheck source=_repo_root.sh
_script_dir="$(dirname "${BASH_SOURCE[0]}")"
# When sbatch copies the script to its spool dir, _repo_root.sh won't be there;
# fall back to the original submission directory.
if [[ ! -f "$_script_dir/_repo_root.sh" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _script_dir="${SLURM_SUBMIT_DIR}/scripts"
fi
source "$_script_dir/_repo_root.sh"
unset _script_dir

usage() {
  cat <<'EOF'
SchedulerLens compare launcher (Hub or local NPZ → CSV + plots).

  ./scripts/schedulerlens_compare.sh [OPTIONS]

Source (one required):
  --repo REPO                 Hub dataset short name or org/repo (e.g. llada-pcfg-traces)
  --local-dir DIR             Local directory of .npz traces

Output:
  -o / --output-dir DIR       Compare output directory (required)

Filters:
  --pattern GLOB              Filename glob (default: *.npz)
  --model NAME                Parsed-stem model filter
  --scheduler NAME            Parsed-stem scheduler filter (pd, bd, fdm, …)
  --steps N                   Parsed-stem step budget filter
  --pcfg-only                 Keep only PCFG-looking stems
  --limit N                   Max files to process
  --namespace ORG             HF org for short --repo and per-stem oracle models

Oracle / PCFG:
  --replay-oracle             Exact Mt/Jt + proxy correlations (PCFG)
  --pcfg                      Force PCFG oracle gating
  --model-path PATH|REPO      Local checkpoint or HF id for all traces
  --hf-model REPO             Fixed HF model for all traces
  --max-samples N             Cap samples per trace

Oracle knobs (optional):
  --oracle-workers N
  --oracle-step-workers N
  --oracle-forward-batch B
  --oracle-jt-exact-max K
  --oracle-jt-markov-order M
  --oracle-jt-large-set N
  --oracle-no-jt

Run:
  --slurm                     wrap in srun (use when submitted via sbatch)
  --resume                    resume compare CSV if present
  --no-purge-cache            keep HF cache between files
  --device DEV                0|1|cuda|cuda:0|cpu (local / non-slurm)
  --quiet

Examples:
  sbatch scripts/schedulerlens_compare.sh --repo llada-pcfg-traces \
    --pattern 'llada-300m-pcfg-1B-pcfg_*.npz' --pcfg-only \
    --replay-oracle --pcfg -o outputs/pcfg_compare_1b --slurm

  ./scripts/schedulerlens_compare.sh --local-dir ./npz_traces --pcfg-only \
    -o outputs/local_compare --device 0
EOF
}

REPO_ID=""
LOCAL_DIR=""
OUTPUT_DIR=""
NAMESPACE=""
REVISION=""
PATTERN="*.npz"
MODEL_FILTER=""
SCHEDULER_FILTER=""
STEPS_FILTER=""
PCFG_ONLY=false
REPLAY_ORACLE=false
FORCE_PCFG=false
MODEL_PATH=""
HF_MODEL=""
MAX_SAMPLES=""
LIMIT=""
RESUME=false
NO_PURGE_CACHE=false
QUIET=false
SLURM_MODE=false
DEVICE=""
ORACLE_WORKERS=""
ORACLE_STEP_WORKERS=""
ORACLE_FORWARD_BATCH=""
ORACLE_JT_EXACT_MAX=""
ORACLE_JT_MARKOV_ORDER=""
ORACLE_JT_LARGE_SET=""
ORACLE_NO_JT=false
STRACE_METRICS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage
      exit 0
      ;;
    --repo|--repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --local-dir)
      LOCAL_DIR="$2"
      shift 2
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --revision)
      REVISION="$2"
      shift 2
      ;;
    --pattern)
      PATTERN="$2"
      shift 2
      ;;
    --model)
      MODEL_FILTER="$2"
      shift 2
      ;;
    --scheduler)
      SCHEDULER_FILTER="$2"
      shift 2
      ;;
    --steps)
      STEPS_FILTER="$2"
      shift 2
      ;;
    --pcfg-only)
      PCFG_ONLY=true
      shift
      ;;
    --replay-oracle)
      REPLAY_ORACLE=true
      shift
      ;;
    --pcfg)
      FORCE_PCFG=true
      shift
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --hf-model)
      HF_MODEL="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --no-purge-cache)
      NO_PURGE_CACHE=true
      shift
      ;;
    --slurm)
      SLURM_MODE=true
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --oracle-workers)
      ORACLE_WORKERS="$2"
      shift 2
      ;;
    --oracle-step-workers)
      ORACLE_STEP_WORKERS="$2"
      shift 2
      ;;
    --oracle-forward-batch)
      ORACLE_FORWARD_BATCH="$2"
      shift 2
      ;;
    --oracle-jt-exact-max)
      ORACLE_JT_EXACT_MAX="$2"
      shift 2
      ;;
    --oracle-jt-markov-order)
      ORACLE_JT_MARKOV_ORDER="$2"
      shift 2
      ;;
    --oracle-jt-large-set)
      ORACLE_JT_LARGE_SET="$2"
      shift 2
      ;;
    --oracle-no-jt)
      ORACLE_NO_JT=true
      shift
      ;;
    --strace-metrics)
      STRACE_METRICS="$2"
      shift 2
      ;;
    -q|--quiet)
      QUIET=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Error: -o / --output-dir is required"
  usage
  exit 1
fi
if [[ -z "$REPO_ID" && -z "$LOCAL_DIR" ]]; then
  echo "Error: provide --repo or --local-dir"
  usage
  exit 1
fi
if [[ -n "$REPO_ID" && -n "$LOCAL_DIR" ]]; then
  echo "Error: use only one of --repo or --local-dir"
  exit 1
fi

# Initialize conda
if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
  eval "$("$HOME/anaconda3/bin/conda" shell.bash hook)"
else
  eval "$(conda shell.bash hook)"
fi
conda activate strace
export PATH="$CONDA_PREFIX/bin:$PATH"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

# Local GPU pin when not using --slurm
if [[ "$SLURM_MODE" != "true" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  if [[ -n "$DEVICE" ]]; then
    case "$DEVICE" in
      cpu|CPU)
        export CUDA_VISIBLE_DEVICES=""
        ;;
      cuda)
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        ;;
      cuda:*)
        export CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}"
        ;;
      [0-9]*)
        export CUDA_VISIBLE_DEVICES="$DEVICE"
        ;;
      *)
        echo "Warning: unrecognized --device '$DEVICE' (leaving CUDA_VISIBLE_DEVICES as-is)"
        ;;
    esac
  else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  fi
fi

mkdir -p job_outputs
mkdir -p "$OUTPUT_DIR"

CMD=(python -m schedulerlens.run compare -o "$OUTPUT_DIR" --pattern "$PATTERN")

if [[ -n "$REPO_ID" ]]; then
  CMD+=(--repo "$REPO_ID")
else
  if [[ -d "$LOCAL_DIR" ]]; then
    :
  elif [[ -d "$REPO_ROOT/$LOCAL_DIR" ]]; then
    LOCAL_DIR="$REPO_ROOT/$LOCAL_DIR"
  else
    echo "Error: --local-dir not found: $LOCAL_DIR"
    exit 1
  fi
  CMD+=(--local-dir "$LOCAL_DIR")
fi

[[ -n "$NAMESPACE" ]] && CMD+=(--namespace "$NAMESPACE")
[[ -n "$REVISION" ]] && CMD+=(--revision "$REVISION")
[[ -n "$MODEL_FILTER" ]] && CMD+=(--model "$MODEL_FILTER")
[[ -n "$SCHEDULER_FILTER" ]] && CMD+=(--scheduler "$SCHEDULER_FILTER")
[[ -n "$STEPS_FILTER" ]] && CMD+=(--steps "$STEPS_FILTER")
[[ "$PCFG_ONLY" == "true" ]] && CMD+=(--pcfg-only)
[[ "$REPLAY_ORACLE" == "true" ]] && CMD+=(--replay-oracle)
[[ "$FORCE_PCFG" == "true" ]] && CMD+=(--pcfg)
[[ -n "$MODEL_PATH" ]] && CMD+=(--model-path "$MODEL_PATH")
[[ -n "$HF_MODEL" ]] && CMD+=(--hf-model "$HF_MODEL")
[[ -n "$MAX_SAMPLES" ]] && CMD+=(--max-samples "$MAX_SAMPLES")
[[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")
[[ "$RESUME" == "true" ]] && CMD+=(--resume)
[[ "$NO_PURGE_CACHE" == "true" ]] && CMD+=(--no-purge-cache)
[[ "$QUIET" == "true" ]] && CMD+=(--quiet)
[[ -n "$ORACLE_WORKERS" ]] && CMD+=(--oracle-workers "$ORACLE_WORKERS")
[[ -n "$ORACLE_STEP_WORKERS" ]] && CMD+=(--oracle-step-workers "$ORACLE_STEP_WORKERS")
[[ -n "$ORACLE_FORWARD_BATCH" ]] && CMD+=(--oracle-forward-batch "$ORACLE_FORWARD_BATCH")
[[ -n "$ORACLE_JT_EXACT_MAX" ]] && CMD+=(--oracle-jt-exact-max "$ORACLE_JT_EXACT_MAX")
[[ -n "$ORACLE_JT_MARKOV_ORDER" ]] && CMD+=(--oracle-jt-markov-order "$ORACLE_JT_MARKOV_ORDER")
[[ -n "$ORACLE_JT_LARGE_SET" ]] && CMD+=(--oracle-jt-large-set "$ORACLE_JT_LARGE_SET")
[[ "$ORACLE_NO_JT" == "true" ]] && CMD+=(--oracle-no-jt)
[[ -n "$STRACE_METRICS" ]] && CMD+=(--strace-metrics "$STRACE_METRICS")

echo "SchedulerLens compare"
echo "  cwd: $REPO_ROOT"
echo "  output: $OUTPUT_DIR"
echo "  source: ${REPO_ID:-$LOCAL_DIR}"
echo "  pattern: $PATTERN"
echo "  pcfg-only: $PCFG_ONLY  replay-oracle: $REPLAY_ORACLE  pcfg: $FORCE_PCFG"
echo "  slurm/srun: $SLURM_MODE  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  cmd: ${CMD[*]}"
echo ""

if [[ "$SLURM_MODE" == "true" ]]; then
  srun "${CMD[@]}"
else
  "${CMD[@]}"
fi
