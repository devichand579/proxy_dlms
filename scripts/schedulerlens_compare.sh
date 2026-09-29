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
# SchedulerLens launcher on SLURM (or interactive GPU).
#
# Compare (default):
#   ./scripts/schedulerlens_compare.sh \
#     --repo llada-pcfg-traces --pattern 'llada-300m-pcfg-1B-pcfg_*.npz' \
#     --pcfg-only --replay-oracle --pcfg -o outputs/pcfg_compare_1b
#
# Prefix sweep, leaving ARC and FHS out:
#   ./scripts/schedulerlens_compare.sh --command prefix-sweep \
#     --exclude-task arc_challenge_chat --exclude-scheduler fhs
#
# Prefix selection:
#   ./scripts/schedulerlens_compare.sh --command prefix-selection --tau 1.0
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
SchedulerLens launcher.

  ./scripts/schedulerlens_compare.sh [OPTIONS]

Command (default: compare):
  --command NAME              compare | prefix-sweep | prefix-sweep-pcfg | prefix-selection

Keep aside:
  --exclude-task NAMES        Comma-separated tasks to skip (e.g. arc_challenge_chat)
  --exclude-scheduler NAMES   Comma-separated schedulers to skip (e.g. fhs,fdm)

Compare source (one required for compare):
  --repo REPO                 Hub dataset short name or org/repo (e.g. llada-pcfg-traces)
  --local-dir DIR             Local directory of .npz traces

Compare output:
  -o / --output-dir DIR       Output directory
                              (compare default: outputs/schedulerlens/compare)
  --csv NAME                  Output CSV filename, or an absolute path

Compare filters:
  --pattern GLOB              Filename glob (default: *.npz)
  --model NAME                Parsed-stem model filter
  --scheduler NAME            Parsed-stem scheduler filter (pd, bd, fdm, …)
  --steps N                   Parsed-stem step budget filter
  --pcfg-only                 Keep only PCFG-looking stems
  --limit N                   Max files to process
  --namespace ORG             HF org for short --repo and per-stem oracle models

Oracle / PCFG (compare):
  --replay-oracle             Exact Mt/Jt + proxy correlations (PCFG)
  --oracle PATH               Reuse saved oracle.npz (file or folder); match by stem
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

Prefix caps (compare):
  --prefix-steps Ks           Extra CSV rows with cumulative M/J/G over step < K
                              (e.g. 8,16,32,64)
  --metrics-only              Fast path: skip completion + cheap PPL re-streams

Prefix selection:
  --runs PATH                 prefix_sweep.csv from prefix-sweep
  --tau GAMMA                 Entropy budget column to rank (default 1.0)

Run:
  --slurm                     wrap in srun (use when submitted via sbatch)
  --resume                    skip filenames already in the compare CSV (default)
  --no-resume                 rewrite the compare CSV from scratch
  --no-purge-cache            keep HF cache between files
  --device DEV                0|1|cuda|cuda:0|cpu (local / non-slurm)
  --quiet

Examples:
  sbatch scripts/schedulerlens_compare.sh --repo llada-pcfg-traces \
    --pattern 'llada-300m-pcfg-1B-pcfg_*.npz' --pcfg-only \
    --replay-oracle --pcfg -o outputs/pcfg_compare_1b --slurm

  ./scripts/schedulerlens_compare.sh --command prefix-sweep \
    --exclude-task arc_challenge_chat --exclude-scheduler fhs

  ./scripts/schedulerlens_compare.sh --command prefix-selection --tau 1.0
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
ORACLE=""
FORCE_PCFG=false
MODEL_PATH=""
HF_MODEL=""
MAX_SAMPLES=""
LIMIT=""
RESUME=false
NO_RESUME=false
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
PREFIX_STEPS=""
METRICS_ONLY=false
COMMAND="compare"
EXCLUDE_TASK=""
EXCLUDE_SCHEDULER=""
RUNS_CSV=""
TAU=""
CSV_NAME=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage
      exit 0
      ;;
    --command)
      COMMAND="$2"
      shift 2
      ;;
    --exclude-task)
      EXCLUDE_TASK="$2"
      shift 2
      ;;
    --exclude-scheduler)
      EXCLUDE_SCHEDULER="$2"
      shift 2
      ;;
    --runs)
      RUNS_CSV="$2"
      shift 2
      ;;
    --tau)
      TAU="$2"
      shift 2
      ;;
    --csv)
      CSV_NAME="$2"
      shift 2
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
    --prefix-steps)
      PREFIX_STEPS="$2"
      shift 2
      ;;
    --metrics-only)
      METRICS_ONLY=true
      shift
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
    --oracle)
      ORACLE="$2"
      shift 2
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
      NO_RESUME=false
      shift
      ;;
    --no-resume)
      NO_RESUME=true
      RESUME=false
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

if [[ "$COMMAND" == "compare" && -z "$REPO_ID" && -z "$LOCAL_DIR" ]]; then
  echo "Error: provide --repo or --local-dir"
  usage
  exit 1
fi
if [[ -n "$REPO_ID" && -n "$LOCAL_DIR" ]]; then
  echo "Error: use only one of --repo or --local-dir"
  exit 1
fi
case "$COMMAND" in
  compare|prefix-sweep|prefix-sweep-pcfg|prefix-selection)
    ;;
  *)
    echo "Error: unknown --command '$COMMAND'"
    usage
    exit 1
    ;;
esac

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
if [[ -n "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi

CMD=(python -m schedulerlens.run "$COMMAND")
[[ -n "$EXCLUDE_TASK" ]] && CMD+=(--exclude-task "$EXCLUDE_TASK")
[[ -n "$EXCLUDE_SCHEDULER" ]] && CMD+=(--exclude-scheduler "$EXCLUDE_SCHEDULER")
[[ "$QUIET" == "true" ]] && CMD+=(--quiet)

if [[ "$COMMAND" == "compare" ]]; then
  CMD+=(-o "$OUTPUT_DIR" --pattern "$PATTERN")
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
  [[ -n "$ORACLE" ]] && CMD+=(--oracle "$ORACLE")
  [[ "$FORCE_PCFG" == "true" ]] && CMD+=(--pcfg)
  [[ -n "$MODEL_PATH" ]] && CMD+=(--model-path "$MODEL_PATH")
  [[ -n "$HF_MODEL" ]] && CMD+=(--hf-model "$HF_MODEL")
  [[ -n "$MAX_SAMPLES" ]] && CMD+=(--max-samples "$MAX_SAMPLES")
  [[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")
  if [[ "$NO_RESUME" == "true" ]]; then
    CMD+=(--no-resume)
  elif [[ "$RESUME" == "true" ]]; then
    CMD+=(--resume)
  fi
  [[ "$NO_PURGE_CACHE" == "true" ]] && CMD+=(--no-purge-cache)
  [[ -n "$ORACLE_WORKERS" ]] && CMD+=(--oracle-workers "$ORACLE_WORKERS")
  [[ -n "$ORACLE_STEP_WORKERS" ]] && CMD+=(--oracle-step-workers "$ORACLE_STEP_WORKERS")
  [[ -n "$ORACLE_FORWARD_BATCH" ]] && CMD+=(--oracle-forward-batch "$ORACLE_FORWARD_BATCH")
  [[ -n "$ORACLE_JT_EXACT_MAX" ]] && CMD+=(--oracle-jt-exact-max "$ORACLE_JT_EXACT_MAX")
  [[ -n "$ORACLE_JT_MARKOV_ORDER" ]] && CMD+=(--oracle-jt-markov-order "$ORACLE_JT_MARKOV_ORDER")
  [[ -n "$ORACLE_JT_LARGE_SET" ]] && CMD+=(--oracle-jt-large-set "$ORACLE_JT_LARGE_SET")
  [[ -n "$PREFIX_STEPS" ]] && CMD+=(--prefix-steps "$PREFIX_STEPS")
  [[ "$METRICS_ONLY" == "true" ]] && CMD+=(--metrics-only)
  [[ -n "$CSV_NAME" ]] && CMD+=(--csv "$CSV_NAME")
elif [[ "$COMMAND" == "prefix-selection" ]]; then
  [[ -n "$OUTPUT_DIR" ]] && CMD+=(-o "$OUTPUT_DIR")
  [[ -n "$CSV_NAME" ]] && CMD+=(--csv "$CSV_NAME")
  [[ -n "$RUNS_CSV" ]] && CMD+=(--runs "$RUNS_CSV")
  [[ -n "$TAU" ]] && CMD+=(--tau "$TAU")
elif [[ "$COMMAND" == "prefix-sweep" || "$COMMAND" == "prefix-sweep-pcfg" ]]; then
  [[ -n "$OUTPUT_DIR" ]] && CMD+=(-o "$OUTPUT_DIR")
  [[ -n "$CSV_NAME" ]] && CMD+=(--csv "$CSV_NAME")
fi

echo "SchedulerLens $COMMAND"
echo "  cwd: $REPO_ROOT"
echo "  exclude-task: ${EXCLUDE_TASK:-none}"
echo "  exclude-scheduler: ${EXCLUDE_SCHEDULER:-none}"
if [[ "$COMMAND" == "compare" ]]; then
  echo "  output: ${OUTPUT_DIR:-outputs/schedulerlens/compare}"
  echo "  source: ${REPO_ID:-$LOCAL_DIR}"
  echo "  pattern: $PATTERN"
  echo "  pcfg-only: $PCFG_ONLY  replay-oracle: $REPLAY_ORACLE  oracle: ${ORACLE:-}  pcfg: $FORCE_PCFG"
fi
echo "  slurm/srun: $SLURM_MODE  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  cmd: ${CMD[*]}"
echo ""

if [[ "$SLURM_MODE" == "true" ]]; then
  srun "${CMD[@]}"
else
  "${CMD[@]}"
fi
