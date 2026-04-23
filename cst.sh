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

# CLI: ./cst.sh --help. Hydra: config/cst.yaml. Docs: docs/cst.md, docs/parallel-runs.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
CST launcher: runs `python launch_cst.py` with Hydra (config: config/cst.yaml).

  ./cst.sh [OPTIONS] [HYDRA_OVERRIDES ...]

Options (shell only):
  --mode train|eval     train (default): collect hidden chunks + fit CST. eval: metrics on saved chunks.
  --hidden_dir DIR      eval: folder of hidden .pt files  →  eval.hidden_dir=DIR
  --checkpoint PATH     eval: CST checkpoint .pt          →  eval.checkpoint=PATH
  --slurm               set SLURM_WRAPPED=1 (use with srun when already under Slurm)
  -h, --help            show this message

Anything else is passed through to Hydra (see config/cst.yaml), for example:
  train.target_staging_gb=2.0
  cst.w=8
  model_path=/path/to/llada
  train.resume_checkpoint=outputs/cst_ckpt/cst_llada_scheduler_pd_k4096_w4_train_latest.pt

Examples:
  ./cst.sh
  ./cst.sh --mode eval --hidden_dir outputs/eval_hidden --checkpoint outputs/cst_ckpt/cst_llada_scheduler_pd_k4096_w4_train_best.pt
  ./cst.sh train.target_staging_gb=0.5 cst.k=2048
EOF
}

MODE="train"
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
    --hidden_dir)
      EXTRA+=(eval.hidden_dir="$2")
      shift 2
      ;;
    --checkpoint)
      EXTRA+=(eval.checkpoint="$2")
      shift 2
      ;;
    --slurm)
      export SLURM_WRAPPED=1
      shift
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

if [[ "${SLURM_WRAPPED:-}" == "1" ]] || [[ "${SLURM_JOB_ID:-}" != "" ]]; then
  RUN=(srun)
else
  RUN=()
fi

if [[ "$MODE" == "eval" ]]; then
  "${RUN[@]}" python launch_cst.py mode=eval "${EXTRA[@]}"
else
  "${RUN[@]}" python launch_cst.py mode=train "${EXTRA[@]}"
fi
