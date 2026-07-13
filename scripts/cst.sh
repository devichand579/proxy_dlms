#!/bin/bash
#SBATCH --job-name=cst
#SBATCH --output=job_outputs/cst_job_out_%j.out
#SBATCH --error=job_outputs/cst_job_err_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=42gb
#SBATCH --time=47:00:00
#SBATCH --partition=gpupart_a40  # gpupart_v100, gpupart_a40, gpu_h100, dibdp (AIRAWAT)
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
# Single CST job via sbatch, e.g.:
#   sbatch scripts/cst.sh --mode store store.scope=train store.samples=512
#   sbatch scripts/cst.sh --mode train train.num_epochs=2
#   sbatch scripts/cst.sh --mode eval --checkpoint outputs/cst_ckpt/cst_..._train_best.pt \
#       --hidden_dir outputs/eval_hidden
#
# Direct (no sbatch): ./scripts/cst.sh --mode train [hydra overrides...]
# Config: config/cst.yaml  |  Entry point: hydra_runners/run_cst.py
########################################################################################

set -euo pipefail

# shellcheck source=_repo_root.sh
source "$(dirname "${BASH_SOURCE[0]}")/_repo_root.sh"

usage() {
  cat <<'EOF'
CST launcher — runs `python hydra_runners/run_cst.py` with Hydra (config: config/cst.yaml).

  ./scripts/cst.sh [OPTIONS] [HYDRA_OVERRIDES ...]

Options:
  --mode store|train|eval
                        store  — collect/upload hiddens to Hub (no CST fit)
                        train  — download Hub slices -> train CST -> delete local
                        eval   — score CST checkpoint on saved hiddens
  --scope train|eval    store only: sets store.scope=train|eval
  --hidden_dir DIR      mode=eval only: local folder of .pt to score (not used for store)
  --checkpoint PATH     eval: CST checkpoint .pt
  --task_name NAME      store/eval: lm-eval task (e.g. gsm8k)
  --slurm               use srun when already inside a Slurm allocation
  -h, --help            show this message

Examples:
  ./scripts/cst.sh --mode store store.scope=train
  ./scripts/cst.sh --mode store --scope eval --task_name gsm8k
  ./scripts/cst.sh --mode train train.target_staging_gb=1.0
  ./scripts/cst.sh --mode eval --checkpoint outputs/cst_ckpt/cst_llada_scheduler_pd_k4096_w4_ls0.001_train_best.pt \
      --hidden_dir outputs/eval_hidden

Slurm:
  sbatch scripts/cst.sh --mode store store.scope=train store.samples=2048
  sbatch scripts/cst.sh --mode train train.resume_from_latest=true
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
    --scope)
      EXTRA+=(store.scope="$2")
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
    --task_name)
      EXTRA+=(store.task_name="$2" eval.task_name="$2")
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

"${RUN[@]}" python hydra_runners/run_cst.py "mode=${MODE}" "${EXTRA[@]}"
