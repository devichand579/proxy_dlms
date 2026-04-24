#!/bin/bash
#SBATCH --job-name=tag_strace
#SBATCH --output=job_outputs/strace_tag_out_%j.out
#SBATCH --error=job_outputs/strace_tag_err_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=42gb
#SBATCH --time=12:00:00
#SBATCH --partition=gpupart_a40
#SBATCH --gres=gpu:1

########################################################################################
# DGX: uncomment the block below and comment out the partition/gres block above
########################################################################################
##SBATCH --partition=dgx2
##SBATCH --qos=gpu2
##SBATCH --gres=gpu:2
##SBATCH --mem=72GB
##SBATCH --time=12:00:00
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --open-mode=append
##SBATCH --output=job_outputs/%j/%j_0_log.out
##SBATCH --error=job_outputs/%j/%j_0_log.err

# CLI: ./tag.sh --help. Hydra: config/tag.yaml. Docs: docs/tag.md, docs/parallel-runs.md
#
# Slurm: sbatch tag.sh --slurm --cst_checkpoint ... --model_path ... --prompt "..."
# Local GPU: ./tag.sh ...   (or CUDA_VISIBLE_DEVICES=0 ./tag.sh ...)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
TAG launcher: runs `python launch_tag.py` with Hydra (config: config/tag.yaml).

  ./tag.sh [OPTIONS] [HYDRA_OVERRIDES ...]

Options (shell only):
  --cst_checkpoint PATH   required unless set in config/tag.yaml
  --model_path PATH      LM directory (for capture + logits; use with prompt or for chunk logits)
  --out_dir DIR          default outputs/tag
  --prompt STR           full run: generate hiddens then TAG (mutually exclusive with --hidden_pt)
  --hidden_pt PATH       chunk-only: existing save_hidden_records .pt (skips generation)
  --device STR           e.g. cuda, cuda:0
  --no_html              do not write tag_view.html
  --with_residual        add residual err nodes/edges
  --slurm                set SLURM_WRAPPED=1 (use with srun when already under Slurm)
  -h, --help             show this message

Any other args are passed to Hydra, e.g.:
  gen_length=128 steps=128 prune_mode=none
  cst_config=/path/to/cst.yaml

Examples:
  ./tag.sh --cst_checkpoint outputs/cst_ckpt/cst_llada_scheduler_pd_k4096_w4_train_best.pt \\
           --model_path ./models/llada --out_dir outputs/tag_run \\
           --prompt "Solve: 2+2="

  ./tag.sh --cst_checkpoint outputs/cst_ckpt/..._train_best.pt --model_path ./models/llada \\
           --hidden_pt outputs/.../hidden/file.pt --out_dir outputs/tag_from_pt
EOF
}

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --cst_checkpoint)
      EXTRA+=(cst_checkpoint="$2")
      shift 2
      ;;
    --model_path)
      EXTRA+=(model_path="$2")
      shift 2
      ;;
    --out_dir)
      EXTRA+=(out_dir="$2")
      shift 2
      ;;
    --prompt)
      EXTRA+=("prompt=$2")
      shift 2
      ;;
    --hidden_pt)
      EXTRA+=(hidden_pt="$2")
      shift 2
      ;;
    --device)
      EXTRA+=(device="$2")
      shift 2
      ;;
    --no_html)
      EXTRA+=(write_html=false)
      shift
      ;;
    --with_residual)
      EXTRA+=(with_residual=true)
      shift
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

mkdir -p job_outputs
"${RUN[@]}" python launch_tag.py "${EXTRA[@]}"
