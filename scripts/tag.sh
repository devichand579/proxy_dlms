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

########################################################################################
# TAG launcher — runs `python hydra_runners/run_tag.py` with Hydra (config: config/tag.yaml)
#
# Slurm: sbatch scripts/tag.sh --slurm --cst_checkpoint ... --model_path ... --prompt "..."
# Local: ./scripts/tag.sh --cst_checkpoint ... --prompt "..."
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
TAG launcher — runs `python hydra_runners/run_tag.py` with Hydra (config: config/tag.yaml).

  ./scripts/tag.sh [OPTIONS] [HYDRA_OVERRIDES ...]

Options (shell only):
  --cst_checkpoint PATH   required unless set in config/tag.yaml
  --model_path PATH       LM directory (full run or chunk logits)
  --out_dir DIR           default outputs/tag
  --prompt STR            full run: generate hiddens then TAG
  --hidden_pt PATH        chunk-only: existing .pt (skips generation)
  --device STR            e.g. cuda, cuda:0
  --no_html               do not write tag_view.html
  --no_logits             chunk-only: skip model load for output sinks
  --with_residual         add residual err nodes/edges
  --output_exclude_eos    mask EOS in logits before softmax (lexical readout)
  --no_eos_harness        full run: disable EOS early stop (reference-style)
  --slurm                 use srun when already inside a Slurm allocation
  -h, --help              show this message

Any other args are passed to Hydra, e.g.:
  gen_length=128 steps=128 influence_node_mass=0.9
  cst_config=/path/to/cst.yaml

Examples:
  ./scripts/tag.sh --cst_checkpoint outputs/cst_ckpt/cst_llada_scheduler_pd_k4096_w4_train_best.pt \\
           --model_path ./models/llada --out_dir outputs/tag_run \\
           --prompt "Solve: 2+2="

  ./scripts/tag.sh --cst_checkpoint outputs/cst_ckpt/..._train_best.pt --model_path ./models/llada \\
           --hidden_pt outputs/.../hidden/file.pt --out_dir outputs/tag_from_pt

Slurm:
  sbatch scripts/tag.sh --slurm --cst_checkpoint ... --model_path ./models/llada --prompt "..."
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
    --no_logits)
      EXTRA+=(no_logits=true)
      shift
      ;;
    --with_residual)
      EXTRA+=(with_residual=true)
      shift
      ;;
    --output_exclude_eos)
      EXTRA+=(output_exclude_eos=true)
      shift
      ;;
    --no_eos_harness)
      EXTRA+=(use_eos_harness=false)
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
"${RUN[@]}" python hydra_runners/run_tag.py "${EXTRA[@]}"
