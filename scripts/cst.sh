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
CST launcher — runs `python hydra_runners/run_cst.py` with Hydra (config: config/cst.yaml).

  ./scripts/cst.sh [OPTIONS] [HYDRA_OVERRIDES ...]

Options:
  --mode store|train|eval
                        store  — collect/upload hiddens to Hub (no CST fit)
                        train  — download each Hub batch -> train CST -> delete local
                        eval   — score CST checkpoint on saved hiddens
  --scope train|eval    store only: sets store.scope=train|eval
  --hidden_dir DIR      mode=eval only: local folder of .pt to score (not used for store)
  --checkpoint PATH     eval: CST checkpoint (sharded dir, or legacy .pt)
  --task_name NAME      store/eval: lm-eval task (gsm8k, all, or gsm8k,humaneval)
  --anchor_block N      train: cap on anchors per backward (default: auto from VRAM)
  --gpu_pinned_gb N     train: cap on bank GB pinned on GPU (default: auto from RAM+VRAM)
  --save_optimizer_state
                        train: also shard Adam moments into checkpoints (3x size)
  --device DEV          local only: 0|1|cuda|cuda:0|cpu
  --slurm               optional: one srun wrap (omit with sbatch unless your site requires it)
  -h, --help            show this message

Examples:
  ./scripts/cst.sh --mode store store.scope=train
  ./scripts/cst.sh --mode store --scope eval --task_name gsm8k
  ./scripts/cst.sh --mode store --scope eval --task_name all
  ./scripts/cst.sh --mode train train.num_epochs=1
  ./scripts/cst.sh --mode train train.auto_memory=false train.anchor_block=16 train.gpu_pinned_gb=110
  ./scripts/cst.sh --mode eval --checkpoint outputs/cst_ckpt/cst_llada_pd_512_256_k4096_w4_ls0.001_best \
      --hidden_dir outputs/eval_hidden

Slurm:
  sbatch scripts/cst.sh --mode store store.scope=train store.samples=2048
  sbatch scripts/cst.sh --mode train train.resume_from_latest=true
EOF
}

MODE="train"
DEVICE=""
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
    --anchor_block)
      EXTRA+=(train.anchor_block="$2")
      shift 2
      ;;
    --gpu_pinned_gb)
      EXTRA+=(train.gpu_pinned_gb="$2")
      shift 2
      ;;
    --save_optimizer_state)
      EXTRA+=(train.save_optimizer_state=true)
      shift
      ;;
    --device)
      DEVICE="$2"
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

if [[ "${SLURM_WRAPPED:-}" == "1" ]]; then
  RUN=(srun)
else
  RUN=()
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
# Streaming the bank between host and device allocates and frees large blocks
# every chunk; expandable segments keep the caching allocator from fragmenting.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Local GPU pin when not using --slurm
if [[ "${SLURM_WRAPPED:-}" != "1" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  if [[ -n "$DEVICE" ]]; then
    case "$DEVICE" in
      cpu|CPU)   export CUDA_VISIBLE_DEVICES="" ;;
      cuda)      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" ;;
      cuda:*)    export CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" ;;
      [0-9]*)    export CUDA_VISIBLE_DEVICES="$DEVICE" ;;
      *)         echo "Warning: unrecognized --device '$DEVICE' (leaving CUDA_VISIBLE_DEVICES as-is)" ;;
    esac
  else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  fi
fi

mkdir -p job_outputs

echo "CST launcher"
echo "  cwd: $REPO_ROOT"
echo "  mode: $MODE"
echo "  slurm/srun: ${SLURM_WRAPPED:-0}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  overrides: ${EXTRA[*]:-none}"
echo ""

"${RUN[@]}" python hydra_runners/run_cst.py "mode=${MODE}" "${EXTRA[@]}"
