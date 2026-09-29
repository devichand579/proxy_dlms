#!/bin/bash
#SBATCH --job-name=cst_store
#SBATCH --partition=dgx_all
#SBATCH --qos=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=2             # 2 GPUs
#SBATCH --gres=gpu:2           # Must match ntasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=350G
#SBATCH --time=3-00:00:00

# This wrapper uses srun to launch multiple independent instances of cst.sh.
# Slurm will automatically set SLURM_NTASKS and SLURM_PROCID for each child,
# and isolate CUDA_VISIBLE_DEVICES.

# NOTE: auto_batch_size MUST be false so all workers compute the exact same batches.
# Once this job finishes, you must run a single-GPU instance to seal the manifest:
# ./scripts/cst.sh --mode store store.auto_batch_size=false store.inference_batch_size=64 "$@"

set -euo pipefail

# Proxy settings for internet access on compute nodes
export http_proxy="http://login3:8888"
export https_proxy="http://login3:8888"
export HTTP_PROXY="http://login3:8888"
export HTTPS_PROXY="http://login3:8888"
export no_proxy="localhost,127.0.0.1,login1,login2,login3,master,slurm,dgx1,dgx2,dgx3,dgx4,dgx5,10.0.0.0/8,172.10.0.0/16,192.168.0.0/16"
export NO_PROXY="$no_proxy"

echo "Checking internet connectivity from $(hostname)..."
curl -I -s --connect-timeout 5 https://huggingface.co > /dev/null && echo "Internet OK!" || { echo "Proxy unreachable!"; exit 1; }

srun --label --gpus-per-task=1 ./scripts/cst.sh --mode store store.auto_batch_size=false store.inference_batch_size=64 "$@"
