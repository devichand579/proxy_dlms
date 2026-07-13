# Environment and models

## Environment

```bash
conda create -n strace -f environment.yaml
conda activate strace
pip install -r requirements.txt
```

## Models

Download checkpoints before running evaluation or scheduler tests.

### LLaDA 8B Base

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='GSAI-ML/LLaDA-8B-Base', local_dir='./models/llada', local_dir_use_symlinks=False)"
```

### Dream 7B

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Dream-org/Dream-v0-Base-7B', local_dir='./models/dream', local_dir_use_symlinks=False)"
```

### LLaDA-1.5

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='GSAI-ML/LLaDA-1.5', local_dir='./models/llada1_5', local_dir_use_symlinks=False)"
```

Point `model_path` / `--model` at the matching directory under `./models/` (see [Evaluation](evaluation.md)).


## AMD ROCm Support

On AMD GPUs you need a **ROCm build of PyTorch**, not the default CUDA wheel from `pip install -r requirements.txt`.

1. Install ROCm drivers and a matching PyTorch ROCm build ([PyTorch ROCm install guide](https://pytorch.org/get-started/locally/)).
2. Verify the GPU is visible to PyTorch (ROCm still exposes the CUDA-compatible API):

```bash
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

3. Run with `device=cuda` as usual — there is no separate `rocm` device string in this repo:

```bash
python tests/test_tracing.py --device cuda
./scripts/eval.sh --device cuda ...
```

If you see `Found no NVIDIA driver on your system`, PyTorch is still the **CUDA** build or ROCm is not set up; reinstall PyTorch for ROCm or use CPU (`--device cpu`) for a slow sanity check.

**Notes:**

- LLaDA configs include ROCm-related workarounds (`amd_compatible` LayerNorm); Dream/LLaDA-1.5 on ROCm are less tested here.
- `torch.compile` (on by default in eval) can be unstable on ROCm; use `compile_model=false` in Hydra/`scripts/eval.sh` if you hit compile errors.
- Hub trace audit/convert pipelines are CPU-only and work without a GPU.
