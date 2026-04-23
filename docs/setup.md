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
