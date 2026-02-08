# STrace (DONT FOLLOW NOT UPDATED)

## Usage

### Environment Setup
Run the following command to create a new environment and install the dependencies.
```bash
conda create -n strace -f environment.yaml
conda activate strace
pip install -r requirements.txt
```

### Downloading the Model
Please download the model configuration and safetensors files before running any scheduler or evaluation scripts.

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='GSAI-ML/LLaDA-8B-Base', local_dir='./models/llada', local_dir_use_symlinks=False)"
```

### Running Evaluation

We provide a convenient shell script `eval.sh` for running evaluations with proper configuration.

#### Basic Usage

```bash
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --compile_model False
```

#### Arguments

- `--batch_size`: Batch size for processing requests (e.g., 2, 4, 8)
- `--mode`: Generation mode
  - `pd` or `pure_diffusion`: Pure diffusion mode
  - `bd` or `block_diffusion`: Block diffusion mode
- `--task_name`: Name of the evaluation task
  - Supported tasks: `gsm8k`, `minerva_math500`, `humaneval`, `mbpp`, `logiqa`, `bbh`, `truthfulqa`
- `--model`: Model name (should match the directory name in `./models/`)
- `--compile_model`: Whether to compile model with `torch.compile` (`True` or `False`)

#### Automatic Configuration

The script automatically sets:

1. **Generation Length & Steps**:
   - `humaneval` and `mbpp`: 512 tokens, 512 steps
   - Other tasks: 256 tokens, 256 steps

2. **Block Length**:
   - `block_diffusion` mode: 32
   - `pure_diffusion` mode: Same as generation length

3. **Few-shot Examples** (`num_fewshot`):
   - `humaneval`: 0 shot
   - `gsm8k`: 4 shot
   - `minerva_math500`: 4 shot
   - `mbpp`: 4 shot
   - `bbh`: 3 shot
   - `truthfulqa`: 0 shot

4. **Wandb Name**: Automatically generated as `{model}_{task_name}_{mode}_{timestamp}`

5. **Output Path**: `outputs/{model}_{task_name}_{mode}_{timestamp}.json`

#### Example Commands

```bash
# Evaluate on GSM8K with pure diffusion
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --compile_model False

# Evaluate on HumanEval with block diffusion
./eval.sh --batch_size 1 --mode bd --task_name humaneval --model llada --compile_model False

# Evaluate on BBH with pure diffusion
./eval.sh --batch_size 4 --mode pd --task_name bbh --model llada --compile_model True
```

#### Logging

- **File Logs**: Saved to `logs/{model}_{task_name}_{mode}_{timestamp}.log`
- **Wandb Logging**: Automatically logs metrics including:
  - Per-chunk metrics (size, generation time, steps, blocks, throughput)
  - Global statistics (total samples, steps, time, blocks, averages)
  - Generation metrics (steps per generation, early stopping)

#### Output Files

- **Results**: `outputs/{model}_{task_name}_{mode}_{timestamp}.json`
- **Logs**: `logs/{model}_{task_name}_{mode}_{timestamp}.log`
- **Samples**: Included in output JSON when using `--log_samples`

### Hydra launcher (SLURM multirun)

For running many eval configs on a SLURM cluster, use the Hydra entry point and the submitit launcher.

**Install:** `pip install hydra-core hydra-submitit-launcher` (or use `requirements.txt`).

**Single run (local, same as run_eval.py):**
```bash
python launch_eval.py task_name=gsm8k mode=pd
```

**Multirun locally (multiple configs, sequential):**
```bash
python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun
```

**Multirun on SLURM (one SLURM job per config combination):**
```bash
python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun launcher=slurm
```

Config: `config/config.yaml`. Launcher defaults to local; use `launcher=slurm` for cluster. SLURM defaults (partition, time, mem, GPUs) are in `config/launcher/slurm.yaml` and match `sample_slurm.sh` (dgx2, 72h, 96G, 1 GPU). Override from CLI if needed, e.g. `hydra.launcher.partition=other`.

































