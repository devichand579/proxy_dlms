# STrace

Diffusion-based text generation with LLaDA-style schedulers. Supports evaluation on lm-eval tasks, batched generation, tracing, and optional interventions.

---

## Environment Setup

```bash
conda create -n strace -f environment.yaml
conda activate strace
pip install -r requirements.txt
```

---

## Model

Download the LLaDA model before running evaluation or scheduler tests:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='GSAI-ML/LLaDA-8B-Base', local_dir='./models/llada', local_dir_use_symlinks=False)"
```

---

## Running Evaluation

### Using `eval.sh`

Convenient wrapper that sets generation length, steps, few-shot count, and output paths from the task name.

**Basic:**
```bash
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --compile_model False
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--batch_size` | Max requests per chunk (e.g. 2, 4, 8). Chunks may be smaller when a length bucket has fewer samples. |
| `--mode` | Scheduler: `pd`, `bd`, `eb`, `fdm`, `fhs`, `top_k_c`, `top_k_m`, `dus` (see below). |
| `--task_name` | Eval task: `gsm8k`, `humaneval`, `mbpp`, `minerva_math500`, `bbh`, `truthfulqa`, etc. |
| `--model` | Model dir under `./models/` (e.g. `llada`). |
| `--compile_model` | `True` / `False` for `torch.compile`. |
| `--wandb` | `True` to log to Weights & Biases. |
| `--tracing_enabled` | `True` to write per-task traces to `trace_outputs/`. |
| `--intervention_mode` | Optional: `drop` or `swap`. |
| `--intervention_steps` | Optional: comma-separated step indices (e.g. `0,128,256`). |
| `--intervention_k` | Optional: tokens per intervention step (default 1). |
| `--slurm` | Use when submitting via `sbatch` so the command runs under `srun`. |

**Mode shortcuts:**

- `pd` — pure diffusion  
- `bd` — block diffusion  
- `eb` — EB sampler  
- `fdm` — FDM  
- `fhs` — FHS  
- `top_k_c` — top-k (confidence)  
- `top_k_m` — top-k (top margin)  
- `dus` — dilated unmasking scheduler  

**Auto-configured from task:**

- **Length & steps:** HumanEval/MBPP use 512 tokens and 512 steps; others use 256.
- **Block length:** Block diffusion uses 32; pure diffusion uses full generation length.
- **Few-shot:** HumanEval/TruthfulQA 0; GSM8K, Minerva Math500 4; MBPP 3; BBH 3.

**Examples:**

```bash
# GSM8K, pure diffusion
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --compile_model False

# HumanEval, block diffusion
./eval.sh --batch_size 1 --mode bd --task_name humaneval --model llada --compile_model False

# With intervention and wandb
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --intervention_mode drop --intervention_steps 0,128,256 --intervention_k 1 --wandb True
```

**Batching:** Requests are grouped by **prompt length buckets** (e.g. 0–32, 32–64, …, 512+ tokens). Chunks are formed within each bucket with size up to `batch_size`. So you may see chunk sizes smaller than `batch_size` when a bucket has fewer samples; this keeps similar-length prompts together and reduces padding and wasted steps.

**Outputs:**

- **Results:** `outputs/{model}_{task_name}_{mode}_{timestamp}.json`
- **Logs:** `logs/{model}_{task_name}_{mode}_{timestamp}.log`
- **Traces:** If tracing is enabled, one file per task: `trace_outputs/{trace_name}.json` (contains all samples for that run in a single JSON structure).

---

## Hydra / `launch_eval.py` (single and multirun)

Config lives in `config/config.yaml`. Override from CLI: `task_name=gsm8k mode=pd batch_size=4`.

**Single run (local):**
```bash
python launch_eval.py task_name=gsm8k mode=pd
```

**Multirun locally (sequential):**
```bash
python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun
```

**Multirun on SLURM (one job per config):**

Launchers are in `config/launcher/`: `local`, `slurm_a40`, `slurm_dgx`, `slurm_h100`, `slurm_l40`, `slurm_v100`.

```bash
python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun launcher=slurm_a40
```

Install for SLURM: `pip install hydra-core hydra-submitit-launcher` (or use `requirements.txt`).

---

## Testing Schedulers

Use `test_schedulers.py` or `test_schedulers.sh` to run one or all schedulers on a fixed prompt (no full eval harness).

**Single scheduler, default (single-sample) tests:**
```bash
python test_schedulers.py --scheduler pure_diffusion
python test_schedulers.py   # run all schedulers
```

**Batched generation tests (shape checks + optional tracing):**
```bash
python test_schedulers.py --batch --batch_size 2
python test_schedulers.py --batch --scheduler pure_diffusion --batch_size 4
```

**With wandb and tracing:**
```bash
python test_schedulers.py --batch --wandb
```

**Using the shell script** (sets env, conda, and forwards flags):

```bash
./test_schedulers.sh --scheduler pure_diffusion
./test_schedulers.sh --batch --batch_size 2
./test_schedulers.sh --batch --wandb
```

Script options: `--scheduler`, `--wandb`, `--wandb_project`, `--wandb_name`, `--intervention_mode`, `--batch`, `--batch_size`, `--tracing_enabled`. Run `./test_schedulers.sh --help` for the full list.

---

