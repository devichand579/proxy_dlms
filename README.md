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

Download the Dream 7B model before running evaluation or scheduler tests:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Dream-org/Dream-v0-Base-7B', local_dir='./models/dream', local_dir_use_symlinks=False)"
```

---

Download the LLaDA-1.5 model before running evaluation or scheduler tests:

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='GSAI-ML/LLaDA-1.5', local_dir='./models/llada1_5', local_dir_use_symlinks=False)"
```

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
| `--task_name` | Eval task: `gsm8k`, `humaneval`, `mbpp`, `minerva_math500`, `arc_challenge_chat`, etc. |
| `--model` | Model dir under `./models/` (e.g. `llada`). |
| `--compile_model` | `True` / `False` for `torch.compile`. |
| `--wandb` | `True` to log to Weights & Biases. |
| `--tracing_enabled` | `True` to write per-task traces to `trace_outputs/`. |
| `--intervention_mode` | Optional: `drop` or `swap`. |
| `--intervention_steps` | Optional: comma-separated step indices (e.g. `0,128,256`). |
| `--intervention_k` | Optional: tokens per intervention step (default 1). |
| `--num_fewshot` | Optional: override per-task few-shot count (defaults: humaneval/arc_challenge_chat 0, gsm8k/math500 4, mbpp 3). |
| `--slurm` | Use when submitting via `sbatch` so the command runs under `srun`. |
| `--resumable` | Checkpointed evaluation: fixed output directory, sample-level resume after crash or preemption (see below). |
| `--run_id` | With `--resumable`, directory name under `outputs/` (default: `{model}_{task}_{mode}`, or includes intervention tags when set). |

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

- **Length & steps:** arc_challenge_chat uses 128 tokens and 128 steps; other tasks use 512.
- **Block length:** block_diffusion 32, fdm 64, dus 16, else gen_length. Override with `--block_length <n>` (eval.sh) or `block_length=<n>` (Hydra).
- **Top-k:** For top_k scheduler, tokens to transfer per step. Default 1. Override with `--top_k <n>` (eval.sh) or `top_k=<n>` (Hydra).
- **Few-shot:** HumanEval, arc_challenge_chat 0; GSM8K, Minerva Math500 4; MBPP 3. Override with `--num_fewshot <n>` (eval.sh) or `num_fewshot: <n>` (Hydra).

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

- **Results (default run):** `outputs/{model}_{task_name}_{mode}_{timestamp}.json`
- **Results (resumable run):** directory `outputs/{run_id}/` with final metrics in `results.json` (see **Resumable evaluation**).
- **Logs:** `logs/{model}_{task_name}_{mode}_{timestamp}.log`
- **Traces:** If tracing is enabled, one file per run: `trace_outputs/{trace_name}.json` (full JSON document), or `{trace_name}.jsonl` (one sample per line) when resumable.

### Resumable evaluation

Long jobs can save partial progress and continue from the same command line.

**Enable**

- **Shell:** add `--resumable` to `./eval.sh` (optional `--run_id my_run` for a custom folder name).
- **Hydra:** set `resumable=true` in `config/config.yaml` or on the CLI; optional `run_id=my_run` overrides the default suffix.

**Stable paths (no timestamp)**  
Non-resumable runs use a timestamp in filenames so each invocation is isolated. With resumable mode, outputs go under `outputs/{run_id}/`, where `run_id` is `--run_id` if set, otherwise `{model}_{task}_{mode}` (and, if interventions are configured, a longer suffix that includes intervention mode, steps, and `k`).

**Checkpoint file**  
The harness writes `outputs/{run_id}/checkpoint.jsonl`. Each line is one completed sample: `doc_id`, `task_name`, `prompt`, and `resps`. On a new process, existing lines are read into a map `(task_name, doc_id) - response`; those requests skip generation and only missing samples are sent through the model. New completions are appended, so a crash mid-run does not discard finished work.

**Metrics and W&B**  
lm-eval writes the aggregated output to `outputs/{run_id}/results.json`. If W&B is enabled, the first run stores the run id in `outputs/{run_id}/.wandb_id`; a later invocation with the same resumable paths passes `resume=allow` so logging continues on the same W&B run.

**Tracing**  
With tracing on, scheduler traces use JSONL append mode under `trace_outputs/` so new samples can be appended after a resume (non-resumable runs use a single JSON array file).

**Example**

```bash
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --compile_model False --resumable
# same command again after failure or cancel — skips samples already in checkpoint.jsonl
```

---

## Hydra / `launch_eval.py` (single and multirun)

Config lives in `config/config.yaml`. Override from CLI: `task_name=gsm8k mode=pd batch_size=4`. For checkpointed runs: `resumable=true` and optionally `run_id=my_run`.

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

