# Evaluation (lm-eval)

STrace plugs into the **lm-evaluation-harness** with a custom `llada` / `llada1_5` / `dream` model. Two common entry points:

- `**eval.sh`** — sets generation length, steps, few-shot count, and paths from the task name.
- `**launch_eval.py**` — Hydra-driven; good for **multirun** and **SLURM** launchers.

Config defaults: `config/lmeval.yaml`. Override on the CLI: `task_name=gsm8k mode=pd batch_size=4`.

Task-specific **generation length** and **diffusion step count** are centralized in `**sampling_defaults.py`** (`get_gen_length_steps` / `resolve_gen_length_steps`): ARC chat defaults to 128/128; most other benchmarks use 512/512 unless you override (see below).

**LLaDA 1.5:** when `model` points at `./models/llada1_5` (or any checkpoint whose directory name is `llada1_5`), the harness passes that path into `schedulers.generate`, which turns on EOS/EoT suppression during unmasking (`load_model.infer_b44_eos_unmask_safety`) so that we force EOS to be excluded from the generatd tokens. Other `model` directories are unchanged. Optional `config.json` override: `strace_b44_eos_unmask`. Background: [LLaDA 1.5 sampling (B.4)](llada1_5-sampling.md).

---

## `eval.sh`

**Basic example:**

```bash
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --compile_model False
```

### Arguments


| Argument                                     | Description                                                                                                                                                                                                                                  |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--batch_size`                               | Max requests per chunk (e.g. 2, 4, 8). Chunks may be smaller when a length bucket has fewer samples.                                                                                                                                         |
| `--mode`                                     | Scheduler: `pd`, `bd`, `eb`, `fdm`, `fhs`, `top_k_c`, `top_k_m`, `dus` (see shortcuts below).                                                                                                                                                |
| `--task_name`                                | Eval task: `gsm8k`, `humaneval`, `mbpp`, `minerva_math500`, `arc_challenge_chat`, etc.                                                                                                                                                       |
| `--parallel_config` | `--parallel_runs_file` | Path to a YAML file: list of per-GPU overrides (merged onto `config/lmeval.yaml`). Same as `python launch_eval.py parallel_runs_file=...`. Skips required `--mode` / `--task_name` / `--model` on the shell line (set per slot in the file). |
| `--model`                                    | Model dir under `./models/` (e.g. `llada`).                                                                                                                                                                                                  |
| `--compile_model`                            | `True` / `False` for `torch.compile`.                                                                                                                                                                                                        |
| `--wandb`                                    | `True` to log to Weights & Biases.                                                                                                                                                                                                           |
| `--tracing_enabled`                          | `True` to write per-task traces to `trace_outputs/`.                                                                                                                                                                                         |
| `--intervention_mode`                        | Optional: `drop` or `swap`.                                                                                                                                                                                                                  |
| `--intervention_steps`                       | Optional: comma-separated step indices (e.g. `0,128,256`).                                                                                                                                                                                   |
| `--intervention_k`                           | Optional: tokens per intervention step (default 1).                                                                                                                                                                                          |
| `--block_length`                             | Optional: scheduler block length (otherwise mode defaults below).                                                                                                                                                                            |
| `--top_k`                                    | Optional: top‑k scheduler only — tokens transferred per diffusion step (`config/lmeval.yaml` often sets `top_k`; can override here or with `top_k=` in Hydra).                                                                               |
| `--steps`                                    | Optional: diffusion step budget (**not** intervention indices); must be ≤ `gen_length`. Defaults follow the task (128 or 512) unless you override.                                                                                           |
| `--gen_length`                               | Optional: generated length in tokens; must be ≥ `steps` when both are set. Defaults follow the task.                                                                                                                                         |
| `--num_fewshot`                              | Optional: override per-task few-shot (defaults vary by task).                                                                                                                                                                                |
| `--slurm`                                    | Use when submitting via `sbatch` so the command runs under `srun`.                                                                                                                                                                           |
| `--resumable`                                | Checkpointed evaluation: fixed output directory, sample-level resume (see below).                                                                                                                                                            |
| `--run_id`                                   | With `--resumable`, directory name under `outputs/` (default: `{model}_{task}_{mode}`, or `{model}_{task}_{top_k}_{strategy}` for `top_k_*` modes, or includes intervention tags when set).                                                  |
| Hidden / CST                                 | Use `**capture_hiddens.sh`** / `**launch_capture_hiddens.py**`, then `**hidden_cst_eval.sh**` / `**launch_hidden_cst_eval.py**` — see [CST](cst.md). `eval.sh` no longer accepts `--capture_pre_logits` or `--cst_checkpoint`.               |


### Mode shortcuts

- `pd` — pure diffusion  
- `bd` — block diffusion  
- `eb` — EB sampler  
- `fdm` — FDM  
- `fhs` — FHS  
- `top_k_c` — top-k (confidence)  
- `top_k_m` — top-k (top margin)  
- `dus` — dilated unmasking scheduler

### Auto-configured from task

- **Length & steps:** `arc_challenge_chat` uses 128/128; most other lm-eval tasks use 512/512 (see `sampling_defaults.py`). Override with `**--steps` / `--gen_length`** in `eval.sh`, or `**steps=` / `gen_length=**` on the Hydra CLI / in YAML (`null` keeps task defaults). The launcher checks `**steps ≤ gen_length**`.  
- **Block length:** `block_diffusion` 32, `fdm` 64, `dus` 16, else `gen_length`. Override: `--block_length` / `block_length=`.  
- **Top-k:** scheduler-specific; override: `--top_k` / `top_k=`.  
- **Few-shot:** e.g. HumanEval / `arc_challenge_chat` 0; GSM8K / Minerva 4; MBPP 3. Override: `--num_fewshot` / `num_fewshot:`.

### Batching

Requests are grouped by **prompt length buckets** (e.g. 0–32, 32–64, …, 512+). Chunks are built within each bucket with size up to `batch_size`, so you may see smaller chunks when a bucket is sparse; this keeps similar lengths together and cuts padding.

### Examples

```bash
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --compile_model False
./eval.sh --parallel_config config/parallel_eval_runs.yaml --wandb True --slurm
./eval.sh --batch_size 1 --mode bd --task_name humaneval --model llada --compile_model False
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --intervention_mode drop --intervention_steps 0,128,256 --intervention_k 1 --wandb True
```

### Outputs

- **Default run:** `outputs/{model}_{task_name}_{mode}_{timestamp}.json`  
- **Resumable run:** `outputs/{run_id}/` with `results.json` (see below)  
- **Logs:** `logs/{model}_{task_name}_{mode}_{timestamp}.log`  
- **Traces (if enabled):** `trace_outputs/{trace_name}.json` or `.jsonl` when resumable

---

## Resumable evaluation

Long jobs can save partial progress and continue with the same command.

**Enable:** `--resumable` in `eval.sh`, or `resumable=true` in `config/lmeval.yaml` / CLI. Optional `run_id=my_run` for a custom folder.

**Paths:** without resumable, filenames get a timestamp. With resumable, outputs live under `outputs/{run_id}/` (`run_id` or `{model}_{task}_{mode}` plus `_{strategy}` for `top_k` when not using explicit `run_id`, plus intervention tags when set).

**Checkpoint file:** `outputs/{run_id}/checkpoint.jsonl` — one line per completed sample (`doc_id`, `task_name`, `prompt`, `resps`). On restart, those pairs are skipped; new lines append so crashes do not lose work.

**Diffusion aggregates (steps / NFE):** With `resumable`, the harness also writes `**outputs/{run_id}/.eval_diffusion_metrics.json`** so a **new Python process** can restore cumulative diffusion counters next to `checkpoint.jsonl`.

**W&B:** `launch_eval.py` passes `resume=allow,id=<run_id>` to lm-eval when `outputs/{run_id}/.wandb_id` exists. The harness writes that file the first time the active W&B run id is known — including at the **start of the first `generate_until`**, because Weights & Biases is often initialized **after** the model is constructed (so writing only in the model constructor was unreliable). Until `.wandb_id` exists, the next launch may start a **new** W&B run.

**Tracing:** with tracing on, JSONL under `trace_outputs/` supports appending after resume.

```bash
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --compile_model False --resumable
# Re-run the same command after failure — already-done samples are skipped
```

---

## Hydra and `launch_eval.py`

**Single run:**

```bash
python launch_eval.py task_name=gsm8k mode=pd
python launch_eval.py task_name=gsm8k mode=pd steps=256 gen_length=512
```

YAML keys `**gen_length**`, `**steps**`, and `**block_length**` live in `config/lmeval.yaml` under the sampling block (`null` = defaults: lengths from `sampling_defaults.py`; block size from mode — see “Auto-configured from task”).

**Multi-GPU (one job):** set `parallel_runs_file` in `config/lmeval.yaml` or on the CLI to a YAML file (list of dicts, one per GPU; see `config/parallel_eval_runs.yaml`). `launch_eval.py` merges each dict onto the base config and binds **GPU `i`** to `CUDA_VISIBLE_DEVICES=i`.

```bash
python launch_eval.py parallel_runs_file=config/parallel_eval_runs.yaml
```

From `eval.sh` (after conda; no inner `srun` for the children — same as other multi-GPU paths):

```bash
./eval.sh --parallel_config config/parallel_eval_runs.yaml
# Optional global defaults still apply on top of lmeval.yaml: --batch_size 4 --wandb True --steps 256 --gen_length 512
```

**Multirun (local, sequential):**

```bash
python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun
```

**Multirun on SLURM** (one job per config): launchers live in `config/launcher/`: `local`, `slurm_a40`, `slurm_a100`, `slurm_dgx`, `slurm_h100`, `slurm_l40`.

```bash
python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun launcher=slurm_a40
```

Install: `pip install hydra-core hydra-submitit-launcher` (or `requirements.txt`).

**EOS harness (`use_eos_harness`, default `true` in `config/lmeval.yaml`):** when `true`, schedulers may exit early on the batch EOS stop condition and chunk decode truncates at the first EOS (legacy harness behavior). For reference LLaDA-style sampling (full step budget, `batch_decode` of the full generated tail), set `**use_eos_harness=false`** in YAML, e.g. `python launch_eval.py task_name=gsm8k mode=pd use_eos_harness=false`, or in `**eval.sh`:** `--no_eos_harness` or `--use_eos_harness false`. The same flag is passed into `LLaDAEvalHarness` via `model_args`. Scheduler smoke tests use `**--no-eos-harness`** (hyphens) on `test_schedulers.py` — see [Scheduler testing](scheduler-testing.md).