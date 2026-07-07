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
| `--trace_hf_upload`                          | `True` / `False` — push final trace file to a Hub dataset when tracing is on (default: follows tracing).                                                                                                                                   |
| `--trace_hf_dataset_repo`                    | Hub dataset id (default `schedulerlens`; use `org/schedulerlens` for org repos).                                                                                                                                                             |
| `--trace_hf_namespace`                       | HF user/org prefix when `trace_hf_dataset_repo` has no slash (else uses `whoami`).                                                                                                                                                            |
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
| `--run_id`                                   | With `--resumable`, fixed folder under `outputs/` (default auto: `{model}_{task}_{mode}_s{steps}`, with `_{strategy}` for `top_k_*` and intervention tags when set). Use a custom id when varying steps/tracing so checkpoints do not collide. |
| `--offline`                                  | Use cached HF datasets/models only and `WANDB_MODE=offline` (no Hub or W&B network). Disables trace Hub upload. Prefetch datasets with `./download_datasets.sh` on a networked machine first. |
| `--device`                                   | Harness device / GPU pin: `0`, `1`, `cuda`, `cuda:0`, `cpu`. Integer or `cuda:N` sets `CUDA_VISIBLE_DEVICES` and passes `device=cuda` to the harness. Default (local, non-SLURM): GPU `0`. Ignored for GPU binding when `--slurm` allocates the job GPU. |
| `--launcher`                                 | With `--parallel_config`: cluster launcher for auto-submit from login node (e.g. `slurm_dgx`, `slurm_h100`). Requests one SLURM job with `--gres=gpu:N` (N = slots in parallel YAML). Omit on an already-allocated node (`launcher=local` default). |
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
# Offline on GPU 1 (datasets must be cached; W&B logs locally)
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --offline --device 1 --wandb True
```

### Offline and device selection

**Offline (`--offline` / `offline=true`):** for compute nodes without outbound network access. Sets `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and `WANDB_MODE=offline`. Trace Hub upload is forced off (`TRACE_HF_UPLOAD=FALSE`). Models must already live under `./models/`; lm-eval datasets must be in the HF cache — run `./download_datasets.sh` once from a machine with network access. W&B metrics are written under a local `wandb/` directory and can be synced later with `wandb sync`.

**Device (`--device` / `device=`):** pick the GPU interactively without exporting `CUDA_VISIBLE_DEVICES` yourself.

| Value | `CUDA_VISIBLE_DEVICES` | Harness `device` |
| ----- | ---------------------- | ---------------- |
| `0`, `1`, … | that index | `cuda` |
| `cuda:0`, `cuda:1` | index after `:` | `cuda` |
| `cuda` | unchanged, or `0` locally if unset | `cuda` |
| `cpu` | cleared | `cpu` |

Locally (without `--slurm`), the default is GPU `0` when `--device` is omitted. With `--slurm`, SLURM/`srun` owns GPU allocation — pass `--device` only if you need to override that binding.

Hydra equivalents: `offline=true`, `device=1`, `device=cuda:0`, `device=cpu`. Defaults in `config/lmeval.yaml`: `offline: false`, `device: null`.

### Outputs

- **Default run:** `outputs/{model}_{task_name}_{mode}_s{steps}_{timestamp}.json`  
- **Resumable run:** `outputs/{run_id}/` with `results.json` (see below)  
- **Logs:** `logs/{model}_{task_name}_{mode}_{timestamp}.log`  
- **Traces (if enabled):** `trace_outputs/{trace_name}.json` or `.jsonl` when resumable (`trace_name` matches the run suffix, including `_s{steps}` unless you set `--run_id`)

---

## Resumable evaluation

Long jobs can save partial progress and continue with the same command.

**Enable:** `--resumable` in `eval.sh`, or `resumable=true` in `config/lmeval.yaml` / CLI. Optional `run_id=my_run` for a custom folder.

**Paths:** without resumable, filenames get a timestamp. With resumable, outputs live under `outputs/{run_id}/` (`run_id` or `{model}_{task}_{mode}_s{steps}` plus `_{strategy}` for `top_k` when not using explicit `run_id`, plus intervention tags when set). Diffusion step count is in auto-generated trace/checkpoint names (e.g. `llada1_5_mbpp_fhs_s512`); override entirely with `--run_id`.

**Checkpoint file:** `outputs/{run_id}/checkpoint.jsonl` — one line per completed sample (`doc_id`, `task_name`, `prompt`, `resps`). On restart, those pairs are skipped; new lines append so crashes do not lose work.

**Diffusion aggregates (steps / NFE):** With `resumable`, the harness also writes `**outputs/{run_id}/.eval_diffusion_metrics.json`** so a **new Python process** can restore cumulative diffusion counters next to `checkpoint.jsonl`.

**W&B:** `launch_eval.py` passes `resume=allow,id=<run_id>` to lm-eval when `outputs/{run_id}/.wandb_id` exists. The harness writes that file the first time the active W&B run id is known — including at the **start of the first `generate_until`**, because Weights & Biases is often initialized **after** the model is constructed (so writing only in the model constructor was unreliable). Until `.wandb_id` exists, the next launch may start a **new** W&B run. With `--offline` / `offline=true`, runs log locally (`WANDB_MODE=offline`); sync later with `wandb sync`.

**Tracing:** with tracing on, JSONL under `trace_outputs/{trace_name}.jsonl` supports appending after resume (`trace_name` includes `_s{steps}` in auto names, e.g. `llada1_5_mbpp_fhs_s512.jsonl`). When tracing is enabled, the final trace file is also pushed to a Hugging Face **dataset** repo (default `schedulerlens`, resolved as `{HF user}/schedulerlens` unless you pass `org/schedulerlens`). Disable with `trace_hf_upload=false` in `config/lmeval.yaml` or `TRACE_HF_UPLOAD=FALSE` / `--trace_hf_upload False` in `eval.sh`. Configure repo via `trace_hf_dataset_repo` and optional `trace_hf_namespace`. Requires `HF_TOKEN` or `huggingface-cli login`. With `--offline` / `offline=true`, Hub trace upload is always disabled — push traces later with `traces/hf_trace_upload.py` (see below).

**Manual trace upload (offline or retry):** upload one file, every file in a directory, or a glob expansion from the shell:

```bash
python traces/hf_trace_upload.py upload trace_outputs/llada_gsm8k_pd_s512.jsonl
python traces/hf_trace_upload.py upload trace_outputs/
python traces/hf_trace_upload.py upload trace_outputs/ --recursive --repo org/schedulerlens
python traces/hf_trace_upload.py download llada_gsm8k_pd_s512.jsonl -o trace_outputs/llada_gsm8k_pd_s512.jsonl
```

Uses `TRACE_HF_DATASET_REPO` / `TRACE_HF_NAMESPACE` from the environment when `--repo` is omitted (same defaults as eval). `--dry-run` lists planned uploads without calling the Hub.

**Changing steps or tracing:** use a **new** `--run_id` (or let auto naming pick a new `_s{steps}` suffix). Do not resume an existing checkpoint with different `--steps` — sample outputs would be inconsistent.

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
python launch_eval.py task_name=gsm8k mode=pd offline=true device=1
```

YAML keys `**gen_length**`, `**steps**`, and `**block_length**` live in `config/lmeval.yaml` under the sampling block (`null` = defaults: lengths from `sampling_defaults.py`; block size from mode — see “Auto-configured from task”). Env keys `offline` and `device` are in the same file.

**Multi-GPU (one job):** set `parallel_runs_file` in `config/lmeval.yaml` or on the CLI to a YAML file (list of dicts, one per GPU; see `config/parallel_eval_runs.yaml`). `launch_eval.py` merges each dict onto the base config and binds **GPU `i`** to `CUDA_VISIBLE_DEVICES=i` unless that slot sets `device:` (e.g. `device: cuda:1`).

**From a SLURM login node (auto-submit one N-GPU job):** add a cluster launcher — the slot count sets `--gres=gpu:N`. Launcher detection uses Hydra's config-group choice (`HydraConfig.runtime.choices.launcher`), not a top-level `cfg.launcher` key. Inside the allocation (or after auto-submit), evals run in parallel on GPUs 0…N−1.

```bash
# Login node → one DGX job with 2 GPUs (matches 2 entries in parallel_eval_runs.yaml)
python launch_eval.py parallel_runs_file=config/parallel_eval_runs.yaml launcher=slurm_dgx offline=true

# Same via eval.sh
./eval.sh --parallel_config config/parallel_eval_runs.yaml --launcher slurm_dgx --offline
```

If you are **already on an allocated node** (`salloc` / inside `sbatch`) with enough GPUs, use `launcher=local` (default) — no re-submit:

```bash
python launch_eval.py parallel_runs_file=config/parallel_eval_runs.yaml offline=true
```

If you `sbatch eval.sh --parallel_config ... --slurm`, set `#SBATCH --gres=gpu:N` to match the number of YAML slots (the script checks this at startup).

```bash
python launch_eval.py parallel_runs_file=config/parallel_eval_runs.yaml
```

From `eval.sh` (after conda; no inner `srun` for the children — same as other multi-GPU paths):

```bash
./eval.sh --parallel_config config/parallel_eval_runs.yaml
# Optional global defaults still apply on top of lmeval.yaml: --batch_size 4 --wandb True --steps 256 --gen_length 512 --offline --device 0
python launch_eval.py parallel_runs_file=config/parallel_eval_runs.yaml offline=true
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

---

## Fetching W&B metrics (`fetch_evals/`)

After eval runs are logged to Weights & Biases, export a flat CSV with task scores and NFE:

```bash
cd /path/to/Strace
python fetch_evals/fetch_wandb_metrics.py
```

**Default output:** `fetch_evals/strace_metrics.csv`

**W&B project:** `devichand579-iit-kharagpur/strace` (override with `--project`).

Run names are parsed as `{model}_{task}_{mode}_s{steps}` (same as `eval.sh` / resumable `outputs/{run_id}/`).

### Metrics exported

| Task | Score column |
| ---- | ------------ |
| `gsm8k` | `strict_match` (not flexible `exact_match`) |
| `minerva_math500` | `exact_match` (fallback `math_verify`) |
| `arc_challenge_chat` | `exact_match` |
| `mbpp`, `humaneval` | `pass_at_1` |
| All runs | `avg_nfe_per_sample` (`global/avg_nfe_per_sample` from eval) |

The script reads W&B run summaries first, then falls back to W&B history, then local checkpoint files under `outputs/{run}/` when `--no-merge-local` is not set (`results.json`, `.eval_diffusion_metrics.json`, optional SchedulerLens `metrics.json`).

### Options

```bash
python fetch_evals/fetch_wandb_metrics.py --include-unfinished   # include crashed/running runs
python fetch_evals/fetch_wandb_metrics.py --no-merge-local       # W&B only
python fetch_evals/fetch_wandb_metrics.py -o fetch_evals/my.csv  # custom output path
```
---

**EOS harness (`use_eos_harness`, default `true` in `config/lmeval.yaml`):** when `true`, schedulers may exit early on the batch EOS stop condition and chunk decode truncates at the first EOS (legacy harness behavior). For reference LLaDA-style sampling (full step budget, `batch_decode` of the full generated tail), set `**use_eos_harness=false`** in YAML, e.g. `python launch_eval.py task_name=gsm8k mode=pd use_eos_harness=false`, or in `**eval.sh`:** `--no_eos_harness` or `--use_eos_harness false`. The same flag is passed into `LLaDAEvalHarness` via `model_args`. Scheduler smoke tests use `**--no-eos-harness`** (hyphens) on `test_schedulers.py` — see [Scheduler testing](scheduler-testing.md).