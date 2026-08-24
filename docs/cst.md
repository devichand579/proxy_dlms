# CST: pre-logit capture, training, and evaluation

**CST** captures last-layer pre-logit hidden states, trains a sparse dictionary on them, and evaluates alignment metrics.

## Two entry points

Collection and training are **separate** (separate config blocks, separate jobs). Run **`mode=store`** once (LM inference -> Hub), then **`mode=train`** later (download from Hub -> fit CST, no LM).

Everything goes through **`hydra_runners/run_cst.py`** / **`./scripts/cst.sh`** with `mode=store`, `mode=train`, or `mode=eval`.

| | **`mode=store`** | **`mode=train`** / **`mode=eval`** |
|---|---|---|
| **Purpose** | Run LM+scheduler on prompts -> **Hugging Face** | **Train** or **score** CST from stored hiddens |
| **Config** | `store.*` only | `train.*` + `cst.*` / `eval.*` |
| **Needs GPU + model** | Yes | **No** for train (Hub download only); eval scoring only |
| **Output** | Chunks on `{whoami}/hiddens-train` or `{whoami}/hiddens-eval` | CST checkpoints under `outputs/cst_ckpt/` or eval metrics |

`huggingface-cli login` (or `HF_TOKEN`) is enough. Repos default to **`{your-username}/hiddens-train`** and **`{your-username}/hiddens-eval`**.

```bash
# 1) Store (once per model / scheduler / gen_length / steps)
python hydra_runners/run_cst.py mode=store store.scope=train
python hydra_runners/run_cst.py mode=store store.scope=eval store.task_name=gsm8k

# 2) Train CST from Hub (no inference). Lookup keys must match the store run.
python hydra_runners/run_cst.py mode=train

# 3) Eval CST from Hub or local folder
python hydra_runners/run_cst.py mode=eval eval.hidden_source=hf_hub eval.task_name=gsm8k \
  eval.checkpoint=outputs/cst_ckpt/cst_llada_pd_512_256_k4096_w4_ls0.001_best.pt
python hydra_runners/run_cst.py mode=eval eval.hidden_dir=/path/to/hiddens eval.checkpoint=...
```

**lm-eval + CST metrics during benchmarks:** `hydra_runners/launch_eval.py` with `cst_checkpoint=...` (separate from the table above).

**Config:** `config/cst.yaml` — `store.*`, `train.*`, `eval.*`, `cst.*`. **Code:** `cst/`, `cst/hf_hiddens/`.

---

## Pre-logit hidden states

Representations **immediately before the vocabulary projection** (post final norm). Implementation: `cst/pre_logits.py`.

Each `.pt` file (minimal on-disk schema):

- `hidden_states` — bfloat16 tensors (one per forward during generation)
- `meta.prompt_len` — padded prompt length for eval splits and TAG

Legacy chunks may still include extra `meta` fields or `token_unmasked` / `token_diffusion_masked`; loaders accept both.

Loaded as `[T, B, L, D]` float32 via `cst.train.hidden_chunk.load_chunk_H`.

**Filename examples:**

- Train: `llada_pd_512_256_000042.pt`
- Eval: `llada_pd_512_256_task_gsm8k_000042.pt`

Pattern: `{name_token}_{batch_idx:06d}.pt` where `name_token` is `{model}_{scheduler}_{gen_length}_{steps}`.

---

## Store hidden data (`mode=store`)

Uses **`store.*` only** (scheduler, gen length, steps, batch size, Hub credentials). Does **not** read `train.*` or `cst.*`.

### Train hiddens (`store.scope=train`)

Runs the collect loop (sample prompts -> LM + scheduler -> one `.pt` per inference batch) and **uploads each batch to Hub**. Does **not** train CST.

```bash
huggingface-cli login
python hydra_runners/run_cst.py mode=store store.scope=train
```

Default repo: **`{whoami}/hiddens-train`**.

### Eval hiddens (`store.scope=eval`)

Runs the **lm-eval task** through the model + scheduler, writing each batch `.pt` and uploading it immediately. Optional `store.samples` caps how many eval instances to capture (default: all task documents).

```bash
python hydra_runners/run_cst.py mode=store store.scope=eval store.task_name=gsm8k
python hydra_runners/run_cst.py mode=store store.scope=eval store.task_name=gsm8k store.samples=128
```

Default repo: **`{whoami}/hiddens-eval`**. Requires `store.task_name` (or `store.name_token` override). Eval tokens append `_task_{task}`.

`manifest.json` (one stored dataset per `name_token`):

```json
{
  "schema_version": 2,
  "scope": "train",
  "name_token": "llada_pd_512_256",
  "chunks": [
    {"batch_idx": 42, "filename": "llada_pd_512_256_000042.pt", "bytes": 12345678}
  ],
  "sampling_fingerprint": {"train_samples_per_epoch": 1000, "train_seed": 0}
}
```

---

## Train CST (`mode=train`)

**Requires** hiddens on Hub from `mode=store store.scope=train` first. Uses **`train.*` + `cst.*`**; Hub lookup is `train.model_path` / `train.scheduler_mode` / `train.gen_length` / `train.steps` (must match the store run) or `train.name_token`.

```bash
python hydra_runners/run_cst.py mode=train
```

Training never downloads the full subfolder at once. Each step:

1. Download the next batch `.pt`
2. Train CST on that file
3. Delete the local file
4. Repeat until all batches in the manifest are consumed
5. If `train.num_epochs > 1`, repeat steps 1–4 for another pass over the **same** stored data

| Setting | Default | Meaning |
|---------|---------|---------|
| `num_epochs` | `1` | CST passes over the stored dataset |
| `store.samples` | `1000` | How many prompts were collected at **store** time; train reads batch count from the manifest |

**Resume:** `train.resume_from_latest=true` — restores CST weights and batch cursor (`data_next_epoch` = CST pass, `data_next_batch_idx`).

---

## Eval CST (`mode=eval`)

Score a CST checkpoint against saved hiddens (no lm-eval loop). Hub lookup uses **`eval.*`**.

```bash
# Local folder
python hydra_runners/run_cst.py mode=eval \
  eval.hidden_dir=outputs/eval_hidden \
  eval.checkpoint=outputs/cst_ckpt/cst_llada_pd_512_256_k4096_w4_ls0.001_best.pt

# From Hub
python hydra_runners/run_cst.py mode=eval eval.hidden_source=hf_hub eval.task_name=gsm8k \
  eval.checkpoint=outputs/cst_ckpt/cst_llada_pd_512_256_k4096_w4_ls0.001_best.pt
```

---

## Checkpoints and naming

**Hub `name_token`:** `{model}_{scheduler}_{gen_length}_{steps}`  
Example: `llada_pd_512_256`  
Eval store: `llada_pd_512_256_task_gsm8k`

**Hidden chunk files** (`store.scratch_dir` / `train.scratch_dir` / Hub `chunks/`):

- `{name_token}_{batch_idx:06d}.pt`

**CST checkpoints** (`train.checkpoint_dir`, default `outputs/cst_ckpt`) add k/w/λ so different CST hypers on the same hiddens do not clobber:

- `cst_llada_pd_512_256_k4096_w4_ls0.001_latest.pt`
- `cst_llada_pd_512_256_k4096_w4_ls0.001_best.pt`
- `cst_llada_pd_512_256_k4096_w4_ls0.001_state.json`

---

## lm-eval with CST

```bash
python hydra_runners/launch_eval.py task_name=gsm8k \
  cst_checkpoint=outputs/cst_ckpt/cst_llada_pd_512_256_k4096_w4_ls0.001_best.pt
```

See [Evaluation](evaluation.md) for resumable eval and CST aggregates.

---

## `cst.train_data`

Mixed prompt pools for **train store** only. GSM8K, Minerva (`minerva_math500` formatter + fixed 4-shot), ARC-Challenge, and ARC-Easy use lm-eval `fewshot_context` with the same `k` as `launch_eval` (GSM8K/Minerva 4, ARC 5). Queries still come from each source's **train** split. TACO is 0-shot (like HumanEval). CommonsenseQA is not in the eval suite, so it keeps the 5-shot chat formatter with labeled validation demos. See `cst/train_data/compose.py`.

---

## Related files

| File | Role |
|------|------|
| `hydra_runners/run_cst.py` | **`mode=store`** / **`train`** / **`eval`** (single entry point) |
| `config/cst.yaml` | Separate `store.*` / `train.*` / `eval.*` blocks |
| `scripts/cst.sh` | Shell / Slurm wrapper |

See also: [Setup](setup.md), [Evaluation](evaluation.md).
