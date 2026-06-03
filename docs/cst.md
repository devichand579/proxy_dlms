# CST: pre-logit capture, training, and evaluation

**CST** captures last-layer pre-logit hidden states, trains a sparse dictionary on them, and evaluates alignment metrics.

## Two entry points

Collection and training are **separate**. Run **`mode=store`** once (LM inference → Hub), then **`mode=train`** (download from Hub → fit CST, no LM).

Everything goes through **`run_cst.py`** / **`./cst.sh`** with `mode=store`, `mode=train`, or `mode=eval`.

| | **`mode=store`** | **`mode=train`** / **`mode=eval`** |
|---|---|---|
| **Purpose** | Run LM+scheduler on prompts → **Hugging Face** (train mix or eval task) | **Train** or **score** CST from stored hiddens |
| **Needs GPU + model** | Yes (collect + scheduler for both train and eval store) | **No** for train (Hub download only); eval scoring only |
| **Output** | Chunks on `{namespace}/hiddens-train` or `{namespace}/hiddens-eval` | CST checkpoints under `outputs/cst_ckpt/` or eval metrics |

`train.hf_hiddens_org` is your **HF username** if you have no org (e.g. `devichand` → [devichand/hiddens-train](https://huggingface.co/datasets/devichand/hiddens-train)), or an org slug otherwise.

```bash
# 1) Store (once per model / scheduler / task)
python run_cst.py mode=store store.scope=train
python run_cst.py mode=store store.scope=eval store.task_name=gsm8k

# 2) Train CST from Hub (no inference)
python run_cst.py mode=train

# 3) Eval CST from Hub or local folder
python run_cst.py mode=eval eval.hidden_source=hf_hub eval.task_name=gsm8k \
  eval.checkpoint=outputs/cst_ckpt/cst_..._train_best.pt
python run_cst.py mode=eval eval.hidden_dir=/path/to/hiddens eval.checkpoint=...
```

**lm-eval + CST metrics during benchmarks:** `launch_eval.py` with `cst_checkpoint=...` (separate from the table above).

**Config:** `config/cst.yaml` — `store.*`, `train.*`, `cst.*`. **Code:** `cst/`, `cst/hf_hiddens/`, `hidden_hf_worker.py`.

---

## Pre-logit hidden states

Representations **immediately before the vocabulary projection** (post final norm). Implementation: `cst/pre_logits.py`.

Each `.pt` file (minimal on-disk schema):

- `hidden_states` — bfloat16 tensors (one per forward during generation)
- `meta.prompt_len` — padded prompt length for eval splits and TAG

Legacy chunks may still include extra `meta` fields or `token_unmasked` / `token_diffusion_masked`; loaders accept both.

Loaded as `[T, B, L, D]` float32 via `cst.train.hidden_chunk.load_chunk_H`.

**Filename examples** (same pattern for train and eval):

- Train: `llada_scheduler_pd_k4096_w4_ls0.001_train_seg000_000042.pt`
- Eval: `llada_scheduler_pd_k4096_w4_ls0.001_task_gsm8k_seg000_000042.pt`

Pattern: `{name_token}_seg{segment:03d}_{batch_idx:06d}.pt`

---

## Store hidden data (`mode=store`)

### Train hiddens (`store.scope=train`)

Runs the full collect loop (sample prompts → LM + scheduler → `.pt` until `target_staging_gb` per segment) and **uploads each segment to Hub**. Does **not** train CST.

```bash
huggingface-cli login
python run_cst.py mode=store store.scope=train
```

Uses `train.target_staging_gb`, `train.scheduler_mode`, `train.samples_per_epoch`, etc.

Default repo: **`{hf_hiddens_org}/hiddens-train`**.

### Eval hiddens (`store.scope=eval`)

Runs the **full lm-eval task** through the model + scheduler (same collect loop as train), writing segments to `train.staging_dir` and uploading each slice to Hub. Optional `store.samples` caps how many eval instances to capture (default: all task documents).

```bash
python run_cst.py mode=store store.scope=eval store.task_name=gsm8k
# Optional: cap instances or segment size
python run_cst.py mode=store store.scope=eval store.task_name=gsm8k store.samples=128
```

Default repo: **`{hf_hiddens_org}/hiddens-eval`**. Requires `store.task_name` (or `store.name_token` override).

Each segment in `manifest.json` (no epoch field — one stored dataset per `name_token`):

```json
{
  "schema_version": 2,
  "scope": "train",
  "name_token": "llada_scheduler_pd_k4096_w4_ls0.001_train",
  "segments": [
    {
      "segment": 0,
      "prefix": "llada_scheduler_pd_k4096_w4_ls0.001_train_seg000",
      "chunks": [{"batch_idx": 42, "filename": "llada_scheduler_pd_k4096_w4_ls0.001_train_seg000_000042.pt", "bytes": 12345678}],
      "total_bytes": 12345678,
      "next_batch_idx": 1,
      "sampling_fingerprint": {"train_samples_per_epoch": 2048, "train_seed": 0}
    }
  ]
}
```

Upload segments in the manifest are bookkeeping only; train uses a **global batch cursor** across them (`batch_idx` in each filename).

### Restricted GPU / worker queue

When the store or train machine cannot reach the Hub:

```yaml
train:
  hf_use_worker_queue: true
  hf_worker_queue_dir: outputs/hf_worker_queue
```

On a login machine: `python hidden_hf_worker.py poll --queue-dir outputs/hf_worker_queue --watch`

---

## Train CST (`mode=train`)

**Requires** hiddens on Hub from `mode=store store.scope=train` first.

```bash
python run_cst.py mode=train
```

Training never downloads the full subfolder at once. Each step:

1. Download the next slice of chunks until `train.target_staging_gb`
2. Train CST on those local `.pt` files
3. Delete local chunks
4. Repeat until all batches in the manifest are consumed
5. If `train.num_epochs > 1`, repeat steps 1–4 for another pass over the **same** stored data

Hub layout is one `{name_token}/` subfolder per model + scheduler + k/w/λ (train) or task (eval). Upload segments in the manifest are bookkeeping only; train uses a **global batch cursor** across them.

| Setting | Default | Meaning |
|---------|---------|---------|
| `target_staging_gb` | `1.0` | Max GB per download slice (store upload segments use the same cap) |
| `num_epochs` | `1` | CST passes over the stored dataset (not store-time epochs) |
| `samples_per_epoch` | `2048` | Used at **store** time (`store.samples` override); train reads batch count from manifest |

**Resume:** `train.resume_from_latest=true` — restores CST weights and batch cursor (`data_next_epoch` = CST pass, `data_next_batch_idx`).

---

## Eval CST (`mode=eval`)

Score a CST checkpoint against saved hiddens (no lm-eval loop):

```bash
# Local folder
python run_cst.py mode=eval \
  eval.hidden_dir=outputs/eval_hidden \
  eval.checkpoint=outputs/cst_ckpt/cst_..._train_best.pt

# From Hub
python run_cst.py mode=eval eval.hidden_source=hf_hub eval.task_name=gsm8k \
  eval.checkpoint=outputs/cst_ckpt/cst_..._train_best.pt
```

---

## Checkpoints and naming

Paths encode **model**, **scheduler**, **`cst.k`**, **`cst.w`**, and **`train.lambda_sparsity`** (as `ls…` in the token).

Hyperparam segment: `k4096_w4_ls0.001` — from `cst.k`, `cst.w`, and `train.lambda_sparsity`.

**Training `name_token`:** `{model}_scheduler_{sched}_k{K}_w{W}_ls{λ}_train`  
**Eval `name_token`:** `{model}_scheduler_{sched}_k{K}_w{W}_ls{λ}_task_{task}`

**Hidden chunk files** (`train.staging_dir` / Hub `chunks/`):

- `{name_token}_seg{segment:03d}_{batch_idx:06d}.pt` — same pattern for train and eval

**CST checkpoints** (`train.checkpoint_dir`, default `outputs/cst_ckpt`):

- `cst_<model>_scheduler_<sched>_k<K>_w<W>_train_latest.pt`
- `cst_..._train_best.pt`
- `cst_..._train_state.json`

---

## lm-eval with CST

```bash
python launch_eval.py task_name=gsm8k \
  cst_checkpoint=outputs/cst_ckpt/cst_llada_scheduler_pd_k4096_w4_ls0.001_train_best.pt
```

See [Evaluation](evaluation.md) for resumable eval and CST aggregates.

---

## `cst.train_data`

Mixed prompt pools for **train store** only: GSM8K, Minerva, ARC, CSQA, TACO. See `cst/train_data/compose.py`.

---

## Related files

| File | Role |
|------|------|
| `run_cst.py` | **`mode=store`** / **`train`** / **`eval`** (single entry point) |
| `hidden_hf_worker.py` | Hub I/O worker + queue |
| `config/cst.yaml` | All modes |
| `cst.sh` | Shell / Slurm wrapper |

See also: [Setup](setup.md), [Evaluation](evaluation.md).
