# CST: pre-logit capture, training, and evaluation

**CST** (context in this repo) refers to the optional path that **captures** last-layer pre-logit hidden states, **trains** a sparse dictionary / bank on those chunks, and **evaluates** alignment metrics—either during lm-eval (per task) or **offline** on a folder of saved `.pt` files.

- **Code:** `cst/` (pre-logit wrapper, `CSTBank`, training loop, eval helpers)  
- **Train config:** `config/cst.yaml` (Hydra for `launch_cst.py` and `cst.sh`)  
- **LM eval config:** `config/config.yaml` (flags like `cst_checkpoint`, `capture_pre_logits`)

---

## Pre-logit hidden states

The harness can record **representations immediately before the vocabulary projection** (post final norm: `ln_f` / RMSNorm, depending on the model). Implementation: `cst/pre_logits.py`, wired from `eval.py` when the model is built with `capture_pre_logits=true` and a writable `hidden_capture_path`.

**Enable**

- **`eval.sh`:** `--capture_pre_logits True` and optionally `--hidden_capture_path /path/to/dir`  
- **`launch_eval.py` / YAML:** `capture_pre_logits=true`, optional `hidden_capture_path=...` (`null` uses the same auto paths as the shell script)

**Where files go (depends on lm-eval `resumable`, not on CST):**

- **Non-resumable (default):** each run uses a new timestamped folder; hiddens default to `outputs/<output_suffix>/hidden/`.  
- **Resumable:** fixed `outputs/{run_id}/` with `checkpoint.jsonl`; hiddens default to `outputs/{run_id}/hidden/`. If you set `hidden_capture_path` explicitly, that overrides the default.

See [Evaluation — Resumable](evaluation.md#resumable-evaluation) and [Resumable eval and CST](#resumable-eval-and-cst-same-resumable-flag) below.

**What is written**

- One PyTorch file per **lm-eval `generate_until` chunk**  
- Filenames include **model**, **scheduler** (short tag), **CST `k` and `w`** (from your CST YAML), **task name**, and chunk indices, for example:  
  `llada_scheduler_pd_k4096_w4_task_gsm8k_g00001_chunk_00000.pt`  
- Payload: dict with `hidden_states` (bfloat16 tensors on disk) and `meta` (chunk index, batch size, `cst_name_token`, `context`, etc.); loads cast to float32; see `cst.pre_logits.save_hidden_records`

Use these for analysis or to feed the **CST training** pipeline below.

---

## Checkpoint and staging names (training)

To avoid clobbering runs, **CST training checkpoints** and **collect staging** `.pt` files encode the LM directory name, **scheduler** short code (`train.scheduler_mode` in `cst.yaml`, e.g. `pd`), **`cst.k` and `cst.w`**, and (for collect) **epoch/segment/batch** indices.

**Training checkpoints** (under `train.checkpoint_dir`, default `outputs/cst_ckpt`):

- `cst_<model>_scheduler_<sched>_k<K>_w<W>_train_latest.pt` — resume / last write  
- `cst_..._train_best.pt` — best mean loss so far  
- `cst_..._train_state.json` — best-loss bookkeeping  

Checkpoint **`.pt`** files store float tensors in **bfloat16** to save space; loading promotes them to float32 for the CST module, optimizer, and ramp.

**Resume:** `train.resume_from_latest=true` loads the matching `*_train_latest.pt`, or `cst_latest.pt` if you still have that filename from an older run. You can set **`train.resume_checkpoint`** to an explicit path regardless of naming style.

**Staging files** (under `train.staging_dir`):  
`<same_token_prefix>_e{epoch}_seg{segment}_{batch:06d}.pt`

**Offline eval default path:** if `outputs/cst_ckpt/cst_best.pt` is missing, `launch_cst.py` can fall back to the newest `cst_*_train_best.pt` in that directory (see [Offline CST eval](#offline-cst-eval)).

---

## CST training (collect → train → checkpoint → delete chunks)

**Entry points**

- `./cst.sh` (wrapper)  
- `python launch_cst.py mode=train`  

**Config:** `config/cst.yaml` — `model_path`, `cst` (e.g. `d_model`, `k`, `w`, `max_steps`), and `train` (staging size per segment, `samples_per_epoch`, `num_epochs`, `scheduler_mode`, learning rate, W&B, etc.).

**Flow (simplified):**  
Sample prompts from `cst.train_data` (see below) → in each **epoch** / **segment**, run LM collection until `target_staging_gb` of hidden `.pt` is written → **train** CST on those files → **save** `latest` / maybe `best` → **delete** the segment’s chunk files → repeat.

**Example overrides**

```bash
python launch_cst.py mode=train train.target_staging_gb=0.5 cst.w=8
./cst.sh train.target_staging_gb=2.0
```

**Resume (YAML / CLI):** `train.resume_checkpoint=/path/to.pt` or `train.resume_from_latest=true` (and correct `train.checkpoint_dir`).

---

## lm-eval with a CST checkpoint

To **load** trained CST weights and run **per-chunk** metrics (and optional staging+flush like training), set on the **eval** side:

- `capture_pre_logits=true`  
- `hidden_capture_path=...` (or rely on auto paths)  
- `cst_checkpoint=/path/to/cst_<...>_train_best.pt` (match `k`/`w` to the YAML in `cst_config_path`)  
- `cst_config_path` if not default `config/cst.yaml`

**`eval.sh`** passes through CST-related flags; see `eval.sh --help` or the table in [Evaluation](evaluation.md).

**`launch_eval.py` example:**

```text
python launch_eval.py task_name=gsm8k cst_checkpoint=outputs/cst_ckpt/cst_llada_scheduler_pd_k4096_w4_train_best.pt capture_pre_logits=true
```

The **hidden file names** on disk use the **same** `k`/`w` as in your CST config YAML so they align with the checkpoint you expect.

### Resumable eval and CST (same `resumable` flag)

- **Yes — it is supported.** The harness uses one **`resumable`** flag (e.g. `./eval.sh --resumable` or `resumable=true` in [launch_eval / config](evaluation.md#resumable-evaluation)). `launch_eval.py` passes `resumable=true` and `output_dir=outputs/{run_id}/` together with `cst_checkpoint` / `capture_pre_logits` so hidden dumps live under **`outputs/{run_id}/hidden/`** next to **`checkpoint.jsonl`**.

- **What resumes:** The **`checkpoint.jsonl`** file stores **finished generations** (prompt + response) so those `(task_name, doc_id)` rows **skip the forward pass** on restart. That is the same resume mechanism as plain lm-eval (no extra CST flag).

- **What CST does on resume:** Hidden `.pt` files and CST **buffer / segment** processing run **only for chunks that still run the model** in this process. Chunks that are **fully** satisfied from `checkpoint.jsonl` do not write hiddens and do not re-run CST on them.

- **Persisted CST summary (resumable):** This is **not** the hidden **`.pt` tensors** — those are still **deleted** after each CST segment (eval then remove), same as before. We only persist a small **JSON** next to `checkpoint.jsonl`, with the same naming policy as hiddens: **`outputs/{run_id}/cst_<model>_scheduler_<sched>_k<K>_w<W>_task_<task>_eval_agg.json`** (e.g. `cst_llada_scheduler_pd_k4096_w4_task_gsm8k_eval_agg.json`). It stores the running lists of per-chunk **scalars** (e.g. mean NMSE / L0 / cos per file) that feed the final summary. On resume, that aggregate is **reloaded** so new segments **append**; you cannot reconstruct old hiddens from it. If the new-style file is missing, the same folder is checked for **`cst_eval_agg.json`**. If you change `cst_checkpoint` or config so the name token no longer matches, the JSON is **ignored** (with a log) to avoid mixing runs.

- **Limitation — staging buffer** (`cst_target_staging_gb`): if the process dies **before** a segment is fully evaluated and **persisted**, in-memory progress for that in-flight segment is lost. **After** a segment completes, the `*_eval_agg.json` file is updated so a resume keeps prior segments’ CST stats.

---

## Offline CST eval

Score a **directory of saved hidden `.pt` files** with a saved CST checkpoint (no full lm-eval loop):

```bash
python launch_cst.py mode=eval eval.hidden_dir=/path/to/hiddens eval.checkpoint=/path/to/cst_..._train_best.pt
```

Defaults merged in `launch_cst.py` if you omit an `eval:` block in `cst.yaml`. See also `./cst.sh --mode eval --hidden_dir ... --checkpoint ...`.

For **on-task** numbers with the same prompts as a benchmark, prefer **lm-eval + `cst_checkpoint`** (previous section) so `task_name` and data match.

---

## `cst.train_data` (mixed prompt pools)

Used by CST **training** to build row batches: math, code-style (TACO), multiple-choice, etc. — similar flavor to lm-eval tasks.

- **`load_pools(...)`** — one list of rows per source: `{id, source, text, meta}`  
- **`sample_uniform(n, ...)`** — `n` distinct examples, proportional split across pools, without replacement per pool, then shuffle  
- **`default_source_specs()`** / **`SourceSpec`** — defaults include `gsm8k`, `minerva_math`, `arc_*`, `commonsense_qa`, `taco`  

**Formatting notes:** GSM8K few-shot from public test; Minerva fixed 4-shot option; CommonsenseQA uses **validation** (HF test has no `answerKey`). See `cst/train_data/compose.py`.

**CLI**

```bash
python -m cst.train_data
python -m cst.train_data --samples 8
```

---

## Related files

| File | Role |
|------|------|
| `config/cst.yaml` | `launch_cst` train defaults, `cst` hyperparameters |
| `config/config.yaml` | lm-eval + `cst_checkpoint`, `capture_pre_logits` |
| `cst.sh` / `eval.sh` | Shell entry points |
| `launch_cst.py` | Hydra: `mode=train` or `mode=eval` |
| `launch_eval.py` | Hydra: lm-eval + CST flags |

See also: [Setup](setup.md), [Evaluation](evaluation.md), [Scheduler smoke tests](scheduler-testing.md).
