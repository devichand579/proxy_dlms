# LLaDA-PCFG

LLaDA-PCFG is a controlled synthetic domain inside Strace for training and evaluating small **LLaDA** (masked diffusion) models on **PCFG-generated code-like strings**. Because the data comes from a known probabilistic grammar, you can compute **exact token marginals** \(q(x_p \mid \text{visible tokens})\) via a Rust CKY inside/outside engine — a ground-truth oracle for scheduler research and model calibration.

---

## Why it exists

| Problem in natural-language eval | LLaDA-PCFG answer |
| -------------------------------- | ---------------- |
| Unknown true conditional distributions | Exact \(q\) from the PCFG engine |
| Expensive / noisy ground truth | Fast Rust CKY with incremental `PCFGState` |
| Hard to isolate scheduler errors | Infinite synthetic corpus with fixed syntax |
| Large-model training cost | 100M–300M LLaDA variants on 51-token vocab |

Use LLaDA-PCFG to prototype schedulers, measure KL to ground truth, and train small diffusion LMs before scaling to GSM8K / HumanEval runs in the main Strace harness.

---

## Layout

```
Strace/
├── scripts/                  # bash launchers (eval, cst, tag, llada-pcfg, …)
├── hydra_runners/            # Hydra entry points (CST, TAG, eval, llada-pcfg)
├── tests/                    # scheduler + tracing smoke tests
├── checkpoints/
├── data/pcfg/
├── eval_outputs/pcfg/
└── llada-pcfg/
    ├── scripts/              # PCFG Python CLIs (train, eval, generate)
    └── …
```

---

## Prerequisites

| Requirement | Why |
| ----------- | --- |
| Python ≥ 3.10 | `pyproject.toml` constraint |
| `numpy`, `datasets` | Corpus I/O |
| `maturin` + Rust toolchain | Build `pcfg_engine_rs` |
| `torch` (optional) | Training only (`pip install -e ".[train]"` from `llada-pcfg/`) |
| `HF_TOKEN` | Hub push/load (set in `.env` or via `repo_env_defaults`) |

Run scripts from the repo root via `./scripts/llada-pcfg.sh` or `llada-pcfg/scripts/` (Python CLIs).

---

## Setup

### Python dependencies

```bash
# from repo root
pip install -e .
pip install -e ".[train]"   # for training
pip install -e ".[hub]"     # for Hub push (huggingface_hub)
```

### Rust extension (`pcfg_engine_rs`)

`pcfg_engine_rs` is the native extension (PyO3 bindings) with two components:

| Component | Python API | Role |
| --------- | ---------- | ---- |
| CKY inside/outside | `pcfg_engine.PCFGEngine` | Exact marginals \(q(x_p \mid \text{visible})\), `Z`, `Mt`, `Jt` |
| Parallel sampler | `pcfg_grammar.Sampler` | Corpus generation via `PyPCFGSampler` (rayon-parallel) |

Build from `llada-pcfg/`:

```bash
# from repo root
pip install maturin
maturin develop --release
```

Verify:

```bash
python -c "from pcfg_engine import PCFGEngine; from pcfg_grammar import Sampler; print('ok')"
```

Usage:

```python
from pcfg_engine import PCFGEngine          # CKY marginals
from pcfg_grammar import Sampler, code_grammar   # corpus generation
from pcfg_engine_rs import PyPCFGSampler    # low-level Rust API

g = code_grammar()
engine = PCFGEngine(g.to_cnf())
sampler = Sampler(g, workers=8)
batch = sampler.sample_batch(1000, 64, 128, seed=0)
```

CKY runs entirely in Rust. Corpus sampling uses `PyPCFGSampler` (rayon-parallel threads).

---

## Grammar

`code_grammar()` in `pcfg_grammar.py` defines a C-like PCFG with ~48 terminals (`int`, `if`, `while`, `id`, `num`, `+`, …) and nonterminals (`Prog`, `Stmt`, `Expr`, `Cond`, …).

| Property | Value |
| -------- | ----- |
| String length range | 64–128 tokens |
| Sequence architecture length | 128 (`ARCH_LEN`) |
| Vocab size | 51 (`<pad>`, `<mask>`, `<eos>` + terminals) |

The `Grammar` class defines `code_grammar()` rules. `to_cnf()` converts automatically (fresh start, terminal lifting, binarization, unit-rule elimination) so the CKY engine can consume the grammar. Corpus sampling is implemented in Rust via `Sampler` in the same module.

---

## Data generation

`pcfg_generate.py` samples strings until a train token budget is met, optionally holds out a fixed number of test strings (disjoint shuffle split), and saves a HuggingFace `DatasetDict`.

### Local only

```bash
# from repo root
python llada-pcfg/scripts/pcfg_generate.py --train-tokens 500000 --test-strings 512 --out-dir data/pcfg
```

### Push to HuggingFace Hub

```bash
# Short name → resolves to <namespace>/pcfg-code via whoami or env
python llada-pcfg/scripts/pcfg_generate.py --train-tokens 1000000 --test-strings 1000 --dataset pcfg-code --push

# Full repo id
python llada-pcfg/scripts/pcfg_generate.py --train-tokens 1000000 --test-strings 1000 --repo-id devichand/pcfg-code --push

# Explicit namespace
python llada-pcfg/scripts/pcfg_generate.py --train-tokens 500000 --dataset pcfg-code --namespace devichand --push
```

### Fast generation (Rust sampler)

Build `pcfg_engine_rs`, then generate with parallel Rust sampling (default):

```bash
python llada-pcfg/scripts/pcfg_generate.py --train-tokens 100000000 --test-strings 512 \
  --workers 8 --dataset pcfg-code --push
```

Observed throughput: **~3–4M tokens/sec**. 100M tokens ≈ **30–60 seconds** of sampling.

### Generation flags

| Flag | Description |
| ---- | ----------- |
| `--train-tokens` | Train split token budget (after test holdout) |
| `--test-strings` | Strings held out from pool for test |
| `--min-len`, `--max-len` | Length range (default 64–128) |
| `--seed` | Sampling seed |
| `--workers` | Rust sampler threads (default: CPU count, max 8) |
| `--progress-interval` | Log every N pool tokens (default 250000; 0 to disable) |
| `--out-dir` | Local `save_to_disk` path (default `data/pcfg_corpus`) |
| `--no-save` | Skip local save |
| `--dataset` | Short Hub dataset name (default `pcfg-code`) |
| `--repo-id` | Full `org/repo` (overrides `--dataset`) |
| `--namespace` | HF org when using a short name |
| `--push` | Push to Hub after generation |
| `--private` | Create private dataset repo |
| `--capacity` | Print grammar capacity notes before sampling |

### Dataset schema

Each split has rows with:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `id` | int | Row index |
| `tokens` | list[str] | Grammar terminal sequence |
| `length` | int | `len(tokens)` |
| `text` | str | Space-joined tokens |

`generation_meta.json` is written alongside the local export with split stats and sampling parameters.

---

## HuggingFace integration

`hf_hub.py` follows the same conventions as trace uploads and CST hidden storage:

- `apply_repo_env_defaults()` — sets `HF_TOKEN` from `.env` / repo defaults
- `normalize_dataset_repo_id()` — resolves short names to `org/repo`
- `ensure_dataset_repo()` — creates the dataset repo before push

### Environment variables

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `HF_TOKEN` | from `.env` | Hub authentication |
| `PCFG_HF_DATASET` | `pcfg-code` | Default short dataset name |
| `PCFG_HF_NAMESPACE` | — | Default org (else `TRACE_HF_NAMESPACE` / whoami) |

### Load corpus in Python

```python
from pcfg_data import load_corpus_from_hub

train, test, ddict = load_corpus_from_hub("pcfg-code")
# or
train, test, ddict = load_corpus_from_hub("devichand/pcfg-code")
```

---

## Training

`train_pcfg_llada.py` trains small LLaDA variants with **independent masking** (LLaDA Eq. 3):

1. Draw \(t \sim \text{Uniform}[0, 1]\) per example
2. Mask each content token independently with probability \(t\)
3. Optimize CE on masked positions with weight \(1 / (t \cdot L)\)

### Model variants

| Variant | d_model | layers | ~params |
| ------- | ------- | ------ | ------- |
| `100m` | 768 | 12 | 85M |
| `200m` | 1024 | 16 | 218M |
| `300m` | 1024 | 24 | 327M |

Load programmatically:

```python
from models.pcfg_models import load_pcfg_llada, load_pcfg_tokenizer

model, config, vocab = load_pcfg_llada("100m")
tokenizer = load_pcfg_tokenizer()
```

### Train from local corpus

```bash
# from repo root
python llada-pcfg/scripts/train_pcfg_llada.py --variant 100m --corpus data/pcfg --epochs 1 --steps 200
```

### Train from Hub

```bash
python llada-pcfg/scripts/train_pcfg_llada.py --variant 100m --hf-dataset pcfg-code --steps 200
```

### Train on-the-fly sampling

```bash
python llada-pcfg/scripts/train_pcfg_llada.py --variant 100m --train-tokens 500000 --steps 200
```

### Training flags

| Flag | Description |
| ---- | ----------- |
| `--variant` | `100m` / `200m` / `300m` |
| `--corpus` | Local `save_to_disk` path from `pcfg_generate` |
| `--hf-dataset` | Hub dataset (short name or `org/repo`) |
| `--hf-namespace` | Org override for short `--hf-dataset` |
| `--train-tokens` | Sample corpus by token budget |
| `--batch-size` | Per-GPU batch size (default 16) |
| `--grad-accum` | Gradient accumulation steps (default 1) |
| `--num-workers` | DataLoader workers per rank (default 2) |
| `--peak-lr`, `--mid-lr`, `--min-lr` | WSD schedule (paper defaults) |
| `--warmup-steps` | Explicit warmup (overrides fraction) |
| `--warmup-fraction` | Auto warmup = fraction of total steps (default 0.02) |
| `--warmup-max` | Cap auto warmup (default 5000) |
| `--steps` | Total optimizer steps |
| `--epochs` | Epochs if `--steps` omitted |
| `--log-every` | Print/log every N steps (default 10) |
| `--save-every` | Checkpoint every N steps (default 5000) |
| `--resume` | Resume from checkpoint directory |
| `--wandb` / `--no-wandb` | W&B logging (default on) |
| `--wandb-project` | W&B project (default `WANDB_PROJECT` / `strace`) |
| `--wandb-run-name` | W&B run name (default auto from variant + hparams) |
| `--wandb-id` | Explicit W&B run id for deterministic resume |
| `--out-dir` | Checkpoint output (default `checkpoints/llada-<variant>-pcfg`) |
| `--val-fraction` | Random train fraction for validation (default 0.05) |
| `--val-every` | Validation every N steps (default 2000; 0=off) |
| `--push-model` | Push `checkpoint-best_val` to HF Hub after training |
| `--hf-model` | HF model repo (default `llada-<variant>-pcfg`) |
| `--hf-model-namespace` | Org override for `--hf-model` |
| `--hf-model-private` | Create private model repo |
| `--amp` / `--no-amp` | Mixed precision on CUDA (default on) |
| `--amp-dtype` | `bfloat16` or `float16` |

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=2 llada-pcfg/scripts/train_pcfg_llada.py \
  --variant 100m --corpus data/pcfg --steps 5000 --batch-size 16
```

Effective batch size = `batch_size × num_gpus × grad_accum`.

For ~1M samples at batch size 128, one epoch is ~7.8k optimizer steps. Defaults (`--val-every 2000`, `--save-every 5000`) give a few validation passes and roughly one rolling checkpoint per epoch without excessive I/O. Warmup defaults to 2% of total steps (capped at 5000).

```bash
torchrun --nproc_per_node=4 llada-pcfg/scripts/train_pcfg_llada.py \
  --variant 100m --hf-dataset pcfg-code --batch-size 128 --epochs 3
```

### Hydra launcher and Slurm (`scripts/llada-pcfg.sh`)

Config defaults live in **`config/llada-pcfg.yaml`**. The shell script exposes flags that **override** config only when set; omitted flags keep YAML defaults.

```bash
# Train — only override what you need; rest from config (batch_size=128, val_every=2000, ...)
./scripts/llada-pcfg.sh --mode train --variant 100m --hf_dataset pcfg-code --epochs 3

# Eval
./scripts/llada-pcfg.sh --mode eval --model_path checkpoints/llada-100m-pcfg/checkpoint-best_val --schedulers all

# Slurm
sbatch scripts/llada-pcfg.sh --mode train --hf_dataset pcfg-code --nproc 4 --slurm

# Hydra multirun ablations (login node → one SLURM job per combo)
python hydra_runners/run_llada-pcfg.py mode=train train.variant=100m,200m train.peak_lr=4e-4,1e-4 \
  --multirun launcher=slurm_h100
```

Run `./scripts/llada-pcfg.sh --help` for the full flag list (train hparams, eval schedulers, W&B, tracing, etc.). Advanced overrides can still be appended as `key=value`, e.g. `train.mid_lr=2e-4`.

| Layer | Role |
| ----- | ---- |
| `config/llada-pcfg.yaml` | Default hyperparameters for train and eval |
| `scripts/llada-pcfg.sh --flag` | Override specific keys without repeating the full config |
| `hydra_runners/run_llada-pcfg.py` | Hydra entry; maps YAML → `llada-pcfg/scripts/` train and eval CLIs |

Edit `#SBATCH` headers in `scripts/llada-pcfg.sh` for partition, memory, and GPU count (same pattern as `scripts/eval.sh` / `scripts/cst.sh`).

### W&B

Uses repo defaults (`WANDB_API_KEY`, `WANDB_PROJECT` from `.env` / `repo_env_defaults`).

Logged per step: `train/loss`, `train/ppl`, `train/lr`, `val/loss`, `val/ppl` (when validation enabled).

Default run name encodes ablation-relevant hparams, e.g. `pcfg-100m-bs128-s50000-lr4e-4-pcfg-code` (variant, batch, steps/epochs, LR, data source). Override with `--wandb-run-name`.

Checkpoints: only `checkpoint-latest/` (replaced each save) and `checkpoint-best_val/` (replaced when val improves), plus `train_state.json` and RNG states for deterministic resume. Previous `latest` / `best_val` dirs are deleted before writing new ones.

### HF model push

```bash
python llada-pcfg/scripts/train_pcfg_llada.py --variant 100m --hf-dataset pcfg-code --steps 5000 \
  --push-model --hf-model llada-100m-pcfg
```

Pushes `checkpoint-best_val` to `https://huggingface.co/<namespace>/llada-100m-pcfg` (requires `HF_TOKEN` and `pip install -e ".[train]"`).

Checkpoints include model weights, config, tokenizer, and `pcfg_vocab.txt`.

---

## Evaluation metrics

| Phase | Metric | Meaning |
| ----- | ------ | ------- |
| Train / val | `masked_ce_loss`, `masked_ppl` | LLaDA Eq. (3) on independently masked tokens |
| Suffix eval | `exact_match` | Generated suffix (content+eos) matches reference |
| Suffix eval | `structural_accuracy` | Syntax token match rate on suffix (excludes `id`/`fld`/`num`/`flt`) |
| Traces (SchedulerLens) | `M_hat`, `J_hat`, oracle `Mt`/`Jt` | Post-hoc via `schedulerlens.run … --replay-oracle` |

Validation holds out a **random 5%** of the train corpus (`--val-fraction`, default 0.05) with deterministic masking per example index. Validation is **masked CE loss only** — schedulers are not run during training. Best checkpoint by validation loss: `checkpoint-best_val/`.

---

## Scheduler eval (8 schedulers + tracing)

After training, evaluate on the test split with all eight Strace schedulers (`pd`, `bd`, `top_k_c`, `top_k_m`, `fhs`, `eb`, `fdm`, `dus`):

```bash
# Via Hydra launcher (repo root)
./scripts/llada-pcfg.sh --mode eval eval.hf_model=username/llada-100m-pcfg eval.schedulers=all

# Direct script (from repo root)
python llada-pcfg/scripts/pcfg_eval.py \
  --model-path checkpoints/llada-100m-pcfg/checkpoint-best_val \
  --hf-dataset pcfg-code --schedulers all --steps 128 --resumable
```

Load from HuggingFace Hub:

```bash
python llada-pcfg/scripts/pcfg_eval.py \
  --hf-model username/llada-100m-pcfg \
  --hf-dataset pcfg-code --schedulers pd,bd,eb --resumable
```

**EOS harness:** `eval.use_eos_harness` defaults to `true` (same as main `eval.py`): schedulers may stop early once EOS is decoded with no masks before it. Use `--no-use-eos-harness` or `eval.use_eos_harness=false` for full-step reference-style generation.

**Harness parity:** prompts are **left-padded** (like `eval.py`); suffix metrics truncate through the first EOS when the harness is on; `eval.top_k` defaults to `2` for top_k schedulers; padded output vocab ids (`vocab_size..embedding_size-1`) are masked for FHS.

Outputs (per scheduler, `run_name = {model}-pcfg_{scheduler}_s{steps}`):
- `eval_outputs/pcfg/pcfg_eval_{model}.json` — aggregate summary
- `eval_outputs/pcfg/{run_name}/checkpoint.jsonl` — resumable sample checkpoint (when `--resumable`)
- `eval_outputs/pcfg/{run_name}/trace_outputs/{run_name}.jsonl` — traces for SchedulerLens

Analyze traces (proxies + perplexity + oracle Mt/Jt) — single CLI:

```bash
# Local NPZ
python -m schedulerlens.run file \
  eval_outputs/pcfg/llada-100m-pcfg-pcfg_pd_s128/trace_outputs/llada-100m-pcfg_pd_s128.npz \
  -o outputs/schedulerlens/pcfg_pd --replay-oracle --model-path /path/to/llada-100m-pcfg

# Hub: inventory → download→analyze→delete → CSV + plots
python -m schedulerlens.run inventory --repo schedulerlens-npz --pcfg-only
python -m schedulerlens.run compare \
  --repo schedulerlens-npz --pattern '*-pcfg_*.npz' \
  -o outputs/schedulerlens/pcfg_compare \
  --replay-oracle --model-path /path/to/llada-100m-pcfg
```

`perplexity.json` / `schedulerlens_metrics.csv` include `model_commit_ppl` and `oracle_ppl`. With `--replay-oracle` + `--model-path`, exact `Mt`/`Jt` and M↔M̂ / J↔Ĵ correlations (full / early / mid / late) are written. See [schedulerlens.md](schedulerlens.md).

PCFG metrics are computed by `pcfg_eval.py` and `schedulerlens` — not the lm-eval trace audit pipeline. PCFG traces (`pcfg_*` / `*-pcfg_*`) get hook/structure checks in `trace_audit.py`. **Sample count** is compared to the **`pcfg-code` Hub test split** (override with `--pcfg-dataset` / `--pcfg-namespace`; lm-eval / W&B accuracy is still skipped).

---

## PCFG engine (ground truth)

The Rust engine (`pcfg_engine_rs`) runs CKY inside/outside on the CNF grammar. Python access is via `pcfg_engine.py`.

### Build the stack

```python
from pcfg_data import build_stack, infer_marginals

g, cnf, engine, vocab = build_stack()
```

### Exact marginals

```python
tokens = [...]  # grammar string, len 64–127
ex = infer_marginals(engine, tokens, n_visible=32, vocab=vocab)

# ex.marginals[pos] → np.ndarray over terminals (sums to 1)
for pos, dist in ex.marginals.items():
    print(pos, dist.argmax(), dist.max())
```

### Key engine methods

| Method | Computes |
| ------ | -------- |
| `Z(L, committed)` | Partition function (parse probability) |
| `marginals_batch(L, committed, positions)` | \(q(x_p \mid \text{committed})\) per position |
| `all_marginals(L, committed)` | All positions at once |
| `Jt_entropy(L, committed, S)` | Joint entropy over masked set \(S\) |
| `Mt(L, committed, S, model_fn)` | KL divergence vs model predictions |
| `make_state` / `PCFGState.commit` | Incremental chart updates for scheduler simulation |

### Benchmarks

```bash
# from repo root
python pcfg_validate.py
python pcfg_validate.py --lengths 32 64 128 --jt-sizes 3 4 5
```

---

## End-to-end workflow

```mermaid
flowchart LR
    G[code_grammar] -->|sample| GEN[pcfg_generate.py]
    GEN -->|HF dataset| HUB[(HuggingFace Hub)]
    GEN -->|local disk| DISK[data/pcfg]
    HUB --> TRAIN[train_pcfg_llada.py]
    DISK --> TRAIN
    G -->|to_cnf| ENG[pcfg_engine_rs]
    TRAIN --> MODEL[LLaDA 100M/200M/300M]
    ENG -->|exact q| EVAL[scheduler / KL eval]
    MODEL -.->|compare| EVAL
```

1. **Generate** — `pcfg_generate.py --train-tokens … --dataset pcfg-code --push`
2. **Train** — `train_pcfg_llada.py --variant 100m --hf-dataset pcfg-code`
3. **Evaluate** — compare model logits to `infer_marginals()` at masked positions; plug `PCFGState` into scheduler unmasking loops

---

## Relation to Strace

| Strace component | LLaDA-PCFG role |
| ---------------- | -------------- |
| `schedulers/` | Test unmasking order against exact \(q\) |
| `schedulerlens/` | M̂/Ĵ metrics with known ground-truth marginals |
| `traces/` | Same HF conventions for dataset push/load |
| `models/llada/` | Full-scale 8B model; LLaDA-PCFG uses small PCFG-tuned copies |

LLaDA-PCFG is the **synthetic sandbox**: infinite known data, exact conditionals, small fast models — before running the full eval harness on real benchmarks.

---

## Troubleshooting

| Issue | Fix |
| ----- | --- |
| `ImportError: pcfg_engine_rs is required` | Run `maturin develop --release` from `llada-pcfg/` |
| `no namespace` on Hub push | Set `--namespace`, `PCFG_HF_NAMESPACE`, or `huggingface-cli login` |
| `content length must be < arch_len` | Grammar strings must be ≤ 127 tokens (room for `<eos>`) |
| Training loss uses wrong formula | Do not use HF `labels` in `LLaDAModelLM.forward`; use `llada_mask_loss` in the training script |
| Slow marginal queries | Use `marginals_batch` or incremental `PCFGState` instead of recomputing full charts |
