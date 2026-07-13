# SchedulerLens

SchedulerLens analyzes **masked diffusion schedulers** from generation traces produced by the eval harness. It reports M̂/Ĵ caps and their sub-proxies, each with **entropy-weighted** and **uniform-step** aggregates plus per-step trends.

| Proxy | Role | Lower is better? |
| ----- | ---- | ---------------- |
| **M̂_t** (`M_hat_t`) | Ordering quality | Yes |
| **mean_regret** | Post-commit regret over `C_t` | Yes |
| **cascade_gain** | Confidence lift on still-masked positions | Context-dependent |
| **Ĵ_t** (`J_hat_t`) | Parallelism error | Yes |
| **instability** | Argmax flips on `S_t` | Yes |
| **rank_volatility** | Confidence rank shuffling over gen region | Yes |

Top-level `M_hat` / `J_hat` are aliases for entropy-weighted `M_hat_t` / `J_hat_t` caps.

---

## Prerequisites

| Requirement | Why |
| ----------- | --- |
| **Trace file** (`.jsonl` or `.json`) | One sample per line (JSONL) or a single trace / `samples` list (JSON) |
| **Gen-only step fields** | `mask_index`, `predictions`, `confidences`, `entropies`, `selected_positions`, `selected_tokens` |

Traces come from `GenerationTrace` in `traces/trace_generator.py`, usually under `trace_outputs/` after eval with tracing enabled. See [evaluation.md](evaluation.md).

---

## Quick start

```bash
python -m schedulerlens.run trace_outputs/your_run.jsonl -o outputs/schedulerlens/run1
python -m schedulerlens.run trace_outputs/your_run.jsonl -o outputs/schedulerlens/run1 --bucket-size 16
```

Flags:

| Flag | Description |
| ---- | ----------- |
| `-o`, `--output-dir` | Output directory (required) |
| `--max-samples N` | Process only the first N samples |
| `--bucket-size N` | Bucket steps in groups of N; adds bucket trends + plots |
| `-q`, `--quiet` | Less logging |

---

## Output

Written to `--output-dir`:

| File | Contents |
| ---- | -------- |
| `metrics.json` | Caps and per-step trends under `proxies` |
| `{stem}_M_trends.png` | M̂_t (main), mean regret, cascade gain |
| `{stem}_J_trends.png` | Ĵ_t (main), instability, rank volatility |
| `{stem}_unmasked_count_trend.png` | Mean tokens unmasked per step |

Each M/J PNG has three stacked subplots: **main proxy on top** (bold), two sub-proxies below.

With `--bucket-size N`:

| File | Contents |
| ---- | -------- |
| `{stem}_bucket{N}_all_trends.png` | All 6 proxy trends, one subplot each |
| `{stem}_bucket{N}_unmasked.png` | Mean tokens unmasked per bucket |

Bucket value at index `b` = mean over steps `[b*N, (b+1)*N)` (per sample, then mean across samples).

When `--bucket-size` is set, the CLI summary prints a table of bucket values after the main caps.

### `metrics.json` structure

Each entry under `proxies` has:

- `entropy`, `uniform` — task-level caps (mean over samples)
- `trend` — per-step mean across samples
- `bucket_trend` — present when `--bucket-size N` is set
- `sample_entropy`, `sample_uniform` — per-sample caps

Proxy names: `M_hat_t`, `mean_regret`, `cascade_gain`, `J_hat_t`, `instability`, `rank_volatility`.

Top-level keys: `M_hat`, `J_hat`, `M_hat_uniform`, `J_hat_uniform`, `n_samples`, `n_steps`, `n_unmasked_trend`, `mean_unmasked_per_step`, and `proxies`.

---

## Metrics

### Per-step notation (gen region only)

| Symbol | Trace field | Meaning |
| ------ | ----------- | ------- |
| `S_t` | `selected_positions` | Unmasked at step `t` |
| `C_t` | `~mask_index` at step start | Committed before step `t` |
| `S̄_t` | `mask_index & ~selected_positions` | Still masked after step `t` |
| `w_t` | sum of `entropies[S_t]` | Entropy-drop weight |

### M-side

1. **mean_regret** — mean positive post-commit regret over `C_t`.
2. **cascade_gain** — mean Δ confidence on `S̄_t` from step `t-1` to `t`.
3. **M̂_t** = mean_regret / `(1 + max(0, cascade_gain))`.

### J-side

1. **instability** — fraction of `S_t` whose argmax changed since `t-1`.
2. **rank_volatility** — `1 - (τ + 1) / 2` over all gen positions (Kendall τ of confidence ranks).
3. **Ĵ_t** = instability × rank_volatility.

### Aggregation

| Level | Entropy-weighted | Uniform-step |
| ----- | ---------------- | ------------ |
| Sample cap | `sum(w_t * x_t) / sum(w_t)` | `mean(x_t)` |
| Task cap | mean over samples | mean over samples |
| Trend at step `t` | mean of raw `x_t` across samples | same (step-level, not re-weighted) |

Every proxy series (`x_t`) gets both cap variants and a trend.

---

## Python API

```python
from schedulerlens import compute_task_metrics, save_trend_plots
from schedulerlens.run import run_schedulerlens

result = run_schedulerlens("trace_outputs/run.jsonl", "outputs/schedulerlens/run1")
print(result["M_t_plot"], result["J_t_plot"])

metrics = compute_task_metrics("trace_outputs/run.jsonl")
print(metrics.summary())
regret = metrics.proxies["mean_regret"]
print(regret.entropy, regret.uniform, regret.trend)
save_trend_plots(metrics, "outputs/schedulerlens/run1", prefix="run")
```

---

## Module layout

| Module | Role |
| ------ | ---- |
| `schedulerlens/proxies.py` | Per-step proxies and sample caps |
| `schedulerlens/metrics.py` | Trace I/O and task-level aggregation |
| `schedulerlens/plots.py` | Trend PNGs |
| `schedulerlens/run.py` | CLI and `run_schedulerlens()` |
| `schedulerlens/pos_analysis.py` | POS tagging and unmask timing |
| `schedulerlens/pos_plots.py` | POS analysis plots |
| `schedulerlens/pos_run.py` | POS analysis CLI |
| `schedulerlens/tokenizer_utils.py` | Model/tokenizer resolution from trace path |

---

## Notes

- All arrays are **gen-only** (`gen_length`); prompt tokens are excluded.
- Samples that fail during proxy computation are skipped with a warning.
- Shorter traces count equally in task caps; late trend steps only include samples that ran that long.

---

## POS unmasking analysis

Separate from M̂/Ĵ proxies: tag unmasked tokens with NLTK POS, report when each tag is committed, and list top-k tokens per step. No ρ_sem metric.

```bash
python -m schedulerlens.pos_run trace_outputs/llada_humaneval_pure_diffusion_s256.jsonl -o outputs/schedulerlens/pos1
python -m schedulerlens.pos_run trace_outputs/llada_humaneval_pure_diffusion_s256.jsonl -o outputs/schedulerlens/pos1 --bucket-size 16 --top-k 10
```

| Flag | Description |
| ---- | ----------- |
| `--top-k N` | Top N (token, POS) pairs per step by count across samples (default 10) |
| `--model-path DIR` | Override tokenizer model dir (default: infer `llada` / `llada1_5` / `dream` from trace filename stem) |
| `--models-root DIR` | Root containing model subdirs (default `./models`) |

**Tokenizer:** Trace basenames follow `{model}_{task}_{mode}_s{steps}` from `hydra_runners/launch_eval.py`. The CLI loads `AutoTokenizer` from `./models/{model}` via `load_model.load_tokenizer`.

**Unmask events:** Uses `selected_positions` / `selected_tokens` when present; otherwise reconstructs from consecutive `mask_index` diffs (same idea as `visualization_json.track_unmasking_order`).

### Output

| File | Contents |
| ---- | -------- |
| `pos_metrics.json` | Mean normalized unmask step by POS, per-step POS trends, top-k tokens per step, `mean_unmask_step_by_pos_bucket` when bucketed |
| `{stem}_pos_mean_unmask_step.png` | Bar chart: mean t/T by POS tag |
| `{stem}_pos_bucket{N}_mean_unmask_step.png` | With `--bucket-size N`: per-bucket bar panels (8 earliest POS tags by mean step within each bucket) |
| `{stem}_pos_unmask_count_trend.png` | Top POS tags unmasked per step |
| `{stem}_pos_bucket{N}_unmask_trend.png` | With `--bucket-size N`: unmask counts per bucket |

### Python API

```python
from schedulerlens import compute_pos_metrics, save_pos_plots

metrics = compute_pos_metrics("trace_outputs/run.jsonl", bucket_size=16, top_k=10)
print(metrics.summary())
save_pos_plots(metrics, "outputs/schedulerlens/pos1", prefix="run")
```
