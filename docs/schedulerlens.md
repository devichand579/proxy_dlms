# SchedulerLens

SchedulerLens analyzes **masked diffusion schedulers** from generation traces. It reports M̂ (γ = 1) and Ĵ caps, completion diagnostics, optional strace accuracy/NFE join, commit / oracle perplexity, and PCFG oracle–proxy correlations.

| Proxy | Definition | Cap |
| ----- | ---------- | --- |
| **M̂_t** (`m_hat`) | lowest-entropy prefix of \(S_t\) with \(\sum H\le\gamma\), \(\gamma=1\); sum of \(-\log c\) on that prefix | **M̂** = \(\sum_t\) M̂_t |
| **Ĵ_t** (`j_hat`) | \(\sum H-\max H\) on \(S_t\) if \(\|S_t\|\ge 2\), else 0 | **Ĵ** = \(\sum_t\) Ĵ_t |
| **Ĝ** (`G_hat`) | — | M̂ + Ĵ |

Sub-proxies (mean_regret, …) are removed.

---

## Prerequisites

| Requirement | Why |
| ----------- | --- |
| **Trace archive** (`.npz` only) | From `traces/trace_npz.py` (convert JSONL via `trace_audit.py convert`) |
| **Gen-only step fields** | `mask_index`, `predictions`, `confidences`, `entropies`; `selected_positions` / `selected_tokens` or inferrable from consecutive masks |

---

## Fail-hard rules

- Missing / inconsistent required fields → **`TraceProxyError`** (no silent zeros, no skipping samples *within* a file).
- Single-file and batch commands abort on **`TraceProxyError`** (fail-hard).
- Proxies and oracle call **`validate_trace_for_metrics`** first. They do **not** silently repair. Unrepaired issues raise hard errors, including:
  - missing `mask_index` / `predictions` / `confidences` / `entropies` (or wrong lengths)
  - `selected_positions` vs mask-diff set/count mismatch (DUS remask / incomplete hooks)
  - missing `selected_*` when mask-diff has unmasks
  - FDM **phantom extra steps** (nested hyp forwards)
  - **untrimmed trailing step** (`schedulerlens_trimmed_tail` missing) — repair drops the last step for **all** schedulers
- On the **last kept step** after trim, `selected_*` is trusted (no next mask to compare).
- Empty selection on non-last steps is valid only when that step’s mask-diff is also empty.
- Repair traces first (`trace_audit.py convert` / `repair-npz`), then run SchedulerLens.
- When **no tokens are unmasked** at a step (`n_selected == 0`), **M̂_t = Ĵ_t = 0**.

---

## CLI (single entrypoint)

All commands go through `python -m schedulerlens.run` (or `python -m schedulerlens`):

| Subcommand | Role |
| ---------- | ---- |
| `file` | One local NPZ → metrics + plots |
| `inventory` | List Hub NPZs (no download) |
| `compare` | Batch Hub/local → CSV |
| `prefix-sweep` | M̂, Ĵ, and Ĝ/f at each prefix for each run |
| `prefix-sweep-pcfg` | Same sweep on PCFG traces; sample Pearson of M̂ with Mt, Ĵ with Jt, and Ĝ with G |
| `prefix-selection` | Finish the scheduler with the lowest prefix Ĝ/f |

`compare` writes the metrics sheet. `prefix-sweep`, `prefix-sweep-pcfg`, and `prefix-selection` read that sheet or the traces behind it. All four run from `scripts/schedulerlens_compare.sh` via `--command`. `--exclude-task` and `--exclude-scheduler` skip comma-separated names on every command (for example `--exclude-task arc_challenge_chat`). `causal` and `plots` are separate.

```bash
# Local NPZ
python -m schedulerlens.run file trace_outputs/your_run.npz -o outputs/schedulerlens/run1
python -m schedulerlens.run file path/to/llada-100m-pcfg_pd_s128.npz -o out/ \
  --replay-oracle --model-path /path/to/pcfg/ckpt

# Hub inventory / batch
python -m schedulerlens.run inventory --repo schedulerlens-npz --pcfg-only
# Lazy-load each stem's model from HF (llada-100m-pcfg-pcfg_bd_s128 → org/llada-100m-pcfg),
# unload previous weights when the model changes:
python -m schedulerlens.run compare \
  --repo llada_pcfg_traces --namespace YOUR_ORG --pattern '*-pcfg_*.npz' --pcfg-only \
  -o outputs/schedulerlens/pcfg_compare \
  --replay-oracle
# (--namespace applies to both the dataset repo and lazy oracle models:
#  stem llada-100m-pcfg-pcfg_bd_s128 → YOUR_ORG/llada-100m-pcfg)
# Omit --namespace to use TRACE_HF_NAMESPACE / HF whoami (token from repo defaults).

# Or pin one model for all traces:
#   --replay-oracle --hf-model YOUR_ORG/llada-300m-pcfg
# Local checkpoint:
#   --replay-oracle --model-path checkpoints/llada-300m-pcfg-1B/checkpoint-best_val

# Offline batch
python -m schedulerlens.run compare --local-dir ./npz_traces -o out/compare --pcfg-only

# Reuse saved oracle.npz (folder or single file) + Hub traces → full CSV (no GPU replay)
python -m schedulerlens.run compare \
  --repo llada_pcfg_traces --namespace YOUR_ORG --pattern '*-pcfg_*.npz' --pcfg-only \
  --oracle outputs/pcfg_oracles -o outputs/schedulerlens/pcfg_compare
```

Hub helpers live in **`schedulerlens.hub`** (library): inventory → temp download → stream NPZ samples → metrics → **delete temp**. Batch jobs purge stale HF cache at start.

| Flag | Description |
| ---- | ----------- |
| `-o` | Output directory |
| `--max-samples N` | First N samples only |
| `--bucket-size N` | Bucketed trends + plots |
| `--no-perplexity` | Skip PPL (`file` only) |
| `--replay-oracle` | Exact Mt/Jt (PCFG; needs `--model-path`) |
| `--pcfg` | Force PCFG oracle gating |
| `--model-path DIR` | PCFG checkpoint for Mt/Jt replay |

### Outputs (`file`)

| File | Contents |
| ---- | -------- |
| `metrics.json` | Caps, trends, meta, optional perplexity / correlations |
| `metrics.csv` | One unified row (same schema as compare) |
| `perplexity.json` | Model commit / oracle PPL (and Mt/Jt when replayed) |
| `{stem}_M_trends.png` / `_J_trends.png` | Per-step proxy panels |
| `compare_surfaces.png` | Step × token heatmaps (confidence, entropy) |

---

## Hub / compare flags

| Flag | Description |
| ---- | ----------- |
| `--repo` / `--local-dir` | Source (mutually exclusive). `-o` defaults to `outputs/schedulerlens/compare` |
| `--pattern` | Filename glob against Hub/local names (default `*.npz`). Example: `*-pcfg_*.npz` keeps only PCFG-looking files by name |
| `--pcfg-only` | After discovery, keep only stems parsed as PCFG (`-pcfg_` / `pcfg_` in the name). Stricter than pattern alone |
| `--model` / `--scheduler` / `--steps` | Extra filters from parsed stem |
| `--exclude-task` / `--exclude-scheduler` | Comma-separated names to skip. Same flags on `prefix-sweep`, `prefix-sweep-pcfg`, and `prefix-selection` |
| `--include-duplicate-budgets` | Compare only. Also analyze step budgets whose NFE repeats a lower budget. By default `compare` and `prefix-sweep` skip those traces using `fetch_evals/strace_metrics.csv` |
| `--pcfg` | Force PCFG oracle gating even if the filename is not PCFG-like |
| `--replay-oracle` | Exact Mt/Jt + M↔M̂ / J↔Ĵ correlations for PCFG |
| `--oracle PATH` | Reuse saved `oracle.npz` (file or folder). Matched to Hub/local traces by stem; fills Mt/Jt/PPL + `corr_*` without live replay. With `--replay-oracle`, missing stems still replay |
| `--oracle-workers N` | Parallel samples in oracle replay (shared model + engine pool; default 1) |
| `--oracle-step-workers N` | Parallel PCFG Mt/Jt across steps within a sample (default `min(8, CPUs)`) |
| `--oracle-forward-batch B` | Batched model forwards over precomputed step inputs (default 8) |
| `--oracle-jt-exact-max K` | Exact PCFG `Jt` only for simultaneous sets with `\|S\|≤K` (default **3**). Larger `S` (DUS/EB parallel unmasks) use Markov `Jt` so oracle replay does not OOM. Env: `SCHEDULERLENS_JT_EXACT_MAX` |
| `--oracle-jt-markov-order M` | Markov order for large-`\|S\|` `Jt` (default **2** → exact joints of size ≤3). Env: `SCHEDULERLENS_JT_MARKOV_ORDER` |
| `--oracle-jt-large-set N` | If `\|S\|>N`, force **pairwise** `Jt` (order 1) for speed (default **6**). EB/DUS steps with `\|S\|=15+` dominate wall time otherwise. `0` disables. Env: `SCHEDULERLENS_JT_LARGE_SET` |
| `--namespace` | HF user/org for short `--repo` names **and** per-stem oracle models (`{ns}/{llada-100m-pcfg}`). Default: `TRACE_HF_NAMESPACE` / whoami |
| `--hf-model` | Pin one HF model for all traces (disables per-stem switch) |
| `--model-path` | Local checkpoint dir (or HF id if path does not exist) |
| `--limit N` | Cap number of files |
| `--resume` / `--no-resume` | Compare skips a filename only when the full row and every requested prefix are already in the CSV. A filename missing a prefix is redone. This is the default. `--no-resume` rewrites the sheet |
| `--no-purge-cache` | Skip Hub cache purge at batch start |
| `--prefix-steps Ks` | Extra CSV rows with cumulative M̂/Ĵ/Ĝ over `step < K` (e.g. `8,16,32,64`) |
| `--metrics-only` | Fast path: proxy caps (+ prefix) + strace join; skip completion + cheap PPL re-streams |

**Oracle gating:** `oracle_ppl` / Mt/Jt / `pearson_M` / `pearson_J` / `pearson_G` only when `--replay-oracle` or a matching `--oracle` artifact is used. Without those, compare still fills M̂/Ĵ + completion (+ cheap `model_commit_ppl` from trace confidences). Normal runs never get a correlation column.

NPZ samples are streamed via `traces.trace_npz.iter_npz_traces` (one sample in RAM at a time for proxies).

### Compare outputs

| File | Contents |
| ---- | -------- |
| `schedulerlens_metrics.csv` | Normal compare: proxies, completion, accuracy and NFE |
| `schedulerlens_metrics_pcfg.csv` | PCFG compare: proxies, completion, oracle M/J and sample Pearson |

CSV columns. A normal compare and a PCFG compare do not share the trailing columns.

Shared: `repo`, `filename`, `model`, `scheduler`, `task`, `steps`, `prefix_steps`, `is_pcfg`, `n_samples`, `n_steps`, `M_hat`, `J_hat`, `G_hat`, `commit_frac`, and the completion diagnostics.

Normal only: `accuracy`, `avg_nfe_per_sample`, `strace_name`, `strace_metric_key`.

PCFG only: `model_commit_ppl`, `oracle_ppl`, `Mt_total`, `Jt_total`, `pearson_M`, `pearson_J`, `pearson_G`.

`prefix_steps` is empty for the full trajectory. On a PCFG compare, prefixes 8, 16, 32, and 64 are always written as extra rows. `M_hat`/`J_hat`/`G_hat` on those rows are cumulative sums over `step < K`. `pearson_M`, `pearson_J`, and `pearson_G` on every row, including the full row, are sample-level Pearson: one point per sample, values are the sums of M̂ with Mt, of Ĵ with Jt, and of Ĝ = M̂+Ĵ with G = Mt+Jt over the same steps. Run-level oracle totals stay on the full row only.

Compare writes `schedulerlens_metrics.csv` for a normal batch and `schedulerlens_metrics_pcfg.csv` for a PCFG batch. Both go in `-o`, which defaults to `outputs/schedulerlens/compare`. `--csv` replaces that filename, or is an absolute path. `file` writes `metrics.csv` in its `-o` directory unless `--csv` is set.

Prefix sweep reads the run list from `fetch_evals/strace_metrics.csv` and downloads each `{name}.npz` from `devichand/schedulerlens-traces`. It uses the same NFE rule as compare: within one model, scheduler, and task, a requested budget whose NFE is within 1 of a lower budget is dropped, and the kept run is the requested budget closest to that NFE. The result is `outputs/schedulerlens/prefix_sweep/prefix_sweep.csv` unless `-o` or `--csv` is set. It writes `tau` (the entropy budget γ) instead of a single γ = 1. Columns are `filename`, `model`, `scheduler`, `task`, `steps`, `avg_nfe_per_sample`, `prefix`, `tau`, `M_hat`, `J_hat`, `G_hat`, `commit_frac`, `accuracy`, `n_samples`. PCFG lists runs from local `oracle_pcfg/*_oracle.npz` files and downloads the matching trace from `devichand/llada-pcfg-traces`. It writes `outputs/schedulerlens/prefix_sweep_pcfg/prefix_sweep_pcfg.csv` unless `-o` or `--csv` is set, with the same sample-level Pearson at prefixes `8`, `16`, `32`, `64`, and `full`, for each `tau`. Ĵ does not depend on `tau`, so `pearson_J` is repeated on each tau row. `pearson_G` does, because Ĝ includes M̂.

---

## Metrics (per step, gen region)

| Symbol | Trace field | Meaning |
| ------ | ----------- | ------- |
| `S_t` | selected (or inferred) | Unmasked at step `t` |
| `c_i`, `H_i` | `confidences` / `entropies` on `S_t` | Max-prob and Shannon entropy |
| `γ` | constant | Entropy budget on \(S_t\) (`1` nat) |

**M-side:** order \(S_t\) by increasing \(H\); \(\hat{M}_t\) sums \(-\log c\) over the prefix whose entropies sum to at most \(\gamma\). A token past the budget adds 0.  
**J-side:** \(\hat{J}_t = (\sum H - \max H)\) on \(S_t\) if \(|S_t|\ge 2\), else \(0\)

Sample caps: \(\hat{M}=\sum_t\hat{M}_t\); \(\hat{J}=\sum_t\hat{J}_t\). A prefix cap uses the same sums over steps with index \(< K\). Task caps: mean over samples. \(\hat{G}=\hat{M}+\hat{J}\).

On PCFG traces, `pearson_M` is the Pearson correlation of per-sample M̂ with per-sample Mt, `pearson_J` is the same for Ĵ and Jt, and `pearson_G` is the same for Ĝ = M̂+Ĵ and G = Mt+Jt. All three are computed at prefixes 8, 16, 32, 64 and over the full trajectory. Compare (γ = 1) and `prefix-sweep-pcfg` use this definition. Normal runs do not write a correlation.

**PCFG oracle (with `--replay-oracle`):** left-padded eval layout; teacher-forced reference prefix (`n_visible`); gen index → content via `n_visible + gen_pos`; Mt = Σ KL(q‖p) and Jt = joint-entropy gap (multi-information) on the **pre-step** committed set. Steps are precomputed from the unmask schedule (independent teacher-forced states), then **batched model forwards** + **parallel PCFG** (engine pool), with optional **multi-sample** workers.

**Large `|S|` / DUS:** exact `Jt` via CKY joint entropy is exponential in the simultaneous unmask set size (`|S|=8` can peak ~35 GB). SchedulerLens therefore:
- computes **exact** `Jt` when `|S| ≤ exact_max` (default 3; BD-style single-token steps stay exact);
- for larger sets, uses a **Markov chain-rule** approximation of order `M` (default 2): `Jt(S) ≈ Σ_i [Jt(S[i−M:i+1]) − Jt(S[i−M:i])]`, so every exact call uses joints of size ≤ `M+1 ≤ exact_max`.

Raise `exact_max` only if you have the RAM. Concurrent exact CKY `Jt` calls are capped by `SCHEDULERLENS_JT_CONCURRENCY` (default **2**) so `--oracle-step-workers 8` cannot stack eight multi-GB chart caches.

---

## Module layout

| Module | Role |
| ------ | ---- |
| `proxies.py` | Per-step M̂/Ĵ proxies |
| `metrics.py` | Aggregation, stem meta, CSV, correlations, row builder |
| `completion.py` | Commit / EOS / mask diagnostics, conf/ent means, strace accuracy/NFE join |
| `oracle.py` | Commit NLL, PCFG Mt/Jt replay, task perplexity |
| `hub.py` | Hub inventory and batch compare |
| `run.py` | **Single CLI** (`file` / `inventory` / `compare` / `causal` / `prefix-sweep` / `prefix-selection`) |
| `causal/` | Predictor–controller audit, swap, and adaptive selection |

`compare` and `file` join accuracy and NFE from `fetch_evals/strace_metrics.csv` onto each normal metrics row (stem == strace `name`; fallback model+task+steps+mode). Override with `--strace-metrics`. PCFG rows omit those columns.

Causal experiments (fixed-history deltas, live PC swap, proxy-guided selection) live in `schedulerlens/causal`. See [causal.md](causal.md).
