# SchedulerLens

SchedulerLens analyzes **masked diffusion schedulers** from generation traces. It reports M̂/Ĵ caps and sub-proxies, commit / oracle perplexity, optional oracle–proxy correlations (PCFG), and comparison plots across runs.

| Proxy | Role | Lower is better? |
| ----- | ---- | ---------------- |
| **M̂_t** (`M_hat_t`) | Ordering quality | Yes |
| **mean_regret** | Post-commit regret over `C_t` | Yes |
| **cascade_gain** | Confidence lift on still-masked positions | Context-dependent |
| **Ĵ_t** (`J_hat_t`) | Parallelism error | Yes |
| **instability** | Argmax flips on `S_t` | Yes |
| **rank_volatility** | Confidence rank shuffling over gen region | Yes |

Top-level `M_hat` / `J_hat` are entropy-weighted caps of `M_hat_t` / `J_hat_t`.

---

## Prerequisites

| Requirement | Why |
| ----------- | --- |
| **Trace archive** (`.npz` only) | From `traces/trace_npz.py` (convert JSONL via `trace_audit.py convert`) |
| **Gen-only step fields** | `mask_index`, `predictions`, `confidences`, `entropies`; `selected_positions` / `selected_tokens` or inferrable from consecutive masks |

---

## Fail-hard rules

- Missing / inconsistent required fields → **`TraceProxyError`**; the run aborts (no silent zeros, no skipping samples).
- Proxies and oracle call **`validate_trace_for_metrics`** first. They do **not** silently repair. Unrepaired issues raise hard errors, including:
  - missing `mask_index` / `predictions` / `confidences` / `entropies` (or wrong lengths)
  - `selected_positions` vs mask-diff set/count mismatch (DUS remask / incomplete hooks)
  - missing `selected_*` when mask-diff has unmasks
  - FDM **phantom extra steps** (nested hyp forwards)
  - **untrimmed trailing step** (`schedulerlens_trimmed_tail` missing) — repair drops the last step for **all** schedulers
- On the **last kept step** after trim, `selected_*` is trusted (no next mask to compare).
- Empty selection on non-last steps is valid only when that step’s mask-diff is also empty.
- Repair traces first (`trace_audit.py convert` / `repair-npz`), then run SchedulerLens.
- When **no tokens are unmasked** at a step (`n_selected == 0`), **all** step proxies are **0** (`M_hat_t`, `J_hat_t`, and every sub-quantity).

---

## CLI (single entrypoint)

All commands go through `python -m schedulerlens.run` (or `python -m schedulerlens`):

| Subcommand | Role |
| ---------- | ---- |
| `file` | One local NPZ → metrics + plots |
| `inventory` | List Hub NPZs (no download) |
| `analyze` | One Hub NPZ (download→analyze→delete) or `--local` |
| `compare` | Batch Hub/local → CSV + plots (`analyze-all` alias) |
| `pos` | POS-stratified unmask analysis |

```bash
# Local NPZ
python -m schedulerlens.run file trace_outputs/your_run.npz -o outputs/schedulerlens/run1
python -m schedulerlens.run file path/to/llada-100m-pcfg_pd_s128.npz -o out/ \
  --replay-oracle --model-path /path/to/pcfg/ckpt

# Hub inventory / one file / batch
python -m schedulerlens.run inventory --repo schedulerlens-npz --pcfg-only
python -m schedulerlens.run analyze llada-300m-pcfg_pd_s128.npz \
  --repo schedulerlens-npz -o outputs/schedulerlens/one \
  --replay-oracle --model-path checkpoints/llada-300m-pcfg-1B/checkpoint-best_val
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

# POS
python -m schedulerlens.run pos trace.npz -o out/pos --bucket-size 16
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
| `--repo` / `--local-dir` | Source (mutually exclusive) |
| `--pattern` | Filename glob against Hub/local names (default `*.npz`). Example: `*-pcfg_*.npz` keeps only PCFG-looking files by name |
| `--pcfg-only` | After discovery, keep only stems parsed as PCFG (`-pcfg_` / `pcfg_` in the name). Stricter than pattern alone |
| `--model` / `--scheduler` / `--steps` | Extra filters from parsed stem |
| `--pcfg` | Force PCFG oracle gating even if the filename is not PCFG-like |
| `--replay-oracle` | Exact Mt/Jt + M↔M̂ / J↔Ĵ correlations for PCFG |
| `--oracle-workers N` | Parallel samples in oracle replay (shared model + engine pool; default 1) |
| `--oracle-step-workers N` | Parallel PCFG Mt/Jt across steps within a sample (default `min(8, CPUs)`) |
| `--oracle-forward-batch B` | Batched model forwards over precomputed step inputs (default 8) |
| `--oracle-jt-exact-max K` | Exact PCFG `Jt` only for simultaneous sets with `\|S\|≤K` (default **3**). Larger `S` (DUS/EB parallel unmasks) use Markov `Jt` so oracle replay does not OOM. Env: `SCHEDULERLENS_JT_EXACT_MAX` |
| `--oracle-jt-markov-order M` | Markov order for large-`\|S\|` `Jt` (default **2** → exact joints of size ≤3). Env: `SCHEDULERLENS_JT_MARKOV_ORDER` |
| `--namespace` | HF user/org for short `--repo` names **and** per-stem oracle models (`{ns}/{llada-100m-pcfg}`). Default: `TRACE_HF_NAMESPACE` / whoami |
| `--hf-model` | Pin one HF model for all traces (disables per-stem switch) |
| `--model-path` | Local checkpoint dir (or HF id if path does not exist) |
| `--limit N` | Cap number of files |
| `--resume` | Skip stems already in output CSV |
| `--no-purge-cache` | Skip Hub cache purge at batch start |

**Oracle gating:** exact Mt/Jt and proxy correlations run only when the trace is PCFG (`-pcfg_` / `pcfg_` in the name, or `--pcfg`) **and** `--replay-oracle` is set. Non-PCFG runs still get M̂/Ĵ + model commit PPL + overview/surface plots.

NPZ samples are streamed via `traces.trace_npz.iter_npz_traces` (one sample in RAM at a time for proxies).

### Compare outputs

| File | Contents |
| ---- | -------- |
| `schedulerlens_metrics.csv` | All runs, unified columns |
| `compare_summary.json` | Same rows as JSON |
| `compare_overview.png` | Scheduler × model bars for M̂, Ĵ, PPL |
| `compare_ppl.png` | Model vs oracle PPL (when oracle present) |
| `compare_proxy_corr.png` | Mean ρ by early/mid/late block |
| `compare_proxy_fidelity.png` | Scatter Mt vs M̂_t, Jt vs Ĵ_t |
| `compare_surfaces.png` | Step × token heatmaps |
| `plot_artifacts/<stem>/oracle.npz` | **Full oracle replay dump** (see below) — keep this; replaying Mt/Jt is expensive |
| `plot_artifacts/<stem>/fidelity.npz` | Mean step trends for fidelity scatter (also inside `oracle.npz`) |
| `plot_artifacts/<stem>/surfaces.npz` | Confidence/entropy surfaces (optional; cheap to recompute) |
| `run_plots/<stem>/` | Per-run PNGs written at analyze time |

### Oracle artifact (`oracle.npz`)

Written whenever `--replay-oracle` produces per-sample detail. Prefer this over re-running oracle.

| Level | Arrays | Contents |
| ----- | ------ | -------- |
| Meta | `jt_exact_max`, `jt_markov_order`, `n_samples`, `stem` | Jt policy used |
| Task | `oracle_ppl`, `model_commit_ppl`, `Mt_total`, `Jt_total`, … | Same as CSV caps |
| Sample | `sample_oracle_nll/ppl`, `sample_model_nll/ppl`, `sample_Mt`, `sample_Jt` | One row per NPZ sample |
| Step | `step_sample_idx`, `step`, `step_Mt`, `step_Jt`, `step_oracle_nll`, `step_model_nll`, `step_n_selected` | Per-sample × step |
| Token | `tok_sample_idx`, `tok_step`, `tok_gen_pos`, `tok_content_pos`, `tok_sel`, `tok_ref`, `tok_mt_kl`, `tok_oracle_nll`, `tok_model_nll` | Per commit |

`Jt` is set-level (one value per step); `Mt` / NLLs are stored per token. Proxy M̂/Ĵ caps are **not** stored here — recompute from traces if needed.

Load: `from schedulerlens.oracle import load_oracle_artifacts`.

### CSV schema (unified)

Identity: `repo`, `filename`, `model`, `scheduler`, `task`, `steps`, `is_pcfg`, `n_samples`, `n_steps`

Caps: `M_hat`, `J_hat`, `M_hat_uniform`, `J_hat_uniform`, plus `*_entropy` / `*_uniform` for sub-proxies

PPL / oracle: `model_commit_ppl`, `oracle_ppl`, `Mt_total`, `Jt_total`

Correlations (PCFG+replay, else empty): `corr_M_{full,early,mid,late}`, `corr_J_{full,early,mid,late}`, `corr_ppl`

Trajectory blocks are equal thirds of `n_steps`.

---

## Metrics (per step, gen region)

| Symbol | Trace field | Meaning |
| ------ | ----------- | ------- |
| `S_t` | selected (or inferred) | Unmasked at step `t` |
| `C_t` | `~mask_index` | Committed before step `t` |
| `S̄_t` | still masked after | Remaining masked |
| `w_t` | `sum(entropies[S_t])` | Entropy weight |

**M-side:** `M̂_t = mean_regret / (1 + max(0, cascade_gain))`  
**J-side:** `Ĵ_t = instability × rank_volatility`

Sample caps: over steps with `n_selected > 0` — entropy-weighted `Σ w_t x_t / Σ w_t` and uniform `mean(x_t)`. Task caps: mean over samples.

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
| `oracle.py` | Commit NLL, PCFG Mt/Jt replay, task perplexity |
| `hub.py` | Hub inventory / download→analyze→delete / batch compare |
| `run.py` | **Single CLI** (`file` / `inventory` / `analyze` / `compare` / `pos`) |
| `pos.py` | POS analysis library |
| `plots/viz.py` | All M/J/compare/quad/eval-join/surface plots |
| `plots/pos.py` | POS unmask timing plots |

Compare also writes:

| File | Contents |
| ---- | -------- |
| `compare_quad_bars.png` | Faceted bars: M̂/Ĵ by scheduler, hue = step budget |
| `compare_quad_heat_M.png` / `_J.png` | Heatmaps scheduler × steps per (model, task) |
| `compare_accuracy_vs_proxy.png` | Accuracy vs M̂/Ĵ (from strace_metrics) |
| `compare_nfe_pareto.png` | NFE vs accuracy colored by M̂/Ĵ |
| `compare_budget_curves.png` | Proxies / accuracy / NFE vs steps when multiple budgets exist |

Join key: NPZ stem == strace `name` (fallback: model+task+steps+mode). Override path with `--strace-metrics`.

---

## POS unmasking analysis

Separate from M̂/Ĵ: `python -m schedulerlens.run pos TRACE.npz -o OUT/`.
