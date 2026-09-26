# SchedulerLens

SchedulerLens analyzes **masked diffusion schedulers** from generation traces. It reports hard-gated M̂ / star Ĵ caps, completion diagnostics, optional strace accuracy/NFE join, commit / oracle perplexity, and PCFG oracle–proxy correlations.

| Proxy | Definition | Cap |
| ----- | ---------- | --- |
| **M̂_t** (`M_hat_t`) | \(\sum_{i\in S_t}(-\log c_i)\cdot\mathbf{1}[H_i\le\tau]\), \(\tau=1.25\) | **M̂** = \(\sum_t\) M̂_t |
| **Ĵ_t** (`J_hat_t`) | star MI: \(\sum H-\max H\) on \(S_t\) if \(\|S_t\|\ge 2\), else 0 | **Ĵ** = \(\sum_t\) Ĵ_t (full); **Ĵ_early** = sum over the **early third** |
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
| `analyze` | One Hub NPZ (download→analyze→delete) or `--local` |
| `compare` | Batch Hub/local → CSV + plots (`analyze-all` alias) |

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
| `--repo` / `--local-dir` | Source (mutually exclusive) |
| `--pattern` | Filename glob against Hub/local names (default `*.npz`). Example: `*-pcfg_*.npz` keeps only PCFG-looking files by name |
| `--pcfg-only` | After discovery, keep only stems parsed as PCFG (`-pcfg_` / `pcfg_` in the name). Stricter than pattern alone |
| `--model` / `--scheduler` / `--steps` | Extra filters from parsed stem |
| `--pcfg` | Force PCFG oracle gating even if the filename is not PCFG-like |
| `--replay-oracle` | Exact Mt/Jt + M↔M̂ / J↔Ĵ correlations for PCFG |
| `--oracle PATH` | Reuse saved `oracle.npz` (file or folder). Matched to Hub/local traces by stem; fills Mt/Jt/PPL + `corr_*` without live replay. With `--replay-oracle`, missing stems still replay |
| `--oracle-workers N` | Parallel samples in oracle replay (shared model + engine pool; default 1) |
| `--oracle-step-workers N` | Parallel PCFG Mt/Jt across steps within a sample (default `min(8, CPUs)`) |
| `--oracle-forward-batch B` | Batched model forwards over precomputed step inputs (default 8) |
| `--oracle-jt-exact-max K` | Exact PCFG `Jt` only for simultaneous sets with `\|S\|≤K` (default **3**). Larger `S` (DUS/EB parallel unmasks) use Markov `Jt` so oracle replay does not OOM. Env: `SCHEDULERLENS_JT_EXACT_MAX` |
| `--oracle-jt-markov-order M` | Markov order for large-`\|S\|` `Jt` (default **2** → exact joints of size ≤3). Env: `SCHEDULERLENS_JT_MARKOV_ORDER` |
| `--oracle-jt-large-set N` | If `\|S\|>N`, force **pairwise** `Jt` (order 1) for speed (default **6**). EB/DUS steps with `\|S\|=15+` dominate wall time otherwise. `0` disables. Env: `SCHEDULERLENS_JT_LARGE_SET` |
| `--oracle-no-jt` | Skip `Jt` entirely (keep Mt + commit NLL/PPL). Fastest for EB full-file runs |
| `--namespace` | HF user/org for short `--repo` names **and** per-stem oracle models (`{ns}/{llada-100m-pcfg}`). Default: `TRACE_HF_NAMESPACE` / whoami |
| `--hf-model` | Pin one HF model for all traces (disables per-stem switch) |
| `--model-path` | Local checkpoint dir (or HF id if path does not exist) |
| `--limit N` | Cap number of files |
| `--resume` | Skip stems already in output CSV |
| `--no-purge-cache` | Skip Hub cache purge at batch start |

**Oracle gating:** `oracle_ppl` / Mt/Jt / `corr_*` only when `--replay-oracle` or a matching `--oracle` artifact is used. Without those, compare still fills M̂/Ĵ + completion (+ cheap `model_commit_ppl` from trace confidences).

NPZ samples are streamed via `traces.trace_npz.iter_npz_traces` (one sample in RAM at a time for proxies).

### Compare outputs

| File | Contents |
| ---- | -------- |
| `schedulerlens_metrics.csv` | All runs: proxy caps, completion diagnostics, strace accuracy/NFE, optional oracle |

CSV columns include:

- **Proxy caps:** `M_hat` (hard-gated NLL sum), `J_hat` (star sum full traj), `J_hat_early` (star sum early third), `G_hat` = M̂+Ĵ
- **Completion diagnostics:** commit / final-mask / EOS rates, `conf_mean` / `ent_mean`
- **Eval join** (from `fetch_evals/strace_metrics.csv` via `--strace-metrics`): `accuracy`, `avg_nfe_per_sample`, `strace_name`, `strace_metric_key`
- **Oracle** (with `--replay-oracle` or `--oracle`): `model_commit_ppl`, `oracle_ppl`, `Mt_total`, `Jt_total`, `corr_*`

Compare writes `schedulerlens_metrics.csv` only. `file` / `analyze` also write per-run `metrics.csv` / `metrics.json` (and `oracle.npz` when replaying).

### CSV schema (unified)

Identity: `repo`, `filename`, `model`, `scheduler`, `task`, `steps`, `is_pcfg`, `n_samples`, `n_steps`

Caps: `M_hat` (hard-gated sum), `J_hat` (star sum full), `J_hat_early` (star sum early third), `G_hat` = M̂+Ĵ

Completion diagnostics: commit / EOS / mask rates, `conf_mean` / `ent_mean`

Eval join: `accuracy`, `avg_nfe_per_sample` from `strace_metrics.csv` (stem == `name`)

PPL / oracle: `model_commit_ppl`, `oracle_ppl`, `Mt_total`, `Jt_total`

Correlations (PCFG + replay or cached `--oracle`, else empty): `corr_M_{full,early,mid,late}`, `corr_J_{full,early,mid,late}`, `corr_ppl`

Trajectory blocks are equal thirds of `n_steps`.

---

## Metrics (per step, gen region)

| Symbol | Trace field | Meaning |
| ------ | ----------- | ------- |
| `S_t` | selected (or inferred) | Unmasked at step `t` |
| `c_i`, `H_i` | `confidences` / `entropies` on `S_t` | Max-prob and Shannon entropy |
| `τ` | constant | Hard gate on entropy (`1.25` nats) |

**M-side:** \(\hat{M}_t = \sum_{i\in S_t}(-\log c_i)\cdot\mathbf{1}[H_i\le\tau]\)  
**J-side:** \(\hat{J}_t = (\sum H - \max H)\) on \(S_t\) if \(|S_t|\ge 2\), else \(0\)

Sample caps: \(\hat{M}=\sum_t\hat{M}_t\); \(\hat{J}=\sum_t\hat{J}_t\); \(\hat{J}_{\mathrm{early}}=\sum_t\hat{J}_t\) over the **early third** of steps. Task caps: mean over samples. \(\hat{G}=\hat{M}+\hat{J}\).

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
| `hub.py` | Hub inventory / download→analyze→delete / batch compare |
| `run.py` | **Single CLI** (`file` / `inventory` / `analyze` / `compare`) |

`compare` / `file` / `analyze` join accuracy and NFE from `fetch_evals/strace_metrics.csv` onto each metrics row (stem == strace `name`; fallback model+task+steps+mode). Override with `--strace-metrics`.
