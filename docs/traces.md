# Traces

Eval runs record **generation traces** for SchedulerLens. This doc covers format, the audit/convert pipeline, and testing.

Enable tracing in eval: `--tracing_enabled True` (see [evaluation.md](evaluation.md)).

---

## Module layout

| Module | Role |
| ------ | ---- |
| `traces/trace_generator.py` | `StepTrace`, `GenerationTrace`, `record_transfer` |
| `traces/trace_wrapper.py` | `@trace_scheduler`, hook implementation |
| `traces/trace_checks.py` | Field validation, duplicate-prompt dedup, mask-diff repair, NPZ round-trip |
| `traces/trace_metrics.py` | Compare trace aggregates vs `strace_metrics.csv` |
| `traces/trace_pipeline.py` | One-file audit and convert orchestration |
| `traces/trace_audit.py` | CLI (`inventory`, `audit`, `audit-all`, `convert`, `convert-all`) |
| `traces/trace_npz.py` | JSONL -> NPZ (lossless: prompt, final_sequence, step extras) |
| `traces/hf_trace_upload.py` | Hub upload/download/delete, cache cleanup, NPZ convert CLI |
| `tests/test_tracing.py` | 3 models × 8 schedulers tracing smoke test with audit |

---

## Trace format

Each sample is a JSON object (one line in `.jsonl`).

| Field | Description |
| ----- | ----------- |
| `prompt_len`, `gen_length` | Generation bounds |
| `step_traces` | Per-diffusion-step records |
| `unmasking_order` | Gen index -> first unmask step |
| `final_sequence` | Full sequence after generation |

### Per-step fields

| Field | Required | Description |
| ----- | -------- | ----------- |
| `mask_index` | yes | Still masked at step start |
| `predictions`, `confidences`, `entropies` | yes | Logits metrics |
| `selected_positions`, `selected_tokens` | no* | Unmask writes this step |

\*Required for SchedulerLens M̂/Ĵ proxies. When missing but mask diffs exist, traces are **inferrable** and can be repaired before NPZ conversion.

---

## Pipeline flow

### 1. Inventory (no downloads)

```bash
python traces/trace_audit.py inventory
```

Compares Hub filenames against `fetch_evals/strace_metrics.csv` for lm-eval trace repos. **PCFG-only repos** (all `.jsonl` names contain `-pcfg_`) skip that comparison and report file counts only.

### 2. Audit -> CSV (one file at a time)

```bash
python traces/trace_audit.py audit-all -o fetch_evals/trace_audit_flagged.csv
python traces/trace_audit.py audit-all --npz-only
python traces/trace_audit.py audit llada_humaneval_pure_diffusion_s256.jsonl
python traces/trace_audit.py audit llada_humaneval_pure_diffusion_s256.npz
python traces/trace_audit.py local trace_outputs/foo.jsonl --all-samples
python traces/trace_audit.py local trace_outputs/foo.npz --all-samples
```

`audit-all` audits both `.jsonl` and `.npz` on Hub by default. Use `--jsonl-only` or `--npz-only` to restrict. NPZ archives are lossless, so NPZ audits include full accuracy recompute when `prompt` / `final_sequence` are present.

Per file: **download -> stream-check hooks -> (optional) lm-eval accuracy -> compare metrics -> delete temp download**.

JSONL/NPZ audits **stream one sample at a time** for structure checks. Full-file load happens only when lm-eval accuracy is enabled for non-PCFG traces.

Trace audit/convert downloads each JSONL into a **temporary directory** (not the shared HF hub cache) and deletes it after processing. `audit-all` / `convert-all` also purge any stale cached revisions for the trace dataset repo at startup.

Output CSV includes `audit_flags` and `is_bad` columns. Resume (`audit-all` default) skips completed rows but **re-runs rows flagged `AUDIT_FAILED`, `MISSING_TRACE_METRIC`, or `MISSING_CODE_METRIC`** from a previous pass. Use `--no-retry-errors` to keep those skipped.

Humaneval/mbpp rows with no `metric_trace` are flagged `MISSING_CODE_METRIC` (pass@1 could not be recomputed from the trace).

**PCFG traces** (`pcfg_*`, `*-pcfg_*`, or `metadata.task=pcfg_suffix`): lm-eval scoring is skipped, but **`n_samples` is compared to the Hugging Face `pcfg-code` test split** (env `PCFG_HF_DATASET` / `PCFG_HF_NAMESPACE`, or `--pcfg-dataset` / `--pcfg-namespace` on audit commands). A complete contiguous `sample_idx` prefix (`0..n-1`) from `--max-samples` evals expects `n` instead of the full split. Mismatches flag `SAMPLE_COUNT_MISMATCH` in the audit CSV.

Accuracy uses lm-eval task definitions. Samples are matched to eval instances by **decoded prompt doc keys** (few-shot safe; e.g. MBPP problem text after the last `here is your task:`), with token-id fallback. Generations are decoded from `final_sequence`.

Audit checks:
- Required step fields present, correct lengths
- Null `selected_positions` / transfer hook status (`explicit_all`, `inferrable`, `missing_transfer`, …)
- **Duplicate prompts** — same `prompt` token ids; column `removable_duplicates` (= `n_samples - unique_prompts`)
- Missing `unmasking_order` when inferrable from hooks or mask diffs
- `metric_trace` vs `metric_wandb` (pass@1, exact_match, strict-match per task)
- Sample count vs expected task size (e.g. humaneval = 164)

When duplicates are present, accuracy scoring **dedupes in memory** (one trace per prompt) so metrics are not skewed; the Hub JSONL is unchanged until convert.

Use `--no-accuracy` to skip lm-eval scoring (faster; humaneval runs code execution).

### 3. Convert -> NPZ -> upload (one file at a time)

```bash
python traces/trace_audit.py convert-all -o fetch_evals/trace_convert.csv
python traces/trace_audit.py convert foo.jsonl --delete-jsonl-on-hub
python traces/trace_audit.py convert-all --output-repo schedulerlens-npz
```

`--output-repo` uploads NPZ to a different dataset repo while JSONL is still read from `--repo` (source). JSONL deletion with `--delete-jsonl-on-hub` always applies to the **source** repo only.

Per file:
1. Download JSONL
2. Stream-audit content (one sample in RAM at a time)
3. **Repair** when needed via streaming rewrite (dedup, hooks, `unmasking_order`). Conversion is **blocked** if repair is required but disabled (`--no-repair`), or if issues remain after repair (`status: repair_blocked`).
4. Stream-convert JSONL to NPZ in **RAM-aware batches** (two passes; batch size from free memory)
5. Validate NPZ round-trip against source (spot-check indices only)
6. Upload NPZ to Hub (`--output-repo` for a separate destination repo)
7. Optionally delete JSONL from Hub (`--delete-jsonl-on-hub`) after validation
8. Delete local cache

Disable repair with `--no-repair` on convert commands (blocks conversion when repair would be needed).

`convert-all` resume (default) skips `status=ok` rows but **automatically retries** `repair_blocked`, `roundtrip_failed`, and `convert_failed` rows. Use `--no-retry-errors` to keep failed rows skipped.

---

## Testing (hooks)

Quick check that tracing hooks work for all schedulers, with batched boolean-transfer schedulers at `batch_size=2`:

```bash
python tests/test_tracing.py
python tests/test_tracing.py --model-path ./models/llada --scheduler pure_diffusion
python tests/test_tracing.py --batch-size 1
```

Sets `TRACING_ENABLED=true` before importing schedulers (required for `@trace_scheduler` to attach). Default: `llada`, `llada1_5`, `dream` × 8 schedulers at `batch_size=2`. Reports `transfer_status` and `hooks_ok` per combo; FHS expects exactly `batch_size` traces per call (no double-wrap).

---

## Hub upload (manual)

```bash
python traces/hf_trace_upload.py upload trace_outputs/my_run.jsonl
python traces/hf_trace_upload.py download my_run.jsonl -o trace_outputs/
```

Env: `TRACE_HF_DATASET_REPO`, `TRACE_HF_NAMESPACE`, `TRACE_HF_REVISION`. Override per command with `--repo`, `--namespace`, `--revision`, and for convert with `--output-repo` / `--output-namespace`.

JSONL pack batching (convert): `TRACE_PACK_MEMORY_FRACTION` (default `0.25` of free RAM for parsed samples), `TRACE_PACK_MEMORY_BYTES` (fixed byte budget), `TRACE_PACK_BATCH_SAMPLES` (fixed batch size), `TRACE_PACK_JSON_EXPANSION` (default `2.5`, JSON line → dict overhead), `TRACE_PACK_MIN_BATCH` / `TRACE_PACK_MAX_BATCH`.

---

## SchedulerLens

See [schedulerlens.md](schedulerlens.md). Traces with complete `selected_positions` give accurate M̂/Ĵ caps. Repaired or inferrable traces work after NPZ conversion.
