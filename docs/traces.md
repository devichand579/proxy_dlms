# Traces

Eval runs record **generation traces** for SchedulerLens. This doc covers format, the audit/convert pipeline, and testing.

Enable tracing in eval: `--tracing_enabled True` (see [evaluation.md](evaluation.md)).

---

## Module layout

| Module | Role |
| ------ | ---- |
| `traces/trace_generator.py` | `StepTrace`, `GenerationTrace`, `record_transfer` |
| `traces/trace_wrapper.py` | `@trace_scheduler`, hook implementation |
| `traces/trace_checks.py` | Field validation, mask-diff repair, NPZ round-trip |
| `traces/trace_metrics.py` | Compare trace aggregates vs `strace_metrics.csv` |
| `traces/trace_pipeline.py` | One-file audit and convert orchestration |
| `traces/trace_audit.py` | CLI (`inventory`, `audit`, `audit-all`, `convert`, `convert-all`) |
| `traces/trace_npz.py` | JSONL -> NPZ v2 |
| `traces/hf_trace_upload.py` | Hub upload/download/delete, cache cleanup, NPZ convert CLI |
| `test_tracing.py` | 3 models × 8 schedulers tracing smoke test with audit |

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

Compares Hub filenames against 360 expected runs in `fetch_evals/strace_metrics.csv`.

### 2. Audit -> CSV (one file at a time)

```bash
python traces/trace_audit.py audit-all -o fetch_evals/trace_audit_flagged.csv
python traces/trace_audit.py audit llada_humaneval_pure_diffusion_s256.jsonl
python traces/trace_audit.py local trace_outputs/foo.jsonl --all-samples
```

Per file: **download -> check hooks -> recompute task accuracy (pass@1 / exact_match / strict-match) -> compare vs W&B CSV -> flag issues -> delete temp download**.

Trace audit/convert downloads each JSONL into a **temporary directory** (not the shared HF hub cache) and deletes it after processing. `audit-all` / `convert-all` also purge any stale cached revisions for the trace dataset repo at startup.

Output CSV includes `audit_flags` and `is_bad` columns. Resume (`audit-all` default) skips completed rows but **re-runs rows flagged `AUDIT_FAILED`, `MISSING_TRACE_METRIC`, or `MISSING_CODE_METRIC`** from a previous pass. Use `--no-retry-errors` to keep those skipped.

Humaneval/mbpp rows with no `metric_trace` are flagged `MISSING_CODE_METRIC` (pass@1 could not be recomputed from the trace).

Accuracy uses lm-eval task definitions with the same few-shot counts as eval. Samples are matched to eval instances by prompt token ids; generations are decoded from `final_sequence`.

Audit checks:
- Required step fields present, correct lengths
- Null `selected_positions` / transfer hook status (`explicit_all`, `inferrable`, `missing_transfer`, …)
- `metric_trace` vs `metric_wandb` (pass@1, exact_match, strict-match per task)
- Sample count vs expected task size (e.g. humaneval = 164)

Use `--no-accuracy` to skip lm-eval scoring (faster; humaneval runs code execution).

### 3. Convert -> NPZ -> upload (one file at a time)

```bash
python traces/trace_audit.py convert-all -o fetch_evals/trace_convert.csv
python traces/trace_audit.py convert foo.jsonl --delete-jsonl-on-hub
```

Per file:
1. Download JSONL
2. Audit content
3. **Repair** missing `selected_positions` from mask diffs when inferrable
4. Convert to NPZ
5. Validate NPZ round-trip against source
6. Upload NPZ to Hub
7. Optionally delete JSONL from Hub (`--delete-jsonl-on-hub`) after validation
8. Delete local cache

Resumable via CSV; skips files that already have NPZ on Hub (use `--force` to re-convert).

---

## Testing (hooks)

Quick check that tracing hooks work for all schedulers, with batched boolean-transfer schedulers at `batch_size=2`:

```bash
python test_tracing.py
python test_tracing.py --model-path ./models/llada --scheduler pure_diffusion
python test_tracing.py --batch-size 1
```

Default: `llada`, `llada1_5`, `dream` × 8 schedulers. Reports `transfer_status` and `hooks_ok` per combo.

---

## Hub upload (manual)

```bash
python traces/hf_trace_upload.py upload trace_outputs/my_run.jsonl
python traces/hf_trace_upload.py download my_run.jsonl -o trace_outputs/
```

Env: `TRACE_HF_DATASET_REPO`, `TRACE_HF_NAMESPACE`, `TRACE_HF_REVISION`.

---

## SchedulerLens

See [schedulerlens.md](schedulerlens.md). Traces with complete `selected_positions` give accurate M̂/Ĵ caps. Repaired or inferrable traces work after NPZ conversion.
