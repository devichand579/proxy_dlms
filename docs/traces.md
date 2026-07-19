# Traces

Eval runs record **generation traces** for SchedulerLens. This doc covers format, the audit/convert pipeline, and testing.

Enable tracing in eval: `--tracing_enabled True` (see [evaluation.md](evaluation.md)).

---

## Module layout

| Module | Role |
| ------ | ---- |
| `traces/trace_generator.py` | `StepTrace`, `GenerationTrace`, `record_transfer` (OR-accumulate) |
| `traces/trace_wrapper.py` | `@trace_scheduler`, `suppress_step_recording` (FDM hyp) |
| `traces/utils.py` | Path helpers, gen-region vector squeeze, setitem utils |
| `traces/trace_checks.py` | Validate, repair (`iter_repaired_jsonl`), NPZ round-trip |
| `traces/trace_npz.py` | JSONL ↔ NPZ; `pack_repaired_jsonl_to_npz` |
| `traces/trace_pipeline.py` | Hub audit / convert / repair-npz orchestration |
| `traces/trace_audit.py` | CLI (`inventory`, `audit*`, `convert*`, `repair-npz*`) |
| `traces/trace_metrics.py` | Compare vs `strace_metrics.csv` / PCFG split |
| `traces/hf_trace_upload.py` | Hub upload/download/delete, cache cleanup |
| `tests/test_tracing.py` | 3 models × 8 schedulers tracing smoke |

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
- **Selection vs mask-diff counts** — when `selected_positions` is present, its True-count must match the mask-index unmask count for that step (steps with a next mask only); mismatches are flagged `SELECTION_MASK_MISMATCH` (repairable; common on DUS when per-position writes overwrite the transfer hook)
- **Phantom extra steps** — FDM (`_fdm_`) and/or FHS (`_fhs_`). FDM: more steps than permanent-unmask plateaus (empty-growth duplicates; Hub SL traces show identical mask/preds across the plateau, with `selected_*` only on the last step); repair keeps the **last** of each plateau. FHS predictor-corrector (`steps > 128`): corrector forwards were also traced (`n_steps ≈ gen_length + steps`); repair **merges** each chunk (keep predictor `mask_index` / `selected_positions`, copy corrector preds/conf/entropy/`selected_tokens`) then drops corrector steps. Live: corrector uses `suppress_step_recording(update_open_step=True)` so logits/tokens refresh the open predictor step.
- **Untrimmed trailing step** — every scheduler’s last `step_trace` has no next `mask_index`, so mask-diff cannot validate it (flag `UNTRIMMED_TAIL`). Repair **drops that last step** after phantom collapse (all schedulers), marks `metadata.schedulerlens_trimmed_tail`, and trusts `selected_*` on the new last step.
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
3. **Repair+pack** (default on): two streaming passes over the source JSONL (no intermediate repaired JSONL). Dedup, collapse FDM plateaus (keep last of each; from `_fdm_` in the filename), align `selected_*` to mask diffs, **drop the trailing step (all schedulers)**, rebuild `unmasking_order`, write NPZ. Use `--workers N` to repair samples in a process pool. Conversion is **blocked** if repair is required but disabled (`--no-repair`), or if issues remain after repair (`status: repair_blocked`).
4. Stream-audit the NPZ, spot-check round-trip on a few repaired samples, upload, optional JSONL delete, cleanup.

Stage timings log at INFO (`download`, `repair+pack`, `post-repair audit`, `roundtrip`, `upload`). NPZ write uses deflate level `TRACE_NPZ_COMPRESSLEVEL` (default `1`, faster than numpy’s old default); set `TRACE_NPZ_COMPRESS=0` for uncompressed (fastest, larger).

NPZ **reads** (SchedulerLens proxies/oracle, `repair-npz`, audits) pick mmap vs eager from free RAM, like pack batching: `TRACE_NPZ_MMAP=auto` (default) memmaps when uncompressed size exceeds `TRACE_NPZ_MEMORY_FRACTION` / `TRACE_PACK_MEMORY_FRACTION` (default 0.25) of available RAM; `=1` always mmap, `=0` always eager.

Disable repair with `--no-repair` on convert commands (blocks conversion when repair would be needed; falls back to plain pack).

`convert-all` resume (default) skips `status=ok` rows but **automatically retries** `repair_blocked`, `roundtrip_failed`, and `convert_failed` rows. Use `--no-retry-errors` to keep failed rows skipped.

### 4. Repair existing NPZs (NPZ → NPZ)

For archives already on Hub (or local) that need the selection/mask-diff alignment:

```bash
# Local
python traces/trace_audit.py repair-npz-local path/to/file.npz -o path/to/fixed.npz

# Hub (re-upload to same or --output-repo)
python traces/trace_audit.py repair-npz llada-100m-pcfg-pcfg_dus_s128.npz --repo llada-pcfg-traces
python traces/trace_audit.py repair-npz-all --repo llada-pcfg-traces -o fetch_evals/trace_repair_npz.csv
```

Skips clean files unless `--force`. Writes `fetch_evals/trace_repair_npz.csv` for batch runs.

---

FDM nested hypothesis forwards and FHS predictor-corrector corrector forwards use `suppress_step_recording()` so they do not open extra `step_traces` (live fix; archive repair still collapses older FDM plateaus / drops FHS correctors).

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

See [schedulerlens.md](schedulerlens.md). SchedulerLens reads `.npz` archives only (single-run or HF compare). Traces with complete `selected_positions` give accurate M̂/Ĵ caps; missing selection is inferrable from mask diffs, otherwise the run fails hard.

