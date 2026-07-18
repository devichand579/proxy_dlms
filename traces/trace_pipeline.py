"""One-file-at-a-time Hub trace audit and NPZ conversion pipelines."""

from __future__ import annotations

import csv
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from traces.hf_trace_upload import (
    cleanup_repo_cache,
    delete_file,
    download_file_to_temp,
    ensure_dataset_repo,
    file_sizes,
    list_repo_files,
    remove_temp_download,
    resolve_repo,
    trace_hf_dataset_private,
    trace_hf_revision,
    upload_file,
)
from traces.trace_checks import (
    TraceCheck,
    check_to_row,
    check_trace_jsonl_streaming,
    check_trace_npz_streaming,
    conversion_repair_blockers,
    deduplicate_prompt_traces,
    stream_repair_jsonl,
    stream_repair_npz,
    trace_needs_repair,
    validate_npz_roundtrip,
    validate_npz_roundtrip_indexed,
)
from traces.trace_metrics import (
    compare_pcfg_metrics_row,
    compare_with_metrics_row,
    expected_trace_names,
    is_pcfg_trace,
    load_metrics_lookup,
    prewarm_scoring_tasks,
)
from traces.trace_npz import (
    compression_ratio,
    convert_jsonl_to_npz,
    load_trace_samples,
    pack_repaired_jsonl_to_npz,
    read_jsonl_task_name,
    read_npz_task_name,
)

logger = logging.getLogger(__name__)


def _stage(label: str, t0: float, **extra: Any) -> float:
    """Log elapsed seconds for a pipeline stage; return duration."""
    dt = time.perf_counter() - t0
    extras = " ".join(f"{k}={v}" for k, v in extra.items() if v is not None)
    if extras:
        logger.info("%s done in %.1fs (%s)", label, dt, extras)
    else:
        logger.info("%s done in %.1fs", label, dt)
    return dt

DEFAULT_METRICS_CSV = Path(__file__).resolve().parent.parent / "fetch_evals/strace_metrics.csv"
DEFAULT_AUDIT_CSV = Path(__file__).resolve().parent.parent / "fetch_evals/trace_audit_flagged.csv"
DEFAULT_CONVERT_CSV = Path(__file__).resolve().parent.parent / "fetch_evals/trace_convert.csv"
DEFAULT_REPAIR_NPZ_CSV = Path(__file__).resolve().parent.parent / "fetch_evals/trace_repair_npz.csv"

_CODE_EVAL_TASKS = frozenset({"humaneval", "openai_humaneval", "mbpp"})
_METRIC_DELTA_FLAG_THRESHOLD = 0.02
_METRIC_DELTA_WITH_COUNT_MISMATCH_THRESHOLD = 0.01


def _repo_is_pcfg_only(jsonl_files: list[str]) -> bool:
    if not jsonl_files:
        return False
    return all(is_pcfg_trace(_stem(f), first_metadata=None) for f in jsonl_files)


def hub_inventory(
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    metrics_csv: str | Path | None = None,
    token: bool | str | None = True,
) -> dict[str, Any]:
    full_repo = resolve_repo(repo_id, namespace)
    rev = revision if revision is not None else trace_hf_revision()
    files = list_repo_files(repo_id=full_repo, revision=rev, token=token)
    jsonl = sorted(f for f in files if f.endswith(".jsonl"))
    npz = sorted(f for f in files if f.endswith(".npz"))
    other = sorted(f for f in files if not f.endswith((".jsonl", ".npz")))

    pcfg_only = _repo_is_pcfg_only(jsonl)
    expected: set[str] = set()
    if not pcfg_only:
        try:
            expected = expected_trace_names(metrics_csv)
        except FileNotFoundError:
            expected = set()

    expected_jsonl = {f"{n}.jsonl" for n in expected}
    expected_npz = {f"{n}.npz" for n in expected}

    return {
        "repo": full_repo,
        "revision": rev,
        "pcfg_only": pcfg_only,
        "expected_variants": len(expected),
        "jsonl_count": len(jsonl),
        "npz_count": len(npz),
        "other_files": other,
        "missing_jsonl": sorted(expected_jsonl - set(jsonl)),
        "extra_jsonl": sorted(set(jsonl) - expected_jsonl),
        "missing_npz": sorted(expected_npz - set(npz)),
        "extra_npz": sorted(set(npz) - expected_npz),
        "jsonl_files": jsonl,
        "npz_files": npz,
    }


def _stem(filename: str) -> str:
    for ext in (".jsonl", ".json", ".npz"):
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return Path(filename).stem


def _load_done_names(csv_path: Path, name_field: str = "name") -> set[str]:
    if not csv_path.is_file():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row.get(name_field, "") for row in csv.DictReader(f) if row.get(name_field)}


def _should_retry_convert_row(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "") != "ok"


def _should_retry_repair_npz_row(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "") not in ("ok", "skipped_clean")


def _load_existing_convert_rows(
    csv_path: Path,
    *,
    retry_errors: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    if not csv_path.is_file():
        return [], set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    done = {
        row["name"]
        for row in rows
        if row.get("name") and not (retry_errors and _should_retry_convert_row(row))
    }
    return rows, done


def _convert_failure_row(
    *,
    name: str,
    filename: str,
    npz_name: str,
    source_repo: str,
    output_repo: str,
    status: str,
    check: TraceCheck,
    repaired: bool,
    n_samples: int,
    jsonl_bytes: int = 0,
    npz_bytes: int = 0,
    compression_ratio: str = "",
    issues: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "jsonl_filename": filename,
        "npz_filename": npz_name,
        "source_repo": source_repo,
        "output_repo": output_repo,
        "status": status,
        "transfer_status": check.transfer_status,
        "repaired": repaired,
        "n_samples": n_samples,
        "jsonl_bytes": jsonl_bytes,
        "npz_bytes": npz_bytes,
        "compression_ratio": compression_ratio,
        "roundtrip_ok": False,
        "jsonl_deleted_on_hub": False,
        "issues": "; ".join(issues),
    }


def _is_audit_error_row(row: dict[str, Any]) -> bool:
    issues = str(row.get("issues") or "")
    return issues.startswith("audit failed:")


def _should_retry_audit_row(row: dict[str, Any]) -> bool:
    if _is_audit_error_row(row):
        return True
    flags = str(row.get("audit_flags") or "")
    return (
        "AUDIT_FAILED" in flags
        or "MISSING_TRACE_METRIC" in flags
        or "MISSING_CODE_METRIC" in flags
    )


def _resolve_output_repo(
    *,
    repo_id: Optional[str],
    namespace: Optional[str],
    output_repo_id: Optional[str],
    output_namespace: Optional[str],
) -> str:
    return resolve_repo(
        output_repo_id or repo_id,
        output_namespace if output_namespace is not None else namespace,
    )


def _audit_done_key(row: dict[str, Any]) -> str:
    return str(row.get("filename") or row.get("name") or "")


def _load_existing_audit_rows(
    csv_path: Path,
    *,
    retry_errors: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    if not csv_path.is_file():
        return [], set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [annotate_audit_row(dict(row)) for row in csv.DictReader(f)]
    done = {
        _audit_done_key(row)
        for row in rows
        if _audit_done_key(row) and not (retry_errors and _should_retry_audit_row(row))
    }
    return rows, done


def _metric_delta_flag_value(row: dict[str, Any]) -> Optional[float]:
    md = row.get("metric_delta")
    if md in (None, ""):
        return None
    try:
        return float(md)
    except (TypeError, ValueError):
        return None


def _sample_count_delta_value(row: dict[str, Any]) -> Optional[int]:
    scd = row.get("sample_count_delta")
    if scd in (None, ""):
        return None
    try:
        return int(float(scd))
    except (TypeError, ValueError):
        return None


def compute_audit_flags(row: dict[str, Any]) -> tuple[str, bool]:
    """Derive audit_flags / is_bad from one audit CSV row."""
    flags: list[str] = []
    issues = str(row.get("issues") or "")
    if issues.startswith("audit failed:"):
        flags.append("AUDIT_FAILED")
    if row.get("transfer_status") == "missing_transfer":
        flags.append("MISSING_TRANSFER")
    try:
        if int(row.get("samples_selection_mask_mismatch") or 0) > 0:
            flags.append("SELECTION_MASK_MISMATCH")
    except (TypeError, ValueError):
        pass
    if "selection/mask-diff count mismatch" in issues:
        if "SELECTION_MASK_MISMATCH" not in flags:
            flags.append("SELECTION_MASK_MISMATCH")
    try:
        if int(row.get("samples_phantom_steps") or 0) > 0:
            flags.append("PHANTOM_STEPS")
        if int(row.get("samples_untrimmed_tail") or 0) > 0:
            flags.append("UNTRIMMED_TAIL")
    except (TypeError, ValueError):
        pass
    if "phantom extra steps" in issues and "PHANTOM_STEPS" not in flags:
        flags.append("PHANTOM_STEPS")
    if "untrimmed trailing step" in issues and "UNTRIMMED_TAIL" not in flags:
        flags.append("UNTRIMMED_TAIL")

    sample_delta = _sample_count_delta_value(row)
    if sample_delta not in (None, 0):
        flags.append(f"SAMPLE_COUNT_MISMATCH({sample_delta:+d})")

    task = str(row.get("task") or "").strip().lower()
    metric_wandb = row.get("metric_wandb")
    metric_trace = row.get("metric_trace")
    metrics_flags = str(row.get("metrics_flags") or "")
    if metric_wandb and not metric_trace:
        if task in _CODE_EVAL_TASKS:
            flags.append("MISSING_CODE_METRIC")
        elif "HF_ALLOW_CODE_EVAL" not in metrics_flags and (
            "lm-eval task load failed" in metrics_flags
            or "Input/output error" in metrics_flags
            or "score failed" in metrics_flags
        ):
            flags.append("MISSING_TRACE_METRIC")

    metric_delta = _metric_delta_flag_value(row)
    if metric_delta is not None:
        threshold = (
            _METRIC_DELTA_WITH_COUNT_MISMATCH_THRESHOLD
            if sample_delta not in (None, 0)
            else _METRIC_DELTA_FLAG_THRESHOLD
        )
        if metric_delta > threshold:
            flags.append(f"METRIC_DELTA({metric_delta:.4f})")

    audit_flags = "|".join(flags)
    return audit_flags, bool(flags)


def annotate_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    audit_flags, is_bad = compute_audit_flags(row)
    row["audit_flags"] = audit_flags
    row["is_bad"] = is_bad
    return row


def _write_csv_rows(rows: list[dict[str, Any]], output_csv: Path, fields: list[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            if out.get("size_bytes"):
                try:
                    out["size_mb"] = f"{float(out['size_bytes']) / 1e6:.2f}"
                except (TypeError, ValueError):
                    pass
            w.writerow(out)


AUDIT_CSV_FIELDS = [
    "name",
    "filename",
    "source_format",
    "on_hub",
    "size_bytes",
    "size_mb",
    "model",
    "task",
    "mode",
    "steps_cfg",
    "transfer_status",
    "hooks_ok",
    "n_samples",
    "samples_checked",
    "samples_proxy_ready",
    "samples_inferrable",
    "samples_no_transfer",
    "samples_with_any_selected",
    "samples_with_all_selected",
    "samples_missing_required",
    "samples_selection_mask_mismatch",
    "samples_phantom_steps",
    "samples_untrimmed_tail",
    "unique_prompts",
    "duplicate_traces",
    "removable_duplicates",
    "ambiguous_duplicate_prompts",
    "metric_wandb",
    "metric_key_wandb",
    "metric_trace",
    "metric_key_trace",
    "metric_delta",
    "n_scored",
    "n_unmatched",
    "n_samples_expected",
    "sample_count_delta",
    "metrics_flags",
    "issues",
    "audit_flags",
    "is_bad",
]

CONVERT_CSV_FIELDS = [
    "name",
    "jsonl_filename",
    "npz_filename",
    "source_repo",
    "output_repo",
    "status",
    "transfer_status",
    "repaired",
    "n_samples",
    "jsonl_bytes",
    "npz_bytes",
    "compression_ratio",
    "roundtrip_ok",
    "jsonl_deleted_on_hub",
    "issues",
]

REPAIR_NPZ_CSV_FIELDS = [
    "name",
    "npz_filename",
    "source_repo",
    "output_repo",
    "status",
    "transfer_status",
    "repaired",
    "n_samples",
    "npz_bytes_in",
    "npz_bytes_out",
    "roundtrip_ok",
    "issues",
]


def _metrics_for_audit(
    *,
    name: str,
    check: TraceCheck,
    meta: dict[str, str],
    metrics_csv: str | Path | None,
    compute_accuracy: bool,
    samples: list[dict] | None = None,
    pcfg_hf_dataset: str | None = None,
    pcfg_hf_namespace: str | None = None,
    token: bool | str | None = True,
) -> dict[str, Any]:
    pcfg = is_pcfg_trace(name, samples, first_metadata=check.first_metadata)
    if pcfg:
        if check.first_metadata:
            m = check.first_metadata
            meta = {
                "model": str(m.get("variant", "pcfg")),
                "task": "pcfg",
                "mode": str(m.get("scheduler", "")),
                "steps": "",
            }
        n_samples = check.n_samples - check.removable_duplicates
        return compare_pcfg_metrics_row(
            n_samples=n_samples,
            sample_indices=check.pcfg_sample_indices,
            hf_dataset=pcfg_hf_dataset,
            namespace=pcfg_hf_namespace,
            token=token,
        )

    if not compute_accuracy:
        return {
            "metric_wandb": None,
            "metric_key_wandb": "",
            "metric_trace": None,
            "metric_key_trace": "",
            "metric_delta": None,
            "n_scored": 0,
            "n_unmatched": 0,
            "n_samples_expected": None,
            "sample_count_delta": None,
            "metrics_flags": "accuracy skipped",
        }

    if samples is None:
        raise ValueError("full trace samples required for lm-eval accuracy audit")

    samples_for_metrics = samples
    if check.removable_duplicates > 0:
        samples_for_metrics, _ = deduplicate_prompt_traces(samples)

    return compare_with_metrics_row(
        run_name=name,
        samples=samples_for_metrics,
        metrics_lookup=load_metrics_lookup(metrics_csv),
        task=meta.get("task", ""),
        model=meta.get("model", ""),
        compute_accuracy=True,
    )


def audit_one_hub_file(
    filename: str,
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    metrics_csv: str | Path | None = None,
    max_samples: Optional[int] = 3,
    compute_accuracy: bool = True,
    pcfg_hf_dataset: str | None = None,
    pcfg_hf_namespace: str | None = None,
    token: bool | str | None = True,
) -> dict[str, Any]:
    """Download one Hub trace file (JSONL or NPZ), audit, compare metrics, cleanup cache."""
    source_format = "npz" if filename.endswith(".npz") else "jsonl"
    full_repo = resolve_repo(repo_id, namespace)
    rev = revision if revision is not None else trace_hf_revision()
    name = _stem(filename)
    meta: dict[str, str] = {}
    try:
        meta = load_metrics_lookup(metrics_csv).get(name, {})
    except FileNotFoundError:
        pass
    t_all = time.perf_counter()
    logger.info("audit %s (%s) from %s", filename, source_format, full_repo)
    t0 = time.perf_counter()
    sizes = file_sizes([filename], repo_id=full_repo, revision=rev, token=token)
    _stage("hub size lookup", t0)

    temp_dir: Path | None = None
    try:
        t0 = time.perf_counter()
        path, temp_dir = download_file_to_temp(
            filename, repo_id=full_repo, revision=rev, token=token
        )
        _stage(
            "download",
            t0,
            mb=f"{path.stat().st_size / 1e6:.1f}",
        )
        name_is_pcfg = is_pcfg_trace(name, first_metadata=None)
        need_full_samples = compute_accuracy and not name_is_pcfg

        t0 = time.perf_counter()
        if source_format == "jsonl":
            check = check_trace_jsonl_streaming(
                path,
                max_samples=max_samples,
                collect_pcfg_audit=True,
            )
            if not name_is_pcfg:
                name_is_pcfg = is_pcfg_trace(name, first_metadata=check.first_metadata)
            if name_is_pcfg or not compute_accuracy:
                need_full_samples = False
        else:
            check = check_trace_npz_streaming(
                path,
                max_samples=max_samples,
                collect_pcfg_audit=True,
            )
            if not name_is_pcfg:
                name_is_pcfg = is_pcfg_trace(name, first_metadata=check.first_metadata)
            if name_is_pcfg or not compute_accuracy:
                need_full_samples = False
        _stage(
            "stream check",
            t0,
            n=check.n_samples,
            checked=check.samples_checked,
            transfer=check.transfer_status,
            sel_mm=check.samples_selection_mask_mismatch,
            phantom=check.samples_phantom_steps,
            untrimmed=check.samples_untrimmed_tail,
        )

        samples: list[dict] | None = None
        if need_full_samples:
            t0 = time.perf_counter()
            _, samples = load_trace_samples(path)
            _stage("load samples for accuracy", t0, n=len(samples) if samples else 0)

        t0 = time.perf_counter()
        metrics_cmp = _metrics_for_audit(
            name=name,
            check=check,
            meta=meta,
            metrics_csv=metrics_csv,
            compute_accuracy=compute_accuracy,
            samples=samples,
            pcfg_hf_dataset=pcfg_hf_dataset,
            pcfg_hf_namespace=pcfg_hf_namespace,
            token=token,
        )
        _stage(
            "metrics",
            t0,
            pcfg=name_is_pcfg,
            accuracy=compute_accuracy and not name_is_pcfg,
        )
    finally:
        if temp_dir is not None:
            remove_temp_download(temp_dir)

    row = check_to_row(check, extra={
        "name": name,
        "filename": filename,
        "source_format": source_format,
        "on_hub": True,
        "size_bytes": sizes.get(filename, check.size_bytes),
        "model": meta.get("model", ""),
        "task": meta.get("task", ""),
        "mode": meta.get("mode", ""),
        "steps_cfg": meta.get("steps", ""),
        **metrics_cmp,
    })
    annotated = annotate_audit_row(row)
    logger.info(
        "audit %s: transfer=%s is_bad=%s flags=%s total=%.1fs",
        filename,
        annotated.get("transfer_status"),
        annotated.get("is_bad"),
        annotated.get("audit_flags") or "",
        time.perf_counter() - t_all,
    )
    return annotated


def run_audit_all(
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    metrics_csv: str | Path | None = None,
    output_csv: str | Path | None = None,
    max_samples: Optional[int] = 3,
    compute_accuracy: bool = True,
    include_jsonl: bool = True,
    include_npz: bool = True,
    limit: Optional[int] = None,
    resume: bool = True,
    retry_errors: bool = True,
    pcfg_hf_dataset: str | None = None,
    pcfg_hf_namespace: str | None = None,
    token: bool | str | None = True,
) -> list[dict[str, Any]]:
    inv = hub_inventory(
        repo_id=repo_id,
        namespace=namespace,
        revision=revision,
        metrics_csv=metrics_csv,
        token=token,
    )
    cleanup_repo_cache(inv["repo"], revision=inv["revision"])
    filenames: list[str] = []
    if include_jsonl:
        filenames.extend(inv["jsonl_files"])
    if include_npz:
        filenames.extend(inv["npz_files"])
    filenames = sorted(filenames)
    if limit is not None:
        filenames = filenames[:limit]

    output_csv = Path(output_csv or DEFAULT_AUDIT_CSV)
    if compute_accuracy and metrics_csv and not inv.get("pcfg_only"):
        lookup = load_metrics_lookup(metrics_csv)
        tasks = {row.get("task", "") for row in lookup.values() if row.get("task")}
        prewarm_errors = prewarm_scoring_tasks(tasks)
        if prewarm_errors:
            logger.warning(
                "lm-eval prewarm incomplete (%d tasks failed): %s",
                len(prewarm_errors),
                ", ".join(sorted(prewarm_errors)),
            )

    if resume:
        rows, done = _load_existing_audit_rows(output_csv, retry_errors=retry_errors)
    else:
        rows, done = [], set()

    logger.info(
        "audit-all: %d files on %s (jsonl=%s npz=%s accuracy=%s)",
        len(filenames),
        inv["repo"],
        include_jsonl,
        include_npz,
        compute_accuracy,
    )
    t_batch = time.perf_counter()
    for i, filename in enumerate(filenames):
        if filename in done:
            continue
        if resume and retry_errors:
            rows = [row for row in rows if _audit_done_key(row) != filename]
        logger.info("[%d/%d] audit %s", i + 1, len(filenames), filename)
        try:
            row = audit_one_hub_file(
                filename,
                repo_id=inv["repo"],
                revision=inv["revision"],
                metrics_csv=metrics_csv,
                max_samples=max_samples,
                compute_accuracy=compute_accuracy,
                pcfg_hf_dataset=pcfg_hf_dataset,
                pcfg_hf_namespace=pcfg_hf_namespace,
                token=token,
            )
        except Exception as exc:
            logger.error("[%d/%d] FAILED %s: %s", i + 1, len(filenames), filename, exc)
            name = _stem(filename)
            meta = load_metrics_lookup(metrics_csv).get(name, {})
            row = annotate_audit_row({
                "name": name,
                "filename": filename,
                "source_format": "npz" if filename.endswith(".npz") else "jsonl",
                "on_hub": True,
                "model": meta.get("model", ""),
                "task": meta.get("task", ""),
                "mode": meta.get("mode", ""),
                "steps_cfg": meta.get("steps", ""),
                "issues": f"audit failed: {exc}",
            })
        rows.append(row)
        _write_csv_rows(rows, output_csv, AUDIT_CSV_FIELDS)
    n_bad = sum(1 for r in rows if r.get("is_bad") in (True, "True", "true", "1"))
    logger.info(
        "audit-all done: rows=%d bad=%d total=%.1fs -> %s",
        len(rows),
        n_bad,
        time.perf_counter() - t_batch,
        output_csv,
    )
    return rows


def convert_one_hub_file(
    filename: str,
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    output_repo_id: Optional[str] = None,
    output_namespace: Optional[str] = None,
    revision: Optional[str] = None,
    repair: bool = True,
    delete_jsonl_on_hub: bool = False,
    workers: int = 1,
    token: bool | str | None = True,
) -> dict[str, Any]:
    """
    Download JSONL -> repair+pack (default) -> validate -> upload NPZ -> cleanup.
    Optionally delete JSONL from the source Hub after successful validation.
    """
    source_repo = resolve_repo(repo_id, namespace)
    output_repo = _resolve_output_repo(
        repo_id=repo_id,
        namespace=namespace,
        output_repo_id=output_repo_id,
        output_namespace=output_namespace,
    )
    rev = revision if revision is not None else trace_hf_revision()
    name = _stem(filename)
    npz_name = f"{name}.npz"
    issues: list[str] = []
    t_all = time.perf_counter()

    temp_dir: Path | None = None
    try:
        logger.info("convert %s: download from %s", filename, source_repo)
        t0 = time.perf_counter()
        path, temp_dir = download_file_to_temp(
            filename, repo_id=source_repo, revision=rev, token=token
        )
        jsonl_bytes = path.stat().st_size
        _stage("download", t0, mb=f"{jsonl_bytes / 1e6:.1f}")

        task_name = read_jsonl_task_name(path)
        repaired = False
        ratio = 0.0
        npz_bytes = 0
        jsonl_deleted = False
        check: TraceCheck

        with tempfile.TemporaryDirectory(prefix="trace_npz_") as tmp:
            work = Path(tmp)
            npz_local = work / npz_name

            if repair:
                # Repair while packing — no intermediate repaired JSONL rewrite.
                t0 = time.perf_counter()
                packed = pack_repaired_jsonl_to_npz(
                    path,
                    npz_local,
                    task_name=task_name,
                    workers=workers,
                )
                _stage(
                    "repair+pack",
                    t0,
                    n=packed.n_samples,
                    repairs=len(packed.repairs),
                )
                repaired = bool(packed.repairs)
                if packed.repairs:
                    issues.extend(packed.repairs[:20])
                    if len(packed.repairs) > 20:
                        issues.append(f"... and {len(packed.repairs) - 20} more repairs")
                if packed.task_name is not None:
                    task_name = packed.task_name
                npz_bytes = npz_local.stat().st_size
                ratio = compression_ratio(path, npz_local)

                t0 = time.perf_counter()
                check = check_trace_npz_streaming(npz_local, max_samples=3)
                _stage(
                    "post-repair audit",
                    t0,
                    transfer=check.transfer_status,
                    phantom=check.samples_phantom_steps,
                    untrimmed=check.samples_untrimmed_tail,
                )

                repair_blockers = conversion_repair_blockers(
                    check,
                    None,
                    repair_enabled=True,
                )
                if repair_blockers:
                    issues.extend(repair_blockers)
                    logger.warning(
                        "convert %s: repair_blocked (%s)",
                        filename,
                        "; ".join(repair_blockers),
                    )
                    return _convert_failure_row(
                        name=name,
                        filename=filename,
                        npz_name=npz_name,
                        source_repo=source_repo,
                        output_repo=output_repo,
                        status="repair_blocked",
                        check=check,
                        repaired=repaired,
                        n_samples=check.n_samples,
                        jsonl_bytes=jsonl_bytes,
                        npz_bytes=npz_bytes,
                        compression_ratio=f"{ratio:.2f}",
                        issues=issues,
                    )

                t0 = time.perf_counter()
                roundtrip_issues = validate_npz_roundtrip_indexed(
                    packed.roundtrip_samples,
                    npz_local,
                    n_samples=packed.n_samples,
                )
            else:
                t0 = time.perf_counter()
                check = check_trace_jsonl_streaming(path, max_samples=3)
                _stage("pre-pack audit", t0, transfer=check.transfer_status)

                repair_blockers = conversion_repair_blockers(
                    check,
                    None,
                    repair_enabled=False,
                )
                if repair_blockers:
                    issues.extend(repair_blockers)
                    return _convert_failure_row(
                        name=name,
                        filename=filename,
                        npz_name=npz_name,
                        source_repo=source_repo,
                        output_repo=output_repo,
                        status="repair_blocked",
                        check=check,
                        repaired=False,
                        n_samples=check.n_samples,
                        jsonl_bytes=jsonl_bytes,
                        issues=issues,
                    )

                t0 = time.perf_counter()
                convert_jsonl_to_npz(path, npz_local, task_name=task_name)
                _stage("pack", t0, n=check.n_samples)
                ratio = compression_ratio(path, npz_local)
                npz_bytes = npz_local.stat().st_size

                t0 = time.perf_counter()
                roundtrip_issues = validate_npz_roundtrip(path, npz_local)

            roundtrip_ok = not roundtrip_issues
            _stage(
                "roundtrip",
                t0,
                ok=roundtrip_ok,
                issues=len(roundtrip_issues),
            )
            if roundtrip_issues:
                issues.extend(roundtrip_issues)

            if not roundtrip_ok:
                logger.warning("convert %s: roundtrip_failed", filename)
                return _convert_failure_row(
                    name=name,
                    filename=filename,
                    npz_name=npz_name,
                    source_repo=source_repo,
                    output_repo=output_repo,
                    status="roundtrip_failed",
                    check=check,
                    repaired=repaired,
                    n_samples=check.n_samples,
                    jsonl_bytes=jsonl_bytes,
                    npz_bytes=npz_bytes,
                    compression_ratio=f"{ratio:.2f}",
                    issues=issues,
                )

            if output_repo != source_repo:
                ensure_dataset_repo(
                    output_repo,
                    token=token,
                    private=trace_hf_dataset_private(),
                )

            t0 = time.perf_counter()
            upload_file(
                npz_local,
                npz_name,
                repo_id=output_repo,
                revision=rev,
                token=token,
                commit_message=f"NPZ trace {npz_name} ({npz_bytes} bytes)",
            )
            _stage("upload", t0, mb=f"{npz_bytes / 1e6:.1f}", repo=output_repo)

            if delete_jsonl_on_hub:
                t0 = time.perf_counter()
                delete_file(
                    filename,
                    repo_id=source_repo,
                    revision=rev,
                    token=token,
                    commit_message=f"remove JSONL after NPZ validated: {filename}",
                )
                jsonl_deleted = True
                _stage("delete-jsonl", t0)

        logger.info(
            "convert %s: ok n=%d ratio=%s total=%.1fs",
            filename,
            check.n_samples,
            f"{ratio:.2f}",
            time.perf_counter() - t_all,
        )
        return {
            "name": name,
            "jsonl_filename": filename,
            "npz_filename": npz_name,
            "source_repo": source_repo,
            "output_repo": output_repo,
            "status": "ok",
            "transfer_status": check.transfer_status,
            "repaired": repaired,
            "n_samples": check.n_samples,
            "jsonl_bytes": jsonl_bytes,
            "npz_bytes": npz_bytes,
            "compression_ratio": f"{ratio:.2f}",
            "roundtrip_ok": True,
            "jsonl_deleted_on_hub": jsonl_deleted,
            "issues": "; ".join(issues) if issues else "",
        }
    finally:
        if temp_dir is not None:
            remove_temp_download(temp_dir)


def run_convert_all(
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    output_repo_id: Optional[str] = None,
    output_namespace: Optional[str] = None,
    revision: Optional[str] = None,
    output_csv: str | Path | None = None,
    repair: bool = True,
    delete_jsonl_on_hub: bool = False,
    skip_existing_npz: bool = True,
    limit: Optional[int] = None,
    resume: bool = True,
    retry_errors: bool = True,
    workers: int = 1,
    token: bool | str | None = True,
) -> list[dict[str, Any]]:
    inv = hub_inventory(repo_id=repo_id, namespace=namespace, revision=revision, token=token)
    cleanup_repo_cache(inv["repo"], revision=inv["revision"])
    output_repo = _resolve_output_repo(
        repo_id=repo_id,
        namespace=namespace,
        output_repo_id=output_repo_id,
        output_namespace=output_namespace,
    )
    if output_repo != inv["repo"]:
        output_inv = hub_inventory(
            repo_id=output_repo_id or repo_id,
            namespace=output_namespace if output_namespace is not None else namespace,
            revision=revision,
            token=token,
        )
        present_npz = {f"{_stem(f)}.npz" for f in output_inv["npz_files"]}
    else:
        present_npz = {f"{_stem(f)}.npz" for f in inv["npz_files"]}
    filenames = inv["jsonl_files"]
    if limit is not None:
        filenames = filenames[:limit]

    output_csv = Path(output_csv or DEFAULT_CONVERT_CSV)
    if resume:
        rows, done = _load_existing_convert_rows(output_csv, retry_errors=retry_errors)
    else:
        rows, done = [], set()

    logger.info(
        "convert-all: %d jsonl on %s -> %s (resume=%s repair=%s)",
        len(filenames),
        inv["repo"],
        output_repo,
        resume,
        repair,
    )
    t_batch = time.perf_counter()
    n_ok = n_fail = n_skip = 0
    for i, filename in enumerate(filenames):
        name = _stem(filename)
        if name in done:
            n_skip += 1
            continue
        if resume and retry_errors:
            rows = [row for row in rows if row.get("name") != name]
        if skip_existing_npz and f"{name}.npz" in present_npz:
            logger.info("skip %s (npz already on %s)", name, output_repo)
            n_skip += 1
            continue
        logger.info("[%d/%d] convert %s -> %s", i + 1, len(filenames), filename, output_repo)
        try:
            row = convert_one_hub_file(
                filename,
                repo_id=inv["repo"],
                revision=inv["revision"],
                output_repo_id=output_repo_id,
                output_namespace=output_namespace,
                repair=repair,
                delete_jsonl_on_hub=delete_jsonl_on_hub,
                workers=workers,
                token=token,
            )
        except Exception as exc:
            logger.error("[%d/%d] FAILED %s: %s", i + 1, len(filenames), filename, exc)
            row = {
                "name": name,
                "jsonl_filename": filename,
                "npz_filename": f"{name}.npz",
                "source_repo": inv["repo"],
                "output_repo": output_repo,
                "status": "convert_failed",
                "transfer_status": "",
                "repaired": False,
                "n_samples": 0,
                "jsonl_bytes": 0,
                "npz_bytes": 0,
                "compression_ratio": "",
                "roundtrip_ok": False,
                "jsonl_deleted_on_hub": False,
                "issues": f"convert failed: {exc}",
            }
        if row.get("status") == "ok":
            n_ok += 1
        else:
            n_fail += 1
        rows.append(row)
        _write_csv_rows(rows, output_csv, CONVERT_CSV_FIELDS)
    logger.info(
        "convert-all done: ok=%d fail=%d skip=%d rows=%d total=%.1fs -> %s",
        n_ok,
        n_fail,
        n_skip,
        len(rows),
        time.perf_counter() - t_batch,
        output_csv,
    )
    return rows


def print_inventory(report: dict[str, Any]) -> int:
    print(f"repo: https://huggingface.co/datasets/{report['repo']}/tree/{report['revision']}")
    print(f"jsonl on hub: {report['jsonl_count']}")
    print(f"npz on hub: {report['npz_count']}")
    if report.get("other_files"):
        print(f"other files: {', '.join(report['other_files'])}")
    if report.get("pcfg_only"):
        print("inventory: PCFG repo (not compared to strace_metrics.csv)")
        print("status: ok")
        return 0
    print(f"expected variants (strace_metrics.csv): {report['expected_variants']}")
    print(f"missing jsonl: {len(report['missing_jsonl'])}")
    for n in report["missing_jsonl"]:
        print(f"  - {n}")
    print(f"extra jsonl: {len(report['extra_jsonl'])}")
    complete = (
        not report["missing_jsonl"]
        and not report["extra_jsonl"]
        and report["jsonl_count"] == report["expected_variants"]
    )
    print("status:", "complete" if complete else "incomplete")
    return 0 if complete else 2


def repair_one_hub_npz(
    filename: str,
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    output_repo_id: Optional[str] = None,
    output_namespace: Optional[str] = None,
    revision: Optional[str] = None,
    force: bool = False,
    workers: int = 1,
    token: bool | str | None = True,
) -> dict[str, Any]:
    """Download NPZ → repair (phantoms/mask-diff/trim) → validate → re-upload."""
    source_repo = resolve_repo(repo_id, namespace)
    output_repo = _resolve_output_repo(
        repo_id=repo_id,
        namespace=namespace,
        output_repo_id=output_repo_id,
        output_namespace=output_namespace,
    )
    rev = revision if revision is not None else trace_hf_revision()
    name = _stem(filename)
    if not filename.endswith(".npz"):
        filename = f"{name}.npz"
    issues: list[str] = []
    t_all = time.perf_counter()
    temp_dir: Path | None = None
    try:
        logger.info("repair-npz %s from %s", filename, source_repo)
        t0 = time.perf_counter()
        path, temp_dir = download_file_to_temp(
            filename, repo_id=source_repo, revision=rev, token=token
        )
        npz_bytes_in = path.stat().st_size
        _stage("download", t0, mb=f"{npz_bytes_in / 1e6:.1f}")
        t0 = time.perf_counter()
        check = check_trace_npz_streaming(path, max_samples=3)
        _stage(
            "pre-repair audit",
            t0,
            transfer=check.transfer_status,
            needs_repair=trace_needs_repair(check, None),
        )
        if not force and not trace_needs_repair(check, None):
            logger.info("repair-npz %s: skipped_clean (%.1fs)", filename, time.perf_counter() - t_all)
            return {
                "name": name,
                "npz_filename": filename,
                "source_repo": source_repo,
                "output_repo": output_repo,
                "status": "skipped_clean",
                "transfer_status": check.transfer_status,
                "repaired": False,
                "n_samples": check.n_samples,
                "npz_bytes_in": npz_bytes_in,
                "npz_bytes_out": npz_bytes_in,
                "roundtrip_ok": True,
                "issues": "",
            }

        with tempfile.TemporaryDirectory(prefix="trace_repair_npz_") as tmp:
            work = Path(tmp)
            out_local = work / filename
            t0 = time.perf_counter()
            n_samples, repairs, _task_name, repaired_samples = stream_repair_npz(
                path,
                out_local,
                task_name=read_npz_task_name(path),
                workers=workers,
            )
            _stage("repair+repack", t0, n=n_samples, repairs=len(repairs))
            if repairs:
                issues.extend(repairs[:20])
                if len(repairs) > 20:
                    issues.append(f"... and {len(repairs) - 20} more repairs")

            t0 = time.perf_counter()
            check_after = check_trace_npz_streaming(out_local, max_samples=3)
            _stage("post-repair audit", t0, transfer=check_after.transfer_status)
            blockers = conversion_repair_blockers(
                check_after, None, repair_enabled=True
            )
            if blockers:
                issues.extend(blockers)
                logger.warning("repair-npz %s: repair_blocked", filename)
                return {
                    "name": name,
                    "npz_filename": filename,
                    "source_repo": source_repo,
                    "output_repo": output_repo,
                    "status": "repair_blocked",
                    "transfer_status": check_after.transfer_status,
                    "repaired": bool(repairs),
                    "n_samples": n_samples,
                    "npz_bytes_in": npz_bytes_in,
                    "npz_bytes_out": out_local.stat().st_size,
                    "roundtrip_ok": False,
                    "issues": "; ".join(issues) if issues else "",
                }

            t0 = time.perf_counter()
            roundtrip_issues = validate_npz_roundtrip(repaired_samples, out_local)
            roundtrip_ok = not roundtrip_issues
            _stage("roundtrip", t0, ok=roundtrip_ok)
            if roundtrip_issues:
                issues.extend(roundtrip_issues[:20])
                return {
                    "name": name,
                    "npz_filename": filename,
                    "source_repo": source_repo,
                    "output_repo": output_repo,
                    "status": "roundtrip_failed",
                    "transfer_status": check_after.transfer_status,
                    "repaired": True,
                    "n_samples": n_samples,
                    "npz_bytes_in": npz_bytes_in,
                    "npz_bytes_out": out_local.stat().st_size,
                    "roundtrip_ok": False,
                    "issues": "; ".join(issues) if issues else "",
                }

            if output_repo != source_repo:
                ensure_dataset_repo(
                    output_repo,
                    token=token,
                    private=trace_hf_dataset_private(),
                )
            t0 = time.perf_counter()
            upload_file(
                out_local,
                filename,
                repo_id=output_repo,
                revision=rev,
                token=token,
                commit_message=f"NPZ repair {filename} (selection/mask-diff align)",
            )
            out_bytes = out_local.stat().st_size
            _stage("upload", t0, mb=f"{out_bytes / 1e6:.1f}", repo=output_repo)
            logger.info(
                "repair-npz %s: ok n=%d total=%.1fs",
                filename,
                n_samples,
                time.perf_counter() - t_all,
            )
            return {
                "name": name,
                "npz_filename": filename,
                "source_repo": source_repo,
                "output_repo": output_repo,
                "status": "ok",
                "transfer_status": check_after.transfer_status,
                "repaired": True,
                "n_samples": n_samples,
                "npz_bytes_in": npz_bytes_in,
                "npz_bytes_out": out_bytes,
                "roundtrip_ok": roundtrip_ok,
                "issues": "; ".join(issues) if issues else "",
            }
    finally:
        if temp_dir is not None:
            remove_temp_download(temp_dir)


def run_repair_npz_all(
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    output_repo_id: Optional[str] = None,
    output_namespace: Optional[str] = None,
    revision: Optional[str] = None,
    output_csv: str | Path | None = None,
    force: bool = False,
    limit: Optional[int] = None,
    resume: bool = True,
    retry_errors: bool = True,
    workers: int = 1,
    token: bool | str | None = True,
) -> list[dict[str, Any]]:
    """Repair Hub NPZs that need repair (or ``--force`` all)."""
    inv = hub_inventory(repo_id=repo_id, namespace=namespace, revision=revision, token=token)
    cleanup_repo_cache(inv["repo"], revision=inv["revision"])
    output_repo = _resolve_output_repo(
        repo_id=repo_id,
        namespace=namespace,
        output_repo_id=output_repo_id,
        output_namespace=output_namespace,
    )
    filenames = list(inv["npz_files"])
    if limit is not None:
        filenames = filenames[:limit]

    output_csv = Path(output_csv or DEFAULT_REPAIR_NPZ_CSV)
    if resume and output_csv.is_file():
        with open(output_csv, newline="", encoding="utf-8") as f:
            rows = [dict(row) for row in csv.DictReader(f)]
        done = {
            row["name"]
            for row in rows
            if row.get("name") and not (retry_errors and _should_retry_repair_npz_row(row))
        }
    else:
        rows, done = [], set()

    for i, filename in enumerate(filenames):
        name = _stem(filename)
        if name in done:
            continue
        if resume and retry_errors:
            rows = [row for row in rows if row.get("name") != name]
        logger.info("[%d/%d] repair-npz %s -> %s", i + 1, len(filenames), filename, output_repo)
        try:
            row = repair_one_hub_npz(
                filename,
                repo_id=inv["repo"],
                revision=inv["revision"],
                output_repo_id=output_repo_id,
                output_namespace=output_namespace,
                force=force,
                workers=workers,
                token=token,
            )
        except Exception as exc:
            logger.error("[%d/%d] FAILED %s: %s", i + 1, len(filenames), filename, exc)
            row = {
                "name": name,
                "npz_filename": filename if filename.endswith(".npz") else f"{name}.npz",
                "source_repo": inv["repo"],
                "output_repo": output_repo,
                "status": "repair_failed",
                "transfer_status": "",
                "repaired": False,
                "n_samples": 0,
                "npz_bytes_in": 0,
                "npz_bytes_out": 0,
                "roundtrip_ok": False,
                "issues": f"repair failed: {exc}",
            }
        rows.append(row)
        _write_csv_rows(rows, output_csv, REPAIR_NPZ_CSV_FIELDS)
    return rows


def print_check(check: TraceCheck) -> None:
    print(f"path: {check.path}")
    print(f"format: {check.format}")
    print(f"size: {check.size_bytes / 1e6:.2f} MB")
    print(f"samples: {check.n_samples} (checked {check.samples_checked})")
    print(f"transfer_status: {check.transfer_status} hooks_ok={check.hooks_ok}")
    if check.any_selection_mask_mismatch:
        print(
            f"selection_mask_mismatch: {check.samples_selection_mask_mismatch} "
            "checked sample(s) (repairable)"
        )
    if check.any_phantom_steps:
        print(
            f"phantom_steps: {check.samples_phantom_steps} "
            "checked sample(s) (repairable)"
        )
    if check.any_untrimmed_tail:
        print(
            f"untrimmed_tail: {check.samples_untrimmed_tail} "
            "checked sample(s) (repairable; all schedulers)"
        )
    for s in check.sample_checks:
        print(
            f"  sample {s.index}: steps={s.n_steps} sel_steps={s.steps_with_selected} "
            f"inferred={s.inferred_unmasks} sel_mismatch={s.selection_mask_mismatches} "
            f"phantom_extra={s.phantom_extra_steps} tail_trimmed={s.tail_trimmed} "
            f"proxy_ready={s.proxy_ready}"
        )
        if s.issues:
            print(f"    issues: {'; '.join(s.issues)}")
