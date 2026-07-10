"""One-file-at-a-time Hub trace audit and NPZ conversion pipelines."""

from __future__ import annotations

import csv
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

from traces.hf_trace_upload import (
    cleanup_cache,
    delete_file,
    download_file,
    file_sizes,
    list_repo_files,
    resolve_repo,
    trace_hf_revision,
    upload_file,
)
from traces.trace_checks import (
    TraceCheck,
    check_to_row,
    check_trace_file,
    repair_samples,
    validate_npz_roundtrip,
)
from traces.trace_metrics import (
    compare_with_metrics_row,
    expected_trace_names,
    load_metrics_lookup,
)
from traces.trace_npz import compression_ratio, convert_jsonl_to_npz, load_trace_samples

logger = logging.getLogger(__name__)

DEFAULT_METRICS_CSV = Path(__file__).resolve().parent.parent / "fetch_evals/strace_metrics.csv"
DEFAULT_AUDIT_CSV = Path(__file__).resolve().parent.parent / "fetch_evals/trace_audit.csv"
DEFAULT_CONVERT_CSV = Path(__file__).resolve().parent.parent / "fetch_evals/trace_convert.csv"


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

    expected = expected_trace_names(metrics_csv)
    expected_jsonl = {f"{n}.jsonl" for n in expected}
    expected_npz = {f"{n}.npz" for n in expected}

    return {
        "repo": full_repo,
        "revision": rev,
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
]

CONVERT_CSV_FIELDS = [
    "name",
    "jsonl_filename",
    "npz_filename",
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


def audit_one_hub_file(
    filename: str,
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    metrics_csv: str | Path | None = None,
    max_samples: Optional[int] = 3,
    compute_accuracy: bool = True,
    token: bool | str | None = True,
) -> dict[str, Any]:
    """Download one Hub JSONL, audit, compare metrics, cleanup cache."""
    full_repo = resolve_repo(repo_id, namespace)
    rev = revision if revision is not None else trace_hf_revision()
    name = _stem(filename)
    meta = load_metrics_lookup(metrics_csv).get(name, {})
    sizes = file_sizes([filename], repo_id=full_repo, revision=rev, token=token)

    path = download_file(filename, repo_id=full_repo, revision=rev, token=token)
    try:
        check = check_trace_file(path, max_samples=max_samples)
        _, samples = load_trace_samples(path)
        metrics_cmp = compare_with_metrics_row(
            run_name=name,
            samples=samples,
            metrics_lookup=load_metrics_lookup(metrics_csv),
            task=meta.get("task", ""),
            model=meta.get("model", ""),
            compute_accuracy=compute_accuracy,
        )
    finally:
        cleanup_cache(full_repo, filename, revision=rev)

    row = check_to_row(check, extra={
        "name": name,
        "filename": filename,
        "on_hub": True,
        "size_bytes": sizes.get(filename, check.size_bytes),
        "model": meta.get("model", ""),
        "task": meta.get("task", ""),
        "mode": meta.get("mode", ""),
        "steps_cfg": meta.get("steps", ""),
        **metrics_cmp,
    })
    return row


def run_audit_all(
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    metrics_csv: str | Path | None = None,
    output_csv: str | Path | None = None,
    max_samples: Optional[int] = 3,
    compute_accuracy: bool = True,
    limit: Optional[int] = None,
    resume: bool = True,
    token: bool | str | None = True,
) -> list[dict[str, Any]]:
    inv = hub_inventory(
        repo_id=repo_id,
        namespace=namespace,
        revision=revision,
        metrics_csv=metrics_csv,
        token=token,
    )
    filenames = inv["jsonl_files"]
    if limit is not None:
        filenames = filenames[:limit]

    output_csv = Path(output_csv or DEFAULT_AUDIT_CSV)
    done = _load_done_names(output_csv) if resume else set()
    rows: list[dict[str, Any]] = []
    if resume and output_csv.exists():
        with open(output_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    for i, filename in enumerate(filenames):
        name = _stem(filename)
        if name in done:
            continue
        logger.info("[%d/%d] audit %s", i + 1, len(filenames), filename)
        row = audit_one_hub_file(
            filename,
            repo_id=inv["repo"],
            revision=inv["revision"],
            metrics_csv=metrics_csv,
            max_samples=max_samples,
            compute_accuracy=compute_accuracy,
            token=token,
        )
        rows.append(row)
        _write_csv_rows(rows, output_csv, AUDIT_CSV_FIELDS)
    return rows


def convert_one_hub_file(
    filename: str,
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    repair: bool = True,
    delete_jsonl_on_hub: bool = False,
    token: bool | str | None = True,
) -> dict[str, Any]:
    """
    Download JSONL -> audit -> optional repair -> NPZ -> validate -> upload NPZ -> cleanup.
    Optionally delete JSONL from Hub after successful validation.
    """
    full_repo = resolve_repo(repo_id, namespace)
    rev = revision if revision is not None else trace_hf_revision()
    name = _stem(filename)
    npz_name = f"{name}.npz"
    issues: list[str] = []

    path = download_file(filename, repo_id=full_repo, revision=rev, token=token)
    jsonl_bytes = path.stat().st_size
    try:
        check = check_trace_file(path, max_samples=None)
        task_name, samples = load_trace_samples(path)
        repaired = False
        repairs: list[str] = []

        needs_hook_repair = check.transfer_status in ("inferrable", "explicit_partial", "mixed")
        needs_order_repair = any(
            sample.get("step_traces") and not sample.get("unmasking_order") for sample in samples
        )
        if repair and (needs_hook_repair or needs_order_repair):
            samples, repairs = repair_samples(samples)
            if repairs:
                repaired = True
                issues.extend(repairs[:20])
                if len(repairs) > 20:
                    issues.append(f"... and {len(repairs) - 20} more repairs")

        with tempfile.TemporaryDirectory(prefix="trace_npz_") as tmp:
            work = Path(tmp)
            if repaired:
                jsonl_work = work / filename
                with open(jsonl_work, "w", encoding="utf-8") as f:
                    if task_name:
                        f.write(f"# task={json.dumps(task_name)}\n")
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                src = jsonl_work
            else:
                src = path

            npz_local = work / npz_name
            convert_jsonl_to_npz(src, npz_local)
            ratio = compression_ratio(src, npz_local)
            npz_bytes = npz_local.stat().st_size

            roundtrip_issues = validate_npz_roundtrip(samples, npz_local)
            roundtrip_ok = not roundtrip_issues
            if roundtrip_issues:
                issues.extend(roundtrip_issues)

            if not roundtrip_ok:
                return {
                    "name": name,
                    "jsonl_filename": filename,
                    "npz_filename": npz_name,
                    "status": "roundtrip_failed",
                    "transfer_status": check.transfer_status,
                    "repaired": repaired,
                    "n_samples": len(samples),
                    "jsonl_bytes": jsonl_bytes,
                    "npz_bytes": npz_bytes,
                    "compression_ratio": f"{ratio:.2f}",
                    "roundtrip_ok": False,
                    "jsonl_deleted_on_hub": False,
                    "issues": "; ".join(issues),
                }

            upload_file(
                npz_local,
                npz_name,
                repo_id=full_repo,
                revision=rev,
                token=token,
                commit_message=f"NPZ trace {npz_name} ({npz_bytes} bytes)",
            )

            jsonl_deleted = False
            if delete_jsonl_on_hub:
                delete_file(
                    filename,
                    repo_id=full_repo,
                    revision=rev,
                    token=token,
                    commit_message=f"remove JSONL after NPZ validated: {filename}",
                )
                jsonl_deleted = True

        return {
            "name": name,
            "jsonl_filename": filename,
            "npz_filename": npz_name,
            "status": "ok",
            "transfer_status": check.transfer_status,
            "repaired": repaired,
            "n_samples": len(samples),
            "jsonl_bytes": jsonl_bytes,
            "npz_bytes": npz_bytes,
            "compression_ratio": f"{ratio:.2f}",
            "roundtrip_ok": True,
            "jsonl_deleted_on_hub": jsonl_deleted,
            "issues": "; ".join(issues) if issues else "",
        }
    finally:
        cleanup_cache(full_repo, filename, revision=rev)
        cleanup_cache(full_repo, npz_name, revision=rev)


def run_convert_all(
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    output_csv: str | Path | None = None,
    repair: bool = True,
    delete_jsonl_on_hub: bool = False,
    skip_existing_npz: bool = True,
    limit: Optional[int] = None,
    resume: bool = True,
    token: bool | str | None = True,
) -> list[dict[str, Any]]:
    inv = hub_inventory(repo_id=repo_id, namespace=namespace, revision=revision, token=token)
    present_npz = {f"{_stem(f)}.npz" for f in inv["npz_files"]}
    filenames = inv["jsonl_files"]
    if limit is not None:
        filenames = filenames[:limit]

    output_csv = Path(output_csv or DEFAULT_CONVERT_CSV)
    done = _load_done_names(output_csv) if resume else set()
    rows: list[dict[str, Any]] = []
    if resume and output_csv.exists():
        with open(output_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    for i, filename in enumerate(filenames):
        name = _stem(filename)
        if name in done:
            continue
        if skip_existing_npz and f"{name}.npz" in present_npz:
            logger.info("skip %s (npz already on hub)", name)
            continue
        logger.info("[%d/%d] convert %s", i + 1, len(filenames), filename)
        row = convert_one_hub_file(
            filename,
            repo_id=inv["repo"],
            revision=inv["revision"],
            repair=repair,
            delete_jsonl_on_hub=delete_jsonl_on_hub,
            token=token,
        )
        rows.append(row)
        _write_csv_rows(rows, output_csv, CONVERT_CSV_FIELDS)
    return rows


def print_inventory(report: dict[str, Any]) -> int:
    print(f"repo: https://huggingface.co/datasets/{report['repo']}/tree/{report['revision']}")
    print(f"expected variants: {report['expected_variants']}")
    print(f"jsonl on hub: {report['jsonl_count']}")
    print(f"npz on hub: {report['npz_count']}")
    if report["other_files"]:
        print(f"other files: {', '.join(report['other_files'])}")
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


def print_check(check: TraceCheck) -> None:
    print(f"path: {check.path}")
    print(f"format: {check.format}")
    print(f"size: {check.size_bytes / 1e6:.2f} MB")
    print(f"samples: {check.n_samples} (checked {check.samples_checked})")
    print(f"transfer_status: {check.transfer_status} hooks_ok={check.hooks_ok}")
    for s in check.sample_checks:
        print(
            f"  sample {s.index}: steps={s.n_steps} sel_steps={s.steps_with_selected} "
            f"inferred={s.inferred_unmasks} proxy_ready={s.proxy_ready}"
        )
        if s.issues:
            print(f"    issues: {'; '.join(s.issues)}")
