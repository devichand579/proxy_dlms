"""Download SchedulerLens NPZs, stream each into a metrics CSV, and delete the temp copy."""
from __future__ import annotations

import csv
import fnmatch
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from schedulerlens.completion import (
    compute_completion_metrics,
    duplicate_run_stems,
    join_rows_with_strace,
)
from schedulerlens.metrics import (
    CORR_PREFIXES,
    build_metrics_row,
    compute_task_metrics,
    expand_prefix_metric_rows,
    parse_prefix_steps,
    parse_run_stem,
    prefix_pearsons,
    read_metrics_csv,
    write_metrics_csv,
)
from schedulerlens.oracle import (
    compute_task_perplexity,
    empty_task_perplexity,
    index_oracle_artifacts,
    resolve_oracle_npz,
    task_perplexity_from_oracle_npz,
    load_oracle_artifacts,
)
from schedulerlens.proxies import TraceProxyError
from traces.hf_trace_upload import (
    cleanup_repo_cache,
    download_file,
    download_file_to_temp,
    list_repo_files,
    remove_temp_download,
    resolve_repo,
    trace_hf_revision,
)

logger = logging.getLogger(__name__)

DEFAULT_METRICS_CSV = Path(__file__).resolve().parent.parent / "outputs/schedulerlens/compare/schedulerlens_metrics.csv"

# Re-export Hub primitives so callers import from one place.
__all__ = [
    "DEFAULT_METRICS_CSV",
    "analyze_one_npz",
    "cleanup_repo_cache",
    "discover_hub_npz",
    "discover_local_npz",
    "download_file",
    "download_file_to_temp",
    "filter_by_meta",
    "hub_inventory",
    "list_repo_files",
    "print_inventory",
    "remove_temp_download",
    "resolve_repo",
    "run_compare",
    "run_hub_compare",
    "should_replay",
    "trace_hf_revision",
]


def discover_local_npz(local_dir: Path, pattern: str = "*.npz") -> List[Path]:
    files = sorted(Path(local_dir).glob("*.npz"))
    return [p for p in files if fnmatch.fnmatch(p.name, pattern)]


def discover_hub_npz(
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    pattern: str = "*.npz",
    token: bool | str | None = True,
) -> List[str]:
    """Hub NPZ paths whose filename matches ``pattern``."""
    files = list_repo_files(
        repo_id=repo_id, namespace=namespace, revision=revision, token=token
    )
    npz = [
        f
        for f in files
        if f.endswith(".npz") and fnmatch.fnmatch(Path(f).name, pattern)
    ]
    repo = resolve_repo(repo_id, namespace)
    logger.info("Hub %s: %d NPZ matching %r", repo, len(npz), pattern)
    return sorted(npz)


def _name_set(spec: Optional[Union[str, Sequence[str]]]) -> set[str]:
    if spec is None:
        return set()
    if isinstance(spec, str):
        parts = spec.split(",")
    else:
        parts = spec
    return {str(p).strip() for p in parts if str(p).strip()}


def filter_by_meta(
    filenames: Sequence[str],
    *,
    model: Optional[str] = None,
    scheduler: Optional[str] = None,
    steps: Optional[int] = None,
    pcfg_only: bool = False,
    exclude_scheduler: Optional[Union[str, Sequence[str]]] = None,
    exclude_task: Optional[Union[str, Sequence[str]]] = None,
) -> List[str]:
    drop_sched = _name_set(exclude_scheduler)
    drop_task = _name_set(exclude_task)
    out: List[str] = []
    for name in filenames:
        meta = parse_run_stem(name)
        if pcfg_only and not meta.is_pcfg:
            continue
        if model is not None and meta.model != model:
            continue
        if scheduler is not None and meta.scheduler != scheduler:
            continue
        if steps is not None and meta.steps != steps:
            continue
        if meta.scheduler in drop_sched:
            continue
        if meta.task in drop_task:
            continue
        out.append(name)
    return out


def hub_inventory(
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    pattern: str = "*.npz",
    model: Optional[str] = None,
    scheduler: Optional[str] = None,
    steps: Optional[int] = None,
    pcfg_only: bool = False,
    exclude_scheduler: Optional[Union[str, Sequence[str]]] = None,
    exclude_task: Optional[Union[str, Sequence[str]]] = None,
    token: bool | str | None = True,
) -> dict[str, Any]:
    """Matching NPZ filenames, without downloading them."""
    full_repo = resolve_repo(repo_id, namespace)
    rev = revision if revision is not None else trace_hf_revision()
    files = list_repo_files(repo_id=full_repo, revision=rev, token=token)
    all_npz = sorted(f for f in files if f.endswith(".npz"))
    matched = [
        f for f in all_npz if fnmatch.fnmatch(Path(f).name, pattern)
    ]
    matched = filter_by_meta(
        matched,
        model=model,
        scheduler=scheduler,
        steps=steps,
        pcfg_only=pcfg_only,
        exclude_scheduler=exclude_scheduler,
        exclude_task=exclude_task,
    )
    pcfg = [f for f in matched if parse_run_stem(f).is_pcfg]
    return {
        "repo": full_repo,
        "revision": rev,
        "pattern": pattern,
        "npz_total": len(all_npz),
        "npz_matched": len(matched),
        "npz_pcfg": len(pcfg),
        "other_files": sorted(
            f for f in files if not f.endswith((".npz", ".jsonl"))
        ),
        "jsonl_count": sum(1 for f in files if f.endswith(".jsonl")),
        "npz_files": matched,
        "pcfg_files": pcfg,
    }


def print_inventory(inv: dict[str, Any]) -> None:
    print(f"repo={inv['repo']}  revision={inv['revision']}")
    print(
        f"NPZ total={inv['npz_total']}  matched={inv['npz_matched']}  "
        f"pcfg={inv['npz_pcfg']}  jsonl={inv['jsonl_count']}"
    )
    for name in inv.get("npz_files") or []:
        meta = parse_run_stem(name)
        tag = "pcfg" if meta.is_pcfg else meta.task
        print(
            f"  {Path(name).name}  model={meta.model}  sched={meta.scheduler}  "
            f"steps={meta.steps}  task={tag}"
        )


def should_replay(*, meta_is_pcfg: bool, force_pcfg: bool, replay_oracle: bool) -> bool:
    if not replay_oracle:
        return False
    return meta_is_pcfg or force_pcfg


def analyze_one_npz(
    trace_path: Path,
    *,
    repo: str = "",
    max_samples: Optional[int] = None,
    bucket_size: Optional[int] = None,
    replay_oracle: bool = False,
    force_pcfg: bool = False,
    model=None,
    device=None,
    engine=None,
    vocab=None,
    verbose: bool = True,
    oracle_workers: int = 1,
    step_workers: Optional[int] = None,
    forward_batch_size: int = 8,
    jt_exact_max: Optional[int] = None,
    jt_markov_order: Optional[int] = None,
    jt_large_set: Optional[int] = None,
    strace_metrics_path: Optional[Union[str, Path]] = None,
    oracle_npz: Optional[Union[str, Path]] = None,
    require_cached_oracle: bool = False,
    prefix_steps: Optional[Sequence[int]] = None,
    metrics_only: bool = False,
    skip_completion: bool = False,
    skip_perplexity: bool = False,
) -> Dict[str, Any]:
    """Metrics rows for one local NPZ. Oracle columns come from a cached file or a live PCFG replay."""
    meta = parse_run_stem(trace_path)
    cached_oracle = Path(oracle_npz) if oracle_npz is not None else None
    do_replay = (
        cached_oracle is None
        and should_replay(
            meta_is_pcfg=meta.is_pcfg,
            force_pcfg=force_pcfg,
            replay_oracle=replay_oracle,
        )
    )
    prefix_ks = parse_prefix_steps(prefix_steps)
    if (meta.is_pcfg or force_pcfg) and not prefix_ks:
        prefix_ks = CORR_PREFIXES
    skip_completion = bool(metrics_only or skip_completion)
    skip_perplexity = bool(metrics_only or skip_perplexity)

    metrics = compute_task_metrics(
        trace_path,
        max_samples=max_samples,
        bucket_size=bucket_size,
        verbose=verbose,
        prefix_steps=prefix_ks or None,
    )

    completion = None
    if not skip_completion:
        steps_budget = int(meta.steps) if meta.steps is not None else int(metrics.n_steps)
        completion = compute_completion_metrics(
            trace_path,
            model=meta.model,
            steps_budget=steps_budget,
            max_samples=max_samples,
        )

    if cached_oracle is not None:
        logger.info("Using cached oracle %s for %s", cached_oracle, meta.stem)
        ppl = task_perplexity_from_oracle_npz(cached_oracle)
    elif do_replay:
        ppl = compute_task_perplexity(
            trace_path,
            engine=engine,
            vocab=vocab,
            model=model,
            device=device,
            replay_oracle=True,
            max_samples=max_samples,
            verbose=verbose,
            oracle_workers=oracle_workers,
            step_workers=step_workers,
            forward_batch_size=forward_batch_size,
            jt_exact_max=jt_exact_max,
            jt_markov_order=jt_markov_order,
            jt_large_set=jt_large_set,
        )
    elif require_cached_oracle or skip_perplexity:
        if require_cached_oracle and not skip_perplexity:
            logger.warning(
                "No cached oracle.npz for stem %r (and --replay-oracle off); "
                "leaving Mt/Jt/PPL/corr empty",
                meta.stem,
            )
        ppl = empty_task_perplexity(n_samples=int(metrics.n_samples))
    else:
        ppl = compute_task_perplexity(
            trace_path,
            engine=engine,
            vocab=vocab,
            model=None,
            device=device,
            replay_oracle=False,
            max_samples=max_samples,
            verbose=verbose,
            oracle_workers=oracle_workers,
            step_workers=step_workers,
            forward_batch_size=forward_batch_size,
            jt_exact_max=jt_exact_max,
            jt_markov_order=jt_markov_order,
            jt_large_set=jt_large_set,
        )

    prefix_pearson = None
    use_oracle = cached_oracle is not None or do_replay
    if use_oracle:
        if cached_oracle is not None:
            arts = load_oracle_artifacts(cached_oracle)
            step_sample_idx = arts.get("step_sample_idx", [])
            step = arts.get("step", [])
            step_mt = arts.get("step_Mt", [])
            step_jt = arts.get("step_Jt", [])
        else:
            tables = getattr(ppl, "oracle_tables", None) or {}
            step_sample_idx = tables.get("step_sample_idx", [])
            step = tables.get("step", [])
            step_mt = tables.get("step_Mt", [])
            step_jt = tables.get("step_Jt", [])
        if len(step):
            prefix_pearson = prefix_pearsons(
                metrics,
                step_sample_idx=step_sample_idx,
                step=step,
                step_mt=step_mt,
                step_jt=step_jt,
            )

    row = build_metrics_row(
        trace_path,
        metrics=metrics,
        perplexity=ppl,
        prefix_pearson=prefix_pearson,
        meta=meta,
        repo=repo,
        completion=completion,
    )
    row = join_rows_with_strace([row], strace_path=strace_metrics_path)[0]
    rows = expand_prefix_metric_rows(row, metrics, prefix_pearson)

    return {
        "row": row,
        "rows": rows,
        "metrics": metrics,
        "perplexity": ppl,
        "prefix_pearson": prefix_pearson,
        "meta": meta,
        "completion": completion,
        "oracle_npz": str(cached_oracle.resolve()) if cached_oracle else None,
    }


def resolve_pcfg_model_dir(
    *,
    model_path: Optional[Union[str, Path]] = None,
    hf_model: Optional[str] = None,
    token: bool | str | None = True,
) -> Optional[Path]:
    """Local checkpoint directory, downloading an HF model id when that is what was given."""
    if model_path is not None:
        path = Path(model_path)
        if path.is_dir():
            return path
        # Allow passing an HF id via --model-path when it is not a local directory.
        text = str(model_path).strip()
        if "/" in text and not path.exists():
            hf_model = text
        else:
            raise FileNotFoundError(f"model path not found: {path}")
    if not hf_model:
        return None
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(hf_model, token=token))
    logger.info("Downloaded PCFG model %s → %s", hf_model, local)
    return local


def _release_model(model) -> None:
    """Drop the model and free the CUDA cache."""
    if model is None:
        return
    try:
        import gc

        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def hf_model_id_for_meta(
    meta,
    *,
    hf_model: Optional[str] = None,
    namespace: Optional[str] = None,
) -> Optional[str]:
    """HF repo id for this trace: the explicit override, or ``{namespace}/{model}``."""
    if hf_model:
        return str(hf_model).strip()
    short = (getattr(meta, "model", None) or "").strip()
    if not short or not getattr(meta, "is_pcfg", False):
        return None
    if "/" in short:
        return short
    ns = (namespace or "").strip()
    if not ns:
        try:
            from traces.hf_trace_upload import normalize_dataset_repo_id

            # Reuse namespace resolution (env / whoami) via a dummy repo name.
            full = normalize_dataset_repo_id("x", namespace=None)
            ns = full.split("/", 1)[0]
        except Exception:
            ns = ""
    if not ns:
        logger.warning(
            "Cannot resolve HF namespace for model %r; set --namespace or --hf-model",
            short,
        )
        return None
    return f"{ns}/{short}"


def _load_oracle_stack(
    *,
    replay_oracle: bool,
    model_path: Optional[Union[str, Path]] = None,
    hf_model: Optional[str] = None,
):
    """PCFG engine, and one fixed checkpoint when a model path is given."""
    from schedulerlens.oracle import _ensure_pcfg_path

    _ensure_pcfg_path()
    engine = vocab = model = device = None
    if not replay_oracle:
        return engine, vocab, model, device
    try:
        from pcfg_data import build_stack

        _g, _cnf, engine, vocab = build_stack()
    except Exception as e:
        logger.warning("PCFG stack unavailable (%s); oracle metrics limited", e)
        engine = vocab = None

    if model_path is None and hf_model is None:
        return engine, vocab, model, device

    try:
        ckpt = resolve_pcfg_model_dir(model_path=model_path, hf_model=hf_model)
    except Exception as e:
        logger.warning("Could not resolve PCFG model (%s); Mt/Jt replay skipped", e)
        ckpt = None

    if ckpt is not None:
        try:
            import torch
            from models.pcfg_models import load_pcfg_llada_checkpoint

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, _cfg, vocab = load_pcfg_llada_checkpoint(ckpt, device=str(device))
            logger.info("Loaded PCFG model from %s on %s", ckpt, device)
        except Exception as e:
            logger.warning(
                "Could not load PCFG model from %s (%s); Mt/Jt replay skipped",
                ckpt,
                e,
            )
            model = None
    return engine, vocab, model, device


class _LazyOracleModels:
    """One PCFG checkpoint at a time, unloaded when the trace model changes."""

    def __init__(
        self,
        *,
        engine,
        vocab,
        device,
        model_path: Optional[Union[str, Path]] = None,
        hf_model: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        self.engine = engine
        self.vocab = vocab
        self.device = device
        self.model_path = model_path
        self.hf_model = hf_model
        self.namespace = namespace
        self._model = None
        self._loaded_key: Optional[str] = None

    def for_meta(self, meta):
        """Model, vocab, and device for this trace."""
        if self.model_path is not None or self.hf_model is not None:
            # Fixed override — already loaded (or failed) once via _load_oracle_stack.
            return self._model, self.vocab, self.device

        key = hf_model_id_for_meta(meta, hf_model=None, namespace=self.namespace)
        if key is None:
            return None, self.vocab, self.device
        if key == self._loaded_key:
            return self._model, self.vocab, self.device

        if self._model is not None:
            logger.info("Unloading PCFG model %s", self._loaded_key)
            _release_model(self._model)
            self._model = None
            self._loaded_key = None

        try:
            import torch
            from models.pcfg_models import load_pcfg_llada_checkpoint

            if self.device is None:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            ckpt = resolve_pcfg_model_dir(hf_model=key)
            self._model, _cfg, self.vocab = load_pcfg_llada_checkpoint(
                ckpt, device=str(self.device)
            )
            self._loaded_key = key
            logger.info("Lazy-loaded PCFG model %s on %s", key, self.device)
        except Exception as e:
            logger.warning("Lazy load failed for %s (%s); Mt/Jt skipped", key, e)
            self._model = None
            self._loaded_key = None
        return self._model, self.vocab, self.device

    def close(self) -> None:
        _release_model(self._model)
        self._model = None
        self._loaded_key = None


def _prefix_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number == int(number):
        return str(int(number))
    return text


def _load_done_stems(
    csv_path: Path,
    prefix_steps: Optional[Sequence[int]] = None,
) -> set[str]:
    """Stems that already have the full row and every requested prefix.

    A stem with only some of those rows is left out, so the next pass
    replaces it.
    """
    if not csv_path.is_file():
        return set()
    required = {str(int(k)) for k in (prefix_steps or [])}
    present: dict[str, set[str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("filename") or row.get("name") or "").strip()
            if not name:
                continue
            present.setdefault(Path(name).stem, set()).add(
                _prefix_label(row.get("prefix_steps"))
            )
    done = {
        stem
        for stem, labels in present.items()
        if "" in labels and required <= labels
    }
    partial = len(present) - len(done)
    if partial:
        logger.info(
            "Resume: %d stems already have every prefix; %d are missing a prefix and will be redone",
            len(done),
            partial,
        )
    elif done:
        logger.info("Resume: skipping %d stems already in %s", len(done), csv_path)
    return done


def run_compare(
    *,
    output_dir: Union[str, Path],
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    local_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.npz",
    model_filter: Optional[str] = None,
    scheduler_filter: Optional[str] = None,
    steps_filter: Optional[int] = None,
    pcfg_only: bool = False,
    exclude_scheduler: Optional[Union[str, Sequence[str]]] = None,
    exclude_task: Optional[Union[str, Sequence[str]]] = None,
    force_pcfg: bool = False,
    replay_oracle: bool = False,
    model_path: Optional[Union[str, Path]] = None,
    hf_model: Optional[str] = None,
    max_samples: Optional[int] = None,
    limit: Optional[int] = None,
    resume: bool = True,
    purge_cache: bool = True,
    verbose: bool = True,
    oracle_workers: int = 1,
    step_workers: Optional[int] = None,
    forward_batch_size: int = 8,
    jt_exact_max: Optional[int] = None,
    jt_markov_order: Optional[int] = None,
    jt_large_set: Optional[int] = None,
    strace_metrics_path: Optional[Union[str, Path]] = None,
    skip_duplicate_budgets: bool = True,
    oracle_dir: Optional[Union[str, Path]] = None,
    prefix_steps: Optional[Sequence[int]] = None,
    metrics_only: bool = False,
    skip_completion: bool = False,
    skip_perplexity: bool = False,
    csv_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze each discovered NPZ and write the compare CSV.

    A normal batch writes ``schedulerlens_metrics.csv``. A PCFG batch writes
    ``schedulerlens_metrics_pcfg.csv``.
    """
    from schedulerlens.oracle import _ensure_pcfg_path

    _ensure_pcfg_path()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_ks = parse_prefix_steps(prefix_steps)

    oracle_index: dict[str, Path] = {}
    if oracle_dir is not None:
        oracle_index = index_oracle_artifacts(oracle_dir)
        logger.info(
            "Indexed %d oracle artifact(s) under %s",
            len(oracle_index),
            Path(oracle_dir),
        )

    fixed_override = model_path is not None or hf_model is not None
    need_live_replay = bool(replay_oracle)
    engine = vocab = model = device = None
    lazy: Optional[_LazyOracleModels] = None
    if need_live_replay:
        engine, vocab, model, device = _load_oracle_stack(
            replay_oracle=True,
            model_path=model_path if fixed_override else None,
            hf_model=hf_model if fixed_override else None,
        )
        if engine is None:
            try:
                from pcfg_data import build_stack

                _g, _cnf, engine, vocab = build_stack()
            except Exception as e:
                logger.warning("PCFG stack unavailable (%s)", e)

        if not fixed_override:
            if device is None:
                try:
                    import torch

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                except Exception:
                    device = "cpu"
            lazy = _LazyOracleModels(
                engine=engine,
                vocab=vocab,
                device=device,
                namespace=namespace,
            )
        else:
            lazy = _LazyOracleModels(
                engine=engine,
                vocab=vocab,
                device=device,
                model_path=model_path,
                hf_model=hf_model,
                namespace=namespace,
            )
            lazy._model = model
            lazy._loaded_key = str(hf_model or model_path) if model is not None else None

    rows: List[Dict[str, Any]] = []
    repo_label = ""

    if local_dir is not None:
        local_dir = Path(local_dir)
        paths = discover_local_npz(local_dir, pattern)
        names = filter_by_meta(
            [p.name for p in paths],
            model=model_filter,
            scheduler=scheduler_filter,
            steps=steps_filter,
            pcfg_only=pcfg_only,
            exclude_scheduler=exclude_scheduler,
            exclude_task=exclude_task,
        )
        path_map = {p.name: p for p in paths}
        repo_label = str(local_dir.resolve())
        work_items = [(name, path_map[name], None) for name in names]
    else:
        repo_label = resolve_repo(repo_id, namespace)
        if purge_cache:
            try:
                freed = cleanup_repo_cache(repo_label, revision=revision)
                if freed:
                    logger.info("Purged %.1f MiB of stale Hub cache for %s", freed / 1e6, repo_label)
            except Exception as e:
                logger.debug("cache purge skipped: %s", e)

        hub_files = discover_hub_npz(
            repo_id=repo_id,
            namespace=namespace,
            revision=revision,
            pattern=pattern,
        )
        hub_files = filter_by_meta(
            hub_files,
            model=model_filter,
            scheduler=scheduler_filter,
            steps=steps_filter,
            pcfg_only=pcfg_only,
            exclude_scheduler=exclude_scheduler,
            exclude_task=exclude_task,
        )
        work_items = [(name, None, name) for name in hub_files]

    pcfg_sheet = bool(pcfg_only or force_pcfg) or (
        bool(work_items)
        and all(parse_run_stem(name).is_pcfg for name, _, _ in work_items)
    )
    csv_name = (
        csv_name
        if csv_name
        else (
            "schedulerlens_metrics_pcfg.csv"
            if pcfg_sheet
            else "schedulerlens_metrics.csv"
        )
    )
    given = Path(csv_name)
    csv_path = given if given.is_absolute() else output_dir / given

    done_stems: set[str] = set()
    if resume and csv_path.is_file():
        required_prefixes = list(prefix_ks)
        if pcfg_sheet and not required_prefixes:
            required_prefixes = list(CORR_PREFIXES)
        done_stems = _load_done_stems(csv_path, required_prefixes)

    if skip_duplicate_budgets:
        dropped = duplicate_run_stems(strace_metrics_path)
        before = len(work_items)
        work_items = [
            (n, lp, hn)
            for n, lp, hn in work_items
            if Path(n).stem not in dropped
        ]
        logger.info(
            "Skipped %d traces whose NFE repeats a lower step budget",
            before - len(work_items),
        )

    if done_stems:
        work_items = [
            (n, lp, hn)
            for n, lp, hn in work_items
            if Path(n).stem not in done_stems
        ]

    # Group by parsed model so lazy HF loads swap as little as possible.
    work_items.sort(key=lambda it: parse_run_stem(it[0]).model)

    if limit is not None:
        work_items = work_items[:limit]

    if not work_items and not (resume and csv_path.is_file()):
        logger.warning("No NPZ traces matched filters")
        write_metrics_csv(csv_path, [], pcfg=pcfg_sheet)
        return {"rows": [], "csv_path": str(csv_path)}

    try:
        for idx, (name, local_path, hub_name) in enumerate(work_items):
            temp_dir = None
            try:
                if local_path is not None:
                    trace_path = Path(local_path)
                else:
                    trace_path, temp_dir = download_file_to_temp(
                        hub_name,
                        repo_id=repo_id,
                        namespace=namespace,
                        revision=revision,
                    )

                meta = parse_run_stem(name)
                stem = Path(name).stem
                cached = resolve_oracle_npz(None, stem, index=oracle_index)
                do_replay = bool(replay_oracle and cached is None)
                cur_model, cur_vocab, cur_device = model, vocab, device
                if lazy is not None and do_replay:
                    cur_model, cur_vocab, cur_device = lazy.for_meta(meta)

                logger.info(
                    "[%d/%d] Analyzing %s (model=%s%s)",
                    idx + 1,
                    len(work_items),
                    name,
                    meta.model,
                    f", oracle={cached.name}" if cached is not None else "",
                )
                result = analyze_one_npz(
                    Path(trace_path),
                    repo=repo_label,
                    max_samples=max_samples,
                    replay_oracle=do_replay,
                    force_pcfg=force_pcfg,
                    model=cur_model,
                    device=cur_device,
                    engine=engine,
                    vocab=cur_vocab,
                    verbose=verbose,
                    oracle_workers=oracle_workers,
                    step_workers=step_workers,
                    forward_batch_size=forward_batch_size,
                    jt_exact_max=jt_exact_max,
                    jt_markov_order=jt_markov_order,
                    jt_large_set=jt_large_set,
                    strace_metrics_path=strace_metrics_path,
                    oracle_npz=cached,
                    require_cached_oracle=bool(oracle_index),
                    prefix_steps=prefix_ks or None,
                    metrics_only=metrics_only,
                    skip_completion=skip_completion,
                    skip_perplexity=skip_perplexity,
                )
                rows.extend(result.get("rows") or [result["row"]])
                perplexity = result.get("perplexity")
                if perplexity is not None:
                    if hasattr(perplexity, "sample_oracle"):
                        perplexity.sample_oracle = []
                    if hasattr(perplexity, "oracle_tables"):
                        perplexity.oracle_tables = {}
                result["metrics"] = None

                if resume and done_stems:
                    prior = read_metrics_csv(csv_path) if csv_path.is_file() else []
                    prior_kept = [
                        r
                        for r in prior
                        if Path(str(r.get("filename") or "")).stem in done_stems
                    ]
                    all_so_far = prior_kept + rows
                else:
                    all_so_far = rows
                write_metrics_csv(csv_path, all_so_far, append=False, pcfg=pcfg_sheet)
                logger.info(
                    "Flushed CSV after %s (%d run(s))",
                    name,
                    len(all_so_far),
                )
            except TraceProxyError as e:
                logger.error("Fail-hard on %s: %s", name, e)
                raise
            finally:
                if temp_dir is not None:
                    remove_temp_download(temp_dir)
    finally:
        if lazy is not None:
            lazy.close()
        elif model is not None:
            _release_model(model)

    if resume and csv_path.is_file() and done_stems:
        prior = read_metrics_csv(csv_path)
        prior_kept = [
            r for r in prior if Path(str(r.get("filename") or "")).stem in done_stems
        ]
        all_rows = prior_kept + rows
        write_metrics_csv(csv_path, all_rows, append=False, pcfg=pcfg_sheet)
    else:
        all_rows = rows
        write_metrics_csv(csv_path, all_rows, append=False, pcfg=pcfg_sheet)

    return {
        "rows": all_rows,
        "csv_path": str(csv_path.resolve()),
    }


# Alias used by CLI / docs.
run_hub_compare = run_compare
