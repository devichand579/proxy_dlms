"""
Discover NPZs on a dataset repo (no download)
One file at a time: temp download -> stream samples -> metrics/plots -> delete temp
Batch start: purge stale HF cache for the repo
Optional CSV resume
"""
from __future__ import annotations

import csv
import fnmatch
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from schedulerlens.metrics import (
    build_metrics_row,
    compute_task_metrics,
    correlations_from_trends,
    parse_run_stem,
    read_metrics_csv,
    write_metrics_csv,
)
from schedulerlens.oracle import compute_task_perplexity, save_oracle_artifacts
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

DEFAULT_METRICS_CSV = Path(__file__).resolve().parent.parent / "outputs/schedulerlens/schedulerlens_metrics.csv"

# Re-export Hub primitives so callers import from one place.
__all__ = [
    "DEFAULT_METRICS_CSV",
    "analyze_hub_file",
    "analyze_local_file",
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
    """List Hub NPZ paths matching ``pattern`` (filename glob)."""
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


def filter_by_meta(
    filenames: Sequence[str],
    *,
    model: Optional[str] = None,
    scheduler: Optional[str] = None,
    steps: Optional[int] = None,
    pcfg_only: bool = False,
) -> List[str]:
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
    token: bool | str | None = True,
) -> dict[str, Any]:
    """Filename inventory for SchedulerLens NPZs (no downloads)."""
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
    compute_surfaces: bool = True,
    oracle_workers: int = 1,
    step_workers: Optional[int] = None,
    forward_batch_size: int = 8,
    jt_exact_max: Optional[int] = None,
    jt_markov_order: Optional[int] = None,
    jt_large_set: Optional[int] = None,
    skip_jt: bool = False,
) -> Dict[str, Any]:
    """Analyze one local NPZ (samples streamed via ``iter_npz_traces``)."""
    meta = parse_run_stem(trace_path)
    do_replay = should_replay(
        meta_is_pcfg=meta.is_pcfg, force_pcfg=force_pcfg, replay_oracle=replay_oracle
    )

    metrics = compute_task_metrics(
        trace_path,
        max_samples=max_samples,
        bucket_size=bucket_size,
        verbose=verbose,
    )

    ppl = compute_task_perplexity(
        trace_path,
        engine=engine,
        vocab=vocab,
        model=model if do_replay else None,
        device=device,
        replay_oracle=do_replay,
        max_samples=max_samples,
        verbose=verbose,
        oracle_workers=oracle_workers,
        step_workers=step_workers,
        forward_batch_size=forward_batch_size,
        jt_exact_max=jt_exact_max,
        jt_markov_order=jt_markov_order,
        jt_large_set=jt_large_set,
        skip_jt=skip_jt,
    )

    correlations = None
    fidelity = None
    if do_replay and ppl.mt_trend and ppl.jt_trend:
        m_hat = metrics.proxies["M_hat_t"].trend
        j_hat = metrics.proxies["J_hat_t"].trend
        n = min(len(ppl.mt_trend), len(m_hat), len(ppl.jt_trend), len(j_hat))
        correlations = correlations_from_trends(
            mt_trend=ppl.mt_trend[:n],
            m_hat_trend=m_hat[:n],
            jt_trend=ppl.jt_trend[:n],
            j_hat_trend=j_hat[:n],
            sample_oracle_nll=ppl.sample_oracle_nll,
            sample_model_nll=ppl.sample_model_nll,
        )
        fidelity = {
            "mt": np.asarray(ppl.mt_trend[:n], dtype=np.float64),
            "m_hat": np.asarray(m_hat[:n], dtype=np.float64),
            "jt": np.asarray(ppl.jt_trend[:n], dtype=np.float64),
            "j_hat": np.asarray(j_hat[:n], dtype=np.float64),
        }

    row = build_metrics_row(
        trace_path,
        metrics=metrics,
        perplexity=ppl,
        correlations=correlations,
        meta=meta,
        repo=repo,
    )

    surfaces = None
    if compute_surfaces:
        from schedulerlens.plots import compute_trace_surfaces

        surfaces = compute_trace_surfaces(trace_path, max_samples=max_samples)

    return {
        "row": row,
        "metrics": metrics,
        "perplexity": ppl,
        "correlations": correlations,
        "surfaces": surfaces,
        "fidelity": fidelity,
        "meta": meta,
    }


def analyze_local_file(
    trace_path: Union[str, Path],
    **kwargs: Any,
) -> Dict[str, Any]:
    return analyze_one_npz(Path(trace_path), **kwargs)


def analyze_hub_file(
    filename: str,
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    keep_download: bool = False,
    dest: Optional[Union[str, Path]] = None,
    **analyze_kwargs: Any,
) -> Dict[str, Any]:
    """Download one Hub NPZ → analyze (streaming samples) → delete temp (default)."""
    full_repo = resolve_repo(repo_id, namespace)
    temp_dir = None
    try:
        if dest is not None:
            trace_path = download_file(
                filename,
                dest=Path(dest),
                repo_id=repo_id,
                namespace=namespace,
                revision=revision,
            )
        else:
            trace_path, temp_dir = download_file_to_temp(
                filename,
                repo_id=repo_id,
                namespace=namespace,
                revision=revision,
            )
        result = analyze_one_npz(
            Path(trace_path),
            repo=full_repo,
            **analyze_kwargs,
        )
        result["hub_filename"] = filename
        result["local_path"] = str(Path(trace_path).resolve())
        return result
    finally:
        if temp_dir is not None and not keep_download:
            remove_temp_download(temp_dir)


def resolve_pcfg_model_dir(
    *,
    model_path: Optional[Union[str, Path]] = None,
    hf_model: Optional[str] = None,
    token: bool | str | None = True,
) -> Optional[Path]:
    """Resolve a local checkpoint dir or download an HF model id via snapshot_download."""
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
    """Drop model weights and free CUDA cache if possible."""
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
    """Resolve HF model repo for a trace.

    Priority: explicit ``hf_model`` override → ``{namespace}/{meta.model}``
    where ``meta.model`` is the short name from the stem (e.g. ``llada-100m-pcfg``).
    Namespace defaults to ``--namespace`` / ``TRACE_HF_NAMESPACE`` / whoami.
    """
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
    """Build PCFG engine; optionally load one fixed checkpoint (non-lazy path)."""
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
    """Load one HF/local PCFG checkpoint at a time; unload on model switch."""

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
        """Return (model, vocab, device) for this trace's model; may unload previous."""
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


def _load_done_stems(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        return set()
    done: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("filename") or row.get("name") or "").strip()
            if name:
                done.add(Path(name).stem)
    return done


def _plot_artifacts_dir(output_dir: Path) -> Path:
    path = Path(output_dir) / "plot_artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_stem_plot_artifacts(
    art_dir: Path,
    stem: str,
    *,
    fidelity: Optional[Dict[str, np.ndarray]] = None,
    surfaces: Any = None,
    perplexity: Any = None,
) -> None:
    """Persist plot/oracle inputs for one run; avoids holding them all in RAM.

    ``oracle.npz`` keeps expensive replay detail (per-sample / step / token).
    Proxy caps are not stored (cheap to recompute from traces).
    """
    stem_dir = art_dir / stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    if perplexity is not None:
        tables = getattr(perplexity, "oracle_tables", None) or {}
        has_tables = bool(tables.get("step_sample_idx"))
        has_objs = any(
            s is not None for s in (getattr(perplexity, "sample_oracle", None) or [])
        )
        if has_tables or has_objs:
            save_oracle_artifacts(stem_dir / "oracle.npz", perplexity, stem=stem)
    if fidelity:
        np.savez_compressed(
            stem_dir / "fidelity.npz",
            **{k: np.asarray(v) for k, v in fidelity.items()},
        )
    if surfaces is not None and getattr(surfaces, "n_samples", 0) > 0:
        mt = getattr(surfaces, "mt_at_commit", None)
        payload = {
            "confidence": np.asarray(surfaces.confidence),
            "entropy": np.asarray(surfaces.entropy),
            "n_samples": np.asarray(int(surfaces.n_samples)),
            "n_steps": np.asarray(int(surfaces.n_steps)),
            "gen_length": np.asarray(int(surfaces.gen_length)),
        }
        if mt is not None and np.asarray(mt).size:
            payload["mt_at_commit"] = np.asarray(mt)
        np.savez_compressed(stem_dir / "surfaces.npz", **payload)


def _load_fidelity_points(art_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    if not art_dir.is_dir():
        return out
    for stem_dir in sorted(art_dir.iterdir()):
        path = stem_dir / "fidelity.npz"
        if not stem_dir.is_dir() or not path.is_file():
            continue
        with np.load(path) as z:
            out[stem_dir.name] = {k: np.asarray(z[k]) for k in z.files}
    return out


def _load_surfaces_by_stem(art_dir: Path) -> Dict[str, Any]:
    from schedulerlens.plots import TraceSurfaces

    out: Dict[str, Any] = {}
    if not art_dir.is_dir():
        return out
    for stem_dir in sorted(art_dir.iterdir()):
        path = stem_dir / "surfaces.npz"
        if not stem_dir.is_dir() or not path.is_file():
            continue
        with np.load(path) as z:
            mt = (
                np.asarray(z["mt_at_commit"])
                if "mt_at_commit" in z.files
                else np.array([])
            )
            out[stem_dir.name] = TraceSurfaces(
                confidence=np.asarray(z["confidence"]),
                entropy=np.asarray(z["entropy"]),
                mt_at_commit=mt,
                n_samples=int(z["n_samples"]),
                n_steps=int(z["n_steps"]),
                gen_length=int(z["gen_length"]),
            )
    return out


def _write_compare_plots(
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    art_dir: Path,
    strace_metrics_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Rebuild compare plots from CSV rows + on-disk step-level artifacts."""
    if not rows:
        return {}
    try:
        from schedulerlens.plots import save_compare_plots
    except ImportError as e:
        logger.warning("Skipping compare plots (missing dependency): %s", e)
        return {}
    fidelity = _load_fidelity_points(art_dir)
    surfaces = _load_surfaces_by_stem(art_dir)
    return save_compare_plots(
        rows,
        output_dir,
        surfaces_by_stem=surfaces or None,
        fidelity_points=fidelity or None,
        strace_metrics_path=strace_metrics_path,
    )


def _write_run_plots(
    output_dir: Path,
    stem: str,
    *,
    metrics: Any,
    fidelity: Optional[Dict[str, np.ndarray]] = None,
    surfaces: Any = None,
    row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Write per-NPZ PNGs (trends + single-run compare panel) under run_plots/<stem>/."""
    run_dir = Path(output_dir) / "run_plots" / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    try:
        from schedulerlens.plots import save_compare_plots, save_trend_plots
    except ImportError as e:
        logger.warning("Skipping run plots for %s (missing dependency): %s", stem, e)
        return paths

    if metrics is not None and getattr(metrics, "n_steps", 0) > 0:
        m_plot, j_plot, unmasked_plot = save_trend_plots(
            metrics, run_dir, prefix=stem
        )
        paths["M_t"] = m_plot
        paths["J_t"] = j_plot
        paths["unmasked"] = unmasked_plot

    if row is not None:
        surf_map = None
        if surfaces is not None and getattr(surfaces, "n_samples", 0) > 0:
            surf_map = {stem: surfaces}
        fid_map = {stem: fidelity} if fidelity else None
        for key, path in save_compare_plots(
            [row],
            run_dir,
            surfaces_by_stem=surf_map,
            fidelity_points=fid_map,
        ).items():
            paths[f"compare_{key}"] = path
    return paths


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
    force_pcfg: bool = False,
    replay_oracle: bool = False,
    model_path: Optional[Union[str, Path]] = None,
    hf_model: Optional[str] = None,
    strace_metrics_path: Optional[Union[str, Path]] = None,
    max_samples: Optional[int] = None,
    limit: Optional[int] = None,
    resume: bool = False,
    purge_cache: bool = True,
    verbose: bool = True,
    oracle_workers: int = 1,
    step_workers: Optional[int] = None,
    forward_batch_size: int = 8,
    jt_exact_max: Optional[int] = None,
    jt_markov_order: Optional[int] = None,
    jt_large_set: Optional[int] = None,
    skip_jt: bool = False,
) -> Dict[str, Any]:
    """Discover NPZs (Hub or local), analyze each, write CSV + comparison plots.

    Hub path: purge cache → per-file temp download → stream analyze → delete temp.
    After each NPZ: flush CSV, persist step-level plot artifacts to disk, and
    regenerate compare plots (so a crash keeps progress and RAM stays flat).

    With ``--replay-oracle`` and no fixed ``--hf-model``/``--model-path``, each
    PCFG stem's ``meta.model`` (e.g. ``llada-100m-pcfg``) is lazy-loaded from
    ``{namespace}/{meta.model}``; the previous weights are unloaded on switch.
    """
    from schedulerlens.oracle import _ensure_pcfg_path

    _ensure_pcfg_path()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "schedulerlens_metrics.csv"
    art_dir = _plot_artifacts_dir(output_dir)

    fixed_override = model_path is not None or hf_model is not None
    engine, vocab, model, device = _load_oracle_stack(
        replay_oracle=replay_oracle,
        model_path=model_path if fixed_override else None,
        hf_model=hf_model if fixed_override else None,
    )
    if replay_oracle and engine is None:
        try:
            from pcfg_data import build_stack

            _g, _cnf, engine, vocab = build_stack()
        except Exception as e:
            logger.warning("PCFG stack unavailable (%s)", e)

    lazy: Optional[_LazyOracleModels] = None
    if replay_oracle and not fixed_override:
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
    elif fixed_override:
        # Keep a holder so we still unload at the end.
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
    plot_paths: Dict[str, Path] = {}
    repo_label = ""

    done_stems: set[str] = set()
    if resume and csv_path.is_file():
        done_stems = _load_done_stems(csv_path)
        if done_stems:
            logger.info("Resume: skipping %d stems already in %s", len(done_stems), csv_path)

    if local_dir is not None:
        local_dir = Path(local_dir)
        paths = discover_local_npz(local_dir, pattern)
        names = filter_by_meta(
            [p.name for p in paths],
            model=model_filter,
            scheduler=scheduler_filter,
            steps=steps_filter,
            pcfg_only=pcfg_only,
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
        )
        work_items = [(name, None, name) for name in hub_files]

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
        write_metrics_csv(csv_path, [])
        return {"rows": [], "csv_path": str(csv_path), "plot_paths": {}}

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
                cur_model, cur_vocab, cur_device = model, vocab, device
                if lazy is not None and replay_oracle:
                    cur_model, cur_vocab, cur_device = lazy.for_meta(meta)

                logger.info(
                    "[%d/%d] Analyzing %s (model=%s)",
                    idx + 1,
                    len(work_items),
                    name,
                    meta.model,
                )
                result = analyze_one_npz(
                    Path(trace_path),
                    repo=repo_label,
                    max_samples=max_samples,
                    replay_oracle=replay_oracle,
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
                    skip_jt=skip_jt,
                )
                rows.append(result["row"])
                stem = result["meta"].stem
                fidelity = result.get("fidelity")
                surfaces = result.get("surfaces")
                metrics = result.get("metrics")
                perplexity = result.get("perplexity")

                # Per-NPZ PNGs first (trends + single-run panel), then persist arrays.
                run_plot_paths = _write_run_plots(
                    output_dir,
                    stem,
                    metrics=metrics,
                    fidelity=fidelity,
                    surfaces=surfaces,
                    row=result["row"],
                )
                _save_stem_plot_artifacts(
                    art_dir,
                    stem,
                    fidelity=fidelity,
                    surfaces=surfaces,
                    perplexity=perplexity,
                )
                # Drop bulky payloads; batch compare reloads from plot_artifacts/.
                result["surfaces"] = None
                result["fidelity"] = None
                if perplexity is not None:
                    if hasattr(perplexity, "sample_oracle"):
                        perplexity.sample_oracle = []
                    if hasattr(perplexity, "oracle_tables"):
                        perplexity.oracle_tables = {}
                result["metrics"] = None

                # Flush CSV + batch compare plots after each file.
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
                write_metrics_csv(csv_path, all_so_far, append=False)
                with (output_dir / "compare_summary.json").open(
                    "w", encoding="utf-8"
                ) as f:
                    json.dump(
                        {
                            "n_runs": len(all_so_far),
                            "repo": repo_label,
                            "rows": all_so_far,
                        },
                        f,
                        indent=2,
                    )
                plot_paths = _write_compare_plots(
                    output_dir,
                    all_so_far,
                    art_dir=art_dir,
                    strace_metrics_path=strace_metrics_path,
                )
                logger.info(
                    "Flushed CSV + plots after %s (%d run(s); run_plots=%d files)",
                    name,
                    len(all_so_far),
                    len(run_plot_paths),
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
        write_metrics_csv(csv_path, all_rows, append=False)
    else:
        all_rows = rows
        write_metrics_csv(csv_path, all_rows, append=False)

    with (output_dir / "compare_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"n_runs": len(all_rows), "repo": repo_label, "rows": all_rows}, f, indent=2)

    # Final pass (covers resume-only / empty-work cases; normally matches last flush).
    if all_rows:
        plot_paths = _write_compare_plots(
            output_dir,
            all_rows,
            art_dir=art_dir,
            strace_metrics_path=strace_metrics_path,
        )

    return {
        "rows": all_rows,
        "csv_path": str(csv_path.resolve()),
        "plot_paths": {k: str(v.resolve()) for k, v in plot_paths.items()},
    }


# Alias used by CLI / docs.
run_hub_compare = run_compare
