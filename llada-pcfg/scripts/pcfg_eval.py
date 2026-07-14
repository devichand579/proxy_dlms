"""
Evaluate a trained PCFG LLaDA model on the test split with diffusion schedulers.

Uses lm-eval-style prefix/suffix decoding: a deterministic visible prefix per sample
(seed + sample index) and a fixed ``gen_length`` generation window.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from utils import ensure_import_paths, eval_outputs_dir, resolve_repo_path

ensure_import_paths()


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return bool(val)


def _tracing_enabled_from_argv_env() -> bool:
    raw = os.environ.get("TRACING_ENABLED")
    if raw is not None and str(raw).strip():
        return _to_bool(raw)
    argv = sys.argv[1:]
    if "--no-tracing-enabled" in argv:
        return False
    for i, flag in enumerate(argv):
        if flag in ("--tracing-enabled", "--tracing_enabled") and i + 1 < len(argv):
            return _to_bool(argv[i + 1])
    return True


# Schedulers bind @trace_scheduler at import time.
if _tracing_enabled_from_argv_env():
    os.environ["TRACING_ENABLED"] = "true"
else:
    os.environ["TRACING_ENABLED"] = "false"

import torch
from hf_hub import apply_pcfg_hf_env, resolve_hf_token
from models.pcfg_models import load_pcfg_llada_checkpoint
from pcfg_data import ARCH_LEN, fit_corpus_for_arch, load_corpus_from_disk, load_corpus_from_hub
from pcfg_metrics import (
    PromptSuffixEvalSample,
    build_prompt_suffix_batch,
    build_prompt_suffix_eval_plan,
    suffix_exact_match,
    suffix_structural_accuracy,
)
from pcfg_utils.wandb_utils import (
    default_eval_run_name,
    init_wandb,
    resolve_eval_model_name,
)
from repo_env_defaults import apply_repo_env_defaults, is_offline_env
from schedulers.generate import generate
from traces.hf_trace_upload import maybe_upload_eval_trace_file, trace_hf_upload_enabled
from traces.trace_wrapper import _task_trace_list_ctx

SCHEDULER_MAP = {
    "pd": ("pure_diffusion", "confidence"),
    "bd": ("block_diffusion", "confidence"),
    "eb": ("eb_sampler", "confidence"),
    "fdm": ("fdm", None),
    "fhs": ("fhs", None),
    "top_k_c": ("top_k", "confidence"),
    "top_k_m": ("top_k", "top_margin"),
    "dus": ("dus", "confidence"),
}
ALL_SCHEDULERS = list(SCHEDULER_MAP.keys())

CHECKPOINT_JSONL = "checkpoint.jsonl"
WANDB_ID_FILE = ".wandb_id"


class StreamingTraceWriter:
    """Append JSONL traces; resume-safe when ``resumable=True``."""

    def __init__(self, filepath: Path, *, resumable: bool) -> None:
        self._path = filepath
        self._resumable = resumable
        self._count = 0
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if resumable and filepath.is_file():
            with filepath.open("r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self._count += 1
            self._f = filepath.open("a", encoding="utf-8")
            if self._count == 0:
                self._f.write('# task="pcfg_suffix"\n')
                self._f.flush()
        else:
            self._f = filepath.open("w", encoding="utf-8")

    def append(self, trace_dict: dict) -> None:
        self._f.write(json.dumps(trace_dict, ensure_ascii=False) + "\n")
        self._count += 1
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    @property
    def count(self) -> int:
        return self._count


def _checkpoint_path(work_dir: Path) -> Path:
    return work_dir / CHECKPOINT_JSONL


def _load_checkpoint_completed(path: Path) -> dict[int, dict]:
    completed: dict[int, dict] = {}
    if not path.is_file():
        return completed
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                sample_idx = rec.get("sample_idx")
                if sample_idx is not None:
                    completed[int(sample_idx)] = rec
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: failed to load checkpoint {path}: {exc}", flush=True)
    return completed


def _read_wandb_id(work_dir: Path) -> str | None:
    path = work_dir / WANDB_ID_FILE
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _maybe_persist_wandb_id(work_dir: Path, wb, *, resumable: bool) -> None:
    if not resumable or wb is None:
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    path = work_dir / WANDB_ID_FILE
    if path.exists():
        return
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(str(wandb.run.id), encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not write {path}: {exc}", flush=True)


def _aggregate_metrics(completed: dict[int, dict]) -> tuple[float, float, int]:
    if not completed:
        return 0.0, 0.0, 0
    n = len(completed)
    em = sum(float(v.get("exact_match", 0.0)) for v in completed.values()) / n
    struct = sum(float(v.get("structural_accuracy", 0.0)) for v in completed.values()) / n
    return em, struct, n


def resolve_block_length(mode: str, gen_length: int, block_length: int | None) -> int | None:
    if block_length is not None:
        return int(block_length)
    if mode == "fhs":
        return None
    if mode == "block_diffusion":
        return min(32, gen_length)
    if mode == "fdm":
        return min(64, gen_length)
    if mode == "dus":
        return min(16, gen_length)
    return gen_length


def resolve_model_path(
    *,
    model_path: Path | None,
    hf_model: str | None,
    token: str | None,
) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"--model-path not found: {path}")
        return path
    if hf_model is None:
        raise ValueError("pass --model-path or --hf-model")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(hf_model, token=token))


def load_test_corpus(args) -> list[list[str]]:
    if args.corpus is not None:
        _train, test, _ = load_corpus_from_disk(args.corpus)
    elif args.hf_dataset is not None:
        _train, test, _ = load_corpus_from_hub(
            args.hf_dataset,
            namespace=args.hf_namespace,
            token=resolve_hf_token(None),
        )
    else:
        raise ValueError("pass --corpus or --hf-dataset for test split")
    if not test:
        raise ValueError("corpus has no test split; regenerate with pcfg_generate holdout")
    test, n_clipped = fit_corpus_for_arch(test, arch_len=ARCH_LEN)
    if n_clipped > 0:
        print(
            f"clipped {n_clipped} test strings to max content {ARCH_LEN - 1}",
            flush=True,
        )
    return test


def parse_schedulers(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return ALL_SCHEDULERS
    keys = [s.strip().lower() for s in raw.split(",") if s.strip()]
    bad = [k for k in keys if k not in SCHEDULER_MAP]
    if bad:
        raise ValueError(f"unknown schedulers {bad}; choose from {ALL_SCHEDULERS}")
    return keys


def _log_eval_wandb(wb, metrics: dict, *, run_name: str, trace_path: str | None) -> None:
    if wb is None:
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    try:
        wandb.log(
            {
                "eval/exact_match": metrics["exact_match"],
                "eval/structural_accuracy": metrics["structural_accuracy"],
                "eval/n_samples": metrics["n_samples"],
                "eval/n_computed": metrics.get("n_computed", metrics["n_samples"]),
                "eval/n_from_checkpoint": metrics.get("n_from_checkpoint", 0),
                "eval/gen_length": metrics["gen_length"],
                "eval/steps": metrics.get("steps"),
                "eval/scheduler": metrics["scheduler"],
                "eval/complete": float(metrics.get("complete", False)),
            }
        )
        if trace_path and Path(trace_path).is_file():
            trace_name_safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", run_name)
            artifact = wandb.Artifact(
                name=f"trace_{trace_name_safe}",
                type="trace",
                description=f"PCFG suffix eval trace ({run_name})",
            )
            artifact.add_file(trace_path, name=Path(trace_path).name)
            wandb.log_artifact(artifact)
    except Exception as exc:
        print(f"wandb log failed: {exc}", flush=True)


def _finish_wandb(wb) -> None:
    if wb is None:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def eval_scheduler(
    scheduler_key: str,
    *,
    model,
    vocab,
    eval_plan: list[PromptSuffixEvalSample],
    device: torch.device,
    steps: int,
    gen_length: int,
    batch_size: int,
    work_dir: Path,
    run_name: str,
    resumable: bool,
    tracing_enabled: bool,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_config: dict,
    wandb_resume_id: str | None,
) -> dict:
    mode, strategy = SCHEDULER_MAP[scheduler_key]
    strategy_kw = {} if strategy is None else {"strategy": strategy}
    bl = resolve_block_length(mode, gen_length, None)

    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(work_dir)
    completed = _load_checkpoint_completed(checkpoint_path) if resumable else {}
    to_compute = [s for s in eval_plan if s.sample_idx not in completed]
    n_from_checkpoint = len(completed)
    n_total = len(eval_plan)

    if resumable and n_from_checkpoint:
        print(
            f"  resume {run_name}: {n_from_checkpoint}/{n_total} from checkpoint, "
            f"{len(to_compute)} remaining",
            flush=True,
        )

    metrics: dict = {
        "scheduler": scheduler_key,
        "mode": mode,
        "strategy": strategy,
        "gen_length": gen_length,
        "steps": steps,
        "run_name": run_name,
        "resumable": resumable,
        "work_dir": str(work_dir),
        "n_samples": 0,
        "n_computed": 0,
        "n_from_checkpoint": 0,
        "exact_match": 0.0,
        "structural_accuracy": 0.0,
        "complete": False,
    }

    trace_path: Path | None = None
    trace_writer: StreamingTraceWriter | None = None
    if tracing_enabled:
        trace_path = work_dir / "trace_outputs" / f"{run_name}.jsonl"
        trace_writer = StreamingTraceWriter(trace_path, resumable=resumable)

    checkpoint_writer = None
    if resumable:
        checkpoint_writer = checkpoint_path.open("a", encoding="utf-8")

    wb = init_wandb(
        enabled=wandb_enabled,
        project=wandb_project,
        run_name=run_name,
        config={
            **wandb_config,
            "scheduler": scheduler_key,
            "run_name": run_name,
            "resumable": resumable,
            "work_dir": str(work_dir),
        },
        resume_id=wandb_resume_id,
    )
    _maybe_persist_wandb_id(work_dir, wb, resumable=resumable)

    n_computed = 0
    prompt_tensor_width = 0

    for start in range(0, len(to_compute), batch_size):
        batch_plan = to_compute[start : start + batch_size]
        input_ids, attention_mask, _prompt_lens = build_prompt_suffix_batch(batch_plan, vocab)
        prompt_tensor_width = int(input_ids.shape[1])
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        traces: list[dict] = []
        token = _task_trace_list_ctx.set(traces)
        try:
            out, _steps_taken, _blocks = generate(
                model,
                input_ids,
                attention_mask,
                steps=steps,
                gen_length=gen_length,
                block_length=bl if bl is not None else gen_length,
                mode=mode,
                top_k=1,
                trace_name=run_name if tracing_enabled else None,
                upload_to_wandb=False,
                mask_id=vocab.mask_id,
                eos_id=vocab.eos_id,
                use_eos_harness=False,
                **strategy_kw,
            )
        finally:
            _task_trace_list_ctx.reset(token)

        if tracing_enabled and not traces:
            traces = [
                {
                    "final_sequence": out[b : b + 1].detach().cpu().tolist(),
                    "prompt_len": prompt_tensor_width,
                    "gen_length": gen_length,
                    "step_traces": [],
                }
                for b in range(out.shape[0])
            ]

        for b, sample in enumerate(batch_plan):
            gen_start = prompt_tensor_width
            gen_end = gen_start + len(sample.ref_suffix_ids)
            gen_suffix = out[b, gen_start:gen_end].detach().cpu().tolist()
            suffix_tokens = sample.tokens[sample.n_visible :]

            em = suffix_exact_match(gen_suffix, sample.ref_suffix_ids)
            struct_acc = suffix_structural_accuracy(
                gen_suffix,
                sample.ref_suffix_ids,
                vocab,
                suffix_tokens=suffix_tokens,
            )

            rec = {
                "sample_idx": sample.sample_idx,
                "scheduler": scheduler_key,
                "run_name": run_name,
                "exact_match": bool(em),
                "structural_accuracy": float(struct_acc),
                "n_visible": sample.n_visible,
                "n_content": sample.n_content,
            }

            if tracing_enabled:
                tr = traces[b] if b < len(traces) else traces[-1]
                tr = dict(tr)
                tr["prompt_len"] = prompt_tensor_width
                tr["gen_length"] = gen_length
                tr["metadata"] = {
                    "task": "pcfg_suffix",
                    "run_name": run_name,
                    "scheduler": scheduler_key,
                    "sample_idx": sample.sample_idx,
                    "n_visible": sample.n_visible,
                    "n_content": sample.n_content,
                    "gen_length": gen_length,
                    "reference_tokens": sample.tokens,
                    "reference_suffix_tokens": suffix_tokens,
                    "reference_suffix_ids": sample.ref_suffix_ids,
                }
                if trace_writer is not None:
                    trace_writer.append(tr)

            if checkpoint_writer is not None:
                checkpoint_writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                checkpoint_writer.flush()
            elif resumable:
                completed[sample.sample_idx] = rec
            else:
                completed[sample.sample_idx] = rec

            n_computed += 1

    if checkpoint_writer is not None:
        checkpoint_writer.close()
    if trace_writer is not None:
        trace_writer.close()

    if resumable:
        completed = _load_checkpoint_completed(checkpoint_path)

    em, struct, n_done = _aggregate_metrics(completed)
    metrics["exact_match"] = em
    metrics["structural_accuracy"] = struct
    metrics["n_samples"] = n_done
    metrics["n_computed"] = n_computed
    metrics["n_from_checkpoint"] = n_from_checkpoint
    metrics["prompt_tensor_width"] = prompt_tensor_width
    metrics["complete"] = n_done == n_total and n_total > 0

    trace_path_str: str | None = None
    if tracing_enabled and trace_path is not None:
        trace_path_str = str(trace_path)
        metrics["trace_path"] = trace_path_str

    if metrics["complete"]:
        if tracing_enabled and trace_hf_upload_enabled() and trace_path_str:
            uploaded = maybe_upload_eval_trace_file(
                trace_path_str,
                trace_name=run_name,
                task_name="pcfg",
                num_samples=n_done,
            )
            metrics["trace_hf_upload"] = uploaded
        _log_eval_wandb(wb, metrics, run_name=run_name, trace_path=trace_path_str)
        (work_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    elif n_done > 0:
        print(
            f"  incomplete {run_name}: {n_done}/{n_total} — re-run with --resumable to continue",
            flush=True,
        )

    _finish_wandb(wb)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--hf-model", type=str, default=None, help="HF model repo id")
    p.add_argument("--variant", type=str, default="100m", help="Fallback label when model name cannot be inferred")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--hf-dataset", type=str, default=None)
    p.add_argument("--hf-namespace", type=str, default=None)
    p.add_argument("--schedulers", type=str, default="all", help="Comma list or 'all'")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gen-length", type=int, default=128,
                   help="Fixed generation window (lm-eval style; suffix unmasked in this region).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for per-sample prompt lengths (seed+sample_idx; shared across schedulers).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--resumable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fixed per-scheduler work dir with checkpoint.jsonl; re-run to resume.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run/trace/W&B stem (default: {model}-pcfg_{scheduler}_s{steps}).",
    )
    p.add_argument(
        "--tracing-enabled",
        dest="tracing_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write traces; HF upload via TRACE_HF_UPLOAD env / eval.trace_hf_upload (default: true).",
    )
    p.add_argument("--wandb", dest="wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use cached HF datasets/models and local W&B (no Hub/network).",
    )
    args = p.parse_args()

    offline = args.offline or is_offline_env()
    apply_repo_env_defaults()
    apply_pcfg_hf_env(offline=offline)

    tracing_enabled = (
        _tracing_enabled_from_argv_env()
        if args.tracing_enabled is None
        else bool(args.tracing_enabled)
    )

    base_out_dir = resolve_repo_path(args.out_dir) or eval_outputs_dir()
    if args.model_path is not None:
        args.model_path = resolve_repo_path(args.model_path)
    if args.corpus is not None:
        args.corpus = resolve_repo_path(args.corpus)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    schedulers = parse_schedulers(args.schedulers)
    model_dir = resolve_model_path(
        model_path=args.model_path,
        hf_model=args.hf_model,
        token=resolve_hf_token(None),
    )
    model_name = resolve_eval_model_name(
        model_dir=model_dir,
        hf_model=args.hf_model,
        variant=args.variant,
    )
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT", "strace")
    wandb_enabled = bool(args.wandb)
    run_id_override = args.run_id

    model, _config, vocab = load_pcfg_llada_checkpoint(model_dir, device=str(device))
    model.eval()

    test = load_test_corpus(args)
    if args.max_samples is not None:
        test = test[: args.max_samples]

    eval_plan = build_prompt_suffix_eval_plan(test, vocab, seed=args.seed)
    n_vis = [s.n_visible for s in eval_plan]
    suffix_lens = [len(s.ref_suffix_ids) for s in eval_plan]

    base_out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_path": str(model_dir),
        "model_name": model_name,
        "variant": args.variant,
        "n_test": len(test),
        "steps": args.steps,
        "gen_length": args.gen_length,
        "seed": args.seed,
        "resumable": bool(args.resumable),
        "tracing_enabled": tracing_enabled,
        "wandb": wandb_enabled,
        "prompt_len_range": [min(n_vis), max(n_vis)] if n_vis else [0, 0],
        "suffix_len_range": [min(suffix_lens), max(suffix_lens)] if suffix_lens else [0, 0],
        "schedulers": {},
    }
    wandb_config = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
        if k not in ("run_id",)
    }
    wandb_config.update(
        {
            "model_path": str(model_dir),
            "model_name": model_name,
            "n_test": len(test),
            "tracing_enabled": tracing_enabled,
            "resumable": bool(args.resumable),
        }
    )

    print(
        f"model={model_dir} name={model_name} test={len(test)} schedulers={schedulers} "
        f"gen_length={args.gen_length} seed={args.seed} resumable={args.resumable} "
        f"tracing={tracing_enabled} wandb={wandb_enabled} device={device}",
        flush=True,
    )
    if eval_plan:
        print(
            f"prompt_len in [1, n//2]: min={min(n_vis)} max={max(n_vis)} "
            f"suffix_len: min={min(suffix_lens)} max={max(suffix_lens)}",
            flush=True,
        )

    for key in schedulers:
        run_name = run_id_override or default_eval_run_name(model_name, key, args.steps)
        if args.resumable:
            work_dir = base_out_dir / run_name
            wandb_resume_id = _read_wandb_id(work_dir)
        else:
            work_dir = base_out_dir
            wandb_resume_id = None

        print(f"eval scheduler={key} run={run_name} work_dir={work_dir} ...", flush=True)
        m = eval_scheduler(
            key,
            model=model,
            vocab=vocab,
            eval_plan=eval_plan,
            device=device,
            steps=args.steps,
            gen_length=args.gen_length,
            batch_size=args.batch_size,
            work_dir=work_dir,
            run_name=run_name,
            resumable=bool(args.resumable),
            tracing_enabled=tracing_enabled,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_config=wandb_config,
            wandb_resume_id=wandb_resume_id,
        )
        summary["schedulers"][key] = m
        print(
            f"  {key}: em={m['exact_match']:.3f} struct={m['structural_accuracy']:.3f} "
            f"done={m['n_samples']}/{len(eval_plan)} complete={m['complete']} "
            f"trace={m.get('trace_path')}",
            flush=True,
        )

    summary_path = base_out_dir / f"pcfg_eval_{model_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
