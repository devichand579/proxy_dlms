"""Fixed-history predictor–controller audit over stored NPZ traces."""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch

from schedulerlens.causal.metrics import StepDelta, step_deltas
from schedulerlens.causal.scores import bundle_for_select, bundle_from_logits
from schedulerlens.causal.select import SUPPORTED_PHI, select_gen
from schedulerlens.causal.state import reconstruct_state
from schedulerlens.metrics import parse_run_stem
from traces.trace_npz import iter_npz_traces

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "filename",
    "trace_model",
    "scheduler",
    "task",
    "steps",
    "predictor",
    "controller",
    "n_samples",
    "n_steps_scored",
    "delta_pred_m",
    "delta_ctrl_m",
    "delta_ctrl_j",
    "delta_own_j",
    "jaccard",
]

_MEAN_KEYS = (
    "delta_pred_m",
    "delta_ctrl_m",
    "delta_ctrl_j",
    "delta_own_j",
    "jaccard",
)


def _split_even(items: list, n: int) -> list[list]:
    n = max(1, min(n, len(items)))
    base, extra = divmod(len(items), n)
    chunks = []
    start = 0
    for i in range(n):
        take = base + (1 if i < extra else 0)
        if take:
            chunks.append(items[start : start + take])
            start += take
    return chunks


def _forward_batch(model, states: list[np.ndarray]) -> torch.Tensor:
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    stacked = torch.tensor(np.stack(states, axis=0), dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(stacked).logits
    return logits.detach().to("cpu")


def _parallel_logits(groups: list[tuple[Any, list[np.ndarray]]]) -> list[torch.Tensor]:
    """Run each ``(model, states)`` forward. Distinct devices run together."""
    if len(groups) == 1:
        return [_forward_batch(groups[0][0], groups[0][1])]
    devices = {str(next(model.parameters()).device) for model, _states in groups}
    if len(devices) == 1:
        return [_forward_batch(model, states) for model, states in groups]
    slots: list[Optional[torch.Tensor]] = [None] * len(groups)
    errors: list[BaseException] = []

    def run(index: int, model, states: list[np.ndarray]) -> None:
        try:
            slots[index] = _forward_batch(model, states)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(i, model, states))
        for i, (model, states) in enumerate(groups)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return [slot for slot in slots if slot is not None]


def _role_logits(models: Sequence[Any], states: list[np.ndarray]) -> torch.Tensor:
    chunks = _split_even(states, len(models))
    parts = _parallel_logits(list(zip(models, chunks)))
    return torch.cat(parts, dim=0)


def _chunk_deltas(
    chunk: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    phi: str,
    predictors: Sequence[Any],
    controllers: Sequence[Any],
    steps_budget: int,
    block_length: Optional[int],
    top_k: int,
    gamma: float,
    prompt_len: int,
    eos_id: int,
    eot_id: Optional[int],
    controller_drop_eos: bool,
) -> list[StepDelta]:
    states = [row[1] for row in chunk]
    same = controllers is predictors
    if same:
        logits_p = _role_logits(predictors, states)
        logits_c = logits_p
    else:
        pred_chunks = _split_even(states, len(predictors))
        ctrl_chunks = _split_even(states, len(controllers))
        groups = [(model, part) for model, part in zip(predictors, pred_chunks)]
        groups += [(model, part) for model, part in zip(controllers, ctrl_chunks)]
        parts = _parallel_logits(groups)
        logits_p = torch.cat(parts[: len(pred_chunks)], dim=0)
        logits_c = torch.cat(parts[len(pred_chunks) :], dim=0)
    bundles_p = bundle_from_logits(logits_p)
    bundles_c = bundles_p if same else bundle_from_logits(logits_c)
    select_c = bundles_c
    if controller_drop_eos:
        select_c = bundle_for_select(
            logits_c, drop_eos=True, eos_id=eos_id, eot_id=eot_id
        )
    out: list[StepDelta] = []
    for i, (step, _state, gen_mask) in enumerate(chunk):
        bundle_pred = bundles_p[i].gen(prompt_len)
        bundle_ctrl = bundles_c[i].gen(prompt_len)
        select_ctrl = select_c[i].gen(prompt_len)
        s0 = select_gen(
            phi, bundle_pred, gen_mask,
            step=step, steps=steps_budget, block_length=block_length,
            top_k=top_k, gamma=gamma,
        )
        s1 = select_gen(
            phi, select_ctrl, gen_mask,
            step=step, steps=steps_budget, block_length=block_length,
            top_k=top_k, gamma=gamma,
        )
        out.append(
            step_deltas(
                conf_pred=bundle_pred.conf,
                ent_pred=bundle_pred.entropy,
                conf_ctrl=bundle_ctrl.conf,
                ent_ctrl=bundle_ctrl.entropy,
                s0=s0,
                s1=s1,
            )
        )
    return out


def audit_trace(
    trace: dict[str, Any],
    *,
    phi: str,
    predictors: Sequence[Any],
    controllers: Sequence[Any],
    mask_id: int,
    steps_budget: int,
    block_length: Optional[int] = None,
    top_k: int = 2,
    gamma: float = 0.1,
    max_steps: Optional[int] = None,
    trace_name: str | None = None,
    eos_id: int = 126081,
    eot_id: Optional[int] = None,
    controller_drop_eos: bool = False,
    batch_size: int = 8,
) -> list[StepDelta]:
    """Score every recorded step of one sample. S0 uses the predictor, S1 the controller.

    ``batch_size`` is how many steps each replica runs. Replicas on different
    GPUs, and the two roles, run at the same time. Selection and the deltas
    stay one value per step.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    n_steps = len(trace.get("step_traces") or [])
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))
    pending: list[tuple[int, np.ndarray, np.ndarray]] = []
    prompt_len = None
    for t in range(n_steps):
        x, gen_mask = reconstruct_state(
            trace, t, mask_id=mask_id, trace_name=trace_name
        )
        if prompt_len is None:
            prompt_len = int(x.shape[0] - gen_mask.shape[0])
        if not gen_mask.any():
            continue
        pending.append((t, x, gen_mask))
    if prompt_len is None:
        return []
    out: list[StepDelta] = []
    wave = batch_size * max(len(predictors), len(controllers))
    for start in range(0, len(pending), wave):
        out.extend(
            _chunk_deltas(
                pending[start : start + wave],
                phi=phi,
                predictors=predictors,
                controllers=controllers,
                steps_budget=steps_budget,
                block_length=block_length,
                top_k=top_k,
                gamma=gamma,
                prompt_len=prompt_len,
                eos_id=eos_id,
                eot_id=eot_id,
                controller_drop_eos=controller_drop_eos,
            )
        )
    return out


def _finite_mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _sample_mean(deltas: list[StepDelta]) -> dict[str, float]:
    return {
        key: _finite_mean([float(getattr(d, key)) for d in deltas])
        for key in _MEAN_KEYS
    }


def audit_npz(
    path: Path,
    *,
    predictors: Sequence[Any],
    controllers: Sequence[Any],
    mask_id: int,
    predictor_name: str,
    controller_name: str,
    block_length: Optional[int] = None,
    top_k: int = 2,
    gamma: float = 0.1,
    max_samples: Optional[int] = None,
    max_steps: Optional[int] = None,
    allowed_models: Iterable[str] = ("llada",),
    eos_id: int = 126081,
    eot_id: Optional[int] = None,
    controller_drop_eos: bool = False,
    batch_size: int = 8,
) -> dict[str, Any]:
    meta = parse_run_stem(path)
    if meta.model not in set(allowed_models):
        raise ValueError(f"{path.name}: trace model {meta.model!r} is not a llada trace")
    if meta.scheduler not in SUPPORTED_PHI:
        raise ValueError(
            f"{path.name}: scheduler {meta.scheduler!r} is not in {SUPPORTED_PHI}"
        )
    steps_budget = int(meta.steps or 0) or 256
    sample_means: list[dict[str, float]] = []
    n_steps_scored = 0
    n_samples = 0
    for i, trace in enumerate(iter_npz_traces(path)):
        if max_samples is not None and i >= max_samples:
            break
        n_samples += 1
        deltas = audit_trace(
            trace,
            phi=meta.scheduler,
            predictors=predictors,
            controllers=controllers,
            mask_id=mask_id,
            steps_budget=steps_budget,
            block_length=block_length,
            top_k=top_k,
            gamma=gamma,
            max_steps=max_steps,
            trace_name=str(path),
            eos_id=eos_id,
            eot_id=eot_id,
            controller_drop_eos=controller_drop_eos,
            batch_size=batch_size,
        )
        if not deltas:
            continue
        n_steps_scored += len(deltas)
        sample_means.append(_sample_mean(deltas))
    row: dict[str, Any] = {
        "filename": meta.filename,
        "trace_model": meta.model,
        "scheduler": meta.scheduler,
        "task": meta.task,
        "steps": meta.steps if meta.steps is not None else "",
        "predictor": predictor_name,
        "controller": controller_name,
        "n_samples": n_samples,
        "n_steps_scored": n_steps_scored,
    }
    for key in _MEAN_KEYS:
        row[key] = _finite_mean([sample[key] for sample in sample_means])
    return row


def load_audit_csv(path: Path) -> list[dict[str, str]]:
    """Rows already written, or an empty list when the CSV is absent."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def iter_npz_paths(
    local_dir: Optional[Path],
    *,
    pattern: str = "*.npz",
) -> list[Path]:
    if local_dir is None:
        return []
    return sorted(Path(local_dir).glob(pattern))


def keep_stem(path: Path, *, steps: Optional[int], exclude_task: Optional[str]) -> bool:
    meta = parse_run_stem(path)
    if meta.model != "llada":
        return False
    if meta.scheduler not in SUPPORTED_PHI:
        return False
    if steps is not None and meta.steps != int(steps):
        return False
    if exclude_task and meta.task == exclude_task:
        return False
    return True


