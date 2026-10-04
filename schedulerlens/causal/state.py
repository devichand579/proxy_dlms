"""Rebuild a frozen diffusion state from a stored trace."""

from __future__ import annotations

from typing import Any

import numpy as np

from schedulerlens.oracle import extract_unmask_events


def prompt_ids(trace: dict[str, Any]) -> np.ndarray:
    prompt = trace.get("prompt")
    if prompt is None:
        prompt = trace.get("prompt_tokens")
    if prompt is None:
        raise ValueError("trace is missing prompt token ids")
    ids = np.asarray(prompt).reshape(-1)
    return ids.astype(np.int64, copy=False)


def reconstruct_state(
    trace: dict[str, Any],
    step: int,
    *,
    mask_id: int,
    pad_id: int,
    trace_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, gen_mask, attention_mask)`` just before ``step`` is applied.

    ``x`` is prompt ‖ generation. ``gen_mask`` is True where the gen region is still masked.
    Stored prompts include left padding, so replay must restore its attention mask.
    """
    prompt = prompt_ids(trace)
    gen_length = int(trace.get("gen_length") or 0)
    if gen_length <= 0:
        steps = trace.get("step_traces") or []
        if not steps or steps[0].get("mask_index") is None:
            raise ValueError("cannot infer gen_length")
        gen_length = int(np.asarray(steps[0]["mask_index"]).reshape(-1).shape[0])
    x = np.full(prompt.shape[0] + gen_length, int(mask_id), dtype=np.int64)
    x[: prompt.shape[0]] = prompt
    events = extract_unmask_events(trace, trace_name=trace_name)
    prompt_len = int(prompt.shape[0])
    for ev_step, gen_pos, token in events:
        if int(ev_step) >= int(step):
            continue
        x[prompt_len + int(gen_pos)] = int(token)
    gen_mask = x[prompt_len:] == int(mask_id)
    attention_mask = np.ones(x.shape[0], dtype=np.int64)
    attention_mask[:prompt_len] = prompt != int(pad_id)
    return x, gen_mask, attention_mask
