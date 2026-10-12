"""Best-of-K stochastic trajectory reranking by SchedulerLens G-cap."""

from __future__ import annotations

from typing import Any, Callable

import torch

from schedulerlens.proxies import compute_sample_proxies, cumulative_caps
from traces.trace_checks import repair_drop_last_step
from traces.trace_generator import TracingContext
from traces.trace_wrapper import ModelTracer, _task_trace_list_ctx

SUPPORTED_SCORES = ("raw", "per_commit")

# Filled by the most recent ``generate_gcap_rerank`` call.
last_rerank_metrics: list[dict[str, Any]] = []
last_random_output: torch.Tensor | None = None
last_candidate_outputs: torch.Tensor | None = None


def _run_traced_candidate(
    candidate_fn: Callable,
    model,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    mask_id: int,
    gen_length: int,
    block_length: int | None,
):
    traces: list[dict[str, Any]] = []
    token = _task_trace_list_ctx.set(traces)
    try:
        context = TracingContext(save_to_disk=False, trace_name="gcap_candidate")
        tracer = ModelTracer(
            candidate_fn,
            mask_id=mask_id,
            tracing_context=context,
        )
        result = tracer(
            model,
            prompt,
            attention_mask,
            gen_length=gen_length,
            block_length=block_length,
        )
    finally:
        _task_trace_list_ctx.reset(token)
    if len(traces) != int(prompt.shape[0]):
        raise RuntimeError(
            "G-cap candidate trace count does not match batch size: "
            f"{len(traces)} != {int(prompt.shape[0])}"
        )
    return result, traces


def _candidate_metrics(
    trace: dict[str, Any],
    score_mode: str,
    selection_prefix_steps: int,
) -> dict[str, Any]:
    repaired, _repairs = repair_drop_last_step(trace)
    proxies = compute_sample_proxies(repaired)
    commits = sum(step.n_selected for step in proxies.step_proxies)
    g_cap = float(proxies.M_hat + proxies.J_hat)
    prefix_m, prefix_j, prefix_g = cumulative_caps(
        proxies.step_proxies,
        selection_prefix_steps,
    )
    prefix_commits = sum(
        step.n_selected
        for step in proxies.step_proxies
        if int(step.step) < int(selection_prefix_steps)
    )
    if score_mode == "raw":
        selection_score = prefix_g
    else:
        selection_score = prefix_g / float(max(prefix_commits, 1))
    return {
        "M_hat": float(proxies.M_hat),
        "J_hat": float(proxies.J_hat),
        "G_hat": g_cap,
        "prefix_steps": int(selection_prefix_steps),
        "prefix_M_hat": float(prefix_m),
        "prefix_J_hat": float(prefix_j),
        "prefix_G_hat": float(prefix_g),
        "prefix_n_committed": int(prefix_commits),
        "selection_score": selection_score,
        "n_committed": int(commits),
        "n_steps": len(proxies.step_proxies),
        "step_proxies": [
            {
                "step": int(step.step),
                "M_hat_t": float(step.M_hat_t),
                "J_hat_t": float(step.J_hat_t),
                "n_committed": int(step.n_committed),
                "n_selected": int(step.n_selected),
                "n_remaining": int(step.n_remaining),
            }
            for step in proxies.step_proxies
        ],
    }


@torch.no_grad()
def generate_gcap_rerank(
    model,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    candidate_fn: Callable,
    k: int,
    score_mode: str,
    selection_seed: int,
    selection_prefix_steps: int,
    mask_id: int,
    gen_length: int,
    block_length: int | None,
):
    """Generate one pool and expose both minimum-G and random selections."""
    global last_candidate_outputs, last_random_output, last_rerank_metrics
    last_candidate_outputs = None
    last_random_output = None
    last_rerank_metrics = []
    if int(k) < 2:
        raise ValueError(f"G-cap reranking requires k >= 2, got {k}")
    if score_mode not in SUPPORTED_SCORES:
        raise ValueError(
            f"unknown G-cap score mode {score_mode!r}; expected {SUPPORTED_SCORES}"
        )

    candidates: list[torch.Tensor] = []
    metrics_by_candidate: list[list[dict[str, Any]]] = []
    total_steps = 0
    total_blocks = 0
    for _candidate_index in range(int(k)):
        result, traces = _run_traced_candidate(
            candidate_fn,
            model,
            prompt,
            attention_mask,
            mask_id=mask_id,
            gen_length=gen_length,
            block_length=block_length,
        )
        generated, steps_taken, blocks = result
        candidates.append(generated)
        metrics_by_candidate.append(
            [
                _candidate_metrics(trace, score_mode, selection_prefix_steps)
                for trace in traces
            ]
        )
        total_steps += int(steps_taken)
        total_blocks += int(blocks)

    batch = int(prompt.shape[0])
    min_indices = [
        min(
            range(int(k)),
            key=lambda candidate: metrics_by_candidate[candidate][row][
                "selection_score"
            ],
        )
        for row in range(batch)
    ]
    prompt_cpu = prompt.detach().to(device="cpu", dtype=torch.long)
    weights = torch.arange(1, prompt_cpu.shape[1] + 1, dtype=torch.long)
    row_seeds = (prompt_cpu * weights.unsqueeze(0)).sum(dim=1) + int(selection_seed)
    random_indices = []
    for row_seed in row_seeds.tolist():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(row_seed) % (2**63 - 1))
        random_indices.append(int(torch.randint(int(k), (), generator=generator).item()))

    stacked = torch.stack(candidates, dim=0)
    last_candidate_outputs = stacked
    rows = torch.arange(batch, device=prompt.device)
    min_selected = torch.tensor(min_indices, dtype=torch.long, device=prompt.device)
    random_selected = torch.tensor(
        random_indices, dtype=torch.long, device=prompt.device
    )
    output = stacked[min_selected, rows]
    last_random_output = stacked[random_selected, rows]
    last_rerank_metrics = [
        {
            "score_mode": score_mode,
            "selection_prefix_steps": int(selection_prefix_steps),
            "min_g_index": int(min_indices[row]),
            "random_index": int(random_indices[row]),
            "candidates": [
                metrics_by_candidate[candidate][row]
                for candidate in range(int(k))
            ],
        }
        for row in range(batch)
    ]
    return output, total_steps, total_blocks
