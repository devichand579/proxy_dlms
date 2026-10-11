"""Separate latent cardinality control from position ranking.

Risk uses confidence and incoming-motion proxies. The response
Gram is an observed co-response screen, not a proven MI/TC estimate; discrete
response probes remain necessary to validate that approximation.
"""
from __future__ import annotations

import torch

from schedulers.latent import _eb_prefix, _fisher_basis, _fisher_project, _target_temporal_risk, _unit
from schedulers.utils import get_mask_by_confidence, token_entropy
from schedulers.temporal_divergence import temporal_divergence

last_metrics = {}


def latent_prefix(positions, ranking, confidence, motion, features, budget=.03, coupling_beta=.25):
    """Largest prefix under a risk-plus-co-response budget; forced singleton if needed."""
    if not len(positions):
        return positions, False, 0.
    order = ranking.argsort(descending=True, stable=True)
    error_proxy = (1 - confidence[order]).clamp_min(0)
    response = motion[order].clamp(0, 1)
    individual = error_proxy * (1 + .15 * response)
    vectors = _unit(features[order])
    # Both aligned and opposed responses can indicate coupling.
    overlap = (vectors @ vectors.T).square()
    stakes = error_proxy.sqrt() * response
    pair = overlap * stakes[:, None] * stakes[None, :]
    pair = pair.tril(diagonal=-1)
    costs = individual.cumsum(0) + coupling_beta * pair.sum(1).cumsum(0)
    count = int((costs <= budget).sum())
    forced = count == 0
    count = max(count, 1)
    return positions[order[:count]], forced, float(costs[count - 1])


@torch.no_grad()
def generate_factorized(model, prompt, attention_mask, *, cells=None, risk_budgets=None,
                        steps=128, gen_length=512, mask_id=126336, eos_id=126081,
                        rank_beta=.15, coupling_beta=.25, fisher_vocab=128,
                        initial_canvas=None, full_attention=None, progress=None,
                        rank_mode="risk", support_beta=.1, risk_readout=None,
                        motion_beta=.25, content_beta=.25, value_readout=None, value_beta=.25,
                        switch_step=64):
    """Cells: eb_eb, eb_latent, latent_eb, latent_latent, one setting per row.

    In eb_latent, k is frozen from EB's confidence-ranked set, then exactly k
    positions are chosen by latent ranking. In latent_latent the risk budget
    is checked on the actual latent-ranked prefix. This exposes the interaction.
    """
    global last_metrics
    batch, prefix = prompt.shape
    cells = cells or ["eb_eb"] * batch
    budgets = risk_budgets or [.03] * batch
    if len(cells) != batch or len(budgets) != batch:
        raise ValueError("one count/rank configuration is required per row")
    if any(c not in ("eb_eb", "eb_cap1", "eb_cap2", "eb_cap4", "warm_eb", "eb_latent", "latent_eb", "latent_latent",
                     "motion_eb", "motion_latent",
                     "eb_content", "content_entropy",
                     "eb_value", "value_entropy",
                     "fisher_entropy", "kl_eb", "kl_entropy", "js_entropy",
                     "eb_learned", "learned_eb", "learned_learned", "learned_latent",
                     "late_learned_latent") for c in cells):
        raise ValueError("unknown factorial cell")
    if any("learned" in c or "content" in c for c in cells) and risk_readout is None:
        raise ValueError("learned count/rank cells require a risk readout")
    if any("value" in c for c in cells) and value_readout is None:
        raise ValueError("value cells require a reveal-value readout")
    if rank_mode not in ("risk", "support"):
        raise ValueError("rank mode must be risk or support")
    if initial_canvas is None:
        x = torch.full((batch, prefix + gen_length), mask_id, device=prompt.device, dtype=torch.long)
        x[:, :prefix] = prompt
    else:
        x = initial_canvas.clone()
    attention = torch.cat([attention_mask, torch.ones(batch, gen_length, dtype=attention_mask.dtype, device=prompt.device)], -1)
    if full_attention is not None:
        attention = full_attention
    head = model.get_output_embeddings()
    previous = None
    previous_logits = None
    cache_logits = any(c.startswith(('kl_','js_')) for c in cells)
    previous_counts = torch.ones(batch, device=x.device)
    ema = torch.full((batch,), 1e-8, device=x.device)
    metrics = {"sequence_evaluations": 0, "forward_calls": 0, "steps_taken": 0,
               "cells": cells, "risk_budgets": budgets, "rank_beta": rank_beta,
               "coupling_beta": coupling_beta, "fisher_vocab": fisher_vocab,
               "rank_mode": rank_mode, "support_beta": support_beta,
               "motion_beta": motion_beta,
               "content_beta": content_beta,
               "value_beta": value_beta,
               "switch_step": switch_step,
               "selected_counts": [], "forced_singletons": [0] * batch,
               "bootstrap_singletons": [0] * batch, "selected_costs": []}
    for step in range(steps):
        masked = x == mask_id
        if not masked[:, prefix:].any():
            break
        captured = []
        hook = head.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0].detach()))
        try:
            out = model(x, attention_mask=attention)
        finally:
            hook.remove()
        logits, hidden = out.logits, captured[0]
        logits[..., mask_id] = -torch.inf
        predicted = logits.argmax(-1)
        confidence = get_mask_by_confidence(logits, predicted)
        entropy = token_entropy(logits, eps=1e-9)
        reveal = torch.zeros_like(masked)
        counts, costs = [], []
        for row in range(batch):
            pos = masked[row, prefix:].nonzero().flatten() + prefix
            if not len(pos):
                counts.append(0); costs.append(0.); continue
            conf = confidence[row, pos]
            estimated_error = None
            body_probability = None
            predicted_value = None
            if "learned" in cells[row]:
                estimated_error = risk_readout.predict(hidden[row, pos], logits[row, pos],
                    model.get_input_embeddings().weight, eos_id).clamp(1e-6, 1 - 1e-6)
            if "content" in cells[row]:
                body_probability = risk_readout.body_probability(hidden[row, pos], logits[row, pos],
                    model.get_input_embeddings().weight, eos_id).clamp_min(1e-6)
            if "value" in cells[row]:
                predicted_value = value_readout.predict(hidden[row, pos], logits[row, pos],
                    model.get_input_embeddings().weight, eos_id, hidden[row, pos])
            eb_set = _eb_prefix(pos, conf, entropy[row, pos])
            ranking = conf.clamp_min(1e-30).log()
            normalized_motion = torch.zeros(len(pos), device=x.device)
            risk = torch.zeros_like(normalized_motion)
            normalized_support = torch.zeros_like(normalized_motion)
            coupling_motion = torch.zeros_like(normalized_motion)
            features = hidden.new_zeros((len(pos), fisher_vocab), dtype=torch.float32)
            if previous is not None and cells[row] not in ("eb_eb", "eb_cap1", "eb_cap2", "eb_cap4", "warm_eb"):
                delta = hidden[row, pos].float() - previous[row, pos].float()
                exact_metric = 'kl' if cells[row].startswith('kl_') else ('js' if cells[row].startswith('js_') else None)
                if exact_metric:
                    risk = temporal_divergence(logits[row,pos],previous_logits[row,pos],exact_metric)
                    signals = (risk, torch.zeros_like(risk)) if rank_mode == 'support' else risk
                else:
                    signals = _target_temporal_risk(logits[row, pos], hidden[row, pos], previous[row, pos], head.weight,
                        fisher_vocab, return_support=rank_mode == "support")
                if rank_mode == "support":
                    risk, support = signals
                    normalized_support = (support / support.abs().mean().clamp_min(1e-8)).clamp(-4, 4)
                else:
                    risk = signals
                normalized_motion = (risk / risk.mean().clamp_min(1e-8)).clamp(max=4)
                # Normalize accumulated movement by the previous actual reveal count.
                per_reveal = risk / previous_counts[row].clamp_min(1)
                ema[row] = .9 * ema[row] + .1 * per_reveal.mean()
                coupling_motion = per_reveal / (ema[row] + per_reveal).clamp_min(1e-8)
                if cells[row].startswith(("latent_", "learned_", "late_learned_")):
                    p, w, _ = _fisher_basis(logits[row, pos], head.weight, fisher_vocab)
                    features = _fisher_project(delta, p, w)
            row_rank_beta = rank_beta[row] if isinstance(rank_beta, (list, tuple)) else rank_beta
            latent_ranking = ranking - row_rank_beta * normalized_motion
            if rank_mode == "support":
                latent_ranking += support_beta * normalized_support
            learned_ranking = (1 - estimated_error).log() - row_rank_beta * normalized_motion if estimated_error is not None else None
            if cells[row] == "eb_eb":
                selected = eb_set; cost = 0.
            elif cells[row].startswith("eb_cap"):
                selected = eb_set[:int(cells[row][6:])]; cost = 0.
            elif cells[row] == "warm_eb":
                selected = eb_set[:1] if previous is None else eb_set
                cost = 0.
                if previous is None:
                    metrics["bootstrap_singletons"][row] += 1
            elif cells[row] == "eb_latent":
                selected = pos[latent_ranking.argsort(descending=True, stable=True)[:len(eb_set)]]; cost = 0.
            elif cells[row] in ('fisher_entropy','kl_entropy','js_entropy','kl_eb'):
                selected = (pos[latent_ranking.argsort(descending=True, stable=True)[:len(eb_set)]]
                            if cells[row] == 'kl_eb' else _eb_prefix(pos,latent_ranking,entropy[row,pos]))
                cost = 0.
            elif cells[row] == 'late_learned_latent' and step < switch_step:
                selected = _eb_prefix(pos,latent_ranking,entropy[row,pos]); cost = 0.
            elif cells[row] == "eb_learned":
                selected = pos[learned_ranking.argsort(descending=True, stable=True)[:len(eb_set)]]; cost = 0.
            elif "content" in cells[row]:
                # Likely code-body membership is a progress proxy, not outgoing MI.
                ordering = latent_ranking + content_beta * body_probability.log()
                selected = (pos[ordering.argsort(descending=True, stable=True)[:len(eb_set)]]
                            if cells[row] == "eb_content" else _eb_prefix(pos, ordering, entropy[row, pos]))
                cost = 0.
            elif "value" in cells[row]:
                row_value_beta = value_beta[row] if isinstance(value_beta, (list, tuple)) else value_beta
                ordering = latent_ranking + row_value_beta * predicted_value.clamp(-4, 4)
                selected = (pos[ordering.argsort(descending=True, stable=True)[:len(eb_set)]]
                            if cells[row] == "eb_value" else _eb_prefix(pos, ordering, entropy[row, pos]))
                cost = 0.
            elif cells[row].startswith("motion_"):
                # Add observed predictive KL movement to the entropy budget.
                # This is a local instability surrogate, not a TC upper bound.
                ordering = ranking if cells[row] == "motion_eb" else latent_ranking
                selected = _eb_prefix(pos, ordering, entropy[row, pos] + motion_beta * risk)
                cost = 0.
            else:
                ordering = ranking if cells[row] in ("latent_eb", "learned_eb") else (learned_ranking if cells[row] == "learned_learned" else latent_ranking)
                count_confidence = 1 - estimated_error if cells[row].startswith(("learned_","late_learned_")) else conf
                if previous is None:
                    selected = pos[ordering.argmax()].reshape(1)
                    cost = float(1 - count_confidence[ordering.argmax()])
                    metrics["bootstrap_singletons"][row] += 1
                else:
                    selected, forced, cost = latent_prefix(pos, ordering, count_confidence, coupling_motion,
                        features, budget=budgets[row], coupling_beta=coupling_beta)
                    metrics["forced_singletons"][row] += int(forced)
            reveal[row, selected] = True
            counts.append(len(selected)); costs.append(cost)
        x[reveal] = predicted[reveal]
        previous = hidden.detach()
        previous_logits = logits.detach() if cache_logits else None
        previous_counts = torch.tensor(counts, device=x.device, dtype=torch.float32)
        metrics["selected_counts"].append(counts)
        metrics["selected_costs"].append(costs)
        metrics["forward_calls"] += 1
        metrics["sequence_evaluations"] += batch
        metrics["steps_taken"] += 1
        del out, logits, predicted, confidence, entropy, captured, hidden
        if progress and (step % 16 == 15 or step == steps - 1):
            progress(dict(metrics))
    metrics["unfinished_masks_by_row"] = (x[:, prefix:] == mask_id).sum(-1).tolist()
    last_metrics = metrics
    return x
