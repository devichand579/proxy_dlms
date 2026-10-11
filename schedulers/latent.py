"""Experimental latent and selectively probed scheduling at fixed cardinality.

Geometry is a heuristic for redundancy, not a mutual-information estimate.
Temporal Fisher movement measures persistence, not correctness. Probe entropy
is evaluated on a common survivor set; its token-risk cost is kept separate.
"""
from __future__ import annotations

import torch

from schedulers.utils import (
    check_eos_stop_condition, get_num_transfer_tokens, get_mask_by_confidence,
    mask_eos_eot_logits, mask_invalid_token_logits, token_entropy,
)

VARIANTS = ("entropy", "raw_diversity", "fisher_diversity", "temporal", "target_temporal",
            "gradient_gain", "response_gain", "response_ready", "hybrid_probe", "eb_confidence",
            "eb_temporal", "eb_target_temporal", "eb_gradient_gain", "eb_response_ready",
            "eb_latent_support", "eb_temporal_support")
last_metrics = {}


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _fisher_basis(logits, weight, vocab=128, return_tokens=False):
    """Arithmetic mixture of masked-position probabilities, then truncation.

    Shared metric for this geometry control; it is not per-position full Fisher.
    Keep tail mass as a diagnostic rather than concealing renormalization.
    """
    mixture = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.float32)
    for start in range(0, logits.shape[0], 32):
        mixture += logits[start:start + 32].float().softmax(-1).sum(0)
    mixture /= logits.shape[0]
    probability, tokens = mixture.topk(min(vocab, mixture.numel()))
    retained = probability.sum()
    result = (probability / retained, weight[tokens].float(), float(retained))
    return (*result, tokens) if return_tokens else result


def _fisher_project(hidden, probability, weight):
    change = hidden.float() @ weight.T
    # Remove the softmax-invariant common logit shift before averaging.
    change = change - change[:, :1]
    return (change - (change @ probability).unsqueeze(-1)) * probability.sqrt()


def _greedy(log_conf, features, budget, diversity_beta=.25, temporal=None, stability_beta=.15):
    """Risk-aware diversity within a confidence shortlist, with a fixed budget."""
    score = log_conf.clone()
    if temporal is not None:
        # Normalize a scale-free movement proxy; do not interpret this as KL risk.
        scale = temporal.mean().clamp_min(1e-8)
        score -= stability_beta * (temporal / scale).clamp(max=4)
    shortlist = score.topk(min(score.numel(), max(budget, 4 * budget))).indices
    local_score = score[shortlist].clone()
    vectors = _unit(features[shortlist]) if features is not None else None
    chosen = []
    for _ in range(min(budget, len(shortlist))):
        best = int(local_score.argmax())
        chosen.append(shortlist[best])
        if vectors is not None:
            # Negative correlations are not rewarded as negative dependence.
            redundancy = (vectors @ vectors[best]).clamp_min(0).square()
            local_score -= diversity_beta * redundancy
        local_score[best] = -torch.inf
    return torch.stack(chosen)


def _probe_utility(before, after, common_targets, source_log_conf):
    gain = (before[common_targets] - after[common_targets]).mean() if common_targets.any() else before.new_tensor(0.)
    return gain + .5 * source_log_conf.mean()


def _eb_prefix(positions, ranking, entropy, gamma=.1):
    """Same prefix entropy constraint as the existing EB sampler, stable ties."""
    if not positions.numel():
        return positions
    order = ranking.argsort(descending=True, stable=True)
    h = entropy[order].double()
    admissible = h.cumsum(0) - h.cummax(0).values <= gamma
    return positions[order[:int(admissible.sum())]]


def _target_temporal_risk(logits, states, previous, weight, vocab, return_support=False):
    """Target-specific truncated Fisher of the actual latent temporal change."""
    risks, supports = [], []
    for start in range(0, len(states), 32):
        stop = min(start + 32, len(states))
        values, indices = logits[start:stop].float().topk(min(vocab, logits.shape[-1]), -1)
        p = values.softmax(-1)
        delta = states[start:stop].float() - previous[start:stop].float()
        changes = torch.einsum("nd,nvd->nv", delta, weight[indices].float())
        changes = changes - changes[:, :1]
        expected = (p * changes).sum(-1, keepdim=True)
        risk = .5 * (p * (changes - expected).square()).sum(-1)
        risks.append(risk)
        if return_support:
            # Signed latent motion towards the token currently being proposed.
            supports.append(-expected[:, 0])
    result = torch.cat(risks)
    return (result, torch.cat(supports)) if return_support else result


def _gradient_score_bonus(relative_gain, confidence, normalized_risk, risk_eta=0., readiness_gate=0.):
    positive = relative_gain.clamp_min(0) * (-readiness_gate * normalized_risk).exp()
    benefit = positive + relative_gain.clamp_max(0)
    damage = (1-confidence) * relative_gain.abs()
    return benefit - risk_eta * damage


def _gradient_reveal_gain(model, ids, attention, masked, shortlist, predicted, mask_id, metrics, metric_row_offset=0):
    """One shared-target backward, excluding every shortlisted source position.

    This estimates a finite reveal action by a local input-embedding derivative.
    It is not an exact discrete response or a candidate-specific full-target MI.
    """
    targets = masked.clone()
    for row, positions in enumerate(shortlist):
        targets[row, positions] = False
    active = targets.any(-1).nonzero().flatten()
    result = torch.zeros(ids.shape, device=ids.device, dtype=torch.float32)
    if not active.numel():
        return result
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.requires_grad_(False)
    try:
        with torch.enable_grad():
            embedding = model.get_input_embeddings()(ids[active]).detach().requires_grad_(True)
            out = model(ids[active], inputs_embeds=embedding, attention_mask=attention[active])
            loss = embedding.new_tensor(0., dtype=torch.float32)
            denominator = targets[active].sum().float()
            for row in range(len(active)):
                positions = targets[active[row]].nonzero().flatten()
                for offset in range(0, len(positions), 16):
                    piece = out.logits[row, positions[offset:offset + 16]].float()
                    logp = torch.log_softmax(piece, -1)
                    loss -= (logp.exp() * logp).sum() / denominator
            gradient = torch.autograd.grad(loss, embedding)[0]
            weights = model.get_input_embeddings().weight
            for local_row, row in enumerate(active.tolist()):
                positions = shortlist[row]
                delta = weights[predicted[row, positions]].float() - weights[mask_id].float()
                result[row, positions] = -(gradient[local_row, positions].float() * delta).sum(-1)
            metrics["forward_calls"] += 1
            metrics["sequence_evaluations"] += len(active)
            metrics["backward_passes"] += 1
            metrics["backward_sequence_evaluations"] += len(active)
            if "forward_work_by_row" in metrics:
                for row in active.tolist():
                    metrics["forward_work_by_row"][row + metric_row_offset] += 1
                    metrics["backward_work_by_row"][row + metric_row_offset] += 1
    finally:
        for p in trainable:
            p.requires_grad_(True)
    return result


def _response_selection(model, forward, x, attention, logits, hidden, conf, entropy,
                        masked, start, end, budgets, metrics, vocab, strength,
                        ready=False, incoming_beta=.2):
    """Discrete single-source branches, latent dependence/coherence separation.

    Fisher response variance is computed across token alternatives at each
    target. The shared vocabulary is truncated, but probabilities inside that
    vocabulary are target-specific. These are local model diagnostics.
    """
    batch = len(x)
    positions = [(masked[row, start:end]).nonzero().flatten() + start for row in range(batch)]
    counts = [len(p) for p in positions]
    k = int(budgets[0])
    fallback = [pos[conf[row, pos].topk(min(int(budgets[row]), len(pos))).indices] for row, pos in enumerate(positions)]
    if not k or (not ready and (len(set(counts)) != 1 or not bool((budgets == k).all()) or counts[0] <= k)):
        return fallback
    rows = torch.arange(batch, device=x.device)
    if ready:
        positions = torch.arange(start, end, device=x.device).repeat(batch, 1)
        sources = 8
        pool, source_valid = [], []
        for row in range(batch):
            live = masked[row, start:end].nonzero().flatten()
            size = min(sources, len(live))
            chosen = live[conf[row, start:end][live].topk(size).indices]
            pool.append(torch.cat([chosen, chosen[-1:].repeat(sources - size)]))
            source_valid.append(torch.arange(sources, device=x.device) < size)
        pool, source_valid = torch.stack(pool), torch.stack(source_valid)
    else:
        positions = torch.stack(positions)
        sources = min(positions.shape[1], 2 * k)
        pool = torch.stack([conf[row, positions[row]].topk(sources).indices for row in range(batch)])
        source_valid = torch.ones((batch, sources), device=x.device, dtype=torch.bool)
    remaining = positions.shape[1]
    source_positions = positions.gather(1, pool)
    source_logits = logits[rows[:, None], source_positions].float()
    alternative_logits, alternative_tokens = source_logits.topk(2, -1)
    rho = alternative_logits.softmax(-1)
    source_conf = conf.gather(1, source_positions)
    retained_sources = source_conf / rho[..., 0].clamp_min(1e-30)
    metrics["response_omitted_source_mass_sum"] += float(((1 - retained_sources).clamp_min(0) * source_valid).sum())
    metrics["response_sources_measured"] += int(source_valid.sum())
    weights, target_probs, baseline_projection = [], [], []
    for row in range(batch):
        live_targets = masked[row, positions[row]]
        _, weight, _, tokens = _fisher_basis(logits[row, positions[row][live_targets]], model.get_output_embeddings().weight, vocab, return_tokens=True)
        selected_logits = logits[row, positions[row]][:, tokens].float()
        maximum = logits[row, positions[row]].amax(-1).float()
        logz = maximum - conf[row, positions[row]].clamp_min(1e-30).log()
        actual_p = (selected_logits - logz[:, None]).exp()
        retained = actual_p.sum(-1)
        metrics["response_target_retained_mass_sum"] += float(retained[live_targets].sum())
        metrics["response_targets_measured"] += int(live_targets.sum())
        target_probs.append(actual_p / retained[:, None].clamp_min(1e-30))
        weights.append(weight)
        baseline_projection.append(hidden[row, positions[row]].float() @ weight.T)
    weights = torch.stack(weights)
    target_probs = torch.stack(target_probs)
    baseline_projection = torch.stack(baseline_projection)
    hypotheses = 2 * sources
    original_rows = rows.repeat(hypotheses)
    source_index = torch.arange(hypotheses, device=x.device).repeat_interleave(batch) // 2
    value_index = torch.arange(hypotheses, device=x.device).repeat_interleave(batch) % 2
    branches = x.repeat(hypotheses, 1)
    branches[torch.arange(len(branches), device=x.device), source_positions[original_rows, source_index]] = alternative_tokens[original_rows, source_index, value_index]
    projections = torch.zeros((len(branches), remaining, weights.shape[1]), device=x.device)
    entropies = torch.zeros((len(branches), remaining), device=x.device)
    valid_hypotheses = source_valid[original_rows, source_index].nonzero().flatten()
    metrics["response_sequence_evaluations"] += len(valid_hypotheses)
    # Up to 80 full sequences per call; use the 192 GiB GPU without retaining all logits.
    for offset in range(0, len(valid_hypotheses), 80):
        selected_hypotheses = valid_hypotheses[offset:offset + 80]
        original = original_rows[selected_hypotheses]
        future_logits, future_hidden = forward(branches[selected_hypotheses], attention[original])
        future_entropy = token_entropy(future_logits)
        local_rows = torch.arange(len(selected_hypotheses), device=x.device)
        target_positions = positions[original]
        entropies[selected_hypotheses] = future_entropy[local_rows[:, None], target_positions]
        target_hidden = future_hidden[local_rows[:, None], target_positions].float()
        projections[selected_hypotheses] = torch.bmm(target_hidden, weights[original].transpose(1, 2))
        del future_logits, future_hidden, future_entropy, target_hidden
    projections = projections.reshape(sources, 2, batch, remaining, -1).permute(2, 0, 1, 3, 4)
    entropies = entropies.reshape(sources, 2, batch, remaining).permute(2, 0, 1, 3)
    difference = projections[:, :, 0] - projections[:, :, 1]
    p = target_probs[:, None]
    expected_difference = (p * difference).sum(-1)
    variance = (p * difference.square()).sum(-1) - expected_difference.square()
    dependence = .5 * (rho[..., 0] * rho[..., 1])[..., None] * variance.clamp_min(0)
    mean_delta = (rho[..., None, None] * (projections - baseline_projection[:, None, None])).sum(2)
    mean = (p * mean_delta).sum(-1)
    defect = .5 * ((p * mean_delta.square()).sum(-1) - mean.square()).clamp_min(0)
    selected = []
    for row in range(batch):
        common = masked[row, positions[row]].clone()
        common[pool[row]] = False
        gain = (entropy[row, positions[row]][None] - entropies[row, :, 0])[:, common].mean(-1) if common.any() else torch.zeros(sources, device=x.device)
        pair = dependence[row].index_select(1, pool[row]).clone()
        pair.fill_diagonal_(0)
        pair *= source_valid[row][:, None] * source_valid[row][None, :]
        defect_pair = defect[row].index_select(1, pool[row]).clone()
        defect_pair.fill_diagonal_(0)
        defect_pair *= source_valid[row][:, None] * source_valid[row][None, :]
        incoming = ((pair + defect_pair).sum(0) / max(int(source_valid[row].sum()) - 1, 1)) if ready else pair.mean(0)
        outgoing = dependence[row][:, common].mean(-1) if common.any() else torch.zeros_like(gain)
        incoherence = defect[row][:, common].mean(-1) if common.any() else torch.zeros_like(gain)

        def normalize(value):
            denominator = value[source_valid[row]].abs().mean() if ready else value.abs().mean()
            return (value / denominator.clamp_min(1e-8)).clamp(-4, 4)

        score = source_conf[row].clamp_min(1e-30).log()
        score += .5 * strength * normalize(gain)
        score -= (incoming_beta if ready else .05) * normalize(incoming)
        if ready:
            # Mean response contributes to target readiness and propagation risk,
            # not an unconditional penalty against a potentially corrective source.
            score += .1 * normalize(outgoing)
            score -= .1 * (1 - source_conf[row]) * normalize(outgoing + incoherence)
        else:
            score -= .05 * (1 - source_conf[row]) * normalize(outgoing)
            score -= .05 * normalize(incoherence)
        score[~source_valid[row]] = -torch.inf
        symmetric = .5 * (pair + pair.T)
        if ready:
            valid_pair = source_valid[row][:, None] & source_valid[row][None, :]
            valid_pair.fill_diagonal_(False)
            scale = symmetric[valid_pair].mean() if valid_pair.any() else symmetric.new_tensor(0.)
        else:
            scale = symmetric.mean()
        symmetric = (symmetric / scale.clamp_min(1e-8)).clamp(max=4)
        picked = []
        for _ in range(int(budgets[row])):
            best = int(score.argmax())
            picked.append(best)
            score -= .25 * strength * symmetric[:, best]
            score[best] = -torch.inf
        selected.append(source_positions[row, torch.tensor(picked, device=x.device)])
    metrics["response_probe_steps"] += 1
    return selected


@torch.no_grad()
def generate_latent(
    model, prompt, attention_mask=None, *, steps=128, gen_length=512,
    block_length=512, variant="fisher_diversity", temperature=0.,
    mask_id=126336, eos_id=126081, eot_id=126348,
    logits_eos_inf=False, confidence_eos_eot_inf=False,
    use_eos_harness=True, invalid_token_ids=None, diversity_beta=.25,
    stability_beta=.15, probe_every=8, fisher_vocab=128, gain_beta=.25,
    response_incoming_beta=.2,
    gradient_pool_min=0, gradient_temporal=False, gradient_probe_batch_size=0,
    gradient_risk_eta=0., gradient_readiness_gate=0.,
    progress=None, initial_canvas=None, canvas_attention_mask=None, **trace_kwargs,
):
    global last_metrics
    if variant not in VARIANTS:
        raise ValueError(f"unknown latent variant {variant}")
    adaptive_eb = variant.startswith("eb_")
    original_variant = variant
    if adaptive_eb:
        variant = variant[len("eb_"):]
    if temperature != 0:
        raise ValueError("initial latent experiments use greedy tokens")
    if not 1 <= steps <= gen_length or block_length <= 0 or probe_every <= 0:
        raise ValueError("invalid generation budget")
    if gen_length % block_length or steps % (gen_length // block_length):
        raise ValueError("generation length, blocks and steps must divide exactly")
    if getattr(model.config, "model_type", None) != "llada":
        raise ValueError("latent head capture currently supports LLaDA checkpoints")
    head = model.get_output_embeddings()
    if getattr(model.config, "weight_tying", False):
        raise ValueError("head capture requires a separate output head")
    batch, prompt_len = prompt.shape
    if gradient_probe_batch_size < 0:
        raise ValueError("gradient probe batch size must be nonnegative")
    pool_values = gradient_pool_min if isinstance(gradient_pool_min, (list,tuple)) else [gradient_pool_min] * batch
    temporal_values = gradient_temporal if isinstance(gradient_temporal, (list,tuple)) else [gradient_temporal] * batch
    gain_values = gain_beta if isinstance(gain_beta, (list,tuple)) else [gain_beta] * batch
    risk_values = gradient_risk_eta if isinstance(gradient_risk_eta, (list,tuple)) else [gradient_risk_eta] * batch
    gate_values = gradient_readiness_gate if isinstance(gradient_readiness_gate, (list,tuple)) else [gradient_readiness_gate] * batch
    if len(pool_values) != batch or len(temporal_values) != batch or len(gain_values) != batch or any(v < 0 for v in pool_values):
        raise ValueError("one nonnegative gradient pool and configuration per row is required")
    if len(risk_values) != batch or len(gate_values) != batch or any(v < 0 for v in (*risk_values,*gate_values)):
        raise ValueError("one nonnegative risk and readiness weight per row is required")
    if any(g and not t for g,t in zip(gate_values,temporal_values)):
        raise ValueError("gradient readiness gating requires temporal features")
    if initial_canvas is None:
        x = torch.full((batch, prompt_len + gen_length), mask_id, dtype=torch.long, device=prompt.device)
        x[:, :prompt_len] = prompt
    else:
        if initial_canvas.shape != (batch, prompt_len + gen_length):
            raise ValueError("initial canvas shape does not match the generation region")
        if not torch.equal(initial_canvas[:, :prompt_len], prompt):
            raise ValueError("initial canvas must preserve the supplied prompt")
        x = initial_canvas.clone()
    attention = torch.cat([attention_mask, torch.ones(batch, gen_length, device=prompt.device, dtype=attention_mask.dtype)], -1) if attention_mask is not None else torch.ones_like(x)
    if canvas_attention_mask is not None:
        if canvas_attention_mask.shape != x.shape:
            raise ValueError("canvas attention mask must match the full canvas")
        attention = canvas_attention_mask
    metrics = {"variant": original_variant, "forward_calls": 0, "sequence_evaluations": 0,
               "cardinality": "eb_entropy_budget" if adaptive_eb else "fixed_linear_quota",
               "probe_steps": 0, "probed_samples": 0, "probe_switches": 0,
               "backward_passes": 0, "backward_sequence_evaluations": 0,
               "response_probe_steps": 0, "response_sources_measured": 0,
               "response_sequence_evaluations": 0,
               "response_omitted_source_mass_sum": 0., "response_targets_measured": 0,
               "response_target_retained_mass_sum": 0.,
               "steps_taken": 0, "fisher_retained_mass_sum": 0., "fisher_measurements": 0,
               "diversity_beta": diversity_beta, "stability_beta": stability_beta,
               "probe_every": probe_every, "fisher_vocab": fisher_vocab}
    metrics["gain_beta"] = gain_beta
    metrics["response_incoming_beta"] = response_incoming_beta
    metrics.update(gradient_pool_min=gradient_pool_min, gradient_temporal=gradient_temporal,
                   gradient_probe_batch_size=gradient_probe_batch_size,
                   gradient_risk_eta=gradient_risk_eta,gradient_readiness_gate=gradient_readiness_gate)
    if variant == "gradient_gain":
        metrics.update(forward_work_by_row=[0] * batch, backward_work_by_row=[0] * batch,
                       gradient_sources_by_row=[0] * batch, gradient_measurements_by_row=[0] * batch)
    need_hidden = variant not in ("confidence", "entropy", "gradient_gain") or (variant == "gradient_gain" and any(temporal_values))

    def forward(ids, attn):
        captured = []
        hook = head.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0].detach())) if need_hidden else None
        try:
            out = model(ids, attention_mask=attn)
        finally:
            if hook is not None:
                hook.remove()
        metrics["forward_calls"] += 1
        metrics["sequence_evaluations"] += ids.shape[0]
        if "forward_work_by_row" in metrics:
            for row in range(batch): metrics["forward_work_by_row"][row] += 1
        lg = out.logits
        mask_eos_eot_logits(lg, logits_eos_inf, confidence_eos_eot_inf, eos_id, eot_id)
        mask_invalid_token_logits(lg, invalid_token_ids)
        lg[..., mask_id] = -torch.inf
        hidden = captured[0] if captured else None
        return lg, hidden

    previous = None
    cached = None
    nblocks = gen_length // block_length
    block_steps = steps // nblocks
    for block in range(nblocks):
        start, end = prompt_len + block * block_length, prompt_len + (block + 1) * block_length
        quotas = get_num_transfer_tokens(x[:, start:end] == mask_id, block_steps)
        for step in range(block_steps):
            masked = x == mask_id
            if not masked[:, start:end].any():
                break
            if cached is None:
                logits, hidden = forward(x, attention)
            else:
                logits, hidden = cached
                cached = None
            metrics["steps_taken"] += 1
            predicted = logits.argmax(-1)
            conf = get_mask_by_confidence(logits, predicted)
            entropy = token_entropy(logits, eps=1e-9 if adaptive_eb else 1e-10) if adaptive_eb or variant in ("entropy", "hybrid_probe", "response_gain", "response_ready") else None
            budgets = quotas[:, step]
            if adaptive_eb:
                budgets = torch.tensor([len(_eb_prefix(pos, conf[row, pos], entropy[row, pos]))
                    for row in range(batch)
                    for pos in [torch.arange(start, end, device=x.device)[masked[row, start:end]]]], device=x.device)
            response_choice = None
            if variant == "response_gain" and step % probe_every == 0:
                response_choice = _response_selection(model, forward, x, attention, logits, hidden, conf, entropy,
                    masked, start, end, budgets, metrics, fisher_vocab, diversity_beta)
            if variant == "response_ready" and step % probe_every == 0:
                response_choice = [_eb_prefix(pos, conf[row, pos], entropy[row, pos]) if adaptive_eb else pos[conf[row, pos].topk(min(int(budgets[row]), len(pos))).indices]
                    for row in range(batch)
                    for pos in [torch.arange(start, end, device=x.device)[masked[row, start:end]]]]
                active = [row for row in range(batch) if 0 < int(budgets[row]) < 8 and int(masked[row, start:end].sum()) > int(budgets[row])]
                if active:
                    active_tensor = torch.tensor(active, device=x.device)
                    ready_choices = _response_selection(model, forward, x[active_tensor], attention[active_tensor],
                        logits[active_tensor], hidden[active_tensor], conf[active_tensor], entropy[active_tensor],
                        masked[active_tensor], start, end, budgets[active_tensor], metrics, fisher_vocab,
                        diversity_beta, ready=True, incoming_beta=response_incoming_beta)
                    for row, choice in zip(active, ready_choices):
                        response_choice[row] = choice
            gradient = None
            if variant == "gradient_gain" and step % probe_every == 0 and any(gain_values):
                shortlist = []
                target_masks = masked.clone()
                target_masks[:, :start] = False; target_masks[:, end:] = False
                for row in range(batch):
                    pos = torch.arange(start, end, device=x.device)[masked[row, start:end]]
                    k = min(int(budgets[row]), pos.numel())
                    size = min(pos.numel(), max(k, 4 * k, pool_values[row]))
                    shortlist.append(pos[conf[row, pos].topk(size).indices])
                    if not gain_values[row]:
                        target_masks[row] = False
                    elif size < pos.numel():
                        metrics["gradient_sources_by_row"][row] += size
                        metrics["gradient_measurements_by_row"][row] += 1
                gradient = torch.zeros_like(x, dtype=torch.float32)
                probe_batch = gradient_probe_batch_size or batch
                for lo in range(0, batch, probe_batch):
                    hi = min(lo + probe_batch, batch)
                    gradient[lo:hi] = _gradient_reveal_gain(model, x[lo:hi], attention[lo:hi], target_masks[lo:hi],
                        shortlist[lo:hi], predicted[lo:hi], mask_id, metrics, metric_row_offset=lo)
            baseline, chosen = [], []
            for row in range(batch):
                pos = torch.arange(start, end, device=x.device)[masked[row, start:end]]
                k = min(int(budgets[row]), pos.numel())
                if k == 0:
                    baseline.append(pos[:0]); chosen.append(pos[:0]); continue
                local_conf = conf[row, pos].clamp_min(1e-30).log()
                row_stability = stability_beta[row] if isinstance(stability_beta, (list, tuple)) else stability_beta
                baseline.append(_eb_prefix(pos, conf[row, pos], entropy[row, pos]) if adaptive_eb else pos[conf[row, pos].topk(k).indices])
                if variant == "confidence":
                    chosen.append(baseline[-1])
                    continue
                if variant == "entropy":
                    chosen.append(pos[(-entropy[row, pos]).topk(k).indices])
                    continue
                if variant == "gradient_gain":
                    ranking = local_conf.clone()
                    normalized_risk = torch.zeros_like(ranking)
                    if temporal_values[row] and previous is not None:
                        risk = _target_temporal_risk(logits[row,pos],hidden[row,pos],previous[row,pos],head.weight,fisher_vocab)
                        normalized_risk = (risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
                        ranking -= row_stability * normalized_risk
                    if gradient is None or not gain_values[row]:
                        chosen.append(_eb_prefix(pos,ranking,entropy[row,pos]) if adaptive_eb else pos[ranking.topk(k).indices])
                    else:
                        sources = shortlist[row]
                        gain = gradient[row, sources]
                        lookup = torch.searchsorted(pos, sources)
                        relative_gain = (gain/gain.abs().mean().clamp_min(1e-8)).clamp(-4,4)
                        # Separate a benefit proxy from estimated wrong-commit
                        # damage. Neither is a calibrated functional guarantee.
                        bonus = _gradient_score_bonus(relative_gain,conf[row,sources],normalized_risk[lookup],
                            risk_values[row],gate_values[row])
                        score = ranking[lookup] + gain_values[row]*bonus
                        if adaptive_eb:
                            ranking[lookup] = score
                            chosen.append(_eb_prefix(pos, ranking, entropy[row, pos]))
                        else:
                            chosen.append(sources[score.topk(k).indices])
                    continue
                if variant in ("response_gain", "response_ready"):
                    proposed = response_choice[row] if response_choice is not None else baseline[-1]
                    if adaptive_eb and response_choice is not None:
                        ranking = local_conf.clone()
                        local_indices = torch.searchsorted(pos, proposed)
                        ranking[local_indices] = 1e6 - torch.arange(len(proposed), device=x.device, dtype=ranking.dtype)
                        proposed = _eb_prefix(pos, ranking, entropy[row, pos])
                    chosen.append(proposed)
                    continue
                states = hidden[row, pos].float()
                features = _unit(states - states.mean(0)) if variant == "raw_diversity" else None
                movement = None
                if variant in ("fisher_diversity", "temporal", "hybrid_probe"):
                    probability, weight, retained = _fisher_basis(logits[row, pos], head.weight, fisher_vocab)
                    metrics["fisher_retained_mass_sum"] += retained
                    metrics["fisher_measurements"] += 1
                    if variant != "temporal":
                        features = _fisher_project(states - states.mean(0), probability, weight)
                    if previous is not None and variant in ("temporal", "hybrid_probe"):
                        delta = states - previous[row, pos].float()
                        movement = .5 * _fisher_project(delta, probability, weight).square().sum(-1)
                if variant == "target_temporal" and previous is not None:
                    movement = _target_temporal_risk(logits[row, pos], states, previous[row, pos], head.weight, fisher_vocab)
                if variant in ("latent_support", "temporal_support") and previous is not None:
                    risk, support = _target_temporal_risk(logits[row, pos], states, previous[row, pos], head.weight, fisher_vocab, return_support=True)
                    local_conf += gain_beta * (support / support.abs().mean().clamp_min(1e-8)).clamp(-4, 4)
                    if variant == "temporal_support":
                        movement = risk
                if adaptive_eb:
                    ranking = local_conf.clone()
                    if movement is not None:
                        ranking -= row_stability * (movement / movement.mean().clamp_min(1e-8)).clamp(max=4)
                    chosen.append(_eb_prefix(pos, ranking, entropy[row, pos]))
                else:
                    selected = _greedy(local_conf, features, k, diversity_beta, movement, row_stability)
                    chosen.append(pos[selected])
            reveal = torch.zeros_like(masked)
            for row, positions in enumerate(chosen):
                reveal[row, positions] = True
            if variant == "hybrid_probe" and step % probe_every == 0:
                disagree = [row for row in range(batch) if not torch.equal(torch.sort(baseline[row]).values, torch.sort(chosen[row]).values)]
                if disagree:
                    # Evaluate both whole-batch actions on identical surviving targets.
                    branch = x.repeat(2, 1)
                    for row in range(batch):
                        branch[row, baseline[row]] = predicted[row, baseline[row]]
                        branch[batch + row, chosen[row]] = predicted[row, chosen[row]]
                    future_logits, future_hidden = forward(branch, attention.repeat(2, 1))
                    future_entropy = token_entropy(future_logits)
                    winners = torch.ones(batch, device=x.device, dtype=torch.long)
                    for row in disagree:
                        common = masked[row].clone()
                        common[:start] = False; common[end:] = False
                        common[baseline[row]] = False; common[chosen[row]] = False
                        u0 = _probe_utility(entropy[row], future_entropy[row], common, conf[row, baseline[row]].clamp_min(1e-30).log())
                        u1 = _probe_utility(entropy[row], future_entropy[batch + row], common, conf[row, chosen[row]].clamp_min(1e-30).log())
                        if u0 >= u1:
                            winners[row] = 0
                            reveal[row] = False
                            reveal[row, baseline[row]] = True
                            metrics["probe_switches"] += 1
                    indices = torch.arange(batch, device=x.device) + batch * winners
                    cached = (future_logits[indices], future_hidden[indices])
                    metrics["probe_steps"] += 1
                    metrics["probed_samples"] += len(disagree)
                    del future_logits, future_hidden, future_entropy, branch
            x[reveal] = predicted[reveal]
            keep_previous = variant in ("temporal", "target_temporal", "latent_support", "temporal_support", "hybrid_probe") or (variant == "gradient_gain" and any(temporal_values))
            previous = hidden.detach() if hidden is not None and keep_previous else None
            del logits, hidden, predicted, conf, entropy
            if progress and (metrics["steps_taken"] % 16 == 0 or step == block_steps - 1):
                progress(dict(metrics))
            if use_eos_harness and check_eos_stop_condition(x, eos_id, mask_id, prompt_len):
                break
        if use_eos_harness and check_eos_stop_condition(x, eos_id, mask_id, prompt_len):
            break
    metrics["unfinished_masks"] = int((x[:, prompt_len:] == mask_id).sum())
    count = metrics["fisher_measurements"]
    metrics["mean_fisher_retained_mass"] = metrics["fisher_retained_mass_sum"] / count if count else None
    nsource, ntarget = metrics["response_sources_measured"], metrics["response_targets_measured"]
    metrics["mean_response_omitted_source_mass"] = metrics["response_omitted_source_mass_sum"] / nsource if nsource else None
    metrics["mean_response_target_retained_mass"] = metrics["response_target_retained_mass_sum"] / ntarget if ntarget else None
    last_metrics = metrics
    return x, metrics["steps_taken"], block + 1
