"""DAS: Dependence-Aware Sampler for LLaDA.

Ranking: log confidence minus normalised temporal Fisher movement (the mentor's
EB + temporal Fisher readiness, reused unchanged).

Count: EB admits a ranked prefix while  sum(H) - max(H) <= gamma.  That is a
worst-case bound on the batch's joint-dependence error TC(X_S).  DAS keeps a
fraction `alpha` of that bound (slack for model error and for dependence that a
pairwise measurement misses) and adds the *measured* pairwise dependence:

    cost(S) = alpha * (sum_S H - max_S H) + sum_{i<j in S} D_ij  <=  gamma

D_ij comes from lookahead: commit candidate i at its greedy value in a copy of
the canvas, re-run the model, and take the KL change of candidate j's predictive
distribution against an unchanged control evaluated in the same batch (matched
shape, so ROCm numerics cancel).  Because greedy decoding commits the argmax,
this is exactly "would j's prediction differ if i were committed first".

alpha=1 and no lookahead reproduces EB + Fisher exactly (count='eb' too).

Count modes
  'eb'        EB prefix on the readiness ranking            (= EB + Fisher)
  'singleton' one token per step
  'fixed'     k_fixed tokens per step; with dep='lookahead', skip candidates
              whose dependence to an already chosen one exceeds skip_delta
  'das'       the dependence budget above (gated lookahead, pass budget)
  'pace'      EB prefix, plus a progress floor: at least ceil(L_rem / steps_left)
              answer tokens per step, where L_rem = masked positions before the
              first committed-or-predicted EOS/EOT.  Relaxes EB only as much as
              the step budget forces (pace_frac < 1 aims to finish earlier).

Every model forward is counted per row in metrics['forward_work_by_row'].
"""
from __future__ import annotations

import math

import torch

from schedulers.utils import get_mask_by_confidence, token_entropy, mask_eos_eot_logits
from schedulers.latent import _target_temporal_risk
from schedulers.latent_allowance import attention_field, gradient_dispersion, allowance_prefix
from schedulers.latent_count import decoder_geometry, read_capture
from schedulers.latent_information import partition_probabilities, response_information

last_metrics = {}


def _info_adjust(model, x, attention, logits, conf, rankings, positions, rows, mask_id, metrics,
                 candidates=4, vocab=128, gain=.5, risk_eta=1., incoming=.15, chunk=160):
    """Mentor's information + risk + incoming-sensitivity score (latent_information.py).

    Branch the top `candidates` positions on their top-2 tokens, measure the response
    information at every other masked position against a matched control, and adjust
    the readiness ranking in place.  Same terms and weights as his 56/164 run.
    """
    pools, tokens, rhos, conds, bases, actions = {}, {}, {}, {}, {}, []
    for r in rows:
        pos = positions[r]
        pool = rankings[r].argsort(descending=True, stable=True)[:candidates]
        pools[r] = pool
        tokens[r] = logits[r, pos].topk(min(vocab, logits.shape[-1] - 1), -1).indices
        src, alts = logits[r, pos[pool]].float().topk(2, -1)
        rhos[r] = src.softmax(-1)
        shape = (len(pool), 2, len(pos), tokens[r].shape[-1] + 1)
        conds[r] = torch.empty(shape, device=x.device)
        bases[r] = torch.empty((shape[0], shape[2], shape[3]), device=x.device)
        for s, i in enumerate(pool.tolist()):
            for a in range(2):
                actions.append((r, s, a, int(pos[i]), int(alts[s, a])))
    for off in range(0, len(actions), chunk):
        group = actions[off:off + chunk]
        ridx = torch.tensor([g[0] for g in group], device=x.device)
        # Control: the unchanged canvas in the same batch shape as the branches.
        lg = model(x[ridx], attention_mask=attention[ridx]).logits
        lg[..., mask_id] = -torch.inf
        for j, (r, s, a, _, _) in enumerate(group):
            if a == 0:
                bases[r][s] = partition_probabilities(lg[j, positions[r]], tokens[r])
        del lg
        branch = x[ridx].clone()
        for j, (_, _, _, i, tok) in enumerate(group):
            branch[j, i] = tok
        lg = model(branch, attention_mask=attention[ridx]).logits
        lg[..., mask_id] = -torch.inf
        for j, (r, s, a, _, _) in enumerate(group):
            conds[r][s, a] = partition_probabilities(lg[j, positions[r]], tokens[r])
        del lg, branch
        for r, *_ in group:
            metrics['forward_work_by_row'][r] += 2
    for r, pool in pools.items():
        info, defect = response_information(conds[r], rhos[r], bases[r])
        ar = torch.arange(len(pool), device=x.device)
        info[ar, pool] = 0.; defect[ar, pool] = 0.
        outgoing = info.sum(-1); damage = (info + defect).sum(-1)
        bonus = (outgoing / outgoing.mean().clamp_min(1e-8)).clamp(max=4)
        harmful = (damage / damage.mean().clamp_min(1e-8)).clamp(max=4)
        bonus = bonus - risk_eta * (1 - conf[r, positions[r][pool]]) * harmful
        rankings[r][pool] += gain * bonus
        inc = (info + defect).sum(0)
        rankings[r] -= incoming * (inc / inc.mean().clamp_min(1e-8)).clamp(max=4)
        metrics['info_probes'] = metrics.get('info_probes', 0) + 1


def _eb_cost(h):
    """Running EB bound for an ordered list of entropies (1-D tensor)."""
    return h.cumsum(0) - h.cummax(0).values


def _lookahead(model, forward, x, attention, predicted, rows, cands, chunk, metrics):
    """Return {row: D (len(c) x len(c)) tensor} of KL(p_j | i committed || p_j control).

    D[a, b] = effect of committing cands[a] on the distribution at cands[b].
    """
    seqs, atts, owner = [], [], []
    for r, c in zip(rows, cands):
        seqs.append(x[r]); atts.append(attention[r]); owner.append((r, -1))
        for a, i in enumerate(c.tolist()):
            y = x[r].clone(); y[i] = predicted[r, i]
            seqs.append(y); atts.append(attention[r]); owner.append((r, a))
    seqs = torch.stack(seqs); atts = torch.stack(atts)
    # Chunk boundaries never split a row's control from its branches, so each
    # control shares a forward call (and batch shape) with its branches.
    bounds, lo, sizes = [], 0, [1 + len(c) for c in cands]
    for s in sizes:
        if bounds and bounds[-1][1] - bounds[-1][0] + s <= chunk:
            bounds[-1][1] += s
        else:
            bounds.append([lo, lo + s])
        lo += s
    logp = {}
    for lo, hi in bounds:
        lg = forward(seqs[lo:hi], atts[lo:hi], count=False)
        for k in range(lg.shape[0]):
            r, a = owner[lo + k]
            c = cands[rows.index(r)]
            logp[(r, a)] = torch.log_softmax(lg[k, c].float(), -1)
        del lg
    out = {}
    for r, c in zip(rows, cands):
        ctrl = logp[(r, -1)]
        n = len(c)
        D = torch.zeros(n, n, device=x.device)
        for a in range(n):
            br = logp[(r, a)]
            p = br.exp()
            # KL(branch || control) per target; tokens with zero mass (e.g. MASK) contribute 0.
            kl = torch.where(p > 0, p * (br - ctrl), torch.zeros_like(p)).sum(-1).clamp_min(0)
            D[a] = kl
        D.fill_diagonal_(0)
        out[r] = D
        metrics['forward_work_by_row'][r] += 1 + n
        metrics['lookahead_sequences'] += 1 + n
        metrics['lookahead_steps_by_row'][r] += 1
    return out


def _extrapolated_logconf(logits, states, previous, weight, vocab, eta):
    out = []
    for start in range(0, len(states), 32):
        stop = min(start + 32, len(states))
        indices = logits[start:stop].float().topk(min(vocab, logits.shape[-1]), -1).indices
        target = states[start:stop].float() + eta * (states[start:stop].float() - previous[start:stop].float())
        z = torch.einsum('nd,nvd->nv', target, weight[indices].float())
        out.append(z.log_softmax(-1)[:, 0])  # column 0 = the current argmax (the proposal)
    return torch.cat(out)


def _pool_steps(lengths, steps, pool, lo, hi):
    """Split len(lengths) * steps passes in proportion to length ** pool (1 = per token,
    0.5 = sqrt), each row within [lo, hi] * steps and at least 2, summing exactly."""
    n, total = len(lengths), len(lengths) * steps
    lo, hi = max(2, int(math.ceil(lo * steps))), int(hi * steps)
    w = [max(float(l), 1.) ** float(pool) for l in lengths]
    alloc = [min(hi, max(lo, int(round(float(total * x / sum(w)))))) for x in w]
    order = sorted(range(n), key=lambda i: -w[i])
    while sum(alloc) != total:
        d = 1 if sum(alloc) < total else -1
        moved = False
        for i in (order if d > 0 else order[::-1]):
            if sum(alloc) == total: break
            if lo <= alloc[i] + d <= hi:
                alloc[i] += d; moved = True
        if not moved: break
    return alloc


def _lens_logconf(logits, states, ln_f, weight, vocab):
    out = []
    for start in range(0, len(states), 32):
        stop = min(start + 32, len(states))
        indices = logits[start:stop].float().topk(min(vocab, logits.shape[-1]), -1).indices
        z = torch.einsum('nd,nvd->nv', ln_f(states[start:stop]).float(), weight[indices].float())
        out.append(z.log_softmax(-1)[:, 0])
    return torch.cat(out)


@torch.no_grad()
def generate_das(model, prompt, attention_mask=None, *, steps=128, gen_length=512,
                 mask_id=126336, eos_id=126081, eot_id=126348,
                 count='das', k_fixed=4, dep='lookahead', skip_delta=0.1,
                 alpha=0.5, gamma=0.1, gamma_eb=0.1, k_max=16, m=8,
                 sym='max', prefix=False, pass_budget=1.0, fallback='eb', pace_frac=1.0, pad_ids=None, pad_run=8, pace_shape='flat', pace_space=0, pace_order='rank', pace_cap=None, remask_tau=None, rank='conf', block_len=None,
                 score='fisher', info_every=16, info_candidates=4, info_gain=.5, info_risk=1., info_incoming=.15, info_chunk=160,
                 base='eb', allow_tol=.5, allow_max=16, remask_scope='extra', chunk=40,
                 stability_beta=.15, fisher_vocab=128, use_fisher=True, b44=False, ext=None, soft=0., soft_vocab=128, div=None, lens=None, lens_w=.5,
                 share=None, share_min=.25, share_max=4., share_at=0., trace_end=False, end_tau=None, **_ignored):
    global last_metrics
    head = model.get_output_embeddings()
    batch, plen = prompt.shape
    x = torch.full((batch, plen + gen_length), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :plen] = prompt
    attention = (torch.cat([attention_mask, torch.ones(batch, gen_length, device=prompt.device,
                 dtype=attention_mask.dtype)], -1) if attention_mask is not None else torch.ones_like(x))
    metrics = dict(count=count, dep=dep, alpha=alpha, gamma=gamma, k_max=k_max, m=m,
                   steps_taken=0, forward_work_by_row=[0] * batch, lookahead_sequences=0,
                   lookahead_steps_by_row=[0] * batch, commits_by_row=[0] * batch,
                   gate_fired=0, gate_skipped_budget=0, extended_beyond_eb=0, shrunk_below_eb=0,
                   coupled_skips=0, dep_values=[])
    budget_total = pass_budget * steps  # extra passes allowed per row

    captured = []
    hook_handle = head.register_forward_pre_hook(lambda _m, inp: captured.append(inp[0].detach()))

    mid = []  # logit lens: residual stream after block `lens`
    lens_handle = None
    if lens is not None:
        lens_handle = model.model.transformer.blocks[lens].register_forward_hook(
            lambda _m, _i, out: mid.append((out[0] if isinstance(out, tuple) else out).detach()))
    ln_f = model.model.transformer.ln_f

    reads = []
    if base == 'allowance':  # mentor's Fisher-geometry allowance count (latent_allowance.py, metric fisher_dispersion)
        projection = decoder_geometry(head.weight)
        dictionary = head.weight.float() @ projection
        dictionary -= dictionary.mean(0)
    elif base != 'eb':
        raise ValueError(base)

    wte = model.model.transformer.wte
    soft_draft = None  # (top ids, renormalised probs) from the previous pass, for soft-token inputs

    def forward(ids, attn, count=True):
        captured.clear(); reads.clear(); mid.clear()
        embeds = None
        if soft > 0 and soft_draft is not None and count:
            # Latent feedback: a still-masked input carries a little of its current draft,
            # (1 - soft) * e[mask] + soft * sum_v p_v e_v  over the draft's top tokens.
            embeds = wte(ids)
            top_i, top_p = soft_draft
            m_in = ids == mask_id
            m_in[:, :plen] = False
            ti, tp = top_i[m_in], top_p[m_in]
            draft = torch.cat([torch.einsum('nk,nkd->nd', tp[c:c + 256], wte.weight[ti[c:c + 256]].float())
                               for c in range(0, len(ti), 256)])
            embeds[m_in] = ((1 - soft) * embeds[m_in].float() + soft * draft).to(embeds.dtype)
        with read_capture(model.model.transformer.blocks[-1], base == 'allowance' and count) as rd:
            lg = (model(input_ids=None, inputs_embeds=embeds, attention_mask=attn).logits if embeds is not None
                  else model(ids, attention_mask=attn).logits)
        reads.extend(rd)
        mask_eos_eot_logits(lg, False, False, eos_id, eot_id)
        lg[..., mask_id] = -torch.inf
        if count:
            for r in range(batch):
                metrics['forward_work_by_row'][r] += active[r]
        return lg

    pad_table = None
    if pad_ids is not None:
        pad_table = torch.zeros(model.get_output_embeddings().weight.shape[0], dtype=torch.bool, device=prompt.device)
        pad_table[torch.as_tensor(pad_ids, device=prompt.device)] = True
    previous = None
    end_ids = torch.tensor(sorted(set(([] if pad_ids is None else list(pad_ids)) + [eos_id, eot_id])),
                           device=prompt.device)
    # share: the batch shares batch * steps passes; after the first pass each row gets a share
    # proportional to its estimated answer length (same mean NFE, even tokens per step).
    row_steps = [steps] * batch
    active = [True] * batch
    first_len = [0] * batch
    pool_step = int(share_at * steps)  # reallocate after this pass, once the answer end is readable
    # remask: positions committed last step that may be re-checked (each position at most once)
    recheck = torch.zeros_like(x, dtype=torch.bool)
    remasked_once = torch.zeros_like(x, dtype=torch.bool)
    try:
        for step in range(steps if share is None else int(math.ceil(share_max * steps))):
            if step >= max(row_steps):
                break
            for r in range(batch):
                active[r] = step < row_steps[r]
            masked = x == mask_id
            masked[:, :plen] = False
            if not masked.any() and not recheck.any():
                break
            logits = forward(x, attention)
            mask_eos_eot_logits(logits, b44, b44, eos_id, eot_id)  # LLaDA App. B.4, instruct models
            hidden = captured[0]
            metrics['steps_taken'] += 1
            if remask_tau is not None and recheck.any() and step < steps - 1:
                # The pass already conditions each last-step commit on everything committed with it,
                # so p(committed token) here is a free post-hoc dependence check (D_{j|i} after the fact).
                p_in = logits.float().softmax(-1).gather(-1, x[..., None])[..., 0]
                bad = recheck & ~remasked_once & (p_in < remask_tau)
                if bad.any():
                    x[bad] = mask_id  # re-predicted next step from a masked input, not committed now
                    remasked_once |= bad
                    metrics['remasked'] = metrics.get('remasked', 0) + int(bad.sum())
            recheck.zero_()
            if not masked.any():
                continue
            predicted = logits.argmax(-1)
            conf = get_mask_by_confidence(logits, predicted)
            entropy = token_entropy(logits, eps=1e-9)

            if base == 'allowance':
                q, k, bias = reads[0]
                graph = attention_field(q, k, bias, plen)
                allow_err = gradient_dispersion(logits[:, plen:], dictionary)
                del q, k, bias
                reads.clear()
            positions, rankings = [], []
            for r in range(batch):
                pos = masked[r].nonzero().flatten()
                positions.append(pos)
                if not pos.numel():
                    rankings.append(None); continue
                if rank == 'margin':  # top-1 minus top-2 probability: low margin = high model error
                    top2 = logits[r, pos].float().softmax(-1).topk(2, dim=-1).values
                    ranking = (top2[:, 0] - top2[:, 1]).clamp_min(1e-30).log()
                else:
                    ranking = conf[r, pos].clamp_min(1e-30).log()
                if ext is not None and previous is not None:
                    # Where is the latent heading? Read the proposal out of h + ext * (h - h_prev)
                    # (the head is linear, no bias): readiness = its log-prob there, over the top tokens.
                    ranking = _extrapolated_logconf(logits[r, pos], hidden[r, pos], previous[r, pos],
                                                    head.weight, fisher_vocab, ext)
                if use_fisher and previous is not None:
                    risk = _target_temporal_risk(logits[r, pos], hidden[r, pos], previous[r, pos],
                                                 head.weight, fisher_vocab)
                    ranking = ranking - stability_beta * (risk / risk.mean().clamp_min(1e-8)).clamp(max=4)
                if lens is not None:
                    # Depth agreement: how early in the network was this answer already there?
                    # log-prob of the final top-1 under the layer-`lens` read-out (top fisher_vocab tokens).
                    ranking = ranking + lens_w * _lens_logconf(logits[r, pos], mid[0][r, pos], ln_f,
                                                               head.weight, fisher_vocab)
                rankings.append(ranking)
            if score == 'info' and step % info_every == 0:
                rows = [r for r in range(batch) if positions[r].numel()]
                _info_adjust(model, x, attention, logits, conf, rankings, positions, rows, mask_id, metrics,
                             info_candidates, fisher_vocab, info_gain, info_risk, info_incoming, info_chunk)
                captured.clear()
            elif score not in ('fisher', 'info'):
                raise ValueError(score)
            orders, ents = [], []
            for r in range(batch):
                pos, ranking = positions[r], rankings[r]
                if ranking is None:
                    orders.append(pos); ents.append(None); continue
                o = ranking.argsort(descending=True, stable=True)
                orders.append(pos[o]); ents.append(entropy[r, pos[o]].double())

            chosen = [None] * batch
            need_rows, need_cands, tails = [], [], {}
            for r in range(batch):
                order, h = orders[r], ents[r]
                if not order.numel():
                    chosen[r] = order; continue
                eb_len = int((_eb_cost(h) <= gamma_eb).sum())
                if base == 'allowance':  # replaces the EB prefix as the base count (singleton on the first step, as his)
                    if step == 0:
                        eb_len = 1
                    else:
                        g = order - plen
                        sel, _ = allowance_prefix(g, allow_err[r], graph[r], allow_tol, allow_max, uncertainty_only=True)
                        eb_len = len(sel)
                if not active[r]:
                    chosen[r] = order[:0]
                elif count == 'eb':
                    chosen[r] = order[:max(eb_len, 1)]
                elif count == 'singleton':
                    chosen[r] = order[:1]
                elif count == 'fixed':
                    if dep == 'lookahead' and order.numel() > 1:
                        need_rows.append(r); need_cands.append(order[:min(m, order.numel())])
                    else:
                        chosen[r] = order[:k_fixed]
                elif count == 'pace':
                    # Answer end: first EOS/EOT, or start of a run of pad_run whitespace-only
                    # tokens (LLaDA pads with blank lines), committed or predicted.
                    tail = torch.where(masked[r, plen:], predicted[r, plen:], x[r, plen:])
                    padlike = (tail == eos_id) | (tail == eot_id)
                    if pad_table is not None:
                        padlike |= pad_table[tail]
                    if end_tau is not None:
                        # A masked position counts as past the end only when the model is confident:
                        # P(end-of-text or whitespace) > end_tau, not merely its top-1 guess.
                        p_end = logits[r, plen:].float().softmax(-1)[:, end_ids].sum(-1)
                        m_r = masked[r, plen:]
                        padlike = torch.where(m_r, p_end > end_tau, padlike)
                        tail = torch.where(m_r & ~(p_end > end_tau) & ((tail == eos_id) | (tail == eot_id)),
                                           torch.full_like(tail, mask_id), tail)
                    run = torch.nn.functional.conv1d(padlike.float()[None, None],
                                                     torch.ones(1, 1, pad_run, device=x.device))[0, 0]
                    hits = ((run >= pad_run).nonzero().flatten())
                    eos_hits = ((tail == eos_id) | (tail == eot_id)).nonzero().flatten()
                    cands_end = [gen_length] + ([int(hits[0])] if hits.numel() else []) + \
                                ([int(eos_hits[0])] if eos_hits.numel() else [])
                    end = plen + min(cands_end)
                    if trace_end:  # diagnostic: estimated answer end and content left, per row per step
                        metrics.setdefault('end_trace', [[] for _ in range(batch)])[r].append(
                            [step, end - plen, int(masked[r, plen:end].sum())])
                    if block_len:
                        # Semi-autoregressive: the earliest block with masked answer tokens goes first
                        # (EB prefix and pace extras), later blocks only take the spill-over.
                        blk = (order - plen) // block_len
                        live = blk[order < end]
                        if live.numel():
                            later = (blk > live.min()).long()
                            o = later.argsort(stable=True)
                            order, h = order[o], h[o]
                            eb_len = min(int((_eb_cost(h) <= gamma_eb).sum()), int((later == 0).sum()))
                    content = order < end
                    l_rem = int(content.sum())
                    if step == pool_step:
                        first_len[r] = l_rem
                    steps_r = row_steps[r]
                    horizon = max(step + 1, int(pace_frac * steps_r))
                    w = torch.arange(step, horizon, dtype=torch.float64)
                    w = {'flat': torch.ones_like(w), 'up': w + 1, 'down': steps_r - w}[pace_shape]
                    need = int(torch.ceil(l_rem * w[0] / w.sum()))
                    take = torch.zeros_like(content)
                    take[:max(eb_len, 1)] = True
                    short = need - int((take & content).sum())
                    if short > 0:
                        pool = (content & ~take).nonzero().flatten()  # indices into order, rank order
                        if end_tau is not None:  # extras never commit filler (end-of-text / whitespace-only)
                            pred_o = predicted[r, order[pool]]
                            filler = (pred_o == eos_id) | (pred_o == eot_id)
                            if pad_table is not None:
                                filler |= pad_table[pred_o]
                            pool = pool[~filler]
                        if pace_order == 'left':
                            pool = pool[order[pool].argsort()]
                        if pace_space > 0:
                            picked = order[take].tolist()
                            added, rest = [], []
                            for k in pool.tolist():
                                pos = int(order[k])
                                if len(added) < short and all(abs(pos - q) >= pace_space for q in picked):
                                    added.append(k); picked.append(pos)
                                else:
                                    rest.append(k)
                            added += rest[:short - len(added)]
                            extra = torch.tensor(added, dtype=torch.long, device=order.device)
                        elif div is not None:
                            # Latent diversity: skip an extra whose hidden state nearly duplicates one
                            # already taken this step (likely the same token twice); refill from the skipped.
                            unit = torch.nn.functional.normalize(hidden[r, order].float(), dim=-1)
                            picked = unit[take]
                            added, rest = [], []
                            for k in pool.tolist():
                                if len(added) >= short:
                                    break
                                if picked.numel() and float((picked @ unit[k]).max()) > div:
                                    rest.append(k); continue
                                added.append(k); picked = torch.cat([picked, unit[k:k + 1]])
                            metrics['div_skipped'] = metrics.get('div_skipped', 0) + len(rest)
                            added += rest[:short - len(added)]
                            extra = torch.tensor(added, dtype=torch.long, device=order.device)
                        else:
                            extra = pool[:short]
                        take[extra] = True
                        metrics['pace_extra'] = metrics.get('pace_extra', 0) + int(extra.numel())
                        if remask_tau is not None and remask_scope == 'extra':
                            recheck[r, order[extra]] = True
                    elif pace_cap is not None:
                        # Two-sided: ahead of schedule, spend the slack on fewer joint commits.
                        cap = max(1, int(math.ceil(pace_cap * need)))
                        over = (take & content).cumsum(0) > cap
                        cut = take & content & over
                        take &= ~cut
                        metrics['pace_cut'] = metrics.get('pace_cut', 0) + int(cut.sum())
                    chosen[r] = order[take]
                    if remask_tau is not None and remask_scope == 'all' and int(take.sum()) > 1:
                        recheck[r, order[take & content]] = True
                elif count == 'das':
                    cand = order[:k_max]
                    opt_len = int((alpha * _eb_cost(h[:k_max]) <= gamma).sum())  # D = 0 optimistic
                    opt_len = max(opt_len, 1)
                    if dep != 'lookahead':
                        chosen[r] = cand[:opt_len]
                    elif opt_len <= max(eb_len, 1):
                        chosen[r] = order[:max(eb_len, 1)]  # lookahead could only shrink: keep EB
                    else:
                        spent = metrics['forward_work_by_row'][r] - metrics['steps_taken']
                        n = min(opt_len, m)
                        pace = budget_total * (step + 1) / steps + (m + 1)
                        if spent + n + 1 <= budget_total and spent <= pace:
                            need_rows.append(r); need_cands.append(cand[:n])
                            if fallback == 'opt': tails[r] = cand[n:opt_len]  # unmeasured: trust the alpha bound
                        else:
                            metrics['gate_skipped_budget'] += 1
                            chosen[r] = cand[:opt_len] if fallback == 'opt' else order[:max(eb_len, 1)]
                else:
                    raise ValueError(count)

            if need_rows:
                metrics['gate_fired'] += len(need_rows)
                Ds = _lookahead(model, forward, x, attention, predicted, need_rows, need_cands, chunk, metrics)
                for r, c in zip(need_rows, need_cands):
                    D = Ds[r]
                    D = torch.maximum(D, D.T) if sym == 'max' else (D + D.T) / 2
                    if D.numel() > 1:
                        metrics['dep_values'].append(float(D[~torch.eye(len(c), dtype=torch.bool, device=D.device)].max()))
                    h = entropy[r, c].double()
                    sel = [0]
                    if count == 'fixed':
                        for a in range(1, len(c)):
                            if len(sel) >= k_fixed: break
                            if float(D[a, sel].max()) <= skip_delta: sel.append(a)
                            else: metrics['coupled_skips'] += 1
                        for a in range(1, len(c)):  # fill to k with the best remaining
                            if len(sel) >= k_fixed: break
                            if a not in sel: sel.append(a)
                    else:
                        for a in range(1, len(c)):
                            trial = sel + [a]
                            hh = h[trial]
                            cost = alpha * float(hh.sum() - hh.max()) + float(D[trial][:, trial].triu(1).sum())
                            if cost <= gamma:
                                sel.append(a)
                            else:
                                metrics['coupled_skips'] += 1
                                if prefix: break
                        eb_len = int((_eb_cost(ents[r]) <= gamma_eb).sum())
                        if len(sel) > eb_len: metrics['extended_beyond_eb'] += 1
                        if len(sel) < eb_len: metrics['shrunk_below_eb'] += 1
                    chosen[r] = c[torch.tensor(sorted(sel), device=c.device)]
                    if r in tails and len(sel) == len(c):  # nothing measured was coupled: keep the rest
                        chosen[r] = torch.cat([chosen[r], tails[r]])

            if share is not None and step == pool_step:
                left = _pool_steps(first_len, steps - step - 1, share, share_min, share_max)
                row_steps = [step + 1 + a for a in left]
                metrics['row_steps'] = row_steps
                metrics['pool_len'] = list(first_len)
            for r in range(batch):
                if chosen[r] is not None and chosen[r].numel():
                    x[r, chosen[r]] = predicted[r, chosen[r]]
                    metrics['commits_by_row'][r] += int(chosen[r].numel())
            previous = hidden if (use_fisher or ext is not None) else None
            if soft > 0:
                tv, ti = logits.float().topk(soft_vocab, -1)
                soft_draft = (ti, tv.softmax(-1))
            del logits
    finally:
        hook_handle.remove()
        if lens_handle is not None:
            lens_handle.remove()
    d = metrics.pop('dep_values')
    metrics['dep_max_mean'] = sum(d) / len(d) if d else None
    metrics['dep_max_median'] = sorted(d)[len(d) // 2] if d else None
    metrics['unfinished_masks'] = int((x[:, plen:] == mask_id).sum())
    if trace_end:  # where the finished answer actually ends (same rule, on the final canvas)
        fin = []
        for r in range(batch):
            tail = x[r, plen:]
            padlike = (tail == eos_id) | (tail == eot_id) | (tail == mask_id)
            if pad_table is not None:
                padlike |= pad_table[tail]
            run = torch.nn.functional.conv1d(padlike.float()[None, None],
                                             torch.ones(1, 1, pad_run, device=x.device))[0, 0]
            hits = (run >= pad_run).nonzero().flatten()
            eos_hits = ((tail == eos_id) | (tail == eot_id)).nonzero().flatten()
            fin.append(min([gen_length] + ([int(hits[0])] if hits.numel() else []) +
                           ([int(eos_hits[0])] if eos_hits.numel() else [])))
        metrics['final_end'] = fin
    last_metrics = metrics
    return x, metrics
