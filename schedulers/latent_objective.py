"""One-forward, variable-cardinality three-term latent Gaussian objective.

Gaussian information quantities describe the declared surrogate, not token MI
or a generation-KL bound. Scheduling never consumes decoder probabilities.
"""
import math
import torch

from schedulers.latent_count import decoder_geometry, known_reads, read_capture

last_metrics = {}


def normalized_energy(field, active):
    energy = field.square().mean(-1)
    mean = (energy * active).sum(-1) / active.sum(-1).clamp_min(1)
    return (energy / mean[:, None].clamp_min(1e-12)).clamp(max=8)


def proposal_risk(final, middle, early, previous, directions, active,
                  measure, curvature_weight=1., temporal_weight=.15):
    """Readout-gradient projections of activation changes, without logit levels."""
    support = ((final-middle)*directions).sum(-1)
    if measure == 'proposal_curvature':
        curvature = ((final-2*middle+early)*directions).sum(-1)
        risk = curvature_weight*normalized_energy(curvature[:, :, None],active)
    elif measure == 'proposal_support':
        scale = ((support.square()*active).sum(-1)/active.sum(-1).clamp_min(1)).sqrt()
        risk = curvature_weight*(-support/scale[:, None].clamp_min(1e-6)).clamp(-8,8).exp().clamp(max=8)
    else:
        raise ValueError('unknown proposal risk')
    if previous is not None:
        movement = ((final-previous.float())*directions).sum(-1)
        risk += temporal_weight*normalized_energy(movement[:, :, None],active)
    return risk


def innovation_features(late, early, active, rank=16, centered=False):
    """Shared RMS normalization, followed by an explicit low-rank truncation."""
    blocks = []
    for field in (late, early):
        if centered:
            mean = (field*active[:, :, None]).sum(1)/active.sum(-1)[:, None].clamp_min(1)
            field = field-mean[:, None]
        scale = ((field.square().mean(-1)*active).sum(-1)
                 / active.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-6)
        blocks.append(field / scale[:, None, None])
    z = torch.cat(blocks, -1) / math.sqrt(late.shape[-1]*2)
    z = (z*active[:, :, None]).double()
    if rank < z.shape[-1]:
        _, vectors = torch.linalg.eigh(z.transpose(-1, -2)@z)
        z = z@vectors[:, :, -rank:]
    return z


def gaussian_select(z, risk, active, *, noise=1., gain_weight=1.,
                    risk_weight=.15, dependence_weight=1., progress_reward=.15,
                    max_count=16, singleton=False):
    """Batched greedy exact marginal increments of the Gaussian objective."""
    if noise <= 0 or max_count < 1:
        raise ValueError('positive noise and maximum count required')
    if min(gain_weight, risk_weight, dependence_weight, progress_reward) < 0:
        raise ValueError('nonnegative objective coefficients required')
    batch, positions, rank = z.shape
    z = z.double()*active[:, :, None]
    risk = risk.double()
    identity = torch.eye(rank, device=z.device, dtype=z.dtype)[None].expand(batch, -1, -1)
    source_inverse = identity.clone()
    target_inverse = torch.linalg.inv(identity + z.transpose(-1, -2)@z/noise)
    diagonal = noise + z.square().sum(-1)
    selected = torch.zeros_like(active)
    available = active.clone()
    running = available.any(-1)
    forced = torch.zeros(batch, device=z.device, dtype=torch.bool)
    objective = torch.zeros(batch, device=z.device, dtype=z.dtype)
    contributions = torch.zeros((batch, 4), device=z.device, dtype=z.dtype)
    rows = torch.arange(batch, device=z.device)
    for iteration in range(1 if singleton else min(max_count, positions)):
        source_product = z@source_inverse
        target_product = z@target_inverse
        conditional_source = noise+(source_product*z).sum(-1)
        precision_target = (1/noise-(target_product*z).sum(-1)/noise**2).clamp_min(1e-12)
        conditional_target = precision_target.reciprocal()
        information = .5*(conditional_source/conditional_target).clamp_min(1e-12).log()
        dependence = .5*(diagonal/conditional_source).clamp_min(1.).log()
        increment = (progress_reward + gain_weight*information-risk_weight*risk
                     - dependence_weight*dependence)
        increment = increment.masked_fill(~available, -torch.inf)
        best, index = increment.max(-1)
        accept = running & ((best > 0) | (iteration == 0))
        forced |= running & (best <= 0) & (iteration == 0)
        # Rows forced to progress stop after that singleton; they do not bypass
        # stopping on further negative increments.
        running &= best > 0
        selected[rows[accept], index[accept]] = True
        available[rows[accept], index[accept]] = False
        objective += torch.where(accept, best, 0.)
        contributions[:, 0] += accept*progress_reward
        contributions[:, 1] += accept*gain_weight*information[rows, index]
        contributions[:, 2] += accept*(-risk_weight*risk[rows, index])
        contributions[:, 3] += accept*(-dependence_weight*dependence[rows, index])
        source_vector = source_product[rows, index]
        target_vector = target_product[rows, index]
        ds = conditional_source[rows, index]
        dt = (noise-(target_vector*z[rows, index]).sum(-1)).clamp_min(1e-12)
        source_inverse -= (source_vector[:, :, None]*source_vector[:, None, :]
                           / ds[:, None, None])*accept[:, None, None]
        target_inverse += (target_vector[:, :, None]*target_vector[:, None, :]
                           / dt[:, None, None])*accept[:, None, None]
        running &= available.any(-1)
        if not bool(running.any()):
            break
    return selected, dict(forced=forced, objective=objective, contributions=contributions)


def incoming_instability(q, k, bias, risk, active, prefix, chunk=32):
    """Actual last-block attention transporting unresolved source instability."""
    if q.shape[1] != k.shape[1]:
        k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
    keys = k.float().transpose(-1, -2)
    source = torch.zeros((len(q), k.shape[-2]), device=q.device)
    source[:, prefix:] = risk*active
    values = []
    for start in range(prefix, q.shape[-2], chunk):
        stop = min(start+chunk, q.shape[-2])
        scores = q[:, :, start:stop].float()@keys*q.shape[-1]**-.5
        if bias is not None:
            local = bias[..., start:stop, :] if bias.shape[-2] != 1 else bias
            scores = (scores.masked_fill(~local, -torch.inf) if local.dtype == torch.bool
                      else scores+local.float())
        attention = scores.softmax(-1).mean(1)
        # Exclude the target itself from incoming transport.
        diagonal = attention[:, torch.arange(stop-start, device=q.device),
                             torch.arange(start, stop, device=q.device)]
        value = (attention*source[:, None]).sum(-1)-diagonal*source[:, start:stop]
        values.append(value)
    return torch.cat(values, -1).clamp_min(0.)


@torch.no_grad()
def generate_objective(model, prompt, attention_mask, *, steps=128, gen_length=512,
                       mask_id=126336, projection=None, rank=16, noise=1.,
                       gain_weight=1., risk_weight=.15, dependence_weight=1.,
                       progress_reward=.15, incoming_weight=.25,
                       temporal_weight=.15, curvature_weight=1., max_count=16,
                       singleton=False, initial_canvas=None, full_attention=None,
                       progress=None, resume_state=None, step_checkpoint=None,
                       risk_measure='projected_norm', readout_mean=None,
                       covariance_mode='raw'):
    global last_metrics
    if min(steps, gen_length, rank, max_count) < 1:
        raise ValueError('positive sizes required')
    if min(incoming_weight, temporal_weight, curvature_weight) < 0:
        raise ValueError('nonnegative risk weights required')
    if risk_measure not in ('projected_norm','proposal_curvature','proposal_support','context_curvature'):
        raise ValueError('unknown risk measure')
    if covariance_mode not in ('raw','centered','risk_scaled'):
        raise ValueError('unknown covariance mode')
    batch, prefix = prompt.shape
    x = (torch.full((batch, prefix+gen_length), mask_id, dtype=torch.long,
                    device=prompt.device) if initial_canvas is None else initial_canvas.clone())
    if x.shape != (batch, prefix+gen_length):
        raise ValueError('canvas shape mismatch')
    x[:, :prefix] = prompt
    attention = (torch.cat([attention_mask, torch.ones((batch, gen_length), device=x.device,
                         dtype=attention_mask.dtype)], -1) if full_attention is None else full_attention)
    if attention.shape != x.shape:
        raise ValueError('attention shape mismatch')
    projection = (decoder_geometry(model.get_output_embeddings().weight, width=16)
                  if projection is None else projection)
    config = dict(rank=rank, noise=noise, gain_weight=gain_weight, risk_weight=risk_weight,
                  dependence_weight=dependence_weight, progress_reward=progress_reward,
                  incoming_weight=incoming_weight, temporal_weight=temporal_weight,
                  curvature_weight=curvature_weight, max_count=max_count, singleton=singleton,
                  risk_measure=risk_measure,covariance_mode=covariance_mode)
    work = dict(steps_taken=0, forward_calls=0, sequence_evaluations=0,
                backward_sequence_evaluations=0, reveal_counts=[], objective_traces=[],
                forced_singletons_by_row=[0]*batch, confidence_scoring=False,
                entropy_scoring=False, eb_count=False, auxiliary_training=False,
                bootstrap_singletons_by_row=[0]*batch, config=config)
    previous = None
    previous_full = None
    if risk_measure != 'projected_norm' and readout_mean is None:
        readout_mean = model.get_output_embeddings().weight.float().mean(0)
    first_step = 0
    if resume_state is not None:
        if resume_state['metrics']['config'] != config:
            raise ValueError('resumed configuration differs')
        restored = resume_state['canvas'].to(x.device)
        if restored.shape != x.shape or not torch.equal(restored[:, :prefix], prompt):
            raise ValueError('resumed prompt differs')
        x = restored
        previous = resume_state['previous_projection'].to(x.device)
        if risk_measure != 'projected_norm':
            previous_full = resume_state['previous_full'].to(x.device)
        work = resume_state['metrics']
        first_step = resume_state['next_step']
        if first_step != work['steps_taken'] or not 0 <= first_step <= steps:
            raise ValueError('invalid resumed step')
    transformer = model.model.transformer
    for step in range(first_step, steps):
        active = (x[:, prefix:] == mask_id) & attention[:, prefix:].bool()
        if not bool(active.any()):
            break
        frames = {}
        hooks = [transformer.blocks[-5].register_forward_hook(
                     lambda _m, _i, o: frames.__setitem__('early', o[0].detach())),
                 transformer.blocks[-3].register_forward_hook(
                     lambda _m, _i, o: frames.__setitem__('middle', o[0].detach())),
                 model.get_output_embeddings().register_forward_pre_hook(
                     lambda _m, i: frames.__setitem__('final', i[0].detach()))]
        try:
            with read_capture(transformer.blocks[-1], incoming_weight > 0 or risk_measure=='context_curvature') as reads:
                output = model(x, attention_mask=attention)
        finally:
            for hook in hooks:
                hook.remove()
        output.logits[..., mask_id] = -torch.inf
        proposals = output.logits[:, prefix:].argmax(-1)
        del output
        final_full = frames['final'][:, prefix:]
        middle_full = transformer.ln_f(frames['middle'])[:, prefix:].float()
        early_full = transformer.ln_f(frames['early'])[:, prefix:].float()
        final = final_full.float()@projection
        middle = middle_full@projection
        early = early_full@projection
        late_delta, early_delta = final-middle, middle-early
        if risk_measure == 'projected_norm':
            risk = curvature_weight*normalized_energy(late_delta-early_delta, active)
            if previous is not None:
                risk += temporal_weight*normalized_energy(final-previous, active)
        else:
            directions = model.get_output_embeddings().weight[proposals].float()-readout_mean
            directions /= directions.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            risk = proposal_risk(final_full.float(),middle_full,early_full,previous_full,
                directions,active,'proposal_curvature' if risk_measure=='context_curvature' else risk_measure,
                curvature_weight,temporal_weight)
            previous_full = final_full.detach().clone()
            del directions
        if risk_measure == 'context_curvature':
            q, k, bias = reads[0]
            observed = (x != mask_id)&attention.bool()
            context = known_reads(q,k,observed.float(),bias)[:,prefix:]
            unresolved = (1-context).clamp(0.,1.)
            scale = (unresolved*active).sum(-1)/active.sum(-1).clamp_min(1)
            risk += (unresolved/scale[:,None].clamp_min(1e-6)).clamp(max=8)
            del context,unresolved,observed
        if incoming_weight > 0:
            q, k, bias = reads[0]
            risk += incoming_weight*incoming_instability(q, k, bias, risk, active, prefix)
        z = innovation_features(late_delta, early_delta, active, rank,
                                centered=covariance_mode=='centered')
        if covariance_mode == 'risk_scaled':
            z *= risk.clamp(0.,8.).sqrt()[:, :, None]
        bootstrap = (previous is None and curvature_weight == 0 and temporal_weight > 0
                     and risk_measure == 'proposal_curvature')
        chosen, details = gaussian_select(z, risk, active, noise=noise,
            gain_weight=gain_weight, risk_weight=risk_weight,
            dependence_weight=dependence_weight, progress_reward=progress_reward,
            max_count=1 if bootstrap else max_count, singleton=singleton)
        if bootstrap:
            work['bootstrap_singletons_by_row'] = [a+int(b) for a,b in zip(
                work['bootstrap_singletons_by_row'],active.any(-1).tolist())]
        tail = x[:, prefix:]
        tail[chosen] = proposals[chosen]
        previous = final.detach()
        work['steps_taken'] += 1
        work['forward_calls'] += 1
        work['sequence_evaluations'] += batch
        work['reveal_counts'].append(chosen.sum(-1).tolist())
        forced = details['forced'].tolist()
        work['forced_singletons_by_row'] = [a+int(b) for a,b in zip(work['forced_singletons_by_row'], forced)]
        work['objective_traces'].append(dict(objective=details['objective'].tolist(),
            contributions=details['contributions'].tolist(), forced=forced))
        del frames, reads, proposals, middle, early, late_delta, early_delta, risk, z, chosen
        del final_full, middle_full, early_full
        if progress and (step % 16 == 15 or step == steps-1):
            progress(work)
        if step_checkpoint and (step % 16 == 15 or step == steps-1):
            state = dict(canvas=x.detach().cpu(), previous_projection=previous.detach().cpu(),
                         next_step=step+1, metrics=work)
            if previous_full is not None:
                state['previous_full'] = previous_full.detach().cpu()
            step_checkpoint(state)
    work['unfinished_masks_by_row'] = ((x[:, prefix:] == mask_id)&attention[:, prefix:].bool()).sum(-1).tolist()
    last_metrics = work
    return x
