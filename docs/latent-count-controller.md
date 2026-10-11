# Isolating count with a forced-mask latent-growth controller

The user's latest direction is to focus on count while preserving a useful
scorer. Because the earlier successful Fisher scorer includes token confidence,
it is used here only as a labelled count-isolation diagnostic. Separate candidate
arms use an activation-only scorer. The new count rule uses no EB entropy budget,
token confidence, original logit margin, learned policy, or fitted risk head.

## What the controller measures

For the current ranked masked positions, consider prefixes of size
1, 2, 4, 8 and 16, clipped to the remaining canvas. The size-one action is a
progress fallback. For every larger prefix S_k, keep its last target j masked,
tentatively reveal the other k−1 proposed greedy values, and run the frozen model.
Compare the hidden state at j against an unchanged input evaluated in the same
probe batch shape. The target's own removal never counts as gain or error.

This leave-one-out probe can expose higher-order dependence: with unresolved
parity, singleton source changes can leave every marginal unchanged, while
revealing all other unresolved bits determines the kept target.

The model's final normalization is also applied to an earlier state two blocks
before the end. Their difference supplies a native layer-change scale. It is an
activation scale, not a calibrated uncertainty estimate.

The first metric sketches the centred frozen decoder-weight covariance, using
fixed random signs. It uses no runtime probabilities but proves too permissive
in arithmetic. The improved metric uses the decoder gradient vectors of its
proposed values before and after the tentative reveal. Those value identities
define directions; their scores and probabilities are discarded. For the union
of up to four value proposals from each context, let d_r=W_r−mean_r W_r. Measure

    error(k) = mean_r [(d_rᵀ(h_j^probe−h_j^control))²]
               / mean_r [(d_rᵀ(h_j^final−h_j^earlier))²].

A small activation-derived floor regularizes the denominator. Since the decoder
is linear on the captured final representation, these are exact readout-gradient
projections of actual finite interventions; no mask-to-token Jacobian
linearization is assumed. No confidence or current-margin term appears.

The stepwise tolerance is η divided by the remaining outer steps. This is a
heuristic relaxation, not a certified total KL or cumulative risk budget. Take a
conservative upper envelope of measured errors across prefix sizes and choose
the largest admissible size. Bootstrap with a singleton. An optional pacing
override is recorded explicitly; the primary tests use the free guard.

Only one kept target per prefix is probed. This is an economical screen, not a
bound on every target's dependence or a guarantee of correctness. The candidate
grid, intrinsic scale, and hand-selected η all need empirical validation.

## Fixed scorers for the count experiment

The activation-only scorer reads the model's last-layer Q/K activations and
measures attention to observed context, with a penalty for recent activation
movement. These attention activations concern input positions, not decoder token
confidence. The historical diagnostic scorer is exactly log-confidence minus
target-specific temporal Fisher movement; it is marked confidence_scoring=True.
Neither diagnostic ranking nor its outcomes qualify as a fully activation-only
scheduler. Both share the same new confidence-free count controller.

All proposals are normal greedy decoder values. Intervention predictions are
not substituted as new committed values; this keeps the experiment about count.
Rankers are held fixed between fixed-count and guarded-count arms. All branch
and unchanged-control sequence evaluations are counted.

## Gates and early results

The oracle-state repeated-bit/parity fixture confirms that the count mechanism
can expose the right phase when activations encode the exact conditional
information. It produces 1,16,16,16,15 for repeated bits and
1,16,16,16,8,4,2,1 for parity in 64 dimensions. This validates count logic only;
it is not a validation of LLaDA's representation or a claim that every fixed
schedule fails. Artifacts: `/root/Strace/outputs/latent_count_toys/diagnostic.json`.

On arithmetic, the tighter proposed-value-gradient guard (η=0.01) with the
historical Fisher scorer scores 73.96% / 70.31% on seeds 0/1, versus the same
scorer's 73.96% / 70.05% with fixed singleton quotas or EB counts. That is a tie
and one additional answer among 384 slots; it is a small diagnostic observation,
not an established performance improvement. It costs 67.625 / 66.625 forwards
per canvas. The activation-only scorer remains weaker, so count is not the sole
unresolved problem under the strict constraint.

The two-seed artifacts are in
`/root/Strace/outputs/latent_count_candidate_gradients/`. An earlier diagnostic
adapter shape error interrupted the first uniform-metric run; the complete
corrected comparison is `/root/Strace/outputs/latent_count_arithmetic_v2/`.
Source snapshots preserve each implementation and diagnostic/candidate flags.

The HumanEval count-isolation run uses 128-step caps, 512 generation positions,
normal batch 20, η=0.01, and candidate-gradient geometry. Its fixed-count Fisher
control scores 3/20, compared with the historical Fisher+EB 14/20. Remaining
guard and activation-only arms are checkpointed under
`/root/Strace/outputs/latent_count_humaneval20/`.

The HumanEval comparison is now complete. With the historical Fisher scorer,
fixed quota scores 3/20, plain singleton 14/20, and the new growth guard 14/20.
Singleton and guard have identical passed tasks. With the activation-only scorer,
fixed quota scores 0/20, singleton 7/20, and growth guard 8/20. The guard wins
HumanEval/9 and /12 and loses /15 against singleton. This is net +1, with no
robust-improvement claim on the repeatedly reused subset.

Both growth guards use 1144 forward evaluations/problem, versus 128 for the
quota and singleton controls. Mean committed positions are 135.05 with the
diagnostic scorer and 140.40 with the activation-only scorer, compared with 128
for singleton. The guard is mostly conservative; its 14/20 is not evidence that
adaptive counts beat simpler safe pacing. Native EB remains 13/20. The strict
activation-only policy remains below that baseline.

The complete paired report is
`/root/Strace/outputs/latent_count_ablation_report/README.md`.

## Relation to the paper

[Wainwright's growth-geometry paper](https://arxiv.org/html/2608.13520) motivates
forced-mask conditional-growth measurements and allocating steps according to
changing difficulty. Our realized greedy latent displacement, selected target,
and intrinsic layer scale are surrogates. They do not inherit the paper's
posterior-KL estimation or certified sampling guarantees. A full growth-profile
clock remains a distinct proposed experiment in `ugc-count-direction.md`.
