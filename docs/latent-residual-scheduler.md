# A scheduler whose score and count come from continuous activations

This prototype implements the user's October 7 constraint: no EB count rule,
confidence, token entropy, probability margin, risk head, or trained policy in
the scheduler. The model's ordinary greedy decoder supplies token values;
selection and cardinality use only activations and their measured changes.

## Source objective and availability

The original `Schedule_tracing.pdf` and `spectral.py` are unavailable in this
checkout. The attached October 3 design memo preserves their quadratic objective
and its audit. This prototype uses that recorded formulation rather than claiming
to have reread the original PDF. It also respects the user's subsequent ban on
the memo's historical learned-head proposals.

For the recorded objective F(v)=vᵀMb−½vᵀMv, the conditional marginal of adding i
is (Mb)ᵢ−(Mv)ᵢ−½Mᵢᵢ. Ordering by M^(1/2)b is not the derived binary optimizer.
Here we construct an observable latent control objective, diagonalize its actual
interaction matrix, and optimize both the subset and its size. This is a
research surrogate, not an observable substitute for true data-distribution KL.

## The measured latent field

Capture normalized activation states after layers L−4, L−2, and L. Apply the
checkpoint's same frozen final normalization to the earlier states. Define

    r_j = h_j^L − 2 h_j^(L−2) + h_j^(L−4).

This is a late-layer curvature residual. It describes how far the representation
is from continuing a straight late-layer trajectory; it does not certify a
correct answer. Only these three frames are retained, not all hidden layers.

The candidate pool contains the positions with the smallest normalized residual
energy plus normalized temporal movement. The temporal coefficient is 0.15.
The pool size is a compute limit (initially four), not a count borrowed from a
baseline. The chosen count ranges from one to the size of that pool.

For candidate i, reveal its normal-pass greedy token in a temporary canvas and
measure the new residual. An unchanged canvas is run with the identical probe
batch shape. On the fixed external target set T (masked positions outside the
whole candidate pool), define

    a_i = vec(r_T^control − r_T^(i revealed)).

Excluding the candidate pool keeps targets identical across subsets and avoids
crediting a source for its own removal. Numerical control-versus-normal residual
differences are recorded separately. Probe residuals are never substituted into
the actual canvas; only selected greedy token values are committed.

## Joint score and count

Let A stack the action responses as rows, r=vec(r_T), and E=||r||². Define

    M = 2 AAᵀ / E
    g_i = 2 a_iᵀr / E + τ − α c_i − β u_i
    F(v) = gᵀv − ½vᵀMv,  v∈{0,1}^k.

The curvature cost c_i is the source's normalized residual energy plus temporal
movement. Incoming susceptibility u_i sums the final-activation changes at i
caused by the other candidate reveals, excluding its own action. Both are
normalized using only the current model activations. Initially α=β=0.25;
τ∈{0.25,0.5,0.75,1} is a fixed progress-versus-stability objective coefficient.
These are explicit hand-selected research parameters, not learned weights or
claims of automatic optimal calibration.

Without the costs and progress reward, the expression equals

    (||r||² − ||r − Aᵀv||²) / E.

Thus the quadratic measures predicted reduction of the surviving latent field.
Interactions represent overlap and cancellation of measured responses. They are
not interpreted as mutual information or total correlation; signed cross terms
are legitimate for this residual-control objective.

Diagonalize M=U diag(λ) Uᵀ and evaluate the energy as Σ λ_m(Uᵀv)_m². For a pool
of four candidates, compare all 16 subsets, including the empty set. Larger
pools up to 12 are supported. This returns the exact optimizer of the surrogate
within the pool, not necessarily the best action under the nonlinear model.
Prefer smaller subsets on objective ties. If the empty set wins, force the best
singleton and record the override. There is no entropy budget or confidence
threshold. Count is sum(v) from the same optimization that chooses positions.

When no external targets remain, the external gain is zero. The explicit
progress reward, latent source cost, and incoming susceptibility still decide
the terminal count. This avoids making terminal progress disappear from the
objective merely because external information is zero.

## What needs to be demonstrated

This is a new independent scheduler in this repository, not an established
paper-level novelty claim. Related approaches already include
[Info-Gain's speculative information objective](https://arxiv.org/html/2602.18176v3),
[temporal probability stability](https://arxiv.org/abs/2604.17068), and
[D5P4's DPP-based selection of intermediate beam sequences](https://arxiv.org/abs/2603.19146).
The candidate distinction here is variable-cardinality token selection using
measured contraction of a continuous internal residual, with activation-only
costs and exact spectral subset optimization. A deeper related-work review and
positive ablations are required before calling that a novel research contribution.

The key empirical risk is that latent smoothness can improve while generated
answers remain wrong. Single-action responses can also fail to add linearly.
Therefore compare predicted versus actual simultaneous latent contraction and
completed task accuracy. Do not promote a method on residual reduction alone.
No tests, answers, reference completions, or reference-token error enter scores.

Five checks cover exact optimization, independent versus correlated counts,
progress override, the quadratic reduction identity including signed terms, and
invariance under changing output confidence without changing activations or
token proposals. Integration checks preserve prompts, gradient state and exact
sequence work accounting.

## First completed-trajectory experiment

    python -m schedulerlens.latent_residual_bench

Uses the existing 32 arithmetic canvases and 12-step cap for seeds 0 and 1.
Both native EB and EB+Fisher are benchmark controls only. Four latent objective
settings use the same frozen model and canvases. References enter only the
post-generation accuracy evaluator. Each probe has a matched unchanged control,
and all sequence evaluations are counted. Results are checkpointed per arm at
`outputs/latent_residual_arithmetic/seed0/` and `seed1/`.

The first two-seed run completed. The best curvature settings scored
64.58% / 69.79%, versus native EB 71.09% / 69.79%. Correcting surviving-target
bookkeeping did not improve those best settings. Diversifying a six-source pool
through latent spectral leverage scored 63.80% / 63.80% at progress weight 0.5.
All answer slots were filled. These candidates were not promoted to HumanEval.
The current source includes the tested `survivors` target mode and `spectral`
pool mode; each historical artifact retains its tested source snapshot.

In `survivors` mode, retain all masked targets and evaluate latent reduction on
R\S separately for every candidate subset. Target-specific response Gram spectra
evaluate each surviving target's quadratic exactly. The target mask depends on
S, so this is not one fixed global quadratic. Tests verify it against direct
subset reconstruction and ensure a revealed source receives no own-removal gain.

Proceed to HumanEval-20 at 128 steps only if subsequent experiments establish a
useful signal. The full benchmark reference is EB 47/164, EB+Fisher 55/164, and
the probability-based information method 56/164. All 164 have now been examined,
so future tuning on their outcomes is exploratory rather than a fresh holdout.
