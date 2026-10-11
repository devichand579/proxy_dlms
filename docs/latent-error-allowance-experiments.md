# Count experiments under a latent error allowance

8 October 2026. Experiment implementation and protocol. Results are in
`outputs/latent_allowance_report/README.md`; the scope of each control is explicit.

## What is held fixed

The main diagnostic uses the existing score: log confidence minus 0.15 times
normalized target temporal Fisher movement. It is held identical across count
controllers. Its confidence term makes it a diagnostic, not the final
confidence-free scheduler requested by the user.

The activation-only arms retain the earlier score: last-block attention to
observed context divided by its current masked-position mean, minus 0.15 times
normalized temporal hidden movement in a fixed decoder-weight metric. The
metric uses a width-64 sketch of the frozen centred output weights. It is
recomputed from model weights, not trained from outcomes. The singleton control
previously passed 7/20 and is included to identify scoring versus count effects.

## New probability-free count controllers

Every step uses the normal frozen-model forward and cached states from the
previous normal pass. No hypothetical canvas, branch forward, trained policy,
or backward pass is used.

The measured response excludes positions committed in the preceding step:
only currently masked positions are targets. For uniform decoder geometry P,
measure temporal movement in P coordinates. For the gradient-union metric,
use the uniform union of up to four decoder value proposals from each of the
current and preceding contexts. Discard their logit levels and probabilities.
At position j, measure

    e_j = Var_uniform-values[W_v^T(h_j(t)-h_j(t-1))]
          / Var_uniform-values[W_v^T(h_j^L-h_j^(L-2))].

The same frozen final normalization is applied to the earlier depth state. A
floor equal to 0.001 times the masked-position mean native variance protects
small denominators; numerical ratios are clipped to 100. Head-common readout
directions cancel in the variance. These are changes in model beliefs relative
to an internal scale, not true per-position model KL errors. The model's
ordinary decoder supplies proposal identities; no confidence or entropy
quantity enters this metric.

### Response feedback

Take the 99th percentile of e over surviving targets. A high quantile limits
dilution by inactive output tails. The controller proposes

    k_next = floor(0.8 k_previous sqrt(allowance / observed_response)),

clipped to [1,16] and at most twice the preceding count. Use the larger of the
instantaneous response and a 0.5 EMA. A grounded variant weights target movement
by attention to other still-masked positions, aiming to distinguish a recently
resolved position from one susceptible to further unknown context.

A relative variant sets the allowance to a fixed multiplier of the first
observed singleton response on that input. This computes an activation-based
reference scale; it does not fit a policy or use correctness labels. The first
step always commits one token because no response history is available.

Feedback reacts after an action and cannot undo an earlier bad batch. Its
square-root scaling is a local approximation, not a nonlinear model bound.
Excess responses and the actual counts are retained in traces.

### Ranked-prefix interaction allowance

Capture the actual last-block attention A during the normal forward. It is
over input positions, not vocabulary probabilities. Define the symmetric
susceptibility proxy

    d_ij = (A_ij+A_ji)/2 sqrt((1+e_i)(1+e_j)),    d_ii=0,
    estimated_cost(S) = sum_(i in S) e_i + sum_(i<j in S) d_ij.

Select the largest ranked prefix satisfying a declared cost allowance. This
separates the singleton response term from pair interaction, without interpreting
a shared hidden Gram as token MI. Risk-only and pair-only ablations remove the
corresponding part. A grounded variant weights e by unresolved incoming reads.
A subset variant retains the top-ranked anchor and skips later candidates
whose incremental cost does not fit, to test selection beyond a strict prefix.

If even the best singleton exceeds the estimated allowance, commit one token
and record the exception. The exception does not satisfy the stated allowance.
Attention is an interaction proxy, not a certified TC estimator; pairwise costs
do not capture all higher-order dependence.

## Separately labelled Fisher-geometry diagnostic

The probability-weighted diagnostic measures the projected Fisher trace of the
existing softmax readout. Let Z_v = W_v P be the fixed vocabulary gradient
dictionary, centred over vocabulary. Then

    u_i = [sum_v p_i(v)||Z_v||^2 - ||sum_v p_i(v)Z_v||^2]
          / mean_v ||Z_v||^2.

This is expected squared model-score gradient in frozen latent directions. It
uses model probabilities as metric weights, with no maximum-confidence feature,
margin or entropy calculation in its count rule. It is nevertheless NOT a
strict probability-free candidate. It is a diagnostic for the difference between
observed motion and current predictive uncertainty.

Its allowance sums u_i and pair interactions weighted by sqrt(u_i u_j), so
estimated interaction vanishes for a concentrated model distribution. It can
distinguish equal-entropy distributions whose candidate gradient vectors have
different geometry. The geometry is not a guarantee of semantic correctness;
confidently wrong predictions remain possible. The projected moment calculations
add readout algebra and softmax work, not another backbone forward or training.

## Protocol and controls

The arithmetic study uses 32 canvases / 384 slots per seed, seeds 0 and 1, and
12-step caps. The HumanEval study uses the original first 20 problems, greedy
raw completion, 512 positions, 128-step caps, and standalone batches of 20.
No early-EOS generation stop is added. Decoding and the three-second correctness
executor match the established study. Reference answers and hidden tests are
used only after generation. These tasks have been repeatedly exposed, so the
study is exploratory.

Native EB, Fisher with EB count, Fisher singleton, and activation singleton are
controls. New count candidates and scoring scopes are reported separately.
Record accuracy, actual forwards, generation time, count trajectories, positions
committed, allowance exceptions, paired outcomes, and identical completions.
Many remaining masks can lie beyond the decoded function; completing more
positions is not itself success.

Experiments snapshot the scheduler, benchmark and relevant helper sources,
plus fixed decoder geometry. State is checkpointed every 16 steps. Source or
configuration changes require a fresh output directory.

## Exact small-distribution checks

`schedulerlens/latent_allowance_toys.py` compares the entropy bound, exact TC,
pairwise approximation, and the draft's natural-weight objective using an
oracle Bayes model on eight bits. This isolates dependence with model error zero:

- Independent bits: entropy-bound scheduling needs eight steps; exact-TC count
  takes one, both with sampling KL zero.
- Copied bits: exact TC and the entropy bound choose [1,7], with KL zero. The
  literal natural-weight draft chooses all eight and incurs 7 log(2) KL.
- Parity: exact TC chooses [7,1] with KL zero; pairwise MI misses the dependence,
  chooses all eight, and incurs log(2) KL.

These oracle sampling results do not validate the neural estimators or establish
greedy HumanEval accuracy. They show why dependence estimation and higher-order
audits matter. The JSON is `outputs/latent_allowance_toys/diagnostic.json`.

The implementation checks cover cost accumulation and exceptions, independent
versus coupled counts, feedback scaling, readout gauge/scale invariance, exclusion
of committed sources, singleton bootstrap, source-prefix preservation, and one
forward per step.

## Completed neural results

The study completed 33 arithmetic settings across two seeds and 13 standalone
HumanEval-20 settings. Native EB reproduced its historical 20 completions and
passed 13/20. The Fisher singleton and activation singleton controls also
reproduced all their historical completions, passing 14/20 and 7/20.

Fisher-ranked probability-free temporal allowance scored 11/20, grounded
allowance 12/20, and relative feedback 13/20. Grounded uniform feedback matched
14/20 but added only five two-token decisions across 20 trajectories, with
128.25 mean committed positions: effectively the singleton control.

Fisher-geometry allowance at 0.01 and 0.05 matched 14/20 with average counts
1.284 and 1.757, and 164.35 and 224.90 committed positions. Every decoded
completion was identical to the Fisher singleton control. More committed
positions therefore did not establish improved useful parallelism. These count
rules are independent of EB and do not use an entropy or maximum-confidence
threshold, but probabilities weight their gradient geometry and confidence
remains in their reference ranking.

The strict activation-only grounded-feedback candidate retained 7/20, with
129.4 mean committed positions. Activation-only temporal allowance scored
6/20. Activation ranking with the Fisher-weighted diagnostic count also scored
6/20. None improved on the cheap activation-only singleton control. No full-164
promotion is warranted by these exploratory results.

Source snapshots, forward/reveal accounting, and correctness totals were
verified for all 79 completed neural trajectories. Eight implementation checks
passed. These results identify estimator weaknesses; the exact toy examples
establish the dependence-budget motivation without validating those estimators.
