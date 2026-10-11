# One-pass scheduling from the three-term objective

Research proposal, 7 October 2026. This is a proposed surrogate and experiment
plan, not a tested improvement on HumanEval.

## Source and constraints

The original `Schedule_tracing.pdf` was unavailable when this proposal and its
experiments were designed; `spectral.py` was also absent. The attached October 3
memo and historical transcript preserved the
objective's three components: marginal model error, within-batch dependence,
and convergence through information delivered to remaining masked positions.
This note works from that recorded formulation; it does not claim to reproduce
unavailable PDF notation or coefficients.

The PDF has now been attached. Its literal Equation 17, count implications,
and relationship to the spectral quadratic are read directly in
`docs/schedule-tracing-objective-readout.md`. The Gaussian formulation below
remains an alternative surrogate, not a direct implementation of Equation 17.

The design uses one ordinary frozen-model forward per denoising step. At 128
steps, the inference budget is 128 forward evaluations per problem. Decoder
token proposals supply the values to commit. Token confidence, vocabulary
entropy, probability margins, EB counts, learned heads, and learned policies do
not enter scheduling. No backward pass or counterfactual forward is required.
Projection, attention reconstruction, matrix factorization, and subset selection
still cost time and memory, which must be measured separately.

## Start with the objective, then define the observables

For masked positions R and proposed batch S, the recorded theoretical target is

    J(S) = lambda_G I_q(X_S; X_(R\S) | C)
           - lambda_M sum_(i in S) KL(q_i || p_i)
           - lambda_D TC_q(X_S | C).

Neither true model error nor token dependence is observable from a normal pass.
The information term excludes the selected positions from the target. Removing
a position cannot count as information delivered to itself.

A terminal batch has zero external information. Consequently this objective
alone cannot express a preference for finishing quickly. The practical
convergence utility must explicitly include progress, or there must be a
separate progress constraint. This proposal uses

    J_hat(S) = lambda_G G_lat(S) + rho |S|
               - lambda_M sum_(i in S) m_i
               - lambda_D D_lat(S).

Here rho is an explicit, fixed computation-versus-risk tradeoff. It is not
derived from the original KL identity, and it is not an EB entropy threshold.
The minimum action is one token; an override when all marginal gains are
nonpositive must be recorded. No uniform four-token quota is imposed merely
because 512/128=4. A 128-step cap does not guarantee all 512 canvas positions
will be committed.

## Observable 1: commitment risk

Reuse the useful activation movement component of temporal Fisher, without its
confidence ranking or runtime probability weighting. Let P be the fixed
decoder-weight projection already available in `schedulers/latent_count.py`.
Its metric approximates uniform centred frozen readout geometry, not the
probability-weighted Fisher metric. At position i, calculate

    temporal_i = || P^T (h_i(t) - h_i(t-1)) ||^2,
    curvature_i = || P^T (h_i^L - 2 h_i^(L-2) + h_i^(L-4)) ||^2.

Earlier depth states are passed through the same final normalization as the
state entering the decoder. Normalize these energies by explicit shared RMS
scales and numerical floors over current masked positions. Retain the scales
in traces. On the first step the temporal term is unavailable; use depth alone,
and record the missing history rather than interpreting it as evidence of
stability.

Let b_i be the sum of the normalized energies. An inexpensive incoming term is

    incoming_i = sum_(j in R, j != i) mean_head(A_ij) b_j,
    m_i = b_i + gamma incoming_i.

A_ij is the last block's actual attention from target i to source j, captured
from the normal pass, including its native mask/bias. It is attention over
input positions, not vocabulary probabilities. Only attention to unresolved
positions enters incoming instability. A cheap ablation omits incoming entirely
so that attention reconstruction overhead and benefit are identifiable.

This is a proxy for susceptibility to future changes, not a prediction of
token correctness or an estimate of KL(q_i || p_i). A stable wrong answer can
have small m_i. Earlier experiments did not establish that this risk proxy alone
produces good activation-only rankings.

## Observables 2 and 3: an explicit latent Gaussian surrogate

Avoid calling an arbitrary Gram matrix token mutual information. Instead state
the surrogate distribution explicitly and calculate its own information terms.
From the same pass, construct depth-innovation features

    z_i = concat(P^T(h_i^L - h_i^(L-2)),
                 P^T(h_i^(L-2) - h_i^(L-4))).

Use shared RMS normalization per feature block, with a numerical floor. Do not
use decoder confidence to weight these features. A candidate initial low-rank
version retains 16 modes of their Gram matrix; this is an explicit approximation
to be ablated, not a token selection ranking. Let Z denote the resulting
features, with their spectral scales retained, and define

    K = sigma^2 I + Z Z^T,       sigma^2 > 0,
    xi_R ~ Normal(0, K).

Sigma is a regularization parameter, not calibrated model uncertainty. Eigen
truncation, feature normalization, and sigma change the surrogate and must be
recorded. Use the same K for both terms below at a particular history.

The within-batch dependence proxy is the exact total correlation of this
Gaussian surrogate:

    D_lat(S) = 1/2 [ sum_(i in S) log K_ii - log det K_SS ].

It is nonnegative and zero for a singleton. This avoids interpreting negative
Gram entries as negative dependence. It does not account for nonlinear token
interactions, such as parity, invisible to a covariance model.

The external information proxy is

    G_lat(S) = 1/2 [ log det K_SS + log det K_TT - log det K_RR ],
    T = R \ S.

It is exactly I(xi_S; xi_T) for the stated Gaussian, and zero for both empty and
terminal batches. Revealed sources are excluded from the target at every
candidate addition. Unlike a sum of source-to-target scores, it accounts for
redundancy within this particular Gaussian representation.

The unresolved empirical assumption is that depth innovations encode useful
remaining uncertainty and that their correlations predict beneficial reveals.
Committing a guessed token does not observe the true Gaussian variable xi_i.
Therefore G_lat is structural acquisition potential, not a measured benefit of
the actual token commitment. The risk proxy does not remove this gap. These
quantities are NOT certified approximations or bounds for token MI, token TC,
or generation KL. Earlier PCA/response experiments were poor; replacing probes
with this algebra is a compute reduction and a clearer hypothesis, not evidence
that the proxy is better.

## One decision for both token identity and count

For a current S and i in T, define Gaussian conditional variances

    v_i(S) = Var(xi_i | xi_S),
    v_i(T\{i}) = Var(xi_i | xi_(T\{i})).

The exact objective increment is

    Delta_i(S) = rho - lambda_M m_i
                 + (lambda_G/2) log [v_i(S) / v_i(T\{i})]
                 - (lambda_D/2) log [K_ii / v_i(S)].

The information increment can be negative: adding a source also removes it
from the surviving targets. Replacing it with an always-positive outgoing
influence score would no longer optimize the stated objective.

Select the greatest positive increment, update S, and recompute increments.
Stop when no positive increment remains, subject to an explicitly recorded
maximum batch size used as a compute bound. Then k=|S| is the count. If the empty
batch wins, force the best singleton and record the exception. This greedy rule
is a local algorithm, not a claim of global optimality. Exact enumeration on a
small candidate pool is an algebraic reference check.

The spectral decomposition supports conditional-variance calculations through
low-rank solves. For K=sigma^2 I+ZZ^T,

    v_i(S) = sigma^2 + z_i^T
               [I + sigma^(-2) Z_S^T Z_S]^(-1) z_i,
    v_i(T\{i}) = 1 / (K_TT^(-1))_ii.

Woodbury and rank-one updates make these computations operate on the retained
latent modes. There is no full vocabulary Fisher tensor or repeated model
execution. Do not rank by M^(1/2)b: that is not the optimizer of either this
objective or the recorded quadratic v^T Mb - 1/2 v^T Mv.

## Experiments that distinguish the hypotheses

First validate the Gaussian algebra on small matrices: conditional-variance
marginals must match direct log-determinant objective differences; independence
must give zero dependence and external gain; correlated batches must incur
positive dependence. This verifies the surrogate, not its relation to the model.

This algebra check is complete: 100 random positive-definite covariance trials
covered 409 candidate additions. The largest objective-increment discrepancy was
4.0e-15 and the largest conditional-variance discrepancy was 6.3e-15. With toy
coefficients rho=0.05 and all three objective weights equal to one, eight
independent zero-risk variables selected a batch of eight, while eight variables
with a shared latent factor selected one. High-risk independent variables
required the recorded singleton fallback. These are constructed Gaussian
examples, not model benchmarks. The report is
`/root/Strace/outputs/latent_three_term_algebra/verification.json`.

For the first model run, use the existing two arithmetic seeds and then the
same HumanEval-20 exploratory set, with 128 outer steps and greedy proposals.
Use a small, declared parameter set rather than a wide search on reused tasks.
Compare these activation-only arms:

1. Full three-term objective.
2. No incoming term in risk.
3. No within-batch dependence penalty.
4. No external-information reward.
5. Same initial singleton marginal ranking, but fixed singleton counts, to
   isolate the usefulness of adaptive count.

Native EB is the baseline. The historical confidence/Fisher method may be shown
as a clearly labelled diagnostic, not called an eligible new method. For all
arms record accuracy, paired outcomes, elapsed time, forward evaluations,
committed positions, count trajectories, objective terms, and forced-singleton
frequency. Parameter sensitivity is central: rho, sigma, and objective weights
can dominate count, and none has a proved optimal setting.

Reject the candidate if its count stays effectively singleton without a quality
gain, if it reproduces premature large batches, or if latent objective gains do
not translate to correctness. Keep any finite-intervention validation in a
separate small audit; do not hide its compute inside the one-pass benchmark.
Full HumanEval-164 follows only a useful small-run result. All these tasks have
already informed earlier exploration, so report subsequent tuning honestly.

## Why this replaces the previous growth guard

The completed count-isolation study used 1144 forwards/problem for the guard,
versus 128 for fixed/singleton controls. With the historical confidence-based
scorer, both singleton and guard passed the same 14/20 tasks. Under activation
only, singleton passed 7/20 and guard 8/20, below EB's 13/20. Those results do
not justify making expensive lookahead the primary design.

This proposal preserves the original separation of three quantities, gives
count and position selection a shared explicit objective, and makes spectral
algebra useful within a normal-forward budget. Its missing evidence is the
quality of the latent surrogate, not the algebra or the availability of GPU
memory.

## Implemented evaluation and gradient refinement

The prototype is `schedulers/latent_objective.py`; run matched benchmarks with
`python -m schedulerlens.latent_objective_bench`. It implements batched exact
Gaussian marginal increments with low-rank inverse updates, normal-pass depth
hooks, an optional last-block attention reconstruction, cached temporal states,
and resumable 16-step checkpoints. The frozen decoder sketch has width 16 and
the Gaussian feature rank is 16 in these experiments.

The initial projected-norm risk and full-weight Gaussian information variants
were poor on arithmetic. The next measured risk variant uses the existing
decoder gradient of the proposed value y_i:

    d_i = (W_(y_i) - mean_v W_v) / ||W_(y_i) - mean_v W_v||,
    c_i = d_i^T (h_i^L - 2 h_i^(L-2) + h_i^(L-4)),
    u_i = d_i^T (h_i(t) - h_i(t-1)),
    b_i = normalized(c_i^2) + 0.15 normalized(u_i^2),
    m_i = b_i + 0.25 sum_(j in R, j != i) mean_head(A_ij) b_j.

Normalized scalar energies use their current shared masked-position mean, a
1e-12 numerical floor, and a maximum of eight. The temporal component is absent
on step zero. This measures activation change along the proposed value's
readout-gradient direction. It does not consume the current logit value,
probability, confidence, or probability margin. Frozen head weights and the
ordinary decoder's proposed token identity are used; no gradient is backpropagated
and no new parameters are fitted. Earlier/final states share the same final
normalization. For this variant only, retain the previous full final hidden
state to calculate the next temporal projection in its current proposal
direction; this is additional memory, not another model pass.

One alternative tested signed late-layer support along d_i, transformed into a
nonnegative risk. It was weaker on the small arithmetic benchmarks.

Among the declared arithmetic settings, proposal-gradient curvature with
lambda_G=0.1, lambda_M=0.15, lambda_D=1, rho=0.05, sigma^2=1 and maximum count 16
scored 81.25% / 79.17% on seeds 0/1. Native EB scored 71.09% / 69.79%. Each used
12 ordinary forwards per canvas. The winning candidate sometimes selected two
tokens and finished individual canvases before others, but mostly selected one;
it is not evidence of a solved count problem. These are exploratory results on
reused, correlated worksheets. The no-information variant scored 81.25% / 78.39%,
and its matching singleton control scored 81.25% / 78.91%.

HumanEval-20 evaluation is complete. Forty arithmetic configurations were
evaluated on both seeds, and 15 configurations on HumanEval with a 128-step cap.
The arithmetic leader passed **0/20** on HumanEval. Native EB reproduced its
historical 20 completions exactly and passed **13/20**.

Additional variants centred depth innovations over masked positions, scaled
covariance by the observed risk, replaced depth curvature with actual temporal
changes, and added attention to observed context into risk. These remain
computed proxies; centring does not make hidden covariance token dependence.
Temporal-only risk bootstraps with one reveal because no history exists on the
first step. A risk-scaled variant filled the entire canvas in 57 steps but
passed 0/20, demonstrating that completing more positions is insufficient.

The best new adaptive one-pass variant, `context_smallgain`, passed **4/20**.
It normalizes unresolved attention mass by its masked-position mean and adds
0.05 proposal-gradient curvature, 0.15 temporal gradient movement, and 0.25
incoming instability. It uses centred depth-innovation covariance, rank 16,
sigma^2=1, lambda_G=0.01, lambda_M=0.15, lambda_D=1, rho=0.15, and maximum count
16. It averaged 1.717 reveals/step and 219.75 committed positions/problem,
with 128 forwards/problem, no backward pass, no confidence, no EB count, and no
training. It has no wins and nine losses against EB on the reused 20 tasks.

Its no-information sibling passed 3/20 versus 0/20 for that sibling's matching
singleton control. This is a small count advantage within a weak scorer. A
singleton control with the exact winning 0.01 information weight was not run;
do not attribute its extra task to count alone. None of the newly evaluated
methods beats EB on HumanEval or justifies a full-164 promotion.

The completed report, per-task outcomes, compute, source snapshots, and
reproduction artifacts are linked from
`/root/Strace/outputs/latent_objective_report/README.md`. The main unresolved
problem is that the commitment-risk and geometric information proxies do not
reliably select correct code tokens. Future work should validate those terms
on code states and retain the earlier cheap activation-only singleton scorer
(7/20) as a scoring control, before changing scoring and count together again.
