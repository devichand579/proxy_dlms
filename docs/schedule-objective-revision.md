# Revising the objective and incorporating measured results

8 October 2026. Proposed formulation and experiment plan, not a new evaluated
scheduler. Source: the original `Schedule_tracing.pdf` and completed experiments.

## Keep the exact quality target

At context C, for a batch S independently sampled from per-position model
distributions, define

    M_C(S) = sum_(i in S) KL(q_i(.|C) || p_i(.|C)),
    D_C(S) = TC_q(X_S | C).

The exact per-batch data-distribution error is M_C(S)+D_C(S). For a complete,
admissible schedule, the sequence KL is the sum of these errors averaged over
the data-conditioned histories. Preserve this decomposition. Do not equate
model error with context similarity or temporal stability.

The overall design target is quality under a model-evaluation budget. A soft
version adds a cost per model evaluation; a hard version imposes the permitted
evaluation count and a completion condition. If every run uses exactly 128
evaluations, a fixed per-evaluation cost is constant and cannot distinguish
their policies. Extra branch forwards and backward work must be accounted for.

Greedy HumanEval accuracy is a separate experimental target: the deterministic
output law is not the softmax sampling distribution in the KL derivation.
Claims about that derivation do not certify program correctness.

## Separate useful benefit from an error allowance

Define external information on surviving positions

    G_C(S) = I_q(X_S; X_(R\S) | C).

This fixes self-removal counting and the overlapping-set approximation. But
external information alone gives no reward to a terminal batch or independent
positions. Introduce an explicit productive-completion benefit P_C(S), separate
from the information-theoretic identity. Its ideal meaning is reduction in the
remaining work needed to complete the relevant output. It is a continuation
value, not automatically the number of masks removed.

One practical framework is

    maximize     U_C(S) = P_C(S) + lambda_G G_C(S)
    subject to   M_C(S) + D_C(S) <= epsilon_t,
                 S nonempty, S contained in R,
                 model-evaluation and completion constraints.

Its penalized surrogate has the familiar three conceptual components

    J_C(S) = U_C(S) - alpha M_C(S) - beta D_C(S).

Keep alpha, beta, and lambda_G independent when calibrating practical proxies.
With the exact errors the quality sum has unit weights; different practical
weights reflect surrogate scaling, not a new KL identity. Do not impose the
draft's lambda_1=lambda_3 relationship, which makes its literal Equation 17
monotone and fills the maximum allowed batch.

P_C(S) and the practical error allowance remain quantities to define and
validate. Treating P as rho times the number of committed tokens is an ablation,
not the default claim of productive work. A completed 512-position canvas in
57 steps passed 0/20; irrelevant tails and incorrect values can dominate counts.
If no nonempty batch satisfies an estimated allowance, a progress override
violates that allowance and must be recorded. Estimated bounds are not certified
data-error bounds.

This formulation separates conservative quality control from the temptation
to maximize a potentially misleading information reward. It does not repair
an incorrect estimator by itself.

## How EB fits mathematically

For true conditional marginal entropies,

    TC_q(X_S|C) = sum_(i in S) H_q(X_i|C) - H_q(X_S|C)
                <= sum_(i in S) H_q(X_i|C) - max_(i in S) H_q(X_i|C).

The inequality follows because joint entropy is at least each marginal entropy.
EB's existing count rule has exactly this upper-bound form, with model marginal
entropies substituted for the unknown data entropies. It can be interpreted as
dependence control, not as an estimate of marginal model error. Substitution
of model entropies needs accuracy/calibration assumptions to become a guarantee
about q.

The current implementation chooses the largest feasible prefix of an existing
ranking. That is a prefix-restricted heuristic, not an exact optimizer over all
subsets. This is the reference to replace when developing an independent count:
estimate batch interaction more selectively than a marginal-entropy upper bound,
then choose productive positions under that allowance. Replacing the estimator
with a raw hidden-state Gram is not justified.

## Put the empirical components in their appropriate roles

| Measured component | Role in the framework | What the result supports |
|---|---|---|
| Confidence plus temporal Fisher | Readiness/risk ranking proxy | 55/164 versus EB's 47/164 at 128 forwards; useful in this hybrid, not proof of a standalone latent model-error estimator. |
| Multi-value reveal information | Candidate external-benefit proxy | The full information+risk+incoming method reached 56/164 at 250.34 forwards; only one additional task over Fisher. |
| Incoming sensitivity | Candidate susceptibility-to-future-context term in readiness | Tested jointly with other changes; its isolated full-set contribution is unmeasured. |
| Source-value harm and mixture consistency defect | Model-update risk diagnostics | They can penalize harmful/inconsistent updates, but are not true KL(q_i||p_i). |
| Gradient curvature | Candidate readiness signal | 81.25%/79.17% arithmetic, but 0/20 HumanEval for its winning setting; task transfer failed. |
| Gaussian covariance information/dependence | Computable geometry surrogate | Best new adaptive one-pass method 4/20 versus EB 13/20; not a validated replacement for token information or TC. |

For the existing hybrid, a diagnostic readiness cost can include negative log
confidence plus normalized temporal Fisher movement. The latter is the local
readout-geometric quantity

    r_i = 1/2 delta_h_i^T F_i delta_h_i.

It measures model-prediction change, not data-distribution model error. Existing
F_i uses model probability weights. Removing the confidence term or replacing
the Fisher metric by a probability-free gradient metric changes the method and
must be tested; the 55/164 result cannot be transferred to that new method.

The final independent latent scheduler must comply with the user's no-learned-
policy, no-confidence-ranking, and low-pass requirements. Historical hybrids
serve as labelled diagnostics. Neither their EB counts nor a trained CST is
implicitly authorized as part of the final method.

## Keep spectral algebra separate from the estimators

A pairwise local approximation may be written

    J_pair(v) = a^T v - 1/2 v^T B v,      B_ii=0,

where a contains singleton productive benefits minus readiness costs. Pairwise
corrections for overlapping gain and batch dependence need distinct estimation
and interpretation, even if their weighted sum is B. General external gain
changes when selected positions leave the targets, and higher-order interactions
require more than this fixed pair approximation.

The exact pair-approximation marginal is

    Delta_i(S) = a_i - sum_(j in S) B_ij.

Conditional positive-gain selection and allowance checks provide both the
ranking and count. This is a local algorithm, not a global optimality claim.

If B is symmetric with nonnegative off-diagonal penalties, let

    d_i = sum_j B_ij,      Q = diag(d) + B.

Q is PSD, and for binary v the identical objective is

    J_pair(v) = (a + d/2)^T v - 1/2 v^T Q v.

This gives a legitimate PSD spectral representation without inventing a
dependence self-cost: the diagonal is compensated in the linear term. This
algebra already appeared in the latent-flow prototype; it is not new performance
evidence. If gain corrections introduce negative pair terms, the PSD construction
does not apply directly. Clipping eigenvalues changes the objective.

Use spectral decomposition to compute the actual objective and its increments,
not to infer that Q^(1/2)b is its optimizer. A shared activation Gram should not
simultaneously stand for model error, information transfer and dependence.

The compensated quadratic identity was checked numerically on all 128 binary
subsets of a seven-node nonnegative interaction graph; Q was PSD. A separate
two-bit calculation illustrates why replacing the entropy bound can matter:
independent fair bits have true TC zero but an entropy bound of log(2), while
two copied fair bits have both true TC and bound equal to log(2). Equal marginal
entropies do not imply equal safe parallelism. These are algebra checks, not
validation of a latent estimator or a new model-performance result.

## The next experiments

1. Freeze the successful confidence/Fisher ranking as an explicitly ineligible
   diagnostic and vary only count: EB, singleton, and the proposed latent batch
   interaction allowance. This isolates whether count estimation helps.
2. Separately validate confidence-free readiness against the existing cheap
   activation-only singleton control (7/20), before enlarging batches. Current
   new readiness signals are weaker on code despite arithmetic gains.
3. Validate candidate interaction estimates against small actual finite reveals,
   including groups with higher-order dependence. Put these probes in a small
   estimator audit, not at every generation step. Attention, Jacobians and
   cached normal-pass responses are cheap approximation candidates, not measured
   counterfactual effects. Backward/JVP compute is not free.
4. Evaluate each objective term by removal and matched count/scoring controls.
   Freeze settings before a broader run. All 164 HumanEval tasks have now been
   exposed to the research process; further tuning is exploratory.

Use existing results as evidence for or against particular estimators. Do not
reshape the theory to declare every successful heuristic optimal, or attribute
joint improvements to individual components without ablations.
