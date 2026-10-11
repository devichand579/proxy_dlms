# Reading the original Schedule_tracing objective

7 October 2026. Read directly from the newly attached 17-page PDF:
`/root/.codex/attachments/89d7fda4-2e90-4766-bff0-e103bf961128/Schedule_tracing.pdf`.
This is an explanation and mathematical audit, not a new evaluated scheduler.

## The objective actually stated

Let C be the revealed positions, R the remaining masked positions, and S a batch
selected from R. Equation 17, page 5, is

    F(S) = lambda_1 sum_(s in S, k in R) I(X_s; X_k | X_C)
           + lambda_2 sum_(s in S, j in C) I(X_s; X_j)
           - lambda_3 sum_(i<j in S) I(X_i; X_j | X_C).

The three terms are future information, past compatibility, and within-batch
redundancy. Past compatibility is intended as a surrogate for small marginal
model error, not an independently derived error estimate. Future information
is intended to accelerate entropy resolution. Redundancy penalizes selecting
positions whose independent predictions destroy their joint dependence.

The exact starting point, Equation 4 on page 3, is

    KL(q_S || product_(i in S) p_i)
        = sum_(i in S) KL(q_i || p_i) + TC_q(X_S | X_C).

Replacing marginal error by predictability, replacing total correlation by a
sum of pairwise MI, and replacing MI by CST feature inner products are successive
assumptions. They are not consequences of the exact KL identity. The data
distribution q is not observable from a model's ordinary forward pass.

## What the stated natural weights imply for count

For fixed R and C, define the first two sums' singleton benefit as a_i. The
exact Equation-17 increment when adding i outside S is

    Delta_i(S) = a_i - lambda_3 sum_(j in S) I(X_i; X_j | X_C).

Page 6 proposes lambda_1=lambda_3=lambda and lambda_2=1-lambda. Substituting gives

    Delta_i(S) = lambda sum_(k in R\S) I(X_i; X_k | X_C)
                 + (1-lambda) sum_(j in C) I(X_i; X_j) >= 0.

Thus, for 0<=lambda<=1, S=R is always an optimum unless an external size limit is
imposed. With positive self-entropy and lambda>0, each addition is strictly
rewarded. Under a prescribed |S|<=k, the objective can choose positions but
generally fills the allowed count. It does not derive conservative independent
counts under its suggested coefficients.

For two perfectly correlated fair bits, no revealed context and lambda=0.5,
the literal Equation 17 gives F({1})=1 and F({1,2})=1.5 in bit units. It prefers
jointly revealing both, despite one bit of total-correlation error when they
are independently sampled from their correct marginal distributions.

## The convergence quantity and its pairwise approximation

The Section-3.6 identity itself is correct:

    I(X_S; X_R | X_C) = H(X_S | X_C), since S is contained in R.

It measures entropy removed from the original masked set, including removal of
the sources. It is not solely information provided to surviving positions.
That distinct target would be I(X_S; X_(R\S) | X_C).

Equation 16 is not justified merely by the disjoint-set pairwise approximation:
its sets overlap. For S={i}, its right side is H(X_i|X_C) plus all external
pairwise MI, whereas its left side is exactly H(X_i|X_C). This already disagrees
for two dependent tokens, where there are no higher-order interactions to omit.
The rate approximation needs a fresh derivation before being used for count.

## The quadratic is a separate feature-matching formulation

Equation 21, page 8, states

    F(v) = v^T M b - 1/2 v^T M v,
    b = lambda r + (1-lambda) z,

where r indicates masked positions, z indicates revealed positions, and v
indicates the selected batch. With M=AA^T, pages 12-13 imply

    u = A^T b,       selected_features = A^T v,
    F(v) = 1/2 ||u||^2 - 1/2 ||selected_features-u||^2.

The geometric interpretation is to select a batch whose aggregate features
match a weighted aggregate of future and past features. Selecting too few can
miss the target; repeatedly selecting the same features can overshoot it.

This identity is exact for the stated Gram quadratic. But Equation 21 is not
the same as Equation 17 under the proposed natural coefficients:

- Its pair penalty has coefficient one instead of lambda_3=lambda.
- It includes a self-cost of 1/2 sum_(i in S) M_ii; Equation 17's redundancy
  sum over i<j has no diagonal cost.
- Its single M is defined as conditional MI given X_C. Its entries to already
  conditioned context variables must be zero, so it cannot also reproduce the
  unconditional past-compatibility MI in Equation 17. A hybrid block matrix or
  separate past and future objects would be needed.

The correlated-bit example above gives F({1})=0.5 and F({1,2})=0 under Equation
21 with M equal to the all-ones matrix. Its preference is the opposite of
Equation 17 with the same lambda, demonstrating the mismatch concretely.

## Spectral ranking versus optimization

The exact singleton-addition increment of Equation 21 is

    Delta_i(S) = (Mb)_i - sum_(j in S) M_ij - 1/2 M_ii.

Greedily selecting the largest positive increment and stopping determines
positions and count from this quadratic, subject to a progress fallback and
explicit compute limits. It is a local solver, not a globally optimal binary
solver. Ranking by M^(1/2)b is a distinct heuristic, not the derived optimizer.

When M is PSD, the unconstrained continuous optimum includes v=b. Restricting
selection to masked positions, imposing binary values or adding a cardinality
constraint changes that optimization. A rotated eigenbasis does not remove
these coupled constraints. Negative eigenvalues invalidate the concave
unconstrained argument; CST Gram matrices themselves have no negative modes.

## Relation to the current research constraints

Section 4 proposes separately trained CST encoders and decoders. That is one
feature-extraction choice, not a mathematical requirement of the abstract
objective. Under the user's no-auxiliary-learning constraint, features must
come from the existing frozen model, its activations, or its gradients.
Their Gram matrix remains a geometry proxy unless its relationship to actual
token dependence and reveal effects is validated.

The right starting point for the next design is therefore the exact error
decomposition, an explicit choice between removal rate and external gain, and
a consistent variable-cardinality objective. Neither the current natural
Equation-17 weights nor square-root spectral ranking alone supplies the
desired independently computed count.
