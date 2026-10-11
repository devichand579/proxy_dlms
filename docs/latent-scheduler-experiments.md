# Latent scheduling: separate count and ranking

## Current research constraints

The subsequent body-progress pilot completed 12 valid HumanEval-20 arms at the
128-step cap. Independent Fisher-dispersion counts retained 14/20 and identical
programs, while mean visible-body commitment advanced from step 48.1 to 38.0.
Exact replay at step 48 passed 13/20 versus singleton's 9/20. This count uses
probability-weighted frozen readout geometry and retains confidence/Fisher
ranking; it is a diagnostic, not a wholly confidence-free scheduler. Corrected
observed-value gradient scoring passed 0/20 with both singleton and adaptive
counts. EOS and fixed-frontier controls did not improve final Fisher scores.
Protocol, partial-backward costs, and limitations:
`docs/latent-body-progress.md`; results:
`outputs/latent_body_progress_report/README.md`.

The October 8 error-allowance study is complete: 33 arithmetic settings on two
seeds and 13 HumanEval-20 settings. Independent Fisher-geometry counts matched
14/20 under the confidence/Fisher diagnostic ranking, with no EB count and one
forward per step; their decoded programs were identical to singleton. The best
strict probability-free activation-only arm stayed at 7/20. Protocol and
limitations: `docs/latent-error-allowance-experiments.md`; complete results:
`outputs/latent_allowance_report/README.md`. Probability weights in gradient
geometry are explicitly diagnostic, not classified as strict probability-free.

The October 7 clarification requires both ranking and count to come from
continuous activations, latent properties, gradients or measured latent reveal
responses. Vocabulary confidence, token entropy, probability margins and EB
count control are excluded from the new scheduler. The frozen decoder still
supplies token values. The earlier probability-based results below are benchmark
history, not methods satisfying this stricter constraint.

The new one-forward three-term study is complete: 40 arithmetic configurations
across two seeds and 15 HumanEval-20 configurations. Proposal-gradient curvature
with a small information weight scores 81.25% / 79.17% on arithmetic, but 0/20
on HumanEval. The best new adaptive one-pass variant scores 4/20, below native
EB's 13/20. Its scorer and count use only internal activations, existing decoder
gradient directions, and a Gaussian spectral surrogate; no trained head,
confidence rule, EB count, backward pass, or extra forward is used. Details:
`docs/latent-three-term-objective.md` and
`outputs/latent_objective_report/README.md`. These are exploratory results on
reused tasks, not a robust improvement.

Earlier activation-only prototypes and their small completed experiments are in
`docs/latent-residual-scheduler.md`, `docs/latent-flow-scheduler.md` and
`outputs/latent_only_ablation/README.md`. Those earlier methods did not beat EB
across both arithmetic seeds. Tests verify exact
spectral set objectives, count behavior and confidence-rescaling invariance.

Auxiliary
trained policies and readout heads are excluded. Earlier trained-head experiments
below are retained only as historical records and are ineligible for the current
comparison. The latest preference is few model passes; the new objective study
uses one normal forward per step. Historical extra-pass studies retain their
actual work accounting alongside the 128 outer-step cap.

The full 164-problem evaluation is complete at 128 outer steps and a 512-position
canvas. All settings were frozen before this run; normal batches contain 20
canonical problems (last batch 4). Each method exactly reproduced all 20 earlier
completions on the tuning subset. The remaining 144 problems were not used in
those scheduler searches.

| Method | Full HumanEval | Remaining 144 | Forward evaluations/problem |
|---|---:|---:|---:|
| Native EB | 47/164 (28.66%) | 34/144 (23.61%) | 128 |
| EB + target-specific temporal Fisher | 55/164 (33.54%) | 41/144 (28.47%) | 128 |
| Information + risk + incoming sensitivity | **56/164 (34.15%)** | **42/144 (29.17%)** | 250.34 |

Fisher has nine wins and one loss against EB; information has ten wins and one
loss. Information adds only HumanEval/35 relative to Fisher, at 1.96 times the
forward work. Fisher is the preferred inexpensive improvement; information has
the highest measured accuracy. Both improvements persist outside the original
20-problem subset. These are single greedy completions, not multi-seed estimates.
Five of Fisher's nine wins repair invalid syntax; four had valid syntax under EB
but failed tests. Overall invalid-syntax outputs decrease from 25 with EB to 14
with either intrinsic method. Most remaining failures have valid syntax.

The methods commit on average 313.88 / 342.57 / 347.65 of the 512 positions,
respectively. The EB entropy rule is unchanged; a different ranking can change
the accepted prefix length. Counts include unused output tails and do not
directly measure quality. All batch artifacts passed task-alignment, outer-step
limit and forward-work checks. Evaluator controls passed 164/164 canonical
solutions and 0/164 deliberately failing solutions; controls never enter model
generation. No method uses training or backward passes.

Artifacts: `outputs/latent_full_humaneval164/README.md`, `analysis.json`,
`per_task.csv`, `comparison.png`, and per-method batch checkpoints. Reproduce
generation with `python -m schedulerlens.latent_full_bench`, then analysis with
`python -m schedulerlens.latent_full_report`.

On the original tuning subset, Fisher and information both score **14/20**,
compared with native EB's **13/20**. Those scores should not substitute for the
full evaluation above.

The input-gradient audit compared 56 candidate actions with actual singleton
reveals on 24 small arithmetic histories. Pearson correlation with external
entropy reduction was -0.0044, Spearman correlation was 0.189 and sign agreement
was 50%. This local derivative poorly predicted finite MASK-to-token effects.
Five wrong reveals reduced entropy while worsening reference likelihood, so
entropy reduction alone is not evidence of correctness. Reference answers enter
only this diagnostic, never scheduler decisions.

The next intrinsic experiment measures actual reveal-and-recompute effects at
surviving masked positions, excluding the revealed source. It combines measured
outgoing entropy reduction with temporal readiness and EB's count rule. Optional
inverse-density weights come from output-Fisher latent geometry without fitting
a readout; these are redundancy heuristics, not MI or correctness estimates.

The first completed direct-reveal run packed four cohorts into a batch of 80.
All used 128 outer steps and a 512-position canvas:

| Setting | Passed / 20 | Forward evaluations per sample |
| --- | ---: | ---: |
| Fisher / EB control | 14 | 128.0 |
| Actual reveal, uniform surviving targets | 14 | 190.0 |
| Actual reveal, latent-density target weights | 13 | 190.4 |
| Actual reveal, latent-density candidate and target weights | 13 | 190.0 |

The uniform-target method gained HumanEval/14 and lost HumanEval/5 relative to
its packed control. It has no aggregate advantage. Its standalone verification
evaluated unchanged probe contexts at the same batch shape as revealed contexts
to remove batch-shape numerical effects from measured entropy changes. This
verified run scored **14/20 with exactly the same passed tasks as temporal
Fisher**, using **252.8 forward evaluations per sample** and no backwards. The
packed task-14 gain did not reproduce; both batch shape and the probe-control
protocol changed, so this comparison does not isolate which change caused it.
It does not justify replacing the cheaper 128-forward Fisher method.
Artifacts: `outputs/latent_finite80/`. These probe gains are outgoing entropy
changes under the model, not information-gain or correctness guarantees.
Standalone artifact: `outputs/latent_finite_verified20/finite_all_targets.json`.

The same intrinsic finite-reveal variants were then checked on both arithmetic
seeds, with 32 canvases, 384 answer slots and a 12-outer-step cap per seed. Each
probe included a batch-matched unchanged-context control:

| Setting | Seed 0 answer accuracy | Seed 1 answer accuracy | Forward evaluations per canvas |
| --- | ---: | ---: | ---: |
| Native EB | 71.09% | 69.79% | 12 |
| Temporal Fisher / EB | 73.96% | 70.05% | 12 |
| Actual reveal, uniform targets | 74.74% | 71.61% | 92.56 / 93.56 |
| Actual reveal, balanced targets | 75.00% | 71.88% | 92.56 / 93.69 |
| Actual reveal, balanced candidates and targets | 74.74% | 73.44% | 92.56 / 93.94 |

Every slot was filled and no backwards were used. Balancing candidates and
targets gained 14 correct answers over EB on each seed (+3.65 percentage points).
It also increased complete-worksheet accuracy from 34.38% to 38.28% on seed 0
and from 38.28% to 39.06% on seed 1. Target-only balancing gave stronger worksheet
accuracy (40.63% / 39.84%), so the best choice depends on the reported metric.
The canvases reuse component worksheets; these are correlated exploratory
checks, not a generalization result. Artifacts:
`outputs/latent_finite_arithmetic_seed0/` and `outputs/latent_finite_arithmetic_seed1/`.

Reproduce the standalone HumanEval probe with
`python -m schedulerlens.latent_finite_bench --configuration finite_all_targets
--matched-probe-control --out outputs/finite_reproduction20` (one shell line).
For arithmetic, use `python -m schedulerlens.latent_small_bench --variants
eb_sampler,eb_target_temporal,finite_all_targets,finite_balanced_targets,finite_balanced_sources_targets
--gain-beta 0.5 --seed 0 --out outputs/finite_reproduction_seed0`, then repeat
with seed 1 and a separate output directory. Use the repository's `strace` env.

The decision is a batch of positions S. Its size answers **how many**, and its
members answer **which**. A quality controller should aggregate commitment risk
and within-batch dependence; a position policy should prefer reliable actions
that help the remaining prediction problem. The quantities need distinct
estimates and joint calibration.

## Intrinsic multi-value information probes

`schedulers/latent_information.py` probes the two highest-probability values of
each shortlisted source. The source weights are their renormalized probabilities;
the omitted source mass is recorded. At each remaining target, a fixed partition
contains its current top-128 vocabulary entries and an OTHER category holding all
remaining probability. Branches use the same partition and matched unchanged
contexts. Coarsening can hide changes within OTHER, but does not discard its mass.

For source weights rho, branch conditionals k and unchanged posterior p, compute:

```
mixture = sum_y rho[y] * k[y]
information = sum_y rho[y] * KL(k[y] || mixture)
consistency_defect = KL(mixture || p)
```

The sum equals expected intervention KL on this partition. The information term
is MI in the constructed two-value model joint; it is not true-data MI. The
revealed source's own target contribution is excluded. Ranking variants add
outgoing information, optionally penalize low-confidence propagated effects,
and optionally defer targets with high incoming information or consistency
change. Temporal Fisher readiness remains active on every step, and EB still
controls the selected prefix's entropy budget. No reference answers, code tests
or auxiliary learned weights enter these computations.

The initial HumanEval ablation uses four candidates every 16 steps, with all
branch and control sequences counted. Run with `python -m
schedulerlens.latent_finite_bench --scheduler information --candidates 4 --out
outputs/latent_information80` on one shell line. The Fisher control incurs no
branch work. The completed packed run scored 14/20 for every setting, with
exactly the same passed tasks as the Fisher control. Information-only,
risk-adjusted and incoming-protected settings used 252.0, 253.6 and 253.6
forwards per sample respectively, versus 128 for Fisher. No backwards were used.

Across the measured probes, the top-two source values retained 86.75% of mass,
while target top-128 entries retained 68.72%; OTHER held the remaining target
mass. Aggregate mixture-consistency KL was about 2.58 times aggregate local MI.
This supports separating the two diagnostics, but neither helped task accuracy
in this run. Artifacts: `outputs/latent_information80/`.

The incoming-protected information method was subsequently verified in a
standalone batch of 20 with four candidates and probes every 16 steps. It scored
**14/20** at 128 outer steps, using **253.6 forward evaluations per sample** and
no backwards. It solved exactly the same tasks as standalone temporal Fisher,
which uses 128 forwards. This confirms the score but does not show a HumanEval
advantage for the extra probes. Artifact:
`outputs/latent_information_readiness_verified20/response_information_readiness.json`.

The same incoming-protected method was then run in a standalone batch of 20
with probes at every step. It scored **13/20**, using **2172 forward evaluations
per sample**, compared with 14/20 and 253.6 forwards for probes every 16 steps.
No backwards or policy training were used. More frequent measurements did not
improve HumanEval accuracy, so the expensive cadence is not promoted.
Artifact: `outputs/latent_information_every_step20/response_information_readiness.json`.

All token-specific scheduler quantities are computed online: current probability
and entropy, the previous normal-pass hidden state, target-specific temporal
Fisher movement, and measured model responses to two source values. Normalization
uses current candidates/targets. The model weights remain frozen and sampling
runs under `torch.no_grad()`. There is no optimizer, auxiliary trained head or
offline-fitted risk/value estimate in this method. Coefficients and probe budgets
are fixed hyperparameters compared across exploratory runs; the method is not
parameter-free. Code-test outcomes do not enter a generation run.

Multi-value probes were also checked on both arithmetic seeds, using a branch
batch size of 256 and four candidates at every outer step:

| Setting | Seed 0 answer accuracy | Seed 1 answer accuracy | Seed 0 / 1 worksheet accuracy | Forward evaluations per canvas |
| --- | ---: | ---: | ---: | ---: |
| Native EB | 71.09% | 69.79% | 34.38% / 38.28% | 12 / 12 |
| Temporal Fisher / EB | 73.96% | 70.05% | 39.06% / 35.16% | 12 / 12 |
| Information only | 74.48% | 69.53% | 44.53% / 32.03% | 174.25 / 177.13 |
| Information + propagated risk | 73.96% | 75.00% | 41.41% / 46.09% | 176.13 / 177.00 |
| Information + risk + incoming sensitivity | **77.08%** | **78.91%** | **45.31% / 53.13%** | 175.63 / 177.13 |

All slots completed within 12 outer steps; no backwards or trained heads were
used. The incoming-protected method added 23 and 35 correct answers over EB on
the two seeds. This is the strongest observed answer accuracy across these
two small studies so far, at substantially greater sequence-evaluation cost.
Information gain alone did not reliably help: it lost one answer versus EB on
seed 1. Component worksheets are reused within each study, so these remain
correlated exploratory ablations. Artifacts:
`outputs/latent_information_arithmetic_seed0/` and
`outputs/latent_information_arithmetic_seed1/`.

Expanding incoming-protected probes to all 12 remaining answer positions gave
75.26% / 77.60% answer accuracy and 40.63% / 49.22% complete-worksheet accuracy
on seeds 0 / 1, below the four-candidate method on both metrics. Cost increased
to 316.75 / 314.50 forwards per canvas. These runs used a 512-sequence probe cap,
versus 256 for four candidates, so this exploratory comparison also changes
batch numerics and is not a clean causal estimate of candidate count alone.
Top-two source retained mass fell from 86.29% to 79.79% in the seed-0 histories;
target retained mass stayed above 99.98%. Candidate histories differ, so this is
a diagnostic association, not proof of the cause of the quality change.
Four candidates remain the best observed arithmetic setting. Artifacts:
`outputs/latent_information_fullpool_seed0/` and
`outputs/latent_information_fullpool_seed1/`.

Current validation: 51 unit checks pass, including model-joint information
identities, zero external response for independent positions, source exclusion,
no-probe recovery of Fisher control, committed-prefix boundary behavior, frozen
parameter gradient state and exact sequence-work accounting. No further
HumanEval improvement was verified by these experiments.

## Measured-support correction and count ablation

The two probed source values are not the whole source posterior. Let r be their
full-posterior retained mass. A new ranking ablation multiplies measured
information and observed intervention KL by r before aggregation. The term
`r * I(Y;Z | Y in probed support)` is the measured-support contribution to a
full-source joint's MI; unobserved source conditionals remain unknown. This
avoids assigning the entire unit of source probability to the shortlisted
values. It is not an estimate of the missing information or true-data MI.

The separate count experiment keeps the latent/Fisher ranking fixed and probes
the argmax value at each source. For two proposed positions, define directed
conflict as the positive drop in the other proposal's log probability after
revealing this one. The prefix controller uses `-log p(proposal)` for probed
positions, EB entropy for unprobed positions, and symmetric pair-conflict costs.
It accepts prefixes while `sum(cost) - max(cost) + sum(pair conflict) <= 0.1`.
The singleton override is retained. This is a heuristic risk controller, not
EB's entropy bound or a TC upper bound. Between probe steps it uses EB.

A five-cohort HumanEval run compares Fisher control, existing information
ranking, support-weighted ranking, count-only change, and the combined change.
Normal batch size is 100; probe batches reach 320. No trained policy or readout
is involved. Completed scored artifacts are stored at
`outputs/latent_support_count100/`.

The completed five-cohort run gave:

| Setting | Passed / 20 | Forward evaluations per sample | Extra reveals at count-probe steps per sample |
| --- | ---: | ---: | ---: |
| Fisher control | 13 | 128.0 | 0 |
| Existing information ranking | 13 | 253.6 | 0 |
| Support-weighted ranking | 13 | 253.6 | 0 |
| Count-only change | 12 | 253.6 | 7.85 |
| Combined weighting and count | 13 | 252.8 | 8.15 |

No variant gained a task over its packed Fisher control; count-only lost
HumanEval/15. Its output used `range(n)` instead of the required inclusive
`range(n + 1)`. The count-only controller changed 4.7 probe decisions per sample,
and the combined controller changed 3.9. Both increased counts; neither selected
fewer positions than EB at these probes. Lower remaining-mask counts therefore
did not imply better code. The batch-100 control scores 13/20 rather than the
verified standalone Fisher score of 14/20, consistent with earlier batch-shape
effects. These trials do not replace the standalone 14/20 result.

The two-seed arithmetic follow-up used 32 canvases, 12 outer steps and the same
256-sequence probe cap for all information/count variants:

| Setting | Seed 0 / 1 answer accuracy | Seed 0 / 1 worksheet accuracy | Seed 0 / 1 forwards per canvas |
| --- | ---: | ---: | ---: |
| EB | 71.09% / 69.79% | 34.38% / 38.28% | 12 / 12 |
| Fisher + EB | 73.96% / 70.05% | 39.06% / 35.16% | 12 / 12 |
| Existing information + risk + incoming | 77.08% / 78.91% | 45.31% / 53.13% | 175.63 / 177.13 |
| Support-weighted ranking | 72.14% / 73.18% | 39.06% / 39.84% | 176.88 / 176.63 |
| Count-only change | 73.96% / 70.05% | 39.06% / 35.16% | 172.63 / 177.00 |
| Combined weighting and count | 72.40% / 72.14% | 39.84% / 39.06% | 170.38 / 173.88 |

All answer slots were filled. Mass weighting has a principled interpretation as
a measured-support information contribution, but it worsened the ranking on
both seeds here. Neither the count-only nor combined change improves the current
best arithmetic result. Keep the unweighted incoming-protected method for these
small studies and Fisher/EB for HumanEval. Artifacts:
`outputs/latent_support_count_arithmetic_seed0/` and
`outputs/latent_support_count_arithmetic_seed1/`. Current checks: 54 pass.

On arithmetic the count-only controller changed 9 / 4 decisions across all 32
canvases on seeds 0 / 1, adding 9 / 4 reveals at those steps. The combined
controller changed 17 / 10 decisions and added 18 / 11 reveals. Thus the count
ablation activated, but its influence was limited in these small canvases.

## Conditioning-evidence ranking

`schedulers/latent_conditioning.py` measures how the original observed context
supports each current argmax proposal. A paired probe evaluates the current
canvas and a copy where only initially visible, attended positions are masked.
Previously generated anchors remain visible in both. This works for code prompts
and for the arithmetic scaffold without using references or a text classifier.
Both probe halves use the same batch shape to isolate context removal.

```
support_i = log p(proposal_i | original context, generated anchors)
          - log p(proposal_i | original context masked, generated anchors)
ranking_i = log confidence_i - temporal_Fisher_penalty_i
          + beta * clamp(support_i / mean(abs(support)), -4, 4)
```

The support is a pointwise frozen-model likelihood ratio, not calibrated token
correctness or true-data MI. Proposed token values still come from the normal
conditional model pass. EB controls the selected prefix with its original full
entropy budget. No guided-logit sampling or trained policy is introduced.

The HumanEval ablation compares beta=0, 0.25, 0.5 and 1 in a normal batch of 80.
Every active step adds one unchanged control and one context-removal sequence
evaluation per sample: at most 384 total forwards versus the control's 128.
Four new invariant checks cover no-probe recovery, independent-target evidence,
fixed-context preservation, generated-anchor preservation and work accounting.
All 58 checks pass. The completed packed HumanEval run scored 14/20 for Fisher,
12/20 for beta=0.25, 11/20 for beta=0.5 and 7/20 for beta=1. Active methods used
340.4, 337.4 and 348.6 forwards per sample respectively. Beta=0.25 gained
HumanEval/6 and /14 but lost /5, /8, /9 and /12; it has no aggregate advantage.
No conditioning variant is promoted over the verified Fisher result.
Artifacts: `outputs/latent_conditioning80/`.

The arithmetic follow-up completed all slots on both seeds. Beta=0.25 gave
70.83% / 73.44%, beta=0.5 gave 68.23% / 72.66%, and beta=1 gave 68.23% / 75.00%
answer accuracy on seeds 0 / 1. Each used roughly 35 forwards per canvas.
None improved both seeds over EB, and none beat incoming-protected information
ranking's 77.08% / 78.91%. Artifacts:
`outputs/latent_conditioning_arithmetic_seed0/` and
`outputs/latent_conditioning_arithmetic_seed1/`.

## Within-forward depth signals

`schedulers/latent_depth.py` compares the final head-input representation with
the representation four transformer layers earlier. The earlier raw state is
normalized once with the checkpoint's existing final norm. No lens or readout
is fitted. For a hidden-state difference d and output-head weight W, compute
the current top-128 output-weighted variance of Wd, plus signed support for the
actual current argmax relative to that weighted mean. The implementation handles
ties using the actual proposal rather than assuming top-k's first entry matches
argmax.

The ablation tests deferring large depth movement, rewarding signed support,
and their combination on top of temporal Fisher. All use EB's unchanged entropy
prefix rule and greedy conditional token values. These are internal-computation
diagnostics, not calibrated probabilities of error or measured token dependence.
Intermediate-layer states are not trained as output posteriors.

The initial run uses four cohorts in a normal batch of 80, with 128 outer steps
and no extra forward/backward evaluations. It retains hidden states, increasing
activation memory. Four invariant checks cover zero-weight recovery, zero motion,
common-logit invariance, deterministic tied proposals, prompt preservation and
work accounting. All 62 checks pass. The completed HumanEval run scored 14/20
for the Fisher control, 12/20 for depth movement penalty, 9/20 for signed support,
and 6/20 for their combination. Every setting used 128 forwards and no backwards.
No depth setting gained a task over the control. The movement penalty lost
HumanEval/9 and /16. Artifacts: `outputs/latent_depth80/`.

The arithmetic follow-up used 32 canvases and a 12-outer-step cap on each seed:

| Setting | Seed 0 / 1 answer accuracy | Seed 0 / 1 worksheet accuracy | Forwards per canvas |
| --- | ---: | ---: | ---: |
| EB | 71.09% / 69.79% | 34.38% / 38.28% | 12 |
| Temporal Fisher / EB | 73.96% / 70.05% | 39.06% / 35.16% | 12 |
| Depth movement penalty | 72.66% / 74.48% | 39.84% / 42.97% | 12 |
| Signed depth support | 64.84% / 62.24% | 21.88% / 28.13% | 12 |
| Depth support + movement penalty | 69.79% / 74.74% | 27.34% / 39.84% | 12 |

All slots filled. Depth movement added 6 and 18 correct answers over EB on the
two seeds, plus 7 and 6 complete worksheets, at no extra sequence evaluations.
It trails temporal Fisher on seed 0 and beats it on seed 1. It also trails the
more expensive incoming-protected information method on both seeds. Retain this
as a promising inexpensive arithmetic signal, without promoting it for HumanEval
or claiming generalization from correlated reused worksheets. Scored artifacts:
`outputs/latent_depth_arithmetic_seed0/` and `outputs/latent_depth_arithmetic_seed1/`.

## Provided-context reconstruction audit

`schedulerlens/context_reconstruction_audit.py` tests a separate intrinsic
semantic-consistency diagnostic. It reuses generated code from standalone
Fisher, sparse incoming-protected information, every-step information, and the
packed conditioning-support beta=0.25 run. Identical bodies per task are
deduplicated. The frozen model sees a code body and predicts masked tokens of
the original supplied prompt. Targets are only prompt tokens; canonical
solutions and hidden tests are never model inputs or scoring targets.

Two fixed schemes remove the whole prompt or one of four interleaved subsets.
The unchanged prompt tokens remain visible for the latter. Each score is mean
log probability of the supplied tokens at masked, attended positions. Selection
uses this score only. Stored code-test outcomes are consulted afterward for the
diagnostic report; no head or policy is trained.

There were 42 distinct candidates covering the first 20 tasks, with a correct
candidate available for 16 tasks. Whole-prompt reconstruction selected 13/20
correct, while interleaved reconstruction selected 11/20. Correct candidates
ranked above incorrect candidates in 7/12 and 5/12 within-task comparisons
respectively. These results do not justify an online semantic-risk controller
based on this score. This is a retrospective audit, not a new 128-step sampler
result or an established ensemble performance result.

The audit used 210 sequence evaluations in batches up to 64 and no backwards.
Three checks verify that scored labels are actually masked, independent targets
cannot distinguish bodies, and selection does not use correctness labels.
Artifact: `outputs/context_reconstruction_audit/audit.json`.

## Combined depth movement and incoming sensitivity

The depth movement penalty was combined with the existing incoming-protected
information ranking. It is captured during the normal forward, with no extra
passes beyond the information probes. EB still controls count. Two fixed depth
weights, 0.05 and 0.15, were compared against zero on both arithmetic seeds:

| Setting | Seed 0 / 1 answer accuracy | Seed 0 / 1 worksheet accuracy | Seed 0 / 1 forwards per canvas |
| --- | ---: | ---: | ---: |
| Information + risk + incoming control | 77.08% / 78.91% | 45.31% / 53.13% | 175.63 / 177.13 |
| Add depth weight 0.05 | 77.34% / 78.13% | 45.31% / 50.78% | 176.50 / 176.50 |
| Add depth weight 0.15 | 76.30% / 74.22% | 45.31% / 45.31% | 175.63 / 174.75 |

All slots filled. The lighter setting gained one answer on seed 0 but lost three
on seed 1, so it does not improve the current best across these studies. The
larger setting worsens both seeds. Combining individually useful signals is not
automatically beneficial. Artifacts:
`outputs/latent_information_depth_arithmetic_seed0/` and
`outputs/latent_information_depth_arithmetic_seed1/`. All 66 invariant checks
pass, including preservation of prompt and accurate work accounting with depth
capture and discrete probes together. The corresponding three-cohort HumanEval
run completed in a batch of 60: control 13/20, depth weight 0.05 13/20, depth
weight 0.15 11/20. Forward costs were 253.6, 253.6 and 253.2 per sample, with no
backwards. No combined setting beat its packed control. This batch's control
scores lower than the verified standalone incoming-protected method's 14/20,
so the packed comparison does not replace that result. Keep the original
incoming-protected method for arithmetic and Fisher/EB for HumanEval.
Artifacts: `outputs/latent_information_depth60/`.

## Attention reads from unresolved positions

`schedulers/latent_attention.py` captures the actual post-RoPE queries and keys
passed to LLaDA's last-block attention operation. The original operation is
called unchanged, and capture wrappers are removed even on error. Attention
probabilities are reconstructed in query chunks with the same scale and padding
bias. It does not retain the entire attention matrix.

For query i, the readiness proxy is mean-head attention to masked keys j != i,
optionally weighted by `1 - current_argmax_probability_j`. Observed input and
already committed positions have zero uncertainty weight. Self attention is
excluded. This estimates reads from unresolved representations, not causal
dependence, true MI or probability of a wrong commitment. Masked representations
may already contain useful inferred information, so the uncertainty-weighted
variant is tested separately from uniform masked-key mass.

The penalty is normalized over current masked targets and added to the existing
temporal Fisher ranking. EB retains its entropy-count rule and greedy conditional
token proposals. No policy/readout is fitted and there are no additional model
passes. Four checks cover self exclusion, known-key exclusion, padding handling,
query-chunk consistency, unchanged attention outputs, cleanup on errors, and
zero-weight recovery of Fisher. The batch-80 HumanEval ablation compares zero,
uniform masked-read weight 0.15, and uncertainty-weighted penalties 0.15 / 0.3.
The completed packed HumanEval run scored 14/20 in every setting, with exactly
the same passed tasks as its Fisher control. All used 128 forwards and no
backwards. The attention penalty changed reveal trajectories and tail-mask
counts but did not improve task accuracy. Artifacts: `outputs/latent_attention80/`.

The two-seed arithmetic follow-up completed all slots within 12 outer steps:

| Setting | Seed 0 / 1 answer accuracy | Seed 0 / 1 worksheet accuracy | Forwards per canvas |
| --- | ---: | ---: | ---: |
| EB | 71.09% / 69.79% | 34.38% / 38.28% | 12 |
| Temporal Fisher / EB | 73.96% / 70.05% | 39.06% / 35.16% | 12 |
| Uniform masked-read weight 0.15 | 75.00% / 66.41% | 39.84% / 31.25% | 12 |
| Uncertainty-weighted read penalty 0.15 | 75.26% / 64.58% | 40.63% / 30.47% | 12 |
| Uncertainty-weighted read penalty 0.3 | 72.92% / 64.58% | 39.06% / 28.91% | 12 |

The apparent seed-0 gain reversed on seed 1. Neither weighting choice supports
promoting attention reads as a reliable commitment-risk estimate. No head or
policy training was used; all 70 checks pass. Artifacts:
`outputs/latent_attention_arithmetic_seed0/` and
`outputs/latent_attention_arithmetic_seed1/`.

## Split-batch commitment consistency

`schedulers/latent_consistency.py` starts with the same confidence/temporal-Fisher
ranking and EB entropy prefix. When the proposed batch has more than one member,
it reveals alternating halves in two speculative copies. Each withheld half is
checked for changes to its original greedy proposals. Two unchanged controls
are evaluated in the same model batch as those speculative copies. If a control
already differs from the normal-pass proposal, that numerical mismatch is
recorded separately and is not called a reveal conflict.

One controller removes conflicting positions while preserving the proposed
order. The other reduces the whole batch to its first singleton if any conflict
occurs. Both retain a singleton fallback when all positions conflict. Token
values still come from the original normal pass; probe predictions never replace
them. This can reduce EB's count and select a stable subset, but cannot increase
count, remask committed tokens or alter the input prompt. It is a model
consistency heuristic, not a correctness certificate or TC bound.

Each active check costs four sequence evaluations: two controls and two reveal
contexts. Checks run at every outer step, only when count is greater than one.
Five checks cover stable ordering, singleton fallback, no-check recovery,
unchanged independent batches, conflicting-commitment serialization, numerical
batch-shape separation, prompt preservation and exact work accounting. All 75
checks pass. The completed batch-60 HumanEval comparison at
`outputs/latent_consistency60/` tied the Fisher control at 14/20 with exactly the
same passed tasks, costing 390.4 forwards/problem versus 128. Only one conflict
and count reduction occurred across all 20 problems. Arithmetic seeds 0/1 also
tied Fisher at 73.96% / 70.05%, costing 12.75 / 12.375 forwards/canvas rather
than 12. These checks have not demonstrated an accuracy benefit.

The next ablation restricts target relevance to positions up to the first
already committed EOS, optionally restricting the source pool too. This boundary
comes from the frozen model's generated end marker, with no fitted code-content
classifier. Before EOS is committed, all positions remain relevant. The target
restriction addresses scoring changes in unused output tails; it is not a true
conditional-MI derivation. Run using the information scheduler and
`--information-plan prefix`.

The completed prefix ablation used normal batches of 80 and probe batches up to
320. Every setting scored 14/20 with exactly the same passed tasks as Fisher.
The mean recorded target relevance was 1.0: no committed EOS boundary excluded
targets at probe times. This is an inactive relevance control, not evidence that
meaningful prefix targeting cannot help. Different allocation/batch shapes also
changed tail mask counts, so those are not used as accuracy evidence. Peak
allocated memory was 96.25 GiB. Artifacts: `outputs/latent_information_prefix80/`.

## Protocol and baseline

- Frozen local LLaDA-8B Base, first 20 canonical HumanEval test rows.
- Raw code-completion prompts, zero shot, greedy token values.
- Main comparison: 512 generation positions, a 128-forward outer-step cap.
- Decode through the first EOS, then apply the installed HumanEval stop filters.
- Existing `schedulers/eb_sampler.py`, gamma=0.1 and confidence ranking, is the
  primary baseline. Its ranking and entropy selection logic is unchanged.
- The optional initial-canvas arguments added to EB support arithmetic infilling.
- Code tests run only after generation. Canonical solutions were used solely
  to verify that the evaluator passes 20 positive controls and rejects 20
  deliberately failing programs; they are never scheduler inputs.

Experiments reuse these 20 problems for exploratory tuning. They do not measure
unseen-task generalization. Completed code can pass while unused canvas positions
remain masked; completion counts are recorded separately.

## Implemented signals

`schedulers/latent.py` includes raw and Fisher diversity, input-gradient rate
estimates with source-excluded shared targets, temporal Fisher movement, signed
latent support, and discrete response probes. The probes distinguish centered
response variance from the mean response and use target-specific probabilities
inside a reported truncated vocabulary. Expensive probe variants have not
consistently improved results; sensitivity is not a correctness certificate.

The most useful cheap signal so far is target-specific temporal Fisher movement:

```
r_i = 0.5 * Var_{v ~ truncated p_i}[ W_v (h_i_current - h_i_previous) ]
ranking_i = log p_i(proposed_token) - beta * min(r_i / mean(r), 4)
```

The hidden state is captured at the actual input to the output head, after final
normalization. This saves all-layer captures and does not store full past logits.
The Fisher is a local approximation under a renormalized top-128 target
distribution, not full KL or a calibrated probability of error.

## Count/rank factorial experiment

`schedulers/latent_control.py` implements four policy cells:

| Cell | Count decision | Position decision |
| --- | --- | --- |
| eb_eb | Existing EB entropy-prefix rule | Confidence |
| eb_latent | Count computed from EB's confidence-ranked prefix | Exactly that many positions by latent readiness |
| latent_eb | Latent risk/coupling budget | Confidence |
| latent_latent | Latent risk/coupling budget on proposed positions | Latent readiness |

The count **rule** is held fixed in the ranking-only cell. Realized counts can
change later because earlier position choices change subsequent contexts. The
runner records every per-prompt count to expose this feedback.

The count prototype combines:

1. A confidence-derived error proxy, adjusted by incoming movement per previous
   actual reveal count and an exponential moving scale.
2. A nonnegative coupling screen from Fisher-projected, observed latent responses.
3. A batch risk budget and a recorded singleton override when no prefix fits.
4. A singleton bootstrap when temporal evidence is missing.

The screen uses squared correlations, including opposed responses. It is a
co-response proxy from the previous actual update, **not** a measured pairwise
MI or a TC upper bound. Common causes can create co-responses. Discrete branches
and higher-order controls are needed before stronger dependence claims.
The confidence proxy has not been independently calibrated as task-error risk.

## Results currently measured

The native EB baseline passes 13/20 at 512 positions. Target-specific temporal
Fisher ranking with EB's entropy controller and beta=0.15 passes 14/20 in a
standalone batch of 20, with 128 forwards and no backward passes per sample.
Shared-Fisher stability passes 12/20. Fixed linear reveal quotas with confidence
pass 3/20; this demonstrates how consequential the count decision is here.

The original factorial run packed eight settings into a batch of 160:

| Count/rank cell | Passed / 20 |
| --- | ---: |
| EB / confidence | 13 |
| EB / latent readiness | 14 |
| Latent budget / confidence | 14 |
| Latent budget / latent readiness | 13 |

All three tested latent budgets (0.015, 0.03, 0.06) give the same aggregate
count-only and combined scores, despite different reveal rates. The original
run used the positive-correlation version of the coupling screen; its exact
source is archived with its results. The current version also penalizes opposed
responses, with an invariant test for that case.

The large-batch EB control changes two individual outcomes relative to the
native batch-20 baseline, while retaining 13 total passes. Standalone checks
are therefore necessary before attributing a one-problem difference to a method.
The corrected count-only controller, rechecked alone at batch size 20 with
risk budget 0.03, passes **13/20**, matching native EB. Its earlier 14/20 packed
result is not a verified improvement. The target-specific Fisher/EB variant's
14/20 standalone result remains the best verified 512-position result so far.
In the packed factorial, the combination underperforms either individual change.
The standalone count check does not establish a count-only improvement. The
evidence does not justify adding independently tuned penalties or claiming that
the new count rule beats EB.

On the 32-canvas arithmetic benchmark (four worksheets and 12 answer slots per
canvas, a 12-step cap), native EB gives 71.09% slot accuracy and 34.38% completely
correct worksheets. Target-specific Fisher readiness gives 73.96% and 39.06%;
the EB gradient variant gives 74.22% and 42.97%, with additional backward work.
The expensive response-readiness probes give lower accuracy in this experiment.
The fresh count/rank arithmetic comparison gives 71.09% slots / 34.38% worksheets
for EB, 71.61% / 35.94% for latent count with confidence ranking, and 73.96% /
39.06% for latent count plus latent readiness. All 384 slots are completed in
these 12-step comparisons. The combined method helps this small domain while
failing to improve the main HumanEval comparison; transfer is unproven.

## Reproduce and inspect

With the `strace` environment active, from the repository root:

```bash
TRACING_ENABLED=false python -m schedulerlens.latent_bench --variants eb_sampler,eb_target_temporal --samples 20 --steps 128 --batch-size 20
TRACING_ENABLED=false python -m schedulerlens.latent_factorial_bench --samples 20 --budgets 0.015,0.03,0.06
TRACING_ENABLED=false python -m schedulerlens.latent_small_bench --steps 12 --samples 32 --variants eb_sampler,eb_target_temporal,control_count,control_both
python -m unittest discover -s tests -p 'test_*.py'
```

Native and first-stage results: `outputs/latent_humaneval20/`.
Factorial counts and per-problem outcomes: `outputs/latent_factorial/`.
Arithmetic results: `outputs/latent_arithmetic32_s12/`.
Standalone count verification: `outputs/latent_count_verified20/`.
Fresh arithmetic count comparison: `outputs/latent_arithmetic_count32/`.

The separate 256-position study is in `outputs/latent_humaneval20_g256/`. It is
not mixed into the main 512-position comparison: reducing the canvas changes
parallelism and improves even the confidence control substantially.

The subsequent experiments below add singleton-bootstrap and capped-EB controls,
separate supervised risk from progress, and evaluate amortized actual-reveal value.

## Learned risk and padding audit

A separate compact readout was trained with frozen LLaDA on 256 cached MBPP
training programs and checked on 64 held-out MBPP validation programs. Every
HumanEval entry-point name was excluded from both collections. No HumanEval
answers enter scheduler training or inference. The head uses 128 projected
hidden dimensions, 32 projected proposed-token embedding dimensions, and four
confidence/margin/end-marker features (about 11,000 trainable parameters).

The training histories contain gold-consistent partial masks, so calibration
there does not establish calibration on generated trajectories. Exact reference
agreement also penalizes alternative valid implementations. Padding labels
accept whitespace/end markers instead of demanding one particular EOS token.
The original strict-padding-label artifact is retained as a failed diagnostic;
`outputs/latent_risk_mbpp_v2/risk_head.pt` is the corrected artifact.

The held-out audit found that the full latent head mainly distinguishes code
content from harmless padding. On substantive code slots, confidence is better:
Brier 0.0858 versus 0.1496 for the latent head; lower is better. Its excellent
whole-canvas calibration therefore does not demonstrate better code-error
prediction. The scalar-only learned control is also worse on code-body slots.

In the first packed batch-80 experiment:

| Policy | Passed / 20 |
| --- | ---: |
| EB count / confidence ranking | 13 |
| EB count / learned-risk ranking | 0 |
| Learned-risk count / confidence ranking | 14 |
| Learned-risk count / learned-risk ranking | 1 |

The learned-count result needs standalone verification. It averages only 1.06
reveals per step and leaves 7,537 canvas positions masked across 20 prompts.
Risk-only ranking favors harmless filler over code progress and is rejected.
A singleton/capped-EB control is needed to isolate whether conservative pacing
explains the count-only result. Results: `outputs/learned_risk_factorial80/`.

A hybrid body gate is now being tested. Its output estimates substantive-body
membership; on predicted body positions it retains confidence-derived error,
and on inferred tail positions it treats non-whitespace content as risky.
This preserves the better body signal rather than replacing it with the
uncalibrated latent error forecast. Head selection uses held-out MBPP gate Brier,
and metadata now reports the saved best checkpoint.

A second count variant augments EB's prefix entropy with observed temporal
Fisher movement, independently of the position ranking. This remains an
instability surrogate, not a proven TC bound or calibrated functional-error
probability. The zero-penalty invariant recovers EB's trajectory exactly.

The packed hybrid-gate count study gives 14/20 at budgets 0.03, 0.1, and
0.3; the simultaneous EB control gives 13/20. All three gates pass the same
14 problems and recover HumanEval/5 relative to that packed control. Their
mean reveals per step are 1.88, 3.08, and 3.84, respectively; remaining masks
are 5,419, 2,348, and 417 across 20 prompts. This is promising across pacing
settings but still needs standalone verification. Source and readout artifacts
are archived in `outputs/body_gate_count80/`.

The gate can also estimate progress rather than risk: content-ranked policies
add `content_beta * log P(code_body)` to confidence/stability ranking. One
cell fixes k from EB's confidence prefix; another applies EB's entropy rule to
the actual reordered prefix. This is an inferred code-content prior, not an
outgoing information-gain estimate. Arithmetic is an explicit transfer check,
since this gate was trained on code.

The hybrid count controller at budget 0.3 was then checked alone on batch 20:
**13/20**, matching native EB. It left only 346 masks across 20 canvases,
versus EB's 3,322, while using the same 128 forwards. Thus its packed 14/20
result is not a verified pass-rate improvement, though it completes substantially
more of the canvas. Standalone results: `outputs/body_gate_count_verified20/`.

## Fresh standalone checks and controls

With the current numerically centered Fisher implementation, native EB again
passes 13/20 and EB/target-temporal beta=0.15 again passes 14/20. The current
results are in `outputs/latent_current_verified20/`. This confirms that the
older improvement survives the common-logit-shift/cancellation fix.

The code-content ranking policy was also checked at standalone batch 20:
**14/20**, using k from EB's confidence prefix and then choosing k positions
with confidence, temporal Fisher readiness (beta 0.15), and inferred code-body
membership (beta 0.25). It leaves 2,853 masks. This ties the best verified score
without adding model forwards; it requires a separately trained compact head.
Results: `outputs/content_verified20/`. The simpler singleton EB control gives
**13/20** alone and leaves 7,680 masks (`outputs/eb_singleton_verified20/`).
Its earlier 14/20 packed result does not survive standalone checking.

Held-out MBPP content-gate Brier is 0.0150 with hidden, token embedding and scalar
features, 0.0369 with token embedding and scalar features, and 0.0851 with only
scalar features. This establishes hidden-state value for content identification
on those histories; it does not establish a HumanEval pass-rate advantage over
the token-only gate, which needs a scheduler ablation.

## Actual-reveal value readout

A separate readout now targets actual reveal effects. For a greedy source action,
its supervised target is the change in mean reference log likelihood at **other
surviving masked code-body positions** after that action. Training uses disjoint
MBPP programs, never HumanEval answers. The source itself is excluded from the
objective; tail positions are excluded from supervised target quality.

Inference features use source hidden state, proposed-token embedding, confidence
features, and the mean hidden state of other currently masked positions. No
reference body boundary, target token or gold eligibility reaches the scheduler.
Training oversamples substantive source actions only to collect useful labels.
The baseline and intervention forwards have matched batch shapes to keep ROCm
numerical differences out of reveal-value labels. All baseline and branch
sequence evaluations are counted. LLaDA stays frozen.

This is an amortized reference-prediction utility, not mutual information or
functional correctness. Training histories remain gold-consistent masks, so
transfer to generated contexts is a hypothesis to check. Its two policy cells
separate a fixed EB count from applying the EB entropy rule to the value-ranked
prefix. Both run the compact head after the ordinary forward and require no
extra deployment branches.

The first reveal-value collector used 128 MBPP training and 32 validation
programs, two masking fractions, and eight source actions per history. Baseline
and actual-branch work totals 5,120 sequence evaluations in 167 seconds on the
local GPU, with baseline/branch batches of 128. The frozen-model labels and
head are stored in `outputs/latent_value_mbpp/`.

Held-out actual-gain prediction gives Pearson 0.462, Spearman 0.430, and MSE
0.213 versus 0.265 for a constant training-mean predictor. On substantive sources,
Spearman is 0.488. The scalar-only learned control gives MSE 0.211 and body-source
Spearman 0.468; the local hidden/token head gives body-source Spearman 0.528.
Thus the full hidden/context readout has **no demonstrated MSE advantage** over
a scalar head. These comparisons motivate task-level and feature ablations;
correlation alone does not establish scheduler benefit.

The packed reveal-value HumanEval study gives 14/20 for the cheap temporal
control, 14/20 with gain weight 0.1, and 12/20 with gain weight 0.3 under both
fixed-EB-k and value-ranked entropy control. No task-level gain is established
for the learned value head; stronger weighting is rejected. Results are archived
in `outputs/reveal_value_factorial80/`.

Two complete arithmetic seeds now compare the additional policies:

| Policy | Seed 0 slot accuracy | Fresh seed 1 slot accuracy |
| --- | ---: | ---: |
| EB | 71.09% | 69.79% |
| Temporal Fisher + EB | 73.96% | 70.05% |
| Gradient gain + EB | 74.22% | 64.58% |
| Hybrid risk count | 71.61% | 69.01% |
| Content ranking | 73.96% | 70.05% |
| Learned reveal-value ranking, beta 0.1 | 73.70% | 69.79% |
| Temporal-movement entropy/count, confidence rank | 71.09% | 69.01% |
| Temporal-movement entropy/count, latent rank | 73.96% | 70.05% |

Every variant completes all 384 slots at the same 12-step outer cap. Gradient
gain also uses additional forwards and backwards. Its seed-0 lead does not
transfer to seed 1, so it is no longer the preferred arithmetic method across
these two seeds. Temporal Fisher improves **slot** accuracy on both seeds, but
seed-1 complete-worksheet accuracy is 35.16%, below EB's 38.28%; there is no
unqualified claim that it improves all task metrics. Canvases reuse component
worksheets, so the slot observations are not independent samples. Raw results:
`outputs/latent_arithmetic_extended_seed0/` and
`outputs/latent_arithmetic_extended_seed1/`.

## Full-posterior movement controls

The next control computes temporal KL or Jensen-Shannon divergence using full
vocabulary logits from consecutive forwards. The GPU can retain past logits;
these policies add no model forwards and avoid the truncated quadratic Fisher
approximation. They still measure observed incoming prediction movement, not
causal source influence, correctness, or mutual information. Tests cover static
distributions, forbidden-mask support, gauge invariance, the categorical KL
definition, JS symmetry/bounds, and recovery of EB at zero ranking penalty.
The `fisher_entropy` policy reproduces the current best ranking/controller
combination as a direct comparison inside the factorial runner.

The live full-posterior/count study packs eight settings into batch 160:
EB; Fisher/EB; KL/EB with beta 0.05 and 0.15; JS/EB with beta 0.15;
hybrid risk count with **confidence-minus-Fisher** position ranking at budgets
0.1 and 0.3; and a delayed switch from the verified Fisher/EB policy to the
hybrid count rule at step 64. Results will be written under
`outputs/posterior_and_count160/`. This combination differs from the failed
risk-only ranking: risk controls count, while confidence/readiness selects
positions. No success is claimed before scoring and standalone verification.

The full-posterior/count batch-160 study completed:

| Policy | Passed / 20 | Remaining masks |
| --- | ---: | ---: |
| EB / confidence | 13 | 3,606 |
| Fisher ranking / EB entropy rule | 14 | 2,854 |
| Full KL ranking / EB, beta 0.05 | 12 | 2,618 |
| Full KL ranking / EB, beta 0.15 | 13 | 2,171 |
| Jensen-Shannon ranking / EB, beta 0.15 | 13 | 2,489 |
| Hybrid risk count + Fisher ranking, budget 0.1 | 13 | 2,228 |
| Hybrid risk count + Fisher ranking, budget 0.3 | 13 | 292 |
| Fisher/EB initially; hybrid count after step 64 | 14 | 816 |

Full-distribution movement is not better here than the truncated local Fisher
signal. More faithful measurement of movement does not guarantee better
readiness ranking. The delayed count change is promising for completion while
retaining pass rate, but needs a standalone batch-20 check. All policies use
128 forwards per sequence and no backward passes. The study occupied about
109 GiB while running, within the 192 GiB device's capacity.

A regenerated scoreboard now consolidates completed trials and keeps protocol
groups and verification status explicit:

```bash
python -m schedulerlens.scheduler_scoreboard
```

Report: `outputs/scheduler_scoreboard/README.md`; full settings and compute counts
are in its HumanEval and arithmetic CSVs. Pending jobs are not scored entries.

The delayed count change at step 64 has now been verified alone at batch 20:
**14/20**, with **820 masks** left, versus 3,238 for the current Fisher/EB
method and 3,322 for native EB. It retains the best pass count and reveals
2,418 more canvas positions than the Fisher/EB policy within the same
128-forward cap. This is a completion improvement, not a further task-score
gain; some extra reveals are unused tail content. It uses a trained compact
readout and a co-response count screen. Its standalone result is archived in
`outputs/late_count_verified20/`.

The consolidated outcome audit found that HumanEval/14 has been solved by the
fixed-quota gradient methods (7/20 overall at gain 0.5 and 5/20 with 64-position
blocks). Thus there is evidence of a potentially useful anchor-selection signal
that their aggressive count rule obscures. The EB gradient prototype's candidate
shortlist currently scales with k: four candidates at k=1, versus sixteen at k=4.
A fixed minimum shortlist can isolate candidate-pool restriction from the count
rule before discarding that signal.

On arithmetic seed 0, full KL and JS improve slot accuracy to 74.74% and
75.00%, respectively, versus EB's 71.09% and Fisher's 73.96%. On seed 1 both
give 69.53%, below EB's 69.79% and Fisher's 70.05%. The delayed hybrid gives
73.70% / 70.05%; all methods finish every slot. Thus full posterior movement
has no consistent advantage across these arithmetic seeds or on HumanEval.
Results: `outputs/latent_posterior_arithmetic_seed0/` and
`outputs/latent_posterior_arithmetic_seed1/`.

## Candidate-pool and incoming/outgoing gradient ablation

The gradient candidate pool can now have a fixed minimum independent of EB k.
The combined score adds source-excluded input-gradient reveal gain to
confidence-minus-target-Fisher readiness. EB's actual reordered-prefix entropy
rule still controls cardinality. Zero gain bonus recovers the verified Fisher
policy, with no backward work. The original gradient behavior remains the
default. Probe microbatches bound activation memory; compute is counted per
logical sample, including skipped probes and extra backward work.

The current five-setting batch-100 study compares the Fisher control, the
original EB gradient policy (adaptive 4*k pool), a minimum-16 pool without
incoming risk, and the minimum-16 incoming/outgoing combination at gain weights
0.25 and 0.5. Normal forwards run at batch 100; gradient probes use up to 20
sequences at a time. Outputs: `outputs/latent_gradient_pool100/`.
No performance claim is made before scoring and standalone confirmation.

The initial 40-sequence backward-probe attempt exhausted VRAM before the first
outer step completed (177.17 GiB allocated, 12.59 GiB reserved but unused).
The process exited, and the same study was restarted with probe microbatches
of 20 while retaining the 100-sequence normal forward batch. Native batch-20
gradient runs already peaked at about 138.7 GiB: backward activation storage
is the constraint here, rather than model-weight size. Failure metadata is
retained in `outputs/latent_gradient_pool100/failure_probe40.json`.
