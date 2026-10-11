# Small information-gain and spectral-scheduler experiments

This lab is for repeated mechanism tests before full benchmark sweeps. It leaves
production schedulers unchanged. `spectral.py` is absent from the checkout used
to build this lab; `raw_gram` is an explicitly labeled legacy-style geometry
control, not a reproduction of a historical run.

## What the Info-Gain paper actually tests

[Yang et al., Improving Sampling for Masked Diffusion Models via Information Gain](https://arxiv.org/html/2602.18176v3)
uses two motivating examples in Section 3.1: decimal factors with a binary
product, and arithmetic judgments with a Yes/No answer. Locally easy output
symbols can be chosen before useful upstream quantities. Figure 3 uses 100
simple arithmetic problems and a 64-token global window to track cumulative
entropy; Figure 5 varies temperatures on that set. These are failure cases and
empirical comparisons, not a general impossibility proof for discrete schedules.

The [authors' method description](https://github.com/yks23/Information-Gain-Sampler/blob/main/docs/method.md)
defines mean masked-state entropy and scores candidate actions by its decrease
minus the selected tokens' marginal entropy. A batched call with N hypotheses
still evaluates N sequences. This lab reports both calls and sequence work.

## Stage 1: seconds on CPU

Activate the existing `strace` environment, then from the repository root:

```bash
python -m unittest discover -s tests -p 'test_infogain_lab.py'
python -m schedulerlens.infogain_lab toy
python -m schedulerlens.infogain_lab sweep outputs/infogain_lab/toy.json --out outputs/infogain_lab/toy_sweep.csv
```

Exact six-bit distributions include independent variables, a noisy informative
anchor plus an easy independent token, and XOR/parity. Branches enumerate all
source values. Additional explicit diagnostics show that an incorrect reveal
can sharpen other predictions while worsening reference likelihood, and that
removing an independent uncertain source can decrease *mean state entropy*
without informing any surviving target.

The toy score's `reference_nll_gain` is expected true log-score improvement
under the enumerated coherent joint (equal to external marginal information).
These numbers are mathematical diagnostics, not learned-model accuracies.
Pairwise TC of the parity variables is zero, whereas their three-variable TC is
one bit. A pairwise controller must not be declared safe from these zeros.

## Stage 2: small LLaDA audit, then GPU-free sweeps

```bash
python -m schedulerlens.infogain_collect --samples 8 --seed 0 --microbatch 4 --out outputs/infogain_lab/arithmetic_seed0.json
python -m schedulerlens.infogain_lab sweep outputs/infogain_lab/arithmetic_seed0.json --alpha 0,0.1,0.25,0.5,1,2 --out outputs/infogain_lab/arithmetic_seed0_sweep.csv
```

The controlled worksheet has three single-token slots: intermediate sum, final
sum, and Yes/No judgment. The visible scaffold and two solved examples are held
fixed. We measure three common states: all slots masked, true intermediate sum
visible, and true answer visible. All selections commit the same argmax token,
use a singleton budget, and compare the same candidate positions. Histories
contain ground-truth *revealed context*, but references never enter scores.
This deliberately isolates position selection from token sampling and batch
size. It is not a reproduction of the paper or a free-form reasoning benchmark.

For eight problems, two token alternatives, and three histories, collection
uses 24 baseline evaluations plus 112 branch evaluations: **136 sequence
evaluations**. Changing `--microbatch` changes call count, not this work. The
collector loads the existing local model and records wall time, memory, model
configuration hash, code hash, input ids, and references. Existing caches are
protected against accidental overwrite. No datasets or models are downloaded;
no external services are written to.

Scores replayed from the cache:

| Arm | Selection signal |
| --- | --- |
| confidence / entropy / margin | Cheap singleton controls |
| paper_ig | Realized mean-state entropy decrease minus source entropy; equation diagnostic only |
| external_ig | Mean entropy reduction on identical surviving targets minus the same source entropy cost |
| external_gain | Realized entropy reduction on the identical surviving positions |
| expected_external_gain | Same target correction, averaged over truncated source alternatives |
| response_kl | Expected target KL after reveal; sensitivity, not correctness |
| response_fisher | Half the target-probability-weighted variance of finite logit changes; local approximation to KL |
| raw_gram | Singleton marginal gain of centered/normalized hidden Gram, with context weight 0.5 |

For gain/response/Gram arms, `alpha` subtracts source commitment surprise
`-log p(argmax)`. This is a cheap risk **proxy**, not a calibrated error estimate.
No risk/dependence/spectral optimality guarantee is claimed. Branch expectations
renormalize the top alternatives: omitted source mass is always reported.
The raw input to the output head is captured directly, rather than saving all
hidden layers. Full target-vocabulary probabilities are used, except MASK is
suppressed to prevent recommitting MASK. No arithmetic vocabulary constraint is
imposed on the model. EOS is not a stopping criterion in this fixed-state audit.

For each policy the CSV reports source token error, remaining reference log-score
improvement, regret against the best measured candidate on those same targets,
and omitted probability mass. Histories are averaged within a problem before
averaging problems. The best reference-scoring action is an **oracle diagnostic**
and never a deployed policy. Report source error alongside target utility:
a confidently wrong, influential source can look helpful to entropy alone.

The source-excluded target differs between singleton candidates. Regret is
therefore against this specific external-effect objective, not total action
correctness or final task score. Joint action utility and complete trajectories
are required before claiming end-to-end gains.

### First local smoke run

The seed-0, eight-problem collection completed on the local AMD GPU with
PyTorch 2.9.1+rocm6.4: 56 model calls, 136 sequence evaluations, 4.16 seconds
of collection after 6.36 seconds of model loading, and 15.26 GiB peak allocated
memory. These figures describe this tiny 106-token worksheet, not throughput
for longer generations. Cached score sweeps require no model or GPU.

On the six validation problems, mean source-error rates were 16.7% for
confidence, 33.3% for pure external entropy gain, and 22.2% for external gain
with commitment-surprise weight 0.25. These are descriptive smoke results,
averaged within problem across the three histories, with no significance
claim. Gain-only selection is not automatically safe. The two test problems
are too few to choose a winning scheduler, and must not become a repeatedly
tuned leaderboard. Keep the source error, target utility, and omitted mass
columns together when assessing a candidate.

## Repeated ablations and the next gates

Use the fixed seed-0 cache for quick engineering comparisons. Select weights
using `validation` rows only; freeze the choice before inspecting `test` rows.
The default eight examples contain six validation and two test problems,
balanced in Yes/No labels: this is a smoke test, not significant evidence.
Fresh problem seeds change the corpus; `make_problems` currently caps collection
at the finite number of unique one-digit triples. For stronger evaluation,
expand problem families and hold out entire families instead of tuning against
the same eight examples indefinitely.

Change one component at a time:

1. Mean-state gain versus source-excluded gain, keeping source cost fixed.
2. Source cost weight, with the gain definition fixed.
3. Argmax reveal versus expectation over 1/2/4 alternatives; inspect omitted mass.
4. Exact KL versus Fisher variance; measure approximation error on the same branches.
5. Raw geometry versus measured response features; condition on cheap controls.
6. In a follow-up batch experiment, k=1 versus k=2 with identical token proposals,
   then pairwise versus actual joint batch effects.

Only promote a signal if it improves held-out reference outcomes and avoids
wrong-anchor failures. Next add short 32–64-token completed trajectories and a
full Info-Gain baseline with a shared candidate pool, followed by the PCFG oracle
and broader tasks. Cache-based local results alone do not establish trajectory
quality. With an exact coherent joint and exact singleton sampling, the expected
sum of committed conditional entropies follows the chain rule and is order
invariant; scheduler failures require imperfect conditionals, approximate batch
sampling, constraints, or a different task objective.

New spectral features should represent directed reveal responses. Compression
rank and set-selection strategy can then be tested against the uncompressed
response baseline. The present lab collects exact effects and a raw-geometry
control; it does not yet implement an amortized response head, full response
spectrum, finite-horizon planner, or continuous background experiment service.
