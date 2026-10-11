# Pace: A Step-Budget Unmasking Schedule for LLaDA

Oct 11, 2026 · Online version: https://claude.ai/code/artifact/c70cef50-c891-4060-a1ac-82517e13519f

## Abstract

Pace, a one-line change to how many tokens a masked diffusion language model unmasks per step, solves 2–9× as many problems as the strongest existing schedule when the step budget is tight, at no extra cost per step. Masked diffusion models such as LLaDA write an answer in a fixed number of forward passes, and the scheduler decides how many and which masked tokens to reveal at each pass. The EB-Sampler and its Fisher-ranked variant (EB+Fisher) choose that count from an entropy bound that ignores the budget, so with 16–32 steps they leave answers unfinished.

Pace keeps EB+Fisher's ranking and count, and adds one floor: reveal at least (answer tokens still masked) / (steps left) each step. It has no tuned parameters and no learned parts. We justify it from a decomposition of the parallel-decoding error, in which EB's rule is a worst-case bound on token dependence and the step budget adds a progress term.

On LLaDA-8B Base, pace beats EB+Fisher at 16 steps on HumanEval (39 vs 17), GSM8K (96 vs 24) and MBPP (107 vs 12), matches EB+Fisher's 128-step HumanEval score at 64 steps, and ties at 128 steps. On LLaDA-1.5 it wins 8 of 10 benchmark × budget settings and ties the other two. More than twenty variants that change the ranking, spacing, schedule shape, remasking or end estimate do not improve on plain pace.

## 1. Introduction

The number of tokens revealed per step, not which tokens are revealed, decides the quality of a masked diffusion model under a tight step budget. Across all our experiments, changes to the ranking moved HumanEval scores by 0–8 problems, while getting the count right moved them by up to 22 on HumanEval and 95 on MBPP.

Masked diffusion language models (MDLMs) such as LLaDA start from a fully masked answer canvas and fill it in over T forward passes. Each pass predicts every masked position at once; the scheduler then commits some of those predictions. Fewer passes means faster generation, but each pass must commit more tokens, and tokens committed together are sampled as if independent. The scheduler therefore trades speed against the error from ignoring dependence between tokens.

The EB-Sampler handles this trade-off with an entropy bound: it keeps adding the most confident tokens until their summed entropy, minus the largest, exceeds a threshold γ. Our mentor's EB+Fisher adds a Fisher-information risk term to the ranking and is the strongest baseline in this codebase. Neither rule looks at the step budget. When T is small, EB commits too few tokens per step and runs out of steps with the answer half-written.

**Contributions.**

1. **Pace**, a budget-aware floor on EB's count: commit at least ⌈L\_rem / (T − t)⌉ tokens at step t, where L\_rem is the number of masked tokens before the estimated end of the answer. It costs nothing per step and has no tuned parameters.
2. **A theory** that explains why the floor is safe. The parallel-decoding error splits into per-token error plus the total correlation among the revealed tokens. EB's rule is the worst case of that correlation, and measured dependence on code is about 10× below it. The step budget enters as a progress term, which pace makes explicit.
3. **Experiments on two models and three benchmarks** (HumanEval, GSM8K, MBPP; LLaDA-8B Base and LLaDA-1.5) at 16–128 steps, with paired tests. Pace is best or tied in every setting, and halves the steps needed for EB+Fisher's best HumanEval score.
4. **A set of negative results** on more than twenty refinements (spacing, schedule shapes, remasking, blocks, latent-space extrapolation, step sharing, confident end estimates), plus a diagnostic showing pace works as a self-paced left-to-right schedule.

## 2. Background

Every scheduler here makes two choices per step: a **ranking** of masked positions (which to reveal first) and a **count** (how many to reveal). The baselines differ in both.

**Masked diffusion decoding.** The prompt is followed by a canvas of L mask tokens (512 for HumanEval, 256 for GSM8K and MBPP). At step t = 0 … T−1 the model returns a distribution p\_i over the vocabulary at every masked position i. With greedy decoding, each position's candidate is its top-1 token. The scheduler commits a set S\_t of candidates; committed tokens are never changed again.

**EB-Sampler.** Rank masked positions by confidence (top-1 probability). Add positions in that order while

```latex
\sum_{i \in S} H_i - \max_{i \in S} H_i \le \gamma
```

where H\_i is the entropy of p\_i and γ = 0.1. At least one token is committed per step. The count n\_EB(t) is whatever size S reaches.

**EB+Fisher (mentor's best).** The same count, but positions are ranked by confidence minus 0.15 × a temporal Fisher risk, which penalises positions whose predictions are still moving between steps. It is our main baseline.

**Mentor's gradient-covariance count.** A different count rule from the mentor's codebase: it grows S while a Fisher-dispersion estimate stays under a tolerance (0.5), up to 16 tokens per step. It is paired with the Fisher ranking.

**Mentor's information score.** A ranking that adds a probe-based information-gain term and an “incoming risk” term. It needs extra forward passes (about 2×), so we compare it at equal total passes.

**LLaDA-1.5 and end-of-text blocking.** LLaDA-1.5 is instruction-tuned and, on a plain completion prompt, tends to commit end-of-text at the start of the canvas. The LLaDA paper (Appendix B.4) blocks end-of-text tokens for such models; the mentor's harness does this automatically. We call this the end-of-text fix and report each method at its better setting.

## 3. Method: the pace rule

Pace commits EB+Fisher's set, then tops it up with the next-best answer tokens until the step has done its share of the remaining work:

```latex
|S_t| = \max\left( n_{EB}(t),\ \left\lceil \frac{L_{rem}(t)}{T - t} \right\rceil \right)
```

Here L\_rem(t) is the number of still-masked positions before the estimated end of the answer, and T − t is the number of steps left. When EB already keeps pace, the floor is inactive and pace is exactly EB+Fisher. When EB falls behind, the floor adds the fewest extra tokens that still finish on time.

**Estimating the answer end.** The canvas is longer than the answer, so L\_rem must exclude the padding after it. For each problem, at each step, we read a “tail” sequence: the committed token where a position is committed, and the model's top-1 prediction where it is masked. The answer end is the earliest of:

- the first end-of-text (EOS or EOT) token in the tail;
- the start of the first run of 8 whitespace-only tokens in the tail (LLaDA pads code answers with blank lines rather than EOS);
- the end of the canvas.

The estimate is recomputed from scratch at every step and costs no extra pass.

**One step of pace** (per problem, after the forward pass):

1. Rank all masked positions by the EB+Fisher score (confidence − 0.15 × Fisher risk).
2. Take EB's prefix of that ranking (n\_EB tokens, at least 1).
3. Estimate the answer end and count L\_rem.
4. need = ⌈L\_rem / (T − t)⌉. If EB's prefix holds fewer than `need` answer tokens, add the next-ranked masked positions before the end until it does.
5. Commit the chosen positions' top-1 tokens.

**Cost.** Pace uses one forward pass per step, like EB. The extra work is a sort that EB already does and a scan of the tail, and measured wall-clock time per step is identical (Section 6.4).

**What pace does not change.** The ranking, the EB threshold γ = 0.1 and the Fisher weight 0.15 are the mentor's settings, unchanged. The run length 8 is the only new constant, and 4 or 16 give the same scores within 1 problem (Section 7).

## 4. Theory

EB's rule is a worst-case bound on token dependence that ignores the step budget; pace is what you get when you account for both the real dependence and the budget.

**4.1 Error from revealing a set at once.** Let q be the true conditional distribution of the masked tokens given the committed context x\_C, and p\_i the model's marginal at position i. Committing a set S in one step samples from the product of marginals. The KL divergence of that product from the truth splits exactly into two parts:

```latex
KL\Big(q(x_S \mid x_C)\ \Big\|\ \prod_{i \in S} p_i\Big) = \sum_{i \in S} KL(q_i \| p_i) + TC_q(X_S \mid x_C)
```

The first term is per-token model error, which no scheduler can remove. The second is the total correlation TC among the revealed tokens: the error from treating dependent tokens as independent. Only the second depends on which tokens are revealed together.

**4.2 EB is the worst case.** Total correlation is bounded by the entropies:

```latex
TC(X_S) \le \sum_{i \in S} H_i - \max_{i \in S} H_i
```

with equality only when one token fully determines the rest. EB's stopping rule is this bound set to γ. So EB is always safe, but it assumes the tokens it reveals are as dependent as they could possibly be.

**4.3 Real dependence is about 10× smaller.** We measured pairwise dependence directly: D\_ij = how much token j's prediction changes when token i is committed first (one extra pass per candidate). On HumanEval, the median largest D\_ij among the top candidates was about 0.01, against EB's budget of 0.1. A sampler that used D\_ij at every step to decide the count never beat a matched control and cost 2× the passes (Exp 3). So D\_ij is evidence that EB over-counts dependence, not a useful per-step signal.

**4.4 The step budget adds a progress term.** A scheduler with a budget is really solving

```latex
\max_S\ \pi |S| \quad \text{s.t.} \quad \sum_{i \in S} KL(q_i \| p_i) + TC(X_S) \le \gamma
```

where π is a Lagrange multiplier on “finish in T steps”. EB sets π = 0 and stops at the bound. With a budget, π grows when the work left exceeds the steps left. Pace takes the simplest form of this: when the even share L\_rem / (T − t) exceeds EB's count, relax the dependence bound just enough to meet it. Section 4.3 says this relaxation is cheap, because the true TC is far below the bound.

**4.5 Why pace is safe at high budgets.** With many steps, EB's count already exceeds the even share at almost every step, so the floor rarely fires and pace reduces to EB+Fisher. A fixed loosened bound (EB with γ/α) cannot do this: it reveals extra tokens even when the budget does not need them, and with α = 0.05 it loses 6 problems at 64 steps and 9 at 128 (Exp 4–5).

**4.6 Pace tracks the writing frontier.** The answer-end estimate is not a forecast of the final length. It hugs the text already written: at 16 steps, the estimate is 34% of the final length a quarter of the way through, 48% halfway and 74% at three quarters (Exp 21). So pace never spreads work over the whole answer. It keeps a moving frontier, commits an even share of what lies behind it, and saves a large share for the last step (median 13.5 tokens, mean 33).

This makes pace a self-paced left-to-right schedule. It explains three results:

- Front-loading the work hurts (Exp 9, “down”: 31 vs 39): early steps have the least context.
- A confidence-thresholded end estimate breaks decoding (0/20): the model is only sure where the answer ends near text it has already written.
- Pace's gain is largest where the baseline stops early. On LLaDA-1.5 GSM8K, blocking end-of-text already prevents early stopping, and pace's lead shrinks to a non-significant 11–13 problems.

## 5. Experimental setup

Every run uses the same model weights, prompts, greedy decoding and seed, so score differences come only from the scheduler.

| Item | Setting |
| --- | --- |
| Models | LLaDA-8B Base; LLaDA-1.5 (instruction-tuned) |
| HumanEval | all 164 problems, pass@1, 512-token canvas |
| GSM8K | test problems 0–399, 4-shot, 256-token canvas, exact match on the final number |
| MBPP | test rows 0–499, 3-shot, 256-token canvas, pass@1 on the given tests |
| Step budgets | 16, 32, 64, 128 forward passes per problem |
| Decoding | greedy (temperature 0), seed 1234, batch 20 |
| Baselines | EB+Fisher (γ = 0.1, Fisher weight 0.15); mentor's count + Fisher; EB with loosened bound γ/α |
| Constraints | no learned heads; at most about 2× the passes of the base sampler |
| Statistics | paired wins / losses per problem, exact two-sided McNemar (binomial) test |

**Noise floor.** Changing only the batch size from 20 to 10 moves HumanEval scores by 2–3 problems, with no significant change (Exp 17). We treat differences of about ±3 problems as ties and rely on paired tests.

**Full sets only.** Early runs used a 40-problem HumanEval screen; it flattered a fixed loosened bound that then lost on the full set (Exp 4). All reported results use the full sets above.

**Code.** Our sampler is `das/das_sampler.py`; the mentor's baselines are called through his own functions and reproduce his outputs exactly. “Exp N” refers to the numbered entries in the experiment log, where every run's settings and output folder are recorded.

## 6. Results

Pace beats EB+Fisher significantly in 15 of 22 model × benchmark × budget settings and ties the other 7; the gap is widest at 16 steps and closes by 128 steps on the base model.

### 6.1 Pace vs EB+Fisher

[Figure (interactive in the online version): % solved vs steps, pace vs EB+Fisher, 2 models × 3 benchmarks. Data: outputs/full, outputs/gsm8k, outputs/mbpp, outputs/llada15 · Exp 7, 8, 15, 18, 22]

On the base model, pace at 16 steps solves 2–9× as many problems as EB+Fisher (HumanEval 39 vs 17, GSM8K 96 vs 24, MBPP 107 vs 12). Pace at 64 steps matches EB+Fisher at 128 on HumanEval (55 vs 55), so it halves the steps for the same score. On LLaDA-1.5, the gap does not close at large budgets, partly because EB+Fisher needs the end-of-text fix there and the fix leaves much of the canvas unused.

**Paired tests** (pace wins / losses against EB+Fisher, per problem):

| Benchmark | Model | 16 steps | 32 steps | 64 steps | 128 steps |
| --- | --- | --- | --- | --- | --- |
| HumanEval | Base | 23 / 1 \*\*\* | 15 / 1 \*\* | 4 / 1 | 3 / 1 |
| GSM8K | Base | 79 / 7 \*\*\* | 104 / 15 \*\*\* | 55 / 23 \*\*\* | 12 / 17 |
| MBPP | Base | 98 / 3 \*\*\* | 75 / 16 \*\*\* | 23 / 12 | 6 / 1 |
| HumanEval | LLaDA-1.5 | 15 / 1 \*\*\* | 27 / 9 \*\* | 28 / 6 \*\*\* | 34 / 8 \*\*\* |
| GSM8K | LLaDA-1.5 | 58 / 47 | 72 / 59 | 91 / 29 \*\*\* | — |
| MBPP | LLaDA-1.5 | 124 / 3 \*\*\* | 78 / 12 \*\*\* | 54 / 25 \*\* | — |

\*\* p < .01, \*\*\* p < .001 (exact McNemar). Unstarred cells are ties.

### 6.2 Against the mentor's count

The mentor's gradient-covariance count beats EB+Fisher at 32 steps but fails at 16 and ties pace at 64 and above; pace is best or tied at every budget.

| Benchmark | Steps | Mentor's count + Fisher | Pace + Fisher | Pace wins / losses | p |
| --- | --- | --- | --- | --- | --- |
| HumanEval | 16 | 19 | **39** | 22 / 2 | < .001 |
| HumanEval | 32 | 47 | 48 | 6 / 5 | 1.0 |
| HumanEval | 64 | 51 | 55 | 6 / 2 | .29 |
| GSM8K | 16 | 46 | **96** | 59 / 9 | < .001 |
| GSM8K | 32 | 143 | **198** | 78 / 23 | < .001 |
| GSM8K | 64 | 255 | 248 | 27 / 34 | .44 |
| GSM8K | 128 | 270 | 261 | 16 / 25 | .21 |
| MBPP | 16 | 14 | **107** | 96 / 3 | < .001 |
| MBPP | 32 | 144 | **161** | 41 / 24 | .046 |
| MBPP | 64 | 192 | 195 | 21 / 18 | .75 |

At 16 steps the mentor's count is capped at 16 tokens per step, so it cannot finish a 512-token code canvas. It has no notion of the steps left, which is exactly what pace's floor adds.

### 6.3 Count matters more than score

On HumanEval at 32 steps, crossing three count rules with three ranking scores shows that the count decides most of the score (Exp 13). With confidence-only ranking, pace scores 46, the mentor's count 35 and EB 25. Adding Fisher to pace raises it only to 48. The mentor's information score, at about twice the passes, also gives 48.

### 6.4 Cost

Pace takes the same wall-clock time per step as EB+Fisher: 1.67 vs 1.66 s per problem at 32 steps, and 3.29 vs 3.30 s at 64 (HumanEval 164, one run at a time on an idle GPU). Its 2× saving in steps is therefore a 2× saving in time: 55/164 in 9 minutes, where EB+Fisher needs about 18.

## 7. Analysis: what does not help

No refinement we tried beats plain pace reliably; the best ones gain 3–5 problems at one budget and lose at another. All rows are LLaDA-8B Base on HumanEval 164 (plain pace: 39 at 16 steps, 48 at 32).

| Group | Variant | 16 steps | 32 steps |
| --- | --- | --- | --- |
| Parts of pace | No Fisher (confidence-only ranking) | 37 | 46 |
| Parts of pace | End = run of 4 whitespace tokens | 39 | 47 |
| Parts of pace | End = run of 16 whitespace tokens | 38 | 47 |
| How extras are picked | Skip extras within 2 positions of each other | 44 | 44 |
| How extras are picked | Skip extras within 4 positions | 39 | 47 |
| How extras are picked | Extras left to right, not by rank | 31 | 41 |
| How extras are picked | Rank by top-1 minus top-2 margin | 35 | 44 |
| Schedule shape | Fewer tokens early, more late | 37 | 49 |
| Schedule shape | More tokens early, fewer late | 31 | 47 |
| Schedule shape | Semi-autoregressive 32-token blocks | 42 | 47 |
| Schedule shape | Two-sided: fewer than EB when ahead | 39 | 47 |
| Repair | Remask committed tokens if p < 0.3 or 0.5 | 39 | 48 |
| Latent space | Extrapolate logits half a step ahead | 43 | 50 |
| Latent space | Extrapolate a full step ahead | 34 | 47 |
| Latent space | Soft-token feedback (10%) | 31 | 43 |

**What the ablations say.**

- **Ranking matters, but only a little.** Without Fisher, pace loses 2 problems at each budget. Taking extras left to right instead of by rank loses 7–8.
- **The end estimate is robust.** Pad-run lengths of 4, 8 and 16 give the same score within 1 problem.
- **Front-loading hurts.** Doing more work early (31 vs 39) fits the frontier-tracking view in Section 4.6.
- **Repair does not fire.** Remasking flipped no outcome. The model's own probability flags random corrupted tokens well (AUROC 0.92), but real errors are plausible tokens it is happy to keep.

**Half-step extrapolation was the one lead, and it did not hold.** It scored 43 vs 39 at 16 steps (10 wins / 6 losses, p = .45). Across three benchmarks and six budgets it won three and lost three: GSM8K 105 vs 96 at 16 steps but 186 vs 198 at 32; MBPP 111 vs 107 and 160 vs 161; HumanEval 44 vs 42 at 24 steps and 53 vs 55 at 64.

**Other ideas, dropped after a 20-problem check.** Skipping near-duplicate hidden states among the extras (14/14/13/13 vs plain pace 14), ranking by a logit-lens score from a middle layer (5–12 vs 14), and pooling the step budget across problems in proportion to answer length (12–14 vs 14).

**The mentor's information score costs more than it gives.** At 32 steps it ties Fisher ranking under pace (48 vs 48) but uses about 64 passes. Spending those passes as steps instead gives 55: pace + Fisher at 64 steps beats pace + information at 64 passes by 8 wins to 1 (p = .04).

## 8. Limitations and future work

Pace's gains are limited to tight budgets, and its answer-end estimate depends on how the model marks the end of an answer.

- **No gain once the budget is generous.** At 128 steps on the base model, pace ties EB+Fisher on every benchmark (HumanEval 57 vs 55, GSM8K 261 vs 266, MBPP 199 vs 194). On the base model it speeds up generation but does not raise the best achievable score. (On LLaDA-1.5 the gap persists at 128 steps, 66 vs 40, but there the baseline is handicapped by the end-of-text fix.)
- **The end estimate relies on end markers.** It needs end-of-text or a run of blank lines. With end-of-text blocked (the LLaDA-1.5 fix), pace loses that signal and falls to EB+Fisher's level, which is why we run pace on LLaDA-1.5 without the fix. A model that pads with something else would need its own pad rule.
- **A late crunch.** Because the end estimate trails the written text, the last step often commits many tokens at once (at 16 steps, 46 of 164 HumanEval problems commit 32 or more). A one-pass forecast of the final length could spread this out, but our confidence-based attempt broke decoding.
- **Scope.** Two models from one family, three benchmarks, greedy decoding only. We have not tested sampling with temperature, longer canvases, or other masked diffusion models such as Dream.
- **Single runs.** Greedy decoding is deterministic, so each number is one run; paired tests give the uncertainty per comparison, but there is no seed spread.

**Future work.** A cheap length forecast that does not collapse decoding (for example from the prompt alone); pace with temperature sampling; and an end rule learned once per model from a few unconstrained generations.

## 9. Conclusion

Under a fixed step budget, the scheduler should set its count from the budget. EB's entropy bound is a worst case of token dependence that is about 10× looser than measured dependence, and it ignores how many steps remain. Pace keeps EB+Fisher and adds a single floor — the masked answer tokens left, divided by the steps left — at no extra cost per step.

That floor is enough to beat EB+Fisher by wide margins at 16–32 steps on HumanEval, GSM8K and MBPP, to reach EB+Fisher's 128-step HumanEval score in 64 steps, and to carry over to LLaDA-1.5. It never loses significantly to EB+Fisher or to the mentor's count at any budget. Every refinement we tried left it unchanged or worse, which suggests that, for a self-paced left-to-right writer like LLaDA, getting the count right is most of what a training-free scheduler can do.
