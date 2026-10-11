# Unmasking Schedulers for LLaDA: Experiment Log

2026-10-09 · Online copy: https://claude.ai/code/artifact/c590cfe8-15cb-485f-a778-2183d6037679 · Write-up: https://claude.ai/code/artifact/c70cef50-c891-4060-a1ac-82517e13519f

## Summary

Our main method is the **pace rule**. It beats EB+Fisher (the mentor's best) by a wide margin when steps are scarce, and it is never worse than EB+Fisher at any budget we tested.

Pace keeps EB+Fisher's ranking and its EB count, then adds one floor: each step must unmask at least (answer tokens still masked) / (steps left). It has no tuned parameters and no learned heads. Like EB, it costs one forward pass per step.

| Benchmark | Steps | EB+Fisher | Pace | Wins / losses |
| --- | --- | --- | --- | --- |
| HumanEval (164) | 16 | 17 | **39** | 23 / 1 (p < .001) |
| HumanEval (164) | 32 | 34 | **48** | 15 / 1 (p = .001) |
| HumanEval (164) | 64 | 52 | **55** | 4 / 1 |
| HumanEval (164) | 128 | 55 | **57** | 3 / 1 |
| GSM8K (400) | 16 | 24 | **96** | 79 / 7 (p < .001) |
| GSM8K (400) | 32 | 109 | **198** | 104 / 15 (p < .001) |
| GSM8K (400) | 64 | 216 | **248** | 55 / 23 (p < .001) |
| GSM8K (400) | 128 | 266 | 261 | 12 / 17 (p = .46, tie) |
| MBPP (500) | 16 | 12 | **107** | 98 / 3 (p < .001) |
| MBPP (500) | 32 | 102 | **161** | 75 / 16 (p < .001) |
| MBPP (500) | 64 | 184 | **195** | 23 / 12 (p = .09) |
| MBPP (500) | 128 | 194 | **199** | 6 / 1 (p = .13) |

In short: on HumanEval, pace at 64 steps matches EB+Fisher at 128 steps, a 2× speed-up. At 128 steps both methods saturate and tie. The same pattern holds on all three benchmarks (Exp 15).

The mentor's count × score grid (Exp 13) confirms the choice. His gradient-covariance count ties pace only at 32 steps (47 vs 48) and loses at 16 (19 vs 39). On GSM8K it loses at 16 and 32 steps (46 vs 96, 143 vs 198) and ties at 64 (255 vs 248). His information score adds nothing at equal compute.

On MBPP the mentor's count shows the same pattern (14 / 144 / 192 vs pace 107 / 161 / 195 at 16 / 32 / 64 steps). On the second model, LLaDA-1.5, pace beats EB+Fisher in 8 of 10 benchmark × budget settings and ties the other 2 (Exp 18, 22).

## Setup

Every result uses the same model, greedy decoding and seed, so differences come only from the scheduler.

- **Model:** LLaDA-8B Base, greedy decoding (temperature 0), seed 1234, batch size 20.
- **HumanEval:** all 164 problems, pass@1, 512-token canvas.
- **GSM8K:** test problems 0–399, 4-shot prompt, 256-token canvas, exact match on the final number.
- **MBPP:** test rows 0–499, 3-shot prompt, 256-token canvas, pass@1 on the provided tests.
- **Step budgets:** 16, 32, 64 and 128 steps (forward passes per problem).
- **Baselines:**
  - EB-Sampler (γ = 0.1).
  - EB+Fisher (the mentor's best): ranks tokens by confidence minus 0.15 × temporal Fisher risk.
- **Constraints:**
  - No learned heads.
  - At most about 2× the passes of the base sampler.
  - The mentor's code in /root/Strace is untouched; our code is in /root/aditya/das.
- **Statistics:** paired win/loss counts per problem, with an exact McNemar test.
- **Noise floor:** changing only the batch size moves a score by 1–2 problems, so differences that small count as ties.
- **Screen vs full set:** early experiments used a 40-problem screen (HumanEval tasks 20–59). It turned out to be too easy and gave a misleading picture (see Exp 4). Every result since then uses the full set.

## Theory recap

The theory says EB is a worst-case bound on how much the tokens depend on each other. It also says a step budget adds a progress term to EB's objective. Pace is what you get when you take both seriously.

**1. Error from unmasking many tokens at once (Prop 1).** When we reveal a set S in one step, the error splits into two parts:
- the per-token model error;
- the total correlation TC among the tokens in S, i.e. how much they depend on each other.

    KL(q(x_S | x_C) ‖ Π_i p_i) = Σ_{i∈S} KL(q_i ‖ p_i) + TC_q(X_S | x_C)

**2. The per-step objective.** Maximise progress π|S| subject to (model error + TC) ≤ γ. The progress term π does not come from the KL identity. It encodes the step budget: it is a Lagrange multiplier on "finish in T steps".

**3. EB is a worst-case bound (Prop 2).**
- TC(X_S) ≤ Σ H_i − max H_i, with equality only when one token determines all the others.
- EB's rule "ΣH − maxH ≤ γ" is exactly our constraint with TC replaced by this worst case.
- So EB is always safe, but too cautious when the tokens are nearly independent.

**4. M, the dependence matrix (our replacement for CSTs).**
- M is measured as D_ij: how much token j's prediction changes if token i is committed first.
- It costs one extra forward pass per candidate.
- On code, the median largest D among the top candidates was about **0.01, against a budget of 0.1**.
- In other words, the real dependence is about 10× below what EB's bound assumes.

**5. How this gives pace.** Since EB over-counts dependence by about 10×, we can safely reveal more tokens per step than EB does. The step budget says how many more: at least L_rem / steps_left.

    |S_t| = max( n_EB(t), ⌈ L_rem / (T − t) ⌉ )

- When EB already keeps pace, the floor is inactive and pace is exactly EB+Fisher.
- When EB falls behind, the floor relaxes the dependence bound by the least amount that still finishes on time.

**Role of M in the final method.** Pace does not compute M at every step, which keeps its cost at 1×. M is the *evidence* that justifies relaxing EB's bound. Measuring it at every step never paid for its 2× cost (Exp 3).

## Experiments

Ten experiments, in order. The decisive finding: *how many* tokens to unmask per step matters far more than *which* tokens. The right count depends on the step budget.

### Exp 1. Fixed counts (40-problem screen)

- **Idea:** check whether the count matters at all, using one token per step and a fixed 4 per step.
- **Result:**
  - One per step: 19 at 64 steps and 22 at 128, against EB+Fisher's 21 and 21.
  - Fixed 4: 14/40.
  - Fixed 4, skipping tokens the lookahead marks as coupled: 16/40, at 10× the passes.
- **Impact:** dropped as methods; kept as sanity checks.
- **Why:**
  - With enough steps, every sensible rule lands at 20–22/40. So the count only matters when steps are scarce.
  - The dependence signal is real (+2) but weak for its cost.

### Exp 2. Loosened EB bound, "α" (screen)

- **Idea:** Prop 2 says EB's bound is worst-case. So try EB+Fisher with threshold γ/α, for α between 0.5 and 0.025.
- **Result:** at 16/32/64/128 steps, α = 0.05 scored 19/21/21/21, against EB+Fisher's 13/16/21/21.
- **Impact:** it looked like a winner, so we froze α = 0.05 and tested it on the full set (Exp 4).
- **Why:** at few steps EB unmasks too little and leaves most of the answer masked. Loosening the bound fixes that.

### Exp 3. Measured dependence M (DAS lookahead)

- **Idea:**
  - Replace EB's worst-case bound with the measured pairwise dependence D_ij (our M).
  - Add tokens greedily while α·(EB slack) + Σ D_ij ≤ γ.
  - Compare against a matched control: same α and ranking, no lookahead.
- **Result:**
  - It never beat its control. It tied, or lost 2–4 problems, at 2× the passes.
  - Median max-D ≈ 0.01 against a budget of 0.1, so the measurement almost never changes a decision.
- **Impact:** M stays in the paper as *evidence* that EB over-counts dependence by about 10×, not as a per-step component.
- **Why:** on code, the top-ranked candidates are almost independent of each other. A measurement that almost always says "no dependence" can't change decisions, so it costs passes and buys nothing.

### Exp 4. Frozen α = 0.05 on all 164 problems

- **Result:**
  - 39/45/46/46 at 16/32/64/128 steps, against EB+Fisher's 17/34/52/55.
  - Large wins at 16 and 32 steps.
  - Losses at 64 steps (6 wins / 12 losses) and 128 steps (4 / 13, p = .05).
- **Impact:** the 40-problem screen was misleading (too easy). Since then every run uses all 164 problems.
- **Why:** a fixed loose bound unmasks many tokens per step *even when the budget doesn't need it*. On harder problems those extra parallel commits cause errors, so accuracy caps at about 46.

### Exp 5. α grid on all 164 problems

- **Idea:** map the full trade-off with α ∈ {0.5, 0.25, 0.1, 0.05} × four budgets.
- **Result:** every fixed α trades high-step accuracy for low-step accuracy. The best α < 1 at each budget gives 39/47/51/54.
- **Impact:** showed that the right looseness depends on the budget. That led directly to pace, which sets it from the budget automatically.

HumanEval passed (of 164):

| Method | 16 | 32 | 64 | 128 |
| --- | --- | --- | --- | --- |
| EB+Fisher | 17 | 34 | 52 | 55 |
| α 0.5 | 21 | 42 | 51 | 54 |
| α 0.25 | 23 | 47 | 50 | 52 |
| α 0.1 | 32 | 44 | 45 | 46 |
| α 0.05 | 39 | 45 | 46 | 46 |
| **Pace** | **39** | **48** | **55** | **57** |

Each fixed α is a different point on the trade-off: the loose ones win at 16 steps and lose at 128. Pace tracks the best of them at every budget without tuning.

### Exp 6. Pace, first version (end = first EOS)

- **Result:** 3/164 at 16 steps.
- **Why:**
  - LLaDA pads the end of a code answer with newlines, not EOS.
  - So the estimated answer length was the whole 512-token canvas, which forced about 32 tokens per step.
- **Fix:** the answer ends at the first EOS/EOT, or at the first run of 8 whitespace-only tokens (committed or predicted).

### Exp 7. Pace (main method), HumanEval 164

- **Result:** 39/48/55/57, against 17/34/52/55 (table in the Summary). Pace is never worse.
- **Holdout check:** on the 124 problems outside the screen, pace vs EB+Fisher was 19 vs 4, 28 vs 18, 32 vs 31 and 34 vs 34.
- **Why it works:**
  - At low budgets, the floor forces exactly enough tokens per step to finish.
  - At high budgets, EB's own count already keeps pace. The floor rarely fires and pace behaves like EB+Fisher, so it avoids the high-step penalty of a fixed α.
  - Untuned, it matches or beats the best per-budget α at every budget.

### Exp 8. Pace on GSM8K (400 problems)

- **Result:**
  - 96 vs 24 at 16 steps, 198 vs 109 at 32, and 248 vs 216 at 64.
  - At 128 steps it ties: 261 vs 266, 12 wins / 17 losses, p = .46.
- **Impact:** the gain carries over from code to maths with no changes.
- **Why:** GSM8K answers are short relative to the 256-token canvas. At few steps EB+Fisher leaves the answer unfinished, and pace makes sure it finishes.

### Exp 9. Pace variants (how the extras are chosen and spread)

Plain pace scores 39 at 16 steps and 48 at 32.

| Variant | What changes | 16 steps | 32 steps |
| --- | --- | --- | --- |
| sp2 | skip extras within 2 positions of each other | 44 | 44 |
| sp4 | same, within 4 positions | 39 | 47 |
| up | fewer tokens early, more late | 37 | 49 |
| down | more tokens early, fewer late | 31 | 47 |
| left | extras taken left to right, not by readiness | 31 | 41 |

- **Impact:** none beats plain pace at both budgets. They become robustness ablations.
- **Why:**
  - Ranking extras by readiness (confidence − Fisher) matters: left to right loses 7–8.
  - Front-loading (down) hurts, because early steps have the least context.
  - Spacing helps at 16 steps but hurts at 32, so it is not a reliable gain.

### Exp 10. Ablations of pace's parts (16 / 32 steps)

| Variant | 16 steps | 32 steps |
| --- | --- | --- |
| Plain pace | 39 | 48 |
| No Fisher (confidence-only ranking) | 37 | 46 |
| Answer end = run of 4 whitespace tokens | 39 | 47 |
| Answer end = run of 16 whitespace tokens | 38 | 47 |

- **Impact:**
  - Fisher helps by about 2 at both budgets, so we keep it.
  - The end estimate barely changes with the run length, which is good for robustness.

## Lessons learned

1. **Count beats ranking.**
   - Every ranking change (Fisher, information gain, spacing, order) moved scores by 0–8.
   - Getting the count right under the budget moved them by up to 22 on HumanEval and 89 on GSM8K.
2. **EB's bound is far too cautious, but this only matters under a tight budget.**
   - Measured dependence is about 10× below the bound, so more tokens per step are safe.
   - With a generous budget, extra parallelism still costs a few problems. So loosen only when needed.
3. **The budget should set the count.** That is the progress term π in our objective. Pace makes it explicit with no tuned parameter.
4. **Measuring M at every step doesn't pay at 2× cost.** M is useful as evidence for the relaxation, not as an online component.
5. **Small screens mislead.** The 40-problem screen flattered the fixed α. We now trust only full-set, paired tests.
6. **Know the model's quirks.** LLaDA ends answers with whitespace padding, not EOS. Ignoring that broke pace's first version (3/164).

## New experiment ideas

Pace's gains are at 16–64 steps; at 128 steps it only ties. So the next ideas target two things:
- fixing the errors that the forced extras cause;
- using spare steps for more caution.

All of them stay within our theory and the 2× pass budget, with no learned parts.

| # | Idea | Link to theory | Cost | Expected effect | Priority |
| --- | --- | --- | --- | --- | --- |
| 1 | **Two-sided pace:** when ahead of schedule, take fewer tokens than EB (down to 1), so every step in the budget gets used | π is a Lagrange multiplier: when the budget is slack, spend it on lower dependence error | 1× | Gains at 64–128 steps, where pace now only ties | High |
| 2 | **Free post-hoc M + remasking:** the next pass already gives p(x_j \| new context) for tokens committed together. If a forced extra's own probability drops sharply, remask it | This is D_{j\|i} measured after the fact at no extra cost: M without lookahead | 1× | Repairs the errors the relaxation causes at 16–32 steps | High |
| 3 | **Lookahead only for forced extras:** measure D_ij only on steps where the pace floor fires, and skip coupled extras | M is applied only where EB's bound is actually relaxed, so the 2× budget goes where it matters | ≤ 2× | Better picks at 16–32 steps | Medium |
| 4 | **Margin ranking for extras:** choose extras by top-1 minus top-2 probability, not by confidence | Low margin = high per-token model error (the first term of Prop 1) | 1× | Small; cheap to check | Medium |
| 5 | **Smoother answer-length estimate:** use the running max of the estimated end across steps, so L_rem doesn't jump around | A more accurate L_rem gives a more accurate π | 1× | Fewer wasted or rushed steps | Medium |
| 6 | **Pace inside semi-autoregressive blocks:** split the canvas into blocks of 32–64 tokens, each with its own budget | Far-apart positions are less dependent, so most dependence is within a block | 1× | Possibly large on GSM8K, where LLaDA is known to do well with blocks | Medium |
| 7 | **Reserve the last few steps for checking:** remask and redo the lowest-confidence tokens | Same model-error term, spent at the end when the context is fullest | 1× | Small gains at high step counts | Low |

**Plan:**
- Run ideas 1 and 2 first, on HumanEval 164 at all four budgets, then on GSM8K.
- Adopt an idea only if it beats plain pace without losing at any budget (paired test, beyond the 1–2 problem noise floor).

**Status, 2026-10-09:**

- **Check behind idea 2.** LLaDA is trained only on masked positions, so it was unclear whether its output at a committed position means anything. Test on 20 GSM8K answers: we corrupted 15% of committed tokens with random ones. The model's probability for the token it was given has median 0.99 on clean tokens and 0.44 on corrupted ones (AUROC 0.92). So the free check carries real signal.
- **Ran on HumanEval 164** (jobs file: das/jobs_pace2.txt; results in Exp 11):
  - two-sided pace with cap 1× and 2× the pace (64/128 steps; cap 1× also at 16/32);
  - remasking at thresholds 0.3 and 0.5 (16/32 steps).

### Exp 11. Two-sided pace and free remasking (ideas 1 and 2): no gain

| Variant | 16 | 32 | 64 | 128 |
| --- | --- | --- | --- | --- |
| Plain pace | 39 | 48 | 55 | 57 |
| Two-sided, cap 1× pace | 39 | 47 | 54 | 57 |
| Two-sided, cap 2× pace | — | — | 54 | 56 |
| Remask if p < 0.3 | 39 | 48 | — | — |
| Remask if p < 0.5 | 39 | 48 | — | — |

- **Two-sided:** cutting tokens early only pushes them to later steps, where the floor forces them back in. Total work per step ends up the same, so accuracy doesn't move.
- **Remasking:** it fired only 1–3 times per problem and flipped no problem. The check catches random-token errors (AUROC 0.92), but real errors are plausible tokens the model is happy to copy.
- **Impact:** both dropped. They are negative results for the paper: the step budget, not when or whether to re-check, is what limits accuracy.

### Exp 12. Semi-autoregressive blocks and margin ranking (ideas 6 and 4): no reliable gain

| Variant | HumanEval 16 | HumanEval 32 | GSM8K 32 |
| --- | --- | --- | --- |
| Plain pace | 39 | 48 | 198 |
| Pace in 32-token blocks | 42 (4 / 1) | 47 (2 / 3) | 190 |
| Pace in 64-token blocks | 39 | 48 | — |
| Margin ranking | 35 (4 / 8) | 44 | — |
| EB+Fisher in 32-token blocks | 0 | 0 | 49 |
| EB+Fisher (no blocks) | 17 | 34 | 109 |

- **Blocks:** +3 at 16 steps, −1 at 32 and −8 on GSM8K. That is within noise or worse, so blocks are not adopted.
- **Margin ranking:** −4 at both budgets. Confidence ranking stays.
- **EB+Fisher with blocks collapses.** The mentor's sampler gives each block a fixed share of the steps and moves on with tokens still masked: 496 of 512 positions stay unfinished at 16 steps. This is the same "unfinished answer" failure pace fixes, so the fair baseline stays EB+Fisher without blocks.
- **Conclusion so far:** plain pace is robust. Every variant we tried (spacing, schedule shape, order, two-sided, remasking, blocks, margin) ties or loses.

**Blocks at larger budgets** (final numbers; blocks still do not help):

| Run | 16 steps | 32 | 64 | 128 |
|---|---|---|---|---|
| HumanEval: plain pace | 39 | 48 | 55 | 57 |
| HumanEval: pace in 32-token blocks | 42 | 47 | 53 | 54 |
| GSM8K: plain pace | 96 | 198 | 248 | 261 |
| GSM8K: pace in 32-token blocks | 102 | 190 | 239 | 266 |
| GSM8K: EB+Fisher in 32-token blocks | 5 | 49 | 168 | 267 |

Extra budgets for the figure: pace at 24 steps scored 42/164. The 48- and 96-step runs crashed (GPU out of memory with too many runs at once) and were dropped as not needed.

### Exp 13. Count × score grid (mentor's request), HumanEval 164 at 32 steps

**Idea.** The mentor's sweep (Codex session 01a10259, /root/Strace) tried several count rules and scoring rules, but only on 20 problems at 128 steps, where every count ties. Here every cell runs at 32 steps on all 164, where count actually matters. Mentor's methods are called through his own helper functions (`score='info'`, `base='allowance'` in das_sampler.py). A smoke test shows they reproduce his `generate_information` / `generate_allowance` outputs exactly. Settings: information score = 4 candidates, probe every 16 steps, weights 0.5 / 1.0 / 0.15. Gradient-covariance count = Fisher dispersion, tolerance 0.5, max 16.

| Count ↓ / Score → | Confidence | Confidence + Fisher | Information + risk + incoming (2× passes) |
|---|---|---|---|
| EB | 25 | 34 | 35 |
| Mentor's gradient-covariance count | 35 | 47 | — |
| Pace (EB base) | 46 | **48** | 48 |
| Pace on mentor's count | — | 45 | — |

**Paired tests (W/L, p):**

| Comparison | W/L | p |
|---|---|---|
| Mentor count + Fisher vs EB + Fisher | 14/1 | 0.001 |
| Mentor count + Fisher vs pace + Fisher | 5/6 | 1.0 (tie) |
| Pace on mentor's count vs pace | 2/5 | 0.45 |
| Pace + info (64 passes) vs pace + Fisher at 64 steps (55) | 1/8 | 0.04 (worse) |
| EB + info (64 passes) vs EB + Fisher at 64 steps (52) | 1/18 | <0.001 (worse) |
| Confidence only: pace (46) vs mentor's count (35) | 13/2 | 0.007 |
| Confidence only: pace (46) vs EB (25) | 22/1 | <0.001 |

**What it shows**

- **Count is the big lever at low budgets.** Both adaptive counts (pace, and the mentor's count) beat EB by 13–14 problems.
- **Pace is robust to the score.** With confidence only it scores 46, close to its 48 with Fisher.
- **The mentor's count depends on Fisher.** It drops from 47 to 35 without it.
- **Stacking the two counts does not help.**
- **The information score is a net loss at equal compute.** It changed 20 of the 164 pace completions but flipped no outcomes. Its extra passes are worth more as extra steps.

**Best count × best score = pace + Fisher, which is our existing method.**

**Follow-up: the mentor's count + Fisher across budgets** (HumanEval 164):

| Steps | EB + Fisher | Mentor's count + Fisher | Pace + Fisher | Pace vs mentor's count (W/L, p) |
|---|---|---|---|---|
| 16 | 17 | 19 | **39** | 22/2, p < 0.001 |
| 32 | 34 | 47 | **48** | 6/5, p = 1.0 |
| 64 | 52 | 51 | **55** | 6/2, p = 0.29 |

- **The mentor's count only helps in a narrow band around 32 steps.**
  - At 16 steps it is capped at 16 tokens per step, so it cannot finish the code. It scores 19, about EB's level.
  - At 64 steps it falls back to EB's level (51 vs 52).
- **Pace is best or tied at every budget.** Its floor scales with the remaining answer length and the steps left, so it adapts to any budget. The mentor's count has no notion of the step budget.

### Exp 14. Wall-clock timing: pace costs nothing extra per step

Pace and EB+Fisher take the same wall-clock time per step, so pace's 2× saving in forward passes is a 2× saving in time. Runs were on HumanEval 164, one at a time on an idle GPU (outputs/timing).

| Steps | Pace | EB+Fisher | Score (pace vs EB+Fisher) |
|---|---|---|---|
| 32 | 4.56 min (1.67 s/problem) | 4.55 min (1.66 s/problem) | 48 vs 34 |
| 64 | 8.99 min (3.29 s/problem) | 9.01 min (3.30 s/problem) | 55 vs 52 |

- Pace at 64 steps reaches 55/164 in 9 minutes. EB+Fisher needs 128 steps (about 18 minutes) for the same score.
- All four timing runs reproduced the earlier runs' per-problem pass/fail exactly, so the results are deterministic.
- The first overnight queue crashed (GPU out of memory: MBPP task 482 has a 4,309-token prompt, and two runs hit that batch together). All runs were redone one at a time (das/jobs_rerun.txt) and finished cleanly; results are in Exp 15–17.

**Figure:** accuracy vs steps for HumanEval, GSM8K and MBPP is in the Claude Doc (summary section). matplotlib isn't installed here, so it isn't saved as a local image.

### Exp 15. MBPP: pace vs EB+Fisher across budgets

**Setup.** MBPP test rows 0–499, 3-shot prompt, 256-token canvas, stop at `[DONE]`, 3 s test timeout (das/bench_mbpp.py; 499/500 reference solutions pass the harness). Batch size 20, same model and seed as everything else (outputs/mbpp).

| Steps | EB+Fisher | Pace | Pace wins / losses | p |
|---|---|---|---|---|
| 16 | 12 | **107** | 98 / 3 | < .001 |
| 32 | 102 | **161** | 75 / 16 | < .001 |
| 64 | 184 | **195** | 23 / 12 | .09 |
| 128 | 194 | **199** | 6 / 1 | .13 |

- **Same shape as HumanEval and GSM8K.** Pace's lead is huge at low budgets and shrinks to a tie as steps grow.
- **At 16 steps EB+Fisher collapses** (12/500, 2.4%). It leaves about 232 of 256 positions masked on average, so most answers are unfinished. Pace at 16 steps (107) beats EB+Fisher at 32 steps (102).
- **Pace at 64 steps (195) matches EB+Fisher at 128 (194)**, the same 2× saving as on HumanEval.
- Pace is never worse than EB+Fisher on any of the three benchmarks below 128 steps.

### Exp 16. Mentor's count on GSM8K

**Setup.** Mentor's gradient-covariance count + Fisher score (same settings as Exp 13) on GSM8K 0–399, no blocks, compared with the existing pace and EB+Fisher runs (outputs/gsm8k).

| Steps | EB+Fisher | Mentor's count + Fisher | Pace + Fisher | Pace vs mentor (W/L, p) | Mentor vs EB+F (W/L, p) |
|---|---|---|---|---|---|
| 16 | 24 | 46 | **96** | 59/9, p < .001 | 28/6, p < .001 |
| 32 | 109 | 143 | **198** | 78/23, p < .001 | 39/5, p < .001 |
| 64 | 216 | **255** | 248 | 27/34, p = .44 (tie) | 46/7, p < .001 |

- **The mentor's count clearly beats EB+Fisher at every budget** on GSM8K, more consistently than on HumanEval.
- **Pace beats it by a wide margin at 16 and 32 steps.** On GSM8K the gap holds at 32 steps too, where on HumanEval they tied.
- **At 64 steps they tie.** The mentor's count is 7 problems ahead, which is not significant.
- Conclusion unchanged: pace is best or tied at every budget; the mentor's count is a real improvement over EB but does not adapt to short budgets.

### Exp 17. Batch-size check

**Setup.** HumanEval 164 at 32 steps, batch size 10 instead of 20 (outputs/full/*_bs10_s32).

| Method | Batch 20 | Batch 10 | Batch 10 better / worse |
|---|---|---|---|
| Pace | 48 | 45 | 0 / 3 (p = .25) |
| EB+Fisher | 34 | 36 | 3 / 1 (p = .63) |

- Batch size changes a few outcomes (padding changes the numerics slightly), but neither change is significant.
- At batch size 10, pace still beats EB+Fisher: 45 vs 36, 10 wins / 1 loss (p = .01).
- **Takeaway:** treat differences of about ±3 problems on HumanEval as noise. All headline comparisons are well beyond that.


### Exp 18. Second model: LLaDA-1.5 (final pace result)

**Setup.** LLaDA-1.5 (instruction-tuned, models/llada1_5), HumanEval 164, same prompts, greedy, batch 20 (outputs/llada15). LLaDA-1.5 tends to emit end-of-text very early on a plain completion prompt. The LLaDA paper's fix for instruction-tuned models (Appendix B.4) blocks end-of-text tokens; the mentor's harness turns it on automatically for this model. We added it as an option (`b44`) to both samplers. Smoke test, 20 problems at 64 steps: EB+Fisher plain 3/20 (empty answers), EB+Fisher + fix 10/20, pace plain 12/20, pace + fix 9/20. So each method is compared at its best setting: pace plain vs EB+Fisher + fix.

| Steps | EB+Fisher + fix | Pace (plain) | Pace wins / losses | p | Pace + fix |
|---|---|---|---|---|---|
| 16 | 12 | **26** | 15 / 1 | < .001 | 19 |
| 32 | 23 | **41** | 27 / 9 | .004 | 21 |
| 64 | 34 | **56** | 28 / 6 | < .001 | 38 |

- **Pace carries over to a second model.** It wins clearly at every budget, and on LLaDA-1.5 the gap holds even at 64 steps (56 vs 34).
- **The end-of-text fix hurts pace.** With end-of-text blocked, pace cannot see where the answer ends, so its floor spreads steps over the whole 512-token canvas. Pace + fix is level with EB+Fisher + fix (16 steps 11/4, p = .12; 32 steps 11/13; 64 steps 13/9).
- EB+Fisher + fix still leaves about 368 of 512 positions masked at 64 steps.

### Exp 19. Latent-space ideas on top of pace: no gain

**Setup.** LLaDA-8B Base, HumanEval 164, batch 20, each idea added to plain pace + Fisher (outputs/latent). Baseline: pace 39 (16 steps), 48 (32 steps). The defaults reproduce plain pace exactly (checked: 0 of 20 completions changed).

- **Extrapolated hidden state (`ext`):** push each masked position's last-layer state along its last move, h + η(h − h_prev), and rank by the model's confidence read out there. η = −0.5 (smoothing toward the previous step) is the control.
- **Soft-token feedback (`soft`):** a still-masked input gets a small share of its current predicted-token embedding instead of the pure mask embedding.

| Setting | 16 steps | W/L vs pace, p | 32 steps | W/L vs pace, p |
|---|---|---|---|---|
| ext η = 1 | 34 | 7/12, .36 | 47 | 3/4, 1 |
| ext η = 0.5 | 43 | 10/6, .45 | 50 | 6/4, .75 |
| ext η = 0.25 | 38 | 4/5, 1 | 48 | 4/4, 1 |
| ext η = −0.5 (control) | 38 | 6/7, 1 | 43 | 3/8, .23 |
| ext η = 1, no Fisher | 35 | 7/11, .48 | 44 | 5/9, .42 |
| soft 5% | 35 | 4/8, .39 | 44 | 3/7, .34 |
| soft 10% | 31 | 4/12, .08 | 43 | 4/9, .27 |

Two more ideas were only smoke-tested (20 problems, 32 steps; pace 14/20):
- **Latent diversity among pace's extra picks (`div`):** skip an extra whose hidden state is nearly identical (cosine > threshold) to one already taken this step. Thresholds 0.97/0.9/0.8/0.7: 14/14/13/13. It changes at most 3 completions; not worth a full run.
- **Depth agreement / logit lens (`lens`):** add the log-prob of the final answer under an intermediate layer's read-out. Layer 24: 5, layer 28: 11, layer 30: 9, layer 28 at weight 0.1: 12. Clearly worse; middle layers of LLaDA are not readable through the final head.

- **Nothing beats plain pace.** Extrapolation at η = 0.5 is the best (+4, +2) but well inside noise; the η sweep has no consistent direction, and the smoothing control is about as good, so the latent direction carries no usable signal for ranking.
- **Soft tokens hurt** at both budgets (up to −8). The model was never trained on blended inputs, so the mask embedding works best as a pure mask.
- Together with Exp 11, this says the ranking within a step is already near what the model can give; pace's gain comes from *how many* tokens to commit, not *which*.

### Exp 20. Checking half-step extrapolation, and sharing steps across problems: no gain

**Half-step extrapolation (ext η = 0.5) on new step counts and new data.** Decoding is greedy, so a rerun at 16/32 steps would be identical; we checked other budgets and benchmarks instead (outputs/latent).

| Benchmark | Steps | Pace | + ext 0.5 | W/L | p |
|---|---|---|---|---|---|
| HumanEval | 16 | 39 | 43 | 10/6 | .45 |
| HumanEval | 24 | 42 | 44 | 8/6 | .79 |
| HumanEval | 32 | 48 | 50 | 6/4 | .75 |
| HumanEval | 64 | 55 | 53 | 4/6 | .75 |
| GSM8K (400) | 16 | 96 | 105 | 32/23 | .28 |
| GSM8K (400) | 32 | 198 | 186 | 28/40 | .18 |
| MBPP (500) | 16 | 107 | 111 | 35/31 | .71 |
| MBPP (500) | 32 | 161 | 160 | 28/29 | 1 |

- Small gains at low budgets, small losses at higher ones, none significant. Not a reliable improvement.

**Sharing steps across the problems in a batch (`share`).** The 20 problems in a batch share 20 × S passes. At a set point, each problem's remaining passes are split in proportion to its remaining answer length (or its square root), between 0.25× and 4× S. Mean passes per problem stay exactly S. Smoke test, HumanEval 0–19 at 32 steps (pace 14/20):

| Reallocate after | Score |
|---|---|
| first pass | 13 |
| 25% of the budget | 12 |
| 25%, square root of length | 13 |
| 50% | 14 |

- No gain. The first-pass length estimate is poor (a 51-character answer was given 58 steps), and later reallocation has too little budget left to move. Not run in full.

### Exp 21. How good is pace's answer-end estimate? (diagnostic)

**Setup.** Pace on HumanEval 164 at 16, 32 and 128 steps with a trace of the estimated answer end at every step (`trace_end`, outputs/endtrace). Scores are unchanged (39 / 48 / 57).

**Finding: the end estimate hugs the written text.** On a mostly masked canvas, end-of-text is the model's top guess just past whatever is already committed. So pace's end starts near the prompt and moves outward as text is written.

| Steps | Estimate / final end at 25% of steps | at 50% | at 75% | Tokens committed in the last pass (median / mean) | Even pace per pass |
|---|---|---|---|---|---|
| 16 | 0.34 | 0.48 | 0.74 | 13.5 / 33 | 3.5 |
| 32 | 0.26 | 0.45 | 0.68 | 16 / 42 | 3.0 |

- Example (problem 0, 16 steps): the estimated end goes 8 → 29 → 53 → 78, then 173 at the last step, which has to commit 106 tokens in one pass.
- At 16 steps, 46 of 164 problems commit 32 or more tokens in the final pass.
- Problems lost at 16 steps but solved at 128 are not truncated. They fail on local slips ("delimeter, delimeter", '0' for '1', `i+11` for `i + 1`).

**Trying a confident end instead (`end_tau`): breaks decoding.** A masked position counts as past the end only if P(end-of-text or whitespace) > τ.
- As is (τ = 0.5 / 0.9): 0/20 on the first batch. Nothing is confidently "over" at step 0, so the end is 512 and pace commits confident blank lines far out, cutting answers to a few tokens.
- With extras barred from committing filler: 0/20 at 16 and 32 steps (plain pace 9/20 at 16 steps). The end never closes (stays at 512), so pace fills the whole canvas with repeated junk.

**What it means.**
- The model only knows where the answer ends close to text it has already written. A better end estimator is not available from one pass.
- So pace really works as a self-paced left-to-right schedule: the end moves with the frontier, and pace spreads the remaining work over the remaining steps.
- This fits Exp 9: forcing more tokens early ("down") hurts, because early steps have the least context. The late crunch is the price of that, and the step budget, not the end rule, is the limit.

### Exp 22. Filling the gaps: LLaDA-1.5 on GSM8K and MBPP, mentor's count on MBPP, 128-step runs

**Setup.** Same settings as Exp 15, 16 and 18 (outputs/llada15, outputs/mbpp, outputs/gsm8k). On LLaDA-1.5, each method again runs at its best setting: pace plain vs EB+Fisher with end-of-text blocked (`b44`). Several MBPP runs ran out of GPU memory when three shared the GPU; they were rerun alone from their saved batches.

**LLaDA-1.5, all benchmarks (pace vs EB+Fisher + fix):**

| Benchmark | Steps | EB+Fisher + fix | Pace | Pace wins / losses | p |
|---|---|---|---|---|---|
| HumanEval (164) | 16 | 12 | **26** | 15 / 1 | < .001 |
| HumanEval (164) | 32 | 23 | **41** | 27 / 9 | .004 |
| HumanEval (164) | 64 | 34 | **56** | 28 / 6 | < .001 |
| HumanEval (164) | 128 | 40 | **66** | 34 / 8 | < .001 |
| GSM8K (400) | 16 | 82 | 93 | 58 / 47 | .33 (tie) |
| GSM8K (400) | 32 | 165 | 178 | 72 / 59 | .29 (tie) |
| GSM8K (400) | 64 | 224 | **286** | 91 / 29 | < .001 |
| MBPP (500) | 16 | 34 | **155** | 124 / 3 | < .001 |
| MBPP (500) | 32 | 141 | **207** | 78 / 12 | < .001 |
| MBPP (500) | 64 | 176 | **205** | 54 / 25 | .001 |

- **Pace wins on LLaDA-1.5 in 8 of 10 settings and never loses.** Unlike the base model, the gap does not close at large budgets: at 128 steps on HumanEval it is still 66 vs 40.
- **GSM8K at 16 and 32 steps is a tie.** With end-of-text blocked, EB+Fisher no longer stops early on GSM8K (82 vs 24 on the base model at 16 steps), which was the main failure pace fixed there. Pace is still 11–13 ahead, but not significantly.
- On MBPP, pace at 32 and 64 steps is level (207 vs 205), so LLaDA-1.5 is saturated by 32 steps there.

**Mentor's count (+ Fisher) on MBPP, and on GSM8K at 128 steps:**

| Benchmark | Steps | EB+Fisher | Mentor's count | Pace | Pace vs mentor (W/L, p) | Mentor vs EB+F (W/L, p) |
|---|---|---|---|---|---|---|
| MBPP | 16 | 12 | 14 | **107** | 96/3, p < .001 | 6/4, p = .75 |
| MBPP | 32 | 102 | 144 | **161** | 41/24, p = .046 | 49/7, p < .001 |
| MBPP | 64 | 184 | 192 | 195 | 21/18, p = .75 (tie) | 19/11, p = .20 |
| GSM8K | 128 | 266 | 270 | 261 | 16/25, p = .21 (tie) | 17/13, p = .59 |

- **Same pattern as HumanEval and GSM8K (Exp 13, 16).** The mentor's count fails at 16 steps (14 vs 107), is a real gain over EB+Fisher at 32 steps but still loses to pace, and all three tie at 64+ steps.
- At 128 steps on GSM8K all three methods are within 9 problems (saturated).

**What it means.** The headline holds across 2 models × 3 benchmarks: pace is best or tied everywhere, with large wins whenever steps are scarce. The one weaker case (LLaDA-1.5 GSM8K at low budgets) is where the baseline's own fix already prevents early stopping, which supports the explanation that pace's gain comes mostly from pacing the answer to its end.
