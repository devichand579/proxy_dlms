# Handoff: Pace, a step-budget unmasking schedule for LLaDA (BTP project)

**Written:** 2026-10-11, at the end of the main experiment phase.
**For:** any agent (or person) continuing this work. Read this first, then `experiment_log.md` for the full results.
**Style rule from the user:** plain, simple language with little jargon. Lead with the idea, put maths second.

---

## 1. The story in one page

**What the project is about.** LLaDA is a *masked diffusion* language model. It writes an answer by starting from a canvas of blank (masked) tokens and filling some of them in at every step. Each step costs one full forward pass of an 8B model. An *unmasking scheduler* decides, at each step:
- **how many** blanks to fill (the count), and
- **which** blanks to fill (the ranking).

Fewer steps means faster generation, but filling many blanks at once causes errors, because the model predicts each blank independently while the real tokens depend on each other.

**Where it started.** The original BTP paper draft (`Schedule_tracing.pdf`, "Scheduler Tracing") derived a scheduler objective from KL(q‖p). Its key quantity was a token-dependence matrix **M**, estimated with Cross-Step Transcoders (CSTs, sparse dictionaries over hidden states). CSTs were dropped on 2026-10-05/06: about 70 TB of traces per model/scheduler pair, and banks of up to ~43B parameters. The plan became: build a CST-free scheduler, and estimate M by *lookahead* instead (commit one candidate, re-run, see how the others' predictions shift).

**What the mentor had.** The mentor (separate Codex sessions, Oct 3–8, code in `/root/Strace`) built many "latent" schedulers. His best was **EB+Fisher**: the EB-Sampler count, with tokens ranked by confidence minus 0.15 × a temporal Fisher risk term. It scored 55/164 on HumanEval at 128 steps (plain EB: 47). He also had a gradient-covariance count ("allowance", Fisher dispersion) and an information-gain score (56/164 at 2× passes).

**What we found, in order** (details in `experiment_log.md`):
1. **The dependence measurement M is real but tiny.** Measured dependence between top candidates is ~0.01 against a budget of 0.1, so EB's worst-case bound is about 10× too cautious. Using M per step never beat a matched control and costs 2× passes (Exp 3). M stays in the paper as *evidence*, not as a component.
2. **Loosening EB's bound works only at low budgets.** A fixed looser threshold (α) wins big at 16 steps and loses at 128 (Exp 2, 4, 5). The 40-problem screen misled us here, so from then on: **always full sets**.
3. **The right looseness depends on the step budget → pace.** Instead of a fixed α, require each step to make enough progress to finish on time:

       |S_t| = max( n_EB(t), ⌈ L_rem / (T − t) ⌉ )

   L_rem is the number of answer tokens still masked, T − t the steps left. When EB already keeps pace, pace *is* EB+Fisher. When EB falls behind, pace adds just enough extra tokens (taken in ranking order) to finish.
4. **Pace wins big when steps are scarce and ties when they are plentiful**, on 3 benchmarks × 2 models. It costs one forward pass per step, same as EB (Exp 14 timing: identical wall-clock per step).
5. **Count beats ranking.** More than twenty variants of *which* tokens to pick (spacing, schedule shape, order, margin, blocks, two-sided pace, remasking, latent extrapolation, soft tokens, latent diversity, logit lens, step sharing across problems, confident-end rules, the mentor's information score) all tie or lose to plain pace (Exp 9–13, 19–21).
6. **The mentor's count** beats EB at ~32 steps but fails at 16 (it's capped at 16 tokens/step and has no notion of the budget) and ties everyone at 64+. Pace is best or tied at every budget (Exp 13, 16, 22).
7. **Why pace works (Exp 21).** The estimated answer end hugs the already-written text and moves outward. So pace really acts as a self-paced left-to-right schedule that spreads the remaining work over the remaining steps.

**Theory, briefly** (full version in the write-up §4 and the log's "Theory recap"):
- Error of unmasking set S at once = Σ per-token model error + TC(X_S), the total correlation among S.
- EB's rule ΣH − maxH ≤ γ is exactly this constraint with TC replaced by its worst-case upper bound.
- Real dependence (D_ij, measured by lookahead) is ~10× below that bound, so relaxing it is safe.
- A step budget adds a progress term π|S| to the objective (a Lagrange multiplier on "finish in T steps"). Pace is the smallest relaxation that meets the budget.

---

## 2. Headline results

LLaDA-8B Base, greedy, seed 1234, batch 20. Numbers = problems solved. W/L = paired per-problem wins/losses for pace, exact McNemar test.

| Benchmark | Steps | EB+Fisher | Mentor's count + Fisher | **Pace** | Pace vs EB+F (W/L, p) |
|---|---|---|---|---|---|
| HumanEval (164) | 16 | 17 | 19 | **39** | 23/1, <.001 |
| | 32 | 34 | 47 | **48** | 15/1, .001 |
| | 64 | 52 | 51 | **55** | 4/1 |
| | 128 | 55 | — | **57** | 3/1 |
| GSM8K (400) | 16 | 24 | 46 | **96** | 79/7, <.001 |
| | 32 | 109 | 143 | **198** | 104/15, <.001 |
| | 64 | 216 | 255 | 248 | 55/23, <.001 |
| | 128 | 266 | 270 | 261 | 12/17, .46 (tie) |
| MBPP (500) | 16 | 12 | 14 | **107** | 98/3, <.001 |
| | 32 | 102 | 144 | **161** | 75/16, <.001 |
| | 64 | 184 | 192 | **195** | 23/12, .09 |
| | 128 | 194 | — | **199** | 6/1, .13 |

LLaDA-1.5 (instruction-tuned), pace (plain) vs EB+Fisher with end-of-text blocked (`b44`, the LLaDA paper's fix, which is EB+F's best setting on this model):

| Benchmark | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| HumanEval | **26** vs 12 | **41** vs 23 | **56** vs 34 | **66** vs 40 |
| GSM8K | 93 vs 82 (tie, p=.33) | 178 vs 165 (tie, p=.29) | **286** vs 224 | — |
| MBPP | **155** vs 34 | **207** vs 141 | **205** vs 176 | — |

Across both models: pace wins significantly in 15 of 22 settings, ties the other 7, never loses significantly. On HumanEval/MBPP, pace at 64 steps ≈ EB+Fisher at 128 (2× speed-up).

**Noise floor:** changing only the batch size moves scores by 1–3 problems (Exp 17). Treat ±3 on HumanEval as a tie.

---

## 3. Where everything lives

### Documents
| What | Where |
|---|---|
| Full experiment log (Exp 1–22, setup, theory, lessons, idea table) | `/root/aditya/experiment_log.md` |
| Online copy of the log (Claude Doc, keep in sync with the file) | https://claude.ai/code/artifact/c590cfe8-15cb-485f-a778-2183d6037679 |
| **Paper-style write-up** (abstract → conclusion, chart, theory §4) | https://claude.ai/code/artifact/c70cef50-c891-4060-a1ac-82517e13519f ("Pace: A Step-Budget Unmasking Schedule for LLaDA"); offline copy `/root/aditya/writeup.md` (chart replaced by a text placeholder) |
| Research plan v2 / v1 (historical) | `plan.md`, `plan_v1_2026-10-06.md` |
| Original paper draft | `Schedule_tracing.pdf` |
| Reference papers (EB-Sampler `2505.24857v1.pdf`, LLaDA, InfoGain, BatchBALD, …) | `papers/`, top-level PDFs |
| Mentor's notes and result summaries | `/root/aditya/mentor/` (copy of his `docs/` and the small result files from his `outputs/`; see `mentor/README.md`) |

### Code: `/root/aditya/das/` (ours)
| File | What it does |
|---|---|
| `das_sampler.py` | **The sampler.** `generate_das(...)` implements every count/score we tried. Pace is `count='pace'` (logic around line 386: answer end = first EOS/EOT or start of a run of `pad_run=8` whitespace tokens, committed or top-1 predicted; extras taken from content positions in ranking order). Defaults reproduce EB+Fisher. Other knobs: `base='allowance'` (mentor's count), `score='info'` (mentor's information score), `b44`, `block_len`, `ext`, `soft`, `div`, `lens`, `share`, `trace_end`, `end_tau`, `remask_tau`, `pace_cap`, `pace_shape`, `pace_space`, `pace_order`, `rank='margin'`, … Each is documented in the log experiment that used it. |
| `bench.py` | HumanEval runner (`--method eb|ebf|das`, `--cfg` JSON kwargs for `generate_das`, `--tasks 0-163`, `--steps`, `--model models/llada|models/llada1_5`, `--out`). Saves `batch_XXX.json` per 20 problems; **resumes** from saved batches. |
| `bench_gsm8k.py`, `bench_mbpp.py` | Same for GSM8K (4-shot, 256 tokens, tasks 0-399) and MBPP (3-shot, 256 tokens, tasks 0-499, 3 s test timeout). |
| `compare.py` | Builds `SUMMARY.txt` tables (score, NFE, unfinished, W/L vs a reference). |
| `pair.py` | Paired W/L + exact McNemar p (reads `results.json`, or `batch_*.json` if missing). Run from `/root/aditya/outputs`: `python3 ../das/pair.py full/pace_s16:full/ebf_s16` |
| `run.sh` | Runs one job: `cd` to the folder above `das/`, sets `PYTHONPATH` to it, runs `python -m ${MODULE:-das.bench}`, logs to `outputs/logs/<name>.log` (overwrites). Env: `MODEL_ROOT` (default `/root/Strace`, where `models/llada` lives), `PYTHON`, `OUT_ROOT`. |
| `queue2.sh <jobfile> <N>` | Runs a job file with N in parallel. Each line is a full command run from `das/`. |
| `jobs_*.txt` | Every job file we ran; the exact command lines for each experiment (e.g. `jobs_gaps.txt` = Exp 22, `jobs_l15.txt` = Exp 18). |
| `retry_gaps.sh`, `retry_after.sh`, `night.sh` | Helpers: rerun jobs whose log has no `RESULT` line, overnight chains. |
| `tests/` | Smoke tests (20-problem quick checks). |
| `vendor/` | Copies of the 17 Strace files das imports (`load_model.py`, parts of `schedulers/`, `schedulerlens/latent_bench.py`, `traces/`). `das/__init__.py` puts it first on `sys.path`. Identical to Strace branch `experiment-set-1`; don't edit. |
| `README.md` | Usage summary. |

Example job line:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODULE=das.bench_mbpp ./run.sh --method das --name mbpp_pace_s32 --steps 32 --tasks 0-499 --out /root/aditya/outputs/mbpp --cfg '{"count":"pace"}'
```

### Results: `/root/aditya/outputs/`
| Folder | Contents |
|---|---|
| `full/` | HumanEval 164, LLaDA-8B Base (main runs, variants, ablations, count × score grid). `SUMMARY.txt`, `SUMMARY2.txt` |
| `gsm8k/`, `mbpp/` | GSM8K 400 and MBPP 500 runs |
| `llada15/` | LLaDA-1.5 runs (prefix `l15_` HumanEval, `l15g_` GSM8K, `l15m_` MBPP) |
| `latent/`, `latent_smoke*/` | Exp 19–20 latent ideas |
| `endtrace/` | Exp 21 answer-end traces |
| `timing/` | Exp 14 wall-clock runs |
| `screen/` | Early 40-problem screen (Exp 1–3; don't trust for conclusions) |
| `logs/` | One log per run; a finished run ends with a `RESULT` line |

Each run folder has `batch_XXX.json` files with a `samples` list (`task`, `passed`, completion, per-row metrics such as forward passes and unfinished tokens). Run names follow `<method>_<variant>_s<steps>`.

### Independence from the mentor's repo, and git
- **`/root/aditya` is self-contained.** The Strace modules our code needs are copied into `das/vendor/`, and the mentor's per-task HumanEval results used by `compare.py` are in `outputs/mentor/`. Only the **model weights** come from outside: `/root/Strace/models/llada` (LLaDA-8B Base, 15 GB) and `/root/Strace/models/llada1_5` (15 GB), found via `MODEL_ROOT`. Checked on 2026-10-11: a 20-problem rerun (pace and EB+Fisher, 16 steps) through the standalone code gave identical generations and pass/fail to the original runs.
- **GitHub repo:** `git@github.com:devichand579/Strace.git` (SSH works from this machine as `devichand579`).
  - Branch **`experiment-set-1`**: the mentor's previously uncommitted work (latent schedulers, schedulerlens benches, tests, docs, small result files from `outputs/`), on top of `main`.
  - Branch **`experiment-set-2`** (on top of set 1): our work. `das/` (without `vendor/`; inside the repo it uses the repo's own modules), `docs/pace/` (write-up, log, this handoff, README mapping the paths), `outputs/pace/` (`results.json` + `config.json` per run and the SUMMARY tables; raw generations not committed).
  - `/root/Strace` itself is checked out on `experiment-set-1`.
- **`/root/aditya/Strace/` is an old copy of the repo (commit adf822b, no `latent_*` modules).** It's not used by anything; ignore it.
- Python: `/root/miniconda3/envs/strace/bin/python`. GPU: 1× AMD MI300X (ROCm). matplotlib isn't installed.
- **Don't edit the mentor's files** (in `/root/Strace` or `das/vendor/`). Our code lives in `/root/aditya/das`; if it changes, also update branch `experiment-set-2`.

---

## 4. Working rules the user has set

1. **Plain language** in all docs and replies.
2. **Run only relevant experiments.** Don't queue nice-to-have runs (extra budgets, extra benchmarks) without proposing them first.
3. **Always full sets** (HumanEval 164, GSM8K 0–399, MBPP 0–499) for any conclusion. 20-problem smoke tests are only for "does it run / is it obviously broken".
4. **Paired tests for every claim.** Differences within the noise floor are ties.
5. **GPU jobs run detached in the background** (`setsid nohup ./queue2.sh jobs_x.txt N > outputs/logs/x.log 2>&1 &`), so they survive session restarts. Waits should be background too.
6. **Parallelism:** 2–3 jobs at once is fine for HumanEval/GSM8K. **MBPP runs one at a time** (task 482 has a 4,309-token prompt; 3 in parallel ran out of memory). Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
7. **Keep docs in sync:** every new experiment goes into `experiment_log.md` **and** the online log doc, and if it changes a headline, the write-up too.
8. **Compute limit:** at most ~2× the forward passes of the base sampler; no learned heads or policies. Confidence, entropy and latent signals are all allowed.

---

## 5. Gotchas

- **LLaDA Base ends code answers with whitespace padding, not EOS.** Pace's first version (end = first EOS) scored 3/164 at 16 steps. Hence the 8-whitespace-run end rule.
- **LLaDA-1.5 emits end-of-text too early** on plain completion prompts. The mentor's/LLaDA paper's fix (`b44`, block end-of-text) helps EB+Fisher but **hurts pace** (pace can no longer see the end). Compare each method at its best setting: pace plain vs EB+F + `b44`.
- **The mentor's semi-autoregressive block sampler collapses at low budgets** (fixed step share per block → 496/512 positions left masked at 16 steps).
- **Greedy decoding is deterministic:** rerunning the same config gives identical per-problem results (checked in Exp 14). Batch size changes a few outcomes through padding numerics.
- **The 40-problem screen (HumanEval 20–59) is too easy** and flattered the fixed-α method. Don't use it.
- `run.sh` **overwrites** the log of the same `--name`, but benches **resume** from existing `batch_*.json`. Delete the run folder to redo a run from scratch.
- Background waiters die when the Claude session restarts; detached GPU jobs don't. After a restart, check `outputs/logs/*.log` for `RESULT` lines.

---

## 6. Status and what's left

**Done:** all experiments (Exp 1–22), the full log, the online log doc, the paper-style write-up with chart and theory, the standalone code, and both GitHub branches. No jobs are running.

**GPU access ends ~2026-10-12 04:00 UTC.** After that, no new runs are possible on this machine. Back up:
- `/root/aditya/` (code, results, docs, mentor copies; self-contained except the model weights; the old `Strace/` copy inside can be skipped).
- Code, docs and result summaries of both the mentor's and our work are also on GitHub (branches `experiment-set-1`, `experiment-set-2`). Raw generations are only on this machine (`/root/aditya/outputs`, `/root/Strace/outputs`).
- Weights can be re-downloaded from Hugging Face (`GSAI-ML/LLaDA-8B-Base`, `GSAI-ML/LLaDA-1.5`).

**Possible next steps** (only if the user asks):
1. **Polish the write-up:** fresh-eyes pass, fix any wording the user/mentor comments on. Keep its numbers consistent with `experiment_log.md`.
2. **Tidy the log:** "Lessons learned" and "New experiment ideas" sit between Exp 10 and Exp 11 (so Exp 11–22 fall under the ideas heading). Move both to the end and update the lessons with Exp 11–22 (MBPP, LLaDA-1.5, 128-step results, "count beats ranking" now backed by 20+ variants). Mirror the change in the online doc.
3. **Missing cells, if GPU time is available somewhere:** LLaDA-1.5 GSM8K/MBPP at 128 steps; mentor's count at 128 steps on HumanEval/MBPP; Dream-7B as a third model (weights not downloaded; `models/dream` is config only).
4. **Ideas not yet tried** (from the log's idea table): lookahead only on steps where the pace floor fires (idea 3); smoother answer-length estimate via running max (idea 5); reserve the last steps for checking (idea 7). Exp 11, 19 and 21 suggest gains here will be small: the step budget, not the ranking or the end estimate, is the limit.

**Paper framing to keep:** pace's contribution is the *count*, derived from the theory (EB = worst-case TC bound; measured dependence ~10× lower; budget = progress term). It needs no tuning, costs 1× passes, and is best or tied at every budget on 2 models × 3 benchmarks.
