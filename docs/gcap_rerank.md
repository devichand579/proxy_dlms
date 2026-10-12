# G-cap stochastic reranking

`gcap_rerank` generates one pool of \(K\) complete stochastic trajectories per
prompt. It stores every decoded candidate plus per-step
\(\widehat M_t,\widehat J_t\), and commit counts. Selection uses the lowest
SchedulerLens \(\widehat G=\widehat M+\widehat J\) over the configured prefix
(32 diffusion steps by default), together with a deterministic random control
from that exact pool.

The initial implementation supports `pd`, `bd`, `eb`, `top_k_c`, `top_k_m`, and
`dus`. FHS and FDM are excluded because their nested model forwards require a
different trace-to-cap treatment. Candidate metrics and both selected indices
are stored under `gcap_rerank` in each resumable checkpoint record.

One command generates the candidates once and runs two lm-eval scoring passes:

```bash
./scripts/eval.sh \
  --batch_size 4 \
  --mode gcap --gcap_scheduler pd \
  --gcap_k 8 --gcap_score_mode per_commit \
  --temperature 0.7 \
  --task_name humaneval --model llada \
  --steps 128 --gen_length 512 \
  --no_eos_harness --resumable --wandb True
```

The minimum-G result is in the normal output directory; the random-control
result is in the sibling directory ending in `_random`. Random selection uses a
separate prompt-derived RNG and does not alter candidate generation.
The defaults are \(K=8\), temperature \(0.7\), a 32-step prefix, M-cap
\(\gamma=1\), and `--gcap_score_mode per_commit`. Raw G-cap can otherwise favor
incomplete outputs. Compute cost is approximately \(K\) times the base
scheduler NFE.

`scripts/gcap_suite.sh` runs HumanEval, MBPP, GSM8K, and MATH-500 over all six
supported base schedulers on eight GPUs. EB and top-k margin each use two
disjoint dataset shards. The final `schedulerlens.gcap_summary` pass reports
exact expected random pass@1 over all candidates, pass@8, and prefix-minimum-G
selected@1.
