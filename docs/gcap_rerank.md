# G-cap stochastic reranking

`gcap_rerank` generates one pool of \(K\) complete stochastic trajectories per
prompt. It selects both the lowest SchedulerLens
\(\widehat G=\widehat M+\widehat J\) trajectory and a deterministic random
control from that exact pool.

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

After both scoring passes finish, the run writes one report,
`gcap_results.json`, in the minimum-G output directory. It contains `min_g`,
`random`, `pool_mean`, and `pool_best` for the task's primary metric.
`pool_mean` averages all \(K\) candidates per sample, then averages over
samples. `pool_best` takes the best candidate per sample, then averages over
samples. The harness still writes `results.json` in that directory and in the
`_random` sibling; those are the inputs to the combined report. Random
selection uses a separate prompt-derived RNG and does not alter candidate
generation. Rows written before candidate texts were stored are left out of
the pool numbers.
The defaults are \(K=8\), temperature \(0.7\), M-cap \(\gamma=1\), and
`--gcap_score_mode per_commit`. Raw G-cap can otherwise favor incomplete
outputs. Compute cost is approximately \(K\) times the base scheduler NFE.
