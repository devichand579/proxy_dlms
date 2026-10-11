# Latent scoring and useful code-body progress

Exploratory HumanEval first 20 tasks, greedy LLaDA-8B Base, 512 positions,
128-step cap, standalone batch 20. No training or extra backbone forwards.
Context-flow methods require one partial backward through the final tokenwise
MLP and final normalization per forward. Full-model backwards remain zero.

Visible-body timing is retrospective: it cannot be used as an online stopping
rule. It excludes unresolved mask holes. Identical programs can have different
commit times; actual execution cost is shown separately.

## Main findings

Independent Fisher-dispersion count: 14/20 versus singleton 14/20, with 20/20 identical final programs.
Mean visible-body commitment moved from step 48.10 to 38.00. This is a progress gain under confidence ranking, not a final-quality or measured runtime gain.

The corrected confidence-free context-flow scorer passed 0/20 with singleton and 0/20 with adaptive allowance, versus activation singleton 7/20. It is not supported as a standalone readiness estimator.

| Method | Passed | Forwards | Partial backwards | Batch seconds | Mean body-ready step | Body multi-reveal events | Mean mask holes | Scope |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| activation_singleton | 7/20 | 128 | 0 | 98.2 | 79.85 | 0 | 12.65 | confidence-free ranking control; fixed count |
| fisher_dispersion05 | 14/20 | 128 | 0 | 113.9 | 38.00 | 117 | 0.00 | confidence-ranking diagnostic; probability-weighted count |
| fisher_singleton | 14/20 | 128 | 0 | 112.8 | 48.10 | 0 | 0.00 | confidence-ranking diagnostic |
| activation_allow01_eos | 6/20 | 128 | 0 | 102.2 | 79.00 | 291 | 30.15 | confidence-free ranking and adaptive count |
| context_flow_allow01 | 0/20 | 128 | 128 | 106.9 | 122.00 | 222 | 95.20 | confidence-free ranking and adaptive count |
| context_flow_singleton | 0/20 | 128 | 128 | 107.2 | 120.75 | 0 | 153.30 | confidence-free ranking control; fixed count |
| fisher_dispersion05_eos | 14/20 | 128 | 0 | 112.3 | 38.00 | 117 | 0.00 | confidence-ranking diagnostic; probability-weighted count |
| fisher_dispersion05_frontier32 | 14/20 | 128 | 0 | 113.3 | 36.90 | 116 | 0.00 | position-window diagnostic; confidence ranking |
| fisher_flow_dispersion05 | 14/20 | 128 | 128 | 120.1 | 37.25 | 114 | 0.00 | confidence-ranking diagnostic; probability-weighted count |
| fisher_flow_singleton005 | 14/20 | 128 | 128 | 116.4 | 47.50 | 0 | 0.00 | confidence-ranking diagnostic |
| fisher_singleton_eos | 14/20 | 128 | 0 | 108.6 | 48.10 | 0 | 0.00 | confidence-ranking diagnostic |
| fisher_singleton_frontier32 | 14/20 | 128 | 0 | 109.9 | 46.10 | 0 | 0.00 | position-window diagnostic; confidence ranking |

## Paired outcomes

- activation_singleton versus activation_singleton: wins []; losses []; identical completions 20/20.
- fisher_dispersion05 versus fisher_singleton: wins []; losses []; identical completions 20/20.
  Identical-output body timing: 18 earlier, 2 equal, 0 later; mean 10.10 steps saved.
- fisher_singleton versus fisher_singleton: wins []; losses []; identical completions 20/20.
- activation_allow01_eos versus activation_singleton: wins ['HumanEval/13']; losses ['HumanEval/4', 'HumanEval/11']; identical completions 4/20.
- context_flow_allow01 versus context_flow_singleton: wins []; losses []; identical completions 0/20.
- context_flow_singleton versus activation_singleton: wins []; losses ['HumanEval/2', 'HumanEval/4', 'HumanEval/7', 'HumanEval/8', 'HumanEval/11', 'HumanEval/15', 'HumanEval/16']; identical completions 0/20.
- fisher_dispersion05_eos versus fisher_dispersion05: wins []; losses []; identical completions 20/20.
  Identical-output body timing: 0 earlier, 20 equal, 0 later; mean 0.00 steps saved.
- fisher_dispersion05_frontier32 versus fisher_singleton_frontier32: wins []; losses []; identical completions 20/20.
  Identical-output body timing: 18 earlier, 2 equal, 0 later; mean 9.20 steps saved.
- fisher_flow_dispersion05 versus fisher_flow_singleton005: wins []; losses []; identical completions 20/20.
  Identical-output body timing: 19 earlier, 1 equal, 0 later; mean 10.25 steps saved.
- fisher_flow_singleton005 versus fisher_singleton: wins []; losses []; identical completions 19/20.
- fisher_singleton_eos versus fisher_singleton: wins []; losses []; identical completions 20/20.
  Identical-output body timing: 0 earlier, 20 equal, 0 later; mean 0.00 steps saved.
- fisher_singleton_frontier32 versus fisher_singleton: wins []; losses []; identical completions 18/20.

## Corrected arithmetic check

32 canvases and 384 masked answers per seed, 12-step cap. Reused correlated
worksheets; these results are exploratory and do not establish transfer to code.

| Method | Seed 0 | Seed 1 | Forwards/canvas | Partial backwards/canvas |
|---|---:|---:|---:|---:|
| activation_singleton | 58.85% | 57.81% | 12/12 | 0/0 |
| context_flow_allow01 | 48.70% | 47.14% | 12/12 | 12/12 |
| context_flow_cosine_singleton | 48.70% | 48.18% | 12/12 | 12/12 |
| context_flow_singleton | 48.70% | 47.66% | 12/12 | 12/12 |
| fisher_flow_dispersion05 | 76.04% | 63.80% | 11/12 | 11/12 |
| fisher_flow_singleton005 | 71.88% | 63.02% | 12/12 | 12/12 |
| fisher_flow_singleton02 | 55.21% | 55.21% | 12/12 | 12/12 |
| fisher_singleton | 73.96% | 70.05% | 12/12 | 0/0 |

## Validation and interpretation

12 trajectories passed source-snapshot, work-count, unique-reveal,
final-token and correctness-total checks. Historical control replay comparisons:
```json
{
  "fisher_singleton": {
    "wins": [],
    "losses": [],
    "identical_completions": 20
  },
  "activation_singleton": {
    "wins": [],
    "losses": [],
    "identical_completions": 20
  }
}
```

The first context-flow run omitted the residual identity derivative. It is
excluded from conclusions and retained with its original source snapshot.
Corrected runs pass an independent complete-residual gradient check.

Observed-key value attribution holds final-block attention fixed and measures
a local readout derivative; observed value vectors already contain earlier-layer
context. It is neither whole-model causal prompt attribution nor a data-error
certificate. Fisher-dispersion count uses probabilities as covariance weights.
Frontier windows use token position and are diagnostics rather than fully latent
schedulers. Final program boundaries and test outcomes never enter generation.

[Per-task body audit](/root/Strace/outputs/latent_body_progress_audit/README.md)
[Saved runs and source snapshots](/root/Strace/outputs/latent_body_progress_humaneval20)
[Corrected runs and source snapshots](/root/Strace/outputs/latent_body_progress_corrected_humaneval20)
[Earlier-step correctness replay](/root/Strace/outputs/latent_body_progress_budget_replay/README.md)
