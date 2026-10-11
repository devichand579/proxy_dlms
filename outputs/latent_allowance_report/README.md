# Latent error allowances and response-feedback counts

One normal LLaDA forward per step. New controllers use actual cached-state changes,
native last-block attention, and frozen decoder-gradient geometry. No policy training,
counterfactual branch forwards, or backward passes. Fisher-geometry diagnostics use
model probabilities as gradient-covariance weights; these are not strict probability-free candidates.

The Fisher ranking includes confidence and is held fixed as a diagnostic. Activation
ranking uses attention to observed context minus temporal movement. Source ranking
and count eligibility are reported separately. Allowances are empirical proxies, not
certified bounds on data KL or program error.

## Findings

Completed 33 arithmetic settings on two seeds and 13 HumanEval-20 settings.

- eb_sampler: 13/20, 128 forwards/problem.
- fisher_eb: 14/20, 128 forwards/problem.
- fisher_singleton: 14/20, 128 forwards/problem.
- activation_singleton: 7/20, 128 forwards/problem.

The probability-weighted Fisher-geometry counts matched 14/20 with independent
non-EB count rules, but all 20 decoded completions were identical to Fisher
singleton. Their additional reveals did not change the decoded programs, so
these results do not establish a quality benefit from adaptive count.

The probability-free grounded feedback count also matched 14/20 under Fisher
ranking, but committed only 128.25 positions/problem and made five two-token
decisions across all 20 trajectories. It effectively reproduced singleton.

The largest tested Fisher-ranked temporal allowance dropped to 11/20; relative
feedback reached 13/20. Larger groups did not reliably improve code.

Under activation-only ranking, grounded feedback retained 7/20 with nearly
singleton counts. The other activation-ranked count arms scored 6/20. No
strict probability-free adaptive candidate improved on the 7/20 scoring control.

A full-164 promotion is not supported by this pilot: tied small-set results
mostly reproduce the same programs or reveal more unused tail positions.

The practical next question is whether an interaction estimate identifies
safe parallelism among actual output-body decisions. Motion magnitude alone
is not wrong-commitment risk, and last-block attention alone is not token TC.

Exact oracle examples, separately from neural results, demonstrate entropy-bound
conservatism for independent bits and the failure of pairwise MI on parity.
[Toy sampling diagnostics](/root/Strace/outputs/latent_allowance_toys/diagnostic.json).

## Arithmetic, two seeds, 32 canvases / 384 answers per seed, 12-step cap

| Method | Seed 0 | Seed 1 | Mean | Forwards/canvas | Ranking | Count metric |
|---|---:|---:|---:|---:|---|---|
| fisher_dispersion05 | 72.92% | 71.35% | 72.14% | 11/11 | diagnostic_fisher | fisher_dispersion |
| fisher_dispersion002 | 73.96% | 70.05% | 72.01% | 12/12 | diagnostic_fisher | fisher_dispersion |
| fisher_dispersion01 | 73.96% | 70.05% | 72.01% | 12/12 | diagnostic_fisher | fisher_dispersion |
| fisher_eb | 73.96% | 70.05% | 72.01% | 12/12 | diagnostic_fisher | uniform |
| fisher_gradient_allow002 | 73.96% | 70.05% | 72.01% | 12/12 | diagnostic_fisher | gradient_union |
| fisher_gradient_fb005 | 73.96% | 70.05% | 72.01% | 12/12 | diagnostic_fisher | gradient_union |
| fisher_gradient_fb025 | 73.96% | 70.05% | 72.01% | 12/12 | diagnostic_fisher | gradient_union |
| fisher_gradient_fb1 | 73.96% | 70.05% | 72.01% | 12/12 | diagnostic_fisher | gradient_union |
| fisher_relative1 | 73.96% | 70.05% | 72.01% | 12/12 | diagnostic_fisher | gradient_union |
| fisher_singleton | 73.96% | 70.05% | 72.01% | 12/12 | diagnostic_fisher | uniform |
| fisher_uniform_fb005 | 73.96% | 70.05% | 72.01% | 12/12 | diagnostic_fisher | uniform |
| fisher_uniform_fb025 | 73.96% | 70.05% | 72.01% | 12/12 | diagnostic_fisher | uniform |
| fisher_uniform_relative4 | 74.22% | 69.79% | 72.01% | 12/12 | diagnostic_fisher | uniform |
| fisher_gradient_allow01 | 73.96% | 69.79% | 71.88% | 12/12 | diagnostic_fisher | gradient_union |
| fisher_gradient_grounded025 | 74.22% | 69.27% | 71.74% | 12/12 | diagnostic_fisher | gradient_union |
| fisher_gradient_allow05 | 71.61% | 71.61% | 71.61% | 10/10 | diagnostic_fisher | gradient_union |
| fisher_uniform_fb1 | 72.92% | 70.31% | 71.61% | 10/12 | diagnostic_fisher | uniform |
| fisher_relative8 | 73.44% | 69.53% | 71.48% | 11/11 | diagnostic_fisher | gradient_union |
| fisher_relative4 | 73.70% | 69.27% | 71.48% | 12/12 | diagnostic_fisher | gradient_union |
| fisher_grounded_allow01 | 73.44% | 69.27% | 71.35% | 12/11 | diagnostic_fisher | gradient_union |
| fisher_gradient_risk01 | 70.31% | 72.40% | 71.35% | 12/12 | diagnostic_fisher | gradient_union |
| fisher_subset_allow01 | 68.49% | 72.66% | 70.57% | 12/12 | diagnostic_fisher | gradient_union |
| fisher_gradient_pair01 | 71.35% | 69.79% | 70.57% | 12/11 | diagnostic_fisher | gradient_union |
| fisher_grounded_allow05 | 73.96% | 66.93% | 70.44% | 7/7 | diagnostic_fisher | gradient_union |
| eb_sampler | 71.09% | 69.79% | 70.44% | 12/12 | eb_confidence | entropy |
| fisher_relative_grounded4 | 71.61% | 68.75% | 70.18% | 12/12 | diagnostic_fisher | gradient_union |
| fisher_uniform_grounded025 | 71.61% | 67.97% | 69.79% | 4/4 | diagnostic_fisher | uniform |
| activation_dispersion01 | 59.11% | 57.81% | 58.46% | 12/12 | activation | fisher_dispersion |
| activation_grounded025 | 58.59% | 58.07% | 58.33% | 12/12 | activation | gradient_union |
| activation_singleton | 58.85% | 57.81% | 58.33% | 12/12 | activation | uniform |
| activation_allow01 | 58.33% | 55.73% | 57.03% | 12/12 | activation | gradient_union |
| activation_relative4 | 56.25% | 57.29% | 56.77% | 12/12 | activation | gradient_union |
| activation_grounded_allow01 | 56.51% | 54.69% | 55.60% | 10/11 | activation | gradient_union |

Arithmetic component worksheets are reused and correlated; small gains do not establish robustness.

## HumanEval-20, 512 positions, 128-step cap, matched standalone batch 20

| Method | Passed | Forwards/problem | Mean committed | Count >1 events | Scope |
|---|---:|---:|---:|---:|---|
| fisher_singleton | 14/20 | 128 | 128.00 | 0 | confidence/Fisher diagnostic or baseline |
| fisher_eb | 14/20 | 128 | 350.10 | 1304 | confidence/Fisher diagnostic or baseline |
| fisher_uniform_grounded025 | 14/20 | 128 | 128.25 | 5 | confidence/Fisher diagnostic or baseline |
| fisher_dispersion05 | 14/20 | 128 | 224.90 | 648 | confidence/Fisher diagnostic or baseline |
| fisher_dispersion01 | 14/20 | 128 | 164.35 | 327 | confidence/Fisher diagnostic or baseline |
| fisher_relative4 | 13/20 | 128 | 130.75 | 32 | confidence/Fisher diagnostic or baseline |
| eb_sampler | 13/20 | 128 | 345.90 | 0 | confidence/Fisher diagnostic or baseline |
| fisher_grounded_allow01 | 12/20 | 128 | 150.60 | 408 | confidence/Fisher diagnostic or baseline |
| fisher_gradient_allow05 | 11/20 | 128 | 201.60 | 908 | confidence/Fisher diagnostic or baseline |
| activation_singleton | 7/20 | 128 | 128.00 | 0 | activation singleton control |
| activation_grounded025 | 7/20 | 128 | 129.40 | 9 | strict candidate |
| activation_allow01 | 6/20 | 128 | 174.65 | 755 | strict candidate |
| activation_dispersion01 | 6/20 | 128 | 151.35 | 262 | activation ranking / Fisher-weighted count diagnostic |

## Fixed-ranking paired comparisons

- fisher_uniform_grounded025 versus fisher_singleton: wins []; losses []; identical completions 20/20.
- fisher_dispersion05 versus fisher_singleton: wins []; losses []; identical completions 20/20.
- fisher_dispersion01 versus fisher_singleton: wins []; losses []; identical completions 20/20.
- fisher_relative4 versus fisher_singleton: wins []; losses ['HumanEval/4']; identical completions 19/20.
- fisher_grounded_allow01 versus fisher_singleton: wins []; losses ['HumanEval/9', 'HumanEval/18']; identical completions 15/20.
- fisher_gradient_allow05 versus fisher_singleton: wins []; losses ['HumanEval/5', 'HumanEval/11', 'HumanEval/13']; identical completions 8/20.
- activation_grounded025 versus activation_singleton: wins []; losses []; identical completions 19/20.
- activation_allow01 versus activation_singleton: wins ['HumanEval/13']; losses ['HumanEval/4', 'HumanEval/11']; identical completions 4/20.
- activation_dispersion01 versus activation_singleton: wins ['HumanEval/3']; losses ['HumanEval/4', 'HumanEval/15']; identical completions 9/20.

## Artifacts and reproduction

- [latent_allowance_arithmetic](/root/Strace/outputs/latent_allowance_arithmetic)
- [latent_allowance_relative_arithmetic](/root/Strace/outputs/latent_allowance_relative_arithmetic)
- [latent_allowance_grounded_arithmetic](/root/Strace/outputs/latent_allowance_grounded_arithmetic)
- [latent_allowance_fisher_geometry_arithmetic](/root/Strace/outputs/latent_allowance_fisher_geometry_arithmetic)
- [latent_allowance_humaneval20](/root/Strace/outputs/latent_allowance_humaneval20)

Each experiment stores source hashes and source snapshots, fixed decoder geometry,
per-step count decisions, state checkpoints, completions, and fresh-child correctness
outcomes. No hidden tests or reference answers enter scheduling.

Run new directories with `python -m schedulerlens.latent_allowance_bench --task arithmetic`
or `--task humaneval`, supplying `--out` and an explicit `--plans` list.

Feedback responds after a commit and cannot undo earlier errors. Attention is an
interaction proxy, not measured token mutual information. Observed displacement can
be beneficial learning from context; its magnitude alone is not an error certificate.
