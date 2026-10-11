# One-pass latent objective experiments

Frozen LLaDA-8B Base on the local AMD GPU. Candidate scoring and count use activations,
existing decoder-gradient directions, and latent Gaussian set objectives. No EB count,
vocabulary confidence/entropy, trained heads, backward passes, or extra model forwards.
Native EB is the baseline only.

These are exploratory results on reused tasks, not a claim of the best possible algorithm.

## Findings

Completed 40 arithmetic configurations on both seeds and 15 HumanEval configurations.

Native EB passed **13/20**. The best new adaptive candidate,
`context_smallgain`, passed **4/20** using
128 ordinary forwards/problem. Singleton controls are excluded from this candidate ranking.

The arithmetic leader scored 81.25% / 79.17%,
but passed **0/20** on HumanEval. Its arithmetic gain did not transfer.

Counts alone did not repair weak selection: a temporal-risk variant committed the
entire 512-position canvas in 57 steps and passed 0/20. The no-information
context arm beat its matching singleton control 3/20 versus 0/20.
Adding a small information weight reached 4/20; a singleton control with
that exact weight was not run, so its additional gain cannot be assigned
to count alone. None of these candidates warrants promotion to a
full-164 run on the evidence here.

The original error term remains an unvalidated commitment-risk proxy; the
Gaussian information and dependence terms describe internal geometry rather
than verified token interactions. Better algebra did not establish better generation.

For context, the earlier activation-only singleton control passed 7/20 at
128 forwards/problem, and its costly growth guard passed 8/20 at 1144
forwards/problem. Those are historical controls, not new adaptive one-pass candidates.
[Earlier count study](/root/Strace/outputs/latent_count_ablation_report/README.md).

## Arithmetic: 32 canvases, 384 slots per seed, 12-step cap

| Method | Seed 0 | Seed 1 | Mean | Forwards/canvas (0 / 1) |
|---|---:|---:|---:|---:|
| grad_curvature_g010_r005 | 81.25% | 79.17% | 80.21% | 12 / 12 |
| grad_curvature_nogain_singleton | 81.25% | 78.91% | 80.08% | 12 / 12 |
| grad_curvature_noinformation | 81.25% | 78.39% | 79.82% | 12 / 12 |
| grad_centered | 79.43% | 79.69% | 79.56% | 12 / 12 |
| grad_curvature_noincoming | 80.73% | 78.39% | 79.56% | 12 / 12 |
| grad_curvature_g0025_r005 | 80.99% | 77.86% | 79.43% | 12 / 12 |
| grad_risk_scaled | 78.65% | 80.21% | 79.43% | 12 / 12 |
| grad_curvature_singleton | 76.56% | 72.14% | 74.35% | 12 / 12 |
| grad_curvature_g0025_r015 | 73.70% | 69.79% | 71.74% | 9 / 8 |
| grad_curvature_nogain_r015 | 73.96% | 67.97% | 70.96% | 9 / 9 |
| eb_sampler | 71.09% | 69.79% | 70.44% | 12 / 12 |
| grad_curvature_r005 | 65.36% | 62.76% | 64.06% | 9 / 9 |
| context_only_centered | 61.46% | 62.50% | 61.98% | 10 / 11 |
| context_centered_r015 | 61.20% | 59.90% | 60.55% | 11 / 11 |
| context_nogain | 59.38% | 59.38% | 59.38% | 11 / 12 |
| context_smallgain | 59.38% | 58.59% | 58.98% | 11 / 11 |
| grad_curvature_r015 | 63.02% | 53.91% | 58.46% | 7 / 6 |
| temporal_only | 57.81% | 59.11% | 58.46% | 5 / 6 |
| singleton | 60.94% | 55.21% | 58.07% | 12 / 12 |
| grad_support_noinformation | 53.39% | 61.20% | 57.29% | 12 / 12 |
| grad_support_singleton | 55.73% | 58.85% | 57.29% | 12 / 12 |
| no_information | 58.85% | 55.21% | 57.03% | 11 / 11 |
| grad_support_r015 | 57.03% | 56.77% | 56.90% | 5 / 5 |
| context_raw | 56.25% | 57.03% | 56.64% | 12 / 12 |
| no_dependence | 58.07% | 54.95% | 56.51% | 5 / 5 |
| full_r035 | 57.03% | 55.47% | 56.25% | 3 / 3 |
| full_r005 | 57.81% | 54.43% | 56.12% | 9 / 9 |
| depth_only | 57.29% | 54.69% | 55.99% | 6 / 6 |
| no_incoming | 55.99% | 55.47% | 55.73% | 7 / 7 |
| context_nogain_singleton | 58.07% | 53.39% | 55.73% | 12 / 12 |
| full_r015 | 56.25% | 54.69% | 55.47% | 7 / 7 |
| grad_curvature_g0025_r035 | 57.29% | 53.65% | 55.47% | 3 / 3 |
| context_centered | 54.69% | 55.47% | 55.08% | 12 / 12 |
| context_singleton | 54.69% | 55.47% | 55.08% | 12 / 12 |
| temporal_scaled_singleton | 52.08% | 56.25% | 54.17% | 12 / 12 |
| no_information_r035 | 54.69% | 53.39% | 54.04% | 3 / 3 |
| full_r075 | 53.65% | 53.91% | 53.78% | 2 / 2 |
| grad_support_r005 | 52.60% | 53.65% | 53.12% | 8 / 9 |
| temporal_scaled_gain1 | 51.30% | 53.39% | 52.34% | 7 / 7 |
| temporal_risk_scaled | 51.82% | 52.60% | 52.21% | 7 / 8 |

Arithmetic slots share worksheets and are not independent observations. Duplicate
settings across stages must reproduce their exact per-slot outcomes.

## HumanEval-20: 128-step cap, 512 positions, standalone batch 20

| Method | Passed | Forwards/problem | Mean committed | Mean batch count | Generation seconds |
|---|---:|---:|---:|---:|---:|
| eb_sampler | 13/20 | 128 | 345.90 | not traced | 139.7 |
| context_smallgain | 4/20 | 128 | 219.75 | 1.72 | 99.1 |
| context_singleton | 3/20 | 128 | 128.00 | 1.00 | 98.9 |
| context_nogain | 3/20 | 128 | 193.35 | 1.51 | 101.9 |
| grad_centered | 2/20 | 128 | 203.50 | 1.59 | 96.3 |
| grad_risk_scaled | 2/20 | 128 | 181.75 | 1.42 | 96.6 |
| context_centered_r015 | 2/20 | 128 | 358.20 | 2.80 | 101.8 |
| grad_curvature_nogain_singleton | 1/20 | 128 | 128.00 | 1.00 | 96.3 |
| grad_curvature_noinformation | 1/20 | 128 | 138.15 | 1.08 | 96.6 |
| grad_curvature_r005 | 1/20 | 128 | 387.70 | 3.03 | 96.8 |
| context_only_centered | 1/20 | 128 | 386.70 | 3.02 | 99.1 |
| temporal_risk_scaled | 0/20 | 57 | 512.00 | 8.98 | 43.2 |
| grad_curvature_g010_r005 | 0/20 | 128 | 146.50 | 1.14 | 96.6 |
| grad_curvature_g0025_r015 | 0/20 | 128 | 218.70 | 1.71 | 96.7 |
| context_nogain_singleton | 0/20 | 128 | 128.00 | 1.00 | 98.9 |

Scores include one greedy completion per problem. Decoder output is truncated at
the first committed EOS and standard task stops, matching the existing protocol.
Committed positions can include unused output tails; they are not a quality metric.

## Paired HumanEval outcomes against EB

- context_smallgain: wins []; losses ['HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/13', 'HumanEval/16', 'HumanEval/18'].
- context_singleton: wins []; losses ['HumanEval/0', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- context_nogain: wins []; losses ['HumanEval/2', 'HumanEval/4', 'HumanEval/5', 'HumanEval/7', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- grad_centered: wins []; losses ['HumanEval/0', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/13', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- grad_risk_scaled: wins []; losses ['HumanEval/0', 'HumanEval/2', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- context_centered_r015: wins []; losses ['HumanEval/0', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/13', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- grad_curvature_nogain_singleton: wins []; losses ['HumanEval/0', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/7', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/13', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- grad_curvature_noinformation: wins []; losses ['HumanEval/0', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/7', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/13', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- grad_curvature_r005: wins []; losses ['HumanEval/0', 'HumanEval/2', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/7', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- context_only_centered: wins []; losses ['HumanEval/0', 'HumanEval/2', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/13', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- temporal_risk_scaled: wins []; losses ['HumanEval/0', 'HumanEval/2', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/7', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/13', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- grad_curvature_g010_r005: wins []; losses ['HumanEval/0', 'HumanEval/2', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/7', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/13', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- grad_curvature_g0025_r015: wins []; losses ['HumanEval/0', 'HumanEval/2', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/7', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/13', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].
- context_nogain_singleton: wins []; losses ['HumanEval/0', 'HumanEval/2', 'HumanEval/3', 'HumanEval/4', 'HumanEval/5', 'HumanEval/7', 'HumanEval/8', 'HumanEval/11', 'HumanEval/12', 'HumanEval/13', 'HumanEval/15', 'HumanEval/16', 'HumanEval/18'].

## Artifacts

- [latent_objective_arithmetic](/root/Strace/outputs/latent_objective_arithmetic)
- [latent_objective_gradient_arithmetic](/root/Strace/outputs/latent_objective_gradient_arithmetic)
- [latent_objective_refined_arithmetic](/root/Strace/outputs/latent_objective_refined_arithmetic)
- [latent_objective_context_arithmetic](/root/Strace/outputs/latent_objective_context_arithmetic)
- [latent_objective_temporal_arithmetic](/root/Strace/outputs/latent_objective_temporal_arithmetic)
- [latent_objective_context_control_arithmetic](/root/Strace/outputs/latent_objective_context_control_arithmetic)
- [latent_objective_humaneval20](/root/Strace/outputs/latent_objective_humaneval20)
- [latent_objective_context_humaneval20](/root/Strace/outputs/latent_objective_context_humaneval20)
- [latent_objective_context_control_humaneval20](/root/Strace/outputs/latent_objective_context_control_humaneval20)
