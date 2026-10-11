# Completed scheduler results

HumanEval results use greedy code completion. The first 20 tasks are reused for tuning; these scores do not establish generalization.
Current constraint: no trained scheduler heads or policies. Tables below contain only intrinsic methods computed from the frozen model. Historical learned-head results remain in the CSV files with eligible_intrinsic_only=False.
Batch-20 checks and packed exploratory runs are separate. The 256-position canvas is kept separate from the main 512-position comparison.

## One-forward three-term study

Activation and decoder-gradient risk, Gaussian information/dependence, and independent adaptive counts.
[Complete arithmetic/HumanEval results and limitations](/root/Strace/outputs/latent_objective_report/README.md).

## Cached-response and error-allowance count study

Fixed-ranking diagnostics, probability-free count arms, and separately labelled Fisher-geometry controls.
[Complete results, paired outcomes, and limitations](/root/Strace/outputs/latent_allowance_report/README.md).

## Useful code-body progress and local gradient scoring

Independent count diagnostics, corrected final-MLP gradients, and retrospective commitment timing.
[Outcome, work accounting, and paired comparisons](/root/Strace/outputs/latent_body_progress_report/README.md).

The latest requirement excludes EB counts and vocabulary confidence/entropy from both scheduler decisions. Historical intrinsic results below do not meet that stricter requirement. [Activation-only pilot results](/root/Strace/outputs/latent_only_ablation/README.md). The CSV records eligibility under both constraints.

## Full HumanEval-164, 128 steps, 512 positions

| Method | Passed | Accuracy | Wins / losses vs EB | Forward evaluations/problem |
|---|---:|---:|---:|---:|
| response_information_readiness | 56/164 | 34.15% | 10 / 1 | 250.34 |
| eb_target_temporal | 55/164 | 33.54% | 9 / 1 | 128.00 |
| eb_sampler | 47/164 | 28.66% | 0 / 0 | 128.00 |

[Full report, remaining-144 results, compute, and paired analysis](/root/Strace/outputs/latent_full_humaneval164/README.md).

Native EB reference: **13/20**, 128 steps, 512 positions. [Artifact](/root/Strace/outputs/latent_allowance_humaneval20/seed1234/eb_sampler/results.json).

## Standalone HumanEval-20, 128 steps, 512 positions

| Method | Passed | Wins / losses vs EB | Forward / full backward / partial backward evaluations | Remaining masks | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| fisher_dispersion05_frontier32 | 14/20 | 1 / 0 | 128.0 / 0.0 / 0.0 | 5348 | [Result](/root/Strace/outputs/latent_body_progress_corrected_humaneval20/seed1234/fisher_dispersion05_frontier32/results.json) |
| fisher_singleton_frontier32 | 14/20 | 1 / 0 | 128.0 / 0.0 / 0.0 | 7680 | [Result](/root/Strace/outputs/latent_body_progress_corrected_humaneval20/seed1234/fisher_singleton_frontier32/results.json) |
| fisher_dispersion05_eos | 14/20 | 1 / 0 | 128.0 / 0.0 / 0.0 | 5930 | [Result](/root/Strace/outputs/latent_body_progress_corrected_humaneval20/seed1234/fisher_dispersion05_eos/results.json) |
| fisher_singleton_eos | 14/20 | 1 / 0 | 128.0 / 0.0 / 0.0 | 7836 | [Result](/root/Strace/outputs/latent_body_progress_corrected_humaneval20/seed1234/fisher_singleton_eos/results.json) |
| fisher_flow_dispersion05 | 14/20 | 1 / 0 | 128.0 / 0.0 / 128.0 | 5420 | [Result](/root/Strace/outputs/latent_body_progress_corrected_humaneval20/seed1234/fisher_flow_dispersion05/results.json) |
| fisher_flow_singleton005 | 14/20 | 1 / 0 | 128.0 / 0.0 / 128.0 | 7680 | [Result](/root/Strace/outputs/latent_body_progress_corrected_humaneval20/seed1234/fisher_flow_singleton005/results.json) |
| fisher_dispersion05 | 14/20 | 1 / 0 | 128.0 / 0.0 / 0.0 | 5742 | [Result](/root/Strace/outputs/latent_body_progress_humaneval20/seed1234/fisher_dispersion05/results.json) |
| fisher_singleton | 14/20 | 1 / 0 | 128.0 / 0.0 / 0.0 | 7680 | [Result](/root/Strace/outputs/latent_body_progress_humaneval20/seed1234/fisher_singleton/results.json) |
| fisher_dispersion01 | 14/20 | 1 / 0 | 128.0 / 0.0 / 0.0 | 6953 | [Result](/root/Strace/outputs/latent_allowance_humaneval20/seed1234/fisher_dispersion01/results.json) |
| fisher_uniform_grounded025 | 14/20 | 1 / 0 | 128.0 / 0.0 / 0.0 | 7675 | [Result](/root/Strace/outputs/latent_allowance_humaneval20/seed1234/fisher_uniform_grounded025/results.json) |
| fisher_eb | 14/20 | 1 / 0 | 128.0 / 0.0 / 0.0 | 3238 | [Result](/root/Strace/outputs/latent_allowance_humaneval20/seed1234/fisher_eb/results.json) |
| diagnostic_fisher_singleton | 14/20 | 1 / 0 | 128.0 / 0.0 / 0.0 | 7680 | [Result](/root/Strace/outputs/latent_count_singleton20/diagnostic_fisher_singleton/results.json) |
| eb_target_temporal (beta=0.15) | 14/20 | 1 / 0 | 128.0 / 0.0 / 0.0 | 3238 | [Result](/root/Strace/outputs/latent_current_verified20/eb_target_temporal_b512_d0.25_s0.15_p8_v128/results.json) |
| eb_gradient_gain (gain=0.5, risk=1.0) | 14/20 | 1 / 0 | 142.9 / 14.9 / 0.0 | 2436 | [Result](/root/Strace/outputs/gradient_damage_verified20/eb_gradient_gain_b512_d0.25_s0.15_p8_v128_g0.5_cp16_ft_mb20_risk1/results.json) |
| eb_gradient_gain (gain=0.5, risk=3.0) | 14/20 | 1 / 0 | 143.4 / 15.4 / 0.0 | 2856 | [Result](/root/Strace/outputs/gradient_damage_verified20/eb_gradient_gain_b512_d0.25_s0.15_p8_v128_g0.5_cp16_ft_mb20_risk3/results.json) |
| finite_all_targets | 14/20 | 1 / 0 | 252.8 / 0.0 / 0.0 | 2902 | [Result](/root/Strace/outputs/latent_finite_verified20/finite_all_targets.json) |
| response_information_readiness | 14/20 | 1 / 0 | 253.6 / 0.0 / 0.0 | 3332 | [Result](/root/Strace/outputs/latent_information_readiness_verified20/response_information_readiness.json) |
| diagnostic_fisher_guard | 14/20 | 1 / 0 | 1144.0 / 0.0 / 0.0 | 7539 | [Result](/root/Strace/outputs/latent_count_humaneval20/diagnostic_fisher_guard/results.json) |
| fisher_relative4 | 13/20 | 1 / 1 | 128.0 / 0.0 / 0.0 | 7625 | [Result](/root/Strace/outputs/latent_allowance_humaneval20/seed1234/fisher_relative4/results.json) |
| eb_sampler | 13/20 | 0 / 0 | 128.0 / 0.0 / 0.0 | 3322 | [Result](/root/Strace/outputs/latent_allowance_humaneval20/seed1234/eb_sampler/results.json) |
| eb_cap1 | 13/20 | 0 / 0 | 128.0 / 0.0 / 0.0 | 7680 | [Result](/root/Strace/outputs/eb_singleton_verified20/eb_cap1_risk0.03.json) |
| eb_sampler | 13/20 | 0 / 0 | 128.0 / 0.0 / 0.0 | 3322 | [Result](/root/Strace/outputs/latent_current_verified20/eb_sampler_b512_d0.25_s0.15_p8_v128/results.json) |
| latent_eb (budget=0.03) | 13/20 | 0 / 0 | 128.0 / 0.0 / 0.0 | 4770 | [Result](/root/Strace/outputs/latent_count_verified20/latent_eb_risk0.03.json) |
| response_information_readiness | 13/20 | 1 / 1 | 2172.0 / 0.0 / 0.0 | 3131 | [Result](/root/Strace/outputs/latent_information_every_step20/response_information_readiness.json) |
| fisher_grounded_allow01 | 12/20 | 0 / 1 | 128.0 / 0.0 / 0.0 | 7228 | [Result](/root/Strace/outputs/latent_allowance_humaneval20/seed1234/fisher_grounded_allow01/results.json) |
| eb_target_temporal (beta=0.05) | 12/20 | 0 / 1 | 128.0 / 0.0 / 0.0 | 3151 | [Result](/root/Strace/outputs/latent_humaneval20/eb_target_temporal_b512_d0.25_s0.05_p8_v128/results.json) |
| eb_temporal (beta=0.15) | 12/20 | 1 / 2 | 128.0 / 0.0 / 0.0 | 1638 | [Result](/root/Strace/outputs/latent_humaneval20/eb_temporal_b512_d0.25_s0.15_p8_v128/results.json) |
| eb_gradient_gain (gain=0.5) | 12/20 | 2 / 3 | 143.1 / 15.1 / 0.0 | 2512 | [Result](/root/Strace/outputs/gradient_combined_verified20/eb_gradient_gain_b512_d0.25_s0.15_p8_v128_g0.5_cp16_ft_mb20/results.json) |
| eb_gradient_gain (gain=0.25) | 12/20 | 1 / 2 | 143.1 / 15.1 / 0.0 | 2686 | [Result](/root/Strace/outputs/latent_humaneval20/eb_gradient_gain_b512_d0.25_s0.15_p8_v128/results.json) |
| fisher_gradient_allow05 (gain=0.25) | 11/20 | 1 / 3 | 128.0 / 0.0 / 0.0 | 6208 | [Result](/root/Strace/outputs/latent_allowance_humaneval20/seed1234/fisher_gradient_allow05/results.json) |
| eb_gradient_gain (gain=0.5) | 11/20 | 0 / 2 | 143.3 / 15.3 / 0.0 | 3267 | [Result](/root/Strace/outputs/latent_humaneval20/eb_gradient_gain_b512_d0.25_s0.15_p8_v128_g0.5/results.json) |
| activation_guard | 8/20 | 1 / 6 | 1144.0 / 0.0 / 0.0 | 7432 | [Result](/root/Strace/outputs/latent_count_humaneval20/activation_guard/results.json) |
| activation_singleton | 7/20 | 0 / 6 | 128.0 / 0.0 / 0.0 | 7680 | [Result](/root/Strace/outputs/latent_body_progress_humaneval20/seed1234/activation_singleton/results.json) |
| activation_grounded025 | 7/20 | 0 / 6 | 128.0 / 0.0 / 0.0 | 7652 | [Result](/root/Strace/outputs/latent_allowance_humaneval20/seed1234/activation_grounded025/results.json) |
| activation_singleton | 7/20 | 0 / 6 | 128.0 / 0.0 / 0.0 | 7680 | [Result](/root/Strace/outputs/latent_count_singleton20/activation_singleton/results.json) |
| gradient_gain (gain=0.5) | 7/20 | 1 / 7 | 144.0 / 16.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_humaneval20/gradient_gain_b512_d0.25_s0.15_p8_v128_g0.5/results.json) |
| activation_allow01_eos | 6/20 | 0 / 7 | 128.0 / 0.0 / 0.0 | 7302 | [Result](/root/Strace/outputs/latent_body_progress_corrected_humaneval20/seed1234/activation_allow01_eos/results.json) |
| activation_dispersion01 | 6/20 | 0 / 7 | 128.0 / 0.0 / 0.0 | 7213 | [Result](/root/Strace/outputs/latent_allowance_humaneval20/seed1234/activation_dispersion01/results.json) |
| activation_allow01 | 6/20 | 0 / 7 | 128.0 / 0.0 / 0.0 | 6747 | [Result](/root/Strace/outputs/latent_allowance_humaneval20/seed1234/activation_allow01/results.json) |
| gradient_gain (block=64, gain=0.25) | 5/20 | 1 / 9 | 144.0 / 16.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_humaneval20/gradient_gain_b64_d0.25_s0.15_p8_v128/results.json) |
| context_smallgain | 4/20 | 0 / 9 | 128.0 / 0.0 / 0.0 | 5845 | [Result](/root/Strace/outputs/latent_objective_context_control_humaneval20/seed1234/context_smallgain/results.json) |
| gradient_gain (gain=0.25) | 4/20 | 0 / 9 | 144.0 / 16.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_humaneval20/gradient_gain_b512_d0.25_s0.15_p8_v128/results.json) |
| response_gain | 4/20 | 0 / 9 | 192.0 / 0.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_humaneval20/response_gain_b512_d0.25_s0.15_p32_v128/results.json) |
| latent_pca_flow | 4/20 | 0 / 9 | 3462.8 / 0.0 / 0.0 | 4424 | [Result](/root/Strace/outputs/latent_pca_flow_humaneval20_lean/latent_pca_flow/results.json) |
| context_nogain | 3/20 | 0 / 10 | 128.0 / 0.0 / 0.0 | 6373 | [Result](/root/Strace/outputs/latent_objective_context_control_humaneval20/seed1234/context_nogain/results.json) |
| context_singleton | 3/20 | 0 / 10 | 128.0 / 0.0 / 0.0 | 7680 | [Result](/root/Strace/outputs/latent_objective_context_humaneval20/seed1234/context_singleton/results.json) |
| diagnostic_fisher_fixed | 3/20 | 0 / 10 | 128.0 / 0.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_count_humaneval20/diagnostic_fisher_fixed/results.json) |
| eb_sampler (block=64) | 3/20 | 0 / 10 | 128.0 / 0.0 / 0.0 | 4704 | [Result](/root/Strace/outputs/latent_humaneval20/eb_sampler_b64_d0.25_s0.15_p8_v128/results.json) |
| confidence (block=64) | 3/20 | 0 / 10 | 128.0 / 0.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_humaneval20/confidence_b64_d0.25_s0.15_p8_v128/results.json) |
| temporal (beta=0.15) | 3/20 | 0 / 10 | 128.0 / 0.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_humaneval20/temporal_b512_d0.25_s0.15_p8_v128/results.json) |
| confidence | 3/20 | 0 / 10 | 128.0 / 0.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_humaneval20/confidence_b512_d0.25_s0.15_p8_v128/results.json) |
| hybrid_probe | 3/20 | 0 / 10 | 144.0 / 0.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_humaneval20/hybrid_probe_b512_d0.25_s0.15_p8_v128/results.json) |
| grad_centered | 2/20 | 0 / 11 | 128.0 / 0.0 / 0.0 | 6170 | [Result](/root/Strace/outputs/latent_objective_context_humaneval20/seed1234/grad_centered/results.json) |
| grad_risk_scaled | 2/20 | 0 / 11 | 128.0 / 0.0 / 0.0 | 6605 | [Result](/root/Strace/outputs/latent_objective_context_humaneval20/seed1234/grad_risk_scaled/results.json) |
| context_centered_r015 | 2/20 | 0 / 11 | 128.0 / 0.0 / 0.0 | 3076 | [Result](/root/Strace/outputs/latent_objective_context_humaneval20/seed1234/context_centered_r015/results.json) |
| fisher_diversity | 2/20 | 0 / 11 | 128.0 / 0.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_humaneval20/fisher_diversity_b512_d0.25_s0.15_p8_v128/results.json) |
| entropy | 2/20 | 0 / 11 | 128.0 / 0.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_humaneval20/entropy_b512_d0.25_s0.15_p8_v128/results.json) |
| context_only_centered | 1/20 | 0 / 12 | 128.0 / 0.0 / 0.0 | 2506 | [Result](/root/Strace/outputs/latent_objective_context_humaneval20/seed1234/context_only_centered/results.json) |
| grad_curvature_r005 | 1/20 | 0 / 12 | 128.0 / 0.0 / 0.0 | 2486 | [Result](/root/Strace/outputs/latent_objective_humaneval20/seed1234/grad_curvature_r005/results.json) |
| grad_curvature_nogain_singleton | 1/20 | 0 / 12 | 128.0 / 0.0 / 0.0 | 7680 | [Result](/root/Strace/outputs/latent_objective_humaneval20/seed1234/grad_curvature_nogain_singleton/results.json) |
| grad_curvature_noinformation | 1/20 | 0 / 12 | 128.0 / 0.0 / 0.0 | 7477 | [Result](/root/Strace/outputs/latent_objective_humaneval20/seed1234/grad_curvature_noinformation/results.json) |
| raw_diversity | 1/20 | 0 / 12 | 128.0 / 0.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_humaneval20/raw_diversity_b512_d0.25_s0.15_p8_v128/results.json) |
| temporal_risk_scaled (beta=None) | 0/20 | 0 / 13 | 57.0 / 0.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_objective_context_humaneval20/seed1234/temporal_risk_scaled/results.json) |
| context_flow_allow01 | 0/20 | 0 / 13 | 128.0 / 0.0 / 128.0 | 6303 | [Result](/root/Strace/outputs/latent_body_progress_corrected_humaneval20/seed1234/context_flow_allow01/results.json) |
| context_flow_singleton | 0/20 | 0 / 13 | 128.0 / 0.0 / 128.0 | 7680 | [Result](/root/Strace/outputs/latent_body_progress_corrected_humaneval20/seed1234/context_flow_singleton/results.json) |
| context_nogain_singleton | 0/20 | 0 / 13 | 128.0 / 0.0 / 0.0 | 7680 | [Result](/root/Strace/outputs/latent_objective_context_control_humaneval20/seed1234/context_nogain_singleton/results.json) |
| grad_curvature_g0025_r015 | 0/20 | 0 / 13 | 128.0 / 0.0 / 0.0 | 5866 | [Result](/root/Strace/outputs/latent_objective_humaneval20/seed1234/grad_curvature_g0025_r015/results.json) |
| grad_curvature_g010_r005 | 0/20 | 0 / 13 | 128.0 / 0.0 / 0.0 | 7310 | [Result](/root/Strace/outputs/latent_objective_humaneval20/seed1234/grad_curvature_g010_r005/results.json) |
| activation_fixed | 0/20 | 0 / 13 | 128.0 / 0.0 / 0.0 | 0 | [Result](/root/Strace/outputs/latent_count_humaneval20/activation_fixed/results.json) |

Settings and every completed packed/standalone trial are in [humaneval.csv](/root/Strace/outputs/scheduler_scoreboard/humaneval.csv).
Remaining masks include unused tail positions; they are not equivalent to incomplete code.

## Latest extended arithmetic checks

| Seed | Method | Answer accuracy | Delta vs EB | Worksheet accuracy |
| ---: | --- | ---: | ---: | ---: |
| 0 | eb_sampler | 71.09% | +0.00 pp | 34.38% |
| 0 | confidence | 71.61% | +0.52 pp | 35.94% |
| 0 | eb_target_temporal | 73.96% | +2.86 pp | 39.06% |
| 0 | eb_gradient_gain | 74.22% | +3.12 pp | 42.97% |
| 0 | motion_count | 71.09% | +0.00 pp | 34.38% |
| 0 | motion_both | 73.96% | +2.86 pp | 39.06% |
| 0 | control_both | 73.96% | +2.86 pp | 39.06% |
| 1 | eb_sampler | 69.79% | +0.00 pp | 38.28% |
| 1 | confidence | 69.01% | -0.78 pp | 35.94% |
| 1 | eb_target_temporal | 70.05% | +0.26 pp | 35.16% |
| 1 | eb_gradient_gain | 64.58% | -5.21 pp | 34.38% |
| 1 | motion_count | 69.01% | -0.78 pp | 35.94% |
| 1 | motion_both | 70.05% | +0.26 pp | 35.16% |
| 1 | control_both | 70.05% | +0.26 pp | 35.16% |

All arithmetic studies: [arithmetic.csv](/root/Strace/outputs/scheduler_scoreboard/arithmetic.csv). Canvases share component worksheets, so their outcomes are correlated.

## Actual-reveal arithmetic checks

| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |
| ---: | --- | ---: | ---: | ---: |
| 0 | eb_sampler | 71.09% | 34.38% | 12.00 |
| 0 | eb_target_temporal | 73.96% | 39.06% | 12.00 |
| 0 | finite_all_targets | 74.74% | 41.41% | 92.56 |
| 0 | finite_balanced_targets | 75.00% | 40.62% | 92.56 |
| 0 | finite_balanced_sources_targets | 74.74% | 38.28% | 92.56 |
| 1 | eb_sampler | 69.79% | 38.28% | 12.00 |
| 1 | eb_target_temporal | 70.05% | 35.16% | 12.00 |
| 1 | finite_all_targets | 71.61% | 38.28% | 93.56 |
| 1 | finite_balanced_targets | 71.88% | 39.84% | 93.69 |
| 1 | finite_balanced_sources_targets | 73.44% | 39.06% | 93.94 |

## Multi-value information arithmetic checks

| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |
| ---: | --- | ---: | ---: | ---: |
| 0 | eb_sampler | 71.09% | 34.38% | 12.00 |
| 0 | eb_target_temporal | 73.96% | 39.06% | 12.00 |
| 0 | response_information (candidates=4) | 74.48% | 44.53% | 174.25 |
| 0 | response_information_risk (candidates=4) | 73.96% | 41.41% | 176.12 |
| 0 | response_information_readiness (candidates=4) | 77.08% | 45.31% | 175.62 |
| 1 | eb_sampler | 69.79% | 38.28% | 12.00 |
| 1 | eb_target_temporal | 70.05% | 35.16% | 12.00 |
| 1 | response_information (candidates=4) | 69.53% | 32.03% | 177.12 |
| 1 | response_information_risk (candidates=4) | 75.00% | 46.09% | 177.00 |
| 1 | response_information_readiness (candidates=4) | 78.91% | 53.12% | 177.12 |
| 0 | response_information_readiness (candidates=12) | 75.26% | 40.62% | 316.75 |
| 1 | response_information_readiness (candidates=12) | 77.60% | 49.22% | 314.50 |

## Source-support and count arithmetic checks

| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |
| ---: | --- | ---: | ---: | ---: |
| 0 | eb_sampler | 71.09% | 34.38% | 12.00 |
| 0 | eb_target_temporal | 73.96% | 39.06% | 12.00 |
| 0 | response_information_readiness | 77.08% | 45.31% | 175.62 |
| 0 | response_source_weighted | 72.14% | 39.06% | 176.88 |
| 0 | response_count_only | 73.96% | 39.06% | 172.62 |
| 0 | response_weighted_count | 72.40% | 39.84% | 170.38 |
| 1 | eb_sampler | 69.79% | 38.28% | 12.00 |
| 1 | eb_target_temporal | 70.05% | 35.16% | 12.00 |
| 1 | response_information_readiness | 78.91% | 53.12% | 177.12 |
| 1 | response_source_weighted | 73.18% | 39.84% | 176.62 |
| 1 | response_count_only | 70.05% | 35.16% | 177.00 |
| 1 | response_weighted_count | 72.14% | 39.06% | 173.88 |

## Within-forward depth arithmetic checks

| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |
| ---: | --- | ---: | ---: | ---: |
| 0 | eb_sampler | 71.09% | 34.38% | 12.00 |
| 0 | eb_target_temporal | 73.96% | 39.06% | 12.00 |
| 0 | depth_motion_risk | 72.66% | 39.84% | 12.00 |
| 0 | depth_signed_support | 64.84% | 21.88% | 12.00 |
| 0 | depth_support_risk | 69.79% | 27.34% | 12.00 |
| 1 | eb_sampler | 69.79% | 38.28% | 12.00 |
| 1 | eb_target_temporal | 70.05% | 35.16% | 12.00 |
| 1 | depth_motion_risk | 74.48% | 42.97% | 12.00 |
| 1 | depth_signed_support | 62.24% | 28.12% | 12.00 |
| 1 | depth_support_risk | 74.74% | 39.84% | 12.00 |

## Combined information and depth arithmetic checks

| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |
| ---: | --- | ---: | ---: | ---: |
| 0 | eb_sampler | 71.09% | 34.38% | 12.00 |
| 0 | eb_target_temporal | 73.96% | 39.06% | 12.00 |
| 0 | response_information_readiness | 77.08% | 45.31% | 175.62 |
| 0 | information_depth005 | 77.34% | 45.31% | 176.50 |
| 0 | information_depth015 | 76.30% | 45.31% | 175.62 |
| 1 | eb_sampler | 69.79% | 38.28% | 12.00 |
| 1 | eb_target_temporal | 70.05% | 35.16% | 12.00 |
| 1 | response_information_readiness | 78.91% | 53.12% | 177.12 |
| 1 | information_depth005 | 78.12% | 50.78% | 176.50 |
| 1 | information_depth015 | 74.22% | 45.31% | 174.75 |

## Unresolved-attention arithmetic checks

| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |
| ---: | --- | ---: | ---: | ---: |
| 0 | eb_sampler | 71.09% | 34.38% | 12.00 |
| 0 | eb_target_temporal | 73.96% | 39.06% | 12.00 |
| 0 | attention_mask_read | 75.00% | 39.84% | 12.00 |
| 0 | attention_uncertain_read015 | 75.26% | 40.62% | 12.00 |
| 0 | attention_uncertain_read03 | 72.92% | 39.06% | 12.00 |
| 1 | eb_sampler | 69.79% | 38.28% | 12.00 |
| 1 | eb_target_temporal | 70.05% | 35.16% | 12.00 |
| 1 | attention_mask_read | 66.41% | 31.25% | 12.00 |
| 1 | attention_uncertain_read015 | 64.58% | 30.47% | 12.00 |
| 1 | attention_uncertain_read03 | 64.58% | 28.91% | 12.00 |

Only completed, scored artifacts are included; this report does not assert that pending jobs have finished.
