# Activation-only scheduler experiments

Completed: 32 arithmetic canvases, 384 answer slots per seed, seeds 0 and 1, 12-step cap. All slots were filled. Canvases reuse component worksheets; these are correlated exploratory tests.

The new policies use no EB count, token confidence, token entropy, probability margin or learned head. EB and EB+Fisher are evaluation controls only. Greedy model proposals supply token values, and source alternatives are uniformly weighted. Scores and subset size use continuous activations.

| Study | Method | Seed 0 accuracy | Seed 1 accuracy | Forward evaluations/canvas, seeds 0 / 1 |
|---|---|---:|---:|---:|
| latent_residual_arithmetic | eb_sampler | 71.09% | 69.79% | 12.00 / 12.00 |
| latent_residual_arithmetic | eb_target_temporal | 73.96% | 70.05% | 12.00 / 12.00 |
| latent_residual_arithmetic | latent_residual_p0.25 | 61.20% | 65.89% | 96.00 / 96.00 |
| latent_residual_arithmetic | latent_residual_p0.5 | 63.28% | 67.45% | 46.81 / 48.31 |
| latent_residual_arithmetic | latent_residual_p0.75 | 64.58% | 69.79% | 28.12 / 28.12 |
| latent_residual_arithmetic | latent_residual_p1 | 64.58% | 69.79% | 27.00 / 27.00 |
| latent_survivors_arithmetic | latent_survivors_p0.25 | 61.98% | 64.58% | 96.00 / 95.75 |
| latent_survivors_arithmetic | latent_survivors_p0.5 | 63.28% | 67.71% | 47.00 / 47.62 |
| latent_survivors_arithmetic | latent_survivors_p0.75 | 64.58% | 69.79% | 28.12 / 28.12 |
| latent_survivors_arithmetic | latent_survivors_p1 | 64.58% | 69.79% | 27.00 / 27.00 |
| latent_spectral_survivors_arithmetic6 | latent_spectral_survivors_p0.5 | 63.80% | 63.80% | 50.38 / 51.50 |
| latent_spectral_survivors_arithmetic6 | latent_spectral_survivors_p0.75 | 63.54% | 63.02% | 27.69 / 27.50 |
| latent_flow_arithmetic12 | latent_flow_defect_w0 | 54.69% | 53.91% | 122.00 / 122.12 |
| latent_flow_arithmetic12 | latent_flow_defect_w0.15 | 54.43% | 54.43% | 132.00 / 134.25 |
| latent_pca_flow_arithmetic12 | latent_pca_flow_defect_w0 | 55.47% | 55.21% | 109.75 / 114.38 |

None of these activation-only candidates beats EB across both seeds. The best curvature settings score 64.58% / 69.79%, versus EB 71.09% / 69.79%. Directed latent flow scores 54.69% / 53.91%; PCA flow scores 55.47% / 55.21%. These results do not establish a better scheduler. They were not promoted to HumanEval.

Both scorer and count are independent of EB in the new modules. Their scores are not disguised log-confidence terms: integration tests rescale logits while preserving activations and proposed values and obtain identical counts and completions. Exact subset tests verify the spectral objective and independent-versus-coupled counts.

Positive latent smoothness or influence does not establish correct token commitments. The directed-flow model especially reduces Yes/No judgment accuracy compared with EB. All probes and controls count toward model work. Historical source snapshots preserve each tested implementation; later code changes do not retroactively alter the trials.

[Detailed results](/root/Strace/outputs/latent_only_ablation/results.csv)

[Residual design](/root/Strace/docs/latent-residual-scheduler.md) · [Directed flow design](/root/Strace/docs/latent-flow-scheduler.md)
