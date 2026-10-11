# Activation-only latent flow on HumanEval

Completed 20 canonical problems at a 128-step cap, 512 generation positions, greedy raw code completion, and matched normal batches of 20. Parameters were frozen before generation. Logits supply probe and committed values; no probabilities, confidence, entropy, or EB count enter the new scheduler.

| Method | Passed | Accuracy | Forward evaluations/problem | Outer steps used |
|---|---:|---:|---:|---:|
| eb_sampler | 13/20 | 65.00% | 128.00 | 128 |
| eb_target_temporal | 14/20 | 70.00% | 128.00 | 128 |
| latent_pca_flow | 4/20 | 20.00% | 3462.80 | 128 |

EB and temporal Fisher are previously completed matched benchmark controls. They do not satisfy the activation-only policy constraint. The new policy stops if its entire canvas is committed before the cap; no-op forwards are not added to pretend that more denoising occurred.

Novel policy generation time: 2563.0s; peak allocated memory: 118.40 GiB. Every intervention and unchanged control counts toward sequence work. No backward passes or auxiliary training.

Against eb_sampler: 0 wins, 9 losses. Wins: none. Losses: HumanEval/0, HumanEval/2, HumanEval/4, HumanEval/5, HumanEval/7, HumanEval/8, HumanEval/11, HumanEval/16, HumanEval/18.

Against eb_target_temporal: 0 wins, 10 losses. Wins: none. Losses: HumanEval/0, HumanEval/2, HumanEval/4, HumanEval/5, HumanEval/7, HumanEval/8, HumanEval/9, HumanEval/11, HumanEval/16, HumanEval/18.

The first 20 problems have been used for repeated exploratory tuning. This is one greedy completion per problem, not a generalization estimate.

[Settings and source hashes](/root/Strace/outputs/latent_pca_flow_humaneval20_lean/manifest.json) · [Completions and test outcomes](/root/Strace/outputs/latent_pca_flow_humaneval20_lean/latent_pca_flow/results.json) · [Paired comparison](/root/Strace/outputs/latent_pca_flow_humaneval20_lean/comparison.json)

Failure analysis: 10 completions have invalid syntax; 6 have valid syntax but fail tests. The method commits 290.80 positions/problem on average, versus 345.90 for EB and 350.10 for Fisher on these tasks. These full-canvas counts include unused tails and do not establish the cause of the accuracy loss. There are 414 forced singleton choices across 2560 sample-step decisions.

This candidate is rejected as a performance improvement. The paper-informed count proposal is a separate future experiment.
