# Full HumanEval comparison

Frozen LLaDA-8B, all 164 canonical problems, greedy raw code completion, 512 generation positions, 128 outer denoising steps, matched batches of 20 (last batch 4). No auxiliary training or learned scheduler policy. Correctness uses the same 3-second evaluator as the earlier runs.

The information scheduler uses additional model evaluations inside the 128 steps; compute is reported separately.

| Method | Full 164 | First 20 used for tuning | Remaining 144 | Forward evaluations/problem | Generation seconds | Peak GPU GiB |
|---|---:|---:|---:|---:|---:|---:|
| EB | 47/164 (28.66%) | 13/20 (65.00%) | 34/144 (23.61%) | 128.00 | 1282.4 | 21.80 |
| EB + temporal Fisher | 55/164 (33.54%) | 14/20 (70.00%) | 41/144 (28.47%) | 128.00 | 1055.2 | 21.95 |
| Information + risk + incoming sensitivity | 56/164 (34.15%) | 14/20 (70.00%) | 42/144 (29.17%) | 250.34 | 1927.3 | 74.05 |

## Paired outcomes

| Comparison against EB | Split | Wins | Losses | Net | Exact paired p-value |
|---|---|---:|---:|---:|---:|
| EB + temporal Fisher | all_164 | 9 | 1 | +8 | 0.0215 |
| Information + risk + incoming sensitivity | all_164 | 10 | 1 | +9 | 0.0117 |
| EB + temporal Fisher | remaining_144 | 8 | 1 | +7 | 0.0391 |
| Information + risk + incoming sensitivity | remaining_144 | 9 | 1 | +8 | 0.0215 |
| EB + temporal Fisher | tuned_first_20 | 1 | 0 | +1 | 1.0000 |
| Information + risk + incoming sensitivity | tuned_first_20 | 1 | 0 | +1 | 1.0000 |

## Failure breakdown

| Method | Invalid syntax | Valid syntax but failed tests | Passed |
|---|---:|---:|---:|
| EB | 25 | 92 | 47 |
| EB + temporal Fisher | 14 | 95 | 55 |
| Information + risk + incoming sensitivity | 14 | 94 | 56 |

## Commitment counts

| Method | Mean positions committed out of 512 | Mean per outer step |
|---|---:|---:|
| EB | 313.88 | 2.45 |
| EB + temporal Fisher | 342.57 | 2.68 |
| Information + risk + incoming sensitivity | 347.65 | 2.72 |

These counts include the full canvas, including positions after the decoded end of a function. They measure scheduler behavior, not completion quality.


## Interpretation

Highest accuracy: **Information + risk + incoming sensitivity**, 56/164. Ties are resolved by lower forward-evaluation cost.

Temporal Fisher changes the position ranking while retaining EB’s entropy-based count rule. Its measured change against EB is +8 problems (+4.88 percentage points) overall, and +7/144 outside the tuning set. Both use 128 forward sequence evaluations per problem. The entropy rule is unchanged, but a different ranking can naturally yield different accepted prefix lengths.

Of Fisher’s 9 wins against EB, 5 repaired invalid syntax, while 4 had valid syntax under EB but failed tests. Syntax validity alone does not imply correct reasoning or successful execution.

The information method changes accuracy by +1 problems relative to Fisher, at 1.96× the forward sequence evaluations. Its extra work includes both intervention branches and unchanged controls; the number of outer denoising steps remains 128.

The absolute scores on all 164 problems should replace the small-set scores when judging code performance. Inspect paired wins and losses before attributing gains to fewer unfinished masks: unused masked tail positions can coexist with a correct decoded function.


The first 20 problems were repeatedly used to choose these settings. The remaining 144 were not used in those earlier scheduler searches. This is one deterministic completion per problem; confidence intervals and paired tests describe uncertainty across problems, not variability across generation seeds. No multiple-comparison correction is applied. Batch shape can affect near-tied ROCm predictions, so all methods use the same normal-pass batching.

[Per-problem outcomes](per_task.csv) · [Detailed analysis](analysis.json) · [Generation settings and source hashes](manifest.json)

Evaluator controls: 164/164 reference solutions passed; 0/164 deliberately failing solutions passed. Control solutions were never provided to generation. [Validation](evaluator_validation164.json)
