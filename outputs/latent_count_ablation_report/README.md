# Count-isolation experiments

HumanEval/0–19, 128 outer steps, 512-position canvas, greedy values, matched standalone batches of 20. The ranking formula stays fixed within each three-row group. Counts and ranking change later through their different generation histories. No references or tests enter decisions.

| Fixed scorer | Count rule | Passed | Forward evaluations/problem | Mean committed positions |
|---|---|---:|---:|---:|
| Earlier Fisher (diagnostic) | Fixed quota | 3/20 | 128 | 512.00 |
| Earlier Fisher (diagnostic) | Singleton | 14/20 | 128 | 128.00 |
| Earlier Fisher (diagnostic) | Latent growth guard | 14/20 | 1144 | 135.05 |
| Activation only | Fixed quota | 0/20 | 128 | 512.00 |
| Activation only | Singleton | 7/20 | 128 | 128.00 |
| Activation only | Latent growth guard | 8/20 | 1144 | 140.40 |

Matched historical controls: native EB 13/20; the earlier Fisher scorer with EB count 14/20. The earlier scorer uses confidence, so its new-count rows are diagnostic controls, not proposed activation-only schedulers.

The growth guard matches singleton quality with the stronger diagnostic scorer, with identical passed tasks. Its 14/20 does not establish a benefit from adaptive counts. With the activation-only scorer it has two wins and one loss against singleton (net +1), at 1144 versus 128 forward evaluations/problem. This small exploratory gain is not enough to beat EB or establish a robust improvement.

The new count itself uses model activations and decoder-gradient directions, not EB, entropy or token confidence. It probes candidate batches while keeping their last target masked, measuring the impact of the other proposed commitments on that target. The uncertainty scale and tolerance are empirical latent surrogates, not certified KL bounds.

[Controller design](/root/Strace/docs/latent-count-controller.md) · [Exact outcomes](/root/Strace/outputs/latent_count_ablation_report/analysis.json)

- [Earlier Fisher (diagnostic), Fixed quota](/root/Strace/outputs/latent_count_humaneval20/diagnostic_fisher_fixed/results.json)
- [Earlier Fisher (diagnostic), Singleton](/root/Strace/outputs/latent_count_singleton20/diagnostic_fisher_singleton/results.json)
- [Earlier Fisher (diagnostic), Latent growth guard](/root/Strace/outputs/latent_count_humaneval20/diagnostic_fisher_guard/results.json)
- [Activation only, Fixed quota](/root/Strace/outputs/latent_count_humaneval20/activation_fixed/results.json)
- [Activation only, Singleton](/root/Strace/outputs/latent_count_singleton20/activation_singleton/results.json)
- [Activation only, Latent growth guard](/root/Strace/outputs/latent_count_humaneval20/activation_guard/results.json)
