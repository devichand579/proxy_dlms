# Decoded-body progress audit

Final output boundaries are used only for retrospective analysis. They never enter the scheduler.
Rendered-body readiness measures when the visible output tokens were committed; mask holes can
still exist inside that span. Safe EOS readiness requires every position up to the first
committed EOS to be known and is the stronger runtime-compatible condition.

| Method | Passed | Full forwards | Partial MLP backwards | Body multi-reveal events | Mean visible-body ready step | Mean body mask holes | Safe EOS examples |
|---|---:|---:|---:|---:|---:|---:|---:|
| activation_singleton | 7/20 | 128 | 0 | 0 | 79.85 | 12.65 | 2/20 |
| fisher_dispersion05 | 14/20 | 128 | 0 | 117 | 38.00 | 0.00 | 2/20 |
| fisher_singleton | 14/20 | 128 | 0 | 0 | 48.10 | 0.00 | 2/20 |
| activation_allow01_eos | 6/20 | 128 | 0 | 291 | 79.00 | 30.15 | 5/20 |
| context_flow_allow01 | 0/20 | 128 | 128 | 222 | 122.00 | 95.20 | 0/20 |
| context_flow_singleton | 0/20 | 128 | 128 | 0 | 120.75 | 153.30 | 0/20 |
| fisher_dispersion05_eos | 14/20 | 128 | 0 | 117 | 38.00 | 0.00 | 2/20 |
| fisher_dispersion05_frontier32 | 14/20 | 128 | 0 | 116 | 36.90 | 0.00 | 1/20 |
| fisher_flow_dispersion05 | 14/20 | 128 | 128 | 114 | 37.25 | 0.00 | 2/20 |
| fisher_flow_singleton005 | 14/20 | 128 | 128 | 0 | 47.50 | 0.00 | 2/20 |
| fisher_singleton_eos | 14/20 | 128 | 0 | 0 | 48.10 | 0.00 | 2/20 |
| fisher_singleton_frontier32 | 14/20 | 128 | 0 | 0 | 46.10 | 0.00 | 1/20 |
