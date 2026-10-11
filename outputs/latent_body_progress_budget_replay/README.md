# Correctness at saved earlier steps

These canvases replay recorded commits; reference answers never enter generation.
This measures a fixed earlier-step exit on the saved trajectory, not an oracle
stopping policy. Partial backward work remains necessary for context-flow arms.

| Method | Step 32 | Step 48 | Step 64 | Step 96 | Step 128 |
|---|---:|---:|---:|---:|---:|
| activation_allow01_eos | 3/20 | 3/20 | 6/20 | 6/20 | 6/20 |
| activation_singleton | 2/20 | 2/20 | 5/20 | 5/20 | 7/20 |
| context_flow_allow01 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| context_flow_singleton | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| fisher_dispersion05 | 7/20 | 13/20 | 13/20 | 14/20 | 14/20 |
| fisher_dispersion05_eos | 7/20 | 13/20 | 13/20 | 14/20 | 14/20 |
| fisher_dispersion05_frontier32 | 7/20 | 13/20 | 14/20 | 14/20 | 14/20 |
| fisher_flow_dispersion05 | 9/20 | 13/20 | 13/20 | 14/20 | 14/20 |
| fisher_flow_singleton005 | 6/20 | 10/20 | 13/20 | 13/20 | 14/20 |
| fisher_singleton | 6/20 | 9/20 | 13/20 | 13/20 | 14/20 |
| fisher_singleton_eos | 6/20 | 9/20 | 13/20 | 13/20 | 14/20 |
| fisher_singleton_frontier32 | 6/20 | 9/20 | 14/20 | 14/20 | 14/20 |
