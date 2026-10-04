# Fixed-history causal audit

Lives in `schedulerlens/causal`. LLaDA and LLaDA-1.5 only (shared token space). Both roles score the same states reconstructed from stored LLaDA traces; the audit does not alter or regenerate those histories.

Histories are LLaDA traces. At each stored state, score with the predictor and the controller, apply the trace's scheduler rule, and compare the sets. The predictor chooses \(S_0\), and the controller chooses \(S_1\). For either role using LLaDA-1.5, EOS and EoT are removed only from the scores used to choose that role's set. M̂ and Ĵ stay on the raw distribution. `delta_ctrl_j` uses the predictor entropy on both sets. `delta_own_j` uses the controller entropy on the controller's set.

- `delta_pred_m`: same set \(S_0\), controller M̂ minus predictor M̂
- `delta_ctrl_m`: predictor M̂ on \(S_1\) minus predictor M̂ on \(S_0\)
- `delta_ctrl_j`: Ĵ of \(S_1\) minus Ĵ of \(S_0\), both under the predictor entropy
- `delta_own_j`: Ĵ of \(S_1\) under the controller entropy minus Ĵ of \(S_0\) under the predictor entropy

Also writes `jaccard`, the overlap of the predictor set and the controller set.

`--steps` keeps the trace whose step budget is the NFE-bucket representative for that model, scheduler, and task. A higher budget with the same NFE, such as DUS at 256 and 512, is skipped. Excluded tasks are removed before those buckets are formed. The replay restores the stored prompt's left-padding attention mask. LLaDA-1.5 EOS/EoT suppression applies only while constructing scheduler sets; M̂ and Ĵ always use each role's raw scores. Each file writes one full-trajectory row (`prefix_steps` empty) and one row for each default prefix 8, 16, 32, and 64. A prefix row averages steps with `step < K`, then averages those sample means. `n_samples` is the number of traces read. `n_steps_scored` is the number of steps inside that window. `--prefix-steps` replaces the default list, including when the audit is launched from `scripts/schedulerlens_compare.sh --command causal`.

Two runs, same traces: both roles LLaDA, then both roles LLaDA-1.5.

```bash
python -m schedulerlens.run causal \
  --repo schedulerlens-traces --namespace devichand \
  --predictor llada --controller llada \
  --steps 256 --exclude-task arc_challenge_chat \
  --device-map split \
  -o outputs/schedulerlens/causal_audit

python -m schedulerlens.run causal \
  --repo schedulerlens-traces --namespace devichand \
  --predictor llada1_5 --controller llada1_5 \
  --steps 256 --exclude-task arc_challenge_chat \
  --device-map split \
  -o outputs/schedulerlens/causal_audit_llada1_5
```
