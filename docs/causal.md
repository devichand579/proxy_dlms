# Causal predictor–controller experiments

Lives in `schedulerlens/causal`. LLaDA and LLaDA-1.5 only (shared token space). The predictor supplies committed token ids. The controller supplies the scores used to choose positions.

## Experiment 1 — fixed-history audit

Histories are LLaDA traces. At each stored state, score with the predictor and the controller, apply the trace's scheduler rule, and compare the sets. The controller's scores choose \(S_1\). When that controller is LLaDA-1.5, EOS and EoT are removed from those selection scores. M̂ and Ĵ stay on the raw distribution. `delta_ctrl_j` uses the predictor entropy on both sets. `delta_own_j` uses the controller entropy on the controller's set.

- `delta_pred_m`: same set \(S_0\), controller M̂ minus predictor M̂
- `delta_ctrl_m`: predictor M̂ on \(S_1\) minus predictor M̂ on \(S_0\)
- `delta_ctrl_j`: Ĵ of \(S_1\) minus Ĵ of \(S_0\), both under the predictor entropy
- `delta_own_j`: Ĵ of \(S_1\) under the controller entropy minus Ĵ of \(S_0\) under the predictor entropy

Also writes `jaccard`, the overlap of the predictor set and the controller set.

`--steps` keeps the trace whose step budget is the NFE-bucket representative for that model, scheduler, and task. A higher budget with the same NFE, such as DUS at 256 and 512, is skipped. Excluded tasks are removed before those buckets are formed. Each file writes one full-trajectory row (`prefix_steps` empty) and one row for each default prefix 8, 16, 32, and 64. A prefix row averages steps with `step < K`, then averages those sample means. `n_samples` is the number of traces read. `n_steps_scored` is the number of steps inside that window. `--prefix-steps` replaces the default list, including when the audit is launched from `scripts/schedulerlens_compare.sh --command causal`.

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

## Experiment 2 — full swap

Four cells: predictor × controller ∈ {llada, llada1_5}. The harness `model` is the predictor. `controller` is the controller. The W&B run and the trace are `{predictor}_{controller}_{pc_phi}_{task}_s{steps}`, for example `llada_llada1_5_eb_gsm8k_s256`. `run_id` replaces that name when set.

```bash
python hydra_runners/launch_eval.py \
  model=llada controller=llada1_5 mode=pc_swap pc_phi=eb \
  task_name=gsm8k steps=256 device_map=share --multirun
```

Sweep the sharp cohort with:

```bash
python hydra_runners/launch_eval.py \
  model=llada,llada1_5 controller=llada,llada1_5 \
  mode=pc_swap pc_phi=eb,bd \
  task_name=gsm8k,humaneval steps=256 --multirun
```

## Experiment 3 — adaptive selection

At each step both LLaDA and LLaDA-1.5 are forwarded. Each model, as controller, builds a set for `pd`, `bd`, `top_k_c`, `top_k_m`, `eb`, and `dus` (12 sets). Ĵ on a set is the controller that built it. Each model then supplies an M̂ on that set, giving 24 values of Ĝ = M̂ + Ĵ. The lowest Ĝ wins. The predictor is the model whose M̂ entered that Ĝ, and its argmax is written. The controller remains the model that built the set. LLaDA-1.5 drops EOS and EoT only from the scores that build its sets and from the token it writes. A set whose positions are already unmasked is skipped. M̂ and Ĵ use only the positions that are still masked.

```bash
python hydra_runners/launch_eval.py \
  model=llada mode=pc_adaptive \
  task_name=gsm8k steps=256 device_map=split
```

`model` is the harness checkpoint and is reused. The other of LLaDA and LLaDA-1.5 is loaded beside it. The W&B run and the checkpoint directory are `pc_adaptive_{task}_s{steps}`, for example `pc_adaptive_gsm8k_s128`. `device_map=split` places that other checkpoint on `cuda:1` when two GPUs are visible. `share` keeps both on the harness GPU. Swap still uses `device_map=split` to place the controller on `cuda:1`.

Each finished sample appends its per-step scheduler, predictor, and controller to `{task}_s{steps}_adaptive_choices.jsonl` in the run output directory. `{task}_s{steps}_adaptive_choices.csv` is rewritten from that log: one row per step, over the samples that still had masked tokens at that step. `scheduler_1` / `scheduler_2` / `scheduler_3` are the three most common winning schedulers. `predictor` and `controller` are the most common roles. Shares are that count divided by `n_samples` at the step. `g_hat`, `m_hat`, `m_hat_other`, `j_hat`, and `j_hat_other` are means over those samples. `g_hat` is the winning Ĝ. `m_hat` is the winning predictor's M̂ and `m_hat_other` is the other model's M̂ on the same set. `j_hat` is the winning controller's Ĵ, the one inside Ĝ. `j_hat_other` is the winning predictor's Ĵ on that set. A later line for the same sample replaces the earlier one, so a resumed run does not double-count.
