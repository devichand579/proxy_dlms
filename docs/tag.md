# Temporal Attribution Graph (TAG)

Implements a **prompt-specific, trajectory-specific** directed graph on top of trained CSTs for one decoding run, following the CLT-style construction:

- **Nodes**: sparse CST features `(i, t, f)` where the post–JumpReLU activation is positive; optional **input** nodes per response position; **out@pos** (readout / logits interface for that position) when you pass `model_path`. **feat → out** edges use `w = a^t_{i,f} · ⟨W^{dec}_{t,δ}[:,f], W_U[v,:]⟩` for every `t` in `[T_{\mathrm{eff}} - w + 1, …, T_{\mathrm{eff}}]` and each high-probability token id `v` (same pruned list as for `p_sink`), with `W_U` the model unembedding (`lm_head` / tied `wte`). If unembedding weights cannot be read, edges fall back to last-step-only with weight `a` only. Optional **error** nodes and **err → feature** edges as in the implementation notes.
- **Cross-step edges** \((i,t,f) \to (j,t+\delta,g)\): weight \(a^t_{i,f} \cdot V^{t\to t+\delta}_{f,g} \cdot a^{t+\delta}_{j,g}\), with \(V\) applied as a **sparse submatrix** \(W^{dec}_{t,\delta}[:,f_i]^\top W^{enc}_{t+\delta}[g_j,:]^\top\) so we never materialize full \(k\times k\) matrices.
- **Within-step lateral** (same \(t\), different positions): the CST only defines decoder offsets `1..w` across time, so the graph has no separate same-step lateral layer.
- **Pruning (default)**: `prune_tag_influence` — \((I-A^T)u = r\) with \(r\) on **out** from `p_sink`. **Optional** `prune_mode=bfs`: per-destination top `prune_p` of in-edges, **out** as sink.

## Run (full: generate + TAG + JSON + HTML)

**Shell (same pattern as `cst.sh` / `eval.sh`):** local GPU `./tag.sh ...`, or Slurm `sbatch tag.sh --slurm ...` (see `./tag.sh --help`).

```bash
./tag.sh \
  --cst_checkpoint outputs/cst_ckpt/cst_llada_scheduler_pd_k4096_w4_train_best.pt \
  --model_path ./models/llada \
  --out_dir outputs/tag_demo \
  --prompt "Your question here"
# Optional: match reference LLaDA-style capture (no EOS early stop / full-tail decode; same idea as eval `use_eos_harness=false`):
#   add --no_eos_harness   (see `python -m tag --help`)
```

Or Hydra only:

```bash
python launch_tag.py \
  model_path=./models/llada \
  cst_checkpoint=outputs/cst_ckpt/cst_llada_scheduler_pd_k4096_w4_train_best.pt \
  prompt="Your question here" \
  out_dir=outputs/tag_demo
```

Or only from an existing hidden chunk (same format as training/lm-eval capture):

```bash
python -m tag \
  --model_path ./models/llada \
  --cst_checkpoint path/to/cst_..._train_best.pt \
  --hidden_pt path/to/file.pt \
  --out_dir outputs/tag_from_pt
```

For **full** capture runs (`--prompt` + `--model_path`), optional **`--no_eos_harness`** turns off the EOS “harness” in the shared scheduler path (aligns with [Evaluation](evaluation.md) `use_eos_harness=false`).

Use `--no_logits` with `--hidden_pt` if you do not need output sink nodes (saves loading the backbone for a logits pass).

**Output sinks and EOS (interpretability):** by default, each `out@pos` node uses a **full-vocab** softmax of last-step logits. If EOS soaks mass and you care about **lexical** top tokens, enable **`--output_exclude_eos`** (CLI) or `output_exclude_eos=true` in `config/tag.yaml` / `launch_tag.py` — the readout **masks `eos_id` to \(-\infty\)** on that logit row before softmax (same renormalization idea as `cst_ntf_first_non_eos` for NTF). `eos_id` defaults to the checkpoint’s `eos_token_id` from `model_path` unless you set `eos_id=...` explicitly.

## Outputs

- `tag_trace.pt` — one `save_hidden_records` chunk (full run only).
- `tag_graph.json` — `nodes` / `edges` for custom tooling.
- `tag_view.html` — static **vis-network** page (requires network to load the CDN script once).

## Library

```python
from tag import run_tag_full, run_tag_from_chunk

run_tag_full(model_path, cst_ckpt, "prompt", "outputs/tag_x")
run_tag_from_chunk("hidden.pt", cst_ckpt, model_path=model_path)
```

## Configuration

Match `config/cst.yaml` `cst:` (`k`, `w`, `max_steps`, `d_model`) to the checkpoint. The scheduler short code used for training does not change the CST weights, but your capture should use a consistent `mode_key` / `steps` / `gen_length` when comparing to other runs.
