# Temporal Attribution Graph (TAG)

TAG builds a **prompt-specific, trajectory-specific** directed graph on top of a trained [CST](cst.md) bank for **one** diffusion decoding run. The goal is interpretability: which sparse CST features at which positions and timesteps influence which other features and, optionally, which vocabulary tokens at the readout.

The construction follows a **CLT-style** bilinear attribution picture: edges are products of activations and learned virtual couplings between feature indices, not full dense hidden-space Jacobians.

---

## Prerequisites


| Requirement                                                     | Why                                                        |
| --------------------------------------------------------------- | ---------------------------------------------------------- |
| **CST checkpoint** (`cst_..._train_best.pt`)                    | Sparse bank weights (`k`, `w`, `max_steps`, `d_model`)     |
| **Matching CST config** (`config/cst.yaml` or `cst_config=...`) | Bank shape must match the checkpoint                       |
| **Hidden trace** `H`                                            | Shape `[T, B, L, d]` from capture or `hidden_pt`           |
| **Local LM** (`model_path`)                                     | Full runs (capture) and readout sinks (logits at `H[T-1]`) |


TAG does not train CSTs; it **reads** a frozen bank and one trajectory.

---

## CST steps on a trajectory

A hidden trace `H` has rows `H[0]` (initial) through `H[T-1]` (after the last recorded forward). Indexing matches `cst.train.hidden_chunk` and eval: `H[k]` is the state **after** `k` forwards.

### What one CST step does

At **CST step** `t` (bank slot `t`):

1. **Encode** the hidden **input to step** `t`:

```
a^t = JumpReLU( W^enc_t * H[t-1] )
```

Tensor shape: `[B, L, k]`.

1. **Decode** with offsets `delta` in `1 .. w`: matrix `W^dec_{t,delta}` maps features toward future rows. Offset `delta` contributes to hidden row `H[t + delta - 1]`.

So step `t` reads the representation **entering** step `t` and explains part of the representations at rows `t, t+1, ... t+w-1`. This is the same window rule as CST training (`cst/train/loss.py`) and eval (`cst/eval/run_eval_cst.py`).

### Sparse features as graph units

Only **active** features (`a^t[i,f] > 0` after JumpReLU) become nodes. The dictionary size is `k`, but the graph stays sparse because activations are sparse.

TAG runs CST steps `t = 1, 2, ... T` on a trace of length `T`, so the last row `H[T-1]` is encoded at step `T` and can participate in cross-step edges and readout paths.

---

## TAG graph definition

The graph is **directed**. Edge weights are signed scalars (products of activations and weights); pruning uses `abs(weight)`.

### Node types


| Kind     | ID pattern      | Meaning                                                                                          |
| -------- | --------------- | ------------------------------------------------------------------------------------------------ |
| **in**   | `in_{i}`        | Source at response position `i` (`prompt_len <= i < L`). Feeds step-1 features only.             |
| **feat** | `f_{i}_{t}_{f}` | CST feature `f` at position `i`, step `t`, activation `a^t[i,f]`.                                |
| **out**  | `out_{i}`       | Readout sink at position `i`: softmax over vocab from last hidden `H[T-1]` (needs `model_path`). |
| **err**  | `err_{i}_{t}`   | Optional reconstruction residual (needs `with_residual=true`).                                   |


Positions `i < prompt_len` are excluded from response-region nodes and cross-position edges.

### Edge types and weights

**1. Input -> feature (step 1)**

```
in_i  -->  (i, 1, f)     weight w = a^1[i,f]
```

**2. Cross-step feature -> feature**

For `delta` in `1 .. w` and `t + delta <= T_eff`:

```
(i, t, f)  -->  (j, t+delta, g)

w = a^t[i,f] * V^{t -> t+delta}[f,g] * a^{t+delta}[j,g]
```

Virtual coupling (only a small block is built in memory, not full `k x k`):

```
V[f,g] = dot( W^dec_{t,delta}[:,f],  W^enc_{t+delta}[g,:] )
```

**3. Feature -> output (readout)**

When `model_path` and unembedding `W_U` (`lm_head` or tied `wte`) are available: for each high-probability token `v` at position `i`, each step `t` from `T_eff - w + 1` to `T_eff`, and decoder offset `delta = T_eff - t + 1`:

```
(i, t, f)  -->  out_i

w = a^t[i,f] * dot( W^dec_{t,delta}[:,f],  W_U[v,:] )
```

If unembedding is unavailable, a fallback links **last-step** features to `out_i` with `w = a^{T_eff}[i,f]` only.

**4. Error -> feature (optional)**

With `with_residual=true`, at each anchor:

```
eps = H[anchor] - H_hat[anchor]
```

Error nodes connect to features using `eps` in place of `a^t[i,f]` on the source side of the bilinear form (one edge per active pair `(f, g)`).

### What TAG does not include

- **No within-step lateral layer** at the same `t` between positions: CST only couples time via decoder offsets `1..w`, not arbitrary same-timestep position mixing in the bank.
- **No full LM Jacobian**: attribution is through the CST bottleneck and explicit virtual weights, not layer-by-layer backprop through the backbone.

---

## Theory: influence pruning

The raw graph can be large (many active features × positions × steps). **Influence pruning** keeps a high-mass subgraph.

### Transition matrix

Build matrix `A` from absolute edge weights:

```
for each edge src --> dst with weight w:
    A[dst, src] += abs(w)
```

Then **column-normalize** each column (each source column sums to 1). Add a small **uniform mix** (`mix_uniform`) for stability.

### Source vector `r`

- **Default:** `out` nodes. Each `out@i` gets mass `p_sink` = sum of softmax probabilities of the top tokens kept for that position (controlled by `top_output_mass` and `max_output_tokens`), normalized over all `out` nodes.
- **No logits** (`no_logits=true`): uniform source on **in** nodes instead.

### Influence scores

Solve the linear system:

```
(I - A^T) * u = r
```

Use **non-negative** node scores: `max(u - r, 0)`.

### What is kept

1. **Nodes:** All `in`, `out`, `err`. Among **feat**, keep the smallest set whose cumulative influence share is at least `influence_node_mass` (default 0.8).
2. **Edges:** On the induced subgraph, recompute influence and keep edges until cumulative edge-mass share is at least `influence_edge_mass` (default 0.98).

Prune metadata in the run JSON includes `source` (`out` or `in`), node/edge counts before and after.

---

## Pipeline


| Stage       | Module                   | Role                                                          |
| ----------- | ------------------------ | ------------------------------------------------------------- |
| Capture     | `tag/capture.py`         | One prompt -> `tag_trace.pt` via `collect_hidden_batch_loaded` |
| Activations | `tag/activations.py`     | `a^t` from `H[t-1]`; optional residuals                       |
| Graph       | `tag/graph.py`           | Nodes, edges, virtual submatrices                             |
| Prune       | `tag/influence_prune.py` | Influence propagation + mass cuts                             |
| Export      | `tag/viz.py`             | JSON + vis-network HTML                                       |


Entry points: `hydra_runners/run_tag.py`, `./scripts/tag.sh`, `python -m tag`.

---

## Configuration (`config/tag.yaml`)

### Paths and run mode


| Key              | Default          | Meaning                                                   |
| ---------------- | ---------------- | --------------------------------------------------------- |
| `cst_checkpoint` | `null`           | **Required** at runtime. Path to `cst_..._train_best.pt`. |
| `cst_config`     | `null`           | Optional CST yaml; else `config/cst.yaml`.                |
| `model_path`     | `./models/llada` | Local LM for capture and/or logits.                       |
| `prompt`         | `null`           | Full run: text to generate from.                          |
| `hidden_pt`      | `null`           | Chunk run: existing `.pt` (skips generation).             |
| `out_dir`        | `outputs_tag`    | Output directory.                                         |


Set `**prompt`** (full) or `**hidden_pt**` (chunk), not neither.

### Capture (full run only)


| Key               | Default | Meaning                                                                            |
| ----------------- | ------- | ---------------------------------------------------------------------------------- |
| `mode_key`        | `pd`    | Scheduler mode; keep consistent with CST capture for comparisons.                  |
| `gen_length`      | `256`   | Generated token count.                                                             |
| `steps`           | `256`   | Diffusion / decode steps.                                                          |
| `use_eos_harness` | `true`  | `false` = full step budget, no EOS early stop (like eval `use_eos_harness=false`). |
| `device`          | `cuda`  | Torch device.                                                                      |


### Pruning


| Key                   | Default  | Meaning                                  |
| --------------------- | -------- | ---------------------------------------- |
| `influence_node_mass` | `0.8`    | Fraction of feat influence mass to keep. |
| `influence_edge_mass` | `0.98`   | Fraction of edge influence mass to keep. |
| `mix_uniform`         | `0.0001` | Uniform damping in the influence matrix. |


### Graph options


| Key             | Default | Meaning                                                              |
| --------------- | ------- | -------------------------------------------------------------------- |
| `with_residual` | `false` | Add `err` nodes and err->feat edges from CST reconstruction residual. |
| `write_html`    | `true`  | Write `tag_view.html`.                                               |


### Readout


| Key                  | Default    | Meaning                                                            |
| -------------------- | ---------- | ------------------------------------------------------------------ |
| `no_logits`          | `false`    | Chunk-only: skip LM load; no `out` nodes; prune from `in` sources. |
| `top_output_mass`    | `0.95`     | Softmax mass cutoff for top tokens per position.                   |
| `max_output_tokens`  | `10`       | Max tokens per position for sinks and feat->out.                    |
| `output_exclude_eos` | `false`    | Mask EOS to negative infinity before softmax on readout rows.      |
| `eos_id`             | (optional) | Override EOS id; else inferred from `model_path`.                  |


---

## How to run

### Full run (generate + TAG + JSON + HTML)

```bash
./scripts/tag.sh \
  --cst_checkpoint outputs/cst_ckpt/cst_llada_pd_512_256_k4096_w4_ls0.001_best.pt \
  --model_path ./models/llada \
  --out_dir outputs/tag_demo \
  --prompt "Your question here"
```

Hydra:

```bash
python hydra_runners/run_tag.py \
  model_path=./models/llada \
  cst_checkpoint=outputs/cst_ckpt/cst_llada_pd_512_256_k4096_w4_ls0.001_best.pt \
  prompt="Your question here" \
  out_dir=outputs/tag_demo
```

### Chunk-only (existing hidden `.pt`)

```bash
python hydra_runners/run_tag.py \
  cst_checkpoint=path/to/cst_..._train_best.pt \
  hidden_pt=path/to/file.pt \
  model_path=./models/llada \
  out_dir=outputs/tag_from_pt
```

### Useful overrides

```bash
# Reference-style decode (no EOS harness)
python hydra_runners/run_tag.py ... use_eos_harness=false

# Lexical readout (hide EOS mass)
python hydra_runners/run_tag.py ... output_exclude_eos=true

# Residual error nodes
python hydra_runners/run_tag.py ... with_residual=true

# Faster chunk run without readout
python hydra_runners/run_tag.py ... hidden_pt=... no_logits=true

# Tighter graph
python hydra_runners/run_tag.py ... influence_node_mass=0.7 influence_edge_mass=0.95
```

Slurm: `sbatch scripts/tag.sh --cst_checkpoint ... --model_path ... --prompt "..."`
Add `--slurm` only if your cluster requires `srun` for GPU binding inside `sbatch`.

---

## Outputs


| File             | When              | Contents                                      |
| ---------------- | ----------------- | --------------------------------------------- |
| `tag_trace.pt`   | Full run          | Hidden chunk (CSTHID01 or legacy `save_hidden_records`) |
| `tag_graph.json` | Always            | `nodes` / `edges` with weights and `meta`     |
| `tag_view.html`  | `write_html=true` | Static vis-network viewer (CDN on first open) |


Node `meta` examples: feat `a`, out `p_sink` / `logit_p_tokens`, err `l2`. Edge `meta`: `type`, `delta`, `f`, `g`, `v` (token id for feat->out).

---

## Configuration

Match `config/cst.yaml` `cst:` (`k`, `w`, `max_steps`, `d_model`) to the checkpoint. The scheduler short code used for training does not change the CST weights, but your capture should use a consistent `mode_key` / `steps` / `gen_length` when comparing to other runs.
