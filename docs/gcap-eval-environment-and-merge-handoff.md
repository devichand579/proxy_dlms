# G-cap evaluation environment and upstream merge handoff

Last updated: 2026-10-05

## Current status

- Local `main` and `origin/main` are aligned at `a24cdcfc`.
- The two upstream fixes were fast-forwarded successfully. Overlapping pre-pull
  files were preserved in `stash@{0}` (`pre-upstream-overlap-backup`).
- The local G-cap functionality described below has been reapplied on top of
  the upstream DUS and memory fixes.
- The Docker container is running idle; no full evaluation is active.
- Smoke tests passed, but the full suite has deliberately not been launched.
- The working tree still contains pre-existing tracked and untracked changes.
  Do not discard or commit them without review.

## Completed post-merge integration

- Each scheduler runs `K=8` independent candidate generations per question.
- Candidate tensors are offloaded to CPU between generations, preserving the
  upstream memory reduction.
- Every checkpoint candidate stores its decoded response, full M/J/G totals,
  per-step M/J/G, cumulative M/J/G, and commit counters.
- Offline selection can be recomputed from the persisted steps for any prefix
  (`step < prefix`) or the full trajectory, using raw or per-commit G.
- Resume validation only skips a G-cap question when all `K` responses and
  step-proxy lists are present.
- Checkpoint-only rows no longer inflate persisted global sample counts after
  restart.
- `pd` and `bd` each use two GPUs/shards. `top_k_m` and `eb` retain two
  deterministic checkpoint shards, processed sequentially on one GPU each.
- Sharded runs disable lm-eval request caching. lm-eval 0.4.9.2 omits
  `--samples` from its request-cache key, which was the reason both old workers
  received all 164 HumanEval requests despite correct 82-document shard files.
- Run IDs and W&B names contain the task, scheduler, score mode, K,
  temperature, step budget, and shard instead of generic `shard1`/`shard2`
  names.

## Server and data layout

- Code: `/home/machaita/Strace`
- External data: `/home/machaita/Strace_data`
- Model: `/home/machaita/Strace_data/models/llada`
- Outputs: `/home/machaita/Strace_data/outputs`
- Logs: `/home/machaita/Strace_data/logs`
- Hugging Face cache: `/home/machaita/Strace_data/hf_cache`
- Hardware: eight AMD Instinct MI350X GPUs with approximately 252 GiB VRAM each.
- ROCm devices: `/dev/kfd` and `/dev/dri`

The model checkpoint and reusable cache are intentionally outside the Git repository. Cleaning run outputs must not remove `models/` or `hf_cache/`.

## Environment setup performed

The requested `strace-primus:v26.7` image was absent and could not be pulled from Docker Hub. A replacement image with that local tag was built from the installed ROCm image:

```dockerfile
FROM hipphysics-mae:rocm72
WORKDIR /workspace/Strace
COPY requirements.txt /tmp/strace-requirements.txt
RUN awk '!/^(torch|triton|nvidia-)/' /tmp/strace-requirements.txt \
      > /tmp/strace-requirements-rocm.txt \
    && python3 -m pip install --no-cache-dir \
      -r /tmp/strace-requirements-rocm.txt \
    && rm -f /tmp/strace-requirements.txt \
      /tmp/strace-requirements-rocm.txt
CMD ["/bin/bash"]
```

`torch`, `triton`, and `nvidia-*` packages were excluded to avoid replacing the working ROCm PyTorch installation with CUDA packages.

Validated runtime versions:

- PyTorch `2.9.1+rocm7.2`
- Transformers `4.57.3`
- Datasets `4.4.2`
- lm-eval `0.4.9.2`
- Accelerate `1.12.0`
- Hydra Core `1.4.0.dev1`

The complete `GSAI-ML/LLaDA-8B-Base` checkpoint was downloaded to the external model directory. It contains six weight shards totalling about 15 GiB. A real bfloat16 model forward pass succeeded on an MI350X.

The persistent container was created with:

```bash
CODE=/home/machaita/Strace
DATA=/home/machaita/Strace_data

mkdir -p "$DATA/outputs" "$DATA/logs" "$DATA/hf_cache"

docker run -d \
  --name llada-gcap-runtime \
  --restart unless-stopped \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --network=host \
  --shm-size=128g \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HF_HOME=/data/hf_cache \
  -v "$CODE:/workspace/Strace" \
  -v "$DATA/models/llada:/workspace/Strace/models/llada:ro" \
  -v "$DATA/outputs:/workspace/Strace/outputs" \
  -v "$DATA/logs:/workspace/Strace/logs" \
  -v "$DATA/hf_cache:/data/hf_cache" \
  -w /workspace/Strace \
  strace-primus:v26.7 \
  sleep infinity
```

The container has restart policy `unless-stopped`, so it remains available when the terminal closes. Detached `docker exec` evaluation processes do not automatically restart after a host/container restart; relaunch the suite explicitly.

## Changes made to the suite launcher

`scripts/gcap_suite.sh` is currently untracked and contains the session-specific launcher work.

### ROCm GPU mapping

The original launcher used the same physical index for both:

```bash
ROCR_VISIBLE_DEVICES="$gpu"
HIP_VISIBLE_DEVICES="$gpu"
```

On ROCm, `ROCR_VISIBLE_DEVICES` first narrows and remaps the selected physical GPU to logical device 0. Setting `HIP_VISIBLE_DEVICES` to the original nonzero physical index then hides the only visible device.

The corrected mapping is:

```bash
export ROCR_VISIBLE_DEVICES="$gpu"
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
```

This was validated for physical GPU indices 1 and 7.

### Scheduler budgets and sharding

Current intended configuration:

- `pd`: 512 steps, two shards
- `bd`: 512 steps, two shards
- `top_k_c`: 256 steps, one shard
- `top_k_m`: 256 steps, two shards
- `dus`: 128 steps, one shard
- `eb`: 128 steps, two shards

The two-shard schedulers use shard indices `0/2` and `1/2`. Human-readable W&B names can later display these as `part1of2` and `part2of2` without changing the zero-based dataset indices.

All eight GPUs are assigned:

```text
GPU 0: pd shard 0/2
GPU 1: pd shard 1/2
GPU 2: bd shard 0/2
GPU 3: bd shard 1/2
GPU 4: top_k_c
GPU 5: dus
GPU 6: top_k_m shards 0/2 then 1/2
GPU 7: eb shards 0/2 then 1/2
```

### W&B

- W&B was enabled for the suite.
- Authentication was verified without printing the token.
- Username: `devichand579`
- Default entity: `devichand579-iit-kharagpur`
- Project: `strace`

The API key is currently hardcoded in `repo_env_defaults.py`. This is a security risk. Rotate it and provide it through an environment variable or secret manager. Never copy its value into this document or a commit.

### Script permissions

`scripts/gcap_suite.sh` and `scripts/eval.sh` were made executable because the checkout initially had mode `0664`, causing `Permission denied`.

Several other tracked scripts currently show accidental executable-bit removals in `git diff --summary`. Review and restore those modes before committing.

## Local working-tree scope

Only a subset of the dirty working tree was created during this session. The important session-created file is:

- `scripts/gcap_suite.sh`

The following existing local work was already present and must be preserved or intentionally discarded after review:

- G-cap prefix-step scoring and candidate persistence changes
- Dataset sharding support
- Resumable checkpoint changes
- Evaluation/Hydra launcher changes
- G-cap summary and sample-sharding modules
- CST and TAG code/configuration
- Numerous tests and analysis utilities

Important modified tracked files include:

- `config/lmeval.yaml`
- `docs/gcap_rerank.md`
- `eval.py`
- `hydra_runners/launch_eval.py`
- `schedulers/gcap_rerank.py`
- `schedulers/generate.py`
- `scripts/eval.sh`

There are also many untracked files and directories, including `tests/`, `cst/`, `tag/`, several configs/docs, and `Schedulerlens_ICLR.pdf`. Review these before creating a safety commit. Do not blindly commit the PDF or credentials.

## Non-negotiable G-cap functionality to reapply after upstream merge

These requirements are the main purpose of the local G-cap work. They must be
preserved when local changes are applied on top of `origin/main`.

### Eight scheduler-generated candidates per question

For every benchmark question and scheduler:

1. Run the selected unmasking scheduler eight independent times (`K=8`).
2. Produce eight complete candidate sequences.
3. Capture the unmasking trace for each candidate.
4. Calculate G-cap metrics separately for each candidate.
5. Persist all eight decoded responses and their metrics.

The scheduler generates the candidate sequences. G-cap is calculated from each
candidate's trace; G-cap does not generate candidates.

Supported schedulers:

- pure diffusion (`pd`)
- block diffusion (`bd`)
- confidence top-k (`top_k_c`)
- top-margin top-k (`top_k_m`)
- dilated unmasking (`dus`)
- entropy-budget sampler (`eb`)

### Per-step M, J, and G persistence

For every one of the eight candidates, retain metrics at every recorded
diffusion step. The post-merge checkpoint schema must include at least:

```json
{
  "candidate_index": 0,
  "response": "...",
  "M_hat": 0.0,
  "J_hat": 0.0,
  "G_hat": 0.0,
  "n_committed": 0,
  "n_steps": 0,
  "step_proxies": [
    {
      "step": 0,
      "M_hat_t": 0.0,
      "J_hat_t": 0.0,
      "G_hat_t": 0.0,
      "cumulative_M_hat": 0.0,
      "cumulative_J_hat": 0.0,
      "cumulative_G_hat": 0.0,
      "n_committed": 0,
      "n_selected": 0,
      "n_remaining": 512
    }
  ]
}
```

The integrated implementation persists:

- full-trajectory `M_hat`, `J_hat`, and `G_hat`
- per-step `M_hat_t` and `J_hat_t`
- `n_committed`, `n_selected`, and `n_remaining`
- all candidate responses

`G_hat_t` and cumulative M/J/G are persisted explicitly, making each
checkpoint self-contained.

### Selection at full trajectory or any requested prefix

The saved candidate pool must support selection without regenerating samples:

- **Full-sequence selection:** choose the candidate with minimum total G-cap.
- **Prefix selection:** choose the candidate with minimum cumulative G-cap up
  to an arbitrary requested diffusion step.
- **Per-commit selection:** when requested, divide cumulative G-cap by the
  number of tokens committed through that prefix.

For prefix `p`, include steps satisfying `step < p`:

```text
prefix_M = sum(M_hat_t)
prefix_J = sum(J_hat_t)
prefix_G = prefix_M + prefix_J
prefix_per_commit = prefix_G / max(prefix_n_committed, 1)
```

The online generation path selects at `GCAP_PREFIX_STEPS` (default 32).
`schedulerlens/gcap_summary.py` independently recomputes candidate selection
from saved `step_proxies`; use `--prefix-steps N` for any prefix or
`--full-trajectory` for all steps. No candidate regeneration is required.

### Resumability

Every run must remain resumable:

- Stable run IDs encode task, scheduler, K, temperature, step budget, and shard.
- Each completed question is appended immediately to `checkpoint.jsonl`.
- The checkpoint is flushed after each question.
- Restarting the same run loads `(task_name, doc_id)` keys and skips completed
  questions.
- A question is complete only after all eight candidates, all per-step
  metrics, and the selected response have been written.
- W&B run IDs are persisted and resumed rather than creating duplicate runs.
- The random-control response must come from the same eight-candidate pool; it
  must not regenerate candidates.

### Scheduler-specific step budgets

The budgets from the last run are required:

- `pd`: 512
- `bd`: 512
- `top_k_c`: 256
- `top_k_m`: 256
- `dus`: 128
- `eb`: 128

Run IDs and W&B names must include the scheduler's actual step budget.

### GPU allocation

The two 512-step bottlenecks are split across two GPUs:

- `pd`: shard `0/2` and shard `1/2`
- `bd`: shard `0/2` and shard `1/2`

`top_k_m` and `eb` keep their existing two-shard checkpoint format, but each
runs shard 0 and shard 1 sequentially on one GPU. `top_k_c` and `dus` use one
GPU each.

Shard requirements:

- deterministic document assignment
- disjoint shard document IDs
- union equals full benchmark coverage
- no duplicate checkpoint records when shard directories are combined
- final summary merges both shard directories before scoring

The previous run's 164-request shard failure was caused by lm-eval's request
cache key omitting `--samples`. Sharded commands now omit `--cache_requests`.
An end-to-end lm-eval HumanEval test confirmed one shard builds and sends
exactly 82 requests.

### Required post-merge combination with upstream

The upstream changes must remain the base:

- DUS remask trace correction
- sequence-chunked stochastic sampling
- sequence-chunked confidence/entropy/margin calculations
- trace tensors moved to CPU
- completed candidate outputs moved to CPU
- GPU references released between candidates

Apply the local behavior on top:

- `K=8` candidate generation
- prefix and full-trajectory metrics
- every-step M/J/G persistence
- all eight decoded candidate responses
- arbitrary offline prefix/full selection
- resumable checkpoints and W&B IDs
- deterministic two-GPU sharding for `pd` and `bd`, with sequential
  checkpoint shards for `top_k_m` and `eb`
- scheduler-specific budgets
- suite summary over merged shard outputs

### Acceptance criteria

The integration is ready only when tests demonstrate:

1. Every question checkpoint has exactly eight candidates.
2. Every candidate has a response and nonempty per-step metrics.
3. Per-step G equals M plus J.
4. Cumulative full-trajectory values equal the stored total values.
5. Selecting at two different prefixes can produce independently recomputed
   candidate indices without regenerating candidates.
6. Full-trajectory selection uses all saved steps.
7. Restarting skips completed questions and continues incomplete coverage.
8. Two shard files are disjoint and merge to exact benchmark coverage.
9. DUS remasking produces a valid trace.
10. Peak VRAM remains bounded with the upstream chunked operations.

## Last run configuration

The last full launch used:

```bash
export WANDB=True
export WANDB_ENTITY=devichand579-iit-kharagpur
export BATCH_SIZE=64
export PD_STEPS=512
export BD_STEPS=512
export TOP_K_C_STEPS=256
export TOP_K_M_STEPS=256
export DUS_STEPS=128
export EB_STEPS=128
export GEN_LENGTH=512
export GCAP_K=8
export GCAP_PREFIX_STEPS=32
export TEMPERATURE=0.7
```

Benchmarks:

```text
humaneval mbpp gsm8k minerva_math500
```

## Failures found

### DUS trace failure

The plain historical DUS evaluations succeeded. The failure occurred when a DUS-generated candidate trace was scored for G-cap.

DUS can write a token and remask it during the same step. The local tracer recorded the write in `selected_positions` but did not remove it after remasking. The metric validator compared `selected_positions` with the net masked-to-unmasked transition and raised:

```text
TraceProxyError: selected_positions vs mask_diff mismatch
```

This is a code issue, not a model-generation failure.

### GPU OOM

All seven non-DUS workers eventually failed with a real HIP capacity OOM. They had completed 83 checkpoint records each when the failure occurred.

The failing allocation was about 45.4 GiB in `add_gumbel_noise()`:

```python
logits = logits.to(torch.float64)
noise = torch.rand_like(logits, dtype=torch.float64)
gumbel_noise = (-torch.log(noise)) ** temperature
return logits.exp() / gumbel_noise
```

At failure, PyTorch had approximately 230.7 GiB allocated and only 12–13 GiB free. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not sufficient because this was genuine capacity exhaustion, not only fragmentation.

This was the historical failure before upstream sequence chunking. Post-merge
memory tests passed at batch sizes 16, 32, and 64.

### Dataset sharding root cause and fix

The two-shard sample files were correct. lm-eval loaded a previously cached
full-task request list because its cache key does not include `--samples`.
Disabling request caching for sharded runs fixes this; a real lm-eval request
construction test sent 82 HumanEval requests.

## Improvements already present on remote main

`origin/main` adds two commits:

### `a7ec7a4` — DUS tracing fix

Relevant improvements:

- Clears a traced selected position when DUS remasks it.
- Keeps `selected_positions` aligned with the next step's mask difference.
- Moves trace predictions, confidences, entropies, transfers, prompts, and final sequences to CPU.
- Avoids retaining unnecessary full-vocabulary logits copies in tracing.
- Offloads completed G-cap candidate sequences to CPU and clears GPU memory between candidates.
- Adds a `majority_acc` metric to prefix-selection analysis.
- Deletes two plotting modules; confirm they are no longer needed:
  - `schedulerlens/plots/atlas_plots.py`
  - `schedulerlens/plots/pcfg_corr_plots.py`

This commit directly fixes the observed DUS trace mismatch.

### `a24cdcfc` — G-cap rank memory optimizations

Relevant improvements:

- Replaces full-tensor Gumbel sampling with sequence-chunked `sample_tokens()`.
- Computes confidence, entropy, top-two margin, and token log-probability in sequence chunks.
- Avoids duplicate full-vocabulary softmax tensors in EB and top-k schedulers.
- Applies the optimized sampling path to PD, BD, DUS, EB, FDM, and top-k schedulers.
- Moves more trace state to CPU.
- Deletes the old root-level `temp` script.

This commit directly addresses the observed 45.4 GiB temporary allocation and OOM.

## Upstream/local overlap

Most upstream files do not overlap the current local modifications. The two direct overlaps are:

- `schedulers/gcap_rerank.py`
- `temp`

`temp` is a server-specific script using an obsolete `/shared_nfs/...` path. Prefer dropping it or moving a corrected version to `scripts/` if it is still needed.

### Required manual combination for `gcap_rerank.py`

Preserve local functionality:

- prefix-only G-cap selection (`gcap_prefix_steps`)
- cumulative prefix metrics
- per-step proxy persistence
- all K candidate responses in checkpoints
- `last_candidate_outputs`

Preserve upstream memory fixes:

- `generated.detach().cpu()` for every candidate
- deleting candidate GPU references after scoring
- `torch.cuda.empty_cache()` between candidates
- stacking/selecting candidates on CPU
- moving only the selected outputs back to the prompt device

`last_candidate_outputs` should remain available for checkpoint decoding, but it should point to the CPU `stacked` tensor rather than keeping all K candidates on GPU.

## Recommended safe merge procedure

Do **not** run `git pull` directly on the current dirty `main`.

1. Review the working tree and remove generated or unwanted files.
2. Create a local safety branch from the current state:

   ```bash
   git switch -c backup/local-gcap-before-upstream
   ```

3. Stage only intentional source/docs/tests. Exclude:

   - credentials
   - run outputs/logs
   - model files
   - caches
   - `Schedulerlens_ICLR.pdf` unless intentionally versioned
   - obsolete `temp`

4. Commit the reviewed local work on the safety branch.
5. Create an integration branch from the fetched remote:

   ```bash
   git switch -c integrate/gcap-upstream origin/main
   ```

6. Cherry-pick the reviewed local commit(s):

   ```bash
   git cherry-pick <local-commit>
   ```

7. Resolve `schedulers/gcap_rerank.py` manually using the combination described above.
8. Keep the upstream DUS tracing and scheduler utility changes.
9. Confirm script executable modes.
10. Run tests and smoke evaluations before any full suite.

This branch-based approach keeps the original dirty state recoverable and avoids losing either side.

## Post-merge validation results

Completed:

- Focused persistence, arbitrary-prefix/full-selection, and sharding tests:
  `4 passed`.
- HumanEval shard 0/2: lm-eval built and sent exactly 82 requests.
- DUS, K=8, 128 steps, one HumanEval question: passed with no trace mismatch.
- EB, K=8, 128 steps, one HumanEval question: passed and persisted all
  candidate data.
- Re-running the EB smoke loaded the complete question from checkpoint and
  performed zero diffusion steps.
- EB stochastic memory path, 512-token prompt + 512 generated tokens:
  batch 16 = 19.59 GiB, batch 32 = 24.18 GiB, batch 64 = 33.34 GiB peak.
- Long-prompt stress test, batch 64 with 1024-token prompt + 512 generated
  tokens: 41.06 GiB peak.

Therefore `BATCH_SIZE=64` is validated for the intended MI350X environment
with substantial VRAM headroom. This is a smoke-tested safe size, not a claim
that 64 is the absolute maximum.

The broader untracked legacy test directory still has three collection errors
from older local APIs (`DEFAULT_ENTROPY_TAU`, `trajectory_blocks`, and
`build_length_aware_chunks`). These are separate from the G-cap integration
and upstream fixes.

Suggested post-merge launch environment:

```bash
export WANDB=True
export WANDB_ENTITY=devichand579-iit-kharagpur
export BATCH_SIZE=64
export PD_STEPS=512
export BD_STEPS=512
export TOP_K_C_STEPS=256
export TOP_K_M_STEPS=256
export DUS_STEPS=128
export EB_STEPS=128
export GEN_LENGTH=512
export GCAP_K=8
export GCAP_PREFIX_STEPS=32
export TEMPERATURE=0.7
```

The merged smoke tests have passed. Do not start the full run until explicitly
approved.
