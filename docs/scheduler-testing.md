# Scheduler smoke tests

Use `test_schedulers.py` or `test_schedulers.sh` to run one or all schedulers on a fixed prompt **without** the full lm-eval harness. Useful for quick shape checks, debugging, and optional W&B or tracing.

## Model path

By default, checkpoints are loaded from `./models/llada` (same idea as `model_path` elsewhere in the repo). Use **`--model_path DIR`** to point at another local model directory. Mask / EOS / EoT token ids for schedulers are read from each checkpoint’s `config.json` via `load_model.infer_diffusion_token_ids` (so LLaDA, LLaDA 1.5, and Dream are handled correctly). For **Dream** checkpoints, `load_model` wraps the base model in `DreamSchedulersCompatWrapper` so that scheduler-style `forward` calls get the same **position_ids** (from the pad mask) and **logit shift** as `DreamModel`’s official `_sample` path; this avoids garbage text from left-padded batches. Native `model.generate` still uses the unwrapped `DreamModel` and is not affected. Repeat **`--model_path`** to run the same test suite on **several** models, **one after another** on the same device; the process frees GPU memory after each model so the next can load. Models are not run in parallel.

**LLaDA 1.5 (directory name `llada1_5`)** — the repo automatically applies LLaDA Appendix B.4–style **EOS (and EoT) suppression while positions are still masked** (`load_model.infer_b44_eos_unmask_safety`: enables scheduler `logits_eos_inf` / `confidence_eos_eot_inf` only for that layout). This avoids spurious all-`<|endoftext|>` fills in full-block + confidence remasking, without changing `./models/llada`, Dream, or other folders. If your checkpoint is not in a directory named `llada1_5`, set `"strace_b44_eos_unmask": true` in `config.json` to opt in, or `false` to opt out. See [LLaDA 1.5 sampling (B.4)](llada1_5-sampling.md) for the full issue description.

**Single scheduler, default (single-sample) tests:**

```bash
python test_schedulers.py --scheduler pure_diffusion
python test_schedulers.py   # all schedulers
```

**Multiple local models (same as above, per model):**

```bash
python test_schedulers.py --model_path ./models/llada --model_path ./models/dream
python test_schedulers.py --scheduler fhs --model_path ./models/llada1_5
```

**Batched generation (shape checks + optional tracing):**

```bash
python test_schedulers.py --batch --batch_size 2
python test_schedulers.py --batch --scheduler pure_diffusion --batch_size 4
```

**Batched, multiple models** (traces and batch `trace_name` are suffixed with each model’s directory name, e.g. `batch_YYYYMMDD_HHMMSS_llada`):

```bash
python test_schedulers.py --batch --model_path ./models/llada --model_path ./models/dream
```

**Reference-style generation** (no EOS early exit in schedulers, full generated tail decoded — like upstream LLaDA scripts): add **`--no-eos-harness`**.

**Debug: tokens before `skip_special_tokens`:** add **`--show-original-tokens`** to print, for each batch row, the full integer tail that will be `batch_decode`d (and a string with `skip_special_tokens=False`) immediately before the usual stripped decode. The printed tail matches the harness (including truncate-at-first-EOS when the EOS harness is on).

**W&B and tracing:**

```bash
python test_schedulers.py --batch --wandb
```

The W&B run `config` includes the list of `model_paths` when set.

**Shell wrapper** (env/conda and forwards flags):

```bash
./test_schedulers.sh --scheduler pure_diffusion
./test_schedulers.sh --batch --batch_size 2
./test_schedulers.sh --batch --wandb
./test_schedulers.sh --model_path ./models/llada --model_path ./models/dream
# Pass-through to Python (reference-style gen + decode):
./test_schedulers.sh --batch --no-eos-harness
```

`./test_schedulers.sh --help` lists options: `--scheduler`, `--model_path` (repeatable), `--wandb`, `--wandb_project`, `--wandb_name`, `--intervention_mode`, `--batch`, `--batch_size`, `--tracing_enabled`, `--no-eos-harness`, and other pass-through flags (e.g. **`--show-original-tokens`** for `test_schedulers.py`).

For full **benchmark** evaluation, use [Evaluation](evaluation.md).
