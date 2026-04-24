# Scheduler smoke tests

Use `test_schedulers.py` or `test_schedulers.sh` to run one or all schedulers on a fixed prompt **without** the full lm-eval harness. Useful for quick shape checks, debugging, and optional W&B or tracing.

## Model path

By default, checkpoints are loaded from `./models/llada` (same idea as `model_path` elsewhere in the repo). Use **`--model_path DIR`** to point at another local model directory. Mask / EOS / EoT token ids for schedulers are read from each checkpoint’s `config.json` via `load_model.infer_diffusion_token_ids` (so LLaDA, LLaDA 1.5, and Dream are handled correctly). Repeat **`--model_path`** to run the same test suite on **several** models, **one after another** on the same device; the process frees GPU memory after each model so the next can load. Models are not run in parallel.

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
```

`./test_schedulers.sh --help` lists options: `--scheduler`, `--model_path` (repeatable), `--wandb`, `--wandb_project`, `--wandb_name`, `--intervention_mode`, `--batch`, `--batch_size`, `--tracing_enabled`, and others.

For full **benchmark** evaluation, use [Evaluation](evaluation.md).
