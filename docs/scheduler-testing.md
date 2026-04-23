# Scheduler smoke tests

Use `test_schedulers.py` or `test_schedulers.sh` to run one or all schedulers on a fixed prompt **without** the full lm-eval harness. Useful for quick shape checks, debugging, and optional W&B or tracing.

**Single scheduler, default (single-sample) tests:**

```bash
python test_schedulers.py --scheduler pure_diffusion
python test_schedulers.py   # all schedulers
```

**Batched generation (shape checks + optional tracing):**

```bash
python test_schedulers.py --batch --batch_size 2
python test_schedulers.py --batch --scheduler pure_diffusion --batch_size 4
```

**W&B and tracing:**

```bash
python test_schedulers.py --batch --wandb
```

**Shell wrapper** (env/conda and forwards flags):

```bash
./test_schedulers.sh --scheduler pure_diffusion
./test_schedulers.sh --batch --batch_size 2
./test_schedulers.sh --batch --wandb
```

`./test_schedulers.sh --help` lists options: `--scheduler`, `--wandb`, `--wandb_project`, `--wandb_name`, `--intervention_mode`, `--batch`, `--batch_size`, `--tracing_enabled`, and others.

For full **benchmark** evaluation, use [Evaluation](evaluation.md).
