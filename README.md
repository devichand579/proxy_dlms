# Scheduler Tracing: Characterization of Unmasking Schedulers in Diffusion Language Models

Diffusion-based text generation with LLaDA-style **schedulers**, **lm-evaluation-harness** integration, batched generation, **tracing**, optional **interventions**, optional **pre-logit hidden capture** and **CST** training/eval (`cst/`), and **mixed training prompt pools** (`cst.train_data`).

## Quick start

```bash
conda create -n strace -f environment.yaml
conda activate strace
pip install -r requirements.txt
```

Download a model (e.g. LLaDA) — see [docs/setup.md](docs/setup.md). Then from the repo root:

```bash
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --compile_model False
```

CST training (collect, fit, checkpoints):

```bash
./cst.sh
# or: python launch_cst.py mode=train
```

## Documentation

| | |
|--|--|
| **[docs/README.md](docs/README.md)** | Doc index and quick command reference |
| [Setup (env & models)](docs/setup.md) | Conda, requirements, Hugging Face downloads |
| [Evaluation](docs/evaluation.md) | `eval.sh`, `launch_eval.py`, resumable runs, Hydra/SLURM, outputs |
| [CST: training & eval](docs/cst.md) | Pre-logit capture, checkpoint naming, training, lm-eval+CST, offline eval, `train_data` |
| [Parallel runs](docs/parallel-runs.md) | Concurrent train/eval without clobbering paths |
| [Scheduler smoke tests](docs/scheduler-testing.md) | `test_schedulers.py` / `.sh` |


## Repository layout (high level)

- `config/` — Hydra YAML (`config.yaml` for `launch_eval`, `cst.yaml` for `launch_cst`)
- `cst/` — pre-logit capture, CST model bank, training and eval utilities
- `eval.py` — lm-eval model implementation
- `launch_eval.py`, `launch_cst.py` — Hydra entry points
- `eval.sh`, `cst.sh` — shell wrappers
- `schedulers/` — generator-facing scheduler implementations
- `docs/` — detailed documentation
