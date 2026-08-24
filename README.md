# Scheduler Tracing: Understanding Diffusion Language Models and Unmasking Schedulers

This repository contains the code for the papers "_____" and "____". A unified framework for understanding and characterizing diffusion-based text generation which includes integration of diffusion language models, unmasking schedulers, evaluations, tracing, interpretability, hidden representations capture, cross-step-transcoder training/evaluation and circuit tracing for unmasking sequences.

## Quick start

```bash
conda create -n strace -f environment.yaml
conda activate strace
pip install -r requirements.txt
```

## Layout

| Path | Purpose |
|------|---------|
| `scripts/` | Bash launchers (`eval.sh`, `cst.sh`, `tag.sh`, `llada-pcfg.sh`, …) |
| `hydra_runners/` | Hydra entry points (`run_cst.py`, `run_tag.py`, `launch_eval.py`, `run_llada-pcfg.py`) |
| `config/` | Hydra YAML configs |

Run everything from the repo root, e.g. `./scripts/eval.sh …`, `python hydra_runners/run_cst.py mode=train`, `python tests/test_schedulers.py --scheduler pure_diffusion`.

## Documentation

| | |
|--|--|
| [Setup (env & models)](docs/setup.md) | Conda, requirements, Hugging Face downloads |
| [Evaluation](docs/evaluation.md) | `scripts/eval.sh`, `hydra_runners/launch_eval.py`, resumable runs, Hydra/SLURM, outputs |
| [CST: training & eval](docs/cst.md) | **`mode=store`** (Hub hiddens) · **`mode=train`** / **`mode=eval`** (CST only) |
| [TAG (attribution graph)](docs/tag.md) | `scripts/tag.sh`, `hydra_runners/run_tag.py`, one prompt or `.pt` chunk |
| Parallel runs | Concurrent train/eval without clobbering paths |
| [Scheduler smoke tests](docs/scheduler-testing.md) | `tests/test_schedulers.py` / `.sh` |

