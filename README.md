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
| [Small scheduler ablations](docs/infogain-lab.md) | Exact CPU checks, short LLaDA reveal probes, and cached information-gain score sweeps |
| [Latent count/rank experiments](docs/latent-scheduler-experiments.md) | HumanEval-20 with EB baseline, count/rank factorials, and completed arithmetic trajectories |
| [One-pass three-term objective](docs/latent-three-term-objective.md) | Activation and decoder-gradient proxies, Gaussian spectral set selection, and independent count/rank ablations |
| [Latent error-allowance counts](docs/latent-error-allowance-experiments.md) | Fixed-ranking count isolation, cached-response feedback, attention coupling, and explicit Fisher-geometry diagnostics |
| [Useful code-body progress](docs/latent-body-progress.md) | Local gradient scoring, independent counts, commitment timing, and earlier-step correctness replay |
| [CST: training & eval](docs/cst.md) | **`mode=store`** (Hub hiddens) · **`mode=train`** / **`mode=eval`** (CST only) |
| [TAG (attribution graph)](docs/tag.md) | `scripts/tag.sh`, `hydra_runners/run_tag.py`, one prompt or `.pt` chunk |
| Parallel runs | Concurrent train/eval without clobbering paths |
| [Scheduler smoke tests](docs/scheduler-testing.md) | `tests/test_schedulers.py` / `.sh` |
