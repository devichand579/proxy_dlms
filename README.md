# Scheduler Tracing: Understanding Diffusion Language Models and Unmasking Schedulers

This repository contains the code for the papers "Scheduler Tracing: Understanding Diffusion Language Models and Unmasking Schedulers" and "Schedulerlens: A Decision Taxonomy and Interpretability Framework for Masked Diffusion Language Models and Schedulers". A unified framework for understanding and characterizing diffusion-based text generation which includes integration of diffusion language models, unmasking schedulers, evaluations, tracing, interpretability, hidden representations capture, cross-step-transcoder training/evaluation and circuit tracing for unmasking sequences.

## Quick start

```bash
conda create -n strace -f environment.yaml
conda activate strace
pip install -r requirements.txt
```

## Documentation

| | |
|--|--|
| [Setup (env & models)](docs/setup.md) | Conda, requirements, Hugging Face downloads |
| [Evaluation](docs/evaluation.md) | `eval.sh`, `launch_eval.py`, resumable runs, Hydra/SLURM, outputs |
| [CST: training & eval](docs/cst.md) | **`mode=store`** (Hub hiddens) · **`mode=train`** / **`mode=eval`** (CST only) |
| [TAG (attribution graph)](docs/tag.md) | `tag.sh`, `run_tag.py`, one prompt or `.pt` chunk |
| [Parallel runs](docs/parallel-runs.md) | Concurrent train/eval without clobbering paths |
| [Scheduler smoke tests](docs/scheduler-testing.md) | `test_schedulers.py` / `.sh` |

