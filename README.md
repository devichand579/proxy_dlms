# Scheduler Tracing: Characterization of Unmasking Schedulers in Diffusion Language Models

Diffusion-based text generation with LLaDA-style **schedulers**, **lm-evaluation-harness** integration, batched generation, **tracing**, optional **interventions**, optional **pre-logit hidden capture** and **CST** training/eval (`cst/`), **TAG** on one prompt or hidden chunk (`tag/`), and **mixed training prompt pools** (`cst.train_data`).

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

