# STrace documentation

This folder contains detailed guides. The [top-level README](../README.md) only holds a short overview and quick links.

| Doc | Contents |
|-----|----------|
| [Setup](setup.md) | Conda environment, `requirements.txt`, downloading LLaDA / Dream / LLaDA-1.5 |
| [Evaluation](evaluation.md) | `eval.sh` and `launch_eval.py`, schedulers, resumable runs, outputs, Hydra multirun / SLURM |
| [CST (training & eval)](cst.md) | Pre-logit hidden capture, checkpoint naming, CST training, lm-eval with CST, offline eval, `cst.train_data` |
| [Scheduler smoke tests](scheduler-testing.md) | `test_schedulers.py` / `test_schedulers.sh` for quick local checks |
| [Parallel runs](parallel-runs.md) | Isolating concurrent train/eval jobs (dirs, `run_id`, no shared `staging`/`ckpt`/`output`) |
| [TAG (attribution graph)](tag.md) | Temporal attribution graph on CST for one prompt |

## Quick commands

```bash
# Environment
conda create -n strace -f environment.yaml && conda activate strace && pip install -r requirements.txt

# One evaluation (from repo root)
./eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada --compile_model False

# CST training (collect + fit + checkpoints)
# ./cst.sh
# or: python launch_cst.py mode=train

# TAG (one prompt or hidden .pt; see docs/tag.md)
# ./tag.sh --help
# ./tag.sh --cst_checkpoint outputs/cst_ckpt/..._train_best.pt --model_path ./models/llada --out_dir outputs/tag --prompt "..."
```

For model downloads, CST configs, and full flag lists, follow the links above.
