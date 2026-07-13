# Shell launchers

Run from the **repo root**. Each script `cd`s to the repo root automatically.

| Script | Purpose |
|--------|---------|
| `eval.sh` | lm-eval harness + tracing |
| `cst.sh` | CST store / train / eval |
| `tag.sh` | Temporal attribution graph |
| `llada-pcfg.sh` | PCFG LLaDA train / scheduler eval |
| `test_schedulers.sh` | Quick scheduler smoke tests |
| `download_datasets.sh` | Prefetch HF datasets for offline eval |

```bash
./scripts/eval.sh --batch_size 2 --mode pd --task_name gsm8k --model llada
./scripts/llada-pcfg.sh --mode train --variant 200m --hf_dataset pcfg-code
sbatch scripts/cst.sh --mode train
```

Python Hydra runners live in `hydra_runners/`; PCFG Python CLIs in `llada-pcfg/scripts/`.
