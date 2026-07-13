# Hydra runners

Hydra entry points — run from the **repo root**:

| Runner | Config | Shell wrapper |
|--------|--------|---------------|
| `run_cst.py` | `config/cst.yaml` | `./scripts/cst.sh` |
| `run_tag.py` | `config/tag.yaml` | `./scripts/tag.sh` |
| `launch_eval.py` | `config/lmeval.yaml` | `./scripts/eval.sh` |
| `run_llada-pcfg.py` | `config/llada-pcfg.yaml` | `./scripts/llada-pcfg.sh` |

```bash
python hydra_runners/run_cst.py mode=store store.scope=train
python hydra_runners/launch_eval.py task_name=gsm8k mode=pd
python hydra_runners/run_llada-pcfg.py mode=train train.hf_dataset=pcfg-code
```

Each runner adds the repo root to `sys.path` automatically.
