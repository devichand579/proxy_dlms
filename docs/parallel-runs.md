# Parallel jobs (train and eval)

File **names** disambiguate **model, scheduler, CST `k`/`w`**, and (for eval) **task**; they do **not** include a run UUID. **Isolation for concurrent jobs** comes from using **separate working directories** (or separate `staging_dir` / `checkpoint_dir` / `output_dir` / `run_id`).

## CST training

- **Same** `model_path` basename, `scheduler_mode`, and `cst.k` / `cst.w` → **same** checkpoint and staging name tokens. Two jobs writing to the **same** `train.staging_dir` and `train.checkpoint_dir` will **race** and clobber (mixed `.pt` chunks, corrupted `latest.pt`).

**Practical setup**

- **Hydra multirun:** set **distinct** `train.staging_dir` and `train.checkpoint_dir` per job, e.g. `train.staging_dir=outputs/stage_${hydra.job.num} train.checkpoint_dir=outputs/ckpt_${hydra.job.num}` or a unique path per experiment.
- **Shell / Slurm:** export different dirs per `srun` task or `CUDA_VISIBLE_DEVICES` slot.
- **Optional:** one job per `model` / `cst` sweep so tokens differ; still prefer **separate dirs** for safety.

## lm-eval (with or without CST)

- **Non-resumable** runs: default `output_suffix` includes a **timestamp** → new folder per start; parallel launches usually **do not** collide.
- **Resumable** runs: output is `outputs/{run_id}/` with fixed `run_id` default `"{model}_{task_name}_{mode}"`. **Two parallel resumable jobs** with the same model/task/mode (and no custom `run_id`) will **corrupt the same** `checkpoint.jsonl`, `hidden/`, and `cst_*_eval_agg.json`.

**Practical setup**

- Give each resumable job a **unique** `run_id` (or `output_suffix` in `eval.sh` terms). Example: `run_id=gsm8k_pd_job7` or `run_id=run_${SLURM_JOBID}`.
- For **parallel sweep** of tasks/modes, the default suffixes already differ (different `task_name` or `mode` in the string).

## Offline `launch_cst` mode=eval

- Default “pick best checkpoint in dir” uses **newest** `cst_*_train_best.pt` if an explicit default path is missing. With **multiple** compatible checkpoints in one directory, that heuristic can pick the **wrong** file. Prefer **explicit** `eval.checkpoint=.../cst_..._train_best.pt` per job.

## Naming architecture (summary)

| Area | What names encode | Collision if… |
|------|-------------------|---------------|
| Train staging + ckpt | `model`, `scheduler`, `k`, `w`, `train`, epoch/seg/batch | Same token **and** same dirs |
| Eval hiddens + resumable JSON | `model`, `scheduler`, `k`, `w`, `task` | Same token **and** same `output_dir` |
| Train ckpt files | `cst_{token}_train_{latest\|best\|state}.*` | Same token **and** same `checkpoint_dir` |

Loss (`windowed_cst_loss_full` in `cst/train/loss.py`) and trainer integration are **per-process**; there is no cross-job shared state beyond the filesystem paths above.

## W&B

- Set **`WANDB_API_KEY`** in the environment (or use logged-in W&B). Do not commit API keys in scripts.

See also: [CST (training & eval)](cst.md), [Evaluation](evaluation.md), [index](README.md).
