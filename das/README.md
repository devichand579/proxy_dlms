# das: pace and other unmasking schedulers for LLaDA

Code for the BTP experiments on unmasking schedulers for LLaDA-8B Base / LLaDA-1.5.
The main method is **pace**: EB+Fisher's count and ranking, plus a floor of
⌈(answer tokens still masked) / (steps left)⌉ tokens per step. Results and reasoning are in
`experiment_log.md`, the paper-style write-up in `writeup.md`, and the context for picking the
work up in `handoff.md` (in `/root/aditya`, or `docs/pace/` in the Strace repo).

## Files

| File | What it does |
|---|---|
| `das_sampler.py` | `generate_das(...)`: every count/score we tried. Pace = `count='pace'`. Defaults reproduce EB+Fisher. |
| `bench.py`, `bench_gsm8k.py`, `bench_mbpp.py` | HumanEval 164 / GSM8K 0-399 / MBPP 0-499 runners. Save one `batch_XXX.json` per batch and resume from them. |
| `compare.py` | Summary table of a results folder against a reference run. |
| `pair.py` | Paired wins/losses + exact McNemar p: `python3 das/pair.py runA_dir:runB_dir` |
| `run.sh` | Runs one bench job and logs to `<OUT_ROOT>/logs/<name>.log`. |
| `queue2.sh <jobfile> <N>` | Runs a job file, N jobs in parallel. |
| `jobs_*.txt` | The exact command lines of every experiment (see the log for which file is which). |
| `tests/` | 20-problem smoke tests. |
| `vendor/` | Standalone checkout only: copies of the Strace modules das imports (see `vendor/README.md`). |

## Running

```
# pace, HumanEval, 32 steps
./das/run.sh --method das --name pace_s32 --steps 32 --tasks 0-163 --out outputs/full --cfg '{"count":"pace"}'
# EB+Fisher baseline
./das/run.sh --method ebf --name ebf_s32 --steps 32 --tasks 0-163 --out outputs/full
# GSM8K / MBPP: prefix with MODULE=das.bench_gsm8k or MODULE=das.bench_mbpp
# LLaDA-1.5: --model models/llada1_5 (EB+Fisher there uses --cfg '{"b44":true}')
```

Environment variables for `run.sh`:
- `MODEL_ROOT` (default `/root/Strace`): `--model models/llada` is resolved against it. The model
  weights are the only thing needed from outside this folder.
- `PYTHON` (default `/root/miniconda3/envs/strace/bin/python`), `OUT_ROOT` (log folder root,
  default `outputs/` next to `das/`).

Datasets are read from the local Hugging Face cache (`~/.cache/huggingface/datasets`), offline.
The job files hard-code `--out /root/aditya/outputs/...`, as they were run.

Pitfalls: run MBPP jobs one at a time (a 4,309-token prompt runs out of GPU memory with 3 in
parallel), set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and remember a rerun with the
same `--name` resumes from saved batches (delete the run folder to redo it).
