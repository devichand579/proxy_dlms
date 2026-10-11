# Pace experiments (experiment set 2)

Our scheduler work on top of the mentor's experiment set 1 (branch `experiment-set-1`).

| File | What it is |
|---|---|
| `writeup.md` | Paper-style write-up: "Pace: A Step-Budget Unmasking Schedule for LLaDA" (online, with the interactive chart: https://claude.ai/code/artifact/c70cef50-c891-4060-a1ac-82517e13519f) |
| `experiment_log.md` | Every experiment (Exp 1-22): idea, results, paired tests, what we took from it |
| `handoff.md` | Context for continuing the work: story, results, code map, rules, pitfalls, next steps |

Where things are in this repo (the docs were written on the machine where the work ran, so they use
those paths):

| Docs say | In this repo |
|---|---|
| `/root/aditya/das/` | `das/` (code; uses this repo's `schedulers/`, `schedulerlens/`, `load_model.py`, `traces/`) |
| `/root/aditya/outputs/<folder>/<run>/` | `outputs/pace/<folder>/<run>/` (`results.json` + `config.json` per run, `SUMMARY*.txt`; raw generations not committed) |
| `/root/Strace/models/` | `models/` (weights not in git) |

Paired test from committed results: `cd outputs/pace && python3 ../../das/pair.py full/pace_s16:full/ebf_s16`
