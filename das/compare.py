"""Summarise screening runs against EB+Fisher.  python -m das.compare [--ref NAME] [dir]"""
import csv
import json
import sys
from math import comb
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# mentor's per-task HumanEval results: local copy (standalone) or the Strace repo's own outputs
MENTOR = next((p for p in (_ROOT / 'outputs/mentor/latent_full_humaneval164_per_task.csv',
                           _ROOT / 'outputs/latent_full_humaneval164/per_task.csv') if p.exists()),
              _ROOT / 'outputs/mentor/latent_full_humaneval164_per_task.csv')


def mcnemar(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    args = sys.argv[1:]
    ref = 'ebf_s128_b20'
    if '--ref' in args:
        i = args.index('--ref'); ref = args[i + 1]; del args[i:i + 2]
    root = Path(args[0]) if args else Path('/root/aditya/outputs/screen')
    runs = {}
    for p in sorted(root.glob('*/results.json')):
        r = json.loads(p.read_text())
        runs[p.parent.name] = (r['summary'], {int(k): v for k, v in r['passed'].items()})
    tasks = sorted(next(iter(runs.values()))[1]) if runs else []
    mentor = {}
    for row in csv.DictReader(MENTOR.open()):
        t = int(row['task_id'].split('/')[1])
        mentor[t] = dict(eb=row['eb_sampler_passed'] == 'True', ebf=row['eb_target_temporal_passed'] == 'True',
                         info=row['response_information_readiness_passed'] == 'True')
    base = runs.get(ref, (None, None))[1]
    print(f'reference: {ref}')
    print(f'{"run":44s} {"pass":>8s} {"nfe":>7s} {"nfeMax":>6s} {"unfin":>6s} {"W":>3s} {"L":>3s} {"p":>6s}')
    for name in (['mentor_eb', 'mentor_ebf', 'mentor_info'] if not any(b in str(root) for b in ('gsm8k', 'mbpp')) else []):
        key = name.split('_')[1]
        sub = {t: mentor[t][key] for t in tasks}
        w = l = 0
        if base:
            w = sum(sub[t] and not base.get(t, False) for t in sub); l = sum(base.get(t, False) and not sub[t] for t in sub)
        print(f'{name:44s} {sum(sub.values()):>4d}/{len(sub):<3d} {"":>7s} {"":>6s} {"":>6s} {w:>3d} {l:>3d} {mcnemar(w, l):6.3f}')
    for name, (s, passed) in runs.items():
        w = l = 0
        if base and set(base) == set(passed):
            w = sum(passed[t] and not base[t] for t in passed); l = sum(base[t] and not passed[t] for t in passed)
        print(f'{name:44s} {s["passed"]:>4d}/{s["n"]:<3d} {s["nfe_mean"]:7.1f} {s["nfe_max"]:6.0f} {s["unfinished_mean"]:6.1f} '
              f'{w:>3d} {l:>3d} {mcnemar(w, l):6.3f}')


if __name__ == '__main__':
    main()
