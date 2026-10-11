"""Small, cached singleton-selection experiments; no production scheduler changes.

Run ``python -m schedulerlens.infogain_lab --help``. All entropies use nats.
The paper-style score is an equation-level diagnostic, not an implementation
of the authors' full sampler (candidate sampling, caching and bypass included).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    positive = p > 0
    return float(-(p[positive] * np.log(p[positive])).sum())


def kl(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    positive = p > 0
    return float((p[positive] * (np.log(p[positive]) - np.log(q[positive].clip(1e-300)))).sum())


def effect_metrics(before_entropy, after_entropy, source):
    """Compare the same surviving targets; source entropy is never external gain."""
    before, after = np.asarray(before_entropy), np.asarray(after_entropy)
    keep = np.arange(len(before)) != source
    external = float((before[keep] - after[keep]).sum())
    paper = float(before.mean() - after[keep].mean()) if keep.any() else 0.0
    return external, paper


def dependence_decomposition(base, branches, weights):
    """Exact identity for a normalized local branch joint, not necessarily data MI."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    mixture = weights @ np.asarray(branches)
    influence = sum(w * kl(p, base) for w, p in zip(weights, branches))
    dependence = sum(w * kl(p, mixture) for w, p in zip(weights, branches))
    return influence, dependence, kl(mixture, base)


def binary_joint(case):
    states = np.asarray(list(itertools.product((0, 1), repeat=6)))
    if case == "independent":
        p1 = np.asarray([.05, .1, .2, .4, .5, .5])
        mass = np.prod(np.where(states == 1, p1, 1 - p1), axis=1)
    elif case == "noisy_anchor":
        matches = states[:, 1:5] == states[:, :1]
        mass = .5 * np.prod(np.where(matches, .9, .1), axis=1)
        mass *= np.where(states[:, 5] == 0, .95, .05)
    elif case == "parity":
        valid = (states[:, 0] ^ states[:, 1]) == states[:, 2]
        mass = valid.astype(float) / 32
    else:
        raise ValueError(case)
    return states, mass / mass.sum()


def marginals(states, mass):
    p1 = mass @ states
    return np.stack([1 - p1, p1], axis=1)


def condition(states, mass, source, value):
    weights = mass * (states[:, source] == value)
    return marginals(states, weights / weights.sum())


def total_correlation(states, mass, positions):
    positions = list(positions)
    joint = {}
    for row, weight in zip(states[:, positions], mass):
        key = tuple(row)
        joint[key] = joint.get(key, 0.0) + weight
    marginal = marginals(states, mass)
    return sum(entropy(marginal[i]) for i in positions) - entropy(list(joint.values()))


def toy_records():
    records = []
    checks = {}
    for case in ("independent", "noisy_anchor", "parity"):
        states, mass = binary_joint(case)
        base = marginals(states, mass)
        before = np.asarray([entropy(p) for p in base])
        for source in range(6):
            branches = [condition(states, mass, source, value) for value in (0, 1)]
            weights = base[source]
            effects = [effect_metrics(before, [entropy(p) for p in b], source) for b in branches]
            expected_gain = sum(w * e[0] for w, e in zip(weights, effects))
            expected_paper = sum(w * e[1] for w, e in zip(weights, effects))
            influence = dependence = defect = 0.0
            for target in range(6):
                if target == source:
                    continue
                a, b, c = dependence_decomposition(base[target], [x[target] for x in branches], weights)
                influence += a
                dependence += b
                defect += c
            records.append({
                "sample_id": case, "history": "initial", "split": "diagnostic",
                "source": source, "confidence": float(base[source].max()),
                "margin": float(abs(base[source, 1] - base[source, 0])),
                "entropy": float(before[source]), "commit_nll": float(-np.log(base[source].max())),
                "external_gain": expected_gain, "paper_ig": expected_paper - before[source],
                "external_ig": expected_gain / 5 - before[source],
                "response_kl": influence, "local_dependence": dependence,
                "coherence_defect": defect, "reference_nll_gain": expected_gain,
                "source_error": None, "omitted_mass": 0.0,
            })
        if case == "parity":
            checks["parity_pairwise_tc_nats"] = [total_correlation(states, mass, pair) for pair in ((0, 1), (0, 2), (1, 2))]
            checks["parity_triple_tc_nats"] = total_correlation(states, mass, (0, 1, 2))
    # A deliberately wrong reveal sharpens predictions of other positions.
    # This is an engineered failure diagnostic, not a trained-model result.
    before = np.asarray([entropy([.1, .9]), entropy([.5, .5]), entropy([.5, .5])])
    after = np.asarray([0.0, entropy([.01, .99]), entropy([.01, .99])])
    checks["wrong_anchor"] = {
        "external_entropy_gain": effect_metrics(before, after, 0)[0],
        "remaining_reference_nll_gain": 2 * (np.log(.01) - np.log(.5)),
        "source_commit_wrong": True,
    }
    # Explicit removal artifact: independent targets have zero external gain.
    before = [entropy([.5, .5]), entropy([.95, .05])]
    checks["independent_removal"] = dict(zip(("external_gain", "paper_state_gain"), effect_metrics(before, before, 0)))
    return records, checks


def policy_scores(rows, name, alpha):
    if name == "confidence":
        return [r["confidence"] for r in rows]
    if name == "margin":
        return [r["margin"] for r in rows]
    if name == "entropy":
        return [-r["entropy"] for r in rows]
    if name in ("paper_ig", "external_ig"):
        return [r[name] for r in rows]
    if name in ("external_gain", "expected_external_gain", "response_kl", "response_fisher", "raw_gram"):
        return [r[name] - alpha * r["commit_nll"] for r in rows]
    raise ValueError(name)


def summarize(records, alphas):
    groups = {}
    for row in records:
        key = (row["split"], row["sample_id"], row["history"])
        groups.setdefault(key, []).append(row)
    available = set.intersection(*(set(row) for row in records))
    rows_out = []
    for name in ("confidence", "margin", "entropy", "paper_ig", "external_ig", "external_gain", "expected_external_gain", "response_kl", "response_fisher", "raw_gram"):
        if name not in available and name not in ("confidence", "margin", "entropy"):
            continue
        for alpha in (alphas if name in ("external_gain", "expected_external_gain", "response_kl", "response_fisher", "raw_gram") else [0.0]):
            by_sample = {}
            for (split, sample, _history), rows in groups.items():
                scores = policy_scores(rows, name, alpha)
                # Explicit deterministic tie-breaking; candidates shared by every arm.
                pick = max(range(len(rows)), key=lambda i: (scores[i], -rows[i]["source"]))
                chosen = rows[pick]
                utility = chosen["reference_nll_gain"]
                regret = max(r["reference_nll_gain"] for r in rows) - utility
                metrics = {"reference_nll_gain": utility, "shortlist_regret": regret,
                           "source_error": chosen["source_error"], "omitted_mass": chosen["omitted_mass"]}
                by_sample.setdefault((split, sample), []).append(metrics)
            for split in sorted({x[0] for x in by_sample}):
                samples = [v for (s, _), v in by_sample.items() if s == split]
                out = {"split": split, "policy": name, "alpha": alpha, "n_samples": len(samples)}
                for key in ("reference_nll_gain", "shortlist_regret", "source_error", "omitted_mass"):
                    means = [np.mean([r[key] for r in sample if r[key] is not None]) for sample in samples if any(r[key] is not None for r in sample)]
                    out[key] = float(np.mean(means)) if means else None
                rows_out.append(out)
    return rows_out


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    toy = sub.add_parser("toy", help="Exact CPU diagnostics; no model or downloads")
    toy.add_argument("--out", type=Path, default=Path("outputs/infogain_lab/toy.json"))
    sweep = sub.add_parser("sweep", help="Compare singleton scores using cached branches")
    sweep.add_argument("cache", type=Path)
    sweep.add_argument("--alpha", default="0,0.25,1")
    sweep.add_argument("--out", type=Path, default=Path("outputs/infogain_lab/sweep.csv"))
    args = parser.parse_args()
    if args.command == "toy":
        records, checks = toy_records()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"schema": 1, "kind": "exact_toy_diagnostics", "checks": checks, "records": records}, indent=2, allow_nan=False) + "\n")
        print(json.dumps(checks, indent=2))
        print(f"Cache: {args.out}")
    else:
        payload = json.loads(args.cache.read_text())
        if payload.get("schema") != 1 or not payload.get("records"):
            raise ValueError("expected nonempty schema=1 measurement cache")
        alphas = [float(x) for x in args.alpha.split(",")]
        if not all(np.isfinite(alphas)):
            raise ValueError("alpha must be finite")
        result = summarize(payload["records"], alphas)
        write_csv(args.out, result)
        print(f"Wrote {len(result)} comparisons to {args.out}; histories averaged within sample.")


if __name__ == "__main__":
    main()
