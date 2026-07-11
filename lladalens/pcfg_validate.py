"""
Speed benchmarks for the Rust PCFGEngine.

Grammar-sampled strings; scattered visible positions (any unmask order).
Default lengths: 32, 48, 64, 128. Jt_entropy at |S| in {3, 4, 5}.
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np

from pcfg_grammar import code_grammar, Sampler
from pcfg_engine import PCFGEngine, engine_backend

BENCH_L = (32, 48, 64, 128)
JT_SIZES = (3, 4, 5)
QUERY_COUNTS = (1, 8, "all_masked")


def banner(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)


def _uniform_model(qm: np.ndarray) -> np.ndarray:
    return np.full_like(qm, 1.0 / len(qm), dtype=float)


def _time_ms(fn, eng=None) -> float:
    if eng is not None and hasattr(eng, "clear_cache"):
        eng.clear_cache()
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


def sample_string(sampler: Sampler, L: int, max_tries: int, seed: int = 0):
    for i in range(max_tries):
        tokens = sampler.sample_exact_length(L, max_tries=1, seed=seed + i)
        if tokens is not None:
            return tokens
    return None


def make_scenario(eng: PCFGEngine, tokens: list[str], rng: random.Random):
    """~half positions visible, scattered (any unmask order)."""
    tmap = eng.t_idx
    L = len(tokens)
    n_vis = max(1, L // 2)
    visible = sorted(rng.sample(range(L), n_vis))
    committed = {i: tmap[tokens[i]] for i in visible}
    masked = [p for p in range(L) if p not in committed]
    return committed, masked


def _query_subset(masked: list[int], count):
    if count == "all_masked":
        return masked
    return masked[: min(int(count), len(masked))]


def _record(rows: list, L: int, function: str, detail: str, ms: float):
    rows.append((L, function, detail, ms))
    print(f"   {L:>4}  {function:<24}  {detail:<24}  {ms:>10.1f}", flush=True)


def bench_engine(
    eng: PCFGEngine,
    sampler: Sampler,
    lengths: tuple[int, ...],
    jt_sizes: tuple[int, ...] = JT_SIZES,
    seed: int = 7,
):
    rows = []
    rng = random.Random(seed)

    print(f"   backend={engine_backend(eng)}  V={eng.V}  NT={eng.n}", flush=True)
    print(f"   commit pattern: scattered (~L/2 visible)", flush=True)
    print(f"   Jt sizes: {list(jt_sizes)}", flush=True)
    print(flush=True)
    print(f"   {'L':>4}  {'function':<24}  {'detail':<24}  {'ms':>10}", flush=True)
    print("   " + "-" * 68, flush=True)

    for L in lengths:
        t0 = time.perf_counter()
        tokens = sample_string(sampler, L, 4000, seed=seed + L)
        sample_ms = (time.perf_counter() - t0) * 1e3
        if tokens is None:
            print(f"   {L:>4}  SKIP  could not sample length-{L}", flush=True)
            continue

        committed, masked = make_scenario(eng, tokens, rng)
        S2 = masked[:2] if len(masked) >= 2 else masked

        _record(rows, L, "sample", f"visible={len(committed)}", sample_ms)
        _record(rows, L, "Z", "", _time_ms(lambda: eng.Z(L, committed), eng))
        _record(rows, L, "all_marginals", f"L={L}", _time_ms(lambda: eng.all_marginals(L, committed), eng))

        for nq in QUERY_COUNTS:
            qs = _query_subset(masked, nq)
            label = len(qs) if nq != "all_masked" else f"all({len(qs)})"
            _record(
                rows, L, "marginals_batch", f"n={label}",
                _time_ms(lambda qs=qs: eng.marginals_batch(L, committed, qs), eng),
            )

        if S2:
            _record(
                rows, L, "Mt", f"|S|={len(S2)}",
                _time_ms(lambda: eng.Mt(L, committed, S2, _uniform_model), eng),
            )

        for k in jt_sizes:
            Sk = masked[:k]
            if len(Sk) < k:
                _record(rows, L, "Jt_entropy", f"|S|={k}", float("nan"))
                print(f"          (skipped: only {len(masked)} masked positions)", flush=True)
                continue
            _record(
                rows, L, "Jt_entropy", f"|S|={k}",
                _time_ms(lambda Sk=Sk: eng.Jt_entropy(L, committed, Sk), eng),
            )

        def _state_path():
            st = eng.make_state(L, committed)
            st.marginals_batch(masked[: min(8, len(masked))])
            if masked:
                st.commit(masked[0], 0)
            st.all_marginals()

        _record(rows, L, "PCFGState", "8q+1commit+all", _time_ms(_state_path, eng))
        print(flush=True)

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=list(BENCH_L))
    p.add_argument("--jt-sizes", type=int, nargs="+", default=list(JT_SIZES))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    lengths = tuple(args.lengths)
    jt_sizes = tuple(args.jt_sizes)
    g = code_grammar()
    sampler = Sampler(g, max_depth=350, workers=1)
    eng = PCFGEngine(g.to_cnf())
    banner(f"PCFGEngine speed @ L in {lengths} (rust)")
    bench_engine(eng, sampler, lengths=lengths, jt_sizes=jt_sizes, seed=args.seed)


if __name__ == "__main__":
    main()
