"""CLI: pool sizes and full ``sample_uniform`` output. Run: ``python -m cst.train_data``."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Print train_data pool sizes and samples.")
    parser.add_argument(
        "--samples",
        type=int,
        default=24,
        help="Rows from sample_uniform (printed in full).",
    )
    args = parser.parse_args()

    from .compose import load_pools, sample_uniform

    pools = load_pools()
    print("=== Pool sizes ===")
    for name in sorted(pools.keys()):
        print(f"  {name}: {len(pools[name])}")
    print()

    if args.samples > 0:
        mix = sample_uniform(args.samples)
        print(f"=== sample_uniform(n={args.samples}) — counts per source ===")
        from collections import Counter

        c = Counter(r["source"] for r in mix)
        from .compose import default_source_specs

        for spec in default_source_specs():
            name = spec.name
            n_pool = len(pools.get(name, []))
            if n_pool == 0:
                print(f"  {name}: (empty pool)")
                continue
            print(f"  {name}: {c.get(name, 0)}  (pool {n_pool})")
        print()
        print("=== Samples (full text) ===")
        for i, row in enumerate(mix):
            print(f"--- [{i}] {row.get('source')} {row.get('id')} ---")
            print(row.get("text", ""))
            print()


if __name__ == "__main__":
    main()
