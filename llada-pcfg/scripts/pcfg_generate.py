"""
Sample PCFG strings and publish raw corpora (before LLaDA masking).

One generation stream: sample until the train token budget is met, then hold out
a fixed number of strings for test (shuffled split, disjoint from train).

Outputs a HuggingFace DatasetDict locally and optionally pushes to the Hub.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from utils import ensure_import_paths, pcfg_corpus_dir, resolve_repo_path

ensure_import_paths()

from hf_hub import apply_pcfg_hf_env, push_pcfg_dataset, resolve_hf_token, resolve_pcfg_repo
from pcfg_grammar import Sampler, code_grammar, default_workers, sample_one

L_MIN = 64
L_MAX = 127  # max content tokens; + <eos> fits ARCH_LEN=128
DEFAULT_MAX_DEPTH = 350
PROGRESS_INTERVAL = 250_000


@dataclass
class SampleRecord:
    id: int
    tokens: list[str]
    length: int

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tokens": self.tokens,
            "length": self.length,
            "text": " ".join(self.tokens),
        }


@dataclass
class CorpusStats:
    split: str
    n_strings: int
    total_tokens: int
    min_length: int
    max_length: int
    target_tokens: int | None = None
    target_strings: int | None = None


@dataclass
class GrammarCapacity:
    """Rough capacity notes for code_grammar() in [L_MIN, L_MAX]."""
    min_len: int
    max_len: int
    n_terminals: int
    mean_length_empirical: float | None
    language: str
    distinct_strings_note: str
    tokens_note: str

    def summary(self) -> str:
        lines = [
            f"Length range: [{self.min_len}, {self.max_len}]",
            f"Terminals: {self.n_terminals}",
            f"Language: {self.language}",
            f"Distinct strings: {self.distinct_strings_note}",
            f"Tokens you can generate: {self.tokens_note}",
        ]
        if self.mean_length_empirical is not None:
            est = int(1_000_000 / self.mean_length_empirical)
            lines.append(
                f"Mean sampled length ≈ {self.mean_length_empirical:.1f} "
                f"→ ~{est:,} strings per 1M tokens (train portion, before holdout)"
            )
        return "\n".join(lines)



def _stats_for(records: list[SampleRecord], split: str, **targets) -> CorpusStats:
    if not records:
        return CorpusStats(
            split=split,
            n_strings=0,
            total_tokens=0,
            min_length=0,
            max_length=0,
            **targets,
        )
    lengths = [r.length for r in records]
    return CorpusStats(
        split=split,
        n_strings=len(records),
        total_tokens=sum(lengths),
        min_length=min(lengths),
        max_length=max(lengths),
        **targets,
    )


def split_holdout(
    pool: list[SampleRecord],
    test_strings: int,
    *,
    rng: random.Random,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Shuffle pool and peel off test_strings for test; remainder is train."""
    if test_strings < 0:
        raise ValueError("test_strings must be non-negative")
    if test_strings == 0:
        return pool, []
    if len(pool) <= test_strings:
        raise ValueError(
            f"pool has {len(pool)} strings but need > {test_strings} to leave a train split"
        )
    shuffled = list(pool)
    rng.shuffle(shuffled)
    test_recs = shuffled[:test_strings]
    train_recs = shuffled[test_strings:]
    for i, rec in enumerate(train_recs):
        rec.id = i
    for i, rec in enumerate(test_recs):
        rec.id = i
    return train_recs, test_recs


def _pool_target_tokens(target_tokens: int, test_strings: int, max_len: int) -> int:
    """Conservative pool size so train split can meet target after holdout."""
    if test_strings <= 0:
        return target_tokens
    return target_tokens + test_strings * max_len


def _records_from_token_lists(token_lists: list[list[str]], start_id: int = 0) -> list[SampleRecord]:
    return [
        SampleRecord(id=start_id + i, tokens=tokens, length=len(tokens))
        for i, tokens in enumerate(token_lists)
    ]


def sample_corpus_with_holdout(
    target_tokens: int,
    test_strings: int,
    *,
    min_len: int = L_MIN,
    max_len: int = L_MAX,
    max_depth: int = DEFAULT_MAX_DEPTH,
    grammar=None,
    rng: random.Random | None = None,
    split_seed: int | None = None,
    workers: int | None = None,
    progress_interval: int = PROGRESS_INTERVAL,
    log_fn=print,
) -> tuple[list[SampleRecord], list[SampleRecord], CorpusStats, CorpusStats]:
    """
    Sample one pool until the train portion (after holdout) has >= target_tokens.
    Then shuffle and assign test_strings to test; the rest is train.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if test_strings < 0:
        raise ValueError("test_strings must be non-negative")
    grammar = grammar or code_grammar()
    rng = rng or random.Random()
    split_rng = random.Random(split_seed if split_seed is not None else rng.randint(0, 2**31 - 1))
    workers = default_workers() if workers is None else max(1, workers)

    sampler = Sampler(grammar, workers=workers, max_depth=max_depth)
    pool_target = _pool_target_tokens(target_tokens, test_strings, max_len)
    seed = rng.randint(0, 2**31 - 1)

    log_fn(
        f"  sampler: rust workers={workers} pool_target={pool_target:,} tokens",
        flush=True,
    )
    t0 = time.perf_counter()
    token_lists, pool_tokens = sampler.fill_pool(
        pool_target,
        min_len=min_len,
        max_len=max_len,
        seed=seed,
        progress_interval=progress_interval,
        log_fn=log_fn,
    )
    elapsed = time.perf_counter() - t0
    if pool_tokens < pool_target and test_strings == 0:
        log_fn(f"  warning: pool_tokens={pool_tokens:,} < target={pool_target:,}", flush=True)
    log_fn(
        f"  sampled {len(token_lists):,} strings, {pool_tokens:,} tokens in {elapsed:.1f}s "
        f"({pool_tokens / max(elapsed, 1e-9):,.0f} tok/s)",
        flush=True,
    )

    pool = _records_from_token_lists(token_lists)
    if test_strings == 0:
        train_recs, test_recs = pool, []
    else:
        train_recs, test_recs = split_holdout(pool, test_strings, rng=split_rng)
        train_tokens = sum(r.length for r in train_recs)
        if train_tokens < target_tokens:
            extra_target = target_tokens - train_tokens
            log_fn(
                f"  holdout shortfall: train={train_tokens:,} < {target_tokens:,}; "
                f"sampling {extra_target:,} more tokens",
                flush=True,
            )
            extra_lists, _ = sampler.fill_pool(
                extra_target + test_strings * max_len,
                min_len=min_len,
                max_len=max_len,
                seed=seed + 1,
                progress_interval=progress_interval,
                log_fn=log_fn,
            )
            pool.extend(_records_from_token_lists(extra_lists, start_id=len(pool)))
            train_recs, test_recs = split_holdout(pool, test_strings, rng=split_rng)

    train_stats = _stats_for(
        train_recs, "train", target_tokens=target_tokens, target_strings=None
    )
    test_stats = _stats_for(
        test_recs,
        "test",
        target_tokens=None,
        target_strings=test_strings if test_strings else 0,
    )
    return train_recs, test_recs, train_stats, test_stats


def sample_train_corpus(
    target_tokens: int,
    *,
    min_len: int = L_MIN,
    max_len: int = L_MAX,
    max_depth: int = DEFAULT_MAX_DEPTH,
    grammar=None,
    rng: random.Random | None = None,
    start_id: int = 0,
    workers: int | None = None,
    progress_interval: int = PROGRESS_INTERVAL,
    log_fn=print,
) -> tuple[list[SampleRecord], CorpusStats]:
    """Sample strings until total token count >= target_tokens (no holdout)."""
    train, _, stats, _ = sample_corpus_with_holdout(
        target_tokens,
        0,
        min_len=min_len,
        max_len=max_len,
        max_depth=max_depth,
        grammar=grammar,
        rng=rng,
        workers=workers,
        progress_interval=progress_interval,
        log_fn=log_fn,
    )
    for i, rec in enumerate(train):
        rec.id = start_id + i
    stats.target_strings = None
    return train, stats


def estimate_grammar_capacity(
    *,
    n_probe: int = 5_000,
    min_len: int = L_MIN,
    max_len: int = L_MAX,
    seed: int = 0,
) -> GrammarCapacity:
    """Empirical probe for mean length and collision rate."""
    grammar = code_grammar()
    sampler = Sampler(grammar, max_depth=DEFAULT_MAX_DEPTH)
    seen: set[tuple[str, ...]] = set()
    lengths: list[int] = []
    collisions = 0
    batch = sampler.sample_batch(n_probe, min_len, max_len, seed=seed)
    for tokens in batch:
        key = tuple(tokens)
        if key in seen:
            collisions += 1
        seen.add(key)
        lengths.append(len(tokens))

    mean_len = sum(lengths) / len(lengths) if lengths else None
    collision_note = (
        f"0 collisions in {n_probe:,} samples — space is enormous"
        if collisions == 0
        else f"{collisions:,} collisions in {n_probe:,} samples"
    )
    return GrammarCapacity(
        min_len=min_len,
        max_len=max_len,
        n_terminals=len(grammar.terminals),
        mean_length_empirical=mean_len,
        language="infinite (recursive PCFG); each fixed L has finitely many parses",
        distinct_strings_note=(
            f"{collision_note}; realistic corpora will not exhaust distinct strings in [64,128]"
        ),
        tokens_note="unlimited — bounded only by compute/storage; sample as many as you need",
    )


def records_to_dataset(records: list[SampleRecord]):
    from datasets import Dataset

    return Dataset.from_list([r.to_row() for r in records])


def build_dataset_dict(
    train_records: list[SampleRecord],
    test_records: list[SampleRecord],
    *,
    train_stats: CorpusStats | None = None,
    test_stats: CorpusStats | None = None,
):
    from datasets import DatasetDict

    splits = {"train": records_to_dataset(train_records)}
    if test_records:
        splits["test"] = records_to_dataset(test_records)
    ddict = DatasetDict(splits)
    meta = {}
    if train_stats is not None:
        meta["train"] = asdict(train_stats)
    if test_stats is not None and test_records:
        meta["test"] = asdict(test_stats)
    return ddict, meta


def save_corpus(
    ddict,
    out_dir: Path,
    *,
    meta: dict | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ddict.save_to_disk(str(out_dir))
    if meta is not None:
        (out_dir / "generation_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def push_corpus(
    ddict,
    *,
    repo_id: str | None = None,
    dataset: str | None = None,
    namespace: str | None = None,
    private: bool = False,
    token: str | None = None,
) -> str:
    return push_pcfg_dataset(
        ddict,
        repo_id=repo_id,
        dataset=dataset,
        namespace=namespace,
        private=private,
        token=token,
    )


def generate_and_publish(
    *,
    train_tokens: int,
    test_strings: int,
    min_len: int = L_MIN,
    max_len: int = L_MAX,
    max_depth: int = DEFAULT_MAX_DEPTH,
    seed: int = 0,
    out_dir: Path | None = None,
    repo_id: str | None = None,
    dataset: str | None = None,
    namespace: str | None = None,
    push: bool = False,
    private: bool = False,
    hf_token: str | None = None,
    print_capacity: bool = False,
    workers: int | None = None,
    progress_interval: int = PROGRESS_INTERVAL,
):
    rng = random.Random(seed)
    grammar = code_grammar()

    if print_capacity:
        cap = estimate_grammar_capacity(min_len=min_len, max_len=max_len, seed=seed)
        print(cap.summary(), flush=True)
        print(flush=True)

    print(
        f"sampling pool until train>={train_tokens:,} tokens, "
        f"then hold out {test_strings:,} strings for test  "
        f"L in [{min_len}, {max_len}]  seed={seed}",
        flush=True,
    )

    train_records, test_records, train_stats, test_stats = sample_corpus_with_holdout(
        train_tokens,
        test_strings,
        min_len=min_len,
        max_len=max_len,
        max_depth=max_depth,
        grammar=grammar,
        rng=rng,
        split_seed=seed + 1,
        workers=workers,
        progress_interval=progress_interval,
    )

    ddict, meta = build_dataset_dict(
        train_records, test_records, train_stats=train_stats, test_stats=test_stats
    )
    meta["seed"] = seed
    meta["split_seed"] = seed + 1
    meta["split_method"] = "holdout_from_single_pool"
    meta["min_len"] = min_len
    meta["max_len"] = max_len
    meta["max_depth"] = max_depth
    meta["sampler"] = "rust"
    meta["workers"] = workers if workers is not None else default_workers()
    meta["pool_strings"] = len(train_records) + len(test_records)
    meta["pool_tokens"] = train_stats.total_tokens + test_stats.total_tokens

    print(
        f"pool:  {meta['pool_strings']:,} strings, {meta['pool_tokens']:,} tokens",
        flush=True,
    )
    print(
        f"train: {train_stats.n_strings:,} strings, {train_stats.total_tokens:,} tokens "
        f"(len {train_stats.min_length}-{train_stats.max_length})",
        flush=True,
    )
    if test_records:
        print(
            f"test:  {test_stats.n_strings:,} strings, {test_stats.total_tokens:,} tokens "
            f"(len {test_stats.min_length}-{test_stats.max_length})  [held out, disjoint]",
            flush=True,
        )

    saved = None
    if out_dir is not None:
        saved = save_corpus(ddict, out_dir, meta=meta)
        print(f"saved -> {saved}", flush=True)

    if push:
        full_repo = push_corpus(
            ddict,
            repo_id=repo_id,
            dataset=dataset,
            namespace=namespace,
            private=private,
            token=hf_token,
        )
        print(f"pushed -> https://huggingface.co/datasets/{full_repo}", flush=True)
    elif repo_id or dataset:
        full_repo = resolve_pcfg_repo(repo_id, dataset=dataset, namespace=namespace)
        print(f"hub target (dry run): https://huggingface.co/datasets/{full_repo}", flush=True)

    return ddict, meta, saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-tokens", type=int, required=True,
                   help="Train split token budget (after test holdout).")
    p.add_argument("--test-strings", type=int, default=0,
                   help="Strings held out from the sampled pool for test (disjoint from train).")
    p.add_argument("--min-len", type=int, default=L_MIN)
    p.add_argument("--max-len", type=int, default=L_MAX)
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--capacity", action="store_true",
                   help="Print grammar capacity notes before sampling.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Local HF dataset directory (save_to_disk; default data/pcfg).")
    p.add_argument("--no-save", action="store_true", help="Skip local save_to_disk.")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Rust sampler threads (default {default_workers()}).",
    )
    p.add_argument(
        "--progress-interval",
        type=int,
        default=PROGRESS_INTERVAL,
        help="Log progress every N pool tokens (0 to disable).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Short Hub dataset name (default: PCFG_HF_DATASET or pcfg-code; "
        "namespace from --namespace / PCFG_HF_NAMESPACE / whoami).",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Full Hub dataset repo, e.g. username/pcfg-code (overrides --dataset).",
    )
    p.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="HF user/org when --dataset has no slash (else PCFG_HF_NAMESPACE / whoami).",
    )
    p.add_argument("--push", action="store_true", help="push_to_hub after generation.")
    p.add_argument("--private", action="store_true", help="Create private HF dataset.")
    p.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HF token (default: HF_TOKEN env or repo_env_defaults).",
    )
    args = p.parse_args()

    apply_pcfg_hf_env()
    out_dir = None
    if not args.no_save:
        out_dir = resolve_repo_path(args.out_dir) or pcfg_corpus_dir()
    generate_and_publish(
        train_tokens=args.train_tokens,
        test_strings=args.test_strings,
        min_len=args.min_len,
        max_len=args.max_len,
        max_depth=args.max_depth,
        seed=args.seed,
        out_dir=out_dir,
        repo_id=args.repo_id,
        dataset=args.dataset,
        namespace=args.namespace,
        push=args.push,
        private=args.private,
        hf_token=resolve_hf_token(args.hf_token),
        print_capacity=args.capacity,
        workers=args.workers,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
