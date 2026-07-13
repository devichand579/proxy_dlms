"""
Training corpus generation (no engine) and inference-time exact q via PCFGEngine.

Raw grammar strings: pcfg_generate.py (L in [64, 127], HF export; max content + <eos> fits ARCH_LEN).
Training (LLaDA-style):
  - Per example: draw t ~ Uniform[0, 1], mask each content token independently
  - Loss weight 1/(t·L) is applied in the training loop (Eq. 3)

Inference:
  - User/model provides visible left prefix; right suffix is masked
  - Sequences end with <eos> then <pad> fill; engine uses grammar tokens only
"""
from dataclasses import dataclass
import random
import numpy as np

from pcfg_grammar import Sampler, code_grammar
from pcfg_engine import PCFGEngine
from pcfg_generate import (
    L_MIN as GEN_L_MIN,
    L_MAX as GEN_L_MAX,
    sample_train_corpus,
    sample_corpus_with_holdout,
)

# Re-export generation defaults (single source of truth in pcfg_generate).
L_MIN = GEN_L_MIN
L_MAX = GEN_L_MAX
ARCH_LEN = 128
MAX_CONTENT_LEN = ARCH_LEN - 1  # one slot reserved for <eos>
PAD = "<pad>"
MASK = "<mask>"
EOS = "<eos>"
LABEL_IGNORE = -100


class Vocab:
    """Grammar terminals plus PAD, MASK, and EOS for model I/O."""

    def __init__(self, grammar_terminals):
        special = [PAD, MASK, EOS]
        self.itos = special + sorted(grammar_terminals)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.mask_id = self.stoi[MASK]
        self.eos_id = self.stoi[EOS]
        self.size = len(self.itos)

    def encode(self, tokens):
        return [self.stoi[t] for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


@dataclass
class TrainExample:
    tokens: list
    n_content: int
    n_visible: int
    masked_positions: list
    mask_t: float
    input_ids: np.ndarray
    labels: np.ndarray
    attention_mask: np.ndarray


@dataclass
class InferExample:
    tokens: list
    n_content: int
    n_visible: int
    input_ids: np.ndarray
    attention_mask: np.ndarray
    query_positions: list
    marginals: dict


def sample_mask_t(rng=None):
    """Draw t ~ Uniform[0, 1] for one training example."""
    return (rng or random).random()


def _independent_mask_positions(n_content, mask_t, rng=None):
    """Mask each content index independently with probability mask_t."""
    rng = rng or random
    return [i for i in range(n_content) if rng.random() < mask_t]


def _mask_count(n_content, mask_ratio):
    return max(1, min(n_content - 1, int(round(n_content * mask_ratio))))


def _rightmost_split(n_content, mask_ratio):
    n_mask = _mask_count(n_content, mask_ratio)
    n_visible = n_content - n_mask
    return n_visible, n_mask


def corpus_length_stats(token_seqs: list[list]) -> tuple[int, int, int]:
    """Return (min_len, max_len, n_over_max_content)."""
    if not token_seqs:
        return 0, 0, 0
    lengths = [len(t) for t in token_seqs]
    over = sum(1 for n in lengths if n > MAX_CONTENT_LEN)
    return min(lengths), max(lengths), over


def fit_tokens_for_arch(tokens, arch_len: int = ARCH_LEN) -> tuple[list, bool]:
    """Truncate content to ``arch_len - 1`` so ``<eos>`` fits in the fixed window."""
    max_content = arch_len - 1
    n = len(tokens)
    if n <= max_content:
        return list(tokens), False
    return list(tokens[:max_content]), True


def fit_corpus_for_arch(
    token_seqs: list[list],
    arch_len: int = ARCH_LEN,
) -> tuple[list[list], int]:
    """Clip every sequence to fit ``ARCH_LEN`` (content + ``<eos>``). Returns (corpus, n_clipped)."""
    out: list[list] = []
    clipped = 0
    for tokens in token_seqs:
        fitted, was_clipped = fit_tokens_for_arch(tokens, arch_len=arch_len)
        if was_clipped:
            clipped += 1
        if fitted:
            out.append(fitted)
    return out, clipped


def make_train_example(tokens, vocab, arch_len=ARCH_LEN, mask_t=None, rng=None):
    """Build one padded training example using LLaDA independent masking."""
    rng = rng or random
    tokens, _ = fit_tokens_for_arch(tokens, arch_len=arch_len)
    n = len(tokens)
    if n < 1:
        raise ValueError("content length must be >= 1 after arch fit")
    t = sample_mask_t(rng) if mask_t is None else float(mask_t)
    masked_positions = _independent_mask_positions(n, t, rng=rng)
    masked_set = set(masked_positions)
    ids = vocab.encode(tokens)
    input_ids = np.full(arch_len, vocab.pad_id, dtype=np.int64)
    labels = np.full(arch_len, LABEL_IGNORE, dtype=np.int64)
    attention = np.zeros(arch_len, dtype=np.int64)
    for i in range(n):
        if i in masked_set:
            input_ids[i] = vocab.mask_id
            labels[i] = ids[i]
        else:
            input_ids[i] = ids[i]
    input_ids[n] = vocab.eos_id
    attention[: n + 1] = 1
    return TrainExample(
        tokens=list(tokens),
        n_content=n,
        n_visible=n - len(masked_positions),
        masked_positions=masked_positions,
        mask_t=t,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
    )


def generate_training_tokens(
    n_examples,
    min_len=L_MIN,
    max_len=L_MAX,
    max_depth=250,
    grammar=None,
    rng=None,
):
    """Sample exactly n_examples grammar strings (legacy count-based API)."""
    grammar = grammar or code_grammar()
    seed = rng.randint(0, 2**31 - 1) if rng else 0
    sampler = Sampler(grammar, max_depth=max_depth)
    out: list[list[str]] = []
    while len(out) < n_examples:
        batch = sampler.sample_batch(
            min(n_examples - len(out), 256),
            min_len,
            max_len,
            seed=seed,
        )
        seed += 1
        if not batch:
            break
        out.extend(batch)
    return out[:n_examples]


def generate_training_corpus(
    n_examples,
    min_len=L_MIN,
    max_len=L_MAX,
    arch_len=ARCH_LEN,
    max_depth=250,
    grammar=None,
    rng=None,
):
    """Return (token_sequences, vocab). Masking is deferred to the training loop."""
    del arch_len  # kept for API compatibility
    token_seqs = generate_training_tokens(
        n_examples,
        min_len=min_len,
        max_len=max_len,
        max_depth=max_depth,
        grammar=grammar,
        rng=rng,
    )
    grammar = grammar or code_grammar()
    vocab = Vocab(grammar.terminals)
    return token_seqs, vocab


def generate_training_corpus_budget(
    target_tokens,
    min_len=L_MIN,
    max_len=L_MAX,
    arch_len=ARCH_LEN,
    max_depth=250,
    grammar=None,
    rng=None,
):
    """Sample until total content tokens >= target_tokens."""
    del arch_len
    grammar = grammar or code_grammar()
    vocab = Vocab(grammar.terminals)
    rng_obj = rng if isinstance(rng, random.Random) else random.Random()
    records, stats = sample_train_corpus(
        target_tokens,
        min_len=min_len,
        max_len=max_len,
        max_depth=max_depth,
        grammar=grammar,
        rng=rng_obj,
    )
    token_seqs = [r.tokens for r in records]
    return token_seqs, vocab, {
        "total_content_tokens": stats.total_tokens,
        "n_examples": stats.n_strings,
    }


def load_corpus_from_disk(path):
    """Load raw token corpus saved by pcfg_generate (HF save_to_disk layout)."""
    from datasets import load_from_disk

    ddict = load_from_disk(str(path))
    train = [row["tokens"] for row in ddict["train"]]
    test = [row["tokens"] for row in ddict["test"]] if "test" in ddict else []
    return train, test, ddict


def load_corpus_from_hub(
    repo_id: str | None = None,
    *,
    dataset: str | None = None,
    namespace: str | None = None,
    token: str | None = None,
):
    """Load raw token corpus from HuggingFace Hub."""
    from hf_hub import load_pcfg_dataset, resolve_hf_token

    ddict = load_pcfg_dataset(
        repo_id,
        dataset=dataset,
        namespace=namespace,
        token=resolve_hf_token(token),
    )
    train = [row["tokens"] for row in ddict["train"]]
    test = [row["tokens"] for row in ddict["test"]] if "test" in ddict else []
    return train, test, ddict


def make_infer_input(tokens, n_visible, vocab, arch_len=ARCH_LEN):
    """Build padded inference input (visible prefix + MASK suffix + EOS). No engine yet."""
    tokens, _ = fit_tokens_for_arch(tokens, arch_len=arch_len)
    n = len(tokens)
    if not (0 < n_visible < n):
        raise ValueError("require 0 < n_visible < n_content")
    ids = vocab.encode(tokens)
    input_ids = np.full(arch_len, vocab.pad_id, dtype=np.int64)
    attention = np.zeros(arch_len, dtype=np.int64)
    input_ids[:n_visible] = ids[:n_visible]
    input_ids[n_visible:n] = vocab.mask_id
    input_ids[n] = vocab.eos_id
    attention[: n + 1] = 1
    query_positions = list(range(n_visible, n))
    return input_ids, attention, query_positions, n


def build_infer_state(engine, tokens, n_visible):
    """PCFGState for scheduler-driven inference (incremental commits)."""
    tmap = engine.t_idx
    committed = {i: tmap[tokens[i]] for i in range(n_visible)}
    return engine.make_state(len(tokens), committed)


def infer_marginals(
    engine,
    tokens,
    n_visible,
    vocab,
    arch_len=ARCH_LEN,
    query_positions=None,
    use_log=None,
    state=None,
):
    """Exact q(x_p | visible tokens). Pass state= to reuse incremental chart."""
    _ = use_log
    _, attention, default_query, n = make_infer_input(tokens, n_visible, vocab, arch_len)
    positions = query_positions if query_positions is not None else default_query
    if state is None:
        tmap = engine.t_idx
        prefix = {i: tmap[tokens[i]] for i in range(n_visible)}
        marginals = engine.marginals_batch(n, prefix, query_positions=positions)
    else:
        marginals = state.marginals_batch(positions)
    input_ids, _, _, _ = make_infer_input(tokens, n_visible, vocab, arch_len)
    return InferExample(
        tokens=list(tokens),
        n_content=n,
        n_visible=n_visible,
        input_ids=input_ids,
        attention_mask=attention,
        query_positions=list(positions),
        marginals=marginals,
    )


def infer_from_mask_ratio(
    engine,
    tokens,
    mask_ratio,
    vocab,
    arch_len=ARCH_LEN,
    query_positions=None,
    use_log=None,
):
    """Inference with rightmost mask_ratio of content masked."""
    n_visible, _ = _rightmost_split(len(tokens), mask_ratio)
    return infer_marginals(
        engine, tokens, n_visible, vocab, arch_len,
        query_positions=query_positions, use_log=use_log,
    )


def build_stack():
    """Grammar -> CNF -> engine + vocab (for inference pipelines)."""
    g = code_grammar()
    cnf = g.to_cnf()
    eng = PCFGEngine(cnf)
    vocab = Vocab(g.terminals)
    return g, cnf, eng, vocab
