"""
PCFGs in readable form (rules may have any RHS length, mixed
terminals/nonterminals, unit rules), then convert to Chomsky Normal Form
automatically so the CKY/inside engine can consume them.

Corpus sampling is implemented in Rust (pcfg_engine_rs.PyPCFGSampler).
"""
from __future__ import annotations

import os
from collections import defaultdict


class Grammar:
    def __init__(self, rules, start, terminals):
        # rules: list of (head:str, rhs:tuple[str], prob:float)
        self.rules = [(h, tuple(r), float(p)) for (h, r, p) in rules]
        self.start = start
        self.terminals = set(terminals)
        self._check()

    def is_term(self, s): return s in self.terminals

    def _check(self):
        tot = defaultdict(float)
        for h, r, p in self.rules: tot[h] += p
        for h, s in tot.items():
            assert abs(s - 1.0) < 1e-9, f"rules for {h} sum to {s}, not 1"

    # ---------------- CNF conversion ----------------
    def to_cnf(self):
        rules = list(self.rules)
        start = self.start

        # 0) fresh start if start appears on any RHS
        if any(start in r for _, r, _ in rules):
            rules.append(('_S0', (start,), 1.0)); start = '_S0'

        # 1) TERM: in rules of length >=2, replace terminals by preterminals
        term_pre = {}
        out = []
        for h, r, p in rules:
            if len(r) >= 2:
                nr = []
                for s in r:
                    if self.is_term(s):
                        if s not in term_pre:
                            term_pre[s] = f'_PT{len(term_pre)}'
                        nr.append(term_pre[s])
                    else:
                        nr.append(s)
                out.append((h, tuple(nr), p))
            else:
                out.append((h, r, p))
        for t, name in term_pre.items():
            out.append((name, (t,), 1.0))
        rules = out

        # 2) BIN: binarize RHS of length > 2
        out = []; bc = 0
        for h, r, p in rules:
            if len(r) <= 2:
                out.append((h, r, p))
            else:
                cur, prob = h, p
                for i in range(len(r) - 2):
                    nxt = f'_BIN{bc}'; bc += 1
                    out.append((cur, (r[i], nxt), prob))
                    cur, prob = nxt, 1.0
                out.append((cur, (r[-2], r[-1]), prob))
        rules = out

        # 3) UNIT: eliminate unit rules A->B (B nonterminal), acyclic assumed
        unit_out = defaultdict(list); nonunit = []
        for h, r, p in rules:
            if len(r) == 1 and not self.is_term(r[0]):
                unit_out[h].append((r[0], p))
            else:
                nonunit.append((h, r, p))
        memo = {}
        def closure(A):
            if A in memo: return memo[A]
            res = defaultdict(float); res[A] += 1.0
            for B, p in unit_out.get(A, []):
                for C, u in closure(B).items():
                    res[C] += p * u
            memo[A] = dict(res); return memo[A]

        heads = set(h for h, _, _ in rules)
        heads |= set(s for _, r, _ in rules for s in r if not self.is_term(s))
        nonunit_by_head = defaultdict(list)
        for h, r, p in nonunit: nonunit_by_head[h].append((r, p))

        merged = defaultdict(float)  # (head, rhs) -> prob
        for A in heads:
            for C, u in closure(A).items():
                for r, q in nonunit_by_head.get(C, []):
                    merged[(A, r)] += u * q

        # split into terminal rules and binary rules
        term_rules = defaultdict(dict)   # head -> {terminal: prob}
        bin_rules = []                   # (head, B, C, prob)
        for (h, r), p in merged.items():
            if len(r) == 1:
                assert self.is_term(r[0])
                term_rules[h][r[0]] = term_rules[h].get(r[0], 0.0) + p
            else:
                bin_rules.append((h, r[0], r[1], p))

        # normalization sanity
        tot = defaultdict(float)
        for h, d in term_rules.items():
            for _, p in d.items(): tot[h] += p
        for h, B, C, p in bin_rules: tot[h] += p
        for h, s in tot.items():
            assert abs(s - 1.0) < 1e-7, f"CNF rules for {h} sum to {s}"

        nonterminals = set(term_rules) | set(h for h, _, _, _ in bin_rules)
        nonterminals |= {start}
        terminals = sorted({t for d in term_rules.values() for t in d})
        return CNFGrammar(nonterminals, terminals, dict(term_rules), bin_rules, start)


class CNFGrammar:
    def __init__(self, nonterminals, terminals, term_rules, bin_rules, start):
        self.NT = sorted(nonterminals)
        self.terminals = list(terminals)
        self.term_rules = term_rules
        self.bin_rules = bin_rules
        self.start = start
    def __repr__(self):
        return (f"CNFGrammar(|NT|={len(self.NT)}, |V|={len(self.terminals)}, "
                f"#term_rules={sum(len(d) for d in self.term_rules.values())}, "
                f"#bin_rules={len(self.bin_rules)}, start={self.start})")


# ===================== grammar definition =====================

def code_grammar():
    """Code-like PCFG for L=64..127 training/inference (syntax only)."""
    T = {
        'int', 'float', 'bool', 'void', 'char',
        'if', 'else', 'while', 'for', 'switch', 'case', 'break', 'cont', 'ret',
        'const', 'struct',
        '(', ')', '{', '}', '[', ']', ';', ',', ':',
        '=', 'pluseq', 'minuseq',
        '+', '-', '*', '/', '%',
        'lt', 'gt', 'le', 'ge', 'eq', 'ne',
        'and', 'or', 'not', 'band', 'bor',
        'id', 'fld', 'num', 'flt',
    }
    rules = [
        ('Prog', ('Stmt',), 0.30),
        ('Prog', ('Stmt', 'Prog'), 0.70),

        ('Stmt', ('Decl',), 0.18),
        ('Stmt', ('Assign',), 0.22),
        ('Stmt', ('If',), 0.14),
        ('Stmt', ('While',), 0.08),
        ('Stmt', ('For',), 0.08),
        ('Stmt', ('Switch',), 0.05),
        ('Stmt', ('Break',), 0.03),
        ('Stmt', ('Cont',), 0.02),
        ('Stmt', ('Ret',), 0.10),
        ('Stmt', ('Struct',), 0.05),
        ('Stmt', ('Block',), 0.05),

        ('Decl', ('Type', 'id', ';'), 0.55),
        ('Decl', ('const', 'Type', 'id', ';'), 0.45),
        ('Type', ('int',), 0.35),
        ('Type', ('float',), 0.25),
        ('Type', ('bool',), 0.20),
        ('Type', ('char',), 0.10),
        ('Type', ('void',), 0.10),

        ('Assign', ('LVal', '=', 'Expr', ';'), 0.50),
        ('Assign', ('LVal', 'pluseq', 'Expr', ';'), 0.25),
        ('Assign', ('LVal', 'minuseq', 'Expr', ';'), 0.25),
        ('LVal', ('id',), 0.55),
        ('LVal', ('id', '[', 'Expr', ']'), 0.30),
        ('LVal', ('fld',), 0.15),

        ('If', ('if', '(', 'Cond', ')', 'Block'), 0.55),
        ('If', ('if', '(', 'Cond', ')', 'Block', 'else', 'Block'), 0.45),
        ('While', ('while', '(', 'Cond', ')', 'Block'), 1.0),
        ('For', ('for', '(', 'Assign', 'Cond', ';', 'Assign', ')', 'Block'), 1.0),
        ('Switch', ('switch', '(', 'Expr', ')', 'Cases'), 1.0),
        ('Cases', ('Case', 'Cases'), 0.60),
        ('Cases', ('Case',), 0.40),
        ('Case', ('case', 'num', ':', 'Stmt'), 1.0),
        ('Break', ('break', ';'), 1.0),
        ('Cont', ('cont', ';'), 1.0),
        ('Ret', ('ret', 'Expr', ';'), 0.70),
        ('Ret', ('ret', ';'), 0.30),
        ('Struct', ('struct', 'id', '{', 'Decl', '}'), 0.60),
        ('Struct', ('struct', 'id', '{', 'Decl', 'Decl', '}'), 0.40),
        ('Block', ('{', 'Prog', '}'), 1.0),

        ('Cond', ('Expr', 'lt', 'Expr'), 0.20),
        ('Cond', ('Expr', 'gt', 'Expr'), 0.15),
        ('Cond', ('Expr', 'le', 'Expr'), 0.15),
        ('Cond', ('Expr', 'ge', 'Expr'), 0.15),
        ('Cond', ('Expr', 'eq', 'Expr'), 0.15),
        ('Cond', ('Expr', 'ne', 'Expr'), 0.10),
        ('Cond', ('Cond', 'and', 'Cond'), 0.05),
        ('Cond', ('Cond', 'or', 'Cond'), 0.03),
        ('Cond', ('not', 'Cond'), 0.02),

        ('Expr', ('Expr', 'band', 'Term'), 0.08),
        ('Expr', ('Expr', 'bor', 'Term'), 0.07),
        ('Expr', ('Expr', '+', 'Term'), 0.20),
        ('Expr', ('Expr', '-', 'Term'), 0.15),
        ('Expr', ('Term',), 0.50),

        ('Term', ('Term', '*', 'Factor'), 0.25),
        ('Term', ('Term', '/', 'Factor'), 0.15),
        ('Term', ('Term', '%', 'Factor'), 0.10),
        ('Term', ('Factor',), 0.50),

        ('Factor', ('(', 'Expr', ')'), 0.15),
        ('Factor', ('id', '[', 'Expr', ']'), 0.12),
        ('Factor', ('id',), 0.35),
        ('Factor', ('num',), 0.23),
        ('Factor', ('flt',), 0.15),
    ]
    return Grammar(rules, 'Prog', terminals=T)


# ===================== Rust sampler (pcfg_engine_rs) =====================

from pcfg_engine_rs import PyPCFGSampler

_PROGRESS_INTERVAL = 250_000


def default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


class Sampler:
    """Parallel corpus sampler backed by PyPCFGSampler."""

    def __init__(
        self,
        grammar: Grammar | None = None,
        *,
        workers: int | None = None,
        max_depth: int = 350,
    ):
        self.grammar = grammar or code_grammar()
        self.max_depth = max_depth
        self.workers = workers if workers is not None else default_workers()
        self._rust = PyPCFGSampler(self.grammar, max_depth, self.workers)

    def sample_batch(
        self,
        n: int,
        min_len: int,
        max_len: int,
        seed: int = 0,
    ) -> list[list[str]]:
        return self._rust.sample_batch(n, min_len, max_len, seed)

    def sample_one(
        self,
        min_len: int,
        max_len: int,
        seed: int = 0,
    ) -> list[str] | None:
        batch = self.sample_batch(1, min_len, max_len, seed)
        return batch[0] if batch else None

    def sample_exact_length(
        self,
        length: int,
        *,
        max_tries: int = 100_000,
        seed: int = 0,
    ) -> list[str] | None:
        for i in range(max_tries):
            batch = self.sample_batch(1, length, length, seed + i)
            if batch and len(batch[0]) == length:
                return batch[0]
        return None

    def fill_pool(
        self,
        target_pool_tokens: int,
        *,
        min_len: int,
        max_len: int,
        seed: int = 0,
        progress_interval: int = _PROGRESS_INTERVAL,
        log_fn=print,
    ) -> tuple[list[list[str]], int]:
        """Sample token lists until total length >= target_pool_tokens."""
        pool: list[list[str]] = []
        pool_tokens = 0
        batch_id = 0

        def maybe_log():
            if progress_interval <= 0 or pool_tokens == 0:
                return
            if pool_tokens % progress_interval < max_len:
                log_fn(
                    f"  progress: {len(pool):,} strings, "
                    f"{pool_tokens:,}/{target_pool_tokens:,} tokens "
                    f"({100.0 * pool_tokens / target_pool_tokens:.1f}%)",
                    flush=True,
                )

        while pool_tokens < target_pool_tokens:
            remaining = target_pool_tokens - pool_tokens
            batch_size = max(64, min(2048, remaining // max_len + 8))
            batch = self.sample_batch(batch_size, min_len, max_len, seed + batch_id)
            batch_id += 1
            if not batch:
                break
            for tokens in batch:
                pool.append(tokens)
                pool_tokens += len(tokens)
                if pool_tokens >= target_pool_tokens:
                    break
            maybe_log()
        return pool, pool_tokens


def sample_one(
    grammar: Grammar | None = None,
    min_len: int = 64,
    max_len: int = 127,
    *,
    max_depth: int = 350,
    seed: int = 0,
    workers: int | None = None,
) -> list[str] | None:
    sampler = Sampler(grammar, workers=workers, max_depth=max_depth)
    return sampler.sample_one(min_len, max_len, seed=seed)
