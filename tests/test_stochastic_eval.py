from types import SimpleNamespace

import pytest
import torch

from eval import (
    build_length_aware_chunks,
    derive_sample_seed,
    extract_gsm8k_answer,
    majority_vote_gsm8k,
    seeded_subset_indices,
    shard_subset_indices,
)
import schedulers.generate as generate_module
from schedulers.utils import add_gumbel_noise


def test_extract_gsm8k_answer_normalizes_commas():
    assert extract_gsm8k_answer("work\n#### 1,234.50") == "1234.50"
    assert extract_gsm8k_answer("no strict answer") is None


def test_majority_vote_and_earliest_tie_break():
    responses = ["#### 10", "#### 20", "#### 20"]
    assert majority_vote_gsm8k(responses) == ("20", responses[1])

    tied = ["#### 7", "#### 9", "#### 9", "#### 7"]
    assert majority_vote_gsm8k(tied) == ("7", tied[0])


def test_sample_seed_is_stable_and_tuple_specific():
    seed = derive_sample_seed(1234, "gsm8k", 5, "pure_diffusion", 2)
    assert seed == derive_sample_seed(1234, "gsm8k", 5, "pure_diffusion", 2)
    assert seed != derive_sample_seed(1234, "gsm8k", 5, "pure_diffusion", 3)
    assert seed != derive_sample_seed(1234, "gsm8k", 6, "pure_diffusion", 2)


def test_seeded_subset_is_reproducible_and_unique():
    first = seeded_subset_indices(1319, 100, 1234)
    assert first == seeded_subset_indices(1319, 100, 1234)
    assert first != seeded_subset_indices(1319, 100, 1235)
    assert len(first) == len(set(first)) == 100
    assert first == sorted(first)
    shard0 = shard_subset_indices(first, 0, 2)
    shard1 = shard_subset_indices(first, 1, 2)
    assert set(shard0).isdisjoint(shard1)
    assert sorted(shard0 + shard1) == first


def test_batching_modes_avoid_fine_bucket_fragmentation():
    lengths = [40, 60, 130, 200, 300, 500]
    current = build_length_aware_chunks(
        lengths, batch_size=6, gen_length=64, mode="current"
    )
    none = build_length_aware_chunks(
        lengths, batch_size=6, gen_length=64, mode="none"
    )
    coarse = build_length_aware_chunks(
        lengths, batch_size=6, gen_length=64, mode="coarse"
    )
    token = build_length_aware_chunks(
        lengths,
        batch_size=6,
        gen_length=64,
        mode="token",
        max_batch_tokens=900,
    )
    assert len(current) > len(coarse) > len(none)
    assert len(none) == 1
    assert sorted(i for chunk in token for i in chunk) == list(range(len(lengths)))
    assert all(
        len(chunk) * (max(lengths[i] for i in chunk) + 64) <= 900
        for chunk in token
    )


def test_fp32_gumbel_is_seeded_and_zero_temperature_is_exact():
    logits = torch.tensor([[[0.1, 0.2, 0.3], [0.5, -0.2, 0.0]]])
    assert add_gumbel_noise(logits, 0.0) is logits
    first = add_gumbel_noise(
        logits,
        0.3,
        generators=[torch.Generator().manual_seed(9)],
        mode="fp32",
    )
    second = add_gumbel_noise(
        logits,
        0.3,
        generators=[torch.Generator().manual_seed(9)],
        mode="fp32",
    )
    assert first.dtype == torch.float32
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "mode,target",
    [
        ("pure_diffusion", "generate_llada"),
        ("block_diffusion", "generate_llada"),
        ("top_k", "generate_topk"),
        ("fhs", "generate_fhs"),
        ("eb_sampler", "generate_EB"),
        ("dus", "generate_dus"),
        ("fdm", "generate_fdm"),
    ],
)
def test_dispatcher_passes_temperature_without_changing_logits(mode, target, monkeypatch):
    seen = {}

    def fake_scheduler(model, input_ids, attention_mask, **kwargs):
        seen.update(kwargs)
        if mode == "fhs":
            return input_ids, 1
        return input_ids, 1, 1

    monkeypatch.setattr(generate_module, target, fake_scheduler)
    model = SimpleNamespace()
    input_ids = torch.ones((1, 2), dtype=torch.long)
    generators = [torch.Generator().manual_seed(4)]
    block_length = None if mode == "fhs" else (1 if mode == "block_diffusion" else 2)

    generate_module.generate(
        model,
        input_ids,
        torch.ones_like(input_ids),
        steps=2,
        gen_length=2,
        block_length=block_length,
        mode=mode,
        strategy="confidence",
        temperature=0.3,
        generators=generators,
    )

    assert seen["temperature"] == pytest.approx(0.3)
    assert seen["generators"] is generators
    if mode != "fhs":
        assert seen["gumbel_mode"] == "fp32"
