"""CST block-structured training: gradient equivalence, residency, checkpoints.

The refactor replaced a single per-chunk ``backward()`` with one backward per
block of consecutive anchors, and an LRU GPU cache with an explicit residency
manager. There was no CST test coverage before, so these are the only guard
against a silent numerical regression.

Run: ``python -m pytest tests/test_cst_block.py -v`` (or ``python tests/test_cst_block.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cst.model.cst_bank import CSTBank  # noqa: E402
from cst.model.residency import CSTResidency, budget_report, steps_for_anchors  # noqa: E402
from cst.train.adam_bf16 import AdamBF16  # noqa: E402
from cst.train.hidden_chunk import iter_anchor_samples  # noqa: E402
from cst.train.loss import windowed_cst_loss_block, windowed_cst_loss_full  # noqa: E402
from cst.train.trainer import (  # noqa: E402
    LambdaRamp,
    load_cst_bank_weights,
    load_cst_checkpoint,
    save_cst_checkpoint,
)

D, K, W, MAX_STEPS = 16, 32, 4, 12
LAM_S, C_TANH, LAM_RAMP, LAM_PRE = 1e-3, 1.0, 0.5, 3e-6


def _bank(seed: int = 0) -> CSTBank:
    torch.manual_seed(seed)
    return CSTBank(max_steps=MAX_STEPS, d=D, k=K, w=W, initial_threshold=0.03)


def _H(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(MAX_STEPS + 1, 2, 5, D, generator=g)


def _grads(bank: CSTBank) -> dict[str, torch.Tensor]:
    return {n: p.grad.detach().clone() for n, p in bank.named_parameters() if p.grad is not None}


def test_block_backward_matches_per_chunk_backward():
    """One backward per block must accumulate the same gradients as one per chunk."""
    H = _H()
    samples = list(iter_anchor_samples(H, W))
    anchors = [a for a, _ in samples]

    ref = _bank()
    losses = [
        windowed_cst_loss_full(
            ref, a, hw, H,
            lambda_sparsity=LAM_S, c_tanh=C_TANH, lambda_ramp=LAM_RAMP, lambda_pre_dead=LAM_PRE,
        )[0]
        for a, hw in samples
    ]
    torch.stack(losses).sum().backward()
    want = _grads(ref)

    for block_size in (1, 3, 5, len(anchors)):
        got_bank = _bank()
        for i in range(0, len(anchors), block_size):
            blk = anchors[i : i + block_size]
            hws = [hw for a, hw in samples if a in blk]
            loss, _ = windowed_cst_loss_block(
                got_bank, blk, hws, H,
                lambda_sparsity=LAM_S, c_tanh=C_TANH, lambda_ramp=LAM_RAMP, lambda_pre_dead=LAM_PRE,
            )
            loss.backward()
        got = _grads(got_bank)
        assert set(got) == set(want)
        for name in want:
            torch.testing.assert_close(
                got[name], want[name], rtol=1e-4, atol=1e-6,
                msg=lambda m, n=name, b=block_size: f"block_size={b} param={n}\n{m}",
            )


def test_checkpointed_sparsity_penalty_matches_uncheckpointed():
    """The activation-memory fix must not change the value or the gradient."""
    bank = _bank()
    cst = bank.get(1)
    A = torch.randn(2, 5, K).abs().requires_grad_(True)

    plain = cst.decoder_sparsity_penalty(A, C_TANH, checkpoint=False)
    (g_plain,) = torch.autograd.grad(plain, A, retain_graph=False)
    ckpt = cst.decoder_sparsity_penalty(A, C_TANH, checkpoint=True)
    (g_ckpt,) = torch.autograd.grad(ckpt, A)

    torch.testing.assert_close(plain, ckpt, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(g_plain, g_ckpt, rtol=1e-6, atol=1e-7)


def test_steps_for_anchors_window_rule():
    assert steps_for_anchors([5], 4) == [2, 3, 4, 5]
    # G consecutive anchors touch G + w - 1 distinct steps, not G * w.
    assert steps_for_anchors([5, 6, 7], 4) == [2, 3, 4, 5, 6, 7]
    assert len(steps_for_anchors(list(range(4, 4 + 24)), 4)) == 24 + 3
    # Steps below 1 are clipped.
    assert steps_for_anchors([2], 4) == [1, 2]


def test_residency_preserves_parameter_identity_and_moves_state():
    """module.to(device) would rebuild Parameters and orphan optimizer state."""
    bank = _bank()
    opt = AdamBF16(bank.parameters(), lr=1e-3)
    ids_before = [id(p) for p in bank.get(3).parameters()]

    res = CSTResidency(bank, torch.device("cpu"), optimizer=opt, pinned_bytes=0)
    res.ensure([3, 4])
    assert [id(p) for p in bank.get(3).parameters()] == ids_before
    assert res.resident_steps == [3, 4]
    res.release(3)
    assert res.resident_steps == [4]

    # A param that has taken a step keeps its moments across a migrate.
    p = next(bank.get(4).parameters())
    p.grad = torch.ones_like(p)
    opt.step()
    assert "exp_avg" in opt.state[p]
    before = opt.state[p]["exp_avg"].clone()
    res.release(4)
    torch.testing.assert_close(opt.state[p]["exp_avg"], before)


def test_pinned_prefix_is_sized_from_budget():
    bank = _bank()
    per_step = sum(x.numel() * x.element_size() for x in bank.get(1).parameters())
    res = CSTResidency(bank, torch.device("cpu"), pinned_bytes=per_step * 5)
    assert res.pinned == {1, 2, 3, 4, 5}
    # Pinned steps drop their grad but stay resident.
    res.ensure([1])
    res.release(1)
    assert 1 in res.resident_steps


def test_adam_bf16_keeps_moments_bf16_and_descends():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(64))
    target = torch.randn(64)
    opt = AdamBF16([p], lr=0.1)
    first = float(((p - target) ** 2).sum().detach())
    for _ in range(60):
        opt.zero_grad()
        ((p - target) ** 2).sum().backward()
        opt.step()
    assert float(((p - target) ** 2).sum().detach()) < first * 0.1
    assert opt.state[p]["exp_avg"].dtype == torch.bfloat16
    assert opt.state[p]["exp_avg_sq"].dtype == torch.bfloat16


def test_adam_bf16_second_moment_does_not_stagnate():
    """Plain bf16 truncation rounds the (1-b2)*g^2 increment away; SR must not."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(4096))
    opt = AdamBF16([p], lr=0.0, betas=(0.9, 0.999), stochastic_rounding=True)
    for _ in range(200):
        p.grad = torch.full_like(p, 0.05)
        opt.step()
    v = opt.state[p]["exp_avg_sq"].float().mean()
    expected = 0.05**2 * (1 - 0.999**200)
    assert v == pytest.approx(expected, rel=0.15), f"v={v} expected~{expected}"


def test_sharded_checkpoint_round_trip(tmp_path):
    bank = _bank(seed=0)
    opt = AdamBF16(bank.parameters(), lr=1e-3)
    ramp = LambdaRamp(100)
    for _ in range(7):
        ramp.step()
    for p in bank.parameters():
        p.grad = torch.randn_like(p)
    opt.step()

    out = tmp_path / "cst_x_train_latest"
    save_cst_checkpoint(
        out, cst_bank=bank, optimizer=opt, ramp=ramp,
        meta={"phase": 3}, save_optimizer_state=True,
    )
    assert (out / "meta.json").is_file()
    assert len(list(out.glob("cst_*.pt"))) == MAX_STEPS

    bank2 = _bank(seed=99)
    opt2 = AdamBF16(bank2.parameters(), lr=1e-3)
    ramp2 = LambdaRamp(1)
    meta = load_cst_checkpoint(out, bank2, opt2, ramp2)

    assert meta["phase"] == 3
    assert ramp2.current == 7 and ramp2.total_steps == 100
    for (n1, p1), (n2, p2) in zip(bank.named_parameters(), bank2.named_parameters()):
        assert n1 == n2
        # Shards are stored bf16, so compare at bf16 resolution.
        torch.testing.assert_close(p1.to(torch.bfloat16).float(), p2, rtol=1e-2, atol=1e-2)
    p_a = next(bank.get(1).parameters())
    p_b = next(bank2.get(1).parameters())
    torch.testing.assert_close(opt.state[p_a]["exp_avg"], opt2.state[p_b]["exp_avg"])


def test_params_only_checkpoint_skips_optimizer(tmp_path):
    bank = _bank()
    opt = AdamBF16(bank.parameters(), lr=1e-3)
    for p in bank.parameters():
        p.grad = torch.randn_like(p)
    opt.step()

    out = tmp_path / "cst_y_train_latest"
    save_cst_checkpoint(out, cst_bank=bank, optimizer=opt, ramp=LambdaRamp(10), meta={})
    blob = torch.load(out / "cst_00001.pt", map_location="cpu", weights_only=False)
    assert "optimizer" not in blob

    bank2 = _bank(seed=5)
    opt2 = AdamBF16(bank2.parameters(), lr=1e-3)
    load_cst_checkpoint(out, bank2, opt2, LambdaRamp(10))
    assert not opt2.state  # moments restart, weights restored


def test_legacy_single_pt_checkpoint_still_loads(tmp_path):
    from cst.storage_dtype import map_float_tensors

    bank = _bank()
    legacy = tmp_path / "cst_old_train_latest.pt"
    torch.save(
        {
            "cst": map_float_tensors(bank.state_dict_for_save(), to_storage=True),
            "ramp": {"current": 4, "total_steps": 50},
            "meta": {"phase": 1},
            "storage_float_dtype": "bfloat16",
        },
        legacy,
    )
    bank2 = _bank(seed=42)
    meta = load_cst_bank_weights(legacy, bank2)
    assert meta["phase"] == 1
    for (_, p1), (_, p2) in zip(bank.named_parameters(), bank2.named_parameters()):
        torch.testing.assert_close(p1.to(torch.bfloat16).float(), p2, rtol=1e-2, atol=1e-2)


def test_incomplete_checkpoint_dir_is_not_treated_as_sharded(tmp_path):
    """meta.json is written last; a half-written dir must not silently load."""
    from cst.train.trainer import _is_sharded_ckpt

    d = tmp_path / "cst_partial_train_latest"
    d.mkdir()
    (d / "cst_00001.pt").write_bytes(b"")
    assert not _is_sharded_ckpt(d)


def test_train_on_chunk_files_end_to_end(tmp_path):
    """Real trainer path: streamed residency, per-step updates, loss decreases."""
    from cst.storage_dtype import map_float_tensors
    from cst.train.trainer import train_on_chunk_files

    H = _H(seed=7)
    chunk = tmp_path / "chunk_000.pt"
    torch.save(
        {"hidden_states": [map_float_tensors(H[i], to_storage=True) for i in range(H.shape[0])],
         "meta": {"prompt_len": 2}},
        chunk,
    )

    bank = _bank()
    opt = AdamBF16(bank.parameters(), lr=1e-2)
    per_step = sum(x.numel() * x.element_size() for x in bank.get(1).parameters())
    res = CSTResidency(bank, torch.device("cpu"), optimizer=opt, pinned_bytes=per_step * 2)

    losses = []
    for _ in range(4):
        stats = train_on_chunk_files(
            [chunk], bank, opt, LambdaRamp(10),
            w=W, lambda_sparsity=LAM_S, c_tanh=C_TANH, lambda_pre_dead=LAM_PRE,
            device=torch.device("cpu"), anchor_block=3, residency=res,
        )
        losses.append(stats["train_loss"])

    assert stats["n_chunks"] == 1.0
    assert stats["train_batches"] == float(len(list(iter_anchor_samples(H, W))))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    # Every step must have been updated exactly once per pass.
    assert all(opt.state[p]["step"] == 4 for p in bank.get(1).parameters())
    # Streaming must leave only the pinned prefix behind.
    assert res.resident_steps == sorted(res.pinned)
    # No stale gradients.
    assert all(p.grad is None for p in bank.parameters())


def test_block_size_does_not_change_the_trained_result(tmp_path):
    """anchor_block is a memory knob, not a hyperparameter."""
    from cst.storage_dtype import map_float_tensors
    from cst.train.trainer import train_on_chunk_files

    H = _H(seed=11)
    chunk = tmp_path / "c.pt"
    torch.save(
        {"hidden_states": [map_float_tensors(H[i], to_storage=True) for i in range(H.shape[0])],
         "meta": {"prompt_len": 2}},
        chunk,
    )

    def run(block):
        bank = _bank()
        opt = AdamBF16(bank.parameters(), lr=1e-3, stochastic_rounding=False)
        train_on_chunk_files(
            [chunk], bank, opt, LambdaRamp(10),
            w=W, lambda_sparsity=LAM_S, c_tanh=C_TANH, lambda_pre_dead=LAM_PRE,
            device=torch.device("cpu"), anchor_block=block, residency=None,
        )
        return {n: p.detach().clone() for n, p in bank.named_parameters()}

    a, b = run(2), run(9)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=1e-4, atol=1e-6, msg=lambda m, n=name: f"{n}\n{m}")


def test_shipped_config_fits_the_target_machine():
    """config/cst.yaml must fit 192 GB GPU / 240 GB CPU at worst-case seq len."""
    from omegaconf import OmegaConf

    c = OmegaConf.load(Path(__file__).resolve().parent.parent / "config" / "cst.yaml")
    assert int(c.train.steps) <= int(c.train.gen_length), "generate.py asserts steps <= gen_length"
    seq = int(c.train.gen_length) + int(c.train.max_prompt_length)
    b = budget_report(
        max_steps=int(c.cst.max_steps), d=int(c.cst.d_model), k=int(c.cst.k), w=int(c.cst.w),
        anchor_block=int(c.train.anchor_block), pinned_gb=float(c.train.gpu_pinned_gb),
        batch=int(c.train.inference_batch_size), seq_len=seq,
    )
    assert b["gpu_gb"] <= 177.0, f"GPU {b['gpu_gb']:.0f} GB exceeds usable 177 GB"
    assert b["host_gb"] <= 240.0, f"host {b['host_gb']:.0f} GB exceeds 240 GB"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
