"""NPZ trace conversion and round-trip fidelity tests."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from traces.trace_checks import (
    check_trace_file,
    check_trace_samples,
    conversion_repair_blockers,
    count_fhs_corrector_extras,
    repair_sample_from_mask_diff,
    validate_npz_roundtrip,
)
from traces.trace_npz import (
    convert_jsonl_to_npz,
    iter_npz_traces,
    load_trace_samples,
    pack_jsonl_to_npz,
    pack_samples_to_npz,
)


def _sample_trace(*, gen_length: int = 8, n_steps: int = 4) -> dict:
    step_traces = []
    mask = [True] * gen_length
    for t in range(n_steps):
        preds = list(range(100, 100 + gen_length))
        conf = [0.5] * gen_length
        ent = [1.0] * gen_length
        sel_pos = None
        sel_tok = None
        if t == 1:
            mask = mask[:]
            mask[0] = False
            sel_pos = [[not m for m in mask]]
            sel_tok = [[preds[i] if not mask[i] else 0 for i in range(gen_length)]]
        step_traces.append(
            {
                "step": t,
                "mask_index": [mask],
                "predictions": [preds],
                "confidences": [conf],
                "entropies": [ent],
                "selected_positions": sel_pos,
                "selected_tokens": sel_tok,
                "intervention_applied": t == 2,
                "intervention_type": "drop" if t == 2 else None,
                "metadata": {"scheduler": "test"} if t == 0 else {},
            }
        )

    prompt = [11, 22, 33]
    final = prompt + list(range(200, 200 + gen_length))
    return {
        "gen_only": True,
        "prompt": [prompt],
        "final_sequence": [final],
        "prompt_len": len(prompt),
        "gen_length": gen_length,
        "step_traces": step_traces,
        "unmasking_order": {"0": 1},
        "start_time": 1.25,
        "end_time": 2.5,
        "metadata": {"task": "pcfg_suffix", "variant": "pcfg"},
    }


def test_npz_roundtrip_preserves_jsonl_fields() -> None:
    sample = _sample_trace()
    with tempfile.TemporaryDirectory(prefix="test_trace_npz_") as tmp:
        jsonl_path = Path(tmp) / "trace.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("# task=\"gsm8k\"\n")
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        npz_path = convert_jsonl_to_npz(jsonl_path)
        issues = validate_npz_roundtrip([sample], npz_path)
        assert issues == [], issues

        _, loaded = load_trace_samples(npz_path)
        assert len(loaded) == 1
        assert loaded[0]["prompt"] == sample["prompt"]
        assert loaded[0]["final_sequence"] == sample["final_sequence"]
        assert loaded[0]["metadata"] == sample["metadata"]
        assert loaded[0]["step_traces"][2]["intervention_applied"] is True
        assert loaded[0]["step_traces"][2]["intervention_type"] == "drop"

        check = check_trace_file(npz_path, max_samples=1)
        assert check.n_samples == 1
        assert check.transfer_status in ("explicit_partial", "explicit_all", "inferrable")

        with __import__("numpy").load(npz_path, allow_pickle=False) as data:
            assert int(data["n_samples"]) == 1
            assert "sidecar_bytes" in data
            assert "prompt_tokens" in data


def test_conversion_blocks_when_repair_disabled() -> None:
    sample = _sample_trace()
    # Remove hooks so repair would be needed; inferrable from mask diffs only if selected missing
    for step in sample["step_traces"]:
        step["selected_positions"] = None
        step["selected_tokens"] = None

    check = check_trace_samples([sample], max_samples=1)
    blockers = conversion_repair_blockers(check, [sample], repair_enabled=False)
    assert blockers
    assert any("disabled" in b for b in blockers)


def test_conversion_blocks_when_hooks_remain_broken() -> None:
    sample = _sample_trace()
    for step in sample["step_traces"]:
        step["mask_index"] = [[True] * sample["gen_length"]]
        step["selected_positions"] = None
        step["selected_tokens"] = None
    sample["unmasking_order"] = {}

    check = check_trace_samples([sample], max_samples=1)
    blockers = conversion_repair_blockers(check, [sample], repair_enabled=True)
    assert blockers
    assert any("hooks not ok" in b for b in blockers)


def test_streaming_jsonl_convert_matches_in_memory() -> None:
    sample = _sample_trace()
    with tempfile.TemporaryDirectory(prefix="test_trace_npz_") as tmp:
        jsonl_path = Path(tmp) / "trace.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("# task=\"gsm8k\"\n")
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        task_name, samples = load_trace_samples(jsonl_path)
        npz_memory = pack_samples_to_npz(
            samples,
            Path(tmp) / "memory.npz",
            task_name=task_name,
        )
        npz_stream = pack_jsonl_to_npz(jsonl_path, Path(tmp) / "stream.npz")

        assert validate_npz_roundtrip([sample], npz_memory) == []
        assert validate_npz_roundtrip(jsonl_path, npz_stream) == []


def test_check_trace_jsonl_streaming() -> None:
    sample = _sample_trace()
    with tempfile.TemporaryDirectory(prefix="test_trace_npz_") as tmp:
        jsonl_path = Path(tmp) / "trace.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        from traces.trace_checks import check_trace_jsonl_streaming

        check = check_trace_jsonl_streaming(jsonl_path)
        assert check.n_samples == 1
        assert check.hooks_ok


def test_convert_jsonl_to_npz_accepts_preloaded_samples() -> None:
    sample = _sample_trace()
    with tempfile.TemporaryDirectory(prefix="test_trace_npz_") as tmp:
        jsonl_path = Path(tmp) / "trace.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("# task=\"gsm8k\"\n")
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        task_name, samples = load_trace_samples(jsonl_path)
        npz_from_disk = convert_jsonl_to_npz(jsonl_path)
        npz_from_memory = convert_jsonl_to_npz(
            jsonl_path,
            Path(tmp) / "trace_memory.npz",
            samples=samples,
            task_name=task_name,
        )

        assert validate_npz_roundtrip(samples, npz_from_disk) == []
        assert validate_npz_roundtrip(samples, npz_from_memory) == []


def test_jsonl_pack_batch_sizes_match() -> None:
    import os

    sample = _sample_trace()
    with tempfile.TemporaryDirectory(prefix="test_trace_npz_") as tmp:
        jsonl_path = Path(tmp) / "trace.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("# task=\"gsm8k\"\n")
            for _ in range(5):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        old = os.environ.get("TRACE_PACK_BATCH_SAMPLES")
        try:
            os.environ["TRACE_PACK_BATCH_SAMPLES"] = "1"
            npz_one = pack_jsonl_to_npz(jsonl_path, Path(tmp) / "batch1.npz")
            os.environ["TRACE_PACK_BATCH_SAMPLES"] = "3"
            npz_many = pack_jsonl_to_npz(jsonl_path, Path(tmp) / "batch3.npz")
        finally:
            if old is None:
                os.environ.pop("TRACE_PACK_BATCH_SAMPLES", None)
            else:
                os.environ["TRACE_PACK_BATCH_SAMPLES"] = old

        assert validate_npz_roundtrip(jsonl_path, npz_one) == []
        assert validate_npz_roundtrip(jsonl_path, npz_many) == []


def test_iter_npz_traces_matches_load_trace_samples() -> None:
    sample = _sample_trace(gen_length=4, n_steps=2)
    with tempfile.TemporaryDirectory(prefix="test_trace_npz_") as tmp:
        jsonl_path = Path(tmp) / "trace.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")
        npz_path = convert_jsonl_to_npz(jsonl_path)
        from_iter = list(iter_npz_traces(npz_path))
        _, from_loader = load_trace_samples(npz_path)
        assert from_iter[0]["final_sequence"] == from_loader[0]["final_sequence"]


def _fhs_pc_sample(*, gen_length: int = 8, step_budget: int = 4) -> dict:
    """Minimal FHS predictor-corrector trace: Ln predictors + 1 corrector per chunk."""
    assert gen_length % step_budget == 0
    ln = gen_length // step_budget
    stride = ln + 1
    mask = [True] * gen_length
    step_traces = []
    pos = 0
    for chunk in range(step_budget):
        for _pred in range(ln):
            preds = list(range(100, 100 + gen_length))
            sel = [False] * gen_length
            if pos < gen_length:
                mask = mask[:]
                mask[pos] = False
                sel[pos] = True
                pos += 1
            step_traces.append(
                {
                    "step": len(step_traces),
                    "mask_index": [mask[:]],
                    "predictions": [preds],
                    "confidences": [[0.5] * gen_length],
                    "entropies": [[1.0] * gen_length],
                    "selected_positions": [sel],
                    "selected_tokens": [[preds[i] if sel[i] else 0 for i in range(gen_length)]],
                }
            )
        # Corrector: same mask, no selection, refreshed logits.
        step_traces.append(
            {
                "step": len(step_traces),
                "mask_index": [mask[:]],
                "predictions": [list(range(200, 200 + gen_length))],
                "confidences": [[0.9] * gen_length],
                "entropies": [[0.1] * gen_length],
                "selected_positions": None,
                "selected_tokens": None,
            }
        )
    assert len(step_traces) == gen_length + step_budget
    assert stride == ln + 1
    return {
        "gen_only": True,
        "prompt": [[1, 2, 3]],
        "final_sequence": [[1, 2, 3] + list(range(300, 300 + gen_length))],
        "prompt_len": 3,
        "gen_length": gen_length,
        "step_traces": step_traces,
        "unmasking_order": {},
        "metadata": {},
    }


def test_fhs_corrector_count_and_merge_clears_phantoms() -> None:
    """s256-style PC: merge must not leave false phantom correctors when len > budget."""
    # PC mode requires step_budget > 128; gen > budget reproduces the s256 false-positive.
    gen_length, step_budget = 512, 256  # Ln=2, stride=3, raw len=768
    sample = _fhs_pc_sample(gen_length=gen_length, step_budget=step_budget)
    assert len(sample["step_traces"]) == gen_length + step_budget
    assert count_fhs_corrector_extras(
        sample["step_traces"], gen_length=gen_length, step_budget=step_budget
    ) == step_budget

    repaired, msgs = repair_sample_from_mask_diff(
        sample, is_fdm=False, is_fhs=True, step_budget=step_budget
    )
    assert any("merged FHS predictor+corrector" in m for m in msgs)
    rsteps = repaired["step_traces"]
    # Merge to gen_length predictors, then drop trailing step.
    assert len(rsteps) == gen_length - 1
    assert len(rsteps) > step_budget  # old detector used len > budget → false phantoms
    assert (
        count_fhs_corrector_extras(
            rsteps, gen_length=gen_length, step_budget=step_budget
        )
        == 0
    )
    check = check_trace_samples(
        [repaired],
        path="toy_fhs_s256.jsonl",
        max_samples=None,
        check_last=True,
        is_fhs=True,
        step_budget=step_budget,
    )
    assert not check.any_phantom_steps
    assert conversion_repair_blockers(check, [repaired], repair_enabled=True) == []
