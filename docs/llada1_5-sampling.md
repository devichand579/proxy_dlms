# LLaDA 1.5: full-block confidence collapse and B.4 mitigation

This note explains a failure mode seen with **LLaMA-backbone LLaDA 1.5** checkpoints when using **long single blocks** and **confidence-based** remasking, and how STrace handles it **without** extra CLI or Hydra flags for the usual `./models/llada1_5` layout.

## What it looks like

For some schedulers (e.g. pure diffusion, top-k with **confidence**, EB with **confidence**), the **generated tail** can be almost entirely token id **126081** (`<|endoftext|>`, i.e. **EOS** in `config.json`). Decoded text looks like an empty answer or a wall of special tokens after a short prefix.

This is **not** caused by:

- Batched left padding mis-handling (Dream has a separate wrapper story; LLaDA 1.5 uses normal pad + attention mask).
- `skip_special_tokens=True` “stripping away” content — the **integer sequence** really is mostly EOS.
- EOS **harness** early exit alone — with `--no-eos-harness` you still see the same id pattern if the model keeps committing EOS at masked positions.

## Why it happens (intuition)

Mask diffusion steps **predict a token at every position** and then **choose which masked positions to commit** each step (e.g. by **highest confidence** in the confidence remasking path). If the model assigns **very high probability to EOS** at many still-masked positions in parallel, the scheduler can **unmask EOS first** across the block. That fills the sequence with EOS even when other schedulers (e.g. **block diffusion** with smaller blocks, or **top-k margin**) look fine, because **block size**, **remasking metric**, and **parallel context** change which positions get committed when.

The LLaDA paper’s **Appendix B.4** discusses **suppressing EOS / EoT in logits and in the confidence used for unmasking** while positions are still `[MASK]`, so the sampler cannot lock in EOS everywhere too early. STrace’s schedulers already expose this as `logits_eos_inf` and `confidence_eos_eot_inf` in `schedulers.utils.mask_eos_eot_logits`; they default to off for compatibility with original LLaDA runs.

## What STrace does automatically

The helper **`load_model.infer_b44_eos_unmask_safety(model_dir)`** returns **true** when:

1. The resolved checkpoint directory’s **basename is exactly `llada1_5`** (e.g. `/path/to/models/llada1_5`), **or**
2. The checkpoint’s **`config.json`** contains `"strace_b44_eos_unmask": true` (for renamed folders or explicit opt-in).

When **true**, **`schedulers.generate.generate`** (and code paths that pass `model_path` into it, including **lm-eval** and **CST collect**) set **both** B.4 flags on for that run. **`./models/llada`**, **Dream**, and other directory names are **unchanged** (still `false`).

There is **no** separate switch in `config/config.yaml`, `eval.sh`, or `cst.yaml` — you only need the **model path** to resolve to `llada1_5` (or use the config key above).

## Opting out or overriding

- **`"strace_b44_eos_unmask": false`** in that checkpoint’s `config.json` forces B.4 **off** even under a directory named `llada1_5` (if you need to reproduce the old behavior for debugging).
- **`"strace_b44_eos_unmask": true`** forces B.4 **on** if weights live in a folder **not** named `llada1_5`.

## Related docs

- [Evaluation](evaluation.md) — model path and harness settings.
- [Scheduler smoke tests](scheduler-testing.md) — `test_schedulers.py`, `--show-original-tokens`, `--no-eos-harness`.

## Code references

- `load_model.py`: `infer_b44_eos_unmask_safety`
- `schedulers/generate.py`: applies B.4 when `model_path` is set and inference returns true
- `eval.py`: passes `model_path` into `generate` so eval matches smoke tests
