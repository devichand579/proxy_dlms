import os
import time
import traceback
import argparse
from datetime import datetime
from typing import List, Optional

import torch

from load_model import (
    LLADA_DEFAULT_EOS_ID,
    LLADA_DEFAULT_EOT_ID,
    LLADA_DEFAULT_MASK_ID,
    infer_diffusion_token_ids,
    load_model_and_tokenizer,
)
from schedulers.llada_schedulers import generate_llada
from schedulers.top_k import generate_topk
from schedulers.fhs import generate_fhs
from schedulers.eb_sampler import generate_EB
from schedulers.dus import generate_dus
from schedulers.fdm import generate_fdm
from schedulers.generate import generate as generate_batch

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def _decode_generated_tails_at_eos(
    tokenizer,
    generated: torch.Tensor,
    prompt_len: int,
    eos_id: int,
) -> List[str]:
    """
    For each row, take ``generated[b, prompt_len:]`` and truncate before the first ``eos_id``,
    matching ``eval.py`` (see ``LLaDAEvalHarness.generate_until`` chunk decoding) so
    ``batch_decode`` is not swamped with mask or post-EOS noise.
    """
    gen_tails = []
    for b in range(generated.shape[0]):
        tail = generated[b, prompt_len:].contiguous()
        eos_positions = (tail == eos_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            tail = tail[: int(eos_positions[0].item())]
        gen_tails.append(tail)
    return tokenizer.batch_decode(gen_tails, skip_special_tokens=True)


def test_pure_diffusion(
    model, input_ids, attention_mask, tokenizer, trace_name=None,
    intervention_mode=None, intervention_steps=None, intervention_k=1,
    mask_id: int = LLADA_DEFAULT_MASK_ID,
    eos_id: int = LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
):
    """Test pure diffusion scheduler"""
    print("Generating with pure diffusion (block_length=256)...")
    start_time = time.time()
    out_pure_diffusion, steps_pure, blocks_pure = generate_llada(
        model, input_ids, attention_mask, steps=256, gen_length=256, 
        block_length=256, temperature=0., remasking='confidence',
        trace_name=trace_name,intervention_mode=intervention_mode, intervention_steps=intervention_steps, intervention_k=intervention_k,
        mask_id=mask_id, eos_id=eos_id, eot_id=eot_id,
    )
    pure_diffusion_time = time.time() - start_time
    print(f"Pure diffusion generation time: {pure_diffusion_time:.2f} seconds")
    print(f"Steps taken: {steps_pure}")
    print(f"Blocks generated: {blocks_pure}")
    print(f"output Shape: {out_pure_diffusion.shape}")
    print()
    output = _decode_generated_tails_at_eos(
        tokenizer, out_pure_diffusion, int(input_ids.shape[1]), eos_id
    )
    return output, "Pure diffusion"


def test_block_diffusion(
    model, input_ids, attention_mask, tokenizer, trace_name=None,
    intervention_mode=None, intervention_steps=None, intervention_k=1,
    mask_id: int = LLADA_DEFAULT_MASK_ID,
    eos_id: int = LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
):
    """Test block diffusion scheduler"""
    print("Generating with block diffusion (block_length=32)...")
    start_time = time.time()
    out_block_diffusion, steps_block, blocks_block = generate_llada(
        model, input_ids, attention_mask, steps=256, gen_length=256, 
        block_length=32, temperature=0., remasking='confidence',
        trace_name=trace_name, intervention_mode=intervention_mode, intervention_steps=intervention_steps, intervention_k=intervention_k,
        mask_id=mask_id, eos_id=eos_id, eot_id=eot_id,
    )
    block_diffusion_time = time.time() - start_time
    print(f"Block diffusion generation time: {block_diffusion_time:.2f} seconds")
    print(f"Steps taken: {steps_block}")
    print(f"Blocks generated: {blocks_block}")
    print(f"output Shape: {out_block_diffusion.shape}")
    print()
    output = _decode_generated_tails_at_eos(
        tokenizer, out_block_diffusion, int(input_ids.shape[1]), eos_id
    )
    return output, "Block diffusion"


def test_top_k_confidence(
    model, input_ids, attention_mask, tokenizer, trace_name=None,
    intervention_mode=None, intervention_steps=None, intervention_k=1,
    mask_id: int = LLADA_DEFAULT_MASK_ID,
    eos_id: int = LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
):
    """Test top-k scheduler with confidence metric"""
    print("Generating with top-k (confidence metric, block_length=256)...")
    start_time = time.time()
    # Note: trace_name not yet supported by generate_topk
    out_top_k_confidence, steps_top_k_confidence, blocks_top_k_confidence = generate_topk(
        model, input_ids, attention_mask, 
        steps=256, gen_length=256, 
        block_length=256, temperature=0., top_k=1, metric='confidence',
        trace_name=trace_name, intervention_mode=intervention_mode, intervention_steps=intervention_steps, intervention_k=intervention_k,
        mask_id=mask_id, eos_id=eos_id, eot_id=eot_id,
    )
    top_k_time = time.time() - start_time
    print(f"Top-k generation time: {top_k_time:.2f} seconds")
    print(f"Steps taken: {steps_top_k_confidence}")
    print(f"Blocks generated: {blocks_top_k_confidence}")
    print(f"output Shape: {out_top_k_confidence.shape}")
    print()
    output = _decode_generated_tails_at_eos(
        tokenizer, out_top_k_confidence, int(input_ids.shape[1]), eos_id
    )
    return output, "Top-k (confidence)"


def test_top_k_margin(
    model, input_ids, attention_mask, tokenizer, trace_name=None,
    intervention_mode=None, intervention_steps=None, intervention_k=1,
    mask_id: int = LLADA_DEFAULT_MASK_ID,
    eos_id: int = LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
):
    """Test top-k scheduler with margin metric"""
    print("Generating with top-k (margin metric, block_length=256)...")
    start_time = time.time()
    out_top_k_margin, steps_top_k_margin, blocks_top_k_margin = generate_topk(
        model, input_ids, attention_mask, steps=256, gen_length=256, 
        block_length=256, temperature=0., top_k=1, metric='top_margin',
        trace_name=trace_name, intervention_mode=intervention_mode, intervention_steps=intervention_steps, intervention_k=intervention_k,
        mask_id=mask_id, eos_id=eos_id, eot_id=eot_id,
    )
    top_k_time = time.time() - start_time
    print(f"Top-k generation time: {top_k_time:.2f} seconds")
    print(f"Steps taken: {steps_top_k_margin}")
    print(f"Blocks generated: {blocks_top_k_margin}")
    print(f"output Shape: {out_top_k_margin.shape}")
    print()
    output = _decode_generated_tails_at_eos(
        tokenizer, out_top_k_margin, int(input_ids.shape[1]), eos_id
    )
    return output, "Top-k (margin)"


def test_fhs(
    model, input_ids, attention_mask, tokenizer, trace_name=None,
    intervention_mode=None, intervention_steps=None, intervention_k=1,
    mask_id: int = LLADA_DEFAULT_MASK_ID,
    eos_id: int = LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
):
    """Test FHS scheduler"""
    print("Generating with FHS (steps=256)...")
    start_time = time.time()
    out_fhs, steps_fhs = generate_fhs(
        model, input_ids, attention_mask=attention_mask, steps=256, 
        gen_length=256, temperature=1.0, pass_time_to_model=True,
        trace_name=trace_name, intervention_mode=intervention_mode, intervention_steps=intervention_steps, intervention_k=intervention_k,
        mask_id=mask_id, eos_id=eos_id, eot_id=eot_id,
    )
    fhs_time = time.time() - start_time
    print(f"FHS generation time: {fhs_time:.2f} seconds")
    print(f"Steps taken: {steps_fhs}")
    print(f"output Shape: {out_fhs.shape}")
    print()
    output = _decode_generated_tails_at_eos(
        tokenizer, out_fhs, int(input_ids.shape[1]), eos_id
    )
    return output, "FHS"


def test_eb_sampler(
    model, input_ids, attention_mask, tokenizer, trace_name=None,
    intervention_mode=None, intervention_steps=None, intervention_k=1,
    mask_id: int = LLADA_DEFAULT_MASK_ID,
    eos_id: int = LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
):
    """Test EB sampler scheduler"""
    print("Generating with EB_sampler (confidence metric, block_length=256)...")
    start_time = time.time()
    out_eb_sampler, steps_eb_sampler, blocks_eb_sampler = generate_EB(
        model, input_ids, attention_mask, steps=256, gen_length=256, 
        block_length=256, gamma=0.1, E='confidence', temperature=0.,
        trace_name=trace_name, intervention_mode=intervention_mode, intervention_steps=intervention_steps, intervention_k=intervention_k,
        mask_id=mask_id, eos_id=eos_id, eot_id=eot_id,
    )
    eb_sampler_time = time.time() - start_time
    print(f"EB_sampler generation time: {eb_sampler_time:.2f} seconds")
    print(f"Steps taken: {steps_eb_sampler}")
    print(f"Blocks generated: {blocks_eb_sampler}")
    print(f"output Shape: {out_eb_sampler.shape}")
    print()
    output = _decode_generated_tails_at_eos(
        tokenizer, out_eb_sampler, int(input_ids.shape[1]), eos_id
    )
    return output, "EB_sampler (confidence)"


def test_dus(
    model, input_ids, attention_mask, tokenizer, trace_name=None,
    intervention_mode=None, intervention_steps=None, intervention_k=1,
    mask_id: int = LLADA_DEFAULT_MASK_ID,
    eos_id: int = LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
):
    """Test DUS scheduler"""
    print("Generating with DUS (dilated schedule, block_length=16)...")
    start_time = time.time()
    out_dus, steps_dus, blocks_dus = generate_dus(
        model, input_ids, attention_mask, gen_length=256,
        block_length=16, temperature=0., remasking='confidence', scheduler='dilated',
        trace_name=trace_name, intervention_mode=intervention_mode, intervention_steps=intervention_steps, intervention_k=intervention_k,
        mask_id=mask_id, eos_id=eos_id, eot_id=eot_id,
    )
    dus_time = time.time() - start_time
    print(f"DUS generation time: {dus_time:.2f} seconds")
    print(f"Steps taken: {steps_dus}")
    print(f"Blocks generated: {blocks_dus}")
    print(f"output Shape: {out_dus.shape}")
    print()
    output = _decode_generated_tails_at_eos(
        tokenizer, out_dus, int(input_ids.shape[1]), eos_id
    )
    return output, "DUS (dilated)"


def test_fdm(
    model, input_ids, attention_mask, tokenizer, trace_name=None,
    intervention_mode=None, intervention_steps=None, intervention_k=1,
    mask_id: int = LLADA_DEFAULT_MASK_ID,
    eos_id: int = LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
):
    """Test FDM scheduler"""
    print("Generating with FDM (steps=256, block_length=64)...")
    start_time = time.time()
    out_fdm, steps_fdm, blocks_fdm = generate_fdm(
        model, input_ids, attention_mask, steps=256, gen_length=256, block_length=64,
        trace_name=trace_name, intervention_mode=intervention_mode, intervention_steps=intervention_steps, intervention_k=intervention_k,
        mask_id=mask_id, eos_id=eos_id, eot_id=eot_id,
    )
    fdm_time = time.time() - start_time
    print(f"FDM generation time: {fdm_time:.2f} seconds")
    print(f"Steps taken: {steps_fdm}, blocks: {blocks_fdm}")
    print(f"output Shape: {out_fdm.shape}")
    print()
    output = _decode_generated_tails_at_eos(
        tokenizer, out_fdm, int(input_ids.shape[1]), eos_id
    )
    return output, "FDM"

def _model_label(model_dir: str) -> str:
    """Short name for logs/traces (last path component)."""
    base = os.path.basename(os.path.normpath(model_dir))
    return base or "model"


# (mode, strategy, block_length) for batch tests via schedulers.generate.generate()
BATCH_TEST_MODES = {
    "pure_diffusion": ("pure_diffusion", "confidence", 256),
    "block_diffusion": ("block_diffusion", "confidence", 32),
    "top_k_confidence": ("top_k", "confidence", 256),
    "top_k_margin": ("top_k", "top_margin", 256),
    "fhs": ("fhs", None, None),
    "eb_sampler": ("eb_sampler", "confidence", 256),
    "dus": ("dus", "confidence", 16),
    "fdm": ("fdm", None, 64),
}


def run_batch_tests(
    model,
    tokenizer,
    device,
    batch_size=2,
    scheduler_name=None,
    trace_name_base=None,
    upload_to_wandb=False,
    model_path=None,
):
    """Run batched generation via schedulers.generate.generate() and assert shapes/outputs.
    When ``model_path`` is set, token ids are read from that checkpoint (``config.json``).
    When upload_to_wandb is True, traces are uploaded to wandb."""
    prompts = [
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "A store sells 5 apples for $2. How much do 15 apples cost?",
        "What is 17 times 23?",
        "If 3x + 5 = 20, what is x?",
    ]
    prompts = (prompts * ((batch_size // len(prompts)) + 1))[:batch_size]
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    assert input_ids.shape[0] == batch_size, f"Expected batch {batch_size}, got {input_ids.shape[0]}"

    if model_path is not None:
        _, batch_eos_id, _ = infer_diffusion_token_ids(model_path)
    else:
        batch_eos_id = LLADA_DEFAULT_EOS_ID
    prompt_len = int(input_ids.shape[1])

    modes_to_run = (
        {scheduler_name: BATCH_TEST_MODES[scheduler_name]}
        if scheduler_name
        else BATCH_TEST_MODES
    )
    failures = []
    for name, (mode, strategy, block_length) in modes_to_run.items():
        try:
            steps = 256
            gen_length = 256
            if mode == "fdm":
                steps = 128
            bl = None if mode == "fhs" else (block_length or gen_length)
            trace_name = f"{trace_name_base}_{name}" if trace_name_base else None
            kwargs = dict(
                steps=steps,
                gen_length=gen_length,
                block_length=bl,
                mode=mode,
                trace_name=trace_name,
                upload_to_wandb=upload_to_wandb,
            )
            if strategy is not None:
                kwargs["strategy"] = strategy
            if model_path is not None:
                kwargs["model_path"] = model_path

            generated, steps_taken, blocks, gen_time = generate_batch(
                model, input_ids, attention_mask, **kwargs
            )
            assert generated.shape[0] == batch_size, (
                f"{name}: expected generated.shape[0] == {batch_size}, got {generated.shape[0]}"
            )
            assert generated.shape[1] == input_ids.shape[1] + gen_length, (
                f"{name}: expected seq_len == prompt_len + gen_length, "
                f"got {generated.shape[1]} vs {input_ids.shape[1]} + {gen_length}"
            )
            decoded = _decode_generated_tails_at_eos(
                tokenizer, generated, prompt_len, batch_eos_id
            )
            assert len(decoded) == batch_size, (
                f"{name}: expected {batch_size} decoded strings, got {len(decoded)}"
            )
            print(f"  [batch] {name}: shape={generated.shape}, steps={steps_taken}, blocks={blocks}, time={gen_time:.2f}s")
            for i, text in enumerate(decoded):
                print(f"    [{name}] sample {i}:")
                print(f"      {text}")
        except Exception as e:
            failures.append((name, e))
            print(f"  [batch] {name}: FAILED - {e}")
            traceback.print_exc()

    if failures:
        raise AssertionError(f"Batch tests failed for: {[f[0] for f in failures]}")
    print(f"Batch tests passed for {len(modes_to_run)} scheduler(s).")


def _free_model(model, tokenizer, device: str) -> None:
    del model, tokenizer
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()


def main(scheduler_name=None, use_wandb=False, wandb_project=None, wandb_name=None,
         intervention_mode=None, intervention_steps=None, intervention_k=1, run_batch_tests_only=False, batch_size=2,
         model_paths=None):
    device = "cuda"
    paths = list(model_paths) if model_paths else ["./models/llada"]

    # Batch tests: run batched generation
    if run_batch_tests_only:
        if use_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: wandb not available. Install with: pip install wandb")
                print("Continuing without wandb logging...")
                use_wandb = False
            elif wandb.run is None:
                if wandb_name is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    suffix = f"_{scheduler_name}" if scheduler_name else "_all"
                    wandb_name = f"test_schedulers_batch{suffix}_{timestamp}"
                wandb.init(
                    settings=wandb.Settings(init_timeout=120),
                    project=wandb_project or "strace",
                    name=wandb_name,
                    config={
                        "scheduler": scheduler_name or "all",
                        "device": device,
                        "batch_size": batch_size,
                        "model_paths": paths,
                    },
                )
                print(f"Initialized wandb run: {wandb_name}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for model_dir in paths:
            mlabel = _model_label(model_dir)
            print(f"\n{'=' * 60}\nModel: {model_dir}  ({mlabel})\n{'=' * 60}\n")
            trace_name_base = f"batch_{timestamp}_{mlabel}"
            model, tokenizer = load_model_and_tokenizer(model_dir, device=device, compile_model=False)
            if tokenizer.padding_side != "left":
                tokenizer.padding_side = "left"
            _mid, _eos, _eot = infer_diffusion_token_ids(model_dir)
            assert tokenizer.pad_token_id != _mid, "pad token must differ from [MASK] for batched tests"
            print(
                f"Running batch tests (batch_size={batch_size}, model={mlabel}, "
                f"tracing={'on' if trace_name_base else 'off'})..."
            )
            try:
                run_batch_tests(
                    model, tokenizer, device,
                    batch_size=batch_size,
                    scheduler_name=scheduler_name,
                    trace_name_base=trace_name_base,
                    upload_to_wandb=use_wandb,
                    model_path=model_dir,
                )
            finally:
                _free_model(model, tokenizer, device)
        if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
        return

    # Initialize wandb if requested
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not available. Install with: pip install wandb")
            print("Continuing without wandb logging...")
            use_wandb = False
        elif wandb.run is None:
            # Generate wandb run name if not provided
            if wandb_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                scheduler_suffix = f"_{scheduler_name}" if scheduler_name else "_all"
                wandb_name = f"test_schedulers{scheduler_suffix}_{timestamp}"
            
            wandb.init(
                settings=wandb.Settings(init_timeout=120),
                project=wandb_project or "strace",
                name=wandb_name,
                config={
                    "scheduler": scheduler_name or "all",
                    "device": device,
                    "model_paths": paths,
                }
            )
            print(f"Initialized wandb run: {wandb_name}")
        else:
            print("Wandb already initialized, using existing run")

    # Define available schedulers
    schedulers = {
        'pure_diffusion': test_pure_diffusion,
        'block_diffusion': test_block_diffusion,
        'top_k_confidence': test_top_k_confidence,
        'top_k_margin': test_top_k_margin,
        'fhs': test_fhs,
        'eb_sampler': test_eb_sampler,
        'dus': test_dus,
        'fdm': test_fdm,
    }

    # Determine which schedulers to run
    if scheduler_name:
        if scheduler_name not in schedulers:
            print(f"Error: Unknown scheduler '{scheduler_name}'")
            print(f"Available schedulers: {', '.join(schedulers.keys())}")
            return
        schedulers_to_run = {scheduler_name: schedulers[scheduler_name]}
    else:
        schedulers_to_run = schedulers

    for model_dir in paths:
        mlabel = _model_label(model_dir)
        print(f"\n{'=' * 60}\nModel: {model_dir}  ({mlabel})\n{'=' * 60}\n")
        model, tokenizer = load_model_and_tokenizer(model_dir, device=device, compile_model=False)
        mask_id, eos_id, eot_id = infer_diffusion_token_ids(model_dir)

        if tokenizer.padding_side != 'left':
            tokenizer.padding_side = 'left'

        assert tokenizer.pad_token_id != mask_id, "pad token must differ from [MASK] for correct inference"

        prompt = "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"

        encoded_outputs = tokenizer(
            prompt,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoded_outputs['input_ids'].to(device)
        attention_mask = encoded_outputs['attention_mask'].to(device)

        results = []
        for name, test_func in schedulers_to_run.items():
            try:
                trace_name = None
                if use_wandb:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if intervention_mode is not None:
                        trace_name = f"{mlabel}_{name}_{intervention_mode}_{ts}"
                    else:
                        trace_name = f"{mlabel}_{name}_{ts}"

                output, label = test_func(
                    model, input_ids, attention_mask, tokenizer, 
                    trace_name=trace_name, intervention_mode=intervention_mode, intervention_steps=intervention_steps, intervention_k=intervention_k,
                    mask_id=mask_id, eos_id=eos_id, eot_id=eot_id,
                )
                results.append((label, output))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error running {name} on {mlabel}: {e}")
                print()

        if results:
            print("Output:")
            for label, output in results:
                for o in output:
                    print(f"[{mlabel}] {label}:")
                    print(o)
                    print('-' * 50)

        _free_model(model, tokenizer, device)
    
    # Finish wandb run if it was initialized
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test different schedulers')
    parser.add_argument(
        '--scheduler', 
        type=str, 
        default=None,
        choices=['pure_diffusion', 'block_diffusion', 'top_k_confidence', 'top_k_margin', 'fhs', 'eb_sampler', 'dus', 'fdm'],
        help='Scheduler name to test. If not provided, all schedulers will be tested.'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable wandb logging and trace uploads'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='Wandb project name (default: "strace")'
    )
    parser.add_argument(
        '--wandb_name',
        type=str,
        default=None,
        help='Wandb run name (default: auto-generated)'
    )
    parser.add_argument(
        '--intervention_mode',
        type=str,
        default=None,
        choices=['drop', 'swap'],
        help='Intervention mode: "drop" to prevent tokens from being unmasked, "swap" to swap tokens'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run batch execution tests only (batched generation via schedulers.generate)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size for --batch tests (default: 2)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        action='append',
        default=None,
        metavar='DIR',
        help='Local model directory (same as config model_path; repeat for multiple models). Default: ./models/llada',
    )
    args = parser.parse_args()

    # Set default intervention parameters (steps=[0], k=1 when mode is specified)
    intervention_steps = [0] if args.intervention_mode is not None else None
    intervention_k = 1

    if args.batch:
        print("... Running batch tests ...")
        if args.model_path:
            print(f"Model path(s): {', '.join(args.model_path)}")
        main(
            scheduler_name=args.scheduler,
            run_batch_tests_only=True,
            batch_size=args.batch_size,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            model_paths=args.model_path,
        )
    else:
        print("... Testing scheduler ...")
        if args.scheduler:
            print(f"Running scheduler: {args.scheduler}")
        else:
            print("Running all schedulers")
        if args.model_path:
            print(f"Model path(s): {', '.join(args.model_path)}")
        else:
            print("Model path: ./models/llada (default)")
        if args.wandb:
            print("Wandb logging enabled")
        if args.intervention_mode:
            print(f"Intervention mode: {args.intervention_mode}")
            print(f"Intervention steps: {intervention_steps} (default)")
            print(f"Intervention k: {intervention_k} (default)")
        print()

        main(
        scheduler_name=args.scheduler,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        intervention_mode=args.intervention_mode,
        intervention_steps=intervention_steps,
        intervention_k=intervention_k,
        model_paths=args.model_path,
        )
