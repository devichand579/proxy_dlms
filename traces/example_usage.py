"""
Example usage of the unified tracing and intervention system.

This demonstrates how to use trace_generation with different schedulers
and intervention types.
"""

import torch
from traces import trace_generation, analyze_bottlenecks
from load_model import load_model_and_tokenizer


def example_basic_tracing():
    """Example: Basic tracing without interventions."""
    # Load model
    model, tokenizer = load_model_and_tokenizer("./models/llada", device="cuda")
    
    # Prepare prompt
    prompt_text = "Explain what artificial intelligence is."
    prompt = torch.tensor(tokenizer(prompt_text)['input_ids']).unsqueeze(0).to("cuda")
    
    # Trace generation
    result, trace = trace_generation(
        scheduler_type="pure_diffusion",
        model=model,
        prompt=prompt,
        steps=64,
        gen_length=64,
        trace_enabled=True
    )
    
    # Analyze the trace
    print(f"Total steps: {len(trace.steps)}")
    print(f"Positions unmasked: {len(trace.unmasking_order)}")
    
    # Check when specific positions were unmasked
    for pos in [10, 20, 30]:
        step = trace.get_unmasking_step(pos)
        if step is not None:
            print(f"Position {pos} was unmasked at step {step}")


def example_force_early():
    """Example: Force tokens to unmask early (causal intervention)."""
    model, tokenizer = load_model_and_tokenizer("./models/llada", device="cuda")
    
    prompt_text = "The capital of France is"
    prompt = torch.tensor(tokenizer(prompt_text)['input_ids']).unsqueeze(0).to("cuda")
    
    # Force positions 0 and 1 (first two tokens of generation) to unmask early
    # This tests if early tokens are necessary for good generation
    result, trace = trace_generation(
        scheduler_type="pure_diffusion",
        model=model,
        prompt=prompt,
        intervention_type="force_early",
        intervention_params={
            "positions": [0, 1],  # Relative to generation start
            # "tokens": [123, 456]  # Optional: force specific tokens
        },
        steps=64,
        gen_length=64
    )
    
    # Decode result
    generated_text = tokenizer.decode(result[0, prompt.shape[1]:], skip_special_tokens=True)
    print(f"Generated: {generated_text}")


def example_force_late():
    """Example: Force tokens to unmask late (test if they're bottlenecks)."""
    model, tokenizer = load_model_and_tokenizer("./models/llada", device="cuda")
    
    prompt_text = "What is machine learning?"
    prompt = torch.tensor(tokenizer(prompt_text)['input_ids']).unsqueeze(0).to("cuda")
    
    # Force positions 5-10 to unmask only after step 30
    # This tests if delaying these tokens affects generation quality
    result, trace = trace_generation(
        scheduler_type="top_k",
        model=model,
        prompt=prompt,
        intervention_type="force_late",
        intervention_params={
            "positions": list(range(5, 11)),
            "steps": [30]  # Don't unmask until step 30
        },
        gen_length=64
    )


def example_prevent_unmasking():
    """Example: Prevent certain positions from unmasking (test necessity)."""
    model, tokenizer = load_model_and_tokenizer("./models/llada", device="cuda")
    
    prompt_text = "The answer is"
    prompt = torch.tensor(tokenizer(prompt_text)['input_ids']).unsqueeze(0).to("cuda")
    
    # Prevent positions 10-15 from ever unmasking
    # This tests if these tokens are necessary for generation
    result, trace = trace_generation(
        scheduler_type="pure_diffusion",
        model=model,
        prompt=prompt,
        intervention_type="prevent",
        intervention_params={
            "positions": list(range(10, 16))
        },
        steps=64,
        gen_length=64
    )


def example_bottleneck_analysis():
    """Example: Analyze which tokens are bottlenecks."""
    model, tokenizer = load_model_and_tokenizer("./models/llada", device="cuda")
    
    prompt_text = "Explain quantum computing."
    prompt = torch.tensor(tokenizer(prompt_text)['input_ids']).unsqueeze(0).to("cuda")
    
    # Generate with tracing
    result, trace = trace_generation(
        scheduler_type="pure_diffusion",
        model=model,
        prompt=prompt,
        steps=64,
        gen_length=64
    )
    
    # Analyze bottlenecks
    bottlenecks = analyze_bottlenecks(trace, metric="confidence")
    
    # Print top bottlenecks
    sorted_bottlenecks = sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True)
    print("Top bottleneck positions:")
    for pos, score in sorted_bottlenecks[:10]:
        step = trace.get_unmasking_step(pos)
        print(f"  Position {pos}: score={score:.3f}, unmasked at step {step}")


def example_compare_schedulers():
    """Example: Compare different schedulers with same intervention."""
    model, tokenizer = load_model_and_tokenizer("./models/llada", device="cuda")
    
    prompt_text = "The weather today is"
    prompt = torch.tensor(tokenizer(prompt_text)['input_ids']).unsqueeze(0).to("cuda")
    
    schedulers = ["pure_diffusion", "top_k", "fdm"]
    results = {}
    
    for scheduler_type in schedulers:
        print(f"\n=== Testing {scheduler_type} ===")
        try:
            result, trace = trace_generation(
                scheduler_type=scheduler_type,
                model=model,
                prompt=prompt,
                steps=64,
                gen_length=64
            )
            
            generated_text = tokenizer.decode(result[0, prompt.shape[1]:], skip_special_tokens=True)
            results[scheduler_type] = {
                "text": generated_text,
                "steps": len(trace.steps),
                "unmasked_positions": len(trace.unmasking_order)
            }
            print(f"Generated: {generated_text[:100]}...")
            print(f"Steps taken: {len(trace.steps)}")
        except Exception as e:
            print(f"Error with {scheduler_type}: {e}")
    
    return results


def example_force_at_step():
    """Example: Force tokens to unmask at specific steps."""
    model, tokenizer = load_model_and_tokenizer("./models/llada", device="cuda")
    
    prompt_text = "The answer to the question is"
    prompt = torch.tensor(tokenizer(prompt_text)['input_ids']).unsqueeze(0).to("cuda")
    
    # Force position 0 to unmask at step 5, position 1 at step 10
    # This tests the causal effect of unmasking timing
    result, trace = trace_generation(
        scheduler_type="pure_diffusion",
        model=model,
        prompt=prompt,
        intervention_type="force_at_step",
        intervention_params={
            "steps": [5, 10],
            "positions": [0, 1],
            # "tokens": [123, 456]  # Optional: force specific tokens
        },
        steps=64,
        gen_length=64
    )
    
    generated_text = tokenizer.decode(result[0, prompt.shape[1]:], skip_special_tokens=True)
    print(f"Generated with forced timing: {generated_text}")
    
    # Verify interventions were applied
    prompt_len = prompt.shape[1]
    for pos in [0, 1]:
        abs_pos = prompt_len + pos
        step = trace.get_unmasking_step(abs_pos)
        print(f"Position {pos} (absolute {abs_pos}) unmasked at step {step}")


def example_force_token_values():
    """Example: Force specific token values at positions."""
    model, tokenizer = load_model_and_tokenizer("./models/llada", device="cuda")
    
    prompt_text = "The capital of France is"
    prompt = torch.tensor(tokenizer(prompt_text)['input_ids']).unsqueeze(0).to("cuda")
    
    # Force specific tokens at positions 0 and 1
    # First, let's see what tokens "Paris" would be
    paris_tokens = tokenizer("Paris", add_special_tokens=False)['input_ids']
    
    result, trace = trace_generation(
        scheduler_type="pure_diffusion",
        model=model,
        prompt=prompt,
        intervention_type="force_token",
        intervention_params={
            "positions": [0, 1],
            "tokens": paris_tokens[:2] if len(paris_tokens) >= 2 else paris_tokens + [paris_tokens[0]]
        },
        steps=64,
        gen_length=64
    )
    
    generated_text = tokenizer.decode(result[0, prompt.shape[1]:], skip_special_tokens=True)
    print(f"Generated with forced tokens: {generated_text}")


def example_causal_intervention_analysis():
    """
    Example: Comprehensive causal intervention analysis.
    
    This demonstrates how to test if early-selected tokens are actually necessary
    by comparing baseline vs. intervention results.
    """
    model, tokenizer = load_model_and_tokenizer("./models/llada", device="cuda")
    
    prompt_text = "Explain what artificial intelligence is."
    prompt = torch.tensor(tokenizer(prompt_text)['input_ids']).unsqueeze(0).to("cuda")
    
    # Baseline: no intervention
    print("=== Baseline (no intervention) ===")
    baseline_result, baseline_trace = trace_generation(
        scheduler_type="pure_diffusion",
        model=model,
        prompt=prompt,
        steps=64,
        gen_length=64
    )
    baseline_text = tokenizer.decode(baseline_result[0, prompt.shape[1]:], skip_special_tokens=True)
    print(f"Generated: {baseline_text[:200]}...")
    
    # Intervention: force early unmasking of first 5 positions
    print("\n=== Intervention: Force early unmasking of positions 0-4 ===")
    intervention_result, intervention_trace = trace_generation(
        scheduler_type="pure_diffusion",
        model=model,
        prompt=prompt,
        intervention_type="force_early",
        intervention_params={
            "positions": list(range(5))
        },
        steps=64,
        gen_length=64
    )
    intervention_text = tokenizer.decode(intervention_result[0, prompt.shape[1]:], skip_special_tokens=True)
    print(f"Generated: {intervention_text[:200]}...")
    
    # Compare results
    print("\n=== Comparison ===")
    print(f"Baseline length: {len(baseline_text)}")
    print(f"Intervention length: {len(intervention_text)}")
    print(f"Baseline steps: {len(baseline_trace.steps)}")
    print(f"Intervention steps: {len(intervention_trace.steps)}")
    
    # Analyze which positions were bottlenecks in baseline
    bottlenecks = analyze_bottlenecks(baseline_trace, metric="confidence")
    print("\nTop 5 bottleneck positions in baseline:")
    for pos, score in sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True)[:5]:
        step = baseline_trace.get_unmasking_step(pos)
        print(f"  Position {pos}: score={score:.3f}, unmasked at step {step}")


def example_analyze_unmasking_order():
    """Example: Analyze the order in which tokens are unmasked."""
    model, tokenizer = load_model_and_tokenizer("./models/llada", device="cuda")
    
    prompt_text = "The solution involves"
    prompt = torch.tensor(tokenizer(prompt_text)['input_ids']).unsqueeze(0).to("cuda")
    
    result, trace = trace_generation(
        scheduler_type="pure_diffusion",
        model=model,
        prompt=prompt,
        steps=64,
        gen_length=64
    )
    
    # Get all unmasked positions in order
    unmasked_positions = trace.get_all_unmasked_positions()
    prompt_len = prompt.shape[1]
    
    print("Unmasking order (first 20 positions):")
    for i, pos in enumerate(unmasked_positions[:20]):
        step = trace.get_unmasking_step(pos)
        rel_pos = pos - prompt_len
        token_id = int(result[0, pos].item())
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"  Step {step}: Position {rel_pos} (abs {pos}) -> {token_text}")


def main():
    """Run all examples."""
    print("=" * 80)
    print("Unified Tracing and Intervention System - Examples")
    print("=" * 80)
    
    examples = [
        ("Basic Tracing", example_basic_tracing),
        ("Force Early Intervention", example_force_early),
        ("Force Late Intervention", example_force_late),
        ("Prevent Unmasking", example_prevent_unmasking),
        ("Force at Step", example_force_at_step),
        ("Force Token Values", example_force_token_values),
        ("Bottleneck Analysis", example_bottleneck_analysis),
        ("Causal Intervention Analysis", example_causal_intervention_analysis),
        ("Compare Schedulers", example_compare_schedulers),
        ("Analyze Unmasking Order", example_analyze_unmasking_order),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nTo run a specific example, call the function directly.")
    print("Example: example_basic_tracing()")
    print("\nOr run all examples by calling main() and selecting an option.")
    
    # Uncomment to run a specific example:
    # example_basic_tracing()
    # example_force_early()
    # example_bottleneck_analysis()


if __name__ == "__main__":
    main()