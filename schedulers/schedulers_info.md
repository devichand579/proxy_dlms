# Diffusion Text Generation Schedulers: Algorithm Documentation

This document provides a comprehensive explanation of each diffusion-based text generation scheduler implemented in this codebase.

---

## Table of Contents

1. [Common Concepts](#common-concepts)
2. [LLaDA Scheduler](#llada-scheduler)
3. [Top-K Scheduler](#top-k-scheduler)
4. [First-Hitting Sampling (FHS)](#first-hitting-sampling-fhs)
5. [Entropy-Based (EB) Sampler](#entropy-based-eb-sampler)
6. [Dilated Unmasking Scheduler (DUS)](#dilated-unmasking-scheduler-dus)
7. [Foreseeing Decoding Method (FDM)](#foreseeing-decoding-method-fdm)
---

## Common Concepts

### Diffusion Process for Text Generation

All schedulers follow a reverse diffusion process:
1. **Initialization**: Start with a sequence of `[MASK]` tokens
2. **Iterative Denoising**: At each step, predict tokens for masked positions
3. **Selective Unmasking**: Choose which predicted tokens to "unmask" (reveal)
4. **Termination**: Stop when all tokens are unmasked or EOS is generated

### Key Components

- **Mask Predictor Model**: A masked diffusion language model (MDLM) that predicts tokens for masked positions
- **Confidence/Ranking Metric**: A score to determine which tokens to unmask (e.g., probability, entropy, margin)
- **Block Processing**: Some schedulers process text in contiguous blocks for efficiency
- **Early Stopping**: Terminate generation when EOS token is decoded and all preceding tokens are unmasked

### Notation

- `x`: Current sequence state (contains both prompt and generated tokens)
- `mask_id` / `eos_id` / `eot_id`: from the model config, or use `load_model.infer_diffusion_token_ids(model_path)`. Code fallbacks: `load_model.LLADA_DEFAULT_MASK_ID` (126336), `LLADA_DEFAULT_EOS_ID` (126081), `LLADA_DEFAULT_EOT_ID` (126348) for LLaDA-style checkpoints. In `schedulers.generate.generate`, pass `model_path` to set all three from `config.json`. Dream uses `eot_id=None`.
- `logits`: Model predictions (shape: `[batch, length, vocab_size]`)
- `x0`: Predicted tokens (argmax of logits after Gumbel noise)
- `confidence`: Ranking score for each position

---

## LLaDA Scheduler

### Overview

The **LLaDA (Large Language Diffusion model)** scheduler implements the diffusion process described in the LLaDA paper. It uses a **linear noise schedule** to determine how many tokens to transfer at each step, ensuring a consistent rate of token revelation.

### Algorithm

**Key Features:**
- **Variable token transfer**: Number of tokens unmasked per step varies based on a linear schedule
- **Block-based processing**: Divides generation into blocks for semi-autoregressive generation
- **Implicit block masking**: Restricts updates to current block via confidence masking

**Step-by-Step Process:**

1. **Initialization**
   ```
   x = [prompt | MASK MASK ... MASK]  (gen_length masks)
   ```

2. **Block Processing**
   - Divide `gen_length` into `num_blocks = gen_length / block_length`
   - Process each block sequentially
   - Steps per block: `steps_per_block = total_steps / num_blocks`

3. **For each block:**
   - Compute linear schedule: `num_transfer_tokens[j, i]` for each step `i`
   - This ensures tokens are transferred at a consistent rate

4. **For each step within block:**
   ```
   a. Forward pass: logits = model(x, attention_mask)
   b. Add Gumbel noise: logits_with_noise = add_gumbel_noise(logits, temperature)
   c. Predict tokens: x0 = argmax(logits_with_noise)
   d. Compute confidence: x0_p = get_mask_by_confidence(logits, x0)
   e. Apply block masking: x0_p[:, block_end:] = -inf
   f. Select top-k: k = num_transfer_tokens[j, i]  (variable k)
   g. Transfer tokens: x[selected_positions] = x0[selected_positions]
   h. Check early stopping: if EOS decoded → return
   ```

### Linear Noise Schedule

The schedule ensures a consistent rate of token revelation:
- Early steps: Transfer fewer tokens (more uncertainty)
- Later steps: Transfer more tokens (less uncertainty)
- Formula: `num_transfer_tokens[j, i]` computed via `get_num_transfer_tokens()`

### Example

```
Initial:  [prompt | M M M M M M M M]
Step 1:   [prompt | T1 M M M M M M M]  (k=1)
Step 2:   [prompt | T1 T2 M M M M M M]  (k=1)
Step 3:   [prompt | T1 T2 T3 T4 M M M M]  (k=2)
Step 4:   [prompt | T1 T2 T3 T4 T5 T6 M M]  (k=2)
...
```

### Parameters

- `steps`: Total diffusion steps (divided across blocks)
- `gen_length`: Length of generated sequence
- `block_length`: Size of each processing block
- `remasking`: Strategy for ranking (`'confidence'` or `'random'`)
- `temperature`: Gumbel noise temperature (0 = deterministic)

### Use Cases

- **Pure Diffusion**: `block_length == gen_length` (single block)
- **Block Diffusion**: `block_length < gen_length` (multiple blocks, semi-autoregressive)

---

## Top-K Scheduler

### Overview

The **Top-K scheduler** is a simple diffusion method that transfers exactly `top_k` tokens per step based on a ranking metric (confidence, entropy, top-margin, or random).

### Algorithm

**Key Features:**
- **Fixed token transfer**: Always transfers exactly `top_k` tokens per step
- **Multiple ranking metrics**: Supports confidence, top-margin, entropy, or random selection
- **Block-based processing**: Similar to LLaDA, processes in blocks

**Step-by-Step Process:**

1. **Initialization** (same as LLaDA)

2. **Block Processing**
   - Divide generation into blocks
   - Process each block sequentially

3. **For each step:**
   ```
   a. Forward pass: logits = model(x, attention_mask)
   b. Add Gumbel noise: logits_with_noise = add_gumbel_noise(logits, temperature)
   c. Predict tokens: x0 = argmax(logits_with_noise)
   d. Compute ranking score based on metric:
      - 'confidence': probability of predicted token
      - 'top_margin': difference between top-1 and top-2 probabilities
      - 'entropy': negative entropy (low entropy = high confidence)
      - 'random': random scores
   e. Apply block masking: x0_p[:, block_end:] = -inf
   f. Select top-k: k = min(top_k, remaining_masks_in_block)
   g. Transfer tokens: x[selected_positions] = x0[selected_positions]
   h. Check early stopping: if EOS decoded → return
   ```

### Ranking Metrics

1. **Confidence**: `P(token | context)` - probability of predicted token
2. **Top-Margin**: `P(top1) - P(top2)` - gap between best and second-best
3. **Entropy**: `-sum(P(v) * log P(v))` - uncertainty measure (lower = more confident)
4. **Random**: Random selection (for baseline)

### Example

```
Initial:  [prompt | M M M M M M M M]
Step 1:   [prompt | T1 T2 T3 T4 M M M M]  (top_k=4)
Step 2:   [prompt | T1 T2 T3 T4 T5 T6 T7 T8]  (top_k=4, done)
```

### Parameters

- `top_k`: Number of tokens to transfer per step (fixed)
- `metric`: Ranking metric (`'confidence'`, `'top_margin'`, `'entropy'`, `'random'`)
- `steps`: Total steps (divided across blocks)
- `block_length`: Block size

---

## First-Hitting Sampling (FHS)

### Overview

**First-Hitting Sampling (FHS)** uses stochastic "first-hitting times" (τ) to schedule token decoding in a **random order**. It employs second-order methods (extrapolation or predictor-corrector) for efficiency.

### Algorithm

**Key Features:**
- **Random-order decoding**: Tokens are decoded in random order based on hitting times
- **Chunk-based processing**: Divides sequence into equal chunks
- **Second-order methods**: Uses extrapolation or predictor-corrector for faster convergence
- **Time conditioning**: Optionally passes timestep to model

**Step-by-Step Process:**

1. **Initialization**
   ```
   x = [prompt | MASK MASK ... MASK]
   tau = [τ1, τ2, ..., τL]  (random hitting times for each position)
   ```

2. **Chunk Schedule**
   - Divide `gen_length` into `steps` equal chunks
   - Each chunk has `chunk_size = gen_length / steps` positions

3. **For each chunk:**
   ```
   a. Identify positions in current chunk
   b. For each position in chunk:
      - Update tau using linear schedule: tau = tau * u^(1/n_masked)
      - When tau <= threshold, decode that position
   c. Forward pass: logits = model(x, attention_mask, t=current_step)
   d. Sample tokens: x0 = sample_categorical(logits, temperature)
   e. Update x: x[decoded_positions] = x0[decoded_positions]
   f. Check early stopping: if EOS decoded → return
   ```

### First-Hitting Times (τ)

- Each masked position has a random "hitting time" τ
- When τ reaches a threshold, that position is decoded
- τ is updated stochastically: `τ_new = τ_old * u^(1/n_masked)` where `u ~ Uniform(0,1)`

### Second-Order Methods

1. **Extrapolation** (for `steps <= 128`):
   - Uses linear extrapolation to predict future states
   - Faster convergence with fewer steps

2. **Predictor-Corrector** (for `steps > 128`):
   - Predictor step: rough prediction
   - Corrector step: refinement
   - More accurate but slower

### Example

```
Initial:  [prompt | M M M M M M M M]
         tau:     [0.8, 0.3, 0.6, 0.2, 0.9, 0.1, 0.5, 0.7]

Chunk 1 (positions 0-3):
  Position 3 (tau=0.2) decoded → [prompt | M M M T4 M M M M]
  Position 1 (tau=0.3) decoded → [prompt | M T2 M T4 M M M M]
  ...

Chunk 2 (positions 4-7):
  Position 5 (tau=0.1) decoded → [prompt | M T2 M T4 M T6 M M]
  ...
```

### Parameters

- `steps`: Number of chunks (must divide `gen_length`)
- `temperature`: Sampling temperature (1.0 = stochastic, 0.0 = deterministic)
- `pass_time_to_model`: Whether to pass timestep to model for time conditioning


---

## Entropy-Based (EB) Sampler

### Overview

The **Entropy-Based (EB) sampler** uses an entropy-based selection rule with a **gamma threshold** to decide how many and which tokens to unmask, ensuring bounded cumulative uncertainty.

### Algorithm

**Key Features:**
- **Entropy constraint**: Selects tokens such that cumulative entropy is bounded
- **Adaptive token count**: Number of tokens transferred per step varies based on uncertainty
- **Block-based processing**: Processes in blocks like LLaDA/Top-K

**Step-by-Step Process:**

1. **Initialization** (same as LLaDA)

2. **Block Processing**
   - Divide generation into blocks
   - Process each block sequentially

3. **For each step:**
   ```
   a. Forward pass: logits = model(x, attention_mask)
   b. Add Gumbel noise: logits_with_noise = add_gumbel_noise(logits, temperature)
   c. Predict tokens: x0 = argmax(logits_with_noise)
   d. Compute ranking score E (confidence, margin, or entropy)
   e. Compute entropy: h = -sum(P(v) * log P(v)) for each position
   f. Sort positions by ranking score (highest first)
   g. Select tokens greedily:
      - For each position in sorted order:
        - Add its entropy to cumulative sum: H.append(h[pos])
        - If sum(H) - max(H) <= gamma:
            → Select this position
        - Else:
            → Stop (entropy constraint violated)
   h. Transfer selected tokens: x[selected] = x0[selected]
   i. Check early stopping: if EOS decoded → return
   ```

### Entropy Constraint

The constraint `sum(H) - max(H) <= gamma` ensures:
- **Bounded uncertainty**: Cumulative entropy is controlled
- **Adaptive selection**: More tokens selected when confident, fewer when uncertain
- **Quality control**: Prevents selecting too many uncertain tokens

### Example

```
Step t:
  Positions: [0, 1, 2, 3, 4]
  Entropy:   [0.1, 0.3, 0.2, 0.4, 0.15]
  Ranking:   [0.9, 0.7, 0.8, 0.6, 0.85]
  gamma = 0.5

  Sorted by ranking: [0, 4, 2, 1, 3]
  
  Selection:
    Position 0: H=[0.1], sum-max=0.0 <= 0.5 ✓ → Select
    Position 4: H=[0.1, 0.15], sum-max=0.05 <= 0.5 ✓ → Select
    Position 2: H=[0.1, 0.15, 0.2], sum-max=0.15 <= 0.5 ✓ → Select
    Position 1: H=[0.1, 0.15, 0.2, 0.3], sum-max=0.35 <= 0.5 ✓ → Select
    Position 3: H=[0.1, 0.15, 0.2, 0.3, 0.4], sum-max=0.6 > 0.5 ✗ → Stop
```

### Parameters

- `gamma`: Entropy threshold (lower = stricter, fewer tokens selected)
- `E`: Ranking metric (`'confidence'`, `'margin'`, or `'entropy'`)
- `steps`: Total steps (divided across blocks)
- `block_length`: Block size


---

## Dilated Unmasking Scheduler (DUS)

### Overview

The **Dilated Unmasking Scheduler (DUS)** uses **predefined hierarchical schedules** (dilated or binary search) to unmask tokens, rather than confidence-based selection. It optionally uses remasking to correct low-confidence predictions.

### Algorithm

**Key Features:**
- **Predefined schedule**: Token positions to unmask are determined by the schedule, not confidence
- **Hierarchical unmasking**: Reveals tokens in coarse-to-fine order
- **Optional remasking**: Can remask low-confidence tokens after initial unmasking
- **Two schedule types**: Binary search or dilated

**Step-by-Step Process:**

1. **Initialization** (same as LLaDA)

2. **Generate Schedule**
   - **Binary Search**: Reveals midpoints first, then recursively fills gaps
   - **Dilated**: Reveals positions at regular intervals, progressively filling gaps

3. **Block Processing**
   - Divide generation into blocks
   - For each block, generate schedule levels

4. **For each schedule level:**
   ```
   a. Get positions to unmask from schedule
   b. Forward pass: logits = model(x, attention_mask)
   c. Add Gumbel noise: logits_with_noise = add_gumbel_noise(logits, temperature)
   d. Predict tokens: x0 = argmax(logits_with_noise)
   e. Unmask scheduled positions: x[scheduled_positions] = x0[scheduled_positions]
   f. Optional remasking:
      - If remasking enabled:
        - Compute confidence/entropy for unmasked positions
        - If confidence < threshold (or entropy > threshold):
            → Remask that position (set back to MASK)
   g. Check early stopping: if EOS decoded → return
   ```

### Schedule Types

#### Binary Search Schedule

Reveals positions in binary search order:
```
Level 0: [4]           (midpoint)
Level 1: [2, 6]         (midpoints of left/right halves)
Level 2: [1, 3, 5, 7]   (remaining positions)
```

#### Dilated Schedule

Reveals positions at regular intervals, progressively filling:
```
Level 0: [0, 4, 8]      (stride=4)
Level 1: [2, 6]          (stride=2)
Level 2: [1, 3, 5, 7]    (stride=1, remainder)
```

### Remasking

After unmasking, low-confidence tokens can be remasked:
- **Confidence-based**: Remask if `confidence < threshold`
- **Entropy-based**: Remask if `entropy > threshold`

This allows correction of uncertain predictions.

### Example

**Binary Search Schedule:**
```
Initial:  [prompt | M M M M M M M M]

Level 0:  [prompt | M M M T4 M M M M]  (midpoint)
Level 1:  [prompt | M T2 M T4 M T6 M M]  (midpoints of halves)
Level 2:  [prompt | T1 T2 T3 T4 T5 T6 T7 T8]  (complete)
```

**Dilated Schedule (base=2):**
```
Initial:  [prompt | M M M M M M M M]

Level 0:  [prompt | T1 M M M T5 M M M]  (stride=4)
Level 1:  [prompt | T1 T3 M T5 T7 M M]  (stride=2)
Level 2:  [prompt | T1 T2 T3 T4 T5 T6 T7 T8]  (stride=1)
```

### Parameters

- `scheduler`: Schedule type (`'binary'` or `'dilated'`)
- `base`: Dilation base factor (for dilated schedule)
- `base_skip`: Initial stride exponent (for dilated schedule)
- `remasking`: Remasking strategy (`'confidence'`, `'entropy'`, or `None`)
- `confidence_threshold`: Threshold for remasking (default: 0.3)


---

## Foreseeing Decoding Method (FDM)

### Overview

The **Foreseeing Decoding Method (FDM)** uses a **"foreseeing" mechanism** that evaluates multiple candidate token placements (hypotheses) and selects the best based on combined local and global confidence scores. It adaptively switches between three modes based on confidence levels.

### Algorithm

**Key Features:**
- **Foreseeing**: Evaluates multiple hypotheses before selecting tokens
- **Global confidence**: Assesses sequence-level quality, not just local confidence
- **Adaptive strategy**: Switches between three modes based on confidence distribution
- **Combined scoring**: Uses `local_confidence + global_scale * global_confidence`

**Step-by-Step Process:**

1. **Initialization** (same as LLaDA)

2. **For each step:**
   ```
   a. Forward pass: logits = model(x, attention_mask)
   b. Add Gumbel noise: logits_with_noise = add_gumbel_noise(logits, temperature)
   c. Predict tokens: x0 = argmax(logits_with_noise)
   d. Compute local confidence: x0_logp = log P(x0 | context)
   e. Categorize positions by confidence:
      - High: confidence > eta1
      - Borderline: eta2 < confidence <= eta1
      - Low: confidence <= eta2
   f. Select mode based on distribution:
      - Case A (Exploration): No high-confidence → FDM1
      - Case B (Balance): Some borderline → FDM2
      - Case C (Acceleration): Many high-confidence → Pure local
   g. Execute selected mode (see below)
   h. Check early stopping: if EOS decoded → return
   ```

3. **Mode Execution:**

   **Case A: FDM1 (Exploration)**
   ```
   - Prune candidates: keep only positions with log_prob > log(gamma)
   - Select top-K1 candidates by local confidence
   - Build K1 hypotheses (each with one candidate placed)
   - Evaluate global confidence for each hypothesis
   - Combine: score = local + global_scale * global
   - Select best 1 token
   ```

   **Case B: FDM2 (Balance)**
   ```
   - Prune candidates: keep only positions with log_prob > log(gamma)
   - Select top-1 candidate by local confidence
   - Build 1 hypothesis
   - Evaluate global confidence
   - Combine scores
   - Select best N tokens (parallel decoding)
   ```

   **Case C: Pure Local (Acceleration)**
   ```
   - Select top-N tokens by local confidence only
   - No foreseeing (fast parallel decoding)
   ```

### Global Confidence

The global confidence estimator evaluates how well a hypothesis fits the overall sequence:

```
For each hypothesis:
  1. Get model logits → log_softmax → probabilities p
  2. Compute expected log-prob: e_lp = sum_v (p_v * log p_v) per position
  3. Average e_lp over masked positions
  4. Return: average expected log-prob (higher = better hypothesis)
```

This captures sequence-level coherence, not just local token probability.

### Example

**Step with confidence distribution:**
```
Positions: [0, 1, 2, 3, 4, 5, 6, 7]
Confidence: [0.5, 0.6, 0.75, 0.85, 0.9, 0.65, 0.55, 0.5]

Analysis:
  High (>0.8): [0.85, 0.9] → 2 tokens
  Borderline (0.7-0.8): [0.75] → 1 token
  Low (<0.7): [0.5, 0.6, 0.65, 0.55, 0.5] → 5 tokens

Decision: Case C (Acceleration)
  → Decode top-4 by local confidence: [0.9, 0.85, 0.75, 0.65]
```

**Step with no high-confidence:**
```
Positions: [0, 1, 2, 3]
Confidence: [0.5, 0.6, 0.65, 0.55]

Analysis:
  High (>0.8): [] → 0 tokens
  Borderline (0.7-0.8): [] → 0 tokens
  Low (<0.7): [0.5, 0.6, 0.65, 0.55] → 4 tokens

Decision: Case A (Exploration - FDM1)
  → Prune candidates (log_prob > log(0.2))
  → Select top-K1=4 candidates
  → Build 4 hypotheses
  → Evaluate global confidence
  → Select best 1 token
```

### Parameters

- `K1`: Deep foreseeing width for FDM1 (number of hypotheses to evaluate)
- `N`: Max parallel tokens for FDM2 and acceleration mode
- `eta1`: High-confidence threshold (default: 0.8)
- `eta2`: Borderline threshold (default: 0.7)
- `gamma`: Pruning threshold for candidate selection (default: 0.2)
- `global_scale`: Weight for global confidence in combined score (default: 1.0)


