# Intervention Analysis: Drop and Swap Effects on Different Schedulers

This document analyzes when drop and swap interventions can cause tokens to remain masked in different scheduler types.

## Overview

Interventions (drop/swap) modify which tokens get unmasked during generation. However, depending on the scheduler architecture, some interventions can cause tokens to remain masked permanently if applied at critical points.

## Scheduler Categories

### 1. Block-Based Schedulers (LLaDA, DUS, EB Sampler, Top-K)

**Architecture:**
- Process generation in blocks (e.g., `block_length=32` or `256`)
- Each block is processed independently
- Once a block completes, it's never revisited

**Intervention Safety:**

#### Case A: `block_length >= gen_length` (Single Block)
**SAFE** - Dropped/swapped tokens can be recovered

- Only one block exists (`num_blocks = 1`)
- All steps process the same block
- Each step considers all masked positions
- Dropped/swapped tokens remain in `mask_index` and can be selected again in later steps

**Example:**
```
block_length = 256, gen_length = 256 → num_blocks = 1

Step 0: Drop pos 0 -> remains masked 
Step 1: mask_index includes pos 0 -> can select again 
Step 2: Still can select pos 0 -> can select again 
```

#### Case B: `block_length < gen_length` (Multiple Blocks)
**RISKY** - Tokens can remain masked if dropped/swapped at block boundaries

- Multiple blocks processed sequentially
- After all steps for Block 0 complete, scheduler moves to Block 1
- Block 0 is never revisited
- If a token is dropped/swapped in the last step of Block 0, it remains masked forever

**Example:**
```
block_length = 32, gen_length = 256 → num_blocks = 8

Block 0: [M M M ... M]  (32 tokens)
Step 31 (last step): Drop pos 0
Result: pos 0 remains masked
Block 0 complete -> Move to Block 1
pos 0 never gets another chance 
```

**Affected Schedulers:**
- `generate_llada` (when `block_length < gen_length`)
- `generate_EB` (when `block_length < gen_length`)
- `generate_topk` (when `block_length < gen_length`)

### 2. DUS (Dilated Unmasking Scheduler) 

**Architecture:**
- Uses a **fixed schedule** (binary or dilated) determined at block start
- Each position is scheduled for specific step(s)
- Schedule doesn't adapt during execution

**Intervention Safety:**

**RISKY** - Even with `block_length >= gen_length`

- Schedule is fixed and doesn't change
- If a token is dropped in its scheduled step, it won't be in later steps' `step_indices`
- The code only processes positions in the current step's schedule

**Example:**
```
Block 0: [M M M M M M M M]  (8 tokens)
Schedule (binary): 
  Step 0: [pos 4]           (midpoint)
  Step 1: [pos 2, pos 6]    (midpoints of halves)
  Step 2: [pos 1, pos 3, pos 5, pos 7]  (remainder)

Step 0: Drop pos 4 -> remains masked 
Step 1: step_indices = [pos 2, pos 6]  (pos 4 NOT in schedule) 
Step 2: step_indices = [pos 1, pos 3, pos 5, pos 7]  (pos 4 NOT in schedule) 

Block complete -> pos 4 remains masked forever! 
```

**Why Remasking Doesn't Help:**
- Remasking only affects positions that were already unmasked in that step
- It doesn't help with dropped positions that were never unmasked

### 3. Non-Block Schedulers (FDM, FHS)

**Architecture:**
- Process entire sequence without block boundaries
- Each step considers all masked positions
- Continue until all tokens are unmasked (or step limit reached)

**Intervention Safety:**

**SAFE** - Dropped/swapped tokens can be recovered

- No block boundaries that could be abandoned
- Each step processes all remaining masked positions
- Dropped/swapped tokens remain in `mask_index` and can be selected again

**Example (FHS):**
```
Step 10: Drop pos 50 -> remains in masked tensor 
Step 11: Randomly selects from ALL masked positions (including pos 50) -> can select again 
Step 12: Still can select pos 50 -> can select again 
Continues until masked.any() == False
```

**Example (FDM):**
```
Step 10: Drop pos 50 -> remains masked 
Step 11: mask_index includes pos 50 -> can select again 
Step 12: Still can select pos 50 -> can select again 
```

**Affected Schedulers:**
- `generate_fdm` - Always safe
- `generate_fhs` - Always safe