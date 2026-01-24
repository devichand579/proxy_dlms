### 3. Batch Processing Optimizations

#### A. Batch Multiple Prompts
```python
def generate_batch(model, prompts, attention_masks=None, ...):
    # Process multiple prompts in parallel
    batch_size = prompts.shape[0]
    # Allocate batch tensor
    x = torch.full((batch_size, ...), mask_id, ...)
    # Process all prompts together
```

#### B. Dynamic Batching
- Group prompts by similar length
- Pad to max length in batch
- Process together

#### C. Clear Intermediate Tensors
```python
# Explicitly delete large tensors
del logits, logits_with_noise, x0_p
torch.cuda.empty_cache()  # If needed
```

### 5. Evaluation-Specific Optimizations



#### B. Parallel Evaluation
```python
# Use multiprocessing or torch.multiprocessing
# Process multiple evaluations in parallel
from torch.multiprocessing import Pool
```

#### C. Reduce Redundant Computations
```python
# In eval_llada.py, batch multiple MC samples
# Instead of sequential, process in batches
```

