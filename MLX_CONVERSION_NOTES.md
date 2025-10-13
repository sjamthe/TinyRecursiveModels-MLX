# PyTorch to MLX Conversion Notes

## Overview

This document describes the changes made to convert `pretrain.py` and `puzzle_dataset.py` from PyTorch to MLX (Apple's machine learning framework).

## Key Changes

### 1. Import Statements

**Before (PyTorch):**

```python
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
```

**After (MLX):**

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
```

### 2. Data Loading

**Changes:**

- Removed `DataLoader` wrapper - MLX works directly with Python iterables
- Changed `PuzzleDataset` from `IterableDataset` to plain class
- Removed PyTorch DataLoader-specific features (num_workers, pin_memory, etc.)
- Changed final tensor conversion from `torch.from_numpy()` to `mx.array()`
- Removed `get_worker_info()` checks

### 3. Model Initialization

**Changes:**

- Removed CUDA device checks (`cuda.is_available()`)
- Removed `torch.device()` context manager
- Removed `torch.compile()` call (MLX compiles automatically)
- Removed distributed training parameter broadcasting

### 4. Optimizers

**Before (PyTorch):**

```python
torch.optim.Adam(
    model.parameters(),
    lr=0,
    weight_decay=config.weight_decay,
    betas=(config.beta1, config.beta2)
)
```

**After (MLX):**

```python
optim.AdamW(
    learning_rate=0,
    betas=[config.beta1, config.beta2],
    weight_decay=config.weight_decay
)
```

**Key Differences:**

- Changed `torch.optim.Adam` to `mlx.optimizers.AdamW`
- Changed parameter name `lr` to `learning_rate`
- Changed `betas` from tuple to list
- Optimizer doesn't take model parameters in constructor (passed during update)
- Changed `CastedSparseEmbeddingSignSGD_Distributed` to `CastedSparseEmbeddingSignSGD`

### 5. Training Loop

**Before (PyTorch):**

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

**After (MLX):**

```python
# Define loss function
def loss_fn(model):
    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
    return loss, (carry, metrics)

# Compute gradients
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
(loss, (carry, metrics)), grads = loss_and_grad_fn(model)

# Update with optimizer
optimizer.learning_rate = lr
optimizer.update(model, grads)
```

**Key Differences:**

- MLX uses functional gradient computation with `nn.value_and_grad()`
- No separate `.backward()` call
- Optimizer updates model in-place with `.update()` method
- Learning rate set directly as attribute, not through param_groups

### 6. Model Training/Eval Modes

**Changes:**

- Removed `model.train()` and `model.eval()` calls
- MLX doesn't have explicit train/eval mode switching
- Behavior controlled by whether gradients are computed

### 7. Checkpoint Saving/Loading

**Before (PyTorch):**

```python
torch.save(model.state_dict(), path)
state_dict = torch.load(path)
model.load_state_dict(state_dict)
```

**After (MLX):**

```python
mx.savez(path, **model.parameters())
weights = mx.load(path)
model.update(weights)
```

### 8. Distributed Training

**Changes:**

- Removed all `torch.distributed` (dist) calls
- Removed process group initialization
- Removed `dist.broadcast()`, `dist.all_reduce()`, `dist.reduce()`
- Added comments noting MLX doesn't have built-in distributed training
- Single-device training only (multi-device would require custom implementation)

### 9. Tensor Operations

**Changes:**

- Removed `.to(device)` calls for moving tensors to GPU
- Changed `.cpu()` to direct numpy conversion
- Changed `.numpy()` to `np.array()`
- Changed `torch.stack()` to `mx.array()` or `mx.concatenate()`
- Changed `torch.zeros()` to `mx.zeros()`
- Changed `torch.mean()` to `mx.mean()`
- Changed parameter `.numel()` to `.size`
- Changed `.expand()` to `mx.broadcast_to()`

### 10. RNG Seeding

**Before (PyTorch):**

```python
torch.random.manual_seed(seed)
```

**After (MLX):**

```python
mx.random.seed(seed)
np.random.seed(seed)
```

### 11. Project Naming

**Change:**

- Changed default project name suffix from "-ACT-torch" to "-ACT-mlx"

## Files Modified

1. **pretrain.py** - Main training script
2. **puzzle_dataset.py** - Dataset loading and batching

## Files Already in MLX Format

These files were already converted:

- `models/sparse_embedding.py`
- `models/ema.py`
- `single_inference.py`

## Limitations

1. **No Distributed Training**: MLX doesn't have built-in distributed training like PyTorch. Multi-GPU training would require custom implementation.
2. **No Explicit Train/Eval Modes**: Unlike PyTorch, MLX doesn't have `.train()` and `.eval()` modes for dropout/batch norm behavior switching.
3. **Different Optimizer API**: Optimizers work differently - they update models in-place rather than stepping through parameter groups.

## Testing Recommendations

1. Test with single GPU/device first
2. Verify checkpoint loading/saving works correctly
3. Check that gradient computation produces expected values
4. Validate that metrics are computed correctly
5. Ensure EMA (Exponential Moving Average) works with MLX models

## Additional Notes

- MLX is optimized for Apple Silicon (M1/M2/M3 chips)
- MLX uses unified memory architecture - no explicit device management needed
- MLX automatically compiles computational graphs for efficiency
