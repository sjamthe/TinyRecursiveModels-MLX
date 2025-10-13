# Tiny Recursive Models - MLX Conversion

This document describes the conversion of the Tiny Recursive Models (TRM) project from PyTorch to MLX.

## Overview

The original project was built using PyTorch with distributed training capabilities. This conversion adapts the codebase to use Apple's MLX framework, which is optimized for Apple Silicon (M1/M2/M3) chips.

## Key Changes

### 1. Dependencies

- **Removed**: `torch`, `torch.distributed`, `torch.utils.data`
- **Added**: `mlx`, `mlx-lm`, `mlx.data`
- **Updated**: `requirements.txt` to reflect MLX dependencies

### 2. Core Components Converted

#### Models (`models/`)

- `common.py` → `common_mlx.py`: MLX-compatible initialization functions
- `layers.py` → `layers_mlx.py`: MLX neural network layers (Linear, Embedding, Attention, etc.)
- `sparse_embedding.py` → `sparse_embedding_mlx.py`: MLX sparse embedding implementation
- `losses.py` → `losses_mlx.py`: MLX-compatible loss functions
- `ema.py` → `ema_mlx.py`: Exponential Moving Average for MLX
- `recursive_reasoning/trm.py` → `recursive_reasoning/trm_mlx.py`: Main TRM model

#### Training (`pretrain.py` → `pretrain_mlx.py`)

- Removed distributed training (MLX handles this differently)
- Updated optimizer usage to MLX patterns
- Simplified data loading (no more DataLoader)
- Updated checkpoint saving/loading to use MLX format

#### Data (`puzzle_dataset.py` → `puzzle_dataset_mlx.py`)

- Removed PyTorch DataLoader dependency
- Direct numpy/MLX array handling
- Simplified batch creation

#### Inference (`single_inference.py` → `single_inference_mlx.py`)

- Updated to use MLX arrays and operations
- Simplified model loading

### 3. Key Technical Changes

#### Array Operations

- `torch.tensor()` → `mx.array()`
- `torch.zeros()` → `mx.zeros()`
- `torch.ones()` → `mx.ones()`
- `torch.randn()` → `mx.random.normal()`
- `torch.randint()` → `mx.random.randint()`

#### Neural Network Operations

- `torch.nn.Module` → `mlx.nn.Module`
- `torch.nn.functional` → `mlx.nn.functional`
- `torch.optim` → `mlx.optimizers`

#### Gradient Computation

- `torch.autograd` → `mlx.value_and_grad()`
- Simplified gradient handling (no `.backward()`)

#### Device Management

- Removed explicit device management (MLX handles this automatically)
- No more `.to(device)` or `.cuda()` calls

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python pretrain_mlx.py --config config/cfg_pretrain.yaml
```

### Testing the Conversion

```bash
python test_mlx_conversion.py
```

### Single Inference

```bash
python single_inference_mlx.py
```

## Limitations and Considerations

### 1. Distributed Training

- The original PyTorch version supported distributed training with `torch.distributed`
- MLX handles parallelism differently, so distributed training is simplified
- For multi-GPU training, you may need to use MLX's built-in parallelism features

### 2. Memory Management

- MLX uses different memory management patterns
- Some operations may have different memory footprints

### 3. Performance

- MLX is optimized for Apple Silicon
- Performance characteristics may differ from PyTorch on other hardware

### 4. Compatibility

- Some PyTorch-specific features may not have direct MLX equivalents
- Custom operations may need reimplementation

## File Structure

```
TinyRecursiveModels/
├── models/
│   ├── common_mlx.py              # MLX utilities
│   ├── layers_mlx.py              # MLX neural network layers
│   ├── sparse_embedding_mlx.py    # MLX sparse embeddings
│   ├── losses_mlx.py              # MLX loss functions
│   ├── ema_mlx.py                 # MLX EMA helper
│   └── recursive_reasoning/
│       └── trm_mlx.py             # Main TRM model
├── pretrain_mlx.py                # MLX training script
├── puzzle_dataset_mlx.py          # MLX data loading
├── single_inference_mlx.py        # MLX inference script
├── test_mlx_conversion.py         # Conversion test script
└── README_MLX.md                  # This file
```

## Testing

The `test_mlx_conversion.py` script verifies that:

1. Models can be created successfully
2. Forward passes work correctly
3. Loss computation functions properly
4. Gradient computation works for training

Run the test to ensure your conversion is working:

```bash
python test_mlx_conversion.py
```

## Migration Notes

If you're migrating from the PyTorch version:

1. **Update imports**: Change all PyTorch imports to MLX equivalents
2. **Check array operations**: Ensure all tensor operations use MLX arrays
3. **Update model definitions**: Use MLX module patterns
4. **Simplify training loops**: Remove distributed training complexity
5. **Update checkpoints**: Use MLX's `.npz` format instead of PyTorch's `.pt`

## Performance Optimization

For best performance on Apple Silicon:

1. Use `mx.compile()` for frequently called functions
2. Leverage MLX's automatic memory management
3. Use appropriate data types (float32 vs bfloat16)
4. Consider MLX's built-in optimizations for attention mechanisms

## Troubleshooting

Common issues and solutions:

1. **Import errors**: Ensure all MLX dependencies are installed
2. **Shape mismatches**: Check array shapes after conversion
3. **Gradient issues**: Verify `mx.value_and_grad()` usage
4. **Memory issues**: Monitor memory usage patterns

For more help, refer to the [MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html).
