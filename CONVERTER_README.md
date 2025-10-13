# PyTorch to MLX Checkpoint Converter

This tool converts PyTorch model checkpoints to MLX format for use with Apple's MLX framework.

## Features

- ✅ Converts PyTorch `state_dict` to MLX `npz` format
- ✅ Handles BFloat16 tensors (converts to Float32)
- ✅ Removes `torch.compile` prefixes (`_orig_mod.`)
- ✅ Verification mode to ensure conversion accuracy
- ✅ Preserves weight shapes and values

## Usage

### Basic Conversion

```bash
python convert_torch_to_mlx.py <pytorch_checkpoint> [mlx_checkpoint]
```

If `mlx_checkpoint` is not specified, it will automatically append `.npz` to the input path.

### Examples

#### Convert the step_6510 checkpoint

```bash
python convert_torch_to_mlx.py \
    checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_6510 \
    checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_6510_mlx.npz
```

#### Convert with verification

```bash
python convert_torch_to_mlx.py \
    checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_6510 \
    checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_6510_mlx.npz \
    --verify
```

#### Auto-generate output path

```bash
python convert_torch_to_mlx.py \
    checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_6510
# Output: step_6510.npz
```

#### Quiet mode

```bash
python convert_torch_to_mlx.py \
    checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_6510 \
    --quiet
```

## Command Line Options

- `torch_checkpoint` (required): Path to the PyTorch checkpoint file
- `mlx_checkpoint` (optional): Path to save the MLX checkpoint (default: `<torch_checkpoint>.npz`)
- `--verify`: Verify conversion by comparing tensors between PyTorch and MLX
- `--quiet`: Suppress verbose output

## Conversion Details

### Key Name Transformations

- Removes `_orig_mod.` prefix (added by `torch.compile`)
- Example: `_orig_mod.model.inner.H_init` → `model.inner.H_init`

### Data Type Handling

- **BFloat16**: Automatically converted to Float32 (MLX/numpy compatibility)
- **Float32, Int32, etc.**: Preserved as-is
- All other PyTorch dtypes are converted to their numpy equivalents

### Weight Layout

- Linear layer weights maintain PyTorch convention: `(out_features, in_features)`
- No transposition is performed by default
- Can be customized via `should_transpose_weight()` function if needed

## Output Format

The MLX checkpoint is saved in `.npz` format using `mlx.savez()`, which is compatible with MLX's loading functions:

```python
import mlx.core as mx

# Load MLX checkpoint
weights = mx.load("step_6510_mlx.npz")

# Access individual weights
h_init = weights["model.inner.H_init"]
```

## Verification

The `--verify` flag performs the following checks:

1. Confirms all PyTorch keys are present in MLX checkpoint
2. Verifies shape consistency
3. Compares tensor values (default tolerance: 1e-5)

## Example Output

```
Loading PyTorch checkpoint from: checkpoints/.../step_6510
Found 15 parameters

Sample keys:
  _orig_mod.model.inner.H_init: shape=torch.Size([512]), dtype=torch.bfloat16
  _orig_mod.model.inner.L_init: shape=torch.Size([512]), dtype=torch.bfloat16
  _orig_mod.model.inner.embed_tokens.embedding_weight: shape=torch.Size([11, 512]), dtype=torch.float32
  ...

Converted 15 parameters to MLX format

Saving MLX checkpoint to: checkpoints/.../step_6510_mlx.npz
Conversion complete!

File sizes:
  PyTorch: 19.19 MB
  MLX: 19.19 MB

Verifying conversion...
PyTorch keys: 15
MLX keys: 15

✓ Verification passed! All tensors match within tolerance.
```

## Requirements

- Python 3.8+
- PyTorch
- MLX
- NumPy

Install dependencies:

```bash
pip install torch mlx numpy
```

## Converted Checkpoint

The converted checkpoint `step_6510_mlx.npz` is ready to use with MLX models. You'll need to:

1. Implement the model architecture in MLX (similar to `models/recursive_reasoning/trm.py`)
2. Load the weights using `mx.load()`
3. Map the weights to your MLX model's parameters

## Notes

- The converter preserves the original checkpoint's structure
- BFloat16 tensors lose precision when converted to Float32
- The output file size is similar to the input (compression depends on npz settings)
- For production use, consider implementing the model in MLX with proper dtype handling
