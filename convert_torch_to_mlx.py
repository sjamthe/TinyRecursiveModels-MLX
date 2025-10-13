#!/usr/bin/env python3
"""
Convert PyTorch checkpoint to MLX format.

This script converts a PyTorch checkpoint (saved with torch.save) to MLX format (saved with mlx.save).
It handles:
- Loading PyTorch state dict
- Converting tensor types from torch to MLX
- Adapting weight layouts if needed (e.g., transposing linear layers)
- Saving in MLX format
"""

import os
import argparse
import shutil
from pathlib import Path
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def torch_to_mlx_array(torch_tensor: torch.Tensor) -> mx.array:
    """
    Convert a PyTorch tensor to an MLX array.
    
    Args:
        torch_tensor: PyTorch tensor to convert
        
    Returns:
        MLX array with the same data
    """
    # Handle BFloat16 by converting to float32 first
    # (numpy doesn't support bfloat16 directly)
    if torch_tensor.dtype == torch.bfloat16:
        torch_tensor = torch_tensor.to(torch.float32)
    
    # Move to CPU and convert to numpy
    numpy_array = torch_tensor.cpu().numpy()
    
    # Convert to MLX array
    mlx_array = mx.array(numpy_array)
    
    return mlx_array


def convert_key_name(key: str) -> str:
    """
    Convert PyTorch key names to MLX-compatible names.
    
    MLX typically uses simpler naming conventions.
    Removes '_orig_mod.' prefix which is added by torch.compile.
    
    Args:
        key: Original PyTorch key name
        
    Returns:
        Converted key name for MLX
    """
    # Remove torch.compile prefix
    if key.startswith('_orig_mod.'):
        key = key[len('_orig_mod.'):]
    
    return key


def should_transpose_weight(key: str) -> bool:
    """
    Determine if a weight tensor should be transposed for MLX.
    
    PyTorch and MLX may have different conventions for linear layer weights.
    PyTorch: (out_features, in_features)
    MLX: (in_features, out_features) - needs verification
    
    Args:
        key: Weight key name
        
    Returns:
        True if the weight should be transposed
    """
    # For now, keep the same layout. MLX typically doesn't require transposition
    # as it uses similar conventions to PyTorch for most layers
    return False


def convert_checkpoint(
    torch_checkpoint_path: str,
    mlx_checkpoint_path: str,
    copy_config: bool = True,
    verbose: bool = True
):
    """
    Convert a PyTorch checkpoint to MLX format.
    
    Args:
        torch_checkpoint_path: Path to PyTorch checkpoint file
        mlx_checkpoint_path: Path to save MLX checkpoint
        verbose: Whether to print conversion details
    """
    if verbose:
        print(f"Loading PyTorch checkpoint from: {torch_checkpoint_path}")
    
    # Load PyTorch checkpoint
    torch_state_dict = torch.load(torch_checkpoint_path, map_location='cpu')
    
    if verbose:
        print(f"Found {len(torch_state_dict)} parameters")
        print(f"\nSample keys:")
        for i, key in enumerate(list(torch_state_dict.keys())[:5]):
            tensor = torch_state_dict[key]
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        if len(torch_state_dict) > 5:
            print(f"  ... and {len(torch_state_dict) - 5} more")
    
    # Convert to MLX format
    mlx_state_dict = {}
    
    for key, value in torch_state_dict.items():
        # Convert key name
        new_key = convert_key_name(key)
        
        # Convert tensor to MLX array
        mlx_array = torch_to_mlx_array(value)
        
        # Transpose if needed
        if should_transpose_weight(key):
            mlx_array = mlx_array.T
            if verbose:
                print(f"  Transposed {new_key}")
        
        mlx_state_dict[new_key] = mlx_array
    
    # Add runtime buffers for CastedSparseEmbedding if puzzle_emb.weights exists
    # These are needed for MLX's update() method to work properly
    if any('puzzle_emb.weights' in k for k in mlx_state_dict.keys()):
        for key in list(mlx_state_dict.keys()):
            if 'puzzle_emb.weights' in key:
                # Extract the prefix (e.g., "model.inner.puzzle_emb")
                prefix = key.replace('.weights', '')
                # Get the embedding dimension from the weights tensor
                emb_dim = mlx_state_dict[key].shape[1]
                # Add local_weights (batch_size=1 for inference) and local_ids
                mlx_state_dict[f"{prefix}.local_weights"] = mx.zeros((1, emb_dim))
                mlx_state_dict[f"{prefix}.local_ids"] = mx.zeros((1,), dtype=mx.int32)
                if verbose:
                    print(f"  Added runtime buffers for {prefix}")
    
    if verbose:
        print(f"\nConverted {len(mlx_state_dict)} parameters to MLX format")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(mlx_checkpoint_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save MLX checkpoint
    if verbose:
        print(f"\nSaving MLX checkpoint to: {mlx_checkpoint_path}")
    
    # MLX uses safetensors or npz format
    # For compatibility, we'll use the standard save format
    mx.savez(mlx_checkpoint_path, **mlx_state_dict)
    
    if verbose:
        print("Conversion complete!")
        print(f"\nMLX checkpoint saved to: {mlx_checkpoint_path}")
        
        # Print file size
        torch_size = os.path.getsize(torch_checkpoint_path) / (1024 ** 2)
        mlx_size = os.path.getsize(mlx_checkpoint_path) / (1024 ** 2)
        print(f"\nFile sizes:")
        print(f"  PyTorch: {torch_size:.2f} MB")
        print(f"  MLX: {mlx_size:.2f} MB")
    
    # Copy config file if it exists
    if copy_config:
        # Look for all_config.yaml in the same directory as the checkpoint
        config_dir = os.path.dirname(torch_checkpoint_path)
        config_path = os.path.join(config_dir, "all_config.yaml")
        
        if os.path.exists(config_path):
            # Copy config to the same directory as MLX checkpoint
            mlx_config_path = os.path.join(os.path.dirname(mlx_checkpoint_path), "all_config.yaml")
            
            # Only copy if paths are different
            if config_path != mlx_config_path:
                shutil.copy(config_path, mlx_config_path)
                if verbose:
                    print(f"\n✓ Copied config file to: {mlx_config_path}")
            elif verbose:
                print(f"\n✓ Config file already exists at: {config_path}")
        elif verbose:
            print(f"\n⚠ Warning: No all_config.yaml found in {config_dir}")
            print("  You will need the config file to reconstruct the model architecture!")


def verify_conversion(
    torch_checkpoint_path: str,
    mlx_checkpoint_path: str,
    tolerance: float = 1e-5
):
    """
    Verify that the conversion was successful by comparing tensors.
    
    Args:
        torch_checkpoint_path: Path to original PyTorch checkpoint
        mlx_checkpoint_path: Path to converted MLX checkpoint
        tolerance: Maximum allowed difference between tensors
    """
    print("\nVerifying conversion...")
    
    # Load both checkpoints
    torch_state_dict = torch.load(torch_checkpoint_path, map_location='cpu')
    mlx_state_dict = mx.load(mlx_checkpoint_path)
    
    print(f"PyTorch keys: {len(torch_state_dict)}")
    print(f"MLX keys: {len(mlx_state_dict)}")
    
    # Check each key
    mismatches = []
    for key, torch_tensor in torch_state_dict.items():
        mlx_key = convert_key_name(key)
        
        if mlx_key not in mlx_state_dict:
            mismatches.append(f"Key {mlx_key} not found in MLX checkpoint")
            continue
        
        mlx_array = mlx_state_dict[mlx_key]
        
        # Convert MLX back to numpy for comparison
        mlx_numpy = np.array(mlx_array)
        
        # Handle BFloat16 conversion
        if torch_tensor.dtype == torch.bfloat16:
            torch_tensor = torch_tensor.to(torch.float32)
        
        torch_numpy = torch_tensor.cpu().numpy()
        
        # Apply transpose if needed
        if should_transpose_weight(key):
            torch_numpy = torch_numpy.T
        
        # Compare shapes
        if mlx_numpy.shape != torch_numpy.shape:
            mismatches.append(f"Shape mismatch for {mlx_key}: MLX={mlx_numpy.shape}, PyTorch={torch_numpy.shape}")
            continue
        
        # Compare values
        max_diff = np.abs(mlx_numpy - torch_numpy).max()
        if max_diff > tolerance:
            mismatches.append(f"Value mismatch for {mlx_key}: max_diff={max_diff}")
    
    if mismatches:
        print(f"\nFound {len(mismatches)} issues:")
        for mismatch in mismatches[:10]:  # Print first 10
            print(f"  - {mismatch}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    else:
        print("\n✓ Verification passed! All tensors match within tolerance.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoint to MLX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a specific checkpoint
  python convert_torch_to_mlx.py checkpoints/model/step_6510 checkpoints/model/step_6510.npz
  
  # Convert with verification
  python convert_torch_to_mlx.py checkpoints/model/step_6510 checkpoints/model/step_6510.npz --verify
  
  # Auto-generate output path
  python convert_torch_to_mlx.py checkpoints/model/step_6510
        """
    )
    
    parser.add_argument(
        'torch_checkpoint',
        type=str,
        help='Path to PyTorch checkpoint file'
    )
    
    parser.add_argument(
        'mlx_checkpoint',
        type=str,
        nargs='?',
        default=None,
        help='Path to save MLX checkpoint (default: <torch_checkpoint>.npz)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify conversion by comparing tensors'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--no-copy-config',
        action='store_true',
        help='Do not copy all_config.yaml file'
    )
    
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if args.mlx_checkpoint is None:
        args.mlx_checkpoint = args.torch_checkpoint + '.npz'
    
    # Convert checkpoint
    convert_checkpoint(
        args.torch_checkpoint,
        args.mlx_checkpoint,
        copy_config=not args.no_copy_config,
        verbose=not args.quiet
    )
    
    # Verify if requested
    if args.verify:
        verify_conversion(
            args.torch_checkpoint,
            args.mlx_checkpoint
        )


if __name__ == '__main__':
    main()

