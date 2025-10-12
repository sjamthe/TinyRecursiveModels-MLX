#!/usr/bin/env python3
"""
Single puzzle inference script for TRM model using MLX.
Loads one puzzle and runs inference on it.
"""

import os
import mlx.core as mx
import numpy as np
from typing import Dict, Any
import yaml

from puzzle_dataset_mlx import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata, create_custom_puzzle_batch
from utils.functions import load_model_class
from models.ema_mlx import EMAHelper


def parse_sudoku_string(sudoku_str: str) -> np.ndarray:
    """
    Parse a Sudoku puzzle string into a 9x9 numpy array.
    
    Expected format: 81 characters, where:
    - '0' or '.' represents empty cell
    - '1'-'9' represents filled cells
    - Spaces/newlines are ignored
    
    Example: "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    """
    # Remove spaces, newlines, and other whitespace
    clean_str = ''.join(sudoku_str.split())
    
    if len(clean_str) != 81:
        raise ValueError(f"Sudoku string must be exactly 81 characters, got {len(clean_str)}")
    
    # Convert to 9x9 array
    puzzle = np.zeros((9, 9), dtype=np.int32)
    for i, char in enumerate(clean_str):
        row = i // 9
        col = i % 9
        if char in '123456789':
            puzzle[row, col] = int(char)
        elif char in '0.':
            puzzle[row, col] = 0
        else:
            raise ValueError(f"Invalid character '{char}' at position {i}")
    
    return puzzle


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, metadata: PuzzleDatasetMetadata):
    """
    Load a trained model from checkpoint and config.
    """
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract model config
    arch_config = config_dict['arch']
    # Check if we have MLX checkpoint, otherwise use PyTorch model
    mlx_checkpoint_path = checkpoint_path + '.npz'
    if os.path.exists(mlx_checkpoint_path):
        print("Using MLX model and checkpoint")
        # Load MLX model classes
        if 'recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1' in arch_config['name']:
            from models.recursive_reasoning.trm_mlx import TinyRecursiveReasoningModel_ACTV1
            model_cls = TinyRecursiveReasoningModel_ACTV1
        else:
            model_cls = load_model_class(arch_config['name'])
        
        if 'losses@ACTLossHead' in arch_config['loss']['name']:
            from models.losses_mlx import ACTLossHead
            loss_head_cls = ACTLossHead
        else:
            loss_head_cls = load_model_class(arch_config['loss']['name'])
    else:
        print("MLX checkpoint not found, using PyTorch model")
        # Use PyTorch model classes
        model_cls = load_model_class(arch_config['name'])
        loss_head_cls = load_model_class(arch_config['loss']['name'])
    
    # Create model config using metadata
    model_cfg = {
        **arch_config,
        'batch_size': 1,  # Single inference
        'vocab_size': metadata.vocab_size,
        'seq_len': metadata.seq_len,
        'num_puzzle_identifiers': metadata.num_puzzle_identifiers,
        'causal': False
    }
    
    model = model_cls(model_cfg)
    # Only pass the loss_type parameter to ACTLossHead, not the entire config
    model = loss_head_cls(model, loss_type=arch_config['loss']['loss_type'])
    
    # Load checkpoint
    if os.path.exists(mlx_checkpoint_path):
        print(f"Loading MLX checkpoint from {mlx_checkpoint_path}")
        state_dict = mx.load(mlx_checkpoint_path)
        model.update(state_dict)
        print(f"Successfully loaded MLX checkpoint")
    elif os.path.exists(checkpoint_path):
        print(f"Loading PyTorch checkpoint from {checkpoint_path}")
        import torch
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle _orig_mod prefix mismatch (from torch.compile)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("Detected _orig_mod prefixes in checkpoint, removing them...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # Load PyTorch model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.device(device):
            model.load_state_dict(state_dict, assign=True)
            model.eval()
        print(f"Successfully loaded PyTorch checkpoint")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path} or {mlx_checkpoint_path}, using random weights")
    
    return model


def run_inference(model, batch: Dict[str, mx.array]):
    """Run inference on a single batch."""
    model.eval()
    
    # Check if this is a PyTorch model
    if hasattr(model, 'load_state_dict'):
        # PyTorch model - convert batch to PyTorch tensors
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_batch = {}
        for k, v in batch.items():
            if isinstance(v, mx.array):
                torch_batch[k] = torch.from_numpy(np.array(v)).to(device)
            else:
                torch_batch[k] = v
        
        carry = model.initial_carry(torch_batch)
        return model(carry=carry, batch=torch_batch, return_keys=["preds"])
    else:
        # MLX model
        carry = model.initial_carry(batch)
        return model(carry=carry, batch=batch, return_keys=["preds"])


def format_sudoku_output(preds: mx.array, original_puzzle: np.ndarray) -> np.ndarray:
    """
    Format the model predictions back into a 9x9 Sudoku grid.
    """
    # Convert MLX array to numpy
    preds_np = np.array(preds).flatten()
    
    # Create output grid
    output_grid = original_puzzle.copy()
    
    # Fill in predictions where original puzzle was empty
    empty_mask = (original_puzzle == 0)
    output_grid[empty_mask] = preds_np[empty_mask.flatten()]
    
    return output_grid


def print_sudoku_grid(grid: np.ndarray, title: str = "Sudoku"):
    """
    Print a 9x9 Sudoku grid in a nice format.
    """
    print(f"\n{title}:")
    print("+" + "-" * 21 + "+")
    for i in range(9):
        row = "|"
        for j in range(9):
            if j % 3 == 0 and j > 0:
                row += "|"
            if grid[i, j] == 0:
                row += " ."
            else:
                row += f" {grid[i, j]}"
        row += "|"
        print(row)
        if (i + 1) % 3 == 0 and i < 8:
            print("+" + "-" * 21 + "+")
    print("+" + "-" * 21 + "+")


def main():
    import sys
    
    # Configuration - use same paths as working PyTorch version
    dataset_path = "data/sudoku-extreme-1k-aug-1000"
    checkpoint_path = "checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_6510"
    config_path = "checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/all_config.yaml"
    
    # Check if custom Sudoku string is provided
    if len(sys.argv) > 1:
        # Custom Sudoku string provided
        sudoku_string = sys.argv[1]
        print("=== Custom Sudoku Puzzle Inference ===")
        print(f"Sudoku string: {sudoku_string}")
        
        # Parse the puzzle
        try:
            puzzle = parse_sudoku_string(sudoku_string)
            print_sudoku_grid(puzzle, "Original Puzzle")
        except ValueError as e:
            print(f"Error parsing Sudoku string: {e}")
            return
            
    else:
        # Use default example puzzle
        sudoku_string = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        print("=== Default Sudoku Puzzle Inference ===")
        puzzle = parse_sudoku_string(sudoku_string)
        print_sudoku_grid(puzzle, "Original Puzzle")
    
    # Load metadata from dataset
    with open(os.path.join(dataset_path, "test", "dataset.json"), "r") as f:
        metadata = PuzzleDatasetMetadata(**yaml.safe_load(f))
    
    # Create puzzle batch
    puzzle_batch = create_custom_puzzle_batch(puzzle, metadata)
    
    try:
        model = load_model_from_checkpoint(checkpoint_path, config_path, metadata)
        
        # Run inference
        print("Running inference...")
        result = run_inference(model, puzzle_batch)
        
        # Extract predictions from result tuple (similar to PyTorch version)
        if isinstance(result, tuple) and len(result) >= 4:
            preds_dict = result[3]  # predictions are in the 4th position
            preds = preds_dict['preds']
        elif hasattr(result, 'preds'):
            preds = result.preds
        elif isinstance(result, dict) and 'preds' in result:
            preds = result['preds']
        else:
            print(f"Unexpected result format: {result}")
            return
        
        # Convert predictions back to Sudoku format
        if hasattr(preds, 'cpu'):  # PyTorch tensor
            pred_np = preds.cpu().numpy()
        else:  # MLX array
            pred_np = np.array(preds)
            
        if pred_np.size >= 81:
            pred_flat = pred_np.flatten()[:81]  # Take first 81 elements
            # Convert model tokens back to Sudoku digits: 1->0 (empty), 2-10->1-9 (digits)
            sudoku_digits = np.where(pred_flat == 1, 0, pred_flat - 1)
            pred_9x9 = sudoku_digits.reshape(9, 9)
            print("\nOutput:")
            print(pred_9x9.astype(int))
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Make sure the checkpoint and config paths are correct.")


if __name__ == "__main__":
    main()
