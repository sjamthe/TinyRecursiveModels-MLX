#!/usr/bin/env python3
"""
Single puzzle inference script for TRM model.
Loads one puzzle and runs inference on it.
"""

import os
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, Any
import yaml

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class
from models.ema import EMAHelper


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


def create_custom_puzzle_batch(sudoku_puzzle: np.ndarray, metadata: PuzzleDatasetMetadata) -> Dict[str, mx.array]:
    """Create a batch from a custom Sudoku puzzle."""
    
    # Convert puzzle to the expected format
    # The model expects a flattened sequence with special tokens
    seq_len = metadata.seq_len
    
    # Create input sequence - need to map Sudoku digits correctly
    # The dataset does: raw Sudoku (0-9) -> add 1 -> model tokens (1-10)
    # So: 0 (empty) -> 1, 1-9 (digits) -> 2-10
    # Model vocab: 0=PAD, 1=empty, 2-10=digits 1-9
    puzzle_flat = sudoku_puzzle.flatten()
    
    # Map values: 0 (empty) -> 1, 1-9 (digits) -> 2-10
    mapped_puzzle = puzzle_flat + 1
    
    # Create input sequence (pad with metadata.pad_id)
    input_seq = np.full(seq_len, metadata.pad_id, dtype=np.int32)
    input_seq[:len(mapped_puzzle)] = mapped_puzzle
    
    # Create labels (same as input for now, but you might want to set this differently)
    label_seq = input_seq.copy()
    
    # Create puzzle identifier (single value per puzzle, not per sequence element)
    puzzle_id = np.array([0], dtype=np.int32)  # Use 0 as dummy identifier
    
    # Create batch
    batch = {
        "inputs": mx.array(input_seq)[None, :],
        "labels": mx.array(label_seq)[None, :],
        "puzzle_identifiers": mx.array(puzzle_id)[None, :]
    }
    
    return batch


def load_single_puzzle(dataset_path: str, puzzle_index: int = 0) -> tuple:
    """Load a single puzzle from the dataset."""
    
    # Load metadata
    with open(os.path.join(dataset_path, "test", "dataset.json"), "r") as f:
        metadata = PuzzleDatasetMetadata(**yaml.safe_load(f))
    
    # Load data files
    data = {}
    for field in ["inputs", "labels", "puzzle_identifiers"]:
        data[field] = np.load(os.path.join(dataset_path, "test", f"all__{field}.npy"))
    
    # Get puzzle indices
    puzzle_indices = np.load(os.path.join(dataset_path, "test", "all__puzzle_indices.npy"))
    
    # Extract single puzzle
    puzzle_start = puzzle_indices[puzzle_index]
    puzzle_end = puzzle_indices[puzzle_index + 1]
    
    # Create batch with single puzzle
    batch = {
        "inputs": mx.array(data["inputs"][puzzle_start:puzzle_end])[None, :],
        "labels": mx.array(data["labels"][puzzle_start:puzzle_end])[None, :],
        "puzzle_identifiers": mx.array(data["puzzle_identifiers"][puzzle_start:puzzle_end])[None, :]
    }
    
    return batch, metadata


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, metadata: PuzzleDatasetMetadata):
    """Load model from checkpoint and config."""
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create model config
    model_cfg = dict(
        **config["arch"],
        batch_size=1,  # Single example
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False
    )
    
    # Load model classes
    model_cls = load_model_class(config["arch"]["name"])
    loss_head_cls = load_model_class(config["arch"]["loss"]["name"])
    
    # Create model with loss head
    print(f"Loading checkpoint from {checkpoint_path}")
    model = model_cls(model_cfg)
    model = loss_head_cls(model, loss_type=config["arch"]["loss"]["loss_type"])
    
    # Load weights using MLX's load_weights method
    model.load_weights(checkpoint_path)
    
    model.eval()
    
    return model


def run_inference(model, batch: Dict[str, mx.array]):
    """Run inference on a single batch."""
    
    carry = model.initial_carry(batch)
    return model(carry=carry, batch=batch, return_keys=["preds"])


def main():
    import sys
    
    # Configuration
    dataset_path = "data/sudoku-extreme-1k-aug-1000"
    checkpoint_path = "checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_6510_mlx.npz"
    config_path = "checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/all_config.yaml"
    
    # Check if custom Sudoku string is provided
    if len(sys.argv) > 1:
        # Custom Sudoku string provided
        sudoku_string = sys.argv[1]
        print("=== Custom Sudoku Puzzle Inference ===")
        print(f"Sudoku string: {sudoku_string}")
        
        # Load metadata first
        with open(os.path.join(dataset_path, "test", "dataset.json"), "r") as f:
            metadata = PuzzleDatasetMetadata(**yaml.safe_load(f))
        
        # Parse custom puzzle
        print("\nParsing custom puzzle...")
        try:
            puzzle = parse_sudoku_string(sudoku_string)
            print("Parsed puzzle:")
            print(puzzle)
            
            # Create batch from custom puzzle
            batch = create_custom_puzzle_batch(puzzle, metadata)
            
        except ValueError as e:
            print(f"Error parsing Sudoku string: {e}")
            return
            
    else:
        # Use dataset puzzle
        puzzle_index = 0  # First puzzle
        print("=== Dataset Puzzle Inference ===")
        print(f"Dataset: {dataset_path}")
        print(f"Puzzle index: {puzzle_index}")
        
        # Load single puzzle from dataset
        print("\nLoading puzzle from dataset...")
        batch, metadata = load_single_puzzle(dataset_path, puzzle_index)
    
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load model
    print("Loading model...")
    model = load_model_from_checkpoint(checkpoint_path, config_path, metadata)
    
    # Run inference
    print("Running inference...")
    result = run_inference(model, batch)
    
    # Extract predictions from result tuple
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
    pred_np = np.array(preds)
    if pred_np.size >= 81:
        pred_flat = pred_np.flatten()[:81]  # Take first 81 elements
        # Convert model tokens back to Sudoku digits: 1->0 (empty), 2-10->1-9 (digits)
        sudoku_digits = np.where(pred_flat == 1, 0, pred_flat - 1)
        pred_9x9 = sudoku_digits.reshape(9, 9)
        print("\nOutput:")
        print(pred_9x9.astype(int))


if __name__ == "__main__":
    main()
