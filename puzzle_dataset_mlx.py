import os
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic

import mlx.core as mx

from models.losses_mlx import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata

from argdantic import ArgParser
from pydantic import BaseModel

def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.


class PuzzleDataset:
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        self.config = config
        self.split = split

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            sets=prev_sets,
            total_puzzles=total_puzzles,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples
        )

        # Load data
        self.inputs = []
        self.labels = []
        self.puzzle_identifiers = []
        self.puzzle_indices = []
        self.group_indices = []
        self.sets = []

        for dataset_path in config.dataset_paths:
            current_data = self._load_data(dataset_path)
            self.inputs.append(current_data["inputs"])
            self.labels.append(current_data["labels"])
            self.puzzle_identifiers.append(current_data["puzzle_identifiers"])
            self.puzzle_indices.append(current_data["puzzle_indices"])
            self.group_indices.append(current_data["group_indices"])
            self.sets.append(current_data["sets"])

        # Concatenate
        self.inputs = np.concatenate(self.inputs, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.puzzle_identifiers = np.concatenate(self.puzzle_identifiers, axis=0)
        self.puzzle_indices = np.concatenate(self.puzzle_indices, axis=0)
        self.group_indices = np.concatenate(self.group_indices, axis=0)
        self.sets = np.concatenate(self.sets, axis=0)

        # RNG
        self.rng = np.random.default_rng(config.seed)

    def _load_metadata(self, dataset_path: str) -> PuzzleDatasetMetadata:
        metadata_path = os.path.join(dataset_path, self.split, "dataset.json")
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)
        return PuzzleDatasetMetadata(**metadata_dict)

    def _load_data(self, dataset_path: str) -> Dict[str, np.ndarray]:
        data_path = os.path.join(dataset_path, self.split)
        return {
            "inputs": np.load(os.path.join(data_path, "all__inputs.npy")),
            "labels": np.load(os.path.join(data_path, "all__labels.npy")),
            "puzzle_identifiers": np.load(os.path.join(data_path, "all__puzzle_identifiers.npy")),
            "puzzle_indices": np.load(os.path.join(data_path, "all__puzzle_indices.npy")),
            "group_indices": np.load(os.path.join(data_path, "all__group_indices.npy")),
            "sets": np.load(os.path.join(data_path, "all__puzzle_identifiers.npy")),  # Using puzzle identifiers as sets
        }

    def __iter__(self):
        # Create group order
        group_order = np.arange(self.metadata.total_groups)
        if not self.config.test_set_mode:
            self.rng.shuffle(group_order)

        start_index = 0
        for _ in range(self.config.epochs_per_iter):
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    self.rng, group_order, self.puzzle_indices, self.group_indices, start_index, self.config.global_batch_size
                )

                # Create batch
                batch = {
                    "inputs": self.inputs[batch_indices],
                    "labels": self.labels[batch_indices],
                    "puzzle_identifiers": self.puzzle_identifiers[batch_indices],
                }

                # Get set name (use first puzzle identifier as set name)
                set_name = self.sets[batch_indices[0]]

                yield set_name, batch, len(batch_indices)


def create_custom_puzzle_batch(sudoku_puzzle: np.ndarray, metadata: PuzzleDatasetMetadata) -> Dict[str, mx.array]:
    """
    Create a custom puzzle batch from a Sudoku puzzle for inference.
    
    Args:
        sudoku_puzzle: 9x9 numpy array with the Sudoku puzzle
        metadata: Dataset metadata containing vocab info
        
    Returns:
        Dictionary with batch data compatible with the model
    """
    # Flatten the puzzle to 1D
    puzzle_flat = sudoku_puzzle.flatten()
    
    # Convert to the expected format (assuming vocab mapping)
    # This is a simplified mapping - you may need to adjust based on your actual vocab
    inputs = puzzle_flat.copy()
    labels = puzzle_flat.copy()
    
    # Set empty cells (0) to pad_id for inputs, and keep as 0 for labels (to be predicted)
    inputs[inputs == 0] = metadata.pad_id
    labels[labels == 0] = metadata.ignore_label_id
    
    # Add batch dimension
    inputs = inputs[None, :]  # Shape: (1, 81)
    labels = labels[None, :]  # Shape: (1, 81)
    
    # Create puzzle identifier (using a dummy value)
    puzzle_identifiers = np.array([0], dtype=np.int32)
    
    # Convert to MLX arrays
    batch = {
        "inputs": mx.array(inputs),
        "labels": mx.array(labels),
        "puzzle_identifiers": mx.array(puzzle_identifiers),
    }
    
    return batch
