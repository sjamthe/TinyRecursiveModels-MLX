# Tiny Recursive Models - MLX

MLX implementation of Tiny Recursive Models (TRM) optimized for Apple Silicon.

## Overview

This project implements recursive reasoning models using Apple's MLX framework. The models are designed to solve reasoning puzzles like Sudoku and ARC challenges through iterative refinement.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
run_name="pretrain_att_arc1concept_4"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

### Inference

```bash
python single_inference.py
```

### Converting PyTorch Checkpoints

If you have a PyTorch checkpoint:

```bash
python convert_torch_to_mlx.py
```

See `CONVERTER_README.md` for detailed conversion instructions.

## Project Structure

```
├── models/                      # Neural network components
│   ├── layers.py               # MLX layers (Linear, Attention, etc.)
│   ├── sparse_embedding.py     # Sparse embedding layers
│   ├── losses.py               # Loss functions
│   ├── ema.py                  # Exponential moving average
│   └── recursive_reasoning/    # TRM model implementations
├── dataset/                    # Dataset builders
│   ├── build_sudoku_dataset.py
│   ├── build_arc_dataset.py
│   └── build_maze_dataset.py
├── config/                     # Model and training configs
├── pretrain.py                 # Training script
├── single_inference.py         # Inference script
└── puzzle_dataset.py          # Dataset loader
```

## Available Models

- **TRM** (`trm.py`): Standard Tiny Recursive Model
- **TRM-Hier6** (`trm_hier6.py`): Hierarchical variant
- **TRM-SingleZ** (`trm_singlez.py`): Single latent state variant
- **HRM** (`hrm.py`): Hierarchical Recursive Model
- **Transformers Baseline** (`transformers_baseline.py`): Comparison baseline

## Key Features

- **MLX-optimized**: Built for Apple Silicon (M1/M2/M3)
- **Automatic memory management**: No manual device handling
- **Simple training**: Unified interface for all models
- **Checkpoint compatibility**: Convert PyTorch models to MLX format

## Configuration

Model and training parameters are configured via YAML files in `config/`:

- `cfg_pretrain.yaml`: Training hyperparameters
- `arch/*.yaml`: Model architectures

## Data

The project includes datasets for:

- **Sudoku**: Extreme difficulty puzzles
- **ARC**: Abstract reasoning challenges

Data is preprocessed and stored in `data/` with train/test splits.

## MLX Notes

MLX handles several things automatically:

- Device placement (CPU/GPU)
- Memory management
- Graph optimization

No need for explicit `.to(device)` or `.cuda()` calls.

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Original TRM Paper](https://arxiv.org/abs/2402.08563)
