import math
import mlx.core as mx
import mlx.nn as nn


def trunc_normal_init_(tensor: mx.array, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """
    MLX version of truncated normal initialization.
    Simplified version using normal distribution with clipping.
    """
    if std == 0:
        return mx.zeros_like(tensor)
    else:
        # Generate normal random values
        result = mx.random.normal(shape=tensor.shape, scale=std)
        # Clip to bounds
        result = mx.clip(result, lower * std, upper * std)
        return result
