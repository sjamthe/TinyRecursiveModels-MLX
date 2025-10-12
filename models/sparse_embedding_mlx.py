from typing import Union, Tuple
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from models.common_mlx import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: mx.Dtype):
        super().__init__()
        self.cast_to = cast_to

        # Real Weights
        # Truncated LeCun normal init
        self.weights = mx.array(
            trunc_normal_init_(mx.zeros((num_embeddings, embedding_dim)), std=init_std)
        )

        # Local weights and IDs
        # Local embeddings, with gradient, not persistent
        self.local_weights = mx.zeros((batch_size, embedding_dim))
        # Local embedding IDs, not persistent
        self.local_ids = mx.zeros((batch_size,), dtype=mx.int32)

    def __call__(self, inputs: mx.array) -> mx.array:
        if not self.training:
            # Test mode, no gradient
            return self.weights[inputs].astype(self.cast_to)
            
        # Training mode, fill puzzle embedding from weights
        # Resize local weights to match input batch size
        batch_size = inputs.shape[0]
        if self.local_weights.shape[0] != batch_size:
            self.local_weights = mx.zeros((batch_size, self.local_weights.shape[1]))
            self.local_ids = mx.zeros((batch_size,), dtype=mx.int32)
        
        self.local_weights = self.weights[inputs]
        self.local_ids = inputs

        return self.local_weights.astype(self.cast_to)


class CastedSparseEmbeddingSignSGD_MLX(optim.Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, mx.array] = 1e-3,
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= learning_rate:
            raise ValueError(f"Invalid learning rate: {learning_rate}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> Tuple[mx.array, dict]:
        # Apply SignSGD with decoupled weight decay
        parameter = parameter * (1.0 - self.learning_rate * self.weight_decay)
        parameter = parameter + mx.sign(gradient) * (-self.learning_rate)
        return parameter, state
