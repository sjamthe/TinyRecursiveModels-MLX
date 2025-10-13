"""
HRM ACT V2: Transformer Baseline for Architecture Ablation

This is an architecture ablation of the Hierarchical Reasoning Model (HRM).
Key changes from V1:
1. REMOVED hierarchical split (no separate H and L levels)
2. REMOVED inner cycles (no H_cycles/L_cycles loops within reasoning)
3. KEPT ACT outer loop structure intact
4. KEPT all data preprocessing, embeddings, and evaluation infrastructure

Architecture: Single-level transformer that processes the full 30x30 grid as a
900-token sequence, with the same positional encodings and sparse embeddings as V1.

"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class Model_ACTV2InnerCarry:
    z_H: mx.array


@dataclass
class Model_ACTV2Carry:
    inner_carry: Model_ACTV2InnerCarry

    steps: mx.array
    halted: mx.array

    current_data: Dict[str, mx.array]


class Model_ACTV2Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int

    H_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    act_enabled: bool = True  # If False, always run halt_max_steps (no early stopping during training)
    act_inference: bool = False  # If True, use adaptive computation during inference

    forward_dtype: str = "bfloat16"


class Model_ACTV2Block(nn.Module):
    def __init__(self, config: Model_ACTV2Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def __call__(self, cos_sin: CosSin, hidden_states: mx.array) -> mx.array:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class Model_ACTV2ReasoningModule(nn.Module):
    def __init__(self, layers: List[Model_ACTV2Block]):
        super().__init__()

        self.layers = layers

    def __call__(self, hidden_states: mx.array, input_injection: mx.array, **kwargs) -> mx.array:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class Model_ACTV2_Inner(nn.Module):
    def __init__(self, config: Model_ACTV2Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(mx, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = Model_ACTV2ReasoningModule(
            layers=[Model_ACTV2Block(self.config) for _i in range(self.config.H_layers)]
        )

        # Initial states
        self.H_init = mx.array(
            trunc_normal_init_(mx.zeros(self.config.hidden_size, dtype=self.forward_dtype), std=1)
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        self.q_head.weight = mx.zeros_like(self.q_head.weight)
        self.q_head.bias = mx.full(self.q_head.bias.shape, -5, dtype=self.q_head.bias.dtype)

    def _input_embeddings(self, input: mx.array, puzzle_identifiers: mx.array):
        # Token embedding
        embedding = self.embed_tokens(input.astype(mx.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                # Pad along the last dimension: (batch_size, current_dim) -> (batch_size, current_dim + pad_count)
                puzzle_embedding = mx.pad(puzzle_embedding, ((0, 0), (0, pad_count)))

            embedding = mx.concatenate(
                [puzzle_embedding.reshape(puzzle_embedding.shape[0], self.puzzle_emb_len, self.config.hidden_size), embedding], axis=-2
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.astype(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return Model_ACTV2InnerCarry(
            z_H=mx.zeros(
                (batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size),
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: mx.array, carry: Model_ACTV2InnerCarry):
        return Model_ACTV2InnerCarry(
            z_H=mx.where(reset_flag[..., None, None], self.H_init, carry.z_H),
        )

    def __call__(
        self, carry: Model_ACTV2InnerCarry, batch: Dict[str, mx.array]
    ) -> Tuple[Model_ACTV2InnerCarry, mx.array, Tuple[mx.array, mx.array]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # 1-step grad
        z_H = self.H_level(carry.z_H, input_embeddings, **seq_info)

        # LM Outputs
        new_carry = Model_ACTV2InnerCarry(
            z_H=z_H,
        )  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).astype(mx.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class Model_ACTV2(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = Model_ACTV2Config(**config_dict)
        self.inner = Model_ACTV2_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, mx.array]):
        batch_size = batch["inputs"].shape[0]

        return Model_ACTV2Carry(
            inner_carry=self.inner.empty_carry(
                batch_size
            ),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=mx.zeros((batch_size,), dtype=mx.int32),
            halted=mx.ones((batch_size,), dtype=mx.bool_),  # Default to halted
            current_data={k: mx.zeros_like(v) for k, v in batch.items()},
        )

    def __call__(
        self,
        carry: Model_ACTV2Carry,
        batch: Dict[str, mx.array],
        compute_target_q: bool = False,
    ) -> Tuple[Model_ACTV2Carry, Dict[str, mx.array]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = mx.where(carry.halted, 0, carry.steps)

        new_current_data = {}
        for k, v in carry.current_data.items():
            # Create broadcast mask for halted sequences
            mask_shape = (batch[k].shape[0],) + (1,) * (batch[k].ndim - 1)
            mask = mx.broadcast_to(carry.halted.reshape(mask_shape), mask_shape)
            new_current_data[k] = mx.where(mask, batch[k], v)

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}

        # Step
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps

        halted = is_last_step

        # Check if adaptive computation should be used
        use_adaptive = (self.config.halt_max_steps > 1) and (
            (self.training and self.config.act_enabled)
            or (not self.training and self.config.act_inference)
        )

        if use_adaptive:
            # Halt signal based on Q-values (but always halt at max steps)
            q_halt_signal = q_halt_logits > q_continue_logits
            halted = halted | q_halt_signal

            # Store actual steps used for logging (only during inference)
            if not self.training:
                outputs["actual_steps"] = new_steps.astype(mx.float32)

            # Exploration (only during training)
            if self.training:
                min_halt_steps = (
                    mx.random.uniform(q_halt_logits.shape) < self.config.halt_exploration_prob
                ) * mx.random.randint(2, self.config.halt_max_steps + 1, q_halt_logits.shape)
                halted = halted & (new_steps >= min_halt_steps)

            # Compute target Q (only during training)
            # NOTE: No replay buffer and target networks for computing target Q-value.
            # As batch_size is large, there're many parallel envs.
            # Similar concept as PQN https://arxiv.org/abs/2407.04811
            if self.training and compute_target_q:
                _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                    new_inner_carry, new_current_data
                )

                outputs["target_q_continue"] = 1.0 / (1.0 + mx.exp(
                    -mx.where(
                        is_last_step,
                        next_q_halt_logits,
                        mx.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                ))

        return Model_ACTV2Carry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs
