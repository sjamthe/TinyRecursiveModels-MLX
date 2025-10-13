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
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: mx.array
    z_L: mx.array


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: mx.array
    halted: mx.array
    
    current_data: Dict[str, mx.array]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

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

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool=False # use mlp on L instead of transformer

class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def __call__(self, cos_sin: CosSin, hidden_states: mx.array) -> mx.array:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = mx.transpose(hidden_states, (0, 2, 1))
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = mx.transpose(hidden_states, (0, 2, 1))
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = layers

    def __call__(self, hidden_states: mx.array, input_injection: mx.array, **kwargs) -> mx.array:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(mx, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = mx.array(trunc_normal_init_(mx.zeros(self.config.hidden_size, dtype=self.forward_dtype), std=1))
        self.L_init = mx.array(trunc_normal_init_(mx.zeros(self.config.hidden_size, dtype=self.forward_dtype), std=1))

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

            embedding = mx.concatenate([puzzle_embedding.reshape(puzzle_embedding.shape[0], self.puzzle_emb_len, self.config.hidden_size), embedding], axis=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.astype(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=mx.zeros((batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size), dtype=self.forward_dtype),
            z_L=mx.zeros((batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size), dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: mx.array, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=mx.where(reset_flag[..., None, None], self.H_init, carry.z_H),
            z_L=mx.where(reset_flag[..., None, None], self.L_init, carry.z_L),
        )

    def __call__(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, mx.array]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, mx.array, Tuple[mx.array, mx.array]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        for _H_step in range(self.config.H_cycles):
            for _L_step in range(self.config.L_cycles):
                if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            if not (_H_step == self.config.H_cycles - 1):
                z_H = self.H_level(z_H, z_L, **seq_info)
        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).astype(mx.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, mx.array]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=mx.zeros((batch_size,), dtype=mx.int32),
            halted=mx.ones((batch_size,), dtype=mx.bool_),  # Default to halted
            
            current_data={k: mx.zeros_like(v) for k, v in batch.items()}
        )
        
    def __call__(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, mx.array]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, mx.array]]:
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
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # Step
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps
        
        halted = is_last_step

        # if training, and ACT is enabled
        if self.training and (self.config.halt_max_steps > 1):
            # Halt signal
            # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
            halted = halted | (q_halt_logits > q_continue_logits)

            # Exploration
            min_halt_steps = (mx.random.uniform(q_halt_logits.shape) < self.config.halt_exploration_prob) * mx.random.randint(2, self.config.halt_max_steps + 1, q_halt_logits.shape)

            halted = halted & (new_steps >= min_halt_steps)

            # Compute target Q
            # NOTE: No replay buffer and target networks for computing target Q-value.
            # As batch_size is large, there're many parallel envs.
            # Similar concept as PQN https://arxiv.org/abs/2407.04811
            _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
            
            outputs["target_q_continue"] = 1.0 / (1.0 + mx.exp(-mx.where(is_last_step, next_q_halt_logits, mx.maximum(next_q_halt_logits, next_q_continue_logits))))

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
