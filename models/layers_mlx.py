from typing import Tuple
import einops
import mlx.core as mx
import mlx.nn as nn

from models.common_mlx import trunc_normal_init_


CosSin = Tuple[mx.array, mx.array]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: mx.array):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.astype(cos.dtype)
    k = k.astype(cos.dtype)

    q_embed = (q * cos[None, :, None, :]) + (rotate_half(q) * sin[None, :, None, :])
    k_embed = (k * cos[None, :, None, :]) + (rotate_half(k) * sin[None, :, None, :])

    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = mx.array(
            trunc_normal_init_(mx.zeros((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = mx.zeros((out_features,))

    def __call__(self, input: mx.array) -> mx.array:
        weight = self.weight.astype(input.dtype)
        bias = self.bias.astype(input.dtype) if self.bias is not None else None
        # Manual linear layer: input @ weight.T + bias
        return mx.matmul(input, weight.T) + (bias if bias is not None else 0)


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: mx.Dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = mx.array(
            trunc_normal_init_(mx.zeros((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def __call__(self, input: mx.array) -> mx.array:
        # Manual embedding lookup: input is indices, embedding_weight is the lookup table
        return self.embedding_weight.astype(self.cast_to)[input]


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        t = mx.arange(max_position_embeddings, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = mx.cos(emb)
        self.sin_cached = mx.sin(emb)

    def __call__(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def __call__(self, cos_sin: CosSin, hidden_states: mx.array) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # scaled dot product attention
        # Manual reshape: (B, S, H, D) -> (B, H, S, D)
        query = query.transpose(0, 2, 1, 3)  # (B, S, H, D) -> (B, H, S, D)
        key = key.transpose(0, 2, 1, 3)      # (B, S, H, D) -> (B, H, S, D)
        value = value.transpose(0, 2, 1, 3)  # (B, S, H, D) -> (B, H, S, D)
        # Manual scaled dot product attention
        scale = 1.0 / (query.shape[-1] ** 0.5)
        # Transpose key: (B, H, S, D) -> (B, H, D, S)
        key_t = key.transpose(0, 1, 3, 2)
        attn_weights = mx.matmul(query, key_t) * scale
        
        if self.causal:
            # Apply causal mask
            seq_len = query.shape[-2]
            causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=query.dtype))
            attn_weights = mx.where(causal_mask == 0, -mx.inf, attn_weights)
        
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = mx.matmul(attn_weights, value)
        # Manual reshape: (B, H, S, D) -> (B, S, H, D)
        attn_output = attn_output.transpose(0, 2, 1, 3)  # (B, H, S, D) -> (B, S, H, D)
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def __call__(self, x):
        if self.reverse:
            return nn.silu(self.linear(x))
        else:
            return self.linear(nn.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def __call__(self, x):
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


def rms_norm(hidden_states: mx.array, variance_epsilon: float) -> mx.array:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(mx.float32)

    variance = mx.mean(hidden_states ** 2, axis=-1, keepdims=True)
    hidden_states = hidden_states * mx.rsqrt(variance + variance_epsilon)
    return hidden_states.astype(input_dtype)
