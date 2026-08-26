# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention layers."""

import math
from alphagenome import typing
from alphagenome_research.model import layers
import chex
from einshape import jax_einshape as einshape  # pylint: disable=g-importing-member
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-importing-member, g-multiple-import

_MAX_RELATIVE_DISTANCE = 8192  # 1Mb / 128bp.


def _shift(
    x: jnp.ndarray,
    query_length: int,
    key_length: int,
) -> jnp.ndarray:
  """Shifts the diagonal of a 2D array."""
  chex.assert_axis_dimension(x, -2, query_length)
  chex.assert_axis_dimension(x, -1, query_length + key_length)
  *batch_shapes, n_rows, n_diags = x.shape
  x = x.reshape(batch_shapes + [n_diags, n_rows])
  x = x[..., 1:, :]
  x = x.reshape(batch_shapes + [n_rows, n_diags - 1])
  return x[..., :key_length]


def _central_mask_features(
    *, distances: jnp.ndarray, feature_size: int, seq_length: int
) -> jnp.ndarray:
  """Positional features using exponentially-spaced central mask."""
  if feature_size > seq_length:
    raise ValueError(f'{feature_size=} must be <= than {seq_length=}. ')
  center_widths = jnp.arange(feature_size) + jnp.geomspace(
      1, seq_length - feature_size + 1, feature_size, endpoint=False
  )
  center_widths = jax.lax.broadcast_to_rank(center_widths, distances.ndim)
  outputs = (center_widths > distances[..., None]).astype(distances.dtype)
  chex.assert_shape(outputs, expected_shapes=distances.shape + (feature_size,))
  return outputs


def apply_rope(
    x: Float[Array, 'B S H C'],
    positions: Int[Array, 'B S'] | None,
    max_position: int,
) -> Float[Array, 'B S H C']:
  """Applies Rotary Position Embeddings to the input tensor."""
  if positions is None:
    positions = jnp.arange(x.shape[1]).astype(x.dtype).reshape(1, x.shape[1])
  num_freq = x.shape[-1] // 2
  inv_freq = 1.0 / (
      jnp.arange(num_freq)
      + jnp.geomspace(1, max_position - num_freq + 1, num_freq)
  ).astype(x.dtype)
  theta = jnp.einsum('bs,f->bsf', positions, inv_freq)
  theta = jnp.repeat(theta, 2, axis=-1)[..., None, :]  # [b, s, 1, c]
  x_rotated = jnp.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(
      x.shape
  )  # [b, s, h, c]
  return x * jnp.cos(theta) + x_rotated * jnp.sin(theta)


class MLPBlock(hk.Module):
  """MLP block for sequence representations."""

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S D'], *, is_training: bool
  ) -> Float[Array, 'B S D']:
    h = layers.RMSBatchNorm()(x, is_training=is_training)
    h = hk.Linear(x.shape[-1] * 2)(h)
    h = jax.nn.relu(h)
    h = hk.Linear(
        x.shape[-1], w_init=hk.initializers.TruncatedNormal(stddev=1e-6)
    )(h)
    return layers.RMSBatchNorm()(h, is_training=is_training)


class PairMLPBlock(hk.Module):
  """MLP block for pairwise representations."""

  @typing.jaxtyped
  def __call__(
      self, pair_input: Float[Array, 'B S S F']
  ) -> Float[Array, 'B S S F']:
    x = layers.LayerNorm(rms_norm=True)(pair_input)
    hidden_channels = 2 * pair_input.shape[-1]
    x = hk.Linear(hidden_channels)(x)
    x = jax.nn.relu(x)
    x = hk.Linear(
        pair_input.shape[-1],
        w_init=hk.initializers.TruncatedNormal(stddev=1e-6),
    )(x)
    return x


class MHABlock(hk.Module):
  """Multi-Head Attention block with residual connection."""

  @typing.jaxtyped
  def __call__(
      self,
      x: Float[Array, 'B S D'],
      attention_bias: Float[Array, 'B H S S'],
      *,
      is_training: bool,
  ) -> Float[Array, 'B S D']:
    batch_size, seq_len, _ = x.shape
    h = layers.RMSBatchNorm()(x, is_training=is_training)
    q = layers.LayerNorm(name='norm_q')(
        hk.Linear(8 * 128, with_bias=False, name='q_layer')(h).reshape(
            batch_size, seq_len, 8, 128
        )
    )
    k = layers.LayerNorm(name='norm_k')(
        hk.Linear(128, with_bias=False, name='k_layer')(h).reshape(
            batch_size, seq_len, 1, 128
        )
    )
    v = layers.LayerNorm(name='norm_v')(
        hk.Linear(192, with_bias=False, name='v_layer')(h).reshape(
            batch_size, seq_len, 1, 192
        )
    )
    q = apply_rope(q, None, max_position=_MAX_RELATIVE_DISTANCE)
    k = apply_rope(k, None, max_position=_MAX_RELATIVE_DISTANCE)

    logits_dtype = jnp.float32
    attention_logits = jnp.einsum(
        'bshc,bS1c->bhsS',
        q,
        k,
        precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        preferred_element_type=logits_dtype,
    )
    attention_logits = attention_logits / math.sqrt(128.0)
    attention_logits = (attention_logits + attention_bias).astype(logits_dtype)
    logits_soft_cap = 5.0
    attention_logits = (
        jnp.tanh(attention_logits / logits_soft_cap) * logits_soft_cap
    )
    attention_weights = jax.nn.softmax(attention_logits, axis=-1)

    y = jnp.einsum(
        'bhsS,bS1c->bshc',
        attention_weights,
        v,
        precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
    ).astype(q.dtype)
    y = hk.Linear(
        x.shape[-1],
        name='linear_embedding',
        w_init=hk.initializers.TruncatedNormal(stddev=1e-6),
    )(y.reshape(batch_size, seq_len, -1))
    return layers.RMSBatchNorm()(y, is_training=is_training)


class AttentionBiasBlock(hk.Module):
  """Generates attention bias for Multi-Head Attention."""

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B s s D'], is_training: bool
  ) -> Float[Array, 'B H S S']:
    x = jax.nn.gelu(layers.RMSBatchNorm()(x, is_training=is_training))
    # 8 = number of heads in sequence MHA.
    x = hk.Linear(8, with_bias=False, w_init=jnp.zeros)(x)
    for axis in [1, 2]:
      x = jnp.repeat(x, repeats=16, axis=axis)  # [B S S H]
    return jnp.moveaxis(x, 3, 1)


class RowAttentionBlock(hk.Module):
  """Self-attention block applied along rows of pairwise representations."""

  @typing.jaxtyped
  def __call__(
      self, pair_input: Float[Array, 'B s s F']
  ) -> Float[Array, 'B s s F']:
    x = layers.LayerNorm(rms_norm=True)(pair_input)
    k = hk.Linear(128, with_bias=False, name='linear_k')(x)
    q = hk.Linear(128, with_bias=False, name='linear_q')(x)
    v = hk.Linear(
        128,
        name='linear_v',
        w_init=hk.initializers.TruncatedNormal(stddev=1e-6),
    )(x)
    x = jnp.einsum(
        'bpqf,bpkf->bpqk',
        q,
        k,
        precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        preferred_element_type=jnp.float32,
    ) / math.sqrt(128)
    x = jax.nn.softmax(x, axis=-1)
    x = jnp.einsum(
        'bpqk,bpkf->bpqf',
        x,
        v,
        precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
    ).astype(q.dtype)
    return x


class SequenceToPairBlock(hk.Module):
  """Converts sequence representations to pairwise representations."""

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S D']
  ) -> Float[Array, 'B S//16 S//16 F']:
    x = layers.LayerNorm(rms_norm=True, name='norm_seq2pair')(
        layers.pool(x, by=16, reduce='mean')
    )
    batch_size, seq_len = x.shape[0], x.shape[1]
    q = hk.Linear(32 * 128, with_bias=False, name='linear_q')(x).reshape(
        batch_size, seq_len, 32, 128
    )
    k = hk.Linear(32 * 128, with_bias=False, name='linear_k')(x).reshape(
        batch_size, seq_len, 32, 128
    )

    relative_positions = jnp.arange(-seq_len, seq_len).astype(jnp.float32)
    pos_features = _central_mask_features(
        distances=jnp.abs(relative_positions),
        feature_size=32,
        seq_length=_MAX_RELATIVE_DISTANCE // 16,
    ).astype(x.dtype)
    pos_features = jnp.concatenate(
        [pos_features, jnp.sign(relative_positions)[..., None] * pos_features],
        axis=-1,
    ).astype(x.dtype)
    pos_encoding = hk.Linear(32 * 128, name='linear_pos_features')(
        pos_features
    ).reshape(2 * seq_len, 32, 128)

    b_init = hk.initializers.VarianceScaling(scale=2.0)
    q_bias = hk.get_parameter('q_r_bias', (1, 1, 32, 128), init=b_init).astype(
        x.dtype
    )
    k_bias = hk.get_parameter('k_r_bias', (1, 1, 32, 128), init=b_init).astype(
        x.dtype
    )

    rel_q_a = _shift(
        jnp.einsum('bqhc,phc->bhqp', q + q_bias, pos_encoding), seq_len, seq_len
    )
    rel_k_a = _shift(
        jnp.einsum('bkhc,phc->bhkp', k + k_bias, pos_encoding), seq_len, seq_len
    )

    a = jnp.einsum('bqhc,bkhc->bqkh', q, k)
    a += 0.5 * (
        einshape('bhqp->bqph', rel_q_a) + einshape('bhkp->bpkh', rel_k_a)
    )

    y_q = hk.Linear(128, with_bias=False, name='linear_y_q')(jax.nn.gelu(x))
    y_k = hk.Linear(128, with_bias=False, name='linear_y_k')(jax.nn.gelu(x))
    pair_act = (
        hk.Linear(128, name='linear_pair')(a)
        + y_q[:, :, None, :]
        + y_k[:, None, :, :]
    )
    return pair_act


class PairUpdateBlock(hk.Module):
  """Updates pairwise representations."""

  @typing.jaxtyped
  def __call__(
      self,
      sequence_input: Float[Array, 'B S C'],
      pair_input: Float[Array, 'B S//16 S//16 F'] | None,
  ):
    y = SequenceToPairBlock()(sequence_input)
    x = y if pair_input is None else (pair_input + y)
    x += RowAttentionBlock()(x)
    x += PairMLPBlock()(x)
    return x
