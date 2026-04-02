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

"""Losses for AlphaGenome."""

from alphagenome import typing
import chex
from einshape import jax_einshape as einshape  # pylint: disable=g-importing-member
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree  # pylint: disable=g-importing-member, g-multiple-import


@typing.jaxtyped
def safe_masked_mean(
    x: Float[Array, '*dims'],
    mask: Bool[Array, '#*dims'] | None = None,
) -> Float[Array, '']:
  """Safe jnp.mean that handles completely masked arrays."""
  if mask is None:
    masked = x
    mask = jnp.ones_like(x)
  else:
    # We need to broadcast mask to compute correct mean.
    mask = jnp.broadcast_to(mask, x.shape)
    masked = x * mask

  return jnp.sum(jnp.asarray(masked), dtype=jnp.float32) / jnp.maximum(
      1.0, jnp.sum(mask, dtype=jnp.float32)
  )


@typing.jaxtyped
def poisson_loss(
    *,
    y_true: Float[Array, '*dims'],
    y_pred: Float[Array, '*dims'],
    mask: Bool[Array, '#*dims'],
) -> Float[Array, '']:
  """Poisson loss."""
  y_true = jnp.abs(y_true).astype(jnp.float32)
  y_pred = y_pred.astype(jnp.float32)
  y_pred_logits = jnp.log(y_pred + 1e-7)
  # Substract the minimum value such that loss is zero at optimal prediction.
  min_value = y_true - y_true * jnp.log(y_true + 1e-7)
  loss = (y_pred - y_true * y_pred_logits) - min_value
  return safe_masked_mean(loss, mask)


@typing.jaxtyped
def multinomial_loss(
    *,
    y_true: Float[Array, '... S C'],
    y_pred: Float[Array, '... S C'],
    mask: Bool[Array, '... 1 C'],
    multinomial_resolution: int,
    positional_weight: float,
) -> PyTree[Float[Array, '']]:
  """Returns sum of multinomial losses and Poison loss on total count.

  Args:
    y_true: Target values.
    y_pred: Model predictions.
    mask: Array of bools.
    multinomial_resolution: We split the input into sub-sequences and compute a
      separate multinomial loss over each sub-sequence.
    positional_weight: Weight of the positional loss.
  """
  chex.assert_equal_shape([y_true, y_pred])
  if y_pred.shape[-2] % multinomial_resolution != 0:
    raise ValueError(
        f'{y_pred.shape[-2]=} must be divisible by {multinomial_resolution=}.'
    )

  num_segments = y_pred.shape[-2] // multinomial_resolution

  # Remove the masked out bins from the totals sum.
  y_true = jnp.maximum(y_true, 0) * mask
  y_pred = y_pred * mask

  # Split sequence into n sub-sequences of size multinomial_resolution.
  y_true = einshape('...(ns)c->...nsc', y_true, n=num_segments)
  y_pred = einshape('...(ns)c->...nsc', y_pred, n=num_segments)

  total_pred = jnp.sum(y_pred, axis=-2, keepdims=True, dtype=jnp.float32)
  total_true = jnp.sum(y_true, axis=-2, keepdims=True, dtype=jnp.float32)
  mask = mask[..., None, :]  # broadcast over segments.

  loss_total_count = poisson_loss(
      y_true=total_true,
      y_pred=total_pred,
      mask=mask,
  )
  # Magnitude of poisson loss is linear with the number of bins involved in the
  # `total_count` prediction/target window. Normalization keeps overall loss
  # magnitude invariant to input `multinomial_resolution`.
  loss_total_count /= multinomial_resolution

  prob_predictions = y_pred.astype(jnp.float32) / (total_pred + 1e-7)
  loss_positional = -y_true * jnp.log(prob_predictions + 1e-7)
  loss_positional = safe_masked_mean(loss_positional, mask=mask)

  return {
      'loss': loss_total_count + positional_weight * loss_positional,
      'loss_total': loss_total_count,
      'loss_positional': loss_positional,
      'max_sum_preds': jnp.max(total_pred),
      'max_preds': jnp.max(y_pred),
      'max_targets': jnp.max(y_true).astype(jnp.float32),
  }


def mse(
    y_pred: Float[Array, '*dims'],
    y_true: Float[Array, '*dims'],
    mask: Bool[Array, '#*dims'],
) -> Float[Array, '']:
  """Mean squared error."""
  return safe_masked_mean(jnp.square(y_pred - y_true), mask)


@typing.jaxtyped
def cross_entropy_loss_from_logits(
    *,
    y_pred_logits: Float[Array, '*dims'],
    y_true: Float[Array, '*dims'],
    mask: Bool[Array, '#*dims'],
    axis: int,
) -> Float[Array, '']:
  """Cross-entropy loss from logits."""
  log_softmax_preds = jax.nn.log_softmax(
      y_pred_logits.astype(jnp.float32), axis=axis
  )
  loss = -jnp.sum(y_true.astype(jnp.float32) * log_softmax_preds, axis=axis)
  mask = jnp.any(mask, axis=axis)
  return safe_masked_mean(loss, mask)


@typing.jaxtyped
def binary_crossentropy_from_logits(
    *,
    y_pred: Float[Array, '*dims'],
    y_true: Float[Array, '*dims'],
    mask: Bool[Array, '#*dims'],
) -> Float[Array, '']:
  """Binary cross-entropy loss from sigmoid logits."""
  loss = (
      jnp.maximum(y_pred, 0)
      - y_pred * y_true
      + jnp.log1p(jnp.exp(-jnp.abs(y_pred)))
  )
  return safe_masked_mean(loss, mask)


@typing.jaxtyped
def cross_entropy_loss(
    *,
    y_true: Float[Array, '*dims'],
    y_pred: Float[Array, '*dims'],
    mask: Bool[Array, '#*dims'],
    axis: int,
    eps: float = 1e-7,
) -> Float[Array, '']:
  """Cross entropy loss on counts."""
  mask = jnp.broadcast_to(mask, y_true.shape)
  chex.assert_equal_shape([y_true, y_pred, mask])
  y_true = jnp.where(mask, y_true.astype(jnp.float32), 0)
  p_true = y_true / jnp.maximum(y_true.sum(axis=axis, keepdims=True), eps)

  log_normalizer = jnp.log((jnp.where(mask, y_pred, 0) + eps).sum(axis=axis))
  log_likelihood = (p_true * jnp.log(y_pred + eps)).sum(axis=axis)
  log_loss = log_normalizer - log_likelihood
  return safe_masked_mean(log_loss, mask.any(axis=axis))
