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

"""Common layers."""

from collections.abc import Callable, Sequence
import functools
from typing import Any

from alphagenome import typing
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint: disable=g-importing-member, g-multiple-import


def gelu(x: jax.Array) -> jax.Array:
  """Gaussian Error Linear Unit activation function."""
  coef = jax.lax.convert_element_type(1.702, x.dtype)
  return jax.nn.sigmoid(coef * x) * x


def maybe_hk_remat(
    f: Callable[..., Any],
    remat: bool = True,
    *,
    static_argnames: Sequence[str] = (),
) -> Callable[..., Any]:
  """Optionally applies hk.remat to f, treating static_argnames as static."""
  if not remat:
    return f

  @functools.wraps(f)
  def wrapped(*args: Any, **kwargs: Any) -> Any:
    static_kwargs = {k: kwargs.pop(k) for k in static_argnames}
    return hk.remat(functools.partial(f, **static_kwargs))(*args, **kwargs)

  return wrapped


@typing.jaxtyped
def pool(
    x: Float[Array, '... S D'], by: int = 2, reduce: str = 'max'
) -> Float[Array, '... S/{by} D']:
  """Applies pooling to the sequence dimension of the input.

  Args:
    x: The input sequence, where the second to last dimension is the sequence
      dimension.
    by: The pooling window size.
    reduce: The pooling reduction method.

  Returns:
    The pooled sequence.
  Raises:
    NotImplementedError: If the reduce method is not supported.
  """
  if reduce == 'max':
    return hk.MaxPool(window_shape=(by, 1), strides=(by, 1), padding='SAME')(x)
  elif reduce in ['avg', 'mean']:
    return hk.AvgPool(window_shape=(by, 1), strides=(by, 1), padding='SAME')(x)
  else:
    raise NotImplementedError(f'Reduce method={reduce} unknown.')


class RMSBatchNorm(hk.Module):
  r"""Root Mean Square Batch Normalization.

  Normalization is applied to the last dimension of the input as
  `x -> x * scale / sqrt(var + epsilon) + offset`.
  The scale and offset are learned parameters. The variance is tracked
  as an exponential moving average.

  Variance is computed across the batch and sequence dimension.
  """

  def __init__(self, decay_rate: float = 0.9, name: str | None = None):
    super().__init__(name=name)
    self._decay_rate = decay_rate

  def __call__(
      self, x: Float[Array, '... D'], *, is_training: bool
  ) -> Float[Array, '... D']:
    original_dtype = x.dtype
    axis = tuple(range(x.ndim - 1))
    param_shape = (1,) * (x.ndim - 1) + (x.shape[-1],)

    variance_ema = hk.get_state(
        'var_ema', param_shape, dtype=jnp.float32, init=jnp.ones
    )

    if is_training:
      var = jax.lax.stop_gradient(
          jnp.mean(jnp.square(x), axis=axis, keepdims=True, dtype=jnp.float32)
      )
      hk.set_state(
          'var_ema',
          self._decay_rate * variance_ema + (1 - self._decay_rate) * var,
      )
    else:
      var = variance_ema

    scale = hk.get_parameter(
        'scale', param_shape, dtype=original_dtype, init=jnp.ones
    ).astype(original_dtype)
    offset = hk.get_parameter(
        'offset', param_shape, dtype=original_dtype, init=jnp.zeros
    ).astype(original_dtype)
    inv = scale * jax.lax.rsqrt(var + 1e-5).astype(original_dtype)
    return x * inv + offset


class LayerNorm(hk.Module):
  """Layer Normalization."""

  def __init__(
      self, rms_norm: bool = False, axis: int = -1, name: str | None = None
  ) -> None:
    """Initializes the LayerNorm module.

    Args:
      rms_norm: If False, the input is centered before computing the
        mean-squared for normalization. If True, the mean-squared is computed
        directly on the uncentered input.
      axis: The axis to apply the normalization to.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._rms_norm = rms_norm
    self._axis = axis

  def __call__(self, x: Float[Array, '... D']) -> Float[Array, '... D']:
    dtype = x.dtype
    scale = hk.get_parameter(
        'scale', (x.shape[-1],), dtype, init=jnp.ones
    ).astype(dtype)
    offset = hk.get_parameter(
        'offset', (x.shape[-1],), dtype, init=jnp.zeros
    ).astype(dtype)
    scale = jax.lax.broadcast_to_rank(scale, x.ndim)
    offset = jax.lax.broadcast_to_rank(offset, x.ndim)

    if not self._rms_norm:
      mean = jnp.mean(
          x, axis=self._axis, dtype=jnp.float32, keepdims=True
      ).astype(dtype)
      x = x - mean

    variance = jnp.mean(
        jnp.square(x), axis=self._axis, dtype=jnp.float32, keepdims=True
    )
    inv = scale * jax.lax.rsqrt(variance + 1e-5).astype(dtype)
    return inv * x + offset
