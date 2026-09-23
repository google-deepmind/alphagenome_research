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

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome_research.model import layers
import chex
import haiku as hk
import jax
import jax.numpy as jnp


def _get_pool_layer(by: int):
  """Helper function to create the pool layer."""

  def pool_fn(x):
    return layers.pool(x, by=by)

  pool_mod = hk.transform(pool_fn)
  return pool_mod


class LayersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._batch_size = 4
    self._sequence_length = 16
    self._num_channels = 8
    self._by = 2

  def test_pool_shape(self):
    """Tests that the pool layer returns the expected shape."""
    x = jnp.ones((self._batch_size, self._sequence_length, self._num_channels))
    pool_mod = _get_pool_layer(by=self._by)
    params = pool_mod.init(None, x)
    out = pool_mod.apply(params, None, x)
    chex.assert_shape(
        out,
        (
            self._batch_size,
            self._sequence_length // self._by,
            self._num_channels,
        ),
    )

  def test_pool_type_error(self):
    """Tests that the pool layer raises a type error with invalid input."""
    with self.subTest('wrong_dtype'):
      x = jnp.ones(
          (self._batch_size, self._sequence_length, self._num_channels),
          dtype=jnp.int32,  # Provide wrong type.
      )
      with self.assertRaises(TypeError):
        pool_mod = _get_pool_layer(by=self._by)
        params = pool_mod.init(None, x)
        pool_mod.apply(params, None, x)

  def test_maybe_hk_remat_respects_flag(self):
    """Tests that the function is returned unwrapped when remat is disabled."""
    fn = lambda x, *, is_training: x + 1 if is_training else x
    self.assertIs(
        layers.maybe_hk_remat(
            fn, remat=False, static_argnames=('is_training',)
        ),
        fn,
    )

  def test_maybe_hk_remat_static_argnames(self):
    """Tests that static_argnames can be used in Python control flow under remat."""

    def fn(x, *, scale_by_two: bool):
      return x * 2.0 if scale_by_two else x

    @hk.without_apply_rng
    @hk.transform
    def forward(x):
      rematted = layers.maybe_hk_remat(
          fn, remat=True, static_argnames=('scale_by_two',)
      )
      return jnp.sum(rematted(x, scale_by_two=True))

    x = jnp.ones((4,))
    params = forward.init(None, x)
    grad_x = jax.grad(lambda inp: forward.apply(params, inp))(x)
    chex.assert_trees_all_close(grad_x, jnp.full((4,), 2.0))


if __name__ == '__main__':
  absltest.main()
