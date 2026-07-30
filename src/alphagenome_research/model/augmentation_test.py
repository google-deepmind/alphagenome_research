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

from collections.abc import Sequence
import functools

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.models import dna_output
from alphagenome_research.model import augmentation
import chex
import jax
from jax import numpy as jnp
from jaxtyping import PyTree  # pylint: disable=g-importing-member
import ml_dtypes
import numpy as np


class AugmentationTest(chex.TestCase, parameterized.TestCase):

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters(
      dict(
          output_type=dna_output.OutputType.SPLICE_JUNCTIONS,
          predictions={
              'predictions': np.array(
                  [[
                      [[0, 1, 2, 3], [4, 5, 6, 7]],
                      [[8, 9, 10, 11], [12, 13, 14, 15]],
                  ]],
                  dtype=np.float32,
              ),
              'splice_site_positions': np.array(
                  [[[0, 1], [-1, 3], [4, -1], [6, 7]]],
                  dtype=np.int32,
              ),
          },
          strand_reindexing=np.array([3, 2, 1, 0], dtype=np.int32),
          expected={
              'predictions': np.array(
                  [[
                      [[3, 2, 1, 0], [7, 6, 5, 4]],
                      [[11, 10, 9, 8], [15, 14, 13, 12]],
                  ]],
                  dtype=np.float32,
              ),
              'splice_site_positions': np.array(
                  [[[-1, -1], [-3, -4], [3, 2], [-1, 0]]],
                  dtype=np.int32,
              ),
          },
          sequence_length=4,
      ),
      dict(
          output_type=dna_output.OutputType.CONTACT_MAPS,
          predictions=np.array(
              [[
                  [[0, 1], [2, 3], [4, 5], [6, 7]],
                  [[8, 9], [10, 11], [12, 13], [14, 15]],
                  [[16, 17], [18, 19], [20, 21], [22, 23]],
                  [[24, 25], [26, 27], [28, 29], [30, 31]],
              ]],
              dtype=ml_dtypes.bfloat16,
          ),
          strand_reindexing=None,
          expected=np.array(
              [[
                  [[30, 31], [28, 29], [26, 27], [24, 25]],
                  [[22, 23], [20, 21], [18, 19], [16, 17]],
                  [[14, 15], [12, 13], [10, 11], [8, 9]],
                  [[6, 7], [4, 5], [2, 3], [0, 1]],
              ]],
              dtype=ml_dtypes.bfloat16,
          ),
          sequence_length=-1,  # Unused.
      ),
      dict(
          output_type=dna_output.OutputType.RNA_SEQ,
          predictions=np.array(
              [[[0, 1], [2, 3], [4, 5], [6, 7]]],
              dtype=ml_dtypes.bfloat16,
          ),
          strand_reindexing=np.array([1, 0], dtype=np.int32),
          expected=np.array(
              [[[7, 6], [5, 4], [3, 2], [1, 0]]],
              dtype=ml_dtypes.bfloat16,
          ),
          sequence_length=-1,  # Unused.
      ),
      dict(
          output_type=dna_output.OutputType.DNASE,
          predictions=np.array(
              [[[0, 1], [2, 3], [4, 5], [6, 7]]],
              dtype=ml_dtypes.bfloat16,
          ),
          strand_reindexing=np.array([1, 0], dtype=np.int32),
          expected=np.array(
              [[[0, 0], [7, 6], [5, 4], [3, 2]]],
              dtype=ml_dtypes.bfloat16,
          ),
          sequence_length=-1,  # Unused.
      ),
  )
  def test_reverse_complement_output_type(
      self,
      output_type: dna_output.OutputType,
      predictions: PyTree[np.ndarray],
      strand_reindexing: np.ndarray,
      expected: PyTree[np.ndarray],
      sequence_length: int,
  ):
    output = self.variant(
        augmentation.reverse_complement_output_type,
        static_argnames=['output_type'],
    )(
        jax.tree.map(jnp.asarray, predictions),
        output_type=output_type,
        strand_reindexing=jnp.asarray(strand_reindexing)
        if strand_reindexing is not None
        else None,
        sequence_length=sequence_length,
    )
    jax.tree.map(np.testing.assert_array_equal, output, expected)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.product(
      mask=[[True, False], [False, False], [True, True]],
  )
  def test_reverse_complement(self, mask: Sequence[bool]):
    sequence_length = 64
    example: dict[dna_output.OutputType, PyTree[np.ndarray]] = {
        dna_output.OutputType.ATAC: (
            jnp.arange(sequence_length * 2, dtype=jnp.bfloat16).reshape(
                1, sequence_length, 2
            )
        ),
        dna_output.OutputType.DNASE: (
            jnp.arange(sequence_length * 2, dtype=jnp.bfloat16).reshape(
                1, sequence_length, 2
            )
        ),
        dna_output.OutputType.SPLICE_JUNCTIONS: {
            'predictions': (
                jnp.arange(32, dtype=jnp.bfloat16).reshape(1, 4, 4, 2)
            ),
            'splice_site_positions': (
                jnp.arange(16, dtype=jnp.int32).reshape(1, 4, 4)
            ),
        },
        dna_output.OutputType.CONTACT_MAPS: (
            jnp.arange(400, dtype=jnp.bfloat16).reshape(1, 10, 10, 4)
        ),
    }
    strand_reindexing = {
        dna_output.OutputType.ATAC: jnp.array([1, 0], dtype=jnp.int32),
        dna_output.OutputType.DNASE: jnp.array([1, 0], dtype=jnp.int32),
        dna_output.OutputType.SPLICE_JUNCTIONS: jnp.array(
            [0, 1], dtype=jnp.int32
        ),
    }
    splice_site_positions_reversed = (
        sequence_length
        - 1
        - example[dna_output.OutputType.SPLICE_JUNCTIONS][
            'splice_site_positions'
        ]
    )
    expected_reversed = {
        dna_output.OutputType.ATAC: example[dna_output.OutputType.ATAC][
            :, ::-1, strand_reindexing[dna_output.OutputType.ATAC]
        ],
        dna_output.OutputType.DNASE: jnp.pad(
            example[dna_output.OutputType.DNASE][
                :, :0:-1, strand_reindexing[dna_output.OutputType.DNASE]
            ],
            ((0, 0), (1, 0), (0, 0)),
        ),
        dna_output.OutputType.SPLICE_JUNCTIONS: {
            'predictions': example[dna_output.OutputType.SPLICE_JUNCTIONS][
                'predictions'
            ],
            'splice_site_positions': splice_site_positions_reversed[
                :,
                (2, 3, 0, 1),
            ],
        },
        dna_output.OutputType.CONTACT_MAPS: example[
            dna_output.OutputType.CONTACT_MAPS
        ][:, ::-1, ::-1],
    }
    batched_example = jax.tree.map(
        lambda x: jnp.repeat(x, len(mask), axis=0), example
    )
    expected = jax.tree.map(
        lambda *x: jnp.concatenate(x, axis=0),
        *[expected_reversed if m else example for m in mask],
    )

    result = self.variant(
        augmentation.reverse_complement,
        static_argnames=['sequence_length'],
    )(
        batched_example,
        jnp.asarray(mask, dtype=bool),
        strand_reindexing=strand_reindexing,
        sequence_length=sequence_length,
    )

    for output_type in dna_output.OutputType:
      if (prediction := result.get(output_type)) is not None:
        jax.tree.map(
            functools.partial(
                np.testing.assert_array_equal, err_msg=f'{output_type=}'
            ),
            prediction,
            expected.get(output_type),
        )


if __name__ == '__main__':
  absltest.main()
