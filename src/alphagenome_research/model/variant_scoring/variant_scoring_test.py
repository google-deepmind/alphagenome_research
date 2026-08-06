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
from alphagenome.data import genome
from alphagenome_research.model.variant_scoring import variant_scoring
import chex
import numpy as np


class AlignAlternateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='snv',
          variant=genome.Variant('chr1', 2, 'C', 'T'),
          expected=np.arange(10, dtype=np.float32),
      ),
      dict(
          testcase_name='snv_negative_strand',
          variant=genome.Variant('chr1', 5, 'A', 'T'),
          expected=np.arange(10, dtype=np.float32),
          strand='-',
      ),
      dict(
          testcase_name='insertion',
          variant=genome.Variant('chr1', 2, 'C', 'TGA'),
          expected=np.array([0, 3, 4, 5, 6, 7, 8, 9, 0, 0], dtype=np.float32),
      ),
      dict(
          testcase_name='insertion_negative_strand',
          variant=genome.Variant('chr1', 2, 'C', 'TGA'),
          expected=np.array([0, 0, 0, 1, 2, 3, 4, 5, 8, 9], dtype=np.float32),
          strand='-',
      ),
      dict(
          testcase_name='multi_insertion',
          variant=genome.Variant('chr1', 2, 'CC', 'CCTGA'),
          expected=np.array([0, 1, 5, 6, 7, 8, 9, 0, 0, 0], dtype=np.float32),
      ),
      dict(
          testcase_name='multi_insertion_negative_strand',
          variant=genome.Variant('chr1', 2, 'CC', 'CCTGA'),
          expected=np.array([0, 0, 0, 0, 1, 2, 3, 7, 8, 9], dtype=np.float32),
          strand='-',
      ),
      dict(
          testcase_name='multi_insertion_long_insertion',
          variant=genome.Variant('chr1', 5, 'A', 'AAAAAA'),
          expected=np.array([0, 1, 2, 3, 9, 0, 0, 0, 0, 0], dtype=np.float32),
      ),
      dict(
          testcase_name='multi_insertion_long_insertion_negative_strand',
          variant=genome.Variant('chr1', 5, 'A', 'AAAAAA'),
          expected=np.array([0, 0, 0, 0, 0, 5, 6, 7, 8, 9], dtype=np.float32),
          strand='-',
      ),
      dict(
          testcase_name='deletion',
          variant=genome.Variant('chr1', 2, 'CGA', 'C'),
          expected=np.array([0, 1, 0, 0, 2, 3, 4, 5, 6, 7], dtype=np.float32),
      ),
      dict(
          testcase_name='deletion_negative_strand',
          variant=genome.Variant('chr1', 2, 'CGA', 'C'),
          expected=np.array([2, 3, 4, 5, 6, 7, 0, 0, 8, 9], dtype=np.float32),
          strand='-',
      ),
      dict(
          testcase_name='multi_deletion',
          variant=genome.Variant('chr1', 2, 'CCCGA', 'CCC'),
          expected=np.array([0, 1, 2, 3, 0, 0, 4, 5, 6, 7], dtype=np.float32),
      ),
      dict(
          testcase_name='multi_deletion_negative_strand',
          variant=genome.Variant('chr1', 2, 'CCCGA', 'CCC'),
          expected=np.array([2, 3, 4, 5, 0, 0, 6, 7, 8, 9], dtype=np.float32),
          strand='-',
      ),
  )
  def test_align_alternate(self, variant, expected, strand='.'):
    interval = genome.Interval('chr1', 0, 10, strand)
    alt = np.arange(10, dtype=np.float32).reshape(-1, 1)
    indel_mask = variant_scoring.IndelMask.from_variant(variant, interval)
    # pyrefly: ignore[bad-argument-type]
    aligned_alt = variant_scoring.align_alternate(alt, indel_mask)
    chex.assert_shape(aligned_alt, (10, 1))
    np.testing.assert_array_equal(aligned_alt, expected.reshape(-1, 1))


if __name__ == '__main__':
  absltest.main()
