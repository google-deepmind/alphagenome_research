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
from alphagenome_research.model import indel_stitch_utils
from alphagenome_research.model import one_hot_encoder
import jax.numpy as jnp
import numpy as np


class IndelStitchUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.encoder = one_hot_encoder.DNAOneHotEncoder()

  def _encode(self, seq: str) -> np.ndarray:
    return self.encoder.encode(seq)

  @parameterized.named_parameters(
      dict(
          testcase_name='within_bin_no_shift',
          stitch_idx=18,
          start_idx=16,
          bin_size=4,
          expected_stitch_bin=4,
          expected_shift_bins=0,
      ),
      dict(
          testcase_name='crosses_bin_boundary_shift',
          stitch_idx=22,
          start_idx=16,
          bin_size=4,
          expected_stitch_bin=5,
          expected_shift_bins=1,
      ),
      dict(
          testcase_name='at_bin_end_plus_one',
          stitch_idx=21,
          start_idx=16,
          bin_size=4,
          expected_stitch_bin=5,
          expected_shift_bins=1,
      ),
      dict(
          testcase_name='at_bin_end_no_shift',
          stitch_idx=23,
          start_idx=16,
          bin_size=4,
          expected_stitch_bin=5,
          expected_shift_bins=1,
      ),
      dict(
          testcase_name='large_bin_within_boundary',
          stitch_idx=132,
          start_idx=128,
          bin_size=128,
          expected_stitch_bin=1,
          expected_shift_bins=0,
      ),
  )
  def test_compute_stitch_params(
      self,
      stitch_idx,
      start_idx,
      bin_size,
      expected_stitch_bin,
      expected_shift_bins,
  ):
    stitch_bin, shift_bins = (
        indel_stitch_utils.compute_stitch_params_coarse_resolution(
            stitch_idx=jnp.array(stitch_idx),
            start_idx=jnp.array(start_idx),
            bin_size=bin_size,
        )
    )
    self.assertEqual(int(stitch_bin), expected_stitch_bin)
    self.assertEqual(int(shift_bins), expected_shift_bins)

  @parameterized.named_parameters([
      dict(
          testcase_name='no_shift',
          left_seq='ACGTACGTACGT',
          right_seq='ACGTACGTACGT',
          left_shift=0,
          right_shift=0,
          stitch_index=12,
          expected='ACGTACGTACGT',
      ),
      dict(
          testcase_name='shift_2bp',
          left_seq='AAAACCCC',
          right_seq='TTTTGGGG',
          left_shift=0,
          right_shift=2,
          stitch_index=4,
          expected='AAAATTGG',
      ),
      dict(
          testcase_name='shift_1bp',
          left_seq='AAAACCCC',
          right_seq='GGGGTTTT',
          left_shift=0,
          right_shift=1,
          stitch_index=4,
          expected='AAAAGTTT',
      ),
      dict(
          testcase_name='shift_5bp',
          left_seq='AAAACCCCCC',
          right_seq='GGGGTTTTTT',
          left_shift=0,
          right_shift=5,
          stitch_index=6,
          expected='AAAACCGGGT',
      ),
      dict(
          testcase_name='shift_3bp',
          left_seq='AAAACCCCCC',
          right_seq='GGGGTTTTTT',
          left_shift=0,
          right_shift=3,
          stitch_index=7,
          expected='AAAACCCTTT',
      ),
      dict(
          testcase_name='neg_strand_shift_2bp',
          left_seq='AAAACCCC',
          right_seq='TTTTGGGG',
          left_shift=2,
          right_shift=0,
          stitch_index=4,
          expected='AACCGGGG',
      ),
      dict(
          testcase_name='neg_strand_shift_2bp_v2',
          left_seq='TTTTGGGG',
          right_seq='AAAACCCC',
          left_shift=2,
          right_shift=0,
          stitch_index=4,
          expected='TTGGCCCC',
      ),
      dict(
          testcase_name='neg_strand_shift_3bp',
          left_seq='GGGGTTTTTT',
          right_seq='AAAACCCCCC',
          left_shift=3,
          right_shift=0,
          stitch_index=5,
          expected='GTTTTCCCCC',
      ),
  ])
  def test_stitch_single_base_res(
      self, left_seq, right_seq, left_shift, right_shift, stitch_index, expected
  ):
    left = self._encode(left_seq)
    right = self._encode(right_seq)
    result = indel_stitch_utils.stitch_sequence_base_resolution(
        left=jnp.array([left]),
        right=jnp.array([right]),
        left_shift=jnp.array([[left_shift]]),
        right_shift=jnp.array([[right_shift]]),
        stitch_index=jnp.array([[stitch_index]]),
    )
    np.testing.assert_array_equal(
        np.asarray(result[0]),
        self._encode(expected),
    )

  def test_stitch_single_base_res_batch(self):
    left1 = self._encode('GGGGTTTT')
    right1 = self._encode('CCCCAAAA')
    result = indel_stitch_utils.stitch_sequence_base_resolution(
        left=jnp.array([left1]),
        right=jnp.array([right1]),
        left_shift=jnp.array([[0]]),
        right_shift=jnp.array([[1]]),
        stitch_index=jnp.array([[5]]),
    )
    np.testing.assert_array_equal(
        np.asarray(result[0]),
        self._encode('GGGGTAAA'),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_shift_within_bin',
          stitch_idx=17,
          start_idx=16,
          bin_size=4,
          n_bins=8,
          expected_stitch_bin=4,
          expected_shift_bins=0,
      ),
      dict(
          testcase_name='shift_crosses_bin_boundary',
          stitch_idx=22,
          start_idx=16,
          bin_size=4,
          n_bins=9,
          expected_stitch_bin=5,
          expected_shift_bins=1,
      ),
      dict(
          testcase_name='no_shift_at_bin_midpoint',
          stitch_idx=20,
          start_idx=16,
          bin_size=4,
          n_bins=8,
          expected_stitch_bin=4,
          expected_shift_bins=0,
      ),
  )
  def test_stitch_single_res_bin_selection(
      self,
      stitch_idx,
      start_idx,
      bin_size,
      n_bins,
      expected_stitch_bin,
      expected_shift_bins,
  ):
    dim = 4
    left = np.zeros((n_bins, dim), dtype=np.float32)
    for i in range(n_bins):
      left[i, :] = i
    right = np.zeros((n_bins, dim), dtype=np.float32)
    for i in range(n_bins):
      right[i, :] = 100 + i

    result = indel_stitch_utils.stitch_sequence_coarse_resolution(
        left=jnp.array([left]),
        right=jnp.array([right]),
        left_shift=jnp.zeros((1, 1), dtype=jnp.int32),
        right_shift=jnp.ones((1, 1), dtype=jnp.int32),
        stitch_index=jnp.array([[stitch_idx]]),
        indel_start_index=jnp.array([[start_idx]]),
        bin_size=bin_size,
    )
    result = np.asarray(result[0])

    stitch_bin, shift_bins = (
        indel_stitch_utils.compute_stitch_params_coarse_resolution(
            stitch_idx=jnp.array(stitch_idx),
            start_idx=jnp.array(start_idx),
            bin_size=bin_size,
        )
    )
    self.assertEqual(int(stitch_bin), expected_stitch_bin)
    self.assertEqual(int(shift_bins), expected_shift_bins)

    for i in range(min(expected_stitch_bin, n_bins)):
      np.testing.assert_array_equal(result[i], left[i])

    for i in range(expected_stitch_bin, n_bins):
      expected_right_idx = (i - expected_shift_bins) % n_bins
      np.testing.assert_array_equal(result[i], right[expected_right_idx])

  def test_stitch_2d_cm_no_shift(self):
    n_bins = 8
    dim = 2
    left = np.arange(n_bins * n_bins * dim, dtype=np.float32).reshape(
        n_bins, n_bins, dim
    )
    right = left + 1000.0

    stitch_bin = 4
    result = indel_stitch_utils.stitch_2d_contact_maps(
        left=jnp.array([left]),
        right=jnp.array([right]),
        left_shift=jnp.zeros((1, 1), dtype=jnp.int32),
        right_shift=jnp.zeros((1, 1), dtype=jnp.int32),
        stitch_index=jnp.array([[20]]),
        indel_start_index=jnp.array([[16]]),
        bin_size=4,
    )
    result = np.asarray(result[0])

    # 2D mask: right used only where both i >= stitch_bin AND j >= stitch_bin
    for i in range(n_bins):
      for j in range(n_bins):
        if i >= stitch_bin and j >= stitch_bin:
          np.testing.assert_array_equal(result[i, j], right[i, j])
        else:
          np.testing.assert_array_equal(result[i, j], left[i, j])

  def test_stitch_2d_cm_with_shift(self):
    n_bins = 8
    dim = 1
    left = np.zeros((n_bins, n_bins, dim), dtype=np.float32)
    right = np.ones((n_bins, n_bins, dim), dtype=np.float32)

    for i in range(n_bins):
      for j in range(n_bins):
        right[i, j, 0] = i * n_bins + j

    result = indel_stitch_utils.stitch_2d_contact_maps(
        left=jnp.array([left]),
        right=jnp.array([right]),
        left_shift=jnp.zeros((1, 1), dtype=jnp.int32),
        right_shift=jnp.ones((1, 1), dtype=jnp.int32),
        stitch_index=jnp.array([[22]]),
        indel_start_index=jnp.array([[16]]),
        bin_size=4,
    )
    result = np.asarray(result[0])
    right_rolled = np.roll(np.roll(right, 1, axis=0), 1, axis=1)

    for i in range(n_bins):
      for j in range(n_bins):
        if i >= 5 and j >= 5:
          np.testing.assert_array_equal(result[i, j], right_rolled[i, j])
        else:
          np.testing.assert_array_equal(result[i, j], left[i, j])

  def test_stitch_single_res_bin_selection_negative_strand(self):
    n_bins = 9
    dim = 4
    left = np.zeros((n_bins, dim), dtype=np.float32)
    for i in range(n_bins):
      left[i, :] = i
    right = np.zeros((n_bins, dim), dtype=np.float32)
    for i in range(n_bins):
      right[i, :] = 100 + i

    result = indel_stitch_utils.stitch_sequence_coarse_resolution(
        left=jnp.array([left]),
        right=jnp.array([right]),
        left_shift=jnp.ones((1, 1), dtype=jnp.int32),
        right_shift=jnp.zeros((1, 1), dtype=jnp.int32),
        stitch_index=jnp.array([[22]]),
        indel_start_index=jnp.array([[16]]),
        bin_size=4,
    )
    result = np.asarray(result[0])

    expected_stitch_bin = 5
    expected_shift_bins = 1

    for i in range(expected_stitch_bin):
      np.testing.assert_array_equal(
          result[i], left[(i + expected_shift_bins) % n_bins]
      )

    for i in range(expected_stitch_bin, n_bins):
      np.testing.assert_array_equal(result[i], right[i])

  def test_stitch_2d_cm_with_shift_negative_strand(self):
    n_bins = 8
    dim = 1
    left = np.zeros((n_bins, n_bins, dim), dtype=np.float32)
    right = np.ones((n_bins, n_bins, dim), dtype=np.float32)

    for i in range(n_bins):
      for j in range(n_bins):
        left[i, j, 0] = i * n_bins + j

    result = indel_stitch_utils.stitch_2d_contact_maps(
        left=jnp.array([left]),
        right=jnp.array([right]),
        left_shift=jnp.ones((1, 1), dtype=jnp.int32),
        right_shift=jnp.zeros((1, 1), dtype=jnp.int32),
        stitch_index=jnp.array([[22]]),
        indel_start_index=jnp.array([[16]]),
        bin_size=4,
    )
    result = np.asarray(result[0])
    left_rolled = np.roll(np.roll(left, -1, axis=0), -1, axis=1)

    stitch_bin = 5
    for i in range(n_bins):
      for j in range(n_bins):
        if i >= stitch_bin and j >= stitch_bin:
          np.testing.assert_array_equal(result[i, j], right[i, j])
        else:
          np.testing.assert_array_equal(result[i, j], left_rolled[i, j])


if __name__ == '__main__':
  absltest.main()
