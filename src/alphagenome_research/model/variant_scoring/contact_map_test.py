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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import genome
from alphagenome.data import track_data
from alphagenome.models import dna_output
from alphagenome.models import variant_scorers
from alphagenome_research.model.variant_scoring import center_mask
from alphagenome_research.model.variant_scoring import contact_map
import anndata
import jax
import jax.numpy as jnp
import numpy as np


def _interval_variant_pairs(interval_width: int, bin_width: int):
  return {
      'interval': genome.Interval('chr1', 0, interval_width),
      'variant': genome.Variant(
          chromosome='chr1',
          position=interval_width // 2,
          reference_bases='A',
          alternate_bases='C',
      ),
      'variant_wrong_chr': genome.Variant(
          chromosome='chrX',
          position=interval_width,
          reference_bases='A',
          alternate_bases='C',
      ),
      'variant_indel': genome.Variant(
          chromosome='chr1',
          position=interval_width // 2,
          reference_bases='A',
          alternate_bases='C' * bin_width * 10,
      ),
  }


class ContactMapVariantScorerTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='snv',
          interval=genome.Interval('chr1', 0, 196608),
          variant=genome.Variant('chr1', 196608 // 2, 'A', 'C'),
      ),
      dict(
          testcase_name='indel_variant',
          interval=genome.Interval('chr1', 0, 196608),
          variant=genome.Variant('chr1', 196608 // 2, 'A', 'C' * 2048 * 10),
      ),
  ])
  def test_masks_and_metadata(
      self, interval: genome.Interval, variant: genome.Variant
  ):
    scorer = contact_map.ContactMapScorer()

    mask, mask_metadata = scorer.get_masks_and_metadata(
        interval,
        variant,
        settings=variant_scorers.ContactMapScorer(),
        track_metadata=dna_output.OutputMetadata(),
    )
    self.assertIsNone(mask_metadata)
    self.assertEqual(mask.sum(), 1)

  def test_no_variant_overlap_errors(self):
    scorer = contact_map.ContactMapScorer()
    with self.assertRaisesRegex(
        ValueError, 'The variant does not affect any positions'
    ):
      scorer.get_masks_and_metadata(
          genome.Interval('chr1', 0, 196608),
          genome.Variant('chrX', 196608, 'A', 'C'),
          settings=variant_scorers.ContactMapScorer(),
          track_metadata=dna_output.OutputMetadata(),
      )

  def test_multiple_bins_raises_error(self):
    scorer = contact_map.ContactMapScorer()
    mock_mask = np.array([False, True, True])  # More than one True.

    # Mock a call to create_center_mask to return a badly formed center mask.
    with mock.patch.object(
        center_mask, 'create_center_mask', autospec=True, return_value=mock_mask
    ):
      with self.assertRaisesRegex(
          ValueError, 'only accepts input variants that affect one bin'
      ):
        scorer.get_masks_and_metadata(
            genome.Interval('chr1', 0, 1644),
            genome.Variant('chr1', 822, 'A', 'C'),
            settings=variant_scorers.ContactMapScorer(),
            track_metadata=dna_output.OutputMetadata(),
        )

  @parameterized.product(transfer_guard=['disallow', 'allow'])
  def test_score_variant(self, transfer_guard: str):
    symmetric_diff = jnp.array(
        [[1, 2, 1], [2, -3, 4], [1, 4, 5]], dtype=jnp.float32
    )

    # Rescaling the second channel to make things slightly different.
    second_channel_scale = 0.5

    # Concatenate on channel dimension to represent two tissues.
    alt_contact_map = jnp.concat(
        [
            symmetric_diff[:, :, jnp.newaxis],
            symmetric_diff[:, :, jnp.newaxis] * second_channel_scale,
        ],
        axis=-1,
    )

    # Ref alt.
    ref = {dna_output.OutputType.CONTACT_MAPS: jnp.zeros_like(alt_contact_map)}
    alt = {dna_output.OutputType.CONTACT_MAPS: alt_contact_map}

    # Simple center mask.
    mask = jnp.array([False, True, False])[:, None]

    # Score center column of both channels of the diff.
    variant_scores = (
        jnp.array([1, second_channel_scale])  # Channel scales.
        * jnp.abs(symmetric_diff[:, 1]).mean()  # Mean abs diff.
    )
    scorer = contact_map.ContactMapScorer()
    with jax.transfer_guard(transfer_guard):
      scores = scorer.score_variant(
          ref, alt, masks=mask, settings=variant_scorers.ContactMapScorer()
      )
    np.testing.assert_array_equal(scores['score'], variant_scores)

  def test_finalize_variant(self):
    scorer = contact_map.ContactMapScorer()
    scores = {'score': np.ones((10,), dtype=np.float32)}
    expected_scores = np.ones((8,), dtype=np.float32)
    track_metadata = dna_output.OutputMetadata(
        contact_maps=track_data.TrackMetadata(
            dict(
                name=np.arange(8).astype(str),
                strand='.',
            )
        )
    )
    finalized_score = scorer.finalize_variant(
        scores,
        track_metadata=track_metadata,
        mask_metadata=None,
        settings=variant_scorers.ContactMapScorer(),
    )
    self.assertIsInstance(finalized_score, anndata.AnnData)
    np.testing.assert_array_equal(finalized_score.X[0], expected_scores)
    self.assertEqual(finalized_score.n_vars, expected_scores.shape[-1])


if __name__ == '__main__':
  absltest.main()
