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
from alphagenome.data import track_data
from alphagenome.models import dna_output
from alphagenome.models import variant_scorers
from alphagenome_research.model.variant_scoring import center_mask
import anndata
import chex
import jax
import jax.numpy as jnp
import numpy as np


class CenterMaskVariantScorerTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          width=None,
          output_type=dna_output.OutputType.ATAC,
          variant=genome.Variant('chr1', 10, '', ''),
          expected_start=0,
          expected_end=2048,
      ),
      dict(
          width=None,
          output_type=dna_output.OutputType.ATAC,
          variant=genome.Variant('chr1', 2049, '', ''),
          expected_start=-1,
          expected_end=-1,
      ),
      dict(
          width=501,
          output_type=dna_output.OutputType.RNA_SEQ,
          variant=genome.Variant('chr1', 1, '', ''),
          expected_start=0,
          expected_end=252,
      ),
      dict(
          width=501,
          output_type=dna_output.OutputType.RNA_SEQ,
          variant=genome.Variant('chr1', 2048, '', ''),
          expected_start=1798,
          expected_end=2048,
      ),
  ])
  def test_masks_and_metadata(
      self,
      width: int | None,
      output_type: dna_output.OutputType,
      variant: genome.Variant,
      expected_start: int,
      expected_end: int,
  ):
    settings = variant_scorers.CenterMaskScorer(
        requested_output=output_type,
        aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
        width=width,
    )
    interval = genome.Interval('chr1', 0, 2048)
    (
        masks,
        metadata,
    ) = center_mask.CenterMaskVariantScorer().get_masks_and_metadata(
        interval,
        variant,
        settings=settings,
        track_metadata=dna_output.OutputMetadata(),
    )
    self.assertIsNone(metadata)

    chex.assert_shape(masks, (2048, 1))
    if expected_start >= 0:
      self.assertEqual(masks.argmax(axis=0).item(), expected_start)
    if expected_end >= 0:
      self.assertEqual(
          masks.shape[0] - masks[::-1].argmax(axis=0).item(), expected_end
      )

    expected_mask = np.zeros_like(masks)
    if expected_start != expected_end:
      expected_mask[expected_start:expected_end] = True
    np.testing.assert_array_equal(masks, expected_mask)

  def test_create_center_mask_uses_zero_based_variant_start(self):
    interval = genome.Interval('chr1', 100, 200)
    variant = genome.Variant('chr1', 151, '', '')

    mask = center_mask.create_center_mask(
        interval,
        variant,
        width=1,
        resolution=1,
    )

    expected = np.zeros((100, 1), dtype=bool)
    expected[50:51] = True
    np.testing.assert_array_equal(mask, expected)

  @parameterized.product(
      [
          dict(
              # (0 + 0 + 0) / 3 - (4 + 5 + 6) / 3 = -5
              aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
              expected_score=-5.0,
          ),
          dict(
              # max((0 + 0 + 0) / 3, (4 + 5 + 6) / 3) = 5
              aggregation_type=variant_scorers.AggregationType.ACTIVE_MEAN,
              expected_score=5.0,
          ),
          dict(
              # (0 + 0 + 0) - (4 + 5 + 6) = -15
              aggregation_type=variant_scorers.AggregationType.DIFF_SUM,
              expected_score=-15.0,
          ),
          dict(
              # max((0 + 0 + 0), (4 + 5 + 6)) = 15
              aggregation_type=variant_scorers.AggregationType.ACTIVE_SUM,
              expected_score=15.0,
          ),
          dict(
              # sqrt((0 - 4)**2 + (0 - 5)**2 + (0 - 6)**2) =~ 8.77
              aggregation_type=variant_scorers.AggregationType.L2_DIFF,
              expected_score=8.774964332580566,
          ),
          dict(
              # sqrt((log1p(0) - log1p(4))**2 + (log1p(0) - log1p(5))**2 +
              #      (log1p(0) - log1p(6))**2) =~ 3.096
              aggregation_type=variant_scorers.AggregationType.L2_DIFF_LOG1P,
              expected_score=3.096329875472752,
          ),
          dict(
              # log2(0 + 1) * 3 - log2(4 + 1) - log2(5 + 1) - log2(6 + 1)
              # =~ -7.71
              aggregation_type=variant_scorers.AggregationType.DIFF_SUM_LOG2,
              expected_score=-7.714245517666122,
          ),
          dict(
              # log2( 1 + (0 + 0 + 0)) - log2( 1 + (4 + 5 + 6)) = -4.0
              aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
              expected_score=-4.0,
          ),
      ],
      transfer_guard=['disallow', 'allow'],
  )
  def test_score_variant(
      self,
      aggregation_type: variant_scorers.AggregationType,
      expected_score: float,
      transfer_guard: str,
  ):
    settings = variant_scorers.CenterMaskScorer(
        requested_output=dna_output.OutputType.ATAC,
        aggregation_type=aggregation_type,
        width=None,
    )

    ref = jnp.arange(10, dtype=jnp.float32).repeat(100).reshape((10, 100))
    alt = jnp.zeros((10, 100), dtype=jnp.float32)
    mask = np.zeros((10, 1), dtype=bool)
    mask[4:7] = True
    mask = jnp.array(mask)

    scorer = center_mask.CenterMaskVariantScorer()

    with jax.transfer_guard(transfer_guard):
      scores = scorer.score_variant(
          {dna_output.OutputType.ATAC: ref},
          {dna_output.OutputType.ATAC: alt},
          masks=jax.device_put(mask),
          settings=settings,
      )['score']

    np.testing.assert_almost_equal(
        scores, np.full_like(scores, expected_score), decimal=6
    )

  def test_finalize_variant(self):
    scorer = center_mask.CenterMaskVariantScorer()
    scores = {'score': np.ones((10,), dtype=np.float32)}
    expected_scores = np.ones((8,), dtype=np.float32)
    track_metadata = dna_output.OutputMetadata(
        atac=track_data.TrackMetadata(
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
        settings=variant_scorers.CenterMaskScorer(
            requested_output=dna_output.OutputType.ATAC,
            width=None,
            aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
        ),
    )
    self.assertIsInstance(finalized_score, anndata.AnnData)
    np.testing.assert_array_equal(finalized_score.X[0], expected_scores)
    self.assertEqual(finalized_score.n_vars, expected_scores.shape[-1])


if __name__ == '__main__':
  absltest.main()
