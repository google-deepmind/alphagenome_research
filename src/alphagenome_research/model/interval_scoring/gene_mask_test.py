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
from alphagenome.models import dna_output
from alphagenome.models import interval_scorers
from alphagenome_research.model.interval_scoring import gene_mask
from alphagenome_research.model.variant_scoring import gene_mask_extractor
from alphagenome_research.model.variant_scoring import variant_scoring
import anndata
import jax.numpy as jnp
import numpy as np
import pandas as pd


def _load_gtf():
  return pd.DataFrame({
      'End': [306, 303, 305, 138, 15],
      'Start': [301, 302, 303, 0, 5],
      'Strand': ['+', '+', '+', '-', '-'],
      'transcript_id': ['', 'T1', 'T2', '', 'T3'],
      'gene_id': ['G1', 'G1', 'G1', 'G2', 'G2'],
      'Feature': [
          'gene',
          'transcript',
          'transcript',
          'gene',
          'transcript',
      ],
      # Fixed Fields
      'Chromosome': 'chr1',
      'Score': '.',
      'Frame': '.',
      'Source': 'ENSEMBL',
      'gene_type': 'protein_coding',
      'gene_name': 'GX_name',
      'transcript_type': 'protein_coding',
  })


class GeneIntervalScorerTest(parameterized.TestCase):

  @parameterized.product(
      aggregation_type=[
          interval_scorers.IntervalAggregationType.SUM,
          interval_scorers.IntervalAggregationType.MEAN,
      ],
      output_type=[dna_output.OutputType.RNA_SEQ, dna_output.OutputType.DNASE],
  )
  def test_gene_scorer_get_masks_and_metadata(
      self, aggregation_type, output_type
  ):
    gtf = _load_gtf()

    settings = interval_scorers.GeneMaskScorer(
        requested_output=output_type,
        aggregation_type=aggregation_type,
        width=501,
    )

    interval = genome.Interval('chr1', 0, 2048)
    interval_scorer = gene_mask.GeneIntervalScorer(
        gene_mask_extractor.GeneMaskExtractor(
            gtf=gtf,
            gene_mask_type=gene_mask_extractor.GeneMaskType.EXONS,
        )
    )

    track_metadata = dna_output.OutputMetadata(**{
        output_type.name.lower(): pd.DataFrame({
            'name': np.arange(9).astype(str),
            'strand': '.',
        })
    })
    masks, metadata = interval_scorer.get_masks_and_metadata(
        interval, settings=settings, track_metadata=track_metadata
    )
    self.assertIsInstance(metadata, pd.DataFrame)
    self.assertContainsSubset(
        ['strand', 'gene_name', 'gene_id', 'gene_type'], metadata.columns
    )

    self.assertEqual(masks.shape[0], interval.width)

  def test_gene_scorer_get_masks_and_metadata_negative_strand(self):
    gtf = _load_gtf()
    interval_scorer = gene_mask.GeneIntervalScorer(
        gene_mask_extractor.GeneMaskExtractor(
            gtf=gtf,
            gene_mask_type=gene_mask_extractor.GeneMaskType.EXONS,
        )
    )
    settings = interval_scorers.GeneMaskScorer(
        requested_output=dna_output.OutputType.RNA_SEQ,
        aggregation_type=interval_scorers.IntervalAggregationType.SUM,
        width=501,
    )
    with self.assertRaisesRegex(
        ValueError, 'IntervalScorers do not support negative strands'
    ):
      _ = interval_scorer.get_masks_and_metadata(
          genome.Interval('chr1', 0, 2048, strand='-'),
          settings=settings,
          track_metadata=dna_output.OutputMetadata(),
      )

  @parameterized.product(
      aggregation_type=[
          interval_scorers.IntervalAggregationType.SUM,
          interval_scorers.IntervalAggregationType.MEAN,
      ],
      output_type=[dna_output.OutputType.RNA_SEQ, dna_output.OutputType.DNASE],
  )
  def test_gene_scorer_score_interval(
      self,
      aggregation_type,
      output_type,
  ):
    # Score a simple example for an interval of length 10, consisting of 100
    # tracks, all of values 0..9 depending on the position. The mask
    # includes positions 4-6.
    tracks = jnp.arange(10, dtype=jnp.float32).repeat(100).reshape((10, 100))
    mask = np.zeros((10, 3), dtype=bool)
    mask[4:7] = True

    gtf = _load_gtf()
    settings = interval_scorers.GeneMaskScorer(
        requested_output=output_type,
        aggregation_type=aggregation_type,
        width=501,
    )
    interval_scorer = gene_mask.GeneIntervalScorer(
        gene_mask_extractor.GeneMaskExtractor(
            gtf=gtf, gene_mask_type=gene_mask_extractor.GeneMaskType.BODY
        ),
    )
    scores = interval_scorer.score_interval(
        {output_type: tracks}, settings=settings, masks=mask
    )

    expected_num_tracks = 100
    match aggregation_type:
      case interval_scorers.IntervalAggregationType.SUM:
        expected_score = 15.0
      case interval_scorers.IntervalAggregationType.MEAN:
        expected_score = 5.0
      case _:
        raise ValueError(f'Unknown aggregation type: {aggregation_type}')

    np.testing.assert_almost_equal(
        scores['score'][0],
        np.full((expected_num_tracks,), expected_score, dtype=jnp.float32),
    )

  @parameterized.product(
      center_mask_width=[501],
      output_type=[
          dna_output.OutputType.RNA_SEQ,
          dna_output.OutputType.CHIP_TF,
      ],
  )
  def test_gene_scorer_with_center_mask_width(
      self,
      center_mask_width,
      output_type,
  ):
    gtf = _load_gtf()
    interval = genome.Interval('chr1', 0, 1024)
    interval_scorer = gene_mask.GeneIntervalScorer(
        gene_mask_extractor.GeneMaskExtractor(
            gtf=gtf,
            gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
        ),
    )
    settings = interval_scorers.GeneMaskScorer(
        requested_output=output_type,
        aggregation_type=interval_scorers.IntervalAggregationType.MEAN,
        width=center_mask_width,
    )
    resolution = variant_scoring.get_resolution(output_type)
    num_bins = interval.width // resolution
    tracks = (
        jnp.arange(num_bins, dtype=jnp.float32).repeat(4).reshape((num_bins, 4))
    )

    track_metadata = dna_output.OutputMetadata(**{
        output_type.name.lower(): pd.DataFrame(
            {'name': np.arange(4).astype(str), 'strand': '.'}
        )
    })
    masks, metadata = interval_scorer.get_masks_and_metadata(
        interval, settings=settings, track_metadata=track_metadata
    )
    scores = interval_scorer.score_interval(
        {output_type: tracks}, settings=settings, masks=masks
    )

    expected_genes = ['G1']
    expected_mask = np.zeros([num_bins, len(expected_genes)], dtype=bool)
    expected_scores = np.full(
        (len(expected_genes), 4), np.nan, dtype=jnp.float32
    )

    if resolution == 1:
      expected_mask[301:306, 0] = True  # G1
      expected_scores[0] = 303.0
    else:
      expected_mask[2, 0] = True  # G1
      expected_scores[0] = 2.0
    np.testing.assert_array_equal(masks, expected_mask)
    np.testing.assert_almost_equal(scores['score'], expected_scores)
    np.testing.assert_array_equal(metadata['gene_id'].values, expected_genes)

  def test_gene_scorer_finalize(self):
    interval = genome.Interval('chr1', 0, 2048)
    track_metadata = dna_output.OutputMetadata(
        rna_seq=pd.DataFrame({
            'name': np.arange(9).astype(str),
            'strand': '.',
        })
    )
    expected_scores = np.full((1, 9), 15.0, dtype=jnp.float32)
    scores = {'score': expected_scores}

    gtf = _load_gtf()
    interval_scorer = gene_mask.GeneIntervalScorer(
        gene_mask_extractor.GeneMaskExtractor(
            gtf=gtf, gene_mask_type=gene_mask_extractor.GeneMaskType.BODY
        )
    )

    settings = interval_scorers.GeneMaskScorer(
        requested_output=dna_output.OutputType.RNA_SEQ,
        aggregation_type=interval_scorers.IntervalAggregationType.SUM,
        width=2001,
    )

    _, mask_metadata = interval_scorer.get_masks_and_metadata(
        interval, track_metadata=track_metadata, settings=settings
    )
    finalized_score = interval_scorer.finalize_interval(
        scores,
        track_metadata=track_metadata,
        mask_metadata=mask_metadata,
        settings=settings,
    )

    self.assertIsInstance(finalized_score, anndata.AnnData)
    np.testing.assert_array_almost_equal(finalized_score.X, expected_scores)


if __name__ == '__main__':
  absltest.main()
