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
from alphagenome.models import variant_scorers
from alphagenome_research.model.variant_scoring import splice_junction
import anndata
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def _create_fake_gtf():
  # Two genes (G1, G2).
  # G1 has 3 transcripts (T1, T2, T3). G2 has 1 transcript (T4). The T4
  # transcript is further annotated as consisting of 2 exons.
  gtf = pd.DataFrame({
      'End': [200, 200, 200, 200, 108, 108, 40, 108],
      'Start': [101, 101, 102, 103, 0, 0, 0, 80],
      'Strand': ['+', '+', '+', '+', '-', '-', '-', '-'],
      'transcript_id': ['', 'T1', 'T2', 'T3', '', 'T4', 'T4', 'T4'],
      'transcript_type': [
          '',
          'protein_coding',
          'protein_coding',
          'protein_coding',
          '',
          'protein_coding',
          'protein_coding',
          'protein_coding',
      ],
      'gene_id': ['G1', 'G1', 'G1', 'G1', 'G2', 'G2', 'G2', 'G2'],
      'gene_name': [
          'gene_1_name',
          'gene_2_name',
          'gene_1_name',
          'gene_1_name',
          'gene_2_name',
          'gene_2_name',
          'gene_2_name',
          'gene_2_name',
      ],
      'Feature': [
          'gene',
          'transcript',
          'transcript',
          'transcript',
          'gene',
          'transcript',
          'exon',
          'exon',
      ],
  })
  gtf['Chromosome'] = 'chr1'
  gtf['Score'] = '.'
  gtf['Frame'] = '.'
  gtf['Source'] = 'ENSEMBL'
  gtf['gene_type'] = 'protein_coding'
  return gtf


def _junction_track_metadata() -> pd.DataFrame:
  return pd.DataFrame({
      'name': [
          'Brain_Cerebellum',
          'Adipose_Subcutaneous',
          'UBERON:0036149',
      ],
      'other': ['foo', 'foo', 'foo'],
      'gtex_tissue': [
          'Brain_Cerebellum',
          'Adipose_Subcutaneous',
          '',
      ],
  })


class SpliceJunctionVariantScorerTest(parameterized.TestCase):

  def test_masks_and_metadata(
      self,
  ):
    settings = variant_scorers.SpliceJunctionScorer()
    interval = genome.Interval('chr1', 0, 2048)
    variant = genome.Variant('chr1', 105, 'C', 'T', name='var_0')
    scorer = splice_junction.SpliceJunctionVariantScorer(_create_fake_gtf())
    (
        masks,
        metadata,
    ) = scorer.get_masks_and_metadata(
        interval,
        variant,
        settings=settings,
        track_metadata=dna_output.OutputMetadata(),
    )
    self.assertIsNone(masks)
    self.assertIsInstance(metadata, pd.DataFrame)
    self.assertContainsSubset(
        ['gene_id', 'Strand', 'Chromosome', 'Start', 'End', 'interval'],
        metadata.columns,
    )
    # We expect to retrieve metadata for 2 genes overlapping the variant.
    self.assertSequenceEqual(list(metadata.gene_id), ['G1', 'G2'])

  @parameterized.product(transfer_guard=['disallow', 'allow'])
  def test_score_variant(self, transfer_guard: str):
    scorer = splice_junction.SpliceJunctionVariantScorer(_create_fake_gtf())

    interval = genome.Interval('chr1', 0, 2048)
    variant = genome.Variant('chr1', 105, 'C', 'T')

    max_splice_sites = splice_junction.MAX_SPLICE_SITES
    num_splice_sites = max_splice_sites + 1
    num_tracks = 5
    splice_site_positions = (
        jnp.arange(num_splice_sites * 4, dtype=jnp.int32).reshape(
            (4, num_splice_sites)
        )
        + 1
    )
    reference_predictions = jnp.arange(
        num_splice_sites * num_splice_sites * num_tracks, dtype=np.float32
    ).reshape((num_splice_sites, num_splice_sites, num_tracks))
    alternative_predictions = jnp.zeros_like(reference_predictions)
    with jax.transfer_guard(transfer_guard):
      scores = scorer.score_variant(
          ref={
              dna_output.OutputType.SPLICE_JUNCTIONS: {
                  'predictions': reference_predictions,
                  'splice_site_positions': splice_site_positions,
              }
          },
          alt={
              dna_output.OutputType.SPLICE_JUNCTIONS: {
                  'predictions': alternative_predictions,
                  'splice_site_positions': splice_site_positions,
              }
          },
          masks=None,
          settings=variant_scorers.SpliceJunctionScorer(),
          interval=interval,
          variant=variant,
      )

    np.testing.assert_array_equal(
        scores['splice_site_positions'],
        splice_site_positions[:, :max_splice_sites],
    )
    chex.assert_shape(
        scores['delta_counts'], (max_splice_sites, max_splice_sites, num_tracks)
    )

  @parameterized.product(return_empty_anndata=[True, False])
  def test_finalize_variant(self, return_empty_anndata: bool):
    settings = variant_scorers.SpliceJunctionScorer()
    chromosome = 'chr2' if return_empty_anndata else 'chr1'
    interval = genome.Interval(chromosome, 0, 2048)
    variant = genome.Variant(chromosome, 10, 'C', 'T', name='var_0')
    gtf = _create_fake_gtf()
    # Create mock track metadata.
    track_metadata = _junction_track_metadata()
    scorer = splice_junction.SpliceJunctionVariantScorer(gtf)
    _, mask_metadata = scorer.get_masks_and_metadata(
        interval,
        variant,
        track_metadata=dna_output.OutputMetadata(
            splice_junctions=track_metadata
        ),
        settings=settings,
    )
    if return_empty_anndata:
      self.assertEmpty(mask_metadata)
    else:
      self.assertLen(mask_metadata, 1)  # Only 1 gene overlapping the variant.

    # Create mock predictions.
    num_tracks = len(track_metadata) * 2  # 2 strands per track.
    max_splice_sites = 5
    predictions = (
        np.arange(max_splice_sites * max_splice_sites * num_tracks)
        .reshape((max_splice_sites, max_splice_sites, num_tracks))
        .astype(np.float32)
    )
    splice_site_positions = (
        np.arange(4 * max_splice_sites, dtype=np.int32).reshape(
            (4, max_splice_sites)
        )
    ) + 1
    # Make sure donor are upstream of acceptor.
    splice_site_positions[1] = splice_site_positions[1] + 1
    splice_site_positions[2] = splice_site_positions[2] + 20
    splice_site_positions[3] = splice_site_positions[3] + 10
    scores = {
        'delta_counts': predictions,
        'splice_site_positions': splice_site_positions,
    }
    results = scorer.finalize_variant(
        scores,
        track_metadata=dna_output.OutputMetadata(
            splice_junctions=track_metadata
        ),
        mask_metadata=mask_metadata,
        settings=settings,
    )

    self.assertIsInstance(results, anndata.AnnData)
    for column in [
        'junction_Start',
        'junction_End',
        'gene_id',
        'Strand',
        'Chromosome',
    ]:
      self.assertIn(column, results.obs.columns)

    names = {
        'Brain_Cerebellum',
        'Adipose_Subcutaneous',
        'UBERON:0036149',
    }
    self.assertEqual(set(results.var['name'].tolist()), names)

    if return_empty_anndata:
      self.assertEqual(results.shape, (0, 3))
    else:
      # We have 1 gene x 3 tissues, and the two junctions have the same
      # junction that has the maximum score.
      self.assertEqual(results.shape, (1, 3))


if __name__ == '__main__':
  absltest.main()
