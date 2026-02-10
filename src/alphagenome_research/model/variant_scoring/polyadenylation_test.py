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
from alphagenome_research.model.variant_scoring import polyadenylation
import anndata
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def _load_pas_scores():
  common = dict(
      Chromosome='chr1',
  )
  # PAS sites for 3 genes. G1 PAS sites fall in the middle of the interval, G2
  # PAS sites fall at the beginning of the interval with 80% of them in the
  # interval, and G3 PAS sites fall outside of the interval.
  apa_gtf = pd.DataFrame({
      'Start': [
          1575,
          1585,
          1605,
          980,
          1000,
          1010,
          1020,
          1045,
          3120,
          3150,
          3180,
          1100,
      ],
      'pas_strand': [
          '+',
          '+',
          '+',
          '-',
          '-',
          '-',
          '-',
          '-',
          '+',
          '+',
          '+',
          '+',
      ],
      'pas_gene_id': [
          'G1',
          'G1',
          'G1',
          'G2',
          'G2',
          'G2',
          'G2',
          'G2',
          'G3',
          'G3',
          'G3',
          'G4',
      ],
  })
  apa_gtf['End'] = apa_gtf['Start'] + 1
  apa_gtf['cutmode'] = apa_gtf['Start']
  apa_gtf['pas_id'] = apa_gtf.index
  apa_gtf['gene_id'] = apa_gtf['pas_gene_id']

  for k, v in common.items():
    apa_gtf[k] = v
  return apa_gtf


def _load_gtf():
  """Loads a GTF PAS scoring test."""
  common = dict(
      Chromosome='chr1',
      Score='.',
      Frame='.',
      Source='ENSEMBL',
      gene_type='protein_coding',
      Feature='gene',
  )
  # Relative to variant at position 1700 in interval (chr1, 1000, 3048)
  # Three genes (G1, G2, G3).
  # One gene is at the beginning of the query interval, middle, and end.
  # Only uses gene elements.
  gtf = pd.DataFrame({
      'Start': [1500, 950, 1600, 1050],
      'End': [1800, 1800, 3100, 1800],
      'Strand': ['+', '-', '+', '+'],
      'gene_id': ['G1', 'G2', 'G3', 'G4'],
      'gene_name': [
          'gene_1_name',
          'gene_2_name',
          'gene_3_name',
          'gene_4_name',
      ],
  })
  for k, v in common.items():
    gtf[k] = v
  return gtf


def _load_pas_scoring_inputs() -> (
    tuple[genome.Variant, genome.Interval, pd.DataFrame, pd.DataFrame]
):
  """Loads PAS scoring test specific gtf, PAS gtf, interval, and variant."""
  interval = genome.Interval('chr1', 1000, 3048)
  variant = genome.Variant('chr1', 1700, 'C', 'T', name='var_0')

  return variant, interval, _load_gtf(), _load_pas_scores()


class PolyadenylationTest(parameterized.TestCase):

  def test_masks_and_metadata(self):
    variant, interval, gtf, pas_gtf = _load_pas_scoring_inputs()

    scorer = polyadenylation.PolyadenylationVariantScorer(gtf, pas_gtf)
    settings = variant_scorers.PolyadenylationScorer()
    masks, metadata = scorer.get_masks_and_metadata(
        interval,
        variant,
        settings=settings,
        track_metadata=dna_output.OutputMetadata(),
    )
    self.assertSameElements(
        [
            'Chromosome',
            'Start',
            'End',
            'interval_start',
            'strand',
            'gene_name',
            'gene_id',
            'gene_type',
            'num_pas',
            'min_pas_var_distance',
        ],
        metadata.columns,
    )
    # Check there is only metadata for the 2 genes with >1 PAS in the interval.
    self.assertLen(metadata, 2)

    chex.assert_shape(masks.gene_mask, (polyadenylation.MAX_GENES,))
    self.assertEqual(masks.gene_mask.sum(), 2)

    chex.assert_shape(
        masks.pas_mask,
        (interval.width, polyadenylation.MAX_GENES, polyadenylation.MAX_PAS),
    )

  @parameterized.product(transfer_guard=['disallow', 'allow'])
  def test_pas_variant_scorer_score_variant(self, transfer_guard: str):
    # Use mask for seq len 15, 5 genes, MAX_PAS sites with PAS mask width == 1
    # so aggregated across sequence length == actual value at that point in the
    # track.
    interval = genome.Interval('chr1', 1000, 1015)
    variant = genome.Variant('chr1', 1005, 'C', 'T', name='var_0')
    gene_pas_mask = np.zeros(
        (interval.width, 6, polyadenylation.MAX_PAS), dtype=bool
    )
    # G1 has one PAS sites
    gene_pas_mask[0, 0, 0] = True
    # G2 has 2 PAS sites
    for i in range(1, 3):
      gene_pas_mask[i, 1, i] = True
    # G3 has 3 PAS sites
    for i in range(3, 6):
      gene_pas_mask[i, 2, i] = True
    # G4 has 4 PAS sites
    for i in range(6, 10):
      gene_pas_mask[i, 3, i] = True
    # G5 has 4 PAS
    for i in range(10, 14):
      gene_pas_mask[i, 4, i] = True

    # Set up tracks such that MAX COVR RATIO is equal for all tracks
    # Score should be zero because only 1 PAS site
    g1_values = jnp.array([10], jnp.float32).repeat(100).reshape(1, 100)
    # Score should be 0.5, only one coverage ratio fc to take the max over
    # because only 2 pas sites to make proximal-distal split
    g2_values = jnp.array([2, 4], jnp.float32).repeat(100).reshape(2, 100)
    # Score should be 5, argmax k = 2
    g3_values = jnp.array([2, 48, 5], jnp.float32).repeat(100).reshape(3, 100)
    # Score should be 2, argmax k = 3
    g4_values = (
        jnp.array([0, 5, 55, 10], jnp.float32).repeat(100).reshape(4, 100)
    )
    # Score should be 10, argmax k = 2
    g5_values = (
        jnp.array([10, 90, 5, 5, 0], jnp.float32).repeat(100).reshape(5, 100)
    )

    # Max ratio coverage uses alt_agg/ref_agg, so set alt tracks to the
    # covr_ratios needed for test case answers and ref tracks to ones.
    alt = jnp.concatenate(
        [g1_values, g2_values, g3_values, g4_values, g5_values],
        axis=0,
    )
    ref = jnp.ones_like(alt)
    masks = polyadenylation.PolyadenylationVariantMasks(
        pas_mask=gene_pas_mask,
        gene_mask=np.array([True] * 5 + [False]),
    )
    _, _, gtf, pas_gtf = _load_pas_scoring_inputs()
    variant_scorer = polyadenylation.PolyadenylationVariantScorer(
        gtf=gtf, pas_gtf=pas_gtf
    )
    with jax.transfer_guard(transfer_guard):
      scores = variant_scorer.score_variant(
          {dna_output.OutputType.RNA_SEQ: ref},
          {dna_output.OutputType.RNA_SEQ: alt},
          masks=jax.device_put(masks),
          settings=variant_scorers.PolyadenylationScorer(),
          interval=interval,
          variant=variant,
      )
    np.testing.assert_array_equal(
        scores['scores'][0, :], jnp.zeros(100, dtype=jnp.float32)
    )
    np.testing.assert_array_equal(
        scores['scores'][1, :],
        jnp.abs(jnp.log2(np.ones(100, dtype=jnp.float32) * 0.5)),
    )
    np.testing.assert_array_equal(
        scores['scores'][2, :],
        jnp.abs(jnp.log2(np.ones(100, dtype=jnp.float32) * (4 / 53))),
    )
    np.testing.assert_array_equal(
        scores['scores'][3, :],
        jnp.abs(jnp.log2(np.ones(100, dtype=jnp.float32) * (5 / 65))),
    )
    np.testing.assert_array_equal(
        scores['scores'][4, :],
        jnp.abs(jnp.log2(np.ones(100, dtype=jnp.float32) * 10)),
    )

  def test_finalize_variant(self):
    variant, interval, gtf, pas_gtf = _load_pas_scoring_inputs()
    settings = variant_scorers.PolyadenylationScorer()
    variant_scorer = polyadenylation.PolyadenylationVariantScorer(
        gtf=gtf, pas_gtf=pas_gtf
    )
    track_metadata = pd.DataFrame(
        {'name': np.arange(10).astype(str), 'Strand': '.', 'padding': False}
    )
    # Scenario of 3 genes in the interval (max possible=4) and 10 output tracks.
    expected_scores = np.ones((4, 10), dtype=jnp.float32)
    scores = {
        'scores': expected_scores,
        'gene_mask': np.array([True, False, True, False]),
    }

    _, mask_metadata = variant_scorer.get_masks_and_metadata(
        interval,
        variant,
        settings=settings,
        track_metadata=dna_output.OutputMetadata(),
    )
    finalized_variant = variant_scorer.finalize_variant(
        scores,
        settings=settings,
        track_metadata=dna_output.OutputMetadata(rna_seq=track_metadata),
        mask_metadata=mask_metadata,
    )

    self.assertIsInstance(finalized_variant, anndata.AnnData)
    self.assertTrue(finalized_variant.X.flags['C_CONTIGUOUS'])

    self.assertLen(finalized_variant.var, expected_scores.shape[-1])


if __name__ == '__main__':
  absltest.main()
