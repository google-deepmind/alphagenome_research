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
from alphagenome_research.model.variant_scoring import gene_mask
from alphagenome_research.model.variant_scoring import gene_mask_extractor as gene_mask_extractor_lib
import anndata
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def _get_mock_gtf():
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


def _get_mock_rnaseq_track_metadata() -> pd.DataFrame:
  return pd.DataFrame({
      'name': [
          'CL:0000047 polyA plus RNA-seq',
          'CL:0000062 total RNA-seq',
          'CL:0000084 polyA plus RNA-seq',
          'CL:0000084 total RNA-seq',
          'CL:0000115 total RNA-seq',
          'CL:0000127 total RNA-seq',
          (
              'EFO:0000572 gtex Cells_EBV-transformed_lymphocytes polyA plus'
              ' RNA-seq'
          ),
          'EFO:0002009 gtex Cells_Cultured_fibroblasts polyA plus RNA-seq',
          'UBERON:0000007 gtex Pituitary polyA plus RNA-seq',
          'UBERON:0000458 gtex Cervix_Endocervix polyA plus RNA-seq',
      ],
      'strand': ['+', '+', '+', '+', '+', '+', '.', '.', '.', '.'],
      'gtex_tissue': [
          '',
          '',
          '',
          '',
          '',
          '',
          'Cells_EBV-transformed_lymphocytes',
          'Cells_Cultured_fibroblasts',
          'Pituitary',
          'Cervix_Endocervix',
      ],
      'padding': [False] * 10,
  })


class GeneMaskVariantScorerTest(parameterized.TestCase):

  def test_get_masks_and_metadata_body(self):
    gene_mask_extractor = gene_mask_extractor_lib.GeneMaskExtractor(
        gtf=_get_mock_gtf(),
        gene_mask_type=gene_mask_extractor_lib.GeneMaskType.BODY,
    )
    pad_num_genes = 32
    gene_variant_scorer = gene_mask.GeneVariantScorer(
        gene_mask_extractor=gene_mask_extractor,
        pad_num_genes=pad_num_genes,
    )
    interval = genome.Interval('chr1', 0, 256)
    variant = genome.Variant('chr1', 10, 'C', 'T')
    settings = variant_scorers.GeneMaskLFCScorer(
        requested_output=dna_output.OutputType.RNA_SEQ,
    )
    track_metadata = dna_output.OutputMetadata(
        rna_seq=_get_mock_rnaseq_track_metadata()
    )
    gene_masks, metadata = gene_variant_scorer.get_masks_and_metadata(
        interval, variant, settings=settings, track_metadata=track_metadata
    )
    self.assertIsInstance(metadata, pd.DataFrame)
    self.assertLen(metadata, 2)
    self.assertSequenceEqual(list(metadata.gene_id), ['G1', 'G2'])

    self.assertEqual(gene_masks.gene_mask.shape, (256, pad_num_genes))
    expected_g1_mask = np.zeros(256, dtype=bool)
    expected_g1_mask[101:200] = True
    expected_g2_mask = np.zeros(256, dtype=bool)
    expected_g2_mask[0:108] = True
    np.testing.assert_array_equal(gene_masks.gene_mask[:, 0], expected_g1_mask)
    np.testing.assert_array_equal(gene_masks.gene_mask[:, 1], expected_g2_mask)

  @parameterized.product(
      [
          dict(
              settings=variant_scorers.GeneMaskLFCScorer(
                  requested_output=dna_output.OutputType.RNA_SEQ,
              ),
              # ref values for gene: 4, 5, 6. mean = 5.
              # alt values for gene: 1, 1, 1. mean = 1.
              # expected_score = log(1 + 1e-3) - log(5 + 1e-3) = -1.6086375
              expected_score=-1.6086375,
          ),
          dict(
              settings=variant_scorers.GeneMaskActiveScorer(
                  requested_output=dna_output.OutputType.RNA_SEQ,
              ),
              # ref values for gene: 4, 5, 6. mean = 5.
              # alt values for gene: 1, 1, 1. mean = 1.
              # expected_score = max(1, 5) = 5.
              expected_score=5.0,
          ),
      ],
      transfer_guard=['disallow', 'allow'],
  )
  def test_score_variant(
      self, settings, expected_score: float, transfer_guard: str
  ):
    gtf = pd.DataFrame({
        'End': [7],
        'Start': [4],
        'Strand': ['+'],
        'gene_id': ['G1'],
        'gene_name': ['gene_1_name'],
        'Feature': ['gene'],
        'Chromosome': ['chr1'],
        'Score': ['.'],
        'Frame': ['.'],
        'Source': ['ENSEMBL'],
        'gene_type': ['protein_coding'],
        'transcript_id': [''],
        'transcript_type': [''],
    })
    gene_mask_extractor = gene_mask_extractor_lib.GeneMaskExtractor(
        gtf=gtf,
        gene_mask_type=gene_mask_extractor_lib.GeneMaskType.BODY,
    )
    pad_num_genes = 32
    gene_variant_scorer = gene_mask.GeneVariantScorer(
        gene_mask_extractor=gene_mask_extractor,
        pad_num_genes=pad_num_genes,
    )
    interval = genome.Interval('chr1', 0, 11)
    variant = genome.Variant('chr1', 1, 'C', 'T')
    track_metadata = dna_output.OutputMetadata(
        rna_seq=_get_mock_rnaseq_track_metadata()
    )
    self.assertIsNotNone(track_metadata.rna_seq)
    num_tracks = len(track_metadata.rna_seq)
    gene_masks, _ = gene_variant_scorer.get_masks_and_metadata(
        interval, variant, settings=settings, track_metadata=track_metadata
    )
    expected_g1_mask = np.zeros((11, pad_num_genes), dtype=bool)
    expected_g1_mask[4:7, 0] = True
    np.testing.assert_array_equal(gene_masks.gene_mask, expected_g1_mask)

    ref = (
        jnp.arange(11, dtype=jnp.float32)
        .reshape(-1, 1)
        .repeat(num_tracks, axis=1)
    )
    alt = jnp.ones((11, num_tracks), dtype=jnp.float32)
    with jax.transfer_guard(transfer_guard):
      scores = gene_variant_scorer.score_variant(
          ref={settings.requested_output: ref},
          alt={settings.requested_output: alt},
          masks=jax.device_put(gene_masks),
          settings=settings,
          variant=variant,
          interval=interval,
      )
    padded_expected_scores = np.full(
        (pad_num_genes, num_tracks), dtype=np.float32, fill_value=np.nan
    )
    padded_expected_scores[0, :] = expected_score
    np.testing.assert_almost_equal(
        scores['score'],
        padded_expected_scores,
        decimal=5,
    )

  @parameterized.product(
      variant=[
          genome.Variant('chr1', 5, 'A', 'AT'),
          genome.Variant('chr1', 4, 'AA', 'AAT'),
          genome.Variant('chr1', 3, 'AAA', 'AAAT'),
      ],
      transfer_guard=['disallow', 'allow'],
  )
  def test_gene_variant_scorer_splicing_insertion(
      self, variant: genome.Variant, transfer_guard: str
  ):
    # Score an insertion that adds a predicted splicing donor at position 5.
    # In addition, there is a predicted splicing acceptor at position 7 in the
    # REF (position 8 in the ALT due to the insertion), which should be neutral.
    gtf = pd.DataFrame({
        'Chromosome': ['chr1'],
        'Start': [0],
        'End': [20],
        'Strand': ['+'],
        'Feature': ['gene'],
        'gene_id': ['G1'],
        'gene_name': ['Gene1'],
        'gene_type': ['protein_coding'],
        'transcript_id': [''],
        'transcript_type': [''],
    })
    variant_scorer = gene_mask.GeneVariantScorer(
        gene_mask_extractor=gene_mask_extractor_lib.GeneMaskExtractor(
            gtf=gtf,
            gene_mask_type=gene_mask_extractor_lib.GeneMaskType.BODY,
            gene_query_type=(
                gene_mask_extractor_lib.GeneQueryType.VARIANT_OVERLAPPING
            ),
        ),
    )
    track_metadata = dna_output.OutputMetadata()
    settings = variant_scorers.GeneMaskSplicingScorer(
        requested_output=dna_output.OutputType.SPLICE_SITES,
        width=None,
    )
    interval = variant.reference_interval.resize(9)
    masks, _ = variant_scorer.get_masks_and_metadata(
        interval, variant, settings=settings, track_metadata=track_metadata
    )

    alt = jnp.zeros((9, 4), dtype=jnp.float32)
    alt = alt.at[4 - interval.start, 0].set(1)
    alt = alt.at[7 - interval.start, 2].set(1)
    ref = jnp.zeros((9, 4), dtype=jnp.float32)
    ref = ref.at[6 - interval.start, 2].set(1)

    with jax.transfer_guard(transfer_guard):
      scores = variant_scorer.score_variant(
          {dna_output.OutputType.SPLICE_SITES: ref},
          {dna_output.OutputType.SPLICE_SITES: alt},
          masks=jax.device_put(masks),
          settings=settings,
          variant=variant,
          interval=interval,
      )
    # A new splicing donor was inserted, so the 0th track should be 1.0. A
    # splicing acceptor was kept intact, so the 1st track should remain 0.0.
    np.testing.assert_array_equal(
        scores['score'][0], np.array([1, 0, 0, 0], dtype=jnp.float32)
    )

  @parameterized.parameters(
      genome.Variant('chr1', 5, 'AT', 'A'),
      genome.Variant('chr1', 4, 'AAT', 'AA'),
      genome.Variant('chr1', 3, 'AAAT', 'AAA'),
  )
  def test_gene_variant_scorer_splicing_deletion(self, variant):
    # Score a deletion that removes a predicted splicing donor at position 5.
    # In addition, there is a predicted splicing acceptor at position 4,
    # which should be neutral.
    gtf = pd.DataFrame({
        'Chromosome': ['chr1'],
        'Start': [0],
        'End': [20],
        'Strand': ['+'],
        'Feature': ['gene'],
        'gene_id': ['G1'],
        'gene_name': ['Gene1'],
        'gene_type': ['protein_coding'],
        'transcript_id': [''],
        'transcript_type': [''],
    })
    variant_scorer = gene_mask.GeneVariantScorer(
        gene_mask_extractor=gene_mask_extractor_lib.GeneMaskExtractor(
            gtf=gtf,
            gene_mask_type=gene_mask_extractor_lib.GeneMaskType.BODY,
            gene_query_type=(
                gene_mask_extractor_lib.GeneQueryType.VARIANT_OVERLAPPING
            ),
        ),
    )
    track_metadata = dna_output.OutputMetadata()
    settings = variant_scorers.GeneMaskSplicingScorer(
        requested_output=dna_output.OutputType.SPLICE_SITES,
        width=None,
    )
    interval = variant.reference_interval.resize(9)
    masks, _ = variant_scorer.get_masks_and_metadata(
        interval, variant, settings=settings, track_metadata=track_metadata
    )

    alt = jnp.zeros((9, 4), dtype=jnp.float32)
    alt = alt.at[4 - interval.start, 2].set(1)
    ref = jnp.zeros((9, 4), dtype=jnp.float32)
    ref = ref.at[5 - interval.start, 0].set(1)
    ref = ref.at[4 - interval.start, 2].set(1)

    scores = variant_scorer.score_variant(
        {dna_output.OutputType.SPLICE_SITES: ref},
        {dna_output.OutputType.SPLICE_SITES: alt},
        masks=jax.device_put(masks),
        settings=settings,
        variant=variant,
        interval=interval,
    )
    # A splicing donor was deleted, so the 0th track should be 1.0. A
    # splicing acceptor was kept intact, so the other tracks should remain 0.0.
    np.testing.assert_array_equal(
        scores['score'][0], np.array([1, 0, 0, 0], dtype=jnp.float32)
    )

  def test_finalize_variant(self):
    gene_mask_extractor = gene_mask_extractor_lib.GeneMaskExtractor(
        gtf=_get_mock_gtf(),
        gene_mask_type=gene_mask_extractor_lib.GeneMaskType.BODY,
    )
    gene_variant_scorer = gene_mask.GeneVariantScorer(
        gene_mask_extractor=gene_mask_extractor,
    )
    interval = genome.Interval('chr1', 0, 256)
    variant = genome.Variant('chr1', 10, 'C', 'T')
    settings = variant_scorers.GeneMaskLFCScorer(
        requested_output=dna_output.OutputType.RNA_SEQ,
    )
    df_track_metadata = _get_mock_rnaseq_track_metadata()
    track_metadata = dna_output.OutputMetadata(rna_seq=df_track_metadata)
    _, mask_metadata = gene_variant_scorer.get_masks_and_metadata(
        interval, variant, settings=settings, track_metadata=track_metadata
    )
    self.assertLen(mask_metadata, 2)  # Two genes.
    scores = {'score': np.ones((2, len(df_track_metadata)), dtype=np.float32)}
    finalized_variant = gene_variant_scorer.finalize_variant(
        scores,
        track_metadata=track_metadata,
        mask_metadata=mask_metadata,
        settings=settings,
    )
    self.assertIsInstance(finalized_variant, anndata.AnnData)
    self.assertLen(finalized_variant.obs, 2)
    self.assertLen(finalized_variant.var, len(df_track_metadata))


if __name__ == '__main__':
  absltest.main()
