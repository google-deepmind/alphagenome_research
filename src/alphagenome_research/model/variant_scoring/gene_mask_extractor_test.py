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
from alphagenome_research.model.variant_scoring import gene_mask_extractor
import chex
import numpy as np
import pandas as pd


def _get_dummy_gtf() -> pd.DataFrame:
  """Returns a dummy GTF DataFrame for testing."""
  data = [
      ['gene', 'chr1', 10, 100, '+', 'gene1', 'name1', 'type1', ''],
      ['transcript', 'chr1', 10, 100, '+', 'gene1', 'name1', 'type1', 't1'],
      ['exon', 'chr1', 10, 50, '+', 'gene1', 'name1', 'type1', 't1'],
      ['exon', 'chr1', 70, 100, '+', 'gene1', 'name1', 'type1', 't1'],
      ['gene', 'chr1', 120, 150, '-', 'gene2', 'name2', 'type2', ''],
      ['transcript', 'chr1', 120, 150, '-', 'gene2', 'name2', 'type2', 't2'],
      ['exon', 'chr1', 120, 150, '-', 'gene2', 'name2', 'type2', 't2'],
  ]
  return pd.DataFrame(
      data,
      columns=[
          'Feature',
          'Chromosome',
          'Start',
          'End',
          'Strand',
          'gene_id',
          'gene_name',
          'gene_type',
          'transcript_id',
      ],
  )


class GeneMaskExtractorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._gtf = _get_dummy_gtf()

  @parameterized.named_parameters([
      # gene_mask_type=BODY, gene_query_type=INTERVAL_CONTAINED
      dict(
          testcase_name='body_interval_two_genes',
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 0, 200),
          expected_num_genes=2,
          expected_masks_segments=[(0, [(10, 100)]), (1, [(120, 150)])],
          expected_gene_ids=['gene1', 'gene2'],
      ),
      dict(
          testcase_name='body_interval_two_genes_shifted_interval',
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 5, 200),
          expected_num_genes=2,
          expected_masks_segments=[(0, [(5, 95)]), (1, [(115, 145)])],
          expected_gene_ids=['gene1', 'gene2'],
      ),
      dict(
          testcase_name=(
              'body_interval_two_genes_shifted_interval_outside_of_gene_range'
          ),
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 15, 200),
          expected_num_genes=1,
          expected_masks_segments=[(0, [(105, 135)])],
          expected_gene_ids=['gene2'],
      ),
      dict(
          testcase_name='body_interval_one_gene',
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 0, 110),
          expected_num_genes=1,
          expected_masks_segments=[(0, [(10, 100)])],
          expected_gene_ids=['gene1'],
      ),
      dict(
          testcase_name='body_interval_no_genes',
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 0, 10),
          expected_num_genes=0,
          expected_masks_segments=[],
          expected_gene_ids=[],
      ),
      dict(
          testcase_name='body_interval_no_genes_in_region',
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 300, 400),
          expected_num_genes=0,
          expected_masks_segments=[],
          expected_gene_ids=[],
      ),
      dict(
          testcase_name='body_interval_wrong_chromosome',
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr2', 0, 200),
          expected_num_genes=0,
          expected_masks_segments=[],
          expected_gene_ids=[],
      ),
      # gene_mask_type=BODY, gene_query_type=VARIANT_OVERLAPPING
      dict(
          testcase_name='body_variant_variant_in_gene1',
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.VARIANT_OVERLAPPING,
          interval=genome.Interval('chr1', 0, 200),
          variant=genome.Variant('chr1', 55, 'A', 'G'),
          expected_num_genes=1,
          expected_masks_segments=[(0, [(10, 100)])],
          expected_gene_ids=['gene1'],
      ),
      dict(
          testcase_name='body_variant_variant_in_gene2',
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.VARIANT_OVERLAPPING,
          interval=genome.Interval('chr1', 0, 200),
          variant=genome.Variant('chr1', 130, 'A', 'G'),
          expected_num_genes=1,
          expected_masks_segments=[(0, [(120, 150)])],
          expected_gene_ids=['gene2'],
      ),
      dict(
          testcase_name='body_variant_variant_between_genes',
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.VARIANT_OVERLAPPING,
          interval=genome.Interval('chr1', 0, 200),
          variant=genome.Variant('chr1', 110, 'A', 'G'),
          expected_num_genes=0,
          expected_masks_segments=[],
          expected_gene_ids=[],
      ),
      dict(
          testcase_name='body_variant_variant_overlapping_two_genes',
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.VARIANT_OVERLAPPING,
          interval=genome.Interval('chr1', 0, 200),
          variant=genome.Variant('chr1', 99, 'A' * 30, 'G'),
          expected_num_genes=2,
          expected_masks_segments=[(0, [(10, 100)]), (1, [(120, 150)])],
          expected_gene_ids=['gene1', 'gene2'],
      ),
      dict(
          testcase_name='body_variant_variant_in_gene1_shifted_interval',
          gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
          gene_query_type=gene_mask_extractor.GeneQueryType.VARIANT_OVERLAPPING,
          interval=genome.Interval('chr1', 10, 110),
          variant=genome.Variant('chr1', 55, 'A', 'G'),
          expected_num_genes=1,
          expected_masks_segments=[(0, [(0, 90)])],
          expected_gene_ids=['gene1'],
      ),
      # gene_mask_type=EXONS, gene_query_type=INTERVAL_CONTAINED
      dict(
          testcase_name='exons_interval_two_genes',
          gene_mask_type=gene_mask_extractor.GeneMaskType.EXONS,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 0, 200),
          expected_num_genes=2,
          expected_masks_segments=[
              (0, [(10, 50), (70, 100)]),
              (1, [(120, 150)]),
          ],
          expected_gene_ids=['gene1', 'gene2'],
      ),
      dict(
          testcase_name='exons_interval_two_genes_shifted_interval',
          gene_mask_type=gene_mask_extractor.GeneMaskType.EXONS,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 5, 200),
          expected_num_genes=2,
          expected_masks_segments=[(0, [(5, 45), (65, 95)]), (1, [(115, 145)])],
          expected_gene_ids=['gene1', 'gene2'],
      ),
      dict(
          testcase_name=(
              'exons_interval_two_genes_shifted_interval_outside_of_gene_range'
          ),
          gene_mask_type=gene_mask_extractor.GeneMaskType.EXONS,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 15, 200),
          expected_num_genes=1,
          expected_masks_segments=[(0, [(105, 135)])],
          expected_gene_ids=['gene2'],
      ),
      dict(
          testcase_name='exons_interval_one_gene',
          gene_mask_type=gene_mask_extractor.GeneMaskType.EXONS,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 0, 110),
          expected_num_genes=1,
          expected_masks_segments=[(0, [(10, 50), (70, 100)])],
          expected_gene_ids=['gene1'],
      ),
      dict(
          testcase_name='exons_interval_no_genes',
          gene_mask_type=gene_mask_extractor.GeneMaskType.EXONS,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 0, 10),
          expected_num_genes=0,
          expected_masks_segments=[],
          expected_gene_ids=[],
      ),
      dict(
          testcase_name='exons_interval_no_genes_in_region',
          gene_mask_type=gene_mask_extractor.GeneMaskType.EXONS,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr1', 300, 400),
          expected_num_genes=0,
          expected_masks_segments=[],
          expected_gene_ids=[],
      ),
      dict(
          testcase_name='exons_interval_wrong_chromosome',
          gene_mask_type=gene_mask_extractor.GeneMaskType.EXONS,
          gene_query_type=gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
          variant=None,
          interval=genome.Interval('chr2', 0, 200),
          expected_num_genes=0,
          expected_masks_segments=[],
          expected_gene_ids=[],
      ),
  ])
  def test_gene_mask_extractor(
      self,
      gene_mask_type,
      gene_query_type,
      interval,
      variant,
      expected_num_genes,
      expected_masks_segments,
      expected_gene_ids,
  ):
    extractor = gene_mask_extractor.GeneMaskExtractor(
        self._gtf,
        gene_mask_type=gene_mask_type,
        gene_query_type=gene_query_type,
    )
    mask, metadata = extractor.extract(interval, variant)

    chex.assert_shape(mask, (interval.width, expected_num_genes))
    self.assertLen(metadata, expected_num_genes)
    if expected_num_genes > 0:
      self.assertListEqual(metadata.gene_id.tolist(), expected_gene_ids)

    for i, segments in expected_masks_segments:
      expected_mask = np.zeros(interval.width, dtype=bool)
      for start, end in segments:
        expected_mask[start:end] = True
      np.testing.assert_array_equal(mask[:, i], expected_mask)

  def test_gene_mask_extractor_with_wrong_gene_id_raises_error(self):
    extractor = gene_mask_extractor._GeneMaskExtractor(self._gtf)  # pylint: disable=protected-access
    interval = genome.Interval('chr1', 0, 200)
    with self.assertRaisesRegex(ValueError, 'Gene ID wrong_gene_id not found'):
      extractor.extract(interval, gene_id='wrong_gene_id')

  def test_exon_extractor_transcript_with_no_exons_returns_empty(self):
    extractor = gene_mask_extractor._ExonExtractor(self._gtf)  # pylint: disable=protected-access
    exons = extractor.extract(transcript_id='wrong_transcript_id')
    self.assertEmpty(exons)


if __name__ == '__main__':
  absltest.main()
