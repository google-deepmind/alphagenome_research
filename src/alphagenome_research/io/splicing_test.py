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
from alphagenome_research.io import splicing
import numpy as np
import pandas as pd


class SpliceSiteAnnotationExtractorTest(parameterized.TestCase):

  @parameterized.product(interval_strand=['+', '-'], with_tissues=[True, False])
  def test_extract(self, interval_strand, with_tissues):
    interval_start = 100
    interval_end = 116

    pos_first_bp_intron_0based = np.asarray([101, 106, 107])
    neg_first_bp_intron_0based = np.asarray([101, 106, 107])
    pos_last_bp_intron_1based = np.asarray([104, 109, 115])
    neg_last_bp_intron_1based = np.asarray([104, 109, 115])
    junction_starts = pd.DataFrame({
        'Chromosome': 'chr1',
        'Start': [*pos_first_bp_intron_0based, *neg_first_bp_intron_0based],
        'Strand': ['+', '+', '+', '-', '-', '-'],
    })
    junction_ends = pd.DataFrame({
        'Chromosome': 'chr1',
        'End': [*pos_last_bp_intron_1based, *neg_last_bp_intron_1based],
        'Strand': ['+', '+', '+', '-', '-', '-'],
    })
    if with_tissues:
      junction_starts['Tissue_0'] = [1, 3, 5, 2, 4, 6]
      junction_starts['Tissue_1'] = [2, 4, 6, 3, 5, 7]
      junction_ends['Tissue_0'] = [1, 3, 5, 2, 4, 6]
      junction_ends['Tissue_1'] = [2, 4, 6, 3, 5, 7]

    interval = genome.Interval(
        'chr1', interval_start, interval_end, interval_strand
    )
    extractor = splicing.SpliceSiteAnnotationExtractor(
        junction_starts, junction_ends
    )
    result = extractor.extract(interval)

    expected_sites = np.zeros((interval.width, 5), dtype=bool)

    # Move all to the exon bp, and to 0-based coordinates within the interval.
    pos_last_exon_before = pos_first_bp_intron_0based - interval_start - 1
    pos_first_exon_after = pos_last_bp_intron_1based - interval_start
    neg_first_exon_after = neg_last_bp_intron_1based - interval_start
    neg_last_exon_before = neg_first_bp_intron_0based - interval_start - 1
    if interval_strand == '+':
      expected_sites[pos_last_exon_before, 0] = True
      expected_sites[pos_first_exon_after, 1] = True
      expected_sites[neg_first_exon_after, 2] = True
      expected_sites[neg_last_exon_before, 3] = True
    elif interval_strand == '-':
      pos_last_exon_before = interval.width - 1 - pos_last_exon_before
      pos_first_exon_after = interval.width - 1 - pos_first_exon_after
      neg_last_exon_before = interval.width - 1 - neg_last_exon_before
      neg_first_exon_after = interval.width - 1 - neg_first_exon_after
      expected_sites[neg_first_exon_after, 0] = True
      expected_sites[neg_last_exon_before, 1] = True
      expected_sites[pos_last_exon_before, 2] = True
      expected_sites[pos_first_exon_after, 3] = True
    expected_sites[:, 4] = np.logical_not(np.any(expected_sites, axis=1))

    np.testing.assert_array_equal(result, expected_sites)

  def test_extract_zero_expression_junction_not_marked(self):
    """Junctions with zero expression in every tissue are not splice sites."""
    interval_start = 100
    interval_end = 116
    junction_starts = pd.DataFrame({
        'Chromosome': 'chr1',
        'Start': [101, 106],
        'Strand': ['+', '+'],
        'Tissue_0': [1, 0],
    })
    junction_ends = pd.DataFrame({
        'Chromosome': 'chr1',
        'End': [104, 109],
        'Strand': ['+', '+'],
        'Tissue_0': [1, 0],
    })
    interval = genome.Interval('chr1', interval_start, interval_end, '+')
    extractor = splicing.SpliceSiteAnnotationExtractor(
        junction_starts, junction_ends
    )
    result = extractor.extract(interval)

    expected_sites = np.zeros((interval.width, 5), dtype=bool)
    # The expressed junction (Start=101, End=104) contributes a donor at
    # Start-1 and an acceptor at End, in 0-based coordinates within the interval.
    expected_sites[0, 0] = True
    expected_sites[4, 1] = True
    # The zero-expression junction (Start=106, End=109) must not be marked.
    expected_sites[:, 4] = np.logical_not(np.any(expected_sites[:, :4], axis=1))

    np.testing.assert_array_equal(result, expected_sites)

  def test_tissue_mismatch_raises(self):
    junction_starts = pd.DataFrame({
        'Chromosome': 'chr1',
        'Start': [101, 106, 107],
        'Strand': ['+', '+', '+'],
        'Tissue_0': [1, 3, 5],
    })
    junction_ends = pd.DataFrame({
        'Chromosome': 'chr1',
        'End': [104, 109, 115],
        'Strand': ['+', '+', '+'],
        'Tissue_1': [1, 3, 5],
    })
    with self.assertRaisesRegex(ValueError, 'Tissues mismatch'):
      _ = splicing.SpliceSiteAnnotationExtractor(junction_starts, junction_ends)


class PositionExtractorTest(parameterized.TestCase):

  def test_extract(self):
    df = pd.DataFrame({'Chromosome': ['chr1'], 'position': [1]})
    ex = splicing.PositionExtractor(df, position_column='position')
    # Inside.
    self.assertLen(ex.extract(genome.Interval('chr1', 0, 2)), 1)
    self.assertLen(ex.extract(genome.Interval('chr1', 1, 2)), 1)
    # Outside.
    self.assertEmpty(ex.extract(genome.Interval('chr2', 0, 2)))
    self.assertEmpty(ex.extract(genome.Interval('chr1', 0, 1)))
    self.assertEmpty(ex.extract(genome.Interval('chr1', 2, 3)))


if __name__ == '__main__':
  absltest.main()
