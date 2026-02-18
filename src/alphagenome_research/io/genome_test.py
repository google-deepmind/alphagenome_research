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
from alphagenome_research.io import fasta
from alphagenome_research.io import genome as genome_io


class VariantTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='Modification',
          sequence='AAAA',
          interval=genome.Interval('chr1', 0, 4),
          variant=genome.Variant('chr1', 3, 'A', 'G'),
          expected='AAGA',
      ),
      dict(
          testcase_name='Insertion',
          sequence='AAAA',
          interval=genome.Interval('chr2', 0, 4),
          variant=genome.Variant('chr2', 2, 'A', 'GGGG'),
          expected='AGGGGAA',
      ),
      dict(
          testcase_name='Deletion',
          sequence='AAAA',
          interval=genome.Interval('chr3', 0, 4),
          variant=genome.Variant('chr3', 2, 'AA', ''),
          expected='AA',
      ),
      dict(
          testcase_name='NoOverlap',
          sequence='AAAA',
          interval=genome.Interval('chr4', 0, 4),
          variant=genome.Variant('chr4', 6, 'A', 'G'),
          expected='AAAA',
      ),
      dict(
          testcase_name='OverlapStart',
          sequence='AAAA',
          interval=genome.Interval('chr4', 4, 8),
          variant=genome.Variant('chr4', 1, 'TTTTT', 'GGGGG'),
          expected='GAAA',
      ),
      dict(
          testcase_name='OverlapEnd',
          sequence='AAAA',
          interval=genome.Interval('chr4', 0, 4),
          variant=genome.Variant('chr4', 4, 'TTTTTT', 'GG'),
          expected='AAAG',
      ),
      dict(
          testcase_name='DifferentChromosomes',
          sequence='AAAA',
          interval=genome.Interval('chr1', 0, 4),
          variant=genome.Variant('chr2', 1, 'A', 'G'),
          expected='AAAA',
      ),
  ])
  def test_insert_alternate_variant(
      self,
      sequence: str,
      interval: genome.Interval,
      variant: genome.Variant,
      expected: str,
  ):
    alternate_sequence = genome_io.insert_alternate_variant(
        sequence, interval, variant
    )
    self.assertEqual(alternate_sequence, expected)

  def test_insert_alternate_variant_negative_strand_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'Interval must not be on the negative strand.'
    ):
      _ = genome_io.insert_alternate_variant(
          sequence='AAAA',
          interval=genome.Interval('chr1', 0, 4, '-'),
          variant=genome.Variant('chr1', 2, 'A', 'G'),
      )

  @parameterized.named_parameters([
      dict(
          testcase_name='Normal',
          sequence='AAAA',
          interval=genome.Interval('chr1', 0, 4),
          variant=genome.Variant('chr1', 2, 'A', 'G'),
          expected='AAAA',
      ),
      dict(
          testcase_name='ModifiedReference',
          sequence='AAAA',
          interval=genome.Interval('chr2', 0, 4),
          variant=genome.Variant('chr2', 2, 'GGG', ''),
          expected='AGGG',
      ),
      dict(
          testcase_name='NoOverlap',
          sequence='AAAA',
          interval=genome.Interval('chr4', 0, 4),
          variant=genome.Variant('chr4', 6, 'G', ''),
          expected='AAAA',
      ),
      dict(
          testcase_name='OverlapStart',
          sequence='AAAA',
          interval=genome.Interval('chr4', 4, 8),
          variant=genome.Variant('chr4', 1, 'GGGGG', ''),
          expected='GAAA',
      ),
      dict(
          testcase_name='OverlapEnd',
          sequence='AAAA',
          interval=genome.Interval('chr4', 0, 4),
          variant=genome.Variant('chr4', 4, 'GGG', ''),
          expected='AAAG',
      ),
      dict(
          testcase_name='DifferentChromosomes',
          sequence='AAAA',
          interval=genome.Interval('chr1', 0, 4),
          variant=genome.Variant('chr2', 1, 'G', ''),
          expected='AAAA',
      ),
  ])
  def test_insert_reference_variant(
      self,
      sequence: str,
      interval: genome.Interval,
      variant: genome.Variant,
      expected: str,
  ):
    reference_sequence = genome_io.insert_reference_variant(
        sequence, interval, variant
    )
    self.assertEqual(reference_sequence, expected)

  def test_insert_reference_variant_negative_strand_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'Interval must not be on the negative strand.'
    ):
      _ = genome_io.insert_reference_variant(
          sequence='AAAA',
          interval=genome.Interval('chr1', 0, 4, '-'),
          variant=genome.Variant('chr1', 2, 'A', 'G'),
      )

  @parameterized.named_parameters([
      dict(
          testcase_name='Normal',
          sequence='AAAA',
          interval=genome.Interval('chr1', 0, 4),
          variant=genome.Variant('chr1', 2, 'A', 'G'),
          expected_reference='AAAA',
          expected_alternate='AGAA',
      ),
      dict(
          testcase_name='ModifiedReference',
          sequence='AAAA',
          interval=genome.Interval('chr2', 0, 4),
          variant=genome.Variant('chr2', 2, 'G', 'G'),
          expected_reference='AGAA',
          expected_alternate='AGAA',
      ),
      dict(
          testcase_name='Insertion',
          sequence='AAAA',
          interval=genome.Interval('chr3', 0, 4),
          variant=genome.Variant('chr3', 2, 'A', 'GGGG'),
          expected_reference='AAAA',
          expected_alternate='AGGG',
      ),
      dict(
          testcase_name='Deletion',
          sequence='GATACA',
          interval=genome.Interval('chr3', 0, 4),
          variant=genome.Variant('chr3', 2, 'AT', ''),
          expected_reference='GATA',
          expected_alternate='GACA',
      ),
      dict(
          testcase_name='NoOverlap',
          sequence='GGGG',
          interval=genome.Interval('chr4', 0, 4),
          variant=genome.Variant('chr4', 6, 'T', 'T'),
          expected_reference='GGGG',
          expected_alternate='GGGG',
      ),
      dict(
          testcase_name='NegativeStrand',
          sequence='AAAAA',
          interval=genome.Interval('chr1', 0, 5, '-'),
          variant=genome.Variant('chr1', 2, 'A', 'G'),
          expected_reference='TTTTT',
          expected_alternate='TTTCT',
      ),
  ])
  def test_extract_variant_sequences(
      self,
      sequence: str,
      interval: genome.Interval,
      variant: genome.Variant,
      expected_reference: str,
      expected_alternate: str,
  ):
    def _extract(interval: genome.Interval) -> str:
      self.assertLen(sequence, interval.width, f'{interval=}')
      return sequence

    extractor = mock.create_autospec(fasta.FastaExtractor)
    extractor.extract.side_effect = _extract
    reference_sequence, alternate_sequence = (
        genome_io.extract_variant_sequences(interval, variant, extractor)
    )
    self.assertLen(reference_sequence, interval.width)
    self.assertLen(alternate_sequence, interval.width)
    self.assertEqual(reference_sequence, expected_reference)
    self.assertEqual(alternate_sequence, expected_alternate)


if __name__ == '__main__':
  absltest.main()
