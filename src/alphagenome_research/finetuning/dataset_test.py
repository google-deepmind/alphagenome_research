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
from alphagenome.models import dna_client
from alphagenome_research.finetuning import dataset
from alphagenome_research.io import fasta
from alphagenome_research.model.metadata import metadata as metadata_lib
import chex
import numpy as np
import pandas as pd
import pyBigWig
import tensorflow as tf

MOCK_CHROM_SIZES = {'chr1': 100}


def get_mock_metadata(num_atac_tracks: int = 2, num_dnase_tracks: int = 1):
  return metadata_lib.AlphaGenomeOutputMetadata(
      atac=pd.DataFrame({
          'file_path': [f'path{i}' for i in range(num_atac_tracks)],
          'name': [f'atac_track{i}' for i in range(num_atac_tracks)],
          'strand': ['.'] * num_atac_tracks,
      }),
      dnase=pd.DataFrame({
          'file_path': [f'path{i}' for i in range(num_dnase_tracks)],
          'name': [f'dnase_track{i}' for i in range(num_dnase_tracks)],
          'strand': ['.'] * num_dnase_tracks,
      }),
  )


def get_mock_bw():
  mock_bw = mock.Mock()
  mock_bw.chroms.return_value = MOCK_CHROM_SIZES

  def mock_values(chrom, start, end, numpy=True):
    del chrom, numpy
    return np.full(end - start, 1.0)

  mock_bw.values = mock_values
  return mock_bw


class MockFastaExtractor:

  def __init__(self, path):
    pass

  def extract(self, interval):
    return 'A' * interval.width


class DataTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='valid_interval',
          interval=genome.Interval('chr1', 10, 20),
          expected_values=np.ones(10),
      ),
      dict(
          testcase_name='interval_start_negative',
          interval=genome.Interval('chr1', -10, 10),
          expected_values=np.concatenate([np.zeros(10), np.ones(10)]),
      ),
      dict(
          testcase_name='interval_end_beyond_chrom_length',
          interval=genome.Interval('chr1', 90, 110),
          expected_values=np.concatenate([np.ones(10), np.zeros(10)]),
      ),
      dict(
          testcase_name='interval_fully_outside_negative',
          interval=genome.Interval('chr1', -20, -10),
          expected_values=np.zeros(10),
      ),
      dict(
          testcase_name='interval_fully_outside_positive',
          interval=genome.Interval('chr1', 110, 120),
          expected_values=np.zeros(10),
      ),
      dict(
          testcase_name='zero_length_interval',
          interval=genome.Interval('chr1', 10, 10),
          expected_values=np.array([]),
      ),
  ])
  @mock.patch.object(pyBigWig, 'open')
  def test_bigwig_extractor(self, mock_bigwig_open, interval, expected_values):
    mock_bigwig_open.return_value = get_mock_bw()
    extractor = dataset.BigWigExtractor('mock_path')
    values = extractor.extract(interval)
    self.assertEqual(values.shape, expected_values.shape)
    np.testing.assert_array_equal(values, expected_values)
    extractor.close()
    mock_bigwig_open.assert_called_once_with('mock_path')

  @mock.patch.object(pyBigWig, 'open')
  def test_bigwig_extractor_wrong_chromosome(self, mock_bigwig_open):
    mock_bigwig_open.return_value = get_mock_bw()
    extractor = dataset.BigWigExtractor('mock_path')
    with self.assertRaisesRegex(ValueError, 'Chromosome chr2 not found'):
      extractor.extract(genome.Interval('chr2', 0, 10))
    extractor.close()
    mock_bigwig_open.assert_called_once_with('mock_path')

  @mock.patch.object(pyBigWig, 'open')
  def test_multi_track_extractor(self, mock_bigwig_open):
    mock_bigwig_open.return_value = get_mock_bw()
    extractor = dataset.MultiTrackExtractor(
        output_metadata=get_mock_metadata(
            num_atac_tracks=3, num_dnase_tracks=4
        ),
        sequence_length=10,
    )
    interval = genome.Interval('chr1', 10, 20)
    result = extractor.extract(interval)

    chex.assert_shape(result['atac'], (10, 3))
    self.assertEqual(result['atac'].dtype, tf.bfloat16)
    chex.assert_shape(result['atac_mask'], (1, 3))
    self.assertEqual(result['atac_mask'].dtype, tf.bool)
    chex.assert_shape(result['dnase'], (10, 4))
    chex.assert_shape(result['dnase_mask'], (1, 4))
    extractor.close()
    self.assertEqual(mock_bigwig_open.call_count, 7)

  @parameterized.named_parameters([
      dict(testcase_name='one_epoch', num_epochs=1),
      dict(testcase_name='three_epochs', num_epochs=3),
  ])
  @mock.patch.object(fasta, 'FastaExtractor', MockFastaExtractor)
  @mock.patch.object(pyBigWig, 'open')
  def test_data_pipeline(self, mock_bigwig_open, num_epochs):
    mock_bigwig_open.return_value = get_mock_bw()
    intervals = pd.DataFrame({
        'chromosome': ['chr1', 'chr1', 'chr1'],
        'start': [10, 50, 90],
        'end': [30, 70, 110],
    })
    sequence_length = 30
    pipeline = dataset.DataPipeline(
        fasta_path='mock_fasta',
        intervals=intervals,
        output_metadata=get_mock_metadata(
            num_atac_tracks=2, num_dnase_tracks=1
        ),
        organism=dna_client.Organism.HOMO_SAPIENS,
        sequence_length=sequence_length,
    )

    generator = pipeline.get_generator(num_epochs=num_epochs, shuffle=False)
    elements = list(generator)
    self.assertLen(elements, len(intervals) * num_epochs)

    element = elements[0]
    chex.assert_shape(element['dna_sequence'], (sequence_length, 4))
    np.testing.assert_array_equal(
        element['dna_sequence'],
        np.tile(np.array([[1, 0, 0, 0]]), (sequence_length, 1)),  # 'A'
    )
    self.assertEqual(element['organism_index'], 0)
    bundles = element['bundles']
    chex.assert_shape(bundles['atac'], (sequence_length, 2))
    chex.assert_shape(bundles['dnase'], (sequence_length, 1))

    # The third interval (non shuffled) is half out of bounds.
    element = elements[2]
    bundles = element['bundles']
    expected_track = np.concatenate([np.ones(15), np.zeros(15)])
    np.testing.assert_array_equal(
        bundles['atac'], np.stack([expected_track] * 2, axis=-1)
    )
    np.testing.assert_array_equal(
        bundles['dnase'], expected_track.reshape(-1, 1)
    )

    pipeline.close()
    self.assertEqual(mock_bigwig_open.call_count, 3)


if __name__ == '__main__':
  absltest.main()
