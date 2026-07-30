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
import pathlib

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import fold_intervals
from alphagenome.data import genome
from alphagenome.models import dna_model
from alphagenome_research.io import bundles
from alphagenome_research.io import dataset
from alphagenome_research.model import schemas
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_tf_example(data):
  """Creates a tf.train.Example message from a data dictionary."""
  features = {}
  for key, value in data.items():
    if isinstance(value, int):
      features[key] = _int64_feature(value)
    elif isinstance(value, str):
      features[key] = _bytes_feature(value.encode('utf-8'))
    elif isinstance(value, np.ndarray):
      features[key] = _bytes_feature(tf.io.serialize_tensor(value).numpy())
    else:
      raise ValueError(f'Unsupported data type for key {key}: {type(value)}')
  return tf.train.Example(
      features=tf.train.Features(feature=features)
  ).SerializeToString()


def _write_tfrecord(path: pathlib.Path, data_list):
  """Writes a list of data to a TFRecord file."""
  path.parent.mkdir(parents=True, exist_ok=True)
  options = tf.io.TFRecordOptions(compression_type='GZIP')
  with tf.io.TFRecordWriter(str(path), options=options) as writer:
    for data in data_list:
      writer.write(to_tf_example(data))


class LoadDataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tmpdir = self.create_tempdir().full_path
    self._num_channels = 7
    self._seq_len = int(2**5)
    self._num_shards = 2
    self._intervals = {
        'chr1': {
            '1': [
                genome.Interval(
                    start=10, end=10 + self._seq_len, chromosome='chr1'
                ),
                genome.Interval(
                    start=100, end=100 + self._seq_len, chromosome='chr1'
                ),
            ],
            '2': [
                genome.Interval(
                    start=200, end=200 + self._seq_len, chromosome='chr1'
                ),
            ],
        },
        'chr3': {
            '1': [
                genome.Interval(
                    start=30, end=30 + self._seq_len, chromosome='chr3'
                ),
            ],
            '2': [
                genome.Interval(
                    start=130, end=130 + self._seq_len, chromosome='chr3'
                ),
                genome.Interval(
                    start=230, end=230 + self._seq_len, chromosome='chr3'
                ),
            ],
        },
    }
    self._organism = dna_model.Organism.HOMO_SAPIENS
    self._fold_split = dna_model.ModelVersion.FOLD_0
    self._subset = fold_intervals.Subset.TRAIN
    self._bundles = [
        bundles.BundleName.ATAC,
        bundles.BundleName.RNA_SEQ,
    ]
    self._create_dummy_files()

  def _create_dummy_files(self):
    """Creates data for two bundles in two chromosomes."""
    for bundle in self._bundles:
      for chromosome, shard_intervals_map in self._intervals.items():
        for shard_idx, intervals in shard_intervals_map.items():
          data_list = []
          for interval in intervals:
            data = {
                'dna_sequence': np.zeros((self._seq_len, 4), dtype=np.float32),
                'interval/chromosome': interval.chromosome,
                'interval/start': interval.start,
                'interval/end': interval.end,
                f'{bundle.value}': np.zeros(
                    (self._seq_len, self._num_channels), dtype=jnp.bfloat16
                ),
                f'{bundle.value}_mask': np.ones(
                    (1, self._num_channels), dtype=bool
                ),
            }
            if bundle == bundles.BundleName.RNA_SEQ:
              data['rna_seq_strand'] = np.zeros(
                  (1, self._num_channels), dtype=np.int32
              )
            data_list.append(data)

          path = (
              pathlib.Path(self.tmpdir)
              / self._fold_split.name
              / self._organism.name
              / self._subset.name
              / bundle.name
              / f'data_{chromosome}_{shard_idx}-{self._num_shards}.gz.tfrecord'
          )
          _write_tfrecord(path, data_list)

  def test_get_tfrecords_df(self):
    df = dataset.get_tfrecords_df(path=self.tmpdir)
    self.assertLen(df, 8)
    self.assertSameElements(df.organism, ['HOMO_SAPIENS'])
    self.assertSameElements(df.bundle, ['ATAC', 'RNA_SEQ'])
    self.assertSameElements(df.fold_split, ['FOLD_0'])
    self.assertSameElements(df.subset, ['TRAIN'])
    self.assertSameElements(df.chromosome, ['chr1', 'chr3'])
    self.assertSameElements(df.shard, [1, 2])

  @parameterized.parameters(True, False)
  def test_create_dataset(self, shuffle_dataset: bool):
    ds = dataset.create_dataset(
        organism=self._organism,
        fold_split=self._fold_split,
        subset=self._subset,
        bundles=self._bundles,
        path=self.tmpdir,
    )
    if shuffle_dataset:
      ds = ds.shuffle(buffer_size=100)
    ds_iterator = ds.as_numpy_iterator()
    intervals = []

    # 6 intervals in total: 3 in chr1, 3 in chr3
    for _ in range(6):
      data = next(ds_iterator)
      self.assertLen(data, 2)  # 2 bundles
      data_atac, data_rna_seq = (
          (data[0], data[1])
          if bundles.BundleName.ATAC.value in data[0]
          else (data[1], data[0])
      )

      self.assertIn('atac', data_atac)
      self.assertIn('rna_seq', data_rna_seq)
      self.assertEqual(
          data_atac['interval/chromosome'], data_rna_seq['interval/chromosome']
      )
      self.assertEqual(
          data_atac['interval/start'], data_rna_seq['interval/start']
      )
      self.assertEqual(data_atac['interval/end'], data_rna_seq['interval/end'])
      intervals.append(
          genome.Interval(
              start=data_atac['interval/start'],
              end=data_atac['interval/end'],
              chromosome=data_atac['interval/chromosome'].decode('utf-8'),
          )
      )
      self.assertEqual(data_atac['atac'].dtype, tf.bfloat16)
      self.assertEqual(data_rna_seq['rna_seq'].dtype, tf.bfloat16)

    def _interval_to_tuple(interval):
      # Convert interval to tuple to check equality.
      return (interval.chromosome, interval.start, interval.end)

    all_intervals = jax.tree.reduce(
        lambda x, y: x + y,
        self._intervals,
        is_leaf=lambda x: isinstance(x, list),
    )
    self.assertEqual(
        set(_interval_to_tuple(i) for i in intervals),
        set(_interval_to_tuple(i) for i in all_intervals),
    )

    with self.subTest('Dataset length is correct'):
      with self.assertRaises(StopIteration):
        next(ds_iterator)

  @parameterized.parameters(
      ([bundles.BundleName.ATAC],),
      ([bundles.BundleName.ATAC, bundles.BundleName.RNA_SEQ],),
  )
  def test_get_numpy_dataset_iterator(self, requested_bundles):
    batch_size = 2
    ds_iterator = dataset.get_numpy_dataset_iterator(
        batch_size=batch_size,
        organism=self._organism,
        model_version=self._fold_split,
        subset=self._subset,
        bundles=requested_bundles,
        path=self.tmpdir,
    )

    num_batches = 0
    for batch, metadata in ds_iterator:
      self.assertIsInstance(batch, schemas.DataBatch)
      self.assertIsNotNone(batch.dna_sequence)
      self.assertIsNotNone(batch.organism_index)
      self.assertEqual(batch.dna_sequence.shape[0], batch_size)
      self.assertEqual(batch.organism_index.shape[0], batch_size)
      for bundle in requested_bundles:
        self.assertIsNotNone(getattr(batch, bundle.value))
        self.assertIsNotNone(getattr(batch, f'{bundle.value}_mask'))
      self.assertLen(metadata, 3)
      self.assertEqual(metadata['interval/start'].shape[0], batch_size)
      num_batches += 1

    self.assertEqual(num_batches, 6 // batch_size)


if __name__ == '__main__':
  absltest.main()
