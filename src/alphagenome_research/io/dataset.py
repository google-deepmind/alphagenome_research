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

"""Reading AlphaGenome data from TFRecords.

This module provides functions to load AlphaGenome datasets from TFRecord files.
Each TFRecord file contains data from a particular bundle (see `bundles.py`)
for a set of intervals.

We recommend using `get_numpy_dataset_iterator` to load the data for a
particular organism, fold split, and subset. The iterator yields batches of
data, where each batch is a tuple of `schemas.DataBatch` and a metadata
dictionary containing the chromosome, start, and end of each interval.
"""

from collections.abc import Iterator, Mapping
import functools
import os
import re
from typing import Any, Sequence

from alphagenome.data import fold_intervals
from alphagenome.models.v1 import dna_model
from alphagenome_research.io import bundles as bundles_lib
from alphagenome_research.model import dna_model as research_dna_model
from alphagenome_research.model import schemas
from etils import epath
import numpy as np
import pandas as pd
import tensorflow as tf

_DEFAULT_PATH = 'gs:///alphagenome-datasets/v1/train/'

BatchMetadata = Mapping[str, Any]

_DNA_SEQUENCE_DTYPE = tf.float32
_DNA_SEQUENCE_FEATURE_SPEC = {
    'dna_sequence': tf.io.FixedLenFeature([], tf.string)
}
_INTERVAL_FEATURE_SPEC = {
    'interval/chromosome': tf.io.FixedLenFeature([], tf.string),
    'interval/start': tf.io.FixedLenFeature([], tf.int64),
    'interval/end': tf.io.FixedLenFeature([], tf.int64),
}
_FILENAME_REGEX = re.compile(
    r'data_(?P<chr>.+)_(?P<shard>\d+)-(?P<num_shards>\d+)\.gz\.tfrecord'
)


def get_tfrecords_df(
    *,
    organism: dna_model.Organism | None = None,
    bundle: bundles_lib.BundleName | None = None,
    fold_split: dna_model.ModelVersion | None = None,
    subset: fold_intervals.Subset | None = None,
    chromosome: str | None = None,
    path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
  """Return a dataframe with metadata about the TFRecord files.

  Args:
    organism: The organism to load. If None, all organisms are loaded.
    bundle: The bundle to load. If None, all bundles are loaded.
    fold_split: The fold split to load. If None, all fold splits are loaded.
    subset: The subset to load. If None, all subsets are loaded.
    chromosome: The chromosome to load. If None, all chromosomes are loaded.
    path: The path to the TFRecord files. If None, the default path is used.
  """
  organism_pattern = organism.name if organism is not None else '*'
  fold_split_pattern = fold_split.name if fold_split is not None else '*'
  subset_pattern = subset.name if subset is not None else '*'
  chromosome_pattern = chromosome if chromosome is not None else '*'
  bundle_pattern = bundle.value.upper() if bundle is not None else '*'
  glob_pattern = '/'.join([
      fold_split_pattern,
      organism_pattern,
      subset_pattern,
      bundle_pattern,
      f'data_{chromosome_pattern}_*-*.gz.tfrecord',
  ])
  tfrecord_paths = epath.Path(path or _DEFAULT_PATH).glob(glob_pattern)

  def _parse_path(tfrecord_path: epath.Path):
    base_name = tfrecord_path.name
    match_ = _FILENAME_REGEX.match(base_name)
    if not match_:
      raise ValueError(f'Could not parse metadata for file: {base_name}')

    parsed = match_.groupdict()
    metadata = {
        'organism': tfrecord_path.parts[-4],
        'bundle': tfrecord_path.parts[-2],
        'fold_split': tfrecord_path.parts[-5],
        'subset': tfrecord_path.parts[-3],
        'chromosome': parsed['chr'],
        'shard': int(parsed['shard']),
        'num_shards': int(parsed['num_shards']),
        'path': str(tfrecord_path),
    }
    return pd.DataFrame(metadata, index=[0])

  if not tfrecord_paths:
    return pd.DataFrame()

  df = pd.concat(
      [_parse_path(epath.Path(p)) for p in tfrecord_paths]
  ).reset_index(drop=True)
  assert isinstance(df, pd.DataFrame)
  return df


def _get_parse_function(bundle: bundles_lib.BundleName):
  """Get parse function for a given output type."""
  feature_spec = (
      _DNA_SEQUENCE_FEATURE_SPEC
      | _INTERVAL_FEATURE_SPEC
      | {
          key: tf.io.FixedLenFeature([], tf.string)
          for key in bundle.get_dtypes().keys()
      }
  )
  output_dtypes = bundle.get_dtypes() | {'dna_sequence': _DNA_SEQUENCE_DTYPE}

  def _parse(proto):
    example = tf.io.parse_single_example(proto, feature_spec)
    for key, dtype in output_dtypes.items():
      example[key] = tf.io.parse_tensor(example[key], dtype)
    return example

  return _parse


def _get_tfrecords_dataset(
    paths: Sequence[str | os.PathLike[str]], bundle: bundles_lib.BundleName
) -> tf.data.Dataset:
  """Returns a dataset for a given output type from a sequence of paths."""
  parser = _get_parse_function(bundle)

  def _get(p):
    ds = tf.data.TFRecordDataset(  # pytype: disable=bad-instantiation
        p, compression_type='GZIP'
    )
    ds = ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

  ds = _get(paths[0])
  for p in paths[1:]:
    ds_next = _get(p)
    ds = ds.concatenate(ds_next)
  return ds


def create_dataset(
    *,
    organism: dna_model.Organism,
    fold_split: dna_model.ModelVersion,
    subset: fold_intervals.Subset,
    bundles: Sequence[bundles_lib.BundleName] | None = None,
    path: str | os.PathLike[str] | None = None,
) -> tf.data.Dataset:
  """Returns AlphaGenome dataset for a given organism, fold and subset.

  Args:
    organism: The organism to load.
    fold_split: The fold split to load.
    subset: The subset to load.
    bundles: The bundles to load. If None, all bundles are loaded.
    path: The path to the TFRecord files. If None, the default path is used.
  """
  bundles: Sequence[bundles_lib.BundleName | None] = bundles or [None]
  records = []
  for bundle in bundles:
    records.append(
        get_tfrecords_df(
            organism=organism,
            bundle=bundle,
            fold_split=fold_split,
            subset=subset,
            path=path,
        )
    )
  df = pd.concat(records)
  num_paths = df.groupby('bundle').agg('path').count()
  if num_paths.nunique() != 1:
    raise ValueError(
        f'Number of TFRecord files per bundle is not the same: {num_paths}'
    )
  dataset_per_bundle = []
  bundles = df['bundle'].unique()
  for bundle in bundles:
    dft = df[df['bundle'] == bundle].sort_values('shard')
    if dft.empty:
      raise ValueError(f'No data found for {bundle=}.')
    dataset_per_bundle.append(
        _get_tfrecords_dataset(
            dft['path'].tolist(), bundles_lib.BundleName(bundle.lower())
        )
    )

  # Zip datasets across bundles.
  return tf.data.Dataset.zip(*dataset_per_bundle)


def _parse_batch(
    element,
    bundles: Sequence[bundles_lib.BundleName] | None,
    organism_index: int,
    batch_size: int,
) -> tuple[schemas.DataBatch, BatchMetadata]:
  """Parses a raw dataset element into Input and Target schemas."""

  if bundles is None:
    bundles = list(bundles_lib.BundleName)
  if len(bundles) == 1:
    element = (element,)

  merged_data = functools.reduce(lambda x, y: x | y, element)
  organism_index: np.ndarray = np.full(
      (batch_size,), organism_index, dtype=np.int32
  )
  metadata = {
      'interval/chromosome': merged_data.pop('interval/chromosome'),
      'interval/start': merged_data.pop('interval/start'),
      'interval/end': merged_data.pop('interval/end'),
  }
  batch = schemas.DataBatch(organism_index=organism_index, **merged_data)
  return batch, metadata


def get_numpy_dataset_iterator(
    *,
    batch_size: int,
    organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
    model_version: dna_model.ModelVersion = dna_model.ModelVersion.FOLD_0,
    subset: fold_intervals.Subset = fold_intervals.Subset.VALID,
    bundles: Sequence[bundles_lib.BundleName] | None = None,
    path: str | os.PathLike[str] | None = None,
) -> Iterator[tuple[schemas.DataBatch, BatchMetadata]]:
  """Yields numpy batches of data from the dataset."""
  ds = create_dataset(
      organism=organism,
      fold_split=model_version,
      subset=subset,
      bundles=bundles,
      path=path,
  )
  organism_index = research_dna_model.convert_to_organism_index(organism)
  ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
  for element in ds.as_numpy_iterator():
    yield _parse_batch(element, bundles, organism_index, batch_size)
