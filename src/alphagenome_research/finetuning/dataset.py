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
"""Data pipeline for reading sequence and tracks for fine-tuning."""

from collections.abc import Iterator, Sequence
import concurrent.futures
from typing import Any, Mapping

from absl import logging
from alphagenome.data import genome
from alphagenome.models import dna_model
from alphagenome.models import dna_output
from alphagenome_research.io import fasta
from alphagenome_research.model import one_hot_encoder
from alphagenome_research.model.metadata import metadata as metadata_lib
import numpy as np
import pandas as pd
import pyBigWig
import tensorflow as tf


class BigWigExtractor:
  """BigWig file extractor using pyBigWig."""

  def __init__(self, file_path: str):
    self._bw = pyBigWig.open(file_path)
    self._chromosomes = set(self._bw.chroms().keys())

  def __del__(self):
    self.close()

  @property
  def chromosomes(self) -> set[str]:
    return self._chromosomes

  def extract(self, interval: genome.Interval) -> tf.Tensor:
    """Extracts values from a BigWig file for a given interval."""
    if interval.chromosome not in self._chromosomes:
      raise ValueError(
          f'Chromosome {interval.chromosome} not found in BigWig. '
          'Check self._bw.chroms() for available chromosomes. '
      )
    start = max(0, interval.start)
    end = min(self._bw.chroms()[interval.chromosome], interval.end)

    if end <= start:
      return tf.cast(np.zeros(interval.width), tf.bfloat16)

    values = self._bw.values(interval.chromosome, start, end, numpy=True)
    if values.shape[0] != interval.width:
      pad_left = start - interval.start
      pad_right = interval.end - end
      values = np.pad(
          values, pad_width=(pad_left, pad_right), constant_values=0
      )
    return tf.cast(np.nan_to_num(values, nan=0.0), tf.bfloat16)

  def close(self):
    if self._bw is not None:
      self._bw.close()


class MultiTrackExtractor:
  """Multi-track BigWig file extractor.

  Returns tracks per output type. Tracks are ordered as per the metadata.
  """

  def __init__(
      self,
      output_metadata: metadata_lib.AlphaGenomeOutputMetadata,
      sequence_length: int,
      max_workers=32,
  ):
    """Initializes the MultiTrackExtractor.

    Args:
      output_metadata: The output metadata containing the track information.
      sequence_length: The length of the sequence to extract.
      max_workers: The maximum number of workers to use for parallel extraction.
    """
    self._output_metadata = output_metadata
    self._sequence_length = sequence_length
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    )

    # group bigwig extractors by output type.
    self._bw_extractors: dict[
        dna_output.OutputType, Sequence[BigWigExtractor]
    ] = {}
    for output_type in dna_output.OutputType:
      if (metadata := output_metadata.get(output_type)) is not None:
        self._bw_extractors[output_type] = [
            BigWigExtractor(file_path) for file_path in metadata['file_path']
        ]

    self._track_masks = {
        f'{output_type.name.lower()}_mask': np.logical_not(mask).reshape(1, -1)
        for output_type, mask in output_metadata.padding.items()
    }

  def get_output_signature(self):
    """Returns the output signature of the dataset."""
    signature = {}
    for output_type, extractors in self._bw_extractors.items():
      num_tracks = len(extractors)
      output_name = output_type.name.lower()
      signature[output_name] = tf.TensorSpec(
          shape=(self._sequence_length, num_tracks),
          dtype=tf.bfloat16,
      )
      signature[f'{output_name}_mask'] = tf.TensorSpec(
          shape=(1, num_tracks),
          dtype=tf.bool,
      )
    return signature

  def extract(self, interval: genome.Interval) -> Mapping[str, tf.Tensor]:
    """Extracts all tracks for all groups in parallel."""

    # Submit
    future_map = {}
    for output_type, extractors in self._bw_extractors.items():
      future_map[output_type] = [
          self._executor.submit(bw.extract, interval) for bw in extractors
      ]

    # Collect
    data = {}
    for output_type, futures in future_map.items():
      try:
        results = [f.result() for f in futures]
        data[output_type.name.lower()] = np.stack(results, axis=-1)
      except Exception as e:
        raise RuntimeError(
            f"Failed to extract tracks for output group '{output_type.name}'"
        ) from e
    return data | self._track_masks

  def close(self):
    for bw_extractor in self._bw_extractors.values():
      for bw in bw_extractor:
        bw.close()


class DataPipeline:
  """Data pipeline for reading sequence and tracks."""

  def __init__(
      self,
      *,
      fasta_path: str,
      intervals: pd.DataFrame,
      output_metadata: metadata_lib.AlphaGenomeOutputMetadata,
      organism: dna_model.Organism,
      sequence_length: int,
  ):
    if organism != dna_model.Organism.HOMO_SAPIENS:
      raise NotImplementedError('Only HOMO_SAPIENS is currently supported.')

    self._fasta_extractor = fasta.FastaExtractor(fasta_path)
    self._intervals = intervals
    self._organsim = organism
    self._organism_index = 0  # Only one organism is supported.
    self._sequence_length = sequence_length
    self._one_hot_encoder = one_hot_encoder.DNAOneHotEncoder()
    self._multi_track_extractor = MultiTrackExtractor(
        output_metadata=output_metadata,
        sequence_length=sequence_length,
    )

  def get_element(self, idx: int) -> Mapping[str, Any]:
    """Returns a single element of the dataset."""
    interval = self._intervals.iloc[idx]
    interval = genome.Interval(
        chromosome=interval['chromosome'],
        start=int(interval['start']),
        end=int(interval['end']),
    )
    interval = interval.resize(self._sequence_length)
    seq_str = self._fasta_extractor.extract(interval)
    seq_one_hot = self._one_hot_encoder.encode(seq_str)
    track_bundles = self._multi_track_extractor.extract(interval)
    return {
        'dna_sequence': seq_one_hot,
        'organism_index': self._organism_index,
        'bundles': track_bundles,
    }

  def get_output_signature(self):
    """Returns the output signature of the dataset."""
    return {
        'dna_sequence': tf.TensorSpec(
            shape=(self._sequence_length, 4), dtype=tf.float32
        ),
        'organism_index': tf.TensorSpec(shape=[], dtype=tf.int32),
        'bundles': self._multi_track_extractor.get_output_signature(),
    }

  def get_generator(
      self, num_epochs: int = -1, shuffle: bool = True, seed: int = 0
  ) -> Iterator[Mapping[str, Any]]:
    """Returns a generator for the dataset."""
    rng = np.random.default_rng(seed=seed)
    num_epochs = num_epochs if num_epochs > 0 else float('inf')
    epoch_idx = 0
    while epoch_idx < num_epochs:
      epoch_idx += 1
      if shuffle:
        self._intervals = self._intervals.sample(
            frac=1, random_state=rng
        ).reset_index(drop=True)
      for idx in range(len(self._intervals)):
        try:
          yield self.get_element(idx)
        except Exception as e:  # pylint: disable=broad-except
          logging.warning(
              'Failed to get interval %s. With error: %s',
              self._intervals.iloc[idx].to_dict(),
              e,
          )

  def close(self):
    self._multi_track_extractor.close()
