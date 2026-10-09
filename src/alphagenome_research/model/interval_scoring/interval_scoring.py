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

"""Base class for interval scorers."""

import abc
from collections.abc import Mapping
from typing import Generic, TypeVar

from alphagenome.data import genome
from alphagenome.models.v1 import dna_output
import anndata
import jax
from jaxtyping import Array, Float32  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np

IntervalMaskT = TypeVar('IntervalMaskT')
IntervalMetadataT = TypeVar('IntervalMetadataT')
IntervalSettingsT = TypeVar('IntervalSettingsT')

ScoreIntervalOutput = Mapping[str, jax.Array | np.ndarray]
ScoreIntervalResult = Mapping[str, np.ndarray]


class IntervalScorer(
    Generic[IntervalMaskT, IntervalMetadataT, IntervalSettingsT],
    metaclass=abc.ABCMeta,
):
  """Abstract class for interval scorers."""

  @abc.abstractmethod
  def get_masks_and_metadata(
      self,
      interval: genome.Interval,
      *,
      settings: IntervalSettingsT,
      track_metadata: dna_output.OutputMetadata,
  ) -> tuple[IntervalMaskT, IntervalMetadataT]:
    """Returns masks and metadata for the given interval and metadata.

    The generated masks and metadata will be passed to `score_interval` and
    `finalize_interval` respectively.

    Args:
      interval: The interval to score.
      settings: The interval scorer settings.
      track_metadata: The model's track metadata.

    Returns:
      A tuple of (masks, metadata), where:
        masks: The masks required to score the interval, such as gene or TSS or
          strand masks. These will be passed into the jitted `score_interval`
          function.
        metadata: The metadata required to finalize the interval. These will
          be passed into the `finalize_interval` function.

      The formats/shapes of masks and metadata will vary across interval scorers
      depending on their individual needs.
    """

  @abc.abstractmethod
  def score_interval(
      self,
      predictions: Mapping[dna_output.OutputType, Float32[Array, 'S T']],
      *,
      masks: IntervalMaskT,
      settings: IntervalSettingsT,
      interval: genome.Interval | None = None,
  ) -> ScoreIntervalOutput:
    """Generates a score per track for the provided predictions.

    Args:
      predictions: Model predictions for the interval.
      masks: The masks for scoring the interval.
      settings: The interval scorer settings.
      interval: The interval to score.

    Returns:
      Dictionary of scores to be passed to `finalize_interval`.
    """

  @abc.abstractmethod
  def finalize_interval(
      self,
      scores: ScoreIntervalResult,
      *,
      track_metadata: dna_output.OutputMetadata,
      mask_metadata: IntervalMetadataT,
      settings: IntervalSettingsT,
  ) -> anndata.AnnData:
    """Returns finalized scores for the given scores and metadata.

    Args:
      scores: Dictionary of scores generated from `score_interval` function.
      track_metadata: Metadata describing the tracks for each output_type.
      mask_metadata: Metadata describing the masks.
      settings: The interval scorer settings.

    Returns:
      An AnnData object containing the final interval outputs. The entries will
      vary across scorers depending on their individual needs.
    """
