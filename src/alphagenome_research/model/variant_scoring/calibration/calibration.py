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

"""Module for loading and applying calibration to variant scorers."""

from collections.abc import Mapping
import dataclasses
import os

from alphagenome import typing
from alphagenome.models.v1 import track_data_utils
from alphagenome_research.protos import calibration_scores_pb2
import anndata
import chex
from etils import epath
from jaxtyping import Bool, Float  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np
import pandas as pd


@typing.jaxtyped
@dataclasses.dataclass(frozen=True, kw_only=True)
class VariantScorerCalibration:
  """Data class for variant scorer calibration data."""

  metadata: pd.DataFrame
  quantiles: Float[np.ndarray, 'T Q']
  quantile_probabilities: Float[np.ndarray, 'Q']
  has_duplicate_quantiles: Bool[np.ndarray, 'T']

  @classmethod
  def from_proto(
      cls,
      calibration: calibration_scores_pb2.VariantScorerCalibration,
  ) -> 'VariantScorerCalibration':
    """Reads variant scorer calibration data from a proto."""
    quantile_probabilities = np.asarray(
        calibration.quantile_probabilities, dtype=np.float32
    )
    quantiles = np.asarray(calibration.quantiles, dtype=np.float32).reshape(
        -1, len(quantile_probabilities)
    )
    metadata = track_data_utils.metadata_from_proto(calibration.tracks_metadata)
    track_metadata = pd.DataFrame({
        'name': metadata['name'],
        'strand': metadata['strand'],
        'ontology_curie': metadata.get('ontology_curie', ''),
    })

    track_metadata.index = metadata.index.map(str)

    has_duplicate_quantiles = np.any(np.diff(quantiles) == 0, axis=1)
    return VariantScorerCalibration(
        metadata=track_metadata,
        quantiles=quantiles,
        quantile_probabilities=quantile_probabilities,
        has_duplicate_quantiles=has_duplicate_quantiles,
    )


class CalibrationScorer:
  """Class for loading and applying calibration to variant scores."""

  def __init__(
      self,
      variant_scorer_to_calibration: Mapping[str, VariantScorerCalibration],
      *,
      rng: np.random.Generator | None = None,
  ):
    self._rng = rng or np.random.default_rng(seed=42)
    self._variant_scorer_to_calibration = variant_scorer_to_calibration

  def has_variant_scorer(self, scorer_name: str) -> bool:
    """Returns True if the variant scorer has calibration data."""
    return scorer_name in self._variant_scorer_to_calibration

  def scorer_metadata(self, scorer_name: str) -> pd.DataFrame:
    """Returns the calibration metadata for the variant scorer."""
    return self._variant_scorer_to_calibration[scorer_name].metadata

  def quantile_values(self, scorer_name: str) -> Float[np.ndarray, 'T Q']:
    """Returns the quantile values for the variant scorer."""
    return self._variant_scorer_to_calibration[scorer_name].quantiles

  def quantile_probabilities(self, scorer_name: str) -> Float[np.ndarray, 'Q']:
    """Returns the quantile probabilities for the variant scorer."""
    return self._variant_scorer_to_calibration[
        scorer_name
    ].quantile_probabilities

  def quantile_scores(
      self,
      scorer_name: str,
      scores: anndata.AnnData,
      *,
      validate: bool = True,
      break_quantile_ties: bool = True,
      seed: int | None = None,
  ) -> np.ndarray:
    """Compute quantile scores from raw scores for a given variant scorer."""
    variant_scorer_calibration = self._variant_scorer_to_calibration.get(
        scorer_name
    )
    if variant_scorer_calibration is None:
      raise ValueError(
          f'No calibration data found for variant scorer: {scorer_name}.'
      )

    if scores.is_view:
      raise ValueError("Quantile scores doesn't support AnnData views.")

    rng = self._rng
    if seed is not None:
      rng = np.random.default_rng(seed=seed)

    if validate:
      if not {'name', 'strand'}.issubset(scores.var.columns):
        raise ValueError(
            'Variant scorer metadata must contain "name" and "strand" columns.'
        )
      track_metadata = pd.DataFrame({
          'name': scores.var['name'],
          'strand': scores.var['strand'],
          'ontology_curie': scores.var.get('ontology_curie', ''),
      })
      track_metadata.index = scores.var.index.map(str)

      if np.any(
          track_metadata[['name', 'strand', 'ontology_curie']]
          != variant_scorer_calibration.metadata
      ):
        raise ValueError(
            f'Variant scorer "{scorer_name}" metadata does not match'
            ' calibration scores.'
        )

    predictions = scores.X.transpose()

    scorer_quantiles = variant_scorer_calibration.quantiles
    quantile_probabilities = variant_scorer_calibration.quantile_probabilities
    duplicate_quantiles = variant_scorer_calibration.has_duplicate_quantiles
    chex.assert_shape(predictions, (scorer_quantiles.shape[0], None))

    track_quantile_scores = np.empty_like(predictions, dtype=np.float32)
    for i, values in enumerate(predictions):
      indices = np.searchsorted(scorer_quantiles[i], values, side='left')
      if break_quantile_ties and duplicate_quantiles[i]:
        end_indices = np.searchsorted(scorer_quantiles[i], values, side='right')
        indices = rng.integers(indices, end_indices, endpoint=True)

      indices = np.minimum(indices, len(quantile_probabilities) - 1)
      track_quantile_scores[i] = quantile_probabilities[indices]
      # Always return NaN if the score is NaN.
      track_quantile_scores[i][np.isnan(values)] = np.nan

    return track_quantile_scores.transpose()


def load(
    path: str | os.PathLike[str], *, rng: np.random.Generator | None = None
) -> CalibrationScorer:
  """Reads calibration scores from a path."""
  scores = calibration_scores_pb2.CalibrationScores.FromString(
      epath.Path(path).read_bytes()
  )
  return CalibrationScorer(
      {
          k: VariantScorerCalibration.from_proto(v)
          for k, v in scores.scorer_to_calibration.items()
      },
      rng=rng,
  )
