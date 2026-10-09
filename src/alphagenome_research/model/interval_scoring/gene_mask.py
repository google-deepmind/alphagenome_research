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

"""Implementation of gene mask interval scoring."""

from collections.abc import Mapping

from alphagenome import typing
from alphagenome.data import genome
from alphagenome.data import track_data
from alphagenome.models.v1 import dna_output
from alphagenome.models.v1 import interval_scorers
from alphagenome_research.model.interval_scoring import interval_scoring
from alphagenome_research.model.variant_scoring import gene_mask_extractor as gene_mask_extractor_lib
from alphagenome_research.model.variant_scoring import variant_scoring
import anndata
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float32  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np
import pandas as pd


class GeneIntervalScorer(
    interval_scoring.IntervalScorer[
        Bool[np.ndarray | Array, 'M G'],
        pd.DataFrame,
        interval_scorers.GeneMaskScorer,
    ]
):
  """Interval scorer that aggregates intervals across different genes."""

  def __init__(
      self, gene_mask_extractor: gene_mask_extractor_lib.GeneMaskExtractor
  ):
    """Initializes the GeneIntervalScorer.

    Args:
      gene_mask_extractor: Gene mask extractor to use.
    """
    self._gene_mask_extractor = gene_mask_extractor

  def get_masks_and_metadata(
      self,
      interval: genome.Interval,
      *,
      settings: interval_scorers.GeneMaskScorer,
      track_metadata: dna_output.OutputMetadata,
  ) -> tuple[Bool[np.ndarray | Array, 'S G'], pd.DataFrame]:
    """Get gene masks and metadata for the given interval.

    Args:
      interval: Genomic interval to extract gene masks for.
      settings: The variant scorer settings.
      track_metadata: Track metadata for the variant.

    Returns:
      Tuple of (gene variant masks, mask metadata). The mask metadata is the
      part of the GTF pandas dataframe that was used to construct the gene
      masks.
    """
    del track_metadata
    if interval.negative_strand:
      raise ValueError(
          'IntervalScorers do not support negative strands (negative strand'
          ' predictions should already be reverse-complemented prior to'
          ' scoring and thus masks should be generated on the positive strand).'
      )
    if settings.width is not None and settings.width > interval.width:
      raise ValueError('Interval width must be >= the center mask width.')
    resolution = variant_scoring.get_resolution(settings.requested_output)
    target_interval = (
        interval.resize(width=settings.width)
        if settings.width is not None
        else interval
    )

    gene_mask, metadata = self._gene_mask_extractor.extract(target_interval)
    interval_padding = interval.width - target_interval.width
    gene_mask = np.pad(
        gene_mask,
        ((interval_padding // 2, (interval_padding + 1) // 2), (0, 0)),
    )
    if resolution > 1:
      gene_mask = gene_mask.reshape(
          (gene_mask.shape[0] // resolution, resolution, -1)
      ).max(axis=1)
    return gene_mask, metadata

  @typing.jaxtyped
  def score_interval(
      self,
      predictions: Mapping[dna_output.OutputType, Float32[Array, 'S T']],
      *,
      masks: Bool[Array | np.ndarray, 'M G'],
      settings: interval_scorers.GeneMaskScorer,
      interval: genome.Interval | None = None,
  ) -> interval_scoring.ScoreIntervalOutput:
    """See base class."""
    del interval
    tracks = predictions[settings.requested_output]

    match settings.aggregation_type:
      case interval_scorers.IntervalAggregationType.MEAN:
        output = jnp.einsum('lt,lg->gt', tracks, masks) / jnp.expand_dims(
            masks.sum(axis=0), axis=-1
        )
      case interval_scorers.IntervalAggregationType.SUM:
        output = jnp.einsum('lt,lg->gt', tracks, masks)
      case _:
        raise ValueError(
            f'Unsupported aggregation type: {settings.aggregation_type}.'
        )

    return {'score': output}

  def finalize_interval(
      self,
      scores: interval_scoring.ScoreIntervalResult,
      *,
      track_metadata: dna_output.OutputMetadata,
      mask_metadata: pd.DataFrame,
      settings: interval_scorers.GeneMaskScorer,
  ) -> anndata.AnnData:
    """Returns summarized scores for the given scores and metadata."""
    output_metadata = track_metadata.get(settings.requested_output)
    assert isinstance(output_metadata, track_data.TrackMetadata)
    return variant_scoring.create_anndata(
        scores['score'],
        obs=mask_metadata,
        var=output_metadata,
    )
