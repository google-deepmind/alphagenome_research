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

"""Implementation of center mask variant scoring."""

import functools
import math

from alphagenome import typing
from alphagenome.data import genome
from alphagenome.data import track_data
from alphagenome.models import dna_output
from alphagenome.models import variant_scorers
from alphagenome_research.model.variant_scoring import variant_scoring
import anndata
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float32  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np


def create_center_mask(
    interval: genome.Interval,
    variant: genome.Variant,
    *,
    width: int | None,
    resolution: int,
) -> Bool[np.ndarray, 'S 1']:
  """Creates a mask centered on a variant for a given interval."""
  if width is None:
    if interval.start <= variant.start < interval.end:
      mask = np.ones([interval.width // resolution, 1], dtype=bool)
    else:
      mask = np.zeros([interval.width // resolution, 1], dtype=bool)
  else:
    target_resolution_width = math.ceil(width / resolution)

    # Determine the position of the variant in the specified resolution.
    base_resolution_center = variant.start - interval.start
    target_resolution_center = base_resolution_center // resolution

    # Compute start and end indices of the variant-centered mask.
    target_resolution_start = max(
        target_resolution_center - target_resolution_width // 2, 0
    )
    target_resolution_end = min(
        (target_resolution_center - target_resolution_width // 2)
        + target_resolution_width,
        interval.width // resolution,
    )

    # If the variant is not within the interval, we return an empty mask.
    # Otherwise, we build the mask using our target_resolution_start/end,
    # taking into account the resolution.
    mask = np.zeros([interval.width // resolution, 1], dtype=bool)
    if interval.start <= variant.start < interval.end:
      mask[target_resolution_start:target_resolution_end] = 1

  return mask


@functools.partial(jax.jit, static_argnames=['aggregation_type'])
@typing.jaxtyped
def _apply_aggregation(
    ref: Float32[Array, 'S T'],
    alt: Float32[Array, 'S T'],
    masks: Bool[Array, 'S 1'],
    aggregation_type: variant_scorers.AggregationType,
):
  match aggregation_type:
    case variant_scorers.AggregationType.DIFF_MEAN:
      return alt.mean(axis=0, where=masks) - ref.mean(axis=0, where=masks)
    case variant_scorers.AggregationType.ACTIVE_MEAN:
      return jnp.maximum(
          alt.mean(axis=0, where=masks),
          ref.mean(axis=0, where=masks),
      )
    case variant_scorers.AggregationType.DIFF_SUM:
      return alt.sum(axis=0, where=masks) - ref.sum(axis=0, where=masks)
    case variant_scorers.AggregationType.ACTIVE_SUM:
      return jnp.maximum(
          alt.sum(axis=0, where=masks),
          ref.sum(axis=0, where=masks),
      )
    case variant_scorers.AggregationType.L2_DIFF:
      return jnp.sqrt(jnp.sum((alt - ref) ** 2, axis=0, where=masks))
    case variant_scorers.AggregationType.L2_DIFF_LOG1P:
      return jnp.sqrt(
          jnp.sum(
              (jnp.log1p(alt) - jnp.log1p(ref)) ** 2,
              axis=0,
              where=masks,
          )
      )
    case variant_scorers.AggregationType.DIFF_SUM_LOG2:
      return jnp.sum(jnp.log2(alt + 1), axis=0, where=masks) - jnp.sum(
          jnp.log2(ref + 1), axis=0, where=masks
      )
    case variant_scorers.AggregationType.DIFF_LOG2_SUM:
      return jnp.log2(1 + jnp.sum(alt, axis=0, where=masks)) - jnp.log2(
          1 + jnp.sum(ref, axis=0, where=masks)
      )
    case _:
      raise ValueError(f'Unknown bin aggregation type: {aggregation_type}.')


class CenterMaskVariantScorer(variant_scoring.VariantScorer):
  """Variant scorer that aggregates ALT - REF in a window around the variant."""

  def get_masks_and_metadata(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      settings: variant_scorers.CenterMaskScorer,
      track_metadata: dna_output.OutputMetadata,
  ) -> tuple[np.ndarray, None]:
    """See base class."""
    del track_metadata

    resolution = variant_scoring.get_resolution(settings.requested_output)
    mask = create_center_mask(
        interval, variant, width=settings.width, resolution=resolution
    )

    return mask, None

  @typing.jaxtyped
  def score_variant(
      self,
      ref: variant_scoring.ScoreVariantInput,
      alt: variant_scoring.ScoreVariantInput,
      *,
      masks: Bool[Array, '_ 1'],
      settings: variant_scorers.CenterMaskScorer,
      variant: genome.Variant | None = None,
      interval: genome.Interval | None = None,
  ) -> variant_scoring.ScoreVariantOutput:
    """See base class."""
    del variant, interval  # Unused.
    alt = alt[settings.requested_output]
    ref = ref[settings.requested_output]

    output = _apply_aggregation(ref, alt, masks, settings.aggregation_type)
    return {'score': output}

  def finalize_variant(
      self,
      scores: variant_scoring.ScoreVariantResult,
      *,
      track_metadata: dna_output.OutputMetadata,
      mask_metadata: None,
      settings: variant_scorers.CenterMaskScorer,
  ) -> anndata.AnnData:
    """See base class."""
    del mask_metadata  # Unused.
    output_metadata = track_metadata.get(settings.requested_output)
    assert isinstance(output_metadata, track_data.TrackMetadata)

    num_tracks = len(output_metadata)
    return variant_scoring.create_anndata(
        scores['score'][np.newaxis, :num_tracks],
        obs=None,
        var=output_metadata,
    )
