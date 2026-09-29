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

"""Implementation of contact map variant scorer."""

from alphagenome.data import genome
from alphagenome.data import track_data
from alphagenome.models import dna_output
from alphagenome.models import variant_scorers
from alphagenome_research.model.variant_scoring import center_mask
from alphagenome_research.model.variant_scoring import variant_scoring
import anndata
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np


class ContactMapScorer(variant_scoring.VariantScorer):
  """Implements the contact map variant scoring strategy from Zhou 2022 (Orca).

  Designed for single nucleotide variant (SNV) scoring, where the expected
  effect is local to the variant position.

  Not compatible with indels or structural variants that involve changing more
  than the number of nucleotides in a 2Kb window.

  Citation: As described in the Zhou manuscript:
    https://doi.org/10.1038/s41588-022-01065-4:

  "The disruption impact on local genome interactions is measured by 1-Mb
  structural impact score, which is the average absolute log fold change of
  interactions between the disruption position and all other positions in the
  1-Mb window"

  The Orca scoring strategy is open sourced here (line 120):
  https://github.com/jzhoulab/orca_manuscript/blob/main/virtual_screen/local_interaction_screen.py
  """

  def get_masks_and_metadata(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      settings: variant_scorers.ContactMapScorer,
      track_metadata: dna_output.OutputMetadata,
  ) -> tuple[np.ndarray, None]:
    """See base class."""
    del track_metadata  # Unused.

    resolution = variant_scoring.get_resolution(settings.requested_output)
    mask = center_mask.create_center_mask(
        interval, variant, width=resolution, resolution=resolution
    )

    if mask.sum() > 1:
      raise ValueError(
          'The ContactMapScorer only accepts input variants that affect one bin'
          ' position. However, there is more than one position affected by the'
          ' variant at this bin resolution. This could indicate a malformed'
          ' center mask. Please check `create_center_mask` logic. Debugging'
          f' details: {variant=}, {interval=}, bin width={resolution},'
          f' {mask.sum()=}.'
      )
    elif mask.sum() == 0:
      raise ValueError(
          'The variant does not affect any positions at this bin resolution.'
          f' Debugging details: {variant=}, {interval=}, bin'
          f' width={resolution}, {mask.sum()=}.'
      )
    return mask, None

  def score_variant(
      self,
      ref: variant_scoring.ScoreVariantInput,
      alt: variant_scoring.ScoreVariantInput,
      *,
      masks: Bool[Array, '_ 1'],
      settings: variant_scorers.ContactMapScorer,
      variant: genome.Variant | None = None,
      interval: genome.Interval | None = None,
  ) -> variant_scoring.ScoreVariantOutput:
    del variant, interval  # Unused.
    ref = ref[settings.requested_output]
    alt = alt[settings.requested_output]

    # Mean absolute difference, reduced over contact map rows.
    # Ref, alt shape: [H, W, C]
    # Temps shape: [W, C]
    abs_diff = jnp.abs(alt - ref).mean(axis=0)  # pyrefly: ignore[unsupported-operation]

    # JAX dynamic slicing does not work with transfer_guard.
    with jax.transfer_guard('allow'):
      # Use center mask to select the variant row.
      # Right now, assumes there is a single value.
      # Output shape: [1, C]
      output = abs_diff[jnp.argmax(masks), :]

    return {'score': output}

  def finalize_variant(
      self,
      scores: variant_scoring.ScoreVariantResult,
      *,
      track_metadata: dna_output.OutputMetadata,
      mask_metadata: None,
      settings: variant_scorers.ContactMapScorer,
  ) -> anndata.AnnData:
    """See base class."""
    del mask_metadata  # Unused.
    output_metadata = track_metadata.get(settings.requested_output)
    assert isinstance(output_metadata, track_data.TrackMetadata)

    num_tracks = len(output_metadata)
    return variant_scoring.create_anndata(
        scores['score'][np.newaxis, :num_tracks], obs=None, var=output_metadata
    )
