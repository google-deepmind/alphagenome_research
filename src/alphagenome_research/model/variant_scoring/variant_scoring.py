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

"""Base class for variant scorers."""

import abc
from collections.abc import Mapping
from typing import Generic, TypeVar

from alphagenome import typing
from alphagenome.data import genome
from alphagenome.models.v1 import dna_output
from alphagenome_research.typing import ArrayType  # pylint: disable=g-importing-member
import anndata
import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float32, Int32, PyTree  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np
import pandas as pd

VariantMaskT = TypeVar('VariantMaskT')
VariantMetadataT = TypeVar('VariantMetadataT')
VariantSettingsT = TypeVar('VariantSettingsT')

ScoreVariantOutput = Mapping[str, jax.Array | np.ndarray]
ScoreVariantResult = Mapping[str, np.ndarray]

ScoreVariantInput = Mapping[
    dna_output.OutputType, PyTree[Float32[Array, '...'] | Int32[Array, '...']]
]


@typing.jaxtyped
def create_anndata(
    scores: Float32[np.ndarray, 'G T'],
    *,
    obs: pd.DataFrame | None,
    var: pd.DataFrame,
) -> anndata.AnnData:
  """Helper function for creating AnnData objects."""
  var = var.copy()
  # We explicitly cast the dataframe indices to str to avoid
  # ImplicitModificationWarning being logged over and over again.
  var.index = var.index.map(str)

  if obs is not None:
    obs = obs.copy()
    obs.index = obs.index.map(str).astype(str)
  return anndata.AnnData(np.ascontiguousarray(scores), obs=obs, var=var)


def get_resolution(output_type: dna_output.OutputType):
  match output_type:
    case dna_output.OutputType.ATAC:
      return 1
    case dna_output.OutputType.CAGE:
      return 1
    case dna_output.OutputType.DNASE:
      return 1
    case dna_output.OutputType.RNA_SEQ:
      return 1
    case dna_output.OutputType.CHIP_HISTONE:
      return 128
    case dna_output.OutputType.CHIP_TF:
      return 128
    case dna_output.OutputType.SPLICE_SITES:
      return 1
    case dna_output.OutputType.SPLICE_SITE_USAGE:
      return 1
    case dna_output.OutputType.SPLICE_JUNCTIONS:
      return 1
    case dna_output.OutputType.CONTACT_MAPS:
      return 2048
    case dna_output.OutputType.PROCAP:
      return 1
    case _:
      raise ValueError(f'Unknown output type: {output_type}.')


class VariantScorer(
    Generic[VariantMaskT, VariantMetadataT, VariantSettingsT],
    metaclass=abc.ABCMeta,
):
  """Abstract class for variant scorers."""

  @abc.abstractmethod
  def get_masks_and_metadata(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      settings: VariantSettingsT,
      track_metadata: dna_output.OutputMetadata,
  ) -> tuple[VariantMaskT, VariantMetadataT]:
    """Returns masks and metadata for the given interval, variant and metadata.

    The generated masks and metadata will be passed to `score_variant` and
    `finalize_variant` respectively.

    Args:
      interval: The interval to score.
      variant: The variant to extract the masks/metadata for.
      settings: The variant scorer settings.
      track_metadata: The track metadata required to finalize the variant. These
        will be passed into the `finalize_variants` function.

    Returns:
      A tuple of (masks, metadata), where:
        masks: The masks required to score the variant, such as gene or TSS or
          strand masks. These will be passed into the jitted `score_variants`
          function.
        metadata: The metadata required to finalize the variant. These will
          be passed into the `finalize_variants` function.

      The formats/shapes of masks and metadata will vary across variant scorers
      depending on their individual needs.
    """

  @abc.abstractmethod
  def score_variant(
      self,
      ref: ScoreVariantInput,
      alt: ScoreVariantInput,
      *,
      masks: VariantMaskT,
      settings: VariantSettingsT,
      variant: genome.Variant | None = None,
      interval: genome.Interval | None = None,
  ) -> ScoreVariantOutput:
    """Generates a score per track for the provided ref/alt predictions.

    Args:
      ref: Reference predictions.
      alt: Alternative predictions.
      masks: The masks for scoring the variant.
      settings: The variant scorer settings.
      variant: The variant to score.
      interval: The interval to score.

    Returns:
      Dictionary of scores to be passed to `finalize_variant`.
    """

  @abc.abstractmethod
  def finalize_variant(
      self,
      scores: ScoreVariantResult,
      *,
      track_metadata: dna_output.OutputMetadata,
      mask_metadata: VariantMetadataT,
      settings: VariantSettingsT,
  ) -> anndata.AnnData:
    """Returns finalized scores for the given scores and metadata.

    Args:
      scores: Dictionary of scores generated from `score_variant` function.
      track_metadata: Metadata describing the tracks for each output_type.
      mask_metadata: Metadata describing the masks.
      settings: The variant scorer settings.

    Returns:
      A VariantOutputType object containing the final variant outputs. The
      entries will vary across scorers depending on their individual needs.
    """


@typing.jaxtyped
@chex.dataclass(frozen=True)
class IndelMask:
  """Masks for aligning ALT with REF predictions in the presence of indels.

  This is used both in variant scoring and variant predictions: the former not
  having a batch dimension and the latter having one.
  """

  variant_is_indel: Bool[ArrayType, '*B 1']
  variant_alt_mask: Bool[ArrayType, '*B S']
  variant_deletion_reindex_mask: Int32[ArrayType, '*B S']
  variant_deletion_zeros_mask: Bool[ArrayType, '*B S']
  variant_insertion_crop_mask: Bool[ArrayType, '*B S']
  variant_insertion_reindex_mask: Int32[ArrayType, '*B S']

  @classmethod
  def from_variant(
      cls,
      variant: genome.Variant,
      interval: genome.Interval,
  ) -> 'IndelMask':
    """Returns the indel alignment masks for the given variant and interval.

    Args:
      variant: A genome.Variant.
      interval: A genome.Interval.

    Deletions:
      - We insert zeroes in the ALT at the rightmost positions of the variant,
      to align with the positions that were deleted from the REF. To do this in
      jit, we set the final N predictions to zeros and reindex the ALT to move
      them to the rightmost positions in the ALT.

    Insertions:
      - We set the rightmost variant positions in the ALT to the maximum of
      those positions, then collapse into a single prediction to align with the
      rightmost prediction in the REF. To do this in jit, we reindex all but 1
      of the rightmost predictions to the end, to be filtered later.

    We assume that insertions/deletions always occur at the end of the variant;
    as a result, this algorithm only applies to left-aligned variants. If
    variants are unnormalized or multi-change indels, alignment is not reliable.
    """

    # Extract variant masks. We need several of those so that `score_variant`
    # can be written in a jittable way and can handle indels.
    insertion_length = len(variant.alternate_bases) - len(
        variant.reference_bases
    )
    deletion_length = -insertion_length
    variant_start_ = variant.start - interval.start
    # We only realign the insertion/deletion portion of the variant, which we
    # assume is left-aligned such that the indel portion is at the end.
    variant_start_ += (
        min(len(variant.reference_bases), len(variant.alternate_bases)) - 1
    )

    # Indicator mask for the alternate insertion bases to maximize over.
    variant_alt_mask = np.zeros(interval.width, dtype=bool)
    if insertion_length > 0:
      variant_alt_mask[
          variant_start_ : variant_start_ + insertion_length + 1
      ] = True

    variant_is_indel = np.asarray([insertion_length != 0])

    # Insertion masks.
    # (1) Create an index mask that moves the ALT scores corresponding to
    # insertions to the end of the ALT vector.
    # (2) Create a mask for ignoring those scores that got moved to the end.
    reindex_mask = np.arange(variant_alt_mask.shape[0], dtype=np.int32)
    if insertion_length > 0:
      reindex_mask = np.hstack([
          reindex_mask[: variant_start_ + 1],
          reindex_mask[variant_start_ + insertion_length + 1 :],
          reindex_mask[
              variant_start_ + 1 : variant_start_ + insertion_length + 1
          ],
      ])
    crop_mask = np.zeros_like(variant_alt_mask)
    if insertion_length > 0:
      crop_mask[-insertion_length:] = True

    # Deletion masks.
    # Create masks for (1) inserting zeros at the end of the ALT vector and
    # (2) for bringing those zeros to the deletion locations.
    # This aligns the REF and ALT vectors.
    zeros_mask = np.zeros_like(variant_alt_mask)
    if deletion_length > 0:
      zeros_mask[-deletion_length:] = True
    deletion_reindex_mask = np.arange(variant_alt_mask.shape[0], dtype=np.int32)
    if deletion_length > 0:
      deletion_reindex_mask = np.hstack([
          deletion_reindex_mask[: variant_start_ + 1],
          deletion_reindex_mask[-deletion_length:],
          deletion_reindex_mask[variant_start_ + 1 : -deletion_length],
      ])
    indel_mask = cls(
        variant_is_indel=variant_is_indel,
        variant_alt_mask=variant_alt_mask,
        variant_insertion_reindex_mask=reindex_mask,
        variant_insertion_crop_mask=crop_mask,
        variant_deletion_zeros_mask=zeros_mask,
        variant_deletion_reindex_mask=deletion_reindex_mask,
    )
    if interval.negative_strand:
      return indel_mask.reverse_complement()
    return indel_mask

  def reverse_complement(self) -> 'IndelMask':
    """Reverse complements the IndelMask."""
    interval_width = self.variant_alt_mask.shape[-1]
    return IndelMask(
        variant_is_indel=self.variant_is_indel,
        variant_alt_mask=self.variant_alt_mask[..., ::-1],
        variant_insertion_reindex_mask=(
            interval_width - 1 - self.variant_insertion_reindex_mask[..., ::-1]
        ),
        variant_insertion_crop_mask=self.variant_insertion_crop_mask[..., ::-1],
        variant_deletion_zeros_mask=self.variant_deletion_zeros_mask[..., ::-1],
        variant_deletion_reindex_mask=(
            interval_width - 1 - self.variant_deletion_reindex_mask[..., ::-1]
        ),
    )


@jax.jit
def align_alternate(
    alt: Float32[Array, 'S T'],
    masks: IndelMask,
) -> Float32[Array, 'S T']:
  """Aligns ALT predictions to match the REF allele's sequence length.

  This function adjusts the `alt` prediction array to account for indels
  (insertions or deletions) present in the `variant`.

  For insertions, the function summarizes the inserted region by taking the
  maximum value across the alternate bases and pads the end with zeros to
  maintain the original sequence length.

  For deletions, zero signal is inserted at the locations corresponding to the
  deleted bases in the reference.

  Args:
    alt: The ALT allele predictions, shape [sequence_length, num_tracks].
    masks: Dataclass containing indel alignment masks for aligning ALT to REF
      predictions.

  Returns:
    The aligned ALT predictions, shape [sequence_length, num_tracks].
  """

  # Summarize potential insertions by computing the maximum score across
  # alternate bases.
  alt_variant_max = jnp.max(
      alt,
      initial=-jnp.inf,
      where=masks.variant_alt_mask[..., None],
      axis=0,
      keepdims=True,
  )
  alt = jnp.where(masks.variant_alt_mask[..., None], alt_variant_max, alt)
  # We cannot alter the ALT shape for jit purposes, so we move unwanted
  # scores to the end of the vector. We will mask out these positions at
  # the end.
  alt = alt[masks.variant_insertion_reindex_mask]

  # Handle potential deletions.
  # We start by inserting zeros to the end of the ALT vector. The number
  # of zeros equals the deletion length, i.e. len(reference_bases) - 1.
  alt = jnp.where(masks.variant_deletion_zeros_mask[..., None], 0.0, alt)
  # Reindex ALT to bring the zeros to their correct place.
  alt = alt[masks.variant_deletion_reindex_mask]
  # ALT is now aligned with REF. We also return the alignment mask,
  # which is derived from the ALT cropping mask due to insertion handling.
  alignment_mask = 1 - masks.variant_insertion_crop_mask[..., None]
  alt *= alignment_mask
  return alt
