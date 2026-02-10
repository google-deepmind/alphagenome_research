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

"""Implementation of gene mask variant scoring."""

import functools

from alphagenome import typing
from alphagenome.data import genome
from alphagenome.data import track_data
from alphagenome.models import dna_output
from alphagenome.models import variant_scorers
from alphagenome_research.model.variant_scoring import gene_mask_extractor as gene_mask_extractor_lib
from alphagenome_research.model.variant_scoring import variant_scoring
import anndata
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float32  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np
import pandas as pd


_VariantScorerSettings = (
    variant_scorers.GeneMaskLFCScorer
    | variant_scorers.GeneMaskActiveScorer
    | variant_scorers.GeneMaskSplicingScorer
)


@functools.partial(jax.jit, static_argnames=['settings'])
@typing.jaxtyped
def _score_gene_variant(
    ref, alt, gene_mask, *, settings: _VariantScorerSettings
) -> Float32[Array, 'G T']:
  """Returns a function to score a variant given the settings."""

  match settings.base_variant_scorer:
    case variant_scorers.BaseVariantScorer.GENE_MASK_LFC:
      # Scores are the log fold change between the mean prediction of REF and
      # ALT within each gene mask.
      gene_mask_sum = gene_mask.sum(axis=0)[:, jnp.newaxis]
      ref_mean = jnp.einsum('lt,lg->gt', ref, gene_mask) / gene_mask_sum
      alt_mean = jnp.einsum('lt,lg->gt', alt, gene_mask) / gene_mask_sum
      return jnp.log(alt_mean + 1e-3) - jnp.log(ref_mean + 1e-3)
    case variant_scorers.BaseVariantScorer.GENE_MASK_ACTIVE:
      # Scores are the maximum of the mean prediction for REF and ALT within
      # each gene mask.
      gene_mask_sum = gene_mask.sum(axis=0)[:, jnp.newaxis]
      ref_score = jnp.einsum('lt,lg->gt', ref, gene_mask) / gene_mask_sum
      alt_score = jnp.einsum('lt,lg->gt', alt, gene_mask) / gene_mask_sum
      return jnp.maximum(alt_score, ref_score)
    case variant_scorers.BaseVariantScorer.GENE_MASK_SPLICING:
      # Scores are the maximum of the absolute difference between REF and ALT
      # within each gene mask.
      # Even for relatively small number of genes the naive implementation
      # runs out of memory. We use a map here to reduce the memory footprint.
      return jax.lax.map(
          lambda mask: jnp.max(jnp.abs(alt - ref) * mask[:, None], axis=0),
          jnp.matrix_transpose(gene_mask),
      )  # shape: (G, T)
    case _:
      raise ValueError(
          f'Unsupported base variant scorer: {settings.base_variant_scorer}.'
      )


class GeneVariantScorer(
    variant_scoring.VariantScorer[
        Bool[np.ndarray | Array, 'S G'],
        pd.DataFrame,
        _VariantScorerSettings,
    ]
):
  """Variant scorer that computes scores for different genes."""

  def __init__(
      self,
      gene_mask_extractor: gene_mask_extractor_lib.GeneMaskExtractor,
  ):
    """Initializes the GeneVariantScorer.

    Args:
      gene_mask_extractor: Gene mask extractor to use.
    """
    self._gene_mask_extractor = gene_mask_extractor

  def get_masks_and_metadata(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      settings: _VariantScorerSettings,
      track_metadata: dna_output.OutputMetadata,
  ) -> tuple[Bool[np.ndarray | Array, 'S G'], pd.DataFrame]:
    """Get gene masks and metadata for the given interval and variant.

    Note that the gene mask returned for the REF allele is just the normal
    gene mask extracted from the GTF file from the interval. However, the gene
    mask can be different for the ALT allele in the case of indels. We handle
    this by extracting the indel alignment masks, that will be used to align the
    ALT to the REF predictions, so that the same gene mask can be applied to
    both.

    Args:
      interval: Genomic interval to extract gene masks for.
      variant: Variant that may alter the gene mask in the case of the ALT
        allele.
      settings: The variant scorer settings.
      track_metadata: Track metadata for the variant.

    Returns:
      Tuple of (gene variant masks, mask metadata). The mask metadata is the
      part of the GTF pandas dataframe that was used to construct the gene
      masks.
    """
    del track_metadata
    if variant_scoring.get_resolution(settings.requested_output) != 1:
      raise ValueError(
          'Only resolution = 1 is supported for gene variant scoring.'
      )
    gene_mask, metadata = self._gene_mask_extractor.extract(interval, variant)
    return (gene_mask, metadata)

  @typing.jaxtyped
  def score_variant(
      self,
      ref: variant_scoring.ScoreVariantInput,
      alt: variant_scoring.ScoreVariantInput,
      *,
      masks: Bool[Array, 'S G'],
      settings: _VariantScorerSettings,
      variant: genome.Variant | None = None,
      interval: genome.Interval | None = None,
  ) -> variant_scoring.ScoreVariantOutput:
    alt = alt[settings.requested_output]
    ref = ref[settings.requested_output]
    gene_mask = masks
    alt = variant_scoring.align_alternate(alt, variant, interval)

    output = _score_gene_variant(ref, alt, gene_mask, settings=settings)
    return {'score': output}

  def finalize_variant(
      self,
      scores: variant_scoring.ScoreVariantResult,
      *,
      track_metadata: dna_output.OutputMetadata,
      mask_metadata: pd.DataFrame,
      settings: _VariantScorerSettings,
  ) -> anndata.AnnData:
    """Returns summarized scores for the given scores and metadata."""
    output_metadata = track_metadata.get(settings.requested_output)
    assert isinstance(output_metadata, track_data.TrackMetadata)
    strand_mask = (
        np.asarray(mask_metadata['strand'].values)[:, None]
        == output_metadata['strand'].values[None]
    ) | (output_metadata['strand'].values[None] == '.')

    scores = np.where(strand_mask, scores['score'], np.nan)
    return variant_scoring.create_anndata(
        scores,
        obs=mask_metadata,
        var=output_metadata,
    )
