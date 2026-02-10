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

"""Implements a variant scorer for polyadenylation."""

from alphagenome import typing
from alphagenome.data import genome
from alphagenome.models import dna_output
from alphagenome.models import variant_scorers
from alphagenome_research.model.variant_scoring import gene_mask_extractor
from alphagenome_research.model.variant_scoring import variant_scoring
import anndata
import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float32  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np
import pandas as pd

MAX_GENES = 22
MAX_PAS = 136

_PAS_MASK_WIDTH = 400


@typing.jaxtyped
@chex.dataclass(frozen=True)
class PolyadenylationVariantMasks:
  pas_mask: Bool[Array | np.ndarray, 'S G P']
  gene_mask: Bool[Array | np.ndarray, 'G']


@jax.jit
@typing.jaxtyped
def _aggregate_maximum_ratio_coverage_fc(
    ref: Float32[Array, 'S T'],
    alt: Float32[Array, 'S T'],
    gene_pas_mask: Bool[Array, 'S G P'],
) -> Float32[Array, 'G T']:
  """Implements the Borzoi statistic for paQTL variant scoring."""

  ref_aggregation = jnp.einsum('pc,pga->gac', ref, gene_pas_mask)
  alt_aggregation = jnp.einsum('pc,pga->gac', alt, gene_pas_mask)
  covr_ratio = alt_aggregation / ref_aggregation
  covr_ratio = jnp.nan_to_num(covr_ratio, posinf=0, neginf=0, nan=0)
  # (gene, pas, tracks)

  # Get proximal vs distal counts for all possible polyadenylation site
  # split versions.
  k_interval = jnp.arange(MAX_PAS)  # All PAS for the interval.
  # Create mask for potential proximal pas site splits across the interval.
  # Each PAS is added to the mask in sequential order for each gene, which
  # ensures that aggregating the proximal counts take the first k PAS sites
  # for each gene.
  proximal_sites = k_interval[None] <= k_interval[:, None]

  # Get total number of pas sites per gene
  k_total = gene_pas_mask.max(axis=0).sum(axis=-1)[:, None]  # (gene, 1)
  # Get number of pas sites included in the proximal split per gene.
  k_gene = gene_pas_mask.max(axis=0).cumsum(axis=-1)  # (gene, k)
  k_scaling = ((k_total - k_gene) / k_gene).T[:, :, None]  # (k, gene, 1)

  proximal_counts = jnp.einsum('gac,ka->kgc', covr_ratio, proximal_sites)
  distal_counts = jnp.einsum('gac,ka->kgc', covr_ratio, ~proximal_sites)

  scores = jnp.abs(jnp.log2(k_scaling * proximal_counts / distal_counts))
  # We are converting nan to num to keep all the padding cases at 0.
  scores = jnp.nan_to_num(scores, posinf=0, neginf=0, nan=0)
  # [k, genes, tracks]
  return scores.max(axis=0)


class PolyadenylationVariantScorer(variant_scoring.VariantScorer):
  """Variant scorer for polyadenylation."""

  def __init__(
      self,
      gtf: pd.DataFrame,
      pas_gtf: pd.DataFrame,
  ):
    self._gene_mask_extractor = gene_mask_extractor.GeneMaskExtractor(
        gtf,
        gene_mask_extractor.GeneMaskType.BODY,
        gene_query_type=gene_mask_extractor.GeneQueryType.VARIANT_OVERLAPPING,
    )
    if 'gene_id_nopatch' not in pas_gtf:
      pas_gtf['gene_id_nopatch'] = pas_gtf['gene_id'].str.split(
          '.', expand=True
      )[0]
    self._pas_per_gene = {
        gene_id_gtf: df
        for gene_id_gtf, df in pas_gtf.groupby('gene_id_nopatch')
    }

  def get_masks_and_metadata(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      settings: variant_scorers.PolyadenylationScorer,
      track_metadata: dna_output.OutputMetadata,
  ) -> tuple[PolyadenylationVariantMasks, pd.DataFrame]:
    """See base class."""
    del settings, track_metadata
    _, gene_metadata = self._gene_mask_extractor.extract(interval, variant)
    if len(gene_metadata) > MAX_GENES:
      raise ValueError(
          f'Too many genes found for interval {interval}: {len(gene_metadata)}'
      )
    gene_metadata_rows = []
    gene_padding_mask = np.zeros(MAX_GENES, dtype=bool)
    pas_mask = np.zeros(
        (interval.width, MAX_GENES, MAX_PAS),
        dtype=bool,
    )
    has_gene_id_nopatch = 'gene_id_nopatch' in gene_metadata.columns
    for gene_index, gene_row in gene_metadata.iterrows():
      gene_id = (
          gene_row['gene_id_nopatch']
          if has_gene_id_nopatch
          else gene_row['gene_id'].split('.')[0]
      )
      if gene_id not in self._pas_per_gene:
        continue
      gene_pas = self._pas_per_gene[gene_id]
      gene_pas = gene_pas[gene_pas['pas_strand'] == gene_row['strand']]
      if (
          gene_pas.shape[0] == 0
          # Check at least 80% of a gene's PAS sites fall within the interval.
          or np.mean((gene_pas['Start'] >= interval.start).values) < 0.8
          or np.mean((gene_pas['End'] < interval.end).values) < 0.8
      ):
        # No PAS sites in interval for the gene.
        continue

      # Only look at PAS sites that fall in the interval.
      gene_pas = gene_pas[
          (gene_pas['Start'] >= interval.start)
          & (gene_pas['End'] < interval.end)
      ]

      if gene_pas.shape[0] == 1:
        # Only one PAS site in interval for the gene.
        continue
      else:
        pas_interval_start = gene_pas['Start'] - interval.start
        gene_pas = gene_pas.sort_values(by='Start')
        # Get PAS metadata for gene.
        gene_row_metadata = gene_row.to_dict()
        dist = np.abs(gene_pas['Start'] - variant.position)
        gene_row_metadata['num_pas'] = len(gene_pas)
        gene_row_metadata['min_pas_var_distance'] = dist.min()
        gene_padding_mask[gene_index] = True
        gene_metadata_rows.append(gene_row_metadata)

      for (pas_index, pas_row), p_interval_start in zip(
          gene_pas.reset_index(drop=True).iterrows(),
          pas_interval_start,
          strict=True,
      ):
        # Defaults`to only doing upstream coverage of PAS site.
        if pas_row.pas_strand == '+':
          bin_end = p_interval_start + 1
          bin_start = bin_end - _PAS_MASK_WIDTH
        else:
          bin_start = p_interval_start
          bin_end = bin_start + _PAS_MASK_WIDTH
        bin_start = max(min(bin_start, interval.width), 0)
        bin_end = max(min(bin_end, interval.width), 0)
        pas_mask[bin_start:bin_end, gene_index, pas_index] = True

    return (
        PolyadenylationVariantMasks(
            pas_mask=pas_mask, gene_mask=gene_padding_mask
        ),
        pd.DataFrame(gene_metadata_rows),
    )

  def score_variant(
      self,
      ref: variant_scoring.ScoreVariantInput,
      alt: variant_scoring.ScoreVariantInput,
      *,
      masks: PolyadenylationVariantMasks,
      settings: variant_scorers.PolyadenylationScorer,
      variant: genome.Variant | None = None,
      interval: genome.Interval | None = None,
  ) -> variant_scoring.ScoreVariantOutput:
    """See base class."""
    ref = ref[settings.requested_output]
    alt = alt[settings.requested_output]

    alt = variant_scoring.align_alternate(alt, variant, interval)
    return {
        'scores': _aggregate_maximum_ratio_coverage_fc(
            ref, alt, jnp.asarray(masks.pas_mask)
        ),
        'gene_mask': masks.gene_mask,
    }

  def finalize_variant(
      self,
      scores: variant_scoring.ScoreVariantResult,
      *,
      track_metadata: dna_output.OutputMetadata,
      mask_metadata: pd.DataFrame,
      settings: variant_scorers.PolyadenylationScorer,
  ) -> anndata.AnnData:
    """See base class."""

    track_metadata = track_metadata.get(settings.requested_output)
    return variant_scoring.create_anndata(
        scores['scores'][scores['gene_mask']],
        obs=mask_metadata,
        var=track_metadata,
    )
