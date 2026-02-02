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

"""Implementation of splice junction variant scoring."""

from collections.abc import Mapping

from alphagenome import typing
from alphagenome.data import genome
from alphagenome.data import junction_data
from alphagenome.models import dna_output
from alphagenome.models import variant_scorers
from alphagenome_research.model.variant_scoring import gene_mask_extractor
from alphagenome_research.model.variant_scoring import variant_scoring
import anndata
import einshape
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Float32, Int32, PyTree, Shaped  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np
import pandas as pd
import pyranges

MAX_SPLICE_SITES = 256
PAD_VALUE = -1


def _create_empty(mask_metadata: pd.DataFrame, track_metadata: pd.DataFrame):
  """Create empty AnnData object for splice junction scoring."""
  junction_columns = [
      'junction_Start',
      'junction_End',
  ]
  return variant_scoring.create_anndata(
      np.zeros((0, len(track_metadata['name'])), dtype=np.float32),
      obs=pd.DataFrame(columns=list(mask_metadata.columns) + junction_columns),
      var=pd.DataFrame({
          'strand': '.',
          'name': track_metadata['name'],
          'gtex_tissue': track_metadata['gtex_tissue'],
          'ontology_curie': track_metadata.get('ontology_curie'),
          'biosample_type': track_metadata.get('biosample_type'),
          'biosample_name': track_metadata.get('biosample_name'),
          'biosample_life_stage': track_metadata.get('biosample_life_stage'),
          'data_source': track_metadata.get('data_source'),
          'Assay title': track_metadata.get('Assay title'),
      }),
  )


def _create(
    junction_scores: pd.DataFrame,
    mask_metadata: pd.DataFrame,
    track_metadata: pd.DataFrame,
) -> anndata.AnnData:
  """Converts a dataframe of junction scores to an AnnData object."""
  if mask_metadata.empty or junction_scores.empty:
    raise ValueError('Both junction_scores and mask_metadata must be non-empty')

  junction_scores = junction_scores[
      junction_scores['gene_id'].isin(mask_metadata['gene_id'])
  ]

  gene_max_scores = []
  track_names = track_metadata['name']
  for gene_id in junction_scores['gene_id'].unique():
    gene_junction_scores = junction_scores[
        junction_scores.gene_id == gene_id
    ].reset_index(drop=True)
    gene_junction_scores = gene_junction_scores.iloc[
        gene_junction_scores[track_names].values.argmax(0)
    ]
    gene_max_scores.append(gene_junction_scores)
  junction_scores = pd.concat(gene_max_scores)
  score_values = junction_scores[track_names].values

  # Merge junction information with mask metadata.
  junctions_all_genes = junction_scores[['gene_id', 'Start', 'End']]
  junctions_all_genes.columns = ['gene_id', 'junction_Start', 'junction_End']
  mask_metadata = junctions_all_genes.merge(
      mask_metadata, on='gene_id', sort=False
  )
  # Create the final track metadata.
  track_metadata = pd.DataFrame({
      'strand': '.',  # We already matched prediction by strand.
      'name': track_names,
      'gtex_tissue': track_metadata['gtex_tissue'],
      'ontology_curie': track_metadata.get('ontology_curie'),
      'biosample_type': track_metadata.get('biosample_type'),
      'biosample_name': track_metadata.get('biosample_name'),
      'biosample_life_stage': track_metadata.get('biosample_life_stage'),
      'data_source': track_metadata.get('data_source'),
      'Assay title': track_metadata.get('Assay title'),
  })
  ann_data = variant_scoring.create_anndata(
      score_values,
      obs=mask_metadata,
      var=track_metadata,
  )
  # Remove duplicated junctions. Per gene, we report junctions that has maximum
  # score in any tissue. For the reported junctions, we return their predictions
  # in all tissues.
  return ann_data[~ann_data.obs.duplicated()].copy()


@typing.jaxtyped
def unstack_junction_predictions(
    splice_junction_prediction: Float[np.ndarray, 'D D _'],
    splice_site_positions: Int32[np.ndarray, '4 D'],
    interval: genome.Interval | None = None,
) -> tuple[
    Float[np.ndarray, 'num_junctions num_tracks'],
    Shaped[np.ndarray, 'num_junctions'],
    Int32[np.ndarray, 'num_junctions'],
    Int32[np.ndarray, 'num_junctions'],
]:
  """Unstack splice junction predictions to long format."""
  # Unpack splice junction predictions.
  splice_junction_prediction = einshape.numpy_einshape(
      'da(st)->dast', splice_junction_prediction, s=2
  )
  # Convert splice site positions.
  remove_padding_fn = lambda x: x[x != PAD_VALUE]
  pos_donors = remove_padding_fn(splice_site_positions[0])
  pos_acceptors = remove_padding_fn(splice_site_positions[1])
  neg_donors = remove_padding_fn(splice_site_positions[2])
  neg_acceptors = remove_padding_fn(splice_site_positions[3])
  junction_pred_pos = splice_junction_prediction[
      : len(pos_donors), : len(pos_acceptors), 0
  ]
  junction_pred_pos = einshape.numpy_einshape('dat->(da)t', junction_pred_pos)
  num_pos_donors = len(pos_donors)
  pos_donors = np.repeat(pos_donors, len(pos_acceptors))
  pos_acceptors = np.tile(pos_acceptors, num_pos_donors)
  junction_pred_neg = splice_junction_prediction[
      : len(neg_donors), : len(neg_acceptors), 1
  ]
  junction_pred_neg = einshape.numpy_einshape('dat->(da)t', junction_pred_neg)
  num_neg_donors = len(neg_donors)
  neg_donors = np.repeat(neg_donors, len(neg_acceptors))
  neg_acceptors = np.tile(neg_acceptors, num_neg_donors)
  # Combine into  a single output.
  junction_predictions = np.concatenate(
      [junction_pred_pos, junction_pred_neg], axis=0
  )
  # Junction start and end positions.
  starts = np.concatenate([pos_donors, neg_acceptors]) + 1
  starts += interval.start if interval is not None else 0
  ends = np.concatenate([pos_acceptors, neg_donors])
  ends += interval.start if interval is not None else 0
  strands = np.array(['+'] * len(pos_donors) + ['-'] * len(neg_donors))
  filter_mask = (starts < ends) & (starts > 0)

  return (
      junction_predictions[filter_mask],
      strands[filter_mask],
      starts[filter_mask],
      ends[filter_mask],
  )


def junction_predictions_to_dataframe(
    splice_junction_prediction: Float[np.ndarray, 'D D _'],
    splice_site_positions: Int32[np.ndarray, 'T_mul_4 D'],
    metadata: junction_data.JunctionMetadata,
    interval: genome.Interval,
) -> pd.DataFrame:
  """Convert splice junction predictions to a dataframe."""
  junction_predictions, strands, starts, ends = unstack_junction_predictions(
      splice_junction_prediction, splice_site_positions, interval
  )
  junctions = pd.DataFrame({
      'Chromosome': interval.chromosome,
      'Start': starts,
      'End': ends,
      'Strand': strands,
  })
  predictions = pd.DataFrame(junction_predictions, columns=metadata['name'])
  return pd.concat([junctions, predictions], axis=1)


class SpliceJunctionVariantScorer(variant_scoring.VariantScorer):
  """Implements the SpliceJunction variant scoring strategy.

  Scores variants by the maximum of absolute delta pair counts of junctions
  within the input interval. Junctions are annotated by overlapping with the
  gtf gene intervals.
  """

  def __init__(self, gtf: pd.DataFrame):
    self._gene_mask_extractor = gene_mask_extractor.GeneMaskExtractor(
        gtf,
        gene_mask_extractor.GeneMaskType.BODY,
        gene_query_type=gene_mask_extractor.GeneQueryType.VARIANT_OVERLAPPING,
        filter_protein_coding=True,
    )

  def get_masks_and_metadata(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      settings: variant_scorers.SpliceJunctionScorer,
      track_metadata: dna_output.OutputMetadata,
  ) -> tuple[None, pd.DataFrame]:
    """See base class."""
    del settings, track_metadata
    _, metadata = self._gene_mask_extractor.extract(interval, variant)
    metadata['interval'] = interval
    return None, metadata

  def score_variant(
      self,
      ref: Mapping[
          dna_output.OutputType,
          PyTree[Float32[Array, 'D D T'] | Int32[Array, 'S D']],
      ],
      alt: Mapping[
          dna_output.OutputType,
          PyTree[Float32[Array, 'D D T'] | Int32[Array, 'S D']],
      ],
      *,
      masks: None,
      settings: variant_scorers.SpliceJunctionScorer,
      variant: genome.Variant | None = None,
      interval: genome.Interval | None = None,
  ) -> variant_scoring.ScoreVariantOutput:
    """See base class."""
    del variant, interval, masks
    ref_junctions = ref[settings.requested_output]['predictions']
    alt_junctions = alt[settings.requested_output]['predictions']

    splice_site_positions = ref[settings.requested_output][
        'splice_site_positions'
    ]

    # JAX dynamic slicing does not work with transfer_guard.
    with jax.transfer_guard('allow'):
      # Ignore splice sites beyond the max_splice_sites specified. This works
      # because padding splice sites are always at the end of the array.
      ref_junctions = ref_junctions[:MAX_SPLICE_SITES, :MAX_SPLICE_SITES]
      alt_junctions = alt_junctions[:MAX_SPLICE_SITES, :MAX_SPLICE_SITES]
      splice_site_positions = splice_site_positions[:, :MAX_SPLICE_SITES]

    @jax.jit
    def _apply_log_offset(x):
      return jnp.log(x + 1e-7)

    ref_junctions = _apply_log_offset(ref_junctions)
    alt_junctions = _apply_log_offset(alt_junctions)

    return {
        'delta_counts': (alt_junctions - ref_junctions).astype(jnp.float16),
        'splice_site_positions': splice_site_positions,
    }

  def finalize_variant(
      self,
      scores: variant_scoring.ScoreVariantResult,
      *,
      track_metadata: dna_output.OutputMetadata,
      mask_metadata: pd.DataFrame,
      settings: variant_scorers.SpliceJunctionScorer,
  ) -> anndata.AnnData:
    """See base class."""
    track_metadata = track_metadata.get(settings.requested_output)

    if mask_metadata.empty:
      return _create_empty(mask_metadata, track_metadata)

    delta_counts = scores['delta_counts']

    interval = mask_metadata['interval'].values[0]
    mask_metadata = mask_metadata.drop(columns=['interval'])

    delta_counts = junction_predictions_to_dataframe(
        np.abs(delta_counts, dtype=np.float32),
        scores['splice_site_positions'],
        metadata=track_metadata,
        interval=interval,
    )
    if delta_counts.empty:
      return _create_empty(mask_metadata, track_metadata)

    junction_scores = (
        pyranges.PyRanges(delta_counts)
        .join(pyranges.PyRanges(mask_metadata), strandedness='same')
        .df
    )

    if not junction_scores.empty:
      junction_scores = junction_scores[
          (junction_scores['Start'] > junction_scores['Start_b'])
          & (junction_scores['End'] < junction_scores['End_b'])
      ]
      return _create(junction_scores, mask_metadata, track_metadata)
    else:
      return _create_empty(mask_metadata, track_metadata)
