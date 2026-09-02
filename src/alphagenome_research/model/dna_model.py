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

"""Implementation of alphagenome DNAModel interface."""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
import dataclasses
import functools
import os
import threading
from typing import TypeAlias

from alphagenome import tensor_utils
from alphagenome import typing
from alphagenome.data import genome
from alphagenome.data import junction_data
from alphagenome.data import ontology
from alphagenome.data import track_data
from alphagenome.interpretation import ism
from alphagenome.io import fasta
from alphagenome.models import dna_model
from alphagenome.models import dna_output
from alphagenome.models import interval_scorers as interval_scorers_lib
from alphagenome.models import variant_scorers as variant_scorers_lib
from alphagenome.models import variant_scoring_utils
from alphagenome_research.io import genome as genome_io
from alphagenome_research.io import splicing as splicing_io
from alphagenome_research.model import augmentation
from alphagenome_research.model import embeddings as embeddings_lib
from alphagenome_research.model import indel_stitch_utils
from alphagenome_research.model import model
from alphagenome_research.model import one_hot_encoder
from alphagenome_research.model import splicing
from alphagenome_research.model.interval_scoring import gene_mask as interval_gene_mask
from alphagenome_research.model.metadata import metadata as metadata_lib
from alphagenome_research.model.variant_scoring import center_mask
from alphagenome_research.model.variant_scoring import contact_map
from alphagenome_research.model.variant_scoring import gene_mask
from alphagenome_research.model.variant_scoring import gene_mask_extractor
from alphagenome_research.model.variant_scoring import polyadenylation
from alphagenome_research.model.variant_scoring import splice_junction
from alphagenome_research.model.variant_scoring import variant_scoring
from alphagenome_research.model.variant_scoring.calibration import calibration
from alphagenome_research.typing import ArrayType  # pylint: disable=g-importing-member
import anndata
import chex
import haiku as hk
import huggingface_hub
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Float32, Int32, PyTree, Shaped  # pylint: disable=g-importing-member, g-multiple-import
import jmp
import kagglehub
from kagglehub import auth as kaggle_auth
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd

AlphaGenomeOutputMetadata: TypeAlias = metadata_lib.AlphaGenomeOutputMetadata
ModelVersion: TypeAlias = dna_model.ModelVersion
Organism: TypeAlias = dna_model.Organism
Output: TypeAlias = dna_output.Output
OutputMetadata: TypeAlias = dna_output.OutputMetadata
OutputType: TypeAlias = dna_output.OutputType
VariantOutput: TypeAlias = dna_output.VariantOutput
Organism: TypeAlias = dna_model.Organism

BatchPrediction: TypeAlias = PyTree[
    Float[Array, 'B ...'] | Int32[Array, 'B ...']
]

ApplyFn = Callable[
    [
        hk.Params,
        hk.State,
        Float32[Array, 'B S 4'],
        Int32[Array, 'B'],
    ],
    BatchPrediction,
]
JunctionsApplyFn = Callable[
    [
        hk.Params,
        hk.State,
        Float32[Array, 'B S D'],
        Int32[Array, 'B 4 K'],
        Int32[Array, 'B'],
    ],
    BatchPrediction,
]
TrunkApplyFn = Callable[
    [
        hk.Params,
        hk.State,
        Float32[Array, 'B S 4'],
        Int32[Array, 'B'],
    ],
    embeddings_lib.Embeddings,
]
HeadsApplyFn = Callable[
    [
        hk.Params,
        hk.State,
        Float32[Array, 'B S D1'],
        Float32[Array, 'B S128 D2'],
        Float32[Array, 'B Sp Sp Dp'],
        Int32[Array, 'B'],
    ],
    PyTree[Array],
]


def extract_predictions(
    predictions: BatchPrediction,
    requested_outputs: Collection[dna_output.OutputType],
) -> Mapping[dna_output.OutputType, BatchPrediction]:
  """Extracts predictions from the model predictions."""
  results = {}
  for output_type in requested_outputs:
    match output_type:
      case dna_output.OutputType.ATAC:
        prediction = predictions.get('atac', {}).get('predictions_1bp')
      case dna_output.OutputType.CAGE:
        prediction = predictions.get('cage', {}).get('predictions_1bp')
      case dna_output.OutputType.DNASE:
        prediction = predictions.get('dnase', {}).get('predictions_1bp')
      case dna_output.OutputType.RNA_SEQ:
        prediction = predictions.get('rna_seq', {}).get('predictions_1bp')
      case dna_output.OutputType.CHIP_HISTONE:
        prediction = predictions.get('chip_histone', {}).get(
            'predictions_128bp'
        )
      case dna_output.OutputType.CHIP_TF:
        prediction = predictions.get('chip_tf', {}).get('predictions_128bp')
      case dna_output.OutputType.SPLICE_SITES:
        prediction = predictions.get('splice_sites_classification', {}).get(
            'predictions'
        )
      case dna_output.OutputType.SPLICE_SITE_USAGE:
        prediction = predictions.get('splice_sites_usage', {}).get(
            'predictions'
        )
      case dna_output.OutputType.SPLICE_JUNCTIONS:
        if (
            splice_site_preds := predictions.get('splice_sites_junction')
        ) is not None:
          prediction = {
              'predictions': splice_site_preds.get('predictions'),
              'splice_site_positions': splice_site_preds.get(
                  'splice_site_positions'
              ),
          }
        else:
          prediction = None
      case dna_output.OutputType.CONTACT_MAPS:
        prediction = predictions.get('contact_maps', {}).get('predictions')
      case dna_output.OutputType.PROCAP:
        prediction = predictions.get('procap', {}).get('predictions_1bp')
      case _:
        raise ValueError(f'Unsupported output type: {output_type}')
    if prediction is not None:
      results[output_type] = prediction
  return results


@typing.jaxtyped
@chex.dataclass(frozen=True)
class _SpliceJunctionVariantMasks:
  """Container for splicing masks used by predict_variant."""

  splice_sites: Bool[ArrayType, 'B S 5'] | None
  reference_genes: Bool[ArrayType, 'B S 1']
  indel_masks: variant_scoring.IndelMask


@typing.jaxtyped
def _predict(
    params: hk.Params,
    state: hk.State,
    sequences: Float32[Array, 'B S 4'],
    organism_indices: Int32[Array, 'B'],
    *,
    requested_outputs: Collection[dna_output.OutputType],
    strand_reindexing: Mapping[dna_output.OutputType, Int32[Array, '_']],
    negative_strand_mask: Bool[Array, 'B'],
    apply_fn: ApplyFn,
) -> Mapping[dna_output.OutputType, BatchPrediction]:
  """Maps predictions to output types and optionally reverse complements."""
  predictions = apply_fn(params, state, sequences, organism_indices)
  predictions = extract_predictions(predictions, requested_outputs)
  predictions = augmentation.reverse_complement(
      predictions,
      negative_strand_mask,
      strand_reindexing=strand_reindexing,
      sequence_length=sequences.shape[1],
  )
  return predictions


def _extract_indel_stitch_input_single(
    variant: genome.Variant,
    interval: genome.Interval,
    fasta_extractor: fasta.FastaExtractor,
    encoder: one_hot_encoder.DNAOneHotEncoder,
    reference_sequence_encoded: Float32[ArrayType, 'B S 4'],
    alternate_sequence_encoded: Float32[ArrayType, 'B S 4'],
) -> indel_stitch_utils.IndelStitchInput | None:
  """Extracts IndelStitchInput for a single variant."""
  if not variant.is_deletion and not variant.is_insertion:
    return None

  interval_length = interval.width
  unstranded = interval.as_unstranded()
  seq_len = reference_sequence_encoded.shape[1]

  if variant.is_deletion:
    deletion_length = len(variant.reference_bases) - len(
        variant.alternate_bases
    )
    shifted_interval = unstranded.shift(deletion_length)
    ref_right_raw = fasta_extractor.extract(shifted_interval)
    ref_right_enc = encoder.encode(ref_right_raw)
    if interval.negative_strand:
      ref_right_enc = ref_right_enc[::-1, ::-1]

    stitch_start_fwd = (
        variant.start + len(variant.alternate_bases) - unstranded.start
    )
    stitch_end_fwd = (
        variant.start - unstranded.start + len(variant.reference_bases)
    )

    if interval.negative_strand:
      left_sequence = ref_right_enc
      right_sequence = reference_sequence_encoded[0]
      left_shift = deletion_length
      right_shift = 0
      stitch_start = seq_len - stitch_end_fwd
      stitch_end = seq_len - stitch_start_fwd + len(variant.alternate_bases)
    else:
      left_sequence = reference_sequence_encoded[0]
      right_sequence = ref_right_enc
      left_shift = 0
      right_shift = deletion_length
      stitch_start = stitch_start_fwd
      stitch_end = stitch_end_fwd
    is_deletion = True

  elif variant.is_insertion:
    insertion_length = len(variant.alternate_bases) - len(
        variant.reference_bases
    )
    raw_seq = fasta_extractor.extract(unstranded)
    alt_raw = genome_io.insert_alternate_variant(raw_seq, unstranded, variant)
    alt_right_raw = alt_raw[-interval_length:]
    alt_right_enc = encoder.encode(alt_right_raw)
    if interval.negative_strand:
      alt_right_enc = alt_right_enc[::-1, ::-1]

    stitch_start_fwd = (
        variant.start + len(variant.reference_bases) - unstranded.start
    )
    stitch_end_fwd = (
        variant.start - unstranded.start + len(variant.alternate_bases)
    )

    if interval.negative_strand:
      left_sequence = alt_right_enc
      right_sequence = alternate_sequence_encoded[0]
      left_shift = insertion_length
      right_shift = 0
      stitch_start = seq_len - stitch_end_fwd
      stitch_end = seq_len - stitch_start_fwd + len(variant.reference_bases)
    else:
      left_sequence = alternate_sequence_encoded[0]
      right_sequence = alt_right_enc
      left_shift = 0
      right_shift = insertion_length
      stitch_start = stitch_start_fwd
      stitch_end = stitch_end_fwd
    is_deletion = False
  else:
    raise ValueError(f'Unsupported variant type: {variant}')

  return indel_stitch_utils.IndelStitchInput(
      left_sequence=jnp.array(left_sequence)[np.newaxis],
      right_sequence=jnp.array(right_sequence)[np.newaxis],
      left_shift=jnp.array([[left_shift]], dtype=np.int32),
      right_shift=jnp.array([[right_shift]], dtype=np.int32),
      stitch_index=jnp.array([[stitch_end]], dtype=np.int32),
      indel_start_index=jnp.array([[stitch_start]], dtype=np.int32),
      is_deletion=jnp.array([is_deletion], dtype=bool),
  )


@typing.jaxtyped
def _predict_variant(
    params: hk.Params,
    state: hk.State,
    reference_sequences: Float32[Array, 'B S 4'],
    alternate_sequences: Float32[Array, 'B S 4'],
    splice_junction_masks: _SpliceJunctionVariantMasks,
    organism_indices: Int32[Array, 'B'],
    strand_reindexing: Mapping[dna_output.OutputType, Int32[Array, '_']],
    negative_strand_mask: Bool[Array, 'B'],
    *,
    requested_outputs: Collection[dna_output.OutputType],
    apply_fn: ApplyFn,
    junctions_apply_fn: JunctionsApplyFn,
    trunk_apply_fn: TrunkApplyFn | None = None,
    heads_apply_fn: HeadsApplyFn | None = None,
    indel_stitch_input: indel_stitch_utils.IndelStitchInput | None = None,
    num_splice_sites: int,
    splice_site_threshold: float,
) -> tuple[
    Mapping[dna_output.OutputType, BatchPrediction],
    Mapping[dna_output.OutputType, BatchPrediction],
]:
  """Computes and reverse-complements variant predictions."""
  chex.assert_equal_shape([reference_sequences, alternate_sequences])
  sequence_length = reference_sequences.shape[1]

  def _select(s, o, is_del):
    mask = jnp.expand_dims(is_del, axis=range(1, s.ndim))
    return jnp.where(mask, s, o)

  if indel_stitch_input is not None:
    if trunk_apply_fn is None or heads_apply_fn is None:
      raise ValueError(
          'Both trunk_apply_fn and heads_apply_fn must be provided when '
          'indel_stitch_input is provided.'
      )
    left_embeddings = trunk_apply_fn(
        params, state, indel_stitch_input.left_sequence, organism_indices
    )
    right_embeddings = trunk_apply_fn(
        params, state, indel_stitch_input.right_sequence, organism_indices
    )

    assert isinstance(left_embeddings.embeddings_1bp, Array)
    assert isinstance(right_embeddings.embeddings_1bp, Array)
    stitched_embeddings_1bp = (
        indel_stitch_utils.stitch_sequence_base_resolution(
            left=left_embeddings.embeddings_1bp,
            right=right_embeddings.embeddings_1bp,
            left_shift=indel_stitch_input.left_shift,
            right_shift=indel_stitch_input.right_shift,
            stitch_index=indel_stitch_input.stitch_index,
        )
    )
    base_res_length = left_embeddings.embeddings_1bp.shape[1]

    assert isinstance(left_embeddings.embeddings_128bp, Array)
    assert isinstance(right_embeddings.embeddings_128bp, Array)
    embeddings_128bp_bin_size = (
        base_res_length // left_embeddings.embeddings_128bp.shape[1]
    )
    stitched_embeddings_128bp = (
        indel_stitch_utils.stitch_sequence_coarse_resolution(
            left=left_embeddings.embeddings_128bp,
            right=right_embeddings.embeddings_128bp,
            left_shift=indel_stitch_input.left_shift,
            right_shift=indel_stitch_input.right_shift,
            stitch_index=indel_stitch_input.stitch_index,
            indel_start_index=indel_stitch_input.indel_start_index,
            bin_size=embeddings_128bp_bin_size,
        )
    )
    assert isinstance(left_embeddings.embeddings_pair, Array)
    assert isinstance(right_embeddings.embeddings_pair, Array)
    embeddings_pair_bin_size = (
        base_res_length // left_embeddings.embeddings_pair.shape[1]
    )
    stitched_embeddings_pair = indel_stitch_utils.stitch_2d_contact_maps(
        left=left_embeddings.embeddings_pair,
        right=right_embeddings.embeddings_pair,
        left_shift=indel_stitch_input.left_shift,
        right_shift=indel_stitch_input.right_shift,
        stitch_index=indel_stitch_input.stitch_index,
        indel_start_index=indel_stitch_input.indel_start_index,
        bin_size=embeddings_pair_bin_size,
    )

    stitched_predictions = heads_apply_fn(
        params,
        state,
        stitched_embeddings_1bp,
        stitched_embeddings_128bp,
        stitched_embeddings_pair,
        organism_indices,
    )

    unstitched_sequence = jnp.where(
        indel_stitch_input.is_deletion[:, None, None],
        alternate_sequences,
        reference_sequences,
    )
    unstitched_predictions = apply_fn(
        params, state, unstitched_sequence, organism_indices
    )

    reference_predictions = jax.tree.map(
        lambda s, o: _select(s, o, indel_stitch_input.is_deletion),
        stitched_predictions,
        unstitched_predictions,
    )
    alternate_predictions = jax.tree.map(
        lambda s, o: _select(o, s, indel_stitch_input.is_deletion),
        stitched_predictions,
        unstitched_predictions,
    )
  else:
    reference_predictions = apply_fn(
        params,
        state,
        reference_sequences,
        organism_indices,
    )
    alternate_predictions = apply_fn(
        params,
        state,
        alternate_sequences,
        organism_indices,
    )
  reference_splice_sites = (
      reference_predictions['splice_sites_classification']['predictions']
      * splice_junction_masks.reference_genes
  )
  alternate_splice_sites = alternate_predictions['splice_sites_classification'][
      'predictions'
  ]

  reference_trunk = reference_predictions['embeddings_1bp']
  alternate_trunk = alternate_predictions['embeddings_1bp']

  def _align_alt(x):
    return jax.vmap(variant_scoring.align_alternate)(
        x, splice_junction_masks.indel_masks
    )

  is_indel = jnp.any(splice_junction_masks.indel_masks.variant_is_indel)

  alternate_trunk = jax.lax.cond(
      is_indel, _align_alt, lambda x: x, alternate_trunk
  )

  if alternate_splice_sites is not None:
    alternate_splice_sites = jax.lax.cond(
        is_indel, _align_alt, lambda x: x, alternate_splice_sites
    )
    alternate_splice_sites = (
        alternate_splice_sites * splice_junction_masks.reference_genes
    )

  # Get union of splice site positions across ref and alt.
  ref_and_alt_splice_site_positions = splicing.generate_splice_site_positions(
      ref=reference_splice_sites,
      alt=alternate_splice_sites,
      splice_sites=jnp.asarray(splice_junction_masks.splice_sites, copy=False),
      k=num_splice_sites,
      pad_to_length=num_splice_sites,
      threshold=splice_site_threshold,
  )
  reference_predictions['splice_sites_junction'] = junctions_apply_fn(
      params,
      state,
      reference_trunk,
      ref_and_alt_splice_site_positions,
      organism_indices,
  )
  alternate_predictions['splice_sites_junction'] = junctions_apply_fn(
      params,
      state,
      alternate_trunk,
      ref_and_alt_splice_site_positions,
      organism_indices,
  )

  def _extract_and_rc(predictions):
    return augmentation.reverse_complement(
        extract_predictions(predictions, requested_outputs),
        negative_strand_mask,
        strand_reindexing=strand_reindexing,
        sequence_length=sequence_length,
    )

  return (
      _extract_and_rc(reference_predictions),
      _extract_and_rc(alternate_predictions),
  )


@typing.jaxtyped
def _filter_predictions(
    predictions: Mapping[dna_output.OutputType, BatchPrediction],
    *,
    track_masks: Mapping[dna_output.OutputType, Bool[Array, '_']],
) -> Mapping[dna_output.OutputType, BatchPrediction]:
  """Filters predictions by mapping of track masks."""
  result = {}
  for output_type, mask in track_masks.items():
    if (prediction := predictions.get(output_type)) is not None:
      # JAX dynamic slicing does not work with transfer_guard.
      with jax.transfer_guard('allow'):
        if output_type == dna_output.OutputType.SPLICE_JUNCTIONS:
          result[output_type] = {
              'predictions': prediction['predictions'][
                  ..., jnp.tile(mask, reps=2)
              ],
              'splice_site_positions': prediction['splice_site_positions'],
          }
        else:
          result[output_type] = prediction[..., mask]
  return result


@functools.partial(jax.jit, static_argnames=['transfer_to_host'])
@typing.jaxtyped
def _upcast_single_batch_predictions(
    x: PyTree[Float[Array, '1 ...'] | Int32[Array, '1 ...']],
    *,
    transfer_to_host: bool = True,
) -> PyTree[
    Float32[Array | np.ndarray, '...'] | Int32[Array | np.ndarray, '...']
]:
  """Helper to upcast and optionally transfer predictions to host."""
  x = jax.tree.map(lambda x: tensor_utils.upcast_floating(x[0]), x)
  return jax.device_put(x, jax.memory.Space.Host) if transfer_to_host else x


@typing.jaxtyped
def _filter_variant_predictions(
    reference_predictions: Mapping[dna_output.OutputType, BatchPrediction],
    alternate_predictions: Mapping[dna_output.OutputType, BatchPrediction],
    *,
    track_masks: Mapping[dna_output.OutputType, Bool[Array, '_']],
) -> tuple[
    Mapping[dna_output.OutputType, BatchPrediction],
    Mapping[dna_output.OutputType, BatchPrediction],
]:
  """Filters variant predictions to a set of output types."""
  return (
      _filter_predictions(reference_predictions, track_masks=track_masks),
      _filter_predictions(alternate_predictions, track_masks=track_masks),
  )


def _convert_ontology_terms(
    ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
) -> Sequence[ontology.OntologyTerm] | None:
  """Convert ontology terms or curies to unique list of OntologyTerms."""
  if ontology_terms is None:
    return None
  result = []
  for term in dict.fromkeys(ontology_terms):
    if isinstance(term, str):
      result.append(ontology.from_curie(term))
    else:
      result.append(term)
  return result


class _DeviceContextManager:
  """Context manager for managing a jax.Device."""

  def __init__(self, device: jax.Device):
    self._device = device
    self._lock = threading.Lock()

  def __enter__(self) -> jax.Device:
    self._lock.acquire()
    return self._device

  def __exit__(self, exc_type, exc_value, traceback):
    del exc_type, exc_value, traceback
    self._lock.release()


class AlphaGenomeModel(dna_model.DnaModel):
  """Abstract class for DNA sequence models."""

  def __init__(
      self,
      *,
      params: hk.Params,
      state: hk.State,
      apply_fn: ApplyFn,
      junctions_apply_fn: JunctionsApplyFn,
      trunk_apply_fn: TrunkApplyFn | None = None,
      heads_apply_fn: HeadsApplyFn | None = None,
      metadata: Mapping[dna_model.Organism, AlphaGenomeOutputMetadata],
      fasta_extractors: (
          Mapping[dna_model.Organism, fasta.FastaExtractor] | None
      ) = None,
      splice_site_extractors: (
          Mapping[dna_model.Organism, splicing_io.SpliceSiteAnnotationExtractor]
          | None
      ) = None,
      gtfs: Mapping[dna_model.Organism, pd.DataFrame] | None = None,
      pas_gtfs: Mapping[dna_model.Organism, pd.DataFrame] | None = None,
      num_splice_sites: int = 512,
      splice_site_threshold: float = 0.1,
      enable_indel_stitching: bool = True,
      calibration_scorers: (
          Mapping[dna_model.Organism, calibration.CalibrationScorer] | None
      ) = None,
      device: jax.Device | None = None,
  ):
    """Initializes the AlphaGenomeModel.

    Args:
      params: Model parameters.
      state: Model state.
      apply_fn: A function that takes model parameters, state, DNA sequence and
        organism index; and returns the model's predictions.
      junctions_apply_fn: A function that takes model parameters, state,
        embeddings and splice site positions; and returns the model's junctions
        predictions.
      trunk_apply_fn: A function that takes model parameters, state, DNA
        sequence, and organism index; and returns the model trunk's embeddings.
      heads_apply_fn: A function that takes model parameters, state, trunk
        embeddings, sequence embeddings, junction predictions, and organism
        index; and returns predictions from the heads.
      metadata: A mapping of organism to OutputMetadata.
      fasta_extractors: Optional mapping of organism to FastaExtractor. If not
        provided, functions that require sequence extraction will fail.
      splice_site_extractors: Optional mapping of organism to
        SpliceSiteAnnotationExtractor. If not provided, reference splice sits
        will not be used in variant predictions.
      gtfs: Optional mapping of organism to GENCODE GTF Pandas dataframe. If not
        provided, variant scorers that require GTFs will not be available.
      pas_gtfs: Optional mapping of organism to polyadenylation annotation
        dataframe. If not provided, variant scorers that require polyadenylation
        annotations will not be available.
      num_splice_sites: The maximum number of splice sites that are extracted
        from the splice site classification predictions.
      splice_site_threshold: The threshold to use for splice site prediction.
      enable_indel_stitching: Whether to enable indel stitching for variant
        scoring on indels. See `indel_stitch_utils.py` for more details.
      calibration_scorers: Optional mapping of organism to CalibrationScorer. If
        not provided, calibration will not be available.
      device: Optional device to use for model prediction. If None, the first
        local device will be used.
    """
    if device is None:
      device = jax.default_device.value or jax.local_devices()[0]
      if device.platform not in {'gpu', 'tpu'}:
        raise ValueError(
            'Cannot find any GPU or TPU devices. We strondly recommend running'
            ' on GPU or TPU, but if you wish to run on CPU, please explicitly'
            ' pass the JAX device to run the model.'
        )
    self._device_context = _DeviceContextManager(device)
    self._params = jax.device_put(params, device)
    self._state = jax.device_put(state, device)
    self._metadata = metadata
    self._one_hot_encoder = one_hot_encoder.DNAOneHotEncoder()
    self._fasta_extractors = fasta_extractors or {}
    self._splice_site_extractors = splice_site_extractors or {}
    self._enable_indel_stitching = enable_indel_stitching
    self._calibration_scorers = calibration_scorers or {}
    self._trunk_apply_fn = trunk_apply_fn
    self._heads_apply_fn = heads_apply_fn

    self._predict = jax.jit(
        functools.partial(_predict, apply_fn=apply_fn),
        static_argnames=['requested_outputs'],
    )
    self._predict_variant = jax.jit(
        functools.partial(
            _predict_variant,
            apply_fn=jax.jit(apply_fn),
            junctions_apply_fn=jax.jit(junctions_apply_fn),
            trunk_apply_fn=jax.jit(trunk_apply_fn) if trunk_apply_fn else None,
            heads_apply_fn=jax.jit(heads_apply_fn) if heads_apply_fn else None,
            num_splice_sites=num_splice_sites,
            splice_site_threshold=splice_site_threshold,
        ),
        static_argnames=['requested_outputs'],
    )

    # Metadata for each organism without padding.
    self._output_metadata_by_organism = {}

    gtfs = gtfs or {}
    pas_gtfs = pas_gtfs or {}

    for organism, organism_metadata in self._metadata.items():
      masks = jax.tree.map(np.logical_not, organism_metadata.padding)
      output_metadata = {
          output_type.name.lower(): m[masks[output_type]]
          for output_type in dna_output.OutputType
          if (m := organism_metadata.get(output_type)) is not None
      }
      self._output_metadata_by_organism[organism] = dna_output.OutputMetadata(
          **output_metadata
      )

    self._variant_scorers = {}
    self._gene_mask_extractors = {}

    for organism in metadata.keys():
      self._variant_scorers[organism] = {
          variant_scorers_lib.BaseVariantScorer.CENTER_MASK: (
              center_mask.CenterMaskVariantScorer()
          ),
          variant_scorers_lib.BaseVariantScorer.CONTACT_MAP: (
              contact_map.ContactMapScorer()
          ),
      }
      if (gtf := gtfs.get(organism)) is not None:
        self._gene_mask_extractors[organism] = (
            gene_mask_extractor.GeneMaskExtractor(
                gtf=gtf,
                gene_query_type=(
                    gene_mask_extractor.GeneQueryType.VARIANT_OVERLAPPING
                ),
                gene_mask_type=gene_mask_extractor.GeneMaskType.BODY,
            )
        )

        gene_scorer = gene_mask.GeneVariantScorer(
            gene_mask_extractor=gene_mask_extractor.GeneMaskExtractor(
                gtf=gtf,
                gene_query_type=(
                    gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED
                ),
                gene_mask_type=gene_mask_extractor.GeneMaskType.EXONS,
            ),
        )
        self._variant_scorers[organism][
            variant_scorers_lib.BaseVariantScorer.GENE_MASK_LFC
        ] = gene_scorer
        self._variant_scorers[organism][
            variant_scorers_lib.BaseVariantScorer.GENE_MASK_ACTIVE
        ] = gene_scorer
        self._variant_scorers[organism][
            variant_scorers_lib.BaseVariantScorer.GENE_MASK_SPLICING
        ] = gene_mask.GeneVariantScorer(
            gene_mask_extractor=self._gene_mask_extractors[organism]
        )
        self._variant_scorers[organism][
            variant_scorers_lib.BaseVariantScorer.SPLICE_JUNCTION
        ] = splice_junction.SpliceJunctionVariantScorer(gtf)
        if (pas_gtf := pas_gtfs.get(organism)) is not None:
          self._variant_scorers[organism][
              variant_scorers_lib.BaseVariantScorer.PA_QTL
          ] = polyadenylation.PolyadenylationVariantScorer(gtf, pas_gtf)

    self._interval_scorers = {}

    for organism in metadata.keys():
      self._interval_scorers[organism] = {}
      if (gtf := gtfs.get(organism)) is not None:
        self._interval_scorers[organism][
            interval_scorers_lib.BaseIntervalScorer.GENE_MASK
        ] = interval_gene_mask.GeneIntervalScorer(
            gene_mask_extractor=gene_mask_extractor.GeneMaskExtractor(
                gtf=gtf,
                gene_query_type=(
                    gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED
                ),
                gene_mask_type=gene_mask_extractor.GeneMaskType.EXONS,
            ),
        )

  def _get_fasta_extractor(
      self, organism: dna_model.Organism
  ) -> fasta.FastaExtractor:
    """Returns the FastaExtractor for a given organism."""
    if extractor := self._fasta_extractors.get(organism):
      return extractor
    else:
      raise ValueError(f'FastaExtractor not found for {organism.name=}')

  def predict_sequence(
      self,
      sequence: str,
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
      interval: genome.Interval | None = None,
  ) -> dna_output.Output:
    """Predicts the sequence."""
    requested_outputs = tuple(set(requested_outputs))
    ontology_terms = _convert_ontology_terms(ontology_terms)
    metadata = self._metadata[organism]
    track_masks = metadata_lib.create_track_masks(
        metadata,
        requested_outputs=requested_outputs,
        requested_ontologies=ontology_terms,
    )

    with self._device_context as device, jax.transfer_guard('disallow'):
      organism_index = jax.device_put(
          np.full((1,), convert_to_organism_index(organism), dtype=np.int32),
          device,
      )
      sequence = jax.device_put(
          np.asarray(self._one_hot_encoder.encode(sequence))[np.newaxis], device
      )
      predictions = self._predict(
          self._params,
          self._state,
          sequence,
          organism_index,
          requested_outputs=requested_outputs,
          negative_strand_mask=jax.device_put(np.asarray([False]), device),
          strand_reindexing=jax.device_put(metadata.strand_reindexing, device),
      )
      predictions = _filter_predictions(
          predictions, track_masks=jax.device_put(track_masks, device)
      )
      predictions = _upcast_single_batch_predictions(predictions)
      return _construct_output_from_predictions(
          predictions,
          track_masks=track_masks,
          metadata=metadata,
          interval=interval,
      )

  def predict_interval(
      self,
      interval: genome.Interval,
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
  ) -> dna_output.Output:
    requested_outputs = tuple(set(requested_outputs))
    ontology_terms = _convert_ontology_terms(ontology_terms)
    sequence = self._get_fasta_extractor(organism).extract(interval)
    metadata = self._metadata[organism]
    track_masks = metadata_lib.create_track_masks(
        metadata,
        requested_outputs=requested_outputs,
        requested_ontologies=ontology_terms,
    )

    with self._device_context as device, jax.transfer_guard('disallow'):
      organism_index = jax.device_put(
          np.full((1,), convert_to_organism_index(organism), dtype=np.int32),
          device,
      )
      sequence = jax.device_put(
          np.asarray(self._one_hot_encoder.encode(sequence))[np.newaxis], device
      )
      predictions = self._predict(
          self._params,
          self._state,
          sequence,
          organism_index,
          requested_outputs=requested_outputs,
          negative_strand_mask=jax.device_put(
              np.asarray([interval.negative_strand]), device
          ),
          strand_reindexing=jax.device_put(metadata.strand_reindexing, device),
      )
      predictions = _filter_predictions(
          predictions, track_masks=jax.device_put(track_masks, device)
      )
      predictions = _upcast_single_batch_predictions(predictions)
      return _construct_output_from_predictions(
          predictions,
          track_masks=track_masks,
          metadata=metadata,
          interval=interval,
      )

  def predict_variant(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      requested_outputs: Iterable[dna_output.OutputType],
      ontology_terms: Iterable[ontology.OntologyTerm | str] | None,
  ) -> dna_output.VariantOutput:
    requested_outputs = tuple(set(requested_outputs))
    ontology_terms = _convert_ontology_terms(ontology_terms)
    reference_sequence, alternate_sequence = (
        genome_io.extract_variant_sequences(
            interval, variant, self._get_fasta_extractor(organism)
        )
    )
    metadata = self._metadata[organism]
    track_masks = metadata_lib.create_track_masks(
        metadata,
        requested_outputs=requested_outputs,
        requested_ontologies=ontology_terms,
    )

    reference_gene_mask = np.ones((1, interval.width, 1), dtype=bool)
    if extractor := self._gene_mask_extractors.get(organism):
      mask, _ = extractor.extract(interval, variant)
      if mask.size > 0:
        reference_gene_mask = mask.max(-1, keepdims=True)[np.newaxis]

    splice_sites = None
    if splice_site_extractor := self._splice_site_extractors.get(organism):
      splice_sites = splice_site_extractor.extract(interval)[np.newaxis]
      splice_sites *= reference_gene_mask

    splice_junction_masks = _SpliceJunctionVariantMasks(
        splice_sites=splice_sites,
        reference_genes=reference_gene_mask,
        indel_masks=jax.tree.map(
            lambda x: x[np.newaxis],
            variant_scoring.IndelMask.from_variant(variant, interval),
        ),
    )

    reference_sequence_encoded = np.asarray(
        self._one_hot_encoder.encode(reference_sequence)
    )[np.newaxis]
    alternate_sequence_encoded = np.asarray(
        self._one_hot_encoder.encode(alternate_sequence)
    )[np.newaxis]

    indel_stitch_input = None
    if (
        self._trunk_apply_fn is not None
        and self._heads_apply_fn is not None
        and (variant.is_deletion or variant.is_insertion)
    ):
      indel_stitch_input = _extract_indel_stitch_input_single(
          variant,
          interval,
          self._get_fasta_extractor(organism),
          self._one_hot_encoder,
          reference_sequence_encoded,
          alternate_sequence_encoded,
      )

    with self._device_context as device, jax.transfer_guard('disallow'):
      reference_sequence_dev = jax.device_put(
          reference_sequence_encoded, device
      )
      alternate_sequence_dev = jax.device_put(
          alternate_sequence_encoded, device
      )
      organism_indices = jax.device_put(
          np.full((1,), convert_to_organism_index(organism), dtype=np.int32),
          device,
      )

      reference_predictions, alt_predictions = self._predict_variant(
          self._params,
          self._state,
          reference_sequence_dev,
          alternate_sequence_dev,
          jax.device_put(splice_junction_masks, device),
          organism_indices,
          requested_outputs=requested_outputs,
          negative_strand_mask=jax.device_put(
              np.asarray([interval.negative_strand]), device
          ),
          strand_reindexing=jax.device_put(metadata.strand_reindexing, device),
          indel_stitch_input=jax.device_put(indel_stitch_input, device)
          if indel_stitch_input
          else None,
      )
      reference_predictions, alt_predictions = _filter_variant_predictions(
          reference_predictions,
          alt_predictions,
          track_masks=jax.device_put(track_masks, device),
      )
      reference_predictions, alt_predictions = _upcast_single_batch_predictions(
          (reference_predictions, alt_predictions)
      )

      return dna_output.VariantOutput(
          reference=_construct_output_from_predictions(
              reference_predictions,
              track_masks=track_masks,
              metadata=metadata,
              interval=interval,
          ),
          alternate=_construct_output_from_predictions(
              alt_predictions,
              track_masks=track_masks,
              metadata=metadata,
              interval=interval,
          ),
      )

  def score_interval(
      self,
      interval: genome.Interval,
      interval_scorers: Sequence[interval_scorers_lib.IntervalScorerTypes] = (),
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
  ) -> list[anndata.AnnData]:
    if not interval_scorers:
      interval_scorers = list(
          interval_scorers_lib.RECOMMENDED_INTERVAL_SCORERS.values()
      )

    sequence = self._get_fasta_extractor(organism).extract(interval)
    sequence = np.array(self._one_hot_encoder.encode(sequence))[np.newaxis]
    organism_indices = np.full(
        (1,), convert_to_organism_index(organism), dtype=np.int32
    )

    requested_outputs = tuple(
        {scorer.requested_output for scorer in interval_scorers}
    )

    track_metadata = self._metadata[organism]
    track_masks = metadata_lib.create_track_masks(
        track_metadata,
        requested_outputs=requested_outputs,
        requested_ontologies=None,
    )

    with self._device_context as device, jax.transfer_guard('disallow'):
      predictions = self._predict(
          self._params,
          self._state,
          jax.device_put(sequence, device),
          jax.device_put(organism_indices, device),
          negative_strand_mask=jax.device_put(
              np.asarray([interval.negative_strand]), device
          ),
          strand_reindexing=jax.device_put(
              track_metadata.strand_reindexing, device
          ),
          requested_outputs=requested_outputs,
      )
      predictions = _filter_predictions(
          predictions,
          track_masks=jax.device_put(track_masks, device),
      )
      predictions = _upcast_single_batch_predictions(
          predictions,
          transfer_to_host=False,
      )
      output_metadata = self.output_metadata(organism)

      results = []
      # Predictions are reverse-complemented by the model before scoring.
      interval = interval.as_unstranded()
      for scorer_settings in interval_scorers:
        scorer = self._interval_scorers[organism][
            scorer_settings.base_interval_scorer
        ]
        masks, metadata = scorer.get_masks_and_metadata(
            interval,
            settings=scorer_settings,
            track_metadata=output_metadata,
        )
        masks = jax.device_put(masks, device)
        scores = scorer.score_interval(
            predictions,
            masks=masks,
            settings=scorer_settings,
            interval=interval,
        )
        result = scorer.finalize_interval(
            jax.device_get(scores),
            track_metadata=output_metadata,
            mask_metadata=metadata,
            settings=scorer_settings,
        )
        result.uns['interval'] = interval
        result.uns['interval_scorer'] = scorer_settings
        results.append(result)

    return results

  def score_variant(
      self,
      interval: genome.Interval,
      variant: genome.Variant,
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
  ) -> list[anndata.AnnData]:
    if not variant_scorers:
      variant_scorers = variant_scorers_lib.get_recommended_scorers(
          organism.to_proto()
      )
    for scorer in variant_scorers:
      if scorer.base_variant_scorer not in self._variant_scorers[organism]:
        raise ValueError(
            f"Scorer '{scorer.base_variant_scorer}' is missing for"
            f' {organism.name=}. This may be due to a missing variant scorer'
            ' metadata.'
        )
    reference_sequence, alternate_sequence = (
        genome_io.extract_variant_sequences(
            interval, variant, self._get_fasta_extractor(organism)
        )
    )
    reference_sequence = np.asarray(
        self._one_hot_encoder.encode(reference_sequence)
    )[np.newaxis]
    alternate_sequence = np.asarray(
        self._one_hot_encoder.encode(alternate_sequence)
    )[np.newaxis]
    organism_indices = np.full(
        (1,), convert_to_organism_index(organism), dtype=np.int32
    )

    reference_gene_mask = np.zeros((1, interval.width, 1), dtype=bool)
    if extractor := self._gene_mask_extractors.get(organism):
      mask, _ = extractor.extract(interval, variant)
      if mask.size > 0:
        reference_gene_mask = mask.max(-1, keepdims=True)[np.newaxis]

    splice_sites = None
    if splice_site_extractor := self._splice_site_extractors.get(organism):
      splice_sites = splice_site_extractor.extract(interval)[np.newaxis]
      splice_sites *= reference_gene_mask

    requested_outputs = tuple(
        {scorer.requested_output for scorer in variant_scorers}
    )

    track_metadata = self._metadata[organism]
    track_masks = metadata_lib.create_track_masks(
        track_metadata,
        requested_outputs=requested_outputs,
        requested_ontologies=None,
    )

    splice_junction_masks = _SpliceJunctionVariantMasks(
        splice_sites=splice_sites,
        reference_genes=reference_gene_mask,
        indel_masks=jax.tree.map(
            lambda x: x[np.newaxis],
            variant_scoring.IndelMask.from_variant(variant, interval),
        ),
    )
    calibration_scorer = self._calibration_scorers.get(organism)

    indel_stitch_input = None
    if (
        self._enable_indel_stitching
        and self._trunk_apply_fn is not None
        and self._heads_apply_fn is not None
        and (variant.is_deletion or variant.is_insertion)
    ):
      indel_stitch_input = _extract_indel_stitch_input_single(
          variant,
          interval,
          self._get_fasta_extractor(organism),
          self._one_hot_encoder,
          reference_sequence,
          alternate_sequence,
      )

    with self._device_context as device, jax.transfer_guard('disallow'):

      reference_predictions, alternate_predictions = self._predict_variant(
          self._params,
          self._state,
          jax.device_put(reference_sequence, device),
          jax.device_put(alternate_sequence, device),
          jax.device_put(splice_junction_masks, device),
          jax.device_put(organism_indices, device),
          requested_outputs=requested_outputs,
          negative_strand_mask=jax.device_put(
              np.asarray([interval.negative_strand]), device
          ),
          strand_reindexing=jax.device_put(
              track_metadata.strand_reindexing, device
          ),
          indel_stitch_input=jax.device_put(indel_stitch_input, device)
          if indel_stitch_input
          else None,
      )
      reference_predictions, alternate_predictions = (
          _filter_variant_predictions(
              reference_predictions,
              alternate_predictions,
              track_masks=jax.device_put(track_masks, device),
          )
      )
      reference_predictions, alternate_predictions = (
          _upcast_single_batch_predictions(
              (reference_predictions, alternate_predictions),
              transfer_to_host=False,
          )
      )
      output_metadata = self.output_metadata(organism)

      results = []
      # Predictions are reverse-complemented by the model before scoring.
      interval = interval.as_unstranded()
      for scorer_settings in variant_scorers:
        scorer = self._variant_scorers[organism][
            scorer_settings.base_variant_scorer
        ]
        masks, metadata = scorer.get_masks_and_metadata(
            interval,
            variant,
            settings=scorer_settings,
            track_metadata=output_metadata,
        )
        scores = scorer.score_variant(
            reference_predictions,
            alternate_predictions,
            masks=jax.device_put(masks, device),
            settings=scorer_settings,
            variant=variant,
            interval=interval,
        )
        result = scorer.finalize_variant(
            jax.device_get(scores),
            track_metadata=output_metadata,
            mask_metadata=metadata,
            settings=scorer_settings,
        )
        result.uns['interval'] = interval
        result.uns['variant'] = variant
        result.uns['variant_scorer'] = scorer_settings

        if (
            calibration_scorer is not None
            and calibration_scorer.has_variant_scorer(scorer_settings.name)
        ):
          result.layers['quantiles'] = calibration_scorer.quantile_scores(
              scorer_settings.name, result
          )

        result = variant_scoring_utils.merge_stranded_gene_tracks(result)
        results.append(result)

    return results

  def score_ism_variants(
      self,
      interval: genome.Interval,
      ism_interval: genome.Interval,
      variant_scorers: Sequence[variant_scorers_lib.VariantScorerTypes] = (),
      *,
      organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
      interval_variant: genome.Variant | None = None,
  ) -> list[list[anndata.AnnData]]:
    """Generate in-silico mutagenesis (ISM) variant scores for a given interval.

    Args:
      interval: DNA interval to make the prediction for.
      ism_interval: Interval to perform ISM.
      variant_scorers: Sequence of variant scorers to use for scoring each
        variant. If no variant scorers are provided, the recommended variant
        scorers for the organism will be used.
      organism: Organism to use for the prediction.
      interval_variant: Optional variant to apply to the sequence. If provided,
        the alternate allele is used for in-silico mutagenesis, otherwise the
        unaltered reference sequence is used.

    Returns:
      List of variant scores for each variant in the ISM interval.
    """
    extractor = self._get_fasta_extractor(organism)
    if interval_variant is not None:
      _, sequence = genome_io.extract_variant_sequences(
          interval, interval_variant, extractor
      )
    else:
      sequence = extractor.extract(interval)

    interval_slice = interval.intersect(ism_interval)
    if interval_slice is None:
      raise ValueError(f'{ism_interval=} not fully contained in {interval=}.')
    start = interval_slice.start - interval.start
    end = interval_slice.end - interval.start
    ism_sequence = sequence[start:end]

    variants = ism.ism_variants(ism_interval, ism_sequence)
    return self.score_variants(
        interval, variants, variant_scorers, organism=organism
    )

  def output_metadata(
      self, organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS
  ) -> dna_output.OutputMetadata:
    """Get the metadata for a given organism.

    Args:
      organism: Organism to get metadata for.

    Returns:
      OutputMetadata for the provided organism.
    """
    return self._output_metadata_by_organism[organism]


@typing.jaxtyped
def _construct_output_from_predictions(
    predictions: Mapping[
        dna_output.OutputType,
        PyTree[Float[Array, '...'] | Int32[Array, '...']],
    ],
    *,
    track_masks: Mapping[dna_output.OutputType, Bool[np.ndarray, '_']],
    metadata: AlphaGenomeOutputMetadata,
    interval: genome.Interval | None = None,
) -> dna_output.Output:
  """Returns a dna_output.Output from model predictions with data on CPU."""

  def _convert_to_track_data(
      output_type: dna_output.OutputType,
  ) -> track_data.TrackData | None:
    track_metadata = metadata.get(output_type)
    prediction = predictions.get(output_type)
    if prediction is None or track_metadata is None:
      # No tracks are predicted for this head.
      return None
    return track_data.TrackData(
        values=jax.device_get(prediction),
        resolution=metadata.resolution(output_type),
        metadata=track_metadata[track_masks[output_type]],
        interval=interval,
    )

  def _convert_to_junction_data() -> junction_data.JunctionData | None:
    """Returns a splice junction prediction."""
    output_type = dna_output.OutputType.SPLICE_JUNCTIONS
    junction_metadata = metadata.get(output_type)
    splice_junction_predictions = predictions.get(output_type)
    if junction_metadata is None or splice_junction_predictions is None:
      return None
    splice_junctions = splice_junction_predictions['predictions']
    splice_site_positions = splice_junction_predictions['splice_site_positions']

    junction_predictions, strands, starts, ends = (
        splice_junction.unstack_junction_predictions(
            jax.device_get(splice_junctions),
            jax.device_get(splice_site_positions),
            interval,
        )
    )
    chromosome = interval.chromosome if interval is not None else ''
    junctions = [
        genome.Junction(chromosome, start, end, strand)
        for start, end, strand in zip(starts, ends, strands)
        if start < end
    ]
    return junction_data.JunctionData(
        junctions=np.asarray(junctions),
        values=junction_predictions,
        metadata=junction_metadata[track_masks[output_type]],
        interval=interval,
    )

  return dna_output.Output(
      atac=_convert_to_track_data(dna_output.OutputType.ATAC),
      dnase=_convert_to_track_data(dna_output.OutputType.DNASE),
      procap=_convert_to_track_data(dna_output.OutputType.PROCAP),
      cage=_convert_to_track_data(dna_output.OutputType.CAGE),
      rna_seq=_convert_to_track_data(dna_output.OutputType.RNA_SEQ),
      chip_histone=_convert_to_track_data(dna_output.OutputType.CHIP_HISTONE),
      chip_tf=_convert_to_track_data(dna_output.OutputType.CHIP_TF),
      contact_maps=_convert_to_track_data(dna_output.OutputType.CONTACT_MAPS),
      splice_sites=_convert_to_track_data(dna_output.OutputType.SPLICE_SITES),
      splice_site_usage=_convert_to_track_data(
          dna_output.OutputType.SPLICE_SITE_USAGE
      ),
      splice_junctions=_convert_to_junction_data(),
  )


def convert_to_organism_index(
    organism: dna_model.Organism,
) -> int:
  """Converts a dna_model.Organism to a organism index."""
  match organism:
    case dna_model.Organism.HOMO_SAPIENS:
      return 0
    case dna_model.Organism.MUS_MUSCULUS:
      return 1
    case _:
      raise ValueError(f'Unsupported organism: {organism}')


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelSettings:
  """Settings for the AlphaGenomeModel."""

  # The maximum number of splice sites that are extracted from the splice site
  # classification predictions.
  num_splice_sites: int = 512
  # The threshold to use for splice site prediction.
  splice_site_threshold: float = 0.1
  # Whether to use indel stitching.
  enable_indel_stitching: bool = True


@dataclasses.dataclass(frozen=True, kw_only=True)
class OrganismSettings:
  """Settings for a specific organism."""

  # Optional output metadata. If None, we load the default AlphaGenome metadata
  # for the organism.
  metadata: AlphaGenomeOutputMetadata | None = None

  # Optional paths to the reference genome and annotation data. If None,
  # functions that accept intervals or variants will fail.
  fasta_path: str | os.PathLike[str] | None = None

  # Optional paths to the reference annotation data. If None, variant scorers
  # that require a GTF will not be available.
  gtf_feather_path: str | os.PathLike[str] | None = None

  # Optional path to reference polyadenylation annotation data. If None,
  # variant scorers that require a PAS GTF will not be available.
  pas_feather_path: str | os.PathLike[str] | None = None

  # Optional paths to the reference splice site data. If None, no reference
  # splice sites will be used during variant predictions.
  splice_site_starts_feather_path: str | os.PathLike[str] | None = None
  splice_site_ends_feather_path: str | os.PathLike[str] | None = None

  # Optional path to variant score calibration. If None, variant scores will not
  # be calibrated.
  calibration_path: str | os.PathLike[str] | None = None


def default_organism_settings() -> (
    Mapping[dna_model.Organism, OrganismSettings]
):
  """Returns the default organism settings for the AlphaGenomeModel."""
  return {
      dna_model.Organism.HOMO_SAPIENS: OrganismSettings(
          fasta_path=(
              'https://storage.googleapis.com/alphagenome/reference/gencode/'
              'hg38/GRCh38.p13.genome.fa'
          ),
          gtf_feather_path=(
              'https://storage.googleapis.com/alphagenome/reference/gencode/'
              'hg38/gencode.v46.annotation.gtf.gz.feather'
          ),
          pas_feather_path=(
              'https://storage.googleapis.com/alphagenome/reference/exon/hg38/'
              'polyadb_human_v3_exon3_contiguous_gtfv46.feather'
          ),
          splice_site_starts_feather_path=(
              'https://storage.googleapis.com/alphagenome/reference/gencode/'
              'hg38/gencode.v46.splice_sites_starts.feather'
          ),
          splice_site_ends_feather_path=(
              'https://storage.googleapis.com/alphagenome/reference/gencode/'
              'hg38/gencode.v46.splice_sites_ends.feather'
          ),
          calibration_path='gs:///alphagenome/data/hg38/calibration_scores.pb',
      ),
      dna_model.Organism.MUS_MUSCULUS: OrganismSettings(
          fasta_path=(
              'https://storage.googleapis.com/alphagenome/reference/gencode/'
              'mm10/GRCm38.p6.genome.fa'
          ),
          gtf_feather_path=(
              'https://storage.googleapis.com/alphagenome/reference/gencode/'
              'mm10/gencode.vM23.annotation.gtf.gz.feather'
          ),
          pas_feather_path=None,
          splice_site_starts_feather_path=(
              'https://storage.googleapis.com/alphagenome/reference/gencode/'
              'mm10/gencode.vM23.splice_sites_starts.feather'
          ),
          splice_site_ends_feather_path=(
              'https://storage.googleapis.com/alphagenome/reference/gencode/'
              'mm10/gencode.vM23.splice_sites_ends.feather'
          ),
          calibration_path=None,
      ),
  }


@typing.jaxtyped
def create_model(
    metadata: Mapping[dna_model.Organism, AlphaGenomeOutputMetadata],
    *,
    num_splice_sites: int = model.DEFAULT_NUM_SPLICE_SITES,
    splice_site_threshold: float = model.DEFAULT_SPLICE_SITE_THRESHOLD,
) -> tuple[
    Callable[
        [chex.PRNGKey, Float[Array, 'B S 4'], Int32[Array, 'B']],
        tuple[hk.Params, hk.State],
    ],
    ApplyFn,
    Callable[
        [hk.Params, hk.State, Float[Array, 'B S 4'], Int32[Array, 'B']],
        embeddings_lib.Embeddings,
    ],
    Callable[
        [
            hk.Params,
            hk.State,
            Float[Array, 'B S D1'],
            Float[Array, 'B S128 D2'],
            Float[Array, 'B Sp Sp Dp'],
            Int32[Array, 'B'],
        ],
        PyTree[Shaped[Array, 'B ...']],
    ],
    JunctionsApplyFn,
]:
  """Helper to create AlphaGenome init and four apply functions."""

  jmp_policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')

  @hk.transform_with_state
  def _forward(
      dna_sequence: Float[Array, 'B S 4'],
      organism_index: Int32[Array, 'B'],
  ):
    """AlphaGenome default forward pass."""
    with hk.mixed_precision.push_policy(model.AlphaGenome, jmp_policy):
      return model.AlphaGenome(
          metadata,
          num_splice_sites=num_splice_sites,
          splice_site_threshold=splice_site_threshold,
      )(dna_sequence, organism_index)

  @hk.transform_with_state
  def _forward_trunk(
      dna_sequence: Float[Array, 'B S 4'],
      organism_index: Int32[Array, 'B'],
  ):
    """AlphaGenome trunk-only forward pass."""
    with hk.mixed_precision.push_policy(model.AlphaGenome, jmp_policy):
      return model.AlphaGenome(
          metadata,
          num_splice_sites=num_splice_sites,
          splice_site_threshold=splice_site_threshold,
      ).forward_trunk(dna_sequence, organism_index)

  @hk.transform_with_state
  def _forward_heads(
      embeddings_1bp: Float[Array, 'B S D1'],
      embeddings_128bp: Float[Array, 'B S//128 D2'],
      embeddings_pair: Float[Array, 'B Sp Sp Dp'],
      organism_index: Int32[Array, 'B'],
  ):
    """AlphaGenome heads-only forward pass."""
    with hk.mixed_precision.push_policy(model.AlphaGenome, jmp_policy):
      return model.AlphaGenome(
          metadata,
          num_splice_sites=num_splice_sites,
          splice_site_threshold=splice_site_threshold,
      ).forward_heads(
          embeddings_lib.Embeddings(
              embeddings_1bp=embeddings_1bp,
              embeddings_128bp=embeddings_128bp,
              embeddings_pair=embeddings_pair,
          ),
          organism_index,
      )

  @hk.transform_with_state
  def _forward_junctions(
      trunk_embeddings, splice_site_positions, organism_index
  ):
    with hk.mixed_precision.push_policy(model.AlphaGenome, jmp_policy):
      return model.AlphaGenome(
          metadata,
          num_splice_sites=num_splice_sites,
          splice_site_threshold=splice_site_threshold,
      ).predict_junctions(
          trunk_embeddings, splice_site_positions, organism_index
      )

  def _apply_fn(
      params: hk.Params,
      state: hk.State,
      dna_sequence: Float[Array, 'B S 4'],
      organism_index: Int32[Array, 'B'],
  ) -> PyTree[Shaped[Array, 'B ...']]:
    """AlphaGenome default apply function."""
    (predictions, _), _ = _forward.apply(
        params, state, None, dna_sequence, organism_index
    )
    return predictions

  def _trunk_apply_fn(
      params: hk.Params,
      state: hk.State,
      dna_sequence: Float[Array, 'B S 4'],
      organism_index: Int32[Array, 'B'],
  ) -> embeddings_lib.Embeddings:
    """AlphaGenome trunk apply function."""
    embeddings, _ = _forward_trunk.apply(
        params, state, None, dna_sequence, organism_index
    )
    return embeddings

  def _heads_apply_fn(
      params: hk.Params,
      state: hk.State,
      embeddings_1bp: Float[Array, 'B S D1'],
      embeddings_128bp: Float[Array, 'B S128 D2'],
      embeddings_pair: Float[Array, 'B Sp Sp Dp'],
      organism_index: Int32[Array, 'B'],
  ) -> PyTree[Shaped[Array, 'B ...']]:
    """AlphaGenome heads apply function."""
    predictions, _ = _forward_heads.apply(
        params,
        state,
        None,
        embeddings_1bp,
        embeddings_128bp,
        embeddings_pair,
        organism_index,
    )
    return predictions

  def _junctions_apply_fn(
      params: hk.Params,
      state: hk.State,
      trunk_embeddings: Float[Array, 'B S D'],
      splice_site_positions: Int32[Array, 'B 4 K'],
      organism_index: Int32[Array, 'B'],
  ) -> BatchPrediction:
    """AlphaGenome junctions apply function."""
    predictions, _ = _forward_junctions.apply(
        params,
        state,
        None,
        trunk_embeddings,
        splice_site_positions,
        organism_index,
    )
    return predictions

  return (
      _forward.init,
      _apply_fn,
      _trunk_apply_fn,
      _heads_apply_fn,
      _junctions_apply_fn,
  )


def create(
    checkpoint_path: str | os.PathLike[str],
    *,
    organism_settings: (
        Mapping[dna_model.Organism, OrganismSettings] | None
    ) = None,
    model_settings: ModelSettings = ModelSettings(),
    device: jax.Device | None = None,
) -> AlphaGenomeModel:
  """Returns a AlphaGenomeModel from a checkpoint stored at the given path.

  Args:
    checkpoint_path: Path to the checkpoint to load.
    organism_settings: Optional organism settings to use. If not set, will use
      default organism settings.
    model_settings: Settings for the model. If not set, will use default model
      settings.
    device: Optional device to use for model prediction. If None, the first
      local device will be used.
  """
  if organism_settings is None:
    organism_settings = default_organism_settings()

  metadata = {}
  fasta_extractors = {}
  splice_site_extractors = {}
  calibration_scorers = {}
  gtfs = {}
  pas_gtfs = {}
  validate_checkpoint = True

  for organism, settings in organism_settings.items():
    if settings.metadata is not None:
      metadata[organism] = settings.metadata
      validate_checkpoint = False
    else:
      metadata[organism] = metadata_lib.load(organism)

    if settings.fasta_path is not None:
      fasta_extractors[organism] = fasta.FastaExtractor(settings.fasta_path)
    if settings.gtf_feather_path is not None:
      gtfs[organism] = pd.read_feather(settings.gtf_feather_path)
    if settings.pas_feather_path is not None:
      pas_gtfs[organism] = pd.read_feather(settings.pas_feather_path)
    if settings.splice_site_starts_feather_path is not None:
      assert settings.splice_site_ends_feather_path is not None
      splice_site_extractors[organism] = (
          splicing_io.SpliceSiteAnnotationExtractor(
              junction_starts=pd.read_feather(
                  settings.splice_site_starts_feather_path
              ),
              junction_ends=pd.read_feather(
                  settings.splice_site_ends_feather_path
              ),
          )
      )
    if settings.calibration_path is not None:
      calibration_scorers[organism] = calibration.load(
          settings.calibration_path
      )

  init_fn, apply_fn, trunk_apply_fn, heads_apply_fn, junctions_apply_fn = (
      create_model(
          metadata,
          num_splice_sites=model_settings.num_splice_sites,
          splice_site_threshold=model_settings.splice_site_threshold,
      )
  )

  dna_sequence_shape = jax.ShapeDtypeStruct((1, 2048, 4), dtype=jnp.float32)
  organism_index_shape = jax.ShapeDtypeStruct((1,), dtype=jnp.int32)
  target_shapes = jax.eval_shape(
      init_fn, jax.random.PRNGKey(0), dna_sequence_shape, organism_index_shape
  )
  checkpointer = ocp.StandardCheckpointer()
  params, state = checkpointer.restore(
      checkpoint_path,
      target=target_shapes if validate_checkpoint else None,
      strict=validate_checkpoint,
  )

  return AlphaGenomeModel(
      params=params,
      state=state,
      apply_fn=apply_fn,
      junctions_apply_fn=junctions_apply_fn,
      trunk_apply_fn=trunk_apply_fn,
      heads_apply_fn=heads_apply_fn,
      metadata=metadata,
      fasta_extractors=fasta_extractors,
      splice_site_extractors=splice_site_extractors,
      gtfs=gtfs,
      pas_gtfs=pas_gtfs,
      num_splice_sites=model_settings.num_splice_sites,
      splice_site_threshold=model_settings.splice_site_threshold,
      enable_indel_stitching=model_settings.enable_indel_stitching,
      calibration_scorers=calibration_scorers,
      device=device,
  )


def create_from_kaggle(
    model_version: str | ModelVersion,
    *,
    organism_settings: (
        Mapping[dna_model.Organism, OrganismSettings] | None
    ) = None,
    device: jax.Device | None = None,
) -> AlphaGenomeModel:
  """Helper function to create a model from Kaggle.

  Args:
    model_version: The model version to use, e.g. all_folds.
    organism_settings: Optional organism settings to use. If unset, will use
      default organism settings.
    device: Optional device to use for model prediction. If None, the first
      local device will be used.

  Returns:
    AlphaGenomeModel created from the Kaggle checkpoint.
  """
  if kaggle_auth.get_username() is None:
    kagglehub.login()

  if isinstance(model_version, ModelVersion):
    model_version = model_version.name

  checkpoint_path = kagglehub.model_download(
      f'google/alphagenome/jax/{model_version.lower()}'
  )
  return create(
      checkpoint_path, organism_settings=organism_settings, device=device
  )


def create_from_huggingface(
    model_version: str | ModelVersion,
    *,
    organism_settings: (
        Mapping[dna_model.Organism, OrganismSettings] | None
    ) = None,
    device: jax.Device | None = None,
) -> AlphaGenomeModel:
  """Helper function to create a DNA model from HuggingFace.

  Args:
    model_version: The model version to use, e.g. all_folds.
    organism_settings: Optional organism settings to use. If unset, will use
      default organism settings.
    device: Optional device to use for model prediction. If None, the first
      local device will be used.

  Returns:
    AlphaGenomeModel created from the Hugging Face checkpoint.
  """

  try:
    huggingface_hub.whoami()
  except huggingface_hub.errors.LocalTokenNotFoundError:
    huggingface_hub.login()

  if isinstance(model_version, ModelVersion):
    model_version = model_version.name

  checkpoint_path = huggingface_hub.snapshot_download(
      repo_id=f'google/alphagenome-{model_version.replace("_", "-").lower()}'
  )
  return create(
      checkpoint_path, organism_settings=organism_settings, device=device
  )
