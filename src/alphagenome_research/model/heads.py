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

"""Heads for AlphaGenome."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import functools
import math
from alphagenome import typing
from alphagenome.data import junction_data
from alphagenome.data import track_data
from alphagenome.models import dna_model
from alphagenome.models import dna_output
from alphagenome_research.io import bundles
from alphagenome_research.model import attention
from alphagenome_research.model import embeddings as embeddings_module
from alphagenome_research.model import losses
from alphagenome_research.model import schemas
from alphagenome_research.model.metadata import metadata as metadata_lib
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PyTree, Shaped  # pylint: disable=g-importing-member, g-multiple-import
import numpy as np

_SOFT_CLIP_VALUE = 10.0


class HeadType(enum.Enum):
  """Head types."""

  GENOME_TRACKS = 'genome_tracks'
  CONTACT_MAPS = 'contact_maps'
  SPLICE_SITES_CLASSIFICATION = 'splice_sites_classification'
  SPLICE_SITES_USAGE = 'splice_sites_usage'
  SPLICE_SITES_JUNCTION = 'splice_sites_junction'


class HeadName(enum.Enum):
  """Output heads."""

  ATAC = 'atac'
  DNASE = 'dnase'
  PROCAP = 'procap'
  CAGE = 'cage'
  RNA_SEQ = 'rna_seq'
  CHIP_TF = 'chip_tf'
  CHIP_HISTONE = 'chip_histone'
  CONTACT_MAPS = 'contact_maps'
  SPLICE_SITES_CLASSIFICATION = 'splice_sites_classification'
  SPLICE_SITES_USAGE = 'splice_sites_usage'
  SPLICE_SITES_JUNCTION = 'splice_sites_junction'


@dataclasses.dataclass
class HeadConfig:
  type: HeadType
  name: str
  output_type: dna_output.OutputType


@dataclasses.dataclass
class GenomeTracksHeadConfig(HeadConfig):
  resolutions: Sequence[int]
  apply_squashing: bool
  bundle: bundles.BundleName


def create_head(
    config: HeadConfig,
    metadata: Mapping[
        dna_model.Organism, metadata_lib.AlphaGenomeOutputMetadata
    ],
    *,
    num_organisms: int | None = None,
) -> 'Head':
  match config.type:
    case HeadType.GENOME_TRACKS:
      assert isinstance(config, GenomeTracksHeadConfig)
      return GenomeTracksHead(
          name=config.name,
          output_type=config.output_type,
          metadata=metadata,
          resolutions=config.resolutions,
          apply_squashing=config.apply_squashing,
          bundle=config.bundle,
          num_organisms=num_organisms,
      )
    case HeadType.CONTACT_MAPS:
      return ContactMapsHead(
          name=config.name,
          output_type=config.output_type,
          metadata=metadata,
          num_organisms=num_organisms,
      )
    case HeadType.SPLICE_SITES_CLASSIFICATION:
      return SpliceSitesClassificationHead(
          name=config.name,
          output_type=config.output_type,
          metadata=metadata,
          num_organisms=num_organisms,
      )
    case HeadType.SPLICE_SITES_USAGE:
      return SpliceSitesUsageHead(
          name=config.name,
          output_type=config.output_type,
          metadata=metadata,
          num_organisms=num_organisms,
      )
    case HeadType.SPLICE_SITES_JUNCTION:
      return SpliceSitesJunctionHead(
          name=config.name,
          output_type=config.output_type,
          metadata=metadata,
          num_organisms=num_organisms,
      )
    case _:
      raise ValueError(f'Unknown head type: {config.type}')


def get_head_config(head_name: HeadName) -> HeadConfig:
  """Returns a head for the given head name."""
  match head_name:
    case HeadName.ATAC:
      return GenomeTracksHeadConfig(
          type=HeadType.GENOME_TRACKS,
          name=HeadName.ATAC.value,
          output_type=dna_output.OutputType.ATAC,
          resolutions=[1, 128],
          apply_squashing=False,
          bundle=bundles.BundleName.ATAC,
      )
    case HeadName.DNASE:
      return GenomeTracksHeadConfig(
          type=HeadType.GENOME_TRACKS,
          name=HeadName.DNASE.value,
          output_type=dna_output.OutputType.DNASE,
          resolutions=[1, 128],
          apply_squashing=False,
          bundle=bundles.BundleName.DNASE,
      )
    case HeadName.PROCAP:
      return GenomeTracksHeadConfig(
          type=HeadType.GENOME_TRACKS,
          name=HeadName.PROCAP.value,
          output_type=dna_output.OutputType.PROCAP,
          resolutions=[1, 128],
          apply_squashing=False,
          bundle=bundles.BundleName.PROCAP,
      )
    case HeadName.CAGE:
      return GenomeTracksHeadConfig(
          type=HeadType.GENOME_TRACKS,
          name=HeadName.CAGE.value,
          output_type=dna_output.OutputType.CAGE,
          resolutions=[1, 128],
          apply_squashing=False,
          bundle=bundles.BundleName.CAGE,
      )
    case HeadName.RNA_SEQ:
      return GenomeTracksHeadConfig(
          type=HeadType.GENOME_TRACKS,
          name=HeadName.RNA_SEQ.value,
          output_type=dna_output.OutputType.RNA_SEQ,
          resolutions=[1, 128],
          apply_squashing=True,
          bundle=bundles.BundleName.RNA_SEQ,
      )
    case HeadName.CHIP_TF:
      return GenomeTracksHeadConfig(
          type=HeadType.GENOME_TRACKS,
          name=HeadName.CHIP_TF.value,
          output_type=dna_output.OutputType.CHIP_TF,
          resolutions=[128],
          apply_squashing=False,
          bundle=bundles.BundleName.CHIP_TF,
      )
    case HeadName.CHIP_HISTONE:
      return GenomeTracksHeadConfig(
          type=HeadType.GENOME_TRACKS,
          name=HeadName.CHIP_HISTONE.value,
          output_type=dna_output.OutputType.CHIP_HISTONE,
          resolutions=[128],
          apply_squashing=False,
          bundle=bundles.BundleName.CHIP_HISTONE,
      )
    case HeadName.CONTACT_MAPS:
      return HeadConfig(
          type=HeadType.CONTACT_MAPS,
          name=HeadName.CONTACT_MAPS.value,
          output_type=dna_output.OutputType.CONTACT_MAPS,
      )
    case HeadName.SPLICE_SITES_CLASSIFICATION:
      return HeadConfig(
          type=HeadType.SPLICE_SITES_CLASSIFICATION,
          name=HeadName.SPLICE_SITES_CLASSIFICATION.value,
          output_type=dna_output.OutputType.SPLICE_SITES,
      )
    case HeadName.SPLICE_SITES_USAGE:
      return HeadConfig(
          type=HeadType.SPLICE_SITES_USAGE,
          name=HeadName.SPLICE_SITES_USAGE.value,
          output_type=dna_output.OutputType.SPLICE_SITE_USAGE,
      )
    case HeadName.SPLICE_SITES_JUNCTION:
      return HeadConfig(
          type=HeadType.SPLICE_SITES_JUNCTION,
          name=HeadName.SPLICE_SITES_JUNCTION.value,
          output_type=dna_output.OutputType.SPLICE_JUNCTIONS,
      )
    case _:
      raise ValueError(f'Unknown head name: {head_name}')


@typing.jaxtyped
def _sum_pool(
    x: Float[Array, 'B S C'], width: int
) -> Float[Array, 'B S//{width} C']:
  return x.reshape((x.shape[0], x.shape[1] // width, width, x.shape[2])).sum(
      axis=-2, dtype=jnp.float32
  )


@typing.jaxtyped
def _get_param_for_index(
    params: Float[ArrayLike, 'P ...'], index: Int[Array, 'B']
) -> Float[ArrayLike, 'B ...']:
  """Returns a parameter for a specific index.

  Embeds the params into the graph.

  Args:
    params: The parameters to embed.
    index: The index to get the parameter for.
  """
  return jnp.asarray(params)[(index,)]


class _MultiOrganismLinear(hk.Module):
  """A linear layer with organism-specific weights and biases."""

  def __init__(
      self,
      output_size: int,
      num_organisms: int,
      name: str | None = 'multi_organism_linear',
  ):
    super().__init__(name=name)
    self._output_size = output_size
    self._num_organisms = num_organisms

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B *S D'], organism_index: Int[Array, 'B']
  ) -> Float[Array, 'B *S {self._output_size}']:
    w_shape = (self._num_organisms, x.shape[-1], self._output_size)
    stddev = 1.0 / np.sqrt(x.shape[-1])
    w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter('w', w_shape, init=w_init).astype(x.dtype)
    w = _get_param_for_index(w, organism_index)
    b_shape = (self._num_organisms, self._output_size)
    b = hk.get_parameter('b', b_shape, init=jnp.zeros).astype(x.dtype)
    b = _get_param_for_index(b, organism_index)
    num_inner_dims = len(x.shape) - 2
    target_b_shape = (b.shape[0],) + (1,) * num_inner_dims + (b.shape[1],)
    return jnp.einsum(
        'b...i,bij->b...j', x, w, preferred_element_type=jnp.float32
    ) + b.reshape(target_b_shape)


def predictions_scaling(
    x: Float[ArrayLike, 'B S C'],
    track_means: Float[ArrayLike, 'B C'],
    resolution: int,
    apply_squashing: bool,
    soft_clip_value: float = _SOFT_CLIP_VALUE,
) -> Float[ArrayLike, 'S C']:
  """Scales predictions to experimental data scale.

  Args:
    x: Experimental target counts.
    track_means: Mean values per track (broadcastable).
    resolution: The bin resolution of the targets (e.g., 1 or 128).
    apply_squashing: Whether to apply power law compression (for RNA-seq).
    soft_clip_value: The value to soft clip the predictions to.

  Returns:
    Scaled predictions.
  """
  xnp = jnp if isinstance(x, jnp.ndarray) else np
  x = xnp.where(
      x > soft_clip_value,
      (x + soft_clip_value) ** 2 / (4 * soft_clip_value),
      x,
  )
  if apply_squashing:
    x = xnp.power(x, 1.0 / 0.75)
  x = x * (track_means[:, None] * resolution).astype(x.dtype)
  return x


@typing.jaxtyped
def targets_scaling(
    targets: Float[ArrayLike, 'B S C'],
    track_means: Float[ArrayLike, 'B C'],
    resolution: int,
    apply_squashing: bool,
    soft_clip_value: float = _SOFT_CLIP_VALUE,
) -> Float[Array, 'B S C']:
  """Scales experimental targets to the model prediction space.

  Args:
    targets: Experimental target counts.
    track_means: Mean values per track.
    resolution: The bin resolution of the targets (e.g., 1 or 128).
    apply_squashing: Whether to apply power law compression (for RNA-seq).
    soft_clip_value: The value to soft clip the targets to.

  Returns:
    Scaled targets ready for loss calculation.
  """
  xnp = jnp if isinstance(targets, jnp.ndarray) else np
  targets = targets / (track_means[:, None] * resolution).astype(targets.dtype)

  if apply_squashing:
    targets = xnp.power(targets, 0.75)

  # Where(targets > 10.0, 2 * Sqrt(x * 10.0) - 10.0, targets)
  return xnp.where(
      targets > soft_clip_value,
      2.0 * jnp.sqrt(targets * soft_clip_value) - soft_clip_value,
      targets,
  )


class Head(metaclass=abc.ABCMeta):
  """Abstract class for a model head."""

  def __init__(
      self,
      *,
      name: str,
      output_type: dna_output.OutputType,
      metadata: Mapping[
          dna_model.Organism,
          metadata_lib.AlphaGenomeOutputMetadata,
      ],
      num_organisms: int | None = None,
  ):
    """Initializes the Head class.

    Args:
      name: The name of the head.
      output_type: The type of output to predict.
      metadata: A dictionary of track metadata for each organism. The metadata
        should be a ordered aligned with the organism index. E.g.,
        organism_index=0 should correspond to the first organism in the metadata
        dictionary.
      num_organisms: Optional number of organisms. If not provided, the number
        of organisms will be inferred from the metadata.
    """

    self._name = name
    self._output_type = output_type
    self._metadata = metadata
    self._num_organisms = (
        num_organisms if num_organisms is not None else len(metadata)
    )
    if not self._metadata:
      raise ValueError('No metadata provided for any organism.')
    self._num_tracks = self._get_num_tracks()

  @property
  def name(self) -> str:
    return self._name

  @property
  def num_tracks(self) -> int:
    """Returns the maximum number of tracks for the head across all organisms."""
    return self._num_tracks

  def _get_num_tracks(self) -> int:
    """Returns the number of tracks for the head."""
    num_tracks = []
    for organism in self._metadata.keys():
      if (track_metadata := self.get_metadata(organism)) is not None:
        num_tracks.append(len(track_metadata))

    if not num_tracks:
      raise ValueError(
          f'No metadata found for any organism for {self._output_type=}.'
      )

    if len(set(num_tracks)) > 1:
      raise ValueError(
          'Number of tracks is not the same for all organisms. Please pad the'
          ' metadata to have the same number of tracks for all organisms.'
      )
    return num_tracks[0]

  @typing.jaxtyped
  def get_multi_organism_track_mask(
      self,
  ) -> Bool[Array, '{len(self._metadata)} {self.num_tracks}']:
    """Returns the track mask for all organisms."""
    track_masks = []
    for organism in self._metadata.keys():
      padding = self._metadata[organism].padding[self._output_type]
      chex.assert_shape(padding, (self.num_tracks,))
      track_masks.append(np.logical_not(padding))
    return jnp.stack(track_masks).astype(bool)

  def get_metadata(
      self, organism: dna_model.Organism
  ) -> track_data.TrackMetadata | junction_data.JunctionMetadata | None:
    return self._metadata.get(organism, {}).get(self._output_type)

  def __call__(
      self,
      embeddings: embeddings_module.Embeddings,
      organism_index: Int[Array, 'B'],
      **kwargs,
  ) -> PyTree[Shaped[Array, 'B ...'] | None]:
    """Calls the head's predict function as a module."""
    return hk.to_module(self.predict)(self._name)(
        embeddings, organism_index, **kwargs
    )

  @abc.abstractmethod
  def predict(
      self,
      embeddings: embeddings_module.Embeddings,
      organism_index: Int[Array, 'B'],
      **kwargs,
  ) -> PyTree[Shaped[Array, 'B ...'] | None]:
    """Returns the predictions for the head."""

  @abc.abstractmethod
  def loss(
      self,
      predictions: PyTree[Shaped[Array, 'B ...']],
      batch: schemas.DataBatch,
  ) -> PyTree[Float[Array, '']]:
    """Returns the loss for the head."""


class GenomeTracksHead(Head):
  """A model head that predicts at multiple resolutions.

  This module takes embeddings at different resolutions and produces predictions
  for a specified number of tracks. It uses organism-specific linear layers and
  learnt scales to generate the predictions.
  """

  def __init__(
      self,
      *,
      name: str,
      output_type: dna_output.OutputType,
      apply_squashing: bool,
      resolutions: Sequence[int],
      bundle: bundles.BundleName | None = None,
      metadata: Mapping[
          dna_model.Organism,
          metadata_lib.AlphaGenomeOutputMetadata,
      ],
      num_organisms: int | None = None,
  ):
    """Initializes the BaseResolutionHead module.

    Args:
      name: The name of the head.
      output_type: The type of output to predict.
      apply_squashing: Whether to apply squashing to the predictions.
      resolutions: The resolutions to predict.
      bundle: The name of the dataset bundle associated with this head. This is
        required to fetch the target tracks for loss computation. If None, the
        `loss` method will not be functional.
      metadata: A dictionary mapping each organism to its track metadata. The
        order of organisms in the dictionary is important, as it must align with
        the organism index provided in the data batches (e.g., organism_index=0
        corresponds to the first organism in `metadata`). For each organism, the
        metadata for `output_type` should contain the same number of rows
        (tracks), padded if necessary. If the metadata includes a 'nonzero_mean'
        column, these values are used to scale predictions and targets.
        Otherwise, scaling is omitted. Note that squashing is only applied if
        `apply_squashing` is True.
      num_organisms: Optional number of organisms. If not provided, the number
        of organisms will be inferred from the metadata.
    """

    super().__init__(
        name=name,
        output_type=output_type,
        metadata=metadata,
        num_organisms=num_organisms,
    )
    self._apply_squashing = apply_squashing
    self._resolutions = sorted(resolutions)
    self._bundle = bundle

    def _get_track_means(organism: dna_model.Organism) -> Float[Array, 'C']:
      metadata = self.get_metadata(organism)
      if metadata is None or metadata.get('nonzero_mean') is None:
        return jnp.ones((self.num_tracks,))
      else:
        return metadata['nonzero_mean'].values

    self._track_means = jnp.stack(
        [_get_track_means(organism) for organism in self._metadata.keys()]
    )

  @typing.jaxtyped
  def unscale(
      self,
      x: Float[Array, 'B S C'],
      organism_index: Int[Array, 'B'],
      resolution: int,
  ) -> Float[Array, 'B S C'] | None:
    """Unscales predictions to experimental data scale.

    Requires the column `nonzero_mean` to be present in the metadata to result
    in valid scaling.

    Args:
      x: The predictions to unscale.
      organism_index: The organism index.
      resolution: The resolution of the predictions.
    """
    track_means = _get_param_for_index(self._track_means, organism_index)
    return predictions_scaling(
        x,
        track_means=track_means,
        resolution=resolution,
        apply_squashing=self._apply_squashing,
    )

  @typing.jaxtyped
  def scale(
      self,
      x: Float[Array, 'B S C'],
      organism_index: Int[Array, 'B'],
      resolution: int,
  ) -> Float[Array, 'B S C']:
    """Scales targets to model predictions scale."""
    track_means = _get_param_for_index(self._track_means, organism_index)
    return targets_scaling(
        x,
        track_means=track_means,
        resolution=resolution,
        apply_squashing=self._apply_squashing,
    )

  @hk.transparent
  @typing.jaxtyped
  def _predict(
      self, x: Float[Array, 'B S D'], organism_index: Int[Array, 'B']
  ) -> Float[Array, 'B S {self.num_tracks}']:
    """Predicts genome tracks."""
    x = _MultiOrganismLinear(self.num_tracks, self._num_organisms)(
        x, organism_index
    )
    residual_scales = hk.get_parameter(
        'learnt_scale', (self._num_organisms, self.num_tracks), init=jnp.ones
    ).astype(x.dtype)
    residual_scale = _get_param_for_index(residual_scales, organism_index)
    return jax.nn.softplus(x) * jax.nn.softplus(residual_scale[:, None, :])

  def predict(
      self,
      embeddings: embeddings_module.Embeddings,
      organism_index: Int[Array, 'B'],
      **kwargs,
  ) -> PyTree[Float[Array, 'B ...'] | None]:
    predictions = {}
    for resolution in self._resolutions:
      with hk.name_scope(f'resolution_{resolution}'):
        scaled_predictions = self._predict(
            embeddings.get_sequence_embeddings(resolution), organism_index
        )
        predictions[f'scaled_predictions_{resolution}bp'] = scaled_predictions
        predictions[f'predictions_{resolution}bp'] = self.unscale(
            scaled_predictions, organism_index, resolution
        )
    return predictions

  @typing.jaxtyped
  def _compute_loss(
      self,
      *,
      organism_index: Int[Array, 'B'],
      predictions: Float[Array, 'B S C'],
      targets: Float[Array, 'B S C'],
      targets_mask: Bool[Array, 'B 1 C'] | None,
      resolution: int,
  ) -> PyTree[Float[Array, '']]:
    """Computes the loss for the head at a given resolution."""
    chex.assert_equal_shape([predictions, targets])
    scaled_targets = self.scale(targets, organism_index, resolution)
    all_losses = losses.multinomial_loss(
        y_pred=predictions,
        y_true=scaled_targets,
        mask=targets_mask,
        positional_weight=5.0,
        multinomial_resolution=int(2**17) // resolution,
    )
    return all_losses

  @typing.jaxtyped
  def loss(
      self,
      predictions: PyTree[Float[Array, 'B ...']],
      batch: schemas.DataBatch,
  ) -> PyTree[Float[Array, '']]:
    """Returns the loss for the head."""
    if self._bundle is None:
      raise ValueError('Bundle is required for loss computation.')

    tracks, mask = batch.get_genome_tracks(self._bundle)

    if mask.shape[-2] != 1:
      raise ValueError(
          'We assume the mask to broadcast over the sequence length.'
      )

    bundle_resolution = self._bundle.get_resolution()
    loss_sum, scalars = 0.0, {}

    for resolution in self._resolutions:
      predictions_for_resolution = predictions[
          f'scaled_predictions_{resolution}bp'
      ]
      if resolution == bundle_resolution:
        targets = tracks
      else:
        targets = _sum_pool(tracks, resolution)

      all_losses = self._compute_loss(
          organism_index=batch.get_organism_index(),
          predictions=predictions_for_resolution,
          targets=targets,
          targets_mask=mask,
          resolution=resolution,
      )
      for k, v in all_losses.items():
        scalars[f'{k}_{resolution}bp'] = v
      loss_sum += all_losses['loss']

    scalars['loss'] = loss_sum
    return scalars


class ContactMapsHead(Head):
  """A model head that predicts contact maps from pairwise embeddings."""

  @typing.jaxtyped
  @hk.transparent
  def _predict(
      self,
      pair_embeddings: Float[Array, 'B S S D'],
      organism_index: Int[Array, 'B'],
  ) -> Float[Array, 'B S S {self.num_tracks}']:
    """Predicts contact maps from pairwise embeddings."""
    return _MultiOrganismLinear(self.num_tracks, self._num_organisms)(
        pair_embeddings, organism_index
    )

  def predict(
      self,
      embeddings: embeddings_module.Embeddings,
      organism_index: Int[Array, 'B'],
      **kwargs,
  ) -> PyTree[Float[Array, 'B ...'] | None]:
    """Predicts contact maps from embeddings."""
    return {
        'predictions': self._predict(embeddings.embeddings_pair, organism_index)
    }

  def _get_targets_mask(
      self,
      organism_index: Int[Array, 'B'],
  ) -> Bool[Array, 'B 1 1 {self.num_tracks}']:
    """Returns a mask for padding channels."""
    track_mask = self.get_multi_organism_track_mask()[(organism_index,)]
    return track_mask[:, None, None, :]

  def loss(
      self,
      predictions: PyTree[Float[Array, 'B ...']],
      batch: schemas.DataBatch,
  ) -> PyTree[Float[Array, '']]:
    """Returns the loss for the head."""
    if (targets := batch.contact_maps) is None:
      raise ValueError('contact_maps target not in batch.')

    contact_predictions = predictions['predictions']
    chex.assert_equal_shape([contact_predictions, targets])

    # Mask out NaN targets (which happens when balancing a missing slice).
    targets_mask = self._get_targets_mask(batch.get_organism_index())
    targets_mask = jnp.where(jnp.isnan(targets), False, targets_mask)
    targets = jnp.where(jnp.isnan(targets), 0.0, targets)

    loss = losses.mse(
        y_pred=contact_predictions, y_true=targets, mask=targets_mask
    )
    return {'loss': loss}


class SpliceSitesClassificationHead(Head):
  """A model head that predicts splice site classification."""

  @typing.jaxtyped
  @hk.transparent
  def _predict_logits(
      self, x: Float[Array, 'B S D'], organism_index: Int[Array, 'B']
  ) -> Float[Array, 'B S {self.num_tracks}']:
    """Splice site classification."""
    return _MultiOrganismLinear(self.num_tracks, self._num_organisms)(
        x, organism_index
    )

  def predict(
      self,
      embeddings: embeddings_module.Embeddings,
      organism_index: Int[Array, 'B'],
      **kwargs,
  ) -> PyTree[Float[Array, 'B ...'] | None]:
    """Predicts splice site classification from embeddings."""
    embeddings_1bp = embeddings.get_sequence_embeddings(1)
    logits = self._predict_logits(embeddings_1bp, organism_index)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    return {'logits': logits, 'predictions': probs}

  def loss(
      self,
      predictions: PyTree[Float[Array, 'B ...']],
      batch: schemas.DataBatch,
  ) -> PyTree[Float[Array, '']]:
    """Returns the loss for the head."""
    if (splice_sites := batch.splice_sites) is None:
      raise ValueError('splice_sites target not in batch.')
    logits = predictions['logits']
    chex.assert_equal_shape([splice_sites, logits])

    classification_mask = jnp.any(splice_sites, axis=-1, keepdims=True)
    loss = losses.cross_entropy_loss_from_logits(
        y_pred_logits=logits,
        # Label smoothing with FP32 machine precision (~1e-7) for 5 classes.
        y_true=(1.0 - 1e-7) * splice_sites.astype(jnp.float32)
        + 1e-7 / self.num_tracks,
        mask=classification_mask,
        axis=-1,
    )
    return {'loss': loss}


class SpliceSitesUsageHead(Head):
  """A model head that predicts splice site usage."""

  @typing.jaxtyped
  @hk.transparent
  def _predict_logits(
      self, x: Float[Array, 'B S D'], organism_index: Int[Array, 'B']
  ) -> Float[Array, 'B S {self.num_tracks}']:
    """Splice site usage."""
    return _MultiOrganismLinear(self.num_tracks, self._num_organisms)(
        x, organism_index
    )

  def predict(
      self,
      embeddings: embeddings_module.Embeddings,
      organism_index: Int[Array, 'B'],
      **kwargs,
  ) -> PyTree[Float[Array, 'B ...'] | None]:
    """Predicts splice site usage from embeddings."""
    embeddings_1bp = embeddings.get_sequence_embeddings(1)
    logits = self._predict_logits(embeddings_1bp, organism_index)
    splice_site_usage = jax.nn.sigmoid(logits.astype(jnp.float32)).astype(
        jnp.float16
    )
    return {'logits': logits, 'predictions': splice_site_usage}

  @typing.jaxtyped
  def _get_targets_mask(
      self,
      organism_index: Int[Array, 'B'],
  ) -> Bool[Array, 'B 1 {self.num_tracks}']:
    """Returns a mask for padding channels."""
    track_mask = self.get_multi_organism_track_mask()[(organism_index,)]
    return track_mask[:, None, :]

  def loss(
      self,
      predictions: PyTree[Float[Array, 'B ...']],
      batch: schemas.DataBatch,
  ) -> PyTree[Float[Array, '']]:
    """Returns the loss for the head."""
    if (splice_site_usage := batch.splice_site_usage) is None:
      raise ValueError('splice_site_usage target not in batch.')
    logits = predictions['logits']
    chex.assert_equal_shape([splice_site_usage, logits])
    loss = losses.binary_crossentropy_from_logits(
        y_pred=logits,
        y_true=jnp.clip(splice_site_usage, 1e-7, 1.0 - 1e-7),
        mask=self._get_targets_mask(batch.get_organism_index()),
    )
    return {'loss': loss}


class SpliceSitesJunctionHead(Head):
  """A model head that predicts splice site junctions."""

  def __init__(
      self,
      *,
      name: str,
      output_type: dna_output.OutputType,
      metadata: Mapping[
          dna_model.Organism,
          metadata_lib.AlphaGenomeOutputMetadata,
      ],
      num_organisms: int | None = None,
  ):
    """Initializes the SpliceSitesJunctionHead module."""
    super().__init__(
        name=name,
        output_type=output_type,
        metadata=metadata,
        num_organisms=num_organisms,
    )
    self._hidden_dim = 768
    self._max_position_encoding_distance = int(2**20)

  def get_num_tissues(self, organism: dna_model.Organism) -> int:
    """Returns the number of tissues for the given organism."""
    metadata = self.get_metadata(organism)
    if metadata is None:
      raise ValueError(f'Metadata not found for organism {organism}.')
    return len(metadata)

  @property
  def max_num_tissues(self) -> int:
    """Returns the maximum number of tissues across all organisms from the metadata."""

    return max(self.get_num_tissues(organism) for organism in self._metadata)

  def _get_num_tracks(self) -> int:
    """Returns the number of tracks.

    Splice junctions metadata contains the tissues per organism, rather than the
    tracks. The number of tracks is twice the number of tissues accounting for
    the two strands.
    """
    return 2 * self.max_num_tissues

  def get_multi_organism_track_mask(
      self,
  ) -> Bool[Array, '{self._num_organisms} {self.num_tracks}']:
    """Returns the track mask for the head for human and mouse.

    Splice junctions metadata contains the tissues per organism, rather than the
    tracks.
    """
    track_masks = []
    for organism in self._metadata:
      num_tissues = self.get_num_tissues(organism)
      tissue_mask = np.arange(self.max_num_tissues) < num_tissues
      # Repeat the mask for the two strands.
      track_mask = np.concatenate([tissue_mask, tissue_mask])
      track_masks.append(track_mask)
    return jnp.stack(track_masks).astype(bool)

  @typing.jaxtyped
  @hk.transparent
  def _predict(
      self,
      x: Float[Array, 'B S D'],
      splice_site_positions: Int[Array, 'B 4 P'],
      organism_index: Int[Array, 'B'],
  ) -> tuple[
      Float[Array, 'B P P {self.num_tracks}'],
      Bool[Array, 'B P P {self.num_tracks}'],
  ]:
    """Splice site junctions."""

    chex.assert_shape(splice_site_positions, (None, 4, None))
    pos_donor_idx = splice_site_positions[:, 0, :]
    pos_accept_idx = splice_site_positions[:, 1, :]
    neg_donor_idx = splice_site_positions[:, 2, :]
    neg_accept_idx = splice_site_positions[:, 3, :]

    def _index_embedding(embedding, indices):
      return jax.vmap(functools.partial(jnp.take, axis=0))(embedding, indices)

    shape = (self._num_organisms, 2, self.max_num_tissues, self._hidden_dim)

    def _apply_rope(x, indices):
      x = _index_embedding(x, indices).astype(jnp.float32)
      params = hk.get_parameter(
          'embeddings',
          (shape[0], math.prod(shape[1:])),
          dtype=x.dtype,
          init=jnp.zeros,
      ).reshape(*shape)
      params = _get_param_for_index(params, organism_index)
      # scale and offset have shape [B, 1, num_tissues, C].
      scale, offset = params[:, [0], :, :], params[:, [1], :, :]
      x = scale * x[:, :, None, :] + offset  # [B, num_indices, num_tissues, C]
      return attention.apply_rope(
          x, indices, self._max_position_encoding_distance
      )

    splice_site_logits = _MultiOrganismLinear(
        self._hidden_dim, self._num_organisms
    )(x, organism_index)

    with hk.name_scope('pos_acceptor_logits'):
      pos_accept_logits = _apply_rope(splice_site_logits, pos_accept_idx)
    with hk.name_scope('pos_donor_logits'):
      pos_donor_logits = _apply_rope(splice_site_logits, pos_donor_idx)
    with hk.name_scope('neg_acceptor_logits'):
      neg_accept_logits = _apply_rope(splice_site_logits, neg_accept_idx)
    with hk.name_scope('neg_donor_logits'):
      neg_donor_logits = _apply_rope(splice_site_logits, neg_donor_idx)
    pos_counts = jax.nn.softplus(
        jnp.einsum(
            'bdtc,batc->bdat',
            pos_donor_logits,
            pos_accept_logits,
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        )
    )
    neg_counts = jax.nn.softplus(
        jnp.einsum(
            'bdtc,batc->bdat',
            neg_donor_logits,
            neg_accept_logits,
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        )
    )
    pos_mask = jnp.einsum('bd,ba->bda', pos_donor_idx >= 0, pos_accept_idx >= 0)
    neg_mask = jnp.einsum('bd,ba->bda', neg_donor_idx >= 0, neg_accept_idx >= 0)
    track_mask = self.get_multi_organism_track_mask()[(
        organism_index,
    )]  # [B, 2 * num_tissues]
    pos_mask = (
        pos_mask[:, :, :, None]
        * track_mask[:, None, None, : self.max_num_tissues]
    )
    neg_mask = (
        neg_mask[:, :, :, None]
        * track_mask[:, None, None, self.max_num_tissues :]
    )
    # Shape [B, D, A, 2 * num_tissues]
    splice_junction_mask = jnp.concatenate([pos_mask, neg_mask], axis=-1)
    pred_counts = jnp.concatenate((pos_counts, neg_counts), axis=-1)
    pred_counts = jnp.where(splice_junction_mask, pred_counts, 0)
    return pred_counts, splice_junction_mask

  def predict(
      self,
      embeddings: embeddings_module.Embeddings,
      organism_index: Int[Array, 'B'],
      **kwargs,
  ) -> PyTree[Shaped[Array, 'B ...'] | None]:
    """Predicts splice site junctions from embeddings."""
    if (splice_site_positions := kwargs.get('splice_site_positions')) is None:
      raise ValueError(
          'splice_site_positions is required for junctions predictions.'
      )
    embeddings_1bp = embeddings.get_sequence_embeddings(1)
    splice_site_junction, splice_junction_mask = self._predict(
        embeddings_1bp, splice_site_positions, organism_index
    )
    return {
        'predictions': splice_site_junction,
        'splice_site_positions': splice_site_positions,
        'splice_junction_mask': splice_junction_mask,
    }

  def loss(
      self,
      predictions: PyTree[Shaped[Array, 'B ...']],
      batch: schemas.DataBatch,
  ) -> PyTree[Float[Array, '']]:
    """Returns the loss for the head."""
    if (count_target := batch.splice_junctions) is None:
      raise ValueError('splice_junctions target not in batch.')

    pred_pair = predictions['predictions']
    pairs_mask = predictions['splice_junction_mask']
    # Junctions shape is [batch, donors, acceptors, 2 * num_tissues].
    chex.assert_equal_shape([pred_pair, count_target, pairs_mask])
    chex.assert_rank(pred_pair, 4)

    def _scale_junction_counts(counts):
      return jnp.where(
          counts > _SOFT_CLIP_VALUE,
          2.0 * jnp.sqrt(counts * _SOFT_CLIP_VALUE) - _SOFT_CLIP_VALUE,
          counts,
      )

    accept_total_loss = losses.poisson_loss(
        y_true=_scale_junction_counts(
            count_target.sum(axis=-2, dtype=jnp.float32, where=pairs_mask)
        ),
        y_pred=pred_pair.sum(axis=-2, dtype=jnp.float32, where=pairs_mask),
        mask=jnp.any(pairs_mask, axis=-2),
    )
    donor_total_loss = losses.poisson_loss(
        y_true=_scale_junction_counts(
            count_target.sum(axis=-3, dtype=jnp.float32, where=pairs_mask)
        ),
        y_pred=pred_pair.sum(axis=-3, dtype=jnp.float32, where=pairs_mask),
        mask=jnp.any(pairs_mask, axis=-3),
    )

    # Ratios with cross entropy loss.
    donor_ratios_loss = losses.cross_entropy_loss(
        y_true=count_target,
        y_pred=pred_pair,
        mask=pairs_mask,
        axis=-3,
    )
    acceptor_ratios_loss = losses.cross_entropy_loss(
        y_true=count_target,
        y_pred=pred_pair,
        mask=pairs_mask,
        axis=-2,
    )
    loss = (
        donor_ratios_loss
        + acceptor_ratios_loss
        + 0.2 * (accept_total_loss + donor_total_loss)
    )
    return {'loss': loss}
