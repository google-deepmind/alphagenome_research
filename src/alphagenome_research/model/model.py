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

"""AlphaGenome model."""

from collections.abc import Mapping

from alphagenome import typing
from alphagenome.models import dna_model
from alphagenome_research.model import attention
from alphagenome_research.model import convolutions
from alphagenome_research.model import embeddings as embeddings_module
from alphagenome_research.model import heads as heads_module
from alphagenome_research.model import layers
from alphagenome_research.model import schemas
from alphagenome_research.model import splicing
from alphagenome_research.model.metadata import metadata as metadata_lib
import haiku as hk
import jax
from jaxtyping import Array, Float, Int, PyTree  # pylint: disable=g-importing-member, g-multiple-import


class SequenceEncoder(hk.Module):
  """Encodes a sequence of DNA into embeddings."""

  @typing.jaxtyped
  def __call__(
      self, dna_sequence: Float[Array, 'B S 4']
  ) -> tuple[Float[Array, 'B S//128 D'], dict[str, Array]]:
    intermediates = {}
    x = convolutions.DnaEmbedder()(dna_sequence)
    intermediates['bin_size_1'] = x
    x = layers.pool(x)
    for block_idx, bin_size in enumerate([2, 4, 8, 16, 32, 64]):
      x = convolutions.DownResBlock(f'downres_block_{block_idx}')(x)
      intermediates[f'bin_size_{bin_size}'] = x
      x = layers.pool(x)
    return x, intermediates


class SequenceDecoder(hk.Module):
  """Decodes a sequence of embeddings."""

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S D'], intermediates: dict[str, Array]
  ) -> Float[Array, 'B S_final D_final']:
    for bin_size in [64, 32, 16, 8, 4, 2, 1]:
      x = convolutions.UpResBlock()(x, intermediates[f'bin_size_{bin_size}'])
    return x


class TransformerTower(hk.Module):
  """Transformer tower with interleaved pairwise updates."""

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S C']
  ) -> tuple[Float[Array, 'B S C'], Float[Array, 'B S//16 S//16 F'] | None]:
    pair_x = None
    for i in range(9):
      if i % 2 == 0:
        pair_x = attention.PairUpdateBlock()(x, pair_x)
      mha_bias = attention.AttentionBiasBlock()(pair_x)
      x += attention.MHABlock()(x, mha_bias)
      x += attention.MLPBlock()(x)
    return x, pair_x


class AlphaGenome(hk.Module):
  """Main AlphaGenome model.

  The model architecture consists of a sequence encoder, a transformer tower,
  and a sequence decoder. The output of the decoder is used to generate
  embeddings at 1bp resolution, while the output of the transformer tower
  is used to generate embeddings at 128bp resolution and pair embeddings.
  These embeddings are then passed to various heads to make predictions.
  """

  def __init__(
      self,
      output_metadata: Mapping[
          dna_model.Organism, metadata_lib.AlphaGenomeOutputMetadata
      ],
      *,
      num_splice_sites: int = 512,
      splice_site_threshold: float = 0.1,
      freeze_trunk_embeddings: bool = False,
      name: str | None = None,
  ):
    """Initializes the AlphaGenome model.

    Args:
      output_metadata: Metadata for the output tracks for each organism.
      num_splice_sites: The maximum number of splice sites that are extracted
        from the splice site classification predictions.
      splice_site_threshold: The threshold to use for splice site prediction.
      freeze_trunk_embeddings: Whether to stop the gradient to the embeddings.
        This is useful for training only the heads in fine-tuning.
      name: The name of the module.
    """
    super().__init__(name=name or 'alphagenome')
    self._output_metadata = output_metadata
    self._num_splice_sites = num_splice_sites
    self._splice_site_threshold = splice_site_threshold
    self._freeze_trunk_embeddings = freeze_trunk_embeddings
    self._heads: dict[heads_module.HeadName, heads_module.Head] = {}
    for head in heads_module.HeadName:
      output_type = heads_module.get_head_config(head).output_type
      organisms_with_metadata = [
          organism
          for organism, metadata in output_metadata.items()
          if metadata.get(output_type) is not None
      ]
      if not organisms_with_metadata:
        # None of the organisms have metadata for this output type. Skip.
        continue
      missing_organisms = set(self._output_metadata.keys()) - set(
          organisms_with_metadata
      )
      if missing_organisms:
        raise ValueError(
            f'No metadata found for output type "{output_type.name}" for the'
            f' following organisms: {missing_organisms}. We expect the same set'
            ' of output types for all organisms. Use padding to account for'
            ' missing tracks.'
        )
      self._heads[head] = heads_module.create_head(
          heads_module.get_head_config(head), self._output_metadata
      )

  @property
  def _num_organisms(self) -> int:
    return len(self._output_metadata)

  @hk.name_like('__call__')
  def predict_junctions(
      self,
      trunk_embeddings: Float[Array, 'B S D'],
      splice_site_positions: Int[Array, 'B 4 K'],
      organism_index: Int[Array, 'B'],
  ) -> PyTree[Float[Array, 'B ...'] | None]:
    """Predicts splice site junctions from embeddings and splice site positions.

    Args:
      trunk_embeddings: The trunk embeddings to use for predictions.
      splice_site_positions: The splice site positions. Format: [batch, 4,
        num_splice_sites] with order: [donor_pos_idx, accept_pos_idx,
        donor_neg_idx, accept_neg_idx]
      organism_index: The organism index.

    Returns:
      The predictions for splice site junctions.
    """
    junction_head = self._heads.get(heads_module.HeadName.SPLICE_SITES_JUNCTION)
    if junction_head is None:
      raise ValueError('Junction head is not supported by this model.')
    with hk.name_scope('head'):
      return junction_head(
          embeddings_module.Embeddings(embeddings_1bp=trunk_embeddings),
          organism_index,
          splice_site_positions=splice_site_positions,
      )

  @typing.jaxtyped
  def __call__(
      self,
      dna_sequence: Float[Array, 'B S 4'],
      organism_index: Int[Array, 'B'],
  ) -> tuple[
      PyTree[Float[Array, 'B ...'] | None], embeddings_module.Embeddings
  ]:
    """Encodes a sequence of DNA and makes predictions for various heads.

    Args:
      dna_sequence: The sequence of DNA to encode.
      organism_index: The organism index.

    Returns:
      A tuple of (predictions, embeddings), where predictions is a dictionary
      of predictions for various heads.
    """
    trunk, intermediates = SequenceEncoder()(dna_sequence)
    if self._num_organisms >= 1:
      organism_embedding_trunk = hk.Embed(self._num_organisms, trunk.shape[-1])(
          organism_index
      )
      trunk += organism_embedding_trunk[:, None, :]
    trunk, pair_activations = TransformerTower()(trunk)

    x = SequenceDecoder()(trunk, intermediates)

    embeddings_128bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        trunk, organism_index
    )
    embeddings_1bp = embeddings_module.OutputEmbedder(self._num_organisms)(
        x, organism_index, embeddings_128bp
    )
    embeddings_pair = embeddings_module.OutputPair(self._num_organisms)(
        pair_activations, organism_index
    )

    embeddings = embeddings_module.Embeddings(
        embeddings_1bp=embeddings_1bp,
        embeddings_128bp=embeddings_128bp,
        embeddings_pair=embeddings_pair,
    )
    if self._freeze_trunk_embeddings:
      embeddings = jax.lax.stop_gradient(embeddings)
    predictions = {
        'embeddings_1bp': embeddings_1bp,
    }
    with hk.name_scope('head'):
      for head_name, head_fn in self._heads.items():
        if head_name == heads_module.HeadName.SPLICE_SITES_JUNCTION:
          # This head is handled separately (see below).
          continue
        predictions[head_name.value] = head_fn(
            embeddings,
            organism_index,
        )

    # Handle the splice junction head separately. It requires splice site
    # positions as input, which are derived from the splice site
    # classification predictions.
    if (
        junction_head := heads_module.HeadName.SPLICE_SITES_JUNCTION
    ) in self._heads:
      if heads_module.HeadName.SPLICE_SITES_CLASSIFICATION not in self._heads:
        raise ValueError(
            'SPLICE_SITES_CLASSIFICATION head is required for junctions'
            ' predictions.'
        )
      splice_sites_probabilities = predictions[
          heads_module.HeadName.SPLICE_SITES_CLASSIFICATION.value
      ]['predictions']
      splice_site_positions = splicing.generate_splice_site_positions(
          splice_sites_probabilities,
          alt=None,
          splice_sites=None,
          k=self._num_splice_sites,
          pad_to_length=self._num_splice_sites,
          threshold=self._splice_site_threshold,
      )
      predictions[junction_head.value] = self.predict_junctions(
          embeddings.embeddings_1bp, splice_site_positions, organism_index
      )
    return predictions, embeddings

  @typing.jaxtyped
  def loss(self, batch: schemas.DataBatch) -> tuple[
      Float[Array, ''],
      PyTree[Float[Array, '']],
      PyTree[Float[Array, 'B ...'] | None],
  ]:
    """Returns the loss for the model."""
    predictions, _ = self(batch.dna_sequence, batch.get_organism_index())
    total_loss, all_scalars = 0.0, {}
    for head_name, head_fn in self._heads.items():
      scalars = head_fn.loss(predictions[head_name.value], batch)
      all_scalars.update(
          {f'{head_name.value}_{k}': v for k, v in scalars.items()}
      )
      total_loss += scalars['loss']
    return total_loss, all_scalars, predictions
