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

"""Utilities for augmenting predictions, e.g. reverse complementation."""

from collections.abc import Mapping
from typing import TypeAlias

from alphagenome import typing
from alphagenome.models.v1 import dna_output
import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int32, PyTree  # pylint: disable=g-importing-member, g-multiple-import

_Predictions: TypeAlias = PyTree[Float[Array, 'B ...'] | Int32[Array, 'B ...']]


@typing.jaxtyped
def reverse_complement_output_type(
    predictions: _Predictions,
    *,
    output_type: dna_output.OutputType,
    strand_reindexing: Int32[Array, '_'] | None,
    sequence_length: Int32[Array, ''] | int,
) -> _Predictions:
  """Reverse complement predictions for a given output type."""
  match output_type:  # pytype: disable=incomplete-match
    case dna_output.OutputType.SPLICE_JUNCTIONS:
      splice_junction_predictions = predictions['predictions']
      chex.assert_rank(splice_junction_predictions, 4)
      # Only need to strand flip for junction outputs. Index flipping is done
      # with splice_site_positions.
      chex.assert_equal_shape(
          (splice_junction_predictions, strand_reindexing), dims=-1
      )
      predictions['predictions'] = splice_junction_predictions[
          ..., strand_reindexing
      ]

      splice_site_positions = predictions['splice_site_positions']
      chex.assert_rank(splice_site_positions, 3)
      padding_predictions = splice_site_positions < 0
      splice_site_positions = sequence_length - 1 - splice_site_positions
      splice_site_positions = jnp.where(
          padding_predictions, -1, splice_site_positions
      )
      splice_site_positions = splice_site_positions[:, jnp.array([2, 3, 0, 1])]

      predictions['splice_site_positions'] = splice_site_positions
    case dna_output.OutputType.CONTACT_MAPS:
      chex.assert_rank(predictions, 4)
      # Contact maps are unstranded, so no need to strand flip.
      return predictions[:, ::-1, ::-1]
    case _:
      chex.assert_rank(predictions, 3)
      predictions = predictions[:, ::-1]
      if strand_reindexing is not None:
        predictions = predictions[..., strand_reindexing]
      # For DNASE, +ve strand predictions are not aligned with -ve strand
      # predictions after reverse-complementation:
      # Input positions:
      #   [ 0 ][ 1 ][ 2 ]
      # Track values:
      # [ A ][ B ][ C ][ D ]
      # In this example, output predictions are determined by the track value
      # at the start of the position. Thus, when read forwards, the + strand
      # will give ABC and the - strand will give DCB. If we want predictions to
      # align after reverse-complementation (for example, to perform TTA or to
      # return aligned, unstranded predictions for both + and -), we need to
      # shift predictions by 1 after rc.
      if output_type == dna_output.OutputType.DNASE:
        predictions = jnp.roll(predictions, 1, axis=1)
        predictions = predictions.at[:, 0].set(0)

  return predictions


@typing.jaxtyped
def reverse_complement(
    predictions: Mapping[dna_output.OutputType, _Predictions],
    reverse_complement_mask: Bool[Array, 'B'],
    *,
    strand_reindexing: Mapping[dna_output.OutputType, Int32[Array, '_']],
    sequence_length: int,
) -> Mapping[dna_output.OutputType, _Predictions]:
  """Reverse complement predictions."""

  def _reverse_complement(predictions: _Predictions):
    result = {}
    for output_type in dna_output.OutputType:
      if (prediction := predictions.get(output_type)) is not None:
        result[output_type] = reverse_complement_output_type(
            prediction,
            output_type=output_type,
            strand_reindexing=strand_reindexing.get(output_type),
            sequence_length=sequence_length,
        )
    return result

  # Add single batch dimension to vmap over.
  predictions = jax.tree.map(lambda x: jnp.expand_dims(x, axis=1), predictions)
  predictions = jax.vmap(
      lambda prediction, should_reverse: jax.lax.cond(
          should_reverse,
          _reverse_complement,
          lambda prediction: prediction,
          prediction,
      )
  )(predictions, reverse_complement_mask)
  return jax.tree.map(lambda x: jnp.squeeze(x, axis=1), predictions)
