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
"""AlphaGenome finetuning script."""

from collections.abc import Iterator, Mapping
from typing import Any, Callable
from alphagenome.data import fold_intervals
from alphagenome.models import dna_model
from alphagenome_research.finetuning import dataset as dataset_lib
from alphagenome_research.model import model
from alphagenome_research.model import schemas
from alphagenome_research.model.metadata import metadata as metadata_lib
import haiku as hk
import jax
import jmp
import optax
import tensorflow as tf

_FASTA_PATH = (
    'https://storage.googleapis.com/alphagenome/reference/gencode/'
    'hg38/GRCh38.p13.genome.fa'
)


def get_dataset_iterator(
    *,
    batch_size: int,
    sequence_length: int,
    output_metadata: metadata_lib.AlphaGenomeOutputMetadata,
    model_version: dna_model.ModelVersion,
    subset: fold_intervals.Subset,
    organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
    fasta_path: str = _FASTA_PATH,
    example_regions_path: str | None = None,
) -> Iterator[schemas.DataBatch]:
  """Converts pipeline output dict to a DataBatch schema.

  Args:
    batch_size: The batch size of the dataset.
    sequence_length: The sequence length of the dataset.
    output_metadata: Metadata for the output tracks for each organism.
    model_version: The model version to use.
    subset: The subset of the dataset.
    organism: The organism to use.
    fasta_path: The path to the reference genome FASTA file.
    example_regions_path: The path to the example regions BED file.

  Returns:
    A schemas.DataBatch object.
  """
  intervals = fold_intervals.get_fold_intervals(
      model_version,
      organism,
      subset,
      example_regions_path=example_regions_path,
  )
  pipeline = dataset_lib.DataPipeline(
      fasta_path=fasta_path,
      intervals=intervals,
      output_metadata=output_metadata,
      organism=organism,
      sequence_length=sequence_length,
  )
  dataset = tf.data.Dataset.from_generator(
      pipeline.get_generator,
      output_signature=pipeline.get_output_signature(),
  )
  dataset = (
      dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
  )

  def iterator():
    for batch in dataset:
      yield schemas.DataBatch(
          dna_sequence=batch['dna_sequence'],
          organism_index=batch['organism_index'],
          **batch['bundles'],
      )

  return iterator()


def get_forward_fn(
    output_metadata: Mapping[
        dna_model.Organism, metadata_lib.AlphaGenomeOutputMetadata
    ],
    jmp_policy: str = 'params=float32,compute=bfloat16,output=bfloat16',
) -> hk.TransformedWithState:
  """Creates a Haiku transformed function for the AlphaGenome model.

  Args:
    output_metadata: Metadata for the output tracks for each organism.
    jmp_policy: The JMP policy to use for mixed precision.

  Returns:
    A `hk.TransformedWithState` object representing the forward pass.
  """
  jmp_policy = jmp.get_policy(jmp_policy)

  @hk.transform_with_state
  def forward(batch: schemas.DataBatch):
    with hk.mixed_precision.push_policy(model.AlphaGenome, jmp_policy):
      return model.AlphaGenome(
          output_metadata, freeze_trunk_embeddings=True
      ).loss(batch)

  return forward


def get_train_step(
    predict_fn: Callable[..., Any],
    optimizer: optax.GradientTransformation,
):
  """Creates a jitted training step function using AlphaGenome.loss.

  Args:
    predict_fn: The Haiku transformed forward function.
    optimizer: An Optax optimizer to apply gradients.

  Returns:
    A jitted function `train_step` that takes `params`, `state`, `opt_state`,
    and a `batch` as input and returns the updated `params`, `next_state`,
    `new_opt_state`, and a dictionary of `metrics`.
  """

  @jax.jit
  def train_step(params, state, opt_state, batch):
    def loss_fn(params, state, batch):
      (loss, scalars, predictions), new_state = predict_fn(
          params, state, None, batch
      )
      del predictions  # Unused.
      return loss, (new_state, scalars)

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (next_state, scalars)), grads = loss_grad_fn(params, state, batch)
    scalars['loss'] = loss
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, next_state, new_opt_state, scalars

  return train_step
