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

"""Evaluates AlphaGenome track prediction performance."""

from collections.abc import Iterator
import pprint
from typing import Callable, Sequence
from absl import app
from absl import logging
from alphagenome.data import fold_intervals
from alphagenome.models import dna_output
from alphagenome_research.evals import regression_metrics
from alphagenome_research.io import bundles as bundles_lib
from alphagenome_research.io import dataset
from alphagenome_research.model import dna_model
from alphagenome_research.model import model as model_lib
from alphagenome_research.model import schemas
from alphagenome_research.model.metadata import metadata as metadata_lib
import haiku as hk
import jax
from jax import sharding
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jaxtyping import PyTree  # pylint: disable=g-importing-member
import jmp
import kagglehub
import orbax.checkpoint as ocp
import pandas as pd
import tensorflow as tf

PS = sharding.PartitionSpec
PredictFn = Callable[
    [
        hk.Params,
        hk.State,
        jax.Array,
        jax.Array,
    ],
    PyTree[jax.Array],
]

_SUBSET = fold_intervals.Subset.VALID
_LOG_FREQUENCY = 5
_EVAL_BUNDLES = [
    bundles_lib.BundleName.ATAC,
    bundles_lib.BundleName.CAGE,
    bundles_lib.BundleName.CHIP_HISTONE,
    bundles_lib.BundleName.CHIP_TF,
    bundles_lib.BundleName.DNASE,
    bundles_lib.BundleName.PROCAP,
    bundles_lib.BundleName.RNA_SEQ,
]


def load_model(
    model_version: dna_model.ModelVersion = dna_model.ModelVersion.FOLD_0,
) -> tuple[hk.Params, hk.State, PredictFn]:
  """Loads the model consiting of params, state and predict function."""
  checkpoint_path = kagglehub.model_download(
      f'google/alphagenome/jax/{model_version.name.lower()}'
  )
  params, state = ocp.StandardCheckpointer().restore(checkpoint_path)
  metadata = {
      organism: metadata_lib.load(organism) for organism in dna_model.Organism
  }

  @hk.transform_with_state
  def forward(dna_sequence, organism_index):
    policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
    with hk.mixed_precision.push_policy(model_lib.AlphaGenome, policy):
      return model_lib.AlphaGenome(metadata)(dna_sequence, organism_index)

  @jax.jit(
      in_shardings=(PS(), PS(), PS('data'), PS('data')),
      out_shardings=PS('data'),
  )
  def predict(params, state, dna_sequence, organism_index) -> PyTree[jax.Array]:
    (predictions, _), _ = forward.apply(
        params, state, None, dna_sequence, organism_index
    )
    requested_outputs = tuple(
        dna_output.OutputType[bundle.name] for bundle in _EVAL_BUNDLES
    )
    predictions = dna_model.extract_predictions(predictions, requested_outputs)
    return predictions

  return params, state, predict


def create_eval_step(
    predict_fn: PredictFn, bundles: Sequence[bundles_lib.BundleName]
):
  """Returns the eval step function."""

  @jax.jit(
      in_shardings=(PS(), PS(), PS('data')),
      out_shardings=PS(),
  )
  def eval_step(params, state, batch: schemas.DataBatch):
    predictions = predict_fn(
        params,
        state,
        jnp.asarray(batch.dna_sequence, copy=False),
        jnp.asarray(batch.organism_index, copy=False),
    )
    metrics_step: dict[str, regression_metrics.RegressionState] = {}
    for bundle in bundles:
      targets_true, mask = batch.get_genome_tracks(bundle)
      targets_pred = predictions[dna_output.OutputType[bundle.name]]
      targets_pred = regression_metrics.crop_sequence_length(
          targets_pred, target_length=targets_true.shape[-2]
      )
      metrics_step[bundle.name] = regression_metrics.update_regression_metrics(
          jnp.asarray(targets_true),
          jnp.asarray(targets_pred),
          jnp.asarray(mask),
      )
    return metrics_step

  return eval_step


def evaluate(
    params: hk.Params,
    state: hk.State,
    predict_fn: PredictFn,
    bundles: Sequence[bundles_lib.BundleName],
    dataset_iterator: Iterator[tuple[schemas.DataBatch, dataset.BatchMetadata]],
):
  """Evaluates the model."""
  # Setup Mesh.
  devices = mesh_utils.create_device_mesh((jax.local_device_count(),))
  mesh = jax.sharding.Mesh(devices, axis_names=('data',))
  sharding_rep = sharding.NamedSharding(mesh, PS())
  sharding_data = sharding.NamedSharding(mesh, PS('data'))

  # Replicate params and state.
  params = jax.device_put(params, sharding_rep)
  state = jax.device_put(state, sharding_rep)

  eval_step = create_eval_step(predict_fn, bundles)
  metrics = {
      b.name: regression_metrics.initialize_regression_metrics()
      for b in bundles
  }
  num_elements = 0

  for i, (batch, _) in enumerate(dataset_iterator):
    assert batch.dna_sequence is not None
    num_elements += batch.dna_sequence.shape[0]
    if i % _LOG_FREQUENCY == 1:
      m = pprint.pformat(
          regression_metrics.finalize_regression_metrics(metrics)
      )
      logging.info('step %d: %s', i, m)

    with jax.set_mesh(mesh):
      batch = jax.device_put(batch, sharding_data)
      step_metrics = eval_step(params, state, batch)

    # Accumulate metrics.
    step_metrics = jax.device_get(step_metrics)
    metrics = regression_metrics.reduce_regression_metrics(
        metrics, step_metrics
    )
  logging.info('num_elements: %d', num_elements)
  return regression_metrics.finalize_regression_metrics(metrics)


def run(
    organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
    model_version: dna_model.ModelVersion = dna_model.ModelVersion.FOLD_0,
) -> pd.DataFrame:
  """Runs the track prediction experiment."""
  logging.info('Starting track prediction experiment.')
  params, state, predict_fn = load_model(model_version)
  dataset_iterator = dataset.get_numpy_dataset_iterator(
      batch_size=jax.local_device_count(),
      organism=organism,
      model_version=model_version,
      bundles=_EVAL_BUNDLES,
      subset=_SUBSET,
  )
  results = evaluate(params, state, predict_fn, _EVAL_BUNDLES, dataset_iterator)
  flattened_results = {}
  for bundle, result in results.items():
    for metric, value in result.items():
      logging.info('bundle: %s, metric: %s, value: %s', bundle, metric, value)
      flattened_results[f'{bundle}_{metric}'] = value
  df = pd.DataFrame(
      {'metric': flattened_results.keys(), 'value': flattened_results.values()}
  )
  return df


if __name__ == '__main__':
  # Hide local GPUs from TF. TF is only used for data loading.
  tf.config.set_visible_devices([], 'GPU')
  app.run(lambda _: run())
