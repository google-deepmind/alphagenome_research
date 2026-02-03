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

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import junction_data
from alphagenome.data import track_data
from alphagenome.models import dna_model
from alphagenome_research.model import embeddings as embeddings_lib
from alphagenome_research.model import model as model_lib
from alphagenome_research.model import schemas
from alphagenome_research.model.metadata import metadata as metadata_lib
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int  # pylint: disable=g-importing-member, g-multiple-import
import numpy as np
import pandas as pd


def _mock_track_metadata(
    num_tracks: int, num_padding: int = 0
) -> track_data.TrackMetadata:
  return pd.DataFrame({
      'name': (
          [f'track_{i}' for i in range(num_tracks)] + ['padding'] * num_padding
      ),
      'nonzero_mean': [1.0] * num_tracks + [0.0] * num_padding,
  })


def _mock_junction_metadata(
    num_tissues: int, num_padding: int = 0
) -> junction_data.JunctionMetadata:
  return pd.DataFrame({
      'tissue': (
          [f'tissue_{i}' for i in range(num_tissues)]
          + ['padding'] * num_padding
      ),
      'name': (
          [f'tissue_{i}' for i in range(num_tissues)]
          + ['padding'] * num_padding
      ),
  })


class ModelTest(parameterized.TestCase):

  def test_create_model_metadata_consistency(self):
    """Tests that the model raises an error if the metadata is inconsistent."""
    output_metadata_missing = {
        dna_model.Organism.HOMO_SAPIENS: metadata_lib.AlphaGenomeOutputMetadata(
            atac=_mock_track_metadata(num_tracks=13)
        ),
        dna_model.Organism.MUS_MUSCULUS: (
            metadata_lib.AlphaGenomeOutputMetadata()
        ),
    }

    @hk.transform_with_state
    def forward(dna_sequence, organism_index):
      return model_lib.AlphaGenome(output_metadata_missing)(
          dna_sequence, organism_index
      )

    with self.assertRaisesRegex(
        ValueError,
        r'No metadata found for output type "ATAC" for the following organisms:'
        r'.*MUS_MUSCULUS.*',
    ):
      jax.eval_shape(
          forward.init,
          jax.random.key(0),
          jax.ShapeDtypeStruct((1, 2048, 4), jnp.float32),
          jax.ShapeDtypeStruct((1,), jnp.int32),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_organisms',
          organisms=(),
      ),
      dict(
          testcase_name='one_organism',
          organisms=(dna_model.Organism.HOMO_SAPIENS,),
      ),
      dict(
          testcase_name='two_organisms',
          organisms=(
              dna_model.Organism.HOMO_SAPIENS,
              dna_model.Organism.MUS_MUSCULUS,
          ),
      ),
  )
  def test_create_headless_model(
      self, organisms: tuple[dna_model.Organism, ...]
  ):
    """Tests that the model can be created without any heads."""
    output_metadata = {
        organism: metadata_lib.AlphaGenomeOutputMetadata()
        for organism in organisms
    }

    @hk.transform_with_state
    def forward(dna_sequence, organism_index):
      return model_lib.AlphaGenome(output_metadata)(
          dna_sequence, organism_index
      )

    params_shape, state_shape = jax.eval_shape(
        forward.init,
        jax.random.key(0),
        seq_input := jax.ShapeDtypeStruct((1, 2048, 4), jnp.float32),
        organism_index := jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    (predictions_shape, embeddings_shape), _ = jax.eval_shape(
        forward.apply,
        params_shape,
        state_shape,
        jax.random.key(0),
        seq_input,
        organism_index,
    )
    # Predictions are only the embeddings in this case.
    chex.assert_trees_all_equal_shapes(
        predictions_shape,
        {'embeddings_1bp': jnp.zeros((1, 2048, 1536))},
    )
    chex.assert_trees_all_equal_shapes(
        embeddings_shape,
        embeddings_lib.Embeddings(
            embeddings_1bp=jnp.zeros((1, 2048, 1536)),
            embeddings_128bp=jnp.zeros((1, 16, 3072), jnp.float32),
            embeddings_pair=jnp.zeros((1, 1, 1, 128), jnp.float32),
        ),
    )

  def test_create_model(self):
    num_tracks = 13
    output_metadata = {
        dna_model.Organism.HOMO_SAPIENS: metadata_lib.AlphaGenomeOutputMetadata(
            atac=_mock_track_metadata(num_tracks=num_tracks)
        ),
        dna_model.Organism.MUS_MUSCULUS: metadata_lib.AlphaGenomeOutputMetadata(
            atac=_mock_track_metadata(num_tracks=7, num_padding=num_tracks - 7)
        ),
    }

    @hk.transform_with_state
    def forward(
        dna_sequence: Float[Array, 'B S 4'], organism_index: Int[Array, 'B']
    ):
      return model_lib.AlphaGenome(output_metadata)(
          dna_sequence, organism_index
      )

    seq_length = int(2**14)
    key = jax.random.PRNGKey(42)
    dna_sequence = jnp.zeros((1, seq_length, 4))
    organism_index = jnp.zeros((1,), dtype=jnp.int32)

    params, state = forward.init(key, dna_sequence, organism_index)

    (predictions, embeddings), _ = forward.apply(
        params, state, key, dna_sequence, organism_index
    )
    self.assertEqual(embeddings.embeddings_1bp.shape, (1, seq_length, 1536))
    self.assertEqual(
        embeddings.embeddings_128bp.shape, (1, seq_length // 128, 3072)
    )
    self.assertEqual(
        embeddings.embeddings_pair.shape,
        (1, seq_length // 128 // 16, seq_length // 128 // 16, 128),
    )
    self.assertEqual(
        predictions['atac']['scaled_predictions_1bp'].shape,
        (1, seq_length, num_tracks),
    )
    self.assertEqual(
        predictions['atac']['scaled_predictions_128bp'].shape,
        (1, seq_length // 128, num_tracks),
    )

  def test_predict_junctions_with_same_params(self):
    seq_length = 2048
    num_tissues = 15
    num_splice_sites = 17
    output_metadata = {
        dna_model.Organism.HOMO_SAPIENS: metadata_lib.AlphaGenomeOutputMetadata(
            splice_sites=_mock_track_metadata(5),
            splice_junctions=_mock_junction_metadata(num_tissues),
        ),
    }

    @hk.transform_with_state
    def forward(
        dna_sequence: Float[Array, 'B S 4'], organism_index: Int[Array, 'B']
    ):
      return model_lib.AlphaGenome(
          output_metadata, num_splice_sites=num_splice_sites
      )(dna_sequence, organism_index)

    key = jax.random.PRNGKey(42)
    dna_sequence = jax.ShapeDtypeStruct((1, seq_length, 4), jnp.float32)
    organism_index = jax.ShapeDtypeStruct((1,), jnp.int32)

    params, state = jax.eval_shape(
        forward.init, key, dna_sequence, organism_index
    )

    (predictions, embeddings), _ = jax.eval_shape(
        forward.apply, params, state, key, dna_sequence, organism_index
    )

    @hk.transform_with_state
    def predict_junctions_fn(embeddings, splice_site_positions, organism_index):
      return model_lib.AlphaGenome(
          output_metadata, num_splice_sites=num_splice_sites
      ).predict_junctions(embeddings, splice_site_positions, organism_index)

    splice_site_positions = predictions['splice_sites_junction'][
        'splice_site_positions'
    ]
    self.assertEqual(splice_site_positions.shape, (1, 4, num_splice_sites))
    junction_predictions, _ = jax.eval_shape(
        predict_junctions_fn.apply,
        params,
        state,
        key,
        embeddings.embeddings_1bp,
        splice_site_positions,
        organism_index,
    )
    self.assertIsNotNone(junction_predictions)
    self.assertEqual(
        junction_predictions['predictions'].shape,
        (1, num_splice_sites, num_splice_sites, num_tissues * 2),
    )
    self.assertEqual(
        junction_predictions['splice_junction_mask'].shape,
        (1, num_splice_sites, num_splice_sites, num_tissues * 2),
    )

  def test_loss_and_grads(self):
    seq_length = 131072  # Must be larger or equal to multinomial_resolution.
    batch_size = 1
    num_splice_sites = 17
    num_tissues_junctions = 1
    output_metadata = {
        dna_model.Organism.HOMO_SAPIENS: metadata_lib.AlphaGenomeOutputMetadata(
            atac=_mock_track_metadata(num_tracks=1),
            contact_maps=_mock_track_metadata(num_tracks=1),
            splice_sites=_mock_track_metadata(num_tracks=5),
            splice_site_usage=_mock_track_metadata(num_tracks=4),
            splice_junctions=_mock_junction_metadata(
                num_tissues=num_tissues_junctions
            ),
        )
    }

    batch = schemas.DataBatch(
        dna_sequence=jnp.zeros((batch_size, seq_length, 4), dtype=jnp.float32),
        organism_index=jnp.zeros((batch_size,), dtype=jnp.int32),
        atac=jnp.zeros((batch_size, seq_length, 1), dtype=jnp.float32),
        atac_mask=jnp.ones((batch_size, 1, 1), dtype=bool),
        contact_maps=jnp.zeros(
            (batch_size, seq_length // 2048, seq_length // 2048, 1),
            dtype=jnp.float32,
        ),
        splice_sites=jnp.zeros((batch_size, seq_length, 5), dtype=bool),
        splice_site_usage=jnp.zeros(
            (batch_size, seq_length, 4), dtype=jnp.float32
        ),
        splice_junctions=jnp.zeros(
            (
                batch_size,
                num_splice_sites,
                num_splice_sites,
                num_tissues_junctions * 2,
            ),
            dtype=jnp.float32,
        ),
    )

    @hk.transform_with_state
    def forward(batch):
      return model_lib.AlphaGenome(
          output_metadata, num_splice_sites=num_splice_sites
      ).loss(batch)

    key = jax.random.PRNGKey(42)
    params, state = jax.eval_shape(forward.init, key, batch)
    (loss, scalars, predictions), _ = jax.eval_shape(
        forward.apply, params, state, key, batch
    )
    chex.assert_shape(loss, ())
    chex.assert_tree_shape(scalars, ())
    chex.assert_tree_shape_prefix(predictions, (batch_size,))
    self.assertContainsSubset(
        [
            'atac_loss',
            'contact_maps_loss',
            'splice_sites_classification_loss',
            'splice_sites_usage_loss',
            'splice_sites_junction_loss',
        ],
        scalars.keys(),
    )

    def grads_fn(params, state, batch):
      def _loss_fn(params):
        (loss, scalars, predictions), new_state = forward.apply(
            params, state, None, batch
        )
        return loss, (scalars, predictions, new_state)

      (loss, (scalars, predictions, new_state)), grads = jax.value_and_grad(
          _loss_fn, has_aux=True
      )(params)
      return loss, scalars, new_state, predictions, grads

    loss2, scalars2, state2, predictions2, grads = jax.eval_shape(
        grads_fn, params, state, batch
    )
    chex.assert_trees_all_equal_shapes(loss, loss2)
    chex.assert_trees_all_equal_shapes(scalars, scalars2)
    chex.assert_trees_all_equal_shapes(state, state2)
    chex.assert_trees_all_equal_shapes(predictions, predictions2)
    chex.assert_trees_all_equal_shapes(params, grads)

  def test_finetuning_weights_merging(self):
    """Tests that the weights can be merged for fine-tuning.

    We test this by creating a base model and a fine-tuning model, where the
    fine-tuning model has fewer heads and tracks than the base model. We then
    merge the weights of the base model and the fine-tuning model, and check
    that the loss can be computed with the merged weights.
    """
    seq_length = 131072
    batch_size = 1
    key = jax.random.key(0)

    # Setup base model.
    base_metadata = {
        dna_model.Organism.HOMO_SAPIENS: metadata_lib.AlphaGenomeOutputMetadata(
            atac=_mock_track_metadata(num_tracks=(atac_tracks_base := 5)),
            rna_seq=_mock_track_metadata(num_tracks=(rna_seq_tracks_base := 3)),
        ),
    }

    @hk.transform_with_state
    def base_forward(batch: schemas.DataBatch):
      return model_lib.AlphaGenome(base_metadata).loss(batch)

    base_batch = schemas.DataBatch(
        dna_sequence=np.zeros((batch_size, seq_length, 4), dtype=np.float32),
        organism_index=np.zeros((batch_size,), dtype=np.int32),
        atac=jnp.zeros(
            (batch_size, seq_length, atac_tracks_base), dtype=np.float32
        ),
        atac_mask=np.ones((batch_size, 1, atac_tracks_base), dtype=bool),
        rna_seq=np.zeros(
            (batch_size, seq_length, rna_seq_tracks_base), dtype=np.float32
        ),
        rna_seq_mask=np.ones((batch_size, 1, rna_seq_tracks_base), dtype=bool),
    )
    base_params, base_state = jax.eval_shape(base_forward.init, key, base_batch)

    # 2. Setup fine-tuning model.
    ft_metadata = {
        dna_model.Organism.HOMO_SAPIENS: metadata_lib.AlphaGenomeOutputMetadata(
            atac=_mock_track_metadata(num_tracks=(atac_tracks_ft := 7)),
        ),
    }

    @hk.transform_with_state
    def ft_forward(batch: schemas.DataBatch):
      return model_lib.AlphaGenome(ft_metadata).loss(batch)

    ft_batch = schemas.DataBatch(
        dna_sequence=np.zeros((batch_size, seq_length, 4), dtype=np.float32),
        organism_index=np.zeros((batch_size,), dtype=np.int32),
        atac=jnp.zeros(
            (batch_size, seq_length, atac_tracks_ft), dtype=np.float32
        ),
        atac_mask=np.ones((batch_size, 1, atac_tracks_ft), dtype=bool),
    )
    ft_params, ft_state = jax.eval_shape(ft_forward.init, key, ft_batch)

    # 3. Merge weights: base model weights + fine-tuning model head weights.
    def merge(base_weights, ft_weights):
      ft_head_weights = hk.data_structures.filter(
          lambda module_name, name, v: 'head' in module_name, ft_weights
      )
      return hk.data_structures.merge(base_weights, ft_head_weights)

    merged_params = merge(base_params, ft_params)
    merged_state = merge(base_state, ft_state)

    # 4. Check that we can compute loss with merged weights
    (loss, scalars, predictions), _ = jax.eval_shape(
        ft_forward.apply, merged_params, merged_state, key, ft_batch
    )
    chex.assert_shape(loss, ())
    self.assertContainsSubset(['atac_loss'], scalars.keys())
    self.assertNotIn('rna_seq_loss', scalars.keys())
    chex.assert_tree_shape_prefix(predictions, (batch_size,))
    self.assertEqual(
        predictions['atac']['scaled_predictions_1bp'].shape,
        (batch_size, seq_length, atac_tracks_ft),
    )

  def test_freeze_trunk_embeddings(self):
    if jax.local_devices()[0].platform.lower() not in {'gpu', 'tpu'}:
      self.skipTest('This test requires accelerator devices.')

    seq_length, batch_size, key = 131072, 1, jax.random.key(0)

    metadata = {
        dna_model.Organism.HOMO_SAPIENS: metadata_lib.AlphaGenomeOutputMetadata(
            atac=_mock_track_metadata(num_tracks=(atac_tracks_base := 5)),
            rna_seq=_mock_track_metadata(num_tracks=(rna_seq_tracks_base := 3)),
        ),
    }

    @hk.transform_with_state
    def forward(batch: schemas.DataBatch):
      return model_lib.AlphaGenome(metadata, freeze_trunk_embeddings=True).loss(
          batch
      )

    @jax.jit
    def grads_fn(params, state, batch):
      def loss_fn(params, state, batch):
        (loss, _, _), _ = forward.apply(params, state, None, batch)
        return loss

      grads_fn = jax.grad(loss_fn)
      return grads_fn(params, state, batch)

    batch = schemas.DataBatch(
        dna_sequence=np.zeros((batch_size, seq_length, 4), dtype=np.float32),
        organism_index=np.zeros((batch_size,), dtype=np.int32),
        atac=jnp.zeros(
            (batch_size, seq_length, atac_tracks_base), dtype=np.float32
        ),
        atac_mask=np.ones((batch_size, 1, atac_tracks_base), dtype=bool),
        rna_seq=np.zeros(
            (batch_size, seq_length, rna_seq_tracks_base), dtype=np.float32
        ),
        rna_seq_mask=np.ones((batch_size, 1, rna_seq_tracks_base), dtype=bool),
    )
    batch = jax.device_put(batch, jax.local_devices()[0])

    params, state = jax.jit(forward.init)(key, batch)
    grads = jax.jit(grads_fn)(params, state, batch)
    chex.assert_trees_all_equal_shapes(params, grads)
    grads_trunk = hk.data_structures.filter(
        lambda module_name, name, v: 'head' not in module_name, grads
    )
    grads_head = hk.data_structures.filter(
        lambda module_name, name, v: 'head' in module_name, grads
    )
    # Gradients in the trunk should be zero.
    chex.assert_trees_all_equal(
        grads_trunk,
        jax.tree.map(np.zeros_like, grads_trunk),
    )
    # Gradients in the head should be non-zero.
    head_grad_sq_norm = jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(jnp.square(x)),
        grads_head,
        initializer=0.0,
    )
    self.assertGreater(head_grad_sq_norm, 0.0)


if __name__ == '__main__':
  absltest.main()
