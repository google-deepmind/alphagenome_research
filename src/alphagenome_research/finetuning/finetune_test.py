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

import os

from absl.testing import absltest
from alphagenome_research.finetuning import finetune
from alphagenome_research.model import dna_model
from alphagenome_research.model import model as model_lib
from alphagenome_research.model import schemas
from alphagenome_research.model.metadata import metadata as metadata_lib
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd


def _create_mock_df(modality_name: str, num_tracks: int) -> pd.DataFrame | None:
  if num_tracks == 0:
    return None
  return pd.DataFrame({
      'file_path': [f'path{i}' for i in range(num_tracks)],
      'name': [f'{modality_name}_track{i}' for i in range(num_tracks)],
      'strand': ['.'] * num_tracks,
  })


def get_mock_metadata(
    num_atac_tracks: int = 0, num_dnase_tracks: int = 0, rna_seq_tracks: int = 0
) -> metadata_lib.AlphaGenomeOutputMetadata:
  return metadata_lib.AlphaGenomeOutputMetadata(
      atac=_create_mock_df('atac', num_atac_tracks),
      dnase=_create_mock_df('dnase', num_dnase_tracks),
      rna_seq=_create_mock_df('rna_seq', rna_seq_tracks),
  )


class FinetuneTest(absltest.TestCase):

  def test_finetune_train_step(self):
    seq_length, batch_size, key = 131072, 1, jax.random.key(0)

    # Setup base model.
    base_metadata = {
        dna_model.Organism.HOMO_SAPIENS: get_mock_metadata(
            num_atac_tracks=(atac_tracks_base := 2),
            rna_seq_tracks=(rna_seq_tracks_base := 1),
        )
    }

    @hk.transform_with_state
    def base_forward(batch: schemas.DataBatch):
      return model_lib.AlphaGenome(base_metadata).loss(batch)

    # Batch contains ATAC and RNA-seq data.
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

    # Setup fine-tuning model with different metadata.
    ft_metadata = {
        dna_model.Organism.HOMO_SAPIENS: get_mock_metadata(
            num_atac_tracks=(atac_tracks_ft := 3),  # More ATAC tracks.
            rna_seq_tracks=0,  # No RNA-seq tracks.
            num_dnase_tracks=(dnase_tracks_ft := 7),  # New DNase tracks.
        )
    }
    batch_ft = schemas.DataBatch(
        dna_sequence=np.zeros((batch_size, seq_length, 4), dtype=np.float32),
        organism_index=np.zeros((batch_size,), dtype=np.int32),
        atac=jnp.zeros(
            (batch_size, seq_length, atac_tracks_ft), dtype=np.float32
        ),
        atac_mask=np.ones((batch_size, 1, atac_tracks_ft), dtype=bool),
        dnase=jnp.zeros(
            (batch_size, seq_length, dnase_tracks_ft), dtype=np.float32
        ),
        dnase_mask=np.ones((batch_size, 1, dnase_tracks_ft), dtype=bool),
    )

    forward_fn = finetune.get_forward_fn(ft_metadata)
    params_ft, _ = jax.eval_shape(forward_fn.init, key, batch_ft)

    def merge(frozen, trainable):
      trainable = hk.data_structures.filter(
          lambda module_name, *_: 'head' in module_name, trainable
      )
      merged = hk.data_structures.merge(frozen, trainable)
      return merged

    params = merge(base_params, params_ft)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = jax.eval_shape(optimizer.init, params)
    train_step = finetune.get_train_step(forward_fn.apply, optimizer)
    _, _, _, scalars = jax.eval_shape(
        train_step, params, base_state, opt_state, batch_ft
    )
    self.assertIn('loss', scalars)
    self.assertIn('atac_loss', scalars)
    self.assertIn('dnase_loss', scalars)
    self.assertNotIn('rna_seq_loss', scalars)  # RNA-seq head not in FT model.

  def test_create_finetuned_dna_sequence_model(self):
    seq_length, batch_size, key = 131072, 1, jax.random.key(0)
    ft_metadata = {
        dna_model.Organism.HOMO_SAPIENS: get_mock_metadata(
            num_atac_tracks=(atac_tracks_ft := 3),  # More ATAC tracks.
            rna_seq_tracks=0,  # No RNA-seq tracks.
            num_dnase_tracks=(dnase_tracks_ft := 7),  # New DNase tracks.
        )
    }
    batch_ft = schemas.DataBatch(
        dna_sequence=np.zeros((batch_size, seq_length, 4), dtype=np.float32),
        organism_index=np.zeros((batch_size,), dtype=np.int32),
        atac=jnp.zeros(
            (batch_size, seq_length, atac_tracks_ft), dtype=np.float32
        ),
        atac_mask=np.ones((batch_size, 1, atac_tracks_ft), dtype=bool),
        dnase=jnp.zeros(
            (batch_size, seq_length, dnase_tracks_ft), dtype=np.float32
        ),
        dnase_mask=np.ones((batch_size, 1, dnase_tracks_ft), dtype=bool),
    )

    forward_fn = finetune.get_forward_fn(ft_metadata)
    params_ft, state_ft = jax.eval_shape(forward_fn.init, key, batch_ft)

    checkpointer = ocp.StandardCheckpointer()
    ckpt_dir = self.create_tempdir().full_path
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint')
    checkpointer.save(
        ckpt_path,
        jax.tree_util.tree_map(
            lambda x: jnp.empty(x.shape, x.dtype), (params_ft, state_ft)
        ),
    )
    checkpointer.wait_until_finished()

    ft_organism_settings = {
        k: dna_model.OrganismSettings(metadata=v)
        for k, v in ft_metadata.items()
    }
    finetuned_dna_sequence_model = dna_model.create(
        ckpt_path,
        organism_settings=ft_organism_settings,
        device=jax.devices('cpu')[0],
    )
    self.assertIsInstance(
        finetuned_dna_sequence_model, dna_model.AlphaGenomeModel
    )


if __name__ == '__main__':
  absltest.main()
