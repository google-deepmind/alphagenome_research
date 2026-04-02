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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome import typing
from alphagenome.data import junction_data
from alphagenome.data import track_data
from alphagenome.models import dna_model
from alphagenome.models import dna_output
from alphagenome_research.io import bundles
from alphagenome_research.model import embeddings
from alphagenome_research.model import heads
from alphagenome_research.model import schemas
from alphagenome_research.model.metadata import metadata as metadata_lib
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


_EMBEDDING_DIM_PAIR = 128
_EMBEDDING_DIM_1BP = 1536
_EMBEDDING_DIM_128BP = 3072

# Number of tracks for human and mouse.
_MOCK_ATAC_NUM_TRACKS = (9, 7)
_MOCK_CHIP_TF_NUM_TRACKS = (11, 10)
_MOCK_CONTACT_MAPS_NUM_TRACKS = (8, 7)
_MOCK_SPLICE_SITES_NUM_TRACKS = (5, 5)
_MOCK_SPLICE_SITES_USAGE_NUM_TRACKS = (10, 9)
_MOCK_SPLICE_SITES_JUNCTION_NUM_TISSUES = (15, 12)  # Tissues, not tracks.


_ORGANSIM_INDEX = {
    dna_model.Organism.HOMO_SAPIENS: 0,
    dna_model.Organism.MUS_MUSCULUS: 1,
}


def get_mock_output_metadata(
    organism: dna_model.Organism,
) -> metadata_lib.AlphaGenomeOutputMetadata:
  """Returns mock output metadata for the given organisms."""

  def _make_metadata_df(
      total_tracks,
      num_tracks,
      strand=None,
  ) -> track_data.TrackMetadata:
    data = {
        'name': [f'track_{i}' for i in range(num_tracks)],
        'nonzero_mean': [1.0] * num_tracks,
    }
    if strand is not None:
      data['strand'] = strand[:num_tracks]
    df_tracks = pd.DataFrame(data)
    padding_data = {
        'name': ['padding'] * (total_tracks - num_tracks),
        'nonzero_mean': [0.0] * (total_tracks - num_tracks),
    }
    if strand is not None:
      padding_data['strand'] = ['.'] * (total_tracks - num_tracks)
    df_padding = pd.DataFrame(padding_data)
    return pd.concat([df_tracks, df_padding], ignore_index=True)

  def _make_junction_metadata_df(num_tissues) -> junction_data.JunctionMetadata:
    return pd.DataFrame({
        'name': [f'tissue_{i}' for i in range(num_tissues)],
    })

  organism_idx = _ORGANSIM_INDEX[organism]
  total_atac = max(_MOCK_ATAC_NUM_TRACKS)
  n_atac = _MOCK_ATAC_NUM_TRACKS[organism_idx]
  # Default strand layout: mix of +, -, and . strands.
  atac_strands = (
      ['+'] * (total_atac // 3)
      + ['-'] * (total_atac // 3)
      + ['.'] * (total_atac - 2 * (total_atac // 3))
  )
  return metadata_lib.AlphaGenomeOutputMetadata(
      atac=_make_metadata_df(total_atac, n_atac, strand=atac_strands),
      chip_tf=_make_metadata_df(
          max(_MOCK_CHIP_TF_NUM_TRACKS),
          _MOCK_CHIP_TF_NUM_TRACKS[organism_idx],
      ),
      contact_maps=_make_metadata_df(
          max(_MOCK_CONTACT_MAPS_NUM_TRACKS),
          _MOCK_CONTACT_MAPS_NUM_TRACKS[organism_idx],
      ),
      splice_sites=_make_metadata_df(
          max(_MOCK_SPLICE_SITES_NUM_TRACKS),
          _MOCK_SPLICE_SITES_NUM_TRACKS[organism_idx],
      ),
      splice_site_usage=_make_metadata_df(
          max(_MOCK_SPLICE_SITES_USAGE_NUM_TRACKS),
          _MOCK_SPLICE_SITES_USAGE_NUM_TRACKS[organism_idx],
      ),
      splice_junctions=_make_junction_metadata_df(
          _MOCK_SPLICE_SITES_JUNCTION_NUM_TISSUES[organism_idx],
      ),
  )


@typing.jaxtyped
def get_mock_embeddings(
    batch_size: int,
    sequence_length: int,
) -> embeddings.Embeddings:
  """Returns mock embeddings for testing."""
  return embeddings.Embeddings(
      embeddings_pair=jnp.zeros(
          (
              batch_size,
              sequence_length // 2048,
              sequence_length // 2048,
              _EMBEDDING_DIM_PAIR,
          ),
          dtype=jnp.bfloat16,
      ),
      embeddings_1bp=jnp.zeros(
          (batch_size, sequence_length, _EMBEDDING_DIM_1BP), dtype=jnp.bfloat16
      ),
      embeddings_128bp=jnp.zeros(
          (batch_size, sequence_length // 128, _EMBEDDING_DIM_128BP),
          dtype=jnp.bfloat16,
      ),
  )


@typing.jaxtyped
def get_mock_batch(
    batch_size: int,
    sequence_length: int,
    num_splice_sites: int,
    organism_index: int = 0,
) -> schemas.DataBatch:
  """Returns mock target data for testing."""
  return schemas.DataBatch(
      dna_sequence=jnp.zeros(
          (batch_size, sequence_length, 4), dtype=jnp.float32
      ),
      organism_index=jnp.ones((batch_size,), dtype=jnp.int32) * organism_index,
      atac=jnp.zeros(
          (batch_size, sequence_length, max(_MOCK_ATAC_NUM_TRACKS)),
          dtype=jnp.float32,
      ),
      atac_mask=jnp.ones(
          (batch_size, 1, max(_MOCK_ATAC_NUM_TRACKS)),
          dtype=bool,
      ),
      chip_tf=jnp.zeros(
          (batch_size, sequence_length // 128, max(_MOCK_CHIP_TF_NUM_TRACKS)),
          dtype=jnp.float32,
      ),
      chip_tf_mask=jnp.ones(
          (batch_size, 1, max(_MOCK_CHIP_TF_NUM_TRACKS)),
          dtype=bool,
      ),
      contact_maps=jnp.zeros(
          (
              batch_size,
              sequence_length // 16 // 128,
              sequence_length // 16 // 128,
              max(_MOCK_CONTACT_MAPS_NUM_TRACKS),
          ),
          dtype=jnp.float32,
      ),
      splice_junctions=jnp.zeros(
          (
              batch_size,
              num_splice_sites,
              num_splice_sites,
              max(_MOCK_SPLICE_SITES_JUNCTION_NUM_TISSUES) * 2,
          ),
          dtype=jnp.float32,
      ),
      splice_site_positions=jnp.zeros(
          (batch_size, 4, num_splice_sites), dtype=jnp.int32
      ),
      splice_site_usage=jnp.zeros(
          (
              batch_size,
              sequence_length,
              max(_MOCK_SPLICE_SITES_USAGE_NUM_TRACKS),
          ),
          dtype=jnp.float32,
      ),
      splice_sites=jnp.ones(
          (batch_size, sequence_length, max(_MOCK_SPLICE_SITES_NUM_TRACKS)),
          dtype=bool,
      ),
  )


class HeadsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 1
    # Set >= multinomial resolution of 2^17.
    self.sequence_length = int(2**17)
    self.num_splice_sites = 128

  def test_head_num_tracks_with_default_metadata(self):
    metadata = {
        dna_model.Organism.HOMO_SAPIENS: metadata_lib.load(
            dna_model.Organism.HOMO_SAPIENS
        ),
        dna_model.Organism.MUS_MUSCULUS: metadata_lib.load(
            dna_model.Organism.MUS_MUSCULUS
        ),
    }
    expected_num_tracks = {
        heads.HeadName.ATAC: 256,
        heads.HeadName.DNASE: 384,
        heads.HeadName.PROCAP: 128,
        heads.HeadName.CAGE: 640,
        heads.HeadName.RNA_SEQ: 768,
        heads.HeadName.CHIP_TF: 1664,
        heads.HeadName.CHIP_HISTONE: 1152,
        heads.HeadName.CONTACT_MAPS: 28,
        heads.HeadName.SPLICE_SITES_CLASSIFICATION: 5,
        heads.HeadName.SPLICE_SITES_USAGE: 734,
        heads.HeadName.SPLICE_SITES_JUNCTION: 734,
    }
    for head_name in heads.HeadName:
      with self.subTest(head_name.value):
        config = heads.get_head_config(head_name)
        head = heads.create_head(config, metadata)
        self.assertEqual(head.num_tracks, expected_num_tracks[head_name])

  def test_genome_tracks_head_no_metadata_raises_error(self):
    metadata = {
        dna_model.Organism.HOMO_SAPIENS: (
            metadata_lib.AlphaGenomeOutputMetadata()
        ),
    }
    with self.assertRaisesRegex(
        ValueError, 'No metadata found for any organism'
    ):
      _ = heads.GenomeTracksHead(
          name='test_head',
          output_type=dna_output.OutputType.ATAC,
          apply_squashing=True,
          resolutions=[1],
          bundle=bundles.BundleName.ATAC,
          metadata=metadata,
      )

  def test_genome_tracks_head_inconsistent_num_tracks_raises_error(self):
    def _make_metadata_df(num_tracks) -> track_data.TrackMetadata:
      return pd.DataFrame({
          'name': [f'track_{i}' for i in range(num_tracks)],
          'nonzero_mean': [1.0] * num_tracks,
      })

    metadata = {
        dna_model.Organism.HOMO_SAPIENS: metadata_lib.AlphaGenomeOutputMetadata(
            atac=_make_metadata_df(10)
        ),
        dna_model.Organism.MUS_MUSCULUS: metadata_lib.AlphaGenomeOutputMetadata(
            atac=_make_metadata_df(11)
        ),
    }
    with self.assertRaisesRegex(
        ValueError,
        'Number of tracks is not the same for all organisms',
    ):
      _ = heads.GenomeTracksHead(
          name='test_head',
          output_type=dna_output.OutputType.ATAC,
          apply_squashing=True,
          resolutions=[1],
          bundle=bundles.BundleName.ATAC,
          metadata=metadata,
      )

  def _test_head(
      self,
      head: heads.Head,
      expected_params_shape: Any,
      expected_output_shape: Any,
  ):
    @hk.transform
    def forward(embeddings_input, batch):
      kwargs = {}
      if isinstance(head, heads.SpliceSitesJunctionHead):
        kwargs['splice_site_positions'] = batch.splice_site_positions
      output = head(embeddings_input, batch.organism_index, **kwargs)
      loss = head.loss(output, batch)
      return output, loss

    rng = jax.random.PRNGKey(42)
    embeddings_input = get_mock_embeddings(
        self.batch_size, self.sequence_length
    )
    batch = get_mock_batch(
        self.batch_size, self.sequence_length, self.num_splice_sites
    )

    params = forward.init(rng, embeddings_input, batch)
    output, loss = forward.apply(params, rng, embeddings_input, batch)

    to_shape = lambda t: jax.tree.map(lambda x: x.shape, t)
    chex.assert_trees_all_equal(to_shape(params), expected_params_shape)
    chex.assert_trees_all_equal(to_shape(output), expected_output_shape)
    self.assertTrue(np.isfinite(loss['loss']).all())

  @parameterized.named_parameters(
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
  def test_genome_tracks_head(self, organisms: tuple[dna_model.Organism, ...]):
    metadata = {
        organism: get_mock_output_metadata(organism) for organism in organisms
    }
    head = heads.GenomeTracksHead(
        name='test_head',
        output_type=dna_output.OutputType.ATAC,
        apply_squashing=True,
        resolutions=[1, 128],
        bundle=bundles.BundleName.ATAC,
        metadata=metadata,
    )
    with self.subTest('num_organisms'):
      self.assertLen(head.get_multi_organism_track_mask(), len(metadata))
    with self.subTest('num_tracks'):
      self.assertEqual(head.num_tracks, max(_MOCK_ATAC_NUM_TRACKS))
    with self.subTest('track_mask'):
      for i, organism in enumerate(organisms):
        organism_idx = _ORGANSIM_INDEX[organism]
        self.assertEqual(
            np.sum(head.get_multi_organism_track_mask()[i]),
            _MOCK_ATAC_NUM_TRACKS[organism_idx],
        )

    num_organisms = len(metadata)
    num_tracks = head.num_tracks
    expected_params_shape = {
        'test_head/resolution_1': {'learnt_scale': (num_organisms, num_tracks)},
        'test_head/resolution_1/multi_organism_linear': {
            'b': (num_organisms, num_tracks),
            'w': (num_organisms, _EMBEDDING_DIM_1BP, num_tracks),
        },
        'test_head/resolution_128': {
            'learnt_scale': (num_organisms, num_tracks)
        },
        'test_head/resolution_128/multi_organism_linear': {
            'b': (num_organisms, num_tracks),
            'w': (num_organisms, _EMBEDDING_DIM_128BP, num_tracks),
        },
    }
    expected_output_shape = {
        'scaled_predictions_1bp': (
            self.batch_size,
            self.sequence_length,
            num_tracks,
        ),
        'predictions_1bp': (self.batch_size, self.sequence_length, num_tracks),
        'scaled_predictions_128bp': (
            self.batch_size,
            self.sequence_length // 128,
            num_tracks,
        ),
        'predictions_128bp': (
            self.batch_size,
            self.sequence_length // 128,
            num_tracks,
        ),
    }
    with self.subTest('params_shape'):
      self._test_head(
          head,
          expected_params_shape,
          expected_output_shape,
      )

  @parameterized.named_parameters(
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
  def test_contact_maps_head(self, organisms: tuple[dna_model.Organism, ...]):
    metadata = {
        organism: get_mock_output_metadata(organism) for organism in organisms
    }
    config = heads.get_head_config(heads.HeadName.CONTACT_MAPS)
    head = heads.create_head(config, metadata)

    with self.subTest('num_organisms'):
      self.assertLen(head.get_multi_organism_track_mask(), len(metadata))
    with self.subTest('num_tracks'):
      self.assertEqual(head.num_tracks, max(_MOCK_CONTACT_MAPS_NUM_TRACKS))
    with self.subTest('track_mask'):
      for i, organism in enumerate(organisms):
        organism_idx = _ORGANSIM_INDEX[organism]
        self.assertEqual(
            np.sum(head.get_multi_organism_track_mask()[i]),
            _MOCK_CONTACT_MAPS_NUM_TRACKS[organism_idx],
        )

    num_organisms = len(metadata)
    num_tracks = head.num_tracks
    prefix = config.name
    expected_params_shape = {
        f'{prefix}/multi_organism_linear': {
            'b': (num_organisms, num_tracks),
            'w': (num_organisms, _EMBEDDING_DIM_PAIR, num_tracks),
        },
    }
    expected_output_shape = {
        'predictions': (
            self.batch_size,
            self.sequence_length // 2048,
            self.sequence_length // 2048,
            num_tracks,
        ),
    }
    with self.subTest('params_shape'):
      self._test_head(
          head,
          expected_params_shape,
          expected_output_shape,
      )

  @parameterized.named_parameters(
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
  def test_splice_sites_classification_head(
      self, organisms: tuple[dna_model.Organism, ...]
  ):
    metadata = {
        organism: get_mock_output_metadata(organism) for organism in organisms
    }
    config = heads.get_head_config(heads.HeadName.SPLICE_SITES_CLASSIFICATION)
    head = heads.create_head(config, metadata)

    with self.subTest('num_organisms'):
      self.assertLen(head.get_multi_organism_track_mask(), len(metadata))
    with self.subTest('num_tracks'):
      self.assertEqual(head.num_tracks, max(_MOCK_SPLICE_SITES_NUM_TRACKS))
    with self.subTest('track_mask'):
      for i, organism in enumerate(organisms):
        organism_idx = _ORGANSIM_INDEX[organism]
        self.assertEqual(
            np.sum(head.get_multi_organism_track_mask()[i]),
            _MOCK_SPLICE_SITES_NUM_TRACKS[organism_idx],
        )

    num_organisms = len(metadata)
    num_tracks = head.num_tracks
    prefix = config.name
    expected_params_shape = {
        f'{prefix}/multi_organism_linear': {
            'b': (num_organisms, num_tracks),
            'w': (num_organisms, _EMBEDDING_DIM_1BP, num_tracks),
        },
    }
    expected_output_shape = {
        'predictions': (
            self.batch_size,
            self.sequence_length,
            num_tracks,
        ),
        'logits': (
            self.batch_size,
            self.sequence_length,
            num_tracks,
        ),
    }
    self._test_head(
        head,
        expected_params_shape,
        expected_output_shape,
    )

  @parameterized.named_parameters(
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
  def test_splice_sites_usage_head(
      self, organisms: tuple[dna_model.Organism, ...]
  ):
    metadata = {
        organism: get_mock_output_metadata(organism) for organism in organisms
    }
    config = heads.get_head_config(heads.HeadName.SPLICE_SITES_USAGE)
    head = heads.create_head(config, metadata)

    with self.subTest('num_organisms'):
      self.assertLen(head.get_multi_organism_track_mask(), len(metadata))
    with self.subTest('num_tracks'):
      self.assertEqual(
          head.num_tracks, max(_MOCK_SPLICE_SITES_USAGE_NUM_TRACKS)
      )
    with self.subTest('track_mask'):
      for i, organism in enumerate(organisms):
        organism_idx = _ORGANSIM_INDEX[organism]
        self.assertEqual(
            np.sum(head.get_multi_organism_track_mask()[i]),
            _MOCK_SPLICE_SITES_USAGE_NUM_TRACKS[organism_idx],
        )

    num_organisms = len(metadata)
    num_tracks = head.num_tracks
    prefix = config.name
    expected_params_shape = {
        f'{prefix}/multi_organism_linear': {
            'b': (num_organisms, num_tracks),
            'w': (num_organisms, _EMBEDDING_DIM_1BP, num_tracks),
        },
    }
    expected_output_shape = {
        'predictions': (
            self.batch_size,
            self.sequence_length,
            num_tracks,
        ),
        'logits': (
            self.batch_size,
            self.sequence_length,
            num_tracks,
        ),
    }
    self._test_head(
        head,
        expected_params_shape,
        expected_output_shape,
    )

  @parameterized.named_parameters(
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
  def test_splice_sites_junction_head(
      self, organisms: tuple[dna_model.Organism, ...]
  ):
    metadata = {
        organism: get_mock_output_metadata(organism) for organism in organisms
    }
    config = heads.get_head_config(heads.HeadName.SPLICE_SITES_JUNCTION)
    head = heads.create_head(config, metadata)
    assert isinstance(head, heads.SpliceSitesJunctionHead)

    with self.subTest('num_organisms'):
      self.assertLen(head.get_multi_organism_track_mask(), len(metadata))
    with self.subTest('num_tracks'):
      self.assertEqual(
          head.num_tracks, 2 * max(_MOCK_SPLICE_SITES_JUNCTION_NUM_TISSUES)
      )
    with self.subTest('track_mask'):
      for i, organism in enumerate(organisms):
        organism_idx = _ORGANSIM_INDEX[organism]
        self.assertEqual(
            np.sum(head.get_multi_organism_track_mask()[i]),
            2 * _MOCK_SPLICE_SITES_JUNCTION_NUM_TISSUES[organism_idx],
        )

    num_organisms = len(metadata)
    num_tracks = head.num_tracks
    num_tissues = head.max_num_tissues
    hidden_dim = 768
    prefix = config.name
    expected_params_shape = {
        f'{prefix}/pos_acceptor_logits': {
            'embeddings': (num_organisms, 2 * num_tissues * hidden_dim),
        },
        f'{prefix}/pos_donor_logits': {
            'embeddings': (num_organisms, 2 * num_tissues * hidden_dim),
        },
        f'{prefix}/neg_acceptor_logits': {
            'embeddings': (num_organisms, 2 * num_tissues * hidden_dim),
        },
        f'{prefix}/neg_donor_logits': {
            'embeddings': (num_organisms, 2 * num_tissues * hidden_dim),
        },
        f'{prefix}/multi_organism_linear': {
            'b': (num_organisms, hidden_dim),
            'w': (num_organisms, _EMBEDDING_DIM_1BP, hidden_dim),
        },
    }
    expected_output_shape = {
        'predictions': (
            self.batch_size,
            self.num_splice_sites,
            self.num_splice_sites,
            num_tracks,
        ),
        'splice_site_positions': (
            self.batch_size,
            4,
            self.num_splice_sites,
        ),
        'splice_junction_mask': (
            self.batch_size,
            self.num_splice_sites,
            self.num_splice_sites,
            num_tracks,
        ),
    }
    self._test_head(
        head,
        expected_params_shape,
        expected_output_shape,
    )

  @parameterized.named_parameters(
      dict(testcase_name='with_squashing', apply_squashing=True),
      dict(testcase_name='without_squashing', apply_squashing=False),
  )
  def test_scaling_and_unscaling(self, apply_squashing: bool):
    batch_size, sequence_length, num_tracks, resolution = 3, 2048, 10, 1
    key = jax.random.PRNGKey(42)
    key_data, key_means = jax.random.split(key)
    track_means = jax.random.uniform(key_means, (batch_size, num_tracks)) + 0.01
    input_data = jax.random.uniform(
        key_data, (batch_size, sequence_length, num_tracks), minval=0, maxval=99
    )
    scaled_data = heads.targets_scaling(
        input_data,
        track_means=track_means,
        resolution=resolution,
        apply_squashing=apply_squashing,
    )
    unscaled_data = heads.predictions_scaling(
        scaled_data,
        track_means=track_means,
        resolution=resolution,
        apply_squashing=apply_squashing,
    )
    chex.assert_trees_all_close(input_data, unscaled_data, rtol=1e-5, atol=1e-5)

  def test_cross_track_loss_is_finite(self):
    """Gene loss with gene_loss_weight > 0 returns finite loss and aux keys."""
    batch_size = 2
    sequence_length = 2**17
    num_tracks = max(_MOCK_ATAC_NUM_TRACKS)
    num_genes = 3

    organisms = (
        dna_model.Organism.HOMO_SAPIENS,
        dna_model.Organism.MUS_MUSCULUS,
    )
    metadata = {
        organism: get_mock_output_metadata(organism) for organism in organisms
    }

    head = heads.GenomeTracksHead(
        name='test_cross_track',
        output_type=dna_output.OutputType.ATAC,
        apply_squashing=False,
        resolutions=[1],
        bundle=bundles.BundleName.ATAC,
        metadata=metadata,
        gene_loss_weight=1.0,
        gene_cross_track_weight=5.0,
    )

    rng = jax.random.PRNGKey(0)
    rng_pred, rng_tgt = jax.random.split(rng)
    scaled_preds = jax.random.uniform(
        rng_pred, (batch_size, sequence_length, num_tracks), minval=0.1
    )
    targets = jax.random.uniform(
        rng_tgt, (batch_size, sequence_length, num_tracks), minval=0.0
    )
    mask = jnp.ones((batch_size, 1, num_tracks), dtype=bool)

    gene_mask = jnp.zeros(
        (batch_size, sequence_length, 2, num_genes), dtype=bool
    )
    gene_mask = gene_mask.at[:, 10:50, 0, 0].set(True)
    gene_mask = gene_mask.at[:, 100:200, 1, 1].set(True)
    gene_mask = gene_mask.at[:, 300:350, 0, 2].set(True)

    batch = schemas.DataBatch(
        organism_index=jnp.zeros((batch_size,), dtype=jnp.int32),
        atac=targets,
        atac_mask=mask,
        gene_mask=gene_mask,
    )
    preds_dict = {'scaled_predictions_1bp': scaled_preds}

    result = head.loss(preds_dict, batch)
    self.assertTrue(np.isfinite(result['loss']))
    self.assertTrue(np.isfinite(result['gene_loss_total_count_1bp']))
    self.assertTrue(np.isfinite(result['gene_loss_positional_1bp']))

  def test_cross_track_loss_strand_filtering(self):
    """Strand filtering with 3 tracks (+, -, .) and deterministic values."""
    batch_size = 1
    sequence_length = 2**17
    num_tracks = 3
    num_genes = 2

    organisms = (
        dna_model.Organism.HOMO_SAPIENS,
        dna_model.Organism.MUS_MUSCULUS,
    )

    # Build minimal metadata: 3 ATAC tracks with strands +, -, .
    def _make_3track_metadata(organism):  # pylint: disable=unused-argument
      atac_df = pd.DataFrame({
          'name': ['track_plus', 'track_minus', 'track_unstranded'],
          'nonzero_mean': [1.0, 1.0, 1.0],
          'strand': ['+', '-', '.'],
      })
      return metadata_lib.AlphaGenomeOutputMetadata(atac=atac_df)

    metadata = {o: _make_3track_metadata(o) for o in organisms}

    head = heads.GenomeTracksHead(
        name='test_strand_3track',
        output_type=dna_output.OutputType.ATAC,
        apply_squashing=False,
        resolutions=[1],
        bundle=bundles.BundleName.ATAC,
        metadata=metadata,
        gene_loss_weight=1.0,
        gene_cross_track_weight=5.0,
    )

    # Predictions: all ones.
    scaled_preds = jnp.ones((batch_size, sequence_length, num_tracks))
    # Targets: (1, 2, 3) per position across the 3 tracks.
    targets = jnp.broadcast_to(
        jnp.array([1.0, 2.0, 3.0]), (batch_size, sequence_length, num_tracks)
    )
    mask = jnp.ones((batch_size, 1, num_tracks), dtype=bool)

    # Two short genes: gene 0 on positive strand, gene 1 on negative strand.
    gene_mask = jnp.zeros(
        (batch_size, sequence_length, 2, num_genes), dtype=bool
    )
    gene_mask = gene_mask.at[:, 10:20, 0, 0].set(True)  # pos-strand gene
    gene_mask = gene_mask.at[:, 100:110, 1, 1].set(True)  # neg-strand gene

    batch = schemas.DataBatch(
        organism_index=jnp.zeros((batch_size,), dtype=jnp.int32),
        atac=targets,
        atac_mask=mask,
        gene_mask=gene_mask,
    )
    preds_dict = {'scaled_predictions_1bp': scaled_preds}

    result = head.loss(preds_dict, batch)

    # Gene lengths: 10 for both active genes.
    # predictions: 1.0. targets: (1.0, 2.0, 3.0) along channels.
    # Gene 0 (pos-strand): tracks + and . are active (indices 0 and 2)
    #   y_pred_0 = 1.0 + 1.0 = 2.0
    #   y_true_0 = 1.0 + 3.0 = 4.0
    # Gene 1 (neg-strand): tracks - and . are active (indices 1 and 2)
    #   y_pred_1 = 1.0 + 1.0 = 2.0
    #   y_true_1 = 2.0 + 3.0 = 5.0
    def expected_poisson(y_true, y_pred):
      return (y_pred - y_true * np.log(y_pred + 1e-7)) - (
          y_true - y_true * np.log(y_true + 1e-7)
      )

    expected_loss_0 = expected_poisson(4.0, 2.0)
    expected_loss_1 = expected_poisson(5.0, 2.0)

    # The _safe_masked_mean averages the loss across the 2 valid genes natively.
    avg_poisson = (expected_loss_0 + expected_loss_1) / 2.0

    # Divide by the max number of active channels among all genes (2 active
    # channels for both).
    expected_total_count = avg_poisson / 2.0

    self.assertAlmostEqual(
        float(result['gene_loss_total_count_1bp']),
        expected_total_count,
        places=5,
    )
    self.assertTrue(np.isfinite(float(result['gene_loss_positional_1bp'])))


if __name__ == '__main__':
  absltest.main()
