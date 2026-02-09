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
import pathlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import genome
from alphagenome.data import ontology
from alphagenome.models import dna_output
from alphagenome.models import interval_scorers
from alphagenome.models import variant_scorers
from alphagenome_research.io import fasta
from alphagenome_research.io import splicing
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata
import chex
import huggingface_hub
import jax
import jax.numpy as jnp
import kagglehub
from kagglehub import auth as kaggle_auth
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd


MOCK_SHAPES = ({}, {})


def _create_mock_gtf() -> pd.DataFrame:
  return pd.DataFrame({
      'Chromosome': 'chr1',
      'Start': [101, 101, 102, 103, 0, 0, 0, 80],
      'End': [200, 200, 200, 200, 108, 108, 40, 108],
      'Strand': ['+', '+', '+', '+', '-', '-', '-', '-'],
      'transcript_id': [None, None, 'T2', 'T2', None, 'T4', 'T4', 'T4'],
      'gene_id': ['G1', 'G1', 'G1', 'G1', 'G2', 'G2', 'G2', 'G2'],
      'Feature': [
          'gene',
          'transcript',
          'transcript',
          'exon',
          'gene',
          'transcript',
          'exon',
          'exon',
      ],
      'gene_type': 'protein_coding',
      'gene_name': 'GX_name',
      'transcript_type': 'protein_coding',
  })


def _create_mock_splice_sites() -> tuple[pd.DataFrame, pd.DataFrame]:
  return pd.DataFrame(
      {'Chromosome': 'chr1', 'Start': [40], 'Strand': ['-'], 'Tissue_0': [1]}
  ), pd.DataFrame(
      {'Chromosome': 'chr1', 'End': [80], 'Strand': ['-'], 'Tissue_0': [1]}
  )


def _create_polya_df_gtf() -> pd.DataFrame:
  return pd.DataFrame({
      'End': [103, 104, 1, 81],
      'Start': [102, 103, 0, 80],
      'cutmode': [102, 103, 0, 80],
      'pas_strand': ['+', '+', '-', '-'],
      'pas_id': ['P1', 'P2', 'P3', 'P4'],
      'gene_id': ['G1', 'G1', 'G2', 'G2'],
      'pas_gene_id': ['G1', 'G1', 'G2', 'G2'],
      'Chromosome': 'chr1',
  })


def _get_test_fasta_path() -> str | os.PathLike[str]:
  root_dir = pathlib.Path(__file__).parent.parent
  return root_dir / 'io' / 'testdata' / 'example.fa'


class DnaModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._num_tissues = 10
    self._num_splice_sites = 100
    self._metadata = metadata.AlphaGenomeOutputMetadata(
        atac=pd.DataFrame({
            'name': ['atac1', 'atac1'],
            'strand': ['+', '-'],
            'ontology_curie': ['CL:0000001', 'CL:0000001'],
            'nonzero_mean': [1.0, 1.0],
        }),
        dnase=pd.DataFrame({
            'name': ['acc_1'],
            'strand': ['.'],
            'ontology_curie': ['UBERON:0000001'],
            'nonzero_mean': [1.0],
        }),
        contact_maps=pd.DataFrame({
            'name': ['hic_1', 'hic_2'],
            'ontology_curie': ['UBERON:0000001', 'UBERON:0000002'],
            'strand': '.',
        }),
        splice_sites=pd.DataFrame({
            'name': ['donor', 'acceptor', 'donor', 'acceptor', 'padding'],
            'strand': ['+', '+', '-', '-', '.'],
        }),
        splice_junctions=pd.DataFrame({
            'name': (
                [f'tissue_{i}' for i in range(self._num_tissues)]
                + ['Padding'] * 2
            ),
            'ontology_curie': ['UBERON:0000001'] * (self._num_tissues + 2),
        }),
        chip_tf=pd.DataFrame({
            'name': ['chip_1', 'chip_2'],
            'strand': ['+', '-'],
            'ontology_curie': 'UBERON:0000001',
            'nonzero_mean': 1.0,
        }),
    )

    def _apply_fn(params, state, dna_sequence, organism_index):
      del params, state, organism_index
      batch_size, sequence_length = dna_sequence.shape[0], dna_sequence.shape[1]
      splice_site_positions = (
          np.ones((batch_size, 4, self._num_splice_sites), dtype=np.int32) * -1
      )
      splice_site_positions[:, 0, 0] = 100  # pos donor
      splice_site_positions[:, 1, 0] = 200  # pos acceptor
      splice_site_positions[:, 2, 0] = 300  # neg donor
      splice_site_positions[:, 3, 0] = 50  # neg acceptor
      return {
          'atac': {
              'predictions_1bp': jnp.zeros(
                  (batch_size, sequence_length, len(self._metadata.atac)),
                  dtype=jnp.bfloat16,
              )
          },
          'dnase': {
              'predictions_1bp': jnp.zeros(
                  (batch_size, sequence_length, len(self._metadata.dnase))
              )
          },
          'contact_maps': {
              'predictions': jnp.zeros((
                  batch_size,
                  sequence_length // 2048,
                  sequence_length // 2048,
                  len(self._metadata.contact_maps),
              ))
          },
          'splice_sites_classification': {
              'predictions': jnp.ones(
                  (
                      batch_size,
                      sequence_length,
                      5,
                  ),
                  dtype=jnp.bfloat16,
              ),
          },
          'splice_sites_junction': {
              'predictions': jnp.ones(
                  (
                      batch_size,
                      self._num_splice_sites,
                      self._num_splice_sites,
                      2 * self._num_tissues + 4,  # Add 4 to mimic padding.
                  ),
                  dtype=jnp.bfloat16,
              ),
              'splice_site_positions': jnp.array(splice_site_positions),
          },
          'embeddings_1bp': jnp.zeros(
              (batch_size, sequence_length, 1536), dtype=jnp.bfloat16
          ),
          'chip_tf': {
              'predictions_128bp': jnp.zeros(
                  (batch_size, sequence_length // 128, 2), dtype=jnp.bfloat16
              )
          },
      }

    self._mock_model = _apply_fn

    def _apply_fn_junctions(
        params, state, trunk_embeddings, splice_site_positions, organism_index
    ):
      del params, state, trunk_embeddings, organism_index
      batch_size, _, num_splice_sites = splice_site_positions.shape
      return {
          'predictions': jnp.ones(
              (
                  batch_size,
                  num_splice_sites,
                  num_splice_sites,
                  2 * self._num_tissues + 4,  # Add 4 to mimic padding.
              ),
              dtype=jnp.bfloat16,
          ),
          'splice_site_positions': jnp.array(splice_site_positions),
      }

    self._mock_model_junctions = _apply_fn_junctions

  @parameterized.parameters([
      dict(organism=dna_model.Organism.HOMO_SAPIENS, expected_index=0),
      dict(organism=dna_model.Organism.MUS_MUSCULUS, expected_index=1),
  ])
  def test_convert_to_organism_index(self, organism, expected_index):
    self.assertEqual(
        dna_model.convert_to_organism_index(organism),
        expected_index,
    )

  @parameterized.parameters([
      dict(
          requested_outputs=[
              dna_output.OutputType.ATAC,
              dna_output.OutputType.DNASE,
          ],
          requested_ontologies=None,
          expected_shapes={
              dna_output.OutputType.ATAC: (65_536, 2),
              dna_output.OutputType.DNASE: (65_536, 1),
          },
      ),
      dict(
          requested_outputs=[
              dna_output.OutputType.ATAC,
              dna_output.OutputType.DNASE,
          ],
          requested_ontologies=[ontology.from_curie('CL:0000001')],
          expected_shapes={
              dna_output.OutputType.ATAC: (65_536, 2),
              dna_output.OutputType.DNASE: (65_536, 0),
          },
      ),
  ])
  def test_predict_sequence(
      self, requested_outputs, requested_ontologies, expected_shapes
  ):
    mock_fasta_extractor = mock.create_autospec(fasta.FastaExtractor)
    mock_fasta_extractor.extract.side_effect = lambda x: 'A' * x.width
    model = dna_model.AlphaGenomeModel(
        params={},
        state={},
        apply_fn=self._mock_model,
        junctions_apply_fn=self._mock_model_junctions,
        metadata={dna_model.Organism.HOMO_SAPIENS: self._metadata},
        fasta_extractors={
            dna_model.Organism.HOMO_SAPIENS: mock_fasta_extractor
        },
        gtfs={dna_model.Organism.HOMO_SAPIENS: _create_mock_gtf()},
        device=jax.local_devices()[0],
    )
    seq = 'ATGC' * int(2**14)
    predictions = model.predict_sequence(
        seq,
        requested_outputs=requested_outputs,
        ontology_terms=requested_ontologies,
    )
    for output_type, expected_shape in expected_shapes.items():
      output = predictions.get(output_type)
      self.assertIsNotNone(output)
      chex.assert_shape(output.values, expected_shape)

  @parameterized.parameters([
      dict(
          requested_outputs=[dna_output.OutputType.ATAC],
          requested_ontologies=None,
          expected_shapes={dna_output.OutputType.ATAC: (2048, 2)},
      ),
      dict(
          requested_outputs=[dna_output.OutputType.DNASE],
          requested_ontologies=[ontology.from_curie('UBERON:0000001')],
          expected_shapes={dna_output.OutputType.DNASE: (2048, 1)},
      ),
  ])
  def test_predict_interval(
      self, requested_outputs, requested_ontologies, expected_shapes
  ):
    mock_fasta_extractor = mock.create_autospec(fasta.FastaExtractor)
    mock_fasta_extractor.extract.side_effect = lambda x: 'A' * x.width
    model = dna_model.AlphaGenomeModel(
        params={},
        state={},
        apply_fn=self._mock_model,
        junctions_apply_fn=self._mock_model_junctions,
        metadata={dna_model.Organism.HOMO_SAPIENS: self._metadata},
        fasta_extractors={
            dna_model.Organism.HOMO_SAPIENS: mock_fasta_extractor
        },
        gtfs={dna_model.Organism.HOMO_SAPIENS: _create_mock_gtf()},
        device=jax.local_devices()[0],
    )
    interval = genome.Interval.from_str('chr1:0-2048:.')
    output = model.predict_interval(
        interval,
        requested_outputs=requested_outputs,
        ontology_terms=requested_ontologies,
    )
    for output_type, expected_shape in expected_shapes.items():
      output = output.get(output_type)
      self.assertIsNotNone(output)
      chex.assert_shape(output.values, expected_shape)

  def test_predict_interval_with_splicing(self):
    mock_fasta_extractor = mock.create_autospec(fasta.FastaExtractor)
    mock_fasta_extractor.extract.side_effect = lambda x: 'A' * x.width
    model = dna_model.AlphaGenomeModel(
        params={},
        state={},
        apply_fn=self._mock_model,
        junctions_apply_fn=self._mock_model_junctions,
        metadata={dna_model.Organism.HOMO_SAPIENS: self._metadata},
        fasta_extractors={
            dna_model.Organism.HOMO_SAPIENS: mock_fasta_extractor
        },
        gtfs={dna_model.Organism.HOMO_SAPIENS: _create_mock_gtf()},
        device=jax.local_devices()[0],
    )
    interval = genome.Interval.from_str('chr1:0-2048:.')
    output = model.predict_interval(
        interval,
        requested_outputs=[
            dna_output.OutputType.ATAC,
            dna_output.OutputType.SPLICE_JUNCTIONS,
        ],
        ontology_terms=None,
    )
    atac_output = output.atac
    self.assertIsNotNone(atac_output)
    chex.assert_shape(atac_output.values, (interval.width, 2))

    splice_junctions_output = output.splice_junctions
    self.assertIsNotNone(splice_junctions_output)
    self.assertLen(splice_junctions_output.junctions, 2)
    chex.assert_shape(splice_junctions_output.values, (2, self._num_tissues))
    self.assertEqual(
        splice_junctions_output.junctions[0],
        genome.Junction('chr1', 101, 200, '+'),
    )
    self.assertEqual(
        splice_junctions_output.junctions[1],
        genome.Junction('chr1', 51, 300, '-'),
    )

  def test_predict_interval_missing_fasta_raises_error(self):
    model = dna_model.AlphaGenomeModel(
        params={},
        state={},
        apply_fn=self._mock_model,
        junctions_apply_fn=self._mock_model_junctions,
        metadata={dna_model.Organism.HOMO_SAPIENS: self._metadata},
        device=jax.local_devices()[0],
    )
    with self.assertRaisesRegex(
        ValueError, "FastaExtractor not found for organism.name='HOMO_SAPIENS'"
    ):
      model.predict_interval(
          genome.Interval.from_str('chr1:0-2048:.'),
          requested_outputs=[dna_output.OutputType.ATAC],
          ontology_terms=None,
      )

  @parameterized.parameters([
      dict(
          requested_outputs=[
              dna_output.OutputType.ATAC,
              dna_output.OutputType.DNASE,
          ],
          requested_ontologies=None,
          expected_shapes={
              dna_output.OutputType.ATAC: (2048, 2),
              dna_output.OutputType.DNASE: (2048, 1),
          },
      ),
      dict(
          requested_outputs=[
              dna_output.OutputType.ATAC,
              dna_output.OutputType.DNASE,
          ],
          requested_ontologies=[ontology.from_curie('CL:0000001')],
          expected_shapes={
              dna_output.OutputType.ATAC: (2048, 2),
              dna_output.OutputType.DNASE: (2048, 0),
          },
      ),
  ])
  def test_predict_variant(
      self, requested_outputs, requested_ontologies, expected_shapes
  ):
    mock_fasta_extractor = mock.create_autospec(fasta.FastaExtractor)
    mock_fasta_extractor.extract.side_effect = lambda x: 'A' * x.width

    mock_splice_sites_extractor = mock.create_autospec(
        splicing.SpliceSiteAnnotationExtractor, instance=True
    )
    mock_splice_sites_extractor.extract.side_effect = lambda x: np.zeros(
        (x.width, 5), dtype=bool
    )

    model = dna_model.AlphaGenomeModel(
        params={},
        state={},
        apply_fn=self._mock_model,
        junctions_apply_fn=self._mock_model_junctions,
        metadata={dna_model.Organism.HOMO_SAPIENS: self._metadata},
        fasta_extractors={
            dna_model.Organism.HOMO_SAPIENS: mock_fasta_extractor
        },
        gtfs={dna_model.Organism.HOMO_SAPIENS: _create_mock_gtf()},
        device=jax.local_devices()[0],
        splice_site_extractors={
            dna_model.Organism.HOMO_SAPIENS: mock_splice_sites_extractor
        },
    )
    interval = genome.Interval.from_str('chr1:0-2048:.')
    predictions = model.predict_variant(
        interval,
        variant=genome.Variant.from_str('chr1:1024:A>C'),
        requested_outputs=requested_outputs,
        ontology_terms=requested_ontologies,
    )
    for output_type, expected_shape in expected_shapes.items():
      output = predictions.reference.get(output_type)
      self.assertIsNotNone(output)
      chex.assert_shape(output.values, expected_shape)
    for output_type, expected_shape in expected_shapes.items():
      output = predictions.alternate.get(output_type)
      self.assertIsNotNone(output)
      chex.assert_shape(output.values, expected_shape)

  def test_score_variant(self):
    mock_fasta_extractor = mock.create_autospec(fasta.FastaExtractor)
    mock_fasta_extractor.extract.side_effect = lambda x: 'A' * x.width

    mock_splice_sites_extractor = mock.create_autospec(
        splicing.SpliceSiteAnnotationExtractor, instance=True
    )
    mock_splice_sites_extractor.extract.side_effect = lambda x: np.zeros(
        (x.width, 5), dtype=bool
    )

    model = dna_model.AlphaGenomeModel(
        params={},
        state={},
        apply_fn=self._mock_model,
        junctions_apply_fn=self._mock_model_junctions,
        metadata={dna_model.Organism.HOMO_SAPIENS: self._metadata},
        fasta_extractors={
            dna_model.Organism.HOMO_SAPIENS: mock_fasta_extractor
        },
        gtfs={dna_model.Organism.HOMO_SAPIENS: _create_mock_gtf()},
        pas_gtfs={dna_model.Organism.HOMO_SAPIENS: _create_polya_df_gtf()},
        device=jax.local_devices()[0],
        splice_site_extractors={
            dna_model.Organism.HOMO_SAPIENS: mock_splice_sites_extractor
        },
    )
    interval = genome.Interval.from_str('chr1:0-2048:.')
    variant = genome.Variant.from_str('chr1:1024:A>C')
    scorers = [
        variant_scorers.CenterMaskScorer(
            aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
            requested_output=dna_output.OutputType.ATAC,
            width=501,
        ),
        variant_scorers.CenterMaskScorer(
            requested_output=dna_output.OutputType.CHIP_TF,
            width=501,
            aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
        ),
        variant_scorers.ContactMapScorer(),
    ]
    output = model.score_variant(interval, variant, variant_scorers=scorers)
    self.assertLen(output, len(scorers))
    for result, scorer in zip(output, scorers):
      self.assertEqual(result.uns['scored_interval'], interval)
      self.assertEqual(result.uns['variant'], variant)
      self.assertEqual(result.uns['variant_scorer'], scorer)

  def test_missing_gtf_raises_scorer_missing_error(self):
    mock_fasta_extractor = mock.create_autospec(fasta.FastaExtractor)
    mock_fasta_extractor.extract.side_effect = lambda x: 'A' * x.width
    model = dna_model.AlphaGenomeModel(
        params={},
        state={},
        apply_fn=self._mock_model,
        junctions_apply_fn=self._mock_model_junctions,
        metadata={dna_model.Organism.HOMO_SAPIENS: self._metadata},
        fasta_extractors={
            dna_model.Organism.HOMO_SAPIENS: mock_fasta_extractor
        },
        device=jax.local_devices()[0],
    )
    interval = genome.Interval.from_str('chr1:0-2048:.')
    variant = genome.Variant.from_str('chr1:1024:A>C')
    scorers = [
        variant_scorers.GeneMaskLFCScorer(
            requested_output=dna_output.OutputType.ATAC
        )
    ]
    with self.assertRaisesRegex(
        ValueError, "Scorer 'BaseVariantScorer.GENE_MASK_LFC' is missing"
    ):
      model.score_variant(interval, variant, variant_scorers=scorers)

  def test_missing_pas_raises_scorer_missing_error(self):
    model = dna_model.AlphaGenomeModel(
        params={},
        state={},
        apply_fn=self._mock_model,
        junctions_apply_fn=self._mock_model_junctions,
        metadata={dna_model.Organism.HOMO_SAPIENS: self._metadata},
        pas_gtfs={dna_model.Organism.HOMO_SAPIENS: _create_polya_df_gtf()},
        device=jax.local_devices()[0],
    )
    interval = genome.Interval.from_str('chr1:0-2048:.')
    variant = genome.Variant.from_str('chr1:1024:A>C')
    scorers = [variant_scorers.PolyadenylationScorer()]
    with self.assertRaisesRegex(
        ValueError, "Scorer 'BaseVariantScorer.PA_QTL' is missing"
    ):
      model.score_variant(interval, variant, variant_scorers=scorers)

  def test_score_interval(self):
    mock_fasta_extractor = mock.create_autospec(fasta.FastaExtractor)
    mock_fasta_extractor.extract.side_effect = lambda x: 'A' * x.width
    model = dna_model.AlphaGenomeModel(
        params={},
        state={},
        apply_fn=self._mock_model,
        junctions_apply_fn=self._mock_model_junctions,
        metadata={dna_model.Organism.HOMO_SAPIENS: self._metadata},
        fasta_extractors={
            dna_model.Organism.HOMO_SAPIENS: mock_fasta_extractor
        },
        gtfs={dna_model.Organism.HOMO_SAPIENS: _create_mock_gtf()},
        pas_gtfs={dna_model.Organism.HOMO_SAPIENS: _create_polya_df_gtf()},
        device=jax.local_devices()[0],
    )
    interval = genome.Interval.from_str('chr1:0-2048:.')
    scorers = [
        interval_scorers.GeneMaskScorer(
            aggregation_type=interval_scorers.IntervalAggregationType.MEAN,
            requested_output=dna_output.OutputType.ATAC,
            width=501,
        )
    ]
    output = model.score_interval(interval, interval_scorers=scorers)
    self.assertLen(output, len(scorers))
    for result, scorer in zip(output, scorers):
      self.assertEqual(result.uns['scored_interval'], interval)
      self.assertEqual(result.uns['interval_scorer'], scorer)

  @parameterized.parameters([
      dict(
          ism_interval=genome.Interval.from_str('chr1:10-11:.'),
          interval_variant=None,
          expected_variants=[
              genome.Variant('chr1', 11, 'A', 'C'),
              genome.Variant('chr1', 11, 'A', 'G'),
              genome.Variant('chr1', 11, 'A', 'T'),
          ],
      ),
      dict(
          ism_interval=genome.Interval.from_str('chr1:10-11:.'),
          interval_variant=genome.Variant.from_str('chr1:11:A>C'),
          expected_variants=[
              genome.Variant('chr1', 11, 'C', 'A'),
              genome.Variant('chr1', 11, 'C', 'G'),
              genome.Variant('chr1', 11, 'C', 'T'),
          ],
      ),
  ])
  def test_score_ism_variants(
      self,
      ism_interval: genome.Interval,
      interval_variant: genome.Variant | None,
      expected_variants: list[genome.Variant],
  ):
    mock_fasta_extractor = mock.create_autospec(fasta.FastaExtractor)
    mock_fasta_extractor.extract.side_effect = lambda x: 'A' * x.width

    mock_splice_sites_extractor = mock.create_autospec(
        splicing.SpliceSiteAnnotationExtractor, instance=True
    )
    mock_splice_sites_extractor.extract.side_effect = lambda x: np.zeros(
        (x.width, 5), dtype=bool
    )

    model = dna_model.AlphaGenomeModel(
        params={},
        state={},
        apply_fn=self._mock_model,
        junctions_apply_fn=self._mock_model_junctions,
        metadata={dna_model.Organism.HOMO_SAPIENS: self._metadata},
        fasta_extractors={
            dna_model.Organism.HOMO_SAPIENS: mock_fasta_extractor
        },
        gtfs={dna_model.Organism.HOMO_SAPIENS: _create_mock_gtf()},
        pas_gtfs={dna_model.Organism.HOMO_SAPIENS: _create_polya_df_gtf()},
        device=jax.local_devices()[0],
        splice_site_extractors={
            dna_model.Organism.HOMO_SAPIENS: mock_splice_sites_extractor
        },
    )
    interval = genome.Interval.from_str('chr1:0-2048:.')

    scores = model.score_ism_variants(
        interval,
        ism_interval,
        interval_variant=interval_variant,
        variant_scorers=[
            variant_scorers.CenterMaskScorer(
                aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
                requested_output=dna_output.OutputType.ATAC,
                width=501,
            )
        ],
    )
    scores = sorted(scores, key=lambda x: str(x[0].uns['variant']))
    self.assertLen(scores, len(expected_variants))
    for expected, scores in zip(expected_variants, scores, strict=True):
      self.assertEqual(expected, scores[0].uns['variant'])

  def test_create(self):
    init_fn, _, _ = dna_model.create_model(
        {dna_model.Organism.HOMO_SAPIENS: self._metadata}
    )
    params, state = jax.jit(init_fn)(
        jax.random.PRNGKey(0),
        jax.ShapeDtypeStruct((1, 2048, 4), dtype=jnp.float32),
        jax.ShapeDtypeStruct((1,), dtype=jnp.int32),
    )
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_dir = os.path.join(self.create_tempdir().full_path, 'ckpt')
    checkpointer.save(checkpoint_dir, (params, state))
    gtf_path = os.path.join(self.create_tempdir().full_path, 'hg38.feather')
    _create_mock_gtf().to_feather(gtf_path)
    polya_gtf_path = os.path.join(
        self.create_tempdir().full_path, 'polya_gtf.feather'
    )
    _create_polya_df_gtf().to_feather(polya_gtf_path)
    splice_starts, splice_ends = _create_mock_splice_sites()
    splice_starts_path = os.path.join(
        self.create_tempdir().full_path, 'splice_starts.feather'
    )
    splice_ends_path = os.path.join(
        self.create_tempdir().full_path, 'splice_ends.feather'
    )
    splice_starts.to_feather(splice_starts_path)
    splice_ends.to_feather(splice_ends_path)
    checkpointer.wait_until_finished()
    model = dna_model.create(
        checkpoint_dir,
        organism_settings={
            dna_model.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(
                fasta_path=_get_test_fasta_path(),
                gtf_feather_path=gtf_path,
                pas_feather_path=polya_gtf_path,
                splice_site_starts_feather_path=splice_starts_path,
                splice_site_ends_feather_path=splice_ends_path,
                metadata=self._metadata,
            )
        },
        device=jax.local_devices()[0],
    )
    self.assertIsInstance(model, dna_model.AlphaGenomeModel)

  def test_default_organism_settings(self):
    organism_settings = dna_model.default_organism_settings()
    self.assertContainsSubset(
        [
            dna_model.Organism.HOMO_SAPIENS,
            dna_model.Organism.MUS_MUSCULUS,
        ],
        organism_settings.keys(),
    )

  def test_create_model(self):
    init, apply, apply_junctions = dna_model.create_model(
        {dna_model.Organism.HOMO_SAPIENS: self._metadata}
    )

    dna_sequence_shape = jax.ShapeDtypeStruct((1, 2048, 4), dtype=jnp.float32)
    organism_index_shape = jax.ShapeDtypeStruct((1,), dtype=jnp.int32)

    params, state = jax.eval_shape(
        init, jax.random.PRNGKey(0), dna_sequence_shape, organism_index_shape
    )

    predictions = jax.eval_shape(
        apply, params, state, dna_sequence_shape, organism_index_shape
    )
    expected_predictions = {
        dna_output.OutputType.ATAC: jax.ShapeDtypeStruct(
            (1, 2048, 2), dtype=jnp.bfloat16
        ),
        dna_output.OutputType.DNASE: jax.ShapeDtypeStruct(
            (1, 2048, 1), dtype=jnp.bfloat16
        ),
        dna_output.OutputType.CHIP_TF: jax.ShapeDtypeStruct(
            (1, 16, 2), dtype=jnp.bfloat16
        ),
        dna_output.OutputType.SPLICE_SITES: jax.ShapeDtypeStruct(
            (1, 2048, 5), dtype=jnp.bfloat16
        ),
        dna_output.OutputType.SPLICE_JUNCTIONS: {
            'predictions': jax.ShapeDtypeStruct(
                (1, 512, 512, 24), dtype=jnp.bfloat16
            ),
            'splice_site_positions': jax.ShapeDtypeStruct(
                (1, 4, 512), dtype=jnp.int32
            ),
        },
        dna_output.OutputType.CONTACT_MAPS: jax.ShapeDtypeStruct(
            (1, 1, 1, 2), dtype=jnp.bfloat16
        ),
    }
    chex.assert_trees_all_equal_shapes_and_dtypes(
        dna_model.extract_predictions(predictions), expected_predictions
    )

    junction_predictions = jax.eval_shape(
        apply_junctions,
        params,
        state,
        predictions['embeddings_1bp'],
        predictions['splice_sites_junction']['splice_site_positions'],
        organism_index_shape,
    )
    expected_junction_predictions = {
        'predictions': jax.ShapeDtypeStruct(
            (1, 512, 512, 24), dtype=jnp.bfloat16
        ),
        'splice_junction_mask': jax.ShapeDtypeStruct(
            (1, 512, 512, 24), dtype=jnp.bool
        ),
        'splice_site_positions': jax.ShapeDtypeStruct(
            (1, 4, 512), dtype=jnp.int32
        ),
    }
    chex.assert_trees_all_equal_shapes_and_dtypes(
        junction_predictions, expected_junction_predictions
    )

  @parameterized.parameters(('all_folds',), (dna_model.ModelVersion.ALL_FOLDS,))
  @mock.patch.object(jax, 'eval_shape', return_value=MOCK_SHAPES, autospec=True)
  def test_create_from_kaggle(self, version, mock_eval_shape):
    del mock_eval_shape
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_dir = os.path.join(self.create_tempdir().full_path, 'ckpt')
    checkpointer.save(checkpoint_dir, ({}, {}))
    checkpointer.wait_until_finished()
    with (
        mock.patch.object(kagglehub, 'model_download') as mock_model_download,
        mock.patch.object(kaggle_auth, 'get_username') as mock_get_username,
        mock.patch.object(kagglehub, 'login') as mock_login,
    ):
      mock_get_username.return_value = None
      mock_model_download.return_value = checkpoint_dir
      model = dna_model.create_from_kaggle(
          version, organism_settings={}, device=jax.local_devices()[0]
      )

    self.assertIsInstance(model, dna_model.AlphaGenomeModel)
    mock_get_username.assert_called_once()
    mock_login.assert_called_once()
    mock_model_download.assert_called_once_with(
        'google/alphagenome/jax/all_folds'
    )

  @parameterized.parameters(('fold_0',), (dna_model.ModelVersion.FOLD_0,))
  @mock.patch.object(jax, 'eval_shape', return_value=MOCK_SHAPES, autospec=True)
  def test_create_from_huggingface(self, version, mock_eval_shape):
    del mock_eval_shape
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_dir = os.path.join(self.create_tempdir().full_path, 'ckpt')
    checkpointer.save(checkpoint_dir, ({}, {}))
    checkpointer.wait_until_finished()
    with (
        mock.patch.object(
            huggingface_hub, 'snapshot_download'
        ) as mock_snapshot_download,
        mock.patch.object(huggingface_hub, 'login') as mock_login,
        mock.patch.object(huggingface_hub, 'whoami') as mock_whoami,
    ):
      mock_snapshot_download.return_value = checkpoint_dir
      mock_whoami.side_effect = huggingface_hub.errors.LocalTokenNotFoundError()
      model = dna_model.create_from_huggingface(
          version, organism_settings={}, device=jax.local_devices()[0]
      )

    self.assertIsInstance(model, dna_model.AlphaGenomeModel)
    mock_login.assert_called_once()
    mock_whoami.assert_called_once()
    mock_snapshot_download.assert_called_once_with(
        repo_id='google/alphagenome-fold-0'
    )

  def test_default_create_with_cpu_raises_error(self):
    with jax.default_device(jax.devices(backend='cpu')[0]):
      with self.assertRaisesRegex(
          ValueError, 'Cannot find any GPU or TPU devices'
      ):
        _ = dna_model.AlphaGenomeModel(
            params={},
            state={},
            apply_fn=self._mock_model,
            junctions_apply_fn=self._mock_model_junctions,
            metadata={},
        )


if __name__ == '__main__':
  absltest.main()
