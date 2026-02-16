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

from collections.abc import Mapping, Sequence
from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.data import ontology
from alphagenome.models import dna_model
from alphagenome.models import dna_output
from alphagenome_research.model.metadata import metadata as metadata_lib
import chex
import jax
import numpy as np
import pandas as pd


class MetadataTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(organism=dna_model.Organism.HOMO_SAPIENS),
      dict(organism=dna_model.Organism.MUS_MUSCULUS),
  ])
  def test_load(self, organism: dna_model.Organism):
    metadata = metadata_lib.load(organism)
    self.assertIsInstance(metadata, dna_output.OutputMetadata)
    reindexing = metadata.strand_reindexing
    self.assertIsInstance(reindexing, dict)
    self.assertCountEqual(dna_output.OutputType, reindexing.keys())

  def test_organism_shapes_match(self):
    metadata = tuple(metadata_lib.load(o) for o in dna_model.Organism)
    for output_type in dna_output.OutputType:
      with self.subTest(f'{output_type=}'):
        track_metadata = tuple(m.get(output_type).values for m in metadata)
        # Only assert the first track dimension matches, as fields may differ.
        chex.assert_equal_shape_prefix(track_metadata, 1)

  @parameterized.named_parameters([
      dict(
          testcase_name='Normal',
          output_type=dna_output.OutputType.ATAC,
          names=[f'track_{i}' for i in range(6)],
          strands=['+', '+', '+', '-', '-', '-'],
          expected=[3, 4, 5, 0, 1, 2],
      ),
      dict(
          testcase_name='Unstranded',
          output_type=dna_output.OutputType.RNA_SEQ,
          names=[f'track_{i}' for i in range(6)],
          strands=['.', '.', '.', '.', '.', '.'],
          expected=[0, 1, 2, 3, 4, 5],
      ),
      dict(
          testcase_name='Interleaved',
          output_type=dna_output.OutputType.RNA_SEQ,
          names=[f'track_{i}' for i in range(6)],
          strands=['+', '-', '+', '-', '+', '-'],
          expected=[1, 0, 3, 2, 5, 4],
      ),
      dict(
          testcase_name='Padding',
          output_type=dna_output.OutputType.DNASE,
          names=[f'track_{i}' for i in range(6)] + ['Padding', 'padding'],
          strands=['+', '-', '+', '-', '+', '-', '.', '.'],
          expected=[1, 0, 3, 2, 5, 4, 6, 7],
      ),
      dict(
          testcase_name='SpliceJunctions',
          output_type=dna_output.OutputType.SPLICE_JUNCTIONS,
          names=[f'track_{i}' for i in range(3)],
          strands=['.'] * 3,
          expected=[3, 4, 5, 0, 1, 2],
      ),
      dict(
          testcase_name='SpliceJunctionPadding',
          output_type=dna_output.OutputType.SPLICE_JUNCTIONS,
          names=[f'track_{i}' for i in range(3)] + ['PADDING'],
          strands=['.'] * 4,
          expected=[4, 5, 6, 7, 0, 1, 2, 3],
      ),
  ])
  def test_strand_reindexing(
      self,
      output_type: dna_output.OutputType,
      names: Sequence[str],
      strands: Sequence[str],
      expected: Sequence[int],
  ):
    metadata = metadata_lib.AlphaGenomeOutputMetadata(
        **{
            output_type.name.lower(): pd.DataFrame(
                {'name': names, 'strand': strands}
            ),
        },
    )
    result = metadata.strand_reindexing.get(output_type)
    result = result.tolist() if result is not None else None
    self.assertEqual(result, expected)

  @parameterized.named_parameters(
      (
          'Normal',
          dna_output.OutputType.ATAC,
          [f'track_{i}' for i in range(6)],
          ['.'] * 6,
          [False] * 6,
      ),
      (
          'Padding',
          dna_output.OutputType.DNASE,
          [f'track_{i}' for i in range(3)] + ['Padding', 'Padding'],
          ['.'] * 3 + ['.', '.'],
          [False] * 3 + [True, True],
      ),
  )
  def test_padding(
      self,
      output_type: dna_output.OutputType,
      names: Sequence[str],
      strands: Sequence[str],
      expected: Sequence[int],
  ):
    metadata = metadata_lib.AlphaGenomeOutputMetadata(
        **{
            output_type.name.lower(): pd.DataFrame(
                {'name': names, 'strand': strands}
            )
        },
    )
    np.testing.assert_array_equal(metadata.padding.get(output_type), expected)

  @parameterized.named_parameters(
      (
          'SingleOutput',
          [dna_output.OutputType.ATAC],
          None,
          {dna_output.OutputType.ATAC: [True, True, True]},
      ),
      (
          'SingleOntology',
          [dna_output.OutputType.ATAC],
          [ontology.from_curie('CL:0000084')],
          {dna_output.OutputType.ATAC: [True, True, False]},
      ),
      (
          'MultipleOutputsMissingOntology',
          [dna_output.OutputType.ATAC, dna_output.OutputType.DNASE],
          [ontology.from_curie('CL:0000001')],
          {
              dna_output.OutputType.ATAC: [False, False, False],
              dna_output.OutputType.DNASE: [True, True, True, False, False],
          },
      ),
      (
          'SpliceSites',
          [dna_output.OutputType.SPLICE_SITES],
          [ontology.from_curie('CL:0000001')],
          {
              dna_output.OutputType.SPLICE_SITES: [True, True, True],
          },
      ),
      (
          'AllOutputs',
          list(dna_output.OutputType),
          None,
          {
              dna_output.OutputType.ATAC: [True, True, True],
              dna_output.OutputType.DNASE: [True, True, True, False, False],
              dna_output.OutputType.SPLICE_SITES: [True, True, True],
          },
      ),
  )
  def test_create_track_masks(
      self,
      requested_outputs: Sequence[dna_output.OutputType],
      requested_ontologies: Sequence[str] | None,
      expected: Mapping[dna_output.OutputType, ...],
  ):
    example_metadata = metadata_lib.AlphaGenomeOutputMetadata(
        atac=pd.DataFrame({
            'name': ['track_1', 'track_2', 'track_3'],
            'strand': ['+', '-', '.'],
            'ontology_curie': ['CL:0000084', 'CL:0000084', 'UBERONE:0000001'],
        }),
        dnase=pd.DataFrame({
            'name': [f'track_{i}' for i in range(3)] + ['Padding', 'Padding'],
            'strand': ['+'] * 3 + ['.', '.'],
            'ontology_curie': ['CL:0000001'] * 3 + ['', ''],
        }),
        splice_sites=pd.DataFrame({
            'name': [f'track_{i}' for i in range(3)],
            'strand': ['.'] * 3,
        }),
    )
    track_masks = metadata_lib.create_track_masks(
        example_metadata,
        requested_outputs=requested_outputs,
        requested_ontologies=requested_ontologies,
    )
    jax.tree.map(np.testing.assert_array_equal, track_masks, expected)


if __name__ == '__main__':
  absltest.main()
