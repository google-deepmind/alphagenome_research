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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from alphagenome.models.v1 import dna_output
from alphagenome.models.v1 import track_data_utils
from alphagenome.models.v1 import variant_scorers
from alphagenome_research.model.variant_scoring.calibration import calibration
from alphagenome_research.protos import calibration_scores_pb2
import anndata
from etils import epath
import numpy as np
import pandas as pd


class CalibrationScoresTest(parameterized.TestCase):

  @parameterized.product(
      validate=[True, False],
      break_quantile_ties=[True, False],
  )
  def test_quantile_scores(self, validate, break_quantile_ties):
    num_quantiles = 5

    variant_scorer = variant_scorers.CenterMaskScorer(
        requested_output=dna_output.OutputType.ATAC,
        width=501,
        aggregation_type=variant_scorers.AggregationType.DIFF_LOG2_SUM,
    )

    metadata = pd.DataFrame(
        {
            'name': ['track_1', 'track_1', 'track_2'],
            'strand': ['+', '-', '.'],
            'ontology_curie': 'CL:0000001',
        },
        index=['0', '1', '2'],
    )

    expected_quantiles = np.array(
        [
            np.linspace(0, 100, num_quantiles),
            np.linspace(0, 10, num_quantiles),
            np.linspace(0, 20, num_quantiles),
        ],
        dtype=np.float32,
    )

    proto = calibration_scores_pb2.CalibrationScores(
        scorer_to_calibration={
            variant_scorer.name: (
                calibration_scores_pb2.VariantScorerCalibration(
                    quantiles=expected_quantiles.flatten().tolist(),
                    quantile_probabilities=np.linspace(
                        0, 1, num_quantiles
                    ).tolist(),
                    tracks_metadata=track_data_utils.metadata_to_proto(
                        metadata
                    ),
                )
            )
        }
    )
    path = epath.Path(self.create_tempdir().full_path, 'scores.pb')
    path.write_bytes(proto.SerializeToString())

    calibration_scorer = calibration.load(str(path))

    self.assertTrue(calibration_scorer.has_variant_scorer(variant_scorer.name))
    pd.testing.assert_frame_equal(
        metadata, calibration_scorer.scorer_metadata(variant_scorer.name)
    )
    np.testing.assert_equal(
        expected_quantiles,
        calibration_scorer.quantile_values(variant_scorer.name),
    )

    expected = np.array([[0.25, 0.75, np.nan]])
    scores = anndata.AnnData(np.array([[24.0, 6.0, np.nan]]), var=metadata)
    quantiles = calibration_scorer.quantile_scores(
        variant_scorer.name,
        scores,
        validate=validate,
        break_quantile_ties=break_quantile_ties,
    )
    np.testing.assert_equal(quantiles, expected)
    np.testing.assert_equal(
        calibration_scorer.quantile_probabilities(variant_scorer.name),
        np.linspace(0, 1, num_quantiles),
    )

  def test_duplicate_quantiles(self):
    num_quantiles = 101
    num_duplicates = 10

    values = np.linspace(0, 100, num_quantiles, dtype=np.float32)
    idx = num_quantiles // 2
    values[idx : idx + num_duplicates] = values[idx]

    metadata = pd.DataFrame(
        {'name': ['track_1'], 'strand': ['.'], 'ontology_curie': ['']},
        index=['0'],
    )
    proto = calibration_scores_pb2.CalibrationScores(
        scorer_to_calibration={
            'scorer_1': calibration_scores_pb2.VariantScorerCalibration(
                quantiles=values.tolist(),
                quantile_probabilities=np.linspace(
                    0, 1, num_quantiles, dtype=np.float32
                ).tolist(),
                tracks_metadata=track_data_utils.metadata_to_proto(metadata),
            )
        }
    )

    path = epath.Path(self.create_tempdir().full_path, 'scores.pb')
    path.write_bytes(proto.SerializeToString())

    rng = mock.create_autospec(np.random.Generator, instance=True)

    # Always return the last index.
    last_duplicate_index = idx + num_duplicates
    rng.integers.return_value = last_duplicate_index
    calibration_scores = calibration.load(str(path), rng=rng)

    results = calibration_scores.quantile_scores(
        'scorer_1',
        anndata.AnnData(np.array([[50.0]]), var=metadata),
        validate=False,
        break_quantile_ties=True,
    )
    expected = np.array([[0.6]], dtype=np.float32)

    np.testing.assert_equal(results, expected)
    rng.integers.assert_called_once_with(
        idx, idx + num_duplicates, endpoint=True
    )

  def test_duplicate_quantiles_with_seed(self):
    num_quantiles = 101
    num_duplicates = 10

    values = np.linspace(0, 100, num_quantiles)
    idx = num_quantiles // 2
    values[idx : idx + num_duplicates] = values[idx]
    metadata = pd.DataFrame(
        {'name': ['track_1'], 'strand': ['.'], 'ontology_curie': ['']},
        index=['0'],
    )

    proto = calibration_scores_pb2.CalibrationScores(
        scorer_to_calibration={
            'scorer_1': calibration_scores_pb2.VariantScorerCalibration(
                quantiles=values.tolist(),
                quantile_probabilities=np.linspace(
                    0, 1, num_quantiles
                ).tolist(),
                tracks_metadata=track_data_utils.metadata_to_proto(metadata),
            )
        }
    )

    path = epath.Path(self.create_tempdir().full_path, 'scores.pb')
    path.write_bytes(proto.SerializeToString())

    # Always return the last index.
    calibration_scores = calibration.load(str(path))

    results_1 = calibration_scores.quantile_scores(
        'scorer_1',
        anndata.AnnData(np.array([[50.0]]), var=metadata),
        validate=False,
        break_quantile_ties=True,
        seed=123,
    )
    results_2 = calibration_scores.quantile_scores(
        'scorer_1',
        anndata.AnnData(np.array([[50.0]]), var=metadata),
        validate=False,
        break_quantile_ties=True,
        seed=123,
    )

    np.testing.assert_equal(results_1, results_2)

  @parameterized.named_parameters([
      [
          'MissingColumns',
          pd.DataFrame({'name': ['track_1']}, index=['0']),
          ValueError,
          'Variant scorer metadata must contain "name" and "strand" columns',
      ],
      [
          'WrongStrand',
          pd.DataFrame(
              {
                  'name': ['track_1'],
                  'strand': ['.'],
                  'ontology_curie': ['CL:0000001'],
              },
              index=['0'],
          ),
          ValueError,
          'Variant scorer ".*" metadata does not match calibration scores',
      ],
      [
          'WrongOntologyCurie',
          pd.DataFrame(
              {
                  'name': ['track_1'],
                  'strand': ['+'],
                  'ontology_curie': ['UBERON:0002149'],
              },
              index=['0'],
          ),
          ValueError,
          'Variant scorer ".*" metadata does not match calibration scores',
      ],
  ])
  def test_quantile_scores_mismatching_metadata_raises(
      self, metadata, expected_error, expected_msg
  ):
    proto = calibration_scores_pb2.CalibrationScores(
        scorer_to_calibration={
            'scorer_1': calibration_scores_pb2.VariantScorerCalibration(
                quantiles=np.linspace(0, 1, 5).tolist(),
                quantile_probabilities=np.linspace(0, 10, 5).tolist(),
                tracks_metadata=track_data_utils.metadata_to_proto(
                    pd.DataFrame({
                        'name': ['track_1'],
                        'strand': ['+'],
                        'ontology_curie': ['CL:0000001'],
                    })
                ),
            )
        }
    )
    path = epath.Path(self.create_tempdir().full_path, 'scores.pb')
    path.write_bytes(proto.SerializeToString())
    calibration_scores = calibration.load(str(path))

    with self.assertRaisesRegex(expected_error, expected_msg):
      calibration_scores.quantile_scores(
          'scorer_1',
          anndata.AnnData(
              np.array([[
                  1.0,
              ]]),
              var=metadata,
          ),
      )

  def test_quantile_scores_view_raises(self):
    metadata = pd.DataFrame(
        {
            'name': ['track_1', 'track_1'],
            'strand': ['+', '-'],
            'ontology_curie': '',
        },
        index=['0', '1'],
    )
    proto = calibration_scores_pb2.CalibrationScores(
        scorer_to_calibration={
            'scorer_1': calibration_scores_pb2.VariantScorerCalibration(
                quantiles=np.array(
                    [np.linspace(0, 100, 5), np.linspace(0, 100, 5)],
                    dtype=np.float32,
                )
                .flatten()
                .tolist(),
                quantile_probabilities=np.linspace(0, 10, 5).tolist(),
                tracks_metadata=track_data_utils.metadata_to_proto(metadata),
            )
        }
    )
    path = epath.Path(self.create_tempdir().full_path, 'scores.pb')
    path.write_bytes(proto.SerializeToString())
    calibration_scores = calibration.load(str(path))

    scores = anndata.AnnData(np.zeros((20, 2)), var=metadata)
    view = scores[:10, :]
    self.assertTrue(view.is_view)

    with self.assertRaisesRegex(
        ValueError, "Quantile scores doesn't support AnnData views"
    ):
      _ = calibration_scores.quantile_scores('scorer_1', view)

  def test_quantile_scores_out_of_bounds_clips(self):
    metadata = pd.DataFrame(
        {'name': ['track_1'], 'strand': '.', 'ontology_curie': ''},
        index=['0'],
    )
    proto = calibration_scores_pb2.CalibrationScores(
        scorer_to_calibration={
            'scorer_1': calibration_scores_pb2.VariantScorerCalibration(
                quantiles=np.array(
                    [np.linspace(0, 10, 5)],
                    dtype=np.float32,
                )
                .flatten()
                .tolist(),
                quantile_probabilities=np.linspace(0, 1, 5).tolist(),
                tracks_metadata=track_data_utils.metadata_to_proto(metadata),
            )
        }
    )
    path = epath.Path(self.create_tempdir().full_path, 'scores.pb')
    path.write_bytes(proto.SerializeToString())

    calibration_scores = calibration.load(str(path))
    quantiles = calibration_scores.quantile_scores(
        'scorer_1', anndata.AnnData(np.array([[1_000_000.0]])), validate=False
    )
    np.testing.assert_equal(quantiles, 1.0)

  def test_invalid_variant_scorer_raises(self):
    proto = calibration_scores_pb2.CalibrationScores(
        scorer_to_calibration={
            'scorer_1': calibration_scores_pb2.VariantScorerCalibration(
                quantiles=[0.0, 1.0],
                quantile_probabilities=[0.0, 1.0],
                tracks_metadata=track_data_utils.metadata_to_proto(
                    pd.DataFrame({
                        'name': ['track_1'],
                        'strand': '.',
                        'ontology_curie': '',
                    })
                ),
            )
        }
    )

    path = epath.Path(self.create_tempdir().full_path, 'scores.pb')
    path.write_bytes(proto.SerializeToString())

    calibration_scores = calibration.load(str(path))

    with self.assertRaisesRegex(
        ValueError, 'No calibration data found for variant scorer'
    ):
      calibration_scores.quantile_scores(
          'invalid_scorer', anndata.AnnData(np.zeros((1, 1)))
      )


if __name__ == '__main__':
  absltest.main()
