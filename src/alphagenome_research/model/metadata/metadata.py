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

"""AlphaGenome output metadata."""

from importlib import resources
from collections.abc import Collection, Mapping
import dataclasses
import functools
import pathlib

from alphagenome import typing
from alphagenome.data import ontology
from alphagenome.data import track_data
from alphagenome.models.v1 import dna_client
from alphagenome.models.v1 import dna_model
from alphagenome.models.v1 import dna_output

from alphagenome.protos import dna_model_service_pb2
from google.protobuf import text_format
from jaxtyping import Bool, Int32  # pylint: disable=g-importing-member, g-multiple-import
import numpy as np

_PATH_METADATA = 'model/metadata/'
_PADDING_TRACK_NAME = 'padding'


def _create_output_strand_reindexing(
    metadata: track_data.TrackMetadata,
    num_tracks: int,
) -> Int32[np.ndarray, '_']:
  """Creates the strand reindexing for a given output type's track metadata."""
  strands = metadata['strand'].values
  reindex_array = np.arange(num_tracks, dtype=np.int32)
  positive_indices = np.flatnonzero(strands == '+')
  negative_indices = np.flatnonzero(strands == '-')
  reindex_array[positive_indices] = negative_indices
  reindex_array[negative_indices] = positive_indices
  return reindex_array


@typing.jaxtyped
@dataclasses.dataclass(frozen=True, kw_only=True)
class AlphaGenomeOutputMetadata(dna_output.OutputMetadata):
  """AlphaGenome output metadata."""

  def resolution(self, output: dna_output.OutputType) -> int:
    """Returns the resolution of the given output type."""
    match output:
      case dna_output.OutputType.ATAC:
        return 1
      case dna_output.OutputType.CAGE:
        return 1
      case dna_output.OutputType.DNASE:
        return 1
      case dna_output.OutputType.RNA_SEQ:
        return 1
      case dna_output.OutputType.CHIP_HISTONE:
        return 128
      case dna_output.OutputType.CHIP_TF:
        return 128
      case dna_output.OutputType.SPLICE_SITES:
        return 1
      case dna_output.OutputType.SPLICE_SITE_USAGE:
        return 1
      case dna_output.OutputType.SPLICE_JUNCTIONS:
        return 1
      case dna_output.OutputType.CONTACT_MAPS:
        return 2048
      case dna_output.OutputType.PROCAP:
        return 1
      case _:
        raise ValueError(f'Unknown {output=}')

  @functools.cached_property
  def padding(self) -> Mapping[dna_output.OutputType, Bool[np.ndarray, '_']]:
    """Returns mapping of output type to padding mask."""
    padding = {}
    for output_type in dna_output.OutputType:
      if (metadata := self.get(output_type)) is not None:
        padding[output_type] = (
            metadata['name'].str.lower() == _PADDING_TRACK_NAME
        ).values
    return padding

  @functools.cached_property
  def strand_reindexing(
      self,
  ) -> Mapping[dna_output.OutputType, Int32[np.ndarray, '_']]:
    """Return mapping of output type to negative strand reindexing."""
    result = {}
    for output_type in dna_output.OutputType:
      if (metadata := self.get(output_type)) is not None:
        num_tracks = len(metadata)
        if output_type == dna_output.OutputType.SPLICE_JUNCTIONS:
          # Splice junction metadata is strand agnostic, but predictions are
          # structured as all positive followed by all negative strands.
          result[output_type] = np.concatenate((
              np.arange(num_tracks, num_tracks * 2),
              np.arange(num_tracks),
          ))
        else:
          result[output_type] = _create_output_strand_reindexing(
              metadata, num_tracks
          )
    return result


@typing.jaxtyped
def create_track_masks(
    metadata: AlphaGenomeOutputMetadata,
    *,
    requested_outputs: Collection[dna_output.OutputType],
    requested_ontologies: Collection[ontology.OntologyTerm] | None,
) -> Mapping[dna_output.OutputType, Bool[np.ndarray, '_']]:
  """Creates track masks for the requested output types and ontologies."""
  track_masks = {}
  for output_type in requested_outputs:
    if (output_metadata := metadata.get(output_type)) is None:
      continue

    if (
        requested_ontologies is not None
        # Splice sites are tissue agnostic.
        and output_type != dna_output.OutputType.SPLICE_SITES
    ):
      ontology_curies = {o.ontology_curie for o in requested_ontologies}
      mask = np.asarray(
          [o in ontology_curies for o in output_metadata['ontology_curie']]
      )
    else:
      mask = ~np.array(metadata.padding[output_type])

    track_masks[output_type] = mask

  return track_masks


def load(organism: dna_model.Organism) -> AlphaGenomeOutputMetadata:
  """Loads the metadata for a given organism."""
  file_name = f'OutputMetadataResponse_ORGANISM_{organism.name}.textproto'
  path = pathlib.Path(_PATH_METADATA, file_name)
  content = resources.files('alphagenome_research').joinpath(path).read_text()
  metadata_response = text_format.Parse(
      content, dna_model_service_pb2.MetadataResponse()
  )
  return AlphaGenomeOutputMetadata(
      **vars(dna_client.construct_output_metadata(iter((metadata_response,))))
  )
