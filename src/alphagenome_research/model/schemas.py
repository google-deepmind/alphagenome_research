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

"""Common schemas for the model."""

from alphagenome import typing
from alphagenome_research.io import bundles
import chex
from jaxtyping import ArrayLike, Bool, Float, Int  # pylint: disable=g-importing-member, g-multiple-import


@typing.jaxtyped
@chex.dataclass(frozen=True)
class DataBatch:
  """Input batch for the model."""

  dna_sequence: Float[ArrayLike, 'B S_DNA 4'] | None = None
  organism_index: Int[ArrayLike, 'B'] | None = None
  atac: Float[ArrayLike, 'B S C_ATAC'] | None = None
  atac_mask: Bool[ArrayLike, 'B #S C_ATAC'] | None = None
  dnase: Float[ArrayLike, 'B S C_DNASE'] | None = None
  dnase_mask: Bool[ArrayLike, 'B #S C_DNASE'] | None = None
  procap: Float[ArrayLike, 'B S C_PROCAP'] | None = None
  procap_mask: Bool[ArrayLike, 'B #S C_PROCAP'] | None = None
  chip_histone: Float[ArrayLike, 'B S//128 C_CHIP_HISTONE'] | None = None
  chip_histone_mask: Bool[ArrayLike, 'B #S//128 C_CHIP_HISTONE'] | None = None
  chip_tf: Float[ArrayLike, 'B S//128 C_CHIP_TF'] | None = None
  chip_tf_mask: Bool[ArrayLike, 'B #S//128 C_CHIP_TF'] | None = None
  rna_seq: Float[ArrayLike, 'B S C_RNA_SEQ'] | None = None
  rna_seq_mask: Bool[ArrayLike, 'B #S C_RNA_SEQ'] | None = None
  rna_seq_strand: Int[ArrayLike, 'B 1 C_RNA_SEQ'] | None = None
  cage: Float[ArrayLike, 'B S C_CAGE'] | None = None
  cage_mask: Bool[ArrayLike, 'B #S C_CAGE'] | None = None
  contact_maps: (
      Float[ArrayLike, 'B S_DNA//2048 S_DNA//2048 C_CONTACT_MAPS'] | None
  ) = None
  splice_junctions: Float[ArrayLike, 'B P P C_SPLICE_JUNCTIONS'] | None = None
  splice_site_positions: Int[ArrayLike, 'B 4 P'] | None = None
  splice_site_usage: Float[ArrayLike, 'B S C_SPLICE_SITE_USAGE'] | None = None
  splice_sites: Bool[ArrayLike, 'B S C_SPLICE_SITES'] | None = None
  gene_mask: Bool[ArrayLike, 'B S 2 G'] | None = None

  def get_organism_index(self) -> Int[ArrayLike, 'B']:
    """Returns the organism index data."""
    if self.organism_index is None:
      raise ValueError('Organism index data is not present in the batch.')
    return self.organism_index

  def get_genome_tracks(
      self, bundle: bundles.BundleName
  ) -> tuple[Float[ArrayLike, 'B S C'], Bool[ArrayLike, 'B #S C']]:
    """Returns the genome tracks data for the given bundle if present."""
    match bundle:
      case bundles.BundleName.ATAC:
        data, mask = self.atac, self.atac_mask
      case bundles.BundleName.DNASE:
        data, mask = self.dnase, self.dnase_mask
      case bundles.BundleName.PROCAP:
        data, mask = self.procap, self.procap_mask
      case bundles.BundleName.CAGE:
        data, mask = self.cage, self.cage_mask
      case bundles.BundleName.RNA_SEQ:
        data, mask = self.rna_seq, self.rna_seq_mask
      case bundles.BundleName.CHIP_TF:
        data, mask = self.chip_tf, self.chip_tf_mask
      case bundles.BundleName.CHIP_HISTONE:
        data, mask = self.chip_histone, self.chip_histone_mask
      case _:
        raise ValueError(
            f'Unknown bundle name: {bundle!r}. Is it a genome tracks bundle?'
        )

    if data is None or mask is None:
      raise ValueError(f'{bundle.name!r} data is not present in the batch.')
    return data, mask
