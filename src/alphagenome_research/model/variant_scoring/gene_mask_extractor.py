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

"""Gene mask extractor."""

import abc
import collections
import dataclasses
import enum
from typing import Sequence

from alphagenome import typing
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from jaxtyping import Array, Bool  # pylint: disable=g-multiple-import, g-importing-member
import numpy as np
import pandas as pd


class GeneMaskType(enum.Enum):
  """Type of gene mask."""

  BODY = enum.auto()
  EXONS = enum.auto()


class GeneQueryType(enum.Enum):
  """Type of gene query."""

  VARIANT_OVERLAPPING = enum.auto()
  INTERVAL_CONTAINED = enum.auto()


class GeneMaskExtractor:
  """Mask extractor yielding tuple (mask, metadata)."""

  def __init__(
      self,
      gtf: pd.DataFrame,
      gene_mask_type: GeneMaskType,
      *,
      gene_query_type: GeneQueryType = GeneQueryType.INTERVAL_CONTAINED,
      filter_protein_coding: bool = False,
      cache_size: int = 2,
  ):
    self._gene_mask_type = gene_mask_type
    self._gene_query_type = gene_query_type
    self._filter_protein_coding = filter_protein_coding
    self._lru_cache = collections.OrderedDict()
    self._cache_size = cache_size
    if self._filter_protein_coding:
      # Use protein-coding annotations at gene level for gene body masks,
      # and at transcript level for exon/TSS masks.
      gtf = gene_annotation.filter_protein_coding(
          gtf, include_gene_entries=(gene_mask_type == GeneMaskType.BODY)
      )

    match self._gene_mask_type:
      case GeneMaskType.BODY:
        self._mask_extractor = _GeneBodyAnnotationExtractor(
            gtf,
            gene_query_type=self._gene_query_type,
        )
      case GeneMaskType.EXONS:
        if self._gene_query_type != GeneQueryType.INTERVAL_CONTAINED:
          raise ValueError(
              'Exon masks only support INTERVAL_CONTAINED query type.'
          )
        self._mask_extractor = _GeneExonAnnotationExtractor(gtf)
      case _:
        raise ValueError(f'Unknown gene mask type: {self._gene_mask_type}')

  @typing.jaxtyped
  def extract(
      self,
      interval: genome.Interval,
      variant: genome.Variant | None = None,
      transcript_ids: Sequence[str] | None = None,
  ) -> tuple[Bool[np.ndarray, 'S G'], pd.DataFrame]:
    """Extracts gene masks and metadata for a given interval."""
    key = hash((str(interval), str(variant), *(transcript_ids or [])))
    if key in self._lru_cache:
      mask, metadata = self._lru_cache[key]
      self._lru_cache.move_to_end(key)
      return mask, metadata.copy()
    else:
      mask, annotations = self._mask_extractor.extract(
          interval=interval, variant=variant, transcript_ids=transcript_ids
      )
      mask.setflags(write=False)
      self._lru_cache[key] = (mask, annotations.get_metadata())
      while len(self._lru_cache) > self._cache_size:
        self._lru_cache.popitem(last=False)
      return mask, annotations.get_metadata()


@dataclasses.dataclass(frozen=True)
class _GeneAnnotation:
  """Gene annotation."""

  gene_id: Sequence[str]
  gene_name: Sequence[str]
  gene_type: Sequence[str]
  strand: Sequence[str]
  interval_start: Sequence[int]
  chromosome: Sequence[str]
  start: Sequence[int]
  end: Sequence[int]

  def get_metadata(self) -> pd.DataFrame:
    """Returns the metadata as a DataFrame."""
    return pd.DataFrame({
        'gene_id': self.gene_id,
        'Strand': self.strand,
        'gene_name': self.gene_name,
        'gene_type': self.gene_type,
        'interval_start': self.interval_start,
        'Chromosome': self.chromosome,
        'Start': self.start,
        'End': self.end,
    }).reset_index(drop=True)


def _get_empty_metadata_table() -> pd.DataFrame:
  """Returns an empty metadata table."""
  return pd.DataFrame(
      columns=[
          'gene_id',
          'Strand',
          'gene_name',
          'gene_type',
          'interval_start',
          'Chromosome',
          'Start',
          'End',
      ]
  )


class _GeneAnnotationExtractor(abc.ABC):
  """Base class for gene annotation extractors."""

  @property
  def bin_size(self) -> int:
    return 1

  @abc.abstractmethod
  def extract(
      self,
      *,
      interval: genome.Interval,
      variant: genome.Variant | None = None,
      transcript_ids: Sequence[str] | None = None,
  ) -> tuple[Bool[np.ndarray, 'S G'], _GeneAnnotation]:
    """Returns a list of gene annotations for the given interval or variant."""
    raise NotImplementedError()


class _GeneBodyAnnotationExtractor(_GeneAnnotationExtractor):
  """Extracts gene annotations for a given interval."""

  _COLUMNS = [
      'Chromosome',
      'Start',
      'End',
      'Strand',
      'gene_id',
      'gene_name',
      'gene_type',
  ]

  def __init__(
      self,
      gtf: pd.DataFrame,
      gene_query_type: GeneQueryType,
  ):
    """Init.

    Args:
      gtf: GTF DataFrame.
      gene_query_type: Type of gene query.
    """
    self._gene_mask_extractor = _GeneMaskExtractor(gtf)
    self._gene_query_type = gene_query_type
    self._df_gene_gtf = gtf[gtf.Feature == 'gene'][self._COLUMNS]
    self._df_start_end = {
        chromosome: (dfc, dfc['Start'].values, dfc['End'].values)
        for chromosome, dfc in self._df_gene_gtf.groupby(
            'Chromosome', observed=False
        )
    }
    self._df_empty = self._df_gene_gtf.iloc[:0]

  def _extract_interval_contained(
      self, interval: genome.Interval
  ) -> pd.DataFrame:
    """Extracts genes contained by the given interval."""
    if interval.chromosome not in self._df_start_end:
      return self._df_empty
    else:
      dfc, start, end = self._df_start_end[interval.chromosome]
      return dfc[(start >= interval.start) & (end <= interval.end)]

  def _extract_variant_overlapping(
      self, variant: genome.Variant
  ) -> pd.DataFrame:
    """Extracts genes overlapping the given variant."""
    if variant.chromosome not in self._df_start_end:
      return self._df_empty
    else:
      dfc, start, end = self._df_start_end[variant.chromosome]
      variant_end = max(
          variant.end, variant.start + len(variant.alternate_bases)
      )
      return dfc[(end > variant.start) & (start < variant_end)]

  def extract(
      self,
      *,
      interval: genome.Interval,
      variant: genome.Variant | None = None,
      transcript_ids: Sequence[str] | None = None,
  ) -> tuple[Bool[np.ndarray, 'S G'], _GeneAnnotation]:
    """Returns a list of gene annotations for the given interval.

    Args:
      interval: Interval to extract at.
      variant: Variant to extract at.
      transcript_ids: Not supported and should not be provided.
    """
    if transcript_ids is not None:
      raise ValueError('transcript_ids not supported for gene body extractor.')

    match self._gene_query_type:
      case GeneQueryType.VARIANT_OVERLAPPING:
        if variant is None:
          raise ValueError('No variant provided for VARIANT_OVERLAPPING query.')
        gene_subset = self._extract_variant_overlapping(variant)
      case GeneQueryType.INTERVAL_CONTAINED:
        gene_subset = self._extract_interval_contained(interval)
      case _:
        raise ValueError(f'Unknown gene query type: {self._gene_query_type}')

    mask = np.empty((interval.width, len(gene_subset)), dtype=bool)

    for i, row in enumerate(gene_subset.itertuples()):
      mask[:, i] = self._gene_mask_extractor.extract(interval, row.gene_id)

    annotations = _GeneAnnotation(
        gene_id=gene_subset.gene_id,
        gene_name=gene_subset.gene_name,
        gene_type=gene_subset.gene_type,
        strand=gene_subset.Strand,
        interval_start=[interval.start] * len(gene_subset),
        chromosome=gene_subset.Chromosome,
        start=gene_subset.Start,
        end=gene_subset.End,
    )
    return mask, annotations


class _GeneMaskExtractor:
  """Generates binary masks for genes."""

  def __init__(self, gtf: pd.DataFrame):
    """Init.

    Args:
      gtf: GTF DataFrame.
    """
    self._gtf = gtf
    self._genes_by_gene_id = gtf[gtf.Feature == 'gene'][
        ['Chromosome', 'Start', 'End', 'Strand', 'gene_id']
    ].groupby('gene_id', sort=False)

  def extract(
      self,
      interval: genome.Interval,
      gene_id: str,
  ) -> Bool[Array | np.ndarray, 'S']:
    """Extracts gene masks for a specific gene.

    Args:
      interval: Interval to extract at.
      gene_id: Gene ID to extract mask for.

    Returns:
      Boolean mask of shape (interval.width, 2).
    """
    if gene_id not in self._gtf['gene_id'].values:
      raise ValueError(f'Gene ID {gene_id} not found in GTF.')

    genes = self._genes_by_gene_id.get_group(gene_id)
    intervals = [
        genome.Interval(chr, start, end, strand)
        for chr, start, end, strand in zip(
            genes.Chromosome, genes.Start, genes.End, genes.Strand
        )
    ]
    mask = np.zeros((interval.width,), dtype=bool)
    for gene_interval in intervals:
      if interval.overlaps(gene_interval):
        relative_start = max(gene_interval.start - interval.start, 0)
        relative_end = min(gene_interval.end - interval.start, interval.width)
        mask[relative_start:relative_end] = True
    return mask


class _GeneExonAnnotationExtractor(_GeneAnnotationExtractor):
  """Generates binary masks of overlapping exons."""

  _GENE_COLUMNS = [
      'Chromosome',
      'Start',
      'End',
      'Strand',
      'gene_id',
      'gene_name',
      'gene_type',
  ]

  def __init__(self, gtf: pd.DataFrame):
    """Init.

    Args:
      gtf: GTF DataFrame.
    """
    self._exon_mask_extractor = _ExonMaskExtractor(gtf)

    # We use transcript's TSS to determine which transcripts to annotate.
    self._tss = _PositionExtractor(
        gene_annotation.extract_tss(gtf), position_column='Start'
    )
    self._gtf = gtf
    self._gene_df = gtf[gtf.Feature == 'gene'][self._GENE_COLUMNS].set_index(
        'gene_id'
    )

  def extract(
      self,
      *,
      interval: genome.Interval,
      variant: genome.Variant | None = None,
      transcript_ids: Sequence[str] | None = None,
  ) -> tuple[Bool[np.ndarray, 'S G'], _GeneAnnotation]:
    """Extracts exon masks.

    Args:
      interval: Interval to extract at.
      variant: Variant to extract at.
      transcript_ids: Optional list of transcript ids to extract.

    Returns:
      A list of gene annotations for the given interval.
    """
    if transcript_ids is not None:
      # Contains start/end positions for the transcripts.
      transcript_subset = self._gtf[
          self._gtf.transcript_id.isin(transcript_ids)
      ]
    else:
      # Contains TSS positions for the transcripts.
      transcript_subset = self._tss.extract(interval)

    # Mask for each gene is the OR of the masks for all its exons.
    gene_masks = {}

    for row in transcript_subset.itertuples():
      transcript_id = row.transcript_id
      gene_id = row.gene_id

      exon_mask = self._exon_mask_extractor.extract(interval, transcript_id)
      if (gene_mask := gene_masks.get(gene_id)) is not None:
        gene_mask |= exon_mask
      else:
        gene_masks[gene_id] = exon_mask

    unique_gene_ids = list(transcript_subset['gene_id'].unique())
    gene_metadata = self._gene_df.loc[unique_gene_ids]
    mask = np.empty((interval.width, len(unique_gene_ids)), dtype=bool)
    for i, gene_id in enumerate(unique_gene_ids):
      mask[:, i] = gene_masks[gene_id]

    annotations = _GeneAnnotation(
        gene_id=unique_gene_ids,
        gene_name=gene_metadata.gene_name,
        gene_type=gene_metadata.gene_type,
        strand=gene_metadata.Strand,
        interval_start=[interval.start] * len(unique_gene_ids),
        chromosome=gene_metadata.Chromosome,
        start=gene_metadata.Start,
        end=gene_metadata.End,
    )
    return mask, annotations


class _PositionExtractor:
  """Extractor focused on single position, rather than an interval.

  This extractor can be up to 20x faster than PyRangesExtractor.

  Interval is considered semi-open [start, end). Rows where
  - chromosome == interval.chromosome
  - position >= interval.start
  - position < interval.end
  are returned.

  Note: This code doesn't consider the stand information.
  """

  def __init__(
      self,
      df: pd.DataFrame,
      position_column: str,
      chromosome_column: str = 'Chromosome',
  ):
    """Init.

    Args:
      df: dataframe to query with `position_column` and `chromosome_column`.
      position_column: Which column in df to use as 0-based position.
      chromosome_column: Which column in df to use as chromosome.
    """
    self._df_position = {
        chromosome: (dfc, dfc[position_column].values)
        for chromosome, dfc in df.groupby(chromosome_column, observed=False)
    }

    self._df_empty = df.iloc[:0]

  def extract(self, interval: genome.Interval) -> pd.DataFrame:
    if interval.chromosome not in self._df_position:
      return self._df_empty
    else:
      dfc, position = self._df_position[interval.chromosome]
      return dfc[(position >= interval.start) & (position < interval.end)]


class _ExonMaskExtractor:
  """Generates binary masks for exons."""

  def __init__(self, gtf: pd.DataFrame):
    """Init.

    Args:
      gtf: GTF DataFrame.
    """
    self._exon_extractor = _ExonExtractor(gtf)

  def extract(
      self,
      interval: genome.Interval,
      transcript_id: str,
  ) -> Bool[Array | np.ndarray, 'S']:
    """Extracts exon masks for a single transcript.

    Args:
      interval: Interval to extract at.
      transcript_id: The transcript to extract exons for.

    Returns:
      Boolean mask of shape (interval.width, 2).
    """
    exons = self._exon_extractor.extract(transcript_id)
    mask = np.zeros((interval.width,), dtype=bool)
    for exon in exons:
      if interval.overlaps(exon):
        relative_start = max(exon.start - interval.start, 0)
        relative_end = min(exon.end - interval.start, interval.width)
        mask[relative_start:relative_end] = True
    return mask


class _ExonExtractor:
  """Extracts exons for a single transcript."""

  def __init__(self, gtf: pd.DataFrame):
    """Init.

    Args:
      gtf: GTF DataFrame.
    """
    self._gtf = gtf
    self._exons_by_transcript_id = gtf[gtf.Feature == 'exon'][
        ['Chromosome', 'Start', 'End', 'Strand', 'transcript_id']
    ].groupby('transcript_id', sort=False)

  def extract(
      self,
      transcript_id: str,
  ) -> list[genome.Interval]:
    """Extracts exons as List[genome.Interval] for a single transcript.

    Args:
      transcript_id: The transcript to extract exons for.

    Returns:
      List of exon intervals for the transcript.
    """
    try:
      exons = self._exons_by_transcript_id.get_group(transcript_id)
      return [
          genome.Interval(chr, start, end, strand)
          for chr, start, end, strand in zip(
              exons.Chromosome, exons.Start, exons.End, exons.Strand
          )
      ]
    except KeyError as e:
      raise ValueError(
          f'Transcript ID {transcript_id} not found in GTF.'
      ) from e
