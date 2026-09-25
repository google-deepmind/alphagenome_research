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

"""Extractors for working with splicing data."""

from alphagenome.data import genome
import numpy as np
import pandas as pd


class PositionExtractor:
  """Extractor focused on single position, rather than an interval.

  Interval is considered semi-open [start, end) rows where:
  - chromosome == interval.chromosome,
  - position >= interval.start,
  - position < interval.end,
  are returned.

  Note: This code doesn't consider stand information.
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
      position_column: Which column in the dataframe to use as 0-based position.
      chromosome_column: Which column in the dataframe to use as chromosome.
    """
    self._positions = {
        chromosome: (values, values[position_column].values)
        for chromosome, values in df.groupby(chromosome_column, observed=False)
    }

    self._empty = df.iloc[:0]

  def extract(self, interval: genome.Interval) -> pd.DataFrame:
    if interval.chromosome not in self._positions:
      return self._empty
    else:
      dfc, position = self._positions[interval.chromosome]
      return dfc[(position >= interval.start) & (position < interval.end)]


class SpliceSiteAnnotationExtractor:
  """Generate binary masks of overlapping splice sites."""

  def __init__(
      self,
      junction_starts: pd.DataFrame,
      junction_ends: pd.DataFrame,
  ):
    """Init.

    Args:
      junction_starts: DataFrame of splice junctions starts with columns:
        Chromosome, Start, Strand and an optional set of tissue columns. The
        `Start` column must be 0-based position of the first nucleotide of the
        intron.
      junction_ends: DataFrame of splice junctions ends with columns:
        Chromosome, End, Strand and an optional set of tissue columns. The `End`
        column must be the 1-based position of the last nucleotide of the
        intron.
    """
    self._tissues = [
        c for c in junction_starts if c not in ['Chromosome', 'Strand', 'Start']
    ]
    ends_tissues = [
        c for c in junction_ends if c not in ['Chromosome', 'Strand', 'End']
    ]
    if self._tissues != ends_tissues:
      raise ValueError('Tissues mismatch:', self._tissues, ends_tissues)

    # Junctions describe intron intervals as [Start, End). Start point to the
    # first nucleotide of the intron and End point to the fist nucleotide of the
    # the next exon. We label DONOR the last nucleotide of the exon before the
    # intron (so we move Start back by 1) and ACCEPTOR the first nucleotide of
    # the exon after the intron (so End are not moved).
    junction_starts.update(junction_starts['Start'] - 1)

    self._start_position_extractor = PositionExtractor(junction_starts, 'Start')
    self._end_position_extractor = PositionExtractor(junction_ends, 'End')

  def extract(self, interval: genome.Interval) -> np.ndarray:
    """Extract splice site masks."""
    splice_junctions_starts = self._start_position_extractor.extract(interval)
    splice_junctions_ends = self._end_position_extractor.extract(interval)
    tissues = self._tissues
    splice_junctions_starts = splice_junctions_starts[
        ['Chromosome', 'Strand', 'Start'] + tissues
    ]
    splice_junctions_ends = splice_junctions_ends[
        ['Chromosome', 'Strand', 'End'] + tissues
    ]

    donors_fw = splice_junctions_starts[splice_junctions_starts.Strand == '+']
    accept_fw = splice_junctions_ends[splice_junctions_ends.Strand == '+']
    donors_rw = splice_junctions_ends[splice_junctions_ends.Strand == '-']
    accept_rw = splice_junctions_starts[splice_junctions_starts.Strand == '-']

    donor_idx_fw = (donors_fw.Start.values - interval.start).astype(np.int32)
    accept_idx_fw = (accept_fw.End.values - interval.start).astype(np.int32)
    donor_idx_rw = (donors_rw.End.values - interval.start).astype(np.int32)
    accept_idx_rw = (accept_rw.Start.values - interval.start).astype(np.int32)

    if tissues:
      donor_theta_fw = donors_fw[tissues].to_numpy()
      accept_theta_fw = accept_fw[tissues].to_numpy()
      donor_theta_rw = donors_rw[tissues].to_numpy()
      accept_theta_rw = accept_rw[tissues].to_numpy()
    else:
      # Without tissue annotations, every junction is treated as expressed, so
      # use a single all-True column per junction.
      donor_theta_fw = np.ones((len(donor_idx_fw), 1), dtype=bool)
      accept_theta_fw = np.ones((len(accept_idx_fw), 1), dtype=bool)
      donor_theta_rw = np.ones((len(donor_idx_rw), 1), dtype=bool)
      accept_theta_rw = np.ones((len(accept_idx_rw), 1), dtype=bool)

    if interval.negative_strand:
      # Then model sees negative strand data as positive strand, swap arrays.
      donor_idx_fw, donor_idx_rw = donor_idx_rw, donor_idx_fw
      accept_idx_fw, accept_idx_rw = accept_idx_rw, accept_idx_fw
      donor_theta_fw, donor_theta_rw = donor_theta_rw, donor_theta_fw
      accept_theta_fw, accept_theta_rw = accept_theta_rw, accept_theta_fw
      # Sequence is flipped, so indices must start at end of sequence.
      donor_idx_fw = interval.width - 1 - donor_idx_fw
      accept_idx_fw = interval.width - 1 - accept_idx_fw
      donor_idx_rw = interval.width - 1 - donor_idx_rw
      accept_idx_rw = interval.width - 1 - accept_idx_rw

    splice_sites = np.zeros((interval.width, 5), dtype=bool)
    splice_sites[donor_idx_fw, 0] = np.any(donor_theta_fw > 0, axis=1)
    splice_sites[accept_idx_fw, 1] = np.any(accept_theta_fw > 0, axis=1)
    splice_sites[donor_idx_rw, 2] = np.any(donor_theta_rw > 0, axis=1)
    splice_sites[accept_idx_rw, 3] = np.any(accept_theta_rw > 0, axis=1)
    splice_sites[:, 4] = np.logical_not(np.any(splice_sites[:, :4], axis=1))

    return splice_sites
