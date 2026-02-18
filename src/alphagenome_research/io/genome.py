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

"""Utilities for working with genome-related objects such as intervals."""

from alphagenome.data import genome
from alphagenome_research.io import fasta


def insert_reference_variant(
    sequence: str,
    interval: genome.Interval,
    variant: genome.Variant,
) -> str:
  """Replace reference genome sequence with variant REF sequences.

  Args:
    sequence: Reference genome DNA sequence.
    interval: Interval from which the reference genome sequence was extracted.
    variant: A variant.

  Returns:
    Sequence of the same length as input `sequence`, but with variant.REF
      integrated in case it mismatches it.
  """
  if interval.negative_strand:
    raise ValueError('Interval must not be on the negative strand.')

  if not variant.reference_overlaps(interval):
    return sequence

  # Relative start/end must be within the interval boundaries.
  relative_start = max(variant.start - interval.start, 0)
  relative_end = min(variant.end - interval.start, interval.width)

  # Remove reference bases outside the interval.
  clip_start = max(interval.start - variant.start, 0)
  clip_end = min(interval.end - variant.start, variant.end - variant.start)
  variant_reference_bases = variant.reference_bases[clip_start:clip_end]

  return (
      sequence[:relative_start]
      + variant_reference_bases
      + sequence[relative_end:]
  )


def insert_alternate_variant(
    sequence: str, interval: genome.Interval, variant: genome.Variant
) -> str:
  """Replace reference genome sequence with a single variant ALT sequence.

  Args:
    sequence: Reference genome DNA sequence.
    interval: Interval corresponding to reference sequence.
    variant: Variant to integrate.

  Returns:
    DNA sequence with integrated variant. Length of the sequence might be
    be different from the input sequence.
  """
  if interval.negative_strand:
    raise ValueError('Interval must not be on the negative strand.')

  if not (
      variant.reference_overlaps(interval)
      or (not variant.reference_bases and variant.alternate_overlaps(interval))
  ):
    return sequence

  if variant.end > interval.end:
    # Variant reference is partially outside of the interval at the end.
    variant, _ = variant.split(interval.end)
    assert isinstance(variant, genome.Variant)

  if variant.start < interval.start:
    # Variant reference is partially outside of the interval at the start.
    _, variant = variant.split(interval.start)
    assert isinstance(variant, genome.Variant)

  relative_start = variant.start - interval.start
  relative_end = relative_start + len(variant.reference_bases)
  return (
      sequence[:relative_start]
      + variant.alternate_bases
      + sequence[relative_end:]
  )


def extract_variant_sequences(
    interval: genome.Interval,
    variant: genome.Variant,
    extractor: fasta.FastaExtractor,
) -> tuple[str, str]:
  """Extracts the interval and inserts variant to generate ref/alt sequences.

  Args:
      interval: Interval of interest from which to query the sequence.
      variant: Variant to insert.
      extractor: FASTA extractor for extracting the sequence.

  Returns:
      Tuple of reference and alternate sequences.
  """

  # If the variant is a deletion, extend the interval to account for the
  # shrinking sequence.
  interval_length = interval.width
  extended_length = max(
      0, len(variant.reference_bases) - len(variant.alternate_bases)
  )
  extended_unstranded_interval = interval.as_unstranded().boundary_shift(
      end_offset=extended_length
  )

  sequence = extractor.extract(extended_unstranded_interval)
  reference_sequence = insert_reference_variant(
      sequence, extended_unstranded_interval, variant
  )[:interval_length]
  alternate_sequence = insert_alternate_variant(
      sequence, extended_unstranded_interval, variant
  )[:interval_length]

  if interval.negative_strand:
    reference_sequence = fasta.reverse_complement(reference_sequence)
    alternate_sequence = fasta.reverse_complement(alternate_sequence)

  return reference_sequence, alternate_sequence
