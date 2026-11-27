"""Sequence starts of a packed micro-batch, as the TPU true on-policy attention pages them."""

import torch

from miles.backends.torchtitan_utils.packed_sequences import count_sequences, sequence_starts


def test_padding_tokens_at_position_zero_join_the_sequence_before_them():
    # THD packing pads with position-0 tokens; counted as sequences, a micro-batch overflowed the page budget
    positions = torch.tensor([0, 1, 2, 3, 4] + [0, 1, 2] + [0, 0, 0] + [0, 1, 2, 3, 4])
    segment_ids = torch.arange(len(positions)) - positions
    assert count_sequences(positions) == 3
    assert sequence_starts(segment_ids, 6).tolist() == [0, 5, 11, 16, 16, 16, 16]
