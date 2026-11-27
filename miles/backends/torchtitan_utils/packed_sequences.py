"""Where packed sequences start in a torchtitan micro-batch, from its per-token positions or segment ids.

Positions restart at 0 in every sample. THD packing pads a micro-batch with position-0 tokens, each its own one-token
segment, and the trainer pads the tail with positions 0, 1, ...; a sequence here is a segment of at least two
tokens, and one-token segments join the sequence before them.
"""

import torch


def count_sequences(positions: torch.Tensor) -> int:
    """How many sequences `sequence_starts` finds in a micro-batch with these positions."""
    return int(((positions[:-1] == 0) & (positions[1:] == 1)).sum())


def sequence_starts(segment_ids: torch.Tensor, max_sequences: int) -> torch.Tensor:
    """[max_sequences + 1] int32 sequence starts; entries past the last sequence equal the token count.

    A segment's id is the index of its first token. A sort, not a cumsum: with the scan emitter off (the TPU trainer's
    libtpu flags), torch_tpu cannot lower an int64 cumsum.
    """
    tokens = segment_ids.shape[0]
    seg = segment_ids.to(torch.int32)
    index = torch.arange(tokens, dtype=torch.int32, device=segment_ids.device)
    next_seg = torch.cat([seg[1:], seg.new_full((1,), -1)])
    starts = torch.where((seg == index) & (next_seg == index), index, tokens)
    first = torch.sort(starts).values[:max_sequences]
    return torch.cat([first, first.new_full((max_sequences + 1 - first.shape[0],), tokens)])
