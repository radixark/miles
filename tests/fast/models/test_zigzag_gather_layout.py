import pytest
import torch

from miles_plugins.models.cp_utils import zigzag_gather_layout


def _zigzag_shard(segments, cp_rank, cp_size):
    """Rank ``cp_rank``'s rows of the packed stream ``arange(sum(segments))``: per segment, chunks ``r`` and
    ``2 * cp_size - 1 - r`` of ``2 * cp_size``, or its contiguous ``1 / cp_size`` when not divisible by that."""
    rows, start = [], 0
    for length in segments:
        if length % (2 * cp_size) == 0:
            half = length // (2 * cp_size)
            back = 2 * cp_size - 1 - cp_rank
            rows += [*range(start + cp_rank * half, start + (cp_rank + 1) * half)]
            rows += [*range(start + back * half, start + (back + 1) * half)]
        else:
            part = length // cp_size
            rows += [*range(start + cp_rank * part, start + (cp_rank + 1) * part)]
        start += length
    return torch.tensor(rows)


@pytest.mark.parametrize("cp_size,segments", [(2, [8, 4, 12, 2]), (4, [16, 8, 4]), (2, [4])])
def test_gathered_shards_reassemble_the_packed_stream(cp_size, segments):
    cu_seqlens = torch.tensor([0, *torch.tensor(segments).cumsum(0).tolist()], dtype=torch.int32)
    shards = [_zigzag_shard(segments, rank, cp_size) for rank in range(cp_size)]
    gathered = torch.cat(shards)
    for rank, shard in enumerate(shards):
        gather_order, positions = zigzag_gather_layout(cu_seqlens, rank, cp_size, shard.numel())
        assert torch.equal(gathered.index_select(0, gather_order), torch.arange(sum(segments)))
        assert torch.equal(positions, shard)
