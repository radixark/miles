"""natural_to_local_rows must agree with brute-force zigzag slicing of an index row."""

import itertools

import pytest
import torch

from miles_plugins.models.muse_spark.vision.mm_towers import natural_to_local_rows


def _zigzag_local_indices(lengths, cp_size, cp_rank):
    """Reference: concatenate slice_with_cp(arange(padded_len)) per sample (miles cp_utils semantics)."""
    rows, cu, start = [], [0], 0
    for length in lengths:
        chunk = -(-length // (2 * cp_size))
        padded = 2 * cp_size * chunk
        natural = torch.arange(start, start + padded)
        first = natural[chunk * cp_rank : chunk * (cp_rank + 1)]
        second = natural[chunk * (2 * cp_size - cp_rank - 1) : chunk * (2 * cp_size - cp_rank)]
        rows.append(torch.cat([first, second]))
        start += padded
        cu.append(start)
    return torch.cat(rows), torch.tensor(cu, dtype=torch.int32)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("sp_size", [1, 2, 4])
def test_matches_brute_force(cp_size, sp_size):
    lengths = [37, 5, 128, 1, 64]
    local_natural, cu = _zigzag_local_indices(lengths, cp_size, cp_rank=0)
    total_local = local_natural.numel()
    local_len = -(-total_local // sp_size)  # the last SP window may be shorter
    for cp_rank, sp_rank in itertools.product(range(cp_size), range(sp_size)):
        local_natural, cu = _zigzag_local_indices(lengths, cp_size, cp_rank)
        window = local_natural[sp_rank * local_len : (sp_rank + 1) * local_len]
        positions = torch.arange(int(cu[-1]))  # every natural row
        local, keep = natural_to_local_rows(positions, cu, cp_size, cp_rank, sp_size, sp_rank, local_len)
        # the kept natural rows are exactly this rank's window, in window order
        assert torch.equal(positions[keep], window)
        assert torch.equal(local[keep], torch.arange(window.numel()))


def test_cp1_identity():
    positions = torch.tensor([3, 9, 10, 500])
    local, keep = natural_to_local_rows(positions, None, 1, 0)
    assert torch.equal(local, positions) and bool(keep.all())


def test_cp_requires_cu_seqlens():
    with pytest.raises(ValueError):
        natural_to_local_rows(torch.tensor([1]), None, 2, 0)
