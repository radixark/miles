"""Kpool pools and the eligible-pool window stay inside each packed sequence, as sglang scores per request."""

import pytest

torch = pytest.importorskip("torch")

from miles_plugins.models.glm5_next.ops.kpool_indexer import pool_boundaries  # noqa: E402

KPOOL = 4
SEQ_LENS = [37, 13, 22]  # none is a multiple of KPOOL


def _cu_seqlens(seq_lens):
    return torch.tensor([0, *seq_lens]).cumsum(0).to(torch.int32)


def test_pools_are_counted_per_sequence():
    cu_seqlens = _cu_seqlens(SEQ_LENS)
    pool_cu_seqlens = pool_boundaries(cu_seqlens, KPOOL)

    counts = (pool_cu_seqlens[1:] - pool_cu_seqlens[:-1]).tolist()
    assert counts == [length // KPOOL for length in SEQ_LENS]

    # A grid laid over the whole buffer would produce more pools than the per-sequence
    # count, and the extra ones are exactly the pools that straddle a boundary.
    assert sum(counts) <= int(cu_seqlens[-1]) // KPOOL


def test_eligible_pool_window_never_leaves_the_query_sequence():
    """Mirrors the index arithmetic in `kpool_select_topk`."""
    cu_seqlens = _cu_seqlens(SEQ_LENS)
    pool_cu_seqlens = pool_boundaries(cu_seqlens, KPOOL)
    total_tokens = int(cu_seqlens[-1])

    token_ids = torch.arange(total_tokens)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    seq_token_base = cu_seqlens[seq_indices]
    pool_base = pool_cu_seqlens[seq_indices]
    local_positions = token_ids - seq_token_base
    eligible_pools = torch.div(local_positions + 1, KPOOL, rounding_mode="floor")

    window_start = pool_base
    window_end = pool_base + eligible_pools
    own_window_end = pool_cu_seqlens[seq_indices + 1]

    assert bool((window_start >= pool_base).all())
    assert bool((window_end <= own_window_end).all()), "a query can see pools belonging to a later sequence"

    # Every query outside the first sequence starts strictly after that sequence's pools,
    # i.e. it can never be handed pool 0 of the packed buffer.
    is_later_seq = seq_indices > 0
    assert bool((window_start[is_later_seq] >= int(pool_cu_seqlens[1])).all())
    assert int(pool_cu_seqlens[1]) > 0


def test_local_positions_restart_per_sequence():
    cu_seqlens = _cu_seqlens(SEQ_LENS)
    total_tokens = int(cu_seqlens[-1])
    token_ids = torch.arange(total_tokens)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    local_positions = token_ids - cu_seqlens[seq_indices]

    expected = torch.cat([torch.arange(length) for length in SEQ_LENS])
    torch.testing.assert_close(local_positions, expected)
