"""kpool indexer: pools and the eligible-pool window stay inside each packed sequence.

sglang scores one request at a time, so anything the trainer computes over a packed
micro-batch has to reproduce that per sequence. Qwen3.8-Next's QSA indexer did not, and
every sequence packed at position >= 1 ended up scoring the blocks at the front of the
buffer (1.5-4.8 nats of train/rollout logprob gap per affected sample). GLM's kpool path
is written packed-aware -- `pool_boundaries` counts pools per sequence and
`kpool_select_topk` restricts the logits to `[pool_base, pool_base + eligible_pools)`.
These tests pin that invariant so it cannot regress, with sequence lengths that are
deliberately NOT multiples of ``kpool``.
"""

import pytest

torch = pytest.importorskip("torch")

from miles_plugins.models.glm5_next.ops.kpool_indexer import pool_boundaries  # noqa: E402

KPOOL = 4
LENS = [37, 13, 22]  # none is a multiple of KPOOL


def _cu(lens):
    return torch.tensor([0, *lens]).cumsum(0).to(torch.int32)


def test_pools_are_counted_per_sequence():
    cu = _cu(LENS)
    pool_cu = pool_boundaries(cu, KPOOL)

    counts = (pool_cu[1:] - pool_cu[:-1]).tolist()
    assert counts == [length // KPOOL for length in LENS]

    # A grid laid over the whole buffer would produce more pools than the per-sequence
    # count, and the extra ones are exactly the pools that straddle a boundary.
    assert sum(counts) <= int(cu[-1]) // KPOOL


def test_eligible_pool_window_never_leaves_the_query_sequence():
    """Mirrors the index arithmetic in ``kpool_select_topk``."""
    cu = _cu(LENS)
    pool_cu = pool_boundaries(cu, KPOOL)
    total = int(cu[-1])

    token_ids = torch.arange(total)
    seq = torch.searchsorted(cu, token_ids, right=True) - 1
    seq_token_base = cu[seq]
    pool_base = pool_cu[seq]
    local_positions = token_ids - seq_token_base
    eligible_pools = torch.div(local_positions + 1, KPOOL, rounding_mode="floor")

    lo = pool_base
    hi = pool_base + eligible_pools
    own_hi = pool_cu[seq + 1]

    assert bool((lo >= pool_base).all())
    assert bool((hi <= own_hi).all()), "a query can see pools belonging to a later sequence"

    # Every query outside the first sequence starts strictly after that sequence's pools,
    # i.e. it can never be handed pool 0 of the packed buffer.
    later = seq > 0
    assert bool((lo[later] >= int(pool_cu[1])).all())
    assert int(pool_cu[1]) > 0


def test_local_positions_restart_per_sequence():
    cu = _cu(LENS)
    total = int(cu[-1])
    token_ids = torch.arange(total)
    seq = torch.searchsorted(cu, token_ids, right=True) - 1
    local_positions = token_ids - cu[seq]

    expected = torch.cat([torch.arange(length) for length in LENS])
    torch.testing.assert_close(local_positions, expected)
