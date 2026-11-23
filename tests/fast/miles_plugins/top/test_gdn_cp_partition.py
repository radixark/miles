"""GDN all-gather CP partition: every rank gets a piece whenever the documents have enough legal cuts."""

import itertools
import random

import pytest

from miles_plugins.top.gdn_cp import _HALO, legal_cut, partition

GRID = 64


def _legal_points(length):
    """Rows serving's chunk prefill can resume at (its grid), inside the halo."""
    return [p for p in range(GRID, length, GRID) if _HALO <= p <= length - _HALO]


def _max_pieces(length):
    """Most pieces of at least _HALO rows with legal cuts (shortest first fits the most)."""
    count, at = 1, 0
    for point in _legal_points(length):
        if point - at >= _HALO:
            count, at = count + 1, point
    return count


def _documents():
    """Single documents around the grid, and small multi-document batches."""
    out = [[n] for n in range(_HALO, 400, 7)] + [[n] for n in (64, 66, 67, 128, 130, 131, 192, 256)]
    rng = random.Random(0)
    for _ in range(300):
        out.append([rng.choice([3, 5, 30, 64, 66, 70, 128, 130, 200, 333]) for _ in range(rng.randint(1, 4))])
    return out


def test_cuts_stay_on_the_grid():
    assert legal_cut(200, 150, 2, 198, grid=GRID) == 128
    assert legal_cut(200, 40, 2, 198, grid=GRID) == 64  # none at or below: the next grid row
    assert legal_cut(66, 40, 2, 64, grid=GRID) is None  # the only grid row leaves a 2-row last piece


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_partition_succeeds_exactly_when_a_legal_split_exists(cp_size):
    for lengths in _documents():
        feasible = sum(_max_pieces(n) for n in lengths) >= cp_size
        try:
            pieces = partition(lengths, cp_size, grid=GRID)
        except ValueError:
            assert not feasible, (cp_size, lengths)
            continue
        assert feasible, (cp_size, lengths, pieces)
        covered = {}
        for rank_pieces in pieces:
            assert rank_pieces, (cp_size, lengths, pieces)
            for d, start, end in rank_pieces:
                assert end - start >= _HALO, (cp_size, lengths, pieces)
                assert start % GRID == 0, (cp_size, lengths, pieces)
                covered.setdefault(d, []).append((start, end))
        for d, spans in covered.items():
            spans.sort()
            assert spans[0][0] == 0 and spans[-1][1] == lengths[d]
            assert all(a[1] == b[0] for a, b in itertools.pairwise(spans)), (lengths, pieces)


def test_a_short_sample_cannot_feed_every_rank():
    with pytest.raises(ValueError, match="cannot give each of 4 ranks a piece"):
        partition([150], 4, grid=GRID)
