import pytest
import torch

from miles.backends.training_utils.data.context_parallel import CPLayout, take_ranges


def _layout(cp_rank, cp_size, *, qkv_format="thd", allgather_cp=False, max_seq_len=None, pad_size=None):
    return CPLayout.build(
        qkv_format=qkv_format,
        allgather_cp=allgather_cp,
        cp_rank=cp_rank,
        cp_size=cp_size,
        max_seq_len=max_seq_len,
        pad_size=pad_size,
    )


@pytest.mark.parametrize(
    "kwargs,lengths,expected_by_rank",
    [
        # Without CP a thd sequence is held whole and a bshd one is padded to max_seq_len.
        (dict(cp_size=1), [10, 3], [[((0, 10),), ((0, 3),)]]),
        (dict(cp_size=1, allgather_cp=True), [10, 3], [[((0, 10),), ((0, 3),)]]),
        (dict(cp_size=1, qkv_format="bshd", max_seq_len=16), [10, 3], [[((0, 16),), ((0, 16),)]]),
        # Zigzag: T = 10 in chunks of 3, so rank 0 holds two padding positions past the end.
        (dict(cp_size=2), [10], [[((0, 3), (9, 12))], [((3, 6), (6, 9))]]),
        (dict(cp_size=2, qkv_format="bshd", max_seq_len=8), [5], [[((0, 2), (6, 8))], [((2, 4), (4, 6))]]),
        # bshd all-gather: one contiguous piece of each padded sequence.
        (
            dict(cp_size=2, qkv_format="bshd", allgather_cp=True, max_seq_len=8),
            [5, 3],
            [[((0, 4),), ((0, 4),)], [((4, 8),), ((4, 8),)]],
        ),
        # thd all-gather: 5 + 4 + 2 = 11 positions are padded to 12 and split at 6; the padding belongs to
        # the last sequence, and a sequence outside this rank's piece holds an empty range.
        (
            dict(cp_size=2, allgather_cp=True, pad_size=2),
            [5, 4, 2],
            [[((0, 5),), ((0, 1),), ((0, 0),)], [((0, 0),), ((1, 4),), ((0, 3),)]],
        ),
        # thd all-gather, cp 3: 2 + 9 + 1 = 12 positions split at 4 and 8, so the middle sequence covers
        # rank 1's whole piece.
        (
            dict(cp_size=3, allgather_cp=True, pad_size=1),
            [2, 9, 1],
            [[((0, 2),), ((0, 2),), ((0, 0),)], [((0, 0),), ((2, 6),), ((0, 0),)], [((0, 0),), ((6, 9),), ((0, 1),)]],
        ),
    ],
)
def test_local_ranges(kwargs, lengths, expected_by_rank):
    for cp_rank, expected in enumerate(expected_by_rank):
        assert _layout(cp_rank, **kwargs).local_ranges(lengths) == expected


def test_build_requires_what_the_layout_reads():
    with pytest.raises(AssertionError):
        _layout(0, 2, qkv_format="bshd")
    with pytest.raises(AssertionError, match="cp_size \\* pad_size"):
        _layout(0, 2, allgather_cp=True)
    with pytest.raises(AssertionError, match="max_seq_len 7 not divisible by cp_size 2"):
        _layout(0, 2, qkv_format="bshd", allgather_cp=True, max_seq_len=7)


def test_take_ranges_pads_rows_past_the_end():
    tensor = torch.tensor([1, 2, 3, 4])
    assert take_ranges(tensor, ((0, 2), (5, 7)), -1).tolist() == [1, 2, -1, -1]
    assert take_ranges(tensor, ((3, 6),), 0).tolist() == [4, 0, 0]
    assert take_ranges(tensor, ((0, 0),), 0).tolist() == []

    def pad_func(data, pad):
        return torch.cat([data, torch.full((pad, 2), -1)])

    rows = torch.arange(6).reshape(3, 2)
    assert take_ranges(rows, ((2, 4),), pad_func).tolist() == [[4, 5], [-1, -1]]
