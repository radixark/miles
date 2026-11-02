from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.data.context_parallel import CPLayout, slice_with_cp, take_ranges


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


# As-is gate: every split method must give each rank the same rows as main's inline slicing before
# CPLayout (0c46547ac36). The `_main_*` helpers restate that code, with its padding step as `pad(t, n)`.
_METHODS = [
    pytest.param("thd", False, id="thd-zigzag"),
    pytest.param("bshd", False, id="bshd-zigzag"),
    pytest.param("bshd", True, id="bshd-allgather"),
    pytest.param("thd", True, id="thd-allgather"),
]
_CP_SIZES = [1, 2, 3, 4, 8]
# Sequences shorter than one chunk, at and around chunk boundaries, and mixed micro batches.
_MICRO_BATCHES = [[1], [2], [7], [16], [17], [33, 5], [5, 4, 2], [100, 1, 63], [2, 9, 1, 30]]


def _pad_tokens(tokens, num_rows):
    return F.pad(tokens, (0, num_rows), value=0)


def _pad_replay(rows, num_rows):
    return torch.cat([rows, torch.full((num_rows, *rows.shape[1:]), -1, dtype=rows.dtype)])


def _main_zigzag(t, pad, qkv_format, max_seq_len, cp_rank, cp_size):
    if cp_size == 1:
        return pad(t, max_seq_len - t.size(0)) if qkv_format == "bshd" else t
    length = t.size(0) if qkv_format == "thd" else max_seq_len
    chunk = (length + 2 * cp_size - 1) // (2 * cp_size)
    t = pad(t, 2 * cp_size * chunk - t.size(0))
    back = 2 * cp_size - cp_rank - 1
    return torch.cat([t[chunk * cp_rank : chunk * (cp_rank + 1)], t[chunk * back : chunk * (back + 1)]])


def _main_bshd_allgather(t, pad, max_seq_len, cp_rank, cp_size):
    local_len = max_seq_len // cp_size
    start = cp_rank * local_len
    return pad(t, max_seq_len - t.size(0))[start : start + local_len]


def _main_thd_allgather(seqs, pad, pad_size, cp_rank, cp_size):
    packed = torch.cat(seqs, dim=0)
    global_pad_size = cp_size * pad_size
    num_pad = (global_pad_size - packed.size(0) % global_pad_size) % global_pad_size
    if num_pad != 0:
        packed = pad(packed, num_pad)
    return packed.chunk(cp_size, dim=0)[cp_rank]


def _main_local_rows(seqs, pad, *, qkv_format, allgather_cp, cp_rank, cp_size, max_seq_len, pad_size):
    if allgather_cp and cp_size > 1:
        if qkv_format == "bshd":
            return torch.cat([_main_bshd_allgather(t, pad, max_seq_len, cp_rank, cp_size) for t in seqs])
        return _main_thd_allgather(seqs, pad, pad_size, cp_rank, cp_size)
    return torch.cat([_main_zigzag(t, pad, qkv_format, max_seq_len, cp_rank, cp_size) for t in seqs])


def _local_rows(seqs, lengths, pad_value, **layout_kwargs):
    ranges = CPLayout.build(**layout_kwargs).local_ranges(lengths)
    return torch.cat([take_ranges(t, r, pad_value) for t, r in zip(seqs, ranges, strict=True)])


def _tokens(lengths):
    """Token ids 1, 2, ... across the micro batch, so every position is distinct from padding (0)."""
    starts = [1 + sum(lengths[:i]) for i in range(len(lengths))]
    return [torch.arange(start, start + n) for start, n in zip(starts, lengths, strict=True)]


def _max_seq_len(lengths, qkv_format, allgather_cp, cp_size, extra):
    if qkv_format != "bshd":
        return None
    max_seq_len = max(lengths) + extra
    return (max_seq_len + cp_size - 1) // cp_size * cp_size if allgather_cp else max_seq_len


@pytest.mark.parametrize("cp_size", _CP_SIZES)
@pytest.mark.parametrize("qkv_format,allgather_cp", _METHODS)
def test_local_rows_match_main(qkv_format, allgather_cp, cp_size):
    for lengths in _MICRO_BATCHES:
        tokens = _tokens(lengths)
        replay = [t[:-1].reshape(-1, 1, 1) for t in tokens]  # row p replays token p
        for extra in (0, 3):
            max_seq_len = _max_seq_len(lengths, qkv_format, allgather_cp, cp_size, extra)
            for pad_size in (1, 3):
                for cp_rank in range(cp_size):
                    layout = dict(
                        qkv_format=qkv_format,
                        allgather_cp=allgather_cp,
                        cp_rank=cp_rank,
                        cp_size=cp_size,
                        max_seq_len=max_seq_len,
                        pad_size=pad_size,
                    )
                    expected = _main_local_rows(tokens, _pad_tokens, **layout)
                    assert torch.equal(_local_rows(tokens, lengths, 0, **layout), expected), (lengths, layout)
                    # main padded each replay's missing final row before slicing.
                    expected = _main_local_rows([_pad_replay(r, 1) for r in replay], _pad_replay, **layout)
                    assert torch.equal(_local_rows(replay, lengths, _pad_replay, **layout), expected), (
                        lengths,
                        layout,
                    )
                    if not allgather_cp:
                        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size))
                        for t in tokens:
                            expected = _main_zigzag(t, _pad_tokens, qkv_format, max_seq_len, cp_rank, cp_size)
                            assert torch.equal(slice_with_cp(t, 0, qkv_format, max_seq_len, state), expected)


@pytest.mark.parametrize("cp_size", _CP_SIZES)
@pytest.mark.parametrize("qkv_format,allgather_cp", _METHODS)
def test_every_position_is_on_exactly_one_rank(qkv_format, allgather_cp, cp_size):
    for lengths in _MICRO_BATCHES:
        layout = dict(
            qkv_format=qkv_format,
            allgather_cp=allgather_cp,
            cp_size=cp_size,
            max_seq_len=_max_seq_len(lengths, qkv_format, allgather_cp, cp_size, extra=0),
            pad_size=3,
        )
        per_rank = [_local_rows(_tokens(lengths), lengths, 0, cp_rank=r, **layout) for r in range(cp_size)]
        assert len({rows.numel() for rows in per_rank}) == 1, (lengths, layout)
        held = torch.cat(per_rank)
        assert sorted(held[held != 0].tolist()) == list(range(1, sum(lengths) + 1)), (lengths, layout)


@pytest.mark.parametrize("cp_size", [2, 3, 4, 8])
def test_thd_allgather_ranges_match_get_batch_tokens(monkeypatch, cp_size):
    """get_batch still slices thd all-gather tokens itself, while replay rows take CPLayout's ranges."""
    lengths = [13, 11, 5, 1]
    rollout = {
        "tokens": _tokens(lengths),
        "loss_masks": [torch.ones(n // 2, dtype=torch.int) for n in lengths],
        "total_lengths": lengths,
        "response_lengths": [n // 2 for n in lengths],
    }
    monkeypatch.setattr(data_utils.current_platform, "current_device", lambda: torch.device("cpu"))
    for cp_rank in range(cp_size):
        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size), tp=SimpleNamespace(rank=0, size=1))
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda state=state: state)
        monkeypatch.setattr(context_parallel, "get_parallel_state", lambda state=state: state)
        batch = data_utils.get_batch(
            data_utils.DataIterator(rollout, micro_batch_size=len(lengths)),
            list(rollout),
            pad_multiplier=3,
            qkv_format="thd",
            allgather_cp=True,
        )
        expected = _local_rows(
            rollout["tokens"],
            lengths,
            0,
            qkv_format="thd",
            allgather_cp=True,
            cp_rank=cp_rank,
            cp_size=cp_size,
            pad_size=3,
        )
        assert torch.equal(batch["tokens"].flatten(), expected)
