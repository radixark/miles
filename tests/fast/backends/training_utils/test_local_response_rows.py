"""``cp_utils.iter_local_response_rows`` must name exactly the local logit rows that score each response.

Each token's value encodes its sample and position, so a yielded row is checked independently of
cp_utils: it must sit at the position just before its token, in the layout the CP mode gives this
rank. Over all ranks, the rows of a response must tile it exactly once.
"""

import random
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import cp_utils

_SAMPLE_STRIDE = 1_000_000  # token value = sample * _SAMPLE_STRIDE + position


def _tokens(total_lengths):
    return [torch.arange(total) + i * _SAMPLE_STRIDE for i, total in enumerate(total_lengths)]


def _zigzag_chunk_size(total_length, max_seq_len, qkv_format, cp_size):
    padded = total_length if qkv_format == "thd" else max_seq_len
    return -(-padded // (2 * cp_size))


class _Layout:
    """The sequence position each local logit row holds, per sample, for one rank."""

    def __init__(self, *, mode, cp_rank, cp_size, total_lengths, qkv_format, max_seq_len):
        self.mode, self.cp_rank, self.cp_size = mode, cp_rank, cp_size
        self.total_lengths, self.qkv_format, self.max_seq_len = total_lengths, qkv_format, max_seq_len
        if mode == "zigzag":
            self.chunks = [_zigzag_chunk_size(t, max_seq_len, qkv_format, cp_size) for t in total_lengths]
            self.num_rows = sum(2 * chunk for chunk in self.chunks)
        elif mode == "allgather":
            assert sum(total_lengths) % cp_size == 0
            self.num_rows = sum(total_lengths) // cp_size
        elif qkv_format == "bshd":
            self.num_rows = max_seq_len * len(total_lengths)
        else:
            self.num_rows = sum(total_lengths)

    def position(self, sample, row):
        """Position within ``sample`` of the token whose logit is local row ``row``."""
        if self.mode == "zigzag":
            local = row - sum(2 * chunk for chunk in self.chunks[:sample])
            chunk = self.chunks[sample]
            half, offset = divmod(local, chunk)
            chunk_index = self.cp_rank if half == 0 else 2 * self.cp_size - 1 - self.cp_rank
            return chunk_index * chunk + offset
        if self.mode == "allgather":
            return self.cp_rank * self.num_rows + row - sum(self.total_lengths[:sample])
        if self.qkv_format == "bshd":
            return row - self.max_seq_len * sample
        return row - sum(self.total_lengths[:sample])


def _check_layout(monkeypatch, *, mode, cp_size, prompt_lengths, response_lengths, qkv_format="thd"):
    total_lengths = [p + r for p, r in zip(prompt_lengths, response_lengths, strict=True)]
    max_seq_len = max(total_lengths) + 3 if qkv_format == "bshd" else None
    tokens = _tokens(total_lengths)
    covered = [[] for _ in total_lengths]
    for cp_rank in range(cp_size):
        parallel_state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size))
        monkeypatch.setattr(cp_utils, "get_parallel_state", lambda state=parallel_state: state)
        layout = _Layout(
            mode=mode,
            cp_rank=cp_rank,
            cp_size=cp_size,
            total_lengths=total_lengths,
            qkv_format=qkv_format,
            max_seq_len=max_seq_len,
        )
        samples = cp_utils.iter_local_response_rows(
            layout.num_rows,
            unconcat_tokens=tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            qkv_format=qkv_format,
            allgather_cp=mode == "allgather",
            max_seq_lens=[max_seq_len] * len(total_lengths) if max_seq_len else None,
            include_response_indices=True,
        )
        for sample, (row_ranges, sample_tokens, response_indices) in enumerate(samples):
            rows = [row for start, end in row_ranges for row in range(start, end)]
            assert all(0 <= start <= end <= layout.num_rows for start, end in row_ranges), row_ranges
            assert sample_tokens.tolist() == [sample * _SAMPLE_STRIDE + layout.position(sample, r) + 1 for r in rows]
            prompt_length = prompt_lengths[sample]
            assert list(response_indices) == [layout.position(sample, r) + 1 - prompt_length for r in rows]
            covered[sample].extend(response_indices)
    for sample, response_length in enumerate(response_lengths):
        assert sorted(covered[sample]) == list(range(response_length)), f"sample {sample} not tiled once"


def test_zigzag_half_without_response_rows_stays_in_bounds(monkeypatch):
    """Rank 1 holds chunks 1 and 2 of the first sample (tokens 25-74 of 100), all prompt: both of
    its halves are empty, and must stay empty at their own start rather than map to a negative row."""
    _check_layout(monkeypatch, mode="zigzag", cp_size=2, prompt_lengths=[90, 10], response_lengths=[10, 30])


def _random_lengths(rng, n_samples, cp_size, divisible_total):
    prompts = [rng.randint(1, 60) for _ in range(n_samples)]
    responses = [rng.randint(0, 60) for _ in range(n_samples)]
    if divisible_total:
        prompts[-1] += -sum(prompts + responses) % cp_size
    return prompts, responses


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize(
    ("mode", "cp_size", "qkv_format"),
    [
        ("none", 1, "thd"),
        ("none", 1, "bshd"),
        ("allgather", 2, "thd"),
        ("allgather", 4, "thd"),
        ("zigzag", 2, "thd"),
        ("zigzag", 4, "thd"),
        ("zigzag", 2, "bshd"),
    ],
)
def test_random_layouts(monkeypatch, seed, mode, cp_size, qkv_format):
    rng = random.Random(seed)
    prompts, responses = _random_lengths(rng, rng.randint(1, 5), cp_size, divisible_total=mode == "allgather")
    _check_layout(
        monkeypatch,
        mode=mode,
        cp_size=cp_size,
        prompt_lengths=prompts,
        response_lengths=responses,
        qkv_format=qkv_format,
    )
