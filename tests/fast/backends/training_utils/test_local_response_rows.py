"""``iter_local_response_rows`` must name the rows that score each response, tiling it once over all ranks."""

import random
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel

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
        elif mode == "allgather" and qkv_format == "bshd":
            assert max_seq_len % cp_size == 0
            self.piece = max_seq_len // cp_size
            self.num_rows = self.piece * len(total_lengths)
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
        if self.mode == "allgather" and self.qkv_format == "bshd":
            return self.cp_rank * self.piece + row - self.piece * sample
        if self.mode == "allgather":
            return self.cp_rank * self.num_rows + row - sum(self.total_lengths[:sample])
        if self.qkv_format == "bshd":
            return row - self.max_seq_len * sample
        return row - sum(self.total_lengths[:sample])


def _check_layout(monkeypatch, *, mode, cp_size, prompt_lengths, response_lengths, qkv_format="thd"):
    total_lengths = [p + r for p, r in zip(prompt_lengths, response_lengths, strict=True)]
    max_seq_len = max(total_lengths) + 3 if qkv_format == "bshd" else None
    if max_seq_len and mode == "allgather":
        max_seq_len += -max_seq_len % cp_size
    tokens = _tokens(total_lengths)
    covered = [[] for _ in total_lengths]
    for cp_rank in range(cp_size):
        parallel_state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size))
        monkeypatch.setattr(context_parallel, "get_parallel_state", lambda state=parallel_state: state)
        layout = _Layout(
            mode=mode,
            cp_rank=cp_rank,
            cp_size=cp_size,
            total_lengths=total_lengths,
            qkv_format=qkv_format,
            max_seq_len=max_seq_len,
        )
        samples = context_parallel.iter_local_response_rows(
            layout.num_rows,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            qkv_format=qkv_format,
            allgather_cp=mode == "allgather",
            max_seq_lens=[max_seq_len] * len(total_lengths) if max_seq_len else None,
        )
        for sample, sample_rows in enumerate(samples):
            rows = [row for start, end in sample_rows.row_ranges for row in range(start, end)]
            assert all(0 <= start <= end <= layout.num_rows for start, end in sample_rows.row_ranges), sample_rows
            prompt_length = prompt_lengths[sample]
            sample_tokens = sample_rows.tokens(tokens[sample], prompt_length)
            assert sample_tokens.tolist() == [sample * _SAMPLE_STRIDE + layout.position(sample, r) + 1 for r in rows]
            response_indices = sample_rows.response_indices()
            assert response_indices == [layout.position(sample, r) + 1 - prompt_length for r in rows]
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
        ("allgather", 2, "bshd"),
        ("allgather", 4, "bshd"),
        ("zigzag", 2, "thd"),
        ("zigzag", 4, "thd"),
        ("zigzag", 2, "bshd"),
    ],
)
def test_random_layouts(monkeypatch, seed, mode, cp_size, qkv_format):
    rng = random.Random(seed)
    divisible_total = mode == "allgather" and qkv_format == "thd"
    prompts, responses = _random_lengths(rng, rng.randint(1, 5), cp_size, divisible_total=divisible_total)
    _check_layout(
        monkeypatch,
        mode=mode,
        cp_size=cp_size,
        prompt_lengths=prompts,
        response_lengths=responses,
        qkv_format=qkv_format,
    )
