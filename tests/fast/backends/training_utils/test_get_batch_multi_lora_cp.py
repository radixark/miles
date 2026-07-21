"""CP=2 tests for get_batch's multi-LoRA per-adapter token counts; CUDA is stubbed to run on CPU."""

from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.cp_utils as cp_utils_mod
import miles.backends.training_utils.data as data_mod
from miles.backends.training_utils.cp_utils import slice_with_cp
from miles.backends.training_utils.data import get_batch


def _parallel_state(cp_rank: int, cp_size: int, tp_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        cp=SimpleNamespace(rank=cp_rank, size=cp_size),
        tp=SimpleNamespace(size=tp_size),
    )


class _FakeIterator:
    def __init__(self, batch: dict, n_adapters: int):
        self._batch = batch
        self.rollout_data = {"n_adapters": n_adapters}

    def get_next(self, keys):
        return {key: self._batch[key] for key in keys}


KEYS = ["tokens", "loss_masks", "total_lengths", "response_lengths", "adapter_slots"]

# 5 samples over 3 of 4 slots (sorted), ragged/odd lengths to exercise the
# per-sample zigzag padding (2 * cp_size chunking).
LENGTHS = [7, 13, 5, 9, 4]
SLOTS = [0, 0, 1, 2, 2]
N_ADAPTERS = 4


def _make_batch(max_seqlen: int | None = None) -> dict:
    # Token values start at 1 so zigzag pad (value 0) is distinguishable.
    tokens = [torch.arange(1, length + 1, dtype=torch.long) for length in LENGTHS]
    response_lengths = [max(1, length // 2) for length in LENGTHS]
    batch = {
        "tokens": tokens,
        "loss_masks": [torch.ones(r, dtype=torch.int) for r in response_lengths],
        "total_lengths": list(LENGTHS),
        "response_lengths": response_lengths,
        "adapter_slots": list(SLOTS),
    }
    if max_seqlen is not None:
        batch["max_seq_lens"] = [max_seqlen] * len(LENGTHS)
    return batch


@pytest.fixture(autouse=True)
def _stub_cuda(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu", raising=False)


def _patch_state(monkeypatch, state: SimpleNamespace) -> None:
    # data.get_batch and cp_utils.slice_with_cp each resolve the state themselves.
    monkeypatch.setattr(data_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils_mod, "get_parallel_state", lambda: state)


def _expected_thd_counts(state: SimpleNamespace, local_total: int) -> torch.Tensor:
    sliced_lengths = [
        slice_with_cp(t, 0, "thd", parallel_state=state).numel()
        for t in (torch.arange(1, length + 1, dtype=torch.long) for length in LENGTHS)
    ]
    expected = torch.zeros(N_ADAPTERS, dtype=torch.int32)
    for slot, sliced_length in zip(SLOTS, sliced_lengths, strict=True):
        expected[slot] += sliced_length
    stream_pad = local_total - int(sum(sliced_lengths))
    assert stream_pad >= 0, "rank-local stream shorter than the sliced samples"
    expected[SLOTS[-1]] += stream_pad
    return expected


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_thd_cp2_adapter_token_counts(monkeypatch, cp_rank):
    state = _parallel_state(cp_rank, cp_size=2)
    _patch_state(monkeypatch, state)

    out = get_batch(_FakeIterator(_make_batch(), N_ADAPTERS), KEYS, pad_multiplier=8, qkv_format="thd")

    counts = out["adapter_token_counts"]
    local_total = out["tokens"].numel()
    assert counts.dtype == torch.int32
    assert counts.tolist() == _expected_thd_counts(state, local_total).tolist()
    assert int(counts.sum()) == local_total, "counts must cover every rank-local token incl. padding"


def test_thd_cp2_counts_identical_across_ranks(monkeypatch):
    # Zigzag gives each rank the same padded share of every sample, so the
    # grouped-GEMM routing counts must not depend on the CP rank.
    per_rank = []
    for cp_rank in (0, 1):
        state = _parallel_state(cp_rank, cp_size=2)
        _patch_state(monkeypatch, state)
        out = get_batch(_FakeIterator(_make_batch(), N_ADAPTERS), KEYS, pad_multiplier=8, qkv_format="thd")
        per_rank.append(out["adapter_token_counts"].tolist())
    assert per_rank[0] == per_rank[1]


def test_unsorted_adapter_slots_rejected(monkeypatch):
    _patch_state(monkeypatch, _parallel_state(cp_rank=0, cp_size=2))
    batch = _make_batch()
    batch["adapter_slots"] = [2, 0, 1, 0, 2]
    with pytest.raises(AssertionError, match="not sorted"):
        get_batch(_FakeIterator(batch, N_ADAPTERS), KEYS, pad_multiplier=8, qkv_format="thd")


def test_bshd_cp2_tokens_are_single_sliced(monkeypatch):
    # Regression: the bshd path used to slice tokens twice under CP>1, zeroing the back half of each rank's tokens.
    max_seqlen = 16  # divisible by 2 * cp_size
    state = _parallel_state(cp_rank=0, cp_size=2)
    _patch_state(monkeypatch, state)

    out = get_batch(
        _FakeIterator(_make_batch(max_seqlen=max_seqlen), N_ADAPTERS),
        KEYS + ["max_seq_lens"],
        pad_multiplier=8,
        qkv_format="bshd",
    )

    expected = torch.stack(
        [
            slice_with_cp(torch.arange(1, length + 1, dtype=torch.long), 0, "bshd", max_seqlen, parallel_state=state)
            for length in LENGTHS
        ]
    )
    assert torch.equal(out["tokens"], expected)
