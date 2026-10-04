"""Packed position resets must agree with the explicit document boundaries."""

from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])


def _get_batch(monkeypatch, lengths, *, tp_size=1, pad_multiplier=8, cp_size=1, cp_rank=0, qkv_format="thd"):
    state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size), tp=SimpleNamespace(rank=0, size=tp_size))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: state)
    rollout = {
        "tokens": [torch.arange(1, n + 1) for n in lengths],
        "loss_masks": [torch.ones(n // 2, dtype=torch.int) for n in lengths],
        "total_lengths": lengths,
        "response_lengths": [n // 2 for n in lengths],
        "witness_ids": [torch.full((n,), i + 1, dtype=torch.long) for i, n in enumerate(lengths)],
    }
    if qkv_format == "bshd":
        rollout["max_seq_lens"] = [8] * len(lengths)
    return data_utils.get_batch(
        data_utils.DataIterator(rollout, micro_batch_size=len(lengths)),
        list(rollout),
        pad_multiplier=pad_multiplier,
        qkv_format=qkv_format,
        get_position_ids=True,
    )


@pytest.mark.parametrize(
    "lengths,tp_size,pad_multiplier",
    [
        pytest.param([3, 1, 4], 1, 8, id="no-padding"),
        pytest.param([3, 1, 3], 1, 8, id="one-pad-token"),
        pytest.param([3, 1, 2], 1, 8, id="multiple-pad-tokens"),
        pytest.param([1, 1], 1, 8, id="single-token-documents"),
        pytest.param([449], 8, 128, id="tp8-large-padding"),
    ],
)
def test_thd_position_resets_match_cu_seqlens(monkeypatch, lengths, tp_size, pad_multiplier):
    batch = _get_batch(monkeypatch, lengths, tp_size=tp_size, pad_multiplier=pad_multiplier)
    positions = batch["position_ids"][0]
    # TorchTitan derives document boundaries from these resets, including the padding document.
    starts = (positions == 0).nonzero(as_tuple=True)[0].tolist()
    assert starts == list(batch["cu_seqlens_host"][:-1])
    assert starts == batch["cu_seqlens"][:-1].tolist()

    real_length = sum(lengths)
    torch.testing.assert_close(positions[:real_length], torch.cat([torch.arange(n) for n in lengths]))
    torch.testing.assert_close(positions[real_length:], torch.arange(positions.numel() - real_length))
    torch.testing.assert_close(batch["tokens"][0, :real_length], torch.cat([torch.arange(1, n + 1) for n in lengths]))
    torch.testing.assert_close(
        batch["witness_ids"][0, :real_length],
        torch.cat([torch.full((n,), i + 1, dtype=torch.long) for i, n in enumerate(lengths)]),
    )
    for key in ("tokens", "input_loss_masks", "witness_ids"):
        assert torch.count_nonzero(batch[key][0, real_length:]) == 0


@pytest.mark.parametrize(
    "cp_rank,expected",
    [(0, [0, 0, 0, 0, 0, 0, 0, 0]), (1, [1, 2, 0, 0, 1, 0, 0, 0])],
)
def test_thd_cp2_keeps_sharded_positions(monkeypatch, cp_rank, expected):
    batch = _get_batch(monkeypatch, [3, 1, 2], cp_size=2, cp_rank=cp_rank)
    assert batch["position_ids"].tolist() == [expected]


def test_bshd_keeps_zero_padding(monkeypatch):
    batch = _get_batch(monkeypatch, [3, 1, 2], qkv_format="bshd")
    assert batch["position_ids"].tolist() == [
        [0, 1, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
    ]
