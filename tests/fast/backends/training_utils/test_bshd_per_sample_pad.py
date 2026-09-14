"""bshd padded lengths are per-sample, not one rollout-global max; CUDA is stubbed to run on CPU."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.cp_utils as cp_utils_mod
import miles.backends.training_utils.data as data_mod
import miles.backends.training_utils.replay_data as replay_data_mod
from miles.backends.training_utils.data import get_batch, get_rollout_data
from miles.backends.training_utils.replay_data import fill_replay_data


def _parallel_state(cp_rank: int = 0, cp_size: int = 1, tp_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        cp=SimpleNamespace(rank=cp_rank, size=cp_size),
        tp=SimpleNamespace(rank=0, size=tp_size),
        effective_dp=SimpleNamespace(rank=0, size=1),
    )


class _FakeIterator:
    def __init__(self, batch: dict):
        self._batch = batch
        self.rollout_data = {}

    def get_next(self, keys):
        return {key: self._batch[key] for key in keys}

    def reset(self):
        return self


KEYS = ["tokens", "loss_masks", "total_lengths", "response_lengths", "max_seq_lens"]


def _make_batch(lengths: list[int], max_seq_lens: list[int]) -> dict:
    response_lengths = [max(1, length // 2) for length in lengths]
    return {
        "tokens": [torch.arange(1, length + 1, dtype=torch.long) for length in lengths],
        "loss_masks": [torch.ones(r, dtype=torch.int) for r in response_lengths],
        "total_lengths": list(lengths),
        "response_lengths": response_lengths,
        "max_seq_lens": list(max_seq_lens),
        "adapter_slots": None,
    }


@pytest.fixture(autouse=True)
def _stub_cuda(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu", raising=False)


def _patch_state(monkeypatch, state: SimpleNamespace) -> None:
    monkeypatch.setattr(data_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(replay_data_mod, "get_parallel_state", lambda: state)


def test_rollout_max_seq_lens_are_per_sample(monkeypatch):
    # Regression: every sample used to be padded to the rollout-global max, so
    # one 130-token outlier multiplied the short samples' padded lengths.
    total_lengths = [7, 130, 24]
    rollout_data = {
        "tokens": [list(range(length)) for length in total_lengths],
        "loss_masks": [[1] * max(1, length // 2) for length in total_lengths],
        "total_lengths": list(total_lengths),
        "response_lengths": [max(1, length // 2) for length in total_lengths],
    }
    _patch_state(monkeypatch, _parallel_state())
    monkeypatch.setattr(data_mod, "process_rollout_data", lambda *args, **kwargs: (rollout_data, None))
    args = Namespace(qkv_format="bshd", enable_witness=False, data_pad_size_multiplier=8, compress_ratios=[])

    out, _ = get_rollout_data(args, rollout_data_ref=None)

    assert out["max_seq_lens"] == [8, 136, 24]


def test_get_batch_bshd_accepts_shared_padded_length(monkeypatch):
    _patch_state(monkeypatch, _parallel_state())

    out = get_batch(_FakeIterator(_make_batch([7, 8], [8, 8])), KEYS, pad_multiplier=8, qkv_format="bshd")

    assert out["tokens"].shape == (2, 8)
    assert out["full_loss_masks"].shape == (2, 8)
    assert torch.equal(out["tokens"][0, :7], torch.arange(1, 8, dtype=torch.long))
    assert torch.all(out["tokens"][0, 7:] == 0)
    assert torch.equal(out["tokens"][1], torch.arange(1, 9, dtype=torch.long))


def test_get_batch_bshd_rejects_mixed_padded_lengths(monkeypatch):
    # Build-time CP slicing uses each sample's own padded length, so mixing
    # padded lengths inside one bshd microbatch must fail loudly, not misalign.
    _patch_state(monkeypatch, _parallel_state())
    batch = _make_batch([7, 130], [8, 136])

    with pytest.raises(AssertionError, match="mixes padded lengths"):
        get_batch(_FakeIterator(batch), KEYS, pad_multiplier=8, qkv_format="bshd")


def test_fill_replay_data_bshd_rejects_mixed_padded_lengths(monkeypatch):
    _patch_state(monkeypatch, _parallel_state())
    batch = _make_batch([7, 130], [8, 136])
    batch["routing"] = [torch.zeros(length - 1, 1, 1) for length in batch["total_lengths"]]

    with pytest.raises(AssertionError, match="mixes padded lengths"):
        fill_replay_data(
            args=Namespace(qkv_format="bshd"),
            models=None,
            data_iterator=[_FakeIterator(batch)],
            num_microbatches=[1],
            rollout_data=batch,
            data_key="routing",
            replay_list=[],
            register_replay_list_func=lambda *_args, **_kwargs: None,
        )
