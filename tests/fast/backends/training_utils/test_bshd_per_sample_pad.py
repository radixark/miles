"""BSHD padding policy and rectangular layout guards; CUDA is stubbed to run on CPU."""

from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel as cp_utils_mod
from miles.backends.training_utils.data import rollout as data_mod
from miles.backends.training_utils.data.rollout import get_batch, get_rollout_data
from miles.backends.training_utils.parallel import GroupInfo, ParallelState
from miles.backends.training_utils.replay import data as replay_data_mod
from miles.backends.training_utils.replay.data import fill_replay_data


def _parallel_state(cp_rank: int = 0, cp_size: int = 1, tp_size: int = 1) -> ParallelState:
    trivial = GroupInfo(rank=0, size=1, group=None)
    return ParallelState(
        cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        tp=GroupInfo(rank=0, size=tp_size, group=None),
        intra_dp=trivial,
        intra_dp_cp=GroupInfo(rank=cp_rank, size=cp_size, group=None),
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
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
def _stub_device(monkeypatch):
    monkeypatch.setattr(data_mod.current_platform, "current_device", lambda: torch.device("cpu"))


def _patch_state(monkeypatch, state: ParallelState) -> None:
    monkeypatch.setattr(data_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(replay_data_mod, "get_parallel_state", lambda: state)


@pytest.mark.parametrize(
    ("mode", "tp_size", "cp_size", "compress_ratios", "expected"),
    [
        ("rollout", 1, 1, [], [136, 136, 136]),
        ("per-sample", 1, 1, [], [8, 136, 24]),
        ("rollout", 2, 1, [], [144, 144, 144]),
        ("per-sample", 2, 1, [], [16, 144, 32]),
        ("rollout", 1, 2, [16], [192, 192, 192]),
        ("per-sample", 1, 2, [16], [64, 192, 64]),
    ],
)
def test_rollout_padding_policy(monkeypatch, mode, tp_size, cp_size, compress_ratios, expected):
    total_lengths = [7, 130, 24]
    rollout_data = {
        "tokens": [list(range(length)) for length in total_lengths],
        "loss_masks": [[1] * max(1, length // 2) for length in total_lengths],
        "total_lengths": list(total_lengths),
        "response_lengths": [max(1, length // 2) for length in total_lengths],
    }
    state = _parallel_state(tp_size=tp_size, cp_size=cp_size)
    _patch_state(monkeypatch, state)
    monkeypatch.setattr(data_mod, "process_rollout_data", lambda *args, **kwargs: (rollout_data, None))
    args = Namespace(
        qkv_format="bshd",
        bshd_padding_mode=mode,
        enable_witness=False,
        data_pad_size_multiplier=8,
        backend=Namespace(compress_ratios=compress_ratios),
    )

    out, _ = get_rollout_data(
        args,
        rollout_data_ref=None,
        train_parallel_config=state.train_parallel_config(supports_precomputed_schedule=True),
    )

    assert out["max_seq_lens"] == expected


@pytest.mark.parametrize(
    ("mode", "cp_rank", "expected"),
    [("rollout", 0, [0, 1, 2]), ("rollout", 1, []), ("per-sample", 0, []), ("per-sample", 1, [0, 1, 2])],
)
def test_padding_policy_applies_before_response_field_cp_slicing(monkeypatch, mode, cp_rank, expected):
    rollout_data = {
        "tokens": [list(range(7)), list(range(130))],
        "loss_masks": [[1] * 3, [1] * 65],
        "total_lengths": [7, 130],
        "response_lengths": [3, 65],
        **{
            key: [list(range(3)), list(range(65))]
            for key in ("rollout_log_probs", "teacher_log_probs", "opd_reverse_kl")
        },
    }
    state = _parallel_state(cp_size=2, cp_rank=cp_rank)
    _patch_state(monkeypatch, state)
    monkeypatch.setattr(data_mod, "process_rollout_data", lambda *args, **kwargs: (rollout_data, None))
    args = Namespace(
        qkv_format="bshd",
        bshd_padding_mode=mode,
        enable_witness=False,
        data_pad_size_multiplier=8,
        backend=Namespace(compress_ratios=[]),
        true_on_policy_mode=False,
    )

    out, _ = get_rollout_data(
        args,
        rollout_data_ref=None,
        train_parallel_config=state.train_parallel_config(supports_precomputed_schedule=True),
    )

    for key in ("rollout_log_probs", "teacher_log_probs", "opd_reverse_kl"):
        assert out[key][0].tolist() == expected


@pytest.mark.parametrize(("lengths", "padded_length"), [([7, 8], 8), ([7, 130], 136)])
def test_get_batch_bshd_accepts_shared_padded_length(monkeypatch, lengths, padded_length):
    _patch_state(monkeypatch, _parallel_state())

    out = get_batch(
        _FakeIterator(_make_batch(lengths, [padded_length] * 2)), KEYS, pad_multiplier=8, qkv_format="bshd"
    )

    assert out["tokens"].shape == (2, padded_length)
    assert out["input_loss_masks"].shape == (2, padded_length)
    for i, length in enumerate(lengths):
        assert torch.equal(out["tokens"][i, :length], torch.arange(1, length + 1, dtype=torch.long))
        assert torch.all(out["tokens"][i, length:] == 0)


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
