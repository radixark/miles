"""Multimodal positions index the natural packed row, including per-sample CP padding."""

from types import SimpleNamespace

import pytest
import torch

import miles.backends.training_utils.data.context_parallel as cp_utils_mod
import miles.backends.training_utils.data.rollout as data_mod
from miles.backends.training_utils.data.rollout import DataIterator, get_batch


@pytest.mark.parametrize(
    "cp_size,cp_rank,allgather_cp,offsets",
    [
        (2, 0, False, [0, 8, 16]),
        (2, 1, False, [0, 8, 16]),
        (1, 0, False, [0, 5, 12]),
        (2, 0, True, [0, 5, 12]),
        (2, 1, True, [0, 5, 12]),
    ],
)
def test_thd_multimodal_positions(monkeypatch, cp_size, cp_rank, allgather_cp, offsets):
    state = SimpleNamespace(cp=SimpleNamespace(size=cp_size, rank=cp_rank), tp=SimpleNamespace(size=1))
    monkeypatch.setattr(data_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils_mod, "get_parallel_state", lambda: state)
    monkeypatch.setattr(data_mod.current_platform, "current_device", lambda: torch.device("cpu"))

    # The text-only middle sample still advances the packed offset. Ragged samples
    # need zigzag padding independently of the final stream padding.
    lengths = [5, 7, 3]
    first = {
        "image_positions": torch.tensor([1, 3]),
        "audio_positions": torch.tensor([2]),
        "image_features": torch.tensor([[10.0], [11.0]]),
    }
    last = {
        "image_positions": torch.tensor([0, 1]),
        "audio_positions": torch.tensor([2]),
        "image_features": torch.tensor([[20.0], [21.0]]),
    }
    rollout_data = {
        "tokens": [torch.arange(length) for length in lengths],
        "loss_masks": [torch.ones(1, dtype=torch.int) for _ in lengths],
        "total_lengths": lengths,
        "response_lengths": [1, 1, 1],
        "multimodal_train_inputs": [first, None, last],
    }
    batch = get_batch(
        DataIterator(rollout_data, micro_batch_size=3),
        list(rollout_data),
        pad_multiplier=8,
        qkv_format="thd",
        allgather_cp=allgather_cp,
    )

    assert batch["cu_seqlens"].tolist()[:3] == offsets
    mm = batch["multimodal_train_inputs"]
    assert mm["image_positions"].tolist() == [1, 3, offsets[2], offsets[2] + 1]
    assert mm["audio_positions"].tolist() == [2, offsets[2] + 2]
    assert mm["image_features"].tolist() == [[10.0], [11.0], [20.0], [21.0]]
    assert batch["multimodal_num_items"] == {
        "image_positions": [2, 2],
        "audio_positions": [1, 1],
        "image_features": [2, 2],
    }
    assert last["image_positions"].tolist() == [0, 1]
    assert last["audio_positions"].tolist() == [2]
