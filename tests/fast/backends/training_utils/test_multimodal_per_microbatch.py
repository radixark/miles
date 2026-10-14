"""Screenshot pixels reach the GPU one micro-batch at a time, not the whole data-parallel share at load."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils

_LENGTHS = [7, 5, 6]


def _mm(patches: int, value: float) -> dict[str, torch.Tensor]:
    return {"pixel_values": torch.full((patches, 4), value), "image_grid_thw": torch.tensor([[1, 2, patches // 2]])}


def _rollout() -> dict:
    return {
        "tokens": [torch.arange(1, n + 1) for n in _LENGTHS],
        "loss_masks": [torch.ones(2, dtype=torch.int) for _ in _LENGTHS],
        "total_lengths": list(_LENGTHS),
        "response_lengths": [2] * len(_LENGTHS),
        "multimodal_train_inputs": [_mm(4, 0.0), None, _mm(6, 2.0)],
    }


@pytest.fixture
def single_rank(monkeypatch):
    state = SimpleNamespace(
        cp=SimpleNamespace(rank=0, size=1),
        tp=SimpleNamespace(rank=0, size=1),
        effective_dp=SimpleNamespace(rank=0, size=1),
    )
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: state)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)


def test_get_rollout_data_leaves_multimodal_inputs_on_the_host(monkeypatch, single_rank):
    rollout = _rollout()
    host_inputs = rollout["multimodal_train_inputs"]
    monkeypatch.setattr(data_utils, "process_rollout_data", lambda *args, **kwargs: (rollout, None))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    data, _ = data_utils.get_rollout_data(Namespace(enable_witness=False, qkv_format="thd"), rollout_data_ref=None)

    assert data["multimodal_train_inputs"] is host_inputs
    assert data["multimodal_train_inputs"][0]["pixel_values"] is host_inputs[0]["pixel_values"]


@pytest.mark.parametrize("multimodal_on_host", [False, True])
def test_get_batch_moves_only_its_micro_batch(monkeypatch, single_rank, multimodal_on_host):
    # "meta" stands in for the GPU: a tensor that reached it shows the move without CUDA
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("meta"))
    rollout = _rollout()
    iterator = data_utils.DataIterator(rollout, micro_batch_indices=[[0, 1], [2]])
    keys = ["tokens", "loss_masks", "multimodal_train_inputs", "total_lengths", "response_lengths"]

    first = data_utils.get_batch(iterator, keys, pad_multiplier=1, multimodal_on_host=multimodal_on_host)
    second = data_utils.get_batch(iterator, keys, pad_multiplier=1, multimodal_on_host=multimodal_on_host)

    expected_device = torch.device("cpu") if multimodal_on_host else torch.device("meta")
    for batch, patches in ((first, 4), (second, 6)):
        for tensor in batch["multimodal_train_inputs"].values():
            assert tensor.device == expected_device
        assert batch["multimodal_train_inputs"]["pixel_values"].shape == (patches, 4)
    if multimodal_on_host:
        assert second["multimodal_train_inputs"]["pixel_values"].eq(2.0).all()
    # the rollout's own copies are never moved
    assert all(mm is None or mm["pixel_values"].device.type == "cpu" for mm in rollout["multimodal_train_inputs"])
