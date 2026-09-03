"""A DistributedOptimizer keeps its master weights and Adam moments out of ``state_dict()``,
so a LoRA checkpoint must save and restore them separately."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

import miles.backends.megatron_utils.lora_utils as lora_utils


class _Child(DistributedOptimizer):
    """Stands in for a real DistributedOptimizer, which needs a model and DDP grad buffers."""

    def __init__(self, *, stub=False, dp_rank=0, numel_unpadded=None):
        self.is_stub_optimizer = stub
        self.data_parallel_group = SimpleNamespace(rank=lambda: dp_rank)
        self.gbuf_ranges = [] if numel_unpadded is None else [{(torch.bfloat16, torch.bfloat16): None}]
        self.buffers = [SimpleNamespace(numel_unpadded=numel_unpadded, params=[torch.zeros(1)])]
        self.loaded = "not called"

    def save_parameter_state(self, filename):
        if self.data_parallel_group.rank() == 0:
            torch.save({"master": 1}, filename)

    def load_parameter_state_from_dp_zero(self, state_dict):
        self.loaded = state_dict


def _write_training_state(directory):
    torch.save(
        {"iteration": 3, "optimizer": {"step": 3}, "opt_param_scheduler": {"num_steps": 8}},
        directory / "training_state_rank0.pt",
    )


def test_parameter_state_round_trips_through_the_data_parallel_root(tmp_path):
    root, stub, peer = _Child(dp_rank=0), _Child(stub=True), _Child(dp_rank=1)
    optimizer = MagicMock(chained_optimizers=[root, stub, peer])

    lora_utils._save_optimizer_param_state(optimizer, tmp_path)
    assert [path.name for path in sorted(tmp_path.iterdir())] == ["optimizer_param_state_rank0_optimizer0.pt"]

    _write_training_state(tmp_path)
    scheduler = MagicMock()
    assert lora_utils._load_training_state(tmp_path, optimizer, scheduler) == 3

    optimizer.load_state_dict.assert_called_once_with({"step": 3})
    assert root.loaded == {"master": 1}
    # Every rank must join the scatter, even the ones that read nothing.
    assert peer.loaded is None
    assert stub.loaded == "not called"
    scheduler.load_state_dict.assert_called_once_with({"num_steps": 8})


def test_checkpoint_without_parameter_state_warm_starts_from_the_adapter(tmp_path):
    child = _Child(dp_rank=0)
    optimizer = MagicMock(chained_optimizers=[child])
    _write_training_state(tmp_path)

    assert lora_utils._load_training_state(tmp_path, optimizer, None) == 3
    assert child.loaded == "not called"


def test_masters_are_refreshed_whenever_adapter_weights_are_written(tmp_path, monkeypatch):
    """The masters are snapshotted at optimizer construction, before this load. Refresh them
    on every path that writes adapter weights, not only the ones that restore training state:
    otherwise the first step() copies the construction-time values back over the adapter."""
    monkeypatch.setattr(
        lora_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(rank=0), pp=SimpleNamespace(rank=0)),
    )
    param = torch.nn.Parameter(torch.zeros(2))
    model = [SimpleNamespace(named_parameters=lambda: iter([("adapter.lora_A.weight", param)]))]
    torch.save({"adapter.lora_A.weight": torch.ones(2)}, tmp_path / "adapter_megatron_rank0.pt")
    optimizer = MagicMock(chained_optimizers=[_Child(dp_rank=0)])

    # No training_state_rank0.pt: the weight-only warm start.
    loaded, iteration = lora_utils.load_lora_adapter(model, str(tmp_path), optimizer=optimizer)

    assert (loaded, iteration) == (True, None)
    optimizer.reload_model_params.assert_called_once_with()


def test_partial_parameter_state_is_rejected(tmp_path):
    optimizer = MagicMock(chained_optimizers=[_Child(dp_rank=0), _Child(dp_rank=0)])
    _write_training_state(tmp_path)
    (tmp_path / "optimizer_param_state_rank0_optimizer1.pt").touch()

    with pytest.raises(RuntimeError, match="Optimizer parameter state is incomplete"):
        lora_utils._load_training_state(tmp_path, optimizer, None)


def test_parameter_state_from_a_different_model_is_rejected_before_the_scatter(tmp_path):
    child = _Child(dp_rank=0, numel_unpadded=64)
    optimizer = MagicMock(chained_optimizers=[child])
    _write_training_state(tmp_path)
    torch.save(
        {0: {(torch.bfloat16, torch.bfloat16): {"numel_unpadded": 32}}},
        tmp_path / "optimizer_param_state_rank0_optimizer0.pt",
    )

    with pytest.raises(RuntimeError, match="Failed to read optimizer parameter state") as failure:
        lora_utils._load_training_state(tmp_path, optimizer, None)
    assert "does not match the model" in str(failure.value.__cause__)
    assert child.loaded == "not called"
