"""LoRA checkpoint coverage for distributed optimizer parameter state."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer

import miles.backends.megatron_utils.lora.utils as lora_utils


class _Child(DistributedOptimizer):
    def __init__(self, *, stub=False, dp_rank=0, stepped=True):
        self.is_stub_optimizer = stub
        self.data_parallel_group = SimpleNamespace(rank=lambda: dp_rank)
        self.optimizer = None if stub else SimpleNamespace(param_groups=[], state={"param": {}} if stepped else {})
        self.loaded_state_dict = "not called"
        self.param_state = "not called"

    def state_dict(self):
        assert not self.is_stub_optimizer
        return {"step": 3}

    def load_state_dict(self, state_dict):
        assert not self.is_stub_optimizer
        self.loaded_state_dict = state_dict

    def get_parameter_state_dp_zero(self):
        return {"master": 1} if self.data_parallel_group.rank() == 0 else None

    def load_parameter_state(self, filename):
        self.param_state = torch.load(filename) if self.data_parallel_group.rank() == 0 else None


def _chain(*children):
    chain = ChainedOptimizer.__new__(ChainedOptimizer)
    chain.chained_optimizers = list(children)
    return chain


def _save(tmp_path, optimizer, scheduler=None):
    args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=False)
    publisher = SimpleNamespace(write_adapter=lambda *_: None)
    lora_utils.save_lora_checkpoint(
        [], args, str(tmp_path), publisher=publisher, optimizer=optimizer, opt_param_scheduler=scheduler, iteration=3
    )


def test_round_trip_skips_stub_children_and_reads_through_the_data_parallel_root(tmp_path):
    scheduler = SimpleNamespace(state_dict=lambda: {"num_steps": 8})
    _save(tmp_path, _chain(_Child(dp_rank=0), _Child(stub=True), _Child(dp_rank=1)), scheduler)
    assert [p.name for p in tmp_path.glob("optimizer_param_state*")] == ["optimizer_param_state_rank0_optimizer0.pt"]

    root, stub, peer = _Child(dp_rank=0), _Child(stub=True), _Child(dp_rank=1)
    scheduler = MagicMock()
    assert lora_utils._load_training_state(tmp_path, _chain(root, stub, peer), scheduler) == (3, True)

    assert (root.loaded_state_dict, peer.loaded_state_dict) == ({"step": 3}, {"step": 3})
    assert (root.param_state, peer.param_state, stub.param_state) == ({"master": 1}, None, "not called")
    scheduler.load_state_dict.assert_called_once_with({"num_steps": 8})


def test_parameter_state_is_gathered_before_a_local_write_can_fail(tmp_path, monkeypatch):
    """write_checkpoint_dir requires every rank to finish its collectives before raising."""
    events = []
    child = _Child(dp_rank=1)
    monkeypatch.setattr(child, "get_parameter_state_dp_zero", lambda: events.append("gather"))

    def failing_save(*_):
        events.append("write")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        _save(tmp_path, _chain(child))
    assert events == ["gather", "write"]


def test_an_optimizer_that_never_stepped_saves_no_parameter_state(tmp_path):
    child = _Child(stepped=False)
    child.get_parameter_state_dp_zero = MagicMock()

    _save(tmp_path, _chain(child))

    child.get_parameter_state_dp_zero.assert_not_called()
    assert not list(tmp_path.glob("optimizer_param_state*"))


def test_checkpoint_without_parameter_state_keeps_the_fresh_optimizer(tmp_path):
    child = _Child(dp_rank=0)
    torch.save({"iteration": 3, "optimizer": [{"step": 3}]}, tmp_path / "training_state_rank0.pt")

    assert lora_utils._load_training_state(tmp_path, _chain(child), None) == (3, False)
    assert (child.loaded_state_dict, child.param_state) == ("not called", "not called")


def test_partial_parameter_state_is_rejected(tmp_path):
    optimizer = _chain(_Child(dp_rank=0), _Child(dp_rank=0))
    torch.save({"iteration": 3, "optimizer": [{"step": 3}, {"step": 3}]}, tmp_path / "training_state_rank0.pt")
    (tmp_path / "optimizer_param_state_rank0_optimizer1.pt").touch()

    with pytest.raises(RuntimeError, match="Optimizer parameter state is incomplete"):
        lora_utils._load_training_state(tmp_path, optimizer, None)


def test_loading_adapter_weights_refreshes_the_masters(tmp_path, monkeypatch):
    rank0 = SimpleNamespace(rank=0)
    monkeypatch.setattr(lora_utils, "get_parallel_state", lambda: SimpleNamespace(tp=rank0, pp=rank0))
    param = torch.nn.Parameter(torch.zeros(2))
    model = [SimpleNamespace(named_parameters=lambda: iter([("adapter.lora_A.weight", param)]))]
    torch.save({"adapter.lora_A.weight": torch.ones(2)}, tmp_path / "adapter_megatron_rank0.pt")
    refreshed_from = []
    optimizer = MagicMock()
    optimizer.reload_model_params.side_effect = lambda: refreshed_from.append(param.detach().clone())

    assert lora_utils.load_lora_adapter(model, str(tmp_path), optimizer=optimizer) == (True, None, False)
    assert len(refreshed_from) == 1 and torch.equal(refreshed_from[0], torch.ones(2))
