"""LoRA checkpoint coverage for distributed optimizer parameter state."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

import miles.backends.megatron_utils.lora_utils as lora_utils


class _Child(DistributedOptimizer):
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


def _parameter_state(numel):
    return {
        0: {
            (torch.bfloat16, torch.bfloat16): {
                "numel_unpadded": numel,
                "param": torch.zeros(numel),
                "exp_avg": torch.zeros(numel),
                "exp_avg_sq": torch.zeros(numel),
            }
        }
    }


def test_training_state_without_stubs_keeps_megatron_format():
    optimizer = MagicMock()
    optimizer.state_dict.return_value = {"step": 11}

    assert lora_utils._optimizer_training_state_dict(optimizer) == {"step": 11}


def test_training_state_skips_stub_optimizer_children():
    active = MagicMock(is_stub_optimizer=False)
    active.state_dict.return_value = {"step": 11}
    stub = MagicMock(is_stub_optimizer=True)
    optimizer = MagicMock(chained_optimizers=[stub, active])

    state = lora_utils._optimizer_training_state_dict(optimizer)

    assert state == {
        "format": "active_optimizer_children_v1",
        "active_child_indices": [1],
        "children": {1: {"step": 11}},
    }
    stub.state_dict.assert_not_called()


def test_training_state_restores_only_matching_active_children():
    stub = MagicMock(is_stub_optimizer=True)
    active = MagicMock(is_stub_optimizer=False)
    active.optimizer.param_groups = [{"params": [object()], "step": 11}]
    optimizer = MagicMock(chained_optimizers=[stub, active])
    state = {
        "format": "active_optimizer_children_v1",
        "active_child_indices": [1],
        "children": {1: {"step": 11}},
    }

    lora_utils._load_optimizer_training_state_dict(optimizer, state)

    stub.load_state_dict.assert_not_called()
    active.load_state_dict.assert_called_once_with({"step": 11})
    optimizer._synchronize_steps.assert_not_called()
    assert active.optimizer.param_groups[0]["step"] == 11


def test_training_state_rejects_changed_active_child_layout():
    optimizer = MagicMock(
        chained_optimizers=[
            MagicMock(is_stub_optimizer=False),
            MagicMock(is_stub_optimizer=True),
        ]
    )
    state = {
        "format": "active_optimizer_children_v1",
        "active_child_indices": [1],
        "children": {1: {"step": 11}},
    }

    with pytest.raises(RuntimeError, match="expected active indices \\[0\\], found \\[1\\]"):
        lora_utils._load_optimizer_training_state_dict(optimizer, state)


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
    monkeypatch.setattr(
        lora_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(rank=0), pp=SimpleNamespace(rank=0)),
    )
    param = torch.nn.Parameter(torch.zeros(2))
    model = [SimpleNamespace(named_parameters=lambda: iter([("adapter.lora_A.weight", param)]))]
    torch.save({"adapter.lora_A.weight": torch.ones(2)}, tmp_path / "adapter_megatron_rank0.pt")
    optimizer = MagicMock(chained_optimizers=[_Child(dp_rank=0)])

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
        _parameter_state(32),
        tmp_path / "optimizer_param_state_rank0_optimizer0.pt",
    )

    with pytest.raises(RuntimeError, match="Failed to read optimizer parameter state") as failure:
        lora_utils._load_training_state(tmp_path, optimizer, None)
    assert "does not match the model" in str(failure.value.__cause__)
    assert child.loaded == "not called"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("param", torch.zeros(63)),
        ("exp_avg", torch.zeros(64, dtype=torch.bfloat16)),
        ("exp_avg_sq", None),
    ],
)
def test_malformed_parameter_tensor_is_rejected_before_the_scatter(tmp_path, key, value):
    child = _Child(dp_rank=0, numel_unpadded=64)
    optimizer = MagicMock(chained_optimizers=[child])
    _write_training_state(tmp_path)
    state = _parameter_state(64)
    state[0][(torch.bfloat16, torch.bfloat16)][key] = value
    torch.save(state, tmp_path / "optimizer_param_state_rank0_optimizer0.pt")

    with pytest.raises(RuntimeError, match="Failed to read optimizer parameter state"):
        lora_utils._load_training_state(tmp_path, optimizer, None)
    assert child.loaded == "not called"
