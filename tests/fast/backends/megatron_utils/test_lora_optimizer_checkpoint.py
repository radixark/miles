from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import miles.backends.megatron_utils.lora_utils as lora_utils


def _optimizer_child(*, stub=False, dp_rank=0):
    child = MagicMock(is_stub_optimizer=stub)
    child.data_parallel_group.rank.return_value = dp_rank
    return child


def _args(**overrides):
    values = dict(
        use_distributed_optimizer=True,
        no_load_optim=False,
        finetune=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_parameter_state_uses_active_children_and_dp_root_files(tmp_path, monkeypatch):
    writer = _optimizer_child(dp_rank=0)
    stub = _optimizer_child(stub=True)
    non_writer = _optimizer_child(dp_rank=1)
    optimizer = MagicMock(chained_optimizers=[writer, stub, non_writer])

    lora_utils._save_optimizer_param_state(_args(), optimizer, tmp_path)
    writer.save_parameter_state.assert_called_once()
    non_writer.save_parameter_state.assert_called_once()
    stub.save_parameter_state.assert_not_called()

    torch.save(
        {
            "optimizer_children": {0: {"step": 3}, 2: {"step": 4}},
            "opt_param_scheduler": {"num_steps": 8},
            "iteration": 3,
        },
        tmp_path / "training_state_rank0.pt",
    )
    writer_path = tmp_path / "optimizer_param_state_rank0_optimizer0.pt"
    torch.save({"master": 1}, writer_path)
    active_chain = MagicMock()
    monkeypatch.setattr(
        "megatron.core.optimizer.optimizer.ChainedOptimizer",
        lambda children: active_chain if children == [writer, non_writer] else None,
    )
    scheduler = MagicMock()

    iteration = lora_utils._load_training_state(tmp_path, _args(), optimizer, scheduler)

    assert iteration == 3
    active_chain.load_state_dict.assert_called_once_with([{"step": 3}, {"step": 4}])
    writer.load_parameter_state_from_dp_zero.assert_called_once_with({"master": 1})
    non_writer.load_parameter_state_from_dp_zero.assert_called_once_with(None)
    stub.load_parameter_state_from_dp_zero.assert_not_called()
    scheduler.load_state_dict.assert_called_once_with({"num_steps": 8})


def test_incomplete_parameter_state_warm_starts_without_partial_restore(tmp_path):
    writer = _optimizer_child(dp_rank=0)
    optimizer = MagicMock(chained_optimizers=[writer])
    torch.save(
        {
            "optimizer": {"step": 3},
            "opt_param_scheduler": {"num_steps": 8},
            "iteration": 3,
        },
        tmp_path / "training_state_rank0.pt",
    )
    scheduler = MagicMock()

    assert lora_utils._load_training_state(tmp_path, _args(), optimizer, scheduler) is None
    optimizer.load_state_dict.assert_not_called()
    writer.load_parameter_state_from_dp_zero.assert_not_called()
    scheduler.load_state_dict.assert_not_called()


@pytest.mark.parametrize(
    ("overrides", "expected_iteration"),
    [
        ({"no_load_optim": True}, 3),
        ({"finetune": True}, None),
    ],
)
def test_resume_flags_skip_optimizer_restore(tmp_path, overrides, expected_iteration):
    optimizer = MagicMock()
    scheduler = MagicMock()
    if not overrides.get("finetune"):
        torch.save(
            {
                "optimizer": {"step": 3},
                "opt_param_scheduler": {"num_steps": 8},
                "iteration": 3,
            },
            tmp_path / "training_state_rank0.pt",
        )

    assert lora_utils._load_training_state(tmp_path, _args(**overrides), optimizer, scheduler) == expected_iteration
    optimizer.load_state_dict.assert_not_called()
    scheduler.load_state_dict.assert_not_called()


def test_changed_active_child_layout_stops_before_parameter_restore(tmp_path):
    stub = _optimizer_child(stub=True)
    active = _optimizer_child(dp_rank=0)
    optimizer = MagicMock(chained_optimizers=[stub, active])
    torch.save(
        {
            "optimizer_children": {0: {"step": 3}},
            "opt_param_scheduler": None,
            "iteration": 3,
        },
        tmp_path / "training_state_rank0.pt",
    )

    with pytest.raises(RuntimeError, match=r"Active optimizer children changed.*saved=\[0\], current=\[1\]"):
        lora_utils._load_training_state(tmp_path, _args(), optimizer, None)

    active.load_parameter_state_from_dp_zero.assert_not_called()


def test_native_adapter_read_failure_is_coordinated_before_optimizer_restore(tmp_path, monkeypatch):
    path = tmp_path / "adapter_megatron_rank0.pt"
    path.write_bytes(b"corrupt")
    monkeypatch.setattr(
        lora_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(rank=0), pp=SimpleNamespace(rank=0)),
    )
    model = MagicMock()
    model.named_parameters.return_value = []

    with pytest.raises(RuntimeError, match="Failed to read native LoRA adapter state"):
        lora_utils.load_lora_adapter([model], Namespace(), str(tmp_path))
