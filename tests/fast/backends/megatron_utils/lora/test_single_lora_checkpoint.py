"""Bridge single LoRA's --save / --lora-adapter-path glue around the adapter library."""

import os
from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch

import pytest
import torch

from megatron.core.optimizer import DistributedOptimizer

from miles.backends.megatron_utils.lora.checkpoint import (
    DistributedOptimizerMembers,
    restore_single_lora,
    single_lora_parametrization,
)
from miles.backends.training_utils.checkpoint.adapter.readers import AdapterSourceKind

_MODULE = "miles.backends.megatron_utils.lora.checkpoint"


def _base_model(hf_checkpoint: str) -> str:
    slots = MagicMock()
    args = SimpleNamespace(
        hf_checkpoint=hf_checkpoint, lora_rank=8, lora_alpha=16.0, lora_type="lora", experts_shared_outer_loras=False
    )
    single_lora_parametrization(slots, args)
    return slots.parametrization.call_args.kwargs["base_model"]


def test_one_local_model_under_another_path_string_is_the_same_base(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (tmp_path / "mount").symlink_to(model)
    expected = os.path.realpath(model)
    assert _base_model(str(model)) == _base_model(f"{model}/") == _base_model(str(tmp_path / "mount")) == expected


def test_a_hub_id_is_kept():
    assert _base_model("Qwen/Qwen3-8B") == "Qwen/Qwen3-8B"


def _restore(path, *, convert_save=None, megatron_to_hf_mode="bridge"):
    """``restore_single_lora`` with the model-facing pieces mocked; returns (iteration, scheduler, mocks)."""
    args = SimpleNamespace(
        hf_checkpoint="/hf",
        lora_adapter_path=str(path),
        lora_adapter_convert_save=convert_save,
        lora_rank=8,
        lora_alpha=16.0,
        lora_type="lora",
        experts_shared_outer_loras=False,
        megatron_to_hf_mode=megatron_to_hf_mode,
        multi_lora=False,
        backend=SimpleNamespace(no_load_optim=True, no_save_optim=False, use_precision_aware_optimizer=False),
    )
    scheduler = MagicMock()
    with (
        patch("miles.backends.megatron_utils.update_weight.hf_weight_iterator.get_hf_weight_iterator"),
        patch("miles.backends.megatron_utils.lora.native.checkpoint.NativeAdapterSlots"),
        patch.multiple(
            _MODULE,
            get_hf_lora_targets=DEFAULT,
            load_hf_config=DEFAULT,
            BridgeAdapterSlots=DEFAULT,
            DistributedOptimizerMembers=DEFAULT,
            load_adapter_dir=DEFAULT,
            save_and_commit_adapter_dir=DEFAULT,
        ) as mocks,
    ):
        iteration = restore_single_lora([], args, optimizer=MagicMock(), opt_param_scheduler=scheduler)
    return iteration, scheduler, mocks


def test_an_old_format_save_resumes_its_iteration_whichever_reader_loads_the_weights(tmp_path):
    # --no-load-optim reads an old-format save's PEFT pair, not its rank shards
    torch.save(
        {"iteration": 7, "optimizer": None, "parameter_state": None, "opt_param_scheduler": {"num_steps": 56}},
        tmp_path / "training_state_rank0.pt",
    )
    iteration, scheduler, _ = _restore(tmp_path)
    assert iteration == 7
    scheduler.load_state_dict.assert_called_once_with({"num_steps": 56})


def test_a_source_without_training_state_restarts_at_rollout_0(tmp_path):
    iteration, scheduler, _ = _restore(tmp_path)
    assert iteration is None
    scheduler.load_state_dict.assert_not_called()


def test_a_weights_only_convert_save_writes_neither_moments_nor_a_placeholder_training_state(tmp_path):
    _, _, mocks = _restore(tmp_path / "source", convert_save=str(tmp_path / "converted"))
    assert mocks["DistributedOptimizerMembers"].call_args.kwargs["save_moments"] is False
    assert mocks["save_and_commit_adapter_dir"].call_args.kwargs["training_state"] is None


@pytest.mark.parametrize("megatron_to_hf_mode, loads_bridge", [("bridge", True), ("raw", False)])
def test_only_the_bridge_backend_loads_a_megatron_bridge_checkpoint(tmp_path, megatron_to_hf_mode, loads_bridge):
    _, _, mocks = _restore(tmp_path, megatron_to_hf_mode=megatron_to_hf_mode)
    loaders = mocks["load_adapter_dir"].call_args.kwargs["in_model_loaders_by_kind"]
    assert (AdapterSourceKind.MEGATRON_BRIDGE_CHECKPOINT in loaders) is loads_bridge


def test_a_mix_of_distributed_and_plain_optimizers_is_refused():
    distributed = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer = SimpleNamespace(chained_optimizers=[distributed, MagicMock()])
    members = DistributedOptimizerMembers(MagicMock(), optimizer, [], precision_aware=False)
    with pytest.raises(AssertionError, match="one kind of optimizer"):
        members._distributed_optimizers()


def test_a_plain_optimizer_is_checkpointed_by_its_replicated_masters():
    """miles runs SGD without the DistributedOptimizer; its masters are on every rank."""
    members = DistributedOptimizerMembers.__new__(DistributedOptimizerMembers)
    members._optimizer = SimpleNamespace(chained_optimizers=[object(), object()])
    assert members._distributed_optimizers() == []
