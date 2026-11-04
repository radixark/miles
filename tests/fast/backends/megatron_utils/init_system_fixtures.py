from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from miles.utils.args.runtime import TrainerConfig


def _install_process_environment(
    *, actor_module: ModuleType, args: TrainerConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miles.backends.megatron_utils import initialize, parallel
    from miles.backends.training_utils import parallel as shared_parallel
    from miles.ray import train_actor
    from miles.utils import distributed_utils
    from miles.utils.replay_base import IndexerReplayManager, RoutingReplayManager
    from miles.utils.test_utils.fault_injector.controller import _FaultHookController

    group = object()
    monkeypatch.setattr(shared_parallel, "_parallel_state", None)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", None)
    monkeypatch.setattr(actor_module, "monkey_patch_torch_dist", lambda: None)
    monkeypatch.setattr(
        train_actor,
        "current_platform",
        SimpleNamespace(
            bind_device_before_process_group=lambda rank: None,
            bind_device_after_process_group=lambda: None,
            nvml_numa_affinity=False,
            device_type="cpu",
        ),
    )
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: None)
    monkeypatch.setattr(torch.distributed, "new_group", lambda *args, **kwargs: group)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *args, **kwargs: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *args, **kwargs: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.serialization, "add_safe_globals", lambda values: None)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(initialize, "set_args", lambda backend: None)
    monkeypatch.setattr(parallel, "get_args", lambda: args.backend)
    monkeypatch.setattr(initialize, "register_hf_config_aliases", lambda: None)
    monkeypatch.setattr(initialize, "init_num_microbatches_calculator", lambda *args, **kwargs: None)
    monkeypatch.setattr(initialize.random, "seed", lambda seed: None)
    monkeypatch.setattr(initialize.np.random, "seed", lambda seed: None)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(initialize.tensor_parallel, "model_parallel_cuda_manual_seed", lambda *args: None)
    monkeypatch.setattr(initialize, "_get_all_rng_states", lambda: ())
    monkeypatch.setattr(initialize, "_build_tokenizer", _build_tokenizer)
    monkeypatch.setattr(train_actor, "load_hf_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(train_actor, "load_tokenizer", lambda *args, **kwargs: SimpleNamespace(vocab_size=128))
    monkeypatch.setattr(actor_module, "fault_hook_controller", _FaultHookController())
    routing = RoutingReplayManager()
    monkeypatch.setattr(actor_module, "all_replay_managers", [routing, IndexerReplayManager()])
    monkeypatch.setattr(actor_module, "routing_replay_manager", routing)
    _install_megatron_groups(monkeypatch=monkeypatch, group=group)
    _install_torchft_groups(monkeypatch)


def _install_megatron_groups(*, monkeypatch: pytest.MonkeyPatch, group: object) -> None:
    from megatron.core import mpu

    monkeypatch.setattr(mpu, "initialize_model_parallel", lambda *args, **kwargs: None)
    for name in (
        "get_data_parallel_rank",
        "get_context_parallel_rank",
        "get_tensor_model_parallel_rank",
        "get_pipeline_model_parallel_rank",
        "get_expert_model_parallel_rank",
        "get_expert_tensor_parallel_rank",
        "get_expert_data_parallel_rank",
    ):
        monkeypatch.setattr(mpu, name, lambda *args, **kwargs: 0)
    for name in (
        "get_data_parallel_world_size",
        "get_context_parallel_world_size",
        "get_tensor_model_parallel_world_size",
        "get_pipeline_model_parallel_world_size",
        "get_expert_model_parallel_world_size",
        "get_expert_tensor_parallel_world_size",
        "get_expert_data_parallel_world_size",
    ):
        monkeypatch.setattr(mpu, name, lambda *args, **kwargs: 1)
    for name in (
        "get_tensor_and_data_parallel_group",
        "get_data_parallel_group",
        "get_data_parallel_group_gloo",
        "get_context_parallel_group",
        "get_tensor_model_parallel_group",
        "get_pipeline_model_parallel_group",
        "get_expert_model_parallel_group",
        "get_expert_tensor_parallel_group",
        "get_expert_data_parallel_group",
    ):
        monkeypatch.setattr(mpu, name, lambda *args, **kwargs: group)
    monkeypatch.setattr(mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda: True)


def _build_tokenizer(backend: Any) -> None:
    backend.padded_vocab_size = 128


class _TorchftGroup:
    def __init__(self, *, timeout: object) -> None:
        self.configuration: dict[str, Any] = {}
        self._replica_id: str | None = None
        self._rank = 0
        self._world_size = 0

    def configure(self, **kwargs: Any) -> None:
        self.configuration = kwargs
        self._replica_id = kwargs["replica_id"]
        self._rank = kwargs["rank"]
        self._world_size = kwargs["world_size"]

    def size(self) -> int:
        return self._world_size


def _install_torchft_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    package = ModuleType("torchft")
    package.__path__ = []
    process_group = ModuleType("torchft.process_group")
    process_group.ProcessGroupGloo = _TorchftGroup
    process_group.ProcessGroupNCCL = _TorchftGroup
    monkeypatch.setitem(sys.modules, "torchft", package)
    monkeypatch.setitem(sys.modules, "torchft.process_group", process_group)
