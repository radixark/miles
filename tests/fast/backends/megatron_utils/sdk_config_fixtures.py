from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.model_companion import ModelCompanion
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.tracking_utils import tracking
from miles.utils.tracking_utils.base import TrackingManager


class _SdkModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(no_sync_func=None, grad_sync_func=None, param_sync_func=None)
        self.zeroed = 0
        self.add_module("model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0)))

    def zero_grad_buffer(self) -> None:
        self.zeroed += 1

    def no_sync(self) -> str:
        return "no-sync"

    def start_grad_sync(self) -> str:
        return "grad-sync"

    def start_param_sync(self) -> str:
        return "param-sync"


@dataclass
class _ModelSdk:
    module: Any
    model: _SdkModel
    scheduler: Mock
    optimizer: Mock
    muon: Mock
    pipeline_calls: list[dict[str, Any]] = field(default_factory=list)
    pipeline_outputs: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def model_sdk(monkeypatch: pytest.MonkeyPatch) -> _ModelSdk:
    from miles.backends.megatron_utils import model as module

    model = _SdkModel()
    sdk = _ModelSdk(module=module, model=model, scheduler=Mock(), optimizer=Mock(), muon=Mock())
    trivial = GroupInfo(rank=0, size=1, group=None)
    state = parallel.ParallelState(
        intra_dp=trivial,
        intra_dp_cp=trivial,
        cp=trivial,
        tp=trivial,
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
    )
    monkeypatch.setattr(parallel, "_parallel_state", state)
    monkeypatch.setattr(tracking, "_manager", TrackingManager({}))
    monkeypatch.setattr(module, "get_model", lambda *_args, **_kwargs: [model])
    monkeypatch.setattr(module, "get_model_config", lambda chunk: chunk.config)
    monkeypatch.setattr(module, "DDP", _SdkModel)
    monkeypatch.setattr(module, "OptimizerParamScheduler", sdk.scheduler)
    monkeypatch.setattr(module, "get_megatron_optimizer", sdk.optimizer)
    monkeypatch.setattr(module, "get_megatron_muon_optimizer", sdk.muon)
    monkeypatch.setattr(module.mpu, "is_pipeline_last_stage", lambda **_kwargs: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda _group, rank: rank)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _group=None: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 1)

    def pipeline(**kwargs: Any) -> list[dict[str, Any]]:
        sdk.pipeline_calls.append(kwargs)
        return sdk.pipeline_outputs

    monkeypatch.setattr(module, "get_forward_backward_func", lambda: pipeline)
    return sdk
