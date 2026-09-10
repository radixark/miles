from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from miles.backends.training_utils.data import DataIterator
from miles.utils.audit_utils.witness.cpu import CpuWitness


class _CpuTrainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.add_module("cpu_witness", CpuWitness(pipeline_rank=0, chunk_index=0, replica_id=(0,)))

    def zero_grad_buffer(self) -> None:
        pass


class _CpuOptimizer:
    def __init__(self, model: _CpuTrainModel) -> None:
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.found_inf = False

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def prepare_grads(self) -> bool:
        return self.found_inf

    def get_grad_norm(self) -> float:
        return 1.0

    def step(self) -> tuple[bool, float, int]:
        self.optimizer.step()
        return True, 1.0, 0


class _CpuScheduler:
    def step(self, *, increment: int) -> None:
        pass


class _CpuForwardBackward:
    def __call__(
        self, *, data_iterator: list[DataIterator], model: list[_CpuTrainModel], num_microbatches: int, **kwargs: Any
    ) -> list:
        for iterator in data_iterator:
            for _ in range(num_microbatches):
                iterator.get_next(["sample_indices"])
        for chunk in model:
            chunk.weight.grad = torch.ones_like(chunk.weight)
        return []


@dataclass
class _CpuStepEnvironment:
    args: Namespace
    model: _CpuTrainModel
    iterator: DataIterator
    optimizer: _CpuOptimizer
    scheduler: _CpuScheduler


@pytest.fixture
def cpu_step_environment(monkeypatch: pytest.MonkeyPatch) -> _CpuStepEnvironment:
    from miles.backends.megatron_utils import model as model_module
    from miles.backends.training_utils import parallel

    args = Namespace(
        debug_disable_optimizer=False,
        multi_lora=False,
        custom_megatron_before_train_step_hook_path=None,
        dumper_enable=False,
        dumper_fwd_bwd=[],
        seq_length=8,
        decoder_seq_length=8,
        micro_batch_size=1,
        check_for_nan_in_loss_and_grad=False,
        ci_test=False,
        enable_mtp_training=False,
        rollout_max_response_len=512,
        enable_witness=False,
        save_local_weight_checksum=False,
        trainer_model_id=None,
    )
    model = _CpuTrainModel()
    iterator = DataIterator(
        rollout_data=dict(sample_indices=[10, 11, 12], group_indices=[7, 7, 8], ownership_lineage_id="run"),
        micro_batch_size=1,
    )
    state = Namespace(indep_dp=Namespace(size=1), effective_dp=Namespace(size=1))
    monkeypatch.setattr(parallel, "_parallel_state", state)
    monkeypatch.setattr(model_module, "get_args", lambda: args)
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: _CpuForwardBackward())
    monkeypatch.setattr(model_module.mpu, "is_pipeline_last_stage", lambda **kwargs: False)
    return _CpuStepEnvironment(
        args=args, model=model, iterator=iterator, optimizer=_CpuOptimizer(model), scheduler=_CpuScheduler()
    )
