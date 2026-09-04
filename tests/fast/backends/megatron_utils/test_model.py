import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch


def test_deterministic_optimizer_scopes_fp64_grad_norm_to_instance(monkeypatch) -> None:
    """Deterministic clipping must use FP64 without changing other optimizer instances."""
    from megatron.core.optimizer import optimizer as megatron_optimizer_module
    from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer

    from miles.backends.megatron_utils.model import _use_deterministic_grad_norm

    reduced_dtypes: list[torch.dtype] = []

    def fake_all_reduce(tensor: torch.Tensor, group: object = None) -> None:
        reduced_dtypes.append(tensor.dtype)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    original_grad_norm = megatron_optimizer_module.get_grad_norm_fp32
    config = object()

    class FakeChildOptimizer:
        get_grad_norm = MegatronOptimizer.get_grad_norm

        def __init__(self, grad_stats_parallel_group: object = None) -> None:
            self.config = config
            self.is_stub_optimizer = False
            self._grad_stats_parallel_group = grad_stats_parallel_group

        def get_grads_for_grad_norm(self, grad_norm_group: str | None = None) -> list[torch.Tensor]:
            return [torch.tensor([3.0, 4.0]), torch.tensor([12.0])]

        def get_grad_stats_parallel_group(self) -> object:
            return self._grad_stats_parallel_group

    child_optimizer = FakeChildOptimizer()
    original_optimizer = ChainedOptimizer([child_optimizer])
    original_marker = object()
    original_optimizer.marker = original_marker
    other_optimizer = ChainedOptimizer([child_optimizer])
    nonshared_optimizer = ChainedOptimizer(
        [
            child_optimizer,
            FakeChildOptimizer(grad_stats_parallel_group=object()),
        ]
    )

    class SpecializedChainedOptimizer(ChainedOptimizer):
        pass

    specialized_optimizer = SpecializedChainedOptimizer([child_optimizer])
    single_specialized_child_optimizer = ChainedOptimizer([specialized_optimizer])
    empty_optimizer = ChainedOptimizer([])
    deterministic_optimizer = _use_deterministic_grad_norm(original_optimizer)

    assert deterministic_optimizer is original_optimizer
    assert deterministic_optimizer.marker is original_marker
    assert deterministic_optimizer.get_grad_norm() == 13.0
    assert reduced_dtypes == [torch.float64]
    assert other_optimizer.get_grad_norm.__func__ is ChainedOptimizer.get_grad_norm
    assert _use_deterministic_grad_norm(nonshared_optimizer) is nonshared_optimizer
    assert nonshared_optimizer.get_grad_norm.__func__ is ChainedOptimizer.get_grad_norm
    assert _use_deterministic_grad_norm(specialized_optimizer) is specialized_optimizer
    assert specialized_optimizer.get_grad_norm.__func__ is ChainedOptimizer.get_grad_norm
    assert _use_deterministic_grad_norm(single_specialized_child_optimizer) is single_specialized_child_optimizer
    assert single_specialized_child_optimizer.get_grad_norm.__func__ is ChainedOptimizer.get_grad_norm
    assert _use_deterministic_grad_norm(empty_optimizer) is empty_optimizer
    assert empty_optimizer.get_grad_norm.__func__ is ChainedOptimizer.get_grad_norm
    assert megatron_optimizer_module.get_grad_norm_fp32 is original_grad_norm


class FakeModelChunk:
    def __init__(self) -> None:
        self.zero_grad_buffer_count = 0

    def zero_grad_buffer(self) -> None:
        self.zero_grad_buffer_count += 1


class FakeDataIterator:
    def __init__(self) -> None:
        self.batches: list[dict[str, Any]] = []


class FakeMpu:
    def __init__(self, *, is_last_pipeline_stage: bool = False) -> None:
        self.is_last_pipeline_stage = is_last_pipeline_stage

    def is_pipeline_last_stage(self, ignore_virtual: bool = False) -> bool:
        return self.is_last_pipeline_stage


class FakeForwardBackwardEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return []


@dataclass
class FakeParallelGroup:
    size: int = 1
    rank: int = 0


@dataclass
class FakeParallelState:
    indep_dp: FakeParallelGroup = field(default_factory=FakeParallelGroup)
    effective_dp: FakeParallelGroup = field(default_factory=FakeParallelGroup)


@dataclass
class TrainOneStepEnv:
    args: Namespace
    model: list[FakeModelChunk]
    data_iterator: list[FakeDataIterator]
    parallel_state: FakeParallelState
    forward_backward_engine: FakeForwardBackwardEngine


def make_train_one_step_args(**overrides: Any) -> Namespace:
    defaults: dict[str, Any] = dict(
        debug_disable_optimizer=False,
        multi_lora=False,
        custom_megatron_before_train_step_hook_path=None,
        dumper_enable=False,
        dumper_fwd_bwd=[],
        seq_length=8,
        decoder_seq_length=8,
        micro_batch_size=1,
        check_for_nan_in_loss_and_grad=True,
        ci_test=False,
        enable_mtp_training=False,
        rollout_max_response_len=512,
        enable_witness=False,
        save_local_weight_checksum=False,
    )
    return Namespace(**{**defaults, **overrides})


@pytest.fixture
def train_one_step_env(monkeypatch) -> TrainOneStepEnv:
    from miles.backends.megatron_utils import model as model_module

    env = TrainOneStepEnv(
        args=make_train_one_step_args(),
        model=[FakeModelChunk()],
        data_iterator=[FakeDataIterator()],
        parallel_state=FakeParallelState(),
        forward_backward_engine=FakeForwardBackwardEngine(),
    )

    monkeypatch.setattr(model_module, "get_args", lambda: env.args)
    monkeypatch.setattr(model_module, "get_parallel_state", lambda: env.parallel_state)
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: env.forward_backward_engine)
    monkeypatch.setattr(model_module, "mpu", FakeMpu())

    return env


class TestTrainOneStepStructuredLog:
    def test_train_one_step_emits_the_train_tag_in_its_structured_event(
        self, train_one_step_env: TrainOneStepEnv, caplog
    ):
        """Log consumers key train-step events off the train tag, so the caller must emit that tag with its fields."""
        from miles.backends.megatron_utils.model import train_one_step

        with caplog.at_level(logging.INFO, logger="miles.backends.megatron_utils.model"):
            train_one_step(
                args=train_one_step_env.args,
                rollout_id=7,
                step_id=3,
                data_iterator=train_one_step_env.data_iterator,
                model=train_one_step_env.model,
                optimizer=None,
                opt_param_scheduler=None,
                num_microbatches=1,
                num_rollouts=1,
                witness_info=None,
                attempt=2,
            )

        assert "train op=train_step rollout=7 step=3 attempt=2 outcome=NORMAL valid_step=true" in caplog.messages
