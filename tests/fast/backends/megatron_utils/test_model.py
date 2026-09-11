import logging
from argparse import Namespace
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch


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


@pytest.mark.parametrize("tensor_norm", [False, True])
def test_ft_discard_stays_invalid_with_finite_gradient_norm(
    train_one_step_env: TrainOneStepEnv, monkeypatch: pytest.MonkeyPatch, tensor_norm: bool
) -> None:
    """A finite gradient norm cannot revive a failed collective and update weights."""
    from miles.backends.megatron_utils import model as model_module

    env = train_one_step_env
    env.args.enable_sample_ownership_checker = False
    env.args.check_for_nan_in_loss_and_grad = False
    env.args.calculate_per_token_loss = False
    env.parallel_state.indep_dp.size = 2

    def reject_optimizer_step() -> None:
        raise AssertionError("A discarded step must not update weights")

    optimizer = SimpleNamespace(
        zero_grad=lambda: None,
        prepare_grads=lambda: False,
        get_grad_norm=lambda: torch.tensor(1.0) if tensor_norm else 1.0,
        step=reject_optimizer_step,
    )
    monkeypatch.setattr(
        model_module, "allreduce_grads_and_losses_across_replicas", lambda *args, **kwargs: (False, {})
    )

    _, _, outcome = model_module.train_one_step(
        args=env.args,
        rollout_id=7,
        step_id=0,
        data_iterator=env.data_iterator,
        model=env.model,
        optimizer=optimizer,
        opt_param_scheduler=None,
        num_microbatches=1,
        num_rollouts=1,
        witness_info=None,
        attempt=2,
    )

    assert outcome is model_module.TrainStepOutcome.DISCARDED_SHOULD_RETRY
