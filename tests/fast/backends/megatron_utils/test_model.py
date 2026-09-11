import logging
from argparse import Namespace
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from miles.backends.training_utils.data import DataIterator
from miles.backends.training_utils.model_companion import ModelCompanion


class FakeModelChunk(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.zero_grad_buffer_count = 0
        self.add_module("model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0)))

    def zero_grad_buffer(self) -> None:
        self.zero_grad_buffer_count += 1


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


class FakeNonfiniteOptimizer:
    def zero_grad(self) -> None:
        pass

    def prepare_grads(self) -> bool:
        return True


class FakeFiniteOptimizer(FakeNonfiniteOptimizer):
    def __init__(self) -> None:
        self.step_count = 0

    def prepare_grads(self) -> bool:
        return False

    def get_grad_norm(self) -> float:
        return 1.0

    def step(self) -> tuple[bool, float, int]:
        self.step_count += 1
        return True, 1.0, 0


class FakeScheduler:
    def step(self, *, increment: int) -> None:
        pass


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
    data_iterator: list[DataIterator]
    parallel_state: FakeParallelState
    forward_backward_engine: FakeForwardBackwardEngine


def make_train_one_step_args(**overrides: Any) -> Namespace:
    defaults: dict[str, Any] = dict(
        debug_disable_optimizer=False,
        multi_lora=False,
        enable_sample_ownership_checker=True,
        custom_megatron_before_train_step_hook_path=None,
        dumper_enable=False,
        dumper_fwd_bwd=[],
        seq_length=8,
        decoder_seq_length=8,
        micro_batch_size=1,
        check_for_nan_in_loss_and_grad=True,
        calculate_per_token_loss=False,
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
        data_iterator=[
            DataIterator(
                {
                    "lineage_source_sample_indices": [],
                    "lineage_output_indices": [],
                    "lineage_output_counts": [],
                },
                micro_batch_size=1,
            )
        ],
        parallel_state=FakeParallelState(),
        forward_backward_engine=FakeForwardBackwardEngine(),
    )

    monkeypatch.setattr(model_module, "get_args", lambda: env.args)
    monkeypatch.setattr(model_module, "get_parallel_state", lambda: env.parallel_state)
    monkeypatch.setattr("miles.backends.training_utils.parallel.get_parallel_state", lambda: env.parallel_state)
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


class TestTrainOneStepModelCompanion:
    def test_disabled_check_skips_witness_identity_collection(
        self, train_one_step_env: TrainOneStepEnv, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ordinary training needs neither a companion nor sample identity fields."""
        from miles.backends.megatron_utils import model as model_module

        train_one_step_env.args.enable_sample_ownership_checker = False
        del train_one_step_env.model[0].model_companion
        train_one_step_env.data_iterator[0].rollout_data.clear()
        optimizer = FakeFiniteOptimizer()

        def reject_witness_work(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("Witness collection must be disabled")

        monkeypatch.setattr(
            model_module.SampleIdentityExtractor, "get_consumed_sample_identities", reject_witness_work
        )
        monkeypatch.setattr(model_module.SampleIdentityExtractor, "gather_sample_identities", reject_witness_work)

        _, _, outcome = model_module.train_one_step(
            args=train_one_step_env.args,
            rollout_id=7,
            step_id=3,
            data_iterator=train_one_step_env.data_iterator,
            model=train_one_step_env.model,
            optimizer=optimizer,
            opt_param_scheduler=FakeScheduler(),
            num_microbatches=1,
            num_rollouts=1,
            witness_info=None,
            attempt=2,
        )

        assert outcome is model_module.TrainStepOutcome.NORMAL
        assert optimizer.step_count == 1

    def test_successful_step_records_consumed_rows_as_trained(
        self, train_one_step_env: TrainOneStepEnv, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful optimizer update records each consumed row as trained."""
        from miles.backends.megatron_utils import model as model_module

        identity_fields = train_one_step_env.data_iterator[0].rollout_data
        identity_fields["lineage_source_sample_indices"] = [7]
        identity_fields["lineage_output_indices"] = [0]
        identity_fields["lineage_output_counts"] = [1]
        train_one_step_env.args.check_for_nan_in_loss_and_grad = False

        def forward_backward_engine(**kwargs: Any) -> list[dict[str, Any]]:
            kwargs["data_iterator"][0].offset = 1
            return []

        monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward_engine)

        _, _, outcome = model_module.train_one_step(
            args=train_one_step_env.args,
            rollout_id=7,
            step_id=3,
            data_iterator=train_one_step_env.data_iterator,
            model=train_one_step_env.model,
            optimizer=FakeFiniteOptimizer(),
            opt_param_scheduler=FakeScheduler(),
            num_microbatches=1,
            num_rollouts=1,
            witness_info=None,
            attempt=2,
        )

        assert outcome is model_module.TrainStepOutcome.NORMAL
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=False) == {
            model_module.SampleLineage(source_sample_index=7, output_index=0, output_count=1): 1
        }
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=True) == {}

    def test_nonfinite_step_records_consumed_rows_as_skipped(
        self, train_one_step_env: TrainOneStepEnv, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A nonfinite optimizer skip records each consumed row without marking it trained."""
        from miles.backends.megatron_utils import model as model_module

        identity_fields = train_one_step_env.data_iterator[0].rollout_data
        identity_fields["lineage_source_sample_indices"] = [7]
        identity_fields["lineage_output_indices"] = [1]
        identity_fields["lineage_output_counts"] = [2]
        train_one_step_env.args.check_for_nan_in_loss_and_grad = False
        train_one_step_env.forward_backward_engine = FakeForwardBackwardEngine()

        def forward_backward_engine(**kwargs: Any) -> list[dict[str, Any]]:
            kwargs["data_iterator"][0].offset = 1
            return []

        monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward_engine)

        _, _, outcome = model_module.train_one_step(
            args=train_one_step_env.args,
            rollout_id=7,
            step_id=3,
            data_iterator=train_one_step_env.data_iterator,
            model=train_one_step_env.model,
            optimizer=FakeNonfiniteOptimizer(),
            opt_param_scheduler=None,
            num_microbatches=1,
            num_rollouts=1,
            witness_info=None,
            attempt=2,
        )

        assert outcome is model_module.TrainStepOutcome.NORMAL
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=False) == {}
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=True) == {
            model_module.SampleLineage(source_sample_index=7, output_index=1, output_count=2): 1
        }

    def test_ft_discard_does_not_record_consumed_rows_as_skipped(
        self, train_one_step_env: TrainOneStepEnv, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A discarded FT attempt leaves both successful outcome counters unchanged."""
        from miles.backends.megatron_utils import model as model_module

        identity_fields = train_one_step_env.data_iterator[0].rollout_data
        identity_fields["lineage_source_sample_indices"] = [7]
        identity_fields["lineage_output_indices"] = [0]
        identity_fields["lineage_output_counts"] = [1]
        train_one_step_env.parallel_state.indep_dp.size = 2
        train_one_step_env.args.check_for_nan_in_loss_and_grad = False

        def forward_backward_engine(**kwargs: Any) -> list[dict[str, Any]]:
            kwargs["data_iterator"][0].offset = 1
            return []

        def discard_attempt(*_args: Any, collect_training_metadata, **_kwargs: Any) -> tuple[bool, dict]:
            collect_training_metadata()
            return False, {}

        optimizer = FakeFiniteOptimizer()
        monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward_engine)
        monkeypatch.setattr(model_module, "allreduce_grads_and_losses_across_replicas", discard_attempt)

        _, _, outcome = model_module.train_one_step(
            args=train_one_step_env.args,
            rollout_id=7,
            step_id=0,
            data_iterator=train_one_step_env.data_iterator,
            model=train_one_step_env.model,
            optimizer=optimizer,
            opt_param_scheduler=None,
            num_microbatches=1,
            num_rollouts=1,
            witness_info=None,
            attempt=2,
        )

        assert outcome is model_module.TrainStepOutcome.DISCARDED_SHOULD_RETRY
        assert optimizer.step_count == 0
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=False) == {}
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=True) == {}


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
