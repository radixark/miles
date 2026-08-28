from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.rollout import fully_async_data_buffer
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultMultiDataBuffer,
)


class _RecordingBuffer(DataBuffer):
    """A custom DataBuffer of the kind --custom-async-data-buffer-path-per-model names."""

    def __init__(self, input: DataBufferConstructorInput) -> None:
        self.input = input
        self.metrics_asked_about: list[str | None] = []

    async def put(self, input: DataBufferInput) -> None:
        raise NotImplementedError

    async def get(self, **context) -> DataBufferInput:
        raise NotImplementedError

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        self.metrics_asked_about.append(trainer_model_id)
        return {"asked": float(len(self.metrics_asked_about))}


def _multi_buffer(monkeypatch: pytest.MonkeyPatch, *, model_ids: list[str]) -> DefaultMultiDataBuffer:
    monkeypatch.setattr(
        fully_async_data_buffer, "resolve_megatron_config", lambda args: SimpleNamespace(model_ids=model_ids)
    )
    monkeypatch.setattr(fully_async_data_buffer, "load_function", lambda path: _RecordingBuffer)
    args = Namespace(custom_async_data_buffer_path_per_model=[f"{one}=recording.Buffer" for one in model_ids])
    return DefaultMultiDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=lambda samples: None))


def _composed(multi: DefaultMultiDataBuffer, model_id: str) -> _RecordingBuffer:
    return multi._inners[model_id]


class TestTheMetricsOfOnePolicy:
    def test_the_policy_the_drain_asked_about_reaches_the_buffer_it_composes(self, monkeypatch):
        """A buffer that selects by policy saw None and could attribute its metrics to the wrong one."""
        multi = _multi_buffer(monkeypatch, model_ids=["solver", "verifier"])

        multi.get_metrics("solver")

        assert _composed(multi, "solver").metrics_asked_about == ["solver"]

    def test_a_policy_is_never_asked_about_another_one(self, monkeypatch):
        """Each policy keeps its own window counters, and a drain resets the ones it reads."""
        multi = _multi_buffer(monkeypatch, model_ids=["solver", "verifier"])

        multi.get_metrics("solver")
        multi.get_metrics("verifier")

        assert _composed(multi, "solver").metrics_asked_about == ["solver"]
        assert _composed(multi, "verifier").metrics_asked_about == ["verifier"]

    def test_the_metrics_returned_are_the_ones_that_policy_buffer_reported(self, monkeypatch):
        """Forwarding the policy may not cost the caller the numbers it came for."""
        multi = _multi_buffer(monkeypatch, model_ids=["solver", "verifier"])

        assert multi.get_metrics("solver") == {"asked": 1.0}
        assert multi.get_metrics("solver") == {"asked": 2.0}

    def test_a_policy_this_run_does_not_train_is_refused(self, monkeypatch):
        """The composed buffers are one per policy of the run, so any other name selects nothing."""
        multi = _multi_buffer(monkeypatch, model_ids=["solver"])

        with pytest.raises(AssertionError, match="trains no policy of this run"):
            multi.get_metrics("stranger")


from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.utils.types import Sample


def _make_args() -> Namespace:
    return Namespace(
        async_data_buffer_capacity_factor=1.0,
        custom_async_data_buffer_path_per_model=None,
        dynamic_sampling_filter_path=None,
        max_weight_staleness=None,
        megatron_config=encode_megatron_config("solver", "verifier"),
        reward_key=None,
        rollout_batch_size=1,
    )


def _make_sample(*, index: int, reward: float, trainer_model_id: str) -> Sample:
    sample = Sample(
        index=index,
        prompt="prompt",
        response="response",
        reward=reward,
        status=Sample.Status.COMPLETED,
    )
    sample.trainer_model_id = trainer_model_id
    return sample


def _ignore_group(group: list[Sample]) -> None:
    pass


class TestPerPolicyMetrics:
    async def test_collecting_one_policy_metrics_does_not_reset_another_policy_window(self) -> None:
        """Collecting one policy preserves another policy's resettable raw-reward window."""
        buffer = DefaultMultiDataBuffer(DataBufferConstructorInput(args=_make_args(), unused_handler_fn=_ignore_group))
        group = [
            _make_sample(index=1, reward=1.0, trainer_model_id="solver"),
            _make_sample(index=2, reward=3.0, trainer_model_id="verifier"),
        ]

        await buffer.put(DataBufferInput(prompt_group=group, group=group))

        assert buffer.get_metrics("solver")["rollout/raw_reward_unfiltered"] == 1.0
        assert buffer.get_metrics("verifier")["rollout/raw_reward_unfiltered"] == 3.0
        assert "rollout/raw_reward_unfiltered" not in buffer.get_metrics("verifier")
