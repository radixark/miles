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
    return DefaultMultiDataBuffer(
        DataBufferConstructorInput(args=args, unused_handler_fn=lambda samples, reason: None)
    )


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


def _ignore_group(group: list[Sample], reason: fully_async_data_buffer.UnusedReason) -> None:
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


# ================================ test setup ================================

import asyncio
from dataclasses import dataclass

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.fully_async_data_buffer import DefaultDataBuffer, Group, UnusedReason
from miles.utils.types import WeightVersionSpan, WeightVersionsPerCall

GROUP_SIZE = 2


def _make_single_policy_args(**overrides) -> Namespace:
    defaults = dict(
        async_data_buffer_capacity_factor=1000.0,
        dynamic_sampling_filter_path=None,
        enable_sample_ownership_checker=False,
        max_weight_staleness=None,
        reward_key=None,
        rollout_batch_size=1,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_finished_group(
    group_index: int,
    *,
    status: Sample.Status = Sample.Status.COMPLETED,
    weight_version: int | None = None,
    reward: float = 1.0,
) -> list[Sample]:
    spans = (
        [WeightVersionSpan(version=str(weight_version), abs_start=0, abs_end=1)] if weight_version is not None else []
    )
    return [
        Sample(
            group_index=group_index,
            index=group_index * 10 + i,
            prompt=f"prompt {group_index}",
            response="ok",
            response_length=1,
            reward=reward,
            status=status,
            weight_versions=[WeightVersionsPerCall(spans=list(spans))] if spans else [],
        )
        for i in range(GROUP_SIZE)
    ]


@dataclass
class _BufferUnderTest:
    buffer: DefaultDataBuffer
    unused: list[tuple[list[Sample], UnusedReason]]


def _make_buffer(**overrides) -> _BufferUnderTest:
    unused: list[tuple[list[Sample], UnusedReason]] = []
    buffer = DefaultDataBuffer(
        DataBufferConstructorInput(
            args=_make_single_policy_args(**overrides),
            unused_handler_fn=lambda prompt_group, reason: unused.append((prompt_group, reason)),
        )
    )
    return _BufferUnderTest(buffer=buffer, unused=unused)


async def _put(buffer: DataBuffer, group: Group) -> None:
    await buffer.put(DataBufferInput(prompt_group=group, group=group))


async def _get_one(buffer: DataBuffer, **context) -> DataBufferInput:
    return await buffer.get(**context)


async def _settle(times: int = 20) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


def _reject_group_1(args, group, **kwargs) -> DynamicFilterOutput:
    keep = group[0].group_index != 1
    return DynamicFilterOutput(keep=keep, reason=None if keep else "rejected")


# =============================== staleness ================================


class TestStalenessFiltering:
    async def test_a_group_exactly_at_the_staleness_limit_is_kept(self) -> None:
        """--max-weight-staleness names the oldest version still worth training on, inclusive."""
        under_test = _make_buffer(max_weight_staleness=2)
        group = _make_finished_group(1, weight_version=8)
        await _put(under_test.buffer, group)

        assert (await _get_one(under_test.buffer, current_version=10)).group is group
        assert under_test.unused == []

    async def test_a_group_one_version_past_the_limit_is_refused(self) -> None:
        """One version beyond the limit is the first group the consumer may not see."""
        under_test = _make_buffer(max_weight_staleness=2)
        stale = _make_finished_group(1, weight_version=7)
        fresh = _make_finished_group(2, weight_version=10)
        await _put(under_test.buffer, stale)
        await _put(under_test.buffer, fresh)

        assert (await _get_one(under_test.buffer, current_version=10)).group is fresh
        assert under_test.unused == [(stale, UnusedReason.STALE)]
        assert under_test.buffer.get_metrics()["rollout/fully_async/stale_groups_filtered"] == 1

    async def test_a_group_dropped_for_staleness_is_left_out_of_the_consumed_staleness_metrics(self) -> None:
        """A recycled group was never consumed, so counting it skews the staleness the trainer trained on."""
        under_test = _make_buffer(max_weight_staleness=2)
        await _put(under_test.buffer, _make_finished_group(1, weight_version=1))
        await _put(under_test.buffer, _make_finished_group(2, weight_version=8))

        assert (await _get_one(under_test.buffer, current_version=10)).group[0].group_index == 2

        metrics = under_test.buffer.get_metrics()
        assert metrics["rollout/fully_async/avg_staleness"] == 2
        assert metrics["rollout/fully_async/max_staleness"] == 2

    async def test_an_unset_max_staleness_keeps_every_group_however_old(self) -> None:
        """The filter is opt-in, and a run without it still wants the staleness it is training on."""
        under_test = _make_buffer(max_weight_staleness=None)
        group = _make_finished_group(1, weight_version=1)
        await _put(under_test.buffer, group)

        assert (await _get_one(under_test.buffer, current_version=1000)).group is group
        assert under_test.buffer.get_metrics()["rollout/fully_async/avg_staleness"] == 999

    async def test_a_group_with_no_weight_version_is_never_counted_as_stale(self) -> None:
        """A group whose generation reported no version has unknown staleness, not zero."""
        under_test = _make_buffer(max_weight_staleness=0)
        group = _make_finished_group(1)
        await _put(under_test.buffer, group)

        assert (await _get_one(under_test.buffer, current_version=5)).group is group
        assert "rollout/fully_async/avg_staleness" not in under_test.buffer.get_metrics()

    async def test_a_stale_group_is_recycled_only_when_the_consumer_reaches_it(self) -> None:
        """Staleness is decided at the pop, so a stale group behind a fresh one waits its turn."""
        under_test = _make_buffer(max_weight_staleness=2)
        fresh, stale, later = (
            _make_finished_group(1, weight_version=10),
            _make_finished_group(2, weight_version=1),
            _make_finished_group(3, weight_version=10),
        )
        for group in (fresh, stale, later):
            await _put(under_test.buffer, group)

        assert (await _get_one(under_test.buffer, current_version=10)).group is fresh
        assert under_test.unused == []

        assert (await _get_one(under_test.buffer, current_version=10)).group is later
        assert under_test.unused == [(stale, UnusedReason.STALE)]

    async def test_the_version_of_the_last_get_is_the_clock_of_the_buffered_staleness_metrics(self) -> None:
        """Nothing else tells the buffer which weights the trainer is on when it reports what it holds."""
        under_test = _make_buffer()
        for group_index, weight_version in ((1, 4), (2, 6), (3, 8)):
            await _put(under_test.buffer, _make_finished_group(group_index, weight_version=weight_version))

        await _get_one(under_test.buffer, current_version=10)
        assert under_test.buffer.get_metrics()["rollout/fully_async/buffer_max_staleness"] == 4

        await _get_one(under_test.buffer, current_version=20)
        assert under_test.buffer.get_metrics()["rollout/fully_async/buffer_max_staleness"] == 12


# =============================== capacity =================================


class TestCapacity:
    @pytest.mark.parametrize(("factor", "rollout_batch_size", "expected"), [(1.5, 4, 6), (0.5, 3, 1), (2.0, 3, 6)])
    def test_the_capacity_is_the_floor_of_the_factor_times_the_rollout_batch_size(
        self, factor, rollout_batch_size, expected
    ) -> None:
        """The factor is expressed in training batches, and a partial group cannot be buffered."""
        under_test = _make_buffer(async_data_buffer_capacity_factor=factor, rollout_batch_size=rollout_batch_size)

        assert under_test.buffer._capacity == expected

    def test_a_capacity_factor_of_zero_is_refused(self) -> None:
        """A buffer that can hold nothing wedges the producer on its very first group."""
        with pytest.raises(AssertionError):
            _make_buffer(async_data_buffer_capacity_factor=0.0)

    async def test_a_blocked_put_resumes_as_soon_as_a_get_frees_a_slot(self) -> None:
        """Back pressure is the whole point of the bound, and it has to lift without a poll."""
        under_test = _make_buffer(async_data_buffer_capacity_factor=2.0)
        await _put(under_test.buffer, _make_finished_group(1))
        await _put(under_test.buffer, _make_finished_group(2))

        blocked = asyncio.create_task(_put(under_test.buffer, _make_finished_group(3)))
        await _settle()
        assert not blocked.done()

        assert (await _get_one(under_test.buffer)).group[0].group_index == 1
        await blocked

        assert under_test.buffer.get_metrics()["rollout/fully_async/queue_size"] == 2

    async def test_a_put_that_is_rejected_by_a_filter_never_takes_a_capacity_slot(self) -> None:
        """Both filters return before the lock, so a rejected group may not cost the producer its bound."""
        under_test = _make_buffer(
            async_data_buffer_capacity_factor=1.0,
            dynamic_sampling_filter_path=f"{__name__}._reject_group_1",
        )
        await _put(under_test.buffer, _make_finished_group(2, status=Sample.Status.ABORTED))
        await _put(under_test.buffer, _make_finished_group(1))

        accepted = asyncio.create_task(_put(under_test.buffer, _make_finished_group(3)))
        await _settle()

        assert accepted.done()
        assert under_test.buffer.get_metrics()["rollout/fully_async/queue_size"] == 1

    async def test_a_get_waiting_on_an_empty_buffer_wakes_on_the_next_put(self) -> None:
        """A consumer that ran ahead of the producer must be woken by the group, not by a timer."""
        under_test = _make_buffer()
        waiting = asyncio.create_task(_get_one(under_test.buffer, current_version=1))
        await _settle()
        assert not waiting.done()

        group = _make_finished_group(1)
        await _put(under_test.buffer, group)

        assert (await waiting).group is group


# ================================ metrics =================================


class TestMetricWindows:
    async def test_a_second_metrics_collection_reports_a_fresh_window(self) -> None:
        """Every counter here is per training step, and a cumulative one draws the wrong curve."""
        under_test = _make_buffer(max_weight_staleness=0)
        await _put(under_test.buffer, _make_finished_group(1, status=Sample.Status.ABORTED))
        await _put(under_test.buffer, _make_finished_group(2, weight_version=1))
        await _put(under_test.buffer, _make_finished_group(3, weight_version=9))
        await _get_one(under_test.buffer, current_version=9)

        first = under_test.buffer.get_metrics()
        assert first["rollout/fully_async/aborted_groups_filtered"] == 1
        assert first["rollout/fully_async/stale_groups_filtered"] == 1
        assert first["rollout/raw_reward_unfiltered"] == 1.0

        second = under_test.buffer.get_metrics()
        assert second["rollout/fully_async/aborted_groups_filtered"] == 0
        assert second["rollout/fully_async/stale_groups_filtered"] == 0
        assert "rollout/fully_async/avg_staleness" not in second
        assert "rollout/raw_reward_unfiltered" not in second

    async def test_the_queue_size_metric_survives_the_window_reset(self) -> None:
        """Queue depth is a snapshot of what is held, not something accumulated since the last step."""
        under_test = _make_buffer()
        await _put(under_test.buffer, _make_finished_group(1))

        assert under_test.buffer.get_metrics()["rollout/fully_async/queue_size"] == 1
        assert under_test.buffer.get_metrics()["rollout/fully_async/queue_size"] == 1


# ============================= put time filters =============================


class TestPutTimeFilters:
    async def test_an_aborted_group_reports_the_aborted_reason_to_the_unused_handler(self) -> None:
        """The configured policy can tell a failed generation from output that merely went stale."""
        under_test = _make_buffer()
        group = _make_finished_group(1, status=Sample.Status.ABORTED)

        await _put(under_test.buffer, group)

        assert under_test.unused == [(group, UnusedReason.ABORTED)]
        assert under_test.buffer.get_metrics()["rollout/fully_async/aborted_groups_filtered"] == 1

    async def test_a_group_with_one_aborted_sample_among_finished_ones_is_refused_whole(self) -> None:
        """The group is the training unit, so a partial one is not something the trainer can use."""
        under_test = _make_buffer()
        group = _make_finished_group(1)
        group[-1].status = Sample.Status.ABORTED

        await _put(under_test.buffer, group)

        assert under_test.unused == [(group, UnusedReason.ABORTED)]
        assert under_test.buffer.get_metrics()["rollout/fully_async/queue_size"] == 0

    async def test_a_group_rejected_by_the_dynamic_filter_never_reaches_the_unused_handler(self) -> None:
        """A filtered group carries no usable gradient signal, so regenerating it would only burn rollout."""
        under_test = _make_buffer(dynamic_sampling_filter_path=f"{__name__}._reject_group_1")

        await _put(under_test.buffer, _make_finished_group(1))

        assert under_test.unused == []
        assert under_test.buffer.get_metrics()["rollout/fully_async/queue_size"] == 0

    async def test_a_dynamic_filter_rejection_is_counted_with_its_reason(self) -> None:
        """A filter that rejects for several reasons is only debuggable if each one is counted apart."""
        under_test = _make_buffer(dynamic_sampling_filter_path=f"{__name__}._reject_group_1")

        await _put(under_test.buffer, _make_finished_group(1))
        await _put(under_test.buffer, _make_finished_group(2))

        assert under_test.buffer.get_metrics()["rollout/dynamic_filter/drop_rejected"] == 1


# =========================== composed buffers ============================


def _make_multi_buffer(**overrides) -> DefaultMultiDataBuffer:
    args = _make_single_policy_args(
        custom_async_data_buffer_path_per_model=None,
        megatron_config=encode_megatron_config("solver", "verifier"),
        **overrides,
    )
    return DefaultMultiDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=_ignore_group))


def _make_policy_group(group_index: int, *trainer_model_ids: str) -> list[Sample]:
    group = _make_finished_group(group_index)
    for sample, trainer_model_id in zip(group, trainer_model_ids, strict=True):
        sample.trainer_model_id = trainer_model_id
    return group


class TestComposedCapacity:
    async def test_a_full_inner_buffer_blocks_the_put_of_every_policy(self) -> None:
        """One producer walks the policies in turn, so the policy behind a full one never gets its half."""
        multi = _make_multi_buffer(async_data_buffer_capacity_factor=1.0)
        await multi.put(DataBufferInput(prompt_group=[], group=_make_policy_group(1, "solver", "solver")))

        blocked = asyncio.create_task(
            multi.put(DataBufferInput(prompt_group=[], group=_make_policy_group(2, "solver", "verifier")))
        )
        await _settle()

        assert not blocked.done()
        assert multi.get_metrics("verifier")["rollout/fully_async/queue_size"] == 0
        blocked.cancel()

    async def test_every_inner_buffer_gets_its_own_capacity(self) -> None:
        """Policies consume at their own pace, and a shared bound would let the faster one starve the slower."""
        multi = _make_multi_buffer(async_data_buffer_capacity_factor=2.0, rollout_batch_size=1)

        assert [inner._capacity for inner in multi._inners.values()] == [2, 2]

        await multi._inners["solver"].put(DataBufferInput(prompt_group=[], group=_make_finished_group(1)))
        await multi._inners["solver"].put(DataBufferInput(prompt_group=[], group=_make_finished_group(2)))
        accepted = asyncio.create_task(
            multi._inners["verifier"].put(DataBufferInput(prompt_group=[], group=_make_finished_group(3)))
        )
        await _settle()

        assert accepted.done()
        assert multi.get_metrics("verifier")["rollout/fully_async/queue_size"] == 1
