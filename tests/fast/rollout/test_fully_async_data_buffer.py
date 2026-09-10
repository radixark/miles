import asyncio
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import pytest

from miles.rollout import fully_async_data_buffer
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DataBufferState,
    DefaultDataBuffer,
    DefaultMultiDataBuffer,
    PutOutcome,
    PutOutcomes,
    UnusedReason,
)


class _RecordingBuffer(DataBuffer):
    """A custom DataBuffer of the kind --custom-async-data-buffer-path-per-model names."""

    def __init__(self, input: DataBufferConstructorInput) -> None:
        self.input = input
        self.metrics_asked_about: list[str | None] = []

    async def put(self, input: DataBufferInput) -> PutOutcomes:
        raise NotImplementedError

    async def get(self, *, num_groups: int, **context: Any) -> list[DataBufferInput]:
        raise NotImplementedError

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        self.metrics_asked_about.append(trainer_model_id)
        return {"asked": float(len(self.metrics_asked_about))}

    def snapshot(self) -> DataBufferState:
        return {}

    def restore(self, state: DataBufferState) -> None:
        raise NotImplementedError


def _multi_buffer(monkeypatch: pytest.MonkeyPatch, *, model_ids: list[str]) -> DefaultMultiDataBuffer:
    monkeypatch.setattr(
        fully_async_data_buffer, "resolve_megatron_config", lambda args: SimpleNamespace(model_ids=model_ids)
    )
    monkeypatch.setattr(fully_async_data_buffer, "load_function", lambda path: _RecordingBuffer)
    args = Namespace(custom_async_data_buffer_path_per_model=[f"{one}=recording.Buffer" for one in model_ids])
    return DefaultMultiDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=_ignore_group))


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

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


def _make_args(**overrides: Any) -> Namespace:
    defaults = dict(
        async_data_buffer_capacity_factor=1.0,
        custom_async_data_buffer_path_per_model=None,
        dynamic_sampling_filter_path=None,
        max_weight_staleness=None,
        megatron_config=encode_megatron_config("solver", "verifier"),
        reward_key=None,
        rollout_batch_size=1,
    )

    defaults.update(overrides)
    return Namespace(**defaults)


def _make_sample(*, index: int, reward: float = 1.0, trainer_model_id: str | None = None) -> Sample:
    sample = Sample(
        index=index,
        prompt="prompt",
        response="response",
        reward=reward,
        status=Sample.Status.COMPLETED,
    )
    sample.trainer_model_id = trainer_model_id
    return sample


def _ignore_group(group: list[Sample], *, reason: UnusedReason, trainer_model_id: str | None) -> None:
    pass


class TestPerPolicyMetrics:
    async def test_stale_eviction_identifies_only_the_consuming_policy(self) -> None:
        """Staleness in one inner buffer must not drop a peer's copy of the same prompt."""
        unused: list[str | None] = []
        buffer = DefaultMultiDataBuffer(
            DataBufferConstructorInput(
                args=_make_args(max_weight_staleness=0),
                unused_handler_fn=lambda group, *, reason, trainer_model_id: unused.append(trainer_model_id),
            )
        )
        stale = _entry(1, weight_version="0")
        fresh = _entry(2, weight_version="1")
        buffer.restore({"solver": [stale], "verifier": [stale, fresh]})

        assert await buffer.get(num_groups=1, trainer_model_id="verifier", current_version=1) == [fresh]
        assert unused == ["verifier"]
        assert buffer.snapshot()["solver"] == [stale]

    async def test_unused_prompts_identify_the_inner_policy(self) -> None:
        """An aborted policy output must not drop the same prompt for another policy."""
        unused: list[str | None] = []
        buffer = DefaultMultiDataBuffer(
            DataBufferConstructorInput(
                args=_make_args(),
                unused_handler_fn=lambda group, *, reason, trainer_model_id: unused.append(trainer_model_id),
            )
        )
        solver = _make_sample(index=1, trainer_model_id="solver")
        verifier = _make_sample(index=1, trainer_model_id="verifier")
        verifier.status = Sample.Status.ABORTED

        await buffer.put(DataBufferInput(prompt_group=[solver], group=[solver, verifier]))

        assert unused == ["verifier"]
        [entry] = await buffer.get(num_groups=1, trainer_model_id="solver")
        assert entry.group == [solver]

    def test_replay_contract_is_selected_per_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One replaying inner buffer cannot disable duplicate checks for a FIFO peer."""
        multi = _multi_buffer(monkeypatch, model_ids=["solver", "verifier"])
        _composed(multi, "verifier").replays_samples = True

        assert multi.replays_samples_of("solver") is False
        assert multi.replays_samples_of("verifier") is True

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


def _single_buffer(unused: list[list[Sample]] | None = None, **overrides: Any) -> DefaultDataBuffer:
    return DefaultDataBuffer(
        DataBufferConstructorInput(
            args=_make_args(**{"async_data_buffer_capacity_factor": 2.0, "rollout_batch_size": 2, **overrides}),
            unused_handler_fn=(
                (lambda group, *, reason, trainer_model_id: unused.append(group))
                if unused is not None
                else _ignore_group
            ),
        )
    )


def _entry(index: int, *, weight_version: str | None = None) -> DataBufferInput:
    sample = _make_sample(index=index)
    if weight_version is not None:
        sample.weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan(version=weight_version, abs_start=0, abs_end=1)])
        ]
    return DataBufferInput(prompt_group=[sample], group=[sample])


class TestBatchGet:
    async def test_a_partial_batch_is_never_handed_out(self):
        """A batch half held by the caller is a batch no checkpoint of the buffer can account for."""
        buffer = _single_buffer()
        await buffer.put(_entry(1))

        waiting = asyncio.create_task(buffer.get(num_groups=2))
        await asyncio.sleep(0)

        assert not waiting.done()
        assert len(buffer.snapshot()[None]) == 1
        waiting.cancel()

    async def test_the_whole_batch_comes_back_at_once_when_it_is_complete(self):
        """The drain asks for rollout_batch_size groups and must get exactly that many in one call."""
        buffer = _single_buffer()
        await buffer.put(_entry(1))
        waiting = asyncio.create_task(buffer.get(num_groups=2))
        await asyncio.sleep(0)

        await buffer.put(_entry(2))
        batch = await waiting

        assert [entry.group[0].index for entry in batch] == [1, 2]
        assert buffer.snapshot() == {None: []}

    async def test_a_buffer_full_of_stale_groups_does_not_wedge_the_producer(self):
        """Evicting only on the way out would leave get short of a batch while put blocks on capacity."""
        unused: list = []
        buffer = _single_buffer(unused, max_weight_staleness=0, async_data_buffer_capacity_factor=1.0)
        await buffer.put(_entry(1, weight_version="1"))
        await buffer.put(_entry(2, weight_version="1"))

        waiting = asyncio.create_task(buffer.get(num_groups=2, current_version=9))
        await asyncio.sleep(0)
        blocked = asyncio.create_task(buffer.put(_entry(3, weight_version="9")))
        await asyncio.sleep(0)

        assert [group[0].index for group in unused] == [1, 2]
        assert blocked.done()
        waiting.cancel()

    async def test_a_capacity_below_one_batch_is_refused(self):
        """The producer would block on a full buffer while the step waits for a batch it can never complete."""
        with pytest.raises(AssertionError, match="below the 2 groups a step drains"):
            _single_buffer(async_data_buffer_capacity_factor=0.5)


class TestSnapshotRestore:
    async def test_what_a_snapshot_reports_is_what_a_restore_puts_back(self):
        """A checkpoint persists the snapshot, so anything it omits is a prompt the resumed run loses."""
        buffer = _single_buffer()
        await buffer.put(_entry(1))
        await buffer.put(_entry(2))
        state = buffer.snapshot()

        restored = _single_buffer()
        restored.restore(state)

        assert [entry.group[0].index for entry in restored.snapshot()[None]] == [1, 2]

    async def test_a_snapshot_does_not_hand_ownership_over(self):
        """The checker reads the same method the checkpoint writes, and reading must not consume."""
        buffer = _single_buffer()
        await buffer.put(_entry(1))

        buffer.snapshot()

        assert [entry.group[0].index for entry in await buffer.get(num_groups=1)] == [1]

    async def test_restoring_onto_a_non_empty_buffer_is_refused(self):
        """Restore runs once, before the producer starts; anything else would duplicate groups."""
        buffer = _single_buffer()
        await buffer.put(_entry(1))

        with pytest.raises(AssertionError, match="already holds"):
            buffer.restore({None: [_entry(2)]})

    async def test_a_restored_group_is_measured_against_the_version_of_the_resumed_run(self):
        """Staleness is a get-time decision, so restore must not re-apply the version it was saved under."""
        buffer = _single_buffer(max_weight_staleness=0)
        buffer.restore({None: [_entry(1, weight_version="9")]})

        assert [entry.group[0].index for entry in await buffer.get(num_groups=1, current_version=9)] == [1]

    async def test_the_multi_policy_snapshot_keys_every_entry_by_its_policy(self):
        """Restoring a policy's groups into another policy's queue would train it on foreign samples."""
        buffer = DefaultMultiDataBuffer(DataBufferConstructorInput(args=_make_args(), unused_handler_fn=_ignore_group))
        group = [
            _make_sample(index=1, reward=1.0, trainer_model_id="solver"),
            _make_sample(index=2, reward=3.0, trainer_model_id="verifier"),
        ]
        await buffer.put(DataBufferInput(prompt_group=group, group=group))

        state = buffer.snapshot()

        assert sorted(state) == ["solver", "verifier"]
        assert [entry.group[0].index for entry in state["solver"]] == [1]
        assert [entry.group[0].index for entry in state["verifier"]] == [2]

    async def test_a_multi_policy_restore_refuses_a_policy_this_run_does_not_train(self):
        """Its groups would sit in a buffer nobody drains, and the run would silently stall."""
        buffer = DefaultMultiDataBuffer(DataBufferConstructorInput(args=_make_args(), unused_handler_fn=_ignore_group))

        with pytest.raises(AssertionError, match="train no policy of this run"):
            buffer.restore({"reviewer": [_entry(1)]})


class TestPutOutcome:
    async def test_a_stored_group_reports_kept(self):
        """The rollout function turns this into the in_flight -> output_buffer ownership event."""
        buffer = _single_buffer()

        assert await buffer.put(_entry(1)) == {None: PutOutcome.KEPT}

    async def test_an_aborted_group_reports_recycled(self):
        """It went to the unused handler, which logs the ownership move itself."""
        unused: list[tuple[list[Sample], UnusedReason]] = []
        buffer = DefaultDataBuffer(
            DataBufferConstructorInput(
                args=_make_args(),
                unused_handler_fn=lambda group, *, reason, trainer_model_id: unused.append((group, reason)),
            )
        )
        entry = _entry(1)
        entry.group[0].status = Sample.Status.ABORTED

        assert await buffer.put(entry) == {None: PutOutcome.RECYCLED}
        assert unused == [(entry.prompt_group, UnusedReason.ABORTED)]

    async def test_a_dynamic_filter_reject_reports_dropped(self):
        """Nothing else records this group's fate, so the outcome is what proves it was not lost."""
        buffer = _single_buffer(dynamic_sampling_filter_path=f"{__name__}.reject_everything")
        entry = _entry(1)

        assert await buffer.put(entry) == {None: PutOutcome.DROPPED}

    async def test_a_multi_policy_put_reports_one_outcome_per_policy(self):
        """A group kept by one policy and dropped by another needs both fates recorded, not one of them."""
        args = _make_args()
        args.dynamic_sampling_filter_path = f"{__name__}.reject_the_verifier"
        buffer = DefaultMultiDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=_ignore_group))
        group = [
            _make_sample(index=1, reward=1.0, trainer_model_id="solver"),
            _make_sample(index=2, reward=3.0, trainer_model_id="verifier"),
        ]

        outcomes = await buffer.put(DataBufferInput(prompt_group=group, group=group))

        assert outcomes == {"solver": PutOutcome.KEPT, "verifier": PutOutcome.DROPPED}


def reject_everything(args, group, **kwargs) -> DynamicFilterOutput:
    return DynamicFilterOutput(keep=False, reason="rejected")


def reject_the_verifier(args, group, **kwargs) -> DynamicFilterOutput:
    if all(sample.trainer_model_id == "verifier" for sample in group):
        return DynamicFilterOutput(keep=False, reason="rejected")
    return DynamicFilterOutput(keep=True)
