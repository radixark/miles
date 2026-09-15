import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.rollout.test_fully_async_rollout import FakeDataSource, make_args, make_fn, make_group, train_input

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.backends.training_utils.model_companion import ModelCompanion, ModelCompanionSampleConsumptionUtils
from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn, _RunningTask
from miles.utils.arguments import _resolve_sample_ownership_check
from miles.utils.audit_utils.event_analyzer.analyzer import run_sample_ownership_analysis
from miles.utils.audit_utils.event_analyzer.rules.sample_ownership.models import SampleOwnershipViolation
from miles.utils.audit_utils.event_logger.logger import EventLogger, get_event_logger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import ExplicitlyDroppedSamplesEvent, TrainGroupStepEndEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
from miles.utils.types import Sample, SampleLineage


class TestCheckpointSampleOwnership:
    @pytest.mark.parametrize("lose_prefetched_batch", [False, True], ids=["healthy", "missing-training-step"])
    async def test_ci_checker_detects_a_prefetched_batch_lost_after_restore(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lose_prefetched_batch: bool
    ) -> None:
        """The CI checker rejects a restored batch that the driver never sends to training."""
        event_logger = EventLogger(log_dir=tmp_path / "events", source=SimpleProcessIdentity(component="main"))
        set_event_logger(event_logger)
        args = _checkpoint_args(tmp_path)
        source = FakeDataSource(scripted=[make_group(9)])
        SampleOwnershipRecorder.install(
            args=SimpleNamespace(enable_sample_ownership_checker=True),
            data_source=source,
            current_rollout_id=lambda: 0,
        )
        fn = make_fn(monkeypatch, args, source)
        [group] = source.get_samples(1)
        await fn._output.put(DataBufferInput(prompt_group=group, group=group))
        fn.save(tmp_path)

        restored = make_fn(monkeypatch, args, source)
        restored.load(tmp_path)
        restored._worker = asyncio.create_task(asyncio.Event().wait())
        checker_args = SimpleNamespace(
            **parser_defaults()
            | dict(
                ci_test=True,
                enable_sample_ownership_checker=None,
                sample_ownership_grace_steps=None,
                custom_convert_samples_to_train_data_path=None,
                save_debug_event_data=str(event_logger.log_dir),
                train_backend="megatron",
                megatron_config=None,
                lora_rank=0,
                lora_adapter_path=None,
                multi_lora=False,
                debug_train_only=False,
                debug_rollout_only=False,
                num_critic_only_steps=0,
            )
        )
        _resolve_sample_ownership_check(checker_args)
        assert checker_args.enable_sample_ownership_checker
        trainer = _CheckpointTrainer()
        try:
            output = await restored._drain(RolloutFnTrainInput(rollout_id=7, weight_version=1))
            assert [[sample.index for sample in batch] for batch in output.samples] == [[90, 91]]
            if not lose_prefetched_batch:
                await trainer.train(rollout_id=7, rollout_data_pack=output.samples)
            for rollout_id in (8, 9):
                await trainer.train(rollout_id=rollout_id, rollout_data_pack=source.get_samples(1))

            if lose_prefetched_batch:
                with pytest.raises(SampleOwnershipViolation, match="source sample had no training outcome"):
                    run_sample_ownership_analysis(args=checker_args, event_dir=event_logger.log_dir)
            else:
                run_sample_ownership_analysis(args=checker_args, event_dir=event_logger.log_dir)
                assert trainer.published == [7, 8, 9]
        finally:
            restored._worker.cancel()
            await asyncio.gather(restored._worker, return_exceptions=True)
            set_event_logger(None)


class _CheckpointTrainer:
    def __init__(self) -> None:
        self.model = [torch.nn.Module()]
        self.model[0].add_module(
            "model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        )
        self.published = []

    async def train(
        self, rollout_id: int, rollout_data_pack: list[list[Sample]], external_data: Any = None
    ) -> list[Any]:
        ModelCompanionSampleConsumptionUtils.record(
            self.model,
            [
                SampleLineage(source_sample_index=sample.index, output_index=0, output_count=1)
                for group in rollout_data_pack
                for sample in group
            ],
        )
        SampleOwnershipRecorder.publish_model_companion_info(
            self.model,
            rollout_id=rollout_id,
            attempt=0,
            cell_index=0,
        )
        get_event_logger().log(
            TrainGroupStepEndEvent,
            dict(
                rollout_id=rollout_id,
                attempt=0,
                role="actor",
                cell_outcomes={0: [TrainStepOutcome.NORMAL]},
            ),
            print_log=False,
        )
        self.published.append(rollout_id)
        return []


def _checkpoint_args(path: Path, **overrides: object) -> Namespace:
    return make_args(save=str(path), load=str(path), rollout_batch_size=1, **overrides)


async def _generate_and_reward(_state, group: list[Sample], **_kwargs: Any) -> list[Sample]:
    for sample in group:
        sample.status = Sample.Status.COMPLETED
        sample.reward = 1
    return group


async def test_running_generation_is_restored_as_a_clean_retry(monkeypatch, tmp_path: Path) -> None:
    """A half-written generation is replayed from an untouched prompt after restore."""
    fn = make_fn(monkeypatch, _checkpoint_args(tmp_path), FakeDataSource())
    group = make_group(3)
    group[0].response = "partial"
    fn._running_tasks.append(_RunningTask(prompt_group=group, task=asyncio.Future()))

    fn.save(tmp_path)
    restored = make_fn(monkeypatch, _checkpoint_args(tmp_path), FakeDataSource())
    restored.load(tmp_path)

    [pending] = restored._retry_buffer
    assert all(sample.response == "" for sample in pending)


async def test_an_incomplete_batch_stays_in_the_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """A drain waiting for a whole batch restores every buffered group exactly once."""
    args = _checkpoint_args(tmp_path)
    args.rollout_batch_size = 2
    fn = make_fn(monkeypatch, args, FakeDataSource())
    first = make_group(5)
    await fn._output.put(DataBufferInput(prompt_group=first, group=first))
    fn._worker = asyncio.create_task(asyncio.Event().wait())
    draining = asyncio.create_task(fn._drain(RolloutFnTrainInput(rollout_id=3, weight_version=1)))
    try:
        await asyncio.sleep(0.01)
        assert not draining.done()
        fn.save(tmp_path)
    finally:
        draining.cancel()
        fn._worker.cancel()
        await asyncio.gather(draining, fn._worker, return_exceptions=True)

    state = torch.load(tmp_path / "state.pt", weights_only=False)
    assert [sample.index for sample in state.output[0].group] == [50, 51]

    restored = make_fn(monkeypatch, args, FakeDataSource())
    restored.load(tmp_path)
    second = make_group(6)
    await restored._output.put(DataBufferInput(prompt_group=second, group=second))
    restored._worker = asyncio.create_task(asyncio.Event().wait())
    try:
        output = await asyncio.wait_for(
            restored._drain(RolloutFnTrainInput(rollout_id=3, weight_version=1)), timeout=5
        )
    finally:
        restored._worker.cancel()
        await asyncio.gather(restored._worker, return_exceptions=True)

    assert [[sample.index for sample in group] for group in output.samples] == [[50, 51], [60, 61]]
    assert restored._output.state_dict() == []


_DRAIN_WAKEUP_YIELDS = 100


async def test_a_save_taken_as_the_drain_wakes_cannot_lose_the_batch(monkeypatch, tmp_path: Path) -> None:
    """A batch that has left the buffer has already reached the drain, so no save sees it nowhere."""
    args = _checkpoint_args(tmp_path)
    fn = make_fn(monkeypatch, args, FakeDataSource())
    fn._worker = asyncio.create_task(asyncio.Event().wait())
    draining = asyncio.create_task(fn._drain(RolloutFnTrainInput(rollout_id=3, weight_version=1)))
    try:
        await asyncio.sleep(0.01)
        assert not draining.done()
        group = make_group(5)
        await fn._output.put(DataBufferInput(prompt_group=group, group=group))

        for _ in range(_DRAIN_WAKEUP_YIELDS):
            fn.save(tmp_path)
            state = torch.load(tmp_path / "state.pt", weights_only=False)
            assert state.running == []
            assert bool(state.output) != draining.done()
            if draining.done():
                break
            await asyncio.sleep(0)

        assert draining.done()
        output = await draining
    finally:
        draining.cancel()
        fn._worker.cancel()
        await asyncio.gather(draining, fn._worker, return_exceptions=True)

    assert [sample.index for sample in output.samples[0]] == [50, 51]


async def test_aborted_group_in_retry_queue_survives_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """An aborted group saved after recycling resumes exactly once from its prompt."""
    fn = make_fn(monkeypatch, _checkpoint_args(tmp_path, async_unused_samples_handler="retry"), FakeDataSource())
    group = make_group(6)
    for sample in group:
        sample.status = Sample.Status.ABORTED
    await fn._output.put(DataBufferInput(prompt_group=group, group=group))

    fn.save(tmp_path)
    restored = make_fn(monkeypatch, fn.args, FakeDataSource())
    restored.load(tmp_path)

    [pending] = restored._retry_buffer
    assert [sample.index for sample in pending] == [60, 61]
    assert all(sample.response == "" for sample in pending)


async def test_a_group_blocked_in_put_is_restored_as_a_clean_retry(monkeypatch, tmp_path: Path) -> None:
    """A group saved while its put waits for buffer capacity is regenerated from its prompt after restore."""
    args = _checkpoint_args(tmp_path, async_data_buffer_capacity_factor=1)
    fn = make_fn(monkeypatch, args, FakeDataSource(), generate=_generate_and_reward)
    first, second = make_group(1), make_group(2)
    await fn._output.put(DataBufferInput(prompt_group=first, group=first))
    blocked = asyncio.create_task(fn._output.put(DataBufferInput(prompt_group=second, group=second)))
    fn._running_tasks.append(_RunningTask(prompt_group=second, task=blocked))
    await asyncio.sleep(0.01)
    assert not blocked.done()
    fn.save(tmp_path)
    blocked.cancel()
    await asyncio.gather(blocked, return_exceptions=True)

    restored = make_fn(monkeypatch, args, FakeDataSource(), generate=_generate_and_reward)
    restored.load(tmp_path)

    [pending] = restored._retry_buffer
    assert [sample.index for sample in pending] == [20, 21]
    assert all(sample.response == "" for sample in pending)
    assert [entry.group[0].group_index for entry in restored._output.state_dict()] == [1]
    try:
        outputs = [await asyncio.wait_for(restored(train_input(rollout_id=i)), timeout=5) for i in (0, 1)]
    finally:
        await restored.dispose()
    trained = [[sample.index for sample in group] for output in outputs for group in output.samples]
    assert trained == [[10, 11], [20, 21]]
    assert not restored._retry_buffer


# ============================ dispose drop records ============================


class TestDisposeRecordsHeldGroups:
    def _make_started_fn(self, monkeypatch) -> FullyAsyncRolloutFn:
        args = make_args(rollout_batch_size=1, enable_sample_ownership_checker=True)
        fn = make_fn(monkeypatch, args, FakeDataSource())
        fn._worker = asyncio.create_task(asyncio.Event().wait())
        return fn

    async def test_dispose_records_every_held_group_and_no_drained_one(self, monkeypatch, tmp_path: Path) -> None:
        """A run that ends really abandons what the producer still holds, and only that."""
        set_event_logger(
            EventLogger(log_dir=tmp_path / "events", source=SimpleProcessIdentity(component="rollout_executor"))
        )
        try:
            fn = self._make_started_fn(monkeypatch)
            drained = make_group(4)
            await fn._output.put(DataBufferInput(prompt_group=drained, group=drained))
            output = await fn._drain(RolloutFnTrainInput(rollout_id=0, weight_version=1))
            assert [sample.index for sample in output.samples[0]] == [40, 41]

            fn._running_tasks.append(_RunningTask(prompt_group=make_group(1), task=asyncio.Future()))
            fn._retry_buffer.append(make_group(2))
            buffered = make_group(3)
            await fn._output.put(DataBufferInput(prompt_group=buffered, group=buffered))

            await fn.dispose()

            [event] = read_events(tmp_path / "events")
            assert isinstance(event, ExplicitlyDroppedSamplesEvent)
            assert sorted(event.source_sample_indices) == [10, 11, 20, 21, 30, 31]
            assert event.reason == "shutdown_in_flight"
        finally:
            set_event_logger(None)

    @pytest.mark.parametrize("source", ["running", "retry", "output"])
    async def test_dispose_records_a_group_held_in_any_single_source(
        self, monkeypatch, tmp_path: Path, source: str
    ) -> None:
        """Each of the three places save() reads is a place a group can be stranded at shutdown."""
        set_event_logger(
            EventLogger(log_dir=tmp_path / "events", source=SimpleProcessIdentity(component="rollout_executor"))
        )
        try:
            fn = self._make_started_fn(monkeypatch)
            group = make_group(7)
            if source == "running":
                fn._running_tasks.append(_RunningTask(prompt_group=group, task=asyncio.Future()))
            elif source == "retry":
                fn._retry_buffer.append(group)
            else:
                await fn._output.put(DataBufferInput(prompt_group=group, group=group))

            await fn.dispose()

            [event] = read_events(tmp_path / "events")
            assert event.source_sample_indices == [70, 71]
        finally:
            set_event_logger(None)

    async def test_dispose_records_nothing_when_the_producer_holds_nothing(self, monkeypatch, tmp_path: Path) -> None:
        """A clean shutdown abandons no sample, so it may not invent a drop."""
        set_event_logger(
            EventLogger(log_dir=tmp_path / "events", source=SimpleProcessIdentity(component="rollout_executor"))
        )
        try:
            fn = self._make_started_fn(monkeypatch)

            await fn.dispose()

            assert read_events(tmp_path / "events") == []
        finally:
            set_event_logger(None)

    async def test_dispose_is_idempotent(self, monkeypatch, tmp_path: Path) -> None:
        """A second dispose would otherwise drop the same held samples twice and read as a lost sample."""
        set_event_logger(
            EventLogger(log_dir=tmp_path / "events", source=SimpleProcessIdentity(component="rollout_executor"))
        )
        try:
            fn = self._make_started_fn(monkeypatch)
            fn._retry_buffer.append(make_group(8))

            await fn.dispose()
            await fn.dispose()

            [event] = read_events(tmp_path / "events")
            assert event.source_sample_indices == [80, 81]
        finally:
            set_event_logger(None)
